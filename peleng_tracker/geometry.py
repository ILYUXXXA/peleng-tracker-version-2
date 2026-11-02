"""Camera calibration, projection and multi-view triangulation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import yaml
from pydantic import BaseModel, Field, field_validator


class CameraCalibrationData(BaseModel):
    image_size: tuple[int, int]
    camera_matrix: list[list[float]]
    distortion_coefficients: list[float] = Field(default_factory=list)
    rotation_world_to_camera: list[list[float]]
    translation_world_to_camera: list[float]
    planar_homography: list[list[float]] | None = None
    camera_forward_axis: Literal["positive_z", "negative_z"] = "positive_z"

    @field_validator("camera_matrix", "rotation_world_to_camera")
    @classmethod
    def validate_matrix_3x3(cls, value: list[list[float]]) -> list[list[float]]:
        if np.asarray(value).shape != (3, 3):
            raise ValueError("matrix must have shape 3x3")
        return value

    @field_validator("translation_world_to_camera")
    @classmethod
    def validate_translation(cls, value: list[float]) -> list[float]:
        if len(value) != 3:
            raise ValueError("translation must contain three values")
        return value


class CalibrationDocument(BaseModel):
    coordinate_system: str = "right-handed"
    units: str = "meters"
    cameras: dict[int, CameraCalibrationData] = Field(default_factory=dict)


@dataclass(slots=True)
class CameraModel:
    camera_id: int
    image_size: tuple[int, int]
    camera_matrix: np.ndarray
    distortion: np.ndarray
    rotation: np.ndarray
    translation: np.ndarray
    planar_homography: np.ndarray | None
    camera_forward_axis: Literal["positive_z", "negative_z"]

    @property
    def projection(self) -> np.ndarray:
        return self.camera_matrix @ np.hstack((self.rotation, self.translation.reshape(3, 1)))


@dataclass(slots=True)
class TriangulationResult:
    position_xyz_m: tuple[float, float, float]
    reprojection_error_px: float


class CalibrationStore:
    def __init__(self, document: CalibrationDocument | None = None) -> None:
        self.document = document or CalibrationDocument()
        self.cameras = {
            camera_id: CameraModel(
                camera_id=camera_id,
                image_size=data.image_size,
                camera_matrix=np.asarray(data.camera_matrix, dtype=np.float64),
                distortion=np.asarray(data.distortion_coefficients, dtype=np.float64),
                rotation=np.asarray(data.rotation_world_to_camera, dtype=np.float64),
                translation=np.asarray(data.translation_world_to_camera, dtype=np.float64),
                planar_homography=(
                    np.asarray(data.planar_homography, dtype=np.float64) if data.planar_homography else None
                ),
                camera_forward_axis=data.camera_forward_axis,
            )
            for camera_id, data in self.document.cameras.items()
        }

    @classmethod
    def load(cls, path: str | Path) -> CalibrationStore:
        calibration_path = Path(path)
        if not calibration_path.exists():
            return cls()
        with calibration_path.open(encoding="utf-8") as handle:
            raw: dict[str, Any] = yaml.safe_load(handle) or {}
        return cls(CalibrationDocument.model_validate(raw))

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(self.document.model_dump(mode="json"), handle, sort_keys=False)

    def status(self) -> dict[str, Any]:
        return {
            "valid": len(self.cameras) >= 2,
            "coordinate_system": self.document.coordinate_system,
            "units": self.document.units,
            "camera_ids": sorted(self.cameras),
        }

    def triangulate(
        self,
        observations: dict[int, tuple[float, float]],
        max_reprojection_error_px: float,
    ) -> TriangulationResult | None:
        models = [self.cameras[camera_id] for camera_id in observations if camera_id in self.cameras]
        if len(models) < 2:
            return None

        rows: list[np.ndarray] = []
        undistorted: dict[int, tuple[float, float]] = {}
        for model in models:
            u, v = self._undistort_pixel(observations[model.camera_id], model)
            undistorted[model.camera_id] = (u, v)
            projection = model.projection
            rows.extend((u * projection[2] - projection[0], v * projection[2] - projection[1]))

        _, _, vh = np.linalg.svd(np.asarray(rows))
        homogeneous = vh[-1]
        if abs(homogeneous[3]) < 1e-12:
            return None
        point_3d = homogeneous[:3] / homogeneous[3]

        errors: list[float] = []
        for model in models:
            camera_point = model.rotation @ point_3d + model.translation
            forward_depth = camera_point[2] if model.camera_forward_axis == "positive_z" else -camera_point[2]
            if forward_depth <= 0:
                return None
            projected = model.projection @ np.append(point_3d, 1.0)
            projected_xy = projected[:2] / projected[2]
            errors.append(float(np.linalg.norm(projected_xy - np.asarray(undistorted[model.camera_id]))))

        mean_error = float(np.mean(errors))
        if mean_error > max_reprojection_error_px:
            return None
        return TriangulationResult(tuple(float(value) for value in point_3d), mean_error)

    def planar_position(self, camera_id: int, point: tuple[float, float]) -> tuple[float, float, float] | None:
        model = self.cameras.get(camera_id)
        if model is None or model.planar_homography is None:
            return None
        transformed = model.planar_homography @ np.asarray([point[0], point[1], 1.0])
        if abs(transformed[2]) < 1e-12:
            return None
        return (float(transformed[0] / transformed[2]), float(transformed[1] / transformed[2]), 0.0)

    @staticmethod
    def _undistort_pixel(point: tuple[float, float], model: CameraModel) -> tuple[float, float]:
        """Invert the common OpenCV radial/tangential distortion model."""
        fx, fy = model.camera_matrix[0, 0], model.camera_matrix[1, 1]
        cx, cy = model.camera_matrix[0, 2], model.camera_matrix[1, 2]
        distorted_x, distorted_y = (point[0] - cx) / fx, (point[1] - cy) / fy
        x, y = distorted_x, distorted_y
        coefficients = np.pad(model.distortion.flatten(), (0, max(0, 5 - model.distortion.size)))
        k1, k2, p1, p2, k3 = coefficients[:5]
        for _ in range(8):
            radius2 = x * x + y * y
            radial = 1 + k1 * radius2 + k2 * radius2**2 + k3 * radius2**3
            delta_x = 2 * p1 * x * y + p2 * (radius2 + 2 * x * x)
            delta_y = p1 * (radius2 + 2 * y * y) + 2 * p2 * x * y
            x = (distorted_x - delta_x) / radial
            y = (distorted_y - delta_y) / radial
        return (float(x * fx + cx), float(y * fy + cy))
