"""Conversion of EPFL POM homographies into Peleng Tracker calibration."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .geometry import CalibrationDocument, CameraCalibrationData


def parse_epfl_calibration(path: str | Path) -> list[dict[str, np.ndarray | float | None]]:
    """Parse the ground/head-plane format used by EPFL pedestrian datasets."""
    cameras: list[dict[str, np.ndarray | float | None]] = []
    current: dict[str, np.ndarray | float | None] | None = None
    reading: str | None = None
    rows: list[list[float]] = []

    for raw_line in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line.startswith("# Camera"):
            if current is not None:
                cameras.append(current)
            current = {"ground": None, "head": None, "head_plane_height": None}
            reading = None
            rows = []
        elif "Ground plane homography" in line:
            reading, rows = "ground", []
        elif "Head plane homography" in line:
            reading, rows = "head", []
        elif "Head plane height" in line:
            reading, rows = "head_plane_height", []
        elif line and not line.startswith("#") and current is not None:
            if reading in {"ground", "head"}:
                values = [float(value) for value in line.split()]
                if len(values) != 3:
                    raise ValueError(f"expected three matrix values, got: {line}")
                rows.append(values)
                if len(rows) == 3:
                    current[reading] = np.asarray(rows, dtype=np.float64)
                    reading, rows = None, []
            elif reading == "head_plane_height":
                current[reading] = float(line)
                reading = None

    if current is not None:
        cameras.append(current)
    if not cameras or any(camera["ground"] is None for camera in cameras):
        raise ValueError("no complete EPFL camera calibration found")
    return cameras


def build_projection_matrix(
    camera: dict[str, np.ndarray | float | None],
    *,
    head_height_m: float,
    ground_units_per_meter: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return metric world-to-image ground homography and 3x4 projection."""
    image_to_ground = np.asarray(camera["ground"], dtype=np.float64)
    head_value = camera["head"]
    if head_value is None:
        raise ValueError("camera has no head-plane homography; full 3D projection is unavailable")
    image_to_head = np.asarray(head_value, dtype=np.float64)
    scale = np.diag([ground_units_per_meter, ground_units_per_meter, 1.0])
    ground_metric = np.linalg.inv(image_to_ground) @ scale
    head_metric = np.linalg.inv(image_to_head) @ scale

    head_xy = head_metric[:, :2].ravel()
    ground_xy = ground_metric[:, :2].ravel()
    denominator = float(np.dot(head_xy, head_xy))
    if denominator <= 1e-12:
        raise ValueError("head-plane homography is degenerate")
    relative_scale = float(np.dot(head_xy, ground_xy) / denominator)
    depth_column = (relative_scale * head_metric[:, 2] - ground_metric[:, 2]) / head_height_m
    projection = np.column_stack(
        [ground_metric[:, 0], ground_metric[:, 1], depth_column, ground_metric[:, 2]]
    )
    if np.linalg.det(projection[:, :3]) < 0:
        projection = -projection
    return ground_metric, projection


def decompose_projection(projection: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decompose P into OpenCV K, world-to-camera R and t."""
    camera_matrix, rotation, center_h, *_ = cv2.decomposeProjectionMatrix(projection)
    camera_matrix = camera_matrix / camera_matrix[2, 2]
    center = center_h[:3, 0] / center_h[3, 0]
    translation = -rotation @ center
    return camera_matrix, rotation, translation


def convert_epfl_calibration(
    source: str | Path,
    *,
    camera_ids: list[int],
    image_size: tuple[int, int] = (360, 288),
    head_height_m: float = 1.75,
    ground_units_per_meter: float = 100.0,
) -> CalibrationDocument:
    """Convert selected EPFL cameras into the Peleng Tracker schema."""
    parsed = parse_epfl_calibration(source)
    converted: dict[int, CameraCalibrationData] = {}
    for camera_id in camera_ids:
        if camera_id < 0 or camera_id >= len(parsed):
            raise ValueError(f"camera {camera_id} is not present in the EPFL file")
        ground, projection = build_projection_matrix(
            parsed[camera_id],
            head_height_m=head_height_m,
            ground_units_per_meter=ground_units_per_meter,
        )
        camera_matrix, rotation, translation = decompose_projection(projection)
        converted[camera_id] = CameraCalibrationData(
            image_size=image_size,
            camera_matrix=camera_matrix.tolist(),
            distortion_coefficients=[0.0, 0.0, 0.0, 0.0, 0.0],
            rotation_world_to_camera=rotation.tolist(),
            translation_world_to_camera=translation.tolist(),
            planar_homography=np.linalg.inv(ground).tolist(),
            camera_forward_axis="negative_z",
        )
    return CalibrationDocument(coordinate_system="right-handed", units="meters", cameras=converted)
