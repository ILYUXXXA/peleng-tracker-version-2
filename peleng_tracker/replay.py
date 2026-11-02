"""Deterministic offline 3D reconstruction and replay bundle support."""

from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import yaml
from pydantic import BaseModel, Field, field_validator

from .geometry import CalibrationStore


class ReplayCamera(BaseModel):
    id: int
    name: str
    video: str


class ReplayManifest(BaseModel):
    schema_version: Literal[1] = 1
    id: str
    title: str
    source: str = "local"
    units: Literal["meters"] = "meters"
    coordinate_system: str = "right-handed"
    fps: float = Field(gt=0)
    duration_s: float = Field(ge=0)
    object_id: str
    observation_source: str
    cameras: list[ReplayCamera]
    calibration: str
    observations: str
    ground_truth: str | None = None
    trajectory: str = "trajectory.json"
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("cameras")
    @classmethod
    def require_unique_cameras(cls, value: list[ReplayCamera]) -> list[ReplayCamera]:
        ids = [camera.id for camera in value]
        if len(ids) < 2:
            raise ValueError("replay bundle requires at least two cameras")
        if len(ids) != len(set(ids)):
            raise ValueError("camera ids must be unique")
        return value


class Observation2D(BaseModel):
    frame_index: int = Field(ge=0)
    timestamp: float = Field(ge=0)
    object_id: str
    camera_id: int
    u: float
    v: float
    confidence: float = Field(default=1.0, ge=0, le=1)


class TrackState3D(BaseModel):
    frame_index: int
    timestamp: float
    object_id: str
    position_xyz_m: tuple[float, float, float]
    velocity_xyz_mps: tuple[float, float, float]
    acceleration_xyz_mps2: tuple[float, float, float]
    speed_mps: float
    acceleration_mps2: float
    source_cameras: list[int]
    reprojection_error_px: float | None
    quality: float = Field(ge=0, le=1)
    status: Literal["measured", "predicted"]


class ReplayTrajectory(BaseModel):
    schema_version: Literal[1] = 1
    scenario_id: str
    object_id: str
    states: list[TrackState3D]


class ConstantAccelerationFilter:
    """Small 9D Kalman filter used by replay and future live adapters."""

    def __init__(self, process_variance: float = 1.0, measurement_variance: float = 0.0001) -> None:
        self.process_variance = process_variance
        self.measurement_variance = measurement_variance
        self.state = np.zeros(9, dtype=np.float64)
        self.covariance = np.eye(9, dtype=np.float64)
        self.initialized = False

    @staticmethod
    def _transition(dt: float) -> np.ndarray:
        transition = np.eye(9, dtype=np.float64)
        for axis in range(3):
            transition[axis, axis + 3] = dt
            transition[axis, axis + 6] = 0.5 * dt * dt
            transition[axis + 3, axis + 6] = dt
        return transition

    def initialize(self, position: tuple[float, float, float]) -> None:
        self.state[:3] = position
        self.covariance = np.diag([0.01] * 3 + [1.0] * 3 + [4.0] * 3)
        self.initialized = True

    def predict(self, dt: float) -> np.ndarray:
        if not self.initialized:
            raise RuntimeError("filter must be initialized before prediction")
        transition = self._transition(dt)
        self.state = transition @ self.state
        scale = max(dt, 1e-3) * self.process_variance
        process_noise = np.diag([scale**4] * 3 + [scale**2] * 3 + [scale] * 3)
        self.covariance = transition @ self.covariance @ transition.T + process_noise
        return self.state.copy()

    def update(self, position: tuple[float, float, float]) -> np.ndarray:
        measurement = np.asarray(position, dtype=np.float64)
        observation = np.zeros((3, 9), dtype=np.float64)
        observation[:, :3] = np.eye(3)
        noise = np.eye(3, dtype=np.float64) * self.measurement_variance
        innovation = measurement - observation @ self.state
        innovation_covariance = observation @ self.covariance @ observation.T + noise
        gain = self.covariance @ observation.T @ np.linalg.inv(innovation_covariance)
        self.state += gain @ innovation
        self.covariance = (np.eye(9) - gain @ observation) @ self.covariance
        return self.state.copy()


@dataclass(slots=True)
class ReplayBundle:
    root: Path
    manifest: ReplayManifest
    calibration: CalibrationStore
    trajectory: ReplayTrajectory | None

    @staticmethod
    def resolve_path(root: Path, relative: str) -> Path:
        candidate = (root / relative).resolve()
        if candidate != root and root not in candidate.parents:
            raise ValueError(f"bundle path escapes root: {relative}")
        return candidate

    @classmethod
    def load(cls, path: str | Path, *, require_trajectory: bool = False) -> ReplayBundle:
        root = Path(path).resolve()
        manifest_path = root / "scenario.yaml"
        if not manifest_path.is_file():
            raise ValueError(f"scenario.yaml not found in {root}")
        with manifest_path.open(encoding="utf-8") as handle:
            manifest = ReplayManifest.model_validate(yaml.safe_load(handle) or {})
        calibration_path = cls.resolve_path(root, manifest.calibration)
        calibration = CalibrationStore.load(calibration_path)
        trajectory_path = cls.resolve_path(root, manifest.trajectory)
        trajectory = None
        if trajectory_path.is_file():
            with trajectory_path.open(encoding="utf-8") as handle:
                trajectory = ReplayTrajectory.model_validate(json.load(handle))
        elif require_trajectory:
            raise ValueError(f"processed trajectory not found: {trajectory_path}")
        return cls(root, manifest, calibration, trajectory)

    def validate(self, *, require_trajectory: bool = False) -> dict[str, Any]:
        files = [self.manifest.calibration, self.manifest.observations]
        files.extend(camera.video for camera in self.manifest.cameras)
        if self.manifest.ground_truth:
            files.append(self.manifest.ground_truth)
        missing = [item for item in files if not self.resolve_path(self.root, item).is_file()]
        camera_ids = {camera.id for camera in self.manifest.cameras}
        calibrated_ids = set(self.calibration.cameras)
        if missing:
            raise ValueError(f"missing bundle files: {', '.join(missing)}")
        if not camera_ids.issubset(calibrated_ids):
            absent = sorted(camera_ids - calibrated_ids)
            raise ValueError(f"missing calibration for cameras: {absent}")
        observations = self.read_observations()
        unknown = sorted({item.camera_id for item in observations} - camera_ids)
        if unknown:
            raise ValueError(f"observations reference unknown cameras: {unknown}")
        if require_trajectory and self.trajectory is None:
            raise ValueError("bundle has not been processed")
        return {
            "valid": True,
            "scenario_id": self.manifest.id,
            "camera_ids": sorted(camera_ids),
            "observation_count": len(observations),
            "state_count": len(self.trajectory.states) if self.trajectory else 0,
            "processed": self.trajectory is not None,
        }

    def read_observations(self) -> list[Observation2D]:
        path = self.resolve_path(self.root, self.manifest.observations)
        with path.open(newline="", encoding="utf-8") as handle:
            return [Observation2D.model_validate(row) for row in csv.DictReader(handle)]

    def process(
        self,
        *,
        max_reprojection_error_px: float = 12.0,
        prediction_ttl_s: float = 0.5,
    ) -> ReplayTrajectory:
        observations = [item for item in self.read_observations() if item.object_id == self.manifest.object_id]
        if not observations:
            raise ValueError(f"no observations for object {self.manifest.object_id}")
        by_frame: dict[int, list[Observation2D]] = {}
        for observation in observations:
            by_frame.setdefault(observation.frame_index, []).append(observation)

        kalman = ConstantAccelerationFilter()
        states: list[TrackState3D] = []
        first_frame, last_frame = min(by_frame), max(by_frame)
        dt = 1.0 / self.manifest.fps
        missing_time = 0.0
        last_timestamp = first_frame * dt
        for frame_index in range(first_frame, last_frame + 1):
            frame_observations = by_frame.get(frame_index, [])
            timestamp = frame_observations[0].timestamp if frame_observations else frame_index * dt
            elapsed = max(dt, timestamp - last_timestamp) if states else dt
            estimate = kalman.predict(elapsed) if kalman.initialized else None

            pixels = {item.camera_id: (item.u, item.v) for item in frame_observations}
            triangulation = self.calibration.triangulate(pixels, max_reprojection_error_px)
            if triangulation is not None:
                if not kalman.initialized:
                    kalman.initialize(triangulation.position_xyz_m)
                else:
                    kalman.update(triangulation.position_xyz_m)
                estimate = kalman.state.copy()
                missing_time = 0.0
                status: Literal["measured", "predicted"] = "measured"
                sources = sorted(pixels)
                reprojection = triangulation.reprojection_error_px
                confidence = sum(item.confidence for item in frame_observations) / len(frame_observations)
                geometry_quality = max(0.0, 1.0 - reprojection / max(max_reprojection_error_px, 1e-6))
                quality = float(np.clip(confidence * geometry_quality, 0.0, 1.0))
            elif estimate is not None:
                missing_time += elapsed
                if missing_time > prediction_ttl_s:
                    last_timestamp = timestamp
                    continue
                status = "predicted"
                sources = []
                reprojection = None
                quality = float(max(0.0, 1.0 - missing_time / prediction_ttl_s) * 0.5)
            else:
                last_timestamp = timestamp
                continue

            assert estimate is not None
            velocity = estimate[3:6]
            acceleration = estimate[6:9]
            states.append(
                TrackState3D(
                    frame_index=frame_index,
                    timestamp=timestamp,
                    object_id=self.manifest.object_id,
                    position_xyz_m=tuple(float(value) for value in estimate[:3]),
                    velocity_xyz_mps=tuple(float(value) for value in velocity),
                    acceleration_xyz_mps2=tuple(float(value) for value in acceleration),
                    speed_mps=float(np.linalg.norm(velocity)),
                    acceleration_mps2=float(np.linalg.norm(acceleration)),
                    source_cameras=sources,
                    reprojection_error_px=reprojection,
                    quality=quality,
                    status=status,
                )
            )
            last_timestamp = timestamp

        trajectory = ReplayTrajectory(scenario_id=self.manifest.id, object_id=self.manifest.object_id, states=states)
        destination = self.resolve_path(self.root, self.manifest.trajectory)
        destination.write_text(trajectory.model_dump_json(indent=2), encoding="utf-8")
        self.trajectory = trajectory
        return trajectory

    def evaluation(self) -> dict[str, float | int | None]:
        if self.trajectory is None:
            raise ValueError("bundle has not been processed")
        if not self.manifest.ground_truth:
            return {"state_count": len(self.trajectory.states), "median_error_m": None, "mean_error_m": None}
        ground_truth_path = self.resolve_path(self.root, self.manifest.ground_truth)
        with ground_truth_path.open(newline="", encoding="utf-8") as handle:
            truth = {
                int(row["frame_index"]): np.asarray([row["x"], row["y"], row["z"]], dtype=float)
                for row in csv.DictReader(handle)
            }
        errors = [
            float(np.linalg.norm(np.asarray(state.position_xyz_m) - truth[state.frame_index]))
            for state in self.trajectory.states
            if state.frame_index in truth and state.status == "measured"
        ]
        if not errors:
            return {"state_count": len(self.trajectory.states), "median_error_m": None, "mean_error_m": None}
        return {
            "state_count": len(self.trajectory.states),
            "evaluated_count": len(errors),
            "median_error_m": float(np.median(errors)),
            "mean_error_m": float(np.mean(errors)),
            "p95_error_m": float(np.percentile(errors, 95)),
            "coverage": len(errors) / max(1, len(truth)),
        }

    def export_csv(self) -> str:
        if self.trajectory is None:
            raise ValueError("bundle has not been processed")
        columns = [
            "frame_index",
            "timestamp",
            "object_id",
            "x",
            "y",
            "z",
            "vx",
            "vy",
            "vz",
            "ax",
            "ay",
            "az",
            "speed_mps",
            "acceleration_mps2",
            "quality",
            "status",
        ]
        rows = [",".join(columns)]
        for state in self.trajectory.states:
            values: list[Any] = [
                state.frame_index,
                state.timestamp,
                state.object_id,
                *state.position_xyz_m,
                *state.velocity_xyz_mps,
                *state.acceleration_xyz_mps2,
                state.speed_mps,
                state.acceleration_mps2,
                state.quality,
                state.status,
            ]
            rows.append(",".join(str(value) for value in values))
        return "\n".join(rows) + "\n"


class ReplayRepository:
    def __init__(self) -> None:
        self.bundle: ReplayBundle | None = None

    def load(self, path: str | Path) -> ReplayBundle:
        self.bundle = ReplayBundle.load(path, require_trajectory=True)
        self.bundle.validate(require_trajectory=True)
        return self.bundle

    def load_from_environment(self) -> None:
        path = os.getenv("PELENG_REPLAY_BUNDLE")
        if path and self.bundle is None:
            self.load(path)

    def require(self) -> ReplayBundle:
        self.load_from_environment()
        if self.bundle is None:
            raise ValueError("no replay bundle loaded")
        return self.bundle


replay_repository = ReplayRepository()
