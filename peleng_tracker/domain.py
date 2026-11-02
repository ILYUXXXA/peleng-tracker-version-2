"""Framework-independent data contracts for the vision pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np


@dataclass(slots=True)
class FramePacket:
    camera_id: int
    frame: np.ndarray
    sequence: int
    captured_at: float


@dataclass(slots=True)
class Detection:
    bbox: tuple[int, int, int, int]
    confidence: float
    class_id: int
    class_name: str

    @property
    def anchor(self) -> tuple[float, float]:
        return self.bbox_center

    @property
    def bbox_center(self) -> tuple[float, float]:
        x, y, width, height = self.bbox
        return (x + width / 2.0, y + height / 2.0)


@dataclass(slots=True)
class Track:
    local_track_id: str
    camera_id: int
    bbox: tuple[int, int, int, int]
    confidence: float
    class_id: int
    class_name: str
    age: int = 1
    global_id: str | None = None
    is_target: bool = False

    @property
    def key(self) -> tuple[int, str]:
        return (self.camera_id, self.local_track_id)

    @property
    def anchor(self) -> tuple[float, float]:
        """Display anchor kept for schema compatibility."""
        return self.bbox_center

    @property
    def bbox_center(self) -> tuple[float, float]:
        x, y, width, height = self.bbox
        return (x + width / 2.0, y + height / 2.0)

    @property
    def ground_anchor(self) -> tuple[float, float]:
        x, y, width, height = self.bbox
        return (x + width / 2.0, y + height)

    @property
    def geometry_anchor(self) -> tuple[float, float]:
        """Use a ground-contact observation for people and the center otherwise."""
        return self.ground_anchor if self.class_id == 0 else self.bbox_center


PositionStatus = Literal["valid", "planar_fallback", "unavailable"]


@dataclass(slots=True)
class AssociationPair:
    left_camera_id: int
    left_local_track_id: str
    right_camera_id: int
    right_local_track_id: str
    ground_distance_m: float | None
    reprojection_error_px: float
    cost: float


@dataclass(slots=True)
class AssociationDiagnostics:
    local_tracks_by_camera: dict[int, int] = field(default_factory=dict)
    fused_world_tracks: int = 0
    single_camera_fallback_tracks: int = 0


@dataclass(slots=True)
class WorldTrack:
    global_id: str
    class_id: int
    class_name: str
    confidence: float
    source_cameras: list[int]
    position_xyz_m: tuple[float, float, float] | None = None
    velocity_xyz_mps: tuple[float, float, float] = (0.0, 0.0, 0.0)
    reprojection_error_px: float | None = None
    position_status: PositionStatus = "unavailable"
    is_target: bool = False
    trajectory: list[tuple[float, float, float, float]] = field(default_factory=list)
    source_track_keys: list[str] = field(default_factory=list)
    association_pairs: list[AssociationPair] = field(default_factory=list)


@dataclass(slots=True)
class VisionResult:
    frame_id: int
    timestamp: float
    processing_time_ms: float
    active_stream_ids: list[int]
    detections: dict[int, list[Detection]]
    tracks: dict[int, list[Track]]
    world_tracks: list[WorldTrack]
    sync_skew_ms: float = 0.0
    association_diagnostics: AssociationDiagnostics = field(default_factory=AssociationDiagnostics)

    @property
    def num_streams(self) -> int:
        return len(self.active_stream_ids)

    @property
    def all_stream_detections(self) -> dict[int, list[Detection]]:
        return self.detections

    @property
    def all_stream_tracks(self) -> dict[int, list[Track]]:
        return self.tracks

    @property
    def bev_tracks(self) -> list[WorldTrack]:
        """Temporary compatibility alias for schema v1 clients."""
        return self.world_tracks
