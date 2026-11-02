"""Runtime facade around the framework-independent vision pipeline."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np

from ..association import CrossCameraAssociator
from ..domain import FramePacket, VisionResult
from ..factory import create_detector, create_local_tracker
from ..geometry import CalibrationDocument, CalibrationStore
from ..pipeline import VisionPipeline
from ..settings import VisionSettings
from ..targets import TargetManager


class VisionAPI:
    def __init__(self, settings: VisionSettings, *, warmup: bool = True) -> None:
        self.settings = settings
        self.targets = TargetManager()
        self.is_enabled = False
        self.last_processed_at = 0.0
        self.cached_result: VisionResult | None = None
        self.processing_times: list[float] = []
        self.pipeline = self._build_pipeline(settings, warmup=warmup)

    def _build_pipeline(self, settings: VisionSettings, *, warmup: bool) -> VisionPipeline:
        detector = create_detector(settings)
        if warmup:
            detector.warmup()
        tracker = create_local_tracker(settings)
        calibration = CalibrationStore.load(settings.geometry.calibration_file)
        associator = CrossCameraAssociator(
            calibration,
            settings.geometry.max_reprojection_error_px,
            settings.geometry.max_ground_distance_m,
        )
        return VisionPipeline(
            detector,
            tracker,
            associator,
            self.targets,
            settings.geometry.max_sync_skew_ms,
        )

    @property
    def detector(self):
        return self.pipeline.detector

    @property
    def tracker(self):
        return self.pipeline.tracker

    def enable_tracking(self) -> None:
        self.is_enabled = True

    def disable_tracking(self) -> None:
        self.is_enabled = False
        self.cached_result = None

    def is_tracking_enabled(self) -> bool:
        return self.is_enabled

    def process_frames(self, packets: list[FramePacket]) -> VisionResult | None:
        if not self.is_enabled:
            return None
        now = time.monotonic()
        interval = 1.0 / self.settings.fps
        if self.cached_result is not None and now - self.last_processed_at < interval:
            return self.cached_result
        self.last_processed_at = now
        result = self.pipeline.process(packets)
        if not self.is_enabled:
            return None
        if result is not None:
            self.cached_result = result
            self.processing_times.append(result.processing_time_ms)
            self.processing_times = self.processing_times[-100:]
        return result

    def process_combined_frame(
        self,
        frame: np.ndarray,
        timestamp: float,
        num_streams: int,
        stream_ids: list[int] | None = None,
    ) -> VisionResult | None:
        """Compatibility wrapper; new callers should pass individual frames."""
        stream_ids = stream_ids or list(range(num_streams))
        height, width = frame.shape[:2]
        columns = max(1, int(np.ceil(np.sqrt(num_streams))))
        rows = max(1, int(np.ceil(num_streams / columns)))
        cell_width, cell_height = width // columns, height // rows
        packets = []
        for index, camera_id in enumerate(stream_ids):
            column, row = index % columns, index // columns
            crop = frame[row * cell_height : (row + 1) * cell_height, column * cell_width : (column + 1) * cell_width]
            packets.append(FramePacket(camera_id, crop, self.pipeline.frame_id + 1, timestamp))
        return self.process_frames(packets)

    def restart(self, settings: VisionSettings | None = None) -> bool:
        candidate_settings = settings or self.settings
        candidate = self._build_pipeline(candidate_settings, warmup=True)
        self.pipeline = candidate
        self.settings = candidate_settings
        self.cached_result = None
        self.processing_times.clear()
        return True

    def get_statistics(self) -> dict[str, Any]:
        return {
            "enabled": self.is_enabled,
            "model": self.settings.detector.weights,
            "detector_backend": self.settings.detector.backend,
            "tracker_backend": self.settings.tracker.backend,
            "vision_fps": self.settings.fps,
            "avg_processing_time_ms": (
                sum(self.processing_times) / len(self.processing_times) if self.processing_times else 0.0
            ),
            "sync_skew_ms": self.cached_result.sync_skew_ms if self.cached_result else 0.0,
            "calibration": self.pipeline.associator.calibration.status(),
            "target": self.targets.snapshot(),
        }

    def get_config_schema(self) -> dict[str, Any]:
        return VisionSettings.model_json_schema()

    def get_current_config(self) -> dict[str, Any]:
        return self.settings.model_dump(mode="json")

    def update_config(self, update: dict[str, Any]) -> None:
        merged = self.settings.model_dump()
        for section, values in update.items():
            if isinstance(values, dict) and isinstance(merged.get(section), dict):
                merged[section].update(values)
            else:
                merged[section] = values
        self.settings = VisionSettings.model_validate(merged)

    def calibration_document(self) -> CalibrationDocument:
        return self.pipeline.associator.calibration.document

    def replace_calibration(self, document: CalibrationDocument) -> None:
        store = CalibrationStore(document)
        store.save(Path(self.settings.geometry.calibration_file))
        self.pipeline.associator.calibration = store


def create_vision_api(settings: VisionSettings, *, warmup: bool = True) -> VisionAPI:
    return VisionAPI(settings, warmup=warmup)
