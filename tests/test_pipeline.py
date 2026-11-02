import numpy as np
import pytest

from peleng_tracker.association import CrossCameraAssociator
from peleng_tracker.domain import Detection, FramePacket, Track
from peleng_tracker.geometry import CalibrationStore
from peleng_tracker.pipeline import VisionPipeline
from peleng_tracker.targets import TargetManager


class Detector:
    def detect(self, frame):
        return [Detection((1, 2, 3, 4), 0.9, 2, "car")]


class Tracker:
    def update(self, detections, frame, camera_id, timestamp):
        return [Track("1", camera_id, detections[0].bbox, 0.9, 2, "car")]

    def reset(self):
        pass


def test_local_ids_are_scoped_by_camera_and_skew_is_reported() -> None:
    pipeline = VisionPipeline(Detector(), Tracker(), CrossCameraAssociator(CalibrationStore(), 5), TargetManager(), 50)
    frame = np.zeros((10, 10, 3), dtype=np.uint8)
    result = pipeline.process([FramePacket(0, frame, 1, 1.0), FramePacket(1, frame, 1, 1.01)])
    assert result is not None and result.sync_skew_ms == pytest.approx(10)
    assert result.tracks[0][0].key != result.tracks[1][0].key


def test_selected_local_track_is_marked_with_global_target() -> None:
    targets = TargetManager()
    targets.select_track(0, "1")
    pipeline = VisionPipeline(Detector(), Tracker(), CrossCameraAssociator(CalibrationStore(), 5), targets, 50)
    frame = np.zeros((10, 10, 3), dtype=np.uint8)

    result = pipeline.process([FramePacket(0, frame, 1, 1.0)])

    assert result is not None
    assert result.world_tracks[0].is_target
    assert result.tracks[0][0].global_id == result.world_tracks[0].global_id
    assert result.tracks[0][0].is_target
