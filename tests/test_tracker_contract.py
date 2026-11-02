import numpy as np

from peleng_tracker.domain import Detection
from peleng_tracker.local_trackers.botsort import BoTSortBackend
from peleng_tracker.local_trackers.deepsort import DeepSortBackend
from peleng_tracker.settings import TrackerSettings


class FakeBotTracker:
    def update(self, payload, frame):
        assert len(payload) == 1 and frame.shape == (100, 100, 3)
        return np.asarray([[10, 20, 40, 70, 5, 0.8, 2, 0]])


def detection() -> Detection:
    return Detection((10, 20, 30, 50), 0.8, 2, "car")


def assert_contract(track, expected_id: str) -> None:
    assert track.local_track_id == expected_id
    assert track.bbox == (10, 20, 30, 50)
    assert (track.class_id, track.class_name) == (2, "car")


def test_deepsort_contract_and_camera_isolation() -> None:
    backend = DeepSortBackend(TrackerSettings())
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    left = backend.update([detection()], frame, 0, 1.0)[0]
    right = backend.update([detection()], frame, 1, 1.0)[0]
    assert_contract(left, "1")
    assert left.key != right.key


def test_botsort_contract_and_camera_isolation() -> None:
    backend = BoTSortBackend(TrackerSettings(backend="botsort"), tracker_factory=FakeBotTracker)
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    left = backend.update([detection()], frame, 0, 1.0)[0]
    right = backend.update([detection()], frame, 1, 1.0)[0]
    assert_contract(left, "5")
    assert left.key != right.key


def test_botsort_is_default_and_real_adapter_persists_id() -> None:
    settings = TrackerSettings()
    assert settings.backend == "botsort"
    backend = BoTSortBackend(settings)
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    ids = [backend.update([detection()], frame, 0, index / 25)[0].local_track_id for index in range(3)]
    assert ids == ["1", "1", "1"]
