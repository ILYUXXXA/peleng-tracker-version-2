from peleng_tracker.domain import WorldTrack
from peleng_tracker.targets import TargetManager


def world(global_id: str, class_id: int = 0) -> WorldTrack:
    return WorldTrack(global_id, class_id, "person", 0.9, [0])


def test_class_target_lost_and_reacquired() -> None:
    manager = TargetManager()
    manager.select_classes([0])
    tracks = [world("g1")]
    manager.apply(tracks, {})
    assert manager.state == "tracking" and tracks[0].is_target
    manager.apply([], {})
    assert manager.state == "lost"
    manager.apply([world("g2")], {})
    assert manager.state == "reacquiring"


def test_track_selection_uses_camera_scoped_local_id() -> None:
    manager = TargetManager()
    manager.select_track(1, "7")
    tracks = [world("right")]
    manager.apply(tracks, {(0, "7"): "wrong", (1, "7"): "right"})
    assert tracks[0].is_target
