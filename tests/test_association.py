from __future__ import annotations

from dataclasses import dataclass

import pytest

from peleng_tracker.association import CrossCameraAssociator
from peleng_tracker.domain import Track
from peleng_tracker.geometry import TriangulationResult


@dataclass
class SyntheticCalibration:
    valid_pairs: dict[tuple[tuple[float, float], tuple[float, float]], float]

    def triangulate(
        self,
        observations: dict[int, tuple[float, float]],
        max_reprojection_error_px: float,
    ) -> TriangulationResult | None:
        if set(observations) != {0, 1}:
            return None
        left, right = observations[0], observations[1]
        error = self.valid_pairs.get((left, right))
        if error is None or error > max_reprojection_error_px:
            return None
        return TriangulationResult(((left[0] + right[0]) / 2.0, 0.0, 0.0), error)

    @staticmethod
    def planar_position(camera_id: int, point: tuple[float, float]) -> tuple[float, float, float]:
        del camera_id
        return (point[0] / 10.0, point[1] / 10.0, 0.0)


def person(local_id: str, camera_id: int, ground_x: float) -> Track:
    return Track(local_id, camera_id, (int(ground_x * 10), -20, 0, 20), 0.9, 0, "person")


def test_person_geometry_uses_bbox_bottom_center_without_changing_display_center() -> None:
    track = Track("1", 0, (10, 20, 30, 40), 0.9, 0, "person")

    assert track.bbox_center == (25.0, 40.0)
    assert track.ground_anchor == (25.0, 60.0)
    assert track.anchor == track.bbox_center
    assert track.geometry_anchor == track.ground_anchor


def test_global_assignment_fuses_all_people_independently_of_input_order() -> None:
    left = [person("a", 0, 0.0), person("b", 0, 0.2), person("c", 0, 1.0), person("d", 0, 2.0)]
    right = [person("x", 1, 0.1), person("y", 1, -0.2), person("z", 1, 1.1), person("w", 1, 2.1)]
    anchors = {track.local_track_id: track.ground_anchor for track in left + right}
    calibration = SyntheticCalibration(
        {
            (anchors["a"], anchors["x"]): 1.0,
            (anchors["a"], anchors["y"]): 2.0,
            (anchors["b"], anchors["x"]): 1.0,
            (anchors["c"], anchors["z"]): 1.0,
            (anchors["d"], anchors["w"]): 1.0,
        }
    )

    expected_pairs = {("a", "y"), ("b", "x"), ("c", "z"), ("d", "w")}
    observed_pair_sets = []
    observed_global_ids = []
    for left_order, right_order in ((left, right), (list(reversed(left)), [right[2], right[0], right[3], right[1]])):
        associator = CrossCameraAssociator(calibration, max_reprojection_error_px=5.0, max_ground_distance_m=0.5)
        world_tracks = associator.associate({0: left_order, 1: right_order}, timestamp=1.0)

        assert len(world_tracks) == 4
        assert all(world_track.source_cameras == [0, 1] for world_track in world_tracks)
        observed_pair_sets.append(
            {
                tuple(key.split(":", 1)[1] for key in world_track.source_track_keys)
                for world_track in world_tracks
            }
        )
        observed_global_ids.append(
            {tuple(key.split(":", 1)[1] for key in track.source_track_keys): track.global_id for track in world_tracks}
        )

    assert observed_pair_sets == [expected_pairs, expected_pairs]
    assert observed_global_ids[0] == observed_global_ids[1]


@pytest.mark.parametrize(
    ("right_x", "reprojection_error"),
    [(1.0, 6.0), (10.0, 1.0)],
    ids=["reprojection-gate", "ground-distance-gate"],
)
def test_invalid_pair_is_rejected_by_each_geometry_gate(right_x: float, reprojection_error: float) -> None:
    left = person("a", 0, 0.0)
    right = person("x", 1, right_x)
    calibration = SyntheticCalibration({(left.ground_anchor, right.ground_anchor): reprojection_error})
    associator = CrossCameraAssociator(calibration, max_reprojection_error_px=5.0, max_ground_distance_m=0.5)

    world_tracks = associator.associate({0: [left], 1: [right]})

    assert len(world_tracks) == 2
    assert all(len(track.source_cameras) == 1 for track in world_tracks)


def test_single_camera_fallback_and_pairing_diagnostics_are_serializable() -> None:
    left = person("a", 0, 0.0)
    right = person("x", 1, 0.1)
    extra = person("b", 0, 5.0)
    calibration = SyntheticCalibration({((0.0, 0.0), (1.0, 0.0)): 1.0})
    associator = CrossCameraAssociator(calibration, max_reprojection_error_px=5.0, max_ground_distance_m=0.5)

    world_tracks = associator.associate({0: [left, extra], 1: [right]}, timestamp=1.0)

    fused = next(track for track in world_tracks if len(track.source_cameras) == 2)
    fallback = next(track for track in world_tracks if len(track.source_cameras) == 1)
    assert fused.source_track_keys == ["0:a", "1:x"]
    assert fused.association_pairs[0].reprojection_error_px == pytest.approx(1.0)
    assert fused.association_pairs[0].ground_distance_m == pytest.approx(0.1)
    assert fallback.position_status == "planar_fallback"
    assert associator.last_diagnostics.local_tracks_by_camera == {0: 2, 1: 1}
    assert associator.last_diagnostics.fused_world_tracks == 1
    assert associator.last_diagnostics.single_camera_fallback_tracks == 1
