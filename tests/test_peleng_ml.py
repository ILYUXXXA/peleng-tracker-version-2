from pathlib import Path

import numpy as np
import pytest

from peleng_tracker.domain import Track
from scripts.peleng_ml import Camera, triangulate


def make_camera(path: Path, camera_id: int, y_mm: float) -> Camera:
    path.write_text(
        "Frame;Loc_X;Loc_Y;Loc_Z;Rot_Roll;Rot_Pitch;Rot_Yaw;Focal length;35\n"
        f"0;0;{y_mm};0;0;0;0;;;;\n",
        encoding="cp1251",
    )
    return Camera(camera_id, path)


def test_custom_class_zero_uses_bbox_center_not_person_ground_anchor() -> None:
    track = Track("1", 0, (10, 20, 30, 40), 0.9, 0, "target")
    assert track.geometry_anchor == (25.0, 40.0)


def test_person_still_uses_ground_contact_anchor() -> None:
    track = Track("1", 0, (10, 20, 30, 40), 0.9, 0, "person")
    assert track.geometry_anchor == (25.0, 60.0)


def test_peleng_camera_projection_triangulates_back_to_world_point(tmp_path: Path) -> None:
    left = make_camera(tmp_path / "left.csv", 0, 0)
    right = make_camera(tmp_path / "right.csv", 1, 1000)
    point_mm = np.array([5000.0, 200.0, 300.0])
    observations = []
    for camera in (left, right):
        projected = camera.project(point_mm)
        assert projected is not None
        observations.append((camera, projected[0], projected[1]))
    recovered = triangulate(observations)
    assert recovered is not None
    expected = np.array([5.0, -0.2, 0.3])
    assert recovered == pytest.approx(expected, abs=1e-6)
