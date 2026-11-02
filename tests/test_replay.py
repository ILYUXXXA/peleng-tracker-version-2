import csv
from pathlib import Path

import numpy as np
import pytest
import yaml
from fastapi.testclient import TestClient

from peleng_tracker.core.app import create_app
from peleng_tracker.geometry import CalibrationDocument, CalibrationStore, CameraCalibrationData
from peleng_tracker.replay import ReplayBundle, replay_repository


def _camera(translation: list[float]) -> CameraCalibrationData:
    return CameraCalibrationData(
        image_size=(640, 480),
        camera_matrix=[[500, 0, 320], [0, 500, 240], [0, 0, 1]],
        distortion_coefficients=[0, 0, 0, 0, 0],
        rotation_world_to_camera=np.eye(3).tolist(),
        translation_world_to_camera=translation,
    )


def _project(store: CalibrationStore, camera_id: int, point: np.ndarray) -> tuple[float, float]:
    projected = store.cameras[camera_id].projection @ np.append(point, 1.0)
    return (float(projected[0] / projected[2]), float(projected[1] / projected[2]))


def build_bundle(root: Path) -> Path:
    calibration = CalibrationStore(
        CalibrationDocument(cameras={0: _camera([0, 0, 0]), 1: _camera([-1, 0, 0])})
    )
    calibration.save(root / "calibration.yaml")
    (root / "camera-0.mp4").write_bytes(b"video-0")
    (root / "camera-1.mp4").write_bytes(b"video-1")
    points = {
        0: np.asarray([0.0, 0.0, 5.0]),
        1: np.asarray([0.1, 0.0, 5.0]),
        3: np.asarray([0.3, 0.1, 5.0]),
    }
    with (root / "observations.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["frame_index", "timestamp", "object_id", "camera_id", "u", "v", "confidence"],
        )
        writer.writeheader()
        for frame_index, point in points.items():
            for camera_id in (0, 1):
                u, v = _project(calibration, camera_id, point)
                writer.writerow(
                    {
                        "frame_index": frame_index,
                        "timestamp": frame_index / 10,
                        "object_id": "drone-1",
                        "camera_id": camera_id,
                        "u": u,
                        "v": v,
                        "confidence": 1,
                    }
                )
    with (root / "ground-truth.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["frame_index", "x", "y", "z"])
        writer.writeheader()
        for frame_index, point in points.items():
            writer.writerow({"frame_index": frame_index, "x": point[0], "y": point[1], "z": point[2]})
    manifest = {
        "schema_version": 1,
        "id": "known-flight",
        "title": "Known flight",
        "fps": 10,
        "duration_s": 0.3,
        "object_id": "drone-1",
        "observation_source": "test",
        "cameras": [
            {"id": 0, "name": "left", "video": "camera-0.mp4"},
            {"id": 1, "name": "right", "video": "camera-1.mp4"},
        ],
        "calibration": "calibration.yaml",
        "observations": "observations.csv",
        "ground_truth": "ground-truth.csv",
    }
    (root / "scenario.yaml").write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
    return root


def test_replay_bundle_processes_3d_trajectory_and_prediction(tmp_path: Path) -> None:
    bundle = ReplayBundle.load(build_bundle(tmp_path))
    assert bundle.validate()["observation_count"] == 6

    trajectory = bundle.process(max_reprojection_error_px=0.1)

    assert [state.frame_index for state in trajectory.states] == [0, 1, 2, 3]
    assert trajectory.states[2].status == "predicted"
    assert trajectory.states[0].position_xyz_m == pytest.approx((0.0, 0.0, 5.0))
    assert bundle.evaluation()["median_error_m"] is not None
    assert "speed_mps" in bundle.export_csv().splitlines()[0]


def test_replay_api_serves_processed_bundle(tmp_path: Path) -> None:
    bundle = ReplayBundle.load(build_bundle(tmp_path))
    bundle.process(max_reprojection_error_px=0.1)
    replay_repository.bundle = bundle
    client = TestClient(create_app())

    assert client.get("/api/replay").json()["manifest"]["id"] == "known-flight"
    assert len(client.get("/api/replay/trajectory").json()["states"]) == 4
    assert client.get("/api/replay/export.csv").status_code == 200
    assert client.get("/api/replay/cameras/0/video").content == b"video-0"
    replay_repository.bundle = None
