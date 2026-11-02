import csv
from pathlib import Path

import numpy as np
import pytest
import yaml

from peleng_tracker import dpjait
from peleng_tracker.geometry import CalibrationStore
from peleng_tracker.replay import ReplayBundle

CALIBRATION_HEADER = (
    "cam_name;cam_x[mm];cam_y[mm];cam_z[mm];cam_or_x[rad];cam_or_y[rad];cam_or_z[rad];"
    "focal_length;cx;cy;coeff_1;coeff_2;coeff_3;coeff_4\n"
)


def _write_sequence(root: Path) -> Path:
    sequence = root / "Real_Data" / "R02_D1"
    (sequence / "dl_data").mkdir(parents=True)
    (sequence / "cameras_calibration.csv").write_text(
        CALIBRATION_HEADER
        + "100;1000;2000;3000;0;0;0;500;320;240;-0.1;0.01;0;0\n"
        + "200;-1000;2000;3000;0;0;0;500;320;240;-0.1;0.01;0;0\n",
        encoding="utf-8",
    )
    for name, u in (("100", 300), ("200", 340)):
        (sequence / f"{name}.avi").write_bytes(b"avi")
        (sequence / "dl_data" / f"{name}.csv").write_text(
            f"0,0,0,10,10,{u},240,0.9\n1,0,0,10,10,{u + 1},240,0.8\n",
            encoding="utf-8",
        )
    (sequence / "R02_D1.c3d").write_bytes(b"c3d")
    return sequence


def test_read_dpjait_calibration_interprets_translation_as_opencv_tvec(tmp_path: Path) -> None:
    sequence = _write_sequence(tmp_path)

    document, names = dpjait.read_dpjait_calibration(sequence / "cameras_calibration.csv")

    assert names == ["100", "200"]
    assert document.cameras[0].translation_world_to_camera == [1.0, 2.0, 3.0]
    assert document.cameras[0].distortion_coefficients == []
    assert np.asarray(document.cameras[0].rotation_world_to_camera) == pytest.approx(np.eye(3))


def test_import_dpjait_sequence_builds_replay_bundle(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    sequence = _write_sequence(tmp_path / "source")
    output = tmp_path / "bundle"

    def fake_ground_truth(
        path: str | Path,
        *,
        frame_count: int,
        video_fps: float = dpjait.VIDEO_FPS,
        offset_samples: int = -3,
        visual_center_below_markers_m: float = dpjait.VISUAL_CENTER_BELOW_MARKERS_M,
    ) -> tuple[list[np.ndarray], float, list[str]]:
        assert Path(path).name == "R02_D1.c3d"
        assert video_fps == 25.0
        assert offset_samples == -3
        assert visual_center_below_markers_m == 0.1
        return [np.asarray([0.0, 0.0, 5.0])] * frame_count, 100.0, ["A", "B", "C", "D"]

    monkeypatch.setattr(dpjait, "read_marker_centroid_ground_truth", fake_ground_truth)

    def fake_converter(source: Path, destination: Path) -> None:
        assert source.suffix == ".avi"
        destination.write_bytes(b"mp4")

    result = dpjait.import_dpjait_sequence(
        sequence,
        output,
        video_converter=fake_converter,
    )

    assert result.camera_count == 2
    assert result.frame_count == 2
    assert result.observation_count == 4
    bundle = ReplayBundle.load(output)
    assert bundle.validate()["observation_count"] == 4
    assert bundle.manifest.metadata["visual_center_below_markers_m"] == 0.1
    assert CalibrationStore.load(output / "calibration.yaml").cameras[0].translation.tolist() == [1.0, 2.0, 3.0]
    manifest = yaml.safe_load((output / "scenario.yaml").read_text(encoding="utf-8"))
    assert manifest["cameras"][0]["video"] == "camera-0.mp4"


def test_read_detection_rows_rejects_incomplete_rows(tmp_path: Path) -> None:
    path = tmp_path / "detections.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        csv.writer(handle).writerow([0, 1, 2])

    with pytest.raises(ValueError, match="invalid DPJAIT detection row"):
        dpjait.read_detection_rows(path)
