"""Import real DPJAIT sequences into deterministic replay bundles."""

from __future__ import annotations

import csv
import shutil
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import ezc3d
import numpy as np
import yaml

from .geometry import CalibrationDocument, CalibrationStore, CameraCalibrationData

VIDEO_FPS = 25.0
VIDEO_SIZE = (1924, 1082)
VISUAL_CENTER_BELOW_MARKERS_M = 0.1


@dataclass(frozen=True, slots=True)
class DPJAITImportResult:
    bundle_path: Path
    camera_count: int
    frame_count: int
    observation_count: int
    vicon_rate_hz: float
    vicon_offset_samples: int


def resolve_real_sequence(source: str | Path, sequence: str) -> Path:
    """Accept the archive root, Real_Data directory, or a sequence directory."""
    root = Path(source).resolve()
    candidates = [root, root / sequence, root / "Real_Data" / sequence, root / "DPJAIT" / "Real_Data" / sequence]
    for candidate in candidates:
        if candidate.name == sequence and (candidate / "cameras_calibration.csv").is_file():
            return candidate
    raise ValueError(f"DPJAIT sequence {sequence!r} was not found below {root}")


def read_dpjait_calibration(path: str | Path, *, use_distortion: bool = False) -> tuple[CalibrationDocument, list[str]]:
    """Convert DPJAIT OpenCV Rodrigues/tvec calibration into the project schema.

    Despite their names, ``cam_x/y/z`` are the world-to-camera translation
    vector stored by OpenCV, not the camera center in world coordinates.
    """
    cameras: dict[int, CameraCalibrationData] = {}
    source_names: list[str] = []
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle, delimiter=";"))
    if len(rows) < 2:
        raise ValueError("DPJAIT calibration must contain at least two cameras")

    for camera_id, row in enumerate(rows):
        source_names.append(row["cam_name"])
        rotation_vector = np.asarray([float(row[f"cam_or_{axis}[rad]"]) for axis in "xyz"])
        rotation, _ = cv2.Rodrigues(rotation_vector)
        focal = float(row["focal_length"])
        coefficients = [float(row[f"coeff_{index}"]) for index in range(1, 5)] if use_distortion else []
        cameras[camera_id] = CameraCalibrationData(
            image_size=VIDEO_SIZE,
            camera_matrix=[
                [focal, 0.0, float(row["cx"])],
                [0.0, focal, float(row["cy"])],
                [0.0, 0.0, 1.0],
            ],
            distortion_coefficients=coefficients,
            rotation_world_to_camera=rotation.tolist(),
            translation_world_to_camera=[float(row[f"cam_{axis}[mm]"]) / 1000.0 for axis in "xyz"],
        )
    return CalibrationDocument(cameras=cameras), source_names


def read_detection_rows(path: str | Path) -> list[tuple[int, float, float, float]]:
    """Read DPJAIT's headerless YOLO rows as frame, center x/y and confidence."""
    detections: list[tuple[int, float, float, float]] = []
    with Path(path).open(newline="", encoding="utf-8") as handle:
        for row in csv.reader(handle):
            if len(row) < 8:
                raise ValueError(f"invalid DPJAIT detection row in {path}: {row}")
            detections.append((int(row[0]), float(row[5]), float(row[6]), float(row[7])))
    return detections


def read_marker_centroid_ground_truth(
    path: str | Path,
    *,
    frame_count: int,
    video_fps: float = VIDEO_FPS,
    offset_samples: int = -3,
    visual_center_below_markers_m: float = VISUAL_CENTER_BELOW_MARKERS_M,
) -> tuple[list[np.ndarray], float, list[str]]:
    """Sample the drone visual center at video-frame timestamps.

    The archive provides four marker coordinates rather than a precomputed
    drone-center signal. DPJAIT's marker-cross diagram defines the visual
    center 100 mm below the marker cross, along the lab's vertical axis.
    """
    data = ezc3d.c3d(str(path))
    points = np.asarray(data["data"]["points"][:3], dtype=np.float64).transpose(2, 1, 0)
    residuals = np.asarray(data["data"]["meta_points"]["residuals"], dtype=np.float64)
    residuals = residuals.reshape(residuals.shape[-2], residuals.shape[-1]).T
    rate = float(data["parameters"]["POINT"]["RATE"]["value"][0])
    labels = [str(label) for label in data["parameters"]["POINT"]["LABELS"]["value"]]
    samples_per_frame = rate / video_fps
    if not np.isclose(samples_per_frame, round(samples_per_frame), atol=1e-6):
        raise ValueError(f"Vicon rate {rate} Hz is not an integer multiple of video rate {video_fps} Hz")

    ground_truth: list[np.ndarray] = []
    for frame_index in range(frame_count):
        sample_index = int(round(frame_index * samples_per_frame)) + offset_samples
        sample_index = min(max(sample_index, 0), len(points) - 1)
        valid = np.isfinite(points[sample_index]).all(axis=1) & (residuals[sample_index] >= 0)
        if not np.any(valid):
            ground_truth.append(np.full(3, np.nan))
        else:
            visual_center = points[sample_index, valid].mean(axis=0) / 1000.0
            visual_center[2] -= visual_center_below_markers_m
            ground_truth.append(visual_center)
    return ground_truth, rate, labels


def _transcode_video(source: Path, destination: Path) -> None:
    if destination.is_file() and destination.stat().st_mtime >= source.stat().st_mtime:
        return
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required to create browser-compatible replay videos")
    subprocess.run(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(source),
            "-map",
            "0:v:0",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "fast",
            "-crf",
            "21",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(destination),
        ],
        check=True,
    )


def import_dpjait_sequence(
    source: str | Path,
    output: str | Path,
    *,
    sequence: str = "R02_D1",
    vicon_offset_samples: int = -3,
    use_distortion: bool = False,
    transcode: bool = True,
    video_converter: Callable[[Path, Path], None] | None = None,
) -> DPJAITImportResult:
    """Create one self-contained Peleng replay bundle from a real sequence."""
    sequence_root = resolve_real_sequence(source, sequence)
    destination = Path(output).resolve()
    destination.mkdir(parents=True, exist_ok=True)

    calibration, camera_names = read_dpjait_calibration(
        sequence_root / "cameras_calibration.csv", use_distortion=use_distortion
    )
    CalibrationStore(calibration).save(destination / "calibration.yaml")

    all_detections: dict[int, list[tuple[int, float, float, float]]] = {}
    for camera_id, camera_name in enumerate(camera_names):
        detection_path = sequence_root / "dl_data" / f"{camera_name}.csv"
        video_path = sequence_root / f"{camera_name}.avi"
        if not detection_path.is_file() or not video_path.is_file():
            raise ValueError(f"sequence is missing data for camera {camera_name}")
        all_detections[camera_id] = read_detection_rows(detection_path)

    frame_counts = {len(rows) for rows in all_detections.values()}
    if len(frame_counts) != 1:
        raise ValueError(f"camera detection files have different frame counts: {sorted(frame_counts)}")
    frame_count = frame_counts.pop()

    with (destination / "observations.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["frame_index", "timestamp", "object_id", "camera_id", "u", "v", "confidence"],
        )
        writer.writeheader()
        for camera_id, detections in all_detections.items():
            for frame_index, u, v, confidence in detections:
                writer.writerow(
                    {
                        "frame_index": frame_index,
                        "timestamp": frame_index / VIDEO_FPS,
                        "object_id": "drone-1",
                        "camera_id": camera_id,
                        "u": u,
                        "v": v,
                        "confidence": confidence,
                    }
                )

    ground_truth, vicon_rate, marker_labels = read_marker_centroid_ground_truth(
        sequence_root / f"{sequence}.c3d",
        frame_count=frame_count,
        offset_samples=vicon_offset_samples,
    )
    with (destination / "ground-truth.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["frame_index", "x", "y", "z"])
        writer.writeheader()
        for frame_index, point in enumerate(ground_truth):
            if np.isfinite(point).all():
                writer.writerow({"frame_index": frame_index, "x": point[0], "y": point[1], "z": point[2]})

    converter = video_converter or _transcode_video
    cameras: list[dict[str, Any]] = []
    for camera_id, camera_name in enumerate(camera_names):
        output_name = f"camera-{camera_id}.mp4"
        if transcode:
            converter(sequence_root / f"{camera_name}.avi", destination / output_name)
        else:
            output_name = f"camera-{camera_id}.avi"
            shutil.copy2(sequence_root / f"{camera_name}.avi", destination / output_name)
        cameras.append({"id": camera_id, "name": f"cam_{camera_id + 1} · {camera_name}", "video": output_name})

    manifest = {
        "schema_version": 1,
        "id": sequence.lower().replace("_", "-"),
        "title": f"DPJAIT {sequence} · one drone · four cameras",
        "source": "DPJAIT real measurements",
        "fps": VIDEO_FPS,
        "duration_s": frame_count / VIDEO_FPS,
        "object_id": "drone-1",
        "observation_source": "DPJAIT YOLOv5 bounding-box centers",
        "cameras": cameras,
        "calibration": "calibration.yaml",
        "observations": "observations.csv",
        "ground_truth": "ground-truth.csv",
        "metadata": {
            "sequence": sequence,
            "source_camera_names": camera_names,
            "source_distortion_coefficients_applied": use_distortion,
            "vicon_rate_hz": vicon_rate,
            "vicon_offset_samples": vicon_offset_samples,
            "vicon_marker_labels": marker_labels,
            "ground_truth_definition": "visual center: valid Vicon-marker centroid shifted 100 mm vertically down",
            "visual_center_below_markers_m": VISUAL_CENTER_BELOW_MARKERS_M,
        },
    }
    (destination / "scenario.yaml").write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
    return DPJAITImportResult(
        bundle_path=destination,
        camera_count=len(cameras),
        frame_count=frame_count,
        observation_count=sum(len(rows) for rows in all_detections.values()),
        vicon_rate_hz=vicon_rate,
        vicon_offset_samples=vicon_offset_samples,
    )
