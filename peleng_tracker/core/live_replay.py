"""Record synchronized live camera sessions as D Replay bundles."""
from __future__ import annotations

import csv
import json
import re
import threading
import time
import uuid
from pathlib import Path
from typing import Any

import cv2
import yaml

from ..replay import ReplayBundle, replay_repository


class LiveReplayRecorder:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._root: Path | None = None
        self._writers: dict[int, Any] = {}
        self._camera_names: dict[int, str] = {}
        self._camera_sizes: dict[int, tuple[int, int]] = {}
        self._camera_ids: list[int] = []
        self._calibration: dict[str, Any] = {}
        self._rows: list[dict[str, Any]] = []
        self._frame_index = 0
        self._started_at = 0.0
        self._last_timestamp = 0.0
        self._fps_samples: list[float] = []
        self._active = False
        self._last_summary: dict[str, Any] | None = None

    @property
    def active(self) -> bool:
        return self._active

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "recording": self._active,
                "session_id": self._root.name if self._root else None,
                "frames": self._frame_index,
                "observations": len(self._rows),
                "last_session": self._last_summary,
            }

    def start(self, calibration: dict[str, Any], cameras: list[dict[str, Any]]) -> dict[str, Any]:
        with self._lock:
            if self._active:
                raise ValueError("A replay recording is already active")
            calibrated = {int(key) for key in calibration.get("cameras", {})}
            selected = [camera for camera in cameras if int(camera["id"]) in calibrated]
            if len(selected) < 2:
                raise ValueError("At least two configured cameras with valid stereo calibration are required")
            session_id = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
            base = Path(__import__("os")..getenv("PELENG_REPLAY_DIR", "replay_sessions")).resolve()
            root = base / session_id
            root.mkdir(parents=True, exist_ok=False)
            self._root = root
            self._camera_ids = [int(item["id"]) for item in selected]
            self._camera_names = {int(item["id"]): str(item["name"]) for item in selected}
            self._calibration = calibration
            self._writers = {}
            self._camera_sizes = {}
            self._rows = []
            self._frame_index = 0
            self._started_at = time.monotonic()
            self._last_timestamp = 0.0
            self._fps_samples = []
            self._active = True
            (root / "calibration.yaml").write_text(
                yaml.safe_dump(calibration, sort_keys=False), encoding="utf-8"
            )
            return self.status()

    def append(self, frames: dict[int, Any], result: Any, timestamps: dict[int, float]) -> None:
        with self._lock:
            if not self._active or self._root is None:
                return
            if not all(camera_id in frames and camera_id in timestamps for camera_id in self._camera_ids):
                return
            times = [float(timestamps[camera_id]) for camera_id in self._camera_ids]
            if max(times) - min(times) > 0.15:
                return
            timestamp = max(0.0, max(times) - min(self._started_at, min(times)))
            if self._frame_index and timestamp <= self._last_timestamp:
                return
            for camera_id in self._camera_ids:
                frame = frames[camera_id]
                height, width = frame.shape[:2]
                size = (width, height)
                if camera_id not in self._writers:
                    self._camera_sizes[camera_id] = size
                    writer = cv2.VideoWriter(
                        str(self._root / f"camera-{camera_id}.mp4"),
                        cv2.VideoWriter_fourcc(*"mp4v"), 30.0, size,
                    )
                    if not writer.isOpened():
                        writer.release()
                        raise RuntimeError(f"Could not create video for camera {camera_id}")
                    self._writers[camera_id] = writer
                if size != self._camera_sizes[camera_id]:
                    frame = cv2.resize(frame, self._camera_sizes[camera_id])
                self._writers[camera_id].write(frame)

            grouped: dict[str, list[tuple[int, float, float, float]]] = {}
            for camera_id in self._camera_ids:
                for track in result.tracks.get(camera_id, []):
                    if track.class_id != 0 or not track.global_id:
                        continue
                    u, v = track.geometry_anchor
                    grouped.setdefault(str(track.global_id), []).append(
                        (camera_id, float(u), float(v), float(track.confidence))
                    )
            for object_id, observations in grouped.items():
                distinct = {item[0] for item in observations}
                if len(distinct) < 2:
                    continue
                for camera_id, u, v, confidence in observations:
                    self._rows.append({
                        "frame_index": self._frame_index, "timestamp": timestamp,
                        "object_id": object_id, "camera_id": camera_id,
                        "u": u, "v": v, "confidence": confidence,
                    })
            self._last_timestamp = timestamp
            self._frame_index += 1
            self._fps_samples.append(timestamp)

    def stop(self) -> dict[str, Any]:
        with self._lock:
            if not self._active or self._root is None:
                raise ValueError("No replay recording is active")
            root = self._root
            for writer in self._writers.values():
                writer.release()
            self._writers.clear()
            if not self._rows:
                self._active = False
                raise ValueError("Recording contains no person observations matched across two calibrated cameras")
            object_counts: dict[str, int] = {}
            for row in self._rows:
                object_counts[row["object_id"]] = object_counts.get(row["object_id"], 0) + 1
            object_id = max(object_counts, key=object_counts.get)
            rows = [row for row in self._rows if row["object_id"] == object_id]
            with (root / "observations.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=[
                    "frame_index", "timestamp", "object_id", "camera_id", "u", "v", "confidence"
                ])
                writer.writeheader()
                writer.writerows(rows)
            fps = 30.0
            duration = max(self._last_timestamp, self._frame_index / fps)
            manifest = {
                "schema_version": 1, "id": root.name, "title": f"Live session {root.name}",
                "source": "real-cameras", "fps": fps, "duration_s": duration,
                "object_id": object_id, "observation_source": "live-person-tracks",
                "cameras": [
                    {"id": camera_id, "name": self._camera_names[camera_id], "video": f"camera-{camera_id}.mp4"}
                    for camera_id in self._camera_ids
                ],
                "calibration": "calibration.yaml", "observations": "observations.csv",
                "metadata": {"recorded_frames": self._frame_index, "matched_observations": len(rows)},
            }
            (root / "scenario.yaml").write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
            self._active = False
            self._root = None
            self._last_summary = {"session_id": root.name, "path": str(root), "frames": self._frame_index, "observations": len(rows)}
        bundle = ReplayBundle.load(root)
        bundle.validate()
        bundle.process()
        replay_repository.load(root)
        return {**self._last_summary, "processed": True, "states": len(bundle.trajectory.states) if bundle.trajectory else 0}


live_replay_recorder = LiveReplayRecorder()
