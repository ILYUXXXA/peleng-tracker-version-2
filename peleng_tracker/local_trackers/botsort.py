"""BoT-SORT adapter isolated from Ultralytics internal tracker APIs."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np

from ..domain import Detection, Track
from ..settings import TrackerSettings


class _BoxesAdapter:
    def __init__(self, detections: list[Detection]) -> None:
        self.xywh = np.asarray(
            [[d.bbox[0] + d.bbox[2] / 2, d.bbox[1] + d.bbox[3] / 2, d.bbox[2], d.bbox[3]] for d in detections],
            dtype=np.float32,
        ).reshape(-1, 4)
        self.conf = np.asarray([d.confidence for d in detections], dtype=np.float32)
        self.cls = np.asarray([d.class_id for d in detections], dtype=np.float32)

    def __len__(self) -> int:
        return len(self.conf)

    def __getitem__(self, item: Any) -> _BoxesAdapter:
        selected = object.__new__(_BoxesAdapter)
        selected.xywh = self.xywh[item]
        selected.conf = self.conf[item]
        selected.cls = self.cls[item]
        if selected.xywh.ndim == 1:
            selected.xywh = selected.xywh[None, :]
            selected.conf = np.atleast_1d(selected.conf)
            selected.cls = np.atleast_1d(selected.cls)
        return selected


class BoTSortBackend:
    name = "botsort"

    def __init__(self, settings: TrackerSettings, tracker_factory: Any | None = None) -> None:
        self.settings = settings
        self._tracker_factory = tracker_factory or self._build_tracker
        self._trackers: dict[int, Any] = {}

    def _build_tracker(self) -> Any:
        from ultralytics.trackers.bot_sort import BOTSORT

        args = SimpleNamespace(
            tracker_type="botsort",
            track_high_thresh=0.25,
            track_low_thresh=0.1,
            new_track_thresh=0.25,
            track_buffer=self.settings.max_age,
            match_thresh=0.8,
            fuse_score=True,
            gmc_method="none",
            proximity_thresh=0.5,
            appearance_thresh=0.8,
            with_reid=self.settings.with_reid,
            model="auto",
            device=None,
        )
        return BOTSORT(args)

    def update(
        self,
        detections: list[Detection],
        frame: np.ndarray,
        camera_id: int,
        timestamp: float,
    ) -> list[Track]:
        del timestamp
        if camera_id not in self._trackers:
            self._trackers[camera_id] = self._tracker_factory()
        tracker = self._trackers[camera_id]
        rows = tracker.update(_BoxesAdapter(detections), frame)
        output: list[Track] = []
        for row in np.asarray(rows):
            x1, y1, x2, y2, track_id, score, class_id = row[:7]
            class_int = int(class_id)
            source = next((item for item in detections if item.class_id == class_int), None)
            output.append(
                Track(
                    local_track_id=str(int(track_id)),
                    camera_id=camera_id,
                    bbox=(int(x1), int(y1), int(x2 - x1), int(y2 - y1)),
                    confidence=float(score),
                    class_id=class_int,
                    class_name=source.class_name if source else str(class_int),
                )
            )
        return output

    def reset(self) -> None:
        self._trackers.clear()
