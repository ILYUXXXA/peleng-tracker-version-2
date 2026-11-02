"""Target selection and lifecycle state."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .domain import WorldTrack

TargetState = Literal["idle", "searching", "tracking", "lost", "reacquiring"]


@dataclass(slots=True)
class TargetSpec:
    mode: Literal["class", "track"]
    class_ids: set[int] | None = None
    track_key: tuple[int, str] | None = None
    global_id: str | None = None


class TargetManager:
    def __init__(self) -> None:
        self.spec: TargetSpec | None = None
        self.state: TargetState = "idle"
        self._lost_frames = 0

    def select_classes(self, class_ids: list[int]) -> None:
        self.spec = TargetSpec(mode="class", class_ids=set(class_ids))
        self.state = "searching"
        self._lost_frames = 0

    def select_track(self, camera_id: int, track_id: str) -> None:
        self.spec = TargetSpec(mode="track", track_key=(camera_id, track_id))
        self.state = "searching"
        self._lost_frames = 0

    def clear(self) -> None:
        self.spec = None
        self.state = "idle"
        self._lost_frames = 0

    def apply(self, world_tracks: list[WorldTrack], key_to_global: dict[tuple[int, str], str]) -> None:
        if self.spec is None:
            return
        if self.spec.mode == "track" and self.spec.global_id is None and self.spec.track_key is not None:
            self.spec.global_id = key_to_global.get(self.spec.track_key)

        matched = False
        for track in world_tracks:
            is_match = (
                self.spec.mode == "class" and self.spec.class_ids is not None and track.class_id in self.spec.class_ids
            ) or (self.spec.mode == "track" and track.global_id == self.spec.global_id)
            track.is_target = is_match
            matched = matched or is_match

        if matched:
            self.state = "tracking" if self._lost_frames == 0 else "reacquiring"
            self._lost_frames = 0
        else:
            self._lost_frames += 1
            self.state = "lost" if self._lost_frames == 1 else "reacquiring"

    def snapshot(self) -> dict[str, object]:
        if self.spec is None:
            return {"state": "idle", "target": None}
        target: dict[str, object] = {"mode": self.spec.mode}
        if self.spec.class_ids is not None:
            target["class_ids"] = sorted(self.spec.class_ids)
        if self.spec.track_key is not None:
            target["camera_id"], target["track_id"] = self.spec.track_key
        if self.spec.global_id is not None:
            target["global_id"] = self.spec.global_id
        return {"state": self.state, "target": target}
