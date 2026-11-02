"""Ports implemented by model, tracker and appearance adapters."""

from __future__ import annotations

from typing import Protocol

import numpy as np

from .domain import Detection, Track


class ObjectDetector(Protocol):
    weights: str

    def detect(self, frame: np.ndarray) -> list[Detection]: ...

    def warmup(self) -> None: ...


class LocalTracker(Protocol):
    name: str

    def update(
        self,
        detections: list[Detection],
        frame: np.ndarray,
        camera_id: int,
        timestamp: float,
    ) -> list[Track]: ...

    def reset(self) -> None: ...


class AppearanceEncoder(Protocol):
    def encode(self, crops: list[np.ndarray]) -> np.ndarray: ...
