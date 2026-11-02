"""Small explicit factories for supported model and tracker adapters."""

from __future__ import annotations

from .detectors import UltralyticsDetector
from .local_trackers import BoTSortBackend, DeepSortBackend
from .ports import LocalTracker, ObjectDetector
from .settings import VisionSettings


def create_detector(settings: VisionSettings, *, model: object | None = None) -> ObjectDetector:
    if settings.detector.backend != "ultralytics":
        raise ValueError(f"unsupported detector backend: {settings.detector.backend}")
    return UltralyticsDetector(settings.detector, model=model)


def create_local_tracker(settings: VisionSettings) -> LocalTracker:
    if settings.tracker.backend == "deepsort":
        return DeepSortBackend(settings.tracker)
    if settings.tracker.backend == "botsort":
        return BoTSortBackend(settings.tracker)
    raise ValueError(f"unsupported tracker backend: {settings.tracker.backend}")
