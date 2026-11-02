"""Compact DeepSORT-style tracker with isolated state per camera."""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from ..domain import Detection, Track
from ..settings import TrackerSettings


@dataclass(slots=True)
class _TrackState:
    identifier: int
    bbox: tuple[int, int, int, int]
    class_id: int
    class_name: str
    confidence: float
    appearance: np.ndarray
    hits: int = 1
    missed: int = 0
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float32))


@dataclass(slots=True)
class _CameraState:
    next_identifier: int = 1
    tracks: list[_TrackState] = field(default_factory=list)


class DeepSortBackend:
    """Motion/IoU association augmented with a compact appearance descriptor."""

    name = "deepsort"

    def __init__(self, settings: TrackerSettings) -> None:
        self.settings = settings
        self._cameras: dict[int, _CameraState] = {}

    def update(
        self,
        detections: list[Detection],
        frame: np.ndarray,
        camera_id: int,
        timestamp: float,
    ) -> list[Track]:
        del timestamp
        camera = self._cameras.setdefault(camera_id, _CameraState())
        descriptors = [self._appearance(frame, detection.bbox) for detection in detections]
        candidates: list[tuple[float, int, int]] = []
        for state_index, state in enumerate(camera.tracks):
            predicted = self._shift(state.bbox, state.velocity)
            for detection_index, detection in enumerate(detections):
                if state.class_id != detection.class_id:
                    continue
                iou = self._iou(predicted, detection.bbox)
                appearance = float(np.dot(state.appearance, descriptors[detection_index]))
                cost = 1.0 - (0.65 * iou + 0.35 * max(0.0, appearance))
                candidates.append((cost, state_index, detection_index))

        matched_states: set[int] = set()
        matched_detections: set[int] = set()
        for cost, state_index, detection_index in sorted(candidates):
            if state_index in matched_states or detection_index in matched_detections:
                continue
            if cost > 1.0 - self.settings.matching_threshold:
                continue
            state = camera.tracks[state_index]
            detection = detections[detection_index]
            previous_center = self._center(state.bbox)
            current_center = self._center(detection.bbox)
            state.velocity = np.subtract(current_center, previous_center, dtype=np.float32)
            state.bbox = detection.bbox
            state.confidence = detection.confidence
            state.class_name = detection.class_name
            state.appearance = 0.8 * state.appearance + 0.2 * descriptors[detection_index]
            state.appearance /= max(float(np.linalg.norm(state.appearance)), 1e-12)
            state.hits += 1
            state.missed = 0
            matched_states.add(state_index)
            matched_detections.add(detection_index)

        for index, state in enumerate(camera.tracks):
            if index not in matched_states:
                state.missed += 1
                state.bbox = self._shift(state.bbox, state.velocity)
        camera.tracks = [state for state in camera.tracks if state.missed <= self.settings.max_age]

        for index, detection in enumerate(detections):
            if index in matched_detections:
                continue
            camera.tracks.append(
                _TrackState(
                    identifier=camera.next_identifier,
                    bbox=detection.bbox,
                    class_id=detection.class_id,
                    class_name=detection.class_name,
                    confidence=detection.confidence,
                    appearance=descriptors[index],
                )
            )
            camera.next_identifier += 1

        return [
            Track(
                local_track_id=str(state.identifier),
                camera_id=camera_id,
                bbox=state.bbox,
                confidence=state.confidence,
                class_id=state.class_id,
                class_name=state.class_name,
                age=state.hits,
            )
            for state in camera.tracks
            if state.missed == 0 and state.hits >= self.settings.min_hits
        ]

    @staticmethod
    def _appearance(frame: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
        x, y, width, height = bbox
        crop = frame[max(0, y) : max(0, y + height), max(0, x) : max(0, x + width)]
        if crop.size == 0:
            return np.zeros(48, dtype=np.float32)
        histogram = cv2.calcHist([crop], [0, 1, 2], None, [16, 1, 1], [0, 256, 0, 256, 0, 256]).flatten()
        descriptor = np.concatenate((histogram, crop.mean(axis=(0, 1)), crop.std(axis=(0, 1)))).astype(np.float32)
        descriptor = np.pad(descriptor, (0, 48 - descriptor.size))
        return descriptor / max(float(np.linalg.norm(descriptor)), 1e-12)

    @staticmethod
    def _center(bbox: tuple[int, int, int, int]) -> np.ndarray:
        return np.asarray([bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] / 2], dtype=np.float32)

    @staticmethod
    def _shift(bbox: tuple[int, int, int, int], velocity: np.ndarray) -> tuple[int, int, int, int]:
        return (int(round(bbox[0] + velocity[0])), int(round(bbox[1] + velocity[1])), bbox[2], bbox[3])

    @staticmethod
    def _iou(left: tuple[int, int, int, int], right: tuple[int, int, int, int]) -> float:
        left_x2, left_y2 = left[0] + left[2], left[1] + left[3]
        right_x2, right_y2 = right[0] + right[2], right[1] + right[3]
        intersection = max(0, min(left_x2, right_x2) - max(left[0], right[0])) * max(
            0, min(left_y2, right_y2) - max(left[1], right[1])
        )
        union = left[2] * left[3] + right[2] * right[3] - intersection
        return intersection / union if union else 0.0

    def reset(self) -> None:
        self._cameras.clear()
