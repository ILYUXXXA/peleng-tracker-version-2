"""In-memory frames published by phone browsers or lightweight mobile clients."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(slots=True)
class PhoneFrame:
    frame: np.ndarray
    captured_at: float
    sequence: int
    name: str


class PhoneFrameStore:
    def __init__(self) -> None:
        self._frames: dict[int, PhoneFrame] = {}
        self._lock = threading.Lock()

    def publish(self, camera_id: int, payload: bytes, name: str) -> PhoneFrame:
        encoded = np.frombuffer(payload, dtype=np.uint8)
        frame = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("frame is not a supported JPEG/PNG image")
        with self._lock:
            previous = self._frames.get(camera_id)
            packet = PhoneFrame(frame, time.monotonic(), 1 if previous is None else previous.sequence + 1, name)
            self._frames[camera_id] = packet
            return packet

    def get(self, camera_id: int) -> PhoneFrame | None:
        with self._lock:
            return self._frames.get(camera_id)

    def status(self, camera_id: int) -> dict[str, object]:
        packet = self.get(camera_id)
        if packet is None:
            return {"connected": False, "sequence": 0, "age_ms": None}
        return {
            "connected": (time.monotonic() - packet.captured_at) < 2.0,
            "sequence": packet.sequence,
            "age_ms": round((time.monotonic() - packet.captured_at) * 1000, 1),
            "name": packet.name,
        }


phone_frames = PhoneFrameStore()
