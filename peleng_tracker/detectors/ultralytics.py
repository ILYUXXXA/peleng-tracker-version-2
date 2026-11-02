"""Ultralytics adapter supporting YOLO26, YOLO11 and custom checkpoints."""

from __future__ import annotations

from typing import Any

import numpy as np

from ..domain import Detection
from ..settings import DetectorSettings


class UltralyticsDetector:
    def __init__(self, settings: DetectorSettings, model: Any | None = None) -> None:
        self.settings = settings
        self.weights = settings.weights
        if model is None:
            from ultralytics import YOLO

            model = YOLO(settings.weights)
        self.model = model

    @property
    def device(self) -> str | None:
        return None if self.settings.device == "auto" else self.settings.device

    def warmup(self) -> None:
        sample = np.zeros((self.settings.image_size, self.settings.image_size, 3), dtype=np.uint8)
        self.detect(sample)

    def detect(self, frame: np.ndarray) -> list[Detection]:
        results = self.model.predict(
            source=frame,
            conf=self.settings.confidence,
            iou=self.settings.iou,
            imgsz=self.settings.image_size,
            device=self.device,
            classes=self.settings.class_ids,
            verbose=False,
        )
        if not results or results[0].boxes is None:
            return []

        result = results[0]
        boxes = result.boxes
        xyxy = boxes.xyxy.cpu().numpy()
        confidences = boxes.conf.cpu().numpy()
        classes = boxes.cls.cpu().numpy().astype(int)
        names = result.names

        detections: list[Detection] = []
        for coordinates, confidence, class_id in zip(xyxy, confidences, classes, strict=True):
            x1, y1, x2, y2 = coordinates
            detections.append(
                Detection(
                    bbox=(int(round(x1)), int(round(y1)), int(round(x2 - x1)), int(round(y2 - y1))),
                    confidence=float(confidence),
                    class_id=int(class_id),
                    class_name=str(names[int(class_id)]),
                )
            )
        return detections
