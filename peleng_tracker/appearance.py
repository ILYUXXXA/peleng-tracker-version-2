"""Small pluggable appearance encoder used by cross-camera strategies."""

from __future__ import annotations

import cv2
import numpy as np


class ColorHistogramEncoder:
    """Deterministic offline-safe baseline; replaceable by a learned ReID model."""

    name = "color-histogram"

    def encode(self, crops: list[np.ndarray]) -> np.ndarray:
        features = []
        for crop in crops:
            if crop.size == 0:
                features.append(np.zeros(4096, dtype=np.float32))
                continue
            histogram = cv2.calcHist([crop], [0, 1, 2], None, [16, 16, 16], [0, 256] * 3)
            features.append(cv2.normalize(histogram, histogram).flatten().astype(np.float32))
        return np.asarray(features, dtype=np.float32)
