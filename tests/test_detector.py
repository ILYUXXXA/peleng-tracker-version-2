import numpy as np

from peleng_tracker.detectors.ultralytics import UltralyticsDetector
from peleng_tracker.settings import DetectorSettings


class Tensor:
    def __init__(self, value):
        self.value = np.asarray(value)

    def cpu(self):
        return self

    def numpy(self):
        return self.value


class Boxes:
    xyxy = Tensor([[10, 20, 50, 80]])
    conf = Tensor([0.9])
    cls = Tensor([2])


class Result:
    boxes = Boxes()
    names = {2: "car"}


class Model:
    def predict(self, **kwargs):
        assert kwargs["classes"] == [2]
        return [Result()]


def test_ultralytics_adapter_converts_xyxy_and_keeps_class() -> None:
    settings = DetectorSettings(weights="custom.pt", class_ids=[2])
    detections = UltralyticsDetector(settings, model=Model()).detect(np.zeros((100, 100, 3), dtype=np.uint8))
    assert detections[0].bbox == (10, 20, 40, 60)
    assert (detections[0].class_id, detections[0].class_name) == (2, "car")


def test_empty_result() -> None:
    model = Model()
    model.predict = lambda **_kwargs: []
    assert UltralyticsDetector(DetectorSettings(), model=model).detect(np.zeros((8, 8, 3), dtype=np.uint8)) == []
