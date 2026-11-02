from fastapi.testclient import TestClient

from peleng_tracker.core.api import vision_control
from peleng_tracker.core.app import create_app
from peleng_tracker.targets import TargetManager


class FakeVisionAPI:
    def __init__(self) -> None:
        self.targets = TargetManager()

    def get_statistics(self):
        return {"enabled": True}


def test_target_api_class_track_and_clear() -> None:
    vision_control.set_vision_api(FakeVisionAPI())
    client = TestClient(create_app())
    assert client.put("/api/vision/target", json={"mode": "class", "class_ids": [0, 2]}).status_code == 200
    response = client.put("/api/vision/target", json={"mode": "track", "camera_id": 1, "track_id": 7})
    assert response.json()["target"]["track_id"] == "7"
    assert client.delete("/api/vision/target").json()["state"] == "idle"


def test_health_endpoint() -> None:
    client = TestClient(create_app())
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["service"] == "peleng-tracker"
