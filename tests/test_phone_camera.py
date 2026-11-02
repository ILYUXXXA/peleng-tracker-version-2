import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from peleng_tracker.core.app import create_app
from peleng_tracker.core.config import ServerConfig
from peleng_tracker.core.stream_combiner import StreamCombinerTrack, stream_combiner_manager
from peleng_tracker.phone_frames import phone_frames


def jpeg() -> bytes:
    success, payload = cv2.imencode(".jpg", np.full((48, 64, 3), 127, dtype=np.uint8))
    assert success
    return payload.tobytes()


def test_phone_frame_registers_dynamic_camera_without_removing_rtsp() -> None:
    ServerConfig.configure_streams(
        [{"id": 7, "name": "pro-camera", "url": "rtsp://camera.local/live", "enabled": True}]
    )
    client = TestClient(create_app())
    response = client.post(
        "/api/cameras/phone/3/frame?name=Pixel",
        files={"frame": ("frame.jpg", jpeg(), "image/jpeg")},
    )
    assert response.status_code == 200
    assert response.json()["shape"] == [48, 64, 3]
    assert {camera["id"] for camera in client.get("/api/cameras").json()} == {3, 7}
    assert phone_frames.status(3)["connected"] is True


def test_phone_frame_rejects_invalid_image() -> None:
    client = TestClient(create_app())
    response = client.post(
        "/api/cameras/phone/4/frame",
        files={"frame": ("frame.jpg", b"not-an-image", "image/jpeg")},
    )
    assert response.status_code == 422


def test_runtime_rtsp_camera_can_be_added_without_removing_existing_source(monkeypatch) -> None:
    ServerConfig.configure_streams(
        [{"id": 7, "name": "pro-camera", "url": "rtsp://camera.local/live", "enabled": True}]
    )
    stream_combiner_manager.track = None
    opened: list[int] = []
    monkeypatch.setattr(stream_combiner_manager, "open_camera_stream", opened.append)
    client = TestClient(create_app())

    response = client.post(
        "/api/cameras",
        json={
            "id": 3,
            "name": "iPhone RTSP",
            "url": "rtsp://alice:secret@phone.local:8554/live?token=secret",
            "enabled": True,
        },
    )

    assert response.status_code == 201
    assert response.json()["persistent"] is False
    assert response.json()["camera"]["url"] == "rtsp://***@phone.local:8554/live?***"
    assert opened == [3]
    cameras = client.get("/api/cameras").json()
    assert {camera["id"] for camera in cameras} == {3, 7}
    assert "alice" not in str(cameras)
    assert "secret" not in str(cameras)


def test_runtime_camera_rejects_unsupported_url() -> None:
    client = TestClient(create_app())
    response = client.post(
        "/api/cameras",
        json={"id": 4, "name": "local file", "url": "file:///tmp/video.mp4", "enabled": True},
    )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_phone_source_is_rendered_by_stream_combiner() -> None:
    ServerConfig.configure_streams([{"id": 9, "name": "phone", "url": "phone://9", "enabled": True}])
    phone_frames.publish(9, jpeg(), "phone")
    track = StreamCombinerTrack()
    frame = await track.recv()
    assert (frame.width, frame.height) == ServerConfig.get_individual_resolution()
    rendered = frame.to_ndarray(format="bgr24")
    assert float(rendered.mean()) > 100.0
    track.cleanup()
