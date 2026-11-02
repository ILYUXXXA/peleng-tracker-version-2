"""Camera inventory and stream controls."""

from __future__ import annotations

import asyncio

import cv2
import numpy as np
from fastapi.responses import StreamingResponse
from typing import Annotated
from urllib.parse import urlsplit, urlunsplit

from fastapi import APIRouter, BackgroundTasks, File, HTTPException, Query, UploadFile
from pydantic import BaseModel, Field, field_validator

from ...phone_frames import phone_frames
from ..config import ServerConfig
from ..stream_combiner import stream_combiner_manager

router = APIRouter()
vision_api = None


def set_vision_api(api) -> None:
    global vision_api
    vision_api = api


class StreamDelayRequest(BaseModel):
    delay_ms: int = Field(ge=0, le=5000)


class StreamDelaysRequest(BaseModel):
    delays: dict[str, int]


class CameraSourceRequest(BaseModel):
    id: int = Field(ge=0)
    name: str = Field(min_length=1, max_length=80)
    url: str = Field(min_length=1, max_length=2048)
    enabled: bool = True

    @field_validator("url")
    @classmethod
    def validate_url(cls, value: str) -> str:
        try:
            parsed = urlsplit(value.strip())
            hostname, _ = parsed.hostname, parsed.port
        except ValueError as error:
            raise ValueError("camera URL is malformed") from error
        if parsed.scheme.lower() not in {"rtsp", "rtsps", "rtmp", "http", "https"}:
            raise ValueError("supported schemes: rtsp, rtsps, rtmp, http, https")
        if not hostname:
            raise ValueError("camera URL must contain a host")
        return value.strip()


def _redact_stream(stream: dict[str, object]) -> dict[str, object]:
    result = dict(stream)
    raw_url = str(result.get("url", result.get("stream_url", "")))
    if raw_url.startswith("phone://"):
        safe_url = raw_url
    else:
        parsed = urlsplit(raw_url)
        host = parsed.hostname or ""
        if ":" in host and not host.startswith("["):
            host = f"[{host}]"
        if parsed.port is not None:
            host = f"{host}:{parsed.port}"
        if parsed.username is not None:
            host = f"***@{host}"
        safe_url = urlunsplit((parsed.scheme, host, parsed.path, "***" if parsed.query else "", ""))
    result["url"] = safe_url
    if "stream_url" in result:
        result["stream_url"] = safe_url
    return result


@router.get("")
@router.get("/")
async def get_cameras() -> list[dict[str, object]]:
    statuses = stream_combiner_manager.get_stats().get("stream_status", {})
    return [
        {
            "id": stream["id"],
            "name": stream["name"],
            "stream_url": _redact_stream(stream)["url"],
            "enabled": stream.get("enabled", True),
            "status": (
                "ready"
                if stream["url"].startswith("phone://") and phone_frames.status(stream["id"])["connected"]
                else statuses.get(stream["id"], "configured")
            ),
        }
        for stream in ServerConfig.get_enabled_streams()
    ]


@router.get("/config")
async def get_camera_config() -> dict[str, object]:
    width, height = ServerConfig.get_individual_resolution()
    combined_width, combined_height = ServerConfig.get_combined_resolution()
    return {
        "cameras": [_redact_stream(stream) for stream in ServerConfig.get_enabled_streams()],
        "individual_resolution": {"width": width, "height": height},
        "combined_resolution": {"width": combined_width, "height": combined_height},
    }


@router.post("", status_code=201)
@router.post("/", status_code=201)
async def add_camera(
    request: CameraSourceRequest, background_tasks: BackgroundTasks
) -> dict[str, object]:
    stream = request.model_dump()
    stream_combiner_manager.register_camera(stream)
    if request.enabled:
        background_tasks.add_task(stream_combiner_manager.open_camera_stream, request.id)
    return {
        "camera": _redact_stream(stream),
        "persistent": False,
        "message": "camera registered for this server session",
    }


@router.get("/stream-delays")
async def get_stream_delays() -> dict[str, object]:
    return {"delays": stream_combiner_manager.get_stream_delays(), "unit": "milliseconds"}


@router.put("/stream-delays/{stream_id}")
async def set_stream_delay(stream_id: int, request: StreamDelayRequest) -> dict[str, object]:
    valid_ids = {stream["id"] for stream in ServerConfig.get_enabled_streams()}
    if stream_id not in valid_ids:
        raise HTTPException(status_code=404, detail="camera is not configured")
    await stream_combiner_manager.set_stream_delay(stream_id, request.delay_ms)
    return {"stream_id": stream_id, "delay_ms": request.delay_ms}


@router.put("/stream-delays")
async def set_all_stream_delays(request: StreamDelaysRequest) -> dict[str, object]:
    valid_ids = {stream["id"] for stream in ServerConfig.get_enabled_streams()}
    parsed = {int(key): value for key, value in request.delays.items()}
    if not set(parsed).issubset(valid_ids) or any(value < 0 or value > 5000 for value in parsed.values()):
        raise HTTPException(status_code=422, detail="invalid camera id or delay")
    await stream_combiner_manager.set_all_delays(request.delays)
    return {"delays": request.delays}


@router.post("/tracking/start")
async def start_tracking() -> dict[str, bool]:
    if vision_api is None:
        raise HTTPException(status_code=503, detail="vision pipeline is not initialized")
    stream_combiner_manager.enable_vision_tracking()
    return {"enabled": True}


@router.post("/tracking/stop")
async def stop_tracking() -> dict[str, bool]:
    stream_combiner_manager.disable_vision_tracking()
    return {"enabled": False}


@router.post("/phone/{camera_id}/frame")
async def publish_phone_frame(
    camera_id: int,
    frame: Annotated[UploadFile, File()],
    name: Annotated[str, Query(max_length=80)] = "Phone camera",
) -> dict[str, object]:
    payload = await frame.read(8 * 1024 * 1024 + 1)
    if len(payload) > 8 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="frame is larger than 8 MB")
    try:
        packet = phone_frames.publish(camera_id, payload, name)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    stream_combiner_manager.register_phone_camera(camera_id, name)
    return {
        "camera_id": camera_id,
        "sequence": packet.sequence,
        "captured_at": packet.captured_at,
        "shape": list(packet.frame.shape),
    }


@router.get("/phone/{camera_id}/status")
async def phone_camera_status(camera_id: int) -> dict[str, object]:
    return phone_frames.status(camera_id)


@router.get("/phone/mjpeg")
async def phone_cameras_mjpeg() -> StreamingResponse:
    """Serve phone camera frames over HTTP; unlike WebRTC this works through HTTPS proxies."""
    async def generate():
        last_sequences: dict[int, int] = {}
        while True:
            streams = [
                item for item in ServerConfig.get_enabled_streams()
                if str(item.get("url", "")).startswith("phone://")
            ]
            if streams:
                individual_width, individual_height = ServerConfig.get_individual_resolution()
                columns, rows = ServerConfig.get_layout(len(streams))
                canvas = np.zeros(
                    (rows * individual_height, columns * individual_width, 3),
                    dtype=np.uint8,
                )
                changed = False
                for index, stream in enumerate(streams):
                    camera_id = int(stream["id"])
                    packet = phone_frames.get(camera_id)
                    if packet is None:
                        continue
                    changed = changed or last_sequences.get(camera_id) != packet.sequence
                    frame = cv2.resize(packet.frame, (individual_width, individual_height))
                    x = (index % columns) * individual_width
                    y = (index // columns) * individual_height
                    canvas[y:y + individual_height, x:x + individual_width] = frame
                    last_sequences[camera_id] = packet.sequence
                if changed:
                    ok, encoded = cv2.imencode(".jpg", canvas, [cv2.IMWRITE_JPEG_QUALITY, 80])
                    if ok:
                        yield (
                            b"--frame\\r\\nContent-Type: image/jpeg\\r\\n"
                            b"Cache-Control: no-cache\\r\\n\\r\\n"
                            + encoded.tobytes()
                            + b"\\r\\n"
                        )
            await asyncio.sleep(0.05)

    return StreamingResponse(
        generate(),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-store, no-cache, must-revalidate", "Pragma": "no-cache"},
    )


@router.get("/{camera_id}")
async def get_camera(camera_id: int) -> dict[str, object]:
    try:
        return _redact_stream(ServerConfig.get_stream_by_id(camera_id))
    except ValueError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
