"""Runtime server configuration shared by streaming adapters."""

from __future__ import annotations

import math
import os
from typing import Any


class ServerConfig:
    SERVER_IP = os.getenv("SERVER_IP", "localhost")
    SERVER_NAME = os.getenv("PELENG_TRACKER_HOST", "127.0.0.1")
    SERVER_PORT = int(os.getenv("PELENG_TRACKER_PORT", "8000"))
    SERVER_RELOAD = False
    CORS_ORIGINS = ["http://localhost:5173", "http://127.0.0.1:5173"]
    STUN_SERVERS = ["stun:stun.l.google.com:19302"]
    WEBRTC_TIMEOUTS = {
        "answer_timeout": 20.0,
        "ice_timeout": 30.0,
        "connection_timeout": 45.0,
        "close_timeout": 5.0,
    }
    VISION_API_ENABLED = True
    CAMERA_RESOLUTION = {
        "individual_width": int(os.getenv("CAMERA_WIDTH", "720")),
        "individual_height": int(os.getenv("CAMERA_HEIGHT", "480")),
        "fps": int(os.getenv("CAMERA_FPS", "15")),
    }
    STREAM_SOURCES: list[dict[str, Any]] = []
    DEFAULT_CAMERAS: list[dict[str, Any]] = []

    @classmethod
    def configure_streams(cls, streams: list[dict[str, Any]]) -> None:
        cls.STREAM_SOURCES = streams
        cls.DEFAULT_CAMERAS = cls.get_default_cameras()

    @classmethod
    def add_or_update_stream(cls, stream: dict[str, Any]) -> None:
        cls.STREAM_SOURCES = [item for item in cls.STREAM_SOURCES if item["id"] != stream["id"]]
        cls.STREAM_SOURCES.append(stream)
        cls.STREAM_SOURCES.sort(key=lambda item: item["id"])
        cls.DEFAULT_CAMERAS = cls.get_default_cameras()

    @classmethod
    def get_enabled_streams(cls) -> list[dict[str, Any]]:
        return [stream for stream in cls.STREAM_SOURCES if stream.get("enabled", True)]

    @classmethod
    def get_active_stream_count(cls) -> int:
        return len(cls.get_enabled_streams())

    @classmethod
    def get_stream_by_id(cls, stream_id: int) -> dict[str, Any]:
        for stream in cls.STREAM_SOURCES:
            if stream["id"] == stream_id:
                return stream
        raise ValueError(f"camera {stream_id} is not configured")

    @classmethod
    def get_default_cameras(cls) -> list[dict[str, Any]]:
        width, height = cls.get_individual_resolution()
        return [
            {
                "id": stream["id"],
                "name": stream["name"],
                "stream_url": stream["url"],
                "enabled": stream.get("enabled", True),
                "resolution": {"width": width, "height": height, "fps": cls.CAMERA_RESOLUTION["fps"]},
            }
            for stream in cls.get_enabled_streams()
        ]

    @classmethod
    def get_camera_config(cls, camera_id: int) -> dict[str, Any]:
        return cls.get_stream_by_id(camera_id)

    @classmethod
    def get_camera_resolution(cls) -> dict[str, Any]:
        width, height = cls.get_individual_resolution()
        combined_width, combined_height = cls.get_combined_resolution()
        return {
            **cls.CAMERA_RESOLUTION,
            "combined_width": combined_width,
            "combined_height": combined_height,
            "individual_width": width,
            "individual_height": height,
        }

    @classmethod
    def get_layout(cls, count: int | None = None) -> tuple[int, int]:
        count = count if count is not None else max(1, cls.get_active_stream_count())
        columns = max(1, math.ceil(math.sqrt(count)))
        rows = max(1, math.ceil(count / columns))
        return columns, rows

    @classmethod
    def get_combined_resolution(cls) -> tuple[int, int]:
        width, height = cls.get_individual_resolution()
        columns, rows = cls.get_layout()
        return width * columns, height * rows

    @classmethod
    def get_individual_resolution(cls) -> tuple[int, int]:
        return cls.CAMERA_RESOLUTION["individual_width"], cls.CAMERA_RESOLUTION["individual_height"]

    @classmethod
    def get_stream_url(cls, stream_name: str) -> str:
        return cls.get_rtsp_url(stream_name)

    @classmethod
    def get_rtsp_url(cls, stream_name: str) -> str:
        return f"rtsp://{cls.SERVER_IP}:8554/{stream_name}"
