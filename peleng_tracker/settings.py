"""Typed application configuration loaded from YAML or JSON."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, field_validator


class DetectorSettings(BaseModel):
    backend: Literal["ultralytics"] = "ultralytics"
    weights: str = "yolo26n.pt"
    # COCO class 0 is person. Default to people-only instead of every COCO class.
    confidence: float = Field(default=0.4, ge=0.0, le=1.0)
    iou: float = Field(default=0.6, ge=0.0, le=1.0)
    device: str = "auto"
    image_size: int = Field(default=960, ge=32)
    class_ids: list[int] | None = Field(default_factory=lambda: [0])


class TrackerSettings(BaseModel):
    backend: Literal["deepsort", "botsort"] = "botsort"
    max_age: int = Field(default=30, ge=1)
    min_hits: int = Field(default=1, ge=1)
    matching_threshold: float = Field(default=0.3, ge=0.0, le=1.0)
    with_reid: bool = False


class GeometrySettings(BaseModel):
    mode: Literal["auto", "3d", "planar"] = "auto"
    calibration_file: str = "config/calibration.yaml"
    max_sync_skew_ms: float = Field(default=50.0, ge=0.0)
    max_reprojection_error_px: float = Field(default=5.0, ge=0.0)
    max_ground_distance_m: float = Field(default=1.5, gt=0.0)


class VisionSettings(BaseModel):
    detector: DetectorSettings = Field(default_factory=DetectorSettings)
    tracker: TrackerSettings = Field(default_factory=TrackerSettings)
    geometry: GeometrySettings = Field(default_factory=GeometrySettings)
    fps: float = Field(default=10.0, ge=1.0, le=60.0)


class CameraSettings(BaseModel):
    id: int
    name: str
    url: str
    enabled: bool = True

    @field_validator("url")
    @classmethod
    def reject_example_credentials(cls, value: str) -> str:
        if "CHANGE_ME" in value or "${" in value:
            raise ValueError("camera URL environment variable is not set")
        return value


class ServerSettings(BaseModel):
    host: str = "127.0.0.1"
    port: int = Field(default=8000, ge=1, le=65535)
    open_browser: bool = False
    tls_certfile: str | None = None
    tls_keyfile: str | None = None


class AppSettings(BaseModel):
    vision: VisionSettings = Field(default_factory=VisionSettings)
    cameras: list[CameraSettings] = Field(default_factory=list)
    server: ServerSettings = Field(default_factory=ServerSettings)

    @classmethod
    def load(cls, path: str | Path | None = None) -> AppSettings:
        config_path = Path(path or os.getenv("PELENG_TRACKER_CONFIG", "config/hackathon.yaml"))
        if not config_path.exists():
            return cls()
        with config_path.open(encoding="utf-8") as handle:
            contents = os.path.expandvars(handle.read())
            data = json.loads(contents) if config_path.suffix.lower() == ".json" else yaml.safe_load(contents) or {}
        return cls.model_validate(data)
