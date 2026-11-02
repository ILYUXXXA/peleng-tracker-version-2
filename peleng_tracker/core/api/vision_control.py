"""Vision configuration, target selection and restart endpoints."""

from __future__ import annotations

from typing import Any, Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ...settings import VisionSettings

router = APIRouter()
vision_api = None


def set_vision_api(api) -> None:
    global vision_api
    vision_api = api


def require_api():
    if vision_api is None:
        raise HTTPException(status_code=503, detail="vision pipeline is not initialized")
    return vision_api


class ConfigUpdate(BaseModel):
    params: dict[str, Any]


class RestartRequest(BaseModel):
    vision: dict[str, Any] | None = None


class TargetRequest(BaseModel):
    mode: Literal["class", "track"]
    class_ids: list[int] | None = None
    camera_id: int | None = None
    track_id: str | int | None = None


@router.get("/config/schema")
async def config_schema() -> dict[str, Any]:
    return require_api().get_config_schema()


@router.get("/config")
async def config() -> dict[str, Any]:
    return require_api().get_current_config()


@router.post("/config")
async def update_config(update: ConfigUpdate) -> dict[str, str]:
    require_api().update_config(update.params)
    return {"message": "configuration staged; restart to apply model or tracker changes"}


@router.get("/status")
async def status() -> dict[str, Any]:
    return require_api().get_statistics()


@router.post("/restart")
async def restart(request: RestartRequest) -> dict[str, Any]:
    api = require_api()
    try:
        settings = VisionSettings.model_validate(request.vision) if request.vision is not None else api.settings
        api.restart(settings)
        return {"success": True, "vision": api.get_statistics()}
    except Exception as error:
        raise HTTPException(status_code=422, detail=f"restart rejected: {error}") from error


@router.get("/target")
async def get_target() -> dict[str, object]:
    return require_api().targets.snapshot()


@router.put("/target")
async def set_target(request: TargetRequest) -> dict[str, object]:
    targets = require_api().targets
    if request.mode == "class":
        if not request.class_ids:
            raise HTTPException(status_code=422, detail="class_ids is required for class mode")
        targets.select_classes(request.class_ids)
    else:
        if request.camera_id is None or request.track_id is None:
            raise HTTPException(status_code=422, detail="camera_id and track_id are required for track mode")
        targets.select_track(request.camera_id, str(request.track_id))
    return targets.snapshot()


@router.delete("/target")
async def clear_target() -> dict[str, object]:
    require_api().targets.clear()
    return require_api().targets.snapshot()
