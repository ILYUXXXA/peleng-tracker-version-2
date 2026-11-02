"""Read-only API for processed replay bundles."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse

from ...replay import ReplayBundle, replay_repository
from ...core.live_replay import live_replay_recorder
from . import calibration as calibration_api
from ..config import ServerConfig

router = APIRouter()


def _bundle() -> ReplayBundle:
    try:
        return replay_repository.require()
    except ValueError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error


@router.get("")
async def replay_info() -> dict[str, object]:
    bundle = _bundle()
    return {
        "manifest": bundle.manifest.model_dump(mode="json"),
        "calibration": bundle.calibration.document.model_dump(mode="json"),
        "evaluation": bundle.evaluation(),
    }


@router.get("/trajectory")
async def replay_trajectory() -> JSONResponse:
    bundle = _bundle()
    assert bundle.trajectory is not None
    return JSONResponse(bundle.trajectory.model_dump(mode="json"))


@router.get("/cameras/{camera_id}/video")
async def replay_video(camera_id: int) -> FileResponse:
    bundle = _bundle()
    camera = next((item for item in bundle.manifest.cameras if item.id == camera_id), None)
    if camera is None:
        raise HTTPException(status_code=404, detail=f"camera {camera_id} not found")
    path = bundle.resolve_path(bundle.root, camera.video)
    media_type = "video/mp4" if path.suffix.lower() == ".mp4" else "application/octet-stream"
    return FileResponse(path, media_type=media_type, filename=Path(camera.video).name)


@router.get("/export.csv")
async def replay_export_csv() -> PlainTextResponse:
    bundle = _bundle()
    return PlainTextResponse(
        bundle.export_csv(),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{bundle.manifest.id}-trajectory.csv"'},
    )


@router.get("/export.json")
async def replay_export_json() -> JSONResponse:
    bundle = _bundle()
    assert bundle.trajectory is not None
    return JSONResponse(
        bundle.trajectory.model_dump(mode="json"),
        headers={"Content-Disposition": f'attachment; filename="{bundle.manifest.id}-trajectory.json"'},
    )


@router.get("/record/status")
async def replay_record_status() -> dict[str, object]:
    return live_replay_recorder.status()


@router.post("/record/start")
async def replay_record_start() -> dict[str, object]:
    api = calibration_api.vision_api
    if api is None:
        raise HTTPException(status_code=503, detail="vision pipeline is not initialized")
    if not api.is_tracking_enabled():
        raise HTTPException(status_code=409, detail="Enable vision tracking before recording")
    try:
        calibration = api.calibration_document().model_dump(mode="json")
        return live_replay_recorder.start(calibration, ServerConfig.get_enabled_streams())
    except (ValueError, RuntimeError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


@router.post("/record/stop")
async def replay_record_stop() -> dict[str, object]:
    try:
        return live_replay_recorder.stop()
    except ValueError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(status_code=422, detail=f"Replay processing failed: {error}") from error
