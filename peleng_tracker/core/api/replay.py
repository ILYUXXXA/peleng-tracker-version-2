"""Read-only API for processed replay bundles."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse

from ...replay import ReplayBundle, replay_repository

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
