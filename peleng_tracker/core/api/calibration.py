"""3D calibration storage and validation API."""

from __future__ import annotations

import json
from typing import Any

import yaml
from fastapi import APIRouter, HTTPException, Request

from ...geometry import CalibrationDocument, CalibrationStore

router = APIRouter()
vision_api = None


def set_vision_api(api) -> None:
    global vision_api
    vision_api = api


@router.get("/3d")
async def get_calibration() -> dict[str, Any]:
    if vision_api is None:
        raise HTTPException(status_code=503, detail="vision pipeline is not initialized")
    return {
        "status": vision_api.pipeline.associator.calibration.status(),
        "calibration": vision_api.calibration_document().model_dump(mode="json"),
    }


@router.post("/3d")
async def set_calibration(request: Request) -> dict[str, Any]:
    if vision_api is None:
        raise HTTPException(status_code=503, detail="vision pipeline is not initialized")
    body = await request.body()
    try:
        raw = json.loads(body) if "json" in request.headers.get("content-type", "") else yaml.safe_load(body)
        document = CalibrationDocument.model_validate(raw)
    except Exception as error:
        raise HTTPException(status_code=422, detail=f"invalid calibration: {error}") from error
    store = CalibrationStore(document)
    if len(store.cameras) < 2:
        raise HTTPException(status_code=422, detail="at least two cameras are required for 3D")
    vision_api.replace_calibration(document)
    return {"success": True, "status": store.status()}
