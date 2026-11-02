"""3D calibration storage and checkerboard stereo-calibration API."""

from __future__ import annotations

import json
from typing import Any

import cv2
import numpy as np
import yaml
from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile

from ...geometry import CalibrationDocument, CalibrationStore, CameraCalibrationData

router = APIRouter()
vision_api = None
MAX_IMAGE_BYTES = 12 * 1024 * 1024


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


def _decode_image(upload: UploadFile) -> np.ndarray:
    payload = upload.file.read(MAX_IMAGE_BYTES + 1)
    if len(payload) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail=f"{upload.filename}: image exceeds 12 MB")
    image = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=422, detail=f"{upload.filename}: unsupported or corrupt image")
    return image


@router.post("/3d/checkerboard")
async def calibrate_stereo_checkerboard(
    camera0_images: list[UploadFile] = File(...),
    camera1_images: list[UploadFile] = File(...),
    camera0_id: int = Form(0, ge=0),
    camera1_id: int = Form(1, ge=0),
    columns: int = Form(9, ge=4, le=20),
    rows: int = Form(6, ge=4, le=20),
    square_size_m: float = Form(0.025, gt=0.001, le=0.5),
) -> dict[str, Any]:
    """Calibrate a synchronized stereo pair from corresponding checkerboard image pairs."""
    if vision_api is None:
        raise HTTPException(status_code=503, detail="vision pipeline is not initialized")
    if camera0_id == camera1_id:
        raise HTTPException(status_code=422, detail="camera IDs must be different")
    if len(camera0_images) != len(camera1_images) or len(camera0_images) < 12:
        raise HTTPException(status_code=422, detail="upload at least 12 corresponding image pairs; both lists must have equal length")

    pattern = (columns, rows)
    object_template = np.zeros((columns * rows, 3), np.float32)
    object_template[:, :2] = np.mgrid[0:columns, 0:rows].T.reshape(-1, 2)
    object_template *= square_size_m
    object_points: list[np.ndarray] = []
    image_points0: list[np.ndarray] = []
    image_points1: list[np.ndarray] = []
    image_size: tuple[int, int] | None = None
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.001)

    for index, (upload0, upload1) in enumerate(zip(camera0_images, camera1_images, strict=True)):
        image0, image1 = _decode_image(upload0), _decode_image(upload1)
        size0 = (image0.shape[1], image0.shape[0])
        size1 = (image1.shape[1], image1.shape[0])
        if size0 != size1:
            raise HTTPException(status_code=422, detail=f"pair {index + 1}: camera image sizes differ")
        if image_size is not None and size0 != image_size:
            raise HTTPException(status_code=422, detail="all calibration images must have the same resolution")
        image_size = size0
        gray0 = cv2.cvtColor(image0, cv2.COLOR_BGR2GRAY)
        gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
        found0, corners0 = cv2.findChessboardCorners(gray0, pattern, None)
        found1, corners1 = cv2.findChessboardCorners(gray1, pattern, None)
        if not found0 or not found1:
            continue
        corners0 = cv2.cornerSubPix(gray0, corners0, (11, 11), (-1, -1), criteria)
        corners1 = cv2.cornerSubPix(gray1, corners1, (11, 11), (-1, -1), criteria)
        object_points.append(object_template.copy())
        image_points0.append(corners0)
        image_points1.append(corners1)

    if len(object_points) < 10:
        raise HTTPException(status_code=422, detail=f"only {len(object_points)} usable pairs; need at least 10 with the full board visible in both images")
    assert image_size is not None
    flags = 0
    rms0, matrix0, distortion0, _, _ = cv2.calibrateCamera(object_points, image_points0, image_size, None, None, flags=flags)
    rms1, matrix1, distortion1, _, _ = cv2.calibrateCamera(object_points, image_points1, image_size, None, None, flags=flags)
    stereo_flags = cv2.CALIB_FIX_INTRINSIC
    stereo_rms, matrix0, distortion0, matrix1, distortion1, rotation, translation, essential, fundamental = cv2.stereoCalibrate(
        object_points, image_points0, image_points1, matrix0, distortion0, matrix1, distortion1,
        image_size, criteria=criteria, flags=stereo_flags,
    )
    if not np.isfinite(stereo_rms) or stereo_rms > 3.0:
        raise HTTPException(status_code=422, detail=f"stereo reprojection RMS is {stereo_rms:.2f}px (must be <= 3px); recapture sharper, more varied board poses")
    identity = np.eye(3, dtype=float)
    zero = np.zeros(3, dtype=float)
    data0 = CameraCalibrationData(
        image_size=image_size, camera_matrix=matrix0.tolist(), distortion_coefficients=distortion0.reshape(-1).tolist(),
        rotation_world_to_camera=identity.tolist(), translation_world_to_camera=zero.tolist(),
    )
    data1 = CameraCalibrationData(
        image_size=image_size, camera_matrix=matrix1.tolist(), distortion_coefficients=distortion1.reshape(-1).tolist(),
        rotation_world_to_camera=rotation.tolist(), translation_world_to_camera=translation.reshape(-1).tolist(),
    )
    document = CalibrationDocument(cameras={camera0_id: data0, camera1_id: data1})
    vision_api.replace_calibration(document)
    return {
        "success": True,
        "status": CalibrationStore(document).status(),
        "quality": {
            "usable_pairs": len(object_points), "stereo_rms_px": float(stereo_rms),
            "camera0_rms_px": float(rms0), "camera1_rms_px": float(rms1),
            "baseline_m": float(np.linalg.norm(translation)),
            "image_size": list(image_size),
        },
        "note": "Camera 0 defines the world origin. Keep both phones fixed after calibration.",
    }
