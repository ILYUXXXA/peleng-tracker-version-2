"""Vision metadata WebSocket using schema version 2."""

from __future__ import annotations

import asyncio
import contextlib
import json
import time
from dataclasses import asdict
from typing import Any

import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from .stream_combiner import stream_combiner_manager

router = APIRouter()


def _serializable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer | np.floating):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _serializable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_serializable(item) for item in value]
    return value


def result_payload(result) -> dict[str, Any]:
    streams = {}
    for camera_id in result.active_stream_ids:
        streams[str(camera_id)] = {
            "detections": [
                {**asdict(item), "id": index, "center": item.anchor}
                for index, item in enumerate(result.detections.get(camera_id, []))
            ],
            "tracks": [
                {**asdict(item), "id": item.local_track_id, "center": item.anchor}
                for item in result.tracks.get(camera_id, [])
            ],
        }
    world_tracks = [asdict(item) for item in result.world_tracks]
    return _serializable(
        {
            "type": "vision_metadata",
            "schema_version": 2,
            "frame_id": result.frame_id,
            "timestamp": result.timestamp,
            "processing_time_ms": result.processing_time_ms,
            "sync_skew_ms": result.sync_skew_ms,
            "num_streams": result.num_streams,
            "active_stream_ids": result.active_stream_ids,
            "association_diagnostics": asdict(result.association_diagnostics),
            "all_streams": streams,
            "world_tracks": world_tracks,
            "bev_tracks": world_tracks,
        }
    )


@router.websocket("/vision-metadata")
async def vision_metadata(websocket: WebSocket) -> None:
    await websocket.accept()
    last_frame_id = None
    last_tracking_enabled = None
    try:
        while True:
            tracking_enabled = stream_combiner_manager.is_vision_tracking_enabled()
            if tracking_enabled != last_tracking_enabled:
                last_tracking_enabled = tracking_enabled
                await websocket.send_text(
                    json.dumps(
                        {
                            "type": "vision_status",
                            "schema_version": 2,
                            "timestamp": time.time(),
                            "tracking_enabled": tracking_enabled,
                            "active_stream_ids": stream_combiner_manager.get_active_stream_ids(),
                            "message": "vision enabled" if tracking_enabled else "vision disabled",
                        },
                        separators=(",", ":"),
                    )
                )
            result = stream_combiner_manager.get_latest_vision_result()
            if result is not None and result.frame_id != last_frame_id:
                last_frame_id = result.frame_id
                await websocket.send_text(json.dumps(result_payload(result), separators=(",", ":")))
            await asyncio.sleep(0.02)
    except WebSocketDisconnect:
        return
    except asyncio.CancelledError:
        raise
    finally:
        with contextlib.suppress(Exception):
            await websocket.close()
