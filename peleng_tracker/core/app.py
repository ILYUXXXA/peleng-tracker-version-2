"""FastAPI application and static dashboard hosting."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from . import vision_websocket
from .api import calibration, cameras, replay, vision_control, webrtc
from .config import ServerConfig


def create_app() -> FastAPI:
    application = FastAPI(
        title="Peleng Tracker API",
        version="0.2.0",
        description="Multicamera object tracking and 3D localization",
    )
    application.add_middleware(
        CORSMiddleware,
        allow_origins=ServerConfig.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    application.include_router(cameras.router, prefix="/api/cameras", tags=["cameras"])
    application.include_router(calibration.router, prefix="/api/calibration", tags=["calibration"])
    application.include_router(vision_control.router, prefix="/api/vision", tags=["vision"])
    application.include_router(replay.router, prefix="/api/replay", tags=["replay"])
    application.include_router(webrtc.router, prefix="/ws", tags=["webrtc"])
    application.include_router(vision_websocket.router, prefix="/ws", tags=["vision"])

    @application.get("/api/health")
    async def health() -> dict[str, object]:
        api = vision_control.vision_api
        return {
            "status": "healthy",
            "service": "peleng-tracker",
            "version": "0.2.0",
            "vision": api.get_statistics() if api else {"enabled": False},
            "cameras": len(ServerConfig.get_enabled_streams()),
        }

    static_dir = Path(__file__).parent.parent / "static"
    if static_dir.exists() and (static_dir / "assets").exists():
        application.mount("/assets", StaticFiles(directory=static_dir / "assets"), name="assets")

        @application.get("/favicon.svg")
        async def favicon() -> FileResponse:
            return FileResponse(static_dir / "favicon.svg", media_type="image/svg+xml")

        @application.get("/{full_path:path}", response_model=None)
        async def frontend(full_path: str):
            if full_path.startswith(("api/", "ws/", "assets/")):
                return {"error": "not found"}
            return FileResponse(static_dir / "index.html")

    return application


app = create_app()
