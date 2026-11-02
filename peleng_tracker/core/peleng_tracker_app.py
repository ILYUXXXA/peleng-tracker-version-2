"""Peleng Tracker application lifecycle."""

from __future__ import annotations

import asyncio
import logging
import threading
import time
import webbrowser

import uvicorn

from ..settings import AppSettings
from .config import ServerConfig
from .stream_combiner import stream_combiner_manager
from .vision_api import create_vision_api

logger = logging.getLogger(__name__)


class PelengTrackerApp:
    def __init__(self, settings: AppSettings, *, warmup: bool = True) -> None:
        self.settings = settings
        self.server: uvicorn.Server | None = None
        self.server_thread: threading.Thread | None = None
        self.is_running = False

        ServerConfig.SERVER_NAME = settings.server.host
        ServerConfig.SERVER_PORT = settings.server.port
        ServerConfig.configure_streams([camera.model_dump() for camera in settings.cameras])
        self.vision_api = create_vision_api(settings.vision, warmup=warmup)
        stream_combiner_manager.set_vision_api(self.vision_api)

        from .api import calibration, cameras, vision_control

        calibration.set_vision_api(self.vision_api)
        cameras.set_vision_api(self.vision_api)
        vision_control.set_vision_api(self.vision_api)

        from .app import app

        self.server = uvicorn.Server(
            uvicorn.Config(
                app,
                host=settings.server.host,
                port=settings.server.port,
                log_level="info",
                ssl_certfile=settings.server.tls_certfile,
                ssl_keyfile=settings.server.tls_keyfile,
            )
        )

    @property
    def server_url(self) -> str:
        scheme = "https" if self.settings.server.tls_certfile else "http"
        return f"{scheme}://{self.settings.server.host}:{self.settings.server.port}"

    def start(self) -> None:
        if self.is_running or self.server is None:
            return
        self.server_thread = threading.Thread(target=self._run_server, daemon=True)
        self.server_thread.start()
        self.is_running = True
        time.sleep(1)
        if self.settings.server.open_browser:
            webbrowser.open(self.server_url)
        logger.info("Peleng Tracker started at %s", self.server_url)

    def _run_server(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        if self.server is not None:
            loop.run_until_complete(self.server.serve())

    def stop(self) -> None:
        if self.server is not None:
            self.server.should_exit = True
        if stream_combiner_manager.is_running:
            asyncio.run(stream_combiner_manager.stop())
        self.is_running = False

    def wait(self) -> None:
        try:
            while self.is_running:
                time.sleep(0.5)
        except KeyboardInterrupt:
            self.stop()

    def enable_tracking(self) -> None:
        stream_combiner_manager.enable_vision_tracking()

    def disable_tracking(self) -> None:
        stream_combiner_manager.disable_vision_tracking()


PelengTrackerConfig = AppSettings
