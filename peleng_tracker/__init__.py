"""Peleng Tracker public entrypoints."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .settings import AppSettings

if TYPE_CHECKING:
    from .core.peleng_tracker_app import PelengTrackerApp

__version__ = "0.2.0"
PelengTrackerConfig = AppSettings
__all__ = ["AppSettings", "PelengTrackerConfig", "launch"]


def launch(config: str | None = None, *, warmup: bool = True) -> PelengTrackerApp:
    from .core.peleng_tracker_app import PelengTrackerApp

    app = PelengTrackerApp(AppSettings.load(config), warmup=warmup)
    app.start()
    return app
