"""Peleng Tracker command line interface."""

from __future__ import annotations

import json
import time
from pathlib import Path

import click
import numpy as np

from .geometry import CalibrationStore
from .replay import ReplayBundle
from .settings import AppSettings


@click.group()
@click.version_option()
def cli() -> None:
    """Multicamera object tracking and 3D localization."""


@cli.command()
@click.option("--config", "config_path", default="config/hackathon.yaml", show_default=True)
def run(config_path: str) -> None:
    """Start the API, WebRTC stream and dashboard."""
    from .core.peleng_tracker_app import PelengTrackerApp

    app = PelengTrackerApp(AppSettings.load(config_path))
    app.start()
    app.wait()


@cli.command("validate-calibration")
@click.argument("path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
def validate_calibration(path: Path) -> None:
    """Validate a 3D camera calibration file."""
    store = CalibrationStore.load(path)
    click.echo(json.dumps(store.status(), indent=2))
    if not store.status()["valid"]:
        raise click.ClickException("at least two calibrated cameras are required for 3D")


@cli.command()
@click.option("--config", "config_path", default="config/hackathon.yaml", show_default=True)
@click.option("--iterations", default=20, type=click.IntRange(1, 1000), show_default=True)
def benchmark(config_path: str, iterations: int) -> None:
    """Measure detector latency on synthetic frames."""
    from .factory import create_detector

    settings = AppSettings.load(config_path)
    detector = create_detector(settings.vision)
    frame = np.zeros((settings.vision.detector.image_size, settings.vision.detector.image_size, 3), dtype=np.uint8)
    detector.warmup()
    started = time.perf_counter()
    for _ in range(iterations):
        detector.detect(frame)
    elapsed = time.perf_counter() - started
    click.echo(f"model={detector.weights} avg_ms={elapsed * 1000 / iterations:.2f} fps={iterations / elapsed:.2f}")


@cli.group()
def demo() -> None:
    """Validate, process, evaluate and open deterministic replay bundles."""


@demo.command("import-dpjait")
@click.argument("source", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--sequence", default="R02_D1", show_default=True)
@click.option("--output", type=click.Path(file_okay=False, path_type=Path), required=True)
@click.option("--vicon-offset-samples", default=-3, type=int, show_default=True)
@click.option("--use-distortion/--ignore-distortion", default=False, show_default=True)
@click.option("--transcode/--copy-videos", default=True, show_default=True)
def import_dpjait_demo(
    source: Path,
    sequence: str,
    output: Path,
    vicon_offset_samples: int,
    use_distortion: bool,
    transcode: bool,
) -> None:
    """Import one real DPJAIT sequence as a replay bundle."""
    from .dpjait import import_dpjait_sequence

    result = import_dpjait_sequence(
        source,
        output,
        sequence=sequence,
        vicon_offset_samples=vicon_offset_samples,
        use_distortion=use_distortion,
        transcode=transcode,
    )
    click.echo(
        json.dumps(
            {
                "bundle": str(result.bundle_path),
                "cameras": result.camera_count,
                "frames": result.frame_count,
                "observations": result.observation_count,
                "vicon_rate_hz": result.vicon_rate_hz,
                "vicon_offset_samples": result.vicon_offset_samples,
            },
            indent=2,
        )
    )


@demo.command("validate")
@click.argument("path", type=click.Path(exists=True, file_okay=False, path_type=Path))
def validate_demo(path: Path) -> None:
    """Validate a local scenario bundle."""
    bundle = ReplayBundle.load(path)
    click.echo(json.dumps(bundle.validate(), indent=2))


@demo.command("process")
@click.argument("path", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--max-reprojection-error-px", default=12.0, type=click.FloatRange(min=0.0), show_default=True)
def process_demo(path: Path, max_reprojection_error_px: float) -> None:
    """Reconstruct and save a 3D trajectory from normalized 2D observations."""
    bundle = ReplayBundle.load(path)
    bundle.validate()
    trajectory = bundle.process(max_reprojection_error_px=max_reprojection_error_px)
    click.echo(json.dumps({"scenario_id": trajectory.scenario_id, "states": len(trajectory.states)}, indent=2))


@demo.command("evaluate")
@click.argument("path", type=click.Path(exists=True, file_okay=False, path_type=Path))
def evaluate_demo(path: Path) -> None:
    """Compare a processed trajectory with optional 3D ground truth."""
    bundle = ReplayBundle.load(path, require_trajectory=True)
    click.echo(json.dumps(bundle.evaluation(), indent=2))


@demo.command("run")
@click.argument("path", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--host", default="127.0.0.1", show_default=True)
@click.option("--port", default=8000, type=click.IntRange(1, 65535), show_default=True)
def run_demo(path: Path, host: str, port: int) -> None:
    """Open an already processed bundle in the replay editor."""
    import os

    import uvicorn

    bundle = ReplayBundle.load(path, require_trajectory=True)
    bundle.validate(require_trajectory=True)
    os.environ["PELENG_REPLAY_BUNDLE"] = str(path.resolve())
    uvicorn.run("peleng_tracker.core.app:app", host=host, port=port, log_level="info")


def main() -> None:
    cli()


if __name__ == "__main__":
    main()
