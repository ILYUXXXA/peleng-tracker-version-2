#!/usr/bin/env python3
"""Convert EPFL Laboratory/POM calibration TXT to Peleng Tracker YAML."""

from __future__ import annotations

import argparse
from pathlib import Path

from peleng_tracker.epfl_calibration import convert_epfl_calibration
from peleng_tracker.geometry import CalibrationStore


def parse_camera_ids(value: str) -> list[int]:
    camera_ids = [int(item.strip()) for item in value.split(",") if item.strip()]
    if len(camera_ids) < 2 or len(set(camera_ids)) != len(camera_ids):
        raise argparse.ArgumentTypeError("provide at least two unique camera IDs")
    return camera_ids


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="EPFL calibration TXT")
    parser.add_argument("destination", type=Path, help="output Peleng Tracker YAML")
    parser.add_argument("--camera-ids", type=parse_camera_ids, default=[0, 1], help="comma-separated IDs (default: 0,1)")
    parser.add_argument("--image-width", type=int, default=360)
    parser.add_argument("--image-height", type=int, default=288)
    parser.add_argument("--head-height-m", type=float, default=1.75)
    parser.add_argument("--ground-units-per-meter", type=float, default=100.0)
    arguments = parser.parse_args()

    document = convert_epfl_calibration(
        arguments.source,
        camera_ids=arguments.camera_ids,
        image_size=(arguments.image_width, arguments.image_height),
        head_height_m=arguments.head_height_m,
        ground_units_per_meter=arguments.ground_units_per_meter,
    )
    CalibrationStore(document).save(arguments.destination)
    print(f"Wrote calibration for cameras {sorted(document.cameras)} to {arguments.destination}")


if __name__ == "__main__":
    main()
