# DPJAIT R02_D1 replay bundle

This directory contains the 60.56-second, four-camera sequence used to verify Peleng Tracker's offline 3D reconstruction and replay editor.

Run it from the repository root:

```bash
uv run peleng-tracker demo run demo/github-slice/r02-d1
```

Open `http://127.0.0.1:8000` and select `3D Replay`.

## What is calculated by Peleng Tracker

`trajectory.json` is the output of the project pipeline. For every video frame, Peleng Tracker triangulates the synchronized 2D observations using `calibration.yaml`, then estimates position, velocity and acceleration with the trajectory filter.

`ground-truth.csv` is not used to reconstruct the trajectory or render it in the editor. It contains the Vicon reference positions used only by `demo evaluate` to calculate reconstruction error.

The current slice uses the YOLOv5 bounding-box centers supplied by DPJAIT in `observations.csv`. It does not yet run the project's own detector over the MP4 files.

## Files

- `camera-0.mp4` … `camera-3.mp4`: synchronized camera recordings, H.264, 25 FPS;
- `calibration.yaml`: four metric camera models;
- `observations.csv`: synchronized 2D bounding-box centers;
- `trajectory.json`: reconstructed Peleng Tracker trajectory;
- `ground-truth.csv`: Vicon reference trajectory for evaluation;
- `scenario.yaml`: replay manifest.

## Source and license

The source sequence is `R02_D1` from [DPJAIT — Multimodal Dataset for Indoor 3D Drone Tracking](https://zenodo.org/records/14748573), distributed under the Creative Commons Attribution 4.0 license. The videos in this directory are transcoded from the original AVI files; calibration, detections and Vicon measurements are derived from the same published sequence.
