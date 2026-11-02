# Live 3D tracking from phone cameras

## Current behaviour

- The detector runs without a class filter by default, so all classes supported by the configured YOLO weights can be detected.
- The dashboard lists local tracks from every active camera. Select a local track to nominate a target.
- The World View now renders calibrated world positions in a rotatable 3D scene.
- Only tracks with `position_status: valid` are plotted as 3D measurements. Planar fallback and unavailable positions are deliberately not presented as real 3D.

## Calibration requirement

Two phone video feeds are not enough to infer metric 3D camera geometry automatically. Each camera needs intrinsic parameters and a shared world coordinate frame (extrinsic parameters). The Settings tab now includes a guided stereo checkerboard wizard. It uploads paired still images to `POST /api/calibration/3d/checkerboard`, estimates both camera intrinsics and the relative pose, and saves a metric calibration. The existing `POST /api/calibration/3d` endpoint remains available for importing a previously computed calibration document.

For a real setup:

1. Print [the included 9 × 6 inner-corner checkerboard](checkerboard-9x6-50mm.svg) at **100% / actual size** (do not fit to page). It has 50 mm squares; measure one square edge after printing and enter the measured value in metres.
2. Rigidly mount both phones. Lock focus and zoom, use the same fixed resolution, and avoid moving either phone after calibration.
3. Capture at least 12 corresponding image pairs. In each pair, the board must be in the same pose in both cameras. Vary distance, tilt and image position across pairs; keep the full board visible and sharply focused.
4. In Settings → Calibration, set the camera IDs, board dimensions and square size. Select the camera-A files and camera-B files in matching order, then start calibration.
5. The endpoint rejects fewer than 10 usable pairs or stereo RMS above 3 px. Prefer RMS around 1–2 px or lower. If rejected, capture more varied, sharper pairs.
6. After success, keep both phones fixed. Camera A defines the world origin; the measured baseline is reported in metres. Recalibrate after moving a phone, changing zoom/focus or changing resolution.

The wizard calibrates a two-camera stereo pair. Additional cameras require their own stereo calibration into the same world coordinate frame; do not simply append a third independent pair.

## Verification

Run the backend and frontend checks from the repository root:

```powershell
uv run pytest
cd web
npm ci
npm run lint
npm run build
```

A valid calibration status alone is not a substitute for checking reprojection error and confirming that both cameras observe the same moving object at synchronized timestamps. 3D positions are only shown when triangulation succeeds; single-camera tracks remain 2D and are not fabricated into depth.
