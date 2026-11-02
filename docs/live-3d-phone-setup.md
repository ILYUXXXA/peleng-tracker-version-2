# Live 3D tracking from phone cameras

## Current behaviour

- The detector runs without a class filter by default, so all classes supported by the configured YOLO weights can be detected.
- The dashboard lists local tracks from every active camera. Select a local track to nominate a target.
- The World View now renders calibrated world positions in a rotatable 3D scene.
- Only tracks with `position_status: valid` are plotted as 3D measurements. Planar fallback and unavailable positions are deliberately not presented as real 3D.

## Calibration requirement

Two phone video feeds are not enough to infer metric 3D camera geometry automatically. Each camera needs intrinsic parameters and a shared world coordinate frame (extrinsic parameters). The current `POST /api/calibration/3d` endpoint validates and loads a calibration document; it does not estimate one from phone footage.

For a real setup, calibrate both phones with a printed checkerboard or ChArUco board using the same board dimensions, fixed focus/zoom, and a rigid camera placement. Estimate intrinsics for each phone and stereo extrinsics for the pair, then provide the resulting camera matrices, distortion coefficients, rotations and translations in the calibration YAML. Recalibrate if camera position, zoom, focus or resolution changes.

## Verification

Run the backend and frontend checks from the repository root:

```powershell
uv run pytest
cd web
npm ci
npm run lint
npm run build
```

A valid calibration status alone is not a substitute for checking reprojection error and confirming that both cameras observe the same moving object at synchronized timestamps.
