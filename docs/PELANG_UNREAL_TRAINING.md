# Peleng Unreal dataset: preparation, training, validation, run

This guide applies to the five MP4 files and six CSV files in the hackathon archive. It adds a YOLO fine-tuning workflow to the existing Peleng Tracker application.

## 1. What the workflow does

- Reads the target trajectory and each camera's Unreal pose/focal length.
- Treats the CSV coordinates as **millimetres** and converts camera calibration to a right-handed world coordinate system in metres.
- Projects an approximate oriented 3D cuboid onto each image to create YOLO pseudo-labels.
- Uses temporal frame differences as a visibility filter and saves review images.
- Fine-tunes YOLO with image augmentation and reports validation precision/recall/mAP.
- Separately reports GT-projection-assisted 2D localization and conditional 3D triangulation errors.

Important: the archive contains trajectory points, not true 2D bounding-box annotations. The cuboid dimensions (defaults: 2.5 m long, 0.7 m wide, 0.8 m high) are an approximation inferred from the rendered videos. Inspect the generated review images before training and adjust dimensions if the red rectangles do not cover the moving truck. Do not claim pseudo-labels are ground-truth manual annotations.

## 2. Install on Windows

Install Python 3.11 or 3.12, Git, Node.js 20+, uv, FFmpeg, Docker Desktop, and 7-Zip.

Clone the feature branch:

~~~powershell
git clone -b feat/peleng-dataset-training https://github.com/ILYUXXXA/peleng-tracker-version-2.git
cd peleng-tracker-version-2
uv sync --extra dev
cd web
npm ci
npm run build
cd ..
~~~

Extract project2.7z into data/peleng-source. That folder should directly contain:
- object_transform_SM_AA_track.csv
- object_transform_CineCameraActor1.csv through object_transform_CineCameraActor5.csv
- Way1.FinalImage.CineCameraActor1.mp4 through Way1.FinalImage.CineCameraActor5.mp4

The archive and data/ are ignored by Git, so videos and weights will not be committed.

## 3. Generate training data and calibration

From the repository root:

~~~powershell
uv run python scripts/peleng_ml.py prepare --source data/peleng-source --output data/peleng-yolo --stride 2 --overwrite
~~~

The command creates images/train and images/val, YOLO labels, data.yaml, labels_manifest.csv, prepare_report.json, review/ previews, and calibration.yaml for all five cameras mapped to application camera IDs 0–4.

Open several images in data/peleng-yolo/review. If boxes are consistently too long/short/tall, regenerate with adjusted dimensions:

~~~powershell
uv run python scripts/peleng_ml.py prepare --source data/peleng-source --output data/peleng-yolo --stride 2 --length 2.8 --width 0.75 --height 0.85 --overwrite
~~~

If motion filtering removes too many valid frames, lower --min-motion-pixels or try --no-motion-filter; then manually inspect labels because occluded objects can receive incorrect boxes. Do not train until the review looks reasonable.

The split is chronological: the first 80% of each video is training data and the final 20% is validation data. This reduces leakage from adjacent frames compared with a random frame split, but all videos still come from one synthetic scene and one trajectory.

## 4. Train with augmentation

GPU example:

~~~powershell
uv run python scripts/peleng_ml.py train --data data/peleng-yolo --base-weights yolo11n.pt --epochs 60 --imgsz 960 --batch 4 --device 0
~~~

CPU fallback (much slower):

~~~powershell
uv run python scripts/peleng_ml.py train --data data/peleng-yolo --base-weights yolo11n.pt --epochs 40 --imgsz 640 --batch 2 --device cpu --workers 0
~~~

On first run, Ultralytics downloads yolo11n.pt. A network connection is required unless the weights are already cached. If CUDA reports out-of-memory, reduce --batch to 2 or --imgsz to 640.

Training uses HSV/color changes, translation, scale, small rotations, horizontal flips, mosaic and light mixup. It writes checkpoints and metrics under runs/peleng/yolo11n-peleng/. The model to use is runs/peleng/yolo11n-peleng/weights/best.pt. Keep best.pt local; model weights are ignored by Git.

## 5. Validate

The trainer's results.csv, PR curves, confusion matrix and mAP plots are the standard YOLO validation results. Run the additional video-level check:

~~~powershell
uv run python scripts/peleng_ml.py validate --source data/peleng-source --weights runs/peleng/yolo11n-peleng/weights/best.pt --output runs/peleng/validation --stride 5
~~~

Outputs: validation_report.json, detections.csv, and triangulation_errors.csv.

The validation script uses the known trajectory projection to associate predicted boxes with the target in offline evaluation. The reported 3D error is **conditional**: it only uses frames with detections in at least two cameras. It also assumes the trajectory point is near the truck's front-centre and shifts by half the configured vehicle length to compare against the estimated mesh centre. This is useful for diagnosing detector/calibration, but it is not a blind end-to-end tracking score.

For a diagnostic baseline, run the same validation command with --weights yolo11n.pt, then compare recall and pixel errors with the fine-tuned model. The baseline has COCO classes and a different training objective, so treat this comparison as diagnostic rather than perfectly controlled.

## 6. Run the full application on the five videos

Copy the example config:

~~~powershell
Copy-Item config/peleng-unreal.example.yaml config/peleng-unreal.local.yaml
~~~

Ensure vision.detector.weights points to best.pt and vision.geometry.calibration_file points to data/peleng-yolo/calibration.yaml.

The sample config sets custom detector class [0], because the training dataset contains one class named target. A code fix in this branch also ensures custom class 0 is not incorrectly treated as a COCO person for the geometry anchor.

Enable host networking in Docker Desktop if your setup requires it. Then run PowerShell:

~~~powershell
Set-ExecutionPolicy -Scope Process Bypass
.\scripts\offline-demo-peleng.ps1 -DatasetDir data\peleng-source -ConfigPath config\peleng-unreal.local.yaml
~~~

The script publishes the five MP4s as RTSP streams, launches the dashboard, and stops publishers when the app exits. Open http://127.0.0.1:8000.

For 3D association, streams must remain synchronized. If too few multi-camera positions are accepted, inspect stream startup/skew and camera calibration before changing reprojection thresholds.

## 7. What is and is not validated

- Camera projection/calibration is derived from the supplied camera pose CSVs, focal length, and assumed sensor dimensions (23.76 × 13.365 mm); lens distortion is assumed zero.
- Bounding boxes are pseudo-labels from a projected cuboid, filtered by motion. They are not hand-labelled annotations.
- YOLO validation mAP is measured against those pseudo-labels, so it measures consistency with generated labels, not independent real-world accuracy.
- 2D localization and 3D errors use GT-assisted offline association. The report records these limits.
- The archive contains one scene/trajectory. Generalization claims require additional scenes, trajectories, lighting, and object appearances.
