#!/usr/bin/env python3
"""Prepare a pseudo-labelled YOLO dataset, fine-tune, and validate on Peleng Unreal videos."""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import shutil
from pathlib import Path

import cv2
import numpy as np
import yaml

W, H, SENSOR_W, SENSOR_H = 1920, 1080, 23.76, 13.365
CAM_IDS = range(1, 6)


def rows(path: Path) -> list[list[str]]:
    with path.open(encoding="cp1251", newline="") as f:
        return list(csv.reader(f, delimiter=";"))


def read_target(path: Path) -> dict[int, np.ndarray]:
    result = {}
    for row in rows(path)[1:]:
        if len(row) < 4:
            continue
        try:
            result[int(float(row[0]))] = np.array([float(row[1]), float(row[2]), float(row[3])])
        except ValueError:
            pass
    if not result:
        raise ValueError(f"No target coordinates in {path}")
    return result


class Camera:
    def __init__(self, camera_id: int, path: Path):
        data = rows(path)
        self.id = camera_id
        self.focal = float(data[0][-1])
        frame, x, y, z, roll, pitch, yaw = map(float, data[1][:7])
        self.pos = np.array([x, y, z])
        r, p, a = np.deg2rad([roll, pitch, yaw])
        cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(a), math.sin(a)
        self.fwd = np.array([cp * cy, cp * sy, sp])
        self.right = np.array([sr * sp * cy - cr * sy, sr * sp * sy + cr * cy, -sr * cp])
        self.up = np.array([-(cr * sp * cy + sr * sy), cy * sr - cr * sp * sy, cr * cp])
        self.fx, self.fy = self.focal / SENSOR_W * W, self.focal / SENSOR_H * H
        self.cx, self.cy = W / 2, H / 2

    def project(self, point: np.ndarray):
        d = point - self.pos
        z = float(np.dot(d, self.fwd))
        if z <= 1:
            return None
        return self.cx + self.fx * np.dot(d, self.right) / z, self.cy - self.fy * np.dot(d, self.up) / z, z / 1000

    def projection_matrix(self):
        center = np.array([self.pos[0], -self.pos[1], self.pos[2]], dtype=float) / 1000
        forward = np.array([self.fwd[0], -self.fwd[1], self.fwd[2]])
        right = np.array([self.right[0], -self.right[1], self.right[2]])
        up = np.array([self.up[0], -self.up[1], self.up[2]])
        rotation = np.vstack([right, -up, forward])
        translation = -rotation @ center
        intrinsic = np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1]], dtype=float)
        return intrinsic @ np.column_stack([rotation, translation])

    def cuboid_box(self, point: np.ndarray, heading: np.ndarray, length: float, width: float, height: float):
        norm = np.linalg.norm(heading)
        if norm < 1e-6:
            return None
        heading = heading / norm
        side = np.array([-heading[1], heading[0]])
        points = []
        for along in (0, -length * 1000):
            for lateral in (-width * 500, width * 500):
                for vertical in (-height * 500, height * 500):
                    q = point.copy()
                    q[0] += heading[0] * along + side[0] * lateral
                    q[1] += heading[1] * along + side[1] * lateral
                    q[2] += vertical
                    proj = self.project(q)
                    if proj:
                        points.append(proj[:2])
        if len(points) < 4:
            return None
        pts = np.asarray(points)
        x1, y1 = pts.min(axis=0)
        x2, y2 = pts.max(axis=0)
        raw_area = max(1, (x2 - x1) * (y2 - y1))
        if x2 <= 0 or y2 <= 0 or x1 >= W or y1 >= H:
            return None
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        xx1, yy1, xx2, yy2 = max(0, x1), max(0, y1), min(W - 1, x2), min(H - 1, y2)
        visible = max(0, xx2 - xx1) * max(0, yy2 - yy1) / raw_area
        if not (0 <= cx < W and 0 <= cy < H) or visible < 0.55:
            return None
        return float(cx), float(cy), float(xx2 - xx1), float(yy2 - yy1)


def load_dataset(source: Path):
    target = read_target(source / "object_transform_SM_AA_track.csv")
    cameras, videos = {}, {}
    for cid in CAM_IDS:
        cameras[cid] = Camera(cid, source / f"object_transform_CineCameraActor{cid}.csv")
        videos[cid] = source / f"Way1.FinalImage.CineCameraActor{cid}.mp4"
        if not videos[cid].is_file():
            raise FileNotFoundError(videos[cid])
    return target, cameras, videos


def write_calibration(path: Path, cameras: dict[int, Camera]):
    doc = {"coordinate_system": "right-handed", "units": "meters", "cameras": {}}
    for source_id, cam in cameras.items():
        center = np.array([cam.pos[0], -cam.pos[1], cam.pos[2]]) / 1000
        forward = np.array([cam.fwd[0], -cam.fwd[1], cam.fwd[2]])
        right = np.array([cam.right[0], -cam.right[1], cam.right[2]])
        up = np.array([cam.up[0], -cam.up[1], cam.up[2]])
        rotation = np.vstack([right, -up, forward])
        translation = -rotation @ center
        doc["cameras"][source_id - 1] = {
            "image_size": [W, H],
            "camera_matrix": [[cam.fx, 0, cam.cx], [0, cam.fy, cam.cy], [0, 0, 1]],
            "distortion_coefficients": [0, 0, 0, 0, 0],
            "camera_forward_axis": "positive_z",
            "rotation_world_to_camera": rotation.tolist(),
            "translation_world_to_camera": translation.tolist(),
        }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")


def prepare(args):
    source, output = Path(args.source).resolve(), Path(args.output).resolve()
    target, cameras, videos = load_dataset(source)
    if output.exists() and any(output.iterdir()):
        if not args.overwrite:
            raise FileExistsError(f"{output} is not empty; pass --overwrite")
        shutil.rmtree(output)
    for split in ("train", "val"):
        (output / "images" / split).mkdir(parents=True)
        (output / "labels" / split).mkdir(parents=True)
    (output / "review").mkdir()
    write_calibration(output / "calibration.yaml", cameras)
    manifest, stats = [], {}
    for cid, video in videos.items():
        cap = cv2.VideoCapture(str(video))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open {video}")
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        lag, previous = [], None
        kept = rejected = 0
        i = -1
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            i += 1
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            lag.append((i, gray))
            lag = lag[-(args.motion_lag + 1):]
            if i % args.stride:
                continue
            point, before, after = target.get(i), target.get(i - 1), target.get(i + 1)
            if point is None or before is None or after is None:
                continue
            box = cameras[cid].cuboid_box(point, after[:2] - before[:2], args.length, args.width, args.height)
            if box is None:
                continue
            old = next((im for fi, im in reversed(lag) if fi == i - args.motion_lag), None)
            score = 0
            if old is not None:
                u, v, bw, bh = box
                x1, x2 = max(0, int(u - bw * 0.75)), min(W, int(u + bw * 0.75))
                y1, y2 = max(0, int(v - bh * 0.75)), min(H, int(v + bh * 0.75))
                # Crop before frame differencing; processing all 1080p pixels per sample is wasteful.
                old_roi, current_roi = old[y1:y2, x1:x2], gray[y1:y2, x1:x2]
                diff = cv2.GaussianBlur(cv2.absdiff(old_roi, current_roi), (5, 5), 0)
                _, mask = cv2.threshold(diff, args.diff_threshold, 255, cv2.THRESH_BINARY)
                mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)
                score = int(np.count_nonzero(mask))
            if args.motion_filter and score < args.min_motion_pixels:
                rejected += 1
                continue
            split = "train" if i < count * args.train_fraction else "val"
            stem = f"cam{cid}_frame{i:04d}"
            image = output / "images" / split / f"{stem}.jpg"
            label = output / "labels" / split / f"{stem}.txt"
            # Resize training images to 1280x720; normalized YOLO labels remain valid.
            training_frame = cv2.resize(frame, (1280, 720), interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(image), training_frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
            u, v, bw, bh = box
            label.write_text(f"0 {u/W:.8f} {v/H:.8f} {bw/W:.8f} {bh/H:.8f}\n", encoding="utf-8")
            manifest.append([cid, i, split, u, v, bw, bh, score, image.relative_to(output).as_posix()])
            kept += 1
            if kept <= args.review_per_camera:
                review = frame.copy()
                cv2.rectangle(review, (int(u-bw/2), int(v-bh/2)), (int(u+bw/2), int(v+bh/2)), (0,0,255), 3)
                cv2.circle(review, (int(u), int(v)), 5, (0,255,255), -1)
                cv2.putText(review, f"cam={cid} frame={i} motion={score}", (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                cv2.imwrite(str(output / "review" / f"{stem}.jpg"), cv2.resize(review, (960,540)))
        cap.release()
        stats[cid] = {"frames": count, "kept": kept, "motion_rejected": rejected}
        print(f"camera {cid}: {stats[cid]}")
    if not manifest or not any(r[2] == "val" for r in manifest):
        raise RuntimeError("No examples or no validation examples. Try --no-motion-filter or --stride 1.")
    with (output / "labels_manifest.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["camera_id","frame_index","split","u","v","box_w","box_h","motion_pixels","image"]); w.writerows(manifest)
    (output / "data.yaml").write_text(f"path: {output.as_posix()}\ntrain: images/train\nval: images/val\nnames:\n  0: target\n", encoding="utf-8")
    (output / "prepare_report.json").write_text(json.dumps({"camera_stats": stats, "examples": len(manifest),
        "train": sum(r[2] == "train" for r in manifest), "val": sum(r[2] == "val" for r in manifest),
        "warning": "Projected cuboid pseudo-labels are approximate. Inspect review/ and adjust dimensions before training."}, indent=2), encoding="utf-8")
    print(f"Dataset written to {output}; inspect {output / 'review'} before training.")


def train(args):
    from ultralytics import YOLO
    random.seed(args.seed); np.random.seed(args.seed)
    model = YOLO(args.base_weights)
    model.train(data=str((Path(args.data).resolve() / "data.yaml")), imgsz=args.imgsz, epochs=args.epochs,
        batch=args.batch, device=args.device, workers=args.workers, patience=args.patience,
        project=args.project, name=args.name, seed=args.seed, deterministic=True, pretrained=True,
        plots=True, cos_lr=True, close_mosaic=10, mosaic=0.5, mixup=0.05,
        hsv_h=0.015, hsv_s=0.5, hsv_v=0.35, degrees=3, translate=0.08, scale=0.4,
        fliplr=0.5, flipud=0.0, copy_paste=0.0, verbose=True)
    print(f"Best weights: {Path(args.project) / args.name / 'weights' / 'best.pt'}")


def triangulate(observations):
    rows = []
    for camera, u, v in observations:
        projection = camera.projection_matrix()
        rows.extend([u * projection[2] - projection[0], v * projection[2] - projection[1]])
    if len(rows) < 4:
        return None
    _, _, vh = np.linalg.svd(np.asarray(rows))
    homogeneous = vh[-1]
    if abs(homogeneous[3]) < 1e-12:
        return None
    return homogeneous[:3] / homogeneous[3]


def validate(args):
    from ultralytics import YOLO
    source = Path(args.source).resolve()
    target, cameras, videos = load_dataset(source)
    model = YOLO(args.weights)
    output = Path(args.output).resolve(); output.mkdir(parents=True, exist_ok=True)
    rows_out, per_camera = [], {}
    observations_by_frame = {}
    for cid, video in videos.items():
        cap = cv2.VideoCapture(str(video)); frame_id = -1; errors = []; eligible = found = 0
        while True:
            ok, frame = cap.read()
            if not ok: break
            frame_id += 1
            if frame_id % args.stride or frame_id not in target: continue
            prev, nxt = target.get(frame_id - 1), target.get(frame_id + 1)
            if prev is None or nxt is None: continue
            gt_box = cameras[cid].cuboid_box(target[frame_id], nxt[:2]-prev[:2], args.length, args.width, args.height)
            if gt_box is None: continue
            eligible += 1; u, v, _, _ = gt_box
            preds = model.predict(frame, imgsz=args.imgsz, conf=args.conf, iou=args.iou, device=args.device, verbose=False)[0]
            candidates = []
            if preds.boxes is not None:
                for b in preds.boxes:
                    x1,y1,x2,y2 = map(float, b.xyxy[0].cpu().numpy())
                    cx,cy=(x1+x2)/2,(y1+y2)/2
                    d=math.hypot(cx-u,cy-v)
                    if d <= args.match_radius: candidates.append((d,float(b.conf[0].cpu()),cx,cy))
            candidates.sort(key=lambda x:(x[0],-x[1]))
            ok_det = bool(candidates)
            err = candidates[0][0] if ok_det else None
            found += int(ok_det)
            if err is not None: errors.append(err)
            rows_out.append({"camera_id":cid,"frame_index":frame_id,"detected":int(ok_det),"pixel_error":err,
                             "confidence":candidates[0][1] if ok_det else 0,
                             "pred_u":candidates[0][2] if ok_det else None,
                             "pred_v":candidates[0][3] if ok_det else None})
            if ok_det:
                observations_by_frame.setdefault(frame_id, []).append((cameras[cid], candidates[0][2], candidates[0][3]))
        cap.release()
        per_camera[cid]={"eligible_frames":eligible,"detected_frames":found,
            "recall_at_match_radius":found/eligible if eligible else None,
            "median_pixel_error":float(np.median(errors)) if errors else None,
            "p95_pixel_error":float(np.percentile(errors,95)) if errors else None}
        print(f"camera {cid}: {per_camera[cid]}")
    with (output / "detections.csv").open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=["camera_id","frame_index","detected","pixel_error","confidence","pred_u","pred_v"])
        writer.writeheader(); writer.writerows(rows_out)
    xyz_errors, xyz_rows = [], []
    for frame_id, observations in observations_by_frame.items():
        if len(observations) < 2 or frame_id - 1 not in target or frame_id + 1 not in target:
            continue
        reconstructed = triangulate(observations)
        if reconstructed is None:
            continue
        heading = target[frame_id + 1][:2] - target[frame_id - 1][:2]
        norm = np.linalg.norm(heading)
        if norm < 1e-6:
            continue
        heading = heading / norm
        # The target transform is near the truck's front-center; compare with the estimated mesh center.
        center = target[frame_id].copy()
        center[:2] -= heading * args.length * 500
        gt = np.array([center[0], -center[1], center[2]]) / 1000
        error = float(np.linalg.norm(reconstructed - gt))
        xyz_errors.append(error)
        xyz_rows.append({"frame_index":frame_id,"x_pred_m":reconstructed[0],"y_pred_m":reconstructed[1],
                         "z_pred_m":reconstructed[2],"x_gt_m":gt[0],"y_gt_m":gt[1],"z_gt_m":gt[2],
                         "error_3d_m":error,"camera_count":len(observations)})
    with (output / "triangulation_errors.csv").open("w",newline="",encoding="utf-8") as f:
        fields = list(xyz_rows[0]) if xyz_rows else ["frame_index","error_3d_m"]
        writer=csv.DictWriter(f,fieldnames=fields); writer.writeheader(); writer.writerows(xyz_rows)
    report={"weights":str(Path(args.weights).resolve()),"per_camera":per_camera,
        "eligible_frames":len(rows_out),"detections":sum(r["detected"] for r in rows_out),
        "recall_at_match_radius":sum(r["detected"] for r in rows_out)/len(rows_out) if rows_out else None,
        "3d_valid_frames":len(xyz_errors),
        "3d_rmse_m":float(np.sqrt(np.mean(np.square(xyz_errors)))) if xyz_errors else None,
        "3d_mae_m":float(np.mean(xyz_errors)) if xyz_errors else None,
        "3d_p95_error_m":float(np.percentile(xyz_errors,95)) if xyz_errors else None,
        "note":"GT-projection-assisted association is used only for offline evaluation. 3D errors are conditional on detections in at least two cameras; the mesh-center comparison assumes the trajectory marks the truck front-center and uses the configured length."}
    (output / "validation_report.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps(report,indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__); sub=p.add_subparsers(dest="cmd",required=True)
    a=sub.add_parser("prepare"); a.add_argument("--source",required=True); a.add_argument("--output",default="data/peleng-yolo")
    a.add_argument("--stride",type=int,default=2); a.add_argument("--train-fraction",type=float,default=.8)
    a.add_argument("--length","--object-length-m",dest="length",type=float,default=2.5); a.add_argument("--width","--object-width-m",dest="width",type=float,default=.7); a.add_argument("--height","--object-height-m",dest="height",type=float,default=.8)
    a.add_argument("--motion-filter",dest="motion_filter",action="store_true",default=True); a.add_argument("--no-motion-filter",dest="motion_filter",action="store_false")
    a.add_argument("--motion-lag",type=int,default=5); a.add_argument("--diff-threshold",type=int,default=18); a.add_argument("--min-motion-pixels",type=int,default=35)
    a.add_argument("--review-per-camera",type=int,default=8); a.add_argument("--overwrite",action="store_true"); a.set_defaults(func=prepare)
    b=sub.add_parser("train"); b.add_argument("--data",default="data/peleng-yolo"); b.add_argument("--base-weights",default="yolo11n.pt")
    b.add_argument("--project",default="runs/peleng"); b.add_argument("--name",default="yolo11n-peleng"); b.add_argument("--epochs",type=int,default=60)
    b.add_argument("--imgsz",type=int,default=960); b.add_argument("--batch",type=int,default=4); b.add_argument("--device",default="auto")
    b.add_argument("--workers",type=int,default=2); b.add_argument("--patience",type=int,default=12); b.add_argument("--seed",type=int,default=42); b.set_defaults(func=train)
    c=sub.add_parser("validate"); c.add_argument("--source",required=True); c.add_argument("--weights",required=True); c.add_argument("--output",default="runs/peleng/validation")
    c.add_argument("--stride",type=int,default=5); c.add_argument("--imgsz",type=int,default=960); c.add_argument("--conf",type=float,default=.15)
    c.add_argument("--iou",type=float,default=.7); c.add_argument("--device",default="auto"); c.add_argument("--match-radius",type=float,default=120)
    c.add_argument("--length","--object-length-m",dest="length",type=float,default=2.5); c.add_argument("--width","--object-width-m",dest="width",type=float,default=.7); c.add_argument("--height","--object-height-m",dest="height",type=float,default=.8); c.set_defaults(func=validate)
    args=p.parse_args(); args.func(args)


if __name__ == "__main__":
    main()
