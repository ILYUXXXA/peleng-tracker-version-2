#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 CAMERA_0_VIDEO CAMERA_1_VIDEO" >&2
  exit 2
fi

docker compose up -d mediamtx
ffmpeg -re -stream_loop -1 -i "$1" -c copy -rtsp_transport tcp -f rtsp rtsp://127.0.0.1:8554/camera0 &
first_pid=$!
ffmpeg -re -stream_loop -1 -i "$2" -c copy -rtsp_transport tcp -f rtsp rtsp://127.0.0.1:8554/camera1 &
second_pid=$!
trap 'kill "$first_pid" ${second_pid:-} 2>/dev/null || true' EXIT
uv run peleng-tracker run --config config/offline-demo.yaml
