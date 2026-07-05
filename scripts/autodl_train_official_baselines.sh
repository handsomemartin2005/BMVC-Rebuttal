#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${REPO_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}"

cd "${REPO_DIR}"

DATA="${DATA:-${REPO_DIR}/datasets/data.yaml}"
PROJECT="${PROJECT:-/root/autodl-tmp/bmvc_rebuttal/official_baselines}"
MODELS="${MODELS:-yolov8n.pt yolov8s.pt yolov10n.pt yolov10s.pt yolo11n.pt yolo11s.pt}"
EPOCHS="${EPOCHS:-500}"
IMGSZ="${IMGSZ:-640}"
BATCH="${BATCH:-32}"
WORKERS="${WORKERS:-8}"
DEVICE="${DEVICE:-0}"
RUNS="${RUNS:-1}"

python rebuttal_train_official_baselines.py \
  --data "${DATA}" \
  --project "${PROJECT}" \
  --models ${MODELS} \
  --epochs "${EPOCHS}" \
  --imgsz "${IMGSZ}" \
  --batch "${BATCH}" \
  --workers "${WORKERS}" \
  --device "${DEVICE}" \
  --runs "${RUNS}" \
  "$@"
