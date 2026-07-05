#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${REPO_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}"

cd "${REPO_DIR}"
export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"

WEIGHTS="${WEIGHTS:?Set WEIGHTS=/path/to/best.pt or last.pt}"
DATA="${DATA:-${REPO_DIR}/datasets/data.yaml}"
PROJECT="${PROJECT:-/root/autodl-tmp/bmvc_rebuttal/eval}"
NAME="${NAME:-eval}"
IMGSZ="${IMGSZ:-640}"
BATCH="${BATCH:-32}"
WORKERS="${WORKERS:-8}"
DEVICE="${DEVICE:-0}"

python rebuttal_eval.py \
  --weights "${WEIGHTS}" \
  --data "${DATA}" \
  --project "${PROJECT}" \
  --name "${NAME}" \
  --imgsz "${IMGSZ}" \
  --batch "${BATCH}" \
  --workers "${WORKERS}" \
  --device "${DEVICE}" \
  "$@"
