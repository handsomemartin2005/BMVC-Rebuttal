#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${REPO_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}"

cd "${REPO_DIR}"
export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"

python -m pip install --upgrade pip
python -m pip install -r requirements.txt

DATA="${DATA:-${REPO_DIR}/datasets/data.yaml}"
PROJECT_ROOT="${PROJECT_ROOT:-/root/autodl-tmp/bmvc_rebuttal}"
EPOCHS="${EPOCHS:-500}"
IMGSZ="${IMGSZ:-640}"
BATCH="${BATCH:-32}"
WORKERS="${WORKERS:-8}"
DEVICE="${DEVICE:-0}"
PRETRAINED="${PRETRAINED:-True}"

POOLING_VARIANTS="${POOLING_VARIANTS:-gap gmp gap_gmp top1 ppap}"
GRAPH_VARIANTS="${GRAPH_VARIANTS:-minimal}"

RUN_MODULE_ABLATION="${RUN_MODULE_ABLATION:-0}"
RUN_POOLING_ABLATION="${RUN_POOLING_ABLATION:-1}"
RUN_GRAPH_ABLATION="${RUN_GRAPH_ABLATION:-1}"
RUN_PROFILE="${RUN_PROFILE:-1}"

if [[ "${RUN_MODULE_ABLATION}" == "1" ]]; then
  python ablation.py \
    --variants six \
    --data "${DATA}" \
    --project "${PROJECT_ROOT}/module_ablation" \
    --name-prefix "module" \
    --epochs "${EPOCHS}" \
    --imgsz "${IMGSZ}" \
    --batch "${BATCH}" \
    --workers "${WORKERS}" \
    --device "${DEVICE}" \
    --pretrained "${PRETRAINED}" \
    "$@"
fi

if [[ "${RUN_POOLING_ABLATION}" == "1" ]]; then
  python rebuttal_pooling_ablation.py \
    --variants ${POOLING_VARIANTS} \
    --data "${DATA}" \
    --project "${PROJECT_ROOT}/pooling_ablation" \
    --name-prefix "pooling" \
    --epochs "${EPOCHS}" \
    --imgsz "${IMGSZ}" \
    --batch "${BATCH}" \
    --workers "${WORKERS}" \
    --device "${DEVICE}" \
    --pretrained "${PRETRAINED}" \
    "$@"
fi

if [[ "${RUN_GRAPH_ABLATION}" == "1" ]]; then
  python rebuttal_graph_ablation.py \
    --variants ${GRAPH_VARIANTS} \
    --data "${DATA}" \
    --project "${PROJECT_ROOT}/graph_ablation" \
    --name-prefix "graph" \
    --epochs "${EPOCHS}" \
    --imgsz "${IMGSZ}" \
    --batch "${BATCH}" \
    --workers "${WORKERS}" \
    --device "${DEVICE}" \
    --pretrained "${PRETRAINED}" \
    "$@"
fi

if [[ "${RUN_PROFILE}" == "1" ]]; then
  python rebuttal_pooling_ablation.py --dry-run --variants ${POOLING_VARIANTS}
  python rebuttal_graph_ablation.py --dry-run --variants ${GRAPH_VARIANTS}
  python rebuttal_profile.py \
    --output "${PROJECT_ROOT}/profile/profile_summary.csv" \
    --device "${DEVICE}" \
    --imgsz "${IMGSZ}" \
    --batch 1 \
    --models \
      "HGDefect=ultralytics/cfg/models/v8/yolov8n-full-innov.yaml" \
      "pooling_gap=ultralytics/cfg/models/v8/rebuttal_generated/yolov8n_pooling_gap.yaml" \
      "pooling_ppap=ultralytics/cfg/models/v8/rebuttal_generated/yolov8n_pooling_ppap.yaml" \
      "graph_none=ultralytics/cfg/models/v8/rebuttal_generated/yolov8n_graph_none.yaml" \
      "graph_kde=ultralytics/cfg/models/v8/rebuttal_generated/yolov8n_graph_kde.yaml"
fi
