# BMVC Rebuttal Experiments for HGDefect-YOLO

This repository contains the runnable code for the BMVC rebuttal experiment queue. Datasets and checkpoints are intentionally not committed. Put datasets on AutoDL and pass their `data.yaml` paths to the scripts.

## What To Run

The GC10-DET run for HGDefect-YOLO is not included in the default training queue because it has already been completed. Use `rebuttal_eval.py` or `scripts/autodl_eval_checkpoint.sh` to export metrics from that completed checkpoint.

Primary rebuttal experiments:

| Concern | Script |
| --- | --- |
| YOLOv8/YOLOv10/YOLOv11 nano/small baselines | `scripts/autodl_train_official_baselines.sh` |
| PPAP vs GAP/GMP/GAP+GMP/Top1 pooling | `rebuttal_pooling_ablation.py` or `scripts/autodl_rebuttal_core.sh` |
| KDE hypergraph vs no graph/kNN/fixed/random/cosine | `rebuttal_graph_ablation.py` or `scripts/autodl_rebuttal_core.sh` |
| Params/GFLOPs/FPS/memory profile | `rebuttal_profile.py` |
| Completed checkpoint eval and class AP table | `rebuttal_eval.py` |

Table templates are in `experiments/rebuttal_tables.md`.

## AutoDL Setup

Clone the repository, then install dependencies from the repository root:

```bash
git clone https://github.com/handsomemartin2005/BMVC-Rebuttal.git
cd BMVC-Rebuttal

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

Recommended dataset layout on AutoDL:

```text
/root/autodl-tmp/datasets/
  NEU-DET/
    images/train
    images/val
    labels/train
    labels/val
  GC10-DET/
    images/train
    images/val
    labels/train
    labels/val
```

Portable dataset yaml templates are provided:

```text
configs/datasets/neu-det.yaml
configs/datasets/gc10-det.yaml
```

Edit their `path:` fields if your AutoDL dataset location differs.

## Run Core Rebuttal Ablations

This runs the reviewer-requested PPAP pooling and graph construction ablations on NEU-DET. By default it runs all pooling variants and the minimal graph set: no graph, kNN, fixed threshold, KDE.

```bash
DATA=/root/autodl-tmp/datasets/NEU-DET/data.yaml \
PROJECT_ROOT=/root/autodl-tmp/bmvc_rebuttal \
EPOCHS=500 BATCH=32 DEVICE=0 \
bash scripts/autodl_rebuttal_core.sh
```

Useful overrides:

```bash
# Run every graph construction variant.
GRAPH_VARIANTS="all" bash scripts/autodl_rebuttal_core.sh

# Also rerun the A/B/C module ablation if needed.
RUN_MODULE_ABLATION=1 bash scripts/autodl_rebuttal_core.sh

# Smoke test only.
EPOCHS=1 BATCH=2 DEVICE=cpu RUN_PROFILE=0 bash scripts/autodl_rebuttal_core.sh
```

## Run Official YOLO Baselines

This script uses a temporary working directory so it imports the official pip-installed Ultralytics package instead of the local modified package. This is needed for YOLOv11.

```bash
DATA=/root/autodl-tmp/datasets/NEU-DET/data.yaml \
PROJECT=/root/autodl-tmp/bmvc_rebuttal/official_baselines_neu \
EPOCHS=500 BATCH=32 DEVICE=0 \
bash scripts/autodl_train_official_baselines.sh
```

For GC10-DET baselines:

```bash
DATA=/root/autodl-tmp/datasets/GC10-DET/data.yaml \
PROJECT=/root/autodl-tmp/bmvc_rebuttal/official_baselines_gc10 \
EPOCHS=300 BATCH=16 DEVICE=0 \
bash scripts/autodl_train_official_baselines.sh
```

To restrict models:

```bash
MODELS="yolov8n.pt yolo11n.pt" bash scripts/autodl_train_official_baselines.sh
```

## Evaluate Completed HGDefect Checkpoints

Use this for the already completed GC10 HGDefect-YOLO run:

```bash
WEIGHTS=/root/autodl-tmp/path/to/best.pt \
DATA=/root/autodl-tmp/datasets/GC10-DET/data.yaml \
PROJECT=/root/autodl-tmp/bmvc_rebuttal/gc10_hgdefect_eval \
NAME=hgdefect_gc10 \
bash scripts/autodl_eval_checkpoint.sh
```

Outputs:

```text
summary_metrics.csv
class_ap.csv
```

## Profile Models

Generate rebuttal YAMLs first, then profile:

```bash
python rebuttal_pooling_ablation.py --dry-run
python rebuttal_graph_ablation.py --dry-run --variants minimal

python rebuttal_profile.py \
  --output /root/autodl-tmp/bmvc_rebuttal/profile/profile_summary.csv \
  --device 0 \
  --models \
    HGDefect=ultralytics/cfg/models/v8/yolov8n-full-innov.yaml \
    graph_none=ultralytics/cfg/models/v8/rebuttal_generated/yolov8n_graph_none.yaml \
    graph_kde=ultralytics/cfg/models/v8/rebuttal_generated/yolov8n_graph_kde.yaml \
    pooling_gap=ultralytics/cfg/models/v8/rebuttal_generated/yolov8n_pooling_gap.yaml \
    pooling_ppap=ultralytics/cfg/models/v8/rebuttal_generated/yolov8n_pooling_ppap.yaml
```

## Git Hygiene

Ignored by design:

```text
datasets/
GC10dataset/
GC10dataset_9_1/
runs/
*.pt
*.zip
```

Keep dataset yaml templates under `configs/datasets/` and write training outputs under `/root/autodl-tmp`.
