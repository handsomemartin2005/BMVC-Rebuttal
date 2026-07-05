# BMVC Rebuttal Experiment Tables

No result in this file is fabricated. Fill `TBD` only from completed runs or verified public reports with matching protocols.

## P0 Queue

| Priority | Experiment | Reviewer concern | Script | Default dataset |
| --- | --- | --- | --- | --- |
| P0 | GC10-DET quantitative evaluation for completed HGDefect checkpoint | R2/R3: GC10 has no quantitative result | `scripts/autodl_eval_checkpoint.sh` | GC10-DET |
| P0 | YOLOv8/YOLOv10/YOLOv11 nano/small baselines | R2/R4: fair complexity and recent YOLO baselines | `scripts/autodl_train_official_baselines.sh` | NEU-DET or GC10-DET |
| P0 | PPAP vs GAP/GMP/GAP+GMP/Top1 | R4: pooling mechanism comparison | `rebuttal_pooling_ablation.py` | NEU-DET |
| P0 | KDE vs no graph/kNN/fixed/cosine/random | R4: graph construction comparison | `rebuttal_graph_ablation.py` | NEU-DET |
| P1 | Params/GFLOPs/FPS/memory | R1/R3: compute and latency breakdown | `rebuttal_profile.py` | none |

## GC10-DET Main Table

| Method | Params | GFLOPs | FPS | P | R | F1 | mAP50 | mAP50:95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| YOLOv8-n | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| YOLOv8-s | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| YOLOv10-n | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| YOLOv10-s | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| YOLOv11-n | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| YOLOv11-s | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| HGDefect-YOLO | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

## GC10-DET Class AP50

| Method | Pu | Wl | Cg | Ws | Os | Ss | Cr | Rp | In | Wf | mAP50 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| YOLOv8-n | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| YOLOv11-n | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| HGDefect-YOLO | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

## Similar-Complexity YOLO Comparison

| Method | Scale | Params | GFLOPs | FPS | mAP50 | mAP50:95 | Cr AP |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| YOLOv8-n | nano | TBD | TBD | TBD | TBD | TBD | TBD |
| YOLOv8-s | small | TBD | TBD | TBD | TBD | TBD | TBD |
| YOLOv10-n | nano | TBD | TBD | TBD | TBD | TBD | TBD |
| YOLOv10-s | small | TBD | TBD | TBD | TBD | TBD | TBD |
| YOLOv11-n | nano | TBD | TBD | TBD | TBD | TBD | TBD |
| YOLOv11-s | small | TBD | TBD | TBD | TBD | TBD | TBD |
| HGDefect-YOLO | ours | TBD | 14.85 | 51 | TBD | TBD | TBD |

## PPAP Pooling Ablation

| Variant | Pooling inside EMA | Params | GFLOPs | FPS | mAP50 | mAP50:95 | Cr AP |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline EMA | GAP | TBD | TBD | TBD | TBD | TBD | TBD |
| GMP-EMA | GMP | TBD | TBD | TBD | TBD | TBD | TBD |
| GAP+GMP-EMA | GAP+GMP | TBD | TBD | TBD | TBD | TBD | TBD |
| Top-1 EMA | max peak | TBD | TBD | TBD | TBD | TBD | TBD |
| PPAP-EMA | top-k peak average | TBD | TBD | TBD | TBD | TBD | TBD |

## Graph Construction Ablation

| Graph construction | Params | GFLOPs | FPS | mAP50 | mAP50:95 | Cr AP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| No graph | TBD | TBD | TBD | TBD | TBD | TBD |
| Random hyperedge | TBD | TBD | TBD | TBD | TBD | TBD |
| kNN graph | TBD | TBD | TBD | TBD | TBD | TBD |
| Fixed threshold / epsilon-ball | TBD | TBD | TBD | TBD | TBD | TBD |
| Cosine threshold | TBD | TBD | TBD | TBD | TBD | TBD |
| KDE hypergraph | TBD | TBD | TBD | TBD | TBD | TBD |

## Module Cost Breakdown

| Variant | Params | GFLOPs | FPS | Memory | Time/epoch | mAP50 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline | TBD | TBD | TBD | TBD | TBD | TBD |
| +PPAP-EMA | TBD | TBD | TBD | TBD | TBD | TBD |
| +KDE-Hypergraph | TBD | TBD | TBD | TBD | TBD | TBD |
| +RGCU-CLAG | TBD | TBD | TBD | TBD | TBD | TBD |
| Full | TBD | 14.85 | 51 | TBD | TBD | TBD |
