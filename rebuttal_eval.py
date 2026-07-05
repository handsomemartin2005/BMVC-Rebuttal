# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import numpy as np

from ablation import get_device_arg


ROOT = Path(__file__).resolve().parent


def to_float(value: Any) -> float | str:
    try:
        return float(value)
    except Exception:
        return ""


def get_names(metrics) -> dict[int, str]:
    names = getattr(metrics, "names", None)
    if isinstance(names, dict):
        return {int(k): str(v) for k, v in names.items()}
    if isinstance(names, list):
        return {i: str(v) for i, v in enumerate(names)}
    return {}


def write_summary(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "weights",
        "data",
        "precision",
        "recall",
        "f1",
        "map50",
        "map50_95",
        "preprocess_ms",
        "inference_ms",
        "postprocess_ms",
        "fps_estimate",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerow(row)


def write_class_ap(path: Path, metrics) -> None:
    names = get_names(metrics)
    box = metrics.box
    ap50 = np.asarray(getattr(box, "ap50", []), dtype=float)
    maps = np.asarray(getattr(box, "maps", []), dtype=float)
    nc = max(len(names), len(ap50), len(maps))

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["class_id", "class_name", "ap50", "map50_95"])
        writer.writeheader()
        for idx in range(nc):
            writer.writerow(
                {
                    "class_id": idx,
                    "class_name": names.get(idx, str(idx)),
                    "ap50": to_float(ap50[idx]) if idx < len(ap50) else "",
                    "map50_95": to_float(maps[idx]) if idx < len(maps) else "",
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a YOLO checkpoint and export rebuttal metrics tables.")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--project", type=Path, default=ROOT / "runs" / "rebuttal_eval")
    parser.add_argument("--name", default="eval")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--split", default="val")
    parser.add_argument("--plots", action="store_true")
    parser.add_argument("--save-json", action="store_true")
    args = parser.parse_args()

    from ultralytics import YOLO

    model = YOLO(str(args.weights))
    metrics = model.val(
        data=str(args.data),
        split=args.split,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=get_device_arg(args.device),
        project=str(args.project),
        name=args.name,
        plots=args.plots,
        save_json=args.save_json,
    )

    box = metrics.box
    precision = to_float(getattr(box, "mp", ""))
    recall = to_float(getattr(box, "mr", ""))
    if isinstance(precision, float) and isinstance(recall, float) and precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = ""

    speed = getattr(metrics, "speed", {}) or {}
    preprocess_ms = to_float(speed.get("preprocess", ""))
    inference_ms = to_float(speed.get("inference", ""))
    postprocess_ms = to_float(speed.get("postprocess", ""))
    if all(isinstance(v, float) for v in [preprocess_ms, inference_ms, postprocess_ms]):
        total_ms = preprocess_ms + inference_ms + postprocess_ms
        fps = 1000.0 / total_ms if total_ms > 0 else ""
    else:
        fps = ""

    out_dir = args.project / args.name
    write_summary(
        out_dir / "summary_metrics.csv",
        {
            "weights": str(args.weights),
            "data": str(args.data),
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "map50": to_float(getattr(box, "map50", "")),
            "map50_95": to_float(getattr(box, "map", "")),
            "preprocess_ms": preprocess_ms,
            "inference_ms": inference_ms,
            "postprocess_ms": postprocess_ms,
            "fps_estimate": fps,
        },
    )
    write_class_ap(out_dir / "class_ap.csv", metrics)

    print("Wrote:", out_dir / "summary_metrics.csv")
    print("Wrote:", out_dir / "class_ap.csv")


if __name__ == "__main__":
    main()
