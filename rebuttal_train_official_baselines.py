# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent


CHILD_CODE = r"""
import csv
import json
from pathlib import Path

cfg = json.loads(__import__("sys").argv[1])

from ultralytics import YOLO

model = YOLO(cfg["model"])
train_results = model.train(
    data=cfg["data"],
    epochs=cfg["epochs"],
    imgsz=cfg["imgsz"],
    batch=cfg["batch"],
    workers=cfg["workers"],
    device=cfg["device"],
    project=cfg["project"],
    name=cfg["name"],
    exist_ok=cfg["exist_ok"],
    pretrained=True,
    save=True,
    save_period=cfg["save_period"],
    cache=cfg["cache"],
    patience=cfg["patience"],
    seed=cfg["seed"],
    deterministic=cfg["deterministic"],
    amp=cfg["amp"],
    optimizer=cfg["optimizer"],
    cos_lr=cfg["cos_lr"],
)

metrics = model.val(
    data=cfg["data"],
    imgsz=cfg["imgsz"],
    batch=cfg["batch"],
    workers=cfg["workers"],
    device=cfg["device"],
    project=cfg["project"],
    name=cfg["name"] + "_val",
    plots=False,
)

box = metrics.box
speed = getattr(metrics, "speed", {}) or {}
p = float(getattr(box, "mp", 0.0))
r = float(getattr(box, "mr", 0.0))
f1 = 2 * p * r / (p + r) if p + r > 0 else 0.0
total_ms = float(speed.get("preprocess", 0.0)) + float(speed.get("inference", 0.0)) + float(speed.get("postprocess", 0.0))
fps = 1000.0 / total_ms if total_ms > 0 else 0.0

summary = Path(cfg["summary"])
summary.parent.mkdir(parents=True, exist_ok=True)
fields = ["model", "run_name", "seed", "precision", "recall", "f1", "map50", "map50_95", "fps_estimate"]
exists = summary.exists()
with summary.open("a", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    if not exists:
        writer.writeheader()
    writer.writerow({
        "model": cfg["model"],
        "run_name": cfg["name"],
        "seed": cfg["seed"],
        "precision": p,
        "recall": r,
        "f1": f1,
        "map50": float(getattr(box, "map50", 0.0)),
        "map50_95": float(getattr(box, "map", 0.0)),
        "fps_estimate": fps,
    })
"""


def clean_env() -> dict[str, str]:
    env = os.environ.copy()
    entries = []
    for item in env.get("PYTHONPATH", "").split(os.pathsep):
        if item and Path(item).resolve() != ROOT:
            entries.append(item)
    env["PYTHONPATH"] = os.pathsep.join(entries)
    return env


def run(cmd: list[str], cwd: Path, env: dict[str, str]) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, cwd=str(cwd), env=env, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train official Ultralytics YOLO baselines in an isolated cwd.")
    parser.add_argument("--data", type=Path, default=ROOT / "datasets" / "data.yaml")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["yolov8n.pt", "yolov8s.pt", "yolov10n.pt", "yolov10s.pt", "yolo11n.pt", "yolo11s.pt"],
    )
    parser.add_argument("--project", type=Path, default=ROOT / "runs" / "official_baselines")
    parser.add_argument("--name-prefix", default="baseline")
    parser.add_argument("--summary", type=Path, default=None)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--seed", type=int, default=4601)
    parser.add_argument("--save-period", type=int, default=10)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--cache", action="store_true")
    parser.add_argument("--exist-ok", action="store_true")
    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--optimizer", default="auto")
    parser.add_argument("--cos-lr", action="store_true")
    parser.add_argument("--install-ultralytics", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    if args.runs < 1:
        raise ValueError("--runs must be >= 1")
    if not args.data.exists():
        raise FileNotFoundError(f"Dataset yaml not found: {args.data}")

    summary = args.summary or (args.project / "official_baseline_summary.csv")
    env = clean_env()
    with tempfile.TemporaryDirectory(prefix="official_ultralytics_") as tmp:
        tmp_dir = Path(tmp)
        if args.install_ultralytics:
            run([sys.executable, "-m", "pip", "install", "-U", "ultralytics"], cwd=tmp_dir, env=env)

        for model_idx, model in enumerate(args.models):
            model_tag = Path(model).stem.replace(".", "_")
            for run_idx in range(args.runs):
                name = f"{args.name_prefix}_{model_tag}" if args.runs == 1 else f"{args.name_prefix}_{model_tag}_{run_idx + 1:02d}"
                cfg = {
                    "model": model,
                    "data": str(args.data.resolve()),
                    "project": str(args.project.resolve()),
                    "name": name,
                    "summary": str(summary.resolve()),
                    "epochs": args.epochs,
                    "imgsz": args.imgsz,
                    "batch": args.batch,
                    "workers": args.workers,
                    "device": args.device,
                    "seed": args.seed + model_idx * 100 + run_idx,
                    "save_period": args.save_period,
                    "patience": args.patience if args.patience is not None else args.epochs,
                    "cache": args.cache,
                    "exist_ok": args.exist_ok,
                    "deterministic": args.deterministic,
                    "amp": args.amp,
                    "optimizer": args.optimizer,
                    "cos_lr": args.cos_lr,
                }
                run([sys.executable, "-c", CHILD_CODE, json.dumps(cfg)], cwd=tmp_dir, env=env)

    print("Wrote:", summary)


if __name__ == "__main__":
    main()
