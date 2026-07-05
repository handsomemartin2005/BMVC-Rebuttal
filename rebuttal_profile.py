# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import torch

from ablation import get_device_arg


ROOT = Path(__file__).resolve().parent


def parse_model_item(item: str) -> tuple[str, str]:
    if "=" in item:
        name, value = item.split("=", 1)
        return name.strip(), value.strip()
    path = Path(item)
    return path.stem, item


def try_gflops(model: torch.nn.Module, dummy: torch.Tensor) -> float | str:
    try:
        from thop import profile

        macs, _ = profile(model, inputs=(dummy,), verbose=False)
        return float(macs) * 2.0 / 1e9
    except Exception as exc:
        return f"unavailable: {exc.__class__.__name__}"


def profile_one(model_ref: str, device, imgsz: int, batch: int, warmup: int, iters: int) -> dict:
    from ultralytics import YOLO

    yolo = YOLO(model_ref)
    model = yolo.model.to(device).eval()
    params_m = sum(p.numel() for p in model.parameters()) / 1e6
    dummy = torch.zeros(batch, 3, imgsz, imgsz, device=device)

    if device != "cpu" and torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    with torch.inference_mode():
        for _ in range(warmup):
            model(dummy)
        if device != "cpu" and torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(iters):
            model(dummy)
        if device != "cpu" and torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

    latency_ms = elapsed * 1000.0 / max(iters, 1)
    fps = batch * 1000.0 / latency_ms if latency_ms > 0 else ""
    memory_mb = ""
    if device != "cpu" and torch.cuda.is_available():
        memory_mb = torch.cuda.max_memory_allocated() / (1024**2)

    return {
        "params_m": params_m,
        "gflops": try_gflops(model, dummy),
        "latency_ms": latency_ms,
        "fps": fps,
        "memory_mb": memory_mb,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile YOLO model configs/checkpoints for rebuttal cost tables.")
    parser.add_argument(
        "--models",
        nargs="+",
        default=[f"HGDefect=ultralytics/cfg/models/v8/yolov8n-full-innov.yaml"],
        help="Items can be name=path or path.",
    )
    parser.add_argument("--output", type=Path, default=ROOT / "runs" / "rebuttal_profile" / "profile_summary.csv")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = get_device_arg(args.device)
    rows = []
    for item in args.models:
        name, model_ref = parse_model_item(item)
        print(f"Profiling {name}: {model_ref}")
        result = profile_one(model_ref, device=device, imgsz=args.imgsz, batch=args.batch, warmup=args.warmup, iters=args.iters)
        rows.append({"model": name, "model_ref": model_ref, "device": device, "imgsz": args.imgsz, "batch": args.batch, **result})

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = ["model", "model_ref", "device", "imgsz", "batch", "params_m", "gflops", "latency_ms", "fps", "memory_mb"]
    with args.output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print("Wrote:", args.output)


if __name__ == "__main__":
    main()
