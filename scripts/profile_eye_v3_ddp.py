"""Pre-flight memory/throughput profile for the exact Eye-v3 DDP graph."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

from jwm import JWM
from jwm.checkpoint_utils import warmstart_eye_physical
from jwm.configs import eye_physical_v3_scale
from jwm.geometry_v3_data import make_counterfactuals, procedural_v3_row, stack_geometry_v3_rows
from jwm.geometry_v3_trainer import (
    missing_trainable_gradients, move_geometry_batch,
    set_eye_v3_physical_trainable,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--warmstart", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--per-gpu-batch", type=int, default=1)
    p.add_argument("--tiny", action="store_true")
    args = p.parse_args()
    world, rank, local = (int(os.environ.get("WORLD_SIZE", "1")),
                          int(os.environ.get("RANK", "0")),
                          int(os.environ.get("LOCAL_RANK", "0")))
    device = torch.device("cuda", local) if torch.cuda.is_available() else torch.device("cpu")
    if device.type == "cuda": torch.cuda.set_device(local)
    if world > 1:
        dist.init_process_group("nccl" if device.type == "cuda" else "gloo",
                                device_id=device if device.type == "cuda" else None)
    torch.manual_seed(20260723 + rank)
    cfg = eye_physical_v3_scale()
    if args.tiny:
        cfg.image_size = 64; cfg.geometry_v3_width = 32
        cfg.geometry_track_points = 12; cfg.geometry_track_iterations = 1
        cfg.geometry_ba_iterations = 1
    model = JWM(cfg); warmstart_eye_physical(model, args.warmstart)
    active_names = set_eye_v3_physical_trainable(model)
    model.to(device)
    trainable = [p for p in model.parameters() if p.requires_grad]
    wrapped = (DDP(model, device_ids=[local] if device.type == "cuda" else None,
                   broadcast_buffers=False) if world > 1 else model)
    optimizer = torch.optim.AdamW(trainable, lr=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    size = cfg.image_size
    cached = [stack_geometry_v3_rows([
        procedural_v3_row(40_000_000 + rank * 100 + i * args.per_gpu_batch + j,
                          6, size) for j in range(args.per_gpu_batch)])
              for i in range(3)]
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device); torch.cuda.synchronize(device)
    started = time.time(); losses = []
    for step in range(args.steps):
        batch = move_geometry_batch(cached[step % len(cached)], device)
        wrong = make_counterfactuals(batch)["wrong_intrinsics"] if step % 2 else None
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16,
                            enabled=device.type == "cuda"):
            loss, _ = wrapped("geometry", batch["image"], batch["depth"],
                              batch["pose_c2w"], batch["depth_valid"],
                              batch["dynamic_mask"], None, batch["intrinsics"],
                              batch["projection_y_sign"], batch["rigid_flow"],
                              batch["rigid_flow_valid"], wrong)
        scaler.scale(loss).backward(); scaler.unscale_(optimizer)
        if step == 0:
            raw = wrapped.module if hasattr(wrapped, "module") else wrapped
            disconnected = missing_trainable_gradients(raw)
            if disconnected:
                raise RuntimeError(
                    "Eye-v3 exact graph has trainable parameters without gradients: "
                    + ", ".join(disconnected))
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        scaler.step(optimizer); scaler.update(); losses.append(float(loss.detach()))
        if rank == 0 and (step == 0 or (step + 1) % 20 == 0):
            print(f"profile [{step+1}/{args.steps}] loss={losses[-1]:.4f}", flush=True)
    if device.type == "cuda": torch.cuda.synchronize(device)
    seconds = time.time() - started
    peak = (torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0)
    total = (torch.cuda.get_device_properties(device).total_memory if device.type == "cuda" else 0)
    tensor = torch.tensor([seconds, float(peak), float(total)], device=device)
    if world > 1: dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    if rank == 0:
        seconds, peak, total = tensor.tolist()
        step_rate = args.steps / max(seconds, 1e-6)
        report = {"valid": bool(all(torch.isfinite(torch.tensor(losses))) and
                                (not total or peak / total < .88)),
                  "devices": world, "tiny": args.tiny, "steps": args.steps,
                  "per_gpu_batch": args.per_gpu_batch,
                  "active_parameter_tensors": len(active_names),
                  "optimizer_steps_per_second": step_rate,
                  "seconds_per_step": 1 / step_rate,
                  "peak_memory_gib_per_rank": peak / 2**30,
                  "memory_utilization": peak / total if total else None,
                  "estimated_hours_7000_steps": 7000 / step_rate / 3600,
                  "loss_first": losses[0], "loss_last": losses[-1]}
        path = Path(args.output); path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2), flush=True)
        if not report["valid"]: raise SystemExit(2)
    if world > 1:
        dist.barrier(); dist.destroy_process_group()


if __name__ == "__main__": main()
