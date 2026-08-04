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
from jwm.checkpoint_utils import (
    warmstart_eye_physical, warmstart_eye_v32, warmstart_eye_v322,
)
from jwm.configs import (eye_physical_v3_scale, eye_physical_v32_scale,
                         eye_physical_v32_smoke_scale)
from jwm.geometry_v3_data import make_counterfactuals, procedural_v3_row, stack_geometry_v3_rows
from jwm.geometry_v3_trainer import (
    missing_trainable_gradients, move_geometry_batch,
    set_eye_v3_physical_trainable,
)
from scripts.train_eye_v3_ddp import (
    STAGES, V32_ARCHITECTURES, V3Sampler, apply_stage_weight_profile,
    make_datasets,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--warmstart", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--architecture", choices=("v31", "v32", "v321", "v322"),
                   default="v31")
    p.add_argument("--steps", type=int, default=250)
    p.add_argument("--per-gpu-batch", type=int, default=1)
    p.add_argument("--tiny", action="store_true")
    for source in ("tartan", "tum", "bonn"):
        for split in ("train", "val", "test"):
            p.add_argument(f"--{source}-{split}", nargs="*", default=[])
    p.add_argument("--anchor-root", type=Path, default=Path("data/real_anchor_v1/raw"))
    p.add_argument("--dataset-registry", type=Path,
                   default=Path("configs/datasets/real_anchor_v1.json"))
    p.add_argument("--synthetic-profile", type=Path,
                   default=Path("data/real_anchor_v1/derived/eye_real_anchor_profile_v1.json"))
    p.add_argument("--synthetic-train-samples", type=int, default=250_000)
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
    cfg = (eye_physical_v32_scale() if args.architecture in V32_ARCHITECTURES
           else eye_physical_v3_scale())
    if args.tiny and args.architecture in V32_ARCHITECTURES:
        cfg = eye_physical_v32_smoke_scale()
    elif args.tiny:
        cfg.image_size = 64; cfg.geometry_v3_width = 32
        cfg.geometry_track_points = 12; cfg.geometry_track_iterations = 1
        cfg.geometry_ba_iterations = 1
    model = JWM(cfg)
    (warmstart_eye_v322(model, args.warmstart)
     if args.architecture == "v322" else
     warmstart_eye_v32(model, args.warmstart)
     if args.architecture in V32_ARCHITECTURES else
     warmstart_eye_physical(model, args.warmstart))
    apply_stage_weight_profile(cfg, 0)
    active_names = set_eye_v3_physical_trainable(model)
    model.to(device)
    trainable = [p for p in model.parameters() if p.requires_grad]
    wrapped = (DDP(model, device_ids=[local] if device.type == "cuda" else None,
                   broadcast_buffers=False) if world > 1 else model)
    optimizer = torch.optim.AdamW(trainable, lr=1e-4)
    # Match the full trainer.  The default 65536 scale can overflow fresh
    # convolution gradients and create a false-negative canary on step one.
    scaler = torch.amp.GradScaler("cuda", init_scale=256,
                                  enabled=device.type == "cuda")
    size = cfg.image_size
    sources = make_datasets(args)["train"]
    if any(name in sources for name in ("tartanair", "tum", "bonn")):
        sampler = V3Sampler(sources, world, rank, 20260731)
        mixture = STAGES[0][5]
        cached = [sampler.batch(i, 0, args.per_gpu_batch, mixture) for i in range(24)]
        source_mode = "actual_g0_mixture"
    else:
        cached = [stack_geometry_v3_rows([
            procedural_v3_row(40_000_000 + rank * 100 + i * args.per_gpu_batch + j,
                              6, size) for j in range(args.per_gpu_batch)])
                  for i in range(12)]
        source_mode = "procedural_fallback"
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device); torch.cuda.synchronize(device)
    started = time.time(); losses = []; recent_metrics = []
    for step in range(args.steps):
        batch = move_geometry_batch(cached[step % len(cached)], device)
        controls = make_counterfactuals(batch)
        mode = step % 3
        wrong = controls["wrong_intrinsics"] if mode == 0 else None
        wrong_window = (controls["reverse_image"] if mode == 1 else
                        controls["wrong_window_image"] if mode == 2 else None)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16,
                            enabled=device.type == "cuda"):
            loss, metrics = wrapped("geometry", batch["image"], batch["depth"],
                                    batch["pose_c2w"], batch["depth_valid"],
                                    batch["dynamic_mask"], wrong_window, batch["intrinsics"],
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
        gradient = torch.nn.utils.clip_grad_norm_(trainable, 1.0,
                                                  error_if_nonfinite=False)
        healthy = torch.tensor(int(bool(torch.isfinite(loss.detach())) and
                                   bool(torch.isfinite(gradient))), device=device)
        if world > 1: dist.all_reduce(healthy, op=dist.ReduceOp.MIN)
        if not bool(healthy):
            raise RuntimeError(f"non-finite loss/gradient in stability canary at step {step+1}")
        scaler.step(optimizer); scaler.update(); losses.append(float(loss.detach()))
        recent_metrics.append(metrics)
        if rank == 0 and (step == 0 or (step + 1) % 20 == 0):
            window = recent_metrics[-20:]
            mean = lambda key: sum(row[key] for row in window) / len(window)
            print(f"profile [{step+1}/{args.steps}] loss={sum(losses[-20:])/len(losses[-20:]):.4f} "
                  f"depth={mean('geometry_depth_nll'):.4f} "
                  f"absrel={mean('geometry_depth_absrel_loss'):.3f} "
                  f"edge={mean('geometry_depth_gradient'):.3f} "
                  f"track={mean('geometry_track_epe'):.4f} "
                  f"valid={mean('geometry_track_valid_fraction'):.3f} "
                  f"ba={mean('geometry_ba_reduction'):.3f} "
                  f"conf={mean('geometry_confidence_bce'):.3f} "
                  f"vis={mean('geometry_visibility_bce'):.3f} "
                  f"temp={mean('geometry_temporal_bce'):.3f} "
                  f"grad={float(gradient):.3f}", flush=True)
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
                  "architecture": args.architecture,
                  "source_mode": source_mode,
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
