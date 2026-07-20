"""Equal-initialization A/B probe for real-anchored synthetic Eye data.

Run this script twice with the same seed and warm-start.  The real-only arm
uses TUM/Bonn windows; the mixed arm replaces half of the optimizer exposure
with deterministic real-anchored synthetic sequences.  Evaluation is always
performed on the same fixed real validation windows.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import random
import sys
import time

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jwm import JWM
from jwm.anchor_discovery import discover_registered_rgbd
from jwm.checkpoint_utils import warmstart_eye_physical
from jwm.configs import eye_physical_v3_scale
from jwm.geometry_v2_data import BonnRGBDWindowDataset, TaggedTUMDataset
from jwm.geometry_v3_data import (CalibratedGeometryDataset,
                                  make_counterfactuals,
                                  validate_geometry_v3_datasets)
from jwm.geometry_v3_trainer import (evaluate_geometry_v3_controls,
                                     missing_trainable_gradients,
                                     move_geometry_batch,
                                     set_eye_v3_physical_trainable)
from jwm.real_anchored_sdg import (RealAnchorProfile,
                                   RealAnchoredSyntheticGeometry)

from train_eye_v3_ddp import (V3Sampler, atomic_save, depth_prior,
                              fixed_eval_batches, distributed)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=("real-only", "real-plus-synthetic"),
                        required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmstart", type=Path, required=True)
    parser.add_argument("--anchor-root", type=Path,
                        default=Path("data/real_anchor_v1/raw"))
    parser.add_argument("--registry", type=Path,
                        default=Path("configs/datasets/real_anchor_v1.json"))
    parser.add_argument("--synthetic-profile", type=Path,
                        default=Path("data/real_anchor_v1/derived/eye_real_anchor_profile_v1.json"))
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--per-gpu-batch", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--lr", type=float, default=8e-5)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument("--eval-windows", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args()


def make_real_sources(args: argparse.Namespace):
    roots = discover_registered_rgbd(args.anchor_root, args.registry)
    size = 64 if args.quick else 256
    result = {"train": {}, "validation": {}}
    for split in result:
        if roots[split]["tum"]:
            result[split]["tum"] = CalibratedGeometryDataset(TaggedTUMDataset(
                roots[split]["tum"], source="tum", frames=6, frame_stride=2,
                window_stride=12, height=size, width=size), default_fps=30)
        if roots[split]["bonn"]:
            result[split]["bonn"] = CalibratedGeometryDataset(BonnRGBDWindowDataset(
                roots[split]["bonn"], frames=6, frame_stride=2,
                window_stride=12, height=size, width=size), default_fps=30)
    if set(result["train"]) != {"tum", "bonn"}:
        raise RuntimeError("A/B probe requires both TUM and Bonn train roots")
    if set(result["validation"]) != {"tum", "bonn"}:
        raise RuntimeError("A/B probe requires both TUM and Bonn validation roots")
    return result


def main() -> None:
    args = arguments()
    world, rank, local, device = distributed()
    torch.set_float32_matmul_precision("high")
    random.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed + rank)

    args.output.mkdir(parents=True, exist_ok=True)
    real = make_real_sources(args)
    train_sources = dict(real["train"])
    mixture = {"tum": .70, "bonn": .30}
    if args.arm == "real-plus-synthetic":
        profile = RealAnchorProfile.from_json(args.synthetic_profile)
        size = 64 if args.quick else 256
        train_sources["synthetic_real_anchored"] = RealAnchoredSyntheticGeometry(
            "train", 250_000, profile, frames=6, size=size)
        mixture = {"tum": .35, "bonn": .15,
                   "synthetic_real_anchored": .50}

    if rank == 0:
        validation = validate_geometry_v3_datasets(
            {"train": train_sources, "validation": real["validation"]},
            args.output / "dataset_validation.json")
        if not validation["valid"]:
            raise RuntimeError(f"A/B data admission failed: {validation['failures']}")
    if world > 1:
        dist.barrier()

    cfg = eye_physical_v3_scale()
    if args.quick:
        cfg.image_size = 64
        cfg.geometry_v3_width = 32
        cfg.geometry_track_points = 12
        cfg.geometry_track_iterations = 1
        cfg.geometry_ba_iterations = 1
    model = JWM(cfg)
    warm_report = warmstart_eye_physical(model, args.warmstart)
    active = set_eye_v3_physical_trainable(model)
    model.to(device)
    trainable = [parameter for parameter in model.parameters()
                 if parameter.requires_grad]
    wrapped = (DDP(model, device_ids=[local] if device.type == "cuda" else None,
                   broadcast_buffers=False, find_unused_parameters=False)
               if world > 1 else model)
    raw = wrapped.module if hasattr(wrapped, "module") else wrapped
    sampler = V3Sampler(train_sources, world, rank, args.seed)
    eval_batches = fixed_eval_batches(real["validation"], 8)
    prior = depth_prior(real["train"])
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, betas=(.9, .95),
                                  weight_decay=.05)
    scaler = torch.amp.GradScaler("cuda", init_scale=256,
                                  enabled=device.type == "cuda")
    steps = 4 if args.quick else args.steps
    resume_path = args.output / "resume.pt"
    start_step = 0
    started = time.time()
    nonfinite_skips = 0
    history = []
    if resume_path.exists():
        state = torch.load(resume_path, map_location=device, weights_only=False)
        if state.get("arm") != args.arm or int(state.get("seed", -1)) != args.seed:
            raise RuntimeError("A/B resume belongs to a different arm or seed")
        raw.load_state_dict(state["model"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        start_step = int(state["step"])
        history = state.get("history", [])
        nonfinite_skips = int(state.get("nonfinite_skips", 0))
        if rank == 0:
            print(f"resume {args.arm} at optimizer step {start_step}", flush=True)
    for step in range(start_step, steps):
        warm = min(100, max(10, steps // 10))
        lr = args.lr * min(1.0, (step + 1) / warm)
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        accumulated = {}
        finite = True
        for micro in range(args.grad_accum):
            batch = move_geometry_batch(sampler.batch(
                step + nonfinite_skips, micro, args.per_gpu_batch, mixture), device)
            wrong_k = make_counterfactuals(batch)["wrong_intrinsics"]
            with torch.autocast(device_type=device.type, dtype=torch.float16,
                                enabled=device.type == "cuda"):
                loss, metrics = wrapped(
                    "geometry", batch["image"], batch["depth"], batch["pose_c2w"],
                    batch["depth_valid"], batch["dynamic_mask"], None,
                    batch["intrinsics"], batch["projection_y_sign"],
                    batch["rigid_flow"], batch["rigid_flow_valid"], wrong_k)
                finite = finite and bool(torch.isfinite(loss.detach()))
                micro_loss = torch.nan_to_num(loss, nan=0.0, posinf=0.0,
                                              neginf=0.0) / args.grad_accum
            scaler.scale(micro_loss).backward()
            for key, value in metrics.items():
                accumulated[key] = accumulated.get(key, 0.0) + value / args.grad_accum
        if step == 0:
            missing = missing_trainable_gradients(raw)
            if missing:
                raise RuntimeError("Disconnected trainable tensors: " + ", ".join(missing))
        scaler.unscale_(optimizer)
        healthy = finite and all(parameter.grad is None or
                                 bool(torch.isfinite(parameter.grad).all())
                                 for parameter in trainable)
        healthy_tensor = torch.tensor(int(healthy), device=device)
        if world > 1:
            dist.all_reduce(healthy_tensor, op=dist.ReduceOp.MIN)
        if not bool(healthy_tensor):
            optimizer.zero_grad(set_to_none=True)
            scaler.update(new_scale=max(float(scaler.get_scale()) / 2, 1.0))
            nonfinite_skips += 1
            if nonfinite_skips >= 3:
                raise RuntimeError("A/B arm blocked after three non-finite steps")
            continue
        gradient = torch.nn.utils.clip_grad_norm_(trainable, 1.0,
                                                  error_if_nonfinite=True)
        scaler.step(optimizer)
        scaler.update()
        record = {"step": step + 1, "loss": accumulated["loss"],
                  "grad_norm": float(gradient), "lr": lr}
        history.append(record)
        if rank == 0 and (step == 0 or (step + 1) % args.log_every == 0):
            rate = (step + 1) / max(time.time() - started, 1e-6)
            print(f"[{args.arm} {step + 1:4d}/{steps}] "
                  f"loss={record['loss']:.4f} grad={record['grad_norm']:.3f} "
                  f"lr={lr:.2e} {rate:.2f} opt-step/s", flush=True)
        if (step + 1) % args.checkpoint_every == 0:
            if rank == 0:
                atomic_save({"arm": args.arm, "seed": args.seed,
                             "step": step + 1, "model": raw.state_dict(),
                             "optimizer": optimizer.state_dict(),
                             "scaler": scaler.state_dict(), "history": history,
                             "nonfinite_skips": nonfinite_skips}, resume_path)
            if world > 1:
                dist.barrier()

    if world > 1:
        dist.barrier()
    if rank == 0:
        report = evaluate_geometry_v3_controls(
            raw, eval_batches, device, args.eval_windows, prior)
        payload = {
            "arm": args.arm,
            "seed": args.seed,
            "optimizer_steps": len(history),
            "global_batch": world * args.per_gpu_batch * args.grad_accum,
            "mixture": mixture,
            "warmstart": warm_report,
            "report": report,
            "history_tail": history[-20:],
            "nonfinite_skips": nonfinite_skips,
        }
        (args.output / "probe_metrics.json").write_text(
            json.dumps(payload, indent=2), encoding="utf-8")
        atomic_save({"cfg": asdict(cfg), "model": raw.state_dict(), **payload},
                    args.output / "probe_checkpoint.pt")
        resume_path.unlink(missing_ok=True)
        print(json.dumps({"arm": args.arm, "report": report}, indent=2),
              flush=True)
    if world > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
