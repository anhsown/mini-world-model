"""Metric-gated Eye Physical training on Kaggle T4x2.

G0 learns exact analytic geometry. G1 is admitted only when real TUM RGB-D
roots are provided and pass validation; it mixes real 30-Hz sensor sequences
with the analytic distribution to control sim-to-real forgetting.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import random
import shutil
import sys
import time

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jwm import JWM
from jwm.checkpoint_utils import warmstart_eye_physical
from jwm.configs import eye_physical_scale
from jwm.geometry_data import render_geometry_sequence, validate_geometry_dataset
from jwm.geometry_trainer import evaluate_geometry
from jwm.tum_rgbd import TUMRGBDWindowDataset, validate_tum_dataset


def args():
    p = argparse.ArgumentParser()
    p.add_argument("--output", required=True)
    p.add_argument("--warmstart", default="")
    p.add_argument("--tum-train", nargs="*", default=[])
    p.add_argument("--tum-val", nargs="*", default=[])
    p.add_argument("--per-gpu-batch", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=2)
    p.add_argument("--steps-g0", type=int, default=1800)
    p.add_argument("--steps-g1", type=int, default=3000)
    p.add_argument("--checkpoint-every", type=int, default=250)
    p.add_argument("--log-every", type=int, default=25)
    p.add_argument("--seed", type=int, default=20260720)
    p.add_argument("--quick", action="store_true")
    return p.parse_args()


def distributed():
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local = int(os.environ.get("LOCAL_RANK", "0"))
    device = torch.device("cuda", local) if torch.cuda.is_available() else torch.device("cpu")
    if device.type == "cuda":
        torch.cuda.set_device(local)
    if world > 1:
        dist.init_process_group("nccl" if device.type == "cuda" else "gloo")
    return world, rank, local, device


def atomic_save(payload, path: Path):
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        torch.save(payload, tmp)
        os.replace(tmp, path)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


def stack(rows: list[dict]):
    out = {}
    for key in ("image", "depth", "pose_c2w"):
        out[key] = torch.stack([row[key] for row in rows])
    if all("depth_valid" in row for row in rows):
        out["depth_valid"] = torch.stack([row["depth_valid"] for row in rows])
    return out


def synthetic_batch(seed: int, batch: int, frames=8, size=256):
    return stack([render_geometry_sequence(seed + i, frames, size, size).as_dict()
                  for i in range(batch)])


def move(batch, device):
    return {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v
            for k, v in batch.items()}


def set_eye_trainable(model):
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for module in (model.vision_stem, model.geometry):
        for parameter in module.parameters():
            parameter.requires_grad_(True)
    model.e_img.requires_grad_(True)


def main():
    a = args(); world, rank, local, device = distributed()
    torch.set_float32_matmul_precision("high")
    random.seed(a.seed + rank); torch.manual_seed(a.seed + rank)
    out = Path(a.output)
    if rank == 0:
        out.mkdir(parents=True, exist_ok=True)
        validation = validate_geometry_dataset(
            output=out / "geometry_data_validation.json")
        if not validation["valid"]:
            raise RuntimeError("analytic geometry data failed admission gates")
    if world > 1: dist.barrier()

    cfg = eye_physical_scale()
    model = JWM(cfg)
    warm_report = None
    if a.warmstart:
        warm_report = warmstart_eye_physical(model, a.warmstart)
    set_eye_trainable(model)
    model.to(device)
    trainable = [p for p in model.parameters() if p.requires_grad]
    if rank == 0:
        print(f"devices={world} total={sum(p.numel() for p in model.parameters())/1e6:.2f}M "
              f"trainable={sum(p.numel() for p in trainable)/1e6:.2f}M", flush=True)
        if warm_report:
            (out / "warmstart_report.json").write_text(
                json.dumps(warm_report, indent=2), encoding="utf-8")
    wrapped = DDP(model, device_ids=[local], find_unused_parameters=False) if world > 1 else model
    raw = wrapped.module if hasattr(wrapped, "module") else wrapped
    scaler = torch.amp.GradScaler("cuda", init_scale=1024.0,
                                  enabled=device.type == "cuda")

    tum_train = TUMRGBDWindowDataset(a.tum_train, frames=8, frame_stride=3,
                                     window_stride=12) if a.tum_train else None
    tum_val = TUMRGBDWindowDataset(a.tum_val, frames=8, frame_stride=3,
                                   window_stride=24) if a.tum_val else None
    if rank == 0 and tum_train is not None:
        tum_report = validate_tum_dataset(tum_train)
        (out / "tum_data_validation.json").write_text(
            json.dumps(tum_report, indent=2), encoding="utf-8")
        if not tum_report["valid"]:
            raise RuntimeError("TUM data failed admission gates")
    if world > 1: dist.barrier()

    stages = [("g0_exact_geometry", a.steps_g0, 3e-4, 0.0)]
    if tum_train is not None and len(tum_train):
        stages.append(("g1_real_rgbd_adapt", a.steps_g1, 1.2e-4, 0.35))
    if a.quick:
        stages = [(name, min(3, steps), lr, real) for name, steps, lr, real in stages]
    history, global_step = [], 0
    optimizer = torch.optim.AdamW(trainable, lr=stages[0][2],
                                  betas=(0.9, 0.95), weight_decay=0.05)

    for stage_index, (name, steps, lr, real_probability) in enumerate(stages):
        for group in optimizer.param_groups: group["lr"] = lr
        if rank == 0: print(f"\n=== {name}: {steps} steps lr={lr} real={real_probability} ===", flush=True)
        start = time.time(); optimizer.zero_grad(set_to_none=True)
        for step in range(steps):
            choose_real = (tum_train is not None and
                           random.random() < real_probability)
            if choose_real:
                rows = [tum_train[(global_step * world * a.per_gpu_batch +
                                   rank * a.per_gpu_batch + i) % len(tum_train)]
                        for i in range(a.per_gpu_batch)]
                batch = stack(rows)
            else:
                seed = a.seed + 10_000_000 * stage_index + \
                    global_step * world * a.per_gpu_batch + rank * a.per_gpu_batch
                batch = synthetic_batch(seed, a.per_gpu_batch)
            batch = move(batch, device)
            with torch.autocast(device_type=device.type, dtype=torch.float16,
                                enabled=device.type == "cuda"):
                loss, metrics = wrapped("geometry", batch["image"], batch["depth"],
                                        batch["pose_c2w"], batch.get("depth_valid"))
                scaled_loss = loss / a.grad_accum
            scaler.scale(scaled_loss).backward()
            if (step + 1) % a.grad_accum == 0:
                scaler.unscale_(optimizer)
                grad = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)
            else:
                grad = torch.tensor(float("nan"))
            global_step += 1
            if rank == 0 and (step == 0 or (step + 1) % a.log_every == 0):
                rate = (step + 1) / max(1e-6, time.time() - start)
                print(f"[{step+1:5d}/{steps}] loss={float(loss.detach()):.4f} "
                      f"depth={metrics['geometry_depth']:.4f} "
                      f"rot={metrics['geometry_abs_rotation_rad']:.4f} "
                      f"rel={metrics['geometry_relative_pose']:.4f} "
                      f"grad={float(grad):.3f} {rate:.2f} step/s", flush=True)
            if rank == 0 and global_step % a.checkpoint_every == 0:
                atomic_save({"version": "jwm-eye-physical-v1", "cfg": asdict(cfg),
                             "model": raw.state_dict(), "optimizer": optimizer.state_dict(),
                             "stage_index": stage_index, "step": step + 1,
                             "global_step": global_step, "history": history},
                            out / "resume.pt")

        if world > 1: dist.barrier()
        if rank == 0:
            if name.startswith("g1") and tum_val is not None and len(tum_val):
                eval_batches = (stack([tum_val[i]]) for i in range(min(16, len(tum_val))))
            else:
                eval_batches = (synthetic_batch(3_000_000 + i, 1) for i in range(12))
            evaluation = evaluate_geometry(raw, eval_batches, device, 16)
            depth_gate = 0.35 if stage_index else 0.25
            ate_gate = 0.35 if stage_index else 0.25
            gate = (evaluation.get("depth_abs_rel", float("inf")) < depth_gate
                    and evaluation.get("ate_anchor_scale", float("inf")) < ate_gate
                    and evaluation.get("rpe_rotation_deg", float("inf")) < 15.0)
            result = {"stage": name, "steps": steps, "evaluation": evaluation,
                      "gate_passed": gate, "seconds": time.time() - start}
            history.append(result)
            print(json.dumps(result, indent=2), flush=True)
            atomic_save({"version": "jwm-eye-physical-v1", "cfg": asdict(cfg),
                         "model": raw.state_dict(), "history": history,
                         "status": "stage_passed" if gate else "blocked_by_metric_gate"},
                        out / f"stage_{stage_index}_{name}.pt")
            (out / "metrics.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        else:
            gate = False
        gate_tensor = torch.tensor([int(gate)], device=device)
        if world > 1: dist.broadcast(gate_tensor, 0)
        if not bool(gate_tensor.item()):
            if rank == 0: print("metric gate failed; stopping without promotion", flush=True)
            break
        if world > 1: dist.barrier()

    if rank == 0:
        atomic_save({"version": "jwm-eye-physical-v1", "cfg": asdict(cfg),
                     "model": raw.state_dict(), "history": history},
                    out / "jwm_eye_physical_v1.pt")
        print(f"saved {out / 'jwm_eye_physical_v1.pt'}", flush=True)
    if world > 1:
        dist.barrier(); dist.destroy_process_group()


if __name__ == "__main__":
    main()
