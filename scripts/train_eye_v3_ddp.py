"""Adaptive T4x2 training for CTPG-Eye v3.

Full training is admitted only after dataset hypotheses and controlled probes
pass. Stage length is decided from held-out causal/OOD metrics under hard
minimum/maximum budgets; training loss alone can never promote a checkpoint.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import random
import shutil
import sys
import time

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jwm import JWM
from jwm.adaptive_training import (
    AdaptiveTrainingBudget, BudgetAction, BudgetConfig, MetricSpec,
    eye_v3_budget_specs,
)
from jwm.checkpoint_utils import warmstart_eye_physical
from jwm.configs import eye_physical_v3_scale
from jwm.geometry_v2_data import (
    BonnRGBDWindowDataset, TaggedTUMDataset, TartanAirWindowDataset,
)
from jwm.geometry_v3_data import (
    CalibratedGeometryDataset, make_counterfactuals, procedural_v3_row,
    stack_geometry_v3_rows, validate_geometry_v3_datasets,
)
from jwm.geometry_v3_trainer import (
    controller_metrics, evaluate_geometry_v3_controls,
    missing_trainable_gradients, move_geometry_batch,
    set_eye_v3_physical_trainable,
)


STAGES = (
    # name, min, max, eval_every, lr, source mixture
    ("g0_calibrated_tracks", 800, 1800, 200, 3e-4,
     {"procedural": .55, "tartanair": .45}),
    ("g1_metric_odometry", 1200, 3200, 300, 1.5e-4,
     {"procedural": .15, "tartanair": .35, "tum": .35, "bonn": .15}),
    ("g2_dynamic_geometry", 800, 2200, 250, 1.0e-4,
     {"procedural": .15, "tartanair": .15, "tum": .20, "bonn": .50}),
    ("g3_causal_ood", 600, 1800, 200, 7e-5,
     {"procedural": .10, "tartanair": .25, "tum": .35, "bonn": .30}),
)


class ProceduralV3Dataset(Dataset):
    OFFSETS = {"train": 10_000_000, "validation": 20_000_000, "test": 30_000_000}
    def __init__(self, split: str, samples: int = 64, frames: int = 6, size: int = 256):
        self.split, self.samples, self.frames, self.size = split, samples, frames, size
    def __len__(self): return self.samples
    def __getitem__(self, index):
        return procedural_v3_row(self.OFFSETS[self.split] + index,
                                 self.frames, self.size)
    def scene_ids(self):
        return {f"procedural-{self.OFFSETS[self.split] + i}" for i in range(self.samples)}


def arguments():
    p = argparse.ArgumentParser()
    p.add_argument("--output", required=True)
    p.add_argument("--warmstart", required=True)
    p.add_argument("--probe-report", default="data/eye_v3_probes/probe_report.json")
    for source in ("tartan", "tum", "bonn"):
        for split in ("train", "val", "test"):
            p.add_argument(f"--{source}-{split}", nargs="*", default=[])
    p.add_argument("--per-gpu-batch", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=2)
    p.add_argument("--checkpoint-every", type=int, default=200)
    p.add_argument("--log-every", type=int, default=25)
    p.add_argument("--eval-windows", type=int, default=12)
    p.add_argument("--seed", type=int, default=20260723)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--allow-partial-data", action="store_true")
    return p.parse_args()


def distributed():
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local = int(os.environ.get("LOCAL_RANK", "0"))
    device = torch.device("cuda", local) if torch.cuda.is_available() else torch.device("cpu")
    if device.type == "cuda": torch.cuda.set_device(local)
    if world > 1:
        dist.init_process_group("nccl" if device.type == "cuda" else "gloo",
                                device_id=device if device.type == "cuda" else None)
    return world, rank, local, device


def atomic_save(payload: dict, path: Path):
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        torch.save(payload, temporary); os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True); raise


def make_datasets(args) -> dict[str, dict[str, Dataset]]:
    result = {split: {"procedural": ProceduralV3Dataset(split)}
              for split in ("train", "validation", "test")}
    mapping = {"train": "train", "validation": "val", "test": "test"}
    for split, flag in mapping.items():
        tartan = getattr(args, f"tartan_{flag}")
        tum = getattr(args, f"tum_{flag}")
        bonn = getattr(args, f"bonn_{flag}")
        if tartan:
            base = TartanAirWindowDataset(tartan, frames=6, frame_stride=2,
                                           window_stride=12, height=256, width=256)
            result[split]["tartanair"] = CalibratedGeometryDataset(base, default_fps=10)
        if tum:
            base = TaggedTUMDataset(tum, frames=6, frame_stride=2,
                                    window_stride=12, height=256, width=256)
            result[split]["tum"] = CalibratedGeometryDataset(base, default_fps=30)
        if bonn:
            base = BonnRGBDWindowDataset(bonn, frames=6, frame_stride=2,
                                         window_stride=12, height=256, width=256)
            result[split]["bonn"] = CalibratedGeometryDataset(base, default_fps=30)
    return result


class V3Sampler:
    def __init__(self, datasets: dict[str, Dataset], world: int, rank: int, seed: int):
        self.datasets, self.world, self.rank, self.seed = datasets, world, rank, seed

    def weights(self, requested: dict[str, float]) -> dict[str, float]:
        admitted = {name: value for name, value in requested.items()
                    if name in self.datasets and len(self.datasets[name])}
        total = sum(admitted.values())
        if total <= 0: raise RuntimeError("no admitted source in stage mixture")
        return {name: value / total for name, value in admitted.items()}

    def batch(self, optimizer_step: int, micro_step: int, batch_size: int,
              requested: dict[str, float]) -> dict:
        weights = self.weights(requested)
        rng = random.Random(self.seed + optimizer_step * 1009 + micro_step * 9176)
        marker, cumulative, source = rng.random(), 0.0, next(iter(weights))
        for name, value in weights.items():
            cumulative += value
            if marker <= cumulative: source = name; break
        rows = []
        dataset = self.datasets[source]
        for item in range(batch_size):
            global_index = ((optimizer_step * self.world + self.rank) * batch_size +
                            item + micro_step * 104729)
            rows.append(dataset[global_index % len(dataset)])
        return stack_geometry_v3_rows(rows)


def fixed_eval_batches(sources: dict[str, Dataset], per_source: int = 3) -> list[dict]:
    batches = []
    for dataset in sources.values():
        count = min(per_source, len(dataset))
        for i in range(count):
            index = min(len(dataset) - 1, math.floor(i * len(dataset) / max(count, 1)))
            batches.append(stack_geometry_v3_rows([dataset[index]]))
    return batches


def depth_prior(sources: dict[str, Dataset]) -> float:
    values = []
    for dataset in sources.values():
        for index in range(min(4, len(dataset))):
            row = dataset[index]
            values.append(row["depth"][row["depth_valid"]][::128])
    return float(torch.cat(values).median()) if values else 3.0


def stage_specs(index: int):
    if index == 0:
        return (MetricSpec("depth_prior_gain", "max", 1.0, 1.05),
                MetricSpec("ba_residual_reduction", "max", 0.0, .10))
    if index == 1:
        return (MetricSpec("depth_prior_gain", "max", 1.0, 1.10),
                MetricSpec("pose_identity_gain", "max", 1.0, 1.10, weight=1.5),
                MetricSpec("ba_residual_reduction", "max", 0.0, .12))
    if index == 2:
        return (MetricSpec("wrong_window_pose_ratio", "max", 1.0, 1.15),
                MetricSpec("reverse_time_rpe_ratio", "max", 1.0, 1.05),
                MetricSpec("wrong_intrinsics_pose_ratio", "max", 1.0, 1.05))
    return eye_v3_budget_specs()


def set_trainable(model: JWM):
    return set_eye_v3_physical_trainable(model)


def main():
    args = arguments(); world, rank, local, device = distributed()
    torch.set_float32_matmul_precision("high")
    random.seed(args.seed + rank); torch.manual_seed(args.seed + rank)
    output = Path(args.output)
    if rank == 0: output.mkdir(parents=True, exist_ok=True)

    probe = json.loads(Path(args.probe_report).read_text(encoding="utf-8"))
    if not probe.get("valid"):
        raise RuntimeError("Eye-v3 controlled probes failed; full training blocked")
    datasets = make_datasets(args)
    if rank == 0:
        admission = validate_geometry_v3_datasets(
            datasets, output / "dataset_validation_v3.json")
        available = set(datasets["train"]) - {"procedural"}
        missing = {"tartanair", "tum", "bonn"} - available
        if missing and not (args.quick or args.allow_partial_data):
            raise RuntimeError(f"missing mandatory train sources: {sorted(missing)}")
        if not admission["valid"]:
            raise RuntimeError(f"dataset hypotheses failed: {admission['failures']}")
        print(json.dumps(admission, indent=2), flush=True)
    if world > 1: dist.barrier()

    cfg = eye_physical_v3_scale()
    if args.quick:
        cfg.image_size = 64; cfg.geometry_v3_width = 32
        cfg.geometry_track_points = 12; cfg.geometry_track_iterations = 1
        cfg.geometry_ba_iterations = 1
    model = JWM(cfg)
    warm_report = warmstart_eye_physical(model, args.warmstart)
    active_names = set_trainable(model); model.to(device)
    trainable = [p for p in model.parameters() if p.requires_grad]
    wrapped = (DDP(model, device_ids=[local] if device.type == "cuda" else None,
                   broadcast_buffers=False, find_unused_parameters=False)
               if world > 1 else model)
    raw = wrapped.module if hasattr(wrapped, "module") else wrapped
    sampler = V3Sampler(datasets["train"], world, rank, args.seed)
    validation_batches = fixed_eval_batches(datasets["validation"], 2)
    prior_m = depth_prior(datasets["train"])
    optimizer = torch.optim.AdamW(trainable, lr=STAGES[0][4], betas=(.9, .95),
                                  weight_decay=.05)
    scaler = torch.amp.GradScaler("cuda", init_scale=256,
                                  enabled=device.type == "cuda")
    history, global_step, start_stage, start_step = [], 0, 0, 0
    resume = output / "resume.pt"
    resume_controller = None
    if resume.exists():
        state = torch.load(resume, map_location=device, weights_only=False)
        if state.get("version") != "jwm-eye-v3-ctpg":
            raise RuntimeError("incompatible Eye-v3 resume checkpoint")
        raw.load_state_dict(state["model"], strict=True)
        optimizer.load_state_dict(state["optimizer"]); scaler.load_state_dict(state["scaler"])
        history = state.get("history", []); global_step = int(state["global_step"])
        start_stage, start_step = int(state["stage_index"]), int(state["stage_step"])
        resume_controller = state.get("controller")
    if rank == 0:
        (output / "warmstart_report.json").write_text(json.dumps(warm_report, indent=2),
                                                       encoding="utf-8")
        print(f"devices={world} total={sum(p.numel() for p in raw.parameters())/1e6:.2f}M "
              f"trainable={sum(p.numel() for p in trainable)/1e6:.2f}M "
              f"active_tensors={len(active_names)} "
              f"depth_prior={prior_m:.3f}m", flush=True)

    pipeline_blocked = False
    last_report = None
    for stage_index, (name, min_steps, max_steps, eval_every, base_lr, mixture) in enumerate(STAGES):
        if stage_index < start_stage: continue
        if args.quick: min_steps, max_steps, eval_every = 2, 4, 2
        controller = (AdaptiveTrainingBudget.from_state_dict(resume_controller)
                      if stage_index == start_stage and resume_controller else
                      AdaptiveTrainingBudget(stage_specs(stage_index), BudgetConfig(
                          min_steps=min_steps, max_steps=max_steps, eval_every=eval_every,
                          slope_window=5, plateau_patience=4, max_lr_decays=2,
                          final_stage=stage_index == len(STAGES) - 1)))
        stage_step = start_step if stage_index == start_stage else 0
        lr_factor = .5 ** controller.lr_decays
        started = time.time(); running_loss = 0.0; running_grad = 0.0
        if rank == 0:
            print(f"\n=== {name} resume={stage_step} budget={min_steps}:{max_steps} "
                  f"eval={eval_every} mix={sampler.weights(mixture)} ===", flush=True)
        while stage_step < max_steps:
            # Short warmup; LR reductions are controlled only by held-out plateaus.
            warm = min(100, max(10, min_steps // 10))
            warm_factor = min(1.0, (stage_step + 1) / warm)
            current_lr = base_lr * lr_factor * warm_factor
            for group in optimizer.param_groups: group["lr"] = current_lr
            optimizer.zero_grad(set_to_none=True); accumulated = {}
            for micro in range(args.grad_accum):
                batch = move_geometry_batch(
                    sampler.batch(global_step, micro, args.per_gpu_batch, mixture), device)
                wrong_k = (make_counterfactuals(batch)["wrong_intrinsics"]
                           if stage_index >= 2 else None)
                with torch.autocast(device_type=device.type, dtype=torch.float16,
                                    enabled=device.type == "cuda"):
                    loss, metrics = wrapped(
                        "geometry", batch["image"], batch["depth"], batch["pose_c2w"],
                        batch["depth_valid"], batch["dynamic_mask"], None,
                        batch["intrinsics"], batch["projection_y_sign"],
                        batch["rigid_flow"], batch["rigid_flow_valid"], wrong_k)
                    micro_loss = loss / args.grad_accum
                scaler.scale(micro_loss).backward()
                for key, value in metrics.items():
                    accumulated[key] = accumulated.get(key, 0.0) + value / args.grad_accum
            if global_step == 0:
                disconnected = missing_trainable_gradients(raw)
                if disconnected:
                    raise RuntimeError(
                        "Eye-v3 exact graph has trainable parameters without gradients: "
                        + ", ".join(disconnected))
            scaler.unscale_(optimizer)
            gradient = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            scaler.step(optimizer); scaler.update()
            stage_step += 1; global_step += 1
            running_loss += accumulated["loss"]; running_grad += float(gradient)
            if rank == 0 and (stage_step == 1 or stage_step % args.log_every == 0):
                rate = stage_step / max(time.time() - started, 1e-6)
                print(f"[{stage_step:5d}/{max_steps}] loss={accumulated['loss']:.4f} "
                      f"depth={accumulated['geometry_depth_nll']:.4f} "
                      f"track={accumulated['geometry_track_epe']:.4f} "
                      f"rot={accumulated['geometry_rotation']:.4f} "
                      f"trans={accumulated['geometry_translation']:.4f} "
                      f"ba={accumulated['geometry_ba_reduction']:.3f} "
                      f"grad={float(gradient):.3f} lr={current_lr:.2e} "
                      f"{rate:.2f} opt-step/s", flush=True)

            decision = None
            if stage_step % eval_every == 0 or stage_step == max_steps:
                if world > 1: dist.barrier()
                if rank == 0:
                    last_report = evaluate_geometry_v3_controls(
                        raw, validation_batches, device, args.eval_windows, prior_m)
                    observation = controller.observe(
                        stage_step, controller_metrics(last_report),
                        running_loss / max(eval_every, 1),
                        running_grad / max(eval_every, 1), current_lr)
                    decision = controller.decide()
                    record = {"stage": name, "stage_step": stage_step,
                              "global_step": global_step, "report": last_report,
                              "controller": asdict(decision)}
                    history.append(record)
                    print(json.dumps(record, indent=2), flush=True)
                    running_loss = running_grad = 0.0
                if world > 1:
                    payload = [decision]
                    dist.broadcast_object_list(payload, src=0); decision = payload[0]
                if decision.action == BudgetAction.REDUCE_LR:
                    controller.acknowledge_lr_decay(); lr_factor *= .5
                elif decision.action in (BudgetAction.ADVANCE_STAGE,
                                          BudgetAction.STOP_CONVERGED):
                    break
                elif decision.action in (BudgetAction.STOP_BLOCKED,
                                          BudgetAction.STOP_OVERFIT,
                                          BudgetAction.STOP_UNSTABLE):
                    pipeline_blocked = True; break
            if rank == 0 and global_step % args.checkpoint_every == 0:
                atomic_save({"version": "jwm-eye-v3-ctpg", "cfg": asdict(cfg),
                             "model": raw.state_dict(), "optimizer": optimizer.state_dict(),
                             "scaler": scaler.state_dict(), "history": history,
                             "stage_index": stage_index, "stage_step": stage_step,
                             "global_step": global_step,
                             "controller": controller.state_dict()}, resume)
        if rank == 0:
            atomic_save({"version": "jwm-eye-v3-ctpg", "cfg": asdict(cfg),
                         "model": raw.state_dict(), "history": history,
                         "stage_index": stage_index, "global_step": global_step,
                         "validation": last_report,
                         "status": "blocked" if pipeline_blocked else "stage_complete"},
                        output / f"stage_{stage_index}_{name}.pt")
        if world > 1: dist.barrier()
        start_step = 0; resume_controller = None
        if pipeline_blocked: break

    if rank == 0:
        status = ("promotion_gate_passed" if last_report and last_report["valid"] and
                  not pipeline_blocked else "blocked_by_causal_ood_gate")
        final = output / ("jwm_eye_v3.pt" if status == "promotion_gate_passed"
                          else "jwm_eye_v3_blocked.pt")
        atomic_save({"version": "jwm-eye-v3-ctpg", "cfg": asdict(cfg),
                     "model": raw.state_dict(), "history": history,
                     "validation": last_report, "status": status}, final)
        (output / "metrics_v3.json").write_text(json.dumps(history, indent=2),
                                                 encoding="utf-8")
        print(f"{status} -> {final}", flush=True)
    if world > 1:
        dist.barrier(); dist.destroy_process_group()


if __name__ == "__main__":
    main()
