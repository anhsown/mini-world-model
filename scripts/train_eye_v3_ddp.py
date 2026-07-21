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
from jwm.checkpoint_utils import warmstart_eye_physical, warmstart_eye_v32
from jwm.configs import (eye_physical_v3_scale, eye_physical_v32_scale,
                         eye_physical_v32_smoke_scale)
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
from jwm.anchor_discovery import discover_registered_rgbd
from jwm.real_anchored_sdg import RealAnchorProfile, RealAnchoredSyntheticGeometry
from jwm.training_metrics_v2 import metric_catalog


STAGES = (
    # name, min, max, eval_every, lr, source mixture
    ("g0_calibrated_tracks", 800, 1800, 200, 1e-4,
     {"procedural": .55, "tartanair": .45}),
    ("g1_metric_odometry", 1200, 3200, 300, 8e-5,
     {"procedural": .15, "tartanair": .35, "tum": .35, "bonn": .15}),
    ("g2_dynamic_geometry", 800, 2200, 250, 5e-5,
     {"procedural": .15, "tartanair": .15, "tum": .20, "bonn": .50}),
    ("g3_causal_ood", 600, 1800, 200, 3e-5,
     {"procedural": .10, "tartanair": .25, "tum": .35, "bonn": .30}),
)
CHECKPOINT_VERSION = "jwm-eye-v3.1-ctpg"


def apply_distributed_lr_decision(action, controller, lr_factor, *, controller_owner):
    """Apply a broadcast LR action without mutating controller replicas."""
    if action != BudgetAction.REDUCE_LR:
        return lr_factor
    if controller_owner:
        controller.acknowledge_lr_decay()
    return lr_factor * .5


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
    p.add_argument("--architecture", choices=("v31", "v32", "v321"), default="v31")
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
    p.add_argument("--anchor-root", type=Path,
                   default=Path("data/real_anchor_v1/raw"))
    p.add_argument("--dataset-registry", type=Path,
                   default=Path("configs/datasets/real_anchor_v1.json"))
    p.add_argument("--synthetic-profile", type=Path,
                   default=Path("data/real_anchor_v1/derived/eye_real_anchor_profile_v1.json"))
    p.add_argument("--synthetic-admission", type=Path,
                   default=Path("data/real_anchor_v1/synthetic_ablation_verdict.json"))
    p.add_argument("--synthetic-train-samples", type=int, default=250_000)
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
    if args.synthetic_profile.exists():
        profile = RealAnchorProfile.from_json(args.synthetic_profile)
        counts = {"train": args.synthetic_train_samples,
                  "validation": min(2_000, args.synthetic_train_samples),
                  "test": min(2_000, args.synthetic_train_samples)}
        result = {split: {"procedural": RealAnchoredSyntheticGeometry(
                    split, counts[split], profile, frames=6, size=256)}
                  for split in ("train", "validation", "test")}
    else:
        result = {split: {"procedural": ProceduralV3Dataset(split)}
                  for split in ("train", "validation", "test")}
    discovered = discover_registered_rgbd(args.anchor_root, args.dataset_registry)
    mapping = {"train": "train", "validation": "val", "test": "test"}
    for split, flag in mapping.items():
        tartan = sorted(set(getattr(args, f"tartan_{flag}") +
                            discovered[split]["tartan"]))
        tum = sorted(set(getattr(args, f"tum_{flag}") + discovered[split]["tum"]))
        bonn = sorted(set(getattr(args, f"bonn_{flag}") + discovered[split]["bonn"]))
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
    datasets = [dataset for dataset in sources.values() if len(dataset)]
    for i in range(per_source):
        for dataset in datasets:
            count = min(per_source, len(dataset))
            if i >= count:
                continue
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
                MetricSpec("track_quality_score", "max", 0.0, .40),
                MetricSpec("track_valid_fraction", "max", 0.0, .25))
    if index == 1:
        return (MetricSpec("depth_prior_gain", "max", 1.0, 1.10),
                MetricSpec("pose_identity_gain", "max", 1.0, 1.10, weight=1.5),
                MetricSpec("ba_residual_reduction", "max", 0.0, .12),
                MetricSpec("ba_pose_gain", "max", 1.0, 1.0),
                MetricSpec("track_valid_fraction", "max", 0.0, .30))
    if index == 2:
        return (MetricSpec("wrong_window_compatibility_gap", "max", 0.0, .10),
                MetricSpec("reverse_time_rpe_ratio", "max", 1.0, 1.05),
                MetricSpec("wrong_intrinsics_pose_ratio", "max", 1.0, 1.05),
                MetricSpec("track_quality_score", "max", 0.0, .65))
    return eye_v3_budget_specs()


def set_trainable(model: JWM):
    return set_eye_v3_physical_trainable(model)


def main():
    args = arguments(); world, rank, local, device = distributed()
    torch.set_float32_matmul_precision("high")
    random.seed(args.seed + rank); torch.manual_seed(args.seed + rank)
    output = Path(args.output)
    if rank == 0: output.mkdir(parents=True, exist_ok=True)
    if rank == 0:
        (output / "metric_catalog_v2.json").write_text(
            json.dumps(metric_catalog(), indent=2), encoding="utf-8")

    probe = json.loads(Path(args.probe_report).read_text(encoding="utf-8"))
    if not probe.get("valid"):
        raise RuntimeError("Eye-v3 controlled probes failed; full training blocked")
    if args.synthetic_profile.exists():
        if not args.synthetic_admission.exists():
            raise RuntimeError("real-anchored synthetic profile requires an A/B admission report")
        synthetic_admission = json.loads(
            args.synthetic_admission.read_text(encoding="utf-8"))
        if (not synthetic_admission.get("valid") or
                synthetic_admission.get("decision") != "admit"):
            raise RuntimeError("synthetic data is quarantined by the real-heldout A/B gate")
        if rank == 0:
            (output / "synthetic_admission_used.json").write_text(
                json.dumps(synthetic_admission, indent=2), encoding="utf-8")
    datasets = make_datasets(args)
    if rank == 0:
        admission = validate_geometry_v3_datasets(
            datasets, output / "dataset_validation_v31.json")
        available = set(datasets["train"]) - {"procedural"}
        missing = {"tartanair", "tum", "bonn"} - available
        if missing and not (args.quick or args.allow_partial_data):
            raise RuntimeError(f"missing mandatory train sources: {sorted(missing)}")
        if not admission["valid"]:
            raise RuntimeError(f"dataset hypotheses failed: {admission['failures']}")
        print(json.dumps(admission, indent=2), flush=True)
    if world > 1: dist.barrier()

    cfg = (eye_physical_v32_scale() if args.architecture in ("v32", "v321")
           else eye_physical_v3_scale())
    if args.quick and args.architecture in ("v32", "v321"):
        cfg = eye_physical_v32_smoke_scale()
    elif args.quick:
        cfg.image_size = 64; cfg.geometry_v3_width = 32
        cfg.geometry_track_points = 12; cfg.geometry_track_iterations = 1
        cfg.geometry_ba_iterations = 1
    model = JWM(cfg)
    warm_report = (warmstart_eye_v32(model, args.warmstart)
                   if args.architecture in ("v32", "v321")
                   else warmstart_eye_physical(model, args.warmstart))
    active_names = set_trainable(model); model.to(device)
    trainable = [p for p in model.parameters() if p.requires_grad]
    wrapped = (DDP(model, device_ids=[local] if device.type == "cuda" else None,
                   broadcast_buffers=False, find_unused_parameters=False)
               if world > 1 else model)
    raw = wrapped.module if hasattr(wrapped, "module") else wrapped
    sampler = V3Sampler(datasets["train"], world, rank, args.seed)
    real_validation = {name: dataset for name, dataset in datasets["validation"].items()
                       if name not in ("procedural", "synthetic_real_anchored")}
    real_test = {name: dataset for name, dataset in datasets["test"].items()
                 if name not in ("procedural", "synthetic_real_anchored")}
    real_train = {name: dataset for name, dataset in datasets["train"].items()
                  if name not in ("procedural", "synthetic_real_anchored")}
    if not real_validation or not real_test:
        raise RuntimeError("checkpoint promotion requires real validation and test sources")
    validation_batches = fixed_eval_batches(real_validation, 8)
    test_batches = fixed_eval_batches(real_test, 8)
    prior_m = depth_prior(real_train)
    optimizer = torch.optim.AdamW(trainable, lr=STAGES[0][4], betas=(.9, .95),
                                  weight_decay=.05)
    scaler = torch.amp.GradScaler("cuda", init_scale=256,
                                  enabled=device.type == "cuda")
    history, global_step, start_stage, start_step = [], 0, 0, 0
    nonfinite_skips = 0
    resume = output / "resume.pt"
    resume_controller = None
    checkpoint_version = ({
        "v32": "jwm-eye-v3.2-depth-ray-registers",
        "v321": "jwm-eye-v3.2.1-robust-causal-geometry",
    }.get(args.architecture, CHECKPOINT_VERSION))
    if resume.exists():
        state = torch.load(resume, map_location=device, weights_only=False)
        if state.get("version") != checkpoint_version:
            raise RuntimeError("incompatible Eye-v3 resume checkpoint")
        raw.load_state_dict(state["model"],
                            strict=state.get("model_kind") != "geometry_delta")
        optimizer.load_state_dict(state["optimizer"]); scaler.load_state_dict(state["scaler"])
        history = state.get("history", []); global_step = int(state["global_step"])
        nonfinite_skips = int(state.get("nonfinite_skips", 0))
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
        nonfinite_streak = 0
        started = time.time(); running_loss = 0.0; running_grad = 0.0
        if rank == 0:
            print(f"\n=== {name} resume={stage_step} budget={min_steps}:{max_steps} "
                  f"eval={eval_every} mix={sampler.weights(mixture)} ===", flush=True)
        while stage_step < max_steps:
            # Short warmup; LR reductions are controlled only by held-out plateaus.
            warm = min(200, max(25, min_steps // 4))
            warm_factor = min(1.0, (stage_step + 1) / warm)
            current_lr = base_lr * lr_factor * warm_factor
            for group in optimizer.param_groups: group["lr"] = current_lr
            optimizer.zero_grad(set_to_none=True); accumulated = {}
            forward_finite = True
            for micro in range(args.grad_accum):
                batch = move_geometry_batch(
                    sampler.batch(global_step + nonfinite_skips, micro,
                                  args.per_gpu_batch, mixture), device)
                counterfactuals = make_counterfactuals(batch)
                # Bound compute to two encoder passes: later stages alternate
                # temporal and calibration negatives instead of running both.
                use_wrong_k = stage_index >= 2 and (global_step + micro) % 2 == 0
                wrong_k = counterfactuals["wrong_intrinsics"] if use_wrong_k else None
                wrong_window = (None if use_wrong_k else
                                counterfactuals["wrong_window_image"])
                with torch.autocast(device_type=device.type, dtype=torch.float16,
                                    enabled=device.type == "cuda"):
                    loss, metrics = wrapped(
                        "geometry", batch["image"], batch["depth"], batch["pose_c2w"],
                        batch["depth_valid"], batch["dynamic_mask"], wrong_window,
                        batch["intrinsics"], batch["projection_y_sign"],
                        batch["rigid_flow"], batch["rigid_flow_valid"], wrong_k)
                    forward_finite = forward_finite and bool(torch.isfinite(loss.detach()))
                    micro_loss = torch.nan_to_num(loss, nan=0.0, posinf=0.0,
                                                  neginf=0.0) / args.grad_accum
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
            local_healthy = forward_finite and all(
                parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
                for parameter in trainable)
            healthy = torch.tensor(int(local_healthy), device=device)
            if world > 1:
                dist.all_reduce(healthy, op=dist.ReduceOp.MIN)
            if not bool(healthy):
                optimizer.zero_grad(set_to_none=True)
                nonfinite_skips += 1; nonfinite_streak += 1
                lr_factor *= .5
                scaler.update(new_scale=max(float(scaler.get_scale()) / 2, 1.0))
                if rank == 0:
                    print(f"SKIP non-finite step: streak={nonfinite_streak} "
                          f"total={nonfinite_skips} next_lr_factor={lr_factor:.4f}",
                          flush=True)
                if nonfinite_streak >= 3:
                    pipeline_blocked = True
                    break
                continue
            gradient = torch.nn.utils.clip_grad_norm_(
                trainable, 1.0, error_if_nonfinite=True)
            scaler.step(optimizer); scaler.update()
            nonfinite_streak = 0
            stage_step += 1; global_step += 1
            running_loss += accumulated["loss"]; running_grad += float(gradient)
            if rank == 0 and (stage_step == 1 or stage_step % args.log_every == 0):
                rate = stage_step / max(time.time() - started, 1e-6)
                print(f"[{stage_step:5d}/{max_steps}] loss={accumulated['loss']:.4f} "
                      f"depth={accumulated['geometry_depth_nll']:.4f} "
                      f"track={accumulated['geometry_track_epe']:.4f} "
                      f"valid={accumulated['geometry_track_valid_fraction']:.3f} "
                      f"rot={accumulated['geometry_rotation']:.4f} "
                      f"trans={accumulated['geometry_translation']:.4f} "
                      f"ba={accumulated['geometry_ba_reduction']:.3f} "
                      f"rigid={accumulated['geometry_rigid_epe']:.3f} "
                      f"dyn={accumulated['geometry_dynamic_bce']:.3f} "
                      f"cf={accumulated['geometry_calibration_contrast']:.3f} "
                      f"ray={accumulated.get('geometry_ray_angular', 0.0):.3f} "
                      f"cycle={accumulated.get('geometry_track_cycle', 0.0):.3f} "
                      f"conf={accumulated.get('geometry_confidence_bce', 0.0):.3f} "
                      f"vis={accumulated.get('geometry_visibility_bce', 0.0):.3f} "
                      f"temp={accumulated.get('geometry_temporal_bce', 0.0):.3f} "
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
                    lr_factor = apply_distributed_lr_decision(
                        decision.action, controller, lr_factor,
                        controller_owner=(rank == 0))
                elif decision.action in (BudgetAction.ADVANCE_STAGE,
                                          BudgetAction.STOP_CONVERGED):
                    break
                elif decision.action in (BudgetAction.STOP_BLOCKED,
                                          BudgetAction.STOP_OVERFIT,
                                          BudgetAction.STOP_UNSTABLE):
                    pipeline_blocked = True; break
            if rank == 0 and global_step % args.checkpoint_every == 0:
                model_state = (dict(raw.geometry.state_dict())
                               if args.architecture in ("v32", "v321") else raw.state_dict())
                if args.architecture in ("v32", "v321"):
                    model_state = {f"geometry.{key}": value
                                   for key, value in model_state.items()}
                atomic_save({"version": checkpoint_version, "cfg": asdict(cfg),
                             "model": model_state,
                             "model_kind": ("geometry_delta" if args.architecture in ("v32", "v321")
                                            else "full"),
                             "optimizer": optimizer.state_dict(),
                             "scaler": scaler.state_dict(), "history": history,
                             "stage_index": stage_index, "stage_step": stage_step,
                             "global_step": global_step,
                             "nonfinite_skips": nonfinite_skips,
                             "controller": controller.state_dict()}, resume)
        if rank == 0:
            stage_state = (dict(raw.geometry.state_dict())
                           if args.architecture in ("v32", "v321") else raw.state_dict())
            if args.architecture in ("v32", "v321"):
                stage_state = {f"geometry.{key}": value for key, value in stage_state.items()}
            atomic_save({"version": checkpoint_version, "cfg": asdict(cfg),
                         "model": stage_state,
                         "model_kind": ("geometry_delta" if args.architecture in ("v32", "v321")
                                        else "full"), "history": history,
                         "stage_index": stage_index, "global_step": global_step,
                         "nonfinite_skips": nonfinite_skips,
                         "validation": last_report,
                         "status": "blocked" if pipeline_blocked else "stage_complete"},
                        output / f"stage_{stage_index}_{name}.pt")
        if world > 1: dist.barrier()
        start_step = 0; resume_controller = None
        if pipeline_blocked: break

    if rank == 0:
        test_report = evaluate_geometry_v3_controls(
            raw, test_batches, device, args.eval_windows, prior_m)
        (output / "final_real_test_metrics.json").write_text(
            json.dumps(test_report, indent=2), encoding="utf-8")
        status = ("promotion_gate_passed" if last_report and last_report["valid"] and
                  test_report["valid"] and not pipeline_blocked
                  else "blocked_by_causal_ood_gate")
        stem = {"v32": "jwm_eye_v32", "v321": "jwm_eye_v321"}.get(
            args.architecture, "jwm_eye_v31")
        final = output / (f"{stem}.pt" if status == "promotion_gate_passed"
                          else f"{stem}_blocked.pt")
        atomic_save({"version": checkpoint_version, "cfg": asdict(cfg),
                     "model": raw.state_dict(), "history": history,
                     "validation": last_report, "test": test_report,
                     "status": status}, final)
        (output / f"metrics_{args.architecture}.json").write_text(
            json.dumps(history, indent=2), encoding="utf-8")
        print(f"{status} -> {final}", flush=True)
    if world > 1:
        dist.barrier(); dist.destroy_process_group()


if __name__ == "__main__":
    main()
