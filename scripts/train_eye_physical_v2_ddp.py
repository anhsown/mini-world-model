"""Metric-gated Eye Physical v2 ablation/full training on Kaggle T4x2."""

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

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jwm import JWM
from jwm.checkpoint_utils import warmstart_eye_physical
from jwm.configs import eye_physical_v2_ablation
from jwm.geometry_trainer import evaluate_geometry_controls
from jwm.geometry_v2_data import (
    BonnRGBDWindowDataset,
    TaggedTUMDataset,
    TartanAirWindowDataset,
    procedural_v2_row,
    validate_geometry_source,
    validate_split_disjoint,
)


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--mode", choices=("ablation", "full"), default="ablation")
    parser.add_argument("--arm", default="auto")
    parser.add_argument("--arms", default="ABCD",
                        help="Ablation arms to run, e.g. A, BD, or ABCD")
    parser.add_argument("--warmstart", required=True)
    parser.add_argument("--tartan-train", nargs="*", default=[])
    parser.add_argument("--tartan-val", nargs="*", default=[])
    parser.add_argument("--tum-train", nargs="*", default=[])
    parser.add_argument("--tum-val", nargs="*", default=[])
    parser.add_argument("--tum-test", nargs="*", default=[])
    parser.add_argument("--bonn-train", nargs="*", default=[])
    parser.add_argument("--bonn-val", nargs="*", default=[])
    parser.add_argument("--bonn-test", nargs="*", default=[])
    parser.add_argument("--pilot-steps", type=int, default=400)
    parser.add_argument("--steps-e0", type=int, default=800)
    parser.add_argument("--steps-e1", type=int, default=1800)
    parser.add_argument("--steps-e2", type=int, default=1000)
    parser.add_argument("--per-gpu-batch", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--checkpoint-every", type=int, default=200)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--eval-windows", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args()


def distributed():
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local = int(os.environ.get("LOCAL_RANK", "0"))
    device = torch.device("cuda", local) if torch.cuda.is_available() else torch.device("cpu")
    if device.type == "cuda":
        torch.cuda.set_device(local)
    if world > 1:
        # LOCAL_RANK already pins each process to one homogeneous Kaggle GPU.
        # Omitting device_id also keeps this script compatible with older
        # PyTorch builds where init_process_group did not accept that keyword.
        dist.init_process_group("nccl" if device.type == "cuda" else "gloo")
    return world, rank, local, device


def atomic_save(payload, path: Path):
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def cosine_lr(base: float, step: int, total: int,
              warmup: int = 100, floor: float = 0.10) -> float:
    """Per-stage linear warmup followed by cosine decay."""
    warmup = min(warmup, max(1, total // 10))
    if step < warmup:
        return base * (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup - 1)
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
    return base * (floor + (1.0 - floor) * cosine)


def stack_rows(rows: list[dict]) -> dict:
    required = ("image", "depth", "depth_valid", "pose_c2w")
    batch = {key: torch.stack([row[key] for row in rows]) for key in required}
    if all("dynamic_mask" in row for row in rows):
        batch["dynamic_mask"] = torch.stack([row["dynamic_mask"] for row in rows])
    batch["source"] = rows[0].get("source", "unknown")
    return batch


def move(batch: dict, device: torch.device) -> dict:
    return {key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
            for key, value in batch.items()}


def make_datasets(args):
    common = dict(frames=8, frame_stride=3, window_stride=12,
                  height=256, width=256)
    datasets = {"train": {}, "validation": {}, "test": {}}
    if args.tum_train:
        datasets["train"]["tum"] = TaggedTUMDataset(args.tum_train, source="tum", **common)
    if args.tum_val:
        datasets["validation"]["tum"] = TaggedTUMDataset(args.tum_val, source="tum", **common)
    if args.tum_test:
        datasets["test"]["tum"] = TaggedTUMDataset(args.tum_test, source="tum", **common)
    if args.bonn_train:
        datasets["train"]["bonn"] = BonnRGBDWindowDataset(args.bonn_train, **common)
    if args.bonn_val:
        datasets["validation"]["bonn"] = BonnRGBDWindowDataset(args.bonn_val, **common)
    if args.bonn_test:
        datasets["test"]["bonn"] = BonnRGBDWindowDataset(args.bonn_test, **common)
    tartan_common = dict(frames=8, frame_stride=2, window_stride=8,
                         height=256, width=256)
    if args.tartan_train:
        datasets["train"]["tartanair"] = TartanAirWindowDataset(
            args.tartan_train, **tartan_common)
    if args.tartan_val:
        datasets["validation"]["tartanair"] = TartanAirWindowDataset(
            args.tartan_val, **tartan_common)
    return datasets


def validate_datasets(datasets: dict, output: Path) -> dict:
    reports = {"sources": {}, "splits": {}}
    for split, sources in datasets.items():
        for name, dataset in sources.items():
            key = f"{split}/{name}"
            # Bonn's value is dynamic foreground and real RGB-D appearance;
            # several valid sequences deliberately keep the camera nearly
            # static. TUM/Tartan remain mandatory camera-motion sources.
            reports["sources"][key] = validate_geometry_source(
                dataset, key, require_camera_motion=(name != "bonn"))
    for source in ("tum", "bonn", "tartanair"):
        available = [(split, datasets[split][source]) for split in datasets
                     if source in datasets[split]]
        if len(available) >= 2:
            by_split = dict(available)
            reports["splits"][source] = validate_split_disjoint(
                by_split.get("train"), by_split.get("validation"), by_split.get("test"))
    reports["valid"] = (all(row.get("valid", False)
                            for row in reports["sources"].values()) and
                        all(row.get("valid", False)
                            for row in reports["splits"].values()))
    reports["failures"] = {
        key: [name for name in row.get("required_hypotheses", [])
              if not row.get("hypotheses", {}).get(name, False)]
        for key, row in reports["sources"].items() if not row.get("valid", False)
    }
    reports["split_failures"] = {
        key: row.get("leaked_scene_ids", [])
        for key, row in reports["splits"].items() if not row.get("valid", False)
    }
    output.write_text(json.dumps(reports, indent=2), encoding="utf-8")
    return reports


MIX_SIM = {"procedural": 0.15, "tartanair": 0.60, "tum": 0.15, "bonn": 0.10}
MIX_REAL = {"procedural": 0.10, "tartanair": 0.20, "tum": 0.40, "bonn": 0.30}


class GeometrySampler:
    def __init__(self, datasets: dict, world: int, rank: int, seed: int):
        self.datasets, self.world, self.rank, self.seed = datasets, world, rank, seed

    def available_weights(self, requested: dict[str, float]):
        admitted = {name: weight for name, weight in requested.items()
                    if name == "procedural" or
                    (name in self.datasets and len(self.datasets[name]) > 0)}
        total = sum(admitted.values())
        return {name: weight / total for name, weight in admitted.items()}

    def batch(self, optimizer_step: int, micro_step: int, batch_size: int,
              requested: dict[str, float], counterfactual: bool) -> dict:
        weights = self.available_weights(requested)
        rng = random.Random(self.seed + optimizer_step * 1009 + micro_step * 37)
        marker = rng.random(); cumulative = 0.0; source = "procedural"
        for name, probability in weights.items():
            cumulative += probability
            if marker <= cumulative:
                source = name; break
        rows, negatives = [], []
        for item in range(batch_size):
            global_index = ((optimizer_step * self.world + self.rank) * batch_size +
                            item + micro_step * 104729)
            if source == "procedural":
                seed = self.seed + 10_000_000 + global_index
                row = procedural_v2_row(seed)
                negative = procedural_v2_row(seed + 2_000_000)
            else:
                dataset = self.datasets[source]
                index = global_index % len(dataset)
                row = dataset[index]
                negative = dataset[(index + max(1, len(dataset) // 2)) % len(dataset)]
            rows.append(row); negatives.append(negative["image"])
        batch = stack_rows(rows)
        if counterfactual:
            batch["negative_image"] = torch.stack(negatives)
        return batch


def set_trainable(model: JWM):
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for module in (model.vision_stem, model.geometry):
        for parameter in module.parameters():
            parameter.requires_grad_(True)
    model.e_img.requires_grad_(True)


def eval_batches(sources: dict, per_source: int):
    result = []
    for dataset in sources.values():
        count = min(per_source, len(dataset))
        if count == 0:
            continue
        for i in range(count):
            index = min(len(dataset) - 1, math.floor(i * len(dataset) / count))
            result.append(stack_rows([dataset[index]]))
    return result


def estimate_depth_prior(sources: dict, procedural_samples: int = 8,
                         max_source_windows: int = 8) -> float:
    """Calibrate one target-independent baseline from training data only."""
    values = []
    for seed in range(procedural_samples):
        row = procedural_v2_row(3_000_000 + seed, frames=4, size=96)
        values.append(row["depth"][row["depth_valid"]][::32])
    for dataset in sources.values():
        count = min(max_source_windows, len(dataset))
        for i in range(count):
            index = min(len(dataset) - 1, math.floor(i * len(dataset) / count))
            row = dataset[index]
            valid = row.get("depth_valid", row["depth"] > 0)
            values.append(row["depth"][valid][::64])
    if not values:
        return 2.5
    return float(torch.cat(values).median())


def arm_score(report: dict) -> float:
    if not report.get("controls"):
        return -1e9
    normal = report["controls"]["normal"]
    passed = sum(report["gates"].values())
    ratios = report["ratios"]
    evidence = sum(min(3.0, value) for value in ratios.values())
    return (100.0 * passed + evidence - 10.0 * normal["depth_abs_rel"] -
            20.0 * normal["ate_metric"])


def train_arm(args, arm: str, datasets: dict, world: int, rank: int, local: int,
              device: torch.device, stages: list[tuple[str, int, float, dict]],
              arm_output: Path, initial_checkpoint: Path | None = None):
    # The same seed before every arm gives an identical shared initialization.
    torch.manual_seed(args.seed)
    cfg = eye_physical_v2_ablation(arm)
    model = JWM(cfg)
    warm_report = warmstart_eye_physical(model, args.warmstart)
    if initial_checkpoint is not None and initial_checkpoint.exists():
        pilot = torch.load(initial_checkpoint, map_location="cpu",
                           weights_only=False)
        if (pilot.get("version") != "jwm-eye-physical-v2" or
                pilot.get("arm") != arm):
            raise RuntimeError(f"incompatible pilot checkpoint: {initial_checkpoint}")
        model.load_state_dict(pilot["model"], strict=True)
        warm_report["continued_from_pilot"] = str(initial_checkpoint)
    set_trainable(model); model.to(device)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    wrapped = (DDP(model, device_ids=[local], find_unused_parameters=False)
               if world > 1 else model)
    raw = wrapped.module if hasattr(wrapped, "module") else wrapped
    optimizer = torch.optim.AdamW(trainable, lr=stages[0][2],
                                  betas=(0.9, 0.95), weight_decay=0.05)
    scaler = torch.amp.GradScaler("cuda", init_scale=1024.0,
                                  enabled=device.type == "cuda")
    # Every ablation arm sees the exact same positive samples in the exact same
    # order. Arm D additionally consumes the deterministic paired negatives.
    sampler = GeometrySampler(datasets["train"], world, rank, args.seed)
    depth_prior_m = estimate_depth_prior(datasets["train"])
    history, global_step = [], 0
    resume_stage, resume_step = 0, 0
    arm_output.mkdir(parents=True, exist_ok=True)
    resume_path = arm_output / "resume.pt"
    if resume_path.exists():
        payload = torch.load(resume_path, map_location=device, weights_only=False)
        if (payload.get("version") != "jwm-eye-physical-v2" or
                payload.get("arm") != arm):
            raise RuntimeError(f"incompatible resume checkpoint: {resume_path}")
        raw.load_state_dict(payload["model"], strict=True)
        optimizer.load_state_dict(payload["optimizer"])
        scaler.load_state_dict(payload["scaler"])
        history = payload.get("history", [])
        global_step = int(payload.get("global_step", 0))
        resume_stage = int(payload.get("stage_index", 0))
        resume_step = int(payload.get("step", 0))
        if rank == 0:
            print(f"RESUME arm={arm} stage={resume_stage} step={resume_step} "
                  f"global_step={global_step}", flush=True)
    if rank == 0:
        (arm_output / "warmstart_report.json").write_text(
            json.dumps(warm_report, indent=2), encoding="utf-8")
        print(f"\n### ARM {arm} total={sum(p.numel() for p in model.parameters())/1e6:.2f}M "
              f"trainable={sum(p.numel() for p in trainable)/1e6:.2f}M "
              f"train-only-depth-prior={depth_prior_m:.3f}m", flush=True)

    for stage_index, (stage_name, steps, lr, mixture) in enumerate(stages):
        if args.quick:
            steps = min(3, steps)
        if stage_index < resume_stage:
            continue
        start_step = resume_step if stage_index == resume_stage else 0
        if start_step > steps:
            raise RuntimeError(
                f"resume step {start_step} exceeds stage length {steps} for {stage_name}")
        started = time.time()
        if rank == 0:
            print(f"=== arm={arm} {stage_name} steps={steps} lr={lr} "
                  f"mix={sampler.available_weights(mixture)} ===", flush=True)
        for step in range(start_step, steps):
            current_lr = cosine_lr(lr, step, steps)
            for group in optimizer.param_groups:
                group["lr"] = current_lr
            optimizer.zero_grad(set_to_none=True)
            accumulated = {}
            for micro in range(args.grad_accum):
                use_counterfactual = cfg.geometry_counterfactual_weight > 0
                batch = move(sampler.batch(global_step, micro, args.per_gpu_batch,
                                           mixture, use_counterfactual), device)
                with torch.autocast(device_type=device.type, dtype=torch.float16,
                                    enabled=device.type == "cuda"):
                    loss, metrics = wrapped(
                        "geometry", batch["image"], batch["depth"],
                        batch["pose_c2w"], batch["depth_valid"],
                        batch.get("dynamic_mask"), batch.get("negative_image"))
                    scaled_loss = loss / args.grad_accum
                scaler.scale(scaled_loss).backward()
                for key, value in metrics.items():
                    accumulated[key] = accumulated.get(key, 0.0) + value / args.grad_accum
            scaler.unscale_(optimizer)
            gradient = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            scaler.step(optimizer); scaler.update()
            global_step += 1
            if rank == 0 and (step == 0 or (step + 1) % args.log_every == 0):
                rate = (step + 1) / max(time.time() - started, 1e-6)
                print(f"[{step+1:5d}/{steps}] loss={accumulated['loss']:.4f} "
                      f"metric_depth={accumulated['geometry_depth_metric']:.4f} "
                      f"rel_rot={accumulated['geometry_relative_rotation']:.4f} "
                      f"cycle={accumulated['geometry_cycle']:.4f} "
                      f"cf={accumulated['geometry_counterfactual']:.4f} "
                      f"grad={float(gradient):.3f} lr={current_lr:.2e} "
                      f"{rate:.2f} opt-step/s", flush=True)
            if rank == 0 and global_step % args.checkpoint_every == 0:
                atomic_save({"version": "jwm-eye-physical-v2", "arm": arm,
                             "cfg": asdict(cfg), "model": raw.state_dict(),
                             "optimizer": optimizer.state_dict(),
                             "scaler": scaler.state_dict(), "history": history,
                             "stage_name": stage_name,
                             "stage_index": stage_index, "step": step + 1,
                             "global_step": global_step, "lr": current_lr},
                            arm_output / "resume.pt")
        if world > 1:
            dist.barrier()
        resume_step = 0
        if rank == 0:
            validation = evaluate_geometry_controls(
                raw, eval_batches(datasets["validation"],
                                  max(1, args.eval_windows //
                                      max(1, len(datasets["validation"])))),
                device, args.eval_windows, depth_prior_m)
            ood_sources = datasets["test"] or datasets["validation"]
            ood = evaluate_geometry_controls(
                raw, eval_batches(ood_sources,
                                  max(1, args.eval_windows // max(1, len(ood_sources)))),
                device, args.eval_windows, depth_prior_m)
            result = {"stage": stage_name, "steps": steps,
                      "seconds": time.time() - started,
                      "train_only_depth_prior_m": depth_prior_m,
                      "validation": validation, "ood": ood,
                      "score": arm_score(ood)}
            history.append(result)
            print(json.dumps(result, indent=2), flush=True)
            atomic_save({"version": "jwm-eye-physical-v2", "arm": arm,
                         "cfg": asdict(cfg), "model": raw.state_dict(),
                         "history": history,
                         "status": ("promotion_gate_passed" if ood.get("valid")
                                    else "blocked_by_ood_gate")},
                        arm_output / f"stage_{stage_index}_{stage_name}.pt")
            (arm_output / "metrics.json").write_text(
                json.dumps(history, indent=2), encoding="utf-8")
        if world > 1:
            dist.barrier()
    if rank == 0:
        atomic_save({"version": "jwm-eye-physical-v2", "arm": arm,
                     "cfg": asdict(cfg), "model": raw.state_dict(),
                     "history": history,
                     "status": ("promotion_gate_passed"
                                if history[-1]["ood"].get("valid")
                                else "blocked_by_ood_gate")},
                    arm_output / "final.pt")
    del wrapped, model, optimizer, scaler
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return history[-1] if rank == 0 else None


def main():
    args = arguments()
    world, rank, local, device = distributed()
    torch.set_float32_matmul_precision("high")
    random.seed(args.seed + rank); torch.manual_seed(args.seed + rank)
    output = Path(args.output)
    if rank == 0:
        output.mkdir(parents=True, exist_ok=True)
    datasets = make_datasets(args)
    if rank == 0:
        report = validate_datasets(datasets, output / "dataset_validation_v2.json")
        required = {"tum", "bonn", "tartanair"}
        missing = required - set(datasets["train"])
        if (not args.quick) and missing:
            raise RuntimeError(f"missing required Eye-v2 train sources: {sorted(missing)}")
        if not report["valid"]:
            raise RuntimeError("dataset admission hypotheses failed")
        print(json.dumps(report, indent=2), flush=True)
    if world > 1:
        dist.barrier()

    if args.mode == "ablation":
        stages = [("pilot_real_rich", args.pilot_steps, 2e-4, MIX_REAL)]
        summaries = {}
        arms = "".join(dict.fromkeys(args.arms.upper()))
        if not arms or any(arm not in "ABCD" for arm in arms):
            raise ValueError("--arms must contain one or more of A, B, C, D")
        for arm in arms:
            result = train_arm(args, arm, datasets, world, rank, local, device,
                               stages, output / f"arm_{arm}")
            if rank == 0:
                summaries[arm] = result
        if rank == 0:
            winner = max(summaries, key=lambda name: summaries[name]["score"])
            passed = [name for name, row in summaries.items()
                      if row["ood"].get("valid")]
            summary = {"winner": winner, "strict_gate_passed_arms": passed,
                       "full_training_admitted": bool(passed), "arms": summaries}
            (output / "ablation_summary.json").write_text(
                json.dumps(summary, indent=2), encoding="utf-8")
            (output / "winner.txt").write_text(winner, encoding="utf-8")
            print(json.dumps(summary, indent=2), flush=True)
    else:
        arm = args.arm.upper()
        if arm == "AUTO":
            winner_file = output / "winner.txt"
            if not winner_file.exists():
                raise RuntimeError("--arm auto requires ablation winner.txt")
            arm = winner_file.read_text().strip().upper()
        stages = [
            ("e0_sim_diverse", args.steps_e0, 3e-4, MIX_SIM),
            ("e1_real_metric_adapt", args.steps_e1, 1.2e-4, MIX_REAL),
            ("e2_hard_counterfactual", args.steps_e2, 8e-5, MIX_REAL),
        ]
        result = train_arm(args, arm, datasets, world, rank, local, device,
                           stages, output / f"full_arm_{arm}",
                           initial_checkpoint=output / f"arm_{arm}" / "final.pt")
        if rank == 0:
            final = output / f"full_arm_{arm}" / "final.pt"
            if result["ood"].get("valid"):
                promoted = output / "jwm_eye_physical_v2.pt"
                shutil.copy2(final, promoted)
                print(f"PROMOTED -> {promoted}", flush=True)
            else:
                blocked = output / "jwm_eye_physical_v2_blocked.pt"
                shutil.copy2(final, blocked)
                print(f"BLOCKED -> {blocked}", flush=True)
    if world > 1:
        dist.barrier(); dist.destroy_process_group()


if __name__ == "__main__":
    main()
