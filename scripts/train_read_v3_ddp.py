"""Train JWM-Read v3 with torchrun on Kaggle T4x2.

Example:
  torchrun --standalone --nproc_per_node=2 scripts/train_read_v3_ddp.py \
    --jsonl /kaggle/working/vdoc/vdoc.jsonl \
    --images /kaggle/working/vdoc/images \
    --output /kaggle/working/jwm_read_v3

The script is deliberately notebook-independent: checkpoints are atomic,
resume is automatic, rank-specific data streams do not duplicate samples, and
stage promotion is controlled by free-running metrics instead of training CE.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jwm.configs import reader_scale_v3
from jwm.model import JWM
from jwm.read_data import find_fonts, load_corpus_lines, load_doc_pairs
from jwm.read_v3_data import (PrefetchReadV3, ReadV3Batcher, build_v3_eval,
                              curate_doc_pairs, split_by_document,
                              validate_read_v3_data)
from jwm.read_v3_trainer import eval_read_v3, eval_vision_gain_v3
from jwm.sdg import CameraParams


STAGES = [
    {
        "name": "s0_glyph_bootstrap", "levels": (1, 2),
        "synth": 1.0, "random": 1.0, "steps": 1800, "lr": 3e-4,
        "ctc": 1.20, "box": 0.50, "contrast": 0.20,
        "gate_kinds": ("randL1", "randL2"), "max_extensions": 2,
        "gate": {"ctc_cer_max": 0.72, "vision_gap_min": 0.015},
    },
    {
        "name": "s1_layout_ocr", "levels": (1, 2, 3, 4),
        "synth": 1.0, "random": 0.80, "steps": 2500, "lr": 2e-4,
        "ctc": 1.00, "box": 0.35, "contrast": 0.16,
        "gate_kinds": ("randL2", "randL3"), "max_extensions": 2,
        "gate": {"ctc_cer_max": 0.82, "box_iou_min": 0.20,
                 "vision_gap_min": 0.020},
    },
    {
        "name": "s2_real_document_adapt", "levels": (1, 2, 3, 4),
        "synth": 0.55, "random": 0.45, "steps": 3000, "lr": 1.2e-4,
        "ctc": 0.65, "box": 0.22, "contrast": 0.14,
        "gate_kinds": ("randL3", "randL4", "doc"), "max_extensions": 1,
        "gate": {"synthetic_cer_max": 1.05, "vision_win_min": 0.55},
    },
    {
        "name": "s3_reasoning_ood", "levels": (2, 3, 4),
        "synth": 0.35, "random": 0.30, "steps": 2200, "lr": 8e-5,
        "ctc": 0.35, "box": 0.15, "contrast": 0.10,
        "gate_kinds": (), "max_extensions": 0, "gate": {},
    },
]
EXTENSION_STEPS = 700


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--jsonl", required=True)
    p.add_argument("--images", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--per-gpu-batch", type=int, default=3)
    p.add_argument("--grad-accum", type=int, default=2)
    p.add_argument("--workers-prefetch", type=int, default=2)
    p.add_argument("--checkpoint-every", type=int, default=250)
    p.add_argument("--log-every", type=int, default=25)
    p.add_argument("--seed", type=int, default=20260719)
    p.add_argument("--limit-pairs", type=int, default=260000)
    p.add_argument("--resume", default="auto")
    p.add_argument("--allow-one-gpu", action="store_true")
    p.add_argument("--quick", action="store_true",
                   help="2-process plumbing smoke test; not a useful model")
    return p.parse_args()


def setup_distributed(args):
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        backend = "nccl"
    else:
        device = torch.device("cpu")
        backend = "gloo"
    if world > 1:
        dist.init_process_group(backend=backend)
    if world != 2 and not args.allow_one_gpu and not args.quick:
        raise RuntimeError(f"JWM-Read v3 expects Kaggle T4x2, found WORLD_SIZE={world}")
    return rank, local_rank, world, device


def rank0(rank, *items):
    if rank == 0:
        print(*items, flush=True)


def barrier(world):
    if world > 1:
        dist.barrier()


def broadcast_bool(value: bool, device, world) -> bool:
    x = torch.tensor([int(value)], device=device)
    if world > 1:
        dist.broadcast(x, src=0)
    return bool(x.item())


def atomic_torch_save(payload, path: Path):
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def write_json(obj, path: Path):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def checkpoint_payload(raw, optimizer, scaler, stage_idx, step_in_stage,
                       global_step, history, cfg):
    return {
        "version": "jwm-read-v3", "cfg": asdict(cfg),
        "model": raw.state_dict(), "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(), "stage_idx": stage_idx,
        "step_in_stage": step_in_stage, "global_step": global_step,
        "history": history,
    }


def load_resume(path: Path, model, rank):
    state = {"stage_idx": 0, "step_in_stage": 0, "global_step": 0,
             "history": [], "optimizer": None, "scaler": None}
    if not path.exists():
        return state
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"], strict=True)
    for key in state:
        if key in ckpt:
            state[key] = ckpt[key]
    rank0(rank, f"RESUME {path}: stage={state['stage_idx']} "
                f"step={state['step_in_stage']} global={state['global_step']}")
    return state


def cosine_lr(base, step, total, warmup=100, floor=0.10):
    if step < warmup:
        return base * (step + 1) / max(1, warmup)
    p = (step - warmup) / max(1, total - warmup)
    return base * (floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * min(1, p))))


def reduce_metrics(metrics: dict, device, world):
    keys = sorted(metrics)
    vals = torch.tensor([float(metrics[k]) for k in keys], device=device)
    if world > 1:
        dist.all_reduce(vals, op=dist.ReduceOp.SUM)
        vals /= world
    return {k: float(v) for k, v in zip(keys, vals.cpu())}


def gate_report(stage, evaluation, vision):
    kinds = stage["gate_kinds"]
    selected = [evaluation["by_kind"][k] for k in kinds
                if k in evaluation["by_kind"]]
    synth = [x for k, x in evaluation["by_kind"].items()
             if k in kinds and k.startswith("rand")]

    def mean(key, groups):
        values = [g[key] for g in groups if key in g]
        return float(np.mean(values)) if values else float("nan")

    measured = {
        "cer": mean("cer", selected),
        "synthetic_cer": mean("cer", synth),
        "ctc_cer": mean("ctc_cer", synth),
        "box_iou": mean("box_iou", synth),
        "vision_gap": vision["loss_gap_shuffled_minus_correct"],
        "vision_win": vision["correct_image_win_rate"],
    }
    rules = stage["gate"]
    checks = {}
    if "ctc_cer_max" in rules:
        checks["ctc_cer"] = measured["ctc_cer"] <= rules["ctc_cer_max"]
    if "synthetic_cer_max" in rules:
        checks["synthetic_cer"] = measured["synthetic_cer"] <= rules["synthetic_cer_max"]
    if "box_iou_min" in rules:
        checks["box_iou"] = measured["box_iou"] >= rules["box_iou_min"]
    if "vision_gap_min" in rules:
        checks["vision_gap"] = measured["vision_gap"] >= rules["vision_gap_min"]
    if "vision_win_min" in rules:
        checks["vision_win"] = measured["vision_win"] >= rules["vision_win_min"]
    return {"passed": bool(checks) and all(checks.values()),
            "checks": checks, "measured": measured, "rules": rules}


def run_eval(raw, descriptors, cfg, device, fonts, corpus, cam, stage=None):
    evaluation = eval_read_v3(raw, descriptors, cfg, device, fonts, corpus, cam,
                              batch_size=2, amp=True)
    vision = eval_vision_gain_v3(raw, descriptors, cfg, device, fonts, corpus, cam,
                                 batch_size=3, amp=True)
    gate = gate_report(stage, evaluation, vision) if stage and stage["gate"] else None
    return {"read": evaluation, "vision_control": vision, "gate": gate}


def main():
    args = parse_args()
    rank, local_rank, world, device = setup_distributed(args)
    torch.set_float32_matmul_precision("high")
    seed = args.seed + 1009 * rank
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    out = Path(args.output)
    if rank == 0:
        out.mkdir(parents=True, exist_ok=True)
    barrier(world)

    cfg = reader_scale_v3()
    if args.quick:
        # Preserve every v3 code path while making a CPU/DDP smoke test cheap.
        cfg.d_model = 64; cfg.n_layers = 2; cfg.n_heads = 2; cfg.head_dim = 32
        cfg.ffn_hidden = 96; cfg.image_size = 192
        cfg.image_height = 256; cfg.image_width = 192
        cfg.patch = 16; cfg.patch_merge = 2
        cfg.vision_local_layers = 1; cfg.vision_local_heads = 4; cfg.vision_window = 4
        cfg.vision_grad_checkpoint = False
        cfg.reasoner_moe = True; cfg.moe_experts = 4; cfg.moe_topk = 2
    fonts = find_fonts()
    if not fonts:
        raise RuntimeError("No Vietnamese-capable TrueType font found")
    rank0(rank, f"devices={world} device={device} fonts={len(fonts)}")

    corpus = load_corpus_lines(args.jsonl, limit=24000)
    pairs = load_doc_pairs(args.jsonl, args.images,
                           max_answer_bytes=300, limit=args.limit_pairs,
                           log=(lambda x: rank0(rank, x)))
    pairs = curate_doc_pairs(pairs, cfg)
    splits = split_by_document(pairs, seed=args.seed, val_pct=3, test_pct=3)
    cam = CameraParams(noise_std=4.0, blur_sigma=0.55, jpeg_q=68,
                       contrast=1.04, wb_shift=4.0, vignette=0.08)
    if args.quick:
        cam = None

    valid_report = None
    if rank == 0:
        valid_report = validate_read_v3_data(cfg, fonts, corpus, splits, cam)
        write_json(valid_report, out / "dataset_validation_v3.json")
        print(json.dumps(valid_report, ensure_ascii=False, indent=2), flush=True)
    data_valid = broadcast_bool(bool(valid_report and valid_report["valid"]), device, world)
    if not data_valid:
        raise RuntimeError("Dataset hypotheses failed; see dataset_validation_v3.json")

    gate_eval = build_v3_eval(cfg, splits["val"], n_each=5, n_doc=18, seed=6060)
    final_eval = build_v3_eval(cfg, splits["test"], n_each=10, n_doc=50, seed=9090)

    raw = JWM(cfg).to(device)
    raw.freeze_generator_for_reader()
    rank0(rank, f"params total={sum(p.numel() for p in raw.parameters())/1e6:.2f}M "
                f"trainable={sum(p.numel() for p in raw.parameters() if p.requires_grad)/1e6:.2f}M "
                f"global_batch={args.per_gpu_batch * world * args.grad_accum}")
    resume_path = out / "jwm_read_v3_resume.pt" if args.resume == "auto" else Path(args.resume)
    state = load_resume(resume_path, raw, rank)

    ddp = DDP(raw, device_ids=[local_rank] if device.type == "cuda" else None,
              output_device=local_rank if device.type == "cuda" else None,
              find_unused_parameters=True) if world > 1 else raw
    # CTC starts with a relatively large loss.  The AMP default scale (65536)
    # overflows its first backward pass on T4; 1024 is finite in the gradient
    # audit and still provides ample FP16 dynamic range.
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda",
                                      init_scale=1024.0, growth_interval=500)
    except AttributeError:  # PyTorch < 2.3 compatibility
        scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda",
                                            init_scale=1024.0, growth_interval=500)
    history = state["history"]
    global_step = int(state["global_step"])
    blocked = None

    stages = STAGES
    if args.quick:
        stages = [{**s, "steps": 2, "max_extensions": 0, "gate": {}}
                  for s in STAGES[:1]]

    for stage_idx in range(int(state["stage_idx"]), len(stages)):
        stage = stages[stage_idx]
        cfg.reader_ctc_weight = stage["ctc"]
        cfg.reader_box_weight = stage["box"]
        cfg.vision_contrast_alpha = stage["contrast"]
        optimizer = torch.optim.AdamW(
            [p for p in raw.parameters() if p.requires_grad],
            lr=stage["lr"], betas=(0.9, 0.95), weight_decay=0.05)
        done = int(state["step_in_stage"]) if stage_idx == state["stage_idx"] else 0
        if stage_idx == state["stage_idx"] and done > 0 and state.get("optimizer"):
            optimizer.load_state_dict(state["optimizer"])
            if state.get("scaler"):
                scaler.load_state_dict(state["scaler"])
        extension = max(0, math.ceil(max(0, done - stage["steps"]) / EXTENSION_STEPS))

        batcher = ReadV3Batcher(
            cfg, splits["train"] if stage["synth"] < 1 else [], fonts, corpus, cam,
            stage["synth"], stage["levels"], stage["random"],
            seed=args.seed + 10000 * stage_idx + rank)
        prefetch = PrefetchReadV3(batcher, args.per_gpu_batch,
                                  depth=args.workers_prefetch)
        stage_start = time.time()

        while True:
            target = stage["steps"] + extension * EXTENSION_STEPS
            rank0(rank, f"\n=== {stage['name']} {done}/{target} lr={stage['lr']} "
                        f"synth={stage['synth']} levels={stage['levels']} ===")
            ddp.train()
            while done < target:
                optimizer.zero_grad(set_to_none=True)
                metric_sum = {}
                for micro in range(args.grad_accum):
                    batch = prefetch.batch(device)
                    sync = micro == args.grad_accum - 1
                    ctx = nullcontext() if sync or world == 1 else ddp.no_sync()
                    with ctx:
                        with torch.autocast("cuda", dtype=torch.float16,
                                            enabled=device.type == "cuda"):
                            loss, metrics = ddp("read_v3", *batch)
                            scaled_loss = loss / args.grad_accum
                        scaler.scale(scaled_loss).backward()
                    for k, v in metrics.items():
                        metric_sum[k] = metric_sum.get(k, 0.0) + float(v) / args.grad_accum
                    metric_sum["loss"] = metric_sum.get("loss", 0.0) + \
                        float(loss.detach()) / args.grad_accum
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    [p for p in raw.parameters() if p.requires_grad], 1.0)
                scaler.step(optimizer); scaler.update()
                done += 1; global_step += 1
                lr = cosine_lr(stage["lr"], done - 1, target)
                for group in optimizer.param_groups:
                    group["lr"] = lr

                if done == 1 or done % args.log_every == 0 or done == target:
                    metric_sum["grad_norm"] = float(grad_norm)
                    metric_sum = reduce_metrics(metric_sum, device, world)
                    if rank == 0:
                        elapsed = time.time() - stage_start
                        ips = done / max(1e-6, elapsed)
                        row = {"stage": stage["name"], "step": done,
                               "global_step": global_step, "lr": lr,
                               "steps_per_sec": ips, **metric_sum}
                        history.append(row)
                        fields = " ".join(f"{k}={v:.4f}" for k, v in metric_sum.items())
                        print(f"[{done:5d}/{target}] {fields} lr={lr:.2e} "
                              f"{ips:.2f} opt-step/s", flush=True)

                if done % args.checkpoint_every == 0 or done == target:
                    if rank == 0:
                        payload = checkpoint_payload(raw, optimizer, scaler,
                                                     stage_idx, done, global_step,
                                                     history, cfg)
                        atomic_torch_save(payload, resume_path)
                        write_json(history, out / "history_read_v3.json")
                    barrier(world)

            ddp.eval(); barrier(world)
            report = None
            if rank == 0:
                report = run_eval(raw, gate_eval, cfg, device, fonts, corpus, cam,
                                  stage=stage)
                write_json(report, out / f"metrics_{stage['name']}.json")
                print(json.dumps(report["gate"], ensure_ascii=False, indent=2), flush=True)
            passed = broadcast_bool(bool(report and (report["gate"] is None or
                                                      report["gate"]["passed"])),
                                    device, world)
            if passed:
                break
            if extension >= stage["max_extensions"]:
                blocked = {"status": "blocked_by_metric_gate", "stage": stage["name"],
                           "step": done, "message": "Stage was not promoted; inspect metrics."}
                break
            extension += 1
            rank0(rank, f"GATE FAILED: extending {stage['name']} by {EXTENSION_STEPS} steps")
            ddp.train(); barrier(world)

        prefetch.stop()
        if blocked:
            if rank == 0:
                payload = checkpoint_payload(raw, optimizer, scaler, stage_idx, done,
                                             global_step, history, cfg)
                atomic_torch_save(payload, out / "jwm_read_v3_blocked.pt")
                write_json(blocked, out / "training_status_v3.json")
            break
        state = {"stage_idx": stage_idx + 1, "step_in_stage": 0,
                 "global_step": global_step, "history": history,
                 "optimizer": None, "scaler": None}
        if rank == 0:
            payload = checkpoint_payload(raw, optimizer, scaler, stage_idx + 1, 0,
                                         global_step, history, cfg)
            atomic_torch_save(payload, resume_path)
        barrier(world)

    if not blocked:
        final = None
        if rank == 0:
            final = run_eval(raw, final_eval, cfg, device, fonts, corpus, cam)
            deploy = {"version": "jwm-read-v3", "cfg": asdict(cfg),
                      "model": raw.state_dict(), "metrics": final}
            atomic_torch_save(deploy, out / "jwm_read_v3.pt")
            write_json(final, out / "metrics_read_v3.json")
            write_json({"status": "complete", "global_step": global_step,
                        "world_size": world}, out / "training_status_v3.json")
            write_json(history, out / "history_read_v3.json")
            print("TRAINING COMPLETE -> jwm_read_v3.pt", flush=True)
        barrier(world)

    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
