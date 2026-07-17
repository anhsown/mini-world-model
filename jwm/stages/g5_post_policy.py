# -*- coding: utf-8 -*-
"""STAGE G5 — ROBOT POLICY POST-TRAINING (Cosmos §4.2.5).

The deployment stage, mirroring Cosmos3-Nano-Policy-DROID: specialize the
action head on the strict post tier, optimize for the FAST sampler (4 steps,
like the paper's policy mode), fit Platt calibration on val, run the FULL test
battery, and emit the deployable brain: jwm/checkpoints/jwm_v3.pt +
metrics_v3.json (picked up automatically by core/world_brain.py)."""

from __future__ import annotations

import argparse
import json

import torch

from ..mathx import expected_calibration_error
from ..trainer import (train_stage, eval_ground, eval_qa, eval_fd, eval_t2i,
                       measure_latency)
from .common import CKPT_DIR, DEV, StageRunner, load_data, load_valtest, subset

NAME = "g5_post_policy"
PREV = "g4_post_image2video"
STEPS, LR, BATCH, WARMUP = 700, 8e-5, 48, 50
MODE_PROBS = {"ground": 1.0}
POLICY_STEPS = 4                        # fast-sampler deployment target


def _platt(conf: torch.Tensor, y: torch.Tensor) -> dict:
    conf = conf.clamp(1e-4, 1 - 1e-4)
    logit = torch.log(conf / (1 - conf))
    a = torch.tensor(1.0, requires_grad=True)
    b = torch.tensor(0.0, requires_grad=True)
    opt = torch.optim.LBFGS([a, b], lr=0.1, max_iter=200)

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            a * logit + b, y.float())
        loss.backward()
        return loss

    opt.step(closure)
    return {"a": round(float(a), 4), "b": round(float(b), 4), "steps": POLICY_STEPS}


def _cal(cal, c):
    c = c.clamp(1e-4, 1 - 1e-4)
    return torch.sigmoid(cal["a"] * torch.log(c / (1 - c)) + cal["b"])


def run(force: bool = False, log=print) -> dict:
    r = StageRunner(NAME, PREV, log)
    if r.output_exists() and not force:
        log(f"  [{NAME}] skip (done)")
        return {}
    model, ae, cam, flags, done = r.load()
    data = load_data("generator_action")
    post = subset(data["ground"], data["generator_action:post_idx"])
    log(f"  [g5] post-tier ACTION: {len(post['q'])} hard curated samples")
    split = {"ground": post}
    if done < STEPS:
        train_stage(model, ae, split, r.cfg, DEV, steps=STEPS - done, lr=LR,
                    batch_size=BATCH, warmup=max(10, WARMUP - done),
                    mode_probs=MODE_PROBS, seed=700 + done, log_every=100, log=log,
                    ckpt_fn=r.ckpt_fn(model, ae, cam, flags, done), ckpt_every=250)

    val, test = load_valtest()
    # calibration on val at the deployment step count
    gr_v = eval_ground(model, ae, val, r.cfg, DEV, n=250, steps=POLICY_STEPS, log=log)
    cal = _platt(torch.tensor(gr_v["conf_all"]), torch.tensor(gr_v["iou_all"]) >= 0.5)
    ece_v = expected_calibration_error(_cal(cal, torch.tensor(gr_v["conf_all"])),
                                       torch.tensor(gr_v["iou_all"]) >= 0.5)
    log(f"  [g5] Platt a={cal['a']} b={cal['b']} | val ECE after: {ece_v:.4f}")

    # FULL battery on test (comparable to v1/v2)
    log("  [g5] final TEST battery:")
    qa_t = eval_qa(model, test, r.cfg, DEV, n=250, log=log)
    gr_t = eval_ground(model, ae, test, r.cfg, DEV, n=250, steps=POLICY_STEPS, log=log)
    gr_t50 = eval_ground(model, ae, test, r.cfg, DEV, n=250, steps=50, log=log)
    fd_t = eval_fd(model, ae, test, r.cfg, DEV, n=48, k=4, steps=50, log=log)
    t2i_t = eval_t2i(model, ae, test, r.cfg, DEV, n=48, steps=50, log=log)
    lat = measure_latency(model, ae, val, r.cfg, DEV, reps=10, log=log)
    ece_t = expected_calibration_error(_cal(cal, torch.tensor(gr_t["conf_all"])),
                                       torch.tensor(gr_t["iou_all"]) >= 0.5)

    metrics = {
        "test": {"qa_acc": qa_t["acc"], "qa_per_kind": qa_t["per_kind"],
                 "iou_at_05_4step": gr_t["iou_at_05"], "miou_4step": gr_t["miou"],
                 "iou_at_05_50step": gr_t50["iou_at_05"], "miou_50step": gr_t50["miou"],
                 "ece_calibrated_4step": ece_t,
                 "fd_psnr_min_over_k": fd_t["fd_psnr_min_over_k"],
                 "fd_copy_baseline": fd_t["fd_psnr_copy_baseline"],
                 "fd_beats_copy": fd_t["beats_copy_frac"],
                 "t2i_self_consistency_pos": t2i_t["t2i_self_consistency_pos"],
                 "t2i_self_consistency_neg": t2i_t["t2i_self_consistency_neg"]},
        "val": {"ece_calibrated": ece_v, "miou_4step": gr_v["miou"]},
        "calibration": cal,
        "latency_ms": lat,
        "params_M": round(model.num_params() / 1e6, 2),
        "pipeline": "7-stage (r1,r2,g1..g5)",
        "reference": {"v1_10.7M": {"qa": 0.564, "miou": 0.201, "ece": 0.0396},
                      "v2_28M": "superseded mid-training"},
    }
    # deployable brain artifact (v4 = MoE-reasoner generation, v3 = dense)
    gen = "jwm_v4" if getattr(r.cfg, "reasoner_moe", False) else "jwm_v3"
    torch.save({"model": model.state_dict(), "ae": ae.state_dict(),
                "cfg": r.cfg.__dict__, "camera": cam.to_dict(),
                "calibration": cal, "metrics": metrics, "flags": flags},
               CKPT_DIR / f"{gen}.pt")
    json.dump(metrics, (CKPT_DIR / f"metrics_{gen.split('_')[1]}.json").open("w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    log(f"  [g5] deployable brain -> {gen}.pt")
    r.finish(model, ae, cam, flags,
             {"steps": STEPS, "lr": LR, "policy_steps": POLICY_STEPS,
              "val_ece_calibrated": ece_v, "test_qa_acc": qa_t["acc"],
              "test_miou_4step": gr_t["miou"]},
             extra={"calibration": cal})
    return metrics


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    run(force=ap.parse_args().force)
