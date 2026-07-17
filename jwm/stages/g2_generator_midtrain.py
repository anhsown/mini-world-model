# -*- coding: utf-8 -*-
"""STAGE G2 — Generator MID-TRAINING (Cosmos §4.2.2).

The ACTION modality enters here (exactly like Cosmos: 'Action and video
transfer data are first introduced during mid-training'), mixed with the
visual modes so existing capabilities stay active. bbox flow + calibrated
confidence-head BCE begin training."""

from __future__ import annotations

import argparse

from ..trainer import train_stage, eval_ground, eval_t2i, eval_fd
from .common import DEV, StageRunner, load_data, load_valtest

NAME = "g2_generator_midtrain"
PREV = "g1_generator_pretrain"
STEPS, LR, BATCH, WARMUP = 2200, 2e-4, 48, 100
MODE_PROBS = {"ground": 0.5, "t2i": 0.25, "fd": 0.25}   # action enters at 50%


def run(force: bool = False, log=print) -> dict:
    r = StageRunner(NAME, PREV, log)
    if r.output_exists() and not force:
        log(f"  [{NAME}] skip (done)")
        return {}
    model, ae, cam, flags, done = r.load()
    split = load_data("generator_image", "generator_video", "generator_action")
    if done < STEPS:
        train_stage(model, ae, split, r.cfg, DEV, steps=STEPS - done, lr=LR,
                    batch_size=BATCH, warmup=max(20, WARMUP - done),
                    mode_probs=MODE_PROBS, seed=400 + done, log_every=100, log=log,
                    ckpt_fn=r.ckpt_fn(model, ae, cam, flags, done), ckpt_every=500)
    val, _ = load_valtest()
    gr = eval_ground(model, ae, val, r.cfg, DEV, n=150, steps=4, log=log)
    t2i = eval_t2i(model, ae, val, r.cfg, DEV, n=24, steps=20, log=log)
    fd = eval_fd(model, ae, val, r.cfg, DEV, n=24, k=2, steps=20, log=log)
    r.finish(model, ae, cam, flags,
             {"steps": STEPS, "lr": LR,
              "val_miou_4step": gr["miou"], "val_iou05_4step": gr["iou_at_05"],
              "val_t2i_pos": t2i["t2i_self_consistency_pos"],
              "val_fd_psnr": fd["fd_psnr_min_over_k"]})
    return {"ground": gr}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    run(force=ap.parse_args().force)
