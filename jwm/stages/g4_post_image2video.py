# -*- coding: utf-8 -*-
"""STAGE G4 — IMAGE-TO-VIDEO POST-TRAINING (Cosmos §4.2.4).

FD (frame -> next frame) specialization on the larger-motion post tier —
the predictive world-model capability that matters for embodied planning.
Evaluated with min-over-k PSNR vs the copy-last-frame baseline."""

from __future__ import annotations

import argparse

from ..trainer import train_stage, eval_fd
from .common import DEV, StageRunner, load_data, load_valtest, subset

NAME = "g4_post_image2video"
PREV = "g3_post_text2image"
STEPS, LR, BATCH, WARMUP = 400, 8e-5, 48, 40
MODE_PROBS = {"fd": 0.9, "t2i": 0.05, "ground": 0.05}


def run(force: bool = False, log=print) -> dict:
    r = StageRunner(NAME, PREV, log)
    if r.output_exists() and not force:
        log(f"  [{NAME}] skip (done)")
        return {}
    model, ae, cam, flags, done = r.load()
    data = load_data("generator_image", "generator_video", "generator_action")
    post = subset(data["fd"], data["generator_video:post_idx"])
    log(f"  [g4] post-tier FD: {len(post['q'])} larger-motion pairs")
    split = {"fd": post, "t2i": data["t2i"], "ground": data["ground"]}
    if done < STEPS:
        train_stage(model, ae, split, r.cfg, DEV, steps=STEPS - done, lr=LR,
                    batch_size=BATCH, warmup=max(10, WARMUP - done),
                    mode_probs=MODE_PROBS, seed=600 + done, log_every=100, log=log,
                    ckpt_fn=r.ckpt_fn(model, ae, cam, flags, done), ckpt_every=200)
    val, _ = load_valtest()
    fd = eval_fd(model, ae, val, r.cfg, DEV, n=48, k=4, steps=50, log=log)
    r.finish(model, ae, cam, flags,
             {"steps": STEPS, "lr": LR, "post_tier_n": len(post["q"]),
              "val_fd_psnr_min_over_k": fd["fd_psnr_min_over_k"],
              "val_fd_copy_baseline": fd["fd_psnr_copy_baseline"],
              "val_fd_beats_copy": fd["beats_copy_frac"]})
    return fd


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    run(force=ap.parse_args().force)
