# -*- coding: utf-8 -*-
"""STAGE G3 — TEXT-TO-IMAGE POST-TRAINING (Cosmos §4.2.3).

Compact, ultra-curated specialization: trains mostly on the strict-quality
post tier of the image data (Cosmos: 'post-training image corpus is a compact,
carefully curated set'), with small regularizer slices of the other modes."""

from __future__ import annotations

import argparse

from ..trainer import train_stage, eval_t2i
from .common import DEV, StageRunner, load_data, load_valtest, subset

NAME = "g3_post_text2image"
PREV = "g2_generator_midtrain"
STEPS, LR, BATCH, WARMUP = 400, 8e-5, 48, 40
MODE_PROBS = {"t2i": 0.9, "fd": 0.05, "ground": 0.05}   # regularizer slices


def run(force: bool = False, log=print) -> dict:
    r = StageRunner(NAME, PREV, log)
    if r.output_exists() and not force:
        log(f"  [{NAME}] skip (done)")
        return {}
    model, ae, cam, flags, done = r.load()
    data = load_data("generator_image", "generator_video", "generator_action")
    post = subset(data["t2i"], data["generator_image:post_idx"])
    log(f"  [g3] post-tier T2I: {len(post['q'])} curated samples")
    split = {"t2i": post, "fd": data["fd"], "ground": data["ground"]}
    if done < STEPS:
        train_stage(model, ae, split, r.cfg, DEV, steps=STEPS - done, lr=LR,
                    batch_size=BATCH, warmup=max(10, WARMUP - done),
                    mode_probs=MODE_PROBS, seed=500 + done, log_every=100, log=log,
                    ckpt_fn=r.ckpt_fn(model, ae, cam, flags, done), ckpt_every=200)
    val, _ = load_valtest()
    t2i = eval_t2i(model, ae, val, r.cfg, DEV, n=48, steps=50, log=log)
    r.finish(model, ae, cam, flags,
             {"steps": STEPS, "lr": LR, "post_tier_n": len(post["q"]),
              "val_t2i_pos": t2i["t2i_self_consistency_pos"],
              "val_t2i_neg": t2i["t2i_self_consistency_neg"],
              "val_t2i_mse": t2i["t2i_latent_mse"]})
    return t2i


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    run(force=ap.parse_args().force)
