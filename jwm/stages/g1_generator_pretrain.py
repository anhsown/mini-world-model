# -*- coding: utf-8 -*-
"""STAGE G1 — Generator PRE-TRAINING (Cosmos §4.2.1).

Two Cosmos rituals happen here, once each:
  1. the ConvAE ("frozen VAE") is trained on generator images, then frozen;
  2. the generator tower is INITIALIZED AS A WEIGHT COPY of the trained
     reasoner tower (Cosmos §4: "trained Reasoner weights initialize the
     Generator"), AdaLN zero-gated.
Then broad visual generation: T2I + FD. The reasoner receives no gradients
(no qa mode -> frozen in effect, per the paper)."""

from __future__ import annotations

import argparse

from ..trainer import train_stage, train_convae, eval_t2i, eval_fd
from .common import DEV, StageRunner, load_data, load_valtest

NAME = "g1_generator_pretrain"
PREV = "r2_reasoner_sft"
STEPS, LR, BATCH, WARMUP = 2200, 3e-4, 48, 150
MODE_PROBS = {"t2i": 0.5, "fd": 0.5}


def run(force: bool = False, log=print) -> dict:
    r = StageRunner(NAME, PREV, log)
    if r.output_exists() and not force:
        log(f"  [{NAME}] skip (done)")
        return {}
    model, ae, cam, flags, done = r.load()
    split = load_data("generator_image", "generator_video")

    if not flags.get("ae_trained"):
        log("  [g1] training ConvAE (then FROZEN — Cosmos frozen-VAE role)")
        train_convae(ae, split["t2i"]["img"], DEV, steps=800, bs=64, log_every=200, log=log)
        flags["ae_trained"] = True
    if not flags.get("generator_initialized"):
        log("  [g1] generator tower <- copy of trained reasoner tower (Cosmos §4)")
        model.init_generator_from_reasoner()
        flags["generator_initialized"] = True

    if done < STEPS:
        train_stage(model, ae, split, r.cfg, DEV, steps=STEPS - done, lr=LR,
                    batch_size=BATCH, warmup=max(20, WARMUP - done),
                    mode_probs=MODE_PROBS, seed=300 + done, log_every=100, log=log,
                    ckpt_fn=r.ckpt_fn(model, ae, cam, flags, done), ckpt_every=500)
    val, _ = load_valtest()
    t2i = eval_t2i(model, ae, val, r.cfg, DEV, n=32, steps=20, log=log)
    fd = eval_fd(model, ae, val, r.cfg, DEV, n=32, k=2, steps=20, log=log)
    r.finish(model, ae, cam, flags,
             {"steps": STEPS, "lr": LR,
              "val_t2i_pos": t2i["t2i_self_consistency_pos"],
              "val_t2i_mse": t2i["t2i_latent_mse"],
              "val_fd_psnr": fd["fd_psnr_min_over_k"],
              "val_fd_copy": fd["fd_psnr_copy_baseline"]})
    return {"t2i": t2i, "fd": fd}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    run(force=ap.parse_args().force)
