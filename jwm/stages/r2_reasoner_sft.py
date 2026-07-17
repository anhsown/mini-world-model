# -*- coding: utf-8 -*-
"""STAGE R2 — Reasoner SUPERVISED FINE-TUNING (Cosmos §4.1.2).

Specializes the reasoner on hard Physical-AI-style QA (spatial emphasis).
Mixes a slice of pretrain data back in at 1:4 — the paper's anti-forgetting
ratio — then evaluates per-kind accuracy."""

from __future__ import annotations

import argparse
import random

import torch

from ..trainer import train_stage, eval_qa
from .common import DEV, StageRunner, load_data, load_valtest, subset

NAME = "r2_reasoner_sft"
PREV = "r1_reasoner_pretrain"
# round 3 (28M proven scale)
STEPS, LR, BATCH, WARMUP = 800, 1.2e-4, 48, 50
MODE_PROBS = {"qa": 1.0}
PRETRAIN_MIX = 0.2                      # 1:4 pretrain-to-SFT (Cosmos §4.1.2)


def run(force: bool = False, log=print) -> dict:
    r = StageRunner(NAME, PREV, log)
    if r.output_exists() and not force:
        log(f"  [{NAME}] skip (done)")
        return {}
    model, ae, cam, flags, done = r.load()
    sft = load_data("reasoner_sft")["qa"]
    pre = load_data("reasoner_pretrain")["qa"]
    rng = random.Random(0)
    n_mix = int(len(sft["q"]) * PRETRAIN_MIX / (1 - PRETRAIN_MIX))
    pre_mix = subset(pre, rng.sample(range(len(pre["q"])), n_mix))
    merged = {k: (torch.cat([sft[k], pre_mix[k]]) if torch.is_tensor(sft[k])
                  else sft[k] + pre_mix[k]) for k in sft}
    split = {"qa": merged}
    if done < STEPS:
        train_stage(model, ae, split, r.cfg, DEV, steps=STEPS - done, lr=LR,
                    batch_size=BATCH, warmup=max(10, WARMUP - done),
                    mode_probs=MODE_PROBS, seed=200 + done, log_every=100, log=log,
                    ckpt_fn=r.ckpt_fn(model, ae, cam, flags, done), ckpt_every=250)
    val, _ = load_valtest()
    qa = eval_qa(model, val, r.cfg, DEV, n=250, log=log)
    r.finish(model, ae, cam, flags,
             {"steps": STEPS, "lr": LR, "pretrain_mix": PRETRAIN_MIX,
              "val_qa_acc": qa["acc"], "val_qa_per_kind": qa["per_kind"]})
    return qa


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    run(force=ap.parse_args().force)
