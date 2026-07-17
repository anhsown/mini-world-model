# -*- coding: utf-8 -*-
"""STAGE R1 — Reasoner PRE-TRAINING (Cosmos §4.1.1).

Fresh model; trains ONLY the AR pathway (qa mode -> gradients reach the
reasoner tower, embeddings and lm_head; the generator tower is untouched).
Broad QA data, sqrt-normalized CE."""

from __future__ import annotations

import argparse

from ..trainer import train_stage, eval_qa
from .common import DEV, StageRunner, load_data, load_valtest

NAME = "r1_reasoner_pretrain"
PREV = None
# round 3 (28M proven scale): v2-equivalent budget @ batch 48, with the
# shape/color attribute curriculum in the rebuilt data
STEPS, LR, BATCH, WARMUP = 3000, 3e-4, 48, 150
MODE_PROBS = {"qa": 1.0}
DATA = ["reasoner_pretrain"]


def run(force: bool = False, log=print) -> dict:
    r = StageRunner(NAME, PREV, log)
    if r.output_exists() and not force:
        log(f"  [{NAME}] skip (done)")
        return {}
    model, ae, cam, flags, done = r.load()
    split = load_data(*DATA)
    if done < STEPS:
        train_stage(model, ae, split, r.cfg, DEV, steps=STEPS - done, lr=LR,
                    batch_size=BATCH, warmup=max(20, WARMUP - done),
                    mode_probs=MODE_PROBS, seed=100 + done, log_every=100, log=log,
                    ckpt_fn=r.ckpt_fn(model, ae, cam, flags, done), ckpt_every=500)
    val, _ = load_valtest()
    qa = eval_qa(model, val, r.cfg, DEV, n=200, log=log)
    r.finish(model, ae, cam, flags,
             {"steps": STEPS, "lr": LR, "val_qa_acc": qa["acc"],
              "val_qa_per_kind": qa["per_kind"]})
    return qa


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    run(force=ap.parse_args().force)
