# -*- coding: utf-8 -*-
"""Reasoner branch, type 1 — PRE-TRAINING data: broad visual QA.

Cosmos analog (§3.1.1): the wide-coverage image-text mixture that gives the
reasoner general understanding before specialization. Loose curation
(threshold-2 spirit): easy + medium scenes, synthetic-background heavy.
"""

from __future__ import annotations

import random

import numpy as np
import torch

from ..sdg import judge, make_qa, place_objects, render_scene, scene_hash, structured_caption
from .common import background_picker, get_camera, save_dataset, train_crops

# 14000 -> 40000: the 68M reasoner showed a train/val generalization gap
# (train tok-acc 0.956, val exact-match stuck ~37%) — unique-sample count was
# the binding constraint, and procedural data is nearly free.
N_QA = 40000
SEED = 1001
REAL_BG = 0.45


def build(log=print) -> dict:
    cam = get_camera(log)
    rng = random.Random(SEED)
    crops = train_crops(rng)
    pick_bg = background_picker(rng, crops, REAL_BG)
    seen: set[str] = set()
    rejected = {"dedup": 0, "judge": 0, "template": 0}
    qa = {"img": [], "q": [], "a": [], "meta": []}
    while len(qa["img"]) < N_QA:
        held = rng.random() < 0.35
        objs = place_objects(rng, 1 if held else rng.randint(1, 4), held, hard=False)
        if not objs:
            continue
        out = make_qa(objs, rng, held)
        if out is None:
            rejected["template"] += 1
            continue
        q, a, target, kind = out
        ok, _ = judge(objs, target, kind)
        if not ok:
            rejected["judge"] += 1
            continue
        h = scene_hash(objs, q)
        if h in seen:
            rejected["dedup"] += 1
            continue
        seen.add(h)
        qa["img"].append(np.asarray(render_scene(objs, pick_bg(), cam, rng), dtype=np.uint8))
        qa["q"].append(q)
        qa["a"].append(a)
        qa["meta"].append({"kind": kind, "caption": structured_caption(objs)})
        if len(qa["img"]) % 2000 == 0:
            log(f"  reasoner_pretrain: {len(qa['img'])}/{N_QA}")
    payload = {"qa": {"img": torch.from_numpy(np.stack(qa["img"])), "q": qa["q"],
                      "a": qa["a"], "meta": qa["meta"]}}
    save_dataset("reasoner_pretrain", payload,
                 {"branch": "reasoner", "type": "pretrain", "n_qa": N_QA,
                  "real_bg": REAL_BG, "rejected": rejected, "seed": SEED}, log)
    return payload


if __name__ == "__main__":
    build()
