# -*- coding: utf-8 -*-
"""Reasoner branch, type 2 — SUPERVISED FINE-TUNING data: Physical-AI hard QA.

Cosmos analog (§3.1.2): the specialization mixture — strict curation
(threshold-5 spirit), harder scenes (occlusion, distractors, small objects),
real-background heavy, emphasis on spatial questions (where/count/referring),
mirroring the 2D-grounding + spatial-QA emphasis of the paper.
"""

from __future__ import annotations

import random

import numpy as np
import torch

from ..sdg import judge, make_qa, place_objects, quality_score, render_scene, scene_hash, structured_caption
from .common import background_picker, get_camera, save_dataset, train_crops

N_QA = 12000          # 4500 -> 12000, same generalization-gap fix as pretrain
SEED = 1002
REAL_BG = 0.75
SPATIAL_KINDS = ("where", "count")


def build(log=print) -> dict:
    cam = get_camera(log)
    rng = random.Random(SEED)
    crops = train_crops(rng)
    pick_bg = background_picker(rng, crops, REAL_BG)
    seen: set[str] = set()
    rejected = {"dedup": 0, "judge": 0, "template": 0, "quality": 0}
    qa = {"img": [], "q": [], "a": [], "meta": []}
    spatial_target = int(N_QA * 0.5)          # spatial emphasis, like Cosmos SFT
    n_spatial = 0
    while len(qa["img"]) < N_QA:
        held = rng.random() < 0.25
        objs = place_objects(rng, 1 if held else rng.randint(2, 5), held, hard=True)
        if not objs:
            continue
        out = make_qa(objs, rng, held)
        if out is None:
            rejected["template"] += 1
            continue
        q, a, target, kind = out
        if kind in SPATIAL_KINDS and n_spatial >= spatial_target and rng.random() < 0.5:
            continue                            # keep the mixture balanced
        ok, _ = judge(objs, target, kind)
        if not ok:
            rejected["judge"] += 1
            continue
        if quality_score(objs) < 0.30:          # strict tier: drop the worst scenes
            rejected["quality"] += 1
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
        n_spatial += int(kind in SPATIAL_KINDS)
        if len(qa["img"]) % 1000 == 0:
            log(f"  reasoner_sft: {len(qa['img'])}/{N_QA}")
    payload = {"qa": {"img": torch.from_numpy(np.stack(qa["img"])), "q": qa["q"],
                      "a": qa["a"], "meta": qa["meta"]}}
    save_dataset("reasoner_sft", payload,
                 {"branch": "reasoner", "type": "sft", "n_qa": N_QA, "real_bg": REAL_BG,
                  "spatial_frac": round(n_spatial / N_QA, 3), "rejected": rejected,
                  "seed": SEED}, log)
    return payload


if __name__ == "__main__":
    build()
