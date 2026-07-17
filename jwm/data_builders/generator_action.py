# -*- coding: utf-8 -*-
"""Generator branch, type 3 — ACTION data: referring-expression grounding (bbox).

Cosmos analog (§3.2.3): the action modality connecting language to spatial
intervention. bbox is JWM's action token. 'post' tier = strict-quality hard
scenes for the g5 policy post-training stage (the deployable brain)."""

from __future__ import annotations

import random

import numpy as np
import torch

from ..sdg import judge, make_ground, obj_bbox, place_objects, quality_score, render_scene, scene_hash, structured_caption
from .common import background_picker, get_camera, save_dataset, train_crops

N_GROUND = 14000
SEED = 4001
REAL_BG = 0.55
POST_QUALITY = 0.62


def build(log=print) -> dict:
    cam = get_camera(log)
    rng = random.Random(SEED)
    crops = train_crops(rng)
    pick_bg = background_picker(rng, crops, REAL_BG)
    seen: set[str] = set()
    rejected = {"dedup": 0, "judge": 0, "template": 0}
    d = {"img": [], "q": [], "bbox": [], "quality": [], "meta": []}
    while len(d["img"]) < N_GROUND:
        hard = rng.random() < 0.5
        held = rng.random() < 0.2
        objs = place_objects(rng, 1 if held else rng.randint(2, 4), held, hard)
        if not objs:
            continue
        out = make_ground(objs, rng)
        if out is None:
            rejected["template"] += 1
            continue
        q, target = out
        ok, _ = judge(objs, target, "ground")
        if not ok:
            rejected["judge"] += 1
            continue
        h = scene_hash(objs, q)
        if h in seen:
            rejected["dedup"] += 1
            continue
        seen.add(h)
        d["img"].append(np.asarray(render_scene(objs, pick_bg(), cam, rng), dtype=np.uint8))
        d["q"].append(q)
        d["bbox"].append(list(obj_bbox(target)))
        d["quality"].append(quality_score(objs))
        d["meta"].append({"kind": "ground", "hard": hard, "caption": structured_caption(objs)})
        if len(d["img"]) % 2000 == 0:
            log(f"  generator_action: {len(d['img'])}/{N_GROUND}")
    quality = torch.tensor(d["quality"])
    hard_q = torch.tensor([d["quality"][i] for i in range(N_GROUND) if d["meta"][i]["hard"]])
    cutoff = max(POST_QUALITY, float(torch.quantile(hard_q, 0.65)))  # top ~35% of hard
    post_idx = [i for i in range(N_GROUND)
                if d["quality"][i] >= cutoff and d["meta"][i]["hard"]]
    payload = {"ground": {"img": torch.from_numpy(np.stack(d["img"])), "q": d["q"],
                          "bbox": torch.tensor(d["bbox"], dtype=torch.float32),
                          "quality": quality, "meta": d["meta"]},
               "post_idx": post_idx}
    save_dataset("generator_action", payload,
                 {"branch": "generator", "type": "action(bbox)", "n": N_GROUND,
                  "post_tier": len(post_idx), "post_threshold": POST_QUALITY,
                  "rejected": rejected, "seed": SEED}, log)
    return payload


if __name__ == "__main__":
    build()
