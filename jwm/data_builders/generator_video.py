# -*- coding: utf-8 -*-
"""Generator branch, type 2 — VIDEO data: forward-dynamics frame pairs.

Cosmos analog (§3.2.1 video + §3.2.3 causality): physics-consistent motion,
the temporal half of the world model. 'post' tier = cleaner, larger-motion
pairs for the g4 image-to-video post-training stage.

(This slot replaces Cosmos' AUDIO type — JWM has no audio modality by design;
JARVIS speech I/O lives in core/speech.py and core/voice.py.)"""

from __future__ import annotations

import hashlib
import json
import random

import numpy as np
import torch

from ..sdg import make_fd
from .common import background_picker, get_camera, save_dataset, train_crops

N_FD = 6000
SEED = 3001
REAL_BG = 0.5
POST_SPEED = 0.15


def build(log=print) -> dict:
    cam = get_camera(log)
    rng = random.Random(SEED)
    crops = train_crops(rng)
    pick_bg = background_picker(rng, crops, REAL_BG)
    seen: set[str] = set()
    d = {"img": [], "img1": [], "q": [], "meta": []}
    while len(d["img"]) < N_FD:
        img_t, img_t1, text, meta = make_fd(rng, cam, pick_bg())
        h = hashlib.md5((text + json.dumps(meta)).encode()).hexdigest()[:16]
        if h in seen:
            continue
        seen.add(h)
        d["img"].append(np.asarray(img_t, dtype=np.uint8))
        d["img1"].append(np.asarray(img_t1, dtype=np.uint8))
        d["q"].append(text)
        d["meta"].append(meta)
        if len(d["img"]) % 1500 == 0:
            log(f"  generator_video: {len(d['img'])}/{N_FD}")
    speeds = sorted(m["speed"] for m in d["meta"])
    cutoff = max(POST_SPEED, speeds[int(len(speeds) * 0.70)])   # top ~30% motion
    post_idx = [i for i, m in enumerate(d["meta"]) if m["speed"] >= cutoff]
    payload = {"fd": {"img": torch.from_numpy(np.stack(d["img"])),
                      "img1": torch.from_numpy(np.stack(d["img1"])),
                      "q": d["q"], "meta": d["meta"]},
               "post_idx": post_idx}
    save_dataset("generator_video", payload,
                 {"branch": "generator", "type": "video(FD)", "n": N_FD,
                  "post_tier": len(post_idx), "post_speed": POST_SPEED, "seed": SEED}, log)
    return payload


if __name__ == "__main__":
    build()
