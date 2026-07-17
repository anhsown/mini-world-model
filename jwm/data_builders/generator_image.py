# -*- coding: utf-8 -*-
"""Generator branch, type 1 — IMAGE data: T2I (caption -> scene image).

Cosmos analog (§3.2.1): the image half of the generator corpus with structured
captions. Per-sample quality_score enables tiering: 'post' indices are the
strict-threshold curated subset used by the g3 T2I post-training stage
(Cosmos post-training corpora are compact, ultra-curated sets)."""

from __future__ import annotations

import random

import numpy as np
import torch

from ..sdg import make_t2i
from .common import background_picker, get_camera, save_dataset, train_crops

N_T2I = 8000
SEED = 2001
REAL_BG = 0.5
POST_QUALITY = 0.62


def build(log=print) -> dict:
    cam = get_camera(log)
    rng = random.Random(SEED)
    crops = train_crops(rng)
    pick_bg = background_picker(rng, crops, REAL_BG)
    seen: set[str] = set()
    d = {"img": [], "q": [], "quality": [], "meta": []}
    while len(d["img"]) < N_T2I:
        out = make_t2i(rng, cam, pick_bg())
        if out is None:
            continue
        caption, img, quality, scap = out
        if caption in seen and rng.random() < 0.8:   # captions repeat legitimately;
            continue                                  # cap exact-duplicate share
        seen.add(caption)
        d["img"].append(img)
        d["q"].append(caption)
        d["quality"].append(quality)
        d["meta"].append({"kind": "t2i", "caption": scap})
        if len(d["img"]) % 2000 == 0:
            log(f"  generator_image: {len(d['img'])}/{N_T2I}")
    quality = torch.tensor(d["quality"])
    # T2I scenes (1-2 large objects) are naturally high-quality, so a fixed
    # threshold barely filters; the post tier is defined as the TOP QUARTILE
    # by quality instead — a genuinely compact curated subset.
    cutoff = torch.quantile(quality, 0.75)
    post_idx = torch.nonzero(quality >= max(float(cutoff), POST_QUALITY)).squeeze(-1).tolist()
    payload = {"t2i": {"img": torch.from_numpy(np.stack(d["img"])), "q": d["q"],
                       "quality": quality, "meta": d["meta"]},
               "post_idx": post_idx}
    save_dataset("generator_image", payload,
                 {"branch": "generator", "type": "image(T2I)", "n": N_T2I,
                  "post_tier": len(post_idx), "post_threshold": POST_QUALITY,
                  "seed": SEED}, log)
    return payload


if __name__ == "__main__":
    build()
