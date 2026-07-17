# -*- coding: utf-8 -*-
"""Build the complete 2-branch data corpus + T2I val/test additions.

    python -m jwm.data_builders.build_all [--force]

Legacy val/test (data/jwm_sdg/val.pt, test.pt) are REUSED unchanged for
qa/ground/fd so every model generation stays directly comparable; only the new
T2I evaluation pairs are added here (backgrounds from held-out real frames)."""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from jwm.sdg import load_real_crops, make_t2i  # noqa: E402
from jwm.data_builders import common  # noqa: E402
from jwm.data_builders import (generator_action, generator_image,  # noqa: E402
                               generator_video, reasoner_pretrain, reasoner_sft)


def build_t2i_valtest(cam, log=print):
    frames = common.real_frames()
    preview = str(ROOT / "data" / "vision_preview.jpg")
    packs = {}
    for split, paths, seed, n in (("val", frames[6:7], 5001, 120),
                                  ("test", frames[7:] + [preview], 5002, 120)):
        rng = random.Random(seed)
        crops = load_real_crops(paths, per_frame=60, rng=rng)
        pick = common.background_picker(rng, crops, 0.6)
        d = {"img": [], "q": [], "quality": [], "meta": []}
        while len(d["img"]) < n:
            out = make_t2i(rng, cam, pick())
            if out is None:
                continue
            caption, img, quality, scap = out
            d["img"].append(img)
            d["q"].append(caption)
            d["quality"].append(quality)
            d["meta"].append({"kind": "t2i", "caption": scap})
        packs[split] = {"img": torch.from_numpy(np.stack(d["img"])), "q": d["q"],
                        "quality": torch.tensor(d["quality"]), "meta": d["meta"]}
        log(f"  t2i_{split}: {n}")
    common.save_dataset("t2i_valtest", packs, {"branch": "eval", "type": "t2i val/test",
                                               "n_val": 120, "n_test": 120}, log)


BUILDERS = [
    ("reasoner_pretrain", reasoner_pretrain.build),
    ("reasoner_sft", reasoner_sft.build),
    ("generator_image", generator_image.build),
    ("generator_video", generator_video.build),
    ("generator_action", generator_action.build),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    cam = common.get_camera()
    for name, fn in BUILDERS:
        out = common.OUT_DIR / f"{name}.pt"
        if out.exists() and not args.force:
            print(f"skip {name} (exists; --force to rebuild)")
            continue
        print(f"== building {name} ==")
        fn()
    if not (common.OUT_DIR / "t2i_valtest.pt").exists() or args.force:
        print("== building t2i val/test ==")
        build_t2i_valtest(cam)
    print(f"\nDONE in {(time.time()-t0)/60:.1f} min -> {common.OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
