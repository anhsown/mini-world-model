"""Shared infrastructure for all data builders: real-frame reference, camera
autotune (validated against real JARVIS frames), save helpers, manifest."""

from __future__ import annotations

import glob
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from ..sdg import (CameraParams, autotune_camera, load_real_crops, place_objects,
                   real_reference_images, render_scene, synth_background)

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "data" / "jwm_v3"
CAMERA_FILE = OUT_DIR / "camera.json"
MANIFEST = OUT_DIR / "manifest.json"

# background-frame separation (anti-leakage): train crops never come from the
# frames whose crops feed val/test in the legacy val/test splits
def real_frames() -> list[str]:
    return sorted(glob.glob(str(ROOT / "data/vision_sessions/*/frames/*-raw.jpg")))


def train_crops(rng: random.Random, per_frame: int = 30):
    return load_real_crops(real_frames()[:6], per_frame, rng)


def get_camera(log=print) -> CameraParams:
    """Autotune once, cache to disk; every builder reuses the same validated model."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if CAMERA_FILE.exists():
        payload = json.load(CAMERA_FILE.open(encoding="utf-8"))
        return CameraParams(**payload["camera"])
    rng = random.Random(0)
    crops = train_crops(rng)
    real_ref = real_reference_images(real_frames(), per_frame=40, seed=0)

    def make_batch(cam_p, n):
        r = random.Random(777)
        out = []
        for _ in range(n):
            held = r.random() < 0.4
            objs = place_objects(r, 1 if held else r.randint(1, 4), held, False)
            bg = r.choice(crops) if r.random() < 0.6 else synth_background(r)
            out.append(np.asarray(render_scene(objs, bg, cam_p, r), dtype=np.uint8))
        return out

    cam, report = autotune_camera(real_ref, make_batch, CameraParams(), rounds=8,
                                  batch=200, log=log)
    if not report["all_pass"]:
        raise RuntimeError("camera autotune failed to match real-frame statistics")
    json.dump({"camera": cam.to_dict(),
               "validation": {k: v for k, v in report.items() if isinstance(v, dict)}},
              CAMERA_FILE.open("w", encoding="utf-8"), indent=1)
    log(f"camera autotune PASS -> {CAMERA_FILE.name}")
    return cam


def save_dataset(name: str, payload: dict, meta: dict, log=print) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"{name}.pt"
    torch.save(payload, path)
    manifest = json.load(MANIFEST.open(encoding="utf-8")) if MANIFEST.exists() else {}
    manifest[name] = {**meta, "file": path.name,
                      "built_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    json.dump(manifest, MANIFEST.open("w", encoding="utf-8"), ensure_ascii=False, indent=1)
    log(f"saved {path.name}: {meta}")
    return path


def background_picker(rng: random.Random, crops, real_ratio: float):
    def pick():
        if crops and rng.random() < real_ratio:
            return rng.choice(crops)
        return synth_background(rng)
    return pick
