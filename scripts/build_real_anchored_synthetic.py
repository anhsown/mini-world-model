"""Fit a real-anchor profile and validate a large on-the-fly synthetic corpus."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jwm.geometry_v2_data import BonnRGBDWindowDataset, TaggedTUMDataset
from jwm.geometry_v3_data import CalibratedGeometryDataset
from jwm.anchor_discovery import discover_registered_rgbd
from jwm.real_anchored_sdg import (RealAnchoredSyntheticGeometry,
                                   profile_real_geometry,
                                   validate_real_anchored_synthetic)


def args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--anchor-root", type=Path,
                        default=ROOT / "data/real_anchor_v1/raw")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "data/real_anchor_v1/derived")
    parser.add_argument("--registry", type=Path,
                        default=ROOT / "configs/datasets/real_anchor_v1.json")
    parser.add_argument("--samples", type=int, default=250_000)
    parser.add_argument("--frames", type=int, default=6)
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--profile-windows", type=int, default=96)
    return parser.parse_args()


def discover_tum_roots(root: Path) -> tuple[list[Path], list[Path]]:
    found = []
    for rgb in root.rglob("rgb.txt") if root.exists() else []:
        parent = rgb.parent
        if (parent / "depth.txt").exists() and (parent / "groundtruth.txt").exists():
            found.append(parent)
    bonn = [path for path in found if "bonn" in str(path).lower()]
    tum = [path for path in found if path not in bonn]
    return tum, bonn


def save_preview(dataset, path: Path, count: int = 6) -> None:
    tiles = []
    for index in range(min(count, len(dataset))):
        row = dataset[index]
        rgb = (row["image"][0].permute(1, 2, 0).clamp(0, 1).numpy() * 255).astype(np.uint8)
        depth = row["depth"][0].numpy()
        valid = row["depth_valid"][0].numpy()
        high = np.quantile(depth[valid], .95) if valid.any() else 1.0
        dep = np.clip(depth / max(float(high), 1e-5), 0, 1)
        dep = (255 * (1 - dep)).astype(np.uint8)
        dep = np.stack((dep, dep, dep), axis=-1)
        dep[~valid] = (128, 0, 128)
        tiles.append((rgb, dep))
    if not tiles:
        return
    height, width = tiles[0][0].shape[:2]
    canvas = Image.new("RGB", (width * 2, height * len(tiles)), "white")
    for row, (rgb, depth) in enumerate(tiles):
        canvas.paste(Image.fromarray(rgb), (0, row * height))
        canvas.paste(Image.fromarray(depth), (width, row * height))
    canvas.save(path)


def main():
    cfg = args(); cfg.output.mkdir(parents=True, exist_ok=True)
    discovered = discover_registered_rgbd(cfg.anchor_root, cfg.registry)
    tum, bonn = ([Path(p) for p in discovered["train"]["tum"]],
                 [Path(p) for p in discovered["train"]["bonn"]])
    sources = []
    if tum:
        sources.append(CalibratedGeometryDataset(TaggedTUMDataset(
            tum, source="tum", frames=cfg.frames, height=cfg.size, width=cfg.size)))
    if bonn:
        sources.append(CalibratedGeometryDataset(BonnRGBDWindowDataset(
            bonn, frames=cfg.frames, height=cfg.size, width=cfg.size)))
    if not sources:
        raise SystemExit("No extracted TUM-format roots found; run prepare_real_anchor_data.py first")
    profile = profile_real_geometry(sources, cfg.profile_windows)
    profile_path = cfg.output / "eye_real_anchor_profile_v1.json"
    profile.save(profile_path)
    synthetic = RealAnchoredSyntheticGeometry(
        "train", cfg.samples, profile, cfg.frames, cfg.size)
    report = validate_real_anchored_synthetic(synthetic, profile)
    report.update({"logical_samples": len(synthetic),
                   "storage_model": "deterministic_on_the_fly",
                   "real_roots_train_only": {"tum": len(tum), "bonn": len(bonn)},
                   "profile_split_policy": "train_only_no_validation_test_statistics"})
    report_path = cfg.output / "synthetic_admission_v1.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    save_preview(synthetic, cfg.output / "synthetic_preview_v1.png")
    print(json.dumps({"profile": str(profile_path), "admission": str(report_path),
                      **report}, indent=2))
    if not report["valid"]:
        raise SystemExit("Synthetic admission blocked")


if __name__ == "__main__":
    main()
