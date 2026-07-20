"""Build a JSON data-health card for the JWM real-anchor/synthetic pack."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jwm.anchor_discovery import discover_registered_rgbd
from jwm.dataset_registry import (load_registry, validate_registry_split_groups)
from jwm.geometry_v2_data import BonnRGBDWindowDataset, TaggedTUMDataset


def entropy(counts: list[int]) -> float:
    total = sum(counts)
    if total <= 0 or len(counts) <= 1:
        return 0.0
    p = [value / total for value in counts if value]
    return -sum(value * math.log(value) for value in p) / math.log(len(counts))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT / "data/real_anchor_v1")
    parser.add_argument("--registry", type=Path,
                        default=ROOT / "configs/datasets/real_anchor_v1.json")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "data/real_anchor_v1/dataset_pack_report_v1.json")
    parser.add_argument("--synthetic-samples", type=int, default=250_000)
    args = parser.parse_args()
    _, assets = load_registry(args.registry)
    discovered = discover_registered_rgbd(args.root / "raw", args.registry)
    windows, scenes = {}, {}
    for split, families in discovered.items():
        for family in ("tum", "bonn"):
            roots = families[family]
            if not roots:
                continue
            cls = BonnRGBDWindowDataset if family == "bonn" else TaggedTUMDataset
            kwargs = {"frames": 6, "frame_stride": 2, "window_stride": 12,
                      "height": 256, "width": 256}
            dataset = cls(roots, **kwargs)
            key = f"{split}:{family}"
            windows[key], scenes[key] = len(dataset), len(roots)
    licenses = Counter(asset.license for asset in assets if asset.kind == "real")
    ready = list((args.root / "raw").rglob(".jwm_extracted.json")) \
        if (args.root / "raw").exists() else []
    profile_path = args.root / "derived/eye_real_anchor_profile_v1.json"
    admission_path = args.root / "derived/synthetic_admission_v1.json"
    geometry_path = args.root / "geometry_admission_v1.json"
    geometry_admission = (json.loads(geometry_path.read_text(encoding="utf-8"))
                          if geometry_path.exists() else None)
    planned = sum("starter" in asset.tier and asset.kind == "real" for asset in assets)
    report = {
        "valid": len(ready) == planned and validate_registry_split_groups(assets)["valid"] and
                 admission_path.exists() and
                 json.loads(admission_path.read_text(encoding="utf-8"))["valid"] and
                 geometry_admission is not None and geometry_admission["valid"],
        "real_assets_ready": len(ready),
        "real_assets_planned": planned,
        "real_windows": windows, "real_scenes": scenes,
        "real_window_total": sum(windows.values()),
        "source_balance_entropy": entropy(list(windows.values())),
        "synthetic_logical_train_samples": args.synthetic_samples,
        "synthetic_storage": "on_the_fly_seeded",
        "licenses": dict(licenses),
        "split_group_validation": validate_registry_split_groups(assets),
        "profile": (json.loads(profile_path.read_text(encoding="utf-8"))
                    if profile_path.exists() else None),
        "synthetic_admission": (json.loads(admission_path.read_text(encoding="utf-8"))
                                if admission_path.exists() else None),
        "real_geometry_admission": geometry_admission,
        "final_ablation_required": True,
        "final_ablation_rule": "real+synthetic must improve >=3/5 real metrics with <=3% regression",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
