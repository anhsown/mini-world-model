"""Validate every downloaded real RGB-D source before it can enter training."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jwm.anchor_discovery import discover_registered_rgbd
from jwm.geometry_v2_data import BonnRGBDWindowDataset, TaggedTUMDataset
from jwm.geometry_v3_data import (CalibratedGeometryDataset,
                                  validate_geometry_v3_datasets)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--anchor-root", type=Path,
                        default=ROOT / "data/real_anchor_v1/raw")
    parser.add_argument("--registry", type=Path,
                        default=ROOT / "configs/datasets/real_anchor_v1.json")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "data/real_anchor_v1/geometry_admission_v1.json")
    parser.add_argument("--frames", type=int, default=6)
    parser.add_argument("--size", type=int, default=128)
    args = parser.parse_args()

    discovered = discover_registered_rgbd(args.anchor_root, args.registry)
    splits = {}
    for split, families in discovered.items():
        sources = {}
        if families["tum"]:
            sources["tum"] = CalibratedGeometryDataset(TaggedTUMDataset(
                families["tum"], source="tum", frames=args.frames,
                frame_stride=2, window_stride=12,
                height=args.size, width=args.size))
        if families["bonn"]:
            sources["bonn"] = CalibratedGeometryDataset(BonnRGBDWindowDataset(
                families["bonn"], frames=args.frames,
                frame_stride=2, window_stride=12,
                height=args.size, width=args.size))
        if sources:
            splits[split] = sources

    if not splits:
        raise SystemExit("No registered RGB-D data found")
    report = validate_geometry_v3_datasets(splits, args.output)
    report["discovered_roots"] = discovered
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    summary = {
        "valid": report["valid"],
        "sources": {
            key: {
                "valid": value["valid"],
                "windows": value.get("metrics", {}).get("windows_total"),
                "reprojection_inlier": value.get("metrics", {}).get(
                    "measured_depth_reprojection_inlier"),
            }
            for key, value in report["sources"].items()
        },
        "failures": report["failures"],
        "output": str(args.output),
    }
    print(json.dumps(summary, indent=2))
    if not report["valid"]:
        raise SystemExit("Real geometry admission blocked")


if __name__ == "__main__":
    main()
