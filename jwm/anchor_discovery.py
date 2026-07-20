"""Discover extracted real-anchor roots without guessing train/test splits."""

from __future__ import annotations

import json
from pathlib import Path


def discover_registered_rgbd(anchor_root: str | Path,
                             registry: str | Path) -> dict[str, dict[str, list[str]]]:
    root = Path(anchor_root)
    payload = json.loads(Path(registry).read_text(encoding="utf-8"))
    specs = {row["id"]: row for row in payload["assets"]}
    result = {split: {"tum": [], "bonn": [], "tartan": []}
              for split in ("train", "validation", "test")}
    if not root.exists():
        return result
    for marker in root.rglob(".jwm_extracted.json"):
        asset_id = json.loads(marker.read_text(encoding="utf-8"))["asset"]
        spec = specs.get(asset_id)
        if not spec or spec.get("split") not in result:
            continue
        source = str(spec.get("source", "")).lower()
        family = ("bonn" if "bonn" in source else
                  "tartan" if "tartan" in source else
                  "tum" if "tum" in source else None)
        if family is None:
            continue
        base = marker.parent
        if family in ("tum", "bonn"):
            roots = [p.parent for p in base.rglob("rgb.txt")
                     if (p.parent / "depth.txt").exists() and
                     (p.parent / "groundtruth.txt").exists()]
        else:
            roots = [p.parent for p in base.rglob("pose_lcam_front.txt")]
        result[spec["split"]][family].extend(str(path) for path in roots)
    for split in result:
        for family in result[split]:
            result[split][family] = sorted(set(result[split][family]))
    return result

