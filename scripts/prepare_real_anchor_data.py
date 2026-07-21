"""Download a quota-controlled JWM real-anchor dataset pack.

Examples:
  python scripts/prepare_real_anchor_data.py --tier starter --dry-run
  python scripts/prepare_real_anchor_data.py --tier starter --branch eye
  python scripts/prepare_real_anchor_data.py --tier starter --include tum_fr1_xyz
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jwm.dataset_registry import (append_manifest, load_registry,
                                  materialize_asset, select_assets,
                                  validate_registry_split_groups)


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path,
                        default=ROOT / "configs/datasets/real_anchor_v1.json")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "data/real_anchor_v1")
    parser.add_argument("--tier", choices=("starter", "full", "kaggle"),
                        default="starter")
    parser.add_argument("--branch", choices=("eye", "reader", "video", "action"))
    parser.add_argument("--include", nargs="*", default=[])
    parser.add_argument("--no-extract", action="store_true")
    parser.add_argument("--reserve-free-gb", type=float,
                        help="Override the registry disk reserve (Kaggle: 5)")
    parser.add_argument("--delete-archives-after-extract", action="store_true",
                        help="Save disk after verified extraction; markers make reruns safe")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main():
    args = arguments()
    meta, assets = load_registry(args.registry)
    split_report = validate_registry_split_groups(assets)
    if not split_report["valid"]:
        raise SystemExit(f"registry split leakage: {split_report['leaked_scene_groups']}")
    selected = select_assets(assets, args.tier, set(args.include), args.branch)
    planned = sum(asset.size_bytes for asset in selected)
    reserve_free_gb = (float(args.reserve_free_gb)
                       if args.reserve_free_gb is not None
                       else float(meta["reserve_free_gb"]))
    if reserve_free_gb < 0:
        raise SystemExit("--reserve-free-gb must be non-negative")
    print(json.dumps({"tier": args.tier, "assets": [a.id for a in selected],
                      "download_gb": round(planned / (1 << 30), 3),
                      "reserve_free_gb": reserve_free_gb,
                      "delete_archives_after_extract":
                          args.delete_archives_after_extract}, indent=2))
    if args.dry_run:
        return
    manifest = args.output / "manifest.jsonl"
    for index, asset in enumerate(selected, 1):
        print(f"[{index}/{len(selected)}] {asset.id}", flush=True)
        try:
            record = materialize_asset(
                asset, args.output, reserve_free_gb,
                extract=not args.no_extract)
            if (args.delete_archives_after_extract and not args.no_extract and
                    record.get("archive")):
                archive = Path(record["archive"])
                if archive.exists():
                    archive.unlink()
                    record["archive_removed_after_verified_extract"] = True
        except Exception as exc:
            record = {"asset_id": asset.id, "status": "failed",
                      "error": f"{type(exc).__name__}: {exc}"}
            append_manifest(manifest, record)
            raise
        append_manifest(manifest, record)
        print(json.dumps(record, indent=2), flush=True)


if __name__ == "__main__":
    main()
