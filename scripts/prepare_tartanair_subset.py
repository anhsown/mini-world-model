"""Quota-oriented TartanAir-v2 subset download for Kaggle/Linux.

The full corpus is multi-terabyte. This script intentionally requests only
front-camera RGB/depth from a small, diverse environment list; pose metadata is
included by the official toolkit. Run the registry/admission pipeline after it
finishes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


DEFAULT_ENVIRONMENTS = (
    "ArchVizTinyHouseDay",   # small indoor domestic
    "AbandonedFactory2",     # low light / industrial indoor
    "AmericanDiner",         # reflective cluttered indoor
    "AncientTowns",          # outdoor urban/rural structure
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--env", nargs="+", default=list(DEFAULT_ENVIRONMENTS))
    parser.add_argument("--difficulty", nargs="+", default=["easy", "hard"])
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    request = {"env": args.env, "difficulty": args.difficulty,
               "modality": ["image", "depth"],
               "camera_name": ["lcam_front"], "unzip": True,
               "delete_zip": True, "num_workers": args.workers}
    print(json.dumps(request, indent=2))
    if args.dry_run:
        return
    try:
        import tartanair as ta
    except ImportError as exc:
        raise SystemExit("Install official toolkit first: pip install tartanair==1.4.0") from exc
    args.output.mkdir(parents=True, exist_ok=True)
    ta.init(str(args.output))
    ta.download(**request)


if __name__ == "__main__":
    main()
