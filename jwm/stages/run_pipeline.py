# -*- coding: utf-8 -*-
"""Unified training pipeline — chains all 7 stages (Cosmos 3 §4 structure):

    python -m jwm.stages.run_pipeline              # run everything (skip done)
    python -m jwm.stages.run_pipeline --from g2    # start from a stage
    python -m jwm.stages.run_pipeline --only r1    # run one stage
    python -m jwm.stages.run_pipeline --force      # redo even if done

Every stage checkpoints every few hundred steps — safe to shut the machine
down at ANY point; rerunning this command resumes exactly where it stopped."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from jwm.stages import STAGE_ORDER  # noqa: E402
from jwm.stages import (r1_reasoner_pretrain, r2_reasoner_sft,  # noqa: E402
                        g1_generator_pretrain, g2_generator_midtrain,
                        g3_post_text2image, g4_post_image2video, g5_post_policy)
from jwm.stages.common import DATA_DIR  # noqa: E402

MODULES = {
    "r1_reasoner_pretrain": r1_reasoner_pretrain,
    "r2_reasoner_sft": r2_reasoner_sft,
    "g1_generator_pretrain": g1_generator_pretrain,
    "g2_generator_midtrain": g2_generator_midtrain,
    "g3_post_text2image": g3_post_text2image,
    "g4_post_image2video": g4_post_image2video,
    "g5_post_policy": g5_post_policy,
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="start", default=None,
                    help="stage name or prefix (e.g. g2)")
    ap.add_argument("--only", default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    if not (DATA_DIR / "manifest.json").exists():
        print("Data corpus missing — build it first:")
        print("  python -m jwm.data_builders.build_all")
        return 1

    stages = list(STAGE_ORDER)
    if args.only:
        stages = [s for s in stages if s.startswith(args.only)]
    elif args.start:
        idx = next(i for i, s in enumerate(stages) if s.startswith(args.start))
        stages = stages[idx:]

    t0 = time.time()
    print(f"=== JWM v3 PIPELINE: {len(stages)} stage(s) ===")
    for name in stages:
        print(f"\n>>> STAGE {name} <<<")
        t1 = time.time()
        MODULES[name].run(force=args.force)
        print(f">>> {name} took {(time.time()-t1)/60:.1f} min")
    print(f"\n=== PIPELINE COMPLETE in {(time.time()-t0)/60:.1f} min ===")
    print("Deployable brain: jwm/checkpoints/jwm_v3.pt (world_brain picks it up automatically)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
