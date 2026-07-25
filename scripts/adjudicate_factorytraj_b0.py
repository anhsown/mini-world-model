"""Adjudicate model predictions against the frozen FactoryTraj-B0 gate."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jwm.factorytraj_b0 import score_b0, split_items


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--full-predictions", type=Path, required=True)
    parser.add_argument("--anonymized-predictions", type=Path, required=True)
    parser.add_argument(
        "--threshold",
        type=Path,
        default=ROOT / "research/factorytraj_bench/b0_pass_threshold_v0.1.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = json.loads(args.data.read_text(encoding="utf-8"))
    evaluation = split_items(payload["items"], "validation") + split_items(
        payload["items"], "test"
    )
    full = score_b0(
        evaluation, json.loads(args.full_predictions.read_text(encoding="utf-8"))
    )
    anonymized = score_b0(
        evaluation,
        json.loads(args.anonymized_predictions.read_text(encoding="utf-8")),
    )
    threshold = json.loads(args.threshold.read_text(encoding="utf-8"))
    primary = threshold["primary_thresholds"]
    checks = {
        "H_b0_macro": full["b0_contract_macro_score"] >= primary["b0_contract_macro_score"],
        **{
            f"H_{name}": full["components"][name] is not None
            and full["components"][name] >= minimum
            for name, minimum in primary.items()
            if name != "b0_contract_macro_score"
        },
        "H_anonymized_robustness": (
            full["b0_contract_macro_score"]
            - anonymized["b0_contract_macro_score"]
            <= threshold["robustness_thresholds"][
                "maximum_full_to_anonymized_macro_drop"
            ]
        ),
    }
    report = {
        "benchmark": "FactoryTraj-B0 v0.1",
        "full": full,
        "anonymized": anonymized,
        "checks": checks,
        "passed_b0": all(checks.values()),
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["passed_b0"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
