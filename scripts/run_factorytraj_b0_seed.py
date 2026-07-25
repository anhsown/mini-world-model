"""Run B0 seed baselines and determine whether B0 can be adjudicated."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jwm.factorytraj_b0 import (
    majority_role_baseline,
    rule_baseline,
    score_b0,
    split_items,
)


PACK = ROOT / "research" / "factorytraj_bench"
DATA = PACK / "b0_seed_v0.1.json"
REPORT = PACK / "b0_seed_results_v0.1.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--report", type=Path, default=REPORT)
    args = parser.parse_args()
    data = json.loads(args.data.read_text(encoding="utf-8"))
    items = data["items"]
    train = split_items(items, "train")
    evaluation = split_items(items, "validation") + split_items(items, "test")
    source_count = len({item["source_dataset"] for item in items})

    results = {
        "majority": score_b0(
            evaluation, majority_role_baseline(train, evaluation)
        ),
        "rule_full_name": score_b0(evaluation, rule_baseline(evaluation)),
        "rule_anonymized_name": score_b0(
            evaluation, rule_baseline(evaluation, anonymized=True)
        ),
        "rule_anonymized_with_docs": score_b0(
            evaluation,
            rule_baseline(
                evaluation, anonymized=True, use_documentation=True
            ),
        ),
    }
    coverage = results["rule_full_name"]["label_coverage"]
    authoritative_range_count = sum(
        (
            item["ground_truth"].get("instrument_range") is not None
            or item["ground_truth"].get("eu_range") is not None
        )
        for item in evaluation
    )
    dataset_admission = {
        "H_item_count_at_least_100": len(items) >= 100,
        "H_at_least_two_source_families": source_count >= 2,
        "H_unit_coverage_at_least_50pct": coverage["engineering_unit"] >= 0.5,
        "H_at_least_50_authoritative_ranges": authoritative_range_count >= 50,
        "H_relationship_coverage_at_least_20pct": coverage["relationships"] >= 0.2,
        "H_no_name_shortcut": (
            results["rule_full_name"]["components"]["role_macro_f1"]
            - results["rule_anonymized_with_docs"]["components"]["role_macro_f1"]
            <= 0.20
        ),
    }
    benchmark_admitted = all(dataset_admission.values())
    report = {
        "benchmark": "FactoryTraj-B0 seed v0.1",
        "items_total": len(items),
        "items_train": len(train),
        "items_evaluation": len(evaluation),
        "source_count": source_count,
        "authoritative_range_count": authoritative_range_count,
        "results": results,
        "dataset_admission": dataset_admission,
        "benchmark_admitted": benchmark_admitted,
        "current_jwm_evaluated": False,
        "current_jwm_passed_b0": False,
        "decision": (
            "ready_to_freeze_threshold_and_evaluate_jwm"
            if benchmark_admitted
            else "blocked_by_b0_dataset_admission"
        ),
        "required_next_data": [
            "authoritative OPC UA tag exports with EngineeringUnits and EURange",
            "SME-reviewed cross-tag relationships",
            "anonymized-name evaluation records with representative samples/docs",
        ],
    }
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
