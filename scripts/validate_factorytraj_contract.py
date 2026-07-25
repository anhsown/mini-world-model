"""Validate FactoryTraj-Bench schemas, examples and semantic invariants."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
import sys

from jsonschema import Draft202012Validator, FormatChecker

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jwm.factorytraj import semantic_errors


PACK = ROOT / "research" / "factorytraj_bench"
ITEM_SCHEMA = PACK / "factorytraj_item.schema.json"
OUTPUT_SCHEMA = PACK / "factorytraj_output.schema.json"
SAMPLES = PACK / "factorybench_samples_v0.1.json"
REPORT = PACK / "factorytraj_validation_report_v0.1.json"


def main() -> None:
    item_schema = json.loads(ITEM_SCHEMA.read_text(encoding="utf-8"))
    output_schema = json.loads(OUTPUT_SCHEMA.read_text(encoding="utf-8"))
    payload = json.loads(SAMPLES.read_text(encoding="utf-8"))

    Draft202012Validator.check_schema(item_schema)
    Draft202012Validator.check_schema(output_schema)
    validator = Draft202012Validator(
        item_schema, format_checker=FormatChecker()
    )
    errors: list[str] = []
    for item in payload["items"]:
        for error in sorted(
            validator.iter_errors(item), key=lambda value: list(value.path)
        ):
            path = ".".join(map(str, error.path))
            errors.append(f"{item.get('item_id')}:{path}: {error.message}")
    errors.extend(semantic_errors(payload["items"]))

    levels = Counter(item["task"]["causal_level"] for item in payload["items"])
    structures = Counter(
        item["trajectory"]["stream_relationship"] for item in payload["items"]
    )
    hypotheses = {
        "H_item_schema_valid": not any("schema" in error for error in errors),
        "H_output_schema_valid": True,
        "H_exactly_ten_examples": len(payload["items"]) == 10,
        "H_levels_1_to_4_covered": set(levels) == {1, 2, 3, 4},
        "H_single_and_paired_stream_covered": {
            "single",
            "paired",
        }.issubset(structures),
        "H_no_model_input_target_leakage": not any(
            "target leakage" in error for error in errors
        ),
        "H_episode_split_invariants": not any(
            "split-group leakage" in error for error in errors
        ),
        "H_research_license_enforced": all(
            not item["governance"]["commercial_training_allowed"]
            for item in payload["items"]
        ),
    }
    valid = not errors and all(hypotheses.values())
    report = {
        "valid": valid,
        "schema_version": payload["schema_version"],
        "items": len(payload["items"]),
        "levels": dict(sorted(levels.items())),
        "stream_relationships": dict(sorted(structures.items())),
        "hypotheses": hypotheses,
        "errors": errors,
    }
    REPORT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(0 if valid else 1)


if __name__ == "__main__":
    main()
