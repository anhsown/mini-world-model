import json
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common.mmad import CANONICAL_TYPES, evaluate_records, parse_prediction


def test_strict_answer_parser():
    assert parse_prediction("A") == "A"
    assert parse_prediction("Final answer: (c)") == "C"
    assert parse_prediction("The answer is D.") == "D"
    assert parse_prediction("A or B") is None
    assert parse_prediction("") is None


def test_frozen_manifest_contract():
    manifest = json.loads((ROOT / "data" / "subset_manifest.json").read_text(encoding="utf-8"))
    records = manifest["records"]
    assert len(records) == 140
    assert Counter(row["question_type"] for row in records) == Counter({task: 20 for task in CANONICAL_TYPES})
    assert len({row["sample_id"] for row in records}) == 140
    assert all(Path(row["image_file"]).name.startswith("image_") for row in records)
    assert all(row["prompt"].endswith("Write your final answer immediately after the </think> tag.") for row in records)
    assert manifest["manifest_sha256"] == "7f6dcad2dda8bdd0a2f876c4b7a740239cf9437b3eb1636da88746ebd0aba50f"


def test_evaluator_counts_and_metrics():
    manifest = json.loads((ROOT / "data" / "subset_manifest.json").read_text(encoding="utf-8"))
    predictions = [
        {"sample_id": row["sample_id"], "status": "ok", "prediction": row["answer"], "latency_seconds": 1.0}
        for row in manifest["records"]
    ]
    summary, rows = evaluate_records(manifest, predictions)
    assert len(rows) == 140
    assert summary["micro_accuracy"] == 1.0
    assert summary["macro_task_accuracy"] == 1.0
    assert summary["completion_rate"] == 1.0
    assert summary["parse_failure_rate"] == 0.0


def test_materialized_images_are_complete():
    manifest = json.loads((ROOT / "data" / "subset_manifest.json").read_text(encoding="utf-8"))
    expected = {row["image_file"] for row in manifest["records"]}
    assert all((ROOT / "data" / relative).stat().st_size > 0 for relative in expected)
