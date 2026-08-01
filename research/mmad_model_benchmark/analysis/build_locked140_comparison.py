from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path


BENCH = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BENCH))

from common.mmad import evaluate_records, load_jsonl, write_evaluation  # noqa: E402


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def bootstrap_accuracy(
    truths: list[str], predictions: list[str | None], *, seed: int = 20260801
) -> dict:
    if not truths:
        return {"estimate": None, "ci95": [None, None]}
    rng = random.Random(seed)
    values = []
    n = len(truths)
    for _ in range(2_000):
        indices = [rng.randrange(n) for _ in range(n)]
        values.append(sum(predictions[i] == truths[i] for i in indices) / n)
    values.sort()
    return {
        "estimate": sum(a == b for a, b in zip(truths, predictions)) / n,
        "ci95": [values[49], values[1949]],
    }


def remap_full_predictions(
    subset: list[dict], full: list[dict], predictions: list[dict]
) -> list[dict]:
    full_by_private_key = {row["private_key"]: row for row in full}
    prediction_by_id = {row.get("sample_id"): row for row in predictions}
    remapped = []
    for target in subset:
        full_row = full_by_private_key[target["private_key"]]
        source = prediction_by_id.get(full_row["sample_id"])
        if source is None:
            continue
        item = dict(source)
        item["source_sample_id"] = source.get("sample_id")
        item["sample_id"] = target["sample_id"]
        item["manifest_sha256"] = None
        remapped.append(item)
    return remapped


def build_majority_baseline(subset: list[dict], full: list[dict]) -> list[dict]:
    locked_keys = {row["private_key"] for row in subset}
    pools: dict[str, Counter] = {}
    for row in full:
        if row["private_key"] in locked_keys:
            continue
        pools.setdefault(row["question_type"], Counter())[row["answer"]] += 1
    majority = {task: counts.most_common(1)[0][0] for task, counts in pools.items()}
    return [
        {
            "sample_id": row["sample_id"],
            "status": "ok",
            "prediction": majority[row["question_type"]],
            "backend": "per-task majority from MMAD excluding locked140",
        }
        for row in subset
    ]


def compact_metrics(summary: dict) -> dict:
    return {
        "expected_records": summary.get("expected_records"),
        "attempted_records": summary.get("attempted_records"),
        "parsed_records": summary.get("parsed_records"),
        "completion_rate": summary.get("completion_rate"),
        "micro_accuracy": summary.get("micro_accuracy"),
        "macro_task_accuracy": summary.get("macro_task_accuracy"),
        "per_task": summary.get("per_task"),
        "anomaly_detection": summary.get("anomaly_detection"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cosmos",
        type=Path,
        help="Locked140 Cosmos predictions.jsonl downloaded from Kaggle.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=BENCH / "outputs" / "locked140_comparison",
    )
    args = parser.parse_args()

    subset_manifest = load_json(BENCH / "data" / "subset_manifest.json")
    full_manifest = load_json(BENCH / "data_full" / "full_manifest.json")
    subset = subset_manifest["records"]
    full = full_manifest["records"]
    assert len(subset) == 140

    qwen_path = (
        BENCH
        / "models"
        / "qwen2_vl"
        / "qwen2_vl_mmad_full_results"
        / "predictions.jsonl"
    )
    qwen = remap_full_predictions(subset, full, load_jsonl(qwen_path))
    majority = build_majority_baseline(subset, full)
    runs = {"qwen2_vl_2b": qwen, "majority_baseline": majority}
    if args.cosmos:
        runs["cosmos3_nano_bnb8"] = load_jsonl(args.cosmos)

    args.output.mkdir(parents=True, exist_ok=True)
    report = {
        "benchmark": "MMAD locked140",
        "manifest_sha256": subset_manifest["manifest_sha256"],
        "selection": "20 questions per task; seed=20260729",
        "published_reference": {
            "gpt4o_overall_accuracy_approx": 0.749,
            "warning": "Paper-wide score; not a paired score on locked140.",
        },
        "models": {},
    }

    truths_by_id = {row["sample_id"]: row["answer"] for row in subset}
    for name, rows in runs.items():
        summary, scored = evaluate_records(subset_manifest, rows)
        write_evaluation(args.output / name, summary, scored)
        latest = {row.get("sample_id"): row for row in rows}
        truths = [truths_by_id[row["sample_id"]] for row in subset]
        predictions = [latest.get(row["sample_id"], {}).get("prediction") for row in subset]
        report["models"][name] = {
            **compact_metrics(summary),
            "micro_accuracy_bootstrap": bootstrap_accuracy(truths, predictions),
        }

    if "cosmos3_nano_bnb8" in runs:
        qwen_by_id = {row["sample_id"]: row.get("prediction") for row in qwen}
        cosmos_by_id = {
            row.get("sample_id"): row.get("prediction")
            for row in runs["cosmos3_nano_bnb8"]
            if row.get("status") == "ok"
        }
        paired = [sid for sid in truths_by_id if sid in qwen_by_id and sid in cosmos_by_id]
        report["paired_comparison"] = {
            "n": len(paired),
            "agreement_rate": (
                sum(qwen_by_id[sid] == cosmos_by_id[sid] for sid in paired) / len(paired)
                if paired
                else None
            ),
            "qwen_only_correct": sum(
                qwen_by_id[sid] == truths_by_id[sid]
                and cosmos_by_id[sid] != truths_by_id[sid]
                for sid in paired
            ),
            "cosmos_only_correct": sum(
                cosmos_by_id[sid] == truths_by_id[sid]
                and qwen_by_id[sid] != truths_by_id[sid]
                for sid in paired
            ),
            "both_wrong": sum(
                cosmos_by_id[sid] != truths_by_id[sid]
                and qwen_by_id[sid] != truths_by_id[sid]
                for sid in paired
            ),
        }

    destination = args.output / "comparison_metrics.json"
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("saved", destination)


if __name__ == "__main__":
    main()
