"""Run a context-blind template-prior baseline on FactoryBench.

This is intentionally a weak floor, not a model claim.  It uses only the
causal level and question template identity.  Any sensor-aware model should
beat it on every level, especially the held-out episode split.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import statistics
import sys
from typing import Any, Iterator

from huggingface_hub import hf_hub_download

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from jwm.factorybench import infer_answer_format, score_factorybench_answer


REPO_ID = "FactoryBench/FactoryBench"


def records(level: int, split: str) -> Iterator[dict[str, Any]]:
    path = hf_hub_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        filename=f"factorybench_qa/level_{level}/{split}.jsonl",
    )
    with open(path, encoding="utf-8") as source:
        for line in source:
            yield json.loads(line)


def key(record: dict[str, Any]) -> tuple[Any, ...]:
    return (
        record.get("level"),
        record.get("template_id"),
        record.get("template_type"),
        infer_answer_format(record),
    )


def numeric_answer(answer: Any) -> list[float] | None:
    if isinstance(answer, (int, float)):
        return [float(answer)]
    if isinstance(answer, str):
        try:
            value = json.loads(answer)
        except json.JSONDecodeError:
            return None
        if isinstance(value, (int, float)):
            return [float(value)]
        if isinstance(value, list):
            try:
                return [float(item) for item in value]
            except (TypeError, ValueError):
                return None
    return None


class TemplatePrior:
    def __init__(self) -> None:
        self.values: dict[tuple[Any, ...], list[Any]] = defaultdict(list)
        self.fallback: dict[tuple[int, str], list[Any]] = defaultdict(list)
        self.predictions: dict[tuple[Any, ...], Any] = {}
        self.fallback_predictions: dict[tuple[int, str], Any] = {}

    @staticmethod
    def _reduce(values: list[Any], family: str) -> Any:
        if family in {"numeric", "numeric_array"}:
            arrays = [numeric_answer(value) for value in values]
            arrays = [value for value in arrays if value]
            if arrays:
                width = Counter(len(value) for value in arrays).most_common(1)[0][0]
                aligned = [value for value in arrays if len(value) == width]
                medians = [
                    statistics.median(value[index] for value in aligned)
                    for index in range(width)
                ]
                if family == "numeric":
                    return medians[0]
                return json.dumps(medians, separators=(",", ":"))
        return Counter(str(value) for value in values).most_common(1)[0][0]

    def fit(self) -> None:
        for level in range(1, 5):
            for record in records(level, "train"):
                family = infer_answer_format(record)
                self.values[key(record)].append(record.get("answer"))
                self.fallback[(level, family)].append(record.get("answer"))
        self.predictions = {
            item_key: self._reduce(values, item_key[-1])
            for item_key, values in self.values.items()
        }
        self.fallback_predictions = {
            item_key: self._reduce(values, item_key[-1])
            for item_key, values in self.fallback.items()
        }

    def predict(self, record: dict[str, Any]) -> Any:
        family = infer_answer_format(record)
        return self.predictions.get(
            key(record), self.fallback_predictions.get((record["level"], family), "")
        )


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def run() -> dict[str, Any]:
    model = TemplatePrior()
    model.fit()
    groups: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    examples: list[dict[str, Any]] = []
    for level in range(1, 5):
        for record in records(level, "test"):
            prediction = model.predict(record)
            score = score_factorybench_answer(record, prediction)
            groups[(level, str(score["family"]))].append(score)
            if len(examples) < 20:
                examples.append(
                    {
                        "id": record["id"],
                        "level": level,
                        "family": score["family"],
                        "target": record["answer"],
                        "prediction": prediction,
                        "score": score,
                    }
                )

    by_group: dict[str, Any] = {}
    by_level_values: dict[int, list[float]] = defaultdict(list)
    for (level, family), scores in sorted(groups.items()):
        primary = [float(item["primary_score"]) for item in scores]
        exact = [float(item["exact_match"]) for item in scores]
        by_level_values[level].extend(primary)
        result = {
            "n": len(scores),
            "primary_score": mean(primary),
            "exact_match": mean(exact),
        }
        for metric in (
            "within_tolerance",
            "multilabel_f1",
            "pairwise_accuracy",
            "token_f1",
            "root_cause_hit",
        ):
            values = [
                float(item[metric])
                for item in scores
                if item.get(metric) is not None
            ]
            if values:
                result[metric] = mean(values)
        by_group[f"L{level}_{family}"] = result

    by_level = {
        f"L{level}": {
            "n": len(by_level_values[level]),
            "micro_primary_score": mean(by_level_values[level]),
        }
        for level in range(1, 5)
    }
    level_macro = mean([value["micro_primary_score"] for value in by_level.values()])
    return {
        "name": "FactoryBench context-blind template-prior floor",
        "uses_telemetry": False,
        "uses_question_text": False,
        "uses_template_identity": True,
        "warning": (
            "This is a leakage-control floor. It is not a competitive baseline "
            "and must not be reported as JWM performance."
        ),
        "by_level": by_level,
        "by_level_and_answer_family": by_group,
        "macro_primary_score_over_levels": level_macro,
        "examples": examples,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default="research/factorybench/context_blind_baseline_results.json",
    )
    args = parser.parse_args()
    report = run()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    rows = "\n".join(
        f"| {level} | {values['n']:,} | "
        f"{values['micro_primary_score']:.4f} |"
        for level, values in report["by_level"].items()
    )
    analysis = f"""# FactoryBench Context-Blind Baseline

This baseline sees only the causal level and template identity. It does **not**
read the question or telemetry and is therefore a leakage-control floor.

| Level | Test records | Task-aware primary score |
|---|---:|---:|
{rows}

Macro score across L1–L4: **{report['macro_primary_score_over_levels']:.4f}**

The relatively high L4 token-F1 is not evidence of machine understanding.
FactoryBench contains repeated, templated remediation language. Future models
must report root-cause accuracy, evidence grounding and answer novelty alongside
token overlap. A sensor-aware JWM should beat this floor on every causal level,
not only in aggregate.
"""
    (output.parent / "FACTORYBENCH_BASELINE.md").write_text(
        analysis, encoding="utf-8"
    )
    print(json.dumps(report["by_level"], indent=2))
    print("macro_primary_score_over_levels", report["macro_primary_score_over_levels"])
    print("output", output.resolve())


if __name__ == "__main__":
    main()
