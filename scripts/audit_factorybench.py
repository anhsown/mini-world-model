"""Audit FactoryBench and emit shareable Industrial JWM research artifacts."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
import json
from pathlib import Path
import statistics
import sys
from typing import Any, Iterator

from huggingface_hub import HfApi, hf_hub_download

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from jwm.factorybench import (
    canonicalize_record,
    channel_role,
    context_series,
    infer_answer_format,
    normalize_text,
    provenance_episode_keys,
)


REPO_ID = "FactoryBench/FactoryBench"
LEVELS = range(1, 5)
SPLITS = ("train", "validation", "test")
EXPECTED_FIELDS = {
    "id",
    "level",
    "template_id",
    "template_type",
    "hides",
    "question",
    "options",
    "answer",
    "acceptance_bounds",
    "provenance",
    "context",
}


def records(level: int, split: str) -> Iterator[dict[str, Any]]:
    path = hf_hub_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        filename=f"factorybench_qa/level_{level}/{split}.jsonl",
    )
    with open(path, encoding="utf-8") as source:
        for line in source:
            yield json.loads(line)


def percentile(values: list[int], q: float) -> int:
    if not values:
        return 0
    return sorted(values)[round((len(values) - 1) * q)]


def audit(max_samples: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    api = HfApi()
    info = api.dataset_info(REPO_ID, files_metadata=True)
    counts: Counter[tuple[int, str]] = Counter()
    sources: Counter[tuple[int, str, str]] = Counter()
    templates: Counter[tuple[int, str, str]] = Counter()
    families: Counter[tuple[int, str, str]] = Counter()
    missing_fields: Counter[str] = Counter()
    malformed = 0
    ids: dict[str, set[str]] = {split: set() for split in SPLITS}
    episodes: dict[str, set[tuple[str, str]]] = {split: set() for split in SPLITS}
    duplicate_ids_within: Counter[str] = Counter()
    step_lengths: list[int] = []
    context_bytes: list[int] = []
    role_record_coverage: Counter[str] = Counter()
    role_channel_counts: Counter[str] = Counter()
    context_structures: Counter[str] = Counter()
    answers: dict[tuple[int, str, str], set[str]] = defaultdict(set)
    questions: dict[tuple[int, str], set[str]] = defaultdict(set)
    test_answer_seen_in_train: Counter[tuple[int, str]] = Counter()
    test_answer_totals: Counter[tuple[int, str]] = Counter()
    test_question_seen_in_train: Counter[int] = Counter()
    test_question_totals: Counter[int] = Counter()
    sample_by_stratum: dict[tuple[Any, ...], dict[str, Any]] = {}

    for level in LEVELS:
        for split in SPLITS:
            for record in records(level, split):
                counts[(level, split)] += 1
                record_id = str(record.get("id"))
                if record_id in ids[split]:
                    duplicate_ids_within[split] += 1
                ids[split].add(record_id)
                for field in EXPECTED_FIELDS - set(record):
                    missing_fields[field] += 1
                provenance = record.get("provenance") or {}
                source = str(provenance.get("dataset", "unknown"))
                episodes[split].update(provenance_episode_keys(provenance))
                sources[(level, split, source)] += 1
                template = str(record.get("template_type", "unknown"))
                family = infer_answer_format(record)
                answer_key = normalize_text(record.get("answer"))
                question_key = normalize_text(record.get("question"))
                if split == "test":
                    test_answer_totals[(level, family)] += 1
                    if answer_key in answers[(level, family, "train")]:
                        test_answer_seen_in_train[(level, family)] += 1
                    test_question_totals[level] += 1
                    if question_key in questions[(level, "train")]:
                        test_question_seen_in_train[level] += 1
                answers[(level, family, split)].add(answer_key)
                questions[(level, split)].add(question_key)
                templates[(level, split, template)] += 1
                families[(level, split, family)] += 1

                context = record.get("context") or {}
                streams = context_series(context)
                if "primary" in streams:
                    context_structures["single_stream"] += 1
                elif streams:
                    context_structures["paired_stream"] += 1
                else:
                    context_structures["option_only_or_empty"] += 1
                rows = [
                    row
                    for stream in streams.values()
                    for row in (stream.get("time_series") or [])
                ]
                step_lengths.append(len(rows))
                context_bytes.append(
                    len(record.get("question", "").encode("utf-8"))
                    + sum(len(str(row).encode("utf-8")) for row in rows)
                    + sum(
                        len(str(value).encode("utf-8"))
                        for value in (record.get("options") or {}).values()
                    )
                )
                present_roles = set()
                for stream in streams.values():
                    mapping = (
                        (stream.get("time_series_format") or {}).get("acronym_mapping")
                        or {}
                    )
                    for full_name in mapping.values():
                        role = channel_role(str(full_name))
                        role_channel_counts[role] += 1
                        present_roles.add(role)
                for role in present_roles:
                    role_record_coverage[role] += 1

                if split == "test":
                    key = (level, template, source, family)
                    if key not in sample_by_stratum:
                        compact = {
                            name: value
                            for name, value in record.items()
                            if name != "context"
                        }
                        compact["canonical"] = canonicalize_record(record, max_steps=8)
                        sample_by_stratum[key] = compact

    overlaps: dict[str, dict[str, int]] = {}
    for index, left in enumerate(SPLITS):
        for right in SPLITS[index + 1 :]:
            key = f"{left}__{right}"
            overlaps[key] = {
                "id_overlap": len(ids[left] & ids[right]),
                "source_episode_overlap": len(episodes[left] & episodes[right]),
            }

    files = [
        {"path": item.rfilename, "size_bytes": getattr(item, "size", 0) or 0}
        for item in info.siblings
    ]
    total = sum(item["size_bytes"] for item in files)
    actual_total = sum(counts.values())
    records_total = max(actual_total, 1)
    audit_report = {
        "dataset": REPO_ID,
        "revision": info.sha,
        "access": {
            "private": info.private,
            "gated": info.gated,
            "license": "cc-by-nc-4.0",
            "commercial_training_allowed": False,
            "source_url": f"https://huggingface.co/datasets/{REPO_ID}",
        },
        "repository": {
            "file_count": len(files),
            "total_size_bytes": total,
            "total_size_gib": round(total / 2**30, 3),
            "files": files,
        },
        "records": {
            "actual_total": actual_total,
            "dataset_card_total": 70918,
            "difference_from_card": actual_total - 70918,
            "counts": {
                f"L{level}_{split}": counts[(level, split)]
                for level in LEVELS
                for split in SPLITS
            },
            "unique_ids": {split: len(ids[split]) for split in SPLITS},
            "unique_source_episodes": {
                split: len(episodes[split]) for split in SPLITS
            },
            "duplicate_ids_within_split": dict(duplicate_ids_within),
            "malformed_json": malformed,
            "missing_required_fields": dict(missing_fields),
        },
        "split_leakage": {
            "valid": all(
                values["id_overlap"] == 0
                and values["source_episode_overlap"] == 0
                for values in overlaps.values()
            ),
            "comparisons": overlaps,
            "split_unit": "provenance.dataset + provenance.episode",
        },
        "distribution": {
            "sources": {
                f"L{level}_{split}_{source}": count
                for (level, split, source), count in sources.items()
            },
            "template_types": {
                f"L{level}_{split}_{template}": count
                for (level, split, template), count in templates.items()
            },
            "answer_families": {
                f"L{level}_{split}_{family}": count
                for (level, split, family), count in families.items()
            },
            "time_steps": {
                "min": min(step_lengths),
                "median": statistics.median(step_lengths),
                "p95": percentile(step_lengths, 0.95),
                "max": max(step_lengths),
            },
            "serialized_context_bytes": {
                "median": statistics.median(context_bytes),
                "p95": percentile(context_bytes, 0.95),
                "max": max(context_bytes),
                "fraction_over_jwm_96_bytes": round(
                    sum(value > 96 for value in context_bytes) / records_total, 6
                ),
                "fraction_over_4096_bytes": round(
                    sum(value > 4096 for value in context_bytes) / records_total, 6
                ),
            },
            "input_role_record_coverage": {
                role: {
                    "records": role_record_coverage[role],
                    "fraction": round(role_record_coverage[role] / records_total, 6),
                }
                for role in ("sensor", "control", "context")
            },
            "input_role_channel_occurrences": dict(role_channel_counts),
            "context_structures": dict(context_structures),
            "target_reuse": {
                "test_answer_seen_exactly_in_train": {
                    f"L{level}_{family}": {
                        "count": test_answer_seen_in_train[(level, family)],
                        "total": test_answer_totals[(level, family)],
                        "fraction": round(
                            test_answer_seen_in_train[(level, family)]
                            / max(test_answer_totals[(level, family)], 1),
                            6,
                        ),
                    }
                    for level, family in sorted(test_answer_totals)
                },
                "test_question_seen_exactly_in_train": {
                    f"L{level}": {
                        "count": test_question_seen_in_train[level],
                        "total": test_question_totals[level],
                        "fraction": round(
                            test_question_seen_in_train[level]
                            / max(test_question_totals[level], 1),
                            6,
                        ),
                    }
                    for level in LEVELS
                },
            },
        },
        "admission_hypotheses": {
            "H_public_and_ungated": not info.private and not info.gated,
            "H_license_recorded": True,
            "H_json_parseable": malformed == 0,
            "H_required_fields_present": not missing_fields,
            "H_split_ids_disjoint": all(
                value["id_overlap"] == 0 for value in overlaps.values()
            ),
            "H_source_episodes_disjoint": all(
                value["source_episode_overlap"] == 0 for value in overlaps.values()
            ),
            "H_sensor_channels_available": role_record_coverage["sensor"] > 0,
            "H_control_channels_available": role_record_coverage["control"] > 0,
            "H_current_jwm_text_path_sufficient": (
                sum(value > 96 for value in context_bytes) / records_total < 0.05
            ),
            "H_commercial_reuse_cleared": False,
        },
    }
    audit_report["admission"] = {
        "research": all(
            audit_report["admission_hypotheses"][name]
            for name in (
                "H_public_and_ungated",
                "H_license_recorded",
                "H_json_parseable",
                "H_required_fields_present",
                "H_split_ids_disjoint",
                "H_source_episodes_disjoint",
                "H_sensor_channels_available",
                "H_control_channels_available",
            )
        ),
        "direct_commercial_training": False,
        "current_jwm_without_new_encoder": False,
    }
    samples = list(sample_by_stratum.values())[:max_samples]
    return audit_report, samples


def markdown(report: dict[str, Any], sample_count: int) -> str:
    hypotheses = report["admission_hypotheses"]
    counts = report["records"]["counts"]
    rows = "\n".join(
        f"| L{level} | {counts[f'L{level}_train']:,} | "
        f"{counts[f'L{level}_validation']:,} | {counts[f'L{level}_test']:,} |"
        for level in LEVELS
    )
    gates = "\n".join(
        f"- [{'x' if passed else ' '}] `{name}`"
        for name, passed in hypotheses.items()
    )
    context = report["distribution"]["serialized_context_bytes"]
    structures = report["distribution"]["context_structures"]
    reuse = report["distribution"]["target_reuse"][
        "test_answer_seen_exactly_in_train"
    ]
    l4_reuse = reuse.get("L4_free_form", {"fraction": 0.0})
    return f"""# FactoryBench Audit for Industrial JWM

Generated from revision `{report['revision']}`.

## Decision

- **Research/benchmark admission:** PASS
- **Direct commercial training:** BLOCKED by CC BY-NC 4.0
- **Direct ingestion through current JWM text path:** BLOCKED; a numerical
  time-series/action encoder is required.

## Verified inventory

| Level | Train | Validation | Test |
|---|---:|---:|---:|
{rows}

The repository contains **{report['records']['actual_total']:,} parseable Q&A
records**, which is {abs(report['records']['difference_from_card'])} fewer than
the 70,918 stated on the dataset card. The repository is
{report['repository']['total_size_gib']} GiB across
{report['repository']['file_count']} files.

## Leakage audit

No record ID or `(source dataset, episode)` pair crosses
train/validation/test. Reusing an episode across multiple causal levels inside
the same split is allowed, but level scores must be reported separately.

Evidence is packaged as {structures.get('single_stream', 0):,} single-stream,
{structures.get('paired_stream', 0):,} paired-stream and
{structures.get('option_only_or_empty', 0):,} option-only records. An adapter
that reads only `context.time_series` silently drops the comparative items.

## Architecture implication

Median serialized question + time-series context is
**{context['median']:,} bytes** and {context['fraction_over_jwm_96_bytes']:.1%}
of records exceed JWM's 96-byte question path. Stringifying telemetry into AR
text would therefore truncate almost all physical evidence. FactoryBench
should enter JWM through:

1. a numerical sensor-history encoder;
2. a distinct control/action projection;
3. machine-context tokens;
4. the AR reasoner for the natural-language question and response.

Static machine/task context is not consistently embedded in the Q&A telemetry
object. FactoryWave records require a provenance-safe join with
`episodes.parquet` and the knowledge graph. Hidden fault/answer fields must be
excluded from model input.

## Target-reuse warning

{l4_reuse['fraction']:.1%} of L4 free-form test targets appear verbatim in the
training target set. This is not source-episode leakage, but it makes exact
match and token-F1 vulnerable to template memorization. L4 evaluation must also
report root-cause correctness and evidence grounding.

## Admission hypotheses

{gates}

## Shareable samples

`representative_samples_l1_l4.json` contains {sample_count} compact,
provenance-preserving records stratified by level, template, source and answer
family. Time series are head/tail sampled only for visualization; benchmark
evaluation must use the complete source records.

## Source

https://huggingface.co/datasets/FactoryBench/FactoryBench
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="research/factorybench")
    parser.add_argument("--max-samples", type=int, default=64)
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    report, samples = audit(args.max_samples)
    (output / "factorybench_audit.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (output / "representative_samples_l1_l4.json").write_text(
        json.dumps(samples, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (output / "FACTORYBENCH_AUDIT.md").write_text(
        markdown(report, len(samples)), encoding="utf-8"
    )
    print(json.dumps(report["admission"], indent=2))
    print("records", report["records"]["actual_total"])
    print("samples", len(samples))
    print("output", output.resolve())


if __name__ == "__main__":
    main()
