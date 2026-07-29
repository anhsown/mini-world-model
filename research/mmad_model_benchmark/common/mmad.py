from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable


MMAD_JSON_URL = "https://huggingface.co/datasets/jiang-cc/MMAD/resolve/main/mmad.json"
ARCHIVE_URLS = {
    name: f"https://huggingface.co/datasets/jiang-cc/MMAD/resolve/main/{name}.zip"
    for name in ("DS-MVTec", "MVTec-AD", "MVTec-LOCO", "VisA", "GoodsAD")
}
CANONICAL_TYPES = (
    "Anomaly Detection",
    "Defect Classification",
    "Defect Localization",
    "Defect Description",
    "Defect Analysis",
    "Object Classification",
    "Object Analysis",
)
SYSTEM_PROMPT = (
    "You are an industrial visual inspector. Use only visible image evidence. "
    "Follow the required response template and put exactly one option letter "
    "A, B, C, or D after the closing think tag."
)


def canonical_question_type(raw: str) -> str:
    if raw in {"Object Structure", "Object Details", "Object Analysis"}:
        return "Object Analysis"
    return raw


def normalize_source(raw: str) -> str:
    return "MVTec-AD" if raw in {"DS-MVTec", "MVTec-AD"} else raw


def is_normal_path(path: str) -> bool:
    value = path.replace("\\", "/").lower()
    return "/good/" in value or "/normal/" in value


def format_prompt(question: str, options: dict[str, str]) -> str:
    choices = "\n".join(f"{letter}. {text}" for letter, text in sorted(options.items()))
    return (
        "Inspect the industrial product image and answer the multiple-choice question.\n\n"
        f"Question:\n{question}\n\nChoices:\n{choices}\n\n"
        "Your final answer must be exactly one letter: A, B, C, or D.\n\n"
        "Answer the question using the following format:\n\n"
        "<think>\nYour reasoning.\n</think>\n\n"
        "Write your final answer immediately after the </think> tag."
    )


def parse_prediction(text: str) -> str | None:
    if not text:
        return None
    cleaned = re.sub(r"<think>.*?</think>", " ", text, flags=re.I | re.S).strip()
    if "\nResponse\n" in cleaned:
        cleaned = cleaned.rsplit("\nResponse\n", 1)[-1].strip()
    exact = re.fullmatch(r"[\s\[\(]*([A-D])[\s\.\]\)]*", cleaned, flags=re.I)
    if exact:
        return exact.group(1).upper()
    patterns = (
        r"(?:final\s+answer|answer|option|choice)\s*(?:is|:|-)?\s*\(?([A-D])\)?\b",
        r"\b([A-D])\s*[\.)]\s*(?:is\s+)?(?:the\s+)?(?:correct|best)\b",
    )
    matches: list[str] = []
    for pattern in patterns:
        matches.extend(re.findall(pattern, cleaned, flags=re.I))
    return matches[-1].upper() if matches else None


def _stable_key(seed: int, value: str) -> str:
    return hashlib.sha256(f"{seed}|{value}".encode()).hexdigest()


def flatten_annotations(raw: dict) -> list[dict]:
    rows = []
    for image_key, item in raw.items():
        parts = image_key.replace("\\", "/").split("/")
        source_raw = parts[0]
        category = parts[1] if len(parts) > 1 else "unknown"
        for question_index, turn in enumerate(item.get("conversation", [])):
            canonical = canonical_question_type(turn.get("type", ""))
            if canonical not in CANONICAL_TYPES:
                continue
            answer = str(turn.get("Answer", "")).upper()
            options = {str(k).upper(): str(v) for k, v in turn.get("Options", {}).items()}
            if answer not in options or not 2 <= len(options) <= 4:
                continue
            private_key = f"{image_key}#q{question_index}"
            rows.append(
                {
                    "private_key": private_key,
                    "source_image_path": image_key,
                    "source_archive": source_raw,
                    "source_dataset": normalize_source(source_raw),
                    "category": category,
                    "is_normal": is_normal_path(image_key),
                    "question_index": question_index,
                    "question_type_raw": turn.get("type", ""),
                    "question_type": canonical,
                    "question": str(turn.get("Question", "")),
                    "options": options,
                    "answer": answer,
                    "annotation_verified": bool(turn.get("annotation", False)),
                }
            )
    return rows


def _balanced_pick(candidates: list[dict], count: int, seed: int) -> list[dict]:
    buckets: dict[str, list[dict]] = defaultdict(list)
    for row in candidates:
        buckets[row["source_dataset"]].append(row)
    for source in buckets:
        buckets[source].sort(key=lambda r: _stable_key(seed, r["private_key"]))
    source_order = sorted(buckets)
    selected: list[dict] = []
    used_categories: set[tuple[str, str]] = set()
    while len(selected) < count and any(buckets.values()):
        made_progress = False
        for source in source_order:
            bucket = buckets[source]
            if not bucket or len(selected) >= count:
                continue
            preferred = next(
                (
                    index
                    for index, row in enumerate(bucket)
                    if (row["source_dataset"], row["category"]) not in used_categories
                ),
                0,
            )
            row = bucket.pop(preferred)
            selected.append(row)
            used_categories.add((row["source_dataset"], row["category"]))
            made_progress = True
        if not made_progress:
            break
    return selected


def build_subset(raw: dict, questions_per_task: int = 20, seed: int = 20260729) -> dict:
    rows = flatten_annotations(raw)
    selected: list[dict] = []
    for task_index, task in enumerate(CANONICAL_TYPES):
        candidates = [row for row in rows if row["question_type"] == task]
        # Only Anomaly Detection is defined on both good and defective images.
        # MMAD's object questions are intentionally attached to good references,
        # while defect questions are attached to anomalous queries.
        if task == "Anomaly Detection":
            normal_count = questions_per_task // 2
            chosen = _balanced_pick(
                [r for r in candidates if r["is_normal"]], normal_count, seed + task_index * 17
            )
            chosen += _balanced_pick(
                [r for r in candidates if not r["is_normal"]],
                questions_per_task - len(chosen),
                seed + task_index * 17 + 1,
            )
        else:
            chosen = _balanced_pick(candidates, questions_per_task, seed + task_index * 17)
        if len(chosen) != questions_per_task:
            raise ValueError(f"not enough valid samples for {task}: {len(chosen)}")
        selected.extend(chosen)

    selected.sort(key=lambda r: (CANONICAL_TYPES.index(r["question_type"]), r["private_key"]))
    image_ids: dict[str, str] = {}
    for row_index, row in enumerate(selected, start=1):
        source_path = row["source_image_path"]
        if source_path not in image_ids:
            suffix = Path(source_path).suffix.lower() or ".png"
            image_ids[source_path] = f"image_{len(image_ids) + 1:04d}{suffix}"
        row["sample_id"] = f"mmad_{row_index:04d}"
        row["image_file"] = f"images/{image_ids[source_path]}"
        row["prompt"] = format_prompt(row["question"], row["options"])

    public_payload = {
        "benchmark": "MMAD",
        "setting": "zero_shot",
        "seed": seed,
        "questions_per_task": questions_per_task,
        "canonical_question_types": list(CANONICAL_TYPES),
        "records": selected,
    }
    canonical = json.dumps(public_payload, sort_keys=True, ensure_ascii=False).encode()
    public_payload["manifest_sha256"] = hashlib.sha256(canonical).hexdigest()
    return public_payload


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    rank = (len(values) - 1) * percentile
    low, high = math.floor(rank), math.ceil(rank)
    if low == high:
        return values[low]
    return values[low] * (high - rank) + values[high] * (rank - low)


def evaluate_records(manifest: dict, predictions: Iterable[dict]) -> tuple[dict, list[dict]]:
    truth = {row["sample_id"]: row for row in manifest["records"]}
    latest = {row.get("sample_id"): row for row in predictions if row.get("sample_id") in truth}
    rows = []
    for sample_id, sample in truth.items():
        pred = latest.get(sample_id, {})
        letter = pred.get("prediction") or parse_prediction(pred.get("raw_response", ""))
        parse_valid = letter in sample["options"]
        rows.append(
            {
                "sample_id": sample_id,
                "model": pred.get("model"),
                "source_dataset": sample["source_dataset"],
                "category": sample["category"],
                "question_type": sample["question_type"],
                "is_normal": sample["is_normal"],
                "truth": sample["answer"],
                "prediction": letter,
                "parse_valid": parse_valid,
                "correct": bool(parse_valid and letter == sample["answer"]),
                "status": pred.get("status", "missing"),
                "latency_seconds": pred.get("latency_seconds"),
            }
        )

    attempted = [row for row in rows if row["status"] != "missing"]
    parsed = [row for row in rows if row["parse_valid"]]
    per_task = {}
    for task in CANONICAL_TYPES:
        group = [row for row in parsed if row["question_type"] == task]
        per_task[task] = {
            "n": len(group),
            "accuracy": sum(row["correct"] for row in group) / len(group) if group else None,
        }
    per_source = {}
    for source in sorted({row["source_dataset"] for row in rows}):
        group = [row for row in parsed if row["source_dataset"] == source]
        per_source[source] = {
            "n": len(group),
            "accuracy": sum(row["correct"] for row in group) / len(group) if group else None,
        }

    detection = [row for row in parsed if row["question_type"] == "Anomaly Detection"]
    tp = fp = tn = fn = 0
    for row in detection:
        sample = truth[row["sample_id"]]
        selected = sample["options"].get(row["prediction"], "").strip().lower()
        predicted_anomaly = selected.startswith("yes")
        actual_anomaly = not sample["is_normal"]
        tp += predicted_anomaly and actual_anomaly
        fp += predicted_anomaly and not actual_anomaly
        tn += not predicted_anomaly and not actual_anomaly
        fn += not predicted_anomaly and actual_anomaly
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = 2 * precision * recall / (precision + recall) if precision and recall else 0.0
    latencies = [float(row["latency_seconds"]) for row in attempted if row["latency_seconds"] is not None]
    task_accuracies = [value["accuracy"] for value in per_task.values() if value["accuracy"] is not None]
    summary = {
        "benchmark": manifest.get("benchmark"),
        "setting": manifest.get("setting"),
        "manifest_sha256": manifest.get("manifest_sha256"),
        "expected_records": len(rows),
        "attempted_records": len(attempted),
        "parsed_records": len(parsed),
        "completion_rate": len(attempted) / len(rows) if rows else 0.0,
        "parse_failure_rate": (len(attempted) - len(parsed)) / len(attempted) if attempted else None,
        "micro_accuracy": sum(row["correct"] for row in parsed) / len(parsed) if parsed else None,
        "macro_task_accuracy": statistics.mean(task_accuracies) if task_accuracies else None,
        "per_task": per_task,
        "per_source": per_source,
        "anomaly_detection": {
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "miss_rate": fn / (tp + fn) if tp + fn else None,
            "overkill_rate": fp / (tn + fp) if tn + fp else None,
        },
        "latency_seconds": {
            "mean": statistics.mean(latencies) if latencies else None,
            "median": statistics.median(latencies) if latencies else None,
            "p95": _percentile(latencies, 0.95),
        },
        "status_counts": dict(Counter(row["status"] for row in rows)),
    }
    return summary, rows


def write_evaluation(output_dir: Path, summary: dict, rows: list[dict]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    fields = list(rows[0]) if rows else ["sample_id"]
    with (output_dir / "predictions_scored.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
