"""FactoryBench adapter and task-aware metrics for Industrial JWM research.

The source dataset stores each time-series row as compact text.  This module
parses that representation without depending on pandas/pyarrow and maps the
channels into the three inputs requested by the Industrial JWM track:

    sensor history + control signals + machine context -> text response

The mapping is deliberately conservative.  Unknown channels remain observable
sensor channels instead of being silently promoted to actions.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import json
import math
import re
from typing import Any, Iterable, Mapping, Sequence


_TIME_ROW = re.compile(r"^\s*t=([^:]+):\s*(.*)$")
_NUMBER = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?")


def channel_role(name: str) -> str:
    """Return ``control``, ``context`` or ``sensor`` for a full channel name."""
    value = name.strip().lower().replace(" ", "_")
    if (
        value.startswith(("setpoint_", "command_", "commanded_", "target_"))
        or "target_torque" in value
        or value.startswith(("digital_output", "control_", "actuator_"))
    ):
        return "control"
    if (
        value.startswith(("robot_mode", "safety_mode", "runtime_state", "task_phase"))
        or value.endswith(("_mode", "_state", "_phase"))
        or value.startswith(("digital_input", "program_state"))
    ):
        return "context"
    return "sensor"


def _scalar(value: str) -> float | int | str | None:
    text = value.strip()
    if text.lower() in {"nan", "none", "null", ""}:
        return None
    if not _NUMBER.fullmatch(text):
        return text
    number = float(text)
    return int(number) if number.is_integer() else number


def parse_time_series_row(row: str) -> tuple[float | int | str, dict[str, Any]]:
    """Parse ``t=<timestamp>: acronym=value, ...`` into timestamp and values."""
    match = _TIME_ROW.match(row)
    if not match:
        raise ValueError(f"invalid FactoryBench time-series row: {row[:80]!r}")
    timestamp = _scalar(match.group(1))
    values: dict[str, Any] = {}
    payload = match.group(2)
    for item in payload.split(","):
        if "=" not in item:
            continue
        key, value = item.split("=", 1)
        values[key.strip()] = _scalar(value)
    return timestamp, values


def _dense_stream(
    rows: Sequence[str], acronym_mapping: Mapping[str, str], role: str
) -> dict[str, Any]:
    parsed = [parse_time_series_row(row) for row in rows]
    selected = [
        acronym
        for acronym, full_name in acronym_mapping.items()
        if channel_role(full_name) == role
    ]
    return {
        "timestamps": [timestamp for timestamp, _ in parsed],
        "channels": [acronym_mapping[name] for name in selected],
        "values": [[values.get(name) for name in selected] for _, values in parsed],
        "validity_mask": [
            [values.get(name) is not None for name in selected] for _, values in parsed
        ],
    }


def context_series(context: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    """Return every telemetry stream in a single- or paired-stream context."""
    if "time_series" in context:
        return {"primary": context}
    streams: dict[str, Mapping[str, Any]] = {}
    for name in ("series_a", "series_b"):
        value = context.get(name)
        if isinstance(value, Mapping) and "time_series" in value:
            streams[name] = value
    return streams


def provenance_episode_keys(provenance: Mapping[str, Any]) -> set[tuple[str, str]]:
    """Extract every source episode referenced by simple and comparative items."""
    keys: set[tuple[str, str]] = set()
    if provenance.get("episode") is not None:
        keys.add(
            (
                str(provenance.get("dataset", "unknown")),
                str(provenance["episode"]),
            )
        )
    for suffix in ("a", "b"):
        episode = provenance.get(f"episode_{suffix}")
        if episode is not None:
            keys.add(
                (
                    str(provenance.get(f"dataset_{suffix}", "unknown")),
                    str(episode),
                )
            )
    for item in provenance.get("episodes") or []:
        if isinstance(item, Mapping) and item.get("episode") is not None:
            keys.add(
                (str(item.get("dataset", "unknown")), str(item["episode"]))
            )
    return keys


def infer_answer_format(record: Mapping[str, Any]) -> str:
    """Infer a stable metric family from the answer/options contract."""
    answer = record.get("answer")
    options = record.get("options") or {}
    template = str(record.get("template_type", "")).lower()
    if isinstance(answer, (int, float)):
        return "numeric"
    text = str(answer).strip()
    if text.startswith("[") and text.endswith("]"):
        try:
            if isinstance(json.loads(text), list):
                return "numeric_array"
        except json.JSONDecodeError:
            pass
    if options and re.fullmatch(r"[TF]+", text):
        return "multilabel_tf"
    if options and re.fullmatch(r"[A-Z]+", text):
        if len(text) > 1 and set(text) == set(options):
            return "ranking"
        return "choice"
    if "troubleshooting" in template or "optimization" in template:
        return "free_form"
    return "text"


def canonicalize_record(
    record: Mapping[str, Any], *, max_steps: int | None = None
) -> dict[str, Any]:
    """Map one source record to the Industrial JWM episode contract."""
    context = record.get("context") or {}
    canonical_streams: dict[str, Any] = {}
    for stream_name, stream in context_series(context).items():
        fmt = stream.get("time_series_format") or {}
        mapping = fmt.get("acronym_mapping") or {}
        rows = list(stream.get("time_series") or [])
        if max_steps is not None and len(rows) > max_steps:
            head = max_steps // 2
            rows = rows[:head] + rows[-(max_steps - head) :]
        canonical_streams[stream_name] = {
            "sensor_history": _dense_stream(rows, mapping, "sensor"),
            "control_signals": _dense_stream(rows, mapping, "control"),
            "machine_context": _dense_stream(rows, mapping, "context"),
            "source_context_description": fmt.get("description"),
        }
    provenance = dict(record.get("provenance") or {})
    return {
        "schema_version": "factorybench-jwm-v1",
        "record_id": record.get("id"),
        "causal_level": record.get("level"),
        "task": {
            "template_id": record.get("template_id"),
            "template_type": record.get("template_type"),
            "question": record.get("question"),
            "options": record.get("options") or {},
            "hidden_fields": record.get("hides") or [],
        },
        "inputs": {
            "streams": canonical_streams,
            "candidate_segments": (
                record.get("options") or {} if not canonical_streams else {}
            ),
            "static_machine_context": {},
        },
        "target": {
            "response": record.get("answer"),
            "answer_format": infer_answer_format(record),
            "acceptance_bounds": record.get("acceptance_bounds"),
            "root_cause": record.get("root_cause"),
        },
        "provenance": provenance,
        "split_group": {
            "source_episodes": [
                {"dataset": dataset, "episode": episode}
                for dataset, episode in sorted(provenance_episode_keys(provenance))
            ],
        },
    }


def normalize_text(value: Any) -> str:
    return " ".join(str(value).strip().lower().split())


def _numbers(value: Any) -> list[float]:
    if isinstance(value, (int, float)):
        return [float(value)]
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, list):
                return [float(item) for item in parsed]
            if isinstance(parsed, (int, float)):
                return [float(parsed)]
        except (json.JSONDecodeError, TypeError, ValueError):
            match = _NUMBER.search(value)
            return [float(match.group())] if match else []
    if isinstance(value, Sequence):
        try:
            return [float(item) for item in value]
        except (TypeError, ValueError):
            return []
    return []


def _token_f1(reference: str, prediction: str) -> float:
    ref = normalize_text(reference).split()
    pred = normalize_text(prediction).split()
    if not ref and not pred:
        return 1.0
    if not ref or not pred:
        return 0.0
    ref_counts: dict[str, int] = defaultdict(int)
    pred_counts: dict[str, int] = defaultdict(int)
    for token in ref:
        ref_counts[token] += 1
    for token in pred:
        pred_counts[token] += 1
    overlap = sum(min(count, pred_counts[token]) for token, count in ref_counts.items())
    precision = overlap / len(pred)
    recall = overlap / len(ref)
    return 2 * precision * recall / (precision + recall) if overlap else 0.0


def _pairwise_ranking_accuracy(reference: str, prediction: str) -> float:
    if set(reference) != set(prediction) or len(reference) < 2:
        return 0.0
    ref_pos = {item: index for index, item in enumerate(reference)}
    pred_pos = {item: index for index, item in enumerate(prediction)}
    pairs = 0
    correct = 0
    symbols = list(reference)
    for i, left in enumerate(symbols):
        for right in symbols[i + 1 :]:
            pairs += 1
            correct += (ref_pos[left] < ref_pos[right]) == (
                pred_pos[left] < pred_pos[right]
            )
    return correct / pairs if pairs else 1.0


def score_factorybench_answer(
    record: Mapping[str, Any], prediction: Any
) -> dict[str, float | str | bool | None]:
    """Score one prediction with metrics appropriate to its answer contract."""
    answer = record.get("answer")
    family = infer_answer_format(record)
    exact = normalize_text(answer) == normalize_text(prediction)
    result: dict[str, float | str | bool | None] = {
        "family": family,
        "exact_match": float(exact),
        "primary_score": float(exact),
    }

    if family in {"numeric", "numeric_array"}:
        reference_values = _numbers(answer)
        predicted_values = _numbers(prediction)
        if len(reference_values) != len(predicted_values) or not reference_values:
            result.update({"mae": math.inf, "within_tolerance": 0.0})
            result["primary_score"] = 0.0
            return result
        errors = [abs(left - right) for left, right in zip(reference_values, predicted_values)]
        result["mae"] = sum(errors) / len(errors)
        bounds = record.get("acceptance_bounds") or {}
        passed: list[bool] = []
        if "margin" in bounds:
            raw_margin = bounds["margin"]
            if isinstance(raw_margin, Sequence) and not isinstance(raw_margin, str):
                margins = [float(item) for item in raw_margin]
            else:
                margins = [float(raw_margin)] * len(errors)
            passed = [
                error <= margin + max(1e-8, abs(margin) * 1e-6)
                for error, margin in zip(errors, margins)
            ]
        elif "min" in bounds and "max" in bounds and len(predicted_values) == 1:
            passed = [float(bounds["min"]) <= predicted_values[0] <= float(bounds["max"])]
        elif "actual_value" in bounds and len(predicted_values) == 1:
            scale = max(abs(float(bounds["actual_value"])), 1.0)
            passed = [errors[0] / scale <= 0.05]
        result["within_tolerance"] = (
            sum(passed) / len(passed) if passed else float(exact)
        )
        result["primary_score"] = result["within_tolerance"]
    elif family == "multilabel_tf":
        ref = str(answer).strip()
        pred = str(prediction).strip().upper()
        if len(ref) == len(pred) and set(pred) <= {"T", "F"}:
            tp = sum(a == b == "T" for a, b in zip(ref, pred))
            fp = sum(a == "F" and b == "T" for a, b in zip(ref, pred))
            fn = sum(a == "T" and b == "F" for a, b in zip(ref, pred))
            denom = 2 * tp + fp + fn
            result["multilabel_f1"] = 2 * tp / denom if denom else 1.0
        else:
            result["multilabel_f1"] = 0.0
        result["primary_score"] = result["multilabel_f1"]
    elif family == "ranking":
        result["pairwise_accuracy"] = _pairwise_ranking_accuracy(
            str(answer).strip(), str(prediction).strip().upper()
        )
        result["primary_score"] = result["pairwise_accuracy"]
    elif family == "free_form":
        result["token_f1"] = _token_f1(str(answer), str(prediction))
        root_cause = record.get("root_cause")
        result["root_cause_hit"] = (
            float(normalize_text(root_cause).replace("_", " ") in normalize_text(prediction))
            if root_cause
            else None
        )
        result["primary_score"] = result["token_f1"]
    return result


@dataclass(frozen=True)
class SplitAudit:
    counts: dict[str, int]
    unique_ids: dict[str, int]
    unique_episodes: dict[str, int]
    id_overlap: dict[str, int]
    episode_overlap: dict[str, int]

    @property
    def valid(self) -> bool:
        return not any(self.id_overlap.values()) and not any(
            self.episode_overlap.values()
        )


def audit_splits(records_by_split: Mapping[str, Iterable[Mapping[str, Any]]]) -> SplitAudit:
    """Check source IDs and source episodes for cross-split leakage."""
    ids: dict[str, set[str]] = {}
    episodes: dict[str, set[tuple[str, str]]] = {}
    counts: dict[str, int] = {}
    for split, records in records_by_split.items():
        ids[split] = set()
        episodes[split] = set()
        counts[split] = 0
        for record in records:
            counts[split] += 1
            ids[split].add(str(record.get("id")))
            provenance = record.get("provenance") or {}
            episodes[split].update(provenance_episode_keys(provenance))
    id_overlap: dict[str, int] = {}
    episode_overlap: dict[str, int] = {}
    names = list(records_by_split)
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            key = f"{left}__{right}"
            id_overlap[key] = len(ids[left] & ids[right])
            episode_overlap[key] = len(episodes[left] & episodes[right])
    return SplitAudit(
        counts=counts,
        unique_ids={key: len(value) for key, value in ids.items()},
        unique_episodes={key: len(value) for key, value in episodes.items()},
        id_overlap=id_overlap,
        episode_overlap=episode_overlap,
    )
