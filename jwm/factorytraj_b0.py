"""FactoryTraj B0 metrics and transparent seed baselines."""

from __future__ import annotations

from collections import Counter
from math import isfinite
from typing import Any, Iterable


ROLES = (
    "sensor_feedback",
    "control_setpoint",
    "actuator_effort",
    "machine_context",
    "identifier",
    "time",
)


def _macro_f1(gold: list[str], pred: list[str]) -> float:
    labels = sorted(set(gold) | set(pred))
    scores = []
    for label in labels:
        tp = sum(g == label and p == label for g, p in zip(gold, pred))
        fp = sum(g != label and p == label for g, p in zip(gold, pred))
        fn = sum(g == label and p != label for g, p in zip(gold, pred))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        scores.append(
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
    return sum(scores) / len(scores) if scores else 0.0


def _relation_key(relation: dict[str, Any]) -> tuple[str, str]:
    return relation["relation"], relation["target_tag_id"]


def _set_f1(gold: set[Any], pred: set[Any]) -> float:
    if not gold and not pred:
        return 1.0
    if not gold or not pred:
        return 0.0
    tp = len(gold & pred)
    precision = tp / len(pred)
    recall = tp / len(gold)
    return (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )


def _range_score(gold: dict[str, float], pred: dict[str, float] | None) -> float:
    if not pred:
        return 0.0
    low, high = gold["low"], gold["high"]
    pred_low, pred_high = pred.get("low"), pred.get("high")
    if not all(
        isinstance(value, (int, float)) and isfinite(value)
        for value in (pred_low, pred_high)
    ):
        return 0.0
    width = max(abs(high - low), 1e-12)
    nre = (abs(pred_low - low) + abs(pred_high - high)) / (2 * width)
    return 1.0 - min(1.0, nre)


def score_b0(
    items: list[dict[str, Any]], predictions: list[dict[str, Any]]
) -> dict[str, Any]:
    """Score predictions, excluding genuinely unavailable labels from metrics."""
    pred_by_id = {prediction["item_id"]: prediction for prediction in predictions}
    missing = [item["item_id"] for item in items if item["item_id"] not in pred_by_id]
    if missing:
        raise ValueError(f"missing predictions for {len(missing)} items")

    gold_types, pred_types = [], []
    gold_roles, pred_roles = [], []
    unit_scores, range_scores, relation_scores = [], [], []
    for item in items:
        gold = item["ground_truth"]
        pred = pred_by_id[item["item_id"]]
        gold_types.append(gold["data_type"])
        pred_types.append(pred.get("data_type", "unknown"))
        gold_roles.append(gold["role"])
        pred_roles.append(pred.get("role", "unknown"))
        if gold.get("engineering_unit") is not None:
            unit_scores.append(
                float(pred.get("engineering_unit") == gold["engineering_unit"])
            )
        authoritative_range = gold.get("instrument_range") or gold.get("eu_range")
        if authoritative_range is not None:
            range_scores.append(_range_score(authoritative_range, pred.get("range")))
        if gold.get("relationships"):
            relation_scores.append(
                _set_f1(
                    {_relation_key(value) for value in gold["relationships"]},
                    {
                        _relation_key(value)
                        for value in pred.get("relationships", [])
                        if "relation" in value and "target_tag_id" in value
                    },
                )
            )

    components: dict[str, float | None] = {
        "data_type_exact_accuracy": sum(
            gold == pred for gold, pred in zip(gold_types, pred_types)
        )
        / len(items),
        "unit_exact_accuracy": (
            sum(unit_scores) / len(unit_scores) if unit_scores else None
        ),
        "role_macro_f1": _macro_f1(gold_roles, pred_roles),
        "range_score": (
            sum(range_scores) / len(range_scores) if range_scores else None
        ),
        "relationship_macro_f1": (
            sum(relation_scores) / len(relation_scores)
            if relation_scores
            else None
        ),
    }
    available = [value for value in components.values() if value is not None]
    coverage = {
        "data_type": 1.0,
        "engineering_unit": len(unit_scores) / len(items),
        "authoritative_range": len(range_scores) / len(items),
        "role": 1.0,
        "relationships": len(relation_scores) / len(items),
    }
    return {
        "b0_contract_macro_score": sum(available) / len(available),
        "components": components,
        "label_coverage": coverage,
        "items": len(items),
    }


def majority_role_baseline(
    train: list[dict[str, Any]], test: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    role = Counter(item["ground_truth"]["role"] for item in train).most_common(1)[0][0]
    data_type = Counter(
        item["ground_truth"]["data_type"] for item in train
    ).most_common(1)[0][0]
    return [
        {
            "item_id": item["item_id"],
            "data_type": data_type,
            "engineering_unit": None,
            "range": None,
            "role": role,
            "relationships": [],
        }
        for item in test
    ]


def _unit_from_name(name: str) -> str | None:
    patterns = (
        ("execution_time", "s"),
        ("time_s", "s"),
        ("temp", "Cel"),
        ("voltage", "V"),
        ("current", "A"),
        ("torque", "N.m"),
        ("force", "N"),
        ("_vel_", "rad/s"),
        ("vel_", "rad/s"),
        ("_acc_", "rad/s2"),
        ("acc_", "rad/s2"),
        ("_pos_", "rad"),
        ("pos_", "rad"),
    )
    for token, unit in patterns:
        if token in name:
            return unit
    return None


def _role_from_name(name: str) -> str:
    if name in {"time_s", "timestamp"}:
        return "time"
    if name in {"episode_id", "dataset_source", "machine_type"}:
        return "identifier"
    if name.startswith("setpoint_"):
        return "control_setpoint"
    if name.startswith("effort_"):
        return "actuator_effort"
    if name.startswith("feedback_") or name.startswith("auxiliary_"):
        return "sensor_feedback"
    if name.startswith("ctx_"):
        return "machine_context"
    return "machine_context"


def _role_from_documentation(documentation: str | None) -> str:
    """Infer role from partial documentation without observing the tag name."""
    text = (documentation or "").lower()
    if any(token in text for token in ("commanded", "setpoint", "target command")):
        return "control_setpoint"
    if any(
        token in text
        for token in ("actuator effort", "motor current", "motor voltage", "applied force")
    ):
        return "actuator_effort"
    if any(
        token in text
        for token in (
            "measured",
            "feedback",
            "sensor reading",
            "flow",
            "pressure",
            "temperature",
            "level",
            "composition",
        )
    ):
        return "sensor_feedback"
    if any(token in text for token in ("identifier", "episode id", "source id")):
        return "identifier"
    if any(token in text for token in ("timestamp", "elapsed time")):
        return "time"
    return "machine_context"


def _paired_target(name: str, candidates: set[str]) -> list[dict[str, str]]:
    pairs = []
    if name.startswith("setpoint_"):
        target = "feedback_" + name[len("setpoint_") :]
        if target in candidates:
            pairs.append({"relation": "commands", "target_tag_id": target})
    elif name.startswith("feedback_"):
        target = "setpoint_" + name[len("feedback_") :]
        if target in candidates:
            pairs.append({"relation": "tracks", "target_tag_id": target})
    return pairs


def rule_baseline(
    items: list[dict[str, Any]],
    *,
    anonymized: bool = False,
    use_documentation: bool = False,
) -> list[dict[str, Any]]:
    """Transparent parser; anonymized mode measures tag-name shortcut reliance."""
    names = {item["input"]["tag_name"] for item in items}
    predictions = []
    for item in items:
        input_data = item["input"]
        name = input_data["anonymized_tag_id"] if anonymized else input_data["tag_name"]
        role = (
            _role_from_documentation(input_data.get("documentation"))
            if use_documentation
            else _role_from_name(name)
        )
        predictions.append(
            {
                "item_id": item["item_id"],
                "data_type": input_data.get("declared_data_type", "unknown"),
                "engineering_unit": _unit_from_name(name),
                "range": None,
                "role": role,
                "relationships": _paired_target(name, names),
            }
        )
    return predictions


def split_items(
    items: Iterable[dict[str, Any]], split: str
) -> list[dict[str, Any]]:
    return [item for item in items if item["split"] == split]
