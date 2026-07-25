"""FactoryTraj-Bench v0.1 contract helpers.

The contract deliberately stores model inputs and evaluation ground truth in
separate top-level objects.  Training/inference code must call
``model_input_view`` rather than passing a complete benchmark item to a model.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Iterable, Mapping


SCHEMA_VERSION = "factorytraj-bench-0.1.0"
OUTPUT_SCHEMA_VERSION = "factorytraj-output-0.1.0"
CONTRACT_CREATED_AT = "2026-07-25T00:00:00+00:00"

_FORBIDDEN_INPUT_KEYS = {
    "answer",
    "answer_key",
    "correct_answer",
    "fault_label",
    "ground_truth",
    "hidden_fields",
    "label",
    "root_cause",
    "target",
}


def _source_episodes(canonical: Mapping[str, Any]) -> list[dict[str, str]]:
    episodes = canonical.get("split_group", {}).get("source_episodes", [])
    return [
        {"dataset": str(item["dataset"]), "episode": str(item["episode"])}
        for item in episodes
    ]


def _signal_block(block: Mapping[str, Any]) -> dict[str, Any]:
    channels = [
        {
            "channel_id": str(name),
            "unit": None,
            "sampling_rate_hz": None,
            "description": None,
        }
        for name in block.get("channels", [])
    ]
    return {
        "channels": channels,
        "values": deepcopy(list(block.get("values", []))),
        "validity_mask": deepcopy(list(block.get("validity_mask", []))),
    }


def factorybench_to_factorytraj(sample: Mapping[str, Any]) -> dict[str, Any]:
    """Map one audited FactoryBench sample to FactoryTraj-Bench v0.1."""
    canonical = sample["canonical"]
    source_episodes = _source_episodes(canonical)
    datasets = sorted({item["dataset"] for item in source_episodes})
    stream_relation = (
        "paired" if len(canonical["inputs"]["streams"]) == 2 else "single"
    )
    streams: list[dict[str, Any]] = []
    for stream_id, stream in canonical["inputs"]["streams"].items():
        timestamps = list(stream["sensor_history"].get("timestamps", []))
        streams.append(
            {
                "stream_id": str(stream_id),
                "relation": (
                    "primary"
                    if stream_id in {"primary", "series_a"}
                    else "comparison"
                ),
                "timestamps": timestamps,
                "timestamp_unit": "source_native",
                "sensor_history": _signal_block(stream["sensor_history"]),
                "control_signals": _signal_block(stream["control_signals"]),
                "machine_context": _signal_block(stream["machine_context"]),
                "media": [],
                "data_quality": {
                    "source_validity_mask_preserved": True,
                    "head_tail_sampled": len(timestamps) == 8,
                    "notes": [
                        "Compact visualization sample; benchmark evaluation "
                        "must load the complete source record."
                    ],
                },
            }
        )

    target = canonical["target"]
    provenance = canonical.get("provenance", {})
    question = canonical["task"]["question"]
    record_id = str(canonical["record_id"])
    split_group = "|".join(
        f"{item['dataset']}:{item['episode']}" for item in source_episodes
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "item_id": f"factorybench:{record_id}",
        "source": {
            "dataset": datasets[0] if len(datasets) == 1 else "multi-source",
            "record_id": record_id,
            "uri": "https://huggingface.co/datasets/FactoryBench/FactoryBench",
            "revision": "e2ad55f2c4a66d3f3190170cfe06a50cc93a2e46",
            "license": "CC-BY-NC-4.0",
            "derivation": "audited_test_sample_head_tail_visualization",
            "source_episodes": source_episodes,
            "original_provenance": deepcopy(provenance),
        },
        "trajectory": {
            "trajectory_id": split_group or f"unknown:{record_id}",
            "run_id": None,
            "stream_relationship": stream_relation,
        },
        "model_input": {
            "production_context": {
                "cell_id": None,
                "machine_id": None,
                "asset_type": "industrial_robot",
                "product": None,
                "work_order": None,
                "machine_mode": None,
                "target_rate": None,
                "metadata": {},
            },
            "observation_window": {
                "requested_history_s": 60.0,
                "actual_duration_s": None,
                "streams": streams,
            },
            "candidate_actions": [],
            "sop": [],
            "question": question,
            "options": deepcopy(canonical["task"].get("options", {})),
        },
        "task": {
            "benchmark_tasks": _benchmark_tasks(
                int(canonical["causal_level"]),
                str(canonical["task"].get("template_type", "")),
            ),
            "causal_level": int(canonical["causal_level"]),
            "template_id": str(canonical["task"].get("template_id")),
            "template_type": str(canonical["task"].get("template_type", "")),
            "answer_format": str(target["answer_format"]),
        },
        "ground_truth": {
            "belief_state": {},
            "events": [],
            "root_causes": (
                [str(target["root_cause"])] if target.get("root_cause") else []
            ),
            "action": None,
            "outcome": {},
            "response": deepcopy(target["response"]),
            "acceptance_bounds": deepcopy(target.get("acceptance_bounds")),
            "evidence": [],
            "label_confidence": 1.0,
            "label_provenance": [
                {
                    "method": "FactoryBench published annotation",
                    "source_record_id": record_id,
                }
            ],
        },
        "governance": {
            "permitted_use": "research_only",
            "commercial_training_allowed": False,
            "redistribution_allowed": True,
            "contains_personal_data": False,
        },
        "split": {
            "name": "test_public",
            "split_group": split_group or f"unknown:{record_id}",
            "ood_tags": [],
        },
        "created_at": CONTRACT_CREATED_AT,
    }


def _benchmark_tasks(level: int, template_type: str) -> list[str]:
    template = template_type.lower()
    tasks: list[str] = []
    if level == 1:
        tasks.extend(["B0", "B1"])
    elif level == 2:
        tasks.extend(["B1", "B3"])
    elif level == 3:
        tasks.extend(["B5", "B9"])
    elif level == 4:
        tasks.extend(["B4", "B6", "B8"])
    if "intervention" in template and "B9" not in tasks:
        tasks.append("B9")
    return tasks


def model_input_view(item: Mapping[str, Any]) -> dict[str, Any]:
    """Return the only payload that may be passed to a model."""
    payload = {
        "schema_version": item["schema_version"],
        "item_id": item["item_id"],
        "source": {
            "dataset": item["source"]["dataset"],
            "record_id": item["source"]["record_id"],
        },
        "trajectory": deepcopy(item["trajectory"]),
        "model_input": deepcopy(item["model_input"]),
        "evaluation_task": {
            "benchmark_tasks": deepcopy(item["task"]["benchmark_tasks"]),
            "answer_format": item["task"]["answer_format"],
        },
    }
    assert_no_target_leakage(payload)
    return payload


def assert_no_target_leakage(payload: Mapping[str, Any]) -> None:
    """Reject target-like keys anywhere in a model-facing payload."""
    leaks: list[str] = []

    def visit(value: Any, path: str) -> None:
        if isinstance(value, Mapping):
            for key, child in value.items():
                key_text = str(key)
                child_path = f"{path}.{key_text}"
                if key_text.lower() in _FORBIDDEN_INPUT_KEYS:
                    leaks.append(child_path)
                visit(child, child_path)
        elif isinstance(value, list):
            for index, child in enumerate(value):
                visit(child, f"{path}[{index}]")

    visit(payload, "$")
    if leaks:
        raise ValueError("target leakage in model input: " + ", ".join(leaks))


def semantic_errors(items: Iterable[Mapping[str, Any]]) -> list[str]:
    """Check invariants JSON Schema cannot conveniently express."""
    errors: list[str] = []
    ids: set[str] = set()
    group_splits: dict[str, str] = {}
    for item in items:
        item_id = str(item.get("item_id", "<missing>"))
        if item_id in ids:
            errors.append(f"{item_id}: duplicate item_id")
        ids.add(item_id)

        try:
            model_input_view(item)
        except (KeyError, ValueError) as exc:
            errors.append(f"{item_id}: {exc}")

        for stream in (
            item.get("model_input", {})
            .get("observation_window", {})
            .get("streams", [])
        ):
            timestamps = stream.get("timestamps", [])
            if any(
                isinstance(a, (int, float))
                and isinstance(b, (int, float))
                and b < a
                for a, b in zip(timestamps, timestamps[1:])
            ):
                errors.append(
                    f"{item_id}:{stream.get('stream_id')}: timestamps not monotonic"
                )
            for role in (
                "sensor_history",
                "control_signals",
                "machine_context",
            ):
                block = stream.get(role, {})
                channels = block.get("channels", [])
                values = block.get("values", [])
                masks = block.get("validity_mask", [])
                if len(values) != len(timestamps) or len(masks) != len(timestamps):
                    errors.append(
                        f"{item_id}:{stream.get('stream_id')}:{role}: "
                        "row count differs from timestamps"
                    )
                width = len(channels)
                for row_index, row in enumerate(values):
                    if len(row) != width:
                        errors.append(
                            f"{item_id}:{stream.get('stream_id')}:{role}: "
                            f"values[{row_index}] width {len(row)} != {width}"
                        )
                for row_index, row in enumerate(masks):
                    if len(row) != width:
                        errors.append(
                            f"{item_id}:{stream.get('stream_id')}:{role}: "
                            f"validity_mask[{row_index}] width {len(row)} != {width}"
                        )

        split = item.get("split", {}).get("name")
        group = item.get("split", {}).get("split_group")
        if group in group_splits and group_splits[group] != split:
            errors.append(f"{item_id}: split-group leakage for {group}")
        if group:
            group_splits[group] = split
        if (
            item.get("governance", {}).get("commercial_training_allowed")
            and item.get("source", {}).get("license") == "CC-BY-NC-4.0"
        ):
            errors.append(f"{item_id}: non-commercial source marked commercial")
    return errors
