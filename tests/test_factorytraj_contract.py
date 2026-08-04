import json
from copy import deepcopy
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator, FormatChecker

from jwm.factorytraj import (
    OUTPUT_SCHEMA_VERSION,
    assert_no_target_leakage,
    factorybench_to_factorytraj,
    model_input_view,
    semantic_errors,
)


ROOT = Path(__file__).resolve().parents[1]
PACK = ROOT / "research" / "factorytraj_bench"


def _schemas():
    item = json.loads(
        (PACK / "factorytraj_item.schema.json").read_text(encoding="utf-8")
    )
    output = json.loads(
        (PACK / "factorytraj_output.schema.json").read_text(encoding="utf-8")
    )
    return item, output


def _samples():
    return json.loads(
        (PACK / "factorybench_samples_v0.1.json").read_text(encoding="utf-8")
    )["items"]


def test_schemas_are_valid_draft_202012():
    item, output = _schemas()
    Draft202012Validator.check_schema(item)
    Draft202012Validator.check_schema(output)


def test_ten_examples_validate_and_cover_levels_and_stream_shapes():
    item_schema, _ = _schemas()
    validator = Draft202012Validator(
        item_schema, format_checker=FormatChecker()
    )
    items = _samples()
    assert len(items) == 10
    assert {item["task"]["causal_level"] for item in items} == {1, 2, 3, 4}
    assert {"single", "paired"}.issubset(
        {item["trajectory"]["stream_relationship"] for item in items}
    )
    assert not [
        error
        for item in items
        for error in validator.iter_errors(item)
    ]
    assert semantic_errors(items) == []
    channels = [
        channel
        for item in items
        for stream in item["model_input"]["observation_window"]["streams"]
        for block_name in ("sensor_history", "control_signals", "machine_context")
        for channel in stream[block_name]["channels"]
    ]
    assert channels
    for channel in channels:
        assert channel["data_type"] in {
            "boolean",
            "integer",
            "float",
            "string",
            "unknown",
        }
        if channel["engineering_unit"] is not None:
            assert set(channel["engineering_unit"]) == {
                "namespace_uri",
                "unit_id",
                "display_name",
                "description",
            }
        assert channel["role"]
        assert isinstance(channel["relationships"], list)
        if channel["data_type"] in {"integer", "float", "boolean"}:
            assert channel["observed_range"]["kind"] == "observed_compact_sample"
        else:
            assert channel["observed_range"] is None


def test_model_view_excludes_ground_truth_and_rejects_injected_target():
    item = _samples()[0]
    payload = model_input_view(item)
    assert "ground_truth" not in payload
    assert "response" not in json.dumps(payload)
    assert "template_id" not in json.dumps(payload)
    assert "causal_level" not in json.dumps(payload)

    poisoned = deepcopy(payload)
    poisoned["model_input"]["root_cause"] = "hidden answer"
    with pytest.raises(ValueError, match="target leakage"):
        assert_no_target_leakage(poisoned)


def test_factorybench_mapping_preserves_paired_stream_provenance():
    source = json.loads(
        (
            ROOT
            / "research"
            / "factorybench"
            / "representative_samples_l1_l4.json"
        ).read_text(encoding="utf-8")
    )
    paired = next(
        item
        for item in source
        if len(item["canonical"]["inputs"]["streams"]) == 2
    )
    mapped = factorybench_to_factorytraj(paired)
    assert mapped["trajectory"]["stream_relationship"] == "paired"
    assert len(mapped["model_input"]["observation_window"]["streams"]) == 2
    assert len(mapped["source"]["source_episodes"]) == 2


def test_semantic_validator_rejects_shape_and_split_leakage():
    first = deepcopy(_samples()[0])
    second = deepcopy(_samples()[1])
    first["model_input"]["observation_window"]["streams"][0]["sensor_history"][
        "values"
    ][0].append(123)
    second["split"]["split_group"] = first["split"]["split_group"]
    second["split"]["name"] = "train"
    errors = semantic_errors([first, second])
    assert any("width" in error for error in errors)
    assert any("split-group leakage" in error for error in errors)


def test_output_contract_requires_abstention_reason():
    _, output_schema = _schemas()
    validator = Draft202012Validator(output_schema)
    prediction = {
        "schema_version": OUTPUT_SCHEMA_VERSION,
        "item_id": "factorybench:test",
        "belief_state": {"state_id": None, "probability": 0.2},
        "evidence": [],
        "next_events": [],
        "root_causes": [],
        "ranked_actions": [],
        "expected_outcomes": [],
        "response": None,
        "confidence": 0.2,
        "abstain": True,
        "abstain_reason": None,
    }
    assert list(validator.iter_errors(prediction))
    prediction["abstain_reason"] = "insufficient evidence"
    assert list(validator.iter_errors(prediction)) == []
