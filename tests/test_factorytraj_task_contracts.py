import json
from pathlib import Path

from scripts.validate_factorytraj_task_contracts import EXPECTED_IDS, validate_contract


ROOT = Path(__file__).resolve().parents[1]
PACK = ROOT / "research" / "factorytraj_bench"


def _contract():
    return json.loads(
        (PACK / "factorytraj_task_contracts_v0.1.json").read_text(encoding="utf-8")
    )


def test_task_contract_passes_all_admission_hypotheses():
    report = validate_contract(_contract())
    assert report["valid"], report["errors"]
    assert all(report["hypotheses"].values())


def test_task_ids_and_primary_metrics_are_complete():
    tasks = _contract()["tasks"]
    assert [task["id"] for task in tasks] == EXPECTED_IDS
    assert len({task["primary_metric"] for task in tasks}) == 11


def test_b0_matches_operational_schema_and_tag_contract():
    b0 = _contract()["tasks"][0]
    outputs = {
        field
        for fields in b0["required_outputs"].values()
        for field in fields
    }
    assert {
        "data_type",
        "engineering_unit",
        "range",
        "role",
        "relationships",
    }.issubset(outputs)
    assert "b0_contract_macro_score" == b0["primary_metric"]


def test_action_causality_and_ood_safety_cannot_be_hidden():
    tasks = {task["id"]: task for task in _contract()["tasks"]}
    b9_controls = " ".join(tasks["B9"]["mandatory_controls"]).lower()
    assert "observation_only" in b9_controls
    assert "shuffled_action" in b9_controls
    b10_metrics = " ".join(
        [tasks["B10"]["primary_metric"], *tasks["B10"]["metric_components"]]
    ).lower()
    assert "risk" in b10_metrics
    assert "unsafe" in b10_metrics
