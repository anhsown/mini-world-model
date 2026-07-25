"""Validate the FactoryTraj-Bench B0-B10 task and metric contract."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PACK = ROOT / "research" / "factorytraj_bench"
CONTRACT = PACK / "factorytraj_task_contracts_v0.1.json"
REPORT = PACK / "factorytraj_task_contracts_validation_v0.1.json"

EXPECTED_IDS = [f"B{i}" for i in range(11)]
REQUIRED_FIELDS = {
    "id",
    "capability",
    "research_question",
    "required_inputs",
    "required_ground_truth",
    "required_outputs",
    "primary_metric",
    "metric_components",
    "mandatory_controls",
    "baseline_families",
    "dataset_coverage",
    "gate",
}


def validate_contract(contract: dict) -> dict:
    errors: list[str] = []
    tasks = contract.get("tasks", [])
    by_id = {task.get("id"): task for task in tasks}

    if [task.get("id") for task in tasks] != EXPECTED_IDS:
        errors.append("tasks must appear exactly once and in order B0-B10")

    for task in tasks:
        task_id = task.get("id", "<missing>")
        missing = sorted(REQUIRED_FIELDS - set(task))
        if missing:
            errors.append(f"{task_id}: missing fields {missing}")
        for field in (
            "required_inputs",
            "required_ground_truth",
            "required_outputs",
            "metric_components",
            "mandatory_controls",
            "baseline_families",
            "dataset_coverage",
        ):
            if not task.get(field):
                errors.append(f"{task_id}: {field} must be non-empty")
        if not isinstance(task.get("primary_metric"), str):
            errors.append(f"{task_id}: primary_metric must be a string")
        if not isinstance(task.get("gate"), (dict, str)) or not task.get("gate"):
            errors.append(f"{task_id}: gate must be a non-empty rule")

    b0 = by_id.get("B0", {})
    required_b0_outputs = {
        "data_type",
        "engineering_unit",
        "range",
        "role",
        "relationships",
    }
    b0_outputs = b0.get("required_outputs", {})
    if isinstance(b0_outputs, dict):
        b0_outputs = {
            field
            for fields in b0_outputs.values()
            for field in fields
        }
    else:
        b0_outputs = set(b0_outputs)
    if not required_b0_outputs.issubset(b0_outputs):
        errors.append("B0: output contract does not cover the full tag contract")

    control_text = {
        task_id: " ".join(by_id.get(task_id, {}).get("mandatory_controls", []))
        .lower()
        for task_id in EXPECTED_IDS
    }
    for task_id in ("B1", "B3", "B4", "B5", "B8", "B9"):
        text = control_text[task_id]
        if not any(term in text for term in ("shuffle", "zero", "blind", "only")):
            errors.append(f"{task_id}: missing shortcut-control ablation")

    b9_text = control_text["B9"]
    if "action" not in b9_text or "observation" not in b9_text:
        errors.append("B9: must compare action-conditioned and observation-only models")

    b10 = by_id.get("B10", {})
    b10_metrics = " ".join(
        [str(b10.get("primary_metric", "")), *b10.get("metric_components", [])]
    ).lower()
    if "risk" not in b10_metrics or "unsafe" not in b10_metrics:
        errors.append("B10: risk-coverage and unsafe-confidence metrics are mandatory")

    basis = contract.get("research_basis", [])
    if len(basis) < 5 or any("url" not in entry for entry in basis):
        errors.append("research_basis: at least five URL-backed references are required")

    hypotheses = {
        "H_complete_B0_B10": [task.get("id") for task in tasks] == EXPECTED_IDS,
        "H_every_task_has_io_gt_metric": not any(
            "missing fields" in error or "must be non-empty" in error
            for error in errors
        ),
        "H_B0_is_operational": not any(error.startswith("B0:") for error in errors),
        "H_causal_tasks_have_controls": not any(
            error.startswith(tuple(f"{task_id}:" for task_id in ("B1", "B3", "B4", "B5", "B8", "B9")))
            for error in errors
        ),
        "H_OOD_abstention_is_safety_aware": not any(
            error.startswith("B10:") for error in errors
        ),
        "H_research_basis_is_traceable": not any(
            error.startswith("research_basis:") for error in errors
        ),
    }
    return {
        "valid": not errors and all(hypotheses.values()),
        "schema_version": contract.get("schema_version"),
        "task_count": len(tasks),
        "task_ids": [task.get("id") for task in tasks],
        "hypotheses": hypotheses,
        "errors": errors,
    }


def main() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    report = validate_contract(contract)
    REPORT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
