"""Build a provenance-explicit B0 seed set from the published FactoryNet schema."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jwm.opcua_nodeset_b0 import extract_official_nodeset_items

OUT = ROOT / "research" / "factorytraj_bench" / "b0_seed_v0.1.json"

FACTORYNET_CARD = "https://huggingface.co/datasets/factorynet/factorynet/blob/main/README.md"
FACTORYNET_VIEWER = "https://huggingface.co/datasets/Forgis/FactoryNet"
TE_SOURCE = (
    ROOT
    / "research"
    / "datasets"
    / "external_official"
    / "tennessee-eastman-dataset"
    / "simulator"
    / "temexd_mod.c"
)
OPC_NODESETS = (
    ROOT
    / "research"
    / "datasets"
    / "external_official"
    / "opc-ua-nodeset"
)

DTYPE = {
    "continuous": "float64",
    "mode": "int64",
    "flag": "uint8",
    "text": "string",
}


def unit_for(name: str) -> str | None:
    for token, unit in (
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
    ):
        if token in name and "cartesian" not in name:
            return unit
    return None


def role_for(name: str) -> str:
    if name == "time_s":
        return "time"
    if name in {"dataset_source", "machine_type", "episode_id"}:
        return "identifier"
    if name.startswith("setpoint_"):
        return "control_setpoint"
    if name.startswith("effort_"):
        return "actuator_effort"
    if name.startswith(("feedback_", "auxiliary_")):
        return "sensor_feedback"
    return "machine_context"


def dtype_for(name: str) -> str:
    if name in {"dataset_source", "machine_type", "ctx_anomaly_label", "episode_id"}:
        return DTYPE["text"]
    if name.startswith("ctx_state_"):
        return DTYPE["flag"]
    if name.startswith("ctx_") and name.endswith(("_mode", "_state")):
        return DTYPE["mode"]
    if "joint_mode" in name:
        return DTYPE["mode"]
    return DTYPE["continuous"]


def relationships(name: str, names: set[str]) -> list[dict[str, str]]:
    if name.startswith("setpoint_"):
        target = "feedback_" + name[len("setpoint_") :]
        if target in names:
            return [{"relation": "commands", "target_tag_id": target}]
    if name.startswith("feedback_"):
        target = "setpoint_" + name[len("feedback_") :]
        if target in names:
            return [{"relation": "tracks", "target_tag_id": target}]
    return []


def documentation_for(name: str) -> str:
    """Partial operator-style documentation derived from the published S-E-F-C schema."""
    quantity = name.replace("_", " ")
    if name == "time_s":
        return "Elapsed time for the trajectory sample."
    if name in {"dataset_source", "machine_type", "episode_id"}:
        return "Identifier metadata for the trajectory source, machine, or episode."
    if name.startswith("setpoint_"):
        return f"Commanded setpoint for {quantity.removeprefix('setpoint ')}."
    if name.startswith("effort_"):
        return f"Actuator effort for {quantity.removeprefix('effort ')}."
    if name.startswith(("feedback_", "auxiliary_")):
        prefix = "feedback " if name.startswith("feedback_") else "auxiliary "
        return f"Measured sensor feedback for {quantity.removeprefix(prefix)}."
    return f"Machine context or runtime state for {quantity.removeprefix('ctx ')}."


def build_names() -> list[str]:
    names = [
        "ctx_execution_time",
        "ctx_main_voltage",
        "ctx_robot_voltage",
        "ctx_robot_current",
        "ctx_speed_scaling",
        "ctx_target_speed_fraction",
        "ctx_momentum",
        "ctx_robot_mode",
        "ctx_safety_mode",
        "ctx_runtime_state",
        "feedback_torque_tool",
        "effort_torque_tool",
        "setpoint_torque_tool",
        "setpoint_torque_gradient_tool",
        "ctx_state_move_to_pin",
        "ctx_state_move_to_home",
        "ctx_state_loosening",
        "ctx_state_tightening",
        "ctx_state_screwdriver_busy",
        "ctx_state_process_nok",
        "ctx_state_process_ok",
    ]
    for joint in range(6):
        names.extend(
            [
                f"setpoint_pos_{joint}",
                f"setpoint_vel_{joint}",
                f"setpoint_acc_{joint}",
                f"setpoint_current_{joint}",
                f"setpoint_torque_{joint}",
                f"feedback_pos_{joint}",
                f"feedback_vel_{joint}",
                f"effort_current_{joint}",
                f"effort_voltage_{joint}",
                f"ctx_temp_{joint}",
                f"ctx_joint_mode_{joint}",
                f"setpoint_pos_cartesian_{joint}",
                f"setpoint_vel_cartesian_{joint}",
                f"feedback_pos_cartesian_{joint}",
                f"feedback_vel_cartesian_{joint}",
                f"effort_force_cartesian_{joint}",
            ]
        )
    names.extend(
        [
            "auxiliary_accel_tool_0",
            "auxiliary_accel_tool_1",
            "auxiliary_accel_tool_2",
            "time_s",
            "dataset_source",
            "machine_type",
            "ctx_anomaly_label",
            "episode_id",
        ]
    )
    return names


def build_tennessee_eastman(start_index: int) -> list[dict]:
    text = TE_SOURCE.read_text(encoding="utf-8", errors="replace")
    measured = text.split("Output 1 - Measured Values", 1)[1].split(
        "Output 2 - Monitoring", 1
    )[0]
    measured_rows = re.findall(
        r"^\s+(\d+)\s+\|([^|\r\n]+)\|([^|\r\n]+)\s*$",
        measured,
        flags=re.MULTILINE,
    )
    inputs = text.split("Inputs (12 manipulated variables", 1)[1].split(
        "Number|Type", 1
    )[0]
    input_rows = re.findall(
        r"^\s+(\d+)\s+\|([^|\r\n]+?)\s*$", inputs, flags=re.MULTILINE
    )
    items = []
    rows = [
        (
            f"XMEAS_{number}",
            description.strip(),
            unit.strip(),
            "sensor_feedback",
            (
                {"low": 0.0, "high": 100.0}
                if "mol %" in unit.lower()
                else None
            ),
        )
        for number, description, unit in measured_rows
    ]
    rows += [
        (
            f"XMV_{number}",
            description.strip(),
            "%",
            "control_setpoint",
            {"low": 0.0, "high": 100.0},
        )
        for number, description in input_rows
        if 1 <= int(number) <= 12
    ]
    for offset, (name, description, unit, role, eu_range) in enumerate(rows):
        items.append(
            {
                "item_id": f"tennessee_eastman:{name}",
                "source_dataset": "TennesseeEastman",
                "split": "test",
                "input": {
                    "tag_name": name,
                    "anonymized_tag_id": f"tag_{start_index + offset:04d}",
                    "declared_data_type": "float64",
                    "representative_samples": [],
                    "documentation": description,
                },
                "ground_truth": {
                    "data_type": "float64",
                    "engineering_unit": unit,
                    "instrument_range": None,
                    "eu_range": eu_range,
                    "role": role,
                    "relationships": [],
                },
                "provenance": {
                    "schema_and_type": str(TE_SOURCE.relative_to(ROOT)),
                    "role": "simulator source-code Inputs/Outputs contract",
                    "unit": "simulator source-code table",
                    "range": (
                        (
                            "simulator composition output is a physical mole fraction multiplied by 100"
                            if role == "sensor_feedback" and eu_range
                            else "simulator initialization states all XMV ranges are 0-100"
                        )
                        if eu_range is not None
                        else None
                    ),
                    "relationship": None,
                },
            }
        )
    return items


def main() -> None:
    names = build_names()
    name_set = set(names)
    items = []
    for index, name in enumerate(names):
        # Whole semantic families are held out together where possible.
        if "cartesian" in name or name.startswith("auxiliary_"):
            split = "test"
        elif index % 10 == 0:
            split = "validation"
        else:
            split = "train"
        dtype = dtype_for(name)
        unit = unit_for(name)
        items.append(
            {
                "item_id": f"factorynet:{name}",
                "source_dataset": "FactoryNet",
                "split": split,
                "input": {
                    "tag_name": name,
                    "anonymized_tag_id": f"tag_{index:04d}",
                    "declared_data_type": dtype,
                    "representative_samples": [],
                    "documentation": documentation_for(name),
                },
                "ground_truth": {
                    "data_type": dtype,
                    "engineering_unit": unit,
                    "instrument_range": None,
                    "eu_range": None,
                    "role": role_for(name),
                    "relationships": relationships(name, name_set),
                },
                "provenance": {
                    "schema_and_type": FACTORYNET_VIEWER,
                    "role": FACTORYNET_CARD,
                    "unit": (
                        "quantity-name convention; requires source-owner review"
                        if unit is not None
                        else None
                    ),
                    "range": None,
                    "relationship": "S-E-F-C naming correspondence",
                },
            }
        )
    items.extend(build_tennessee_eastman(len(items)))
    items.extend(extract_official_nodeset_items(OPC_NODESETS, len(items)))
    payload = {
        "schema_version": "factorytraj-b0-seed-0.1.0",
        "status": "research_seed_not_yet_authoritative",
        "items": items,
        "limitations": [
            "authoritative ranges exist only for Tennessee Eastman XMV controls",
            "unit labels inferred from quantity names and require owner review",
            "tag names expose S-E-F-C role prefixes",
            "FactoryNet partial documentation is generated only from its published S-E-F-C field semantics",
            "Tennessee Eastman is simulated process data, not a real machine",
            "OPC Foundation records are normative schema metadata, not live trajectories",
        ],
    }
    OUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(items)} B0 items -> {OUT}")


if __name__ == "__main__":
    main()
