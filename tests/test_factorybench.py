import pytest

from jwm.factorybench import (
    audit_splits,
    canonicalize_record,
    channel_role,
    provenance_episode_keys,
    parse_time_series_row,
    score_factorybench_answer,
)


def _record(answer="A", options=None, bounds=None, root_cause=None):
    return {
        "id": "x",
        "level": 1,
        "template_id": 1,
        "template_type": "identification",
        "hides": [],
        "question": "Which state?",
        "options": options or {"A": "normal", "B": "fault"},
        "answer": answer,
        "acceptance_bounds": bounds,
        "root_cause": root_cause,
        "provenance": {"dataset": "factorywave", "episode": "e1"},
        "context": {
            "time_series_format": {
                "description": "test",
                "acronym_mapping": {
                    "sp": "setpoint_pos_0",
                    "fb": "feedback_pos_0",
                    "rm": "robot_mode",
                },
            },
            "time_series": ["t=0: sp=1, fb=0.9, rm=7", "t=1: sp=2, fb=null, rm=7"],
        },
    }


def test_channel_roles_and_row_parser():
    assert channel_role("setpoint_pos_0") == "control"
    assert channel_role("effort_target_torque_0") == "control"
    assert channel_role("robot_mode") == "context"
    assert channel_role("feedback_pos_0") == "sensor"
    timestamp, values = parse_time_series_row("t=10.5: sp=1, fb=-2.5, x=null")
    assert timestamp == 10.5
    assert values == {"sp": 1, "fb": -2.5, "x": None}


def test_canonical_contract_preserves_roles_and_masks():
    item = canonicalize_record(_record())
    inputs = item["inputs"]["streams"]["primary"]
    assert inputs["control_signals"]["channels"] == ["setpoint_pos_0"]
    assert inputs["sensor_history"]["channels"] == ["feedback_pos_0"]
    assert inputs["machine_context"]["channels"] == ["robot_mode"]
    assert inputs["sensor_history"]["validity_mask"] == [[True], [False]]


def test_paired_series_and_provenance_are_not_dropped():
    item = _record()
    primary = item["context"]
    item["context"] = {"series_a": primary, "series_b": primary}
    item["provenance"] = {
        "dataset_a": "aursad",
        "episode_a": "a",
        "dataset_b": "factorywave",
        "episode_b": "b",
    }
    canonical = canonicalize_record(item)
    assert set(canonical["inputs"]["streams"]) == {"series_a", "series_b"}
    assert provenance_episode_keys(item["provenance"]) == {
        ("aursad", "a"),
        ("factorywave", "b"),
    }


def test_task_specific_metrics():
    assert score_factorybench_answer(_record(), "A")["primary_score"] == 1
    tf = _record(answer="TFFT", options={"A": 1, "B": 2, "C": 3, "D": 4})
    assert score_factorybench_answer(tf, "TFFT")["multilabel_f1"] == 1
    ranking = _record(answer="ABCD", options={"A": 1, "B": 2, "C": 3, "D": 4})
    score = score_factorybench_answer(ranking, "ACBD")
    assert score["pairwise_accuracy"] == pytest.approx(5 / 6)
    numeric = _record(answer="[1,2]", options={}, bounds={"margin": [0.2, 0.1]})
    score = score_factorybench_answer(numeric, "[1.1,2.2]")
    assert score["within_tolerance"] == 0.5
    numeric["acceptance_bounds"] = {"margin": 0.2}
    assert score_factorybench_answer(numeric, "[1.1,2.2]")["within_tolerance"] == 1
    free = _record(
        answer="Stop the machine and inspect joint two.",
        options={},
        root_cause="joint_friction",
    )
    free["template_type"] = "troubleshooting"
    score = score_factorybench_answer(free, "Joint friction: stop and inspect joint two.")
    assert score["root_cause_hit"] == 1
    assert score["token_f1"] > 0


def test_split_audit_rejects_episode_leakage():
    train = [_record()]
    test = [_record()]
    test[0]["id"] = "different"
    report = audit_splits({"train": train, "test": test})
    assert not report.valid
    assert report.id_overlap["train__test"] == 0
    assert report.episode_overlap["train__test"] == 1
