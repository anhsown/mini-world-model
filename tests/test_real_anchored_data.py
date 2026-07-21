import json
import tarfile
from pathlib import Path

import torch

from jwm.dataset_registry import (DatasetAsset, materialize_asset, safe_extract, select_assets,
                                  validate_registry_split_groups)
from jwm.real_anchored_sdg import (RealAnchorProfile,
                                   RealAnchoredSyntheticGeometry,
                                   validate_real_anchored_synthetic)
from jwm.training_metrics_v2 import expected_calibration_error, harmonic_score
from scripts.admit_synthetic_ablation import admission_report


def test_real_anchored_synthetic_is_deterministic_and_disjoint():
    profile = RealAnchorProfile()
    train = RealAnchoredSyntheticGeometry("train", 4, profile, frames=3, size=32)
    val = RealAnchoredSyntheticGeometry("validation", 4, profile, frames=3, size=32)
    left, right = train[0], train[0]
    assert torch.equal(left["image"], right["image"])
    assert torch.equal(left["rigid_flow"], right["rigid_flow"])
    assert train.scene_ids().isdisjoint(val.scene_ids())
    assert validate_real_anchored_synthetic(train, profile, 2)["valid"]


def test_metric_helpers_reward_calibration_and_balance():
    assert expected_calibration_error([.9, .1], [1, 0], bins=2) < .11
    assert harmonic_score([1.0, 1.0, .1]) < .3


def _probe_report(errors, capability, worst, gates=.5):
    keys = ("depth_abs_rel", "ate_metric", "rpe_translation", "track_epe_p90",
            "track_ece")
    return {"report": {"controls": {"normal": dict(zip(keys, errors))},
                       "summary": {"probe_capability_score": capability,
                                   "worst_source_score": worst,
                                   "causal_gate_pass_rate": gates}}}


def test_synthetic_admission_requires_real_and_worst_source_gain():
    base = _probe_report([1, 1, 1, 1, 1], .4, .3)
    mixed = _probe_report([.9, .9, .9, 1.01, 1.01], .42, .3)
    assert admission_report(base, mixed)["valid"]
    collapsed = _probe_report([.9, .9, .9, 1.01, 1.01], .42, .2)
    report = admission_report(base, collapsed)
    assert not report["valid"]
    assert not report["hypotheses"]["H_worst_source_within_tolerance"]


def test_registry_selection():
    asset = DatasetAsset("x", "eye", "source", "real", ("starter",),
                         "train", "scene", "terms", "https://example/x", "file")
    assert select_assets([asset], "starter", branch="eye") == [asset]
    assert not select_assets([asset], "full")


def test_registry_rejects_scene_group_leak():
    common = dict(branch="eye", source="source", kind="real", tier=("starter",),
                  scene_group="same-room", license="terms", url="https://example/x",
                  archive="file")
    left = DatasetAsset(id="a", split="train", **common)
    right = DatasetAsset(id="b", split="test", **common)
    assert not validate_registry_split_groups([left, right])["valid"]


def test_safe_extract_rejects_parent_escape(tmp_path: Path):
    archive = tmp_path / "bad.tgz"
    payload = tmp_path / "payload.txt"; payload.write_text("x")
    with tarfile.open(archive, "w:gz") as package:
        package.add(payload, arcname="../escape.txt")
    try:
        safe_extract(archive, tmp_path / "out", "tgz")
    except ValueError:
        pass
    else:
        raise AssertionError("unsafe archive was accepted")


def test_materialize_resumes_from_extraction_marker_without_archive(tmp_path: Path):
    asset = DatasetAsset("done", "eye", "source", "real", ("starter",),
                         "train", "scene", "terms", "https://invalid/x.tgz",
                         "tgz", size_bytes=10**12)
    output = tmp_path / "raw/source/done"
    output.mkdir(parents=True)
    (output / ".jwm_extracted.json").write_text(
        json.dumps({"asset": "done", "sha256": "abc"}), encoding="utf-8")
    record = materialize_asset(asset, tmp_path, reserve_free_gb=10**6)
    assert record["status"] == "already_ready"
    assert record["sha256"] == "abc"
    assert record["archive"] is None
