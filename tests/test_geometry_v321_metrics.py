import torch

from jwm.geometry_v3_trainer import track_metrics
from jwm.geometry_math_v3 import resize_flow_with_valid
from jwm.configs import eye_physical_v32_smoke_scale
from scripts.train_eye_v3_ddp import (
    STAGES, V3Sampler, apply_stage_weight_profile, validate_stage_mixtures,
)


def test_confidence_contract_reports_pck1_and_pck3_separately():
    # A two-pixel error is wrong at PCK1 but correct at PCK3. Confidence=1
    # must therefore be calibrated for the BA/PCK3 contract, not PCK1.
    output = {
        "feature_hw": (4, 4),
        "track_source": torch.tensor([[[[0.0, 0.0]]]]),
        "track_target": torch.tensor([[[[2.0, 0.0]]]]),
        "track_confidence": torch.ones(1, 1, 1),
        "dynamic_probability": torch.zeros(1, 1, 1),
    }
    batch = {
        "rigid_flow": torch.zeros(1, 1, 4, 4, 2),
        "rigid_flow_valid": torch.ones(1, 1, 4, 4, dtype=torch.bool),
        "dynamic_mask": torch.zeros(1, 2, 4, 4),
    }
    metrics = track_metrics(output, batch)
    assert metrics["track_pck1"] == 0.0
    assert metrics["track_pck3"] == 1.0
    assert metrics["track_ece"] == 0.0
    assert metrics["track_ece_pck1"] == 1.0


def test_invalid_flow_sentinel_cannot_bleed_into_valid_resize():
    flow = torch.ones(1, 1, 8, 8, 2)
    valid = torch.ones(1, 1, 8, 8, dtype=torch.bool)
    flow[:, :, :, 4:] = 1e9
    valid[:, :, :, 4:] = False
    resized, mask = resize_flow_with_valid(flow, valid, (4, 4))
    assert torch.isfinite(resized).all()
    assert float(resized[mask[:, :, None].expand_as(resized)].max()) < 2.0


def test_g0_fallback_cannot_silently_become_all_synthetic():
    datasets = {"procedural": [0], "tum": [0], "bonn": [0]}
    sampler = V3Sampler(datasets, world=1, rank=0, seed=7)
    weights = sampler.weights(STAGES[0][5])
    assert weights["tum"] + weights["bonn"] >= 0.50
    assert weights["procedural"] < 0.50


def test_stage_mixture_contract_blocks_synthetic_only_fallback():
    sampler = V3Sampler({"procedural": [0]}, world=1, rank=0, seed=7)
    report = validate_stage_mixtures(sampler)
    assert not report["valid"]
    assert set(report["failures"]) == {stage[0] for stage in STAGES}


def test_stage_profiles_prioritize_the_capability_being_learned():
    cfg = eye_physical_v32_smoke_scale()
    g0 = apply_stage_weight_profile(cfg, 0)
    assert g0["geometry_depth_absrel_weight"] > g0["geometry_rel_pose_weight"]
    g1 = apply_stage_weight_profile(cfg, 1)
    assert g1["geometry_initial_pose_weight"] >= g1["geometry_depth_weight"]
    g2 = apply_stage_weight_profile(cfg, 2)
    assert g2["geometry_dynamic_weight"] > g1["geometry_dynamic_weight"]
    g3 = apply_stage_weight_profile(cfg, 3)
    assert g3["geometry_temporal_weight"] >= g2["geometry_temporal_weight"]
