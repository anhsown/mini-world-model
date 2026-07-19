import math

import torch

from jwm import JWM, JWMConfig
from jwm.geometry_data import render_geometry_sequence
from jwm.geometry_trainer import (constant_identity_metrics_v2,
                                  evaluate_geometry_controls,
                                  geometry_batch_metrics)


def test_geometry_metrics_are_finite():
    cfg = JWMConfig(d_model=32, n_layers=1, n_heads=4, head_dim=8,
                    ffn_hidden=64, rope_sections=(2, 1, 1), image_size=16,
                    patch=8, geometry_enabled=True, geometry_layers=1,
                    geometry_register_tokens=2, geometry_anchor_frames=2,
                    geometry_local_window=2)
    model = JWM(cfg)
    sample = render_geometry_sequence(5, frames=3, height=16, width=16)
    batch = {"image": sample.image.unsqueeze(0),
             "depth": sample.depth.unsqueeze(0),
             "pose_c2w": sample.pose_c2w.unsqueeze(0)}
    metrics = geometry_batch_metrics(model, batch)
    assert all(math.isfinite(value) for value in metrics.values())


def _v2_cfg():
    return JWMConfig(d_model=32, n_layers=1, n_heads=4, head_dim=8,
                     ffn_hidden=64, rope_sections=(2, 1, 1), image_size=16,
                     patch=8, geometry_enabled=True,
                     geometry_version="v2_pairwise", geometry_layers=1,
                     geometry_register_tokens=2, geometry_anchor_frames=2,
                     geometry_local_window=2, geometry_motion_radius=1,
                     geometry_min_valid_fraction=0.2)


def _v2_batch(seed):
    sample = render_geometry_sequence(seed, frames=3, height=16, width=16)
    return {"image": sample.image.unsqueeze(0),
            "depth": sample.depth.unsqueeze(0),
            "depth_valid": torch.ones_like(sample.depth,
                                           dtype=torch.bool).unsqueeze(0),
            "pose_c2w": sample.pose_c2w.unsqueeze(0)}


def test_v2_metrics_and_fixed_prior_are_metric_and_finite():
    model = JWM(_v2_cfg())
    batch = _v2_batch(7)
    measured = geometry_batch_metrics(model, batch)
    prior = constant_identity_metrics_v2(batch, model.cfg, constant_depth_m=3.0)
    expected = {"depth_abs_rel", "depth_rmse_metric", "depth_delta1",
                "ate_metric", "abs_rotation_deg", "rpe_translation_metric",
                "rpe_rotation_deg"}
    assert set(measured) == expected
    assert set(prior) == expected
    assert all(math.isfinite(value) for value in measured.values())
    assert all(math.isfinite(value) for value in prior.values())


def test_v2_control_report_contains_all_causal_gates():
    model = JWM(_v2_cfg())
    rows = [_v2_batch(11), _v2_batch(12)]
    report = evaluate_geometry_controls(
        model, rows, torch.device("cpu"), max_batches=2,
        constant_depth_m=4.0)
    assert report["constant_depth_prior_m"] == 4.0
    assert set(report["controls"]) == {
        "normal", "black", "reverse_time", "wrong_window",
        "constant_identity_prior"}
    assert len(report["gates"]) == 6
