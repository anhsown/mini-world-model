import math

import torch

from jwm import JWM, JWMConfig
from jwm.geometry_v3_data import procedural_v3_row, stack_geometry_v3_rows
from jwm.geometry_v3_trainer import (
    missing_trainable_gradients, set_eye_v3_physical_trainable,
)


def tiny_v3():
    return JWMConfig(
        d_model=32, n_layers=1, n_heads=4, head_dim=8,
        ffn_hidden=64, rope_sections=(2, 1, 1), image_size=32, patch=8,
        geometry_enabled=True, geometry_version="v3_ctpg",
        geometry_v3_width=32, geometry_track_points=8,
        geometry_track_radius=1, geometry_track_iterations=1,
        geometry_ba_iterations=1)


def test_ctpg_shapes_are_metric_calibrated_and_bounded():
    model = JWM(tiny_v3()).eval()
    batch = stack_geometry_v3_rows([procedural_v3_row(11, 3, 32)])
    with torch.no_grad():
        out = model.encode_geometry_sequence(
            batch["image"], intrinsics=batch["intrinsics"],
            projection_y_sign=batch["projection_y_sign"])
    assert out["depth"].shape == (1, 3, 32, 32)
    assert out["pointmap_camera"].shape == (1, 3, 32, 32, 3)
    assert out["track_target"].shape == (1, 2, 8, 2)
    assert out["relative_pose"].shape == (1, 2, 4, 4)
    assert out["world_tokens"].shape[:2] == (1, 3)
    assert bool((out["depth"] > 0).all())
    points = out["track_source"]
    assert bool((points >= 2).all()) and bool((points <= 5).all())
    quadrants = (points[..., 0] >= 4).long() + 2 * (points[..., 1] >= 4).long()
    assert torch.unique(quadrants).numel() == 4


def test_ctpg_loss_is_finite_and_track_pose_depth_receive_gradient():
    model = JWM(tiny_v3())
    batch = stack_geometry_v3_rows([procedural_v3_row(12, 3, 32)])
    loss, metrics = model(
        "geometry", batch["image"], batch["depth"], batch["pose_c2w"],
        batch["depth_valid"], batch["dynamic_mask"], None,
        batch["intrinsics"], batch["projection_y_sign"],
        batch["rigid_flow"], batch["rigid_flow_valid"])
    assert torch.isfinite(loss)
    assert all(math.isfinite(value) for value in metrics.values())
    loss.backward()
    for parameter in (model.geometry.depth[-1].weight,
                      model.geometry.tracker.context.weight,
                      model.geometry.pose_head[-1].weight):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0


def test_ctpg_refuses_uncalibrated_input():
    model = JWM(tiny_v3())
    images = torch.rand(1, 3, 3, 32, 32)
    try:
        model.encode_geometry_sequence(images)
    except ValueError as error:
        assert "intrinsics" in str(error)
    else:
        raise AssertionError("v3 accepted images without camera intrinsics")


def test_exact_physical_graph_has_no_orphan_trainable_parameters():
    model = JWM(tiny_v3())
    active = set_eye_v3_physical_trainable(model)
    batch = stack_geometry_v3_rows([procedural_v3_row(13, 3, 32)])
    loss, _ = model(
        "geometry", batch["image"], batch["depth"], batch["pose_c2w"],
        batch["depth_valid"], batch["dynamic_mask"], None,
        batch["intrinsics"], batch["projection_y_sign"],
        batch["rigid_flow"], batch["rigid_flow_valid"])
    loss.backward()
    assert active
    assert missing_trainable_gradients(model) == []


def test_no_valid_rigid_tracks_remains_finite_and_backpropagates():
    model = JWM(tiny_v3())
    set_eye_v3_physical_trainable(model)
    batch = stack_geometry_v3_rows([procedural_v3_row(14, 3, 32)])
    batch["rigid_flow_valid"].zero_()
    loss, metrics = model(
        "geometry", batch["image"], batch["depth"], batch["pose_c2w"],
        batch["depth_valid"], batch["dynamic_mask"], None,
        batch["intrinsics"], batch["projection_y_sign"],
        batch["rigid_flow"], batch["rigid_flow_valid"])
    assert torch.isfinite(loss)
    assert metrics["geometry_track_valid_fraction"] == 0.0
    loss.backward()
    assert all(parameter.grad is None or torch.isfinite(parameter.grad).all()
               for parameter in model.parameters())


def test_short_optimizer_stress_has_finite_loss_and_gradients():
    model = JWM(tiny_v3())
    set_eye_v3_physical_trainable(model)
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=1e-4)
    for seed in range(20, 24):
        batch = stack_geometry_v3_rows([procedural_v3_row(seed, 3, 32)])
        optimizer.zero_grad(set_to_none=True)
        loss, _ = model(
            "geometry", batch["image"], batch["depth"], batch["pose_c2w"],
            batch["depth_valid"], batch["dynamic_mask"], None,
            batch["intrinsics"], batch["projection_y_sign"],
            batch["rigid_flow"], batch["rigid_flow_valid"])
        assert torch.isfinite(loss)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        assert torch.isfinite(norm)
        optimizer.step()
