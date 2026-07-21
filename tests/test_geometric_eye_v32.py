import torch

from jwm import JWM
from jwm.configs import eye_physical_v32_scale, eye_physical_v32_smoke_scale
from jwm.geometry_v3_data import procedural_v3_row, stack_geometry_v3_rows
from jwm.geometry_v3_trainer import (
    missing_trainable_gradients, set_eye_v3_physical_trainable,
)


def _batch(seed=31):
    return stack_geometry_v3_rows([procedural_v3_row(seed, 3, 32)])


def test_v32_capacity_budget_is_the_declared_380m_class():
    with torch.device("meta"):
        model = JWM(eye_physical_v32_scale())
    total = sum(parameter.numel() for parameter in model.parameters())
    geometry = sum(parameter.numel() for parameter in model.geometry.parameters())
    assert 350_000_000 <= total <= 410_000_000
    assert geometry >= 10_000_000


def test_v32_safe_motion_initialization_survives_global_init():
    model = JWM(eye_physical_v32_smoke_scale())
    assert torch.count_nonzero(model.geometry.pose_head[-1].weight) == 0
    assert torch.count_nonzero(model.geometry.tracker.delta.weight) == 0
    assert torch.count_nonzero(model.geometry.ray_head.weight) == 0


def test_v32_depth_ray_register_track_contract_and_gradients():
    model = JWM(eye_physical_v32_smoke_scale())
    active = set_eye_v3_physical_trainable(model)
    batch = _batch()
    output = model.encode_geometry_sequence(
        batch["image"], intrinsics=batch["intrinsics"],
        projection_y_sign=batch["projection_y_sign"])
    assert output["scene_context"].shape == (1, 3, 48)
    assert output["ray_map_feature"].shape == (1, 3, 8, 8, 3)
    assert output["track_backward_target"].shape == (1, 2, 8, 2)
    loss, metrics = model(
        "geometry", batch["image"], batch["depth"], batch["pose_c2w"],
        batch["depth_valid"], batch["dynamic_mask"], None,
        batch["intrinsics"], batch["projection_y_sign"],
        batch["rigid_flow"], batch["rigid_flow_valid"])
    assert torch.isfinite(loss)
    assert {"geometry_ray_angular", "geometry_track_cycle",
            "geometry_confidence_bce"} <= set(metrics)
    loss.backward()
    assert active
    assert missing_trainable_gradients(model) == []


def test_scene_registers_are_strictly_causal():
    model = JWM(eye_physical_v32_smoke_scale()).eval()
    batch = _batch(33)
    changed = batch["image"].clone()
    changed[:, 2] = torch.rand_like(changed[:, 2])
    with torch.no_grad():
        first = model.encode_geometry_sequence(
            batch["image"], intrinsics=batch["intrinsics"],
            projection_y_sign=batch["projection_y_sign"])["scene_context"]
        second = model.encode_geometry_sequence(
            changed, intrinsics=batch["intrinsics"],
            projection_y_sign=batch["projection_y_sign"])["scene_context"]
    torch.testing.assert_close(first[:, :2], second[:, :2], rtol=1e-5, atol=1e-6)
    assert not torch.allclose(first[:, 2], second[:, 2])
