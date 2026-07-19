"""Mathematical, causal and anti-shortcut checks for Eye Physical v2."""

import math

import torch

from jwm import JWM, JWMConfig
from jwm.geometric_memory_v2 import (
    LocalPairwiseMotion,
    masked_depth_pool,
    pose_from_9d,
)
from jwm.geometry_data import render_geometry_sequence


def tiny_v2():
    return JWMConfig(
        d_model=32, n_layers=1, n_heads=4, head_dim=8,
        ffn_hidden=64, rope_sections=(2, 1, 1),
        image_size=32, patch=8,
        vision_stem="local", vision_local_layers=1,
        vision_local_heads=4, vision_window=2,
        geometry_enabled=True, geometry_version="v2_pairwise",
        geometry_layers=1, geometry_register_tokens=2,
        geometry_anchor_frames=2, geometry_local_window=2,
        geometry_motion_radius=1,
        geometry_metric_depth_weight=1.0,
        geometry_cycle_weight=0.2, geometry_dynamic_weight=0.2,
        geometry_counterfactual_weight=0.2)


def test_pose_from_zero_is_identity():
    pose = pose_from_9d(torch.zeros(3, 9))
    assert torch.allclose(pose, torch.eye(4).expand_as(pose), atol=1e-6)


def test_masked_depth_pool_does_not_average_missing_pixels_as_zero():
    depth = torch.tensor([[[[2.0, 0.0], [2.0, 0.0]]]])
    valid = depth > 0
    pooled, fraction = masked_depth_pool(depth, valid, 1, 1)
    assert pooled.item() == 2.0
    assert fraction.item() == 0.5


def test_pairwise_motion_depends_on_frame_order():
    module = LocalPairwiseMotion(d=16, radius=1, hidden=32)
    first = torch.randn(2, 16, 16)
    second = torch.randn(2, 16, 16)
    forward = module(first, second, 4, 4)
    reverse = module(second, first, 4, 4)
    assert forward.shape == reverse.shape == (2, 16, 16)
    assert not torch.allclose(forward, reverse)


def test_v2_shapes_metric_depth_identity_origin_and_so3():
    model = JWM(tiny_v2()).eval()
    images = torch.rand(2, 4, 3, 32, 32)
    with torch.no_grad():
        output = model.encode_geometry_sequence(images)
    assert output["depth_tokens"].shape == (2, 4, 16)
    assert output["relative_pose"].shape == (2, 3, 4, 4)
    assert output["dynamic_probability"].shape == (2, 4, 16)
    assert bool((output["depth_tokens"] > 0).all())
    identity = torch.eye(4).expand_as(output["pose_c2w"][:, 0])
    assert torch.allclose(output["pose_c2w"][:, 0], identity, atol=1e-7)
    rotation = output["rotation"]
    eye3 = torch.eye(3).expand_as(rotation)
    assert torch.allclose(rotation.transpose(-1, -2) @ rotation, eye3, atol=1e-5)
    assert torch.allclose(torch.det(rotation), torch.ones_like(torch.det(rotation)),
                          atol=1e-5)


def test_v2_loss_all_terms_finite_and_pose_head_gets_gradient():
    model = JWM(tiny_v2())
    sample = render_geometry_sequence(19, frames=4, height=32, width=32)
    image = sample.image.unsqueeze(0)
    loss, metrics = model(
        "geometry", image, sample.depth.unsqueeze(0),
        sample.pose_c2w.unsqueeze(0), None,
        sample.dynamic_mask.unsqueeze(0), image.flip(1))
    assert torch.isfinite(loss)
    assert all(math.isfinite(value) for value in metrics.values())
    loss.backward()
    gradient = model.geometry.relative_pose_head.weight.grad
    assert gradient is not None and torch.isfinite(gradient).all()
    assert gradient.abs().sum() > 0


def test_wrong_or_reversed_visual_evidence_changes_v2_output():
    model = JWM(tiny_v2()).eval()
    sample = render_geometry_sequence(23, frames=4, height=32, width=32)
    with torch.no_grad():
        normal = model.encode_geometry_sequence(sample.image.unsqueeze(0))
        reversed_output = model.encode_geometry_sequence(
            sample.image.flip(0).unsqueeze(0))
    assert not torch.allclose(normal["relative_pose"],
                              reversed_output["relative_pose"])
    assert not torch.allclose(normal["depth_tokens"],
                              reversed_output["depth_tokens"])
