"""Mathematical and streaming invariants for JWM Geometric Context Memory."""

from __future__ import annotations

import math

import pytest
import torch

from jwm import JWM, JWMConfig
from jwm import tokenizer as tok
from jwm.geometric_memory import GeometricContextMemory, temporal_sinusoid
from jwm.mathx import (
    anchor_depth_scale,
    relative_pose_c2w,
    rotation_geodesic,
)


torch.manual_seed(7)


def _pose(rotation: torch.Tensor | None = None,
          translation=(0.0, 0.0, 0.0)) -> torch.Tensor:
    out = torch.eye(4)
    if rotation is not None:
        out[:3, :3] = rotation
    out[:3, 3] = torch.tensor(translation)
    return out


def _module(local_window: int = 2, trajectory: int = 16):
    return GeometricContextMemory(
        d=32, heads=4, hidden=64, layers=2,
        grid_h=2, grid_w=3, register_tokens=2,
        anchor_frames=2, local_window=local_window,
        max_trajectory_frames=trajectory,
    )


def test_temporal_sinusoid_shape_and_determinism():
    a = temporal_sinusoid(torch.tensor([0, 3, 9]), 31)
    b = temporal_sinusoid(torch.tensor([0, 3, 9]), 31)
    assert a.shape == (3, 31)
    assert torch.equal(a, b)
    assert not torch.equal(a[0], a[1])


def test_rotation_geodesic_identity_and_quarter_turn():
    eye = torch.eye(3)
    rz = torch.tensor([[0.0, -1.0, 0.0],
                       [1.0,  0.0, 0.0],
                       [0.0,  0.0, 1.0]])
    assert rotation_geodesic(eye, eye).item() == pytest.approx(0.0, abs=1e-7)
    assert rotation_geodesic(eye, rz).item() == pytest.approx(math.pi / 2, rel=1e-6)


def test_relative_pose_is_invariant_to_world_frame_change():
    rz = torch.tensor([[0.0, -1.0, 0.0],
                       [1.0,  0.0, 0.0],
                       [0.0,  0.0, 1.0]])
    pi = _pose(translation=(1.0, 2.0, 3.0))
    pj = _pose(rz, translation=(2.0, 2.0, 3.0))
    world_change = _pose(rz, translation=(10.0, -4.0, 2.0))
    rel = relative_pose_c2w(pi, pj)
    rel_changed = relative_pose_c2w(world_change @ pi, world_change @ pj)
    assert torch.allclose(rel, rel_changed, atol=1e-6)
    assert torch.allclose(rel[:3, 3], torch.tensor([1.0, 0.0, 0.0]))


def test_anchor_scale_uses_only_valid_anchor_depth():
    depth = torch.tensor([[[1.0, 3.0], [5.0, 7.0], [100.0, 100.0]]])
    valid = torch.tensor([[[1, 0], [1, 1], [1, 1]]], dtype=torch.bool)
    scale = anchor_depth_scale(depth, valid, anchor_frames=2)
    assert scale.item() == pytest.approx((1.0 + 5.0 + 7.0) / 3.0)


def test_structured_memory_retains_anchor_local_and_compact_trajectory():
    model = _module(local_window=2)
    visual = torch.randn(1, 7, 6, 32)
    out = model.forward_sequence(visual, detach_state=True)
    state = out["memory_state"]
    assert state.frame_index == 7
    for mem in state.layers:
        assert len(mem.anchors) == 2
        assert len(mem.local) == 2
        assert len(mem.trajectory) == 3
        assert all(x.shape[1] == model.num_special for x in mem.trajectory)


def test_forward_shapes_positive_depth_and_so3():
    model = _module()
    visual = torch.randn(2, 5, 6, 32)
    out = model.forward_sequence(visual)
    assert out["pose_c2w"].shape == (2, 5, 4, 4)
    assert out["depth_tokens"].shape == (2, 5, 6)
    assert out["world_tokens"].shape == (2, 5, 10, 32)
    assert bool((out["depth_tokens"] > 0).all())
    rotation = out["rotation"]
    eye = torch.eye(3).expand_as(rotation)
    assert torch.allclose(rotation.transpose(-1, -2) @ rotation, eye, atol=1e-5)
    assert torch.allclose(torch.det(rotation), torch.ones_like(torch.det(rotation)), atol=1e-5)


def test_stream_is_causal_after_anchor_bootstrap():
    model = _module()
    model.eval()
    visual = torch.randn(1, 5, 6, 32)
    changed = visual.clone()
    changed[:, 4] = torch.randn_like(changed[:, 4]) * 100.0
    with torch.no_grad():
        out_a = model.forward_sequence(visual)
        out_b = model.forward_sequence(changed)
    assert torch.allclose(out_a["world_tokens"][:, :4],
                          out_b["world_tokens"][:, :4], atol=1e-6)
    assert not torch.allclose(out_a["world_tokens"][:, 4],
                              out_b["world_tokens"][:, 4])


def test_geometry_loss_is_finite_and_backpropagates():
    model = _module()
    visual = torch.randn(2, 4, 6, 32, requires_grad=True)
    output = model.forward_sequence(visual, detach_state=False)
    depth = torch.rand(2, 4, 8, 12) * 4.0 + 0.5
    poses = torch.eye(4).view(1, 1, 4, 4).repeat(2, 4, 1, 1)
    poses[:, :, 0, 3] = torch.arange(4).float().view(1, 4) * 0.1
    loss, metrics = model.loss(output, depth, poses)
    assert torch.isfinite(loss)
    assert all(math.isfinite(v) for v in metrics.values())
    loss.backward()
    assert visual.grad is not None and torch.isfinite(visual.grad).all()
    assert visual.grad.abs().sum() > 0


def test_jwm_geometry_integration_and_dynamic_ar_tokens():
    cfg = JWMConfig(
        d_model=32, n_layers=1, n_heads=4, head_dim=8, ffn_hidden=64,
        rope_sections=(2, 1, 1), image_size=16, patch=8,
        max_q_bytes=8, max_a_bytes=8, geometry_enabled=True,
        geometry_layers=1, geometry_register_tokens=2,
        geometry_anchor_frames=2, geometry_local_window=2,
    )
    model = JWM(cfg)
    images = torch.rand(1, 3, 3, 16, 16)
    depth = torch.rand(1, 3, 16, 16) + 0.5
    poses = torch.eye(4).view(1, 1, 4, 4).repeat(1, 3, 1, 1)
    loss, metrics = model("geometry", images, depth, poses)
    assert torch.isfinite(loss) and math.isfinite(metrics["loss"])

    geo = model.encode_geometry_sequence(images)
    world = geo["world_tokens"][:, -1]
    q = torch.tensor([[10, 11]])
    valid = torch.ones_like(q, dtype=torch.bool)
    emb, coords, mask = model._build_ar(
        images[:, -1], q, valid, tok.BOA, img_tokens=world)
    assert emb.shape[:2] == coords.shape[:2] == mask.shape
    assert world.shape[1] == cfg.n_img_tokens + model.geometry.num_special
