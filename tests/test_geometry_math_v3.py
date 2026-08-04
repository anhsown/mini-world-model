import torch

from jwm.geometry_math_v3 import (
    backproject_depth, bundle_adjust_pair, project_points, resize_crop_intrinsics,
    rigid_flow, se3_exp,
)


def test_resize_crop_intrinsics_is_analytic():
    k = torch.tensor([[500., 0., 320.], [0., 510., 240.], [0., 0., 1.]])
    out = resize_crop_intrinsics(k, .5, .25, 7, 11)
    expected = torch.tensor([[250., 0., 167.], [0., 127.5, 71.], [0., 0., 1.]])
    assert torch.allclose(out, expected)


def test_backproject_project_round_trip_for_both_axis_conventions():
    k = torch.tensor([[[80., 0., 7.], [0., 75., 5.], [0., 0., 1.]]])
    depth = torch.linspace(1, 3, 11 * 15).reshape(1, 11, 15)
    for sign in (1.0, -1.0):
        points = backproject_depth(depth, k, sign).reshape(1, -1, 3)
        pixels, recovered = project_points(points, k, sign)
        y, x = torch.meshgrid(torch.arange(11.), torch.arange(15.), indexing="ij")
        expected = torch.stack((x, y), -1).reshape(1, -1, 2)
        assert torch.allclose(pixels, expected, atol=1e-5)
        assert torch.allclose(recovered, depth.flatten(1), atol=1e-6)


def test_identity_rigid_flow_is_zero():
    depth = torch.ones(1, 8, 9) * 2
    k = torch.tensor([[[20., 0., 4.], [0., 20., 3.5], [0., 0., 1.]]])
    eye = torch.eye(4).unsqueeze(0)
    flow, valid = rigid_flow(depth, k, k, eye)
    assert torch.allclose(flow, torch.zeros_like(flow), atol=1e-5)
    assert valid.float().mean() > .99


def test_bundle_adjustment_reduces_reprojection_and_pose_error():
    torch.manual_seed(5)
    b, n = 1, 80
    points = torch.randn(b, n, 3)
    points[..., 2] = points[..., 2].abs() + 3.0
    k = torch.tensor([[[120., 0., 64.], [0., 118., 48.], [0., 0., 1.]]])
    truth = se3_exp(torch.tensor([[.08, -.03, .02, .01, -.02, .015]]))
    target, _ = project_points(
        (truth[:, None, :3, :3] @ points[..., None]).squeeze(-1) + truth[:, None, :3, 3], k)
    initial = se3_exp(torch.tensor([[.02, .01, -.01, -.01, .01, 0.]]))
    refined, history, _ = bundle_adjust_pair(points, target, k, initial,
                                              iterations=5, damping=1e-4)
    before = torch.linalg.vector_norm(initial[:, :3, 3] - truth[:, :3, 3], dim=-1)
    after = torch.linalg.vector_norm(refined[:, :3, 3] - truth[:, :3, 3], dim=-1)
    assert history[0, -1] < history[0, 0] * .05
    assert after.item() < before.item() * .2


def test_degenerate_bundle_adjustment_is_finite_monotonic_and_differentiable():
    points = torch.tensor([[[0.0, 0.0, 2.0]] * 16], requires_grad=True)
    k = torch.tensor([[[90., 0., 32.], [0., 90., 24.], [0., 0., 1.]]])
    twist = torch.tensor([[.02, 0., 0., 0., .01, 0.]], requires_grad=True)
    initial = se3_exp(twist)
    target = torch.full((1, 16, 2), 500.0)
    refined, history, prediction = bundle_adjust_pair(
        points, target, k, initial, iterations=3, damping=1e-2)
    assert torch.isfinite(refined).all()
    assert torch.isfinite(history).all()
    assert torch.isfinite(prediction).all()
    assert bool((history[..., 1:] <= history[..., :-1] + 1e-5).all())
    (refined.square().mean() + history.mean()).backward()
    assert points.grad is not None and torch.isfinite(points.grad).all()
    assert twist.grad is not None and torch.isfinite(twist.grad).all()
