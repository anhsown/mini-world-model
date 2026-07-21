"""Camera-calibrated geometry primitives for CTPG-Eye v3.

All transforms are explicit. ``pose_c2w`` maps camera coordinates to world
coordinates, while ``camera_transform(src, dst)`` maps source-camera points
into the destination camera. Pixel coordinates use centres at integer indices.
``y_sign`` is +1 for OpenCV (image y and camera y both point down) and -1 for
the analytic JWM renderer (camera y points up).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def resize_flow_with_valid(flow: torch.Tensor, valid: torch.Tensor,
                           size: tuple[int, int]) -> tuple[torch.Tensor, torch.Tensor]:
    """Resize flow without bleeding invalid sentinels into valid pixels.

    Args:
        flow: ``(B,P,H,W,2)`` pixel displacement.
        valid: ``(B,P,H,W)`` validity/visibility mask.
        size: target ``(height,width)``.
    Returns:
        Rescaled flow ``(B,P,2,h,w)`` and conservative valid mask
        ``(B,P,h,w)``.
    """
    if flow.shape[:-1] != valid.shape or flow.shape[-1] != 2:
        raise ValueError("flow/valid shapes must be (B,P,H,W,2)/(B,P,H,W)")
    b, pairs, height, width = valid.shape
    out_h, out_w = size
    weight = valid.to(flow.dtype).reshape(b * pairs, 1, height, width)
    channels = torch.nan_to_num(flow, nan=0.0, posinf=0.0, neginf=0.0)
    channels = channels.permute(0, 1, 4, 2, 3).reshape(b * pairs, 2, height, width)
    numerator = F.interpolate(channels * weight, size, mode="bilinear",
                              align_corners=False)
    denominator = F.interpolate(weight, size, mode="bilinear",
                                align_corners=False)
    resized = numerator / denominator.clamp_min(1e-6)
    resized[:, 0] *= out_w / width
    resized[:, 1] *= out_h / height
    # Require most of the interpolation footprint to be valid. This prevents
    # supervision directly on depth/occlusion boundaries.
    resized_valid = (denominator[:, 0] >= .75).reshape(b, pairs, out_h, out_w)
    return resized.reshape(b, pairs, 2, out_h, out_w), resized_valid


def ensure_frame_intrinsics(intrinsics: torch.Tensor, frames: int) -> torch.Tensor:
    """Normalize ``(3,3)`` or ``(T,3,3)`` intrinsics to ``(T,3,3)``."""
    if intrinsics.shape == (3, 3):
        return intrinsics.unsqueeze(0).expand(frames, -1, -1).clone()
    if intrinsics.shape == (frames, 3, 3):
        return intrinsics
    raise ValueError(f"intrinsics must be (3,3) or ({frames},3,3), got {tuple(intrinsics.shape)}")


def resize_crop_intrinsics(intrinsics: torch.Tensor, scale_x: float,
                           scale_y: float, left: float = 0.0,
                           top: float = 0.0) -> torch.Tensor:
    """Apply resize then placement/crop offsets to a pinhole matrix."""
    k = intrinsics.clone()
    k[..., 0, 0] *= scale_x
    k[..., 1, 1] *= scale_y
    k[..., 0, 2] = k[..., 0, 2] * scale_x + left
    k[..., 1, 2] = k[..., 1, 2] * scale_y + top
    return k


def pixel_grid(height: int, width: int, *, device=None, dtype=None) -> torch.Tensor:
    y, x = torch.meshgrid(torch.arange(height, device=device, dtype=dtype),
                          torch.arange(width, device=device, dtype=dtype),
                          indexing="ij")
    return torch.stack((x, y), dim=-1)


def camera_rays(intrinsics: torch.Tensor, height: int, width: int,
                y_sign: torch.Tensor | float = 1.0,
                normalize: bool = False) -> torch.Tensor:
    """Return rays with shape ``(...,H,W,3)`` for batched intrinsics."""
    k = intrinsics
    grid = pixel_grid(height, width, device=k.device, dtype=k.dtype)
    lead = k.shape[:-2]
    grid = grid.view(*([1] * len(lead)), height, width, 2)
    fx = k[..., 0, 0][..., None, None]
    fy = k[..., 1, 1][..., None, None]
    cx = k[..., 0, 2][..., None, None]
    cy = k[..., 1, 2][..., None, None]
    sign = torch.as_tensor(y_sign, device=k.device, dtype=k.dtype)
    while sign.ndim < len(lead):
        sign = sign.unsqueeze(-1)
    sign = sign[..., None, None]
    rays = torch.stack(((grid[..., 0] - cx) / fx,
                        sign * (grid[..., 1] - cy) / fy,
                        torch.ones_like((grid[..., 0] - cx) / fx)), dim=-1)
    if normalize:
        rays = rays / torch.linalg.vector_norm(rays, dim=-1, keepdim=True).clamp_min(1e-8)
    return rays


def backproject_depth(depth: torch.Tensor, intrinsics: torch.Tensor,
                      y_sign: torch.Tensor | float = 1.0) -> torch.Tensor:
    """Backproject z-depth ``(...,H,W)`` to camera points ``(...,H,W,3)``."""
    h, w = depth.shape[-2:]
    return camera_rays(intrinsics, h, w, y_sign, normalize=False) * depth[..., None]


def project_points(points: torch.Tensor, intrinsics: torch.Tensor,
                   y_sign: torch.Tensor | float = 1.0) -> tuple[torch.Tensor, torch.Tensor]:
    """Project camera points ``(...,N,3)`` into pixels and return z-depth."""
    z = points[..., 2].clamp_min(1e-8)
    k = intrinsics
    fx, fy = k[..., 0, 0][..., None], k[..., 1, 1][..., None]
    cx, cy = k[..., 0, 2][..., None], k[..., 1, 2][..., None]
    sign = torch.as_tensor(y_sign, device=points.device, dtype=points.dtype)
    while sign.ndim < points.ndim - 2:
        sign = sign.unsqueeze(-1)
    sign = sign[..., None]
    u = fx * points[..., 0] / z + cx
    v = sign * fy * points[..., 1] / z + cy
    return torch.stack((u, v), dim=-1), points[..., 2]


def transform_points(transform: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    return (transform[..., None, :3, :3] @ points[..., None]).squeeze(-1) + \
        transform[..., None, :3, 3]


def camera_transform(pose_c2w_src: torch.Tensor,
                     pose_c2w_dst: torch.Tensor) -> torch.Tensor:
    """Transform source-camera coordinates into destination-camera coordinates."""
    return torch.linalg.inv(pose_c2w_dst) @ pose_c2w_src


def skew(vector: torch.Tensor) -> torch.Tensor:
    x, y, z = vector.unbind(-1)
    zero = torch.zeros_like(x)
    return torch.stack((zero, -z, y, z, zero, -x, -y, x, zero), dim=-1).reshape(
        *vector.shape[:-1], 3, 3)


def _taylor_coefficients(theta: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    theta2 = theta.square()
    small = theta.abs() < 1e-4
    a = torch.where(small, 1 - theta2 / 6 + theta2.square() / 120,
                    torch.sin(theta) / theta.clamp_min(1e-8))
    b = torch.where(small, 0.5 - theta2 / 24 + theta2.square() / 720,
                    (1 - torch.cos(theta)) / theta2.clamp_min(1e-8))
    c = torch.where(small, 1 / 6 - theta2 / 120 + theta2.square() / 5040,
                    (theta - torch.sin(theta)) /
                    (theta2 * theta).clamp_min(1e-8))
    return a, b, c


def se3_exp(twist: torch.Tensor) -> torch.Tensor:
    """SE(3) exponential for twist order ``[tx,ty,tz,rx,ry,rz]``."""
    velocity, omega = twist[..., :3], twist[..., 3:]
    theta = torch.linalg.vector_norm(omega, dim=-1, keepdim=True)
    a, b, c = _taylor_coefficients(theta)
    w = skew(omega)
    eye = torch.eye(3, device=twist.device, dtype=twist.dtype).expand(*twist.shape[:-1], 3, 3)
    rotation = eye + a[..., None] * w + b[..., None] * (w @ w)
    v_matrix = eye + b[..., None] * w + c[..., None] * (w @ w)
    translation = (v_matrix @ velocity[..., None]).squeeze(-1)
    out = torch.zeros(*twist.shape[:-1], 4, 4, device=twist.device, dtype=twist.dtype)
    out[..., :3, :3] = rotation
    out[..., :3, 3] = translation
    out[..., 3, 3] = 1
    return out


def rigid_flow(depth: torch.Tensor, intrinsics_src: torch.Tensor,
               intrinsics_dst: torch.Tensor, transform_dst_src: torch.Tensor,
               y_sign: torch.Tensor | float = 1.0) -> tuple[torch.Tensor, torch.Tensor]:
    """Dense source->destination flow and positive-depth visibility mask."""
    h, w = depth.shape[-2:]
    points = backproject_depth(depth, intrinsics_src, y_sign)
    flat = points.reshape(*points.shape[:-3], h * w, 3)
    moved = transform_points(transform_dst_src, flat)
    target, z = project_points(moved, intrinsics_dst, y_sign)
    source = pixel_grid(h, w, device=depth.device, dtype=depth.dtype).reshape(h * w, 2)
    source = source.view(*([1] * (target.ndim - 2)), h * w, 2)
    flow = (target - source).reshape(*depth.shape, 2)
    valid = ((z > 1e-5) & (target[..., 0] >= 0) & (target[..., 0] <= w - 1) &
             (target[..., 1] >= 0) & (target[..., 1] <= h - 1)).reshape(depth.shape)
    return flow, valid


def bilinear_sample(feature: torch.Tensor, coordinates: torch.Tensor) -> torch.Tensor:
    """Sample ``B,C,H,W`` at pixel coordinates ``B,N,2`` -> ``B,N,C``."""
    _, _, h, w = feature.shape
    x = 2 * coordinates[..., 0] / max(w - 1, 1) - 1
    y = 2 * coordinates[..., 1] / max(h - 1, 1) - 1
    grid = torch.stack((x, y), dim=-1).unsqueeze(2)
    return torch.nn.functional.grid_sample(feature, grid, align_corners=True,
                                            mode="bilinear", padding_mode="zeros") \
        .squeeze(-1).transpose(1, 2)


def bundle_adjust_pair(points_src: torch.Tensor, target_pixels: torch.Tensor,
                       intrinsics_dst: torch.Tensor, initial: torch.Tensor,
                       weights: torch.Tensor | None = None,
                       y_sign: torch.Tensor | float = 1.0, iterations: int = 3,
                       damping: float = 1e-3,
                       huber_delta: float = 3.0) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Unrolled robust Gauss-Newton pose-only bundle adjustment.

    Args use ``(B,N,*)`` and the returned transform maps source to destination.
    Gradients flow through the incoming pose and direct residual paths; the
    ill-conditioned solver update uses a truncated gradient for stability.
    """
    # The system is tiny (6x6), so force the complete unrolled solve to FP32.
    # This nested context also defeats an enclosing CUDA autocast region.
    output_dtype = points_src.dtype
    device_type = points_src.device.type
    with torch.autocast(device_type=device_type, enabled=False):
        points_src = points_src.float()
        target_pixels = target_pixels.float()
        intrinsics_dst = intrinsics_dst.float()
        transform = initial.float()
        sign_value = torch.as_tensor(y_sign, device=points_src.device).float()
        base_weights = (torch.ones(points_src.shape[:2], device=points_src.device)
                        if weights is None else weights.float())
        base_weights = torch.nan_to_num(base_weights, nan=0.0).clamp(0, 1)
        residual_history = []
        eye6 = torch.eye(6, device=points_src.device)[None]

        def objective(candidate: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor,
                                                         torch.Tensor, torch.Tensor]:
            moved = transform_points(candidate, points_src)
            pixels, depth = project_points(moved, intrinsics_dst, sign_value)
            residual = torch.nan_to_num(pixels - target_pixels, nan=0.0,
                                        posinf=1e3, neginf=-1e3).clamp(-1e3, 1e3)
            magnitude = torch.linalg.vector_norm(residual, dim=-1).clamp_min(1e-6)
            robust = torch.where(magnitude <= huber_delta, torch.ones_like(magnitude),
                                 huber_delta / magnitude)
            finite_target = torch.isfinite(target_pixels).all(-1)
            visible = ((depth > 0.1) & finite_target).to(moved.dtype)
            weight = base_weights * robust * visible
            cost = (weight * residual.square().sum(-1)).sum(-1) / weight.sum(-1).clamp_min(1)
            return cost, moved, residual, weight

        for _ in range(iterations):
            current_cost, points, residual, weight = objective(transform)
            residual_history.append(current_cost)
            x, y, z_safe = points.unbind(-1)
            z_safe = z_safe.clamp_min(0.1)
            fx = intrinsics_dst[:, 0, 0][:, None]
            fy = intrinsics_dst[:, 1, 1][:, None]
            sign = sign_value
            if sign.ndim == 0:
                sign = sign.expand(points.shape[0])
            sign = sign.reshape(points.shape[0], -1)[:, :1]
            zeros = torch.zeros_like(x)
            j_proj = torch.stack((fx / z_safe, zeros, -fx * x / z_safe.square(),
                                  zeros, sign * fy / z_safe,
                                  -sign * fy * y / z_safe.square()), dim=-1)
            j_proj = j_proj.reshape(points.shape[0], points.shape[1], 2, 3)
            dpoint = torch.cat((torch.eye(3, device=points.device)
                                .view(1, 1, 3, 3)
                                .expand(points.shape[0], points.shape[1], -1, -1),
                                -skew(points)), dim=-1)
            jacobian = torch.nan_to_num(j_proj @ dpoint, nan=0.0,
                                        posinf=1e4, neginf=-1e4).clamp(-1e4, 1e4)
            root_weight = weight[..., None, None].sqrt()
            weighted_j = jacobian * root_weight
            weighted_r = residual * weight[..., None].sqrt()
            hessian = torch.einsum("bnij,bnik->bjk", weighted_j, weighted_j)
            gradient = torch.einsum("bnij,bni->bj", weighted_j, weighted_r)
            hessian = torch.nan_to_num(hessian, nan=0.0, posinf=1e6, neginf=-1e6)
            gradient = torch.nan_to_num(gradient, nan=0.0, posinf=1e4, neginf=-1e4)
            diagonal_scale = hessian.diagonal(dim1=-2, dim2=-1).abs().mean(-1)
            lm = (float(damping) * diagonal_scale.clamp(1e-3, 1e4)).view(-1, 1, 1)
            rhs = -gradient.unsqueeze(-1)
            system = hessian + lm * eye6
            solution, info = torch.linalg.solve_ex(system, rhs)
            retry, retry_info = torch.linalg.solve_ex(hessian + (100 * lm + 1e-2) * eye6,
                                                       rhs)
            solution = torch.where((info == 0).view(-1, 1, 1), solution, retry)
            solved = ((info == 0) | (retry_info == 0)).view(-1, 1)
            delta = torch.where(solved, solution.squeeze(-1),
                                torch.zeros_like(solution.squeeze(-1)))
            delta = torch.nan_to_num(delta, nan=0.0, posinf=0.0, neginf=0.0)
            translation, rotation = delta[:, :3], delta[:, 3:]
            translation = translation * (
                0.10 / translation.norm(dim=-1, keepdim=True).clamp_min(0.10)
            ).clamp(max=1)
            rotation = rotation * (
                0.087 / rotation.norm(dim=-1, keepdim=True).clamp_min(0.087)
            ).clamp(max=1)
            # Truncated differentiable BA: the accepted transform remains
            # differentiable with respect to the incoming pose, while the
            # ill-conditioned derivative of the linear solver itself is
            # stopped.  Depth/tracks retain their direct supervised losses.
            delta = torch.cat((translation, rotation), dim=-1).detach()

            # Trust-region rollback: accept an update only when it improves
            # the robust objective; otherwise try a quarter step, then keep
            # the previous transform.  This makes BA monotonic by contract.
            full = se3_exp(delta) @ transform
            full_cost, _, _, _ = objective(full)
            small = se3_exp(0.25 * delta) @ transform
            small_cost, _, _, _ = objective(small)
            accept_full = torch.isfinite(full_cost) & (full_cost <= current_cost)
            best = torch.where(accept_full[:, None, None], full, transform)
            best_cost = torch.where(accept_full, full_cost, current_cost)
            accept_small = torch.isfinite(small_cost) & (small_cost < best_cost)
            transform = torch.where(accept_small[:, None, None], small, best)

        final_cost, points, residual, weight = objective(transform)
        residual_history.append(final_cost)
        prediction, _ = project_points(points, intrinsics_dst, sign_value)
        prediction = torch.nan_to_num(prediction, nan=0.0, posinf=1e3, neginf=-1e3)
    return (transform.to(output_dtype),
            torch.stack(residual_history, dim=-1).to(output_dtype),
            prediction.to(output_dtype))
