"""Pairwise metric geometry for JWM Eye Physical v2.

V1 could infer absolute camera pose from a per-frame camera token, which made
an identity/trajectory prior competitive with real visual evidence.  V2 makes
the causal path explicit:

    (F[t-1], F[t]) -> local correlation -> relative SE(3) -> C2W integration

Depth is split into a relative shape and a learned metric scale.  A dynamic
probability masks moving regions out of ego-motion pooling.  Forward/reverse
pair predictions provide an SE(3) cycle loss, and optional wrong-image outputs
support a bounded counterfactual ranking loss.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .geometric_memory import GeometricContextMemory, temporal_sinusoid
from .layers import RMSNorm, SwiGLU
from .mathx import relative_pose_c2w, rot6d_to_matrix, rotation_geodesic


def pose_from_9d(raw: torch.Tensor) -> torch.Tensor:
    """Identity-centred 6D rotation plus metric translation -> SE(3)."""
    identity6 = raw.new_tensor([1., 0., 0., 0., 1., 0.])
    rotation = rot6d_to_matrix(raw[..., :6] + identity6)
    pose = raw.new_zeros(*raw.shape[:-1], 4, 4)
    pose[..., :3, :3] = rotation
    pose[..., :3, 3] = raw[..., 6:]
    pose[..., 3, 3] = 1.0
    return pose


def masked_depth_pool(depth: torch.Tensor, valid: torch.Tensor | None,
                      grid_h: int, grid_w: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Pool depth without treating missing sensor pixels as zero metres.

    Returns token depth and the valid-pixel fraction of every token.  V1 kept
    only 100%-valid patches; this weighted form retains boundary evidence while
    still down-weighting partially observed regions.
    """
    b, t, h, w = depth.shape
    mask = torch.isfinite(depth) & (depth > 1e-6)
    if valid is not None:
        mask &= valid.bool()
    x = torch.where(mask, depth, torch.zeros_like(depth)).reshape(b * t, 1, h, w)
    m = mask.float().reshape(b * t, 1, h, w)
    numerator = F.adaptive_avg_pool2d(x, (grid_h, grid_w))
    fraction = F.adaptive_avg_pool2d(m, (grid_h, grid_w))
    pooled = numerator / fraction.clamp(min=1e-6)
    return pooled.reshape(b, t, -1), fraction.reshape(b, t, -1)


def weighted_uncertainty_depth_loss(pred: torch.Tensor, log_sigma: torch.Tensor,
                                    target: torch.Tensor, weight: torch.Tensor,
                                    min_valid_fraction: float) -> torch.Tensor:
    mask = (weight >= min_valid_fraction) & torch.isfinite(target) & (target > 1e-6)
    error = (pred.clamp(min=1e-6).log() - target.clamp(min=1e-6).log()).abs()
    ls = log_sigma.clamp(-6.0, 6.0)
    value = torch.exp(-ls) * error + ls
    effective = weight * mask.float()
    return (value * effective).sum() / effective.sum().clamp(min=1.0)


class LocalPairwiseMotion(nn.Module):
    """Dense local cost volume and appearance-difference fusion."""

    def __init__(self, d: int, radius: int, hidden: int):
        super().__init__()
        if radius < 0:
            raise ValueError("motion radius must be non-negative")
        self.d, self.radius = d, radius
        self.correlation_channels = (2 * radius + 1) ** 2
        self.prev_norm = RMSNorm(d)
        self.curr_norm = RMSNorm(d)
        self.in_norm = nn.LayerNorm(3 * d + self.correlation_channels)
        self.proj = nn.Linear(3 * d + self.correlation_channels, d, bias=False)
        self.ff_norm = RMSNorm(d)
        self.ff = SwiGLU(d, hidden)

    def forward(self, previous: torch.Tensor, current: torch.Tensor,
                grid_h: int, grid_w: int) -> torch.Tensor:
        b, n, d = current.shape
        if n != grid_h * grid_w:
            raise ValueError("pair tokens do not match the configured visual grid")
        prev = self.prev_norm(previous)
        curr = self.curr_norm(current)
        prev_grid = prev.transpose(1, 2).reshape(b, d, grid_h, grid_w)
        r = self.radius
        neighbourhood = F.unfold(prev_grid, kernel_size=2 * r + 1, padding=r)
        neighbourhood = neighbourhood.view(b, d, self.correlation_channels, n)
        neighbourhood = neighbourhood.permute(0, 3, 2, 1)  # B,N,K,D
        correlation = (curr.unsqueeze(2) * neighbourhood).sum(-1) / math.sqrt(d)
        fused = torch.cat((curr, prev, curr - prev, correlation), dim=-1)
        motion = self.proj(self.in_norm(fused))
        return motion + self.ff(self.ff_norm(motion))


class PairwiseGeometricContextMemory(GeometricContextMemory):
    """GCM with image-pair-constrained pose and metric depth decoding."""

    version = "v2_pairwise"

    def __init__(self, d: int, heads: int, hidden: int, layers: int,
                 grid_h: int, grid_w: int, register_tokens: int = 4,
                 anchor_frames: int = 2, local_window: int = 8,
                 max_trajectory_frames: int = 512, motion_radius: int = 2,
                 min_valid_fraction: float = 0.20):
        super().__init__(d, heads, hidden, layers, grid_h, grid_w,
                         register_tokens, anchor_frames, local_window,
                         max_trajectory_frames)
        # Remove v1 shortcut heads from both the graph and state dict.
        del self.pose_head
        del self.depth_head
        self.min_valid_fraction = min_valid_fraction
        self.motion = LocalPairwiseMotion(d, motion_radius, hidden)
        self.motion_norm = RMSNorm(d)
        self.dynamic_head = nn.Linear(d, 1)
        self.relative_pose_head = nn.Linear(d, 9)
        self.depth_head_v2 = nn.Linear(d, 2)
        self.metric_scale_head = nn.Sequential(
            RMSNorm(d), nn.Linear(d, max(32, d // 4)), nn.SiLU(),
            nn.Linear(max(32, d // 4), 1))

    def _pairwise(self, visual: torch.Tensor):
        b, t, n, d = visual.shape
        zero = visual.new_zeros(b, n, d)
        forward, reverse = [zero], [zero]
        for i in range(1, t):
            forward.append(self.motion(visual[:, i - 1], visual[:, i],
                                       self.grid_h, self.grid_w))
            reverse.append(self.motion(visual[:, i], visual[:, i - 1],
                                       self.grid_h, self.grid_w))
        return torch.stack(forward, dim=1), torch.stack(reverse, dim=1)

    def _contextualize(self, visual: torch.Tensor, detach_state: bool):
        """V1 anchor/local/trajectory memory, returning tokens before decode."""
        b, t, _, d = visual.shape
        a = min(t, self.anchor_frames)
        state = self.new_state(b)
        specials = [self._specials(b, i, True, visual.device, visual.dtype)
                    for i in range(a)]
        h = torch.stack([torch.cat((specials[i], visual[:, i]), dim=1)
                         for i in range(a)], dim=1)
        p = h.shape[2]
        for layer_index, layer in enumerate(self.layers):
            flat = layer.frame(h.reshape(b * a, p, d)).reshape(b, a, p, d)
            joined = layer.context(flat.reshape(b, a * p, d),
                                   flat.reshape(b, a * p, d))
            h = joined.reshape(b, a, p, d)
            for i in range(a):
                self._commit(state.layers[layer_index], h[:, i], i, detach_state)
        chunks = [h[:, i] for i in range(a)]
        state.frame_index = a
        for frame_index in range(a, t):
            x = torch.cat((self._specials(b, frame_index, False,
                                          visual.device, visual.dtype),
                           visual[:, frame_index] + temporal_sinusoid(
                               frame_index, d, visual.device,
                               visual.dtype).view(1, 1, d)), dim=1)
            for layer_index, layer in enumerate(self.layers):
                x = layer.frame(x)
                context = self._context(state.layers[layer_index])
                kv = x if context is None else torch.cat((context, x), dim=1)
                x = layer.context(x, kv)
                self._commit(state.layers[layer_index], x, frame_index,
                             detach_state)
            chunks.append(x)
            state.frame_index += 1
        return torch.stack(chunks, dim=1), state

    def _relative_poses(self, motion: torch.Tensor,
                        dynamic_probability: torch.Tensor) -> torch.Tensor:
        # t=0 has no pair.  Moving pixels are softly excluded from ego motion.
        static = (1.0 - dynamic_probability[:, 1:]).clamp(min=0.05)
        pooled = ((motion[:, 1:] * static.unsqueeze(-1)).sum(2) /
                  static.sum(2, keepdim=True).clamp(min=1e-4))
        return pose_from_9d(self.relative_pose_head(self.motion_norm(pooled)))

    @staticmethod
    def _integrate(relative: torch.Tensor) -> torch.Tensor:
        b = relative.shape[0]
        current = torch.eye(4, device=relative.device,
                            dtype=relative.dtype).view(1, 4, 4).repeat(b, 1, 1)
        poses = [current]
        for i in range(relative.shape[1]):
            current = current @ relative[:, i]
            poses.append(current)
        return torch.stack(poses, dim=1)

    def forward_sequence(self, visual: torch.Tensor,
                         detach_state: bool = False):
        if visual.ndim != 4 or visual.shape[1] < 1:
            raise ValueError("visual must have shape (B,T,N,D), T>=1")
        forward_motion, reverse_motion = self._pairwise(visual)
        dynamic_logits = self.dynamic_head(forward_motion).squeeze(-1)
        dynamic = torch.sigmoid(dynamic_logits)
        dynamic = torch.cat((torch.zeros_like(dynamic[:, :1]), dynamic[:, 1:]), dim=1)
        fused = visual + forward_motion
        context, state = self._contextualize(fused, detach_state)
        context = self.final_norm(context)
        visual_context = context[..., self.num_special:, :]

        forward_relative = self._relative_poses(forward_motion, dynamic)
        reverse_dynamic = torch.sigmoid(self.dynamic_head(reverse_motion).squeeze(-1))
        reverse_relative = self._relative_poses(reverse_motion, reverse_dynamic)
        pose = self._integrate(forward_relative)

        depth_raw = self.depth_head_v2(visual_context)
        relative_depth = F.softplus(depth_raw[..., 0]) + 1e-4
        log_sigma = depth_raw[..., 1].clamp(-6.0, 6.0)
        log_scale = self.metric_scale_head(visual_context.mean(dim=2)).clamp(-2.3, 3.0)
        metric_scale = log_scale.exp()
        metric_depth = relative_depth * metric_scale
        world_tokens = torch.cat((visual_context,
                                  context[..., :self.num_special, :]), dim=-2)
        return {
            "pose_c2w": pose,
            "rotation": pose[..., :3, :3],
            "translation": pose[..., :3, 3],
            "relative_pose": forward_relative,
            "inverse_relative_pose": reverse_relative,
            "depth_tokens": metric_depth,
            "depth_relative_tokens": relative_depth,
            "depth_metric_scale": metric_scale.squeeze(-1),
            "depth_log_sigma": log_sigma,
            "dynamic_logits": dynamic_logits,
            "dynamic_probability": dynamic,
            "pair_motion_tokens": forward_motion,
            "context_tokens": context[..., :self.num_special, :],
            "world_tokens": world_tokens,
            "memory_state": state,
        }

    def _normalise_gt_pose(self, pose: torch.Tensor) -> torch.Tensor:
        origin_inverse = torch.linalg.inv(pose[:, :1])
        return origin_inverse @ pose

    def _sample_energy(self, output: dict, target_metric: torch.Tensor,
                       valid_weight: torch.Tensor,
                       pose_gt: torch.Tensor) -> torch.Tensor:
        mask = valid_weight >= self.min_valid_fraction
        depth_error = (output["depth_tokens"].clamp(min=1e-5).log() -
                       target_metric.clamp(min=1e-5).log()).abs()
        weight = valid_weight * mask.float()
        dims = tuple(range(1, depth_error.ndim))
        depth_energy = (depth_error * weight).sum(dims) / weight.sum(dims).clamp(min=1.0)
        predicted_rel = output["relative_pose"]
        target_rel = relative_pose_c2w(pose_gt[:, :-1], pose_gt[:, 1:])
        rotation = rotation_geodesic(predicted_rel[..., :3, :3],
                                     target_rel[..., :3, :3]).mean(1)
        translation = (predicted_rel[..., :3, 3] -
                       target_rel[..., :3, 3]).norm(dim=-1).mean(1)
        return depth_energy + rotation + translation

    def loss(self, output: dict[str, torch.Tensor], depth_gt: torch.Tensor,
             pose_gt_c2w: torch.Tensor, depth_valid: torch.Tensor | None = None,
             depth_weight: float = 0.5, metric_depth_weight: float = 1.0,
             abs_pose_weight: float = 0.25, rel_pose_weight: float = 1.0,
             rel_translation_weight: float = 1.0, cycle_weight: float = 0.25,
             dynamic_gt: torch.Tensor | None = None,
             dynamic_weight: float = 0.20,
             negative_output: dict[str, torch.Tensor] | None = None,
             counterfactual_weight: float = 0.25,
             counterfactual_margin: float = 0.20):
        target_metric, valid_weight = masked_depth_pool(
            depth_gt, depth_valid, self.grid_h, self.grid_w)
        anchor_weight = valid_weight[:, :self.anchor_frames]
        anchor_depth = target_metric[:, :self.anchor_frames]
        dims = tuple(range(1, anchor_depth.ndim))
        anchor_scale = ((anchor_depth * anchor_weight).sum(dims) /
                        anchor_weight.sum(dims).clamp(min=1.0)).clamp(min=1e-4).detach()
        target_relative = target_metric / anchor_scale.view(-1, 1, 1)

        relative_depth_loss = weighted_uncertainty_depth_loss(
            output["depth_relative_tokens"], output["depth_log_sigma"],
            target_relative, valid_weight, self.min_valid_fraction)
        metric_depth_loss = weighted_uncertainty_depth_loss(
            output["depth_tokens"], output["depth_log_sigma"],
            target_metric, valid_weight, self.min_valid_fraction)

        gt = self._normalise_gt_pose(pose_gt_c2w)
        absolute_rotation = rotation_geodesic(
            output["rotation"], gt[..., :3, :3]).mean()
        absolute_translation = F.smooth_l1_loss(
            output["translation"], gt[..., :3, 3])
        absolute_pose = absolute_rotation + absolute_translation

        target_relative_pose = relative_pose_c2w(gt[:, :-1], gt[:, 1:])
        predicted_relative = output["relative_pose"]
        relative_rotation = rotation_geodesic(
            predicted_relative[..., :3, :3],
            target_relative_pose[..., :3, :3]).mean()
        relative_translation = F.smooth_l1_loss(
            predicted_relative[..., :3, 3],
            target_relative_pose[..., :3, 3])
        relative_pose_loss = (relative_rotation +
                              rel_translation_weight * relative_translation)

        cycle = output["relative_pose"] @ output["inverse_relative_pose"]
        identity_rotation = torch.eye(3, device=cycle.device,
                                      dtype=cycle.dtype).expand_as(cycle[..., :3, :3])
        cycle_loss = (rotation_geodesic(cycle[..., :3, :3], identity_rotation).mean() +
                      cycle[..., :3, 3].norm(dim=-1).mean())

        dynamic_loss = output["dynamic_probability"].sum() * 0.0
        if dynamic_gt is not None:
            target_dynamic = F.adaptive_avg_pool2d(
                dynamic_gt.float().reshape(-1, 1, *dynamic_gt.shape[-2:]),
                (self.grid_h, self.grid_w)).reshape_as(
                    output["dynamic_probability"])
            # Frame zero has no preceding pair, so its dynamic probability is
            # intentionally fixed to zero and must not receive an impossible
            # positive target. Dynamic labels supervise only real transitions.
            if target_dynamic.shape[1] > 1:
                dynamic_loss = F.binary_cross_entropy_with_logits(
                    output["dynamic_logits"][:, 1:], target_dynamic[:, 1:])

        counterfactual_loss = output["depth_tokens"].sum() * 0.0
        if negative_output is not None:
            correct_energy = self._sample_energy(output, target_metric,
                                                 valid_weight, gt)
            wrong_energy = self._sample_energy(negative_output, target_metric,
                                               valid_weight, gt)
            counterfactual_loss = F.relu(
                counterfactual_margin + correct_energy - wrong_energy).mean()

        total = (depth_weight * relative_depth_loss +
                 metric_depth_weight * metric_depth_loss +
                 abs_pose_weight * absolute_pose +
                 rel_pose_weight * relative_pose_loss +
                 cycle_weight * cycle_loss +
                 dynamic_weight * dynamic_loss +
                 counterfactual_weight * counterfactual_loss)
        metrics = {
            "geometry_depth_relative": float(relative_depth_loss.detach()),
            "geometry_depth_metric": float(metric_depth_loss.detach()),
            "geometry_abs_rotation_rad": float(absolute_rotation.detach()),
            "geometry_abs_translation": float(absolute_translation.detach()),
            "geometry_relative_rotation": float(relative_rotation.detach()),
            "geometry_relative_translation": float(relative_translation.detach()),
            "geometry_cycle": float(cycle_loss.detach()),
            "geometry_dynamic": float(dynamic_loss.detach()),
            "geometry_counterfactual": float(counterfactual_loss.detach()),
            "geometry_static_fraction": float(
                (1.0 - output["dynamic_probability"]).mean().detach()),
            "geometry_metric_scale": float(
                output["depth_metric_scale"].mean().detach()),
        }
        return total, metrics
