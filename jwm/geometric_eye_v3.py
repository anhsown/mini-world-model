"""CTPG-Eye v3: calibrated tracks, point geometry and unrolled BA.

This module is deliberately independent from the language tower. It consumes
RGB windows and per-frame camera calibration, predicts metric depth and sparse
tracks, separates dynamic evidence, estimates relative SE(3), then refines it
with safeguarded pose-only bundle adjustment and a stable truncated gradient.
Compact world tokens are the
only interface required by the rest of JWM.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .geometry_math_v3 import (
    backproject_depth, bilinear_sample, bundle_adjust_pair, camera_rays, camera_transform,
    project_points, resize_flow_with_valid, se3_exp, transform_points,
)
from .mathx import rotation_geodesic


def _group_norm(channels: int) -> nn.GroupNorm:
    return nn.GroupNorm(min(8, channels), channels)


class ResidualConv(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            _group_norm(channels), nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            _group_norm(channels))

    def forward(self, x):
        return F.silu(x + self.net(x))


class CalibratedPyramid(nn.Module):
    """1/4, 1/8 and 1/16 features with ray conditioning at every frame."""

    def __init__(self, width: int = 96):
        super().__init__()
        c4, c8, c16 = width, width + 32, width + 64
        self.stem = nn.Sequential(
            nn.Conv2d(3, width // 2, 5, stride=2, padding=2, bias=False),
            _group_norm(width // 2), nn.SiLU(),
            nn.Conv2d(width // 2, c4, 3, stride=2, padding=1, bias=False),
            _group_norm(c4), nn.SiLU(), ResidualConv(c4))
        self.ray = nn.Sequential(nn.Conv2d(3, c4, 1), nn.SiLU(),
                                 nn.Conv2d(c4, c4, 3, padding=1, groups=8))
        self.down8 = nn.Sequential(nn.Conv2d(c4, c8, 3, stride=2, padding=1),
                                   _group_norm(c8), nn.SiLU(), ResidualConv(c8))
        self.down16 = nn.Sequential(nn.Conv2d(c8, c16, 3, stride=2, padding=1),
                                    _group_norm(c16), nn.SiLU(), ResidualConv(c16))
        self.channels = (c4, c8, c16)

    def forward(self, image: torch.Tensor, intrinsics: torch.Tensor,
                y_sign: torch.Tensor) -> tuple[torch.Tensor, ...]:
        f4 = self.stem(image)
        h4, w4 = f4.shape[-2:]
        k4 = intrinsics.clone()
        k4[:, 0] /= image.shape[-1] / w4
        k4[:, 1] /= image.shape[-2] / h4
        rays = camera_rays(k4, h4, w4, y_sign).permute(0, 3, 1, 2)
        f4 = f4 + self.ray(rays)
        f8 = self.down8(f4)
        return f4, f8, self.down16(f8)


class SparseTrackUpdater(nn.Module):
    """RAFT-style recurrent local correlation on a sparse adaptive point set."""

    def __init__(self, channels: int, hidden: int = 128, radius: int = 2,
                 iterations: int = 3):
        super().__init__()
        self.radius, self.iterations = radius, iterations
        offsets = torch.stack(torch.meshgrid(
            torch.arange(-radius, radius + 1), torch.arange(-radius, radius + 1),
            indexing="ij"), dim=-1)[..., [1, 0]].reshape(-1, 2).float()
        self.register_buffer("offsets", offsets, persistent=False)
        self.context = nn.Linear(channels, hidden)
        self.gru = nn.GRUCell(len(offsets) + 2, hidden)
        self.delta = nn.Linear(hidden, 2)
        self.confidence = nn.Linear(hidden, 1)
        self.visibility = nn.Linear(hidden, 1)
        self.log_scale = nn.Linear(hidden, 1)
        self.dynamic = nn.Linear(hidden, 1)
        self.residual_3d_flow = nn.Linear(hidden, 3)
        nn.init.zeros_(self.delta.weight); nn.init.zeros_(self.delta.bias)

    def forward(self, source: torch.Tensor, target: torch.Tensor,
                points: torch.Tensor) -> dict:
        b, channels, _, _ = source.shape
        n = points.shape[1]
        source_descriptor = bilinear_sample(source, points)
        hidden = torch.tanh(self.context(source_descriptor)).reshape(b * n, -1)
        current = points.clone()
        for _ in range(self.iterations):
            candidates = current[:, :, None, :] + self.offsets[None, None]
            sampled = bilinear_sample(target, candidates.reshape(b, -1, 2))
            sampled = sampled.reshape(b, n, len(self.offsets), channels)
            correlation = (sampled * source_descriptor[:, :, None]).sum(-1) / math.sqrt(channels)
            displacement = current - points
            update = torch.cat((correlation, displacement), dim=-1).reshape(b * n, -1)
            hidden = self.gru(update, hidden)
            current = current + self.delta(hidden).reshape(b, n, 2).tanh()
            # Keep tracks in the image domain.  Sampling padded zeros outside
            # the frame creates unconstrained correlations and can poison BA.
            current = torch.stack((current[..., 0].clamp(0, source.shape[-1] - 1),
                                   current[..., 1].clamp(0, source.shape[-2] - 1)), dim=-1)
        hidden_2d = hidden.reshape(b, n, -1)
        dynamic_logit = self.dynamic(hidden_2d).squeeze(-1)
        visibility_logit = self.visibility(hidden_2d).squeeze(-1)
        confidence_logit = self.confidence(hidden_2d).squeeze(-1)
        return {"target": current, "hidden": hidden_2d,
                "confidence_logit": confidence_logit,
                "confidence": torch.sigmoid(confidence_logit),
                "visibility_logit": visibility_logit,
                "visibility_probability": torch.sigmoid(visibility_logit),
                "log_scale": self.log_scale(hidden_2d).squeeze(-1).clamp(-3.0, 4.0),
                "dynamic_logit": dynamic_logit,
                "dynamic_probability": torch.sigmoid(dynamic_logit),
                "scene_flow_residual": self.residual_3d_flow(hidden_2d)}


class BoundedWorldMemory:
    """Inference-only detached ring buffer; memory never grows with runtime."""

    def __init__(self, max_frames: int = 32):
        self.max_frames = max_frames
        self.tokens: list[torch.Tensor] = []
        self.poses: list[torch.Tensor] = []

    def append(self, tokens: torch.Tensor, pose: torch.Tensor) -> None:
        self.tokens.append(tokens.detach())
        self.poses.append(pose.detach())
        del self.tokens[:-self.max_frames]
        del self.poses[:-self.max_frames]

    def clear(self) -> None:
        self.tokens.clear(); self.poses.clear()


class CausalSceneRegisterMixer(nn.Module):
    """Compact causal temporal state instead of all-to-all video attention.

    Each frame contributes one pooled visual token plus a small bank of scene
    registers.  A token may attend to its own and earlier frames only.  This
    keeps sequence cost O((T(R+1))^2) with R << spatial patches and gives pose
    and metric scale a persistent, explicitly causal scene representation.
    """

    def __init__(self, input_channels: int, width: int, registers: int,
                 layers: int, heads: int, max_frames: int):
        super().__init__()
        if width % heads:
            raise ValueError("scene width must be divisible by scene heads")
        self.registers = registers
        self.input_projection = nn.Linear(input_channels, width)
        self.scene_registers = nn.Parameter(torch.randn(registers, width) * 0.02)
        self.frame_embedding = nn.Parameter(torch.randn(max_frames, width) * 0.01)
        layer = nn.TransformerEncoderLayer(
            width, heads, dim_feedforward=4 * width, activation="gelu",
            batch_first=True, norm_first=True, dropout=0.0)
        self.layers = nn.TransformerEncoder(layer, layers)
        self.norm = nn.LayerNorm(width)

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        # feature: (B,T,C,H,W) -> one causal context vector per frame.
        b, t, channels, _, _ = feature.shape
        if t > self.frame_embedding.shape[0]:
            raise ValueError(f"{t} frames exceed scene memory {self.frame_embedding.shape[0]}")
        pooled = self.input_projection(feature.mean(dim=(-1, -2)))
        frame_pos = self.frame_embedding[:t].view(1, t, 1, -1)
        registers = self.scene_registers.view(1, 1, self.registers, -1).expand(b, t, -1, -1)
        tokens = torch.cat((pooled.unsqueeze(2), registers), dim=2) + frame_pos
        per_frame = self.registers + 1
        flat = tokens.reshape(b, t * per_frame, -1)
        frame_id = torch.arange(t, device=feature.device).repeat_interleave(per_frame)
        # PyTorch bool encoder mask: True means blocked.
        blocked = frame_id[None, :] > frame_id[:, None]
        mixed = self.layers(flat, mask=blocked)
        mixed = self.norm(mixed.reshape(b, t, per_frame, -1))
        return mixed[:, :, 1:].mean(dim=2)


class CTPGPhysicalEye(nn.Module):
    def __init__(self, world_dim: int, width: int = 96, track_points: int = 64,
                 track_radius: int = 2, track_iterations: int = 3,
                 ba_iterations: int = 2, memory_frames: int = 32,
                 scene_registers: int = 0, scene_layers: int = 0,
                 scene_width: int = 0, scene_heads: int = 8,
                 pose_context: int = 0, ray_residual: float = 0.0):
        super().__init__()
        self.track_points = track_points
        self.ba_iterations = ba_iterations
        self.ray_residual_scale = float(ray_residual)
        self.pyramid = CalibratedPyramid(width)
        c4, _, c16 = self.pyramid.channels
        self.depth = nn.Sequential(ResidualConv(c4), nn.Conv2d(c4, 2, 1))
        self.scene_mixer = None
        self.depth_film = None
        self.pose_context = None
        context_size = 0
        if scene_registers > 0 and scene_layers > 0:
            scene_width = scene_width or world_dim
            self.scene_mixer = CausalSceneRegisterMixer(
                c16, scene_width, scene_registers, scene_layers, scene_heads,
                memory_frames)
            self.depth_film = nn.Linear(scene_width, 2 * c4)
            context_size = pose_context or max(32, scene_width // 4)
            self.pose_context = nn.Linear(2 * scene_width, context_size)
        self.ray_head = (nn.Conv2d(c4, 2, 1)
                         if self.ray_residual_scale > 0 else None)
        self.tracker = SparseTrackUpdater(c4, hidden=128, radius=track_radius,
                                          iterations=track_iterations)
        self.pose_head = nn.Sequential(nn.Linear(128 + 7 + context_size, 192), nn.SiLU(),
                                       nn.Linear(192, 128), nn.SiLU(),
                                       nn.Linear(128, 6))
        self.world_projection = nn.Linear(c16, world_dim)
        self.track_projection = nn.Linear(128, world_dim)
        temporal_dim = scene_width if self.scene_mixer is not None else c16
        self.temporal_compatibility = nn.Sequential(
            nn.Linear(4 * temporal_dim, max(64, temporal_dim // 2)), nn.SiLU(),
            nn.Linear(max(64, temporal_dim // 2), 1))
        self.memory = BoundedWorldMemory(memory_frames)
        self.reset_safe_initialization()

    def reset_safe_initialization(self) -> None:
        """Restore identity-motion priors after JWM's global initializer.

        JWM recursively initializes every Linear after submodules are built.
        Without this explicit post-init hook the intended zero-delta pose and
        track priors are silently overwritten, causing unstable first updates.
        """
        nn.init.zeros_(self.pose_head[-1].weight)
        nn.init.zeros_(self.pose_head[-1].bias)
        nn.init.zeros_(self.tracker.delta.weight)
        nn.init.zeros_(self.tracker.delta.bias)
        nn.init.zeros_(self.tracker.residual_3d_flow.weight)
        nn.init.zeros_(self.tracker.residual_3d_flow.bias)
        if self.ray_head is not None:
            nn.init.zeros_(self.ray_head.weight); nn.init.zeros_(self.ray_head.bias)
        if self.depth_film is not None:
            nn.init.zeros_(self.depth_film.weight); nn.init.zeros_(self.depth_film.bias)

    @staticmethod
    def _scale_intrinsics(k: torch.Tensor, input_hw: tuple[int, int],
                          output_hw: tuple[int, int]) -> torch.Tensor:
        ih, iw = input_hw; oh, ow = output_hw
        out = k.clone()
        out[..., 0, :] *= ow / iw
        out[..., 1, :] *= oh / ih
        return out

    @staticmethod
    def _trackability_score(feature: torch.Tensor) -> torch.Tensor:
        """Parameter-free local structure score for sparse point selection.

        Hard top-k has no gradient with respect to a learned saliency head.
        Local feature variance is deterministic and image-dependent, avoiding
        an orphan trainable branch in DistributedDataParallel.
        """
        mean = F.avg_pool2d(feature, 3, stride=1, padding=1)
        second = F.avg_pool2d(feature.square(), 3, stride=1, padding=1)
        variance = (second - mean.square()).clamp_min(0).mean(1, keepdim=True)
        dx = F.pad((feature[..., 1:] - feature[..., :-1]).square().mean(1, keepdim=True),
                   (0, 1, 0, 0))
        dy = F.pad((feature[..., 1:, :] - feature[..., :-1, :]).square().mean(1, keepdim=True),
                   (0, 0, 0, 1))
        return variance + 0.5 * (dx + dy)

    def _select_points(self, feature: torch.Tensor) -> torch.Tensor:
        score = self._trackability_score(feature)
        b, _, h, w = score.shape
        count = min(self.track_points, h * w)
        # One maximum per spatial cell prevents all points collapsing onto a
        # high-contrast border.  The margin keeps the correlation window and
        # rigid-flow supervision inside the image.
        margin = min(max(self.tracker.radius + 1, 1), max((min(h, w) - 1) // 2, 0))
        while margin > 0 and (h - 2 * margin) * (w - 2 * margin) < count:
            margin -= 1
        inner_h, inner_w = max(h - 2 * margin, 1), max(w - 2 * margin, 1)
        rows = max(1, min(inner_h, round(math.sqrt(count * inner_h / inner_w))))
        cols = max(1, min(inner_w, math.ceil(count / rows)))
        while rows * cols < count and rows < inner_h:
            rows += 1
        while rows * cols < count and cols < inner_w:
            cols += 1
        y_edges = torch.linspace(margin, h - margin, rows + 1, device=score.device).round().long()
        x_edges = torch.linspace(margin, w - margin, cols + 1, device=score.device).round().long()
        selected = []
        for row in range(rows):
            for col in range(cols):
                y0, y1 = int(y_edges[row]), int(y_edges[row + 1])
                x0, x1 = int(x_edges[col]), int(x_edges[col + 1])
                if y1 <= y0 or x1 <= x0:
                    continue
                local = score[:, 0, y0:y1, x0:x1].flatten(1)
                index = local.argmax(-1)
                local_w = x1 - x0
                y = torch.div(index, local_w, rounding_mode="floor") + y0
                x = index % local_w + x0
                selected.append(torch.stack((x, y), dim=-1))
        if len(selected) < count:
            raise RuntimeError(f"could select only {len(selected)}/{count} track points")
        if len(selected) > count:
            # Sample the complete cell lattice uniformly instead of dropping
            # the final row/column, which would reintroduce spatial bias.
            keep = torch.linspace(0, len(selected) - 1, count).round().long().tolist()
            selected = [selected[index] for index in keep]
        return torch.stack(selected, dim=1).to(score.dtype)

    def forward_sequence(self, images: torch.Tensor, intrinsics: torch.Tensor,
                         projection_y_sign: torch.Tensor | None = None,
                         detach_state: bool = False) -> dict:
        if images.ndim != 5:
            raise ValueError("images must be (B,T,3,H,W)")
        b, t, _, height, width = images.shape
        if intrinsics.shape != (b, t, 3, 3):
            raise ValueError(f"intrinsics must be {(b,t,3,3)}, got {tuple(intrinsics.shape)}")
        sign = (torch.ones(b, device=images.device, dtype=images.dtype)
                if projection_y_sign is None else projection_y_sign.to(images))
        flat_image = images.reshape(b * t, 3, height, width)
        flat_k = intrinsics.reshape(b * t, 3, 3)
        flat_sign = sign[:, None].expand(b, t).reshape(-1)
        f4, _, f16 = self.pyramid(flat_image, flat_k, flat_sign)
        h4, w4 = f4.shape[-2:]
        scene_context = None
        if self.scene_mixer is not None:
            f16_sequence = f16.reshape(b, t, *f16.shape[1:])
            scene_context = self.scene_mixer(f16_sequence)
            film = self.depth_film(scene_context.reshape(b * t, -1))
            scale, shift = film.chunk(2, dim=-1)
            f4_for_depth = f4 * (1 + 0.1 * scale[..., None, None].tanh()) + shift[..., None, None]
        else:
            f4_for_depth = f4
        raw_depth = self.depth(f4_for_depth)
        metric_depth4 = raw_depth[:, :1].clamp(-4.0, 4.0).exp()
        log_sigma4 = raw_depth[:, 1:2].clamp(-5.0, 3.0)
        depth = F.interpolate(metric_depth4, (height, width), mode="bilinear",
                              align_corners=False).reshape(b, t, height, width)
        log_sigma = F.interpolate(log_sigma4, (height, width), mode="bilinear",
                                  align_corners=False).reshape(b, t, height, width)
        f4 = f4.reshape(b, t, *f4.shape[1:])
        k4 = self._scale_intrinsics(intrinsics, (height, width), (h4, w4))
        analytic_rays = camera_rays(
            k4.reshape(b * t, 3, 3), h4, w4,
            sign[:, None].expand(b, t).reshape(-1))
        if self.ray_head is not None:
            residual_xy = self.ray_residual_scale * torch.tanh(
                self.ray_head(f4_for_depth)).permute(0, 2, 3, 1)
            predicted_rays = analytic_rays.clone()
            predicted_rays[..., :2] = predicted_rays[..., :2] + residual_xy
        else:
            predicted_rays = analytic_rays
        predicted_rays = predicted_rays.reshape(b, t, h4, w4, 3)
        points_all, targets, confidences, confidence_logits = [], [], [], []
        dynamics, dynamic_logits = [], []
        visibility_logits, visibilities, track_log_scales = [], [], []
        residual_flows, track_hidden = [], []
        backward_targets, backward_visibilities = [], []
        initial_transforms, refined_transforms = [], []
        ba_histories = []
        for frame in range(t - 1):
            source, target = f4[:, frame], f4[:, frame + 1]
            points = self._select_points(source)
            track = self.tracker(source, target, points)
            sampled_depth = bilinear_sample(metric_depth4.reshape(b, t, 1, h4, w4)[:, frame],
                                             points).squeeze(-1)
            fx, fy = k4[:, frame, 0, 0][:, None], k4[:, frame, 1, 1][:, None]
            cx, cy = k4[:, frame, 0, 2][:, None], k4[:, frame, 1, 2][:, None]
            point3d = torch.stack(((points[..., 0] - cx) / fx * sampled_depth,
                                   sign[:, None] * (points[..., 1] - cy) / fy * sampled_depth,
                                   sampled_depth), dim=-1)
            if self.ray_head is not None:
                sampled_ray = bilinear_sample(
                    predicted_rays[:, frame].permute(0, 3, 1, 2), points)
                point3d = sampled_ray * sampled_depth[..., None]
            flow = track["target"] - points
            normalized = torch.stack((points[..., 0] / max(w4 - 1, 1),
                                      points[..., 1] / max(h4 - 1, 1)), dim=-1)
            pose_features = torch.cat((track["hidden"], flow / max(h4, w4),
                                       normalized, sampled_depth[..., None] / 10.0,
                                       track["confidence"][..., None],
                                       track["dynamic_probability"][..., None]), dim=-1)
            static_weight = (track["confidence"] * track["visibility_probability"] *
                             (1 - track["dynamic_probability"]))
            pooled = ((pose_features * static_weight[..., None]).sum(1) /
                      static_weight.sum(1, keepdim=True).clamp_min(1e-4))
            if scene_context is not None:
                context_pair = torch.cat((scene_context[:, frame],
                                          scene_context[:, frame + 1]), dim=-1)
                pooled = torch.cat((pooled, self.pose_context(context_pair)), dim=-1)
            initial = se3_exp(self.pose_head(pooled))
            # BA and its Jacobians are a small system; numerical reliability is
            # more valuable than Tensor-Core speed here.
            with torch.autocast(device_type=images.device.type, enabled=False):
                refined, ba_history, _ = bundle_adjust_pair(
                    point3d.float(), track["target"].float(),
                    k4[:, frame + 1].float(), initial.float(),
                    static_weight.float(), sign.float(), iterations=self.ba_iterations)
            points_all.append(points); targets.append(track["target"])
            confidences.append(track["confidence"])
            confidence_logits.append(track["confidence_logit"])
            dynamics.append(track["dynamic_probability"])
            dynamic_logits.append(track["dynamic_logit"])
            visibility_logits.append(track["visibility_logit"])
            visibilities.append(track["visibility_probability"])
            track_log_scales.append(track["log_scale"])
            residual_flows.append(track["scene_flow_residual"])
            track_hidden.append(track["hidden"]); initial_transforms.append(initial)
            refined_transforms.append(refined); ba_histories.append(ba_history)
            if self.ray_head is not None:
                backward = self.tracker(target, source, track["target"])
                backward_targets.append(backward["target"])
                backward_visibilities.append(backward["visibility_probability"])

        relative = torch.stack(refined_transforms, dim=1)
        initial_relative = torch.stack(initial_transforms, dim=1)
        pose = [torch.eye(4, device=images.device, dtype=images.dtype)
                .unsqueeze(0).expand(b, -1, -1)]
        for frame in range(t - 1):
            pose.append(pose[-1] @ torch.linalg.inv(relative[:, frame]))
        pose_c2w = torch.stack(pose, dim=1)
        pointmap = backproject_depth(
            depth.reshape(b * t, height, width), flat_k, flat_sign)
        pointmap = pointmap.reshape(b, t, height, width, 3)
        world = self.world_projection(f16.flatten(2).transpose(1, 2))
        world = world.reshape(b, t, world.shape[1], world.shape[2])
        summary = torch.stack([h.mean(1) for h in track_hidden], dim=1)
        track_tokens = self.track_projection(summary)
        temporal_source = (scene_context if scene_context is not None else
                           f16.reshape(b, t, *f16.shape[1:]).mean(dim=(-1, -2)))
        left, right = temporal_source[:, :-1], temporal_source[:, 1:]
        temporal_features = torch.cat((left, right, right - left, left * right), dim=-1)
        temporal_logits = self.temporal_compatibility(temporal_features).squeeze(-1)
        if detach_state:
            world = world.detach(); track_tokens = track_tokens.detach()
        result = {
            "depth": depth, "depth_log_sigma": log_sigma,
            "ray_map_feature": predicted_rays,
            "pointmap_camera": pointmap,
            "pose_c2w": pose_c2w, "relative_pose": relative,
            "initial_relative_pose": initial_relative,
            "track_source": torch.stack(points_all, dim=1),
            "track_target": torch.stack(targets, dim=1),
            "track_confidence": torch.stack(confidences, dim=1),
            "track_confidence_logit": torch.stack(confidence_logits, dim=1),
            "track_visibility_logit": torch.stack(visibility_logits, dim=1),
            "track_visibility": torch.stack(visibilities, dim=1),
            "track_log_scale": torch.stack(track_log_scales, dim=1),
            "dynamic_probability": torch.stack(dynamics, dim=1),
            "dynamic_logit": torch.stack(dynamic_logits, dim=1),
            "scene_flow_residual": torch.stack(residual_flows, dim=1),
            "ba_residual_history": torch.stack(ba_histories, dim=1),
            "world_tokens": world, "track_tokens": track_tokens,
            "feature_hw": (h4, w4), "intrinsics_feature": k4,
            "scene_context": scene_context,
            "temporal_compatibility_logit": temporal_logits,
            "temporal_compatibility": torch.sigmoid(temporal_logits),
        }
        if backward_targets:
            result["track_backward_target"] = torch.stack(backward_targets, dim=1)
            result["track_backward_visibility"] = torch.stack(backward_visibilities, dim=1)
        return result

    def loss(self, output: dict, depth_gt: torch.Tensor, pose_gt_c2w: torch.Tensor,
             depth_valid: torch.Tensor, intrinsics: torch.Tensor,
             projection_y_sign: torch.Tensor, dynamic_gt: torch.Tensor | None = None,
             rigid_flow: torch.Tensor | None = None,
             rigid_flow_valid: torch.Tensor | None = None,
             counterfactual_output: dict | None = None,
             temporal_negative_output: dict | None = None,
             weights: dict[str, float] | None = None) -> tuple[torch.Tensor, dict]:
        weights = weights or {}
        w = lambda name, default: float(weights.get(name, default))
        valid = depth_valid & torch.isfinite(depth_gt) & (depth_gt > 1e-4)
        log_error = (output["depth"].clamp_min(1e-4).log() -
                     depth_gt.clamp_min(1e-4).log()).abs()
        depth_terms = (log_error * torch.exp(-output["depth_log_sigma"]) +
                       output["depth_log_sigma"])
        depth_nll = (depth_terms.masked_select(valid).mean() if bool(valid.any()) else
                     torch.nan_to_num(depth_terms).sum() * 0)
        # Aleatoric NLL alone can improve by inflating sigma while metric depth
        # remains worse than a constant prior. Keep uncertainty-independent
        # metric and edge terms.
        relative_error = ((output["depth"] - depth_gt).abs() /
                          depth_gt.clamp_min(0.10)).clamp_max(5.0)
        depth_absrel = (relative_error.masked_select(valid).mean()
                        if bool(valid.any()) else relative_error.sum() * 0)
        pred_log = output["depth"].clamp_min(1e-4).log()
        truth_log = depth_gt.clamp_min(1e-4).log()
        grad_terms = []
        for axis in (-1, -2):
            pred_delta = torch.diff(pred_log, dim=axis)
            truth_delta = torch.diff(truth_log, dim=axis)
            pair_valid = (valid[..., 1:] & valid[..., :-1] if axis == -1 else
                          valid[..., 1:, :] & valid[..., :-1, :])
            delta = (pred_delta - truth_delta).abs()
            if bool(pair_valid.any()):
                grad_terms.append(delta.masked_select(pair_valid).mean())
        depth_gradient = (torch.stack(grad_terms).mean() if grad_terms else
                          depth_absrel.new_zeros(()))

        gt_relative = torch.stack([
            camera_transform(pose_gt_c2w[:, i], pose_gt_c2w[:, i + 1])
            for i in range(pose_gt_c2w.shape[1] - 1)], dim=1)
        pred_relative = output["relative_pose"]
        rotation_loss = rotation_geodesic(pred_relative[..., :3, :3],
                                          gt_relative[..., :3, :3]).mean()
        gt_motion = gt_relative[..., :3, 3].norm(dim=-1).detach().clamp_min(.01)
        translation_loss = ((pred_relative[..., :3, 3] - gt_relative[..., :3, 3])
                            .norm(dim=-1) / gt_motion).clamp(max=10).mean()
        initial_relative = output["initial_relative_pose"]
        initial_rotation = rotation_geodesic(initial_relative[..., :3, :3],
                                             gt_relative[..., :3, :3]).mean()
        initial_translation = ((initial_relative[..., :3, 3] -
                                gt_relative[..., :3, 3]).norm(dim=-1) /
                               gt_motion).clamp(max=10).mean()
        initial_pose_loss = initial_rotation + initial_translation

        track_loss = depth_nll.new_zeros(())
        dynamic_loss = depth_nll.new_zeros(())
        rigid_consistency = depth_nll.new_zeros(())
        track_valid_fraction = depth_nll.new_zeros(())
        confidence_loss = depth_nll.new_zeros(())
        visibility_loss = depth_nll.new_zeros(())
        temporal_loss = F.binary_cross_entropy_with_logits(
            output["temporal_compatibility_logit"],
            torch.ones_like(output["temporal_compatibility_logit"]))
        track_cycle = depth_nll.new_zeros(())
        ray_loss = depth_nll.new_zeros(())
        if "ray_map_feature" in output:
            h4, w4 = output["feature_hw"]
            b, t = intrinsics.shape[:2]
            signs = projection_y_sign[:, None].expand(b, t).reshape(-1)
            target_ray = camera_rays(
                output["intrinsics_feature"].reshape(-1, 3, 3), h4, w4, signs)
            predicted_ray = output["ray_map_feature"].reshape_as(target_ray)
            ray_loss = (1 - (F.normalize(predicted_ray, dim=-1) *
                             F.normalize(target_ray, dim=-1)).sum(-1)).mean()
        if "track_backward_target" in output:
            cycle_error = torch.linalg.vector_norm(
                output["track_backward_target"] - output["track_source"], dim=-1)
            cycle_weight = (output["track_visibility"] *
                            output["track_backward_visibility"]).detach()
            track_cycle = ((cycle_error * cycle_weight).sum() /
                           cycle_weight.sum().clamp_min(1))
        if rigid_flow is not None:
            b, pairs, h, width, _ = rigid_flow.shape
            h4, w4 = output["feature_hw"]
            source_valid = (rigid_flow_valid if rigid_flow_valid is not None else
                            torch.ones(b, pairs, h, width, dtype=torch.bool,
                                       device=rigid_flow.device))
            flow4, valid4 = resize_flow_with_valid(
                rigid_flow, source_valid, (h4, w4))
            source = output["track_source"]
            gt_track, gt_ok = [], []
            for index in range(pairs):
                sampled = bilinear_sample(flow4[:, index], source[:, index])
                gt_track.append(source[:, index] + sampled)
                gt_ok.append(bilinear_sample(
                    valid4[:, index:index + 1].float(), source[:, index]
                ).squeeze(-1) > .5)
            gt_track = torch.stack(gt_track, 1); gt_ok = torch.stack(gt_ok, 1)
            epe = torch.linalg.vector_norm(output["track_target"] - gt_track, dim=-1)
            finite = torch.isfinite(epe)
            target_inside = ((gt_track[..., 0] >= 0) & (gt_track[..., 0] <= w4 - 1) &
                             (gt_track[..., 1] >= 0) & (gt_track[..., 1] <= h4 - 1))
            supervised = gt_ok & finite & target_inside
            safe_epe = torch.nan_to_num(epe, nan=1e3, posinf=1e3, neginf=1e3).clamp_max(1e3)
            track_valid_fraction = supervised.float().mean()
            # Heteroscedastic Laplace NLL is robust to the heavy-tailed tracking
            # failures observed on real held-out video. Invalid/occluded points
            # do not contribute a fabricated coordinate target.
            scale = output["track_log_scale"].exp().clamp_min(1e-3)
            robust_track = safe_epe / scale + output["track_log_scale"]
            track_loss = (robust_track.masked_select(supervised).mean()
                          if bool(supervised.any()) else robust_track.sum() * 0)
            threshold = w("confidence_threshold_px", 3.0)
            confidence_target = (supervised & (safe_epe <= threshold)).to(
                output["track_confidence"].dtype)
            confidence_bce = F.binary_cross_entropy_with_logits(
                output["track_confidence_logit"], confidence_target,
                reduction="none")
            confidence_loss = confidence_bce.mean()
            visibility_target = supervised.to(output["track_visibility_logit"].dtype)
            visibility_loss = F.binary_cross_entropy_with_logits(
                output["track_visibility_logit"], visibility_target)
            static = ((1 - output["dynamic_probability"]) *
                      output["track_confidence"] * output["track_visibility"])
            static = static * supervised.to(static.dtype)
            rigid_consistency = ((safe_epe * static).sum() /
                                 static.sum().clamp_min(1))
        if dynamic_gt is not None:
            h4, w4 = output["feature_hw"]
            labels = []
            for index in range(dynamic_gt.shape[1] - 1):
                mask = F.interpolate(dynamic_gt[:, index:index+1].float(),
                                     (h4, w4), mode="nearest")
                labels.append(bilinear_sample(mask, output["track_source"][:, index]).squeeze(-1))
            labels = torch.stack(labels, 1)
            # Balanced focal BCE prevents the all-static solution on sources
            # where dynamic points are rare.
            positives = labels.sum().detach()
            negatives = labels.numel() - positives
            pos_weight = (negatives / positives.clamp_min(1)).clamp(1.0, 20.0)
            dynamic_bce = F.binary_cross_entropy_with_logits(
                output["dynamic_logit"], labels, pos_weight=pos_weight, reduction="none")
            probability = output["dynamic_probability"]
            pt = labels * probability + (1 - labels) * (1 - probability)
            focal = ((1 - pt).square() * dynamic_bce).mean()
            intersection = (probability * labels).sum()
            dice = 1 - ((2 * intersection + 1) /
                        (probability.sum() + labels.sum() + 1))
            dynamic_loss = focal + dice
        ba = torch.nan_to_num(output["ba_residual_history"], nan=1e3,
                              posinf=1e3, neginf=1e3)
        ba_monotonic = F.relu(ba[..., -1] - ba[..., 0]).mean()
        calibration_contrast = depth_nll.new_zeros(())
        pose_counterfactual = depth_nll.new_zeros(())
        if counterfactual_output is not None:
            wrong_ba = counterfactual_output["ba_residual_history"][..., -1]
            calibration_contrast = F.relu(.05 + ba[..., -1] - wrong_ba).mean()
            wrong_relative = counterfactual_output["initial_relative_pose"]
            wrong_rotation = rotation_geodesic(
                wrong_relative[..., :3, :3], gt_relative[..., :3, :3])
            wrong_translation = ((wrong_relative[..., :3, 3] -
                                  gt_relative[..., :3, 3]).norm(dim=-1) /
                                 gt_motion).clamp(max=10)
            normal_error = (
                rotation_geodesic(initial_relative[..., :3, :3],
                                  gt_relative[..., :3, :3]) +
                ((initial_relative[..., :3, 3] - gt_relative[..., :3, 3])
                 .norm(dim=-1) / gt_motion).clamp(max=10))
            pose_counterfactual = F.relu(
                .10 + normal_error - (wrong_rotation + wrong_translation)).mean()
        if temporal_negative_output is not None:
            negative_logits = temporal_negative_output["temporal_compatibility_logit"]
            temporal_loss = temporal_loss + F.binary_cross_entropy_with_logits(
                negative_logits, torch.zeros_like(negative_logits))
            temporal_loss = temporal_loss + F.relu(
                w("temporal_margin", 0.0) -
                output["temporal_compatibility_logit"] + negative_logits).mean()
        total = (w("depth", 1.0) * depth_nll + w("rotation", 1.0) * rotation_loss +
                 w("translation", 1.0) * translation_loss + w("track", 1.0) * track_loss +
                 w("depth_absrel", 0.0) * depth_absrel +
                 w("depth_gradient", 0.0) * depth_gradient +
                 w("initial_pose", 0.0) * initial_pose_loss +
                 w("rigid", .25) * rigid_consistency + w("dynamic", .2) * dynamic_loss +
                 w("ba", .2) * ba_monotonic +
                 w("counterfactual", .15) * calibration_contrast +
                 w("pose_counterfactual", 0.0) * pose_counterfactual +
                 w("ray", 0.0) * ray_loss +
                 w("track_cycle", 0.0) * track_cycle +
                 w("confidence", 0.0) * confidence_loss +
                 w("visibility", 0.0) * visibility_loss +
                 w("temporal", 0.0) * temporal_loss)
        ba_initial = ba[..., 0]
        ba_reduction = torch.where(
            ba_initial > .25,
            1 - ba[..., -1] / ba_initial.clamp_min(1e-6),
            torch.zeros_like(ba_initial)).mean()
        metrics = {"geometry_depth_nll": float(depth_nll.detach()),
                   "geometry_depth_absrel_loss": float(depth_absrel.detach()),
                   "geometry_depth_gradient": float(depth_gradient.detach()),
                   "geometry_rotation": float(rotation_loss.detach()),
                   "geometry_translation": float(translation_loss.detach()),
                   "geometry_initial_pose": float(initial_pose_loss.detach()),
                   "geometry_track_epe": float(track_loss.detach()),
                   "geometry_track_valid_fraction": float(track_valid_fraction.detach()),
                   "geometry_rigid_epe": float(rigid_consistency.detach()),
                   "geometry_dynamic_bce": float(dynamic_loss.detach()),
                   "geometry_ba_monotonic": float(ba_monotonic.detach()),
                   "geometry_calibration_contrast": float(calibration_contrast.detach()),
                   "geometry_pose_counterfactual": float(pose_counterfactual.detach()),
                   "geometry_ray_angular": float(ray_loss.detach()),
                   "geometry_track_cycle": float(track_cycle.detach()),
                   "geometry_confidence_bce": float(confidence_loss.detach()),
                   "geometry_visibility_bce": float(visibility_loss.detach()),
                   "geometry_temporal_bce": float(temporal_loss.detach()),
                   "geometry_ba_reduction": float(ba_reduction.detach())}
        return total, metrics
