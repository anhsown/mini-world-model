"""CTPG-Eye v3: calibrated tracks, point geometry and unrolled BA.

This module is deliberately independent from the language tower. It consumes
RGB windows and per-frame camera calibration, predicts metric depth and sparse
tracks, separates dynamic evidence, estimates relative SE(3), then refines it
with differentiable pose-only bundle adjustment. Compact world tokens are the
only interface required by the rest of JWM.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .geometry_math_v3 import (
    backproject_depth, bilinear_sample, bundle_adjust_pair, camera_rays, camera_transform,
    project_points, se3_exp, transform_points,
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
        hidden_2d = hidden.reshape(b, n, -1)
        dynamic_logit = self.dynamic(hidden_2d).squeeze(-1)
        return {"target": current, "hidden": hidden_2d,
                "confidence": torch.sigmoid(self.confidence(hidden_2d).squeeze(-1)),
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


class CTPGPhysicalEye(nn.Module):
    def __init__(self, world_dim: int, width: int = 96, track_points: int = 64,
                 track_radius: int = 2, track_iterations: int = 3,
                 ba_iterations: int = 2, memory_frames: int = 32):
        super().__init__()
        self.track_points = track_points
        self.ba_iterations = ba_iterations
        self.pyramid = CalibratedPyramid(width)
        c4, _, c16 = self.pyramid.channels
        self.depth = nn.Sequential(ResidualConv(c4), nn.Conv2d(c4, 2, 1))
        self.tracker = SparseTrackUpdater(c4, hidden=128, radius=track_radius,
                                          iterations=track_iterations)
        self.pose_head = nn.Sequential(nn.Linear(128 + 7, 128), nn.SiLU(),
                                       nn.Linear(128, 6))
        nn.init.zeros_(self.pose_head[-1].weight)
        nn.init.zeros_(self.pose_head[-1].bias)
        self.world_projection = nn.Linear(c16, world_dim)
        self.track_projection = nn.Linear(128, world_dim)
        self.memory = BoundedWorldMemory(memory_frames)
        # No metric residual-flow target exists in this curriculum.  Start
        # from the conservative rigid-world prior instead of random flow.
        nn.init.zeros_(self.tracker.residual_3d_flow.weight)
        nn.init.zeros_(self.tracker.residual_3d_flow.bias)

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
        index = score.flatten(1).topk(count, dim=-1).indices
        y = torch.div(index, w, rounding_mode="floor")
        x = index % w
        return torch.stack((x, y), dim=-1).to(score.dtype)

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
        raw_depth = self.depth(f4)
        metric_depth4 = raw_depth[:, :1].clamp(-4.0, 4.0).exp()
        log_sigma4 = raw_depth[:, 1:2].clamp(-5.0, 3.0)
        depth = F.interpolate(metric_depth4, (height, width), mode="bilinear",
                              align_corners=False).reshape(b, t, height, width)
        log_sigma = F.interpolate(log_sigma4, (height, width), mode="bilinear",
                                  align_corners=False).reshape(b, t, height, width)
        f4 = f4.reshape(b, t, *f4.shape[1:])
        k4 = self._scale_intrinsics(intrinsics, (height, width), (h4, w4))
        points_all, targets, confidences, dynamics, dynamic_logits = [], [], [], [], []
        residual_flows, track_hidden = [], []
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
            flow = track["target"] - points
            normalized = torch.stack((points[..., 0] / max(w4 - 1, 1),
                                      points[..., 1] / max(h4 - 1, 1)), dim=-1)
            pose_features = torch.cat((track["hidden"], flow / max(h4, w4),
                                       normalized, sampled_depth[..., None] / 10.0,
                                       track["confidence"][..., None],
                                       track["dynamic_probability"][..., None]), dim=-1)
            static_weight = track["confidence"] * (1 - track["dynamic_probability"])
            pooled = ((pose_features * static_weight[..., None]).sum(1) /
                      static_weight.sum(1, keepdim=True).clamp_min(1e-4))
            initial = se3_exp(self.pose_head(pooled))
            refined, ba_history, _ = bundle_adjust_pair(
                point3d, track["target"], k4[:, frame + 1], initial,
                static_weight, sign, iterations=self.ba_iterations)
            points_all.append(points); targets.append(track["target"])
            confidences.append(track["confidence"]); dynamics.append(track["dynamic_probability"])
            dynamic_logits.append(track["dynamic_logit"])
            residual_flows.append(track["scene_flow_residual"])
            track_hidden.append(track["hidden"]); initial_transforms.append(initial)
            refined_transforms.append(refined); ba_histories.append(ba_history)

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
        if detach_state:
            world = world.detach(); track_tokens = track_tokens.detach()
        return {
            "depth": depth, "depth_log_sigma": log_sigma,
            "pointmap_camera": pointmap,
            "pose_c2w": pose_c2w, "relative_pose": relative,
            "initial_relative_pose": initial_relative,
            "track_source": torch.stack(points_all, dim=1),
            "track_target": torch.stack(targets, dim=1),
            "track_confidence": torch.stack(confidences, dim=1),
            "dynamic_probability": torch.stack(dynamics, dim=1),
            "dynamic_logit": torch.stack(dynamic_logits, dim=1),
            "scene_flow_residual": torch.stack(residual_flows, dim=1),
            "ba_residual_history": torch.stack(ba_histories, dim=1),
            "world_tokens": world, "track_tokens": track_tokens,
            "feature_hw": (h4, w4), "intrinsics_feature": k4,
        }

    def loss(self, output: dict, depth_gt: torch.Tensor, pose_gt_c2w: torch.Tensor,
             depth_valid: torch.Tensor, intrinsics: torch.Tensor,
             projection_y_sign: torch.Tensor, dynamic_gt: torch.Tensor | None = None,
             rigid_flow: torch.Tensor | None = None,
             rigid_flow_valid: torch.Tensor | None = None,
             counterfactual_output: dict | None = None,
             weights: dict[str, float] | None = None) -> tuple[torch.Tensor, dict]:
        weights = weights or {}
        w = lambda name, default: float(weights.get(name, default))
        valid = depth_valid & torch.isfinite(depth_gt) & (depth_gt > 1e-4)
        log_error = (output["depth"].clamp_min(1e-4).log() -
                     depth_gt.clamp_min(1e-4).log()).abs()
        depth_nll = ((log_error * torch.exp(-output["depth_log_sigma"]) +
                      output["depth_log_sigma"]).masked_select(valid).mean())

        gt_relative = torch.stack([
            camera_transform(pose_gt_c2w[:, i], pose_gt_c2w[:, i + 1])
            for i in range(pose_gt_c2w.shape[1] - 1)], dim=1)
        pred_relative = output["relative_pose"]
        rotation_loss = rotation_geodesic(pred_relative[..., :3, :3],
                                          gt_relative[..., :3, :3]).mean()
        gt_motion = gt_relative[..., :3, 3].norm(dim=-1).detach().clamp_min(.01)
        translation_loss = ((pred_relative[..., :3, 3] - gt_relative[..., :3, 3])
                            .norm(dim=-1) / gt_motion).clamp(max=10).mean()

        track_loss = depth_nll.new_zeros(())
        dynamic_loss = depth_nll.new_zeros(())
        rigid_consistency = depth_nll.new_zeros(())
        if rigid_flow is not None:
            b, pairs, h, width, _ = rigid_flow.shape
            h4, w4 = output["feature_hw"]
            flow4 = F.interpolate(rigid_flow.permute(0, 1, 4, 2, 3).reshape(-1, 2, h, width),
                                  (h4, w4), mode="bilinear", align_corners=False)
            flow4[:, 0] *= w4 / width; flow4[:, 1] *= h4 / h
            flow4 = flow4.reshape(b, pairs, 2, h4, w4)
            source = output["track_source"]
            gt_track, gt_ok = [], []
            for index in range(pairs):
                sampled = bilinear_sample(flow4[:, index], source[:, index])
                gt_track.append(source[:, index] + sampled)
                if rigid_flow_valid is None:
                    gt_ok.append(torch.ones_like(source[:, index, :, 0], dtype=torch.bool))
                else:
                    mask = F.interpolate(rigid_flow_valid[:, index:index+1].float(),
                                         (h4, w4), mode="nearest")
                    gt_ok.append(bilinear_sample(mask, source[:, index]).squeeze(-1) > .5)
            gt_track = torch.stack(gt_track, 1); gt_ok = torch.stack(gt_ok, 1)
            epe = torch.linalg.vector_norm(output["track_target"] - gt_track, dim=-1)
            track_loss = epe.masked_select(gt_ok).mean() if bool(gt_ok.any()) else epe.mean() * 0
            static = (1 - output["dynamic_probability"]) * output["track_confidence"]
            rigid_consistency = (epe * static).sum() / static.sum().clamp_min(1)
        if dynamic_gt is not None:
            h4, w4 = output["feature_hw"]
            labels = []
            for index in range(dynamic_gt.shape[1] - 1):
                mask = F.interpolate(dynamic_gt[:, index:index+1].float(),
                                     (h4, w4), mode="nearest")
                labels.append(bilinear_sample(mask, output["track_source"][:, index]).squeeze(-1))
            labels = torch.stack(labels, 1)
            dynamic_loss = F.binary_cross_entropy_with_logits(output["dynamic_logit"], labels)
        ba = output["ba_residual_history"]
        ba_monotonic = F.relu(ba[..., -1] - ba[..., 0]).mean()
        calibration_contrast = depth_nll.new_zeros(())
        if counterfactual_output is not None:
            wrong_ba = counterfactual_output["ba_residual_history"][..., -1]
            calibration_contrast = F.relu(.05 + ba[..., -1] - wrong_ba).mean()
        total = (w("depth", 1.0) * depth_nll + w("rotation", 1.0) * rotation_loss +
                 w("translation", 1.0) * translation_loss + w("track", 1.0) * track_loss +
                 w("rigid", .25) * rigid_consistency + w("dynamic", .2) * dynamic_loss +
                 w("ba", .2) * ba_monotonic +
                 w("counterfactual", .15) * calibration_contrast)
        ba_initial = ba[..., 0]
        ba_reduction = torch.where(
            ba_initial > .25,
            1 - ba[..., -1] / ba_initial.clamp_min(1e-6),
            torch.zeros_like(ba_initial)).mean()
        metrics = {"geometry_depth_nll": float(depth_nll.detach()),
                   "geometry_rotation": float(rotation_loss.detach()),
                   "geometry_translation": float(translation_loss.detach()),
                   "geometry_track_epe": float(track_loss.detach()),
                   "geometry_rigid_epe": float(rigid_consistency.detach()),
                   "geometry_dynamic_bce": float(dynamic_loss.detach()),
                   "geometry_ba_monotonic": float(ba_monotonic.detach()),
                   "geometry_calibration_contrast": float(calibration_contrast.detach()),
                   "geometry_ba_reduction": float(ba_reduction.detach())}
        return total, metrics
