"""Held-out metrics, causal controls and DDP helpers for CTPG-Eye v3."""

from __future__ import annotations

import math
from typing import Iterable

import torch
import torch.nn.functional as F

from .geometry_math_v3 import bilinear_sample, camera_transform
from .geometry_v3_data import make_counterfactuals
from .mathx import rotation_geodesic


def move_geometry_batch(batch: dict, device: torch.device) -> dict:
    return {key: (value.to(device, non_blocking=True)
                  if torch.is_tensor(value) else value)
            for key, value in batch.items()}


def depth_metrics(prediction: torch.Tensor, target: torch.Tensor,
                  valid: torch.Tensor) -> dict[str, float]:
    mask = valid & torch.isfinite(prediction) & torch.isfinite(target) & (target > 0)
    pred, truth = prediction[mask].clamp_min(1e-4), target[mask].clamp_min(1e-4)
    if not pred.numel():
        return {"depth_abs_rel": math.inf, "depth_rmse": math.inf, "depth_delta1": 0.0}
    ratio = torch.maximum(pred / truth, truth / pred)
    return {"depth_abs_rel": float(((pred - truth).abs() / truth).mean()),
            "depth_rmse": float(torch.sqrt((pred - truth).square().mean())),
            "depth_delta1": float((ratio < 1.25).float().mean())}


def pose_metrics(prediction: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    target = torch.linalg.inv(target[:, :1]) @ target
    translation = torch.linalg.vector_norm(
        prediction[..., :3, 3] - target[..., :3, 3], dim=-1)
    pred_rel, gt_rel = [], []
    for index in range(target.shape[1] - 1):
        pred_rel.append(camera_transform(prediction[:, index], prediction[:, index + 1]))
        gt_rel.append(camera_transform(target[:, index], target[:, index + 1]))
    pred_rel, gt_rel = torch.stack(pred_rel, 1), torch.stack(gt_rel, 1)
    rpe_t = torch.linalg.vector_norm(pred_rel[..., :3, 3] - gt_rel[..., :3, 3], dim=-1)
    rpe_r = rotation_geodesic(pred_rel[..., :3, :3], gt_rel[..., :3, :3])
    return {"ate_metric": float(torch.sqrt(translation.square().mean())),
            "rpe_translation": float(rpe_t.mean()),
            "rpe_rotation_deg": float(rpe_r.mean() * 180 / math.pi)}


def track_metrics(output: dict, batch: dict) -> dict[str, float]:
    rigid = batch["rigid_flow"]
    valid = batch["rigid_flow_valid"]
    b, pairs, height, width, _ = rigid.shape
    h4, w4 = output["feature_hw"]
    flow = F.interpolate(rigid.permute(0, 1, 4, 2, 3).reshape(-1, 2, height, width),
                         (h4, w4), mode="bilinear", align_corners=False)
    flow[:, 0] *= w4 / width; flow[:, 1] *= h4 / height
    flow = flow.reshape(b, pairs, 2, h4, w4)
    errors, confidences = [], []
    for index in range(pairs):
        points = output["track_source"][:, index]
        target = points + bilinear_sample(flow[:, index], points)
        mask4 = F.interpolate(valid[:, index:index + 1].float(), (h4, w4), mode="nearest")
        mask = bilinear_sample(mask4, points).squeeze(-1) > .5
        epe = torch.linalg.vector_norm(output["track_target"][:, index] - target, dim=-1)
        if bool(mask.any()):
            errors.append(epe[mask]); confidences.append(output["track_confidence"][:, index][mask])
    if not errors:
        return {"track_epe": math.inf, "track_confidence": 0.0}
    return {"track_epe": float(torch.cat(errors).mean()),
            "track_confidence": float(torch.cat(confidences).mean())}


@torch.no_grad()
def _infer(model, batch: dict, image: torch.Tensor | None = None,
           intrinsics: torch.Tensor | None = None) -> tuple[dict, dict]:
    output = model.encode_geometry_sequence(
        batch["image"] if image is None else image,
        intrinsics=batch["intrinsics"] if intrinsics is None else intrinsics,
        projection_y_sign=batch["projection_y_sign"])
    metrics = depth_metrics(output["depth"], batch["depth"], batch["depth_valid"])
    metrics.update(pose_metrics(output["pose_c2w"], batch["pose_c2w"]))
    metrics.update(track_metrics(output, batch))
    ba = output["ba_residual_history"]
    initial = ba[..., 0]
    reduction = torch.where(initial > .25,
                            1 - ba[..., -1] / initial.clamp_min(1e-6),
                            torch.zeros_like(initial))
    metrics["ba_residual_reduction"] = float(reduction.mean())
    return output, metrics


@torch.no_grad()
def evaluate_geometry_v3_controls(model, batches: Iterable[dict],
                                  device: torch.device, max_windows: int,
                                  depth_prior_m: float) -> dict:
    """Evaluate fixed OOD windows plus black/frozen/reverse/wrong-window/K."""
    model.eval()
    keys = ("depth_abs_rel", "depth_rmse", "depth_delta1", "ate_metric",
            "rpe_translation", "rpe_rotation_deg", "track_epe",
            "track_confidence", "ba_residual_reduction")
    accum = {name: {key: [] for key in keys}
             for name in ("normal", "black", "frozen", "reverse",
                          "wrong_window", "wrong_intrinsics")}
    baseline_absrel, identity_ate = [], []
    windows = 0
    for raw in batches:
        if windows >= max_windows:
            break
        batch = move_geometry_batch(raw, device)
        controls = make_counterfactuals(batch)
        variants = {
            "normal": (batch["image"], batch["intrinsics"]),
            "black": (torch.zeros_like(batch["image"]), batch["intrinsics"]),
            "frozen": (controls["frozen_image"], batch["intrinsics"]),
            "reverse": (controls["reverse_image"], batch["intrinsics"]),
            "wrong_window": (controls["wrong_window_image"], batch["intrinsics"]),
            "wrong_intrinsics": (batch["image"], controls["wrong_intrinsics"]),
        }
        for name, (image, k) in variants.items():
            _, metrics = _infer(model, batch, image, k)
            for key in keys:
                accum[name][key].append(metrics[key])
        prior = torch.full_like(batch["depth"], depth_prior_m)
        baseline_absrel.append(depth_metrics(prior, batch["depth"],
                                             batch["depth_valid"])["depth_abs_rel"])
        identity = torch.eye(4, device=device).view(1, 1, 4, 4).expand_as(batch["pose_c2w"])
        identity_ate.append(pose_metrics(identity, batch["pose_c2w"])["ate_metric"])
        windows += batch["image"].shape[0]
    controls = {name: {key: sum(values) / len(values) if values else math.inf
                       for key, values in metrics.items()}
                for name, metrics in accum.items()}
    normal = controls["normal"]
    depth_baseline = sum(baseline_absrel) / max(len(baseline_absrel), 1)
    pose_baseline = sum(identity_ate) / max(len(identity_ate), 1)
    ratio = lambda numerator, denominator: numerator / max(denominator, 1e-8)
    ratios = {
        "depth_prior_gain": ratio(depth_baseline, normal["depth_abs_rel"]),
        "pose_identity_gain": ratio(pose_baseline, normal["ate_metric"]),
        "ba_residual_reduction": normal["ba_residual_reduction"],
        "wrong_window_depth_ratio": ratio(controls["wrong_window"]["depth_abs_rel"],
                                            normal["depth_abs_rel"]),
        "wrong_window_pose_ratio": ratio(controls["wrong_window"]["ate_metric"],
                                           normal["ate_metric"]),
        "reverse_time_rpe_ratio": ratio(controls["reverse"]["rpe_translation"],
                                         normal["rpe_translation"]),
        "wrong_intrinsics_pose_ratio": ratio(controls["wrong_intrinsics"]["ate_metric"],
                                               normal["ate_metric"]),
        "black_depth_ratio": ratio(controls["black"]["depth_abs_rel"],
                                     normal["depth_abs_rel"]),
        "frozen_pose_ratio": ratio(controls["frozen"]["ate_metric"],
                                    normal["ate_metric"]),
    }
    gates = {
        "G_depth_beats_prior_20pct": ratios["depth_prior_gain"] >= 1.20,
        "G_pose_beats_identity_20pct": ratios["pose_identity_gain"] >= 1.20,
        "G_ba_reduces_residual_15pct": ratios["ba_residual_reduction"] >= .15,
        "G_wrong_window_hurts_depth_25pct": ratios["wrong_window_depth_ratio"] >= 1.25,
        "G_wrong_window_hurts_pose_25pct": ratios["wrong_window_pose_ratio"] >= 1.25,
        "G_reverse_time_detected": ratios["reverse_time_rpe_ratio"] >= 1.10,
        "G_wrong_intrinsics_detected": ratios["wrong_intrinsics_pose_ratio"] >= 1.15,
    }
    return {"valid": all(gates.values()), "windows": windows,
            "controls": controls, "baselines": {
                "fixed_depth_abs_rel": depth_baseline, "identity_ate": pose_baseline},
            "ratios": ratios, "gates": gates}


def controller_metrics(report: dict) -> dict[str, float]:
    """Exact metric dictionary consumed by ``AdaptiveTrainingBudget``."""
    return {name: float(report["ratios"][name]) for name in (
        "depth_prior_gain", "pose_identity_gain", "ba_residual_reduction",
        "wrong_window_depth_ratio", "wrong_window_pose_ratio",
        "reverse_time_rpe_ratio", "wrong_intrinsics_pose_ratio")}
