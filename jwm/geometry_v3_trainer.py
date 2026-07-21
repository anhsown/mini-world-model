"""Held-out metrics, causal controls and DDP helpers for CTPG-Eye v3."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Iterable

import torch
import torch.nn.functional as F

from .geometry_math_v3 import bilinear_sample, camera_transform
from .geometry_v3_data import make_counterfactuals
from .mathx import rotation_geodesic
from .training_metrics_v2 import expected_calibration_error, harmonic_score


# These branches have no target in isolated physical-eye training.  They are
# unlocked later for semantic token alignment or metric scene-flow training.
PHYSICAL_EYE_FROZEN_PREFIXES = (
    "pyramid.down8",
    "pyramid.down16",
    "world_projection",
    "track_projection",
    "tracker.residual_3d_flow",
)


def set_eye_v3_physical_trainable(model) -> list[str]:
    """Freeze JWM except the exactly supervised Eye-v3 physical graph."""
    if getattr(model, "geometry", None) is None:
        raise ValueError("model has no geometry branch")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    active = []
    frozen_prefixes = PHYSICAL_EYE_FROZEN_PREFIXES
    if getattr(model.cfg, "geometry_version", "v1") == "v32_ctpg":
        # v3.2 scene registers consume the 1/16 pyramid, so those feature
        # stages are part of a supervised pose/depth path and must be learned.
        frozen_prefixes = tuple(prefix for prefix in frozen_prefixes
                                if prefix not in ("pyramid.down8", "pyramid.down16"))
    for name, parameter in model.geometry.named_parameters():
        enabled = not name.startswith(frozen_prefixes)
        parameter.requires_grad_(enabled)
        if enabled:
            active.append(f"geometry.{name}")
    if not active:
        raise RuntimeError("Eye-v3 physical graph has no trainable parameters")
    return active


def missing_trainable_gradients(model) -> list[str]:
    """List active parameters disconnected from the latest backward graph."""
    return [name for name, parameter in model.named_parameters()
            if parameter.requires_grad and parameter.grad is None]


def move_geometry_batch(batch: dict, device: torch.device) -> dict:
    return {key: (value.to(device, non_blocking=True)
                  if torch.is_tensor(value) else value)
            for key, value in batch.items()}


def depth_metrics(prediction: torch.Tensor, target: torch.Tensor,
                  valid: torch.Tensor) -> dict[str, float]:
    mask = valid & torch.isfinite(prediction) & torch.isfinite(target) & (target > 0)
    pred, truth = prediction[mask].clamp_min(1e-4), target[mask].clamp_min(1e-4)
    if not pred.numel():
        return {"depth_abs_rel": math.inf, "depth_rmse": math.inf,
                "depth_log_rmse": math.inf, "depth_silog": math.inf,
                "depth_delta1": 0.0, "depth_delta2": 0.0, "depth_delta3": 0.0}
    ratio = torch.maximum(pred / truth, truth / pred)
    log_delta = pred.log() - truth.log()
    silog = torch.sqrt((log_delta.square().mean() - .5 * log_delta.mean().square())
                       .clamp_min(0))
    return {"depth_abs_rel": float(((pred - truth).abs() / truth).mean()),
            "depth_rmse": float(torch.sqrt((pred - truth).square().mean())),
            "depth_log_rmse": float(torch.sqrt(log_delta.square().mean())),
            "depth_silog": float(silog),
            "depth_delta1": float((ratio < 1.25).float().mean()),
            "depth_delta2": float((ratio < 1.25 ** 2).float().mean()),
            "depth_delta3": float((ratio < 1.25 ** 3).float().mean())}


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
    absolute_rotation = rotation_geodesic(prediction[..., :3, :3],
                                          target[..., :3, :3])
    return {"ate_metric": float(torch.sqrt(translation.square().mean())),
            "ate_median": float(translation.median()),
            "ate_rotation_deg": float(absolute_rotation.mean() * 180 / math.pi),
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
    errors, confidences, dynamic_pred, dynamic_true = [], [], [], []
    valid_count, total_count = 0, 0
    for index in range(pairs):
        points = output["track_source"][:, index]
        target = points + bilinear_sample(flow[:, index], points)
        mask4 = F.interpolate(valid[:, index:index + 1].float(), (h4, w4), mode="nearest")
        mask = bilinear_sample(mask4, points).squeeze(-1) > .5
        epe = torch.linalg.vector_norm(output["track_target"][:, index] - target, dim=-1)
        mask = mask & torch.isfinite(epe)
        valid_count += int(mask.sum()); total_count += mask.numel()
        if bool(mask.any()):
            errors.append(epe[mask]); confidences.append(output["track_confidence"][:, index][mask])
            if "dynamic_mask" in batch:
                dyn = F.interpolate(batch["dynamic_mask"][:, index:index + 1].float(),
                                    (h4, w4), mode="nearest")
                dynamic_true.append(bilinear_sample(dyn, points).squeeze(-1)[mask])
                dynamic_pred.append(output["dynamic_probability"][:, index][mask])
    if not errors:
        return {"track_epe": 1e3, "track_confidence": 0.0,
                "track_valid_fraction": 0.0, "track_pck1": 0.0,
                "track_pck3": 0.0, "track_outlier_rate": 1.0,
                "track_ece": 1.0, "track_brier": 1.0,
                "dynamic_precision": 0.0, "dynamic_recall": 0.0,
                "dynamic_f1": 0.0, "dynamic_iou": 0.0}
    error, confidence = torch.cat(errors), torch.cat(confidences)
    correct = (error < 3.0).float()
    result = {"track_epe": float(error.mean()),
              "track_epe_median": float(error.median()),
              "track_epe_p90": float(torch.quantile(error, .9)),
              "track_confidence": float(confidence.mean()),
              "track_valid_fraction": valid_count / max(total_count, 1),
              "track_pck1": float((error < 1.0).float().mean()),
              "track_pck3": float(correct.mean()),
              "track_outlier_rate": float((error > 5.0).float().mean()),
              "track_ece": expected_calibration_error(confidence.tolist(),
                                                       correct.tolist()),
              "track_brier": float((confidence - correct).square().mean())}
    if dynamic_true:
        truth = torch.cat(dynamic_true) > .5
        pred = torch.cat(dynamic_pred) > .5
        tp = (pred & truth).sum().float(); fp = (pred & ~truth).sum().float()
        fn = (~pred & truth).sum().float()
        precision = tp / (tp + fp).clamp_min(1)
        recall = tp / (tp + fn).clamp_min(1)
        result.update({"dynamic_precision": float(precision),
                       "dynamic_recall": float(recall),
                       "dynamic_f1": float(2 * precision * recall /
                                           (precision + recall).clamp_min(1e-8)),
                       "dynamic_iou": float(tp / (tp + fp + fn).clamp_min(1))})
    else:
        result.update({"dynamic_precision": 0.0, "dynamic_recall": 0.0,
                       "dynamic_f1": 0.0, "dynamic_iou": 0.0})
    return result


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
    ba = torch.nan_to_num(output["ba_residual_history"], nan=1e3,
                          posinf=1e3, neginf=1e3)
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
    keys = ("depth_abs_rel", "depth_rmse", "depth_log_rmse", "depth_silog",
            "depth_delta1", "depth_delta2", "depth_delta3", "ate_metric",
            "ate_median", "ate_rotation_deg", "rpe_translation",
            "rpe_rotation_deg", "track_epe", "track_epe_median",
            "track_epe_p90", "track_confidence", "track_valid_fraction",
            "track_pck1", "track_pck3", "track_outlier_rate", "track_ece",
            "track_brier", "dynamic_precision", "dynamic_recall", "dynamic_f1",
            "dynamic_iou", "ba_residual_reduction")
    accum = {name: {key: [] for key in keys}
             for name in ("normal", "black", "frozen", "reverse",
                          "wrong_window", "wrong_intrinsics")}
    source_accum = defaultdict(lambda: {key: [] for key in keys})
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
                if name == "normal":
                    labels = batch.get("source", ["unknown"])
                    source = str(labels[0] if isinstance(labels, (list, tuple)) else labels)
                    source_accum[source][key].append(metrics[key])
        prior = torch.full_like(batch["depth"], depth_prior_m)
        baseline_absrel.append(depth_metrics(prior, batch["depth"],
                                             batch["depth_valid"])["depth_abs_rel"])
        identity = torch.eye(4, device=device).view(1, 1, 4, 4).expand_as(batch["pose_c2w"])
        identity_ate.append(pose_metrics(identity, batch["pose_c2w"])["ate_metric"])
        windows += batch["image"].shape[0]
    controls = {name: {key: sum(values) / len(values) if values else math.inf
                       for key, values in metrics.items()}
                for name, metrics in accum.items()}
    per_source = {
        source: {key: sum(values) / len(values) if values else math.inf
                 for key, values in metrics.items()}
        for source, metrics in source_accum.items()
    }
    normal = controls["normal"]
    depth_baseline = sum(baseline_absrel) / max(len(baseline_absrel), 1)
    pose_baseline = sum(identity_ate) / max(len(identity_ate), 1)
    ratio = lambda numerator, denominator: numerator / max(denominator, 1e-8)
    ratios = {
        "depth_prior_gain": ratio(depth_baseline, normal["depth_abs_rel"]),
        "pose_identity_gain": ratio(pose_baseline, normal["ate_metric"]),
        "ba_residual_reduction": normal["ba_residual_reduction"],
        "track_valid_fraction": normal["track_valid_fraction"],
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
    gate_pass_rate = sum(gates.values()) / len(gates)
    # Bounded [0,1] checkpoint score. A harmonic mean makes a collapsed
    # capability dominate instead of being hidden by strong depth alone.
    probe_capability_score = harmonic_score((
        math.exp(-normal["depth_abs_rel"]),
        math.exp(-normal["ate_metric"] / .25),
        math.exp(-normal["track_epe_p90"] / 3.0),
        1.0 - min(1.0, normal["track_ece"]),
    ))
    capability_score = harmonic_score((
        probe_capability_score,
        max(0.0, min(1.0, normal["ba_residual_reduction"])),
        gate_pass_rate,
    ))
    def source_score(row: dict) -> float:
        return harmonic_score((math.exp(-row["depth_abs_rel"]),
                               math.exp(-row["ate_metric"] / .25),
                               math.exp(-row["track_epe_p90"] / 3.0),
                               1.0 - min(1.0, row["track_ece"])))
    source_scores = {source: source_score(row) for source, row in per_source.items()}
    worst_source_score = min(source_scores.values(), default=0.0)
    synthetic_scores = [score for source, score in source_scores.items()
                        if "synthetic" in source or "procedural" in source]
    real_scores = [score for source, score in source_scores.items()
                   if "synthetic" not in source and "procedural" not in source]
    sim_to_real_gap = (abs(sum(synthetic_scores) / len(synthetic_scores) -
                           sum(real_scores) / len(real_scores))
                       if synthetic_scores and real_scores else math.nan)
    return {"valid": all(gates.values()), "windows": windows,
            "controls": controls, "baselines": {
                "fixed_depth_abs_rel": depth_baseline, "identity_ate": pose_baseline},
            "ratios": ratios, "gates": gates, "per_source": per_source,
            "summary": {"capability_score": capability_score,
                        "probe_capability_score": probe_capability_score,
                        "causal_gate_pass_rate": gate_pass_rate,
                        "worst_source_score": worst_source_score,
                        "source_scores": source_scores,
                        "sim_to_real_gap": sim_to_real_gap,
                        "promotion_rule": "all causal gates AND no metric regression"}}


def controller_metrics(report: dict) -> dict[str, float]:
    """Exact metric dictionary consumed by ``AdaptiveTrainingBudget``."""
    return {name: float(report["ratios"][name]) for name in (
        "depth_prior_gain", "pose_identity_gain", "ba_residual_reduction",
        "track_valid_fraction",
        "wrong_window_depth_ratio", "wrong_window_pose_ratio",
        "reverse_time_rpe_ratio", "wrong_intrinsics_pose_ratio")}
