"""Evaluation metrics for the Eye Physical geometry pathway."""

from __future__ import annotations

import math

import torch

from .geometric_memory_v2 import masked_depth_pool
from .mathx import (anchor_depth_scale, relative_pose_c2w,
                    rotation_geodesic)


@torch.no_grad()
def geometry_batch_metrics(model, batch: dict) -> dict[str, float]:
    output = model.encode_geometry_sequence(batch["image"], detach_state=True)
    target = batch["depth"]
    valid = batch.get("depth_valid", torch.isfinite(target) & (target > 1e-6))
    b, t = target.shape[:2]
    if getattr(model.cfg, "geometry_version", "v1") == "v2_pairwise":
        target_tokens, valid_weight = masked_depth_pool(
            target, valid, model.cfg.img_grid_h, model.cfg.img_grid_w)
        mask = valid_weight >= model.cfg.geometry_min_valid_fraction
        weight = valid_weight * mask.float()
        error = output["depth_tokens"] - target_tokens
        denom = weight.sum().clamp(min=1.0)
        abs_rel = ((error.abs() / target_tokens.clamp(min=1e-4)) *
                   weight).sum() / denom
        rmse = torch.sqrt(((error ** 2) * weight).sum() / denom)
        ratio = torch.maximum(output["depth_tokens"] / target_tokens.clamp(min=1e-4),
                              target_tokens / output["depth_tokens"].clamp(min=1e-4))
        delta1 = ((ratio < 1.25).float() * weight).sum() / denom

        gt = torch.linalg.inv(batch["pose_c2w"][:, :1]) @ batch["pose_c2w"]
        pred = output["pose_c2w"]
        ate = torch.sqrt(((pred[..., :3, 3] - gt[..., :3, 3]) ** 2)
                         .sum(-1).mean())
        abs_rotation = rotation_geodesic(pred[..., :3, :3],
                                         gt[..., :3, :3]).mean()
        pred_rel = relative_pose_c2w(pred[:, :-1], pred[:, 1:])
        gt_rel = relative_pose_c2w(gt[:, :-1], gt[:, 1:])
        rpe_translation = (pred_rel[..., :3, 3] -
                           gt_rel[..., :3, 3]).norm(dim=-1).mean()
        rpe_rotation = rotation_geodesic(pred_rel[..., :3, :3],
                                         gt_rel[..., :3, :3]).mean()
        return {"depth_abs_rel": float(abs_rel),
                "depth_rmse_metric": float(rmse),
                "depth_delta1": float(delta1),
                "ate_metric": float(ate),
                "abs_rotation_deg": float(abs_rotation * 180 / math.pi),
                "rpe_translation_metric": float(rpe_translation),
                "rpe_rotation_deg": float(rpe_rotation * 180 / math.pi)}

    target_tokens = torch.nn.functional.adaptive_avg_pool2d(
        target.reshape(b * t, 1, *target.shape[-2:]),
        (model.cfg.img_grid_h, model.cfg.img_grid_w)).reshape_as(output["depth_tokens"])
    valid_tokens = torch.nn.functional.adaptive_avg_pool2d(
        valid.float().reshape(b * t, 1, *valid.shape[-2:]),
        (model.cfg.img_grid_h, model.cfg.img_grid_w)).reshape_as(target_tokens) > 0.99
    scale = anchor_depth_scale(target_tokens, valid_tokens,
                               model.cfg.geometry_anchor_frames)
    target_tokens = target_tokens / scale.view(b, 1, 1)
    mask = valid_tokens & torch.isfinite(target_tokens) & (target_tokens > 1e-6)
    error = output["depth_tokens"] - target_tokens
    abs_rel = (error.abs() / target_tokens.clamp(min=1e-4))[mask].mean()
    rmse = torch.sqrt((error[mask] ** 2).mean())

    gt = batch["pose_c2w"].clone()
    gt[..., :3, 3] /= scale.view(b, 1, 1)
    pred = output["pose_c2w"]
    ate = torch.sqrt(((pred[..., :3, 3] - gt[..., :3, 3]) ** 2).sum(-1).mean())
    abs_rotation = rotation_geodesic(pred[..., :3, :3], gt[..., :3, :3]).mean()
    pred_rel = relative_pose_c2w(pred[:, :-1], pred[:, 1:])
    gt_rel = relative_pose_c2w(gt[:, :-1], gt[:, 1:])
    rpe_translation = (pred_rel[..., :3, 3] - gt_rel[..., :3, 3]).norm(dim=-1).mean()
    rpe_rotation = rotation_geodesic(pred_rel[..., :3, :3],
                                     gt_rel[..., :3, :3]).mean()
    return {"depth_abs_rel": float(abs_rel), "depth_rmse_anchor_scale": float(rmse),
            "ate_anchor_scale": float(ate),
            "abs_rotation_deg": float(abs_rotation * 180 / math.pi),
            "rpe_translation": float(rpe_translation),
            "rpe_rotation_deg": float(rpe_rotation * 180 / math.pi)}


@torch.no_grad()
def evaluate_geometry(model, batches, device, max_batches: int = 16):
    model.eval()
    collected = []
    for i, batch in enumerate(batches):
        if i >= max_batches:
            break
        moved = {k: v.to(device) if torch.is_tensor(v) else v
                 for k, v in batch.items()}
        collected.append(geometry_batch_metrics(model, moved))
    model.train()
    if not collected:
        return {}
    return {key: sum(row[key] for row in collected) / len(collected)
            for key in collected[0]}


@torch.no_grad()
def constant_identity_metrics_v2(batch: dict, cfg,
                                 constant_depth_m: float = 2.5) -> dict[str, float]:
    """Fixed, target-independent prior used as a mandatory promotion baseline."""
    target, valid_weight = masked_depth_pool(
        batch["depth"], batch.get("depth_valid"), cfg.img_grid_h, cfg.img_grid_w)
    predicted = torch.full_like(target, constant_depth_m)
    mask = valid_weight >= cfg.geometry_min_valid_fraction
    weight = valid_weight * mask.float()
    denom = weight.sum().clamp(min=1.0)
    error = predicted - target
    abs_rel = ((error.abs() / target.clamp(min=1e-4)) * weight).sum() / denom
    rmse = torch.sqrt(((error ** 2) * weight).sum() / denom)
    ratio = torch.maximum(predicted / target.clamp(min=1e-4),
                          target / predicted.clamp(min=1e-4))
    delta1 = ((ratio < 1.25).float() * weight).sum() / denom

    gt = torch.linalg.inv(batch["pose_c2w"][:, :1]) @ batch["pose_c2w"]
    pred = torch.eye(4, device=gt.device, dtype=gt.dtype).view(1, 1, 4, 4)
    pred = pred.expand_as(gt)
    ate = torch.sqrt(((pred[..., :3, 3] - gt[..., :3, 3]) ** 2)
                     .sum(-1).mean())
    abs_rotation = rotation_geodesic(pred[..., :3, :3], gt[..., :3, :3]).mean()
    pred_rel = relative_pose_c2w(pred[:, :-1], pred[:, 1:])
    gt_rel = relative_pose_c2w(gt[:, :-1], gt[:, 1:])
    rpe_translation = (pred_rel[..., :3, 3] -
                       gt_rel[..., :3, 3]).norm(dim=-1).mean()
    rpe_rotation = rotation_geodesic(pred_rel[..., :3, :3],
                                     gt_rel[..., :3, :3]).mean()
    return {"depth_abs_rel": float(abs_rel), "depth_rmse_metric": float(rmse),
            "depth_delta1": float(delta1), "ate_metric": float(ate),
            "abs_rotation_deg": float(abs_rotation * 180 / math.pi),
            "rpe_translation_metric": float(rpe_translation),
            "rpe_rotation_deg": float(rpe_rotation * 180 / math.pi)}


def _mean_metrics(rows: list[dict[str, float]]) -> dict[str, float]:
    return {key: sum(row[key] for row in rows) / len(rows)
            for key in rows[0]}


@torch.no_grad()
def evaluate_geometry_controls(model, batches, device,
                               max_batches: int = 16,
                               constant_depth_m: float = 2.5) -> dict:
    """Correct/black/reversed/wrong-window plus fixed-prior causal audit."""
    rows = []
    for index, batch in enumerate(batches):
        if index >= max_batches:
            break
        rows.append({key: value.clone() if torch.is_tensor(value) else value
                     for key, value in batch.items()})
    if not rows:
        return {"valid": False, "reason": "no evaluation batches"}
    model.eval()
    controls = {}
    for name in ("normal", "black", "reverse_time", "wrong_window"):
        measured = []
        for index, row in enumerate(rows):
            batch = {key: value.to(device) if torch.is_tensor(value) else value
                     for key, value in row.items()}
            if name == "black":
                batch["image"] = torch.zeros_like(batch["image"])
            elif name == "reverse_time":
                batch["image"] = batch["image"].flip(1)
            elif name == "wrong_window":
                wrong = rows[(index + max(1, len(rows) // 2)) % len(rows)]["image"]
                batch["image"] = wrong.to(device)
            measured.append(geometry_batch_metrics(model, batch))
        controls[name] = _mean_metrics(measured)
    priors = []
    for row in rows:
        batch = {key: value.to(device) if torch.is_tensor(value) else value
                 for key, value in row.items()}
        priors.append(constant_identity_metrics_v2(
            batch, model.cfg, constant_depth_m=constant_depth_m))
    controls["constant_identity_prior"] = _mean_metrics(priors)
    model.train()

    normal, black = controls["normal"], controls["black"]
    wrong, reverse = controls["wrong_window"], controls["reverse_time"]
    prior = controls["constant_identity_prior"]
    ratios = {
        "depth_prior_over_model": prior["depth_abs_rel"] /
            max(normal["depth_abs_rel"], 1e-9),
        "ate_prior_over_model": prior["ate_metric"] /
            max(normal["ate_metric"], 1e-9),
        "black_depth_over_normal": black["depth_abs_rel"] /
            max(normal["depth_abs_rel"], 1e-9),
        "wrong_depth_over_normal": wrong["depth_abs_rel"] /
            max(normal["depth_abs_rel"], 1e-9),
        "wrong_ate_over_normal": wrong["ate_metric"] /
            max(normal["ate_metric"], 1e-9),
        "reverse_rpe_translation_over_normal": reverse["rpe_translation_metric"] /
            max(normal["rpe_translation_metric"], 1e-9),
    }
    gates = {
        "G_depth_beats_fixed_prior_20pct": ratios["depth_prior_over_model"] >= 1.20,
        "G_pose_beats_identity_prior_20pct": ratios["ate_prior_over_model"] >= 1.20,
        "G_black_image_hurts_depth_25pct": ratios["black_depth_over_normal"] >= 1.25,
        "G_wrong_window_hurts_depth_25pct": ratios["wrong_depth_over_normal"] >= 1.25,
        "G_wrong_window_hurts_pose_25pct": ratios["wrong_ate_over_normal"] >= 1.25,
        "G_reverse_time_hurts_motion_10pct":
            ratios["reverse_rpe_translation_over_normal"] >= 1.10,
    }
    return {"valid": all(gates.values()),
            "constant_depth_prior_m": float(constant_depth_m),
            "controls": controls, "ratios": ratios, "gates": gates}
