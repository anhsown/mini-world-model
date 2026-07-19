"""Evaluation metrics for the Eye Physical geometry pathway."""

from __future__ import annotations

import math

import torch

from .mathx import (anchor_depth_scale, relative_pose_c2w,
                    rotation_geodesic)


@torch.no_grad()
def geometry_batch_metrics(model, batch: dict) -> dict[str, float]:
    output = model.encode_geometry_sequence(batch["image"], detach_state=True)
    target = batch["depth"]
    valid = batch.get("depth_valid", torch.isfinite(target) & (target > 1e-6))
    b, t = target.shape[:2]
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

