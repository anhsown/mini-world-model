"""Selective warm-start utilities for architecture-corrective training."""

from __future__ import annotations

from pathlib import Path
import re

import torch


READER_REINITIALIZE_PREFIXES = (
    "ocr_head.", "box_queries", "box_attn.", "coord_head.", "reader_roi.",
)


def warmstart_reader_v31(model, checkpoint: str | Path) -> dict:
    """Reuse v3 vision/reasoner weights and reinitialize failed OCR heads.

    Shape mismatches and generator-only extras are reported rather than hidden.
    The function accepts training or deploy checkpoints (`model` or
    `state_dict`) and never loads optimizer/scaler state from the failed run.
    """
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    source = payload.get("model", payload.get("state_dict", payload))
    target = model.state_dict()
    admitted, skipped = {}, {}
    for name, tensor in source.items():
        if any(name.startswith(prefix) for prefix in READER_REINITIALIZE_PREFIXES):
            skipped[name] = "corrective_head_reset"
        elif name not in target:
            skipped[name] = "not_in_target"
        elif target[name].shape != tensor.shape:
            skipped[name] = f"shape:{tuple(tensor.shape)}->{tuple(target[name].shape)}"
        else:
            admitted[name] = tensor
    result = model.load_state_dict(admitted, strict=False)
    return {
        "loaded_tensors": len(admitted),
        "reinitialized_tensors": sum(v == "corrective_head_reset"
                                     for v in skipped.values()),
        "skipped": skipped,
        "missing_after_load": list(result.missing_keys),
        "unexpected_after_load": list(result.unexpected_keys),
    }


def warmstart_eye_physical(model, checkpoint: str | Path) -> dict:
    """Load all shape-compatible JWM-v4 semantics; initialize the new eye fresh."""
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    source = payload.get("model", payload.get("state_dict", payload))
    target = model.state_dict()
    fresh_prefixes = ("patch_embed.", "vision_stem.", "geometry.",
                      "ocr_head.", "reader_roi.", "box_", "coord_head.")
    admitted, skipped = {}, {}
    for name, tensor in source.items():
        if name.startswith(fresh_prefixes):
            skipped[name] = "new_eye_reset"
        elif name not in target:
            skipped[name] = "not_in_target"
        elif target[name].shape != tensor.shape:
            skipped[name] = f"shape:{tuple(tensor.shape)}->{tuple(target[name].shape)}"
        else:
            admitted[name] = tensor
    result = model.load_state_dict(admitted, strict=False)
    return {
        "loaded_tensors": len(admitted), "skipped": skipped,
        "missing_after_load": list(result.missing_keys),
        "unexpected_after_load": list(result.unexpected_keys),
    }


def _overlap_copy(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor | None:
    """Copy the maximal common hyper-rectangle without inventing values."""
    if source.ndim != target.ndim:
        return None
    result = target.clone()
    slices = tuple(slice(0, min(a, b)) for a, b in zip(source.shape, target.shape))
    result[slices] = source[slices].to(dtype=result.dtype)
    return result


def warmstart_eye_v32(model, checkpoint: str | Path) -> dict:
    """Function-preserving-ish expansion from the 384x8 JWM-v4 backbone.

    Source blocks are placed at even target depths; inserted odd blocks are
    explicit residual identities.  Width-expanded tensors retain their common
    subspace while new channels keep the target initializer.  The physical eye
    is always reset because it is the corrective architecture being trained.
    """
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    source = payload.get("model", payload.get("state_dict", payload))
    target = model.state_dict()
    fresh_prefixes = ("patch_embed.", "vision_stem.", "geometry.",
                      "ocr_head.", "reader_roi.", "box_", "coord_head.")
    source_layers = 1 + max((int(match.group(1)) for name in source
                             if (match := re.match(r"blocks\.(\d+)\.", name))),
                            default=-1)
    target_layers = len(model.blocks)
    admitted, skipped, expanded = {}, {}, 0
    for target_name, target_tensor in target.items():
        if target_name.startswith(fresh_prefixes):
            skipped[target_name] = "new_eye_reset"; continue
        source_name = target_name
        match = re.match(r"blocks\.(\d+)\.(.*)", target_name)
        if match:
            target_index = int(match.group(1))
            # Interleave source layers with identity layers when depth doubles.
            if target_layers >= 2 * source_layers and target_index % 2:
                skipped[target_name] = "inserted_identity_layer"; continue
            source_index = (target_index // 2 if target_layers >= 2 * source_layers
                            else min(source_layers - 1,
                                     target_index * source_layers // target_layers))
            source_name = f"blocks.{source_index}.{match.group(2)}"
        source_tensor = source.get(source_name)
        if source_tensor is None:
            skipped[target_name] = "not_in_source"; continue
        if source_tensor.shape == target_tensor.shape:
            admitted[target_name] = source_tensor
        else:
            copied = _overlap_copy(source_tensor, target_tensor)
            if copied is None:
                skipped[target_name] = (f"rank:{source_tensor.ndim}->{target_tensor.ndim}")
            else:
                admitted[target_name] = copied; expanded += 1
    result = model.load_state_dict(admitted, strict=False)
    # A residual block is exactly identity when its output projections are 0.
    inserted_identity_layers = (list(range(1, target_layers, 2))
                                if target_layers >= 2 * source_layers else [])
    if inserted_identity_layers:
        for index in inserted_identity_layers:
            block = model.blocks[index]
            torch.nn.init.zeros_(block.r_attn.wo.weight)
            torch.nn.init.zeros_(block.g_attn.wo.weight)
            torch.nn.init.zeros_(block.g_ffn.w_down.weight)
            if hasattr(block.r_ffn, "w_down"):
                torch.nn.init.zeros_(block.r_ffn.w_down)
                torch.nn.init.zeros_(block.r_ffn.shared_down.weight)
            else:
                torch.nn.init.zeros_(block.r_ffn.w_down.weight)
    if hasattr(model.geometry, "reset_safe_initialization"):
        model.geometry.reset_safe_initialization()
    return {
        "source_layers": source_layers, "target_layers": target_layers,
        "loaded_tensors": len(admitted), "expanded_tensors": expanded,
        "identity_inserted_layers": inserted_identity_layers,
        "skipped": skipped, "missing_after_load": list(result.missing_keys),
        "unexpected_after_load": list(result.unexpected_keys),
    }


EYE_V322_CORRECTIVE_RESET_PREFIXES = (
    "geometry.depth.", "geometry.depth_film.", "geometry.pose_head.",
    "geometry.temporal_compatibility.", "geometry.tracker.dynamic.",
    "geometry.ray_head.",
)


def warmstart_eye_v322(model, checkpoint: str | Path) -> dict:
    """Reuse v3.2.1's valid tracker while resetting its collapsed heads.

    A generic JWM-v4 checkpoint still follows the ordinary v3.2 expansion.
    A blocked v3.2.1 checkpoint is admitted only selectively: visual features,
    scene registers, tracker coordinates, confidence, visibility, and scale are
    retained; depth, pose, temporal, dynamic, and ray-calibration heads restart.
    Optimizer/controller state is never inherited.
    """
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if payload.get("version") != "jwm-eye-v3.2.1-robust-causal-geometry":
        return warmstart_eye_v32(model, checkpoint)
    source = payload.get("model", payload.get("state_dict", payload))
    target = model.state_dict()
    admitted, skipped = {}, {}
    for name, tensor in source.items():
        if any(name.startswith(prefix)
               for prefix in EYE_V322_CORRECTIVE_RESET_PREFIXES):
            skipped[name] = "corrective_head_reset"
        elif name not in target:
            skipped[name] = "not_in_target"
        elif target[name].shape != tensor.shape:
            skipped[name] = f"shape:{tuple(tensor.shape)}->{tuple(target[name].shape)}"
        else:
            admitted[name] = tensor
    result = model.load_state_dict(admitted, strict=False)
    return {
        "source_version": payload.get("version"),
        "loaded_tensors": len(admitted),
        "corrective_reset_tensors": sum(
            value == "corrective_head_reset" for value in skipped.values()),
        "retained_tracker_tensors": sum(
            name.startswith("geometry.tracker.") and
            not any(name.startswith(prefix)
                    for prefix in EYE_V322_CORRECTIVE_RESET_PREFIXES)
            for name in admitted),
        "skipped": skipped,
        "missing_after_load": list(result.missing_keys),
        "unexpected_after_load": list(result.unexpected_keys),
    }
