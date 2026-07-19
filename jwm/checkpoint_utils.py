"""Selective warm-start utilities for architecture-corrective training."""

from __future__ import annotations

from pathlib import Path

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
