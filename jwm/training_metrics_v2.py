"""Capability metrics for JWM training and checkpoint promotion.

Loss answers whether the optimizer moved. These metrics answer whether the
model learned vision, geometry, calibration and causal dependence. Every
metric has an explicit direction and scope so dashboards cannot accidentally
treat a decreasing error as an increasing score.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Iterable


@dataclass(frozen=True)
class MetricDefinition:
    name: str
    direction: str
    scope: str
    unit: str
    purpose: str
    promotion: bool = False


EYE_METRICS_V2 = (
    MetricDefinition("loss_total", "down", "train", "loss", "Optimization health only"),
    MetricDefinition("grad_norm", "bounded", "train", "L2", "Detect exploding or dead gradients"),
    MetricDefinition("nonfinite_skip_rate", "down", "train", "fraction", "Numerical stability", True),
    MetricDefinition("depth_abs_rel", "down", "real_heldout", "ratio", "Metric depth relative error", True),
    MetricDefinition("depth_silog", "down", "real_heldout", "log-RMSE", "Scale-aware depth structure", True),
    MetricDefinition("depth_delta1", "up", "real_heldout", "fraction", "Depth within x1.25", True),
    MetricDefinition("ate_metric", "down", "real_heldout", "metres", "Absolute camera trajectory error", True),
    MetricDefinition("rpe_translation", "down", "real_heldout", "metres/frame", "Local translation drift", True),
    MetricDefinition("rpe_rotation_deg", "down", "real_heldout", "degrees/frame", "Local rotation drift", True),
    MetricDefinition("track_epe", "down", "real_heldout", "feature-pixels", "Unbounded mean tracking diagnostic"),
    MetricDefinition("track_epe_p90", "down", "real_heldout", "feature-pixels", "Robust tail tracking error", True),
    MetricDefinition("track_outlier_rate", "down", "real_heldout", "fraction", "Catastrophic track failures", True),
    MetricDefinition("track_pck3", "up", "real_heldout", "fraction", "Tracks within 3 feature pixels", True),
    MetricDefinition("track_ece", "down", "real_heldout", "fraction", "Track confidence calibration", True),
    MetricDefinition("dynamic_f1", "up", "real_heldout", "fraction", "Dynamic/static separation", True),
    MetricDefinition("ba_residual_reduction", "up", "real_heldout", "fraction", "Bundle-adjustment usefulness", True),
    MetricDefinition("causal_gate_pass_rate", "up", "causal", "fraction", "Dependence on image/time/K", True),
    MetricDefinition("worst_source_score", "up", "ood", "normalized", "No source hidden by an average", True),
    MetricDefinition("sim_to_real_gap", "down", "ood", "normalized", "Synthetic/real generalization gap", True),
    MetricDefinition("probe_capability_score", "up", "real_heldout", "harmonic", "A/B depth-pose-track-calibration score", True),
)

READER_METRICS_V2 = (
    MetricDefinition("qa_token_accuracy", "up", "train", "fraction", "Teacher-forced optimization diagnostic"),
    MetricDefinition("cer", "down", "real_heldout", "edits/character", "Free-running OCR error", True),
    MetricDefinition("anls", "up", "real_heldout", "fraction", "Answer similarity robust to small OCR errors", True),
    MetricDefinition("exact_match", "up", "real_heldout", "fraction", "Whole-answer correctness", True),
    MetricDefinition("box_iou", "up", "real_heldout", "fraction", "Text-region grounding", True),
    MetricDefinition("vision_gain", "up", "causal", "delta", "Real image versus blind-image control", True),
    MetricDefinition("worst_kind_cer", "down", "ood", "edits/character", "Prevent easy OCR kinds hiding failures", True),
)

REASONER_METRICS_V2 = (
    MetricDefinition("free_run_exact", "up", "real_heldout", "fraction", "End-to-end answer correctness", True),
    MetricDefinition("token_accuracy", "up", "train", "fraction", "Teacher-forced diagnostic only"),
    MetricDefinition("grounding_miou", "up", "real_heldout", "IoU", "Spatial grounding quality", True),
    MetricDefinition("grounding_iou50", "up", "real_heldout", "fraction", "Usable localized answers", True),
    MetricDefinition("answer_ece", "down", "real_heldout", "fraction", "Confidence calibration", True),
    MetricDefinition("vision_gain", "up", "causal", "delta", "Image dependence over blind control", True),
    MetricDefinition("worst_capability_score", "up", "ood", "normalized", "No QA kind hidden by average", True),
)

GENERATOR_METRICS_V2 = (
    MetricDefinition("flow_matching_loss", "down", "train", "loss", "Optimization diagnostic only"),
    MetricDefinition("t2i_prompt_accuracy", "up", "real_heldout", "fraction", "Prompt content coverage", True),
    MetricDefinition("t2i_negative_rejection", "up", "real_heldout", "fraction", "Avoid forbidden alternatives", True),
    MetricDefinition("fd_psnr", "up", "real_heldout", "dB", "Forward video reconstruction", True),
    MetricDefinition("fd_copy_margin", "up", "real_heldout", "dB", "Must beat last-frame copy", True),
    MetricDefinition("temporal_warp_error", "down", "real_heldout", "pixels", "Motion consistency", True),
    MetricDefinition("action_mse", "down", "real_heldout", "normalized", "Inverse/action prediction", True),
    MetricDefinition("policy_success", "up", "closed_loop", "fraction", "Task completion", True),
    MetricDefinition("video_action_consistency", "up", "closed_loop", "PSNR", "Predicted action agrees with future video", True),
)

DATA_HEALTH_METRICS_V2 = (
    MetricDefinition("scene_leak_count", "down", "data", "count", "Train/eval independence", True),
    MetricDefinition("duplicate_rate", "down", "data", "fraction", "Effective sample diversity", True),
    MetricDefinition("label_valid_fraction", "up", "data", "fraction", "Usable exact supervision", True),
    MetricDefinition("source_balance_entropy", "up", "data", "normalized", "Mixture coverage"),
    MetricDefinition("real_synthetic_domain_gap", "down", "data", "normalized", "Sim-to-real admission", True),
)


def metric_catalog() -> dict:
    return {"eye": [asdict(metric) for metric in EYE_METRICS_V2],
            "reader": [asdict(metric) for metric in READER_METRICS_V2],
            "reasoner": [asdict(metric) for metric in REASONER_METRICS_V2],
            "generator": [asdict(metric) for metric in GENERATOR_METRICS_V2],
            "data_health": [asdict(metric) for metric in DATA_HEALTH_METRICS_V2]}


def expected_calibration_error(confidence: Iterable[float],
                               correct: Iterable[float], bins: int = 10) -> float:
    pairs = [(float(c), float(y)) for c, y in zip(confidence, correct)
             if math.isfinite(float(c)) and math.isfinite(float(y))]
    if not pairs:
        return 1.0
    total, score = len(pairs), 0.0
    for index in range(bins):
        lower, upper = index / bins, (index + 1) / bins
        selected = [(c, y) for c, y in pairs
                    if ((lower <= c <= upper) if index == bins - 1
                        else (lower <= c < upper))]
        if selected:
            mean_c = sum(c for c, _ in selected) / len(selected)
            mean_y = sum(y for _, y in selected) / len(selected)
            score += len(selected) / total * abs(mean_c - mean_y)
    return score


def harmonic_score(values: Iterable[float], floor: float = 1e-6) -> float:
    finite = [max(floor, min(1.0, float(value))) for value in values
              if math.isfinite(float(value))]
    return len(finite) / sum(1 / value for value in finite) if finite else 0.0
