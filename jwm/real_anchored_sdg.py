"""Real-anchored synthetic geometry for JWM Eye.

Synthetic labels remain analytic. Real data controls only plausible nuisance
distributions (appearance, FoV, metric scale and motion magnitude), avoiding
teacher hallucinations and geometry-breaking image transformations.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from .geometry_data import render_geometry_sequence
from .geometry_math_v3 import camera_transform
from .geometry_v3_data import standardize_geometry_row, validate_geometry_v3_source
from .mathx import rotation_geodesic


@dataclass(frozen=True)
class RealAnchorProfile:
    image_mean: float = 0.50
    image_std: float = 0.22
    gradient_mean: float = 0.08
    saturation_fraction: float = 0.02
    depth_median_m: float = 3.0
    depth_q10_m: float = 0.7
    depth_q90_m: float = 8.0
    depth_valid_fraction: float = 0.80
    fov_q10_deg: float = 55.0
    fov_median_deg: float = 67.0
    fov_q90_deg: float = 90.0
    translation_median_m: float = 0.04
    rotation_median_deg: float = 1.0
    fps_median: float = 10.0
    windows_profiled: int = 0

    @classmethod
    def from_json(cls, path: str | Path) -> "RealAnchorProfile":
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))

    def save(self, path: str | Path) -> None:
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")


def _quantile(values: list[float], q: float, fallback: float) -> float:
    if not values:
        return fallback
    tensor = torch.tensor(values, dtype=torch.float64)
    return float(torch.quantile(tensor, q))


def profile_real_geometry(datasets: Iterable[Dataset], max_windows: int = 128) -> RealAnchorProfile:
    """Fit robust scalar distributions without retaining user/source images."""
    datasets = list(datasets)
    means, stds, gradients, saturations = [], [], [], []
    depth_values, valid_fracs, fovs = [], [], []
    translations, rotations, fps = [], [], []
    count = 0
    for dataset in datasets:
        if len(dataset) == 0:
            continue
        take = min(len(dataset), max(1, max_windows // max(1, len(datasets))))
        indices = torch.linspace(0, len(dataset) - 1, steps=take).round().long().unique()
        for index in indices.tolist():
            row = dataset[index]
            image = row["image"].float().clamp(0, 1)
            means.append(float(image.mean())); stds.append(float(image.std()))
            dx = (image[..., 1:] - image[..., :-1]).abs().mean()
            dy = (image[..., 1:, :] - image[..., :-1, :]).abs().mean()
            gradients.append(float((dx + dy) * .5))
            saturations.append(float(((image < .01) | (image > .99)).float().mean()))
            depth = row["depth"].float()
            valid = row.get("depth_valid", torch.isfinite(depth) & (depth > 0))
            selected = depth[valid & torch.isfinite(depth) & (depth > 0)]
            if selected.numel():
                # Fixed subsampling prevents large RGB-D pages dominating RAM/time.
                selected = selected.flatten()[::max(1, selected.numel() // 4096)]
                depth_values.extend(selected.tolist())
            valid_fracs.append(float(valid.float().mean()))
            k = torch.as_tensor(row["intrinsics"])
            if k.ndim == 3: k = k[0]
            width = image.shape[-1]
            fovs.append(math.degrees(2 * math.atan(width / (2 * float(k[0, 0])))))
            pose = row["pose_c2w"].float()
            if pose.shape[0] > 1:
                rel = torch.stack([camera_transform(pose[i], pose[i + 1])
                                   for i in range(pose.shape[0] - 1)])
                translations.extend(torch.linalg.vector_norm(rel[:, :3, 3], dim=-1).tolist())
                rotations.extend((rotation_geodesic(
                    rel[:, :3, :3], torch.eye(3).expand_as(rel[:, :3, :3])) *
                    180 / math.pi).tolist())
            timestamp = row.get("timestamp")
            if timestamp is not None and len(timestamp) > 1:
                delta = torch.as_tensor(timestamp, dtype=torch.float64).diff()
                delta = delta[torch.isfinite(delta) & (delta > 0)]
                if delta.numel(): fps.append(float(1 / delta.median()))
            count += 1
            if count >= max_windows:
                break
        if count >= max_windows:
            break
    if not count:
        return RealAnchorProfile()
    return RealAnchorProfile(
        image_mean=_quantile(means, .5, .5), image_std=_quantile(stds, .5, .22),
        gradient_mean=_quantile(gradients, .5, .08),
        saturation_fraction=_quantile(saturations, .5, .02),
        depth_median_m=_quantile(depth_values, .5, 3.0),
        depth_q10_m=_quantile(depth_values, .1, .7),
        depth_q90_m=_quantile(depth_values, .9, 8.0),
        depth_valid_fraction=_quantile(valid_fracs, .5, .8),
        fov_q10_deg=_quantile(fovs, .1, 55.0),
        fov_median_deg=_quantile(fovs, .5, 67.0),
        fov_q90_deg=_quantile(fovs, .9, 90.0),
        translation_median_m=_quantile(translations, .5, .04),
        rotation_median_deg=_quantile(rotations, .5, 1.0),
        fps_median=_quantile(fps, .5, 10.0), windows_profiled=count)


def _sample_uniform(generator: torch.Generator, low: float, high: float) -> float:
    return low + (high - low) * float(torch.rand((), generator=generator))


def _real_anchor_photometric(image: torch.Tensor, profile: RealAnchorProfile,
                             generator: torch.Generator) -> torch.Tensor:
    """Appearance-only augmentation; never modifies depth/pose/flow labels."""
    contrast = _sample_uniform(generator, .72, 1.30)
    target_mean = _sample_uniform(generator,
                                  max(.08, profile.image_mean - .12),
                                  min(.92, profile.image_mean + .12))
    gamma = _sample_uniform(generator, .72, 1.38)
    white_balance = torch.empty(3).uniform_(.88, 1.12, generator=generator)
    out = (image - image.mean(dim=(-2, -1), keepdim=True)) * contrast + target_mean
    out = out.clamp(0, 1).pow(gamma) * white_balance.view(1, 3, 1, 1)
    noise_std = _sample_uniform(generator, .002, max(.004, profile.image_std * .06))
    out = out + torch.randn(out.shape, generator=generator) * noise_std
    if float(torch.rand((), generator=generator)) < .35:
        out = F.avg_pool2d(out, 3, stride=1, padding=1)
    # Match the robust real contrast statistic after all nonlinear transforms.
    # This changes appearance only; geometric labels remain untouched.
    target_std = _sample_uniform(generator, max(.05, profile.image_std * .72),
                                 min(.38, profile.image_std * 1.12))
    out = ((out - out.mean()) / out.std().clamp_min(1e-4) * target_std + target_mean)
    return out.clamp(0, 1)


class RealAnchoredSyntheticGeometry(Dataset):
    """Deterministic on-the-fly corpus; a million samples cost only the profile."""

    OFFSETS = {"train": 10_000_000, "validation": 20_000_000,
               "val": 20_000_000, "test": 30_000_000}

    def __init__(self, split: str, samples: int, profile: RealAnchorProfile,
                 frames: int = 6, size: int = 128):
        if split not in self.OFFSETS:
            raise ValueError(f"unknown split: {split}")
        self.split, self.samples, self.profile = split, int(samples), profile
        self.frames, self.size = int(frames), int(size)

    def __len__(self):
        return self.samples

    def scene_ids(self):
        return {f"rasdg-{self.split}-{i}" for i in range(self.samples)}

    def __getitem__(self, index):
        if not 0 <= index < self.samples:
            raise IndexError(index)
        seed = self.OFFSETS[self.split] + index
        generator = torch.Generator().manual_seed(seed)
        fov = _sample_uniform(generator, self.profile.fov_q10_deg,
                              self.profile.fov_q90_deg)
        # Base renderer median translation is about 7 cm/frame.
        motion = min(4.0, max(.15, self.profile.translation_median_m / .07))
        motion *= _sample_uniform(generator, .65, 1.45)
        sequence = render_geometry_sequence(seed, self.frames, self.size, self.size,
                                            fov_degrees=fov, motion_scale=motion)
        row = sequence.as_dict()
        # A global metric scaling is a valid scene similarity transform: pixel
        # correspondences stay unchanged while depth and translation remain exact.
        base_depth = float(row["depth"].median())
        scale = min(3.0, max(.25, self.profile.depth_median_m / max(base_depth, .1)))
        scale *= _sample_uniform(generator, .80, 1.25)
        row["depth"] = row["depth"] * scale
        row["pose_c2w"] = row["pose_c2w"].clone()
        row["pose_c2w"][:, :3, 3] *= scale
        row["image"] = _real_anchor_photometric(row["image"], self.profile, generator)
        row["source"] = "synthetic_real_anchored"
        row["timestamp"] = torch.arange(self.frames, dtype=torch.float64) / max(
            self.profile.fps_median, 1.0)
        row["anchor_profile_version"] = 1
        return standardize_geometry_row(row, strict_camera=True)


def validate_real_anchored_synthetic(dataset: Dataset,
                                     profile: RealAnchorProfile,
                                     windows: int = 12) -> dict:
    rows = [dataset[i] for i in range(min(windows, len(dataset)))]
    image_mean = sum(float(row["image"].mean()) for row in rows) / max(len(rows), 1)
    image_std = sum(float(row["image"].std()) for row in rows) / max(len(rows), 1)
    gradient = sum(float(((row["image"][..., 1:] - row["image"][..., :-1]).abs().mean() +
                         (row["image"][..., 1:, :] - row["image"][..., :-1, :]).abs().mean()) * .5)
                   for row in rows) / max(len(rows), 1)
    depth_median = sum(float(row["depth"][row["depth_valid"]].median())
                       for row in rows) / max(len(rows), 1)
    reprojection_finite = all(bool(torch.isfinite(row["rigid_flow"]).all()) for row in rows)
    split_ids = dataset.scene_ids() if hasattr(dataset, "scene_ids") else set()
    mean_gap = abs(image_mean - profile.image_mean)
    std_ratio = max(image_std, profile.image_std) / max(min(image_std, profile.image_std), 1e-5)
    gradient_ratio = max(gradient, profile.gradient_mean) / max(
        min(gradient, profile.gradient_mean), 1e-5)
    depth_ratio = max(depth_median, profile.depth_median_m) / max(
        min(depth_median, profile.depth_median_m), 1e-5)
    geometry = validate_geometry_v3_source(dataset, "synthetic_real_anchored",
                                           max_windows=min(8, windows))
    hypotheses = {
        "H_samples_available": bool(rows),
        "H_exact_labels_finite": reprojection_finite,
        "H_exact_geometry_contract": geometry["valid"],
        "H_deterministic_unique_scene_ids": len(split_ids) == len(dataset),
        "H_real_appearance_mean_gap": mean_gap <= .18,
        "H_real_appearance_std_ratio": std_ratio <= 3.0,
        "H_real_gradient_ratio": gradient_ratio <= 3.0,
        "H_real_depth_median_ratio": depth_ratio <= 2.0,
    }
    return {"valid": all(hypotheses.values()), "hypotheses": hypotheses,
            "metrics": {"windows_checked": len(rows), "image_mean": image_mean,
                        "real_image_mean": profile.image_mean,
                        "mean_abs_gap": mean_gap, "image_std": image_std,
                        "real_image_std": profile.image_std,
                        "image_std_ratio": std_ratio, "gradient_mean": gradient,
                        "real_gradient_mean": profile.gradient_mean,
                        "gradient_ratio": gradient_ratio,
                        "depth_median_m": depth_median,
                        "real_depth_median_m": profile.depth_median_m,
                        "depth_median_ratio": depth_ratio},
            "geometry_contract": geometry}
