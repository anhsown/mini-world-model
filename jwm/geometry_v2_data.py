"""Real/sim geometry adapters and admission gates for Eye Physical v2."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset

from .geometry_data import render_geometry_sequence
from .mathx import relative_pose_c2w, rotation_geodesic
from .tum_rgbd import (TUMRGBDWindowDataset, quaternion_xyzw_to_matrix)


def normalize_pose_origin(pose_c2w: torch.Tensor) -> torch.Tensor:
    """Express every C2W pose in the first-camera coordinate frame."""
    return torch.linalg.inv(pose_c2w[:1]) @ pose_c2w


def procedural_v2_row(seed: int, frames: int = 8, size: int = 256) -> dict:
    row = render_geometry_sequence(seed, frames, size, size).as_dict()
    row["pose_c2w"] = normalize_pose_origin(row["pose_c2w"])
    row["depth_valid"] = torch.isfinite(row["depth"]) & (row["depth"] > 1e-6)
    row["source"] = "procedural_exact"
    return row


def _letterbox(x: torch.Tensor, height: int, width: int, mode: str,
               fill: float = 0.0) -> tuple[torch.Tensor, float, int, int]:
    h, w = x.shape[-2:]
    scale = min(height / h, width / w)
    nh, nw = max(1, round(h * scale)), max(1, round(w * scale))
    kwargs = {"mode": mode}
    if mode in ("bilinear", "bicubic"):
        kwargs["align_corners"] = False
    y = F.interpolate(x.unsqueeze(0), size=(nh, nw), **kwargs).squeeze(0)
    out = x.new_full((x.shape[0], height, width), fill)
    top, left = (height - nh) // 2, (width - nw) // 2
    out[:, top:top + nh, left:left + nw] = y
    return out, scale, top, left


def load_tartanair_depth(path: str | Path,
                         png_depth_scale: float = 1000.0) -> np.ndarray:
    """Decode TartanAir v1 NPY or v2 lossless float32 RGBA depth.

    TartanAir v2 stores each float32 depth value as its four raw bytes in an
    RGBA PNG. Ordinary 16-bit depth PNGs remain supported for converted
    mirrors and are interpreted in ``png_depth_scale`` units per metre.
    """
    path = Path(path)
    if path.suffix.lower() == ".npy":
        return np.load(path).astype(np.float32)
    with Image.open(path) as depth_file:
        encoded = np.asarray(depth_file).copy()
    if encoded.dtype == np.uint8 and encoded.ndim == 3 and encoded.shape[-1] == 4:
        contiguous = np.ascontiguousarray(encoded)
        return contiguous.view("<f4").reshape(encoded.shape[:2]).copy()
    depth = encoded.astype(np.float32)
    if depth.ndim == 3:
        depth = depth[..., 0]
    return depth / png_depth_scale


class TaggedTUMDataset(TUMRGBDWindowDataset):
    """TUM-format RGB-D windows with source identity preserved."""

    def __init__(self, roots, source: str = "tum", **kwargs):
        self.source = source
        super().__init__(roots, **kwargs)

    def __getitem__(self, index):
        row = super().__getitem__(index)
        row["source"] = self.source
        return row

    def scene_ids(self) -> set[str]:
        return {name for name, _ in self.sequences}


class BonnRGBDWindowDataset(TaggedTUMDataset):
    """Bonn Dynamic uses the official TUM RGB-D stream format."""

    def __init__(self, roots, **kwargs):
        super().__init__(roots, source="bonn_dynamic", **kwargs)


class TartanAirWindowDataset(Dataset):
    """Filesystem adapter for TartanAir V1/V2 front-camera trajectories.

    A trajectory directory must contain ``pose_lcam_front.txt`` together with
    ``image_lcam_front`` and ``depth_lcam_front``.  Depth may be ``.npy`` or
    16-bit PNG. Optional ``dynamic_mask_lcam_front`` files are admitted when
    present. Dataset download remains outside this class so license and storage
    decisions stay explicit in the notebook.
    """

    def __init__(self, roots: list[str | Path], frames: int = 8,
                 frame_stride: int = 1, window_stride: int = 8,
                 height: int = 256, width: int = 256,
                 camera: str = "lcam_front", png_depth_scale: float = 1000.0,
                 assume_static: bool = True):
        self.frames, self.frame_stride = frames, frame_stride
        self.height, self.width = height, width
        self.camera, self.png_depth_scale = camera, png_depth_scale
        self.assume_static = assume_static
        self.trajectories, self.windows = [], []
        span = (frames - 1) * frame_stride + 1
        for supplied in roots:
            root = Path(supplied)
            pose_files = ([root / f"pose_{camera}.txt"]
                          if (root / f"pose_{camera}.txt").exists()
                          else sorted(root.rglob(f"pose_{camera}.txt")))
            for pose_file in pose_files:
                parent = pose_file.parent
                image_dir = parent / f"image_{camera}"
                depth_dir = parent / f"depth_{camera}"
                if not image_dir.is_dir() or not depth_dir.is_dir():
                    continue
                images = sorted([p for p in image_dir.iterdir()
                                 if p.suffix.lower() in (".png", ".jpg", ".jpeg")])
                depths = sorted([p for p in depth_dir.iterdir()
                                 if p.suffix.lower() in (".npy", ".png", ".tiff")])
                poses = self._load_poses(pose_file)
                count = min(len(images), len(depths), len(poses))
                if count < span:
                    continue
                mask_dir = parent / f"dynamic_mask_{camera}"
                masks = sorted(mask_dir.iterdir()) if mask_dir.is_dir() else []
                identity = hashlib.sha1(str(parent.resolve()).encode()).hexdigest()[:16]
                trajectory_index = len(self.trajectories)
                self.trajectories.append({
                    "scene_id": f"tartanair-{identity}",
                    "images": images[:count], "depths": depths[:count],
                    "poses": poses[:count], "masks": masks[:count],
                })
                for start in range(0, count - span + 1, window_stride):
                    self.windows.append((trajectory_index, start))

    @staticmethod
    def _load_poses(path: Path) -> torch.Tensor:
        rows = np.loadtxt(path, dtype=np.float64)
        rows = np.atleast_2d(rows)
        if rows.shape[1] < 7:
            raise ValueError(f"expected xyz+xyzw pose rows in {path}")
        translation = torch.from_numpy(rows[:, :3]).float()
        quaternion = torch.from_numpy(rows[:, 3:7]).float()
        rotation = quaternion_xyzw_to_matrix(quaternion)
        pose = torch.eye(4).view(1, 4, 4).repeat(len(rows), 1, 1)
        pose[:, :3, :3], pose[:, :3, 3] = rotation, translation
        return normalize_pose_origin(pose)

    def __len__(self):
        return len(self.windows)

    def scene_ids(self) -> set[str]:
        return {row["scene_id"] for row in self.trajectories}

    def __getitem__(self, index):
        trajectory_index, start = self.windows[index]
        trajectory = self.trajectories[trajectory_index]
        ids = [start + i * self.frame_stride for i in range(self.frames)]
        images, depths, masks = [], [], []
        intrinsics = None
        for frame_index in ids:
            with Image.open(trajectory["images"][frame_index]) as image_file:
                image_np = np.asarray(image_file.convert("RGB"), dtype=np.uint8).copy()
            image = torch.from_numpy(image_np).permute(2, 0, 1).float() / 255.0
            depth_path = trajectory["depths"][frame_index]
            depth_np = load_tartanair_depth(depth_path, self.png_depth_scale)
            if depth_np.ndim == 3:
                depth_np = depth_np[..., 0]
            depth = torch.from_numpy(depth_np).unsqueeze(0)
            original_h, original_w = image.shape[-2:]
            image, scale, top, left = _letterbox(
                image, self.height, self.width, "bilinear", 0.5)
            depth, _, _, _ = _letterbox(
                depth, self.height, self.width, "nearest", 0.0)
            if intrinsics is None:
                # TartanAir pinhole front cameras use 90-degree FoV.
                focal = 0.5 * original_w
                intrinsics = torch.tensor([
                    [focal * scale, 0.0, original_w * 0.5 * scale + left],
                    [0.0, focal * scale, original_h * 0.5 * scale + top],
                    [0.0, 0.0, 1.0]], dtype=torch.float32)
            images.append(image); depths.append(depth.squeeze(0))
            if trajectory["masks"]:
                with Image.open(trajectory["masks"][frame_index]) as mask_file:
                    mask_np = np.asarray(mask_file.convert("L"), dtype=np.uint8).copy()
                mask = torch.from_numpy(mask_np).float().unsqueeze(0) / 255.0
                mask, _, _, _ = _letterbox(mask, self.height, self.width,
                                            "nearest", 0.0)
                masks.append(mask.squeeze(0) > 0.5)
        depth = torch.stack(depths)
        row = {
            "image": torch.stack(images), "depth": depth,
            "depth_valid": torch.isfinite(depth) & (depth > 1e-6),
            "pose_c2w": normalize_pose_origin(trajectory["poses"][ids]),
            "intrinsics": intrinsics,
            "scene_id": trajectory["scene_id"], "source": "tartanair",
        }
        if masks:
            row["dynamic_mask"] = torch.stack(masks)
        elif self.assume_static:
            # TartanAir's raw RGB/depth/pose sequences are static-world
            # simulations unless a dynamic environment is deliberately
            # selected.  Notebook v2 admits only environments marked static;
            # this gives the dynamic head reliable negative supervision.
            row["dynamic_mask"] = torch.zeros_like(depth, dtype=torch.bool)
        return row


def dataset_scene_ids(dataset) -> set[str]:
    if hasattr(dataset, "scene_ids"):
        return set(dataset.scene_ids())
    if hasattr(dataset, "sequences"):
        return {name for name, _ in dataset.sequences}
    return set()


def validate_split_disjoint(train, validation, test=None) -> dict:
    groups = {"train": dataset_scene_ids(train),
              "validation": dataset_scene_ids(validation)}
    if test is not None:
        groups["test"] = dataset_scene_ids(test)
    leaked = set()
    names = list(groups)
    for i, left in enumerate(names):
        for right in names[i + 1:]:
            leaked |= groups[left] & groups[right]
    return {"valid": not leaked,
            "hypotheses": {"H_scene_split_no_leak": not leaked},
            "scene_counts": {key: len(value) for key, value in groups.items()},
            "leaked_scene_ids": sorted(leaked)}


def validate_geometry_source(dataset, source: str, max_windows: int = 12,
                             output: str | Path | None = None) -> dict:
    if len(dataset) == 0:
        report = {"valid": False, "source": source, "reason": "no windows"}
    else:
        count = min(max_windows, len(dataset))
        indices = sorted(set(min(len(dataset) - 1,
                                 math.floor(i * len(dataset) / count))
                             for i in range(count)))
        rows = [dataset[index] for index in indices]
        images = torch.cat([row["image"].flatten() for row in rows])
        valid_fraction = torch.stack([
            row.get("depth_valid", row["depth"] > 0).float().mean() for row in rows])
        depth_values = torch.cat([
            row["depth"][row.get("depth_valid", row["depth"] > 0)].flatten()[::16]
            for row in rows])
        depth_values = depth_values[torch.isfinite(depth_values)]
        depth_median = depth_values.median() if depth_values.numel() else torch.tensor(float("nan"))
        depth_p99 = (torch.quantile(depth_values, 0.99) if depth_values.numel()
                     else torch.tensor(float("nan")))
        rotations = torch.cat([row["pose_c2w"][:, :3, :3] for row in rows])
        identity3 = torch.eye(3).expand_as(rotations)
        so3_error = (rotations.transpose(-1, -2) @ rotations - identity3).abs().max()
        determinant_error = (torch.det(rotations) - 1.0).abs().max()
        translation = torch.cat([
            relative_pose_c2w(row["pose_c2w"][:-1],
                              row["pose_c2w"][1:])[:, :3, 3].norm(dim=-1)
            for row in rows])
        rotation = torch.cat([
            rotation_geodesic(row["pose_c2w"][:-1, :3, :3],
                              row["pose_c2w"][1:, :3, :3])
            for row in rows])
        moving = (translation > 0.003) | (rotation > math.radians(0.15))
        first_identity = all(torch.allclose(row["pose_c2w"][0], torch.eye(4),
                                            atol=1e-4) for row in rows)
        dynamic_rows = [row["dynamic_mask"].float().mean()
                        for row in rows if "dynamic_mask" in row]
        hypotheses = {
            "H_images_finite_nonconstant": bool(torch.isfinite(images).all() and
                                                 images.std() > 0.01),
            "H_real_depth_available": bool(valid_fraction.mean() > 0.20),
            "H_metric_depth_scale_plausible": bool(
                torch.isfinite(depth_median) and 0.10 < depth_median < 100.0 and
                torch.isfinite(depth_p99) and depth_p99 < 1000.0),
            "H_pose_is_SO3": bool(so3_error < 1e-4 and determinant_error < 1e-4),
            "H_pose_normalized_to_first_frame": first_identity,
            "H_motion_not_identity_dominated": bool(moving.float().mean() > 0.25),
        }
        report = {
            "valid": all(hypotheses.values()), "source": source,
            "hypotheses": hypotheses,
            "metrics": {
                "windows_total": len(dataset), "windows_checked": len(rows),
                "depth_valid_fraction": float(valid_fraction.mean()),
                "depth_median_m": float(depth_median),
                "depth_p99_m": float(depth_p99),
                "rgb_mean": float(images.mean()), "rgb_std": float(images.std()),
                "pose_so3_max_error": float(so3_error),
                "pose_det_max_error": float(determinant_error),
                "translation_step_median": float(translation.median()),
                "translation_step_p90": float(torch.quantile(translation, 0.9)),
                "rotation_step_deg_median": float(rotation.median() * 180 / math.pi),
                "rotation_step_deg_p90": float(torch.quantile(rotation, 0.9) * 180 / math.pi),
                "non_identity_step_fraction": float(moving.float().mean()),
                "dynamic_label_available": bool(dynamic_rows),
                "dynamic_pixel_fraction": (float(torch.stack(dynamic_rows).mean())
                                           if dynamic_rows else None),
            }}
    if output is not None:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
