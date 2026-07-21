"""Calibrated data contract and admission gates for CTPG-Eye v3.

V3 never silently invents camera geometry inside the model. Dataset adapters
are normalized here to a per-frame pinhole matrix, timestamps and a projection
axis convention. Exact rigid flow is generated from metric depth and C2W poses
for static pixels; dynamic pixels are explicitly excluded.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterable

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from .geometry_math_v3 import (
    backproject_depth, camera_rays, camera_transform, ensure_frame_intrinsics,
    project_points, resize_crop_intrinsics, rigid_flow, transform_points,
)
from .geometry_v2_data import normalize_pose_origin, procedural_v2_row


TUM_INTRINSICS = {
    "freiburg1": (517.3, 516.5, 318.6, 255.3),
    "freiburg2": (520.9, 521.0, 325.1, 249.7),
    "freiburg3": (535.4, 539.2, 320.1, 247.6),
    # Official Bonn Dynamic RGB camera calibration. Depth is already
    # registered to RGB, so the generic ROS-default 525 focal length is wrong.
    "bonn": (542.822841, 542.576870, 315.593520, 237.756098),
}


def _profile_intrinsics(source: str, scene_id: str, height: int,
                        width: int) -> torch.Tensor | None:
    """Known public RGB-D calibration, transformed into the letterbox canvas."""
    label = f"{source} {scene_id}".lower()
    profile = None
    for name, values in TUM_INTRINSICS.items():
        if name in label:
            profile = values
            break
    if profile is None:
        return None
    fx, fy, cx, cy = profile
    raw_h, raw_w = 480, 640
    scale = min(height / raw_h, width / raw_w)
    nh, nw = round(raw_h * scale), round(raw_w * scale)
    top, left = (height - nh) // 2, (width - nw) // 2
    k = torch.tensor([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])
    return resize_crop_intrinsics(k, scale, scale, left, top)


def standardize_geometry_row(row: dict, *, strict_camera: bool = True,
                             default_fps: float = 10.0,
                             add_flow: bool = True) -> dict:
    """Return a copied row satisfying the immutable Eye-v3 data contract."""
    out = dict(row)
    images = out["image"].float()
    if images.ndim != 4 or images.shape[1] != 3:
        raise ValueError("image must have shape (T,3,H,W)")
    frames, _, height, width = images.shape
    source = str(out.get("source", "unknown"))
    scene_id = str(out.get("scene_id", "unknown"))
    intrinsics = out.get("intrinsics")
    if intrinsics is None:
        intrinsics = _profile_intrinsics(source, scene_id, height, width)
    if intrinsics is None:
        if strict_camera:
            raise ValueError(f"no calibrated intrinsics for {source}/{scene_id}")
        focal = 0.5 * width / math.tan(math.radians(70.0) / 2)
        intrinsics = torch.tensor([[focal, 0.0, width / 2],
                                   [0.0, focal, height / 2],
                                   [0.0, 0.0, 1.0]])
        out["camera_is_assumed"] = True
    out["intrinsics"] = ensure_frame_intrinsics(
        torch.as_tensor(intrinsics, dtype=torch.float32), frames)
    out["image"] = images
    out["pose_c2w"] = normalize_pose_origin(out["pose_c2w"].float())
    out["depth"] = out["depth"].float()
    out["depth_valid"] = (out.get("depth_valid",
                                   torch.isfinite(out["depth"]) & (out["depth"] > 0))
                            .bool())
    if "dynamic_mask" not in out:
        out["dynamic_mask"] = torch.zeros_like(out["depth"], dtype=torch.bool)
        out["dynamic_labels_missing"] = True
    else:
        out["dynamic_mask"] = out["dynamic_mask"].bool()
        out["dynamic_label_source"] = str(
            out.get("dynamic_label_source", "measured_or_simulator"))
    timestamp = out.get("timestamp")
    if timestamp is None:
        timestamp = torch.arange(frames, dtype=torch.float64) / float(default_fps)
    out["timestamp"] = torch.as_tensor(timestamp, dtype=torch.float64)
    if out["timestamp"].numel() != frames:
        raise ValueError("timestamp count must match frame count")
    # The analytic renderer uses camera +y up. Public RGB-D datasets use the
    # OpenCV projection convention (+y down).
    out["projection_y_sign"] = torch.tensor(
        -1.0 if source.startswith("procedural") else 1.0, dtype=torch.float32)
    out["source"] = source
    out["scene_id"] = scene_id
    if out.pop("dynamic_labels_missing", False):
        if source.startswith("bonn"):
            out["dynamic_mask"] = derive_depth_motion_mask(out)
            out["dynamic_labels_pseudo_depth_motion"] = True
            out["dynamic_label_source"] = "depth_motion_pseudo"
        else:
            out["dynamic_labels_assumed_static"] = True
            out["dynamic_label_source"] = "assumed_static"
    if add_flow:
        out.update(derive_rigid_supervision(out))
    return out


def derive_depth_motion_mask(row: dict, relative_threshold: float = .08) -> torch.Tensor:
    """Motion/occlusion pseudo-label from calibrated depth reprojection.

    Bonn does not ship per-pixel actor masks. Its metric depth and mocap pose do
    let us identify pixels that violate the rigid-scene hypothesis. Boundary
    occlusions are intentionally retained as uncertain/dynamic evidence rather
    than mislabeled static supervision.
    """
    depth, pose, k = row["depth"], row["pose_c2w"], row["intrinsics"]
    valid, sign = row["depth_valid"], row["projection_y_sign"]
    h, w = depth.shape[-2:]
    masks = []
    for index in range(depth.shape[0] - 1):
        points = backproject_depth(depth[index], k[index], sign).reshape(-1, 3)
        moved = transform_points(camera_transform(pose[index], pose[index + 1]),
                                 points.unsqueeze(0)).squeeze(0)
        pixels, projected_z = project_points(moved.unsqueeze(0), k[index + 1], sign)
        pixels = pixels.squeeze(0)
        gx = 2 * pixels[:, 0] / max(w - 1, 1) - 1
        gy = 2 * pixels[:, 1] / max(h - 1, 1) - 1
        grid = torch.stack((gx, gy), -1).reshape(1, h, w, 2)
        target = F.grid_sample(depth[index + 1:index + 2, None], grid,
                               mode="bilinear", align_corners=True).reshape(-1)
        target_valid = F.grid_sample(valid[index + 1:index + 2, None].float(), grid,
                                     mode="nearest", align_corners=True).reshape(-1) > .5
        inside = ((pixels[:, 0] >= 0) & (pixels[:, 0] <= w - 1) &
                  (pixels[:, 1] >= 0) & (pixels[:, 1] <= h - 1) &
                  (projected_z > 1e-4) & target_valid)
        relative = (target - projected_z).abs() / target.clamp_min(1e-3)
        mask = (inside & (relative > relative_threshold)).reshape(h, w)
        masks.append(mask & valid[index])
    masks.append(masks[-1].clone() if masks else torch.zeros_like(depth[0], dtype=torch.bool))
    return torch.stack(masks)


def derive_rigid_supervision(row: dict) -> dict:
    depth, valid = row["depth"], row["depth_valid"]
    pose, k = row["pose_c2w"], row["intrinsics"]
    dynamic = row["dynamic_mask"]
    sign = row["projection_y_sign"]
    flows, masks = [], []
    for index in range(depth.shape[0] - 1):
        transform = camera_transform(pose[index], pose[index + 1])
        flow, visible = rigid_flow(depth[index], k[index], k[index + 1],
                                   transform, sign)
        mask = visible & valid[index] & ~dynamic[index]
        flows.append(flow); masks.append(mask)
    return {"rigid_flow": torch.stack(flows),
            "rigid_flow_valid": torch.stack(masks)}


def procedural_v3_row(seed: int, frames: int = 6, size: int = 128) -> dict:
    row = procedural_v2_row(seed, frames=frames, size=size)
    row["source"] = "procedural_exact"
    row["timestamp"] = torch.arange(frames, dtype=torch.float64) / 30.0
    return standardize_geometry_row(row, strict_camera=True)


class CalibratedGeometryDataset(Dataset):
    """Contract wrapper for existing TUM/Bonn/Tartan/procedural adapters."""

    def __init__(self, dataset: Dataset, *, strict_camera: bool = True,
                 default_fps: float = 10.0):
        self.dataset = dataset
        self.strict_camera = strict_camera
        self.default_fps = default_fps

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return standardize_geometry_row(self.dataset[index],
                                        strict_camera=self.strict_camera,
                                        default_fps=self.default_fps)

    def scene_ids(self):
        if hasattr(self.dataset, "scene_ids"):
            return self.dataset.scene_ids()
        if hasattr(self.dataset, "sequences"):
            return {name for name, _ in self.dataset.sequences}
        return set()


def stack_geometry_v3_rows(rows: list[dict]) -> dict:
    """Stack without dropping camera state (the critical Eye-v2 bug fix)."""
    required = ("image", "depth", "depth_valid", "pose_c2w", "intrinsics",
                "timestamp", "projection_y_sign", "dynamic_mask",
                "rigid_flow", "rigid_flow_valid")
    missing = {key for key in required if any(key not in row for row in rows)}
    if missing:
        raise KeyError(f"Eye-v3 rows missing required fields: {sorted(missing)}")
    batch = {key: torch.stack([row[key] for row in rows]) for key in required}
    batch["source"] = [row["source"] for row in rows]
    batch["scene_id"] = [row["scene_id"] for row in rows]
    batch["dynamic_label_source"] = [
        row.get("dynamic_label_source", "unknown") for row in rows]
    return batch


def make_counterfactuals(batch: dict) -> dict:
    """Causal controls: frozen/reversed/shuffled windows and perturbed K."""
    result = {}
    images = batch["image"]
    result["frozen_image"] = images[:, :1].expand_as(images).clone()
    result["reverse_image"] = images.flip(1)
    if images.shape[0] > 1:
        # A different sample is a strong, semantically valid negative.
        result["wrong_window_image"] = images.roll(1, 0)
    else:
        # Circular shift preserves almost every adjacent pair.  Even/odd
        # interleaving breaks local chronology without changing frame content.
        time = images.shape[1]
        order = torch.cat((torch.arange(0, time, 2, device=images.device),
                           torch.arange(1, time, 2, device=images.device)))
        result["wrong_window_image"] = images.index_select(1, order)
    wrong_k = batch["intrinsics"].clone()
    wrong_k[..., 0, 0] *= 0.65
    wrong_k[..., 1, 1] *= 1.35
    wrong_k[..., 0, 2] += images.shape[-1] * 0.12
    result["wrong_intrinsics"] = wrong_k
    return result


def _reprojection_inlier(row: dict, stride: int = 8) -> float:
    """Actual depth/pose/K agreement against the next measured depth map."""
    depth, valid = row["depth"], row["depth_valid"]
    pose, k, sign = row["pose_c2w"], row["intrinsics"], row["projection_y_sign"]
    dynamic = row["dynamic_mask"]
    h, w = depth.shape[-2:]
    scores = []
    for index in range(depth.shape[0] - 1):
        ys, xs = torch.meshgrid(torch.arange(0, h, stride),
                                torch.arange(0, w, stride), indexing="ij")
        z = depth[index, ys, xs]
        fx, fy = k[index, 0, 0], k[index, 1, 1]
        cx, cy = k[index, 0, 2], k[index, 1, 2]
        points = torch.stack(((xs - cx) / fx * z,
                              sign * (ys - cy) / fy * z, z), dim=-1).reshape(1, -1, 3)
        moved = transform_points(camera_transform(pose[index], pose[index + 1]),
                                 points).squeeze(0)
        pixels, projected_z = project_points(moved.unsqueeze(0), k[index + 1], sign)
        pixels, projected_z = pixels.squeeze(0), projected_z.squeeze(0)
        u, v = pixels.round().long().unbind(-1)
        inside = ((projected_z > 1e-4) & (u >= 0) & (u < w) & (v >= 0) & (v < h) &
                  valid[index, ys, xs].reshape(-1) &
                  ~dynamic[index, ys, xs].reshape(-1))
        if bool(inside.any()):
            u_ok, v_ok = u[inside], v[inside]
            target_valid = valid[index + 1, v_ok, u_ok]
            target = depth[index + 1, v_ok, u_ok]
            relative = (target - projected_z[inside]).abs() / target.clamp_min(1e-3)
            if bool(target_valid.any()):
                scores.append((relative[target_valid] < .08).float().mean())
    return float(torch.stack(scores).mean()) if scores else 0.0


def validate_geometry_v3_source(dataset: Dataset, source: str,
                                max_windows: int = 8) -> dict:
    if len(dataset) == 0:
        return {"valid": False, "source": source, "reason": "no windows",
                "hypotheses": {"H_windows_available": False}}
    count = min(max_windows, len(dataset))
    indices = sorted(set(min(len(dataset) - 1, math.floor(i * len(dataset) / count))
                         for i in range(count)))
    rows, errors = [], []
    for index in indices:
        try:
            rows.append(dataset[index])
        except Exception as exc:
            errors.append(f"{type(exc).__name__}: {exc}")
    if not rows:
        return {"valid": False, "source": source, "reason": errors[0],
                "hypotheses": {"H_camera_contract_parseable": False}}
    ks = torch.cat([row["intrinsics"] for row in rows])
    images = torch.cat([row["image"].flatten() for row in rows])
    heights = [row["image"].shape[-2] for row in rows]
    widths = [row["image"].shape[-1] for row in rows]
    plausible_k = all(bool((row["intrinsics"][:, 0, 0] > 0).all() and
                           (row["intrinsics"][:, 1, 1] > 0).all() and
                           (row["intrinsics"][:, 0, 2].abs() < 2 * widths[i]).all() and
                           (row["intrinsics"][:, 1, 2].abs() < 2 * heights[i]).all())
                      for i, row in enumerate(rows))
    rays_ok = all(bool(torch.isfinite(camera_rays(row["intrinsics"],
                                                  row["image"].shape[-2],
                                                  row["image"].shape[-1],
                                                  row["projection_y_sign"])).all())
                  for row in rows)
    timestamp_ok = all(bool((row["timestamp"][1:] > row["timestamp"][:-1]).all())
                       for row in rows)
    rotations = torch.cat([row["pose_c2w"][:, :3, :3] for row in rows])
    eye = torch.eye(3).expand_as(rotations)
    so3 = float((rotations.transpose(-1, -2) @ rotations - eye).abs().max())
    determinant = float((torch.det(rotations) - 1).abs().max())
    valid_depth = torch.stack([row["depth_valid"].float().mean() for row in rows])
    flow_coverage = torch.stack([row["rigid_flow_valid"].float().mean() for row in rows])
    dynamic_fraction = torch.stack([row["dynamic_mask"].float().mean() for row in rows])
    dynamic_label_sources = sorted({row.get("dynamic_label_source", "unknown")
                                    for row in rows})
    reprojection = [_reprojection_inlier(row) for row in rows]
    # Wrong K must alter ray geometry; otherwise a later model could ignore K
    # without the admission report noticing a degenerate calibration field.
    sensitivity = []
    for row in rows:
        wrong = row["intrinsics"].clone(); wrong[:, 0, 0] *= 0.7
        good_ray = camera_rays(row["intrinsics"], 5, 5, row["projection_y_sign"])
        bad_ray = camera_rays(wrong, 5, 5, row["projection_y_sign"])
        sensitivity.append(float((good_ray - bad_ray).abs().mean()))
    hypotheses = {
        "H_camera_contract_parseable": not errors,
        "H_intrinsics_per_frame_finite": bool(torch.isfinite(ks).all()),
        "H_intrinsics_pinhole_plausible": plausible_k,
        "H_camera_rays_finite": rays_ok,
        "H_timestamp_strictly_increasing": timestamp_ok,
        "H_rgb_finite_nonconstant": bool(torch.isfinite(images).all() and images.std() > .01),
        "H_metric_depth_available": bool(valid_depth.mean() > .15),
        "H_pose_is_SE3": so3 < 1e-4 and determinant < 1e-4,
        "H_static_flow_coverage": bool(flow_coverage.mean() > .05),
        "H_rgbd_pose_k_reprojection_consistent": sum(reprojection) / len(reprojection) > .20,
        "H_intrinsics_counterfactual_effective": sum(sensitivity) / len(sensitivity) > 1e-3,
    }
    return {"valid": all(hypotheses.values()), "source": source,
            "hypotheses": hypotheses, "errors": errors,
            "metrics": {"windows_total": len(dataset), "windows_checked": len(rows),
                        "depth_valid_fraction": float(valid_depth.mean()),
                        "static_flow_coverage": float(flow_coverage.mean()),
                        "dynamic_positive_fraction": float(dynamic_fraction.mean()),
                        "dynamic_label_sources": dynamic_label_sources,
                        "measured_depth_reprojection_inlier": sum(reprojection) / len(reprojection),
                        "camera_ray_counterfactual_delta": sum(sensitivity) / len(sensitivity),
                        "pose_so3_max_error": so3,
                        "pose_det_max_error": determinant}}


def validate_geometry_v3_datasets(splits: dict[str, dict[str, Dataset]],
                                  output: str | Path) -> dict:
    report = {"sources": {}, "scene_leaks": {}, "valid": True}
    for split, sources in splits.items():
        for name, dataset in sources.items():
            key = f"{split}:{name}"
            report["sources"][key] = validate_geometry_v3_source(dataset, key)
    names = sorted({name for sources in splits.values() for name in sources})
    for name in names:
        ids = {}
        for split, sources in splits.items():
            if name in sources and hasattr(sources[name], "scene_ids"):
                ids[split] = set(sources[name].scene_ids())
        leaks = set()
        keys = list(ids)
        for i, left in enumerate(keys):
            for right in keys[i + 1:]:
                leaks |= ids[left] & ids[right]
        report["scene_leaks"][name] = sorted(leaks)
    train_reports = [item for key, item in report["sources"].items()
                     if key.startswith("train:")]
    dynamic_supervision = any(
        item.get("metrics", {}).get("dynamic_positive_fraction", 0.0) > 1e-5 and
        any(mode not in ("assumed_static", "unknown")
            for mode in item.get("metrics", {}).get("dynamic_label_sources", []))
        for item in train_reports)
    report["global_hypotheses"] = {
        "H_scene_groups_do_not_leak": not any(report["scene_leaks"].values()),
        "H_train_has_positive_dynamic_supervision": dynamic_supervision,
    }
    report["valid"] = (all(item["valid"] for item in report["sources"].values()) and
                       all(report["global_hypotheses"].values()))
    report["failures"] = {
        key: [name for name, passed in item.get("hypotheses", {}).items() if not passed]
        for key, item in report["sources"].items() if not item["valid"]}
    report["failures"]["global"] = [
        name for name, passed in report["global_hypotheses"].items() if not passed]
    if not report["failures"]["global"]:
        report["failures"].pop("global")
    path = Path(output); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
