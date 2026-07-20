"""TUM RGB-D adapter for real-domain Eye Physical training/evaluation."""

from __future__ import annotations

import bisect
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset


def _records(path: Path, fields: int):
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) >= fields:
            rows.append(parts)
    return rows


def _nearest(timestamps: list[float], value: float, max_delta: float):
    at = bisect.bisect_left(timestamps, value)
    candidates = [i for i in (at - 1, at) if 0 <= i < len(timestamps)]
    if not candidates:
        return None
    best = min(candidates, key=lambda i: abs(timestamps[i] - value))
    return best if abs(timestamps[best] - value) <= max_delta else None


def quaternion_xyzw_to_matrix(q: torch.Tensor) -> torch.Tensor:
    q = q / q.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    x, y, z, w = q.unbind(-1)
    return torch.stack((
        1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w),
        2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w),
        2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y),
    ), dim=-1).reshape(*q.shape[:-1], 3, 3)


def load_tum_associations(root: str | Path, max_rgb_depth_delta: float = 0.03,
                          max_pose_delta: float = 0.03) -> list[dict]:
    """Associate official timestamped RGB, depth and mocap trajectory streams."""
    root = Path(root)
    rgb = _records(root / "rgb.txt", 2)
    depth = _records(root / "depth.txt", 2)
    pose = _records(root / "groundtruth.txt", 8)
    depth_t = [float(x[0]) for x in depth]
    pose_t = [float(x[0]) for x in pose]
    rows = []
    for rgb_row in rgb:
        timestamp = float(rgb_row[0])
        di = _nearest(depth_t, timestamp, max_rgb_depth_delta)
        pi = _nearest(pose_t, timestamp, max_pose_delta)
        if di is None or pi is None:
            continue
        p = pose[pi]
        translation = torch.tensor([float(v) for v in p[1:4]])
        rotation = quaternion_xyzw_to_matrix(
            torch.tensor([float(v) for v in p[4:8]]))
        c2w = torch.eye(4)
        c2w[:3, :3], c2w[:3, 3] = rotation, translation
        rows.append({"timestamp": timestamp,
                     "rgb": root / rgb_row[1],
                     "depth": root / depth[di][1],
                     "pose_c2w": c2w})
    return rows


def _letterbox_tensor(x: torch.Tensor, height: int, width: int,
                      mode: str, fill: float = 0.0) -> torch.Tensor:
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
    return out


class TUMRGBDWindowDataset(Dataset):
    """Fixed windows from one or more TUM sequences, normalized to first pose."""

    def __init__(self, roots: list[str | Path], frames: int = 8,
                 frame_stride: int = 3, window_stride: int = 24,
                 height: int = 256, width: int = 256,
                 depth_scale: float = 5000.0):
        self.frames, self.frame_stride = frames, frame_stride
        self.height, self.width, self.depth_scale = height, width, depth_scale
        self.sequences, self.windows = [], []
        span = (frames - 1) * frame_stride + 1
        for root in roots:
            rows = load_tum_associations(root)
            sequence_index = len(self.sequences)
            self.sequences.append((str(Path(root).name), rows))
            for start in range(0, max(0, len(rows) - span + 1), window_stride):
                self.windows.append((sequence_index, start))

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, index):
        sequence_index, start = self.windows[index]
        name, rows = self.sequences[sequence_index]
        selected = [rows[start + i * self.frame_stride]
                    for i in range(self.frames)]
        images, depths, poses = [], [], []
        origin_inv = torch.linalg.inv(selected[0]["pose_c2w"])
        for row in selected:
            with Image.open(row["rgb"]) as image:
                arr = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
            image = torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0
            with Image.open(row["depth"]) as depth_image:
                dep = np.asarray(depth_image, dtype=np.uint16).copy()
            depth = torch.from_numpy(dep).float().unsqueeze(0) / self.depth_scale
            image = _letterbox_tensor(image, self.height, self.width, "bilinear", 0.5)
            depth = _letterbox_tensor(depth, self.height, self.width, "nearest", 0.0)
            images.append(image); depths.append(depth.squeeze(0))
            poses.append(origin_inv @ row["pose_c2w"])
        depth = torch.stack(depths)
        return {"image": torch.stack(images), "depth": depth,
                "depth_valid": depth > 1e-5, "pose_c2w": torch.stack(poses),
                "scene_id": name,
                # Unix-scale RGB-D timestamps lose all sub-second motion when
                # first materialized as float32. Keep float64 at the adapter.
                "timestamp": torch.tensor([r["timestamp"] for r in selected],
                                          dtype=torch.float64)}


def validate_tum_dataset(dataset: TUMRGBDWindowDataset) -> dict:
    """Real-data admission report; distinct sequence roots are split upstream."""
    if not len(dataset):
        return {"valid": False, "reason": "no associated windows"}
    indices = sorted(set([0, len(dataset) // 2, len(dataset) - 1]))
    rows = [dataset[i] for i in indices]
    valid_fraction = torch.cat([r["depth_valid"].flatten() for r in rows]).float().mean()
    rotations = torch.cat([r["pose_c2w"][:, :3, :3] for r in rows])
    eye = torch.eye(3).expand_as(rotations)
    so3_error = (rotations.transpose(-1, -2) @ rotations - eye).abs().max()
    motion = torch.cat([(r["pose_c2w"][1:, :3, 3] -
                         r["pose_c2w"][:-1, :3, 3]).norm(dim=-1) for r in rows])
    hypotheses = {
        "H_rgb_depth_pose_associated": len(dataset) > 0,
        "H_real_depth_available": float(valid_fraction) > 0.20,
        "H_pose_is_SO3": float(so3_error) < 1e-4,
        "H_camera_motion_present": float(motion.max()) > 1e-4,
        "H_pose_normalized_to_first_frame": all(
            torch.allclose(r["pose_c2w"][0], torch.eye(4), atol=1e-4) for r in rows),
    }
    return {"valid": all(hypotheses.values()), "hypotheses": hypotheses,
            "metrics": {"windows": len(dataset),
                        "depth_valid_fraction": float(valid_fraction),
                        "pose_so3_max_error": float(so3_error),
                        "translation_per_frame_mean_m": float(motion.mean()),
                        "translation_per_frame_max_m": float(motion.max())}}
