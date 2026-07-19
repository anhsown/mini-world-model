"""Validated procedural RGB-D/pose sequences for Eye Physical bootstrapping.

This is intentionally a geometry *unit-test distribution*, not a replacement
for photorealistic data. Rays intersect metric planes and moving spheres, so
RGB, z-depth, dynamic masks, intrinsics and C2W poses are generated from one
consistent scene definition. Later stages mix TartanAir/Hypersim and real TUM
RGB-D data only after the same validation contract passes.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import Dataset


@dataclass
class GeometrySequence:
    image: torch.Tensor          # (T,3,H,W), float [0,1]
    depth: torch.Tensor          # (T,H,W), planar z-depth in metres
    pose_c2w: torch.Tensor       # (T,4,4)
    intrinsics: torch.Tensor     # (3,3)
    dynamic_mask: torch.Tensor   # (T,H,W), bool
    scene_id: str

    def as_dict(self) -> dict:
        return {
            "image": self.image, "depth": self.depth,
            "pose_c2w": self.pose_c2w, "intrinsics": self.intrinsics,
            "dynamic_mask": self.dynamic_mask, "scene_id": self.scene_id,
        }


def _yaw(angle: torch.Tensor) -> torch.Tensor:
    c, s = torch.cos(angle), torch.sin(angle)
    return torch.stack((c, torch.zeros_like(c), s,
                        torch.zeros_like(c), torch.ones_like(c), torch.zeros_like(c),
                        -s, torch.zeros_like(c), c)).reshape(3, 3)


def _camera_rays(height: int, width: int, fov_degrees: float = 70.0):
    focal = 0.5 * width / math.tan(math.radians(fov_degrees) / 2.0)
    y, x = torch.meshgrid(torch.arange(height, dtype=torch.float32),
                          torch.arange(width, dtype=torch.float32), indexing="ij")
    # Camera convention: +x right, +y up, +z forward; depth is camera z.
    rays = torch.stack(((x + 0.5 - width / 2) / focal,
                        -(y + 0.5 - height / 2) / focal,
                        torch.ones_like(x)), dim=-1)
    K = torch.tensor([[focal, 0.0, width / 2],
                      [0.0, focal, height / 2],
                      [0.0, 0.0, 1.0]], dtype=torch.float32)
    return rays, K


def render_geometry_sequence(seed: int, frames: int = 8, height: int = 128,
                             width: int = 128) -> GeometrySequence:
    """Render a deterministic metric scene using analytic ray intersections."""
    gen = torch.Generator().manual_seed(int(seed))
    rays_cam, intrinsics = _camera_rays(height, width)
    n_spheres = int(torch.randint(3, 7, (1,), generator=gen))
    centers = torch.empty(n_spheres, 3).uniform_(-1.0, 1.0, generator=gen)
    centers[:, 0] *= 3.0
    centers[:, 1] = torch.empty(n_spheres).uniform_(-0.45, 0.8, generator=gen)
    centers[:, 2] = torch.empty(n_spheres).uniform_(3.0, 10.0, generator=gen)
    radii = torch.empty(n_spheres).uniform_(0.3, 0.9, generator=gen)
    colours = torch.empty(n_spheres, 3).uniform_(0.15, 0.95, generator=gen)
    dynamic_index = seed % n_spheres

    images, depths, poses, dynamic_masks = [], [], [], []
    phase = float(torch.rand((), generator=gen) * math.tau)
    speed = float(torch.empty(()).uniform_(0.035, 0.11, generator=gen))
    yaw_speed = float(torch.empty(()).uniform_(-0.018, 0.018, generator=gen))

    for frame in range(frames):
        camera_x = (frame - (frames - 1) / 2) * speed
        camera_y = 0.05 * math.sin(phase + frame * 0.31)
        camera_z = 0.08 * math.cos(phase + frame * 0.19)
        rotation = _yaw(torch.tensor((frame - (frames - 1) / 2) * yaw_speed))
        origin = torch.tensor([camera_x, camera_y, camera_z])
        directions = rays_cam @ rotation.T

        pose = torch.eye(4)
        pose[:3, :3], pose[:3, 3] = rotation, origin

        # Start with a far wall. Parameter t equals planar z-depth because the
        # unnormalised camera ray has z=1 and rigid rotation preserves t.
        denom = directions[..., 2].clamp(min=1e-5)
        best_t = ((12.0 - origin[2]) / denom).clamp(min=0.05, max=30.0)
        hit_colour = torch.full((height, width, 3), 0.58)
        hit_normal = torch.zeros(height, width, 3)
        hit_normal[..., 2] = -1.0
        dynamic = torch.zeros(height, width, dtype=torch.bool)

        # Floor y=-1 and ceiling y=2.5 add long planes and occlusion edges.
        for plane_y, colour, normal_y in ((-1.0, (0.38, 0.34, 0.28), 1.0),
                                           (2.5, (0.72, 0.74, 0.78), -1.0)):
            t_plane = (plane_y - origin[1]) / directions[..., 1].clamp(
                min=-1e8, max=1e8)
            valid = (t_plane > 0.05) & (t_plane < best_t)
            best_t = torch.where(valid, t_plane, best_t)
            c = torch.tensor(colour).view(1, 1, 3)
            hit_colour = torch.where(valid[..., None], c, hit_colour)
            normal = torch.tensor([0.0, normal_y, 0.0]).view(1, 1, 3)
            hit_normal = torch.where(valid[..., None], normal, hit_normal)
            dynamic &= ~valid

        frame_centers = centers.clone()
        frame_centers[dynamic_index, 0] += 0.45 * math.sin(phase + frame * 0.35)
        for sphere_index in range(n_spheres):
            oc = origin - frame_centers[sphere_index]
            a = (directions * directions).sum(-1)
            b = 2.0 * (directions * oc).sum(-1)
            c = (oc * oc).sum() - radii[sphere_index] ** 2
            discriminant = b * b - 4.0 * a * c
            root = (-b - torch.sqrt(discriminant.clamp(min=0.0))) / (2.0 * a)
            valid = (discriminant >= 0.0) & (root > 0.05) & (root < best_t)
            point = origin + root[..., None] * directions
            normal = (point - frame_centers[sphere_index]) / radii[sphere_index]
            best_t = torch.where(valid, root, best_t)
            colour = colours[sphere_index].view(1, 1, 3)
            hit_colour = torch.where(valid[..., None], colour, hit_colour)
            hit_normal = torch.where(valid[..., None], normal, hit_normal)
            dynamic = torch.where(valid,
                                  torch.full_like(dynamic, sphere_index == dynamic_index),
                                  dynamic)

        light = torch.tensor([-0.35, 0.8, -0.48])
        light = light / light.norm()
        diffuse = (hit_normal * light).sum(-1).clamp(min=0.0)
        rgb = hit_colour * (0.35 + 0.65 * diffuse[..., None])
        # Deterministic, small sensor perturbation; geometry remains exact.
        noise = torch.randn(rgb.shape, generator=gen) * 0.008
        rgb = (rgb + noise).clamp(0.0, 1.0)
        images.append(rgb.permute(2, 0, 1))
        depths.append(best_t.clamp(0.05, 30.0))
        poses.append(pose)
        dynamic_masks.append(dynamic)

    identity = hashlib.sha1(f"jwm-geometry-{seed}".encode()).hexdigest()[:16]
    return GeometrySequence(torch.stack(images), torch.stack(depths),
                            torch.stack(poses), intrinsics,
                            torch.stack(dynamic_masks), identity)


class ProceduralGeometryDataset(Dataset):
    """Seed-disjoint deterministic split for unit-scale geometry training."""

    OFFSETS = {"train": 0, "val": 1_000_000, "test": 2_000_000}

    def __init__(self, split: str, samples: int, frames: int = 8,
                 height: int = 256, width: int = 256):
        if split not in self.OFFSETS:
            raise ValueError(f"unknown split {split}")
        self.split, self.samples = split, samples
        self.frames, self.height, self.width = frames, height, width

    def __len__(self):
        return self.samples

    def __getitem__(self, index):
        if not 0 <= index < self.samples:
            raise IndexError(index)
        return render_geometry_sequence(self.OFFSETS[self.split] + index,
                                        self.frames, self.height,
                                        self.width).as_dict()


def _reprojection_consistency(sequence: GeometrySequence, stride: int = 8) -> float:
    """Fraction of static, visible t->t+1 points agreeing with target depth."""
    depth, poses, K = sequence.depth, sequence.pose_c2w, sequence.intrinsics
    h, w = depth.shape[-2:]
    ys, xs = torch.meshgrid(torch.arange(0, h, stride),
                            torch.arange(0, w, stride), indexing="ij")
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    scores = []
    for i in range(depth.shape[0] - 1):
        z = depth[i, ys, xs]
        points_i = torch.stack(((xs + 0.5 - cx) / fx * z,
                                -(ys + 0.5 - cy) / fy * z, z), dim=-1)
        world = points_i @ poses[i, :3, :3].T + poses[i, :3, 3]
        points_j = (world - poses[i + 1, :3, 3]) @ poses[i + 1, :3, :3]
        zj = points_j[..., 2]
        u = torch.round(fx * points_j[..., 0] / zj + cx - 0.5).long()
        v = torch.round(-fy * points_j[..., 1] / zj + cy - 0.5).long()
        inside = (zj > 0.05) & (u >= 0) & (u < w) & (v >= 0) & (v < h)
        inside &= ~sequence.dynamic_mask[i, ys, xs]
        if bool(inside.any()):
            target = depth[i + 1, v[inside], u[inside]]
            rel = (target - zj[inside]).abs() / target.clamp(min=1e-3)
            # Occluded points disagree by a large amount; they remain failures.
            scores.append((rel < 0.04).float().mean())
    return float(torch.stack(scores).mean()) if scores else 0.0


def validate_geometry_dataset(samples_per_split: int = 4, frames: int = 6,
                              height: int = 96, width: int = 96,
                              output: str | Path | None = None) -> dict:
    """Validate assumptions before procedural data is admitted to training."""
    sets = {split: [render_geometry_sequence(offset + i, frames, height, width)
                    for i in range(samples_per_split)]
            for split, offset in ProceduralGeometryDataset.OFFSETS.items()}
    ids = {split: {x.scene_id for x in rows} for split, rows in sets.items()}
    no_leak = not ((ids["train"] & ids["val"]) or
                   (ids["train"] & ids["test"]) or
                   (ids["val"] & ids["test"]))
    all_rows = sum(sets.values(), [])
    depth_ok = all(bool(torch.isfinite(x.depth).all() and (x.depth > 0).all())
                   for x in all_rows)
    rotations = torch.cat([x.pose_c2w[:, :3, :3] for x in all_rows])
    eye = torch.eye(3).expand_as(rotations)
    pose_error = float((rotations.transpose(-1, -2) @ rotations - eye).abs().max())
    det_error = float((torch.det(rotations) - 1.0).abs().max())
    motion = torch.cat([(x.pose_c2w[1:, :3, 3] -
                         x.pose_c2w[:-1, :3, 3]).norm(dim=-1)
                        for x in all_rows])
    reprojection = [_reprojection_consistency(x) for x in all_rows]
    dynamic_fraction = float(torch.cat([x.dynamic_mask.flatten() for x in all_rows])
                             .float().mean())
    report = {
        "valid": bool(no_leak and depth_ok and pose_error < 1e-5 and
                      det_error < 1e-5 and float(motion.mean()) > 0.02 and
                      sum(reprojection) / len(reprojection) > 0.75),
        "hypotheses": {
            "H_scene_split_no_leak": no_leak,
            "H_depth_positive_finite": depth_ok,
            "H_pose_is_SO3": pose_error < 1e-5 and det_error < 1e-5,
            "H_camera_motion_nontrivial": float(motion.mean()) > 0.02,
            "H_static_reprojection_consistent": sum(reprojection) / len(reprojection) > 0.75,
            "H_dynamic_objects_present": dynamic_fraction > 0.001,
        },
        "metrics": {
            "samples": len(all_rows),
            "rotation_orthogonality_max": pose_error,
            "rotation_det_error_max": det_error,
            "mean_translation_per_frame_m": float(motion.mean()),
            "mean_static_reprojection_inlier": sum(reprojection) / len(reprojection),
            "min_static_reprojection_inlier": min(reprojection),
            "dynamic_pixel_fraction": dynamic_fraction,
            "rgb_mean": float(torch.cat([x.image.flatten() for x in all_rows]).mean()),
            "depth_median_m": float(torch.cat([x.depth.flatten() for x in all_rows]).median()),
        },
    }
    if output is not None:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
