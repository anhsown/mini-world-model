"""Controlled pre-training gates for CTPG-Eye v3.

No full dataset is touched. Each probe isolates one causal mechanism and writes
a machine-readable admission report. The full trainer requires ``valid=true``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jwm.geometric_eye_v3 import BoundedWorldMemory, SparseTrackUpdater
from jwm.geometry_math_v3 import (
    bundle_adjust_pair, camera_transform, project_points, rigid_flow, se3_exp,
    transform_points,
)
from jwm.geometry_v3_data import procedural_v3_row


def camera_probe() -> dict:
    row = procedural_v3_row(71, 3, 48)
    depth, k, pose = row["depth"], row["intrinsics"], row["pose_c2w"]
    transform = camera_transform(pose[0], pose[1])
    reference, valid = rigid_flow(depth[0], k[0], k[1], transform,
                                  row["projection_y_sign"])
    wrong = k[1].clone(); wrong[0, 0] *= .65; wrong[1, 1] *= 1.35
    normal, _ = rigid_flow(depth[0], k[0], k[1], transform,
                           row["projection_y_sign"])
    counterfactual, _ = rigid_flow(depth[0], k[0], wrong, transform,
                                   row["projection_y_sign"])
    normal_epe = torch.linalg.vector_norm(normal - reference, dim=-1)[valid].mean()
    wrong_epe = torch.linalg.vector_norm(counterfactual - reference, dim=-1)[valid].mean()
    return {"normal_epe_px": float(normal_epe), "wrong_k_epe_px": float(wrong_epe),
            "passed": bool(normal_epe < 1e-5 and wrong_epe > .5)}


def track_probe(device: torch.device, steps: int = 120) -> dict:
    torch.manual_seed(72)
    channels, height, width = 16, 12, 12
    source = torch.randn(1, channels, height, width, device=device)
    target = torch.roll(source, shifts=1, dims=3)
    y, x = torch.meshgrid(torch.arange(2, height - 2, device=device),
                          torch.arange(2, width - 3, device=device), indexing="ij")
    points = torch.stack((x, y), -1).reshape(1, -1, 2).float()
    truth = points + torch.tensor([1., 0.], device=device)
    tracker = SparseTrackUpdater(channels, hidden=48, radius=2,
                                 iterations=2).to(device)
    optimizer = torch.optim.AdamW(tracker.parameters(), lr=3e-3)
    with torch.no_grad():
        initial = torch.linalg.vector_norm(tracker(source, target, points)["target"] - truth,
                                           dim=-1).mean()
    for _ in range(steps):
        output = tracker(source, target, points)
        loss = F.smooth_l1_loss(output["target"], truth)
        optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
    with torch.no_grad():
        final = torch.linalg.vector_norm(tracker(source, target, points)["target"] - truth,
                                         dim=-1).mean()
    return {"steps": steps, "initial_epe_px": float(initial),
            "final_epe_px": float(final),
            "passed": bool(final < .10 and final < initial * .2)}


def ba_probe(device: torch.device) -> dict:
    torch.manual_seed(73)
    points = torch.randn(1, 120, 3, device=device)
    points[..., 2] = points[..., 2].abs() + 3
    k = torch.tensor([[[120., 0., 64.], [0., 118., 48.], [0., 0., 1.]]], device=device)
    truth = se3_exp(torch.tensor([[.08, -.03, .02, .01, -.02, .015]], device=device))
    target, _ = project_points(transform_points(truth, points), k)
    initial = se3_exp(torch.tensor([[.02, .01, -.01, -.01, .01, 0.]], device=device))
    refined, history, _ = bundle_adjust_pair(points, target, k, initial,
                                              iterations=5, damping=1e-4)
    before = torch.linalg.vector_norm(initial[:, :3, 3] - truth[:, :3, 3], dim=-1)
    after = torch.linalg.vector_norm(refined[:, :3, 3] - truth[:, :3, 3], dim=-1)
    reduction = 1 - history[:, -1] / history[:, 0].clamp_min(1e-8)
    return {"reprojection_reduction": float(reduction),
            "translation_error_before_m": float(before),
            "translation_error_after_m": float(after),
            "passed": bool(reduction > .95 and after < before * .2)}


def dynamic_probe(device: torch.device) -> dict:
    torch.manual_seed(74)
    count = 160
    points = torch.randn(1, count, 3, device=device)
    points[..., 2] = points[..., 2].abs() + 3
    k = torch.tensor([[[120., 0., 64.], [0., 120., 48.], [0., 0., 1.]]], device=device)
    truth = se3_exp(torch.tensor([[.07, .01, -.02, .005, -.015, .01]], device=device))
    target, _ = project_points(transform_points(truth, points), k)
    dynamic = torch.zeros(1, count, dtype=torch.bool, device=device)
    dynamic[:, -50:] = True
    target[:, -50:] += torch.tensor([18., -12.], device=device)
    initial = torch.eye(4, device=device).unsqueeze(0)
    all_pose, _, _ = bundle_adjust_pair(points, target, k, initial,
                                        iterations=5, damping=1e-3)
    static_pose, _, _ = bundle_adjust_pair(points, target, k, initial,
                                           (~dynamic).float(), iterations=5,
                                           damping=1e-3)
    all_error = torch.linalg.vector_norm(all_pose[:, :3, 3] - truth[:, :3, 3], dim=-1)
    static_error = torch.linalg.vector_norm(static_pose[:, :3, 3] - truth[:, :3, 3], dim=-1)
    return {"all_tracks_translation_error_m": float(all_error),
            "static_tracks_translation_error_m": float(static_error),
            "passed": bool(static_error < all_error * .25)}


def memory_probe() -> dict:
    memory = BoundedWorldMemory(max_frames=4)
    for index in range(19):
        memory.append(torch.tensor([float(index)]), torch.eye(4))
    return {"frames_inserted": 19, "frames_retained": len(memory.tokens),
            "oldest_retained": float(memory.tokens[0]),
            "passed": len(memory.tokens) == 4 and float(memory.tokens[0]) == 15.0}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/eye_v3_probes/probe_report.json")
    parser.add_argument("--track-steps", type=int, default=120)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    probes = {"camera": camera_probe(), "tracks": track_probe(device, args.track_steps),
              "bundle_adjustment": ba_probe(device), "dynamic_mask": dynamic_probe(device),
              "bounded_memory": memory_probe()}
    report = {"valid": all(row["passed"] for row in probes.values()),
              "device": str(device), "probes": probes}
    path = Path(args.output); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    if not report["valid"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

