"""Independent Day-5 benchmark for the JWM physical-eye checkpoint.

The benchmark deliberately separates capability from image dependence:

* held-out procedural test scenes use seeds disjoint from train/validation;
* black, frozen-frame, reversed-time and wrong-scene controls keep the same
  depth/pose targets while corrupting only the visual evidence;
* an identity-pose/constant-depth baseline guards against easy dataset priors;
* optional TUM roots provide an external real RGB-D benchmark.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jwm.config import JWMConfig
from jwm.geometry_data import render_geometry_sequence
from jwm.geometry_trainer import geometry_batch_metrics
from jwm.model import JWM
from jwm.tum_rgbd import TUMRGBDWindowDataset, validate_tum_dataset


METRICS = ("depth_abs_rel", "depth_rmse_anchor_scale", "ate_anchor_scale",
           "abs_rotation_deg", "rpe_translation", "rpe_rotation_deg")


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--stage0")
    parser.add_argument("--tum", nargs="*", default=[])
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--tum-samples", type=int, default=24)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def stack(row: dict) -> dict:
    return {key: value.unsqueeze(0) if torch.is_tensor(value) else value
            for key, value in row.items()}


def move(batch: dict, device: torch.device) -> dict:
    return {key: value.to(device) if torch.is_tensor(value) else value
            for key, value in batch.items()}


def average(rows: list[dict[str, float]]) -> dict[str, float]:
    return {key: sum(row[key] for row in rows) / len(rows) for key in METRICS}


def procedural_row(seed: int, cfg: JWMConfig, control: str) -> dict:
    row = render_geometry_sequence(seed, frames=8,
                                   height=cfg.input_height,
                                   width=cfg.input_width).as_dict()
    image = row["image"]
    if control == "black":
        image = torch.zeros_like(image)
    elif control == "mean_only":
        image = image.mean(dim=(-2, -1), keepdim=True).expand_as(image).clone()
    elif control == "frozen_first":
        image = image[:1].expand_as(image).clone()
    elif control == "reverse_time":
        image = image.flip(0)
    elif control == "wrong_scene":
        image = render_geometry_sequence(
            seed + 500_000, frames=8, height=cfg.input_height,
            width=cfg.input_width).image
    elif control != "normal":
        raise ValueError(control)
    row["image"] = image
    return row


class ConstantPrior(torch.nn.Module):
    """No-image baseline in the same normalized depth/pose coordinates."""

    def __init__(self, cfg: JWMConfig):
        super().__init__()
        self.cfg = cfg

    def encode_geometry_sequence(self, images: torch.Tensor,
                                 detach_state: bool = True) -> dict:
        b, t = images.shape[:2]
        n = self.cfg.img_grid_h * self.cfg.img_grid_w
        depth = torch.ones(b, t, n, device=images.device)
        pose = torch.eye(4, device=images.device).view(1, 1, 4, 4).repeat(b, t, 1, 1)
        return {"depth_tokens": depth, "pose_c2w": pose}


def load_model(path: str | Path, device: torch.device):
    blob = torch.load(path, map_location="cpu", weights_only=False)
    cfg = JWMConfig(**blob["cfg"])
    model = JWM(cfg)
    model.load_state_dict(blob["model"], strict=True)
    model.eval().to(device)
    return model, cfg, blob


@torch.inference_mode()
def evaluate_rows(model, rows, device: torch.device) -> tuple[dict, float]:
    results = []
    if device.type == "cuda":
        torch.cuda.synchronize()
    started = time.perf_counter()
    for row in rows:
        with torch.autocast(device_type=device.type, dtype=torch.float16,
                            enabled=device.type == "cuda"):
            results.append(geometry_batch_metrics(model, move(stack(row), device)))
    if device.type == "cuda":
        torch.cuda.synchronize()
    seconds = time.perf_counter() - started
    return average(results), seconds


def ratio(bad: float, good: float) -> float:
    return bad / max(good, 1e-12)


def markdown(report: dict) -> str:
    labels = {
        "depth_abs_rel": "Depth AbsRel",
        "depth_rmse_anchor_scale": "Depth RMSE",
        "ate_anchor_scale": "ATE",
        "abs_rotation_deg": "Abs rotation (deg)",
        "rpe_translation": "RPE translation",
        "rpe_rotation_deg": "RPE rotation (deg)",
    }
    lines = ["# JWM Eye Physical v1 benchmark", "",
             "All metrics are lower-is-better.", "",
             "| Evaluation | " + " | ".join(labels[x] for x in METRICS) + " |",
             "|---|" + "---:|" * len(METRICS)]
    for name, block in report["evaluations"].items():
        values = block["metrics"]
        lines.append("| " + name + " | " +
                     " | ".join(f"{values[x]:.5f}" for x in METRICS) + " |")
    lines += ["", "## Image-dependence verdict", "",
              f"- Wrong-scene Depth AbsRel ratio: {report['vision_dependence']['wrong_scene_depth_ratio']:.3f}x",
              f"- Wrong-scene ATE ratio: {report['vision_dependence']['wrong_scene_ate_ratio']:.3f}x",
              f"- Black-image Depth AbsRel ratio: {report['vision_dependence']['black_depth_ratio']:.3f}x",
              f"- Visual evidence gate: **{report['vision_dependence']['gate']}**"]
    return "\n".join(lines) + "\n"


def main():
    args = arguments()
    device = torch.device(args.device)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    final, cfg, blob = load_model(args.checkpoint, device)
    seeds = [2_100_000 + index for index in range(args.samples)]
    evaluations = {}

    controls = ("normal", "black", "mean_only", "frozen_first",
                "reverse_time", "wrong_scene")
    for control in controls:
        rows = [procedural_row(seed, cfg, control) for seed in seeds]
        metrics, seconds = evaluate_rows(final, rows, device)
        evaluations[f"procedural/{control}"] = {
            "metrics": metrics, "samples": len(rows), "seconds": seconds,
            "sequence_per_second": len(rows) / seconds,
            "frame_per_second": len(rows) * 8 / seconds,
        }
        print(control, json.dumps(evaluations[f"procedural/{control}"], indent=2))

    prior = ConstantPrior(cfg).to(device)
    prior_rows = [procedural_row(seed, cfg, "normal") for seed in seeds]
    metrics, seconds = evaluate_rows(prior, prior_rows, device)
    evaluations["procedural/constant_identity_prior"] = {
        "metrics": metrics, "samples": len(prior_rows), "seconds": seconds,
    }

    if args.stage0:
        del final
        if device.type == "cuda":
            torch.cuda.empty_cache()
        stage0, stage0_cfg, _ = load_model(args.stage0, device)
        if stage0_cfg != cfg:
            raise RuntimeError("stage0 and final configurations differ")
        stage0_rows = [procedural_row(seed, cfg, "normal") for seed in seeds]
        metrics, seconds = evaluate_rows(stage0, stage0_rows, device)
        evaluations["procedural/stage0_normal"] = {
            "metrics": metrics, "samples": len(stage0_rows), "seconds": seconds,
        }
        del stage0
        if device.type == "cuda":
            torch.cuda.empty_cache()
        final, _, _ = load_model(args.checkpoint, device)

    tum_report = None
    if args.tum:
        dataset = TUMRGBDWindowDataset(args.tum, frames=8, frame_stride=3,
                                       window_stride=24,
                                       height=cfg.input_height,
                                       width=cfg.input_width)
        tum_report = validate_tum_dataset(dataset)
        count = min(args.tum_samples, len(dataset))
        indices = [math.floor(i * len(dataset) / count) for i in range(count)] if count else []
        rows = [dataset[index] for index in indices]
        for control in ("normal", "black", "frozen_first", "reverse_time",
                        "wrong_window"):
            controlled = []
            for row_index, row in enumerate(rows):
                clone = dict(row)
                if control == "black":
                    clone["image"] = torch.zeros_like(row["image"])
                elif control == "frozen_first":
                    clone["image"] = row["image"][:1].expand_as(row["image"]).clone()
                elif control == "reverse_time":
                    clone["image"] = row["image"].flip(0)
                elif control == "wrong_window":
                    wrong_index = (row_index + max(1, len(rows) // 2)) % len(rows)
                    clone["image"] = rows[wrong_index]["image"]
                controlled.append(clone)
            metrics, seconds = evaluate_rows(final, controlled, device)
            evaluations[f"tum/{control}"] = {
                "metrics": metrics, "samples": len(controlled), "seconds": seconds,
            }
        metrics, seconds = evaluate_rows(prior, rows, device)
        evaluations["tum/constant_identity_prior"] = {
            "metrics": metrics, "samples": len(rows), "seconds": seconds,
        }

    normal = evaluations["procedural/normal"]["metrics"]
    wrong = evaluations["procedural/wrong_scene"]["metrics"]
    black = evaluations["procedural/black"]["metrics"]
    wrong_depth_ratio = ratio(wrong["depth_abs_rel"], normal["depth_abs_rel"])
    wrong_ate_ratio = ratio(wrong["ate_anchor_scale"], normal["ate_anchor_scale"])
    black_depth_ratio = ratio(black["depth_abs_rel"], normal["depth_abs_rel"])
    dependence_gate = "PASS" if (wrong_depth_ratio > 1.25 and
                                  wrong_ate_ratio > 1.25 and
                                  black_depth_ratio > 1.25) else "FAIL"
    report = {
        "benchmark": "JWM-Eye-Physical Independent Geometry + Blind Controls",
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_status": blob.get("status", "final"),
        "device": str(device),
        "evaluations": evaluations,
        "vision_dependence": {
            "wrong_scene_depth_ratio": wrong_depth_ratio,
            "wrong_scene_ate_ratio": wrong_ate_ratio,
            "black_depth_ratio": black_depth_ratio,
            "gate": dependence_gate,
        },
        "tum_dataset_validation": tum_report,
    }
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    output.with_suffix(".md").write_text(markdown(report), encoding="utf-8")
    print(f"saved {output}")


if __name__ == "__main__":
    main()
