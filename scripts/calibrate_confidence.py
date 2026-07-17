# -*- coding: utf-8 -*-
"""Post-hoc Platt scaling for the JWM confidence head (critique-doc recipe).

Fits sigmoid(a * logit(conf) + b) on the VAL split against IoU>=0.5 labels,
reports ECE before/after on val AND test, and writes the calibration into the
checkpoint so WorldBrain applies it at inference.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from jwm import JWM, JWMConfig, ConvAE  # noqa: E402
from jwm.mathx import expected_calibration_error  # noqa: E402
from jwm.trainer import eval_ground  # noqa: E402

CKPT = ROOT / "jwm" / "checkpoints" / "jwm_v1.pt"
STEPS = 4  # calibrate for the fast sampler used in deployment


def main() -> int:
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    payload = torch.load(CKPT, map_location=dev, weights_only=False)
    cfg = JWMConfig(**{k: v for k, v in payload["cfg"].items()
                       if k in JWMConfig.__dataclass_fields__})
    model = JWM(cfg)
    model.load_state_dict(payload["model"])
    model.to(dev).eval()
    ae = ConvAE(cfg.z_ch)
    ae.load_state_dict(payload["ae"])
    ae.to(dev).eval()

    def collect(split_name: str):
        split = torch.load(ROOT / "data" / "jwm_sdg" / f"{split_name}.pt", weights_only=False)
        r = eval_ground(model, ae, split, cfg, dev, n=250, steps=STEPS, log=lambda *a: None)
        return (torch.tensor(r["conf_all"]).clamp(1e-4, 1 - 1e-4),
                torch.tensor(r["iou_all"]) >= 0.5)

    print(f"collecting val/test confidence at steps={STEPS} ...")
    conf_v, y_v = collect("val")
    conf_t, y_t = collect("test")

    # Platt scaling on VAL: sigmoid(a * logit(c) + b)
    logit_v = torch.log(conf_v / (1 - conf_v))
    a = torch.tensor(1.0, requires_grad=True)
    b = torch.tensor(0.0, requires_grad=True)
    opt = torch.optim.LBFGS([a, b], lr=0.1, max_iter=200)

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            a * logit_v + b, y_v.float())
        loss.backward()
        return loss

    opt.step(closure)
    a_f, b_f = float(a), float(b)

    def apply_cal(c):
        lg = torch.log(c / (1 - c))
        return torch.sigmoid(a_f * lg + b_f)

    report = {
        "platt": {"a": round(a_f, 4), "b": round(b_f, 4), "steps": STEPS},
        "val": {"ece_before": round(expected_calibration_error(conf_v, y_v), 4),
                "ece_after": round(expected_calibration_error(apply_cal(conf_v), y_v), 4),
                "base_rate": round(float(y_v.float().mean()), 4)},
        "test": {"ece_before": round(expected_calibration_error(conf_t, y_t), 4),
                 "ece_after": round(expected_calibration_error(apply_cal(conf_t), y_t), 4),
                 "base_rate": round(float(y_t.float().mean()), 4)},
    }
    print(json.dumps(report, indent=1))

    payload["calibration"] = report["platt"]
    payload.setdefault("metrics", {})["calibration"] = report
    torch.save(payload, CKPT)
    metrics_path = ROOT / "jwm" / "checkpoints" / "metrics_v1.json"
    metrics = json.load(metrics_path.open(encoding="utf-8"))
    metrics["calibration"] = report
    json.dump(metrics, metrics_path.open("w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("calibration written into jwm_v1.pt + metrics_v1.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
