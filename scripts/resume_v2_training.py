# -*- coding: utf-8 -*-
"""Resume JWM v2 training after a shutdown — headless, shutdown-safe, one command:

    python scripts/resume_v2_training.py

Loads jwm/checkpoints/jwm_v2_partial.pt, continues the remaining curriculum
(pretrain -> SFT), checkpoints every 500 steps (safe to shut down again any
time), then runs full evaluation + Platt calibration and writes jwm_v2.pt +
metrics_v2.json. Datasets are read from data/jwm_sdg/ (already on disk).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from jwm import JWM, JWMConfig, ConvAE  # noqa: E402
from jwm.mathx import expected_calibration_error  # noqa: E402
from jwm.sdg import CameraParams  # noqa: E402
from jwm.trainer import train_stage, eval_qa, eval_ground, eval_fd, measure_latency  # noqa: E402

CKPT_DIR = ROOT / "jwm" / "checkpoints"
PARTIAL = CKPT_DIR / "jwm_v2_partial.pt"
FINAL = CKPT_DIR / "jwm_v2.pt"

# The first partial was saved from an interrupted run whose 'approx_step' field
# is unreliable; the true progress at that save was ~700/6000 pretrain steps.
FALLBACK_PRETRAIN_DONE = 700
TOTAL_PRETRAIN = 6000
TOTAL_SFT = 1500


def main() -> int:
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"== Resume JWM v2 on {dev} ==")
    payload = torch.load(PARTIAL, map_location=dev, weights_only=False)
    cfg = JWMConfig(**{k: v for k, v in payload["cfg"].items()
                       if k in JWMConfig.__dataclass_fields__})
    model = JWM(cfg)
    model.load_state_dict(payload["model"])
    model.to(dev)
    ae = ConvAE(cfg.z_ch)
    ae.load_state_dict(payload["ae"])
    ae.to(dev).eval()
    for p in ae.parameters():
        p.requires_grad_(False)
    cam = CameraParams(**payload["camera"])

    resume = payload.get("resume", {}) or {}
    pretrain_done = int(resume.get("pretrain_done", resume.get("approx_step") or 0)
                        or FALLBACK_PRETRAIN_DONE)
    if pretrain_done <= 0:
        pretrain_done = FALLBACK_PRETRAIN_DONE
    sft_done = int(resume.get("sft_done", 0))
    print(f"   model {model.num_params()/1e6:.2f}M | pretrain_done={pretrain_done} "
          f"sft_done={sft_done}")

    splits = {name: torch.load(ROOT / "data" / "jwm_sdg" / f"{name}.pt", weights_only=False)
              for name in ("pretrain_v2", "sft_v2", "val", "test")}

    def save_partial(stage: str, base: int):
        def fn(done: int):
            info = {"pretrain_done": pretrain_done, "sft_done": sft_done}
            info[f"{stage}_done"] = base + done
            torch.save({"model": model.state_dict(), "ae": ae.state_dict(),
                        "cfg": cfg.__dict__, "camera": cam.to_dict(),
                        "resume": {**info, "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S")}},
                       PARTIAL)
        return fn

    # ---- Stage 1 remainder: pretrain (ground-heavy) ----
    remaining = max(0, TOTAL_PRETRAIN - pretrain_done)
    if remaining:
        print(f"-- pretrain remainder: {remaining} steps --")
        train_stage(model, ae, splits["pretrain_v2"], cfg, dev,
                    steps=remaining, lr=3e-4, batch_size=48, warmup=100,
                    mode_probs={"qa": 0.30, "ground": 0.50, "fd": 0.20},
                    seed=42 + pretrain_done, log_every=200,
                    ckpt_fn=save_partial("pretrain", pretrain_done), ckpt_every=500)
        pretrain_done = TOTAL_PRETRAIN

    # ---- Stage 2 remainder: SFT ----
    remaining_sft = max(0, TOTAL_SFT - sft_done)
    if remaining_sft:
        print(f"-- SFT remainder: {remaining_sft} steps --")
        train_stage(model, ae, splits["sft_v2"], cfg, dev,
                    steps=remaining_sft, lr=8e-5, batch_size=48, warmup=80,
                    mode_probs={"qa": 0.25, "ground": 0.55, "fd": 0.20},
                    seed=43 + sft_done, log_every=200,
                    ckpt_fn=save_partial("sft", sft_done), ckpt_every=500)
        sft_done = TOTAL_SFT

    # ---- Evaluation ----
    print("-- eval val --")
    qa_v = eval_qa(model, splits["val"], cfg, dev, n=250)
    gr_v50 = eval_ground(model, ae, splits["val"], cfg, dev, n=250, steps=50)
    gr_v4 = eval_ground(model, ae, splits["val"], cfg, dev, n=250, steps=4)
    fd_v = eval_fd(model, ae, splits["val"], cfg, dev, n=48, k=4, steps=50)

    # Platt calibration on val @ 4 steps
    conf_v = torch.tensor(gr_v4["conf_all"]).clamp(1e-4, 1 - 1e-4)
    y_v = torch.tensor(gr_v4["iou_all"]) >= 0.5
    logit_v = torch.log(conf_v / (1 - conf_v))
    a = torch.tensor(1.0, requires_grad=True)
    b = torch.tensor(0.0, requires_grad=True)
    opt = torch.optim.LBFGS([a, b], lr=0.1, max_iter=200)

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(a * logit_v + b, y_v.float())
        loss.backward()
        return loss

    opt.step(closure)
    a_f, b_f = float(a), float(b)
    cal = lambda c: torch.sigmoid(a_f * torch.log(c / (1 - c)) + b_f)  # noqa: E731
    CAL = {"a": round(a_f, 4), "b": round(b_f, 4), "steps": 4}
    print(f"   Platt a={a_f:.3f} b={b_f:.3f} | val ECE "
          f"{expected_calibration_error(conf_v, y_v):.4f} -> "
          f"{expected_calibration_error(cal(conf_v), y_v):.4f}")

    print("-- eval test --")
    qa_t = eval_qa(model, splits["test"], cfg, dev, n=250)
    gr_t50 = eval_ground(model, ae, splits["test"], cfg, dev, n=250, steps=50)
    gr_t4 = eval_ground(model, ae, splits["test"], cfg, dev, n=250, steps=4)
    fd_t = eval_fd(model, ae, splits["test"], cfg, dev, n=48, k=4, steps=50)
    conf_t = torch.tensor(gr_t4["conf_all"]).clamp(1e-4, 1 - 1e-4)
    y_t = torch.tensor(gr_t4["iou_all"]) >= 0.5
    ece_t = expected_calibration_error(cal(conf_t), y_t)
    lat = measure_latency(model, ae, splits["val"], cfg, dev, reps=15)

    metrics = {
        "v1_test_reference": {"qa_acc": 0.564, "miou": 0.201, "iou_at_05_4step": 0.184,
                              "ece_cal": 0.0396, "fd_psnr": 20.10, "fd_copy": 21.39,
                              "params_M": 10.74},
        "val": {"qa_acc": qa_v["acc"], "iou_at_05": gr_v50["iou_at_05"],
                "miou_50": gr_v50["miou"], "miou_4step": gr_v4["miou"],
                "iou_at_05_4step": gr_v4["iou_at_05"],
                "fd_psnr": fd_v["fd_psnr_min_over_k"], "fd_copy": fd_v["fd_psnr_copy_baseline"]},
        "test": {"qa_acc": qa_t["acc"], "qa_per_kind": qa_t["per_kind"],
                 "iou_at_05": gr_t50["iou_at_05"], "miou": gr_t50["miou"],
                 "iou_at_05_4step": gr_t4["iou_at_05"], "miou_4step": gr_t4["miou"],
                 "ece_calibrated_4step": ece_t,
                 "fd_psnr": fd_t["fd_psnr_min_over_k"], "fd_copy": fd_t["fd_psnr_copy_baseline"],
                 "fd_beats_copy": fd_t["beats_copy_frac"]},
        "calibration": CAL,
        "latency_ms": lat,
        "params_M": model.num_params() / 1e6,
        "train": {"pretrain_steps": TOTAL_PRETRAIN, "sft_steps": TOTAL_SFT,
                  "resumed": True},
    }
    torch.save({"model": model.state_dict(), "ae": ae.state_dict(), "cfg": cfg.__dict__,
                "camera": cam.to_dict(), "calibration": CAL, "metrics": metrics}, FINAL)
    json.dump(metrics, (CKPT_DIR / "metrics_v2.json").open("w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("\n== DONE ==")
    print(json.dumps(metrics["test"], ensure_ascii=False, indent=1))
    print(f"Saved {FINAL.name} + metrics_v2.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
