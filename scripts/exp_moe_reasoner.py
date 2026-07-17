# -*- coding: utf-8 -*-
"""Day-2 controlled experiment: Inkling-mini MoE reasoner vs v3 dense baseline.

ONE variable changed (reasoner FFN dense -> MoE); everything else identical to
the round-3 pipeline that produced the v3 baseline (r1 54.5% / r2 57.2%):
same data, LR, batch 48, steps, seeds. Checkpoints go to exp_moe_* names so the
shipped v3 pipeline artifacts are untouched. Shutdown-safe every 500 steps.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from jwm import JWM  # noqa: E402
from jwm.configs import pipeline_scale_moe  # noqa: E402
from jwm.model import ConvAE  # noqa: E402
from jwm.stages.common import DEV, load_data, load_valtest, subset  # noqa: E402
from jwm.trainer import train_stage, eval_qa  # noqa: E402

CKPT = ROOT / "jwm" / "checkpoints" / "exp_moe_reasoner.pt"
REPORT = ROOT / "jwm" / "checkpoints" / "exp_moe_reasoner.report.json"
BASELINE = {"r1": 0.545, "r2": 0.572}          # v3 dense, same data/recipe

R1_STEPS, R2_STEPS = 3000, 800


def save(model, cfg, extra):
    torch.save({"model": model.state_dict(), "cfg": cfg.__dict__, **extra}, CKPT)


def main() -> int:
    t0 = time.time()
    cfg = pipeline_scale_moe()
    done = {"r1": 0, "r2": 0}
    model = JWM(cfg)
    if CKPT.exists():
        payload = torch.load(CKPT, map_location=DEV, weights_only=False)
        model.load_state_dict(payload["model"])
        done = payload.get("done", done)
        print(f"RESUME: {done}")
    model.to(DEV)
    ae = ConvAE(cfg.z_ch).to(DEV)                 # unused by qa; API requirement
    print(f"Inkling-mini reasoner: {model.num_params()/1e6:.2f}M total")

    val, _ = load_valtest()
    report = {"baseline_dense_v3": BASELINE, "params_M": model.num_params() / 1e6}

    # ---- r1-equivalent ----
    if done["r1"] < R1_STEPS:
        split = load_data("reasoner_pretrain")
        def ck1(n):
            save(model, cfg, {"done": {"r1": done["r1"] + n, "r2": 0}})
        train_stage(model, ae, split, cfg, DEV, steps=R1_STEPS - done["r1"], lr=3e-4,
                    batch_size=48, warmup=max(20, 150 - done["r1"]),
                    mode_probs={"qa": 1.0}, seed=100 + done["r1"], log_every=200,
                    ckpt_fn=ck1, ckpt_every=500)
        done["r1"] = R1_STEPS
    qa1 = eval_qa(model, val, cfg, DEV, n=250)
    report["r1_moe"] = {"val_qa_acc": qa1["acc"], "per_kind": qa1["per_kind"]}
    print(f"== r1-MoE: {qa1['acc']:.3f} (dense baseline {BASELINE['r1']}) ==")
    save(model, cfg, {"done": done, "report": report})

    # ---- r2-equivalent (SFT + 1:4 pretrain mix) ----
    if done["r2"] < R2_STEPS:
        import random
        sft = load_data("reasoner_sft")["qa"]
        pre = load_data("reasoner_pretrain")["qa"]
        rng = random.Random(0)
        n_mix = int(len(sft["q"]) * 0.2 / 0.8)
        pre_mix = subset(pre, rng.sample(range(len(pre["q"])), n_mix))
        merged = {k: (torch.cat([sft[k], pre_mix[k]]) if torch.is_tensor(sft[k])
                      else sft[k] + pre_mix[k]) for k in sft}
        def ck2(n):
            save(model, cfg, {"done": {"r1": R1_STEPS, "r2": done["r2"] + n}})
        train_stage(model, ae, {"qa": merged}, cfg, DEV, steps=R2_STEPS - done["r2"],
                    lr=1.2e-4, batch_size=48, warmup=max(10, 50 - done["r2"]),
                    mode_probs={"qa": 1.0}, seed=200 + done["r2"], log_every=200,
                    ckpt_fn=ck2, ckpt_every=250)
        done["r2"] = R2_STEPS
    qa2 = eval_qa(model, val, cfg, DEV, n=250)
    report["r2_moe"] = {"val_qa_acc": qa2["acc"], "per_kind": qa2["per_kind"]}

    # router health after full training
    from jwm.moe import MoEFFN
    from jwm.data import imgs_to_float, pad_text
    d = val["qa"]
    img = imgs_to_float(d["img"][:32], DEV)
    q_ids, q_valid = pad_text(d["q"][:32], cfg.max_q_bytes)
    emb, coords, valid = model._build_ar(img, q_ids.to(DEV), q_valid.to(DEV), 260)
    health = {}
    h = emb
    from jwm.mathx import mrope_angles
    from jwm.layers import build_ar_mask
    ang = mrope_angles(coords, cfg.rope_sections, cfg.rope_base)
    mask = build_ar_mask(valid)
    with torch.no_grad():
        for i, blk in enumerate(model.blocks):
            if isinstance(blk.r_ffn, MoEFFN):
                s = blk.r_ffn.routing_stats(blk.r_norm2(h))
                health[f"layer{i}"] = {"entropy": round(s["entropy"], 3),
                                       "max_entropy": round(s["max_entropy"], 3),
                                       "dead": s["dead_experts"]}
            h, _, _ = blk.forward_ar(h, ang, mask)
    report["router_health"] = health

    verdict = "WIN" if qa2["acc"] > BASELINE["r2"] else "LOSS"
    report["verdict"] = verdict
    report["minutes"] = round((time.time() - t0) / 60, 1)
    save(model, cfg, {"done": done, "report": report})
    json.dump(report, REPORT.open("w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"== r2-MoE: {qa2['acc']:.3f} (dense baseline {BASELINE['r2']}) -> {verdict} ==")
    print(json.dumps(report["router_health"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
