# -*- coding: utf-8 -*-
"""Decisive reasoner diagnostic (runs on CPU while GPU trains).

Separates the failure hypotheses with 4 numbers + a visual dump:
  train_tok vs val_tok  -> generalization gap? (memorization / distribution shift)
  train_exact vs val_exact -> does the gap survive teacher forcing?
  + PNG grid of val what_held samples with GT vs prediction (human-inspectable:
    are objects even recognizable at 64px after camera degradation?)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from jwm import JWM, JWMConfig  # noqa: E402
from jwm import tokenizer as tok  # noqa: E402
from jwm.data import pad_text, pad_answers, imgs_to_float  # noqa: E402

CKPT = ROOT / "jwm" / "checkpoints" / "stage_r1_reasoner_pretrain.pt"
N_TOK, N_EXACT = 160, 60


def teacher_forced_tok_acc(model, cfg, d, idx, dev) -> float:
    accs = []
    for i0 in range(0, len(idx), 16):
        ids = idx[i0:i0 + 16]
        img = imgs_to_float(d["img"][ids], dev)
        q_ids, q_valid = pad_text([d["q"][i] for i in ids], cfg.max_q_bytes)
        a_ids, a_valid = pad_answers([d["a"][i] for i in ids], cfg.max_a_bytes)
        with torch.no_grad():
            _, m = model.loss_qa(img, q_ids.to(dev), q_valid.to(dev),
                                 a_ids.to(dev), a_valid.to(dev))
        accs.append(m["qa_tok_acc"])
    return float(np.mean(accs))


def exact_match(model, cfg, d, idx, dev) -> tuple[float, list]:
    ok, preds = 0, []
    for i0 in range(0, len(idx), 12):
        ids = idx[i0:i0 + 12]
        img = imgs_to_float(d["img"][ids], dev)
        q_ids, q_valid = pad_text([d["q"][i] for i in ids], cfg.max_q_bytes)
        with torch.no_grad():
            ans = model.generate_answer(img, q_ids.to(dev), q_valid.to(dev))
        for j, a in enumerate(ans):
            i = ids[j]
            hit = a.strip().lower() == d["a"][i].strip().lower()
            ok += int(hit)
            preds.append({"i": int(i), "q": d["q"][i], "gt": d["a"][i],
                          "pred": a, "ok": hit, "kind": d["meta"][i]["kind"]})
    return ok / len(idx), preds


def main() -> int:
    dev = "cpu"
    payload = torch.load(CKPT, map_location=dev, weights_only=False)
    cfg = JWMConfig(**{k: v for k, v in payload["cfg"].items()
                       if k in JWMConfig.__dataclass_fields__})
    model = JWM(cfg)
    model.load_state_dict(payload["model"])
    model.to(dev).eval()

    train = torch.load(ROOT / "data/jwm_v3/reasoner_pretrain.pt", weights_only=False)["qa"]
    val = torch.load(ROOT / "data/jwm_sdg/val.pt", weights_only=False)["qa"]
    rng = np.random.default_rng(0)
    tr_idx = rng.choice(len(train["q"]), N_TOK, replace=False).tolist()
    va_idx = rng.choice(len(val["q"]), min(N_TOK, len(val["q"])), replace=False).tolist()

    print("teacher-forced token accuracy:")
    tr_tok = teacher_forced_tok_acc(model, cfg, train, tr_idx, dev)
    va_tok = teacher_forced_tok_acc(model, cfg, val, va_idx, dev)
    print(f"  train_tok = {tr_tok:.4f}\n  val_tok   = {va_tok:.4f}  (gap {tr_tok-va_tok:+.4f})")

    print("exact-match (greedy decode):")
    tr_ex, _ = exact_match(model, cfg, train, tr_idx[:N_EXACT], dev)
    va_ex, va_preds = exact_match(model, cfg, val, va_idx[:N_EXACT], dev)
    print(f"  train_exact = {tr_ex:.3f}\n  val_exact   = {va_ex:.3f}")

    # visual dump: val what_held/what failures at native 64px, upscaled for viewing
    fails = [p for p in va_preds if not p["ok"] and p["kind"] in ("what_held", "what", "where")][:10]
    cell = 192
    grid = Image.new("RGB", (cell * 5, (cell + 58) * 2), (8, 12, 20))
    dr = ImageDraw.Draw(grid)
    for k, p in enumerate(fails[:10]):
        r, c = divmod(k, 5)
        im = Image.fromarray(val["img"][p["i"]].numpy()).resize((cell, cell), Image.NEAREST)
        y = r * (cell + 58)
        grid.paste(im, (c * cell, y))
        dr.text((c * cell + 3, y + cell + 2), f"Q:{p['q'][:30]}", fill=(140, 220, 255))
        dr.text((c * cell + 3, y + cell + 20), f"GT:{p['gt'][:30]}", fill=(120, 255, 160))
        dr.text((c * cell + 3, y + cell + 38), f"PR:{p['pred'][:30]}", fill=(255, 140, 140))
    out_png = ROOT / "data" / "diagnose_reasoner_val_fails.png"
    grid.save(out_png)
    json.dump(va_preds, (ROOT / "data" / "diagnose_reasoner_val_preds.json").open("w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(f"saved {out_png.name} + predictions json")

    # verdict hints
    if tr_tok - va_tok > 0.03:
        print("=> GAP LỚN train/val: distribution shift hoặc memorization")
    elif va_tok < 0.97:
        print("=> KHÔNG gap nhưng val_tok thấp: undertrained / nhiệm vụ khó nội tại (ảnh 64px)")
    else:
        print("=> val_tok cao mà exact thấp: vấn đề decode/EOS/format")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
