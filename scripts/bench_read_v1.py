"""Benchmark harness for jwm_read_v1 — find the model's dead points.

Tiers:
  A. Synthetic diagnostic ladder (char -> word size sweep -> line -> paragraph,
     +camera degrade) — locates the capability cliff
  B. Real doc pages (held-out slice of viet_doc_reasoning, read from tars)
  C. MTVQA Vietnamese test subset (true OOD scene/doc photos)

Metrics per tier: mean/median CER, exact, containment, stop-rate (hit max_new
without EOS), pred/ref length. Writes data/bench_read_v1.json + prints a table.
"""

from __future__ import annotations

import io
import json
import os
import random
import statistics
import sys
import tarfile
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from jwm.config import JWMConfig
from jwm.data import pad_text
from jwm.mathx import char_error_rate
from jwm.model import JWM
from jwm.read_data import (READ_QUESTIONS, find_fonts, letterbox,
                           load_doc_pairs, render_read_sample)
from jwm.sdg import CameraParams, camera_degrade

import argparse

_ap = argparse.ArgumentParser()
_ap.add_argument("--ckpt", default=str(ROOT / "jwm_read_v1.pt"))
_ap.add_argument("--tag", default="v1")
_ARGS, _ = _ap.parse_known_args()

CKPT = Path(_ARGS.ckpt)
VDOC = Path(os.environ.get("JWM_VDOC", ROOT / "data" / "viet_doc_reasoning"))
OUT_JSON = ROOT / "data" / f"bench_read_{_ARGS.tag}.json"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SIZE = 768
BATCH = 8

_WORDS = ("máy tính điện thoại bàn ghế cửa sổ quyển sách cây bút màn hình "
          "bàn phím con chuột tài liệu trang giấy dòng chữ").split()


# ---------------------------------------------------------------- rendering

def render_word(rng: random.Random, text: str, px: int, fonts: list[str],
                cam=None) -> np.ndarray:
    bg = rng.randint(235, 252)
    img = Image.new("RGB", (SIZE, SIZE), (bg, bg, bg))
    d = ImageDraw.Draw(img)
    font = ImageFont.truetype(rng.choice(fonts), px)
    ink = rng.randint(10, 60)
    d.text((rng.randint(30, 90), rng.randint(int(SIZE * 0.25), int(SIZE * 0.55))),
           text, font=font, fill=(ink, ink, ink))
    if cam is not None:
        img = camera_degrade(img, cam, rng, out_size=SIZE)
    return np.asarray(img, dtype=np.uint8)


def ladder_samples(fonts, cam):
    """(tier_name, image, ref) triplets. 12 per tier, seeded."""
    out = []
    rng = random.Random(2026)
    chars = list("AEKMRTVXĐ") + list("035789")
    for _ in range(12):
        c = rng.choice(chars)
        out.append(("T0_char_200px", render_word(rng, c, 200, fonts), c))
    for px in (120, 80, 48, 28):
        for _ in range(12):
            w = rng.choice(_WORDS)
            out.append((f"T1_word_{px}px", render_word(rng, w, px, fonts), w))
    for _ in range(12):
        arr, ref = render_read_sample(rng, 2, fonts, [], size=SIZE)
        out.append(("T2_line_L2", arr, ref))
    for _ in range(12):
        arr, ref = render_read_sample(rng, 4, fonts, [], size=SIZE)
        out.append(("T3_para_L4", arr, ref))
    for _ in range(12):
        w = rng.choice(_WORDS)
        out.append(("T4_word80_degraded", render_word(rng, w, 80, fonts, cam), w))
    return out


# ---------------------------------------------------------------- data tiers

def doc_samples(n=40):
    pairs = load_doc_pairs(str(VDOC / "data" / "vdoc.jsonl"), "", log=lambda *a: None)
    pool = pairs[:400]                       # same held-out slice as Kaggle eval
    picks = random.Random(999).sample(pool, n)
    # member-name -> tar index (tar member names use forward slashes)
    tars = sorted((VDOC / "shards").glob("images-*.tar"))
    handles = [tarfile.open(t) for t in tars]
    index = {}
    for h in handles:
        for m in h.getmembers():
            index[m.name] = (h, m)
    out = []
    for p in picks:
        member = p["img"].replace("\\", "/").lstrip("/")
        hit = index.get(member)
        if not hit:
            continue
        h, m = hit
        img = Image.open(io.BytesIO(h.extractfile(m).read())).convert("RGB")
        out.append(("B_doc_real", np.asarray(letterbox(img, SIZE), dtype=np.uint8),
                    p["a"], p["q"]))
    return out


def mtvqa_samples(n=50):
    from datasets import load_dataset
    ds = load_dataset("ByteDance/MTVQA", split="test")
    vi = [i for i, l in enumerate(ds["lang"]) if str(l).upper().startswith("VI")]
    picks = random.Random(7).sample(vi, min(n, len(vi)))
    out = []
    for i in picks:
        row = ds[i]
        raw = row["qa_pairs"]
        if isinstance(raw, str):
            try:
                qa = json.loads(raw)
            except Exception:
                import ast
                qa = ast.literal_eval(raw)      # python-repr with single quotes
        else:
            qa = raw
        if not qa:
            continue
        q, a = qa[0]["question"], qa[0]["answer"]
        img = row["image"].convert("RGB")
        out.append(("C_mtvqa_vi", np.asarray(letterbox(img, SIZE), dtype=np.uint8),
                    str(a), str(q)))
    return out


# ---------------------------------------------------------------- evaluation

@torch.no_grad()
def run_tier(model, cfg, samples, max_new, log):
    """samples: (tier, img, ref[, q]). Returns per-tier metric dicts + rows."""
    rows = []
    for i0 in range(0, len(samples), BATCH):
        chunk = samples[i0:i0 + BATCH]
        img = torch.from_numpy(np.stack([s[1] for s in chunk])) \
            .permute(0, 3, 1, 2).float().div(255.0).to(DEVICE)
        qs = [s[3] if len(s) > 3 else READ_QUESTIONS[0] for s in chunk]
        q_ids, q_valid = pad_text(qs, cfg.max_q_bytes)
        t0 = time.perf_counter()
        with torch.autocast("cuda", dtype=torch.float16, enabled=DEVICE == "cuda"):
            preds = model.generate_answer(img, q_ids.to(DEVICE), q_valid.to(DEVICE),
                                          max_new=max_new)
        dt = (time.perf_counter() - t0) / len(chunk)
        for s, pred in zip(chunk, preds):
            ref = " ".join(s[2].split())
            pd = " ".join(pred.split())
            rl, pl = ref.lower(), pd.lower()
            rows.append({
                "tier": s[0], "ref": ref, "pred": pd,
                "cer": char_error_rate(pd, ref),
                "exact": pl == rl,
                "contains": bool(rl and rl in pl),
                "stopped": len(pred.encode("utf-8")) < max_new - 1,
                "sec": dt,
            })
        log(f"  {samples[i0][0]}: {min(i0 + BATCH, len(samples))}/{len(samples)} "
            f"({dt:.1f}s/sample)")
    return rows


def summarize(rows):
    tiers = {}
    for r in rows:
        tiers.setdefault(r["tier"], []).append(r)
    out = {}
    for t, rs in sorted(tiers.items()):
        cers = [r["cer"] for r in rs]
        out[t] = {
            "n": len(rs),
            "cer_mean": round(statistics.mean(cers), 3),
            "cer_median": round(statistics.median(cers), 3),
            "exact": round(sum(r["exact"] for r in rs) / len(rs), 3),
            "contains": round(sum(r["contains"] for r in rs) / len(rs), 3),
            "stop_rate": round(sum(r["stopped"] for r in rs) / len(rs), 3),
            "sec_per_sample": round(statistics.mean(r["sec"] for r in rs), 1),
        }
    return out


def main():
    log = print
    log(f"device={DEVICE}")
    blob = torch.load(CKPT, map_location="cpu", weights_only=False)
    cfg = JWMConfig(**blob["cfg"])
    model = JWM(cfg)
    model.load_state_dict(blob["model"])
    model.eval().to(DEVICE)
    fonts = find_fonts()
    cam = CameraParams(noise_std=7.2, blur_sigma=0.54, jpeg_q=55,
                       contrast=1.25, wb_shift=6.0, vignette=0.12)

    all_rows = []
    log("== Tier A: synthetic ladder ==")
    all_rows += run_tier(model, cfg, ladder_samples(fonts, cam), max_new=64, log=log)
    log("== Tier B: real doc pages (held-out) ==")
    all_rows += run_tier(model, cfg, doc_samples(40), max_new=160, log=log)
    log("== Tier C: MTVQA-VI ==")
    try:
        all_rows += run_tier(model, cfg, mtvqa_samples(50), max_new=64, log=log)
    except Exception as e:
        log(f"MTVQA tier failed: {e}")

    summary = summarize(all_rows)
    OUT_JSON.parent.mkdir(exist_ok=True)
    OUT_JSON.write_text(json.dumps({"summary": summary, "rows": all_rows},
                                   ensure_ascii=False, indent=1), encoding="utf-8")
    log("\n=== SUMMARY ===")
    log(f"{'tier':22s} {'n':>3} {'CERm':>6} {'CERmd':>6} {'exact':>6} "
        f"{'cont':>5} {'stop':>5} {'s/smp':>6}")
    for t, s in summary.items():
        log(f"{t:22s} {s['n']:3d} {s['cer_mean']:6.2f} {s['cer_median']:6.2f} "
            f"{s['exact']:6.2f} {s['contains']:5.2f} {s['stop_rate']:5.2f} "
            f"{s['sec_per_sample']:6.1f}")
    log(f"\nwrote {OUT_JSON}")


if __name__ == "__main__":
    main()
