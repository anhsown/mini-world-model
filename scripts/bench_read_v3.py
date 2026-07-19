"""Dimension-correct benchmark for JWM-Read v3 checkpoints.

The legacy Reader benchmark renders square 768x768 inputs.  Reader v3 uses a
1024x768 canvas and a Vietnamese character tokenizer, so silently reusing the
old harness measures an out-of-contract input.  This script keeps the same
capability ladder (glyph, word-size sweep, line, paragraph, degraded word,
real documents and MTVQA-VI) while materializing every sample with the shape
and tokenizer stored in the checkpoint.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import random
import statistics
import sys
import tarfile
import time
import unicodedata
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from jwm.config import JWMConfig
from jwm.data import pad_answers, pad_text
from jwm.mathx import char_error_rate
from jwm.model import JWM
from jwm.read_data import READ_QUESTIONS, find_fonts, load_doc_pairs
from jwm.read_v3_data import letterbox_hw, render_read_v3_sample
from jwm.read_v3_trainer import xyxy_iou
from jwm.sdg import CameraParams, camera_degrade


WORDS = (
    "máy tính điện thoại bàn ghế cửa sổ quyển sách cây bút màn hình "
    "bàn phím con chuột tài liệu trang giấy dòng chữ hóa đơn biểu mẫu"
).split()


def norm(s: str) -> str:
    return " ".join(unicodedata.normalize("NFC", s).lower().split())


def render_text(rng: random.Random, text: str, px: int, fonts: list[str],
                cfg: JWMConfig, cam=None):
    h, w = cfg.input_height, cfg.input_width
    bg = rng.randint(235, 252)
    img = Image.new("RGB", (w, h), (bg, bg, bg))
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype(rng.choice(fonts), px)
    bb0 = draw.textbbox((0, 0), text, font=font)
    tw, th = bb0[2] - bb0[0], bb0[3] - bb0[1]
    x = rng.randint(24, max(24, w - tw - 24))
    y = rng.randint(max(24, h // 4), max(25, min(h - th - 24, 3 * h // 5)))
    draw.text((x, y), text, font=font, fill=(rng.randint(8, 55),) * 3)
    bb = draw.textbbox((x, y), text, font=font)
    box = np.asarray([bb[0] / w, bb[1] / h, bb[2] / w, bb[3] / h], np.float32)
    if cam is not None:
        img = camera_degrade(img, cam, rng, out_size=None)
    return np.asarray(img.convert("RGB"), np.uint8), box


def synthetic_ladder(cfg, fonts, cam, n: int):
    out = []
    rng = random.Random(2026)
    chars = list("AEKMRTVXĐ035789")
    for _ in range(n):
        text = rng.choice(chars)
        arr, box = render_text(rng, text, 200, fonts, cfg)
        out.append(("T0_char_200px", arr, text, READ_QUESTIONS[0], box))
    for px in (120, 80, 48, 28):
        for _ in range(n):
            text = rng.choice(WORDS)
            arr, box = render_text(rng, text, px, fonts, cfg)
            out.append((f"T1_word_{px}px", arr, text, READ_QUESTIONS[0], box))
    for level, tier in ((2, "T2_line_L2"), (4, "T3_para_L4")):
        for _ in range(n):
            arr, text, box, _ = render_read_v3_sample(
                rng, level, fonts, [], cfg, random_text=True, degrade_prob=0.0)
            out.append((tier, arr, text, READ_QUESTIONS[0], box))
    for _ in range(n):
        text = rng.choice(WORDS)
        arr, box = render_text(rng, text, 80, fonts, cfg, cam)
        out.append(("T4_word80_degraded", arr, text, READ_QUESTIONS[0], box))
    return out


def document_samples(cfg, root: Path, n: int):
    pairs = load_doc_pairs(str(root / "data" / "vdoc.jsonl"), "", log=lambda *_: None)
    picks = random.Random(999).sample(pairs[:400], min(n, len(pairs[:400])))
    handles = [tarfile.open(p) for p in sorted((root / "shards").glob("images-*.tar"))]
    index = {}
    for handle in handles:
        for member in handle.getmembers():
            index[member.name] = (handle, member)
    out = []
    for row in picks:
        hit = index.get(row["img"].replace("\\", "/").lstrip("/"))
        if hit is None:
            continue
        handle, member = hit
        img = Image.open(io.BytesIO(handle.extractfile(member).read())).convert("RGB")
        arr = np.asarray(letterbox_hw(img, cfg.input_height, cfg.input_width), np.uint8)
        out.append(("B_doc_real", arr, str(row["a"]), str(row["q"]), None))
    for handle in handles:
        handle.close()
    return out


def mtvqa_samples(cfg, n: int):
    from datasets import load_dataset
    ds = load_dataset("ByteDance/MTVQA", split="test")
    ids = [i for i, lang in enumerate(ds["lang"])
           if str(lang).upper().startswith("VI")]
    out = []
    for i in random.Random(7).sample(ids, min(n, len(ids))):
        row = ds[i]
        raw = row["qa_pairs"]
        if isinstance(raw, str):
            try:
                qa = json.loads(raw)
            except Exception:
                import ast
                qa = ast.literal_eval(raw)
        else:
            qa = raw
        if not qa:
            continue
        img = letterbox_hw(row["image"].convert("RGB"),
                           cfg.input_height, cfg.input_width)
        out.append(("C_mtvqa_vi", np.asarray(img, np.uint8),
                    str(qa[0]["answer"]), str(qa[0]["question"]), None))
    return out


@torch.no_grad()
def evaluate(model, cfg, samples, device, log_every=4):
    rows = []
    for i, (tier, arr, ref, question, box) in enumerate(samples):
        img = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).float().div(255).to(device)
        q, qv = pad_text([question], cfg.max_q_bytes, cfg.tokenizer_mode)
        q, qv = q.to(device), qv.to(device)
        max_new = min(cfg.max_a_bytes, 160 if tier == "B_doc_real" else 64)
        t0 = time.perf_counter()
        with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            pred = model.generate_answer(img, q, qv, max_new=max_new)[0]
            ctc = model.predict_ocr_ctc(img)[0] if box is not None else None
            box_pred = model.predict_text_box(img).float().cpu() if box is not None else None
        row = {
            "tier": tier, "reference": ref, "prediction": pred,
            "cer": float(char_error_rate(norm(pred), norm(ref))),
            "exact": float(norm(pred) == norm(ref)),
            "contains": float(bool(norm(ref)) and norm(ref) in norm(pred)),
            "stopped": float(len(pred) < max_new - 1),
            "seconds": time.perf_counter() - t0,
        }
        if ctc is not None:
            row["ctc_prediction"] = ctc
            row["ctc_cer"] = float(char_error_rate(norm(ctc), norm(ref)))
            target = torch.as_tensor(box).view(1, 4)
            row["box_iou"] = float(xyxy_iou(box_pred, target)[0])
        rows.append(row)
        if (i + 1) % log_every == 0 or i + 1 == len(samples):
            print(f"  {i + 1}/{len(samples)} {tier} {row['seconds']:.1f}s/sample", flush=True)
    return rows


@torch.no_grad()
def blind_control(model, cfg, samples, device, batch_size=2):
    materialized = [s for s in samples if s[0] in
                    ("T1_word_80px", "T2_line_L2", "T3_para_L4", "B_doc_real")]
    gaps, wins = [], []
    for start in range(0, len(materialized), batch_size):
        chunk = materialized[start:start + batch_size]
        if len(chunk) < 2:
            continue
        img = torch.from_numpy(np.stack([r[1] for r in chunk])).permute(0, 3, 1, 2)
        img = img.float().div(255).to(device)
        q, qv = pad_text([r[3] for r in chunk], cfg.max_q_bytes, cfg.tokenizer_mode)
        a, av = pad_answers([r[2] for r in chunk], cfg.max_a_bytes, cfg.tokenizer_mode)
        q, qv, a, av = (x.to(device) for x in (q, qv, a, av))
        with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            vis = model._img_tokens(img)
            good, _ = model._qa_ce_from_vis(img, vis, q, qv, a, av, a)
            bad, _ = model._qa_ce_from_vis(img, vis.roll(1, 0), q, qv, a, av, a)
        gap = float((bad - good).float())
        gaps.append(gap); wins.append(float(gap > 0))
    return {
        "loss_gap_shuffled_minus_correct": statistics.mean(gaps) if gaps else None,
        "correct_image_win_rate": statistics.mean(wins) if wins else None,
        "n_batches": len(gaps),
    }


def summarize(rows):
    groups = {}
    for tier in sorted({r["tier"] for r in rows}):
        rs = [r for r in rows if r["tier"] == tier]
        rec = {"n": len(rs)}
        for key in ("cer", "exact", "contains", "stopped", "ctc_cer", "box_iou", "seconds"):
            vals = [r[key] for r in rs if key in r]
            if vals:
                rec[key] = float(statistics.mean(vals))
        groups[tier] = rec
    return groups


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tag", default="v3")
    ap.add_argument("--n-synth", type=int, default=12)
    ap.add_argument("--n-doc", type=int, default=40)
    ap.add_argument("--n-mtvqa", type=int, default=50)
    ap.add_argument("--skip-mtvqa", action="store_true")
    ap.add_argument("--vdoc", default=os.environ.get("JWM_VDOC", ""),
                    help="optional root of the local Vietnamese document dataset")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    blob = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = JWMConfig(**blob["cfg"])
    model = JWM(cfg)
    model.load_state_dict(blob["model"], strict=True)
    model.eval().to(device)
    fonts = find_fonts()
    cam = CameraParams(noise_std=7.2, blur_sigma=0.54, jpeg_q=55,
                       contrast=1.25, wb_shift=6.0, vignette=0.12)

    samples = synthetic_ladder(cfg, fonts, cam, args.n_synth)
    if args.vdoc:
        samples += document_samples(cfg, Path(args.vdoc), args.n_doc)
    if not args.skip_mtvqa:
        try:
            samples += mtvqa_samples(cfg, args.n_mtvqa)
        except Exception as exc:
            print(f"MTVQA unavailable: {exc}", flush=True)
    print(f"device={device} input={cfg.input_height}x{cfg.input_width} n={len(samples)}", flush=True)
    rows = evaluate(model, cfg, samples, device)
    result = {
        "benchmark": "JWM-EyeRead-v3",
        "checkpoint": str(Path(args.ckpt).resolve()),
        "summary": summarize(rows),
        "blind_control": blind_control(model, cfg, samples, device),
        "rows": rows,
    }
    out = ROOT / "data" / f"bench_read_{args.tag}.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n=== SUMMARY ===", flush=True)
    for tier, rec in result["summary"].items():
        print(tier, json.dumps(rec, ensure_ascii=False), flush=True)
    print("blind", json.dumps(result["blind_control"]), flush=True)
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
