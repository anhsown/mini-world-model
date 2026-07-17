# -*- coding: utf-8 -*-
"""World Brain trial harness.

Every trial records: audio, transcript, frame, predicted region, answer,
confidence, latency, ground truth and failure category.
(-> data/world_brain_trials/<session>/trials.jsonl)

Two arms:
  A "synthetic" — held-out test split (GT bbox + answer known exactly)
  B "real"      — real ASR audio (data/asr_logs WAVs -> faster-whisper) paired
                  with real camera frames; GT = hand-annotated held-object boxes.
                  The synthetic-trained model is OUT-OF-DISTRIBUTION here: the
                  CORRECT behavior is low confidence -> abstain (calibration test).
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.world_brain import WorldBrain  # noqa: E402
from jwm.mathx import bbox_iou  # noqa: E402

# Hand-annotated ground truth for real frames (held object region, normalized cx,cy,w,h)
REAL_GT = {
    "vision-165843-12712d3f-raw.jpg": {
        "bbox": [0.250, 0.517, 0.141, 0.354],
        "note": "black power adapter held in right hand",
    },
    "vision-165428-4ca674ce-raw.jpg": {
        "bbox": [0.295, 0.549, 0.113, 0.319],
        "note": "dark rounded object (mouse-like) held up in right hand",
    },
    "vision-165911-eeae4079-raw.jpg": {
        "bbox": [0.348, 0.514, 0.180, 0.361],
        "note": "black square charger held with both hands",
    },
}

VISION_QUESTION_HINTS = ("gì", "đâu", "màu", "cầm", "bao nhiêu", "what", "where",
                         "holding", "how many", "color")


def transcribe(wav: str, model) -> dict:
    t0 = time.perf_counter()
    segs, info = model.transcribe(wav, language=None, beam_size=1, vad_filter=True)
    text = "".join(s.text for s in segs).strip()
    return {"text": text, "language": getattr(info, "language", None),
            "language_probability": round(float(getattr(info, "language_probability", 0) or 0), 3),
            "asr_latency_ms": round((time.perf_counter() - t0) * 1000, 1)}


def classify_synthetic(out: dict, gt_bbox, gt_answer) -> str:
    iou = float(bbox_iou(torch.tensor(out["bbox"]), torch.tensor(gt_bbox)))
    region_ok = iou >= 0.5
    ans_ok = gt_answer is None or out["answer"].strip().lower() == gt_answer.strip().lower()
    if out["abstain"]:
        return "correct_abstain" if not region_ok else "wrong_abstain"
    if region_ok and ans_ok:
        return "ok"
    if region_ok and not ans_ok:
        return "wrong_answer"
    if not region_ok and ans_ok:
        return "grounding_miss"
    return "both_wrong"


def classify_real(out: dict, gt_bbox) -> str:
    if gt_bbox is None:
        return "no_ground_truth"
    iou = float(bbox_iou(torch.tensor(out["bbox"]), torch.tensor(gt_bbox)))
    if out["abstain"]:
        return "correct_abstain_ood" if iou < 0.3 else "wrong_abstain"
    return "ok_region" if iou >= 0.3 else "grounding_miss"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-synthetic", type=int, default=40)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--no-real", action="store_true")
    ap.add_argument("--device", default=None, help="cuda|cpu (default: auto)")
    args = ap.parse_args()

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = ROOT / "data" / "world_brain_trials" / stamp
    (out_dir / "frames").mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "trials.jsonl"

    print(f"== World Brain trials -> {out_dir}")
    brain = WorldBrain(device=args.device)
    print(f"   checkpoint loaded on {brain.device}; abstain_threshold={brain.abstain_threshold}")

    trials: list[dict] = []

    def record(rec: dict):
        trials.append(rec)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
        cat = rec["failure_category"]
        print(f"  [{rec['trial_id']}] {cat:20s} conf={rec['confidence']:.2f} "
              f"lat={rec['latency_ms']['total']:.0f}ms q={rec['transcript'][:36]!r}")

    # ---------------- Arm A: synthetic test split ----------------
    test = torch.load(ROOT / "data" / "jwm_sdg" / "test.pt", weights_only=False)
    g = test["ground"]
    n = min(args.n_synthetic, len(g["q"]))
    print(f"\n-- Arm A: {n} synthetic trials (GT exact) --")
    for i in range(n):
        frame = g["img"][i].numpy()
        fpath = out_dir / "frames" / f"synthetic_{i:03d}.png"
        Image.fromarray(frame).save(fpath)
        try:
            out = brain.analyze(frame, g["q"][i], steps=args.steps)
            cat = classify_synthetic(out, g["bbox"][i].tolist(), None)
            iou = float(bbox_iou(torch.tensor(out["bbox"]), g["bbox"][i]))
        except Exception as exc:  # noqa: BLE001
            out = {"answer": None, "bbox": None, "confidence": 0.0, "abstain": True,
                   "latency_ms": {"total": 0.0}, "reflection": None}
            cat, iou = "error", 0.0
            print("   ERROR:", exc)
        record({
            "trial_id": f"A{i:03d}",
            "arm": "synthetic",
            "audio": None,
            "transcript": g["q"][i],
            "frame": str(fpath.relative_to(ROOT)),
            "predicted_region": out["bbox"],
            "answer": out["answer"],
            "confidence": out["confidence"],
            "latency_ms": out["latency_ms"],
            "ground_truth": {"bbox": g["bbox"][i].tolist(), "answer": None,
                             "iou": round(iou, 4)},
            "failure_category": cat,
            "abstain": out["abstain"],
            "reflection": out.get("reflection"),
        })

    # ---------------- Arm B: real audio + real frames ----------------
    if not args.no_real:
        wavs = sorted(glob.glob(str(ROOT / "data" / "asr_logs" / "*" / "*.wav")))
        frames = sorted(glob.glob(str(ROOT / "data" / "vision_sessions" / "*" / "frames" / "*-raw.jpg")))
        print(f"\n-- Arm B: {len(wavs)} real-audio trials over {len(frames)} real frames --")
        from faster_whisper import WhisperModel
        asr = WhisperModel("base", device="cpu", compute_type="int8")
        for j, wav in enumerate(wavs):
            asr_out = transcribe(wav, asr)
            text = asr_out["text"]
            usable = bool(text) and any(h in text.lower() for h in VISION_QUESTION_HINTS)
            query = text if usable else "tôi đang cầm vật gì?"
            fpath_src = Path(frames[j % len(frames)])
            frame = np.asarray(Image.open(fpath_src).convert("RGB"))
            fpath = out_dir / "frames" / f"real_{j:03d}_{fpath_src.name}"
            Image.fromarray(frame).save(fpath)
            gt = REAL_GT.get(fpath_src.name)
            try:
                out = brain.analyze(frame, query, steps=args.steps)
                cat = classify_real(out, gt["bbox"] if gt else None)
                if not usable:
                    cat = f"asr_fallback+{cat}"
                iou = (float(bbox_iou(torch.tensor(out["bbox"]), torch.tensor(gt["bbox"])))
                       if gt else None)
            except Exception as exc:  # noqa: BLE001
                out = {"answer": None, "bbox": None, "confidence": 0.0, "abstain": True,
                       "latency_ms": {"total": 0.0}, "reflection": None}
                cat, iou = "error", None
                print("   ERROR:", exc)
            record({
                "trial_id": f"B{j:03d}",
                "arm": "real",
                "audio": str(Path(wav).relative_to(ROOT)),
                "transcript": text,
                "asr": asr_out,
                "asr_usable_question": usable,
                "query_used": query,
                "frame": str(fpath.relative_to(ROOT)),
                "predicted_region": out["bbox"],
                "answer": out["answer"],
                "confidence": out["confidence"],
                "latency_ms": out["latency_ms"],
                "ground_truth": ({"bbox": gt["bbox"], "note": gt["note"],
                                  "iou": round(iou, 4) if iou is not None else None}
                                 if gt else None),
                "failure_category": cat,
                "abstain": out["abstain"],
                "reflection": out.get("reflection"),
            })

    # ---------------- summary ----------------
    cats: dict[str, int] = {}
    for t in trials:
        cats[t["failure_category"]] = cats.get(t["failure_category"], 0) + 1
    synth = [t for t in trials if t["arm"] == "synthetic"]
    ok = sum(1 for t in synth if t["failure_category"] == "ok")
    ious = [t["ground_truth"]["iou"] for t in synth if t["ground_truth"]]
    lat = [t["latency_ms"]["total"] for t in trials if t["latency_ms"]["total"]]
    summary = {
        "session": stamp,
        "n_trials": len(trials),
        "categories": cats,
        "synthetic": {
            "n": len(synth),
            "ok_rate": round(ok / max(1, len(synth)), 3),
            "mean_iou": round(float(np.mean(ious)), 3) if ious else None,
            "iou_at_05": round(float(np.mean([i >= 0.5 for i in ious])), 3) if ious else None,
        },
        "mean_latency_ms": round(float(np.mean(lat)), 1) if lat else None,
    }
    json.dump(summary, (out_dir / "summary.json").open("w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("\n== SUMMARY ==")
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
