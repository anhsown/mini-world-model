"""Evaluation utilities for JWM-Read v3.

Training CE is intentionally not a promotion metric.  These routines measure
free-running recognition, direct CTC OCR, text-region localization and whether
the answer actually becomes worse when the image is shuffled.
"""

from __future__ import annotations

from collections import defaultdict
from contextlib import nullcontext
import re
import unicodedata

import numpy as np
import torch

from .data import pad_answers, pad_text
from .mathx import char_error_rate
from .read_v3_data import realize_v3_eval


def _norm(text: str) -> str:
    text = unicodedata.normalize("NFC", text).lower().strip()
    return re.sub(r"\s+", " ", text)


def xyxy_iou(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Aligned IoU for normalized xyxy boxes."""
    lt = torch.maximum(a[..., :2], b[..., :2])
    rb = torch.minimum(a[..., 2:], b[..., 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    aa = ((a[..., 2] - a[..., 0]).clamp(min=0) *
          (a[..., 3] - a[..., 1]).clamp(min=0))
    bb = ((b[..., 2] - b[..., 0]).clamp(min=0) *
          (b[..., 3] - b[..., 1]).clamp(min=0))
    return inter / (aa + bb - inter).clamp(min=1e-8)


def _amp_context(device: torch.device, enabled: bool):
    if enabled and device.type == "cuda":
        return torch.autocast("cuda", dtype=torch.float16)
    return nullcontext()


@torch.no_grad()
def eval_read_v3(model, descriptors: list[dict], cfg, device,
                 fonts, corpus, cam, batch_size: int = 2,
                 max_new: int | None = None, amp: bool = True) -> dict:
    """Run one deterministic benchmark pass and return JSON-safe metrics."""
    model.eval()
    rows = []
    failures = []
    for item in descriptors:
        try:
            arr, q, answer, box = realize_v3_eval(item, cfg, fonts, corpus, cam)
            rows.append((item["kind"], arr, q, answer, box))
        except Exception as exc:
            failures.append({"kind": item.get("kind", "unknown"), "error": str(exc)})

    records = []
    for start in range(0, len(rows), batch_size):
        chunk = rows[start:start + batch_size]
        imgs = torch.from_numpy(np.stack([r[1] for r in chunk])).permute(0, 3, 1, 2)
        imgs = imgs.float().div(255).to(device)
        q, qv = pad_text([r[2] for r in chunk], cfg.max_q_bytes, cfg.tokenizer_mode)
        a, av = pad_answers([r[3] for r in chunk], cfg.max_a_bytes, cfg.tokenizer_mode)
        q, qv, a, av = (x.to(device) for x in (q, qv, a, av))
        with _amp_context(device, amp):
            pred = model.generate_answer(imgs, q, qv, max_new=max_new or cfg.max_a_bytes)
            _, teacher = model.loss_qa(imgs, q, qv, a, av)
            synth_idx = [i for i, r in enumerate(chunk) if r[4] is not None]
            ctc_pred = model.predict_ocr_ctc(imgs[synth_idx]) if synth_idx else []
            box_pred = model.predict_text_box(imgs[synth_idx]).float().cpu() if synth_idx else None

        synth_pos = {row_i: j for j, row_i in enumerate(synth_idx)}
        for i, (kind, _, question, ref, box) in enumerate(chunk):
            rec = {
                "kind": kind, "question": question, "reference": ref,
                "prediction": pred[i],
                "cer": float(char_error_rate(_norm(pred[i]), _norm(ref))),
                "exact": float(_norm(pred[i]) == _norm(ref)),
                "teacher_tok_acc": float(teacher["qa_tok_acc"]),
            }
            similarity = max(0.0, 1.0 - rec["cer"])
            rec["anls"] = similarity if similarity >= 0.5 else 0.0
            if i in synth_pos:
                j = synth_pos[i]
                rec["ctc_prediction"] = ctc_pred[j]
                rec["ctc_cer"] = float(char_error_rate(_norm(ctc_pred[j]), _norm(ref)))
                target = torch.as_tensor(box, dtype=torch.float32).view(1, 4)
                rec["box_iou"] = float(xyxy_iou(box_pred[j:j + 1], target)[0])
            records.append(rec)

    def aggregate(group):
        out = {"n": len(group)}
        for key in ("cer", "anls", "exact", "ctc_cer", "box_iou"):
            vals = [r[key] for r in group if key in r and np.isfinite(r[key])]
            if vals:
                out[key] = float(np.mean(vals))
        return out

    by_kind = {k: aggregate([r for r in records if r["kind"] == k])
               for k in sorted({r["kind"] for r in records})}
    synthetic = [r for r in records if r["kind"].startswith("rand")]
    documents = [r for r in records if r["kind"] == "doc"]
    finite_kinds = [value for value in by_kind.values() if "cer" in value]
    worst_kind_cer = max((value["cer"] for value in finite_kinds), default=float("nan"))
    worst_kind_exact = min((value.get("exact", 0.0) for value in finite_kinds),
                           default=float("nan"))
    return {
        "overall": aggregate(records),
        "synthetic": aggregate(synthetic),
        "documents": aggregate(documents),
        "by_kind": by_kind,
        "robustness": {"worst_kind_cer": worst_kind_cer,
                       "worst_kind_exact": worst_kind_exact},
        "failures": failures,
        "examples": records[:12],
    }


@torch.no_grad()
def eval_vision_gain_v3(model, descriptors: list[dict], cfg, device,
                        fonts, corpus, cam, batch_size: int = 4,
                        amp: bool = True) -> dict:
    """Teacher-forced control: correct images must beat shuffled images.

    A positive loss gap means the model consumes visual evidence.  If the gap
    stays near zero while QA token accuracy rises, training has found a language
    shortcut and the stage must not be promoted.
    """
    model.eval()
    gaps, wins = [], []
    materialized = []
    for item in descriptors:
        try:
            arr, q, answer, _ = realize_v3_eval(item, cfg, fonts, corpus, cam)
            materialized.append((arr, q, answer))
        except Exception:
            continue
    for start in range(0, len(materialized), batch_size):
        chunk = materialized[start:start + batch_size]
        if len(chunk) < 2:
            continue
        img = torch.from_numpy(np.stack([r[0] for r in chunk])).permute(0, 3, 1, 2)
        img = img.float().div(255).to(device)
        q, qv = pad_text([r[1] for r in chunk], cfg.max_q_bytes, cfg.tokenizer_mode)
        a, av = pad_answers([r[2] for r in chunk], cfg.max_a_bytes, cfg.tokenizer_mode)
        q, qv, a, av = (x.to(device) for x in (q, qv, a, av))
        with _amp_context(device, amp):
            vis = model._img_tokens(img)
            good, _ = model._qa_ce_from_vis(img, vis, q, qv, a, av, a)
            bad, _ = model._qa_ce_from_vis(img, vis.roll(1, 0), q, qv, a, av, a)
        gap = float((bad - good).float())
        gaps.append(gap)
        wins.append(float(gap > 0))
    return {
        "loss_gap_shuffled_minus_correct": float(np.mean(gaps)) if gaps else float("nan"),
        "correct_image_win_rate": float(np.mean(wins)) if wins else float("nan"),
        "n_batches": len(gaps),
    }
