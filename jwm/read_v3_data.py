"""Validated data pipeline for JWM-Read v3.

The v3 curriculum separates labels by what they actually supervise:
synthetic samples have exact transcript + xyxy text box labels; real document
QA has QA supervision only. No document answer is incorrectly treated as a
full-page OCR transcript.
"""

from __future__ import annotations

import hashlib
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from . import tokenizer as tok
from .config import JWMConfig
from .data import pad_answers, pad_text
from .read_data import READ_QUESTIONS, _phrase, random_phrase
from .sdg import CameraParams, camera_degrade


V3_CURRICULUM = {
    # lines, words/line, font px
    1: ((1, 1), (1, 2), (100, 180)),
    2: ((1, 1), (3, 8), (52, 92)),
    3: ((2, 5), (3, 9), (30, 56)),
    4: ((4, 9), (4, 10), (18, 34)),
}


def letterbox_hw(img: Image.Image, height: int, width: int,
                 fill=(235, 235, 235)) -> Image.Image:
    """Aspect-preserving resize to a portrait/landscape canvas."""
    w, h = img.size
    scale = min(width / max(1, w), height / max(1, h))
    nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
    canvas = Image.new("RGB", (width, height), fill)
    canvas.paste(img.resize((nw, nh), Image.Resampling.LANCZOS),
                 ((width - nw) // 2, (height - nh) // 2))
    return canvas


def _make_lines(rng: random.Random, level: int, corpus: list[str],
                random_text: bool, max_tokens: int, mode: str) -> list[str]:
    (l0, l1), (w0, w1), _ = V3_CURRICULUM[level]
    n_lines = rng.randint(l0, l1)
    lines: list[str] = []
    for _ in range(n_lines):
        n_words = rng.randint(w0, w1)
        line = random_phrase(rng, n_words) if random_text else _phrase(rng, corpus, n_words)
        candidate = "\n".join(lines + [line])
        if len(tok.encode(candidate, mode=mode)) > max_tokens:
            break
        lines.append(line)
    if not lines:
        line = random_phrase(rng, 1) if random_text else _phrase(rng, corpus, 1)
        lines = [line]
    return lines


def render_read_v3_sample(rng: random.Random, level: int, fonts: list[str],
                          corpus: list[str], cfg: JWMConfig,
                          cam: CameraParams | None = None,
                          random_text: bool = True,
                          degrade_prob: float = 0.65) -> tuple[np.ndarray, str, np.ndarray, dict]:
    """Render one exact transcript and its union xyxy box in normalized coords."""
    h, w = cfg.input_height, cfg.input_width
    lines = _make_lines(rng, level, corpus, random_text,
                        cfg.max_a_bytes - 1, cfg.tokenizer_mode)
    text = "\n".join(lines)
    _, _, (f0, f1) = V3_CURRICULUM[level]
    font_px = rng.randint(f0, f1)
    font_path = rng.choice(fonts)

    paper = rng.randint(226, 252)
    base = np.full((h, w, 3), paper, dtype=np.int16)
    # Low-frequency paper/lighting variation, not independent pixel snow.
    yy = np.linspace(-1, 1, h, dtype=np.float32)[:, None]
    xx = np.linspace(-1, 1, w, dtype=np.float32)[None, :]
    shade = rng.uniform(-8, 8) * xx + rng.uniform(-7, 7) * yy
    base = np.clip(base + shade[..., None], 0, 255).astype(np.uint8)
    img = Image.fromarray(base, "RGB")
    draw = ImageDraw.Draw(img)
    ink = rng.randint(8, 75)
    gap_scale = rng.uniform(1.20, 1.52)
    # Fit before drawing. Clipped transcripts create contradictory OCR labels,
    # so reducing the font is preferable to training on missing glyphs.
    while True:
        font = ImageFont.truetype(font_path, font_px)
        max_line_w = max(draw.textbbox((0, 0), line, font=font)[2] for line in lines)
        line_gap = int(font_px * gap_scale)
        if (max_line_w <= w - 32 and line_gap * len(lines) <= h - 72) or font_px <= 6:
            break
        font_px -= 2
    total_h = line_gap * len(lines)
    y = rng.randint(36, max(37, h - total_h - 36))
    boxes = []
    for line in lines:
        # The fit loop above guarantees the complete transcript is visible.
        bb = draw.textbbox((0, 0), line, font=font)
        text_w = bb[2] - bb[0]
        x_max = max(8, w - text_w - 8)
        x = rng.randint(8, min(90, x_max))
        draw.text((x, y), line, font=font, fill=(ink, ink, ink))
        boxes.append(draw.textbbox((x, y), line, font=font))
        y += line_gap

    x1 = min(b[0] for b in boxes); y1 = min(b[1] for b in boxes)
    x2 = max(b[2] for b in boxes); y2 = max(b[3] for b in boxes)
    bbox = np.asarray([x1 / w, y1 / h, x2 / w, y2 / h], dtype=np.float32)

    degraded = cam is not None and rng.random() < degrade_prob
    if degraded:
        # Photometric degradation preserves the exact box geometry.
        img = camera_degrade(img, cam, rng, out_size=None)
        if img.size != (w, h):
            img = img.resize((w, h), Image.Resampling.LANCZOS)
    arr = np.asarray(img.convert("RGB"), dtype=np.uint8)
    meta = {"level": level, "font_px": font_px, "random_text": random_text,
            "degraded": degraded, "n_lines": len(lines)}
    return arr, text, bbox, meta


def curate_doc_pairs(pairs: list[dict], cfg: JWMConfig) -> list[dict]:
    out = []
    for p in pairs:
        if not Path(p["img"]).exists():
            continue
        if len(tok.encode(p["q"], mode=cfg.tokenizer_mode)) > cfg.max_q_bytes:
            continue
        if not (1 <= len(tok.encode(p["a"], mode=cfg.tokenizer_mode)) < cfg.max_a_bytes):
            continue
        out.append(p)
    return out


def split_by_document(pairs: list[dict], seed: int = 2026,
                      val_pct: int = 2, test_pct: int = 2):
    """Stable image-level split: conversations from one page never leak."""
    groups: dict[str, list[dict]] = defaultdict(list)
    for p in pairs:
        groups[str(Path(p["img"]).resolve())].append(p)
    split = {"train": [], "val": [], "test": []}
    for image, rows in groups.items():
        key = hashlib.sha1(f"{seed}:{image}".encode()).digest()[0] % 100
        name = "test" if key < test_pct else "val" if key < test_pct + val_pct else "train"
        split[name].extend(rows)
    return split


def _image_stats(arr: np.ndarray) -> np.ndarray:
    gray = arr.astype(np.float32).mean(axis=2) / 255.0
    gx = np.abs(np.diff(gray, axis=1)).mean()
    gy = np.abs(np.diff(gray, axis=0)).mean()
    return np.asarray([gray.mean(), gray.std(), gx + gy, (gray < 0.45).mean()], np.float32)


def validate_read_v3_data(cfg: JWMConfig, fonts: list[str], corpus: list[str],
                          splits: dict, cam: CameraParams | None,
                          n_synth: int = 64, n_real: int = 32,
                          seed: int = 8181) -> dict:
    """Validate training hypotheses before any GPU optimization begins."""
    rng = random.Random(seed)
    texts, synth_stats = [], []
    valid_boxes = token_ok = visible = 0
    for i in range(n_synth):
        arr, text, box, _ = render_read_v3_sample(
            rng, 1 + i % 4, fonts, corpus, cfg, cam, random_text=True)
        texts.append(text)
        synth_stats.append(_image_stats(arr))
        valid_boxes += int(np.all(box >= 0) and np.all(box <= 1) and
                           box[2] > box[0] and box[3] > box[1])
        token_ok += int(len(tok.encode(text, mode=cfg.tokenizer_mode)) < cfg.max_a_bytes)
        visible += int(arr.min() < 120 and arr.std() > 8)

    real_stats = []
    for p in splits["train"][:n_real]:
        try:
            with Image.open(p["img"]) as im:
                arr = np.asarray(letterbox_hw(im.convert("RGB"), cfg.input_height,
                                              cfg.input_width), dtype=np.uint8)
            real_stats.append(_image_stats(arr))
        except Exception:
            continue
    train_imgs = {p["img"] for p in splits["train"]}
    val_imgs = {p["img"] for p in splits["val"]}
    test_imgs = {p["img"] for p in splits["test"]}
    leakage = len(train_imgs & val_imgs) + len(train_imgs & test_imgs) + len(val_imgs & test_imgs)

    ss = np.stack(synth_stats).mean(0)
    rs = np.stack(real_stats).mean(0) if real_stats else np.full(4, np.nan)
    # Descriptive distance: reported for curation, not used as a fake universal threshold.
    distance = float(np.nanmean(np.abs(ss - rs))) if real_stats else float("inf")
    hypotheses = {
        "H_random_labels_unique": len(set(texts)) / len(texts) >= 0.98,
        "H_boxes_exact_and_valid": valid_boxes == n_synth,
        "H_no_token_truncation": token_ok == n_synth,
        "H_text_visible": visible == n_synth,
        "H_document_split_no_leak": leakage == 0,
        "H_real_pages_available": len(real_stats) >= min(8, n_real),
    }
    return {
        "valid": all(hypotheses.values()), "hypotheses": hypotheses,
        "counts": {k: len(v) for k, v in splits.items()},
        "synthetic_stats": ss.round(5).tolist(),
        "real_stats": rs.round(5).tolist(),
        "mean_abs_domain_gap": round(distance, 5), "leaked_images": leakage,
    }


class ReadV3Batcher:
    def __init__(self, cfg: JWMConfig, doc_pairs: list[dict], fonts: list[str],
                 corpus: list[str], cam: CameraParams | None,
                 synth_ratio: float, levels: tuple[int, ...],
                 random_text_ratio: float, seed: int):
        self.cfg, self.doc, self.fonts, self.corpus, self.cam = cfg, doc_pairs, fonts, corpus, cam
        self.synth_ratio, self.levels = synth_ratio, levels
        self.random_text_ratio, self.rng = random_text_ratio, random.Random(seed)

    def _synth(self):
        level = self.rng.choice(self.levels)
        random_text = self.rng.random() < self.random_text_ratio
        arr, text, bbox, meta = render_read_v3_sample(
            self.rng, level, self.fonts, self.corpus, self.cfg, self.cam,
            random_text=random_text)
        return arr, self.rng.choice(READ_QUESTIONS), text, bbox, True, True, meta

    def _one(self):
        if not self.doc or self.rng.random() < self.synth_ratio:
            return self._synth()
        p = self.rng.choice(self.doc)
        try:
            with Image.open(p["img"]) as im:
                arr = np.asarray(letterbox_hw(im.convert("RGB"), self.cfg.input_height,
                                              self.cfg.input_width), dtype=np.uint8)
            return arr, p["q"], p["a"], np.zeros(4, np.float32), False, False, {"real": True}
        except Exception:
            return self._synth()

    def batch(self, n: int, device="cpu"):
        rows = [self._one() for _ in range(n)]
        imgs, qs, ans, boxes, cm, bm, _ = zip(*rows)
        img = torch.from_numpy(np.stack(imgs)).permute(0, 3, 1, 2).float().div(255.0)
        q, qv = pad_text(list(qs), self.cfg.max_q_bytes, self.cfg.tokenizer_mode)
        a, av = pad_answers(list(ans), self.cfg.max_a_bytes, self.cfg.tokenizer_mode)
        tensors = (img, q, qv, a, av, torch.from_numpy(np.stack(boxes)),
                   torch.tensor(cm, dtype=torch.bool), torch.tensor(bm, dtype=torch.bool))
        return tuple(t.to(device, non_blocking=True) for t in tensors)


class PrefetchReadV3:
    def __init__(self, inner: ReadV3Batcher, batch_size: int, depth: int = 2):
        import queue
        import threading
        self.inner, self.batch_size = inner, batch_size
        self.queue = queue.Queue(maxsize=depth)
        self.stop_flag = False
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

    def _worker(self):
        import queue
        while not self.stop_flag:
            batch = self.inner.batch(self.batch_size, "cpu")
            try:
                self.queue.put(batch, timeout=2)
            except queue.Full:
                continue

    def batch(self, device):
        return tuple(t.to(device, non_blocking=True) for t in self.queue.get())

    def stop(self):
        self.stop_flag = True


def build_v3_eval(cfg: JWMConfig, doc_pairs: list[dict], n_each: int = 12,
                  n_doc: int = 40, seed: int = 6060) -> list[dict]:
    rng = random.Random(seed)
    out = []
    for level in (1, 2, 3, 4):
        for _ in range(n_each):
            out.append({"kind": f"randL{level}", "level": level,
                        "seed": rng.getrandbits(63), "random_text": True})
    for p in rng.sample(doc_pairs, min(n_doc, len(doc_pairs))):
        out.append({"kind": "doc", **p})
    return out


def realize_v3_eval(item: dict, cfg: JWMConfig, fonts, corpus, cam):
    if item["kind"] == "doc":
        with Image.open(item["img"]) as im:
            arr = np.asarray(letterbox_hw(im.convert("RGB"), cfg.input_height,
                                          cfg.input_width), dtype=np.uint8)
        return arr, item["q"], item["a"], None
    rng = random.Random(item["seed"])
    arr, text, bbox, _ = render_read_v3_sample(
        rng, item["level"], fonts, corpus, cfg, cam,
        random_text=item.get("random_text", True), degrade_prob=0.5)
    return arr, READ_QUESTIONS[0], text, bbox
