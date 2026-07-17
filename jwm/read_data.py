"""JWM-Read data: unlimited synthetic Vietnamese text rendering (the TrOCR/Donut
bootstrap trick) + real document-page QA from the TranNhiem Vietnamese dataset.

Everything is LAZY — 768px images are rendered/loaded per batch, never
pre-tensorized (64K pages at 768^2 would be ~50GB).
"""

from __future__ import annotations

import json
import random
import re
import tarfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from .config import JWMConfig
from .data import pad_text, pad_answers
from .sdg import CameraParams, camera_degrade

# ----------------------------------------------------------------------------
# fonts + corpus
# ----------------------------------------------------------------------------

FONT_CANDIDATES = [
    r"C:\Windows\Fonts\arial.ttf", r"C:\Windows\Fonts\segoeui.ttf",
    r"C:\Windows\Fonts\times.ttf", r"C:\Windows\Fonts\tahoma.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
]

_WORDS = ("học sinh giáo viên bài tập kiểm tra điểm số trường lớp sách vở "
          "văn bản tác giả tác phẩm câu hỏi trả lời nội dung ý nghĩa đoạn văn "
          "bảng số liệu thống kê sản lượng doanh thu kết quả năm tháng ngày "
          "thành phố nông nghiệp công nghiệp phát triển kinh tế xã hội "
          "chương trình khoa học tự nhiên lịch sử địa lý toán học vật lý "
          "hóa học sinh học tiếng việt ngữ văn ví dụ chú ý ghi nhớ ôn tập").split()


def find_fonts() -> list[str]:
    return [f for f in FONT_CANDIDATES if Path(f).exists()]


def load_corpus_lines(vdoc_jsonl: str | None, limit: int = 20000) -> list[str]:
    """Natural Vietnamese lines mined from the doc dataset's own questions."""
    lines: list[str] = []
    if vdoc_jsonl and Path(vdoc_jsonl).exists():
        with open(vdoc_jsonl, encoding="utf-8") as f:
            for i, ln in enumerate(f):
                if i >= limit:
                    break
                try:
                    rec = json.loads(ln)
                    for turn in rec["conversations"]:
                        if turn["from"] == "human" and isinstance(turn["value"], str):
                            t = turn["value"].strip()
                            if 12 <= len(t) <= 160:
                                lines.append(t)
                except Exception:
                    continue
    return lines


def _phrase(rng: random.Random, corpus: list[str], n_words: int) -> str:
    if corpus and rng.random() < 0.6:
        words = rng.choice(corpus).split()
        if len(words) >= n_words:
            s = rng.randrange(0, len(words) - n_words + 1)
            return " ".join(words[s : s + n_words])
    return " ".join(rng.choice(_WORDS) for _ in range(n_words))


# ----------------------------------------------------------------------------
# synthetic text rendering (curriculum L1..L4)
# ----------------------------------------------------------------------------

CURRICULUM = {
    # level: (n_lines rng, words/line rng, font px rng)
    1: ((1, 1), (1, 2), (70, 130)),
    2: ((1, 1), (3, 7), (44, 76)),
    3: ((2, 4), (3, 8), (30, 52)),
    4: ((4, 8), (4, 9), (20, 36)),
}


def render_read_sample(rng: random.Random, level: int, fonts: list[str],
                       corpus: list[str], size: int = 768,
                       cam: CameraParams | None = None) -> tuple[np.ndarray, str]:
    """One synthetic sample: rendered Vietnamese text image + ground-truth text."""
    (l_lo, l_hi), (w_lo, w_hi), (f_lo, f_hi) = CURRICULUM[level]
    n_lines = rng.randint(l_lo, l_hi)
    lines = [_phrase(rng, corpus, rng.randint(w_lo, w_hi)) for _ in range(n_lines)]
    text = "\n".join(lines)

    bg = rng.randint(232, 252)
    img = Image.new("RGB", (size, size),
                    (bg + rng.randint(-6, 3), bg + rng.randint(-6, 3), bg + rng.randint(-8, 0)))
    d = ImageDraw.Draw(img)
    font_px = rng.randint(f_lo, f_hi)
    font = ImageFont.truetype(rng.choice(fonts), font_px)
    ink = rng.randint(10, 70)
    y = rng.randint(30, max(31, size - n_lines * int(font_px * 1.4) - 40))
    for ln in lines:
        x = rng.randint(24, 80)
        d.text((x, y), ln, font=font, fill=(ink, ink, ink))
        y += int(font_px * rng.uniform(1.25, 1.55))
    if cam is not None and rng.random() < 0.7:
        img = camera_degrade(img, cam, rng, out_size=size)
    return np.asarray(img.convert("RGB"), dtype=np.uint8), text


# ----------------------------------------------------------------------------
# real document pages (TranNhiem vdoc.jsonl + extracted image tree)
# ----------------------------------------------------------------------------

_MD_STRIP = re.compile(r"[*_#`]|\s+", re.UNICODE)


def _clean_answer(a: str) -> str:
    a = a.replace("**", "").replace("*", "").replace("`", "")
    return " ".join(a.split())


def extract_tars(shards_dir: str, out_dir: str, log=print) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for tar_path in sorted(Path(shards_dir).glob("images-*.tar")):
        marker = out / (tar_path.stem + ".done")
        if marker.exists():
            continue
        log(f"extracting {tar_path.name} ...")
        with tarfile.open(tar_path) as tf:
            tf.extractall(out)
        marker.touch()


def load_doc_pairs(vdoc_jsonl: str, images_root: str, max_answer_bytes: int = 220,
                   limit: int | None = None, log=print) -> list[dict]:
    """Flatten conversations into single-turn (image, question, VI answer) pairs.

    English reasoning traces are dropped (Jarvis answers in Vietnamese; the
    192-224B answer budget favors concise answers). Long answers are filtered,
    not truncated — truncated supervision teaches truncation.
    """
    root = Path(images_root)
    pairs: list[dict] = []
    kept = skipped_len = skipped_img = 0
    with open(vdoc_jsonl, encoding="utf-8") as f:
        for ln in f:
            try:
                rec = json.loads(ln)
            except Exception:
                continue
            img_path = root / rec["image"]
            conv = rec["conversations"]
            for i in range(0, len(conv) - 1, 2):
                q = conv[i].get("value")
                gv = conv[i + 1].get("value")
                if not isinstance(q, str) or not isinstance(gv, dict):
                    continue
                a = _clean_answer(str(gv.get("answer", "")))
                if a and len(a.encode("utf-8")) > max_answer_bytes:
                    # long answers usually open with the direct answer sentence,
                    # then explain — keep the first sentence if it stands alone
                    first = a.split(". ")[0].strip()
                    if 20 <= len(first.encode("utf-8")) <= max_answer_bytes and \
                            not first.endswith(":"):
                        a = first if first.endswith((".", "!", "?")) else first + "."
                    else:
                        a = ""
                if not a or len(a.encode("utf-8")) > max_answer_bytes or \
                        len(q.encode("utf-8")) > 90:
                    skipped_len += 1
                    continue
                pairs.append({"img": str(img_path), "q": q, "a": a})
                kept += 1
            if limit and kept >= limit:
                break
    # existence check on a sample only (checking 300K paths is slow)
    probe = [p for p in pairs[:200] if not Path(p["img"]).exists()]
    if probe:
        skipped_img = len(probe)
        log(f"WARNING: {skipped_img}/200 probed image paths missing — check extraction")
    log(f"doc pairs: kept={kept} skipped_long={skipped_len}")
    return pairs


def letterbox(img: Image.Image, size: int) -> Image.Image:
    """Aspect-preserving resize onto a size x size gray canvas."""
    w, h = img.size
    s = size / max(w, h)
    nw, nh = max(1, round(w * s)), max(1, round(h * s))
    canvas = Image.new("RGB", (size, size), (128, 128, 128))
    canvas.paste(img.resize((nw, nh), Image.LANCZOS), ((size - nw) // 2, (size - nh) // 2))
    return canvas


# ----------------------------------------------------------------------------
# lazy batcher (train_stage-compatible, qa mode only)
# ----------------------------------------------------------------------------

READ_QUESTIONS = ("đọc chữ trong ảnh", "hãy chép lại văn bản trong ảnh",
                  "transcribe the text in the image")


class LazyReadBatcher:
    """Mixes synthetic READ samples and real doc-QA pairs, materializing 768px
    images only at batch time. Exposes the ModeBatcher interface used by
    train_stage (pick_mode + batch_qa)."""

    def __init__(self, cfg: JWMConfig, doc_pairs: list[dict], fonts: list[str],
                 corpus: list[str], cam: CameraParams | None,
                 synth_ratio: float = 0.5, levels: tuple = (1, 2, 3, 4),
                 seed: int = 0):
        self.cfg = cfg
        self.doc = doc_pairs
        self.fonts = fonts
        self.corpus = corpus
        self.cam = cam
        self.synth_ratio = synth_ratio
        self.levels = levels
        self.rng = random.Random(seed)

    def pick_mode(self, probs: dict) -> str:
        return "qa"

    def _one(self) -> tuple[np.ndarray, str, str]:
        s = self.cfg.image_size
        if not self.doc or self.rng.random() < self.synth_ratio:
            level = self.rng.choice(self.levels)
            img, text = render_read_sample(self.rng, level, self.fonts,
                                           self.corpus, size=s, cam=self.cam)
            return img, self.rng.choice(READ_QUESTIONS), text
        p = self.rng.choice(self.doc)
        try:
            im = Image.open(p["img"]).convert("RGB")
        except Exception:
            level = self.rng.choice(self.levels)
            img, text = render_read_sample(self.rng, level, self.fonts,
                                           self.corpus, size=s, cam=self.cam)
            return img, self.rng.choice(READ_QUESTIONS), text
        return np.asarray(letterbox(im, s), dtype=np.uint8), p["q"], p["a"]

    def batch_qa(self, n: int, device):
        imgs, qs, ans = [], [], []
        for _ in range(n):
            im, q, a = self._one()
            imgs.append(im)
            qs.append(q)
            ans.append(a)
        img = torch.from_numpy(np.stack(imgs)).permute(0, 3, 1, 2).float().div(255.0).to(device)
        q_ids, q_valid = pad_text(qs, self.cfg.max_q_bytes)
        a_ids, a_valid = pad_answers(ans, self.cfg.max_a_bytes)
        return img, q_ids.to(device), q_valid.to(device), a_ids.to(device), a_valid.to(device)


class PrefetchBatcher:
    """Wraps a LazyReadBatcher with a background thread so CPU image
    rendering/decoding overlaps GPU compute. Same interface as the inner batcher."""

    def __init__(self, inner: LazyReadBatcher, batch_size: int, depth: int = 3):
        import queue
        import threading
        self.inner = inner
        self.batch_size = batch_size
        self.q: "queue.Queue" = queue.Queue(maxsize=depth)
        self._stop = False
        self.t = threading.Thread(target=self._worker, daemon=True)
        self.t.start()

    def _worker(self):
        import queue as _q
        while not self._stop:
            b = self.inner.batch_qa(self.batch_size, "cpu")
            while not self._stop:            # don't discard a built batch on Full
                try:
                    self.q.put(b, timeout=2)
                    break
                except _q.Full:
                    continue

    def pick_mode(self, probs: dict) -> str:
        return "qa"

    def batch_qa(self, n: int, device):
        assert n == self.batch_size, "PrefetchBatcher is fixed-batch-size"
        return tuple(t.to(device, non_blocking=True) for t in self.q.get())

    def stop(self):
        self._stop = True


def build_eval_set(cfg: JWMConfig, doc_pairs: list[dict], fonts, corpus, cam,
                   n_synth_per_level: int = 12, n_doc: int = 60, seed: int = 999) -> list[dict]:
    """Fixed held-out eval samples (descriptors; images realized at eval time)."""
    rng = random.Random(seed)
    out = []
    for level in (1, 2, 3, 4):
        for _ in range(n_synth_per_level):
            out.append({"kind": f"synthL{level}", "level": level, "seed": rng.random()})
    for p in rng.sample(doc_pairs, min(n_doc, len(doc_pairs))):
        out.append({"kind": "doc", **p})
    return out
