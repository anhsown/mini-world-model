"""SDG-JarvisSim — synthetic data generation for JWM (Task: dataset, DESIGN-aligned).

Mirrors the Cosmos 3 data methodology at micro scale:
  * procedural scenes with physics-consistent motion (FD pairs)
  * structured JSON captions (schema subset of Cosmos)
  * programmatic 3-axis judge (Faithfulness / Completeness / Correctness)
  * scene-hash dedup
  * staged splits: pretrain (broad) / sft (hard, real-background) / val / test
  * REAL-CASE VALIDATION: distribution distances vs real JARVIS camera frames,
    with an auto-tune loop over camera-degradation params until thresholds pass.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFilter

# ----------------------------------------------------------------------------
# vocab of the synthetic world
# ----------------------------------------------------------------------------

PALETTE = {
    "đỏ": (214, 45, 38),
    "xanh dương": (36, 98, 210),
    "xanh lá": (52, 158, 72),
    "vàng": (230, 194, 41),
    "cam": (236, 124, 38),
    "tím": (142, 68, 173),
}
EN_COLOR = {"đỏ": "red", "xanh dương": "blue", "xanh lá": "green",
            "vàng": "yellow", "cam": "orange", "tím": "purple"}

SHAPES = ["hình tròn", "hình vuông", "tam giác", "ngôi sao", "điện thoại", "cái cốc"]
EN_SHAPE = {"hình tròn": "circle", "hình vuông": "square", "tam giác": "triangle",
            "ngôi sao": "star", "điện thoại": "phone", "cái cốc": "cup"}

LOCATIONS = ["bên trái", "ở giữa", "bên phải"]
DIRS = {"sang trái": (-1, 0), "sang phải": (1, 0), "lên trên": (0, -1), "xuống dưới": (0, 1)}

CANVAS = 256          # render high-res then LANCZOS down to 64 (anti-aliasing)
OUT = 64


@dataclass
class CameraParams:
    """Webcam degradation model — tuned by the real-case validation loop."""
    noise_std: float = 6.0          # gaussian noise, 0-255 scale
    blur_sigma: float = 0.9         # gaussian blur radius at 64px scale
    jpeg_q: int = 55                # jpeg round-trip quality
    brightness: float = 0.0        # additive shift
    contrast: float = 1.0          # multiplicative around mean
    wb_shift: float = 6.0           # random per-channel white-balance amplitude
    vignette: float = 0.12

    def to_dict(self):
        return self.__dict__.copy()


# ----------------------------------------------------------------------------
# object rendering
# ----------------------------------------------------------------------------

def _shade(color, k):
    return tuple(int(max(0, min(255, c * k))) for c in color)


def _draw_obj(draw: ImageDraw.ImageDraw, o: dict, S: int):
    """Draw one object at canvas scale S. Shapes get simple shading for depth."""
    cx, cy, s = o["cx"] * S, o["cy"] * S, o["size"] * S
    col = o["rgb"]
    x1, y1, x2, y2 = cx - s / 2, cy - s / 2, cx + s / 2, cy + s / 2
    shape = o["shape"]
    dark, lite = _shade(col, 0.75), _shade(col, 1.15)
    if shape == "hình tròn":
        draw.ellipse([x1, y1, x2, y2], fill=col, outline=dark, width=max(2, S // 128))
        draw.ellipse([x1 + s * 0.15, y1 + s * 0.12, x1 + s * 0.45, y1 + s * 0.38], fill=lite)
    elif shape == "hình vuông":
        draw.rectangle([x1, y1, x2, y2], fill=col, outline=dark, width=max(2, S // 128))
        draw.rectangle([x1, y1, x2, y1 + s * 0.18], fill=lite)
    elif shape == "tam giác":
        pts = [(cx, y1), (x2, y2), (x1, y2)]
        draw.polygon(pts, fill=col, outline=dark)
    elif shape == "ngôi sao":
        pts = []
        for i in range(10):
            r = s / 2 if i % 2 == 0 else s / 4.6
            a = -math.pi / 2 + i * math.pi / 5
            pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
        draw.polygon(pts, fill=col, outline=dark)
    elif shape == "điện thoại":
        w, h = s * 0.55, s
        draw.rounded_rectangle([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2],
                               radius=s * 0.09, fill=col, outline=dark, width=max(2, S // 128))
        draw.rounded_rectangle([cx - w * 0.36, cy - h * 0.40, cx + w * 0.36, cy + h * 0.34],
                               radius=s * 0.04, fill=_shade(col, 0.45))
    elif shape == "cái cốc":
        w, h = s * 0.72, s * 0.92
        draw.rectangle([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2],
                       fill=col, outline=dark, width=max(2, S // 128))
        draw.ellipse([cx - w / 2, cy - h / 2 - s * 0.07, cx + w / 2, cy - h / 2 + s * 0.07],
                     fill=lite, outline=dark)
        draw.arc([cx + w / 2 - s * 0.06, cy - h * 0.25, cx + w / 2 + s * 0.30, cy + h * 0.25],
                 start=270, end=90, fill=dark, width=max(3, S // 64))


def obj_bbox(o: dict) -> tuple[float, float, float, float]:
    """Axis-aligned (cx, cy, w, h) in [0,1]. Phones are narrower; cups slightly."""
    w = o["size"] * (0.55 if o["shape"] == "điện thoại" else 0.80 if o["shape"] == "cái cốc" else 1.0)
    h = o["size"] * (0.92 if o["shape"] == "cái cốc" else 1.0)
    cx = min(max(o["cx"], w / 2), 1 - w / 2)
    cy = min(max(o["cy"], h / 2), 1 - h / 2)
    return (cx, cy, w, h)


# ----------------------------------------------------------------------------
# backgrounds
# ----------------------------------------------------------------------------

def load_real_crops(paths: list[str], per_frame: int, rng: random.Random,
                    size: int = CANVAS) -> list[Image.Image]:
    """Random crops from real JARVIS frames — the composite backgrounds."""
    crops = []
    for p in paths:
        img = Image.open(p).convert("RGB")
        W, H = img.size
        for _ in range(per_frame):
            s = rng.randint(int(min(W, H) * 0.45), min(W, H))
            x = rng.randint(0, W - s)
            y = rng.randint(0, H - s)
            crops.append(img.crop((x, y, x + s, y + s)).resize((size, size), Image.LANCZOS))
    return crops


def synth_background(rng: random.Random, size: int = CANVAS) -> Image.Image:
    """Indoor-ish synthetic background: wall/floor split or soft gradient + speckle."""
    base = np.zeros((size, size, 3), np.float32)
    kind = rng.choice(["grad", "room", "flat"])
    c1 = np.array([rng.randint(60, 200) for _ in range(3)], np.float32)
    c2 = c1 * rng.uniform(0.55, 0.9)
    if kind == "grad":
        t = np.linspace(0, 1, size)[:, None, None]
        base = c1 * (1 - t) + c2 * t
        base = np.broadcast_to(base, (size, size, 3)).copy()
    elif kind == "room":
        split = int(size * rng.uniform(0.55, 0.8))
        base[:split] = c1
        base[split:] = c2
    else:
        base[:] = c1
    base += np.random.default_rng(rng.randint(0, 10 ** 9)).normal(0, 4, base.shape)
    return Image.fromarray(np.clip(base, 0, 255).astype(np.uint8))


# ----------------------------------------------------------------------------
# camera degradation (validated against real frames)
# ----------------------------------------------------------------------------

def camera_degrade(img: Image.Image, cam: CameraParams, rng: random.Random,
                   out_size: int = OUT) -> Image.Image:
    """Apply the webcam model at out_size. Order: blur -> photometric -> noise -> jpeg."""
    OUT = out_size  # noqa: N806 — keep the body identical for any output size
    img = img.resize((OUT, OUT), Image.LANCZOS)
    if cam.blur_sigma > 0:
        img = img.filter(ImageFilter.GaussianBlur(radius=cam.blur_sigma * rng.uniform(0.6, 1.4)))
    a = np.asarray(img).astype(np.float32)
    # white balance + brightness/contrast
    for c in range(3):
        a[..., c] += rng.uniform(-cam.wb_shift, cam.wb_shift)
    mean = a.mean()
    a = (a - mean) * cam.contrast * rng.uniform(0.92, 1.08) + mean + \
        cam.brightness + rng.uniform(-8, 8)
    # vignette
    yy, xx = np.mgrid[0:OUT, 0:OUT].astype(np.float32) / OUT - 0.5
    a *= (1.0 - cam.vignette * (xx ** 2 + yy ** 2) * 4)[..., None]
    # sensor noise
    a += np.random.default_rng(rng.randint(0, 10 ** 9)).normal(0, cam.noise_std, a.shape)
    img = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
    # jpeg round-trip
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=int(cam.jpeg_q * rng.uniform(0.85, 1.15)))
    return Image.open(buf).convert("RGB")


# ----------------------------------------------------------------------------
# scene construction + physics
# ----------------------------------------------------------------------------

def place_objects(rng: random.Random, n: int, held: bool, hard: bool) -> list[dict]:
    objs = []
    tries = 0
    while len(objs) < n and tries < 200:
        tries += 1
        first_is_held = held and not objs
        size = rng.uniform(0.34, 0.52) if first_is_held else (
            rng.uniform(0.12, 0.22) if hard else rng.uniform(0.16, 0.30))
        cx = rng.uniform(0.38, 0.62) if first_is_held else rng.uniform(0.12, 0.88)
        cy = rng.uniform(0.38, 0.62) if first_is_held else rng.uniform(0.15, 0.85)
        o = {"shape": rng.choice(SHAPES), "color": rng.choice(list(PALETTE)),
             "cx": cx, "cy": cy, "size": size}
        o["rgb"] = tuple(int(min(255, max(0, v + rng.uniform(-14, 14)))) for v in PALETTE[o["color"]])
        min_gap = 0.35 if not hard else 0.18       # hard split allows partial occlusion
        if all(math.hypot(o["cx"] - p["cx"], o["cy"] - p["cy"]) >
               (o["size"] + p["size"]) / 2 * min_gap for p in objs):
            objs.append(o)
    return objs


def visible_fraction(o: dict, later: list[dict]) -> float:
    """Approximate visibility: 1 - covered area by objects drawn after o."""
    bx = obj_bbox(o)
    ax1, ay1 = bx[0] - bx[2] / 2, bx[1] - bx[3] / 2
    ax2, ay2 = bx[0] + bx[2] / 2, bx[1] + bx[3] / 2
    own = max(bx[2] * bx[3], 1e-8)
    covered = 0.0
    for p in later:
        pb = obj_bbox(p)
        px1, py1 = pb[0] - pb[2] / 2, pb[1] - pb[3] / 2
        px2, py2 = pb[0] + pb[2] / 2, pb[1] + pb[3] / 2
        iw = max(0.0, min(ax2, px2) - max(ax1, px1))
        ih = max(0.0, min(ay2, py2) - max(ay1, py1))
        covered += iw * ih
    return max(0.0, 1.0 - covered / own)


def render_scene(objs: list[dict], bg: Image.Image, cam: CameraParams,
                 rng: random.Random) -> Image.Image:
    canvas = bg.copy()
    draw = ImageDraw.Draw(canvas)
    for o in objs:                       # draw order = z order (later occludes earlier)
        _draw_obj(draw, o, CANVAS)
    return camera_degrade(canvas, cam, rng)


def loc_word(cx: float) -> str:
    return LOCATIONS[0] if cx < 0.37 else (LOCATIONS[2] if cx > 0.63 else LOCATIONS[1])


def structured_caption(objs: list[dict]) -> dict:
    """Cosmos-style structured JSON caption (schema subset)."""
    return {
        "subjects": [{
            "description": f"{o['shape']} màu {o['color']}",
            "location": loc_word(o["cx"]),
            "relative_size": "lớn" if o["size"] > 0.3 else "nhỏ",
            "bbox_norm": [round(v, 4) for v in obj_bbox(o)],
        } for o in objs],
        "background_setting": "indoor scene, webcam viewpoint",
        "lighting": {"conditions": "indoor mixed lighting"},
        "number_of_subjects": len(objs),
    }


# ----------------------------------------------------------------------------
# question factories -> (q, a, target_obj | None, kind)
# ----------------------------------------------------------------------------

def make_qa(objs: list[dict], rng: random.Random, held: bool):
    en = rng.random() < 0.25
    kind = rng.choice(["count", "where", "what", "exist", "shape", "color"]
                      if not held else ["what_held", "shape", "color"])
    if kind in ("shape", "color"):
        # attribute-decomposed curriculum: SHORT answers isolating the exact
        # perceptual skill (diagnosed failure mode: shape words systematically
        # wrong inside long composite answers)
        o = max(objs, key=lambda p: p["size"] - 0.35 * math.hypot(p["cx"] - 0.5, p["cy"] - 0.5))
        subject_vi = "vật tôi đang cầm" if held else "vật ở giữa"
        subject_en = "the object I am holding" if held else "the middle object"
        if kind == "shape":
            q = (f"what shape is {subject_en}?" if en else f"{subject_vi} là hình gì?")
            a = EN_SHAPE[o["shape"]] if en else o["shape"]
        else:
            q = (f"what color is {subject_en}?" if en else f"{subject_vi} màu gì?")
            a = EN_COLOR[o["color"]] if en else o["color"]
        return q, a, o, kind
    if kind == "count":
        shape = rng.choice(SHAPES)
        n = sum(1 for o in objs if o["shape"] == shape)
        q = f"how many {EN_SHAPE[shape]}s are there?" if en else f"có bao nhiêu {shape} trong hình?"
        return q, str(n), None, "count"
    if kind == "where":
        colors = [o["color"] for o in objs]
        uniq = [c for c in set(colors) if colors.count(c) == 1]
        if not uniq:
            return None
        c = rng.choice(uniq)
        o = next(o for o in objs if o["color"] == c)
        q = f"where is the {EN_COLOR[c]} object?" if en else f"vật màu {c} nằm ở đâu?"
        return q, loc_word(o["cx"]), o, "where"
    if kind == "exist":
        c = rng.choice(list(PALETTE))
        present = any(o["color"] == c for o in objs)
        q = f"is there a {EN_COLOR[c]} object?" if en else f"có vật màu {c} nào không?"
        return q, ("yes" if en else "có") if present else ("no" if en else "không"), None, "exist"
    # what / what_held: the largest, most central object
    o = max(objs, key=lambda p: p["size"] - 0.35 * math.hypot(p["cx"] - 0.5, p["cy"] - 0.5))
    if kind == "what_held":
        q = "what am I holding?" if en else rng.choice(
            ["tôi đang cầm vật gì?", "vật tôi đang cầm là gì?"])
    else:
        q = "what is in the middle?" if en else "vật ở giữa khung hình là gì?"
    a = f"{EN_SHAPE[o['shape']]} {EN_COLOR[o['color']]}" if en else f"{o['shape']} màu {o['color']}"
    return q, a, o, "what_held" if kind == "what_held" else "what"


def make_ground(objs: list[dict], rng: random.Random):
    """Referring expression with a UNIQUE referent -> bbox target."""
    en = rng.random() < 0.25
    cands = []
    for o in objs:
        same_color = sum(1 for p in objs if p["color"] == o["color"])
        same_pair = sum(1 for p in objs if p["color"] == o["color"] and p["shape"] == o["shape"])
        if same_color == 1:
            cands.append((f"the {EN_COLOR[o['color']]} object" if en else f"vật màu {o['color']}", o))
        elif same_pair == 1:
            cands.append((f"the {EN_COLOR[o['color']]} {EN_SHAPE[o['shape']]}" if en
                          else f"{o['shape']} màu {o['color']}", o))
    if not cands:
        return None
    ref, o = rng.choice(cands)
    q = (f"where is {ref}?" if en else f"{ref} ở đâu?")
    return q, o


def make_fd(rng: random.Random, cam: CameraParams, bg: Image.Image):
    """Physics pair: one moving object (+static distractors), dt at fd_fps.

    Linear motion with elastic bounce off frame borders — returns
    (img_t, img_t1, motion_text, meta).
    """
    objs = place_objects(rng, rng.randint(1, 3), held=False, hard=False)
    mover = objs[0]
    dname = rng.choice(list(DIRS))
    dx, dy = DIRS[dname]
    speed = rng.uniform(0.10, 0.22)
    img_t = render_scene(objs, bg, cam, rng)
    # advance physics
    nx, ny = mover["cx"] + dx * speed, mover["cy"] + dy * speed
    half = mover["size"] / 2
    if nx < half or nx > 1 - half:
        nx = min(max(nx, half), 1 - half)
    if ny < half or ny > 1 - half:
        ny = min(max(ny, half), 1 - half)
    moved = dict(mover, cx=nx, cy=ny)
    img_t1 = render_scene([moved] + objs[1:], bg, cam, rng)
    en = rng.random() < 0.25
    text = (f"the {EN_COLOR[mover['color']]} object moves" if en
            else f"vật màu {mover['color']} di chuyển {dname}")
    meta = {"mover": {"shape": mover["shape"], "color": mover["color"],
                      "from": [mover["cx"], mover["cy"]], "to": [nx, ny]},
            "direction": dname, "speed": round(speed, 4)}
    return img_t, img_t1, text, meta


# ----------------------------------------------------------------------------
# programmatic 3-axis judge (Cosmos AI-judge analog) + dedup
# ----------------------------------------------------------------------------

def judge(objs: list[dict], target: dict | None, kind: str) -> tuple[bool, str]:
    """Faithfulness: target visible & large enough. Completeness: unique referent.
    Correctness is guaranteed by construction but re-checked for referents."""
    if target is not None:
        idx = objs.index(target)
        if visible_fraction(target, objs[idx + 1:]) < 0.55:
            return False, "faithfulness:occluded"
        if target["size"] < 0.10:
            return False, "faithfulness:too_small"
        same = [o for o in objs if o["color"] == target["color"]]
        if kind in ("where", "ground") and len(same) != 1:
            same2 = [o for o in same if o["shape"] == target["shape"]]
            if len(same2) != 1:
                return False, "completeness:ambiguous_referent"
    return True, "ok"


def scene_hash(objs: list[dict], q: str) -> str:
    key = json.dumps([[o["shape"], o["color"], round(o["cx"], 2), round(o["cy"], 2),
                       round(o["size"], 2)] for o in objs]) + q
    return hashlib.md5(key.encode()).hexdigest()[:16]


# ----------------------------------------------------------------------------
# real-case validation (distribution distance vs real JARVIS frames)
# ----------------------------------------------------------------------------

def frame_stats(img: np.ndarray) -> dict:
    """img: (64,64,3) uint8. Photometric + structural statistics."""
    g = img.astype(np.float32).mean(axis=2)
    gx = np.abs(np.diff(g, axis=1)).mean()
    gy = np.abs(np.diff(g, axis=0)).mean()
    lap = np.abs(4 * g[1:-1, 1:-1] - g[:-2, 1:-1] - g[2:, 1:-1] - g[1:-1, :-2] - g[1:-1, 2:])
    hsv = np.asarray(Image.fromarray(img).convert("HSV"))
    hist, _ = np.histogram(hsv[..., 0], bins=16, range=(0, 255), density=True)
    hist = hist + 1e-9
    return {
        "lum_mean": float(g.mean()),
        "lum_std": float(g.std()),
        "grad_energy": float(gx + gy),
        "log_sharpness": float(np.log1p(lap.mean())),
        "color_entropy": float(-(hist * np.log(hist)).sum() * (255 / 16)) / 16,
    }


def wasserstein1(a: np.ndarray, b: np.ndarray) -> float:
    """1-D Wasserstein distance via sorted quantile coupling."""
    qa = np.quantile(a, np.linspace(0, 1, 101))
    qb = np.quantile(b, np.linspace(0, 1, 101))
    return float(np.abs(qa - qb).mean())


THRESHOLDS = {"lum_mean": 16.0, "lum_std": 12.0, "grad_energy": 4.5,
              "log_sharpness": 0.55, "color_entropy": 0.45}


def validate_against_real(synth: list[np.ndarray], real: list[np.ndarray]) -> dict:
    ss = {k: np.array([frame_stats(i)[k] for i in synth]) for k in THRESHOLDS}
    rs = {k: np.array([frame_stats(i)[k] for i in real]) for k in THRESHOLDS}
    report = {}
    for k, thr in THRESHOLDS.items():
        d = wasserstein1(ss[k], rs[k])
        report[k] = {"W1": round(d, 3), "threshold": thr, "pass": bool(d <= thr),
                     "synth_mean": round(float(ss[k].mean()), 2),
                     "real_mean": round(float(rs[k].mean()), 2)}
    report["all_pass"] = all(v["pass"] for k, v in report.items() if isinstance(v, dict))
    return report


def autotune_camera(real: list[np.ndarray], make_batch, cam: CameraParams,
                    rounds: int = 8, batch: int = 160, log=print) -> tuple[CameraParams, dict]:
    """Greedy auto-tune of camera params until validation passes (or rounds end).

    make_batch(cam, n) -> list[np.ndarray] of degraded synthetic frames.
    """
    report = None
    for r in range(rounds):
        synth = make_batch(cam, batch)
        report = validate_against_real(synth, real)
        log(f"[autotune r{r}] " + " ".join(
            f"{k}:W1={v['W1']}{'(ok)' if v['pass'] else '(X)'}"
            for k, v in report.items() if isinstance(v, dict)))
        if report["all_pass"]:
            break
        # greedy nudges, one per failing stat
        if not report["lum_mean"]["pass"]:
            cam.brightness += 0.5 * (report["lum_mean"]["real_mean"] - report["lum_mean"]["synth_mean"])
        if not report["lum_std"]["pass"]:
            cam.contrast *= 1.0 + 0.25 * math.copysign(
                1, report["lum_std"]["real_mean"] - report["lum_std"]["synth_mean"])
        if not report["grad_energy"]["pass"]:
            d = report["grad_energy"]["real_mean"] - report["grad_energy"]["synth_mean"]
            if d < 0:
                cam.blur_sigma = min(2.2, cam.blur_sigma * 1.3)
            else:
                cam.blur_sigma = max(0.2, cam.blur_sigma * 0.75)
                cam.noise_std = min(14.0, cam.noise_std * 1.2)
        if not report["log_sharpness"]["pass"]:
            d = report["log_sharpness"]["real_mean"] - report["log_sharpness"]["synth_mean"]
            cam.blur_sigma = max(0.2, min(2.2, cam.blur_sigma * (0.8 if d > 0 else 1.25)))
        if not report["color_entropy"]["pass"]:
            cam.wb_shift = min(16.0, cam.wb_shift * 1.25)
    return cam, report


# ----------------------------------------------------------------------------
# dataset builder
# ----------------------------------------------------------------------------

@dataclass
class BuildSpec:
    n_qa: int
    n_ground: int
    n_fd: int
    hard: bool
    real_bg_ratio: float
    seed: int


def _to_u8(img: Image.Image) -> np.ndarray:
    return np.asarray(img, dtype=np.uint8)


def build_split(spec: BuildSpec, cam: CameraParams, real_crops: list[Image.Image],
                log=print) -> dict:
    """Generate one split. Returns dict of stacked tensors + metadata lists."""
    rng = random.Random(spec.seed)
    seen: set[str] = set()
    rejected = {"dedup": 0, "judge": 0, "template": 0}

    def background():
        if real_crops and rng.random() < spec.real_bg_ratio:
            return rng.choice(real_crops)
        return synth_background(rng)

    qa = {"img": [], "q": [], "a": [], "meta": []}
    while len(qa["img"]) < spec.n_qa:
        held = rng.random() < 0.35
        objs = place_objects(rng, 1 if held else rng.randint(1, 4), held, spec.hard)
        if not objs:
            continue
        out = make_qa(objs, rng, held)
        if out is None:
            rejected["template"] += 1
            continue
        q, a, target, kind = out
        ok, why = judge(objs, target, kind)
        if not ok:
            rejected["judge"] += 1
            continue
        h = scene_hash(objs, q)
        if h in seen:
            rejected["dedup"] += 1
            continue
        seen.add(h)
        img = render_scene(objs, background(), cam, rng)
        qa["img"].append(_to_u8(img))
        qa["q"].append(q)
        qa["a"].append(a)
        qa["meta"].append({"kind": kind, "caption": structured_caption(objs)})

    gr = {"img": [], "q": [], "bbox": [], "meta": []}
    while len(gr["img"]) < spec.n_ground:
        held = rng.random() < 0.2
        objs = place_objects(rng, 1 if held else rng.randint(2, 4), held, spec.hard)
        if len(objs) < 1:
            continue
        out = make_ground(objs, rng)
        if out is None:
            rejected["template"] += 1
            continue
        q, target = out
        ok, why = judge(objs, target, "ground")
        if not ok:
            rejected["judge"] += 1
            continue
        h = scene_hash(objs, q)
        if h in seen:
            rejected["dedup"] += 1
            continue
        seen.add(h)
        img = render_scene(objs, background(), cam, rng)
        gr["img"].append(_to_u8(img))
        gr["q"].append(q)
        gr["bbox"].append(list(obj_bbox(target)))
        gr["meta"].append({"kind": "ground", "caption": structured_caption(objs)})

    fd = {"img": [], "img1": [], "q": [], "meta": []}
    while len(fd["img"]) < spec.n_fd:
        img_t, img_t1, text, meta = make_fd(rng, cam, background())
        h = hashlib.md5((text + json.dumps(meta)).encode()).hexdigest()[:16]
        if h in seen:
            rejected["dedup"] += 1
            continue
        seen.add(h)
        fd["img"].append(_to_u8(img_t))
        fd["img1"].append(_to_u8(img_t1))
        fd["q"].append(text)
        fd["meta"].append(meta)

    log(f"  built qa={len(qa['img'])} ground={len(gr['img'])} fd={len(fd['img'])} "
        f"rejected={rejected}")
    return {
        "qa": {"img": torch.from_numpy(np.stack(qa["img"])), "q": qa["q"], "a": qa["a"],
               "meta": qa["meta"]},
        "ground": {"img": torch.from_numpy(np.stack(gr["img"])), "q": gr["q"],
                   "bbox": torch.tensor(gr["bbox"], dtype=torch.float32), "meta": gr["meta"]},
        "fd": {"img": torch.from_numpy(np.stack(fd["img"])),
               "img1": torch.from_numpy(np.stack(fd["img1"])), "q": fd["q"], "meta": fd["meta"]},
        "rejected": rejected,
        "camera": cam.to_dict(),
    }


def serialize_caption(objs: list[dict], max_bytes: int = 60) -> str:
    """Structured caption -> compact Vietnamese text prompt for T2I.

    'hình tròn màu đỏ bên trái; ngôi sao màu vàng ở giữa'
    """
    parts = [f"{o['shape']} màu {o['color']} {loc_word(o['cx'])}" for o in objs]
    text = "; ".join(parts)
    while len(text.encode("utf-8")) > max_bytes and len(parts) > 1:
        parts.pop()
        text = "; ".join(parts)
    return text


def quality_score(objs: list[dict]) -> float:
    """Sample-quality proxy in [0,1] — the micro analog of Cosmos' AI-judge score.

    High = large, unoccluded, well-separated objects (post-training tier);
    the pretrain tier accepts everything the basic judge accepts (threshold-2 vs
    threshold-5 curation, Cosmos §3.1.1)."""
    if not objs:
        return 0.0
    size = min(1.0, sum(o["size"] for o in objs) / len(objs) / 0.35)
    vis = min(visible_fraction(o, objs[i + 1:]) for i, o in enumerate(objs))
    sep = 1.0
    for i, a in enumerate(objs):
        for b in objs[i + 1:]:
            d = math.hypot(a["cx"] - b["cx"], a["cy"] - b["cy"]) / ((a["size"] + b["size"]) / 2 + 1e-6)
            sep = min(sep, min(1.0, d))
    return round(0.4 * size + 0.4 * vis + 0.2 * sep, 4)


def make_t2i(rng: random.Random, cam: CameraParams, bg) -> tuple | None:
    """One T2I pair: (caption_text, degraded_image_u8, quality, structured_caption)."""
    objs = place_objects(rng, rng.randint(1, 2), held=rng.random() < 0.3, hard=False)
    if not objs:
        return None
    ok, _ = judge(objs, objs[0], "ground")
    if not ok:
        return None
    caption = serialize_caption(objs)
    img = render_scene(objs, bg, cam, rng)
    return caption, np.asarray(img, dtype=np.uint8), quality_score(objs), structured_caption(objs)


def real_reference_images(paths: list[str], per_frame: int = 40, seed: int = 0) -> list[np.ndarray]:
    """64x64 crops of real frames — the validation reference distribution."""
    rng = random.Random(seed)
    out = []
    for c in load_real_crops(paths, per_frame, rng, size=OUT):
        out.append(np.asarray(c, dtype=np.uint8))
    return out
