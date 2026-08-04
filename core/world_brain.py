"""World Brain — JWM (Jarvis World Model) integration for JARVIS.

Loads the trained micro world model (jwm/checkpoints/jwm_v1.pt) and exposes:
  * analyze(frame, query)  -> answer + predicted region (bbox) + CALIBRATED
                              confidence + abstain flag + latency breakdown
  * imagine(frame, motion) -> forward-dynamics prediction of the next frame

Deployment modes (config.WORLD_BRAIN_MODE):
  "off"     — not loaded
  "shadow"  — runs alongside the existing Qwen3-VL reasoner; its predictions are
              logged for evaluation but never spoken to the user (safe onboarding)
  "primary" — its answers are used directly (synthetic-domain scenes only)

Includes the inference-time REFLECTION PASS (improvement #3 from
jwm/COSMOS3_CRITIQUE.md): after the generator denoises a bbox, the reasoner
re-inspects the cropped region and the confidence is cross-checked — a
pipeline-level DM->AR feedback loop the Cosmos 3 attention scheme forbids.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

_ROOT = Path(__file__).resolve().parents[1]


def _latest_checkpoint() -> Path:
    """Prefer the newest model generation available (v2 > v1)."""
    ckpt_dir = _ROOT / "jwm" / "checkpoints"
    for name in ("jwm_v4.pt", "jwm_v3.pt", "jwm_v2.pt", "jwm_v1.pt"):
        if (ckpt_dir / name).exists():
            return ckpt_dir / name
    return ckpt_dir / "jwm_v1.pt"



_COLOR_WORDS = ("đỏ", "xanh dương", "xanh lá", "vàng", "cam", "tím",
                "red", "blue", "green", "yellow", "orange", "purple")
_VI_EN = {"red": "đỏ", "blue": "xanh dương", "green": "xanh lá",
          "yellow": "vàng", "orange": "cam", "purple": "tím"}


class WorldBrain:
    def __init__(self, ckpt_path: Path | str | None = None, device: str | None = None,
                 abstain_threshold: float = 0.35):
        ckpt_path = Path(ckpt_path) if ckpt_path else _latest_checkpoint()
        self.ckpt_path = ckpt_path
        import sys
        if str(_ROOT) not in sys.path:
            sys.path.insert(0, str(_ROOT))
        from jwm import JWM, JWMConfig, ConvAE

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        payload = torch.load(str(ckpt_path), map_location=self.device, weights_only=False)
        self.cfg = JWMConfig(**{k: v for k, v in payload["cfg"].items()
                                if k in JWMConfig.__dataclass_fields__})
        self.model = JWM(self.cfg)
        self.model.load_state_dict(payload["model"])
        self.model.to(self.device).eval()
        self.ae = ConvAE(self.cfg.z_ch)
        self.ae.load_state_dict(payload["ae"])
        self.ae.to(self.device).eval()
        self.metrics = payload.get("metrics", {})
        # Post-hoc Platt calibration (fit on val at the fast step count):
        # conf_calibrated = sigmoid(a * logit(conf_raw) + b)
        self.calibration = payload.get("calibration")
        # With calibrated P(IoU>=0.5), abstaining below 0.5 is the principled rule.
        self.abstain_threshold = 0.5 if self.calibration else abstain_threshold

    # ------------------------------------------------------------------
    def _prep(self, frame: np.ndarray) -> torch.Tensor:
        """RGB uint8 (H,W,3) any size -> (1,3,64,64) float [0,1]."""
        img = Image.fromarray(frame).convert("RGB").resize(
            (self.cfg.image_size, self.cfg.image_size), Image.LANCZOS)
        t = torch.from_numpy(np.array(img, dtype=np.uint8, copy=True))
        return t.permute(2, 0, 1).float().div(255.0).unsqueeze(0).to(self.device)

    def _text(self, q: str):
        from jwm.data import pad_text
        ids, valid = pad_text([q], self.cfg.max_q_bytes)
        return ids.to(self.device), valid.to(self.device)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def analyze(self, frame: np.ndarray, query: str, *, steps: int | None = None,
                reflect: bool = True) -> dict:
        """Full pipeline: QA answer + grounded region + calibrated confidence."""
        t_all = time.perf_counter()
        from jwm.model import merge_latent

        img = self._prep(frame)
        q_ids, q_valid = self._text(query)

        t0 = time.perf_counter()
        answer = self.model.generate_answer(img, q_ids, q_valid)[0]
        t_qa = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        z = merge_latent(self.ae.encode_std(img))
        n_steps = steps or (self.calibration["steps"] if self.calibration
                            else self.cfg.sample_steps_fast)
        bbox_t, conf_t = self.model.sample_bbox(
            img, q_ids, q_valid, z, steps=n_steps, shift=self.cfg.sample_shift)
        bbox = [float(v) for v in bbox_t[0]]
        confidence = float(conf_t[0])
        if self.calibration:
            c = min(max(confidence, 1e-4), 1 - 1e-4)
            import math
            confidence = 1.0 / (1.0 + math.exp(-(self.calibration["a"]
                                                 * math.log(c / (1 - c))
                                                 + self.calibration["b"])))
        t_ground = (time.perf_counter() - t0) * 1000

        reflection = None
        if reflect:
            t0 = time.perf_counter()
            reflection = self._reflect(frame, query, bbox)
            if reflection["verdict"] == "contradicted":
                confidence *= 0.5
            elif reflection["verdict"] == "supported":
                confidence = min(1.0, confidence * 1.1)
            t_reflect = (time.perf_counter() - t0) * 1000
        else:
            t_reflect = 0.0

        return {
            "answer": answer,
            "bbox": bbox,                                # (cx, cy, w, h) normalized
            "confidence": round(confidence, 4),
            "abstain": confidence < self.abstain_threshold,
            "reflection": reflection,
            "latency_ms": {
                "qa": round(t_qa, 1),
                "ground": round(t_ground, 1),
                "reflect": round(t_reflect, 1),
                "total": round((time.perf_counter() - t_all) * 1000, 1),
            },
        }

    @torch.no_grad()
    def _reflect(self, frame: np.ndarray, query: str, bbox: list[float]) -> dict:
        """Reflection pass: crop the predicted region, let the REASONER re-inspect
        it, and check consistency with the color mentioned in the query."""
        H, W = frame.shape[:2]
        cx, cy, w, h = bbox
        m = 1.6                                          # crop margin around the box
        x1 = int(max(0, (cx - w * m / 2) * W))
        x2 = int(min(W, (cx + w * m / 2) * W))
        y1 = int(max(0, (cy - h * m / 2) * H))
        y2 = int(min(H, (cy + h * m / 2) * H))
        if x2 - x1 < 8 or y2 - y1 < 8:
            return {"verdict": "unverifiable", "crop_answer": None}
        crop = frame[y1:y2, x1:x2]
        img = self._prep(crop)
        q_ids, q_valid = self._text("vật ở giữa khung hình là gì?")
        crop_answer = self.model.generate_answer(img, q_ids, q_valid)[0]

        q_low = query.lower()
        asked = [c for c in _COLOR_WORDS if c in q_low]
        asked_vi = {_VI_EN.get(c, c) for c in asked}
        if not asked_vi:
            return {"verdict": "unverifiable", "crop_answer": crop_answer}
        if any(c in crop_answer.lower() for c in asked_vi):
            return {"verdict": "supported", "crop_answer": crop_answer}
        return {"verdict": "contradicted", "crop_answer": crop_answer}

    @torch.no_grad()
    def imagine(self, frame: np.ndarray, motion_text: str, *, k: int = 1,
                steps: int | None = None) -> list[np.ndarray]:
        """Forward dynamics: predicted next frame(s) as RGB uint8 (64,64,3)."""
        from jwm.model import merge_latent, unmerge_latent

        img = self._prep(frame)
        q_ids, q_valid = self._text(motion_text)
        z_cur = merge_latent(self.ae.encode_std(img))
        outs = []
        for _ in range(k):
            z_hat = self.model.sample_next_latent(
                img, q_ids, q_valid, z_cur, steps=steps or self.cfg.sample_steps,
                shift=self.cfg.sample_shift)
            rec = self.ae.decode_std(unmerge_latent(z_hat, C=self.cfg.z_ch,
                                                    grid=self.cfg.lat_grid))
            outs.append((rec[0].permute(1, 2, 0).clamp(0, 1).cpu().numpy()
                         * 255).astype(np.uint8))
        return outs


_brain: WorldBrain | None = None


def get_brain() -> WorldBrain | None:
    """Lazy singleton. Returns None when disabled or checkpoint missing."""
    global _brain
    import config
    mode = getattr(config, "WORLD_BRAIN_MODE", "off")
    if mode == "off":
        return None
    if _brain is None and _latest_checkpoint().exists():
        _brain = WorldBrain()
    return _brain


def available() -> bool:
    return _latest_checkpoint().exists()
