"""Stage-runner infrastructure: checkpoint chaining, within-stage resume,
data assembly, reports. Every stage is shutdown-safe (partial every 500 steps)."""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch

from ..config import JWMConfig
from ..configs import pipeline_scale
from ..model import JWM, ConvAE
from ..sdg import CameraParams

ROOT = Path(__file__).resolve().parents[2]
CKPT_DIR = ROOT / "jwm" / "checkpoints"
DATA_DIR = ROOT / "data" / "jwm_v3"
LEGACY_DIR = ROOT / "data" / "jwm_sdg"
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def stage_ckpt(name: str) -> Path:
    return CKPT_DIR / f"stage_{name}.pt"


def stage_partial(name: str) -> Path:
    return CKPT_DIR / f"stage_{name}.partial.pt"


def stage_report_path(name: str) -> Path:
    return CKPT_DIR / f"stage_{name}.report.json"


def load_data(*names: str) -> dict:
    """Merge builder outputs into one split dict keyed by mode (qa/ground/fd/t2i)."""
    out: dict = {}
    for n in names:
        payload = torch.load(DATA_DIR / f"{n}.pt", weights_only=False)
        for k, v in payload.items():
            if k.endswith("_idx"):
                out[f"{n}:{k}"] = v
            elif k in out:
                raise ValueError(f"duplicate mode key {k} while merging {n}")
            else:
                out[k] = v
    return out


def subset(cols: dict, idx: list[int]) -> dict:
    """Index-select a column-dict (tensors and lists) — used for post-tier data."""
    out = {}
    for k, v in cols.items():
        if torch.is_tensor(v):
            out[k] = v[idx]
        else:
            out[k] = [v[i] for i in idx]
    return out


def load_valtest() -> tuple[dict, dict]:
    """Legacy val/test (qa/ground/fd — identical across model generations) plus
    the new T2I evaluation pairs."""
    val = torch.load(LEGACY_DIR / "val.pt", weights_only=False)
    test = torch.load(LEGACY_DIR / "test.pt", weights_only=False)
    t2i = torch.load(DATA_DIR / "t2i_valtest.pt", weights_only=False)
    val["t2i"] = t2i["val"]
    test["t2i"] = t2i["test"]
    return val, test


class StageRunner:
    """Uniform lifecycle for one training stage."""

    def __init__(self, name: str, prev: str | None, log=print):
        self.name = name
        self.prev = prev
        self.log = log
        CKPT_DIR.mkdir(parents=True, exist_ok=True)

    # ---------------- lifecycle ----------------

    def output_exists(self) -> bool:
        return stage_ckpt(self.name).exists()

    def load(self) -> tuple[JWM, ConvAE, CameraParams, dict, int]:
        """Returns (model, ae, camera, flags, steps_already_done_in_this_stage)."""
        partial = stage_partial(self.name)
        if partial.exists():
            payload = torch.load(partial, map_location=DEV, weights_only=False)
            done = int(payload.get("stage_steps_done", 0))
            self.log(f"  [{self.name}] RESUME from partial ({done} steps done)")
        elif self.prev is not None:
            src = stage_ckpt(self.prev)
            if not src.exists():
                raise FileNotFoundError(f"stage {self.name} needs {src.name} — run {self.prev} first")
            payload = torch.load(src, map_location=DEV, weights_only=False)
            done = 0
        else:
            payload = None
            done = 0

        if payload is None:
            cfg = pipeline_scale()
            model = JWM(cfg)
            ae = ConvAE(cfg.z_ch)
            cam_file = DATA_DIR / "camera.json"
            cam = CameraParams(**json.load(cam_file.open(encoding="utf-8"))["camera"])
            flags = {"ae_trained": False, "generator_initialized": False}
        else:
            cfg = JWMConfig(**{k: v for k, v in payload["cfg"].items()
                               if k in JWMConfig.__dataclass_fields__})
            model = JWM(cfg)
            model.load_state_dict(payload["model"])
            ae = ConvAE(cfg.z_ch)
            ae.load_state_dict(payload["ae"])
            cam = CameraParams(**payload["camera"])
            flags = dict(payload.get("flags") or
                         {"ae_trained": False, "generator_initialized": False})
        model.to(DEV)
        ae.to(DEV).eval()
        if flags.get("ae_trained"):
            for p in ae.parameters():
                p.requires_grad_(False)
        self.cfg = cfg
        return model, ae, cam, flags, done

    def _payload(self, model, ae, cam, flags, extra=None):
        return {"model": model.state_dict(), "ae": ae.state_dict(),
                "cfg": self.cfg.__dict__, "camera": cam.to_dict(), "flags": flags,
                "stage": self.name, **(extra or {})}

    def ckpt_fn(self, model, ae, cam, flags, base_done: int):
        """Shutdown-safe periodic saver for train_stage(ckpt_fn=...)."""
        def fn(done_now: int):
            p = self._payload(model, ae, cam, flags,
                              {"stage_steps_done": base_done + done_now,
                               "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S")})
            torch.save(p, stage_partial(self.name))
        return fn

    def finish(self, model, ae, cam, flags, report: dict, extra=None) -> None:
        torch.save(self._payload(model, ae, cam, flags, extra), stage_ckpt(self.name))
        report = {"stage": self.name, "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                  "params_M": round(model.num_params() / 1e6, 2), **report}
        json.dump(report, stage_report_path(self.name).open("w", encoding="utf-8"),
                  ensure_ascii=False, indent=1)
        partial = stage_partial(self.name)
        if partial.exists():
            partial.unlink()
        self.log(f"  [{self.name}] DONE -> {stage_ckpt(self.name).name}")
        self.log("  report: " + json.dumps(
            {k: v for k, v in report.items() if not isinstance(v, (list, dict))},
            ensure_ascii=False))
