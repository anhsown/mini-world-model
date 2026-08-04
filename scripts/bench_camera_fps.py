# -*- coding: utf-8 -*-
"""Benchmark the 30fps vision pipeline on the real camera.

Measures three stages so the bottleneck is unambiguous:
  1. raw camera read fps (what the sensor/driver actually delivers)
  2. read + BGR->RGB convert (reasoner/memory path)
  3. read + resize + JPEG encode + base64 (full HUD path minus evaluate_js)
"""

from __future__ import annotations

import base64
import sys
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import config  # noqa: E402


def measure(fn, seconds: float = 4.0) -> tuple[float, float]:
    """Returns (fps, mean_ms)."""
    n = 0
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < seconds:
        fn()
        n += 1
    dt = time.perf_counter() - t0
    return n / dt, dt / max(n, 1) * 1000


def main() -> int:
    idx = int(getattr(config, "VISION_CAMERA_INDEX", 0))
    width = int(getattr(config, "VISION_CAMERA_WIDTH", 1280))
    height = int(getattr(config, "VISION_CAMERA_HEIGHT", 720))
    target_fps = int(getattr(config, "VISION_CAMERA_FPS", 30))
    disp_w = int(getattr(config, "VISION_DISPLAY_WIDTH", 800))
    quality = int(getattr(config, "VISION_DISPLAY_JPEG_QUALITY", 60))

    cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, target_fps)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        print("KHONG MO DUOC CAMERA (dang bi chiem? khong co camera?)")
        return 1
    reported = cap.get(cv2.CAP_PROP_FPS)
    ok = False
    for _ in range(5):
        ok, frame = cap.read()
    if not ok or frame is None:
        print("camera mo duoc nhung khong doc duoc frame")
        return 1
    print(f"camera idx={idx} {frame.shape[1]}x{frame.shape[0]} | driver bao cao FPS={reported:.0f}")

    fps1, ms1 = measure(lambda: cap.read())
    print(f"1) raw read              : {fps1:5.1f} fps ({ms1:5.1f} ms/frame)")

    def read_convert():
        _, f = cap.read()
        cv2.cvtColor(f, cv2.COLOR_BGR2RGB)

    fps2, ms2 = measure(read_convert)
    print(f"2) read + BGR->RGB       : {fps2:5.1f} fps ({ms2:5.1f} ms/frame)")

    def full_hud_path():
        _, f = cap.read()
        cv2.cvtColor(f, cv2.COLOR_BGR2RGB)          # reasoner path (worst case: every frame)
        h, w = f.shape[:2]
        if w > disp_w:
            s = disp_w / w
            f = cv2.resize(f, (disp_w, max(1, round(h * s))), interpolation=cv2.INTER_AREA)
        ok2, enc = cv2.imencode(".jpg", f, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        base64.b64encode(enc.tobytes())

    fps3, ms3 = measure(full_hud_path)
    print(f"3) full pipeline (no JS) : {fps3:5.1f} fps ({ms3:5.1f} ms/frame) "
          f"[resize->{disp_w}px, jpeg q{quality}]")

    cap.release()
    budget = 1000.0 / target_fps
    verdict = "DAT" if fps3 >= target_fps * 0.95 else "CHUA DAT"
    print(f"\nKet luan: muc tieu {target_fps}fps (ngan sach {budget:.1f}ms/frame) -> {verdict}")
    print("(evaluate_js cua pywebview chay async fire-and-forget, khong chan camera thread)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
