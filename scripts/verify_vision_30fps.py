# -*- coding: utf-8 -*-
"""Headless verification of the 30fps vision pipeline on the REAL camera.

Runs VisionController with the real capture + display threads for ~10s,
replacing the HUD push with a cost-identical stub (resize + JPEG + base64) and
disabling the Qwen reasoner preload (GPU is busy / not needed for this test).
Reports measured capture fps, display fps, and unique-frame ratio.
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
config.VISION_LOG_ROOT = str(ROOT / "data" / "tmp_vision_test")  # isolate session logs

from core import hud, vision  # noqa: E402

pushed = {"n": 0, "hashes": set(), "encode_ms": []}


def fake_push(frame_bgr, max_width=None):
    t0 = time.perf_counter()
    h, w = frame_bgr.shape[:2]
    limit = int(max_width or config.VISION_DISPLAY_WIDTH)
    img = frame_bgr
    if w > limit:
        s = limit / w
        img = cv2.resize(img, (limit, max(1, round(h * s))), interpolation=cv2.INTER_AREA)
    ok, enc = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY),
                                         int(config.VISION_DISPLAY_JPEG_QUALITY)])
    b = base64.b64encode(enc.tobytes())
    pushed["n"] += 1
    pushed["hashes"].add(hash(b[:4096]))
    pushed["encode_ms"].append((time.perf_counter() - t0) * 1000)


def main() -> int:
    hud.update_camera_frame_bgr = fake_push          # cost-identical stub
    vision.hud.update_camera_frame_bgr = fake_push
    ctrl = vision.VisionController()
    ctrl._preload_reasoner_async = lambda: None      # GPU is training JWM v2

    print("mở camera + 2 thread (capture native / display 30Hz)...")
    reply = ctrl.start()
    print("  ->", reply)
    t0 = time.time()
    DURATION = 10.0
    time.sleep(DURATION)
    n_pushed = pushed["n"]
    elapsed = time.time() - t0
    ctrl.stop()

    disp_fps = n_pushed / elapsed
    uniq = len(pushed["hashes"])
    enc = sorted(pushed["encode_ms"])
    print(f"\nKẾT QUẢ ({elapsed:.1f}s):")
    print(f"  display push rate : {disp_fps:5.1f} fps (mục tiêu {config.VISION_DISPLAY_FPS})")
    print(f"  khung hình RIÊNG BIỆT đã render: {uniq} ({uniq/elapsed:.1f}/s) "
          f"[interpolate={config.VISION_DISPLAY_INTERPOLATE}]")
    print(f"  chi phí encode/push: p50={enc[len(enc)//2]:.1f}ms p95={enc[int(len(enc)*0.95)]:.1f}ms")
    ok = disp_fps >= config.VISION_DISPLAY_FPS * 0.93
    print(f"\n{'DAT 30FPS' if ok else 'CHUA DAT'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
