"""Visual smoke test for the fullscreen JARVIS HUD.

This intentionally does not load the visual reasoner. It verifies only the
desktop window, fullscreen layout, DirectShow feed, and Python-to-HUD bridge.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import cv2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core import hud


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("idle", "camera"), default="idle")
    parser.add_argument("--duration", type=float, default=15.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    def run() -> None:
        hud.show()
        if args.mode == "idle":
            hud.enter_idle_mode()
            time.sleep(args.duration)
            return

        hud.enter_camera_mode()
        hud.set_vision_status("CAMERA PREVIEW · UI SMOKE TEST")
        hud.set_transcript("Preview mode", "Live DirectShow feed — Reasoner not loaded in this visual test.")
        capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        deadline = time.monotonic() + args.duration
        try:
            while time.monotonic() < deadline:
                ok, frame_bgr = capture.read()
                if ok and frame_bgr is not None:
                    hud.update_camera_frame(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
                time.sleep(0.12)
        finally:
            capture.release()

    hud.start(run)


if __name__ == "__main__":
    main()
