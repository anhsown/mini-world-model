"""End-to-end diagnostic: recorded/live query -> camera -> isolated reasoner."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import speech_recognition as sr

from core import speech, vision


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", type=Path)
    parser.add_argument("--query", default="What am I holding?")
    args = parser.parse_args()
    report: dict = {}
    query = args.query

    if args.audio:
        with sr.AudioFile(str(args.audio)) as source:
            audio = sr.Recognizer().record(source)
        speech.preload()
        started = time.perf_counter()
        query = speech._transcribe_whisper(audio, context="vision") or query
        report["asr"] = {
            "query": query,
            "wall_ms": round((time.perf_counter() - started) * 1000.0, 2),
            "diagnostics": speech.last_diagnostics(),
        }

    try:
        vision.preload_async()
        deadline = time.monotonic() + 70
        while time.monotonic() < deadline and vision.status() not in ("preloaded", "ready", "reasoner_error"):
            time.sleep(0.25)
        report["preload_status"] = vision.status()

        started = time.perf_counter()
        report["camera_reply"] = vision.start()
        report["camera_start_ms"] = round((time.perf_counter() - started) * 1000.0, 2)

        started = time.perf_counter()
        report["answer"] = vision.ask(query)
        report["vision_wall_ms"] = round((time.perf_counter() - started) * 1000.0, 2)
        report["vision_status_after"] = vision.status()
        report["session_dir"] = str(vision.session_dir())
        report["still_active_after_answer"] = vision.active()
    except Exception as exc:
        report["fatal"] = repr(exc)
    finally:
        vision.shutdown()

    print("E2E_RESULT=" + json.dumps(report, ensure_ascii=False, default=str), flush=True)
    return 0 if "fatal" not in report else 1


if __name__ == "__main__":
    raise SystemExit(main())

