"""Crash-safe, append-only telemetry for the complete JARVIS turn pipeline."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import threading
from typing import Any

import config


_LOCK = threading.Lock()


def _log_path() -> Path:
    configured = Path(getattr(config, "RUNTIME_EVENT_LOG", "data/runtime/events.jsonl")).expanduser()
    if not configured.is_absolute():
        configured = Path(__file__).resolve().parents[1] / configured
    return configured.resolve()


def event(event_type: str, **payload: Any) -> None:
    """Record an event without ever being allowed to break the assistant."""
    row = {
        "schema_version": "jarvis-runtime-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "event_type": event_type,
        "process_id": os.getpid(),
        "thread": threading.current_thread().name,
        **payload,
    }
    try:
        path = _log_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(row, ensure_ascii=False, default=str)
        with _LOCK:
            with path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
    except Exception:
        # Observability is important, but it must never become a new failure mode.
        return
