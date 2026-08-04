"""Camera and visual-reasoning mode for the desktop JARVIS application."""

from __future__ import annotations

from concurrent.futures import Future, TimeoutError as FutureTimeoutError
from datetime import datetime, timezone
import json
from pathlib import Path
import threading
import time
from typing import Any
from uuid import uuid4

import cv2
import numpy as np
from PIL import Image, ImageDraw

import config
from core import hud, telemetry
from core.vision_runtime import IsolatedQwen3VLReasoner, ReasonerConfig, TemporalFrameMemory


OPEN_CAMERA_TERMS = (
    "mo camera",
    "bat camera",
    "truy cap camera",
    "kich hoat camera",
    "mo thi giac",
    "bat thi giac",
    "nhin xung quanh",
    "quan sat xung quanh",
    "camera on",
    "open camera",
    "enable vision",
)

CLOSE_CAMERA_TERMS = (
    "tat camera",
    "dong camera",
    "dung camera",
    "ngung quan sat",
    "tat thi giac",
    "camera off",
    "close camera",
    "disable vision",
)

_CAMERA_OBJECT_TERMS = ("camera", "cam", "thi giac", "vision")
_OPEN_ACTION_TERMS = (
    "mo",
    "bat",
    "truy cap",
    "kich hoat",
    "nhin",
    "quan sat",
    "open",
    "access",
    "enable",
)
_CLOSE_ACTION_TERMS = ("tat", "dong", "dung", "ngung", "close", "disable", "off")


def is_open_camera_command(normalized_text: str) -> bool:
    text = " ".join(normalized_text.casefold().split())
    if any(term in text for term in CLOSE_CAMERA_TERMS):
        return False
    if any(term in text for term in OPEN_CAMERA_TERMS):
        return True
    has_camera_object = any(term in text for term in _CAMERA_OBJECT_TERMS)
    has_open_action = any(term in text for term in _OPEN_ACTION_TERMS)
    return has_camera_object and has_open_action


def is_close_camera_command(normalized_text: str) -> bool:
    text = " ".join(normalized_text.casefold().split())
    if any(term in text for term in CLOSE_CAMERA_TERMS):
        return True
    has_camera_object = any(term in text for term in _CAMERA_OBJECT_TERMS)
    has_close_action = any(term in text for term in _CLOSE_ACTION_TERMS)
    return has_camera_object and has_close_action


class VisionController:
    def __init__(self) -> None:
        self.memory = TemporalFrameMemory(max_frames=getattr(config, "VISION_MEMORY_FRAMES", 8))
        self.reasoner = IsolatedQwen3VLReasoner(
            ReasonerConfig(
                model_id=getattr(
                    config,
                    "VISION_MODEL",
                    "unsloth/Qwen3-VL-2B-Instruct-bnb-4bit",
                ),
                max_new_tokens=getattr(config, "VISION_MAX_NEW_TOKENS", 160),
                max_image_edge=getattr(config, "VISION_MAX_IMAGE_EDGE", 448),
                max_gpu_memory=getattr(config, "VISION_MAX_GPU_MEMORY", "2300MiB"),
                max_cpu_memory=getattr(config, "VISION_MAX_CPU_MEMORY", "8GiB"),
            ),
            startup_timeout=float(getattr(config, "VISION_WORKER_STARTUP_TIMEOUT", 60)),
            response_timeout=float(getattr(config, "VISION_WORKER_RESPONSE_TIMEOUT", 120)),
        )
        self._capture: cv2.VideoCapture | None = None
        self._capture_thread: threading.Thread | None = None
        self._display_thread: threading.Thread | None = None
        self._bgr_lock = threading.Lock()
        self._bgr_prev: np.ndarray | None = None      # (frame, arrival_time)
        self._bgr_prev_t = 0.0
        self._bgr_last: np.ndarray | None = None
        self._bgr_last_t = 0.0
        self._preload_thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._frame_lock = threading.Lock()
        self._model_lock = threading.Lock()
        self._inference_state_lock = threading.Lock()
        self._log_lock = threading.Lock()
        self._reasoner_ready = threading.Event()
        self._inference_future: Future | None = None
        self._inference_query = ""
        self._latest_frame: np.ndarray | None = None
        self._active = False
        self._model_error = ""
        self._status = "offline"
        self._session_id = ""
        self._session_dir: Path | None = None

    @property
    def active(self) -> bool:
        return self._active

    @property
    def session_dir(self) -> Path | None:
        return self._session_dir

    @property
    def status(self) -> str:
        return self._status

    def preload_async(self) -> None:
        """Load the visual model in the background independently of camera use."""
        self._preload_reasoner_async()

    def start(self) -> str:
        if self._active:
            hud.enter_camera_mode()
            return "Vision mode is already active, sir."

        started = time.perf_counter()
        camera_index = int(getattr(config, "VISION_CAMERA_INDEX", 0))
        capture, frame_bgr = self._open_capture(camera_index)

        self._begin_session()
        self.memory.clear()
        self._stop_event.clear()
        self._capture = capture
        self._active = True
        self._status = "camera_online"
        first_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        self._accept_frame(first_frame, add_to_memory=True)

        hud.enter_camera_mode()
        hud.set_vision_status("CAMERA ONLINE · LOADING REASONER")
        hud.update_camera_frame(first_frame)
        startup_ms = round((time.perf_counter() - started) * 1000.0, 2)
        self._write_event("camera_started", {"camera_index": camera_index, "startup_ms": startup_ms})
        telemetry.event("camera_started", camera_index=camera_index, startup_ms=startup_ms)

        self._capture_thread = threading.Thread(
            target=self._capture_loop,
            name="jarvis-camera",
            daemon=True,
        )
        self._capture_thread.start()
        self._display_thread = threading.Thread(
            target=self._display_loop,
            name="jarvis-camera-display",
            daemon=True,
        )
        self._display_thread.start()
        self._preload_reasoner_async()
        return "Vision systems are online, sir. You may ask me about what I can see."

    def stop(self) -> str:
        if not self._active:
            hud.enter_idle_mode()
            return "Vision mode is already offline, sir."
        self._write_event("camera_stopping", {})
        self._active = False
        self._status = "stopping"
        self._stop_event.set()
        if self._capture_thread and self._capture_thread.is_alive():
            self._capture_thread.join(timeout=2.5)
        if self._display_thread and self._display_thread.is_alive():
            self._display_thread.join(timeout=1.5)
        self._display_thread = None
        with self._bgr_lock:
            self._bgr_prev = self._bgr_last = None
        if self._capture is not None:
            try:
                self._capture.release()
            except Exception:
                pass
        self._capture = None
        self._capture_thread = None
        self.memory.clear()
        with self._frame_lock:
            self._latest_frame = None
        hud.enter_idle_mode()
        hud.set_vision_status("VISION OFFLINE")
        self._status = "offline"
        telemetry.event("camera_stopped", session_id=self._session_id)
        return "Vision systems are offline, sir."

    def shutdown(self) -> None:
        if self._active:
            self.stop()
        close = getattr(self.reasoner, "close", None)
        if callable(close):
            close()

    def ask(self, query: str) -> str:
        if not self._active:
            self.start()
        frame = self._wait_for_frame(timeout=3.0)
        if frame is None:
            self._write_event("vision_error", {"query": query, "error": "no camera frame"})
            return "I cannot obtain a camera frame at the moment, sir."

        self.memory.add(frame)
        frame_count = max(1, int(getattr(config, "VISION_REASONING_FRAMES", 1)))
        frames = self.memory.latest(frame_count)
        hud.set_transcript(query, "Analysing the current scene...")
        self._preload_reasoner_async()
        self._status = "waiting_for_reasoner"
        hud.set_vision_status("WAITING FOR VISUAL REASONER")

        ready_timeout = float(getattr(config, "VISION_REASONER_READY_TIMEOUT", 30))
        wait_started = time.perf_counter()
        if not self._reasoner_ready.wait(timeout=ready_timeout):
            wait_ms = round((time.perf_counter() - wait_started) * 1000.0, 2)
            if self._model_error:
                message = "My visual reasoner could not be loaded, sir. The camera will remain online while I recover."
                error = self._model_error
                self._status = "reasoner_error"
            else:
                message = "My visual reasoner is still warming up, sir. The camera will remain online; please ask again shortly."
                error = "reasoner readiness timeout"
                self._status = "reasoner_loading"
            self._write_event("vision_not_ready", {"query": query, "error": error, "wait_ms": wait_ms})
            telemetry.event("vision_not_ready", query=query, error=error, wait_ms=wait_ms)
            hud.set_transcript(query, message)
            hud.set_vision_status("CAMERA ONLINE · REASONER WARMING UP")
            return message

        with self._inference_state_lock:
            previous = self._inference_future
            if previous is not None and not previous.done():
                message = "I am still analysing the previous view, sir. The camera remains online."
                self._write_event("vision_busy", {"query": query, "active_query": self._inference_query})
                telemetry.event("vision_busy", query=query, active_query=self._inference_query)
                hud.set_transcript(query, message)
                hud.set_vision_status("REASONER BUSY")
                return message
            future: Future = Future()
            self._inference_future = future
            self._inference_query = query

        def infer() -> None:
            try:
                future.set_result(self.reasoner.analyze(frames, query))
            except BaseException as exc:
                future.set_exception(exc)

        # A daemon worker gives us a real response timeout without allowing a
        # stuck third-party model call to prevent JARVIS from shutting down.
        threading.Thread(target=infer, name="jarvis-vision-inference", daemon=True).start()
        self._status = "reasoning"
        hud.set_vision_status("REASONER ACTIVE")
        inference_timeout = float(getattr(config, "VISION_INFERENCE_TIMEOUT", 45))

        try:
            result = future.result(timeout=inference_timeout)
        except FutureTimeoutError:
            message = "Visual analysis is taking longer than expected, sir. I have kept the camera session alive."
            self._write_event("vision_timeout", {"query": query, "timeout_s": inference_timeout})
            telemetry.event("vision_timeout", query=query, timeout_s=inference_timeout)
            hud.set_transcript(query, message)
            hud.set_vision_status("REASONER BUSY · CAMERA ONLINE")
            self._status = "reasoning_timeout"
            return message
        except Exception as exc:
            message = f"Vision reasoning failed: {exc}"
            self._write_event("vision_error", {"query": query, "error": str(exc)})
            hud.set_transcript(query, message)
            hud.set_vision_status("VISION DEGRADED · CAMERA ONLINE")
            self._status = "reasoner_error"
            telemetry.event("vision_error", query=query, error=str(exc))
            if not self.reasoner.loaded:
                self._reasoner_ready.clear()
                telemetry.event("vision_worker_restart_requested", error=str(exc))
                self._preload_reasoner_async()
            return "My visual reasoner encountered a problem, sir. The camera remains online and the failure has been logged."

        interaction_id = self._record_result(query, frame, result)
        self._shadow_world_brain(query, frame, interaction_id)
        bbox = list(result.evidence.bbox_norm) if result.evidence.bbox_norm else None
        hud.set_vision_result(
            query=query,
            answer=result.answer,
            bbox=bbox,
            confidence=result.confidence,
            latency_ms=result.latency_ms,
            interaction_id=interaction_id,
        )
        hud.set_vision_status("CAMERA ONLINE · REASONER READY")
        self._status = "ready"
        telemetry.event(
            "vision_answered",
            query=query,
            answer=result.answer,
            latency_ms=result.latency_ms,
            confidence=result.confidence,
            interaction_id=interaction_id,
        )
        if result.abstain:
            return "I cannot identify that reliably from the current view, sir."
        return result.answer

    def _preload_reasoner_async(self) -> None:
        if self.reasoner.loaded:
            self._model_error = ""
            self._reasoner_ready.set()
            hud.set_vision_status("CAMERA ONLINE · REASONER READY")
            return
        if self._preload_thread and self._preload_thread.is_alive():
            return
        self._reasoner_ready.clear()

        def load() -> None:
            started = time.perf_counter()
            self._status = "reasoner_loading" if self._active else "preloading"
            try:
                with self._model_lock:
                    self._model_error = ""
                    self.reasoner.load()
                load_ms = round((time.perf_counter() - started) * 1000.0, 2)
                self._reasoner_ready.set()
                self._write_event(
                    "reasoner_loaded",
                    {"model_id": self.reasoner.config.model_id, "load_ms": load_ms},
                )
                telemetry.event(
                    "reasoner_loaded",
                    model_id=self.reasoner.config.model_id,
                    load_ms=load_ms,
                    worker_pid=getattr(self.reasoner, "worker_pid", None),
                )
                hud.set_vision_status("CAMERA ONLINE · REASONER READY")
                self._status = "ready" if self._active else "preloaded"
            except Exception as exc:
                self._model_error = str(exc)
                self._reasoner_ready.clear()
                self._write_event("reasoner_load_failed", {"error": str(exc)})
                telemetry.event("reasoner_load_failed", error=str(exc))
                hud.set_vision_status("REASONER LOAD FAILED")
                self._status = "reasoner_error"

        self._preload_thread = threading.Thread(target=load, name="jarvis-vision-loader", daemon=True)
        self._preload_thread.start()

    def _capture_loop(self) -> None:
        """Producer: reads at the camera's NATIVE rate (V380: ~21.5fps @720p,
        hardware ceiling — measured) and publishes the two most recent BGR frames
        for the 30Hz display thread. Reasoner/memory consume RGB at their own
        cadence. Capture fps is reported via telemetry every 5s."""
        memory_interval = max(0.2, float(getattr(config, "VISION_MEMORY_INTERVAL", 0.75)))
        next_memory = time.monotonic() + memory_interval
        failures = 0
        reads = 0
        window_start = time.monotonic()
        try:
            while not self._stop_event.is_set() and self._capture is not None:
                try:
                    ok, frame_bgr = self._capture.read()
                except Exception as exc:
                    ok, frame_bgr = False, None
                    telemetry.event("camera_read_exception", error=str(exc))
                if not ok or frame_bgr is None:
                    failures += 1
                    maximum_failures = max(3, int(getattr(config, "VISION_CAMERA_MAX_READ_FAILURES", 12)))
                    if failures >= maximum_failures:
                        self._write_event("camera_read_failed", {"consecutive_failures": failures})
                        telemetry.event("camera_signal_lost", consecutive_failures=failures)
                        hud.set_vision_status("CAMERA SIGNAL LOST")
                        self._status = "camera_reconnecting"
                        if self._reconnect_capture():
                            self._write_event("camera_reconnected", {})
                            telemetry.event("camera_reconnected")
                            hud.set_vision_status("CAMERA ONLINE · REASONER READY")
                            self._status = "ready" if self._reasoner_ready.is_set() else "reasoner_loading"
                        failures = 0
                    time.sleep(float(getattr(config, "VISION_CAMERA_RECONNECT_SECONDS", 1.0)))
                    continue
                failures = 0
                now = time.monotonic()
                with self._bgr_lock:
                    self._bgr_prev, self._bgr_prev_t = self._bgr_last, self._bgr_last_t
                    self._bgr_last, self._bgr_last_t = frame_bgr, now
                if now >= next_memory:
                    frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                    self._accept_frame(frame, add_to_memory=True)
                    next_memory = now + memory_interval
                reads += 1
                if now - window_start >= 5.0:
                    telemetry.event("camera_capture_fps",
                                    fps=round(reads / (now - window_start), 1))
                    reads = 0
                    window_start = now
        finally:
            if self._capture is not None:
                self._capture.release()

    def _display_loop(self) -> None:
        """Consumer: renders the HUD at a strict VISION_DISPLAY_FPS (30Hz) clock.

        The camera delivers ~21.5fps, so a plain re-push would repeat pixels and
        the HUD would still LOOK 21.5fps. With VISION_DISPLAY_INTERPOLATE the
        loop cross-fades between the two latest camera frames (standard
        frame-rate conversion): every 33ms tick renders a DISTINCT frame
        alpha = clamp((now - t_last)/(t_last - t_prev)), out = (1-a)*prev + a*last
        at the cost of one addWeighted (~1-2ms) and one capture-period of extra
        latency (~46ms). Auto-degrades display width if the loop can't hold rate.
        """
        target = max(1.0, float(getattr(config, "VISION_DISPLAY_FPS", 30)))
        interval = 1.0 / target
        interpolate = bool(getattr(config, "VISION_DISPLAY_INTERPOLATE", True))
        display_width = int(getattr(config, "VISION_DISPLAY_WIDTH", 800))
        auto_degrade = bool(getattr(config, "VISION_DISPLAY_AUTO_DEGRADE", True))
        shown = 0
        slow_windows = 0
        window_start = time.monotonic()
        next_tick = time.monotonic()
        while not self._stop_event.is_set() and self._active:
            now = time.monotonic()
            if now < next_tick:
                time.sleep(min(0.004, next_tick - now))
                continue
            next_tick += interval
            if next_tick < now:                      # fell behind: resync clock
                next_tick = now + interval
            with self._bgr_lock:
                prev, prev_t = self._bgr_prev, self._bgr_prev_t
                last, last_t = self._bgr_last, self._bgr_last_t
            if last is None:
                continue
            if interpolate and prev is not None and prev.shape == last.shape:
                span = max(1e-3, last_t - prev_t)
                a = min(1.0, max(0.0, (now - last_t) / span))
                frame = cv2.addWeighted(prev, 1.0 - a, last, a, 0) if a < 1.0 else last
            else:
                frame = last
            hud.update_camera_frame_bgr(frame, max_width=display_width)
            shown += 1
            if now - window_start >= 5.0:
                fps = shown / (now - window_start)
                telemetry.event("camera_display_fps", fps=round(fps, 1),
                                width=display_width, target=target,
                                interpolate=interpolate)
                if auto_degrade and fps < 0.8 * target:
                    slow_windows += 1
                    if slow_windows >= 2 and display_width > 512:
                        display_width = 640 if display_width > 640 else 512
                        telemetry.event("camera_display_degraded",
                                        new_width=display_width, measured_fps=round(fps, 1))
                        self._write_event("camera_display_degraded",
                                          {"new_width": display_width, "fps": round(fps, 1)})
                        slow_windows = 0
                else:
                    slow_windows = 0
                shown = 0
                window_start = now

    def _open_capture(self, camera_index: int) -> tuple[cv2.VideoCapture, np.ndarray]:
        backends = []
        for name in ("CAP_DSHOW", "CAP_MSMF", "CAP_ANY"):
            backend = getattr(cv2, name, None)
            if backend is not None and backend not in backends:
                backends.append(backend)
        errors: list[str] = []
        for backend in backends:
            capture = cv2.VideoCapture(camera_index, backend)
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, int(getattr(config, "VISION_CAMERA_WIDTH", 1280)))
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, int(getattr(config, "VISION_CAMERA_HEIGHT", 720)))
            capture.set(cv2.CAP_PROP_FPS, int(getattr(config, "VISION_CAMERA_FPS", 30)))
            capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            if not capture.isOpened():
                errors.append(f"backend={backend}: open failed")
                capture.release()
                continue
            frame_bgr = None
            warmup_frames = max(1, int(getattr(config, "VISION_CAMERA_WARMUP_FRAMES", 3)))
            for _ in range(warmup_frames):
                try:
                    ok, candidate = capture.read()
                except Exception as exc:
                    errors.append(f"backend={backend}: {exc}")
                    ok, candidate = False, None
                if ok and candidate is not None:
                    frame_bgr = candidate
            if frame_bgr is not None:
                return capture, frame_bgr
            errors.append(f"backend={backend}: no frame")
            capture.release()
        raise RuntimeError(
            f"Cannot obtain camera index {camera_index}. " + "; ".join(errors)
        )

    def _reconnect_capture(self) -> bool:
        if self._stop_event.is_set():
            return False
        old_capture = self._capture
        try:
            if old_capture is not None:
                old_capture.release()
            capture, frame_bgr = self._open_capture(int(getattr(config, "VISION_CAMERA_INDEX", 0)))
            self._capture = capture
            frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            self._accept_frame(frame, add_to_memory=True)
            hud.update_camera_frame(frame)
            return True
        except Exception as exc:
            self._write_event("camera_reconnect_failed", {"error": str(exc)})
            telemetry.event("camera_reconnect_failed", error=str(exc))
            return False

    def _accept_frame(self, frame: np.ndarray, *, add_to_memory: bool) -> None:
        rgb = np.ascontiguousarray(frame[:, :, :3].astype(np.uint8))
        with self._frame_lock:
            self._latest_frame = rgb.copy()
        if add_to_memory:
            self.memory.add(rgb)

    def _wait_for_frame(self, timeout: float) -> np.ndarray | None:
        """Freshest frame for the reasoner: convert the newest BGR on demand
        (the RGB memory path only runs at VISION_MEMORY_INTERVAL cadence)."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self._bgr_lock:
                bgr = None if self._bgr_last is None else self._bgr_last.copy()
            if bgr is not None:
                return np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            with self._frame_lock:
                if self._latest_frame is not None:
                    return self._latest_frame.copy()
            time.sleep(0.05)
        return None

    def _begin_session(self) -> None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self._session_id = f"{stamp}-{uuid4().hex[:8]}"
        root = Path(getattr(config, "VISION_LOG_ROOT", "data/vision_sessions")).resolve()
        self._session_dir = root / self._session_id
        (self._session_dir / "frames").mkdir(parents=True, exist_ok=True)
        self._write_event(
            "session_started",
            {
                "session_id": self._session_id,
                "model_id": self.reasoner.config.model_id,
            },
        )

    def _record_result(self, query: str, frame: np.ndarray, result: Any) -> str:
        interaction_id = f"vision-{datetime.now(timezone.utc).strftime('%H%M%S')}-{uuid4().hex[:8]}"
        frame_rel = f"frames/{interaction_id}-raw.jpg"
        overlay_rel = f"frames/{interaction_id}-overlay.jpg"
        if self._session_dir is not None:
            Image.fromarray(frame).save(self._session_dir / frame_rel, quality=92)
            overlay = Image.fromarray(frame.copy())
            if result.evidence.bbox_norm:
                x1, y1, x2, y2 = result.evidence.bbox_norm
                width, height = overlay.size
                draw = ImageDraw.Draw(overlay)
                draw.rectangle(
                    (x1 * width, y1 * height, x2 * width, y2 * height),
                    outline=(0, 238, 255),
                    width=max(3, width // 320),
                )
            overlay.save(self._session_dir / overlay_rel, quality=92)
        self._write_event(
            "vision_result",
            {
                "interaction_id": interaction_id,
                "query": query,
                "answer": result.answer,
                "object_name": result.object_name,
                "bbox_norm": list(result.evidence.bbox_norm) if result.evidence.bbox_norm else None,
                "visual_attributes": result.evidence.visual_attributes,
                "confidence": result.confidence,
                "abstain": result.abstain,
                "latency_ms": result.latency_ms,
                "model_id": result.model_id,
                "raw_model_output": result.raw_text,
                "frame_file": frame_rel,
                "overlay_file": overlay_rel,
            },
        )
        return interaction_id

    def _shadow_world_brain(self, query: str, frame: np.ndarray, interaction_id: str) -> None:
        """WORLD_BRAIN_MODE == "shadow": run the JWM world brain alongside the
        reasoner and log its prediction without changing the spoken answer.
        Never allowed to break a live vision turn."""
        try:
            import config as _cfg
            if getattr(_cfg, "WORLD_BRAIN_MODE", "off") != "shadow":
                return
            from core import world_brain
            brain = world_brain.get_brain()
            if brain is None:
                return
            out = brain.analyze(frame, query, steps=brain.cfg.sample_steps_fast)
            self._write_event(
                "world_brain_shadow",
                {
                    "interaction_id": interaction_id,
                    "query": query,
                    "answer": out["answer"],
                    "bbox": out["bbox"],
                    "confidence": out["confidence"],
                    "abstain": out["abstain"],
                    "latency_ms": out["latency_ms"],
                },
            )
            telemetry.event("world_brain_shadow", interaction_id=interaction_id,
                            answer=out["answer"], confidence=out["confidence"],
                            latency_ms=out["latency_ms"]["total"])
        except Exception as exc:
            telemetry.event("world_brain_shadow_error", error=repr(exc))

    def _write_event(self, event_type: str, payload: dict[str, Any]) -> None:
        if self._session_dir is None:
            return
        event = {
            "schema_version": "jarvis-vision-v1",
            "session_id": self._session_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "event_type": event_type,
            **payload,
        }
        line = json.dumps(event, ensure_ascii=False, default=str)
        with self._log_lock:
            with (self._session_dir / "events.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")


_controller = VisionController()


def active() -> bool:
    return _controller.active


def status() -> str:
    return _controller.status


def preload_async() -> None:
    _controller.preload_async()


def start() -> str:
    return _controller.start()


def stop() -> str:
    return _controller.stop()


def ask(query: str) -> str:
    return _controller.ask(query)


def shutdown() -> None:
    _controller.shutdown()


def session_dir() -> Path | None:
    return _controller.session_dir
