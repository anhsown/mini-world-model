"""Process-isolated Qwen visual reasoner with crash detection and restart support."""

from __future__ import annotations

from dataclasses import asdict
import multiprocessing as mp
from queue import Empty
import threading
import time
import traceback
from typing import Sequence
from uuid import uuid4

from .config import ReasonerConfig
from .memory import FrameObservation
from .reasoner import Qwen3VLReasoner, VisionLanguageReasoner
from .schemas import ReasonerResult


def _worker_main(request_queue, response_queue, config_payload: dict) -> None:
    """Own all CUDA/model state so a native failure cannot kill JARVIS."""
    try:
        import torch

        # torch 2.5.1/cuDNN 9.1 on this GTX 1650 crashes inside the Qwen visual
        # encoder while resolving cudnnGetLibConfig. CUDA kernels remain active;
        # only cuDNN graph dispatch is disabled.
        torch.backends.cudnn.enabled = False
        reasoner = Qwen3VLReasoner(ReasonerConfig(**config_payload))
        reasoner.load()
        response_queue.put({"type": "ready"})
        while True:
            request = request_queue.get()
            if request.get("type") == "shutdown":
                return
            if request.get("type") != "analyze":
                continue
            request_id = request["request_id"]
            try:
                result = reasoner.analyze(request["frames"], request["query"])
                response_queue.put({"type": "result", "request_id": request_id, "result": result})
            except BaseException as exc:
                response_queue.put(
                    {
                        "type": "error",
                        "request_id": request_id,
                        "error": repr(exc),
                        "traceback": traceback.format_exc(),
                    }
                )
    except BaseException as exc:
        try:
            response_queue.put(
                {"type": "fatal", "error": repr(exc), "traceback": traceback.format_exc()}
            )
        except Exception:
            pass


class IsolatedQwen3VLReasoner(VisionLanguageReasoner):
    """Supervise a disposable model worker and surface native crashes as errors."""

    def __init__(
        self,
        config: ReasonerConfig,
        *,
        startup_timeout: float = 45.0,
        response_timeout: float = 120.0,
        worker_target=None,
    ) -> None:
        self.config = config
        self.startup_timeout = startup_timeout
        self.response_timeout = response_timeout
        self._worker_target = worker_target or _worker_main
        self._context = mp.get_context("spawn")
        self._process = None
        self._request_queue = None
        self._response_queue = None
        self._ready = False
        self._lifecycle_lock = threading.Lock()

    @property
    def loaded(self) -> bool:
        return bool(self._ready and self._process is not None and self._process.is_alive())

    @property
    def worker_pid(self) -> int | None:
        return self._process.pid if self._process is not None and self._process.is_alive() else None

    def load(self) -> None:
        with self._lifecycle_lock:
            if self.loaded:
                return
            self._dispose_worker(force=True)
            self._request_queue = self._context.Queue(maxsize=2)
            self._response_queue = self._context.Queue(maxsize=2)
            self._process = self._context.Process(
                target=self._worker_target,
                args=(self._request_queue, self._response_queue, asdict(self.config)),
                name="jarvis-visual-reasoner",
                daemon=True,
            )
            self._process.start()
            deadline = time.monotonic() + self.startup_timeout
            while time.monotonic() < deadline:
                if not self._process.is_alive():
                    exit_code = self._process.exitcode
                    self._ready = False
                    raise RuntimeError(f"Visual worker crashed during startup (exit code {exit_code}).")
                try:
                    message = self._response_queue.get(timeout=0.25)
                except Empty:
                    continue
                if message.get("type") == "ready":
                    self._ready = True
                    return
                if message.get("type") == "fatal":
                    raise RuntimeError(f"Visual worker failed to load: {message.get('error')}")
            self._dispose_worker(force=True)
            raise TimeoutError(f"Visual worker did not become ready within {self.startup_timeout:.0f} seconds.")

    def analyze(self, frames: Sequence[FrameObservation], query: str) -> ReasonerResult:
        self.load()
        request_id = uuid4().hex
        self._request_queue.put(
            {"type": "analyze", "request_id": request_id, "frames": list(frames), "query": query},
            timeout=2.0,
        )
        deadline = time.monotonic() + self.response_timeout
        while time.monotonic() < deadline:
            if self._process is None or not self._process.is_alive():
                exit_code = self._process.exitcode if self._process is not None else None
                self._ready = False
                raise RuntimeError(f"Visual worker crashed during inference (exit code {exit_code}).")
            try:
                message = self._response_queue.get(timeout=0.25)
            except Empty:
                continue
            message_type = message.get("type")
            if message_type == "fatal":
                self._ready = False
                raise RuntimeError(f"Visual worker fatal error: {message.get('error')}")
            if message.get("request_id") != request_id:
                continue
            if message_type == "result":
                return message["result"]
            if message_type == "error":
                raise RuntimeError(f"Visual inference failed: {message.get('error')}")
        raise TimeoutError(f"Visual worker did not answer within {self.response_timeout:.0f} seconds.")

    def close(self) -> None:
        with self._lifecycle_lock:
            self._dispose_worker(force=False)

    def _dispose_worker(self, *, force: bool) -> None:
        process = self._process
        if process is not None and process.is_alive() and not force:
            try:
                self._request_queue.put({"type": "shutdown"}, timeout=0.5)
                process.join(timeout=2.0)
            except Exception:
                pass
        if process is not None and process.is_alive():
            process.terminate()
            process.join(timeout=2.0)
        for queue in (self._request_queue, self._response_queue):
            if queue is not None:
                try:
                    queue.close()
                except Exception:
                    pass
        self._process = None
        self._request_queue = None
        self._response_queue = None
        self._ready = False
