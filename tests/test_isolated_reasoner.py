from __future__ import annotations

import os

import numpy as np
import pytest

from core.vision_runtime import FrameObservation, IsolatedQwen3VLReasoner, ReasonerConfig


def _native_crash_fixture(request_queue, response_queue, _config_payload) -> None:
    response_queue.put({"type": "ready"})
    request_queue.get()
    os._exit(73)


def test_native_worker_crash_is_contained_in_parent_process() -> None:
    reasoner = IsolatedQwen3VLReasoner(
        ReasonerConfig(model_id="fault-injection"),
        startup_timeout=5,
        response_timeout=5,
        worker_target=_native_crash_fixture,
    )
    reasoner.load()
    frame = FrameObservation(index=0, timestamp="fixture", image=np.zeros((8, 8, 3), dtype=np.uint8))
    with pytest.raises(RuntimeError, match=r"crashed during inference.*73"):
        reasoner.analyze([frame], "fixture")
    assert not reasoner.loaded
    reasoner.close()

