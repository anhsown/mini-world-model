from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from threading import Lock

import numpy as np


@dataclass
class FrameObservation:
    index: int
    timestamp: str
    image: np.ndarray


class TemporalFrameMemory:
    def __init__(self, max_frames: int = 8) -> None:
        if max_frames < 1:
            raise ValueError("max_frames must be positive")
        self._frames: deque[FrameObservation] = deque(maxlen=max_frames)
        self._next_index = 0
        self._lock = Lock()

    def add(self, image: np.ndarray, timestamp: str | None = None) -> FrameObservation:
        if image is None or image.ndim != 3 or image.shape[2] not in (3, 4):
            raise ValueError("camera image must be HxWx3/4")
        if image.shape[2] == 4:
            image = image[:, :, :3]
        observation = FrameObservation(
            index=self._next_index,
            timestamp=timestamp or datetime.now(timezone.utc).isoformat(),
            image=np.ascontiguousarray(image[:, :, :3].astype(np.uint8)).copy(),
        )
        with self._lock:
            self._frames.append(observation)
            self._next_index += 1
        return observation

    def latest(self, count: int = 1) -> list[FrameObservation]:
        with self._lock:
            return list(self._frames)[-max(1, count) :]

    def clear(self) -> None:
        with self._lock:
            self._frames.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._frames)

    def status(self) -> str:
        frames = self.latest(len(self) or 1)
        if not frames:
            return "Memory: 0 frame"
        return f"Memory: {len(frames)} frame | latest index={frames[-1].index} @ {frames[-1].timestamp}"
