from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class TimestampedFrame:
    timestamp_ns: int
    frame: np.ndarray
    metadata: dict[str, Any]


class ImageRingBuffer:
    def __init__(self, capacity: int = 256) -> None:
        self._frames: deque[TimestampedFrame] = deque(maxlen=capacity)
        self._lock = threading.Lock()

    def push(self, frame: np.ndarray, timestamp_ns: int, metadata: dict[str, Any] | None = None) -> None:
        item = TimestampedFrame(timestamp_ns=timestamp_ns, frame=np.asarray(frame), metadata=metadata or {})
        with self._lock:
            self._frames.append(item)

    def latest(self) -> TimestampedFrame | None:
        with self._lock:
            if not self._frames:
                return None
            return self._frames[-1]

    def nearest(self, timestamp_ns: int, max_delta_ns: int | None = None) -> TimestampedFrame | None:
        with self._lock:
            if not self._frames:
                return None
            best = min(self._frames, key=lambda f: abs(f.timestamp_ns - timestamp_ns))
        if max_delta_ns is not None and abs(best.timestamp_ns - timestamp_ns) > max_delta_ns:
            return None
        return best

    def __len__(self) -> int:
        with self._lock:
            return len(self._frames)
