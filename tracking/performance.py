"""Measure delivered frames separately from camera FPS requested in settings."""

import time


class PerformanceMonitor:
    def __init__(self, now=None, captured=0):
        self._started = time.monotonic() if now is None else now
        self._processed = 0
        self._inference_ms = 0.0
        self._control_ms = 0.0
        self._last_captured = captured
        self._skipped = 0
        self.label = "Measuring FPS..."

    def record(self, captured, skipped, inference_ms, control_ms, now=None):
        now = time.monotonic() if now is None else now
        self._processed += 1
        self._inference_ms += inference_ms
        self._control_ms += control_ms
        self._skipped += skipped
        elapsed = now - self._started
        if elapsed < 1.0:
            return None
        self.label = (
            f"camera {(captured - self._last_captured) / elapsed:.1f} FPS | "
            f"tracking {self._processed / elapsed:.1f} FPS | "
            f"inference {self._inference_ms / self._processed:.1f}ms | "
            f"frame-to-control {self._control_ms / self._processed:.1f}ms | "
            f"skipped {self._skipped}"
        )
        self._started = now
        self._last_captured = captured
        self._processed = 0
        self._inference_ms = 0.0
        self._control_ms = 0.0
        self._skipped = 0
        return self.label
