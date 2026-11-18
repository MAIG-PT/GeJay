"""Continuously drain the camera and retain only its most recent frame."""

import threading
import time
from dataclasses import dataclass
from typing import Any, Optional


@dataclass(frozen=True)
class CameraFrame:
    image: Any
    captured_at: float  # Host time when read() returned, not sensor exposure time.
    sequence: int


class LatestFrameCapture:
    """Own the capture device and drop frames when inference falls behind.

    Only the worker calls the underlying read/release methods. The consumer
    waits for a new sequence, so an old image is never processed twice.
    """

    def __init__(self, capture):
        self._capture = capture
        self._condition = threading.Condition()
        self._stop = threading.Event()
        self._latest = None
        self._ended = False
        self._error = None
        self._sequence = 0
        self._thread = threading.Thread(target=self._run, name="camera-capture", daemon=True)

    @property
    def sequence(self):
        with self._condition:
            return self._sequence

    def start(self):
        self._thread.start()

    def _run(self):
        try:
            while not self._stop.is_set():
                ok, image = self._capture.read()
                captured_at = time.monotonic()
                if not ok:
                    break
                with self._condition:
                    self._sequence += 1
                    self._latest = CameraFrame(image, captured_at, self._sequence)
                    self._condition.notify_all()
        except Exception as exc:
            self._error = exc
        finally:
            try:
                self._capture.release()
            finally:
                with self._condition:
                    self._ended = True
                    self._condition.notify_all()

    def read(self, after_sequence=0, timeout=2.0) -> Optional[CameraFrame]:
        deadline = time.monotonic() + timeout
        with self._condition:
            while True:
                if self._latest is not None and self._latest.sequence > after_sequence:
                    return self._latest
                if self._error is not None:
                    raise RuntimeError(f"Camera capture failed: {self._error}") from self._error
                if self._ended or self._stop.is_set():
                    return None
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(f"Camera did not deliver a new frame for {timeout:g} seconds")
                self._condition.wait(remaining)

    def close(self):
        self._stop.set()
        with self._condition:
            self._condition.notify_all()
        if self._thread.ident is not None:
            # Never release a device concurrently with its native read() call.
            # The daemon worker releases it in finally when that read returns.
            self._thread.join(timeout=2.0)
        else:
            self._capture.release()
