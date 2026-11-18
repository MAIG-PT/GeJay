"""Check camera backlog handling without requiring a webcam or MediaPipe."""

import queue
import threading
import unittest

from tracking.camera import LatestFrameCapture
from tracking.performance import PerformanceMonitor


class ControlledCapture:
    def __init__(self):
        self.frames = queue.Queue()
        self.released = threading.Event()
        self.release_count = 0

    def read(self):
        item = self.frames.get(timeout=1.0)
        if isinstance(item, Exception):
            raise item
        return item is not None, item

    def release(self):
        self.release_count += 1
        self.released.set()


class LatestFrameTests(unittest.TestCase):
    def test_slow_consumer_gets_newest_frame_and_not_a_queue(self):
        capture = ControlledCapture()
        reader = LatestFrameCapture(capture)
        reader.start()
        try:
            capture.frames.put("first")
            first = reader.read()
            self.assertEqual((first.sequence, first.image), (1, "first"))
            capture.frames.put("second")
            capture.frames.put("third")
            capture.frames.put(None)
            self.assertTrue(capture.released.wait(timeout=1.0))
            latest = reader.read(after_sequence=first.sequence)
            self.assertEqual((latest.sequence, latest.image), (3, "third"))
            self.assertGreaterEqual(latest.captured_at, first.captured_at)
            self.assertIsNone(reader.read(after_sequence=latest.sequence))
        finally:
            capture.frames.put(None)
            reader.close()
        self.assertEqual(capture.release_count, 1)

    def test_same_frame_is_not_returned_twice(self):
        capture = ControlledCapture()
        reader = LatestFrameCapture(capture)
        reader.start()
        try:
            capture.frames.put("first")
            first = reader.read()
            with self.assertRaises(TimeoutError):
                reader.read(after_sequence=first.sequence, timeout=0.01)
            capture.frames.put("second")
            second = reader.read(after_sequence=first.sequence)
            self.assertEqual(second.image, "second")
            self.assertEqual(second.sequence, 2)
        finally:
            capture.frames.put(None)
            reader.close()
        self.assertEqual(capture.release_count, 1)

    def test_disconnect_is_reported_and_device_is_released(self):
        capture = ControlledCapture()
        capture.frames.put(RuntimeError("disconnected"))
        reader = LatestFrameCapture(capture)
        reader.start()
        try:
            with self.assertRaisesRegex(RuntimeError, "disconnected"):
                reader.read()
        finally:
            reader.close()
        self.assertEqual(capture.release_count, 1)


class PerformanceTests(unittest.TestCase):
    def test_capture_rate_is_distinct_from_processed_rate(self):
        monitor = PerformanceMonitor(now=0.0)
        self.assertIsNone(monitor.record(1, 0, 10.0, 12.0, now=0.4))
        label = monitor.record(3, 1, 20.0, 22.0, now=1.0)
        self.assertIn("camera 3.0 FPS", label)
        self.assertIn("tracking 2.0 FPS", label)
        self.assertIn("inference 15.0ms", label)
        self.assertIn("frame-to-control 17.0ms", label)
        self.assertIn("skipped 1", label)
        next_label = monitor.record(8, 4, 10.0, 12.0, now=2.0)
        self.assertIn("camera 5.0 FPS", next_label)
        self.assertIn("tracking 1.0 FPS", next_label)
        self.assertIn("skipped 4", next_label)


if __name__ == "__main__":
    unittest.main()
