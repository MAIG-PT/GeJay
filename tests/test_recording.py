"""Verify real encoded A/V duration, delayed audio, sparse video, and UI toggles."""

import contextlib
import io
import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from recording.controls import RecordingControls
from recording.session import RecordingSession
from synth.engine import SynthEngine


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg is required")
class EncodedRecordingTests(unittest.TestCase):
    def test_sparse_video_preserves_duration_and_delayed_audio_stays_aligned(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "performance.mp4"
            session = RecordingSession(output, (64, 48), 48000, fps=10)
            session.start(now=100.0)
            try:
                # Two input images over one second must produce ten output frames.
                session.push_video(np.full((48, 64, 3), (0, 0, 230), np.uint8), 100.0)
                session.push_video(np.full((48, 64, 3), (230, 0, 0), np.uint8), 100.5)
                # Playback begins at 200 ms, then runs on its sample clock.
                tone = 0.2 * np.sin(2 * np.pi * 440 * np.arange(38400) / 48000)
                for offset in range(0, len(tone), 256):
                    mono = tone[offset:offset + 256].astype(np.float32)
                    stereo = np.repeat(mono[:, None], 2, axis=1)
                    session.push_audio(stereo, 100.2 + offset / 48000)
            finally:
                session.stop(now=101.0)
            self.assertTrue(session.wait(timeout=15), "Recorder did not finalize")
            self.assertIsNone(session.error)
            probe = subprocess.run([
                "ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(output),
            ], check=True, capture_output=True, text=True)
            metadata = json.loads(probe.stdout)
            streams = {stream["codec_type"]: stream for stream in metadata["streams"]}
            self.assertEqual(streams["video"]["codec_name"], "h264")
            self.assertEqual(int(streams["video"]["nb_frames"]), 10)
            self.assertEqual(streams["audio"]["codec_name"], "aac")
            self.assertEqual(streams["audio"]["channels"], 2)
            self.assertAlmostEqual(float(metadata["format"]["duration"]), 1.0, delta=0.04)
            decoded = subprocess.run([
                "ffmpeg", "-v", "error", "-i", str(output), "-f", "f32le", "-acodec", "pcm_f32le",
                "-ac", "1", "-ar", "48000", "pipe:1",
            ], check=True, capture_output=True)
            audio = np.frombuffer(decoded.stdout, dtype="<f4")
            self.assertLess(np.max(np.abs(audio[:7000])), 0.002)
            self.assertGreater(np.sqrt(np.mean(audio[14000:30000] ** 2)), 0.1)
            video = subprocess.run([
                "ffmpeg", "-v", "error", "-i", str(output), "-an", "-pix_fmt", "rgb24",
                "-f", "rawvideo", "pipe:1",
            ], check=True, capture_output=True)
            images = np.frombuffer(video.stdout, dtype=np.uint8).reshape(-1, 48, 64, 3)
            self.assertGreater(images[2, :, :, 0].mean(), 180)  # First frame held across the gap.
            self.assertGreater(images[7, :, :, 2].mean(), 180)
            self.assertFalse(session.work_dir.exists(), "Successful take left intermediate files")

    def test_audio_discontinuity_inserts_silence_instead_of_shortening_the_take(self):
        with tempfile.TemporaryDirectory() as directory:
            session = RecordingSession(Path(directory) / "gap.mp4", (64, 48), 48000, fps=10)
            session.start(now=10.0)
            session.push_video(np.zeros((48, 64, 3), np.uint8), 10.0)
            block = np.full((4800, 2), 0.15, np.float32)
            session.push_audio(block, 10.0)
            session.push_audio(block, 10.5, discontinuity=True)
            session.stop(now=10.7)
            self.assertTrue(session.wait(timeout=15))
            self.assertIsNone(session.error)
            decoded = subprocess.run([
                "ffmpeg", "-v", "error", "-i", str(session.output), "-f", "f32le",
                "-ac", "1", "-ar", "48000", "pipe:1",
            ], check=True, capture_output=True)
            audio = np.frombuffer(decoded.stdout, dtype="<f4")
            self.assertLess(np.max(np.abs(audio[12000:20000])), 0.002)
            self.assertGreater(np.mean(audio[25000:27000]), 0.1)


class RecordingControlTests(unittest.TestCase):
    def test_start_failure_does_not_leave_a_session_that_hangs_on_close(self):
        synth = Mock(sample_rate=48000)
        controls = RecordingControls(synth, (640, 480), 30)
        with patch("recording.controls.RecordingSession") as factory:
            factory.return_value.start.side_effect = FileExistsError("already exists")
            with self.assertRaises(FileExistsError):
                controls.start("existing.mp4")
        self.assertIsNone(controls.session)
        controls.close()

    def test_click_and_keyboard_requests_start_and_stop_takes(self):
        synth = Mock(sample_rate=48000, recording_error=None)
        controls = RecordingControls(synth, (640, 480), 30)
        session = Mock(active=True, finished=False, output=Path("take.mp4"), error=None)
        controls._button_rect = (470, 12, 628, 50)
        with patch.dict("sys.modules", {"cv2": SimpleNamespace(EVENT_LBUTTONDOWN=1)}), \
             patch("recording.controls.RecordingSession", return_value=session), \
             contextlib.redirect_stdout(io.StringIO()):
            controls.on_mouse(1, 500, 25, 0, None)
            controls.update()
            session.start.assert_called_once()
            synth.set_audio_sink.assert_called_with(session.push_audio)
            controls.request_toggle()
            controls.update()
            session.stop.assert_called_once()
            synth.set_audio_sink.assert_called_with(None)

    def test_existing_files_are_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "existing.mp4"
            output.write_bytes(b"original")
            with patch("recording.session.shutil.which", return_value="ffmpeg"):
                with self.assertRaises(FileExistsError):
                    RecordingSession(output, (64, 48), 48000).start()
            self.assertEqual(output.read_bytes(), b"original")

    def test_audio_sink_receives_playback_samples_and_errors_do_not_stop_audio(self):
        engine = SynthEngine()
        expected = np.linspace(-0.2, 0.2, 256, dtype=np.float32)
        output = np.zeros((256, 2), np.float32)
        status = SimpleNamespace(output_underflow=False)
        timing = SimpleNamespace(currentTime=50.0, outputBufferDacTime=50.02)
        sink = Mock()
        engine.set_audio_sink(sink)
        with patch.object(engine, "render", return_value=expected), \
             patch("synth.engine.time.monotonic", return_value=100.0):
            engine._callback(output, 256, timing, status)
            self.assertAlmostEqual(sink.call_args.args[1], 100.02)
            np.testing.assert_array_equal(sink.call_args.args[0][:, 0], expected)
            np.testing.assert_array_equal(sink.call_args.args[0][:, 1], expected)
            sink.side_effect = RuntimeError("recorder failed")
            engine._callback(output, 256, timing, status)
            np.testing.assert_array_equal(output[:, 0], expected)
        self.assertEqual(engine.recording_error, "recorder failed")
        self.assertIsNone(engine.callback_error)


if __name__ == "__main__":
    unittest.main()
