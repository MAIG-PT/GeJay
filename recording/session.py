"""Bounded recording queues, timestamped video, and asynchronous MP4 encoding."""

import math
import queue
import shutil
import subprocess
import tempfile
import threading
import time
import wave
from pathlib import Path

import numpy as np


class RecordingSession:
    """One take. Producers enqueue copies; only the worker writes to disk.

    Video is resampled onto a constant-rate timeline by repeating the previous
    image over missing frames. Audio follows its sample clock after the first
    scheduled playback timestamp, and reanchors if PortAudio reports an underrun.
    """

    def __init__(self, output, frame_size, sample_rate, fps=30.0):
        self.output = Path(output).expanduser().resolve()
        self.frame_size = tuple(frame_size)
        self.sample_rate = sample_rate
        self.fps = fps
        if self.output.suffix.lower() != ".mp4":
            raise ValueError("Recording filename must end in .mp4")
        if not math.isfinite(fps) or fps <= 0 or sample_rate <= 0:
            raise ValueError("Recording FPS and sample rate must be positive")
        if any(dimension <= 0 or dimension % 2 for dimension in self.frame_size):
            raise ValueError("Recording dimensions must be positive even numbers")
        self._audio = queue.Queue(maxsize=512)
        self._video = queue.Queue(maxsize=8)
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._done = threading.Event()
        self._accepting = False
        self.started_at = None
        self.ended_at = None
        self.error = None
        self.work_dir = None
        self.dropped_video = 0
        self._thread = None

    @property
    def active(self):
        return self._accepting

    @property
    def finished(self):
        return self._done.is_set()

    @property
    def elapsed(self):
        if self.started_at is None:
            return 0.0
        end = self.ended_at if self.ended_at is not None else time.monotonic()
        return max(0.0, end - self.started_at)

    def start(self, now=None):
        if self._thread is not None:
            raise RuntimeError("A recording session can only be started once")
        if shutil.which("ffmpeg") is None:
            raise RuntimeError("FFmpeg is required to save camera video with synth audio")
        if self.output.exists():
            raise FileExistsError(f"Recording already exists: {self.output}")
        self.output.parent.mkdir(parents=True, exist_ok=True)
        self.work_dir = Path(tempfile.mkdtemp(prefix=".take-", dir=self.output.parent))
        self._thread = threading.Thread(target=self._run, name="performance-recorder", daemon=True)
        self._thread.start()
        if not self._ready.wait(timeout=5.0):
            self.stop()
            raise RuntimeError("Recording writer did not start within five seconds")
        if self.error:
            raise RuntimeError(self.error)
        self.started_at = time.monotonic() if now is None else now
        self._accepting = True

    def push_audio(self, samples, timestamp, discontinuity=False):
        if not self._accepting:
            return
        # No encoding or disk access on the audio callback.
        try:
            self._audio.put_nowait((timestamp, samples.copy(), discontinuity))
        except queue.Full:
            self.error = "Recording disk writer fell behind; audio queue is full"
            self.stop()

    def push_video(self, image, timestamp):
        if not self._accepting:
            return
        item = (timestamp, image.copy())
        try:
            self._video.put_nowait(item)
        except queue.Full:
            # Bound memory and retain a recent frame; the writer fills the gap.
            try:
                self._video.get_nowait()
            except queue.Empty:
                pass
            self.dropped_video += 1
            try:
                self._video.put_nowait(item)
            except queue.Full:
                self.dropped_video += 1

    def stop(self, now=None):
        if self.ended_at is None:
            self.ended_at = time.monotonic() if now is None else now
        self._accepting = False
        self._stop.set()

    def wait(self, timeout=None):
        return self._done.wait(timeout)

    def _run(self):
        video = None
        audio = None
        try:
            import cv2

            video_path = self.work_dir / "camera.avi"
            audio_path = self.work_dir / "synth.wav"
            video = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"MJPG"),
                                    self.fps, self.frame_size)
            if not video.isOpened():
                raise RuntimeError("Cannot open the MJPEG video writer")
            audio = wave.open(str(audio_path), "wb")
            audio.setnchannels(2)
            audio.setsampwidth(2)
            audio.setframerate(self.sample_rate)
            self._ready.set()
            audio_written = 0
            next_audio = None
            video_written = 0
            last_frame = None
            silence = bytes(4096 * 4)

            def pad_audio(target):
                nonlocal audio_written
                while audio_written < target:
                    count = min(4096, target - audio_written)
                    audio.writeframesraw(silence[:count * 4])
                    audio_written += count

            def write_video_until(target):
                nonlocal video_written
                if last_frame is not None:
                    while video_written < target:
                        video.write(last_frame)
                        video_written += 1

            while not self._stop.is_set() or not self._audio.empty() or not self._video.empty():
                if self.error:
                    raise RuntimeError(self.error)
                worked = False
                # Limit each batch so video is not starved by audio blocks.
                for _ in range(64):
                    try:
                        timestamp, samples, discontinuity = self._audio.get_nowait()
                    except queue.Empty:
                        break
                    worked = True
                    if next_audio is None or discontinuity:
                        next_audio = round((timestamp - self.started_at) * self.sample_rate)
                    first = next_audio
                    next_audio += len(samples)
                    trim = max(0, audio_written - first)
                    if trim >= len(samples):
                        continue
                    pad_audio(max(0, first))
                    samples = samples[trim:]
                    pcm = (np.clip(samples, -1.0, 1.0) * 32767).astype("<i2")
                    audio.writeframesraw(pcm.tobytes())
                    audio_written += len(samples)
                try:
                    timestamp, image = self._video.get_nowait()
                except queue.Empty:
                    pass
                else:
                    worked = True
                    if image.shape[:2] != (self.frame_size[1], self.frame_size[0]):
                        raise RuntimeError("Camera frame dimensions changed during recording")
                    index = max(0, int((timestamp - self.started_at) * self.fps))
                    if last_frame is None:
                        last_frame = image
                    write_video_until(index)
                    last_frame = image
                if not worked:
                    self._stop.wait(0.003)

            if self.error:
                raise RuntimeError(self.error)
            if last_frame is None:
                raise RuntimeError("No camera frames were received during recording")
            duration = max(1.0 / self.fps, self.elapsed)
            write_video_until(math.ceil(duration * self.fps))
            pad_audio(round(duration * self.sample_rate))
            video.release()
            video = None
            audio.close()
            audio = None

            # Encode in the worker; stopping a take never blocks the camera loop.
            with (self.work_dir / "ffmpeg.log").open("w") as log:
                result = subprocess.run([
                    shutil.which("ffmpeg"), "-nostdin", "-hide_banner", "-loglevel", "error", "-n",
                    "-i", str(video_path), "-i", str(audio_path),
                    "-map", "0:v:0", "-map", "1:a:0", "-c:v", "libx264",
                    "-preset", "veryfast", "-crf", "20", "-threads", "2",
                    "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k",
                    "-t", f"{duration:.6f}", "-movflags", "+faststart", str(self.output),
                ], stdout=subprocess.DEVNULL, stderr=log, check=False)
            if result.returncode:
                details = (self.work_dir / "ffmpeg.log").read_text().strip()
                raise RuntimeError(f"FFmpeg could not save the recording: {details}")
            shutil.rmtree(self.work_dir)
        except Exception as exc:
            self.error = str(exc)
        finally:
            self._accepting = False
            try:
                if video is not None:
                    video.release()
                if audio is not None:
                    audio.close()
            except Exception as exc:
                self.error = self.error or str(exc)
            finally:
                self._ready.set()
                self._done.set()
