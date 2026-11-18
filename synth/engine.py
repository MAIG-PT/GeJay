"""A NumPy synthesizer with spectral low-pass filtering and Schroeder reverb.

The camera thread only publishes immutable control snapshots. PortAudio renders
audio independently; it never waits for tracking, console output, or TCP.
"""

import time
from dataclasses import replace

import numpy as np

from config.synth_config import SYNTH_CONFIG
from synth.controls import SynthParameters


class _Delay:
    def __init__(self, length: int, feedback: float, allpass: bool = False):
        self.buffer = np.zeros(length, dtype=np.float64)
        self.position = 0
        self.feedback = feedback
        self.allpass = allpass

    def process(self, source: np.ndarray) -> np.ndarray:
        result = np.empty_like(source)
        start = 0
        # Chunk at the ring boundary, also supporting blocks longer than a delay.
        while start < len(source):
            count = min(len(source) - start, len(self.buffer) - self.position)
            section = slice(self.position, self.position + count)
            delayed = self.buffer[section].copy()
            incoming = source[start:start + count]
            if self.allpass:
                output = delayed - self.feedback * incoming
                self.buffer[section] = incoming + self.feedback * output
            else:
                output = delayed
                self.buffer[section] = incoming + self.feedback * delayed
            result[start:start + count] = output
            self.position = (self.position + count) % len(self.buffer)
            start += count
        return result


class SynthEngine:
    """Monophonic instrument. update() is called once per camera frame."""

    def __init__(self, sample_rate=None, block_size=None, master_gain=None):
        self.sample_rate = sample_rate or SYNTH_CONFIG["sample_rate"]
        self.block_size = block_size or SYNTH_CONFIG["block_size"]
        self.master_gain = (SYNTH_CONFIG["master_gain"]
                            if master_gain is None else master_gain)
        if self.sample_rate < 16000 or self.block_size < 32:
            raise ValueError("Use a sample rate >= 16000 and a block size >= 32")
        if not 0 <= self.master_gain <= 1:
            raise ValueError("Master gain must be between 0 and 1")
        self._snapshot = (SynthParameters(), float("-inf"))
        # Smooth log frequency/cutoff, wet mix, volume, and tracking gate.
        self._state = np.array([np.log(130.8128), np.log(200.0), 0., 0., 0.])
        self._phase = 0.0
        self._harmonics = np.arange(1, 25, dtype=np.float64)
        self._combs = [_Delay(round(self.sample_rate * delay), 0.78)
                       for delay in (0.0297, 0.0371, 0.0411, 0.0437)]
        self._allpasses = [_Delay(round(self.sample_rate * delay), 0.5, True)
                          for delay in (0.005, 0.0017)]
        self._stream = None
        self.callback_status = ""
        self.callback_error = None
        self.underruns = 0
        self._audio_sink = None
        self.recording_error = None

    def set_audio_sink(self, callback):
        """Attach a nonblocking consumer of the actual stereo playback buffers."""
        if callback is not None:
            self.recording_error = None
        self._audio_sink = callback

    def update(self, parameters: SynthParameters, now=None):
        self._snapshot = (parameters, time.monotonic() if now is None else now)

    def render(self, frames: int, now=None) -> np.ndarray:
        """Render mono float32 PCM; also usable offline without audio hardware."""
        if frames <= 0:
            return np.empty(0, dtype=np.float32)
        parameters, updated = self._snapshot
        now = time.monotonic() if now is None else now
        active = parameters.active and now - updated <= SYNTH_CONFIG["tracking_timeout"]
        target = np.array([
            np.log(parameters.frequency), np.log(parameters.cutoff),
            parameters.reverb, parameters.gain if active else 0.0, float(active),
        ])
        decay = np.exp(-np.arange(1, frames + 1) /
                       (self.sample_rate * SYNTH_CONFIG["smoothing_seconds"]))
        smoothed = target + decay[:, None] * (self._state - target)
        self._state = smoothed[-1].copy()
        frequency = np.exp(smoothed[:, 0])
        cutoff = np.minimum(np.exp(smoothed[:, 1]), self.sample_rate * 0.45)

        # Additive saw-like voice. Suppress partials at Nyquist to avoid aliasing;
        # apply a 12 dB/oct spectral low-pass envelope to its harmonics.
        phase = self._phase + 2.0 * np.pi * np.cumsum(frequency) / self.sample_rate
        self._phase = phase[-1] % (2.0 * np.pi)
        partial_frequency = frequency[:, None] * self._harmonics
        weights = 1.0 / np.sqrt(1.0 + (partial_frequency / cutoff[:, None]) ** 4)
        weights *= partial_frequency < self.sample_rate * 0.45
        oscillator = (np.sin(phase[:, None] * self._harmonics)
                      * weights / self._harmonics).sum(axis=1) * (2.0 / np.pi)
        dry = oscillator * smoothed[:, 3]
        wet = sum(delay.process(dry) for delay in self._combs) / len(self._combs)
        for delay in self._allpasses:
            wet = delay.process(wet)
        mix = smoothed[:, 2]
        # Tracking gate also fades the reverb tail when hands disappear/stall.
        output = ((1.0 - mix) * dry + mix * wet) * smoothed[:, 4] * self.master_gain
        return np.tanh(output).astype(np.float32)

    def _callback(self, outdata, frames, timing, status):
        if status:
            self.callback_status = str(status)
            if status.output_underflow:
                self.underruns += 1
        try:
            outdata[:] = self.render(frames)[:, None]
        except Exception as exc:
            # PortAudio callback exceptions do not propagate to the camera thread.
            outdata.fill(0)
            self.callback_error = str(exc)
            import sounddevice as sd
            raise sd.CallbackAbort from exc
        sink = self._audio_sink
        if sink is not None:
            try:
                # Map PortAudio's scheduled DAC time onto our monotonic clock.
                timestamp = time.monotonic() + float(timing.outputBufferDacTime - timing.currentTime)
                sink(outdata, timestamp, bool(status.output_underflow))
            except Exception as exc:
                # A failed recorder must never interrupt the musical performance.
                self.recording_error = str(exc)
                self._audio_sink = None

    def start(self, device=None):
        import sounddevice as sd

        self._stream = sd.OutputStream(
            samplerate=self.sample_rate, blocksize=self.block_size,
            channels=2, dtype="float32", latency="low", device=device,
            callback=self._callback,
        )
        try:
            self._stream.start()
        except Exception:
            self._stream.close()
            self._stream = None
            raise

    def check_audio(self):
        if self.callback_error:
            raise RuntimeError(f"Audio callback failed: {self.callback_error}")
        if self._stream is not None and not self._stream.active:
            raise RuntimeError("Audio stream stopped unexpectedly")

    def close(self):
        if self._stream is not None:
            parameters, _ = self._snapshot
            self.update(replace(parameters, gain=0.0, active=False))
            try:
                time.sleep(0.15)  # Let the callback fade before closing the device.
                self._stream.stop()
            finally:
                self._stream.close()
                self._stream = None
