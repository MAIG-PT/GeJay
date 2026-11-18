"""Preview recording button and take lifecycle; no disk I/O in mouse callbacks."""

from datetime import datetime
from pathlib import Path

from recording.session import RecordingSession


class RecordingControls:
    def __init__(self, synth, frame_size, fps, directory="recordings"):
        self.synth = synth
        self.frame_size = frame_size
        self.fps = fps
        self.directory = Path(directory)
        self.session = None
        self._toggle_requested = False
        self._button_rect = None
        self._reported = False
        self._last_status = "r: record"

    @property
    def active(self):
        return self.session is not None and self.session.active

    @property
    def saving(self):
        return self.session is not None and not self.active and not self.session.finished

    def request_toggle(self):
        self._toggle_requested = True

    def on_mouse(self, event, x, y, flags, parameter):
        import cv2
        if event == cv2.EVENT_LBUTTONDOWN and self._button_rect:
            left, top, right, bottom = self._button_rect
            if left <= x <= right and top <= y <= bottom:
                self.request_toggle()

    def start(self, output=None):
        if self.active or self.saving:
            return
        if output is None:
            name = datetime.now().strftime("performance_%Y%m%d_%H%M%S_%f.mp4")
            output = self.directory / name
        session = RecordingSession(output, self.frame_size, self.synth.sample_rate, self.fps)
        session.start()
        self.session = session
        self._reported = False
        self.synth.set_audio_sink(self.session.push_audio)
        print(f"Recording: {self.session.output}")

    def update(self):
        if self.active and self.synth.recording_error:
            self.session.error = self.synth.recording_error
            self.synth.recording_error = None
            self.stop()
        if self._toggle_requested:
            self._toggle_requested = False
            if self.active:
                self.stop()
            elif not self.saving:
                try:
                    self.start()
                except (OSError, RuntimeError, ValueError) as exc:
                    self._last_status = "Recording unavailable"
                    print(f"Recording error: {exc}")
        if self.session is not None and self.session.finished and not self._reported:
            self._reported = True
            self.synth.set_audio_sink(None)
            if self.session.error:
                self._last_status = "Recording failed; see terminal"
                print(f"Recording error: {self.session.error}")
                if self.session.work_dir:
                    print(f"Recording intermediate files: {self.session.work_dir}")
            else:
                self._last_status = "Saved | r: new take"
                print(f"Saved recording: {self.session.output} ({self.session.elapsed:.1f}s)")

    def stop(self):
        self.synth.set_audio_sink(None)
        if self.session is not None and self.session.active:
            self.session.stop()
            print("Saving recording...")

    def draw(self, frame):
        import cv2
        right = frame.shape[1] - 12
        left, top, bottom = max(0, right - 158), 12, 50
        self._button_rect = (left, top, right, bottom)
        label = "Stop recording" if self.active else "Saving..." if self.saving else "Record"
        color = (35, 35, 190) if self.active else (65, 65, 65)
        cv2.rectangle(frame, (left, top), (right, bottom), color, -1)
        cv2.putText(frame, label, (left + 10, top + 25), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 1, cv2.LINE_AA)
        status = self._last_status
        if self.active:
            seconds = int(self.session.elapsed)
            status = f"REC {seconds // 60:02d}:{seconds % 60:02d}"
        elif self.saving:
            status = "Finishing MP4..."
        cv2.putText(frame, status, (left, bottom + 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, (0, 0, 255) if self.active else (220, 220, 220), 1, cv2.LINE_AA)

    def close(self):
        self._toggle_requested = False
        self.stop()
        if self.session is not None and not self.session.finished:
            # Finish the MP4 before the process exits; the live stream is already closed.
            self.session.wait()
        self.update()
