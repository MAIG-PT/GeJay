"""Camera-free checks for gesture mapping, audio output, and tracking loss."""

import unittest
import contextlib
import io
import sys
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from synth.controls import DISTANCE_KEYS, GestureMapper, extract_distances
from synth.engine import SynthEngine, _Delay


def distances(index=0.5, thumb=0.5, left=0.0, right=1.0):
    return dict(zip(DISTANCE_KEYS, (index, thumb, left, right)))


def render(engine, parameters, seconds=0.6, start=0.0, refresh=True):
    blocks = []
    total = round(seconds * engine.sample_rate)
    for offset in range(0, total, engine.block_size):
        now = start + offset / engine.sample_rate
        if refresh:
            engine.update(parameters, now=now)
        blocks.append(engine.render(min(engine.block_size, total - offset), now=now))
    return np.concatenate(blocks)


class MappingTests(unittest.TestCase):
    def test_four_individual_controls(self):
        base = GestureMapper().map(distances())
        for key, attribute in zip(DISTANCE_KEYS, ("frequency", "cutoff", "reverb", "gain")):
            with self.subTest(key=key):
                changed = distances()
                changed[key] = 0.9 if changed[key] != 1.0 else 0.4
                result = GestureMapper().map(changed)
                self.assertNotEqual(getattr(base, attribute), getattr(result, attribute))
                for other in ("frequency", "cutoff", "reverb", "gain"):
                    if other != attribute:
                        self.assertEqual(getattr(base, other), getattr(result, other))

    def test_pitch_endpoints_and_continuous_mode(self):
        mapper = GestureMapper()
        self.assertEqual(mapper.map(distances(index=0)).midi_note, 48)
        self.assertEqual(mapper.map(distances(index=1)).midi_note, 72)
        self.assertAlmostEqual(GestureMapper("continuous").map(distances(index=0.123)).midi_note,
                               48 + 24 * 0.123)

    def test_hysteresis_stops_boundary_chatter(self):
        mapper = GestureMapper()
        mapper.map(distances(index=0))
        for index in (0.049, 0.051, 0.048, 0.052):
            self.assertEqual(mapper.map(distances(index=index)).midi_note, 48)
        self.assertEqual(mapper.map(distances(index=0.064)).midi_note, 50)
        for index in (0.049, 0.051):
            self.assertEqual(mapper.map(distances(index=index)).midi_note, 50)

    def test_missing_invalid_and_closed_pinch(self):
        mapper = GestureMapper()
        self.assertTrue(mapper.map(distances(right=0)).active)
        self.assertEqual(mapper.parameters.gain, 0)
        for invalid in (None, float("nan"), float("inf"), "0.5"):
            frame = distances()
            frame[DISTANCE_KEYS[0]] = invalid
            self.assertFalse(mapper.map(frame).active)
        self.assertFalse(mapper.map({}).active)
        self.assertTrue(mapper.map(distances()).active)

    def test_frame_extraction_preserves_zero_and_absence(self):
        frame = {"index_to_index_normalized": 0.4,
                 "left": {"index_to_thumb_normalized": 0.0}}
        extracted = extract_distances(frame)
        self.assertEqual(extracted[DISTANCE_KEYS[0]], 0.4)
        self.assertEqual(extracted[DISTANCE_KEYS[2]], 0.0)
        self.assertIsNone(extracted[DISTANCE_KEYS[1]])
        self.assertIsNone(extracted[DISTANCE_KEYS[3]])


class AudioTests(unittest.TestCase):
    def test_frequency_and_output_bounds(self):
        parameters = GestureMapper().map(distances(index=0.5, thumb=1))
        audio = render(SynthEngine(), parameters)
        self.assertEqual(audio.dtype, np.float32)
        self.assertTrue(np.isfinite(audio).all())
        self.assertLess(np.max(np.abs(audio)), 1.0)
        spectrum = np.abs(np.fft.rfft(audio[-24000:] * np.hanning(24000)))
        peak_hz = np.argmax(spectrum) * 48000 / 24000
        self.assertAlmostEqual(peak_hz, parameters.frequency, delta=2.0)

    def test_brightness_increases_high_frequency_energy(self):
        high_energy = []
        for thumb in (0.0, 1.0):
            parameters = GestureMapper().map(distances(thumb=thumb))
            audio = render(SynthEngine(), parameters)[-24000:]
            spectrum = np.abs(np.fft.rfft(audio * np.hanning(len(audio)))) ** 2
            frequencies = np.fft.rfftfreq(len(audio), 1 / 48000)
            high_energy.append(spectrum[frequencies > 1500].sum())
        self.assertGreater(high_energy[1], high_energy[0] * 20)

    def test_closed_pinch_is_silent(self):
        audio = render(SynthEngine(), GestureMapper().map(distances(right=0)))
        self.assertEqual(np.count_nonzero(audio), 0)

    def test_missing_frame_and_stalled_camera_fade_out(self):
        for lost_frame in (True, False):
            with self.subTest(lost_frame=lost_frame):
                engine = SynthEngine()
                mapper = GestureMapper()
                parameters = mapper.map(distances(left=1, thumb=1))
                initial = render(engine, parameters)
                self.assertGreater(np.max(np.abs(initial)), 0.01)
                if lost_frame:
                    parameters = mapper.map({})
                faded = render(engine, parameters, seconds=1.0, start=0.6, refresh=lost_frame)
                self.assertLess(np.max(np.abs(faded[-1000:])), 1e-7)

    def test_reverb_delay_impulse_and_arbitrary_block_lengths(self):
        comb = _Delay(5, 0.5)
        impulse = np.zeros(23)
        impulse[0] = 1
        result = comb.process(impulse)
        self.assertEqual(result[5], 1)
        self.assertEqual(result[10], 0.5)
        self.assertEqual(result[15], 0.25)
        for allpass in (False, True):
            whole = _Delay(7, 0.5, allpass).process(impulse)
            split_delay = _Delay(7, 0.5, allpass)
            split = np.concatenate([split_delay.process(impulse[:3]),
                                    split_delay.process(impulse[3:18]),
                                    split_delay.process(impulse[18:])])
            np.testing.assert_array_equal(whole, split)

    def test_live_reverb_changes_audio_and_remains_bounded(self):
        dry = render(SynthEngine(), GestureMapper().map(distances(left=0)))
        wet = render(SynthEngine(), GestureMapper().map(distances(left=1)))
        self.assertGreater(np.sqrt(np.mean((dry - wet) ** 2)), 0.005)
        self.assertLess(np.max(np.abs(wet)), 1.0)


class TrackerIntegrationTests(unittest.TestCase):
    def test_camera_frames_update_synth_and_tcp_then_release_resources(self):
        from scripts import hand_distance_tracker as tracker

        frame = np.zeros((20, 20, 3), dtype=np.uint8)
        camera = Mock()
        camera.get.side_effect = lambda key: {3: 640, 4: 480, 5: 30, 6: 1196444237}[key]
        camera.read.side_effect = [(True, frame), (True, frame), (False, None)]
        hands = Mock()
        hands.detect_hands.side_effect = [
            ([{"handedness": "Left"}, {"handedness": "Right"}], None), (None, None),
        ]
        calculator = Mock()
        calculator.get_hand_distances.side_effect = [
            {"index_to_thumb_normalized": 0.2}, {"index_to_thumb_normalized": 0.8},
        ]
        calculator.get_index_to_index_distance.return_value = {DISTANCE_KEYS[0]: 0.4}
        calculator.get_thumb_to_thumb_distance.return_value = {DISTANCE_KEYS[1]: 0.6}
        fake_cv = SimpleNamespace(VideoCapture=Mock(return_value=camera),
                                  flip=lambda image, axis: image,
                                  CAP_PROP_FRAME_WIDTH=3, CAP_PROP_FRAME_HEIGHT=4,
                                  CAP_PROP_FPS=5, CAP_PROP_FOURCC=6, CAP_PROP_BUFFERSIZE=38,
                                  VideoWriter_fourcc=lambda *letters: 1196444237)
        fake_utils = SimpleNamespace(HandTracker=Mock(return_value=hands),
                                     DistanceCalculator=Mock(return_value=calculator),
                                     Visualizer=Mock())
        synth = Mock(sample_rate=48000, underruns=0)
        connection = Mock()
        with patch.dict(sys.modules, {"cv2": fake_cv, "utils": fake_utils}), \
             patch("synth.engine.SynthEngine", return_value=synth), \
             patch.object(tracker, "send_tcp_payload", return_value=connection) as send, \
             contextlib.redirect_stdout(io.StringIO()):
            tracker.main(show_visualization=False, enable_synth=True,
                         stream_tcp=True, verbose=False, threaded_camera=False)

        first, lost = [call.args[0] for call in synth.update.call_args_list]
        self.assertTrue(first.active)
        self.assertFalse(lost.active)
        first_payload = send.call_args_list[0].args[1]
        self.assertEqual([first_payload[key] for key in DISTANCE_KEYS], [0.4, 0.6, 0.2, 0.8])
        self.assertEqual(first_payload["interhand_average"], 0.5)
        self.assertEqual(first_payload["thumb_index_average"], 0.5)
        lost_payload = send.call_args_list[1].args[1]
        self.assertTrue(all(lost_payload[key] is None for key in DISTANCE_KEYS))
        synth.start.assert_called_once_with(device=None)
        synth.close.assert_called_once()
        camera.release.assert_called_once()
        hands.close.assert_called_once()
        connection.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
