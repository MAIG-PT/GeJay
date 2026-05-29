"""
Gesture Orchestra
A FoxDot song designed for the controller architecture.

- Strange mode:
    Distance-based generative melody.

- Charm mode:
    Uses left/right gestures to change beat patterns, bass logic,
    fills, accents, and transient sound effects.

- Beauty mode:
    Uses LEFT-HAND GESTURE to control effect profiles
    (delay, reverb, lpf, bpm, texture, octave, resonance),
    while right pinch continuously shapes timing and brightness.
"""

import random
from typing import Optional, List
from FoxDot import *
from songs.song import Song as BaseSong


class Song(BaseSong):
    def __init__(self, max_vec_size: int = 24, max_variation: int = 16):
        self.max_vec_size = max_vec_size
        self.max_variation = max_variation

        self.last_vec_size = None
        self.last_variation_span = None
        self.last_melody = [0, 2, 4, 7]
        self.current_scale = Scale.minor
        self.current_root = 0
        self.current_bpm = 110
        self.last_charm_signature = None
        self.last_beauty_profile = None

    def start(self):
        Clock.bpm = self.current_bpm

        d1 >> play("x-o-", dur=1/2, amp=0.8, room=0.15)
        d2 >> play("  * ", dur=1, amp=0.25, room=0.25)
        b1 >> bass([0, 0, 3, 2], dur=2, oct=4, amp=0.45, scale=self.current_scale)
        p1 >> pluck(self.last_melody, dur=1/4, amp=0.35, room=0.25, mix=0.2, scale=self.current_scale)
        p2 >> pads([0, 2, 4, 7], dur=4, sus=6, amp=0.28, room=0.7, mix=0.45, oct=5, scale=self.current_scale)
        h1 >> bell([(0, 4, 7)], dur=8, sus=6, amp=0.18, room=0.85, mix=0.5, oct=6, scale=self.current_scale)

    def stop(self):
        for player in (d1, d2, p1, p2, b1, h1):
            try:
                player.stop()
            except Exception:
                pass

    def handle_key(self, key):
        if key == "a":
            Clock.bpm = 90
        elif key == "s":
            Clock.bpm = 110
        elif key == "d":
            Clock.bpm = 135
        elif key == "+":
            b1 >> bass([0, 0, 3, 2], dur=2, oct=4, amp=0.55, scale=self.current_scale)
        elif key == "-":
            try:
                b1.stop()
            except Exception:
                pass

    def _clamp(self, x: float, a: float, b: float) -> float:
        return max(a, min(b, x))

    def _safe_stop(self, player):
        try:
            player.stop()
        except Exception:
            pass

    def _set_scale_from_count(self, left_count: int) -> None:
        scales = [
            Scale.minor,
            Scale.major,
            Scale.dorian,
            Scale.mixolydian,
            Scale.lydian,
            Scale.phrygian,
        ]
        self.current_scale = scales[left_count % len(scales)]

    def _reseed_melody(self, length_value: float, variation_value: float) -> List[int]:
        vec_size = max(2, int(length_value * self.max_vec_size))
        variation_span = max(2, int(variation_value * self.max_variation))

        if vec_size == self.last_vec_size and variation_span == self.last_variation_span:
            return self.last_melody

        melody = [random.randint(0, variation_span - 1) for _ in range(vec_size)]
        self.last_vec_size = vec_size
        self.last_variation_span = variation_span
        self.last_melody = melody
        return melody

    def update_from_strange(
        self,
        left_it: float,
        right_it: float,
        index_index: float,
        thumb_thumb: float,
    ) -> None:
        length_value = (index_index + thumb_thumb) / 2
        variation_value = (left_it + right_it) / 2

        melody = self._reseed_melody(length_value, variation_value)

        blur_amt = 1 + int(self._clamp(variation_value, 0.0, 1.0) * 3)
        amp_amt = 0.25 + self._clamp(length_value, 0.0, 1.0) * 0.4

        p1 >> orient(
            melody,
            dur=1/4,
            blur=blur_amt,
            bend=0.08,
            room=0.25,
            mix=0.2,
            amp=amp_amt,
            scale=self.current_scale,
        )

        b1 >> bass(
            [0, 0, 2, 3],
            dur=max(1/2, 2 - self._clamp(index_index, 0.0, 1.0)),
            oct=4,
            amp=0.35 + self._clamp(thumb_thumb, 0.0, 1.0) * 0.2,
            scale=self.current_scale,
        )

    def update_from_charm(
        self,
        left_count: int,
        right_count: int,
        left_gesture: Optional[str] = None,
        right_gesture: Optional[str] = None,
        two_hand_gesture: Optional[str] = None,
    ) -> None:
        self._set_scale_from_count(left_count)

        signature = (left_count, right_count, left_gesture, right_gesture, two_hand_gesture)
        if signature == self.last_charm_signature:
            return
        self.last_charm_signature = signature

        # Base drum density from right-hand finger count
        drum_patterns = {
            0: "-",
            1: "x---",
            2: "x-o-",
            3: "x-o[--]",
            4: "x[--]o[--]",
            5: "x-o-(*[--])",
        }
        main_pattern = drum_patterns.get(min(right_count, 5), "x-o-")
        hat_pattern = ["  - ", " - -", "-- -", "-*-*", "*-*-", "****"][min(right_count, 5)]

        # Gesture-driven overrides / effects
        kick_amp = 0.85
        hat_amp = 0.22
        bass_amp = 0.45
        extra_room = 0.2
        extra_mix = 0.15
        lead_amp = 0.28
        fx_note = None
        fx_synth = None

        gesture_actions = {
            "Heart": lambda: ("x-o(o[--])", " - *", "pads", [0, 2, 4, 7]),
            "Circle": lambda: ("(xo)-", " -- ", "sinepad", [0, 1, 2, 3]),
            "Fist": lambda: ("x---", "    ", "bass", [0]),
            "Point": lambda: ("x--o", "  - ", "pluck", [7, 9]),
            "Victory": lambda: ("x-o-", "- - ", "bell", [0, 4, 7, 11]),
            "Thumbs Up": lambda: ("x[xx]o-", " ** ", "charm", [7, 12]),
            "Thumbs Down": lambda: ("x---", ". . ", "dbass", [-2, 0]),
            "Cool Sign": lambda: ("x-o-", "****", "pads", [7, 9, 11, 14]),
            "Open Hand": lambda: ("x-o[--]", "-*-*", "glass", [0, 2, 4, 6]),
            "Rock Sign": lambda: ("x[--]x-", "**-*", "sawbass", [0, 3, 5]),
            "Three Fingers": lambda: ("x-o-", " - -", "karp", [0, 2, 5]),
            "Four Fingers": lambda: ("x-o(o-)", "--*-", "prophet", [0, 4, 7, 9]),
            "Mini Heart": lambda: ("x-ox", "  * ", "bell", [12, 14]),
            "Looser": lambda: ("x..o", "-  -", "pulse", [0, 5]),
            "Crossed Fingers": lambda: ("x-o-", "*  *", "blip", [1, 6, 8]),
            "Middle Finger": lambda: ("x---", "    ", "noise", [0]),
            "Spock": lambda: ("x-o-", "-*-*", "klank", [0, 2, 7, 9]),
            "Stop": lambda: ("----", "    ", "none", []),
            "Call Me": lambda: ("x-o-", " * *", "bell", [9, 12, 14]),
            "Fancy": lambda: ("x(o[--])-", "-***", "sitar", [0, 3, 7, 10]),
            "Gun": lambda: ("x.xxx", "  * ", "star", [12]),
        }

        chosen_gesture = right_gesture or left_gesture

        if chosen_gesture in gesture_actions:
            main_pattern, hat_pattern, fx_synth, fx_note = gesture_actions[chosen_gesture]()

        if chosen_gesture == "Fist":
            bass_amp = 0.7
            lead_amp = 0.0
            extra_room = 0.05

        elif chosen_gesture == "Point":
            lead_amp = 0.45

        elif chosen_gesture == "Victory":
            Clock.bpm = 122
            extra_room = 0.35

        elif chosen_gesture == "Thumbs Up":
            Clock.bpm = 128
            kick_amp = 1.0

        elif chosen_gesture == "Thumbs Down":
            Clock.bpm = 84
            bass_amp = 0.6

        elif chosen_gesture == "Cool Sign":
            extra_room = 0.9
            extra_mix = 0.7
            lead_amp = 0.2

        elif chosen_gesture == "Open Hand":
            extra_room = 0.55
            extra_mix = 0.35

        elif chosen_gesture == "Rock Sign":
            kick_amp = 1.1
            bass_amp = 0.72

        elif chosen_gesture == "Three Fingers":
            self.current_root = 2

        elif chosen_gesture == "Four Fingers":
            self.current_root = 4

        elif chosen_gesture == "Mini Heart":
            lead_amp = 0.18
            extra_room = 0.8

        elif chosen_gesture == "Looser":
            self.current_root = -1

        elif chosen_gesture == "Crossed Fingers":
            extra_mix = 0.55

        elif chosen_gesture == "Middle Finger":
            lead_amp = 0.05
            bass_amp = 0.2
            kick_amp = 0.5

        elif chosen_gesture == "Spock":
            self.current_scale = Scale.wholeTone

        elif chosen_gesture == "Stop":
            self._safe_stop(d1)
            self._safe_stop(d2)
            self._safe_stop(b1)
            self._safe_stop(p1)
            self._safe_stop(p2)
            return

        elif chosen_gesture == "Call Me":
            extra_mix = 0.5
            extra_room = 0.65

        elif chosen_gesture == "Fancy":
            Clock.bpm = 105
            self.current_scale = Scale.harmonicMinor
            extra_room = 0.75

        elif chosen_gesture == "Gun":
            fx_synth = "noise"
            fx_note = [12]
            kick_amp = 1.2

        if two_hand_gesture == "Heart":
            p2 >> pads(
                [0, 2, 4, 7],
                dur=8,
                sus=7,
                amp=0.42,
                room=1.0,
                mix=0.8,
                oct=6,
                scale=self.current_scale,
            )

        d1 >> play(main_pattern, dur=1/2, amp=kick_amp, room=extra_room)
        d2 >> play(hat_pattern, dur=1/2, amp=hat_amp, room=0.25, mix=0.2)

        b1 >> bass(
            [self.current_root, self.current_root, self.current_root + 2, self.current_root + 4],
            dur=2,
            oct=4,
            amp=bass_amp,
            room=0.2,
            scale=self.current_scale,
        )

        if self.last_melody:
            p1 >> pluck(
                self.last_melody,
                dur=1/4,
                amp=lead_amp,
                room=extra_room,
                mix=extra_mix,
                scale=self.current_scale,
            )

        if fx_synth == "pads":
            p2 >> pads(fx_note, dur=1/4, amp=0.3, room=0.9, mix=0.7, oct=6, scale=self.current_scale)
        elif fx_synth == "sinepad":
            p2 >> sinepad(fx_note, dur=2, sus=4, amp=0.24, room=0.8, mix=0.5, oct=5, scale=self.current_scale)
        elif fx_synth == "pluck":
            h1 >> pluck(fx_note, dur=1/4, amp=0.28, room=0.4, mix=0.25, oct=6, scale=self.current_scale)
        elif fx_synth == "bell":
            h1 >> bell(fx_note, dur=1/2, sus=2, amp=0.3, room=0.8, mix=0.5, oct=6, scale=self.current_scale)
        elif fx_synth == "charm":
            h1 >> charm(fx_note, dur=1/4, amp=0.22, room=0.7, mix=0.35, oct=6, scale=self.current_scale)
        elif fx_synth == "dbass":
            h1 >> dbass(fx_note, dur=1, amp=0.25, room=0.2, mix=0.1, oct=4, scale=self.current_scale)
        elif fx_synth == "glass":
            h1 >> glass(fx_note, dur=1/2, amp=0.2, room=0.85, mix=0.6, oct=6, scale=self.current_scale)
        elif fx_synth == "sawbass":
            h1 >> sawbass(fx_note, dur=1/2, amp=0.25, room=0.15, mix=0.2, oct=4, scale=self.current_scale)
        elif fx_synth == "karp":
            h1 >> karp(fx_note, dur=1/4, amp=0.22, room=0.4, mix=0.2, oct=5, scale=self.current_scale)
        elif fx_synth == "prophet":
            h1 >> prophet(fx_note, dur=2, sus=3, amp=0.24, room=0.8, mix=0.55, oct=5, scale=self.current_scale)
        elif fx_synth == "pulse":
            h1 >> pulse(fx_note, dur=1/4, amp=0.18, room=0.35, mix=0.2, oct=5, scale=self.current_scale)
        elif fx_synth == "blip":
            h1 >> blip(fx_note, dur=1/4, amp=0.22, room=0.5, mix=0.3, oct=6, scale=self.current_scale)
        elif fx_synth == "noise":
            h1 >> noise(dur=1/2, amp=0.16, room=0.5, mix=0.2)
        elif fx_synth == "klank":
            h1 >> klank(fx_note, dur=1/2, amp=0.24, room=0.75, mix=0.45, oct=5, scale=self.current_scale)
        elif fx_synth == "sitar":
            h1 >> sitar(fx_note, dur=1/4, amp=0.22, room=0.55, mix=0.35, oct=5, scale=self.current_scale)
        elif fx_synth == "star":
            h1 >> star(fx_note, dur=1/8, amp=0.25, room=0.45, mix=0.25, oct=6, scale=self.current_scale)
        elif fx_synth == "none":
            self._safe_stop(h1)
            self._safe_stop(p2)

    def update_from_beauty(
        self,
        gesture: int,
        pinch: float,
        left_gesture: Optional[str] = None,
    ) -> None:
        pinch = self._clamp(pinch, 0.0, 1.0)

        # Finger-count voicings
        voicings = [
            [(0, 7)],
            [(0, 4, 7)],
            [(0, 3, 7, 10)],
            [(0, 4, 7, 11)],
            [(0, 2, 4, 7, 9)],
            [(0, 3, 6, 9, 12)],
        ]
        chord = voicings[gesture % len(voicings)]

        # Default beauty profile
        bpm = 105
        room = 0.55
        mix = 0.35
        lpf = 900
        delay = 0.0
        dur = 1 + pinch * 5
        octv = 4 + int(pinch * 2)
        amp = 0.32
        synth_name = "bell"

        # Left-gesture -> effect profile
        beauty_profiles = {
            "Heart": {"bpm": 90, "room": 0.95, "mix": 0.75, "lpf": 700, "delay": 0.5, "synth": "pads"},
            "Circle": {"bpm": 100, "room": 0.7, "mix": 0.45, "lpf": 1200, "delay": 0.25, "synth": "sinepad"},
            "Fist": {"bpm": 72, "room": 0.15, "mix": 0.1, "lpf": 500, "delay": 0.0, "synth": "bass"},
            "Point": {"bpm": 118, "room": 0.3, "mix": 0.15, "lpf": 2200, "delay": 0.125, "synth": "pluck"},
            "Victory": {"bpm": 128, "room": 0.45, "mix": 0.3, "lpf": 1800, "delay": 0.25, "synth": "bell"},
            "Thumbs Up": {"bpm": 132, "room": 0.5, "mix": 0.25, "lpf": 2500, "delay": 0.125, "synth": "prophet"},
            "Thumbs Down": {"bpm": 68, "room": 0.8, "mix": 0.55, "lpf": 450, "delay": 0.5, "synth": "dbass"},
            "Cool Sign": {"bpm": 110, "room": 0.92, "mix": 0.72, "lpf": 1400, "delay": 0.75, "synth": "pads"},
            "Open Hand": {"bpm": 96, "room": 0.88, "mix": 0.6, "lpf": 1000, "delay": 0.375, "synth": "glass"},
            "Rock Sign": {"bpm": 140, "room": 0.35, "mix": 0.18, "lpf": 2800, "delay": 0.125, "synth": "sawbass"},
            "Three Fingers": {"bpm": 104, "room": 0.62, "mix": 0.34, "lpf": 1300, "delay": 0.25, "synth": "karp"},
            "Four Fingers": {"bpm": 108, "room": 0.66, "mix": 0.38, "lpf": 1450, "delay": 0.25, "synth": "bell"},
            "Mini Heart": {"bpm": 92, "room": 0.98, "mix": 0.78, "lpf": 850, "delay": 0.75, "synth": "charm"},
            "Looser": {"bpm": 112, "room": 0.4, "mix": 0.2, "lpf": 1900, "delay": 0.125, "synth": "pulse"},
            "Crossed Fingers": {"bpm": 98, "room": 0.74, "mix": 0.48, "lpf": 1100, "delay": 0.375, "synth": "blip"},
            "Middle Finger": {"bpm": 60, "room": 0.05, "mix": 0.0, "lpf": 300, "delay": 0.0, "synth": "noise"},
            "Spock": {"bpm": 102, "room": 0.82, "mix": 0.58, "lpf": 1600, "delay": 0.5, "synth": "klank"},
            "Stop": {"bpm": 40, "room": 0.0, "mix": 0.0, "lpf": 250, "delay": 0.0, "synth": "none"},
            "Call Me": {"bpm": 116, "room": 0.73, "mix": 0.52, "lpf": 1700, "delay": 0.375, "synth": "bell"},
            "Fancy": {"bpm": 88, "room": 0.93, "mix": 0.68, "lpf": 1250, "delay": 0.625, "synth": "sitar"},
            "Gun": {"bpm": 145, "room": 0.2, "mix": 0.1, "lpf": 3200, "delay": 0.0, "synth": "star"},
        }

        if left_gesture in beauty_profiles:
            profile = beauty_profiles[left_gesture]
            bpm = profile["bpm"]
            room = profile["room"]
            mix = profile["mix"]
            lpf = profile["lpf"]
            delay = profile["delay"]
            synth_name = profile["synth"]

        profile_sig = (gesture, left_gesture, round(pinch, 2))
        if profile_sig == self.last_beauty_profile:
            return
        self.last_beauty_profile = profile_sig

        Clock.bpm = bpm

        if left_gesture == "Stop":
            self._safe_stop(h1)
            self._safe_stop(p2)
            self._safe_stop(p1)
            return

        if gesture == 0:
            self._safe_stop(h1)
        else:
            if synth_name == "pads":
                h1 >> pads(chord, dur=dur, sus=max(1, dur - 0.5), amp=amp, room=room, mix=mix, oct=octv, lpf=lpf, delay=delay, scale=self.current_scale)
            elif synth_name == "sinepad":
                h1 >> sinepad(chord, dur=dur, sus=max(1, dur - 0.5), amp=amp, room=room, mix=mix, oct=octv, lpf=lpf, delay=delay, scale=self.current_scale)
            elif synth_name == "bass":
                h1 >> bass([0], dur=dur, amp=0.25, room=room, mix=mix, oct=3, lpf=lpf, delay=delay, scale=self.current_scale)
            elif synth_name == "pluck":
                h1 >> pluck(chord, dur=max(1/4, dur / 4), amp=amp, room=room, mix=mix, oct=octv + 1, lpf=lpf, delay=delay, scale=self.current_scale)
            elif synth_name == "bell":
                h1 >> bell(chord, dur=dur, sus=max(1, dur - 1), amp=amp, room=room, mix=mix, oct=octv, lpf=lpf, delay=delay, scale=self.current_scale)
            elif synth_name == "prophet":
                h1 >> prophet(chord, dur=dur, sus=max(1, dur - 0.25), amp=amp, room=room, mix=mix, oct=octv, lpf=lpf, delay=delay, scale=self.current_scale)
            elif synth_name == "dbass":
                h1 >> dbass(chord, dur=max(1/2, dur / 2), amp=0.22, room=room, mix=mix, oct=3, lpf=lpf, delay=delay, scale=self.current_scale)
            elif synth_name == "glass":
                h1 >> glass(chord, dur=dur, sus=max(1, dur - 0.5), amp=amp * 0.8, room=room, mix=mix, oct=octv + 1, lpf=lpf, delay=delay, scale=self.current_scale)
            elif synth_name == "sawbass":
                h1 >> sawbass(chord, dur=max(1/2, dur / 2), amp=0.22, room=room, mix=mix, oct=4, lpf=lpf, delay=delay, scale=self.current_scale)
            elif synth_name == "karp":
                h1 >> karp(chord, dur=max(1/4, dur / 3), amp=amp * 0.75, room=room, mix=mix, oct=octv, lpf=lpf, delay=delay, scale=self.current_scale)
            elif synth_name == "charm":
                h1 >> charm(chord, dur=max(1/2, dur / 2), amp=amp * 0.75, room=room, mix=mix, oct=octv + 1, lpf=lpf, delay=delay, scale=self.current_scale)
            elif synth_name == "pulse":
                h1 >> pulse(chord, dur=max(1/4, dur / 3), amp=amp * 0.75, room=room, mix=mix, oct=octv, lpf=lpf, delay=delay, scale=self.current_scale)
            elif synth_name == "blip":
                h1 >> blip(chord, dur=max(1/4, dur / 4), amp=amp * 0.7, room=room, mix=mix, oct=octv + 1, lpf=lpf, delay=delay, scale=self.current_scale)
            elif synth_name == "noise":
                h1 >> noise(dur=max(1/2, dur / 2), amp=0.1, room=room, mix=mix, lpf=lpf, delay=delay)
            elif synth_name == "klank":
                h1 >> klank(chord, dur=max(1/2, dur / 2), amp=amp * 0.8, room=room, mix=mix, oct=octv, lpf=lpf, delay=delay, scale=self.current_scale)
            elif synth_name == "sitar":
                h1 >> sitar(chord, dur=max(1/4, dur / 3), amp=amp * 0.75, room=room, mix=mix, oct=octv, lpf=lpf, delay=delay, scale=self.current_scale)
            elif synth_name == "star":
                h1 >> star(chord, dur=max(1/4, dur / 4), amp=amp * 0.8, room=room, mix=mix, oct=octv + 1, lpf=lpf, delay=delay, scale=self.current_scale)

        variation = max(1, int(pinch * self.max_variation))
        if self.last_vec_size:
            new_melody = [random.randint(0, variation) for _ in range(self.last_vec_size)]
        else:
            new_melody = [random.randint(0, variation) for _ in range(8)]

        p1 >> pluck(
            new_melody,
            dur=max(1/8, 1/2 - pinch * 0.3),
            amp=0.22 + pinch * 0.18,
            room=room * 0.7,
            mix=mix * 0.7,
            oct=5 + int(pinch),
            lpf=lpf + int(pinch * 900),
            delay=delay,
            scale=self.current_scale,
        )

        p2 >> pads(
            [0, 2, 4, 7] if left_gesture != "Cool Sign" else [7, 9, 11, 14],
            dur=max(2, dur * 1.5),
            sus=max(2, dur),
            amp=0.2 + pinch * 0.12,
            room=min(1, room + 0.15),
            mix=min(1, mix + 0.1),
            oct=max(4, octv),
            lpf=max(400, lpf - 150 + int(pinch * 300)),
            delay=delay,
            scale=self.current_scale,
        )