"""
Ambient Drift — an atmospheric, evolving ambient song controlled by hand gestures.

Strange mode: Left/right pinch controls pad filter sweep, inter-hand distances
              control pluck density and reverb depth.
Charm mode:   Left fingers select scale, right fingers control arpeggio speed.
              Cool sign triggers a shimmering high-register texture.
Beauty mode:  Left gesture selects chord voicing, right pinch sweeps
              through duration and octave range for evolving pads.
"""

import random
from typing import List
from FoxDot import *
from songs.song import Song


class Song(Song):
    """
    Ambient Drift: a lush, evolving ambient soundscape controlled by hand gestures.
    Uses pads, plucks, and layered textures for a cinematic, meditative feel.
    """

    def __init__(self, max_vec_size=16, max_variation=12):
        self.max_vec_size = max_vec_size
        self.max_variation = max_variation
        self.drum_pattern = "- - - -"
        self.last_vec_size = None
        self.last_variation_span = None
        self.last_melody = None
        self.current_scale = Scale.minor
        self.last_chord_idx = -1

    def start(self):
        """Start with a minimal ambient foundation — soft kick pulse and airy pad."""
        d1 >> play("- x ", dur=1, amp=0.4, room=0.6, mix=0.5)
        p2 >> pads(
            [0, 2, 4, 7],
            dur=4,
            sus=6,
            amp=0.35,
            room=0.8,
            mix=0.6,
            oct=5,
            scale=Scale.minor,
        )

    def stop(self):
        """Stop all active players."""
        for player in (d1, p1, p2, b1, h1):
            try:
                player.stop()
            except Exception:
                pass

    def handle_key(self, key):
        """
        Keyboard shortcuts for live performance control.

        a/s/d: Switch between minimal, mid, and complex percussion patterns.
        +: Layer in a deep bass drone and bell shimmer.
        -: Strip bass and bell layers back out.
        """
        if key == "a":
            self.drum_pattern = "- x "
            d1 >> play(self.drum_pattern, dur=1, amp=0.4, room=0.6, mix=0.5)

        elif key == "s":
            self.drum_pattern = "-x-[--]"
            d1 >> play(self.drum_pattern, dur=1/2, amp=0.5, room=0.5, mix=0.4)

        elif key == "d":
            self.drum_pattern = "<-x[--]><  o >"
            d1 >> play(self.drum_pattern, dur=1/2, amp=0.55, room=0.5, mix=0.4)

        elif key == "+":
            b1 >> bass(
                [0, 0, -2, 0],
                dur=8,
                oct=3,
                amp=0.5,
                room=0.7,
                mix=0.5,
                sus=8,
            )
            h1 >> bell(
                [(0, 4, 7), (2, 5, 9), (4, 7, 11)],
                dur=6,
                sus=5,
                amp=0.3,
                oct=6,
                room=0.8,
                mix=0.6,
            )

        elif key == "-":
            b1.stop()
            h1.stop()

    def update_from_strange(
        self,
        left_it: float,
        right_it: float,
        index_index: float,
        thumb_thumb: float,
    ) -> None:
        """
        Strange mode: shape the ambient texture with hand distances.

        - Pinch distances (left_it, right_it) control the pad's sustain and amp warmth.
        - Index-to-index distance controls the density of the pluck melody.
        - Thumb-to-thumb distance controls reverb depth and spatial width.

        Args:
            left_it: Left hand index-to-thumb normalized pinch distance [0, 1].
            right_it: Right hand index-to-thumb normalized pinch distance [0, 1].
            index_index: Inter-hand index tip-to-tip normalized distance [0, 1].
            thumb_thumb: Inter-hand thumb tip-to-tip normalized distance [0, 1].
        """
        # Pad warmth from pinch average
        pinch_avg = (left_it + right_it) / 2 if (left_it + right_it) > 0 else 0.3
        pad_amp = max(0.15, min(0.6, pinch_avg * 0.7))
        pad_sus = max(2, int(pinch_avg * 10))

        p2 >> pads(
            [0, 2, 4, 7],
            dur=4,
            sus=pad_sus,
            amp=pad_amp,
            room=0.8,
            mix=0.6,
            oct=5,
            scale=self.current_scale,
        )

        # Pluck melody density from index-index distance
        density = max(1, int(index_index * self.max_vec_size))
        variation = max(1, int(thumb_thumb * self.max_variation))

        if density != self.last_vec_size or variation != self.last_variation_span:
            self.last_melody = [random.randint(0, max(1, variation - 1)) for _ in range(density)]
            self.last_vec_size = density
            self.last_variation_span = variation

        if self.last_melody:
            # Reverb depth from thumb-thumb distance
            room_val = max(0.2, min(0.9, thumb_thumb))

            p1 >> pluck(
                self.last_melody,
                dur=1 / 2,
                amp=0.45,
                room=room_val,
                mix=0.5,
                scale=self.current_scale,
                oct=5,
            )

    def update_from_charm(
        self,
        left_count: int,
        right_count: int,
        left_gesture: bool = False,
        right_gesture: bool = False,
    ) -> None:
        """
        Charm mode: shape the ambient soundscape with finger counting.

        - Left fingers select the scale (6 options cycling through modes).
        - Right fingers control the arpeggio tempo / note duration.
        - Cool sign triggers a shimmering high-register texture.

        Args:
            left_count: Extended finger count of left hand (0-5).
            right_count: Extended finger count of right hand (0-5).
            left_gesture: Gesture name of left hand ('Thumbs Up', 'Open Hand', 'Cool Sign').
            right_gesture: Gesture name of right hand ('Thumbs Up', 'Open Hand', 'Cool Sign').
        """
        scales = [
            Scale.minor,
            Scale.major,
            Scale.dorian,
            Scale.lydian,
            Scale.mixolydian,
            Scale.chromatic,
        ]
        self.current_scale = scales[left_count % len(scales)]

        # Right hand controls arpeggio speed
        durations = [2, 1, 1 / 2, 1 / 4, 1 / 8]
        arp_dur = durations[min(right_count, len(durations) - 1)]

        # Subtle evolving arpeggio
        if self.last_melody:
            p1 >> pluck(
                self.last_melody,
                dur=arp_dur,
                amp=0.4,
                room=0.7,
                mix=0.5,
                scale=self.current_scale,
                oct=5,
            )
        else:
            p1 >> pluck(
                [0, 2, 4, 7, 9],
                dur=arp_dur,
                amp=0.4,
                room=0.7,
                mix=0.5,
                scale=self.current_scale,
                oct=5,
            )

        # Cool sign triggers shimmer texture
        if left_gesture == "Cool Sign" or right_gesture == "Cool Sign":
            p2 >> pads(
                [7, 9, 11, 14],
                dur=1 / 4,
                amp=0.3,
                room=0.9,
                mix=0.7,
                oct=6,
                scale=self.current_scale,
            )
        else:
            p2 >> pads(
                [0, 2, 4, 7],
                dur=4,
                sus=6,
                amp=0.35,
                room=0.8,
                mix=0.6,
                oct=5,
                scale=self.current_scale,
            )

    def update_from_beauty(self, gesture: int, pinch: float) -> None:
        """
        Beauty mode: sculpt chord voicings and evolving pad textures.

        - Left hand gesture (finger count) selects chord voicing.
        - Right hand pinch controls the pad duration sweep and octave shimmer.

        Args:
            gesture: Left hand extended finger count (0-5).
            pinch: Right hand normalized index-to-thumb pinch distance [0, 1].
        """
        # Chord voicings — from sparse to dense and exotic
        voicings = [
            [(0, 7)],                      # Power chord (minimal)
            [(0, 4, 7)],                   # Major triad
            [(0, 3, 7, 10)],              # Minor 7th
            [(0, 4, 7, 11)],              # Major 7th
            [(0, 2, 4, 7, 9)],            # Pentatonic cluster
            [(0, 3, 6, 9, 12)],           # Diminished spread
        ]

        chord = voicings[gesture % len(voicings)]

        # Pinch controls duration and octave range
        pad_dur = max(1, int(pinch * 8))
        pad_oct = 4 + int(pinch * 2)  # Ranges from oct 4 to 6

        if gesture == 0:
            # Silence the chord layer when fist is shown
            try:
                h1.stop()
            except Exception:
                pass
        else:
            h1 >> bell(
                chord,
                dur=pad_dur,
                sus=pad_dur - 1,
                amp=0.4,
                room=0.8,
                mix=0.6,
                oct=pad_oct,
                scale=self.current_scale,
            )

        # Evolving pluck texture based on pinch
        variation = max(1, int(pinch * self.max_variation))
        if self.last_vec_size:
            new_melody = [random.randint(0, variation) for _ in range(self.last_vec_size)]
            p1 >> pluck(
                new_melody,
                dur=1 / 4,
                amp=0.35,
                room=0.6,
                mix=0.4,
                scale=self.current_scale,
            )
