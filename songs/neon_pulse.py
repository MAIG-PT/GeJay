"""
Neon Pulse — a glitchy, rhythmic electronic song controlled by hand gestures.

A funky, percussive soundscape built on karp arpeggios, spark leads,
and choppy rhythmic patterns. Designed for energetic, playful performance.

Strange mode: Pinch distances control arpeggio pitch range and chop speed.
              Inter-hand distances control filter-like amp modulation and swing.
Charm mode:   Left fingers select root note, right fingers control rhythmic density.
              Cool sign triggers a wild glitch burst.
Beauty mode:  Left gesture picks a synth voice, right pinch sweeps tempo and octave.
"""

import random
from typing import List
from FoxDot import *
from songs.song import Song


class Song(Song):
    """
    Neon Pulse: a glitchy, funky electronic groove controlled by hand gestures.
    Uses karp, spark, and choppy percussion for an energetic, playful feel.
    """

    def __init__(self, max_vec_size=12, max_variation=14):
        self.max_vec_size = max_vec_size
        self.max_variation = max_variation
        self.drum_pattern = "x-o-x-o[--]"
        self.last_vec_size = None
        self.last_variation_span = None
        self.last_melody = None
        self.current_scale = Scale.minor
        self.current_root = 0

    def start(self):
        """Start with a punchy beat and a bouncy karp arpeggio."""
        d1 >> play(self.drum_pattern, dur=1/2, amp=0.8, room=0.3, mix=0.3)
        p2 >> karp(
            [0, 3, 5, 7],
            dur=[1/4, 1/4, 1/2, 1/4],
            amp=0.5,
            room=0.4,
            mix=0.3,
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
        Keyboard shortcuts for live control.

        a: Minimal groove.
        s: Syncopated beat.
        d: Full glitch percussion.
        +: Layer in a wobbly bass and bright spark lead.
        -: Strip layers back.
        """
        if key == "a":
            self.drum_pattern = "x-o-"
            d1 >> play(self.drum_pattern, dur=1/2, amp=0.8, room=0.3, mix=0.3)

        elif key == "s":
            self.drum_pattern = "x-o-x-o[--]"
            d1 >> play(self.drum_pattern, dur=1/2, amp=0.8, room=0.3, mix=0.3)

        elif key == "d":
            self.drum_pattern = "<x[--]o[--]><  X >"
            d1 >> play(self.drum_pattern, dur=1/4, amp=0.85, room=0.3, mix=0.3)

        elif key == "+":
            b1 >> bass(
                [0, 0, 3, 5],
                dur=1,
                oct=3,
                amp=0.65,
                room=0.3,
                mix=0.3,
                sus=0.8,
            )
            h1 >> spark(
                [(0, 4), (3, 7), (5, 9)],
                dur=2,
                sus=1.5,
                amp=0.35,
                oct=6,
                room=0.5,
                mix=0.4,
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
        Strange mode: shape the electronic groove with hand distances.

        - Pinch distances control arpeggio note count and pitch variation.
        - Index-index distance controls rhythmic chop speed.
        - Thumb-thumb distance controls swing feel and amp modulation.

        Args:
            left_it: Left hand index-to-thumb normalized pinch distance [0, 1].
            right_it: Right hand index-to-thumb normalized pinch distance [0, 1].
            index_index: Inter-hand index tip-to-tip normalized distance [0, 1].
            thumb_thumb: Inter-hand thumb tip-to-tip normalized distance [0, 1].
        """
        # Arpeggio density from pinch average
        pinch_avg = (left_it + right_it) / 2 if (left_it + right_it) > 0 else 0.3
        note_count = max(2, int(pinch_avg * self.max_vec_size))
        variation = max(1, int(index_index * self.max_variation))

        if note_count != self.last_vec_size or variation != self.last_variation_span:
            self.last_melody = [random.randint(0, max(1, variation - 1)) for _ in range(note_count)]
            self.last_vec_size = note_count
            self.last_variation_span = variation

        # Chop speed from index-index: closer hands = faster chops
        chop_speeds = [1, 1/2, 1/4, 1/8]
        speed_idx = min(len(chop_speeds) - 1, int(index_index * len(chop_speeds)))
        chop_dur = chop_speeds[speed_idx]

        # Amp modulation from thumb-thumb
        amp_mod = max(0.3, min(0.7, 0.3 + thumb_thumb * 0.4))

        if self.last_melody:
            p1 >> karp(
                self.last_melody,
                dur=chop_dur,
                amp=amp_mod,
                room=0.4,
                mix=0.3,
                scale=self.current_scale,
                oct=5,
            )

        # Update bouncy arp layer too
        p2 >> spark(
            [0, 3, 5, 7, 10],
            dur=chop_dur * 2,
            amp=max(0.2, amp_mod * 0.5),
            room=0.5,
            mix=0.4,
            oct=6,
            scale=self.current_scale,
        )

    def update_from_charm(
        self,
        left_count: int,
        right_count: int,
        left_gesture: bool = False,
        right_gesture: bool = False,
    ) -> None:
        """
        Charm mode: funky finger-controlled groove.

        - Left fingers select root note (shifting the key center).
        - Right fingers control rhythmic density of the beat.
        - Cool sign triggers a wild glitch burst effect.

        Args:
            left_count: Extended finger count of left hand (0-5).
            right_count: Extended finger count of right hand (0-5).
            left_cool: Whether left hand is making the 'cool' sign.
            right_cool: Whether right hand is making the 'cool' sign.
        """
        # Left hand selects root note offset
        roots = [0, 2, 3, 5, 7, 9]
        self.current_root = roots[left_count % len(roots)]

        # Right hand controls drum density
        patterns = [
            "-",
            "x---",
            "x-o-",
            "x-o-x-o[--]",
            "x[--]o[--]x[--]o",
            "<x[--]o[--]><[xx]o >",
        ]
        self.drum_pattern = patterns[min(right_count, len(patterns) - 1)]
        d1 >> play(self.drum_pattern, dur=1/2, amp=0.8, room=0.3, mix=0.3)

        # Shift the arpeggio root
        arp_notes = [self.current_root + n for n in [0, 3, 5, 7]]
        p2 >> karp(
            arp_notes,
            dur=[1/4, 1/4, 1/2, 1/4],
            amp=0.5,
            room=0.4,
            mix=0.3,
            oct=5,
            scale=self.current_scale,
        )

        # Cool sign: glitch burst
        if left_gesture == "Cool Sign" or right_gesture == "Cool Sign":
            p1 >> spark(
                [0, 1, 2, 3, 4, 5, 6, 7],
                dur=1/8,
                amp=0.55,
                room=0.6,
                mix=0.5,
                oct=6,
                scale=self.current_scale,
            )
        else:
            if self.last_melody:
                p1 >> karp(
                    self.last_melody,
                    dur=1/4,
                    amp=0.45,
                    room=0.4,
                    mix=0.3,
                    scale=self.current_scale,
                    oct=5,
                )

    def update_from_beauty(self, gesture: int, pinch: float) -> None:
        """
        Beauty mode: expressive synth voice selection with pinch modulation.

        - Left hand gesture selects the lead synth voice.
        - Right hand pinch sweeps duration and octave for expressive control.

        Args:
            gesture: Left hand extended finger count (0-5).
            pinch: Right hand normalized index-to-thumb pinch distance [0, 1].
        """
        # Duration sweep from pinch
        dur_val = max(1/8, pinch * 2)
        play_oct = 4 + int(pinch * 3)  # oct 4 to 7

        if gesture == 0:
            # Silence the lead when fist is shown
            try:
                h1.stop()
            except Exception:
                pass
        else:
            # Different synth voices per gesture
            phrases = [
                [0, 3, 5, 7],         # Simple minor tetrad
                [0, 2, 4, 5, 7],      # Pentatonic run
                [0, 1, 3, 5, 6, 8],   # Chromatic tension
                [0, 4, 7, 11, 14],    # Wide major 7 spread
                [0, 3, 6, 9, 12, 15], # Diminished cascade
            ]
            phrase = phrases[(gesture - 1) % len(phrases)]

            h1 >> spark(
                phrase,
                dur=dur_val,
                sus=dur_val * 0.8,
                amp=0.45,
                room=0.5,
                mix=0.4,
                oct=play_oct,
                scale=self.current_scale,
            )

        # Evolving karp texture
        variation = max(1, int(pinch * self.max_variation))
        if self.last_vec_size:
            new_melody = [
                self.current_root + random.randint(0, variation)
                for _ in range(self.last_vec_size)
            ]
            p1 >> karp(
                new_melody,
                dur=1/4,
                amp=0.4,
                room=0.4,
                mix=0.3,
                scale=self.current_scale,
            )
