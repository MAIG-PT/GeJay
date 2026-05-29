"""
DerpyNaive implementation of interactive hand gestures mapped to FoxDot audio players.
Inherits from the base Song class, only overriding gesture-handling hooks.
"""

import random
from typing import List
from FoxDot import *
from songs.song import Song


class Song(Song):
    """
    A concrete interactive song that maps hand landmarks to musical expressions.
    Inherits default drum setup, sound players, and keyboard controls from the base Song class.
    """

    def __init__(self, max_vec_size=20, max_variation=20):
        self.max_vec_size = max_vec_size
        self.max_variation = max_variation
        self.drum_pattern = "x-o-"
        self.last_vec_size = None
        self.last_variation_span = None
        self.last_melody = None

    def start(self):
        d1 >> play(self.drum_pattern, dur=1/2, amp=0.9, room=0.2)
        p2 >> blip([0, PWhite(0, 7)], dur=[rest(1/2), 1/2], scale=Scale.minor, amp=0.6)

    def stop(self):
        for player in (d1, p1, p2, b1, h1):
            try:
                player.stop()
            except Exception:
                pass

    def handle_key(self, key):
        if key == "a":
            self.drum_pattern = "x-o-"
            d1 >> play(self.drum_pattern, dur=1/2, amp=0.9, room=0.2)

        elif key == "s":
            self.drum_pattern = "x--[--]"
            d1 >> play(self.drum_pattern, dur=1/2, amp=0.9, room=0.2)

        elif key == "d":
            self.drum_pattern = "<---[--]>< o >"
            d1 >> play(self.drum_pattern, dur=1/2, amp=0.9, room=0.2)

        elif key == "+":
            b1 >> bass([0, 0, -1, 0], dur=4, oct=4, amp=0.8, room=0.2)
            h1 >> bell(
                [(0, 3, 7), (2, 5, 6), (4, 7, 9), (7, 9, 11)],
                dur=8, sus=6, amp=0.7, oct=6, room=0.2
            )

        elif key == "-":
            b1.stop()
            h1.stop()

    def update_from_strange(self,  
        left_it: float,
        right_it: float,
        index_index: float,
        thumb_thumb: float
        ) -> None:
        """
        Update the song structure using 'strange' mode inputs.

        Generates a new melody where length depends on hand pinching distance
        and note variance depends on finger-to-finger distance.

        Args:
            length_value: Average normalized distance of index-to-thumb pinched fingers.
            variation_value: Average normalized distance between opposite hands.
        """

        length_value = (index_index + thumb_thumb) / 2 
        variation_value = (left_it + right_it)/2

        vec_size = max(1, int(length_value * self.max_vec_size))
        variation_span = max(1, int(variation_value * self.max_variation))

        if vec_size == self.last_vec_size and variation_span == self.last_variation_span:
            melody = self.last_melody
        else:
            melody = [random.randint(0, variation_span - 1) for _ in range(vec_size)]
            self.last_vec_size = vec_size
            self.last_variation_span = variation_span
            self.last_melody = melody

        p1 >> orient(
            melody,
            dur=1/4,
            blur=[1, 2],
            bend=0.1,
            scale=Scale.minor,
            room=0.2,
            amp=0.8
        )

    def update_from_charm(
        self,
        left_count: int,
        right_count: int,
        left_gesture: bool = False,
        right_gesture: bool = False
    ) -> None:
        """
        Update the song structure using 'charm' mode inputs (extended fingers).

        - Left fingers: Selects a scale (e.g., Major, Minor, Dorian).
        - Right fingers: Controls the drum beat density and tempo layout.
        - Cool sign: Triggers a high-register ambient sound effect.

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
            Scale.phrygian,
            Scale.lydian,
            Scale.mixolydian
        ]
        current_scale = scales[left_count % len(scales)]

        # Update drums based on right hand finger count
        if right_count == 0:
            self.drum_pattern = "-"
        elif right_count == 1:
            self.drum_pattern = "x---"
        elif right_count == 2:
            self.drum_pattern = "x-o-"
        elif right_count == 3:
            self.drum_pattern = "x-o[--]"
        elif right_count >= 4:
            self.drum_pattern = "x[--]o[--]"

        d1 >> play(self.drum_pattern, dur=1/2, amp=0.9, room=0.2)

        # Special effect for cool sign
        if left_gesture == "Cool Sign" or right_gesture == "Cool Sign":
            p2 >> blip([12, 14, 16], dur=1/8, amp=0.7, room=0.5, oct=6)
        else:
            try:
                p2.stop()
            except Exception:
                pass

        # Keep melody running with current scale
        if self.last_melody:
            p1 >> orient(self.last_melody, dur=1/4, scale=current_scale, amp=0.8)


    def update_from_beauty(self, gesture: int, pinch: float) -> None:
        """
        Update the song structure using 'beauty' mode inputs.

        - Left hand gesture: Triggers different chord voicings or arpeggio shapes.
        - Right hand pinch: Sets duration speed and melody variation range.

        Args:
            gesture: Left hand extended finger count (0-5).
            pinch: Right hand normalized index-to-thumb pinch distance [0, 1].
        """
        durations = [1/4, 1/2, 1, 2, 4]
        current_dur = durations[int(pinch * (len(durations) - 1))]

        if gesture == 0:
                h1.stop()
        else:
            # Different chord/arpeggio shapes based on gesture
            shapes = [
                [0, 2, 4],
                [0, 3, 7],
                [0, 4, 7, 11],
                [0, 2, 5, 7],
                [0, 3, 6, 9]
            ]
            shape = shapes[(gesture - 1) % len(shapes)]
            h1 >> bell(shape, dur=current_dur, amp=0.7, room=0.3)

        # Update melody variation based on pinch
        variation_span = max(1, int(pinch * self.max_variation))
        if self.last_vec_size:
            new_melody = [random.randint(0, variation_span) for _ in range(self.last_vec_size)]
            p1 >> orient(new_melody, dur=1/4, amp=0.8)

