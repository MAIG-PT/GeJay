"""
Base song class for hand-controlled music generation.
Provides the interface and default implementations for handling keyboard inputs,
managing audio players, and reacting to hand gestures from different tracking modes.
"""

from typing import Optional, List
from FoxDot import *


class Song:
    """Base class for all interactive hand-controlled songs."""

    def __init__(self, max_vec_size: int = 20, max_variation: int = 20) -> None:
        """
        Initialize the base Song class.

        Args:
            max_vec_size: Maximum size of the generated melody note vectors.
            max_variation: Maximum variation/range of note pitches.
        """
        self.max_vec_size: int = max_vec_size
        self.max_variation: int = max_variation
        self.drum_pattern: str = "x-o-"
        self.last_vec_size: Optional[int] = None
        self.last_variation_span: Optional[int] = None
        self.last_melody: Optional[List[int]] = None

    def start(self) -> None:
        """Start playing the song's default channels and patterns."""
        d1 >> play(self.drum_pattern, dur=1/2, amp=0.9, room=0.2)
        p2 >> blip([0, PWhite(0, 7)], dur=[rest(1/2), 1/2], scale=Scale.minor, amp=0.6)

    def stop(self) -> None:
        """Stop all active audio players/synthesizers."""
        for player in (d1, p1, p2, b1, h1):
            try:
                player.stop()
            except Exception:
                pass

    def handle_key(self, key: str) -> None:
        """
        Handle a physical keyboard press forwarded by the orchestrator main loop.

        Args:
            key: Single character representing the pressed key.
        """
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

    def update_from_strange(
        self,
        left_it: float,
        right_it: float,
        index_index: float,
        thumb_thumb: float
    ) -> None:
        """
        Update audio properties in response to 'strange' mode inputs.

        Args:
            left_it: Left hand index-to-thumb normalized pinch distance [0, 1].
            right_it: Right hand index-to-thumb normalized pinch distance [0, 1].
            index_index: Inter-hand index tip-to-tip normalized distance [0, 1].
            thumb_thumb: Inter-hand thumb tip-to-tip normalized distance [0, 1].
        """
        pass

    def update_from_charm(
        self,
        left_count: int,
        right_count: int,
        left_gesture: bool = False,
        right_gesture: bool = False
    ) -> None:
        """
        Update audio properties in response to 'charm' mode inputs (finger counting).

        Args:
            left_count: Number of fingers extended on the left hand (0-5).
            right_count: Number of fingers extended on the right hand (0-5).
            left_gesture: The gesture in the left hand . ("thumbs_up", "open_hand", "rock_sign"...)
            right_gesture: The gesture in the right hand . ("thumbs_up", "open_hand", "rock_sign"...)
        """
        pass

    def update_from_beauty(self, gesture: int, pinch: float) -> None:
        """
        Update audio properties in response to 'beauty' mode inputs (mixed count + scale).

        Args:
            gesture: Extended finger count on the left hand (0-5).
            pinch: Normalized index-to-thumb pinch distance on the right hand [0, 1].
        """
        pass

