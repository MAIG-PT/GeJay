"""
Distance calculation and normalization utilities for hand landmarks.
Provides tools to measure distances between specific fingers on a single hand
and between fingers of two separate hands, mapping raw metrics to a [0.0, 1.0] scale.
"""

import math
from typing import Dict, Tuple, Optional, Any

from config import DISTANCE_CONFIG

class DistanceCalculator:
    """Calculates and normalizes spatial distances between hand landmarks."""

    def __init__(self) -> None:
        """Initialize the DistanceCalculator with boundaries loaded from configuration."""
        self.distance_config: Dict[str, Dict[str, float]] = DISTANCE_CONFIG

    def calculate_distance(
        self, point1: Tuple[float, float], point2: Tuple[float, float]
    ) -> float:
        """
        Calculate the standard Euclidean distance between two points in 2D space.

        Args:
            point1: Normalized (x, y) coordinates of the first point.
            point2: Normalized (x, y) coordinates of the second point.

        Returns:
            The Euclidean distance as a float.
        """
        dx = point2[0] - point1[0]
        dy = point2[1] - point1[1]
        return math.sqrt(dx**2 + dy**2)

    def normalize_distance(
        self, raw_distance: float, distance_type: str = "index_to_thumb"
    ) -> float:
        """
        Normalize a raw spatial distance to a [0.0, 1.0] range using config bounds.

        Clamps the distance value first, then maps it linearly between the min and max limits.

        Args:
            raw_distance: The raw measured distance float value.
            distance_type: Type identifier ('index_to_thumb', 'index_to_index', 'thumb_to_thumb').

        Returns:
            A normalized float value in [0.0, 1.0].
        """
        config = self.distance_config.get(distance_type, {})
        min_dist = config.get("min", 0.0)
        max_dist = config.get("max", 1.0)

        # Clamp raw distance to defined config ranges
        clamped = max(min_dist, min(max_dist, raw_distance))

        # Linear mapping to [0.0, 1.0]
        normalized = (clamped - min_dist) / (max_dist - min_dist)
        return normalized

    def get_hand_distances(self, hand_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate raw and normalized distances between index finger and thumb tip of a single hand.

        Args:
            hand_data: Hand dictionary retrieved from HandTracker.

        Returns:
            A dictionary containing:
            - 'index_to_thumb_raw': Raw distance float.
            - 'index_to_thumb_normalized': Clamped and scaled float in [0.0, 1.0].
            - 'index_pos': (x, y) coordinates of the index finger tip.
            - 'thumb_pos': (x, y) coordinates of the thumb tip.
        """
        landmarks = hand_data["landmarks_list"]

        # Index tip = landmark 8, Thumb tip = landmark 4
        index_pos = landmarks[8]
        thumb_pos = landmarks[4]

        # Calculate raw spatial distance
        raw_distance = self.calculate_distance(index_pos, thumb_pos)

        # Normalize to 0-1 scale
        normalized_distance = self.normalize_distance(
            raw_distance, distance_type="index_to_thumb"
        )

        return {
            "index_to_thumb_raw": raw_distance,
            "index_to_thumb_normalized": normalized_distance,
            "index_pos": index_pos,
            "thumb_pos": thumb_pos,
        }

    def get_index_to_index_distance(
        self, left_hand_data: Optional[Dict[str, Any]], right_hand_data: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Calculate raw and normalized distance between index finger tips of opposite hands.

        Args:
            left_hand_data: Left hand dictionary or None.
            right_hand_data: Right hand dictionary or None.

        Returns:
            A dictionary with results if both hands are active, otherwise an empty dictionary.
            Contains:
            - 'index_to_index_raw': Raw distance float.
            - 'index_to_index_normalized': Clamped and scaled float in [0.0, 1.0].
            - 'left_index_pos': Left index tip (x, y).
            - 'right_index_pos': Right index tip (x, y).
        """
        if left_hand_data is None or right_hand_data is None:
            return {}

        left_landmarks = left_hand_data["landmarks_list"]
        right_landmarks = right_hand_data["landmarks_list"]

        left_index = left_landmarks[8]
        right_index = right_landmarks[8]

        # Raw distance between left and right index finger tips
        raw_distance = self.calculate_distance(left_index, right_index)

        # Normalize distance based on Index-to-Index parameters
        normalized_distance = self.normalize_distance(
            raw_distance, distance_type="index_to_index"
        )

        return {
            "index_to_index_raw": raw_distance,
            "index_to_index_normalized": normalized_distance,
            "left_index_pos": left_index,
            "right_index_pos": right_index,
        }

    def get_thumb_to_thumb_distance(
        self, left_hand_data: Optional[Dict[str, Any]], right_hand_data: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Calculate raw and normalized distance between thumb tips of opposite hands.

        Args:
            left_hand_data: Left hand dictionary or None.
            right_hand_data: Right hand dictionary or None.

        Returns:
            A dictionary with results if both hands are active, otherwise an empty dictionary.
            Contains:
            - 'thumb_to_thumb_raw': Raw distance float.
            - 'thumb_to_thumb_normalized': Clamped and scaled float in [0.0, 1.0].
            - 'left_thumb_pos': Left thumb tip (x, y).
            - 'right_thumb_pos': Right thumb tip (x, y).
        """
        if left_hand_data is None or right_hand_data is None:
            return {}

        left_landmarks = left_hand_data["landmarks_list"]
        right_landmarks = right_hand_data["landmarks_list"]

        left_thumb = left_landmarks[4]
        right_thumb = right_landmarks[4]

        # Raw distance between left and right thumb tips
        raw_distance = self.calculate_distance(left_thumb, right_thumb)

        # Normalize distance based on Thumb-to-Thumb parameters
        normalized_distance = self.normalize_distance(
            raw_distance, distance_type="thumb_to_thumb"
        )

        return {
            "thumb_to_thumb_raw": raw_distance,
            "thumb_to_thumb_normalized": normalized_distance,
            "left_thumb_pos": left_thumb,
            "right_thumb_pos": right_thumb,
        }

