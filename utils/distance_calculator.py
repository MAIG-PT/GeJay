"""
Calculate distances between hand landmarks and normalize them.
"""

import math
from typing import Dict, Tuple, Optional, List

from config.distances_config import DISTANCE_CONFIG


class DistanceCalculator:
    """Calculate and normalize distances between hand landmarks."""

    def __init__(self):
        """Initialize distance calculator with config values."""
        self.distance_config = DISTANCE_CONFIG

    def calculate_distance(
        self, point1: Tuple[float, float], point2: Tuple[float, float]
    ) -> float:
        """
        Calculate Euclidean distance between two points.

        Args:
            point1: (x, y) normalized coordinates
            point2: (x, y) normalized coordinates

        Returns:
            Distance between points
        """
        dx = point2[0] - point1[0]
        dy = point2[1] - point1[1]
        return math.sqrt(dx**2 + dy**2)

    def normalize_distance(
        self, raw_distance: float, distance_type: str = "index_to_thumb"
    ) -> float:
        """
        Normalize distance to 0-1 range using min/max config.

        Args:
            raw_distance: Raw measured distance
            distance_type: Type of distance ('index_to_thumb', 'index_to_index', or 'thumb_to_thumb')

        Returns:
            Normalized distance in range [0, 1]
        """
        config = self.distance_config.get(distance_type, {})
        min_dist = config.get("min", 0.0)
        max_dist = config.get("max", 1.0)

        # Clamp the distance to min/max range and normalize
        clamped = max(min_dist, min(max_dist, raw_distance))
        normalized = (clamped - min_dist) / (max_dist - min_dist)
        return normalized

    def get_hand_distances(self, hand_data: dict) -> Dict[str, float]:
        """
        Calculate both raw and normalized distances for a hand.

        Args:
            hand_data: Hand information dict from HandTracker

        Returns:
            Dict with raw and normalized distances:
            {
                'index_to_thumb_raw': float,
                'index_to_thumb_normalized': float,
            }
        """
        landmarks = hand_data["landmarks_list"]

        # Index finger is landmark 8, thumb tip is landmark 4
        index_pos = landmarks[8]
        thumb_pos = landmarks[4]

        # Calculate raw distance
        raw_distance = self.calculate_distance(index_pos, thumb_pos)

        # Normalize distance
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
        self, left_hand_data: Optional[dict], right_hand_data: Optional[dict]
    ) -> Dict[str, float]:
        """
        Calculate distance between index fingers of both hands.

        Args:
            left_hand_data: Left hand information or None
            right_hand_data: Right hand information or None

        Returns:
            Dict with raw and normalized distances, or empty dict if not both hands present
        """
        if left_hand_data is None or right_hand_data is None:
            return {}

        left_landmarks = left_hand_data["landmarks_list"]
        right_landmarks = right_hand_data["landmarks_list"]

        left_index = left_landmarks[8]
        right_index = right_landmarks[8]

        # Calculate raw distance
        raw_distance = self.calculate_distance(left_index, right_index)

        # Normalize distance
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
        self, left_hand_data: Optional[dict], right_hand_data: Optional[dict]
    ) -> Dict[str, float]:
        """
        Calculate distance between thumbs of both hands.

        Args:
            left_hand_data: Left hand information or None
            right_hand_data: Right hand information or None

        Returns:
            Dict with raw and normalized distances, or empty dict if not both hands present
        """
        if left_hand_data is None or right_hand_data is None:
            return {}

        left_landmarks = left_hand_data["landmarks_list"]
        right_landmarks = right_hand_data["landmarks_list"]

        left_thumb = left_landmarks[4]
        right_thumb = right_landmarks[4]

        raw_distance = self.calculate_distance(left_thumb, right_thumb)
        normalized_distance = self.normalize_distance(
            raw_distance, distance_type="thumb_to_thumb"
        )

        return {
            "thumb_to_thumb_raw": raw_distance,
            "thumb_to_thumb_normalized": normalized_distance,
            "left_thumb_pos": left_thumb,
            "right_thumb_pos": right_thumb,
        }
