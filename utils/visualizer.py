"""
Visualization utilities for hand tracking and distances.
"""

import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple

from config.distances_config import VISUALIZATION_CONFIG


class Visualizer:
    """Visualize hand landmarks and calculated distances."""

    def __init__(self):
        """Initialize visualizer with config values."""
        self.config = VISUALIZATION_CONFIG

    def draw_hand_landmarks(
        self, frame: np.ndarray, hand_data: dict
    ) -> np.ndarray:
        """
        Draw hand landmarks on the frame.

        Args:
            frame: Input frame (RGB format, height x width x 3)
            hand_data: Hand information dict from HandTracker

        Returns:
            Frame with drawn landmarks
        """
        landmarks = hand_data["landmarks_list"]
        h, w, _ = frame.shape

        # Draw circles at each landmark
        for landmark in landmarks:
            x = int(landmark[0] * w)
            y = int(landmark[1] * h)
            cv2.circle(
                frame,
                (x, y),
                self.config["landmark_circle_radius"],
                self.config["landmark_circle_color"],
                -1,
            )

        # Draw connections between hand joints
        self._draw_hand_connections(frame, landmarks, h, w)

        return frame

    def _draw_hand_connections(
        self,
        frame: np.ndarray,
        landmarks: List[Tuple[float, float]],
        frame_height: int,
        frame_width: int,
    ) -> None:
        """Draw lines connecting hand joints."""
        # MediaPipe hand connections
        connections = [
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 4),  # Thumb
            (0, 5),
            (5, 6),
            (6, 7),
            (7, 8),  # Index
            (0, 9),
            (9, 10),
            (10, 11),
            (11, 12),  # Middle
            (0, 13),
            (13, 14),
            (14, 15),
            (15, 16),  # Ring
            (0, 17),
            (17, 18),
            (18, 19),
            (19, 20),  # Pinky
        ]

        for connection in connections:
            start_idx, end_idx = connection
            start_pos = landmarks[start_idx]
            end_pos = landmarks[end_idx]

            start_x = int(start_pos[0] * frame_width)
            start_y = int(start_pos[1] * frame_height)
            end_x = int(end_pos[0] * frame_width)
            end_y = int(end_pos[1] * frame_height)

            cv2.line(
                frame,
                (start_x, start_y),
                (end_x, end_y),
                self.config["line_color"],
                self.config["line_thickness"],
            )

    def draw_distance_line(
        self,
        frame: np.ndarray,
        point1: Tuple[float, float],
        point2: Tuple[float, float],
        label: str = "",
        color: Optional[Tuple[int, int, int]] = None,
    ) -> np.ndarray:
        """
        Draw a line between two points representing a distance.

        Args:
            frame: Input frame
            point1: (x, y) normalized coordinates
            point2: (x, y) normalized coordinates
            label: Text label for the distance
            color: RGB color tuple, defaults to config color

        Returns:
            Frame with drawn distance line
        """
        if color is None:
            color = (255, 0, 255)  # Magenta for distance lines

        h, w, _ = frame.shape
        x1 = int(point1[0] * w)
        y1 = int(point1[1] * h)
        x2 = int(point2[0] * w)
        y2 = int(point2[1] * h)

        # Draw line
        cv2.line(frame, (x1, y1), (x2, y2), color, 2)

        # Draw label at midpoint
        if label:
            mid_x = (x1 + x2) // 2
            mid_y = (y1 + y2) // 2
            cv2.putText(
                frame,
                label,
                (mid_x, mid_y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                self.config["text_font_scale"],
                color,
                2,
            )

        return frame

    def draw_text_info(
        self,
        frame: np.ndarray,
        distances_data: Dict[str, Dict],
        position: Tuple[int, int] = (10, 30),
    ) -> np.ndarray:
        """
        Draw distance information as text on the frame.

        Args:
            frame: Input frame
            distances_data: Dictionary with distance measurements for each hand
            position: Starting (x, y) position for text

        Returns:
            Frame with drawn text
        """
        y_offset = position[1]
        line_height = 25

        # Draw title
        cv2.putText(
            frame,
            "Hand Distances (Normalized 0-1):",
            (position[0], y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0),
            2,
        )
        y_offset += line_height

        # Draw left hand info
        if "left" in distances_data:
            left_data = distances_data["left"]
            cv2.putText(
                frame,
                f"L Index-Thumb: {left_data['index_to_thumb_normalized']:.3f}",
                (position[0], y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                self.config["text_font_scale"],
                self.config["text_color"],
                1,
            )
            y_offset += line_height

        # Draw right hand info
        if "right" in distances_data:
            right_data = distances_data["right"]
            cv2.putText(
                frame,
                f"R Index-Thumb: {right_data['index_to_thumb_normalized']:.3f}",
                (position[0], y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                self.config["text_font_scale"],
                self.config["text_color"],
                1,
            )
            y_offset += line_height

        # Draw index-to-index distance if both hands present
        if "index_to_index_normalized" in distances_data:
            cv2.putText(
                frame,
                f"Index-to-Index: {distances_data['index_to_index_normalized']:.3f}",
                (position[0], y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                self.config["text_font_scale"],
                (0, 255, 255),
                1,
            )
            y_offset += line_height

        # Draw thumb-to-thumb distance if both hands present
        if "thumb_to_thumb_normalized" in distances_data:
            cv2.putText(
                frame,
                f"Thumb-to-Thumb: {distances_data['thumb_to_thumb_normalized']:.3f}",
                (position[0], y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                self.config["text_font_scale"],
                (255, 165, 0),
                1,
            )

        return frame
