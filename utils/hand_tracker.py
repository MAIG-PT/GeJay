"""
Hand tracking using MediaPipe.
"""

import cv2
import mediapipe as mp
from typing import Optional, List, Tuple
import numpy as np

from config.distances_config import MEDIAPIPE_CONFIG


class HandTracker:
    """Tracks hand landmarks using MediaPipe."""

    def __init__(self, model_complexity: Optional[int] = None):
        """Initialize MediaPipe hand detector."""
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=MEDIAPIPE_CONFIG["static_image_mode"],
            max_num_hands=MEDIAPIPE_CONFIG["max_num_hands"],
            model_complexity=(MEDIAPIPE_CONFIG["model_complexity"]
                              if model_complexity is None else model_complexity),
            min_detection_confidence=MEDIAPIPE_CONFIG["min_detection_confidence"],
            min_tracking_confidence=MEDIAPIPE_CONFIG["min_tracking_confidence"],
        )

        # Hand landmark indices
        self.THUMB_TIP = 4
        self.INDEX_FINGER_TIP = 8

    def detect_hands(
        self, frame: np.ndarray
    ) -> Tuple[Optional[List[dict]], np.ndarray]:
        """
        Detect hands in the frame.

        Args:
            frame: Input video frame (BGR format from OpenCV)

        Returns:
            Tuple of (hands_data, annotated_frame) where hands_data is a list of dicts
            containing hand information, or None if no hands detected.
        """
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, _ = frame.shape

        # Process frame
        rgb_frame.flags.writeable = False
        try:
            results = self.hands.process(rgb_frame)
        finally:
            rgb_frame.flags.writeable = True

        hands_data = None
        if results.multi_hand_landmarks and results.multi_handedness:
            hands_data = []
            for hand_landmarks, handedness in zip(
                results.multi_hand_landmarks, results.multi_handedness
            ):
                hand_info = {
                    "landmarks": hand_landmarks,
                    "landmarks_list": self._landmarks_to_list(hand_landmarks, h, w),
                    "handedness": handedness.classification[0].label,  # 'Left' or 'Right'
                    "confidence": handedness.classification[0].score,
                }
                hands_data.append(hand_info)

        return hands_data, rgb_frame

    def _landmarks_to_list(
        self, hand_landmarks, frame_height: int, frame_width: int
    ) -> List[Tuple[float, float]]:
        """Convert MediaPipe landmarks to list of normalized coordinates."""
        landmarks_list = []
        for landmark in hand_landmarks.landmark:
            x = landmark.x
            y = landmark.y
            landmarks_list.append((x, y))
        return landmarks_list

    def get_landmark_position(
        self, hand_data: dict, landmark_index: int
    ) -> Tuple[float, float]:
        """
        Get normalized coordinates of a specific landmark.

        Args:
            hand_data: Hand information dict from detect_hands
            landmark_index: Landmark index (0-20)

        Returns:
            Tuple of (x, y) normalized coordinates
        """
        landmarks = hand_data["landmarks_list"]
        if landmark_index < len(landmarks):
            return landmarks[landmark_index]
        return (0, 0)

    def close(self):
        """Close the hand tracker."""
        self.hands.close()
