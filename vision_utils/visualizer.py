"""
Visualizer utility class using OpenCV for hand tracking systems.
Provides rendering overlays for hand skeletons, connection segments, distance lines,
labels, finger counts, and UI mode banners onto camera frames.
"""

from pandas.core import frame
from typing import Dict, List, Optional, Tuple, Any
import cv2
import numpy as np

from config import VISUALIZATION_CONFIG


class Visualizer:
    """Provides methods to render overlays for landmarks, skeletons, text info, and modes."""

    def __init__(self) -> None:
        """Initialize visualizer with parameters from configuration."""
        self.config: Dict[str, Any] = VISUALIZATION_CONFIG

    def draw_hand_landmarks(
        self, frame: np.ndarray, hand_data: Dict[str, Any]
    ) -> np.ndarray:
        """
        Draw hand joint landmarks and connecting skeleton bones on a frame.

        Args:
            frame: Video image frame array (BGR).
            hand_data: Hand dictionary retrieved from HandTracker.

        Returns:
            The modified frame array with skeleton overlay.
        """
        landmarks = hand_data["landmarks_list"]
        h, w, _ = frame.shape

        # Draw circles at each landmark joint
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

        # Draw lines connecting hand joints
        self._draw_hand_connections(frame, landmarks, h, w)

        return frame

    def _draw_hand_connections(
        self,
        frame: np.ndarray,
        landmarks: List[Tuple[float, float]],
        frame_height: int,
        frame_width: int,
    ) -> None:
        """Draw lines representing skeletal connections between hand joints."""
        # MediaPipe standard 21-joint connections
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
        Draw a straight line between two joint points and label it with a metric value.

        Args:
            frame: Video image frame array (BGR).
            point1: Start coordinate tuple (normalized x, y).
            point2: End coordinate tuple (normalized x, y).
            label: Text description or measured distance overlay.
            color: RGB/BGR tuple. Defaults to magenta.

        Returns:
            The modified frame array with the distance line overlay.
        """
        if color is None:
            color = (255, 0, 255)  # Magenta

        h, w, _ = frame.shape
        x1 = int(point1[0] * w)
        y1 = int(point1[1] * h)
        x2 = int(point2[0] * w)
        y2 = int(point2[1] * h)

        # Draw standard lines
        cv2.line(frame, (x1, y1), (x2, y2), color, 2)

        # Draw text at midpoint of the line segment
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
        distances_data: Dict[str, Any],
        position: Tuple[int, int] = (10, 30),
    ) -> np.ndarray:
        """
        Render dynamic metrics text (distances, fingers) at a fixed corner location.

        Args:
            frame: Video image frame array (BGR).
            distances_data: Normalized distance dictionaries for active tracking states.
            position: Initial top-left corner offset (x, y).

        Returns:
            The modified frame array with standard debug overlay text.
        """
        y_offset = position[1]
        line_height = 25

        # Section Header
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

        # Left Hand Index-to-Thumb
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

        # Right Hand Index-to-Thumb
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

        # Index-to-Index Distance (Dual Hands)
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

        # Thumb-to-Thumb Distance (Dual Hands)
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

    def draw_text(
    self,
    frame,
    text: str,
    position: tuple[int, int],
    font_scale: float = 0.6,
    color: tuple[int, int, int] = (255, 255, 255),
    thickness: int = 1,
    font_face: int = cv2.FONT_HERSHEY_SIMPLEX,
    line_type: int = cv2.LINE_AA,
    ):
        """
        Draw text on a frame.

        Args:
            frame: The image/frame to draw on.
            text: Text to render.
            position: Bottom-left origin of the text (x, y).
            font_scale: OpenCV font scale factor.
            color: BGR text color.
            thickness: Stroke thickness.
            font_face: OpenCV font face.
            line_type: OpenCV line type, default anti-aliased.

        Returns:
            The modified frame.
        """
        cv2.putText(
            frame,
            text,
            position,
            font_face,
            font_scale,
            color,
            thickness,
            line_type,
        )
        return frame

    # BGR Mode Color Mappings
    _MODE_COLOURS: Dict[str, Tuple[int, int, int]] = {
        "strange": (127, 255, 0),    # spring green
        "charm":   (255,  50, 200),  # violet/pinkish
        "beauty":  (255,  200, 50),  # sky blue
    }

    def draw_mode_label(self, frame: np.ndarray, mode_name: str) -> np.ndarray:
        """
        Draw a premium pill-shaped banner indicating the active hand control mode.

        Args:
            frame: Video image frame array (BGR).
            mode_name: Active mode string (e.g., 'strange', 'charm', 'beauty').

        Returns:
            The modified frame array with the pill banner.
        """
        color = self._MODE_COLOURS.get(mode_name, (255, 255, 255))
        h, w = frame.shape[:2]
        label = f"[ {mode_name.upper()} ]"
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.8
        thickness = 2
        (tw, th), baseline = cv2.getTextSize(label, font, scale, thickness)
        x = w - tw - 12
        y = th + 12

        # Draw a dark backing pill rectangle
        cv2.rectangle(
            frame,
            (x - 6, y - th - 4),
            (x + tw + 6, y + baseline + 2),
            (20, 20, 20),
            -1
        )

        # Draw mode text label inside the backing pill
        cv2.putText(frame, label, (x, y), font, scale, color, thickness)
        return frame

    def draw_finger_count(
        self, frame: np.ndarray, hand_data: Dict[str, Any], count: int, label: str = ""
    ) -> np.ndarray:
        """
        Draw the finger count as a distinct numeric digit with dark circle background near the wrist.

        Args:
            frame: Video image frame array (BGR).
            hand_data: Hand tracker data structure.
            count: Number of fingers extended.
            label: Optional text modifier (e.g., "COOL").

        Returns:
            The modified frame array.
        """
        landmarks = hand_data["landmarks_list"]
        h, w = frame.shape[:2]
        wrist = landmarks[0]
        cx = int(wrist[0] * w)
        cy = int(wrist[1] * h) - 30  # Slightly above wrist landmark

        font = cv2.FONT_HERSHEY_SIMPLEX

        # Draw modern backplate circle
        cv2.circle(frame, (cx, cy), 28, (20, 20, 20), -1)

        # Text dimensions for alignment
        (tw, th), _ = cv2.getTextSize(str(count), font, 1.4, 3)
        cv2.putText(
            frame,
            str(count),
            (cx - tw // 2, cy + th // 2),
            font,
            1.4,
            (255, 220, 0),
            3
        )

        # Optional label text (e.g., "COOL!")
        if label:
            cv2.putText(frame, label, (cx + 35, cy + 10), font, 0.8, (0, 255, 255), 2)

        return frame

    def draw_gesture_tag(
        self,
        frame: np.ndarray,
        label: str,
        position: Tuple[int, int],
        color: Tuple[int, int, int],
        is_centered: bool = False,
    ) -> np.ndarray:
        """
        Draw a premium glassmorphic pill banner overlay representing a gesture tag.

        Args:
            frame: Video image frame array (BGR).
            label: Text to display in the tag (e.g. 'FIST', 'VICTORY', 'HEART').
            position: Center coordinate tuple (x, y) for the tag (if not centered).
            color: Accent color (BGR) matching the gesture's theme.
            is_centered: If True, centers the tag horizontally and vertically on the frame.

        Returns:
            The modified frame array.
        """
        h, w = frame.shape[:2]

        if is_centered:
            cx = w // 2
            cy = h // 2 - 40
        else:
            cx, cy = position

        font = cv2.FONT_HERSHEY_DUPLEX
        scale = 0.75
        thickness = 2

        # Get text size
        (text_w, text_h), baseline = cv2.getTextSize(label, font, scale, thickness)

        # Define pill bounds
        padding_x = 24
        padding_y = 12
        x1 = cx - text_w // 2 - padding_x
        y1 = cy - text_h // 2 - padding_y
        x2 = cx + text_w // 2 + padding_x
        y2 = cy + text_h // 2 + padding_y

        # Clip coordinates to frame bounds to avoid crashes/weird rendering
        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(0, min(x2, w - 1))
        y2 = max(0, min(y2, h - 1))

        # 1. Semi-transparent backing rectangle
        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (20, 20, 20), -1)
        # Apply blending
        alpha = 0.75
        cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0, frame)

        # 2. Modern accent border and left strip
        # Draw dynamic glowing border
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1, lineType=cv2.LINE_AA)

        # Accent indicator bar on the left edge
        bar_w = 4
        cv2.rectangle(
            frame,
            (x1 + 4, y1 + 6),
            (x1 + 4 + bar_w, y2 - 6),
            color,
            -1,
            lineType=cv2.LINE_AA
        )

        # 3. High-contrast centered text with a dark shadow drop
        tx = cx - text_w // 2 + (bar_w // 2 + 2)
        ty = cy + text_h // 2 - 1

        # Shadow
        cv2.putText(frame, label, (tx + 1, ty + 1), font, scale, (10, 10, 10), thickness, cv2.LINE_AA)
        # Text
        cv2.putText(frame, label, (tx, ty), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)

        return frame

    def create_window(self, window_name: str, width: int, height: int, resizable: bool = True) -> None:
        flag = cv2.WINDOW_NORMAL if resizable else cv2.WINDOW_AUTOSIZE
        cv2.namedWindow(window_name, flag)
        if resizable:
            cv2.resizeWindow(window_name, width, height)

    def show_frame(self, window_name: str, frame: np.ndarray) -> None:
        rect = cv2.getWindowImageRect(window_name)
        win_w, win_h = rect[2], rect[3]

        if win_w > 0 and win_h > 0:
            fh, fw = frame.shape[:2]
            scale = min(win_w / fw, win_h / fh)
            new_w, new_h = int(fw * scale), int(fh * scale)

            resized = cv2.resize(frame, (new_w, new_h))
            canvas = np.zeros((win_h, win_w, 3), dtype=np.uint8)

            x0 = (win_w - new_w) // 2
            y0 = (win_h - new_h) // 2
            canvas[y0:y0 + new_h, x0:x0 + new_w] = resized
            cv2.imshow(window_name, canvas)
        else:
            cv2.imshow(window_name, frame)

    def is_window_open(self, window_name: str) -> bool:
        try:
            return cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) >= 1
        except cv2.error:
            return False