"""
Hand tracking implementation using the MediaPipe Tasks API.
Provides high-level tracking of hands, coordinate retrieval, and gesture recognition
such as counting fingers and detecting special hand signs (e.g., the 'cool' sign).
"""

from itertools import count
from FoxDot.lib import Buffers
import time
from typing import Optional, List, Tuple, Dict, Any
from collections import deque, Counter
import cv2
import mediapipe as mp
import numpy as np
import math

from config import MEDIAPIPE_CONFIG

class HandTracker:
    """Tracks hand landmarks in real-time using MediaPipe HandLandmarker."""
    counter = 0

    def __init__(self) -> None:
        """
        Initialize the HandTracker with parameters defined in the configuration.
        Loads the pre-trained MediaPipe hand landmark model using the right API.
        """
        self.backend = MEDIAPIPE_CONFIG.get("backend", "auto")  # auto | tasks | solutions

        self.THUMB_TIP = 4
        self.INDEX_FINGER_TIP = 8

        self.hands = None
        self._last_timestamp_ms = -1
        self._start_time = time.perf_counter()

        if self.backend in ("auto", "tasks"):
            if self._init_tasks():
                self.backend = "tasks"
                return
            if self.backend == "tasks":
                raise RuntimeError("MediaPipe Tasks backend requested but could not be initialized.")

        if self.backend in ("auto", "solutions"):
            if self._init_solutions():
                self.backend = "solutions"
                return

        raise RuntimeError("Could not initialize any MediaPipe hand tracking backend.")

    def _init_tasks(self) -> bool:
        """
        Initialize the HandTracker with parameters defined in the configuration.
        Loads the pre-trained MediaPipe hand landmark model using the Tasks API.
        """
        try:
            try:
                from mediapipe.tasks import python
                from mediapipe.tasks.python import vision

                BaseOptions = python.BaseOptions
                HandLandmarker = vision.HandLandmarker
                HandLandmarkerOptions = vision.HandLandmarkerOptions
                RunningMode = vision.RunningMode
            except Exception:
                BaseOptions = mp.tasks.BaseOptions
                HandLandmarker = mp.tasks.vision.HandLandmarker
                HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
                RunningMode = mp.tasks.vision.RunningMode

            model_path = MEDIAPIPE_CONFIG.get("model_asset_path")
            if not model_path:
                return False

            self.hands = HandLandmarker.create_from_options(
                HandLandmarkerOptions(
                    base_options=BaseOptions(model_asset_path=model_path),
                    running_mode=RunningMode.VIDEO,
                    num_hands=MEDIAPIPE_CONFIG.get("max_num_hands", 2),
                    min_hand_detection_confidence=MEDIAPIPE_CONFIG.get(
                        "min_hand_detection_confidence",
                        MEDIAPIPE_CONFIG.get("min_detection_confidence", 0.5),
                    ),
                    min_hand_presence_confidence=MEDIAPIPE_CONFIG.get(
                        "min_hand_presence_confidence", 0.5
                    ),
                    min_tracking_confidence=MEDIAPIPE_CONFIG.get(
                        "min_tracking_confidence", 0.5
                    ),
                )
            )
            return True
        except Exception:
            return False

    def _init_solutions(self) -> bool:
        """
        Initialize the HandTracker with parameters defined in the configuration.
        Loads the pre-trained MediaPipe hand landmark model using the Solutions API.
        """
        try:
            self.mp_hands = mp.solutions.hands
            self.hands = self.mp_hands.Hands(
                static_image_mode=MEDIAPIPE_CONFIG.get("static_image_mode", False),
                max_num_hands=MEDIAPIPE_CONFIG.get("max_num_hands", 2),
                model_complexity=MEDIAPIPE_CONFIG.get("model_complexity", 1),
                min_detection_confidence=MEDIAPIPE_CONFIG.get(
                    "min_detection_confidence",
                    MEDIAPIPE_CONFIG.get("min_hand_detection_confidence", 0.5),
                ),
                min_tracking_confidence=MEDIAPIPE_CONFIG.get(
                    "min_tracking_confidence", 0.5
                ),
            )
            return True
        except Exception:
            return False

    def _next_timestamp_ms(self) -> int:
        ts = int((time.perf_counter() - self._start_time) * 1000)
        if ts <= self._last_timestamp_ms:
            ts = self._last_timestamp_ms + 1
        self._last_timestamp_ms = ts
        return ts



    def detect_hands(
        self,
        frame: np.ndarray,
        timestamp_ms: Optional[int] = None,
    ) -> Tuple[Optional[List[Dict[str, Any]]], np.ndarray]:
        """
        Detect hands and retrieve landmarks from a video frame.

        Args:
            frame: Input video frame from the camera (BGR format).
            timestamp_ms: Optional timestamp for Tasks VIDEO mode.

        Returns:
            A tuple containing:
            1. hands_data: List of hand data dictionaries if hands are detected, else None.
               Each dictionary has:
               {
                   'landmarks': MediaPipe normalized landmarks list,
                   'landmarks_list': List of (x, y) floats in [0, 1],
                   'handedness': 'Left' or 'Right',
                   'confidence': float
               }
            2. rgb_frame: The RGB representation of the input frame.
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if self.backend == "solutions":
            return self._detect_hands_solutions(rgb_frame), rgb_frame

        return self._detect_hands_tasks(rgb_frame, timestamp_ms), rgb_frame


    def _detect_hands_solutions(
        self, rgb_frame: np.ndarray
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Detect hands and retrieve landmarks from a video frame using Solutions API.

        Args:
            rgb_frame: Input video frame from the camera (RGB format).

        Returns:
            A tuple containing:
            1. hands_data: List of hand data dictionaries if hands are detected, else None.
               Each dictionary has:
               {
                   'landmarks': MediaPipe normalized landmarks list,
                   'landmarks_list': List of (x, y) floats in [0, 1],
                   'handedness': 'Left' or 'Right',
                   'confidence': float
               }
        """
        results = self.hands.process(rgb_frame)

        hands_data = None
        if results.multi_hand_landmarks and results.multi_handedness:
            hands_data = []
            for hand_landmarks, handedness in zip(
                results.multi_hand_landmarks,
                results.multi_handedness,
            ):
                hands_data.append(
                    {
                        "landmarks": hand_landmarks,
                        "landmarks_list": self._landmarks_to_list_solutions(hand_landmarks),
                        "handedness": handedness.classification[0].label,
                        "confidence": handedness.classification[0].score,
                    }
                )
        return hands_data

    def _detect_hands_tasks(
        self,
        rgb_frame: np.ndarray,
        timestamp_ms: Optional[int] = None,
        
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Detect hands and retrieve landmarks from a video frame using Solutions API.

        Args:
            rgb_frame: Input video frame from the camera (RGB format).
            timestamp_ms: Optional timestamp for Tasks VIDEO mode.

        Returns:
            A tuple containing:
            1. hands_data: List of hand data dictionaries if hands are detected, else None.
               Each dictionary has:
               {
                   'landmarks': MediaPipe normalized landmarks list,
                   'landmarks_list': List of (x, y) floats in [0, 1],
                   'handedness': 'Left' or 'Right',
                   'confidence': float
               }
        """
        if timestamp_ms is None:
            timestamp_ms = self._next_timestamp_ms()

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        results = self.hands.detect_for_video(mp_image, timestamp_ms)

        hands_data = None
        if results.hand_landmarks and results.handedness:
            hands_data = []
            for hand_landmarks, handedness_list in zip(
                results.hand_landmarks,
                results.handedness,
            ):
                handedness = handedness_list[0]
                hands_data.append(
                    {
                        "landmarks": hand_landmarks,
                        "landmarks_list": self._landmarks_to_list_tasks(hand_landmarks),
                        "handedness": handedness.category_name,
                        "confidence": handedness.score,
                    }
                )
        return hands_data



    def _landmarks_to_list_solutions(
        self, hand_landmarks: Any
    ) -> List[Tuple[float, float]]:
        """
        Convert MediaPipe landmarks object to a raw list of coordinate tuples for Solutions API.

        Args:
            hand_landmarks: The MediaPipe hand landmarks output object.

        Returns:
            List of (x, y) coordinates normalized between 0.0 and 1.0.
            - x represents horizontal position (0.0 = left edge, 1.0 = right edge).
            - y represents vertical position (0.0 = top edge, 1.0 = bottom edge).
        """
        return [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]

    def _landmarks_to_list_tasks(
        self, hand_landmarks: Any
    ) -> List[Tuple[float, float]]:
        """
        Convert MediaPipe landmarks object to a raw list of coordinate tuples for Tasks API.

        Args:
            hand_landmarks: The MediaPipe hand landmarks output object.

        Returns:
            List of (x, y) coordinates normalized between 0.0 and 1.0.
            - x represents horizontal position (0.0 = left edge, 1.0 = right edge).
            - y represents vertical position (0.0 = top edge, 1.0 = bottom edge).
        """
        return [(lm.x, lm.y, lm.z) for lm in hand_landmarks]

    def get_landmark_position(
        self, hand_data: Dict[str, Any], landmark_index: int
    ) -> Tuple[float, float]:
        """
        Get the coordinate of a specific hand landmark index.

        Args:
            hand_data: Hand data dictionary returned by detect_hands.
            landmark_index: MediaPipe joint identifier index (0 to 20).

        Returns:
            (x, y) normalized coordinate tuple. Defaults to (0.0, 0.0) if index is invalid.
        """
        landmarks = hand_data["landmarks_list"]
        if 0 <= landmark_index < len(landmarks):
            return landmarks[landmark_index]
        return (0.0, 0.0)

    def close(self) -> None:
        """Release underlying MediaPipe landmarker resources."""
        if self.hands is not None:
            self.hands.close()

    def _dist(self, p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
        """Calculate the Euclidean distance between two 2D points."""
        import math
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

    def _vec3(self, p1: Tuple[float, float, float], p2: Tuple[float, float, float]) -> Tuple[float, float, float]:
        return (p2[0] - p1[0], p2[1] - p1[1], p2[2] - p1[2])

    def _norm3(self, v: Tuple[float, float, float]) -> float:
        return math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)

    def _unit3(self, v: Tuple[float, float, float]) -> Tuple[float, float, float]:
        n = self._norm3(v)
        if n < 1e-6:
            return (0.0, 0.0, 0.0)
        return (v[0] / n, v[1] / n, v[2] / n)

    def _scalar_projection(self, point, origin, axis_unit) -> float:
        return (
            (point[0] - origin[0]) * axis_unit[0] +
            (point[1] - origin[1]) * axis_unit[1]
        )

    def _unit2(self, v):
        n = math.hypot(v[0], v[1])
        if n < 1e-6:
            return (0.0, 0.0)
        return (v[0] / n, v[1] / n)

    def _cross3(self, a, b):
        return (
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        )

    def _dot3(self, a, b):
        return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]

    def _midpoint3(self, a, b):
        return (
            (a[0] + b[0]) * 0.5,
            (a[1] + b[1]) * 0.5,
            (a[2] + b[2]) * 0.5,
        )

    def _angle3(self, a, b, c):
        """
        Angle ABC in degrees.
        """
        ba = (a[0] - b[0], a[1] - b[1], a[2] - b[2])
        bc = (c[0] - b[0], c[1] - b[1], c[2] - b[2])

        uba = self._unit3(ba)
        ubc = self._unit3(bc)

        dot = self._dot3(uba, ubc)
        dot = max(-1.0, min(1.0, dot))
        return math.degrees(math.acos(dot))

    def _orient2d(self, a, b, c) -> float:
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

    def _segments_intersect_2d(self, p1, p2, q1, q2) -> bool:
        o1 = self._orient2d(p1, p2, q1)
        o2 = self._orient2d(p1, p2, q2)
        o3 = self._orient2d(q1, q2, p1)
        o4 = self._orient2d(q1, q2, p2)

        return (o1 * o2 < 0) and (o3 * o4 < 0)

    def fingers_cross(
    self,
    hand_data: Dict[str, Any],
    finger_a: str,
    finger_b: str) -> bool:
        landmarks = hand_data["landmarks_list"]

        finger_a_propos = self.get_finger_properties(hand_data, finger_a)
        finger_b_propos = self.get_finger_properties(hand_data, finger_b)
  
        return self._segments_intersect_2d(finger_a_propos["base_point"],
            finger_a_propos["pip_point"],
            finger_b_propos["base_point"], 
            finger_b_propos["pip_point"]) or \
            self._segments_intersect_2d(finger_a_propos["pip_point"],
            finger_a_propos["tip_point"],
            finger_b_propos["pip_point"], 
            finger_b_propos["tip_point"]) or \
            self._segments_intersect_2d(finger_a_propos["base_point"],
            finger_a_propos["tip_point"],
            finger_b_propos["base_point"],
            finger_b_propos["tip_point"])

    def fingers_angle(self,
    hand_data: Dict[str, Any],
    finger_a: str,
    finger_b: str) -> float:
        a = self.get_finger_properties(hand_data, finger_a)
        b = self.get_finger_properties(hand_data, finger_b)
        ua = a["vector"]
        ub = b["vector"]
        dot = ua[0] * ub[0] + ua[1] * ub[1] + ua[2] * ub[2]
        dot = max(-1.0, min(1.0, dot))
        angle_deg = math.degrees(math.acos(dot))
        return angle_deg


    def finger_spacing_state(
    self,
    hand_data: Dict[str, Any],
    finger_a: str,
    finger_b: str,
    require_open: bool = True,
    ) -> str:
        a = self.get_finger_properties(hand_data, finger_a)
        b = self.get_finger_properties(hand_data, finger_b)

        if require_open and (not a["open"] or not b["open"]):
            return "joint"

        if self.fingers_cross(hand_data, finger_a, finger_b):
            return "joint"

        landmarks = hand_data["landmarks_list"]
        palm_width = self._dist(landmarks[5], landmarks[17])
        if palm_width < 1e-6:
            return "joint"

        tip_ratio = self._dist(a["tip_point"], b["tip_point"]) / palm_width

        ua = a["vector"]
        ub = b["vector"]
        dot = ua[0] * ub[0] + ua[1] * ub[1] + ua[2] * ub[2]
        dot = max(-1.0, min(1.0, dot))
        angle_deg = math.degrees(math.acos(dot))

        pair = tuple(sorted((finger_a, finger_b)))

        thresholds = {
            ("index", "thumb"): {
                "joint": {"tip_ratio_max": 1.9, "angle_deg_max": 10},
                "spread": {"tip_ratio_min": 2.0, "angle_deg_min": 50},
            },
            ("index", "middle"): {
                "joint": {"tip_ratio_max": 0.79, "angle_deg_max": 10.4},
                "spread": {"tip_ratio_min": 0.8, "angle_deg_min": 14},
            },
            ("middle", "ring"): {
                "joint": {"tip_ratio_max": 0.46, "angle_deg_max": 3.0},
                "spread": {"tip_ratio_min": 0.68, "angle_deg_min": 10.9},
            },
            ("pinky", "ring"): {
                "joint": {"tip_ratio_max": 0.8, "angle_deg_max": 8},
                "spread": {"tip_ratio_min": 1.14, "angle_deg_min": 25.1},
            },
        }

        default_thresholds = {
            "joint": {"tip_ratio_max": 0.50, "angle_deg_max": 8.0},
            "spread": {"tip_ratio_min": 0.90, "angle_deg_min": 18.0},
        }

        t = thresholds.get(pair, default_thresholds)

        #print(f"fingers: {pair}, tip_ratio: {tip_ratio}, angle_deg: {angle_deg}, t: {t}")

        is_joint = (
            tip_ratio <= t["joint"]["tip_ratio_max"] and
            angle_deg <= t["joint"]["angle_deg_max"]
        )
        if is_joint:
            return "joint"

        is_spread = (
            tip_ratio >= t["spread"]["tip_ratio_min"] and
            angle_deg >= t["spread"]["angle_deg_min"]
        )
        if is_spread:
            return "spread"

        return "natural"

    def get_finger_count(self, hand_data: Dict[str, Any]) -> int:
        """
        Count the number of extended fingers (0 to 5) on a given hand.

        Uses an extremely robust rotation-invariant relative-distance algorithm:
        - Non-thumb fingers are open if the Tip distance to Wrist is greater than PIP distance to Wrist.
        - The thumb is open if the Tip is spread laterally away from the middle finger base MCP (9).

        Args:
            hand_data: Hand data dictionary containing landmarks.

        Returns:
            Count of extended fingers.
        """
        landmarks = hand_data["landmarks_list"]
        count = 0

        thumb_props = self.get_finger_properties(hand_data, "thumb")
        index_props = self.get_finger_properties(hand_data, "index")
        middle_props = self.get_finger_properties(hand_data, "middle")
        ring_props = self.get_finger_properties(hand_data, "ring")
        pinky_props = self.get_finger_properties(hand_data, "pinky")

        thumb_up = thumb_props["open"]
        index_up = index_props["open"]
        middle_up = middle_props["open"]
        ring_up = ring_props["open"]
        pinky_up = pinky_props["open"]

        count = int(thumb_up) + int(index_up) + int(middle_up) + int(ring_up) + int(pinky_up)

        return count

    def get_palm_properties(
    self,
    hand_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Palm properties dict.

        Requires landmarks_list to store (x, y, z).
        """
        landmarks = hand_data["landmarks_list"]
        handedness = hand_data.get("handedness", None)

        # ---- Palm facing computation ----
        wrist_pt = landmarks[0]
        index_mcp = landmarks[5]
        pinky_mcp = landmarks[17]

        v_wrist_index = self._vec3(wrist_pt, index_mcp)
        v_wrist_pinky = self._vec3(wrist_pt, pinky_mcp)

        if handedness == "Left":
            palm_normal = self._unit3(self._cross3(v_wrist_index, v_wrist_pinky))
        elif handedness == "Right":
            palm_normal = self._unit3(self._cross3(v_wrist_pinky, v_wrist_index))
        else:
            palm_normal = self._unit3(self._cross3(v_wrist_index, v_wrist_pinky))

        # In MediaPipe-style image/world conventions, lower z generally means closer to camera.
        # So a palm normal with negative z points more toward the camera.
        palm_facing_camera = palm_normal[2]

        min_angle = 0.5
        return {
            # palm facing
            "palm_normal": palm_normal,
            "palm_facing_camera_metric": palm_facing_camera,
            "palm_facing_camera": palm_facing_camera > min_angle,
            "palm_facing_away_from_camera": palm_facing_camera < -min_angle,
            "palm_sideways": abs(palm_facing_camera) <= min_angle,
        }

    def _thumb_is_closed(self, hand_data: Dict[str, Any]) -> bool:
        landmarks = hand_data["landmarks_list"]

        thumb_tip = landmarks[4]
        index_mcp = landmarks[5]
        pinky_mcp = landmarks[17]

        a = index_mcp
        b = pinky_mcp
        p = thumb_tip

        ab = (b[0] - a[0], b[1] - a[1], b[2] - a[2])
        ap = (p[0] - a[0], p[1] - a[1], p[2] - a[2])

        ab_len_sq = ab[0]**2 + ab[1]**2 + ab[2]**2
        if ab_len_sq < 1e-8:
            return True

        t = (ap[0]*ab[0] + ap[1]*ab[1] + ap[2]*ab[2]) / ab_len_sq

        return -0.4 <= t <= 1.4


    def _thumb_is_open(self, hand_data: Dict[str, Any]) -> bool:
        return not self._thumb_is_closed(hand_data)

    def get_finger_properties(
    self,
    hand_data: Dict[str, Any],
    finger_name: str
    ) -> Dict[str, Any]:
        """
        Per-finger properties dict.

        Requires landmarks_list to store (x, y, z).
        """
        landmarks = hand_data["landmarks_list"]

        finger_map = {
            "thumb":  {"base": 2,  "pip": 3,  "tip": 4},
            "index":  {"base": 5,  "pip": 6,  "tip": 8},
            "middle": {"base": 9,  "pip": 10, "tip": 12},
            "ring":   {"base": 13, "pip": 14, "tip": 16},
            "pinky":  {"base": 17, "pip": 18, "tip": 20},
        }

        if finger_name not in finger_map:
            raise ValueError(f"Unknown finger name: {finger_name}")

        base_idx = finger_map[finger_name]["base"]
        pip_idx = finger_map[finger_name]["pip"]
        tip_idx = finger_map[finger_name]["tip"]

        base = landmarks[base_idx]
        pip = landmarks[pip_idx]
        tip = landmarks[tip_idx]
        wrist = landmarks[0]

        v = self._vec3(base, tip)
        u = self._unit3(v)

        # Your existing open/closed logic
        palm_facing_camera = self.get_palm_properties(hand_data)["palm_facing_camera"]

        if finger_name == "thumb":
            is_open = self._thumb_is_open(hand_data)
        else:
            is_open = self._dist(tip, wrist) > self._dist(pip, wrist)

        handedness = hand_data.get("handedness", None)

        pointing_inward = False
        pointing_outward = False

        if handedness == "Left":
            # Left hand: screen-right is inward, screen-left is outward
            pointing_inward = u[0] < -0.35
            pointing_outward = u[0] > 0.35

        elif handedness == "Right":
            # Right hand: screen-left is inward, screen-right is outward
            pointing_inward = u[0] > 0.35
            pointing_outward = u[0] < -0.35

        return {
            "finger": finger_name,
            "base_index": base_idx,
            "pip_index": pip_idx,
            "tip_index": tip_idx,
            "base_point": base,
            "pip_point": pip,
            "tip_point": tip,
            "wrist_point": wrist,
            "open": is_open,
            "vector": u,              # (x, y, z)
            "horizontal": u[0],       # +right / -left
            "vertical": u[1],         # +down / -up
            "depth": u[2],            # +farther / -closer
            "magnitude": self._norm3(v),

            # convenient booleans
            "pointing_up": u[1] < -0.35,
            "pointing_down": u[1] > 0.35,
            "pointing_right": u[0] > 0.35,
            "pointing_left": u[0] < -0.35,
            "toward_camera": u[2] < -0.25,
            "away_from_camera": u[2] > 0.25,
            "pointing_inward": pointing_inward,
            "pointing_outward": pointing_outward,
        }

    def get_active_gesture(self, hand_data: Dict[str, Any]) -> str:
        """
        Classify the active single-hand gesture based on extended fingers.

        Provides a highly extensible registry of hand poses.

        Args:
            hand_data: Hand data dictionary.

        Returns:
            String name of the active gesture (e.g., 'Fist', 'Victory', 'Cool Sign', etc.).
        """
        landmarks = hand_data["landmarks_list"]

        # Determine individual finger properties
        thumb_props = self.get_finger_properties(hand_data, "thumb")
        index_props = self.get_finger_properties(hand_data, "index")
        middle_props = self.get_finger_properties(hand_data, "middle")
        ring_props = self.get_finger_properties(hand_data, "ring")
        pinky_props = self.get_finger_properties(hand_data, "pinky")

        palm_props = self.get_palm_properties(hand_data)

        thumb_up = thumb_props["open"]
        index_up = index_props["open"]
        middle_up = middle_props["open"]
        ring_up = ring_props["open"]
        pinky_up = pinky_props["open"]

        thumb_index_spacing = self.finger_spacing_state(hand_data, "thumb", "index")
        index_middle_spacing = self.finger_spacing_state(hand_data, "index", "middle") 
        middle_ring_spacing = self.finger_spacing_state(hand_data, "middle", "ring") 
        ring_pinky_spacing = self.finger_spacing_state(hand_data, "ring", "pinky") 

        thumb_index_crossed = self.fingers_cross(hand_data, "thumb", "index") 
        index_middle_crossed = self.fingers_cross(hand_data, "index", "middle")
        middle_ring_crossed = self.fingers_cross(hand_data, "middle", "ring")
        ring_pinky_crossed = self.fingers_cross(hand_data, "ring", "pinky")

        # if HandTracker.counter == 20:
        #     print("--------Hand Info----------")
        #     print("thumb up", thumb_up)
        #     print("index up", index_up)
        #     print("middle up", middle_up)
        #     print("ring up", ring_up)
        #     print("pinky up", pinky_up)
        #     print("thumb index spacing", thumb_index_spacing)
        #     print("index middle spacing", index_middle_spacing)
        #     print("middle ring spacing", middle_ring_spacing)
        #     print("ring pinky spacing", ring_pinky_spacing)
        #     print(palm_props)
        #     HandTracker.counter = 0
        # else:
        #     HandTracker.counter += 1

        #print(f"-----\nthumb up: {thumb_up}, \nindex up: {index_up},\nmiddle up: {middle_up}, \nring up: {ring_up}, \npinky up: {pinky_up}")

        # Sum of extended fingers
        up_count = sum([thumb_up, index_up, middle_up, ring_up, pinky_up])

        # 1. Thumbs Up / Thumbs Down: Thumb up, other fingers folded)
        if thumb_up and not (index_up or middle_up or ring_up or pinky_up) and self.fingers_angle(hand_data, "thumb", "index") > 40:
            # Check if thumb tip is physically above the thumb MCP (upright screen coordinates)
            if thumb_props["pointing_up"]:
                return "Thumbs Up"
            else:
                return "Thumbs Down"

        # 2. Rock Sign: Index and Pinky up, Middle and Ring folded
        if index_up and pinky_up and not middle_up and not ring_up:
            return "Rock Sign"

        # 3. Cool Sign: Thumb up, pinky up, ring down, index down
        if thumb_up and pinky_up and not middle_up and not ring_up and not index_up and thumb_props["pointing_outward"] and pinky_props["pointing_inward"]:
            return "Cool Sign"

        # 4. Call Me Sign: Thumb up, pinky up, ring down, index down, middle down
        if thumb_up and pinky_up and not middle_up and not ring_up and not index_up and thumb_props["pointing_up"] and pinky_props["pointing_down"]:
            return "Call Me"

        # 5. Victory: Index and Middle up, Ring and Pinky folded
        if index_up and middle_up and not (ring_up or pinky_up or thumb_up) and not index_middle_crossed and palm_props["palm_facing_camera"] and not index_middle_spacing == "joint":
            return "Victory"

        # 6. Mini heart/Finger heart: crossing index and thumb 
        if index_up and thumb_up and thumb_index_crossed and not (middle_up or ring_up or pinky_up):
            return "Mini Heart"

        # 7. Fist: 0 fingers up
        if up_count == 0:
            return "Fist"

        # 8. Looser: Index up vertically, thumb inward, other fingers down
        if index_props["open"] and \
           index_props["pointing_up"] and \
           thumb_props["open"] and \
           thumb_props["pointing_inward"] and \
           not (middle_up or ring_up or pinky_up) and\
           thumb_index_spacing == "spread":
            return "Looser"

        # 9. Point: Only Index is up, others are down
        if index_up and not (middle_up or ring_up or pinky_up):
            return "Point"

        # 10. Crossed Fingers: Index and Middle up, crossed
        if index_up and middle_up and not (ring_up or pinky_up) and index_middle_crossed:
            return "Crossed Fingers"

        # 12. Middle Finger: Thumb down, pinky down, ring down, index down, middle up
        if middle_up and not pinky_up and not ring_up and not index_up:
            return "Middle Finger"

        # 13. Fancy: Pinky up, thumb down, ring down, index down, middle down
        if pinky_up and not thumb_up and not middle_up and not ring_up and not index_up:
            return "Fancy"

        # 14. Spock: all up but index, middle and ring are apart and close to each other 
        if pinky_up and middle_up and ring_up and index_up and index_middle_spacing == "joint" and ring_pinky_spacing == "joint" and middle_ring_spacing == "spread" and palm_props["palm_facing_camera"]:
            return "Spock"

        # 15. Stop: all up and close to each other palm face to the camera
        if pinky_up and thumb_up and middle_up and ring_up and index_up and index_middle_spacing == "joint" and ring_pinky_spacing == "joint" and  middle_ring_spacing == "joint" and palm_props["palm_facing_camera"]:
            return "Stop"
        
        # 16. Gun:
        if thumb_up and \
            index_up and middle_up and not (ring_up or pinky_up) and\
            not thumb_index_spacing == "joint" and index_middle_spacing == "joint":
            return "Gun"

        # 17. Open Hand: All 5 fingers up
        if up_count == 5:
            return "Open Hand"

        # Standard intermediate counts
        if up_count == 1:
            return "One Fingers"
        if up_count == 2:
            return "Two Fingers"
        if up_count == 3:
            return "Three Fingers"
        if up_count == 4:
            return "Four Fingers"

        return "Unknown"

    def detect_two_hand_gesture(
        self, left_hand_data: Optional[Dict[str, Any]], right_hand_data: Optional[Dict[str, Any]]
    ) -> Optional[str]:
        """
        Detect simple two-hand gestures:
        - Circle
        - Heart

        Geometry:
        - Build an axis from thumb-tip midpoint to index-tip midpoint.
        - Project key landmarks onto that axis.
        - Circle: thumb/index tips are the extreme points on that axis.
        - Heart: index tips are lower on that axis than the first index knuckle.
        """
        if left_hand_data is None or right_hand_data is None:
            return None

        left = left_hand_data["landmarks_list"]
        right = right_hand_data["landmarks_list"]

        THUMB_TIP = 4
        INDEX_MCP = 5
        INDEX_PIP = 6   # "first knuckle" in most practical gesture code
        INDEX_DIP = 7
        INDEX_TIP = 8
        MIDDLE_TIP = 12
        RING_TIP = 16
        PINKY_TIP = 20

        def dist(a, b) -> float:
            return math.hypot(a[0] - b[0], a[1] - b[1])

        def avg(a, b):
            return ((a[0] + b[0]) * 0.5, (a[1] + b[1]) * 0.5)

        def sub(a, b):
            return (a[0] - b[0], a[1] - b[1])

        def dot(a, b) -> float:
            return a[0] * b[0] + a[1] * b[1]

        def norm(v) -> float:
            return math.hypot(v[0], v[1])

        def unit(v):
            n = norm(v)
            if n < 1e-6:
                return (0.0, 0.0)
            return (v[0] / n, v[1] / n)

        def hand_scale(lm) -> float:
            return max(
                0.5 * (dist(lm[0], lm[9]) + dist(lm[5], lm[17])),
                1e-6
            )

        def project_scalar(point, origin, axis_unit) -> float:
            return dot(sub(point, origin), axis_unit)

        def finger_extended(lm, tip_idx, pip_idx) -> bool:
            return dist(lm[tip_idx], lm[0]) > dist(lm[pip_idx], lm[0])

        left_scale = hand_scale(left)
        right_scale = hand_scale(right)
        pair_scale = 0.5 * (left_scale + right_scale)

        left_thumb = left[THUMB_TIP]
        right_thumb = right[THUMB_TIP]
        left_index_tip = left[INDEX_TIP]
        right_index_tip = right[INDEX_TIP]
        left_index_pip = left[INDEX_PIP]
        right_index_pip = right[INDEX_PIP]

        thumb_mid = avg(left_thumb, right_thumb)
        index_mid = avg(left_index_tip, right_index_tip)

        axis_vec = sub(index_mid, thumb_mid)
        axis_len = norm(axis_vec)

        if axis_len < 0.15 * pair_scale:
            return None

        axis_u = unit(axis_vec)
        axis_origin = thumb_mid

        # Projections onto thumb->index axis
        l_thumb_s = project_scalar(left_thumb, axis_origin, axis_u)
        r_thumb_s = project_scalar(right_thumb, axis_origin, axis_u)
        l_index_s = project_scalar(left_index_tip, axis_origin, axis_u)
        r_index_s = project_scalar(right_index_tip, axis_origin, axis_u)
        l_pip_s = project_scalar(left_index_pip, axis_origin, axis_u)
        r_pip_s = project_scalar(right_index_pip, axis_origin, axis_u)

        all_s = [l_thumb_s, r_thumb_s, l_index_s, r_index_s]
        min_s = min(all_s)
        max_s = max(all_s)

        # Sideways deviation from axis, useful so the hands are not wildly misaligned
        def point_line_distance(point, origin, axis_unit) -> float:
            v = sub(point, origin)
            parallel = dot(v, axis_unit)
            proj = (axis_unit[0] * parallel, axis_unit[1] * parallel)
            perp = (v[0] - proj[0], v[1] - proj[1])
            return norm(perp)

        thumb_axis_error = (
            point_line_distance(left_thumb, axis_origin, axis_u) +
            point_line_distance(right_thumb, axis_origin, axis_u)
        ) / (2.0 * pair_scale)

        index_axis_error = (
            point_line_distance(left_index_tip, axis_origin, axis_u) +
            point_line_distance(right_index_tip, axis_origin, axis_u)
        ) / (2.0 * pair_scale)

        # Optional fold checks to avoid open palms being mistaken for gestures
        left_middle_up = finger_extended(left, MIDDLE_TIP, 10)
        left_ring_up = finger_extended(left, RING_TIP, 14)
        left_pinky_up = finger_extended(left, PINKY_TIP, 18)
        right_middle_up = finger_extended(right, MIDDLE_TIP, 10)
        right_ring_up = finger_extended(right, RING_TIP, 14)
        right_pinky_up = finger_extended(right, PINKY_TIP, 18)

        other_fingers_folded = not (
            left_middle_up or left_ring_up or left_pinky_up or
            right_middle_up or right_ring_up or right_pinky_up
        )

        # Hand-level pinches: thumb and index touching on each hand
        left_pinch = dist(left_thumb, left_index_tip) / left_scale < 0.55
        right_pinch = dist(right_thumb, right_index_tip) / right_scale < 0.55

        # Circle:
        # 1) both hands pinch
        # 2) thumb tips occupy the low end of the axis
        # 3) index tips occupy the high end of the axis
        # 4) tips are reasonably aligned with the axis
        thumbs_at_low_end = (
            abs(l_thumb_s - min_s) < 0.22 * pair_scale and
            abs(r_thumb_s - min_s) < 0.22 * pair_scale
        )
        indexes_at_high_end = (
            abs(l_index_s - max_s) < 0.22 * pair_scale and
            abs(r_index_s - max_s) < 0.22 * pair_scale
        )

        if (
            left_pinch and right_pinch and
            thumbs_at_low_end and
            indexes_at_high_end and
            thumb_axis_error < 0.30 and
            index_axis_error < 0.30
        ):
            return "Circle"

        # Heart:
        # Same general thumb/index arrangement, but curled index tips:
        # index tip projection is behind its own PIP projection on the thumb->index axis.
        # In other words, the first knuckle is farther along the axis than the fingertip.
        left_heart_index = l_index_s < l_pip_s
        right_heart_index = r_index_s < r_pip_s

        thumbs_grouped = dist(left_thumb, right_thumb) / pair_scale < 0.9
        indexes_grouped = dist(left_index_tip, right_index_tip) / pair_scale < 0.9

        if (
            thumbs_grouped and
            indexes_grouped and
            left_heart_index and
            right_heart_index and
            other_fingers_folded
        ):
            return "Heart"

        return None