from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Tuple

from vision_utils import HandTracker, DistanceCalculator


WINDOW_NAME = "Gejay - Control Music using your hands"


@dataclass
class SmoothedGesture:
    active: Optional[str] = None
    counter: int = 0


@dataclass
class ControlState:
    mode: str
    hands_data: Optional[List[Dict[str, Any]]] = None
    left_hand_data: Optional[Dict[str, Any]] = None
    right_hand_data: Optional[Dict[str, Any]] = None

    left_gesture_raw: Optional[str] = None
    right_gesture_raw: Optional[str] = None
    two_hand_gesture_raw: Optional[str] = None

    left_gesture: Optional[str] = None
    right_gesture: Optional[str] = None
    two_hand_gesture: Optional[str] = None

    frame_data: Dict[str, Any] = field(default_factory=dict)
    song_data: Dict[str, Any] = field(default_factory=dict)
    debug_lines: List[str] = field(default_factory=list)


@dataclass
class RenderContext:
    frame: Any
    state: ControlState
    visualizer: Any
    gesture_colors: Dict[str, Tuple[int, int, int]]


class BaseMode(ABC):
    name: str = "base"

    def update(self, state: ControlState, controller: "ModeController") -> None:
        pass

    @abstractmethod
    def render(self, ctx: RenderContext) -> Any:
        raise NotImplementedError


class StrangeMode(BaseMode):
    name = "strange"

    def render(self, ctx: RenderContext) -> Any:
        frame = ctx.frame
        state = ctx.state
        visualizer = ctx.visualizer

        left_hand_data = state.left_hand_data
        right_hand_data = state.right_hand_data
        frame_data = state.frame_data

        for hand_data in [left_hand_data, right_hand_data]:
            if hand_data:
                visualizer.config["line_color"] = (180, 180, 180)
                visualizer.config["landmark_circle_color"] = (200, 200, 200)
                frame = visualizer.draw_hand_landmarks(frame, hand_data)

        if left_hand_data and "left" in frame_data:
            left_dist = frame_data["left"]
            frame = visualizer.draw_distance_line(
                frame,
                left_dist["index_pos"],
                left_dist["thumb_pos"],
                label=f"PINCH: {left_dist['index_to_thumb_normalized']:.2f}",
                color=(255, 120, 0),
            )

        if right_hand_data and "right" in frame_data:
            right_dist = frame_data["right"]
            frame = visualizer.draw_distance_line(
                frame,
                right_dist["index_pos"],
                right_dist["thumb_pos"],
                label=f"PINCH: {right_dist['index_to_thumb_normalized']:.2f}",
                color=(255, 120, 0),
            )

        if left_hand_data and right_hand_data:
            if "left_index_pos" in frame_data and "right_index_pos" in frame_data:
                frame = visualizer.draw_distance_line(
                    frame,
                    frame_data["left_index_pos"],
                    frame_data["right_index_pos"],
                    label=f"INDEX-INDEX: {frame_data['index_to_index_normalized']:.2f}",
                    color=(0, 165, 255),
                )
            if "left_thumb_pos" in frame_data and "right_thumb_pos" in frame_data:
                frame = visualizer.draw_distance_line(
                    frame,
                    frame_data["left_thumb_pos"],
                    frame_data["right_thumb_pos"],
                    label=f"THUMB-THUMB: {frame_data['thumb_to_thumb_normalized']:.2f}",
                    color=(255, 50, 200),
                )

        return frame


class CharmMode(BaseMode):
    name = "charm"

    def render(self, ctx: RenderContext) -> Any:
        frame = ctx.frame
        state = ctx.state
        visualizer = ctx.visualizer
        colors = ctx.gesture_colors

        h, w = frame.shape[:2]
        left_hand_data = state.left_hand_data
        right_hand_data = state.right_hand_data

        for hand_data in [left_hand_data, right_hand_data]:
            if hand_data:
                visualizer.config["line_color"] = (240, 50, 180)
                visualizer.config["landmark_circle_color"] = (0, 215, 255)
                frame = visualizer.draw_hand_landmarks(frame, hand_data)

        if left_hand_data:
            left_count = state.song_data.get("left_finger_count", 0)
            frame = visualizer.draw_finger_count(frame, left_hand_data, left_count, label="")

        if right_hand_data:
            right_count = state.song_data.get("right_finger_count", 0)
            frame = visualizer.draw_finger_count(frame, right_hand_data, right_count, label="")

        if state.two_hand_gesture:
            color = colors.get(state.two_hand_gesture, (255, 255, 255))
            frame = visualizer.draw_gesture_tag(
                frame,
                state.two_hand_gesture.upper(),
                (w // 2, h // 2 - 40),
                color,
                is_centered=True,
            )
            return frame

        if left_hand_data and state.left_gesture:
            wrist_pos = left_hand_data["landmarks_list"][0]
            cx = int(wrist_pos[0] * w)
            cy = int(wrist_pos[1] * h) - 120
            color = colors.get(state.left_gesture, (255, 255, 255))
            frame = visualizer.draw_gesture_tag(
                frame,
                state.left_gesture.upper(),
                (cx, cy),
                color,
                is_centered=False,
            )

        if right_hand_data and state.right_gesture:
            wrist_pos = right_hand_data["landmarks_list"][0]
            cx = int(wrist_pos[0] * w)
            cy = int(wrist_pos[1] * h) - 120
            color = colors.get(state.right_gesture, (255, 255, 255))
            frame = visualizer.draw_gesture_tag(
                frame,
                state.right_gesture.upper(),
                (cx, cy),
                color,
                is_centered=False,
            )

        return frame


class BeautyMode(BaseMode):
    name = "beauty"

    def render(self, ctx: RenderContext) -> Any:
        frame = ctx.frame
        state = ctx.state
        visualizer = ctx.visualizer
        colors = ctx.gesture_colors

        h, w = frame.shape[:2]
        left_hand_data = state.left_hand_data
        right_hand_data = state.right_hand_data
        frame_data = state.frame_data

        if left_hand_data:
            visualizer.config["line_color"] = (100, 200, 50)
            visualizer.config["landmark_circle_color"] = (220, 220, 220)
            frame = visualizer.draw_hand_landmarks(frame, left_hand_data)

            left_count = state.song_data.get("left_finger_count", 0)
            frame = visualizer.draw_finger_count(frame, left_hand_data, left_count)

            if state.left_gesture:
                wrist_pos = left_hand_data["landmarks_list"][0]
                cx = int(wrist_pos[0] * w)
                cy = int(wrist_pos[1] * h) - 120
                color = colors.get(state.left_gesture, (255, 255, 255))
                frame = visualizer.draw_gesture_tag(
                    frame,
                    state.left_gesture.upper(),
                    (cx, cy),
                    color,
                    is_centered=False,
                )

        if right_hand_data and "right" in frame_data:
            right_dist = frame_data["right"]
            frame = visualizer.draw_distance_line(
                frame,
                right_dist["index_pos"],
                right_dist["thumb_pos"],
                label=f"PINCH: {right_dist['index_to_thumb_normalized']:.2f}",
                color=(255, 50, 180),
            )

        return frame


class SpinMode(BaseMode):
    name = "spin"

    def render(self, ctx: RenderContext) -> Any:
        frame = ctx.frame
        state = ctx.state
        visualizer = ctx.visualizer

        for hand_data in [state.left_hand_data, state.right_hand_data]:
            if hand_data:
                visualizer.config["line_color"] = (255, 180, 0)
                visualizer.config["landmark_circle_color"] = (255, 255, 255)
                frame = visualizer.draw_hand_landmarks(frame, hand_data)

        return frame


class ModeController:
    def __init__(self, visualizer: Any, tolerance_frames: int = 8) -> None:
        self.visualizer = visualizer
        self.hand_tracker = HandTracker()
        self.distance_calculator = DistanceCalculator()

        self.tolerance_frames = tolerance_frames

        self.gesture_colors: Dict[str, Tuple[int, int, int]] = {
            "Heart": (50, 50, 255),
            "Circle": (255, 180, 0),
            "Fist": (100, 100, 100),
            "Point": (255, 180, 120),
            "Victory": (50, 205, 50),
            "Thumbs Up": (0, 215, 255),
            "Thumbs Down": (50, 50, 200),
            "Cool Sign": (255, 50, 200),
            "Open Hand": (220, 220, 220),
            "Rock Sign": (180, 100, 255),
            "Three Fingers": (160, 220, 120),
            "Four Fingers": (220, 220, 120),
            "Mini Heart": (0, 0, 255),
            "Looser": (255, 0, 0),
            "Crossed Fingers": (100, 200, 200),
            "Middle Finger": (50, 50, 50),
            "Spock": (144,107,0),
            "Stop": (0,140,255),
            "Call Me": (0, 180, 180),
            "Fancy": (180, 0, 180),
            "Gun": (10,10, 10)
        }

        self.modes: Dict[str, BaseMode] = {
            "strange": StrangeMode(),
            "beauty": BeautyMode(),
            "charm": CharmMode(),
            "spin": SpinMode(),
        }
        self.active_mode: BaseMode = self.modes["strange"]

        self.left_smoothed = SmoothedGesture()
        self.right_smoothed = SmoothedGesture()
        self.two_smoothed = SmoothedGesture()

        self.state = ControlState(mode=self.active_mode.name)

        self.key_to_mode: Dict[int, str] = {
            ord("!"): "strange",
            ord('"'): "beauty",
            ord("#"): "charm",
            ord("$"): "spin",
        }

    @property
    def name(self) -> str:
        return self.active_mode.name

    def close(self) -> None:
        self.hand_tracker.close()

    def switch(self, mode_name: str) -> None:
        if mode_name in self.modes:
            self.active_mode = self.modes[mode_name]
            self.state.mode = self.active_mode.name

    def get_help_text(self) -> str:
        lines = [
            "Starting Gejay hand-control system.",
            "",
            "Controls:",
            "  q / Esc  -> Quit",
            "  v        -> Toggle visualization window",
            "  r        -> Reload current song module",
            "  !        -> Strange mode",
            '  "        -> Beauty mode',
            "  #        -> Charm mode",
            "  $        -> Spin mode",
            "",
            "Modes:",
            "  Strange  -> Skeletons + distance lines only",
            "  Beauty   -> Left-hand counting HUD + right-hand pinch line",
            "  Charm    -> Skeletons + counting + gesture tags",
            "  Spin     -> Placeholder experimental mode",
        ]
        return "\n".join(lines)

    def get_required_tracking(self) -> Dict[str, bool]:
        return {
            "hands": True,
            "single_hand_gestures": True,
            "two_hand_gestures": True,
            "distances": True,
        }

    def detect(self, frame: Any) -> Optional[List[Dict[str, Any]]]:
        hands_data, _ = self.hand_tracker.detect_hands(frame)
        return hands_data

    def update(self, frame: Any) -> None:
        hands_data = self.detect(frame)

        state = ControlState(mode=self.active_mode.name, hands_data=hands_data)

        left_hand_data: Optional[Dict[str, Any]] = None
        right_hand_data: Optional[Dict[str, Any]] = None
        frame_data: Dict[str, Any] = {}

        if hands_data:
            for hand_data in hands_data:
                handedness = hand_data["handedness"]
                distances = self.distance_calculator.get_hand_distances(hand_data)

                if handedness == "Left":
                    left_hand_data = hand_data
                    frame_data["left"] = distances
                else:
                    right_hand_data = hand_data
                    frame_data["right"] = distances

            if left_hand_data and right_hand_data:
                frame_data.update(
                    self.distance_calculator.get_index_to_index_distance(
                        left_hand_data, right_hand_data
                    )
                )
                frame_data.update(
                    self.distance_calculator.get_thumb_to_thumb_distance(
                        left_hand_data, right_hand_data
                    )
                )

        state.left_hand_data = left_hand_data
        state.right_hand_data = right_hand_data
        state.frame_data = frame_data

        raw_left = None
        raw_right = None
        raw_two = None

        if left_hand_data:
            raw_left = self.hand_tracker.get_active_gesture(left_hand_data)
            if raw_left not in self.gesture_colors:
                raw_left = None

        if right_hand_data:
            raw_right = self.hand_tracker.get_active_gesture(right_hand_data)
            if raw_right not in self.gesture_colors:
                raw_right = None

        if left_hand_data and right_hand_data:
            raw_two = self.hand_tracker.detect_two_hand_gesture(left_hand_data, right_hand_data)
            if raw_two not in self.gesture_colors:
                raw_two = None

        state.left_gesture_raw = raw_left
        state.right_gesture_raw = raw_right
        state.two_hand_gesture_raw = raw_two

        state.left_gesture = self._smooth(raw_left, self.left_smoothed)
        state.right_gesture = self._smooth(raw_right, self.right_smoothed)
        state.two_hand_gesture = self._smooth(raw_two, self.two_smoothed)

        state.song_data = self._build_song_data(state)
        state.debug_lines = self._build_debug_lines(state)

        self.active_mode.update(state, self)
        self.state = state

    def render(self, frame: Any) -> Any:
        ctx = RenderContext(
            frame=frame,
            state=self.state,
            visualizer=self.visualizer,
            gesture_colors=self.gesture_colors,
        )
        frame = self.active_mode.render(ctx)
        frame = self._draw_hud(frame)
        return frame

    def get_song_data(self) -> Dict[str, Any]:
        return self.state.song_data

    def handle_key(self, key: int) -> bool:
        if key in (ord("q"), 27):
            return True

        if key in self.key_to_mode:
            mode_name = self.key_to_mode[key]
            self.switch(mode_name)
            print(f"Switched to {mode_name.capitalize()} Mode")
            return False

        return False

    def _smooth(self, raw_gesture: Optional[str], smoothed: SmoothedGesture) -> Optional[str]:
        if raw_gesture is not None:
            if raw_gesture == smoothed.active:
                smoothed.counter = self.tolerance_frames
            else:
                if smoothed.active is None:
                    smoothed.active = raw_gesture
                    smoothed.counter = self.tolerance_frames
                else:
                    smoothed.counter -= 1
                    if smoothed.counter <= 0:
                        smoothed.active = raw_gesture
                        smoothed.counter = self.tolerance_frames
        else:
            if smoothed.counter > 0:
                smoothed.counter -= 1
                if smoothed.counter == 0:
                    smoothed.active = None
            else:
                smoothed.active = None

        return smoothed.active

    def _build_song_data(self, state: ControlState) -> Dict[str, Any]:
        song_data: Dict[str, Any] = {
            "mode": state.mode,
            "left_gesture": state.left_gesture,
            "right_gesture": state.right_gesture,
            "two_hand_gesture": state.two_hand_gesture,
            "left_finger_count": 0,
            "right_finger_count": 0,
        }

        if state.left_hand_data:
            song_data["left_finger_count"] = self.hand_tracker.get_finger_count(state.left_hand_data)

        if state.right_hand_data:
            song_data["right_finger_count"] = self.hand_tracker.get_finger_count(state.right_hand_data)

        if "left" in state.frame_data:
            song_data["left_pinch"] = state.frame_data["left"].get("index_to_thumb_normalized")

        if "right" in state.frame_data:
            song_data["right_pinch"] = state.frame_data["right"].get("index_to_thumb_normalized")

        if "index_to_index_normalized" in state.frame_data:
            song_data["index_to_index"] = state.frame_data["index_to_index_normalized"]

        if "thumb_to_thumb_normalized" in state.frame_data:
            song_data["thumb_to_thumb"] = state.frame_data["thumb_to_thumb_normalized"]

        return song_data

    def _build_debug_lines(self, state: ControlState) -> List[str]:
        lines = [
            f"Mode: {state.mode}",
            f"Left Gesture: {state.left_gesture or '-'}",
            f"Right Gesture: {state.right_gesture or '-'}",
            f"Two Hand: {state.two_hand_gesture or '-'}",
        ]

        if "left" in state.frame_data:
            lines.append(
                f"Left Pinch: {state.frame_data['left'].get('index_to_thumb_normalized', 0.0):.2f}"
            )

        if "right" in state.frame_data:
            lines.append(
                f"Right Pinch: {state.frame_data['right'].get('index_to_thumb_normalized', 0.0):.2f}"
            )

        if "index_to_index_normalized" in state.frame_data:
            lines.append(f"Index-Index: {state.frame_data['index_to_index_normalized']:.2f}")

        if "thumb_to_thumb_normalized" in state.frame_data:
            lines.append(f"Thumb-Thumb: {state.frame_data['thumb_to_thumb_normalized']:.2f}")

        return lines

    def _draw_hud(self, frame: Any) -> Any:
        h, w = frame.shape[:2]

        frame = self.visualizer.draw_mode_label(frame, self.state.mode)

        y = 28
        for line in self.state.debug_lines:
            frame = self.visualizer.draw_text(
                frame,
                line,
                (16, y),
                font_scale=0.6,
                color=(230, 230, 230),
                thickness=1,
            )
            y += 24

        if self.state.two_hand_gesture:
            color = self.gesture_colors.get(self.state.two_hand_gesture, (255, 255, 255))
            frame = self.visualizer.draw_gesture_tag(
                frame,
                self.state.two_hand_gesture.upper(),
                (w // 2, h // 2 - 40),
                color,
                is_centered=True,
            )

        return frame