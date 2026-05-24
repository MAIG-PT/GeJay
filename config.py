"""
Configuration for hand distance measurements and normalization.
"""
# Distance measurement ranges (in normalized coordinates, 0-1)
DISTANCE_CONFIG = {
    # Distance between index fingers (both hands)
    # Min: fingers touching (0.0), Max: arms fully stretched apart
    "index_to_index": {
        "min": 0.05,  # Minimum distance when fingers are close
        "max": 0.7,   # Maximum distance when arms are fully stretched
    },
    # Index to thumb distance for each hand
    # Min: pinching (thumb and index together), Max: fully spread
    "index_to_thumb": {
        "min": 0.01,  # Minimum distance (pinch)
        "max": 0.25,   # Maximum distance (spread)
    },
    # Distance between thumbs (both hands)
    # Min: thumbs touching, Max: arms/hands spread apart
    "thumb_to_thumb": {
        "min": 0.02,
        "max": 0.7,
    },
}
# Visualization settings
VISUALIZATION_CONFIG = {
    "show_landmarks": True,
    "show_distances": True,
    "show_normalized_values": True,
    "landmark_circle_radius": 5,
    "landmark_circle_color": (0, 255, 0),
    "line_color": (0, 128, 255),
    "line_thickness": 2,
    "text_color": (255, 255, 255),
    "text_font_scale": 0.6,
}
# Camera settings
CAMERA_CONFIG = {
    "camera_id": 0,  # Default camera
    "frame_width": 640,
    "frame_height": 480,
    "fps": 30,
}
# MediaPipe settings


MEDIAPIPE_CONFIG = {
    "backend": "auto",  # "solutions" (legacy only for Mediapipe 10.9), "tasks" (for more recent versions), or "auto"

    # Shared
    "max_num_hands": 2,
    "min_hand_detection_confidence": 0.7,
    "min_tracking_confidence": 0.5,

    "static_image_mode": False,
    "max_num_hands": 2,
    "model_complexity": 1,  # 0 = lite, 1 = full

    "min_tracking_confidence": 0.5,
    "model_asset_path": "hand_landmarker.task",
}