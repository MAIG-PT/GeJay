"""
Real-time hand distance tracking from camera feed.

This script captures video from the camera and calculates:
- Distance between index fingers (both hands)
- Distance between thumbs (both hands)
- Index to thumb distance for each hand
All distances are normalized to 0-1 range based on configured min/max values.
"""

import cv2
import json
import sys
import argparse
import socket
from pathlib import Path
from typing import Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils import HandTracker, DistanceCalculator, Visualizer
from config.distances_config import CAMERA_CONFIG, TCP_CONFIG, VISUALIZATION_CONFIG


def connect_tcp_client(host: str, port: int) -> Optional[socket.socket]:
    """Create a TCP connection to the receiver."""
    try:
        client = socket.create_connection((host, port), timeout=1.0)
        client.settimeout(1.0)
        return client
    except OSError as exc:
        print(f"TCP connection unavailable at {host}:{port} ({exc})")
        return None


def send_tcp_payload(
    client: Optional[socket.socket], payload: dict
) -> Optional[socket.socket]:
    """Send one JSON payload over TCP and reconnect on failure."""
    message = json.dumps(payload) + "\n"

    if client is None:
        client = connect_tcp_client(TCP_CONFIG["host"], TCP_CONFIG["port"])
        if client is None:
            return None

    try:
        client.sendall(message.encode("utf-8"))
        return client
    except OSError:
        try:
            client.close()
        except OSError:
            pass
        return connect_tcp_client(TCP_CONFIG["host"], TCP_CONFIG["port"])


def main(show_visualization: bool = True):
    """
    Main function to run real-time hand distance tracking.

    Args:
        show_visualization: Whether to display the video feed with landmarks
    """
    # Initialize hand tracker, distance calculator, and visualizer
    hand_tracker = HandTracker()
    distance_calculator = DistanceCalculator()
    visualizer = Visualizer()

    # Open camera
    cap = cv2.VideoCapture(CAMERA_CONFIG["camera_id"])

    if not cap.isOpened():
        print("Error: Could not open camera")
        return

    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_CONFIG["frame_width"])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_CONFIG["frame_height"])
    cap.set(cv2.CAP_PROP_FPS, CAMERA_CONFIG["fps"])

    print("Starting hand distance tracker...")
    print(f"Visualization: {'ON' if show_visualization else 'OFF'}")
    print(
        f"TCP stream target: {TCP_CONFIG['host']}:{TCP_CONFIG['port']}"
    )
    print("Press 'q' to quit, 'v' to toggle visualization")

    frame_count = 0
    tcp_client = None

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            # Flip frame horizontally for selfie view
            frame = cv2.flip(frame, 1)

            # Detect hands
            hands_data, rgb_frame = hand_tracker.detect_hands(frame)

            # Initialize distance data
            frame_data = {}

            if hands_data:
                # Process each detected hand
                left_hand_data = None
                right_hand_data = None

                for hand_data in hands_data:
                    handedness = hand_data["handedness"]
                    distances = distance_calculator.get_hand_distances(hand_data)

                    if handedness == "Left":
                        left_hand_data = hand_data
                        frame_data["left"] = distances
                    else:
                        right_hand_data = hand_data
                        frame_data["right"] = distances

                    # Draw landmarks if visualization is on
                    if show_visualization:
                        frame = visualizer.draw_hand_landmarks(frame, hand_data)
                        # Draw index to thumb distance line
                        frame = visualizer.draw_distance_line(
                            frame,
                            distances["index_pos"],
                            distances["thumb_pos"],
                            label=f"{distances['index_to_thumb_normalized']:.2f}",
                            color=(0, 255, 127),
                        )

                # Calculate index-to-index distance if both hands present
                if left_hand_data and right_hand_data:
                    index_data = distance_calculator.get_index_to_index_distance(
                        left_hand_data, right_hand_data
                    )
                    frame_data.update(index_data)

                    thumb_data = distance_calculator.get_thumb_to_thumb_distance(
                        left_hand_data, right_hand_data
                    )
                    frame_data.update(thumb_data)

                    if show_visualization:
                        frame = visualizer.draw_distance_line(
                            frame,
                            index_data["left_index_pos"],
                            index_data["right_index_pos"],
                            label=f"{index_data['index_to_index_normalized']:.2f}",
                            color=(255, 255, 0),
                        )
                        frame = visualizer.draw_distance_line(
                            frame,
                            thumb_data["left_thumb_pos"],
                            thumb_data["right_thumb_pos"],
                            label=f"{thumb_data['thumb_to_thumb_normalized']:.2f}",
                            color=(255, 165, 0),
                        )

                # Draw text info
                if show_visualization:
                    frame = visualizer.draw_text_info(frame, frame_data)

            interhand_average = None
            if "index_to_index_normalized" in frame_data and "thumb_to_thumb_normalized" in frame_data:
                interhand_average = (
                    frame_data["index_to_index_normalized"]
                    + frame_data["thumb_to_thumb_normalized"]
                ) / 2.0

            left_thumb_index = frame_data.get("left", {}).get("index_to_thumb_normalized")
            right_thumb_index = frame_data.get("right", {}).get("index_to_thumb_normalized")

            if left_thumb_index is not None and right_thumb_index is not None:
                thumb_index_average = (left_thumb_index + right_thumb_index) / 2.0
            else:
                thumb_index_average = None

            tcp_payload = {
                "frame": frame_count,
                "interhand_average": interhand_average,
                "thumb_index_average": thumb_index_average,
            }
            tcp_client = send_tcp_payload(tcp_client, tcp_payload)

            # Print current distances to console
            if hands_data:
                print(f"\n--- Frame {frame_count} ---")
                if "left" in frame_data:
                    print(
                        f"Left  Index-Thumb (normalized): {frame_data['left']['index_to_thumb_normalized']:.4f}"
                    )
                if "right" in frame_data:
                    print(
                        f"Right Index-Thumb (normalized): {frame_data['right']['index_to_thumb_normalized']:.4f}"
                    )
                if "index_to_index_normalized" in frame_data:
                    print(
                        f"Index-to-Index (normalized): {frame_data['index_to_index_normalized']:.4f}"
                    )
                if "thumb_to_thumb_normalized" in frame_data:
                    print(
                        f"Thumb-to-Thumb (normalized): {frame_data['thumb_to_thumb_normalized']:.4f}"
                    )
                print(
                    f"Interhand average: {interhand_average:.4f}"
                    if interhand_average is not None
                    else "Interhand average: n/a"
                )
                print(
                    f"Thumb-Index average: {thumb_index_average:.4f}"
                    if thumb_index_average is not None
                    else "Thumb-Index average: n/a"
                )

            # Display frame if visualization is on
            if show_visualization:
                cv2.imshow("Hand Distance Tracker", frame)

            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("v"):
                show_visualization = not show_visualization
                if show_visualization:
                    print("Visualization ON")
                else:
                    print("Visualization OFF")
                    cv2.destroyAllWindows()

    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        # Cleanup
        cap.release()
        if tcp_client is not None:
            try:
                tcp_client.close()
            except OSError:
                pass
        if show_visualization:
            cv2.destroyAllWindows()
        hand_tracker.close()
        print("Hand distance tracker closed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Real-time hand distance tracking from camera"
    )
    parser.add_argument(
        "--no-viz",
        action="store_true",
        help="Disable visualization (headless mode)",
    )

    args = parser.parse_args()
    main(show_visualization=not args.no_viz)
