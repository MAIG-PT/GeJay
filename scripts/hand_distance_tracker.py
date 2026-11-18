"""
Real-time hand distance tracking from camera feed.

This script captures video from the camera and calculates:
- Distance between index fingers (both hands)
- Distance between thumbs (both hands)
- Index to thumb distance for each hand
All distances are normalized to 0-1 range based on configured min/max values.
"""

import json
import sys
import argparse
import socket
import time
from pathlib import Path
from typing import Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.distances_config import CAMERA_CONFIG, MEDIAPIPE_CONFIG, TCP_CONFIG
from config.synth_config import SYNTH_CONFIG
from synth.controls import SCALES, GestureMapper, extract_distances, note_name
from tracking.camera import LatestFrameCapture
from tracking.performance import PerformanceMonitor


def connect_tcp_client(host: str, port: int) -> Optional[socket.socket]:
    """Create a TCP connection to the receiver."""
    try:
        client = socket.create_connection((host, port), timeout=0.01)
        client.settimeout(0.01)
        return client
    except OSError as exc:
        print(f"TCP connection unavailable at {host}:{port} ({exc})")
        return None


def send_tcp_payload(
    client: Optional[socket.socket], payload: dict
) -> Optional[socket.socket]:
    """Send a JSON line; let the main loop schedule reconnection on failure."""
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
        return None


def main(show_visualization: bool = True, enable_synth: bool = False,
         stream_tcp: bool = True, audio_device=None, scale=None,
         sample_rate=None, block_size=None, verbose: bool = False,
         camera_width=None, camera_height=None, camera_fps=None,
         model_complexity=None, threaded_camera=True, profile=False, duration=None,
         record_output=None, recording_dir="recordings"):
    """
    Main function to run real-time hand distance tracking.

    Args:
        show_visualization: Whether to display the video feed with landmarks
    """
    if record_output is not None and not enable_synth:
        raise ValueError("Camera and synth recording requires the synthesizer")
    import cv2
    from utils import HandTracker, DistanceCalculator, Visualizer

    # Initialize hand tracker, distance calculator, and visualizer
    hand_tracker = HandTracker(model_complexity=model_complexity)
    distance_calculator = DistanceCalculator()
    visualizer = Visualizer()

    # Open camera
    cap = cv2.VideoCapture(CAMERA_CONFIG["camera_id"])

    if not cap.isOpened():
        cap.release()
        hand_tracker.close()
        raise RuntimeError("Could not open camera; check CAMERA_CONFIG['camera_id']")

    # Set camera properties
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*CAMERA_CONFIG["fourcc"]))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, camera_width or CAMERA_CONFIG["frame_width"])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_height or CAMERA_CONFIG["frame_height"])
    cap.set(cv2.CAP_PROP_FPS, camera_fps or CAMERA_CONFIG["fps"])
    cap.set(cv2.CAP_PROP_BUFFERSIZE, CAMERA_CONFIG["buffer_size"])
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    encoding = "".join(chr((fourcc >> (8 * n)) & 255) for n in range(4)).strip("\0")

    print("Starting hand distance tracker...")
    print(f"Visualization: {'ON' if show_visualization else 'OFF'}")
    print(f"Camera reports: {actual_width}x{actual_height}, {actual_fps:g} FPS, {encoding}")
    if camera_fps is not None and actual_fps < camera_fps:
        print(f"Requested {camera_fps:g} FPS; this mode reports {actual_fps:g} FPS instead.")
    if stream_tcp:
        print(f"TCP stream target: {TCP_CONFIG['host']}:{TCP_CONFIG['port']}")
    print("Press 'q' to quit, 'v' to hide visualization; Ctrl+C also quits")

    frame_count = 0
    tcp_client = None
    next_tcp_attempt = 0.0
    synth = None
    mapper = GestureMapper(scale) if enable_synth else None
    latest_capture = None
    sequence = 0
    run_started = None
    metrics = None
    recording = None
    auto_record_pending = record_output is not None

    try:
        if threaded_camera:
            latest_capture = LatestFrameCapture(cap)
            latest_capture.start()
        if enable_synth:
            try:
                from synth.engine import SynthEngine
                synth = SynthEngine(sample_rate=sample_rate, block_size=block_size)
                synth.start(device=audio_device)
            except Exception as exc:
                raise RuntimeError(
                    f"Cannot start local synth: {exc}. Install requirements-synth.txt "
                    "and run python scripts/local_synth.py --list-devices."
                ) from exc
            print(f"Local synth: {mapper.scale}, "
                  f"{note_name(mapper.notes[0])}-{note_name(mapper.notes[-1])}, "
                  f"{synth.sample_rate} Hz")
            print("Index spacing=pitch; thumb spacing=brightness; "
                  "left pinch=reverb; right pinch=volume")
            print("Show both hands and open the right pinch to play.")
            from recording.controls import RecordingControls
            recording = RecordingControls(
                synth, (actual_width, actual_height),
                actual_fps if actual_fps > 0 else CAMERA_CONFIG["fps"], recording_dir,
            )
            if show_visualization:
                cv2.namedWindow("Hand Distance Tracker")
                cv2.setMouseCallback("Hand Distance Tracker", recording.on_mouse)
                print("Click Record or press 'r' to record camera video with synth audio.")

        run_started = time.monotonic()
        metrics = PerformanceMonitor(now=run_started,
                                     captured=latest_capture.sequence if latest_capture else 0)
        while duration is None or time.monotonic() - run_started < duration:
            if latest_capture is not None:
                sample = latest_capture.read(after_sequence=sequence)
                if sample is None:
                    break
                skipped = sample.sequence - sequence - 1
                sequence = sample.sequence
                frame, captured_at = sample.image, sample.captured_at
            else:
                ret, frame = cap.read()
                captured_at = time.monotonic()
                if not ret:
                    break
                sequence += 1
                skipped = 0

            frame_count += 1

            # Flip frame horizontally for selfie view
            frame = cv2.flip(frame, 1)
            if recording is not None:
                if auto_record_pending:
                    recording.start(record_output or None)
                    auto_record_pending = False
                recording.update()
                if recording.active:
                    # Save the clean mirrored camera image, before preview overlays.
                    recording.session.push_video(frame, captured_at)

            # Detect hands
            inference_started = time.monotonic()
            hands_data, _ = hand_tracker.detect_hands(frame)
            inference_ms = (time.monotonic() - inference_started) * 1000.0

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

            individual_distances = extract_distances(frame_data)
            if synth is not None:
                parameters = mapper.map(individual_distances)
                synth.update(parameters, now=captured_at)
                synth.check_audio()
            control_at = time.monotonic()
            captured = latest_capture.sequence if latest_capture is not None else sequence
            report = metrics.record(captured, skipped, inference_ms,
                                    (control_at - captured_at) * 1000.0, now=control_at)
            if report is not None and profile:
                print(f"{report} | hands {len(hands_data or [])}")

            # Publish audio controls before spending time drawing the preview.
            if show_visualization:
                for hand_data in hands_data or []:
                    frame = visualizer.draw_hand_landmarks(frame, hand_data)
                    side = "left" if hand_data["handedness"] == "Left" else "right"
                    distances = frame_data[side]
                    frame = visualizer.draw_distance_line(
                        frame, distances["index_pos"], distances["thumb_pos"],
                        label=f"{distances['index_to_thumb_normalized']:.2f}",
                        color=(0, 255, 127),
                    )
                if "index_to_index_normalized" in frame_data:
                    frame = visualizer.draw_distance_line(
                        frame, frame_data["left_index_pos"], frame_data["right_index_pos"],
                        label=f"{frame_data['index_to_index_normalized']:.2f}",
                        color=(255, 255, 0),
                    )
                    frame = visualizer.draw_distance_line(
                        frame, frame_data["left_thumb_pos"], frame_data["right_thumb_pos"],
                        label=f"{frame_data['thumb_to_thumb_normalized']:.2f}",
                        color=(255, 165, 0),
                    )
                frame = visualizer.draw_text_info(frame, frame_data)
                if synth is not None:
                    label = (parameters.label if parameters.active
                             else "Synth muted: show both hands")
                    cv2.putText(frame, label, (10, frame.shape[0] - 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 127), 1)
                preview_stats = metrics.label.split(" | inference")[0]
                cv2.putText(frame, preview_stats, (10, frame.shape[0] - 45),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                if recording is not None:
                    recording.draw(frame)

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
                **individual_distances,
                "interhand_average": interhand_average,
                "thumb_index_average": thumb_index_average,
            }
            if stream_tcp and (tcp_client is not None or time.monotonic() >= next_tcp_attempt):
                tcp_client = send_tcp_payload(tcp_client, tcp_payload)
                if tcp_client is None:
                    next_tcp_attempt = time.monotonic() + 1.0

            # Print current distances to console
            if hands_data and verbose:
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
            key = (cv2.waitKey(1) & 0xFF) if show_visualization else -1
            if key == ord("q"):
                break
            elif key == ord("r") and recording is not None:
                recording.request_toggle()
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
        try:
            if synth is not None:
                synth.close()
                if synth.underruns:
                    print(f"Audio underruns: {synth.underruns}; "
                          "try --block-size 512 or 1024 if you hear crackling.")
        finally:
            if latest_capture is not None:
                latest_capture.close()
            else:
                cap.release()
            if tcp_client is not None:
                try:
                    tcp_client.close()
                except OSError:
                    pass
            if show_visualization:
                cv2.destroyAllWindows()
            hand_tracker.close()
            if recording is not None:
                recording.close()
        print("Hand distance tracker closed")
        if profile and run_started is not None:
            elapsed = time.monotonic() - run_started
            print(f"Processed {frame_count} frames in {elapsed:.2f}s "
                  "(including shutdown)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Real-time hand distance tracking from camera"
    )
    parser.add_argument(
        "--no-viz",
        action="store_true",
        help="Disable visualization (headless mode)",
    )
    parser.add_argument("--synth", action="store_true", help="Play the local synthesizer")
    parser.add_argument("--audio-device", help="Output device index or name")
    parser.add_argument("--scale", choices=[*SCALES, "continuous"],
                        default=SYNTH_CONFIG["scale"])
    parser.add_argument("--sample-rate", type=int, default=SYNTH_CONFIG["sample_rate"])
    parser.add_argument("--block-size", type=int, default=SYNTH_CONFIG["block_size"])
    parser.add_argument("--verbose", action="store_true", help="Print measurements for every tracked frame")
    parser.add_argument("--width", type=int, default=CAMERA_CONFIG["frame_width"])
    parser.add_argument("--height", type=int, default=CAMERA_CONFIG["frame_height"])
    parser.add_argument("--fps", type=float, default=CAMERA_CONFIG["fps"],
                        help="Requested camera FPS; actual rate depends on hardware")
    parser.add_argument("--model-complexity", type=int, choices=[0, 1],
                        default=MEDIAPIPE_CONFIG["model_complexity"])
    parser.add_argument("--sync-camera", action="store_true",
                        help="Read frames synchronously for diagnostics")
    parser.add_argument("--profile", action="store_true", help="Report measured FPS and timing each second")
    parser.add_argument("--duration", type=float, help="Stop after this many seconds of tracking")
    parser.add_argument("--record", nargs="?", const="", metavar="MP4",
                        help="Start recording camera + synth audio immediately; implies --synth")
    parser.add_argument("--recording-dir", default="recordings",
                        help="Directory for takes started with the Record button (default: recordings)")
    tcp_options = parser.add_mutually_exclusive_group()
    tcp_options.add_argument("--tcp", action="store_true", help="Enable TCP alongside the synth")
    tcp_options.add_argument("--no-tcp", action="store_true", help="Disable TCP streaming")

    args = parser.parse_args()
    import math
    if any(not math.isfinite(value) or value <= 0
           for value in (args.width, args.height, args.fps)):
        parser.error("Camera width, height, and FPS must be positive and finite")
    if args.duration is not None and (not math.isfinite(args.duration) or args.duration <= 0):
        parser.error("--duration must be positive and finite")
    if args.record and Path(args.record).suffix.lower() != ".mp4":
        parser.error("--record filename must end in .mp4")
    use_synth = args.synth or args.record is not None
    device = args.audio_device
    if device is not None and device.isdigit():
        device = int(device)
    try:
        main(show_visualization=not args.no_viz, enable_synth=use_synth,
             stream_tcp=not args.no_tcp and (args.tcp or not use_synth),
             audio_device=device, scale=args.scale, sample_rate=args.sample_rate,
             block_size=args.block_size, verbose=args.verbose,
             camera_width=args.width, camera_height=args.height, camera_fps=args.fps,
             model_complexity=args.model_complexity, threaded_camera=not args.sync_camera,
             profile=args.profile, duration=args.duration,
             record_output=args.record, recording_dir=args.recording_dir)
    except (ImportError, OSError, RuntimeError, ValueError) as exc:
        parser.exit(1, f"Error: {exc}\n")
