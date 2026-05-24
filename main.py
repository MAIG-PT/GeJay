#!/usr/bin/env python3
"""
Main orchestrator script for the hand-controlled FoxDot melody system.
Integrates live camera frames, hand landmark tracking via MediaPipe,
and distance calculations to update dynamic musical parameters in real-time.
"""

import sys
import time
import argparse

from pathlib import Path
from typing import Optional, List, Dict, Any
import cv2
import numpy as _numpy

from vision_utils import Visualizer
from mode_controller import ModeController
from config import CAMERA_CONFIG

# Global State Variables
running: bool = True
last_update_time: float = 0.0
update_interval: float = 0.25

window_name: str = "Gejay - Control Music using your hands"

def main(show_visualization: bool = True, 
         song_name: str = "songs.derpy_naive") -> None:
    visualizer = Visualizer()
    controller = ModeController(visualizer=visualizer)
    #song_controller = SongController(song_name=song_name)

    cap = cv2.VideoCapture(CAMERA_CONFIG["camera_id"], cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_CONFIG["frame_width"])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_CONFIG["frame_height"])
    cap.set(cv2.CAP_PROP_FPS, CAMERA_CONFIG["fps"])
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc("M", "J", "P", "G"))

    print(controller.get_help_text())

    if show_visualization:
        visualizer.create_window(
            window_name,
            CAMERA_CONFIG["frame_width"],
            CAMERA_CONFIG["frame_height"],
        )

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)

            controller.update(frame)
            
            song_data = controller.get_song_data()
            #song_controller.update(song_data)

            if show_visualization:
                out = controller.render(frame.copy())
                visualizer.show_frame(window_name, out)


            if show_visualization:
                out_frame = controller.render(frame.copy())
                visualizer.show_frame(window_name, out_frame)

            key = cv2.waitKey(1) & 0xFF
            if controller.handle_key(key):
                break

            if show_visualization and not visualizer.is_window_open(window_name):
                break

    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("System cleanly shut down.")

main()