# Hand Distance Tracker

Real-time hand distance tracking using MediaPipe and OpenCV. This project calculates hand distances in real-time from a camera feed:

- **Index Finger to Thumb Distance**: Measured for each hand (left and right)
- **Index Finger to Index Finger Distance**: Distance between index fingers of both hands
- **Normalized Values**: All distances are normalized to 0-1 range based on configurable min/max values

## Project Structure

```
├── config/
│   └── distances_config.py      # Configuration for distances, camera, and visualization
├── scripts/
│   └── hand_distance_tracker.py # Main script to run the tracker
├── utils/
│   ├── __init__.py
│   ├── hand_tracker.py          # MediaPipe hand detection
│   ├── distance_calculator.py   # Distance calculations and normalization
│   └── visualizer.py            # Visualization utilities
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## Installation

1. **Clone or navigate to the project directory**:
   ```bash
   cd /path/to/GeJay
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Basic Usage

Run the tracker with visualization enabled (default):

```bash
python scripts/hand_distance_tracker.py
```

### Run in Headless Mode

Run without visualization (useful for servers or processing):

```bash
python scripts/hand_distance_tracker.py --no-viz
```

### Controls

- **q**: Quit the application
- **v**: Toggle visualization on/off

## Configuration

All configuration is done in `config/distances_config.py`:

### Distance Ranges

Adjust the min/max values for distance normalization:

```python
DISTANCE_CONFIG = {
    "index_to_index": {
        "min": 0.05,  # Minimum distance
        "max": 1.0,   # Maximum distance
    },
    "index_to_thumb": {
        "min": 0.01,  # Minimum pinch distance
        "max": 0.3,   # Maximum spread distance
    },
}
```

### Camera Settings

Modify camera resolution and FPS:

```python
CAMERA_CONFIG = {
    "camera_id": 0,      # Camera device ID
    "frame_width": 1280,
    "frame_height": 720,
    "fps": 30,
}
```

### Visualization Settings

Customize the appearance of landmarks and text:

```python
VISUALIZATION_CONFIG = {
    "show_landmarks": True,
    "show_distances": True,
    "landmark_circle_radius": 5,
    # ... more settings
}
```

### MediaPipe Settings

Fine-tune hand detection performance:

```python
MEDIAPIPE_CONFIG = {
    "static_image_mode": False,
    "max_num_hands": 2,
    "model_complexity": 1,  # 0=lite, 1=full
    "min_detection_confidence": 0.7,
    "min_tracking_confidence": 0.5,
}
```

## Output

The program outputs normalized distances (0-1 range) to the console:

```
--- Frame 42 ---
Left  Index-Thumb (normalized): 0.1234
Right Index-Thumb (normalized): 0.2567
Index-to-Index (normalized): 0.8901
```

## Normalization

All distances are normalized to a 0-1 range:

$$\text{normalized\_distance} = \frac{\text{raw\_distance} - \text{min}}{\text{max} - \text{min}}$$

- **0.0**: Distance equals the configured minimum
- **1.0**: Distance equals the configured maximum
- Values outside the range are clamped to [0, 1]

## Hand Landmarks

The program uses MediaPipe's hand landmark detection:
- **Landmark 4**: Thumb tip
- **Landmark 8**: Index finger tip
- **Other landmarks**: Other finger tips and joints (0-20)

## Performance Notes

- **Model Complexity**: Set to 1 (full model) for better accuracy; use 0 (lite) for faster processing on slower machines
- **Detection Confidence**: Adjust `min_detection_confidence` to filter out low-confidence detections
- **Frame Rate**: Reduce camera FPS if your machine can't keep up

## Troubleshooting

### Camera not opening
- Ensure your camera is connected and not being used by another application
- Check the `camera_id` in `CAMERA_CONFIG` (try 0, 1, 2, etc.)

### Poor hand detection
- Ensure good lighting conditions
- Move hands closer to the camera
- Adjust `min_detection_confidence` in `MEDIAPIPE_CONFIG`

### Slow performance
- Reduce `CAMERA_CONFIG` resolution
- Set `model_complexity` to 0 in `MEDIAPIPE_CONFIG`
- Run with `--no-viz` flag to disable rendering

## Dependencies

- **OpenCV** (`opencv-python`): Video capture and rendering
- **MediaPipe** (`mediapipe`): Hand detection and landmark tracking
- **NumPy** (`numpy`): Numerical operations

## Future Enhancements

- Save distance data to CSV/JSON files
- Real-time plotting of distance over time
- Hand gesture recognition
- Multi-camera support
- Recording and playback
- Distance alerts/thresholds

## License

[Add your license here]

## Author

[Your name/organization]
