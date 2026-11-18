# Hand Distance Tracker

Real-time hand distance tracking using MediaPipe and OpenCV. This project calculates hand distances in real-time from a camera feed:

- **Index Finger to Thumb Distance**: Measured for each hand (left and right)
- **Index Finger to Index Finger Distance**: Distance between index fingers of both hands
- **Thumb to Thumb Distance**: Distance between thumbs of both hands
- **Normalized Values**: All distances are normalized to 0-1 range based on configurable min/max values
- **Local Synthesizer**: Play a monophonic synthesizer directly from all four distances

## Play the local synthesizer

Use the same Python environment as your working camera tracker (Python 3.10 or
3.11 for the existing pinned camera dependencies). Install the audio extra:

```bash
conda activate gejay
python -m pip install -r requirements-synth.txt
python scripts/hand_distance_tracker.py --synth
```

The `gejay` environment on this machine already includes `sounddevice`. No DAW,
MIDI routing, synth server, or TCP receiver is needed. The path is:

```text
Camera → MediaPipe → four normalized distances → gesture mapper
                                                 ↓
                                local audio callback → speakers
```

| Measurement | Musical control | Default behavior |
|---|---|---|
| Index to index | Pitch | Apart → higher notes, C3–C5 in C major pentatonic |
| Thumb to thumb | Brightness | Apart → higher spectral low-pass cutoff, 200–8000 Hz |
| Left index to thumb | Reverb | Open → more reverb, up to 55% wet |
| Right index to thumb | Volume | Open → louder; closed pinch stops the voice |

Show both hands and open your right pinch to start playing. The camera window
shows the target note and musical controls. A right pinch below 0.08 is treated
as closed; reverb may briefly ring after releasing a note. Missing either hand
fades the entire output, including reverb. A stalled camera also fades the sound
after 300 ms without a control update. Press **q** or **Ctrl+C** to stop.

The instrument uses a harmonic oscillator, a spectral low-pass envelope, and
four feedback comb delays followed by two all-pass delays for reverb. Controls
use a 12 ms smoothing time constant, and scale boundaries have hysteresis to reduce note
chatter. Audio runs independently of camera inference using a
[`sounddevice.OutputStream`](https://python-sounddevice.readthedocs.io/en/latest/api/streams.html).
It defaults to 48 kHz with 256-sample audio blocks; total latency also depends on
the camera, inference speed, and output device. Both speaker channels receive
the same mono instrument.

Try another scale or continuous pitch:

```bash
python scripts/hand_distance_tracker.py --synth --scale minor
python scripts/hand_distance_tracker.py --synth --scale continuous
```

Available scales: `pentatonic`, `major`, `minor`, `chromatic`, and `continuous`.
Edit `config/synth_config.py` for the root note, pitch range, master gain, filter
range, reverb limit, and smoothing. Adjust `config/distances_config.py` to
calibrate the four distances to your comfortable hand movements. The four
measurements are mapped separately, although moving one finger can physically
affect more than one measurement.

### Record camera video with synth audio

```bash
python scripts/hand_distance_tracker.py --synth
```

Click **Record** at the top right of the preview, or press **r**. Click
**Stop recording** or press **r** again to finish the take. A red `REC` timer
shows when recording is active. The button shows `Saving...` while the MP4 is
encoded; tracking and live audio continue. Once it finishes, you can start
another take. The terminal prints the saved file's full path.

Each take saves to `recordings/performance_<timestamp>.mp4` by default, with a
clean mirrored camera image and the synthesizer's stereo output, including
filtering, reverb, and volume changes. Tracking landmarks, text, and the Record
button appear only in the live preview. Audio comes directly from the synth,
so the camera microphone and other applications are not included. The output
uses H.264 video and AAC audio. Generated takes are ignored by Git.

To start recording immediately, optionally supply a filename:

```bash
python scripts/hand_distance_tracker.py --synth --record
python scripts/hand_distance_tracker.py --record recordings/my_take.mp4
python scripts/hand_distance_tracker.py --record /tmp/my_take.mp4 --no-viz --duration 10
```

`--record` automatically enables the synth. `--recording-dir PATH` changes the
directory used by the button and automatic filenames. Existing files are never
overwritten. Press **q** or **Ctrl+C** to quit and finish an active take; leave
the process running until it prints `Saved recording`.

Recording requires **FFmpeg** with H.264/AAC support, already installed on this
machine. On Debian/Ubuntu, install it with `sudo apt install ffmpeg` if missing.
Camera frames and synth buffers go through bounded queues to a separate disk
writer. Video repeats the previous frame over gaps to preserve real elapsed
time rather than speeding up a slow camera feed. Audio uses scheduled playback
timestamps and its sample clock; sensor/driver timestamp limitations still
apply. Final MP4 encoding also happens in the background. If encoding fails,
the terminal identifies the error and the intermediate camera/audio files kept
beside the destination for recovery.

### Fast finger movements

The tracker defaults to **640×480 MJPEG at a requested 30 FPS** and MediaPipe's
lite model. Camera capture runs in a separate thread and retains only the newest
frame, so slow inference or drawing cannot accumulate a queue of old gestures.
Musical controls are published before drawing the preview. Per-frame console
logging is off unless `--verbose` is supplied.

```bash
python scripts/hand_distance_tracker.py --synth --profile
python scripts/hand_distance_tracker.py --synth --no-viz --profile
```

The preview displays measured camera and tracking FPS. `--profile` also reports
mean inference time, skipped frames, and mean time from the host receiving a frame
to publishing its controls. That last measurement excludes sensor exposure, USB
and driver buffering, audio buffering, and control smoothing; it is not total
motion-to-sound latency. Skipping frames when overloaded is intentional: the
tracker processes the latest available hand position rather than catching up.

On this machine, the previous 1280×720 configuration used YUYV and negotiated only
10 FPS. Selecting MJPEG negotiated 30 FPS. Requesting 60 FPS at 640×480 also
negotiated 30 FPS, so setting `--fps 60` alone does not make this tested mode run
at 60 FPS. Startup prints the mode actually reported by the driver; `--profile`
measures delivered frames.

In the live camera-and-synth check, tracking reached about 30 FPS after startup,
including intervals with both hands detected. Two driver buffers performed better
than one on this camera; the capture thread drains them continuously and retains
one latest frame for inference.

For a camera/mode supporting a higher rate, request it explicitly:

```bash
python scripts/hand_distance_tracker.py --synth --fps 60 --profile
```

At 30 FPS, successive frames are about 33 ms apart, so a pinch that closes and
opens entirely between frames can be missed. Good lighting helps reduce motion
blur. Keep both hands visible, and recalibrate distance ranges if the new framing
changes your comfortable gesture range. For more landmark accuracy, try
`--model-complexity 1`; the [MediaPipe model setting](https://chuoling.github.io/mediapipe/solutions/hands.html#model_complexity)
trades latency against accuracy. `--width 1280 --height 720` restores HD capture
while keeping MJPEG. `--sync-camera` disables the capture thread for diagnostics.
Use `--duration 10 --no-viz --profile` for a bounded benchmark.

### Check audio without the camera

```bash
python scripts/local_synth.py --list-devices
python scripts/local_synth.py --test-tone --seconds 3
python scripts/local_synth.py --test-tone --audio-device default
python scripts/local_synth.py --render-demo /tmp/gejay-demo.wav
```

The WAV demo renders offline, so it works without an audio device. To choose an
output for live tracking, pass `--audio-device` with a device name or index from
the list. If you hear crackling, try `--block-size 512` or `1024`; this increases
buffering. If a device does not accept 48 kHz, try `--sample-rate 44100`.
On Debian/Ubuntu, a missing PortAudio library can be installed with
`sudo apt install libportaudio2`.

### TCP compatibility

Synth mode disables TCP by default to keep network waits out of the camera loop.
Use `--synth --tcp` to enable it alongside audio. Tracking without `--synth`
continues to stream TCP by default; `--no-tcp` disables streaming.
The newline-delimited JSON now includes four individual fields:

```text
index_to_index_normalized
thumb_to_thumb_normalized
left_index_to_thumb_normalized
right_index_to_thumb_normalized
```

Missing measurements are `null`; zero is a valid measurement. The existing
`frame`, `interhand_average`, and `thumb_index_average` fields remain available.
TCP connection/send attempts have a short timeout and failed connections are
retried at most once a second. Frame-by-frame logging is off by default;
add `--verbose` if needed.

### Verification

```bash
python -m unittest discover -s tests -v
```

These tests run without a camera or audio device and cover independent control
mapping, scale boundaries, pitch output, brightness, reverb, silence, fading
on lost/stalled tracking, and dropping stale camera frames.
When FFmpeg is available, they also encode and inspect real MP4s to check audio,
video duration, frame repetition, and timing offsets.

## Project Structure

```
├── config/
│   ├── distances_config.py      # Configuration for distances, camera, and visualization
│   └── synth_config.py          # Musical mapping and audio defaults
├── scripts/
│   ├── hand_distance_tracker.py # Main script to run the tracker
│   ├── local_synth.py           # Audio device check, test tone, offline demo
│   └── tcp_distance_receiver.py # Optional TCP monitor
├── synth/                      # Gesture mapper and audio engine
├── tracking/                   # Latest-frame camera capture and FPS measurements
├── recording/                  # Record button, bounded disk writer, and MP4 encoding
├── tests/                      # Camera-free mapping and synthesis checks
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
- **v**: Hide visualization (Ctrl+C still stops tracking)
- **r / Record button**: Start or stop a camera-and-synth recording in synth mode

## Configuration

All configuration is done in `config/distances_config.py`:

### Distance Ranges

Adjust the min/max values for distance normalization:

```python
DISTANCE_CONFIG = {
    "index_to_index": {
        "min": 0.05,  # Minimum distance
        "max": 0.7,   # Maximum distance
    },
    "index_to_thumb": {
        "min": 0.01,  # Minimum pinch distance
        "max": 0.25,  # Maximum spread distance
    },
    "thumb_to_thumb": {
        "min": 0.02,  # Minimum thumb-touch distance
        "max": 0.7,   # Maximum spread distance
    },
}
```

### Camera Settings

Modify camera resolution and FPS:

```python
CAMERA_CONFIG = {
    "camera_id": 0,      # Camera device ID
    "frame_width": 640,
    "frame_height": 480,
    "fps": 30,
    "fourcc": "MJPG",
    "buffer_size": 2,
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
    "model_complexity": 0,  # 0=lite, 1=full
    "min_detection_confidence": 0.7,
    "min_tracking_confidence": 0.5,
}
```

## Output

Use `--verbose` to print normalized distances (0-1 range) to the console:

```
--- Frame 42 ---
Left  Index-Thumb (normalized): 0.1234
Right Index-Thumb (normalized): 0.2567
Index-to-Index (normalized): 0.8901
Thumb-to-Thumb (normalized): 0.7452
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
- **Frame Rate**: Check measured FPS with `--profile`; requesting a rate is not a guarantee the camera supports it

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
