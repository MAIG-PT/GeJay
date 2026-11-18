# GeJay — teaching outline

- **Core idea:** turn hand movements into a musical instrument using a camera and Python.
- **Lesson objective:** understand how computer vision, geometry, interaction design, and audio synthesis work together in a real-time system.
- **Useful prerequisites:** functions, classes, lists, dictionaries, and basic knowledge of coordinates and sound waves.

## 1. Introduce the project

- GeJay detects up to two hands, tracks finger landmarks, and calculates four distances.
- Those distances become musical controls: pitch, brightness, reverb, and volume.
- The instrument is monophonic: it plays one note at a time; both output channels receive the same sound.
- The project supports a tracking preview, TCP measurement streaming, and video recording with synthesizer audio.
- Opening question: “How can we play music without physically touching an instrument?”

## 2. Explain the data flow

- **Camera → image → MediaPipe → hand landmarks → distances → values between 0 and 1 → musical parameters → audio.**
- The camera supplies images; the model identifies reference points called *landmarks*.
- The code calculates geometric features and maps them to synthesizer parameters.
- AI is provided by a pretrained detection and tracking model; this project does not train a new model.
- The mapping from distances to sound uses rules and formulas defined in the code.

## 3. Important libraries and tools

- **OpenCV (`cv2`; `opencv-python`):** open the camera, read images, convert colors, and draw the interface. Learn that OpenCV uses BGR while MediaPipe receives RGB.
- **MediaPipe (`mediapipe`):** detect hands and estimate 21 landmarks per hand. Learn to access landmarks and left/right hand labels.
- **NumPy (`numpy`):** represent images and signals as arrays and calculate audio blocks. Learn array shapes, numeric types, vectorized operations, and broadcasting.
- **sounddevice:** send audio blocks to the output device through a callback, using PortAudio. Learn about sample rate, block size, and uninterrupted playback.
- **Python standard library:** `math` for formulas; `dataclasses` for parameters; `threading` and `queue` for concurrency; `time` for measurements; `argparse` for command-line options; `socket` and `json` for communication; `pathlib` and `subprocess` for files and processes.
- **FFmpeg:** an external tool used to encode MP4 recordings with H.264 video and AAC audio; it is not a Python library.
- **`unittest`:** automated tests that can check parts of the system without a camera or audio device.
- **Virtual environments and dependencies:** understand `requirements.txt` and `requirements-synth.txt`. The pinned vision dependencies were prepared for Python 3.10/3.11; installation should follow the project's dependencies.

## 4. Computer vision and geometry

- Each hand has 21 landmarks; landmarks **4** and **8** are the thumb tip and index fingertip.
- The project uses their **normalized 2D `(x, y)` coordinates** to calculate distances.
- Euclidean distance: `d = sqrt((x2 - x1)^2 + (y2 - y1)^2)`.
- The four measurements are index–index, thumb–thumb, left index–thumb, and right index–thumb distances.
- Control normalization: `u = (d - minimum) / (maximum - minimum)`, clamped to `[0, 1]`.
- Example: with minimum `0.01`, maximum `0.25`, and distance `0.13`, the normalized control is `0.5`.
- Distinguish normalized image coordinates from distance normalization for control: these are separate operations.
- These measurements are not centimeters or physical 3D distances; they depend on framing, perspective, and position relative to the camera. Different image-axis scales also affect the metric.
- Calibrate minimum and maximum values in `config/distances_config.py` for the user's comfortable movements.
- Distinguish **zero** from **missing detection**: a closed pinch is valid; `None`/`null` means information is missing.

## 5. From gestures to music

- **Move the index fingers apart:** raise pitch; the default is a C major pentatonic scale from C3 to C5.
- **Move the thumbs apart:** increase brightness by raising the spectral cutoff parameter from 200 to 8000 Hz.
- **Open the left pinch:** increase reverb, up to a 55% wet mix.
- **Open the right pinch:** increase volume; below the normalized threshold of `0.08`, the voice stops being excited and reverb may briefly continue.
- The measurements are handled separately in code, but physical movements can affect several measurements at once.
- Compare continuous pitch control with selecting notes from a scale.
- Explain **hysteresis**: requiring an extra margin before changing notes prevents small tracking fluctuations from causing rapid note switching.
- Explain **smoothing**: gradual parameter transitions reduce sudden changes and clicks; stability and responsiveness involve a tradeoff.
- Distinguish a MIDI note number from sending MIDI messages: here the number is used to calculate frequency, `f = 440 × 2^((note - 69) / 12)`.

## 6. How sound is generated

- The synthesizer adds sinusoidal harmonic components: a form of **additive synthesis**.
- The fundamental determines pitch; harmonics contribute to timbre.
- Brightness is controlled by attenuating harmonics with a spectral low-pass envelope.
- Reverb uses feedback delay lines and all-pass filters.
- The default configuration uses **48,000 samples per second** and **256-sample blocks**.
- A 256-sample block at 48 kHz represents about **5.3 ms of audio**; this is not the system's total latency.
- Introduce the Nyquist frequency and aliasing: the engine suppresses harmonics near the sampling limit.

## 7. Real-time behavior, performance, and robustness

- Camera capture runs in a thread and keeps only the latest image, avoiding a backlog of old gestures.
- Audio is generated independently of image inference; the callback should not wait for networking, disk writes, or camera processing.
- Recording uses bounded queues and background writing/encoding.
- **FPS and latency are different:** FPS measures update frequency; latency measures the delay between a gesture and its audible response.
- At 30 FPS, images are about 33 ms apart; a very fast gesture can occur between captures.
- Requesting 60 FPS does not guarantee 60 FPS: measure the delivered rate and check the camera's supported mode.
- `--profile` measures internal performance, but does not measure the full motion-to-sound latency.
- Losing either hand, or receiving no control update for 300 ms, fades the output, including reverb.
- Discuss tradeoffs between resolution, model accuracy, lighting, FPS, and delay.

## 8. Guide students through the code

- `scripts/hand_distance_tracker.py`: entry point connecting the components.
- `utils/hand_tracker.py`: MediaPipe integration and landmark extraction.
- `utils/distance_calculator.py`: geometry and normalization.
- `synth/controls.py`: mapping distances to musical parameters.
- `synth/engine.py`: audio generation and playback.
- `tracking/camera.py` and `tracking/performance.py`: concurrent capture and performance measurement.
- `recording/`: recording interface, queues, and audio/video synchronization.
- `config/`: adjustable parameters; `tests/`: automated behavior checks.
- Highlight separation of responsibilities: make mathematics and mapping testable without hardware.

## 9. Suggested demonstration

- Run `python scripts/hand_distance_tracker.py --synth`.
- Show both hands and open the right pinch to start hearing sound; demonstrate each control with slow movements.
- Run `python scripts/hand_distance_tracker.py --synth --scale continuous` to compare with the pentatonic scale.
- Remove one hand from view and observe the sound fading out.
- Use `--profile` to relate the experience to performance measurements.
- Press `r` to start/stop recording and `q` to quit; wait for the recording completion message.
- Without a camera or audio output device, generate an offline demonstration: `python scripts/local_synth.py --render-demo /tmp/gejay-demo.wav`.
- Show automated verification: `python -m unittest discover -s tests -v`.

## 10. Exercises and learning outcomes

- **Geometry:** calculate and normalize a distance by hand; confirm the result in code.
- **Calibration:** adjust the limits for two users and explain why the experience changes.
- **Mapping:** swap the gestures controlling volume and reverb while preserving silence when detection is missing.
- **Music and audio:** experiment with scales, pitch range, and smoothing; describe the audible effect.
- **Performance:** compare operation with and without visualization, recording FPS and inference time; explain what those measurements leave out.
- **Testing:** propose cases for minimum/maximum distances, out-of-range values, zero measurements, and missing hands.
- **Extension:** add a control based on another hand landmark and define how to test it.
- By the end, students should be able to explain the complete **image → data → decision → sound** pipeline, identify the libraries involved, and justify calibration, interaction, and performance choices.
