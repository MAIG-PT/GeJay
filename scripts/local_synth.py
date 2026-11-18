"""Inspect audio devices, play a test tone, or render a camera-free WAV demo."""

import argparse
import math
import sys
import time
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.synth_config import SYNTH_CONFIG
from synth.controls import DISTANCE_KEYS, GestureMapper


def render_demo(path: str, seconds: float, sample_rate: int, block_size: int):
    import numpy as np
    from synth.engine import SynthEngine

    engine = SynthEngine(sample_rate=sample_rate, block_size=block_size)
    mapper = GestureMapper()
    total = round(seconds * sample_rate)
    with wave.open(str(path), "wb") as output:
        output.setnchannels(2)
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        for start in range(0, total, block_size):
            position = start / total
            # Sweep pitch/brightness, then close volume for a fade at the end.
            distances = dict(zip(DISTANCE_KEYS, (
                position, 0.2 + 0.8 * position, 0.6,
                min(1.0, max(0.0, (1.0 - position) * 10.0)),
            )))
            now = start / sample_rate
            engine.update(mapper.map(distances), now=now)
            audio = engine.render(min(block_size, total - start), now=now)
            pcm = (np.repeat(audio[:, None], 2, axis=1) * 32767).astype("<i2")
            output.writeframes(pcm.tobytes())
    print(f"Rendered {seconds:g}s stereo demo to {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--list-devices", action="store_true")
    actions.add_argument("--test-tone", action="store_true", help="Play C4 without a camera")
    actions.add_argument("--render-demo", metavar="WAV", help="Render offline; no audio device needed")
    parser.add_argument("--seconds", type=float, default=4.0)
    parser.add_argument("--audio-device", help="Output device index or name")
    parser.add_argument("--sample-rate", type=int, default=SYNTH_CONFIG["sample_rate"])
    parser.add_argument("--block-size", type=int, default=SYNTH_CONFIG["block_size"])
    args = parser.parse_args()
    if not math.isfinite(args.seconds) or args.seconds <= 0:
        parser.error("--seconds must be positive and finite")
    if round(args.seconds * args.sample_rate) < 1:
        parser.error("Duration must contain at least one audio sample")

    try:
        if args.render_demo:
            render_demo(args.render_demo, args.seconds, args.sample_rate, args.block_size)
            return
        import sounddevice as sd
        if args.list_devices:
            print(sd.query_devices())
            return

        from synth.engine import SynthEngine
        mapper = GestureMapper()
        # Default two-octave scale: its midpoint is C4.
        parameters = mapper.map(dict(zip(DISTANCE_KEYS, (0.5, 0.75, 0.35, 0.75))))
        device = args.audio_device
        if device is not None and device.isdigit():
            device = int(device)
        engine = SynthEngine(sample_rate=args.sample_rate, block_size=args.block_size)
        try:
            engine.start(device=device)
            print(f"Playing {parameters.label} for {args.seconds:g}s")
            deadline = time.monotonic() + args.seconds
            while time.monotonic() < deadline:
                engine.update(parameters)
                engine.check_audio()
                time.sleep(1.0 / 30.0)
        finally:
            engine.close()
        print(f"Audio test complete; underruns={engine.underruns}")
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        parser.exit(1, f"Audio error: {exc}\n")


if __name__ == "__main__":
    main()
