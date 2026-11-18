"""Map the four normalized hand distances to musical controls."""

import math
from dataclasses import dataclass, replace
from typing import Mapping, Optional

from config.synth_config import SYNTH_CONFIG


DISTANCE_KEYS = (
    "index_to_index_normalized",
    "thumb_to_thumb_normalized",
    "left_index_to_thumb_normalized",
    "right_index_to_thumb_normalized",
)

SCALES = {
    "major": (0, 2, 4, 5, 7, 9, 11),
    "minor": (0, 2, 3, 5, 7, 8, 10),
    "pentatonic": (0, 2, 4, 7, 9),
    "chromatic": tuple(range(12)),
}


def extract_distances(frame_data: dict) -> dict:
    """Keep missing measurements as None, distinct from a closed pinch (0)."""
    return {
        DISTANCE_KEYS[0]: frame_data.get(DISTANCE_KEYS[0]),
        DISTANCE_KEYS[1]: frame_data.get(DISTANCE_KEYS[1]),
        DISTANCE_KEYS[2]: frame_data.get("left", {}).get("index_to_thumb_normalized"),
        DISTANCE_KEYS[3]: frame_data.get("right", {}).get("index_to_thumb_normalized"),
    }


def note_name(midi_note: float) -> str:
    number = round(midi_note)
    names = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
    return f"{names[number % 12]}{number // 12 - 1}"


@dataclass(frozen=True)
class SynthParameters:
    frequency: float = 130.8128
    cutoff: float = 200.0
    reverb: float = 0.0
    gain: float = 0.0
    midi_note: float = 48.0
    active: bool = False

    @property
    def label(self) -> str:
        return (
            f"{note_name(self.midi_note)} {self.frequency:.0f}Hz | "
            f"filter {self.cutoff:.0f}Hz | reverb {self.reverb:.0%} | "
            f"volume {self.gain:.0%}"
        )


class GestureMapper:
    """Scale snapping with hysteresis; incomplete/invalid frames mute the voice."""

    def __init__(self, scale: Optional[str] = None):
        self.scale = scale or SYNTH_CONFIG["scale"]
        if self.scale not in (*SCALES, "continuous"):
            raise ValueError(f"Unknown scale: {self.scale}")
        root = SYNTH_CONFIG["root_midi"]
        octaves = SYNTH_CONFIG["octaves"]
        degrees = SCALES.get(self.scale, SCALES["chromatic"])
        self.notes = [root + 12 * octave + degree
                      for octave in range(octaves) for degree in degrees]
        self.notes.append(root + 12 * octaves)
        self._note_index = None
        self.parameters = SynthParameters()

    def map(self, distances: Mapping) -> SynthParameters:
        values = []
        for key in DISTANCE_KEYS:
            value = distances.get(key)
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                self._note_index = None
                self.parameters = replace(self.parameters, gain=0.0, active=False)
                return self.parameters
            values.append(max(0.0, min(1.0, value)))

        index_spacing, thumb_spacing, left_pinch, right_pinch = values
        if self.scale == "continuous":
            midi_note = self.notes[0] + index_spacing * (self.notes[-1] - self.notes[0])
        else:
            position = index_spacing * (len(self.notes) - 1)
            # Cross a boundary by an extra 12% of a step before changing note.
            if self._note_index is None or abs(position - self._note_index) > 0.62:
                self._note_index = min(len(self.notes) - 1, int(position + 0.5))
            midi_note = self.notes[self._note_index]

        minimum = SYNTH_CONFIG["cutoff_min"]
        maximum = SYNTH_CONFIG["cutoff_max"]
        deadzone = SYNTH_CONFIG["volume_deadzone"]
        volume = max(0.0, (right_pinch - deadzone) / (1.0 - deadzone))
        self.parameters = SynthParameters(
            frequency=440.0 * 2.0 ** ((midi_note - 69.0) / 12.0),
            cutoff=minimum * (maximum / minimum) ** thumb_spacing,
            reverb=SYNTH_CONFIG["max_reverb"] * left_pinch,
            gain=volume ** 2,
            midi_note=midi_note,
            active=True,
        )
        return self.parameters
