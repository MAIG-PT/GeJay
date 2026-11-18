"""Defaults for the local hand-controlled synthesizer."""

SYNTH_CONFIG = {
    "sample_rate": 48000,
    "block_size": 256,
    "master_gain": 0.2,
    "root_midi": 48,  # C3
    "octaves": 2,
    "scale": "pentatonic",
    "cutoff_min": 200.0,
    "cutoff_max": 8000.0,
    "max_reverb": 0.55,
    "volume_deadzone": 0.08,
    "smoothing_seconds": 0.012,  # Keep quick finger movements responsive.
    "tracking_timeout": 0.3,
}
