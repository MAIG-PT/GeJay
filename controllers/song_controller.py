#!/usr/bin/env python3
"""
Song controller for the Gejay hand-controlled FoxDot system.

Responsibilities:
- Dynamically load / reload a song module.
- Instantiate and start the song object.
- Throttle musical updates.
- Dispatch ModeController song_data to the song's mode-specific hooks.
- Forward keyboard input to the current song.
- Stop / clean up FoxDot resources safely.
"""

from __future__ import annotations

import importlib
import time
from pathlib import Path
from typing import Any, Dict, Optional

import sys

# Ensure local project and songs package are importable, matching the old main.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from FoxDot import Clock  # type: ignore


class SongController:
    def __init__(
        self,
        song_name: str = "songs.derpy_naive",
        update_interval: float = 0.25,
    ) -> None:
        self.song_name = song_name
        self.update_interval = update_interval

        self.song_module: Optional[Any] = None
        self.song: Optional[Any] = None
        self.last_update_time: float = 0.0

        self.load_song(song_name, force_reload=False)

    def load_song(self, module_name: Optional[str] = None, force_reload: bool = False) -> None:
        module_name = module_name or self.song_name

        if self.song is not None:
            try:
                self.song.stop()
            except Exception:
                pass

        importlib.invalidate_caches()

        if self.song_module is not None and getattr(self.song_module, "__name__", None) == module_name:
            if force_reload:
                self.song_module = importlib.reload(self.song_module)
        else:
            self.song_module = importlib.import_module(module_name)
            if force_reload:
                self.song_module = importlib.reload(self.song_module)

        self.song_name = module_name
        self.song = self.song_module.Song()
        self.song.start()
        self.last_update_time = 0.0

        print(f"Loaded song: {module_name}")

    def reload(self) -> None:
        self.load_song(self.song_name, force_reload=True)

    def update(self, song_data: Dict[str, Any]) -> None:
        if self.song is None:
            return

        now = time.time()
        if (now - self.last_update_time) < self.update_interval:
            return

        mode = song_data.get("mode", "strange")

        if mode == "strange":
            left_it = float(song_data.get("left_pinch", 0.0) or 0.0)
            right_it = float(song_data.get("right_pinch", 0.0) or 0.0)
            index_index = float(song_data.get("index_to_index", 0.0) or 0.0)
            thumb_thumb = float(song_data.get("thumb_to_thumb", 0.0) or 0.0)

            self.song.update_from_strange(
                left_it,
                right_it,
                index_index,
                thumb_thumb,
            )

        elif mode == "charm":
            left_count = int(song_data.get("left_finger_count", 0) or 0)
            right_count = int(song_data.get("right_finger_count", 0) or 0)

            left_gesture = song_data.get("left_gesture")
            right_gesture = song_data.get("right_gesture")

            self.song.update_from_charm(
                left_count,
                right_count,
                left_gesture,
                right_gesture,
            )

        elif mode == "beauty":
            left_count = int(song_data.get("left_finger_count", 0) or 0)
            right_pinch = float(song_data.get("right_pinch", 0.0) or 0.0)

            self.song.update_from_beauty(
                left_count,
                right_pinch,
            )

        elif mode == "spin":
            update_spin = getattr(self.song, "update_from_spin", None)
            if callable(update_spin):
                update_spin(song_data)

        self.last_update_time = now

    def handle_key(self, key: int) -> bool:
        if key == ord("r"):
            self.reload()
            return True

        if key != 255 and self.song is not None:
            try:
                self.song.handle_key(chr(key))
                return True
            except Exception:
                return False

        return False

    def stop(self) -> None:
        if self.song is not None:
            try:
                self.song.stop()
            except Exception:
                pass

        try:
            Clock.clear()
        except Exception:
            pass