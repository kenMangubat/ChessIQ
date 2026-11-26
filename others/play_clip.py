"""Tiny audio clip preview player for auditioning voice clips or timestamp ranges."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import numpy as np
import pygame

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def play_clip(
    path: Path,
    *,
    start_sec: float | None = None,
    end_sec: float | None = None,
    volume: float = 0.9,
) -> None:
    """Play an audio file (or a specific [start_sec, end_sec] slice) via pygame.mixer."""
    if not path.is_file():
        raise FileNotFoundError(f"Audio file not found: {path}")

    pygame.mixer.init(frequency=44100, size=-16, channels=2)
    try:
        sound = pygame.mixer.Sound(str(path))
        sr, _fmt, _ch = pygame.mixer.get_init()
        if start_sec is not None or end_sec is not None:
            arr = pygame.sndarray.array(sound)
            n_total = len(arr)
            s_idx = 0 if start_sec is None else max(0, int(round(start_sec * sr)))
            e_idx = n_total if end_sec is None else min(n_total, int(round(end_sec * sr)))
            if e_idx <= s_idx:
                raise ValueError(f"Invalid slice [{start_sec}, {end_sec}]")
            sound = pygame.sndarray.make_sound(np.ascontiguousarray(arr[s_idx:e_idx]))
            print(f"Playing {path.name} [{s_idx / sr:.2f}s .. {e_idx / sr:.2f}s] ({sound.get_length():.2f}s)")
        else:
            print(f"Playing {path} ({sound.get_length():.2f}s)")
        sound.set_volume(max(0.0, min(1.0, float(volume))))
        channel = sound.play()
        if channel is None:
            return
        while channel.get_busy():
            time.sleep(0.03)
    finally:
        pygame.mixer.quit()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Preview a voice clip or timestamp slice.")
    parser.add_argument("path", type=Path)
    parser.add_argument("--start", type=float, default=None)
    parser.add_argument("--end", type=float, default=None)
    parser.add_argument("--volume", type=float, default=0.9)
    args = parser.parse_args(argv)
    play_clip(args.path, start_sec=args.start, end_sec=args.end, volume=args.volume)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
