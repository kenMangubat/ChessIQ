"""Extract and auto-trim chess sound effects from assets/chesssound.mp4.

Produces normalized 16-bit PCM WAV clips in assets/sounds/ with tight leading
trim (~5 ms pre-roll), 10 ms tail fade-out, and 200 ms gap merging so the
double-clack castling cue is never split.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import wave

import numpy as np

from others.config import ASSETS_DIR, SOUNDS_DIR

SOURCE_VIDEO = ASSETS_DIR / "chesssound.mp4"
SAMPLE_RATE = 44100
FRAME_MS = 5
MERGE_GAP_MS = 200
MIN_SEGMENT_MS = 20
ONSET_THRESHOLD = 0.015
TAIL_THRESHOLD = 0.002
PRE_ROLL_MS = 5
TAIL_PAD_MS = 10
FADE_OUT_MS = 10
TARGET_PEAK = 0.90


@dataclass(frozen=True)
class DetectedSegment:
    """Audio segment detected by amplitude onset/silence scanning."""

    index: int
    start_sec: float
    end_sec: float
    peak: float

    @property
    def duration_ms(self) -> float:
        return (self.end_sec - self.start_sec) * 1000.0


def resolve_ffmpeg() -> str:
    """Locate FFmpeg on PATH, falling back to imageio-ffmpeg."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        return ffmpeg

    try:
        import imageio_ffmpeg  # type: ignore[import-not-found]

        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        print("FFmpeg not found on PATH; installing imageio-ffmpeg into active venv...")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "imageio-ffmpeg"],
            check=True,
        )
        import imageio_ffmpeg  # type: ignore[import-not-found]

        return imageio_ffmpeg.get_ffmpeg_exe()


def extract_full_wav(ffmpeg_exe: str, source_video: Path, wav_out: Path) -> None:
    """Extract uncompressed 16-bit PCM WAV audio from *source_video*."""
    subprocess.run(
        [
            ffmpeg_exe,
            "-y",
            "-nostdin",
            "-i",
            str(source_video),
            "-vn",
            "-ar",
            str(SAMPLE_RATE),
            "-acodec",
            "pcm_s16le",
            str(wav_out),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def load_wav(wav_path: Path) -> tuple[np.ndarray, int, int]:
    """Read 16-bit PCM WAV into float32 array of shape (n_frames, n_channels)."""
    with wave.open(str(wav_path), "rb") as wf:
        n_channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        sample_rate = wf.getframerate()
        if sample_width != 2:
            raise ValueError(f"Expected 16-bit PCM WAV, got {sample_width * 8}-bit")
        raw = wf.readframes(wf.getnframes())
    data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    return data.reshape(-1, n_channels), sample_rate, n_channels


def save_wav(wav_path: Path, samples: np.ndarray, sample_rate: int, n_channels: int) -> None:
    """Write float32 array (-1..1) as 16-bit PCM WAV."""
    clipped = np.clip(samples, -1.0, 1.0)
    pcm = np.round(clipped * 32767.0).astype(np.int16)
    wav_path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(wav_path), "wb") as wf:
        wf.setnchannels(n_channels)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())


def detect_segments(samples: np.ndarray, sample_rate: int) -> list[DetectedSegment]:
    """Detect non-silent audio segments and merge gaps shorter than MERGE_GAP_MS."""
    mono_abs = np.max(np.abs(samples), axis=1)
    hop = max(1, int(sample_rate * FRAME_MS / 1000))
    n_frames = len(mono_abs) // hop
    env = np.max(mono_abs[: n_frames * hop].reshape(n_frames, hop), axis=1)

    active = env > TAIL_THRESHOLD
    raw_runs: list[list[int]] = []
    in_run = False
    run_start = 0
    for idx, is_active in enumerate(active):
        if is_active and not in_run:
            in_run = True
            run_start = idx
        elif not is_active and in_run:
            in_run = False
            raw_runs.append([run_start, idx])
    if in_run:
        raw_runs.append([run_start, len(active)])

    merged: list[list[int]] = []
    for start_f, end_f in raw_runs:
        if merged and (start_f - merged[-1][1]) * FRAME_MS < MERGE_GAP_MS:
            merged[-1][1] = end_f
        else:
            merged.append([start_f, end_f])

    segments: list[DetectedSegment] = []
    for start_f, end_f in merged:
        dur_ms = (end_f - start_f) * FRAME_MS
        peak = float(np.max(env[start_f:end_f])) if end_f > start_f else 0.0
        if dur_ms >= MIN_SEGMENT_MS and peak >= ONSET_THRESHOLD:
            segments.append(
                DetectedSegment(
                    index=len(segments) + 1,
                    start_sec=start_f * FRAME_MS / 1000.0,
                    end_sec=end_f * FRAME_MS / 1000.0,
                    peak=peak,
                )
            )
    return segments


def slice_and_process(
    samples: np.ndarray,
    sample_rate: int,
    segment: DetectedSegment,
) -> np.ndarray:
    """Slice *segment* with tight ~5 ms pre-roll, 10 ms fade-out, and peak normalization."""
    n_total = len(samples)
    seg_start = max(0, int(segment.start_sec * sample_rate))
    seg_end = min(n_total, int(segment.end_sec * sample_rate))
    mono_abs = np.max(np.abs(samples[seg_start:seg_end]), axis=1)

    # Tighten leading onset to the first sample crossing the tail threshold
    # so softer first notes of two-tone chimes (e.g. game_end) are preserved
    onset_hits = np.flatnonzero(mono_abs >= TAIL_THRESHOLD)
    onset_offset = int(onset_hits[0]) if len(onset_hits) else 0
    pre_roll = int(sample_rate * PRE_ROLL_MS / 1000)
    tail_pad = int(sample_rate * TAIL_PAD_MS / 1000)

    clip_start = max(0, seg_start + onset_offset - pre_roll)
    clip_end = min(n_total, seg_end + tail_pad)
    clip = samples[clip_start:clip_end].copy()

    # Peak-normalize consistently across clips
    peak = float(np.max(np.abs(clip))) if len(clip) else 0.0
    if peak > 1e-6:
        clip *= TARGET_PEAK / peak

    # Apply short linear fade-out at the tail to avoid pops/clicks
    fade_len = min(len(clip), max(1, int(sample_rate * FADE_OUT_MS / 1000)))
    if fade_len > 1:
        fade = np.linspace(1.0, 0.0, fade_len, dtype=np.float32)[:, np.newaxis]
        clip[-fade_len:] *= fade

    return clip


def _pick_cleanest(a: DetectedSegment, b: DetectedSegment) -> DetectedSegment:
    """Select the stronger/cleaner take between two repeated cue recordings."""
    return a if a.peak >= b.peak else b


def extract_sounds(source_video: Path = SOURCE_VIDEO, out_dir: Path = SOUNDS_DIR) -> dict[str, float]:
    """Extract, slice, normalize, and save all chess WAV cues. Returns {filename: duration_ms}."""
    if not source_video.is_file():
        raise FileNotFoundError(f"Source audio video not found: {source_video}")

    ffmpeg_exe = resolve_ffmpeg()
    with tempfile.TemporaryDirectory() as tmp_dir:
        temp_wav = Path(tmp_dir) / "chesssound_full.wav"
        extract_full_wav(ffmpeg_exe, source_video, temp_wav)
        samples, sample_rate, n_channels = load_wav(temp_wav)

    segments = detect_segments(samples, sample_rate)
    print(f"Detected {len(segments)} audio segments in {source_video.name}:")
    print(f"{'Index':>5}  {'Start (s)':>10}  {'End (s)':>10}  {'Duration (ms)':>14}  {'Peak':>6}")
    print("-" * 55)
    for seg in segments:
        print(
            f"{seg.index:5d}  {seg.start_sec:10.3f}  {seg.end_sec:10.3f}  "
            f"{seg.duration_ms:14.1f}  {seg.peak:6.3f}"
        )

    if len(segments) < 12:
        raise RuntimeError(
            f"Expected 12 sound cues in {source_video.name}, found {len(segments)}."
        )

    # Known cue order from chesssound.mp4:
    # 1: Game Start | 2-3: Move (x2) | 4-5: Castling (x2) | 6-7: Capture (x2)
    # 8-9: Check (x2) | 10: Game Over | 11: Stalemate | 12: Checkmate
    mapping: dict[str, DetectedSegment] = {
        "game_start.wav": segments[0],
        "move_self.wav": segments[1],
        "move_opponent.wav": segments[2],
        "castle.wav": _pick_cleanest(segments[3], segments[4]),
        "capture.wav": _pick_cleanest(segments[5], segments[6]),
        "check.wav": _pick_cleanest(segments[7], segments[8]),
        "game_end.wav": segments[9],
        "stalemate.wav": segments[10],
        "checkmate.wav": segments[11],
        # Promotion is not separate in the source clip; reuse game_start chime
        "promote.wav": segments[0],
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    durations_ms: dict[str, float] = {}
    print(f"\nSaved trimmed WAV clips to {out_dir}:")
    for filename, seg in mapping.items():
        clip = slice_and_process(samples, sample_rate, seg)
        out_path = out_dir / filename
        save_wav(out_path, clip, sample_rate, n_channels)
        dur_ms = len(clip) / sample_rate * 1000.0
        durations_ms[filename] = dur_ms
        print(f"  {filename:<18} <- seg #{seg.index:02d}  ({dur_ms:6.1f} ms, {out_path.stat().st_size} bytes)")

    return durations_ms


if __name__ == "__main__":
    extract_sounds()
