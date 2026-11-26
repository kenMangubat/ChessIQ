"""Build-time audio clipping, silence trimming, and speech segment detector.

Decodes WAV via stdlib ``wave`` and MP3/OGG via ``pygame.mixer`` + ``pygame.sndarray``
(falling back to optional build-time ``ffmpeg`` only if needed). Never overwrites
or deletes the original source file and adds no runtime dependencies.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import wave

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from others.config import BLUNDER_CLIP_PATH, BLUNDER_SOURCE_AUDIO_PATH


FRAME_MS: float = 10.0
DEFAULT_PAD_MS: float = 70.0
DEFAULT_FADE_MS: float = 8.0
DEFAULT_TARGET_PEAK_DBFS: float = -1.0
MIN_SILENCE_RMS: float = 0.004
BED_NOISE_THRESHOLD_DBFS: float = -34.0
SPEECH_SNR_DB: float = 8.5
MERGE_GAP_MS: float = 320.0
MIN_SEGMENT_MS: float = 250.0


@dataclass(frozen=True)
class AudioSegment:
    """Detected speech/energy segment in an audio signal."""

    index: int
    start_sec: float
    end_sec: float
    peak_dbfs: float
    rms_dbfs: float

    @property
    def duration_sec(self) -> float:
        return max(0.0, self.end_sec - self.start_sec)


def _linear_to_dbfs(val: float) -> float:
    return 20.0 * math.log10(max(float(val), 1e-6))


def _dbfs_to_linear(db: float) -> float:
    return 10.0 ** (float(db) / 20.0)


def load_wav(path: Path) -> tuple[np.ndarray, int, int]:
    """Read a 16-bit PCM WAV file into float32 samples of shape ``(n_frames, n_channels)``."""
    with wave.open(str(path), "rb") as wf:
        n_channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        sample_rate = wf.getframerate()
        if sample_width != 2:
            raise ValueError(f"Unsupported WAV sample width: {sample_width * 8}-bit")
        raw = wf.readframes(wf.getnframes())
    samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    return samples.reshape(-1, n_channels), sample_rate, n_channels


def save_wav(
    path: Path,
    samples: np.ndarray,
    sample_rate: int,
    n_channels: int | None = None,
) -> None:
    """Write float32 samples ``[-1.0, 1.0]`` as 16-bit PCM WAV using stdlib ``wave``."""
    arr = np.asarray(samples, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[:, np.newaxis]
    ch = int(n_channels) if n_channels is not None else int(arr.shape[1])
    if arr.shape[1] != ch:
        if ch == 1:
            arr = np.mean(arr, axis=1, keepdims=True)
        else:
            raise ValueError(f"Channel mismatch: array has {arr.shape[1]}, requested {ch}")

    clipped = np.clip(arr, -1.0, 1.0)
    pcm = np.round(clipped * 32767.0).astype(np.int16)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(ch)
        wf.setsampwidth(2)
        wf.setframerate(int(sample_rate))
        wf.writeframes(pcm.tobytes())


def decode_audio(path: Path, target_sample_rate: int = 44100) -> tuple[np.ndarray, int, int]:
    """Decode WAV, MP3, or OGG into float32 ``(n_frames, n_channels)`` without extra deps."""
    if not path.is_file():
        raise FileNotFoundError(f"Source audio file not found: {path}")

    if path.suffix.lower() == ".wav":
        return load_wav(path)

    try:
        os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
        import pygame

        already_init = pygame.mixer.get_init() is not None
        if not already_init:
            pygame.mixer.init(frequency=target_sample_rate, size=-16, channels=2)
        try:
            init_sr, _init_fmt, init_ch = pygame.mixer.get_init()
            snd = pygame.mixer.Sound(str(path))
            arr = pygame.sndarray.array(snd)
            if arr.ndim == 1:
                arr = arr[:, np.newaxis]
            if arr.dtype == np.int16:
                samples = arr.astype(np.float32) / 32768.0
            elif arr.dtype == np.int32:
                samples = arr.astype(np.float32) / 2147483648.0
            elif arr.dtype == np.uint8:
                samples = (arr.astype(np.float32) - 128.0) / 128.0
            else:
                samples = arr.astype(np.float32)
            return samples, int(init_sr), int(samples.shape[1] or init_ch)
        finally:
            if not already_init:
                pygame.mixer.quit()
    except Exception:
        pass

    ffmpeg_exe = shutil.which("ffmpeg")
    if not ffmpeg_exe:
        raise RuntimeError(
            f"Could not decode {path.name} with pygame.mixer and 'ffmpeg' was not found on PATH."
        )
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_wav = Path(tmp_dir) / "decoded.wav"
        subprocess.run(
            [
                ffmpeg_exe,
                "-y",
                "-nostdin",
                "-i",
                str(path),
                "-ar",
                str(target_sample_rate),
                "-acodec",
                "pcm_s16le",
                str(tmp_wav),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return load_wav(tmp_wav)


def detect_speech_segments(
    samples: np.ndarray,
    sample_rate: int,
    *,
    frame_ms: float = FRAME_MS,
    merge_gap_ms: float = MERGE_GAP_MS,
    min_segment_ms: float = MIN_SEGMENT_MS,
) -> tuple[list[AudioSegment], float, bool]:
    """Detect candidate speech segments via short-time RMS energy."""
    arr = np.asarray(samples, dtype=np.float32)
    mono = np.mean(arr, axis=1) if arr.ndim == 2 else arr

    hop = max(1, int(round(sample_rate * frame_ms / 1000.0)))
    n_frames = len(mono) // hop
    if n_frames == 0:
        return [], -96.0, False

    framed = mono[: n_frames * hop].reshape(n_frames, hop)
    rms = np.sqrt(np.mean(framed * framed, axis=1) + 1e-12)
    peak_env = np.max(np.abs(framed), axis=1)

    floor_rms = float(np.percentile(rms, 25)) if len(rms) else 0.0
    floor_dbfs = _linear_to_dbfs(floor_rms)
    has_background_bed = floor_dbfs > BED_NOISE_THRESHOLD_DBFS

    if has_background_bed:
        snr_ratio = _dbfs_to_linear(SPEECH_SNR_DB)
        high_thresh = max(MIN_SILENCE_RMS * 3.0, floor_rms * snr_ratio)
        low_thresh = max(MIN_SILENCE_RMS * 2.0, floor_rms * _dbfs_to_linear(SPEECH_SNR_DB * 0.55))
    else:
        max_rms = float(np.max(rms))
        high_thresh = max(MIN_SILENCE_RMS, max_rms * 0.04)
        low_thresh = max(MIN_SILENCE_RMS * 0.6, max_rms * 0.02)

    raw_runs: list[list[int]] = []
    idx = 0
    while idx < n_frames:
        if rms[idx] >= high_thresh:
            start = idx
            while start > 0 and rms[start - 1] >= low_thresh:
                start -= 1
            end = idx + 1
            while end < n_frames and rms[end] >= low_thresh:
                end += 1
            raw_runs.append([start, end])
            idx = end
        else:
            idx += 1

    gap_frames = max(1, int(round(merge_gap_ms / frame_ms)))
    merged: list[list[int]] = []
    for start_f, end_f in raw_runs:
        if merged and (start_f - merged[-1][1]) <= gap_frames:
            merged[-1][1] = max(merged[-1][1], end_f)
        else:
            merged.append([start_f, end_f])

    min_frames = max(1, int(round(min_segment_ms / frame_ms)))
    segments: list[AudioSegment] = []
    for start_f, end_f in merged:
        if (end_f - start_f) < min_frames:
            continue
        seg_peak = float(np.max(peak_env[start_f:end_f]))
        seg_rms = float(np.sqrt(np.mean(rms[start_f:end_f] ** 2)))
        segments.append(
            AudioSegment(
                index=len(segments) + 1,
                start_sec=start_f * hop / sample_rate,
                end_sec=end_f * hop / sample_rate,
                peak_dbfs=_linear_to_dbfs(seg_peak),
                rms_dbfs=_linear_to_dbfs(seg_rms),
            )
        )

    return segments, floor_dbfs, has_background_bed


def clip_and_normalize(
    samples: np.ndarray,
    sample_rate: int,
    *,
    start_sec: float | None = None,
    end_sec: float | None = None,
    pad_ms: float = DEFAULT_PAD_MS,
    fade_ms: float = DEFAULT_FADE_MS,
    target_peak_dbfs: float = DEFAULT_TARGET_PEAK_DBFS,
) -> np.ndarray:
    """Slice or auto-trim audio, apply subtle anti-click fades, and peak-normalize cleanly."""
    arr = np.asarray(samples, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[:, np.newaxis]
    n_total = len(arr)
    if n_total == 0:
        return arr.copy()

    pad_samples = max(0, int(round(sample_rate * pad_ms / 1000.0)))

    if start_sec is not None or end_sec is not None:
        s_idx = 0 if start_sec is None else max(0, int(round(start_sec * sample_rate)))
        e_idx = n_total if end_sec is None else min(n_total, int(round(end_sec * sample_rate)))
        if e_idx <= s_idx:
            raise ValueError(f"Invalid cut window: start={start_sec}, end={end_sec}")
        clip_start = max(0, s_idx - pad_samples)
        clip_end = min(n_total, e_idx + pad_samples)
    else:
        mono = np.mean(arr, axis=1)
        hop = max(1, int(round(sample_rate * FRAME_MS / 1000.0)))
        n_frames = len(mono) // hop
        if n_frames > 0:
            framed = mono[: n_frames * hop].reshape(n_frames, hop)
            rms = np.sqrt(np.mean(framed * framed, axis=1) + 1e-12)
            thresh = max(MIN_SILENCE_RMS, float(np.max(rms)) * 0.02)
            active = np.flatnonzero(rms >= thresh)
            if len(active):
                s_idx = int(active[0]) * hop
                e_idx = min(n_total, (int(active[-1]) + 1) * hop)
                clip_start = max(0, s_idx - pad_samples)
                clip_end = min(n_total, e_idx + pad_samples)
            else:
                clip_start, clip_end = 0, n_total
        else:
            clip_start, clip_end = 0, n_total

    clip = arr[clip_start:clip_end].copy()
    if len(clip) == 0:
        return clip

    target_linear = min(0.99, _dbfs_to_linear(target_peak_dbfs))
    peak = float(np.max(np.abs(clip)))
    if peak > 1e-6:
        clip *= target_linear / peak

    fade_samples = min(len(clip) // 4, max(1, int(round(sample_rate * fade_ms / 1000.0))))
    if fade_samples > 1:
        fade_in = np.linspace(0.0, 1.0, fade_samples, dtype=np.float32)[:, np.newaxis]
        fade_out = np.linspace(1.0, 0.0, fade_samples, dtype=np.float32)[:, np.newaxis]
        clip[:fade_samples] *= fade_in
        clip[-fade_samples:] *= fade_out

    return np.clip(clip, -1.0, 1.0)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Inspect speech segments and clip/normalize voice lines to WAV."
    )
    parser.add_argument("--in", dest="input_path", type=Path, default=BLUNDER_SOURCE_AUDIO_PATH)
    parser.add_argument("--out", dest="output_path", type=Path, default=BLUNDER_CLIP_PATH)
    parser.add_argument("--start", type=float, default=None)
    parser.add_argument("--end", type=float, default=None)
    parser.add_argument("--segment", type=int, default=None)
    parser.add_argument("--pad-ms", type=float, default=None)
    parser.add_argument("--fade-ms", type=float, default=DEFAULT_FADE_MS)
    parser.add_argument("--mono", action="store_true")
    args = parser.parse_args(argv)

    src: Path = args.input_path
    dst: Path = args.output_path
    if src.resolve() == dst.resolve():
        raise ValueError("Refusing to overwrite original source audio file.")

    samples, sr, n_ch = decode_audio(src)

    if args.start is not None or args.end is not None:
        pad_ms = 0.0 if args.pad_ms is None else args.pad_ms
        clip = clip_and_normalize(
            samples, sr, start_sec=args.start, end_sec=args.end, pad_ms=pad_ms, fade_ms=args.fade_ms
        )
        out_ch = 1 if args.mono else n_ch
        save_wav(dst, clip, sr, n_channels=out_ch)
        print(f"Saved manual clip: {dst} ({len(clip) / sr:.3f}s, {dst.stat().st_size:,} bytes)")
        return 0

    segments, floor_dbfs, has_bed = detect_speech_segments(samples, sr)
    print(f"Source: {src} ({len(samples)/sr:.2f}s) | Noise floor: {floor_dbfs:.1f} dBFS")
    for seg in segments:
        print(f"  #{seg.index:2d}  {seg.start_sec:7.3f}s .. {seg.end_sec:7.3f}s  ({seg.duration_sec:.3f}s)  peak={seg.peak_dbfs:.1f}dB")

    if args.segment is not None:
        seg = segments[args.segment - 1]
        pad_ms = DEFAULT_PAD_MS if args.pad_ms is None else args.pad_ms
        clip = clip_and_normalize(
            samples, sr, start_sec=seg.start_sec, end_sec=seg.end_sec, pad_ms=pad_ms, fade_ms=args.fade_ms
        )
        save_wav(dst, clip, sr, n_channels=1 if args.mono else n_ch)
        print(f"Exported segment #{seg.index} -> {dst} ({len(clip) / sr:.3f}s)")
        return 0

    if len(segments) == 1 and not has_bed:
        seg = segments[0]
        pad_ms = DEFAULT_PAD_MS if args.pad_ms is None else args.pad_ms
        clip = clip_and_normalize(
            samples, sr, start_sec=seg.start_sec, end_sec=seg.end_sec, pad_ms=pad_ms, fade_ms=args.fade_ms
        )
        save_wav(dst, clip, sr, n_channels=1 if args.mono else n_ch)
        print(f"Auto-trimmed single segment -> {dst} ({len(clip) / sr:.3f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
