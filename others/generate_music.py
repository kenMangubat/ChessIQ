"""Procedural ambient focus music generator for ChessIQ ("Chess Ken").

Synthesizes a calm, futuristic emerald-cyan ambient synth loop (64 seconds,
60 BPM, D Dorian / D minor 9) using NumPy stereo DSP with circular delay and
reverb tails so bar 16 wraps seamlessly into bar 1 with zero click or seam.

Encodes to OGG Vorbis (``assets/sounds/bg_music.ogg``) via ``ffmpeg`` when
available (gapless looping in SDL_mixer, compact file size), or falls back to
writing a lossless WAV file if ``ffmpeg`` is unavailable.

Usage (from repository root):
    python -m others.generate_music
"""

from __future__ import annotations

import math
from pathlib import Path
import shutil
import subprocess
import tempfile
import wave

import numpy as np

from others.config import BG_MUSIC_PATH, SOUNDS_DIR

SAMPLE_RATE = 44100
BPM = 60.0
BEATS_PER_BAR = 4
NUM_BARS = 16
BAR_SEC = (60.0 / BPM) * BEATS_PER_BAR  # 4.0 s
TOTAL_SEC = BAR_SEC * NUM_BARS          # 64.0 s
TOTAL_SAMPLES = int(round(TOTAL_SEC * SAMPLE_RATE))


def _midi_to_hz(midi: float) -> float:
    return 440.0 * (2.0 ** ((midi - 69.0) / 12.0))


def _circular_add(
    buf_l: np.ndarray,
    buf_r: np.ndarray,
    start_sample: int,
    sig_l: np.ndarray,
    sig_r: np.ndarray,
) -> None:
    """Add stereo signal into circular loop buffers with modulo wrap-around."""
    n = len(buf_l)
    m = len(sig_l)
    idx = (start_sample + np.arange(m)) % n
    np.add.at(buf_l, idx, sig_l)
    np.add.at(buf_r, idx, sig_r)


def _synth_pad_chord(
    midi_notes: tuple[int, ...],
    dur_sec: float,
    gain: float = 0.16,
    brightness: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Lush detuned stereo analog pad with gentle breathing filter envelope."""
    tail_sec = 1.8
    n = int(round((dur_sec + tail_sec) * SAMPLE_RATE))
    t = np.arange(n, dtype=np.float64) / SAMPLE_RATE

    # Smooth cosine-shaped attack and long release
    attack = 1.15
    release = tail_sec
    env = np.ones(n, dtype=np.float64)
    atk_n = min(n, int(round(attack * SAMPLE_RATE)))
    env[:atk_n] = 0.5 * (1.0 - np.cos(np.pi * np.linspace(0.0, 1.0, atk_n)))
    rel_start = max(atk_n, int(round((dur_sec - 0.35) * SAMPLE_RATE)))
    rel_n = n - rel_start
    if rel_n > 0:
        env[rel_start:] = 0.5 * (1.0 + np.cos(np.pi * np.linspace(0.0, 1.0, rel_n)))

    # Slow organic swell in harmonic shimmer
    swell = 0.65 + 0.35 * np.sin(np.pi * np.clip(t / max(dur_sec, 0.1), 0.0, 1.0))

    left = np.zeros(n, dtype=np.float64)
    right = np.zeros(n, dtype=np.float64)
    scale = gain / max(1, len(midi_notes))

    for i, note in enumerate(midi_notes):
        f = _midi_to_hz(note)
        # Subtle stereo detuning + gentle pan spread across voicing
        detune = 0.24 + 0.05 * (i % 3)
        pan = -0.35 + 0.70 * (i / max(1, len(midi_notes) - 1))
        g_l = math.cos((pan + 1.0) * math.pi / 4.0)
        g_r = math.sin((pan + 1.0) * math.pi / 4.0)

        p1 = 2.0 * np.pi * (f - detune) * t + (i * 0.73)
        p2 = 2.0 * np.pi * (f + detune) * t + (i * 1.41)

        # Warm mellow harmonic recipe (no harsh high partials)
        v_l = (
            1.00 * np.sin(p1)
            + (0.28 * brightness * swell) * np.sin(2.0 * p1 + 0.2)
            + (0.10 * brightness * swell) * np.sin(3.0 * p1)
            + (0.04 * brightness * swell) * np.sin(4.0 * p1)
        )
        v_r = (
            1.00 * np.sin(p2)
            + (0.28 * brightness * swell) * np.sin(2.0 * p2 - 0.2)
            + (0.10 * brightness * swell) * np.sin(3.0 * p2)
            + (0.04 * brightness * swell) * np.sin(4.0 * p2)
        )

        left += (v_l * g_l + v_r * (0.25 * g_l)) * scale
        right += (v_r * g_r + v_l * (0.25 * g_r)) * scale

    return left * env, right * env


def _synth_sub_bass(midi_note: int, dur_sec: float, gain: float = 0.22) -> tuple[np.ndarray, np.ndarray]:
    """Deep, calm fundamental sub-bass with soft cosine envelope."""
    tail_sec = 0.65
    n = int(round((dur_sec + tail_sec) * SAMPLE_RATE))
    t = np.arange(n, dtype=np.float64) / SAMPLE_RATE
    f = _midi_to_hz(midi_note)

    env = np.ones(n, dtype=np.float64)
    atk_n = min(n, int(round(0.45 * SAMPLE_RATE)))
    env[:atk_n] = 0.5 * (1.0 - np.cos(np.pi * np.linspace(0.0, 1.0, atk_n)))
    rel_start = max(atk_n, int(round((dur_sec - 0.15) * SAMPLE_RATE)))
    rel_n = n - rel_start
    if rel_n > 0:
        env[rel_start:] = 0.5 * (1.0 + np.cos(np.pi * np.linspace(0.0, 1.0, rel_n)))

    # Centered low end with gentle 2nd harmonic warmth
    sig = (np.sin(2.0 * np.pi * f * t) + 0.14 * np.sin(4.0 * np.pi * f * t)) * env * gain
    return sig, sig.copy()


def _synth_glass_pluck(
    midi_note: int,
    gain: float = 0.085,
    pan: float = 0.0,
    decay_sec: float = 0.85,
    fm_index: float = 0.95,
) -> tuple[np.ndarray, np.ndarray]:
    """Crystalline cyan-glass FM mallet pluck (calm, futuristic, zero harshness)."""
    dur_sec = decay_sec * 4.2
    n = int(round(dur_sec * SAMPLE_RATE))
    t = np.arange(n, dtype=np.float64) / SAMPLE_RATE
    f = _midi_to_hz(midi_note)

    # Soft rounded attack (14 ms) so plucks never feel percussive or distracting
    atk = 1.0 - np.exp(-t / 0.014)
    amp_env = atk * np.exp(-t / decay_sec)
    mod_env = fm_index * np.exp(-t / 0.14)

    mod = mod_env * np.sin(2.0 * np.pi * (2.0 * f) * t)
    sig = (
        0.82 * np.sin(2.0 * np.pi * f * t + mod)
        + 0.18 * np.sin(2.0 * np.pi * (3.0 * f) * t) * np.exp(-t / 0.22)
    ) * amp_env * gain

    g_l = math.cos((pan + 1.0) * math.pi / 4.0)
    g_r = math.sin((pan + 1.0) * math.pi / 4.0)
    return sig * g_l, sig * g_r


def _apply_circular_stereo_delay(
    left: np.ndarray,
    right: np.ndarray,
    delay_l_sec: float = 0.375,
    delay_r_sec: float = 0.75,
    feedback: float = 0.36,
    wet: float = 0.42,
    taps: int = 8,
) -> tuple[np.ndarray, np.ndarray]:
    """Circular stereo ping-pong delay so echoes wrap seamlessly across loop boundary."""
    d_l = int(round(delay_l_sec * SAMPLE_RATE))
    d_r = int(round(delay_r_sec * SAMPLE_RATE))

    out_l = left.copy()
    out_r = right.copy()

    cur_l = left.copy()
    cur_r = right.copy()
    g = wet
    for _ in range(taps):
        # Cross-feed L <-> R with gentle high-frequency damping via 3-tap smoothing
        nxt_l = np.roll(cur_r, d_l)
        nxt_r = np.roll(cur_l, d_r)
        nxt_l = 0.25 * np.roll(nxt_l, -1) + 0.5 * nxt_l + 0.25 * np.roll(nxt_l, 1)
        nxt_r = 0.25 * np.roll(nxt_r, -1) + 0.5 * nxt_r + 0.25 * np.roll(nxt_r, 1)

        out_l += nxt_l * g
        out_r += nxt_r * g
        cur_l, cur_r = nxt_l, nxt_r
        g *= feedback

    return out_l, out_r


def _apply_circular_reverb(
    left: np.ndarray,
    right: np.ndarray,
    wet: float = 0.34,
) -> tuple[np.ndarray, np.ndarray]:
    """Smooth circular multi-tap ambient reverb for spacious depth and seamless looping."""
    tap_specs = (
        (0.053, 0.067, 0.32),
        (0.113, 0.131, 0.26),
        (0.197, 0.223, 0.21),
        (0.311, 0.347, 0.16),
        (0.461, 0.503, 0.12),
        (0.673, 0.719, 0.08),
        (0.941, 1.013, 0.05),
    )
    rev_l = np.zeros_like(left)
    rev_r = np.zeros_like(right)

    for dl_s, dr_s, amp in tap_specs:
        dl = int(round(dl_s * SAMPLE_RATE))
        dr = int(round(dr_s * SAMPLE_RATE))
        rev_l += np.roll(left, dl) * amp + np.roll(right, dr) * (amp * 0.35)
        rev_r += np.roll(right, dr) * amp + np.roll(left, dl) * (amp * 0.35)

    # Soft low-pass diffusion on the reverb tail (5-point moving average, circular)
    for _ in range(2):
        rev_l = (
            np.roll(rev_l, -2) + 2.0 * np.roll(rev_l, -1) + 3.0 * rev_l
            + 2.0 * np.roll(rev_l, 1) + np.roll(rev_l, 2)
        ) / 9.0
        rev_r = (
            np.roll(rev_r, -2) + 2.0 * np.roll(rev_r, -1) + 3.0 * rev_r
            + 2.0 * np.roll(rev_r, 1) + np.roll(rev_r, 2)
        ) / 9.0

    return left + rev_l * wet, right + rev_r * wet


def synthesize_ambient_loop() -> np.ndarray:
    """Render the 64-second stereo ChessIQ ambient focus synth loop as int16 PCM."""
    pad_l = np.zeros(TOTAL_SAMPLES, dtype=np.float64)
    pad_r = np.zeros(TOTAL_SAMPLES, dtype=np.float64)
    sub_l = np.zeros(TOTAL_SAMPLES, dtype=np.float64)
    sub_r = np.zeros(TOTAL_SAMPLES, dtype=np.float64)
    arp_l = np.zeros(TOTAL_SAMPLES, dtype=np.float64)
    arp_r = np.zeros(TOTAL_SAMPLES, dtype=np.float64)
    mel_l = np.zeros(TOTAL_SAMPLES, dtype=np.float64)
    mel_r = np.zeros(TOTAL_SAMPLES, dtype=np.float64)

    # 16-bar D Dorian / D minor 9 progression: (root_sub_midi, pad_chord_midis, arp_notes)
    progression: tuple[tuple[int, tuple[int, ...], tuple[int, ...]], ...] = (
        # Cycle 1: Warm emerald pad intro (bars 0..3)
        (38, (50, 57, 60, 64, 69), (62, 69, 72, 76)),  # Dm9
        (34, (46, 53, 57, 62, 64), (62, 65, 69, 74)),  # BbMaj7(#11)
        (31, (43, 50, 53, 57, 62), (58, 62, 65, 69)),  # Gm9
        (33, (45, 52, 55, 60, 62), (60, 64, 67, 72)),  # Am7(11)
        # Cycle 2: Crystal arpeggio enters (bars 4..7)
        (38, (50, 57, 60, 64, 69), (62, 64, 69, 72)),  # Dm9
        (34, (46, 53, 57, 62, 65), (65, 69, 74, 72)),  # BbMaj7
        (29, (41, 48, 52, 57, 60), (60, 64, 69, 72)),  # Fmaj9
        (33, (45, 52, 55, 62, 64), (62, 64, 69, 71)),  # Asus4 / Am9
        # Cycle 3: Full ambient depth + focus motif (bars 8..11)
        (38, (50, 57, 60, 64, 69), (62, 69, 72, 76)),  # Dm9
        (36, (48, 55, 59, 62, 67), (64, 67, 71, 74)),  # Cmaj9
        (34, (46, 53, 57, 62, 65), (62, 65, 69, 74)),  # BbMaj7
        (31, (43, 50, 53, 57, 64), (60, 62, 67, 69)),  # Gm9 / Am7
        # Cycle 4: Spacious calm resolve back to bar 0 (bars 12..15)
        (38, (50, 57, 60, 64, 69), (62, 64, 69, 72)),  # Dm9
        (34, (46, 53, 57, 62, 65), (62, 65, 69, 72)),  # BbMaj7
        (31, (43, 50, 53, 57, 62), (58, 62, 65, 69)),  # Gm9
        (33, (45, 52, 55, 60, 64), (60, 64, 67, 71)),  # Am9
    )

    for bar_idx, (root_midi, chord_midis, arp_notes) in enumerate(progression):
        bar_start = int(round(bar_idx * BAR_SEC * SAMPLE_RATE))

        # Evolving brightness curve across the 16 bars (peaks gently in bars 8..12)
        brightness = 0.78 + 0.32 * math.sin(math.pi * (bar_idx + 0.5) / NUM_BARS)
        p_l, p_r = _synth_pad_chord(chord_midis, BAR_SEC, gain=0.17, brightness=brightness)
        _circular_add(pad_l, pad_r, bar_start, p_l, p_r)

        # Warm sub-bass enters smoothly from bar 0 (slightly fuller in cycles 2-3)
        sub_gain = 0.16 if bar_idx < 4 or bar_idx >= 14 else 0.21
        s_l, s_r = _synth_sub_bass(root_midi, BAR_SEC, gain=sub_gain)
        _circular_add(sub_l, sub_r, bar_start, s_l, s_r)

        # Gentle glass arpeggio (bars 4..14, sparse in bar 15 for clean breath before loop)
        if 4 <= bar_idx <= 14:
            step_beats = (0.0, 0.75, 1.5, 2.0, 2.75, 3.5)
            for step_i, beat_off in enumerate(step_beats):
                note = arp_notes[step_i % len(arp_notes)]
                # Occasional upper-octave shimmer on beat 2.75
                if step_i == 4 and bar_idx % 2 == 0:
                    note += 12
                pan = -0.45 if step_i % 2 == 0 else 0.45
                vel = 0.052 if step_i in (0, 3) else 0.038
                if bar_idx == 4:
                    vel *= 0.65
                elif bar_idx == 14:
                    vel *= 0.70
                g_l, g_r = _synth_glass_pluck(
                    note, gain=vel, pan=pan, decay_sec=0.72, fm_index=0.85
                )
                pos = bar_start + int(round(beat_off * (60.0 / BPM) * SAMPLE_RATE))
                _circular_add(arp_l, arp_r, pos, g_l, g_r)

    # Sparse contemplative high focus melody (bars 8..13)
    melody_events: tuple[tuple[float, int, float, float], ...] = (
        # (beat_from_start, midi_note, gain, pan)
        (32.0, 81, 0.068, -0.18),  # Bar 8 beat 0: A5
        (34.5, 79, 0.055,  0.15),  # G5
        (36.0, 76, 0.065, -0.12),  # Bar 9 beat 0: E5
        (39.0, 74, 0.052,  0.20),  # D5
        (40.0, 77, 0.064, -0.16),  # Bar 10 beat 0: F5
        (42.5, 76, 0.054,  0.18),  # E5
        (44.0, 74, 0.062, -0.10),  # Bar 11 beat 0: D5
        (46.5, 72, 0.052,  0.14),  # C5
        (48.0, 76, 0.060, -0.14),  # Bar 12 beat 0: E5
        (51.0, 74, 0.048,  0.12),  # D5
        (52.0, 69, 0.050,  0.00),  # Bar 13 beat 0: A4 resolve
    )
    for beat_pos, note, vel, pan in melody_events:
        m_l, m_r = _synth_glass_pluck(
            note, gain=vel, pan=pan, decay_sec=1.35, fm_index=0.55
        )
        pos = int(round(beat_pos * (60.0 / BPM) * SAMPLE_RATE))
        _circular_add(mel_l, mel_r, pos, m_l, m_r)

    # Apply circular stereo delay to arpeggio & melody for lush futuristic atmosphere
    arp_l, arp_r = _apply_circular_stereo_delay(arp_l, arp_r, 0.375, 0.75, feedback=0.35, wet=0.42)
    mel_l, mel_r = _apply_circular_stereo_delay(mel_l, mel_r, 0.50, 1.00, feedback=0.32, wet=0.38)

    # Sum stems and apply spacious circular ambient reverb to upper layers (keep sub tight)
    upper_l = pad_l + arp_l + mel_l
    upper_r = pad_r + arp_r + mel_r
    upper_l, upper_r = _apply_circular_reverb(upper_l, upper_r, wet=0.36)

    mix_l = upper_l + sub_l
    mix_r = upper_r + sub_r

    # Remove any DC offset
    mix_l -= np.mean(mix_l)
    mix_r -= np.mean(mix_r)

    # Gentle soft-knee warmth and target -3.0 dBFS peak headroom
    peak = max(float(np.max(np.abs(mix_l))), float(np.max(np.abs(mix_r))), 1e-6)
    mix_l = np.tanh((mix_l / peak) * 1.15)
    mix_r = np.tanh((mix_r / peak) * 1.15)
    peak2 = max(float(np.max(np.abs(mix_l))), float(np.max(np.abs(mix_r))), 1e-6)
    target_peak = 0.70  # ~ -3.1 dBFS so background music sits calmly under SFX
    mix_l = (mix_l / peak2) * target_peak
    mix_r = (mix_r / peak2) * target_peak

    stereo = np.empty((TOTAL_SAMPLES, 2), dtype=np.int16)
    stereo[:, 0] = np.clip(np.round(mix_l * 32767.0), -32767, 32767).astype(np.int16)
    stereo[:, 1] = np.clip(np.round(mix_r * 32767.0), -32767, 32767).astype(np.int16)
    return stereo


def write_music_file(output_path: Path = BG_MUSIC_PATH) -> Path:
    """Generate and save the background music file (OGG Vorbis via ffmpeg, WAV fallback)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    stereo = synthesize_ambient_loop()

    ffmpeg_bin = shutil.which("ffmpeg")
    if output_path.suffix.lower() == ".ogg" and ffmpeg_bin:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_wav = Path(tmp.name)
        try:
            with wave.open(str(tmp_wav), "wb") as wf:
                wf.setnchannels(2)
                wf.setsampwidth(2)
                wf.setframerate(SAMPLE_RATE)
                wf.writeframes(stereo.tobytes())

            subprocess.run(
                [
                    ffmpeg_bin,
                    "-y",
                    "-loglevel",
                    "error",
                    "-i",
                    str(tmp_wav),
                    "-c:a",
                    "libvorbis",
                    "-q:a",
                    "6",
                    str(output_path),
                ],
                check=True,
            )
            return output_path
        finally:
            tmp_wav.unlink(missing_ok=True)

    # Fallback: write directly as WAV
    wav_path = output_path if output_path.suffix.lower() == ".wav" else output_path.with_suffix(".wav")
    with wave.open(str(wav_path), "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(stereo.tobytes())
    return wav_path


def main() -> int:
    out = write_music_file(BG_MUSIC_PATH)
    size_kb = out.stat().st_size / 1024.0
    print(f"Generated ChessIQ ambient focus music loop: {out} ({TOTAL_SEC:.1f}s, {size_kb:.1f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
