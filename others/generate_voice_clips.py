"""Build-time voice-over generator for ChessIQ bot bubble lines.

Supports two build-time backends (never imported at game runtime):
- ``omnivoice`` (default): Zero-shot voice cloning via ``k2-fsa/OmniVoice`` using
  a cleaned reference sample from ``assets/walang kanen buseng.mp3``, with native
  Cebuano (``ceb``), Filipino (``fil``), and English (``en``) synthesis plus
  subtle per-bot vocal shading via FFmpeg ``rubberband``.
- ``edge``: Lightweight fallback via ``edge-tts`` (``fil-PH-AngeloNeural``).
"""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass
import datetime
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from others.clip_audio import (
    DEFAULT_FADE_MS,
    DEFAULT_PAD_MS,
    DEFAULT_TARGET_PEAK_DBFS,
    clip_and_normalize,
    decode_audio,
    load_wav,
    save_wav,
)
from others.config import BLUNDER_CLIP_PATH, BLUNDER_SOURCE_AUDIO_PATH, VOICE_DIR
from ui.speech_bubble import SpeechLine, load_speech_lines


TARGET_VOICE_SAMPLE_RATE: int = 22050
OMNIVOICE_MODEL_ID: str = "k2-fsa/OmniVoice"
OMNIVOICE_DESIGN_INSTRUCT: str = "male, middle-aged, low pitch"
OMNIVOICE_DESIGN_SEED: int = 77
OMNIVOICE_DESIGN_ANCHOR_TEXT: str = (
    "Oy bossing! Handa ka na ba sa matinding laban sa chess ngayon? Galingan mo ha!"
)
OMNIVOICE_REF_TEXT: str = (
    "Hoy! Maya, walang kanin bossing! "
    "Kanina pa, kanina pa kami nanghihingi dito!"
)


@dataclass(frozen=True)
class BotVoiceProfile:
    bot_id: str
    bot_name: str
    voice: str
    rate: str
    pitch: str
    ffmpeg_filter: str


BOT_VOICE_PROFILES: dict[str, BotVoiceProfile] = {
    "sick_ken": BotVoiceProfile(
        "sick_ken",
        "Sick Ken",
        "fil-PH-AngeloNeural",
        "-16%",
        "-6Hz",
        "rubberband=pitch=0.966:tempo=0.93:formant=preserved,highshelf=f=4000:g=-2.0",
    ),
    "focus_ken": BotVoiceProfile(
        "focus_ken",
        "Focus Ken",
        "fil-PH-AngeloNeural",
        "+6%",
        "+4Hz",
        "",
    ),
    "condition_ken": BotVoiceProfile(
        "condition_ken",
        "Condition Ken",
        "fil-PH-AngeloNeural",
        "-6%",
        "-16Hz",
        "rubberband=pitch=0.933:tempo=0.98:formant=preserved,bass=g=2.5:f=160",
    ),
}

# Spoken-form overrides so TTS pronounces Bisaya slang and abbreviations cleanly
# without altering the displayed text in data/speech_lines.json.
TTS_TEXT_OVERRIDES: dict[str, str] = {
    "blunder_bossing": "Oy bossing! Kamusta kabuhay-buhay?",
    "blunder_esmeringhoy": "Naah, ma-esmeringhoy na imong posisyon ron!",
    "mistake_sayop": "Sayop man ka, buseng!",
    "mistake_pataka": "Nagpataka ra man kag tira, oy!",
    "mistake_bushet": "Bushet man nang tiraha, oy!",
    "dubious_sure_ka": "Ha? Shur ka ana?",
    "dubious_wa_ka": "Wa ka ma-bo?",
    "dubious_mura_sakto": "Ah, mura mag sakto!",
    "great_pa_kiss": "Wow, pa-kiss ko bi!",
    "great_grabe_sir": "Grabeh, sir! Ayaw ana, sir!",
    "brilliant_omg": "Oh my god! Unsa mana tiraha oy!",
    "brilliant_cheer": "Brilliant!",
    "brilliant_nakita": "Ah shet, shet, nakita ako!",
    "check": "Check! Watch the king!",
    "castle": "King safely tucked away!",
    "bot_win": "Gee gee! Want a rematch?",
}

# Per-bot greeting spoken overrides (greeting shares id="greeting" across bots)
GREETING_OVERRIDES: dict[str, tuple[str, str]] = {
    "sick_ken": ("Bai, gihilantan ko, pildiha daw ko bi!", "ceb"),
    "focus_ken": ("Ha? Shur ka makigduwa ka nako?", "ceb"),
    "condition_ken": ("Hmm, depende kung three yan!", "fil"),
}

# Native OmniVoice ISO language IDs per line_id
LINE_LANGUAGES: dict[str, str] = {
    "blunder_bossing": "fil",
    "blunder_esmeringhoy": "ceb",
    "mistake_sayop": "ceb",
    "mistake_pataka": "ceb",
    "mistake_bushet": "ceb",
    "dubious_sure_ka": "ceb",
    "dubious_wa_ka": "ceb",
    "dubious_mura_sakto": "ceb",
    "great_pa_kiss": "ceb",
    "great_grabe_sir": "ceb",
    "brilliant_omg": "ceb",
    "brilliant_cheer": "en",
    "brilliant_nakita": "fil",
    "check": "en",
    "promotion": "en",
    "heavy_capture": "en",
    "castle": "en",
    "draw": "en",
    "bot_loss": "en",
    "bot_win": "en",
}

_BASE_EVENTS = {"GREETING", "CHECK", "PROMOTION", "HEAVY_CAPTURE", "CASTLE", "DRAW", "WIN", "LOSS"}


def _require_ffmpeg() -> str:
    ffmpeg_exe = shutil.which("ffmpeg")
    if not ffmpeg_exe:
        raise RuntimeError("ffmpeg is required on PATH for reference cleanup and audio conversion.")
    return ffmpeg_exe


def _decode_to_mono_wav(audio_path: Path, sample_rate: int = TARGET_VOICE_SAMPLE_RATE, af_filter: str = "") -> tuple[np.ndarray, int]:
    ffmpeg_exe = shutil.which("ffmpeg")
    if ffmpeg_exe:
        with tempfile.TemporaryDirectory() as td:
            tmp_wav = Path(td) / "out.wav"
            cmd = [ffmpeg_exe, "-y", "-nostdin", "-i", str(audio_path)]
            if af_filter:
                cmd += ["-af", af_filter]
            cmd += ["-ac", "1", "-ar", str(sample_rate), "-acodec", "pcm_s16le", str(tmp_wav)]
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            samples, sr, _ = load_wav(tmp_wav)
            return samples, sr
    samples, sr, _ = decode_audio(audio_path, target_sample_rate=sample_rate)
    if samples.ndim == 2 and samples.shape[1] > 1:
        samples = np.mean(samples, axis=1, keepdims=True)
    return samples, sr


def prepare_reference_audio(source_mp3: Path, out_wav: Path) -> Path:
    """Extract and gently denoise a clean 4.5s speaker reference from the source MP3."""
    ffmpeg_exe = _require_ffmpeg()
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    filter_complex = (
        "[0:a]atrim=3.05:5.30,asetpts=PTS-STARTPTS[a1];"
        "aevalsrc=0:d=0.20:s=24000[s1];"
        "[0:a]atrim=31.85:34.10,asetpts=PTS-STARTPTS[a2];"
        "aevalsrc=0:d=0.25:s=24000[s2];"
        "[a1]aformat=channel_layouts=mono:sample_rates=24000[b1];"
        "[a2]aformat=channel_layouts=mono:sample_rates=24000[b2];"
        "[b1][s1][b2][s2]concat=n=4:v=0:a=1,"
        "highpass=f=100,lowpass=f=7600,afftdn=nf=-20:nr=18:tn=1,"
        "loudnorm=I=-16:TP=-1.5:LRA=11[out]"
    )
    subprocess.run(
        [
            ffmpeg_exe, "-y", "-nostdin",
            "-i", str(source_mp3),
            "-filter_complex", filter_complex,
            "-map", "[out]",
            str(out_wav),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return out_wav


def _spoken_and_lang(line: SpeechLine) -> tuple[str, str]:
    if line.id == "greeting" and len(line.bots) == 1 and line.bots[0] in GREETING_OVERRIDES:
        return GREETING_OVERRIDES[line.bots[0]]
    spoken = TTS_TEXT_OVERRIDES.get(line.id, line.text)
    lang = LINE_LANGUAGES.get(line.id, "ceb")
    return spoken, lang


def _master_key(line: SpeechLine) -> str:
    if line.id == "greeting" and len(line.bots) == 1:
        return f"greeting_{line.bots[0]}"
    return line.id


def generate_clips_omnivoice(
    out_root: Path,
    *,
    include_reactions: bool = True,
    mode: str = "design",
    ref_audio: Path | None = None,
    ref_text: str | None = None,
    instruct: str = OMNIVOICE_DESIGN_INSTRUCT,
    design_seed: int = OMNIVOICE_DESIGN_SEED,
    master_cache_dir: Path | None = None,
    num_step: int = 16,
    force: bool = False,
) -> list[dict[str, object]]:
    """Generate consistent Filipino/Cebuano AI voice clips via k2-fsa/OmniVoice.

    In ``mode="design"`` (default), synthesizes an original studio-clean Filipino
    male anchor clip using OmniVoice Voice Design (no real-person reference audio)
    so every line shares one coherent synthetic AI voice identity.
    """
    try:
        import soundfile as sf  # type: ignore[import-not-found]
        import torch  # type: ignore[import-not-found]
        from omnivoice import OmniVoice  # type: ignore[import-not-found]
        from omnivoice.models.omnivoice import OmniVoiceGenerationConfig  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError(
            "Install build-time 'omnivoice' + CPU PyTorch in a separate venv to use the omnivoice backend."
        ) from exc

    lines_to_process = [
        line
        for line in load_speech_lines()
        if line.audio is None and (include_reactions or line.event in _BASE_EVENTS)
    ]

    cache_dir = master_cache_dir if master_cache_dir is not None else Path(tempfile.mkdtemp(prefix="chessiq_ov_"))
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Check whether any master WAV still needs OmniVoice inference
    missing_masters = [
        line for line in lines_to_process if force or not (cache_dir / f"{_master_key(line)}.wav").is_file()
    ]

    if missing_masters:
        torch.set_num_threads(2)
        t0 = time.perf_counter()
        print(f"Loading {OMNIVOICE_MODEL_ID} on CPU (fp32, mode={mode}, steps={num_step})...", flush=True)
        model = OmniVoice.from_pretrained(OMNIVOICE_MODEL_ID, device_map="cpu", dtype=torch.float32, load_asr=False)
        print(f"Model loaded in {time.perf_counter() - t0:.1f}s (sr={model.sampling_rate})", flush=True)
        cfg = OmniVoiceGenerationConfig(num_step=num_step, guidance_scale=2.0, denoise=True)

        if mode == "design":
            effective_ref_text = ref_text or OMNIVOICE_DESIGN_ANCHOR_TEXT
            if ref_audio is None:
                anchor_wav = cache_dir / "synthetic_filipino_anchor.wav"
                if force or not anchor_wav.is_file():
                    torch.manual_seed(design_seed)
                    np.random.seed(design_seed)
                    anchor_raw = model.generate(
                        text=effective_ref_text,
                        language="fil",
                        instruct=instruct,
                        generation_config=cfg,
                    )[0]
                    if hasattr(anchor_raw, "detach"):
                        anchor_raw = anchor_raw.detach().cpu().numpy()
                    anchor_arr = np.asarray(anchor_raw, dtype=np.float32).squeeze()
                    pad = np.zeros(int(model.sampling_rate * 0.25), dtype=np.float32)
                    sf.write(str(anchor_wav), np.concatenate([anchor_arr, pad]), model.sampling_rate)
                ref_audio = anchor_wav
            elif not ref_audio.is_file():
                raise FileNotFoundError(f"Anchor audio not found: {ref_audio}")
        else:
            effective_ref_text = ref_text or OMNIVOICE_REF_TEXT
            if ref_audio is None:
                ref_audio = prepare_reference_audio(BLUNDER_SOURCE_AUDIO_PATH, cache_dir / "ref_speaker.wav")
            elif not ref_audio.is_file():
                raise FileNotFoundError(f"Reference audio not found: {ref_audio}")

        prompt = model.create_voice_clone_prompt(
            ref_audio=str(ref_audio),
            ref_text=effective_ref_text,
            preprocess_prompt=True,
        )

        total = len(missing_masters)
        for idx, line in enumerate(missing_masters, start=1):
            key = _master_key(line)
            master_wav = cache_dir / f"{key}.wav"
            spoken, lang = _spoken_and_lang(line)
            ts = time.perf_counter()
            audios = model.generate(
                text=spoken,
                language=lang,
                voice_clone_prompt=prompt,
                generation_config=cfg,
            )
            wav = audios[0]
            if hasattr(wav, "detach"):
                wav = wav.detach().cpu().numpy()
            wav = np.asarray(wav, dtype=np.float32).squeeze()
            sf.write(str(master_wav), wav, model.sampling_rate)
            dt = time.perf_counter() - ts
            print(
                f"[{idx}/{total}] {key} ({lang}): {len(wav) / model.sampling_rate:.2f}s audio in {dt:.1f}s -> '{spoken}'",
                flush=True,
            )

    # Render per-bot WAVs (22050 Hz mono 16-bit PCM, trimmed & normalized to -1 dBFS)
    today = datetime.date.today().isoformat()
    records: list[dict[str, object]] = []
    voice_label = "OmniVoice Filipino AI" if mode == "design" else "OmniVoice clone"
    for line in lines_to_process:
        key = _master_key(line)
        master_wav = cache_dir / f"{key}.wav"
        spoken, lang = _spoken_and_lang(line)
        for bot_id in line.bots:
            if bot_id not in BOT_VOICE_PROFILES:
                continue
            profile = BOT_VOICE_PROFILES[bot_id]
            samples, sr = _decode_to_mono_wav(
                master_wav,
                sample_rate=TARGET_VOICE_SAMPLE_RATE,
                af_filter=profile.ffmpeg_filter,
            )
            trimmed = clip_and_normalize(
                samples,
                sr,
                pad_ms=DEFAULT_PAD_MS,
                fade_ms=DEFAULT_FADE_MS,
                target_peak_dbfs=DEFAULT_TARGET_PEAK_DBFS,
            )
            dst = out_root / bot_id / f"{line.id}.wav"
            save_wav(dst, trimmed, sr, n_channels=1)
            records.append(
                {
                    "rel_path": f"{bot_id}/{line.id}.wav",
                    "bot_id": bot_id,
                    "event": line.event,
                    "lang": lang,
                    "text": line.text,
                    "voice": f"{voice_label} ({lang}, {profile.bot_name})",
                    "duration_sec": len(trimmed) / sr,
                    "date": today,
                }
            )
    return records


async def generate_clips_edge(out_root: Path, *, include_reactions: bool = True) -> list[dict[str, object]]:
    """Fallback build-time generator using edge-tts (fil-PH-AngeloNeural)."""
    try:
        import edge_tts  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError("Install build-time tool 'edge-tts' in a separate venv: pip install edge-tts") from exc

    records: list[dict[str, object]] = []
    today = datetime.date.today().isoformat()

    with tempfile.TemporaryDirectory() as td:
        tmp_dir = Path(td)
        for line in load_speech_lines():
            if not include_reactions and line.event not in _BASE_EVENTS:
                continue
            if line.audio is not None:
                continue
            spoken, lang = _spoken_and_lang(line)
            for bot_id in line.bots:
                if bot_id not in BOT_VOICE_PROFILES:
                    continue
                profile = BOT_VOICE_PROFILES[bot_id]
                mp3_path = tmp_dir / f"{bot_id}_{line.id}.mp3"
                await edge_tts.Communicate(
                    text=spoken, voice=profile.voice, rate=profile.rate, pitch=profile.pitch
                ).save(str(mp3_path))
                samples, sr = _decode_to_mono_wav(mp3_path, TARGET_VOICE_SAMPLE_RATE)
                trimmed = clip_and_normalize(
                    samples,
                    sr,
                    pad_ms=DEFAULT_PAD_MS,
                    fade_ms=DEFAULT_FADE_MS,
                    target_peak_dbfs=DEFAULT_TARGET_PEAK_DBFS,
                )
                dst = out_root / bot_id / f"{line.id}.wav"
                save_wav(dst, trimmed, sr, n_channels=1)
                records.append(
                    {
                        "rel_path": f"{bot_id}/{line.id}.wav",
                        "bot_id": bot_id,
                        "event": line.event,
                        "lang": lang,
                        "text": line.text,
                        "voice": f"{profile.voice} ({profile.rate}/{profile.pitch})",
                        "duration_sec": len(trimmed) / sr,
                        "date": today,
                    }
                )
    return records


def write_audio_readme(out_root: Path, records: list[dict[str, object]], backend: str) -> None:
    """Write assets/sounds/voice/README.md documenting voice clip provenance."""
    blunder_note = (
        f"- Direct blunder clip (`{BLUNDER_CLIP_PATH.name}`): sliced from "
        f"`{BLUNDER_SOURCE_AUDIO_PATH.name}` (`3.05s–5.32s`) using `others/clip_audio.py` and normalized to `-1.0 dBFS`."
    )
    if backend == "omnivoice-design":
        engine_note = (
            f"- Generated at build time using `others/generate_voice_clips.py --backend omnivoice-design` via "
            f"`{OMNIVOICE_MODEL_ID}` synthetic Filipino AI voice design (`{OMNIVOICE_DESIGN_INSTRUCT}`, seed `{OMNIVOICE_DESIGN_SEED}`, "
            f"native `ceb` / `fil` / `en` synthesis), with subtle per-bot vocal shading (`focus_ken` master, "
            f"`sick_ken` slower/warmer, `condition_ken` deeper) and `-1.0 dBFS` normalization via `others/clip_audio.py`."
        )
    elif backend == "omnivoice":
        engine_note = (
            f"- Generated at build time using `others/generate_voice_clips.py --backend omnivoice` via "
            f"`{OMNIVOICE_MODEL_ID}` zero-shot voice cloning from `{BLUNDER_SOURCE_AUDIO_PATH.name}` "
            f"(native `ceb` / `fil` / `en` synthesis), with subtle per-bot vocal shading (`focus_ken` master, "
            f"`sick_ken` slower/warmer, `condition_ken` deeper) and `-1.0 dBFS` normalization via `others/clip_audio.py`."
        )
    else:
        engine_note = (
            "- Generated at build time using `others/generate_voice_clips.py --backend edge` via "
            "`edge-tts` (`fil-PH-AngeloNeural`), normalized to `-1.0 dBFS` via `others/clip_audio.py`."
        )

    lines = [
        "# ChessIQ Voice Clips",
        "",
        "Per-bot voice-over WAV clips (`22050 Hz`, mono, 16-bit PCM) loaded by `ui/sound_manager.py`.",
        "",
        "## Generation & Provenance",
        engine_note,
        blunder_note,
        "- Audition any clip with: `.\\.venv\\Scripts\\python.exe others/play_clip.py <path-to-wav>`",
        "",
        "| File | Bot | Event | Lang | Duration | Date | Text |",
        "|---|---|---|---|---:|---|---|",
    ]
    if BLUNDER_CLIP_PATH.is_file():
        samples, sr, _ = load_wav(BLUNDER_CLIP_PATH)
        lines.append(
            f"| `{BLUNDER_CLIP_PATH.name}` | `all` | `PLAYER_BLUNDER` | `fil` | {len(samples) / sr:.2f}s | {datetime.date.today().isoformat()} | Walang kanin Buseng! |"
        )
    for rec in records:
        lines.append(
            f"| `{rec['rel_path']}` | `{rec['bot_id']}` | `{rec['event']}` | `{rec['lang']}` | {rec['duration_sec']:.2f}s | {rec['date']} | {rec['text']} |"
        )
    readme_path = out_root / "README.md"
    readme_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate ChessIQ bot voice-over WAV clips.")
    parser.add_argument("--backend", choices=("omnivoice-design", "omnivoice", "edge"), default="omnivoice-design")
    parser.add_argument("--base-only", action="store_true", help="Only generate greeting and base match events.")
    parser.add_argument("--ref-audio", type=Path, default=None, help="Optional pre-cleaned anchor/reference WAV for OmniVoice.")
    parser.add_argument("--cache-dir", type=Path, default=None, help="Directory to cache master OmniVoice WAVs (resume-safe).")
    parser.add_argument("--steps", type=int, default=16, help="OmniVoice diffusion steps (default: 16).")
    parser.add_argument("--force", action="store_true", help="Regenerate cached master WAVs.")
    args = parser.parse_args()

    include_reactions = not args.base_only
    if args.backend in ("omnivoice-design", "omnivoice"):
        records = generate_clips_omnivoice(
            VOICE_DIR,
            include_reactions=include_reactions,
            mode="design" if args.backend == "omnivoice-design" else "clone",
            ref_audio=args.ref_audio,
            master_cache_dir=args.cache_dir,
            num_step=args.steps,
            force=args.force,
        )
    else:
        records = asyncio.run(generate_clips_edge(VOICE_DIR, include_reactions=include_reactions))

    write_audio_readme(VOICE_DIR, records, backend=args.backend)
    print(f"Generated {len(records)} WAV clips in {VOICE_DIR} ({args.backend}) and updated README.md.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
