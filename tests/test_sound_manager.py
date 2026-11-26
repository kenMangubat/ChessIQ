"""Unit tests for ui.sound_manager and extracted assets/sounds/*.wav clips."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch
import wave

import chess
import pygame

from others.config import BG_MUSIC_PATH, SOUNDS_DIR
from ui.sound_manager import (
    DEFAULT_MUSIC_VOLUME_RATIO,
    SOUND_NAMES,
    SoundManager,
    select_move_sound,
)


def _apply(fen: str, uci: str) -> tuple[chess.Board, chess.Move, chess.Board]:
    board_before = chess.Board(fen)
    move = chess.Move.from_uci(uci)
    assert move in board_before.legal_moves, f"{uci} is not legal in {fen}"
    board_after = board_before.copy()
    board_after.push(move)
    return board_before, move, board_after


# ── Priority order tests ─────────────────────────────────────────────


def test_priority_checkmate_beats_promotion_and_capture() -> None:
    # White pawn on g7 captures rook on h8 and promotes to queen (defended by Be5) with checkmate
    fen = "5pkr/6P1/6K1/4B3/8/8/8/8 w - - 0 1"
    before, move, after = _apply(fen, "g7h8q")
    assert before.is_capture(move)
    assert move.promotion == chess.QUEEN
    assert after.is_checkmate()
    assert select_move_sound(before, move, after, is_player_move=True) == "checkmate"


def test_priority_stalemate_beats_capture() -> None:
    # Black king a8, black pawn b6; White king c7, white queen b1 -> Qxb6 is capture + stalemate
    before, move, after = _apply("k7/2K5/1p6/8/8/8/8/1Q6 w - - 0 1", "b1b6")
    assert before.is_capture(move)
    assert after.is_stalemate()
    assert select_move_sound(before, move, after, is_player_move=True) == "stalemate"


def test_priority_game_end_insufficient_material() -> None:
    # White king captures last black knight -> K vs K insufficient material draw
    before, move, after = _apply("7k/8/8/8/4n3/4K3/8/8 w - - 0 1", "e3e4")
    assert before.is_capture(move)
    assert not after.is_checkmate()
    assert not after.is_stalemate()
    assert after.is_game_over(claim_draw=True)
    assert select_move_sound(before, move, after, is_player_move=True) == "game_end"


def test_priority_check_beats_promotion_and_capture() -> None:
    # White pawn on e7 captures knight on d8 and promotes to Queen with check (not mate)
    before, move, after = _apply("3n2k1/4P3/8/8/8/8/8/4K3 w - - 0 1", "e7d8q")
    assert before.is_capture(move)
    assert move.promotion == chess.QUEEN
    assert after.is_check()
    assert not after.is_checkmate()
    assert select_move_sound(before, move, after, is_player_move=True) == "check"


def test_priority_promote_beats_capture() -> None:
    # White pawn captures rook on b8 and promotes without giving check (Black king on g7)
    before, move, after = _apply("1r6/P5k1/6p1/8/8/8/8/4K3 w - - 0 1", "a7b8q")
    assert before.is_capture(move)
    assert move.promotion == chess.QUEEN
    assert not after.is_check()
    assert select_move_sound(before, move, after, is_player_move=True) == "promote"


def test_priority_castle() -> None:
    before, move, after = _apply("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1", "e1g1")
    assert before.is_castling(move)
    assert not after.is_check()
    assert select_move_sound(before, move, after, is_player_move=True) == "castle"


def test_priority_capture_and_en_passant() -> None:
    # Normal capture
    before, move, after = _apply("rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2", "e4d5")
    assert before.is_capture(move)
    assert select_move_sound(before, move, after, is_player_move=True) == "capture"

    # En passant capture
    before_ep, move_ep, after_ep = _apply("rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3", "e5f6")
    assert before_ep.is_en_passant(move_ep)
    assert select_move_sound(before_ep, move_ep, after_ep, is_player_move=False) == "capture"


def test_priority_quiet_move_self_vs_opponent() -> None:
    before, move, after = _apply(chess.STARTING_FEN, "e2e4")
    assert select_move_sound(before, move, after, is_player_move=True) == "move_self"
    assert select_move_sound(before, move, after, is_player_move=False) == "move_opponent"


# ── Mute, volume, and headless/missing-audio fallback tests ──────────


def test_mute_toggle_and_volume_control() -> None:
    fake_sound = MagicMock()
    with patch("pygame.mixer.get_init", return_value=(44100, -16, 2)), patch(
        "pygame.mixer.Sound", return_value=fake_sound
    ):
        sm = SoundManager(SOUNDS_DIR, volume=0.8)
        assert sm.available is True
        assert sm.muted is False
        assert abs(sm.volume - 0.8) < 1e-6

        assert sm.play("move_self") is True
        assert fake_sound.play.call_count == 1

        # Mute blocks playback
        assert sm.toggle_mute() is True
        assert sm.muted is True
        assert sm.play("move_self") is False
        assert fake_sound.play.call_count == 1

        # Unmute restores playback
        assert sm.toggle_mute() is False
        assert sm.muted is False
        assert sm.play("move_self") is True
        assert fake_sound.play.call_count == 2

        # Volume clamping
        sm.set_volume(1.5)
        assert sm.volume == 1.0
        sm.set_volume(-0.2)
        assert sm.volume == 0.0


def test_headless_mixer_error_and_missing_dir_fallback(tmp_path: Path) -> None:
    with patch("pygame.mixer.get_init", return_value=None), patch(
        "pygame.mixer.init", side_effect=pygame.error("No audio device")
    ):
        sm = SoundManager(tmp_path / "nonexistent")
        assert sm.available is False
        assert sm.loaded_sounds == ()
        assert sm.play("game_start") is False
        before, move, after = _apply(chess.STARTING_FEN, "e2e4")
        assert sm.play_for_move(before, move, after, is_player_move=True) == "move_self"


def test_extracted_wav_files_exist_and_castle_keeps_both_clacks() -> None:
    assert SOUNDS_DIR.is_dir()
    for name in SOUND_NAMES:
        wav_path = SOUNDS_DIR / f"{name}.wav"
        assert wav_path.is_file(), f"Missing extracted sound: {wav_path}"
        assert wav_path.stat().st_size > 44, f"Empty WAV file: {wav_path}"
        with wave.open(str(wav_path), "rb") as wf:
            duration_ms = wf.getnframes() / wf.getframerate() * 1000.0
        assert duration_ms >= 40.0, f"{wav_path.name} too short: {duration_ms:.1f} ms"
        if name == "castle":
            # Double-clack castling cue is ~95 ms between clacks plus tail (>140 ms total)
            assert duration_ms >= 140.0, f"castle.wav appears split: {duration_ms:.1f} ms"


# ── Background music tests ───────────────────────────────────────────


def test_background_music_asset_exists() -> None:
    """Generated looping background music file must exist and be non-empty."""
    assert BG_MUSIC_PATH.is_file(), f"Missing background music: {BG_MUSIC_PATH}"
    assert BG_MUSIC_PATH.stat().st_size > 10_000, "Background music file is unexpectedly small"


def test_background_music_loop_volume_and_mute() -> None:
    """SoundManager starts, scales, mutes, unmutes, and stops pygame.mixer.music cleanly."""
    fake_sound = MagicMock()
    with patch("pygame.mixer.get_init", return_value=(44100, -16, 2)), patch(
        "pygame.mixer.Sound", return_value=fake_sound
    ), patch("pygame.mixer.music.load") as mock_load, patch(
        "pygame.mixer.music.play"
    ) as mock_play, patch("pygame.mixer.music.set_volume") as mock_vol, patch(
        "pygame.mixer.music.stop"
    ) as mock_stop:
        sm = SoundManager(SOUNDS_DIR, volume=0.8)
        assert sm.start_music() is True
        assert sm.music_playing is True
        mock_load.assert_called_once_with(str(BG_MUSIC_PATH))
        mock_play.assert_called_once_with(loops=-1, fade_ms=800)
        assert abs(mock_vol.call_args[0][0] - 0.8 * DEFAULT_MUSIC_VOLUME_RATIO) < 1e-6

        # Adjusting volume updates music proportionally beneath SFX
        sm.set_volume(0.5)
        assert abs(mock_vol.call_args[0][0] - 0.5 * DEFAULT_MUSIC_VOLUME_RATIO) < 1e-6

        # Muting silences music immediately; unmuting restores volume
        sm.toggle_mute()
        assert sm.music_playing is False
        assert mock_vol.call_args[0][0] == 0.0
        sm.toggle_mute()
        assert sm.music_playing is True
        assert abs(mock_vol.call_args[0][0] - 0.5 * DEFAULT_MUSIC_VOLUME_RATIO) < 1e-6

        sm.stop_music()
        assert sm.music_playing is False
        mock_stop.assert_called_once()


def test_background_music_headless_fallback(tmp_path: Path) -> None:
    """Background music degrades silently when audio device or file is absent."""
    with patch("pygame.mixer.get_init", return_value=None), patch(
        "pygame.mixer.init", side_effect=pygame.error("No audio device")
    ):
        sm = SoundManager(tmp_path, music_path=tmp_path / "missing.ogg")
        assert sm.start_music() is False
        assert sm.music_playing is False
        sm.stop_music()


def test_menu_music_stops_in_game_while_match_sfx_stay_active() -> None:
    """Menu music stays seamless across menu screens, fades out on game/time-control, and preserves SFX."""
    from main import App

    fake_sound = MagicMock()
    with patch("pygame.mixer.get_init", return_value=(44100, -16, 2)), patch(
        "pygame.mixer.Sound", return_value=fake_sound
    ), patch("pygame.mixer.music.load") as mock_load, patch(
        "pygame.mixer.music.play"
    ) as mock_play, patch("pygame.mixer.music.set_volume"), patch(
        "pygame.mixer.music.fadeout"
    ) as mock_fadeout:
        sm = SoundManager(SOUNDS_DIR, volume=0.7)
        app = App.__new__(App)
        app._sound_manager = sm
        app._settings_return_screen = "dashboard"

        # Dashboard starts menu music
        app._sync_menu_music("dashboard")
        assert sm.music_playing is True
        assert mock_load.call_count == 1
        assert mock_play.call_count == 1

        # Opponent select & menu-opened Settings keep menu music playing without restarting
        app._sync_menu_music("opponent_select")
        app._sync_menu_music("settings")
        assert sm.music_playing is True
        assert mock_load.call_count == 1
        assert mock_play.call_count == 1

        # Entering GameScreen (Time Control selection + match) fades out menu music
        app._sync_menu_music("game")
        assert sm.music_playing is False
        mock_fadeout.assert_called_once_with(400)

        # Chess match SFX (castle, check, capture, etc.) remain active while menu music is off
        for cue in ("castle", "check", "capture", "move_self"):
            assert sm.play(cue) is True
        assert fake_sound.play.call_count == 4

        # Settings opened from inside a match keeps menu music off
        app._settings_return_screen = "game"
        app._sync_menu_music("settings")
        assert sm.music_playing is False

        # Returning to dashboard resumes menu music smoothly
        app._settings_return_screen = "dashboard"
        app._sync_menu_music("dashboard")
        assert sm.music_playing is True
        assert mock_load.call_count == 2
        assert mock_play.call_count == 2


# ── Bot voice playback & clipping tests ──────────────────────────────


def test_voice_channel_volume_mute_missing_file_and_failure(tmp_path: Path) -> None:
    """Verify dedicated voice channel playback, caching, volume/mute sync, and silent fallbacks."""
    from others.config import VOICE_DIR

    fake_sound = MagicMock()
    fake_channel = MagicMock()
    fake_channel.get_busy.return_value = True

    with patch("pygame.mixer.get_init", return_value=(44100, -16, 2)), patch(
        "pygame.mixer.get_num_channels", return_value=8
    ), patch("pygame.mixer.set_reserved"), patch(
        "pygame.mixer.Channel", return_value=fake_channel
    ), patch("pygame.mixer.Sound", return_value=fake_sound):
        sm = SoundManager(SOUNDS_DIR, volume=0.75, voice_dir=VOICE_DIR)
        assert sm.play_voice("sick_ken", "greeting") is True
        assert fake_channel.play.call_count == 1
        assert sm.is_voice_playing() is True

        # Missing clip degrades silently without raising
        assert sm.play_voice("sick_ken", "nonexistent_line_xyz", explicit_audio="missing_file.wav") is False

        # Volume updates propagate to dedicated voice channel
        sm.set_volume(0.35)
        fake_channel.set_volume.assert_called_with(0.35)

        # Mute stops voice immediately and blocks further playback
        assert sm.toggle_mute() is True
        fake_channel.stop.assert_called()
        assert sm.play_voice("sick_ken", "greeting") is False

        # Playback error is caught cleanly without crashing
        sm.toggle_mute()
        fake_channel.play.side_effect = pygame.error("Device disconnected")
        assert sm.play_voice("focus_ken", "greeting") is False

    # Unavailable mixer never crashes
    with patch("pygame.mixer.get_init", return_value=None), patch(
        "pygame.mixer.init", side_effect=pygame.error("No audio device")
    ):
        sm_off = SoundManager(tmp_path, voice_dir=VOICE_DIR)
        assert sm_off.play_voice("sick_ken", "greeting") is False
        assert sm_off.is_voice_playing() is False
        sm_off.stop_voice()


def test_clip_audio_trims_synthetic_signal_without_clipping() -> None:
    """Verify others.clip_audio trims silence with ~70 ms padding and normalizes without clipping."""
    import numpy as np
    from others.clip_audio import (
        DEFAULT_FADE_MS,
        DEFAULT_PAD_MS,
        DEFAULT_TARGET_PEAK_DBFS,
        clip_and_normalize,
        detect_speech_segments,
    )

    sr = 22050
    sil = np.zeros((int(sr * 0.8), 1), dtype=np.float32)
    t = np.arange(int(sr * 1.0), dtype=np.float32) / sr
    tone = (0.25 * np.sin(2.0 * np.pi * 440.0 * t))[:, np.newaxis]
    signal = np.vstack([sil, tone, sil])

    segs, floor_db, has_bed = detect_speech_segments(signal, sr)
    assert len(segs) == 1
    assert has_bed is False
    assert abs(segs[0].start_sec - 0.8) <= 0.03
    assert abs(segs[0].end_sec - 1.8) <= 0.03

    clipped = clip_and_normalize(
        signal,
        sr,
        pad_ms=DEFAULT_PAD_MS,
        fade_ms=DEFAULT_FADE_MS,
        target_peak_dbfs=DEFAULT_TARGET_PEAK_DBFS,
    )
    dur = len(clipped) / sr
    expected_dur = 1.0 + 2.0 * (DEFAULT_PAD_MS / 1000.0)
    assert abs(dur - expected_dur) <= 0.04, f"Expected ~{expected_dur:.3f}s, got {dur:.3f}s"

    peak = float(np.max(np.abs(clipped)))
    target_lin = 10.0 ** (DEFAULT_TARGET_PEAK_DBFS / 20.0)
    assert 0.85 <= peak <= 0.92, f"Expected peak near {target_lin:.3f}, got {peak:.4f}"
    # Verify anti-click fades start and end cleanly at zero
    assert abs(float(clipped[0, 0])) < 1e-4
    assert abs(float(clipped[-1, 0])) < 1e-4


def test_generated_voice_clips_exist_and_are_valid_wav() -> None:
    """Every generated per-bot greeting, match, and Cebuano reaction WAV clip exists and has valid PCM audio."""
    from others.config import VOICE_DIR
    from ui.speech_bubble import load_speech_lines

    expected_lines = [line for line in load_speech_lines() if line.audio is None]
    assert len(expected_lines) >= 21

    for line in expected_lines:
        for bot_id in line.bots:
            wav_path = VOICE_DIR / bot_id / f"{line.id}.wav"
            assert wav_path.is_file(), f"Missing generated voice clip: {wav_path}"
            with wave.open(str(wav_path), "rb") as wf:
                assert wf.getnchannels() == 1
                assert wf.getsampwidth() == 2
                assert wf.getframerate() == 22050
                dur = wf.getnframes() / wf.getframerate()
            assert dur >= 0.45, f"Voice clip unexpectedly short ({dur:.2f}s): {wav_path}"
