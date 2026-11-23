"""Unit tests for ui.sound_manager and extracted assets/sounds/*.wav clips."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch
import wave

import chess
import pygame

from others.config import SOUNDS_DIR
from ui.sound_manager import SOUND_NAMES, SoundManager, select_move_sound


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
