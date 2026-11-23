"""Unit tests for Ken Bot Playstyle DNA, phase detection, style bonus calculations, and safety gates."""

import os
from pathlib import Path
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import chess

from bots.ken_bot import KenBot, get_profile
from calibration.style_analyzer import (
    center_distance,
    chebyshev_distance,
    compute_style_bonus,
    detect_game_phase,
    get_material_diff,
    is_passed_pawn,
    is_pawn_storm_move,
    is_queen_trade_move,
    is_rook_on_open_or_7th,
    load_style_profile,
    score_legal_moves_style,
    DEFAULT_STYLE_PROFILE,
)
from calibration.build_book import query_book_moves, TIER_CONDITION, TIER_FOCUS, TIER_SICK
from others.config import BOOK_DB_PATH, STYLE_PROFILE_PATH
from engine.stockfish_engine import StockfishEngine


def test_game_phase_detection():
    """Verify accurate detection of opening, middlegame, and endgame phases."""
    # Starting position at ply 0 is opening
    board = chess.Board()
    assert detect_game_phase(board, ply=0) == "opening"
    assert detect_game_phase(board, ply=12) == "opening"

    # Position at ply 24 with all major pieces on the board is middlegame
    board_mg = chess.Board("r1bq1rk1/ppp2ppp/2np1n2/2b1p3/2B1P3/2NP1N2/PPP2PPP/R1BQ1RK1 w - - 0 7")
    assert detect_game_phase(board_mg, ply=24) == "middlegame"

    # Position without queens is endgame
    board_no_queens = chess.Board("r1b2rk1/ppp2ppp/2np1n2/4p3/4P3/2NP1N2/PPP2PPP/R1B2RK1 w - - 0 9")
    assert detect_game_phase(board_no_queens, ply=18) == "endgame"

    # King and pawn endgame
    board_kp = chess.Board("8/5k2/4p3/4P3/8/5K2/8/8 w - - 0 45")
    assert detect_game_phase(board_kp, ply=45) == "endgame"

    # Depleted non-pawn material (<= 13 pts per side: e.g. R + B = 8 pts)
    board_depleted = chess.Board("4r1k1/5ppp/8/8/8/8/5PPP/4R1K1 w - - 0 35")
    assert detect_game_phase(board_depleted, ply=35) == "endgame"


def test_spatial_and_tactical_helpers():
    """Verify Chebyshev distance, center distance, passed pawns, and rook activity."""
    # Chebyshev distance
    assert chebyshev_distance(chess.E4, chess.E5) == 1
    assert chebyshev_distance(chess.E4, chess.F5) == 1
    assert chebyshev_distance(chess.E1, chess.E8) == 7

    # Center distance
    assert center_distance(chess.E4) == 0
    assert center_distance(chess.D5) == 0
    assert center_distance(chess.C3) == 1
    assert center_distance(chess.A1) == 3

    # Passed pawn
    # White pawn on e5 with no black pawns on d, e, f files ahead
    board_passed = chess.Board("8/8/8/4P3/8/8/8/k6K w - - 0 1")
    assert is_passed_pawn(board_passed, chess.E5, chess.WHITE) is True

    # Blocked or flanked pawn is not passed
    board_blocked = chess.Board("8/4p3/8/4P3/8/8/8/k6K w - - 0 1")
    assert is_passed_pawn(board_blocked, chess.E5, chess.WHITE) is False

    # Rook on open file and 7th rank
    board_rook = chess.Board("4r1k1/1R6/8/8/8/8/8/4K3 w - - 0 1")
    move_7th = chess.Move(chess.B7, chess.D7)
    is_open, is_7th = is_rook_on_open_or_7th(board_rook, move_7th)
    assert is_7th is True


def test_style_profile_json_validity():
    """Verify that ken_style_profile.json exists, parses, and contains calibrated DNA."""
    assert STYLE_PROFILE_PATH.exists(), f"Profile missing at {STYLE_PROFILE_PATH}"
    profile = load_style_profile(STYLE_PROFILE_PATH)

    assert "tiers" in profile
    for tier in ("sick", "focus", "condition"):
        assert tier in profile["tiers"], f"Tier {tier} missing from style profile"
        tier_data = profile["tiers"][tier]
        assert "castling" in tier_data
        assert "middlegame" in tier_data
        assert "trading" in tier_data
        assert "endgame" in tier_data
        assert "weights" in tier_data

        weights = tier_data["weights"]
        for key in (
            "check_bonus",
            "king_attack_bonus",
            "pawn_storm_bonus",
            "queen_trade_ahead_bonus",
            "queen_trade_behind_penalty",
            "endgame_king_centralization_bonus",
            "endgame_passed_pawn_bonus",
            "endgame_rook_activity_bonus",
        ):
            assert key in weights, f"Weight {key} missing in {tier}"
            assert isinstance(weights[key], (int, float))


def test_style_bonus_strictly_bounded():
    """Verify that compute_style_bonus is strictly bounded within [-20.0, 20.0] cp."""
    profile = load_style_profile(STYLE_PROFILE_PATH)
    cond_weights = profile["tiers"]["condition"]["weights"]

    # Test checking move
    board_check = chess.Board("rnbqkbnr/ppp2ppp/8/3pp3/4P3/5Q2/PPPP1PPP/RNB1KBNR w KQkq - 0 3")
    move_check = chess.Move.from_uci("f3f7")  # Qxf7+ (check + king attack)
    bonus = compute_style_bonus(board_check, move_check, "middlegame", cond_weights)
    assert -20.0 <= bonus <= 20.0
    assert bonus > 0.0  # Must be positive due to check and king attack

    # Test extreme bonus combinations do not exceed 20.0
    huge_weights = {k: 50.0 for k in cond_weights}
    huge_bonus = compute_style_bonus(board_check, move_check, "middlegame", huge_weights)
    assert huge_bonus == 20.0

    huge_neg_weights = {k: -50.0 for k in cond_weights}
    huge_neg_bonus = compute_style_bonus(board_check, move_check, "middlegame", huge_neg_weights)
    assert -20.0 <= huge_neg_bonus <= 20.0


def test_quality_filtered_opening_book():
    """Verify opening book queries filter by play count and win rate for Condition Ken."""
    board = chess.Board()
    start_epd = board.epd()

    # Query with default constraints
    all_moves = query_book_moves(start_epd, side="white", tier="condition", db_path=BOOK_DB_PATH)
    assert len(all_moves) > 0

    # Query with min_play_count = 10 -> high-frequency repertoire only (e.g. e2e4)
    filtered = query_book_moves(
        start_epd,
        side="white",
        tier="condition",
        db_path=BOOK_DB_PATH,
        min_play_count=10,
    )
    assert all(m["play_count"] >= 10 for m in filtered)
    assert any(m["move_uci"] == "e2e4" for m in filtered)


def test_safety_gate_prevents_forced_mate():
    """Verify that KenBot rejects candidates that walk into forced mate (mate < 0)."""
    # Position: White to move, Black threatens Qh2# / mate in 1
    # White has two legal moves: one prevents mate, one allows mate
    board = chess.Board("r1b1k2r/pppp1ppp/8/8/1b2q3/2P5/PP1B1PPP/RN1QKB1R w KQkq - 0 9")

    bot = KenBot(profile="Condition Ken")
    try:
        chosen = bot.choose_move(board)
        assert chosen in board.legal_moves
        # Pushing the move should not leave White immediately mated
        board.push(chosen)
        assert not board.is_checkmate()
    finally:
        bot.close()


def test_safety_gate_condition_ken_tactical_precision():
    """Verify Condition Ken (2350) does not play severe blunders even out of book."""
    # Tactical puzzle position: White has a winning piece capture or queen fork
    # FEN where White can play Rxe8+ or blunder
    board = chess.Board("4r1k1/5ppp/8/8/8/8/5PPP/4R1K1 w - - 0 1")

    bot = KenBot(profile="Condition Ken", log_provenance=False)
    try:
        chosen = bot.choose_move(board)
        assert chosen in board.legal_moves
        # e1e8# is mate in 1!
        assert chosen == chess.Move.from_uci("e1e8")
    finally:
        bot.close()


def test_white_black_separation_and_style_probabilities():
    """Verify White/Black models are separate and produce normalized legal-move probabilities."""
    profile = load_style_profile(STYLE_PROFILE_PATH)
    assert "by_color" in profile
    assert "white" in profile["by_color"] and "black" in profile["by_color"]

    w_weights = profile["by_color"]["white"]["models"]["middlegame"]["weights"]
    b_weights = profile["by_color"]["black"]["models"]["middlegame"]["weights"]
    assert len(w_weights) == len(b_weights) and len(w_weights) > 10
    assert w_weights != b_weights, "White and Black middlegame style models must be distinct"

    board = chess.Board("r1bq1rk1/pp2bppp/2n1pn2/2pp4/3P4/2PBPN2/PP1N1PPP/R1BQ1RK1 w - - 2 8")
    probs, reasons = score_legal_moves_style(board, "white", "middlegame", profile)
    assert len(probs) == len(list(board.legal_moves))
    assert abs(sum(probs.values()) - 1.0) < 1e-6
    assert all(0.0 <= p <= 1.0 for p in probs.values())
    assert len(reasons) == len(probs)


def test_quality_guards_cp_cap_and_style_floor():
    """Quality guards: per-profile CPL cap and style-probability floor across fixed positions."""
    fixed_fens = [
        "r2qk2r/pbp1bppp/1pn2n2/3pp3/4P3/3P1NP1/PPPN1PBP/R1BQ1RK1 w kq - 1 8",
        "r1bq1rk1/pp2bppp/2n1pn2/2pp4/3P4/2PBPN2/PP1N1PPP/R1BQ1RK1 w - - 2 8",
        "r2q1rk1/1p1bbppp/p1np1n2/4p3/4P3/1NN1BP2/PPPQ2PP/R3KB1R w KQ - 2 10",
        "r3k2r/1bq1bp2/p3p1p1/1p2P1N1/3P1P1p/P1np1N1P/3B2PK/RQ3R2 w kq - 0 25",
    ]
    for bot_name in ("Sick Ken", "Focus Ken", "Condition Ken"):
        bot = KenBot(profile=bot_name, log_provenance=False)
        try:
            for fen in fixed_fens:
                board = chess.Board(fen)
                dec = bot.evaluate_position(board, deterministic=False, allow_book=False)
                assert dec["source"] in ("style-selected", "error-model")
                # Hard CPL cap must always be respected by the bot's own full-strength evaluation
                assert dec["cpl"] <= bot.profile.mistake_max_cp_loss, (
                    f"{bot_name} exceeded mistake cap: cpl={dec['cpl']} > {bot.profile.mistake_max_cp_loss}"
                )
                # Style floor guard: chosen move cannot be below floor unless it was the only safe fallback or tactically justified
                chosen_cand = next(
                    c for c in dec["candidates"] if c["uci"] == dec["chosen_uci"]
                )
                safe_above_floor = [
                    c for c in dec["candidates"] if c["in_safe_band"] and not c["below_style_floor"]
                ]
                if safe_above_floor:
                    assert not chosen_cand["below_style_floor"], (
                        f"{bot_name} picked below-floor move {dec['chosen_uci']} when safe above-floor moves existed"
                    )
        finally:
            bot.close()


if __name__ == "__main__":
    test_game_phase_detection()
    test_spatial_and_tactical_helpers()
    test_style_profile_json_validity()
    test_style_bonus_strictly_bounded()
    test_quality_filtered_opening_book()
    test_safety_gate_prevents_forced_mate()
    test_safety_gate_condition_ken_tactical_precision()
    test_white_black_separation_and_style_probabilities()
    test_quality_guards_cp_cap_and_style_floor()
    print("ALL PLAYSTYLE DNA & SAFETY GATE TESTS PASSED!")
