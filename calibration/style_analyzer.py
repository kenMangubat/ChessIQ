"""Playstyle, Move-Level Style Model & Mistake Analyzer for Ken Bot personalities.

Trains White/Black separated conditional-logit move-level style models and
smoothed piece-square priors from Ken's ingested games in SQLite (`ken_book.db`),
runs incremental offline Stockfish mistake profiling, evaluates hold-out
generalization against a plain Stockfish baseline, and provides bounded
style bonuses and legal-move style probability inference for KenBot.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import datetime
import json
import logging
import math
from pathlib import Path
import random
import sqlite3
import time
from typing import Any

import chess
import chess.pgn
import numpy as np

from calibration.build_book import (
    KEN_USERNAMES,
    TIER_CONDITION,
    TIER_FOCUS,
    TIER_SICK,
    TIERS,
    determine_tier,
    identify_ken_in_game,
    init_book_db,
    query_book_moves,
)
from others.config import BOOK_DB_PATH, PGN_FILES, STYLE_PROFILE_PATH
from engine.stockfish_engine import StockfishEngine

logger = logging.getLogger(__name__)

# Center and flank squares for spatial evaluations
CENTER_SQUARES: frozenset[chess.Square] = frozenset(
    {chess.D4, chess.E4, chess.D5, chess.E5}
)
EXTENDED_CENTER_SQUARES: frozenset[chess.Square] = frozenset(
    {
        chess.C3, chess.D3, chess.E3, chess.F3,
        chess.C4, chess.D4, chess.E4, chess.F4,
        chess.C5, chess.D5, chess.E5, chess.F5,
        chess.C6, chess.D6, chess.E6, chess.F6,
    }
)
LONG_DIAGONAL_SQUARES: frozenset[chess.Square] = frozenset(
    {
        chess.A1, chess.B2, chess.C3, chess.D4, chess.E5, chess.F6, chess.G7, chess.H8,
        chess.H1, chess.G2, chess.F3, chess.E4, chess.D5, chess.C6, chess.B7, chess.A8,
    }
)

PIECE_VALUES: dict[chess.PieceType, int] = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 0,
}

PHASES: tuple[str, ...] = ("opening", "middlegame", "endgame")
COLORS: tuple[str, ...] = ("white", "black")
ERROR_TYPES: tuple[str, ...] = (
    "hanging_piece",
    "missed_tactic",
    "positional_drift",
    "premature_attack",
    "endgame_technique",
)

# Ordered feature vector names used by the conditional logit move-level style model.
# Every feature here is extracted at training AND inference time.
STYLE_FEATURE_NAMES: tuple[str, ...] = (
    "piece_pawn",
    "piece_knight",
    "piece_bishop",
    "piece_rook",
    "piece_queen",
    "piece_king",
    "is_capture",
    "is_equal_trade",
    "is_winning_capture",
    "is_pawn_capture",
    "is_recapture",
    "gives_check",
    "is_kingside_castle",
    "is_queenside_castle",
    "is_minor_development",
    "is_fianchetto",
    "is_center_pawn_push",
    "is_flank_pawn_push",
    "is_prophylaxis",
    "is_pawn_storm",
    "king_ring_attack",
    "queen_trade_ahead",
    "queen_trade_behind",
    "queen_trade_equal",
    "knight_centralize",
    "bishop_long_diag",
    "rook_open_file",
    "rook_7th_rank",
    "king_centralize",
    "passed_pawn_push",
    "move_forward",
    "dest_center_prox",
    "to_undefended_attacked",
    "creates_doubled_pawn",
    "repeats_own_piece",
    "psq_log_prior",
    # ── v2.1 rich human behavioral & tactical-style features ──
    "saves_attacked_piece",
    "attacks_enemy_piece",
    "attacks_undefended_enemy",
    "recapture_towards_center",
    "capture_high_value_victim",
    "is_losing_capture",
    "is_retreat_unprovoked",
    "is_retreat_from_attack",
    "develops_queen_early",
    "connects_rooks",
    "doubles_rooks_or_battery",
    "trading_minor_for_minor",
    "from_to_log_prior",
)

NUM_STYLE_FEATURES: int = len(STYLE_FEATURE_NAMES)
_EVAL_CLAMP_CP = 1000


def chebyshev_distance(sq1: chess.Square, sq2: chess.Square) -> int:
    """Chebyshev (chessboard king) distance between two squares."""
    return max(
        abs(chess.square_file(sq1) - chess.square_file(sq2)),
        abs(chess.square_rank(sq1) - chess.square_rank(sq2)),
    )


def center_distance(sq: chess.Square) -> int:
    """Minimum distance from a square to the central 4 squares {d4, d5, e4, e5}."""
    return min(chebyshev_distance(sq, c) for c in CENTER_SQUARES)


def effective_ply(board: chess.Board) -> int:
    """Return best estimate of current ply count even for FEN-initialized boards."""
    ply_from_stack = board.ply()
    ply_from_fullmove = max(0, (board.fullmove_number - 1) * 2 + (0 if board.turn == chess.WHITE else 1))
    return max(ply_from_stack, ply_from_fullmove)


def detect_game_phase(board: chess.Board, ply: int | None = None) -> str:
    """Determine whether a board position is in opening, middlegame, or endgame.

    - Endgame: queens are off OR non-pawn material <= 13 points per side.
    - Opening: early game (ply <= 16, or ply <= 20 with major material intact)
      and not an endgame.
    - Middlegame: between opening and endgame.
    """
    eff_ply = effective_ply(board) if ply is None else ply
    white_queens = len(board.pieces(chess.QUEEN, chess.WHITE))
    black_queens = len(board.pieces(chess.QUEEN, chess.BLACK))

    white_np_mat = (
        3 * len(board.pieces(chess.KNIGHT, chess.WHITE))
        + 3 * len(board.pieces(chess.BISHOP, chess.WHITE))
        + 5 * len(board.pieces(chess.ROOK, chess.WHITE))
        + 9 * white_queens
    )
    black_np_mat = (
        3 * len(board.pieces(chess.KNIGHT, chess.BLACK))
        + 3 * len(board.pieces(chess.BISHOP, chess.BLACK))
        + 5 * len(board.pieces(chess.ROOK, chess.BLACK))
        + 9 * black_queens
    )

    if (white_queens == 0 and black_queens == 0) or (
        white_np_mat <= 13 and black_np_mat <= 13
    ):
        return "endgame"

    if eff_ply <= 16:
        return "opening"

    if eff_ply <= 20 and (white_np_mat + black_np_mat >= 50):
        return "opening"

    return "middlegame"


def get_material_diff(board: chess.Board, color: chess.Color) -> int:
    """Compute material advantage in centipawns for the specified color."""
    my_mat = 0
    opp_mat = 0
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece is not None:
            val = PIECE_VALUES.get(piece.piece_type, 0)
            if piece.color == color:
                my_mat += val
            else:
                opp_mat += val
    return my_mat - opp_mat


def is_passed_pawn(board: chess.Board, square: chess.Square, color: chess.Color) -> bool:
    """Return True if the pawn at square is passed (no opposing pawns blocking or flanking)."""
    piece = board.piece_at(square)
    if not piece or piece.piece_type != chess.PAWN or piece.color != color:
        return False

    file_idx = chess.square_file(square)
    rank_idx = chess.square_rank(square)
    enemy_pawns = board.pieces(chess.PAWN, not color)

    for enemy_sq in enemy_pawns:
        e_file = chess.square_file(enemy_sq)
        e_rank = chess.square_rank(enemy_sq)
        if abs(e_file - file_idx) <= 1:
            if color == chess.WHITE and e_rank > rank_idx:
                return False
            if color == chess.BLACK and e_rank < rank_idx:
                return False
    return True


def is_pawn_storm_move(board: chess.Board, move: chess.Move) -> bool:
    """Check if a move is a flank pawn advancing against an enemy castled king."""
    piece = board.piece_at(move.from_square)
    if not piece or piece.piece_type != chess.PAWN:
        return False

    opp_color = not board.turn
    opp_king_sq = board.king(opp_color)
    if opp_king_sq is None:
        return False

    opp_king_file = chess.square_file(opp_king_sq)
    from_file = chess.square_file(move.from_square)
    to_rank = chess.square_rank(move.to_square)
    from_rank = chess.square_rank(move.from_square)

    is_advancing = (to_rank > from_rank) if board.turn == chess.WHITE else (to_rank < from_rank)
    if not is_advancing:
        return False

    if opp_king_file >= 5 and from_file >= 5:
        return True
    if opp_king_file <= 2 and from_file <= 2:
        return True
    return False


def is_queen_trade_move(board: chess.Board, move: chess.Move) -> bool:
    """Check if a move initiates or accepts a Queen trade."""
    piece = board.piece_at(move.from_square)
    if not piece:
        return False

    opp_queens = board.pieces(chess.QUEEN, not board.turn)
    if not opp_queens:
        return False

    if move.to_square in opp_queens:
        return True

    if piece.piece_type == chess.QUEEN:
        for q_sq in opp_queens:
            if q_sq == move.to_square:
                return True
            if move.to_square in chess.SquareSet(board.attacks(q_sq)):
                return True

    return False


def is_rook_on_open_or_7th(board: chess.Board, move: chess.Move) -> tuple[bool, bool]:
    """Determine if a rook move places it on an open/semi-open file or the 7th/2nd rank."""
    piece = board.piece_at(move.from_square)
    if not piece or piece.piece_type != chess.ROOK:
        return False, False

    to_file = chess.square_file(move.to_square)
    to_rank = chess.square_rank(move.to_square)
    is_7th = to_rank == 6 if board.turn == chess.WHITE else to_rank == 1

    friendly_pawns = [
        sq for sq in board.pieces(chess.PAWN, board.turn) if chess.square_file(sq) == to_file
    ]
    is_open_or_semi = len(friendly_pawns) == 0
    return is_open_or_semi, is_7th


def _empty_psq_log_prior() -> list[list[float]]:
    """Return a neutral (all-zero) 6x64 piece-square log-prior table."""
    return [[0.0] * 64 for _ in range(6)]


def _default_mistake_phase() -> dict[str, Any]:
    return {
        "analyzed_moves": 0,
        "avg_cpl": 45.0,
        "mistake_rate_60_150": 0.12,
        "mistake_rate_150_300": 0.05,
        "blunder_rate_300plus": 0.02,
        "error_type_dist": {
            "hanging_piece": 0.18,
            "missed_tactic": 0.28,
            "positional_drift": 0.34,
            "premature_attack": 0.12,
            "endgame_technique": 0.08,
        },
    }


def _default_color_block() -> dict[str, Any]:
    return {
        "games_count": 0,
        "otb_games": 0,
        "phase_moves": {"opening": 0, "middlegame": 0, "endgame": 0},
        "models": {
            phase: {
                "weights": [0.0] * NUM_STYLE_FEATURES,
                "psq_log_prior": _empty_psq_log_prior(),
                "train_positions": 0,
            }
            for phase in PHASES
        },
        "mistake_profile": {phase: _default_mistake_phase() for phase in PHASES},
        "human_engine_agreement": {
            phase: {"top1_rate": 0.42 if phase != "opening" else 0.25, "analyzed_moves": 0}
            for phase in PHASES
        },
    }


# Default neutral baseline style profile (backward-compatible + v2 fields)
DEFAULT_STYLE_PROFILE: dict[str, Any] = {
    "version": "2.0",
    "updated_at": "default",
    "feature_names": list(STYLE_FEATURE_NAMES),
    "by_color": {
        "white": _default_color_block(),
        "black": _default_color_block(),
    },
    "tiers": {
        TIER_SICK: {
            "castling": {"kingside_rate": 0.84, "queenside_rate": 0.08, "uncastled_rate": 0.08, "avg_ply": 13.5},
            "middlegame": {
                "check_rate": 0.088,
                "capture_rate": 0.240,
                "king_attack_rate": 0.032,
                "pawn_storm_rate": 0.025,
                "knight_ratio": 0.520,
            },
            "trading": {
                "queen_trade_ahead_rate": 0.70,
                "queen_trade_behind_rate": 0.45,
                "late_knight_retention": 0.51,
            },
            "endgame": {
                "king_move_rate": 0.18,
                "avg_king_center_dist": 1.7,
                "passed_pawn_push_rate": 0.28,
                "rook_open_file_rate": 0.35,
                "rook_7th_rank_rate": 0.12,
            },
            "weights": {
                "check_bonus": 6.0,
                "king_attack_bonus": 8.0,
                "pawn_storm_bonus": 5.0,
                "knight_bias": 2.0,
                "bishop_bias": -1.0,
                "queen_trade_ahead_bonus": 8.0,
                "queen_trade_behind_penalty": -8.0,
                "endgame_king_centralization_bonus": 6.0,
                "endgame_passed_pawn_bonus": 7.0,
                "endgame_rook_activity_bonus": 5.0,
            },
        },
        TIER_FOCUS: {
            "castling": {"kingside_rate": 0.85, "queenside_rate": 0.09, "uncastled_rate": 0.06, "avg_ply": 13.0},
            "middlegame": {
                "check_rate": 0.070,
                "capture_rate": 0.221,
                "king_attack_rate": 0.026,
                "pawn_storm_rate": 0.022,
                "knight_ratio": 0.523,
            },
            "trading": {
                "queen_trade_ahead_rate": 0.75,
                "queen_trade_behind_rate": 0.38,
                "late_knight_retention": 0.52,
            },
            "endgame": {
                "king_move_rate": 0.20,
                "avg_king_center_dist": 1.5,
                "passed_pawn_push_rate": 0.35,
                "rook_open_file_rate": 0.42,
                "rook_7th_rank_rate": 0.15,
            },
            "weights": {
                "check_bonus": 4.0,
                "king_attack_bonus": 7.0,
                "pawn_storm_bonus": 4.0,
                "knight_bias": 2.0,
                "bishop_bias": -1.0,
                "queen_trade_ahead_bonus": 10.0,
                "queen_trade_behind_penalty": -10.0,
                "endgame_king_centralization_bonus": 7.0,
                "endgame_passed_pawn_bonus": 9.0,
                "endgame_rook_activity_bonus": 6.0,
            },
        },
        TIER_CONDITION: {
            "castling": {"kingside_rate": 0.86, "queenside_rate": 0.08, "uncastled_rate": 0.06, "avg_ply": 12.5},
            "middlegame": {
                "check_rate": 0.064,
                "capture_rate": 0.223,
                "king_attack_rate": 0.023,
                "pawn_storm_rate": 0.020,
                "knight_ratio": 0.522,
            },
            "trading": {
                "queen_trade_ahead_rate": 0.82,
                "queen_trade_behind_rate": 0.30,
                "late_knight_retention": 0.53,
            },
            "endgame": {
                "king_move_rate": 0.22,
                "avg_king_center_dist": 1.3,
                "passed_pawn_push_rate": 0.42,
                "rook_open_file_rate": 0.48,
                "rook_7th_rank_rate": 0.18,
            },
            "weights": {
                "check_bonus": 3.0,
                "king_attack_bonus": 6.0,
                "pawn_storm_bonus": 4.0,
                "knight_bias": 1.5,
                "bishop_bias": -0.8,
                "queen_trade_ahead_bonus": 12.0,
                "queen_trade_behind_penalty": -12.0,
                "endgame_king_centralization_bonus": 8.0,
                "endgame_passed_pawn_bonus": 10.0,
                "endgame_rook_activity_bonus": 7.0,
            },
        },
    },
}


# ── Fast Move-Level Feature Extraction ───────────────────────────────


def _build_board_context(
    board: chess.Board,
    prev_own_move: chess.Move | None = None,
    prev_opp_move: chess.Move | None = None,
) -> dict[str, Any]:
    """Precompute shared board context once per position for fast legal-move scoring."""
    turn = board.turn
    opp = not turn
    opp_king_sq = board.king(opp)
    own_king_sq = board.king(turn)
    opp_queens = board.pieces(chess.QUEEN, opp)

    opp_queen_attacks = 0
    for q_sq in opp_queens:
        opp_queen_attacks |= board.attacks_mask(q_sq)

    enemy_attacks = 0
    for sq in chess.scan_forward(board.occupied_co[opp]):
        enemy_attacks |= board.attacks_mask(sq)

    friendly_attacks = 0
    for sq in chess.scan_forward(board.occupied_co[turn]):
        friendly_attacks |= board.attacks_mask(sq)

    friendly_pawn_files: dict[int, int] = {}
    passed_pawn_squares: set[int] = set()
    enemy_pawns_list = list(board.pieces(chess.PAWN, opp))

    for sq in board.pieces(chess.PAWN, turn):
        f = chess.square_file(sq)
        r = chess.square_rank(sq)
        friendly_pawn_files[f] = friendly_pawn_files.get(f, 0) + 1
        passed = True
        for e_sq in enemy_pawns_list:
            if abs(chess.square_file(e_sq) - f) <= 1:
                e_r = chess.square_rank(e_sq)
                if (turn == chess.WHITE and e_r > r) or (turn == chess.BLACK and e_r < r):
                    passed = False
                    break
        if passed:
            passed_pawn_squares.add(sq)

    # Compute material diff fast from bitboards
    my_mat = (
        100 * len(board.pieces(chess.PAWN, turn))
        + 320 * len(board.pieces(chess.KNIGHT, turn))
        + 330 * len(board.pieces(chess.BISHOP, turn))
        + 500 * len(board.pieces(chess.ROOK, turn))
        + 900 * len(board.pieces(chess.QUEEN, turn))
    )
    opp_mat = (
        100 * len(board.pieces(chess.PAWN, opp))
        + 320 * len(board.pieces(chess.KNIGHT, opp))
        + 330 * len(board.pieces(chess.BISHOP, opp))
        + 500 * len(board.pieces(chess.ROOK, opp))
        + 900 * len(board.pieces(chess.QUEEN, opp))
    )

    enemy_major_minor_mask = (
        board.pieces_mask(chess.KNIGHT, opp)
        | board.pieces_mask(chess.BISHOP, opp)
        | board.pieces_mask(chess.ROOK, opp)
        | board.pieces_mask(chess.QUEEN, opp)
    )
    enemy_undefended_major_minor = enemy_major_minor_mask & (~enemy_attacks)
    friendly_heavy_files: dict[int, int] = {}
    for pt_h in (chess.ROOK, chess.QUEEN):
        for sq in board.pieces(pt_h, turn):
            f_h = chess.square_file(sq)
            friendly_heavy_files[f_h] = friendly_heavy_files.get(f_h, 0) + 1
    back_rank_idx = 0 if turn == chess.WHITE else 7
    back_rank_between_rooks = 0
    own_rooks = list(board.pieces(chess.ROOK, turn))
    if len(own_rooks) == 2:
        r_files = sorted(chess.square_file(sq) for sq in own_rooks)
        if all(chess.square_rank(sq) == back_rank_idx for sq in own_rooks):
            for f_bt in range(r_files[0] + 1, r_files[1]):
                sq_bt = chess.square(f_bt, back_rank_idx)
                if board.piece_at(sq_bt) is not None:
                    back_rank_between_rooks |= chess.BB_SQUARES[sq_bt]

    return {
        "turn": turn,
        "opp": opp,
        "mat_diff": my_mat - opp_mat,
        "opp_king_sq": opp_king_sq,
        "own_king_sq": own_king_sq,
        "opp_king_file": chess.square_file(opp_king_sq) if opp_king_sq is not None else -1,
        "own_king_file": chess.square_file(own_king_sq) if own_king_sq is not None else -1,
        "opp_queens": opp_queens,
        "opp_queen_attacks": opp_queen_attacks,
        "enemy_attacks": enemy_attacks,
        "friendly_attacks": friendly_attacks,
        "friendly_pawn_files": friendly_pawn_files,
        "passed_pawn_squares": passed_pawn_squares,
        "prev_own_to": prev_own_move.to_square if prev_own_move is not None else -1,
        "prev_opp_to": prev_opp_move.to_square if prev_opp_move is not None else -1,
        "enemy_major_minor_mask": enemy_major_minor_mask,
        "enemy_undefended_major_minor": enemy_undefended_major_minor,
        "friendly_heavy_files": friendly_heavy_files,
        "back_rank_between_rooks": back_rank_between_rooks,
        "eff_ply": effective_ply(board),
    }


def extract_legal_move_features(
    board: chess.Board,
    legal_moves: list[chess.Move],
    psq_log_prior: list[list[float]] | None = None,
    prev_own_move: chess.Move | None = None,
    prev_opp_move: chess.Move | None = None,
    from_to_log_prior: dict[str, float] | None = None,
) -> np.ndarray:
    """Extract a (len(legal_moves), NUM_STYLE_FEATURES) float64 feature matrix."""
    n = len(legal_moves)
    feats = np.zeros((n, NUM_STYLE_FEATURES), dtype=np.float64)
    if n == 0:
        return feats

    ctx = _build_board_context(board, prev_own_move=prev_own_move, prev_opp_move=prev_opp_move)
    turn = ctx["turn"]
    is_white = turn == chess.WHITE
    back_rank = 0 if is_white else 7
    seventh_rank = 6 if is_white else 1
    opp_king_sq = ctx["opp_king_sq"]
    opp_king_file = ctx["opp_king_file"]
    own_king_file = ctx["own_king_file"]
    opp_queens = ctx["opp_queens"]
    opp_queen_attacks = ctx["opp_queen_attacks"]
    enemy_attacks = ctx["enemy_attacks"]
    friendly_attacks = ctx["friendly_attacks"]
    friendly_pawn_files = ctx["friendly_pawn_files"]
    passed_pawn_squares = ctx["passed_pawn_squares"]
    mat_diff = ctx["mat_diff"]
    prev_own_to = ctx["prev_own_to"]
    prev_opp_to = ctx["prev_opp_to"]
    enemy_major_minor_mask = ctx["enemy_major_minor_mask"]
    enemy_undefended_major_minor = ctx["enemy_undefended_major_minor"]
    friendly_heavy_files = ctx["friendly_heavy_files"]
    back_rank_between_rooks = ctx["back_rank_between_rooks"]
    eff_ply = ctx["eff_ply"]

    for i, move in enumerate(legal_moves):
        from_sq = move.from_square
        to_sq = move.to_square
        piece = board.piece_at(from_sq)
        if piece is None:
            continue

        pt = piece.piece_type
        pt_idx = pt - 1  # 0..5 for P, N, B, R, Q, K
        feats[i, pt_idx] = 1.0

        from_file = chess.square_file(from_sq)
        from_rank = chess.square_rank(from_sq)
        to_file = chess.square_file(to_sq)
        to_rank = chess.square_rank(to_sq)

        # Captures & trading
        is_cap = board.is_capture(move)
        if is_cap:
            feats[i, 6] = 1.0
            victim = board.piece_at(to_sq)
            victim_val = PIECE_VALUES.get(victim.piece_type, 100) if victim else 100  # en passant = pawn
            att_val = PIECE_VALUES.get(pt, 0)
            if abs(victim_val - att_val) <= 50:
                feats[i, 7] = 1.0
            if victim_val > att_val + 50 or not ((enemy_attacks >> to_sq) & 1):
                feats[i, 8] = 1.0
            if pt == chess.PAWN:
                feats[i, 9] = 1.0
            if to_sq == prev_opp_to:
                feats[i, 10] = 1.0

        # Checks
        if board.gives_check(move):
            feats[i, 11] = 1.0

        # Castling
        if pt == chess.KING and abs(to_file - from_file) == 2:
            if to_file == 6:
                feats[i, 12] = 1.0
            else:
                feats[i, 13] = 1.0

        # Minor development off back rank
        if pt in (chess.KNIGHT, chess.BISHOP) and from_rank == back_rank and to_rank != back_rank:
            feats[i, 14] = 1.0

        # Fianchetto (bishop to g2/b2/g7/b7 or g3/b3/g6/b6 pawn setup)
        if pt == chess.BISHOP and to_sq in (
            (chess.G2, chess.B2) if is_white else (chess.G7, chess.B7)
        ):
            feats[i, 15] = 1.0
        elif pt == chess.PAWN and to_sq in (
            (chess.G3, chess.B3) if is_white else (chess.G6, chess.B6)
        ):
            feats[i, 15] = 1.0

        # Pawn pushes: center, flank, prophylaxis, pawn storm, passed pawn, doubled pawn
        if pt == chess.PAWN:
            if to_sq in (chess.D4, chess.E4, chess.D5, chess.E5, chess.C4, chess.C5):
                feats[i, 16] = 1.0
            if from_file in (0, 1, 6, 7):
                feats[i, 17] = 1.0
            # Prophylaxis h3/a3 (white) or h6/a6 (black)
            if (is_white and to_sq in (chess.H3, chess.A3)) or (
                not is_white and to_sq in (chess.H6, chess.A6)
            ):
                feats[i, 18] = 1.0
            # Pawn storm against castled enemy king
            advancing = (to_rank > from_rank) if is_white else (to_rank < from_rank)
            if advancing and opp_king_file != -1:
                if (opp_king_file >= 5 and from_file >= 5) or (opp_king_file <= 2 and from_file <= 2):
                    feats[i, 19] = 1.0
            if from_sq in passed_pawn_squares:
                feats[i, 29] = 1.0
            if is_cap and friendly_pawn_files.get(to_file, 0) > 0:
                feats[i, 33] = 1.0
        elif pt == chess.KING:
            # Prophylactic king tuck Kh1/Kb1 or Kh8/Kb8 after castling
            if (is_white and from_sq == chess.G1 and to_sq == chess.H1) or (
                not is_white and from_sq == chess.G8 and to_sq == chess.H8
            ):
                feats[i, 18] = 1.0
            if center_distance(to_sq) < center_distance(from_sq):
                feats[i, 28] = 1.0

        # King ring attack
        if opp_king_sq is not None and chebyshev_distance(to_sq, opp_king_sq) <= 1:
            feats[i, 20] = 1.0

        # Queen trade context
        if opp_queens:
            is_qt = (to_sq in opp_queens) or (
                pt == chess.QUEEN and bool((opp_queen_attacks >> to_sq) & 1)
            )
            if is_qt:
                if mat_diff > 50:
                    feats[i, 21] = 1.0
                elif mat_diff < -50:
                    feats[i, 22] = 1.0
                else:
                    feats[i, 23] = 1.0

        # Knight centralization & bishop long diagonal
        if pt == chess.KNIGHT and center_distance(to_sq) < center_distance(from_sq):
            feats[i, 24] = 1.0
        if pt == chess.BISHOP and to_sq in LONG_DIAGONAL_SQUARES:
            feats[i, 25] = 1.0

        # Rook open/semi-open file & 7th rank
        if pt == chess.ROOK:
            if friendly_pawn_files.get(to_file, 0) == 0:
                feats[i, 26] = 1.0
            if to_rank == seventh_rank:
                feats[i, 27] = 1.0

        # Forward direction (+1 forward, 0 lateral, -1 backward)
        rank_delta = (to_rank - from_rank) if is_white else (from_rank - to_rank)
        if rank_delta > 0:
            feats[i, 30] = 1.0
        elif rank_delta < 0:
            feats[i, 30] = -1.0

        # Destination center proximity [0, 1]
        feats[i, 31] = (3.0 - center_distance(to_sq)) / 3.0

        # Moving to enemy-attacked square without friendly defense (risk indicator)
        if ((enemy_attacks >> to_sq) & 1) and not ((friendly_attacks >> to_sq) & 1):
            feats[i, 32] = 1.0

        # Moving the same piece twice in a row
        if from_sq == prev_own_to:
            feats[i, 34] = 1.0

        # Empirical-Bayes smoothed piece-square log-prior
        if psq_log_prior is not None:
            feats[i, 35] = float(psq_log_prior[pt_idx][to_sq])

        # ── v2.1 behavioral & tactical-style features (indices 36..48) ──
        from_attacked = bool((enemy_attacks >> from_sq) & 1)
        to_attacked = bool((enemy_attacks >> to_sq) & 1)

        # 36: Moving a currently threatened non-pawn piece to safety
        if pt != chess.PAWN and from_attacked and not to_attacked:
            feats[i, 36] = 1.0

        # 37-38: Threatening enemy minor/major pieces (and undefended ones) from to_sq
        if pt != chess.KING:
            if pt == chess.PAWN:
                att_mask = chess.BB_PAWN_ATTACKS[turn][to_sq]
            elif pt == chess.KNIGHT:
                att_mask = chess.BB_KNIGHT_ATTACKS[to_sq]
            elif pt == chess.BISHOP:
                att_mask = chess.BB_DIAG_ATTACKS[to_sq][chess.BB_DIAG_MASKS[to_sq] & board.occupied]
            elif pt == chess.ROOK:
                occ = board.occupied
                att_mask = (
                    chess.BB_RANK_ATTACKS[to_sq][chess.BB_RANK_MASKS[to_sq] & occ]
                    | chess.BB_FILE_ATTACKS[to_sq][chess.BB_FILE_MASKS[to_sq] & occ]
                )
            else:  # Queen
                occ = board.occupied
                att_mask = (
                    chess.BB_DIAG_ATTACKS[to_sq][chess.BB_DIAG_MASKS[to_sq] & occ]
                    | chess.BB_RANK_ATTACKS[to_sq][chess.BB_RANK_MASKS[to_sq] & occ]
                    | chess.BB_FILE_ATTACKS[to_sq][chess.BB_FILE_MASKS[to_sq] & occ]
                )
            if att_mask & enemy_major_minor_mask:
                feats[i, 37] = 1.0
            if att_mask & enemy_undefended_major_minor:
                feats[i, 38] = 1.0

        # 39-41, 47: Capture nuances (towards center, high-value victim, losing capture, minor-for-minor)
        if is_cap:
            victim = board.piece_at(to_sq)
            v_pt = victim.piece_type if victim else chess.PAWN
            v_val = PIECE_VALUES.get(v_pt, 100)
            a_val = PIECE_VALUES.get(pt, 0)
            if pt == chess.PAWN and abs(to_file - 3.5) < abs(from_file - 3.5):
                feats[i, 39] = 1.0
            if v_val >= 320:
                feats[i, 40] = 1.0
            if a_val > v_val + 50 and to_attacked:
                feats[i, 41] = 1.0
            if pt in (chess.KNIGHT, chess.BISHOP) and v_pt in (chess.KNIGHT, chess.BISHOP):
                feats[i, 47] = 1.0

        # 42-43: Unprovoked retreat vs defensive retreat from attack
        if rank_delta < 0 and pt in (chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN) and not is_cap:
            if from_attacked:
                feats[i, 43] = 1.0
            else:
                feats[i, 42] = 1.0

        # 44: Early queen sortie (opening ply <= 14)
        if pt == chess.QUEEN and eff_ply <= 14 and from_rank == back_rank and to_rank != back_rank:
            feats[i, 44] = 1.0

        # 45: Clearing the last piece between back-rank rooks (connecting rooks)
        if back_rank_between_rooks and ((back_rank_between_rooks >> from_sq) & 1) and to_rank != back_rank:
            if (back_rank_between_rooks & ~chess.BB_SQUARES[from_sq]) == 0:
                feats[i, 45] = 1.0

        # 46: Doubling rooks or forming a Q+R vertical battery on a file
        if pt in (chess.ROOK, chess.QUEEN) and to_file != from_file:
            if friendly_heavy_files.get(to_file, 0) >= 1:
                feats[i, 46] = 1.0

        # 48: Empirical-Bayes smoothed (piece, from_sq -> to_sq) maneuver prior
        if from_to_log_prior is not None:
            feats[i, 48] = float(from_to_log_prior.get(f"{pt_idx}:{move.uci()[:4]}", 0.0))

    return feats


def _fit_conditional_logit(
    samples: list[tuple[np.ndarray, int, float]],
    l2_reg: float = 5.0,
    max_iter: int = 18,
) -> list[float]:
    """Fit L2-regularized conditional logistic regression over legal-move sets via Newton-Raphson."""
    if len(samples) < 15:
        return [0.0] * NUM_STYLE_FEATURES

    theta = np.zeros(NUM_STYLE_FEATURES, dtype=np.float64)
    total_w = sum(w for _, _, w in samples)
    norm_scale = len(samples) / max(total_w, 1e-9)

    def _loss_grad_hess(w_vec: np.ndarray, need_hess: bool = True):
        loss = 0.5 * l2_reg * float(np.dot(w_vec, w_vec))
        grad = l2_reg * w_vec.copy()
        hess = l2_reg * np.eye(NUM_STYLE_FEATURES, dtype=np.float64) if need_hess else None

        for X, y_idx, raw_w in samples:
            w = raw_w * norm_scale
            logits = X @ w_vec
            logits -= np.max(logits)
            exp_l = np.exp(logits)
            sum_exp = float(np.sum(exp_l))
            probs = exp_l / sum_exp
            loss -= w * float(logits[y_idx] - math.log(sum_exp))

            mean_x = probs @ X
            grad += w * (mean_x - X[y_idx])
            if need_hess and hess is not None:
                # Covariance matrix: X^T diag(p) X - mean_x mean_x^T
                weighted_x = X * probs[:, None]
                cov = X.T @ weighted_x - np.outer(mean_x, mean_x)
                hess += w * cov
        return loss, grad, hess

    for _ in range(max_iter):
        loss, grad, hess = _loss_grad_hess(theta, need_hess=True)
        grad_norm = float(np.linalg.norm(grad))
        if grad_norm < 1e-4:
            break
        try:
            step = np.linalg.solve(hess, grad)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(hess, grad, rcond=None)[0]

        # Backtracking line search
        alpha = 1.0
        accepted = False
        for _ in range(10):
            cand = theta - alpha * step
            cand_loss, _, _ = _loss_grad_hess(cand, need_hess=False)
            if cand_loss <= loss + 1e-9:
                theta = cand
                accepted = True
                break
            alpha *= 0.5
        if not accepted or float(np.linalg.norm(alpha * step)) < 1e-5:
            break

    # Clip weights modestly to avoid numerical extremes on rare features
    theta = np.clip(theta, -4.0, 4.0)
    return [round(float(v), 4) for v in theta]


def score_legal_moves_style(
    board: chess.Board,
    color_str: str,
    phase: str,
    style_profile: dict[str, Any],
    prev_own_move: chess.Move | None = None,
    prev_opp_move: chess.Move | None = None,
    tier: str | None = None,
) -> tuple[dict[str, float], dict[str, list[tuple[str, float]]]]:
    """Compute calibrated style probabilities over all legal moves for `(color_str, phase)`.

    When `tier` is provided and tier-specific weights exist in `style_profile["by_tier_color"]`,
    smoothly blends the tier-specific model with Ken's global color model via Empirical Bayes
    shrinkage so each bot mirrors Ken's behavior at its own Elo band.

    Returns `(probs_by_uci, top_reasons_by_uci)` where `probs_by_uci` sums to 1.0
    and `top_reasons_by_uci` maps each UCI move to its top 3 contributing feature
    `(feature_name, contribution)` pairs for explainability.
    """
    legal_moves = list(board.legal_moves)
    if not legal_moves:
        return {}, {}

    n = len(legal_moves)
    uniform_prob = 1.0 / n
    color_key = "white" if color_str.lower() == "white" else "black"
    phase_key = phase if phase in PHASES else "middlegame"

    model = (
        style_profile.get("by_color", {})
        .get(color_key, {})
        .get("models", {})
        .get(phase_key)
    )

    if not model or not model.get("weights"):
        return (
            {m.uci(): uniform_prob for m in legal_moves},
            {m.uci(): [] for m in legal_moves},
        )

    weights = np.array(model["weights"], dtype=np.float64)
    if weights.shape[0] != NUM_STYLE_FEATURES:
        return (
            {m.uci(): uniform_prob for m in legal_moves},
            {m.uci(): [] for m in legal_moves},
        )

    psq_prior = model.get("psq_log_prior")
    from_to_prior = model.get("from_to_log_prior")
    train_pos = int(model.get("train_positions", 0))
    # Empirical shrinkage toward uniform when training data in this bucket is thin
    confidence = train_pos / (train_pos + 40.0) if train_pos > 0 else 0.0

    X = extract_legal_move_features(
        board,
        legal_moves,
        psq_log_prior=psq_prior,
        prev_own_move=prev_own_move,
        prev_opp_move=prev_opp_move,
        from_to_log_prior=from_to_prior,
    )
    eff_weights = weights * confidence

    # Blend tier-specific Ken behavior model when available
    if tier is not None:
        t_model = (
            style_profile.get("by_tier_color", {})
            .get(tier, {})
            .get(color_key, {})
            .get(phase_key)
        )
        if t_model and t_model.get("weights"):
            t_w = np.array(t_model["weights"], dtype=np.float64)
            t_pos = int(t_model.get("train_positions", 0))
            if t_w.shape[0] == NUM_STYLE_FEATURES and t_pos >= 50:
                t_conf = t_pos / (t_pos + 40.0)
                t_alpha = min(0.65, t_pos / (t_pos + 500.0))
                eff_weights = (1.0 - t_alpha) * eff_weights + t_alpha * (t_w * t_conf)

    logits = X @ eff_weights
    logits -= np.max(logits)
    exp_l = np.exp(logits)
    probs = exp_l / float(np.sum(exp_l))

    probs_by_uci: dict[str, float] = {}
    reasons_by_uci: dict[str, list[tuple[str, float]]] = {}
    for idx, mv in enumerate(legal_moves):
        uci = mv.uci()
        probs_by_uci[uci] = float(probs[idx])
        contribs = X[idx] * eff_weights
        # Pick top 3 features by absolute contribution (nonzero only)
        top_idx = np.argsort(np.abs(contribs))[::-1][:3]
        reasons_by_uci[uci] = [
            (STYLE_FEATURE_NAMES[j], round(float(contribs[j]), 3))
            for j in top_idx
            if abs(contribs[j]) >= 0.03
        ]

    return probs_by_uci, reasons_by_uci


# ── Error Classification & Incremental Offline Stockfish Analysis ────


def classify_error_type(
    board: chess.Board,
    move: chess.Move,
    best_move: chess.Move | None,
    phase: str,
    cpl: int,
) -> str:
    """Categorize a move's centipawn loss into a recurring human mistake bucket."""
    if cpl < 60:
        return "none"

    opp = not board.turn
    piece = board.piece_at(move.from_square)
    # 1. Hanging piece: moving a piece (worth >= 300) onto an enemy-attacked undefended square
    #    or ignoring a non-pawn piece capture
    if piece is not None and PIECE_VALUES.get(piece.piece_type, 0) >= 300:
        if board.is_attacked_by(opp, move.to_square) and not board.is_attacked_by(
            board.turn, move.to_square
        ):
            if not board.is_capture(move):
                return "hanging_piece"

    # 2. Endgame technique errors
    if phase == "endgame":
        return "endgame_technique"

    # 3. Premature attack (flank pawn storm / king-ring thrust / speculative check losing >= 80 cp)
    if cpl >= 80 and (
        is_pawn_storm_move(board, move)
        or board.gives_check(move)
        or (
            board.king(opp) is not None
            and chebyshev_distance(move.to_square, board.king(opp)) <= 1
        )
    ):
        return "premature_attack"

    # 4. Missed tactic: best move was a capture/check/promotion or large tactical swing >= 170 cp
    if best_move is not None and (
        board.is_capture(best_move)
        or board.gives_check(best_move)
        or best_move.promotion is not None
        or cpl >= 170
    ):
        return "missed_tactic"

    return "positional_drift"


def _clamp_cp(val: int) -> int:
    return max(-_EVAL_CLAMP_CP, min(_EVAL_CLAMP_CP, val))


def _analyze_single_game_worker(
    game_row: tuple[str, str, str, str],
    depth: int = 8,
    max_plies_to_analyze: int = 110,
) -> tuple[str, list[tuple[Any, ...]]]:
    """Analyze Ken's moves in a single game with full-strength Stockfish."""
    dedupe_key, ken_side, tier, moves_str = game_row
    ken_color = chess.WHITE if ken_side == "white" else chess.BLACK
    moves = moves_str.split() if moves_str else []

    results: list[tuple[Any, ...]] = []
    engine = StockfishEngine(threads=1, hash_mb=16)
    try:
        board = chess.Board()
        for ply_idx, uci in enumerate(moves):
            if ply_idx >= max_plies_to_analyze:
                break
            try:
                move = chess.Move.from_uci(uci)
            except ValueError:
                break
            if move not in board.legal_moves:
                break

            if board.turn == ken_color:
                phase = detect_game_phase(board, ply=ply_idx)
                pre = engine.analyse_candidates(
                    board, multipv=1, time_limit=0.0, depth=depth, uci_elo=None
                )
                if pre:
                    best_cp = _clamp_cp(int(pre[0]["score_cp"]))
                    best_uci = str(pre[0]["uci"])
                    best_move = pre[0]["move"]
                    is_top1 = 1 if uci == best_uci else 0

                    board.push(move)
                    if is_top1 or board.is_checkmate():
                        cpl = 0
                    else:
                        post = engine.analyse_candidates(
                            board,
                            multipv=1,
                            time_limit=0.0,
                            depth=max(4, depth - 1),
                            uci_elo=None,
                        )
                        after_cp = -_clamp_cp(int(post[0]["score_cp"])) if post else best_cp
                        cpl = max(0, best_cp - after_cp)
                    board.pop()

                    err_type = classify_error_type(board, move, best_move, phase, cpl)
                    results.append(
                        (
                            dedupe_key,
                            ply_idx,
                            phase,
                            ken_side,
                            tier,
                            uci,
                            best_uci,
                            cpl,
                            is_top1,
                            err_type,
                        )
                    )
            board.push(move)
    finally:
        engine.stop()

    return dedupe_key, results


def analyze_ken_mistakes_incremental(
    db_path: Path = BOOK_DB_PATH,
    max_new_games: int = 120,
    depth: int = 8,
    workers: int = 4,
    verbose: bool = True,
) -> dict[str, Any]:
    """Incrementally analyze unanalyzed Ken games with full-strength Stockfish.

    Prioritizes ALL OTB games first, then draws a balanced mix across tiers,
    colors, and train/holdout splits. Results are cached in `game_move_analysis`
    and `ingested_games.analyzed_at` so re-running never repeats prior work.
    """
    t0 = time.time()
    conn = init_book_db(db_path)

    already_analyzed = int(
        conn.execute("SELECT COUNT(*) FROM ingested_games WHERE analyzed_at IS NOT NULL").fetchone()[0]
    )

    if max_new_games <= 0:
        total_rows = int(conn.execute("SELECT COUNT(*) FROM game_move_analysis").fetchone()[0])
        conn.close()
        return {
            "newly_analyzed_games": 0,
            "newly_analyzed_moves": 0,
            "total_analyzed_games": already_analyzed,
            "total_analyzed_moves": total_rows,
            "elapsed_seconds": 0.0,
        }

    # 1. Always prioritize unanalyzed OTB games
    otb_rows = conn.execute(
        """
        SELECT dedupe_key, ken_side, tier, moves_uci FROM ingested_games
        WHERE analyzed_at IS NULL AND is_otb = 1 AND ply_count >= 10
        ORDER BY game_date DESC, dedupe_key DESC
        """
    ).fetchall()

    selected: list[tuple[str, str, str, str]] = list(otb_rows[:max_new_games])
    selected_ids = {r[0] for r in selected}
    remaining = max_new_games - len(selected)

    # 2. Fill remaining budget stratified across (tier, ken_side) prioritizing recent games
    if remaining > 0:
        buckets: list[list[tuple[str, str, str, str]]] = []
        for tier in TIERS:
            for side in COLORS:
                rows = conn.execute(
                    """
                    SELECT dedupe_key, ken_side, tier, moves_uci FROM ingested_games
                    WHERE analyzed_at IS NULL AND is_otb = 0
                      AND tier = ? AND ken_side = ? AND ply_count >= 16
                    ORDER BY game_date DESC, dedupe_key DESC
                    LIMIT ?
                    """,
                    (tier, side, remaining),
                ).fetchall()
                buckets.append([r for r in rows if r[0] not in selected_ids])

        # Round-robin pick from all 6 (tier, side) buckets
        idx_map = [0] * len(buckets)
        while remaining > 0:
            progressed = False
            for b_i, bucket in enumerate(buckets):
                if remaining <= 0:
                    break
                if idx_map[b_i] < len(bucket):
                    row = bucket[idx_map[b_i]]
                    idx_map[b_i] += 1
                    if row[0] not in selected_ids:
                        selected.append(row)
                        selected_ids.add(row[0])
                        remaining -= 1
                        progressed = True
            if not progressed:
                break

    if not selected:
        total_rows = int(conn.execute("SELECT COUNT(*) FROM game_move_analysis").fetchone()[0])
        conn.close()
        if verbose:
            print(f"No unanalyzed games remaining (total analyzed: {already_analyzed}).")
        return {
            "newly_analyzed_games": 0,
            "newly_analyzed_moves": 0,
            "total_analyzed_games": already_analyzed,
            "total_analyzed_moves": total_rows,
            "elapsed_seconds": 0.0,
        }

    if verbose:
        print(
            f"Analyzing {len(selected)} unanalyzed games (depth={depth}, workers={workers}, "
            f"OTB={sum(1 for r in otb_rows if r[0] in selected_ids)})..."
        )

    now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
    new_moves_count = 0
    done_games = 0

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        future_map = {
            pool.submit(_analyze_single_game_worker, row, depth): row[0]
            for row in selected
        }
        for future in as_completed(future_map):
            dedupe_key, move_rows = future.result()
            with conn:
                if move_rows:
                    conn.executemany(
                        """
                        INSERT OR REPLACE INTO game_move_analysis (
                            dedupe_key, ply, phase, ken_side, tier,
                            ken_move_uci, best_move_uci, cpl, is_top1, error_type
                        )
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        move_rows,
                    )
                conn.execute(
                    "UPDATE ingested_games SET analyzed_at = ? WHERE dedupe_key = ?",
                    (now_iso, dedupe_key),
                )
            done_games += 1
            new_moves_count += len(move_rows)
            if verbose and (done_games % 20 == 0 or done_games == len(selected)):
                print(
                    f"  Analyzed {done_games}/{len(selected)} games "
                    f"({new_moves_count} moves, {time.time() - t0:.1f}s)..."
                )

    total_analyzed_games = int(
        conn.execute("SELECT COUNT(*) FROM ingested_games WHERE analyzed_at IS NOT NULL").fetchone()[0]
    )
    total_analyzed_moves = int(conn.execute("SELECT COUNT(*) FROM game_move_analysis").fetchone()[0])
    conn.close()
    elapsed = round(time.time() - t0, 2)
    return {
        "newly_analyzed_games": done_games,
        "newly_analyzed_moves": new_moves_count,
        "total_analyzed_games": total_analyzed_games,
        "total_analyzed_moves": total_analyzed_moves,
        "elapsed_seconds": elapsed,
    }


# ── Unified Style Model Trainer (White/Black + Phase + Mistakes) ─────


def _aggregate_mistake_stats(
    conn: sqlite3.Connection,
    split_filter: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Aggregate real mistake profiles and human SF-top-1 agreement by color and by tier."""
    where_clause = "WHERE g.split = ?" if split_filter else ""
    params = (split_filter,) if split_filter else ()

    rows = conn.execute(
        f"""
        SELECT g.ken_side, g.tier, a.phase, a.cpl, a.is_top1, a.error_type, g.source_weight
        FROM game_move_analysis a
        JOIN ingested_games g ON g.dedupe_key = a.dedupe_key
        {where_clause}
        """,
        params,
    ).fetchall()

    def _new_bucket() -> dict[str, Any]:
        return {
            "count": 0,
            "w_sum": 0.0,
            "cpl_sum": 0.0,
            "top1_sum": 0.0,
            "m_60_150": 0.0,
            "m_150_300": 0.0,
            "m_300p": 0.0,
            "err_counts": {e: 0.0 for e in ERROR_TYPES},
        }

    by_color: dict[str, dict[str, dict[str, Any]]] = {
        c: {p: _new_bucket() for p in PHASES} for c in COLORS
    }
    by_tier: dict[str, dict[str, dict[str, Any]]] = {
        t: {p: _new_bucket() for p in PHASES} for t in TIERS
    }

    for side, tier, phase, cpl, is_top1, err_type, weight in rows:
        if side not in by_color or phase not in PHASES:
            continue
        w = float(weight) if weight else 1.0
        for bucket in (
            by_color[side][phase],
            by_tier[tier][phase] if tier in by_tier else None,
        ):
            if bucket is None:
                continue
            bucket["count"] += 1
            bucket["w_sum"] += w
            bucket["cpl_sum"] += w * float(cpl)
            bucket["top1_sum"] += w * float(is_top1)
            if 60 <= cpl < 150:
                bucket["m_60_150"] += w
            elif 150 <= cpl < 300:
                bucket["m_150_300"] += w
            elif cpl >= 300:
                bucket["m_300p"] += w
            if err_type in bucket["err_counts"]:
                bucket["err_counts"][err_type] += w

    def _finalize(bucket: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        cnt = int(bucket["count"])
        ws = max(bucket["w_sum"], 1e-9)
        if cnt == 0:
            default_m = _default_mistake_phase()
            return default_m, {"top1_rate": 0.40, "analyzed_moves": 0}
        err_total = sum(bucket["err_counts"].values())
        if err_total > 0:
            err_dist = {
                k: round(v / err_total, 3) for k, v in bucket["err_counts"].items()
            }
        else:
            err_dist = _default_mistake_phase()["error_type_dist"]

        mistake_out = {
            "analyzed_moves": cnt,
            "avg_cpl": round(bucket["cpl_sum"] / ws, 1),
            "mistake_rate_60_150": round(bucket["m_60_150"] / ws, 4),
            "mistake_rate_150_300": round(bucket["m_150_300"] / ws, 4),
            "blunder_rate_300plus": round(bucket["m_300p"] / ws, 4),
            "error_type_dist": err_dist,
        }
        agreement_out = {
            "top1_rate": round(bucket["top1_sum"] / ws, 4),
            "analyzed_moves": cnt,
        }
        return mistake_out, agreement_out

    color_out: dict[str, dict[str, Any]] = {}
    for c in COLORS:
        color_out[c] = {"mistake_profile": {}, "human_engine_agreement": {}}
        for p in PHASES:
            m_dict, a_dict = _finalize(by_color[c][p])
            color_out[c]["mistake_profile"][p] = m_dict
            color_out[c]["human_engine_agreement"][p] = a_dict

    tier_out: dict[str, dict[str, Any]] = {}
    for t in TIERS:
        tier_out[t] = {"mistake_profile": {}, "human_engine_agreement": {}}
        for p in PHASES:
            m_dict, a_dict = _finalize(by_tier[t][p])
            tier_out[t]["mistake_profile"][p] = m_dict
            tier_out[t]["human_engine_agreement"][p] = a_dict

    return color_out, tier_out


def train_ken_style_profile(
    db_path: Path = BOOK_DB_PATH,
    split_filter: str | None = None,
    max_positions_per_bucket: int = 6500,
    max_tier_positions_per_bucket: int = 2400,
    seed: int = 42,
    verbose: bool = True,
) -> dict[str, Any]:
    """Train Ken's White/Black & tier-specific move-level style models from `ingested_games`.

    Pass `split_filter="train"` when evaluating on `split="holdout"` so there is
    zero leakage between training and evaluation.
    """
    t0 = time.time()
    rng = random.Random(seed)
    conn = init_book_db(db_path)

    query = """
        SELECT dedupe_key, ken_side, tier, is_otb, source_weight, game_date, moves_uci
        FROM ingested_games
    """
    params: tuple[Any, ...] = ()
    if split_filter is not None:
        query += " WHERE split = ?"
        params = (split_filter,)
    query += " ORDER BY is_otb DESC, game_date DESC, dedupe_key DESC"
    rows = conn.execute(query, params).fetchall()

    # Pass 1: Aggregate behavioral stats + smoothed piece-square & from-to maneuver tables
    psq_counts: dict[str, dict[str, np.ndarray]] = {
        c: {p: np.zeros((6, 64), dtype=np.float64) for p in PHASES}
        for c in COLORS
    }
    from_to_counts: dict[str, dict[str, dict[str, float]]] = {
        c: {p: {} for p in PHASES}
        for c in COLORS
    }
    from_to_totals: dict[str, dict[str, float]] = {
        c: {p: 0.0 for p in PHASES}
        for c in COLORS
    }
    color_meta: dict[str, dict[str, Any]] = {
        c: {
            "games_count": 0,
            "otb_games": 0,
            "phase_moves": {p: 0 for p in PHASES},
        }
        for c in COLORS
    }
    tier_stats: dict[str, dict[str, Any]] = {
        tier: {
            "games_count": 0,
            "total_moves": 0,
            "kingside_castles": 0,
            "queenside_castles": 0,
            "uncastled_games": 0,
            "castling_plies": [],
            "mg_moves": 0,
            "mg_checks": 0,
            "mg_captures": 0,
            "mg_king_attacks": 0,
            "mg_pawn_storms": 0,
            "mg_knight_moves": 0,
            "mg_bishop_moves": 0,
            "queen_trades_ahead": 0,
            "queen_trades_behind": 0,
            "queen_opps_ahead": 0,
            "queen_opps_behind": 0,
            "late_knights_count": 0,
            "late_bishops_count": 0,
            "eg_moves": 0,
            "eg_king_moves": 0,
            "eg_king_center_dists": [],
            "eg_passed_pushes": 0,
            "eg_passed_opps": 0,
            "eg_rook_moves": 0,
            "eg_rook_open_files": 0,
            "eg_rook_7th_ranks": 0,
        }
        for tier in TIERS
    }

    # Candidate position pools stratified by (tier, color, phase) so stronger/recent tiers
    # are never drowned out by early beginner games, and each tier trains its own model too.
    # Tuple: (fen, chosen_uci, weight, prev_own_uci, prev_opp_uci, is_otb)
    tier_position_pools: dict[tuple[str, str, str], list[tuple[str, str, float, str | None, str | None, int]]] = {
        (t, c, p): [] for t in TIERS for c in COLORS for p in PHASES
    }
    tier_phase_seen: dict[tuple[str, str, str], int] = {
        (t, c, p): 0 for t in TIERS for c in COLORS for p in PHASES
    }

    for _, side, tier, is_otb, weight, _, moves_str in rows:
        if side not in COLORS or tier not in tier_stats:
            continue
        w = float(weight) if weight else 1.0
        ken_color = chess.WHITE if side == "white" else chess.BLACK
        color_meta[side]["games_count"] += 1
        if is_otb:
            color_meta[side]["otb_games"] += 1

        t_data = tier_stats[tier]
        t_data["games_count"] += 1

        moves = moves_str.split() if moves_str else []
        board = chess.Board()
        ken_castled = False
        prev_own_uci: str | None = None
        prev_opp_uci: str | None = None

        for ply_idx, uci in enumerate(moves):
            try:
                move = chess.Move.from_uci(uci)
            except ValueError:
                break
            if move not in board.legal_moves:
                break

            if ply_idx == 30:
                t_data["late_knights_count"] += len(board.pieces(chess.KNIGHT, ken_color))
                t_data["late_bishops_count"] += len(board.pieces(chess.BISHOP, ken_color))

            if board.turn == ken_color:
                phase = detect_game_phase(board, ply=ply_idx)
                color_meta[side]["phase_moves"][phase] += 1
                t_data["total_moves"] += 1

                piece = board.piece_at(move.from_square)
                if piece is not None:
                    pt_i = piece.piece_type - 1
                    psq_counts[side][phase][pt_i, move.to_square] += w
                    ft_key = f"{pt_i}:{uci[:4]}"
                    ft_bucket = from_to_counts[side][phase]
                    ft_bucket[ft_key] = ft_bucket.get(ft_key, 0.0) + w
                    from_to_totals[side][phase] += w

                if board.is_castling(move):
                    ken_castled = True
                    t_data["castling_plies"].append(ply_idx)
                    if board.is_kingside_castling(move):
                        t_data["kingside_castles"] += 1
                    else:
                        t_data["queenside_castles"] += 1

                # Accurate queen trade opportunity tracking
                if board.pieces(chess.QUEEN, ken_color) and board.pieces(chess.QUEEN, not ken_color):
                    mat_diff = get_material_diff(board, ken_color)
                    is_qt = is_queen_trade_move(board, move)
                    if mat_diff > 50:
                        t_data["queen_opps_ahead"] += 1
                        if is_qt:
                            t_data["queen_trades_ahead"] += 1
                    elif mat_diff < -50:
                        t_data["queen_opps_behind"] += 1
                        if is_qt:
                            t_data["queen_trades_behind"] += 1

                if phase == "middlegame":
                    t_data["mg_moves"] += 1
                    if board.gives_check(move):
                        t_data["mg_checks"] += 1
                    if board.is_capture(move):
                        t_data["mg_captures"] += 1
                    opp_k = board.king(not ken_color)
                    if opp_k is not None and chebyshev_distance(move.to_square, opp_k) <= 1:
                        t_data["mg_king_attacks"] += 1
                    if is_pawn_storm_move(board, move):
                        t_data["mg_pawn_storms"] += 1
                    if piece is not None:
                        if piece.piece_type == chess.KNIGHT:
                            t_data["mg_knight_moves"] += 1
                        elif piece.piece_type == chess.BISHOP:
                            t_data["mg_bishop_moves"] += 1

                elif phase == "endgame":
                    t_data["eg_moves"] += 1
                    # True passed-pawn opportunity denominator
                    has_passed = any(
                        is_passed_pawn(board, sq, ken_color)
                        for sq in board.pieces(chess.PAWN, ken_color)
                    )
                    if has_passed:
                        t_data["eg_passed_opps"] += 1
                    if piece is not None:
                        if piece.piece_type == chess.KING:
                            t_data["eg_king_moves"] += 1
                            t_data["eg_king_center_dists"].append(center_distance(move.to_square))
                        elif piece.piece_type == chess.PAWN:
                            if is_passed_pawn(board, move.from_square, ken_color):
                                t_data["eg_passed_pushes"] += 1
                        elif piece.piece_type == chess.ROOK:
                            t_data["eg_rook_moves"] += 1
                            is_open, is_7th = is_rook_on_open_or_7th(board, move)
                            if is_open:
                                t_data["eg_rook_open_files"] += 1
                            if is_7th:
                                t_data["eg_rook_7th_ranks"] += 1

                # Reservoir-collect positions per (tier, side, phase) (always keep all OTB)
                if ply_idx >= 4:
                    tk = (tier, side, phase)
                    tier_phase_seen[tk] += 1
                    pool_list = tier_position_pools[tk]
                    cap_pool = max_tier_positions_per_bucket * 2
                    if is_otb or len(pool_list) < cap_pool:
                        pool_list.append(
                            (board.fen(), uci, w, prev_own_uci, prev_opp_uci, int(is_otb))
                        )
                    else:
                        j = rng.randint(0, tier_phase_seen[tk] - 1)
                        if j < len(pool_list) and pool_list[j][5] == 0:
                            pool_list[j] = (
                                board.fen(), uci, w, prev_own_uci, prev_opp_uci, 0
                            )

                prev_own_uci = uci
            else:
                prev_opp_uci = uci

            board.push(move)

        if not ken_castled:
            t_data["uncastled_games"] += 1

    # Convert PSQ weighted counts into Empirical-Bayes smoothed log-priors relative to uniform 1/64
    psq_log_priors: dict[str, dict[str, list[list[float]]]] = {c: {} for c in COLORS}
    alpha_smooth = 2.0
    uniform_sq = 1.0 / 64.0
    for c in COLORS:
        for p in PHASES:
            mat = psq_counts[c][p]
            table: list[list[float]] = []
            for pt_idx in range(6):
                row_sum = float(np.sum(mat[pt_idx]))
                denom = row_sum + 64.0 * alpha_smooth
                probs = (mat[pt_idx] + alpha_smooth) / denom
                log_p = np.clip(np.log(probs / uniform_sq), -2.5, 2.5)
                table.append([round(float(v), 4) for v in log_p])
            psq_log_priors[c][p] = table

    # Convert (piece, from_sq -> to_sq) maneuver counts into compact Empirical-Bayes log-priors
    from_to_log_priors: dict[str, dict[str, dict[str, float]]] = {c: {} for c in COLORS}
    for c in COLORS:
        for p in PHASES:
            ft_map = from_to_counts[c][p]
            tot = max(from_to_totals[c][p], 1.0)
            n_keys = max(len(ft_map), 1)
            avg_count = tot / n_keys
            scored_ft: list[tuple[str, float]] = []
            for k, cnt in ft_map.items():
                if cnt < 3.0:
                    continue
                val = math.log((cnt + 3.0) / (avg_count + 3.0))
                val = max(-2.0, min(2.5, val))
                if abs(val) >= 0.12:
                    scored_ft.append((k, round(val, 4)))
            scored_ft.sort(key=lambda kv: abs(kv[1]), reverse=True)
            from_to_log_priors[c][p] = dict(scored_ft[:240])

    def _build_samples(
        items: list[tuple[str, str, float, str | None, str | None, int]],
        psq_table: list[list[float]],
        ft_prior: dict[str, float],
    ) -> list[tuple[np.ndarray, int, float]]:
        samples: list[tuple[np.ndarray, int, float]] = []
        for fen, uci, w, prev_own_s, prev_opp_s, _ in items:
            b = chess.Board(fen)
            legal = list(b.legal_moves)
            if len(legal) <= 1:
                continue
            target_idx = -1
            for idx, mv in enumerate(legal):
                if mv.uci() == uci:
                    target_idx = idx
                    break
            if target_idx == -1:
                continue
            prev_own_mv = chess.Move.from_uci(prev_own_s) if prev_own_s else None
            prev_opp_mv = chess.Move.from_uci(prev_opp_s) if prev_opp_s else None
            X = extract_legal_move_features(
                b,
                legal,
                psq_log_prior=psq_table,
                prev_own_move=prev_own_mv,
                prev_opp_move=prev_opp_mv,
                from_to_log_prior=ft_prior,
            )
            samples.append((X, target_idx, w))
        return samples

    # Pass 2: Fit conditional logit weights per (color, phase) using tier-stratified pools
    # PLUS per-(tier, color, phase) specialized style models!
    color_models: dict[str, dict[str, dict[str, Any]]] = {c: {} for c in COLORS}
    tier_color_models: dict[str, dict[str, dict[str, Any]]] = {
        t: {c: {} for c in COLORS} for t in TIERS
    }
    tier_quotas = {
        TIER_CONDITION: 0.40,  # prioritize Ken's strongest + OTB play in core DNA
        TIER_FOCUS: 0.38,
        TIER_SICK: 0.22,
    }
    for c in COLORS:
        for p in PHASES:
            psq_table = psq_log_priors[c][p]
            ft_prior = from_to_log_priors[c][p]

            # Build per-tier samples first (reused for both tier model and global stratified pool)
            tier_chosen: dict[str, list[tuple[str, str, float, str | None, str | None, int]]] = {}
            tier_built_samples: dict[str, list[tuple[np.ndarray, int, float]]] = {}
            for t in TIERS:
                pool_items = tier_position_pools[(t, c, p)]
                otb_items = [item for item in pool_items if item[5] == 1]
                online_items = [item for item in pool_items if item[5] == 0]
                t_budget = max(0, max_tier_positions_per_bucket - len(otb_items))
                if len(online_items) > t_budget:
                    online_items = rng.sample(online_items, t_budget)
                chosen_t = otb_items + online_items
                tier_chosen[t] = chosen_t
                s_t = _build_samples(chosen_t, psq_table, ft_prior)
                tier_built_samples[t] = s_t
                t_weights = _fit_conditional_logit(s_t, l2_reg=10.0) if len(s_t) >= 40 else [0.0] * NUM_STYLE_FEATURES
                tier_color_models[t][c][p] = {
                    "weights": t_weights,
                    "train_positions": len(s_t),
                }

            # Assemble stratified global (color, phase) sample set up to max_positions_per_bucket
            global_samples: list[tuple[np.ndarray, int, float]] = []
            leftover: list[tuple[np.ndarray, int, float]] = []
            for t in TIERS:
                s_t = tier_built_samples[t]
                q = int(round(max_positions_per_bucket * tier_quotas[t]))
                if len(s_t) <= q:
                    global_samples.extend(s_t)
                else:
                    idxs = list(range(len(s_t)))
                    rng.shuffle(idxs)
                    global_samples.extend(s_t[i] for i in idxs[:q])
                    leftover.extend(s_t[i] for i in idxs[q:])
            if len(global_samples) < max_positions_per_bucket and leftover:
                need = max_positions_per_bucket - len(global_samples)
                global_samples.extend(rng.sample(leftover, min(need, len(leftover))))

            weights = _fit_conditional_logit(global_samples, l2_reg=8.0)
            color_models[c][p] = {
                "weights": weights,
                "psq_log_prior": psq_table,
                "from_to_log_prior": ft_prior,
                "train_positions": len(global_samples),
            }
            if verbose:
                top_f = sorted(
                    zip(STYLE_FEATURE_NAMES, weights),
                    key=lambda kv: abs(kv[1]),
                    reverse=True,
                )[:4]
                top_str = ", ".join(f"{k}={v:+.2f}" for k, v in top_f)
                t_counts_str = "/".join(str(tier_color_models[t][c][p]["train_positions"]) for t in TIERS)
                print(
                    f"  Trained [{c.upper():5s}/{p:10s}] on {len(global_samples):4d} pos "
                    f"(S/F/C={t_counts_str}) | top: {top_str}"
                )

    # Aggregate offline Stockfish mistake stats & real human engine-top-1 agreement
    color_mistakes, tier_mistakes = _aggregate_mistake_stats(conn, split_filter=split_filter)
    total_db_games = int(conn.execute("SELECT COUNT(*) FROM ingested_games").fetchone()[0])
    analyzed_games = int(
        conn.execute("SELECT COUNT(*) FROM ingested_games WHERE analyzed_at IS NOT NULL").fetchone()[0]
    )
    conn.close()

    # Build backward-compatible tiers section with every weight empirically derived & used
    tiers_out: dict[str, dict[str, Any]] = {}
    for tier in TIERS:
        raw = tier_stats[tier]
        g_count = max(1, raw["games_count"])
        mg_moves = max(1, raw["mg_moves"])
        eg_moves = max(1, raw["eg_moves"])
        minor_moves = max(1, raw["mg_knight_moves"] + raw["mg_bishop_moves"])
        late_minors = max(1, raw["late_knights_count"] + raw["late_bishops_count"])
        eg_rooks = max(1, raw["eg_rook_moves"])

        castled_count = raw["kingside_castles"] + raw["queenside_castles"]
        ks_rate = raw["kingside_castles"] / max(1, castled_count)
        qs_rate = raw["queenside_castles"] / max(1, castled_count)
        uncastled_rate = raw["uncastled_games"] / g_count
        avg_castling_ply = (
            sum(raw["castling_plies"]) / len(raw["castling_plies"])
            if raw["castling_plies"]
            else 13.0
        )

        check_rate = raw["mg_checks"] / mg_moves
        capture_rate = raw["mg_captures"] / mg_moves
        king_att_rate = raw["mg_king_attacks"] / mg_moves
        pawn_storm_rate = raw["mg_pawn_storms"] / mg_moves
        knight_ratio = raw["mg_knight_moves"] / minor_moves

        q_ahead_rate = raw["queen_trades_ahead"] / max(1, raw["queen_opps_ahead"])
        q_behind_rate = raw["queen_trades_behind"] / max(1, raw["queen_opps_behind"])
        late_n_retention = raw["late_knights_count"] / late_minors

        eg_k_rate = raw["eg_king_moves"] / eg_moves
        avg_eg_k_dist = (
            sum(raw["eg_king_center_dists"]) / len(raw["eg_king_center_dists"])
            if raw["eg_king_center_dists"]
            else 1.5
        )
        passed_push_rate = raw["eg_passed_pushes"] / max(1, raw["eg_passed_opps"])
        rook_open_rate = raw["eg_rook_open_files"] / eg_rooks
        rook_7th_rate = raw["eg_rook_7th_ranks"] / eg_rooks

        # Derive every weight directly from empirical rates (no dead hardcoded weights)
        tier_scale = {TIER_SICK: 1.0, TIER_FOCUS: 1.15, TIER_CONDITION: 1.3}[tier]
        minor_delta = knight_ratio - 0.5
        weights_dict = {
            "check_bonus": round(max(1.0, min(10.0, check_rate * 70.0 / tier_scale)), 1),
            "king_attack_bonus": round(max(2.0, min(10.0, king_att_rate * 250.0)), 1),
            "pawn_storm_bonus": round(max(1.5, min(8.0, pawn_storm_rate * 220.0)), 1),
            "knight_bias": round(max(-5.0, min(5.0, minor_delta * 25.0)), 1),
            "bishop_bias": round(max(-5.0, min(5.0, -minor_delta * 25.0)), 1),
            "queen_trade_ahead_bonus": round(max(4.0, min(14.0, (0.5 + q_ahead_rate) * 10.0 * tier_scale)), 1),
            "queen_trade_behind_penalty": round(-max(4.0, min(14.0, (1.0 - q_behind_rate) * 9.0 * tier_scale)), 1),
            "endgame_king_centralization_bonus": round(max(4.0, min(12.0, (3.0 - avg_eg_k_dist) * 4.5 * tier_scale)), 1),
            "endgame_passed_pawn_bonus": round(max(4.0, min(14.0, (0.35 + passed_push_rate) * 11.0 * tier_scale)), 1),
            "endgame_rook_activity_bonus": round(max(3.5, min(12.0, (rook_open_rate + rook_7th_rate) * 10.0 * tier_scale)), 1),
        }

        tiers_out[tier] = {
            "games_count": raw["games_count"],
            "total_moves": raw["total_moves"],
            "castling": {
                "kingside_rate": round(ks_rate, 3),
                "queenside_rate": round(qs_rate, 3),
                "uncastled_rate": round(uncastled_rate, 3),
                "avg_ply": round(avg_castling_ply, 1),
            },
            "middlegame": {
                "check_rate": round(check_rate, 4),
                "capture_rate": round(capture_rate, 4),
                "king_attack_rate": round(king_att_rate, 4),
                "pawn_storm_rate": round(pawn_storm_rate, 4),
                "knight_ratio": round(knight_ratio, 3),
            },
            "trading": {
                "queen_trade_ahead_rate": round(q_ahead_rate, 3),
                "queen_trade_behind_rate": round(q_behind_rate, 3),
                "late_knight_retention": round(late_n_retention, 3),
            },
            "endgame": {
                "king_move_rate": round(eg_k_rate, 3),
                "avg_king_center_dist": round(avg_eg_k_dist, 2),
                "passed_pawn_push_rate": round(passed_push_rate, 3),
                "rook_open_file_rate": round(rook_open_rate, 3),
                "rook_7th_rank_rate": round(rook_7th_rate, 3),
            },
            "weights": weights_dict,
            "mistake_profile": tier_mistakes[tier]["mistake_profile"],
            "human_engine_agreement": tier_mistakes[tier]["human_engine_agreement"],
        }

    # Build thin-data honesty notices
    thin_warnings: list[str] = []
    for c in COLORS:
        if color_meta[c]["otb_games"] < 15:
            thin_warnings.append(
                f"{c.title()} OTB sample is small ({color_meta[c]['otb_games']} games); weighted 3x alongside online games."
            )
        for p in PHASES:
            if color_models[c][p]["train_positions"] < 500:
                thin_warnings.append(
                    f"{c.title()} {p} has {color_models[c][p]['train_positions']} training positions."
                )

    elapsed = round(time.time() - t0, 2)
    profile: dict[str, Any] = {
        "version": "2.1",
        "updated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "split_filter": split_filter or "all",
        "total_games_scanned": total_db_games,
        "valid_games_parsed": len(rows),
        "analyzed_games_for_mistakes": analyzed_games,
        "analysis_seconds": elapsed,
        "feature_names": list(STYLE_FEATURE_NAMES),
        "thin_data_notes": thin_warnings,
        "by_color": {
            c: {
                "games_count": color_meta[c]["games_count"],
                "otb_games": color_meta[c]["otb_games"],
                "phase_moves": color_meta[c]["phase_moves"],
                "models": color_models[c],
                "mistake_profile": color_mistakes[c]["mistake_profile"],
                "human_engine_agreement": color_mistakes[c]["human_engine_agreement"],
            }
            for c in COLORS
        },
        "by_tier_color": tier_color_models,
        "tiers": tiers_out,
    }

    if verbose:
        print(
            f"Style profile v2.1 trained in {elapsed:.2f}s over {len(rows)} games "
            f"(White={color_meta['white']['games_count']}, Black={color_meta['black']['games_count']}, "
            f"OTB={color_meta['white']['otb_games'] + color_meta['black']['otb_games']})."
        )
    return profile


def analyze_ken_playstyle(
    pgn_files: list[Path] = PGN_FILES,
    verbose: bool = True,
) -> dict[str, Any]:
    """Backward-compatible wrapper: ingests PGNs if needed and trains v2.0 style profile."""
    from calibration.build_book import build_opening_book

    if pgn_files:
        build_opening_book(pgn_files=pgn_files, db_path=BOOK_DB_PATH, verbose=False)
    return train_ken_style_profile(db_path=BOOK_DB_PATH, verbose=verbose)


# ── Hold-Out Generalization & Engine-Likeness Evaluator ──────────────


def evaluate_holdout_style(
    db_path: Path = BOOK_DB_PATH,
    max_holdout_games: int = 25,
    eval_depth: int = 8,
    verbose: bool = True,
) -> dict[str, Any]:
    """Evaluate bots strictly on `split='holdout'` using a model trained ONLY on `split='train'`.

    Compares each KenBot (Sick, Focus, Condition) against a plain Stockfish
    baseline at matching UCI_Elo on out-of-book positions across phases:
    - Top-1 and Top-3 match rate against Ken's actual move
    - Engine top-1 agreement vs Ken's real human engine-top-1 agreement
    """
    from bots.ken_bot import KEN_PROFILES, KenBot
    from calibration.build_book import rebuild_book_from_db
    import tempfile

    t0 = time.time()

    # Build an isolated train-only DB copy & style profile so hold-out evaluation has ZERO leakage
    conn_src = init_book_db(db_path)
    holdout_rows = conn_src.execute(
        """
        SELECT dedupe_key, ken_side, tier, is_otb, game_date, moves_uci
        FROM ingested_games
        WHERE split = 'holdout' AND ply_count >= 16
        ORDER BY is_otb DESC, game_date DESC, dedupe_key DESC
        LIMIT ?
        """,
        (max_holdout_games,),
    ).fetchall()
    conn_src.close()

    if not holdout_rows:
        return {"error": "No holdout games available.", "positions_evaluated": 0}

    train_profile = train_ken_style_profile(
        db_path=db_path, split_filter="train", max_positions_per_bucket=5000, verbose=False
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_db = Path(tmpdir) / "train_only_book.db"
        tmp_style = Path(tmpdir) / "train_only_style.json"
        save_style_profile(train_profile, path=tmp_style)

        # Copy train rows into isolated DB and build train-only book
        conn_in = sqlite3.connect(db_path)
        conn_out = init_book_db(tmp_db)
        train_games = conn_in.execute(
            """
            SELECT dedupe_key, source_file, game_index, ken_side, white_name, black_name,
                   ken_elo, tier, result, eco, event, site, game_date, time_control,
                   is_otb, source_weight, ply_count, moves_uci, split, analyzed_at, ingested_at
            FROM ingested_games WHERE split = 'train'
            """
        ).fetchall()
        conn_in.close()
        with conn_out:
            conn_out.executemany(
                """
                INSERT INTO ingested_games (
                    dedupe_key, source_file, game_index, ken_side, white_name, black_name,
                    ken_elo, tier, result, eco, event, site, game_date, time_control,
                    is_otb, source_weight, ply_count, moves_uci, split, analyzed_at, ingested_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                train_games,
            )
        conn_out.close()
        rebuild_book_from_db(db_path=tmp_db, split_filter="train", verbose=False)

        # Sample up to 6 positions per holdout game across opening/middlegame/endgame
        positions: list[tuple[str, str, str, str, int, str | None, str | None]] = []
        rng = random.Random(101)
        for _, side, _, _, _, moves_str in holdout_rows:
            ken_color = chess.WHITE if side == "white" else chess.BLACK
            moves = moves_str.split()
            board = chess.Board()
            game_pos_by_phase: dict[str, list[tuple[str, str, str, str, int, str | None, str | None]]] = {
                p: [] for p in PHASES
            }
            prev_own: str | None = None
            prev_opp: str | None = None
            for ply_idx, uci in enumerate(moves):
                try:
                    mv = chess.Move.from_uci(uci)
                except ValueError:
                    break
                if mv not in board.legal_moves:
                    break
                if board.turn == ken_color and ply_idx >= 4:
                    phase = detect_game_phase(board, ply=ply_idx)
                    game_pos_by_phase[phase].append(
                        (board.fen(), uci, side, phase, ply_idx, prev_own, prev_opp)
                    )
                    prev_own = uci
                else:
                    prev_opp = uci
                board.push(mv)

            for p in PHASES:
                cands = game_pos_by_phase[p]
                if cands:
                    picks = rng.sample(cands, min(2, len(cands)))
                    positions.extend(picks)

        engine = StockfishEngine(threads=1, hash_mb=32)
        bots = {
            name: KenBot(
                profile=name,
                db_path=tmp_db,
                engine=engine,
                style_profile_path=tmp_style,
                log_provenance=False,
            )
            for name in KEN_PROFILES
        }

        def _empty_metrics() -> dict[str, Any]:
            return {
                p: {
                    "n": 0,
                    "top1": 0,
                    "top3": 0,
                    "sf_top1": 0,
                    "oob_n": 0,
                    "oob_top1": 0,
                    "oob_top3": 0,
                    "oob_sf_top1": 0,
                    "book_n": 0,
                }
                for p in PHASES
            }

        bot_metrics = {name: _empty_metrics() for name in KEN_PROFILES}
        sf_baseline_metrics = {name: _empty_metrics() for name in KEN_PROFILES}
        human_sf_top1 = {p: [0, 0] for p in PHASES}  # [matches, total]

        try:
            for fen, ken_uci, side, phase, _, prev_own_s, prev_opp_s in positions:
                board = chess.Board(fen)
                # Reference full-strength SF candidates (top 8)
                ref_cands = engine.analyse_candidates(
                    board, multipv=8, time_limit=0.0, depth=eval_depth, uci_elo=None
                )
                if not ref_cands:
                    continue
                sf_top1_uci = ref_cands[0]["uci"]
                human_sf_top1[phase][1] += 1
                if ken_uci == sf_top1_uci:
                    human_sf_top1[phase][0] += 1

                # Plain Stockfish baseline at each profile's UCI_Elo
                for name, profile_obj in KEN_PROFILES.items():
                    plain_cands = engine.analyse_candidates(
                        board, multipv=3, time_limit=0.03, uci_elo=profile_obj.elo
                    )
                    plain_ucis = [c["uci"] for c in plain_cands]
                    m_sf = sf_baseline_metrics[name][phase]
                    m_sf["n"] += 1
                    m_sf["oob_n"] += 1
                    if plain_ucis and plain_ucis[0] == ken_uci:
                        m_sf["top1"] += 1
                        m_sf["oob_top1"] += 1
                    if ken_uci in plain_ucis[:3]:
                        m_sf["top3"] += 1
                        m_sf["oob_top3"] += 1
                    if plain_ucis and plain_ucis[0] == sf_top1_uci:
                        m_sf["sf_top1"] += 1
                        m_sf["oob_sf_top1"] += 1

                    # Evaluate KenBot (deterministic ranking via choose_move + ranked candidates)
                    bot = bots[name]
                    prev_own_mv = chess.Move.from_uci(prev_own_s) if prev_own_s else None
                    prev_opp_mv = chess.Move.from_uci(prev_opp_s) if prev_opp_s else None
                    bot._last_bot_move = prev_own_mv
                    bot._last_opp_move = prev_opp_mv

                    decision = bot.evaluate_position(board, precomputed_candidates=ref_cands)
                    chosen_uci = decision["chosen_uci"]
                    ranked_ucis = decision["ranked_ucis"]
                    is_book = decision["source"] == "book"

                    m_b = bot_metrics[name][phase]
                    m_b["n"] += 1
                    if is_book:
                        m_b["book_n"] += 1
                    else:
                        m_b["oob_n"] += 1
                    if chosen_uci == ken_uci:
                        m_b["top1"] += 1
                        if not is_book:
                            m_b["oob_top1"] += 1
                    if ken_uci in ranked_ucis[:3]:
                        m_b["top3"] += 1
                        if not is_book:
                            m_b["oob_top3"] += 1
                    if chosen_uci == sf_top1_uci:
                        m_b["sf_top1"] += 1
                        if not is_book:
                            m_b["oob_sf_top1"] += 1
        finally:
            engine.stop()

    def _summarize(metrics: dict[str, dict[str, int]]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        tot_n = sum(metrics[p]["n"] for p in PHASES)
        tot_oob = sum(metrics[p]["oob_n"] for p in PHASES)
        out["overall"] = {
            "n": tot_n,
            "top1_pct": round(100.0 * sum(metrics[p]["top1"] for p in PHASES) / max(1, tot_n), 1),
            "top3_pct": round(100.0 * sum(metrics[p]["top3"] for p in PHASES) / max(1, tot_n), 1),
            "sf_top1_pct": round(100.0 * sum(metrics[p]["sf_top1"] for p in PHASES) / max(1, tot_n), 1),
            "oob_n": tot_oob,
            "oob_top1_pct": round(100.0 * sum(metrics[p]["oob_top1"] for p in PHASES) / max(1, tot_oob), 1),
            "oob_top3_pct": round(100.0 * sum(metrics[p]["oob_top3"] for p in PHASES) / max(1, tot_oob), 1),
            "oob_sf_top1_pct": round(100.0 * sum(metrics[p]["oob_sf_top1"] for p in PHASES) / max(1, tot_oob), 1),
        }
        for p in PHASES:
            d = metrics[p]
            out[p] = {
                "n": d["n"],
                "book_n": d["book_n"],
                "top1_pct": round(100.0 * d["top1"] / max(1, d["n"]), 1),
                "top3_pct": round(100.0 * d["top3"] / max(1, d["n"]), 1),
                "sf_top1_pct": round(100.0 * d["sf_top1"] / max(1, d["n"]), 1),
                "oob_n": d["oob_n"],
                "oob_top1_pct": round(100.0 * d["oob_top1"] / max(1, d["oob_n"]), 1),
                "oob_top3_pct": round(100.0 * d["oob_top3"] / max(1, d["oob_n"]), 1),
                "oob_sf_top1_pct": round(100.0 * d["oob_sf_top1"] / max(1, d["oob_n"]), 1),
            }
        return out

    human_agree_summary = {
        p: {
            "n": human_sf_top1[p][1],
            "sf_top1_pct": round(100.0 * human_sf_top1[p][0] / max(1, human_sf_top1[p][1]), 1),
        }
        for p in PHASES
    }
    tot_h_n = sum(human_sf_top1[p][1] for p in PHASES)
    human_agree_summary["overall"] = {
        "n": tot_h_n,
        "sf_top1_pct": round(100.0 * sum(human_sf_top1[p][0] for p in PHASES) / max(1, tot_h_n), 1),
    }

    report = {
        "elapsed_seconds": round(time.time() - t0, 2),
        "holdout_games": len(holdout_rows),
        "positions_evaluated": len(positions),
        "human_sf_top1_agreement": human_agree_summary,
        "ken_bots": {name: _summarize(bot_metrics[name]) for name in KEN_PROFILES},
        "plain_stockfish": {name: _summarize(sf_baseline_metrics[name]) for name in KEN_PROFILES},
    }

    if verbose:
        print(
            f"\nHold-out Evaluation ({len(holdout_rows)} games, {len(positions)} positions, "
            f"{report['elapsed_seconds']}s):"
        )
        h = human_agree_summary
        print(
            f"  Ken's actual SF top-1 agreement: overall={h['overall']['sf_top1_pct']}% "
            f"(OP={h['opening']['sf_top1_pct']}%, MG={h['middlegame']['sf_top1_pct']}%, "
            f"EG={h['endgame']['sf_top1_pct']}%)"
        )
        for name in KEN_PROFILES:
            kb = report["ken_bots"][name]["overall"]
            sb = report["plain_stockfish"][name]["overall"]
            print(
                f"  {name:14s} | KenBot Top1={kb['top1_pct']:4.1f}% Top3={kb['top3_pct']:4.1f}% "
                f"(OOB Top1={kb['oob_top1_pct']:4.1f}% Top3={kb['oob_top3_pct']:4.1f}% SF#1={kb['oob_sf_top1_pct']:4.1f}%) "
                f"vs Plain SF Top1={sb['top1_pct']:4.1f}% Top3={sb['top3_pct']:4.1f}% SF#1={sb['sf_top1_pct']:4.1f}%"
            )

    return report


# ── Persistence & Bounded Style Bonus ────────────────────────────────


def save_style_profile(profile_data: dict[str, Any], path: Path = STYLE_PROFILE_PATH) -> None:
    """Save style profile dictionary to JSON disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(profile_data, f, indent=2)


def load_style_profile(path: Path = STYLE_PROFILE_PATH) -> dict[str, Any]:
    """Load style profile dictionary from JSON disk, returning default on failure."""
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict) and "tiers" in data:
                    return data
        except Exception as exc:
            logger.warning("Could not read style profile at %s (%s); using default.", path, exc)
    return DEFAULT_STYLE_PROFILE


def compute_style_bonus(
    board: chess.Board,
    move: chess.Move,
    phase: str,
    tier_weights: dict[str, Any],
) -> float:
    """Calculate bounded centipawn style bonus for a candidate move.

    Guaranteed to return a value strictly within [-20.0, 20.0] centipawns,
    using all tier behavioral weights (castling, middlegame, trading, endgame).
    """
    weights = tier_weights.get("weights", tier_weights)
    castling_stats = tier_weights.get("castling", {})
    bonus = 0.0

    piece = board.piece_at(move.from_square)
    if not piece:
        return 0.0

    # 1. Castling preference driven by empirical castling rates
    if board.is_castling(move):
        ks_rate = float(castling_stats.get("kingside_rate", 0.85))
        qs_rate = float(castling_stats.get("queenside_rate", 0.15))
        if board.is_kingside_castling(move):
            bonus += round(6.0 * ks_rate, 2)
        else:
            bonus += round(6.0 * qs_rate + 1.0, 2)

    # 2. Middlegame aggression & minor piece balance
    if phase == "middlegame":
        if board.gives_check(move):
            bonus += float(weights.get("check_bonus", 0.0))

        opp_k = board.king(not board.turn)
        if opp_k is not None and chebyshev_distance(move.to_square, opp_k) <= 1:
            bonus += float(weights.get("king_attack_bonus", 0.0))

        if is_pawn_storm_move(board, move):
            bonus += float(weights.get("pawn_storm_bonus", 0.0))

        if piece.piece_type == chess.KNIGHT:
            bonus += float(weights.get("knight_bias", 0.0))
        elif piece.piece_type == chess.BISHOP:
            bonus += float(weights.get("bishop_bias", 0.0))

    # 3. Positional trading / simplification (Middlegame & Endgame)
    if is_queen_trade_move(board, move):
        mat_diff = get_material_diff(board, board.turn)
        if mat_diff > 0:
            bonus += float(weights.get("queen_trade_ahead_bonus", 0.0))
        elif mat_diff < 0:
            bonus += float(weights.get("queen_trade_behind_penalty", 0.0))

    # 4. Endgame technique
    if phase == "endgame":
        if piece.piece_type == chess.KING:
            old_dist = center_distance(move.from_square)
            new_dist = center_distance(move.to_square)
            if new_dist < old_dist:
                bonus += float(weights.get("endgame_king_centralization_bonus", 0.0))
        elif piece.piece_type == chess.PAWN:
            if is_passed_pawn(board, move.from_square, board.turn):
                bonus += float(weights.get("endgame_passed_pawn_bonus", 0.0))
        elif piece.piece_type == chess.ROOK:
            is_open, is_7th = is_rook_on_open_or_7th(board, move)
            if is_7th or is_open:
                bonus += float(weights.get("endgame_rook_activity_bonus", 0.0))

    clamped_bonus = max(-20.0, min(20.0, bonus))
    return round(clamped_bonus, 2)


if __name__ == "__main__":
    profile_data = train_ken_style_profile(db_path=BOOK_DB_PATH, verbose=True)
    save_style_profile(profile_data)
    print(f"Style profile v2.0 successfully saved to {STYLE_PROFILE_PATH}")
