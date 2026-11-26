"""Calibrated Ken Bot personalities, style-first move pipeline, and threaded worker.

Move Selection Pipeline (in strict order):
  1. Weighted Opening Book (`book`)
  2. Candidate generation via full-strength Stockfish MultiPV (`uci_elo=None`)
  3. Safety Filter (no forced mate allowed, forced mate-in-1/2 taken, max cp loss cap)
  4. Style Selection (PRIMARY selector via White/Black move-level style probabilities)
  5. Error Model (calibrated to Ken's real mistake profile, capped) (`error-model`)
  6. Consistency / Plan Continuity (penalizes immediate undo / unprovoked retreat;
     enforces style-probability floor unless tactically justified)
  7. Fallback Heuristic ONLY on engine failure (`fallback`, logged loudly at ERROR)
"""

from __future__ import annotations

from dataclasses import dataclass
import datetime
import json
import logging
import math
from pathlib import Path
import queue
import random
import threading
import time
from typing import Any
import uuid

import chess
import numpy as np

from calibration.build_book import (
    TIER_CONDITION,
    TIER_FOCUS,
    TIER_SICK,
    query_book_moves,
)
from calibration.style_analyzer import (
    classify_error_type,
    compute_style_bonus,
    detect_game_phase,
    effective_ply,
    load_style_profile,
    score_legal_moves_style,
)
from others.config import BOOK_DB_PATH, PROVENANCE_LOG_DIR, STYLE_PROFILE_PATH
from engine.stockfish_engine import EngineNotFoundError, StockfishEngine
from ui.speech_bubble import get_greeting_text

logger = logging.getLogger(__name__)

_MAX_PROVENANCE_LOG_FILES = 30
_MAX_PROVENANCE_LOG_BYTES = 2 * 1024 * 1024  # 2 MB per session file

# ── Human move-quality & blunder evaluation thresholds (centipawns) ──
BLUNDER_CP_LOSS: int = 200              # >= 200 cp loss (or newly allowed mate) = Blunder
MISTAKE_CP_LOSS: int = 115              # 115..199 cp loss = Mistake
DUBIOUS_CP_LOSS: int = 55               # 55..114 cp loss = Dubious / Inaccuracy
GREAT_ONLY_MOVE_GAP_CP: int = 160       # Best move beats 2nd-best by >= 160 cp = Great
BRILLIANT_GAP_CP: int = 150             # Sacrificial best move beating 2nd-best by >= 150 cp = Brilliant
BEST_MOVE_TOLERANCE_CP: int = 15        # Within 15 cp of engine top-1 counts as best
DECIDED_EVAL_CP: int = 800              # Skip positions already completely decided in same direction
PLAYER_EVAL_TIME_LIMIT: float = 0.12    # Fast full-strength Stockfish analysis per position (s)
PLAYER_EVAL_CACHE_MAX: int = 256        # Bounded EPD evaluation cache inside BotWorker


@dataclass(frozen=True)
class KenProfile:
    """Configuration for a specific Ken bot personality tier."""

    name: str
    tier: str
    elo: int
    temperature: float               # Opening book sampling temperature
    engine_time_limit: float         # Full-strength Stockfish MultiPV time limit (s)
    engine_multipv: int              # Number of full-strength candidate lines (5-8)
    engine_temp: float               # Style-selection softmax temperature
    blunder_chance: float            # Legacy/heuristic mistake chance scale
    tagline: str
    description: str
    max_cp_loss: int = 80            # Safety gate: max cp loss for normal style-selected moves
    mistake_max_cp_loss: int = 180   # Hard cp loss cap even when error model fires
    min_book_play_count: int = 1     # Minimum times Ken played a book move
    min_book_win_rate: float = 0.0   # Minimum win rate for opening book moves
    book_fallback_tiers: tuple[str, ...] = (TIER_FOCUS, TIER_CONDITION)
    eval_tiebreak_weight: float = 0.45   # Mild eval tie-breaker weight (style stays primary)
    error_rate_scale: float = 0.35       # Multiplier on Ken's measured phase mistake rate
    min_style_prob_ratio: float = 0.25   # Style floor relative to uniform legal-move prob
    tactical_override_cp: int = 120      # Margin over #2 move that justifies overriding style floor
    consistency_penalty: float = 1.6     # Logit penalty for immediate undo / unprovoked retreat


KEN_PROFILES: dict[str, KenProfile] = {
    "Sick Ken": KenProfile(
        name="Sick Ken",
        tier=TIER_SICK,
        elo=1350,
        temperature=1.05,
        engine_time_limit=0.05,
        engine_multipv=10,
        engine_temp=1.15,
        blunder_chance=0.25,
        tagline=get_greeting_text("sick_ken", "Bai gihilantan ko, pildiha daw ko bi"),
        description="A weaker take on your style,\nwith occasional tactical errors.",
        max_cp_loss=175,
        mistake_max_cp_loss=245,
        min_book_play_count=1,
        min_book_win_rate=0.0,
        book_fallback_tiers=(TIER_SICK, TIER_FOCUS, TIER_CONDITION),
        eval_tiebreak_weight=0.04,
        error_rate_scale=1.70,
        min_style_prob_ratio=0.30,
        tactical_override_cp=160,
        consistency_penalty=1.4,
    ),
    "Focus Ken": KenProfile(
        name="Focus Ken",
        tier=TIER_FOCUS,
        elo=1800,
        temperature=0.60,
        engine_time_limit=0.08,
        engine_multipv=9,
        engine_temp=0.90,
        blunder_chance=0.08,
        tagline=get_greeting_text("focus_ken", "Ha?? sure ka makig duwa ka nako"),
        description="Consistent decisions and\nyour familiar openings.",
        max_cp_loss=125,
        mistake_max_cp_loss=210,
        min_book_play_count=1,
        min_book_win_rate=0.15,
        book_fallback_tiers=(TIER_FOCUS, TIER_CONDITION, TIER_SICK),
        eval_tiebreak_weight=0.12,
        error_rate_scale=1.05,
        min_style_prob_ratio=0.30,
        tactical_override_cp=110,
        consistency_penalty=1.7,
    ),
    "Condition Ken": KenProfile(
        name="Condition Ken",
        tier=TIER_CONDITION,
        elo=2350,
        temperature=0.30,
        engine_time_limit=0.11,
        engine_multipv=8,
        engine_temp=0.78,
        blunder_chance=0.0,
        tagline=get_greeting_text("condition_ken", "hmmm dependi kong 3 yan!"),
        description="Advanced tactical and\npositional precision.",
        max_cp_loss=78,
        mistake_max_cp_loss=145,
        min_book_play_count=1,
        min_book_win_rate=0.22,
        book_fallback_tiers=(TIER_CONDITION, TIER_FOCUS),
        eval_tiebreak_weight=0.22,
        error_rate_scale=0.42,
        min_style_prob_ratio=0.25,
        tactical_override_cp=70,
        consistency_penalty=1.8,
    ),
}


def get_profile(identifier: str) -> KenProfile:
    """Retrieve a KenProfile by name or tier key."""
    if identifier in KEN_PROFILES:
        return KEN_PROFILES[identifier]

    norm = identifier.strip().lower()
    for profile in KEN_PROFILES.values():
        if norm in (profile.name.lower(), profile.tier.lower()):
            return profile

    return KEN_PROFILES["Focus Ken"]


_PIECE_VALUES: dict[chess.PieceType, int] = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 20000,
}

_CENTER_SQUARES: frozenset[chess.Square] = frozenset(
    {chess.D4, chess.E4, chess.D5, chess.E5}
)
_EXTENDED_CENTER: frozenset[chess.Square] = frozenset(
    {
        chess.C3, chess.D3, chess.E3, chess.F3,
        chess.C4, chess.F4, chess.C5, chess.F5,
        chess.C6, chess.D6, chess.E6, chess.F6,
    }
)


def fallback_heuristic_move(board: chess.Board, profile: KenProfile) -> chess.Move:
    """Lightweight rule-based move selection used ONLY when Stockfish is unavailable."""
    legal_moves = list(board.legal_moves)
    if not legal_moves:
        raise ValueError("No legal moves available on this board.")

    scored_moves: list[tuple[float, chess.Move]] = []
    for move in legal_moves:
        score = 0.0
        captured = board.piece_at(move.to_square)
        attacker = board.piece_at(move.from_square)

        if captured:
            victim_val = _PIECE_VALUES.get(captured.piece_type, 0)
            attacker_val = _PIECE_VALUES.get(attacker.piece_type, 0) if attacker else 0
            score += 1000.0 + victim_val - (attacker_val * 0.1)

        if move.promotion:
            score += _PIECE_VALUES.get(move.promotion, 800)

        if board.gives_check(move):
            score += 60.0

        if move.to_square in _CENTER_SQUARES:
            score += 25.0
        elif move.to_square in _EXTENDED_CENTER:
            score += 12.0

        if board.is_castling(move):
            score += 45.0

        opponent_color = not board.turn
        if board.is_attacked_by(opponent_color, move.to_square):
            if attacker:
                score -= _PIECE_VALUES.get(attacker.piece_type, 0) * 0.5

        noise = random.gauss(0, 30.0 * max(0.1, profile.temperature))
        scored_moves.append((score + noise, move))

    scored_moves.sort(key=lambda item: item[0], reverse=True)

    if (
        profile.blunder_chance > 0
        and random.random() < profile.blunder_chance
        and len(scored_moves) > 1
    ):
        suboptimal_idx = min(
            len(scored_moves) - 1, random.randint(1, min(3, len(scored_moves) - 1))
        )
        return scored_moves[suboptimal_idx][1]

    return scored_moves[0][1]


def _compute_continuity_penalty(
    board: chess.Board,
    move: chess.Move,
    prev_bot_move: chess.Move | None,
    penalty_strength: float,
) -> tuple[float, str]:
    """Compute plan-continuity logit penalty for immediate undos or unprovoked retreats."""
    if prev_bot_move is None or penalty_strength <= 0.0:
        return 0.0, ""

    # 1. Exact immediate undo (moving the same piece straight back to where it came from)
    if (
        move.from_square == prev_bot_move.to_square
        and move.to_square == prev_bot_move.from_square
    ):
        if not board.is_capture(move) and not board.gives_check(move):
            return -penalty_strength, "immediate_undo"

    # 2. Unprovoked retreat of the piece we just advanced
    if move.from_square == prev_bot_move.to_square:
        piece = board.piece_at(move.from_square)
        if piece is not None and piece.piece_type in (
            chess.KNIGHT,
            chess.BISHOP,
            chess.ROOK,
            chess.QUEEN,
        ):
            is_white = board.turn == chess.WHITE
            prev_adv = (
                chess.square_rank(prev_bot_move.to_square)
                - chess.square_rank(prev_bot_move.from_square)
            ) * (1 if is_white else -1)
            now_adv = (
                chess.square_rank(move.to_square) - chess.square_rank(move.from_square)
            ) * (1 if is_white else -1)
            if (
                prev_adv > 0
                and now_adv < 0
                and not board.is_capture(move)
                and not board.gives_check(move)
                and not board.is_attacked_by(not board.turn, move.from_square)
            ):
                return -0.65 * penalty_strength, "unprovoked_retreat"

    return 0.0, ""


class KenBot:
    """AI player calibrated to Ken's repertoire, White/Black style model, and mistake profile."""

    def __init__(
        self,
        profile: KenProfile | str = "Focus Ken",
        db_path: Path = BOOK_DB_PATH,
        engine: StockfishEngine | None = None,
        style_profile_path: Path = STYLE_PROFILE_PATH,
        log_provenance: bool = True,
        provenance_dir: Path = PROVENANCE_LOG_DIR,
    ) -> None:
        self.profile: KenProfile = (
            profile if isinstance(profile, KenProfile) else get_profile(profile)
        )
        self.db_path = db_path
        self._engine = engine or StockfishEngine()
        self.style_profile = load_style_profile(style_profile_path)
        self.tier_style = self.style_profile.get("tiers", {}).get(self.profile.tier, {})

        self.move_count: int = 0
        self.book_move_count: int = 0
        self.engine_move_count: int = 0
        self.style_move_count: int = 0
        self.error_move_count: int = 0
        self.fallback_count: int = 0
        self.last_move_source: str = "none"
        self.last_move_provenance: dict[str, Any] = {}

        # Plan-continuity state (auto-resets if board ply jumps backward)
        self._last_bot_move: chess.Move | None = None
        self._last_opp_move: chess.Move | None = None
        self._last_seen_ply: int = -1

        # Bounded provenance logging
        self.log_provenance = log_provenance
        self._provenance_dir = provenance_dir
        self._session_log_path: Path | None = None

    @property
    def engine(self) -> StockfishEngine:
        return self._engine

    @property
    def session_log_path(self) -> Path | None:
        return self._session_log_path

    def reset_stats(self) -> None:
        """Reset telemetry counters and plan-continuity memory for a new game."""
        self.move_count = 0
        self.book_move_count = 0
        self.engine_move_count = 0
        self.style_move_count = 0
        self.error_move_count = 0
        self.fallback_count = 0
        self.last_move_source = "none"
        self.last_move_provenance = {}
        self._last_bot_move = None
        self._last_opp_move = None
        self._last_seen_ply = -1
        self._session_log_path = None

    def _sync_game_state(self, board: chess.Board) -> None:
        """Keep plan-continuity state aligned with the current board position."""
        cur_ply = effective_ply(board)
        if cur_ply < self._last_seen_ply or cur_ply <= 1:
            self._last_bot_move = None
            self._last_opp_move = None
            self._session_log_path = None
        self._last_seen_ply = cur_ply
        if board.move_stack:
            self._last_opp_move = board.move_stack[-1]

    def _write_provenance_record(self, record: dict[str, Any]) -> None:
        """Append a single JSONL provenance record with bounded file count & size."""
        if not self.log_provenance:
            return
        try:
            self._provenance_dir.mkdir(parents=True, exist_ok=True)
            if (
                self._session_log_path is None
                or (
                    self._session_log_path.exists()
                    and self._session_log_path.stat().st_size >= _MAX_PROVENANCE_LOG_BYTES
                )
            ):
                stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                slug = self.profile.tier
                short_id = uuid.uuid4().hex[:6]
                self._session_log_path = self._provenance_dir / f"game_{slug}_{stamp}_{short_id}.jsonl"
                # Rotate older logs if count exceeds limit
                existing = sorted(
                    self._provenance_dir.glob("game_*.jsonl"),
                    key=lambda p: p.stat().st_mtime,
                )
                while len(existing) >= _MAX_PROVENANCE_LOG_FILES:
                    oldest = existing.pop(0)
                    try:
                        oldest.unlink()
                    except OSError:
                        break

            with open(self._session_log_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, separators=(",", ":")) + "\n")
        except OSError as exc:
            logger.debug("Could not write move provenance log: %s", exc)

    def _get_phase_mistake_config(self, side: str, phase: str) -> tuple[float, dict[str, float]]:
        """Read Ken's measured mistake rate and error-type distribution for (side, phase)."""
        color_m = (
            self.style_profile.get("by_color", {})
            .get(side, {})
            .get("mistake_profile", {})
            .get(phase)
        )
        tier_m = (
            self.tier_style.get("mistake_profile", {}).get(phase)
            if isinstance(self.tier_style, dict)
            else None
        )
        m_data = color_m if (color_m and color_m.get("analyzed_moves", 0) >= 15) else (tier_m or color_m)
        if not m_data:
            base_rate = 0.14
            err_dist = {
                "hanging_piece": 0.18,
                "missed_tactic": 0.28,
                "positional_drift": 0.34,
                "premature_attack": 0.12,
                "endgame_technique": 0.08,
            }
            return base_rate, err_dist

        # Combine inaccuracies (60-150 cp) and mistakes (150-300 cp) proportionally
        base_rate = (
            0.70 * float(m_data.get("mistake_rate_60_150", 0.12))
            + 1.00 * float(m_data.get("mistake_rate_150_300", 0.05))
        )
        err_dist = dict(m_data.get("error_type_dist", {}))
        return max(0.02, min(0.35, base_rate)), err_dist

    def _select_book_move(
        self,
        board: chess.Board,
        side: str,
        deterministic: bool = False,
    ) -> dict[str, Any] | None:
        """Step 1: Weighted Opening Book lookup & sampling."""
        book_candidates = query_book_moves(
            board.epd(),
            side=side,
            tier=self.profile.tier,
            db_path=self.db_path,
            allow_fallback=True,
            min_play_count=self.profile.min_book_play_count,
            min_win_rate=self.profile.min_book_win_rate,
            fallback_tiers=list(self.profile.book_fallback_tiers),
        )

        valid_book: list[tuple[chess.Move, dict[str, Any]]] = []
        for entry in book_candidates:
            try:
                mv = chess.Move.from_uci(entry["move_uci"])
                if mv in board.legal_moves:
                    valid_book.append((mv, entry))
            except ValueError:
                continue

        if not valid_book:
            return None

        # Weight by source-weighted count (OTB=3x, recent rapid=2x, etc.) and win rate
        weights = [
            (max(0.5, float(c.get("weighted_count", c["play_count"]))) ** 0.85)
            * (float(c["win_rate"]) + 0.25)
            for _, c in valid_book
        ]
        weights_arr = np.array(weights, dtype=np.float64)
        temp = max(0.01, self.profile.temperature)

        # Rank order by book weight for explainability & Top-3 holdout metrics
        order = np.argsort(weights_arr)[::-1]
        ranked_ucis = [valid_book[int(i)][0].uci() for i in order]

        if deterministic or temp <= 0.05:
            chosen_idx = int(order[0])
            probs = weights_arr / max(float(np.sum(weights_arr)), 1e-9)
        else:
            max_w = float(np.max(weights_arr))
            norm_w = weights_arr / max_w if max_w > 0 else weights_arr
            scaled = norm_w ** (1.0 / temp)
            sum_s = float(np.sum(scaled))
            probs = scaled / sum_s if sum_s > 0 else np.full(len(valid_book), 1.0 / len(valid_book))
            chosen_idx = int(np.random.choice(len(valid_book), p=probs))

        chosen_move, chosen_entry = valid_book[chosen_idx]
        return {
            "move": chosen_move,
            "chosen_uci": chosen_move.uci(),
            "ranked_ucis": ranked_ucis,
            "source": "book",
            "style_prob": round(float(probs[chosen_idx]), 4),
            "book_candidates": [
                {
                    "uci": mv.uci(),
                    "san": board.san(mv),
                    "play_count": int(c["play_count"]),
                    "weighted_count": round(float(c.get("weighted_count", c["play_count"])), 1),
                    "win_rate": round(float(c["win_rate"]), 3),
                    "prob": round(float(probs[i]), 4),
                }
                for i, (mv, c) in enumerate(valid_book[:6])
            ],
        }

    def evaluate_position(
        self,
        board: chess.Board,
        precomputed_candidates: list[dict[str, Any]] | None = None,
        deterministic: bool = False,
        allow_book: bool = True,
    ) -> dict[str, Any]:
        """Run the full 7-step decision pipeline and return detailed move provenance.

        Used by `choose_move`, `evaluate_holdout_style`, and `calibration/explain_move.py`.
        """
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            raise ValueError("No legal moves available on this board.")

        side = "white" if board.turn == chess.WHITE else "black"
        cur_ply = effective_ply(board)
        phase = detect_game_phase(board, ply=cur_ply)

        # ── Step 1: Weighted Opening Book ────────────────────────────
        if allow_book:
            book_res = self._select_book_move(board, side=side, deterministic=deterministic)
            if book_res is not None:
                chosen_mv = book_res["move"]
                return {
                    "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    "bot": self.profile.name,
                    "tier": self.profile.tier,
                    "fen": board.fen(),
                    "ply": cur_ply,
                    "side": side,
                    "phase": phase,
                    "source": "book",
                    "move": chosen_mv,
                    "chosen_uci": chosen_mv.uci(),
                    "chosen_san": board.san(chosen_mv),
                    "ranked_ucis": book_res["ranked_ucis"],
                    "engine_top1_uci": None,
                    "equals_engine_top1": None,
                    "cpl": 0,
                    "style_prob": book_res["style_prob"],
                    "continuity_penalty": 0.0,
                    "continuity_reason": "",
                    "error_type": "none",
                    "tactical_override": False,
                    "book_candidates": book_res["book_candidates"],
                    "candidates": [],
                }

        # ── Step 2: Candidate Generation via Full-Strength Stockfish ─
        try:
            if precomputed_candidates is not None:
                engine_candidates = precomputed_candidates[: self.profile.engine_multipv]
            else:
                engine_candidates = self._engine.analyse_candidates(
                    board,
                    multipv=self.profile.engine_multipv,
                    time_limit=self.profile.engine_time_limit,
                    uci_elo=None,  # Full strength: eliminates low-Elo search hallucinations
                )
            valid_cands = [
                c for c in engine_candidates if c.get("move") in board.legal_moves
            ]
        except EngineNotFoundError as exc:
            logger.error(
                "FALLBACK TRIGGERED: Stockfish binary missing for %s (%s); using heuristic fallback.",
                self.profile.name,
                exc,
            )
            valid_cands = []
        except Exception as exc:
            logger.error(
                "FALLBACK TRIGGERED: Stockfish analysis failed for %s (%s); using heuristic fallback.",
                self.profile.name,
                exc,
            )
            valid_cands = []

        # ── Step 7: Fallback Heuristic ONLY on Engine Failure ────────
        if not valid_cands:
            fb_move = fallback_heuristic_move(board, self.profile)
            return {
                "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "bot": self.profile.name,
                "tier": self.profile.tier,
                "fen": board.fen(),
                "ply": cur_ply,
                "side": side,
                "phase": phase,
                "source": "fallback",
                "move": fb_move,
                "chosen_uci": fb_move.uci(),
                "chosen_san": board.san(fb_move),
                "ranked_ucis": [fb_move.uci()],
                "engine_top1_uci": None,
                "equals_engine_top1": None,
                "cpl": None,
                "style_prob": None,
                "continuity_penalty": 0.0,
                "continuity_reason": "",
                "error_type": "none",
                "tactical_override": False,
                "book_candidates": [],
                "candidates": [],
            }

        # Sort candidates descending by score_cp
        valid_cands = sorted(valid_cands, key=lambda c: int(c["score_cp"]), reverse=True)
        best_cand = valid_cands[0]
        best_cp = int(best_cand["score_cp"])
        engine_top1_uci = str(best_cand["uci"])
        best_move_obj: chess.Move = best_cand["move"]
        second_cp = int(valid_cands[1]["score_cp"]) if len(valid_cands) > 1 else best_cp - 500
        top1_margin = best_cp - second_cp

        # ── Step 3: Mandatory Safety Filter ──────────────────────────
        non_mating = [
            c for c in valid_cands if c.get("mate") is None or int(c["mate"]) > 0
        ]
        safe_pool = non_mating if non_mating else [valid_cands[0]]
        safe_best_cp = max(int(c["score_cp"]) for c in safe_pool)

        # Forced mate-in-1 (all tiers) or mate-in-2 (Focus/Condition): always execute
        best_mate = best_cand.get("mate")
        forced_mate_take = (
            best_mate is not None
            and best_mate > 0
            and (best_mate == 1 or (best_mate <= 2 and self.profile.tier != TIER_SICK))
        )

        # Compute move-level style probabilities over ALL legal moves
        style_probs, style_reasons = score_legal_moves_style(
            board,
            color_str=side,
            phase=phase,
            style_profile=self.style_profile,
            prev_own_move=self._last_bot_move,
            prev_opp_move=self._last_opp_move,
        )

        uniform_prob = 1.0 / max(1, len(legal_moves))
        min_style_floor = min(0.08, self.profile.min_style_prob_ratio * uniform_prob)

        # Annotate every candidate with safety status, style prob, continuity, and error type
        err_band_low = max(45, int(0.55 * self.profile.max_cp_loss))
        annotated: list[dict[str, Any]] = []
        for c in valid_cands:
            mv: chess.Move = c["move"]
            uci = str(c["uci"])
            cpl = max(0, safe_best_cp - int(c["score_cp"]))
            allows_mate = c.get("mate") is not None and int(c["mate"]) < 0
            in_safe_band = (not allows_mate) and (cpl <= self.profile.max_cp_loss)
            in_error_band = (
                (not allows_mate)
                and (uci != engine_top1_uci)
                and (err_band_low <= cpl <= self.profile.mistake_max_cp_loss)
            )
            s_prob = float(style_probs.get(uci, uniform_prob))
            # Include bounded tier DNA bonus as a tiny calibrated refinement (max ±0.12 logit)
            dna_bonus = compute_style_bonus(board, mv, phase, self.tier_style)
            cont_pen, cont_reason = _compute_continuity_penalty(
                board, mv, self._last_bot_move, self.profile.consistency_penalty
            )
            tactical_just = (
                uci == engine_top1_uci and top1_margin >= self.profile.tactical_override_cp
            )
            below_floor = (s_prob < min_style_floor) and (not tactical_just)

            # Style-PRIMARY score: log(style_prob) dominates; engine CPL is mild tie-breaker
            style_logit = (
                math.log(max(s_prob, 1e-5))
                + (dna_bonus / 150.0)
                + (0.0 if tactical_just else cont_pen)
                - (self.profile.eval_tiebreak_weight * (cpl / 100.0))
            )

            annotated.append(
                {
                    "move": mv,
                    "uci": uci,
                    "san": board.san(mv),
                    "score_cp": int(c["score_cp"]),
                    "mate": c.get("mate"),
                    "cpl": cpl,
                    "allows_mate": allows_mate,
                    "in_safe_band": in_safe_band,
                    "in_error_band": in_error_band,
                    "style_prob": round(s_prob, 4),
                    "below_style_floor": below_floor,
                    "tactical_override": tactical_just,
                    "continuity_penalty": round(0.0 if tactical_just else cont_pen, 3),
                    "continuity_reason": "" if tactical_just else cont_reason,
                    "style_logit": round(style_logit, 4),
                    "top_features": style_reasons.get(uci, []),
                    "error_type": classify_error_type(board, mv, best_move_obj, phase, cpl),
                }
            )

        # Safe candidates surviving Step 3
        safe_list = [a for a in annotated if a["in_safe_band"]]
        if not safe_list:
            # Always keep the single best non-mating move (or best overall if all forced mate)
            safe_list = [annotated[0]]

        # Step 6: Apply style-probability floor unless all safe candidates are below floor
        floor_filtered = [a for a in safe_list if not a["below_style_floor"]]
        eligible_safe = floor_filtered if floor_filtered else safe_list

        # Rank eligible_safe by style-PRIMARY logit
        eligible_safe_sorted = sorted(
            eligible_safe, key=lambda a: a["style_logit"], reverse=True
        )
        ranked_ucis = [a["uci"] for a in eligible_safe_sorted]
        # Append remaining safe/annotated moves to ranked_ucis for Top-3 diagnostics
        for a in annotated:
            if a["uci"] not in ranked_ucis:
                ranked_ucis.append(a["uci"])

        # ── Forced Mate Shortcut ─────────────────────────────────────
        if forced_mate_take:
            chosen = annotated[0]
            source = "style-selected"
        else:
            # ── Step 5: Calibrated Error Model ───────────────────────
            error_band = [
                a for a in annotated if a["in_error_band"] and not a["below_style_floor"]
            ]
            trigger_error = False
            if (
                not deterministic
                and self.profile.error_rate_scale > 0.0
                and error_band
                and len(safe_list) >= 1
            ):
                base_err_rate, err_dist = self._get_phase_mistake_config(side, phase)
                effective_err_prob = min(0.45, base_err_rate * self.profile.error_rate_scale)
                if random.random() < effective_err_prob:
                    trigger_error = True

            if trigger_error:
                # Score error-band moves by style probability × Ken's recurring error-type likelihood
                _, err_dist = self._get_phase_mistake_config(side, phase)
                err_logits = np.array(
                    [
                        math.log(max(a["style_prob"], 1e-4))
                        + math.log(max(float(err_dist.get(a["error_type"], 0.10)), 0.05))
                        + a["continuity_penalty"]
                        for a in error_band
                    ],
                    dtype=np.float64,
                )
                err_logits -= np.max(err_logits)
                err_exp = np.exp(err_logits / max(0.35, self.profile.engine_temp))
                err_probs = err_exp / float(np.sum(err_exp))
                pick_idx = int(np.random.choice(len(error_band), p=err_probs))
                chosen = error_band[pick_idx]
                source = "error-model"
            else:
                # ── Step 4: Style-Primary Selection ──────────────────
                if deterministic or len(eligible_safe_sorted) == 1:
                    chosen = eligible_safe_sorted[0]
                else:
                    s_logits = np.array(
                        [a["style_logit"] for a in eligible_safe_sorted], dtype=np.float64
                    )
                    temp = max(0.10, self.profile.engine_temp)
                    scaled = (s_logits - np.max(s_logits)) / temp
                    exp_vals = np.exp(scaled)
                    probs = exp_vals / float(np.sum(exp_vals))
                    pick_idx = int(np.random.choice(len(eligible_safe_sorted), p=probs))
                    chosen = eligible_safe_sorted[pick_idx]
                source = "style-selected"

        return {
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "bot": self.profile.name,
            "tier": self.profile.tier,
            "fen": board.fen(),
            "ply": cur_ply,
            "side": side,
            "phase": phase,
            "source": source,
            "move": chosen["move"],
            "chosen_uci": chosen["uci"],
            "chosen_san": chosen["san"],
            "ranked_ucis": ranked_ucis,
            "engine_top1_uci": engine_top1_uci,
            "equals_engine_top1": chosen["uci"] == engine_top1_uci,
            "cpl": chosen["cpl"],
            "style_prob": chosen["style_prob"],
            "continuity_penalty": chosen["continuity_penalty"],
            "continuity_reason": chosen["continuity_reason"],
            "error_type": chosen["error_type"],
            "tactical_override": chosen["tactical_override"],
            "book_candidates": [],
            "candidates": [
                {
                    "uci": a["uci"],
                    "san": a["san"],
                    "score_cp": a["score_cp"],
                    "cpl": a["cpl"],
                    "in_safe_band": a["in_safe_band"],
                    "in_error_band": a["in_error_band"],
                    "style_prob": a["style_prob"],
                    "below_style_floor": a["below_style_floor"],
                    "continuity_penalty": a["continuity_penalty"],
                    "style_logit": a["style_logit"],
                    "top_features": a["top_features"],
                    "error_type": a["error_type"],
                }
                for a in annotated
            ],
        }

    def choose_move(self, board: chess.Board) -> chess.Move:
        """Choose the next move using the 7-step style-first pipeline and record provenance."""
        self._sync_game_state(board)
        self.move_count += 1

        decision = self.evaluate_position(board, deterministic=False, allow_book=True)
        chosen_move: chess.Move = decision["move"]
        source: str = decision["source"]

        self.last_move_source = source
        if source == "book":
            self.book_move_count += 1
        elif source == "style-selected":
            self.engine_move_count += 1
            self.style_move_count += 1
        elif source == "error-model":
            self.engine_move_count += 1
            self.error_move_count += 1
        else:
            self.fallback_count += 1

        self._last_bot_move = chosen_move

        # Store serializable provenance (omit chess.Move object)
        prov = {k: v for k, v in decision.items() if k != "move"}
        self.last_move_provenance = prov
        self._write_provenance_record(prov)
        return chosen_move

    def close(self) -> None:
        """Release engine resources."""
        self._engine.stop()


@dataclass(frozen=True)
class PositionEval:
    """Full-strength engine evaluation snapshot from side-to-move's perspective."""

    best_cp: int
    second_best_cp: int | None
    best_mate: int | None
    best_uci: str | None


@dataclass(frozen=True)
class PlayerMoveEvaluation:
    """Classified human move result emitted asynchronously by BotWorker."""

    generation: int
    move_ply: int
    player_move_number: int
    event: str | None
    cp_loss: int


def classify_blunder(
    eval_before_cp: int,
    eval_after_cp: int,
    mate_before: int | None = None,
    mate_after: int | None = None,
    mover_color: chess.Color = chess.WHITE,
    *,
    scores_from_white: bool = False,
) -> bool:
    """Return True if the human mover's move was a blunder from their perspective.

    Rules:
    - If ``scores_from_white=True``, converts White-POV centipawns & mate scores to
      mover's POV (negating when ``mover_color == chess.BLACK``). Otherwise expects
      scores already expressed from the mover's perspective.
    - Allows a forced mate against the mover that was not there before -> True.
    - Throws away a forced mate for the mover into a non-winning eval (< 200 cp) -> True.
    - Skips positions already decided (``|eval| >= DECIDED_EVAL_CP`` in the same
      direction both before and after) unless a new forced mate against was allowed.
    - Loses ``>= BLUNDER_CP_LOSS`` (200 cp) versus best play -> True.
    """
    sign = -1 if (scores_from_white and mover_color == chess.BLACK) else 1
    before_cp = int(eval_before_cp) * sign
    after_cp = int(eval_after_cp) * sign
    m_before = (-int(mate_before) if mate_before is not None else None) if sign < 0 else mate_before
    m_after = (-int(mate_after) if mate_after is not None else None) if sign < 0 else mate_after

    # Allowing a forced mate that was not forced before is always a blunder
    had_mate_against = m_before is not None and m_before < 0
    now_mate_against = m_after is not None and m_after < 0
    if now_mate_against and not had_mate_against:
        return True

    # Skip positions already completely decided in the same direction (mop-up phase)
    if (
        abs(before_cp) >= DECIDED_EVAL_CP
        and abs(after_cp) >= DECIDED_EVAL_CP
        and (before_cp > 0) == (after_cp > 0)
    ):
        return False

    # Throwing away a forced mating sequence into an unclear/losing position
    if m_before is not None and m_before > 0 and m_after is None and after_cp < 200:
        return True

    return (before_cp - after_cp) >= BLUNDER_CP_LOSS


def is_tactical_sacrifice(board_before: chess.Board, move: chess.Move) -> bool:
    """Heuristic check for a real tactical piece sacrifice (used for Brilliant detection)."""
    mover_color = board_before.turn
    opp_color = not mover_color
    moving_piece = board_before.piece_at(move.from_square)
    if moving_piece is None:
        return False

    captured_piece = board_before.piece_at(move.to_square)
    captured_val = _PIECE_VALUES.get(captured_piece.piece_type, 0) if captured_piece else 0
    moving_val = _PIECE_VALUES.get(moving_piece.piece_type, 0)

    board_after = board_before.copy(stack=False)
    board_after.push(move)

    # 1. Moving a minor/major piece onto an opponent-attacked square for less immediate gain
    if moving_piece.piece_type in (chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN):
        if board_after.is_attacked_by(opp_color, move.to_square):
            attackers = board_after.attackers(opp_color, move.to_square)
            cheapest_attacker = min(
                (
                    _PIECE_VALUES.get(board_after.piece_at(sq).piece_type, 20000)
                    for sq in attackers
                    if board_after.piece_at(sq) is not None
                ),
                default=20000,
            )
            defended = board_after.is_attacked_by(mover_color, move.to_square)
            if (not defended and moving_val > captured_val + 100) or (
                cheapest_attacker < moving_val and moving_val > captured_val + 100
            ):
                return True

    # 2. Quiet/tactical move leaving another friendly non-pawn piece newly en prise
    for sq in chess.SQUARES:
        if sq == move.to_square:
            continue
        p = board_after.piece_at(sq)
        if p is None or p.color != mover_color:
            continue
        if p.piece_type not in (chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN):
            continue
        val = _PIECE_VALUES.get(p.piece_type, 0)
        if val <= captured_val + 100:
            continue
        if board_after.is_attacked_by(opp_color, sq) and not board_after.is_attacked_by(
            mover_color, sq
        ):
            if not board_before.is_attacked_by(opp_color, sq):
                return True

    return False


def classify_player_move(
    eval_before_cp: int,
    eval_after_cp: int,
    *,
    mate_before: int | None = None,
    mate_after: int | None = None,
    second_best_cp: int | None = None,
    mover_color: chess.Color = chess.WHITE,
    scores_from_white: bool = False,
    is_sacrifice: bool = False,
    ply: int = 10,
    legal_move_count: int = 10,
) -> str | None:
    """Classify a human move into one of the bot reaction event categories or ``None``.

    Possible return values:
      ``'PLAYER_BLUNDER'``, ``'PLAYER_MISTAKE'``, ``'PLAYER_DUBIOUS'``,
      ``'PLAYER_BRILLIANT'``, ``'PLAYER_GREAT'``, or ``None``.
    """
    if classify_blunder(
        eval_before_cp,
        eval_after_cp,
        mate_before=mate_before,
        mate_after=mate_after,
        mover_color=mover_color,
        scores_from_white=scores_from_white,
    ):
        return "PLAYER_BLUNDER"

    sign = -1 if (scores_from_white and mover_color == chess.BLACK) else 1
    before_cp = int(eval_before_cp) * sign
    after_cp = int(eval_after_cp) * sign
    sec_cp = (int(second_best_cp) * sign) if second_best_cp is not None else None
    m_after = (-int(mate_after) if mate_after is not None else None) if sign < 0 else mate_after

    # Skip positions already completely decided in the same direction
    if (
        abs(before_cp) >= DECIDED_EVAL_CP
        and abs(after_cp) >= DECIDED_EVAL_CP
        and (before_cp > 0) == (after_cp > 0)
    ):
        return None

    cp_loss = max(0, before_cp - after_cp)
    if cp_loss >= MISTAKE_CP_LOSS:
        return "PLAYER_MISTAKE"
    if cp_loss >= DUBIOUS_CP_LOSS:
        return "PLAYER_DUBIOUS"

    # Positive accolades require non-forced, post-opening tactical accuracy
    if legal_move_count < 3 or cp_loss > BEST_MOVE_TOLERANCE_CP or after_cp < -50:
        return None

    best_gap = (before_cp - sec_cp) if sec_cp is not None else 0

    if ply >= 8 and is_sacrifice and before_cp < 650:
        if best_gap >= BRILLIANT_GAP_CP or (
            m_after is not None and m_after >= 2 and (sec_cp is None or sec_cp < 300)
        ):
            return "PLAYER_BRILLIANT"

    if ply >= 6 and abs(before_cp) < 700 and sec_cp is not None and best_gap >= GREAT_ONLY_MOVE_GAP_CP:
        return "PLAYER_GREAT"

    return None


class BotWorker:
    """Non-blocking background thread worker for bot moves and full-strength player-move eval."""

    def __init__(self, bot: KenBot) -> None:
        self.bot = bot
        self._request_queue: queue.Queue[
            tuple[str, chess.Board | None, float, dict[str, Any] | None]
        ] = queue.Queue()
        self._result_queue: queue.Queue[
            tuple[chess.Move | None, Exception | None, str, float]
        ] = queue.Queue()
        self._eval_result_queue: queue.Queue[PlayerMoveEvaluation] = queue.Queue()
        self._eval_cache: dict[str, PositionEval] = {}
        self._stop_event = threading.Event()
        self._is_computing = threading.Event()
        self.last_move_source: str = "none"
        self.last_compute_seconds: float = 0.0
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="BotWorkerThread"
        )
        self._thread.start()

    def request_move(
        self,
        board: chess.Board,
        delay_seconds: float = 0.0,
        player_move_context: dict[str, Any] | None = None,
    ) -> None:
        """Submit a board position to compute a bot move (and optional post-move player eval)."""
        self._is_computing.set()
        self._request_queue.put(("move", board.copy(), delay_seconds, player_move_context))

    def request_position_eval(self, board: chess.Board) -> None:
        """Pre-evaluate a human-to-move position in the background while idle."""
        if board.is_game_over():
            return
        if board.epd() in self._eval_cache:
            return
        self._request_queue.put(("pre_eval", board.copy(stack=False), 0.0, None))

    def poll_move(self) -> chess.Move | None:
        """Poll for a computed bot move non-blockingly. Returns Move or None."""
        try:
            move, exc, source, compute_s = self._result_queue.get_nowait()
            self._is_computing.clear()
            self.last_move_source = source
            self.last_compute_seconds = compute_s
            if exc is not None:
                raise exc
            return move
        except queue.Empty:
            return None

    def poll_player_eval(self) -> PlayerMoveEvaluation | None:
        """Poll non-blockingly for an asynchronously classified human move evaluation."""
        try:
            return self._eval_result_queue.get_nowait()
        except queue.Empty:
            return None

    @property
    def is_thinking(self) -> bool:
        """Return True if the worker is actively computing or has pending bot move requests."""
        return self._is_computing.is_set()

    def stop(self) -> None:
        """Signal the worker to terminate and wait briefly for thread exit."""
        self._stop_event.set()
        self._request_queue.put(("stop", None, 0.0, None))
        if self._thread.is_alive():
            self._thread.join(timeout=1.0)
        self.bot.close()

    def _cache_put(self, epd: str, pos_eval: PositionEval) -> None:
        if len(self._eval_cache) >= PLAYER_EVAL_CACHE_MAX and epd not in self._eval_cache:
            oldest_key = next(iter(self._eval_cache))
            self._eval_cache.pop(oldest_key, None)
        self._eval_cache[epd] = pos_eval

    def _evaluate_full_strength(
        self, board: chess.Board, *, multipv: int = 2
    ) -> PositionEval | None:
        """Run or retrieve cached full-strength Stockfish eval (uci_elo=None) on worker thread."""
        epd = board.epd()
        cached = self._eval_cache.get(epd)
        if cached is not None and (multipv == 1 or cached.second_best_cp is not None):
            return cached
        if board.is_game_over():
            if board.is_checkmate():
                res = PositionEval(best_cp=-10000, second_best_cp=None, best_mate=0, best_uci=None)
            else:
                res = PositionEval(best_cp=0, second_best_cp=None, best_mate=None, best_uci=None)
            self._cache_put(epd, res)
            return res

        try:
            cands = self.bot.engine.analyse_candidates(
                board,
                multipv=multipv,
                time_limit=PLAYER_EVAL_TIME_LIMIT,
                uci_elo=None,  # Full uncapped strength independent of bot Elo
            )
        except Exception:
            return None
        if not cands:
            return None

        cands_sorted = sorted(cands, key=lambda c: int(c["score_cp"]), reverse=True)
        top = cands_sorted[0]
        second_cp = int(cands_sorted[1]["score_cp"]) if len(cands_sorted) > 1 else None
        res = PositionEval(
            best_cp=int(top["score_cp"]),
            second_best_cp=second_cp,
            best_mate=top.get("mate"),
            best_uci=str(top.get("uci")) if top.get("uci") else None,
        )
        self._cache_put(epd, res)
        return res

    def _cache_bot_candidates(self, board: chess.Board) -> None:
        """Reuse full-strength candidates just computed by KenBot to avoid redundant searches."""
        prov = self.bot.last_move_provenance
        cands = prov.get("candidates") if isinstance(prov, dict) else None
        if not cands:
            return
        try:
            ordered = sorted(cands, key=lambda c: int(c["score_cp"]), reverse=True)
            top = ordered[0]
            second_cp = int(ordered[1]["score_cp"]) if len(ordered) > 1 else None
            self._cache_put(
                board.epd(),
                PositionEval(
                    best_cp=int(top["score_cp"]),
                    second_best_cp=second_cp,
                    best_mate=top.get("mate"),
                    best_uci=str(top.get("uci")) if top.get("uci") else None,
                ),
            )
        except Exception:
            pass

    def _evaluate_player_move_context(
        self, board_after: chess.Board, ctx: dict[str, Any]
    ) -> None:
        """Evaluate the human's move at full strength and publish any reaction event."""
        if self._stop_event.is_set():
            return
        try:
            board_before = chess.Board(str(ctx["board_before_fen"]))
            move = chess.Move.from_uci(str(ctx["move_uci"]))
        except Exception:
            return

        legal_count = sum(1 for _ in board_before.legal_moves)
        eval_before = self._evaluate_full_strength(board_before, multipv=2)
        if eval_before is None or self._stop_event.is_set():
            return

        eval_after_opp = self._evaluate_full_strength(board_after, multipv=1)
        if eval_after_opp is None or self._stop_event.is_set():
            return

        # Convert opponent-to-move score after human's move into human mover's POV
        after_mover_cp = -eval_after_opp.best_cp
        if board_after.is_checkmate():
            # Human delivered checkmate!
            after_mover_mate: int | None = 1
        elif eval_after_opp.best_mate is not None:
            after_mover_mate = -int(eval_after_opp.best_mate)
        else:
            after_mover_mate = None

        sac = is_tactical_sacrifice(board_before, move)
        event = classify_player_move(
            eval_before.best_cp,
            after_mover_cp,
            mate_before=eval_before.best_mate,
            mate_after=after_mover_mate,
            second_best_cp=eval_before.second_best_cp,
            mover_color=board_before.turn,
            scores_from_white=False,
            is_sacrifice=sac,
            ply=int(ctx.get("move_ply", 0)),
            legal_move_count=legal_count,
        )
        cp_loss = max(0, eval_before.best_cp - after_mover_cp)
        self._eval_result_queue.put(
            PlayerMoveEvaluation(
                generation=int(ctx["generation"]),
                move_ply=int(ctx["move_ply"]),
                player_move_number=int(ctx["player_move_number"]),
                event=event,
                cp_loss=cp_loss,
            )
        )

    def _run(self) -> None:
        engine_warmed = False
        while not self._stop_event.is_set():
            try:
                kind, board, delay, ctx = self._request_queue.get(timeout=0.05)
            except queue.Empty:
                # Silently pre-warm Stockfish while idle so the first out-of-book move has no hitch
                if not engine_warmed and not self._stop_event.is_set():
                    engine_warmed = True
                    try:
                        self.bot.engine.start()
                    except Exception:
                        pass
                continue

            if self._stop_event.is_set() or kind == "stop" or board is None:
                break

            if kind == "pre_eval":
                # Skip stale pre-eval if a real bot move request is already waiting
                if self._request_queue.empty():
                    self._evaluate_full_strength(board, multipv=2)
                continue

            # kind == "move": compute and emit bot reply FIRST so bot pacing is never delayed
            try:
                t_start = time.perf_counter()
                if delay > 0:
                    time.sleep(delay)
                move = self.bot.choose_move(board)
                elapsed = time.perf_counter() - t_start
                self._cache_bot_candidates(board)
                self._result_queue.put((move, None, self.bot.last_move_source, elapsed))
            except Exception as exc:
                self._result_queue.put((None, exc, "error", 0.0))
                continue

            # Evaluate the human's move right after releasing the bot move (reusing cached scores)
            if ctx is not None and not self._stop_event.is_set():
                self._evaluate_player_move_context(board, ctx)
