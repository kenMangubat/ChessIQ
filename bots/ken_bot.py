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

logger = logging.getLogger(__name__)

_MAX_PROVENANCE_LOG_FILES = 30
_MAX_PROVENANCE_LOG_BYTES = 2 * 1024 * 1024  # 2 MB per session file


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
        tagline="Bai gihilantan ko, pildiha daw ko bi",
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
        tagline="Ha?? sure ka makig duwa ka nako",
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
        tagline="hmmm dependi kong 3 yan!",
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


class BotWorker:
    """Non-blocking background thread worker to compute bot moves without freezing Pygame."""

    def __init__(self, bot: KenBot) -> None:
        self.bot = bot
        self._request_queue: queue.Queue[tuple[chess.Board | None, float]] = queue.Queue()
        self._result_queue: queue.Queue[
            tuple[chess.Move | None, Exception | None, str, float]
        ] = queue.Queue()
        self._stop_event = threading.Event()
        self._is_computing = threading.Event()
        self.last_move_source: str = "none"
        self.last_compute_seconds: float = 0.0
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="BotWorkerThread"
        )
        self._thread.start()

    def request_move(self, board: chess.Board, delay_seconds: float = 0.0) -> None:
        """Submit a board position to compute a move in the background."""
        self._is_computing.set()
        self._request_queue.put((board.copy(), delay_seconds))

    def poll_move(self) -> chess.Move | None:
        """Poll for a computed move non-blockingly. Returns Move or None."""
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

    @property
    def is_thinking(self) -> bool:
        """Return True if the worker is actively computing or has pending requests."""
        return self._is_computing.is_set() or not self._request_queue.empty()

    def stop(self) -> None:
        """Signal the worker to terminate and wait briefly for thread exit."""
        self._stop_event.set()
        self._request_queue.put((None, 0.0))
        if self._thread.is_alive():
            self._thread.join(timeout=1.0)
        self.bot.close()

    def _run(self) -> None:
        engine_warmed = False
        while not self._stop_event.is_set():
            try:
                board, delay = self._request_queue.get(timeout=0.05)
            except queue.Empty:
                # Silently pre-warm Stockfish while idle so the first out-of-book move has no hitch
                if not engine_warmed and not self._stop_event.is_set():
                    engine_warmed = True
                    try:
                        self.bot.engine.start()
                    except Exception:
                        pass
                continue

            if self._stop_event.is_set() or board is None:
                break

            try:
                t_start = time.perf_counter()
                if delay > 0:
                    time.sleep(delay)
                move = self.bot.choose_move(board)
                elapsed = time.perf_counter() - t_start
                self._result_queue.put((move, None, self.bot.last_move_source, elapsed))
            except Exception as exc:
                self._result_queue.put((None, exc, "error", 0.0))
