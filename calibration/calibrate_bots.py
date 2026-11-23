"""Offline bot strength calibration harness for ChessIQ ("Chess Ken").

Measures the actual playing strength, Average Centipawn Loss (ACPL), blunder rate
(>200 cp loss per 100 moves), and fallback usage of each KenBot profile against
fixed Stockfish UCI_Elo reference levels across varied opening positions.

Usage:
    python -m calibration.calibrate_bots            # standard calibration run
    python -m calibration.calibrate_bots --quick    # fast smoke-test mode
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import datetime
import json
import math
from pathlib import Path
import time
from typing import Any

import chess

from bots.ken_bot import KEN_PROFILES, KenBot, get_profile
from others.config import BOOK_DB_PATH, CALIBRATION_CSV_PATH, CALIBRATION_JSON_PATH
from engine.stockfish_engine import StockfishEngine

# ── Standard varied opening suites (0 to 4 plies) ───────────────────
OPENING_POSITIONS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("Start Position", ()),
    ("Sicilian Defense (1.e4 c5)", ("e2e4", "c7c5")),
    ("Open Game (1.e4 e5 2.Nf3 Nc6)", ("e2e4", "e7e5", "g1f3", "b8c6")),
    ("Caro-Kann Defense (1.e4 c6)", ("e2e4", "c7c6")),
    ("French Defense (1.e4 e6 2.d4 d5)", ("e2e4", "e7e6", "d2d4", "d7d5")),
    ("Queen's Gambit (1.d4 d5 2.c4 e6)", ("d2d4", "d7d5", "c2c4", "e7e6")),
    ("Indian Defense (1.d4 Nf6 2.c4 e6)", ("d2d4", "g8f6", "c2c4", "e7e6")),
    ("Pirc / Modern (1.e4 d6 2.d4 Nf6)", ("e2e4", "d7d6", "d2d4", "g8f6")),
)

DEFAULT_REFERENCE_ELOS: tuple[int, ...] = (1350, 1600, 1800, 2000, 2200, 2400)

# Profile-targeted reference ladders for fast mode so each bot is tested near its target
QUICK_PROFILE_ELOS: dict[str, tuple[int, ...]] = {
    "Sick Ken": (1350, 1550),
    "Focus Ken": (1650, 1950),
    "Condition Ken": (2200, 2400),
}

_EVAL_CLAMP_CP = 1000
_BLUNDER_THRESHOLD_CP = 200


def _clamp_cp(score_cp: int) -> int:
    return max(-_EVAL_CLAMP_CP, min(_EVAL_CLAMP_CP, score_cp))


def estimate_elo_mle(
    game_results: list[tuple[int, float]],
    prior_elo: float | None = None,
) -> tuple[float, float, float]:
    """Estimate Elo rating and 95% confidence interval via Maximum Likelihood Estimation.

    Parameters
    ----------
    game_results : list of (opponent_elo, score)
        Each entry has opponent_elo (int) and score in {1.0, 0.5, 0.0}.
    prior_elo : float | None
        Weak regularizing anchor (equivalent to one 0.5 draw at the mean opponent
        Elo) to keep 0% or 100% small-sample scores finite.

    Returns
    -------
    (estimated_elo, ci_low, ci_high)
    """
    if not game_results:
        return (1500.0, 1200.0, 1800.0)

    mean_opp = sum(r for r, _ in game_results) / len(game_results)
    anchor = prior_elo if prior_elo is not None else mean_opp

    # Weak Bayesian prior of 0.5 virtual draw at anchor so 100%/0% samples converge cleanly
    augmented: list[tuple[float, float, float]] = [
        (float(opp_elo), float(score), 1.0) for opp_elo, score in game_results
    ]
    augmented.append((float(anchor), 0.5, 0.5))

    target_sum = sum(score * weight for _, score, weight in augmented)

    lo, hi = 600.0, 3200.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        expected = sum(
            weight / (1.0 + 10.0 ** ((opp - mid) / 400.0))
            for opp, _, weight in augmented
        )
        if expected < target_sum:
            lo = mid
        else:
            hi = mid

    est_elo = 0.5 * (lo + hi)

    # Fisher information for standard error
    scale = math.log(10.0) / 400.0
    fisher_info = 0.0
    for opp, _, weight in augmented:
        p = 1.0 / (1.0 + 10.0 ** ((opp - est_elo) / 400.0))
        fisher_info += weight * (scale ** 2) * p * (1.0 - p)

    se = 1.0 / math.sqrt(max(fisher_info, 1e-6))
    half_width = min(450.0, 1.96 * se)
    return (
        round(est_elo, 1),
        round(est_elo - half_width, 1),
        round(est_elo + half_width, 1),
    )


def play_calibration_game(
    profile_name: str,
    ref_elo: int,
    bot_is_white: bool,
    opening_name: str,
    opening_moves: tuple[str, ...],
    ref_time_limit: float = 0.05,
    eval_depth: int = 8,
    max_plies: int = 120,
    db_path: Path = BOOK_DB_PATH,
) -> dict[str, Any]:
    """Play a single calibration game between a KenBot profile and a reference Stockfish level."""
    board = chess.Board()
    for uci in opening_moves:
        move = chess.Move.from_uci(uci)
        if move in board.legal_moves:
            board.push(move)

    bot_color = chess.WHITE if bot_is_white else chess.BLACK

    # Dedicated engine instances per game worker to avoid state collisions
    bot_engine = StockfishEngine(threads=1, hash_mb=16)
    ref_engine = StockfishEngine(threads=1, hash_mb=16)
    bot = KenBot(
        profile=profile_name,
        db_path=db_path,
        engine=bot_engine,
        log_provenance=False,
    )

    bot_cpls: list[int] = []
    blunders = 0
    plies_played = 0
    oob_moves = 0
    oob_sf_top1 = 0
    erratic_moves = 0
    worst_offenders: list[dict[str, Any]] = []

    try:
        while not board.is_game_over(claim_draw=True) and plies_played < max_plies:
            plies_played += 1

            if board.turn == bot_color:
                pre_fen = board.fen()
                # 1. Full-strength baseline evaluation before bot's move
                pre_cands = ref_engine.analyse_candidates(
                    board,
                    multipv=1,
                    time_limit=0.0,
                    depth=eval_depth,
                    uci_elo=None,
                )
                best_cp = _clamp_cp(pre_cands[0]["score_cp"]) if pre_cands else 0
                best_uci = pre_cands[0]["uci"] if pre_cands else ""

                # 2. Bot selects its move
                chosen_move = bot.choose_move(board)
                if chosen_move not in board.legal_moves:
                    chosen_move = next(iter(board.legal_moves))
                chosen_san = board.san(chosen_move)
                prov = bot.last_move_provenance

                board.push(chosen_move)

                # 3. Compute centipawn loss vs best engine choice
                if chosen_move.uci() == best_uci:
                    cpl = 0
                elif board.is_checkmate():
                    cpl = 0
                else:
                    post_cands = ref_engine.analyse_candidates(
                        board,
                        multipv=1,
                        time_limit=0.0,
                        depth=max(4, eval_depth - 1),
                        uci_elo=None,
                    )
                    after_cp = -_clamp_cp(post_cands[0]["score_cp"]) if post_cands else best_cp
                    cpl = max(0, best_cp - after_cp)

                bot_cpls.append(cpl)
                if cpl > _BLUNDER_THRESHOLD_CP:
                    blunders += 1

                # Track out-of-book engine-top-1 agreement and erratic moves
                src = prov.get("source", bot.last_move_source)
                if src != "book":
                    oob_moves += 1
                    if chosen_move.uci() == best_uci or prov.get("equals_engine_top1") is True:
                        oob_sf_top1 += 1
                    cap = bot.profile.mistake_max_cp_loss
                    # Erratic: violates hard mistake cap (beyond depth noise margin) or
                    # falls below style floor without tactical justification
                    chosen_cand = next(
                        (c for c in prov.get("candidates", []) if c["uci"] == chosen_move.uci()),
                        None,
                    )
                    is_unjustified_low_style = bool(
                        cpl > 30
                        and chosen_cand
                        and chosen_cand.get("below_style_floor")
                    )
                    if cpl > cap + 50 or is_unjustified_low_style:
                        erratic_moves += 1
                        if len(worst_offenders) < 3:
                            worst_offenders.append(
                                {
                                    "fen": pre_fen,
                                    "move_san": chosen_san,
                                    "move_uci": chosen_move.uci(),
                                    "best_uci": best_uci,
                                    "cpl": cpl,
                                    "source": src,
                                    "style_prob": prov.get("style_prob"),
                                    "reason": "low_style_prob" if is_unjustified_low_style else "exceeded_cap",
                                }
                            )
            else:
                # Reference Stockfish opponent plays at fixed UCI_Elo
                ref_move = ref_engine.play_move(
                    board,
                    time_limit=ref_time_limit,
                    uci_elo=ref_elo,
                )
                board.push(ref_move)

        # Determine game score for bot (1.0 win, 0.5 draw, 0.0 loss)
        outcome = board.outcome(claim_draw=True)
        if outcome is not None:
            if outcome.winner is None:
                bot_score = 0.5
                termination = outcome.termination.name
            elif outcome.winner == bot_color:
                bot_score = 1.0
                termination = outcome.termination.name
            else:
                bot_score = 0.0
                termination = outcome.termination.name
        else:
            # Adjudicate capped game by full-strength evaluation (>250 cp = win, <-250 cp = loss)
            adj = ref_engine.analyse_candidates(
                board, multipv=1, time_limit=0.0, depth=eval_depth, uci_elo=None
            )
            raw_cp = adj[0]["score_cp"] if adj else 0
            bot_persp_cp = raw_cp if board.turn == bot_color else -raw_cp
            if bot_persp_cp >= 250:
                bot_score = 1.0
            elif bot_persp_cp <= -250:
                bot_score = 0.0
            else:
                bot_score = 0.5
            termination = "MAX_PLIES_ADJUDICATED"

        return {
            "profile": profile_name,
            "ref_elo": ref_elo,
            "bot_side": "white" if bot_is_white else "black",
            "opening": opening_name,
            "plies": plies_played,
            "bot_score": bot_score,
            "termination": termination,
            "bot_moves": bot.move_count,
            "book_moves": bot.book_move_count,
            "engine_moves": bot.engine_move_count,
            "style_moves": bot.style_move_count,
            "error_moves": bot.error_move_count,
            "fallback_moves": bot.fallback_count,
            "oob_moves": oob_moves,
            "oob_sf_top1": oob_sf_top1,
            "erratic_moves": erratic_moves,
            "worst_offenders": worst_offenders,
            "sum_cpl": sum(bot_cpls),
            "evaluated_moves": len(bot_cpls),
            "blunders": blunders,
        }
    finally:
        bot.close()
        ref_engine.stop()


def run_calibration(
    profile_names: list[str] | None = None,
    reference_elos: list[int] | None = None,
    games_per_level: int = 4,
    ref_time_limit: float = 0.05,
    eval_depth: int = 7,
    max_plies: int = 90,
    workers: int = 4,
    quick: bool = False,
    json_path: Path = CALIBRATION_JSON_PATH,
    csv_path: Path = CALIBRATION_CSV_PATH,
    verbose: bool = True,
) -> dict[str, Any]:
    """Run the calibration suite across profiles and export JSON/CSV reports."""
    t0 = time.time()
    names = profile_names or list(KEN_PROFILES.keys())

    if quick:
        games_per_level = min(games_per_level, 2)
        max_plies = min(max_plies, 60)
        eval_depth = min(eval_depth, 6)
        ref_time_limit = min(ref_time_limit, 0.03)

    tasks: list[tuple[str, int, bool, str, tuple[str, ...]]] = []
    for p_name in names:
        if reference_elos is not None:
            elos = reference_elos
        elif quick and p_name in QUICK_PROFILE_ELOS:
            elos = list(QUICK_PROFILE_ELOS[p_name])
        else:
            elos = list(DEFAULT_REFERENCE_ELOS)

        for r_elo in elos:
            for g_idx in range(games_per_level):
                bot_white = (g_idx % 2) == 0
                op_name, op_moves = OPENING_POSITIONS[g_idx % len(OPENING_POSITIONS)]
                tasks.append((p_name, r_elo, bot_white, op_name, op_moves))

    if verbose:
        mode_str = "QUICK SMOKE TEST" if quick else "FULL CALIBRATION"
        print(f"Starting {mode_str}: {len(tasks)} games across {len(names)} profiles (workers={workers})...")

    game_records: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        future_map = {
            pool.submit(
                play_calibration_game,
                p_name,
                r_elo,
                bot_white,
                op_name,
                op_moves,
                ref_time_limit,
                eval_depth,
                max_plies,
            ): (p_name, r_elo, bot_white, op_name)
            for (p_name, r_elo, bot_white, op_name, op_moves) in tasks
        }

        for idx, future in enumerate(as_completed(future_map), start=1):
            rec = future.result()
            game_records.append(rec)
            if verbose:
                print(
                    f"  [{idx:2d}/{len(tasks):2d}] {rec['profile']:13s} vs SF-{rec['ref_elo']} "
                    f"({rec['bot_side']:5s}) -> score={rec['bot_score']:.1f}, "
                    f"plies={rec['plies']:2d}, blunders={rec['blunders']}"
                )

    # Aggregate metrics per profile
    profile_summaries: dict[str, dict[str, Any]] = {}
    for p_name in names:
        profile_obj = get_profile(p_name)
        p_games = [g for g in game_records if g["profile"] == p_name]

        by_level: dict[int, dict[str, Any]] = {}
        for g in p_games:
            lvl = int(g["ref_elo"])
            bucket = by_level.setdefault(
                lvl, {"games": 0, "points": 0.0, "wins": 0, "draws": 0, "losses": 0}
            )
            bucket["games"] += 1
            bucket["points"] += float(g["bot_score"])
            if g["bot_score"] == 1.0:
                bucket["wins"] += 1
            elif g["bot_score"] == 0.5:
                bucket["draws"] += 1
            else:
                bucket["losses"] += 1

        for lvl, bucket in by_level.items():
            bucket["score_pct"] = round(100.0 * bucket["points"] / max(1, bucket["games"]), 1)

        mle_pairs = [(int(g["ref_elo"]), float(g["bot_score"])) for g in p_games]
        est_elo, ci_low, ci_high = estimate_elo_mle(mle_pairs, prior_elo=float(profile_obj.elo))

        total_eval_moves = sum(int(g["evaluated_moves"]) for g in p_games)
        total_cpl = sum(int(g["sum_cpl"]) for g in p_games)
        total_blunders = sum(int(g["blunders"]) for g in p_games)
        total_bot_moves = sum(int(g["bot_moves"]) for g in p_games)
        total_book_moves = sum(int(g["book_moves"]) for g in p_games)
        total_engine_moves = sum(int(g["engine_moves"]) for g in p_games)
        total_fallback_moves = sum(int(g["fallback_moves"]) for g in p_games)
        total_oob_moves = sum(int(g.get("oob_moves", 0)) for g in p_games)
        total_oob_sf_top1 = sum(int(g.get("oob_sf_top1", 0)) for g in p_games)
        total_erratic_moves = sum(int(g.get("erratic_moves", 0)) for g in p_games)
        all_offenders: list[dict[str, Any]] = []
        for g in p_games:
            all_offenders.extend(g.get("worst_offenders", []))
        all_offenders.sort(key=lambda o: int(o.get("cpl", 0)), reverse=True)

        acpl = round(total_cpl / max(1, total_eval_moves), 1)
        blunder_rate_per_100 = round(100.0 * total_blunders / max(1, total_eval_moves), 2)
        fallback_rate_pct = round(100.0 * total_fallback_moves / max(1, total_bot_moves), 2)
        engine_top1_rate_pct = round(100.0 * total_oob_sf_top1 / max(1, total_oob_moves), 1)
        erratic_rate_pct = round(100.0 * total_erratic_moves / max(1, total_oob_moves), 2)

        profile_summaries[p_name] = {
            "profile": p_name,
            "target_elo": profile_obj.elo,
            "estimated_elo": est_elo,
            "ci_95": [ci_low, ci_high],
            "elo_delta": round(est_elo - profile_obj.elo, 1),
            "games_played": len(p_games),
            "total_points": sum(float(g["bot_score"]) for g in p_games),
            "acpl": acpl,
            "blunder_rate_per_100": blunder_rate_per_100,
            "total_blunders": total_blunders,
            "total_bot_moves": total_bot_moves,
            "book_moves": total_book_moves,
            "engine_moves": total_engine_moves,
            "oob_moves": total_oob_moves,
            "engine_top1_rate_pct": engine_top1_rate_pct,
            "erratic_moves": total_erratic_moves,
            "erratic_rate_pct": erratic_rate_pct,
            "worst_offenders": all_offenders[:5],
            "fallback_moves": total_fallback_moves,
            "fallback_rate_pct": fallback_rate_pct,
            "scores_by_level": {str(k): by_level[k] for k in sorted(by_level)},
        }

    elapsed = round(time.time() - t0, 2)
    report = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "elapsed_seconds": elapsed,
        "quick_mode": quick,
        "games_per_level": games_per_level,
        "ref_time_limit": ref_time_limit,
        "eval_depth": eval_depth,
        "max_plies": max_plies,
        "profiles": profile_summaries,
        "games": game_records,
    }

    _write_outputs(report, json_path=json_path, csv_path=csv_path)
    if verbose:
        print_calibration_table(report)
    return report


def _write_outputs(report: dict[str, Any], json_path: Path, csv_path: Path) -> None:
    """Persist calibration summary to JSON and CSV inside DATA_DIR."""
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as jf:
        json.dump(report, jf, indent=2)

    fieldnames = [
        "profile",
        "target_elo",
        "estimated_elo",
        "ci_95_low",
        "ci_95_high",
        "elo_delta",
        "games_played",
        "total_points",
        "acpl",
        "blunder_rate_per_100",
        "total_blunders",
        "total_bot_moves",
        "book_moves",
        "engine_moves",
        "fallback_moves",
        "fallback_rate_pct",
    ]
    with open(csv_path, "w", encoding="utf-8", newline="") as cf:
        writer = csv.DictWriter(cf, fieldnames=fieldnames)
        writer.writeheader()
        for p_summary in report["profiles"].values():
            writer.writerow(
                {
                    "profile": p_summary["profile"],
                    "target_elo": p_summary["target_elo"],
                    "estimated_elo": p_summary["estimated_elo"],
                    "ci_95_low": p_summary["ci_95"][0],
                    "ci_95_high": p_summary["ci_95"][1],
                    "elo_delta": p_summary["elo_delta"],
                    "games_played": p_summary["games_played"],
                    "total_points": p_summary["total_points"],
                    "acpl": p_summary["acpl"],
                    "blunder_rate_per_100": p_summary["blunder_rate_per_100"],
                    "total_blunders": p_summary["total_blunders"],
                    "total_bot_moves": p_summary["total_bot_moves"],
                    "book_moves": p_summary["book_moves"],
                    "engine_moves": p_summary["engine_moves"],
                    "fallback_moves": p_summary["fallback_moves"],
                    "fallback_rate_pct": p_summary["fallback_rate_pct"],
                }
            )


def print_calibration_table(report: dict[str, Any]) -> None:
    """Print a formatted summary table of calibration results."""
    print("\n" + "=" * 96)
    print("                     CHESS KEN BOT STRENGTH CALIBRATION REPORT")
    print("=" * 96)
    header = (
        f"{'Profile':14s} | {'Label':>5s} | {'Est. Elo (95% CI)':>21s} | "
        f"{'Score':>9s} | {'ACPL':>6s} | {'Blunders/100':>12s} | {'Fallback%':>9s}"
    )
    print(header)
    print("-" * 96)
    for p_name, s in report["profiles"].items():
        ci_low, ci_high = s["ci_95"]
        elo_str = f"{s['estimated_elo']:6.1f} [{ci_low:.0f}..{ci_high:.0f}]"
        score_str = f"{s['total_points']:.1f}/{s['games_played']}"
        print(
            f"{p_name:14s} | {s['target_elo']:5d} | {elo_str:>21s} | "
            f"{score_str:>9s} | {s['acpl']:6.1f} | {s['blunder_rate_per_100']:12.2f} | "
            f"{s['fallback_rate_pct']:8.2f}%"
        )
        print(
            f"   Style  -> OOB SF#1: {s.get('engine_top1_rate_pct', 0.0):4.1f}% | "
            f"Erratic: {s.get('erratic_rate_pct', 0.0):4.2f}% ({s.get('erratic_moves', 0)}/{s.get('oob_moves', 0)})"
        )
        lvl_parts = [
            f"vs {lvl}: {b['points']:.1f}/{b['games']} ({b['score_pct']:.0f}%)"
            for lvl, b in s["scores_by_level"].items()
        ]
        print(f"   Levels -> {' | '.join(lvl_parts)}")
    print("=" * 96)
    print(f"Elapsed: {report['elapsed_seconds']}s | Saved JSON: {CALIBRATION_JSON_PATH.name} | CSV: {CALIBRATION_CSV_PATH.name}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline KenBot strength calibration harness.")
    parser.add_argument("--quick", action="store_true", help="Fast smoke-test mode (2 games/level).")
    parser.add_argument("--games-per-level", type=int, default=4, help="Games per reference Elo level.")
    parser.add_argument("--workers", type=int, default=4, help="Parallel game worker threads.")
    parser.add_argument("--max-plies", type=int, default=90, help="Maximum plies per calibration game.")
    parser.add_argument("--ref-time", type=float, default=0.05, help="Move time for reference Stockfish.")
    parser.add_argument("--eval-depth", type=int, default=7, help="Depth for ACPL reference evaluation.")
    parser.add_argument(
        "--profiles",
        nargs="*",
        default=None,
        help="Optional subset of profiles (e.g. 'Sick Ken' 'Focus Ken').",
    )
    parser.add_argument(
        "--levels",
        nargs="*",
        type=int,
        default=None,
        help="Optional list of reference Stockfish UCI_Elo levels.",
    )
    args = parser.parse_args()

    run_calibration(
        profile_names=args.profiles,
        reference_elos=args.levels,
        games_per_level=args.games_per_level,
        ref_time_limit=args.ref_time,
        eval_depth=args.eval_depth,
        max_plies=args.max_plies,
        workers=args.workers,
        quick=args.quick,
        verbose=True,
    )


if __name__ == "__main__":
    main()
