"""Explainability & Move Provenance Inspection CLI for Ken Bots.

Replays logged bot games from `data/move_logs/` or evaluates any FEN position
with a chosen KenBot profile, printing candidate scores, safety filter status,
style probabilities, top style features, continuity penalties, and final choice.

Usage:
    python -m calibration.explain_move --fen "<FEN>" --bot "Focus Ken"
    python -m calibration.explain_move --latest
    python -m calibration.explain_move --log data/move_logs/<file>.jsonl --ply 14
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import chess

from bots.ken_bot import KenBot
from config import BOOK_DB_PATH, PROVENANCE_LOG_DIR, STYLE_PROFILE_PATH


def format_decision(rec: dict[str, Any]) -> str:
    """Format a move provenance record into a concise human-readable report."""
    lines: list[str] = []
    lines.append("=" * 84)
    lines.append(
        f"Bot: {rec.get('bot')} ({rec.get('tier')}) | Side: {rec.get('side')} | "
        f"Ply: {rec.get('ply')} | Phase: {rec.get('phase')}"
    )
    lines.append(f"FEN: {rec.get('fen')}")
    lines.append(
        f"Chosen: {rec.get('chosen_san')} ({rec.get('chosen_uci')}) | "
        f"Source: {rec.get('source')} | CPL: {rec.get('cpl')} | "
        f"StyleProb: {rec.get('style_prob')} | SF#1: {rec.get('engine_top1_uci')} "
        f"(match={rec.get('equals_engine_top1')})"
    )
    if rec.get("continuity_reason"):
        lines.append(
            f"Continuity: {rec.get('continuity_reason')} ({rec.get('continuity_penalty'):+})"
        )
    if rec.get("tactical_override"):
        lines.append("Note: Tactical override active (engine #1 margin exceeded threshold).")
    if rec.get("error_type") and rec.get("error_type") != "none":
        lines.append(f"Error classification: {rec.get('error_type')}")

    book_cands = rec.get("book_candidates") or []
    if book_cands:
        lines.append("\nOpening Book Candidates:")
        lines.append(f"  {'Move':8s} {'UCI':6s} {'Count':>6s} {'Weight':>7s} {'WinRate':>8s} {'SampleP':>8s}")
        for b in book_cands:
            marker = "*" if b["uci"] == rec.get("chosen_uci") else " "
            lines.append(
                f"{marker} {b['san']:8s} {b['uci']:6s} {b['play_count']:6d} "
                f"{b['weighted_count']:7.1f} {b['win_rate']*100:7.1f}% {b['prob']*100:7.1f}%"
            )

    cands = rec.get("candidates") or []
    if cands:
        lines.append("\nEngine + Style Candidates:")
        lines.append(
            f"  {'Move':8s} {'Eval':>6s} {'CPL':>4s} {'Band':>6s} {'StyleP':>7s} "
            f"{'Cont':>6s} {'Logit':>7s}  Top Style Features"
        )
        for c in cands:
            marker = "*" if c["uci"] == rec.get("chosen_uci") else " "
            if c.get("in_safe_band"):
                band = "SAFE"
            elif c.get("in_error_band"):
                band = "ERR"
            else:
                band = "REJ"
            if c.get("below_style_floor"):
                band += "!"
            feats = ", ".join(
                f"{name}:{val:+.2f}" for name, val in (c.get("top_features") or [])
            )
            lines.append(
                f"{marker} {c['san']:8s} {c['score_cp']:+6d} {c['cpl']:4d} {band:>6s} "
                f"{c['style_prob']*100:6.1f}% {c['continuity_penalty']:+6.2f} "
                f"{c['style_logit']:+7.2f}  {feats}"
            )
    lines.append("=" * 84)
    return "\n".join(lines)


def explain_fen(
    fen: str,
    bot_name: str = "Focus Ken",
    db_path: Path = BOOK_DB_PATH,
    style_path: Path = STYLE_PROFILE_PATH,
    no_book: bool = False,
) -> dict[str, Any]:
    """Evaluate a FEN with the specified KenBot and return its full decision record."""
    board = chess.Board(fen)
    bot = KenBot(
        profile=bot_name,
        db_path=db_path,
        style_profile_path=style_path,
        log_provenance=False,
    )
    try:
        rec = bot.evaluate_position(board, deterministic=True, allow_book=not no_book)
        rec.pop("move", None)
        return rec
    finally:
        bot.close()


def replay_log(log_path: Path, target_ply: int | None = None) -> list[dict[str, Any]]:
    """Read provenance JSONL records from `log_path`, optionally filtered to a single ply."""
    records: list[dict[str, Any]] = []
    with open(log_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if target_ply is None or rec.get("ply") == target_ply:
                records.append(rec)
    return records


def find_latest_log(log_dir: Path = PROVENANCE_LOG_DIR) -> Path | None:
    """Return the most recently modified JSONL provenance log file, or None."""
    if not log_dir.exists():
        return None
    logs = sorted(log_dir.glob("game_*.jsonl"), key=lambda p: p.stat().st_mtime)
    return logs[-1] if logs else None


def main() -> None:
    """CLI entrypoint for explaining FEN positions or replaying move logs."""
    parser = argparse.ArgumentParser(description="Explain KenBot move choices and replay move logs.")
    parser.add_argument("--fen", type=str, default=None, help="FEN string to analyze.")
    parser.add_argument("--bot", type=str, default="Focus Ken", help="Bot profile name or tier.")
    parser.add_argument("--no-book", action="store_true", help="Skip opening book and show engine+style step.")
    parser.add_argument("--log", type=str, default=None, help="Path to a JSONL move provenance log.")
    parser.add_argument("--latest", action="store_true", help="Replay the latest logged game.")
    parser.add_argument("--ply", type=int, default=None, help="Optional ply filter when replaying a log.")
    args = parser.parse_args()

    if args.fen:
        rec = explain_fen(args.fen, bot_name=args.bot, no_book=args.no_book)
        print(format_decision(rec))
        return

    log_path: Path | None = Path(args.log) if args.log else (find_latest_log() if args.latest else None)
    if log_path is None:
        print("Provide --fen \"<FEN>\", --log <path>, or --latest.")
        return
    if not log_path.exists():
        print(f"Log file not found: {log_path}")
        return

    records = replay_log(log_path, target_ply=args.ply)
    if not records:
        print(f"No matching moves found in {log_path}.")
        return
    for rec in records:
        print(format_decision(rec))


if __name__ == "__main__":
    main()
