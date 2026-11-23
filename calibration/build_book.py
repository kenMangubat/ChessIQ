"""Offline idempotent PGN ingestion and SQLite opening/game database builder.

Parses Ken's historical games across online archives and OTB Lichess studies,
deduplicates games idempotently into SQLite (`ingested_games`), tracks source
weights and stratified White/Black hold-out splits, and materializes weighted
opening repertoire statistics into `book_moves`.
"""

from __future__ import annotations

import datetime
import hashlib
from pathlib import Path
import re
import shutil
import sqlite3
import time
from typing import Any

import chess
import chess.pgn

from config import BOOK_DB_PATH, DATA_DIR, PGN_FILES, STYLE_PROFILE_PATH

# ── Ken identification & source-weight constants ─────────────────────
KEN_USERNAMES: frozenset[str] = frozenset(
    {
        "em_ken",
        "ken_is_not_here",
        "ken_uclm1",
        "thissskenn",
        "uclm_ken2",
        "ken_uclm",
        "ken_human",
    }
)

# Real-name / OTB aliases confirmed from tournament PGN studies annotated by UCLM_Ken2
KEN_OTB_NAMES: frozenset[str] = frozenset(
    {
        "mangubat",
        "ken mangubat",
        "ken harvey mangubat",
        "k. mangubat",
    }
)

# Bot names that must NEVER be mistaken for Ken's own human moves
BOT_NAME_EXCLUSIONS: frozenset[str] = frozenset(
    {
        "sick ken",
        "focus ken",
        "condition ken",
    }
)

# Explicit source weights so OTB and recent games carry stronger style weight
SOURCE_WEIGHT_OTB: float = 3.0
SOURCE_WEIGHT_RECENT_RAPID: float = 2.0
SOURCE_WEIGHT_RECENT_BLITZ: float = 1.5
SOURCE_WEIGHT_OLDER_ONLINE: float = 1.0
SOURCE_WEIGHT_BOT_GAME: float = 0.5

MAX_BOOK_PLIES = 24  # 12 full moves per player
MIN_TRAINING_PLIES = 6
HOLDOUT_FRACTION = 0.20

TIER_SICK = "sick"            # Elo < 1400
TIER_FOCUS = "focus"          # 1400 <= Elo < 1750
TIER_CONDITION = "condition"  # Elo >= 1750

TIERS: tuple[str, ...] = (TIER_SICK, TIER_FOCUS, TIER_CONDITION)

TIER_FALLBACK_ORDER: dict[str, list[str]] = {
    TIER_SICK: [TIER_SICK, TIER_FOCUS, TIER_CONDITION],
    TIER_FOCUS: [TIER_FOCUS, TIER_CONDITION, TIER_SICK],
    TIER_CONDITION: [TIER_CONDITION, TIER_FOCUS, TIER_SICK],
}

_VS_PATTERN = re.compile(
    r"(?:\(|:\s*|^)\s*([A-Za-zÀ-ÿ0-9_\-\s\.]+?)\s+vs\.?\s+([A-Za-zÀ-ÿ0-9_\-\s\.]+?)\s*(?:\)|$|\(Round)",
    re.IGNORECASE,
)

_OTB_EVENT_MARKERS: tuple[str, ...] = (
    "tournament",
    "cesafi",
    "bcc",
    "social hall",
    "mahi",
    "otb",
    "over the board",
    "practice",
)


def determine_tier(elo: int) -> str:
    """Map a numerical rating to a bot personality tier."""
    if elo < 1400:
        return TIER_SICK
    if elo < 1750:
        return TIER_FOCUS
    return TIER_CONDITION


def is_ken_player_name(name: str) -> bool:
    """Return True if *name* confidently identifies Ken (excluding bot accounts)."""
    cleaned = name.strip().lower()
    if not cleaned or cleaned == "?":
        return False
    if "(bot)" in cleaned or cleaned in BOT_NAME_EXCLUSIONS:
        return False
    if cleaned in KEN_USERNAMES or cleaned in KEN_OTB_NAMES:
        return True
    # Token-level check for OTB surnames like "Mangubat, Ken" or "Round 6 Mangubat"
    tokens = re.findall(r"[a-z0-9_]+", cleaned)
    if "mangubat" in tokens:
        return True
    return False


def _extract_study_matchup(text: str) -> tuple[str, str] | None:
    """Parse 'PlayerA vs PlayerB' from a Lichess study ChapterName or Event header."""
    if not text or " vs " not in text.lower():
        return None
    # Strip leading tournament prefix before ':' if present
    segment = text.split(":", 1)[-1].strip()
    # Prefer parenthesized matchup if present, e.g. "Round 6 (Estrera vs Mangubat)"
    paren_match = re.search(r"\(([^()]*\bvs\.?\b[^()]*)\)", segment, re.IGNORECASE)
    if paren_match:
        segment = paren_match.group(1).strip()
    parts = re.split(r"\s+vs\.?\s+", segment, maxsplit=1, flags=re.IGNORECASE)
    if len(parts) != 2:
        return None
    left = re.sub(r"\(.*?\)", "", parts[0]).strip()
    right = re.sub(r"\(.*?\)", "", parts[1]).strip()
    if left and right:
        return (left, right)
    return None


def identify_ken_in_game(
    headers: chess.pgn.Headers | dict[str, str],
    source_file: str = "",
) -> tuple[str | None, bool, str, str]:
    """Identify Ken's side ('white'/'black'/None), whether the game is OTB, and resolved player names.

    Handles both standard online PGNs (White/Black headers) and Lichess Study OTB PGNs
    where White/Black are '?' and player identity is recorded in ChapterName/Event +
    Annotator + Orientation.
    """
    w_raw = str(headers.get("White", "")).strip()
    b_raw = str(headers.get("Black", "")).strip()
    event = str(headers.get("Event", "")).strip()
    site = str(headers.get("Site", "")).strip()
    study_name = str(headers.get("StudyName", "")).strip()
    chapter_name = str(headers.get("ChapterName", "")).strip()
    annotator = str(headers.get("Annotator", "")).strip().lower()
    orientation = str(headers.get("Orientation", "")).strip().lower()

    ken_is_white = is_ken_player_name(w_raw)
    ken_is_black = is_ken_player_name(b_raw)

    # Skip self-play / ambiguous games where both players match Ken
    if ken_is_white and ken_is_black:
        return (None, False, w_raw, b_raw)

    is_study = bool(study_name or chapter_name or "lichess_study" in source_file.lower())
    event_lower = event.lower()
    site_lower = site.lower()
    is_online_platform = (
        "chess.com" in site_lower
        or ("lichess.org" in site_lower and not is_study)
    )
    is_otb = bool(
        is_study
        or (
            not is_online_platform
            and (
                any(marker in event_lower for marker in _OTB_EVENT_MARKERS)
                or (site_lower not in ("", "?") and "play vs bot" not in event_lower)
            )
        )
    )

    if ken_is_white:
        return ("white", is_otb, w_raw, b_raw)
    if ken_is_black:
        return ("black", is_otb, w_raw, b_raw)

    # Fallback for Lichess study exports where White/Black headers are '?'
    if (w_raw in ("", "?") and b_raw in ("", "?")) and (is_study or annotator):
        annotator_is_ken = any(u in annotator for u in KEN_USERNAMES) or any(
            u in source_file.lower() for u in KEN_USERNAMES
        )
        if not annotator_is_ken:
            return (None, is_otb, w_raw, b_raw)

        matchup = _extract_study_matchup(chapter_name) or _extract_study_matchup(event)
        if matchup is not None:
            left_name, right_name = matchup
            left_is_ken = is_ken_player_name(left_name)
            right_is_ken = is_ken_player_name(right_name)
            if left_is_ken and not right_is_ken:
                # Cross-check with orientation when provided
                if orientation in ("", "white"):
                    return ("white", True, left_name, right_name)
            elif right_is_ken and not left_is_ken:
                if orientation in ("", "black"):
                    return ("black", True, left_name, right_name)

    return (None, is_otb, w_raw, b_raw)


def _parse_base_seconds(time_control: str) -> int:
    """Parse initial clock seconds from PGN TimeControl header (e.g. '600+5', '300')."""
    tc = time_control.strip()
    if not tc or tc in ("-", "?"):
        return 0
    base_part = tc.split("+", 1)[0].split("|", 1)[0]
    try:
        return int(base_part)
    except ValueError:
        return 0


def _extract_year(date_str: str) -> int:
    """Extract 4-digit year from PGN Date string ('YYYY.MM.DD'), or 0 if invalid."""
    if not date_str or len(date_str) < 4:
        return 0
    try:
        year = int(date_str[:4])
        return year if 2000 <= year <= 2100 else 0
    except ValueError:
        return 0


def compute_source_weight(
    is_otb: bool,
    game_date: str,
    time_control: str,
    event: str,
) -> float:
    """Assign an explicit training weight reflecting OTB and recency priority."""
    if "play vs bot" in event.lower():
        return SOURCE_WEIGHT_BOT_GAME
    if is_otb:
        return SOURCE_WEIGHT_OTB
    year = _extract_year(game_date)
    base_sec = _parse_base_seconds(time_control)
    is_rapid_or_slower = base_sec >= 600 or "rapid" in event.lower() or "classical" in event.lower()
    if year >= 2026:
        return SOURCE_WEIGHT_RECENT_RAPID if is_rapid_or_slower else SOURCE_WEIGHT_RECENT_BLITZ
    return SOURCE_WEIGHT_OLDER_ONLINE


def compute_game_dedupe_key(
    site: str,
    game_date: str,
    white: str,
    black: str,
    result: str,
    moves_uci: list[str],
) -> str:
    """Compute a stable SHA-1 dedupe key from game metadata and mainline move list."""
    moves_hash = hashlib.sha1(" ".join(moves_uci).encode("utf-8")).hexdigest()[:16]
    payload = "|".join(
        [
            site.strip().lower(),
            game_date.strip(),
            white.strip().lower(),
            black.strip().lower(),
            result.strip(),
            moves_hash,
        ]
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def backup_training_artifacts(
    db_path: Path = BOOK_DB_PATH,
    style_path: Path = STYLE_PROFILE_PATH,
) -> list[Path]:
    """Create timestamped backups of ken_book.db and ken_style_profile.json in DATA_DIR."""
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    created: list[Path] = []
    for target in (db_path, style_path):
        if target.exists():
            backup_path = target.with_name(f"{target.name}.bak_{ts}")
            shutil.copy2(target, backup_path)
            created.append(backup_path)
    return created


def init_book_db(db_path: Path) -> sqlite3.Connection:
    """Initialize SQLite book & game database with non-destructive schema migration."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    with conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS book_moves (
                epd TEXT NOT NULL,
                side TEXT NOT NULL,
                move_uci TEXT NOT NULL,
                tier TEXT NOT NULL,
                play_count INTEGER NOT NULL DEFAULT 0,
                wins INTEGER NOT NULL DEFAULT 0,
                draws INTEGER NOT NULL DEFAULT 0,
                losses INTEGER NOT NULL DEFAULT 0,
                weighted_count REAL NOT NULL DEFAULT 0.0,
                PRIMARY KEY (epd, side, move_uci, tier)
            )
            """
        )
        # Non-destructive migration: add weighted_count if DB came from v1 schema
        cols = {
            row[1]
            for row in conn.execute("PRAGMA table_info(book_moves)").fetchall()
        }
        if "weighted_count" not in cols:
            conn.execute(
                "ALTER TABLE book_moves ADD COLUMN weighted_count REAL NOT NULL DEFAULT 0.0"
            )
            conn.execute(
                "UPDATE book_moves SET weighted_count = CAST(play_count AS REAL) WHERE weighted_count = 0.0"
            )

        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_book_lookup
            ON book_moves (epd, side, tier)
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_book_fallback
            ON book_moves (epd, side)
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS book_metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS ingested_games (
                dedupe_key TEXT PRIMARY KEY,
                source_file TEXT NOT NULL,
                game_index INTEGER NOT NULL,
                ken_side TEXT NOT NULL,
                white_name TEXT NOT NULL DEFAULT '',
                black_name TEXT NOT NULL DEFAULT '',
                ken_elo INTEGER NOT NULL,
                tier TEXT NOT NULL,
                result TEXT NOT NULL,
                eco TEXT NOT NULL DEFAULT '',
                event TEXT NOT NULL DEFAULT '',
                site TEXT NOT NULL DEFAULT '',
                game_date TEXT NOT NULL DEFAULT '',
                time_control TEXT NOT NULL DEFAULT '',
                is_otb INTEGER NOT NULL DEFAULT 0,
                source_weight REAL NOT NULL DEFAULT 1.0,
                ply_count INTEGER NOT NULL DEFAULT 0,
                moves_uci TEXT NOT NULL,
                split TEXT NOT NULL DEFAULT 'train',
                analyzed_at TEXT,
                ingested_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_ingested_side_split
            ON ingested_games (ken_side, split)
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_ingested_analyzed
            ON ingested_games (analyzed_at)
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS game_move_analysis (
                dedupe_key TEXT NOT NULL,
                ply INTEGER NOT NULL,
                phase TEXT NOT NULL,
                ken_side TEXT NOT NULL,
                tier TEXT NOT NULL,
                ken_move_uci TEXT NOT NULL,
                best_move_uci TEXT NOT NULL,
                cpl INTEGER NOT NULL,
                is_top1 INTEGER NOT NULL,
                error_type TEXT NOT NULL DEFAULT 'none',
                PRIMARY KEY (dedupe_key, ply)
            )
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_analysis_phase_tier
            ON game_move_analysis (phase, tier, ken_side)
            """
        )
    return conn


def assign_stratified_holdout_splits(
    conn: sqlite3.Connection,
    holdout_fraction: float = HOLDOUT_FRACTION,
) -> dict[str, int]:
    """Assign deterministic 80/20 train/holdout splits stratified by (ken_side, is_otb).

    Selects the most recent games (by game_date descending) in each stratum as holdout
    so evaluation tests generalization to Ken's current play.
    """
    rows = conn.execute(
        """
        SELECT dedupe_key, ken_side, is_otb, game_date, ply_count
        FROM ingested_games
        WHERE ply_count >= ?
        ORDER BY game_date DESC, dedupe_key DESC
        """,
        (MIN_TRAINING_PLIES,),
    ).fetchall()

    strata: dict[tuple[str, int], list[str]] = {}
    for dkey, side, is_otb, _gdate, _ply in rows:
        strata.setdefault((str(side), int(is_otb)), []).append(str(dkey))

    holdout_keys: set[str] = set()
    for (_side, _otb), keys in strata.items():
        if len(keys) < 2:
            continue
        n_holdout = max(1, int(round(len(keys) * holdout_fraction)))
        if n_holdout >= len(keys):
            n_holdout = len(keys) - 1
        holdout_keys.update(keys[:n_holdout])

    with conn:
        conn.execute("UPDATE ingested_games SET split = 'train'")
        if holdout_keys:
            conn.executemany(
                "UPDATE ingested_games SET split = 'holdout' WHERE dedupe_key = ?",
                [(k,) for k in holdout_keys],
            )

    train_count = conn.execute(
        "SELECT COUNT(*) FROM ingested_games WHERE split = 'train'"
    ).fetchone()[0]
    holdout_count = conn.execute(
        "SELECT COUNT(*) FROM ingested_games WHERE split = 'holdout'"
    ).fetchone()[0]
    return {"train": int(train_count), "holdout": int(holdout_count)}


def ingest_pgn_files(
    pgn_files: list[Path] | None = None,
    db_path: Path = BOOK_DB_PATH,
    recompute_splits: bool = True,
    verbose: bool = True,
) -> dict[str, Any]:
    """Idempotently ingest PGN archives into SQLite `ingested_games`.

    Re-running on the same PGN files adds 0 new games. New files add only unseen games.
    """
    t0 = time.time()
    files = list(pgn_files) if pgn_files is not None else sorted(DATA_DIR.glob("*.pgn"))

    conn = init_book_db(db_path)
    existing_keys: set[str] = {
        row[0] for row in conn.execute("SELECT dedupe_key FROM ingested_games").fetchall()
    }

    total_scanned = 0
    new_added = 0
    already_present = 0
    skipped_variants = 0
    skipped_non_ken = 0
    skipped_zero_moves = 0
    now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
    to_insert: list[tuple[Any, ...]] = []

    for pgn_file in files:
        if not pgn_file.exists():
            if verbose:
                print(f"[WARN] Skipping missing PGN file: {pgn_file}")
            continue
        if verbose:
            print(f"Scanning: {pgn_file.name} ...")

        game_idx = 0
        with open(pgn_file, mode="r", encoding="utf-8", errors="replace") as f:
            while True:
                try:
                    game = chess.pgn.read_game(f)
                except Exception:
                    continue
                if game is None:
                    break
                game_idx += 1
                total_scanned += 1
                headers = game.headers

                # Filter out non-standard variants and custom starting positions
                variant = headers.get("Variant", "").strip().lower()
                if variant and variant != "standard":
                    skipped_variants += 1
                    continue
                if headers.get("SetUp", "").strip() == "1":
                    skipped_variants += 1
                    continue
                fen_header = headers.get("FEN", "").strip()
                if fen_header and fen_header != chess.STARTING_FEN:
                    skipped_variants += 1
                    continue

                ken_side, is_otb, w_name, b_name = identify_ken_in_game(
                    headers, source_file=pgn_file.name
                )
                if ken_side is None:
                    skipped_non_ken += 1
                    continue

                # Extract mainline moves only (ignores study annotations & sidelines)
                board = game.board()
                moves_uci: list[str] = []
                for mv in game.mainline_moves():
                    if mv not in board.legal_moves:
                        break
                    moves_uci.append(mv.uci())
                    board.push(mv)

                if not moves_uci:
                    skipped_zero_moves += 1
                    continue

                event = headers.get("Event", "").strip()
                site = headers.get("Site", "").strip()
                if site in ("", "?"):
                    site = headers.get("ChapterURL", "").strip() or pgn_file.name
                game_date = headers.get("Date", "").strip()
                if game_date in ("", "????.??.??") or game_date.startswith("1970."):
                    game_date = headers.get("UTCDate", "").strip() or game_date
                result = headers.get("Result", "").strip() or "*"
                if result == "*" and board.is_checkmate():
                    result = "1-0" if board.turn == chess.BLACK else "0-1"

                dedupe_key = compute_game_dedupe_key(
                    site=site,
                    game_date=game_date,
                    white=w_name,
                    black=b_name,
                    result=result,
                    moves_uci=moves_uci,
                )
                if dedupe_key in existing_keys:
                    already_present += 1
                    continue
                existing_keys.add(dedupe_key)

                elo_str = headers.get(
                    "WhiteElo" if ken_side == "white" else "BlackElo", ""
                ).strip()
                try:
                    ken_elo = int(elo_str)
                except ValueError:
                    ken_elo = 1850 if is_otb else 1500
                tier = determine_tier(ken_elo)
                eco = headers.get("ECO", "").strip()
                time_control = headers.get("TimeControl", "").strip()
                weight = compute_source_weight(is_otb, game_date, time_control, event)

                to_insert.append(
                    (
                        dedupe_key,
                        pgn_file.name,
                        game_idx,
                        ken_side,
                        w_name,
                        b_name,
                        ken_elo,
                        tier,
                        result,
                        eco,
                        event,
                        site,
                        game_date,
                        time_control,
                        1 if is_otb else 0,
                        weight,
                        len(moves_uci),
                        " ".join(moves_uci),
                        "train",
                        None,
                        now_iso,
                    )
                )
                new_added += 1

    if to_insert:
        with conn:
            conn.executemany(
                """
                INSERT OR IGNORE INTO ingested_games (
                    dedupe_key, source_file, game_index, ken_side, white_name, black_name,
                    ken_elo, tier, result, eco, event, site, game_date, time_control,
                    is_otb, source_weight, ply_count, moves_uci, split, analyzed_at, ingested_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                to_insert,
            )

    split_counts = (
        assign_stratified_holdout_splits(conn)
        if (recompute_splits and (new_added > 0 or conn.execute("SELECT COUNT(*) FROM ingested_games WHERE split='holdout'").fetchone()[0] == 0))
        else {
            "train": int(
                conn.execute("SELECT COUNT(*) FROM ingested_games WHERE split='train'").fetchone()[0]
            ),
            "holdout": int(
                conn.execute("SELECT COUNT(*) FROM ingested_games WHERE split='holdout'").fetchone()[0]
            ),
        }
    )

    total_db_games = int(conn.execute("SELECT COUNT(*) FROM ingested_games").fetchone()[0])
    white_games = int(conn.execute("SELECT COUNT(*) FROM ingested_games WHERE ken_side='white'").fetchone()[0])
    black_games = int(conn.execute("SELECT COUNT(*) FROM ingested_games WHERE ken_side='black'").fetchone()[0])
    otb_games = int(conn.execute("SELECT COUNT(*) FROM ingested_games WHERE is_otb=1").fetchone()[0])
    tier_rows = conn.execute("SELECT tier, COUNT(*) FROM ingested_games GROUP BY tier").fetchall()
    tier_counts = {t: 0 for t in TIERS}
    for t_name, cnt in tier_rows:
        if t_name in tier_counts:
            tier_counts[t_name] = int(cnt)

    conn.close()
    elapsed = round(time.time() - t0, 2)

    summary = {
        "elapsed_seconds": elapsed,
        "total_games_scanned": total_scanned,
        "new_games_added": new_added,
        "already_present": already_present,
        "skipped_variants": skipped_variants,
        "skipped_non_ken": skipped_non_ken,
        "skipped_zero_moves": skipped_zero_moves,
        "total_db_games": total_db_games,
        "white_games": white_games,
        "black_games": black_games,
        "otb_games": otb_games,
        "online_games": total_db_games - otb_games,
        "tier_counts": tier_counts,
        "splits": split_counts,
    }
    if verbose:
        print(
            f"Ingest done in {elapsed:.2f}s | scanned={total_scanned}, new={new_added}, "
            f"existing={already_present}, total_in_db={total_db_games} "
            f"(White={white_games}, Black={black_games}, OTB={otb_games}, "
            f"Train={split_counts['train']}, Holdout={split_counts['holdout']})"
        )
    return summary


def rebuild_book_from_db(
    db_path: Path = BOOK_DB_PATH,
    split_filter: str | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    """Rebuild weighted `book_moves` from `ingested_games` (preserving all prior ingested games)."""
    t0 = time.time()
    conn = init_book_db(db_path)

    query = """
        SELECT ken_side, tier, result, source_weight, moves_uci
        FROM ingested_games
    """
    params: tuple[Any, ...] = ()
    if split_filter is not None:
        query += " WHERE split = ?"
        params = (split_filter,)

    rows = conn.execute(query, params).fetchall()

    # key: (epd, side, move_uci, tier) -> [play_count, wins, draws, losses, weighted_count]
    stats: dict[tuple[str, str, str, str], list[float]] = {}
    tier_counts: dict[str, int] = {t: 0 for t in TIERS}

    for side, tier, result, weight, moves_str in rows:
        if tier in tier_counts:
            tier_counts[tier] += 1
        ken_is_white = side == "white"
        if ken_is_white:
            win_val = 1 if result == "1-0" else 0
            draw_val = 1 if result == "1/2-1/2" else 0
            loss_val = 1 if result == "0-1" else 0
        else:
            win_val = 1 if result == "0-1" else 0
            draw_val = 1 if result == "1/2-1/2" else 0
            loss_val = 1 if result == "1-0" else 0

        board = chess.Board()
        seen_in_game: set[tuple[str, str, str, str]] = set()
        moves = moves_str.split() if moves_str else []

        for ply_idx, uci in enumerate(moves):
            if ply_idx >= MAX_BOOK_PLIES:
                break
            try:
                move = chess.Move.from_uci(uci)
            except ValueError:
                break
            if move not in board.legal_moves:
                break

            is_kens_turn = (board.turn == chess.WHITE and ken_is_white) or (
                board.turn == chess.BLACK and not ken_is_white
            )
            if is_kens_turn:
                epd = board.epd()
                key = (epd, side, uci, tier)
                if key not in seen_in_game:
                    seen_in_game.add(key)
                    entry = stats.setdefault(key, [0, 0, 0, 0, 0.0])
                    entry[0] += 1
                    entry[1] += win_val
                    entry[2] += draw_val
                    entry[3] += loss_val
                    entry[4] += float(weight)
            board.push(move)

    with conn:
        conn.execute("DELETE FROM book_moves")
        conn.executemany(
            """
            INSERT INTO book_moves (
                epd, side, move_uci, tier, play_count, wins, draws, losses, weighted_count
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    epd,
                    side,
                    move_uci,
                    tier,
                    int(counts[0]),
                    int(counts[1]),
                    int(counts[2]),
                    int(counts[3]),
                    round(float(counts[4]), 3),
                )
                for (epd, side, move_uci, tier), counts in stats.items()
            ],
        )

        cursor = conn.cursor()
        unique_positions = int(
            cursor.execute("SELECT COUNT(DISTINCT epd) FROM book_moves").fetchone()[0]
        )
        total_move_records = int(
            cursor.execute("SELECT COUNT(*) FROM book_moves").fetchone()[0]
        )

        now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
        metadata = [
            ("valid_games_parsed", str(len(rows))),
            ("unique_positions", str(unique_positions)),
            ("total_move_records", str(total_move_records)),
            ("book_split_filter", split_filter or "all"),
            ("built_at", now_iso),
        ]
        conn.executemany(
            "INSERT OR REPLACE INTO book_metadata (key, value) VALUES (?, ?)",
            metadata,
        )

    conn.close()
    elapsed = round(time.time() - t0, 2)
    summary = {
        "elapsed_seconds": elapsed,
        "valid_games_parsed": len(rows),
        "tier_counts": tier_counts,
        "unique_positions": unique_positions,
        "total_move_records": total_move_records,
        "db_path": str(db_path),
    }
    if verbose:
        print_summary(summary, db_path)
    return summary


def build_opening_book(
    pgn_files: list[Path] | None = None,
    db_path: Path = BOOK_DB_PATH,
    split_filter: str | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    """Ingest PGN files idempotently and rebuild weighted opening book in SQLite."""
    ingest_stats = ingest_pgn_files(pgn_files=pgn_files, db_path=db_path, verbose=verbose)
    book_stats = rebuild_book_from_db(db_path=db_path, split_filter=split_filter, verbose=verbose)
    return {
        **ingest_stats,
        **book_stats,
        "elapsed_seconds": round(
            ingest_stats["elapsed_seconds"] + book_stats["elapsed_seconds"], 2
        ),
    }


def query_book_moves(
    epd: str,
    side: str,
    tier: str = TIER_FOCUS,
    db_path: Path = BOOK_DB_PATH,
    allow_fallback: bool = True,
    min_play_count: int = 1,
    min_win_rate: float = 0.0,
    fallback_tiers: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Query weighted candidate book moves for a given position EPD, side, and tier."""
    if not db_path.exists():
        return []

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    side_normalized = side.strip().lower()
    tier_normalized = tier.strip().lower()

    cols = {
        row[1] for row in cursor.execute("PRAGMA table_info(book_moves)").fetchall()
    }
    has_weighted = "weighted_count" in cols
    w_expr = "CASE WHEN weighted_count > 0 THEN weighted_count ELSE CAST(play_count AS REAL) END" if has_weighted else "CAST(play_count AS REAL)"

    def _filter_and_format(rows: list[Any], is_fb: bool) -> list[dict[str, Any]]:
        results = []
        for r in rows:
            p_count = int(r["play_count"])
            if p_count < min_play_count:
                continue
            wins = int(r["wins"])
            draws = int(r["draws"])
            losses = int(r["losses"])
            decided = wins + draws + losses
            w_rate = round(wins / decided, 3) if decided > 0 else 0.5
            if decided > 0 and w_rate < min_win_rate:
                continue
            score = round((wins + 0.5 * draws) / decided, 3) if decided > 0 else 0.5
            w_count = round(float(r["weighted_count"]), 3)
            results.append(
                {
                    "move_uci": r["move_uci"],
                    "tier": r["tier"],
                    "play_count": p_count,
                    "weighted_count": w_count,
                    "wins": wins,
                    "draws": draws,
                    "losses": losses,
                    "win_rate": w_rate,
                    "score": score,
                    "is_fallback": is_fb,
                }
            )
        return results

    try:
        if tier_normalized == "all":
            cursor.execute(
                f"""
                SELECT move_uci,
                       'all' AS tier,
                       SUM(play_count) AS play_count,
                       SUM({w_expr}) AS weighted_count,
                       SUM(wins) AS wins,
                       SUM(draws) AS draws,
                       SUM(losses) AS losses
                FROM book_moves
                WHERE epd = ? AND side = ?
                GROUP BY move_uci
                ORDER BY SUM({w_expr}) DESC, SUM(play_count) DESC, SUM(wins) DESC
                """,
                (epd, side_normalized),
            )
            return _filter_and_format(cursor.fetchall(), is_fb=False)

        cursor.execute(
            f"""
            SELECT move_uci, tier, play_count, ({w_expr}) AS weighted_count, wins, draws, losses
            FROM book_moves
            WHERE epd = ? AND side = ? AND tier = ?
            ORDER BY ({w_expr}) DESC, play_count DESC, wins DESC
            """,
            (epd, side_normalized, tier_normalized),
        )
        exact_results = _filter_and_format(cursor.fetchall(), is_fb=False)
        if exact_results:
            return exact_results

        if allow_fallback:
            candidate_fb_tiers = (
                fallback_tiers
                if fallback_tiers is not None
                else TIER_FALLBACK_ORDER.get(
                    tier_normalized, [t for t in TIERS if t != tier_normalized]
                )
            )
            for fb_tier in candidate_fb_tiers:
                if fb_tier == tier_normalized:
                    continue
                cursor.execute(
                    f"""
                    SELECT move_uci, tier, play_count, ({w_expr}) AS weighted_count, wins, draws, losses
                    FROM book_moves
                    WHERE epd = ? AND side = ? AND tier = ?
                    ORDER BY ({w_expr}) DESC, play_count DESC, wins DESC
                    """,
                    (epd, side_normalized, fb_tier),
                )
                fb_results = _filter_and_format(cursor.fetchall(), is_fb=True)
                if fb_results:
                    return fb_results

        return []
    finally:
        conn.close()


def export_human_game_to_pgn(
    moves_uci: list[str],
    human_color: chess.Color,
    bot_name: str,
    result: str = "*",
    out_path: Path | None = None,
    event: str = "Chess Ken Human vs Bot",
) -> Path:
    """Opt-in helper to export a human-vs-bot game PGN with ONLY the human flagged as Ken.

    The bot side is tagged with '(Bot)' so ingestion will strictly exclude bot moves
    and learn only from the human player's moves.
    """
    target = out_path or (DATA_DIR / "human_vs_bot_games.pgn")
    target.parent.mkdir(parents=True, exist_ok=True)
    game = chess.pgn.Game()
    game.headers["Event"] = event
    game.headers["Site"] = "ChessKen Local"
    game.headers["Date"] = datetime.date.today().strftime("%Y.%m.%d")
    game.headers["Result"] = result
    if human_color == chess.WHITE:
        game.headers["White"] = "Ken_Human"
        game.headers["Black"] = f"{bot_name} (Bot)"
    else:
        game.headers["White"] = f"{bot_name} (Bot)"
        game.headers["Black"] = "Ken_Human"

    board = chess.Board()
    node: chess.pgn.GameNode = game
    for uci in moves_uci:
        mv = chess.Move.from_uci(uci)
        if mv not in board.legal_moves:
            break
        node = node.add_variation(mv)
        board.push(mv)

    exporter = chess.pgn.StringExporter(headers=True, variations=False, comments=False)
    pgn_text = game.accept(exporter)
    with open(target, "a", encoding="utf-8") as f:
        f.write(pgn_text + "\n\n")
    return target


def print_summary(summary: dict[str, Any], db_path: Path) -> None:
    """Print readable summary stats of the built opening book."""
    print("=" * 68)
    print("           CHESS KEN OPENING & STYLE BOOK SUMMARY")
    print("=" * 68)
    print(f"Database File:         {db_path}")
    print(f"Build Time:            {summary['elapsed_seconds']}s")
    if "total_games_scanned" in summary:
        print(f"Games Scanned:         {summary['total_games_scanned']}")
    print(f"Valid Games in Book:   {summary['valid_games_parsed']}")
    print(f"Unique Positions:      {summary['unique_positions']}")
    print(f"Total Move Records:    {summary['total_move_records']}")
    print("-" * 68)
    tc = summary["tier_counts"]
    print(f"Tier Breakdown:        Sick: {tc['sick']} | Focus: {tc['focus']} | Condition: {tc['condition']}")
    print("-" * 68)

    start_epd = chess.Board().epd()
    print("\n--- TOP WHITE OPENING MOVES (from starting position) ---")
    for tier in TIERS:
        moves = query_book_moves(start_epd, "white", tier=tier, db_path=db_path, allow_fallback=False)
        formatted = ", ".join(
            f"{m['move_uci']} ({m['play_count']}x, w={m['weighted_count']:.0f}, {int(m['win_rate']*100)}% W)"
            for m in moves[:4]
        ) or "None"
        print(f"  [{tier.upper():9s}] {formatted}")

    after_e4 = chess.Board()
    after_e4.push_san("e4")
    after_e4_epd = after_e4.epd()
    print("\n--- TOP BLACK RESPONSES TO 1. e4 ---")
    for tier in TIERS:
        moves = query_book_moves(after_e4_epd, "black", tier=tier, db_path=db_path, allow_fallback=False)
        formatted = ", ".join(
            f"{m['move_uci']} ({m['play_count']}x, w={m['weighted_count']:.0f}, {int(m['win_rate']*100)}% W)"
            for m in moves[:4]
        ) or "None"
        print(f"  [{tier.upper():9s}] {formatted}")

    print("=" * 68)


def run_continuous_training(
    db_path: Path = BOOK_DB_PATH,
    style_path: Path = STYLE_PROFILE_PATH,
    analyze_limit: int = 120,
    eval_games: int = 25,
    workers: int = 4,
    skip_backup: bool = False,
    verbose: bool = True,
) -> dict[str, Any]:
    """One-command safe continuous training pipeline.

    1. Backs up previous DB & style profile in DATA_DIR
    2. Runs idempotent PGN ingest + weighted opening book rebuild
    3. Runs incremental offline Stockfish mistake analysis (only unanalyzed games)
    4. Trains White/Black move-level style models & updates ken_style_profile.json
    5. Runs hold-out match rate & engine-likeness quality checks and prints a report
    """
    from calibration.style_analyzer import (
        analyze_ken_mistakes_incremental,
        evaluate_holdout_style,
        save_style_profile,
        train_ken_style_profile,
    )

    backups: list[Path] = []
    if not skip_backup:
        backups = backup_training_artifacts(db_path=db_path, style_path=style_path)
        if verbose and backups:
            print("Backed up previous artifacts:", ", ".join(b.name for b in backups))

    pgn_files = sorted(DATA_DIR.glob("*.pgn"))
    book_summary = build_opening_book(pgn_files=pgn_files, db_path=db_path, verbose=verbose)

    if verbose:
        print("\nRunning incremental offline Stockfish analysis on unanalyzed games...")
    mistake_summary = analyze_ken_mistakes_incremental(
        db_path=db_path,
        max_new_games=analyze_limit,
        workers=workers,
        verbose=verbose,
    )

    if verbose:
        print("\nTraining White/Black move-level style models...")
    profile_data = train_ken_style_profile(
        db_path=db_path,
        split_filter=None,
        verbose=verbose,
    )
    save_style_profile(profile_data, path=style_path)

    if verbose:
        print("\nRunning hold-out quality checks (trained strictly on train split)...")
    holdout_report = evaluate_holdout_style(
        db_path=db_path,
        max_holdout_games=eval_games,
        verbose=verbose,
    )

    return {
        "backups": [str(b) for b in backups],
        "book": book_summary,
        "mistakes": mistake_summary,
        "profile_version": profile_data.get("version"),
        "holdout": holdout_report,
    }


def main() -> None:
    """CLI entrypoint for building book, style training, or full continuous training."""
    import argparse

    parser = argparse.ArgumentParser(description="Ken Opening Book & Continuous Style Trainer")
    parser.add_argument("--style", action="store_true", help="Train style profile after ingest")
    parser.add_argument("--all", action="store_true", help="Build book and train style profile")
    parser.add_argument(
        "--train",
        action="store_true",
        help="Run full continuous training (backup, idempotent ingest, incremental analysis, style train, quality checks)",
    )
    parser.add_argument(
        "--analyze-limit",
        type=int,
        default=120,
        help="Max new games to analyze with Stockfish in incremental pass",
    )
    parser.add_argument(
        "--eval-games",
        type=int,
        default=25,
        help="Number of hold-out games for quality check",
    )
    parser.add_argument("--workers", type=int, default=4, help="Parallel worker threads for engine analysis")
    args = parser.parse_args()

    if args.train:
        run_continuous_training(
            analyze_limit=args.analyze_limit,
            eval_games=args.eval_games,
            workers=args.workers,
            verbose=True,
        )
        return

    print("Starting Ken's idempotent PGN ingest and opening book builder...")
    pgn_files = sorted(DATA_DIR.glob("*.pgn"))
    build_opening_book(pgn_files, BOOK_DB_PATH, verbose=True)

    if args.style or args.all:
        from calibration.style_analyzer import save_style_profile, train_ken_style_profile
        print("\nStarting Ken's playstyle & move-level model training...")
        profile_data = train_ken_style_profile(db_path=BOOK_DB_PATH, verbose=True)
        save_style_profile(profile_data)


if __name__ == "__main__":
    main()
