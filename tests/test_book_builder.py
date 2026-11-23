"""Unit tests for the offline PGN-to-SQLite opening book builder and query helpers."""

import os
from pathlib import Path
import sqlite3
import sys
import tempfile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import chess

from config import BOOK_DB_PATH
from calibration.build_book import (
    determine_tier,
    identify_ken_in_game,
    ingest_pgn_files,
    init_book_db,
    is_ken_player_name,
    query_book_moves,
    rebuild_book_from_db,
    TIER_SICK,
    TIER_FOCUS,
    TIER_CONDITION,
)


def test_determine_tier():
    """Verify rating thresholds for the 3 bot personality tiers."""
    assert determine_tier(1200) == TIER_SICK
    assert determine_tier(1399) == TIER_SICK
    assert determine_tier(1400) == TIER_FOCUS
    assert determine_tier(1650) == TIER_FOCUS
    assert determine_tier(1749) == TIER_FOCUS
    assert determine_tier(1750) == TIER_CONDITION
    assert determine_tier(2200) == TIER_CONDITION


def test_database_exists_and_valid():
    """Verify that ken_book.db was created and contains valid tables and records."""
    assert BOOK_DB_PATH.exists(), f"Database missing at {BOOK_DB_PATH}"
    assert BOOK_DB_PATH.stat().st_size > 0, "Database file is empty"

    conn = sqlite3.connect(BOOK_DB_PATH)
    cursor = conn.cursor()

    # Check table existence
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='book_moves'")
    assert cursor.fetchone() is not None, "Table 'book_moves' does not exist"

    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='book_metadata'")
    assert cursor.fetchone() is not None, "Table 'book_metadata' does not exist"

    # Verify positions and moves were populated
    cursor.execute("SELECT COUNT(*) FROM book_moves")
    total_moves = cursor.fetchone()[0]
    assert total_moves > 10000, f"Expected > 10,000 moves, found {total_moves}"

    cursor.execute("SELECT COUNT(DISTINCT epd) FROM book_moves")
    unique_epds = cursor.fetchone()[0]
    assert unique_epds > 8000, f"Expected > 8,000 unique EPDs, found {unique_epds}"

    conn.close()


def test_query_starting_position_white():
    """Verify that White's primary opening move from starting position is 1. e4 (e2e4)."""
    start_epd = chess.Board().epd()

    # Test each tier
    for tier in (TIER_SICK, TIER_FOCUS, TIER_CONDITION):
        moves = query_book_moves(start_epd, side="white", tier=tier, db_path=BOOK_DB_PATH)
        assert len(moves) > 0, f"No moves found for White starting position in tier {tier}"
        top_move = moves[0]
        assert top_move["move_uci"] == "e2e4", f"Expected top move e2e4, got {top_move['move_uci']}"
        assert top_move["tier"] == tier
        assert top_move["play_count"] > 0
        assert top_move["is_fallback"] is False

    # Test 'all' tiers aggregation
    moves_all = query_book_moves(start_epd, side="white", tier="all", db_path=BOOK_DB_PATH)
    assert len(moves_all) > 0
    assert moves_all[0]["move_uci"] == "e2e4"
    assert moves_all[0]["play_count"] > 1000


def test_query_black_responses_to_e4():
    """Verify that Black's primary response to 1. e4 is 1... c5 (c7c5, Sicilian)."""
    board = chess.Board()
    board.push_san("e4")
    after_e4_epd = board.epd()

    for tier in (TIER_SICK, TIER_FOCUS, TIER_CONDITION):
        moves = query_book_moves(after_e4_epd, side="black", tier=tier, db_path=BOOK_DB_PATH)
        assert len(moves) > 0, f"No moves found for Black after 1. e4 in tier {tier}"
        top_move = moves[0]
        assert top_move["move_uci"] == "c7c5", f"Expected top move c7c5, got {top_move['move_uci']}"
        assert top_move["tier"] == tier
        assert top_move["play_count"] > 0
        assert top_move["is_fallback"] is False

    moves_all = query_book_moves(after_e4_epd, side="black", tier="all", db_path=BOOK_DB_PATH)
    assert len(moves_all) > 0
    assert moves_all[0]["move_uci"] == "c7c5"
    assert moves_all[0]["play_count"] > 500


def test_fallback_behavior():
    """Verify fallback when an exact tier has no record but another tier does."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    try:
        conn = init_book_db(tmp_path)
        with conn:
            # Insert a position recorded ONLY for 'sick' tier
            conn.execute(
                """
                INSERT INTO book_moves (epd, side, move_uci, tier, play_count, wins, draws, losses)
                VALUES ('test_epd_123', 'white', 'd2d4', 'sick', 10, 6, 2, 2)
                """
            )
        conn.close()

        # Querying with tier='condition' and allow_fallback=True should fallback to 'sick'
        fallback_moves = query_book_moves(
            "test_epd_123",
            side="white",
            tier="condition",
            db_path=tmp_path,
            allow_fallback=True,
        )
        assert len(fallback_moves) == 1
        assert fallback_moves[0]["move_uci"] == "d2d4"
        assert fallback_moves[0]["tier"] == "sick"
        assert fallback_moves[0]["is_fallback"] is True
        assert fallback_moves[0]["play_count"] == 10

        # Querying with allow_fallback=False should return empty list
        no_fallback_moves = query_book_moves(
            "test_epd_123",
            side="white",
            tier="condition",
            db_path=tmp_path,
            allow_fallback=False,
        )
        assert len(no_fallback_moves) == 0
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def test_missing_position():
    """Verify querying an unrecorded position returns empty list."""
    moves = query_book_moves(
        "8/8/8/8/8/8/8/8 w - -",
        side="white",
        tier="focus",
        db_path=BOOK_DB_PATH,
    )
    assert moves == []


def test_alias_matching_study_and_bot_exclusion():
    """Verify Ken aliases, OTB study ChapterName parsing, and bot-name exclusion."""
    assert is_ken_player_name("Ken_is_not_here") is True
    assert is_ken_player_name("Mangubat") is True
    assert is_ken_player_name("UCLM_Ken2") is True
    assert is_ken_player_name("Sick Ken") is False
    assert is_ken_player_name("Focus Ken (bot)") is False
    assert is_ken_player_name("UCLM_Estremos") is False

    # Lichess study OTB format (White/Black = '?')
    study_headers = {
        "Event": "CESAFI 2026: Mangubat vs Peteros (Round 5)",
        "Site": "https://lichess.org/study/abc/def",
        "White": "?",
        "Black": "?",
        "Annotator": "UCLM_Ken2",
        "Orientation": "white",
        "ChapterName": "Mangubat vs Peteros (Round 5)",
    }
    side, is_otb, w_name, b_name = identify_ken_in_game(
        study_headers, source_file="lichess_study_cesafi-2026_by_UCLM_Ken2.pgn"
    )
    assert side == "white"
    assert is_otb is True
    assert w_name == "Mangubat"
    assert b_name == "Peteros"


def test_idempotent_ingest_holdout_split_and_weighted_book():
    """Verify idempotent PGN ingest, White/Black separation, 80/20 holdout split, and OTB weights."""
    sample_pgn = """
[Event "BCC Tournament: Mangubat vs OpponentA"]
[Site "https://lichess.org/study/s1/c1"]
[Date "2026.10.05"]
[White "?"]
[Black "?"]
[Result "1-0"]
[Annotator "UCLM_Ken2"]
[Orientation "white"]
[ChapterName "Mangubat vs OpponentA"]

1. d4 d5 2. c4 e6 3. Nc3 Nf6 4. Nf3 Be7 1-0

[Event "Rated Rapid game"]
[Site "https://lichess.org/game1"]
[Date "2024.06.01"]
[White "Ken_is_not_here"]
[Black "OnlineOpp1"]
[WhiteElo "1600"]
[Result "1-0"]
[TimeControl "600+0"]

1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 1-0

[Event "Rated Rapid game"]
[Site "https://lichess.org/game2"]
[Date "2024.06.02"]
[White "OnlineOpp2"]
[Black "Ken_is_not_here"]
[BlackElo "1620"]
[Result "0-1"]
[TimeControl "600+0"]

1. e4 c5 2. Nf3 d6 3. d4 cxd4 4. Nxd4 Nf6 0-1

[Event "Rated Rapid game"]
[Site "https://lichess.org/game3"]
[Date "2024.06.03"]
[White "Ken_is_not_here"]
[Black "OnlineOpp3"]
[WhiteElo "1650"]
[Result "1-0"]
[TimeControl "600+0"]

1. e4 c5 2. Nf3 Nc6 3. d4 cxd4 4. Nxd4 e5 1-0

[Event "Rated Rapid game"]
[Site "https://lichess.org/game4"]
[Date "2024.06.04"]
[White "OnlineOpp4"]
[Black "Ken_is_not_here"]
[BlackElo "1660"]
[Result "0-1"]
[TimeControl "600+0"]

1. d4 Nf6 2. c4 g6 3. Nc3 Bg7 4. e4 d6 0-1

[Event "Bot Sparring"]
[Site "ChessIQ"]
[Date "2026.10.20"]
[White "Focus Ken"]
[Black "RandomOpp"]
[Result "1-0"]

1. e4 e5 2. Nf3 Nc6 3. Bc4 Bc5 1-0
""".strip()

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_pgn = Path(tmpdir) / "lichess_study_test.pgn"
        tmp_db = Path(tmpdir) / "test_book.db"
        tmp_pgn.write_text(sample_pgn, encoding="utf-8")

        # Pass 1: ingest 5 Ken games (1 OTB study + 4 online; bot game excluded)
        s1 = ingest_pgn_files([tmp_pgn], db_path=tmp_db, verbose=False)
        assert s1["new_games_added"] == 5
        assert s1["white_games"] == 3
        assert s1["black_games"] == 2
        assert s1["otb_games"] >= 1
        assert s1["splits"]["train"] > 0
        assert s1["splits"]["holdout"] > 0
        assert s1["splits"]["train"] + s1["splits"]["holdout"] == 5

        # Pass 2: idempotency check (0 new games added)
        s2 = ingest_pgn_files([tmp_pgn], db_path=tmp_db, verbose=False)
        assert s2["new_games_added"] == 0
        assert s2["already_present"] == 5
        assert s2["total_db_games"] == 5

        # Rebuild book and verify OTB move (d2d4) received higher weighted_count than a single older online move
        rebuild_book_from_db(db_path=tmp_db, verbose=False)
        start_moves = query_book_moves(
            chess.Board().epd(), side="white", tier="all", db_path=tmp_db
        )
        by_uci = {m["move_uci"]: m for m in start_moves}
        assert "d2d4" in by_uci and "e2e4" in by_uci
        assert by_uci["d2d4"]["weighted_count"] >= 3.0  # OTB weight = 3.0


if __name__ == "__main__":
    test_determine_tier()
    test_database_exists_and_valid()
    test_query_starting_position_white()
    test_query_black_responses_to_e4()
    test_fallback_behavior()
    test_missing_position()
    test_alias_matching_study_and_bot_exclusion()
    test_idempotent_ingest_holdout_split_and_weighted_book()
    print("ALL BOOK BUILDER TESTS PASSED!")
