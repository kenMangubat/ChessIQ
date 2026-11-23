"""Unit tests for StockfishEngine, KenBot profiles, book selection, heuristic fallback, and BotWorker."""

import os
from pathlib import Path
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import chess

from bots.ken_bot import (
    BotWorker,
    fallback_heuristic_move,
    get_profile,
    KenBot,
    KEN_PROFILES,
)
from calibration.build_book import query_book_moves
from calibration.calibrate_bots import estimate_elo_mle, play_calibration_game
from config import BOOK_DB_PATH, get_stockfish_path
from engine.stockfish_engine import EngineNotFoundError, StockfishEngine


def test_profiles_configuration():
    """Verify all 3 Ken bot profiles exist and have proper calibrations."""
    assert len(KEN_PROFILES) == 3
    for name in ("Sick Ken", "Focus Ken", "Condition Ken"):
        profile = get_profile(name)
        assert profile.name == name
        assert profile.elo > 1000
        assert profile.temperature > 0
        assert profile.engine_time_limit > 0

    # Test alias and fallback lookups
    assert get_profile("sick").name == "Sick Ken"
    assert get_profile("focus").name == "Focus Ken"
    assert get_profile("condition").name == "Condition Ken"
    assert get_profile("unknown_bot").name == "Focus Ken"


def test_book_move_selection_white():
    """Verify that all three Ken bots pick a recorded opening move from the starting position."""
    board = chess.Board()
    start_epd = board.epd()

    for bot_name in ("Sick Ken", "Focus Ken", "Condition Ken"):
        bot = KenBot(profile=bot_name)
        chosen = bot.choose_move(board)
        assert chosen in board.legal_moves, f"{bot_name} chose an illegal move"

        book_candidates = query_book_moves(start_epd, side="white", tier=bot.profile.tier, db_path=BOOK_DB_PATH)
        book_ucis = {c["move_uci"] for c in book_candidates}
        assert chosen.uci() in book_ucis, f"{bot_name} move {chosen.uci()} not in book {book_ucis}"
        # e2e4 is overwhelmingly Ken's primary move
        assert "e2e4" in book_ucis
        bot.close()


def test_book_move_selection_black():
    """Verify that all three Ken bots pick a recorded response to 1. e4 as Black."""
    board = chess.Board()
    board.push_san("e4")
    after_e4_epd = board.epd()

    for bot_name in ("Sick Ken", "Focus Ken", "Condition Ken"):
        bot = KenBot(profile=bot_name)
        chosen = bot.choose_move(board)
        assert chosen in board.legal_moves, f"{bot_name} chose an illegal move as Black"

        book_candidates = query_book_moves(after_e4_epd, side="black", tier=bot.profile.tier, db_path=BOOK_DB_PATH)
        book_ucis = {c["move_uci"] for c in book_candidates}
        assert chosen.uci() in book_ucis, f"{bot_name} response {chosen.uci()} not in book {book_ucis}"
        # c7c5 (Sicilian) is in book for all tiers
        assert "c7c5" in book_ucis
        bot.close()


def test_out_of_book_fallback_heuristic():
    """Verify that an out-of-book position produces a legal move via heuristic fallback."""
    # A position definitely not in opening book (endgame position)
    board = chess.Board("8/8/4k3/8/8/4K3/4P3/8 w - - 0 1")

    profile = get_profile("Focus Ken")
    move = fallback_heuristic_move(board, profile)
    assert move in board.legal_moves

    # Test KenBot directly on out-of-book position without Stockfish
    bot = KenBot(profile="Focus Ken")
    chosen = bot.choose_move(board)
    assert chosen in board.legal_moves
    bot.close()


def test_stockfish_engine_missing_binary():
    """Verify StockfishEngine raises EngineNotFoundError when binary is absent."""
    engine = StockfishEngine(executable_path=Path("non_existent_engine_path_123.exe"))
    try:
        engine.analyse_candidates(chess.Board())
        assert False, "Expected EngineNotFoundError was not raised"
    except EngineNotFoundError:
        pass


def test_bot_worker_async():
    """Verify BotWorker executes move calculation in background without blocking."""
    bot = KenBot(profile="Focus Ken")
    worker = BotWorker(bot)

    board = chess.Board()
    # Initially not thinking
    assert not worker.is_thinking

    # Request move with small delay to test async polling
    worker.request_move(board, delay_seconds=0.05)
    assert worker.is_thinking

    # Immediate poll might be None while thinking
    poll1 = worker.poll_move()

    # Wait up to 1.5 seconds for computation to finish
    computed_move = None
    for _ in range(30):
        time.sleep(0.05)
        computed_move = worker.poll_move()
        if computed_move is not None:
            break

    assert computed_move is not None, "Worker failed to return a move"
    assert computed_move in board.legal_moves
    assert not worker.is_thinking

    # Clean shutdown
    worker.stop()


def test_stockfish_uci_options_and_fallback_telemetry():
    """Verify Stockfish binary resolution, UCI_Elo clamping/reset, and zero fallback when active."""
    sf_path = get_stockfish_path()
    assert sf_path is not None and sf_path.is_file(), f"Expected Stockfish binary, got {sf_path}"

    with StockfishEngine() as eng:
        min_elo, max_elo = eng.uci_elo_range
        assert min_elo <= 1350 and max_elo >= 2400

        board = chess.Board("r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3")
        # Request with out-of-bounds low uci_elo -> clamps cleanly
        cands_limited = eng.analyse_candidates(board, multipv=2, time_limit=0.03, uci_elo=1000)
        assert len(cands_limited) == 2
        assert eng.applied_options["uci_limit_strength"] is True
        assert eng.applied_options["uci_elo"] == min_elo

        # Request with uci_elo=None -> resets UCI_LimitStrength to False
        cands_full = eng.analyse_candidates(board, multipv=2, time_limit=0.03, uci_elo=None)
        assert len(cands_full) == 2
        assert eng.applied_options["uci_limit_strength"] is False
        assert eng.applied_options["uci_elo"] is None

    # Out-of-book move with real Stockfish must use style-first engine pipeline (fallback_count == 0)
    endgame_board = chess.Board("8/8/4k3/8/8/4K3/4P3/8 w - - 0 1")
    bot = KenBot(profile="Focus Ken", log_provenance=False)
    try:
        chosen = bot.choose_move(endgame_board)
        assert chosen in endgame_board.legal_moves
        assert bot.fallback_count == 0
        assert bot.engine_move_count == 1
        assert bot.last_move_source in ("style-selected", "error-model")
    finally:
        bot.close()

    # Missing engine binary increments fallback_count
    missing_bot = KenBot(
        profile="Focus Ken",
        engine=StockfishEngine(executable_path=Path("missing_sf_binary_xyz.exe")),
        log_provenance=False,
    )
    chosen_fb = missing_bot.choose_move(endgame_board)
    assert chosen_fb in endgame_board.legal_moves
    assert missing_bot.fallback_count == 1
    assert missing_bot.last_move_source == "fallback"


def test_pipeline_order_safety_supremacy_and_provenance():
    """Verify pipeline order (book first, zero fallback), safety supremacy, and provenance logging."""
    import tempfile
    from calibration.explain_move import replay_log

    with tempfile.TemporaryDirectory() as tmpdir:
        log_dir = Path(tmpdir)
        bot = KenBot(profile="Condition Ken", log_provenance=True, provenance_dir=log_dir)
        try:
            # 1. Starting position must come from opening book (Step 1)
            start_board = chess.Board()
            m1 = bot.choose_move(start_board)
            assert m1 in start_board.legal_moves
            assert bot.last_move_source == "book"
            assert bot.fallback_count == 0

            # 2. Safety-filter supremacy: position where mate-in-1 (Rxe8#) exists
            mate_board = chess.Board("4r1k1/5ppp/8/8/8/8/5PPP/4R1K1 w - - 0 1")
            m2 = bot.choose_move(mate_board)
            assert m2 == chess.Move.from_uci("e1e8")
            assert bot.last_move_source == "style-selected"
            assert bot.fallback_count == 0

            # 3. Verify provenance JSONL log was created and contains required fields
            assert bot.session_log_path is not None and bot.session_log_path.exists()
            records = replay_log(bot.session_log_path)
            assert len(records) >= 1
            last_rec = records[-1]
            for key in ("source", "chosen_uci", "engine_top1_uci", "equals_engine_top1", "cpl", "style_prob"):
                assert key in last_rec, f"Missing provenance key {key}"
        finally:
            bot.close()


def test_calibration_harness_smoke():
    """Verify MLE Elo estimation and a quick single-game calibration smoke test."""
    est, lo, hi = estimate_elo_mle([(1800, 1.0), (1800, 0.0), (1800, 0.5)], prior_elo=1800.0)
    assert 1750.0 <= est <= 1850.0
    assert lo < est < hi

    rec = play_calibration_game(
        profile_name="Focus Ken",
        ref_elo=1600,
        bot_is_white=True,
        opening_name="Start Position",
        opening_moves=(),
        ref_time_limit=0.02,
        eval_depth=5,
        max_plies=12,
    )
    assert rec["profile"] == "Focus Ken"
    assert rec["fallback_moves"] == 0
    assert rec["plies"] == 12
    assert rec["bot_score"] in (0.0, 0.5, 1.0)


if __name__ == "__main__":
    test_profiles_configuration()
    test_book_move_selection_white()
    test_book_move_selection_black()
    test_out_of_book_fallback_heuristic()
    test_stockfish_engine_missing_binary()
    test_bot_worker_async()
    test_stockfish_uci_options_and_fallback_telemetry()
    test_pipeline_order_safety_supremacy_and_provenance()
    test_calibration_harness_smoke()
    print("ALL KEN BOT TESTS PASSED!")
