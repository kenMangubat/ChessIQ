"""Comprehensive tests for ChessGame / ChessMoveLogic."""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from chess_logic import ChessGame, ChessMoveLogic, STARTING_FEN


def test_initial_state():
    game = ChessGame()
    assert game.turn == "white"
    assert game.is_white_turn is True
    assert game.is_black_turn is False
    assert game.fullmove_number == 1
    assert game.is_check is False
    assert game.is_checkmate is False
    assert game.is_stalemate is False
    assert game.is_game_over is False
    assert game.fen == STARTING_FEN
    assert len(game.legal_moves()) == 20
    assert len(game.captured_by_white) == 0
    assert len(game.captured_by_black) == 0
    assert game.can_undo is False
    assert game.can_redo is False
    assert game.piece_symbol_at("e1") == "K"
    assert game.piece_symbol_at("e8") == "k"
    assert game.piece_symbol_at("e4") == "."


def test_coordinate_helpers():
    game = ChessGame()
    # String square
    assert game.parse_square("e4") == 28
    assert game.square_name(28) == "e4"
    # Integer square
    assert game.parse_square(0) == 0
    assert game.square_name(0) == "a1"
    assert game.square_name(63) == "h8"
    # Top-down coordinates: (row 0, col 0) -> a8 (square 56)
    assert game.square_from_coords(0, 0) == 56
    assert game.coords_from_square(56) == (0, 0)
    # (row 7, col 4) -> e1 (square 4)
    assert game.square_from_coords(7, 4) == 4
    assert game.coords_from_square(4) == (7, 4)


def test_legal_moves_query():
    game = ChessGame()
    # e2 pawn has e3 and e4
    targets = game.legal_moves_from_as_names("e2")
    assert sorted(targets) == ["e3", "e4"]

    # g1 knight has f3 and h3
    targets_knight = game.legal_moves_from_as_names("g1")
    assert sorted(targets_knight) == ["f3", "h3"]

    # Empty square has no legal moves
    assert game.legal_moves_from("e4") == []


def test_make_moves_and_history():
    game = ChessGame()
    # 1. e4
    info1 = game.make_move("e2", "e4")
    assert info1 is not None
    assert info1.san == "e4"
    assert info1.uci == "e2e4"
    assert info1.piece == "P"
    assert info1.color == "white"
    assert game.turn == "black"

    # 1... e5
    info2 = game.make_uci("e7e5")
    assert info2 is not None
    assert info2.san == "e5"
    assert game.turn == "white"

    # 2. Nf3
    info3 = game.make_san("Nf3")
    assert info3 is not None
    assert info3.piece == "N"
    assert info3.san == "Nf3"

    assert game.san_moves == ["e4", "e5", "Nf3"]
    assert game.fullmove_number == 2


def test_illegal_moves():
    game = ChessGame()
    # Illegal move: bishop through pawn
    assert game.make_move("f1", "c4") is None
    # Illegal move: pawn sideways
    assert game.make_move("e2", "d3") is None
    # Empty square
    assert game.make_move("e4", "e5") is None
    # Moving opponent's piece on White's turn
    assert game.make_move("e7", "e5") is None
    # Bad UCI string
    assert game.make_uci("invalid") is None
    # Bad SAN string
    assert game.make_san("Qxe8#") is None


def test_captures_and_material():
    game = ChessGame()
    game.make_san("e4")
    game.make_san("d5")
    assert game.is_capture_move("e4", "d5") is True

    # White takes pawn on d5
    info = game.make_move("e4", "d5")
    assert info is not None
    assert info.is_capture is True
    assert info.captured_piece == "P"
    assert game.captured_by_white == ["P"]
    assert game.captured_by_black == []

    # Material difference: White is +1
    diff = game.material_diff()
    assert diff == 1


def test_castling():
    game = ChessGame()
    # Move pieces out to allow White kingside castling
    game.make_san("e4")
    game.make_san("e5")
    game.make_san("Nf3")
    game.make_san("Nc6")
    game.make_san("Bc4")
    game.make_san("Bc5")

    # Castles O-O
    info = game.make_move("e1", "g1")
    assert info is not None
    assert info.is_castling is True
    assert info.castling_side == "kingside"
    assert info.san == "O-O"
    assert game.piece_symbol_at("g1") == "K"
    assert game.piece_symbol_at("f1") == "R"


def test_en_passant():
    game = ChessGame()
    game.make_san("e4")
    game.make_san("a6")
    game.make_san("e5")
    game.make_san("d5")  # Black moves 2 squares forward adjacent to e5

    # White captures en passant
    assert game.is_capture_move("e5", "d6") is True
    info = game.make_move("e5", "d6")
    assert info is not None
    assert info.is_en_passant is True
    assert info.is_capture is True
    assert info.captured_piece == "P"
    # Black pawn on d5 is gone
    assert game.piece_symbol_at("d5") == "."
    assert game.piece_symbol_at("d6") == "P"
    assert game.captured_by_white == ["P"]


def test_promotion():
    # Setup board one move away from promotion
    fen = "8/4P3/8/8/8/8/8/4K2k w - - 0 1"
    game = ChessGame(fen)
    assert game.is_promotion_move("e7", "e8") is True

    # Default promotion to Queen
    info = game.make_move("e7", "e8")
    assert info is not None
    assert info.is_promotion is True
    assert info.promotion_piece == "Q"
    assert game.piece_symbol_at("e8") == "Q"

    # Under-promotion to Knight on custom FEN
    game2 = ChessGame(fen)
    info2 = game2.make_move("e7", "e8", promotion="n")
    assert info2 is not None
    assert info2.promotion_piece == "N"
    assert game2.piece_symbol_at("e8") == "N"


def test_undo_and_redo():
    game = ChessGame()
    game.make_san("e4")
    game.make_san("d5")
    game.make_san("exd5")
    assert game.captured_by_white == ["P"]
    assert len(game.move_history) == 3

    # Undo capture
    undone = game.undo_move()
    assert undone is not None
    assert undone.san == "exd5"
    assert len(game.move_history) == 2
    assert game.captured_by_white == []
    assert game.can_redo is True

    # Redo capture
    redone = game.redo_move()
    assert redone is not None
    assert redone.san == "exd5"
    assert len(game.move_history) == 3
    assert game.captured_by_white == ["P"]


def test_checkmate():
    # Scholar's Mate
    game = ChessGame()
    game.make_san("e4")
    game.make_san("e5")
    game.make_san("Qh5")
    game.make_san("Nc6")
    game.make_san("Bc4")
    game.make_san("Nf6")
    info = game.make_san("Qxf7#")

    assert info is not None
    assert info.is_checkmate is True
    assert game.is_checkmate is True
    assert game.is_game_over is True

    outcome = game.outcome
    assert outcome.is_over is True
    assert outcome.status == "checkmate"
    assert outcome.winner == "white"
    assert "White wins" in outcome.description


def test_stalemate():
    # Known stalemate position
    fen = "k7/8/1Q6/8/8/8/8/4K3 b - - 0 1"
    game = ChessGame(fen)
    assert game.is_stalemate is True
    assert game.is_game_over is True
    outcome = game.outcome
    assert outcome.status == "stalemate"
    assert outcome.winner is None


def test_pgn_export():
    game = ChessGame()
    game.make_san("e4")
    game.make_san("e5")
    game.make_san("Nf3")
    pgn_str = game.get_pgn(white="Alice", black="Bob")
    assert '[White "Alice"]' in pgn_str
    assert '[Black "Bob"]' in pgn_str
    assert "1. e4 e5 2. Nf3" in pgn_str


def test_alias():
    # Verify ChessMoveLogic is an exact alias of ChessGame
    assert ChessMoveLogic is ChessGame


def test_legal_targets_deduplicated_for_promotion():
    game = ChessGame("8/P6k/8/8/8/8/8/K7 w - - 0 1")
    targets = game.legal_moves_from("a7")
    assert targets == [game.parse_square("a8")]
    assert game.is_promotion_move("a7", "a8") is True


def test_checked_king_square():
    game = ChessGame()
    assert game.checked_king_square() is None
    # Fool's mate: White king on e1 is in check (and mated)
    for san in ("f3", "e5", "g4", "Qh4#"):
        game.make_san(san)
    assert game.checked_king_square() == game.parse_square("e1")


def test_resign_and_agree_draw():
    game = ChessGame()
    game.make_san("e4")
    outcome = game.resign("white")
    assert game.is_game_over is True
    assert outcome.status == "resignation"
    assert outcome.winner == "black"
    assert game.outcome.termination == "Resignation"
    # Moves are refused after the game has ended
    assert game.make_san("e5") is None
    assert game.make_uci("e7e5") is None
    assert game.make_move("e7", "e5") is None
    # Resigning again does not override the result
    assert game.resign("black").winner == "black"

    game.reset()
    assert game.is_game_over is False
    assert game.outcome.status == "ongoing"
    draw = game.agree_draw()
    assert draw.status == "draw_agreed"
    assert draw.winner is None
    assert game.is_game_over is True


def test_game_over_reasons():
    stalemate = ChessGame("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1")
    assert stalemate.outcome.status == "stalemate"

    insufficient = ChessGame("8/8/8/4k3/8/8/8/4K3 w - - 0 1")
    assert insufficient.outcome.status == "insufficient_material"

    fifty = ChessGame("4k3/8/8/8/8/8/8/R3K3 w - - 100 80")
    assert fifty.outcome.status == "fifty_moves"
    assert fifty.is_game_over is True

    threefold = ChessGame()
    for _ in range(2):
        for san in ("Nf3", "Nf6", "Ng1", "Ng8"):
            threefold.make_san(san)
    assert threefold.outcome.status == "threefold_repetition"


if __name__ == "__main__":
    test_initial_state()
    test_coordinate_helpers()
    test_legal_moves_query()
    test_make_moves_and_history()
    test_illegal_moves()
    test_captures_and_material()
    test_castling()
    test_en_passant()
    test_promotion()
    test_undo_and_redo()
    test_checkmate()
    test_stalemate()
    test_pgn_export()
    test_alias()
    test_legal_targets_deduplicated_for_promotion()
    test_checked_king_square()
    test_resign_and_agree_draw()
    test_game_over_reasons()
    print("ALL CHESS LOGIC TESTS PASSED!")
