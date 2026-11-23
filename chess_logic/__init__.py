"""Chess logic package for ChessKen."""

from chess_logic.chess_game import (
    ChessGame,
    GameOutcome,
    MoveInfo,
    PIECE_VALUES,
    STARTING_FEN,
)

# Friendly alias
ChessMoveLogic = ChessGame

__all__ = [
    "ChessGame",
    "ChessMoveLogic",
    "GameOutcome",
    "MoveInfo",
    "PIECE_VALUES",
    "STARTING_FEN",
]
