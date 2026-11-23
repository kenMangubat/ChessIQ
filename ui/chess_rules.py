"""Legal chess rules shared by gameplay and the animated menu backdrop."""

from __future__ import annotations

import chess

START_FEN = chess.STARTING_FEN


class ChessRules:
    """Small gameplay wrapper around python-chess' authoritative board."""

    def __init__(self, fen: str = START_FEN) -> None:
        self.board = chess.Board(fen)
        self._san_moves: list[str] = []

    @staticmethod
    def board_coords_to_square(row: int, col: int) -> chess.Square:
        """Convert top-down board row/column coordinates to a chess square."""
        if not (0 <= row < 8 and 0 <= col < 8):
            raise ValueError(f"board coordinates out of range: {(row, col)}")
        return chess.square(col, 7 - row)

    @staticmethod
    def square_to_board_coords(square: chess.Square) -> tuple[int, int]:
        """Convert a chess square to top-down board row/column coordinates."""
        if not 0 <= square < 64:
            raise ValueError(f"invalid chess square: {square}")
        return 7 - chess.square_rank(square), chess.square_file(square)

    @property
    def turn(self) -> str:
        return "w" if self.board.turn == chess.WHITE else "b"

    @property
    def in_check(self) -> bool:
        return self.board.is_check()

    @property
    def last_move(self) -> chess.Move | None:
        return self.board.peek() if self.board.move_stack else None

    @property
    def move_stack(self) -> list[chess.Move]:
        return list(self.board.move_stack)

    @property
    def san_list(self) -> list[str]:
        return list(self._san_moves)

    def piece_at(self, file: int, rank: int) -> str:
        piece = self.board.piece_at(chess.square(file, rank))
        return piece.symbol() if piece is not None else "."

    def legal_moves_from(self, square: chess.Square) -> list[chess.Square]:
        if not 0 <= square < 64:
            return []
        return list(dict.fromkeys(
            move.to_square
            for move in self.board.legal_moves
            if move.from_square == square
        ))

    def is_promotion_move(
        self, from_sq: chess.Square, to_sq: chess.Square
    ) -> bool:
        if not (0 <= from_sq < 64 and 0 <= to_sq < 64):
            return False
        return any(
            move.from_square == from_sq
            and move.to_square == to_sq
            and move.promotion is not None
            for move in self.board.legal_moves
        )

    def is_capture_move(self, from_sq: chess.Square, to_sq: chess.Square) -> bool:
        if not (0 <= from_sq < 64 and 0 <= to_sq < 64):
            return False
        return self.board.is_capture(chess.Move(from_sq, to_sq))

    def make_move(
        self,
        from_sq: chess.Square,
        to_sq: chess.Square,
        promotion: str | int | None = None,
    ) -> bool:
        if not (0 <= from_sq < 64 and 0 <= to_sq < 64):
            return False
        promotion_types = {
            "q": chess.QUEEN,
            "r": chess.ROOK,
            "b": chess.BISHOP,
            "n": chess.KNIGHT,
        }
        if isinstance(promotion, str):
            promotion_type = promotion_types.get(promotion.lower())
        elif promotion in promotion_types.values():
            promotion_type = promotion
        elif promotion is None:
            promotion_type = None
        else:
            return False
        if promotion is not None and promotion_type is None:
            return False

        move = chess.Move(from_sq, to_sq, promotion=promotion_type)
        if not self.board.is_legal(move):
            return False

        self._san_moves.append(self.board.san(move))
        self.board.push(move)
        return True

    def game_state(self) -> tuple[str, str | None]:
        outcome = self.board.outcome(claim_draw=True)
        if outcome is None:
            return "ongoing", None
        state_names = {
            chess.Termination.CHECKMATE: "checkmate",
            chess.Termination.STALEMATE: "stalemate",
            chess.Termination.INSUFFICIENT_MATERIAL: "insufficient_material",
            chess.Termination.FIFTY_MOVES: "fifty_move",
            chess.Termination.THREEFOLD_REPETITION: "threefold_repetition",
            chess.Termination.FIVEFOLD_REPETITION: "fivefold",
            chess.Termination.SEVENTYFIVE_MOVES: "seventyfive_move",
        }
        winner = None
        if outcome.winner is not None:
            winner = "white" if outcome.winner == chess.WHITE else "black"
        return state_names[outcome.termination], winner

    def reset(self) -> None:
        self.board.reset()
        self._san_moves.clear()


class Position(ChessRules):
    """Legacy backdrop API backed by the same python-chess rules engine."""

    def push_uci(self, uci: str) -> dict[str, object]:
        try:
            move = chess.Move.from_uci(uci)
        except ValueError as exc:
            raise ValueError(f"bad move {uci}") from exc
        if not self.board.is_legal(move):
            raise ValueError(f"{uci}: illegal move")

        moving_piece = self.board.piece_at(move.from_square)
        if moving_piece is None:
            raise ValueError(f"{uci}: no piece on the origin")
        captured_piece = self.board.piece_at(move.to_square)
        en_passant = self.board.is_en_passant(move)
        if en_passant:
            captured_square = move.to_square - 8 if self.board.turn else move.to_square + 8
            captured_piece = self.board.piece_at(captured_square)
        san = self.board.san(move)
        is_castling = self.board.is_castling(move)
        castle = None
        if is_castling:
            rank = chess.square_rank(move.from_square)
            if chess.square_file(move.to_square) == 6:
                castle = ((7, rank), (5, rank))
            else:
                castle = ((0, rank), (3, rank))

        if not self.make_move(move.from_square, move.to_square, move.promotion):
            raise ValueError(f"{uci}: illegal move")

        capture = None
        if captured_piece is not None:
            capture = (moving_piece.symbol().upper(), captured_piece.symbol().upper())
        return {
            "uci": uci,
            "san": san,
            "piece": moving_piece.symbol().upper(),
            "color": "w" if moving_piece.color == chess.WHITE else "b",
            "fr": (chess.square_file(move.from_square), chess.square_rank(move.from_square)),
            "to": (chess.square_file(move.to_square), chess.square_rank(move.to_square)),
            "capture": capture,
            "ep": en_passant,
            "cap_sq": (chess.square_file(captured_square), chess.square_rank(captured_square)) if en_passant else None,
            "castle": castle,
            "promo": chess.piece_symbol(move.promotion) if move.promotion else "",
            "check": self.board.is_check(),
            "mate": self.board.is_checkmate(),
        }


# Three opening lines used by the animated dashboard background.
# UCI only — SAN and capture names are produced by the rules, not typed by hand.
LINES: list[dict] = [
    {
        "title": "Sicilian Defense",
        "subtitle": "Najdorf Variation, Bg5 Line",
        "moves": [
            "e2e4", "c7c5", "g1f3", "d7d6", "d2d4", "c5d4", "f3d4", "g8f6",
            "b1c3", "a7a6", "c1g5", "e7e6", "f2f4", "f8e7", "d1f3", "d8c7",
            "e1c1", "b8d7", "g2g4", "b7b5",
        ],
    },
    {
        "title": "Ruy Lopez",
        "subtitle": "Closed Spanish Main Line",
        "moves": [
            "e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "a7a6", "b5a4", "g8f6",
            "e1g1", "f8e7", "f1e1", "b7b5", "a4b3", "d7d6", "c2c3", "e8g8",
            "h2h3", "c6b8", "d2d4", "b8d7",
        ],
    },
    {
        "title": "Queen's Gambit Declined",
        "subtitle": "Classical Development",
        "moves": [
            "d2d4", "d7d5", "c2c4", "e7e6", "b1c3", "g8f6", "c1g5", "f8e7",
            "e2e3", "e8g8", "g1f3", "h7h6", "g5h4", "b7b6", "c4d5", "f6d5",
            "h4e7", "d8e7",
        ],
    },
]

PIECE_NAMES = {
    "K": "King",
    "Q": "Queen",
    "R": "Rook",
    "B": "Bishop",
    "N": "Knight",
    "P": "Pawn",
}


def verify_lines() -> list[str]:
    """Play every backdrop line. Raise if any move is illegal or mislabeled."""
    report: list[str] = []
    for line in LINES:
        pos = Position(line.get("fen", START_FEN))
        sans: list[str] = []
        for uci in line["moves"]:
            info = pos.push_uci(uci)
            sans.append(info["san"])
            if info["capture"]:
                attacker, victim = info["capture"]
                expect = f"{PIECE_NAMES[attacker]} takes {PIECE_NAMES[victim]}"
                if expect != f"{PIECE_NAMES[info['piece']]} takes {PIECE_NAMES[victim]}":
                    raise AssertionError(expect)
        if line.get("ends_in_mate") and not sans[-1].endswith("#"):
            raise AssertionError(f"{line['title']} did not end in mate: {sans[-1]}")
        report.append(f"{line['title']}: {' '.join(sans)}")
    return report
