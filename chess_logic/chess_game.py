"""Core chess move and game logic.

Wraps python-chess to provide authoritative move validation, execution,
undo/redo, capture tracking, game outcome evaluation, and coordinate conversions.
No engine or bot dependencies — pure rule-based chess logic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import datetime
import chess
import chess.pgn

STARTING_FEN = chess.STARTING_FEN

PIECE_VALUES: dict[str, int] = {
    "P": 1,
    "N": 3,
    "B": 3,
    "R": 5,
    "Q": 9,
    "K": 0,
}


@dataclass
class MoveInfo:
    """Detailed metadata for a single executed chess move."""

    move: chess.Move
    san: str
    uci: str
    from_square: int
    to_square: int
    from_square_name: str
    to_square_name: str
    piece: str
    color: str
    is_capture: bool
    captured_piece: str | None = None
    captured_square: int | None = None
    is_castling: bool = False
    castling_side: str | None = None
    is_en_passant: bool = False
    is_promotion: bool = False
    promotion_piece: str | None = None
    is_check: bool = False
    is_checkmate: bool = False
    fen_before: str = ""
    fen_after: str = ""


@dataclass
class GameOutcome:
    """Current state and outcome of the game."""

    is_over: bool
    status: str
    winner: str | None
    termination: str | None
    description: str


class ChessGame:
    """Full-featured chess game and move logic engine.

    Operates on a python-chess board, handling move validation, execution,
    undo/redo history, capture accounting, and game termination checks.
    """

    def __init__(self, fen: str = STARTING_FEN) -> None:
        self._initial_fen = fen
        self._board = chess.Board(fen)
        self._move_history: list[MoveInfo] = []
        self._redo_stack: list[chess.Move] = []
        self._captured_by_white: list[str] = []
        self._captured_by_black: list[str] = []
        # Set by resign() / agree_draw(); overrides board-derived outcome.
        self._custom_outcome: GameOutcome | None = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def board(self) -> chess.Board:
        """Underlying python-chess Board object."""
        return self._board

    @property
    def fen(self) -> str:
        """Current FEN string."""
        return self._board.fen()

    @property
    def initial_fen(self) -> str:
        """Starting FEN string of this game."""
        return self._initial_fen

    @property
    def turn(self) -> str:
        """Current player's turn ('white' or 'black')."""
        return "white" if self._board.turn == chess.WHITE else "black"

    @property
    def turn_color(self) -> chess.Color:
        """Current player's chess.Color (chess.WHITE or chess.BLACK)."""
        return self._board.turn

    @property
    def is_white_turn(self) -> bool:
        """True if it is White's turn."""
        return self._board.turn == chess.WHITE

    @property
    def is_black_turn(self) -> bool:
        """True if it is Black's turn."""
        return self._board.turn == chess.BLACK

    @property
    def fullmove_number(self) -> int:
        """The number of the full moves. Starts at 1 and is incremented after Black's move."""
        return self._board.fullmove_number

    @property
    def halfmove_clock(self) -> int:
        """Number of halfmoves since the last capture or pawn advance."""
        return self._board.halfmove_clock

    @property
    def is_check(self) -> bool:
        """True if the king of the side to move is currently under check."""
        return self._board.is_check()

    @property
    def is_checkmate(self) -> bool:
        """True if the side to move is checkmated."""
        return self._board.is_checkmate()

    @property
    def is_stalemate(self) -> bool:
        """True if the side to move is stalemated (draw)."""
        return self._board.is_stalemate()

    @property
    def is_insufficient_material(self) -> bool:
        """True if neither side has sufficient material to checkmate."""
        return self._board.is_insufficient_material()

    @property
    def is_fivefold_repetition(self) -> bool:
        """True if the exact position has occurred at least five times (FIDE automatic draw)."""
        return self._board.is_fivefold_repetition()

    @property
    def is_seventyfive_moves(self) -> bool:
        """True if 75 moves have occurred without a capture or pawn move (FIDE automatic draw)."""
        return self._board.is_seventyfive_moves()

    @property
    def can_claim_fifty_moves(self) -> bool:
        """True if a player can claim a draw by the 50-move rule."""
        return self._board.can_claim_fifty_moves()

    @property
    def can_claim_threefold_repetition(self) -> bool:
        """True if a player can claim a draw by threefold repetition."""
        return self._board.can_claim_threefold_repetition()

    @property
    def can_claim_draw(self) -> bool:
        """True if a draw can be claimed (either threefold repetition or 50 moves)."""
        return self._board.can_claim_draw()

    @property
    def is_game_over(self) -> bool:
        """True if the game is over (by checkmate, stalemate, insufficient material,
        claimable draw, resignation, or agreed draw)."""
        return self._custom_outcome is not None or self._board.is_game_over(claim_draw=True)

    @property
    def outcome(self) -> GameOutcome:
        """Evaluate the current status and outcome of the game."""
        if self._custom_outcome is not None:
            return self._custom_outcome

        if self._board.is_checkmate():
            winner = "black" if self._board.turn == chess.WHITE else "white"
            return GameOutcome(
                is_over=True,
                status="checkmate",
                winner=winner,
                termination="Checkmate",
                description=f"Checkmate — {winner.capitalize()} wins!",
            )

        if self._board.is_stalemate():
            return GameOutcome(
                is_over=True,
                status="stalemate",
                winner=None,
                termination="Stalemate",
                description="Draw by stalemate.",
            )

        if self._board.is_insufficient_material():
            return GameOutcome(
                is_over=True,
                status="insufficient_material",
                winner=None,
                termination="Insufficient Material",
                description="Draw by insufficient material.",
            )

        if self._board.is_seventyfive_moves():
            return GameOutcome(
                is_over=True,
                status="seventyfive_moves",
                winner=None,
                termination="75-Move Rule",
                description="Draw by 75-move rule.",
            )

        if self._board.is_fivefold_repetition():
            return GameOutcome(
                is_over=True,
                status="fivefold_repetition",
                winner=None,
                termination="Fivefold Repetition",
                description="Draw by fivefold repetition.",
            )

        if self._board.can_claim_threefold_repetition():
            return GameOutcome(
                is_over=True,
                status="threefold_repetition",
                winner=None,
                termination="Threefold Repetition",
                description="Draw can be claimed by threefold repetition.",
            )

        if self._board.can_claim_fifty_moves():
            return GameOutcome(
                is_over=True,
                status="fifty_moves",
                winner=None,
                termination="50-Move Rule",
                description="Draw can be claimed by 50-move rule.",
            )

        return GameOutcome(
            is_over=False,
            status="ongoing",
            winner=None,
            termination=None,
            description=f"Ongoing — {self.turn.capitalize()} to move.",
        )

    @property
    def move_history(self) -> list[MoveInfo]:
        """Chronological list of all executed moves with rich metadata."""
        return list(self._move_history)

    @property
    def san_moves(self) -> list[str]:
        """List of moves in Standard Algebraic Notation (SAN), e.g. ['e4', 'e5', 'Nf3']."""
        return [info.san for info in self._move_history]

    @property
    def uci_moves(self) -> list[str]:
        """List of moves in UCI format, e.g. ['e2e4', 'e7e5', 'g1f3']."""
        return [info.uci for info in self._move_history]

    @property
    def captured_by_white(self) -> list[str]:
        """Pieces captured by White (uppercase symbols: 'P', 'N', 'B', 'R', 'Q')."""
        return list(self._captured_by_white)

    @property
    def captured_by_black(self) -> list[str]:
        """Pieces captured by Black (uppercase symbols: 'P', 'N', 'B', 'R', 'Q')."""
        return list(self._captured_by_black)

    @property
    def can_undo(self) -> bool:
        """True if there is at least one move to undo."""
        return len(self._move_history) > 0

    @property
    def can_redo(self) -> bool:
        """True if there is an undone move available to redo."""
        return len(self._redo_stack) > 0

    @property
    def last_move(self) -> MoveInfo | None:
        """The most recently played move, or None if no moves have been made."""
        return self._move_history[-1] if self._move_history else None

    # ------------------------------------------------------------------
    # Square & Coordinate Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def parse_square(sq: int | str | tuple[int, int]) -> int:
        """Convert an int (0..63), string ('e4'), or top-down (row, col) to a chess.Square.

        Top-down row coordinates: row 0 = rank 8 (top), row 7 = rank 1 (bottom).
        Col coordinates: col 0 = file a (left), col 7 = file h (right).
        """
        if isinstance(sq, int):
            if not 0 <= sq < 64:
                raise ValueError(f"Square index out of range: {sq}")
            return sq

        if isinstance(sq, str):
            sq_clean = sq.strip().lower()
            if len(sq_clean) != 2:
                raise ValueError(f"Invalid algebraic square string: '{sq}'")
            file = ord(sq_clean[0]) - ord("a")
            rank = int(sq_clean[1]) - 1
            if not (0 <= file < 8 and 0 <= rank < 8):
                raise ValueError(f"Square coordinates out of range: '{sq}'")
            return chess.square(file, rank)

        if isinstance(sq, (tuple, list)):
            if len(sq) != 2:
                raise ValueError(f"Expected 2-tuple (row, col), got: {sq}")
            row, col = sq
            if not (0 <= row < 8 and 0 <= col < 8):
                raise ValueError(f"Board coordinates out of range: (row={row}, col={col})")
            return chess.square(col, 7 - row)

        raise TypeError(f"Cannot parse square from type {type(sq)}: {sq}")

    @staticmethod
    def square_name(square: int) -> str:
        """Return the algebraic name for a square (e.g. 0 -> 'a1', 63 -> 'h8')."""
        return chess.square_name(square)

    @staticmethod
    def square_from_coords(row: int, col: int) -> int:
        """Convert top-down board coordinates (row: 0..7, col: 0..7) to chess.Square."""
        if not (0 <= row < 8 and 0 <= col < 8):
            raise ValueError(f"Coordinates out of range: (row={row}, col={col})")
        return chess.square(col, 7 - row)

    @staticmethod
    def coords_from_square(square: int) -> tuple[int, int]:
        """Convert a chess.Square (0..63) to top-down board coordinates (row, col)."""
        if not 0 <= square < 64:
            raise ValueError(f"Square out of range: {square}")
        return 7 - chess.square_rank(square), chess.square_file(square)

    # ------------------------------------------------------------------
    # Board Inspection
    # ------------------------------------------------------------------

    def piece_at(self, square: int | str | tuple[int, int]) -> chess.Piece | None:
        """Get the chess.Piece at the given square, or None if empty."""
        sq = self.parse_square(square)
        return self._board.piece_at(sq)

    def piece_symbol_at(self, square: int | str | tuple[int, int]) -> str:
        """Return the piece symbol ('P', 'n', etc.) at the square, or '.' if empty."""
        piece = self.piece_at(square)
        return piece.symbol() if piece is not None else "."

    def legal_moves(self) -> list[chess.Move]:
        """List all legal moves in the current position."""
        return list(self._board.legal_moves)

    def legal_moves_from(self, square: int | str | tuple[int, int]) -> list[int]:
        """Get all unique legal target squares for a piece at the given square.

        Promotion moves (four python-chess moves per target) yield the target once.
        """
        sq = self.parse_square(square)
        return list(dict.fromkeys(
            move.to_square
            for move in self._board.legal_moves
            if move.from_square == sq
        ))

    def checked_king_square(self) -> int | None:
        """Return the square of the side-to-move's king if it is in check, else None."""
        if not self._board.is_check():
            return None
        return self._board.king(self._board.turn)

    def legal_moves_from_as_names(
        self, square: int | str | tuple[int, int]
    ) -> list[str]:
        """Get all legal target square names (e.g. ['e3', 'e4']) for piece at square."""
        return [self.square_name(sq) for sq in self.legal_moves_from(square)]

    def is_legal_move(
        self,
        from_sq: int | str | tuple[int, int],
        to_sq: int | str | tuple[int, int],
        promotion: str | None = None,
    ) -> bool:
        """Check if a move is legal."""
        try:
            f = self.parse_square(from_sq)
            t = self.parse_square(to_sq)
        except (ValueError, TypeError):
            return False

        promo_type = self._parse_promotion(promotion)
        move = chess.Move(f, t, promotion=promo_type)
        if self._board.is_legal(move):
            return True

        # If promotion was omitted but move is a promotion, test default Queen
        if promo_type is None and self.is_promotion_move(f, t):
            return self._board.is_legal(chess.Move(f, t, promotion=chess.QUEEN))

        return False

    def is_promotion_move(
        self,
        from_sq: int | str | tuple[int, int],
        to_sq: int | str | tuple[int, int],
    ) -> bool:
        """Check if a move from from_sq to to_sq constitutes a pawn promotion."""
        try:
            f = self.parse_square(from_sq)
            t = self.parse_square(to_sq)
        except (ValueError, TypeError):
            return False

        piece = self._board.piece_at(f)
        if piece is None or piece.piece_type != chess.PAWN:
            return False

        dest_rank = chess.square_rank(t)
        if piece.color == chess.WHITE and dest_rank == 7:
            return any(
                m.from_square == f and m.to_square == t and m.promotion is not None
                for m in self._board.legal_moves
            )
        if piece.color == chess.BLACK and dest_rank == 0:
            return any(
                m.from_square == f and m.to_square == t and m.promotion is not None
                for m in self._board.legal_moves
            )
        return False

    def is_capture_move(
        self,
        from_sq: int | str | tuple[int, int],
        to_sq: int | str | tuple[int, int],
    ) -> bool:
        """Check if moving from from_sq to to_sq results in a capture."""
        try:
            f = self.parse_square(from_sq)
            t = self.parse_square(to_sq)
        except (ValueError, TypeError):
            return False

        promo = chess.QUEEN if self.is_promotion_move(f, t) else None
        move = chess.Move(f, t, promotion=promo)
        if not self._board.is_legal(move):
            return False
        return self._board.is_capture(move)

    # ------------------------------------------------------------------
    # Move Execution
    # ------------------------------------------------------------------

    def make_move(
        self,
        from_sq: int | str | tuple[int, int],
        to_sq: int | str | tuple[int, int],
        promotion: str | None = None,
    ) -> MoveInfo | None:
        """Execute a move if legal. Returns MoveInfo on success, None on illegal move.

        If the move requires promotion and promotion is None, defaults to Queen ('q').
        Returns None once the game has ended by resignation or agreement.
        """
        if self._custom_outcome is not None:
            return None
        f = self.parse_square(from_sq)
        t = self.parse_square(to_sq)

        promo_type = self._parse_promotion(promotion)
        if promo_type is None and self.is_promotion_move(f, t):
            promo_type = chess.QUEEN

        move = chess.Move(f, t, promotion=promo_type)
        if not self._board.is_legal(move):
            return None

        return self._execute_move(move)

    def make_uci(self, uci_str: str) -> MoveInfo | None:
        """Execute a move given in UCI string format (e.g. 'e2e4', 'e7e8q')."""
        if self._custom_outcome is not None:
            return None
        try:
            move = chess.Move.from_uci(uci_str.strip())
        except ValueError:
            return None

        if not self._board.is_legal(move):
            return None

        return self._execute_move(move)

    def make_san(self, san_str: str) -> MoveInfo | None:
        """Execute a move given in Standard Algebraic Notation (e.g. 'Nf3', 'O-O')."""
        if self._custom_outcome is not None:
            return None
        try:
            move = self._board.parse_san(san_str.strip())
        except ValueError:
            return None

        return self._execute_move(move)

    def _execute_move(self, move: chess.Move) -> MoveInfo:
        """Internal helper to push a validated move and record metadata."""
        fen_before = self._board.fen()

        moving_piece = self._board.piece_at(move.from_square)
        assert moving_piece is not None, "Internal error: moving piece is None"
        piece_symbol = moving_piece.symbol().upper()
        color_str = "white" if moving_piece.color == chess.WHITE else "black"

        # Check captures and en passant
        is_capture = self._board.is_capture(move)
        is_en_passant = self._board.is_en_passant(move)
        captured_piece_symbol: str | None = None
        captured_sq: int | None = None

        if is_en_passant:
            captured_sq = (
                move.to_square - 8 if moving_piece.color == chess.WHITE else move.to_square + 8
            )
            captured_p = self._board.piece_at(captured_sq)
            captured_piece_symbol = captured_p.symbol().upper() if captured_p else "P"
        elif is_capture:
            captured_sq = move.to_square
            captured_p = self._board.piece_at(captured_sq)
            captured_piece_symbol = captured_p.symbol().upper() if captured_p else None

        # Check castling
        is_castling = self._board.is_castling(move)
        castling_side: str | None = None
        if is_castling:
            castling_side = (
                "kingside" if self._board.is_kingside_castling(move) else "queenside"
            )

        # Check promotion
        is_promo = move.promotion is not None
        promo_symbol = (
            chess.piece_symbol(move.promotion).upper() if move.promotion else None
        )

        # SAN string before pushing
        san = self._board.san(move)
        uci = move.uci()

        # Push the move onto the board
        self._board.push(move)
        fen_after = self._board.fen()

        # Update capture tallies
        if captured_piece_symbol is not None:
            if moving_piece.color == chess.WHITE:
                self._captured_by_white.append(captured_piece_symbol)
            else:
                self._captured_by_black.append(captured_piece_symbol)

        # New move invalidates redo stack
        self._redo_stack.clear()

        info = MoveInfo(
            move=move,
            san=san,
            uci=uci,
            from_square=move.from_square,
            to_square=move.to_square,
            from_square_name=self.square_name(move.from_square),
            to_square_name=self.square_name(move.to_square),
            piece=piece_symbol,
            color=color_str,
            is_capture=is_capture,
            captured_piece=captured_piece_symbol,
            captured_square=captured_sq,
            is_castling=is_castling,
            castling_side=castling_side,
            is_en_passant=is_en_passant,
            is_promotion=is_promo,
            promotion_piece=promo_symbol,
            is_check=self._board.is_check(),
            is_checkmate=self._board.is_checkmate(),
            fen_before=fen_before,
            fen_after=fen_after,
        )
        self._move_history.append(info)
        return info

    # ------------------------------------------------------------------
    # Undo / Redo
    # ------------------------------------------------------------------

    def undo_move(self) -> MoveInfo | None:
        """Undo the last played move. Returns the popped MoveInfo, or None if no moves."""
        if not self._move_history:
            return None

        last_info = self._move_history.pop()
        popped_move = self._board.pop()
        self._redo_stack.append(popped_move)

        # Revert capture list if necessary
        if last_info.captured_piece is not None:
            if last_info.color == "white" and self._captured_by_white:
                self._captured_by_white.pop()
            elif last_info.color == "black" and self._captured_by_black:
                self._captured_by_black.pop()

        return last_info

    def redo_move(self) -> MoveInfo | None:
        """Redo the most recently undone move. Returns the re-executed MoveInfo or None."""
        if not self._redo_stack:
            return None

        move = self._redo_stack.pop()
        # Execute without clearing the remaining redo stack
        saved_redo = list(self._redo_stack)
        info = self._execute_move(move)
        self._redo_stack = saved_redo
        return info

    def reset(self, fen: str = STARTING_FEN) -> None:
        """Reset the game back to the starting position or specified FEN."""
        self._initial_fen = fen
        self._board.set_fen(fen)
        self._move_history.clear()
        self._redo_stack.clear()
        self._captured_by_white.clear()
        self._captured_by_black.clear()
        self._custom_outcome = None

    # ------------------------------------------------------------------
    # Game Termination
    # ------------------------------------------------------------------

    def resign(self, color: str) -> GameOutcome:
        """End the game with *color* ('white' or 'black') resigning.

        Has no effect if the game is already over; returns the current outcome.
        """
        loser = color.strip().lower()
        if loser not in ("white", "black"):
            raise ValueError(f"Invalid color: {color!r}")
        if self.is_game_over:
            return self.outcome
        winner = "black" if loser == "white" else "white"
        self._custom_outcome = GameOutcome(
            is_over=True,
            status="resignation",
            winner=winner,
            termination="Resignation",
            description=f"{loser.capitalize()} resigns — {winner.capitalize()} wins!",
        )
        return self._custom_outcome

    def agree_draw(self, reason: str = "Draw by agreement.") -> GameOutcome:
        """End the game as a draw. Has no effect if the game is already over."""
        if self.is_game_over:
            return self.outcome
        self._custom_outcome = GameOutcome(
            is_over=True,
            status="draw_agreed",
            winner=None,
            termination="Agreement",
            description=reason,
        )
        return self._custom_outcome

    def timeout(self, color: str) -> GameOutcome:
        """End the game with *color* ('white' or 'black') losing on time.

        Has no effect if the game is already over; returns the current outcome.
        """
        loser = color.strip().lower()
        if loser not in ("white", "black"):
            raise ValueError(f"Invalid color: {color!r}")
        if self.is_game_over:
            return self.outcome
        winner = "black" if loser == "white" else "white"
        self._custom_outcome = GameOutcome(
            is_over=True,
            status="timeout",
            winner=winner,
            termination="Time Out",
            description=f"{loser.capitalize()} ran out of time — {winner.capitalize()} wins!",
        )
        return self._custom_outcome

    # ------------------------------------------------------------------
    # Material Accounting
    # ------------------------------------------------------------------

    def material_count(self) -> dict[str, int]:
        """Compute current material value for White and Black (P=1, N/B=3, R=5, Q=9)."""
        white_pts = 0
        black_pts = 0

        for sq in chess.SQUARES:
            piece = self._board.piece_at(sq)
            if piece is None or piece.piece_type == chess.KING:
                continue
            val = PIECE_VALUES.get(piece.symbol().upper(), 0)
            if piece.color == chess.WHITE:
                white_pts += val
            else:
                black_pts += val

        return {"white": white_pts, "black": black_pts}

    def material_diff(self) -> int:
        """Material difference from White's perspective (positive = White leads)."""
        counts = self.material_count()
        return counts["white"] - counts["black"]

    # ------------------------------------------------------------------
    # PGN Export
    # ------------------------------------------------------------------

    def get_pgn(
        self,
        white: str = "Player 1",
        black: str = "Player 2",
        event: str = "Casual Game",
        site: str = "ChessKen",
        round_str: str = "1",
        result: str | None = None,
    ) -> str:
        """Export the played game to standard PGN string."""
        game = chess.pgn.Game()
        game.headers["Event"] = event
        game.headers["Site"] = site
        game.headers["Date"] = datetime.date.today().strftime("%Y.%m.%d")
        game.headers["Round"] = round_str
        game.headers["White"] = white
        game.headers["Black"] = black

        if result is not None:
            game.headers["Result"] = result
        else:
            outcome = self.outcome
            if outcome.is_over:
                if outcome.winner == "white":
                    game.headers["Result"] = "1-0"
                elif outcome.winner == "black":
                    game.headers["Result"] = "0-1"
                else:
                    game.headers["Result"] = "1/2-1/2"
            else:
                game.headers["Result"] = "*"

        node = game
        for info in self._move_history:
            node = node.add_variation(info.move)

        exporter = chess.pgn.StringExporter(headers=True, variations=False, comments=False)
        return game.accept(exporter)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_promotion(promotion: str | int | None) -> int | None:
        """Parse promotion string ('q', 'r', 'b', 'n') or int into chess piece type."""
        if promotion is None:
            return None
        if isinstance(promotion, int):
            if promotion in (chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT):
                return promotion
            return None
        if isinstance(promotion, str):
            mapping = {
                "q": chess.QUEEN,
                "r": chess.ROOK,
                "b": chess.BISHOP,
                "n": chess.KNIGHT,
            }
            return mapping.get(promotion.strip().lower())
        return None


@dataclass(frozen=True)
class TimeControlPreset:
    """Structured time-control preset definition."""

    category: str
    base_seconds: int
    increment_seconds: int
    label: str

    @property
    def display_label(self) -> str:
        """Label shown in the collapsed selector, e.g. '3 + 2 (Blitz)' or 'No Timer'."""
        if self.base_seconds <= 0 or self.category.upper() == "NO TIMER":
            return "No Timer"
        return f"{self.label} ({self.category.title()})"


def format_clock_seconds(seconds: float) -> str:
    """Format a clock reading in seconds as M:SS (or 0:0S.d below 10 seconds)."""
    clamped = max(0.0, float(seconds))
    if clamped <= 0.0:
        return "0:00"
    if clamped < 10.0:
        tenths = int(clamped * 10)
        whole = tenths // 10
        frac = tenths % 10
        return f"0:0{whole}.{frac}"
    total_sec = int(clamped)
    minutes = total_sec // 60
    rem_sec = total_sec % 60
    return f"{minutes}:{rem_sec:02d}"


class MatchClock:
    """Pure, testable dual chess clock with Fischer increment and flag-fall detection."""

    def __init__(self, preset: TimeControlPreset | None = None) -> None:
        self._preset: TimeControlPreset | None = (
            preset if (preset is not None and preset.base_seconds > 0) else None
        )
        base = float(self._preset.base_seconds) if self._preset is not None else 0.0
        self.white_seconds: float = base
        self.black_seconds: float = base
        self.active_color: str = "white"
        self.paused: bool = False
        self.flagged_color: str | None = None

    @property
    def preset(self) -> TimeControlPreset | None:
        """Active structured time-control preset, or None for No Timer."""
        return self._preset

    @property
    def is_timed(self) -> bool:
        """True when a timed preset is active (False for No Timer)."""
        return self._preset is not None and self._preset.base_seconds > 0

    @staticmethod
    def _normalize_color(color: str | chess.Color) -> str:
        if isinstance(color, bool):
            return "white" if color == chess.WHITE else "black"
        c = str(color).strip().lower()
        if c not in ("white", "black"):
            raise ValueError(f"Invalid color: {color!r}")
        return c

    def remaining(self, color: str | chess.Color) -> float:
        """Return remaining seconds for *color* ('white' or 'black')."""
        norm = self._normalize_color(color)
        return self.white_seconds if norm == "white" else self.black_seconds

    def format_time(self, color: str | chess.Color) -> str:
        """Return formatted clock string for *color*."""
        return format_clock_seconds(self.remaining(color))

    def tick(
        self, dt: float, active_color: str | chess.Color | None = None
    ) -> str | None:
        """Advance the active side's clock by *dt* seconds.

        Returns the flagged color ('white' or 'black') if a flag-fall occurs,
        otherwise None.
        """
        if not self.is_timed or self.paused or self.flagged_color is not None:
            return self.flagged_color
        if active_color is not None:
            self.active_color = self._normalize_color(active_color)
        elapsed = max(0.0, float(dt))
        if elapsed <= 0.0:
            return None

        if self.active_color == "white":
            self.white_seconds = max(0.0, self.white_seconds - elapsed)
            if self.white_seconds <= 0.0:
                self.white_seconds = 0.0
                self.flagged_color = "white"
                self.paused = True
        else:
            self.black_seconds = max(0.0, self.black_seconds - elapsed)
            if self.black_seconds <= 0.0:
                self.black_seconds = 0.0
                self.flagged_color = "black"
                self.paused = True
        return self.flagged_color

    def apply_increment(self, moved_color: str | chess.Color) -> None:
        """Apply Fischer increment to *moved_color* after a move and switch active side."""
        if not self.is_timed or self.paused or self.flagged_color is not None:
            return
        assert self._preset is not None
        norm = self._normalize_color(moved_color)
        inc = float(max(0, self._preset.increment_seconds))
        if norm == "white":
            self.white_seconds += inc
            self.active_color = "black"
        else:
            self.black_seconds += inc
            self.active_color = "white"

    def pause(self) -> None:
        """Pause the clock (e.g. when the game ends)."""
        self.paused = True

    def resume(self) -> None:
        """Resume the clock if not flagged."""
        if self.flagged_color is None:
            self.paused = False

    def reset(self, preset: TimeControlPreset | None = None) -> None:
        """Reset both clocks to the base time of *preset* (or current preset)."""
        if preset is not None:
            self._preset = preset if preset.base_seconds > 0 else None
        base = float(self._preset.base_seconds) if self._preset is not None else 0.0
        self.white_seconds = base
        self.black_seconds = base
        self.active_color = "white"
        self.paused = False
        self.flagged_color = None

