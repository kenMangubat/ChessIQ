"""Animated chessboard: opening-demo backdrop and interactive game board.

Demo mode (default) renders a Lichess-inspired green board that replays the
opening LINES with smooth piece movement and fade transitions.

Interactive mode (after attach_game()) renders a live ChessGame position with
selection, legal-target markers, last-move / check highlights, distance-based
eased move animation with queue support, capture fade-out, an optional flipped
orientation, resize support, and a pawn-promotion selector.
All timing uses delta-time; no blocking calls.
"""

from __future__ import annotations

from collections import deque
import io
import math
from pathlib import Path
from xml.dom import minidom

import chess
import pygame

from chess_logic.chess_game import ChessGame, MoveInfo
from others.config import ASSETS_DIR
from ui.chess_rules import Position, START_FEN, LINES, PIECE_NAMES


# ── Centralized animation & bot-pacing constants ────────────────────
# Animation timing (seconds)
ANIMATION_BASE_DURATION: float = 0.17
ANIMATION_DURATION_PER_SQUARE: float = 0.022
ANIMATION_MAX_DURATION: float = 0.32
ANIMATION_KNIGHT_DISTANCE: float = 2.0  # crisp medium distance for knight hops
PREMOVE_ANIMATION_DURATION: float = 0.13  # snappy, smooth glide for premoves

# Frame timing clamp (prevents pieces from teleporting after a frame hitch)
MAX_ANIMATION_DT: float = 1.0 / 20.0

# Easing curve ("ease_out_cubic" | "ease_in_out_cubic")
ANIMATION_EASING: str = "ease_in_out_cubic"

# Subtle lift while traveling (pre-cached scaled surfaces; no per-frame rescaling)
ANIMATION_SCALE_START: float = 1.00
ANIMATION_SCALE_PEAK: float = 1.03

# Captured piece fade-out after moving piece lands (seconds)
CAPTURE_FADE_DURATION: float = 0.10

# Bot pacing: minimum total perceived thinking time ranges (seconds)
# Measured AFTER the player's piece finishes sliding so bot replies never feel instant.
# Sick Ken: faster and more impulsive
SICK_BOOK_THINK_RANGE: tuple[float, float] = (0.55, 0.90)
SICK_ENGINE_THINK_RANGE: tuple[float, float] = (0.75, 1.20)

# Focus Ken: balanced thinking
FOCUS_BOOK_THINK_RANGE: tuple[float, float] = (0.65, 1.05)
FOCUS_ENGINE_THINK_RANGE: tuple[float, float] = (0.95, 1.45)

# Condition Ken: slower and more deliberate
CONDITION_BOOK_THINK_RANGE: tuple[float, float] = (0.85, 1.35)
CONDITION_ENGINE_THINK_RANGE: tuple[float, float] = (1.20, 1.95)

BOT_THINK_RANGES: dict[str, dict[str, tuple[float, float]]] = {
    "sick": {"book": SICK_BOOK_THINK_RANGE, "engine": SICK_ENGINE_THINK_RANGE},
    "focus": {"book": FOCUS_BOOK_THINK_RANGE, "engine": FOCUS_ENGINE_THINK_RANGE},
    "condition": {"book": CONDITION_BOOK_THINK_RANGE, "engine": CONDITION_ENGINE_THINK_RANGE},
}

# Dashboard demo timing (seconds)
_MOVE_DURATION = 0.55             # demo piece slide
_PAUSE_BETWEEN_MOVES = 0.70       # dwell after a demo move lands
_PAUSE_BETWEEN_OPENINGS = 2.0     # dwell after final move of a demo opening
_FADE_DURATION = 1.0              # cross-fade between demo openings


# ── Pure animation & pacing math helpers ────────────────────────────


def ease_progress(t: float, mode: str = ANIMATION_EASING) -> float:
    """Map normalized linear progress *t* in [0, 1] to eased progress in [0, 1]."""
    clamped = 0.0 if t <= 0.0 else (1.0 if t >= 1.0 else t)
    if mode == "ease_in_out_cubic":
        if clamped < 0.5:
            return 4.0 * clamped * clamped * clamped
        p = 2.0 * clamped - 2.0
        return 0.5 * p * p * p + 1.0
    # Default: cubic ease-out (fast, responsive start; smooth deceleration into square)
    inv = 1.0 - clamped
    return 1.0 - inv * inv * inv


def move_distance_in_squares(
    from_sq: int,
    to_sq: int,
    piece_symbol: str = "",
) -> float:
    """Return effective board distance in squares for timing a move."""
    if piece_symbol.upper() == "N":
        return ANIMATION_KNIGHT_DISTANCE
    df = chess.square_file(to_sq) - chess.square_file(from_sq)
    dr = chess.square_rank(to_sq) - chess.square_rank(from_sq)
    return math.hypot(df, dr)


def compute_move_duration(
    from_sq: int,
    to_sq: int,
    piece_symbol: str = "",
) -> float:
    """Compute distance-based animation duration capped at ANIMATION_MAX_DURATION."""
    dist = move_distance_in_squares(from_sq, to_sq, piece_symbol)
    raw = ANIMATION_BASE_DURATION + dist * ANIMATION_DURATION_PER_SQUARE
    return min(raw, ANIMATION_MAX_DURATION)


def get_bot_think_range(tier: str, is_book: bool) -> tuple[float, float]:
    """Return (min_seconds, max_seconds) perceived thinking range for a bot tier."""
    tier_ranges = BOT_THINK_RANGES.get(tier.strip().lower(), BOT_THINK_RANGES["focus"])
    return tier_ranges["book" if is_book else "engine"]


def _convert_surface(surf: pygame.Surface, alpha: bool = True) -> pygame.Surface:
    """Convert surface to active display pixel format when a display exists."""
    if pygame.display.get_surface() is not None:
        try:
            return surf.convert_alpha() if alpha else surf.convert()
        except pygame.error:
            return surf
    return surf


def _load_svg_at_size(svg_path: Path, size: int) -> pygame.Surface:
    document = minidom.parse(str(svg_path))
    root = document.documentElement
    root.setAttribute("width", f"{size}px")
    root.setAttribute("height", f"{size}px")
    svg_data = document.toxml(encoding="utf-8")
    return _convert_surface(pygame.image.load(io.BytesIO(svg_data), str(svg_path)), alpha=True)


# ── board themes ────────────────────────────────────────────────────
# Default design is Lichess green.
# Colours sampled pixel-perfect from the reference screenshot:
#   Light squares: #ffffdd  Dark squares: #86a666
#   Border:        #211e1a  Coord-on-dark: #e4f0c6  Coord-on-light: #9ca989
BOARD_THEMES: dict[str, dict[str, tuple[int, ...]]] = {
    "lichess_green": {
        "light_sq": (255, 255, 221),       # #ffffdd
        "dark_sq": (134, 166, 102),        # #86a666
        "border": (33, 30, 26),            # #211e1a
        "highlight": (205, 210, 106, 160), # lichess last-move yellow-green
        "highlight_border": (170, 162, 58, 200),
        "coord_on_light": (156, 169, 137), # #9ca989 (muted green on light sq)
        "coord_on_dark": (228, 240, 198),  # #e4f0c6 (pale cream on dark sq)
    },
    "wood": {
        "light_sq": (237, 217, 189),       # #edd9bd
        "dark_sq": (174, 139, 112),        # #ae8b70
        "border": (18, 18, 26),            # #12121a
        "highlight": (114, 182, 93, 115),
        "highlight_border": (114, 182, 93, 220),
        "coord_on_light": (174, 139, 112),
        "coord_on_dark": (237, 217, 189),
    },
    "green": {
        "light_sq": (238, 238, 210),       # #eeeed2
        "dark_sq": (118, 150, 86),         # #769656
        "border": (18, 18, 26),
        "highlight": (186, 202, 68, 115),
        "highlight_border": (186, 202, 68, 220),
        "coord_on_light": (118, 150, 86),
        "coord_on_dark": (238, 238, 210),
    },
}
DEFAULT_THEME = "lichess_green"

_BG = (23, 23, 23)                # #171717

# ── SVG piece asset mapping ─────────────────────────────────────────
_PIECE_SVG_MAP: dict[str, str] = {
    "K": "wK.svg", "Q": "wQ.svg", "R": "wR.svg",
    "B": "wB.svg", "N": "wN.svg", "P": "wP.svg",
    "k": "bK.svg", "q": "bQ.svg", "r": "bR.svg",
    "b": "bB.svg", "n": "bN.svg", "p": "bP.svg",
}

# ── piece unicode glyphs (fallback if SVG cannot be loaded) ─────────
_WHITE_GLYPHS = {"K": "♔", "Q": "♕", "R": "♖", "B": "♗", "N": "♘", "P": "♙"}
_BLACK_GLYPHS = {"K": "♚", "Q": "♛", "R": "♜", "B": "♝", "N": "♞", "P": "♟"}

# ── interactive overlay colours ─────────────────────────────────────
_SELECT_COLOR = (20, 85, 30, 120)
_MARKER_COLOR = (20, 85, 30, 110)
_PREMOVE_COLOR = (224, 72, 64, 150)
_PREMOVE_DOT_COLOR = (224, 72, 64, 125)
_CHECK_COLOR = (230, 30, 30)
_PROMO_DIM = (0, 0, 0, 140)
_PROMO_TILE = (240, 240, 240)
_PROMO_TILE_BORDER = (60, 60, 60)
_PROMOTION_CHOICES: tuple[str, ...] = ("q", "r", "b", "n")


def premove_targets_from(
    board: chess.Board, square: int, player_color: chess.Color
) -> list[int]:
    """Return candidate destination squares for a premove from *square*.

    Follows standard Chess.com / Lichess premove geometry: rays ignore current
    blockers, pawns may premove diagonally even onto empty/friendly squares,
    and kings may premove castling when castling rights remain.
    """
    piece = board.piece_at(square)
    if piece is None or piece.color != player_color:
        return []

    targets: set[int] = set()
    ptype = piece.piece_type
    file_idx = chess.square_file(square)
    rank_idx = chess.square_rank(square)

    if ptype == chess.PAWN:
        step = 1 if player_color == chess.WHITE else -1
        start_rank = 1 if player_color == chess.WHITE else 6
        next_rank = rank_idx + step
        if 0 <= next_rank <= 7:
            targets.add(chess.square(file_idx, next_rank))
            if file_idx > 0:
                targets.add(chess.square(file_idx - 1, next_rank))
            if file_idx < 7:
                targets.add(chess.square(file_idx + 1, next_rank))
        if rank_idx == start_rank:
            two_rank = rank_idx + 2 * step
            if 0 <= two_rank <= 7:
                targets.add(chess.square(file_idx, two_rank))
    elif ptype == chess.KNIGHT:
        targets.update(chess.SquareSet(chess.BB_KNIGHT_ATTACKS[square]))
    elif ptype == chess.BISHOP:
        targets.update(chess.SquareSet(chess.BB_DIAG_ATTACKS[square][0]))
    elif ptype == chess.ROOK:
        targets.update(
            chess.SquareSet(
                chess.BB_RANK_ATTACKS[square][0] | chess.BB_FILE_ATTACKS[square][0]
            )
        )
    elif ptype == chess.QUEEN:
        targets.update(
            chess.SquareSet(
                chess.BB_DIAG_ATTACKS[square][0]
                | chess.BB_RANK_ATTACKS[square][0]
                | chess.BB_FILE_ATTACKS[square][0]
            )
        )
    elif ptype == chess.KING:
        targets.update(chess.SquareSet(chess.BB_KING_ATTACKS[square]))
        if player_color == chess.WHITE and square == chess.E1:
            if board.has_kingside_castling_rights(chess.WHITE):
                targets.add(chess.G1)
            if board.has_queenside_castling_rights(chess.WHITE):
                targets.add(chess.C1)
        elif player_color == chess.BLACK and square == chess.E8:
            if board.has_kingside_castling_rights(chess.BLACK):
                targets.add(chess.G8)
            if board.has_queenside_castling_rights(chess.BLACK):
                targets.add(chess.C8)

    targets.discard(square)
    return sorted(targets)


class AnimatedBoard:
    """Self-contained animated chessboard that draws onto a parent surface.

    Parameters
    ----------
    width, height : int
        Size of the area available for drawing.
    scale : float
        UI scale factor from the dashboard or game screen.
    """

    def __init__(
        self,
        width: int,
        height: int,
        scale: float,
        theme: str = DEFAULT_THEME,
        square_scale: float = 0.115,
    ) -> None:
        self._w = width
        self._h = height
        self._scale = scale
        self._square_scale = square_scale
        self._theme_name = theme if theme in BOARD_THEMES else DEFAULT_THEME
        self._theme = BOARD_THEMES[self._theme_name]

        # orientation: False = White at the bottom, True = Black at the bottom
        self._flipped = False

        # pre-rendered caches (populated in _rebuild_geometry)
        self._piece_cache: dict[str, pygame.Surface] = {}
        self._piece_lift_mid_cache: dict[str, pygame.Surface] = {}
        self._piece_lift_peak_cache: dict[str, pygame.Surface] = {}
        self._fade_old_surf: pygame.Surface | None = None
        self._fade_new_surf: pygame.Surface | None = None
        self._rebuild_geometry(width, height, scale)

        # ── demo animation state ─────────────────────────────────────
        self._line_idx = 0
        self._move_idx = 0
        self._position = Position(LINES[0].get("fen", START_FEN))

        # phase: "move", "pause", "opening_pause", "fade"
        self._phase = "pause"
        self._timer: float = 0.0
        self._phase_dur: float = 0.8  # initial pause before first move

        # demo move animation temps
        self._anim_piece: str = ""
        self._anim_from: tuple[int, int] = (0, 0)
        self._anim_to: tuple[int, int] = (0, 0)
        self._anim_rook_piece: str = ""
        self._anim_rook_from: tuple[int, int] = (0, 0)
        self._anim_rook_to: tuple[int, int] = (0, 0)
        self._anim_castle: bool = False
        self._last_to: tuple[int, int] | None = None

        # demo fade state
        self._fade_alpha: float = 0.0
        self._next_position: Position | None = None

        # ── interactive state (active only after attach_game) ────────
        self._game: ChessGame | None = None
        self._selected: int | None = None
        self._legal_targets: list[int] = []
        self._capture_targets: set[int] = set()
        self._promotion: tuple[int, int] | None = None
        self._hint_squares: tuple[int, int] | None = None
        self._premove_selected: int | None = None
        self._premove_targets: list[int] = []
        self._premove: tuple[int, int, str | None] | None = None
        # Smooth click-and-drag state (supports both live moves and premoves)
        self._drag_origin: int | None = None
        self._drag_start_pos: tuple[int, int] = (0, 0)
        self._drag_pos: tuple[int, int] = (0, 0)
        self._drag_active: bool = False
        self._drag_is_premove: bool = False
        self._move_anim: MoveInfo | None = None
        self._move_anim_base_board: chess.Board | None = None
        self._move_anim_elapsed: float = 0.0
        self._move_anim_duration: float = ANIMATION_BASE_DURATION
        self._anim_queue: deque[tuple[MoveInfo, float | None]] = deque()
        # Cosmetic captured-piece fade: (symbol, square, elapsed, cached_surface)
        self._capture_fade: tuple[str, int, float, pygame.Surface] | None = None

    def _rebuild_geometry(self, width: int, height: int, scale: float) -> None:
        """Compute board layout and build cached board, piece, and overlay surfaces."""
        self._w = width
        self._h = height
        self._scale = scale
        self._sq = max(40, int(min(width, height) * self._square_scale))
        board_px = self._sq * 8
        self._ox = (width - board_px) // 2
        self._oy = (height - board_px) // 2
        self._border_w = max(3, int(self._sq * 0.06))

        coord_size = max(10, int(self._sq * 0.18))
        self._coord_font = pygame.font.SysFont("Segoe UI", coord_size, bold=True)

        piece_size = max(24, int(self._sq * 0.78))
        self._piece_font = pygame.font.SysFont("Segoe UI Symbol", piece_size)

        self._board_surf = self._render_empty_board()
        self._piece_cache.clear()
        self._piece_lift_mid_cache.clear()
        self._piece_lift_peak_cache.clear()
        self._build_piece_cache()

        self._hl_surf = _convert_surface(
            pygame.Surface((self._sq, self._sq), pygame.SRCALPHA), alpha=True
        )
        self._update_highlight_surf()
        self._build_marker_surfaces()
        self._fade_old_surf = None
        self._fade_new_surf = None

    def resize(self, width: int, height: int, scale: float) -> None:
        """Resize board surfaces safely without interrupting active animations."""
        if width <= 0 or height <= 0:
            return
        self._rebuild_geometry(width, height, scale)
        if self._capture_fade is not None:
            sym, sq, elapsed, _ = self._capture_fade
            self._capture_fade = (sym, sq, elapsed, self._get_piece_surface(sym).copy())

    # ------------------------------------------------------------------
    # Theme Management
    # ------------------------------------------------------------------

    @property
    def board_rect(self) -> pygame.Rect:
        """Outer bounding rectangle of the 8x8 board including its border."""
        board_px = self._sq * 8
        return pygame.Rect(
            self._ox - self._border_w,
            self._oy - self._border_w,
            board_px + self._border_w * 2,
            board_px + self._border_w * 2,
        )

    @property
    def theme(self) -> str:
        """Name of the active board theme."""
        return self._theme_name

    def set_theme(self, theme_name: str) -> None:
        """Switch board theme at runtime and re-render board cache."""
        if theme_name in BOARD_THEMES and theme_name != self._theme_name:
            self._theme_name = theme_name
            self._theme = BOARD_THEMES[theme_name]
            self._board_surf = self._render_empty_board()
            self._update_highlight_surf()

    def _update_highlight_surf(self) -> None:
        """Re-render move highlight tile matching theme accent."""
        self._hl_surf.fill((0, 0, 0, 0))
        self._hl_surf.fill(self._theme["highlight"])
        pygame.draw.rect(
            self._hl_surf,
            self._theme["highlight_border"],
            (0, 0, self._sq, self._sq),
            width=max(2, int(self._sq * 0.04)),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, dt: float) -> None:
        """Advance the animation by *dt* seconds (clamped against frame hitches)."""
        clamped_dt = min(max(0.0, dt), MAX_ANIMATION_DT)
        if self._game is not None:
            self._update_move_animation(clamped_dt)
            return

        self._timer += clamped_dt

        if self._phase == "pause":
            if self._timer >= self._phase_dur:
                self._start_next_move()

        elif self._phase == "move":
            if self._timer >= self._phase_dur:
                self._finish_move()

        elif self._phase == "opening_pause":
            if self._timer >= self._phase_dur:
                self._start_fade()

        elif self._phase == "fade":
            self._fade_alpha = min(self._timer / _FADE_DURATION, 1.0)
            if self._fade_alpha >= 1.0:
                self._finish_fade()

    def draw(self, surface: pygame.Surface) -> None:
        """Draw the board and pieces onto *surface*."""
        if self._game is not None:
            self._draw_interactive(surface)
            return

        rect = self.board_rect
        if self._phase == "fade" and self._next_position is not None:
            if self._fade_old_surf is None or self._fade_old_surf.get_size() != rect.size:
                self._fade_old_surf = _convert_surface(pygame.Surface(rect.size), alpha=False)
                self._fade_new_surf = _convert_surface(pygame.Surface(rect.size), alpha=False)
            assert self._fade_old_surf is not None and self._fade_new_surf is not None
            self._render_demo_into(self._fade_old_surf, self._position, None, rect.topleft)
            self._render_demo_into(self._fade_new_surf, self._next_position, None, rect.topleft)
            self._fade_old_surf.set_alpha(int(255 * (1.0 - self._fade_alpha)))
            surface.blit(self._fade_old_surf, rect.topleft)
            self._fade_new_surf.set_alpha(int(255 * self._fade_alpha))
            surface.blit(self._fade_new_surf, rect.topleft)
        elif self._phase == "move":
            t = min(self._timer / max(self._phase_dur, 1e-6), 1.0)
            self._draw_with_animation(surface, t)
        else:
            surface.blit(self._board_surf, rect.topleft)
            if self._last_to is not None:
                hf, hr = self._last_to
                surface.blit(self._hl_surf, self._file_rank_to_pixel(hf, hr))
            for rank in range(8):
                for file in range(8):
                    piece = self._position.piece_at(file, rank)
                    if piece == ".":
                        continue
                    self._blit_piece_centered(
                        surface, piece, self._file_rank_center(file, rank)
                    )

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _render_empty_board(self) -> pygame.Surface:
        """Render the cached opaque empty board tightly cropped to `board_rect`.

        Layout preserves the exact Lichess green reference design:
        - Solid dark border around the 8x8 grid, sharp corners, flat design
        - File letters (a-h) bottom-left on the visually bottom rank
        - Rank numbers (8-1) top-right on the visually right file
        """
        rect = self.board_rect
        surf = pygame.Surface(rect.size)
        surf.fill(self._theme["border"])

        light_color = self._theme["light_sq"]
        dark_color = self._theme["dark_sq"]
        label_file = 0 if self._flipped else 7
        label_rank = 7 if self._flipped else 0

        for rank in range(8):
            for file in range(8):
                gx, gy = self._file_rank_to_pixel(file, rank)
                x = gx - rect.left
                y = gy - rect.top
                light = (file + rank) % 2 != 0
                color = light_color if light else dark_color
                pygame.draw.rect(surf, color, (x, y, self._sq, self._sq))

                if file == label_file:
                    coord_color = (
                        self._theme["coord_on_light"] if light
                        else self._theme["coord_on_dark"]
                    )
                    lbl = self._coord_font.render(str(rank + 1), True, coord_color)
                    cx = x + self._sq - lbl.get_width() - max(2, int(self._sq * 0.05))
                    cy = y + max(1, int(self._sq * 0.04))
                    surf.blit(lbl, (cx, cy))

                if rank == label_rank:
                    coord_color = (
                        self._theme["coord_on_light"] if light
                        else self._theme["coord_on_dark"]
                    )
                    lbl = self._coord_font.render(
                        chr(ord("a") + file), True, coord_color
                    )
                    cx = x + max(2, int(self._sq * 0.05))
                    cy = y + self._sq - lbl.get_height() - max(1, int(self._sq * 0.03))
                    surf.blit(lbl, (cx, cy))

        return _convert_surface(surf, alpha=False)

    def _render_demo_into(
        self,
        target: pygame.Surface,
        pos: Position,
        highlight: tuple[int, int] | None,
        offset: tuple[int, int],
    ) -> None:
        """Render a demo position into a board_rect-sized target surface."""
        ox, oy = offset
        target.blit(self._board_surf, (0, 0))
        if highlight is not None:
            hx, hy = self._file_rank_to_pixel(*highlight)
            target.blit(self._hl_surf, (hx - ox, hy - oy))
        for rank in range(8):
            for file in range(8):
                piece = pos.piece_at(file, rank)
                if piece == ".":
                    continue
                cx, cy = self._file_rank_center(file, rank)
                self._blit_piece_centered(target, piece, (cx - ox, cy - oy))

    def _draw_with_animation(self, surface: pygame.Surface, t: float) -> None:
        """Draw the demo board with one piece smoothly sliding (zero surface copies)."""
        surface.blit(self._board_surf, self.board_rect.topleft)
        surface.blit(self._hl_surf, self._file_rank_to_pixel(*self._anim_to))

        for rank in range(8):
            for file in range(8):
                piece = self._position.piece_at(file, rank)
                if piece == ".":
                    continue
                if (file, rank) == self._anim_from:
                    continue
                if self._anim_castle and (file, rank) == self._anim_rook_from:
                    continue
                self._blit_piece_centered(surface, piece, self._file_rank_center(file, rank))

        s = ease_progress(t, "ease_in_out_cubic")

        fx, fy = self._file_rank_center(*self._anim_from)
        tx, ty = self._file_rank_center(*self._anim_to)
        self._blit_piece_centered(
            surface, self._anim_piece, (fx + (tx - fx) * s, fy + (ty - fy) * s)
        )

        if self._anim_castle:
            rfx, rfy = self._file_rank_center(*self._anim_rook_from)
            rtx, rty = self._file_rank_center(*self._anim_rook_to)
            self._blit_piece_centered(
                surface,
                self._anim_rook_piece,
                (rfx + (rtx - rfx) * s, rfy + (rty - rfy) * s),
            )

    # ------------------------------------------------------------------
    # Orientation mapping (single source of truth for square <-> pixel)
    # ------------------------------------------------------------------

    @property
    def flipped(self) -> bool:
        """True when the board is drawn from Black's side (rank 1 at the top)."""
        return self._flipped

    def set_flipped(self, flipped: bool) -> None:
        """Change orientation and re-render the cached empty board."""
        if flipped != self._flipped:
            self._flipped = flipped
            self._board_surf = self._render_empty_board()

    def _file_rank_to_pixel(self, file: int, rank: int) -> tuple[int, int]:
        """Top-left pixel of (file, rank) on the board surface, honouring orientation."""
        col = 7 - file if self._flipped else file
        row = rank if self._flipped else 7 - rank
        return self._ox + col * self._sq, self._oy + row * self._sq

    def _file_rank_center(self, file: int, rank: int) -> tuple[float, float]:
        x, y = self._file_rank_to_pixel(file, rank)
        half = self._sq * 0.5
        return x + half, y + half

    def square_to_pixel(self, square: int) -> tuple[int, int]:
        """Top-left pixel (board-surface coordinates) of a chess square 0..63."""
        if not 0 <= square < 64:
            raise ValueError(f"Square out of range: {square}")
        return self._file_rank_to_pixel(chess.square_file(square), chess.square_rank(square))

    def _square_center(self, square: int) -> tuple[float, float]:
        x, y = self.square_to_pixel(square)
        half = self._sq * 0.5
        return x + half, y + half

    def pixel_to_square(self, x: int, y: int) -> int | None:
        """Chess square under board-surface pixel (x, y), or None if off the grid."""
        dx = x - self._ox
        dy = y - self._oy
        if dx < 0 or dy < 0:
            return None
        col = dx // self._sq
        row = dy // self._sq
        if col > 7 or row > 7:
            return None
        file = 7 - col if self._flipped else col
        rank = row if self._flipped else 7 - row
        return chess.square(file, rank)

    # ------------------------------------------------------------------
    # Interactive mode — state & input
    # ------------------------------------------------------------------

    def attach_game(self, game: ChessGame, flipped: bool = False) -> None:
        """Switch to interactive mode, rendering *game* in the given orientation."""
        self._game = game
        self.set_flipped(flipped)
        self.clear_interaction()

    @property
    def is_animating(self) -> bool:
        """True while an interactive move slide (or queued slides) is in progress."""
        return self._move_anim is not None or bool(self._anim_queue)

    @property
    def animation_progress(self) -> float:
        """Normalized [0.0, 1.0] progress of the active interactive move animation."""
        if self._move_anim is None or self._move_anim_duration <= 0.0:
            return 0.0
        return min(1.0, max(0.0, self._move_anim_elapsed / self._move_anim_duration))

    @property
    def animation_queue_length(self) -> int:
        """Number of moves currently waiting behind the active animation."""
        return len(self._anim_queue)

    @property
    def selected_square(self) -> int | None:
        return self._selected

    @property
    def legal_targets(self) -> list[int]:
        return list(self._legal_targets)

    @property
    def promotion_pending(self) -> bool:
        """True while the promotion selector overlay is open."""
        return self._promotion is not None

    @property
    def hint_squares(self) -> tuple[int, int] | None:
        """Currently highlighted hint squares (from_sq, to_sq), or None."""
        return self._hint_squares

    @property
    def premove(self) -> tuple[int, int, str | None] | None:
        """Currently queued premove `(from_sq, to_sq, promotion)`, or None."""
        return self._premove

    @property
    def premove_selected(self) -> int | None:
        """Currently selected origin square for premove planning, or None."""
        return self._premove_selected

    @property
    def is_dragging(self) -> bool:
        """True while the user is actively dragging a piece across the board."""
        return self._drag_active and self._drag_origin is not None

    def cancel_drag(self) -> None:
        """Abort any in-progress piece drag cleanly."""
        self._drag_origin = None
        self._drag_active = False
        self._drag_is_premove = False

    def clear_premove(self) -> None:
        """Cancel any queued premove and premove piece selection."""
        self._premove = None
        self._premove_selected = None
        self._premove_targets = []
        if self._drag_is_premove:
            self.cancel_drag()

    def pop_premove(self) -> tuple[int, int, str | None] | None:
        """Consume and return the queued premove, clearing premove overlays."""
        queued = self._premove
        self.clear_premove()
        return queued

    def promote_premove_selection(self, player_color: chess.Color) -> None:
        """Convert an in-progress premove piece selection into a live selection on player's turn."""
        if self._game is None or self._premove is not None or self._premove_selected is None:
            return
        sq = self._premove_selected
        self._premove_selected = None
        self._premove_targets = []
        piece = self._game.piece_at(sq)
        if piece is not None and piece.color == player_color:
            self.select_square(sq)
            if self._drag_origin == sq:
                self._drag_is_premove = False
        else:
            if self._drag_origin == sq:
                self.cancel_drag()

    def set_hint(self, squares: tuple[int, int] | None) -> None:
        """Set or clear the (from_sq, to_sq) hint highlight."""
        self._hint_squares = squares

    def select_square(self, square: int) -> None:
        """Select a square and cache its legal targets from the attached game."""
        if self._game is None:
            return
        self._hint_squares = None
        self._selected = square
        self._legal_targets = self._game.legal_moves_from(square)
        self._capture_targets = {
            target for target in self._legal_targets
            if self._game.is_capture_move(square, target)
        }

    def clear_selection(self) -> None:
        """Deselect and close the promotion overlay."""
        self._selected = None
        self._legal_targets = []
        self._capture_targets = set()
        self._promotion = None
        self._hint_squares = None

    def clear_interaction(self) -> None:
        """Reset selection, premove, drag, promotion overlay, and animations."""
        self.clear_selection()
        self.clear_premove()
        self.cancel_drag()
        self._move_anim = None
        self._move_anim_base_board = None
        self._move_anim_elapsed = 0.0
        self._move_anim_duration = ANIMATION_BASE_DURATION
        self._anim_queue.clear()
        self._capture_fade = None

    def cancel_promotion(self) -> None:
        """Close the promotion overlay, keeping the pawn selected."""
        self._promotion = None

    def animate_move(
        self, info: MoveInfo, duration_override: float | None = None
    ) -> None:
        """Slide a move smoothly (or queue it if another animation is currently active).

        Human and bot moves both enter this exact same pipeline.
        """
        self.clear_selection()
        self._hint_squares = None
        if self._move_anim is not None:
            self._anim_queue.append((info, duration_override))
            return
        self._begin_move_animation(info, duration_override=duration_override)

    def _begin_move_animation(
        self, info: MoveInfo, duration_override: float | None = None
    ) -> None:
        """Initialize timing and base visual state for a single MoveInfo slide."""
        self._move_anim = info
        self._move_anim_elapsed = 0.0
        if duration_override is not None:
            self._move_anim_duration = max(0.06, duration_override)
        else:
            self._move_anim_duration = compute_move_duration(
                info.from_square, info.to_square, info.piece
            )
        # Snapshot the pre-move board so queued/future logical moves never leak early
        if info.fen_before:
            self._move_anim_base_board = chess.Board(info.fen_before)
        else:
            self._move_anim_base_board = None

    def promotion_rects(self) -> list[tuple[str, pygame.Rect]]:
        """Overlay tiles (choice, rect) stacked from the promotion square inwards."""
        if self._promotion is None:
            return []
        x, y = self.square_to_pixel(self._promotion[1])
        step = self._sq if y < self._oy + 4 * self._sq else -self._sq
        return [
            (choice, pygame.Rect(x, y + index * step, self._sq, self._sq))
            for index, choice in enumerate(_PROMOTION_CHOICES)
        ]

    def promotion_choice_at(self, x: int, y: int) -> str | None:
        """Promotion piece ('q', 'r', 'b', 'n') under (x, y), or None."""
        for choice, rect in self.promotion_rects():
            if rect.collidepoint(x, y):
                return choice
        return None

    def handle_click(
        self, x: int, y: int, player_color: chess.Color
    ) -> tuple[int, int, str | None] | None:
        """Process a left click at board-surface (x, y).

        Returns (from_square, to_square, promotion) when a move is complete,
        otherwise None. The caller is responsible for turn / game-over gating.
        """
        if self._game is None or self.is_animating:
            return None

        if self._promotion is not None:
            choice = self.promotion_choice_at(x, y)
            if choice is None:
                self.cancel_promotion()
                return None
            from_sq, to_sq = self._promotion
            self.clear_selection()
            return from_sq, to_sq, choice

        square = self.pixel_to_square(x, y)
        if square is None:
            self.clear_selection()
            return None

        if self._selected is not None and square in self._legal_targets:
            from_sq = self._selected
            if self._game.is_promotion_move(from_sq, square):
                self._promotion = (from_sq, square)
                return None
            self.clear_selection()
            return from_sq, square, None

        piece = self._game.piece_at(square)
        if piece is not None and piece.color == player_color and square != self._selected:
            self.select_square(square)
        else:
            self.clear_selection()
        return None

    def handle_premove_click(
        self, x: int, y: int, player_color: chess.Color
    ) -> tuple[int, int, str | None] | None:
        """Process a left click during the opponent's turn to queue or cancel a premove.

        Returns `(from_square, to_square, promotion)` when a premove is queued,
        otherwise None.
        """
        if self._game is None:
            return None

        square = self.pixel_to_square(x, y)
        if square is None:
            self.clear_premove()
            return None

        if self._premove_selected is not None and square in self._premove_targets:
            from_sq = self._premove_selected
            piece = self._game.piece_at(from_sq)
            promo: str | None = None
            if piece is not None and piece.piece_type == chess.PAWN:
                promo_rank = 7 if player_color == chess.WHITE else 0
                if chess.square_rank(square) == promo_rank:
                    promo = "q"
            self._premove = (from_sq, square, promo)
            self._premove_selected = None
            self._premove_targets = []
            return self._premove

        piece = self._game.piece_at(square)
        if (
            piece is not None
            and piece.color == player_color
            and square != self._premove_selected
        ):
            self._premove_selected = square
            self._premove_targets = premove_targets_from(
                self._game.board, square, player_color
            )
            return None

        self.clear_premove()
        return None

    def begin_drag(
        self, x: int, y: int, player_color: chess.Color, is_premove: bool
    ) -> None:
        """Track a potential drag starting on a player's piece (activates after 6px move)."""
        self.cancel_drag()
        if self._game is None or self._promotion is not None:
            return
        square = self.pixel_to_square(x, y)
        if square is None:
            return
        piece = self._game.piece_at(square)
        if piece is not None and piece.color == player_color:
            self._drag_origin = square
            self._drag_start_pos = (x, y)
            self._drag_pos = (x, y)
            self._drag_active = False
            self._drag_is_premove = is_premove

    def update_drag(self, x: int, y: int) -> None:
        """Update cursor position during a press; activates smooth drag once moved >= 6px."""
        if self._drag_origin is None:
            return
        self._drag_pos = (x, y)
        if not self._drag_active:
            dx = x - self._drag_start_pos[0]
            dy = y - self._drag_start_pos[1]
            if dx * dx + dy * dy >= 36:
                self._drag_active = True

    def end_drag(
        self,
        x: int,
        y: int,
        player_color: chess.Color,
        can_play_live: bool,
        can_premove: bool,
    ) -> tuple[str, tuple[int, int, str | None] | None]:
        """Finish a drag on mouse-up.

        Returns `("move", (from, to, promo))`, `("premove", (from, to, promo))`,
        `("promotion", None)`, or `("none", None)`.
        """
        if self._drag_origin is None:
            return "none", None
        origin = self._drag_origin
        was_active = self._drag_active
        self.cancel_drag()
        if not was_active or self._game is None:
            return "none", None

        drop_sq = self.pixel_to_square(x, y)
        if drop_sq is None or drop_sq == origin:
            return "none", None

        if can_play_live:
            legal = self._game.legal_moves_from(origin)
            if drop_sq in legal:
                if self._game.is_promotion_move(origin, drop_sq):
                    self._selected = origin
                    self._promotion = (origin, drop_sq)
                    return "promotion", None
                self.clear_selection()
                return "move", (origin, drop_sq, None)
            return "none", None

        if can_premove:
            targets = premove_targets_from(self._game.board, origin, player_color)
            if drop_sq in targets:
                piece = self._game.piece_at(origin)
                promo: str | None = None
                if piece is not None and piece.piece_type == chess.PAWN:
                    promo_rank = 7 if player_color == chess.WHITE else 0
                    if chess.square_rank(drop_sq) == promo_rank:
                        promo = "q"
                self._premove = (origin, drop_sq, promo)
                self._premove_selected = None
                self._premove_targets = []
                return "premove", self._premove
        return "none", None

    # ------------------------------------------------------------------
    # Interactive mode — animation & rendering
    # ------------------------------------------------------------------

    def _update_move_animation(self, clamped_dt: float) -> None:
        # Advance cosmetic capture fade-out
        if self._capture_fade is not None:
            sym, sq, elapsed, surf = self._capture_fade
            elapsed += clamped_dt
            if elapsed >= CAPTURE_FADE_DURATION:
                self._capture_fade = None
            else:
                self._capture_fade = (sym, sq, elapsed, surf)

        if self._move_anim is None:
            if self._anim_queue:
                next_info, next_dur = self._anim_queue.popleft()
                self._begin_move_animation(next_info, duration_override=next_dur)
            return

        self._move_anim_elapsed = min(
            self._move_anim_elapsed + clamped_dt, self._move_anim_duration
        )
        if self._move_anim_elapsed >= self._move_anim_duration:
            finished = self._move_anim
            # Trigger smooth captured-piece fade-out once the moving piece lands
            if (
                CAPTURE_FADE_DURATION > 0.0
                and finished.captured_piece
                and finished.captured_square is not None
            ):
                mover_white = finished.color == "white"
                cap_sym = (
                    finished.captured_piece.lower()
                    if mover_white
                    else finished.captured_piece.upper()
                )
                self._capture_fade = (
                    cap_sym,
                    finished.captured_square,
                    0.0,
                    self._get_piece_surface(cap_sym).copy(),
                )

            if self._anim_queue:
                next_info, next_dur = self._anim_queue.popleft()
                self._begin_move_animation(next_info, duration_override=next_dur)
            else:
                self._move_anim = None
                self._move_anim_base_board = None
                self._move_anim_elapsed = 0.0

    @staticmethod
    def _castling_rook_squares(info: MoveInfo) -> tuple[int, int] | None:
        """(rook_from, rook_to) for a castling move, else None."""
        if not info.is_castling:
            return None
        rank = chess.square_rank(info.from_square)
        if info.castling_side == "kingside":
            return chess.square(7, rank), chess.square(5, rank)
        return chess.square(0, rank), chess.square(3, rank)

    def _animation_hidden_squares(self) -> set[int]:
        """Destination squares whose (already-moved) pieces must not be drawn yet."""
        info = self._move_anim
        if info is None:
            return set()
        hidden = {info.to_square}
        rook = self._castling_rook_squares(info)
        if rook is not None:
            hidden.add(rook[1])
        return hidden

    def _animation_source_hidden_squares(self, info: MoveInfo) -> set[int]:
        """Origin squares hidden on `fen_before` while their pieces are mid-flight."""
        hidden = {info.from_square}
        rook = self._castling_rook_squares(info)
        if rook is not None:
            hidden.add(rook[0])
        return hidden

    def _build_marker_surfaces(self) -> None:
        """Pre-render selection, check, legal-target, shadow, and overlay tiles once."""
        sq = self._sq
        self._select_surf = _convert_surface(
            pygame.Surface((sq, sq), pygame.SRCALPHA), alpha=True
        )
        self._select_surf.fill(_SELECT_COLOR)

        self._premove_surf = _convert_surface(
            pygame.Surface((sq, sq), pygame.SRCALPHA), alpha=True
        )
        self._premove_surf.fill(_PREMOVE_COLOR)

        self._premove_dot_surf = _convert_surface(
            pygame.Surface((sq, sq), pygame.SRCALPHA), alpha=True
        )
        pygame.draw.circle(
            self._premove_dot_surf,
            _PREMOVE_DOT_COLOR,
            (sq // 2, sq // 2),
            max(4, int(sq * 0.16)),
        )

        self._check_surf = _convert_surface(
            pygame.Surface((sq, sq), pygame.SRCALPHA), alpha=True
        )
        steps = 8
        for index in range(steps):
            radius = int(sq * 0.5 * (1.0 - index / steps))
            alpha = int(40 + 160 * index / steps)
            pygame.draw.circle(
                self._check_surf, (*_CHECK_COLOR, alpha), (sq // 2, sq // 2), max(1, radius)
            )

        self._dot_surf = _convert_surface(
            pygame.Surface((sq, sq), pygame.SRCALPHA), alpha=True
        )
        pygame.draw.circle(
            self._dot_surf, _MARKER_COLOR, (sq // 2, sq // 2), max(4, int(sq * 0.16))
        )

        self._ring_surf = _convert_surface(
            pygame.Surface((sq, sq), pygame.SRCALPHA), alpha=True
        )
        pygame.draw.circle(
            self._ring_surf,
            _MARKER_COLOR,
            (sq // 2, sq // 2),
            sq // 2,
            width=max(3, int(sq * 0.09)),
        )

        self._promo_dim_surf = _convert_surface(
            pygame.Surface((sq * 8, sq * 8), pygame.SRCALPHA), alpha=True
        )
        self._promo_dim_surf.fill(_PROMO_DIM)

        self._hint_surf = _convert_surface(
            pygame.Surface((sq, sq), pygame.SRCALPHA), alpha=True
        )
        self._hint_surf.fill((16, 185, 129, 95))
        pygame.draw.rect(
            self._hint_surf,
            (16, 185, 129, 230),
            (0, 0, sq, sq),
            width=max(3, int(sq * 0.05)),
        )

        # Subtle soft ground shadow cached once for lifted sliding pieces
        sh_w = max(16, int(sq * 0.66))
        sh_h = max(8, int(sq * 0.24))
        self._lift_shadow_surf = _convert_surface(
            pygame.Surface((sh_w, sh_h), pygame.SRCALPHA), alpha=True
        )
        pygame.draw.ellipse(self._lift_shadow_surf, (0, 0, 0, 65), (0, 0, sh_w, sh_h))

    def _blit_piece_centered(
        self,
        surface: pygame.Surface,
        symbol: str,
        center: tuple[float, float],
        lift_level: int = 0,
    ) -> None:
        """Blit a cached piece surface rounded once to stable screen pixel coords."""
        if lift_level >= 2 and symbol in self._piece_lift_peak_cache:
            glyph = self._piece_lift_peak_cache[symbol]
        elif lift_level == 1 and symbol in self._piece_lift_mid_cache:
            glyph = self._piece_lift_mid_cache[symbol]
        else:
            glyph = self._get_piece_surface(symbol)
        surface.blit(glyph, glyph.get_rect(center=(round(center[0]), round(center[1]))))

    def _draw_interactive(self, surface: pygame.Surface) -> None:
        """Render the attached game: highlights, pieces, markers, animation, overlay."""
        game = self._game
        assert game is not None
        surface.blit(self._board_surf, self.board_rect.topleft)

        # Highlight matches active animating move so queued moves never flash prematurely
        active_hl = self._move_anim if self._move_anim is not None else game.last_move
        if active_hl is not None:
            surface.blit(self._hl_surf, self.square_to_pixel(active_hl.from_square))
            surface.blit(self._hl_surf, self.square_to_pixel(active_hl.to_square))

        # Check highlight only when not showing an earlier pre-move board snapshot
        if self._move_anim is None:
            king_sq = game.checked_king_square()
            if king_sq is not None:
                surface.blit(self._check_surf, self.square_to_pixel(king_sq))

        if self._selected is not None:
            surface.blit(self._select_surf, self.square_to_pixel(self._selected))

        if self._premove is not None:
            surface.blit(self._premove_surf, self.square_to_pixel(self._premove[0]))
            surface.blit(self._premove_surf, self.square_to_pixel(self._premove[1]))

        if self._premove_selected is not None:
            surface.blit(
                self._premove_surf, self.square_to_pixel(self._premove_selected)
            )

        if self._hint_squares is not None:
            surface.blit(self._hint_surf, self.square_to_pixel(self._hint_squares[0]))
            surface.blit(self._hint_surf, self.square_to_pixel(self._hint_squares[1]))

        # Draw static pieces from pre-move snapshot while animating, else live board
        if self._move_anim is not None and self._move_anim_base_board is not None:
            static_board = self._move_anim_base_board
            hidden = self._animation_source_hidden_squares(self._move_anim)
        else:
            static_board = game.board
            hidden = self._animation_hidden_squares()

        # Draw fading captured piece cleanly underneath (even if a premove starts sliding)
        if self._capture_fade is not None:
            _, cap_sq, cap_elapsed, cap_surf = self._capture_fade
            if self._move_anim is None or cap_sq != self._move_anim.from_square:
                fade_t = min(1.0, cap_elapsed / max(CAPTURE_FADE_DURATION, 1e-6))
                cap_surf.set_alpha(max(0, round(255 * (1.0 - fade_t))))
                cx, cy = self._square_center(cap_sq)
                surface.blit(cap_surf, cap_surf.get_rect(center=(round(cx), round(cy))))

        drag_hide_sq = self._drag_origin if self.is_dragging else None
        for square in chess.SQUARES:
            if square in hidden or square == drag_hide_sq:
                continue
            piece = static_board.piece_at(square)
            if piece is not None:
                self._blit_piece_centered(surface, piece.symbol(), self._square_center(square))

        for target in self._legal_targets:
            marker = self._ring_surf if target in self._capture_targets else self._dot_surf
            surface.blit(marker, self.square_to_pixel(target))

        for target in self._premove_targets:
            surface.blit(self._premove_dot_surf, self.square_to_pixel(target))

        if self._move_anim is not None:
            self._draw_move_animation(surface, self._move_anim)

        if self.is_dragging and self._drag_origin is not None:
            drag_piece = game.board.piece_at(self._drag_origin)
            if drag_piece is not None:
                dx, dy = self._drag_pos
                self._lift_shadow_surf.set_alpha(120)
                surface.blit(
                    self._lift_shadow_surf,
                    self._lift_shadow_surf.get_rect(
                        center=(dx, dy + round(self._sq * 0.34))
                    ),
                )
                self._blit_piece_centered(surface, drag_piece.symbol(), (dx, dy))

        if self._promotion is not None:
            self._draw_promotion_overlay(surface)

    def _draw_move_animation(self, surface: pygame.Surface, info: MoveInfo) -> None:
        mover_white = info.color == "white"

        # If base snapshot wasn't available, keep captured piece visible under sliding piece
        if (
            self._move_anim_base_board is None
            and info.captured_piece
            and info.captured_square is not None
        ):
            captured = info.captured_piece.lower() if mover_white else info.captured_piece.upper()
            self._blit_piece_centered(surface, captured, self._square_center(info.captured_square))

        t = self.animation_progress
        s = ease_progress(t, ANIMATION_EASING)
        arc = math.sin(math.pi * t)  # 0 at endpoints, 1 at midpoint
        # Smooth continuous shadow & gentle 2px lift — no discrete bitmap scale-popping
        shadow_alpha = int(115 * arc)
        shadow_offset_y = round(self._sq * 0.34)
        lift_y = 2.0 * arc

        def slide(symbol: str, from_sq: int, to_sq: int) -> None:
            fx, fy = self._square_center(from_sq)
            tx, ty = self._square_center(to_sq)
            cx = fx + (tx - fx) * s
            cy = fy + (ty - fy) * s
            if shadow_alpha > 6:
                self._lift_shadow_surf.set_alpha(shadow_alpha)
                surface.blit(
                    self._lift_shadow_surf,
                    self._lift_shadow_surf.get_rect(
                        center=(round(cx), round(cy) + shadow_offset_y)
                    ),
                )
            self._blit_piece_centered(surface, symbol, (cx, cy - lift_y), lift_level=0)

        rook = self._castling_rook_squares(info)
        if rook is not None:
            slide("R" if mover_white else "r", rook[0], rook[1])
        piece = info.piece if mover_white else info.piece.lower()
        slide(piece, info.from_square, info.to_square)

    def _draw_promotion_overlay(self, surface: pygame.Surface) -> None:
        assert self._game is not None and self._promotion is not None
        surface.blit(self._promo_dim_surf, (self._ox, self._oy))
        pawn = self._game.piece_at(self._promotion[0])
        white = pawn is None or pawn.color == chess.WHITE
        radius = max(4, int(self._sq * 0.12))
        for choice, rect in self.promotion_rects():
            pygame.draw.rect(surface, _PROMO_TILE, rect, border_radius=radius)
            pygame.draw.rect(surface, _PROMO_TILE_BORDER, rect, width=2, border_radius=radius)
            symbol = choice.upper() if white else choice
            self._blit_piece_centered(surface, symbol, rect.center)

    # ------------------------------------------------------------------
    # Piece cache
    # ------------------------------------------------------------------

    def _build_piece_cache(self) -> None:
        """Load SVG pieces at board square resolution and pre-cache subtle lift scales."""
        target_size = max(24, int(self._sq * 0.96))
        mid_scale = 0.5 * (ANIMATION_SCALE_START + ANIMATION_SCALE_PEAK)
        mid_size = max(target_size, round(target_size * mid_scale))
        peak_size = max(target_size, round(target_size * ANIMATION_SCALE_PEAK))

        for piece_char, svg_filename in _PIECE_SVG_MAP.items():
            svg_path = ASSETS_DIR / svg_filename
            loaded = False

            if svg_path.is_file():
                try:
                    base_surf = _load_svg_at_size(svg_path, target_size)
                    self._piece_cache[piece_char] = base_surf
                    loaded = True
                except Exception:
                    loaded = False

            if not loaded:
                is_white = piece_char.isupper()
                glyph = _WHITE_GLYPHS[piece_char] if is_white else _BLACK_GLYPHS[piece_char.upper()]
                color = (250, 248, 238) if is_white else (34, 36, 42)
                outline_color = (20, 22, 26) if is_white else (205, 210, 215)
                base_surf = self._render_piece_glyph(glyph, color, outline_color)
                self._piece_cache[piece_char] = base_surf

            # Pre-cache subtle 1.015x and 1.03x surfaces so animation never rescales per frame
            base = self._piece_cache[piece_char]
            if mid_size != base.get_width():
                self._piece_lift_mid_cache[piece_char] = _convert_surface(
                    pygame.transform.smoothscale(base, (mid_size, mid_size)), alpha=True
                )
            else:
                self._piece_lift_mid_cache[piece_char] = base
            if peak_size != base.get_width():
                self._piece_lift_peak_cache[piece_char] = _convert_surface(
                    pygame.transform.smoothscale(base, (peak_size, peak_size)), alpha=True
                )
            else:
                self._piece_lift_peak_cache[piece_char] = base

    @staticmethod
    def _apply_piece_palette(surface: pygame.Surface, piece: str) -> None:
        """Use solid black pieces and white pieces with dark outlines."""
        if piece.islower():
            surface.fill((0, 0, 0, 255), special_flags=pygame.BLEND_RGBA_MULT)
            return

        for x in range(surface.get_width()):
            for y in range(surface.get_height()):
                red, green, blue, alpha = surface.get_at((x, y))
                if alpha == 0:
                    continue
                color = (0, 0, 0) if (red + green + blue) // 3 < 100 else (255, 255, 255)
                surface.set_at((x, y), (*color, alpha))

    def _render_piece_glyph(
        self,
        glyph: str,
        color: tuple[int, int, int],
        outline_color: tuple[int, int, int],
    ) -> pygame.Surface:
        main = self._piece_font.render(glyph, True, color)
        outline = self._piece_font.render(glyph, True, outline_color)
        w = main.get_width() + 4
        h = main.get_height() + 4
        surf = pygame.Surface((w, h), pygame.SRCALPHA)
        for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            surf.blit(outline, (2 + dx, 2 + dy))
        surf.blit(main, (2, 2))
        return _convert_surface(surf, alpha=True)

    def _get_piece_surface(self, piece: str) -> pygame.Surface:
        return self._piece_cache[piece]

    # ------------------------------------------------------------------
    # Demo animation state machine
    # ------------------------------------------------------------------

    def _start_next_move(self) -> None:
        """Begin animating the next move in the current opening."""
        line = LINES[self._line_idx]
        moves = line["moves"]

        if self._move_idx >= len(moves):
            self._phase = "opening_pause"
            self._timer = 0.0
            self._phase_dur = _PAUSE_BETWEEN_OPENINGS
            return

        uci = moves[self._move_idx]

        f1, r1 = ord(uci[0]) - 97, int(uci[1]) - 1
        f2, r2 = ord(uci[2]) - 97, int(uci[3]) - 1
        self._anim_piece = self._position.piece_at(f1, r1)
        self._anim_from = (f1, r1)
        self._anim_to = (f2, r2)

        piece = self._position.piece_at(f1, r1)
        self._anim_castle = False
        if piece.upper() == "K" and abs(f2 - f1) == 2:
            self._anim_castle = True
            if f2 == 6:
                self._anim_rook_from = (7, r1)
                self._anim_rook_to = (5, r1)
            else:
                self._anim_rook_from = (0, r1)
                self._anim_rook_to = (3, r1)
            self._anim_rook_piece = self._position.piece_at(
                self._anim_rook_from[0], self._anim_rook_from[1]
            )

        self._phase = "move"
        self._timer = 0.0
        self._phase_dur = _MOVE_DURATION

    def _finish_move(self) -> None:
        """Land the piece and apply the move to the position."""
        line = LINES[self._line_idx]
        uci = line["moves"][self._move_idx]
        self._position.push_uci(uci)
        self._last_to = self._anim_to
        self._move_idx += 1

        self._phase = "pause"
        self._timer = 0.0
        self._phase_dur = _PAUSE_BETWEEN_MOVES

    def _start_fade(self) -> None:
        """Begin cross-fade to the next opening."""
        self._line_idx = (self._line_idx + 1) % len(LINES)
        next_line = LINES[self._line_idx]
        self._next_position = Position(next_line.get("fen", START_FEN))

        self._phase = "fade"
        self._timer = 0.0
        self._phase_dur = _FADE_DURATION
        self._fade_alpha = 0.0

    def _finish_fade(self) -> None:
        """Complete the cross-fade; reset for the new opening."""
        self._position = self._next_position  # type: ignore[assignment]
        self._next_position = None
        self._move_idx = 0
        self._last_to = None

        self._phase = "pause"
        self._timer = 0.0
        self._phase_dur = 0.8

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _ease_in_out(t: float) -> float:
        """Cubic ease-in-out for smooth piece sliding."""
        return ease_progress(t, "ease_in_out_cubic")
