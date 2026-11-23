"""Bot match setup and interactive game screen for Chess Ken."""

from collections import deque
import ctypes
import logging
import math
import os
from pathlib import Path
import random
import sys

import chess
import numpy as np
import pygame

from bots.ken_bot import BotWorker, KenBot, fallback_heuristic_move, get_profile
from chess_logic.chess_game import (
    ChessGame,
    GameOutcome,
    MatchClock,
    MoveInfo,
    TimeControlPreset,
    format_clock_seconds,
)
from config import ASSETS_DIR
from screens.base_screen import BaseScreen
from ui.animated_board import (
    AnimatedBoard,
    MAX_ANIMATION_DT,
    PREMOVE_ANIMATION_DURATION,
    ease_progress,
    get_bot_think_range,
)
from ui.button import Button
from ui.hamburger_menu import HamburgerMenu, MATCH_ACTIONS


logger = logging.getLogger(__name__)

_NOTICE_DURATION = 2.5  # seconds a short sidebar notice stays visible
PLAYER_USERNAME = "Player"
PLAYER_AVATAR_IMAGE = "focusken.png"

# ── Chess.com-inspired warm charcoal & green palette ─────────────────
THEME_BG = (38, 36, 33)                 # #262421 main window canvas
THEME_SIDEBAR = (48, 46, 43)            # #302E2B warm dark charcoal sidebar
THEME_PANEL = (58, 56, 53)              # #3A3835 card / control surface
THEME_PANEL_ALT = (43, 41, 38)          # #2B2926 recessed / alternating row
THEME_SURFACE = (58, 56, 53)            # #3A3835 button base surface
THEME_SURFACE_HOVER = (69, 66, 64)      # #454240 button hover surface
THEME_POPOVER_BG = (42, 40, 37)         # #2A2825 expanded preset container
THEME_BORDER = (74, 71, 68)             # #4A4744 thin subtle border
THEME_POPOVER_BORDER = (84, 80, 76)     # #54504C popover outline
THEME_SHADOW = (20, 19, 17)             # subtle elevation shadow
THEME_TEXT = (242, 240, 236)            # #F2F0EC near-white primary text
THEME_MUTED = (168, 164, 158)           # #A8A49E secondary / section label text
THEME_BUBBLE = (255, 255, 255)          # #FFFFFF speech bubble
THEME_BUBBLE_BORDER = THEME_BUBBLE
THEME_BUBBLE_TEXT = (38, 36, 33)        # #262421 speech bubble text
THEME_GREEN = (129, 182, 76)            # #81B64C primary action / selected outline green
THEME_GREEN_DARK = (94, 143, 54)        # #5E8F36 gradient bottom / darker green edge
THEME_GREEN_HOVER_TOP = (145, 198, 88)  # #91C658 hover gradient top
THEME_GREEN_HOVER_BOT = (108, 160, 62)  # #6CA03E hover gradient bottom
THEME_SELECTED_FILL = (64, 62, 59)      # #403E3B selected preset fill
THEME_SELECTED_OUTLINE = THEME_GREEN    # #81B64C selected preset green outline
THEME_CLOCK_ACTIVE_BG = (238, 236, 232)
THEME_CLOCK_ACTIVE_TEXT = (38, 36, 33)
THEME_CLOCK_INACTIVE_BG = (43, 41, 38)
THEME_CLOCK_INACTIVE_TEXT = (168, 164, 158)
THEME_CLOCK_LOW_BG = (178, 58, 48)
THEME_CLOCK_LOW_TEXT = (255, 248, 246)

# Consistent 8 / 12 / 16 / 24 px spacing & corner radius scale
SPACE_XS = 8
SPACE_SM = 12
SPACE_MD = 16
SPACE_LG = 24
SPACE_XL = 32
RADIUS_SM = 8
RADIUS_MD = 10
RADIUS_BUBBLE = 12

SETUP_HEADER_FONT_SIZE = 26
SETUP_NAME_FONT_SIZE = 30
SETUP_ELO_FONT_SIZE = 16
SETUP_SECTION_FONT_SIZE = 16
SETUP_BUBBLE_FONT_SIZE = 18
SETUP_CHOICE_FONT_SIZE = 18
SETUP_PLAY_FONT_SIZE = 30
SETUP_MIN_HEADER_FONT_SIZE = 18
SETUP_MIN_NAME_FONT_SIZE = 20
SETUP_MIN_ELO_FONT_SIZE = 14
SETUP_MIN_SECTION_FONT_SIZE = 14
SETUP_MIN_BUBBLE_FONT_SIZE = 16
SETUP_MIN_CHOICE_FONT_SIZE = 16
SETUP_MIN_PLAY_FONT_SIZE = 22
SETUP_AVATAR_SIZE = 64
SETUP_BUTTON_HEIGHT = 52
SETUP_PLAY_BUTTON_HEIGHT = 58
SETUP_MIN_BUTTON_HEIGHT = 44
SETUP_MIN_PLAY_BUTTON_HEIGHT = 48
SETUP_BUBBLE_PADDING_X = SPACE_MD
SETUP_BUBBLE_PADDING_Y = SPACE_SM
SETUP_BUBBLE_TAIL_HALF_HEIGHT = SPACE_XS
SETUP_BUBBLE_TAIL_INSET = SPACE_XS // 4
SETUP_CHEVRON_RADIUS = 6
SETUP_MIN_CHEVRON_RADIUS = 5
SETUP_PROFILE_GAP = SPACE_MD
SETUP_MENU_BUBBLE_GAP = SPACE_XS
SETUP_SECTION_GAP = SPACE_LG
SETUP_LABEL_BUTTON_GAP = SPACE_XS
SETUP_PLAY_GAP = SPACE_XL
SETUP_BUTTON_GAP = SPACE_XS
SETUP_NAME_ELO_GAP = SPACE_XS

_BG = THEME_BG
_SIDEBAR = THEME_SIDEBAR
_PANEL = THEME_PANEL
_PANEL_ALT = THEME_PANEL_ALT
_BORDER = THEME_BORDER
_TEXT = THEME_TEXT
_BUBBLE_TEXT = THEME_BUBBLE_TEXT
_BUBBLE = THEME_BUBBLE
_MUTED = THEME_MUTED
_ACCENT = THEME_GREEN
_ACTIVE_PILL = (74, 104, 48)
_PILL_TEXT = (248, 250, 245)

# ── Warm Charcoal & Emerald Time Control design constants ────────────
# Derived directly from THEME_SIDEBAR (#302E2B) — zero blue-gray cast.
TC_BG_DEEP = THEME_SIDEBAR              # #302E2B existing sidebar warm charcoal
TC_TRIGGER_BG = (56, 54, 51)            # #383633 elevated warm charcoal control
TC_TRIGGER_HOVER = (65, 62, 58)         # #413E3A warm hover state
TC_TRIGGER_OPEN = (52, 50, 47)          # #34322F settled warm open state
TC_PANEL_BG = (41, 39, 37)              # #292725 unified expanded warm charcoal
TC_PANEL_TOP = (46, 44, 41)             # #2E2C29 subtle warm top sheen
TC_BTN_BG = (54, 52, 49)                # #363431 secondary warm charcoal button
TC_BTN_HOVER = (66, 63, 59)             # #423F3B refined warm hover surface
TC_ACCENT = (16, 185, 129)              # #10B981 primary emerald accent
TC_ACCENT_BRIGHT = (52, 211, 153)       # #34D399 crisp emerald highlight text
TC_SELECTED_ACCENT = (132, 204, 22)     # #84CC16 subtle lime-emerald harmony
TC_SELECTED_BG = (33, 59, 49)           # subtle emerald-tinted warm charcoal
TC_SELECTED_HOVER = (39, 71, 58)        # active hover warm emerald tint
TC_TEXT_PRIMARY = (245, 245, 244)       # #F5F5F4 warm crisp primary text
TC_TEXT_SECONDARY = (168, 162, 158)     # #A8A29E warm stone secondary text
TC_TEXT_SELECTED = (236, 253, 245)      # high-contrast mint-white active text
TC_BORDER_SUBTLE = (70, 67, 63)         # ~rgba(255, 255, 255, 0.10) on warm charcoal
TC_BTN_BORDER = (76, 73, 69)            # refined warm resting button border
TC_BTN_BORDER_HOVER = (102, 98, 93)     # refined warm hover button border
TC_TRIGGER_BORDER_IDLE = (78, 75, 71)   # warm neutral subtle trigger outline
TC_TRIGGER_SPECULAR = (74, 71, 67)      # warm top highlight (no cool tint)
TC_PANEL_SPECULAR = (68, 65, 61, 115)   # subtle warm panel top edge sheen
TC_DIVIDER_COLOR = (60, 58, 54)         # delicate warm section hairline
TC_CHIP_BG = (42, 40, 38)               # #2A2826 warm recessed icon chip
TC_CHIP_BORDER = (68, 65, 61)           # warm chip border
TC_PILL_BG = (32, 58, 48)               # emerald-tinted category pill fill
TC_PILL_BORDER = (44, 128, 96)          # emerald-tinted category pill border
TC_SHADOW_RGB = (16, 15, 13)            # warm charcoal shadow (never blue-black)
TC_STOPWATCH_ICON_FILE = "timecontrol.png"

# Consistent Time Control spacing & animation system
TC_SECTION_GAP = 26                     # profile -> TIME CONTROL heading (24-28px)
TC_HEADING_DROPDOWN_GAP = 10            # heading -> selected dropdown (10-12px)
TC_DROPDOWN_PANEL_GAP = 9               # dropdown -> options panel (8-10px)
TC_PANEL_PAD = 14                       # panel internal padding (12-14px)
TC_CATEGORY_GAP = 12                    # category separation (10-12px)
TC_CAT_LABEL_GAP = 6                    # category label -> button row
TC_BUTTON_GAP = 9                       # horizontal/vertical button gap (8-10px)
TC_BUTTON_HEIGHT = 38                   # equal height for No Timer & all presets
TC_PANEL_RADIUS = 12
TC_BTN_RADIUS = 9
TC_DROPDOWN_ANIM_DURATION = 0.24        # smooth 240ms expand/collapse
TC_SELECT_ANIM_SPEED = 10.0             # smooth ~220ms selection transition

# ── Chess.com-style Speech Bubble animation constants ────────────────
BUBBLE_ANIM_DURATION = 0.50             # 500ms bubble emergence (450-600ms)
BUBBLE_TEXT_DELAY = 0.28                # 280ms delay before text fades in (250-350ms)
BUBBLE_TEXT_DURATION = 0.30             # 300ms text fade + subtle rise (250-350ms)
BUBBLE_TEXT_CHANGE_DURATION = 0.26      # 260ms smooth transition on dialogue updates
BUBBLE_TEXT_SLIDE_PX = 7                # gentle upward text glide
BUBBLE_SCALE_START = 0.38               # emerges naturally from avatar tail tip


def _prefers_reduced_motion() -> bool:
    """Return True when OS or environment prefers reduced UI motion."""
    env_val = os.environ.get("CHESSIQ_REDUCED_MOTION", "").strip().lower()
    if env_val in ("1", "true", "yes", "on"):
        return True
    if env_val in ("0", "false", "no", "off"):
        return False
    if sys.platform == "win32":
        try:
            enabled = ctypes.c_int(1)
            # SPI_GETCLIENTAREAANIMATION = 0x1042
            ok = ctypes.windll.user32.SystemParametersInfoW(
                0x1042, 0, ctypes.byref(enabled), 0
            )
            if ok and enabled.value == 0:
                return True
        except Exception:
            pass
    return False


def _ease_out_back_soft(t: float) -> float:
    """Subtle spring-like ease-out curve (~3.5% settle, no cartoon bounce)."""
    x = max(0.0, min(1.0, t))
    c1 = 1.35
    c3 = c1 + 1.0
    xm = x - 1.0
    return 1.0 + c3 * xm * xm * xm + c1 * xm * xm


def _lerp_tc_rgb(
    a: tuple[int, int, int], b: tuple[int, int, int], t: float
) -> tuple[int, int, int]:
    t_c = max(0.0, min(1.0, t))
    return (
        int(a[0] + (b[0] - a[0]) * t_c),
        int(a[1] + (b[1] - a[1]) * t_c),
        int(a[2] + (b[2] - a[2]) * t_c),
    )


def _load_tc_font(size: int, weight: str = "regular") -> pygame.font.Font:
    """Load Manrope / Plus Jakarta Sans (with real-weight Segoe UI fallback)."""
    size = max(10, size)
    if weight == "bold":
        for family in ("Manrope", "Plus Jakarta Sans", "Segoe UI"):
            path = pygame.font.match_font(family, bold=True)
            if path:
                return pygame.font.Font(path, size)
        return pygame.font.SysFont("Segoe UI", size, bold=True)
    if weight == "semibold":
        for family in (
            "Manrope SemiBold",
            "Plus Jakarta Sans SemiBold",
            "Segoe UI Semibold",
            "Manrope",
            "Plus Jakarta Sans",
            "Segoe UI",
        ):
            path = pygame.font.match_font(family)
            if path:
                return pygame.font.Font(path, size)
        return pygame.font.SysFont("Segoe UI", size, bold=True)
    for family in ("Manrope", "Plus Jakarta Sans", "Segoe UI"):
        path = pygame.font.match_font(family)
        if path:
            if path.lower().endswith("segoeuil.ttf"):
                alt = Path(path).with_name("segoeui.ttf")
                if alt.is_file():
                    return pygame.font.Font(str(alt), size)
            return pygame.font.Font(path, size)
    return pygame.font.SysFont("Segoe UI", size)


# ── Modern Game Over Dialog design constants ─────────────────────────
DIALOG_MAX_WIDTH = 440
DIALOG_MIN_WIDTH = 270
DIALOG_CORNER_RADIUS = 16
DIALOG_PAD_X = 24
DIALOG_PAD_Y = 22
DIALOG_SECTION_GAP = 16
DIALOG_AVATAR_BASE = 80
DIALOG_AVATAR_MIN = 48
DIALOG_RESULT_ICON_BASE = 54
DIALOG_RESULT_ICON_MIN = 34
DIALOG_BUTTON_HEIGHT_BASE = 54
DIALOG_BUTTON_HEIGHT_MIN = 40
DIALOG_BUTTON_ICON_BASE = 22
DIALOG_BUTTON_GAP = 12
DIALOG_CLOSE_SIZE = 40
DIALOG_CLOSE_ICON_SIZE = 20
DIALOG_ENTRANCE_DURATION = 0.22
DIALOG_ENTRANCE_SLIDE_PX = 16
DIALOG_ENTRANCE_SCALE_START = 0.96
DIALOG_OVERLAY_ALPHA = 158  # ~62% dim

DIALOG_CARD_BG = (44, 42, 39)            # #2C2A27 elevated warm charcoal
DIALOG_CARD_BORDER = (80, 76, 71)        # #504C47 subtle card outline
DIALOG_SUBTITLE_COLOR = (218, 214, 208)  # high-contrast readable subtitle
DIALOG_META_COLOR = (158, 154, 148)      # subtle secondary stats
DIALOG_WIN_COLOR = THEME_GREEN           # #81B64C ChessIQ green
DIALOG_LOSS_COLOR = (232, 96, 92)        # soft red
DIALOG_DRAW_COLOR = (134, 166, 204)      # neutral blue-gray
DIALOG_CROWN_GOLD = (245, 198, 72)       # winner crown accent
DIALOG_LOSER_RING = (92, 88, 84)         # muted neutral ring
DIALOG_CLOSE_BG = (58, 55, 52)
DIALOG_CLOSE_HOVER = (78, 74, 70)
DIALOG_SECONDARY_BG = (60, 57, 54)
DIALOG_SECONDARY_HOVER = (78, 74, 70)
DIALOG_ROBOT_ACCENT = (245, 178, 42)     # golden-orange robot accent
DIALOG_ROBOT_ICON_FILE = "chessiq-robot.png"


def format_endgame_dialog_text(
    outcome: GameOutcome,
    player_color: chess.Color,
    opponent_name: str,
) -> tuple[str, str, str]:
    """Return `(category, headline, subtitle)` from the human player's perspective.

    `category` is `'win'`, `'loss'`, or `'draw'`. Supports every termination
    produced by `ChessGame` and falls back gracefully on unknown reasons.
    """
    status = (outcome.status or "").strip().lower()
    if outcome.winner is None:
        category = "draw"
        headline = "Draw"
        draw_subtitles = {
            "stalemate": "Draw by stalemate",
            "draw_agreed": "Draw by agreement",
            "insufficient_material": "Draw by insufficient material",
            "threefold_repetition": "Draw by repetition",
            "fivefold_repetition": "Draw by repetition",
            "fifty_moves": "Draw by the 50-move rule",
            "seventyfive_moves": "Draw by the 75-move rule",
        }
        if status in draw_subtitles:
            subtitle = draw_subtitles[status]
        elif outcome.termination:
            subtitle = f"Draw by {outcome.termination.lower()}"
        else:
            subtitle = "Game ended in a draw"
        return category, headline, subtitle

    player_won = (outcome.winner == "white") == (player_color == chess.WHITE)
    category = "win" if player_won else "loss"
    headline = "You Won" if player_won else "You Lost"

    if status == "checkmate":
        subtitle = (
            "You won by checkmate" if player_won else f"{opponent_name} won by checkmate"
        )
    elif status == "resignation":
        subtitle = f"{opponent_name} resigned" if player_won else "You resigned"
    elif status == "timeout":
        subtitle = (
            f"{opponent_name} ran out of time" if player_won else "You ran out of time"
        )
    elif outcome.termination:
        winner_label = "You" if player_won else opponent_name
        subtitle = f"{winner_label} won by {outcome.termination.lower()}"
    else:
        subtitle = "You won the game" if player_won else f"{opponent_name} won the game"

    return category, headline, subtitle

# ── Structured time-control presets (No Timer + 9 timed presets) ─────
NO_TIMER_PRESET: TimeControlPreset | None = None

TIME_CONTROL_PRESETS: tuple[TimeControlPreset, ...] = (
    # Bullet
    TimeControlPreset("Bullet", 60, 0, "1 min"),
    TimeControlPreset("Bullet", 60, 1, "1 + 1"),
    TimeControlPreset("Bullet", 120, 1, "2 + 1"),
    # Blitz
    TimeControlPreset("Blitz", 180, 0, "3 min"),
    TimeControlPreset("Blitz", 180, 2, "3 + 2"),
    TimeControlPreset("Blitz", 300, 0, "5 min"),
    # Rapid
    TimeControlPreset("Rapid", 600, 0, "10 min"),
    TimeControlPreset("Rapid", 600, 5, "10 + 5"),
    TimeControlPreset("Rapid", 900, 10, "15 + 10"),
)

PRESET_BY_LABEL: dict[str, TimeControlPreset | None] = {
    "No Timer": None,
    **{preset.label: preset for preset in TIME_CONTROL_PRESETS},
}
DEFAULT_TIME_CONTROL_LABEL = "5 min"

TIME_CONTROL_GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("BULLET", tuple(p.label for p in TIME_CONTROL_PRESETS if p.category == "Bullet")),
    ("BLITZ", tuple(p.label for p in TIME_CONTROL_PRESETS if p.category == "Blitz")),
    ("RAPID", tuple(p.label for p in TIME_CONTROL_PRESETS if p.category == "Rapid")),
)

TIME_CONTROL_PRESET_GROUPS: tuple[tuple[str, tuple[TimeControlPreset, ...]], ...] = (
    ("Bullet", tuple(p for p in TIME_CONTROL_PRESETS if p.category == "Bullet")),
    ("Blitz", tuple(p for p in TIME_CONTROL_PRESETS if p.category == "Blitz")),
    ("Rapid", tuple(p for p in TIME_CONTROL_PRESETS if p.category == "Rapid")),
)

_TIME_CONTROL_TO_GROUP: dict[str, str] = {
    preset.label: preset.category.upper() for preset in TIME_CONTROL_PRESETS
}


def resolve_time_control_preset(
    value: str | TimeControlPreset | None,
) -> TimeControlPreset | None:
    """Resolve a preset label or TimeControlPreset into a structured preset (None for No Timer)."""
    if value is None:
        return None
    if isinstance(value, TimeControlPreset):
        return None if value.base_seconds <= 0 else value
    return PRESET_BY_LABEL.get(value)


def get_time_control_category(value: str | TimeControlPreset | None) -> str:
    """Return the category key ('NO TIMER', 'BULLET', 'BLITZ', or 'RAPID') for a time control."""
    preset = resolve_time_control_preset(value)
    if preset is None:
        return "NO TIMER"
    return preset.category.upper()


def format_time_control_label(value: str | TimeControlPreset | None) -> str:
    """Format a selected time control for display in the selector bar."""
    preset = resolve_time_control_preset(value)
    if preset is None:
        return "No Timer"
    return preset.display_label


_PIECE_FIGURINES: dict[str, str] = {
    "K": "♔",
    "Q": "♕",
    "R": "♖",
    "B": "♗",
    "N": "♘",
}

_OPENING_BOOK_LINES: dict[tuple[str, ...], str] = {
    # 1-ply
    ("e4",): "King's Pawn Game",
    ("d4",): "Queen's Pawn Game",
    ("c4",): "English Opening",
    ("Nf3",): "Réti Opening",
    ("f4",): "Bird's Opening",
    ("b3",): "Nimzo-Larsen Attack",
    ("g3",): "King's Fianchetto Opening",
    ("Nc3",): "Dunst Opening",
    ("b4",): "Polish Opening",
    # Pirc & Modern
    ("e4", "d6"): "Pirc Defense",
    ("e4", "d6", "d4"): "Pirc Defense",
    ("e4", "d6", "d4", "Nf6"): "Pirc Defense: Main Line",
    ("e4", "d6", "d4", "Nf6", "Nc3"): "Pirc Defense: Main Line",
    ("e4", "d6", "d4", "Nf6", "Nc3", "g6"): "Pirc Defense: Main Line",
    ("e4", "d6", "d4", "Nf6", "Nc3", "g6", "Bc4"): "Pirc Defense: Main Line, Kholmov System",
    ("e4", "d6", "d4", "Nf6", "Nc3", "g6", "Be3"): "Pirc Defense: Main Line, 150 Attack",
    ("e4", "d6", "d4", "Nf6", "Nc3", "g6", "f4"): "Pirc Defense: Austrian Attack",
    ("e4", "d6", "d4", "Nf6", "Nc3", "g6", "Nf3"): "Pirc Defense: Classical Variation",
    ("e4", "d6", "d4", "Nf6", "Nc3", "g6", "Be2"): "Pirc Defense: Classical Variation",
    ("e4", "g6"): "Modern Defense",
    ("e4", "g6", "d4", "Bg7"): "Modern Defense: Standard Line",
    ("e4", "g6", "d4", "Bg7", "Nc3", "d6"): "Modern Defense: Standard Line",
    # Sicilian Defense
    ("e4", "c5"): "Sicilian Defense",
    ("e4", "c5", "d4"): "Sicilian Defense: Smith-Morra Gambit",
    ("e4", "c5", "d4", "cxd4"): "Sicilian Defense: Smith-Morra Gambit",
    ("e4", "c5", "d4", "cxd4", "c3"): "Sicilian Defense: Smith-Morra Gambit",
    ("e4", "c5", "d4", "cxd4", "c3", "dxc3"): "Sicilian Defense: Smith-Morra Gambit, Accepted",
    ("e4", "c5", "d4", "cxd4", "c3", "dxc3", "Nxc3"): "Sicilian Defense: Smith-Morra Gambit, Accepted",
    ("e4", "c5", "d4", "cxd4", "c3", "Nf6"): "Sicilian Defense: Smith-Morra Gambit, Chicago Defense",
    ("e4", "c5", "d4", "cxd4", "c3", "d6"): "Sicilian Defense: Smith-Morra Gambit, Declined",
    ("e4", "c5", "d4", "cxd4", "c3", "d5"): "Sicilian Defense: Smith-Morra Gambit, Alapin Formation",
    ("e4", "c5", "d4", "cxd4", "Qxd4"): "Sicilian Defense: Chekhover Variation",
    ("e4", "c5", "d4", "cxd4", "Nf3"): "Sicilian Defense: Morra-transposed Open",
    ("e4", "c5", "Nf3"): "Sicilian Defense: Open",
    ("e4", "c5", "Nf3", "d6"): "Sicilian Defense: Open",
    ("e4", "c5", "Nf3", "d6", "d4"): "Sicilian Defense: Open",
    ("e4", "c5", "Nf3", "d6", "d4", "cxd4"): "Sicilian Defense: Open",
    ("e4", "c5", "Nf3", "d6", "d4", "cxd4", "Nxd4"): "Sicilian Defense: Open",
    ("e4", "c5", "Nf3", "d6", "d4", "cxd4", "Nxd4", "Nf6"): "Sicilian Defense: Open, Main Line",
    ("e4", "c5", "Nf3", "d6", "d4", "cxd4", "Nxd4", "Nf6", "Nc3"): "Sicilian Defense: Open, Main Line",
    ("e4", "c5", "Nf3", "d6", "d4", "cxd4", "Nxd4", "Nf6", "Nc3", "a6"): "Sicilian Defense: Najdorf Variation",
    ("e4", "c5", "Nf3", "d6", "d4", "cxd4", "Nxd4", "Nf6", "Nc3", "a6", "Bg5"): "Sicilian Defense: Najdorf, Bg5 Line",
    ("e4", "c5", "Nf3", "d6", "d4", "cxd4", "Nxd4", "Nf6", "Nc3", "a6", "Be3"): "Sicilian Defense: Najdorf, English Attack",
    ("e4", "c5", "Nf3", "d6", "d4", "cxd4", "Nxd4", "Nf6", "Nc3", "a6", "Be2"): "Sicilian Defense: Najdorf, Classical",
    ("e4", "c5", "Nf3", "d6", "d4", "cxd4", "Nxd4", "Nf6", "Nc3", "a6", "Bc4"): "Sicilian Defense: Najdorf, Fischer-Sozin Attack",
    ("e4", "c5", "Nf3", "d6", "d4", "cxd4", "Nxd4", "Nf6", "Nc3", "g6"): "Sicilian Defense: Dragon Variation",
    ("e4", "c5", "Nf3", "d6", "d4", "cxd4", "Nxd4", "Nf6", "Nc3", "g6", "Be3"): "Sicilian Defense: Dragon, Yugoslav Attack",
    ("e4", "c5", "Nf3", "d6", "d4", "cxd4", "Nxd4", "Nf6", "Nc3", "e6"): "Sicilian Defense: Scheveningen Variation",
    ("e4", "c5", "Nf3", "d6", "d4", "cxd4", "Nxd4", "Nf6", "Nc3", "Nc6"): "Sicilian Defense: Classical Variation",
    ("e4", "c5", "Nf3", "d6", "Bb5"): "Sicilian Defense: Canal-Sokolsky Attack",
    ("e4", "c5", "Nf3", "Nc6"): "Sicilian Defense: Old Sicilian",
    ("e4", "c5", "Nf3", "Nc6", "Bb5"): "Sicilian Defense: Rossolimo Variation",
    ("e4", "c5", "Nf3", "Nc6", "d4", "cxd4", "Nxd4"): "Sicilian Defense: Open",
    ("e4", "c5", "Nf3", "Nc6", "d4", "cxd4", "Nxd4", "Nf6", "Nc3", "e5"): "Sicilian Defense: Sveshnikov Variation",
    ("e4", "c5", "Nf3", "Nc6", "d4", "cxd4", "Nxd4", "e5"): "Sicilian Defense: Kalashnikov Variation",
    ("e4", "c5", "Nf3", "Nc6", "d4", "cxd4", "Nxd4", "g6"): "Sicilian Defense: Accelerated Dragon",
    ("e4", "c5", "Nf3", "Nc6", "d4", "cxd4", "Nxd4", "e6"): "Sicilian Defense: Taimanov Variation",
    ("e4", "c5", "Nf3", "e6"): "Sicilian Defense: French Variation",
    ("e4", "c5", "Nf3", "e6", "d4", "cxd4", "Nxd4", "a6"): "Sicilian Defense: Kan Variation",
    ("e4", "c5", "Nf3", "g6"): "Sicilian Defense: Hyperaccelerated Dragon",
    ("e4", "c5", "c3"): "Sicilian Defense: Alapin Variation",
    ("e4", "c5", "c3", "Nf6"): "Sicilian Defense: Alapin Variation",
    ("e4", "c5", "c3", "d5"): "Sicilian Defense: Alapin Variation",
    ("e4", "c5", "Nc3"): "Sicilian Defense: Closed",
    ("e4", "c5", "Nc3", "Nc6", "f4"): "Sicilian Defense: Grand Prix Attack",
    ("e4", "c5", "f4"): "Sicilian Defense: McDonnell Attack",
    ("e4", "c5", "Bc4"): "Sicilian Defense: Bowdler Attack",
    ("e4", "c5", "b4"): "Sicilian Defense: Wing Gambit",
    ("e4", "c5", "a3"): "Sicilian Defense: Mengarini Variation",
    ("e4", "c5", "c4"): "Sicilian Defense: Staunton-Cochrane Variation",
    # 1. e4 e5
    ("e4", "e5"): "King's Pawn Game: Open",
    ("e4", "e5", "Nf3"): "King's Knight Opening",
    ("e4", "e5", "Nf3", "Nc6"): "King's Knight Opening",
    ("e4", "e5", "Nf3", "Nc6", "Bb5"): "Ruy Lopez",
    ("e4", "e5", "Nf3", "Nc6", "Bb5", "a6"): "Ruy Lopez: Morphy Defense",
    ("e4", "e5", "Nf3", "Nc6", "Bb5", "a6", "Bxc6"): "Ruy Lopez: Exchange Variation",
    ("e4", "e5", "Nf3", "Nc6", "Bb5", "a6", "Ba4", "Nf6"): "Ruy Lopez: Morphy Defense",
    ("e4", "e5", "Nf3", "Nc6", "Bb5", "a6", "Ba4", "Nf6", "O-O", "Be7"): "Ruy Lopez: Closed Spanish Main Line",
    ("e4", "e5", "Nf3", "Nc6", "Bb5", "a6", "Ba4", "Nf6", "O-O", "Nxe4"): "Ruy Lopez: Open Variation",
    ("e4", "e5", "Nf3", "Nc6", "Bb5", "a6", "Ba4", "Nf6", "O-O", "Be7", "Re1", "b5", "Bb3", "O-O", "c3", "d5"): "Ruy Lopez: Marshall Attack",
    ("e4", "e5", "Nf3", "Nc6", "Bb5", "Nf6"): "Ruy Lopez: Berlin Defense",
    ("e4", "e5", "Nf3", "Nc6", "Bb5", "f5"): "Ruy Lopez: Schliemann Defense",
    ("e4", "e5", "Nf3", "Nc6", "Bb5", "Nd4"): "Ruy Lopez: Bird's Defense",
    ("e4", "e5", "Nf3", "Nc6", "Bc4"): "Italian Game",
    ("e4", "e5", "Nf3", "Nc6", "Bc4", "Bc5"): "Italian Game: Giuoco Piano",
    ("e4", "e5", "Nf3", "Nc6", "Bc4", "Bc5", "b4"): "Italian Game: Evans Gambit",
    ("e4", "e5", "Nf3", "Nc6", "Bc4", "Bc5", "c3", "Nf6", "d3"): "Italian Game: Giuoco Pianissimo",
    ("e4", "e5", "Nf3", "Nc6", "Bc4", "Nf6"): "Italian Game: Two Knights Defense",
    ("e4", "e5", "Nf3", "Nc6", "Bc4", "Nf6", "Ng5"): "Italian Game: Knight Attack",
    ("e4", "e5", "Nf3", "Nc6", "d4"): "Scotch Game",
    ("e4", "e5", "Nf3", "Nc6", "d4", "exd4", "Bc4"): "Scotch Game: Scotch Gambit",
    ("e4", "e5", "Nf3", "Nc6", "Nc3", "Nf6"): "Four Knights Game",
    ("e4", "e5", "Nf3", "Nc6", "Nc3", "Nf6", "Bb5"): "Four Knights Game: Spanish Variation",
    ("e4", "e5", "Nf3", "Nf6"): "Petrov's Defense",
    ("e4", "e5", "Nf3", "Nf6", "Nxe5"): "Petrov's Defense: Classical Attack",
    ("e4", "e5", "Nf3", "Nf6", "d4"): "Petrov's Defense: Steinitz Attack",
    ("e4", "e5", "Nf3", "d6"): "Philidor Defense",
    ("e4", "e5", "Nc3"): "Vienna Game",
    ("e4", "e5", "Nc3", "Nf6", "f4"): "Vienna Game: Vienna Gambit",
    ("e4", "e5", "Bc4"): "Bishop's Opening",
    ("e4", "e5", "f4"): "King's Gambit",
    ("e4", "e5", "f4", "exf4"): "King's Gambit Accepted",
    ("e4", "e5", "f4", "d5"): "King's Gambit Declined: Falkbeer Countergambit",
    ("e4", "e5", "d4"): "Center Game",
    ("e4", "e5", "d4", "exd4", "c3"): "Center Game: Danish Gambit",
    # Other 1. e4 responses
    ("e4", "e6"): "French Defense",
    ("e4", "e6", "d4", "d5"): "French Defense",
    ("e4", "e6", "d4", "d5", "Nc3"): "French Defense: Classical / Winawer",
    ("e4", "e6", "d4", "d5", "Nc3", "Nf6"): "French Defense: Classical Variation",
    ("e4", "e6", "d4", "d5", "Nc3", "Bb4"): "French Defense: Winawer Variation",
    ("e4", "e6", "d4", "d5", "Nc3", "dxe4"): "French Defense: Rubinstein Variation",
    ("e4", "e6", "d4", "d5", "Nd2"): "French Defense: Tarrasch Variation",
    ("e4", "e6", "d4", "d5", "e5"): "French Defense: Advance Variation",
    ("e4", "e6", "d4", "d5", "exd5"): "French Defense: Exchange Variation",
    ("e4", "c6"): "Caro-Kann Defense",
    ("e4", "c6", "d4", "d5"): "Caro-Kann Defense",
    ("e4", "c6", "d4", "d5", "Nc3"): "Caro-Kann Defense: Main Line",
    ("e4", "c6", "d4", "d5", "Nc3", "dxe4", "Nxe4", "Bf5"): "Caro-Kann Defense: Classical Variation",
    ("e4", "c6", "d4", "d5", "Nc3", "dxe4", "Nxe4", "Nd7"): "Caro-Kann Defense: Karpov Variation",
    ("e4", "c6", "d4", "d5", "Nc3", "dxe4", "Nxe4", "Nf6"): "Caro-Kann Defense: Tartakower Variation",
    ("e4", "c6", "d4", "d5", "e5"): "Caro-Kann Defense: Advance Variation",
    ("e4", "c6", "d4", "d5", "exd5"): "Caro-Kann Defense: Exchange Variation",
    ("e4", "c6", "d4", "d5", "exd5", "cxd5", "c4"): "Caro-Kann Defense: Panov Attack",
    ("e4", "c6", "Nc3", "d5", "Nf3"): "Caro-Kann Defense: Two Knights Attack",
    ("e4", "d5"): "Scandinavian Defense",
    ("e4", "d5", "exd5", "Qxd5"): "Scandinavian Defense: Mieses-Kotroc Variation",
    ("e4", "d5", "exd5", "Nf6"): "Scandinavian Defense: Modern Variation",
    ("e4", "Nf6"): "Alekhine's Defense",
    ("e4", "Nf6", "e5", "Nd5"): "Alekhine's Defense: Normal Variation",
    ("e4", "Nc6"): "Nimzowitsch Defense",
    ("e4", "b6"): "Owen's Defense",
    # 1. d4 openings
    ("d4", "d5"): "Closed Game",
    ("d4", "d5", "c4"): "Queen's Gambit",
    ("d4", "d5", "c4", "e6"): "Queen's Gambit Declined",
    ("d4", "d5", "c4", "e6", "Nc3", "Nf6"): "Queen's Gambit Declined: Classical Development",
    ("d4", "d5", "c4", "e6", "Nf3", "Nf6"): "Queen's Gambit Declined: Three Knights",
    ("d4", "d5", "c4", "e6", "Nc3", "Nf6", "Bg5"): "Queen's Gambit Declined: Orthodox Defense",
    ("d4", "d5", "c4", "e6", "Nc3", "c5"): "Queen's Gambit Declined: Tarrasch Defense",
    ("d4", "d5", "c4", "c6"): "Slav Defense",
    ("d4", "d5", "c4", "c6", "Nf3", "Nf6", "Nc3", "e6"): "Semi-Slav Defense",
    ("d4", "d5", "c4", "dxc4"): "Queen's Gambit Accepted",
    ("d4", "d5", "c4", "e5"): "Queen's Gambit Declined: Albin Countergambit",
    ("d4", "d5", "c4", "Nc6"): "Queen's Gambit Declined: Chigorin Defense",
    ("d4", "d5", "Bf4"): "London System",
    ("d4", "d5", "Nf3", "Nf6", "Bf4"): "London System",
    ("d4", "d5", "Nf3", "Nf6", "e3"): "Colle System",
    ("d4", "d5", "e4"): "Blackmar-Diemer Gambit",
    ("d4", "Nf6"): "Indian Defense",
    ("d4", "Nf6", "Bf4"): "London System",
    ("d4", "Nf6", "Bg5"): "Trompowsky Attack",
    ("d4", "Nf6", "c4"): "Indian Game",
    ("d4", "Nf6", "c4", "g6"): "King's Indian Defense",
    ("d4", "Nf6", "c4", "g6", "Nc3", "Bg7"): "King's Indian Defense: Main Line",
    ("d4", "Nf6", "c4", "g6", "Nc3", "Bg7", "e4", "d6"): "King's Indian Defense: Normal Variation",
    ("d4", "Nf6", "c4", "g6", "Nc3", "Bg7", "e4", "d6", "Nf3", "O-O", "Be2"): "King's Indian Defense: Classical Variation",
    ("d4", "Nf6", "c4", "g6", "Nc3", "Bg7", "e4", "d6", "f3"): "King's Indian Defense: Sämisch Variation",
    ("d4", "Nf6", "c4", "g6", "Nc3", "Bg7", "e4", "d6", "f4"): "King's Indian Defense: Four Pawns Attack",
    ("d4", "Nf6", "c4", "g6", "Nc3", "d5"): "Grünfeld Defense",
    ("d4", "Nf6", "c4", "g6", "Nc3", "d5", "cxd5", "Nxd5", "e4"): "Grünfeld Defense: Exchange Variation",
    ("d4", "Nf6", "c4", "e6"): "Indian Defense: East Indian",
    ("d4", "Nf6", "c4", "e6", "Nc3", "Bb4"): "Nimzo-Indian Defense",
    ("d4", "Nf6", "c4", "e6", "Nc3", "Bb4", "e3"): "Nimzo-Indian Defense: Rubinstein System",
    ("d4", "Nf6", "c4", "e6", "Nc3", "Bb4", "Qc2"): "Nimzo-Indian Defense: Classical Variation",
    ("d4", "Nf6", "c4", "e6", "Nc3", "Bb4", "a3"): "Nimzo-Indian Defense: Sämisch Variation",
    ("d4", "Nf6", "c4", "e6", "Nf3", "b6"): "Queen's Indian Defense",
    ("d4", "Nf6", "c4", "e6", "Nf3", "Bb4"): "Bogo-Indian Defense",
    ("d4", "Nf6", "c4", "e6", "g3"): "Catalan Opening",
    ("d4", "Nf6", "c4", "e6", "g3", "d5", "Bg2"): "Catalan Opening: Closed",
    ("d4", "Nf6", "c4", "c5"): "Benoni Defense",
    ("d4", "Nf6", "c4", "c5", "d5", "e6"): "Benoni Defense: Modern Variation",
    ("d4", "Nf6", "c4", "c5", "d5", "b5"): "Benko Gambit",
    ("d4", "Nf6", "c4", "e5"): "Budapest Gambit",
    ("d4", "Nf6", "c4", "d6"): "Old Indian Defense",
    ("d4", "f5"): "Dutch Defense",
    ("d4", "f5", "g3", "Nf6", "Bg2", "g6"): "Dutch Defense: Leningrad Variation",
    ("d4", "f5", "c4", "e6", "g3", "d5"): "Dutch Defense: Stonewall Variation",
    # Flank openings
    ("c4", "e5"): "English Opening: Reversed Sicilian",
    ("c4", "c5"): "English Opening: Symmetrical Variation",
    ("c4", "Nf6"): "English Opening: Anglo-Indian Defense",
    ("c4", "e6"): "English Opening: Agincourt Defense",
    ("Nf3", "d5"): "Réti Opening: King's Indian Attack",
    ("Nf3", "Nf6"): "Réti Opening: Symmetrical",
    ("Nf3", "d5", "c4"): "Réti Opening: Réti Gambit",
}

_OPENING_EPD_MAP: dict[str, tuple[int, str]] | None = None


def _get_opening_epd_map() -> dict[str, tuple[int, str]]:
    """Build a lazy transposition lookup mapping EPD -> (canonical_ply, opening_name)."""
    global _OPENING_EPD_MAP
    if _OPENING_EPD_MAP is not None:
        return _OPENING_EPD_MAP

    epd_map: dict[str, tuple[int, str]] = {}
    for line, name in _OPENING_BOOK_LINES.items():
        board = chess.Board()
        valid = True
        for san in line:
            try:
                board.push_san(san)
            except ValueError:
                valid = False
                break
        if valid:
            epd = board.epd()
            existing = epd_map.get(epd)
            if existing is None or len(line) > existing[0]:
                epd_map[epd] = (len(line), name)

    _OPENING_EPD_MAP = epd_map
    return epd_map


def identify_opening(san_moves: list[str]) -> str:
    """Identify the ECO opening name from SAN moves via longest prefix and transposition."""
    if not san_moves:
        return "Starting Position"

    cleaned = [m.rstrip("+#") for m in san_moves]
    best_ply = 0
    best_name = "Custom Position"

    limit = min(len(cleaned), 18)
    for length in range(limit, 0, -1):
        prefix = tuple(cleaned[:length])
        if prefix in _OPENING_BOOK_LINES:
            best_ply = length
            best_name = _OPENING_BOOK_LINES[prefix]
            break

    # Check position transpositions up to book depth
    if best_ply < len(cleaned):
        epd_map = _get_opening_epd_map()
        board = chess.Board()
        for ply, san in enumerate(cleaned[:18], start=1):
            try:
                board.push_san(san)
            except ValueError:
                break
            match = epd_map.get(board.epd())
            if match is not None and match[0] > best_ply:
                best_ply = match[0]
                best_name = match[1]

    return best_name


def format_san_figurine(san: str) -> str:
    """Convert leading/promotion piece letters in SAN into chess figurine symbols."""
    if not san:
        return san
    if san[0] in _PIECE_FIGURINES:
        san = _PIECE_FIGURINES[san[0]] + san[1:]
    if "=" in san:
        head, tail = san.split("=", 1)
        if tail and tail[0] in _PIECE_FIGURINES:
            san = f"{head}={_PIECE_FIGURINES[tail[0]]}{tail[1:]}"
    return san


class GameScreen(BaseScreen):
    """Configure a bot match, then play it on an interactive board."""

    _asset_cache: dict[tuple[str, tuple[int, int]], pygame.Surface] = {}

    def __init__(
        self,
        screen: pygame.Surface,
        opponent: dict[str, object],
        premove_enabled: bool = True,
    ) -> None:
        super().__init__(screen)
        self._scale = min(self.width / 1280, self.height / 720)
        self._layout_scale = max(self._scale, min(self.height / 720, 1.2))
        self._premove_enabled = bool(premove_enabled)
        self._opponent_name = str(opponent["name"])
        self._opponent_elo = int(opponent["elo"])
        self._opponent_tagline = str(opponent["tagline"])
        self._selected_preset: TimeControlPreset | None = PRESET_BY_LABEL[
            DEFAULT_TIME_CONTROL_LABEL
        ]
        self._selected_time_control = DEFAULT_TIME_CONTROL_LABEL
        self._selected_side = "Random"
        self._game_started = False

        # ── match state (always backed by ChessGame, never demo replay) ─
        self._game: ChessGame = ChessGame()
        self._clock: MatchClock = MatchClock(self._selected_preset)
        self._resolved_side = "White"
        self._player_color: chess.Color = chess.WHITE
        self._bot: KenBot | None = None
        self._bot_worker: BotWorker | None = None
        # Incremented on restart / resign / draw so in-flight bot results are discarded.
        self._match_generation = 0
        # Generation of each outstanding bot request, in submission (FIFO) order.
        self._pending_requests: deque[int] = deque()
        self._bot_think_elapsed: float = 0.0
        self._bot_wait_remaining: float = 0.0
        self._ready_bot_move: tuple[chess.Move | None, bool] | None = None
        self._result_text: str | None = None
        self._notice_text: str | None = None
        self._notice_timer = 0.0
        self._chat_text = self._opponent_tagline
        self._reduced_motion = _prefers_reduced_motion()
        self._last_bubble_msg = self._opponent_tagline
        self._bubble_anim_elapsed: float = (
            BUBBLE_ANIM_DURATION + BUBBLE_TEXT_DURATION
            if self._reduced_motion
            else 0.0
        )
        self._bubble_text_anim_elapsed: float = BUBBLE_TEXT_CHANGE_DURATION
        self._current_opening = "Starting Position"
        self._cached_material_diff: int = 0
        self._move_scroll_y = 0
        self._show_endgame_popup = False
        self._pending_endgame_popup = False
        self._popup_anim_elapsed: float = 0.0
        self._popup_card_surface: pygame.Surface | None = None
        self._popup_shadow_surface: pygame.Surface | None = None
        self._opponent_image: str = str(opponent["image"])
        self._san_surface_cache: dict[tuple[str, bool], pygame.Surface] = {}
        self._text_surface_cache: dict[
            tuple[int, str, tuple[int, int, int]], pygame.Surface
        ] = {}
        self._wrapped_lines_cache: dict[
            tuple[int, str, int, tuple[int, int, int]], list[pygame.Surface]
        ] = {}

        sidebar_width = min(
            max(1, int(430 * self._scale)),
            max(1, int(self.width * 0.45)),
        )
        self._sidebar_rect = pygame.Rect(
            self.width - sidebar_width, 0, sidebar_width, self.height
        )
        self._board_area = pygame.Rect(0, 0, self._sidebar_rect.left, self.height)
        self._board_surface = self.screen.subsurface(self._board_area)
        self._board = AnimatedBoard(
            self._board_area.width,
            self._board_area.height,
            self._scale,
            square_scale=0.102,
        )
        self._board.attach_game(self._game, flipped=False)
        # ── fonts ─────────────────────────────────────────────────────
        self._font_header = pygame.font.SysFont(
            "Segoe UI", max(16, int(19 * self._scale)), bold=True
        )
        self._font_name = pygame.font.SysFont(
            "Segoe UI", max(15, int(20 * self._scale)), bold=True
        )
        self._font_bar = pygame.font.SysFont(
            "Segoe UI", max(14, int(16 * self._scale)), bold=True
        )
        self._font_elo = pygame.font.SysFont(
            "Segoe UI", max(12, int(14 * self._scale)), bold=True
        )
        self._font_label = pygame.font.SysFont(
            "Segoe UI", max(12, int(13 * self._scale)), bold=True
        )
        self._font_body = pygame.font.SysFont(
            "Segoe UI", max(12, int(14 * self._scale))
        )
        self._font_small = pygame.font.SysFont(
            "Segoe UI", max(11, int(12 * self._scale)), bold=True
        )
        self._font_moves = pygame.font.SysFont(
            "Segoe UI", max(12, int(14 * self._scale)), bold=True
        )
        self._font_symbol = pygame.font.SysFont(
            "Segoe UI Symbol", max(13, int(15 * self._scale))
        )
        self._font_choice = pygame.font.SysFont(
            "Segoe UI", max(12, int(14 * self._scale)), bold=True
        )
        self._font_clock = pygame.font.SysFont(
            "Consolas", max(14, int(18 * self._scale)), bold=True
        )
        self._font_play = pygame.font.SysFont(
            "Segoe UI", max(18, int(22 * self._scale)), bold=True
        )
        self._font_setup_header = pygame.font.SysFont(
            "Segoe UI",
            max(SETUP_MIN_HEADER_FONT_SIZE, round(SETUP_HEADER_FONT_SIZE * self._scale)),
            bold=True,
        )
        self._font_setup_name = pygame.font.SysFont(
            "Segoe UI",
            max(SETUP_MIN_NAME_FONT_SIZE, round(SETUP_NAME_FONT_SIZE * self._scale)),
            bold=True,
        )
        self._font_setup_elo = pygame.font.SysFont(
            "Segoe UI",
            max(SETUP_MIN_ELO_FONT_SIZE, round(SETUP_ELO_FONT_SIZE * self._scale)),
            bold=True,
        )
        self._font_setup_label = pygame.font.SysFont(
            "Segoe UI",
            max(
                SETUP_MIN_SECTION_FONT_SIZE,
                round(SETUP_SECTION_FONT_SIZE * self._scale),
            ),
            bold=True,
        )
        self._font_setup_bubble = _load_tc_font(
            max(
                SETUP_MIN_BUBBLE_FONT_SIZE,
                round(SETUP_BUBBLE_FONT_SIZE * self._scale),
            ),
            "regular",
        )
        self._font_setup_choice = pygame.font.SysFont(
            "Segoe UI",
            max(
                SETUP_MIN_CHOICE_FONT_SIZE,
                round(SETUP_CHOICE_FONT_SIZE * self._scale),
            ),
            bold=True,
        )
        self._font_setup_play = pygame.font.SysFont(
            "Segoe UI",
            max(SETUP_MIN_PLAY_FONT_SIZE, round(SETUP_PLAY_FONT_SIZE * self._scale)),
            bold=True,
        )

        # ── sidebar top box & avatars ─────────────────────────────────
        padding = max(SPACE_SM, int(18 * self._scale))
        content_left = self._sidebar_rect.left + padding
        content_width = self._sidebar_rect.width - padding * 2
        self._header_pos = (
            content_left,
            max(SPACE_SM, round(SPACE_MD * self._scale)),
        )
        self._header_setup_lines = self._get_wrapped_lines(
            "BOT MATCH SETUP",
            self._font_setup_header,
            THEME_TEXT,
            content_width,
        )
        self._header_play_surf = self._font_header.render(
            "Play Bots", True, THEME_TEXT
        )
        self._name_surface = self._font_setup_name.render(
            self._opponent_name, True, THEME_TEXT
        )
        self._elo_surface = self._font_setup_elo.render(
            f"ELO  {self._opponent_elo}", True, THEME_GREEN
        )
        self._avatar_size = max(40, round(SETUP_AVATAR_SIZE * self._scale))
        self._avatar = self._load_avatar(
            str(opponent["image"]), self._avatar_size
        )
        self._bar_avatar_size = max(28, int(38 * self._scale))
        self._bar_bot_avatar = self._load_avatar(
            str(opponent["image"]), self._bar_avatar_size
        )
        self._bar_player_avatar = self._load_avatar(
            PLAYER_AVATAR_IMAGE, self._bar_avatar_size
        )

        self._group_icon_size = max(14, int(18 * self._scale))
        self._group_icons = {
            "BULLET": self._load_group_icon("bullet.jpg", self._group_icon_size),
            "BLITZ": self._load_group_icon("blitz.webp", self._group_icon_size),
            "RAPID": self._load_group_icon("rapid.png", self._group_icon_size),
        }
        self._trigger_icon_size = max(18, round(23 * self._scale))
        self._trigger_icons = {
            "BULLET": self._load_group_icon("bullet.jpg", self._trigger_icon_size),
            "BLITZ": self._load_group_icon("blitz.webp", self._trigger_icon_size),
            "RAPID": self._load_group_icon("rapid.png", self._trigger_icon_size),
        }
        profile_top = (
            self._header_pos[1]
            + len(self._header_setup_lines) * self._font_setup_header.get_linesize()
            + max(SPACE_SM, round(SETUP_PROFILE_GAP * self._scale))
        )
        menu_bubble_gap = max(
            SPACE_XS, round(SETUP_MENU_BUBBLE_GAP * self._scale)
        )
        self._menu = HamburgerMenu(
            self.width,
            self.height,
            self._scale,
            actions=MATCH_ACTIONS,
            toggle_bottom_limit=profile_top - menu_bubble_gap,
        )
        avatar_top = max(
            profile_top,
            self._menu.toggle_rect.bottom + menu_bubble_gap,
        )
        self._avatar_rect = pygame.Rect(
            content_left,
            avatar_top,
            self._avatar_size,
            self._avatar_size,
        )
        bubble_gap = max(SPACE_XS, round(SPACE_SM * self._scale))
        bubble_width = max(
            1, content_width - self._avatar_size - bubble_gap
        )
        self._bubble_padding_x = max(
            SPACE_XS, round(SETUP_BUBBLE_PADDING_X * self._scale)
        )
        self._bubble_padding_y = max(
            SPACE_XS, round(SETUP_BUBBLE_PADDING_Y * self._scale)
        )
        self._tagline_rect = pygame.Rect(
            self._avatar_rect.right + bubble_gap,
            self._avatar_rect.top,
            bubble_width,
            1,
        )
        tagline_lines = self._get_wrapped_lines(
            self._opponent_tagline,
            self._font_setup_bubble,
            THEME_BUBBLE_TEXT,
            max(1, bubble_width - self._bubble_padding_x * 2),
        )
        self._tagline_rect.height = (
            max(1, len(tagline_lines)) * self._font_setup_bubble.get_linesize()
            + self._bubble_padding_y * 2
        )

        profile_bottom = max(self._avatar_rect.bottom, self._tagline_rect.bottom)
        name_gap = max(SPACE_XS, round(SETUP_PROFILE_GAP * self._scale))
        self._name_pos = (content_left, profile_bottom + name_gap)
        flag_width = max(16, round(26 * self._scale))
        flag_height = max(10, round(17 * self._scale))
        name_elo_gap = max(SPACE_XS, round(SETUP_NAME_ELO_GAP * self._scale))
        name_elo_width = (
            self._name_surface.get_width()
            + name_elo_gap
            + self._elo_surface.get_width()
            + name_elo_gap
            + flag_width
        )
        elo_y = self._name_pos[1]
        if name_elo_width > content_width:
            elo_y += self._font_setup_name.get_linesize() + name_elo_gap
            elo_x = content_left
        else:
            elo_x = content_left + self._name_surface.get_width() + name_elo_gap
        self._elo_pos = (elo_x, elo_y)
        self._flag_rect = pygame.Rect(0, 0, flag_width, flag_height)
        self._flag_rect.midleft = (
            self._elo_pos[0] + self._elo_surface.get_width() + name_elo_gap,
            self._elo_pos[1] + self._font_setup_elo.get_linesize() // 2,
        )
        self._profile_bottom = max(
            self._name_pos[1] + self._font_setup_name.get_linesize(),
            self._elo_pos[1] + self._font_setup_elo.get_linesize(),
            self._flag_rect.bottom,
        )
        self._flag_image = self._load_philippine_flag(
            (flag_width, flag_height)
        )
        bar_flag_w = max(18, int(24 * self._scale))
        bar_flag_h = max(12, int(16 * self._scale))
        self._bar_flag_image = self._load_philippine_flag((bar_flag_w, bar_flag_h))
        action_icon_size = max(16, int(19 * self._scale))
        self._action_icons: dict[str, pygame.Surface] = {
            "resign": self._load_action_icon("resign.png", action_icon_size, _TEXT),
            "newgame": self._load_action_icon("newgame", action_icon_size, _TEXT),
            "restart": self._load_action_icon("restart.jpg", action_icon_size, (255, 255, 255)),
        }
        self._bar_opp_label = self._font_bar.render(
            f"{self._opponent_name} ({self._opponent_elo})", True, THEME_TEXT
        )
        self._bar_player_label = self._font_bar.render(
            PLAYER_USERNAME, True, THEME_TEXT
        )

        # ── board bars & top-right settings gear ──────────────────────
        board_rect = self._board.board_rect
        top_cy = max(self._bar_avatar_size // 2 + 4, board_rect.top // 2)
        gear_size = max(28, int(36 * self._scale))
        gear_rect = pygame.Rect(0, 0, gear_size, gear_size)
        gear_rect.midright = (board_rect.right, top_cy)
        self._btn_board_settings = Button(
            gear_rect,
            "",
            self._font_label,
            color=THEME_SURFACE,
            hover_color=THEME_SURFACE_HOVER,
            text_color=THEME_TEXT,
            border_radius=max(6, int(RADIUS_SM * self._scale)),
            border_color=THEME_BORDER,
            border_width=1,
            accent_color=THEME_GREEN,
        )

        # ── setup controls (Modern Dark-Emerald Time Control selector) ─
        self._time_dropdown_open = False
        self._time_dropdown_anim: float = 0.0
        tc_section_gap = max(18, round(TC_SECTION_GAP * self._scale))
        tc_heading_gap = max(8, round(TC_HEADING_DROPDOWN_GAP * self._scale))
        tc_panel_gap = max(6, round(TC_DROPDOWN_PANEL_GAP * self._scale))

        # Typography dedicated to Time Control (Manrope / Plus Jakarta Sans -> Segoe UI)
        self._font_tc_heading = _load_tc_font(
            max(11, round(13 * self._scale)), "bold"
        )
        trigger_font_px = (
            max(12, round(14 * self._scale))
            if content_width < 200
            else max(15, round(18 * self._scale))
        )
        self._font_tc_trigger = _load_tc_font(trigger_font_px, "bold")
        self._font_tc_trigger_cat = _load_tc_font(
            max(10, round(12 * self._scale)), "semibold"
        )
        self._font_tc_category = _load_tc_font(
            max(11, round(13 * self._scale)), "semibold"
        )

        self._no_timer_label_y = self._profile_bottom + tc_section_gap
        self._tc_heading_surf = self._build_tc_heading_surface()
        dropdown_h = max(
            SETUP_MIN_BUTTON_HEIGHT,
            round(SETUP_BUTTON_HEIGHT * self._scale),
        )
        dropdown_top = (
            self._no_timer_label_y
            + self._tc_heading_surf.get_height()
            + tc_heading_gap
        )
        self._time_dropdown_rect = pygame.Rect(
            content_left,
            dropdown_top,
            content_width,
            dropdown_h,
        )
        self._tc_panel_radius = max(8, round(TC_PANEL_RADIUS * self._scale))
        self._tc_btn_radius = max(7, round(TC_BTN_RADIUS * self._scale))
        self._btn_time_dropdown = Button(
            self._time_dropdown_rect,
            "",
            self._font_tc_trigger,
            color=TC_TRIGGER_BG,
            hover_color=TC_TRIGGER_HOVER,
            text_color=TC_TEXT_PRIMARY,
            border_radius=self._tc_panel_radius,
            border_color=TC_TRIGGER_BORDER_IDLE,
            border_width=1,
            accent_color=TC_ACCENT,
            press_feedback=True,
        )

        # Layout of grouped presets inside the expanded Time Control panel
        panel_top = self._time_dropdown_rect.bottom + tc_panel_gap
        bottom_margin = max(8, round(12 * self._scale))
        avail_panel_h = max(160, self.height - panel_top - bottom_margin)

        inner_pad = max(8, round(TC_PANEL_PAD * self._scale))
        button_gap = max(6, round(TC_BUTTON_GAP * self._scale))
        button_height = max(28, round(TC_BUTTON_HEIGHT * self._scale))
        cat_label_gap = max(4, round(TC_CAT_LABEL_GAP * self._scale))
        header_h = max(
            self._group_icon_size, self._font_tc_category.get_linesize()
        ) + cat_label_gap
        section_gap = max(7, round(TC_CATEGORY_GAP * self._scale))
        divider_space = max(10, round(14 * self._scale))
        inner_width = max(1, content_width - inner_pad * 2)

        # Choose crisp responsive option font so 3 columns fit whenever space permits
        choice_px = max(13, round(15 * self._scale))
        if inner_width < 230:
            choice_px = max(12, min(choice_px, 13))
        if inner_width < 160:
            choice_px = 11
        self._font_tc_choice = _load_tc_font(choice_px, "bold")
        max_preset_txt_w = max(
            self._font_tc_choice.size(preset.label)[0]
            for preset in TIME_CONTROL_PRESETS
        )
        min_choice_w = max_preset_txt_w + max(12, round(18 * self._scale))
        if min_choice_w * 3 + button_gap * 2 <= inner_width:
            time_column_count = 3
        elif min_choice_w * 2 + button_gap <= inner_width:
            time_column_count = 2
        else:
            time_column_count = 1

        # Compact vertical spacing proportionally only when viewport height is tight
        num_groups = len(TIME_CONTROL_PRESET_GROUPS)
        total_preset_rows = sum(
            (len(presets) + time_column_count - 1) // time_column_count
            for _, presets in TIME_CONTROL_PRESET_GROUPS
        )
        natural_h = (
            inner_pad * 2
            + button_height
            + divider_space
            + num_groups * header_h
            + total_preset_rows * button_height
            + (total_preset_rows - num_groups) * button_gap
            + (num_groups - 1) * section_gap
        )
        if natural_h > avail_panel_h:
            fit_ratio = avail_panel_h / max(1, natural_h)
            inner_pad = max(6, int(inner_pad * fit_ratio))
            button_gap = max(4, int(button_gap * fit_ratio))
            button_height = max(24, int(button_height * fit_ratio))
            cat_label_gap = max(3, int(cat_label_gap * fit_ratio))
            header_h = max(16, int(header_h * fit_ratio))
            section_gap = max(5, int(section_gap * fit_ratio))
            divider_space = max(6, int(divider_space * fit_ratio))

        inner_left = content_left + inner_pad
        inner_width = max(1, content_width - inner_pad * 2)
        choice_width = (
            inner_width - button_gap * (time_column_count - 1)
        ) // time_column_count
        # Center any rounding remainder (1-2px) evenly inside the panel
        grid_total_w = choice_width * time_column_count + button_gap * (time_column_count - 1)
        grid_left = inner_left + (inner_width - grid_total_w) // 2

        self._time_buttons: dict[str, Button] = {}
        self._time_btn_select_t: dict[str, float] = {}
        self._side_buttons: dict[str, Button] = {}

        cursor_y = panel_top + inner_pad
        no_timer_rect = pygame.Rect(
            grid_left,
            cursor_y,
            grid_total_w,
            button_height,
        )
        self._time_buttons["No Timer"] = self._make_choice_button(
            "No Timer", no_timer_rect
        )
        self._time_btn_select_t["No Timer"] = (
            1.0 if self._selected_time_control == "No Timer" else 0.0
        )
        self._tc_divider_y = no_timer_rect.bottom + divider_space // 2
        cursor_y = no_timer_rect.bottom + divider_space

        row_step = button_height + button_gap
        self._tc_header_row_h = header_h - cat_label_gap
        self._group_label_positions: dict[str, int] = {}
        for group_title, presets in TIME_CONTROL_PRESET_GROUPS:
            group_key = group_title.upper()
            self._group_label_positions[group_key] = cursor_y
            cursor_y += header_h
            num_rows = (len(presets) + time_column_count - 1) // time_column_count
            for index, preset in enumerate(presets):
                row = index // time_column_count
                column = index % time_column_count
                rect = pygame.Rect(
                    grid_left + column * (choice_width + button_gap),
                    cursor_y + row * row_step,
                    choice_width,
                    button_height,
                )
                self._time_buttons[preset.label] = self._make_choice_button(
                    preset.label, rect
                )
                self._time_btn_select_t[preset.label] = (
                    1.0 if self._selected_time_control == preset.label else 0.0
                )
            cursor_y += num_rows * row_step - button_gap + section_gap

        panel_height = (cursor_y - section_gap + inner_pad) - panel_top
        self._time_dropdown_panel_rect = pygame.Rect(
            content_left,
            panel_top,
            content_width,
            panel_height,
        )
        self._dropdown_inner_left = grid_left
        self._tc_grid_width = grid_total_w
        self._tc_panel_shadow, self._tc_panel_card = self._build_tc_panel_surfaces(
            self._time_dropdown_panel_rect.size, self._tc_panel_radius
        )
        self._tc_trigger_shadow, _ = self._build_tc_panel_surfaces(
            self._time_dropdown_rect.size, self._tc_panel_radius, shadow_pad=10
        )
        self._tc_check_icon = self._build_tc_check_icon(max(10, round(12 * self._scale)))
        self._tc_no_timer_icon_idle = self._build_tc_no_timer_icon(
            self._group_icon_size, TC_TEXT_SECONDARY
        )
        self._tc_no_timer_icon_active = self._build_tc_no_timer_icon(
            self._group_icon_size, TC_ACCENT
        )
        self._tc_trigger_no_timer_icon = self._build_tc_no_timer_icon(
            self._trigger_icon_size, TC_ACCENT_BRIGHT
        )

        # Side selection ("PLAY AS") and green PLAY button (no Standard/Rated/Daily)
        setup_content_gap = tc_section_gap
        side_btn_h = max(
            SETUP_MIN_BUTTON_HEIGHT,
            round(SETUP_BUTTON_HEIGHT * self._scale),
        )
        self._side_label_y = (
            self._time_dropdown_rect.bottom + setup_content_gap
        )
        side_y = (
            self._side_label_y
            + self._font_setup_label.get_linesize()
            + max(SPACE_XS, round(SETUP_LABEL_BUTTON_GAP * self._scale))
        )
        side_button_gap = max(
            SPACE_XS, round(SETUP_BUTTON_GAP * self._scale)
        )
        side_button_min_width = max(
            64, self._font_setup_choice.size("Random")[0] + SPACE_MD
        )
        side_is_stacked = content_width < (
            side_button_min_width * 3 + side_button_gap * 2
        )
        side_choice_w = (
            content_width
            if side_is_stacked
            else (content_width - side_button_gap * 2) // 3
        )
        for index, side in enumerate(("White", "Random", "Black")):
            row = index if side_is_stacked else 0
            column = 0 if side_is_stacked else index
            rect = pygame.Rect(
                content_left + column * (side_choice_w + side_button_gap),
                side_y + row * (side_btn_h + side_button_gap),
                side_choice_w,
                side_btn_h,
            )
            self._side_buttons[side] = self._make_choice_button(side, rect)

        side_bottom = max(
            button.base_rect.bottom for button in self._side_buttons.values()
        )
        play_h = max(
            SETUP_MIN_PLAY_BUTTON_HEIGHT,
            round(SETUP_PLAY_BUTTON_HEIGHT * self._scale),
        )
        play_y = side_bottom + max(
            SPACE_XL, round(SETUP_PLAY_GAP * self._scale)
        )
        play_rect = pygame.Rect(
            content_left,
            play_y,
            content_width,
            play_h,
        )
        self._btn_play = Button(
            play_rect,
            "PLAY",
            self._font_setup_play,
            color=THEME_GREEN,
            hover_color=THEME_GREEN_HOVER_TOP,
            text_color=(255, 255, 255),
            border_radius=max(8, int(RADIUS_MD * self._scale)),
            accent=True,
            gradient_colors=(THEME_GREEN, THEME_GREEN_DARK),
            hover_gradient_colors=(THEME_GREEN_HOVER_TOP, THEME_GREEN_HOVER_BOT),
            shadow_color=THEME_SHADOW,
            border_color=THEME_GREEN_DARK,
            border_width=1,
            accent_color=THEME_GREEN,
            press_feedback=True,
        )
        self._refresh_choice_styles()

        # ── gameplay sidebar layout ───────────────────────────────────
        opening_top = int(150 * self._layout_scale)
        opening_h = max(30, int(36 * self._scale))
        self._opening_rect = pygame.Rect(
            content_left, opening_top, content_width, opening_h
        )

        bottom_pad = max(12, int(SPACE_MD * self._scale))
        ctrl_h = max(38, int(46 * self._scale))
        ctrl_top = self.height - bottom_pad - ctrl_h
        footer_h = max(20, int(22 * self._scale))
        footer_top = ctrl_top - max(4, int(6 * self._scale)) - footer_h
        self._footer_rect = pygame.Rect(
            content_left, footer_top, content_width, footer_h
        )

        moves_top = self._opening_rect.bottom + max(6, int(SPACE_XS * self._scale))
        moves_bottom = footer_top - max(6, int(SPACE_XS * self._scale))
        self._moves_rect = pygame.Rect(
            content_left,
            moves_top,
            content_width,
            max(80, moves_bottom - moves_top),
        )
        self._moves_pad_y = max(6, int(SPACE_XS * self._scale))
        self._move_row_h = max(24, int(30 * self._scale))

        # Lower control button (Resign / New Game when match ends)
        self._btn_resign = self._make_control_button(
            pygame.Rect(content_left, ctrl_top, content_width, ctrl_h)
        )

        # ── endgame popup modal (responsive layout) ───────────────────
        self._build_endgame_dialog_layout()

    @classmethod
    def _load_avatar(cls, filename: str, size: int) -> pygame.Surface:
        cache_key = (f"avatar:{filename}", (size, size))
        if cache_key in cls._asset_cache:
            return cls._asset_cache[cache_key]

        image = pygame.image.load(str(ASSETS_DIR / filename)).convert_alpha()
        width, height = image.get_size()
        scale = max(size / width, size / height)
        image = pygame.transform.smoothscale(
            image, (round(width * scale), round(height * scale))
        )
        crop = pygame.Rect(0, 0, size, size)
        crop.center = image.get_rect().center
        avatar = image.subsurface(crop).copy()
        mask = pygame.Surface((size, size), pygame.SRCALPHA)
        pygame.draw.circle(mask, (255, 255, 255, 255), (size // 2, size // 2), size // 2)
        avatar.blit(mask, (0, 0), special_flags=pygame.BLEND_RGBA_MULT)
        cls._asset_cache[cache_key] = avatar
        return avatar

    @classmethod
    def _load_group_icon(cls, filename: str, size: int) -> pygame.Surface:
        cache_key = (filename, (size, size))
        if cache_key in cls._asset_cache:
            return cls._asset_cache[cache_key]

        image = pygame.image.load(str(ASSETS_DIR / filename)).convert_alpha()
        width, height = image.get_size()
        if filename == "bullet.jpg":
            work_side = max(128, size * 6)
            if width > work_side or height > work_side:
                scale = work_side / max(width, height)
                image = pygame.transform.smoothscale(
                    image,
                    (max(1, round(width * scale)), max(1, round(height * scale))),
                )
            corner_r, corner_g, corner_b, _ = image.get_at((0, 0))
            if min(corner_r, corner_g, corner_b) > 120:
                cls._remove_checkerboard_background(image)
            else:
                cls._remove_dark_background(image)
        elif filename == "rapid.png":
            side = min(width, height)
            crop = pygame.Rect((width - side) // 2, (height - side) // 2, side, side)
            image = image.subsurface(crop).copy()
            cls._remove_edge_white_background(image)

        bounds = image.get_bounding_rect(min_alpha=24)
        if bounds.width > 0 and bounds.height > 0:
            image = image.subsurface(bounds).copy()

        bw, bh = image.get_size()
        fit_scale = min(size / max(1, bw), size / max(1, bh))
        scaled_w = max(1, min(size, round(bw * fit_scale)))
        scaled_h = max(1, min(size, round(bh * fit_scale)))
        scaled_img = pygame.transform.smoothscale(image, (scaled_w, scaled_h))

        icon = pygame.Surface((size, size), pygame.SRCALPHA)
        icon.blit(scaled_img, scaled_img.get_rect(center=(size // 2, size // 2)))
        cls._asset_cache[cache_key] = icon
        return icon

    @classmethod
    def _load_action_icon(
        cls,
        filename: str,
        size: int,
        tint_color: tuple[int, int, int] = _TEXT,
    ) -> pygame.Surface:
        """Load a dark-on-white action icon asset, strip white background, tint, and cache."""
        cache_key = (f"{filename}:{tint_color}", (size, size))
        if cache_key in cls._asset_cache:
            return cls._asset_cache[cache_key]

        image = pygame.image.load(str(ASSETS_DIR / filename)).convert_alpha()
        rgb = pygame.surfarray.array3d(image).astype(np.float32)
        src_alpha = pygame.surfarray.array_alpha(image).astype(np.float32)

        lum = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
        glyph_alpha = np.clip((245.0 - lum) * (255.0 / 210.0), 0.0, 255.0)
        glyph_alpha = np.where(lum >= 242.0, 0.0, glyph_alpha)
        final_alpha = np.clip(glyph_alpha * (src_alpha / 255.0), 0.0, 255.0).astype(
            np.uint8
        )

        tinted = pygame.Surface(image.get_size(), pygame.SRCALPHA)
        t_rgb = pygame.surfarray.pixels3d(tinted)
        t_rgb[:, :, 0] = tint_color[0]
        t_rgb[:, :, 1] = tint_color[1]
        t_rgb[:, :, 2] = tint_color[2]
        del t_rgb
        t_alpha = pygame.surfarray.pixels_alpha(tinted)
        t_alpha[:, :] = final_alpha
        del t_alpha

        bounds = tinted.get_bounding_rect(min_alpha=20)
        if bounds.width > 0 and bounds.height > 0:
            tinted = tinted.subsurface(bounds).copy()

        bw, bh = tinted.get_size()
        fit_scale = min(size / max(1, bw), size / max(1, bh))
        scaled_w = max(1, min(size, round(bw * fit_scale)))
        scaled_h = max(1, min(size, round(bh * fit_scale)))
        scaled_img = pygame.transform.smoothscale(tinted, (scaled_w, scaled_h))

        icon = pygame.Surface((size, size), pygame.SRCALPHA)
        icon.blit(scaled_img, scaled_img.get_rect(center=(size // 2, size // 2)))
        cls._asset_cache[cache_key] = icon
        return icon

    @classmethod
    def _load_philippine_flag(
        cls, size: tuple[int, int]
    ) -> pygame.Surface:
        cache_key = ("philippine.png", size)
        if cache_key in cls._asset_cache:
            return cls._asset_cache[cache_key]

        image = pygame.image.load(
            str(ASSETS_DIR / "philippine.png")
        ).convert_alpha()
        width, height = image.get_size()
        crop = pygame.Rect(0, round(height * 0.15), width, round(height * 0.70))
        image = image.subsurface(crop).copy()
        cls._remove_edge_white_background(image)
        flag = pygame.transform.smoothscale(image, size)
        cls._asset_cache[cache_key] = flag
        return flag

    @staticmethod
    def _remove_dark_background(image: pygame.Surface) -> None:
        rgb = pygame.surfarray.array3d(image)
        alpha = pygame.surfarray.pixels_alpha(image)
        dark_mask = np.max(rgb, axis=2) < 32
        alpha[dark_mask] = 0
        del alpha

    @staticmethod
    def _remove_checkerboard_background(image: pygame.Surface) -> None:
        """Extract the warm gold bullet glyph while removing the checkerboard and dark border."""
        rgb = pygame.surfarray.array3d(image).astype(np.int16)
        warmth = rgb[:, :, 0] - rgb[:, :, 2]
        alpha_vals = np.clip(((warmth - 35) * 2.2).astype(np.int16), 0, 255)
        alpha_vals = np.where((warmth <= 35) | (rgb[:, :, 0] < 100), 0, alpha_vals)
        alpha = pygame.surfarray.pixels_alpha(image)
        alpha[:, :] = alpha_vals.astype(np.uint8)
        del alpha

    @staticmethod
    def _remove_edge_white_background(image: pygame.Surface) -> None:
        width, height = image.get_size()
        visited = bytearray(width * height)
        pending: deque[tuple[int, int]] = deque()

        def enqueue(x: int, y: int) -> None:
            index = y * width + x
            if not visited[index]:
                visited[index] = 1
                pending.append((x, y))

        for x in range(width):
            enqueue(x, 0)
            enqueue(x, height - 1)
        for y in range(height):
            enqueue(0, y)
            enqueue(width - 1, y)

        while pending:
            x, y = pending.popleft()
            red, green, blue, _ = image.get_at((x, y))
            if min(red, green, blue) < 250:
                continue
            image.set_at((x, y), (red, green, blue, 0))
            if x > 0:
                enqueue(x - 1, y)
            if x + 1 < width:
                enqueue(x + 1, y)
            if y > 0:
                enqueue(x, y - 1)
            if y + 1 < height:
                enqueue(x, y + 1)

    @property
    def selected_time_control(self) -> TimeControlPreset | None:
        """Return the selected structured TimeControlPreset, or None for No Timer."""
        return self._selected_preset

    @property
    def match_clock(self) -> MatchClock:
        """Return the active MatchClock instance for the match."""
        return self._clock

    def select_time_control(self, value: str | TimeControlPreset | None) -> None:
        """Set the active time control from a preset object, label string, or None."""
        preset = resolve_time_control_preset(value)
        self._selected_preset = preset
        self._selected_time_control = "No Timer" if preset is None else preset.label
        self._refresh_choice_styles()

    def _get_cached_text(
        self,
        font: pygame.font.Font,
        text: str,
        color: tuple[int, int, int],
    ) -> pygame.Surface:
        """Return a cached rendered text surface for (font, text, color)."""
        key = (id(font), text, color)
        surf = self._text_surface_cache.get(key)
        if surf is None:
            surf = font.render(text, True, color)
            self._text_surface_cache[key] = surf
        return surf

    def _make_choice_button(self, label: str, rect: pygame.Rect) -> Button:
        return Button(
            rect,
            label,
            self._font_setup_choice,
            color=THEME_SURFACE,
            hover_color=THEME_SURFACE_HOVER,
            text_color=THEME_TEXT,
            border_radius=max(6, int(RADIUS_SM * self._scale)),
            border_color=THEME_BORDER,
            border_width=1,
            accent_color=THEME_GREEN,
            press_feedback=True,
        )

    def _make_control_button(
        self, rect: pygame.Rect, color: tuple[int, int, int] = THEME_SURFACE
    ) -> Button:
        return Button(
            rect,
            "",
            self._font_choice,
            color=color,
            hover_color=THEME_SURFACE_HOVER,
            text_color=THEME_TEXT,
            border_radius=max(6, int(RADIUS_SM * self._scale)),
            border_color=THEME_BORDER,
            border_width=1,
            accent_color=THEME_GREEN,
            press_feedback=True,
        )

    def _refresh_choice_styles(self) -> None:
        for value, button in self._time_buttons.items():
            selected = value == self._selected_time_control
            button.color = TC_SELECTED_BG if selected else TC_BTN_BG
            button.hover_color = TC_SELECTED_HOVER if selected else TC_BTN_HOVER
            button.text_color = TC_TEXT_SELECTED if selected else TC_TEXT_PRIMARY
            button.border_color = TC_ACCENT if selected else TC_BTN_BORDER
            button.border_width = max(2, round(2 * self._scale)) if selected else 1
            button.accent = selected
            button._accent_color = TC_ACCENT
        for value, button in self._side_buttons.items():
            selected = value == self._selected_side
            button.color = THEME_SELECTED_FILL if selected else THEME_SURFACE
            button.hover_color = THEME_SURFACE_HOVER
            button.text_color = THEME_TEXT if selected else THEME_MUTED
            button.border_color = (
                THEME_SELECTED_OUTLINE if selected else THEME_BORDER
            )
            button.border_width = max(2, int(2 * self._scale)) if selected else 1
            button.accent = selected

    def handle_events(self, events: list[pygame.event.Event]) -> None:
        for event in events:
            # When the Game Over modal is open (and menu closed), route clicks inside
            # the modal directly to the dialog so narrow-window close buttons stay usable.
            if (
                self._game_started
                and self._show_endgame_popup
                and not self._menu.is_open
            ):
                if event.type == pygame.KEYDOWN or (
                    hasattr(event, "pos")
                    and (
                        self._popup_rect.collidepoint(event.pos)
                        or self._popup_overlay_rect.collidepoint(event.pos)
                    )
                ):
                    self._handle_match_event(event)
                    continue

            menu_was_open = self._menu.is_open
            action = self._menu.handle_event(event)
            if action is not None:
                self._handle_menu_action(action)
                continue
            if menu_was_open or self._menu.is_open:
                continue

            if self._btn_board_settings.handle_event(event):
                self.next_screen = "settings"
                continue

            if self._game_started:
                self._handle_match_event(event)
                continue

            if (
                event.type == pygame.KEYDOWN
                and event.key == pygame.K_ESCAPE
                and self._time_dropdown_open
            ):
                self._time_dropdown_open = False
                continue

            if self._btn_time_dropdown.handle_event(event):
                self._time_dropdown_open = not self._time_dropdown_open
                continue

            if self._time_dropdown_open:
                selected_any = False
                for value, button in self._time_buttons.items():
                    if button.handle_event(event):
                        self.select_time_control(value)
                        self._time_dropdown_open = False
                        selected_any = True
                        break
                if selected_any:
                    continue
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    if not self._time_dropdown_panel_rect.collidepoint(event.pos):
                        self._time_dropdown_open = False
                    continue
                if (
                    event.type == pygame.MOUSEMOTION
                    and self._time_dropdown_panel_rect.collidepoint(event.pos)
                ):
                    continue

            for value, button in self._side_buttons.items():
                if button.handle_event(event):
                    self._selected_side = value
                    self._refresh_choice_styles()
                    if value in ("White", "Black"):
                        self._board.set_flipped(value == "Black")
                    break

            if self._btn_play.handle_event(event):
                self._start_match()

    def update(self, dt: float) -> None:
        self._menu.update(dt)
        self._btn_board_settings.update(dt)
        if self._notice_timer > 0:
            self._notice_timer = max(0.0, self._notice_timer - dt)

        # Speech bubble entrance & dynamic dialogue text transition
        clamped_dt = min(max(0.0, dt), MAX_ANIMATION_DT)
        current_bubble_msg = (
            self._chat_text if self._game_started else self._opponent_tagline
        )
        if current_bubble_msg != self._last_bubble_msg:
            self._last_bubble_msg = current_bubble_msg
            if self._reduced_motion:
                self._bubble_text_anim_elapsed = BUBBLE_TEXT_CHANGE_DURATION
            else:
                self._bubble_text_anim_elapsed = 0.0
        if self._reduced_motion:
            self._bubble_anim_elapsed = BUBBLE_ANIM_DURATION + BUBBLE_TEXT_DURATION
            self._bubble_text_anim_elapsed = BUBBLE_TEXT_CHANGE_DURATION
        else:
            max_entrance = max(
                BUBBLE_ANIM_DURATION, BUBBLE_TEXT_DELAY + BUBBLE_TEXT_DURATION
            )
            if self._bubble_anim_elapsed < max_entrance:
                self._bubble_anim_elapsed = min(
                    max_entrance, self._bubble_anim_elapsed + clamped_dt
                )
            if self._bubble_text_anim_elapsed < BUBBLE_TEXT_CHANGE_DURATION:
                self._bubble_text_anim_elapsed = min(
                    BUBBLE_TEXT_CHANGE_DURATION,
                    self._bubble_text_anim_elapsed + clamped_dt,
                )

        if self._game_started:
            for button in (
                self._btn_resign,
                self._btn_popup_rematch,
                self._btn_popup_bots,
                self._btn_popup_close,
            ):
                button.update(dt)
            self._board.update(dt)
            self._try_execute_premove()
            if self._pending_endgame_popup and not self._board.is_animating:
                self._pending_endgame_popup = False
                self._open_endgame_popup()
            if self._show_endgame_popup and self._popup_anim_elapsed < DIALOG_ENTRANCE_DURATION:
                clamped_dt = min(max(0.0, dt), MAX_ANIMATION_DT)
                self._popup_anim_elapsed = min(
                    DIALOG_ENTRANCE_DURATION, self._popup_anim_elapsed + clamped_dt
                )
            self._update_clock(dt)
            self._update_bot(dt)
        else:
            clamped_dt = min(max(0.0, dt), MAX_ANIMATION_DT)
            dd_step = clamped_dt / max(0.01, TC_DROPDOWN_ANIM_DURATION)
            if self._time_dropdown_open:
                self._time_dropdown_anim = min(1.0, self._time_dropdown_anim + dd_step)
            else:
                self._time_dropdown_anim = max(0.0, self._time_dropdown_anim - dd_step)
            sel_step = min(TC_SELECT_ANIM_SPEED * clamped_dt, 1.0)
            for value in self._time_buttons:
                target = 1.0 if value == self._selected_time_control else 0.0
                cur = self._time_btn_select_t.get(value, target)
                self._time_btn_select_t[value] = cur + (target - cur) * sel_step
            self._btn_time_dropdown.update(dt)
            self._btn_play.update(dt)
            for button in (*self._time_buttons.values(), *self._side_buttons.values()):
                button.update(dt)

    def _update_clock(self, dt: float) -> None:
        """Advance the active side's clock and trigger timeout if a flag falls."""
        if self._game is None:
            return
        if self._game.is_game_over:
            self._clock.pause()
            return
        if not self._clock.is_timed:
            return
        flagged = self._clock.tick(dt, self._game.turn_color)
        if flagged is not None:
            self._game.timeout(flagged)
            self._invalidate_pending_search()
            self._refresh_result()

    def captures_escape(self) -> bool:
        """Esc cancels open dropdowns, premoves, promotion selector, or endgame popup."""
        if not self._game_started:
            return self._time_dropdown_open
        return (
            self._board.promotion_pending
            or self._show_endgame_popup
            or self._board.premove is not None
            or self._board.premove_selected is not None
        )

    def close(self) -> None:
        """Stop the bot worker thread and its engine process."""
        if self._bot_worker is not None:
            self._bot_worker.stop()
        self._bot_worker = None
        self._bot = None
        self._pending_requests.clear()

    # ------------------------------------------------------------------
    # Match lifecycle
    # ------------------------------------------------------------------

    def _start_match(self) -> None:
        """Resolve the side, create the game and bot worker, and begin play."""
        self._time_dropdown_open = False
        self._time_dropdown_anim = 0.0
        self._game_started = True
        side = self._selected_side
        if side == "Random":
            side = random.choice(("White", "Black"))
        self._resolved_side = side
        self._player_color = chess.WHITE if side == "White" else chess.BLACK
        self._game = ChessGame()
        self._ensure_bot_worker()
        self._begin_new_game()

    def _ensure_bot_worker(self) -> None:
        if self._bot_worker is None:
            self._bot = KenBot(self._opponent_name)
            self._bot_worker = BotWorker(self._bot)

    def _begin_new_game(self) -> None:
        """Reset to the starting position with the current match settings."""
        assert self._game is not None
        self._invalidate_pending_search()
        self._game.reset()
        self._selected_preset = resolve_time_control_preset(
            self._selected_time_control
        )
        self._clock = MatchClock(self._selected_preset)
        self._board.attach_game(self._game, flipped=self._player_color == chess.BLACK)
        self._result_text = None
        self._show_endgame_popup = False
        self._pending_endgame_popup = False
        self._popup_anim_elapsed = 0.0
        self._notice_text = None
        self._notice_timer = 0.0
        self._chat_text = self._opponent_tagline
        self._sync_move_state(auto_scroll=True)

    def _invalidate_pending_search(self) -> None:
        """Mark any in-flight bot search as stale; its result will be discarded."""
        self._match_generation += 1
        self._bot_think_elapsed = 0.0
        self._bot_wait_remaining = 0.0
        self._ready_bot_move = None

    @property
    def _is_game_active(self) -> bool:
        return self._game_started and self._game is not None and not self._game.is_game_over

    @property
    def _bot_thinking(self) -> bool:
        return (
            self._match_generation in self._pending_requests
            or self._ready_bot_move is not None
        )

    def _show_notice(self, text: str) -> None:
        self._notice_text = text
        self._notice_timer = _NOTICE_DURATION

    def _sync_move_state(self, auto_scroll: bool = False) -> None:
        """Refresh opening banner, cached material diff, and move-list scroll bounds."""
        if self._game is None:
            self._current_opening = "Starting Position"
            self._cached_material_diff = 0
            self._move_scroll_y = 0
            return
        self._current_opening = identify_opening(self._game.san_moves)
        self._cached_material_diff = self._game.material_diff()
        max_scroll = self._max_move_scroll()
        if auto_scroll:
            self._move_scroll_y = max_scroll
        else:
            self._move_scroll_y = max(0, min(self._move_scroll_y, max_scroll))

    def _max_move_scroll(self) -> int:
        if self._game is None:
            return 0
        num_rows = (len(self._game.san_moves) + 1) // 2
        total_height = num_rows * self._move_row_h + self._moves_pad_y * 2
        return max(0, total_height - self._moves_rect.height)

    def _open_endgame_popup(self) -> None:
        """Open the Game Over dialog and start its smooth entrance animation."""
        self._show_endgame_popup = True
        self._popup_anim_elapsed = 0.0

    @property
    def popup_entrance_progress(self) -> float:
        """Normalized [0.0, 1.0] progress of the Game Over dialog entrance animation."""
        if not self._show_endgame_popup or DIALOG_ENTRANCE_DURATION <= 0.0:
            return 0.0
        return min(1.0, max(0.0, self._popup_anim_elapsed / DIALOG_ENTRANCE_DURATION))

    def _refresh_result(self) -> None:
        """Lock the board, pause clocks, and reveal result UI once final animation finishes."""
        if self._game is not None and self._game.is_game_over:
            self._clock.pause()
            self._result_text = self._describe_outcome(self._game.outcome)
            if self._board.is_animating:
                self._pending_endgame_popup = True
            else:
                self._pending_endgame_popup = False
                self._open_endgame_popup()
            self._board.clear_selection()
            self._board.clear_premove()
            self._chat_text = self._endgame_chat_line(self._game.outcome)

    def _describe_outcome(self, outcome: GameOutcome) -> str:
        if outcome.winner is None:
            return f"Draw — {outcome.termination}"
        player_won = (outcome.winner == "white") == (self._player_color == chess.WHITE)
        verdict = "You win!" if player_won else f"{self._opponent_name} wins."
        return f"{outcome.termination} — {verdict}"

    def _endgame_chat_line(self, outcome: GameOutcome) -> str:
        if outcome.winner is None:
            return "Good game! Solid draw."
        player_won = (outcome.winner == "white") == (self._player_color == chess.WHITE)
        if player_won:
            return "Nice game! You got me that time."
        return "GG! Want a rematch?"

    def _update_chat_for_move(self, info: MoveInfo) -> None:
        if info.is_checkmate:
            return
        if info.is_check:
            self._chat_text = "Check! Watch the king."
        elif info.is_promotion:
            self._chat_text = "Fresh queen on the board!"
        elif info.is_capture and info.captured_piece in ("Q", "R"):
            self._chat_text = "Heavy piece off the board!"
        elif info.is_castling:
            self._chat_text = "King safely tucked away."
        else:
            self._chat_text = self._opponent_tagline

    # ------------------------------------------------------------------
    # Input & Sidebar Actions
    # ------------------------------------------------------------------

    def _handle_menu_action(self, action: str) -> None:
        if action == "restart":
            if self._game_started:
                self._begin_new_game()
            else:
                self._game.reset()
                self._board.attach_game(
                    self._game,
                    flipped=self._selected_side == "Black",
                )
                self._sync_move_state(auto_scroll=True)
        elif action == "opponent_select":
            self.next_screen = "opponent_select"
        elif action == "dashboard":
            self.next_screen = "dashboard"
        elif action == "resign":
            self._resign()
        elif action == "offer_draw":
            self._offer_draw()

    def _resign(self) -> None:
        if not self._is_game_active:
            return
        assert self._game is not None
        self._game.resign("white" if self._player_color == chess.WHITE else "black")
        self._invalidate_pending_search()
        self._refresh_result()

    def _offer_draw(self) -> None:
        if not self._is_game_active:
            return
        assert self._game is not None
        if self._game.can_claim_draw:
            self._game.agree_draw()
            self._invalidate_pending_search()
            self._refresh_result()
        else:
            self._show_notice("Draw declined")

    def _handle_match_event(self, event: pygame.event.Event) -> None:
        if self._show_endgame_popup:
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self._show_endgame_popup = False
                    return
                if event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                    self._begin_new_game()
                    return
            if self._btn_popup_rematch.handle_event(event):
                self._begin_new_game()
                return
            if self._btn_popup_bots.handle_event(event):
                self.next_screen = "opponent_select"
                return
            if self._btn_popup_close.handle_event(event):
                self._show_endgame_popup = False
                return
            if event.type in (
                pygame.MOUSEBUTTONDOWN,
                pygame.MOUSEBUTTONUP,
                pygame.MOUSEMOTION,
            ):
                if self._popup_rect.collidepoint(event.pos):
                    return
                if (
                    event.type == pygame.MOUSEBUTTONDOWN
                    and event.button == 1
                    and self._popup_overlay_rect.collidepoint(event.pos)
                ):
                    return

        if self._handle_sidebar_match_event(event):
            return

        if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            if self._board.is_dragging:
                self._board.cancel_drag()
                return
            if self._board.promotion_pending:
                self._board.cancel_promotion()
                return
            if self._board.premove is not None or self._board.premove_selected is not None:
                self._board.clear_premove()
                return

        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 3:
            self._board.cancel_drag()
            if self._board.premove is not None or self._board.premove_selected is not None:
                self._board.clear_premove()
            return

        if event.type == pygame.MOUSEMOTION:
            if self._board._drag_origin is not None:
                local = self._screen_to_board(event.pos)
                if local is not None:
                    self._board.update_drag(local[0], local[1])
            return

        if event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            if self._board._drag_origin is not None:
                local = self._screen_to_board(event.pos)
                if local is None:
                    self._board.cancel_drag()
                else:
                    kind, payload = self._board.end_drag(
                        local[0],
                        local[1],
                        self._player_color,
                        can_play_live=self._can_accept_input(),
                        can_premove=self._can_accept_premove(),
                    )
                    if kind == "move" and payload is not None:
                        self._apply_player_move(*payload, animate=False)
            return

        if not (event.type == pygame.MOUSEBUTTONDOWN and event.button == 1):
            return
        if self._show_endgame_popup:
            return

        if self._can_accept_input():
            local = self._screen_to_board(event.pos)
            if local is None:
                # Click outside the board: cancel promotion first, else deselect.
                if self._board.promotion_pending:
                    self._board.cancel_promotion()
                else:
                    self._board.clear_selection()
                return

            move = self._board.handle_click(local[0], local[1], self._player_color)
            if move is not None:
                self._apply_player_move(*move)
            elif not self._board.promotion_pending:
                self._board.begin_drag(
                    local[0], local[1], self._player_color, is_premove=False
                )
            return

        if self._can_accept_premove():
            local = self._screen_to_board(event.pos)
            if local is None:
                self._board.clear_premove()
                return
            queued = self._board.handle_premove_click(
                local[0], local[1], self._player_color
            )
            if queued is None and self._board.premove_selected is not None:
                self._board.begin_drag(
                    local[0], local[1], self._player_color, is_premove=True
                )

    def _handle_sidebar_match_event(self, event: pygame.event.Event) -> bool:
        scroll_step = max(24, self._move_row_h)
        if event.type == pygame.MOUSEWHEEL:
            if self._moves_rect.collidepoint(pygame.mouse.get_pos()):
                self._move_scroll_y = max(
                    0,
                    min(self._move_scroll_y - event.y * scroll_step, self._max_move_scroll()),
                )
                return True
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button in (4, 5):
            if self._moves_rect.collidepoint(event.pos):
                delta = -scroll_step if event.button == 4 else scroll_step
                self._move_scroll_y = max(
                    0,
                    min(self._move_scroll_y + delta, self._max_move_scroll()),
                )
                return True

        if self._btn_resign.handle_event(event):
            if self._game is not None and self._game.is_game_over:
                self._begin_new_game()
            else:
                self._resign()
            return True
        return False

    def _screen_to_board(self, pos: tuple[int, int]) -> tuple[int, int] | None:
        """Convert screen coordinates to board-surface coordinates."""
        if not self._board_area.collidepoint(pos):
            return None
        return pos[0] - self._board_area.left, pos[1] - self._board_area.top

    @property
    def premove_enabled(self) -> bool:
        """Whether premoving during the opponent's turn is enabled."""
        return self._premove_enabled

    def _can_accept_input(self) -> bool:
        return (
            self._is_game_active
            and self._game is not None
            and self._game.turn_color == self._player_color
            and not self._board.is_animating
            and not self._bot_thinking
        )

    def _can_accept_premove(self) -> bool:
        if (
            not self._premove_enabled
            or not self._is_game_active
            or self._show_endgame_popup
            or self._game is None
        ):
            return False
        # Allow queueing a premove during the opponent's turn OR while the bot's reply is sliding
        return (
            self._game.turn_color != self._player_color
            or self._board.is_animating
        )

    def _try_execute_premove(self) -> None:
        """Execute a queued premove (or promote premove selection) once bot slide lands."""
        if (
            not self._is_game_active
            or self._show_endgame_popup
            or self._game is None
            or self._game.turn_color != self._player_color
            or self._board.is_animating
        ):
            return

        # If the player selected a piece before the bot moved but hadn't clicked a destination yet,
        # seamlessly promote it to a live selection so their next click plays immediately.
        if self._board.premove is None:
            if self._board.premove_selected is not None:
                self._board.promote_premove_selection(self._player_color)
            return

        if not self._premove_enabled:
            self._board.clear_premove()
            return

        queued = self._board.pop_premove()
        if queued is None:
            return
        from_sq, to_sq, promotion = queued
        self._apply_player_move(
            from_sq,
            to_sq,
            promotion,
            duration_override=PREMOVE_ANIMATION_DURATION,
        )

    def _apply_player_move(
        self,
        from_sq: int,
        to_sq: int,
        promotion: str | None,
        *,
        animate: bool = True,
        duration_override: float | None = None,
    ) -> None:
        assert self._game is not None
        info = self._game.make_move(from_sq, to_sq, promotion)
        if info is None:
            self._board.clear_selection()
            return
        if self._game.is_game_over:
            self._clock.pause()
        else:
            self._clock.apply_increment(info.color)
        if animate:
            self._board.animate_move(info, duration_override=duration_override)
        else:
            self._board.clear_selection()
        self._sync_move_state(auto_scroll=True)
        self._update_chat_for_move(info)
        self._refresh_result()

    # ------------------------------------------------------------------
    # Bot turn (background thread via BotWorker + natural perceived pacing)
    # ------------------------------------------------------------------

    def _pick_bot_think_target(self, is_book: bool) -> float:
        """Select a randomized minimum total perceived thinking duration for this bot."""
        tier = (
            self._bot.profile.tier
            if self._bot is not None
            else get_profile(self._opponent_name).tier
        )
        lo, hi = get_bot_think_range(tier, is_book=is_book)
        return random.uniform(lo, hi)

    def _update_bot(self, dt: float) -> None:
        # Advance perceived thinking timer only after the player's own animation lands
        if self._match_generation in self._pending_requests and not self._board.is_animating:
            self._bot_think_elapsed += max(0.0, dt)

        # Drain finished worker moves non-blockingly
        self._poll_bot_results()

        # Wait asynchronously for remaining minimum perceived thinking time AFTER player anim lands
        if self._ready_bot_move is not None:
            if not self._board.is_animating and self._bot_wait_remaining > 0.0:
                self._bot_wait_remaining = max(
                    0.0, self._bot_wait_remaining - max(0.0, dt)
                )
            # Never start bot animation while the player's animation is still running
            if not self._board.is_animating and self._bot_wait_remaining <= 0.0:
                move, _ = self._ready_bot_move
                self._ready_bot_move = None
                self._bot_think_elapsed = 0.0
                if (
                    self._is_game_active
                    and self._game is not None
                    and self._game.turn_color != self._player_color
                ):
                    self._apply_bot_move(move)
            return

        if self._should_request_bot_move():
            assert self._bot_worker is not None and self._game is not None
            # Start background thinking immediately (even while player animation runs)
            self._bot_think_elapsed = 0.0
            self._bot_worker.request_move(self._game.board)
            self._pending_requests.append(self._match_generation)

    def _should_request_bot_move(self) -> bool:
        return (
            self._bot_worker is not None
            and self._is_game_active
            and self._game is not None
            and self._game.turn_color != self._player_color
            and not self._bot_thinking
        )

    def _poll_bot_results(self) -> None:
        """Drain finished bot results without blocking; discard stale ones."""
        if self._bot_worker is None:
            return
        while self._pending_requests:
            failed = False
            try:
                move = self._bot_worker.poll_move()
            except Exception:
                logger.exception("Bot search failed; using fallback heuristic.")
                move, failed = None, True
            if move is None and not failed:
                return

            generation = self._pending_requests.popleft()
            if (
                generation != self._match_generation
                or not self._is_game_active
                or self._game is None
                or self._game.turn_color == self._player_color
            ):
                continue

            source = str(getattr(self._bot_worker, "last_move_source", ""))
            is_book = source == "book"
            compute_s = float(getattr(self._bot_worker, "last_compute_seconds", 0.0))
            actual_think = max(self._bot_think_elapsed, compute_s)
            target_think = self._pick_bot_think_target(is_book=is_book)
            self._bot_wait_remaining = max(0.0, target_think - actual_think)
            self._ready_bot_move = (move, failed)

    def _apply_bot_move(self, move: chess.Move | None) -> None:
        """Validate and apply the bot's move, falling back to the heuristic if needed."""
        assert self._game is not None
        info = self._game.make_uci(move.uci()) if move is not None else None
        if info is None:
            logger.warning("Bot returned illegal move %r; using fallback heuristic.", move)
            profile = self._bot.profile if self._bot is not None else get_profile(self._opponent_name)
            fallback = fallback_heuristic_move(self._game.board, profile)
            info = self._game.make_uci(fallback.uci())
        if info is None:
            return
        if self._game.is_game_over:
            self._clock.pause()
        else:
            self._clock.apply_increment(info.color)
        self._board.animate_move(info)
        self._sync_move_state(auto_scroll=True)
        self._update_chat_for_move(info)
        self._refresh_result()

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def draw(self) -> None:
        self.screen.fill(_BG)
        self._board.draw(self._board_surface)
        self._draw_player_bars()

        pygame.draw.rect(self.screen, _SIDEBAR, self._sidebar_rect)
        pygame.draw.line(
            self.screen,
            _BORDER,
            self._sidebar_rect.topleft,
            self._sidebar_rect.bottomleft,
            width=max(1, int(2 * self._scale)),
        )
        self._draw_profile()

        if self._game_started:
            self._draw_started_panel()
        else:
            self._draw_setup_panel()
        self._menu.draw(self.screen)
        if self._game_started and self._show_endgame_popup and not self._menu.is_open:
            self._draw_endgame_popup()

    def _draw_player_bars(self) -> None:
        """Draw the Top Opponent Bar and Bottom Player Bar aligned to the board."""
        board_rect = self._board.board_rect
        gap = max(6, int(10 * self._scale))
        diff = self._cached_material_diff if self._game is not None else 0
        player_is_white = self._player_color == chess.WHITE
        bot_adv = -diff if player_is_white else diff
        player_adv = diff if player_is_white else -diff
        bot_color = chess.BLACK if player_is_white else chess.WHITE

        # Top Opponent Bar
        top_cy = max(self._bar_avatar_size // 2 + 4, board_rect.top // 2)
        bot_av_rect = self._bar_bot_avatar.get_rect(midleft=(board_rect.left, top_cy))
        self.screen.blit(self._bar_bot_avatar, bot_av_rect)
        pygame.draw.circle(
            self.screen,
            _ACCENT if self._bot_thinking else _BORDER,
            bot_av_rect.center,
            self._bar_avatar_size // 2,
            width=max(1, int(2 * self._scale)),
        )
        opp_label_rect = self._bar_opp_label.get_rect(
            midleft=(bot_av_rect.right + gap, top_cy)
        )
        self.screen.blit(self._bar_opp_label, opp_label_rect)

        top_flag_rect = self._bar_flag_image.get_rect(
            midleft=(opp_label_rect.right + gap, top_cy)
        )
        self.screen.blit(self._bar_flag_image, top_flag_rect)
        cursor_x = top_flag_rect.right + gap

        if bot_adv > 0:
            adv_surf = self._get_cached_text(self._font_small, f"+{bot_adv}", _MUTED)
            self.screen.blit(adv_surf, adv_surf.get_rect(midleft=(cursor_x, top_cy)))
            cursor_x += adv_surf.get_width() + gap

        if self._bot_thinking:
            self._draw_thinking_pill((cursor_x, top_cy))

        # Top-right settings gear button aligned to board_rect.right
        self._btn_board_settings.draw(self.screen)
        self._draw_gear_icon(
            self.screen,
            self._btn_board_settings.base_rect.center,
            max(7, int(9 * self._scale)),
            _TEXT,
        )

        if self._game_started and self._clock.is_timed:
            bot_clock_right = self._btn_board_settings.base_rect.left - gap
            self._draw_clock_badge(
                bot_color,
                midright=(bot_clock_right, top_cy),
            )

        # Bottom Player Bar
        bot_cy = board_rect.bottom + (self.height - board_rect.bottom) // 2
        pl_av_rect = self._bar_player_avatar.get_rect(midleft=(board_rect.left, bot_cy))
        self.screen.blit(self._bar_player_avatar, pl_av_rect)
        pygame.draw.circle(
            self.screen,
            _BORDER,
            pl_av_rect.center,
            self._bar_avatar_size // 2,
            width=max(1, int(2 * self._scale)),
        )
        pl_label_rect = self._bar_player_label.get_rect(
            midleft=(pl_av_rect.right + gap, bot_cy)
        )
        self.screen.blit(self._bar_player_label, pl_label_rect)

        bot_flag_rect = self._bar_flag_image.get_rect(
            midleft=(pl_label_rect.right + gap, bot_cy)
        )
        self.screen.blit(self._bar_flag_image, bot_flag_rect)

        if player_adv > 0:
            adv_surf = self._get_cached_text(
                self._font_small, f"+{player_adv}", _MUTED
            )
            self.screen.blit(
                adv_surf, adv_surf.get_rect(midleft=(bot_flag_rect.right + gap, bot_cy))
            )

        if self._game_started and self._clock.is_timed:
            self._draw_clock_badge(
                self._player_color,
                midright=(board_rect.right, bot_cy),
            )

    def _draw_clock_badge(
        self, color: chess.Color, midright: tuple[int, int]
    ) -> None:
        """Render a Chess.com-style digital clock pill for *color*."""
        remaining = self._clock.remaining(color)
        time_str = format_clock_seconds(remaining)
        is_active = (
            self._is_game_active
            and self._game is not None
            and self._game.turn_color == color
            and not self._clock.paused
        )
        is_low = remaining < 10.0 and is_active

        if is_low:
            bg_color = THEME_CLOCK_LOW_BG
            text_color = THEME_CLOCK_LOW_TEXT
            border_col = THEME_CLOCK_LOW_TEXT
        elif is_active:
            bg_color = THEME_CLOCK_ACTIVE_BG
            text_color = THEME_CLOCK_ACTIVE_TEXT
            border_col = THEME_GREEN
        else:
            bg_color = THEME_CLOCK_INACTIVE_BG
            text_color = THEME_CLOCK_INACTIVE_TEXT
            border_col = THEME_BORDER

        text_surf = self._get_cached_text(self._font_clock, time_str, text_color)
        pad_x = max(10, int(SPACE_SM * self._scale))
        pad_y = max(4, int(6 * self._scale))
        badge_w = max(int(82 * self._scale), text_surf.get_width() + pad_x * 2)
        badge_h = max(int(30 * self._scale), text_surf.get_height() + pad_y * 2)
        badge_rect = pygame.Rect(0, 0, badge_w, badge_h)
        badge_rect.midright = midright
        radius = max(5, int(RADIUS_SM * self._scale))

        pygame.draw.rect(self.screen, bg_color, badge_rect, border_radius=radius)
        pygame.draw.rect(
            self.screen,
            border_col,
            badge_rect,
            width=max(1, int(2 * self._scale)) if is_active else 1,
            border_radius=radius,
        )
        self.screen.blit(text_surf, text_surf.get_rect(center=badge_rect.center))

    def _draw_thinking_pill(self, midleft: tuple[int, int]) -> None:
        dots = "." * ((pygame.time.get_ticks() // 350) % 3 + 1)
        surf = self._get_cached_text(self._font_small, f"thinking{dots}", _ACCENT)
        pad_x = max(7, int(8 * self._scale))
        pad_y = max(3, int(4 * self._scale))
        pill = pygame.Rect(
            midleft[0],
            midleft[1] - (surf.get_height() + pad_y * 2) // 2,
            surf.get_width() + pad_x * 2,
            surf.get_height() + pad_y * 2,
        )
        pygame.draw.rect(
            self.screen, THEME_PANEL_ALT, pill, border_radius=max(6, int(8 * self._scale))
        )
        pygame.draw.rect(
            self.screen,
            _ACCENT,
            pill,
            width=1,
            border_radius=max(6, int(8 * self._scale)),
        )
        self.screen.blit(surf, surf.get_rect(center=pill.center))

    def _draw_profile(self) -> None:
        if self._game_started:
            self.screen.blit(self._header_play_surf, self._header_pos)
        else:
            for index, header_line in enumerate(self._header_setup_lines):
                self.screen.blit(
                    header_line,
                    (
                        self._header_pos[0],
                        self._header_pos[1]
                        + index * self._font_setup_header.get_linesize(),
                    ),
                )
        self.screen.blit(self._avatar, self._avatar_rect)
        pygame.draw.circle(
            self.screen,
            _ACCENT if self._bot_thinking else _BORDER,
            self._avatar_rect.center,
            self._avatar_size // 2,
            width=max(1, int(2 * self._scale)),
        )
        if not self._game_started:
            self.screen.blit(self._name_surface, self._name_pos)
            self.screen.blit(self._elo_surface, self._elo_pos)
            self.screen.blit(self._flag_image, self._flag_rect)
        elif self._bot_thinking:
            self._draw_thinking_pill(
                (self._avatar_rect.left, self._avatar_rect.bottom + max(8, int(10 * self._scale)))
            )

        bubble_msg = self._chat_text if self._game_started else self._opponent_tagline
        if bubble_msg != self._last_bubble_msg:
            self._last_bubble_msg = bubble_msg
            self._bubble_text_anim_elapsed = (
                BUBBLE_TEXT_CHANGE_DURATION if self._reduced_motion else 0.0
            )
        self._draw_speech_bubble(bubble_msg)

    def _speech_bubble_progress(self) -> tuple[float, float]:
        """Return `(bubble_t, text_t)` animation progress values in [0.0, 1.0]."""
        if self._reduced_motion:
            return 1.0, 1.0
        bubble_t = min(
            1.0, max(0.0, self._bubble_anim_elapsed / max(0.01, BUBBLE_ANIM_DURATION))
        )
        if self._bubble_anim_elapsed <= BUBBLE_TEXT_DELAY:
            entrance_text_t = 0.0
        else:
            entrance_text_t = min(
                1.0,
                (self._bubble_anim_elapsed - BUBBLE_TEXT_DELAY)
                / max(0.01, BUBBLE_TEXT_DURATION),
            )
        change_text_t = min(
            1.0,
            max(
                0.0,
                self._bubble_text_anim_elapsed
                / max(0.01, BUBBLE_TEXT_CHANGE_DURATION),
            ),
        )
        return bubble_t, min(entrance_text_t, change_text_t)

    def _draw_speech_bubble(self, bubble_msg: str) -> None:
        """Render the opponent speech bubble with Chess.com-style emergence & text reveal."""
        bubble_t, text_t = self._speech_bubble_progress()
        if bubble_t <= 0.005:
            return

        inner_w = max(1, self._tagline_rect.width - self._bubble_padding_x * 2)
        bubble_lines = self._get_wrapped_lines(
            bubble_msg,
            self._font_setup_bubble,
            THEME_BUBBLE_TEXT,
            inner_w,
        )
        line_h = self._font_setup_bubble.get_linesize()
        bubble_rect = self._tagline_rect.copy()
        bubble_rect.height = (
            max(1, len(bubble_lines)) * line_h + self._bubble_padding_y * 2
        )
        bubble_rect.top = self._avatar_rect.top

        tail_y = self._avatar_rect.centery
        tail_width = max(4, bubble_rect.left - self._avatar_rect.right)
        tail_half_h = max(
            SPACE_XS, round(SETUP_BUBBLE_TAIL_HALF_HEIGHT * self._scale)
        )
        bubble_rad = max(10, round((RADIUS_BUBBLE + 2) * self._scale))

        # Compose bubble + tail onto a crisp local SRCALPHA canvas anchored at the tail tip
        shadow_pad = max(3, round(4 * self._scale))
        canvas_w = tail_width + bubble_rect.width + shadow_pad * 2
        canvas_h = bubble_rect.height + shadow_pad * 2
        canvas = pygame.Surface((canvas_w, canvas_h), pygame.SRCALPHA)

        local_rect = pygame.Rect(
            shadow_pad + tail_width,
            shadow_pad,
            bubble_rect.width,
            bubble_rect.height,
        )
        local_tail_y = max(
            local_rect.top + tail_half_h + 2,
            min(local_rect.bottom - tail_half_h - 2, shadow_pad + (tail_y - bubble_rect.top)),
        )
        tail_poly = (
            (local_rect.left + SETUP_BUBBLE_TAIL_INSET + 2, local_tail_y - tail_half_h),
            (shadow_pad, local_tail_y),
            (local_rect.left + SETUP_BUBBLE_TAIL_INSET + 2, local_tail_y + tail_half_h),
        )

        # Subtle soft drop shadow beneath the bubble
        shadow_rect = local_rect.move(0, max(2, round(2 * self._scale)))
        pygame.draw.rect(
            canvas, (14, 13, 12, 46), shadow_rect, border_radius=bubble_rad
        )
        # Seamless white bubble body + tail pointing to opponent avatar
        pygame.draw.polygon(canvas, (*_BUBBLE, 255), tail_poly)
        pygame.draw.rect(
            canvas, (*_BUBBLE, 255), local_rect, border_radius=bubble_rad
        )

        # Step 2: Delayed dialogue text fade-in + subtle upward glide (clipped to bubble interior)
        if text_t > 0.01:
            text_ease = ease_progress(text_t, "ease_out_cubic")
            text_alpha = int(255 * text_ease)
            text_slide_y = round(BUBBLE_TEXT_SLIDE_PX * (1.0 - text_ease) * self._scale)
            text_block_h = len(bubble_lines) * line_h
            avail_h = max(1, local_rect.height - self._bubble_padding_y * 2)
            base_y = (
                local_rect.top
                + self._bubble_padding_y
                + max(0, (avail_h - text_block_h) // 2)
                + text_slide_y
            )
            text_layer = pygame.Surface(canvas.get_size(), pygame.SRCALPHA)
            clip_rect = local_rect.inflate(
                -self._bubble_padding_x, -max(2, self._bubble_padding_y // 2)
            )
            text_layer.set_clip(clip_rect)
            for idx, line_surf in enumerate(bubble_lines):
                text_layer.blit(
                    line_surf,
                    (local_rect.left + self._bubble_padding_x, base_y + idx * line_h),
                )
            text_layer.set_clip(None)
            text_layer.set_alpha(text_alpha)
            canvas.blit(text_layer, (0, 0))

        # Step 1: Smooth spring scale + opacity emerging from avatar-facing tail tip
        pivot_x = bubble_rect.left - tail_width
        pivot_y = bubble_rect.top + (local_tail_y - shadow_pad)
        if bubble_t < 0.999:
            spring = _ease_out_back_soft(bubble_t)
            scale = max(0.05, BUBBLE_SCALE_START + (1.0 - BUBBLE_SCALE_START) * spring)
            opacity = min(1.0, ease_progress(min(1.0, bubble_t * 1.55), "ease_out_cubic"))
            sw = max(2, round(canvas_w * scale))
            sh = max(2, round(canvas_h * scale))
            scaled = pygame.transform.smoothscale(canvas, (sw, sh))
            scaled.set_alpha(int(255 * opacity))
            draw_x = round(pivot_x - shadow_pad * scale)
            draw_y = round(pivot_y - local_tail_y * scale)
            self.screen.blit(scaled, (draw_x, draw_y))
        else:
            self.screen.blit(
                canvas, (bubble_rect.left - tail_width - shadow_pad, bubble_rect.top - shadow_pad)
            )

    def _draw_setup_panel(self) -> None:
        padding = max(SPACE_SM, int(18 * self._scale))
        content_left = self._sidebar_rect.left + padding

        # Modern TIME CONTROL heading (subtle timer icon + tracked uppercase label)
        self.screen.blit(
            self._tc_heading_surf,
            (content_left, self._no_timer_label_y),
        )
        self._draw_time_dropdown_trigger()

        # Resolve dropdown animation progress (renders immediately if toggled without update)
        anim_t = (
            self._time_dropdown_anim
            if (self._time_dropdown_anim > 0.01 or not self._time_dropdown_open)
            else 1.0
        )

        # Draw lower controls underneath while closed or mid-transition
        if not self._time_dropdown_open or anim_t < 0.99:
            self._draw_label(
                "PLAY AS",
                (content_left, self._side_label_y),
                font=self._font_setup_label,
            )
            for button in self._side_buttons.values():
                button.draw(self.screen)

            self._btn_play.draw(self.screen)

        if self._time_dropdown_open or self._time_dropdown_anim > 0.01:
            self._draw_time_dropdown_menu(anim_t)

    def _draw_time_dropdown_trigger(self) -> None:
        btn = self._btn_time_dropdown
        btn.accent = self._time_dropdown_open
        rect = btn.base_rect
        press_offset = round(btn._pressed_t) if btn.press_feedback else 0
        draw_rect = rect.move(0, press_offset)
        radius = self._tc_panel_radius

        anim_t = (
            self._time_dropdown_anim
            if (self._time_dropdown_anim > 0.01 or not self._time_dropdown_open)
            else 1.0
        )

        # Soft diffused elevation shadow (Chess.com warm charcoal card depth)
        self.screen.blit(
            self._tc_trigger_shadow,
            self._tc_trigger_shadow.get_rect(center=draw_rect.center),
        )

        # Warm charcoal fill matching Chess.com reference (#3A3835 -> #44413D hover)
        base_fill = _lerp_tc_rgb(THEME_PANEL, TC_TRIGGER_HOVER, btn._hover_t * 0.90)
        fill_col = _lerp_tc_rgb(base_fill, TC_TRIGGER_OPEN, anim_t * 0.55)
        pygame.draw.rect(self.screen, fill_col, draw_rect, border_radius=radius)

        # Subtle warm top edge highlight
        if draw_rect.width > radius * 2 + 4:
            pygame.draw.line(
                self.screen,
                _lerp_tc_rgb(TC_TRIGGER_SPECULAR, TC_ACCENT, 0.18 * anim_t),
                (draw_rect.left + radius, draw_rect.top + 1),
                (draw_rect.right - radius, draw_rect.top + 1),
                width=1,
            )

        # Clean subtle resting border; highlights cleanly in emerald when expanded
        border_col = _lerp_tc_rgb(
            _lerp_tc_rgb(TC_TRIGGER_BORDER_IDLE, TC_BTN_BORDER_HOVER, btn._hover_t),
            TC_ACCENT,
            anim_t,
        )
        border_w = 2 if (self._time_dropdown_open or anim_t > 0.5) else 1
        btn.border_color = border_col
        pygame.draw.rect(
            self.screen, border_col, draw_rect, width=border_w, border_radius=radius
        )

        # Chess.com-style centered [icon + bold label (Category)] with crisp right chevron
        pad_x = max(10, round(16 * self._scale))
        cy = draw_rect.centery
        category = get_time_control_category(self._selected_time_control)
        chevron_r = max(5, round(7 * self._scale))
        chevron_cx = draw_rect.right - pad_x - chevron_r
        left_limit = draw_rect.left + pad_x
        right_limit = chevron_cx - chevron_r - max(8, round(10 * self._scale))

        if category in self._trigger_icons:
            icon = self._trigger_icons[category]
        else:
            icon = self._tc_trigger_no_timer_icon

        icon_w = icon.get_width()
        icon_gap = max(7, round(10 * self._scale))
        max_label_w = max(1, (right_limit - left_limit) - icon_w - icon_gap)
        label_text = self._ellipsize(
            format_time_control_label(self._selected_time_control),
            self._font_tc_trigger,
            max_label_w,
        )
        label_surf = self._get_cached_text(
            self._font_tc_trigger, label_text, TC_TEXT_PRIMARY
        )

        group_w = icon_w + icon_gap + label_surf.get_width()
        group_left = draw_rect.centerx - group_w // 2
        # Clamp safely between left padding and right chevron on narrow screens
        if group_left + group_w > right_limit:
            group_left = max(left_limit, right_limit - group_w)
        elif group_left < left_limit:
            group_left = left_limit

        self.screen.blit(icon, icon.get_rect(midleft=(group_left, cy)))
        self.screen.blit(
            label_surf,
            label_surf.get_rect(midleft=(group_left + icon_w + icon_gap, cy)),
        )

        # Clean bold Chess.com-style right chevron (rotates smoothly when expanded)
        chevron_col = _lerp_tc_rgb(TC_TEXT_SECONDARY, TC_TEXT_PRIMARY, max(anim_t, btn._hover_t * 0.5))
        self._draw_tc_chevron(self.screen, (chevron_cx, cy), chevron_r, chevron_col, anim_t)

    def _draw_time_dropdown_menu(self, anim_t: float = 1.0) -> None:
        panel = self._time_dropdown_panel_rect
        eased = ease_progress(anim_t, "ease_out_cubic")
        if eased <= 0.01:
            return

        # Render complete panel content onto a clean local surface for crisp animation & clipping
        w, h = panel.size
        panel_surf = pygame.Surface((w, h), pygame.SRCALPHA)
        panel_surf.blit(self._tc_panel_card, (0, 0))

        ox, oy = panel.left, panel.top

        # Subtle divider separating No Timer from timed categories
        div_y = self._tc_divider_y - oy
        pygame.draw.line(
            panel_surf,
            TC_DIVIDER_COLOR,
            (self._dropdown_inner_left - ox, div_y),
            (self._dropdown_inner_left - ox + self._tc_grid_width, div_y),
            width=1,
        )

        # Category headers (aligned left with icons vertically centered)
        selected_cat = get_time_control_category(self._selected_time_control)
        icon_gap = max(6, round(8 * self._scale))
        for group_key, y in self._group_label_positions.items():
            local_y = y - oy
            row_cy = local_y + self._tc_header_row_h // 2
            icon = self._group_icons[group_key]
            icon_rect = icon.get_rect(midleft=(self._dropdown_inner_left - ox, row_cy))
            panel_surf.blit(icon, icon_rect)

            is_active_cat = group_key == selected_cat
            lbl_col = TC_TEXT_PRIMARY if is_active_cat else TC_TEXT_SECONDARY
            lbl_surf = self._get_cached_text(
                self._font_tc_category, group_key.title(), lbl_col
            )
            panel_surf.blit(
                lbl_surf,
                lbl_surf.get_rect(midleft=(icon_rect.right + icon_gap, row_cy)),
            )

        # Draw No Timer + all 9 timer preset buttons with sleek dark-emerald styling
        for value, button in self._time_buttons.items():
            self._draw_tc_option_button(panel_surf, value, button, ox, oy)

        # Apply smooth expand/collapse reveal (slide + height clip + fade)
        slide_y = round(-6.0 * (1.0 - eased) * self._scale)
        vis_h = max(1, round(h * (0.28 + 0.72 * eased)))

        shadow = self._tc_panel_shadow
        shadow_rect = shadow.get_rect(center=(panel.centerx, panel.centery + slide_y))
        if eased < 0.99:
            shadow_copy = shadow.copy()
            shadow_copy.set_alpha(int(255 * eased))
            clip_shadow = pygame.Rect(
                shadow_rect.left,
                shadow_rect.top,
                shadow_rect.width,
                min(shadow_rect.height, vis_h + 28),
            )
            prev_clip = self.screen.get_clip()
            self.screen.set_clip(clip_shadow)
            self.screen.blit(shadow_copy, shadow_rect.topleft)
            self.screen.set_clip(prev_clip)

            panel_surf.set_alpha(int(255 * min(1.0, eased * 1.15)))
            self.screen.blit(
                panel_surf,
                (panel.left, panel.top + slide_y),
                area=pygame.Rect(0, 0, w, vis_h),
            )
        else:
            self.screen.blit(shadow, shadow_rect.topleft)
            self.screen.blit(panel_surf, panel.topleft)

    def _draw_tc_option_button(
        self,
        surface: pygame.Surface,
        value: str,
        button: Button,
        ox: int,
        oy: int,
    ) -> None:
        """Render a single sleek Time Control option button onto the panel surface."""
        sel_t = self._time_btn_select_t.get(
            value, 1.0 if value == self._selected_time_control else 0.0
        )
        # Ensure immediate state reflection when drawn without intermediate update()
        if value == self._selected_time_control and sel_t < 0.05 and not self._time_dropdown_anim:
            sel_t = 1.0

        rect = button.base_rect.move(-ox, -oy)
        press_y = round(button._pressed_t) if button.press_feedback else 0
        if press_y:
            rect = rect.move(0, press_y)

        radius = min(self._tc_btn_radius, max(4, rect.height // 2))
        hover_t = button._hover_t

        # Layered fill: secondary dark panel -> subtle emerald tint when selected
        idle_fill = _lerp_tc_rgb(TC_BTN_BG, TC_BTN_HOVER, hover_t)
        active_fill = _lerp_tc_rgb(TC_SELECTED_BG, TC_SELECTED_HOVER, hover_t)
        fill_col = _lerp_tc_rgb(idle_fill, active_fill, sel_t)
        pygame.draw.rect(surface, fill_col, rect, border_radius=radius)

        # Subtle warm inner top specular edge
        if rect.width > radius * 2 + 2:
            top_line_col = _lerp_tc_rgb(
                (70, 67, 63), (48, 128, 98), sel_t
            )
            pygame.draw.line(
                surface,
                top_line_col,
                (rect.left + radius, rect.top + 1),
                (rect.right - radius, rect.top + 1),
                width=1,
            )

        # Refined warm border -> crisp emerald (#10B981) on active selection
        idle_border = _lerp_tc_rgb(TC_BTN_BORDER, TC_BTN_BORDER_HOVER, hover_t)
        border_col = _lerp_tc_rgb(idle_border, TC_ACCENT, sel_t)
        border_w = 2 if sel_t > 0.35 else 1
        pygame.draw.rect(surface, border_col, rect, width=border_w, border_radius=radius)

        # Label text + non-color selection indicator
        txt_col = _lerp_tc_rgb(TC_TEXT_PRIMARY, TC_TEXT_SELECTED, sel_t)
        lbl_surf = self._get_cached_text(self._font_tc_choice, value, txt_col)

        if value == "No Timer":
            # Centered icon + label; adds crisp check icon on right when selected
            icon = self._tc_no_timer_icon_active if sel_t > 0.45 else self._tc_no_timer_icon_idle
            gap = max(6, round(8 * self._scale))
            total_w = icon.get_width() + gap + lbl_surf.get_width()
            sx = rect.centerx - total_w // 2
            surface.blit(icon, icon.get_rect(midleft=(sx, rect.centery)))
            surface.blit(
                lbl_surf,
                lbl_surf.get_rect(midleft=(sx + icon.get_width() + gap, rect.centery)),
            )
            if sel_t > 0.05:
                chk = self._tc_check_icon.copy()
                chk.set_alpha(int(255 * sel_t))
                pad_r = max(9, round(12 * self._scale))
                surface.blit(
                    chk, chk.get_rect(midright=(rect.right - pad_r, rect.centery))
                )
        else:
            # Show non-color emerald indicator bar + checkmark/dot on selected preset
            if sel_t > 0.05:
                # Subtle left emerald accent pill bar (works cleanly even on narrow buttons)
                bar_h = max(12, int(rect.height * 0.48))
                bar_w = max(3, round(3 * self._scale))
                bar_rect = pygame.Rect(
                    rect.left + max(4, round(5 * self._scale)),
                    rect.centery - bar_h // 2,
                    bar_w,
                    bar_h,
                )
                bar_surf = pygame.Surface(bar_rect.size, pygame.SRCALPHA)
                pygame.draw.rect(
                    bar_surf,
                    (*TC_ACCENT, int(255 * sel_t)),
                    bar_surf.get_rect(),
                    border_radius=bar_w,
                )
                surface.blit(bar_surf, bar_rect.topleft)

                # If button has enough horizontal room, also render crisp emerald checkmark
                chk = self._tc_check_icon
                if rect.width >= lbl_surf.get_width() + chk.get_width() + 26:
                    chk_copy = chk.copy()
                    chk_copy.set_alpha(int(255 * sel_t))
                    pad_r = max(7, round(9 * self._scale))
                    surface.blit(
                        chk_copy,
                        chk_copy.get_rect(midright=(rect.right - pad_r, rect.centery)),
                    )

            surface.blit(lbl_surf, lbl_surf.get_rect(center=rect.center))

    def _build_tc_heading_surface(self) -> pygame.Surface:
        """Build cached 'TIME CONTROL' heading using custom transparent stopwatch icon."""
        # Crisp 28-34px icon size (transparent background, never inside a square box)
        icon_sz = max(28, min(34, round(32 * max(0.85, self._scale))))
        icon_surf = self._load_tc_stopwatch_icon(icon_sz)
        tracking = max(1, round(1.5 * self._scale))
        text = "TIME CONTROL"
        glyphs = [
            self._font_tc_heading.render(ch, True, TC_TEXT_SECONDARY) for ch in text
        ]
        txt_w = sum(g.get_width() for g in glyphs) + tracking * max(0, len(glyphs) - 1)
        txt_h = max(g.get_height() for g in glyphs)
        gap = max(7, round(9 * self._scale))
        total_w = icon_surf.get_width() + gap + txt_w
        total_h = max(icon_surf.get_height(), txt_h)

        surf = pygame.Surface((total_w, total_h), pygame.SRCALPHA)
        surf.blit(icon_surf, icon_surf.get_rect(midleft=(0, total_h // 2)))
        x = icon_surf.get_width() + gap
        for g in glyphs:
            surf.blit(g, g.get_rect(midleft=(x, total_h // 2)))
            x += g.get_width() + tracking
        return surf

    @classmethod
    def _load_tc_stopwatch_icon(cls, size: int) -> pygame.Surface:
        """Load assets/timecontrol.png preserving transparency with crisp multi-step scaling."""
        cache = getattr(cls, "_tc_stopwatch_cache", None)
        if cache is None:
            cache = {}
            cls._tc_stopwatch_cache = cache
        if size in cache:
            return cache[size]

        icon_path = ASSETS_DIR / TC_STOPWATCH_ICON_FILE
        if icon_path.is_file():
            try:
                raw = pygame.image.load(str(icon_path))
                if pygame.display.get_surface() is not None:
                    raw = raw.convert_alpha()
                else:
                    raw = raw.copy()
                # Trim empty transparent borders so the stopwatch fills its target size cleanly
                bbox = raw.get_bounding_rect(min_alpha=12)
                if bbox.width > 0 and bbox.height > 0:
                    raw = raw.subsurface(bbox).copy()
                # Preserve exact aspect ratio and downsample in crisp halving steps
                rw, rh = raw.get_size()
                scale = size / max(1, max(rw, rh))
                tw = max(1, round(rw * scale))
                th = max(1, round(rh * scale))
                cur = raw
                while cur.get_width() > tw * 2 and cur.get_height() > th * 2:
                    cur = pygame.transform.smoothscale(
                        cur, (max(tw, cur.get_width() // 2), max(th, cur.get_height() // 2))
                    )
                final = pygame.transform.smoothscale(cur, (tw, th))
                cache[size] = final
                return final
            except Exception as exc:
                logger.warning("Failed to load %s: %s", icon_path, exc)

        fallback = cls._build_tc_clock_icon(size)
        cache[size] = fallback
        return fallback

    @staticmethod
    def _build_tc_clock_icon(size: int) -> pygame.Surface:
        """Create a crisp 2x-supersampled minimalist stopwatch icon for the heading."""
        hi = max(24, size * 2)
        surf = pygame.Surface((hi, hi), pygame.SRCALPHA)
        cx, cy = hi // 2, int(hi * 0.56)
        r = int(hi * 0.36)
        stroke = max(2, hi // 11)
        # Top stopwatch crown
        pygame.draw.line(
            surf, TC_TEXT_SECONDARY, (cx - r // 2, cy - r - stroke), (cx + r // 2, cy - r - stroke), stroke
        )
        pygame.draw.line(
            surf, TC_TEXT_SECONDARY, (cx, cy - r - stroke), (cx, cy - r + 1), stroke
        )
        # Dial ring + crisp emerald minute hand
        pygame.draw.circle(surf, TC_TEXT_SECONDARY, (cx, cy), r, width=stroke)
        pygame.draw.line(surf, TC_ACCENT, (cx, cy), (cx, cy - int(r * 0.54)), stroke)
        pygame.draw.line(
            surf, TC_ACCENT, (cx, cy), (cx + int(r * 0.42), cy + int(r * 0.12)), stroke
        )
        return pygame.transform.smoothscale(surf, (size, size))

    @staticmethod
    def _build_tc_no_timer_icon(
        size: int, color: tuple[int, int, int]
    ) -> pygame.Surface:
        """Create a crisp 2x-supersampled untimed clock-slash icon."""
        hi = max(28, size * 2)
        surf = pygame.Surface((hi, hi), pygame.SRCALPHA)
        cx, cy = hi // 2, hi // 2
        r = int(hi * 0.38)
        stroke = max(2, hi // 10)
        pygame.draw.circle(surf, color, (cx, cy), r, width=stroke)
        pygame.draw.line(surf, color, (cx, cy), (cx, cy - int(r * 0.52)), stroke)
        pygame.draw.line(surf, color, (cx, cy), (cx + int(r * 0.44), cy), stroke)
        slash = int(r * 0.88)
        pygame.draw.line(
            surf, color, (cx - slash, cy + slash), (cx + slash, cy - slash), stroke
        )
        return pygame.transform.smoothscale(surf, (size, size))

    @staticmethod
    def _build_tc_check_icon(size: int) -> pygame.Surface:
        """Create a crisp 2x-supersampled emerald checkmark icon for selected presets."""
        hi = max(22, size * 2)
        surf = pygame.Surface((hi, hi), pygame.SRCALPHA)
        stroke = max(3, hi // 7)
        pts = (
            (int(hi * 0.16), int(hi * 0.54)),
            (int(hi * 0.40), int(hi * 0.78)),
            (int(hi * 0.84), int(hi * 0.24)),
        )
        pygame.draw.lines(surf, TC_ACCENT, False, pts, stroke)
        return pygame.transform.smoothscale(surf, (size, size))

    @staticmethod
    def _build_tc_panel_surfaces(
        size: tuple[int, int], radius: int, shadow_pad: int = 12
    ) -> tuple[pygame.Surface, pygame.Surface]:
        """Build cached soft warm-charcoal shadow + rounded layered dark panel surface."""
        w, h = max(1, size[0]), max(1, size[1])
        shadow = pygame.Surface((w + shadow_pad * 2, h + shadow_pad * 2), pygame.SRCALPHA)
        for spread, alpha, dy in ((8, 14, 5), (4, 24, 4), (1, 38, 2)):
            if spread > shadow_pad:
                continue
            pygame.draw.rect(
                shadow,
                (*TC_SHADOW_RGB, alpha),
                (
                    shadow_pad - spread,
                    shadow_pad - spread + dy,
                    w + spread * 2,
                    h + spread * 2,
                ),
                border_radius=radius + spread,
            )

        card = pygame.Surface((w, h), pygame.SRCALPHA)
        for y in range(h):
            t = y / max(1, h - 1)
            col = _lerp_tc_rgb(TC_PANEL_TOP, TC_PANEL_BG, t)
            pygame.draw.line(card, (*col, 250), (0, y), (w - 1, y))
        mask = pygame.Surface((w, h), pygame.SRCALPHA)
        pygame.draw.rect(mask, (255, 255, 255, 255), (0, 0, w, h), border_radius=radius)
        card.blit(mask, (0, 0), special_flags=pygame.BLEND_RGBA_MULT)

        # Delicate warm top edge specular highlight + refined warm border
        if w > radius * 2 + 4:
            pygame.draw.line(
                card, TC_PANEL_SPECULAR, (radius, 1), (w - radius, 1), width=1
            )
        pygame.draw.rect(
            card, TC_BORDER_SUBTLE, (0, 0, w, h), width=1, border_radius=radius
        )
        return shadow, card

    @staticmethod
    def _draw_tc_chevron(
        surf: pygame.Surface,
        center: tuple[int, int],
        r: int,
        color: tuple[int, int, int],
        open_t: float,
    ) -> None:
        """Draw a crisp antialiased chevron smoothly interpolating between closed (v) and open (^)."""
        eased = ease_progress(open_t, "ease_out_cubic")
        hi = max(24, r * 4)
        tmp = pygame.Surface((hi, hi), pygame.SRCALPHA)
        cx, cy = hi // 2, hi // 2
        span = int(hi * 0.34)
        # Smooth vertical flip from +half_h (down) to -half_h (up)
        half_h = int(hi * 0.17 * (1.0 - 2.0 * eased))
        pts = (
            (cx - span, cy - half_h),
            (cx, cy + half_h),
            (cx + span, cy - half_h),
        )
        stroke = max(4, hi // 7)
        pygame.draw.lines(tmp, color, False, pts, stroke)
        for px, py in pts:
            pygame.draw.circle(tmp, color, (px, py), stroke // 2)
        out_sz = max(14, r * 2 + 2)
        scaled = pygame.transform.smoothscale(tmp, (out_sz, out_sz))
        surf.blit(scaled, scaled.get_rect(center=center))

    def _draw_started_panel(self) -> None:
        # 1. Opening banner
        pygame.draw.rect(
            self.screen,
            _PANEL,
            self._opening_rect,
            border_radius=max(6, int(8 * self._scale)),
        )
        pygame.draw.rect(
            self.screen,
            _BORDER,
            self._opening_rect,
            width=1,
            border_radius=max(6, int(8 * self._scale)),
        )
        pad_x = max(8, int(12 * self._scale))
        opening_text = self._ellipsize(
            f"Opening: {self._current_opening}",
            self._font_small,
            self._opening_rect.width - pad_x * 2,
        )
        opening_surf = self._get_cached_text(self._font_small, opening_text, _TEXT)
        self.screen.blit(
            opening_surf,
            opening_surf.get_rect(
                midleft=(self._opening_rect.left + pad_x, self._opening_rect.centery)
            ),
        )

        # 2. Scrollable two-column SAN move list
        self._draw_move_list()

        # 3. Footer status / temporary notice
        if self._notice_text and self._notice_timer > 0:
            footer_text, footer_color = self._notice_text, _ACCENT
        else:
            footer_text = self._status_text()
            footer_color = _ACCENT if self._result_text else _MUTED
        footer_surf = self._get_cached_text(
            self._font_small,
            self._ellipsize(footer_text, self._font_small, self._footer_rect.width),
            footer_color,
        )
        self.screen.blit(
            footer_surf,
            footer_surf.get_rect(
                midleft=(self._footer_rect.left, self._footer_rect.centery)
            ),
        )

        # 4. Lower control button (Resign during match, New Game when over)
        if self._game is not None and self._game.is_game_over:
            self._draw_labeled_icon_button(
                self._btn_resign, "New Game", icon_surface=self._action_icons["newgame"]
            )
        else:
            self._draw_labeled_icon_button(
                self._btn_resign, "Resign", icon_surface=self._action_icons["resign"]
            )

    def _draw_move_list(self) -> None:
        rect = self._moves_rect
        radius = max(6, int(9 * self._scale))
        pygame.draw.rect(self.screen, THEME_BG, rect, border_radius=radius)
        pygame.draw.rect(self.screen, _BORDER, rect, width=1, border_radius=radius)

        san_moves = self._game.san_moves if self._game is not None else []
        if not san_moves:
            hint = self._get_cached_text(
                self._font_body, "Make a move to start notation", _MUTED
            )
            self.screen.blit(hint, hint.get_rect(center=rect.center))
            return

        pad_x = max(8, int(12 * self._scale))
        num_w = max(30, int(38 * self._scale))
        col_w = (rect.width - pad_x * 2 - num_w - int(10 * self._scale)) // 2
        white_x = rect.left + pad_x + num_w
        black_x = white_x + col_w + int(4 * self._scale)
        last_ply = len(san_moves) - 1

        prev_clip = self.screen.get_clip()
        self.screen.set_clip(rect.inflate(-2, -2))

        total_pairs = (len(san_moves) + 1) // 2
        for pair_idx in range(total_pairs):
            row_y = (
                rect.top
                + self._moves_pad_y
                + pair_idx * self._move_row_h
                - self._move_scroll_y
            )
            if row_y + self._move_row_h < rect.top or row_y > rect.bottom:
                continue

            row_rect = pygame.Rect(
                rect.left + 2, row_y, rect.width - 4, self._move_row_h
            )
            if pair_idx % 2 == 1:
                pygame.draw.rect(self.screen, _PANEL_ALT, row_rect)

            num_surf = self._get_cached_text(
                self._font_moves, f"{pair_idx + 1}.", _MUTED
            )
            self.screen.blit(
                num_surf,
                num_surf.get_rect(midleft=(rect.left + pad_x, row_rect.centery)),
            )

            w_ply = pair_idx * 2
            self._draw_san_cell(
                san_moves[w_ply],
                pygame.Rect(white_x, row_y + 2, col_w - 4, self._move_row_h - 4),
                active=(w_ply == last_ply),
            )

            b_ply = w_ply + 1
            if b_ply < len(san_moves):
                self._draw_san_cell(
                    san_moves[b_ply],
                    pygame.Rect(black_x, row_y + 2, col_w - 4, self._move_row_h - 4),
                    active=(b_ply == last_ply),
                )

        self.screen.set_clip(prev_clip)

        max_scroll = self._max_move_scroll()
        if max_scroll > 0:
            track_w = max(4, int(5 * self._scale))
            track_rect = pygame.Rect(
                rect.right - track_w - 4,
                rect.top + 6,
                track_w,
                rect.height - 12,
            )
            total_h = total_pairs * self._move_row_h + self._moves_pad_y * 2
            thumb_h = max(20, int(track_rect.height * (rect.height / total_h)))
            ratio = self._move_scroll_y / max_scroll
            thumb_y = track_rect.top + int((track_rect.height - thumb_h) * ratio)
            pygame.draw.rect(
                self.screen,
                THEME_BORDER,
                (track_rect.left, thumb_y, track_w, thumb_h),
                border_radius=track_w // 2,
            )

    def _get_san_surface(self, san: str, active: bool) -> pygame.Surface:
        key = (san, active)
        cached = self._san_surface_cache.get(key)
        if cached is not None:
            return cached

        text_color = _PILL_TEXT if active else _TEXT
        figurine = format_san_figurine(san)
        glyphs = [
            (
                self._font_symbol
                if ch in _PIECE_FIGURINES.values()
                else self._font_moves
            ).render(ch, True, text_color)
            for ch in figurine
        ]
        width = max(1, sum(g.get_width() for g in glyphs))
        height = max(
            (g.get_height() for g in glyphs),
            default=self._font_moves.get_linesize(),
        )
        surf = pygame.Surface((width, height), pygame.SRCALPHA)
        x = 0
        for g in glyphs:
            surf.blit(g, (x, (height - g.get_height()) // 2))
            x += g.get_width()
        self._san_surface_cache[key] = surf
        return surf

    def _draw_san_cell(self, san: str, cell_rect: pygame.Rect, active: bool) -> None:
        if active:
            pill_r = max(5, int(6 * self._scale))
            pygame.draw.rect(self.screen, _ACTIVE_PILL, cell_rect, border_radius=pill_r)
            pygame.draw.rect(
                self.screen, _ACCENT, cell_rect, width=1, border_radius=pill_r
            )
        text_surf = self._get_san_surface(san, active)
        self.screen.blit(
            text_surf,
            text_surf.get_rect(
                midleft=(
                    cell_rect.left + max(6, int(8 * self._scale)),
                    cell_rect.centery,
                )
            ),
        )

    # ------------------------------------------------------------------
    # Modern Game Over Dialog — fonts, icons, responsive layout & draw
    # ------------------------------------------------------------------

    @staticmethod
    def _load_dialog_font(size: int, weight: str = "regular") -> pygame.font.Font:
        """Load crisp UI typography using true system font-weight files when available."""
        size = max(10, size)
        if weight == "bold":
            path = pygame.font.match_font("Segoe UI", bold=True)
            if path:
                return pygame.font.Font(path, size)
            return pygame.font.SysFont("Segoe UI", size, bold=True)
        if weight == "semibold":
            path = pygame.font.match_font("Segoe UI Semibold")
            if path:
                return pygame.font.Font(path, size)
            return pygame.font.SysFont("Segoe UI", size, bold=True)
        # Regular weight: prefer true regular segoeui.ttf over light variants
        reg_path = pygame.font.match_font("Segoe UI")
        if reg_path and reg_path.lower().endswith("segoeuil.ttf"):
            alt = Path(reg_path).with_name("segoeui.ttf")
            if alt.is_file():
                return pygame.font.Font(str(alt), size)
        if reg_path:
            return pygame.font.Font(reg_path, size)
        return pygame.font.SysFont("Segoe UI", size)

    @classmethod
    def _make_dialog_result_icon(cls, category: str, size: int) -> pygame.Surface:
        """Create and cache a 2x-supersampled result icon badge (win trophy / loss flag / draw)."""
        cache_key = (f"dialog_result:{category}", (size, size))
        if cache_key in cls._asset_cache:
            return cls._asset_cache[cache_key]

        hi = max(32, size * 2)
        surf = pygame.Surface((hi, hi), pygame.SRCALPHA)
        cx, cy = hi // 2, hi // 2
        radius = hi // 2 - 2

        if category == "win":
            accent = DIALOG_WIN_COLOR
            pygame.draw.circle(surf, (*accent, 36), (cx, cy), radius)
            pygame.draw.circle(surf, (*accent, 115), (cx, cy), radius, width=3)
            # Trophy cup silhouette (warm gold / green-accented modern emblem)
            gold = DIALOG_CROWN_GOLD
            cup_w = int(hi * 0.42)
            cup_h = int(hi * 0.34)
            cup_top = int(hi * 0.24)
            cup_rect = pygame.Rect(cx - cup_w // 2, cup_top, cup_w, cup_h)
            pygame.draw.rect(
                surf,
                gold,
                cup_rect,
                border_top_left_radius=4,
                border_top_right_radius=4,
                border_bottom_left_radius=cup_w // 2,
                border_bottom_right_radius=cup_w // 2,
            )
            # Side handles
            h_r = int(hi * 0.11)
            h_y = cup_top + int(cup_h * 0.35)
            pygame.draw.circle(surf, gold, (cup_rect.left - h_r // 3, h_y), h_r, width=max(3, hi // 26))
            pygame.draw.circle(surf, gold, (cup_rect.right + h_r // 3, h_y), h_r, width=max(3, hi // 26))
            # Stem & pedestal base
            stem_w = max(5, int(hi * 0.08))
            stem_h = int(hi * 0.14)
            pygame.draw.rect(surf, gold, (cx - stem_w // 2, cup_rect.bottom - 2, stem_w, stem_h))
            base_w = int(hi * 0.34)
            base_h = max(6, int(hi * 0.08))
            pygame.draw.rect(
                surf,
                gold,
                (cx - base_w // 2, cup_rect.bottom + stem_h - 3, base_w, base_h),
                border_radius=base_h // 2,
            )
        elif category == "loss":
            accent = DIALOG_LOSS_COLOR
            pygame.draw.circle(surf, (*accent, 36), (cx, cy), radius)
            pygame.draw.circle(surf, (*accent, 115), (cx, cy), radius, width=3)
            flag_sz = max(16, int(hi * 0.50))
            flag_surf = cls._load_action_icon("resign.png", flag_sz, accent)
            surf.blit(flag_surf, flag_surf.get_rect(center=(cx, cy)))
        else:
            accent = DIALOG_DRAW_COLOR
            pygame.draw.circle(surf, (*accent, 36), (cx, cy), radius)
            pygame.draw.circle(surf, (*accent, 115), (cx, cy), radius, width=3)
            # Clean balanced equals emblem for Draw
            bar_w = int(hi * 0.44)
            bar_h = max(6, int(hi * 0.10))
            gap = int(hi * 0.10)
            r_top = pygame.Rect(cx - bar_w // 2, cy - gap // 2 - bar_h, bar_w, bar_h)
            r_bot = pygame.Rect(cx - bar_w // 2, cy + gap // 2, bar_w, bar_h)
            pygame.draw.rect(surf, accent, r_top, border_radius=bar_h // 2)
            pygame.draw.rect(surf, accent, r_bot, border_radius=bar_h // 2)

        final = pygame.transform.smoothscale(surf, (size, size))
        cls._asset_cache[cache_key] = final
        return final

    @classmethod
    def _make_crown_badge(cls, size: int) -> pygame.Surface:
        """Create and cache a 2x-supersampled gold crown badge for the winner's avatar."""
        cache_key = ("dialog_crown", (size, size))
        if cache_key in cls._asset_cache:
            return cls._asset_cache[cache_key]

        hi = max(24, size * 2)
        surf = pygame.Surface((hi, hi), pygame.SRCALPHA)
        pad_x = int(hi * 0.12)
        base_y = int(hi * 0.80)
        mid_y = int(hi * 0.48)
        top_y = int(hi * 0.20)
        pts = (
            (pad_x, base_y),
            (pad_x + int(hi * 0.06), int(hi * 0.32)),
            (int(hi * 0.34), mid_y),
            (hi // 2, top_y),
            (int(hi * 0.66), mid_y),
            (hi - pad_x - int(hi * 0.06), int(hi * 0.32)),
            (hi - pad_x, base_y),
        )
        pygame.draw.polygon(surf, DIALOG_CROWN_GOLD, pts)
        pygame.draw.rect(
            surf,
            DIALOG_CROWN_GOLD,
            (pad_x, base_y - 2, hi - pad_x * 2, max(4, int(hi * 0.12))),
            border_radius=3,
        )
        final = pygame.transform.smoothscale(surf, (size, size))
        cls._asset_cache[cache_key] = final
        return final

    @classmethod
    def _make_bot_icon(cls, size: int, color: tuple[int, int, int]) -> pygame.Surface:
        """Load and cache `assets/chessiq-robot.png` with transparent alpha and crisp downscaling."""
        cache_key = (f"dialog_bot_icon:{DIALOG_ROBOT_ICON_FILE}", (size, size))
        if cache_key in cls._asset_cache:
            return cls._asset_cache[cache_key]

        path = ASSETS_DIR / DIALOG_ROBOT_ICON_FILE
        if path.is_file():
            image = pygame.image.load(str(path)).convert_alpha()
            bounds = image.get_bounding_rect(min_alpha=8)
            if bounds.width > 0 and bounds.height > 0:
                image = image.subsurface(bounds).copy()

            # Two-stage smooth downscale preserves glowing eyes and crisp outlines
            bw, bh = image.get_size()
            mid_side = max(128, size * 4)
            if max(bw, bh) > mid_side:
                mid_scale = mid_side / max(bw, bh)
                image = pygame.transform.smoothscale(
                    image,
                    (max(1, round(bw * mid_scale)), max(1, round(bh * mid_scale))),
                )
                bw, bh = image.get_size()

            fit_scale = min(size / max(1, bw), size / max(1, bh))
            scaled = pygame.transform.smoothscale(
                image,
                (
                    max(1, min(size, round(bw * fit_scale))),
                    max(1, min(size, round(bh * fit_scale))),
                ),
            )
            icon = pygame.Surface((size, size), pygame.SRCALPHA)
            icon.blit(scaled, scaled.get_rect(center=(size // 2, size // 2)))
            cls._asset_cache[cache_key] = icon
            return icon

        # Fallback vector icon if chessiq-robot.png is absent
        hi = max(24, size * 2)
        surf = pygame.Surface((hi, hi), pygame.SRCALPHA)
        cx = hi // 2
        stroke = max(3, hi // 14)
        pygame.draw.line(surf, color, (cx, int(hi * 0.14)), (cx, int(hi * 0.30)), stroke)
        pygame.draw.circle(surf, color, (cx, int(hi * 0.14)), max(3, hi // 12))
        head = pygame.Rect(int(hi * 0.16), int(hi * 0.30), int(hi * 0.68), int(hi * 0.56))
        pygame.draw.rect(surf, color, head, width=stroke, border_radius=max(6, hi // 6))
        final = pygame.transform.smoothscale(surf, (size, size))
        cls._asset_cache[cache_key] = final
        return final

    @classmethod
    def _make_close_icon(cls, size: int, color: tuple[int, int, int]) -> pygame.Surface:
        """Create and cache a crisp 2x-supersampled '×' icon."""
        cache_key = (f"dialog_close:{color}", (size, size))
        if cache_key in cls._asset_cache:
            return cls._asset_cache[cache_key]

        hi = max(24, size * 2)
        surf = pygame.Surface((hi, hi), pygame.SRCALPHA)
        pad = int(hi * 0.22)
        stroke = max(4, hi // 10)
        pygame.draw.line(surf, color, (pad, pad), (hi - pad, hi - pad), stroke)
        pygame.draw.line(surf, color, (hi - pad, pad), (pad, hi - pad), stroke)
        final = pygame.transform.smoothscale(surf, (size, size))
        cls._asset_cache[cache_key] = final
        return final

    @staticmethod
    def _make_dialog_shadow(size: tuple[int, int], radius: int) -> pygame.Surface:
        """Pre-render a soft layered elevation shadow behind the Game Over card."""
        pad = 22
        w, h = size
        shadow = pygame.Surface((w + pad * 2, h + pad * 2), pygame.SRCALPHA)
        for spread, alpha in ((18, 18), (12, 32), (6, 52), (2, 75)):
            rect = pygame.Rect(
                pad - spread,
                pad - spread + 6,
                w + spread * 2,
                h + spread * 2,
            )
            pygame.draw.rect(
                shadow, (0, 0, 0, alpha), rect, border_radius=radius + spread
            )
        return shadow

    def _build_endgame_dialog_layout(self) -> None:
        """Compute responsive Game Over dialog geometry, fonts, icons, and buttons."""
        # Use board area on wide screens; span full screen on narrow windows (e.g. 455px)
        use_full_screen = self._board_area.width < 340 or self.width <= 560
        container = (
            pygame.Rect(0, 0, self.width, self.height)
            if use_full_screen
            else self._board_area.copy()
        )
        self._popup_overlay_rect = container
        self._popup_overlay = pygame.Surface(container.size, pygame.SRCALPHA)
        self._popup_overlay.fill((12, 11, 10, DIALOG_OVERLAY_ALPHA))

        margin_x = 12 if container.width <= 480 else 24
        margin_y = 10 if container.height <= 460 else 20
        avail_w = max(DIALOG_MIN_WIDTH, container.width - margin_x * 2)
        avail_h = max(260, container.height - margin_y * 2)

        desired_w = round(DIALOG_MAX_WIDTH * max(0.88, min(1.05, self._scale)))
        card_w = max(DIALOG_MIN_WIDTH, min(DIALOG_MAX_WIDTH, avail_w, desired_w))
        # On narrow ~455px windows, let the card use generous width for clean breathing room
        if container.width <= 520:
            card_w = min(DIALOG_MAX_WIDTH, avail_w)

        # Adaptive density scale based on actual available width & height
        h_ratio = min(1.0, avail_h / 455.0)
        w_ratio = min(1.0, card_w / 400.0)
        d_scale = max(0.68, min(1.0, min(h_ratio, w_ratio)))

        pad_x = max(14, round(DIALOG_PAD_X * d_scale))
        pad_y = max(12, round(DIALOG_PAD_Y * h_ratio))
        section_gap = max(8, round(DIALOG_SECTION_GAP * h_ratio))
        btn_gap = max(8, round(DIALOG_BUTTON_GAP * w_ratio))

        # Typography using real font-weight files
        self._font_dlg_title = self._load_dialog_font(
            max(22, round(32 * d_scale)), "bold"
        )
        self._font_dlg_sub = self._load_dialog_font(
            max(13, round(16 * d_scale)), "regular"
        )
        self._font_dlg_meta = self._load_dialog_font(
            max(11, round(13 * d_scale)), "regular"
        )
        self._font_dlg_player = self._load_dialog_font(
            max(13, round(17 * d_scale)), "semibold"
        )
        self._font_dlg_elo = self._load_dialog_font(
            max(11, round(13 * d_scale)), "regular"
        )
        self._font_dlg_vs = self._load_dialog_font(
            max(11, round(14 * d_scale)), "bold"
        )
        self._font_dlg_btn = self._load_dialog_font(
            max(13, round(16 * d_scale)), "bold"
        )

        result_icon_sz = max(
            DIALOG_RESULT_ICON_MIN, round(DIALOG_RESULT_ICON_BASE * h_ratio)
        )
        avatar_sz = max(DIALOG_AVATAR_MIN, round(DIALOG_AVATAR_BASE * min(h_ratio, w_ratio)))
        crown_sz = max(16, round(22 * d_scale))
        btn_h = max(DIALOG_BUTTON_HEIGHT_MIN, round(DIALOG_BUTTON_HEIGHT_BASE * h_ratio))
        btn_icon_sz = max(18, min(24, round(DIALOG_BUTTON_ICON_BASE * d_scale)))

        inner_w = card_w - pad_x * 2
        stack_buttons = inner_w < 260
        buttons_block_h = (btn_h * 2 + btn_gap) if stack_buttons else btn_h

        title_h = self._font_dlg_title.get_linesize()
        sub_h = self._font_dlg_sub.get_linesize()
        meta_h = self._font_dlg_meta.get_linesize()
        name_h = self._font_dlg_player.get_linesize()
        elo_h = self._font_dlg_elo.get_linesize()

        # Compress gently if total vertical content exceeds avail_h (e.g. very short windows)
        crown_gap = max(2, round(3 * h_ratio))
        name_gap = max(5, round(8 * h_ratio))
        title_gap = max(4, round(6 * h_ratio))
        sub_gap = max(2, round(4 * h_ratio))

        def _compute_total_h(icon_s: int, av_s: int, py: int, s_gap: int, bh: int) -> int:
            b_block = (bh * 2 + btn_gap) if stack_buttons else bh
            return (
                py
                + icon_s
                + title_gap
                + title_h
                + sub_gap
                + sub_h
                + sub_gap
                + meta_h
                + s_gap
                + crown_sz
                + crown_gap
                + av_s
                + name_gap
                + name_h
                + elo_h
                + s_gap
                + b_block
                + py
            )

        total_h = _compute_total_h(result_icon_sz, avatar_sz, pad_y, section_gap, btn_h)
        if total_h > avail_h:
            shrink = avail_h / max(1, total_h)
            result_icon_sz = max(28, int(result_icon_sz * shrink))
            avatar_sz = max(40, int(avatar_sz * shrink))
            pad_y = max(10, int(pad_y * shrink))
            section_gap = max(6, int(section_gap * shrink))
            btn_h = max(36, int(btn_h * shrink))
            total_h = _compute_total_h(result_icon_sz, avatar_sz, pad_y, section_gap, btn_h)

        card_h = min(avail_h, max(total_h, 240))
        self._popup_rect = pygame.Rect(0, 0, card_w, card_h)
        if use_full_screen:
            self._popup_rect.center = container.center
        else:
            self._popup_rect.center = self._board.board_rect.center
            self._popup_rect.clamp_ip(container.inflate(-margin_x * 2, -margin_y * 2))

        self._popup_radius = max(12, round(DIALOG_CORNER_RADIUS * d_scale))
        self._popup_pad_x = pad_x
        self._popup_pad_y = pad_y
        self._popup_avatar_size = avatar_sz
        self._popup_result_icon_size = result_icon_sz
        self._popup_crown_size = crown_sz

        # Cache dialog-sized avatars, result icons, crown badge, and button icons
        self._popup_player_avatar = self._load_avatar(PLAYER_AVATAR_IMAGE, avatar_sz)
        self._popup_bot_avatar = self._load_avatar(self._opponent_image, avatar_sz)
        self._popup_result_icons = {
            "win": self._make_dialog_result_icon("win", result_icon_sz),
            "loss": self._make_dialog_result_icon("loss", result_icon_sz),
            "draw": self._make_dialog_result_icon("draw", result_icon_sz),
        }
        self._popup_crown_icon = self._make_crown_badge(crown_sz)
        self._popup_replay_icon = self._load_action_icon(
            "restart.jpg", btn_icon_sz, (255, 255, 255)
        )
        bot_icon_sz = max(22, min(28, round((DIALOG_BUTTON_ICON_BASE + 5) * d_scale)))
        self._popup_bot_icon = self._make_bot_icon(bot_icon_sz, THEME_TEXT)
        self._popup_close_icon = self._make_close_icon(DIALOG_CLOSE_ICON_SIZE, THEME_TEXT)
        self._popup_shadow_surface = self._make_dialog_shadow(
            self._popup_rect.size, self._popup_radius
        )

        # Compute internal vertical anchor coordinates (relative to self._popup_rect.top)
        y = self._popup_rect.top + pad_y
        self._popup_icon_cy = y + result_icon_sz // 2
        y += result_icon_sz + title_gap
        self._popup_title_cy = y + title_h // 2
        y += title_h + sub_gap
        self._popup_sub_cy = y + sub_h // 2
        y += sub_h + sub_gap
        self._popup_meta_cy = y + meta_h // 2
        y += meta_h + section_gap

        self._popup_crown_cy = y + crown_sz // 2
        y += crown_sz + crown_gap
        self._popup_avatar_cy = y + avatar_sz // 2
        y += avatar_sz + name_gap
        self._popup_name_top = y
        self._popup_elo_top = y + name_h

        # Balanced horizontal columns for Player | VS | Bot
        vs_w = max(34, self._font_dlg_vs.size("VS")[0] + 16)
        col_w = max(avatar_sz + 8, (inner_w - vs_w - 8) // 2)
        self._popup_col_w = col_w
        self._popup_player_cx = self._popup_rect.left + pad_x + col_w // 2
        self._popup_bot_cx = self._popup_rect.right - pad_x - col_w // 2

        # Close button: minimum 40x40 hit target, comfortably inset top-right
        close_s = DIALOG_CLOSE_SIZE
        close_inset = max(10, min(14, pad_x // 2))
        self._btn_popup_close = Button(
            pygame.Rect(
                self._popup_rect.right - close_s - close_inset,
                self._popup_rect.top + close_inset,
                close_s,
                close_s,
            ),
            "",
            self._font_dlg_btn,
            color=DIALOG_CLOSE_BG,
            hover_color=DIALOG_CLOSE_HOVER,
            text_color=THEME_TEXT,
            border_radius=max(8, close_s // 4),
            border_color=DIALOG_CARD_BORDER,
            border_width=1,
            accent_color=THEME_GREEN,
            press_feedback=True,
        )

        # Action buttons at bottom of card (side-by-side or stacked on ultra-narrow screens)
        btn_rad = max(8, round(10 * d_scale))
        if stack_buttons:
            rematch_rect = pygame.Rect(
                self._popup_rect.left + pad_x,
                self._popup_rect.bottom - pad_y - btn_h * 2 - btn_gap,
                inner_w,
                btn_h,
            )
            bots_rect = pygame.Rect(
                self._popup_rect.left + pad_x,
                self._popup_rect.bottom - pad_y - btn_h,
                inner_w,
                btn_h,
            )
        else:
            btn_w = (inner_w - btn_gap) // 2
            btn_y = self._popup_rect.bottom - pad_y - btn_h
            rematch_rect = pygame.Rect(
                self._popup_rect.left + pad_x, btn_y, btn_w, btn_h
            )
            bots_rect = pygame.Rect(
                rematch_rect.right + btn_gap, btn_y, inner_w - btn_w - btn_gap, btn_h
            )

        self._btn_popup_rematch = Button(
            rematch_rect,
            "",
            self._font_dlg_btn,
            color=THEME_GREEN,
            hover_color=THEME_GREEN_HOVER_TOP,
            text_color=(255, 255, 255),
            border_radius=btn_rad,
            accent=True,
            gradient_colors=(THEME_GREEN, THEME_GREEN_DARK),
            hover_gradient_colors=(THEME_GREEN_HOVER_TOP, THEME_GREEN_HOVER_BOT),
            border_color=THEME_GREEN_DARK,
            border_width=1,
            accent_color=THEME_GREEN,
            press_feedback=True,
        )
        self._btn_popup_bots = Button(
            bots_rect,
            "",
            self._font_dlg_btn,
            color=DIALOG_SECONDARY_BG,
            hover_color=DIALOG_SECONDARY_HOVER,
            text_color=THEME_TEXT,
            border_radius=btn_rad,
            border_color=DIALOG_CARD_BORDER,
            border_width=1,
            accent=True,
            accent_color=DIALOG_ROBOT_ACCENT,
            press_feedback=True,
        )
        self._popup_card_surface = pygame.Surface(self._popup_rect.size, pygame.SRCALPHA)

    def _render_endgame_card_contents(
        self, target: pygame.Surface, ox: int, oy: int
    ) -> None:
        """Draw the Game Over card interior onto *target* offset by `(-ox, -oy)`."""
        assert self._game is not None
        outcome = self._game.outcome
        category, headline, subtitle = format_endgame_dialog_text(
            outcome, self._player_color, self._opponent_name
        )

        if category == "win":
            accent = DIALOG_WIN_COLOR
        elif category == "loss":
            accent = DIALOG_LOSS_COLOR
        else:
            accent = DIALOG_DRAW_COLOR

        r = self._popup_rect.move(-ox, -oy)
        rad = self._popup_radius

        # Card surface + subtle top highlight + crisp border
        pygame.draw.rect(target, DIALOG_CARD_BG, r, border_radius=rad)
        pygame.draw.rect(target, DIALOG_CARD_BORDER, r, width=1, border_radius=rad)
        # Subtle colored accent bar along top edge (clamped inside corner radius)
        bar_inset = rad + 2
        if r.width > bar_inset * 2:
            pygame.draw.line(
                target,
                accent,
                (r.left + bar_inset, r.top + 1),
                (r.right - bar_inset, r.top + 1),
                width=2,
            )

        # 1. Result icon + Headline + Subtitle + optional concise stats line
        res_icon = self._popup_result_icons[category]
        target.blit(res_icon, res_icon.get_rect(center=(r.centerx, self._popup_icon_cy - oy)))

        max_text_w = max(80, r.width - self._popup_pad_x * 2 - 20)
        title_text = self._ellipsize(headline, self._font_dlg_title, max_text_w)
        title_surf = self._get_cached_text(self._font_dlg_title, title_text, accent)
        target.blit(
            title_surf,
            title_surf.get_rect(center=(r.centerx, self._popup_title_cy - oy)),
        )

        sub_text = self._ellipsize(subtitle, self._font_dlg_sub, max_text_w)
        sub_surf = self._get_cached_text(
            self._font_dlg_sub, sub_text, DIALOG_SUBTITLE_COLOR
        )
        target.blit(
            sub_surf,
            sub_surf.get_rect(center=(r.centerx, self._popup_sub_cy - oy)),
        )

        num_plies = len(self._game.san_moves)
        full_moves = (num_plies + 1) // 2
        move_lbl = f"{full_moves} move" if full_moves == 1 else f"{full_moves} moves"
        meta_raw = f"{move_lbl}  •  {self._selected_time_control}"
        meta_text = self._ellipsize(meta_raw, self._font_dlg_meta, max_text_w)
        meta_surf = self._get_cached_text(
            self._font_dlg_meta, meta_text, DIALOG_META_COLOR
        )
        target.blit(
            meta_surf,
            meta_surf.get_rect(center=(r.centerx, self._popup_meta_cy - oy)),
        )

        # 2. Player vs Bot avatars, winner ring + crown badge, names, and ELO
        av_cy = self._popup_avatar_cy - oy
        pl_cx = self._popup_player_cx - ox
        bot_cx = self._popup_bot_cx - ox
        av_r = self._popup_avatar_size // 2

        pl_ring_color = (
            DIALOG_WIN_COLOR
            if category == "win"
            else (DIALOG_DRAW_COLOR if category == "draw" else DIALOG_LOSER_RING)
        )
        bot_ring_color = (
            DIALOG_LOSS_COLOR
            if category == "loss"
            else (DIALOG_DRAW_COLOR if category == "draw" else DIALOG_LOSER_RING)
        )
        pl_ring_w = 3 if category == "win" else 2
        bot_ring_w = 3 if category == "loss" else 2

        # Winner crown badge above winning player
        if category == "win":
            target.blit(
                self._popup_crown_icon,
                self._popup_crown_icon.get_rect(
                    center=(pl_cx, self._popup_crown_cy - oy)
                ),
            )
        elif category == "loss":
            target.blit(
                self._popup_crown_icon,
                self._popup_crown_icon.get_rect(
                    center=(bot_cx, self._popup_crown_cy - oy)
                ),
            )

        pl_rect = self._popup_player_avatar.get_rect(center=(pl_cx, av_cy))
        bot_rect = self._popup_bot_avatar.get_rect(center=(bot_cx, av_cy))
        target.blit(self._popup_player_avatar, pl_rect)
        target.blit(self._popup_bot_avatar, bot_rect)

        pygame.draw.circle(target, pl_ring_color, pl_rect.center, av_r + 1, width=pl_ring_w)
        pygame.draw.circle(target, bot_ring_color, bot_rect.center, av_r + 1, width=bot_ring_w)

        # Subtle VS pill between avatars
        vs_surf = self._get_cached_text(self._font_dlg_vs, "VS", DIALOG_META_COLOR)
        vs_pill = vs_surf.get_rect(center=(r.centerx, av_cy)).inflate(14, 8)
        pygame.draw.rect(
            target, THEME_PANEL_ALT, vs_pill, border_radius=vs_pill.height // 2
        )
        pygame.draw.rect(
            target, DIALOG_CARD_BORDER, vs_pill, width=1, border_radius=vs_pill.height // 2
        )
        target.blit(vs_surf, vs_surf.get_rect(center=vs_pill.center))

        # Labels under avatars (strictly clipped to column width, zero overlap)
        col_max_w = max(48, self._popup_col_w - 6)
        name_top = self._popup_name_top - oy
        elo_top = self._popup_elo_top - oy

        you_surf = self._get_cached_text(self._font_dlg_player, "You", THEME_TEXT)
        target.blit(you_surf, you_surf.get_rect(midtop=(pl_cx, name_top)))

        bot_name_txt = self._ellipsize(
            self._opponent_name, self._font_dlg_player, col_max_w
        )
        bot_name_surf = self._get_cached_text(
            self._font_dlg_player, bot_name_txt, THEME_TEXT
        )
        target.blit(bot_name_surf, bot_name_surf.get_rect(midtop=(bot_cx, name_top)))

        elo_txt = self._ellipsize(
            f"{self._opponent_elo} ELO", self._font_dlg_elo, col_max_w
        )
        elo_surf = self._get_cached_text(self._font_dlg_elo, elo_txt, DIALOG_META_COLOR)
        target.blit(elo_surf, elo_surf.get_rect(midtop=(bot_cx, elo_top)))

        # 3. Close button and action buttons (drawn offset-aware for smooth entrance)
        if ox == 0 and oy == 0:
            self._btn_popup_close.draw(target)
            target.blit(
                self._popup_close_icon,
                self._popup_close_icon.get_rect(
                    center=self._btn_popup_close.base_rect.center
                ),
            )
            self._draw_labeled_icon_button(
                self._btn_popup_rematch,
                "Play Again",
                icon_surface=self._popup_replay_icon,
                text_color=(255, 255, 255),
                font=self._font_dlg_btn,
                surface=target,
            )
            self._draw_labeled_icon_button(
                self._btn_popup_bots,
                "New Bot",
                icon_surface=self._popup_bot_icon,
                text_color=THEME_TEXT,
                font=self._font_dlg_btn,
                surface=target,
            )
        else:
            for btn, lbl, icon, txt_col in (
                (self._btn_popup_close, "", self._popup_close_icon, THEME_TEXT),
                (self._btn_popup_rematch, "Play Again", self._popup_replay_icon, (255, 255, 255)),
                (self._btn_popup_bots, "New Bot", self._popup_bot_icon, THEME_TEXT),
            ):
                orig = btn.base_rect.copy()
                btn.base_rect = orig.move(-ox, -oy)
                if lbl:
                    self._draw_labeled_icon_button(
                        btn, lbl, icon_surface=icon, text_color=txt_col, font=self._font_dlg_btn, surface=target
                    )
                else:
                    btn.draw(target)
                    target.blit(icon, icon.get_rect(center=btn.base_rect.center))
                btn.base_rect = orig

    def _draw_endgame_popup(self) -> None:
        t = self.popup_entrance_progress
        eased = ease_progress(t, "ease_out_cubic")

        # 1. Dimmed translucent backdrop fading in cleanly
        overlay_alpha = round(DIALOG_OVERLAY_ALPHA * eased)
        self._popup_overlay.fill((12, 11, 10, overlay_alpha))
        self.screen.blit(self._popup_overlay, self._popup_overlay_rect.topleft)

        # 2. Soft drop shadow + card (subtle scale 0.96->1.0 + slide + fade during entrance)
        r = self._popup_rect
        if eased >= 0.999:
            if self._popup_shadow_surface is not None:
                self._popup_shadow_surface.set_alpha(255)
                self.screen.blit(
                    self._popup_shadow_surface,
                    self._popup_shadow_surface.get_rect(center=r.center),
                )
            self._render_endgame_card_contents(self.screen, 0, 0)
            return

        assert self._popup_card_surface is not None
        self._popup_card_surface.fill((0, 0, 0, 0))
        self._render_endgame_card_contents(self._popup_card_surface, r.left, r.top)

        scale_f = DIALOG_ENTRANCE_SCALE_START + (1.0 - DIALOG_ENTRANCE_SCALE_START) * eased
        slide_y = round(DIALOG_ENTRANCE_SLIDE_PX * (1.0 - eased))
        scaled_w = max(1, round(r.width * scale_f))
        scaled_h = max(1, round(r.height * scale_f))
        anim_card = pygame.transform.smoothscale(
            self._popup_card_surface, (scaled_w, scaled_h)
        )
        anim_card.set_alpha(max(0, min(255, round(255 * eased))))
        dest_center = (r.centerx, r.centery + slide_y)

        if self._popup_shadow_surface is not None:
            self._popup_shadow_surface.set_alpha(max(0, min(255, round(255 * eased))))
            self.screen.blit(
                self._popup_shadow_surface,
                self._popup_shadow_surface.get_rect(center=dest_center),
            )
        self.screen.blit(anim_card, anim_card.get_rect(center=dest_center))

    def _draw_labeled_icon_button(
        self,
        button: Button,
        label: str,
        icon_drawer=None,
        icon_surface: pygame.Surface | None = None,
        text_color: tuple[int, int, int] = _TEXT,
        font: pygame.font.Font | None = None,
        surface: pygame.Surface | None = None,
    ) -> None:
        target = surface if surface is not None else self.screen
        button.draw(target)
        rect = button.base_rect
        gap = max(6, int(8 * self._scale))
        active_font = font or self._font_choice
        max_lbl_w = max(20, rect.width - (icon_surface.get_width() if icon_surface else 14) - gap - 16)
        label_txt = self._ellipsize(label, active_font, max_lbl_w)
        text_surf = self._get_cached_text(active_font, label_txt, text_color)
        press_y = round(button._pressed_t) if button.press_feedback else 0
        cy = rect.centery + press_y
        if icon_surface is not None:
            icon_w = icon_surface.get_width()
            total_w = icon_w + gap + text_surf.get_width()
            start_x = rect.centerx - total_w // 2
            target.blit(
                icon_surface,
                icon_surface.get_rect(midleft=(start_x, cy)),
            )
            target.blit(
                text_surf,
                text_surf.get_rect(midleft=(start_x + icon_w + gap, cy)),
            )
        elif icon_drawer is not None:
            icon_r = max(6, int(7 * self._scale))
            total_w = icon_r * 2 + gap + text_surf.get_width()
            start_x = rect.centerx - total_w // 2
            icon_drawer(target, (start_x + icon_r, cy), icon_r, text_color)
            target.blit(
                text_surf,
                text_surf.get_rect(midleft=(start_x + icon_r * 2 + gap, cy)),
            )

    # ── vector icon helpers ───────────────────────────────────────────

    @staticmethod
    def _draw_flag_icon(
        surf: pygame.Surface,
        center: tuple[int, int],
        r: int,
        color: tuple[int, int, int],
    ) -> None:
        cx, cy = center
        x0 = cx - r // 2
        pygame.draw.line(surf, color, (x0, cy - r), (x0, cy + r), 2)
        pygame.draw.polygon(
            surf,
            color,
            ((x0 + 1, cy - r), (x0 + r + 2, cy - r // 2), (x0 + 1, cy + 1)),
        )

    @staticmethod
    def _draw_gear_icon(
        surf: pygame.Surface,
        center: tuple[int, int],
        r: int,
        color: tuple[int, int, int],
    ) -> None:
        cx, cy = center
        pygame.draw.circle(surf, color, (cx, cy), max(3, int(r * 0.55)), width=2)
        for angle_deg in range(0, 360, 45):
            rad = math.radians(angle_deg)
            p1 = (
                cx + int(math.cos(rad) * (r * 0.55)),
                cy + int(math.sin(rad) * (r * 0.55)),
            )
            p2 = (cx + int(math.cos(rad) * r), cy + int(math.sin(rad) * r))
            pygame.draw.line(surf, color, p1, p2, 2)

    @staticmethod
    def _draw_restart_icon(
        surf: pygame.Surface,
        center: tuple[int, int],
        r: int,
        color: tuple[int, int, int],
    ) -> None:
        cx, cy = center
        pygame.draw.arc(
            surf,
            color,
            pygame.Rect(cx - r + 1, cy - r + 1, r * 2 - 2, r * 2 - 2),
            0.4,
            math.pi * 1.85,
            2,
        )
        tip = (cx + r - 1, cy - 2)
        pygame.draw.polygon(
            surf, color, (tip, (tip[0] - 5, tip[1] - 4), (tip[0] - 5, tip[1] + 2))
        )

    @staticmethod
    def _draw_no_timer_icon(
        surf: pygame.Surface,
        center: tuple[int, int],
        r: int,
        color: tuple[int, int, int],
    ) -> None:
        """Draw a clean infinity / untimed clock symbol for the 'No Timer' option."""
        cx, cy = center
        pygame.draw.circle(surf, color, (cx, cy), r, width=2)
        pygame.draw.line(surf, color, (cx, cy), (cx, cy - max(2, r - 3)), 2)
        pygame.draw.line(surf, color, (cx, cy), (cx + max(2, r - 4), cy), 2)
        slash = max(2, int(r * 0.72))
        pygame.draw.line(
            surf, color, (cx - slash, cy + slash), (cx + slash, cy - slash), 2
        )

    @staticmethod
    def _draw_chevron_icon(
        surf: pygame.Surface,
        center: tuple[int, int],
        r: int,
        color: tuple[int, int, int],
        is_open: bool,
    ) -> None:
        """Draw a crisp up ('^') or down ('v') chevron for the dropdown trigger."""
        cx, cy = center
        half_h = max(2, r // 2)
        if is_open:
            p_left = (cx - r, cy + half_h)
            p_mid = (cx, cy - half_h)
            p_right = (cx + r, cy + half_h)
        else:
            p_left = (cx - r, cy - half_h)
            p_mid = (cx, cy + half_h)
            p_right = (cx + r, cy - half_h)
        pygame.draw.lines(surf, color, False, (p_left, p_mid, p_right), 2)

    def _status_text(self) -> str:
        if self._result_text:
            return self._result_text
        if self._game is None:
            return ""
        if self._bot_thinking:
            return f"{self._opponent_name} is thinking..."
        if self._game.turn_color == self._player_color:
            return "Your move — Check!" if self._game.is_check else "Your move"
        return f"{self._opponent_name} to move"

    def _draw_label(
        self,
        text: str,
        position: tuple[int, int],
        color: tuple[int, int, int] = THEME_MUTED,
        font: pygame.font.Font | None = None,
    ) -> None:
        surf = self._get_cached_text(font or self._font_label, text, color)
        self.screen.blit(surf, position)

    def _draw_time_group_icon(
        self, label: str, position: tuple[int, int]
    ) -> None:
        icon = self._group_icons[label]
        rect = icon.get_rect(
            midleft=(
                position[0],
                position[1] + self._font_label.get_linesize() // 2,
            )
        )
        self.screen.blit(icon, rect)

    @staticmethod
    def _ellipsize(text: str, font: pygame.font.Font, max_width: int) -> str:
        if font.size(text)[0] <= max_width:
            return text
        ellipsis = "..."
        trimmed = text
        while trimmed and font.size(trimmed + ellipsis)[0] > max_width:
            trimmed = trimmed[:-1]
        return trimmed + ellipsis

    def _draw_wrapped(
        self,
        text: str,
        font: pygame.font.Font,
        color: tuple[int, int, int],
        position: tuple[int, int],
        max_width: int,
        center_height: int | None = None,
    ) -> None:
        rendered_lines = self._get_wrapped_lines(text, font, color, max_width)
        line_height = font.get_linesize()
        if center_height is not None:
            position = (
                position[0],
                position[1]
                + max(0, (center_height - len(rendered_lines) * line_height) // 2),
            )
        for index, surf in enumerate(rendered_lines):
            self.screen.blit(
                surf,
                (round(position[0]), round(position[1] + index * line_height)),
            )

    def _get_wrapped_lines(
        self,
        text: str,
        font: pygame.font.Font,
        color: tuple[int, int, int],
        max_width: int,
    ) -> list[pygame.Surface]:
        """Return cached, antialiased lines that fit within the requested width."""
        max_width = max(1, max_width)
        cache_key = (id(font), text, max_width, color)
        rendered_lines = self._wrapped_lines_cache.get(cache_key)
        if rendered_lines is None:
            lines: list[str] = []
            current = ""
            for word in text.split():
                candidate = f"{current} {word}".strip()
                if current and font.size(candidate)[0] > max_width:
                    lines.append(current)
                    current = ""
                while font.size(word)[0] > max_width:
                    split_at = 1
                    while (
                        split_at < len(word)
                        and font.size(word[: split_at + 1])[0] <= max_width
                    ):
                        split_at += 1
                    if current:
                        lines.append(current)
                        current = ""
                    lines.append(word[:split_at])
                    word = word[split_at:]
                if word:
                    current = f"{current} {word}".strip()
            if current:
                lines.append(current)
            if not lines:
                lines.append("")
            rendered_lines = [font.render(line, True, color) for line in lines]
            self._wrapped_lines_cache[cache_key] = rendered_lines
        return rendered_lines
