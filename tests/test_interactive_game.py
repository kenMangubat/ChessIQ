"""Tests for the interactive board (orientation, selection, promotion, animation)
and GameScreen bot integration (stale results, fallback, cleanup).

Runs headless via SDL_VIDEODRIVER=dummy.
"""

import os
import sys

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import chess
import pygame

from bots.ken_bot import KenBot
from chess_logic import ChessGame
from chess_logic.chess_game import (
    GameOutcome,
    MatchClock,
    TimeControlPreset,
    format_clock_seconds,
)
from screens.game import (
    BUBBLE_ANIM_DURATION,
    BUBBLE_TEXT_CHANGE_DURATION,
    BUBBLE_TEXT_DELAY,
    BUBBLE_TEXT_DURATION,
    DIALOG_CLOSE_SIZE,
    DIALOG_ENTRANCE_DURATION,
    GameScreen,
    NO_TIMER_PRESET,
    PLAYER_USERNAME,
    PRESET_BY_LABEL,
    TC_BTN_BG,
    TC_PANEL_BG,
    TC_TRIGGER_BG,
    TIME_CONTROL_GROUPS,
    TIME_CONTROL_PRESETS,
    format_endgame_dialog_text,
    format_san_figurine,
    format_time_control_label,
    get_time_control_category,
    identify_opening,
)
from screens.opponent_select import OpponentSelectScreen, _BOTS
from screens.settings import SettingsScreen
from ui.animated_board import (
    ANIMATION_BASE_DURATION,
    ANIMATION_KNIGHT_DISTANCE,
    ANIMATION_MAX_DURATION,
    CAPTURE_FADE_DURATION,
    MAX_ANIMATION_DT,
    AnimatedBoard,
    compute_move_duration,
    ease_progress,
    get_bot_think_range,
    move_distance_in_squares,
    premove_targets_from,
)

_OPPONENT = {
    "name": "Sick Ken",
    "elo": 1350,
    "image": "sickken.png",
    "tagline": "test",
}


def _board(game: ChessGame, flipped: bool = False) -> AnimatedBoard:
    pygame.init()
    board = AnimatedBoard(800, 720, 1.0)
    board.attach_game(game, flipped=flipped)
    return board


def _click(board: AnimatedBoard, square: int, color: chess.Color):
    x, y = board.square_to_pixel(square)
    return board.handle_click(x + 5, y + 5, color)


class FakeWorker:
    """Stand-in for BotWorker: records requests, returns queued results on poll."""

    def __init__(self) -> None:
        self.requests: list[str] = []
        self.results: list[chess.Move | None] = []
        self.stopped = False
        self.last_move_source: str = "book"
        self.last_compute_seconds: float = 0.0

    def request_move(self, board: chess.Board, delay_seconds: float = 0.0) -> None:
        self.requests.append(board.fen())

    def poll_move(self) -> chess.Move | None:
        return self.results.pop(0) if self.results else None

    def stop(self) -> None:
        self.stopped = True


def _game_screen(side: str) -> tuple[GameScreen, FakeWorker]:
    pygame.init()
    surface = pygame.display.set_mode((1280, 720))
    screen = GameScreen(surface, _OPPONENT)
    worker = FakeWorker()
    screen._bot = KenBot("Sick Ken", log_provenance=False)
    screen._bot_worker = worker
    screen._selected_side = side
    screen._start_match()
    return screen, worker


def _finish_animation(screen: GameScreen) -> None:
    for _ in range(40):
        screen.update(1 / 60)


def _advance_bot_pacing(screen: GameScreen, max_steps: int = 160) -> None:
    """Advance simulated 60 FPS frames until any pending bot thinking pause elapses."""
    for _ in range(max_steps):
        if screen._ready_bot_move is None and not screen._pending_requests:
            break
        screen.update(1 / 60)


# ── orientation ─────────────────────────────────────────────────────


def test_orientation_round_trip():
    game = ChessGame()
    for flipped in (False, True):
        board = _board(game, flipped)
        for square in chess.SQUARES:
            x, y = board.square_to_pixel(square)
            assert board.pixel_to_square(x, y) == square
            assert board.pixel_to_square(x + board._sq - 1, y + board._sq - 1) == square


def test_orientation_corners_and_off_board():
    game = ChessGame()
    white = _board(game, flipped=False)
    black = _board(game, flipped=True)
    ox, oy, sq = white._ox, white._oy, white._sq
    # White view: a8 top-left, a1 bottom-left
    assert white.square_to_pixel(chess.A8) == (ox, oy)
    assert white.square_to_pixel(chess.A1) == (ox, oy + 7 * sq)
    # Black view: h1 top-left, a1 top-right
    assert black.square_to_pixel(chess.H1) == (ox, oy)
    assert black.square_to_pixel(chess.A1) == (ox + 7 * sq, oy)
    for board in (white, black):
        assert board.pixel_to_square(ox - 1, oy) is None
        assert board.pixel_to_square(ox, oy + 8 * sq) is None


# ── selection & move completion ─────────────────────────────────────


def test_selection_flow():
    game = ChessGame()
    board = _board(game)
    assert _click(board, chess.E2, chess.WHITE) is None
    assert board.selected_square == chess.E2
    assert sorted(board.legal_targets) == [chess.E3, chess.E4]

    # Clicking another own piece switches the selection
    assert _click(board, chess.G1, chess.WHITE) is None
    assert board.selected_square == chess.G1

    # Clicking an illegal empty square deselects
    assert _click(board, chess.E5, chess.WHITE) is None
    assert board.selected_square is None

    # Opponent pieces cannot be selected
    assert _click(board, chess.E7, chess.WHITE) is None
    assert board.selected_square is None

    # Clicking off the grid deselects
    _click(board, chess.E2, chess.WHITE)
    assert board.handle_click(0, 0, chess.WHITE) is None
    assert board.selected_square is None

    _click(board, chess.E2, chess.WHITE)
    assert _click(board, chess.E4, chess.WHITE) == (chess.E2, chess.E4, None)
    assert board.selected_square is None


def test_capture_markers():
    game = ChessGame()
    for san in ("e4", "d5"):
        game.make_san(san)
    board = _board(game)
    _click(board, chess.E4, chess.WHITE)
    assert board._capture_targets == {chess.D5}


def test_promotion_flow_white_and_flipped_black():
    for flipped, fen, from_sq, to_sq, color in (
        (False, "8/P6k/8/8/8/8/8/K7 w - - 0 1", chess.A7, chess.A8, chess.WHITE),
        (True, "k7/8/8/8/8/8/p6K/8 b - - 0 1", chess.A2, chess.A1, chess.BLACK),
    ):
        game = ChessGame(fen)
        board = _board(game, flipped)
        _click(board, from_sq, color)
        assert _click(board, to_sq, color) is None
        assert board.promotion_pending

        # The overlay opens on the promotion square and stacks inward
        rects = board.promotion_rects()
        assert [choice for choice, _ in rects] == ["q", "r", "b", "n"]
        assert rects[0][1].topleft == board.square_to_pixel(to_sq)

        # Clicking outside cancels but keeps the pawn selected
        far = board.square_to_pixel(chess.H4)
        assert board.handle_click(far[0] + 5, far[1] + 5, color) is None
        assert not board.promotion_pending
        assert board.selected_square == from_sq

        # Re-open and pick each piece type
        for choice, _ in rects:
            board.select_square(from_sq)
            _click(board, to_sq, color)
            rect = dict(board.promotion_rects())[choice]
            assert board.handle_click(rect.centerx, rect.centery, color) == (from_sq, to_sq, choice)
            assert not board.promotion_pending


# ── animation ───────────────────────────────────────────────────────


def test_animation_hides_destination_until_landed():
    game = ChessGame("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1")
    board = _board(game)
    info = game.make_san("O-O")
    board.animate_move(info)
    assert board.is_animating
    assert board._animation_hidden_squares() == {chess.G1, chess.F1}
    # Clicks are ignored during animation
    assert _click(board, chess.A1, chess.WHITE) is None
    assert board.selected_square is None

    surf = pygame.Surface((800, 720), pygame.SRCALPHA)
    for _ in range(30):
        board.update(1 / 60)
        board.draw(surf)
    assert not board.is_animating
    assert board._animation_hidden_squares() == set()


# ── GameScreen integration ──────────────────────────────────────────


def test_player_black_bot_moves_first_and_board_flipped():
    screen, worker = _game_screen("Black")
    assert screen._board.flipped is True
    screen.update(1 / 60)
    assert len(worker.requests) == 1
    assert screen._bot_thinking
    assert not screen._can_accept_input()

    worker.results.append(chess.Move.from_uci("e2e4"))
    screen.update(1 / 60)
    # Bot enters natural perceived thinking pause before sliding the piece
    assert screen._bot_thinking
    _advance_bot_pacing(screen)
    assert screen._game.uci_moves == ["e2e4"]
    _finish_animation(screen)
    assert screen._can_accept_input()
    screen.close()


def test_stale_bot_result_discarded_after_restart():
    screen, worker = _game_screen("Black")
    screen.update(1 / 60)
    assert len(worker.requests) == 1

    screen._handle_menu_action("restart")
    screen.update(1 / 60)
    assert len(worker.requests) == 2  # new search for the fresh game

    worker.results.append(chess.Move.from_uci("e2e4"))  # answer to the old search
    screen.update(1 / 60)
    assert screen._game.uci_moves == []
    assert screen._bot_thinking

    worker.results.append(chess.Move.from_uci("d2d4"))  # answer to the new search
    screen.update(1 / 60)
    _advance_bot_pacing(screen)
    assert screen._game.uci_moves == ["d2d4"]
    screen.close()


def test_illegal_bot_move_falls_back():
    screen, worker = _game_screen("Black")
    screen.update(1 / 60)
    worker.results.append(chess.Move.from_uci("e2e5"))  # illegal
    screen.update(1 / 60)
    _advance_bot_pacing(screen)
    assert len(screen._game.uci_moves) == 1
    assert screen._game.turn_color == chess.BLACK
    screen.close()


def test_resign_draw_offer_and_close():
    screen, worker = _game_screen("White")
    assert screen._board.flipped is False
    screen._handle_menu_action("offer_draw")
    assert screen._notice_text == "Draw declined"
    assert not screen._game.is_game_over

    screen._handle_menu_action("resign")
    assert screen._game.is_game_over
    assert screen._result_text == "Resignation — Sick Ken wins."
    assert not screen._can_accept_input()

    screen.close()
    assert worker.stopped is True
    assert screen._bot_worker is None
    screen.close()  # idempotent


def test_escape_captured_only_during_promotion():
    screen, _ = _game_screen("White")
    assert screen.captures_escape() is False
    screen._game.reset("8/P6k/8/8/8/8/8/K7 w - - 0 1")
    screen._board.select_square(chess.A7)
    x, y = screen._board.square_to_pixel(chess.A8)
    screen._board.handle_click(x + 5, y + 5, chess.WHITE)
    assert screen.captures_escape() is True
    screen.handle_events([pygame.event.Event(pygame.KEYDOWN, key=pygame.K_ESCAPE)])
    assert screen.captures_escape() is False
    assert screen._board.selected_square == chess.A7
    screen.close()


def test_board_backed_by_chess_game_and_bars_layout():
    pygame.init()
    assert PLAYER_USERNAME == "Player"
    for resolution in ((1280, 720), (1920, 1080)):
        surface = pygame.display.set_mode(resolution)
        screen = GameScreen(surface, _OPPONENT)
        # Immediately backed by ChessGame (not demo replay)
        assert screen._board._game is screen._game
        board_rect = screen._board.board_rect
        assert board_rect.top >= screen._bar_avatar_size + 8
        assert resolution[1] - board_rect.bottom >= screen._bar_avatar_size + 8
        assert screen._btn_board_settings.base_rect.right == board_rect.right
        assert screen._btn_board_settings.base_rect.bottom <= board_rect.top
        assert screen._opening_rect.top > screen._tagline_rect.bottom
        assert screen._moves_rect.top > screen._opening_rect.bottom
        assert screen._moves_rect.bottom < screen._footer_rect.top
        assert screen._footer_rect.bottom < screen._btn_resign.base_rect.top
        assert screen._btn_resign.base_rect.width == screen._moves_rect.width
        assert screen._btn_resign.base_rect.bottom <= resolution[1]
        for removed in (
            "_btn_takeback",
            "_btn_hint",
            "_btn_share",
            "_btn_download",
            "_btn_sidebar_settings",
            "_btn_restart_bar",
        ):
            assert not hasattr(screen, removed)
        screen.draw()

        # Clicking top-right settings gear navigates to settings
        gear = screen._btn_board_settings.base_rect
        screen.handle_events(
            [pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=gear.center)]
        )
        assert screen.next_screen == "settings"
        screen.close()


def test_opening_banner_and_figurine_formatting():
    assert identify_opening([]) == "Starting Position"
    assert identify_opening(["e4", "c5"]) == "Sicilian Defense"
    assert identify_opening(["e4", "c5", "d4"]) == "Sicilian Defense: Smith-Morra Gambit"
    assert (
        identify_opening(["e4", "c5", "d4", "cxd4", "c3"])
        == "Sicilian Defense: Smith-Morra Gambit"
    )
    assert (
        identify_opening(["e4", "c5", "d4", "cxd4", "c3", "dxc3", "Nxc3"])
        == "Sicilian Defense: Smith-Morra Gambit, Accepted"
    )
    assert (
        identify_opening(["e4", "c5", "Nf3", "d6", "d4", "cxd4", "Nxd4", "Nf6", "Nc3", "a6"])
        == "Sicilian Defense: Najdorf Variation"
    )
    assert identify_opening(["e4", "c6"]) == "Caro-Kann Defense"
    assert identify_opening(["d4", "Nf6", "c4", "g6"]) == "King's Indian Defense"
    # Transposition check: 1. d4 Nf6 2. c4 e6 3. Nf3 d5 -> QGD Three Knights
    assert (
        identify_opening(["d4", "Nf6", "c4", "e6", "Nf3", "d5"])
        == "Queen's Gambit Declined: Three Knights"
    )
    # Out-of-book continuation retains deepest matched opening name
    assert (
        identify_opening(["e4", "c5", "d4", "cxd4", "c3", "a6", "h4"])
        == "Sicilian Defense: Smith-Morra Gambit"
    )

    pirc = ["e4", "d6", "d4", "Nf6", "Nc3", "g6", "Bc4"]
    assert identify_opening(pirc) == "Pirc Defense: Main Line, Kholmov System"
    assert format_san_figurine("Nf6") == "♘f6"
    assert format_san_figurine("e8=Q") == "e8=♕"
    assert format_san_figurine("O-O") == "O-O"

    # Verify live updates on GameScreen after every human and bot move
    screen, _ = _game_screen("White")
    expected_progression = [
        ("e4", "King's Pawn Game"),
        ("c5", "Sicilian Defense"),
        ("d4", "Sicilian Defense: Smith-Morra Gambit"),
        ("cxd4", "Sicilian Defense: Smith-Morra Gambit"),
        ("c3", "Sicilian Defense: Smith-Morra Gambit"),
        ("a6", "Sicilian Defense: Smith-Morra Gambit"),
    ]
    for idx, (san, expected_opening) in enumerate(expected_progression):
        move = screen._game.board.parse_san(san)
        if idx % 2 == 0:
            screen._apply_player_move(move.from_square, move.to_square, None)
        else:
            screen._apply_bot_move(move)
        _finish_animation(screen)
        assert screen._current_opening == expected_opening
    screen.draw()
    screen.close()


def test_move_list_scroll_and_clamping():
    screen, _ = _game_screen("White")
    # Play 30 half-moves (15 full moves) to exceed visible move list height
    moves = ["Nf3", "Nf6", "Ng1", "Ng8"] * 7 + ["e4", "e5"]
    for idx, san in enumerate(moves):
        move = screen._game.board.parse_san(san)
        if idx % 2 == 0:
            screen._apply_player_move(move.from_square, move.to_square, None)
        else:
            screen._apply_bot_move(move)
        _finish_animation(screen)
    max_scroll = screen._max_move_scroll()
    assert max_scroll > 0
    assert screen._move_scroll_y == max_scroll

    # Scroll up via legacy wheel button 4 inside move list rect
    screen.handle_events(
        [
            pygame.event.Event(
                pygame.MOUSEBUTTONDOWN, button=4, pos=screen._moves_rect.center
            )
        ]
    )
    assert 0 <= screen._move_scroll_y < max_scroll

    # Clamp at 0 when scrolling far up
    for _ in range(30):
        screen.handle_events(
            [
                pygame.event.Event(
                    pygame.MOUSEBUTTONDOWN, button=4, pos=screen._moves_rect.center
                )
            ]
        )
    assert screen._move_scroll_y == 0
    screen.draw()
    screen.close()


def test_sidebar_resign_becomes_new_game_when_over():
    screen, _ = _game_screen("White")
    resign_pos = screen._btn_resign.base_rect.center
    screen.handle_events(
        [pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=resign_pos)]
    )
    assert screen._game.is_game_over
    assert screen._result_text == "Resignation — Sick Ken wins."
    screen.draw()

    # Clicking the bottom button again starts a New Game
    screen.handle_events(
        [pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=resign_pos)]
    )
    assert not screen._game.is_game_over
    assert screen._show_endgame_popup is False
    assert screen._game.san_moves == []
    screen.close()


def test_endgame_popup_and_rematch_flow():
    screen, _ = _game_screen("White")
    # Play Fool's Mate against White (player loses on Checkmate)
    screen._apply_player_move(chess.F2, chess.F3, None)
    _finish_animation(screen)
    screen._apply_bot_move(chess.Move.from_uci("e7e5"))
    _finish_animation(screen)
    screen._apply_player_move(chess.G2, chess.G4, None)
    _finish_animation(screen)
    screen._apply_bot_move(chess.Move.from_uci("d8h4"))
    _finish_animation(screen)

    assert screen._game.is_checkmate
    assert screen._show_endgame_popup is True
    assert screen._result_text == "Checkmate — Sick Ken wins."
    assert not screen._can_accept_input()
    screen.draw()

    # Clicking Play Again restarts cleanly with the same settings
    rematch_pos = screen._btn_popup_rematch.base_rect.center
    screen.handle_events(
        [pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=rematch_pos)]
    )
    assert screen._show_endgame_popup is False
    assert screen._game.san_moves == []
    assert not screen._game.is_game_over
    assert screen._can_accept_input()
    screen.close()


def test_time_control_presets_and_exclusions():
    """Verify the 9 structured presets + No Timer and exclusion of Daily/Rated/Standard."""
    assert NO_TIMER_PRESET is None
    assert PRESET_BY_LABEL["No Timer"] is None
    assert len(TIME_CONTROL_PRESETS) == 9

    expected = [
        ("Bullet", 60, 0, "1 min"),
        ("Bullet", 60, 1, "1 + 1"),
        ("Bullet", 120, 1, "2 + 1"),
        ("Blitz", 180, 0, "3 min"),
        ("Blitz", 180, 2, "3 + 2"),
        ("Blitz", 300, 0, "5 min"),
        ("Rapid", 600, 0, "10 min"),
        ("Rapid", 600, 5, "10 + 5"),
        ("Rapid", 900, 10, "15 + 10"),
    ]
    actual = [
        (p.category, p.base_seconds, p.increment_seconds, p.label)
        for p in TIME_CONTROL_PRESETS
    ]
    assert actual == expected

    categories = {g[0].upper() for g in TIME_CONTROL_GROUPS}
    assert categories == {"BULLET", "BLITZ", "RAPID"}
    assert "DAILY" not in categories

    pygame.init()
    for res in ((1280, 720), (1024, 600), (1920, 1080)):
        surface = pygame.display.set_mode(res)
        screen = GameScreen(surface, _OPPONENT)
        assert not hasattr(screen, "_mode_rect")
        assert not hasattr(screen, "_btn_rated")
        assert set(screen._time_buttons.keys()) == {
            "No Timer",
            *(p.label for p in TIME_CONTROL_PRESETS),
        }
        assert screen._time_dropdown_panel_rect.bottom <= res[1]
        screen.close()


def test_time_control_dropdown():
    pygame.init()
    surface = pygame.display.set_mode((1280, 720))
    screen = GameScreen(surface, _OPPONENT)

    # 1. Default state and label formatting
    assert screen._selected_time_control == "5 min"
    assert screen.selected_time_control == TimeControlPreset("Blitz", 300, 0, "5 min")
    assert format_time_control_label(screen._selected_time_control) == "5 min (Blitz)"
    assert screen._time_buttons["5 min"].accent is True
    assert screen._time_buttons["No Timer"].accent is False
    assert screen._time_dropdown_open is False
    assert screen.captures_escape() is False
    assert get_time_control_category("No Timer") == "NO TIMER"
    assert get_time_control_category("1 min") == "BULLET"
    assert get_time_control_category("3 min") == "BLITZ"
    assert get_time_control_category("5 min") == "BLITZ"
    assert get_time_control_category("10 min") == "RAPID"
    assert format_time_control_label("No Timer") == "No Timer"
    assert format_time_control_label("1 min") == "1 min (Bullet)"
    assert format_time_control_label("3 + 2") == "3 + 2 (Blitz)"
    assert format_time_control_label("5 min") == "5 min (Blitz)"
    assert format_time_control_label("10 min") == "10 min (Rapid)"

    # Clicking a hidden time option (not overlapping Play) while dropdown is closed must NOT select it
    bullet_btn_pos = screen._time_buttons["1 min"].base_rect.center
    assert not screen._btn_play.base_rect.collidepoint(bullet_btn_pos)
    screen.handle_events(
        [pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=bullet_btn_pos)]
    )
    assert screen._selected_time_control == "5 min"
    assert screen.selected_time_control == TimeControlPreset("Blitz", 300, 0, "5 min")
    assert screen._time_dropdown_open is False
    assert screen._game_started is False

    blitz_btn_pos = screen._time_buttons["5 min"].base_rect.center

    # 2. Clicking the trigger opens the dropdown; clicking again closes it
    trigger_pos = screen._btn_time_dropdown.base_rect.center
    screen.handle_events(
        [pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=trigger_pos)]
    )
    assert screen._time_dropdown_open is True
    assert screen.captures_escape() is True
    screen.draw()

    screen.handle_events(
        [pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=trigger_pos)]
    )
    assert screen._time_dropdown_open is False
    assert screen.captures_escape() is False

    # 3. Open dropdown and select a time control option ("5 min")
    screen.handle_events(
        [pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=trigger_pos)]
    )
    assert screen._time_dropdown_open is True
    screen.handle_events(
        [pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=blitz_btn_pos)]
    )
    assert screen._selected_time_control == "5 min"
    assert screen.selected_time_control == TimeControlPreset("Blitz", 300, 0, "5 min")
    assert screen._time_dropdown_open is False
    assert screen._time_buttons["5 min"].accent is True
    assert screen._time_buttons["No Timer"].accent is False
    assert screen._game_started is False
    screen.draw()

    # 4. Open dropdown and close via outside click without changing selection or starting match
    screen.handle_events(
        [pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=trigger_pos)]
    )
    assert screen._time_dropdown_open is True
    outside_pos = (50, 50)
    screen.handle_events(
        [pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=outside_pos)]
    )
    assert screen._time_dropdown_open is False
    assert screen._selected_time_control == "5 min"
    assert screen._game_started is False

    # 5. Open dropdown and close via Escape key without changing selection
    screen.handle_events(
        [pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=trigger_pos)]
    )
    assert screen._time_dropdown_open is True
    screen.handle_events([pygame.event.Event(pygame.KEYDOWN, key=pygame.K_ESCAPE)])
    assert screen._time_dropdown_open is False
    assert screen._selected_time_control == "5 min"
    screen.close()


def test_time_control_modern_spacing_animation_and_responsiveness():
    """Verify consistent Time Control spacing, equal button geometry, smooth animations, and responsive fit."""
    pygame.init()
    for res in ((1280, 720), (1024, 600), (720, 1280), (480, 800)):
        surf = pygame.display.set_mode(res)
        gs = GameScreen(surf, _OPPONENT)

        # Vertical rhythm & alignment
        assert gs._no_timer_label_y > gs._profile_bottom
        assert gs._time_dropdown_rect.top > gs._no_timer_label_y
        assert gs._time_dropdown_panel_rect.top > gs._time_dropdown_rect.bottom
        assert gs._time_dropdown_rect.left == gs._time_dropdown_panel_rect.left
        assert gs._time_dropdown_rect.width == gs._time_dropdown_panel_rect.width
        assert gs._time_dropdown_panel_rect.right <= res[0]
        assert gs._time_dropdown_panel_rect.bottom <= res[1]

        # All 10 timer buttons have equal height, stay inside panel, and never overlap
        btn_rects = list(gs._time_buttons.values())
        heights = {b.base_rect.height for b in btn_rects}
        assert len(heights) == 1, f"Timer buttons must share equal height at {res}: {heights}"
        for b in btn_rects:
            assert gs._time_dropdown_panel_rect.contains(b.base_rect), f"Button overflow at {res}"
        for i in range(len(btn_rects)):
            for j in range(i + 1, len(btn_rects)):
                assert not btn_rects[i].base_rect.colliderect(btn_rects[j].base_rect), (
                    f"Timer buttons overlap at {res}"
                )

        # Desktop, compact, and 720px portrait must keep 3 columns per category
        if res[0] >= 720:
            blitz_rects = [gs._time_buttons[k].base_rect for k in ("3 min", "3 + 2", "5 min")]
            assert len({r.top for r in blitz_rects}) == 1, f"Expected 3-column Blitz row at {res}"
            gap_01 = blitz_rects[1].left - blitz_rects[0].right
            gap_12 = blitz_rects[2].left - blitz_rects[1].right
            assert gap_01 == gap_12 and gap_01 >= 4

        # Smooth expand/collapse + selection transition + real match clock config propagation
        trigger_pos = gs._btn_time_dropdown.base_rect.center
        gs.handle_events([pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=trigger_pos)])
        assert gs._time_dropdown_open is True
        gs.update(0.05)
        assert 0.0 < gs._time_dropdown_anim < 1.0
        gs.draw()
        for _ in range(15):
            gs.update(1 / 60.0)
        assert gs._time_dropdown_anim == 1.0
        gs.draw()

        # Select 10 + 5 Rapid and verify smooth selection state + collapse + MatchClock
        rapid_pos = gs._time_buttons["10 + 5"].base_rect.center
        gs.handle_events([pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=rapid_pos)])
        assert gs._selected_time_control == "10 + 5"
        assert gs._time_dropdown_open is False
        for _ in range(20):
            gs.update(1 / 60.0)
        assert gs._time_dropdown_anim == 0.0
        assert gs._time_btn_select_t["10 + 5"] > 0.9
        assert gs._time_btn_select_t["5 min"] < 0.1
        gs.draw()

        gs._start_match()
        gs._bot_worker.stop()
        gs._bot_worker = None
        assert gs.selected_time_control == TimeControlPreset("Rapid", 600, 5, "10 + 5")
        assert gs.match_clock.is_timed is True
        assert gs.match_clock.white_seconds == 600.0
        gs.match_clock.apply_increment("white")
        assert gs.match_clock.white_seconds == 605.0
        gs.close()

    # Verify warm charcoal palette consistency (no blue-gray cast) and custom stopwatch icon
    for rgb in (TC_TRIGGER_BG, TC_PANEL_BG, TC_BTN_BG):
        assert rgb[0] >= rgb[2], f"Expected warm charcoal (R >= B), got {rgb}"
    icon = GameScreen._load_tc_stopwatch_icon(30)
    assert icon.get_flags() & pygame.SRCALPHA
    assert 24 <= max(icon.get_size()) <= 34


def test_speech_bubble_entrance_dynamic_text_and_reduced_motion():
    """Verify Chess.com-style speech bubble entrance, dynamic text update, and reduced motion."""
    pygame.init()
    prev_env = os.environ.get("CHESSIQ_REDUCED_MOTION")
    try:
        # 1. Animated mode: bubble emerges first, text delays 250-350ms, then fades in
        os.environ["CHESSIQ_REDUCED_MOTION"] = "0"
        surf = pygame.display.set_mode((1280, 720))
        gs = GameScreen(surf, _OPPONENT)
        assert gs._reduced_motion is False
        assert gs._speech_bubble_progress() == (0.0, 0.0)

        # Advance ~140ms: bubble emerging, text still hidden
        for _ in range(9):
            gs.update(1 / 60.0)
        bt, tt = gs._speech_bubble_progress()
        assert 0.0 < bt < 1.0
        assert tt == 0.0
        gs.draw()

        # Advance past text delay (~380ms total): text fading in
        for _ in range(14):
            gs.update(1 / 60.0)
        bt, tt = gs._speech_bubble_progress()
        assert 0.5 < bt <= 1.0
        assert 0.0 < tt < 1.0
        gs.draw()

        # Finish entrance (~850ms total)
        for _ in range(30):
            gs.update(1 / 60.0)
        assert gs._speech_bubble_progress() == (1.0, 1.0)
        gs.draw()

        # Unrelated UI update (opening Time Control dropdown) must NOT replay bubble entrance
        gs.handle_events(
            [
                pygame.event.Event(
                    pygame.MOUSEBUTTONDOWN,
                    button=1,
                    pos=gs._btn_time_dropdown.base_rect.center,
                )
            ]
        )
        gs.update(1 / 60.0)
        assert gs._speech_bubble_progress() == (1.0, 1.0)

        # Dynamic dialogue text change re-animates ONLY text (bubble stays at 1.0)
        gs._game_started = True
        gs._chat_text = "Check! Watch the king."
        gs.update(1 / 60.0)
        bt_dyn, tt_dyn = gs._speech_bubble_progress()
        assert bt_dyn == 1.0
        assert 0.0 < tt_dyn < 1.0
        gs.draw()
        for _ in range(25):
            gs.update(1 / 60.0)
        assert gs._speech_bubble_progress() == (1.0, 1.0)
        gs.close()

        # 2. Reduced-motion preference: immediately visible without entrance delay
        os.environ["CHESSIQ_REDUCED_MOTION"] = "1"
        gs_rm = GameScreen(surf, _OPPONENT)
        assert gs_rm._reduced_motion is True
        assert gs_rm._speech_bubble_progress() == (1.0, 1.0)
        gs_rm._game_started = True
        gs_rm._chat_text = "King safely tucked away."
        gs_rm.update(1 / 60.0)
        assert gs_rm._speech_bubble_progress() == (1.0, 1.0)
        gs_rm.draw()
        gs_rm.close()
    finally:
        if prev_env is None:
            os.environ.pop("CHESSIQ_REDUCED_MOTION", None)
        else:
            os.environ["CHESSIQ_REDUCED_MOTION"] = prev_env


def test_premove_settings_toggle_and_gameplay():
    """Verify Settings sliding toggle (default ON) and full Premove gameplay lifecycle."""
    pygame.init()
    surf = pygame.display.set_mode((1280, 720))

    # 1. SettingsScreen sliding toggle defaults to ON and notifies callback cleanly
    changes: list[bool] = []
    settings = SettingsScreen(surf, on_premove_changed=changes.append)
    assert settings.premove_enabled is True
    assert settings._premove_knob_t == 1.0
    settings.draw()

    # Click toggle -> OFF
    settings.handle_events(
        [
            pygame.event.Event(
                pygame.MOUSEBUTTONDOWN,
                button=1,
                pos=settings._premove_switch_rect.center,
            )
        ]
    )
    assert settings.premove_enabled is False
    assert changes == [False]
    for _ in range(30):
        settings.update(1 / 60.0)
    assert settings._premove_knob_t < 0.05
    settings.draw()

    # Click toggle -> ON again
    settings.handle_events(
        [
            pygame.event.Event(
                pygame.MOUSEBUTTONDOWN,
                button=1,
                pos=settings._premove_switch_rect.center,
            )
        ]
    )
    assert settings.premove_enabled is True
    assert changes == [False, True]
    for _ in range(30):
        settings.update(1 / 60.0)
    assert settings._premove_knob_t > 0.95
    settings.draw()

    # 2. Pure premove target rules (pawn diagonals onto empty squares, sliding over blockers)
    start_board = chess.Board()
    e2_targets = set(premove_targets_from(start_board, chess.E2, chess.WHITE))
    assert {chess.E3, chess.E4, chess.D3, chess.F3}.issubset(e2_targets)
    a1_targets = set(premove_targets_from(start_board, chess.A1, chess.WHITE))
    assert chess.A8 in a1_targets and chess.H1 in a1_targets

    # 3. GameScreen premove: queue, cancel (right-click & ESC), auto-execute legal, discard illegal
    screen, _ = _game_screen("White")
    assert screen.premove_enabled is True

    # White plays 1. e4 -> now Black's turn
    screen._apply_player_move(chess.E2, chess.E4, None)
    _finish_animation(screen)
    assert screen._game.turn_color == chess.BLACK

    def click_sq(sq: int, button: int = 1) -> None:
        cx, cy = screen._board._square_center(sq)
        pos = (int(cx) + screen._board_area.left, int(cy) + screen._board_area.top)
        screen.handle_events(
            [pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=button, pos=pos)]
        )

    # Queue premove g1 -> f3 during Black's turn
    click_sq(chess.G1)
    assert screen._board.premove_selected == chess.G1
    assert screen.captures_escape() is True
    click_sq(chess.F3)
    assert screen._board.premove == (chess.G1, chess.F3, None)
    assert screen.captures_escape() is True
    screen.draw()

    # Right-click cancels queued premove
    click_sq(chess.D4, button=3)
    assert screen._board.premove is None
    assert screen.captures_escape() is False

    # Queue premove again and cancel with ESC
    click_sq(chess.G1)
    click_sq(chess.F3)
    assert screen._board.premove == (chess.G1, chess.F3, None)
    screen.handle_events([pygame.event.Event(pygame.KEYDOWN, key=pygame.K_ESCAPE)])
    assert screen._board.premove is None

    # Queue legal premove Nf3, then bot plays e5 -> Nf3 auto-executes after bot animation!
    click_sq(chess.G1)
    click_sq(chess.F3)
    assert screen._board.premove == (chess.G1, chess.F3, None)
    screen._apply_bot_move(chess.Move.from_uci("e7e5"))
    _finish_animation(screen)
    # One more update executes the queued premove
    screen.update(1 / 60.0)
    _finish_animation(screen)
    assert screen._board.premove is None
    assert screen._game.san_moves == ["e4", "e5", "Nf3"]
    assert screen._game.turn_color == chess.BLACK

    # Queue illegal premove (f3 -> e5 is legal only if e5 stays; test d1 -> h5 then bot plays d5?
    # Better: queue e4 -> e5 while e5 is occupied by Black pawn — illegal after Black plays Nc6!)
    click_sq(chess.E4)
    click_sq(chess.E5)
    assert screen._board.premove == (chess.E4, chess.E5, None)
    screen._apply_bot_move(chess.Move.from_uci("b8c6"))
    _finish_animation(screen)
    screen.update(1 / 60.0)
    assert screen._board.premove is None
    assert screen._game.san_moves == ["e4", "e5", "Nf3", "Nc6"]
    assert screen._game.turn_color == chess.WHITE
    screen.close()

    # 4. Premove queued WHILE bot's reply is still animating + natural bot pacing after premove
    screen2, worker2 = _game_screen("White")
    worker2.last_move_source = "book"
    worker2.last_compute_seconds = 0.002
    screen2._apply_player_move(chess.E2, chess.E4, None)
    screen2.update(1 / 60.0)
    worker2.results.append(chess.Move.from_uci("e7e5"))
    screen2.update(1 / 60.0)
    # Bot wait timer must NOT finish while player's e4 is still sliding
    wait_while_anim = screen2._bot_wait_remaining
    assert screen2._board.is_animating and wait_while_anim >= 0.50
    screen2.update(1 / 60.0)
    assert screen2._bot_wait_remaining == wait_while_anim, "Bot wait must not drain during player slide"
    _advance_bot_pacing(screen2)
    assert screen2._board.is_animating and screen2._game.turn_color == chess.WHITE

    def click2(sq: int, button: int = 1) -> None:
        cx, cy = screen2._board._square_center(sq)
        pos = (int(cx) + screen2._board_area.left, int(cy) + screen2._board_area.top)
        screen2.handle_events(
            [pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=button, pos=pos)]
        )
        screen2.handle_events(
            [pygame.event.Event(pygame.MOUSEBUTTONUP, button=button, pos=pos)]
        )

    # Queue Nf3 mid-bot-animation -> fires immediately once bot's e5 lands
    click2(chess.G1)
    click2(chess.F3)
    assert screen2._board.premove == (chess.G1, chess.F3, None)
    _finish_animation(screen2)
    assert screen2._game.san_moves == ["e4", "e5", "Nf3"]

    # 5. Half-finished premove selection promotes cleanly to live selection when bot lands
    click2(chess.F1)
    assert screen2._board.premove_selected == chess.F1
    worker2.results.append(chess.Move.from_uci("b8c6"))
    _advance_bot_pacing(screen2)
    _finish_animation(screen2)
    assert screen2._board.premove_selected is None
    assert screen2._board.selected_square == chess.F1
    click2(chess.C4)
    _finish_animation(screen2)
    assert screen2._game.san_moves == ["e4", "e5", "Nf3", "Nc6", "Bc4"]

    # 6. Drag-and-drop works for both premoves (during opponent's turn) and live moves
    def drag2(from_sq: int, to_sq: int) -> None:
        fx, fy = screen2._board._square_center(from_sq)
        tx, ty = screen2._board._square_center(to_sq)
        p0 = (int(fx) + screen2._board_area.left, int(fy) + screen2._board_area.top)
        p1 = (int(tx) + screen2._board_area.left, int(ty) + screen2._board_area.top)
        screen2.handle_events([pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=p0)])
        screen2.handle_events([pygame.event.Event(pygame.MOUSEMOTION, pos=p1, buttons=(1, 0, 0))])
        screen2.draw()
        screen2.handle_events([pygame.event.Event(pygame.MOUSEBUTTONUP, button=1, pos=p1)])

    # Drag d2 -> d3 during Black's turn queues premove
    assert screen2._game.turn_color == chess.BLACK
    drag2(chess.D2, chess.D3)
    assert screen2._board.premove == (chess.D2, chess.D3, None)
    worker2.results.append(chess.Move.from_uci("g8f6"))
    _advance_bot_pacing(screen2)
    _finish_animation(screen2)
    assert screen2._game.san_moves == ["e4", "e5", "Nf3", "Nc6", "Bc4", "Nf6", "d3"]

    # Bot plays Bc5, then live drag O-O (e1 -> g1) drops cleanly
    worker2.results.append(chess.Move.from_uci("f8c5"))
    _advance_bot_pacing(screen2)
    _finish_animation(screen2)
    drag2(chess.E1, chess.G1)
    assert screen2._game.san_moves[-1] == "O-O"
    assert not screen2._board.is_animating  # dropped directly onto target square
    screen2.close()

    # 7. When premove_enabled=False, clicks during opponent's turn do NOT queue premoves
    disabled_screen = GameScreen(surf, _OPPONENT, premove_enabled=False)
    disabled_screen._selected_side = "White"
    disabled_screen._start_match()
    disabled_screen._bot_worker.stop()
    disabled_screen._bot_worker = None
    assert disabled_screen.premove_enabled is False
    disabled_screen._apply_player_move(chess.E2, chess.E4, None)
    _finish_animation(disabled_screen)
    cx, cy = disabled_screen._board._square_center(chess.G1)
    disabled_screen.handle_events(
        [pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=(int(cx), int(cy)))]
    )
    assert disabled_screen._board.premove_selected is None
    assert disabled_screen._board.premove is None
    disabled_screen.close()


def test_hamburger_toggle_does_not_overlap_profile_bubble():
    pygame.init()
    for resolution in ((1280, 720), (1920, 1080), (720, 1280), (480, 800)):
        screen = GameScreen(pygame.display.set_mode(resolution), _OPPONENT)
        assert not screen._menu.toggle_rect.colliderect(screen._tagline_rect)
        assert screen._menu.toggle_rect.bottom <= screen._tagline_rect.top
        screen.close()


def test_hamburger_menu_actions_after_game_over():
    """Hamburger menu hides Resign/Offer Draw once game ends, showing only Resume/Restart/Quit."""
    screen, _ = _game_screen("White")
    # Active match includes Resign and Offer Draw
    assert "resign" in screen._menu.actions
    assert "offer_draw" in screen._menu.actions

    # End game by resignation -> only Resume, Restart, Quit remain
    screen._resign()
    assert screen._game.is_game_over
    assert screen._menu.actions == ("resume", "restart", "opponent_select")
    assert "resign" not in screen._menu.actions
    assert "offer_draw" not in screen._menu.actions

    # Open menu and draw cleanly
    screen._show_endgame_popup = False
    screen.handle_events(
        [
            pygame.event.Event(
                pygame.MOUSEBUTTONDOWN, button=1, pos=screen._menu.toggle_rect.center
            )
        ]
    )
    assert screen._menu.is_open is True
    screen.draw()

    # Clicking Restart starts a fresh game and restores active match actions
    restart_btn = dict(screen._menu._items)["restart"]
    screen.handle_events(
        [
            pygame.event.Event(
                pygame.MOUSEBUTTONDOWN, button=1, pos=restart_btn.base_rect.center
            )
        ]
    )
    assert not screen._game.is_game_over
    assert "resign" in screen._menu.actions
    assert "offer_draw" in screen._menu.actions

    # Checkmate also switches menu to Resume / Restart / Quit
    screen._apply_player_move(chess.F2, chess.F3, None)
    _finish_animation(screen)
    screen._apply_bot_move(chess.Move.from_uci("e7e5"))
    _finish_animation(screen)
    screen._apply_player_move(chess.G2, chess.G4, None)
    _finish_animation(screen)
    screen._apply_bot_move(chess.Move.from_uci("d8h4"))
    _finish_animation(screen)
    assert screen._game.is_checkmate
    assert screen._menu.actions == ("resume", "restart", "opponent_select")

    # Clicking Quit navigates to opponent_select
    screen._show_endgame_popup = False
    screen._menu.toggle()
    quit_btn = dict(screen._menu._items)["opponent_select"]
    screen.handle_events(
        [
            pygame.event.Event(
                pygame.MOUSEBUTTONDOWN, button=1, pos=quit_btn.base_rect.center
            )
        ]
    )
    assert screen.next_screen == "opponent_select"
    screen.close()


def test_move_review_arrow_keys_click_and_nav_bar():
    """Verify Chess.com/Lichess move review via Left/Right/Home/End keys, move clicks, and nav buttons."""
    screen, _ = _game_screen("White")
    # Play 4 plies: 1. e4 e5 2. Nf3 Nc6
    for idx, san in enumerate(("e4", "e5", "Nf3", "Nc6")):
        mv = screen._game.board.parse_san(san)
        if idx % 2 == 0:
            screen._apply_player_move(mv.from_square, mv.to_square, None)
        else:
            screen._apply_bot_move(mv)
        _finish_animation(screen)

    live_fen = screen._game.fen
    assert screen.is_reviewing is False
    assert screen.viewed_ply == 3

    # 1. Left arrow steps backward through history without mutating live ChessGame
    screen.handle_events([pygame.event.Event(pygame.KEYDOWN, key=pygame.K_LEFT)])
    assert screen.is_reviewing is True
    assert screen.viewed_ply == 2
    assert screen._board.is_reviewing is True
    assert screen._board._review_board.fen() == screen._game.move_history[2].fen_after
    assert screen._game.fen == live_fen
    assert not screen._can_accept_input()
    assert screen.captures_escape() is True
    screen.draw()

    # Step all the way back to starting position (-1) and clamp cleanly
    for _ in range(5):
        screen.handle_events([pygame.event.Event(pygame.KEYDOWN, key=pygame.K_LEFT)])
    assert screen.viewed_ply == -1
    assert screen._board._review_board.fen() == screen._game.initial_fen
    assert screen._current_opening == "Starting Position"
    screen.draw()

    # 2. Right arrow steps forward to ply 0 (1. e4), then End returns to live
    screen.handle_events([pygame.event.Event(pygame.KEYDOWN, key=pygame.K_RIGHT)])
    assert screen.viewed_ply == 0
    assert screen._board._review_board.fen() == screen._game.move_history[0].fen_after
    screen.handle_events([pygame.event.Event(pygame.KEYDOWN, key=pygame.K_END)])
    assert screen.is_reviewing is False
    assert screen.viewed_ply == 3
    assert screen._board.is_reviewing is False

    # 3. Clicking a specific move cell in the move list jumps directly to that ply
    cell_1 = screen._move_cell_rect(1)  # 1... e5
    screen.handle_events(
        [pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=cell_1.center)]
    )
    assert screen.is_reviewing is True
    assert screen.viewed_ply == 1
    assert screen._board._review_board.fen() == screen._game.move_history[1].fen_after
    screen.draw()

    # 4. All 4 on-screen navigation buttons (first, prev, next, last) work identically
    assert tuple(screen._review_buttons.keys()) == ("first", "prev", "next", "last")
    assert "play" not in screen._review_buttons
    # Equal widths and non-overlapping inside sidebar
    nav_rects = [b.base_rect for b in screen._review_buttons.values()]
    assert len({r.width for r in nav_rects}) == 1
    assert len({r.height for r in nav_rects}) == 1
    for i in range(len(nav_rects) - 1):
        assert nav_rects[i].right < nav_rects[i + 1].left

    screen.handle_events(
        [
            pygame.event.Event(
                pygame.MOUSEBUTTONDOWN,
                button=1,
                pos=screen._review_buttons["first"].base_rect.center,
            )
        ]
    )
    assert screen.viewed_ply == -1
    screen.handle_events(
        [
            pygame.event.Event(
                pygame.MOUSEBUTTONDOWN,
                button=1,
                pos=screen._review_buttons["next"].base_rect.center,
            )
        ]
    )
    assert screen.viewed_ply == 0
    screen.handle_events(
        [
            pygame.event.Event(
                pygame.MOUSEBUTTONDOWN,
                button=1,
                pos=screen._review_buttons["prev"].base_rect.center,
            )
        ]
    )
    assert screen.viewed_ply == -1
    screen.handle_events(
        [
            pygame.event.Event(
                pygame.MOUSEBUTTONDOWN,
                button=1,
                pos=screen._review_buttons["last"].base_rect.center,
            )
        ]
    )
    assert screen.is_reviewing is False

    # 5. Works after game ends (with popup dismissed)
    screen._resign()
    screen._show_endgame_popup = False
    screen.handle_events([pygame.event.Event(pygame.KEYDOWN, key=pygame.K_LEFT)])
    assert screen.is_reviewing is True
    assert screen.viewed_ply == 2
    screen.handle_events([pygame.event.Event(pygame.KEYDOWN, key=pygame.K_ESCAPE)])
    assert screen.is_reviewing is False
    screen.draw()
    screen.close()


def test_match_clock_countdown_increment_and_flag_fall():
    """Verify pure MatchClock countdown, increment, flag-fall, pause, and GameScreen clock integration."""
    # 1. Pure MatchClock unit checks
    untimed = MatchClock(None)
    assert untimed.is_timed is False
    assert untimed.tick(10.0, "white") is None
    assert untimed.flagged_color is None

    preset_3_2 = PRESET_BY_LABEL["3 + 2"]
    assert preset_3_2 is not None
    clock = MatchClock(preset_3_2)
    assert clock.is_timed is True
    assert clock.white_seconds == 180.0
    assert clock.black_seconds == 180.0
    assert format_clock_seconds(180.0) == "3:00"

    assert clock.tick(5.0, "white") is None
    assert clock.white_seconds == 175.0
    assert clock.black_seconds == 180.0

    # White completes a move -> receives +2s increment, active switches to black
    clock.apply_increment("white")
    assert clock.white_seconds == 177.0
    assert clock.active_color == "black"

    assert clock.tick(10.0) is None
    assert clock.black_seconds == 170.0

    # Pause stops countdown
    clock.pause()
    assert clock.tick(25.0, "black") is None
    assert clock.black_seconds == 170.0
    clock.resume()

    # Flag-fall when clock reaches zero
    flagged = clock.tick(200.0, "black")
    assert flagged == "black"
    assert clock.black_seconds == 0.0
    assert clock.flagged_color == "black"
    assert clock.paused is True
    assert format_clock_seconds(0.0) == "0:00"

    # 2. GameScreen integration with timed preset vs No Timer
    pygame.init()
    surface = pygame.display.set_mode((1280, 720))
    screen = GameScreen(surface, _OPPONENT)
    screen._selected_side = "White"
    screen.select_time_control("1 + 1")
    screen._start_match()
    screen._bot_worker.stop()
    screen._bot_worker = None

    assert screen.match_clock.is_timed is True
    assert screen.match_clock.white_seconds == 60.0
    assert screen.match_clock.black_seconds == 60.0

    # Tick 4 seconds on White's turn, then play 1. e4 (+1s increment)
    screen.update(4.0)
    assert abs(screen.match_clock.white_seconds - 56.0) < 1e-6
    screen._apply_player_move(chess.E2, chess.E4, None)
    _finish_animation(screen)
    assert abs(screen.match_clock.white_seconds - 57.0) < 1e-6

    # Tick 65 seconds on Black's turn -> Black flags on time, White wins
    screen.update(65.0)
    assert screen._game.is_game_over is True
    assert screen._game.outcome.status == "timeout"
    assert screen._game.outcome.winner == "white"
    assert screen._game.outcome.termination == "Time Out"
    assert screen.match_clock.paused is True
    assert screen._show_endgame_popup is True
    screen.draw()
    screen.close()


def test_animation_mathematics():
    """Verify easing bounds/monotonicity, distance scaling, cap, progress bounds, and dt clamping."""
    for mode in ("ease_out_cubic", "ease_in_out_cubic"):
        assert ease_progress(-0.5, mode) == 0.0
        assert ease_progress(0.0, mode) == 0.0
        assert ease_progress(1.0, mode) == 1.0
        assert ease_progress(1.5, mode) == 1.0
        prev = -1.0
        for i in range(101):
            val = ease_progress(i / 100.0, mode)
            assert 0.0 <= val <= 1.0
            assert val >= prev
            prev = val

    d_short = compute_move_duration(chess.E2, chess.E3, "P")
    d_mid = compute_move_duration(chess.D1, chess.D5, "Q")
    d_long = compute_move_duration(chess.A1, chess.H8, "B")
    d_knight = compute_move_duration(chess.G1, chess.F3, "N")
    assert ANIMATION_BASE_DURATION <= d_short < d_mid <= d_long <= ANIMATION_MAX_DURATION
    assert move_distance_in_squares(chess.G1, chess.F3, "N") == ANIMATION_KNIGHT_DISTANCE
    assert d_knight < d_long

    game = ChessGame()
    board = _board(game)
    info = game.make_san("e4")
    board.animate_move(info)
    dur = board._move_anim_duration
    assert dur > MAX_ANIMATION_DT

    board.update(2.0)  # large frame hitch clamped to MAX_ANIMATION_DT
    assert board.is_animating, "Single clamped hitch must not teleport move to completion"
    assert board.animation_progress <= (MAX_ANIMATION_DT / dur) + 1e-6
    assert 0.0 < board.animation_progress < 1.0

    for _ in range(20):
        board.update(2.0)
    assert not board.is_animating
    assert board.animation_progress == 0.0


def test_animation_queue_captures_resize_and_flip():
    """Verify FIFO animation queue, capture fade, castling, en passant, resize, and flip mid-animation."""
    game = ChessGame()
    board = _board(game)
    surf = pygame.Surface((800, 720), pygame.SRCALPHA)

    m1 = game.make_san("e4")
    m2 = game.make_san("d5")
    m3 = game.make_san("exd5")
    board.animate_move(m1)
    board.animate_move(m2)
    board.animate_move(m3)

    assert board._move_anim is m1
    assert board.animation_queue_length == 2
    board.update(0.05)
    board.draw(surf)
    assert board._move_anim is m1

    # Flip and resize mid-animation must preserve progress and never crash
    prog_before = board.animation_progress
    board.set_flipped(True)
    board.resize(960, 760, 1.1)
    assert abs(board.animation_progress - prog_before) < 1e-6
    surf2 = pygame.Surface((960, 760), pygame.SRCALPHA)
    board.draw(surf2)

    seen_moves = [board._move_anim.uci]
    for _ in range(80):
        board.update(1 / 60)
        board.draw(surf2)
        if board._move_anim is not None and board._move_anim.uci != seen_moves[-1]:
            seen_moves.append(board._move_anim.uci)
    assert seen_moves == ["e2e4", "d7d5", "e4d5"]
    assert not board.is_animating

    # Capture fade-out triggers after capture lands and finishes cleanly
    game_cap = ChessGame("rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2")
    board_cap = _board(game_cap)
    cap_info = game_cap.make_san("exd5")
    board_cap.animate_move(cap_info)
    while board_cap.is_animating:
        board_cap.update(1 / 60)
        board_cap.draw(surf)
    assert board_cap._capture_fade is not None
    steps = int(CAPTURE_FADE_DURATION / (1 / 60)) + 4
    for _ in range(steps):
        board_cap.update(1 / 60)
        board_cap.draw(surf)
    assert board_cap._capture_fade is None

    # En passant capture animation renders cleanly
    game_ep = ChessGame("rnbqkbnr/ppp1pppp/8/3pP3/8/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 3")
    board_ep = _board(game_ep)
    ep_info = game_ep.make_san("exd6")
    assert ep_info.is_en_passant and ep_info.captured_square == chess.D5
    board_ep.animate_move(ep_info)
    for _ in range(30):
        board_ep.update(1 / 60)
        board_ep.draw(surf)
    assert not board_ep.is_animating


def test_bot_pacing_waits_for_player_animation_and_respects_engine_time():
    """Verify bot starts thinking in background, waits for player anim, applies pacing, and subtracts engine time."""
    assert get_bot_think_range("sick", True)[0] < get_bot_think_range("condition", True)[0]
    assert get_bot_think_range("focus", True)[1] <= get_bot_think_range("focus", False)[1]

    screen, worker = _game_screen("White")
    worker.last_move_source = "book"
    worker.last_compute_seconds = 0.002

    screen._apply_player_move(chess.E2, chess.E4, None)
    assert screen._board.is_animating

    # Bot search begins in background even while player animation is active
    screen.update(1 / 60)
    assert len(worker.requests) == 1
    assert screen._bot_thinking

    # Bot returns immediately (book move) while player animation still runs
    worker.results.append(chess.Move.from_uci("c7c5"))
    screen.update(1 / 60)
    assert screen._board.is_animating
    assert screen._game.uci_moves == ["e2e4"], "Bot must not start moving while player anim runs"
    assert screen._bot_thinking, "Thinking indicator must stay visible during pause"
    assert screen._bot_wait_remaining > 0.25

    # Advance until player anim finishes AND pacing pause completes -> bot glides via same pipeline
    _advance_bot_pacing(screen)
    assert screen._game.uci_moves == ["e2e4", "c7c5"]
    assert screen._board.is_animating, "Bot move must enter the same animation pipeline"
    _finish_animation(screen)
    assert not screen._board.is_animating
    screen.close()

    # Existing engine thinking time is subtracted (no double wait)
    screen2, worker2 = _game_screen("White")
    worker2.last_move_source = "style-selected"
    worker2.last_compute_seconds = 2.5

    screen2._apply_player_move(chess.E2, chess.E4, None)
    _finish_animation(screen2)
    assert len(worker2.requests) == 1
    worker2.results.append(chess.Move.from_uci("c7c5"))
    screen2.update(1 / 60)
    assert screen2._game.uci_moves == ["e2e4", "c7c5"]
    assert screen2._board.is_animating
    screen2.close()


def test_endgame_popup_waits_for_final_animation():
    """Verify checkmate popup waits until the checkmating piece finishes sliding."""
    screen, _ = _game_screen("White")
    screen._apply_player_move(chess.F2, chess.F3, None)
    _finish_animation(screen)
    screen._apply_bot_move(chess.Move.from_uci("e7e5"))
    _finish_animation(screen)
    screen._apply_player_move(chess.G2, chess.G4, None)
    _finish_animation(screen)

    screen._apply_bot_move(chess.Move.from_uci("d8h4"))
    assert screen._game.is_checkmate
    assert screen._board.is_animating
    assert screen._show_endgame_popup is False

    _finish_animation(screen)
    assert not screen._board.is_animating
    assert screen._show_endgame_popup is True
    screen.close()


def test_perf_overlay_toggle_and_stats():
    """Verify F3 performance overlay starts hidden, toggles cleanly, and computes percentiles."""
    from main import _PerfOverlay

    overlay = _PerfOverlay()
    assert overlay.visible is False
    overlay.toggle()
    assert overlay.visible is True
    for ms in (15.0, 16.0, 16.5, 17.0, 18.0, 22.0):
        overlay.record(ms)
    s = overlay.stats()
    assert s["worst"] == 22.0
    assert s["p95"] >= s["avg"] > 0.0
    assert s["fps"] > 0.0
    surf = pygame.Surface((400, 300))
    overlay.draw(surf)
    overlay.toggle()
    assert overlay.visible is False


def test_game_over_dialog_result_content():
    """Verify player-perspective headlines and subtitles for every supported outcome and fallback."""
    cases = [
        # (GameOutcome, player_color, expected_category, expected_headline, expected_subtitle)
        (GameOutcome(True, "checkmate", "white", "Checkmate", ""), chess.WHITE, "win", "You Won", "You won by checkmate"),
        (GameOutcome(True, "checkmate", "white", "Checkmate", ""), chess.BLACK, "loss", "You Lost", "Sick Ken won by checkmate"),
        (GameOutcome(True, "resignation", "black", "Resignation", ""), chess.WHITE, "loss", "You Lost", "You resigned"),
        (GameOutcome(True, "resignation", "white", "Resignation", ""), chess.WHITE, "win", "You Won", "Sick Ken resigned"),
        (GameOutcome(True, "timeout", "black", "Time Out", ""), chess.WHITE, "loss", "You Lost", "You ran out of time"),
        (GameOutcome(True, "timeout", "white", "Time Out", ""), chess.WHITE, "win", "You Won", "Sick Ken ran out of time"),
        (GameOutcome(True, "stalemate", None, "Stalemate", ""), chess.WHITE, "draw", "Draw", "Draw by stalemate"),
        (GameOutcome(True, "draw_agreed", None, "Agreement", ""), chess.WHITE, "draw", "Draw", "Draw by agreement"),
        (GameOutcome(True, "insufficient_material", None, "Insufficient Material", ""), chess.WHITE, "draw", "Draw", "Draw by insufficient material"),
        (GameOutcome(True, "threefold_repetition", None, "Threefold Repetition", ""), chess.WHITE, "draw", "Draw", "Draw by repetition"),
        (GameOutcome(True, "fivefold_repetition", None, "Fivefold Repetition", ""), chess.WHITE, "draw", "Draw", "Draw by repetition"),
        (GameOutcome(True, "fifty_moves", None, "50-Move Rule", ""), chess.WHITE, "draw", "Draw", "Draw by the 50-move rule"),
        (GameOutcome(True, "seventyfive_moves", None, "75-Move Rule", ""), chess.WHITE, "draw", "Draw", "Draw by the 75-move rule"),
        # Unknown fallbacks must never crash
        (GameOutcome(True, "mystery_reason", None, None, ""), chess.WHITE, "draw", "Draw", "Game ended in a draw"),
        (GameOutcome(True, "mystery_reason", "white", "Forfeit", ""), chess.WHITE, "win", "You Won", "You won by forfeit"),
    ]
    for outcome, p_col, exp_cat, exp_head, exp_sub in cases:
        cat, head, sub = format_endgame_dialog_text(outcome, p_col, "Sick Ken")
        assert (cat, head, sub) == (exp_cat, exp_head, exp_sub), (
            f"Mismatch for {outcome.status}: got {(cat, head, sub)}"
        )


def test_game_over_dialog_responsive_layout_and_interaction():
    """Verify Game Over dialog fits cleanly across narrow (455x414), small, and wide windows."""
    pygame.init()
    for size in ((455, 414), (360, 360), (800, 600), (1280, 720), (1920, 1080)):
        surf = pygame.display.set_mode(size)
        gs = GameScreen(surf, _OPPONENT)
        gs._bot = KenBot("Sick Ken", log_provenance=False)
        gs._bot_worker = FakeWorker()
        gs._selected_side = "White"
        gs._start_match()
        gs._resign()
        for _ in range(25):
            gs.update(1 / 60)
        gs.draw()

        screen_rect = surf.get_rect()
        r = gs._popup_rect
        assert screen_rect.contains(r), f"Card {r} overflows window {size}"

        # Close button hit area >= 40x40 and inside card
        close_r = gs._btn_popup_close.base_rect
        assert close_r.width >= DIALOG_CLOSE_SIZE and close_r.height >= DIALOG_CLOSE_SIZE
        assert r.contains(close_r), f"Close button {close_r} overflows card {r} at {size}"

        # Action buttons inside card and non-overlapping
        rem_r = gs._btn_popup_rematch.base_rect
        bot_r = gs._btn_popup_bots.base_rect
        assert r.contains(rem_r) and r.contains(bot_r), f"Buttons overflow card at {size}"
        assert not rem_r.colliderect(bot_r), f"Buttons overlap at {size}"

        # Avatars inside card, non-overlapping with each other or buttons
        av_r = gs._popup_avatar_size // 2
        pl_av = pygame.Rect(
            gs._popup_player_cx - av_r, gs._popup_avatar_cy - av_r, av_r * 2, av_r * 2
        )
        bot_av = pygame.Rect(
            gs._popup_bot_cx - av_r, gs._popup_avatar_cy - av_r, av_r * 2, av_r * 2
        )
        assert r.contains(pl_av) and r.contains(bot_av), f"Avatars overflow card at {size}"
        assert not pl_av.colliderect(bot_av), f"Avatars overlap at {size}"
        assert not pl_av.colliderect(rem_r) and not bot_av.colliderect(bot_r)
        gs.close()

    # Interaction checks at narrow 455x414: close button works, clicks don't leak, Enter/New Bot work
    surf = pygame.display.set_mode((455, 414))
    gs = GameScreen(surf, _OPPONENT)
    gs._bot = KenBot("Sick Ken", log_provenance=False)
    gs._bot_worker = FakeWorker()
    gs._selected_side = "White"
    gs._start_match()
    gs._resign()
    assert gs._show_endgame_popup is True

    # Click inside dialog body must not leak to board or close dialog
    gs.handle_events(
        [pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=gs._popup_rect.center)]
    )
    assert gs._show_endgame_popup is True
    assert gs._board.selected_square is None

    # Click enlarged close button closes dialog (even at 455x414 where hamburger sits behind it)
    gs.handle_events(
        [pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=gs._btn_popup_close.base_rect.center)]
    )
    assert gs._show_endgame_popup is False
    assert gs._menu.is_open is False

    # Re-open and verify Enter triggers Play Again
    gs._open_endgame_popup()
    gs.handle_events([pygame.event.Event(pygame.KEYDOWN, key=pygame.K_RETURN)])
    assert gs._show_endgame_popup is False
    assert not gs._game.is_game_over

    # Resign again and verify New Bot navigates to opponent_select
    gs._resign()
    gs.handle_events(
        [pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=gs._btn_popup_bots.base_rect.center)]
    )
    assert gs.next_screen == "opponent_select"
    gs.close()


def test_game_over_dialog_entrance_animation():
    """Verify dialog entrance starts at 0, clamps large dt, stays in [0, 1], and completes."""
    screen, _ = _game_screen("White")
    screen._resign()
    assert screen._show_endgame_popup is True
    assert screen.popup_entrance_progress == 0.0

    # Large frame hitch (2.0s) must be clamped to MAX_ANIMATION_DT
    screen.update(2.0)
    assert 0.0 < screen.popup_entrance_progress <= (MAX_ANIMATION_DT / DIALOG_ENTRANCE_DURATION) + 1e-6
    assert screen.popup_entrance_progress < 1.0
    screen.draw()

    # Repeated updates complete cleanly and never exceed 1.0
    for _ in range(15):
        screen.update(2.0)
        assert 0.0 <= screen.popup_entrance_progress <= 1.0
        screen.draw()
    assert screen.popup_entrance_progress == 1.0
    screen.close()


def test_opponent_select_screen_redesign():
    """Verify OpponentSelectScreen profiles, navigation, hover animation, and responsive layouts."""
    pygame.init()

    # 1. Preserved opponent identities, ratings, and recommended spotlight
    assert [(b["name"], b["elo"], bool(b["recommended"])) for b in _BOTS] == [
        ("Sick Ken", 1350, False),
        ("Focus Ken", 1800, True),
        ("Condition Ken", 2350, False),
    ]

    # 2. Selecting each bot sets selected_opponent and navigates to game
    for idx, expected_bot in enumerate(_BOTS):
        surf = pygame.display.set_mode((1280, 720))
        scr = OpponentSelectScreen(surf)
        assert len(scr._avatars) == 3
        assert all(av.get_width() == av.get_height() == scr._avatar_size for av in scr._avatars)

        # Hover card smoothly increases _card_hover_t
        scr.handle_events(
            [pygame.event.Event(pygame.MOUSEMOTION, pos=scr._cards[idx].center)]
        )
        for _ in range(15):
            scr.update(1 / 60.0)
        assert scr._card_hover_t[idx] > 0.5
        scr.draw()

        # Click SELECT button (aligned with current hovered card lift)
        btn_center = scr._buttons[idx].base_rect.center
        scr.handle_events(
            [pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=btn_center)]
        )
        assert scr.selected_opponent is expected_bot
        assert scr.next_screen == "game"

    # 3. BACK button returns to dashboard
    surf = pygame.display.set_mode((1280, 720))
    scr = OpponentSelectScreen(surf)
    scr.handle_events(
        [pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=scr._btn_back.base_rect.center)]
    )
    assert scr.next_screen == "dashboard"

    # 4. Responsive layouts: desktop, tablet, mobile
    for size, expected_mode in (((1280, 720), "desktop"), ((800, 600), "tablet"), ((455, 720), "mobile")):
        w, h = size
        surf = pygame.display.set_mode(size)
        scr = OpponentSelectScreen(surf)
        assert scr._layout_mode == expected_mode
        assert len(scr._cards) == 3
        for idx, card in enumerate(scr._cards):
            assert 0 <= card.left and card.right <= w, f"Card {idx} horizontal overflow at {size}"
            assert card.contains(scr._buttons[idx].base_rect), f"SELECT overflows card {idx} at {size}"
        # Cards must never overlap each other
        for i in range(3):
            for j in range(i + 1, 3):
                assert not scr._cards[i].colliderect(scr._cards[j]), f"Cards {i}/{j} overlap at {size}"
        if expected_mode in ("desktop", "tablet"):
            assert scr._cards[0].right < scr._cards[1].left
            assert scr._cards[1].right < scr._cards[2].left
            assert scr._btn_back.base_rect.bottom <= h
        else:
            assert scr._cards[0].bottom < scr._cards[1].top
            assert scr._cards[1].bottom < scr._cards[2].top
            assert scr._max_scroll_y > 0
            scr.handle_events([pygame.event.Event(pygame.MOUSEWHEEL, x=0, y=-3)])
            assert scr._scroll_y > 0
        scr.update(1 / 60.0)
        scr.draw()


def test_user_chooses_promotion_piece_live_and_premove():
    """Verify live drag, premove click, and premove drag always open Q/R/B/N picker instead of auto-queening."""
    def _sq_screen_center(screen: GameScreen, sq: int) -> tuple[int, int]:
        px, py = screen._board.square_to_pixel(sq)
        half = screen._board._sq // 2
        return (
            screen._board_area.left + px + half,
            screen._board_area.top + py + half,
        )

    def _promo_screen_center(screen: GameScreen, choice: str) -> tuple[int, int]:
        rect = dict(screen._board.promotion_rects())[choice]
        return (
            screen._board_area.left + rect.centerx,
            screen._board_area.top + rect.centery,
        )

    # 1. Live drag-and-drop promotion opens picker and promotes to chosen Rook (not Queen)
    screen, _worker = _game_screen("White")
    screen._game = ChessGame("8/P6k/8/8/8/8/8/K7 w - - 0 1")
    screen._board.attach_game(screen._game, flipped=False)
    a7 = _sq_screen_center(screen, chess.A7)
    a8 = _sq_screen_center(screen, chess.A8)
    screen.handle_events([pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=a7)])
    screen.handle_events([pygame.event.Event(pygame.MOUSEMOTION, pos=a8)])
    screen.handle_events([pygame.event.Event(pygame.MOUSEBUTTONUP, button=1, pos=a8)])
    assert screen._board.promotion_pending is True
    assert screen._game.piece_at(chess.A8) is None  # did NOT auto-promote to queen
    r_pos = _promo_screen_center(screen, "r")
    screen.handle_events([pygame.event.Event(pygame.MOUSEMOTION, pos=r_pos)])
    assert screen._board._promotion_hover == "r"
    screen.draw()
    screen.handle_events([pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=r_pos)])
    assert screen._board.promotion_pending is False
    assert screen._game.piece_at(chess.A8) == chess.Piece(chess.ROOK, chess.WHITE)
    screen.close()

    # 2. Premove click promotion opens picker during Black's turn and executes chosen Bishop
    screen, worker = _game_screen("White")
    screen._game = ChessGame("8/P6k/8/8/8/8/8/K7 b - - 0 1")
    screen._board.attach_game(screen._game, flipped=False)
    a7 = _sq_screen_center(screen, chess.A7)
    a8 = _sq_screen_center(screen, chess.A8)
    screen.handle_events([pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=a7)])
    screen.handle_events([pygame.event.Event(pygame.MOUSEBUTTONUP, button=1, pos=a7)])
    screen.handle_events([pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=a8)])
    assert screen._board.promotion_pending is True
    assert screen._board.premove is None  # did NOT auto-queue queen premove
    b_pos = _promo_screen_center(screen, "b")
    screen.draw()
    screen.handle_events([pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=b_pos)])
    assert screen._board.promotion_pending is False
    assert screen._board.premove == (chess.A7, chess.A8, "b")

    # Bot plays Kh8 -> premove fires and promotes to Bishop (NOT Queen)
    worker.results.append(chess.Move.from_uci("h7h8"))
    screen._update_bot(0.016)
    screen._bot_wait_remaining = 0.0
    _advance_bot_pacing(screen)
    _finish_animation(screen)
    assert screen._game.piece_at(chess.A8) == chess.Piece(chess.BISHOP, chess.WHITE)
    screen.close()

    # 3. Premove drag promotion opens picker; if bot replies while picker is open,
    #    picker converts cleanly to live mode and user's Knight choice executes
    screen, worker = _game_screen("White")
    screen._game = ChessGame("8/P6k/8/8/8/8/8/K7 b - - 0 1")
    screen._board.attach_game(screen._game, flipped=False)
    a7 = _sq_screen_center(screen, chess.A7)
    a8 = _sq_screen_center(screen, chess.A8)
    screen.handle_events([pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=a7)])
    screen.handle_events([pygame.event.Event(pygame.MOUSEMOTION, pos=a8)])
    screen.handle_events([pygame.event.Event(pygame.MOUSEBUTTONUP, button=1, pos=a8)])
    assert screen._board.promotion_pending is True
    assert screen._board._promotion_is_premove is True

    # Bot plays Kh8 while promotion picker is still open
    worker.results.append(chess.Move.from_uci("h7h8"))
    screen._update_bot(0.016)
    screen._bot_wait_remaining = 0.0
    _advance_bot_pacing(screen)
    _finish_animation(screen)
    assert screen._board.promotion_pending is True
    assert screen._board._promotion_is_premove is False

    n_pos = _promo_screen_center(screen, "n")
    screen.handle_events([pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=n_pos)])
    assert screen._board.promotion_pending is False
    assert screen._game.piece_at(chess.A8) == chess.Piece(chess.KNIGHT, chess.WHITE)
    screen.close()


if __name__ == "__main__":
    test_orientation_round_trip()
    test_orientation_corners_and_off_board()
    test_selection_flow()
    test_capture_markers()
    test_promotion_flow_white_and_flipped_black()
    test_user_chooses_promotion_piece_live_and_premove()
    test_animation_hides_destination_until_landed()
    test_animation_mathematics()
    test_animation_queue_captures_resize_and_flip()
    test_player_black_bot_moves_first_and_board_flipped()
    test_stale_bot_result_discarded_after_restart()
    test_illegal_bot_move_falls_back()
    test_bot_pacing_waits_for_player_animation_and_respects_engine_time()
    test_endgame_popup_waits_for_final_animation()
    test_game_over_dialog_result_content()
    test_game_over_dialog_responsive_layout_and_interaction()
    test_game_over_dialog_entrance_animation()
    test_opponent_select_screen_redesign()
    test_perf_overlay_toggle_and_stats()
    test_resign_draw_offer_and_close()
    test_escape_captured_only_during_promotion()
    test_board_backed_by_chess_game_and_bars_layout()
    test_opening_banner_and_figurine_formatting()
    test_move_list_scroll_and_clamping()
    test_sidebar_resign_becomes_new_game_when_over()
    test_endgame_popup_and_rematch_flow()
    test_time_control_presets_and_exclusions()
    test_time_control_dropdown()
    test_time_control_modern_spacing_animation_and_responsiveness()
    test_speech_bubble_entrance_dynamic_text_and_reduced_motion()
    test_premove_settings_toggle_and_gameplay()
    test_hamburger_toggle_does_not_overlap_profile_bubble()
    test_hamburger_menu_actions_after_game_over()
    test_move_review_arrow_keys_click_and_nav_bar()
    test_match_clock_countdown_increment_and_flag_fall()
    print("ALL INTERACTIVE GAME TESTS PASSED!")
