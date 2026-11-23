"""Test and verification for the reference-style AnimatedBoard.

Tests asset loading, caching, animation steps, and captures a full
dashboard screenshot with the animated background.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pygame

from others.config import ASSETS_DIR
from screens.dashboard import DashboardScreen
from ui.animated_board import AnimatedBoard, _PIECE_SVG_MAP, _load_svg_at_size

PREVIEW_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "dashboard_neo_preview.png"))


def test_svg_piece_renders_at_requested_size():
    pygame.init()
    piece = _load_svg_at_size(ASSETS_DIR / "wK.svg", 90)

    assert piece.get_size() == (90, 90)


def test_board_assets():
    pygame.init()
    surf = pygame.Surface((1920, 1080))
    board = AnimatedBoard(1920, 1080, 1.5)

    # Verify default theme is Lichess Green matching the reference design
    assert board.theme == "lichess_green", f"Expected default theme 'lichess_green', got {board.theme}"
    assert board._theme["light_sq"] == (255, 255, 221), "Light square mismatch"
    assert board._theme["dark_sq"] == (134, 166, 102), "Dark square mismatch"
    assert board._theme["border"] == (33, 30, 26), "Border mismatch"

    # Verify theme switching
    board.set_theme("wood")
    assert board.theme == "wood"
    board.set_theme("lichess_green")
    assert board.theme == "lichess_green"

    print("Checking AnimatedBoard piece cache:")
    for piece_char, filename in _PIECE_SVG_MAP.items():
        assert piece_char in board._piece_cache, f"Piece {piece_char} missing from cache"
        piece_surf = board._piece_cache[piece_char]
        print(f"  [OK] {piece_char} ({filename}): cached surface size = {piece_surf.get_size()}")

    black_piece = board._piece_cache["p"]
    assert all(
        black_piece.get_at((x, y))[3] == 0
        or black_piece.get_at((x, y))[:3] == (0, 0, 0)
        for x in range(black_piece.get_width())
        for y in range(black_piece.get_height())
    ), "Black pieces should render as solid black silhouettes"

    white_piece = board._piece_cache["P"]
    white_piece_colors = {
        white_piece.get_at((x, y))[:3]
        for x in range(white_piece.get_width())
        for y in range(white_piece.get_height())
        if white_piece.get_at((x, y))[3] > 0
    }
    assert any(min(color) >= 240 for color in white_piece_colors), "White pieces should have white fills"
    assert any(max(color) <= 10 for color in white_piece_colors), "White pieces should retain dark outlines"

    # Simulate 5 seconds of animation (running through several moves)
    dt = 0.05
    for step in range(100):
        board.update(dt)
        board.draw(surf)

    print("Simulated 5s of board animation successfully!")


def test_dashboard_full_render():
    pygame.init()
    surf = pygame.Surface((1920, 1080))
    dash = DashboardScreen(surf)

    # Advance 1.5 seconds so a move has started/landed and fade-in is complete
    dt = 0.05
    for _ in range(30):
        dash.update(dt)

    dash.draw()
    pygame.image.save(surf, PREVIEW_PATH)
    print(f"Saved full dashboard preview to: {PREVIEW_PATH}")


if __name__ == "__main__":
    test_board_assets()
    test_dashboard_full_render()
    print("ALL TESTS PASSED")
