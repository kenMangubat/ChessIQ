"""Tests for Chess Ken logo processing, icon files, and dashboard logo integration."""

from __future__ import annotations

import os
import struct
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pygame

from others.config import (
    ICON_ICO_PATH,
    ICON_MASTER_PATH,
    ICON_PREVIEW_PATH,
    LOGO_SOURCE_PATH,
)
from others.make_icon import ICO_SIZES
from screens.dashboard import DashboardScreen
from screens.opponent_select import OpponentSelectScreen


def test_icon_files_exist_via_config() -> None:
    """All logo/icon assets resolved through others/config.py must exist."""
    assert LOGO_SOURCE_PATH.is_file(), f"Missing source logo: {LOGO_SOURCE_PATH}"
    assert ICON_MASTER_PATH.is_file(), f"Missing master icon: {ICON_MASTER_PATH}"
    assert ICON_ICO_PATH.is_file(), f"Missing .ico file: {ICON_ICO_PATH}"
    assert ICON_PREVIEW_PATH.is_file(), f"Missing preview sheet: {ICON_PREVIEW_PATH}"


def test_master_icon_is_square_and_transparent() -> None:
    """Processed master PNG must be square, high-res, and carry an alpha channel."""
    pygame.init()
    surf = pygame.image.load(str(ICON_MASTER_PATH))
    w, h = surf.get_size()
    assert w == h, f"Master icon is not square: {w}x{h}"
    assert w >= 256, f"Master icon is unexpectedly small: {w}x{h}"
    assert surf.get_bitsize() == 32, "Master icon should be 32-bit RGBA"
    # Corners should be transparent after exterior background removal
    assert surf.get_at((0, 0)).a == 0, "Top-left corner of master should be transparent"


def test_ico_contains_expected_sizes() -> None:
    """Windows .ico header must list all required icon resolutions."""
    data = ICON_ICO_PATH.read_bytes()
    reserved, ico_type, count = struct.unpack_from("<HHH", data, 0)
    assert reserved == 0
    assert ico_type == 1
    assert count == len(ICO_SIZES)

    found_sizes: list[int] = []
    for i in range(count):
        width_b, height_b, _, _, _, _, size_bytes, offset = struct.unpack_from(
            "<BBBBHHII", data, 6 + i * 16
        )
        w = 256 if width_b == 0 else width_b
        h = 256 if height_b == 0 else height_b
        assert w == h
        assert size_bytes > 0
        assert offset + size_bytes <= len(data)
        found_sizes.append(w)

    assert sorted(found_sizes) == sorted(ICO_SIZES)


def test_dashboard_logo_scaling_and_caching() -> None:
    """DashboardScreen should preserve aspect ratio and cache scaled logo surfaces."""
    pygame.init()
    DashboardScreen._logo_raw_cache.clear()
    DashboardScreen._logo_scaled_cache.clear()

    surf_720 = pygame.Surface((1280, 720))
    dash_720 = DashboardScreen(surf_720)
    w720, h720 = dash_720._icon_surf.get_size()
    assert w720 == h720, f"Dashboard logo distorted at 720p: {w720}x{h720}"
    assert len(DashboardScreen._logo_raw_cache) == 1
    assert len(DashboardScreen._logo_scaled_cache) == 1

    # Re-instantiating at the same size reuses cached scaled surface
    dash_720_b = DashboardScreen(surf_720)
    assert dash_720_b._icon_surf is dash_720._icon_surf

    # Different window size builds a new cached scaled surface without reloading raw
    surf_1080 = pygame.Surface((1920, 1080))
    dash_1080 = DashboardScreen(surf_1080)
    w1080, h1080 = dash_1080._icon_surf.get_size()
    assert w1080 == h1080, f"Dashboard logo distorted at 1080p: {w1080}x{h1080}"
    assert w1080 > w720
    assert len(DashboardScreen._logo_raw_cache) == 1
    assert len(DashboardScreen._logo_scaled_cache) == 2

    dash_720.draw()
    dash_1080.draw()


def test_opponent_select_uses_3d_logo_consistently() -> None:
    """OpponentSelectScreen header crest should use the square 3D logo across layouts."""
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    pygame.init()
    if pygame.display.get_surface() is None:
        pygame.display.set_mode((1, 1))
    for dims in ((1280, 720), (800, 720), (480, 820)):
        surf = pygame.Surface(dims)
        scr = OpponentSelectScreen(surf)
        ew, eh = scr._knight_emblem.get_size()
        assert ew == eh, f"Header logo distorted at {dims}: {ew}x{eh}"
        assert ew >= 32
        # Verify logo does not overlap the gradient headline
        logo_bottom = scr._header_knight_y + eh // 2
        title_top = scr._header_title_y - scr._header_title_surf.get_height() // 2
        assert logo_bottom <= title_top + 4, (
            f"Header logo overlaps title at {dims}: bottom={logo_bottom}, title_top={title_top}"
        )
        scr.draw()


if __name__ == "__main__":
    test_icon_files_exist_via_config()
    test_master_icon_is_square_and_transparent()
    test_ico_contains_expected_sizes()
    test_dashboard_logo_scaling_and_caching()
    test_opponent_select_uses_3d_logo_consistently()
    print("ALL ICON TESTS PASSED!")
