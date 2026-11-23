"""Dashboard — the main menu screen for Chess Ken.

A centered menu card with a large title, a readable tagline, and three
buttons: PLAY, SETTINGS, EXIT.  Sizing is relative to the display so the
type stays large on a full-screen window.
"""

import math
from pathlib import Path
import numpy as np
import pygame

from others.config import ICON_MASTER_PATH, LOGO_SOURCE_PATH
from screens.base_screen import BaseScreen
from ui.button import Button
from ui.animated_board import AnimatedBoard


# ── colour palette (deep charcoal + emerald & cyan accents) ─────────
_BG_TOP = (9, 13, 17)
_BG_BOT = (16, 22, 29)
_TITLE_GRAD_LEFT = (0, 214, 160)       # #00D6A0 emerald
_TITLE_GRAD_RIGHT = (54, 191, 255)     # #36BFFF cyan
_TITLE_COLOR = _TITLE_GRAD_LEFT
_SUBTITLE_COLOR = (176, 190, 206)
_CARD = (15, 22, 28)
_ACCENT = (0, 214, 160)
_ACCENT_CYAN = (54, 191, 255)

# button colours (harmonized emerald-teal gradients)
_BTN_PLAY_GRADIENT = ((16, 198, 138), (5, 138, 108))
_BTN_PLAY_HOVER_GRADIENT = ((34, 216, 158), (12, 164, 130))
_BTN_SETTINGS_GRADIENT = ((21, 56, 60), (14, 38, 44))
_BTN_SETTINGS_HOVER_GRADIENT = ((26, 76, 78), (16, 54, 60))
_BTN_EXIT_GRADIENT = ((18, 34, 40), (12, 22, 28))
_BTN_EXIT_HOVER_GRADIENT = ((24, 48, 56), (16, 34, 42))
_BTN_SHADOW = (0, 180, 145)
_BTN_TEXT = (244, 248, 252)
_BUTTON_FONT_FAMILIES = (
    "Plus Jakarta Sans",
    "Manrope",
    "BlinkMacSystemFont",
    "-apple-system",
    "Segoe UI",
    "Roboto",
    "Oxygen",
    "Ubuntu",
    "Cantarell",
    "Fira Sans",
    "Droid Sans",
    "Helvetica Neue",
    "Helvetica",
    "Arial",
)


class DashboardScreen(BaseScreen):
    """Main dashboard with PLAY / SETTINGS / EXIT."""

    _logo_raw_cache: dict[Path, pygame.Surface] = {}
    _logo_scaled_cache: dict[tuple[Path, int, int], pygame.Surface] = {}

    def __init__(self, screen: pygame.Surface) -> None:
        super().__init__(screen)

        # Design baseline is 1280×720, so a 1080p display scales type up
        # instead of leaving a small cluster in a large empty window.
        scale = min(self.width / 1280, self.height / 720)
        self._scale = scale

        self._font_title = self._load_modern_font(max(46, int(66 * scale)), "bold")
        self._font_sub = self._load_modern_font(max(18, int(23 * scale)), "regular")
        self._font_btn = self._make_button_font(max(17, int(17.5 * scale)))
        self._font_icon = pygame.font.SysFont("Segoe UI Symbol", max(36, int(52 * scale)))
        self._font_ver = self._load_modern_font(max(14, int(16 * scale)), "semibold")

        self._title_surf = self._render_gradient_title("Chess Ken")
        self._sub_surf = self._font_sub.render("Can you beat me?", True, _SUBTITLE_COLOR)
        self._logo_target_size = max(60, int(92 * scale))
        self._icon_surf = self._load_logo_surface(self._logo_target_size)
        self._icon_shadow = self._make_logo_shadow(self._icon_surf.get_size())
        self._ver_surf = self._font_ver.render("v0.1", True, (130, 146, 160))

        self._bg = self._make_gradient(self.width, self.height, _BG_TOP, _BG_BOT)
        self._grid = self._make_grid_overlay(self.width, self.height, scale)
        self._anim_board = AnimatedBoard(self.width, self.height, scale)

        cx = self.width // 2
        btn_w = int(460 * scale)
        btn_h = int(66 * scale)
        btn_h_play = int(76 * scale)
        gap = int(15 * scale)
        radius = int(14 * scale)

        icon_h = self._icon_surf.get_height()
        title_h = self._title_surf.get_height()
        sub_h = self._sub_surf.get_height()
        block_gap = int(10 * scale)
        rule_gap = int(20 * scale)
        after_sub = int(28 * scale)

        content_h = (
            icon_h + block_gap
            + title_h + rule_gap
            + sub_h + after_sub
            + btn_h_play + gap
            + btn_h + gap
            + btn_h
        )
        pad_x = int(56 * scale)
        pad_y = int(38 * scale)
        card_w = btn_w + pad_x * 2
        card_h = content_h + pad_y * 2
        self._card = pygame.Rect(0, 0, card_w, card_h)
        self._card.center = (cx, self.height // 2)
        strip_h = max(28, int(36 * scale))
        bottom_limit = self.height - strip_h - int(28 * scale)
        if self._card.bottom > bottom_limit:
            self._card.bottom = bottom_limit
        top_limit = int(28 * scale)
        if self._card.top < top_limit:
            self._card.top = top_limit
        self._card_radius = int(24 * scale)

        y = self._card.top + pad_y
        self._icon_pos = self._icon_surf.get_rect(midtop=(cx, y))
        y = self._icon_pos.bottom + block_gap
        self._title_pos = self._title_surf.get_rect(midtop=(cx, y))

        rule_y = self._title_pos.bottom + rule_gap // 2
        rule_w = int(132 * scale)
        self._rule = ((cx - rule_w // 2, rule_y), (cx + rule_w // 2, rule_y))
        self._rule_surf = self._make_gradient_rule(rule_w, max(2, int(3 * scale)))

        self._sub_pos = self._sub_surf.get_rect(midtop=(cx, self._title_pos.bottom + rule_gap))
        group_top = self._sub_pos.bottom + after_sub
        self._drag_header = pygame.Rect(
            self._card.left,
            self._card.top,
            self._card.width,
            group_top - self._card.top,
        )
        play_rect = pygame.Rect(0, 0, btn_w, btn_h_play)
        play_rect.midtop = (cx, group_top)

        settings_rect = pygame.Rect(0, 0, btn_w, btn_h)
        settings_rect.midtop = (cx, play_rect.bottom + gap)

        exit_rect = pygame.Rect(0, 0, btn_w, btn_h)
        exit_rect.midtop = (cx, settings_rect.bottom + gap)

        self._btn_play = Button(
            play_rect, "PLAY", self._font_btn,
            color=_BTN_PLAY_GRADIENT[0], hover_color=_BTN_PLAY_HOVER_GRADIENT[0],
            text_color=_BTN_TEXT, border_radius=radius,
            gradient_colors=_BTN_PLAY_GRADIENT,
            hover_gradient_colors=_BTN_PLAY_HOVER_GRADIENT,
            shadow_color=_BTN_SHADOW, border_color=(82, 236, 206),
            lift_on_hover=True, press_feedback=True,
        )
        self._btn_settings = Button(
            settings_rect, "SETTINGS", self._font_btn,
            color=_BTN_SETTINGS_GRADIENT[0], hover_color=_BTN_SETTINGS_HOVER_GRADIENT[0],
            text_color=_BTN_TEXT, border_radius=radius,
            gradient_colors=_BTN_SETTINGS_GRADIENT,
            hover_gradient_colors=_BTN_SETTINGS_HOVER_GRADIENT,
            shadow_color=_BTN_SHADOW, border_color=(42, 134, 128),
            lift_on_hover=True, press_feedback=True,
        )
        self._btn_exit = Button(
            exit_rect, "EXIT", self._font_btn,
            color=_BTN_EXIT_GRADIENT[0], hover_color=_BTN_EXIT_HOVER_GRADIENT[0],
            text_color=_BTN_TEXT, border_radius=radius,
            gradient_colors=_BTN_EXIT_GRADIENT,
            hover_gradient_colors=_BTN_EXIT_HOVER_GRADIENT,
            shadow_color=_BTN_SHADOW, border_color=(46, 78, 88),
            lift_on_hover=True, press_feedback=True,
        )
        self._buttons = [self._btn_play, self._btn_settings, self._btn_exit]

        self._glow = self._make_glow(int(280 * scale))
        self._card_padding = max(6, int(8 * scale))
        self._card_surf = self._make_card(card_w, card_h, self._card_radius, self._card_padding)
        self._drag_offset: tuple[int, int] | None = None
        self._header_hovered = False

        self._time: float = 0.0
        self._fade_in: float = 0.0
        self._FADE_SPEED: float = 2.5

    # ------------------------------------------------------------------
    # BaseScreen interface
    # ------------------------------------------------------------------

    def handle_events(self, events: list[pygame.event.Event]) -> None:
        for event in events:
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if self._drag_header.collidepoint(event.pos):
                    self._drag_offset = (
                        event.pos[0] - self._card.left,
                        event.pos[1] - self._card.top,
                    )
                    self._header_hovered = True
                    continue
            elif event.type == pygame.MOUSEMOTION:
                if self._drag_offset is not None:
                    self._move_panel(
                        event.pos[0] - self._drag_offset[0],
                        event.pos[1] - self._drag_offset[1],
                    )
                    continue
                self._header_hovered = self._drag_header.collidepoint(event.pos)
            elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                if self._drag_offset is not None:
                    self._drag_offset = None
                    self._header_hovered = self._drag_header.collidepoint(event.pos)
                    continue

            if self._btn_play.handle_event(event):
                self.next_screen = "opponent_select"
            elif self._btn_settings.handle_event(event):
                self.next_screen = "settings"
            elif self._btn_exit.handle_event(event):
                self.next_screen = "exit"

    def _move_panel(self, left: int, top: int) -> None:
        """Move the panel and its contents while keeping it inside the window."""
        min_left = min(self._card_padding, max(0, self.width - self._card.width))
        min_top = min(self._card_padding, max(0, self.height - self._card.height))
        max_left = max(min_left, self.width - self._card.width - self._card_padding)
        max_top = max(min_top, self.height - self._card.height - self._card_padding)
        left = max(min_left, min(left, max_left))
        top = max(min_top, min(top, max_top))
        dx = left - self._card.left
        dy = top - self._card.top
        if not dx and not dy:
            return

        self._card.move_ip(dx, dy)
        self._drag_header.move_ip(dx, dy)
        self._icon_pos.move_ip(dx, dy)
        self._title_pos.move_ip(dx, dy)
        self._sub_pos.move_ip(dx, dy)
        self._rule = (
            (self._rule[0][0] + dx, self._rule[0][1] + dy),
            (self._rule[1][0] + dx, self._rule[1][1] + dy),
        )
        for button in self._buttons:
            button.base_rect.move_ip(dx, dy)

    def update(self, dt: float) -> None:
        self._time += dt
        self._fade_in = min(self._fade_in + self._FADE_SPEED * dt, 1.0)
        self._anim_board.update(dt)
        for btn in self._buttons:
            btn.update(dt)

    def draw(self) -> None:
        # 1. Dark charcoal background
        self.screen.blit(self._bg, (0, 0))
        self.screen.blit(self._grid, (0, 0))

        # 2 & 3. Animated chessboard & moving chess pieces (Lichess green theme)
        self._anim_board.draw(self.screen)

        # 4. Subtle emerald lighting
        pulse = 0.72 + 0.28 * math.sin(self._time * 1.4)
        self._glow.set_alpha(int(255 * pulse))
        self.screen.blit(
            self._glow,
            (self._card.centerx - self._glow.get_width() // 2,
             self._card.top - int(40 * self._scale)),
        )

        # 5. Semi-transparent main menu panel (shows animated board through it)
        self.screen.blit(
            self._card_surf,
            (self._card.left - self._card_padding, self._card.top - self._card_padding),
        )
        if self._header_hovered or self._drag_offset is not None:
            highlight = pygame.Surface(
                (self._drag_header.width, self._drag_header.height), pygame.SRCALPHA
            )
            pygame.draw.rect(
                highlight,
                (16, 185, 129, 18),
                highlight.get_rect(),
                border_radius=self._card_radius,
            )
            self.screen.blit(highlight, self._drag_header.topleft)
            grip_color = (87, 211, 165)
            grip_y = self._drag_header.top + int(14 * self._scale)
            grip_width = int(38 * self._scale)
            pygame.draw.line(
                self.screen,
                grip_color,
                (self._drag_header.centerx - grip_width // 2, grip_y),
                (self._drag_header.centerx + grip_width // 2, grip_y),
                width=max(2, int(3 * self._scale)),
            )

        # 6. Menu typography & buttons
        shadow_rect = self._icon_shadow.get_rect(
            center=(self._icon_pos.centerx, self._icon_pos.centery + max(2, int(4 * self._scale)))
        )
        self.screen.blit(self._icon_shadow, shadow_rect)
        self.screen.blit(self._icon_surf, self._icon_pos)
        self.screen.blit(self._title_surf, self._title_pos)

        rule_rect = self._rule_surf.get_rect(
            midleft=(self._rule[0][0], self._rule[0][1])
        )
        self.screen.blit(self._rule_surf, rule_rect)
        self.screen.blit(self._sub_surf, self._sub_pos)

        for btn in self._buttons:
            btn.draw(self.screen)

        margin = int(28 * self._scale)
        self.screen.blit(
            self._ver_surf,
            (self.width - self._ver_surf.get_width() - margin,
             self.height - self._ver_surf.get_height() - margin),
        )

        if self._fade_in < 1.0:
            overlay = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, int(255 * (1.0 - self._fade_in))))
            self.screen.blit(overlay, (0, 0))

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_logo_surface(self, target_size: int) -> pygame.Surface:
        """Load the logo once and cache its aspect-preserved scaled surface."""
        logo_path = ICON_MASTER_PATH if ICON_MASTER_PATH.is_file() else LOGO_SOURCE_PATH
        if not logo_path.is_file():
            return self._font_icon.render("♔", True, _TITLE_COLOR)

        raw = self._logo_raw_cache.get(logo_path)
        if raw is None:
            loaded = pygame.image.load(str(logo_path))
            if pygame.display.get_surface() is not None:
                loaded = loaded.convert_alpha()
            raw = loaded
            self._logo_raw_cache[logo_path] = raw

        src_w, src_h = raw.get_size()
        ratio = min(target_size / max(src_w, 1), target_size / max(src_h, 1))
        out_w = max(1, int(round(src_w * ratio)))
        out_h = max(1, int(round(src_h * ratio)))

        cache_key = (logo_path, out_w, out_h)
        scaled = self._logo_scaled_cache.get(cache_key)
        if scaled is None:
            scaled = pygame.transform.smoothscale(raw, (out_w, out_h))
            self._logo_scaled_cache[cache_key] = scaled
        return scaled

    @staticmethod
    def _load_modern_font(size: int, weight: str = "regular") -> pygame.font.Font:
        """Load Plus Jakarta Sans / Manrope / Segoe UI using genuine weight files."""
        size = max(10, size)
        if weight == "bold":
            for family in ("Plus Jakarta Sans", "Manrope", "Segoe UI"):
                path = pygame.font.match_font(family, bold=True)
                if path:
                    return pygame.font.Font(path, size)
            return pygame.font.SysFont("Segoe UI", size, bold=True)

        if weight == "semibold":
            for family in ("Plus Jakarta Sans SemiBold", "Manrope SemiBold", "Segoe UI Semibold", "Segoe UI"):
                path = pygame.font.match_font(family)
                if path:
                    return pygame.font.Font(path, size)
            return pygame.font.SysFont("Segoe UI", size, bold=True)

        for family in ("Plus Jakarta Sans", "Manrope", "Segoe UI"):
            path = pygame.font.match_font(family)
            if path:
                if path.lower().endswith("segoeuil.ttf"):
                    alt = Path(path).with_name("segoeui.ttf")
                    if alt.is_file():
                        return pygame.font.Font(str(alt), size)
                return pygame.font.Font(path, size)
        return pygame.font.SysFont("Segoe UI", size)

    def _render_gradient_title(self, text: str) -> pygame.Surface:
        """Render the menu title with a clean emerald-to-cyan gradient."""
        raw = self._font_title.render(text, True, (255, 255, 255))
        w, h = raw.get_size()
        grad = pygame.Surface((w, h), pygame.SRCALPHA)
        t = np.linspace(0.0, 1.0, max(1, w), dtype=np.float32)
        c1 = np.array(_TITLE_GRAD_LEFT, dtype=np.float32)
        c2 = np.array(_TITLE_GRAD_RIGHT, dtype=np.float32)
        colors = (c1[None, :] * (1.0 - t[:, None]) + c2[None, :] * t[:, None]).astype(np.uint8)
        rgb = pygame.surfarray.pixels3d(grad)
        rgb[:, :, 0] = colors[:, 0:1]
        rgb[:, :, 1] = colors[:, 1:2]
        rgb[:, :, 2] = colors[:, 2:3]
        del rgb
        alpha = pygame.surfarray.pixels_alpha(grad)
        alpha[:, :] = pygame.surfarray.array_alpha(raw)
        del alpha
        return grad

    @staticmethod
    def _make_gradient_rule(w: int, h: int) -> pygame.Surface:
        """Create a subtle tapered emerald-to-cyan horizontal accent rule."""
        w = max(8, w)
        h = max(2, h)
        surf = pygame.Surface((w, h), pygame.SRCALPHA)
        for x in range(w):
            t = x / max(w - 1, 1)
            r = int(_ACCENT[0] + (_ACCENT_CYAN[0] - _ACCENT[0]) * t)
            g = int(_ACCENT[1] + (_ACCENT_CYAN[1] - _ACCENT[1]) * t)
            b = int(_ACCENT[2] + (_ACCENT_CYAN[2] - _ACCENT[2]) * t)
            edge_fade = min(1.0, min(t, 1.0 - t) * 5.0)
            a = int(215 * edge_fade)
            pygame.draw.line(surf, (r, g, b, a), (x, 0), (x, h - 1))
        return surf

    @staticmethod
    def _make_logo_shadow(icon_size: tuple[int, int]) -> pygame.Surface:
        """Create a soft, grounded drop shadow under the 3D knight app icon."""
        w, h = icon_size
        pad = max(8, int(min(w, h) * 0.14))
        surf = pygame.Surface((w + pad * 2, h + pad * 2), pygame.SRCALPHA)
        radius = max(8, int(min(w, h) * 0.24))
        for step in range(4, 0, -1):
            expand = step * max(1, pad // 4)
            rect = pygame.Rect(
                pad - expand // 2,
                pad - expand // 2,
                w + expand,
                h + expand,
            )
            alpha = int(28 * (1.0 - (step - 1) / 4.0))
            pygame.draw.rect(surf, (0, 0, 0, alpha), rect, border_radius=radius + expand // 3)
        return surf

    @staticmethod
    def _make_button_font(size: int) -> pygame.font.Font:
        """Select the first installed family from the requested UI font stack."""
        for family in _BUTTON_FONT_FAMILIES:
            font_path = pygame.font.match_font(family, bold=True)
            if font_path:
                return pygame.font.Font(font_path, size)
        return pygame.font.SysFont(None, size, bold=True)

    @staticmethod
    def _make_gradient(
        w: int, h: int,
        top: tuple[int, int, int],
        bot: tuple[int, int, int],
    ) -> pygame.Surface:
        """Return a vertical linear-gradient surface."""
        surf = pygame.Surface((w, h))
        for y in range(h):
            t = y / max(h - 1, 1)
            r = int(top[0] + (bot[0] - top[0]) * t)
            g = int(top[1] + (bot[1] - top[1]) * t)
            b = int(top[2] + (bot[2] - top[2]) * t)
            pygame.draw.line(surf, (r, g, b), (0, y), (w, y))
        return surf

    @staticmethod
    def _make_grid_overlay(w: int, h: int, scale: float) -> pygame.Surface:
        """Faint chess-inspired grid, stronger toward the middle."""
        surf = pygame.Surface((w, h), pygame.SRCALPHA)
        step = max(48, int(72 * scale))
        color = (255, 255, 255, 12)
        for x in range(0, w, step):
            pygame.draw.line(surf, color, (x, 0), (x, h))
        for y in range(0, h, step):
            pygame.draw.line(surf, color, (0, y), (w, y))
        return surf

    @staticmethod
    def _make_glow(radius: int) -> pygame.Surface:
        """Subtle emerald-cyan ambient glow cached once and pulsed with set_alpha."""
        surf = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
        for i in range(radius, 0, -4):
            ratio = i / radius
            alpha = int(14 * ratio)
            r = int(_ACCENT[0] * (1.0 - ratio) + _ACCENT_CYAN[0] * ratio)
            g = int(_ACCENT[1] * (1.0 - ratio) + _ACCENT_CYAN[1] * ratio)
            b = int(_ACCENT[2] * (1.0 - ratio) + _ACCENT_CYAN[2] * ratio)
            pygame.draw.circle(surf, (r, g, b, alpha), (radius, radius), i)
        return surf

    @staticmethod
    def _make_card(w: int, h: int, radius: int, pad: int = 8) -> pygame.Surface:
        """Rounded frosted deep-charcoal panel with refined emerald-cyan top highlight."""
        surf = pygame.Surface((w + pad * 2, h + pad * 2), pygame.SRCALPHA)
        inner = pygame.Rect(pad, pad, w, h)

        # Subtle ambient shadow halo around the card
        glow_rect = inner.inflate(pad, pad)
        pygame.draw.rect(surf, (0, 0, 0, 56), glow_rect, border_radius=radius + 4)

        # Deep charcoal frosted surface keeps logo & typography crisp over the board
        pygame.draw.rect(surf, (*_CARD, 236), inner, border_radius=radius)
        pygame.draw.rect(
            surf,
            (68, 92, 108, 210),
            inner,
            width=1,
            border_radius=radius,
        )
        # Subtle emerald-to-cyan top hairline accent
        line_x0 = inner.left + radius
        line_x1 = inner.right - radius
        span = max(1, line_x1 - line_x0)
        for x in range(line_x0, line_x1 + 1):
            t = (x - line_x0) / span
            r = int(_ACCENT[0] + (_ACCENT_CYAN[0] - _ACCENT[0]) * t)
            g = int(_ACCENT[1] + (_ACCENT_CYAN[1] - _ACCENT[1]) * t)
            b = int(_ACCENT[2] + (_ACCENT_CYAN[2] - _ACCENT[2]) * t)
            edge_fade = min(1.0, min(t, 1.0 - t) * 4.0)
            surf.set_at((x, inner.top + 1), (r, g, b, int(145 * edge_fade)))
        return surf
