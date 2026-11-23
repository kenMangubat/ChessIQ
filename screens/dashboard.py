"""Dashboard — the main menu screen for Chess Ken.

A centered menu card with a large title, a readable tagline, and three
buttons: PLAY, SETTINGS, EXIT.  Sizing is relative to the display so the
type stays large on a full-screen window.
"""

import math
import pygame

from screens.base_screen import BaseScreen
from ui.button import Button
from ui.animated_board import AnimatedBoard


# ── colour palette ──────────────────────────────────────────────────
_BG_TOP = (12, 14, 18)
_BG_BOT = (20, 24, 32)
_TITLE_COLOR = (52, 211, 153)
_SUBTITLE_COLOR = (196, 202, 214)
_CARD = (24, 28, 36)
_ACCENT = (16, 185, 129)

# button colours
_BTN_PLAY_GRADIENT = ((34, 197, 94), (21, 128, 61))
_BTN_PLAY_HOVER_GRADIENT = ((22, 163, 74), (22, 101, 52))
_BTN_SETTINGS_GRADIENT = ((22, 101, 52), (20, 83, 45))
_BTN_SETTINGS_HOVER_GRADIENT = ((21, 128, 61), (6, 78, 59))
_BTN_EXIT_GRADIENT = ((20, 83, 45), (5, 46, 22))
_BTN_EXIT_HOVER_GRADIENT = ((22, 101, 52), (20, 83, 45))
_BTN_SHADOW = (34, 197, 94)
_BTN_TEXT = (255, 255, 255)
_BUTTON_FONT_FAMILIES = (
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

    def __init__(self, screen: pygame.Surface) -> None:
        super().__init__(screen)

        # Design baseline is 1280×720, so a 1080p display scales type up
        # instead of leaving a small cluster in a large empty window.
        scale = min(self.width / 1280, self.height / 720)
        self._scale = scale

        self._font_title = pygame.font.SysFont("Segoe UI", max(48, int(68 * scale)), bold=True)
        self._font_sub = pygame.font.SysFont("Segoe UI", max(20, int(26 * scale)))
        self._font_btn = self._make_button_font(max(17, int(16.8 * scale)))
        self._font_icon = pygame.font.SysFont("Segoe UI Symbol", max(36, int(52 * scale)))
        self._font_ver = pygame.font.SysFont("Segoe UI", max(16, int(18 * scale)))

        self._title_surf = self._font_title.render("Chess Ken", True, _TITLE_COLOR)
        self._sub_surf = self._font_sub.render("Can you beat me?", True, _SUBTITLE_COLOR)
        self._icon_surf = self._font_icon.render("♔", True, _TITLE_COLOR)
        self._ver_surf = self._font_ver.render("v0.1", True, (130, 138, 150))

        self._bg = self._make_gradient(self.width, self.height, _BG_TOP, _BG_BOT)
        self._grid = self._make_grid_overlay(self.width, self.height, scale)
        self._anim_board = AnimatedBoard(self.width, self.height, scale)

        cx = self.width // 2
        btn_w = int(460 * scale)
        btn_h = int(68 * scale)
        btn_h_play = int(78 * scale)
        gap = int(16 * scale)
        radius = int(14 * scale)

        icon_h = self._icon_surf.get_height()
        title_h = self._title_surf.get_height()
        sub_h = self._sub_surf.get_height()
        block_gap = int(6 * scale)
        rule_gap = int(20 * scale)
        after_sub = int(30 * scale)

        content_h = (
            icon_h + block_gap
            + title_h + rule_gap
            + sub_h + after_sub
            + btn_h_play + gap
            + btn_h + gap
            + btn_h
        )
        pad_x = int(56 * scale)
        pad_y = int(40 * scale)
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
        rule_w = int(120 * scale)
        self._rule = ((cx - rule_w // 2, rule_y), (cx + rule_w // 2, rule_y))

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
            shadow_color=_BTN_SHADOW, border_color=(74, 222, 128),
            lift_on_hover=True, press_feedback=True,
        )
        self._btn_settings = Button(
            settings_rect, "SETTINGS", self._font_btn,
            color=_BTN_SETTINGS_GRADIENT[0], hover_color=_BTN_SETTINGS_HOVER_GRADIENT[0],
            text_color=_BTN_TEXT, border_radius=radius,
            gradient_colors=_BTN_SETTINGS_GRADIENT,
            hover_gradient_colors=_BTN_SETTINGS_HOVER_GRADIENT,
            shadow_color=_BTN_SHADOW, border_color=(34, 197, 94),
            lift_on_hover=True, press_feedback=True,
        )
        self._btn_exit = Button(
            exit_rect, "EXIT", self._font_btn,
            color=_BTN_EXIT_GRADIENT[0], hover_color=_BTN_EXIT_HOVER_GRADIENT[0],
            text_color=_BTN_TEXT, border_radius=radius,
            gradient_colors=_BTN_EXIT_GRADIENT,
            hover_gradient_colors=_BTN_EXIT_HOVER_GRADIENT,
            shadow_color=_BTN_SHADOW, border_color=(22, 101, 52),
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
        self.screen.blit(self._icon_surf, self._icon_pos)
        self.screen.blit(self._title_surf, self._title_pos)

        pygame.draw.line(self.screen, _ACCENT, self._rule[0], self._rule[1], width=max(2, int(3 * self._scale)))
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
        """Soft emerald glow cached once and pulsed with set_alpha."""
        surf = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
        for i in range(radius, 0, -4):
            alpha = int(28 * (i / radius))
            pygame.draw.circle(surf, (16, 185, 129, alpha), (radius, radius), i)
        return surf

    @staticmethod
    def _make_card(w: int, h: int, radius: int, pad: int = 8) -> pygame.Surface:
        """Rounded semi-transparent panel with subtle emerald border and soft glow."""
        surf = pygame.Surface((w + pad * 2, h + pad * 2), pygame.SRCALPHA)
        inner = pygame.Rect(pad, pad, w, h)

        # Soft green glow halo around the card
        glow_rect = inner.inflate(pad, pad)
        pygame.draw.rect(surf, (*_ACCENT, 22), glow_rect, border_radius=radius + 4)

        # A mostly opaque surface keeps the menu crisp over the animated board.
        pygame.draw.rect(surf, (*_CARD, 232), inner, border_radius=radius)
        pygame.draw.rect(
            surf,
            (77, 88, 102, 220),
            inner,
            width=1,
            border_radius=radius,
        )
        pygame.draw.line(
            surf,
            (*_ACCENT, 120),
            (inner.left + radius, inner.top + 1),
            (inner.right - radius, inner.top + 1),
            width=1,
        )
        return surf
