"""Settings screen for Chess Ken."""

from collections.abc import Callable

import pygame

from screens.base_screen import BaseScreen
from ui.button import Button


_BG = (16, 19, 25)
_PANEL = (27, 32, 41)
_TEXT = (235, 239, 244)
_MUTED = (155, 164, 178)
_ACCENT = (16, 185, 129)
_TRACK = (55, 64, 76)


class SettingsScreen(BaseScreen):
    """Adjust audio volume and gameplay preferences (such as Premove)."""

    def __init__(
        self,
        screen: pygame.Surface,
        volume: int = 70,
        on_volume_changed: Callable[[int], None] | None = None,
        return_screen: str = "dashboard",
        premove_enabled: bool = True,
        on_premove_changed: Callable[[bool], None] | None = None,
    ) -> None:
        super().__init__(screen)
        self._scale = min(self.width / 1280, self.height / 720)
        self._volume = max(0, min(100, volume))
        self._on_volume_changed = on_volume_changed
        self._return_screen = return_screen
        self._premove_enabled = bool(premove_enabled)
        self._on_premove_changed = on_premove_changed
        self._premove_knob_t: float = 1.0 if self._premove_enabled else 0.0
        self._dragging = False
        self._dragging_premove = False

        self._font_title = pygame.font.SysFont(
            "Segoe UI", max(36, int(54 * self._scale)), bold=True
        )
        self._font_label = pygame.font.SysFont(
            "Segoe UI", max(20, int(26 * self._scale)), bold=True
        )
        self._font_value = pygame.font.SysFont(
            "Segoe UI", max(24, int(32 * self._scale)), bold=True
        )
        self._font_hint = pygame.font.SysFont(
            "Segoe UI", max(14, int(17 * self._scale))
        )
        self._font_state = pygame.font.SysFont(
            "Segoe UI", max(14, int(18 * self._scale)), bold=True
        )
        self._font_button = pygame.font.SysFont(
            "Segoe UI", max(18, int(22 * self._scale)), bold=True
        )

        self._panel = pygame.Rect(
            0, 0, int(700 * self._scale), int(410 * self._scale)
        )
        self._panel.center = (self.width // 2, int(self.height * 0.52))
        self._title = self._font_title.render("SETTINGS", True, _TEXT)
        self._label = self._font_label.render("Music / Audio Volume", True, _TEXT)
        self._hint = self._font_hint.render(
            "Drag the slider to adjust volume", True, _MUTED
        )
        self._premove_label = self._font_label.render("Premove", True, _TEXT)
        self._premove_hint = self._font_hint.render(
            "Queue your next move during the opponent's turn", True, _MUTED
        )

        inner_pad = int(58 * self._scale)
        self._track = pygame.Rect(
            self._panel.left + inner_pad,
            self._panel.top + int(152 * self._scale),
            self._panel.width - inner_pad * 2,
            max(8, int(12 * self._scale)),
        )
        self._value_right = (
            self._panel.right - inner_pad,
            self._panel.top + int(82 * self._scale),
        )
        self._divider_y = self._panel.top + int(222 * self._scale)

        # Sliding toggle switch for Premove
        sw_w = max(52, int(68 * self._scale))
        sw_h = max(28, int(36 * self._scale))
        self._premove_row_y = self._panel.top + int(306 * self._scale)
        self._premove_switch_rect = pygame.Rect(0, 0, sw_w, sw_h)
        self._premove_switch_rect.midright = (
            self._panel.right - inner_pad,
            self._premove_row_y,
        )

        back_rect = pygame.Rect(
            0, 0, int(250 * self._scale), int(58 * self._scale)
        )
        back_rect.center = (self.width // 2, int(self.height * 0.88))
        self._btn_back = Button(
            back_rect,
            "BACK",
            self._font_button,
            color=(37, 44, 54),
            hover_color=(52, 62, 74),
            text_color=_TEXT,
            border_radius=max(8, int(12 * self._scale)),
        )

    @property
    def volume(self) -> int:
        """Current volume as an integer percentage."""
        return self._volume

    @property
    def premove_enabled(self) -> bool:
        """Whether premove is currently enabled."""
        return self._premove_enabled

    def set_premove_enabled(self, enabled: bool) -> None:
        """Update the premove toggle state and notify listener if changed."""
        enabled = bool(enabled)
        if enabled != self._premove_enabled:
            self._premove_enabled = enabled
            if self._on_premove_changed is not None:
                self._on_premove_changed(enabled)

    def _set_volume_from_x(self, x: int) -> None:
        ratio = (x - self._track.left) / self._track.width
        volume = round(max(0.0, min(1.0, ratio)) * 100)
        if volume != self._volume:
            self._volume = volume
            if self._on_volume_changed is not None:
                self._on_volume_changed(volume)

    def handle_events(self, events: list[pygame.event.Event]) -> None:
        for event in events:
            if self._btn_back.handle_event(event):
                self.next_screen = self._return_screen
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if self._track.inflate(0, int(36 * self._scale)).collidepoint(event.pos):
                    self._dragging = True
                    self._set_volume_from_x(event.pos[0])
                elif self._premove_switch_rect.inflate(16, 16).collidepoint(event.pos):
                    self._dragging_premove = True
                    self.set_premove_enabled(not self._premove_enabled)
            elif event.type == pygame.MOUSEMOTION:
                if self._dragging:
                    self._set_volume_from_x(event.pos[0])
                elif self._dragging_premove:
                    self.set_premove_enabled(
                        event.pos[0] >= self._premove_switch_rect.centerx
                    )
            elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                self._dragging = False
                self._dragging_premove = False

    def update(self, dt: float) -> None:
        self._btn_back.update(dt)
        target = 1.0 if self._premove_enabled else 0.0
        step = min(14.0 * max(0.0, dt), 1.0)
        self._premove_knob_t += (target - self._premove_knob_t) * step

    def draw(self) -> None:
        self.screen.fill(_BG)
        title_rect = self._title.get_rect(
            center=(self.width // 2, int(self.height * 0.14))
        )
        self.screen.blit(self._title, title_rect)

        panel_rad = max(12, int(22 * self._scale))
        pygame.draw.rect(self.screen, _PANEL, self._panel, border_radius=panel_rad)
        pygame.draw.rect(
            self.screen,
            (48, 57, 69),
            self._panel,
            width=max(1, int(2 * self._scale)),
            border_radius=panel_rad,
        )

        # Row 1: Music / Audio Volume slider
        label_rect = self._label.get_rect(
            midleft=(self._track.left, self._panel.top + int(78 * self._scale))
        )
        self.screen.blit(self._label, label_rect)
        value_surface = self._font_value.render(f"{self._volume}%", True, _ACCENT)
        self.screen.blit(
            value_surface, value_surface.get_rect(midright=self._value_right)
        )
        hint_rect = self._hint.get_rect(
            midleft=(self._track.left, self._panel.top + int(112 * self._scale))
        )
        self.screen.blit(self._hint, hint_rect)

        pygame.draw.rect(
            self.screen, _TRACK, self._track, border_radius=self._track.height // 2
        )
        fill_width = round(self._track.width * self._volume / 100)
        if fill_width:
            fill_rect = pygame.Rect(
                self._track.left, self._track.top, fill_width, self._track.height
            )
            pygame.draw.rect(
                self.screen, _ACCENT, fill_rect, border_radius=self._track.height // 2
            )
        knob_x = self._track.left + fill_width
        pygame.draw.circle(
            self.screen,
            (218, 255, 242),
            (knob_x, self._track.centery),
            max(9, int(15 * self._scale)),
        )
        pygame.draw.circle(
            self.screen,
            _ACCENT,
            (knob_x, self._track.centery),
            max(5, int(8 * self._scale)),
        )

        # Subtle section divider
        pygame.draw.line(
            self.screen,
            (45, 53, 66),
            (self._track.left, self._divider_y),
            (self._track.right, self._divider_y),
            width=1,
        )

        # Row 2: Premove sliding toggle button
        pm_label_rect = self._premove_label.get_rect(
            midleft=(self._track.left, self._premove_row_y - int(16 * self._scale))
        )
        self.screen.blit(self._premove_label, pm_label_rect)
        pm_hint_rect = self._premove_hint.get_rect(
            midleft=(self._track.left, self._premove_row_y + int(18 * self._scale))
        )
        self.screen.blit(self._premove_hint, pm_hint_rect)

        sw = self._premove_switch_rect
        t = max(0.0, min(1.0, self._premove_knob_t))
        track_col = (
            int(_TRACK[0] + (_ACCENT[0] - _TRACK[0]) * t),
            int(_TRACK[1] + (_ACCENT[1] - _TRACK[1]) * t),
            int(_TRACK[2] + (_ACCENT[2] - _TRACK[2]) * t),
        )
        pygame.draw.rect(self.screen, track_col, sw, border_radius=sw.height // 2)
        border_col = (
            int(74 + (52 - 74) * t),
            int(86 + (211 - 86) * t),
            int(102 + (153 - 102) * t),
        )
        pygame.draw.rect(
            self.screen, border_col, sw, width=1, border_radius=sw.height // 2
        )

        pad = max(3, int(4 * self._scale))
        knob_r = max(8, sw.height // 2 - pad)
        knob_min_x = sw.left + pad + knob_r
        knob_max_x = sw.right - pad - knob_r
        sw_knob_x = round(knob_min_x + (knob_max_x - knob_min_x) * t)
        pygame.draw.circle(
            self.screen, (242, 248, 246), (sw_knob_x, sw.centery), knob_r
        )

        state_txt = "ON" if self._premove_enabled else "OFF"
        state_col = _ACCENT if self._premove_enabled else _MUTED
        state_surf = self._font_state.render(state_txt, True, state_col)
        self.screen.blit(
            state_surf,
            state_surf.get_rect(
                midright=(sw.left - max(12, int(16 * self._scale)), sw.centery)
            ),
        )

        self._btn_back.draw(self.screen)
