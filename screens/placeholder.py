"""PlaceholderScreen — temporary stand-in for screens not yet built.

Shows a centred label and a BACK button that returns to the dashboard.
Used for Opponent Selection and Settings until those screens are implemented.
"""

import pygame

from screens.base_screen import BaseScreen
from ui.button import Button


_BG = (22, 22, 28)
_LABEL_COLOR = (160, 160, 170)
_BTN_COLOR = (38, 38, 44)
_BTN_HOVER = (50, 50, 58)
_BTN_TEXT = (225, 225, 230)


class PlaceholderScreen(BaseScreen):
    """A minimal screen with a title label and a BACK button."""

    def __init__(self, screen: pygame.Surface, title: str = "Coming Soon") -> None:
        super().__init__(screen)
        scale = min(self.width / 1920, self.height / 1080)

        self._font_title = pygame.font.SysFont("Segoe UI", int(48 * scale), bold=True)
        self._font_hint = pygame.font.SysFont("Segoe UI", int(22 * scale))
        self._font_btn = pygame.font.SysFont("Segoe UI", int(24 * scale), bold=True)

        self._title_surf = self._font_title.render(title, True, _LABEL_COLOR)
        self._hint_surf = self._font_hint.render(
            "This screen is not implemented yet.", True, (90, 90, 100)
        )

        btn_w = int(260 * scale)
        btn_h = int(56 * scale)
        radius = int(12 * scale)
        back_rect = pygame.Rect(0, 0, btn_w, btn_h)
        back_rect.center = (self.width // 2, int(self.height * 0.62))

        self._btn_back = Button(
            back_rect, "B A C K", self._font_btn,
            color=_BTN_COLOR, hover_color=_BTN_HOVER,
            text_color=_BTN_TEXT, border_radius=radius,
        )

    # ------------------------------------------------------------------
    # BaseScreen interface
    # ------------------------------------------------------------------

    def handle_events(self, events: list[pygame.event.Event]) -> None:
        for event in events:
            if self._btn_back.handle_event(event):
                self.next_screen = "dashboard"

    def update(self, dt: float) -> None:
        self._btn_back.update(dt)

    def draw(self) -> None:
        self.screen.fill(_BG)
        title_rect = self._title_surf.get_rect(
            center=(self.width // 2, int(self.height * 0.38))
        )
        self.screen.blit(self._title_surf, title_rect)

        hint_rect = self._hint_surf.get_rect(
            center=(self.width // 2, int(self.height * 0.46))
        )
        self.screen.blit(self._hint_surf, hint_rect)

        self._btn_back.draw(self.screen)
