"""Reusable hamburger button with a dropdown action menu."""

import numpy as np
import pygame

from others.config import ASSETS_DIR
from ui.button import Button


_ACTIONS = (
    ("Resume", "resume"),
    ("Restart", "restart"),
    ("Quit", "opponent_select"),
)
_ACTION_ICON_FILES: dict[str, str] = {
    "resign": "resign.png",
    "restart": "restart.jpg",
}
_MENU_TEXT_COLOR = (235, 239, 244)
_TOGGLE_MARGIN = 22
_MIN_TOGGLE_MARGIN = 12
_TOGGLE_SIZE = 50
_MIN_TOGGLE_SIZE = 42
_MIN_TOGGLE_TOP = 8

# In-match menu: the default actions plus Resign / Offer Draw / Exit to dashboard.
MATCH_ACTIONS = (
    ("Resume", "resume"),
    ("Resign", "resign"),
    ("Offer Draw", "offer_draw"),
    ("Restart", "restart"),
    ("Quit", "opponent_select"),
    ("Exit", "dashboard"),
)

# Post-game menu (after the match has ended): only Resume, Restart, and Quit.
ENDED_MATCH_ACTIONS = _ACTIONS


class HamburgerMenu:
    """Toggle and render a screen-corner dropdown menu."""

    _icon_cache: dict[tuple[str, int], pygame.Surface] = {}

    @classmethod
    def _load_menu_icon(cls, filename: str, size: int) -> pygame.Surface | None:
        cache_key = (filename, size)
        if cache_key in cls._icon_cache:
            return cls._icon_cache[cache_key]
        path = ASSETS_DIR / filename
        if not path.is_file():
            return None
        image = pygame.image.load(str(path)).convert_alpha()
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
        t_rgb[:, :, 0] = _MENU_TEXT_COLOR[0]
        t_rgb[:, :, 1] = _MENU_TEXT_COLOR[1]
        t_rgb[:, :, 2] = _MENU_TEXT_COLOR[2]
        del t_rgb
        t_alpha = pygame.surfarray.pixels_alpha(tinted)
        t_alpha[:, :] = final_alpha
        del t_alpha
        bounds = tinted.get_bounding_rect(min_alpha=20)
        if bounds.width > 0 and bounds.height > 0:
            tinted = tinted.subsurface(bounds).copy()
        bw, bh = tinted.get_size()
        fit_scale = min(size / max(1, bw), size / max(1, bh))
        scaled = pygame.transform.smoothscale(
            tinted,
            (max(1, min(size, round(bw * fit_scale))), max(1, min(size, round(bh * fit_scale)))),
        )
        icon = pygame.Surface((size, size), pygame.SRCALPHA)
        icon.blit(scaled, scaled.get_rect(center=(size // 2, size // 2)))
        cls._icon_cache[cache_key] = icon
        return icon

    def __init__(
        self,
        width: int,
        height: int,
        scale: float,
        actions: tuple[tuple[str, str], ...] = _ACTIONS,
        toggle_bottom_limit: int | None = None,
    ) -> None:
        self.is_open = False
        self._scale = scale
        self._height = height
        self._font = pygame.font.SysFont(
            "Segoe UI", max(15, int(17 * scale)), bold=True
        )
        self._menu_font = pygame.font.SysFont(
            "Segoe UI", max(15, int(16 * scale))
        )

        margin = max(_MIN_TOGGLE_MARGIN, int(_TOGGLE_MARGIN * scale))
        self._margin = margin
        toggle_size = max(_MIN_TOGGLE_SIZE, int(_TOGGLE_SIZE * scale))
        toggle_top = margin
        if toggle_bottom_limit is not None:
            min_top = max(_MIN_TOGGLE_TOP, round(_MIN_TOGGLE_TOP * scale))
            toggle_top = max(
                min_top,
                min(margin, toggle_bottom_limit - toggle_size),
            )
        toggle_rect = pygame.Rect(
            width - margin - toggle_size,
            toggle_top,
            toggle_size,
            toggle_size,
        )
        self._toggle = Button(
            toggle_rect,
            "",
            self._font,
            color=(37, 44, 54),
            hover_color=(52, 62, 74),
            border_radius=max(8, int(10 * scale)),
        )

        self._actions: tuple[tuple[str, str], ...] = ()
        self._panel = pygame.Rect(0, 0, 0, 0)
        self._item_icons: dict[str, pygame.Surface] = {}
        self._item_labels: dict[str, pygame.Surface] = {}
        self._items: list[tuple[str, Button]] = []
        self.set_actions(actions)

    @property
    def actions(self) -> tuple[str, ...]:
        """Return the ordered action keys currently shown in the dropdown."""
        return tuple(action for action, _ in self._items)

    def set_actions(self, actions: tuple[tuple[str, str], ...]) -> None:
        """Replace the menu items and resize the dropdown panel cleanly."""
        actions_tuple = tuple(actions)
        if actions_tuple == self._actions and self._items:
            return
        self._actions = actions_tuple
        scale = self._scale
        toggle_rect = self._toggle.base_rect
        panel_width = max(220, int(270 * scale))
        padding = max(6, int(8 * scale))
        item_height = max(36, int(44 * scale))
        item_gap = max(3, int(5 * scale))
        count = len(actions_tuple)
        panel_height = (
            padding * 2
            + count * item_height
            + max(0, count - 1) * item_gap
        )
        panel_left = toggle_rect.right - panel_width
        panel_top = toggle_rect.bottom + max(5, int(8 * scale))
        if panel_top + panel_height > self._height - self._margin:
            panel_top = max(self._margin, self._height - self._margin - panel_height)
        self._panel = pygame.Rect(panel_left, panel_top, panel_width, panel_height)

        item_width = panel_width - padding * 2
        icon_size = max(15, int(18 * scale))
        self._item_icons = {}
        self._item_labels = {}
        self._items = []
        for index, (label, action) in enumerate(actions_tuple):
            rect = pygame.Rect(
                panel_left + padding,
                panel_top + padding + index * (item_height + item_gap),
                item_width,
                item_height,
            )
            icon_file = _ACTION_ICON_FILES.get(action)
            icon_surf = (
                self._load_menu_icon(icon_file, icon_size) if icon_file else None
            )
            if icon_surf is not None:
                self._item_icons[action] = icon_surf
                self._item_labels[action] = self._menu_font.render(
                    label, True, _MENU_TEXT_COLOR
                )
            self._items.append(
                (
                    action,
                    Button(
                        rect,
                        "" if icon_surf is not None else label,
                        self._menu_font,
                        color=(37, 44, 54),
                        hover_color=(52, 62, 74),
                        text_color=_MENU_TEXT_COLOR,
                        border_radius=max(6, int(8 * scale)),
                    ),
                )
            )

    @property
    def toggle_rect(self) -> pygame.Rect:
        """Return the bounds of the hamburger toggle."""
        return self._toggle.base_rect.copy()

    def toggle(self) -> None:
        """Open or close the dropdown."""
        self.is_open = not self.is_open

    def handle_event(self, event: pygame.event.Event) -> str | None:
        """Return the chosen action key, if any."""
        if self._toggle.handle_event(event):
            self.toggle()
            return None

        if not self.is_open:
            return None

        for action, button in self._items:
            if button.handle_event(event):
                self.is_open = False
                return action

        if (
            event.type == pygame.MOUSEBUTTONDOWN
            and event.button == 1
            and not self._panel.collidepoint(event.pos)
        ):
            self.is_open = False
        return None

    def update(self, dt: float) -> None:
        self._toggle.update(dt)
        for _, button in self._items:
            button.update(dt)

    def draw(self, surface: pygame.Surface) -> None:
        self._toggle.draw(surface)
        rect = self._toggle.base_rect
        line_width = max(2, int(3 * self._scale))
        line_length = int(rect.width * 0.42)
        center_x = rect.centerx
        for offset in (-1, 0, 1):
            y = rect.centery + offset * max(5, int(7 * self._scale))
            pygame.draw.line(
                surface,
                (235, 239, 244),
                (center_x - line_length // 2, y),
                (center_x + line_length // 2, y),
                line_width,
            )

        if self.is_open:
            pygame.draw.rect(
                surface,
                (24, 28, 36),
                self._panel,
                border_radius=max(8, int(12 * self._scale)),
            )
            pygame.draw.rect(
                surface,
                (48, 57, 69),
                self._panel,
                width=max(1, int(2 * self._scale)),
                border_radius=max(8, int(12 * self._scale)),
            )
            icon_gap = max(6, int(8 * self._scale))
            for action, button in self._items:
                button.draw(surface)
                icon_surf = self._item_icons.get(action)
                label_surf = self._item_labels.get(action)
                if icon_surf is not None and label_surf is not None:
                    rect = button.base_rect
                    total_w = icon_surf.get_width() + icon_gap + label_surf.get_width()
                    start_x = rect.centerx - total_w // 2
                    surface.blit(
                        icon_surf,
                        icon_surf.get_rect(midleft=(start_x, rect.centery)),
                    )
                    surface.blit(
                        label_surf,
                        label_surf.get_rect(
                            midleft=(start_x + icon_surf.get_width() + icon_gap, rect.centery)
                        ),
                    )
