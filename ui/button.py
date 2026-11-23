"""Reusable animated button component for ChessIQ.

Supports hover glow, smooth scale animation, and resolution-adaptive sizing.
"""

import pygame
import math


class Button:
    """A clickable, animated rectangular button.

    Parameters
    ----------
    rect : pygame.Rect
        The base (un-scaled) bounding rectangle.
    text : str
        Label drawn on the button.
    font : pygame.font.Font
        Font used for the label.
    color : tuple[int, int, int]
        Base fill colour.
    hover_color : tuple[int, int, int]
        Fill colour when hovered.
    text_color : tuple[int, int, int]
        Label colour.
    border_radius : int
        Corner rounding in pixels.
    accent : bool
        If True the button receives a subtle glow outline when hovered.
    gradient_colors : tuple[tuple[int, int, int], tuple[int, int, int]] | None
        Optional start and end colours for a vertical gradient fill.
    hover_gradient_colors : tuple[tuple[int, int, int], tuple[int, int, int]] | None
        Optional start and end colours used while hovered.
    """

    def __init__(
        self,
        rect: pygame.Rect,
        text: str,
        font: pygame.font.Font,
        color: tuple[int, int, int] = (45, 45, 50),
        hover_color: tuple[int, int, int] = (55, 55, 62),
        text_color: tuple[int, int, int] = (230, 230, 230),
        border_radius: int = 12,
        accent: bool = False,
        gradient_colors: tuple[
            tuple[int, int, int], tuple[int, int, int]
        ] | None = None,
        hover_gradient_colors: tuple[
            tuple[int, int, int], tuple[int, int, int]
        ] | None = None,
        shadow_color: tuple[int, int, int] = (10, 12, 16),
        border_color: tuple[int, int, int] | None = None,
        border_width: int | None = None,
        accent_color: tuple[int, int, int] = (16, 185, 129),
        lift_on_hover: bool = False,
        press_feedback: bool = False,
    ) -> None:
        self.base_rect = pygame.Rect(rect)
        self.text = text
        self.font = font
        self.color = color
        self.hover_color = hover_color
        self.text_color = text_color
        self.border_radius = border_radius
        self.accent = accent
        self.gradient_colors = gradient_colors
        self.hover_gradient_colors = hover_gradient_colors or gradient_colors
        self.shadow_color = shadow_color
        self.border_color = border_color
        self.border_width = border_width
        self.lift_on_hover = lift_on_hover
        self.press_feedback = press_feedback

        # animation state
        self._hover = False
        self._hover_t: float = 0.0          # 0.0 → 1.0 smooth transition
        self._pressed_t: float = 0.0
        self._pressed = False
        self._scale: float = 1.0
        self._target_scale: float = 1.0
        self._HOVER_SPEED: float = 15.0 if lift_on_hover else 6.0
        self._SCALE_HOVER: float = 1.04     # subtle enlarge on hover

        # accent glow colour
        self._accent_color: tuple[int, int, int] = accent_color
        self._gradient_surface = (
            self._make_gradient_surface(
                self.base_rect.size, self.gradient_colors, self.border_radius
            )
            if self.gradient_colors
            else None
        )
        self._hover_gradient_surface = (
            self._make_gradient_surface(
                self.base_rect.size, self.hover_gradient_colors, self.border_radius
            )
            if self.hover_gradient_colors
            else None
        )
        self._soft_shadow_surface = (
            self._make_soft_shadow_surface(
                self.base_rect.size, self.border_radius, self.shadow_color
            )
            if self.gradient_colors
            else None
        )
        self._label_cache_key: tuple[str, tuple[int, int, int]] | None = None
        self._label_surface: pygame.Surface | None = None
        self._glow_cache_key: tuple[int, int, int, tuple[int, int, int]] | None = None
        self._glow_surface: pygame.Surface | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def handle_event(self, event: pygame.event.Event) -> bool:
        """Return True if the button was clicked this event."""
        if event.type == pygame.MOUSEMOTION:
            self._hover = self.base_rect.collidepoint(event.pos)
            self._target_scale = (
                self._SCALE_HOVER if self._hover and not self.lift_on_hover else 1.0
            )
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.base_rect.collidepoint(event.pos):
                self._pressed = self.press_feedback
                return True
        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            self._pressed = False
        return False

    def update(self, dt: float) -> None:
        """Advance hover / scale animations by *dt* seconds."""
        # smooth hover interpolation
        target = 1.0 if self._hover else 0.0
        diff = target - self._hover_t
        self._hover_t += diff * min(self._HOVER_SPEED * dt, 1.0)

        # smooth scale interpolation
        s_diff = self._target_scale - self._scale
        self._scale += s_diff * min(self._HOVER_SPEED * dt, 1.0)

        pressed_target = 1.0 if self._pressed else 0.0
        pressed_diff = pressed_target - self._pressed_t
        self._pressed_t += pressed_diff * min(self._HOVER_SPEED * dt, 1.0)

    def draw(self, surface: pygame.Surface) -> None:
        """Draw the button onto *surface*."""
        # compute scaled rect (centred on base_rect centre)
        press_scale = 1.0 - 0.015 * self._pressed_t if self.press_feedback else 1.0
        w = int(self.base_rect.width * self._scale * press_scale)
        h = int(self.base_rect.height * self._scale * press_scale)
        draw_rect = pygame.Rect(0, 0, w, h)
        draw_rect.center = self.base_rect.center
        lift = -round(2 * self._hover_t) if self.lift_on_hover else 0
        press_offset = round(self._pressed_t) if self.press_feedback else 0
        draw_rect.y += lift + press_offset
        radius = min(self.border_radius, max(1, min(w, h) // 2))
        shadow_offset = max(2, min(5, radius // 3))
        shadow_rect = draw_rect.move(0, shadow_offset)

        # interpolated fill colour
        fill = self._lerp_color(self.color, self.hover_color, self._hover_t)

        if self.gradient_colors:
            self._draw_soft_shadow(surface, shadow_rect, radius)
        else:
            pygame.draw.rect(
                surface,
                self.shadow_color,
                shadow_rect,
                border_radius=radius,
            )

        # accent glow outline (drawn slightly larger behind the button)
        if self.accent and self._hover_t > 0.01:
            glow_alpha = int(100 * self._hover_t)
            glow_rect = draw_rect.inflate(6, 6)
            glow_key = (glow_rect.width, glow_rect.height, radius + 3, self._accent_color)
            if self._glow_cache_key != glow_key or self._glow_surface is None:
                self._glow_surface = pygame.Surface(
                    (glow_rect.width, glow_rect.height), pygame.SRCALPHA
                )
                pygame.draw.rect(
                    self._glow_surface,
                    (*self._accent_color, 255),
                    self._glow_surface.get_rect(),
                    border_radius=radius + 3,
                )
                self._glow_cache_key = glow_key
            self._glow_surface.set_alpha(glow_alpha)
            surface.blit(self._glow_surface, glow_rect.topleft)

        # main button body
        if self._gradient_surface is not None:
            gradient = self._gradient_surface
            if gradient.get_size() != draw_rect.size:
                gradient = pygame.transform.smoothscale(gradient, draw_rect.size)
            surface.blit(gradient, draw_rect)
            if self._hover_gradient_surface is not None and self._hover_t > 0:
                hover_gradient = self._hover_gradient_surface
                if hover_gradient.get_size() != draw_rect.size:
                    hover_gradient = pygame.transform.smoothscale(
                        hover_gradient, draw_rect.size
                    )
                hover_gradient.set_alpha(int(255 * self._hover_t))
                surface.blit(hover_gradient, draw_rect)
        else:
            pygame.draw.rect(surface, fill, draw_rect, border_radius=radius)

        # A soft top highlight and restrained outline give flat controls depth.
        if self.border_color is not None:
            base_border = self.border_color
            hover_border = (
                self._accent_color
                if self.accent
                else self._lerp_color(base_border, (255, 255, 255), 0.18)
            )
            border_color = self._lerp_color(base_border, hover_border, self._hover_t)
        elif self.gradient_colors:
            border_color = self._lerp_color(
                self.color, self.hover_color, self._hover_t
            )
        else:
            inset = min(radius, max(2, w // 3))
            highlight = self._lerp_color(fill, (255, 255, 255), 0.16)
            pygame.draw.line(
                surface,
                highlight,
                (draw_rect.left + inset, draw_rect.top + 1),
                (draw_rect.right - inset, draw_rect.top + 1),
            )
            border_color = self._lerp_color(
                (67, 75, 86), self._accent_color if self.accent else (100, 110, 122),
                self._hover_t,
            )
        b_width = (
            self.border_width
            if self.border_width is not None
            else max(1, min(2, h // 20))
        )
        pygame.draw.rect(
            surface,
            border_color,
            draw_rect,
            width=b_width,
            border_radius=radius,
        )

        # label (cached until text or text_color changes)
        if self.text:
            label_key = (self.text, self.text_color)
            if self._label_cache_key != label_key or self._label_surface is None:
                self._label_surface = self.font.render(self.text, True, self.text_color)
                self._label_cache_key = label_key
            label_rect = self._label_surface.get_rect(center=draw_rect.center)
            surface.blit(self._label_surface, label_rect)

    def _draw_soft_shadow(
        self, surface: pygame.Surface, rect: pygame.Rect, radius: int
    ) -> None:
        """Draw the cached layered shadow behind the button."""
        shadow = self._soft_shadow_surface
        if shadow is None:
            return
        padding = max(6, int(radius * 0.7))
        strength = min(255, int(205 + 50 * self._hover_t - 100 * self._pressed_t))
        shadow.set_alpha(strength)
        surface.blit(shadow, (rect.centerx - shadow.get_width() // 2, rect.top - padding))

    @staticmethod
    def _make_soft_shadow_surface(
        size: tuple[int, int],
        radius: int,
        color: tuple[int, int, int],
    ) -> pygame.Surface:
        """Pre-render a layered green shadow for reuse while drawing."""
        width, height = size
        padding = max(6, int(radius * 0.7))
        shadow = pygame.Surface(
            (width + padding * 2, height + padding * 2), pygame.SRCALPHA
        )
        for spread, alpha in ((padding, 12), (padding // 2, 24), (2, 48)):
            shadow_rect = pygame.Rect(
                padding - spread,
                padding - spread + padding // 3,
                width + spread * 2,
                height + spread * 2,
            )
            pygame.draw.rect(
                shadow,
                (*color, alpha),
                shadow_rect,
                border_radius=radius + spread,
            )
        return shadow

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _make_gradient_surface(
        size: tuple[int, int],
        colors: tuple[tuple[int, int, int], tuple[int, int, int]],
        radius: int,
    ) -> pygame.Surface:
        """Build a rounded vertical gradient once for this button."""
        width, height = size
        gradient = pygame.Surface(size, pygame.SRCALPHA)
        radius = min(radius, width // 2, height // 2)
        for y in range(height):
            if y < radius:
                offset = radius - math.sqrt(radius * radius - (radius - y) ** 2)
            elif y >= height - radius:
                offset = radius - math.sqrt(
                    radius * radius - (radius - (height - 1 - y)) ** 2
                )
            else:
                offset = 0
            inset = int(offset)
            t = y / max(height - 1, 1)
            color = Button._lerp_color(colors[0], colors[1], t)
            pygame.draw.line(
                gradient,
                (*color, 255),
                (inset, y),
                (width - 1 - inset, y),
            )
        return gradient

    @staticmethod
    def _lerp_color(
        a: tuple[int, int, int],
        b: tuple[int, int, int],
        t: float,
    ) -> tuple[int, int, int]:
        return (
            int(a[0] + (b[0] - a[0]) * t),
            int(a[1] + (b[1] - a[1]) * t),
            int(a[2] + (b[2] - a[2]) * t),
        )
