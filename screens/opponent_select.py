"""Modernized Choose Your Opponent screen for ChessIQ ("Chess Ken").

Features a futuristic dark-emerald glassmorphism aesthetic, atmospheric
chess-themed backdrop with subtle ambient particles, emerald-gradient headline,
prominent circular character portraits with accent-colored halos, pill ELO
badges, Focus Ken recommended spotlight, smooth hover micro-interactions, and
responsive desktop / tablet / mobile layouts.
"""

from __future__ import annotations

import math
from pathlib import Path
import random
from typing import Any

import numpy as np
import pygame

from config import ASSETS_DIR
from screens.base_screen import BaseScreen
from ui.animated_board import MAX_ANIMATION_DT, _load_svg_at_size
from ui.button import Button


# ── Futuristic Dark-Emerald Design Palette ───────────────────────────
_BG_MAIN = (5, 11, 13)                   # #050B0D
_BG_SECONDARY = (11, 20, 24)             # #0B1418
_CARD_BG = (16, 28, 34)                  # #101C22
_CARD_BG_HOVER = (22, 38, 46)            # elevated glass hover tint
_EMERALD_PRIMARY = (0, 214, 160)         # #00D6A0
_EMERALD_SECONDARY = (0, 143, 112)       # #008F70
_EMERALD_HOVER_TOP = (28, 232, 180)
_EMERALD_HOVER_BOT = (0, 166, 130)
_ELECTRIC_BLUE = (54, 191, 255)          # #36BFFF
_GOLD_ACCENT = (255, 194, 71)            # #FFC247
_TEXT_PRIMARY = (241, 245, 249)          # #F1F5F9
_TEXT_SECONDARY = (148, 163, 184)        # #94A3B8
_BORDER_SUBTLE_RGB = (58, 76, 86)        # crisp resting glass border (~20% slate on dark)
_BORDER_RECOMMENDED_RGB = (0, 164, 126)  # refined emerald resting border
_ELO_PILL_BG = (8, 16, 20, 220)
_BACK_BG = (16, 28, 34)
_BACK_HOVER = (24, 42, 50)

_HOVER_ANIM_SPEED = 9.0  # ~220 ms smooth settle
_PARTICLE_COUNT = 18

# Preserved opponent profiles (names, ELOs, images, taglines, descriptions)
# plus per-bot visual theme accents.
_BOTS: tuple[dict[str, Any], ...] = (
    {
        "name": "Sick Ken",
        "elo": 1350,
        "image": "sickken.png",
        "tagline": "Bai gihilantan ko, pildiha daw ko bi",
        "description": "A weaker take on your style,\nwith occasional tactical errors.",
        "accent": _GOLD_ACCENT,
        "badge_icon": "spark",
        "recommended": False,
    },
    {
        "name": "Focus Ken",
        "elo": 1800,
        "image": "focusken.png",
        "tagline": "Ha?? sure ka makig duwa ka nako",
        "description": "Consistent decisions and\nyour familiar openings.",
        "accent": _EMERALD_PRIMARY,
        "badge_icon": "target",
        "recommended": True,
    },
    {
        "name": "Condition Ken",
        "elo": 2350,
        "image": "conditionken.jpg",
        "tagline": "hmmm dependi kong 3 yan!",
        "description": "Advanced tactical and\npositional precision.",
        "accent": _ELECTRIC_BLUE,
        "badge_icon": "crown",
        "recommended": False,
    },
)


def _lerp_rgb(
    a: tuple[int, int, int], b: tuple[int, int, int], t: float
) -> tuple[int, int, int]:
    t_c = max(0.0, min(1.0, t))
    return (
        int(a[0] + (b[0] - a[0]) * t_c),
        int(a[1] + (b[1] - a[1]) * t_c),
        int(a[2] + (b[2] - a[2]) * t_c),
    )


def _load_modern_font(size: int, weight: str = "regular") -> pygame.font.Font:
    """Load clean modern UI typography using genuine font-weight files when installed."""
    size = max(10, size)
    preferred_bold = ("Plus Jakarta Sans", "Manrope", "Segoe UI")
    preferred_semi = ("Plus Jakarta Sans SemiBold", "Manrope SemiBold", "Segoe UI Semibold", "Segoe UI")
    preferred_reg = ("Plus Jakarta Sans", "Manrope", "Segoe UI")

    if weight == "bold":
        for family in preferred_bold:
            path = pygame.font.match_font(family, bold=True)
            if path:
                return pygame.font.Font(path, size)
        return pygame.font.SysFont("Segoe UI", size, bold=True)

    if weight == "semibold":
        for family in preferred_semi:
            path = pygame.font.match_font(family)
            if path:
                return pygame.font.Font(path, size)
        return pygame.font.SysFont("Segoe UI", size, bold=True)

    for family in preferred_reg:
        path = pygame.font.match_font(family)
        if path:
            if path.lower().endswith("segoeuil.ttf"):
                alt = Path(path).with_name("segoeui.ttf")
                if alt.is_file():
                    return pygame.font.Font(str(alt), size)
            return pygame.font.Font(path, size)
    return pygame.font.SysFont("Segoe UI", size)


class OpponentSelectScreen(BaseScreen):
    """Choose one of the three Chess Ken AI opponent profiles."""

    _avatar_cache: dict[tuple[str, int], pygame.Surface] = {}
    _icon_cache: dict[tuple[str, int, tuple[int, int, int]], pygame.Surface] = {}

    def __init__(self, screen: pygame.Surface) -> None:
        super().__init__(screen)
        self._scale = min(self.width / 1280, self.height / 720)
        self.selected_opponent: dict[str, Any] | None = None
        self._hovered_card: int | None = None
        self._card_hover_t: list[float] = [0.0 for _ in _BOTS]
        self._time: float = 0.0
        self._scroll_y: int = 0
        self._max_scroll_y: int = 0

        # Responsive mode: desktop (>=960), tablet (620..959), mobile (<620)
        if self.width < 620:
            self._layout_mode = "mobile"
        elif self.width < 960:
            self._layout_mode = "tablet"
        else:
            self._layout_mode = "desktop"

        ui_s = max(0.72, min(1.18, self._scale)) if self._layout_mode != "mobile" else max(0.78, self.width / 460.0)
        self._ui_scale = ui_s

        # ── Typography (real weight files, cached once) ───────────────
        title_px = (
            max(22, round(28 * ui_s))
            if self._layout_mode == "mobile"
            else max(32, round(46 * ui_s))
        )
        self._font_title = _load_modern_font(title_px, "bold")
        self._font_subtitle = _load_modern_font(max(13, round(18 * ui_s)), "regular")
        self._font_name = _load_modern_font(max(18, round(25 * ui_s)), "bold")
        self._font_elo_lbl = _load_modern_font(max(12, round(14 * ui_s)), "semibold")
        self._font_elo_num = _load_modern_font(max(14, round(17 * ui_s)), "bold")
        self._font_description = _load_modern_font(max(12, round(15 * ui_s)), "regular")
        self._font_button = _load_modern_font(max(14, round(17 * ui_s)), "bold")
        self._font_back = _load_modern_font(max(14, round(16 * ui_s)), "semibold")
        self._font_badge = _load_modern_font(max(10, round(12 * ui_s)), "bold")
        self._font_status = _load_modern_font(max(13, round(15 * ui_s)), "semibold")

        # ── Responsive Card, Portrait & Button Geometry ───────────────
        self._cards: list[pygame.Rect] = []
        self._button_base_rects: list[pygame.Rect] = []
        self._buttons: list[Button] = []
        self._build_layout()

        # ── Cached Visual Assets & Surfaces ───────────────────────────
        self._knight_emblem = self._load_knight_emblem(max(22, round(30 * ui_s)))
        self._header_title_surf = self._render_gradient_headline()
        self._subtitle_surf = self._font_subtitle.render(
            "Pick a version of Ken to challenge", True, _TEXT_SECONDARY
        )
        self._back_arrow_surf = self._make_back_arrow_icon(
            max(14, round(17 * ui_s)), _TEXT_PRIMARY
        )
        self._recommended_badge_surf = self._build_recommended_badge()

        # Circular portraits + accent halos + small archetype icons
        self._avatars: list[pygame.Surface] = [
            self._load_avatar(str(bot["image"]), self._avatar_size) for bot in _BOTS
        ]
        self._avatar_halos: list[pygame.Surface] = [
            self._make_radial_glow(int(self._avatar_size * 0.92), bot["accent"], peak_alpha=72)
            for bot in _BOTS
        ]
        badge_icon_sz = max(18, round(24 * ui_s))
        self._bot_emblems: list[pygame.Surface] = [
            self._make_archetype_badge(str(bot["badge_icon"]), badge_icon_sz, bot["accent"])
            for bot in _BOTS
        ]

        # Pre-rendered card texts, ELO pills, glass surfaces, shadows, and glows
        self._card_name_surfs: list[pygame.Surface] = []
        self._card_elo_pills: list[pygame.Surface] = []
        self._card_desc_lines: list[list[pygame.Surface]] = []
        self._card_shadows: list[pygame.Surface] = []
        self._card_glass_surfs: list[pygame.Surface] = []
        self._card_hover_glows: list[pygame.Surface] = []
        self._build_card_caches()

        # Pre-rendered atmospheric chess background + lightweight ambient particles
        self._bg_surface = self._build_background()
        self._particle_surf = self._make_radial_glow(6, _EMERALD_PRIMARY, peak_alpha=150)
        rng = random.Random(2026)
        self._particles: list[dict[str, float]] = [
            {
                "x": rng.uniform(0.04, 0.96) * self.width,
                "y": rng.uniform(0.06, 0.94) * self.height,
                "vy": rng.uniform(-11.0, -4.5),
                "vx": rng.uniform(-4.0, 4.0),
                "phase": rng.uniform(0.0, math.tau),
                "alpha": rng.uniform(0.20, 0.55),
            }
            for _ in range(_PARTICLE_COUNT)
        ]

        self._status_surface: pygame.Surface | None = None

    # ------------------------------------------------------------------
    # Responsive Layout Construction
    # ------------------------------------------------------------------

    def _build_layout(self) -> None:
        """Compute responsive header, card, portrait, and button geometry."""
        ui_s = self._ui_scale
        self._card_radius = max(14, round(18 * ui_s))

        if self._layout_mode == "mobile":
            margin_x = max(16, int(self.width * 0.05))
            card_w = max(240, self.width - margin_x * 2)
            self._avatar_size = max(72, min(92, round(82 * ui_s)))
            card_h = max(304, round(316 * ui_s))
            gap = max(26, round(30 * ui_s))

            self._header_knight_y = max(26, round(32 * ui_s))
            self._header_title_y = self._header_knight_y + max(28, round(36 * ui_s))
            self._header_sub_y = self._header_title_y + max(26, round(32 * ui_s))
            cards_top = self._header_sub_y + max(34, round(40 * ui_s))

            btn_w = min(card_w - 36, max(160, round(card_w * 0.72)))
            btn_h = max(42, round(46 * ui_s))

            for idx, bot in enumerate(_BOTS):
                top = cards_top + idx * (card_h + gap)
                card = pygame.Rect((self.width - card_w) // 2, top, card_w, card_h)
                self._cards.append(card)
                btn_rect = pygame.Rect(0, 0, btn_w, btn_h)
                btn_rect.center = (card.centerx, card.bottom - max(36, round(40 * ui_s)))
                self._button_base_rects.append(btn_rect.copy())
                self._buttons.append(self._make_select_button(btn_rect))

            back_w = min(card_w, max(170, round(200 * ui_s)))
            back_h = max(42, round(46 * ui_s))
            back_top = self._cards[-1].bottom + max(24, round(28 * ui_s))
            self._back_base_rect = pygame.Rect((self.width - back_w) // 2, back_top, back_w, back_h)
            total_content_h = self._back_base_rect.bottom + max(24, round(28 * ui_s))
            self._max_scroll_y = max(0, total_content_h - self.height)
            self._status_y = self._cards[-1].bottom + max(10, round(12 * ui_s))
        else:
            # Desktop & Tablet: 3 balanced horizontal cards
            margin_x = max(20, int(self.width * 0.045))
            gap = max(16, round(28 * ui_s)) if self._layout_mode == "desktop" else max(12, round(18 * ui_s))
            max_card_w = round(344 * ui_s)
            avail_card_w = (self.width - margin_x * 2 - gap * 2) // 3
            card_w = max(180, min(max_card_w, avail_card_w))

            self._header_knight_y = max(26, int(self.height * 0.068))
            self._header_title_y = max(self._header_knight_y + 32, int(self.height * 0.132))
            self._header_sub_y = self._header_title_y + max(30, round(38 * ui_s))

            back_h = max(42, round(48 * ui_s))
            back_w = max(164, round(196 * ui_s))
            back_bottom_pad = max(20, int(self.height * 0.04))
            back_y = self.height - back_bottom_pad - back_h
            self._back_base_rect = pygame.Rect((self.width - back_w) // 2, back_y, back_w, back_h)

            cards_top_min = self._header_sub_y + max(38, round(48 * ui_s))
            cards_bottom_max = back_y - max(24, round(32 * ui_s))
            avail_h = max(260, cards_bottom_max - cards_top_min)
            card_h = min(avail_h, max(310, round(412 * ui_s)))
            card_top = cards_top_min + max(0, (avail_h - card_h) // 2)

            self._avatar_size = max(
                76,
                min(round(126 * ui_s), int(card_w * 0.40), int(card_h * 0.31)),
            )

            total_w = card_w * len(_BOTS) + gap * (len(_BOTS) - 1)
            first_left = (self.width - total_w) // 2

            btn_w = max(130, card_w - max(36, round(52 * ui_s)))
            btn_h = max(42, round(50 * ui_s))

            for idx, bot in enumerate(_BOTS):
                # Focus Ken (recommended) gets a refined subtle prominence (+8px height, -4px top)
                is_rec = bool(bot["recommended"])
                extra_h = round(10 * ui_s) if is_rec and (card_h + 10 <= avail_h) else 0
                c_top = card_top - extra_h // 2
                card = pygame.Rect(
                    first_left + idx * (card_w + gap),
                    c_top,
                    card_w,
                    card_h + extra_h,
                )
                self._cards.append(card)
                btn_rect = pygame.Rect(0, 0, btn_w, btn_h)
                btn_rect.center = (card.centerx, card.bottom - max(34, round(40 * ui_s)))
                self._button_base_rects.append(btn_rect.copy())
                self._buttons.append(self._make_select_button(btn_rect))

            self._max_scroll_y = 0
            self._status_y = (cards_bottom_max + back_y) // 2

        self._btn_back = Button(
            self._back_base_rect.copy(),
            "",
            self._font_back,
            color=_BACK_BG,
            hover_color=_BACK_HOVER,
            text_color=_TEXT_PRIMARY,
            border_radius=max(10, round(12 * ui_s)),
            border_color=_BORDER_SUBTLE_RGB,
            border_width=1,
            accent=True,
            accent_color=_EMERALD_PRIMARY,
            press_feedback=True,
        )
        self._back_label_surf = self._font_back.render("BACK", True, _TEXT_PRIMARY)

    def _make_select_button(self, rect: pygame.Rect) -> Button:
        """Create a modern emerald-gradient SELECT button."""
        return Button(
            rect,
            "SELECT",
            self._font_button,
            color=_EMERALD_PRIMARY,
            hover_color=_EMERALD_HOVER_TOP,
            text_color=(255, 255, 255),
            border_radius=max(9, round(11 * self._ui_scale)),
            accent=True,
            gradient_colors=(_EMERALD_PRIMARY, _EMERALD_SECONDARY),
            hover_gradient_colors=(_EMERALD_HOVER_TOP, _EMERALD_HOVER_BOT),
            shadow_color=(0, 160, 120),
            border_color=(72, 238, 194),
            border_width=1,
            accent_color=_EMERALD_PRIMARY,
            lift_on_hover=True,
            press_feedback=True,
        )

    # ------------------------------------------------------------------
    # Cached Rendering & Asset Helpers
    # ------------------------------------------------------------------

    @classmethod
    def _load_avatar(cls, filename: str, size: int) -> pygame.Surface:
        """Load circular portrait preserving aspect ratio (object-fit: cover), cached."""
        cache_key = (filename, size)
        if cache_key in cls._avatar_cache:
            return cls._avatar_cache[cache_key]

        image = pygame.image.load(str(ASSETS_DIR / filename)).convert_alpha()
        width, height = image.get_size()
        scale = max(size / max(1, width), size / max(1, height))
        scaled_size = (max(1, round(width * scale)), max(1, round(height * scale)))
        image = pygame.transform.smoothscale(image, scaled_size)

        crop = pygame.Rect(0, 0, size, size)
        crop.center = image.get_rect().center
        avatar = image.subsurface(crop).copy()
        circle_mask = pygame.Surface((size, size), pygame.SRCALPHA)
        pygame.draw.circle(
            circle_mask, (255, 255, 255, 255), (size // 2, size // 2), size // 2
        )
        avatar.blit(circle_mask, (0, 0), special_flags=pygame.BLEND_RGBA_MULT)
        cls._avatar_cache[cache_key] = avatar
        return avatar

    @staticmethod
    def _make_radial_glow(
        radius: int, color: tuple[int, int, int], peak_alpha: int = 90
    ) -> pygame.Surface:
        """Create a smooth Gaussian-like radial glow surface using NumPy."""
        radius = max(4, radius)
        d = radius * 2
        yy, xx = np.mgrid[0:d, 0:d]
        dx = (xx - radius + 0.5) / radius
        dy = (yy - radius + 0.5) / radius
        dist2 = dx * dx + dy * dy
        falloff = np.where(dist2 < 1.0, np.exp(-3.2 * dist2) * (1.0 - dist2), 0.0)
        alpha = np.clip(falloff * peak_alpha, 0, 255).astype(np.uint8)

        surf = pygame.Surface((d, d), pygame.SRCALPHA)
        rgb_arr = pygame.surfarray.pixels3d(surf)
        rgb_arr[:, :, 0] = color[0]
        rgb_arr[:, :, 1] = color[1]
        rgb_arr[:, :, 2] = color[2]
        del rgb_arr
        alpha_arr = pygame.surfarray.pixels_alpha(surf)
        alpha_arr[:, :] = alpha.T
        del alpha_arr
        return surf

    def _load_knight_emblem(self, size: int) -> pygame.Surface:
        """Load the existing white knight SVG tinted in emerald as the header crest."""
        key = ("header_knight", size, _EMERALD_PRIMARY)
        if key in self._icon_cache:
            return self._icon_cache[key]

        svg_path = ASSETS_DIR / "wN.svg"
        if svg_path.is_file():
            raw = _load_svg_at_size(svg_path, size)
            tinted = pygame.Surface(raw.get_size(), pygame.SRCALPHA)
            src_a = pygame.surfarray.array_alpha(raw)
            rgb = pygame.surfarray.pixels3d(tinted)
            rgb[:, :, 0] = _EMERALD_PRIMARY[0]
            rgb[:, :, 1] = _EMERALD_PRIMARY[1]
            rgb[:, :, 2] = _EMERALD_PRIMARY[2]
            del rgb
            dst_a = pygame.surfarray.pixels_alpha(tinted)
            dst_a[:, :] = src_a
            del dst_a
            self._icon_cache[key] = tinted
            return tinted

        fallback = pygame.font.SysFont("Segoe UI Symbol", size).render(
            "♞", True, _EMERALD_PRIMARY
        )
        self._icon_cache[key] = fallback
        return fallback

    def _render_gradient_headline(self) -> pygame.Surface:
        """Render 'CHOOSE YOUR OPPONENT' with an emerald-to-cyan gradient on 'YOUR'."""
        left_s = self._font_title.render("CHOOSE ", True, _TEXT_PRIMARY)
        mid_raw = self._font_title.render("YOUR", True, (255, 255, 255))
        right_s = self._font_title.render(" OPPONENT", True, _TEXT_PRIMARY)

        # Apply smooth horizontal emerald gradient onto "YOUR"
        mw, mh = mid_raw.get_size()
        mid_grad = pygame.Surface((mw, mh), pygame.SRCALPHA)
        t = np.linspace(0.0, 1.0, max(1, mw), dtype=np.float32)
        c1 = np.array(_EMERALD_PRIMARY, dtype=np.float32)
        c2 = np.array((74, 240, 212), dtype=np.float32)
        grad_rgb = (c1[None, :] * (1.0 - t[:, None]) + c2[None, :] * t[:, None]).astype(
            np.uint8
        )
        rgb = pygame.surfarray.pixels3d(mid_grad)
        rgb[:, :, 0] = grad_rgb[:, 0:1]
        rgb[:, :, 1] = grad_rgb[:, 1:2]
        rgb[:, :, 2] = grad_rgb[:, 2:3]
        del rgb
        alpha = pygame.surfarray.pixels_alpha(mid_grad)
        alpha[:, :] = pygame.surfarray.array_alpha(mid_raw)
        del alpha

        total_w = left_s.get_width() + mw + right_s.get_width()
        total_h = max(left_s.get_height(), mh, right_s.get_height())
        combined = pygame.Surface((total_w, total_h), pygame.SRCALPHA)
        x = 0
        for part in (left_s, mid_grad, right_s):
            combined.blit(part, (x, (total_h - part.get_height()) // 2))
            x += part.get_width()

        # Fit safely inside screen width on very narrow mobile screens
        max_w = max(200, self.width - 28)
        if combined.get_width() > max_w:
            ratio = max_w / combined.get_width()
            combined = pygame.transform.smoothscale(
                combined, (max_w, max(1, round(total_h * ratio)))
            )
        return combined

    @classmethod
    def _make_archetype_badge(
        cls, kind: str, size: int, accent: tuple[int, int, int]
    ) -> pygame.Surface:
        """Create a crisp 2x-supersampled circular mini-emblem (spark / target / crown)."""
        key = (f"bot_emblem:{kind}", size, accent)
        if key in cls._icon_cache:
            return cls._icon_cache[key]

        hi = max(32, size * 2)
        surf = pygame.Surface((hi, hi), pygame.SRCALPHA)
        cx, cy = hi // 2, hi // 2
        r = hi // 2 - 2
        pygame.draw.circle(surf, (10, 18, 22, 235), (cx, cy), r)
        pygame.draw.circle(surf, accent, (cx, cy), r, width=3)

        if kind == "spark":
            # Crisp lightning bolt
            pts = (
                (cx + int(r * 0.08), cy - int(r * 0.56)),
                (cx - int(r * 0.34), cy + int(r * 0.06)),
                (cx + int(r * 0.02), cy + int(r * 0.06)),
                (cx - int(r * 0.12), cy + int(r * 0.56)),
                (cx + int(r * 0.36), cy - int(r * 0.08)),
                (cx - int(r * 0.02), cy - int(r * 0.08)),
            )
            pygame.draw.polygon(surf, accent, pts)
        elif kind == "target":
            # Crisp focus reticle
            pygame.draw.circle(surf, accent, (cx, cy), int(r * 0.52), width=3)
            pygame.draw.circle(surf, accent, (cx, cy), max(3, int(r * 0.18)))
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                pygame.draw.line(
                    surf,
                    accent,
                    (cx + dx * int(r * 0.42), cy + dy * int(r * 0.42)),
                    (cx + dx * int(r * 0.68), cy + dy * int(r * 0.68)),
                    3,
                )
        else:
            # Crisp crown
            w = int(r * 0.96)
            h = int(r * 0.68)
            left = cx - w // 2
            bot = cy + h // 2
            top = cy - h // 2
            pts = (
                (left, bot),
                (left + int(w * 0.08), top + int(h * 0.25)),
                (cx - int(w * 0.20), cy + int(h * 0.08)),
                (cx, top),
                (cx + int(w * 0.20), cy + int(h * 0.08)),
                (left + w - int(w * 0.08), top + int(h * 0.25)),
                (left + w, bot),
            )
            pygame.draw.polygon(surf, accent, pts)

        final = pygame.transform.smoothscale(surf, (size, size))
        cls._icon_cache[key] = final
        return final

    @classmethod
    def _make_back_arrow_icon(
        cls, size: int, color: tuple[int, int, int]
    ) -> pygame.Surface:
        """Create a crisp 2x-supersampled left arrow for the BACK button."""
        key = ("back_arrow", size, color)
        if key in cls._icon_cache:
            return cls._icon_cache[key]

        hi = max(24, size * 2)
        surf = pygame.Surface((hi, hi), pygame.SRCALPHA)
        cy = hi // 2
        stroke = max(3, hi // 10)
        left_x = int(hi * 0.22)
        right_x = int(hi * 0.80)
        head = int(hi * 0.26)
        pygame.draw.line(surf, color, (left_x + 2, cy), (right_x, cy), stroke)
        pygame.draw.lines(
            surf,
            color,
            False,
            ((left_x + head, cy - head), (left_x, cy), (left_x + head, cy + head)),
            stroke,
        )
        final = pygame.transform.smoothscale(surf, (size, size))
        cls._icon_cache[key] = final
        return final

    def _build_recommended_badge(self) -> pygame.Surface:
        """Pre-render the '★ RECOMMENDED' emerald pill badge above Focus Ken."""
        star_s = max(12, round(13 * self._ui_scale))
        hi = star_s * 2
        star = pygame.Surface((hi, hi), pygame.SRCALPHA)
        cx, cy = hi // 2, hi // 2
        pts = []
        for i in range(10):
            ang = math.radians(-90 + i * 36)
            rad = (hi * 0.44) if i % 2 == 0 else (hi * 0.20)
            pts.append((cx + math.cos(ang) * rad, cy + math.sin(ang) * rad))
        pygame.draw.polygon(star, _EMERALD_PRIMARY, pts)
        star_icon = pygame.transform.smoothscale(star, (star_s, star_s))

        txt = self._font_badge.render("RECOMMENDED", True, _EMERALD_PRIMARY)
        pad_x = max(10, round(12 * self._ui_scale))
        pad_y = max(4, round(5 * self._ui_scale))
        gap = max(5, round(6 * self._ui_scale))
        w = pad_x * 2 + star_s + gap + txt.get_width()
        h = max(star_s, txt.get_height()) + pad_y * 2

        pill = pygame.Surface((w, h), pygame.SRCALPHA)
        pygame.draw.rect(pill, (7, 28, 25, 240), (0, 0, w, h), border_radius=h // 2)
        pygame.draw.rect(
            pill, (*_EMERALD_PRIMARY, 195), (0, 0, w, h), width=1, border_radius=h // 2
        )
        pill.blit(star_icon, star_icon.get_rect(midleft=(pad_x, h // 2)))
        pill.blit(txt, txt.get_rect(midleft=(pad_x + star_s + gap, h // 2)))
        return pill

    def _build_card_caches(self) -> None:
        """Pre-render glassmorphism card surfaces, ELO pills, and wrapped text once."""
        # Compute uniform ELO pill width across all 3 cards
        pill_pad_x = max(14, round(18 * self._ui_scale))
        pill_pad_y = max(5, round(6 * self._ui_scale))
        elo_gap = max(6, round(8 * self._ui_scale))
        lbl_surf = self._font_elo_lbl.render("ELO", True, _TEXT_SECONDARY)
        num_surfs = [
            self._font_elo_num.render(str(bot["elo"]), True, bot["accent"])
            for bot in _BOTS
        ]
        pill_w = max(
            lbl_surf.get_width() + elo_gap + ns.get_width() + pill_pad_x * 2
            for ns in num_surfs
        )
        pill_h = max(lbl_surf.get_height(), max(ns.get_height() for ns in num_surfs)) + pill_pad_y * 2

        for idx, (bot, card, num_s) in enumerate(zip(_BOTS, self._cards, num_surfs)):
            accent = bot["accent"]
            # 1. Name surface (fitted cleanly to card width)
            max_txt_w = max(80, card.width - 28)
            name_surf = self._font_name.render(str(bot["name"]), True, _TEXT_PRIMARY)
            if name_surf.get_width() > max_txt_w:
                ratio = max_txt_w / name_surf.get_width()
                name_surf = pygame.transform.smoothscale(
                    name_surf, (max_txt_w, max(1, round(name_surf.get_height() * ratio)))
                )
            self._card_name_surfs.append(name_surf)

            # 2. Uniform ELO pill badge
            pill = pygame.Surface((pill_w, pill_h), pygame.SRCALPHA)
            pygame.draw.rect(pill, _ELO_PILL_BG, (0, 0, pill_w, pill_h), border_radius=pill_h // 2)
            pygame.draw.rect(
                pill, (*accent, 145), (0, 0, pill_w, pill_h), width=1, border_radius=pill_h // 2
            )
            content_w = lbl_surf.get_width() + elo_gap + num_s.get_width()
            sx = (pill_w - content_w) // 2
            pill.blit(lbl_surf, lbl_surf.get_rect(midleft=(sx, pill_h // 2)))
            pill.blit(num_s, num_s.get_rect(midleft=(sx + lbl_surf.get_width() + elo_gap, pill_h // 2)))
            self._card_elo_pills.append(pill)

            # 3. Wrapped description lines
            desc_text = str(bot["description"]).replace("\n", " ")
            self._card_desc_lines.append(
                self._wrap_text_lines(desc_text, self._font_description, _TEXT_SECONDARY, max_txt_w)
            )

            # 4. Card glass surface (subtle vertical glass sheen + top specular accent)
            w, h = card.size
            rad = self._card_radius
            glass = pygame.Surface((w, h), pygame.SRCALPHA)
            # Subtle vertical dark-glass gradient
            top_col = (24, 42, 50) if bot["recommended"] else (20, 34, 41)
            bot_col = (12, 22, 27)
            for y in range(h):
                t = y / max(1, h - 1)
                c = _lerp_rgb(top_col, bot_col, t)
                pygame.draw.line(glass, (*c, 228), (0, y), (w - 1, y))
            # Mask into rounded rect
            mask = pygame.Surface((w, h), pygame.SRCALPHA)
            pygame.draw.rect(mask, (255, 255, 255, 255), (0, 0, w, h), border_radius=rad)
            glass.blit(mask, (0, 0), special_flags=pygame.BLEND_RGBA_MULT)
            # Subtle top inner gloss line
            gloss_inset = rad + 2
            if w > gloss_inset * 2:
                gloss_col = (*accent, 110) if bot["recommended"] else (241, 245, 249, 32)
                pygame.draw.line(
                    glass, gloss_col, (gloss_inset, 1), (w - gloss_inset, 1), width=1
                )
            self._card_glass_surfs.append(glass)

            # 5. Soft shadow + hover accent glow surface
            pad = 20
            shadow = pygame.Surface((w + pad * 2, h + pad * 2), pygame.SRCALPHA)
            for spread, alpha in ((16, 16), (10, 28), (5, 48)):
                pygame.draw.rect(
                    shadow,
                    (0, 0, 0, alpha),
                    (pad - spread, pad - spread + 7, w + spread * 2, h + spread * 2),
                    border_radius=rad + spread,
                )
            self._card_shadows.append(shadow)

            glow = pygame.Surface((w + pad * 2, h + pad * 2), pygame.SRCALPHA)
            for spread, alpha in ((14, 14), (8, 26), (3, 44)):
                pygame.draw.rect(
                    glow,
                    (*accent, alpha),
                    (pad - spread, pad - spread, w + spread * 2, h + spread * 2),
                    border_radius=rad + spread,
                )
            self._card_hover_glows.append(glow)

    def _build_background(self) -> pygame.Surface:
        """Build the atmospheric dark-emerald chess backdrop once using NumPy."""
        w, h = self.width, self.height
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        nx = xx / max(1.0, float(w))
        ny = yy / max(1.0, float(h))

        # 1. Deep charcoal vertical gradient (#050B0D -> #0B1418 -> #050B0D)
        mid_curve = np.sin(ny * math.pi)
        r = _BG_MAIN[0] + (_BG_SECONDARY[0] - _BG_MAIN[0]) * mid_curve
        g = _BG_MAIN[1] + (_BG_SECONDARY[1] - _BG_MAIN[1]) * mid_curve
        b = _BG_MAIN[2] + (_BG_SECONDARY[2] - _BG_MAIN[2]) * mid_curve

        # 2. Subtle emerald radial glow behind the opponent cards
        dx_c = (nx - 0.50) * 1.45
        dy_c = (ny - 0.54) * 1.75
        radial = np.exp(-3.4 * (dx_c * dx_c + dy_c * dy_c))
        r += _EMERALD_PRIMARY[0] * 0.085 * radial
        g += _EMERALD_PRIMARY[1] * 0.085 * radial
        b += _EMERALD_PRIMARY[2] * 0.085 * radial

        # 3. Subtle cool atmospheric edge light at top center
        dx_t = (nx - 0.50) * 2.0
        dy_t = (ny - 0.08) * 3.0
        top_glow = np.exp(-3.0 * (dx_t * dx_t + dy_t * dy_t))
        r += _EMERALD_PRIMARY[0] * 0.045 * top_glow
        g += _EMERALD_PRIMARY[1] * 0.045 * top_glow
        b += _EMERALD_PRIMARY[2] * 0.045 * top_glow

        # 4. Very faint chessboard pattern (softly masked to center/lower region)
        sq_px = max(36.0, round(58.0 * max(0.75, self._scale)))
        parity = (((xx // sq_px).astype(np.int32) + (yy // sq_px).astype(np.int32)) & 1).astype(np.float32)
        board_mask = np.clip(1.0 - 1.55 * ((nx - 0.5) ** 2 + (ny - 0.58) ** 2), 0.0, 1.0)
        checker = parity * board_mask * 6.5
        r += checker * 0.45
        g += checker * 1.10
        b += checker * 0.90

        # 5. Soft edge vignette
        edge_d = (nx - 0.5) ** 2 + (ny - 0.5) ** 2
        vignette = np.clip(1.0 - 0.55 * edge_d, 0.65, 1.0)
        rgb = np.stack([r * vignette, g * vignette, b * vignette], axis=2)
        rgb_u8 = np.clip(rgb, 0, 255).astype(np.uint8)

        bg = pygame.Surface((w, h))
        arr = pygame.surfarray.pixels3d(bg)
        arr[:, :, :] = np.transpose(rgb_u8, (1, 0, 2))
        del arr

        # 6. Low-opacity chess piece silhouettes in the left & right atmospheric wings
        sil_size = max(120, min(280, round(240 * max(0.65, self._scale))))
        for svg_name, pos_frac in (("wN.svg", (0.09, 0.72)), ("wQ.svg", (0.91, 0.70))):
            svg_path = ASSETS_DIR / svg_name
            if svg_path.is_file():
                try:
                    raw = _load_svg_at_size(svg_path, sil_size)
                    sil = pygame.Surface(raw.get_size(), pygame.SRCALPHA)
                    src_a = pygame.surfarray.array_alpha(raw)
                    s_rgb = pygame.surfarray.pixels3d(sil)
                    s_rgb[:, :, 0] = _EMERALD_PRIMARY[0]
                    s_rgb[:, :, 1] = _EMERALD_PRIMARY[1]
                    s_rgb[:, :, 2] = _EMERALD_PRIMARY[2]
                    del s_rgb
                    s_a = pygame.surfarray.pixels_alpha(sil)
                    s_a[:, :] = (src_a.astype(np.float32) * 0.06).astype(np.uint8)
                    del s_a
                    bg.blit(sil, sil.get_rect(center=(int(w * pos_frac[0]), int(h * pos_frac[1]))))
                except Exception:
                    pass

        if pygame.display.get_surface() is not None:
            try:
                return bg.convert()
            except pygame.error:
                return bg
        return bg

    @staticmethod
    def _wrap_text_lines(
        text: str,
        font: pygame.font.Font,
        color: tuple[int, int, int],
        max_width: int,
    ) -> list[pygame.Surface]:
        words = text.split()
        lines: list[str] = []
        current = ""
        for word in words:
            cand = f"{current} {word}".strip()
            if current and font.size(cand)[0] > max_width:
                lines.append(current)
                current = word
            else:
                current = cand
        if current:
            lines.append(current)
        return [font.render(line, True, color) for line in lines]

    def _card_draw_rect(self, index: int) -> pygame.Rect:
        """Return the active screen rect for card *index* including hover lift & scroll."""
        lift = round(-8.0 * self._card_hover_t[index] * self._ui_scale)
        return self._cards[index].move(0, lift - self._scroll_y)

    # ------------------------------------------------------------------
    # BaseScreen Interface
    # ------------------------------------------------------------------

    def handle_events(self, events: list[pygame.event.Event]) -> None:
        for event in events:
            if self._max_scroll_y > 0:
                if event.type == pygame.MOUSEWHEEL:
                    self._scroll_y = max(
                        0, min(self._max_scroll_y, self._scroll_y - event.y * 36)
                    )
                    self._sync_scrolled_button_rects()
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button in (4, 5):
                    delta = -36 if event.button == 4 else 36
                    self._scroll_y = max(
                        0, min(self._max_scroll_y, self._scroll_y + delta)
                    )
                    self._sync_scrolled_button_rects()

            if event.type == pygame.MOUSEMOTION:
                self._hovered_card = next(
                    (
                        index
                        for index in range(len(self._cards))
                        if self._card_draw_rect(index).collidepoint(event.pos)
                    ),
                    None,
                )

            if self._btn_back.handle_event(event):
                self.next_screen = "dashboard"
                continue

            for index, button in enumerate(self._buttons):
                if button.handle_event(event):
                    self.selected_opponent = _BOTS[index]
                    self._buttons[index].text = "SELECTED"
                    self.next_screen = "game"
                    self._status_surface = self._font_status.render(
                        f"{self.selected_opponent['name']} selected",
                        True,
                        _EMERALD_PRIMARY,
                    )

    def _sync_scrolled_button_rects(self) -> None:
        for idx, button in enumerate(self._buttons):
            lift = round(-8.0 * self._card_hover_t[idx] * self._ui_scale)
            button.base_rect = self._button_base_rects[idx].move(0, lift - self._scroll_y)
        self._btn_back.base_rect = self._back_base_rect.move(0, -self._scroll_y)

    def update(self, dt: float) -> None:
        clamped_dt = min(max(0.0, dt), MAX_ANIMATION_DT)
        self._time += clamped_dt

        # Smooth card hover micro-interactions (~220ms)
        step = min(_HOVER_ANIM_SPEED * clamped_dt, 1.0)
        for idx, bot in enumerate(_BOTS):
            target = (
                1.0
                if (idx == self._hovered_card or bot is self.selected_opponent)
                else 0.0
            )
            self._card_hover_t[idx] += (target - self._card_hover_t[idx]) * step
        self._sync_scrolled_button_rects()

        # Subtle floating ambient light motes
        for p in self._particles:
            p["x"] += p["vx"] * clamped_dt
            p["y"] += p["vy"] * clamped_dt
            if p["y"] < -12:
                p["y"] = self.height + 10
            if p["x"] < -12:
                p["x"] = self.width + 10
            elif p["x"] > self.width + 12:
                p["x"] = -10

        self._btn_back.update(clamped_dt)
        for button in self._buttons:
            button.update(clamped_dt)

    def draw(self) -> None:
        # 1. Atmospheric dark-emerald chess background + faint drifting particles
        self.screen.blit(self._bg_surface, (0, 0))
        for p in self._particles:
            pulse = 0.55 + 0.45 * math.sin(self._time * 1.4 + p["phase"])
            self._particle_surf.set_alpha(int(255 * p["alpha"] * pulse))
            self.screen.blit(
                self._particle_surf,
                self._particle_surf.get_rect(center=(round(p["x"]), round(p["y"]))),
            )

        sy = self._scroll_y

        # 2. Header section (Knight crest + tapered emerald lines + gradient title + subtitle)
        cx = self.width // 2
        knight_y = self._header_knight_y - sy
        self.screen.blit(
            self._knight_emblem,
            self._knight_emblem.get_rect(center=(cx, knight_y)),
        )
        line_len = max(46, min(140, int(self.width * 0.11)))
        line_gap = self._knight_emblem.get_width() // 2 + max(10, round(14 * self._ui_scale))
        for sign in (-1, 1):
            x0 = cx + sign * line_gap
            x1 = x0 + sign * line_len
            pygame.draw.line(self.screen, _EMERALD_PRIMARY, (x0, knight_y), (x0 + sign * int(line_len * 0.55), knight_y), 2)
            pygame.draw.line(self.screen, _EMERALD_SECONDARY, (x0 + sign * int(line_len * 0.55), knight_y), (x1, knight_y), 1)

        self.screen.blit(
            self._header_title_surf,
            self._header_title_surf.get_rect(center=(cx, self._header_title_y - sy)),
        )
        self.screen.blit(
            self._subtitle_surf,
            self._subtitle_surf.get_rect(center=(cx, self._header_sub_y - sy)),
        )

        # 3. Three Opponent Glassmorphism Cards
        for idx, (bot, button) in enumerate(zip(_BOTS, self._buttons)):
            card = self._card_draw_rect(idx)
            ht = self._card_hover_t[idx]
            accent: tuple[int, int, int] = bot["accent"]
            is_rec = bool(bot["recommended"])

            # Soft elevation shadow
            shadow = self._card_shadows[idx]
            self.screen.blit(shadow, shadow.get_rect(center=card.center))

            # Subtle accent glow (always softly present on Focus Ken; blooms on hover for all)
            glow_strength = max(0.32 if is_rec else 0.0, ht * 0.92)
            if glow_strength > 0.02:
                glow = self._card_hover_glows[idx]
                glow.set_alpha(int(255 * glow_strength))
                self.screen.blit(glow, glow.get_rect(center=card.center))

            # Glass card fill
            self.screen.blit(self._card_glass_surfs[idx], card.topleft)

            # Hover interior brightening overlay
            if ht > 0.01:
                tint_col = _lerp_rgb(_CARD_BG, _CARD_BG_HOVER, ht)
                pygame.draw.rect(
                    self.screen,
                    tint_col,
                    card.inflate(-4, -4),
                    width=2,
                    border_radius=max(8, self._card_radius - 2),
                )

            # Refined card border
            rest_border = _BORDER_RECOMMENDED_RGB if is_rec else _BORDER_SUBTLE_RGB
            border_col = _lerp_rgb(rest_border, accent, ht)
            border_w = 2 if (is_rec or ht > 0.25) else 1
            pygame.draw.rect(
                self.screen,
                border_col,
                card,
                width=border_w,
                border_radius=self._card_radius,
            )

            # "★ RECOMMENDED" floating badge above Focus Ken
            if is_rec:
                self.screen.blit(
                    self._recommended_badge_surf,
                    self._recommended_badge_surf.get_rect(center=(card.centerx, card.top)),
                )

            # Circular portrait + ambient glow + crisp colored ring + archetype emblem
            av_cy = card.top + max(
                self._avatar_size // 2 + (24 if self._layout_mode == "mobile" else 18),
                int(card.height * (0.28 if self._layout_mode != "mobile" else 0.21)),
            )
            av_center = (card.centerx, av_cy)

            halo = self._avatar_halos[idx]
            halo.set_alpha(int(255 * (0.62 + 0.38 * ht)))
            self.screen.blit(halo, halo.get_rect(center=av_center))

            avatar = self._avatars[idx]
            self.screen.blit(avatar, avatar.get_rect(center=av_center))

            ring_base = _lerp_rgb(accent, (255, 255, 255), 0.22 * ht)
            ring_w = max(2, round((3.0 + 0.8 * ht) * self._ui_scale))
            pygame.draw.circle(
                self.screen,
                ring_base,
                av_center,
                self._avatar_size // 2 + 1,
                width=ring_w,
            )

            # Small themed emblem badge anchored at bottom-right of portrait ring
            emblem = self._bot_emblems[idx]
            emb_angle = math.radians(40)
            emb_r = self._avatar_size // 2 - 2
            emb_pos = (
                round(av_center[0] + math.cos(emb_angle) * emb_r),
                round(av_center[1] + math.sin(emb_angle) * emb_r),
            )
            self.screen.blit(emblem, emblem.get_rect(center=emb_pos))

            # Opponent Name
            name_surf = self._card_name_surfs[idx]
            name_y = av_cy + self._avatar_size // 2 + max(18, round(24 * self._ui_scale))
            self.screen.blit(name_surf, name_surf.get_rect(center=(card.centerx, name_y)))

            # Pill ELO Badge
            elo_pill = self._card_elo_pills[idx]
            elo_y = name_y + name_surf.get_height() // 2 + elo_pill.get_height() // 2 + max(7, round(10 * self._ui_scale))
            self.screen.blit(elo_pill, elo_pill.get_rect(center=(card.centerx, elo_y)))

            # Description lines (centered cleanly between ELO pill and SELECT button)
            desc_lines = self._card_desc_lines[idx]
            line_h = self._font_description.get_linesize()
            desc_block_h = len(desc_lines) * line_h
            desc_area_top = elo_y + elo_pill.get_height() // 2 + max(6, round(8 * self._ui_scale))
            desc_area_bot = button.base_rect.top - max(6, round(8 * self._ui_scale))
            desc_start_y = desc_area_top + max(0, (desc_area_bot - desc_area_top - desc_block_h) // 2)
            for l_idx, line_surf in enumerate(desc_lines):
                self.screen.blit(
                    line_surf,
                    line_surf.get_rect(
                        midtop=(card.centerx, desc_start_y + l_idx * line_h)
                    ),
                )

            button.text = "SELECTED" if bot is self.selected_opponent else "SELECT"
            button.draw(self.screen)

        # 4. Status notice (if set) + Glass BACK button with left arrow
        if self._status_surface is not None:
            self.screen.blit(
                self._status_surface,
                self._status_surface.get_rect(center=(cx, self._status_y - sy)),
            )

        self._btn_back.draw(self.screen)
        b_rect = self._btn_back.base_rect
        gap = max(7, round(9 * self._ui_scale))
        total_w = self._back_arrow_surf.get_width() + gap + self._back_label_surf.get_width()
        sx = b_rect.centerx - total_w // 2
        press_y = round(self._btn_back._pressed_t)
        self.screen.blit(
            self._back_arrow_surf,
            self._back_arrow_surf.get_rect(midleft=(sx, b_rect.centery + press_y)),
        )
        self.screen.blit(
            self._back_label_surf,
            self._back_label_surf.get_rect(
                midleft=(sx + self._back_arrow_surf.get_width() + gap, b_rect.centery + press_y)
            ),
        )
