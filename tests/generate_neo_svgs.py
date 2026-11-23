"""Generator for Neo-style ChessIQ SVG pieces.

Produces 12 vector SVG pieces (wK, wQ, wR, wB, wN, wP, bK, bQ, bR, bB, bN, bP)
inspired by the modern Chess.com Neo style with smooth curves,
balanced proportions, and clean vector contours.
"""

import os
import pygame

ASSETS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "assets"))


def make_svg(piece_type: str, is_white: bool) -> str:
    """Return the SVG content for a given piece ('P', 'N', 'B', 'R', 'Q', 'K')."""
    if is_white:
        body_start = "#FFFFFF"
        body_end = "#EFECE6"
        shade_start = "#E6E2D8"
        shade_end = "#D1CCC0"
        outline = "#25272C"
        hi_stroke = "#FFFFFF"
        hi_opacity = "0.8"
        accent_fill = "#DDD9CE"
        eye_color = "#25272C"
        notch_color = "#D1CCC0"
    else:
        body_start = "#444750"
        body_end = "#2A2C33"
        shade_start = "#26282E"
        shade_end = "#1D1E22"
        outline = "#15161A"
        hi_stroke = "#868D9E"
        hi_opacity = "0.7"
        accent_fill = "#23252B"
        eye_color = "#868D9E"
        notch_color = "#1D1E22"

    grad_id = f"g_{piece_type}_{'w' if is_white else 'b'}"
    shade_id = f"s_{piece_type}_{'w' if is_white else 'b'}"

    defs = f"""  <defs>
    <linearGradient id="{grad_id}" x1="0.15" y1="0.0" x2="0.85" y2="1.0">
      <stop offset="0%" stop-color="{body_start}"/>
      <stop offset="100%" stop-color="{body_end}"/>
    </linearGradient>
    <linearGradient id="{shade_id}" x1="0.0" y1="0.0" x2="0.0" y2="1.0">
      <stop offset="0%" stop-color="{shade_start}"/>
      <stop offset="100%" stop-color="{shade_end}"/>
    </linearGradient>
  </defs>"""

    def base_group(width_scale: float = 1.0) -> str:
        half_w = 33.0 * width_scale
        x1 = 50.0 - half_w
        x2 = 50.0 + half_w
        torus_w = half_w - 4.5
        tx1 = 50.0 - torus_w
        tx2 = 50.0 + torus_w

        return f"""
    <!-- Base Plinth -->
    <path d="M {x1:.1f} 89.5 C {x1:.1f} 87.0 {x1+3:.1f} 85.0 {x1+6:.1f} 85.0 L {x2-6:.1f} 85.0 C {x2-3:.1f} 85.0 {x2:.1f} 87.0 {x2:.1f} 89.5 Z"
          fill="url(#{shade_id})" stroke="{outline}" stroke-width="3.5" stroke-linejoin="round"/>
    <!-- Base Torus -->
    <path d="M {tx1:.1f} 85.0 C {tx1:.1f} 80.5 {tx1+4:.1f} 78.5 {tx1+7:.1f} 78.5 L {tx2-7:.1f} 78.5 C {tx2-4:.1f} 78.5 {tx2:.1f} 80.5 {tx2:.1f} 85.0 Z"
          fill="url(#{grad_id})" stroke="{outline}" stroke-width="3.2" stroke-linejoin="round"/>"""

    # ─────────────────────────────────────────────────────────────────
    # PAWN
    # ─────────────────────────────────────────────────────────────────
    if piece_type == "P":
        content = f"""{defs}
  <g id="pawn">
    {base_group(0.82)}
    <!-- Body -->
    <path d="M 28 78.5 C 33 65 39 56 42 50 L 58 50 C 61 56 67 65 72 78.5 Z"
          fill="url(#{grad_id})" stroke="{outline}" stroke-width="3.5" stroke-linejoin="round"/>
    <!-- Neck Ring -->
    <path d="M 38 50 C 38 47 43 45.0 50 45.0 C 57 45.0 62 47 62 50 C 62 53 57 55.0 50 55.0 C 43 55.0 38 53 38 50 Z"
          fill="url(#{shade_id})" stroke="{outline}" stroke-width="3.0"/>
    <!-- Head Ball -->
    <circle cx="50" cy="32" r="14.5"
            fill="url(#{grad_id})" stroke="{outline}" stroke-width="3.5"/>
    <!-- Head Shading Arc -->
    <path d="M 40 23 C 44 20.5 54 20.5 60 25"
          fill="none" stroke="{hi_stroke}" stroke-width="2.4" stroke-linecap="round" opacity="{hi_opacity}"/>
    <!-- Body Highlight -->
    <path d="M 45 56 C 43 63 40 70 36 76"
          fill="none" stroke="{hi_stroke}" stroke-width="2.2" stroke-linecap="round" opacity="{hi_opacity}"/>
  </g>"""

    # ─────────────────────────────────────────────────────────────────
    # KNIGHT (Neo style faces left, proud chest, arched mane)
    # ─────────────────────────────────────────────────────────────────
    elif piece_type == "N":
        content = f"""{defs}
  <g id="knight">
    {base_group(0.92)}
    <!-- Knight Body & Head (Facing Left like Chess.com Neo) -->
    <path d="M 72 78.5
             C 72 68 70 56 66 46
             C 63 38 59 30 54 24
             C 52 21 52 17 56 16
             C 53 14 47 16 44 20
             C 41 24 37 28 32 31
             C 27 34 22 39 23 44
             C 24 49 29 48 35 46
             C 40 45 42 47 38 54
             C 34 61 32 68 31 78.5
             Z"
          fill="url(#{grad_id})" stroke="{outline}" stroke-width="3.5" stroke-linejoin="round"/>
    <!-- Mane Arc Line / Depth Notch -->
    <path d="M 52 28 C 58 35 60 46 62 58"
          fill="none" stroke="{outline}" stroke-width="2.6" stroke-linecap="round"/>
    <!-- Jaw / Throat Contour -->
    <path d="M 35 46 C 42 47 46 54 44 62"
          fill="none" stroke="{outline}" stroke-width="2.4" stroke-linecap="round"/>
    <!-- Eye (Sleek tilted almond) -->
    <ellipse cx="34" cy="36" rx="2.2" ry="3.2" transform="rotate(-15 34 36)" fill="{eye_color}" stroke="{outline}" stroke-width="0.8"/>
    <!-- Nostril curve -->
    <path d="M 25 43 C 27 44 29 43 29 41" fill="none" stroke="{outline}" stroke-width="2.2" stroke-linecap="round"/>
    <!-- Mane Highlight -->
    <path d="M 53 20 C 60 26 63 36 65 48"
          fill="none" stroke="{hi_stroke}" stroke-width="2.4" stroke-linecap="round" opacity="{hi_opacity}"/>
    <!-- Snout Highlight -->
    <path d="M 42 24 C 36 29 30 33 26 38"
          fill="none" stroke="{hi_stroke}" stroke-width="2.0" stroke-linecap="round" opacity="{hi_opacity}"/>
  </g>"""

    # ─────────────────────────────────────────────────────────────────
    # BISHOP (Smooth miter dome, signature notch, finial ball)
    # ─────────────────────────────────────────────────────────────────
    elif piece_type == "B":
        content = f"""{defs}
  <g id="bishop">
    {base_group(0.90)}
    <!-- Lower Body -->
    <path d="M 28 78.5 C 33 66 38 60 42 55 L 58 55 C 62 60 67 66 72 78.5 Z"
          fill="url(#{grad_id})" stroke="{outline}" stroke-width="3.5" stroke-linejoin="round"/>
    <!-- Collar Ring -->
    <path d="M 37 55 C 37 52.5 43 50.5 50 50.5 C 57 50.5 63 52.5 63 55 C 63 57.5 57 59.5 50 59.5 C 43 59.5 37 57.5 37 55 Z"
          fill="url(#{shade_id})" stroke="{outline}" stroke-width="3.0"/>
    <!-- Miter Head Dome -->
    <path d="M 32 45
             C 32 32 40 21 50 21
             C 60 21 68 32 68 45
             C 68 51 61 54 50 54
             C 39 54 32 51 32 45 Z"
          fill="url(#{grad_id})" stroke="{outline}" stroke-width="3.5" stroke-linejoin="round"/>
    <!-- Bishop Slit / Cut (Cutout notch) -->
    <path d="M 54 26 L 63 37"
          fill="none" stroke="{outline}" stroke-width="4.0" stroke-linecap="round"/>
    <path d="M 54 26 L 63 37"
          fill="none" stroke="{notch_color}" stroke-width="1.8" stroke-linecap="round"/>
    <!-- Finial Ball on Top -->
    <circle cx="50" cy="18" r="4.2"
            fill="url(#{shade_id})" stroke="{outline}" stroke-width="2.8"/>
    <!-- Miter Left Highlight -->
    <path d="M 37 38 C 37 30 42 25 47 23"
          fill="none" stroke="{hi_stroke}" stroke-width="2.4" stroke-linecap="round" opacity="{hi_opacity}"/>
    <!-- Body Highlight -->
    <path d="M 45 60 C 42 66 39 71 36 76"
          fill="none" stroke="{hi_stroke}" stroke-width="2.0" stroke-linecap="round" opacity="{hi_opacity}"/>
  </g>"""

    # ─────────────────────────────────────────────────────────────────
    # ROOK (Sturdy castle, 3 crenel notches, clean cornice)
    # ─────────────────────────────────────────────────────────────────
    elif piece_type == "R":
        content = f"""{defs}
  <g id="rook">
    {base_group(0.94)}
    <!-- Tower Body -->
    <path d="M 29 78.5 C 33 66 35 56 35 45 L 65 45 C 65 56 67 66 71 78.5 Z"
          fill="url(#{grad_id})" stroke="{outline}" stroke-width="3.5" stroke-linejoin="round"/>
    <!-- Cornice Band -->
    <path d="M 29 45 L 71 45 L 73 39 L 27 39 Z"
          fill="url(#{shade_id})" stroke="{outline}" stroke-width="3.2" stroke-linejoin="round"/>
    <!-- Battlements: 3 Wide Merlons, 2 Notches -->
    <path d="M 27 39
             L 27 24
             L 37 24
             L 37 31
             L 45 31
             L 45 24
             L 55 24
             L 55 31
             L 63 31
             L 63 24
             L 73 24
             L 73 39
             Z"
          fill="url(#{grad_id})" stroke="{outline}" stroke-width="3.5" stroke-linejoin="round"/>
    <!-- Cornice Highlight Line -->
    <path d="M 30 41.5 L 70 41.5"
          fill="none" stroke="{hi_stroke}" stroke-width="2.0" stroke-linecap="round" opacity="{hi_opacity}"/>
    <!-- Tower Body Highlight -->
    <path d="M 40 49 C 39 58 37 67 35 76"
          fill="none" stroke="{hi_stroke}" stroke-width="2.0" stroke-linecap="round" opacity="{hi_opacity}"/>
  </g>"""

    # ─────────────────────────────────────────────────────────────────
    # QUEEN (Coronet with 5 rounded pearl points, elegant waist)
    # ─────────────────────────────────────────────────────────────────
    elif piece_type == "Q":
        content = f"""{defs}
  <g id="queen">
    {base_group(0.96)}
    <!-- Lower Gown -->
    <path d="M 28 78.5 C 33 66 38 58 42 50 L 58 50 C 62 58 67 66 72 78.5 Z"
          fill="url(#{grad_id})" stroke="{outline}" stroke-width="3.5" stroke-linejoin="round"/>
    <!-- Waist Collar -->
    <path d="M 37 50 C 37 47.5 43 45.5 50 45.5 C 57 45.5 63 47.5 63 50 C 63 52.5 57 54.5 50 54.5 C 43 54.5 37 52.5 37 50 Z"
          fill="url(#{shade_id})" stroke="{outline}" stroke-width="3.0"/>
    <!-- Coronet / Crown Body -->
    <path d="M 30 47
             L 23 30
             L 36 38
             L 50 25
             L 64 38
             L 77 30
             L 70 47
             Z"
          fill="url(#{grad_id})" stroke="{outline}" stroke-width="3.5" stroke-linejoin="round"/>
    <!-- 5 Jewels on Crown Points -->
    <circle cx="23" cy="28" r="3.2" fill="url(#{shade_id})" stroke="{outline}" stroke-width="2.5"/>
    <circle cx="36.5" cy="36" r="3.2" fill="url(#{shade_id})" stroke="{outline}" stroke-width="2.5"/>
    <circle cx="50" cy="23" r="3.8" fill="url(#{shade_id})" stroke="{outline}" stroke-width="2.5"/>
    <circle cx="63.5" cy="36" r="3.2" fill="url(#{shade_id})" stroke="{outline}" stroke-width="2.5"/>
    <circle cx="77" cy="28" r="3.2" fill="url(#{shade_id})" stroke="{outline}" stroke-width="2.5"/>
    <!-- Crown Arch Highlight -->
    <path d="M 34 44.5 C 44 46.5 56 46.5 66 44.5"
          fill="none" stroke="{hi_stroke}" stroke-width="2.0" stroke-linecap="round" opacity="{hi_opacity}"/>
    <!-- Gown Highlight -->
    <path d="M 45 54 C 42 62 39 70 36 76"
          fill="none" stroke="{hi_stroke}" stroke-width="2.0" stroke-linecap="round" opacity="{hi_opacity}"/>
  </g>"""

    # ─────────────────────────────────────────────────────────────────
    # KING (Majestic arched crown, rounded cross finial)
    # ─────────────────────────────────────────────────────────────────
    elif piece_type == "K":
        content = f"""{defs}
  <g id="king">
    {base_group(1.0)}
    <!-- Lower Body / Robe -->
    <path d="M 27 78.5 C 32 66 37 58 41 48 L 59 48 C 63 58 68 66 73 78.5 Z"
          fill="url(#{grad_id})" stroke="{outline}" stroke-width="3.5" stroke-linejoin="round"/>
    <!-- Waist Collar -->
    <path d="M 37 48 C 37 45.5 43 43.5 50 43.5 C 57 43.5 63 45.5 63 48 C 63 50.5 57 52.5 50 52.5 C 43 52.5 37 50.5 37 48 Z"
          fill="url(#{shade_id})" stroke="{outline}" stroke-width="3.0"/>
    <!-- Crown Pavilion Dome -->
    <path d="M 31 44
             C 27 34 32 25 50 25
             C 68 25 73 34 69 44
             Z"
          fill="url(#{grad_id})" stroke="{outline}" stroke-width="3.5" stroke-linejoin="round"/>
    <!-- Crown Inner Arches -->
    <path d="M 50 25 L 50 44"
          fill="none" stroke="{outline}" stroke-width="2.6" stroke-linecap="round"/>
    <path d="M 39 29 C 44 37 45 42 45 44"
          fill="none" stroke="{outline}" stroke-width="2.2" stroke-linecap="round"/>
    <path d="M 61 29 C 56 37 55 42 55 44"
          fill="none" stroke="{outline}" stroke-width="2.2" stroke-linecap="round"/>
    <!-- Rounded Latin Cross Finial -->
    <path d="M 50 11 L 50 25"
          stroke="{outline}" stroke-width="4.2" stroke-linecap="round"/>
    <path d="M 43.5 16.5 L 56.5 16.5"
          stroke="{outline}" stroke-width="4.2" stroke-linecap="round"/>
    <path d="M 50 12 L 50 24 M 44.5 16.5 L 55.5 16.5"
          stroke="{hi_stroke}" stroke-width="2.0" stroke-linecap="round" opacity="{hi_opacity}"/>
    <!-- Crown Dome Highlight -->
    <path d="M 34 34 C 36 28 42 26 48 26"
          fill="none" stroke="{hi_stroke}" stroke-width="2.0" stroke-linecap="round" opacity="{hi_opacity}"/>
    <!-- Body Highlight -->
    <path d="M 45 53 C 42 61 39 70 36 76"
          fill="none" stroke="{hi_stroke}" stroke-width="2.0" stroke-linecap="round" opacity="{hi_opacity}"/>
  </g>"""

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100" width="160" height="160">
{content}
</svg>
"""
    return svg


def main():
    import sys
    if "--force" not in sys.argv and os.environ.get("GENERATE_NEO_SVGS") != "1":
        print("Skipping SVG generation to protect canonical piece assets. Use --force to regenerate.")
        return

    pieces = [
        ("wK", "K", True), ("wQ", "Q", True), ("wR", "R", True),
        ("wB", "B", True), ("wN", "N", True), ("wP", "P", True),
        ("bK", "K", False), ("bQ", "Q", False), ("bR", "R", False),
        ("bB", "B", False), ("bN", "N", False), ("bP", "P", False),
    ]

    os.makedirs(ASSETS_DIR, exist_ok=True)

    print(f"Generating 12 Neo-style SVG pieces into: {ASSETS_DIR}")
    for filename, ptype, is_white in pieces:
        filepath = os.path.join(ASSETS_DIR, f"{filename}.svg")
        svg_content = make_svg(ptype, is_white)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(svg_content)
        print(f"  [+] Updated {filename}.svg ({len(svg_content)} bytes)")


if __name__ == "__main__":
    main()
