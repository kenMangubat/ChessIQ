"""Visual verification test for Neo chess pieces.

Renders all 12 pieces onto alternating green and cream squares,
verifying contrast, centering, proportions, and alpha transparency.
"""

import os
import pygame

ASSETS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "assets"))
OUTPUT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "pieces_preview.png"))

DARK_SQ = (134, 166, 102)   # #86a666 (Lichess green)
LIGHT_SQ = (255, 255, 221)  # #ffffdd (Lichess cream)


def test_visual():
    pygame.init()

    sq_size = 120
    cols = 6
    rows = 4  # Row 0: White on Dark, Row 1: White on Light, Row 2: Black on Dark, Row 3: Black on Light

    width = cols * sq_size
    height = rows * sq_size

    surface = pygame.Surface((width, height))

    # Types: Pawn, Knight, Bishop, Rook, Queen, King
    piece_types = ["P", "N", "B", "R", "Q", "K"]

    for r in range(rows):
        is_white_piece = (r < 2)
        base_color_is_dark = (r % 2 == 0)

        for c, ptype in enumerate(piece_types):
            sq_color = DARK_SQ if base_color_is_dark else LIGHT_SQ
            # Checker the squares slightly across columns
            if c % 2 == 1:
                sq_color = LIGHT_SQ if sq_color == DARK_SQ else DARK_SQ

            x = c * sq_size
            y = r * sq_size
            rect = pygame.Rect(x, y, sq_size, sq_size)
            pygame.draw.rect(surface, sq_color, rect)

            prefix = "w" if is_white_piece else "b"
            filename = f"{prefix}{ptype}.svg"
            svg_path = os.path.join(ASSETS_DIR, filename)

            raw_surf = pygame.image.load(svg_path)
            # Scale piece to ~88% of square size for perfect margins
            piece_target_size = int(sq_size * 0.88)
            scaled = pygame.transform.smoothscale(raw_surf, (piece_target_size, piece_target_size))

            piece_rect = scaled.get_rect(center=rect.center)
            surface.blit(scaled, piece_rect)

    pygame.image.save(surface, OUTPUT_PATH)
    print(f"Saved visual verification preview to: {OUTPUT_PATH}")
    print(f"Dimensions: {width}x{height}")


if __name__ == "__main__":
    test_visual()
