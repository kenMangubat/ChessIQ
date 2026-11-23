"""Build-time icon generator for ChessIQ / "Chess Ken".

Reads the source logo configured in ``others/config.py`` (without modifying the
original artwork), centers it cleanly on a square transparent canvas, writes a
1024x1024 master PNG, a multi-size Windows ``.ico`` (16, 24, 32, 48, 64, 128,
256 px), and a dark/light size-comparison preview sheet.

Uses Pillow (``Image.Resampling.LANCZOS``) when available, and falls back to an
exact premultiplied-alpha NumPy Lanczos-3 resampler + PNG-in-ICO writer so no
extra runtime dependencies are required.

Usage (from repository root):
    python -m others.make_icon

Future PyInstaller packaging:
    pyinstaller --windowed --name "Chess Ken" --icon=assets/icons/chessken.ico main.py
"""

from __future__ import annotations

from collections import deque
import io
import math
from pathlib import Path
import struct
import sys

import numpy as np
import pygame

from others.config import (
    ICON_ICO_PATH,
    ICON_MASTER_PATH,
    ICON_PREVIEW_PATH,
    ICONS_DIR,
    LOGO_SOURCE_PATH,
)

ICO_SIZES: tuple[int, ...] = (16, 24, 32, 48, 64, 128, 256)
PREVIEW_SIZES: tuple[int, ...] = (256, 128, 64, 48, 32, 16)
MASTER_TARGET_SIZE = 1024
MIN_RECOMMENDED_SIZE = 256
# Padding fraction added around the artwork if tight or re-centered
TARGET_PADDING_RATIO = 0.045

try:
    from PIL import Image as _PILImage  # type: ignore[import-not-found]
except ImportError:
    _PILImage = None


# ---------------------------------------------------------------------------
# Low-level RGBA array helpers (H, W, 4) uint8
# ---------------------------------------------------------------------------

def _load_rgba_array(path: Path) -> tuple[np.ndarray, bool]:
    """Load an image as (H, W, 4) uint8 RGBA and report whether source had alpha."""
    if _PILImage is not None:
        with _PILImage.open(path) as img:
            has_alpha = img.mode in ("RGBA", "LA") or (
                img.mode == "P" and "transparency" in img.info
            )
            rgba = np.asarray(img.convert("RGBA"), dtype=np.uint8).copy()
            return rgba, has_alpha

    pygame.init()
    surf = pygame.image.load(str(path))
    has_alpha = surf.get_bitsize() == 32 and surf.get_masks()[3] != 0
    rgb = pygame.surfarray.array3d(surf)  # (W, H, 3)
    rgb = np.transpose(rgb, (1, 0, 2)).copy()  # (H, W, 3)
    if has_alpha:
        alpha = np.transpose(pygame.surfarray.array_alpha(surf), (1, 0)).copy()
    else:
        alpha = np.full(rgb.shape[:2], 255, dtype=np.uint8)
    rgba = np.dstack((rgb, alpha)).astype(np.uint8)
    return rgba, has_alpha


def _rgba_to_surface(rgba: np.ndarray) -> pygame.Surface:
    """Convert (H, W, 4) uint8 RGBA array to a 32-bit SRCALPHA Pygame Surface."""
    h, w, _ = rgba.shape
    surf = pygame.Surface((w, h), pygame.SRCALPHA, 32)
    pygame.surfarray.blit_array(surf, np.transpose(rgba[:, :, :3], (1, 0, 2)))
    alpha_view = pygame.surfarray.pixels_alpha(surf)
    alpha_view[:] = np.transpose(rgba[:, :, 3], (1, 0))
    del alpha_view
    return surf


def _save_png_rgba(rgba: np.ndarray, path: Path) -> None:
    """Save (H, W, 4) uint8 RGBA array to a lossless PNG file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if _PILImage is not None:
        _PILImage.fromarray(rgba, mode="RGBA").save(path, format="PNG")
        return
    surf = _rgba_to_surface(rgba)
    pygame.image.save(surf, str(path))


def _encode_png_bytes(rgba: np.ndarray) -> bytes:
    """Encode (H, W, 4) uint8 RGBA array to in-memory PNG bytes."""
    buf = io.BytesIO()
    if _PILImage is not None:
        _PILImage.fromarray(rgba, mode="RGBA").save(buf, format="PNG")
        return buf.getvalue()
    surf = _rgba_to_surface(rgba)
    pygame.image.save(surf, buf, "icon.png")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# High-quality Lanczos-3 downscaling (with premultiplied alpha)
# ---------------------------------------------------------------------------

def _lanczos_kernel(x: np.ndarray, a: int = 3) -> np.ndarray:
    """Evaluate the Lanczos-a kernel."""
    out = np.zeros_like(x, dtype=np.float64)
    mask = np.abs(x) < a
    xm = x[mask]
    # np.sinc(t) == sin(pi*t)/(pi*t)
    out[mask] = np.sinc(xm) * np.sinc(xm / a)
    return out


def _resample_1d(arr: np.ndarray, dst_len: int, axis: int, a: int = 3) -> np.ndarray:
    """Resample a float64 array along *axis* using Lanczos-a."""
    src_len = arr.shape[axis]
    if src_len == dst_len:
        return arr.copy()

    scale = src_len / float(dst_len)
    filter_scale = max(scale, 1.0)
    support = a * filter_scale

    # Pixel centers in source space
    centers = (np.arange(dst_len, dtype=np.float64) + 0.5) * scale - 0.5
    left = np.floor(centers - support).astype(np.int64)
    right = np.ceil(centers + support).astype(np.int64)
    max_taps = int(np.max(right - left)) + 1

    offsets = np.arange(max_taps, dtype=np.int64)[None, :]
    indices = left[:, None] + offsets
    valid = (indices >= 0) & (indices < src_len)
    clamped = np.clip(indices, 0, src_len - 1)

    dist = (indices - centers[:, None]) / filter_scale
    weights = _lanczos_kernel(dist, a=a) * valid
    weight_sums = weights.sum(axis=1, keepdims=True)
    weight_sums = np.where(weight_sums == 0.0, 1.0, weight_sums)
    weights /= weight_sums  # (dst_len, max_taps)

    # Move target axis to 0 -> (src_len, ...)
    moved = np.moveaxis(arr, axis, 0)
    orig_tail = moved.shape[1:]
    flat = moved.reshape(src_len, -1)

    # Gather and weighted sum per destination coordinate
    out = np.empty((dst_len, flat.shape[1]), dtype=np.float64)
    for i in range(dst_len):
        w = weights[i]
        nz = w != 0.0
        out[i] = w[nz] @ flat[clamped[i, nz]]

    out = out.reshape((dst_len, *orig_tail))
    return np.moveaxis(out, 0, axis)


def resize_lanczos(rgba: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
    """High-quality Lanczos-3 resize with premultiplied alpha to avoid edge halos."""
    h, w, _ = rgba.shape
    if w == target_w and h == target_h:
        return rgba.copy()

    if _PILImage is not None:
        pil_img = _PILImage.fromarray(rgba, mode="RGBA")
        resized = pil_img.resize((target_w, target_h), _PILImage.Resampling.LANCZOS)
        return np.asarray(resized, dtype=np.uint8)

    f = rgba.astype(np.float64) / 255.0
    alpha = f[:, :, 3:4]
    premult = np.concatenate((f[:, :, :3] * alpha, alpha), axis=2)

    tmp = _resample_1d(premult, target_w, axis=1, a=3)
    res = _resample_1d(tmp, target_h, axis=0, a=3)

    res_a = np.clip(res[:, :, 3:4], 0.0, 1.0)
    safe_a = np.where(res_a > 1e-6, res_a, 1.0)
    res_rgb = np.where(res_a > 1e-6, np.clip(res[:, :, :3] / safe_a, 0.0, 1.0), 0.0)
    out = np.concatenate((res_rgb, res_a), axis=2)
    return np.round(out * 255.0).astype(np.uint8)


# ---------------------------------------------------------------------------
# Exterior solid-black background extraction (preserves interior artwork 100%)
# ---------------------------------------------------------------------------

def _remove_exterior_solid_black(rgba: np.ndarray, threshold: int = 12) -> tuple[np.ndarray, bool]:
    """Strip solid black exterior margin outside a closed badge without altering interior art.

    Flood-fills only from the 4 outer edges through near-black pixels
    (``max(R, G, B) <= threshold``), then applies 1px anti-aliased edge alpha
    only on pixels directly touching the exterior mask so the rounded corners
    sit cleanly on both dark and light taskbars/cards.
    """
    h, w, _ = rgba.shape
    rgb = rgba[:, :, :3]
    corners = (rgb[0, 0], rgb[0, w - 1], rgb[h - 1, 0], rgb[h - 1, w - 1])
    if any(int(c.max()) > threshold for c in corners):
        return rgba.copy(), False

    max_ch = rgb.max(axis=2)
    dark = max_ch <= threshold
    bg_mask = np.zeros((h, w), dtype=bool)
    queue: deque[tuple[int, int]] = deque()

    for x in range(w):
        if dark[0, x] and not bg_mask[0, x]:
            bg_mask[0, x] = True
            queue.append((0, x))
        if dark[h - 1, x] and not bg_mask[h - 1, x]:
            bg_mask[h - 1, x] = True
            queue.append((h - 1, x))
    for y in range(h):
        if dark[y, 0] and not bg_mask[y, 0]:
            bg_mask[y, 0] = True
            queue.append((y, 0))
        if dark[y, w - 1] and not bg_mask[y, w - 1]:
            bg_mask[y, w - 1] = True
            queue.append((y, w - 1))

    while queue:
        y, x = queue.popleft()
        if y > 0 and not bg_mask[y - 1, x] and dark[y - 1, x]:
            bg_mask[y - 1, x] = True
            queue.append((y - 1, x))
        if y + 1 < h and not bg_mask[y + 1, x] and dark[y + 1, x]:
            bg_mask[y + 1, x] = True
            queue.append((y + 1, x))
        if x > 0 and not bg_mask[y, x - 1] and dark[y, x - 1]:
            bg_mask[y, x - 1] = True
            queue.append((y, x - 1))
        if x + 1 < w and not bg_mask[y, x + 1] and dark[y, x + 1]:
            bg_mask[y, x + 1] = True
            queue.append((y, x + 1))

    if not bg_mask.any():
        return rgba.copy(), False

    out = rgba.copy()
    out[bg_mask, 3] = 0
    out[bg_mask, :3] = 0

    # Smooth 1-pixel fringe directly bordering the exterior mask
    neighbor_bg = np.zeros((h, w), dtype=bool)
    neighbor_bg[1:, :] |= bg_mask[:-1, :]
    neighbor_bg[:-1, :] |= bg_mask[1:, :]
    neighbor_bg[:, 1:] |= bg_mask[:, :-1]
    neighbor_bg[:, :-1] |= bg_mask[:, 1:]
    fringe = (~bg_mask) & neighbor_bg & (max_ch < 48)
    if fringe.any():
        fringe_alpha = np.clip((max_ch[fringe].astype(np.float64) - 6.0) / 42.0, 0.0, 1.0)
        out[fringe, 3] = np.round(fringe_alpha * 255.0).astype(np.uint8)

    return out, True


# ---------------------------------------------------------------------------
# Master preparation, .ico writer, and size preview
# ---------------------------------------------------------------------------

def inspect_and_build_master(source_path: Path) -> tuple[np.ndarray, dict[str, object]]:
    """Inspect source logo and construct the centered transparent square master."""
    rgba, has_alpha = _load_rgba_array(source_path)
    src_h, src_w, _ = rgba.shape
    aspect = src_w / float(src_h)

    has_transparency = bool((rgba[:, :, 3] < 255).any())
    if has_transparency:
        cleaned = rgba.copy()
        removed_solid_bg = False
        bg_kind = "transparent"
    else:
        cleaned, removed_solid_bg = _remove_exterior_solid_black(rgba)
        bg_kind = "solid black (removed exterior margin for alpha)" if removed_solid_bg else "solid opaque"

    # Measure artwork bounding box (non-transparent pixels)
    ys, xs = np.where(cleaned[:, :, 3] > 0)
    if len(xs) == 0:
        raise RuntimeError(f"Logo image appears completely empty: {source_path}")

    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    art_w = x1 - x0 + 1
    art_h = y1 - y0 + 1
    pad_l, pad_t = x0, y0
    pad_r, pad_b = src_w - 1 - x1, src_h - 1 - y1

    # Center cropped artwork on a square transparent canvas with safe padding
    art_crop = cleaned[y0 : y1 + 1, x0 : x1 + 1]
    max_art = max(art_w, art_h)
    pad_px = max(int(round(max_art * TARGET_PADDING_RATIO)), 1)
    canvas_side = max_art + pad_px * 2

    canvas = np.zeros((canvas_side, canvas_side, 4), dtype=np.uint8)
    ox = (canvas_side - art_w) // 2
    oy = (canvas_side - art_h) // 2
    canvas[oy : oy + art_h, ox : ox + art_w] = art_crop

    # Never upscale blurry: cap master size to source resolution if smaller than 1024
    master_side = min(MASTER_TARGET_SIZE, canvas_side)
    master = resize_lanczos(canvas, master_side, master_side)

    report: dict[str, object] = {
        "source_path": str(source_path),
        "source_size": (src_w, src_h),
        "aspect_ratio": round(aspect, 4),
        "has_alpha_channel": has_alpha,
        "has_transparent_pixels": has_transparency,
        "background": bg_kind,
        "artwork_bbox": (x0, y0, x1, y1),
        "artwork_size": (art_w, art_h),
        "source_padding_ltrb": (pad_l, pad_t, pad_r, pad_b),
        "too_small_warning": min(src_w, src_h) < MIN_RECOMMENDED_SIZE,
        "master_size": (master_side, master_side),
        "resampler": "Pillow LANCZOS" if _PILImage is not None else "NumPy Lanczos-3 (premultiplied alpha)",
    }
    return master, report


def write_ico(master: np.ndarray, ico_path: Path, sizes: tuple[int, ...] = ICO_SIZES) -> None:
    """Write a multi-size Windows .ico file with Lanczos-downscaled PNG frames."""
    ico_path.parent.mkdir(parents=True, exist_ok=True)
    if _PILImage is not None:
        pil_master = _PILImage.fromarray(master, mode="RGBA")
        # Pillow downscales using Lanczos internally when saving multi-size ICO
        frames = [
            pil_master.resize((s, s), _PILImage.Resampling.LANCZOS) for s in sizes
        ]
        frames[-1].save(
            ico_path,
            format="ICO",
            append_images=frames[:-1],
            sizes=[(s, s) for s in sizes],
        )
        return

    # Standard Windows Vista+ PNG-in-ICO container
    entries: list[tuple[int, bytes]] = []
    for size in sizes:
        scaled = resize_lanczos(master, size, size)
        png_bytes = _encode_png_bytes(scaled)
        entries.append((size, png_bytes))

    count = len(entries)
    # ICONDIR header: reserved=0 (H), type=1 icon (H), count (H)
    header = struct.pack("<HHH", 0, 1, count)
    dir_size = 6 + count * 16
    offset = dir_size
    directory = bytearray()
    payload = bytearray()

    for size, png_bytes in entries:
        dim_byte = 0 if size >= 256 else size
        # ICONDIRENTRY: width(B), height(B), colors(B), reserved(B),
        # planes(H), bpp(H), bytes_in_res(I), image_offset(I)
        directory.extend(
            struct.pack("<BBBBHHII", dim_byte, dim_byte, 0, 0, 1, 32, len(png_bytes), offset)
        )
        payload.extend(png_bytes)
        offset += len(png_bytes)

    with open(ico_path, "wb") as f:
        f.write(header)
        f.write(directory)
        f.write(payload)


def build_preview_sheet(
    master: np.ndarray,
    preview_path: Path,
    sizes: tuple[int, ...] = PREVIEW_SIZES,
) -> None:
    """Create a dark + light background preview strip at 256, 128, 64, 48, 32, 16 px."""
    pygame.init()
    pad = 24
    gap = 28
    label_h = 26
    row_h = max(sizes) + label_h + pad * 2
    total_w = pad * 2 + sum(sizes) + gap * (len(sizes) - 1)
    total_h = row_h * 2

    canvas = pygame.Surface((total_w, total_h), pygame.SRCALPHA, 32)
    font = pygame.font.SysFont("Segoe UI", 14, bold=True)

    rows = (
        ("Dark background (#121418)", (18, 20, 24), (210, 220, 232)),
        ("Light background (#f3f4f6)", (243, 244, 246), (30, 36, 44)),
    )

    scaled_surfs = {
        s: _rgba_to_surface(resize_lanczos(master, s, s)) for s in sizes
    }

    for row_idx, (title, bg_color, text_color) in enumerate(rows):
        y0 = row_idx * row_h
        pygame.draw.rect(canvas, bg_color, (0, y0, total_w, row_h))
        title_surf = font.render(title, True, text_color)
        canvas.blit(title_surf, (pad, y0 + 8))

        baseline_y = y0 + row_h - pad - label_h
        x = pad
        for s in sizes:
            icon_surf = scaled_surfs[s]
            canvas.blit(icon_surf, (x, baseline_y - s))
            lbl = font.render(f"{s}px", True, text_color)
            lbl_x = x + (s - lbl.get_width()) // 2
            canvas.blit(lbl, (lbl_x, baseline_y + 6))
            x += s + gap

    preview_path.parent.mkdir(parents=True, exist_ok=True)
    pygame.image.save(canvas, str(preview_path))


def generate_all(source_path: Path = LOGO_SOURCE_PATH) -> dict[str, object]:
    """Run full icon generation pipeline and return the inspection report."""
    if not source_path.is_file():
        raise FileNotFoundError(f"Source logo not found: {source_path}")

    ICONS_DIR.mkdir(parents=True, exist_ok=True)
    master, report = inspect_and_build_master(source_path)
    _save_png_rgba(master, ICON_MASTER_PATH)
    write_ico(master, ICON_ICO_PATH, ICO_SIZES)
    build_preview_sheet(master, ICON_PREVIEW_PATH, PREVIEW_SIZES)
    return report


def main() -> int:
    report = generate_all(LOGO_SOURCE_PATH)
    print("=== Chess Ken Icon Generation Report ===")
    for k, v in report.items():
        print(f"  {k}: {v}")
    print(f"  master saved : {ICON_MASTER_PATH}")
    print(f"  ico saved    : {ICON_ICO_PATH} (sizes={ICO_SIZES})")
    print(f"  preview saved: {ICON_PREVIEW_PATH} (sizes={PREVIEW_SIZES})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
