"""Download and SHA-256 verify the official Stockfish Windows binary.

Invoked automatically by ``others/run_game.bat`` when ``engine/stockfish.exe``
is missing. Uses only the Python standard library (no admin rights needed) and
downloads exclusively from the official Stockfish GitHub Releases page.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
import shutil
import sys
import tempfile
import urllib.error
import urllib.request
import zipfile

from others.config import ENGINE_DIR

# ── Pinned official Stockfish release constants ───────────────────────
STOCKFISH_VERSION = "19"
STOCKFISH_RELEASE_TAG = "sf_19"
STOCKFISH_SOURCE_URL = "https://github.com/official-stockfish/Stockfish"
STOCKFISH_RELEASE_URL = (
    f"{STOCKFISH_SOURCE_URL}/releases/tag/{STOCKFISH_RELEASE_TAG}"
)
STOCKFISH_ARCHIVE_NAME = "stockfish-windows-x86-64-universal.zip"
STOCKFISH_DOWNLOAD_URL = (
    f"{STOCKFISH_SOURCE_URL}/releases/download/"
    f"{STOCKFISH_RELEASE_TAG}/{STOCKFISH_ARCHIVE_NAME}"
)
# Official SHA-256 digest from GitHub Releases for sf_19 Windows x86-64 universal ZIP
STOCKFISH_ARCHIVE_SHA256 = (
    "3c8bf1f9ea66a09350a40df4f632288285ac206d99f33ab5842c408fc30b48a7"
)
# Expected SHA-256 digest of the extracted stockfish executable inside the ZIP
# (set to None if not pinned; verified when set).
STOCKFISH_BINARY_SHA256: str | None = (
    "45bc8e4969147db9c2eb533810637994619bff0eacc81ccfd9854394901bcbd0"
)

_CHUNK_SIZE = 256 * 1024  # 256 KB
_USER_AGENT = "ChessIQ-Stockfish-Setup/1.0"


def sha256_file(path: Path) -> str:
    """Compute lowercase hexadecimal SHA-256 digest of *path*."""
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(_CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest().lower()


def verify_sha256(path: Path, expected_sha256: str) -> tuple[bool, str]:
    """Verify *path* matches *expected_sha256*. Deletes *path* on mismatch."""
    actual = sha256_file(path)
    expected = expected_sha256.strip().lower()
    if actual != expected:
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass
        return False, actual
    return True, actual


def print_manual_instructions(target_path: Path) -> None:
    """Print clear manual download and USB fallback instructions."""
    print()
    print("=" * 68)
    print("Manual Stockfish Setup (if automatic download is blocked):")
    print("=" * 68)
    print("Option 1 — Download manually from the official Stockfish page:")
    print(f"  1. Visit: {STOCKFISH_RELEASE_URL}")
    print(f"  2. Download: {STOCKFISH_ARCHIVE_NAME}")
    print(f"  3. Extract the .exe file from the ZIP and save it as:")
    print(f"     {target_path}")
    print()
    print("Option 2 — Copy by USB from your laptop (works offline):")
    print("  1. Copy 'engine\\stockfish.exe' from your laptop onto a USB drive.")
    print(f"  2. Paste it into this computer's folder: {target_path.parent}")
    print(f"     so the file path is: {target_path}")
    print("=" * 68)
    print()


def _download_with_progress(url: str, dest: Path, timeout: int = 30) -> None:
    """Stream *url* to *dest* with inline progress output."""
    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        total_header = resp.headers.get("Content-Length")
        total_bytes = int(total_header) if total_header and total_header.isdigit() else 0
        downloaded = 0
        last_pct = -1
        with dest.open("wb") as out:
            while chunk := resp.read(_CHUNK_SIZE):
                out.write(chunk)
                downloaded += len(chunk)
                if total_bytes > 0:
                    pct = int(downloaded * 100 / total_bytes)
                    if pct >= last_pct + 5 or downloaded >= total_bytes:
                        last_pct = pct
                        mb_done = downloaded / (1024 * 1024)
                        mb_tot = total_bytes / (1024 * 1024)
                        print(
                            f"\rDownloading Stockfish {STOCKFISH_VERSION}: "
                            f"{mb_done:5.1f} / {mb_tot:5.1f} MB ({pct:3d}%)",
                            end="",
                            flush=True,
                        )
                else:
                    mb_done = downloaded / (1024 * 1024)
                    print(
                        f"\rDownloading Stockfish {STOCKFISH_VERSION}: {mb_done:5.1f} MB",
                        end="",
                        flush=True,
                    )
    print()


def _extract_exe_from_zip(zip_path: Path, out_exe: Path) -> str:
    """Safely extract the Stockfish .exe member from *zip_path* into *out_exe*."""
    with zipfile.ZipFile(zip_path, "r") as zf:
        exe_members = [
            info
            for info in zf.infolist()
            if not info.is_dir() and info.filename.lower().endswith(".exe")
        ]
        if not exe_members:
            raise RuntimeError(
                f"No .exe executable found inside archive {zip_path.name}."
            )
        # Prefer the largest .exe (the Stockfish engine binary itself)
        chosen = max(exe_members, key=lambda info: info.file_size)
        with zf.open(chosen, "r") as src, out_exe.open("wb") as dst:
            shutil.copyfileobj(src, dst, length=_CHUNK_SIZE)
        return chosen.filename


def ensure_stockfish(target_path: Path | None = None) -> bool:
    """Ensure ``engine/stockfish.exe`` exists, downloading and verifying if missing.

    Returns True if Stockfish is ready on disk, False on download/verification failure.
    """
    dest = target_path if target_path is not None else (ENGINE_DIR / "stockfish.exe")
    if dest.is_file() and dest.stat().st_size > 0:
        print(f"Stockfish binary already present: {dest}")
        return True

    print(f"Stockfish not found at {dest}.")
    print(f"Fetching official Stockfish {STOCKFISH_VERSION} ({STOCKFISH_ARCHIVE_NAME})...")
    print(f"Source: {STOCKFISH_DOWNLOAD_URL}")

    dest.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="chessiq_sf_") as tmp_dir:
        tmp_path = Path(tmp_dir)
        zip_path = tmp_path / STOCKFISH_ARCHIVE_NAME
        extracted_exe = tmp_path / "stockfish.exe"

        try:
            _download_with_progress(STOCKFISH_DOWNLOAD_URL, zip_path)
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            zip_path.unlink(missing_ok=True)
            print(f"\n[ERROR] Failed to download Stockfish: {exc}")
            print_manual_instructions(dest)
            return False

        print("Verifying archive SHA-256 hash...")
        ok, actual_zip_hash = verify_sha256(zip_path, STOCKFISH_ARCHIVE_SHA256)
        if not ok:
            print("[ERROR] Archive SHA-256 mismatch! Deleted downloaded file.")
            print(f"  Expected: {STOCKFISH_ARCHIVE_SHA256}")
            print(f"  Actual:   {actual_zip_hash}")
            print_manual_instructions(dest)
            return False
        print(f"Archive SHA-256 verified: {actual_zip_hash}")

        try:
            member_name = _extract_exe_from_zip(zip_path, extracted_exe)
            print(f"Extracted {member_name}")
        except (zipfile.BadZipFile, RuntimeError, OSError) as exc:
            extracted_exe.unlink(missing_ok=True)
            print(f"[ERROR] Failed to extract Stockfish binary: {exc}")
            print_manual_instructions(dest)
            return False

        if STOCKFISH_BINARY_SHA256:
            print("Verifying extracted binary SHA-256 hash...")
            bin_ok, actual_bin_hash = verify_sha256(
                extracted_exe, STOCKFISH_BINARY_SHA256
            )
            if not bin_ok:
                print("[ERROR] Extracted binary SHA-256 mismatch! Deleted file.")
                print(f"  Expected: {STOCKFISH_BINARY_SHA256}")
                print(f"  Actual:   {actual_bin_hash}")
                print_manual_instructions(dest)
                return False
            print(f"Binary SHA-256 verified: {actual_bin_hash}")

        shutil.move(str(extracted_exe), str(dest))

    print(f"Stockfish {STOCKFISH_VERSION} installed to {dest}")
    return True


def main() -> int:
    return 0 if ensure_stockfish() else 1


if __name__ == "__main__":
    raise SystemExit(main())
