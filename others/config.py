"""Shared project paths."""

from pathlib import Path
import shutil


BASE_DIR = Path(__file__).resolve().parent.parent
ASSETS_DIR = BASE_DIR / "assets"
ICONS_DIR = ASSETS_DIR / "icons"
SOUNDS_DIR = ASSETS_DIR / "sounds"
BG_MUSIC_PATH = SOUNDS_DIR / "bg_music.ogg"
DATA_DIR = BASE_DIR / "data"
ENGINE_DIR = BASE_DIR / "engine"

# Logo & application icon paths
# Checks assets/chesslogo.png first, falling back to assets/gamelogo.png.
LOGO_SOURCE_PATH = (
    ASSETS_DIR / "chesslogo.png"
    if (ASSETS_DIR / "chesslogo.png").is_file()
    else ASSETS_DIR / "gamelogo.png"
)
ICON_MASTER_PATH = ICONS_DIR / "chessken_master.png"
# Future PyInstaller build: pass --icon=assets/icons/chessken.ico (ICON_ICO_PATH)
ICON_ICO_PATH = ICONS_DIR / "chessken.ico"
ICON_PREVIEW_PATH = ICONS_DIR / "preview_sizes.png"


def get_stockfish_path() -> Path | None:
    """Find the Stockfish executable path in ENGINE_DIR or system PATH.

    Checks ENGINE_DIR for standard executable names first, then searches for
    any stockfish binary in ENGINE_DIR, and finally falls back to system PATH.
    Returns None if no binary is found.
    """
    for candidate in ("stockfish.exe", "stockfish"):
        local_path = ENGINE_DIR / candidate
        if local_path.is_file():
            return local_path

    if ENGINE_DIR.exists():
        for candidate_path in ENGINE_DIR.glob("stockfish*"):
            if candidate_path.is_file() and (
                candidate_path.suffix.lower() in (".exe", "") or not candidate_path.suffix
            ):
                return candidate_path

    which_path = shutil.which("stockfish")
    if which_path:
        return Path(which_path)

    return None


# Opening book and style database
BOOK_DB_PATH = DATA_DIR / "ken_book.db"
STYLE_PROFILE_PATH = DATA_DIR / "ken_style_profile.json"
OPENINGS_TSV_PATH = DATA_DIR / "chess_openings.tsv"

# Offline bot calibration output files
CALIBRATION_JSON_PATH = DATA_DIR / "calibration_results.json"
CALIBRATION_CSV_PATH = DATA_DIR / "calibration_results.csv"

# Per-game bot move provenance logs (bounded & rotated)
PROVENANCE_LOG_DIR = DATA_DIR / "move_logs"

# PGN game archives located directly inside data/
PGN_FILES: list[Path] = sorted(DATA_DIR.glob("*.pgn"))

# Explicit PGN path constants
EMKEN_GAMES_PGN = DATA_DIR / "emkengames.pgn"
KEN_IS_NOT_HERE_PGN = DATA_DIR / "ken_is_not_here.pgn"
LICHESS_KEN_UCLM1_BLACK_PGN = DATA_DIR / "lichess_KEN_UCLM1_blackk.pgn"
LICHESS_KEN_UCLM1_WHITE_PGN = DATA_DIR / "lichess_KEN_UCLM1_white.pgn"
LICHESS_THISSS_KENN_PGN = DATA_DIR / "lichess_thisssKenn.pgn"
LICHESS_UCLM_KEN2_PGN = DATA_DIR / "lichess_UCLM_Ken2.pgn"

# Mapping by stem name for convenient lookup
PGN_MAP: dict[str, Path] = {path.stem: path for path in PGN_FILES}

