# ChessIQ
Personalized AI chess platform in Python & Pygame with style-calibrated bots trained on real over-the-board and online games.

## Setup & Run

- Install dependencies: `pip install -r others/requirements.txt`
- Launch on Windows: `others\run_game.bat` (or `python main.py` from the repository root)
- Shared paths and configuration live in `others/config.py`.

## Running on another PC

1. **Install Python:** Install Python 3.10+ from https://www.python.org/downloads/ (check **"Add Python to PATH"** during installation; no admin rights needed if installed for the current user).
2. **Launch the game:** Double-click `others\run_game.bat` (or run `.\others\run_game.bat` in PowerShell / Command Prompt).
3. **What happens on first run:**
   - Creates a local `.venv` in the project folder (no admin rights required).
   - Installs the exact pinned dependencies from `others/requirements.txt` (`pygame-ce`, `python-chess`, `numpy`).
   - Checks for `engine/stockfish.exe`. If missing, automatically downloads the pinned official **Stockfish 19** Windows x86-64 universal build (`sf_19`) from the official Stockfish GitHub Releases page, verifies its SHA-256 hash, and extracts `engine/stockfish.exe`.
   - Launches `main.py` with Windows DPI awareness enabled so the board renders crisply at 100%, 125%, and 150% display scaling.
4. **USB fallback (if a school or restricted network blocks downloads):**
   - On your laptop, copy `engine\stockfish.exe` onto a USB drive.
   - On the other computer, paste it into the `engine\` folder inside this repo (`engine\stockfish.exe`).
   - Alternatively, manually download `stockfish-windows-x86-64-universal.zip` from https://github.com/official-stockfish/Stockfish/releases/tag/sf_19, extract the `.exe`, and rename it to `engine\stockfish.exe`.
   - If Stockfish is missing at launch, ChessIQ displays a clear in-game warning banner instead of silently falling back to the basic heuristic bot.

## Third-Party Engine License (Stockfish)

ChessIQ uses the external **Stockfish** chess engine (`engine/stockfish.exe`), which is free software licensed under the **GNU General Public License v3.0 (GPL-3.0)**. Stockfish is downloaded directly from its official release and is not committed to this repository.

- Official Stockfish source code & license: https://github.com/official-stockfish/Stockfish
- Official Stockfish 19 release: https://github.com/official-stockfish/Stockfish/releases/tag/sf_19
