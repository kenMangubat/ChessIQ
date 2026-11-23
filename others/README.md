# ChessIQ
Personalized AI chess platform in Python &amp; Pygame with style-calibrated bots trained on real over-the-board and online games.

## Setup &amp; Run

- Install dependencies: `pip install -r others/requirements.txt`
- Launch on Windows: `others\run_game.bat` (or `python main.py` from the repository root)
- Shared paths and configuration live in `others/config.py`.

## Icons &amp; Packaging Notes

- Regenerate the master icon, `.ico`, and size preview from the source logo: `python -m others.make_icon`
  - Optional build-time dependency: `Pillow` (not required at runtime; do not add to `others/requirements.txt`). When Pillow is not installed, `others/make_icon.py` falls back to a built-in NumPy Lanczos-3 resampler.
- Future PyInstaller build icon flag: `--icon=assets/icons/chessken.ico`
