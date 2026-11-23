
# ChessIQ — Agent Rules

## GOLDEN RULE: REUSE EXISTING FILES FIRST

> This rule has the highest priority for all development tasks.

1. Never create a file or folder without checking whether it already exists.
2. Always inspect the existing project structure before implementation.
3. Reuse and modify existing files whenever possible.
4. Never create duplicate files with the same name or duplicate modules serving the same purpose.
5. Only create a new file when the requested functionality genuinely requires a separate module and no suitable existing file can handle it.
6. Place new files inside the appropriate existing directory.
7. Never create unnecessary folders.
8. Never rename, replace, or delete existing files without explicit permission.
9. Preserve existing functionality when modifying files.
10. Before creating anything, check the actual project directory, not just your assumptions.

### Example

If the project already contains:

ChessIQ/
├── main.py
└── others/
    ├── AGENTS.md
    └── ken.py

You are NOT allowed to create another ken.py.

If a separate feature genuinely requires another module, use a distinct and descriptive filename, such as shoe_brand.py, and place it appropriately.

The example filename is illustrative only.

### Mandatory Workflow

Before implementation:
- Inspect the project structure.
- Identify relevant existing files.
- Determine which files need modification.
- Determine whether any new file is genuinely necessary.

After implementation:
- Check for accidental duplicate files.
- Check for unnecessary folders.
- Report modified files.
- Report newly created files and explain why they were needed.

---

## Stack

- Python, Pygame, python-chess, Stockfish (UCI), sqlite3, NumPy.

## Architecture

- Follow the existing folder structure and never add folders without asking.
- **Never place new files in the repository root.** `main.py` is the only file allowed at the repository root. If a task genuinely requires creating an extra, helper, script, or miscellaneous file that does not belong in `assets/`, `bots/`, `calibration/`, `chess_logic/`, `data/`, `engine/`, `screens/`, `style/`, `sets/`, `tests/`, or `ui/`, it MUST be placed inside `others/`.
- Every screen is a class inheriting from BaseScreen in screens/base_screen.py; main.py holds the App class and screen manager.
- Build one phase at a time; do not implement anything outside the current task.
- Do not add features or settings not explicitly requested; do not assume unspecified parts.
- Keep the UI responsive: fixed 60 FPS loop, engine analysis and PGN parsing in background threads, cached assets, no blocking calls in the main loop.
- Do not hardcode absolute paths; use `others/config.py` (`from others.config import ...`).
- Install dependencies with `pip install -r others/requirements.txt` or launch via `others/run_game.bat`.
- Write simple tests in tests/ for non-UI logic when asked.

---

## CHESSBOARD DESIGN AND ASSET PRESERVATION

1. **Do not change the design of the chessboard**: Maintain the existing board layout, geometry, borders, coordinate notation, and default color themes unless the user explicitly requests a design change.
2. **Preserve piece assets**: Never overwrite, modify, regenerate, or replace the canonical chess piece SVGs in `assets/` (`wK.svg`, `bP.svg`, etc.) with alternative or experimental styles.
3. **Guard test/generator scripts**: Automated tests or utility scripts must never mutate, overwrite, or clean up production visual assets in `assets/`.

---

## TOKEN AND CONTEXT OPTIMIZATION

### Context Management
1. Inspect only files relevant to the current task.
2. Do not repeatedly scan the entire project.
3. Do not reread unchanged files unnecessarily.
4. Avoid loading unrelated source files, assets, documentation, and dependencies.
5. Exclude generated and dependency directories from broad inspection, including:
   - __pycache__
   - .git
   - build
   - dist
   - node_modules
   - .venv
6. Use targeted searches instead of dumping entire directory contents.
7. Retrieve additional context only when necessary for accurate implementation.
8. Reuse information already established in the current task when still valid.

### Token Efficiency
1. Keep responses concise and task-focused.
2. Do not print entire source files unless explicitly requested.
3. Avoid lengthy explanations of straightforward changes.
4. Do not repeat the user's requirements unnecessarily.
5. Avoid unnecessary planning for small tasks.
6. Avoid generating multiple solutions unless requested.
7. Do not produce unnecessary documentation or reports.
8. Use the minimum necessary tool calls to complete the task correctly.

### Task Scope
1. Implement only the requested feature or fix.
2. Do not refactor unrelated modules.
3. Do not redesign existing interfaces unless requested.
4. Do not introduce unnecessary dependencies.
5. Do not modify unrelated configuration files.
6. Complete the current task before suggesting additional improvements.

---

## CODING QUALITY

1. Follow existing naming conventions and coding style.
2. Reuse existing classes, functions, and components.
3. Maintain compatibility with the current architecture.
4. Keep functions and modules maintainable.
5. Avoid unnecessary abstractions and overengineering.
6. Handle errors appropriately.
7. Preserve existing application behavior.
8. Avoid hardcoded values when existing configuration options are available.

---

## DEBUGGING AND ERROR HANDLING

1. Identify the actual root cause before modifying code.
2. Inspect relevant error messages and stack traces.
3. Modify only the files necessary to fix the issue.
4. Do not rewrite entire modules to fix small bugs.
5. Do not repeatedly attempt the same failed solution.
6. Verify that the fix does not break existing functionality.
7. Report unresolved errors honestly.

---

## VERIFICATION

After implementation:
1. Run relevant tests when available.
2. Check Python syntax for modified Python files.
3. Verify imports and dependencies when relevant.
4. Check affected functionality.
5. Confirm that no unrelated files were modified.
6. Never claim successful testing if no test was performed.

---

## SAFETY AND PROJECT PRESERVATION

1. Never expose API keys, passwords, or environment secrets.
2. Never delete user data without explicit authorization.
3. Never overwrite important configurations without inspecting them.
4. Never commit, push, or deploy without authorization. When pushing to GitHub, always target the official repository `https://github.com/kenMangubat/ChessIQ` without creating another folder or another repository.
5. Never perform destructive database operations without authorization.
6. Preserve existing project settings and dependencies.
7. Never alter the chessboard design or overwrite chess piece assets without explicit user request.

---

## GITHUB AND REPOSITORY MANAGEMENT

1. **Official Repository**: When the user requests to push or upload to GitHub, always use the designated repository URL:
   `https://github.com/kenMangubat/ChessIQ`
2. **Account & Username**: The GitHub account username is `kenMangubat`. Authentication is handled securely via Git Credential Manager; never store or commit passwords or tokens to this file or repository.
3. **No Extra Repositories or Nested Folders**: Never create another repository and never wrap the project files in an unnecessary nested subfolder (e.g., `ChessIQ/ChessIQ/`). Put files directly at the repository root.
4. **Preserve Repository**: Never delete or remove the repository.
5. **Binary & Cache Exclusions**: Always ensure `others/.gitignore` (and `.git/info/exclude` / `core.excludesFile`) excludes `.venv/`, `__pycache__/`, `.pytest_cache/`, `*.bak_*` backups, and large executable binaries exceeding GitHub limits (e.g. `engine/stockfish.exe`).

---

## CONTINUOUS BOT TRAINING & DATA LAYOUT

To keep Sick Ken (1350), Focus Ken (1800), and Condition Ken (2350) trained on new games:

1. **Add new games**: Drop any new PGN or Lichess study export into `data/`.
2. **One-command training** (backs up prior DB & profile, runs idempotent ingest, incrementally analyzes only unanalyzed games with Stockfish, retrains White/Black style models, and runs hold-out quality checks):
   ```powershell
   .\.venv\Scripts\python.exe -m calibration.build_book --train
   ```
3. **Strength calibration**:
   ```powershell
   .\.venv\Scripts\python.exe -m calibration.calibrate_bots --quick
   ```
4. **Move explainability / debugging** (inspect any FEN or replay logged bot moves):
   ```powershell
   .\.venv\Scripts\python.exe -m calibration.explain_move --latest
   .\.venv\Scripts\python.exe -m calibration.explain_move --fen "<FEN>" --bot "Focus Ken"
   ```
5. **Opt-in human game export**: Use `calibration.build_book.export_human_game_to_pgn(...)` to append completed human-vs-bot games to `data/human_vs_bot_games.pgn`. Only the human side (`Ken_human`) is ingested; bot moves are never used for training.
6. **`data/` layout**:
   - `*.pgn` — Source PGN archives and Lichess studies (never overwritten).
   - `ken_book.db` — SQLite store for `ingested_games` (with train/holdout split), `book_moves` (weighted), and `game_move_analysis`.
   - `ken_style_profile.json` — Trained v2.0 White/Black move-level style models, piece-square priors, mistake profiles, and tier DNA.
   - `calibration_results.json` / `.csv` — Latest strength & style calibration report.
   - `move_logs/` — Bounded/rotated JSONL per-move provenance logs for `explain_move`.
   - `*.bak_<timestamp>` — Automatic timestamped backups created before retraining.

---

## FINAL RESPONSE FORMAT

After completing a task, provide a concise report:

### Completed
- Brief description of implemented changes.

### Modified Files
- List only files actually modified.

### New Files
- List newly created files and explain why they were necessary.
- If none, state "None."

### Verification
- Tests or checks actually performed.
- Any remaining issues.

Avoid lengthy explanations unless explicitly requested.

---

## FINAL PRIORITY

Correctness and preservation of the existing ChessIQ project take priority over token reduction.

Never sacrifice code quality, functionality, or project integrity just to minimize context or tool usage.
