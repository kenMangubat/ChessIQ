"""Stockfish UCI engine wrapper.

Provides lifecycle management of the Stockfish process, confirmed UCI option
configuration (with automatic UCI_Elo clamping and UCI_LimitStrength reset),
multi-PV candidate move analysis, and reference-level play for calibration.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
import subprocess
import sys
from types import TracebackType
from typing import Any

import chess
import chess.engine

from config import ENGINE_DIR, get_stockfish_path


class EngineNotFoundError(FileNotFoundError):
    """Raised when the Stockfish executable binary is not found on disk or PATH."""


class StockfishEngine:
    """Wrapper around chess.engine.SimpleEngine for Stockfish analysis and play."""

    def __init__(
        self,
        executable_path: Path | str | None = None,
        threads: int = 1,
        hash_mb: int = 32,
    ) -> None:
        self._custom_path = Path(executable_path) if executable_path else None
        self._threads = max(1, int(threads))
        self._hash_mb = max(16, int(hash_mb))
        self._engine: chess.engine.SimpleEngine | None = None
        self._engine_name: str = "Stockfish"
        self._uci_elo_min: int = 1320
        self._uci_elo_max: int = 3190
        self._current_limit_strength: bool = False
        self._current_uci_elo: int | None = None
        self._current_skill_level: int = 20

    @property
    def is_running(self) -> bool:
        """Return True if the engine process is active."""
        return self._engine is not None

    @property
    def uci_elo_range(self) -> tuple[int, int]:
        """Return the (min_elo, max_elo) supported by the active Stockfish binary."""
        self.start()
        return (self._uci_elo_min, self._uci_elo_max)

    @property
    def applied_options(self) -> dict[str, Any]:
        """Return a snapshot of the confirmed UCI configuration on this engine."""
        return {
            "engine_name": self._engine_name,
            "threads": self._threads,
            "hash_mb": self._hash_mb,
            "uci_limit_strength": self._current_limit_strength,
            "uci_elo": self._current_uci_elo,
            "skill_level": self._current_skill_level,
            "uci_elo_range": (self._uci_elo_min, self._uci_elo_max),
        }

    def _resolve_binary_path(self) -> Path:
        """Locate the Stockfish binary or raise EngineNotFoundError."""
        if self._custom_path is not None:
            if self._custom_path.is_file():
                return self._custom_path
            raise EngineNotFoundError(
                f"Specified Stockfish binary does not exist: {self._custom_path}"
            )
        resolved = get_stockfish_path()
        if resolved is None or not resolved.is_file():
            raise EngineNotFoundError(
                f"Stockfish binary not found. Place 'stockfish.exe' inside '{ENGINE_DIR}' "
                "or install Stockfish and add it to system PATH."
            )
        return resolved

    def start(self) -> StockfishEngine:
        """Start the engine process if not already running and apply base options."""
        if self._engine is None:
            binary_path = self._resolve_binary_path()
            try:
                popen_kwargs: dict[str, Any] = {}
                if sys.platform == "win32":
                    popen_kwargs["creationflags"] = (
                        subprocess.BELOW_NORMAL_PRIORITY_CLASS
                        | subprocess.CREATE_NO_WINDOW
                    )
                self._engine = chess.engine.SimpleEngine.popen_uci(
                    str(binary_path), **popen_kwargs
                )
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to start Stockfish process at {binary_path}: {exc}"
                ) from exc

            self._engine_name = self._engine.id.get("name", "Stockfish")
            elo_opt = self._engine.options.get("UCI_Elo")
            if elo_opt is not None:
                if elo_opt.min is not None:
                    self._uci_elo_min = int(elo_opt.min)
                if elo_opt.max is not None:
                    self._uci_elo_max = int(elo_opt.max)

            init_cfg: dict[str, Any] = {}
            if "Threads" in self._engine.options:
                init_cfg["Threads"] = self._threads
            if "Hash" in self._engine.options:
                init_cfg["Hash"] = self._hash_mb
            if init_cfg:
                self._engine.configure(init_cfg)

            self._current_limit_strength = False
            self._current_uci_elo = None
            self._current_skill_level = 20
        return self

    def _apply_strength_options(
        self,
        uci_elo: int | None = None,
        skill_level: int | None = None,
    ) -> None:
        """Configure UCI_LimitStrength / UCI_Elo / Skill Level with proper clamping and reset."""
        assert self._engine is not None
        cfg: dict[str, Any] = {}

        if uci_elo is not None:
            clamped_elo = max(self._uci_elo_min, min(self._uci_elo_max, int(uci_elo)))
            if not self._current_limit_strength or self._current_uci_elo != clamped_elo:
                if "UCI_LimitStrength" in self._engine.options:
                    cfg["UCI_LimitStrength"] = True
                if "UCI_Elo" in self._engine.options:
                    cfg["UCI_Elo"] = clamped_elo
                self._current_limit_strength = True
                self._current_uci_elo = clamped_elo
        else:
            if self._current_limit_strength:
                if "UCI_LimitStrength" in self._engine.options:
                    cfg["UCI_LimitStrength"] = False
                self._current_limit_strength = False
                self._current_uci_elo = None

        target_skill = 20 if skill_level is None else max(0, min(20, int(skill_level)))
        if target_skill != self._current_skill_level and "Skill Level" in self._engine.options:
            cfg["Skill Level"] = target_skill
            self._current_skill_level = target_skill

        if cfg:
            self._engine.configure(cfg)

    def stop(self) -> None:
        """Shut down the engine process cleanly."""
        if self._engine is not None:
            try:
                self._engine.quit()
            except Exception:
                pass
            finally:
                self._engine = None
                self._current_limit_strength = False
                self._current_uci_elo = None
                self._current_skill_level = 20

    def analyse_candidates(
        self,
        board: chess.Board,
        multipv: int = 3,
        time_limit: float = 0.15,
        uci_elo: int | None = None,
        depth: int | None = None,
        root_moves: Iterable[chess.Move] | None = None,
    ) -> list[dict[str, Any]]:
        """Evaluate candidate moves for the given board position.

        Returns a list of dictionaries with keys:
          - 'move': chess.Move
          - 'uci': str (e.g. 'e2e4')
          - 'score_cp': int (centipawn score relative to side to move)
          - 'mate': int | None (moves to mate, positive if side-to-move mates, negative if mated)
          - 'depth': int (search depth reached)
        """
        if root_moves is not None:
            root_list = [m for m in root_moves if m in board.legal_moves]
            legal_count = len(root_list)
        else:
            root_list = None
            legal_count = sum(1 for _ in board.legal_moves)

        if legal_count == 0:
            return []

        # Ensure engine is running (raises EngineNotFoundError if missing)
        self.start()
        assert self._engine is not None

        effective_multipv = min(max(1, multipv), legal_count)
        self._apply_strength_options(uci_elo=uci_elo, skill_level=None)

        limit = chess.engine.Limit(
            time=max(0.01, time_limit) if time_limit > 0 else None,
            depth=depth,
        )
        try:
            info = self._engine.analyse(
                board,
                limit=limit,
                multipv=effective_multipv,
                root_moves=root_list,
            )
        except Exception as exc:
            self.stop()
            raise RuntimeError(f"Engine analysis failed: {exc}") from exc

        info_list = info if isinstance(info, list) else [info]

        candidates: list[dict[str, Any]] = []
        seen_moves: set[chess.Move] = set()
        for entry in info_list:
            pv = entry.get("pv")
            if not pv:
                continue
            move = pv[0]
            if move in seen_moves:
                continue
            seen_moves.add(move)
            score_obj = entry.get("score")
            score_cp = 0
            mate = None
            if score_obj is not None:
                rel = score_obj.relative
                raw_cp = rel.score(mate_score=10000)
                score_cp = int(raw_cp) if raw_cp is not None else 0
                mate = rel.mate()

            candidates.append(
                {
                    "move": move,
                    "uci": move.uci(),
                    "score_cp": score_cp,
                    "mate": mate,
                    "depth": int(entry.get("depth", 0)),
                }
            )

        return candidates

    def play_move(
        self,
        board: chess.Board,
        time_limit: float = 0.05,
        depth: int | None = None,
        uci_elo: int | None = None,
        skill_level: int | None = None,
    ) -> chess.Move:
        """Play a single move using Stockfish's own move selector (for reference opponents)."""
        self.start()
        assert self._engine is not None
        self._apply_strength_options(uci_elo=uci_elo, skill_level=skill_level)
        limit = chess.engine.Limit(
            time=max(0.005, time_limit) if time_limit > 0 else None,
            depth=depth,
        )
        result = self._engine.play(board, limit=limit)
        if result.move is None:
            raise RuntimeError("Stockfish returned no move for non-terminal board.")
        return result.move

    def __enter__(self) -> StockfishEngine:
        self.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.stop()
