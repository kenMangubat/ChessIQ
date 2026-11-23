"""Non-blocking chess sound effect manager backed by pygame.mixer.

Pre-loads trimmed WAV clips from ``others.config.SOUNDS_DIR`` once at startup
and degrades gracefully (silent no-op) in headless or audio-less environments.
"""

from __future__ import annotations

import logging
from pathlib import Path

import chess
import pygame

from others.config import SOUNDS_DIR

logger = logging.getLogger(__name__)

SOUND_NAMES: tuple[str, ...] = (
    "game_start",
    "move_self",
    "move_opponent",
    "castle",
    "capture",
    "check",
    "promote",
    "game_end",
    "stalemate",
    "checkmate",
)


def select_move_sound(
    board_before: chess.Board,
    move: chess.Move,
    board_after: chess.Board,
    is_player_move: bool,
) -> str:
    """Choose the highest-priority sound cue name for a chess move.

    Priority:
        1. ``checkmate``
        2. ``stalemate``
        3. ``game_end`` (any other game-over condition)
        4. ``check``
        5. ``promote``
        6. ``castle``
        7. ``capture``
        8. ``move_self`` / ``move_opponent``
    """
    if board_after.is_checkmate():
        return "checkmate"
    if board_after.is_stalemate():
        return "stalemate"
    if board_after.is_game_over(claim_draw=True):
        return "game_end"
    if board_after.is_check():
        return "check"
    if move.promotion is not None:
        return "promote"
    if board_before.is_castling(move):
        return "castle"
    if board_before.is_capture(move):
        return "capture"
    return "move_self" if is_player_move else "move_opponent"


class SoundManager:
    """Pre-loaded, non-blocking sound player with safe headless fallback."""

    def __init__(
        self,
        sounds_dir: Path | None = None,
        *,
        volume: float = 0.7,
        muted: bool = False,
    ) -> None:
        self._sounds_dir: Path = sounds_dir if sounds_dir is not None else SOUNDS_DIR
        self._volume: float = self._clamp_volume(volume)
        self._muted: bool = bool(muted)
        self._available: bool = False
        self._sounds: dict[str, pygame.mixer.Sound] = {}

        self._init_mixer()
        if self._available:
            self._preload_sounds()

    @staticmethod
    def _clamp_volume(volume: float) -> float:
        return max(0.0, min(1.0, float(volume)))

    def _init_mixer(self) -> None:
        """Initialize pygame.mixer safely without crashing on headless hosts."""
        try:
            if pygame.mixer.get_init() is None:
                pygame.mixer.init()
            self._available = pygame.mixer.get_init() is not None
        except pygame.error as exc:
            logger.debug("pygame.mixer unavailable; audio disabled (%s).", exc)
            self._available = False

    def _preload_sounds(self) -> None:
        """Load all WAV clips from disk once at startup."""
        if not self._sounds_dir.is_dir():
            return
        for wav_path in sorted(self._sounds_dir.glob("*.wav")):
            try:
                sound = pygame.mixer.Sound(str(wav_path))
                sound.set_volume(self._volume)
                self._sounds[wav_path.stem] = sound
            except pygame.error as exc:
                logger.warning("Failed to load sound %s: %s", wav_path.name, exc)

    @property
    def muted(self) -> bool:
        """Whether sound playback is currently muted."""
        return self._muted

    @property
    def volume(self) -> float:
        """Current playback volume in [0.0, 1.0]."""
        return self._volume

    @property
    def available(self) -> bool:
        """Whether pygame.mixer initialized successfully."""
        return self._available

    @property
    def loaded_sounds(self) -> tuple[str, ...]:
        """Sorted tuple of pre-loaded sound cue names."""
        return tuple(sorted(self._sounds.keys()))

    def set_volume(self, volume: float) -> None:
        """Set playback volume in [0.0, 1.0] for all pre-loaded clips."""
        self._volume = self._clamp_volume(volume)
        for sound in self._sounds.values():
            try:
                sound.set_volume(self._volume)
            except pygame.error:
                pass

    def toggle_mute(self) -> bool:
        """Toggle mute state and return the new muted value."""
        self._muted = not self._muted
        return self._muted

    def play(self, name: str) -> bool:
        """Play a pre-loaded sound cue by name without blocking.

        Returns True if playback started, False if muted, unavailable, or missing.
        """
        if self._muted or not self._available:
            return False
        sound = self._sounds.get(name)
        if sound is None:
            return False
        try:
            sound.play()
            return True
        except pygame.error:
            return False

    def play_for_move(
        self,
        board_before: chess.Board,
        move: chess.Move,
        board_after: chess.Board,
        is_player_move: bool,
    ) -> str:
        """Select and play the appropriate sound for a move, returning its cue name."""
        cue = select_move_sound(board_before, move, board_after, is_player_move)
        self.play(cue)
        return cue
