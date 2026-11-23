"""Non-blocking chess sound effect manager backed by pygame.mixer.

Pre-loads trimmed WAV clips from ``others.config.SOUNDS_DIR`` once at startup
and degrades gracefully (silent no-op) in headless or audio-less environments.
"""

from __future__ import annotations

import logging
from pathlib import Path

import chess
import pygame

from others.config import BG_MUSIC_PATH, SOUNDS_DIR

logger = logging.getLogger(__name__)

# Background music sits calmly beneath chess SFX so captures/checks stay crisp.
DEFAULT_MUSIC_VOLUME_RATIO: float = 0.45

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
        music_path: Path | None = None,
        music_volume_ratio: float = DEFAULT_MUSIC_VOLUME_RATIO,
    ) -> None:
        self._sounds_dir: Path = sounds_dir if sounds_dir is not None else SOUNDS_DIR
        self._music_path: Path = music_path if music_path is not None else BG_MUSIC_PATH
        self._volume: float = self._clamp_volume(volume)
        self._music_volume_ratio: float = self._clamp_volume(music_volume_ratio)
        self._muted: bool = bool(muted)
        self._available: bool = False
        self._music_playing: bool = False
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

    @property
    def music_playing(self) -> bool:
        """Whether background music playback is active and unmuted."""
        return self._music_playing and not self._muted

    def _effective_music_volume(self) -> float:
        """Return the active mixer.music volume respecting mute and balance ratio."""
        if self._muted:
            return 0.0
        return self._volume * self._music_volume_ratio

    def _sync_music_volume(self) -> None:
        """Apply current effective volume to pygame.mixer.music safely."""
        if not self._available:
            return
        try:
            pygame.mixer.music.set_volume(self._effective_music_volume())
        except pygame.error:
            pass

    def start_music(self, *, loop: bool = True, fade_ms: int = 800) -> bool:
        """Stream and loop the background music track without blocking.

        Idempotent if music is already playing (avoids restarting mid-loop on
        menu screen transitions). Degrades gracefully to a silent no-op if audio
        is unavailable or the music file does not exist.
        """
        if not self._available or not self._music_path.is_file():
            return False
        if self._music_playing:
            self._sync_music_volume()
            return True
        try:
            pygame.mixer.music.load(str(self._music_path))
            pygame.mixer.music.set_volume(self._effective_music_volume())
            pygame.mixer.music.play(loops=-1 if loop else 0, fade_ms=max(0, int(fade_ms)))
            self._music_playing = True
            return True
        except pygame.error as exc:
            logger.warning("Failed to start background music %s: %s", self._music_path.name, exc)
            self._music_playing = False
            return False

    def stop_music(self, fade_ms: int = 0) -> None:
        """Stop or fade out background music playback safely."""
        if not self._available:
            self._music_playing = False
            return
        try:
            if fade_ms > 0:
                pygame.mixer.music.fadeout(int(fade_ms))
            else:
                pygame.mixer.music.stop()
        except pygame.error:
            pass
        self._music_playing = False

    def set_volume(self, volume: float) -> None:
        """Set playback volume in [0.0, 1.0] for all pre-loaded clips and music."""
        self._volume = self._clamp_volume(volume)
        for sound in self._sounds.values():
            try:
                sound.set_volume(self._volume)
            except pygame.error:
                pass
        self._sync_music_volume()

    def toggle_mute(self) -> bool:
        """Toggle mute state for SFX and background music and return the new value."""
        self._muted = not self._muted
        self._sync_music_volume()
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
