"""Data-driven bot speech lines and time-based speech bubble state machine.

Loaded once from ``others.config.SPEECH_LINES_PATH`` so bot taglines, move-quality
reactions (blunder, mistake, dubious, great, brilliant), and match/endgame lines
have a single source of truth shared across screens, bots, and tests.

State machine lifecycle:
    HIDDEN -> APPEARING -> VISIBLE -> DISMISSING (roll/slide left) -> HIDDEN
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import json
import logging
from pathlib import Path
import random
from typing import Callable

from others.config import SPEECH_LINES_PATH

logger = logging.getLogger(__name__)


# ── Named animation, timing, cooldown & rate-limit constants ─────────
BUBBLE_APPEAR_DURATION: float = 0.25        # Smooth appear / pop-in from avatar tail
BUBBLE_TEXT_DELAY: float = 0.16             # Crisp staggered text fade-in
BUBBLE_TEXT_DURATION: float = 0.26          # Text fade + gentle lift duration
BUBBLE_TEXT_CHANGE_DURATION: float = 0.24   # Quick cross-transition when preempted
BUBBLE_VISIBLE_DURATION: float = 5.0        # Stay visible 5.0 s before rolling left
BUBBLE_AUDIO_TAIL_DELAY: float = 0.35       # Short pause after voice clip ends before dismissing
BUBBLE_DISMISS_DURATION: float = 0.30       # Smooth roll/slide-left dismissal (ease-in)
BUBBLE_FAST_PREEMPT_DURATION: float = 0.12  # Fast roll-away when preempted by higher priority
BUBBLE_MAX_DT: float = 1.0 / 20.0           # Clamp dt against frame hitches

# Reading-time calculation when no explicit fixed hold is requested
BUBBLE_READING_SEC_PER_CHAR: float = 0.06
BUBBLE_READING_MIN_SEC: float = 1.8
BUBBLE_READING_MAX_SEC: float = 6.0

# Cooldown & per-game blunder guardrails
BUBBLE_GLOBAL_COOLDOWN: float = 8.0         # Minimum gap between normal reactions (s)
BLUNDER_MIN_PLAYER_MOVES_BETWEEN: int = 6   # At most 1 blunder reaction per 6 human moves
BLUNDER_MAX_PER_GAME: int = 3               # At most 3 blunder reactions per game

# Setup panel behavior (stays visible by default; set True to enable auto roll-left there)
SETUP_BUBBLE_AUTO_DISMISS: bool = False


class BubblePhase(str, Enum):
    """Lifecycle phases of the opponent speech bubble."""

    HIDDEN = "hidden"
    APPEARING = "appearing"
    VISIBLE = "visible"
    DISMISSING = "dismissing"


@dataclass(frozen=True)
class SpeechLine:
    """Single data-driven bot speech line entry from ``data/speech_lines.json``."""

    id: str
    bots: tuple[str, ...]
    event: str
    text: str
    audio: str | None = None
    weight: float = 1.0
    priority: int = 50
    cooldown: float = 0.0


_DEFAULT_FALLBACK_LINES: tuple[SpeechLine, ...] = (
    SpeechLine("greeting", ("sick_ken",), "GREETING", "Bai gihilantan ko, pildiha daw ko bi", None, 1.0, 20, 0.0),
    SpeechLine("greeting", ("focus_ken",), "GREETING", "Ha?? sure ka makig duwa ka nako", None, 1.0, 20, 0.0),
    SpeechLine("greeting", ("condition_ken",), "GREETING", "hmmm dependi kong 3 yan!", None, 1.0, 20, 0.0),
    SpeechLine(
        "blunder_walang_kanin",
        ("sick_ken", "focus_ken", "condition_ken"),
        "PLAYER_BLUNDER",
        "Walang kanin Buseng!",
        "walang_kanin_buseng.wav",
        1.4,
        100,
        0.0,
    ),
)

_CACHED_LINES: tuple[SpeechLine, ...] | None = None


def normalize_bot_id(identifier: str) -> str:
    """Convert bot display names or tiers (e.g. 'Sick Ken', 'sick') to canonical bot_id."""
    norm = identifier.strip().lower().replace("-", "_").replace(" ", "_")
    mapping = {
        "sick": "sick_ken",
        "sick_ken": "sick_ken",
        "focus": "focus_ken",
        "focus_ken": "focus_ken",
        "condition": "condition_ken",
        "condition_ken": "condition_ken",
    }
    return mapping.get(norm, norm or "focus_ken")


def load_speech_lines(path: Path | None = None, *, force_reload: bool = False) -> tuple[SpeechLine, ...]:
    """Load and cache all bot speech lines from JSON (falls back safely if missing)."""
    global _CACHED_LINES
    if path is None and _CACHED_LINES is not None and not force_reload:
        return _CACHED_LINES

    target = path if path is not None else SPEECH_LINES_PATH
    loaded: list[SpeechLine] = []
    if target.is_file():
        try:
            raw = json.loads(target.read_text(encoding="utf-8"))
            for item in raw.get("lines", []):
                bots_raw = item.get("bots") or item.get("bot") or ()
                if isinstance(bots_raw, str):
                    bots = (normalize_bot_id(bots_raw),)
                else:
                    bots = tuple(normalize_bot_id(str(b)) for b in bots_raw)
                loaded.append(
                    SpeechLine(
                        id=str(item["id"]),
                        bots=bots,
                        event=str(item["event"]).strip().upper(),
                        text=str(item["text"]),
                        audio=str(item["audio"]) if item.get("audio") else None,
                        weight=max(0.01, float(item.get("weight", 1.0))),
                        priority=int(item.get("priority", 50)),
                        cooldown=max(0.0, float(item.get("cooldown", 0.0))),
                    )
                )
        except Exception as exc:
            logger.warning("Failed to load speech lines from %s: %s", target, exc)

    result = tuple(loaded) if loaded else _DEFAULT_FALLBACK_LINES
    if path is None:
        _CACHED_LINES = result
    return result


def get_lines_for_event(
    event: str,
    bot_id: str,
    *,
    lines: tuple[SpeechLine, ...] | None = None,
) -> list[SpeechLine]:
    """Return all configured ``SpeechLine`` entries matching ``(event, bot_id)``."""
    norm_event = event.strip().upper()
    norm_bot = normalize_bot_id(bot_id)
    pool = lines if lines is not None else load_speech_lines()
    return [
        line
        for line in pool
        if line.event == norm_event and (not line.bots or norm_bot in line.bots)
    ]


def get_greeting_text(bot_identifier: str, fallback: str = "") -> str:
    """Return the canonical GREETING text for a bot from ``data/speech_lines.json``."""
    matches = get_lines_for_event("GREETING", bot_identifier)
    if matches:
        return matches[0].text
    return fallback


def compute_visible_duration(
    text: str,
    *,
    fixed_duration: float | None = BUBBLE_VISIBLE_DURATION,
) -> float:
    """Compute bubble visible hold time (5.0 s default, or clamped reading time if fixed is None)."""
    if fixed_duration is not None:
        return max(0.5, float(fixed_duration))
    reading = len(text) * BUBBLE_READING_SEC_PER_CHAR
    return max(BUBBLE_READING_MIN_SEC, min(BUBBLE_READING_MAX_SEC, reading))


class SpeechBubbleController:
    """Manages speech bubble lifecycle, roll-left dismissal, priority, cooldowns, and voice sync."""

    def __init__(
        self,
        bot_id: str,
        *,
        lines: tuple[SpeechLine, ...] | None = None,
        reduced_motion: bool = False,
        rng: random.Random | None = None,
        is_audio_playing: Callable[[], bool] | None = None,
        play_audio: Callable[[SpeechLine, str], bool] | None = None,
        stop_audio: Callable[[], None] | None = None,
    ) -> None:
        self._bot_id: str = normalize_bot_id(bot_id)
        self._lines: tuple[SpeechLine, ...] = (
            lines if lines is not None else load_speech_lines()
        )
        self._reduced_motion: bool = bool(reduced_motion)
        self._rng: random.Random = rng if rng is not None else random.Random()
        self._is_audio_playing = is_audio_playing
        self._play_audio = play_audio
        self._stop_audio = stop_audio

        # Animation & lifecycle state
        self._phase: BubblePhase = BubblePhase.HIDDEN
        self._current_line: SpeechLine | None = None
        self._current_text: str = ""
        self._phase_elapsed: float = 0.0
        self._visible_elapsed: float = 0.0
        self._visible_duration: float = BUBBLE_VISIBLE_DURATION
        self._auto_dismiss: bool = True
        self._dismiss_duration: float = BUBBLE_DISMISS_DURATION
        self._audio_started: bool = False
        self._audio_done_elapsed: float = 0.0
        self._pending_line: SpeechLine | None = None
        self._pending_auto_dismiss: bool = True

        # Cooldown, dedup, and per-game blunder guardrails
        self._clock: float = 0.0
        self._last_bubble_start: float = -1e9
        self._line_last_fired: dict[str, float] = {}
        self._last_id_by_event: dict[str, str] = {}
        self._reacted_plies: set[int] = set()
        self._blunder_count: int = 0
        self._last_blunder_player_move: int = -1000

    # ── Read-only state properties ───────────────────────────────────

    @property
    def phase(self) -> BubblePhase:
        return self._phase

    @property
    def is_visible(self) -> bool:
        return self._phase != BubblePhase.HIDDEN and bool(self._current_text)

    @property
    def text(self) -> str:
        return self._current_text

    @property
    def current_line(self) -> SpeechLine | None:
        return self._current_line

    @property
    def blunder_count(self) -> int:
        return self._blunder_count

    def set_bot(self, bot_identifier: str) -> None:
        self._bot_id = normalize_bot_id(bot_identifier)

    def reset_for_match(self) -> None:
        """Reset per-match cooldowns, move deduplication, and blunder counters."""
        self.dismiss_immediately()
        self._clock = 0.0
        self._last_bubble_start = -1e9
        self._line_last_fired.clear()
        self._reacted_plies.clear()
        self._blunder_count = 0
        self._last_blunder_player_move = -1000

    def dismiss_immediately(self) -> None:
        """Stop voice playback and hide the bubble immediately (restart/resign/exit)."""
        if self._stop_audio is not None:
            try:
                self._stop_audio()
            except Exception:
                pass
        self._phase = BubblePhase.HIDDEN
        self._current_line = None
        self._current_text = ""
        self._phase_elapsed = 0.0
        self._visible_elapsed = 0.0
        self._audio_started = False
        self._audio_done_elapsed = 0.0
        self._pending_line = None

    def show_setup_greeting(self, fallback_text: str = "") -> SpeechLine | None:
        """Display the bot's greeting in the setup panel and play its greeting clip once."""
        matches = get_lines_for_event("GREETING", self._bot_id, lines=self._lines)
        line = (
            matches[0]
            if matches
            else SpeechLine("greeting", (self._bot_id,), "GREETING", fallback_text, None, 1.0, 20, 0.0)
        )
        if not line.text:
            return None
        self._activate_line(
            line,
            auto_dismiss=SETUP_BUBBLE_AUTO_DISMISS,
            bypass_cooldown=True,
        )
        return line

    def trigger_event(
        self,
        event: str,
        *,
        move_ply: int | None = None,
        player_move_number: int | None = None,
        auto_dismiss: bool = True,
        bypass_cooldown: bool = False,
        visible_duration: float | None = BUBBLE_VISIBLE_DURATION,
    ) -> SpeechLine | None:
        """Attempt to trigger a speech reaction for *event* subject to priority & limits."""
        norm_event = event.strip().upper()

        # Prevent duplicate reactions firing twice for the exact same move
        if move_ply is not None and move_ply in self._reacted_plies:
            return None

        # Enforce strict PLAYER_BLUNDER limits (max 3/game, at least 6 human moves apart)
        if norm_event == "PLAYER_BLUNDER":
            if self._blunder_count >= BLUNDER_MAX_PER_GAME:
                return None
            if (
                player_move_number is not None
                and (player_move_number - self._last_blunder_player_move)
                < BLUNDER_MIN_PLAYER_MOVES_BETWEEN
            ):
                return None

        candidates = get_lines_for_event(norm_event, self._bot_id, lines=self._lines)
        if not candidates:
            return None

        # Filter by per-line cooldown
        eligible = [
            c
            for c in candidates
            if bypass_cooldown
            or c.cooldown <= 0.0
            or (self._clock - self._line_last_fired.get(c.id, -1e9)) >= c.cooldown
        ]
        if not eligible:
            return None

        # Prefer not repeating the exact same line back-to-back when alternatives exist
        last_id = self._last_id_by_event.get(norm_event)
        varied = [c for c in eligible if c.id != last_id]
        pool = varied if varied else eligible
        chosen = self._weighted_pick(pool)

        # Priority & global cooldown arbitration
        active_busy = self._phase in (BubblePhase.APPEARING, BubblePhase.VISIBLE)
        cur_priority = self._current_line.priority if self._current_line is not None else -1

        if active_busy:
            if chosen.priority <= cur_priority:
                return None
            # Higher-priority line preempts current bubble quickly & stops its audio
            if self._stop_audio is not None:
                try:
                    self._stop_audio()
                except Exception:
                    pass
            self._record_trigger_metadata(chosen, move_ply, player_move_number)
            self._pending_line = chosen
            self._pending_auto_dismiss = auto_dismiss
            self._visible_duration = compute_visible_duration(
                chosen.text, fixed_duration=visible_duration
            )
            if self._reduced_motion:
                self._activate_line(
                    chosen,
                    auto_dismiss=auto_dismiss,
                    bypass_cooldown=True,
                    visible_duration=visible_duration,
                )
            else:
                self._phase = BubblePhase.DISMISSING
                self._phase_elapsed = 0.0
                self._dismiss_duration = BUBBLE_FAST_PREEMPT_DURATION
            return chosen

        # Global cooldown between sequential bubbles (bypassed for high-priority events >= 80)
        if (
            not bypass_cooldown
            and chosen.priority < 80
            and (self._clock - self._last_bubble_start) < BUBBLE_GLOBAL_COOLDOWN
        ):
            return None

        self._record_trigger_metadata(chosen, move_ply, player_move_number)
        self._activate_line(
            chosen,
            auto_dismiss=auto_dismiss,
            bypass_cooldown=True,
            visible_duration=visible_duration,
        )
        return chosen

    def _record_trigger_metadata(
        self,
        line: SpeechLine,
        move_ply: int | None,
        player_move_number: int | None,
    ) -> None:
        if move_ply is not None:
            self._reacted_plies.add(move_ply)
        self._last_id_by_event[line.event] = line.id
        self._line_last_fired[line.id] = self._clock
        if line.event == "PLAYER_BLUNDER":
            self._blunder_count += 1
            if player_move_number is not None:
                self._last_blunder_player_move = player_move_number

    def _weighted_pick(self, pool: list[SpeechLine]) -> SpeechLine:
        if len(pool) == 1:
            return pool[0]
        weights = [c.weight for c in pool]
        return self._rng.choices(pool, weights=weights, k=1)[0]

    def _activate_line(
        self,
        line: SpeechLine,
        *,
        auto_dismiss: bool,
        bypass_cooldown: bool = True,
        visible_duration: float | None = BUBBLE_VISIBLE_DURATION,
    ) -> None:
        self._current_line = line
        self._current_text = line.text
        self._pending_line = None
        self._auto_dismiss = bool(auto_dismiss)
        self._visible_duration = compute_visible_duration(
            line.text, fixed_duration=visible_duration
        )
        self._dismiss_duration = BUBBLE_DISMISS_DURATION
        self._phase_elapsed = 0.0
        self._visible_elapsed = 0.0
        self._audio_done_elapsed = 0.0
        self._last_bubble_start = self._clock

        if self._reduced_motion:
            self._phase = BubblePhase.VISIBLE
            self._phase_elapsed = BUBBLE_APPEAR_DURATION + BUBBLE_TEXT_DURATION
        else:
            self._phase = BubblePhase.APPEARING

        self._audio_started = False
        if self._play_audio is not None:
            try:
                self._audio_started = bool(self._play_audio(line, self._bot_id))
            except Exception:
                self._audio_started = False

    def _voice_currently_playing(self) -> bool:
        if not self._audio_started or self._is_audio_playing is None:
            return False
        try:
            return bool(self._is_audio_playing())
        except Exception:
            return False

    def update(self, dt: float) -> None:
        """Advance state machine using clamped delta-time (never teleports on hitches)."""
        clamped = min(max(0.0, float(dt)), BUBBLE_MAX_DT)
        self._clock += clamped

        if self._phase == BubblePhase.HIDDEN:
            return

        self._phase_elapsed += clamped

        if self._phase == BubblePhase.APPEARING:
            if self._phase_elapsed >= BUBBLE_APPEAR_DURATION:
                self._phase = BubblePhase.VISIBLE
                self._phase_elapsed = 0.0
                self._visible_elapsed = 0.0
            return

        if self._phase == BubblePhase.VISIBLE:
            self._visible_elapsed += clamped
            if not self._auto_dismiss:
                return

            playing = self._voice_currently_playing()
            if playing:
                # Never slide away while voice audio is still playing
                self._audio_done_elapsed = 0.0
                return

            if self._audio_started:
                self._audio_done_elapsed += clamped

            # Dismiss after visible hold duration (5.0 s) AND after any playing audio + tail delay
            audio_ready = (not self._audio_started) or (
                self._audio_done_elapsed >= BUBBLE_AUDIO_TAIL_DELAY
            )
            if self._visible_elapsed >= self._visible_duration and audio_ready:
                if self._reduced_motion:
                    self._phase = BubblePhase.HIDDEN
                    self._current_line = None
                    self._current_text = ""
                else:
                    self._phase = BubblePhase.DISMISSING
                    self._phase_elapsed = 0.0
                    self._dismiss_duration = BUBBLE_DISMISS_DURATION
            return

        if self._phase == BubblePhase.DISMISSING:
            if self._phase_elapsed >= self._dismiss_duration:
                if self._pending_line is not None:
                    nxt = self._pending_line
                    nxt_auto = self._pending_auto_dismiss
                    self._pending_line = None
                    self._activate_line(
                        nxt,
                        auto_dismiss=nxt_auto,
                        visible_duration=self._visible_duration,
                    )
                else:
                    self._phase = BubblePhase.HIDDEN
                    self._current_line = None
                    self._current_text = ""
                    self._phase_elapsed = 0.0
                    self._audio_started = False

    def animation_values(self) -> tuple[float, float, float]:
        """Return ``(appear_t, text_t, dismiss_t)`` in ``[0.0, 1.0]`` for rendering.

        - ``appear_t``: 0 -> 1 during APPEARING, 1 during VISIBLE/DISMISSING
        - ``text_t``: 0 -> 1 staggered fade-in
        - ``dismiss_t``: 0 during APPEARING/VISIBLE, 0 -> 1 during roll-left DISMISSING
        """
        if self._phase == BubblePhase.HIDDEN:
            return 0.0, 0.0, 1.0
        if self._reduced_motion:
            return 1.0, 1.0, 0.0
        if self._phase == BubblePhase.APPEARING:
            appear_t = min(1.0, self._phase_elapsed / max(0.01, BUBBLE_APPEAR_DURATION))
            if self._phase_elapsed <= BUBBLE_TEXT_DELAY:
                text_t = 0.0
            else:
                text_t = min(
                    1.0,
                    (self._phase_elapsed - BUBBLE_TEXT_DELAY)
                    / max(0.01, BUBBLE_TEXT_DURATION),
                )
            return appear_t, text_t, 0.0
        if self._phase == BubblePhase.VISIBLE:
            total_since_show = BUBBLE_APPEAR_DURATION + self._visible_elapsed
            if total_since_show <= BUBBLE_TEXT_DELAY:
                text_t = 0.0
            else:
                text_t = min(
                    1.0,
                    (total_since_show - BUBBLE_TEXT_DELAY)
                    / max(0.01, BUBBLE_TEXT_DURATION),
                )
            return 1.0, text_t, 0.0
        # DISMISSING: smooth roll/slide left
        dismiss_t = min(1.0, self._phase_elapsed / max(0.01, self._dismiss_duration))
        return 1.0, 1.0, dismiss_t
