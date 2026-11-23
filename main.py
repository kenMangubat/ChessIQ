"""ChessIQ — main entry point.

Initialises Pygame, opens a full-screen window, and runs the screen-manager
loop.  All UI / drawing logic lives in the individual screen classes.
"""

from collections import deque
import sys
import time
import pygame

from screens.dashboard import DashboardScreen
from screens.game import GameScreen
from screens.opponent_select import OpponentSelectScreen
from screens.settings import SettingsScreen

# ── constants ────────────────────────────────────────────────────────
FPS = 60
WINDOW_TITLE = "Chess Ken"


class _PerfOverlay:
    """Lightweight F3-toggleable frame-time overlay (disabled by default)."""

    _WINDOW_SAMPLES = 240
    _REFRESH_INTERVAL = 0.15

    def __init__(self) -> None:
        self.visible: bool = False
        self._samples_ms: deque[float] = deque(maxlen=self._WINDOW_SAMPLES)
        self._font: pygame.font.Font | None = None
        self._bg: pygame.Surface | None = None
        self._lines: list[pygame.Surface] = []
        self._last_refresh: float = 0.0

    def toggle(self) -> None:
        self.visible = not self.visible
        if self.visible:
            self._last_refresh = 0.0

    def record(self, frame_ms: float) -> None:
        self._samples_ms.append(frame_ms)

    def stats(self) -> dict[str, float]:
        if not self._samples_ms:
            return {"fps": 0.0, "last": 0.0, "avg": 0.0, "p95": 0.0, "p99": 0.0, "worst": 0.0}
        ordered = sorted(self._samples_ms)
        n = len(ordered)
        avg = sum(ordered) / n
        last = self._samples_ms[-1]
        p95 = ordered[min(n - 1, int(n * 0.95))]
        p99 = ordered[min(n - 1, int(n * 0.99))]
        worst = ordered[-1]
        fps = 1000.0 / avg if avg > 0.0 else 0.0
        return {"fps": fps, "last": last, "avg": avg, "p95": p95, "p99": p99, "worst": worst}

    def draw(self, surface: pygame.Surface) -> None:
        if not self.visible:
            return
        if self._font is None:
            self._font = pygame.font.SysFont("Consolas", 14, bold=True)
        now = time.perf_counter()
        if not self._lines or (now - self._last_refresh) >= self._REFRESH_INTERVAL:
            self._last_refresh = now
            s = self.stats()
            texts = (
                f"FPS:   {s['fps']:5.1f}",
                f"Last:  {s['last']:5.2f} ms",
                f"Avg:   {s['avg']:5.2f} ms",
                f"P95:   {s['p95']:5.2f} ms",
                f"P99:   {s['p99']:5.2f} ms",
                f"Worst: {s['worst']:5.2f} ms",
            )
            self._lines = [
                self._font.render(t, True, (220, 255, 220)) for t in texts
            ]
            w = max(line.get_width() for line in self._lines) + 16
            h = sum(line.get_height() for line in self._lines) + 12
            if self._bg is None or self._bg.get_size() != (w, h):
                self._bg = pygame.Surface((w, h), pygame.SRCALPHA)
                self._bg.fill((12, 16, 22, 200))
                pygame.draw.rect(self._bg, (52, 211, 153, 180), self._bg.get_rect(), width=1, border_radius=6)

        assert self._bg is not None
        x, y = 12, 12
        surface.blit(self._bg, (x, y))
        cy = y + 6
        for line in self._lines:
            surface.blit(line, (x + 8, cy))
            cy += line.get_height()


# =====================================================================
# App
# =====================================================================

class App:
    """Top-level application: owns the display and screen manager."""

    def __init__(self) -> None:
        pygame.init()

        # full-screen display
        self.screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
        pygame.display.set_caption(WINDOW_TITLE)
        pygame.mouse.set_visible(True)

        self.clock = pygame.time.Clock()
        self._perf_overlay = _PerfOverlay()
        self.running: bool = True
        self._volume_percent = 70
        self._premove_enabled: bool = True
        self._selected_opponent: dict[str, object] | None = None
        self._settings_return_screen = "dashboard"

        # screen registry — maps names to factory callables
        self._screen_factories: dict[str, callable] = {
            "dashboard": lambda: DashboardScreen(self.screen),
            "opponent_select": lambda: OpponentSelectScreen(self.screen),
            "game": lambda: GameScreen(
                self.screen,
                self._get_selected_opponent(),
                premove_enabled=self._premove_enabled,
            ),
            "settings": lambda: SettingsScreen(
                self.screen,
                self._volume_percent,
                self._set_volume,
                self._settings_return_screen,
                premove_enabled=self._premove_enabled,
                on_premove_changed=self._set_premove_enabled,
            ),
        }

        self._current_screen = self._screen_factories["dashboard"]()

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Run the fixed-FPS application loop until exit."""
        while self.running:
            frame_start = time.perf_counter()
            dt = self.clock.tick(FPS) / 1000.0

            # collect events once per frame
            events = pygame.event.get()
            for event in events:
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_F3:
                    self._perf_overlay.toggle()
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    if not self._current_screen.captures_escape():
                        self._navigate("dashboard")

            # delegate to current screen
            self._current_screen.handle_events(events)
            self._current_screen.update(dt)
            self._current_screen.draw()
            self._perf_overlay.draw(self.screen)

            # check for navigation request
            if self._current_screen.next_screen is not None:
                self._navigate(self._current_screen.next_screen)

            pygame.display.flip()
            self._perf_overlay.record((time.perf_counter() - frame_start) * 1000.0)
  
        self._shutdown()

    # ------------------------------------------------------------------
    # Navigation
    # ------------------------------------------------------------------

    def _navigate(self, target: str) -> None:
        """Switch to a different screen by name, or exit."""
        if target == "exit":
            self.running = False
            return

        if target == "game" and isinstance(
            self._current_screen, OpponentSelectScreen
        ):
            self._selected_opponent = self._current_screen.selected_opponent
        if target == "game" and self._selected_opponent is None:
            raise RuntimeError("Cannot open the game before selecting an opponent.")
        if target == "settings":
            self._settings_return_screen = (
                "game" if isinstance(self._current_screen, GameScreen) else "dashboard"
            )

        factory = self._screen_factories.get(target)
        self._current_screen.close()
        if factory is not None:
            self._current_screen = factory()
        else:
            # unknown screen — fall back to dashboard
            self._current_screen = self._screen_factories["dashboard"]()

    def _get_selected_opponent(self) -> dict[str, object]:
        if self._selected_opponent is None:
            raise RuntimeError("No opponent has been selected.")
        return self._selected_opponent

    def _set_volume(self, volume: int) -> None:
        """Keep the settings value when the settings screen is reopened."""
        self._volume_percent = volume

    def _set_premove_enabled(self, enabled: bool) -> None:
        """Keep the premove setting across screens."""
        self._premove_enabled = bool(enabled)

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    def _shutdown(self) -> None:
        """Close the active screen (stopping any bot worker), then quit."""
        self._current_screen.close()
        pygame.quit()
        sys.exit()


# ── run ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    App().run()
 