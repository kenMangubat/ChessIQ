"""ChessIQ — main entry point.

Initialises Pygame, opens a full-screen window, and runs the screen-manager
loop.  All UI / drawing logic lives in the individual screen classes.
"""

from collections import deque
import ctypes
import importlib.metadata
import logging
import platform
import sys
import time
import pygame

from others.config import (
    ASSETS_DIR,
    BOOK_DB_PATH,
    ENGINE_DIR,
    ICON_MASTER_PATH,
    LOGO_SOURCE_PATH,
    STYLE_PROFILE_PATH,
    get_stockfish_path,
)
from screens.dashboard import DashboardScreen
from screens.game import GameScreen
from screens.opponent_select import OpponentSelectScreen
from screens.settings import SettingsScreen
from ui.sound_manager import SoundManager

# ── constants ────────────────────────────────────────────────────────
FPS = 60
WINDOW_TITLE = "Chess Ken"
_REQUIRED_PIECE_SVGS = (
    "wK.svg", "wQ.svg", "wR.svg", "wB.svg", "wN.svg", "wP.svg",
    "bK.svg", "bQ.svg", "bR.svg", "bB.svg", "bN.svg", "bP.svg",
)

logger = logging.getLogger("chessiq.startup")


def _enable_windows_dpi_awareness() -> None:
    """Enable per-monitor DPI awareness on Windows before creating the display."""
    if sys.platform != "win32":
        return
    try:
        # DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2 (-4)
        ctypes.windll.user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4))
        return
    except Exception:
        pass
    try:
        # PROCESS_PER_MONITOR_DPI_AWARE (2)
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
        return
    except Exception:
        pass
    try:
        ctypes.windll.user32.SetProcessDPIAware()
    except Exception:
        pass


def _set_windows_app_user_model_id() -> None:
    """Set a custom Windows AppUserModelID so the taskbar shows our icon."""
    if sys.platform != "win32":
        return
    try:
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("ChessIQ.ChessKen")
    except Exception:
        pass


def _set_window_icon() -> None:
    """Set the Pygame window/taskbar icon from the processed master before set_mode."""
    icon_path = ICON_MASTER_PATH if ICON_MASTER_PATH.is_file() else LOGO_SOURCE_PATH
    if not icon_path.is_file():
        return
    try:
        raw = pygame.image.load(str(icon_path))
        icon_surf = pygame.transform.smoothscale(raw, (64, 64))
        pygame.display.set_icon(icon_surf)
    except Exception as exc:
        logger.warning("Could not set window icon from %s: %s", icon_path, exc)


def _run_startup_self_check() -> list[str]:
    """Log runtime versions and verify required assets, data, and Stockfish.

    Returns a list of user-facing warning messages for missing critical files.
    """
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="[%(levelname)s] %(name)s: %(message)s",
        )

    is_ce = bool(getattr(pygame, "IS_CE", False))
    pkg_label = "pygame-ce" if is_ce else "pygame"
    logger.info(
        "Runtime: Python %s | %s %s (SDL %s)",
        platform.python_version(),
        pkg_label,
        pygame.version.ver,
        ".".join(str(x) for x in pygame.version.SDL),
    )

    warnings: list[str] = []

    try:
        importlib.metadata.distribution("pygame")
        logger.warning(
            "Plain 'pygame' package detected alongside/instead of 'pygame-ce'. "
            "SVG board pieces require 'pygame-ce'. Run: pip uninstall pygame && pip install pygame-ce==2.5.8"
        )
        if not is_ce:
            warnings.append("Plain 'pygame' detected instead of 'pygame-ce'. Pieces may not render.")
    except importlib.metadata.PackageNotFoundError:
        pass

    missing_svgs = [
        name for name in _REQUIRED_PIECE_SVGS if not (ASSETS_DIR / name).is_file()
    ]
    if missing_svgs:
        logger.warning("Missing piece SVG assets in %s: %s", ASSETS_DIR, ", ".join(missing_svgs))
        warnings.append(f"Missing {len(missing_svgs)} piece SVG(s) in assets/.")
    else:
        logger.info("Assets check OK (%s)", ASSETS_DIR)

    if not BOOK_DB_PATH.is_file():
        logger.warning("Opening book DB not found: %s", BOOK_DB_PATH)
        warnings.append(f"Missing book DB: {BOOK_DB_PATH.name}")
    else:
        logger.info("Book DB OK (%s)", BOOK_DB_PATH.name)

    if not STYLE_PROFILE_PATH.is_file():
        logger.warning("Style profile not found: %s", STYLE_PROFILE_PATH)
        warnings.append(f"Missing style profile: {STYLE_PROFILE_PATH.name}")
    else:
        logger.info("Style profile OK (%s)", STYLE_PROFILE_PATH.name)

    stockfish_path = get_stockfish_path()
    if stockfish_path is None:
        logger.error(
            "Stockfish binary not found. Place 'stockfish.exe' in '%s' or add it to PATH.",
            ENGINE_DIR,
        )
        warnings.append(
            f"Stockfish engine not found! Place 'stockfish.exe' in '{ENGINE_DIR.name}/' (bots will not use Stockfish)."
        )
    else:
        logger.info("Stockfish OK (%s)", stockfish_path)

    return warnings


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


class _StartupWarningBanner:
    """Top-center dismissible banner for missing runtime dependencies (e.g. Stockfish)."""

    def __init__(self, warnings: list[str]) -> None:
        self.warnings = warnings
        self.dismissed: bool = not warnings
        self._rect: pygame.Rect | None = None
        self._font: pygame.font.Font | None = None
        self._hint_font: pygame.font.Font | None = None
        self._cached_size: tuple[int, int] = (0, 0)
        self._bg: pygame.Surface | None = None
        self._lines: list[pygame.Surface] = []
        self._hint_surf: pygame.Surface | None = None

    def handle_event(self, event: pygame.event.Event) -> bool:
        """Dismiss the banner on click; return True if the click was consumed."""
        if self.dismissed or self._rect is None:
            return False
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self._rect.collidepoint(event.pos):
                self.dismissed = True
                return True
        return False

    def draw(self, surface: pygame.Surface) -> None:
        if self.dismissed or not self.warnings:
            return
        sw, sh = surface.get_size()
        if self._font is None or self._cached_size != (sw, sh):
            self._cached_size = (sw, sh)
            scale = min(sw / 1280, sh / 720)
            self._font = pygame.font.SysFont("Segoe UI", max(14, int(16 * scale)), bold=True)
            self._hint_font = pygame.font.SysFont("Segoe UI", max(11, int(12 * scale)))
            self._lines = [
                self._font.render(msg, True, (254, 243, 199)) for msg in self.warnings
            ]
            self._hint_surf = self._hint_font.render(
                "Click this banner to dismiss", True, (253, 224, 71)
            )
            pad_x = max(16, int(20 * scale))
            pad_y = max(10, int(12 * scale))
            w = max(
                max(line.get_width() for line in self._lines),
                self._hint_surf.get_width(),
            ) + pad_x * 2
            w = min(w, sw - 24)
            h = sum(line.get_height() for line in self._lines) + self._hint_surf.get_height() + pad_y * 2 + 6
            self._rect = pygame.Rect((sw - w) // 2, 12, w, h)
            self._bg = pygame.Surface((w, h), pygame.SRCALPHA)
            self._bg.fill((120, 32, 24, 230))
            pygame.draw.rect(
                self._bg, (248, 113, 113, 230), self._bg.get_rect(), width=2, border_radius=8
            )

        assert self._bg is not None and self._rect is not None and self._hint_surf is not None
        surface.blit(self._bg, self._rect.topleft)
        cy = self._rect.top + 10
        for line in self._lines:
            lx = self._rect.centerx - line.get_width() // 2
            surface.blit(line, (lx, cy))
            cy += line.get_height()
        hx = self._rect.centerx - self._hint_surf.get_width() // 2
        surface.blit(self._hint_surf, (hx, cy + 4))


# =====================================================================
# App
# =====================================================================

class App:
    """Top-level application: owns the display and screen manager."""

    def __init__(self) -> None:
        _enable_windows_dpi_awareness()
        _set_windows_app_user_model_id()
        pygame.init()
        _set_window_icon()

        # full-screen display (uses physical pixels when DPI awareness is active)
        self.screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
        pygame.display.set_caption(WINDOW_TITLE)
        pygame.mouse.set_visible(True)

        startup_warnings = _run_startup_self_check()
        logger.info("Display surface: %dx%d", self.screen.get_width(), self.screen.get_height())

        self.clock = pygame.time.Clock()
        self._perf_overlay = _PerfOverlay()
        self._warning_banner = _StartupWarningBanner(startup_warnings)
        self.running: bool = True
        self._volume_percent = 70
        self._sound_manager = SoundManager(volume=self._volume_percent / 100.0)
        self._premove_enabled: bool = True
        self._selected_opponent: dict[str, object] | None = None
        self._settings_return_screen = "dashboard"
        self._sync_menu_music("dashboard")

        # screen registry — maps names to factory callables
        self._screen_factories: dict[str, callable] = {
            "dashboard": lambda: DashboardScreen(self.screen),
            "opponent_select": lambda: OpponentSelectScreen(self.screen),
            "game": lambda: GameScreen(
                self.screen,
                self._get_selected_opponent(),
                premove_enabled=self._premove_enabled,
                sound_manager=self._sound_manager,
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
        # Pre-decode opponent portraits off the main thread while on the Dashboard
        OpponentSelectScreen.preload_assets()

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
            filtered_events: list[pygame.event.Event] = []
            for event in events:
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_F3:
                    self._perf_overlay.toggle()
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    if not self._current_screen.captures_escape():
                        self._navigate("dashboard")
                elif self._warning_banner.handle_event(event):
                    continue
                filtered_events.append(event)

            # delegate to current screen
            self._current_screen.handle_events(filtered_events)
            self._current_screen.update(dt)
            self._current_screen.draw()
            self._warning_banner.draw(self.screen)
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
            resolved_target = target
            self._current_screen = factory()
        else:
            # unknown screen — fall back to dashboard
            resolved_target = "dashboard"
            self._current_screen = self._screen_factories["dashboard"]()

        self._sync_menu_music(resolved_target)

    def _should_play_menu_music(self, screen_name: str) -> bool:
        """Menu music plays on main menu screens and stops for time-control/match."""
        if screen_name in ("dashboard", "opponent_select"):
            return True
        if screen_name == "settings":
            return self._settings_return_screen != "game"
        return False

    def _sync_menu_music(self, screen_name: str) -> None:
        """Start or fade out background menu music based on the active screen."""
        if self._should_play_menu_music(screen_name):
            self._sound_manager.start_music(fade_ms=600)
        else:
            self._sound_manager.stop_music(fade_ms=400)

    def _get_selected_opponent(self) -> dict[str, object]:
        if self._selected_opponent is None:
            raise RuntimeError("No opponent has been selected.")
        return self._selected_opponent

    def _set_volume(self, volume: int) -> None:
        """Keep the settings value when the settings screen is reopened."""
        self._volume_percent = volume
        self._sound_manager.set_volume(volume / 100.0)

    def _set_premove_enabled(self, enabled: bool) -> None:
        """Keep the premove setting across screens."""
        self._premove_enabled = bool(enabled)

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    def _shutdown(self) -> None:
        """Close the active screen (stopping any bot worker), then quit."""
        self._current_screen.close()
        self._sound_manager.stop_music()
        pygame.quit()
        sys.exit()


# ── run ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    App().run()
  