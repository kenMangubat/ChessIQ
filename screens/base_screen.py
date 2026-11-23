"""BaseScreen — abstract parent for every screen in ChessIQ.

Subclasses must override handle_events(), update(), and draw().
The screen manager in main.py calls these three methods each frame.
"""

import pygame


class BaseScreen:
    """Base class for all application screens.

    Parameters
    ----------
    screen : pygame.Surface
        The main display surface (full-screen).
    """

    def __init__(self, screen: pygame.Surface) -> None:
        self.screen = screen
        self.width = screen.get_width()
        self.height = screen.get_height()
        self.next_screen: str | None = None  # set by subclass to request navigation

    # ------------------------------------------------------------------
    # Interface — subclasses override these
    # ------------------------------------------------------------------

    def handle_events(self, events: list[pygame.event.Event]) -> None:
        """Process input events for this screen."""
        raise NotImplementedError

    def update(self, dt: float) -> None:
        """Advance animations / state by *dt* seconds."""
        raise NotImplementedError

    def draw(self) -> None:
        """Render the screen to *self.screen*."""
        raise NotImplementedError

    def captures_escape(self) -> bool:
        """Return True to handle Esc locally instead of returning to the dashboard."""
        return False

    def close(self) -> None:
        """Release resources (threads, processes) when the screen is left."""
