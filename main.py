"""Entry point for the hand-controlled mouse.

Control your computer's cursor with one hand in front of the webcam:

    - Move      : point with the index finger.
    - Click/Drag: pinch thumb + index (quick = click, hold = drag).
    - Right clic: pinch thumb + middle finger.
    - Quit      : press 'q' or ESC in the preview window, or Ctrl+C here.

Run with:  python main.py

On macOS the program needs Camera and Accessibility permissions
(System Settings -> Privacy & Security) for the terminal or IDE it runs from.
"""

from __future__ import annotations

import logging
import sys

from hand_mouse.app import HandMouseApp
from hand_mouse.camera import CameraError
from hand_mouse.config import AppConfig

logger = logging.getLogger("hand_mouse")


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )


def main() -> int:
    """Build the application from default configuration and run it.

    Returns a process exit code: 0 on a clean stop, 1 on a camera failure.
    """
    _configure_logging()
    try:
        HandMouseApp(AppConfig()).run()
    except CameraError as error:
        logger.error("Camera error: %s", error)
        return 1
    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
