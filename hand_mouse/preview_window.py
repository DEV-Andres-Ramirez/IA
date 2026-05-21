"""The camera-preview window, pinned above every other window.

OpenCV's own ``WND_PROP_TOPMOST`` flag only lifts the window above ordinary
windows; on macOS it does not rise above an application in full-screen mode.
Native macOS full-screen moves the app onto its own *space* (the side-to-side
slide), and three native settings -- which this module applies through PyObjC
-- are needed for the preview to follow it there:

* **accessory activation policy** -- a regular Dock application's windows are
  confined to their own spaces; an accessory application's windows are allowed
  onto another app's full-screen space (this is how menu-bar apps behave);
* **canJoinAllSpaces collection behaviour** -- places the window on every
  space, full-screen spaces included;
* **the screen-shielding window level** -- stacks it above all other content.

The level and collection behaviour are re-applied every frame, since the user
may switch an app to full-screen at any time. Every macOS call is best-effort:
if PyObjC is missing the preview simply behaves as an ordinary always-on-top
window and the application keeps running.
"""

from __future__ import annotations

import logging
import sys

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class PreviewWindow:
    """Owns the OpenCV preview window and keeps it above every other window."""

    def __init__(self, window_name: str) -> None:
        self._name = window_name
        self._is_open = False
        self._macos_topmost = (
            _MacOSTopmost() if sys.platform == "darwin" else None
        )

    def open(self) -> None:
        """Create the window and request always-on-top behaviour."""
        cv2.namedWindow(self._name, cv2.WINDOW_AUTOSIZE)
        self._apply_opencv_topmost()
        self._is_open = True
        logger.info("Preview window '%s' created (always on top).", self._name)

    def show(self, frame: np.ndarray) -> None:
        """Display one frame and re-assert the always-on-top behaviour."""
        cv2.imshow(self._name, frame)
        if self._macos_topmost is not None:
            self._macos_topmost.apply()

    def quit_requested(self, quit_keys: tuple[int, ...]) -> bool:
        """Pump the GUI event loop and report whether a quit key was pressed.

        OpenCV only receives keystrokes while its window holds focus, so the
        preview must be the focused window for this to detect the key.
        """
        key = cv2.waitKey(1) & 0xFF
        return key in quit_keys

    def close(self) -> None:
        """Destroy the window if it is open."""
        if not self._is_open:
            return
        cv2.destroyWindow(self._name)
        cv2.waitKey(1)  # let the GUI event loop process the destruction
        self._is_open = False
        logger.info("Preview window closed.")

    def _apply_opencv_topmost(self) -> None:
        """Set OpenCV's cross-platform always-on-top flag (best-effort)."""
        try:
            cv2.setWindowProperty(self._name, cv2.WND_PROP_TOPMOST, 1.0)
        except (cv2.error, AttributeError) as error:
            logger.warning("Could not set the topmost window flag: %s", error)


def _macos_shielding_level() -> int:
    """Cocoa window level above every window, full-screen apps included.

    ``CGShieldingWindowLevel`` is the level macOS uses to shield the screen.
    It sits above all application content -- browsers in full-screen mode too
    -- but below the mouse cursor, so the cursor stays visible.
    """
    try:
        import Quartz

        return int(Quartz.CGShieldingWindowLevel())
    except Exception:
        # Assistive-technology level: still above full-screen apps.
        return 1500


class _MacOSTopmost:
    """Keeps the process's windows above everything else on macOS, including
    other applications' full-screen spaces.

    See the module docstring for why all three native settings are required.
    Degrades to a no-op when PyObjC is unavailable.
    """

    def __init__(self) -> None:
        self._app = None
        self._behavior = 0
        self._level = 0
        self._accessory_policy = 0
        self._configured = False
        self._error_logged = False
        try:
            from AppKit import (
                NSApplication,
                NSApplicationActivationPolicyAccessory,
                NSWindowCollectionBehaviorCanJoinAllSpaces,
                NSWindowCollectionBehaviorFullScreenAuxiliary,
                NSWindowCollectionBehaviorStationary,
            )
        except ImportError as error:
            logger.warning(
                "PyObjC unavailable; preview cannot float over full-screen "
                "apps: %s",
                error,
            )
            return
        self._app = NSApplication.sharedApplication()
        self._accessory_policy = NSApplicationActivationPolicyAccessory
        # canJoinAllSpaces puts the window on every space, full-screen spaces
        # included; stationary keeps it steady while spaces are switched.
        self._behavior = (
            NSWindowCollectionBehaviorCanJoinAllSpaces
            | NSWindowCollectionBehaviorFullScreenAuxiliary
            | NSWindowCollectionBehaviorStationary
        )
        self._level = _macos_shielding_level()

    def apply(self) -> None:
        """Pin every window of this process on top (best-effort, per frame)."""
        if self._app is None:
            return
        try:
            windows = list(self._app.windows())
            for window in windows:
                window.setLevel_(self._level)
                window.setCollectionBehavior_(self._behavior)
            if not self._configured and windows:
                self._configure(windows)
        except Exception as error:  # never let this break the render loop
            if not self._error_logged:
                self._error_logged = True
                logger.warning("Could not pin the preview on top: %s", error)

    def _configure(self, windows: list) -> None:
        """One-time setup once the preview window actually exists.

        OpenCV creates the window under a regular (Dock) activation policy;
        switching to an accessory policy afterwards is what allows the window
        onto another application's full-screen space.
        """
        self._configured = True
        self._app.setActivationPolicy_(self._accessory_policy)
        for window in windows:
            window.orderFrontRegardless()
        logger.info(
            "Preview pinned above full-screen apps "
            "(accessory app, level %d, %d window(s)).",
            self._level,
            len(windows),
        )
