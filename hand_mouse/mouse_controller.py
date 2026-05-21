"""Maps recognised gestures onto operating-system mouse actions."""

from __future__ import annotations

import logging
import sys

import pyautogui

from .config import ControlConfig
from .gesture import HandGestures
from .hand_tracker import Point

logger = logging.getLogger(__name__)

# Hand control routinely pushes the cursor into a screen corner. PyAutoGUI's
# fail-safe treats that as an abort signal, which would crash the program, so
# it is disabled deliberately; the user quits with the keyboard instead.
# PAUSE is zeroed so moveTo and button calls do not sleep between frames.
pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0.0


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _double_click_macos(x: float, y: float) -> None:
    """Post a native double-click at (x, y) on macOS.

    The decisive detail is ``kCGMouseEventClickState``: macOS only treats the
    second press as a double-click when that field is set to 2. pyautogui
    leaves it unset, which is why its doubleClick() often registers as two
    separate single clicks instead.
    """
    import Quartz

    position = (x, y)
    button = Quartz.kCGMouseButtonLeft

    def post(event_type: int, click_state: int) -> None:
        event = Quartz.CGEventCreateMouseEvent(
            None, event_type, position, button
        )
        Quartz.CGEventSetIntegerValueField(
            event, Quartz.kCGMouseEventClickState, click_state
        )
        Quartz.CGEventPost(Quartz.kCGHIDEventTap, event)

    # First click of the pair, then the second one marked as click-state 2.
    post(Quartz.kCGEventLeftMouseDown, 1)
    post(Quartz.kCGEventLeftMouseUp, 1)
    post(Quartz.kCGEventLeftMouseDown, 2)
    post(Quartz.kCGEventLeftMouseUp, 2)


class MouseController:
    """Owns the cursor: maps hand position to the screen, smooths motion,
    and turns pinches into button events.

    The left pinch is mapped to button-down on press and button-up on
    release. A quick pinch therefore becomes a click, while a held pinch
    becomes a drag -- one rule covers both.
    """

    def __init__(self, config: ControlConfig) -> None:
        self._config = config
        self._screen_width, self._screen_height = pyautogui.size()

        self._smoothed: Point | None = None
        self._left_button_down = False
        logger.info(
            "Screen resolution detected: %dx%d.",
            self._screen_width,
            self._screen_height,
        )

    def update(self, gestures: HandGestures) -> None:
        """Apply one frame of gestures to the operating-system cursor."""
        self._move_cursor(gestures.cursor_target)
        self._apply_left_pinch(gestures.left_pinch)
        self._apply_double_click(gestures.double_click)

    def release(self) -> None:
        """Release any held button and drop smoothing state.

        Called when the hand leaves the frame or the app exits, so a drag is
        never left stuck down.
        """
        if self._left_button_down:
            pyautogui.mouseUp()
            self._left_button_down = False
            logger.debug("Left button released (tracking lost).")
        self._smoothed = None

    def _move_cursor(self, target: Point) -> None:
        screen_point = self._map_to_screen(target)
        self._smoothed = self._smooth(screen_point)
        pyautogui.moveTo(self._smoothed.x, self._smoothed.y)

    def _apply_left_pinch(self, pinching: bool) -> None:
        if pinching and not self._left_button_down:
            pyautogui.mouseDown()
            self._left_button_down = True
            logger.debug("Left button down.")
        elif not pinching and self._left_button_down:
            pyautogui.mouseUp()
            self._left_button_down = False
            logger.debug("Left button up.")

    def _apply_double_click(self, should_fire: bool) -> None:
        # `should_fire` is already a one-shot pulse from the recogniser, which
        # also enforces the re-arm and cooldown rules, so no extra state here.
        if not should_fire:
            return
        self._perform_double_click()
        logger.info("Double click performed.")

    def _perform_double_click(self) -> None:
        """Double-click at the current cursor position.

        macOS uses a native Quartz event because pyautogui's doubleClick()
        posts two independent clicks that macOS frequently reads as two
        separate single clicks rather than a double-click.
        """
        if sys.platform == "darwin" and self._smoothed is not None:
            try:
                _double_click_macos(self._smoothed.x, self._smoothed.y)
                return
            except Exception as error:  # fall back if Quartz misbehaves
                logger.warning(
                    "Native double-click failed, using fallback: %s", error
                )
        pyautogui.doubleClick()

    def _map_to_screen(self, target: Point) -> Point:
        """Map a normalised hand position to absolute screen coordinates.

        Only the central active region of the frame is used, so the user can
        reach every screen edge without the hand leaving the camera view.
        """
        margin = self._config.active_region_margin
        usable_span = 1.0 - 2.0 * margin

        normalised_x = _clamp((target.x - margin) / usable_span, 0.0, 1.0)
        normalised_y = _clamp((target.y - margin) / usable_span, 0.0, 1.0)
        return Point(
            normalised_x * self._screen_width,
            normalised_y * self._screen_height,
        )

    def _smooth(self, target: Point) -> Point:
        """Exponential moving average to damp camera jitter."""
        if self._smoothed is None:
            return target
        alpha = self._config.smoothing
        return Point(
            self._smoothed.x + alpha * (target.x - self._smoothed.x),
            self._smoothed.y + alpha * (target.y - self._smoothed.y),
        )
