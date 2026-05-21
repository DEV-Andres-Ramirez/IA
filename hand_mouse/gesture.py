"""Translates raw hand landmarks into high-level gestures.

The recogniser is stateful on purpose: pinch detection uses hysteresis, so it
needs to remember whether each pinch was open or closed on the previous frame.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum, auto

from .config import ControlConfig
from .hand_tracker import HandLandmarks, Landmark, Point


class PinchState(Enum):
    OPEN = auto()
    CLOSED = auto()


@dataclass(frozen=True)
class HandGestures:
    """High-level interpretation of a single video frame.

    Attributes:
        cursor_target: Normalised position the cursor should track.
        left_pinch:    Thumb-index pinch held -> left button down (click/drag).
        double_click:  True on the single frame a double-click should fire.
        two_gesture:   True while the index+middle "two" sign is being held;
                       used for on-screen feedback.
    """

    cursor_target: Point
    left_pinch: bool
    double_click: bool
    two_gesture: bool


class GestureRecognizer:
    """Maps landmarks to gestures, keeping pinch state stable across frames."""

    def __init__(self, config: ControlConfig) -> None:
        self._config = config
        self._left_state = PinchState.OPEN
        self._two_gesture_active = False
        self._last_double_click_at = 0.0

    def recognize(self, hand: HandLandmarks) -> HandGestures:
        """Interpret one hand. Cursor follows the index fingertip."""
        thumb_index_gap = hand[Landmark.THUMB_TIP].distance_to(
            hand[Landmark.INDEX_TIP]
        )
        self._left_state = self._next_pinch_state(
            self._left_state, thumb_index_gap, hand.palm_size
        )
        showing_two = _is_two_gesture(hand)
        return HandGestures(
            cursor_target=hand[Landmark.INDEX_TIP],
            left_pinch=self._left_state is PinchState.CLOSED,
            double_click=self._detect_double_click(showing_two),
            two_gesture=showing_two,
        )

    def reset(self) -> None:
        """Forget transient gesture state; call when the hand leaves the frame.

        The double-click cooldown timer is kept on purpose, so the gesture
        cannot be repeated quickly by waving the hand out of view and back.
        """
        self._left_state = PinchState.OPEN
        self._two_gesture_active = False

    def _detect_double_click(self, showing_two: bool) -> bool:
        """Return True on the single frame a double-click should be performed.

        The index+middle "two" gesture triggers it, but only on the rising
        edge: the hand must drop the gesture and form it again to fire another
        double-click. A cooldown additionally blocks repeats within
        `double_click_cooldown` seconds, guarding against accidents.
        """
        rising_edge = showing_two and not self._two_gesture_active
        self._two_gesture_active = showing_two
        if not rising_edge:
            return False

        now = time.monotonic()
        if now - self._last_double_click_at < self._config.double_click_cooldown:
            return False
        self._last_double_click_at = now
        return True

    def _next_pinch_state(
        self, current: PinchState, raw_gap: float, palm_size: float
    ) -> PinchState:
        """Advance one pinch's state machine using hysteresis thresholds."""
        if palm_size <= 0.0:
            return PinchState.OPEN

        normalised_gap = raw_gap / palm_size
        if (
            current is PinchState.OPEN
            and normalised_gap < self._config.pinch_close_threshold
        ):
            return PinchState.CLOSED
        if (
            current is PinchState.CLOSED
            and normalised_gap > self._config.pinch_open_threshold
        ):
            return PinchState.OPEN
        return current


def _is_finger_extended(
    hand: HandLandmarks, tip: Landmark, pip: Landmark
) -> bool:
    """True when a finger is straightened out rather than folded.

    A straight finger reaches its tip farther from the wrist than its middle
    (PIP) joint; a folded finger curls the tip back toward the palm. Measuring
    against the wrist keeps the test independent of how the hand is rotated.
    """
    wrist = hand[Landmark.WRIST]
    return wrist.distance_to(hand[tip]) > wrist.distance_to(hand[pip])


def _is_two_gesture(hand: HandLandmarks) -> bool:
    """True for the "two" / V sign: index and middle up, ring and pinky down."""
    return (
        _is_finger_extended(hand, Landmark.INDEX_TIP, Landmark.INDEX_PIP)
        and _is_finger_extended(hand, Landmark.MIDDLE_TIP, Landmark.MIDDLE_PIP)
        and not _is_finger_extended(hand, Landmark.RING_TIP, Landmark.RING_PIP)
        and not _is_finger_extended(hand, Landmark.PINKY_TIP, Landmark.PINKY_PIP)
    )
