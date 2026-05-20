"""Translates raw hand landmarks into high-level gestures.

The recogniser is stateful on purpose: pinch detection uses hysteresis, so it
needs to remember whether each pinch was open or closed on the previous frame.
"""

from __future__ import annotations

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
    """

    cursor_target: Point
    left_pinch: bool


class GestureRecognizer:
    """Maps landmarks to gestures, keeping pinch state stable across frames."""

    def __init__(self, config: ControlConfig) -> None:
        self._config = config
        self._left_state = PinchState.OPEN

    def recognize(self, hand: HandLandmarks) -> HandGestures:
        """Interpret one hand. Cursor follows the index fingertip."""
        thumb_index_gap = hand[Landmark.THUMB_TIP].distance_to(
            hand[Landmark.INDEX_TIP]
        )
        self._left_state = self._next_pinch_state(
            self._left_state, thumb_index_gap, hand.palm_size
        )
        return HandGestures(
            cursor_target=hand[Landmark.INDEX_TIP],
            left_pinch=self._left_state is PinchState.CLOSED,
        )

    def reset(self) -> None:
        """Forget pinch state; call when the hand leaves the frame."""
        self._left_state = PinchState.OPEN

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
