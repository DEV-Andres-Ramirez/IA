"""Hand detection on top of MediaPipe Hands.

This module is the only place that knows about MediaPipe. It converts the
library's raw output into small, immutable value objects (`Point`,
`HandLandmarks`) so the rest of the application stays decoupled from it.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from enum import IntEnum

import cv2
import mediapipe as mp
import numpy as np

from .config import TrackingConfig

logger = logging.getLogger(__name__)


class Landmark(IntEnum):
    """MediaPipe hand-landmark indices used by this application.

    MediaPipe returns 21 landmarks per hand; only the ones we actually read
    are named here. See the MediaPipe Hands documentation for the full map.
    """

    WRIST = 0
    THUMB_TIP = 4
    INDEX_MCP = 5  # knuckle at the base of the index finger
    INDEX_PIP = 6  # middle joint of the index finger
    INDEX_TIP = 8
    MIDDLE_PIP = 10
    MIDDLE_TIP = 12
    RING_PIP = 14
    RING_TIP = 16
    PINKY_PIP = 18
    PINKY_TIP = 20


@dataclass(frozen=True)
class Point:
    """A 2D point in normalised image coordinates, each axis in [0, 1]."""

    x: float
    y: float

    def distance_to(self, other: "Point") -> float:
        return math.hypot(self.x - other.x, self.y - other.y)


@dataclass(frozen=True)
class HandLandmarks:
    """Normalised landmark positions for a single detected hand."""

    points: tuple[Point, ...]

    def __getitem__(self, landmark: Landmark) -> Point:
        return self.points[landmark]

    @property
    def palm_size(self) -> float:
        """Wrist-to-index-knuckle distance.

        Used as a scale reference so pinch distances can be normalised: the
        same gesture yields the same value whether the hand is near or far
        from the camera.
        """
        return self[Landmark.WRIST].distance_to(self[Landmark.INDEX_MCP])


class HandTracker:
    """Detects a single hand per frame and exposes its landmarks."""

    def __init__(self, config: TrackingConfig) -> None:
        self._hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=config.max_hands,
            min_detection_confidence=config.min_detection_confidence,
            min_tracking_confidence=config.min_tracking_confidence,
        )
        logger.info("MediaPipe Hands initialised.")

    def detect(self, frame_bgr: np.ndarray) -> HandLandmarks | None:
        """Return landmarks for the most prominent hand, or None if absent."""
        # MediaPipe expects RGB; marking the buffer read-only lets it skip a
        # defensive copy, which is a measurable speed-up per frame.
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False
        result = self._hands.process(frame_rgb)

        if not result.multi_hand_landmarks:
            return None
        raw_hand = result.multi_hand_landmarks[0]
        points = tuple(Point(lm.x, lm.y) for lm in raw_hand.landmark)
        return HandLandmarks(points=points)

    def close(self) -> None:
        """Release MediaPipe resources."""
        self._hands.close()
        logger.info("MediaPipe Hands closed.")
