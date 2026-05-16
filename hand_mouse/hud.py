"""Heads-up display drawn on the camera-preview window.

Pure rendering: this module reads state and draws it, it never changes it.
"""

from __future__ import annotations

import cv2
import numpy as np

from .gesture import HandGestures
from .hand_tracker import HandLandmarks

# Landmark index pairs that form the hand skeleton, grouped by finger.
_HAND_CONNECTIONS: tuple[tuple[int, int], ...] = (
    (0, 1), (1, 2), (2, 3), (3, 4),          # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),          # index
    (5, 9), (9, 10), (10, 11), (11, 12),     # middle
    (9, 13), (13, 14), (14, 15), (15, 16),   # ring
    (13, 17), (17, 18), (18, 19), (19, 20),  # pinky
    (0, 17),                                 # palm base
)

# Colours in BGR.
_WHITE = (255, 255, 255)
_GREY = (160, 160, 160)
_GREEN = (0, 220, 0)
_RED = (40, 40, 235)
_CYAN = (230, 230, 0)
_BLACK = (0, 0, 0)

_FONT = cv2.FONT_HERSHEY_SIMPLEX


def draw_overlay(
    frame: np.ndarray,
    hand: HandLandmarks | None,
    gestures: HandGestures | None,
    fps: float,
    active_region_margin: float,
) -> np.ndarray:
    """Draw the active region, hand skeleton, cursor marker and status panel.

    Mutates and returns `frame` for convenient chaining.
    """
    height, width = frame.shape[:2]

    _draw_active_region(frame, width, height, active_region_margin)
    if hand is not None:
        _draw_hand_skeleton(frame, hand, width, height)
    if hand is not None and gestures is not None:
        _draw_cursor_marker(frame, gestures, width, height)
    _draw_status_panel(frame, hand, gestures, fps)
    return frame


def _draw_active_region(
    frame: np.ndarray, width: int, height: int, margin: float
) -> None:
    """Outline the part of the frame mapped to the screen."""
    top_left = (int(margin * width), int(margin * height))
    bottom_right = (int((1.0 - margin) * width), int((1.0 - margin) * height))
    cv2.rectangle(frame, top_left, bottom_right, _GREY, 1)


def _draw_hand_skeleton(
    frame: np.ndarray, hand: HandLandmarks, width: int, height: int
) -> None:
    pixels = [(int(p.x * width), int(p.y * height)) for p in hand.points]
    for start, end in _HAND_CONNECTIONS:
        cv2.line(frame, pixels[start], pixels[end], _WHITE, 2)
    for pixel in pixels:
        cv2.circle(frame, pixel, 4, _CYAN, cv2.FILLED)


def _draw_cursor_marker(
    frame: np.ndarray, gestures: HandGestures, width: int, height: int
) -> None:
    """Mark the index fingertip; colour reflects the active gesture."""
    centre = (
        int(gestures.cursor_target.x * width),
        int(gestures.cursor_target.y * height),
    )
    if gestures.left_pinch:
        colour = _RED
    elif gestures.right_pinch:
        colour = _CYAN
    else:
        colour = _GREEN
    cv2.circle(frame, centre, 14, colour, 2)
    cv2.circle(frame, centre, 3, colour, cv2.FILLED)


def _gesture_label(
    hand: HandLandmarks | None, gestures: HandGestures | None
) -> tuple[str, tuple[int, int, int]]:
    if hand is None or gestures is None:
        return "NO HAND", _GREY
    if gestures.left_pinch:
        return "LEFT CLICK / DRAG", _RED
    if gestures.right_pinch:
        return "RIGHT CLICK", _CYAN
    return "MOVE", _GREEN


def _draw_status_panel(
    frame: np.ndarray,
    hand: HandLandmarks | None,
    gestures: HandGestures | None,
    fps: float,
) -> None:
    """Semi-transparent panel with FPS, current gesture and a usage hint."""
    label, label_colour = _gesture_label(hand, gestures)
    lines = [
        (f"FPS: {fps:4.1f}", _WHITE),
        (f"Gesture: {label}", label_colour),
        ("Index = move  |  Pinch index = click/drag", _GREY),
        ("Pinch middle = right click  |  'q'/ESC = quit", _GREY),
    ]

    panel_width, panel_height = 430, 24 * len(lines) + 16
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (panel_width, panel_height), _BLACK, cv2.FILLED)
    cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)

    y = 28
    for text, colour in lines:
        cv2.putText(frame, text, (12, y), _FONT, 0.55, colour, 1, cv2.LINE_AA)
        y += 24
