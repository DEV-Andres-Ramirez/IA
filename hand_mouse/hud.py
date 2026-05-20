"""Heads-up display drawn on the camera-preview window.

Pure rendering: this module reads state and draws it, it never changes it.
"""

from __future__ import annotations

import cv2
import numpy as np

from .gesture import HandGestures
from .hand_tracker import HandLandmarks, Point

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

# Neon palette for the active-region border. Its colour reacts to the
# fingertip: bright green deep inside, shading through yellow to red near the
# edge, and dark purple once the fingertip leaves the region entirely.
_NEON_GREEN = (0, 255, 0)
_NEON_YELLOW = (0, 255, 255)
_NEON_RED = (0, 0, 255)
_DARK_PURPLE = (110, 0, 85)

_FONT = cv2.FONT_HERSHEY_SIMPLEX

# Line thickness, in pixels, of the rectangle that marks the screen-mapped area.
_ACTIVE_REGION_THICKNESS = 4


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
    cursor_target = gestures.cursor_target if gestures is not None else None

    _draw_active_region(
        frame, width, height, active_region_margin, cursor_target
    )
    if hand is not None:
        _draw_hand_skeleton(frame, hand, width, height)
    if hand is not None and gestures is not None:
        _draw_cursor_marker(frame, gestures, width, height)
    _draw_status_panel(frame, hand, gestures, fps)
    return frame


def _draw_active_region(
    frame: np.ndarray,
    width: int,
    height: int,
    margin: float,
    cursor_target: Point | None,
) -> None:
    """Outline the part of the frame mapped to the screen.

    Drawn as a thick rectangle whose neon colour reacts to the fingertip:
    bright green while it stays safely inside, shading through yellow to red
    as it nears the edge, and dark purple once it leaves the region.
    """
    top_left = (int(margin * width), int(margin * height))
    bottom_right = (int((1.0 - margin) * width), int((1.0 - margin) * height))
    colour = _active_region_color(cursor_target, margin)
    cv2.rectangle(
        frame, top_left, bottom_right, colour, _ACTIVE_REGION_THICKNESS
    )


def _active_region_color(
    cursor_target: Point | None, margin: float
) -> tuple[int, int, int]:
    """Pick the border colour from how deep the fingertip sits in the region.

    Returns neon green at the centre, fading through yellow to red as the
    fingertip nears the boundary, and dark purple once it leaves the region.
    """
    if cursor_target is None:
        return _NEON_GREEN

    half_span = (1.0 - 2.0 * margin) / 2.0
    if half_span <= 0.0:
        return _NEON_GREEN

    # Distance from the fingertip to the nearest region edge on each axis.
    inset_x = min(cursor_target.x - margin, (1.0 - margin) - cursor_target.x)
    inset_y = min(cursor_target.y - margin, (1.0 - margin) - cursor_target.y)
    # depth: 1.0 at the centre, 0.0 on the border, negative once outside.
    depth = min(inset_x, inset_y) / half_span

    if depth < 0.0:
        return _DARK_PURPLE
    if depth >= 0.5:
        return _lerp_color(_NEON_YELLOW, _NEON_GREEN, (depth - 0.5) / 0.5)
    return _lerp_color(_NEON_RED, _NEON_YELLOW, depth / 0.5)


def _lerp_color(
    start: tuple[int, int, int],
    end: tuple[int, int, int],
    fraction: float,
) -> tuple[int, int, int]:
    """Linearly interpolate two BGR colours; fraction is clamped to [0, 1]."""
    fraction = max(0.0, min(1.0, fraction))
    return (
        int(round(start[0] + (end[0] - start[0]) * fraction)),
        int(round(start[1] + (end[1] - start[1]) * fraction)),
        int(round(start[2] + (end[2] - start[2]) * fraction)),
    )


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
    colour = _RED if gestures.left_pinch else _GREEN
    cv2.circle(frame, centre, 14, colour, 2)
    cv2.circle(frame, centre, 3, colour, cv2.FILLED)


def _gesture_label(
    hand: HandLandmarks | None, gestures: HandGestures | None
) -> tuple[str, tuple[int, int, int]]:
    if hand is None or gestures is None:
        return "NO HAND", _GREY
    if gestures.left_pinch:
        return "LEFT CLICK / DRAG", _RED
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
        ("'q' / ESC = quit", _GREY),
    ]

    panel_width, panel_height = 430, 24 * len(lines) + 16
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (panel_width, panel_height), _BLACK, cv2.FILLED)
    cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)

    y = 28
    for text, colour in lines:
        cv2.putText(frame, text, (12, y), _FONT, 0.55, colour, 1, cv2.LINE_AA)
        y += 24
