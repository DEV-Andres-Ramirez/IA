"""Central, typed configuration for the hand-controlled mouse.

All tunable values live here so the rest of the code is free of magic numbers.
Every section is a frozen dataclass: configuration is read-only at runtime.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class CameraConfig:
    """Webcam capture settings."""

    device_index: int = 0
    frame_width: int = 1280
    frame_height: int = 720
    # Mirror the image so moving the hand right moves the cursor right.
    flip_horizontal: bool = True


@dataclass(frozen=True)
class TrackingConfig:
    """MediaPipe Hands detection settings."""

    max_hands: int = 1
    min_detection_confidence: float = 0.7
    min_tracking_confidence: float = 0.6


@dataclass(frozen=True)
class ControlConfig:
    """How hand motion and pinches are translated into mouse actions."""

    # Fraction of the frame (per side) excluded from the mapping so the user
    # reaches every screen edge without moving the hand out of view.
    active_region_margin: float = 0.15

    # Exponential smoothing factor for cursor motion: 0 freezes the cursor,
    # 1 disables smoothing. Lower values are steadier but feel heavier.
    smoothing: float = 0.4

    # Pinch distance thresholds, normalised by hand size so they are
    # independent of how close the hand is to the camera. Two thresholds give
    # hysteresis: a pinch must clearly open before it can close again, which
    # removes flicker around the boundary.
    pinch_close_threshold: float = 0.45
    pinch_open_threshold: float = 0.65

    # Minimum seconds between two consecutive right clicks.
    right_click_cooldown: float = 0.6

    def __post_init__(self) -> None:
        if not 0.0 <= self.active_region_margin < 0.5:
            raise ValueError("active_region_margin must be in [0.0, 0.5).")
        if not 0.0 < self.smoothing <= 1.0:
            raise ValueError("smoothing must be in (0.0, 1.0].")
        if self.pinch_open_threshold <= self.pinch_close_threshold:
            raise ValueError(
                "pinch_open_threshold must be greater than pinch_close_threshold."
            )


@dataclass(frozen=True)
class AppConfig:
    """Top-level configuration aggregating every section."""

    camera: CameraConfig = field(default_factory=CameraConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    control: ControlConfig = field(default_factory=ControlConfig)

    show_window: bool = True
    window_name: str = "Hand Mouse"
    # Keys that stop the application: 'q' or ESC.
    quit_keys: tuple[int, ...] = (ord("q"), 27)
