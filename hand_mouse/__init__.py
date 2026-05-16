"""Hand-controlled mouse.

Turns a webcam and one hand into a pointing device: move the cursor with the
index finger, pinch thumb + index to click or drag, pinch thumb + middle to
right click.

The package is split into single-responsibility modules:

    config            tunable parameters (dataclasses, no magic numbers)
    camera            context-managed webcam capture
    hand_tracker      MediaPipe wrapper -> normalised hand landmarks
    gesture           landmarks -> high-level gestures (with hysteresis)
    mouse_controller  gestures -> operating-system mouse actions
    hud               camera-preview overlay
    app               orchestration of the capture/track/act loop
"""

__all__ = ["__version__"]
__version__ = "1.0.0"
