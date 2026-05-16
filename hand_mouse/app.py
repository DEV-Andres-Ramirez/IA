"""Application orchestration: the capture -> track -> recognise -> act loop."""

from __future__ import annotations

import logging
import time

import cv2

from .camera import Camera
from .config import AppConfig
from .gesture import GestureRecognizer, HandGestures
from .hand_tracker import HandTracker
from .hud import draw_overlay
from .mouse_controller import MouseController

logger = logging.getLogger(__name__)


class _FpsMeter:
    """Exponentially smoothed frames-per-second estimate."""

    def __init__(self, smoothing: float = 0.9) -> None:
        self._smoothing = smoothing
        self._last_tick = time.monotonic()
        self._fps = 0.0

    def tick(self) -> float:
        now = time.monotonic()
        elapsed = now - self._last_tick
        self._last_tick = now
        if elapsed > 0.0:
            instant_fps = 1.0 / elapsed
            self._fps = (
                self._smoothing * self._fps
                + (1.0 - self._smoothing) * instant_fps
            )
        return self._fps


class HandMouseApp:
    """Wires the components together and runs the main loop.

    Responsibilities are delegated: this class only decides *when* each
    component runs and how the program shuts down cleanly.
    """

    def __init__(self, config: AppConfig) -> None:
        self._config = config
        self._tracker = HandTracker(config.tracking)
        self._recognizer = GestureRecognizer(config.control)
        self._mouse = MouseController(config.control)
        self._fps_meter = _FpsMeter()

    def run(self) -> None:
        """Open the camera and process frames until the user quits."""
        logger.info("Starting Hand Mouse. Press 'q' or ESC to quit.")
        with Camera(self._config.camera) as camera:
            try:
                self._main_loop(camera)
            finally:
                self._shutdown()

    def _main_loop(self, camera: Camera) -> None:
        while True:
            frame = camera.read()
            hand = self._tracker.detect(frame)

            if hand is not None:
                gestures: HandGestures | None = self._recognizer.recognize(hand)
                self._mouse.update(gestures)
            else:
                # Hand left the view: forget pinch state and release any
                # held button so a drag is never stuck down.
                self._recognizer.reset()
                self._mouse.release()
                gestures = None

            fps = self._fps_meter.tick()
            if self._config.show_window:
                draw_overlay(
                    frame,
                    hand,
                    gestures,
                    fps,
                    self._config.control.active_region_margin,
                )
                cv2.imshow(self._config.window_name, frame)
                if self._quit_requested():
                    break

    def _quit_requested(self) -> bool:
        # waitKey also yields to the GUI event loop, so the window stays
        # responsive; 1 ms keeps the loop running near camera frame rate.
        key = cv2.waitKey(1) & 0xFF
        return key in self._config.quit_keys

    def _shutdown(self) -> None:
        self._mouse.release()
        self._tracker.close()
        cv2.destroyAllWindows()
        logger.info("Hand Mouse stopped.")
