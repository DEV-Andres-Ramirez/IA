"""Context-managed wrapper around an OpenCV video-capture device."""

from __future__ import annotations

import logging

import cv2
import numpy as np

from .config import CameraConfig

logger = logging.getLogger(__name__)


class CameraError(RuntimeError):
    """Raised when the camera cannot be opened or a frame cannot be read."""


class Camera:
    """Owns the lifecycle of a webcam capture.

    Use it as a context manager so the device is always released, even when
    the surrounding loop raises:

        with Camera(config) as camera:
            frame = camera.read()
    """

    def __init__(self, config: CameraConfig) -> None:
        self._config = config
        self._capture: cv2.VideoCapture | None = None

    def __enter__(self) -> "Camera":
        capture = cv2.VideoCapture(self._config.device_index)
        if not capture.isOpened():
            raise CameraError(
                f"Could not open camera at index {self._config.device_index}. "
                "Check that a webcam is connected and not used by another app."
            )
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, self._config.frame_width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self._config.frame_height)
        self._capture = capture
        logger.info("Camera %d opened.", self._config.device_index)
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None
            logger.info("Camera released.")

    def read(self) -> np.ndarray:
        """Return the next frame in BGR, mirrored when configured.

        Raises CameraError if the device is closed or returns no frame.
        """
        if self._capture is None:
            raise CameraError("Camera is not open; use it as a context manager.")
        ok, frame = self._capture.read()
        if not ok or frame is None:
            raise CameraError("Failed to read a frame from the camera.")
        if self._config.flip_horizontal:
            frame = cv2.flip(frame, 1)
        return frame
