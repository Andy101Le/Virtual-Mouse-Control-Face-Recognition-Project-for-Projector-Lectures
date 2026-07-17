"""
camera_manager.py
─────────────────
Camera abstraction used by main.py.

Picks the Raspberry Pi CSI camera (picamera2) when available, falling
back to a standard USB webcam via OpenCV otherwise. Both are exposed
through the same tiny read()/release() interface so the rest of the
pipeline never has to know which one is active.
"""

import cv2

try:
    from picamera2 import Picamera2
    _HAS_PICAMERA2 = True
except ImportError:
    _HAS_PICAMERA2 = False


class CameraManager:
    def __init__(self, width=640, height=480, fps=30):
        self.width  = width
        self.height = height
        self._picam = None
        self._cap   = None

        if _HAS_PICAMERA2:
            self._picam = Picamera2()
            self._picam.configure(self._picam.create_video_configuration(
                main={"size": (width, height), "format": "RGB888"}))
            self._picam.start()
            self.description = "RPi CSI (picamera2)"
        else:
            self._cap = cv2.VideoCapture(0)
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            self._cap.set(cv2.CAP_PROP_FPS,          fps)
            self._cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)
            self.description = "USB webcam (cv2)"

        print(f"Camera: {self.description}")

    def is_opened(self):
        if self._picam is not None:
            return True
        return self._cap.isOpened()

    def read(self):
        if self._picam is not None:
            return True, self._picam.capture_array()
        return self._cap.read()

    def release(self):
        if self._picam is not None:
            self._picam.stop()
        elif self._cap is not None:
            self._cap.release()
