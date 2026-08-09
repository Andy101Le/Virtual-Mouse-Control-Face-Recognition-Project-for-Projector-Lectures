"""
camera_manager.py
─────────────────
Camera abstraction used by main.py.

Picks the Raspberry Pi CSI camera (picamera2) when available, falling
back to a standard USB webcam via OpenCV otherwise. Both are exposed
through the same tiny read()/release() interface so the rest of the
pipeline never has to know which one is active.

EXPOSURE METADATA. On the CSI path each frame is fetched as a *request*
rather than a bare array, so the sensor's own account of the frame --
exposure time, analogue gain, estimated lux -- arrives with the pixels
it describes. picamera2's other option, capture_metadata(), waits for
the following frame, which would both halve the capture rate and
attribute the wrong settings to the frame in hand.

This is what makes low-light behaviour diagnosable: "no face found" and
"no face found, at 33 ms and 16x gain, with the scene four times
brighter than the subject" are very different bug reports. See
scene_light.py, which consumes last_metadata.
"""

import logging

import cv2

log = logging.getLogger(__name__)

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
        # Sensor metadata for the most recent frame read(). Empty on the USB
        # path, which exposes nothing comparable.
        self.last_metadata = {}

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

    def read(self):
        if self._picam is not None:
            # The request owns a camera buffer, so it must be released even
            # if make_array throws -- holding one back starves the pipeline
            # after a few frames and the stream stalls.
            req = self._picam.capture_request()
            try:
                frame = req.make_array("main")
                self.last_metadata = req.get_metadata()
            finally:
                req.release()
            return True, frame
        return self._cap.read()

    def set_controls(self, controls):
        """
        Apply libcamera controls (exposure, gain, EV bias, metering...).
        No-op on the USB path. Returns True if they were applied.

        Controls take effect a few frames later, not on the next one: the
        sensor is already several frames deep in the pipeline. Anything
        driving this in a loop has to tolerate that delay rather than
        treating an unchanged reading as a failed write.
        """
        if self._picam is None or not controls:
            return False
        try:
            self._picam.set_controls(controls)
            return True
        except Exception:
            log.exception("Failed to apply camera controls: %r", controls)
            return False

    def release(self):
        if self._picam is not None:
            self._picam.stop()
        elif self._cap is not None:
            self._cap.release()
