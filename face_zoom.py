"""
face_zoom.py
─────────────
Digital zoom/crop that keeps the recognised user's FACE nicely framed
in the output view — adapted from the body/torso-tracking auto-zoom in
Zoom_webcam.py, retargeted to track just the face instead.

Unlike Zoom_webcam.py (which ran its own PoseLandmarker), this reuses
the face landmarks main.py's LandmarkPipeline/AuthManager already
compute every FACE_DETECT_INTERVAL frames — no extra model or detection
pass needed.

This is a DISPLAY-ONLY effect: call update() with the tracked face's
landmarks each frame, then apply() right before cv2.imshow() to crop
and rescale the already-composed (HUD-drawn) frame. Hand/pose/gesture
detection and cursor control keep running on the original, un-cropped
frame — this class never touches that pipeline, so cursor accuracy is
unaffected by zoom.
"""

import numpy as np
import cv2


class FaceZoomController:
    TARGET_RATIO    = 0.55   # face bounding-box height as a fraction of frame height
    FAR_THRESH      = 0.18   # below this face-height ratio, subject is considered "far"
    FAR_BOOST       = 0.15   # extra target ratio applied when far, to zoom in more
    PAD_TOP         = 0.90   # headroom above the face (relative to face height)
    PAD_BOTTOM      = 1.60   # room below the face for shoulders/chest
    PAD_SIDES       = 0.90   # room to each side of the face (relative to face width)
    ZOOM_IN_SMOOTH  = 0.14
    ZOOM_OUT_SMOOTH = 0.07
    PAN_SMOOTH      = 0.12
    MAX_ZOOM        = 4.0
    MIN_ZOOM        = 1.0

    def __init__(self, enabled=True):
        self.enabled = enabled
        self.zoom    = 1.0
        self.crop_cx = None
        self.crop_cy = None

    def toggle(self):
        self.enabled = not self.enabled
        return self.enabled

    def update(self, face_landmarks, w, h):
        """
        face_landmarks: the tracked person's face landmark list (478 pts,
        normalized 0-1), or None if they're not currently visible/
        recognised. Called every frame; smoothly relaxes back to zoom=1.0
        when nobody is tracked.
        """
        if self.crop_cx is None:
            self.crop_cx, self.crop_cy = w / 2.0, h / 2.0

        if not self.enabled:
            return

        if face_landmarks is None:
            # Nobody to track right now — ease back out to the full frame
            # rather than freezing on the last known crop.
            target_zoom = self.MIN_ZOOM
            smooth = self.ZOOM_OUT_SMOOTH
            self.zoom += (target_zoom - self.zoom) * smooth
            self.crop_cx += (w / 2.0 - self.crop_cx) * self.PAN_SMOOTH
            self.crop_cy += (h / 2.0 - self.crop_cy) * self.PAN_SMOOTH
            return

        xs = np.array([lm.x for lm in face_landmarks], dtype=np.float32) * w
        ys = np.array([lm.y for lm in face_landmarks], dtype=np.float32) * h

        face_x1, face_x2 = float(xs.min()), float(xs.max())
        face_y1, face_y2 = float(ys.min()), float(ys.max())
        face_w = max(face_x2 - face_x1, 1.0)
        face_h = max(face_y2 - face_y1, 1.0)

        top      = face_y1 - self.PAD_TOP * face_h
        bottom   = face_y2 + self.PAD_BOTTOM * face_h
        region_h = bottom - top

        cx_face  = (face_x1 + face_x2) / 2.0
        left     = cx_face - (face_w / 2.0) - self.PAD_SIDES * face_w
        right    = cx_face + (face_w / 2.0) + self.PAD_SIDES * face_w
        region_w = max(right - left, 1.0)

        face_ratio = face_h / h
        far_factor = max(0.0, min(1.0, (self.FAR_THRESH - face_ratio) / self.FAR_THRESH))
        eff_target = min(self.TARGET_RATIO + self.FAR_BOOST * far_factor, 0.95)

        zoom_for_height = (eff_target * h) / region_h
        fit_zoom_h      = h / region_h
        fit_zoom_w      = w / region_w
        target_zoom     = min(zoom_for_height, fit_zoom_h, fit_zoom_w, self.MAX_ZOOM)
        target_zoom     = max(target_zoom, self.MIN_ZOOM)

        smooth = self.ZOOM_IN_SMOOTH if target_zoom > self.zoom else self.ZOOM_OUT_SMOOTH
        self.zoom += (target_zoom - self.zoom) * smooth

        person_cx = (face_x1 + face_x2) / 2.0
        person_cy = (top + bottom) / 2.0
        self.crop_cx += (person_cx - self.crop_cx) * self.PAN_SMOOTH
        self.crop_cy += (person_cy - self.crop_cy) * self.PAN_SMOOTH

    def apply(self, frame):
        """
        Crop the frame around the tracked face and resize back to the
        original frame size. No-op (returns frame unchanged) when
        disabled or before a face has ever been seen.
        """
        if not self.enabled or self.crop_cx is None:
            return frame

        h, w = frame.shape[:2]
        crop_w = max(int(w / self.zoom), 1)
        crop_h = max(int(h / self.zoom), 1)
        x1 = int(self.crop_cx - crop_w / 2)
        y1 = int(self.crop_cy - crop_h / 2)
        x1 = max(0, min(x1, w - crop_w))
        y1 = max(0, min(y1, h - crop_h))
        x2, y2 = x1 + crop_w, y1 + crop_h

        cropped = frame[y1:y2, x1:x2]
        if cropped.size == 0:
            return frame
        return cv2.resize(cropped, (w, h))
