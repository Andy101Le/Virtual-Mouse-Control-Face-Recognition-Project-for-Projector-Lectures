"""
zoom_webcam.py
───────────────
Digital zoom/crop that keeps the tracked person framed, ported from the
original standalone zoom_webcam.py auto-zoom prototype into a reusable
class main.py can call into.

The core tracking math is UNCHANGED from the original script: same POSE
landmark indices (face/shoulder/hip), same target-ratio + far-subject
zoom boost, same independent zoom-in/zoom-out smoothing rates. What's
different is the packaging — instead of running its own PoseLandmarker
and camera capture loop, this takes the pose landmarks main.py's
LandmarkPipeline already computes each frame, and exposes update()/
apply() so main.py can drive it directly.

ADDED vs. the original script: the original assumed a full-body framing
(head down to hips always visible), which isn't realistic for a
face/podium-level camera — hips are very often out of frame or low-
confidence there, and the original code just froze at zoom=1.0 whenever
hips weren't detected. This version degrades gracefully instead:
  - hips visible      → original "waist-up" framing (unchanged math)
  - only shoulders visible → "chest-up" framing, extrapolated from
    head-to-shoulder distance
  - only face visible → "face" framing, padded from face height
draw_debug() also reproduces the zoom-level readout + tracked-region
box the original script drew, so it's visible on screen whether/how
much auto-zoom is active.

Also added: a control-zone-aware zoom cap. This project uses the
camera feed for two things at once — framing the shot, AND detecting
hand gestures across CursorController's CAM_MARGIN control zone (the
region the person's hand must be able to reach across to move the
cursor). Those two purposes were fighting: without a cap, auto-zoom
would happily crop in tighter than the control zone itself, so the
user's outstretched arm would go off-screen even though it was still
being tracked correctly by (un-cropped) hand detection. Pass
control_zone_margin (CursorController.CAM_MARGIN) in and zoom will
never crop past what that margin requires stay visible.

This is a DISPLAY-ONLY effect: call update() with the tracked person's
pose landmarks each frame, apply() right before cv2.imshow() to crop/
rescale the frame, then draw_debug() on the result to overlay the
zoom readout. Hand/gesture detection and cursor control keep running
on the original, un-cropped frame — this class never touches that
pipeline, so cursor accuracy is unaffected by zoom.
"""

import cv2
import numpy as np


class ZoomWebcamController:
    # ── Same tunables as the original zoom_webcam.py ────────────────────────
    TARGET_RATIO    = 0.80
    FAR_THRESH      = 0.45
    FAR_BOOST       = 0.18
    HEAD_PAD        = 0.20
    HIP_PAD         = 0.10
    ZOOM_IN_SMOOTH  = 0.14
    ZOOM_OUT_SMOOTH = 0.07
    PAN_SMOOTH      = 0.12
    MAX_ZOOM        = 4.0    # absolute ceiling. Raised back up from 2.5 so a
                              # distant subject can actually be framed tightly;
                              # at CLOSE range the arm-reach control-zone cap
                              # (see update()) still keeps zoom comfortable for
                              # gestures, so this ceiling only bites far away.
    FAR_MAX_ZOOM    = 3.5    # how far the distance-adaptive control-zone cap is
                              # allowed to relax once the subject is far
                              # (far_factor→1). Up close the cap stays pinned at
                              # the strict control-zone value; see update().
    VIS_THRESH      = 0.5

    # ── Fallback framing when the original head-to-hip region isn't visible ─
    CHEST_FALLBACK_RATIO = 1.6   # extend below shoulders by this multiple of
                                 # head-to-shoulder distance ("chest-up")
    FACE_FALLBACK_RATIO  = 3.0   # extend below the face by this multiple of
                                 # face height, when even shoulders aren't seen

    FACE_IDS     = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    SHOULDER_IDS = [11, 12]
    HIP_IDS      = [23, 24]

    # ── Optical-zoom handoff ────────────────────────────────────────────────
    # When the lens has a working zoom motor, magnification should come from
    # it first: optical zoom is lossless, while every bit of digital zoom is
    # just upscaled crop. So while the motor still has travel left toward
    # telephoto, this class refuses to zoom IN any further and lets the motor
    # do the work; it only takes over once the motor is at its stop — that's
    # the "gap" it fills. Zooming OUT is never held back, because backing off
    # a crop is instant and free while the motor takes seconds.
    #
    # No feedback loop is needed to keep the two in agreement: update() reads
    # the person's size out of the frame the optical zoom already magnified,
    # so as the motor racks in, region_h grows and the digital target falls
    # out on its own.
    OPTICAL_HEADROOM_EPS = 0.02   # treat below this as "motor is at its stop"

    def __init__(self, enabled=True, control_zone_margin=None):
        self.enabled = enabled
        self.zoom    = 1.0
        self.crop_cx = None
        self.crop_cy = None

        # Latest optical zoom state, refreshed each frame by the vision loop
        # via set_optical(). The default describes "no optical zoom at all",
        # under which update() behaves exactly as it did before the motor
        # existed.
        self._opt_mag      = 1.0
        self._opt_headroom = 0.0
        self._opt_moving   = False

        # If given the cursor control zone's margin (CursorController.CAM_MARGIN),
        # cap zoom so the crop never excludes that zone — i.e. the person can
        # always see their own hand across its full reachable range.
        # margin=0.15 means the middle 70% of the frame is the control zone,
        # so zoom must never exceed 1 / (1 - 2*margin) ≈ 1.43x.
        #
        # This cap is the FLOOR of a distance-adaptive range, not a hard
        # ceiling: update() holds zoom at exactly this value while the
        # subject is close (so their full arm reach stays visible for
        # gestures), then smoothly relaxes it toward FAR_MAX_ZOOM as they
        # move away — a distant subject's reachable area occupies only a
        # small patch of the frame, so cropping tighter than the raw
        # control zone no longer cuts their real reach out of view. That
        # keeps a far-away presenter framed tightly (the original problem
        # a fixed 1.43x cap could never solve) without sacrificing close-
        # range gesture room. See the eff_cap logic in update() and the
        # per-axis clamp guard in apply(). Pass control_zone_margin=None to
        # disable the arm-reach cap entirely (zoom then governed only by
        # MAX_ZOOM, but part of the reach may be cropped at high zoom).
        self.control_zone_margin = control_zone_margin
        if control_zone_margin is not None and 0 < control_zone_margin < 0.5:
            self.control_zone_zoom_cap = 1.0 / (1.0 - 2 * control_zone_margin)
        else:
            self.control_zone_zoom_cap = None

        # Populated by update()/apply() for draw_debug() to visualise.
        self.last_region    = None   # (rx1, top, rx2, bottom) in raw-frame px
        self.last_mode      = None   # "torso" | "upper-body" | "face" | None
        self.last_crop_rect = None   # (x1, y1, crop_w, crop_h) in raw-frame px
        self.last_scale_x   = 1.0    # exact scale apply() last used (for transforms)
        self.last_scale_y   = 1.0

    def toggle(self):
        self.enabled = not self.enabled
        return self.enabled

    def set_optical(self, state):
        """
        Feed in PTZController.get_zoom_state() (or None when there's no PTZ)
        so update() knows how much magnification the lens is already
        providing and whether the motor still has room to provide more.
        """
        if not state:
            self._opt_mag, self._opt_headroom, self._opt_moving = 1.0, 0.0, False
            return
        self._opt_mag      = float(state.get("mag", 1.0))
        self._opt_headroom = float(state.get("headroom", 0.0))
        self._opt_moving   = bool(state.get("moving", False))

    @property
    def total_mag(self):
        """Combined optical x digital magnification — what the viewer
        actually sees, and the honest number to report in telemetry."""
        return self._opt_mag * self.zoom

    @classmethod
    def _pts(cls, pose_landmarks, ids, w, h):
        return [(pose_landmarks[i].x * w, pose_landmarks[i].y * h)
                for i in ids if pose_landmarks[i].visibility > cls.VIS_THRESH]

    def update(self, pose_landmarks, w, h):
        """
        pose_landmarks: the tracked person's 33-point MediaPipe pose
        landmarks, or None if they're not currently visible/recognised.
        Called every frame; smoothly relaxes back to zoom=1.0 (full
        frame) when nobody is tracked.
        """
        if self.crop_cx is None:
            self.crop_cx, self.crop_cy = w / 2.0, h / 2.0

        if not self.enabled:
            return

        if pose_landmarks is None:
            self.zoom    += (1.0 - self.zoom) * self.ZOOM_OUT_SMOOTH
            self.crop_cx += (w / 2.0 - self.crop_cx) * self.PAN_SMOOTH
            self.crop_cy += (h / 2.0 - self.crop_cy) * self.PAN_SMOOTH
            self.last_region = None
            self.last_mode   = None
            return

        face      = self._pts(pose_landmarks, self.FACE_IDS,     w, h)
        shoulders = self._pts(pose_landmarks, self.SHOULDER_IDS, w, h)
        hips      = self._pts(pose_landmarks, self.HIP_IDS,      w, h)

        if not face:
            # Not even the face is confidently visible — hold the last
            # zoom/pan rather than snapping back; nothing new to size.
            return

        head_y = min(p[1] for p in face)

        if hips:
            # ── Original script's exact math: head-to-hip torso ──────────
            hip_y   = sum(p[1] for p in hips) / len(hips)
            torso_h = max(hip_y - head_y, 1)
            bottom  = hip_y + self.HIP_PAD * torso_h
            mode    = "torso"
        elif shoulders:
            # ── Fallback: hips out of frame (very common on a podium/
            # desk webcam) — frame chest-up instead of freezing at 1x ──
            shoulder_y       = sum(p[1] for p in shoulders) / len(shoulders)
            head_to_shoulder = max(shoulder_y - head_y, 1)
            bottom  = shoulder_y + self.CHEST_FALLBACK_RATIO * head_to_shoulder
            torso_h = head_to_shoulder
            mode    = "upper-body"
        else:
            # ── Last resort: only the face is visible ────────────────────
            face_ys = [p[1] for p in face]
            face_h  = max(max(face_ys) - min(face_ys), 1)
            bottom  = head_y + self.FACE_FALLBACK_RATIO * face_h
            torso_h = face_h
            mode    = "face"

        top      = head_y - self.HEAD_PAD * torso_h
        region_h = bottom - top

        all_x    = [p[0] for p in face + shoulders + hips]
        rx1, rx2 = min(all_x), max(all_x)
        region_w = max(rx2 - rx1, 1) * 1.30   # arm room

        torso_ratio = torso_h / h
        far_factor  = max(0.0, min(1.0, (self.FAR_THRESH - torso_ratio) / self.FAR_THRESH))
        eff_target  = min(self.TARGET_RATIO + self.FAR_BOOST * far_factor, 0.95)

        zoom_for_height = (eff_target * h) / region_h
        fit_zoom_h      = h / region_h
        fit_zoom_w      = w / region_w

        target_zoom = min(zoom_for_height, fit_zoom_h, fit_zoom_w, self.MAX_ZOOM)
        if self.control_zone_zoom_cap is not None:
            # Distance-adaptive arm-reach cap. Up close (far_factor≈0) we hold
            # the strict control-zone cap so the user's outstretched arm stays
            # fully on screen for gesture control. As the subject recedes
            # (far_factor→1) their reachable area shrinks to a small patch of
            # the frame, so we smoothly relax the cap toward FAR_MAX_ZOOM —
            # letting the camera stay tight on a distant presenter without
            # actually cropping their (now much smaller) real reach out of view.
            eff_cap = self.control_zone_zoom_cap + far_factor * (
                max(self.FAR_MAX_ZOOM, self.control_zone_zoom_cap)
                - self.control_zone_zoom_cap)
            target_zoom = min(target_zoom, eff_cap)
        target_zoom = max(target_zoom, 1.0)

        # ── Defer to the optical zoom before cropping (see class comment) ──
        if self._opt_moving:
            # The motor is mid-travel and the frame is being magnified out
            # from under us. Anything measured now is stale by the time the
            # crop lands, so hold still rather than chase it.
            target_zoom = self.zoom
        elif (self._opt_headroom > self.OPTICAL_HEADROOM_EPS
                and target_zoom > self.zoom):
            # More magnification is wanted and the lens can still deliver it
            # losslessly — don't spend crop on what the motor is about to do.
            # Backing OUT stays unrestricted, which is why this only clamps
            # the zoom-in direction.
            target_zoom = self.zoom

        smooth = self.ZOOM_IN_SMOOTH if target_zoom > self.zoom else self.ZOOM_OUT_SMOOTH
        self.zoom += (target_zoom - self.zoom) * smooth

        person_cx = (rx1 + rx2) / 2
        person_cy = (top + bottom) / 2
        self.crop_cx += (person_cx - self.crop_cx) * self.PAN_SMOOTH
        self.crop_cy += (person_cy - self.crop_cy) * self.PAN_SMOOTH

        self.last_region = (rx1, top, rx2, bottom)
        self.last_mode   = mode

    def apply(self, frame):
        """
        Crop the frame around the tracked person and resize back to the
        original frame size. No-op (returns frame unchanged) when
        disabled or before anyone has ever been tracked.

        Call this BEFORE drawing any HUD elements — draw everything
        onto the frame this returns, using transform_point()/
        transform_array() to place each element correctly, rather than
        drawing on the raw frame and cropping the finished composite
        (which works geometrically but blurs text/thin lines under
        magnification).
        """
        if not self.enabled or self.crop_cx is None:
            self.last_crop_rect = None
            self.last_scale_x   = 1.0
            self.last_scale_y   = 1.0
            return frame

        h, w = frame.shape[:2]
        crop_w = max(int(w / self.zoom), 1)
        crop_h = max(int(h / self.zoom), 1)
        x1 = int(self.crop_cx - crop_w / 2)
        y1 = int(self.crop_cy - crop_h / 2)
        x1 = max(0, min(x1, w - crop_w))
        y1 = max(0, min(y1, h - crop_h))

        if self.control_zone_margin is not None:
            # The zoom-level cap already guarantees the crop is WIDE ENOUGH
            # to contain the control zone — this additionally constrains
            # WHERE it's centered, so panning toward an off-center subject
            # can't push the zone (partially) out of the visible crop. The
            # final re-clamp to frame bounds is just a safety net; given the
            # zoom cap, it shouldn't actually change anything.
            cz_x1, cz_y1 = self.control_zone_margin * w, self.control_zone_margin * h
            cz_x2, cz_y2 = w - cz_x1, h - cz_y1
            cz_w,  cz_h  = cz_x2 - cz_x1, cz_y2 - cz_y1
            # Only pin the crop to contain the control zone while the crop is
            # actually wider/taller than the zone. Once the distance-adaptive
            # cap lets zoom crop TIGHTER than the control zone (far-subject
            # case), these constraints would flip (min-bound > max-bound) and
            # shove the crop into a corner — so we skip them per-axis and keep
            # the person-centered crop instead, then clamp to frame bounds.
            if crop_w >= cz_w:
                x1 = min(x1, cz_x1)
                x1 = max(x1, cz_x2 - crop_w)
            if crop_h >= cz_h:
                y1 = min(y1, cz_y1)
                y1 = max(y1, cz_y2 - crop_h)
            x1 = max(0, min(x1, w - crop_w))
            y1 = max(0, min(y1, h - crop_h))

        x1, y1 = int(x1), int(y1)
        x2, y2 = x1 + crop_w, y1 + crop_h

        self.last_crop_rect = (x1, y1, crop_w, crop_h)
        # Use the actual ratio the resize will apply, not the theoretical
        # self.zoom value — int() rounding on crop_w/crop_h means they can
        # differ slightly, and every overlay needs to agree with cv2.resize
        # pixel-for-pixel or HUD elements will drift off their targets.
        self.last_scale_x = w / crop_w
        self.last_scale_y = h / crop_h

        cropped = frame[y1:y2, x1:x2]
        if cropped.size == 0:
            return frame
        return cv2.resize(cropped, (w, h))

    def transform_point(self, x_px, y_px):
        """
        Maps a pixel coordinate in the ORIGINAL raw camera frame to
        where that same physical point appears in the cropped/zoomed
        display frame apply() last produced. Returns the coordinate
        unchanged if no crop is currently active (disabled, or nobody
        tracked yet) — safe to call unconditionally.
        """
        if self.last_crop_rect is None:
            return x_px, y_px
        x1, y1, crop_w, crop_h = self.last_crop_rect
        return (x_px - x1) * self.last_scale_x, (y_px - y1) * self.last_scale_y

    def transform_array(self, pts):
        """Vectorized transform_point for an (N, 2) array of raw pixel
        coordinates. Returns a float32 array of the same shape."""
        pts = np.asarray(pts, dtype=np.float32)
        if self.last_crop_rect is None:
            return pts
        x1, y1, crop_w, crop_h = self.last_crop_rect
        out = pts.copy()
        out[:, 0] = (out[:, 0] - x1) * self.last_scale_x
        out[:, 1] = (out[:, 1] - y1) * self.last_scale_y
        return out

    def get_scale(self):
        """
        Current display magnification factor. Use this to scale font
        sizes / line thicknesses / marker radii so HUD elements stay
        legible and proportionally sized under zoom instead of looking
        tiny relative to a magnified subject. Returns 1.0 when no crop
        is active.
        """
        if self.last_crop_rect is None:
            return 1.0
        return (self.last_scale_x + self.last_scale_y) / 2.0

    def draw_debug(self, frame):
        """
        Overlays a live "Zoom: X.XXx" readout (or "Zoom: OFF") plus the
        tracked-region box, remapped into the cropped/resized frame's
        own coordinate space — the same kind of feedback the original
        zoom_webcam.py prototype drew. Call this AFTER apply(), on the
        frame apply() returned.
        """
        h, w = frame.shape[:2]
        if not self.enabled:
            status = "Zoom: OFF"
        elif self._opt_mag > 1.001:
            # Split the readout once the lens is contributing, so it's clear
            # at a glance how much of the magnification is lossless and
            # whether the handoff to digital has kicked in yet.
            status = (f"Zoom: {self.total_mag:.2f}x "
                      f"(opt {self._opt_mag:.2f} x dig {self.zoom:.2f})")
        else:
            status = f"Zoom: {self.zoom:.2f}x"
        cv2.putText(frame, status, (10, h - 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2, cv2.LINE_AA)

        if self.enabled and self.last_region is not None and self.last_crop_rect is not None:
            rx1, top, rx2, bottom = self.last_region
            x1, y1, crop_w, crop_h = self.last_crop_rect
            dbx1 = int((rx1 - x1) * self.last_scale_x)
            dby1 = int((top  - y1) * self.last_scale_y)
            dbx2 = int((rx2 - x1) * self.last_scale_x)
            dby2 = int((bottom - y1) * self.last_scale_y)
            cv2.rectangle(frame, (dbx1, dby1), (dbx2, dby2), (0, 255, 0), 2)
            label = {"torso": "Waist-up", "upper-body": "Chest-up",
                     "face": "Face"}.get(self.last_mode, "Tracked")
            cv2.putText(frame, label, (dbx1, max(dby1 - 8, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2, cv2.LINE_AA)

        return frame