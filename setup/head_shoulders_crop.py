"""
head_shoulders_crop.py
───────────────────────
Digital "virtual zoom" for the Virtual Mouse project: crops the camera
frame to a head-and-shoulders region around the REGISTERED user and
resizes it back up to a fixed output size, so the displayed view stays
framed on the user's upper body regardless of how close or far they
are — without touching the physical PTZ zoom motor at all.

Why digital crop instead of relying on PTZController's optical zoom:
  - Instant: no motor travel time, no settle delay, no risk of the
    "zoom stuck mid-travel" issues we hit with the physical lens.
  - No focus-breathing coupling: optical zoom shifts the lens's sharp
    focus point as it moves (see ptz_controller.py's FOCUS_BREATHING_
    RATIO handling); a digital crop has no such effect at all.
  - Can run independently of, or alongside, PTZController — this
    module never touches Focuser/hardware, it only operates on the
    already-captured frame array.

Usage in main.py (minimal):

    from head_shoulders_crop import HeadShouldersCropper

    cropper = HeadShouldersCropper(active_user=ACTIVE_USER)
    ...
    # inside the main loop, after face/pose detection updates each frame,
    # right before cv2.imshow(...):
    frame = cropper.apply(
        frame,
        face_landmarks=cached_face_lms[0] if (cached_face_lms and recognised_user == ACTIVE_USER) else None,
        pose_landmarks_list=cached_pose_lms,
        is_registered_face_visible=(user_active and recognised_user == ACTIVE_USER),
    )

That's the only change needed — `frame` comes back at the same fixed
output resolution every time (so the rest of main.py's HUD-drawing code
keeps working unmodified), either cropped-and-zoomed onto the user, or
the original full frame when nobody's tracked.
"""

import numpy as np
import cv2


class HeadShouldersCropper:
    # ── Tunables ─────────────────────────────────────────────────────────────
    # Margins around the raw head+shoulders box, as a fraction of the
    # box's own size — gives breathing room so the crop doesn't hug the
    # subject's outline too tightly.
    MARGIN_TOP    = 0.35   # extra space above the head
    MARGIN_SIDES  = 0.30   # extra space left/right of shoulders
    MARGIN_BOTTOM = 0.15   # extra space below shoulders

    # When pose (shoulders) isn't available, expand the face box alone
    # by this multiple to approximate a head+shoulders region.
    FACE_ONLY_EXPAND = 2.6

    # Smoothing: exponential moving average on the crop box's center and
    # half-size, so it resizes/re-centers gradually rather than jumping
    # frame to frame. Lower = smoother/slower to respond, higher = snappier.
    SMOOTH_ALPHA = 0.12

    # Hard limits on how tight/wide the crop is allowed to get, as a
    # fraction of the FULL frame's smaller dimension. Prevents an
    # unreasonably extreme crop from a single noisy detection.
    MIN_CROP_FRACTION = 0.18   # tightest allowed crop (most "zoomed in")
    MAX_CROP_FRACTION = 1.0    # loosest allowed crop (= no zoom, full frame)

    # Output resolution the cropped region gets resized to. Fixed, so
    # cv2.imshow always receives a consistent size regardless of how
    # tight or loose the current crop is.
    OUTPUT_SIZE = (640, 480)   # (width, height) — matches main.py's camera config

    # Pose proximity gating, mirroring PTZController's identity-continuity
    # guard: only trust a pose as "the registered user" if its nose is
    # close to where the face was last actually confirmed.
    POSE_PROXIMITY_THRESH = 0.22
    POSE_VIS_THRESH        = 0.4

    def __init__(self, active_user=None, output_size=None, enabled=True):
        self.active_user = active_user
        self.enabled     = enabled
        if output_size is not None:
            self.OUTPUT_SIZE = output_size

        # Smoothed crop state: center (cx, cy) and half-size (hw, hh),
        # all in normalized 0-1 frame coordinates. None until first lock.
        self._smoothed_cx = None
        self._smoothed_cy = None
        self._smoothed_hw = None
        self._smoothed_hh = None

        self._last_known_face_pos = None  # for pose proximity gating

    def set_enabled(self, enabled: bool):
        self.enabled = enabled

    def reset(self):
        """Clears smoothed state — call if the user re-registers or the
        camera is repositioned, so old crop geometry doesn't linger."""
        self._smoothed_cx = self._smoothed_cy = None
        self._smoothed_hw = self._smoothed_hh = None
        self._last_known_face_pos = None

    # ── Public API ────────────────────────────────────────────────────────
    def apply(self, frame, face_landmarks, pose_landmarks_list,
              is_registered_face_visible):
        """
        frame: the current BGR frame (numpy array) from main.py's camera loop.
        face_landmarks: the registered user's raw 478-point face landmark
            list this frame, or None if not currently recognised.
        pose_landmarks_list: main.py's `cached_pose_lms` — list of all
            detected poses this frame (each a list of 33 landmarks), or [].
        is_registered_face_visible: bool, True only when the active_user's
            face is currently recognised & authenticated this frame.

        Returns a frame at self.OUTPUT_SIZE — either the cropped/zoomed
        region around the user, or the original frame resized to the
        same output size if nobody's currently tracked.
        """
        if not self.enabled:
            return self._resize_to_output(frame)

        h, w = frame.shape[:2]
        region = None

        if is_registered_face_visible and face_landmarks is not None:
            region = self._region_from_face(face_landmarks, pose_landmarks_list, w, h)
            nose = face_landmarks[4]
            self._last_known_face_pos = (float(nose.x), float(nose.y))
        else:
            region = self._region_from_pose_only(pose_landmarks_list, w, h)

        if region is None:
            # Nobody trackable this frame — hold the last known crop
            # briefly (smoothing naturally decays toward nothing useful
            # if this persists, but a single missed frame shouldn't
            # snap back to full frame) only if we have smoothed state;
            # otherwise just show the full frame.
            if self._smoothed_cx is None:
                return self._resize_to_output(frame)
            cx, cy, hw, hh = self._smoothed_cx, self._smoothed_cy, self._smoothed_hw, self._smoothed_hh
        else:
            cx, cy, hw, hh = self._smooth(*region)

        crop = self._extract_crop(frame, cx, cy, hw, hh, w, h)
        return self._resize_to_output(crop)

    # ── Region computation ──────────────────────────────────────────────────
    def _region_from_face(self, face_landmarks, pose_landmarks_list, w, h):
        """Builds a head+shoulders box from the face bounding box, widened
        using pose shoulder landmarks when a matching pose is available."""
        xs = [lm.x for lm in face_landmarks]
        ys = [lm.y for lm in face_landmarks]
        face_x0, face_x1 = min(xs), max(xs)
        face_y0, face_y1 = min(ys), max(ys)
        face_cx = (face_x0 + face_x1) / 2.0
        face_w  = face_x1 - face_x0
        face_h  = face_y1 - face_y0

        pose = self._find_matching_pose(pose_landmarks_list, (face_cx, (face_y0 + face_y1) / 2.0))

        if pose is not None:
            l_sh, r_sh = pose[11], pose[12]
            if l_sh.visibility >= self.POSE_VIS_THRESH and r_sh.visibility >= self.POSE_VIS_THRESH:
                shoulder_x0 = min(l_sh.x, r_sh.x)
                shoulder_x1 = max(l_sh.x, r_sh.x)
                shoulder_y  = max(l_sh.y, face_y1)  # shoulders are below the face

                x0 = min(face_x0, shoulder_x0)
                x1 = max(face_x1, shoulder_x1)
                y0 = face_y0
                y1 = shoulder_y

                box_w = x1 - x0
                box_h = y1 - y0
                cx = (x0 + x1) / 2.0
                cy = (y0 + y1) / 2.0
                hw = (box_w / 2.0) * (1 + self.MARGIN_SIDES)
                hh_top    = (cy - y0) * (1 + self.MARGIN_TOP)
                hh_bottom = (y1 - cy) * (1 + self.MARGIN_BOTTOM)
                hh = max(hh_top, hh_bottom)
                return self._clamp_region(cx, cy, hw, hh, w, h)

        # No matching pose / shoulders not visible — approximate from
        # face box alone by expanding symmetrically.
        cx, cy = face_cx, (face_y0 + face_y1) / 2.0
        hw = (face_w / 2.0) * self.FACE_ONLY_EXPAND
        hh = (face_h / 2.0) * self.FACE_ONLY_EXPAND
        return self._clamp_region(cx, cy, hw, hh, w, h)

    def _region_from_pose_only(self, pose_landmarks_list, w, h):
        """No confirmed face this frame — fall back to a pose whose nose
        is close to the last known face position (same identity-continuity
        guard PTZController uses for its SEARCHING_BY_POSE state)."""
        if not pose_landmarks_list or self._last_known_face_pos is None:
            return None
        pose = self._find_matching_pose(pose_landmarks_list, self._last_known_face_pos)
        if pose is None:
            return None

        nose = pose[0]
        l_sh, r_sh = pose[11], pose[12]
        if not (l_sh.visibility >= self.POSE_VIS_THRESH and r_sh.visibility >= self.POSE_VIS_THRESH):
            return None

        shoulder_x0 = min(l_sh.x, r_sh.x)
        shoulder_x1 = max(l_sh.x, r_sh.x)
        shoulder_y  = max(l_sh.y, r_sh.y)
        head_y0     = nose.y - (shoulder_y - nose.y) * 0.8  # approximate top-of-head above nose

        cx = (shoulder_x0 + shoulder_x1) / 2.0
        cy = (head_y0 + shoulder_y) / 2.0
        hw = ((shoulder_x1 - shoulder_x0) / 2.0) * (1 + self.MARGIN_SIDES)
        hh_top    = (cy - head_y0) * (1 + self.MARGIN_TOP)
        hh_bottom = (shoulder_y - cy) * (1 + self.MARGIN_BOTTOM)
        hh = max(hh_top, hh_bottom)
        return self._clamp_region(cx, cy, hw, hh, w, h)

    def _find_matching_pose(self, pose_landmarks_list, anchor_xy):
        best, best_dist = None, self.POSE_PROXIMITY_THRESH
        for pose_lms in pose_landmarks_list:
            if len(pose_lms) <= 12:
                continue
            nose = pose_lms[0]
            dx, dy = nose.x - anchor_xy[0], nose.y - anchor_xy[1]
            dist = (dx * dx + dy * dy) ** 0.5
            if dist < best_dist:
                best, best_dist = pose_lms, dist
        return best

    def _clamp_region(self, cx, cy, hw, hh, w, h):
        """Enforces MIN/MAX_CROP_FRACTION and keeps the box within frame
        bounds, preserving the output aspect ratio."""
        out_w, out_h = self.OUTPUT_SIZE
        aspect = out_w / out_h

        # Make the box match the output aspect ratio (whichever
        # dimension is the limiting one gets widened to match).
        if hw / hh > aspect:
            hh = hw / aspect
        else:
            hw = hh * aspect

        min_half = self.MIN_CROP_FRACTION * min(w, h) / (2 * min(w, h))  # in normalized units
        max_half = self.MAX_CROP_FRACTION * min(w, h) / (2 * min(w, h))
        hw = self._clamp(hw, min_half, max_half * aspect)
        hh = self._clamp(hh, min_half, max_half)

        # Keep the box fully inside [0,1] normalized bounds.
        cx = self._clamp(cx, hw, 1.0 - hw)
        cy = self._clamp(cy, hh, 1.0 - hh)
        return cx, cy, hw, hh

    # ── Smoothing ────────────────────────────────────────────────────────────
    def _smooth(self, cx, cy, hw, hh):
        if self._smoothed_cx is None:
            self._smoothed_cx, self._smoothed_cy = cx, cy
            self._smoothed_hw, self._smoothed_hh = hw, hh
        else:
            a = self.SMOOTH_ALPHA
            self._smoothed_cx = a * cx + (1 - a) * self._smoothed_cx
            self._smoothed_cy = a * cy + (1 - a) * self._smoothed_cy
            self._smoothed_hw = a * hw + (1 - a) * self._smoothed_hw
            self._smoothed_hh = a * hh + (1 - a) * self._smoothed_hh
        return self._smoothed_cx, self._smoothed_cy, self._smoothed_hw, self._smoothed_hh

    # ── Extraction / output ──────────────────────────────────────────────────
    def _extract_crop(self, frame, cx, cy, hw, hh, w, h):
        x0 = int(round((cx - hw) * w))
        x1 = int(round((cx + hw) * w))
        y0 = int(round((cy - hh) * h))
        y1 = int(round((cy + hh) * h))
        x0, x1 = max(0, x0), min(w, x1)
        y0, y1 = max(0, y0), min(h, y1)
        if x1 <= x0 or y1 <= y0:
            return frame  # degenerate box — bail to full frame rather than crash
        return frame[y0:y1, x0:x1]

    def _resize_to_output(self, frame):
        return cv2.resize(frame, self.OUTPUT_SIZE, interpolation=cv2.INTER_LINEAR)

    @staticmethod
    def _clamp(v, lo, hi):
        return max(lo, min(hi, v))
