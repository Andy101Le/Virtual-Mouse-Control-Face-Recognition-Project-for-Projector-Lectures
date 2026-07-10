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
    # ── Tunables: proportional control on crop tightness ────────────────────
    # The crop's SIZE is now driven by proportional control toward a target
    # face-box-to-control-zone ratio, mirroring how PTZController's optical
    # zoom targets TARGET_FACE_SIZE — just applied to a digital crop instead
    # of a hardware register. The control zone is always CAM_MARGIN_FRACTION
    # of the OUTPUT frame on each side (matching main.py's CAM_MARGIN=0.15),
    # so "face box size relative to the control zone" reduces to a fixed
    # target expressed directly in output-frame terms — no live geometry
    # feedback loop needed, since CAM_MARGIN_FRACTION never changes.
    CAM_MARGIN_FRACTION = 0.15   # MUST match main.py's CAM_MARGIN

    # ── Tunables: direct anatomical sizing (no proportional control) ────────
    # The crop's size is computed DIRECTLY from measured anatomy each frame
    # — head height and shoulder width — rather than chasing a target
    # face-to-zone ratio. This avoids a real problem the ratio approach hit:
    # at moderate-to-close range, a ratio-based target can demand a crop
    # WIDER than the camera's actual field of view, which is impossible to
    # satisfy and causes the box to collapse to whatever fits, disconnected
    # from any deliberate framing. Measuring real anatomy directly can never
    # demand more than what's actually captured.
    #
    # Standard figure-drawing proportion: the navel sits roughly 3 head-
    # heights down from the crown (chin-to-chest-to-navel spans roughly
    # heads 2-3.5 in the classical 7.5-head canon). MediaPipe's face
    # landmark box runs forehead-to-chin, not crown-to-chin, so a small
    # correction is folded into HEAD_TOP_MARGIN below.
    HEAD_TO_STOMACH_HEADS = 3.0   # navel/lower-stomach depth, in face-heights,
                                   # measured down from the TOP OF THE HEAD
    HEAD_TOP_MARGIN       = 0.45  # extra space above the face box (forehead/
                                   # hairline/skull-top aren't in the face
                                   # landmark box), as a fraction of face height

    # Horizontal: half-width is the larger of (a) shoulder span with a side
    # margin for arms hanging at the sides, or (b) face width with its own
    # margin, so a face-only (no pose) fallback still gets a sane width.
    ARM_SIDE_MARGIN  = 0.35   # extra width beyond each shoulder, as a
                                # fraction of shoulder-to-shoulder span,
                                # to include arms hanging at the sides
    FACE_ONLY_WIDTH_MULT = 2.2  # width multiplier on face width when no
                                  # pose/shoulders are available at all

    # Smoothing: lighter than before (snappier per request) — still an EMA
    # on the crop box's center and half-size so it doesn't jump on a single
    # noisy detection, but reacts much faster than the old SMOOTH_ALPHA.
    SMOOTH_ALPHA = 0.35

    # Hard limits on how tight/wide the crop is allowed to get, as a
    # fraction of the FULL frame's smaller dimension. Prevents an
    # unreasonably extreme crop from a single noisy detection.
    #
    # NOTE: 0.18 (the original value) clamps the floor at hw=0.12, which
    # turned out to cover face_diag values up to ~0.13 — i.e. almost the
    # ENTIRE realistic range of face sizes at normal working distances
    # got clamped to the identical crop tightness, so the face appeared
    # at wildly different actual sizes within that supposedly-uniform
    # crop. Lowered so the floor only engages for genuinely extreme
    # close-up detections, not normal distances.
    MIN_CROP_FRACTION = 0.10   # tightest allowed crop (most "zoomed in")
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
        self._last_frame_dims = None      # cached (w, h) for get_crop_region()

    def set_enabled(self, enabled: bool):
        self.enabled = enabled

    def reset(self):
        """Clears smoothed state — call if the user re-registers or the
        camera is repositioned, so old crop geometry doesn't linger."""
        self._smoothed_cx = self._smoothed_cy = None
        self._smoothed_hw = self._smoothed_hh = None
        self._last_known_face_pos = None
        self._last_frame_dims = None

    # ── Public API ────────────────────────────────────────────────────────
    def get_crop_region(self, face_landmarks, pose_landmarks_list,
                         is_registered_face_visible, frame_dims=None):
        """
        Computes (and smooths) the current crop region WITHOUT touching
        any frame pixels — returns (cx, cy, hw, hh) in normalized 0-1
        coordinates relative to the RAW (uncropped) frame, or None if
        nobody is currently trackable and there's no prior smoothed state
        to fall back on.

        Sizing is computed DIRECTLY from measured anatomy each frame
        (head height, shoulder width) — not from proportional control
        toward a target ratio. This guarantees the box always reflects
        real, physically-present extent regardless of distance, and can
        never demand a crop bigger than the camera's actual field of
        view the way ratio-based sizing could at moderate-to-close range.

        IMPORTANT: this updates internal smoothing state, so call it at
        most ONCE per frame — see apply()'s docstring for why.

        frame_dims: optional (width, height) of the frame this region
        will be applied to. If omitted, assumes the frame passed to the
        next apply_with_region() call has the same dimensions as the
        last one seen (cached internally) — pass it explicitly the
        first time, or whenever frame size might have changed.
        """
        if frame_dims is not None:
            self._last_frame_dims = frame_dims
        if self._last_frame_dims is None:
            return None
        w, h = self._last_frame_dims

        if not self.enabled:
            return None

        if is_registered_face_visible and face_landmarks is not None:
            box = self._box_from_face(face_landmarks, pose_landmarks_list)
            nose = face_landmarks[4]
            self._last_known_face_pos = (float(nose.x), float(nose.y))
        else:
            box = self._box_from_pose_only(pose_landmarks_list)

        if box is None:
            if self._smoothed_cx is None:
                return None
            return (self._smoothed_cx, self._smoothed_cy, self._smoothed_hw, self._smoothed_hh)

        cx, cy, hw, hh = box
        return self._smooth(cx, cy, hw, hh, w, h)

    def apply_with_region(self, frame, region):
        """
        Crops+resizes `frame` using an already-computed region tuple
        (cx, cy, hw, hh) from get_crop_region() — does NOT recompute or
        re-smooth anything. Use this together with get_crop_region()
        when you need the same region for other coordinate conversions
        in the same frame (see main.py).
        """
        h, w = frame.shape[:2]
        if region is None:
            return self._resize_to_output(frame)
        cx, cy, hw, hh = region
        crop = self._extract_crop(frame, cx, cy, hw, hh, w, h)
        return self._resize_to_output(crop)

    def apply(self, frame, face_landmarks, pose_landmarks_list,
              is_registered_face_visible):
        """
        Convenience one-call version: computes the region AND crops the
        frame in one step. Equivalent to:
            region = self.get_crop_region(face_landmarks, pose_landmarks_list,
                                           is_registered_face_visible,
                                           frame_dims=frame.shape[1::-1])
            return self.apply_with_region(frame, region)
        Use the two-step version instead (as main.py does) when you also
        need `region` for converting other coordinates into crop-relative
        space — calling `apply()` AND `get_crop_region()` separately in
        the same frame would smooth twice and desync the two results.

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
        h, w = frame.shape[:2]
        region = self.get_crop_region(face_landmarks, pose_landmarks_list,
                                       is_registered_face_visible, frame_dims=(w, h))
        return self.apply_with_region(frame, region)

    # ── Direct anatomical box computation (centering AND sizing together) ───
    def _box_from_face(self, face_landmarks, pose_landmarks_list):
        """
        Returns (cx, cy, hw, hh) computed DIRECTLY from measured anatomy:
          - Top edge: top of face box, minus HEAD_TOP_MARGIN for hair/skull.
          - Bottom edge: top of face box, PLUS HEAD_TO_STOMACH_HEADS face-
            heights down — i.e. wherever the navel/lower-stomach actually
            is this frame, regardless of distance.
          - Width: shoulder span (when available) plus ARM_SIDE_MARGIN on
            each side for arms hanging down, else a multiple of face width.
        No proportional control, no target ratio — this can never demand
        a box bigger than what's physically there, so it can't fight an
        unreachable target the way ratio-based sizing did.
        """
        xs = [lm.x for lm in face_landmarks]
        ys = [lm.y for lm in face_landmarks]
        face_x0, face_x1 = min(xs), max(xs)
        face_y0, face_y1 = min(ys), max(ys)
        face_cx = (face_x0 + face_x1) / 2.0
        face_w  = face_x1 - face_x0
        face_h  = face_y1 - face_y0

        head_top    = face_y0 - face_h * self.HEAD_TOP_MARGIN
        stomach_bot = face_y0 + face_h * self.HEAD_TO_STOMACH_HEADS

        pose = self._find_matching_pose(pose_landmarks_list, (face_cx, (face_y0 + face_y1) / 2.0))

        if pose is not None:
            l_sh, r_sh = pose[11], pose[12]
            if l_sh.visibility >= self.POSE_VIS_THRESH and r_sh.visibility >= self.POSE_VIS_THRESH:
                shoulder_x0 = min(l_sh.x, r_sh.x)
                shoulder_x1 = max(l_sh.x, r_sh.x)
                shoulder_span = shoulder_x1 - shoulder_x0
                margin = shoulder_span * self.ARM_SIDE_MARGIN
                box_x0 = min(face_x0, shoulder_x0 - margin)
                box_x1 = max(face_x1, shoulder_x1 + margin)
                return self._box_from_extent(box_x0, box_x1, head_top, stomach_bot)

        # No matching pose — approximate width from face alone.
        half_w = (face_w / 2.0) * self.FACE_ONLY_WIDTH_MULT
        return self._box_from_extent(face_cx - half_w, face_cx + half_w, head_top, stomach_bot)

    def _box_from_pose_only(self, pose_landmarks_list):
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
        # Approximate face height from nose-to-shoulder distance — a face
        # is roughly 1/2.2 of the nose-to-shoulder span for an upright pose.
        approx_face_h = (shoulder_y - nose.y) / 2.2
        head_top      = nose.y - approx_face_h * (1.0 + self.HEAD_TOP_MARGIN)
        stomach_bot   = nose.y - approx_face_h + approx_face_h * self.HEAD_TO_STOMACH_HEADS

        shoulder_span = shoulder_x1 - shoulder_x0
        margin = shoulder_span * self.ARM_SIDE_MARGIN
        return self._box_from_extent(shoulder_x0 - margin, shoulder_x1 + margin,
                                       head_top, stomach_bot)

    def _box_from_extent(self, x0, x1, y0, y1):
        """Converts an (x0,x1,y0,y1) anatomical extent into (cx, cy, hw, hh),
        expanding whichever dimension is needed to match the output aspect
        ratio (so the box never distorts the displayed image).

        IMPORTANT: hw/hh are hard-capped at 0.5 regardless of aspect ratio
        or MAX_CROP_FRACTION, since a normalized half-size > 0.5 ALWAYS
        means the box extends past the actual camera frame in that
        dimension — there is no aspect-ratio correction that changes
        this; it's a hard fact about normalized coordinates. The
        previous version multiplied the MAX_CROP_FRACTION ceiling by
        the aspect ratio, which let hw reach ~0.667 (out of bounds).
        get_crop_region() then reported that out-of-bounds box to
        main.py for ALL its coordinate math, while _extract_crop
        separately and silently fell back to the full frame when asked
        to actually extract it — main.py kept dividing landmark
        coordinates by the wrong, too-large half-size while the
        displayed image was really the full frame at a different
        effective scale, producing badly wrong landmark positions (the
        "small box in a random corner" symptom). Capping here instead
        keeps what get_crop_region() reports and what gets displayed
        always consistent with each other.
        """
        cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
        hw, hh = (x1 - x0) / 2.0, (y1 - y0) / 2.0

        out_w, out_h = self.OUTPUT_SIZE
        aspect = out_w / out_h
        if hw / hh > aspect:
            hh = hw / aspect
        else:
            hw = hh * aspect

        min_half = self.MIN_CROP_FRACTION / 2.0
        max_half = self.MAX_CROP_FRACTION / 2.0
        # Hard cap at 0.5 (true frame half-width/height) regardless of
        # aspect-ratio scaling — see docstring above.
        hw = self._clamp(hw, min_half * aspect, min(max_half * aspect, 0.5))
        hh = hw / aspect
        # Re-clamp hh too, in case deriving it from a capped hw still
        # leaves hh > 0.5 in an unusual aspect-ratio configuration.
        if hh > 0.5:
            hh = 0.5
            hw = hh * aspect
        return cx, cy, hw, hh

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

    # ── Smoothing ────────────────────────────────────────────────────────────
    def _smooth(self, cx, cy, hw, hh, w, h):
        """
        EMA-smooths center (cx, cy) AND size (hw, hh). The old
        proportional-control sizing approach rate-limited itself
        (MAX_STEP_FRAC), so size was passed through unsmoothed here to
        avoid double-lagging it. Direct anatomical measurement has no
        such self-rate-limiting — it's a fresh geometric measurement
        every frame — so it needs this smoothing to avoid visibly
        jittering with normal landmark noise.
        """
        if self._smoothed_cx is None:
            self._smoothed_cx, self._smoothed_cy = cx, cy
            self._smoothed_hw, self._smoothed_hh = hw, hh
        else:
            a = self.SMOOTH_ALPHA
            self._smoothed_cx = a * cx + (1 - a) * self._smoothed_cx
            self._smoothed_cy = a * cy + (1 - a) * self._smoothed_cy
            self._smoothed_hw = a * hw + (1 - a) * self._smoothed_hw
            self._smoothed_hh = a * hh + (1 - a) * self._smoothed_hh

        # Keep the box fully inside [0,1] normalized bounds given the
        # current half-size (centering can otherwise push the box out
        # of frame near the edges). IMPORTANT: when the box is wide/tall
        # enough that 2*hw or 2*hh >= 1.0 (i.e. it already spans the
        # whole frame in that dimension), the "valid" clamp range
        # [hw, 1-hw] becomes INVERTED (hw > 1-hw) — max(lo, min(hi, v))
        # with lo > hi collapses to a fixed, wrong value regardless of
        # where the box actually wants to be centered, which is exactly
        # what caused tracking to appear "offset" once CHEST_EXTENSION
        # and the lower TARGET_FACE_TO_ZONE_RATIO made the box this big.
        # When that happens, just center on 0.5 along that axis instead.
        if 2 * hw >= 1.0:
            self._smoothed_cx = 0.5
        else:
            self._smoothed_cx = self._clamp(self._smoothed_cx, hw, 1.0 - hw)
        if 2 * hh >= 1.0:
            self._smoothed_cy = 0.5
        else:
            self._smoothed_cy = self._clamp(self._smoothed_cy, hh, 1.0 - hh)
        return self._smoothed_cx, self._smoothed_cy, self._smoothed_hw, self._smoothed_hh

    # ── Extraction / output ──────────────────────────────────────────────────
    def _extract_crop(self, frame, cx, cy, hw, hh, w, h):
        """
        Returns a region of EXACTLY (2*hw*w, 2*hh*h) pixels — the full
        size implied by (cx, cy, hw, hh) — by SHIFTING the box's position
        (not padding/replicating pixels) whenever it would otherwise
        extend past the real frame boundaries. The box may end up not
        perfectly centered on the original (cx, cy) when that's too
        close to an edge for the requested size to fit, but the
        returned crop is always real camera content, never synthetic
        edge-replicated pixels, and always exactly the requested size.

        The SIZE guarantee matters because main.py's coordinate-
        conversion math (used to draw hand/pose/face landmarks aligned
        with the displayed crop) assumes the returned crop spans EXACTLY
        (cx-hw, cy-hh) to (cx+hw, cy+hh) in normalized space. Simply
        clamping the slice to the frame's bounds without shifting would
        silently return a SMALLER region near edges while the
        conversion math still divided by the original, larger size —
        that mismatch is what caused hand/pose skeletons to visibly
        drift from the real person's position. Shifting (rather than
        padding) preserves that fix while avoiding any replicated-pixel
        artifacts in the displayed image.
        """
        x0 = int(round((cx - hw) * w))
        x1 = int(round((cx + hw) * w))
        y0 = int(round((cy - hh) * h))
        y1 = int(round((cy + hh) * h))

        target_w = x1 - x0
        target_h = y1 - y0
        if target_w <= 0 or target_h <= 0:
            return frame  # degenerate box — bail to full frame rather than crash

        # If the box is wider/taller than the frame itself, it can never
        # fit regardless of shifting — fall back to the full frame
        # rather than distorting the request.
        if target_w > w or target_h > h:
            return frame

        # Shift x0..x1 so the box fits within [0, w), preserving its
        # width exactly. Same for y0..y1 within [0, h).
        if x0 < 0:
            x1 -= x0; x0 = 0
        elif x1 > w:
            x0 -= (x1 - w); x1 = w
        if y0 < 0:
            y1 -= y0; y0 = 0
        elif y1 > h:
            y0 -= (y1 - h); y1 = h

        return frame[y0:y1, x0:x1]

    def _resize_to_output(self, frame):
        return cv2.resize(frame, self.OUTPUT_SIZE, interpolation=cv2.INTER_LINEAR)

    @staticmethod
    def _clamp(v, lo, hi):
        return max(lo, min(hi, v))