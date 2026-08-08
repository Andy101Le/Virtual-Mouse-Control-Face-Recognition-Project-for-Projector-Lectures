"""
bt_cursor_controller.py
────────────────────────
Drop-in replacement for
that drives
the paired PC over Bluetooth HID instead of the local machine via
pyautogui.

Deliberately keeps the SAME public interface (CAM_MARGIN,
absolute_to_screen, handle_action) so nothing else in the pipeline has
to change — main.py / gesture_session.py just construct this one
instead.

KEY DIFFERENCE FROM THE PYAUTOGUI VERSION
  pyautogui needed to convert normalized camera coords into *pixel*
  coordinates, so it had to know the screen resolution. Bluetooth HID
  absolute reporting doesn't: we hand the host a 0..32767 fraction on
  each axis and the host maps it across whatever screen it has. So
  there's no screen_w/screen_h here, and no resolution to configure —
  the same Pi drives a 1080p laptop and a 4K monitor identically.

  This is also why "corner of the control zone = corner of the screen"
  holds for free: the control-zone margin math produces a clean 0.0–1.0,
  and 0.0/1.0 are by definition the host's screen edges.
"""

import time
import numpy as np


class BTCursorController:
    # Fallback control zone, used only when nobody is tracked and there is
    # therefore no measured body scale to work from.
    CAM_MARGIN     = 0.15
    CURSOR_SMOOTH  = 0.35
    CLICK_COOLDOWN = 2.0
    SCROLL_COOLDOWN = 1.2
    SCROLL_CLICKS   = 3     # wheel notches per ZOOM IN / ZOOM OUT gesture

    # ── Reach-based control zone ────────────────────────────────────────────
    # The zone used to be a fixed box (0.15..0.85 of the frame) at every
    # distance, which meant reaching the screen edge required sweeping your
    # hand 85% of the way across the CAMERA FRAME however far away you
    # stood. Up close you fill the frame and that's easy; at 20 ft you span
    # maybe 15% of it, so a fully extended arm only ever reached the middle
    # fifth of the screen.
    #
    # So the zone is now sized from how big you appear. Anthropometry:
    # arm span ~= height, so reach from the body midline to the fingertip is
    # ~0.5 x height; head height is ~1/7.5 of height and get_face_size()
    # reports the face-mesh bbox DIAGONAL, ~1.2x head height, i.e. ~0.16 x
    # height. That puts horizontal half-reach at ~3.1 face diagonals. We use
    # slightly less so the screen edge lands at a comfortable extension
    # rather than demanding a full strain, and less again vertically —
    # arms sweep a shorter arc up/down than side to side.
    REACH_HALF_W = 2.6      # half-width of the zone, in face diagonals

    # Vertical reach is NOT symmetric, and treating it as if it were is what
    # made the bottom of the screen unreachable. The old symmetric height
    # plus a chest-drop offset worked out to 0.7 face diagonals above the
    # nose and 2.9 below — so raising your arm did almost nothing, while
    # hitting the bottom of the screen meant pointing down at your own hip
    # (~0.46 x body height below the nose), which is both awkward and often
    # occluded by your torso. People point with the hand raised, so the zone
    # now reaches further up than down.
    REACH_UP     = 1.6      # above the nose, in face diagonals
    REACH_DOWN   = 1.5      # below the nose, in face diagonals

    # Room the zone must leave between its edge and the edge of what the
    # camera can see. MediaPipe needs the WHOLE hand to detect it, not just
    # the fingertip we steer with — so the fingertip cannot go right up to
    # the frame edge, and how much room it needs scales with how big the
    # hand appears, i.e. with face_size.
    #
    # It is also strongly asymmetric. In the MOVE gesture the index finger
    # points up and the palm and wrist hang BELOW the fingertip, so putting
    # the fingertip low in frame pushes the rest of the hand off the bottom
    # and detection simply drops. A hand is ~0.106 x body height and a face
    # diagonal ~0.16 x height, so a hand is ~0.66 face diagonals long; below
    # gets that plus slack, above only needs enough for the fingertip
    # itself. This is what a fixed 0.02 inset got wrong: at 1-3 ft it left a
    # 0.020 gap where the hand needed 0.23.
    HAND_MARGIN_BELOW = 0.90    # in face diagonals
    HAND_MARGIN_ABOVE = 0.25
    HAND_MARGIN_SIDE  = 0.50    # a hand turned sideways is about as wide

    # Absolute floor, for when nobody is tracked and there is no face_size.
    EDGE_INSET   = 0.02
    MIN_HALF     = 0.06
    ZONE_SMOOTH  = 0.15     # EMA so the zone glides as you move rather than
                            # snapping and dragging the cursor with it

    def __init__(self, hid_device):
        """hid_device: a started BluetoothHIDDevice."""
        self.hid = hid_device

        # Cursor position is tracked in normalized 0–1 space now, not px.
        self.cursor_nx = 0.5
        self.cursor_ny = 0.5

        # Current control zone as (x0, y0, x1, y1) in normalized frame
        # coords. Starts as the legacy fixed box and is replaced by the
        # reach-based one as soon as a body scale is known.
        m = self.CAM_MARGIN
        self.zone = (m, m, 1.0 - m, 1.0 - m)
        self._have_scale = False

        self._last_click_time  = {}
        self._last_scroll_time = {}

    # ── Control zone ────────────────────────────────────────────────────────
    def set_control_zone(self, anchor, face_size, detect_window=None):
        """
        Resize/reposition the control zone for the tracked user. Call once
        per frame with their nose position and face size (both normalized),
        or with None to fall back to the fixed box.

        detect_window: (x0, y0, x1, y1) normalized bounds of the region
        MediaPipe is actually looking at this frame, when the detector
        telephoto has cropped it. The zone is clamped inside it — a zone
        extending past the crop would map part of the screen to hand
        positions that cannot be detected at all, so that part of the screen
        would simply be unreachable.
        """
        if anchor is None or not face_size:
            self._have_scale = False
            m = self.CAM_MARGIN
            self._blend_zone((m, m, 1.0 - m, 1.0 - m))
            return

        fs = float(face_size)
        half_w = max(self.MIN_HALF, self.REACH_HALF_W * fs)
        up     = max(self.MIN_HALF, self.REACH_UP * fs)
        down   = max(self.MIN_HALF, self.REACH_DOWN * fs)

        # Bounds the FINGERTIP may occupy: the frame (or the detection crop
        # when one is active), inset by enough room for the rest of the hand
        # to stay visible. Asymmetric because the palm hangs below the
        # fingertip in the pointing gesture — see HAND_MARGIN_BELOW.
        bx0, by0, bx1, by1 = detect_window or (0.0, 0.0, 1.0, 1.0)
        m_side  = max(self.EDGE_INSET, self.HAND_MARGIN_SIDE  * fs)
        m_above = max(self.EDGE_INSET, self.HAND_MARGIN_ABOVE * fs)
        m_below = max(self.EDGE_INSET, self.HAND_MARGIN_BELOW * fs)
        bx0, bx1 = bx0 + m_side,  bx1 - m_side
        by0, by1 = by0 + m_above, by1 - m_below

        # Up close the frame is simply not tall or wide enough to contain a
        # full arm sweep plus that margin. Degenerate bounds would otherwise
        # invert and fold the zone inside out, so collapse to a centred
        # sliver instead and let the MIN_HALF floors below take over.
        if bx1 <= bx0:
            bx0 = bx1 = (bx0 + bx1) / 2.0
        if by1 <= by0:
            by0 = by1 = (by0 + by1) / 2.0

        # A zone larger than that region would put the screen edge at a hand
        # position the camera cannot resolve, so cap it there.
        half_w = min(half_w, max((bx1 - bx0) / 2.0, self.MIN_HALF))
        span_v = max(by1 - by0, 2 * self.MIN_HALF)
        if up + down > span_v:
            # Preserve the up:down ratio while fitting — squashing only one
            # side would silently reintroduce the imbalance this replaced.
            k = span_v / (up + down)
            up, down = up * k, down * k

        cx = float(anchor[0])
        cy = float(anchor[1])

        # Shift (don't shrink) the box to fit. Clipping one side instead
        # would make the same hand movement travel further on screen going
        # left than going right.
        cx = min(max(cx, bx0 + half_w), bx1 - half_w)
        cy = min(max(cy, by0 + up),     by1 - down)

        self._have_scale = True
        self._blend_zone((cx - half_w, cy - up, cx + half_w, cy + down))

    def _blend_zone(self, target):
        a = self.ZONE_SMOOTH
        self.zone = tuple(z + a * (t - z) for z, t in zip(self.zone, target))

    @property
    def reach_radius(self):
        """Half the zone's width — the normalized distance a fully extended
        arm covers. Used to scale anything else that needs to reason about
        'how far from the user is still the user', instead of hardcoding a
        radius that only holds at one distance."""
        x0, _, x1, _ = self.zone
        return max((x1 - x0) / 2.0, self.MIN_HALF)

    @property
    def connected(self):
        return self.hid.connected

    def absolute_to_screen(self, nx, ny):
        """
        Maps a raw camera coordinate into normalized screen space by
        rescaling it across the current control zone. Returns 0.0–1.0 (not
        pixels — see the module docstring), so the zone's corners are the
        host screen's corners at any distance.
        """
        x0, y0, x1, y1 = self.zone
        sx = float(np.clip((nx - x0) / max(x1 - x0, 1e-6), 0.0, 1.0))
        sy = float(np.clip((ny - y0) / max(y1 - y0, 1e-6), 0.0, 1.0))
        return sx, sy

    def handle_action(self, confirmed, hand_id, tip_xy):
        """
        confirmed: debounced gesture label ('MOVE', 'LEFT CLICK', ...)
        tip_xy:    normalized (x, y) fingertip position, used for MOVE.

        No-ops safely when no host is connected, so the gesture pipeline
        can keep running (and previewing) before anything is paired.
        """
        if not self.hid.connected:
            return

        now = time.perf_counter()
        self._last_click_time.setdefault(hand_id, 0.0)
        self._last_scroll_time.setdefault(hand_id, 0.0)

        if confirmed == 'MOVE':
            tgt_x, tgt_y = self.absolute_to_screen(float(tip_xy[0]), float(tip_xy[1]))
            self.cursor_nx += self.CURSOR_SMOOTH * (tgt_x - self.cursor_nx)
            self.cursor_ny += self.CURSOR_SMOOTH * (tgt_y - self.cursor_ny)
            self.hid.move_absolute(self.cursor_nx, self.cursor_ny)

        elif confirmed == 'LEFT CLICK':
            if (now - self._last_click_time[hand_id]) > self.CLICK_COOLDOWN:
                self.hid.click('left')
                self._last_click_time[hand_id] = now

        elif confirmed == 'RIGHT CLICK':
            if (now - self._last_click_time[hand_id]) > self.CLICK_COOLDOWN:
                self.hid.click('right')
                self._last_click_time[hand_id] = now

        elif confirmed == 'ZOOM IN':
            # Ctrl+wheel — the chord virtually every host app binds to
            # zoom (a bare wheel would just scroll the page). Needs the
            # composite pointer+keyboard HID descriptor in bluetooth_hid.
            if (now - self._last_scroll_time[hand_id]) > self.SCROLL_COOLDOWN:
                self.hid.ctrl_scroll(self.SCROLL_CLICKS)
                self._last_scroll_time[hand_id] = now

        elif confirmed == 'ZOOM OUT':
            if (now - self._last_scroll_time[hand_id]) > self.SCROLL_COOLDOWN:
                self.hid.ctrl_scroll(-self.SCROLL_CLICKS)
                self._last_scroll_time[hand_id] = now
