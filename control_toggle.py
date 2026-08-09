"""
control_toggle.py
──────────────────
Hand-gesture master switch for cursor control.

Raise an open hand — all five fingers extended — hold it for about a
second, and every control output stops: no cursor movement, no clicks, no
scrolls. Do it again to bring them back. The camera keeps tracking and the
preview keeps running throughout, so the gesture that re-enables control is
still seen and you stay framed while paused.

WHY THIS IS NOT A MODEL CLASS. GestureEngine's gesture model has exactly five
outputs (MOVE, LEFT/RIGHT CLICK, ZOOM IN/OUT); adding a sixth would mean
retraining it. An open palm is also one of the easiest poses to identify
geometrically, so it is detected here directly from the 21 hand landmarks
instead. That is not a compromise — a landmark test is scale-invariant and
rotation-invariant for free, where the model has to be taught both, and it
lets the switch take priority over whatever the model would have said about
the same frame. Every finger test compares distances from the wrist in 3D,
so it holds however the hand is turned.

RE-ARMING. An open palm occurs naturally when talking, waving or just
resting a hand, so firing once and immediately watching for the next open
palm would toggle repeatedly. After a toggle the switch will not fire again
until it has seen a deliberate reset: a peace sign, then the hand lowered
(or simply out of the pose). Two distinct poses in sequence, which does not
happen by accident.

    raise open palm, hold ~1s  ->  TOGGLE FIRES
    peace sign                 ->  reset acknowledged
    lower hand                 ->  re-armed
"""

import numpy as np

# ── Landmark groups (MediaPipe hand: 0 wrist, then 4 per finger) ────────────
_FINGERS = {                 # name: (mcp, pip, tip)
    "index":  (5, 6, 8),
    "middle": (9, 10, 12),
    "ring":   (13, 14, 16),
    "pinky":  (17, 18, 20),
}
_WRIST      = 0
_THUMB_IP   = 3
_THUMB_TIP  = 4
_PINKY_MCP  = 17

# A finger counts as extended when its tip is this much further from the
# wrist than its own PIP joint. Ratios rather than absolute distances, so
# the test is independent of how large the hand appears.
_EXTEND_RATIO = 1.15
_FOLD_RATIO   = 1.02     # below this the finger is curled

# The thumb folds across the palm rather than toward the wrist, so it is
# measured against the pinky knuckle instead: an extended thumb puts its tip
# further from that knuckle than its own IP joint.
_THUMB_EXTEND_RATIO = 1.06

# States
ARMED       = "armed"          # watching for an open palm
AWAIT_PEACE = "await_peace"    # fired; needs a peace sign to begin resetting
AWAIT_LOWER = "await_lower"    # peace seen; needs the hand lowered to re-arm


def _dist(a, b):
    return float(np.linalg.norm(np.asarray(a, dtype=np.float64) -
                                np.asarray(b, dtype=np.float64)))


def _extension(pts, mcp, pip, tip):
    """How far past its own PIP joint a finger's tip reaches, as a ratio."""
    base = _dist(pts[pip], pts[_WRIST])
    if base < 1e-6:
        return 0.0
    return _dist(pts[tip], pts[_WRIST]) / base


def finger_states(pts):
    """{finger: extension ratio} for the four fingers, plus 'thumb'."""
    out = {name: _extension(pts, *ids) for name, ids in _FINGERS.items()}
    base = _dist(pts[_THUMB_IP], pts[_PINKY_MCP])
    out["thumb"] = (_dist(pts[_THUMB_TIP], pts[_PINKY_MCP]) / base
                    if base > 1e-6 else 0.0)
    return out


def is_open_palm(pts):
    """All five fingers extended."""
    s = finger_states(pts)
    return (all(s[f] >= _EXTEND_RATIO for f in _FINGERS) and
            s["thumb"] >= _THUMB_EXTEND_RATIO)


def is_peace_sign(pts):
    """Index and middle extended, ring and pinky curled. The thumb is
    deliberately ignored — people make this sign both ways."""
    s = finger_states(pts)
    return (s["index"] >= _EXTEND_RATIO and s["middle"] >= _EXTEND_RATIO and
            s["ring"] <= _FOLD_RATIO and s["pinky"] <= _FOLD_RATIO)


def is_toggle_pose(pts):
    """
    True if this hand is making either switch gesture.

    The Keras model classifies every hand it sees, and an open palm or a
    peace sign will still come back as one of its five labels — usually
    MOVE. Without this check, raising a palm to switch controls OFF would
    first drag the cursor to the palm for a second, and the peace sign used
    to reset could fire a click or a scroll. A hand that is talking to the
    switch is not also driving the pointer.
    """
    return is_open_palm(pts) or is_peace_sign(pts)


class ControlToggle:
    HOLD_SECONDS  = 1.0     # open palm must be held this long to fire
    PEACE_SECONDS = 0.25    # peace sign must persist this long to count
    LOWER_SECONDS = 0.25    # and the hand must stay out of the pose this long

    # The hand must be RAISED, not just open — a hand resting at your side
    # with fingers straight is an open palm too. Measured against the face,
    # so it scales with distance like everything else: the wrist must sit
    # above (nose_y + this many face diagonals). Roughly chin height.
    RAISE_ABOVE = 0.5

    def __init__(self, enabled=True):
        self.enabled = bool(enabled)   # are controls currently ON?
        self.state   = ARMED
        self._pose_since = None        # when the current qualifying pose began
        self._elapsed    = 0.0         # how long the open palm has been held
        self._last_toggle_t = 0.0

    # ── Reporting ───────────────────────────────────────────────────────────
    @property
    def owns_peace_sign(self):
        """
        True while the switch is using the peace sign as its own reset step,
        so nothing else may act on it.

        This covers the whole post-fire sequence, not just the frame the
        peace sign appears. Once an open palm has toggled, the switch is
        waiting for peace-then-lower; a peace sign anywhere in that window
        belongs to the reset. Only after the hand has been lowered and the
        switch is ARMED again does the peace sign go back to meaning
        "middle click".

        Without this the second half of a toggle would double as a click:
        switching controls back ON leaves them enabled, so the very next
        gesture — the reset peace sign — would fire a middle click the user
        never asked for.
        """
        return self.state != ARMED

    @property
    def hold_progress(self):
        """0..1 through the open-palm hold, for the HUD countdown. 0 unless
        an open palm is actually being held right now."""
        if self.state != ARMED or self._pose_since is None:
            return 0.0
        return min(1.0, self._elapsed / self.HOLD_SECONDS)

    def status(self):
        return {
            "controls_enabled": self.enabled,
            "toggle_state":     self.state,
            "hold_progress":    round(self.hold_progress, 2),
        }

    def set_enabled(self, enabled):
        """Manual override from the dashboard. Resets the gesture state so
        the switch can't immediately undo a deliberate click."""
        self.enabled = bool(enabled)
        self.state   = ARMED
        self._pose_since = None
        return self.enabled

    # ── Per-frame update ────────────────────────────────────────────────────
    def update(self, user_hands, now_t, nose_pos=None, face_size=None):
        """
        user_hands: (21, 3) landmark arrays for hands confirmed to belong to
        the tracked user — a bystander must not be able to switch off
        somebody else's controls.

        Returns True on the frame the switch actually toggles.
        """
        open_palm = any(self._raised(p, nose_pos, face_size) and is_open_palm(p)
                        for p in user_hands)
        peace     = any(is_peace_sign(p) for p in user_hands)

        if self.state == ARMED:
            return self._tick_armed(open_palm, now_t)
        if self.state == AWAIT_PEACE:
            self._advance(peace, now_t, self.PEACE_SECONDS, AWAIT_LOWER)
            return False
        # AWAIT_LOWER — the pose must be released before we re-arm, so a hand
        # left up in a peace sign doesn't silently rearm the switch.
        self._advance(not (peace or open_palm), now_t,
                      self.LOWER_SECONDS, ARMED)
        return False

    def _tick_armed(self, open_palm, now_t):
        if not open_palm:
            self._pose_since = None
            return False
        if self._pose_since is None:
            self._pose_since = now_t
        self._elapsed = now_t - self._pose_since
        if self._elapsed < self.HOLD_SECONDS:
            return False
        self.enabled        = not self.enabled
        self.state          = AWAIT_PEACE
        self._pose_since    = None
        self._last_toggle_t = now_t
        return True

    def _advance(self, condition, now_t, needed, next_state):
        if not condition:
            self._pose_since = None
            return
        if self._pose_since is None:
            self._pose_since = now_t
        if (now_t - self._pose_since) >= needed:
            self.state = next_state
            self._pose_since = None

    def _raised(self, pts, nose_pos, face_size):
        """Is this hand held up, rather than hanging at rest? Skipped when
        there's no face to measure against — better to accept the gesture
        than to make the switch unusable."""
        if nose_pos is None or not face_size:
            return True
        return float(pts[_WRIST][1]) < float(nose_pos[1]) + self.RAISE_ABOVE * float(face_size)
