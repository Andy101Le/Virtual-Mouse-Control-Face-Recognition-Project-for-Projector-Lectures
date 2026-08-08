"""Offline check of the detector-telephoto crop, landmark remap, and the
crop's two escape hatches (walk-away tightening, bystander watchdog)."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from landmark_pipeline import LandmarkPipeline
import gesture_session as gs

W, H = gs.CAPTURE_W, gs.CAPTURE_H


class LM:
    def __init__(self, x, y, z=0.0, vis=1.0):
        self.x, self.y, self.z, self.visibility = x, y, z, vis


class FakeAuth:
    def __init__(self, nose=None, size=None, recognised=False):
        self.face_nose_pos = nose
        self.face_size = size
        self.is_registered_face_visible = recognised


class FakeSession:
    """Just enough state to exercise the ROI logic."""
    _detect_window   = gs.GestureSession._detect_window
    _update_roi      = gs.GestureSession._update_roi
    _blend_face_size = gs.GestureSession._blend_face_size
    _pose_face_size  = staticmethod(gs.GestureSession._pose_face_size)

    def __init__(self, center=None, face_size=None):
        self._roi_center = center
        self._roi_face_size = face_size
        self._roi_last_seen = 0.0
        self._roi_recognised_t = 0.0
        self._roi_search_until = 0.0
        self.auth = FakeAuth()


def body(cx=0.5, cy=0.4, head=0.10, vis=0.9):
    """A pose whose head landmarks span `head` of the frame."""
    lms = [LM(cx, cy, vis=0.0) for _ in range(33)]
    lms[0] = LM(cx, cy, vis=vis)                       # nose
    lms[2] = LM(cx - head * .2, cy - head * .1, vis=vis)
    lms[5] = LM(cx + head * .2, cy - head * .1, vis=vis)
    lms[7] = LM(cx - head * .5, cy, vis=vis)           # ears set the span
    lms[8] = LM(cx + head * .5, cy, vis=vis)
    return lms


print("== _detect_window ==")
assert FakeSession(None, None)._detect_window(W, H) is None
print("  no anchor            -> full frame            OK")
assert FakeSession((0.5, 0.5), 0.30)._detect_window(W, H) is None
print("  close subject        -> full frame            OK")

win = FakeSession((0.5, 0.5), 0.02)._detect_window(W, H)
x0, y0, cw, ch = win
print(f"  distant subject      -> crop {cw}x{ch} at ({x0},{y0})")
assert abs((cw / ch) - (W / H)) < 0.01, "crop must keep frame aspect ratio"
gain = (gs.DETECT_W / cw) / (gs.DETECT_W / W)
print(f"  -> {gain:.2f}x the pixels on a distant face")
assert gain > 1.9

for cx, cy in ((0.0, 0.0), (1.0, 1.0), (0.02, 0.98)):
    x0, y0, cw, ch = FakeSession((cx, cy), 0.02)._detect_window(W, H)
    assert 0 <= x0 <= W - cw and 0 <= y0 <= H - ch, f"crop escaped at {cx},{cy}"
print("  edge anchors         -> clamped in frame      OK")

print()
print("== REGRESSION: walk-away must keep tightening the crop ==")
# Was: face size froze at its last measured value once the face became too
# small to detect, so the crop never engaged in the walk-away case.
s = FakeSession()
s.auth = FakeAuth(nose=(0.5, 0.4), size=0.22, recognised=True)
s._update_roi([body(head=0.22)], 0.0)
assert s._detect_window(W, H, 1.0) is None, "close subject should be full-frame"
print(f"  at 6ft  face~0.22 -> crop {s._detect_window(W, H, 1.0)}")

# Face now undetectable; only the body remains, and it shrinks as they recede.
s.auth = FakeAuth(nose=None, size=None, recognised=False)
t = 1.0
for head in (0.16, 0.11, 0.07, 0.05, 0.04, 0.03, 0.03, 0.03):
    t += 0.2
    s.auth.is_registered_face_visible = True    # still recognised by pose ROI
    s._roi_recognised_t = t                     # keep watchdog quiet
    s._update_roi([body(head=head)], t)
win = s._detect_window(W, H, t)
print(f"  walked away, pose-only -> face_size est {s._roi_face_size:.3f}, crop {win}")
assert win is not None, "crop MUST engage once the subject is distant"
assert win[2] <= W * 0.6, "crop should be tight, not near-full-frame"

print()
print("== REGRESSION: bystander must not hold the crop forever ==")
# Was: a bystander's pose kept _roi_last_seen fresh, so the reset never fired
# and the real user, outside the crop, could never be found again.
s = FakeSession()
s.auth = FakeAuth(nose=None, size=None, recognised=False)
t = 0.0
for _ in range(10):                     # bystander only, never recognised
    t += 0.5
    s._update_roi([body(cx=0.2, head=0.05)], t)
    if s._roi_center is None:
        break
print(f"  after {t:.1f}s unrecognised -> roi_center = {s._roi_center}")
assert s._roi_center is None, "watchdog must drop the crop"
assert s._detect_window(W, H, t) is None, "and force a full-frame re-search"
assert s._detect_window(W, H, t + gs.ROI_RESEARCH_SECONDS - 0.1) is None, \
    "re-search window must hold full-frame briefly"
print(f"  full-frame re-search held for {gs.ROI_RESEARCH_SECONDS}s  OK")

print()
print("== anchor continuity: nearest pose wins, not most visible ==")
s = FakeSession(center=(0.75, 0.4), face_size=0.05)
s.auth = FakeAuth()
s._roi_recognised_t = 100.0
# Bystander at 0.2 is MORE visible; tracked user at 0.75 is less visible.
s._update_roi([body(cx=0.2, vis=0.99), body(cx=0.75, vis=0.55)], 100.0)
print(f"  bystander vis .99 @0.20 vs user vis .55 @0.75 -> {s._roi_center[0]:.3f}")
assert s._roi_center[0] > 0.6, "must stay on the previously tracked person"

print()
print("== _remap round-trip ==")
x0, y0, cw, ch = FakeSession((0.35, 0.60), 0.02)._detect_window(W, H)
transform = (x0 / W, y0 / H, cw / W)
cases = {
    "crop centre": ((0.5, 0.5), ((x0 + cw / 2) / W, (y0 + ch / 2) / H)),
    "crop top-left": ((0.0, 0.0), (x0 / W, y0 / H)),
    "crop bottom-right": ((1.0, 1.0), ((x0 + cw) / W, (y0 + ch) / H)),
}
for name, (src, want) in cases.items():
    lms = [[LM(*src)]]
    LandmarkPipeline._remap(lms, transform)
    got = (lms[0][0].x, lms[0][0].y)
    assert abs(got[0] - want[0]) < 1e-6 and abs(got[1] - want[1]) < 1e-6
    print(f"  {name:<20} {src} -> ({got[0]:.4f}, {got[1]:.4f})  OK")

lms = [[LM(0.5, 0.5, 0.4)]]
LandmarkPipeline._remap(lms, transform)
assert abs(lms[0][0].z - 0.4 * (cw / W)) < 1e-9
print("  z scaled uniformly                             OK")
lms = [[LM(0.31, 0.72, 0.1)]]
LandmarkPipeline._remap(lms, (0.0, 0.0, 1.0))
assert (lms[0][0].x, lms[0][0].y, lms[0][0].z) == (0.31, 0.72, 0.1)
print("  identity transform -> unchanged                OK")

print()
print("all telephoto assertions passed")
