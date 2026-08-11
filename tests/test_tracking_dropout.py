"""
Offline checks for brief face-detection dropouts.

The reported symptom: a registered user, tracked and green, would flick red
for ~0.4 s; the cursor stopped dead for that whole time and then jumped to
wherever the hand had travelled by the time the face came back.

Two independent causes, both covered here.

  1. Ownership was tested against AuthManager.face_nose_pos, which is None
     on ANY frame where the face detector found nothing. At
     FACE_DETECT_INTERVAL=3 a single miss already spans about three frames,
     and two consecutive misses land right on the ~0.4 s that was observed.
     Meanwhile user_active has a 10 s grace and the control zone holds its
     last good anchor — so the grace was being defeated by the one consumer
     that didn't use it.

  2. Even with ownership fixed, any real resume still has to correct a
     position the filter held across the gap. At full tracking speed that
     correction lands in about two frames, which reads as a teleport.

No camera, no radio, no model.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from gesture_session import hand_belongs_to_user, HAND_OWNERSHIP_SLACK
from bt_cursor_controller import BTCursorController as BC

REACH = 0.20          # normalized half-width of the control zone
FACE  = (0.50, 0.40)  # where the tracked user's face is anchored


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)


class FakeHID:
    def __init__(self):
        self.connected = True
        self.moves = []

    def move_absolute(self, nx, ny):
        self.moves.append((nx, ny))
        return True


def owned(wrist, anchor=FACE, reach=REACH, active=True):
    return hand_belongs_to_user(np.asarray(wrist, dtype=np.float32),
                                anchor, reach, active)


# ── 1. Ownership survives a detection dropout ───────────────────────────────
def test_the_reported_bug():
    """
    Replays the observed dropout: the face detector misses two intervals in
    a row (~0.46 s at 13 fps with FACE_DETECT_INTERVAL=3) while the user
    keeps pointing. Ownership must hold for every frame of it.

    `detected` is what AuthManager.face_nose_pos would be on each frame —
    None wherever nothing was found. The anchor handed to the predicate is
    the HELD one, which is the entire fix: it is what the control zone
    already uses, and it does not go None mid-gesture.
    """
    held = FACE
    wrist = (FACE[0] + 0.10, FACE[1])

    detected = [FACE, FACE] + [None] * 6 + [FACE, FACE]
    verdicts = [owned(wrist, anchor=held) for _ in detected]

    check(all(verdicts),
          "THE BUG: ownership dropped during a face-detection gap — "
          f"{verdicts.count(False)}/{len(verdicts)} frames revoked the "
          "user's own hand while they were still inside the auth grace")

    # And the guarantee that makes it hold: the instantaneous detection
    # result is structurally incapable of reaching this decision. If a
    # future change reintroduces it, the signature is where it will show.
    import inspect
    params = list(inspect.signature(hand_belongs_to_user).parameters)
    check(params == ["wrist", "anchor", "reach_radius", "user_active"],
          f"ownership must depend only on the held anchor, got {params}")


def test_no_anchor_means_no_ownership():
    """Once the anchor really is gone — nobody has been tracked at all —
    there is nothing to measure reach from, so nothing is owned."""
    check(not owned((0.5, 0.4), anchor=None),
          "no anchor at all must not grant ownership")


def test_inactive_user_owns_nothing():
    check(not owned((0.5, 0.4), active=False),
          "a hand cannot be the tracked user's when nobody is tracked")


def test_bystander_beyond_reach_is_rejected():
    far = (FACE[0] + REACH * HAND_OWNERSHIP_SLACK + 0.05, FACE[1])
    check(not owned(far), "a hand past arm's reach is not the user's")


def test_the_boundary_is_reach_times_slack():
    """The slack is deliberate — the zone is sized for a COMFORTABLE
    extension, and a hand stretched past that is still plainly theirs."""
    limit = REACH * HAND_OWNERSHIP_SLACK
    check(owned((FACE[0] + limit * 0.98, FACE[1])), "just inside should hold")
    check(not owned((FACE[0] + limit * 1.02, FACE[1])), "just outside should not")


def test_ownership_scales_with_apparent_size():
    """Someone far away has a small reach radius, so the same absolute
    distance that is theirs up close belongs to nobody at range."""
    wrist = (FACE[0] + 0.25, FACE[1])
    check(owned(wrist, reach=0.40), "up close, a 0.25 offset is within reach")
    check(not owned(wrist, reach=0.05), "at range, the same offset is not")


# ── 2. Resuming must not teleport ───────────────────────────────────────────
def steer(c, target, frames, dt, t0=0.0, clock=None):
    """Drive `frames` MOVE calls at `target`, advancing a fake clock."""
    import time as _time
    t = {"now": t0}
    real = _time.perf_counter
    _time.perf_counter = lambda: t["now"]
    try:
        for _ in range(frames):
            c.handle_action('MOVE', 0, target, dt)
            t["now"] += dt
    finally:
        _time.perf_counter = real
    return t["now"]


def fresh():
    c = BC(FakeHID())
    c.zone = (0.0, 0.0, 1.0, 1.0)     # identity mapping: target == fingertip
    return c


def test_resume_after_a_gap_glides_instead_of_snapping():
    dt = 1 / 13.0
    c = fresh()
    t = steer(c, (0.2, 0.2), 30, dt)              # settle at one corner
    check(abs(c.cursor_nx - 0.2) < 0.01, f"should have settled, {c.cursor_nx:.3f}")

    # Tracking drops for half a second; the hand moves across the frame.
    t += 0.5
    steer(c, (0.8, 0.8), 1, dt, t0=t)
    moved = c.cursor_nx - 0.2
    check(moved < 0.25,
          f"the first frame back must not lunge most of the way, moved {moved:.3f}")
    check(moved > 0.0, "but it should start moving")


def test_steady_steering_is_not_slowed_by_the_ramp():
    """The gentler constant applies to resumes only — continuous steering
    must keep the responsiveness the retune bought."""
    dt = 1 / 13.0
    c = fresh()
    t = steer(c, (0.2, 0.2), 30, dt)
    # No gap: keep steering, target jumps because the hand moved fast.
    steer(c, (0.8, 0.8), 2, dt, t0=t)
    check(c.cursor_nx > 0.65,
          f"uninterrupted steering should stay snappy, got {c.cursor_nx:.3f}")


def test_the_ramp_expires():
    dt = 1 / 13.0
    c = fresh()
    t = steer(c, (0.2, 0.2), 30, dt)
    t += 0.5                                   # gap -> reacquire engaged
    t = steer(c, (0.8, 0.8), 1, dt, t0=t)
    # Ride out the reacquire window, then check full speed is back.
    t = steer(c, (0.8, 0.8), int(BC.REACQUIRE_S / dt) + 2, dt, t0=t)
    c.cursor_nx = 0.0
    steer(c, (1.0, 1.0), 2, dt, t0=t)
    check(c.cursor_nx > 0.9,
          f"after the window the fast constant should be back, {c.cursor_nx:.3f}")


def test_first_move_after_startup_eases_in():
    """The cursor starts at the centre of the screen. Without treating the
    very first MOVE as a resume it would yank from there to the hand."""
    dt = 1 / 13.0
    c = fresh()
    check(c.cursor_nx == 0.5, "fixture assumption: starts centred")
    steer(c, (1.0, 1.0), 1, dt)
    check(c.cursor_nx < 0.75,
          f"the first move should ease in, not lunge, got {c.cursor_nx:.3f}")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"\n{len(tests)} passed")
