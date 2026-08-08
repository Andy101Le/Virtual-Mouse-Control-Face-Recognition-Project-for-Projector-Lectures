"""
Offline checks of the open-palm master switch.

Covers the pose detectors (geometric, so they're testable without a model
or a camera) and the arm/fire/reset state machine.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import control_toggle as ct
from control_toggle import ControlToggle, ARMED, AWAIT_PEACE, AWAIT_LOWER

# MediaPipe hand layout: 0 wrist, then 4 landmarks per finger.
_CHAINS = {"thumb": (1, 2, 3, 4), "index": (5, 6, 7, 8),
           "middle": (9, 10, 11, 12), "ring": (13, 14, 15, 16),
           "pinky": (17, 18, 19, 20)}
# Splay each finger sideways so they don't sit on top of one another.
_SPLAY = {"thumb": -0.9, "index": -0.3, "middle": 0.0, "ring": 0.3, "pinky": 0.6}


def hand(extended, wrist=(0.5, 0.5)):
    """
    Synthetic hand pointing 'up' (-y). `extended` names the fingers that are
    straight; the rest curl back toward the palm, which is what makes the
    tip land closer to the wrist than its own PIP joint.
    """
    pts = np.zeros((21, 3), dtype=np.float32)
    pts[0] = (wrist[0], wrist[1], 0.0)
    for name, (a, b, c, d) in _CHAINS.items():
        dx = _SPLAY[name] * 0.05
        out = name in extended
        for k, idx in enumerate((a, b, c, d), start=1):
            if out:
                reach = 0.05 * k                     # straight: tip furthest
            else:
                reach = 0.05 * min(k, 2) - 0.035 * max(0, k - 2)   # curled back
            pts[idx] = (wrist[0] + dx * k / 4.0, wrist[1] - reach, 0.0)
    return pts


ALL = ("thumb", "index", "middle", "ring", "pinky")
PEACE = ("index", "middle")

print("== pose detectors ==")
assert ct.is_open_palm(hand(ALL)), "all five extended must read as open palm"
assert not ct.is_open_palm(hand(PEACE)), "peace sign is not an open palm"
assert not ct.is_open_palm(hand(())), "closed fist is not an open palm"
assert not ct.is_open_palm(hand(("index",))), "pointing is not an open palm"
print("  open palm: yes for 5 fingers, no for peace/fist/point   OK")

assert ct.is_peace_sign(hand(PEACE)), "index+middle out must read as peace"
assert ct.is_peace_sign(hand(("thumb", "index", "middle"))), \
    "thumb is ignored for the peace sign"
assert not ct.is_peace_sign(hand(ALL)), "open palm is not a peace sign"
assert not ct.is_peace_sign(hand(("index",))), "one finger is not a peace sign"
print("  peace sign: thumb-agnostic, rejects palm/point          OK")

# Both must survive the hand being held at any angle, since people don't
# present a perfectly upright palm.
def rotate(pts, deg):
    r = np.radians(deg)
    c, s = np.cos(r), np.sin(r)
    out = pts.copy()
    rel = pts[:, :2] - pts[0, :2]
    out[:, 0] = pts[0, 0] + rel[:, 0] * c - rel[:, 1] * s
    out[:, 1] = pts[0, 1] + rel[:, 0] * s + rel[:, 1] * c
    return out

for deg in (0, 30, 90, 180, 270):
    assert ct.is_open_palm(rotate(hand(ALL), deg)), f"open palm failed at {deg} deg"
    assert ct.is_peace_sign(rotate(hand(PEACE), deg)), f"peace failed at {deg} deg"
print("  both hold under rotation (0-270 deg)                    OK")

# ...and at any apparent size, since the user may be 1 ft or 20 ft away.
for scale in (0.25, 1.0, 3.0):
    small = hand(ALL).copy()
    small[:, :2] = small[0, :2] + (small[:, :2] - small[0, :2]) * scale
    assert ct.is_open_palm(small), f"open palm failed at scale {scale}"
print("  scale-invariant                                         OK")

print()
print("== state machine: hold to fire, peace + lower to re-arm ==")
t = ControlToggle(enabled=True)
now = 0.0
palm = [hand(ALL)]


def run(pts, seconds, step=0.05):
    """Feed a pose for a while; return how many times it toggled."""
    global now
    fires = 0
    for _ in range(int(seconds / step)):
        now += step
        if t.update(pts, now):
            fires += 1
    return fires

assert t.enabled and t.state == ARMED
assert run(palm, 0.5) == 0, "must not fire before the hold completes"
print(f"  0.5s of open palm -> no fire (progress {t.hold_progress:.0%})")

assert run(palm, 0.8) == 1, "should fire once the hold passes 1s"
assert not t.enabled, "controls must now be OFF"
assert t.state == AWAIT_PEACE
print("  1.3s total        -> FIRED, controls OFF, awaiting peace")

# The critical property: holding the palm up must not toggle repeatedly.
assert run(palm, 5.0) == 0, "must not re-fire while the palm stays up"
assert not t.enabled
print("  +5s of open palm  -> no repeat fire                     OK")

assert run([hand(PEACE)], 0.5) == 0
assert t.state == AWAIT_LOWER
print("  peace sign        -> reset acknowledged")

# A hand left in the peace sign must not silently re-arm.
assert run([hand(PEACE)], 2.0) == 0 and t.state == AWAIT_LOWER
assert run([], 0.5) == 0
assert t.state == ARMED
print("  lower hand        -> re-armed                           OK")

assert run(palm, 1.3) == 1 and t.enabled, "second toggle should switch back ON"
print("  open palm again   -> FIRED, controls back ON            OK")

print()
print("== only the tracked user's hands count ==")
# update() is fed only hands already confirmed as the user's, so an empty
# list is how a bystander-only frame arrives.
t2 = ControlToggle(enabled=True)
now2 = 0.0
for _ in range(60):
    now2 += 0.05
    assert not t2.update([], now2), "no user hands must never toggle"
assert t2.enabled
print("  bystander-only frames never toggle                      OK")

print()
print("== hand must be RAISED, not just open ==")
t3 = ControlToggle(enabled=True)
low = [hand(ALL, wrist=(0.5, 0.95))]          # hanging at the side
nose, fs = (0.5, 0.30), 0.10
now3 = 0.0
for _ in range(60):
    now3 += 0.05
    assert not t3.update(low, now3, nose_pos=nose, face_size=fs), \
        "a hand hanging low must not fire"
high = [hand(ALL, wrist=(0.5, 0.28))]
fired = False
for _ in range(60):
    now3 += 0.05
    fired = fired or t3.update(high, now3, nose_pos=nose, face_size=fs)
assert fired, "a raised open palm must fire"
print("  low hand ignored, raised hand fires                     OK")

print()
print("== manual override resets the gesture state ==")
t4 = ControlToggle(enabled=True)
now4 = 0.0
for _ in range(40):
    now4 += 0.05
    t4.update(palm, now4)
assert not t4.enabled and t4.state == AWAIT_PEACE
t4.set_enabled(True)
assert t4.enabled and t4.state == ARMED, \
    "dashboard override must re-arm, not leave the switch mid-sequence"
print("  set_enabled() re-arms cleanly                           OK")

print()
print("all control-toggle assertions passed")
