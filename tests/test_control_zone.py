"""
Offline checks of the reach-based cursor control zone.

The bug this guards: the zone used to be a fixed box (0.15..0.85 of the
frame) at every distance, so reaching the host screen's edge required
sweeping your hand 85% of the way across the CAMERA FRAME however far away
you stood. Close up you fill the frame and that's easy; far away your fully
extended arm covers only a small slice of the frame, so it mapped to the
middle of the screen and the edges were unreachable.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bt_cursor_controller import BTCursorController as BC


class FakeHID:
    connected = False


# Nominal frame time. The zone filter is time-based now, so its calls need
# one; the exact value doesn't matter here because these helpers iterate to
# convergence.
DT = 1.0 / 13.0


def settled(face_size, anchor=(0.5, 0.45), detect_window=None):
    """A controller whose zone EMA has converged for this apparent size."""
    c = BC(FakeHID())
    for _ in range(400):
        c.set_control_zone(anchor, face_size, DT, detect_window=detect_window)
    return c


def reach_fraction(c, face_size, anchor=(0.5, 0.45)):
    """How much of the host screen a fully extended arm can cover, given the
    real physical reach implied by that apparent size."""
    half = BC.REACH_HALF_W * face_size          # true half-reach, normalized
    lo, _ = c.absolute_to_screen(anchor[0] - half, anchor[1])
    hi, _ = c.absolute_to_screen(anchor[0] + half, anchor[1])
    return hi - lo


print("== the original complaint: screen coverage at full arm extension ==")
print("  face_size   OLD fixed zone   NEW reach zone")
old_scale = 1.0 / (1.0 - 2 * BC.CAM_MARGIN)
for fs in (0.30, 0.20, 0.12, 0.06, 0.03):
    half = BC.REACH_HALF_W * fs
    lo = max(0.0, min(1.0, (0.5 - half - BC.CAM_MARGIN) * old_scale))
    hi = max(0.0, min(1.0, (0.5 + half - BC.CAM_MARGIN) * old_scale))
    c = settled(fs)
    new = reach_fraction(c, fs)
    print(f"  {fs:<11} {hi - lo:>6.0%}          {new:>6.0%}")
    assert new > 0.98, f"full extension must span the screen at face_size={fs}"

# The far cases are the ones the user reported; confirm the old mapping was
# genuinely broken there so this test fails loudly if someone reverts it.
half = BC.REACH_HALF_W * 0.03
old_far = ((0.5 + half - BC.CAM_MARGIN) * old_scale
           - (0.5 - half - BC.CAM_MARGIN) * old_scale)
print(f"\n  at face_size 0.03 the old zone gave {old_far:.0%} of the screen")
assert old_far < 0.30, "sanity: the old mapping really was this bad far away"

print()
print("== zone stays inside the frame, and stays symmetric ==")
for anchor in ((0.5, 0.45), (0.02, 0.02), (0.98, 0.98), (0.5, 0.9)):
    for fs in (0.03, 0.12, 0.30):
        c = settled(fs, anchor)
        x0, y0, x1, y1 = c.zone
        assert 0.0 <= x0 < x1 <= 1.0 and 0.0 <= y0 < y1 <= 1.0, \
            f"zone escaped frame at anchor={anchor} fs={fs}: {c.zone}"
        # Shifted to fit, never squashed on one side — otherwise the same
        # hand movement would travel further on screen one way than the other.
        # Width is the lesser of the user's reach and what the frame can
        # show with room for the hand itself at each side.
        m_side = max(BC.EDGE_INSET, BC.HAND_MARGIN_SIDE * fs)
        w_expect = 2 * min(BC.REACH_HALF_W * fs, max(0.5 - m_side, BC.MIN_HALF))
        assert abs((x1 - x0) - w_expect) < 1e-3, \
            f"zone width {x1-x0:.3f} != expected {w_expect:.3f} (fs={fs})"
print("  clamped by shifting, width preserved                     OK")

print()
print("== REGRESSION: bottom of screen must be reachable ==")
# Was: symmetric height plus a chest-drop put the zone bottom ~2.9 face
# diagonals BELOW the nose (roughly hip level) and only 0.7 above it, so
# reaching the bottom of the host screen meant pointing at your own thigh
# while raising your arm did nothing.
for fs in (0.20, 0.12, 0.05):
    c = settled(fs, anchor=(0.5, 0.45))
    x0, y0, x1, y1 = c.zone
    up_fs   = (0.45 - y0) / fs      # reach above the nose, in face diagonals
    down_fs = (y1 - 0.45) / fs      # reach below the nose
    print(f"  fs={fs:<5} up {up_fs:.2f} diag / down {down_fs:.2f} diag")
    assert up_fs > 1.0, "raising the arm must actually move the cursor up"
    assert down_fs < 2.0, "bottom of screen must not require reaching hip level"
    assert up_fs > down_fs, "reach up should exceed reach down"

print()
print("== REGRESSION: zone must stay inside the detection crop ==")
# Was: the detector-telephoto crop follows the face, and the zone extended
# past its bottom edge — so the lower part of the screen mapped to hand
# positions MediaPipe never saw.
for fs, crop_half in ((0.20, 0.50), (0.12, 0.30), (0.08, 0.25)):
    win = (0.5 - crop_half, 0.45 - crop_half, 0.5 + crop_half, 0.45 + crop_half)
    c = settled(fs, anchor=(0.5, 0.45), detect_window=win)
    x0, y0, x1, y1 = c.zone
    inside = (x0 >= win[0] - 1e-6 and y0 >= win[1] - 1e-6 and
              x1 <= win[2] + 1e-6 and y1 <= win[3] + 1e-6)
    print(f"  fs={fs:<5} crop +/-{crop_half}  zone y {y0:.3f}..{y1:.3f}  "
          f"{'inside' if inside else 'ESCAPED'}")
    assert inside, f"zone escaped the detection window: {c.zone} vs {win}"
    # Fitting must preserve the up:down ratio, not squash one side flat.
    up_d, down_d = 0.45 - y0, y1 - 0.45
    assert up_d > 0 and down_d > 0
    assert abs((up_d / down_d) - (BC.REACH_UP / BC.REACH_DOWN)) < 0.05, \
        "clamping must keep the up:down balance"
print("  clamped into the crop, ratio preserved                   OK")

print()
print("== REGRESSION: room below the zone for the hand to be detected ==")
# Was: a fixed 0.02 inset, so at 1-3 ft the zone ran to y=0.980 and left a
# 0.020 gap where the pointing hand needed ~0.23 of frame height. The
# fingertip is what we steer with, but MediaPipe needs the whole hand, and
# in the MOVE gesture the palm and wrist hang BELOW the fingertip.
HAND_LEN = 0.66        # face diagonals, from anthropometry
print("  fs     zone bottom   gap below   hand needs")
for fs in (0.45, 0.35, 0.25, 0.20, 0.12):
    c = settled(fs, anchor=(0.5, 0.35))
    _, _, _, y1 = c.zone
    gap, need = 1.0 - y1, HAND_LEN * fs
    print(f"  {fs:<6.2f} {y1:.3f}         {gap:.3f}       {need:.3f}")
    assert gap >= need * 0.95, (
        f"only {gap:.3f} below the zone at fs={fs}, hand needs {need:.3f} "
        f"— the pointing hand will fall off the bottom and stop detecting")

# The margin must scale rather than be a constant. Only compare within the
# close range where the FRAME is what limits the zone — further away the
# zone is limited by the user's reach instead, and then the leftover gap is
# large for reasons that have nothing to do with the hand margin.
clamped = [1.0 - settled(fs, anchor=(0.5, 0.35)).zone[3]
           for fs in (0.25, 0.35, 0.45)]
assert clamped == sorted(clamped), \
    f"bottom clearance must grow with apparent hand size: {clamped}"
print(f"  clearance grows with hand size while frame-limited: "
      f"{[round(c, 3) for c in clamped]}  OK")

print()
print("== zone never inverts, however cramped the bounds ==")
for fs in (0.60, 0.50, 0.45):
    for win in (None, (0.3, 0.3, 0.7, 0.7), (0.45, 0.45, 0.55, 0.55)):
        c = settled(fs, anchor=(0.5, 0.4), detect_window=win)
        x0, y0, x1, y1 = c.zone
        assert x1 > x0 and y1 > y0, f"zone inverted at fs={fs} win={win}: {c.zone}"
        sx, sy = c.absolute_to_screen(0.5, 0.4)
        assert 0.0 <= sx <= 1.0 and 0.0 <= sy <= 1.0
print("  stays a valid rectangle under impossible constraints     OK")

print()
print("== corners of the zone are the corners of the screen ==")
c = settled(0.08)
x0, y0, x1, y1 = c.zone
assert c.absolute_to_screen(x0, y0) == (0.0, 0.0)
assert c.absolute_to_screen(x1, y1) == (1.0, 1.0)
mid = c.absolute_to_screen((x0 + x1) / 2, (y0 + y1) / 2)
assert abs(mid[0] - 0.5) < 1e-6 and abs(mid[1] - 0.5) < 1e-6
print("  zone corners -> screen corners, centre -> centre         OK")

print()
print("== out-of-zone hands clamp, they don't wrap ==")
c = settled(0.08)
assert c.absolute_to_screen(-5.0, -5.0) == (0.0, 0.0)
assert c.absolute_to_screen(5.0, 5.0) == (1.0, 1.0)
print("  clamped to 0..1                                          OK")

print()
print("== untracked user falls back to the fixed zone ==")
c = settled(0.08)
for _ in range(400):
    c.set_control_zone(None, None, DT)
m = BC.CAM_MARGIN
assert all(abs(a - b) < 1e-3 for a, b in zip(c.zone, (m, m, 1 - m, 1 - m))), c.zone
assert not c._have_scale
print("  reverts to the legacy 0.15 margin box                    OK")

print()
print("== reach_radius tracks the zone (used for hand ownership) ==")
near, far = settled(0.30), settled(0.03)
print(f"  face_size 0.30 -> radius {near.reach_radius:.3f}")
print(f"  face_size 0.03 -> radius {far.reach_radius:.3f}")
assert near.reach_radius > far.reach_radius, \
    "ownership radius must tighten as the user gets smaller in frame"
assert far.reach_radius >= BC.MIN_HALF, "must never collapse to zero"
# The old fixed 0.70 radius covered most of the room at range.
assert far.reach_radius * 1.6 < 0.70, \
    "far-range ownership must be far tighter than the old fixed 0.70"
print("  tightens with distance, floored at MIN_HALF              OK")

print()
print("all control-zone assertions passed")
