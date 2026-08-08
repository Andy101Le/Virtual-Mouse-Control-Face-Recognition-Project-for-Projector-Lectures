"""
Peace-sign middle click, and its interaction with the master switch.

The peace sign has two jobs: normally it sends a middle click, but the
open-palm master switch also borrows it as the reset step of its toggle
sequence. These tests pin the boundary between the two — a reset peace sign
must never click, and a normal one always must.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bt_cursor_controller import BTCursorController as BC
from control_toggle import ControlToggle, is_peace_sign, ARMED, AWAIT_PEACE
from _hands import hand, ALL, PEACE


class FakeHID:
    """Records clicks instead of sending them."""
    connected = True

    def __init__(self):
        self.clicks = []

    def click(self, button="left", hold_s=0.02):
        self.clicks.append(button)
        return True


def new_cursor():
    return BC(FakeHID())


print("== a sustained peace sign sends exactly one middle click ==")
c = new_cursor()
t = 0.0
fired = 0
for _ in range(60):                       # 3s of held peace sign
    t += 0.05
    fired += bool(c.peace_click(0, True, t))
print(f"  3s of peace sign -> {fired} click(s), hid got {c.hid.clicks}")
assert fired == 1, "must fire once, not stream"
assert c.hid.clicks == ["middle"], "must be a MIDDLE click"

print()
print("== releasing and re-showing fires again (after the cooldown) ==")
for _ in range(10):
    t += 0.05
    c.peace_click(0, False, t)            # hand lowered -> latch released
t += BC.CLICK_COOLDOWN
fired = 0
for _ in range(20):
    t += 0.05
    fired += bool(c.peace_click(0, True, t))
assert fired == 1, "a second, separate peace sign should click again"
print(f"  second peace sign -> 1 click (total {len(c.hid.clicks)})")

print()
print("== a brief flicker of the pose does NOT click ==")
c = new_cursor()
t = 0.0
for _ in range(2):                        # ~0.1s, under MIDDLE_HOLD_SECONDS
    t += 0.05
    assert not c.peace_click(0, True, t)
    c.peace_click(0, False, t)
assert c.hid.clicks == [], "a passing pose must not fire"
print(f"  0.1s flickers -> {len(c.hid.clicks)} clicks               OK")

print()
print("== THE CONFLICT: the switch's reset peace sign must not click ==")
# Full sequence, starting with controls OFF so the toggle turns them ON —
# the dangerous direction, because afterwards controls are enabled and the
# reset peace sign would otherwise be a live middle click.
tog = ControlToggle(enabled=False)
c   = new_cursor()
t   = 0.0
palm = [hand(ALL)]


def frame(pts_list, tog, cur, now):
    """One vision-loop frame, mirroring gesture_session's ordering."""
    peace_for_click = (tog.enabled
                       and not tog.owns_peace_sign
                       and any(is_peace_sign(p) for p in pts_list))
    clicked = cur.peace_click(0, peace_for_click, now)
    fired = tog.update(pts_list, now)
    return clicked, fired


for _ in range(30):                       # open palm 1.5s -> toggle ON
    t += 0.05
    clicked, fired = frame(palm, tog, c, t)
    assert not clicked
assert tog.enabled, "controls should now be ON"
assert tog.state == AWAIT_PEACE
print("  open palm held  -> controls ON, switch awaiting peace")

for _ in range(30):                       # the RESET peace sign
    t += 0.05
    clicked, _ = frame([hand(PEACE)], tog, c, t)
    assert not clicked, "reset peace sign must NOT middle click"
print(f"  reset peace sign-> {len(c.hid.clicks)} clicks (must be 0)      OK")

for _ in range(20):                       # lower the hand -> re-armed
    t += 0.05
    frame([], tog, c, t)
assert tog.state == ARMED
assert c.hid.clicks == [], "still no click after the full reset"
print("  hand lowered    -> re-armed, still 0 clicks           OK")

print()
print("== once re-armed, a peace sign clicks normally again ==")
t += BC.CLICK_COOLDOWN
clicked_any = False
for _ in range(20):
    t += 0.05
    clicked, _ = frame([hand(PEACE)], tog, c, t)
    clicked_any = clicked_any or clicked
assert clicked_any, "after re-arming the peace sign must click again"
assert c.hid.clicks == ["middle"]
print(f"  peace sign      -> {c.hid.clicks}                     OK")

print()
print("== no clicks at all while controls are disabled ==")
tog2 = ControlToggle(enabled=False)
c2   = new_cursor()
t2   = 0.0
for _ in range(40):
    t2 += 0.05
    clicked, _ = frame([hand(PEACE)], tog2, c2, t2)
    assert not clicked
assert c2.hid.clicks == []
print("  peace sign while paused -> 0 clicks                   OK")

print()
print("== a disconnected host never latches a click away ==")
c3 = new_cursor()
c3.hid.connected = False
t3 = 0.0
for _ in range(40):
    t3 += 0.05
    c3.peace_click(0, True, t3)
assert c3.hid.clicks == []
print("  not connected -> nothing sent                         OK")

print()
print("all middle-click assertions passed")
