"""
Offline checks of cursor-control ownership across two devices.

The bug this exists to prevent: sign in on a laptop and a desktop under one
account, sign out on the one holding the cursor, and the other one stays
connected and authenticated but silently stops working — the recogniser had
been retargeted to nobody and nothing ever re-armed it.

No camera, no radio, no Flask: the registry takes its effects as callbacks,
so the whole arbitration is testable here.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from control_registry import ControlRegistry

LAPTOP  = "AA:BB:CC:DD:EE:01"
DESKTOP = "AA:BB:CC:DD:EE:02"


class Effects:
    """Records what the registry pushed down to the camera and the radio."""
    def __init__(self):
        self.user = "unset"
        self.peer = "unset"
        self.broadcasts = 0

    def make(self):
        return ControlRegistry(
            on_active_user=self._set_user,
            on_preferred_peer=self._set_peer,
            on_broadcast=self._bump,
        )

    def _set_user(self, u):    self.user = u
    def _set_peer(self, m):    self.peer = m
    def _bump(self):           self.broadcasts += 1


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)


# ── The reported bug ────────────────────────────────────────────────────────
def test_logout_on_one_device_leaves_the_other_working():
    fx = Effects()
    reg = fx.make()

    # Same account, two machines.
    reg.add_viewer("sid-laptop",  "Andy101Le", "dev-laptop")
    reg.add_viewer("sid-desktop", "Andy101Le", "dev-desktop")
    reg.take("sid-laptop", "Andy101Le", "dev-laptop", LAPTOP, "Laptop")
    check(fx.user == "Andy101Le", "laptop should be tracked after taking control")
    check(fx.peer == LAPTOP,      "cursor should go to the laptop")

    # The laptop signs out. The desktop is untouched and must keep working.
    reg.forget_device("dev-laptop")
    check(reg.holder is None, "control should not survive its device signing out")
    check(fx.user == "Andy101Le",
          "THE BUG: desktop is still connected, so the camera must keep "
          f"following its user, got {fx.user!r}")
    check(fx.peer is None,
          "the HID pin must be released so the desktop can reconnect, "
          f"got {fx.peer!r}")
    check(not reg.has_device("dev-laptop"), "laptop viewers should be gone")
    check(reg.has_device("dev-desktop"),    "desktop viewer should remain")


def test_logout_on_the_last_device_stops_tracking():
    fx = Effects()
    reg = fx.make()
    reg.add_viewer("sid-1", "Andy101Le", "dev-laptop")
    reg.forget_device("dev-laptop")
    check(fx.user is None, "nobody left — the camera must stop following")
    check(fx.peer is None, "nobody left — the HID pin must be released")


# ── Fallback: a single-device user never presses anything ───────────────────
def test_unclaimed_falls_back_to_newest_viewer():
    fx = Effects()
    reg = fx.make()
    reg.add_viewer("sid-a", "Andy101Le", "dev-a")
    check(fx.user == "Andy101Le", "a lone viewer should be tracked with no claim")
    check(fx.peer is None, "an unclaimed cursor must not pin the radio")

    reg.add_viewer("sid-b", "Machu1287", "dev-b")
    check(fx.user == "Machu1287", "newest viewer wins while unclaimed")

    # Explicit claim beats recency, and survives a newer viewer arriving.
    reg.take("sid-a", "Andy101Le", "dev-a", LAPTOP)
    reg.add_viewer("sid-c", "Machu1287", "dev-c")
    check(fx.user == "Andy101Le", "an explicit claim must outrank a newer viewer")
    check(fx.peer == LAPTOP,      "claimed cursor stays on the claimed machine")


# ── Handoff ─────────────────────────────────────────────────────────────────
def test_taking_control_moves_the_cursor():
    fx = Effects()
    reg = fx.make()
    reg.add_viewer("sid-laptop",  "Andy101Le", "dev-laptop")
    reg.add_viewer("sid-desktop", "Andy101Le", "dev-desktop")

    reg.take("sid-laptop", "Andy101Le", "dev-laptop", LAPTOP)
    check(fx.peer == LAPTOP, "cursor should follow the claim")

    reg.take("sid-desktop", "Andy101Le", "dev-desktop", DESKTOP)
    check(fx.peer == DESKTOP, "a second claim should move the cursor")
    check(reg.holder["device_id"] == "dev-desktop", "holder should be the desktop")

    reg.release("test")
    check(reg.holder is None, "release should clear the holder")
    check(fx.peer is None,    "release should unpin the radio")
    check(fx.user == "Andy101Le", "released, but viewers remain — keep tracking")


def test_take_normalises_mac_case():
    fx = Effects()
    reg = fx.make()
    reg.add_viewer("sid-a", "Andy101Le", "dev-a")
    reg.take("sid-a", "Andy101Le", "dev-a", LAPTOP.lower())
    check(fx.peer == LAPTOP, f"MAC should be upper-cased, got {fx.peer!r}")
    check(reg.drop_mac(LAPTOP.lower()),
          "drop_mac should match regardless of case")


# ── Refresh tolerance ───────────────────────────────────────────────────────
def test_refresh_does_not_drop_control():
    """A refresh disconnects and reconnects. Yanking the cursor away in that
    window would make the dashboard unusable, so only forget_device drops it."""
    fx = Effects()
    reg = fx.make()
    reg.add_viewer("sid-old", "Andy101Le", "dev-laptop")
    reg.take("sid-old", "Andy101Le", "dev-laptop", LAPTOP)

    device_id = reg.remove_viewer("sid-old")
    check(device_id == "dev-laptop", "remove_viewer should report the device")
    check(reg.holder is not None, "control must survive a socket drop")

    # Reconnects before the grace period expires.
    reg.add_viewer("sid-new", "Andy101Le", "dev-laptop")
    check(reg.has_device("dev-laptop"), "device is back")
    check(fx.peer == LAPTOP, "cursor should still be on the laptop")


def test_unpairing_the_controlling_machine_releases_it():
    fx = Effects()
    reg = fx.make()
    reg.add_viewer("sid-a", "Andy101Le", "dev-a")
    reg.take("sid-a", "Andy101Le", "dev-a", LAPTOP)

    check(not reg.drop_mac(DESKTOP), "unpairing another machine changes nothing")
    check(fx.peer == LAPTOP, "an unrelated unpair must not move the cursor")

    check(reg.drop_mac(LAPTOP), "unpairing the holder should release control")
    check(fx.peer is None, "radio must not stay pinned to an unpaired MAC")


def test_broadcasts_only_on_real_changes():
    fx = Effects()
    reg = fx.make()
    reg.add_viewer("sid-a", "Andy101Le", "dev-a")
    before = fx.broadcasts
    reg.release("nothing held")
    check(fx.broadcasts == before, "releasing nothing should not broadcast")
    reg.drop_mac(DESKTOP)
    check(fx.broadcasts == before, "dropping an unheld MAC should not broadcast")
    reg.forget_device("dev-nonexistent")
    check(fx.broadcasts == before, "forgetting an unknown device should not broadcast")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"\n{len(tests)} passed")
