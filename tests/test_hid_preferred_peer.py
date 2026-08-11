"""
Offline checks of HID peer pinning.

The Pi has one HID link. Before this, who held it was a race: the accept
loop took whoever dialed in, and the reconnect loop dialed every paired
host until one answered — so with a laptop and a desktop both paired, the
cursor landed wherever the race finished, and logging in had no say in it.

Pinning is what makes a control handoff stick. Without it a sleeping PC
takes the cursor back the moment it wakes, or the reconnect loop quietly
undoes the handoff a few seconds after it happened.

No radio and no sockets: only the decision logic is exercised.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bluetooth_hid import BluetoothHIDDevice

LAPTOP  = "AA:BB:CC:DD:EE:01"
DESKTOP = "AA:BB:CC:DD:EE:02"


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)


def make(paired=(LAPTOP, DESKTOP)):
    hid = BluetoothHIDDevice()
    hid._reconnect_targets = lambda: list(paired)
    return hid


def test_unpinned_allows_and_dials_everything():
    hid = make()
    check(hid.preferred_peer is None, "should start unpinned")
    check(hid._peer_allowed(LAPTOP),  "unpinned must accept any host")
    check(hid._peer_allowed(DESKTOP), "unpinned must accept any host")
    check(hid._dial_targets() == [LAPTOP, DESKTOP],
          "unpinned must dial the whole paired list")


def test_pinning_refuses_every_other_host():
    hid = make()
    hid.set_preferred_peer(LAPTOP)
    check(hid.preferred_peer == LAPTOP, "pin should be recorded")
    check(hid._peer_allowed(LAPTOP), "the holder must still be accepted")
    check(not hid._peer_allowed(DESKTOP),
          "THE BUG: a woken second PC must not take the cursor back")
    check(hid._dial_targets() == [LAPTOP],
          "pinned, the reconnect loop must not dial the other machine")


def test_pin_is_case_insensitive():
    hid = make()
    hid.set_preferred_peer(LAPTOP.lower())
    check(hid.preferred_peer == LAPTOP, "pin should be stored upper-case")
    check(hid._peer_allowed(LAPTOP.lower()), "lower-case peer should match")
    check(hid._peer_allowed(LAPTOP),         "upper-case peer should match")


def test_switching_pin_drops_the_stale_link():
    hid = make()
    # Pretend the laptop holds a live link.
    hid._peer_addr = LAPTOP
    hid._ctrl_conn = _FakeSock()
    hid._intr_conn = _FakeSock()
    check(hid.connected, "fixture should look connected")

    hid.set_preferred_peer(DESKTOP)
    check(not hid.connected,
          "handing control to another machine must drop the old link")
    check(hid.peer_address is None, "stale peer address must be cleared")
    check(hid.preferred_peer == DESKTOP, "pin should have moved")


def test_repinning_the_current_holder_keeps_the_link():
    hid = make()
    hid.set_preferred_peer(LAPTOP)
    hid._peer_addr = LAPTOP
    hid._ctrl_conn = _FakeSock()
    hid._intr_conn = _FakeSock()

    hid.set_preferred_peer(LAPTOP)
    check(hid.connected, "re-pinning the same machine must not bounce the link")


def test_releasing_the_pin_keeps_the_link_but_reopens_the_field():
    hid = make()
    hid.set_preferred_peer(LAPTOP)
    hid._peer_addr = LAPTOP
    hid._ctrl_conn = _FakeSock()
    hid._intr_conn = _FakeSock()

    hid.set_preferred_peer(None)
    check(hid.connected,
          "releasing control must not hang up on the machine in use")
    check(hid.preferred_peer is None, "pin should be cleared")
    check(hid._peer_allowed(DESKTOP), "unpinned again — anyone may connect")
    check(hid._dial_targets() == [LAPTOP, DESKTOP],
          "unpinned again — dial the whole paired list")


def test_dial_targets_survives_a_broken_lookup():
    hid = BluetoothHIDDevice()
    def boom():
        raise RuntimeError("bluez went away")
    hid._reconnect_targets = boom
    check(hid._dial_targets() == [],
          "a failing target lookup must not kill the reconnect loop")

    hid._reconnect_targets = None
    check(hid._dial_targets() == [], "no lookup configured means nothing to dial")


class _FakeSock:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"\n{len(tests)} passed")
