#!/usr/bin/env python3
"""
test_hid.py — isolate the Bluetooth HID transport from the camera/gesture/web
stack, so we can answer one question: does the Pi actually move the paired
host's cursor?

Run as root, with web_server.py NOT running (they can't both own PSM 17/19):

    sudo ../.venv/bin/python test_hid.py

Then, on the already-paired laptop, toggle Bluetooth off/on (or just wait) so
it reconnects to the Pi's HID channels. Watch the two things this prints:

  1. "HOST CONNECTED" + connected=True   -> the interrupt channel opened.
     If you never see this, the host is not opening PSM 19; the cursor code
     is fine and the problem is the connection/host side.

  2. A visible cursor sweep once connected. If connected=True but the cursor
     does not move, the transport works and the suspect is the absolute
     descriptor vs. the host OS (see bluetooth_hid.py docstring: macOS in
     particular ignores absolute Mouse pointers).
"""

import sys
import time

from bluetooth_hid import BluetoothHIDDevice


def main():
    hid = BluetoothHIDDevice()

    print("Registering HID profile with BlueZ ...")
    hid.register_profile()

    print("Binding L2CAP PSM 17/19 and listening ...")
    hid.start()

    print("\nWaiting for the paired host to open the HID channels.")
    print("On the laptop: toggle Bluetooth off/on to force a reconnect.\n")

    waited = 0.0
    while not hid.connected:
        time.sleep(0.5)
        waited += 0.5
        # heartbeat so it's obvious we're alive and still waiting
        print(f"  ... waiting  connected={hid.connected}  ({waited:.0f}s)",
              end="\r", flush=True)
        if waited > 120:
            print("\n\nTimed out after 120s with no interrupt-channel connection.")
            print("=> The host is not opening PSM 19. This is a CONNECTION problem,")
            print("   not a cursor-report problem. Check: device still paired+trusted")
            print("   (bluetoothctl info <mac>), bluetoothd running with")
            print("   --noplugin=input, and that no other process holds the PSMs.")
            hid.stop()
            return 1

    print(f"\n\nHOST CONNECTED: {hid.peer_address}   connected={hid.connected}")
    print("Sweeping the cursor. Watch the laptop screen.\n")

    # 1) Big diagonal sweep, corner to corner, twice — impossible to miss.
    for _ in range(2):
        for i in range(0, 101, 4):
            f = i / 100.0
            ok = hid.move_absolute(f, f)
            if not ok:
                print("  move_absolute() returned False — host dropped. Stopping.")
                hid.stop()
                return 1
            time.sleep(0.02)
        for i in range(100, -1, -4):
            f = i / 100.0
            hid.move_absolute(f, 1.0 - f)
            time.sleep(0.02)

    # 2) Park in the centre and do a left click.
    hid.move_absolute(0.5, 0.5)
    time.sleep(0.3)
    print("Left click at centre ...")
    hid.click("left")

    print("\nDone. If the cursor moved, the HID transport is healthy and the")
    print("problem is upstream (gesture pipeline never calling handle_action,")
    print("or hid.connected being False inside the running server).")
    hid.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())