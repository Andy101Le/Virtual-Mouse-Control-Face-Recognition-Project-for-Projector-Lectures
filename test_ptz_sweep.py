"""
test_ptz_sweep.py — hardware exerciser for the PTZ servos.

Sweeps one axis slowly through a large, visually unmissable range so you
can tell whether the servo physically responds to board commands. Used to
isolate wiring/servo faults from tracking-software issues (the tracking
code path is identical for both axes, so if one axis sweeps here and the
other doesn't, the difference is hardware).

Usage:
    python3 test_ptz_sweep.py tilt            # sweep tilt via the PTZ board
    python3 test_ptz_sweep.py pan             # sweep pan  via the PTZ board
    python3 test_ptz_sweep.py tilt --gpio 18  # sweep tilt via a Pi GPIO pin
                                              # (gpio_tilt.py bypass wiring)

Best run with web_server.py stopped, so the auto-tracker can't fight the
sweep — but a full-range sweep is visible even if it's running.
"""

import sys
import time

from Focuser import Focuser

AXES = {
    "pan":  Focuser.OPT_MOTOR_X,
    "tilt": Focuser.OPT_MOTOR_Y,
}

SWEEP_LO, SWEEP_HI = 45, 140
CYCLES = 3
DWELL_S = 1.8


def main():
    args = sys.argv[1:]
    gpio_pin = None
    if "--gpio" in args:
        i = args.index("--gpio")
        gpio_pin = int(args[i + 1]) if len(args) > i + 1 else 18
        del args[i:i + 2]

    axis = args[0].lower() if args else "tilt"
    if axis not in AXES:
        print(f"Unknown axis {axis!r} — use 'pan' or 'tilt'.")
        sys.exit(1)
    opt = AXES[axis]

    if gpio_pin is not None:
        from gpio_tilt import GpioTiltFocuser
        print(f"driving tilt from GPIO{gpio_pin} (board tilt channel bypassed)")
        f = GpioTiltFocuser(1, tilt_pin=gpio_pin)
    else:
        f = Focuser(1)
    orig = f.get(opt)
    print(f"{axis} starting at {orig}° — sweeping "
          f"{SWEEP_LO}°↔{SWEEP_HI}° x{CYCLES}, WATCH THE CAMERA")

    for cycle in range(1, CYCLES + 1):
        for target in (SWEEP_LO, SWEEP_HI):
            f.set(opt, target)
            time.sleep(DWELL_S)
            readback = f.get(opt)
            ok = "OK" if readback == target else "MISMATCH"
            print(f"  cycle {cycle}: commanded {target:3d}° "
                  f"-> board readback {readback:3d}°  {ok}")

    f.set(opt, orig)
    print(f"done — {axis} restored to {f.get(opt)}°. If the board readbacks "
          f"were OK but the camera never moved, the fault is the servo, its "
          f"lead, or the header connection — not software.")


if __name__ == "__main__":
    main()
