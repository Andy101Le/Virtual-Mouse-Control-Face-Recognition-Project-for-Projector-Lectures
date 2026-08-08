"""
test_zoom_sweep.py — hardware exerciser for the Arducam zoom motor.

The auto-zoom range extension drives OPT_ZOOM, but this board has form:
its tilt channel ACKs every I2C write and reads values back while never
driving the header (hence the GPIO 18 bypass). Run this FIRST to prove
the zoom channel actually moves the lens before trusting any software
built on it.

ZOOM IS NOT FOCUS. This is the single easiest thing to get wrong here,
because both motors live on the same board behind the same I2C interface
and both "do something visible" when you drive them:

    FOCUS changes SHARPNESS.   Everything stays exactly the same size and
                               in the same place; it just goes crisp or
                               soft. A face getting clearer as someone
                               walks toward the camera is focus.
    ZOOM  changes FIELD OF VIEW. The subject and the whole scene get
                               BIGGER or SMALLER. At the telephoto end
                               you see LESS of the room — objects near
                               the frame edges slide out of shot
                               entirely. Sharpness is beside the point.

Only a field-of-view change proves the zoom channel is alive. So this
script runs two stages back to back:

    Stage A — FOCUS reference. Zoom is parked; focus sweeps. Watch what
              focus alone looks like, so you have something to compare
              against. Framing must NOT change here.
    Stage B — ZOOM test. Focus is parked at a fixed mid value so
              autofocus-style sharpness changes cannot masquerade as
              zoom; zoom sweeps. If framing changes, the motor works.

If Stage B's readbacks all say OK but the framing never changes, the
zoom channel is dead like the tilt channel was — start the server with
PTZ_ZOOM=0 to disable optical auto-zoom until the hardware is sorted.

Note WHICH END of the register range is telephoto: the auto-zoom code
assumes MAX_VALUE (20000) is tele; if your lens is tightest at the LOW
end, set PTZController.ZOOM_TELE_AT_MAX = False.

Usage:
    python3 test_zoom_sweep.py              # focus reference, then 2 zoom cycles
    python3 test_zoom_sweep.py --cycles 4
    python3 test_zoom_sweep.py --skip-focus # straight to the zoom test

Run with web_server.py stopped so nothing else owns the camera or the
I2C bus — the running server's autofocus would fight the parked focus in
Stage B and reintroduce exactly the ambiguity this script exists to
remove. Preview the image in a second terminal BEFORE starting this:
    libcamera-hello -t 0
"""

import sys
import time

from Focuser import Focuser

ZOOM_LO = Focuser.opts[Focuser.OPT_ZOOM]["MIN_VALUE"]    # 3000
ZOOM_HI = Focuser.opts[Focuser.OPT_ZOOM]["MAX_VALUE"]    # 20000
FOCUS_LO = Focuser.opts[Focuser.OPT_FOCUS]["MIN_VALUE"]  # 0
FOCUS_HI = Focuser.opts[Focuser.OPT_FOCUS]["MAX_VALUE"]  # 20000

# Parked focus for Stage B. Mid-range keeps a room-scale subject roughly
# in the ballpark, so the picture stays interpretable while zoom moves —
# the point is that it stays CONSTANT, not that it is optimal.
FOCUS_PARK = 10000

DWELL_S = 2.5   # zoom travel is a slow stepper: give it time to finish
                # and give you time to see the framing change


def _sweep(f, opt, targets, label_for, dwell=DWELL_S):
    """Drive one channel to each target in turn, reporting the readback."""
    results = []
    for target in targets:
        f.set(opt, target)   # blocks until the motor settles
        time.sleep(dwell)
        readback = f.get(opt)
        ok = "OK" if readback == target else "MISMATCH"
        print(f"    commanded {target:5d} ({label_for(target)}) "
              f"-> board readback {readback:5d}  {ok}")
        results.append(readback == target)
    return all(results)


def stage_focus(f):
    """Show what focus alone looks like: sharpness moves, framing doesn't."""
    print()
    print("STAGE A — FOCUS reference (zoom parked)")
    print("  WATCH: the picture should go soft and sharp. Nothing should")
    print("  change SIZE, and nothing should enter or leave the frame.")
    print()
    orig = f.get(Focuser.OPT_FOCUS)
    _sweep(f, Focuser.OPT_FOCUS, (FOCUS_LO, FOCUS_HI, FOCUS_PARK),
           lambda v: {FOCUS_LO: "near", FOCUS_HI: "far"}.get(v, "park"))
    print(f"  focus parked at {f.get(Focuser.OPT_FOCUS)} "
          f"(was {orig}) — it will stay here for Stage B.")


def stage_zoom(f, cycles):
    """The actual test: does the zoom channel change the field of view?"""
    orig = f.get(Focuser.OPT_ZOOM)
    print()
    print("STAGE B — ZOOM test (focus parked, so sharpness cannot fool you)")
    print(f"  zoom starting at {orig} — sweeping {ZOOM_LO}<->{ZOOM_HI} x{cycles}")
    print("  WATCH THE FRAMING, NOT THE SHARPNESS: does the scene get")
    print("  bigger/smaller? Do things at the edges leave the frame?")
    print()

    label = lambda v: "wide end" if v == ZOOM_LO else "tele end"
    for cycle in range(1, cycles + 1):
        print(f"  cycle {cycle}:")
        _sweep(f, Focuser.OPT_ZOOM, (ZOOM_LO, ZOOM_HI), label)

    f.set(Focuser.OPT_ZOOM, orig)
    print(f"  done — zoom restored to {f.get(Focuser.OPT_ZOOM)}.")


def verdict():
    print()
    print("=" * 68)
    print("WHAT DID YOU SEE IN STAGE B?")
    print()
    print("  The scene got BIGGER and SMALLER (things left/entered the frame)")
    print("      -> the zoom motor works. Note which end looked TELEPHOTO:")
    print(f"         {ZOOM_HI} (high) => keep ZOOM_TELE_AT_MAX = True")
    print(f"         {ZOOM_LO} (low)  => set  ZOOM_TELE_AT_MAX = False")
    print()
    print("  Only SHARPNESS changed, framing identical")
    print("      -> that is the focus motor, not zoom. The zoom channel is")
    print("         not driving the zoom lens. Run the server with PTZ_ZOOM=0.")
    print()
    print("  NOTHING changed at all")
    print("      -> dead channel, same signature as the tilt header. The")
    print("         readbacks above will still say OK; that proves only that")
    print("         the register accepted the write. Run with PTZ_ZOOM=0.")
    print("=" * 68)


def main():
    cycles = 2
    if "--cycles" in sys.argv:
        i = sys.argv.index("--cycles")
        cycles = int(sys.argv[i + 1]) if len(sys.argv) > i + 1 else 2

    f = Focuser(1)

    if "--skip-focus" not in sys.argv:
        stage_focus(f)
    else:
        # Stage B's whole premise is that sharpness is held constant, so
        # park focus even when the reference sweep is skipped.
        f.set(Focuser.OPT_FOCUS, FOCUS_PARK)
        print(f"focus parked at {f.get(Focuser.OPT_FOCUS)} (reference sweep skipped)")

    stage_zoom(f, cycles)
    verdict()


if __name__ == "__main__":
    main()
