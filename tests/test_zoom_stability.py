"""
Offline checks of optical auto-zoom stability at range.

The reported symptom: as the user walks away, recognition starts flickering
green/red, and the camera begins zooming in and out and refocusing
continuously — which makes the image worse, which makes recognition worse.

That is a feedback loop with two amplifiers, and both are fixed here.

  1. `_auto_zoom_track`'s dwell counter was direction-agnostic. It exists to
     "ride out single noisy size readings", but because it incremented for
     ANY out-of-band reading regardless of sign, three ticks of "face too
     small" followed by ONE bogus oversized reading fired a zoom OUT. A
     single noisy sample got to spend the credit the opposite direction had
     built up. At range that is the common case, because a barely-resolved
     face reports an unreliable bounding box.

  2. Every settled zoom triggered a full autofocus re-hunt, including an
     in-then-out pair that left the lens exactly where it started. Hunting
     blurs the image while it runs, so this fed straight back into the
     recognition failure that started the cycle.

Runs with enabled=False, so no I2C and no motors.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ptz_controller import PTZController as P
import gesture_session as gs

TELE = 1 if P.ZOOM_TELE_AT_MAX else -1     # sign of a "zoom in" delta
START = 10000


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)


def ptz():
    p = P(enabled=False, verbose=False)
    p._targets.zoom = START
    return p


def feed(p, sizes, t0=0.0, step=1.0):
    """Push face-size readings in, one per 'snapshot', and report the net
    zoom movement. `step` clears ZOOM_STEP_INTERVAL comfortably."""
    before = p._targets.zoom
    for i, fs in enumerate(sizes):
        p._auto_zoom_track(fs, now=t0 + i * step)
    return p._targets.zoom - before


# ── 1. The dwell counter ────────────────────────────────────────────────────
def test_one_bogus_reading_cannot_reverse_the_zoom():
    """THE BUG. Three ticks of 'face too small' then a single oversized
    misdetection used to fire a zoom OUT."""
    p = ptz()
    moved = feed(p, [0.02, 0.02, 0.02, 0.55])
    check(moved == 0,
          f"a lone noisy reading must not move the zoom, moved {moved:+d}")


def test_a_consistent_direction_still_zooms():
    """The fix must not simply freeze the zoom — a face that really is too
    small has to bring the lens in."""
    p = ptz()
    moved = feed(p, [0.02] * P.ZOOM_DWELL_TICKS)
    check(moved == P.ZOOM_STEP * TELE,
          f"a consistently small face should zoom IN one step, got {moved:+d}")

    p = ptz()
    moved = feed(p, [0.60] * P.ZOOM_DWELL_TICKS)
    check(moved == -P.ZOOM_STEP * TELE,
          f"a consistently large face should zoom OUT one step, got {moved:+d}")


def test_symmetric_noise_around_a_good_size_holds_still():
    """A correctly-framed face measured noisily must leave the lens alone.

    Note what this does NOT claim: the decision is made on the SMOOTHED
    size, so noise whose mean sits outside the band will still move the
    lens — correctly, because by then the estimate genuinely is out of band.
    The dwell defends against single outliers, not against a biased
    estimator. Inputs here are symmetric about mid-band for that reason.
    """
    lo, hi = P.FACE_TARGET_LO, P.FACE_TARGET_HI
    mid = (lo + hi) / 2
    swing = (hi - lo)                       # excursions well past both edges
    p = ptz()
    moved = feed(p, [mid - swing, mid + swing] * 12)
    check(moved == 0,
          f"noise centred in the band must not drive the motor, {moved:+d}")


def test_direction_flip_restarts_the_count():
    p = ptz()
    feed(p, [0.02, 0.02, 0.02])                   # 3 ticks toward tele
    check(p._zoom_dwell == 3 and p._zoom_dwell_dir == 1, "fixture check")
    feed(p, [0.60], t0=100.0)                     # one tick the other way
    check(p._zoom_dwell == 1 and p._zoom_dwell_dir == -1,
          "a flip should restart the counter, not inherit its credit")
    check(p._targets.zoom == START, "and nothing should have moved yet")


def test_in_band_clears_the_dwell():
    mid = (P.FACE_TARGET_LO + P.FACE_TARGET_HI) / 2
    p = ptz()
    # Seeded rather than fed in: coming from far below, the EMA spends
    # several samples climbing INTO the band, and the lens moving during
    # that climb is correct — the smoothed size really was still too small.
    # What is being tested here is the in-band state itself.
    p._smoothed_face_size = mid
    p._zoom_dwell, p._zoom_dwell_dir = 3, 1

    feed(p, [mid] * 6)
    check(p._zoom_dwell == 0 and p._zoom_dwell_dir == 0,
          "a face inside the target band should clear the dwell entirely")
    check(p._targets.zoom == START, "and should never move the lens")


def test_disabled_zoom_and_missing_size_are_no_ops():
    p = ptz()
    p.zoom_enabled = False
    check(feed(p, [0.02] * 10) == 0, "disabled zoom must not move")

    p = ptz()
    check(feed(p, [None] * 10) == 0, "a missing face size must not move")


# ── 2. Autofocus re-hunts ───────────────────────────────────────────────────
class Settled:
    def __init__(self):
        self.calls = []

    def __call__(self, units):
        self.calls.append(units)


def test_in_then_out_does_not_ask_for_a_refocus():
    """The lens ends where it began, so the focus plane never moved."""
    p = ptz()
    p.on_zoom_settled = spy = Settled()
    # Times start at 1.0: _zoom_last_step_t is 0.0 at construction, so a
    # write at now=0.0 is (correctly) swallowed by ZOOM_STEP_INTERVAL.
    p._zoom_write_delta(+P.ZOOM_STEP, now=1.0)
    p._zoom_write_delta(-P.ZOOM_STEP, now=2.0)
    check(p._targets.zoom == START, "fixture: the pair should cancel")
    p._auto_zoom_check_settled(now=2.0 + P.ZOOM_SETTLE_S + 0.1)
    check(spy.calls == [],
          f"a cancelled zoom must not trigger a hunt, got {spy.calls}")


def test_a_real_ramp_reports_its_net_travel():
    p = ptz()
    p.on_zoom_settled = spy = Settled()
    for i in range(3):
        p._zoom_write_delta(+P.ZOOM_STEP, now=(i + 1) * 1.0)
    p._auto_zoom_check_settled(now=3.0 + P.ZOOM_SETTLE_S)
    check(spy.calls == [3 * P.ZOOM_STEP],
          f"should report net travel once, got {spy.calls}")


def test_refocus_threshold_ignores_a_single_nudge():
    """A settled zoom only earns a hunt if the lens went somewhere."""
    class FakeAF:
        auto_enabled = True
        def __init__(self):  self.hunts = 0
        def trigger_refocus(self): self.hunts += 1

    s = object.__new__(gs.GestureSession)      # no camera, no models
    s.autofocus = FakeAF()

    s._on_zoom_settled(P.ZOOM_STEP)            # one nudge
    check(s.autofocus.hunts == 0,
          "a single zoom step should not be worth an autofocus hunt")

    s._on_zoom_settled(gs.ZOOM_REFOCUS_MIN_UNITS)
    check(s.autofocus.hunts == 1, "a real ramp should still refocus")


def test_manual_focus_is_still_never_overridden():
    """The pre-existing guard: an automatic zoom event must not rack a lens
    the user parked by hand, while the UI still says MANUAL."""
    class ManualAF:
        auto_enabled = False
        def __init__(self):  self.hunts = 0
        def trigger_refocus(self): self.hunts += 1

    s = object.__new__(gs.GestureSession)
    s.autofocus = ManualAF()
    s._on_zoom_settled(100000)
    check(s.autofocus.hunts == 0, "manual focus must stay where it was put")

    s.autofocus = None
    s._on_zoom_settled(100000)                 # must not raise


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"\n{len(tests)} passed")
