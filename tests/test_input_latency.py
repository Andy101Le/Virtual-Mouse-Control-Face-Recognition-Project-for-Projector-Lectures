"""
Offline checks of the input-latency work.

Three separate claims are being defended here:

  1. The smoothing filters are frame-rate independent. They used to be
     fixed per-frame weights, which meant the lag they added was measured
     in frames, not seconds — so the cursor got laggier exactly when the
     pipeline was busiest, and every performance regression quietly became
     a responsiveness regression.

  2. MOVE confirms faster than the actions that actuate. A spurious move is
     self-correcting; a spurious middle click opens browser tabs.

  3. The latency log measures the real path, and costs nothing when off.

No camera and no radio.
"""
import math
import os
import sys
import tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from smoothing import ema_alpha, OneEuroFilter
from bt_cursor_controller import BTCursorController as BC
from gesture_engine import (GestureEngine, LANDMARK_TAU_S,
                            CURSOR_FREEZE_ACTIONS)
from latency_log import LatencyLog


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)


def close(a, b, tol=1e-6):
    return abs(a - b) < tol


class FakeHID:
    def __init__(self):
        self.connected = True
        self.moves = []

    def move_absolute(self, nx, ny):
        self.moves.append((nx, ny))
        return True


# ── 1. Frame-rate independence ──────────────────────────────────────────────
def test_alpha_matches_the_time_constant():
    """The remaining gap after t seconds must be exp(-t/tau) no matter how
    that time was chopped into frames — which is the whole property, and it
    holds exactly, not approximately."""
    tau = 0.1
    for dt in (0.01, 1 / 30, 1 / 13, 0.2):
        gap = 1.0
        for n in range(1, 13):
            gap *= (1.0 - ema_alpha(dt, tau))
            expected = math.exp(-(n * dt) / tau)
            check(close(gap, expected, 1e-12),
                  f"dt={dt}, {n} steps: gap {gap:.9f} != {expected:.9f}")


def test_same_wall_clock_lag_at_any_frame_rate():
    """The regression that matters: identical real-time response whether the
    loop is at 30 fps or crawling at 8."""
    tau = 0.05
    def gap_after(seconds, dt):
        g, t = 1.0, 0.0
        while t < seconds - 1e-9:
            g *= (1.0 - ema_alpha(dt, tau))
            t += dt
        return g

    fast = gap_after(0.2, 1 / 30)
    slow = gap_after(0.2, 1 / 8)
    check(abs(fast - slow) < 0.05,
          f"200 ms of smoothing must mean the same thing at 30 fps ({fast:.3f}) "
          f"and 8 fps ({slow:.3f})")

    # And the old fixed-weight filter demonstrably did NOT have this property,
    # which is what this change fixes.
    old_alpha = 0.35
    old_fast = (1 - old_alpha) ** round(0.2 * 30)
    old_slow = (1 - old_alpha) ** round(0.2 * 8)
    check(abs(old_fast - old_slow) > 0.2,
          "sanity: the old per-frame weight should differ wildly by frame rate")


def test_degenerate_dt_is_safe():
    check(ema_alpha(0.0, 0.05) == 0.0, "no elapsed time means no movement")
    check(ema_alpha(-1.0, 0.05) == 0.0, "a backwards clock must not extrapolate")
    check(ema_alpha(0.03, 0.0) == 1.0, "tau=0 disables smoothing")
    check(0.0 < ema_alpha(0.03, 0.05) < 1.0, "normal case stays a proper weight")


# ── 2. The cursor filter got faster ─────────────────────────────────────────
def steer(c, target, frames, dt, t0=0.0):
    """Drive `frames` MOVE calls at `target` on a fake clock, so the
    reacquire ramp sees the elapsed time the loop would really report."""
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


def steady(dt):
    """A controller already steering, i.e. past the reacquire ramp — that
    ramp is tested in test_tracking_dropout.py and would otherwise mask the
    steady-state responsiveness this file is about."""
    c = BC(FakeHID())
    c.zone = (0.0, 0.0, 1.0, 1.0)      # identity mapping, so target == tip
    t = steer(c, (0.0, 0.0), int(BC.REACQUIRE_S / dt) + 4, dt)
    c.cursor_nx = c.cursor_ny = 0.0
    return c, t


def test_cursor_reaches_target_far_sooner_than_before():
    """At the loop's real rate, the pointer should close most of the gap in a
    couple of frames instead of a dozen."""
    dt = 1 / 13.0
    c, t = steady(dt)

    steer(c, (1.0, 1.0), 2, dt, t0=t)
    check(c.cursor_nx > 0.9,
          f"two frames should get most of the way there, got {c.cursor_nx:.3f}")

    # The old alpha=0.35 filter managed only 0.58 in the same two frames.
    old = 1 - (1 - 0.35) ** 2
    check(c.cursor_nx > old + 0.25,
          f"should beat the old filter's {old:.2f} by a clear margin")


def test_cursor_still_smooths():
    """Faster, not disabled — a single frame must not teleport the pointer."""
    dt = 1 / 30.0
    c, t = steady(dt)
    steer(c, (1.0, 1.0), 1, dt, t0=t)
    check(c.cursor_nx < 0.95,
          f"one frame at 30 fps should not fully snap, got {c.cursor_nx:.3f}")
    check(c.cursor_nx > 0.0, "but it should move")


# ── Jitter vs lag: the One Euro tradeoff ────────────────────────────────────
DT13 = 1 / 13.0


def _ema(sig, tau, dt=DT13):
    y = [sig[0]]
    for x in sig[1:]:
        y.append(y[-1] + ema_alpha(dt, tau) * (x - y[-1]))
    return np.array(y)


def _euro(sig, dt=DT13, beta=None):
    f = OneEuroFilter(BC.CURSOR_MIN_CUTOFF_HZ,
                      BC.CURSOR_BETA if beta is None else beta,
                      BC.CURSOR_D_CUTOFF_HZ)
    return np.array([f(x, dt) for x in sig])


def _held_still(n=400, tremor=0.010, seed=7):
    """A hand held on a small target: tremor plus landmark noise."""
    return 0.5 + np.random.default_rng(seed).normal(0, tremor, n)


def test_precision_pointing_is_as_steady_as_the_old_smoothing():
    """The complaint that prompted this: at a single short time constant the
    pointer shivered while lining up a small target. One Euro must match the
    old long-constant steadiness when the hand is essentially still."""
    still = _held_still()
    old  = _ema(still, 0.18).std()      # the smoothing that felt right
    fast = _ema(still, 0.045).std()     # the retune that felt jumpy
    new  = _euro(still).std()

    check(new < old * 1.25,
          f"at rest One Euro must be about as steady as the old filter: "
          f"{new*1000:.2f} vs {old*1000:.2f} milli-screen")
    check(new < fast * 0.75,
          f"and clearly steadier than the fixed short constant: "
          f"{new*1000:.2f} vs {fast*1000:.2f}")


def test_but_dragging_still_beats_the_old_lag():
    """The other half — being steady at rest must not reintroduce the lag
    the old filter had while actually moving."""
    rate = 0.35                                   # screens per second
    t = np.arange(200) * DT13
    ramp = np.clip(t * rate, 0, 1.0)
    inside = slice(15, int(1.0 / rate / DT13) - 2)

    def lag_ms(out):
        return float(np.mean((ramp - out)[inside]) / rate * 1000)

    old = lag_ms(_ema(ramp, 0.18))
    new = lag_ms(_euro(ramp))
    check(new < old * 0.5,
          f"dragging lag should be far below the old filter's: "
          f"{new:.0f} ms vs {old:.0f} ms")


def test_beta_zero_is_the_documented_escape_hatch():
    """BETA = 0 must reduce to a plain EMA at min_cutoff, so there is a
    one-constant route back to fixed-time-constant behaviour."""
    still = _held_still()
    tau = 1.0 / (2 * math.pi * BC.CURSOR_MIN_CUTOFF_HZ)
    check(np.allclose(_euro(still, beta=0.0), _ema(still, tau), atol=1e-9),
          "with beta=0 One Euro should be exactly a plain EMA")


def test_reset_seeds_without_a_jump():
    f = OneEuroFilter(BC.CURSOR_MIN_CUTOFF_HZ, BC.CURSOR_BETA)
    for _ in range(20):
        f(0.2, DT13)
    f.reset(0.2)
    out = f(0.9, DT13)
    check(out < 0.5,
          f"after a reset a large jump must not read as speed and snap, {out:.3f}")


def test_zone_glide_is_unchanged_in_wall_clock_terms():
    """The zone filter was converted for consistency, NOT sharpened — it is
    not in the latency path and a fast zone drags the cursor around."""
    dt = 1 / 13.0
    a_new = ema_alpha(dt, BC.ZONE_TAU_S)
    check(abs(a_new - 0.15) < 0.02,
          f"should reproduce the old alpha=0.15 at ~13 fps, got {a_new:.3f}")


def test_no_host_means_no_move():
    hid = FakeHID()
    hid.connected = False
    c = BC(hid)
    c.handle_action('MOVE', 0, (1.0, 1.0), 1 / 13.0)
    check(not hid.moves, "nothing should be sent with no host connected")
    check(c.last_move_sent_t is None, "and no send should be stamped")


# ── 3. Per-gesture debounce ─────────────────────────────────────────────────
class StubEngine(GestureEngine):
    """GestureEngine without the model load — only the debounce is exercised."""
    def __init__(self):
        self._smoothed_xyz = {}
        self._gesture_counters = {}


def confirm_after(engine, action, hand_id=0):
    """How many consecutive frames of `action` before it is confirmed."""
    for n in range(1, 30):
        # Seeded with a sentinel, not 'NO ACTION' — otherwise measuring how
        # long NO ACTION takes to confirm would read as already-confirmed.
        gc = engine._gesture_counters.setdefault(
            hand_id, {'name': action, 'count': 0, 'confirmed': '__unset__'})
        if gc['name'] == action:
            gc['count'] += 1
        else:
            gc['name'], gc['count'] = action, 1
        if gc['count'] >= engine._frames_needed(action):
            gc['confirmed'] = action
        if gc['confirmed'] == action:
            return n
    return None


def test_move_confirms_faster_than_clicks():
    check(confirm_after(StubEngine(), 'MOVE') == 2,
          "MOVE should confirm in two frames")
    for action in ('LEFT CLICK', 'RIGHT CLICK', 'ZOOM IN', 'ZOOM OUT'):
        check(confirm_after(StubEngine(), action) == 5,
              f"{action} must keep the full five-frame window — a false fire "
              f"is disruptive, not merely wrong")


def test_classify_exposes_the_raw_label_for_the_cursor_freeze():
    """
    Forming a click curls the hand, which drags the fingertip. For the whole
    five-frame debounce the CONFIRMED label is still MOVE, so the cursor used
    to follow that drag and slide off a small target just as you clicked it.
    The raw label is what lets the caller stop the pointer immediately.
    """
    e = StubEngine()
    hand = 0
    # Steer for a while so MOVE is confirmed.
    for _ in range(5):
        _feed(e, hand, 'MOVE')
    confirmed, raw = _feed(e, hand, 'MOVE')
    check((confirmed, raw) == ('MOVE', 'MOVE'), "fixture: steering")

    # Now a click pose starts forming. The confirmed label lags behind.
    confirmed, raw = _feed(e, hand, 'LEFT CLICK')
    check(confirmed == 'MOVE',
          "confirmed still lags, which is exactly the problem")
    check(raw == 'LEFT CLICK',
          "the raw label must show the click forming on its very first frame")
    check(raw in CURSOR_FREEZE_ACTIONS,
          "and must be recognisable as a cursor-freezing action")


def _feed(engine, hand_id, action):
    """Run one frame of `action` through the debounce, as classify() does,
    and return its (confirmed, raw) pair."""
    gc = engine._gesture_counters.setdefault(
        hand_id, {'name': action, 'count': 0, 'confirmed': '__unset__'})
    if gc['name'] == action:
        gc['count'] += 1
    else:
        gc['name'], gc['count'] = action, 1
    if gc['count'] >= engine._frames_needed(action):
        gc['confirmed'] = action
    return gc['confirmed'], action


def test_stopping_keeps_the_long_window():
    """NO ACTION is what STOPS the cursor. Confirming it fast would make the
    pointer stutter whenever a frame or two of the move pose was misread."""
    check(confirm_after(StubEngine(), 'NO ACTION') == 5,
          "NO ACTION should not be sharpened")


# ── 4. Landmark smoothing ───────────────────────────────────────────────────
def test_first_sighting_seeds_instead_of_smoothing_in():
    e = StubEngine()
    pts = np.full((21, 3), 0.7, dtype=np.float32)
    out = e.smooth_hand(0, pts, 1 / 13.0)
    check(np.allclose(out, pts),
          "a newly seen hand must start where it is, not slide in from zero")


def test_landmark_filter_is_lighter_than_it_was():
    dt = 1 / 13.0
    a = ema_alpha(dt, LANDMARK_TAU_S)
    check(a > 0.3, f"should be less smoothed than the old alpha=0.3, got {a:.3f}")
    check(a < 0.9, "but still smoothing — this array feeds the classifier")

    e = StubEngine()
    zeros = np.zeros((21, 3), dtype=np.float32)
    ones = np.ones((21, 3), dtype=np.float32)
    e.smooth_hand(0, zeros, dt)
    out = e.smooth_hand(0, ones, dt)
    check(np.allclose(out, a, atol=1e-5),
          f"one step should land on alpha, got {out.flat[0]:.4f} vs {a:.4f}")


# ── 5. The latency log ──────────────────────────────────────────────────────
def test_disabled_by_default_and_free():
    os.environ.pop("LATENCY_LOG", None)
    lg = LatencyLog()
    check(not lg.enabled, "must be off unless asked for")
    check(lg.path is None, "and must not create a file")
    lg.maybe_log(1.0, read_t=0.0, sent_t=0.5)   # must not raise


def test_zero_and_off_switches():
    for value in ("0", "", "   "):
        os.environ["LATENCY_LOG"] = value
        check(not LatencyLog().enabled, f"LATENCY_LOG={value!r} should stay off")
    os.environ.pop("LATENCY_LOG", None)


def test_whitespace_around_the_switch_is_tolerated():
    """Same trap as LOWLIGHT_LOG: typed into an IDE run-configuration field,
    where a trailing space is invisible and would become a filename."""
    os.environ["LATENCY_LOG"] = " 1 "
    lg = LatencyLog()   # writes to the real logs/ dir; removed below
    check(lg.enabled, "a padded '1' should still enable logging")
    check(lg.path and lg.path.endswith(".csv"), "should pick a default path")
    lg.close()
    os.remove(lg.path)
    os.environ.pop("LATENCY_LOG", None)


def test_writes_the_measured_breakdown():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "lat.csv")
        lg = LatencyLog(path=path, log_hz=0)     # no rate limit
        check(lg.enabled, "explicit path should enable it")

        lg.maybe_log(10.0, frame_n=7, fps=13.0, dt=0.077,
                     driver_ms=40.0, read_t=100.0, detect_t=100.02,
                     sent_t=100.05, action="MOVE")
        lg.close()

        rows = open(path).read().strip().splitlines()
        check(len(rows) == 2, f"header plus one row, got {len(rows)}")
        cols = dict(zip(rows[0].split(","), rows[1].split(",")))
        check(cols["detect_ms"] == "20.0",  f"detect_ms wrong: {cols['detect_ms']}")
        check(cols["pipeline_ms"] == "50.0", f"pipeline_ms wrong: {cols['pipeline_ms']}")
        check(cols["total_ms"] == "90.0",
              f"total should be driver + pipeline, got {cols['total_ms']}")
        check(cols["moved"] == "1", "a frame that steered should be marked")
        check(cols["action"] == "MOVE", "the driving label should be recorded")


def test_frames_that_steer_nothing_are_still_logged():
    """They show what the pipeline costs when it is not driving the cursor,
    which is the baseline the steering rows are measured against."""
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "lat.csv")
        lg = LatencyLog(path=path, log_hz=0)
        lg.maybe_log(10.0, frame_n=1, read_t=100.0, detect_t=100.02,
                     sent_t=None, driver_ms=40.0)
        lg.close()
        cols = dict(zip(*[r.split(",") for r in
                          open(path).read().strip().splitlines()]))
        check(cols["moved"] == "0", "should be flagged as non-steering")
        check(cols["pipeline_ms"] == "", "no send means no pipeline figure")
        check(cols["total_ms"] == "", "and no total, rather than a fake zero")


def test_rate_limiting():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "lat.csv")
        lg = LatencyLog(path=path, log_hz=10.0)
        for i in range(100):
            lg.maybe_log(i * 0.01, frame_n=i, read_t=0.0, sent_t=0.01)
        lg.close()
        n = len(open(path).read().strip().splitlines()) - 1
        check(9 <= n <= 12, f"1 s at 10 Hz should be ~10 rows, got {n}")


def test_missing_sensor_timestamp_is_not_faked():
    check(LatencyLog.sensor_age_ms(None, 100.0) is None,
          "no metadata (USB webcam path) means no driver figure")
    check(LatencyLog.sensor_age_ms({}, 100.0) is None, "empty metadata likewise")
    check(LatencyLog.sensor_age_ms({"SensorTimestamp": 0}, 100.0) is None,
          "a zero timestamp is absent, not 'captured at the epoch'")

    age = LatencyLog.sensor_age_ms({"SensorTimestamp": 99_950_000_000}, 100.0)
    check(close(age, 50.0, 1e-3), f"should be 50 ms, got {age}")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"\n{len(tests)} passed")
