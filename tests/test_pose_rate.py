"""Offline check of the pose landmarker's time-based schedule.

Pose is the most expensive detector in the loop, so how often it runs is a
performance decision. It used to run every Nth frame, which ties its rate to
whatever frame rate the rest of the pipeline happens to be achieving — the
one coupling you don't want on the component you are rate-limiting. It now
runs at a fixed Hz off the frame timestamp.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from landmark_pipeline import LandmarkPipeline
import gesture_session as gs

failures = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{'  — ' + detail if detail else ''}")
    if not cond:
        failures.append(name)


class Sched:
    """LandmarkPipeline's pose scheduler, without loading any models."""
    _pose_due = LandmarkPipeline._pose_due

    def __init__(self, hz):
        self._pose_period_ms = 1000.0 / hz if hz > 0 else float("inf")
        self._last_pose_ms = None

    def step(self, ts_ms):
        """One frame. Returns True if pose detection would run."""
        if self._pose_due(ts_ms):
            self._last_pose_ms = ts_ms
            return True
        return False


def run(hz, fps, seconds):
    """Detections per second when the loop runs at `fps`."""
    s = Sched(hz)
    n_frames = int(round(fps * seconds))
    fired = sum(s.step(int(round(i * 1000.0 / fps))) for i in range(n_frames))
    return fired / seconds


print("Pose detection rate vs loop frame rate")
print(f"  target {gs.POSE_DETECT_HZ} Hz\n")
print("  loop fps | old (every 2nd frame) | new (time-based)")
for fps in (30, 20, 12, 8):
    old = fps / 2.0
    new = run(gs.POSE_DETECT_HZ, fps, 10.0)
    print(f"  {fps:>8} | {old:>21.1f} | {new:>16.1f}")
print()

# ── The rate is the rate, whatever the loop is doing ────────────────────────
for fps in (30, 20, 12, 8):
    got = run(gs.POSE_DETECT_HZ, fps, 10.0)
    # A frame every 1/fps s can only land on the deadline or just after it,
    # so the achieved rate is bounded above by the target and below by the
    # target scaled by that quantisation.
    lo = gs.POSE_DETECT_HZ / (1 + gs.POSE_DETECT_HZ / fps)
    check(f"{fps} fps loop -> pose stays near {gs.POSE_DETECT_HZ} Hz",
          lo - 0.11 <= got <= gs.POSE_DETECT_HZ + 0.11, f"{got:.2f} Hz")

# Detection can only happen ON a frame, so a slow loop rounds the rate DOWN
# (12 fps gives 4 Hz: the frame after each 200 ms deadline lands at 250 ms).
# That is the safe direction — the target is a ceiling on how much time pose
# is allowed to cost, and undershooting it only makes the loop faster. What
# must never happen is exceeding it.
check("the target rate is a ceiling, never exceeded",
      all(run(gs.POSE_DETECT_HZ, fps, 10.0) <= gs.POSE_DETECT_HZ + 1e-9
          for fps in (8, 12, 20, 30, 60, 120)))

# The bug this replaces ran in the other direction: with a frame interval of
# 2, pose ran at half the loop rate, so every speed-up of the rest of the
# pipeline spent itself on more pose detections — 6 Hz at the ~12 fps this
# loop actually achieves, but 15 Hz if it ever reached 30.
check("a faster loop does not spend the gain on more pose detections",
      run(gs.POSE_DETECT_HZ, 60, 10.0) <= run(gs.POSE_DETECT_HZ, 12, 10.0) + 1.01,
      "60 fps vs 12 fps")

# ── Scheduling edges ────────────────────────────────────────────────────────
s = Sched(5.0)
check("first frame always detects", s.step(0))
check("no second detect inside the period", not s.step(150))
check("detects again on the period", s.step(200))

# Falling behind must not queue catch-up detections: after a stall, one
# detection, then back on period. Bursting is the opposite of what a rate
# limit on the most expensive component is for.
s = Sched(5.0)
s.step(0)
check("one detect after a 2 s stall", s.step(2000))
check("stall does not queue a burst", not s.step(2050))
check("period resumes from the stall", s.step(2200))

# A restarted session clock would otherwise stall pose until the old
# deadline came around again.
s = Sched(5.0)
s.step(60000)
check("clock reset reschedules immediately", s.step(0))

s = Sched(0)
check("0 Hz disables pose entirely after the first frame",
      s.step(0) and not s.step(10 ** 9))

# ── Freshness cost, which is what we trade for the time ─────────────────────
# The cached pose an ROI/PTZ/zoom update reads can be one period plus one
# frame old. At the loop's real ~12 fps that is ~283 ms, up from ~167 ms when
# pose ran every 2nd frame. Only the crop anchor's *freshness* is affected —
# all three consumers re-read the cache and re-smooth every frame, so their
# response rates are unchanged.
SLOWEST_FPS = 8.0
worst_stale_ms = 1000.0 / gs.POSE_DETECT_HZ + 1000.0 / SLOWEST_FPS
check("cached pose stays fresh enough for a smoothed anchor",
      worst_stale_ms <= 400.0, f"{worst_stale_ms:.0f} ms worst case")

print()
if failures:
    print(f"{len(failures)} FAILED: " + ", ".join(failures))
    sys.exit(1)
print("All pose-rate checks passed.")
