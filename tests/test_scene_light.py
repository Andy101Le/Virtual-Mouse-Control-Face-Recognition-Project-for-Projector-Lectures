"""Offline check of the low-light instrumentation.

The point of scene_light is to tell two failure modes apart that look
identical from inside the detector: a uniformly dim room, and a subject
who is dark only because the scene behind them is bright. These tests build
both situations synthetically and assert the numbers separate them, then
check the region-selection fallbacks that keep the measurement alive after
the face becomes undetectable -- which is exactly when it matters.
"""
import sys
import os
import csv
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import scene_light as sl

failures = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{'  — ' + detail if detail else ''}")
    if not cond:
        failures.append(name)


W, H = 1280, 960


class LM:
    def __init__(self, x, y, z=0.0, vis=1.0):
        self.x, self.y, self.z, self.visibility = x, y, z, vis


def face_mesh(cx, cy, size=0.10, n=478):
    """A synthetic face mesh spread over a box of `size` around (cx, cy)."""
    rng = np.random.default_rng(1)
    pts = rng.uniform(-0.5, 0.5, (n, 2)) * size
    # Guarantee the extremes so the bbox is exactly `size` across.
    pts[0] = (-size / 2, -size / 2)
    pts[1] = (size / 2, size / 2)
    return [LM(cx + p[0], cy + p[1]) for p in pts]


def pose_body(cx=0.5, cy=0.35, head=0.09, vis=0.9, n=33):
    lms = [LM(cx, cy, vis=vis) for _ in range(n)]
    for i, (dx, dy) in zip(sl.POSE_HEAD_IDS,
                           [(0, 0), (-.3, -.2), (.3, -.2), (-.5, 0), (.5, 0)]):
        lms[i] = LM(cx + dx * head, cy + dy * head, vis=vis)
    return lms


def scene(bg, subject=None, rect=None):
    """A frame of uniform `bg`, optionally with a brighter/darker patch."""
    f = np.full((H, W, 3), bg, dtype=np.uint8)
    if subject is not None and rect is not None:
        x0, y0, w, h = (int(v) for v in rect)
        f[y0:y0 + h, x0:x0 + w] = subject
    return f


print("Telling a dim room apart from a backlit subject")

# Subject box used by both scenarios, matching face_mesh(0.5, 0.4, 0.10).
FACE = face_mesh(0.5, 0.4, 0.10)
frect = sl.face_rect(FACE, W, H)

# 1. Uniformly dim room: subject and scene are both dark, and similar.
dim = scene(28, 30, frect)
dim_scene = sl.luma_stats(dim)[0]
dim_subj = sl.luma_stats(dim, frect)[0]

# 2. Backlit: bright scene, dark subject -- the projector case.
lit = scene(200, 30, frect)
lit_scene = sl.luma_stats(lit)[0]
lit_subj = sl.luma_stats(lit, frect)[0]

print(f"  dim room : scene {dim_scene:5.1f}  subject {dim_subj:5.1f}  "
      f"ratio {dim_subj / dim_scene:.2f}")
print(f"  backlit  : scene {lit_scene:5.1f}  subject {lit_subj:5.1f}  "
      f"ratio {lit_subj / lit_scene:.2f}")

check("both scenarios read the subject as dark",
      dim_subj < 60 and lit_subj < 60)
check("the dim room reads dark overall", dim_scene < 60)
check("the backlit scene reads bright overall", lit_scene > 150)
# This ratio is the whole diagnosis: it is the only thing that separates
# "needs more light" from "is metering the wrong thing".
check("subject/scene ratio separates the two",
      (dim_subj / dim_scene) > 0.5 and (lit_subj / lit_scene) < 0.5,
      f"{dim_subj / dim_scene:.2f} vs {lit_subj / lit_scene:.2f}")

# ── Region selection, in the order evidence degrades ────────────────────────
print("\nSubject region falls back as evidence disappears")
f = scene(40)
r, src = sl.subject_rect(f, face_landmarks=FACE, pose_lms=pose_body())
check("a detected face wins", src == "face")

r, src = sl.subject_rect(f, face_landmarks=None, pose_lms=pose_body())
check("the body carries the measurement once the face is gone", src == "pose")
check("the pose head box is on the head",
      r is not None and abs((r[0] + r[2] / 2) / W - 0.5) < 0.02
      and abs((r[1] + r[3] / 2) / H - 0.35) < 0.02)

r, src = sl.subject_rect(f, None, None, roi_center=(0.3, 0.6), roi_face_size=0.1)
check("the tracked crop is the next fallback", src == "roi")

r, src = sl.subject_rect(f, None, None)
check("with no evidence at all it reads the whole frame",
      src == "frame" and r is None)

# An invisible body must not produce a bogus reading of the background.
r, src = sl.subject_rect(f, None, pose_body(vis=0.1))
check("an invisible head is not measured", src == "frame")

# ── Rect handling ───────────────────────────────────────────────────────────
print("\nMeasurement edges")
check("a rect off the frame returns None",
      sl.luma_stats(f, (W + 10, 10, 50, 50)) is None)
edge = sl.luma_stats(f, (-40, -40, 100, 100))
check("a rect overhanging the edge is clamped, not rejected",
      edge is not None and abs(edge[0] - 40) < 1.0)
check("a zero-size rect returns None", sl.luma_stats(f, (10, 10, 0, 10)) is None)

# Subsampling must not change the answer materially.
grad = np.tile(np.linspace(0, 255, W, dtype=np.uint8), (H, 1))
grad = np.repeat(grad[:, :, None], 3, axis=2)
full = float((grad[:, :, 0].astype(np.float32)).mean())
check("subsampling tracks the true mean on a gradient",
      abs(sl.luma_stats(grad)[0] - full) < 2.0,
      f"{sl.luma_stats(grad)[0]:.1f} vs {full:.1f}")

# Flat field has no spread; a half/half field has a lot. std is the
# "is there any detail left" signal.
check("std is ~0 on a flat field", sl.luma_stats(scene(50))[1] < 0.5)
half = scene(0); half[:, : W // 2] = 255
check("std is large on a split field", sl.luma_stats(half)[1] > 100)

# ── Pose selection must follow the tracked person, not detection order ──────
print("\nBystander handling")
mine, theirs = pose_body(cx=0.7), pose_body(cx=0.2)
picked = sl.nearest_pose([theirs, mine], roi_center=(0.72, 0.35))
check("the pose nearest the tracked anchor is measured",
      picked is mine)
check("with no anchor the first pose is used",
      sl.nearest_pose([theirs, mine], None) is theirs)
check("an empty pose list gives None", sl.nearest_pose([], (0.5, 0.5)) is None)

# ── The logger ──────────────────────────────────────────────────────────────
print("\nLogger")
lg = sl.SceneLightLog(path="0")
check("disabled by default without the env var", not lg.enabled)
lg.maybe_log(0.0, f)          # must be a no-op, not a crash

tmp = os.path.join(tempfile.mkdtemp(), "ll.csv")
lg = sl.SceneLightLog(path=tmp, log_hz=5.0)
check("enabled when given a path", lg.enabled)

md = {"ExposureTime": 33000, "AnalogueGain": 8.0, "Lux": 12.5}
lg.maybe_log(0.0, lit, metadata=md, face_landmarks=FACE, n_faces=1,
             recognised="andy", score=0.51, user_active=True)
# Inside the period and with no state change: must not write.
lg.maybe_log(0.05, lit, metadata=md, face_landmarks=FACE, n_faces=1,
             recognised="andy", score=0.51, user_active=True)
# Losing the face is a state change, so it must write immediately even
# though only 60 ms have passed -- this is the row worth having.
lg.maybe_log(0.11, lit, metadata=md, n_faces=0, pose_lms_list=[pose_body()],
             recognised="UNKNOWN", score=0.0, user_active=True)
lg.close()

with open(tmp) as fh:
    rows = list(csv.DictReader(fh))
check("rate limiting drops the redundant sample", len(rows) == 2,
      f"{len(rows)} rows")
check("losing the face forces a row regardless of rate",
      rows[-1]["n_faces"] == "0" and rows[-1]["subj_src"] == "pose")
check("camera metadata is recorded",
      rows[0]["exp_us"] == "33000" and rows[0]["again"] == "8.000")
check("the backlit signature is visible in the row",
      float(rows[0]["subj_luma"]) < 60 and float(rows[0]["scene_luma"]) > 150)
check("missing metadata keys leave blanks rather than crashing",
      rows[0]["lux"] == "12.500" and rows[-1]["dgain"] == "")

print()
if failures:
    print(f"{len(failures)} FAILED: " + ", ".join(failures))
    sys.exit(1)
print("All scene-light checks passed.")
