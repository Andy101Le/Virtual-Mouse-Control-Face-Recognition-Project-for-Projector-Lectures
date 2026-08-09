"""
scene_light.py
───────────────
Photometric instrumentation for the low-light problem: how bright is the
person we care about, versus how bright is the room, and what is the camera's
auto-exposure doing about it.

WHY THIS EXISTS. Face detection and SFace recognition both fail in the dark,
but "dark" has at least two very different causes and they need opposite
fixes. If the whole scene is dim, the camera needs more light — longer
exposure or more gain. If the scene is BACKLIT — a presenter standing in
front of a bright projector screen — the sensor is already exposing
correctly for what it was asked to meter, and the subject is dark *because*
the average is bright. More exposure won't be granted; the metering target
is what's wrong.

Those two look identical from inside the detection code (no face found) and
are trivially distinguishable from outside it: compare the luminance of the
subject against the luminance of the whole frame. A dim room has both low.
A backlit subject has a low subject reading against a high scene reading.
That ratio is the single most useful number here, and nothing in this
project measured it before.

The subject region is chosen from the best evidence available, which matters
because the interesting moments are exactly the ones where the face is NOT
detectable:

    face landmarks  -> the face's own bounding box (best)
    pose landmarks  -> the head, from the same points _pose_face_size uses
    tracked ROI     -> where the crop is currently pointed
    nothing         -> the whole frame

So when the face drops out, measurement falls back to the body and keeps
reporting — which is the case we are trying to diagnose.

This module only MEASURES. It deliberately does not adjust anything: the
exposure controller that will use these readings is a separate change, and
it should be built against real numbers from a real lecture room rather
than against a guess.
"""

import csv
import logging
import os
import time

import numpy as np

log = logging.getLogger(__name__)

# Rows are written at this rate, plus one forced row whenever the state we
# care about changes (see SceneLightLog.maybe_log). Steady-state rows show
# the trend; the forced ones catch the exact frame recognition was lost,
# which is the row worth having.
LOG_HZ = 5.0

# Luminance is measured on a subsample rather than every pixel: a face box
# can be 300x300 and this runs inside a frame budget that is already over.
# ~1200 samples is far more than enough for a mean and a spread.
TARGET_SAMPLES = 1200

# BGR luma weights (Rec.601). Frames from picamera2's "RGB888" and from
# cv2.VideoCapture are both BGR-ordered in numpy.
_LUMA_B, _LUMA_G, _LUMA_R = 0.114, 0.587, 0.299

# Pose landmark indices for the head, and the visibility below which a point
# is not worth trusting. Kept in step with gesture_session's POSE_HEAD_IDS /
# POSE_HEAD_VIS_THRESH.
POSE_HEAD_IDS        = (0, 2, 5, 7, 8)
POSE_HEAD_VIS_THRESH = 0.35


def luma_stats(frame, rect=None):
    """
    (mean, std) luminance in 0-255 over `rect`, or over the whole frame when
    rect is None. Returns None if the rect lands outside the frame.

    rect is (x0, y0, w, h) in pixels. It is clamped to the frame rather than
    rejected, because a face box near the edge is still worth measuring.
    """
    h, w = frame.shape[:2]
    if rect is None:
        x0, y0, rw, rh = 0, 0, w, h
    else:
        x0, y0, rw, rh = (int(round(v)) for v in rect)
        x0, y0 = max(0, x0), max(0, y0)
        rw = min(rw, w - x0)
        rh = min(rh, h - y0)
    if rw <= 0 or rh <= 0:
        return None

    region = frame[y0:y0 + rh, x0:x0 + rw]
    step = max(1, int((region.shape[0] * region.shape[1] / TARGET_SAMPLES) ** 0.5))
    # Offset by half a step so samples sit in the centre of their cells.
    # Starting at 0 biases the mean toward the top-left of the region, which
    # on a lit-from-one-side face is a real error, not a rounding one.
    off = step // 2
    sub = region[off::step, off::step]
    if sub.size == 0:
        return None

    # float32 keeps this cheap; the dot is over a ~1200x3 array.
    px = sub.reshape(-1, 3).astype(np.float32)
    lum = px[:, 0] * _LUMA_B + px[:, 1] * _LUMA_G + px[:, 2] * _LUMA_R
    return float(lum.mean()), float(lum.std())


def face_rect(face_landmarks, w, h, pad=0.15):
    """Pixel bounding box of a face mesh, padded outward by `pad` of its own
    size so the reading is of the face rather than of its centre."""
    xs = [lm.x for lm in face_landmarks]
    ys = [lm.y for lm in face_landmarks]
    x0, x1 = min(xs) * w, max(xs) * w
    y0, y1 = min(ys) * h, max(ys) * h
    dx, dy = (x1 - x0) * pad, (y1 - y0) * pad
    return (x0 - dx, y0 - dy, (x1 - x0) + 2 * dx, (y1 - y0) + 2 * dy)


def pose_head_rect(pose_lms, w, h):
    """Pixel box around the head, from the pose points that are visible
    enough to trust. None if the head isn't visible."""
    pts = [(lm.x, lm.y) for i in POSE_HEAD_IDS
           for lm in (pose_lms[i],) if lm.visibility > POSE_HEAD_VIS_THRESH]
    if len(pts) < 2:
        return None
    xs = [p[0] * w for p in pts]
    ys = [p[1] * h for p in pts]
    span = max(max(xs) - min(xs), max(ys) - min(ys))
    if span < 1e-3:
        return None
    # The visible head points cover the eyes/ears, not the whole head, so
    # the box is grown around their centre to cover a plausible face.
    cx, cy = (min(xs) + max(xs)) / 2.0, (min(ys) + max(ys)) / 2.0
    side = span * 1.6
    return (cx - side / 2, cy - side / 2, side, side)


def nearest_pose(pose_lms_list, roi_center):
    """The pose whose nose sits closest to where we are already looking, or
    the first one when nothing is being tracked yet. Picking by continuity
    rather than by detection order matters for the same reason it does in
    _update_roi: a bystander must not capture the measurement."""
    if not pose_lms_list:
        return None
    if roi_center is None:
        return pose_lms_list[0]
    best, best_d = None, None
    for lms in pose_lms_list:
        if not len(lms):
            continue
        d = ((lms[0].x - roi_center[0]) ** 2 + (lms[0].y - roi_center[1]) ** 2)
        if best_d is None or d < best_d:
            best, best_d = lms, d
    return best


def subject_rect(frame, face_landmarks=None, pose_lms=None,
                 roi_center=None, roi_face_size=None):
    """
    Where to measure the subject, and which evidence it came from:
    (rect, source) with source one of "face" / "pose" / "roi" / "frame".

    Ordered by how directly the evidence locates a face. The fallbacks are
    the point of this function -- when the face is too dark to detect, the
    body usually still is, and a reading from the body is what tells us
    whether the failure is darkness or something else.
    """
    h, w = frame.shape[:2]
    if face_landmarks:
        return face_rect(face_landmarks, w, h), "face"
    if pose_lms:
        r = pose_head_rect(pose_lms, w, h)
        if r is not None:
            return r, "pose"
    if roi_center is not None:
        side = max(0.04, float(roi_face_size or 0.12)) * w
        return (roi_center[0] * w - side / 2,
                roi_center[1] * h - side / 2, side, side), "roi"
    return None, "frame"


_COLUMNS = [
    "t", "frame", "fps",
    # What the camera decided to do. exp_us and again together are the
    # camera's answer to "how dark is it"; ev is our bias, 0 until the
    # exposure controller exists.
    "exp_us", "again", "dgain", "lux", "ev",
    # The diagnosis. subj_luma well below scene_luma means backlit; both low
    # means a dim room. subj_std is how much facial detail survives.
    "scene_luma", "scene_std", "subj_luma", "subj_std", "subj_src",
    # What the pipeline made of it.
    "n_faces", "n_poses", "face_size", "recog", "score",
    "auth_temp", "user_active", "roi_active",
]


class SceneLightLog:
    """
    Writes one CSV row per sample. Enabled by the LOWLIGHT_LOG environment
    variable; when off, maybe_log() returns immediately and costs nothing,
    so this can stay wired into the loop permanently.

    Set LOWLIGHT_LOG=1 to log to logs/lowlight_<timestamp>.csv, or
    LOWLIGHT_LOG=/some/path.csv to choose the file.
    """

    def __init__(self, path=None, log_hz=LOG_HZ):
        self.enabled = False
        self._fh = None
        self._writer = None
        self._period = 1.0 / log_hz if log_hz > 0 else 0.0
        self._last_t = -1e9
        self._last_state = None
        self.path = None

        # Stripped because this is usually typed into an IDE run-configuration
        # field, where a trailing space is invisible and would otherwise be
        # taken as a filename -- logging to a file called "1 " instead of
        # failing loudly enough to notice.
        env = (os.environ.get("LOWLIGHT_LOG", "") if path is None
               else str(path)).strip()
        if not env or env == "0":
            return

        if env in ("1", "true", "yes"):
            d = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
            os.makedirs(d, exist_ok=True)
            env = os.path.join(d, time.strftime("lowlight_%Y%m%d_%H%M%S.csv"))

        try:
            self._fh = open(env, "w", newline="", buffering=1)
        except OSError:
            log.exception("Could not open low-light log '%s'", env)
            return
        self._writer = csv.writer(self._fh)
        self._writer.writerow(_COLUMNS)
        self.enabled = True
        self.path = env
        log.info("Low-light logging to %s", env)
        print(f"Low-light logging to {env}")

    def close(self):
        if self._fh is not None:
            self._fh.close()
            self._fh = None
        self.enabled = False

    def maybe_log(self, now_t, frame, *, frame_n=0, fps=0.0, metadata=None,
                  face_landmarks=None, pose_lms_list=None, n_faces=0,
                  roi_center=None, roi_face_size=None, face_size=None,
                  recognised=None, score=0.0, auth_temp=0.0,
                  user_active=False):
        """
        Sample the frame if it is time to, or if something we care about just
        changed. The forced-on-change behaviour is the important half: a
        purely periodic log at 5 Hz can easily miss the transition that
        matters, and the transition is the whole reason for the file.
        """
        if not self.enabled:
            return

        # Whether a face is being seen at all, and whether it is being
        # recognised, are the two transitions worth catching exactly.
        state = (n_faces > 0, bool(user_active), recognised or "")
        due = (now_t - self._last_t) >= self._period
        if not due and state == self._last_state:
            return
        self._last_t = now_t
        self._last_state = state

        pose_lms = nearest_pose(pose_lms_list, roi_center)
        rect, src = subject_rect(frame, face_landmarks, pose_lms,
                                 roi_center, roi_face_size)
        scene = luma_stats(frame) or (float("nan"), float("nan"))
        subj = luma_stats(frame, rect) if rect is not None else scene

        md = metadata or {}
        self._writer.writerow([
            f"{now_t:.3f}", frame_n, f"{fps:.1f}",
            md.get("ExposureTime", ""), _f(md.get("AnalogueGain")),
            _f(md.get("DigitalGain")), _f(md.get("Lux")),
            _f(md.get("ExposureValue"), default="0"),
            f"{scene[0]:.1f}", f"{scene[1]:.1f}",
            f"{subj[0]:.1f}", f"{subj[1]:.1f}", src,
            n_faces, len(pose_lms_list or ()),
            _f(face_size, "{:.4f}"), recognised or "", f"{score:.3f}",
            f"{auth_temp:.2f}", int(bool(user_active)),
            int(roi_center is not None),
        ])


def _f(v, fmt="{:.3f}", default=""):
    """Format a metadata value that may be absent."""
    if v is None:
        return default
    try:
        return fmt.format(float(v))
    except (TypeError, ValueError):
        return default
