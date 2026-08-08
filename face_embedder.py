"""
face_embedder.py
─────────────────
Real face-recognition embeddings, replacing the landmark-geometry
"embedding" this project used to compare faces with.

WHY THIS EXISTS. The old approach flattened all 478 MediaPipe face-mesh
landmarks, centred them on the nose tip and L2-normalised. Measured on the
three faces actually enrolled in login_system.db, that descriptor scored
0.935 / 0.983 / 0.978 cosine similarity between DIFFERENT people, against a
0.82 accept threshold — every enrolled user matched every other one. Each
was also 0.98+ similar to the average of all three, which is the diagnosis:
the vector encodes "this is a human face" almost entirely, and identity is
a small residual buried under it. No threshold could separate those.

The fix is a descriptor trained for identity rather than derived from
geometry. SFace (OpenCV Zoo, ~99.6% on LFW) maps an aligned 112x112 face
crop to 128 dimensions where cosine similarity means what we need it to
mean. OpenCV ships the runtime in opencv-contrib, so this is a model file,
not a new Python dependency.

ALIGNMENT. SFace expects the face pre-aligned by a similarity transform
onto a canonical template, driven by five points: both eyes, the nose tip
and both mouth corners. cv2.FaceRecognizerSF.alignCrop wants them in
YuNet's row layout, which orders them by IMAGE side (its "right eye" is the
one nearer x=0). We derive all five from the mesh we already compute, and
assign left/right by comparing x rather than trusting a fixed landmark
index — the vision loop hands us a MIRRORED frame (cv2.flip), so a
hardcoded "left iris is index 468" would silently swap the eyes and feed
alignCrop a horizontally reflected transform. Ordering by x is correct
under mirroring, and identical at enrollment and at recognition time,
which is what actually matters.

THREADING. One embedding costs ~56 ms on a Pi 5 — far too much to run
inline on a 33 ms vision-loop budget. Embedding therefore happens on a
worker thread: the vision loop calls submit() with a frame and this
frame's faces (non-blocking, drops rather than queues), and reads whatever
the worker last published via result(). Identity that is 100-200 ms stale
is fine; the auth state machine debounces over far longer than that.
"""

import logging
import os
import threading

import cv2
import numpy as np

log = logging.getLogger(__name__)

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "face_recognition_sface.onnx")

EMBED_DIM = 128

# Mesh indices for the five alignment points. The two iris centres come from
# the 478-point (refined-landmark) model; 468/473 are the iris centres and
# 33/263 are the eye corners used as a fallback if a 468-point model is
# loaded instead.
IRIS_A, IRIS_B         = 468, 473
EYE_CORNER_A, EYE_CORNER_B = 33, 263
NOSE_TIP               = 1
MOUTH_A, MOUTH_B       = 61, 291

# How many faces one submission may embed. Each costs ~56 ms on the worker,
# so an unbounded list would make identity arbitrarily stale in a crowded
# room. Faces are ranked before truncation (see FaceEmbedder.submit).
MAX_FACES_PER_CYCLE = 3


def five_points(face_landmarks, w, h):
    """
    The five alignment points in pixel coords, in YuNet row order:
    right eye, left eye, nose tip, right mouth corner, left mouth corner —
    where "right" means nearer x=0 in the image, not anatomically right.
    Returns None if the mesh is too short to supply them.
    """
    n = len(face_landmarks)
    if n <= max(MOUTH_B, NOSE_TIP):
        return None

    def px(i):
        lm = face_landmarks[i]
        return (lm.x * w, lm.y * h)

    if n > max(IRIS_A, IRIS_B):
        eye_a, eye_b = px(IRIS_A), px(IRIS_B)
    else:
        eye_a, eye_b = px(EYE_CORNER_A), px(EYE_CORNER_B)

    # Order each pair by image x so mirroring can't swap them.
    r_eye, l_eye = sorted((eye_a, eye_b), key=lambda p: p[0])
    r_mouth, l_mouth = sorted((px(MOUTH_A), px(MOUTH_B)), key=lambda p: p[0])
    return r_eye, l_eye, px(NOSE_TIP), r_mouth, l_mouth


def _align_row(face_landmarks, w, h):
    """Build the 15-float row alignCrop expects: bbox then the five points."""
    pts = five_points(face_landmarks, w, h)
    if pts is None:
        return None
    xs = [lm.x * w for lm in face_landmarks]
    ys = [lm.y * h for lm in face_landmarks]
    x0, y0 = min(xs), min(ys)
    row = [x0, y0, max(xs) - x0, max(ys) - y0]
    for p in pts:
        row.extend(p)
    row.append(1.0)                      # detection score, unused by alignCrop
    return np.array(row, dtype=np.float32)


class FaceEmbedder:
    """Synchronous SFace wrapper. Thread-confined: give each thread its own,
    or call it only from the worker below."""

    def __init__(self, model_path=MODEL_PATH):
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Face recognition model missing: {model_path}. "
                f"Run setup/get_models.sh to download it.")
        self._rec = cv2.FaceRecognizerSF.create(model_path, "")

    def embed(self, frame, face_landmarks):
        """L2-normalised 128-d identity vector for one face, or None if the
        face can't be aligned. `frame` is the BGR image the landmarks were
        detected against, in that image's own pixel coordinates."""
        h, w = frame.shape[:2]
        row = _align_row(face_landmarks, w, h)
        if row is None:
            return None
        try:
            crop = self._rec.alignCrop(frame, row)
            vec  = self._rec.feature(crop).flatten().astype(np.float32)
        except cv2.error:
            log.debug("alignCrop/feature failed for a face", exc_info=True)
            return None
        norm = float(np.linalg.norm(vec)) or 1.0
        return vec / norm            # so comparison is a plain dot product


class AsyncFaceEmbedder:
    """
    Runs FaceEmbedder on a worker thread. submit() never blocks and never
    queues more than one job — a backlog would only serve stale identities.
    """

    def __init__(self, model_path=MODEL_PATH):
        self._embedder = FaceEmbedder(model_path)

        self._job_lock = threading.Lock()
        self._job      = None            # (frame, [landmark lists], token)
        self._wake     = threading.Event()

        self._res_lock = threading.Lock()
        self._result   = None            # (token, [(anchor_xy, embedding)])

        self._stop   = threading.Event()
        self._worker = threading.Thread(target=self._run, daemon=True,
                                        name="face-embed")
        self._worker.start()

    def close(self):
        self._stop.set()
        self._wake.set()
        self._worker.join(timeout=2.0)

    def submit(self, frame, face_landmarks_list, anchor=None, token=0):
        """
        Queue this frame's faces for embedding, replacing any job not yet
        started. `anchor` is the normalized (x, y) we're currently tracking;
        faces are ranked by closeness to it (then by size) before the list is
        truncated, so the person we care about is embedded first when the
        room is busy.
        """
        if not face_landmarks_list:
            return
        ranked = sorted(
            face_landmarks_list,
            key=lambda lms: _rank_key(lms, anchor))[:MAX_FACES_PER_CYCLE]
        # The camera buffer is reused in place, so the worker must own a copy.
        with self._job_lock:
            self._job = (frame.copy(), ranked, token)
        self._wake.set()

    def result(self):
        """Latest (token, [(anchor_xy, embedding)]) the worker published, or
        None. anchor_xy is the face's normalized nose position, used to match
        a possibly-stale embedding back to a face in the current frame."""
        with self._res_lock:
            return self._result

    def _run(self):
        while not self._stop.is_set():
            self._wake.wait(timeout=0.2)
            self._wake.clear()
            with self._job_lock:
                job, self._job = self._job, None
            if job is None:
                continue
            frame, faces, token = job
            h, w = frame.shape[:2]
            out = []
            for lms in faces:
                if self._stop.is_set():
                    break
                vec = self._embedder.embed(frame, lms)
                if vec is not None:
                    out.append(((float(lms[NOSE_TIP].x), float(lms[NOSE_TIP].y)),
                                vec))
            with self._res_lock:
                self._result = (token, out)


def _rank_key(face_landmarks, anchor):
    """Sort key: distance from the tracked anchor, falling back to biggest
    face first when nothing is being tracked yet."""
    nose = face_landmarks[NOSE_TIP]
    if anchor is not None:
        return ((nose.x - anchor[0]) ** 2 + (nose.y - anchor[1]) ** 2)
    xs = [lm.x for lm in face_landmarks]
    ys = [lm.y for lm in face_landmarks]
    return -((max(xs) - min(xs)) * (max(ys) - min(ys)))
