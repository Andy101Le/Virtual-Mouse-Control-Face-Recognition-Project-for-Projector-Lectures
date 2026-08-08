"""
face_recognizer.py
──────────────────
Face recognition against the accounts in login_system.db.

CHANGED: identity now comes from a trained face-recognition embedding
(face_embedder.SFace, 128-d) instead of flattened face-mesh geometry. The
old geometric descriptor could not separate people at all — measured on the
faces actually enrolled here it scored 0.935-0.983 between DIFFERENT users
against a 0.82 accept threshold, so every user matched every other one. See
face_embedder's module docstring for the full diagnosis.

Consequences of that change:

  - Stored embeddings from the old scheme are 1434 floats; new ones are a
    multiple of 128. Old rows are detected by length and treated as NOT
    registered, so those users are prompted to re-register rather than
    being silently compared against an incompatible vector.
  - Enrollment keeps several samples per user rather than collapsing them
    to one mean. A face varies with pose and expression, and max-similarity
    over a few samples tolerates that far better than the distance to a
    single averaged point.
  - In active_user mode we score ONLY the logged-in user. Previously the
    code took the best match across the whole DB and then rejected it if
    the winner wasn't the active user — so a registered bystander who
    happened to out-score you took your lock away and left you UNKNOWN.
    Bystanders are now simply irrelevant.
"""

import logging
import os
import sqlite3

import numpy as np

from face_embedder import EMBED_DIM

log = logging.getLogger(__name__)

_SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
DATABASE_NAME = os.path.join(_SCRIPT_DIR, "login_system.db")

# SFace's reference operating point is 0.363 cosine similarity. Genuine
# pairs cluster far above it and impostors far below, so unlike the old
# geometric threshold this number actually separates people.
RECOGNITION_THRESHOLD = 0.363
UNKNOWN_LABEL         = "UNKNOWN"

# Distance-aware threshold relaxation. A face at 20 ft resolves to a handful
# of pixels, so its embedding is noisier than at enrollment distance and a
# fixed threshold starts rejecting the genuine user intermittently. Below
# SMALL_FACE_SIZE the threshold ramps down linearly, bottoming out at
# RECOGNITION_THRESHOLD - SMALL_FACE_MAX_RELAX by TINY_FACE_SIZE. Modest on
# purpose: an impostor's SFace similarity sits well below even the relaxed
# floor, and AuthManager's temperature debounce still requires sustained
# matches before anything activates.
SMALL_FACE_SIZE      = 0.12
TINY_FACE_SIZE       = 0.05
SMALL_FACE_MAX_RELAX = 0.08

MAX_ENROLL_SAMPLES = 8


class FaceRecognizer:
    def __init__(self, active_user=None):
        """
        active_user: if provided, ONLY this username is compared against.
        Other registered users in frame are bystanders and can neither match
        nor interfere.
        """
        self.face_db      = {}     # username -> (n, 128) float32, L2-normalised rows
        self.active_user  = active_user
        self.stale_users  = []     # enrolled under the old scheme; must re-register
        self._load_db()

    # ── DB management ─────────────────────────────────────────────────────────
    def _load_db(self):
        if not os.path.exists(DATABASE_NAME):
            log.warning("No DB found at '%s'. Create accounts and register "
                        "faces before expecting recognition.", DATABASE_NAME)
            return

        conn = sqlite3.connect(DATABASE_NAME)
        cur  = conn.cursor()
        cur.execute("""
            SELECT username, face_embedding
            FROM users
            WHERE face_registered = 1 AND face_embedding IS NOT NULL
        """)
        rows = cur.fetchall()
        conn.close()

        for username, blob in rows:
            arr = np.frombuffer(blob, dtype=np.float32)
            if arr.size == 0 or arr.size % EMBED_DIM != 0:
                # Old geometric descriptor (1434 floats). Comparing it to an
                # SFace vector is meaningless, and silently scoring 0 would
                # look like "the camera can't see you" rather than "you need
                # to register again".
                self.stale_users.append(username)
                continue
            samples = arr.reshape(-1, EMBED_DIM).astype(np.float32)
            norms   = np.linalg.norm(samples, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            self.face_db[username] = samples / norms

        if self.stale_users:
            log.warning("These users were enrolled with the old geometric "
                        "descriptor and must re-register their face before "
                        "they can be recognised: %s",
                        ", ".join(self.stale_users))
        log.info("Loaded %d user(s) with usable face data%s",
                 len(self.face_db),
                 f"; active session = '{self.active_user}'" if self.active_user
                 else f": {list(self.face_db)}")

    def reload_db(self):
        self.face_db     = {}
        self.stale_users = []
        self._load_db()

    def needs_registration(self, username):
        """True when this user cannot be recognised until they re-enroll."""
        return username in self.stale_users or username not in self.face_db

    # ── Matching ─────────────────────────────────────────────────────────────
    @staticmethod
    def pack_samples(embeddings):
        """
        The (n, 128) array of enrollment samples to store for a user, keeping
        the most diverse MAX_ENROLL_SAMPLES of them, or None if there are
        none. Diversity beats recency: samples taken milliseconds apart are
        near-duplicates and buy no pose tolerance.

        Returned as an array rather than bytes so UserDatabase.save_face_
        embedding's own .astype(float32).tobytes() applies — row-major, which
        is exactly what _load_db's reshape(-1, EMBED_DIM) reconstructs.
        """
        embs = [e for e in embeddings if e is not None]
        if not embs:
            return None
        M = np.stack(embs).astype(np.float32)
        if len(M) > MAX_ENROLL_SAMPLES:
            keep = [0]
            while len(keep) < MAX_ENROLL_SAMPLES:
                # Farthest-point selection: repeatedly take the sample least
                # similar to everything kept so far.
                sims = M @ M[keep].T           # (n, k)
                worst = np.argsort(sims.max(axis=1))
                for idx in worst:
                    if idx not in keep:
                        keep.append(int(idx))
                        break
            M = M[sorted(keep)]
        return M

    @classmethod
    def _threshold_for_size(cls, face_size):
        """Threshold to apply for a face of this normalized size (bbox
        diagonal). Full strictness at/above SMALL_FACE_SIZE, ramping down to
        the relaxed floor at/below TINY_FACE_SIZE."""
        if face_size is None or face_size >= SMALL_FACE_SIZE:
            return RECOGNITION_THRESHOLD
        frac = (SMALL_FACE_SIZE - face_size) / (SMALL_FACE_SIZE - TINY_FACE_SIZE)
        frac = min(max(frac, 0.0), 1.0)
        return RECOGNITION_THRESHOLD - SMALL_FACE_MAX_RELAX * frac

    def score(self, embedding, username):
        """Best cosine similarity between this embedding and any of the
        user's enrolled samples. 0.0 if that user has no usable face data."""
        samples = self.face_db.get(username)
        if samples is None or embedding is None:
            return 0.0
        return float(np.max(samples @ embedding))

    def recognise(self, embedding, face_size=None):
        """
        Identify one face from its embedding.
        Returns (username, similarity_score), or (UNKNOWN_LABEL, best_score).

        In active_user mode only the logged-in user is considered — a
        bystander cannot win the comparison and therefore cannot displace
        the real user's match.
        """
        if embedding is None or not self.face_db:
            return UNKNOWN_LABEL, 0.0

        threshold = self._threshold_for_size(face_size)
        candidates = ([self.active_user] if self.active_user
                      else list(self.face_db))

        best_name, best_score = UNKNOWN_LABEL, 0.0
        for name in candidates:
            s = self.score(embedding, name)
            if s > best_score:
                best_score = s
                best_name  = name if s >= threshold else UNKNOWN_LABEL

        return best_name, best_score

    # ── Geometry helpers for limb tracking (unchanged) ───────────────────────
    @staticmethod
    def get_nose_tip(face_landmarks):
        lm = face_landmarks[4]
        return np.array([lm.x, lm.y], dtype=np.float32)

    @staticmethod
    def get_face_size(face_landmarks):
        pts  = np.array([[lm.x, lm.y] for lm in face_landmarks], dtype=np.float32)
        diag = np.linalg.norm(pts.max(axis=0) - pts.min(axis=0))
        return max(diag, 1e-4)
