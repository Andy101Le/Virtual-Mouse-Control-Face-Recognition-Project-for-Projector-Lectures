"""
face_recognizer.py
──────────────────
Stateful face recognition module used by main.py.

Loads face embeddings from faces.pkl and compares them against live
MediaPipe FaceLandmarker output using cosine similarity.

Also exposes:
  - get_face_anchor()  : nose-tip position (normalised 0-1) for limb tracking
  - get_face_size()    : face bounding-box diagonal in normalised units
  - get_arm_vector()   : unit vector from nose → index fingertip (body-relative cursor)
"""

import pickle
import os
import numpy as np
import time

FACES_DB_PATH     = "faces.pkl"
RECOGNITION_THRESHOLD = 0.82   # cosine similarity; above this → recognised
UNKNOWN_LABEL         = "UNKNOWN"
# How many consecutive recognised frames before we consider the user authenticated
AUTH_FRAMES_NEEDED    = 10

class FaceRecognizer:
    def __init__(self):
        self.face_db      = {}
        self._auth_count  = {}   # username → consecutive recognised frames
        self.current_user = None
        self.authenticated= False
        self._load_db()

    # ── DB management ─────────────────────────────────────────────────────────
    def _load_db(self):
        if os.path.exists(FACES_DB_PATH):
            with open(FACES_DB_PATH, "rb") as f:
                self.face_db = pickle.load(f)
            print(f"[FaceRecognizer] Loaded {len(self.face_db)} user(s): {list(self.face_db.keys())}")
        else:
            print(f"[FaceRecognizer] No face DB found at '{FACES_DB_PATH}'.")
            print("  Run face_register.py first to register users.")

    def reload_db(self):
        """Hot-reload the DB without restarting."""
        self._load_db()

    # ── Core embedding + recognition ─────────────────────────────────────────
    @staticmethod
    def extract_embedding(face_landmarks):
        """
        Same normalisation as face_register.py — must match exactly.
        Returns L2-normalised flat array of shape (1434,).
        """
        pts = np.array([[lm.x, lm.y, lm.z] for lm in face_landmarks], dtype=np.float32)
        pts -= pts[4]                              # centre on nose tip (idx 4)
        scale = np.max(np.abs(pts)) or 1.0
        pts  /= scale
        flat  = pts.flatten()
        norm  = np.linalg.norm(flat) or 1.0
        return flat / norm

    def recognise(self, face_landmarks):
        """
        Compare live face against all registered embeddings.
        Returns (username, similarity_score) or (UNKNOWN_LABEL, best_score).
        Updates self.authenticated and self.current_user.
        """
        if not self.face_db:
            return UNKNOWN_LABEL, 0.0

        live_emb   = self.extract_embedding(face_landmarks)
        best_name  = UNKNOWN_LABEL
        best_score = 0.0

        for name, stored_emb in self.face_db.items():
            # Cosine similarity: 1.0 = identical, 0.0 = orthogonal
            score = float(np.dot(live_emb, stored_emb))   # both L2-normalised
            if score > best_score:
                best_score = score
                best_name  = name if score >= RECOGNITION_THRESHOLD else UNKNOWN_LABEL

        # Debounce: require AUTH_FRAMES_NEEDED consecutive matches
        if best_name != UNKNOWN_LABEL:
            self._auth_count[best_name] = self._auth_count.get(best_name, 0) + 1
            if self._auth_count[best_name] >= AUTH_FRAMES_NEEDED:
                self.current_user  = best_name
                self.authenticated = True
        else:
            # Reset all counters on unknown
            self._auth_count    = {}
            self.current_user   = None
            self.authenticated  = False

        return best_name, best_score

    # ── Geometry helpers for limb tracking ───────────────────────────────────
    @staticmethod
    def get_nose_tip(face_landmarks):
        """
        Returns (x, y) normalised [0,1] of nose tip (landmark 4).
        Used as the head anchor point for arm-vector computation.
        """
        lm = face_landmarks[4]
        return np.array([lm.x, lm.y], dtype=np.float32)

    @staticmethod
    def get_face_size(face_landmarks):
        """
        Returns the approximate face size as the diagonal of the landmark
        bounding box in normalised coords. Used to make arm-vector scale-invariant.
        """
        pts = np.array([[lm.x, lm.y] for lm in face_landmarks], dtype=np.float32)
        diag = np.linalg.norm(pts.max(axis=0) - pts.min(axis=0))
        return max(diag, 1e-4)

    @staticmethod
    def get_arm_vector(nose_pos, fingertip_pos, face_size):
        """
        Compute body-relative arm extension vector.

        Args:
          nose_pos:     (x, y) normalised — head anchor
          fingertip_pos:(x, y) normalised — index fingertip (hand landmark 8)
          face_size:    float — face diagonal in normalised units (scale reference)

        Returns:
          arm_vec: (dx, dy) — signed arm extension in face-size units
                   e.g. (2.0, 0.5) means hand is 2 face-widths to the right,
                   0.5 face-widths below.

        Why this matters:
          Without this, the cursor position depends on where the hand is
          on-screen absolutely. With this, only the arm's EXTENSION relative
          to the body matters — so moving your whole body doesn't move the cursor,
          and the system works the same whether you're close or far from the camera.
        """
        raw_vec = fingertip_pos - nose_pos        # unnormalised direction
        return raw_vec / face_size                # normalised by face size


UNKNOWN_LABEL = "UNKNOWN"   # re-export for main.py
