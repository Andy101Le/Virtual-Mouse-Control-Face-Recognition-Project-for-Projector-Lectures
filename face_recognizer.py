"""
face_recognizer.py
──────────────────
Stateful face recognition module used by main.py.

CHANGED from the pickle-based version:
  - Embeddings load from SQLite (login_system.db) instead of faces.pkl.
  - Constructor takes an optional `active_user` argument. When set,
    only that user's embedding is considered a valid match — bystanders
    who happen to be registered in the DB but are NOT the logged-in user
    are treated as UNKNOWN. This enforces the account boundary.

Everything else (cosine similarity, AUTH_FRAMES_NEEDED debouncing, the
limb-tracking geometry helpers) is unchanged.
"""

import os
import sqlite3
import numpy as np

DATABASE_NAME         = "login_system.db"
RECOGNITION_THRESHOLD = 0.82   # cosine similarity; above this → recognised
UNKNOWN_LABEL         = "UNKNOWN"
AUTH_FRAMES_NEEDED    = 10     # consecutive recognised frames before authenticated


class FaceRecognizer:
    def __init__(self, active_user=None):
        """
        active_user: if provided, only this username will be accepted as
        a positive match. Other registered users seen by the camera will
        be reported as UNKNOWN (they're bystanders for this session).
        """
        self.face_db      = {}
        self._auth_count  = {}
        self.current_user = None
        self.authenticated= False
        self.active_user  = active_user
        self._load_db()

    # ── DB management ─────────────────────────────────────────────────────────
    def _load_db(self):
        if not os.path.exists(DATABASE_NAME):
            print(f"[FaceRecognizer] No DB found at '{DATABASE_NAME}'.")
            print("  Run login_system.py first to create accounts and register faces.")
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
            emb = np.frombuffer(blob, dtype=np.float32)
            self.face_db[username] = emb

        if self.active_user:
            print(f"[FaceRecognizer] Loaded {len(self.face_db)} user(s); "
                  f"active session = '{self.active_user}'")
        else:
            print(f"[FaceRecognizer] Loaded {len(self.face_db)} user(s): "
                  f"{list(self.face_db.keys())}")

    def reload_db(self):
        self.face_db = {}
        self._load_db()

    # ── Core embedding + recognition ─────────────────────────────────────────
    @staticmethod
    def extract_embedding(face_landmarks):
        """Same normalisation used at enrollment time — must match exactly."""
        pts = np.array([[lm.x, lm.y, lm.z] for lm in face_landmarks], dtype=np.float32)
        pts -= pts[4]                                # centre on nose tip
        scale = np.max(np.abs(pts)) or 1.0
        pts  /= scale
        flat  = pts.flatten()
        norm  = np.linalg.norm(flat) or 1.0
        return flat / norm

    def recognise(self, face_landmarks):
        """
        Compare live face against the DB.
        Returns (username, similarity_score) or (UNKNOWN_LABEL, best_score).
        If active_user is set, only that user counts as a positive match.
        """
        if not self.face_db:
            return UNKNOWN_LABEL, 0.0

        live_emb   = self.extract_embedding(face_landmarks)
        best_name  = UNKNOWN_LABEL
        best_score = 0.0

        for name, stored_emb in self.face_db.items():
            score = float(np.dot(live_emb, stored_emb))   # both L2-normalised
            if score > best_score:
                best_score = score
                if score >= RECOGNITION_THRESHOLD:
                    # Account-boundary enforcement: in active_user mode,
                    # only the logged-in user's match counts.
                    if self.active_user is None or name == self.active_user:
                        best_name = name
                    else:
                        best_name = UNKNOWN_LABEL
                else:
                    best_name = UNKNOWN_LABEL

        # Debounce
        if best_name != UNKNOWN_LABEL:
            self._auth_count[best_name] = self._auth_count.get(best_name, 0) + 1
            if self._auth_count[best_name] >= AUTH_FRAMES_NEEDED:
                self.current_user  = best_name
                self.authenticated = True
        else:
            self._auth_count    = {}
            self.current_user   = None
            self.authenticated  = False

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
