"""
auth_manager.py
────────────────
Face-recognition based authentication state machine.

Wraps FaceRecognizer with a temperature-style debounce: recognising the
active user's face raises an "authentication temperature", not seeing them
lets it decay. The user is considered ACTIVE once temperature crosses
TEMP_ACTIVATE, and stays active through brief look-aways thanks to a grace
period, only deactivating once temperature drops below TEMP_DEACTIVATE AND
the grace period has elapsed.

FIXED: which face gets trusted. This used to process only face index 0 —
`for fi, face_lms in enumerate(...): if fi == 0:` — so the identity, the
nose position driving the PTZ, and the auth temperature all came from
whichever face MediaPipe happened to return first. That order is detection
order, not "the person we're tracking". A bystander landing at index 0 took
over the tracker and decayed the real user's temperature, while the actual
user sat at index 1 with their recognition result computed and discarded.
Now every face is scored and the best match for the active user wins, with
position continuity breaking ties among unrecognised faces.

Identity comes from embeddings computed off-thread (see face_embedder), so
this consumes whatever the worker last published and tolerates it being a
frame or two stale.
"""

import numpy as np

from face_recognizer import FaceRecognizer, UNKNOWN_LABEL

# How far (normalized) a published embedding's face may sit from a face in
# the current frame and still be considered the same person. Faces move a
# little between the submitted frame and now; different people in a room
# are much further apart than this.
EMBED_MATCH_RADIUS = 0.12


class AuthManager:
    TEMP_ACTIVATE   = 0.60
    TEMP_DEACTIVATE = 0.25
    GRACE_SECONDS   = 10.0

    def __init__(self, active_user, face_detect_interval=3):
        self.face_rec        = FaceRecognizer(active_user=active_user)
        self.num_registered  = len(self.face_rec.face_db)
        self.active_user     = active_user

        # Compensate for skipped frames between face-detect intervals
        self.temp_rise = 0.08 * face_detect_interval
        self.temp_fall = 0.04 * face_detect_interval

        self.auth_temp      = 0.0
        self.user_active    = False
        self.last_seen_time = 0.0

        self.recognised_user = UNKNOWN_LABEL
        self.recog_score     = 0.0
        self.face_nose_pos   = None
        self.face_size       = None

    def update(self, face_landmarks_list, now_t, embeddings=None):
        """
        Call only on frames where a fresh face-detection result is
        available (LandmarkPipeline.detect()'s face_updated flag).

        embeddings: [(anchor_xy, vector)] most recently published by the
        async embedder, or None. Each is matched back to a face in this
        frame by nose proximity, since it may describe a slightly older
        frame.
        """
        prev_anchor = self.face_nose_pos

        self.recognised_user = UNKNOWN_LABEL
        self.recog_score     = 0.0
        self.face_nose_pos   = None
        self.face_size       = None

        if not face_landmarks_list:
            self._decay(now_t)
            return self.user_active

        best = None          # (score, is_match, face_lms)
        for face_lms in face_landmarks_list:
            nose = self.face_rec.get_nose_tip(face_lms)
            size = self.face_rec.get_face_size(face_lms)
            emb  = self._embedding_for(nose, embeddings)
            user, score = self.face_rec.recognise(emb, face_size=size)

            is_match = (user != UNKNOWN_LABEL and
                        (self.active_user is None or user == self.active_user))
            # Rank: a genuine match always beats a non-match; among equals,
            # prefer the face nearest where we were already looking, so an
            # unrecognised frame doesn't hop the tracker to a bystander.
            key = (1 if is_match else 0, score if is_match
                   else -_dist(nose, prev_anchor))
            if best is None or key > best[0]:
                best = (key, user, score, nose, size)

        _, user, score, nose, size = best
        self.recognised_user = user
        self.recog_score     = score
        self.face_nose_pos   = nose
        self.face_size       = size

        if user != UNKNOWN_LABEL:
            self.auth_temp      = min(1.0, self.auth_temp + self.temp_rise)
            self.last_seen_time = now_t
        else:
            self.auth_temp = max(0.0, self.auth_temp - self.temp_fall)

        self._apply_thresholds(now_t)
        return self.user_active

    def _embedding_for(self, nose, embeddings):
        """The published embedding whose face sits closest to this one, or
        None if nothing published is near enough to be the same person."""
        if not embeddings:
            return None
        best_vec, best_d = None, EMBED_MATCH_RADIUS
        for anchor, vec in embeddings:
            d = _dist(nose, anchor)
            if d < best_d:
                best_vec, best_d = vec, d
        return best_vec

    def _decay(self, now_t):
        self.auth_temp = max(0.0, self.auth_temp - self.temp_fall)
        self._apply_thresholds(now_t)

    def _apply_thresholds(self, now_t):
        if not self.user_active:
            if self.auth_temp >= self.TEMP_ACTIVATE:
                self.user_active = True
        else:
            if (self.auth_temp < self.TEMP_DEACTIVATE and
                    (now_t - self.last_seen_time) > self.GRACE_SECONDS):
                self.user_active = False

    @property
    def limb_mode(self):
        return self.user_active

    @property
    def is_registered_face_visible(self):
        return self.user_active and self.recognised_user != UNKNOWN_LABEL

    def grace_remaining(self, now_t):
        return max(0.0, self.GRACE_SECONDS - (now_t - self.last_seen_time))


def _dist(a, b):
    if a is None or b is None:
        return 1e9
    return float(np.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1])))
