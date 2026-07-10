"""
auth_manager.py
────────────────
Face-recognition based authentication state machine used by main.py.

Wraps FaceRecognizer (cosine-similarity matching against the DB) with
a temperature-style debounce: recognising the active user's face raises
an "authentication temperature", not seeing them lets it decay. The
user is considered ACTIVE once temperature crosses TEMP_ACTIVATE, and
stays active through brief look-aways thanks to a grace period, only
deactivating once temperature drops below TEMP_DEACTIVATE AND the grace
period has elapsed.
"""

from face_recognizer import FaceRecognizer, UNKNOWN_LABEL


class AuthManager:
    TEMP_ACTIVATE   = 0.60
    TEMP_DEACTIVATE = 0.25
    GRACE_SECONDS   = 10.0

    def __init__(self, active_user, face_detect_interval=3):
        self.face_rec        = FaceRecognizer(active_user=active_user)
        self.num_registered  = len(self.face_rec.face_db)

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

    def update(self, face_landmarks_list, now_t):
        """
        Call only on frames where a fresh face-detection result is
        available (LandmarkPipeline.detect()'s face_updated flag).
        Updates recognised_user / face_nose_pos / user_active.
        Returns self.user_active for convenience.
        """
        self.recognised_user = UNKNOWN_LABEL
        self.recog_score     = 0.0
        self.face_nose_pos   = None
        self.face_size       = None

        for fi, face_lms in enumerate(face_landmarks_list):
            user, score = self.face_rec.recognise(face_lms)
            if fi == 0:
                self.recognised_user = user
                self.recog_score     = score
                self.face_nose_pos   = self.face_rec.get_nose_tip(face_lms)
                self.face_size       = self.face_rec.get_face_size(face_lms)

                if user != UNKNOWN_LABEL:
                    self.auth_temp      = min(1.0, self.auth_temp + self.temp_rise)
                    self.last_seen_time = now_t
                else:
                    self.auth_temp = max(0.0, self.auth_temp - self.temp_fall)

                if not self.user_active:
                    if self.auth_temp >= self.TEMP_ACTIVATE:
                        self.user_active = True
                else:
                    if (self.auth_temp < self.TEMP_DEACTIVATE and
                            (now_t - self.last_seen_time) > self.GRACE_SECONDS):
                        self.user_active = False

        return self.user_active

    @property
    def limb_mode(self):
        return self.user_active

    @property
    def is_registered_face_visible(self):
        return self.user_active and self.recognised_user != UNKNOWN_LABEL

    def grace_remaining(self, now_t):
        return max(0.0, self.GRACE_SECONDS - (now_t - self.last_seen_time))
