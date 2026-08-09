"""
landmark_pipeline.py
─────────────────────
Wraps the three MediaPipe VIDEO-mode landmarkers (hand / face / pose)
used by main.py, and owns the "run every N frames, cache the rest"
scheduling so main.py's loop doesn't have to.

Hand detection is cheap and responsive enough to run every frame.
Face and pose move more slowly, so their results are only refreshed
periodically and cached in between: face every FACE_DETECT_INTERVAL
frames, pose at a fixed POSE_DETECT_HZ rate.

Pose is scheduled in *time*, not frames, because it is the most
expensive detector and the loop's frame rate is neither 30 fps nor
stable. A frame-count interval silently changes the pose rate whenever
the rest of the pipeline speeds up or slows down — exactly the coupling
you don't want on the component you are rate-limiting to save time.
Face stays frame-counted: AuthManager derives its temperature
rise/fall per update from face_detect_interval, so the two must agree.
"""

import os
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions


class LandmarkPipeline:
    def __init__(self,
                 hand_task_path, face_task_path, pose_task_path,
                 num_hands=2, face_detect_interval=3, pose_detect_hz=5.0):
        self.face_detect_interval = face_detect_interval
        self.pose_detect_hz       = pose_detect_hz
        self._pose_period_ms      = (1000.0 / pose_detect_hz
                                     if pose_detect_hz > 0 else float("inf"))
        self._last_pose_ms        = None

        self.hand_landmarker = vision.HandLandmarker.create_from_options(
            vision.HandLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=hand_task_path),
                running_mode=vision.RunningMode.VIDEO,
                num_hands=num_hands,
            )
        )

        self.face_landmarker = None
        if os.path.exists(face_task_path):
            self.face_landmarker = vision.FaceLandmarker.create_from_options(
                vision.FaceLandmarkerOptions(
                    base_options=BaseOptions(model_asset_path=face_task_path),
                    running_mode=vision.RunningMode.VIDEO,
                    num_faces=4,
                    # Lowered from 0.45 for long-range use: a face at 20+ ft
                    # is small and its detector confidence sits well below
                    # the close-range default. False positives this admits
                    # are filtered by the recognition gate (an unrecognised
                    # face is UNKNOWN and can't control anything).
                    min_face_detection_confidence=0.30,
                    min_face_presence_confidence=0.30,
                    min_tracking_confidence=0.30,
                )
            )
            print("Face landmarker: ready")
        else:
            print(f"WARNING: '{face_task_path}' not found — face recognition disabled.")

        self.pose_landmarker = None
        if os.path.exists(pose_task_path):
            self.pose_landmarker = vision.PoseLandmarker.create_from_options(
                vision.PoseLandmarkerOptions(
                    base_options=BaseOptions(model_asset_path=pose_task_path),
                    running_mode=vision.RunningMode.VIDEO,
                    num_poses=4,
                    # Lowered with the face thresholds: pose is the
                    # long-range fallback that keeps the PTZ tracking when
                    # the face flickers, so it must keep firing on a small,
                    # distant body. Identity safety comes from the PTZ's
                    # proximity gate, not from here.
                    min_pose_detection_confidence=0.30,
                    min_pose_presence_confidence=0.30,
                    min_tracking_confidence=0.30,
                )
            )
            print("Pose landmarker: ready")
        else:
            print("pose_landmarker.task not found — body skeleton disabled.")

        # Cached results, refreshed on their own interval
        self.cached_face_lms = []
        self.cached_pose_lms = []

    @staticmethod
    def _remap(landmark_lists, transform):
        """
        Rewrite normalized landmarks detected on a CROP of the frame into
        full-frame normalized coordinates, in place.

        transform = (off_x, off_y, scale): the crop's top-left corner and
        side length as fractions of the full frame (the crop keeps the
        frame's aspect ratio, so one scale serves both axes). z scales by
        the same factor so hand/face geometry stays uniform — the gesture
        and face-embedding features are scale-normalized, so a uniformly
        scaled shape classifies identically.
        """
        off_x, off_y, s = transform
        for lms in landmark_lists:
            for lm in lms:
                lm.x = off_x + lm.x * s
                lm.y = off_y + lm.y * s
                lm.z = lm.z * s

    def _pose_due(self, timestamp_ms):
        """
        True when POSE_DETECT_HZ says pose detection should run on this
        frame. The elapsed time is deliberately measured against the last
        frame pose actually RAN on, not against a fixed grid: falling
        behind must not queue up catch-up detections on the one component
        being rate-limited to keep the loop fast.

        A timestamp that goes backwards (a restarted clock) reschedules
        immediately rather than stalling until the old deadline.
        """
        if self._last_pose_ms is None:
            return True
        elapsed = timestamp_ms - self._last_pose_ms
        return elapsed >= self._pose_period_ms or elapsed < 0

    def detect(self, mp_image, frame_number, timestamp_ms, transform=None):
        """
        Runs hand detection every call; face detection every
        face_detect_interval frames and pose detection at pose_detect_hz
        (results are cached in between).

        transform: pass (off_x, off_y, scale) when mp_image is a crop of
        the full frame (the long-range "detector telephoto"): every FRESH
        result is remapped into full-frame normalized coordinates before
        being returned/cached, so downstream consumers (cursor mapping,
        PTZ error, HUD, embeddings) never know cropping happened. Cached
        face/pose results were remapped when they were produced — they are
        deliberately NOT remapped again.

        Returns:
            hand_result       — this frame's hand landmarker result
            cached_face_lms   — most recently detected face landmarks list
            cached_pose_lms   — most recently detected pose landmarks list
            face_updated      — True only on frames where face detection
                                 actually ran (used by AuthManager, which
                                 must not "see" the same frame twice)
        """
        hand_result = self.hand_landmarker.detect_for_video(mp_image, timestamp_ms)
        if transform is not None and hand_result.hand_landmarks:
            # Only the image-space landmarks — hand_world_landmarks are
            # metric and camera-independent, so they need no remapping.
            self._remap(hand_result.hand_landmarks, transform)

        face_updated = False
        if self.face_landmarker is not None and frame_number % self.face_detect_interval == 0:
            f_result = self.face_landmarker.detect_for_video(mp_image, timestamp_ms)
            if transform is not None and f_result.face_landmarks:
                self._remap(f_result.face_landmarks, transform)
            self.cached_face_lms = f_result.face_landmarks
            face_updated = True

        if self.pose_landmarker is not None and self._pose_due(timestamp_ms):
            self._last_pose_ms = timestamp_ms
            p_result = self.pose_landmarker.detect_for_video(mp_image, timestamp_ms)
            if transform is not None and p_result.pose_landmarks:
                self._remap(p_result.pose_landmarks, transform)
            self.cached_pose_lms = p_result.pose_landmarks

        return hand_result, self.cached_face_lms, self.cached_pose_lms, face_updated

    def close(self):
        self.hand_landmarker.close()
        if self.face_landmarker:
            self.face_landmarker.close()
        if self.pose_landmarker:
            self.pose_landmarker.close()
