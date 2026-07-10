"""
landmark_pipeline.py
─────────────────────
Wraps the three MediaPipe VIDEO-mode landmarkers (hand / face / pose)
used by main.py, and owns the "run every N frames, cache the rest"
scheduling so main.py's loop doesn't have to.

Hand detection is cheap and responsive enough to run every frame.
Face and pose move more slowly, so their results are only refreshed
every FACE_DETECT_INTERVAL / POSE_DETECT_INTERVAL frames and cached
in between.
"""

import os
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions


class LandmarkPipeline:
    def __init__(self,
                 hand_task_path, face_task_path, pose_task_path,
                 num_hands=2, face_detect_interval=3, pose_detect_interval=2):
        self.face_detect_interval = face_detect_interval
        self.pose_detect_interval = pose_detect_interval

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
                    min_face_detection_confidence=0.45,
                    min_face_presence_confidence=0.45,
                    min_tracking_confidence=0.45,
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
                    min_pose_detection_confidence=0.45,
                    min_pose_presence_confidence=0.45,
                    min_tracking_confidence=0.45,
                )
            )
            print("Pose landmarker: ready")
        else:
            print("pose_landmarker.task not found — body skeleton disabled.")

        # Cached results, refreshed on their own interval
        self.cached_face_lms = []
        self.cached_pose_lms = []

    def detect(self, mp_image, frame_number, timestamp_ms):
        """
        Runs hand detection every call, and face/pose detection only on
        their own interval (results are cached in between).

        Returns:
            hand_result       — this frame's hand landmarker result
            cached_face_lms   — most recently detected face landmarks list
            cached_pose_lms   — most recently detected pose landmarks list
            face_updated      — True only on frames where face detection
                                 actually ran (used by AuthManager, which
                                 must not "see" the same frame twice)
        """
        hand_result = self.hand_landmarker.detect_for_video(mp_image, timestamp_ms)

        face_updated = False
        if self.face_landmarker is not None and frame_number % self.face_detect_interval == 0:
            f_result = self.face_landmarker.detect_for_video(mp_image, timestamp_ms)
            self.cached_face_lms = f_result.face_landmarks
            face_updated = True

        if self.pose_landmarker is not None and frame_number % self.pose_detect_interval == 0:
            p_result = self.pose_landmarker.detect_for_video(mp_image, timestamp_ms)
            self.cached_pose_lms = p_result.pose_landmarks

        return hand_result, self.cached_face_lms, self.cached_pose_lms, face_updated

    def close(self):
        self.hand_landmarker.close()
        if self.face_landmarker:
            self.face_landmarker.close()
        if self.pose_landmarker:
            self.pose_landmarker.close()
