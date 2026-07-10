"""
face_capture.py
────────────────
Captures and averages face-landmark samples into a single embedding,
used both when a user registers their own face and when an admin
enrolls someone else's.

Uses CameraManager (picamera2 on the Pi, cv2.VideoCapture fallback on a
laptop) so this works unmodified on both platforms — this is the same
camera-selection fix applied to main.py.
"""

import os
import time
import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions
from tkinter import messagebox

from camera_manager import CameraManager


class FaceCapture:
    SAMPLES_NEEDED = 60
    CAPTURE_DELAY  = 0.05

    def __init__(self, face_task_path):
        self.face_task_path = face_task_path

    @staticmethod
    def extract_embedding(face_landmarks):
        """Same normalisation used everywhere in the project — must match exactly."""
        pts = np.array([[lm.x, lm.y, lm.z] for lm in face_landmarks], dtype=np.float32)
        pts -= pts[4]                            # centre on nose tip
        scale = np.max(np.abs(pts)) or 1.0
        pts  /= scale
        flat  = pts.flatten()
        norm  = np.linalg.norm(flat) or 1.0
        return flat / norm                       # L2-normalised for cosine similarity

    def capture_for_user(self, target_username):
        """
        Opens a CameraManager + OpenCV window, captures SAMPLES_NEEDED
        face samples while SPACE is held, averages them into one
        embedding.

        Returns the embedding (np.float32 array) on success, or None if
        cancelled / nothing was captured. Saving to the database is the
        caller's responsibility (see UserDatabase.save_face_embedding) —
        this class only knows how to capture, not how to persist.
        """
        if not os.path.exists(self.face_task_path):
            messagebox.showerror(
                "Missing model",
                f"'{self.face_task_path}' not found. Run download_models.py first."
            )
            return None

        options = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=self.face_task_path),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        landmarker = vision.FaceLandmarker.create_from_options(options)
        cam        = CameraManager(width=640, height=480, fps=30)

        start_time        = time.perf_counter()
        sample_embeddings = []
        capturing         = False
        last_capture_t     = 0.0

        print(f"\n=== Capturing face for '{target_username}' ===")
        print("Hold SPACE to capture 60 frames.  Q to cancel.")

        try:
            while cam.is_opened():
                success, frame = cam.read()
                if not success:
                    break

                frame  = np.ascontiguousarray(frame[:, ::-1, :])    # mirror
                h, w   = frame.shape[:2]
                rgb    = np.ascontiguousarray(frame[:, :, ::-1])
                mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                ts_ms  = int((time.perf_counter() - start_time) * 1000)
                result = landmarker.detect_for_video(mp_img, ts_ms)

                face_detected = bool(result.face_landmarks)
                n_collected   = len(sample_embeddings)
                done          = n_collected >= self.SAMPLES_NEEDED

                self._draw_hud(frame, target_username, n_collected, done,
                               capturing, face_detected, w, h)
                if result.face_landmarks:
                    self._draw_face_overlay(frame, result.face_landmarks[0], capturing, done, w, h)

                cv2.imshow(f"Face Registration - {target_username}", frame)
                key = cv2.waitKey(1) & 0xFF
                now = time.perf_counter()

                if key == 32 and face_detected and not done:
                    capturing = True

                if capturing and face_detected and not done and (now - last_capture_t) >= self.CAPTURE_DELAY:
                    emb = self.extract_embedding(result.face_landmarks[0])
                    sample_embeddings.append(emb)
                    last_capture_t = now
                    if len(sample_embeddings) >= self.SAMPLES_NEEDED:
                        capturing = False
                        print(f"  Captured {self.SAMPLES_NEEDED} samples - press Q to save.")

                if key == ord('q') or key == 27:
                    break
        finally:
            cam.release()
            cv2.destroyAllWindows()
            landmarker.close()

        if not sample_embeddings:
            print("No samples collected - nothing saved.")
            return None

        mean_emb  = np.mean(sample_embeddings, axis=0)
        mean_emb /= (np.linalg.norm(mean_emb) or 1.0)   # L2-normalise for cosine sim
        print(f"Captured embedding for '{target_username}'.")
        return mean_emb

    # ── Drawing helpers ────────────────────────────────────────────────────
    @staticmethod
    def _draw_hud(frame, target_username, n_collected, done, capturing, face_detected, w, h):
        cv2.rectangle(frame, (0, 0), (w, 75), (20, 20, 20), -1)
        cv2.putText(frame, f"User: {target_username}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        if done:
            status_txt = "DONE - press Q to save"
        elif capturing:
            status_txt = f"CAPTURED {n_collected}/{FaceCapture.SAMPLES_NEEDED}"
        else:
            status_txt = "Hold SPACE to capture"
        status_col = (0, 220, 0) if face_detected else (0, 0, 220)
        cv2.putText(frame, status_txt, (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_col, 2)

        bar_w = int((n_collected / FaceCapture.SAMPLES_NEEDED) * w)
        cv2.rectangle(frame, (0, h - 10), (bar_w, h), (0, 200, 80), -1)

    @staticmethod
    def _draw_face_overlay(frame, face, capturing, done, w, h):
        for idx in [4, 33, 263, 61, 291, 199]:
            lm = face[idx]
            cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 4, (0, 255, 0), -1)
        xs = [int(lm.x * w) for lm in face]
        ys = [int(lm.y * h) for lm in face]
        x1, x2 = max(min(xs) - 10, 0), min(max(xs) + 10, w)
        y1, y2 = max(min(ys) - 10, 0), min(max(ys) + 10, h)
        box_col = (0, 255, 0) if (capturing and not done) else (200, 200, 200)
        cv2.rectangle(frame, (x1, y1), (x2, y2), box_col, 2)
