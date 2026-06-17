"""
face_register.py
────────────────
Register one or more users' faces for the gesture control system.

Uses MediaPipe FaceLandmarker to extract 478 3D facial landmarks per frame,
averages them across N samples, and saves a compact embedding to faces.pkl.

Usage:
    python face_register.py
    → Follow on-screen prompts, press SPACE to capture, Q to finish.

Output:
    faces.pkl  — dict mapping username → embedding array (shape: 478×3)
"""

import cv2
import time
import pickle
import os
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions
from camera_utils import open_camera, read_frame, close_camera

FACE_TASK_PATH  = "face_landmarker.task"   # download from MediaPipe model zoo
FACES_DB_PATH   = "faces.pkl"
SAMPLES_NEEDED  = 60    # frames averaged per registration
CAPTURE_DELAY   = 0.05  # seconds between auto-captures once SPACE is held

# ── Verify model file ─────────────────────────────────────────────────────────
if not os.path.exists(FACE_TASK_PATH):
    print(f"\n ERROR: '{FACE_TASK_PATH}' not found.")
    print("  Download it from:")
    print("  https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task")
    print("  Place it in the same folder as this script.\n")
    raise SystemExit(1)

# ── Load existing face DB ─────────────────────────────────────────────────────
if os.path.exists(FACES_DB_PATH):
    with open(FACES_DB_PATH, "rb") as f:
        face_db = pickle.load(f)
    print(f"Loaded existing face DB: {list(face_db.keys())}")
else:
    face_db = {}
    print("Starting new face DB.")

# ── MediaPipe FaceLandmarker ──────────────────────────────────────────────────
options = vision.FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=FACE_TASK_PATH),
    running_mode=vision.RunningMode.VIDEO,
    num_faces=1,
    min_face_detection_confidence=0.5,
    min_face_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)
landmarker = vision.FaceLandmarker.create_from_options(options)

# ── Haar cascade for drawing face box (lightweight, no extra model needed) ────
cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

cap = open_camera(width=640, height=480)
start_time = time.perf_counter()

def extract_embedding(face_landmarks):
    """
    Convert 478 MediaPipe face landmarks to a normalised embedding.
    Steps:
      1. Stack into (478, 3) array.
      2. Centre on nose tip (landmark 4) — position-invariant.
      3. Divide by max absolute value — scale-invariant.
    Returns: flat numpy array of shape (478*3,) = (1434,)
    """
    pts = np.array([[lm.x, lm.y, lm.z] for lm in face_landmarks], dtype=np.float32)
    pts -= pts[4]                              # centre on nose tip
    scale = np.max(np.abs(pts)) or 1.0
    pts  /= scale
    return pts.flatten()

print("\n=== Face Registration ===")
print("Commands:  SPACE = start/stop capturing  |  N = new user  |  D = delete user  |  Q = quit\n")

username       = input("Enter username to register (or press Enter to list existing): ").strip()
if not username:
    print("Existing users:", list(face_db.keys()) or ["none"])
    username = input("Enter username to register: ").strip()

if not username:
    print("No username entered. Exiting.")
    raise SystemExit(0)

sample_embeddings = []
capturing         = False
last_capture_t    = 0.0

print(f"\nRegistering: '{username}'")
print("Look at the camera, then hold SPACE to capture 60 frames.")
print("Try turning your head slightly left/right/up/down for variety.\n")

while True:
    rgb_raw = read_frame(cap)
    frame   = rgb_raw
    success = frame is not None
    if not success:
        break

    frame   = cv2.flip(frame, 1)           # mirror horizontally
    h, w    = frame.shape[:2]
    rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)   # BGR → RGB for MediaPipe
    mp_img  = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    ts_ms   = int((time.perf_counter() - start_time) * 1000)
    results = landmarker.detect_for_video(mp_img, ts_ms)

    face_detected = bool(results.face_landmarks)
    n_collected   = len(sample_embeddings)
    done          = n_collected >= SAMPLES_NEEDED

    # ── Draw HUD ─────────────────────────────────────────────────────────────
    cv2.rectangle(frame, (0, 0), (w, 75), (20, 20, 20), -1)
    cv2.putText(frame, f"User: {username}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    status_txt = (f"CAPTURED {n_collected}/{SAMPLES_NEEDED}" if capturing
                  else ("DONE — press Q to save" if done
                        else "Hold SPACE to capture"))
    status_col = (0, 220, 0) if face_detected else (0, 0, 220)
    cv2.putText(frame, status_txt, (10, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_col, 2)

    # Progress bar
    bar_w = int((n_collected / SAMPLES_NEEDED) * w)
    cv2.rectangle(frame, (0, h - 10), (bar_w, h), (0, 200, 80), -1)

    # Draw face landmarks if detected
    if results.face_landmarks:
        face = results.face_landmarks[0]
        # Draw a few key points (nose, eyes, mouth corners) — not all 478
        key_idxs = [4, 33, 263, 61, 291, 199]   # nose, L-eye, R-eye, mouth corners, chin
        for idx in key_idxs:
            lm  = face[idx]
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)

        # Face bounding box from landmark extremes
        xs = [int(lm.x * w) for lm in face]
        ys = [int(lm.y * h) for lm in face]
        x1, x2 = max(min(xs) - 10, 0), min(max(xs) + 10, w)
        y1, y2 = max(min(ys) - 10, 0), min(max(ys) + 10, h)
        box_col = (0, 255, 0) if (capturing and not done) else (200, 200, 200)
        cv2.rectangle(frame, (x1, y1), (x2, y2), box_col, 2)

    cv2.imshow("Face Registration", frame)
    key = cv2.waitKey(1) & 0xFF
    now = time.perf_counter()

    # SPACE held → auto-capture frames
    if key == 32 and face_detected and not done:
        capturing = True

    if capturing and face_detected and not done and (now - last_capture_t) >= CAPTURE_DELAY:
        emb = extract_embedding(results.face_landmarks[0])
        sample_embeddings.append(emb)
        last_capture_t = now
        if len(sample_embeddings) >= SAMPLES_NEEDED:
            capturing = False
            print(f"  Captured {SAMPLES_NEEDED} samples — press Q to save or keep holding SPACE for more.")

    if key == ord('q') or key == 27:
        break

# ── Save ─────────────────────────────────────────────────────────────────────
if sample_embeddings:
    # Average all captured embeddings → single robust template
    mean_emb = np.mean(sample_embeddings, axis=0)
    mean_emb /= (np.linalg.norm(mean_emb) or 1.0)   # L2-normalise for cosine sim
    face_db[username] = mean_emb
    with open(FACES_DB_PATH, "wb") as f:
        pickle.dump(face_db, f)
    print(f"\nSaved '{username}' to {FACES_DB_PATH}.")
    print(f"Registered users: {list(face_db.keys())}")
else:
    print("\nNo samples collected — nothing saved.")

close_camera(cap)
cv2.destroyAllWindows()
landmarker.close()
