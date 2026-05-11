# ##FIRST WORKING VERSION##
# # collect_landmarks.py
# import cv2
# import mediapipe as mp
# import numpy as np
# import csv
# import os
# from mediapipe.tasks.python import vision
# from mediapipe.tasks.python.core.base_options import BaseOptions
#
# GESTURES = ["MOVE", "LEFT CLICK", "RIGHT CLICK", "ZOOM IN", "ZOOM OUT"]
# CSV_FILE = "landmark_dataset.csv"
# NUM_SAMPLES_PER_GESTURE = 300
#
# # Create CSV with header if it doesn't exist
# if not os.path.exists(CSV_FILE):
#     with open(CSV_FILE, 'w', newline='') as f:
#         writer = csv.writer(f)
#         header = ['label'] + [f'lm_{i}' for i in range(63)]
#         writer.writerow(header)
#
# # MediaPipe setup
# options = vision.HandLandmarkerOptions(
#     base_options=BaseOptions(model_asset_path="hand_landmarker.task"),
#     running_mode=vision.RunningMode.VIDEO,
#     num_hands=1
# )
# landmarker = vision.HandLandmarker.create_from_options(options)
#
# cap = cv2.VideoCapture(0)
# current_gesture = 0
# count = 0
#
# print("=== Landmark Collection ===")
# print("Press SPACE to capture samples for current gesture")
# print("Press N to go to next gesture")
# print("Press Q to quit\n")
#
# while current_gesture < len(GESTURES):
#     gesture_name = GESTURES[current_gesture]
#     print(f"Current gesture: {gesture_name} ({count}/{NUM_SAMPLES_PER_GESTURE})")
#
#     success, frame = cap.read()
#     if not success:
#         break
#
#     rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
#     timestamp = int(cv2.getTickCount() / cv2.getTickFrequency() * 1000)
#     results = landmarker.detect_for_video(mp_image, timestamp)
#
#     h, w = frame.shape[:2]
#
#     if results.hand_landmarks:
#         hand = results.hand_landmarks[0]
#         landmarks = []
#         for lm in hand:
#             landmarks.extend([lm.x, lm.y, lm.z])
#
#         # Draw
#         for i, lm in enumerate(hand):
#             cx, cy = int(lm.x * w), int(lm.y * h)
#             cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)
#
#         cv2.putText(frame, f"{gesture_name} {count}/{NUM_SAMPLES_PER_GESTURE}", (10, 30),
#                     cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
#
#     cv2.imshow("Collect Landmarks", frame)
#     key = cv2.waitKey(1) & 0xFF
#
#     if key == 32 and len(landmarks) == 63:  # SPACE = save sample
#         with open(CSV_FILE, 'a', newline='') as f:
#             writer = csv.writer(f)
#             writer.writerow([current_gesture] + landmarks)
#         count += 1
#         print(f"  Saved sample {count}")
#
#     elif key == ord('n'):  # Next gesture
#         if count > 0:
#             print(f"Finished {gesture_name} with {count} samples")
#             current_gesture += 1
#             count = 0
#         else:
#             print("Collect at least a few samples before skipping!")
#
#     elif key == ord('q'):
#         break
#
# cap.release()
# cv2.destroyAllWindows()
# landmarker.close()
# print("Collection finished!")






# collect_gestures.py
# Saves wrist-relative + scale-normalized landmark vectors.
# This MUST match the normalization applied in main.py so training
# and inference are consistent.

import cv2
import mediapipe as mp
import numpy as np
import csv
import os
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions

GESTURES = ['MOVE', 'LEFT CLICK', 'RIGHT CLICK', 'ZOOM IN', 'ZOOM OUT']
CSV_FILE = "landmark_dataset.csv"
NUM_SAMPLES_PER_GESTURE = 300

# ── Gesture descriptions shown on screen ──────────────────────────────────────
GESTURE_HINTS = {
    'MOVE':        'Point with INDEX finger, others curled',
    'LEFT CLICK':  'FIST — all fingers closed',
    'RIGHT CLICK': 'THUMBS DOWN — fist with thumb pointing down',
    'ZOOM IN':     'ROCK SIGN — index + pinky extended',
    'ZOOM OUT':    'PEACE — index + middle finger extended (V)',
}

if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        header = ['label'] + [f'lm_{i}' for i in range(63)]
        writer.writerow(header)
    print(f"Created new {CSV_FILE}")
else:
    # Count existing samples per class
    existing = np.loadtxt(CSV_FILE, delimiter=',', skiprows=1)
    if existing.ndim == 1:
        existing = existing.reshape(1, -1)
    print(f"Appending to existing {CSV_FILE}  ({len(existing)} samples already)")

# ── MediaPipe setup ───────────────────────────────────────────────────────────
options = vision.HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="hand_landmarker.task"),
    running_mode=vision.RunningMode.VIDEO,
    num_hands=1
)
landmarker = vision.HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)
current_gesture = 0
count = 0
landmarks = []

print("\n=== Landmark Collection ===")
print("Normalization: wrist-relative + scale  (matches main.py exactly)")
print("SPACE = capture sample | N = next gesture | Q = quit\n")

while current_gesture < len(GESTURES):
    gesture_name = GESTURES[current_gesture]
    hint         = GESTURE_HINTS.get(gesture_name, '')

    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)  # mirror so it feels natural
    h, w  = frame.shape[:2]

    rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    timestamp = int(cv2.getTickCount() / cv2.getTickFrequency() * 1000)
    results   = landmarker.detect_for_video(mp_image, timestamp)

    landmarks = []

    if results.hand_landmarks:
        hand = results.hand_landmarks[0]
        raw  = np.array([[lm.x, lm.y, lm.z] for lm in hand])  # (21, 3)

        # ── Normalize: wrist-relative + scale-invariant ───────────────────────
        wrist    = raw[0]
        centered = raw - wrist
        scale    = np.max(np.abs(centered)) or 1.0
        norm     = centered / scale                             # (21, 3) in [-1, 1]
        landmarks = norm.flatten().tolist()                     # 63 floats

        # Draw skeleton
        CONNECTIONS = [
            (0,1),(1,2),(2,3),(3,4),
            (0,5),(5,6),(6,7),(7,8),
            (0,9),(9,10),(10,11),(11,12),
            (0,13),(13,14),(14,15),(15,16),
            (0,17),(17,18),(18,19),(19,20)
        ]
        pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand]
        for s, e in CONNECTIONS:
            cv2.line(frame, pts[s], pts[e], (255, 0, 0), 2)
        for p in pts:
            cv2.circle(frame, p, 4, (0, 255, 0), -1)

    # ── HUD ───────────────────────────────────────────────────────────────────
    ready  = len(landmarks) == 63
    status = "READY — press SPACE" if ready else "NO HAND DETECTED"
    color  = (0, 220, 0) if ready else (0, 0, 220)

    cv2.rectangle(frame, (0, 0), (w, 90), (30, 30, 30), -1)
    cv2.putText(frame, f"Gesture {current_gesture+1}/{len(GESTURES)}: {gesture_name}",
                (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(frame, hint,
                (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
    cv2.putText(frame, f"{status}   [{count}/{NUM_SAMPLES_PER_GESTURE}]",
                (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

    progress_w = int((count / NUM_SAMPLES_PER_GESTURE) * w)
    cv2.rectangle(frame, (0, h - 8), (progress_w, h), (0, 200, 80), -1)

    cv2.imshow("Collect Landmarks", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == 32 and ready:   # SPACE
        with open(CSV_FILE, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([current_gesture] + landmarks)
        count += 1
        if count % 50 == 0:
            print(f"  [{gesture_name}] {count}/{NUM_SAMPLES_PER_GESTURE}")
        if count >= NUM_SAMPLES_PER_GESTURE:
            print(f"  [{gesture_name}] Done! Press N to continue.\n")

    elif key == ord('n'):
        if count >= NUM_SAMPLES_PER_GESTURE:
            current_gesture += 1
            count = 0
        else:
            print(f"  Need {NUM_SAMPLES_PER_GESTURE - count} more samples before continuing.")

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
landmarker.close()
print("Collection complete!")
