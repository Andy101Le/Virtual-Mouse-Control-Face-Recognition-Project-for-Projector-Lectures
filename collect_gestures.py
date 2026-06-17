import cv2
import mediapipe as mp
import numpy as np
import csv
import os
import time
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions
from camera_utils import open_camera, read_frame, close_camera

GESTURES             = ["MOVE", "LEFT CLICK", "RIGHT CLICK", "ZOOM IN", "ZOOM OUT"]
CSV_FILE             = "landmark_dataset.csv"
NUM_SAMPLES_PER_GESTURE = 300

# Create CSV with header if it does not exist
if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["label"] + ["lm_%d" % i for i in range(63)])

options = vision.HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="hand_landmarker.task"),
    running_mode=vision.RunningMode.VIDEO,
    num_hands=1)
landmarker = vision.HandLandmarker.create_from_options(options)

cap = open_camera(width=640, height=480)

t0 = time.perf_counter()
current_gesture = 0
count = 0
landmarks = []

print("=== Gesture Data Collection ===")
print("SPACE = capture sample for current gesture")
print("N     = move to next gesture")
print("Q     = quit")
print("")
print("Gesture 1/%d: %s" % (len(GESTURES), GESTURES[0]))

while current_gesture < len(GESTURES):
    gesture_name = GESTURES[current_gesture]
    frame = read_frame(cap)
    if frame is None:
        break

    frame = cv2.flip(frame, 1)
    h, w  = frame.shape[:2]

    rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # BGR frame → RGB for MediaPipe
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    ts       = int((time.perf_counter() - t0) * 1000)
    results  = landmarker.detect_for_video(mp_image, ts)

    landmarks = []
    if results.hand_landmarks:
        hand = results.hand_landmarks[0]
        pts  = np.array([[lm.x, lm.y, lm.z] for lm in hand], dtype=np.float32)

        # Wrist-relative + scale normalize (matches main.py inference)
        pts  -= pts[0]
        scale = np.max(np.abs(pts)) or 1.0
        pts  /= scale
        landmarks = pts.flatten().tolist()

        # Draw
        sp = (np.array([[lm.x, lm.y] for lm in hand]) * [w, h]).astype(int)
        for pt in sp:
            cv2.circle(frame, tuple(pt), 4, (0, 255, 0), -1)
        connections = [(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),
                       (0,9),(9,10),(10,11),(11,12),(0,13),(13,14),(14,15),(15,16),
                       (0,17),(17,18),(18,19),(19,20)]
        for s, e in connections:
            cv2.line(frame, tuple(sp[s]), tuple(sp[e]), (255, 0, 0), 2)

    # HUD
    col = (0, 220, 0) if landmarks else (0, 0, 220)
    cv2.putText(frame,
                "Gesture %d/%d: %s  [%d/%d]" % (
                    current_gesture+1, len(GESTURES), gesture_name,
                    count, NUM_SAMPLES_PER_GESTURE),
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, col, 2)
    cv2.putText(frame,
                "READY - press SPACE" if landmarks else "No hand detected",
                (10, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.52, col, 2)
    cv2.putText(frame, "SPACE=capture  N=next  Q=quit",
                (10, h-12), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (120,120,120), 1)

    pw = int((count / NUM_SAMPLES_PER_GESTURE) * w)
    cv2.rectangle(frame, (0, h-8), (pw, h), (0, 200, 80), -1)

    if count >= NUM_SAMPLES_PER_GESTURE:
        cv2.putText(frame, "DONE! Press N for next gesture.",
                    (10, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 120), 2)

    cv2.imshow("Gesture Collection", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == 32 and len(landmarks) == 63 and count < NUM_SAMPLES_PER_GESTURE:
        with open(CSV_FILE, "a", newline="") as f:
            csv.writer(f).writerow([current_gesture] + landmarks)
        count += 1
        if count % 50 == 0:
            print("  %s: %d/%d" % (gesture_name, count, NUM_SAMPLES_PER_GESTURE))
        if count >= NUM_SAMPLES_PER_GESTURE:
            print("  %s complete. Press N." % gesture_name)

    elif key == ord("n"):
        if count >= NUM_SAMPLES_PER_GESTURE:
            current_gesture += 1
            count = 0
            if current_gesture < len(GESTURES):
                print("Gesture %d/%d: %s" % (
                      current_gesture+1, len(GESTURES), GESTURES[current_gesture]))
        else:
            print("  Need %d more samples." % (NUM_SAMPLES_PER_GESTURE - count))

    elif key == ord("q"):
        break

close_camera(cap)
cv2.destroyAllWindows()
landmarker.close()
print("Collection complete. Run train_gesture_model.py to train the model.")
