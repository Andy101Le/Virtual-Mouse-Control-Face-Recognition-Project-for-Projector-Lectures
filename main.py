# import cv2
# import time
# import mediapipe as mp
# from mediapipe.tasks.python import vision
# from mediapipe.tasks.python.core.base_options import BaseOptions
# import numpy as np
# from tensorflow.keras.models import load_model
#
# MODEL_PATH = "hand_landmarker.task"
# GESTURE_MODEL_PATH = "gesture_model.h5" # trained CNN model
#
# # ----------------------------
# # Config
# # ----------------------------
# prev_time = time.time()
# fps = 0.0
#
# NUM_HANDS = 2
# # Official MediaPipe hand skeleton connections
# HAND_CONNECTIONS = [
#     # Thumb
#     (0, 1), (1, 2), (2, 3), (3, 4),
#
#     # Index finger
#     (0, 5), (5, 6), (6, 7), (7, 8),
#
#     # Middle finger
#     (0, 9), (9, 10), (10, 11), (11, 12),
#
#     # Ring finger
#     (0, 13), (13, 14), (14, 15), (15, 16),
#
#     # Pinky
#     (0, 17), (17, 18), (18, 19), (19, 20)
# ]
#
# SMOOTHING_ALPHA = 0.3
# DEBOUNCE_FRAMES = 5
#
# # ----------------------------
# # Load gesture model
# # ----------------------------
# gesture_model = load_model(GESTURE_MODEL_PATH)
# GESTURE_CLASSES = ['OPEN', 'FIST', 'PEACE', 'THUMBS_DOWN', 'ROCK_SIGN']
#
# # ----------------------------
# # Landmarker
# # ----------------------------
# options = vision.HandLandmarkerOptions(
#     base_options=BaseOptions(model_asset_path=MODEL_PATH),
#     running_mode=vision.RunningMode.VIDEO,
#     num_hands=NUM_HANDS
# )
#
# landmarker = vision.HandLandmarker.create_from_options(options)
#
# # ----------------------------
# # Webcam
# # ----------------------------
# cap = cv2.VideoCapture(0)
# start_time = time.time()
#
# # State storage
# smoothed_landmarks = {}
# gesture_counters = {}
#
# while cap.isOpened():
#     success, frame = cap.read()
#     if not success:
#         break
#
#     current_time = time.time()
#     dt = current_time - prev_time
#     prev_time = current_time
#
#     if dt > 0:
#         fps = 0.9 * fps + 0.1 * (1.0 / dt)  # smoothing
#     rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
#
#     timestamp_ms = int((time.time() - start_time) * 1000)
#
#     results = landmarker.detect_for_video(mp_image, timestamp_ms)
#
#     h, w, _ = frame.shape
#
#     if results.hand_landmarks:
#         for hand_id, hand in enumerate(results.hand_landmarks):
#             # ----------------------------
#             # Smoothing landmarks
#             # ----------------------------
#             if hand_id not in smoothed_landmarks:
#                 smoothed_landmarks[hand_id] = [(lm.x, lm.y) for lm in hand]
#
#             smoothed_points = []
#             for i, lm in enumerate(hand):
#                 prev_x, prev_y = smoothed_landmarks[hand_id][i]
#                 new_x = SMOOTHING_ALPHA * lm.x + (1 - SMOOTHING_ALPHA) * prev_x
#                 new_y = SMOOTHING_ALPHA * lm.y + (1 - SMOOTHING_ALPHA) * prev_y
#                 smoothed_landmarks[hand_id][i] = (new_x, new_y)
#                 smoothed_points.append((int(new_x * w), int(new_y * h)))
#
#             # Draw landmarks
#             for x, y in smoothed_points:
#                 cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)
#
#             # Draw bones
#             for start, end in HAND_CONNECTIONS:
#                 cv2.line(frame, smoothed_points[start], smoothed_points[end], (255, 0, 0), 2)
#
#             # ----------------------------
#             # Crop hand for CNN
#             # ----------------------------
#             x_coords = [p[0] for p in smoothed_points]
#             y_coords = [p[1] for p in smoothed_points]
#             x_min, x_max = max(min(x_coords) - 10, 0), min(max(x_coords) + 10, w)
#             y_min, y_max = max(min(y_coords) - 10, 0), min(max(y_coords) + 10, h)
#             hand_crop = frame[y_min:y_max, x_min:x_max]
#
#             if hand_crop.size != 0:
#                 hand_input = cv2.resize(hand_crop, (128, 128))
#                 hand_input = hand_input.astype('float32') / 255.0
#                 hand_input = np.expand_dims(hand_input, axis=0)
#
#                 pred = gesture_model.predict(hand_input, verbose=0)
#                 gesture_idx = np.argmax(pred)
#                 gesture_name = GESTURE_CLASSES[gesture_idx]
#
#                 # ----------------------------
#                 # Debouncing
#                 # ----------------------------
#                 if hand_id not in gesture_counters:
#                     gesture_counters[hand_id] = {'name': gesture_name, 'count': 0}
#
#                 if gesture_counters[hand_id]['name'] == gesture_name:
#                     gesture_counters[hand_id]['count'] += 1
#                 else:
#                     gesture_counters[hand_id]['name'] = gesture_name
#                     gesture_counters[hand_id]['count'] = 1
#
#                 if gesture_counters[hand_id]['count'] >= DEBOUNCE_FRAMES:
#                     display_gesture = gesture_name
#                 else:
#                     display_gesture = '...'
#
#                 cv2.putText(frame,
#                             f"Hand {hand_id}: {display_gesture}",
#                             (10, 50 + hand_id * 20),
#                             cv2.FONT_HERSHEY_SIMPLEX,
#                             0.6, (0, 0, 255), 2, cv2.LINE_AA)
#
#     # FPS overlay
#     cv2.putText(frame,
#                 f"FPS: {fps:.1f}",
#                 (10, 30),
#                 cv2.FONT_ITALIC,
#                 0.46, (100, 255, 100), 2, cv2.LINE_AA)
#
#     cv2.imshow("Hand Tracking + Multi-Gesture CNN", frame)
#     if cv2.waitKey(1) & 0xFF == ord("q"):
#         break
#
# cap.release()
# cv2.destroyAllWindows()
# landmarker.close()



#
# import cv2
# import time
# import mediapipe as mp
# import numpy as np
# from tensorflow.keras.models import load_model
# from mediapipe.tasks.python import vision
# from mediapipe.tasks.python.core.base_options import BaseOptions
#
# # ----------------------------
# # CONFIG
# # ----------------------------
# MODEL_PATH = "hand_landmarker.task"
# GESTURE_MODEL_PATH = "gesture_model.h5"
# NUM_HANDS = 2
# IMAGE_SIZE = 128
#
# # Official hand connections
# HAND_CONNECTIONS = [
#     (0, 1), (1, 2), (2, 3), (3, 4),      # Thumb
#     (0, 5), (5, 6), (6, 7), (7, 8),      # Index
#     (0, 9), (9, 10), (10, 11), (11, 12), # Middle
#     (0, 13), (13, 14), (14, 15), (15, 16), # Ring
#     (0, 17), (17, 18), (18, 19), (19, 20)  # Pinky
# ]
#
# # ----------------------------
# # Load gesture model
# # ----------------------------
# gesture_model = load_model(GESTURE_MODEL_PATH)
# gesture_classes = {v: k for k, v in gesture_model.class_indices.items()} if hasattr(gesture_model, 'class_indices') else {
#     0: "OPEN", 1: "FIST", 2: "PEACE", 3: "THUMBS_DOWN", 4: "ROCK_SIGN"
# }
#
# # ----------------------------
# # MediaPipe HandLandmarker
# # ----------------------------
# options = vision.HandLandmarkerOptions(
#     base_options=BaseOptions(model_asset_path=MODEL_PATH),
#     running_mode=vision.RunningMode.VIDEO,
#     num_hands=NUM_HANDS
# )
# landmarker = vision.HandLandmarker.create_from_options(options)
#
# # ----------------------------
# # Webcam + FPS
# # ----------------------------
# cap = cv2.VideoCapture(0)
# prev_time = time.time()
# fps = 0.0
# start_time = time.time()
#
# while cap.isOpened():
#     success, frame = cap.read()
#     if not success:
#         break
#
#     current_time = time.time()
#     dt = current_time - prev_time
#     prev_time = current_time
#     if dt > 0:
#         fps = 0.9 * fps + 0.1 * (1.0 / dt)  # smoothed FPS
#
#     rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
#     timestamp_ms = int((time.time() - start_time) * 1000)
#     results = landmarker.detect_for_video(mp_image, timestamp_ms)
#
#     h, w, _ = frame.shape
#
#     # ----------------------------
#     # Process each detected hand
#     # ----------------------------
#     if results.hand_landmarks:
#         for hand_id, hand in enumerate(results.hand_landmarks):
#             # Extract landmark points
#             points = []
#             x_coords = []
#             y_coords = []
#             for lm in hand:
#                 cx, cy = int(lm.x * w), int(lm.y * h)
#                 points.append((cx, cy))
#                 x_coords.append(cx)
#                 y_coords.append(cy)
#                 cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)
#
#             # Draw bones
#             for start, end in HAND_CONNECTIONS:
#                 cv2.line(frame, points[start], points[end], (255, 0, 0), 2)
#
#             # Crop hand for gesture recognition
#             x_min, x_max = max(min(x_coords)-10,0), min(max(x_coords)+10,w)
#             y_min, y_max = max(min(y_coords)-10,0), min(max(y_coords)+10,h)
#             hand_crop = frame[y_min:y_max, x_min:x_max]
#             if hand_crop.size != 0:
#                 hand_resized = cv2.resize(hand_crop, (IMAGE_SIZE, IMAGE_SIZE))
#                 hand_input = np.expand_dims(hand_resized/255.0, axis=0)
#                 pred = gesture_model.predict(hand_input)
#                 gesture_id = np.argmax(pred)
#                 gesture_name = gesture_classes.get(gesture_id, "UNKNOWN")
#
#                 # Display gesture above hand
#                 cv2.putText(frame, f"Hand {hand_id+1}: {gesture_name}",
#                             (x_min, y_min-10),
#                             cv2.FONT_HERSHEY_SIMPLEX,
#                             0.7,
#                             (0, 0, 255),
#                             2,
#                             cv2.LINE_AA)
#
#     # ----------------------------
#     # Display FPS
#     # ----------------------------
#     cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 255, 100), 2, cv2.LINE_AA)
#
#     cv2.imshow("Hand Tracking + Gesture Recognition", frame)
#     if cv2.waitKey(1) & 0xFF == ord("q"):
#         break
#
# cap.release()
# cv2.destroyAllWindows()
# landmarker.close()

# import os
# # Disable TensorFlow oneDNN warning
# os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
#
# import cv2
# import numpy as np
# import tensorflow as tf
# from tensorflow.keras.models import load_model
#
# # =========================
# # Gesture classes
# # =========================
# GESTURES = ["OPEN", "FIST", "PEACE", "THUMBS_DOWN", "ROCK_SIGN"]
#
# # =========================
# # Load trained model
# # =========================
# gesture_model = load_model("landmark_gesture_model.h5")
#
# # =========================
# # MediaPipe imports
# # =========================
# from mediapipe.tasks.python import vision
# from mediapipe.tasks.python.vision import hand_landmarker
#
# # Load the hand landmarker model
# base_options = hand_landmarker.HandLandmarkerModelOptions(
#     model_asset_path='hand_landmarker.task'  # <-- download the latest MediaPipe task model
# )
# landmarker = hand_landmarker.HandLandmarker.create_from_options(base_options)
#
# # =========================
# # Open webcam
# # =========================
# cap = cv2.VideoCapture(0)
#
# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break
#
#     # Convert BGR to RGB
#     rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#
#     # MediaPipe expects RGB
#     results = landmarker.detect(rgb)
#
#     if results.hands:
#         for hand in results.hands:
#             # Collect 21 landmarks (x, y, z)
#             wrist = hand.hand_landmarks[0]
#             landmarks = []
#             for lm in hand.hand_landmarks:
#                 # relative to wrist
#                 landmarks.extend([lm.x - wrist.x, lm.y - wrist.y, lm.z - wrist.z])
#
#             landmarks_input = np.array(landmarks, dtype=np.float32).reshape(1, -1)
#
#             # Predict gesture
#             pred_probs = gesture_model.predict(landmarks_input, verbose=0)
#             pred_idx = np.argmax(pred_probs)
#             gesture_name = GESTURES[pred_idx]
#
#             # Draw landmarks
#             for lm in hand.hand_landmarks:
#                 cx, cy = int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0])
#                 cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)
#
#             # Display gesture name
#             cv2.putText(frame, gesture_name, (50, 50),
#                         cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
#
#     cv2.imshow("Gesture Recognition", frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
# cap.release()
# cv2.destroyAllWindows()








# ##MOST CURRENT WORKING VERSION 4/2/2026##
# import cv2
# import time
# import mediapipe as mp
# from mediapipe.tasks.python import vision
# from mediapipe.tasks.python.core.base_options import BaseOptions
# import numpy as np
# from tensorflow.keras.models import load_model
#
# MODEL_PATH = "hand_landmarker.task"
# GESTURE_MODEL_PATH = "landmark_gesture_model.h5"
#
# # ----------------------------
# # Config
# # ----------------------------
# prev_time = time.time()
# fps = 0.0
#
# NUM_HANDS = 2
#
# HAND_CONNECTIONS = [
#     (0, 1), (1, 2), (2, 3), (3, 4),
#     (0, 5), (5, 6), (6, 7), (7, 8),
#     (0, 9), (9, 10), (10, 11), (11, 12),
#     (0, 13), (13, 14), (14, 15), (15, 16),
#     (0, 17), (17, 18), (18, 19), (19, 20)
# ]
#
# SMOOTHING_ALPHA = 0.3
# DEBOUNCE_FRAMES = 5
# CONFIDENCE_THRESHOLD = 0.6  # below this → orange label with ~ prefix
#
# # ----------------------------
# # Load gesture model
# # ----------------------------
# gesture_model = load_model(GESTURE_MODEL_PATH)
# GESTURE_CLASSES = ['MOVE', 'LEFT CLICK', 'RIGHT CLICK', 'ZOOM IN', 'ZOOM OUT']
#
# # ----------------------------
# # Landmarker
# # ----------------------------
# options = vision.HandLandmarkerOptions(
#     base_options=BaseOptions(model_asset_path=MODEL_PATH),
#     running_mode=vision.RunningMode.VIDEO,
#     num_hands=NUM_HANDS
# )
# landmarker = vision.HandLandmarker.create_from_options(options)
#
# # ----------------------------
# # Webcam
# # ----------------------------
# cap = cv2.VideoCapture(0)
# start_time = time.time()
#
# smoothed_xyz = {}
# gesture_counters = {}
#
# while cap.isOpened():
#     success, frame = cap.read()
#     if not success:
#         break
#
#     current_time = time.time()
#     dt = current_time - prev_time
#     prev_time = current_time
#     if dt > 0:
#         fps = 0.9 * fps + 0.1 * (1.0 / dt)
#
#     rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
#     timestamp_ms = int((time.time() - start_time) * 1000)
#     results = landmarker.detect_for_video(mp_image, timestamp_ms)
#
#     h, w, _ = frame.shape
#
#     if results.hand_landmarks:
#         for hand_id, hand in enumerate(results.hand_landmarks):
#
#             # ----------------------------
#             # Smooth landmarks
#             # ----------------------------
#             if hand_id not in smoothed_xyz:
#                 smoothed_xyz[hand_id] = [(lm.x, lm.y, lm.z) for lm in hand]
#
#             smoothed_points = []
#             new_xyz = []
#
#             for i, lm in enumerate(hand):
#                 px, py, pz = smoothed_xyz[hand_id][i]
#                 sx = SMOOTHING_ALPHA * lm.x + (1 - SMOOTHING_ALPHA) * px
#                 sy = SMOOTHING_ALPHA * lm.y + (1 - SMOOTHING_ALPHA) * py
#                 sz = SMOOTHING_ALPHA * lm.z + (1 - SMOOTHING_ALPHA) * pz
#                 smoothed_xyz[hand_id][i] = (sx, sy, sz)
#                 smoothed_points.append((int(sx * w), int(sy * h)))
#                 new_xyz.append((sx, sy, sz))
#
#             # Draw
#             for x, y in smoothed_points:
#                 cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)
#             for start, end in HAND_CONNECTIONS:
#                 cv2.line(frame, smoothed_points[start], smoothed_points[end], (255, 0, 0), 2)
#
#             # ----------------------------
#             # Normalize landmarks relative to wrist + scale
#             # This makes predictions position- and scale-invariant.
#             # NOTE: if this normalization fixes inference but accuracy is still
#             # low, add the same normalization to collect_gestures.py and retrain
#             # so training data matches inference exactly.
#             # ----------------------------
#             wrist_x, wrist_y, wrist_z = new_xyz[0]
#             centered = [(x - wrist_x, y - wrist_y, z - wrist_z) for x, y, z in new_xyz]
#
#             all_vals = [v for pt in centered for v in pt]
#             scale = max(abs(v) for v in all_vals) or 1.0
#             normalized = [(x / scale, y / scale, z / scale) for x, y, z in centered]
#
#             landmark_vector = [v for pt in normalized for v in pt]
#             landmark_input = np.array(landmark_vector, dtype=np.float32).reshape(1, 63)
#
#             pred = gesture_model.predict(landmark_input, verbose=0)[0]
#
#             # Always pick highest-confidence class — no threshold gate on selection
#             gesture_idx = int(np.argmax(pred))
#             gesture_name = GESTURE_CLASSES[gesture_idx]
#             confidence = float(pred[gesture_idx])
#
#             # Console debug: shows all class scores so you can see the model's distribution
#             scores = "  ".join(f"{GESTURE_CLASSES[i]}:{pred[i]:.2f}" for i in range(len(GESTURE_CLASSES)))
#             print(f"Hand {hand_id}: {scores}  → {gesture_name} ({confidence:.0%})")
#
#             # ----------------------------
#             # Debounce + always-visible best guess
#             # ----------------------------
#             if hand_id not in gesture_counters:
#                 gesture_counters[hand_id] = {'name': gesture_name, 'count': 0, 'confirmed': gesture_name}
#
#             if gesture_counters[hand_id]['name'] == gesture_name:
#                 gesture_counters[hand_id]['count'] += 1
#             else:
#                 gesture_counters[hand_id]['name'] = gesture_name
#                 gesture_counters[hand_id]['count'] = 1
#
#             if gesture_counters[hand_id]['count'] >= DEBOUNCE_FRAMES:
#                 gesture_counters[hand_id]['confirmed'] = gesture_name
#
#             confirmed = gesture_counters[hand_id]['confirmed']
#
#             if confidence >= CONFIDENCE_THRESHOLD:
#                 display_gesture = f"{confirmed} ({confidence:.0%})"
#                 label_color = (0, 0, 255)    # red — confident
#             else:
#                 display_gesture = f"~{gesture_name} ({confidence:.0%})"
#                 label_color = (0, 165, 255)  # orange — uncertain
#
#             cv2.putText(frame,
#                         f"Hand {hand_id}: {display_gesture}",
#                         (10, 50 + hand_id * 25),
#                         cv2.FONT_HERSHEY_SIMPLEX,
#                         0.6, label_color, 2, cv2.LINE_AA)
#
#     # Clean up state for hands no longer in frame
#     detected_ids = set(range(len(results.hand_landmarks))) if results.hand_landmarks else set()
#     for old_id in list(smoothed_xyz.keys()):
#         if old_id not in detected_ids:
#             del smoothed_xyz[old_id]
#     for old_id in list(gesture_counters.keys()):
#         if old_id not in detected_ids:
#             del gesture_counters[old_id]
#
#     cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 255, 100), 2, cv2.LINE_AA)
#
#     cv2.imshow("Hand Tracking + Landmark Gesture Model", frame)
#     if cv2.waitKey(1) & 0xFF == ord("q"):
#         break
#
# cap.release()
# cv2.destroyAllWindows()
# landmarker.close()





# ##SECOND WORKING VERSION - EXACTLY WHAT WE WANT##
# import cv2
# import time
# import mediapipe as mp
# from mediapipe.tasks.python import vision
# from mediapipe.tasks.python.core.base_options import BaseOptions
# import numpy as np
# from tensorflow.keras.models import load_model
# import pyautogui
#
# pyautogui.FAILSAFE = False
# pyautogui.PAUSE = 0
#
# MODEL_PATH = "hand_landmarker.task"
# GESTURE_MODEL_PATH = "landmark_gesture_model.h5"
#
# # ----------------------------
# # Config
# # ----------------------------
# prev_time = time.time()
# fps = 0.0
# NUM_HANDS = 2
#
# HAND_CONNECTIONS = [
#     (0, 1), (1, 2), (2, 3), (3, 4),
#     (0, 5), (5, 6), (6, 7), (7, 8),
#     (0, 9), (9, 10), (10, 11), (11, 12),
#     (0, 13), (13, 14), (14, 15), (15, 16),
#     (0, 17), (17, 18), (18, 19), (19, 20)
# ]
#
# SMOOTHING_ALPHA = 0.3
# DEBOUNCE_FRAMES = 5
# CONFIDENCE_THRESHOLD = 0.75
#
# # ----------------------------
# # Cursor control config
# # ----------------------------
# SCREEN_W, SCREEN_H = pyautogui.size()
# CAM_MARGIN     = 0.15
# CURSOR_SMOOTH  = 0.4
# CLICK_COOLDOWN = 2.0
# ZOOM_COOLDOWN  = 1.2
#
# # ----------------------------
# # Label map
# # Maps model output index → action name.
# # Your model was trained with:
# #   0: MOVE  1: LEFT CLICK  2: RIGHT CLICK  3: ZOOM IN  4: ZOOM OUT
# # ----------------------------
# LABEL_MAP = {
#     0: 'MOVE',
#     1: 'LEFT CLICK',
#     2: 'RIGHT CLICK',
#     3: 'ZOOM IN',
#     4: 'ZOOM OUT',
# }
#
# # Actions that freeze the cursor so it stays still while performing them.
# # Only MOVE is allowed to update the cursor position.
# CURSOR_FREEZE_ACTIONS = {'LEFT CLICK', 'RIGHT CLICK', 'ZOOM IN', 'ZOOM OUT'}
#
# # ----------------------------
# # Load gesture model
# # ----------------------------
# gesture_model = load_model(GESTURE_MODEL_PATH)
# num_classes = gesture_model.output_shape[-1]
# print(f"Model loaded — {num_classes} output classes")
# print(f"Label map: {LABEL_MAP}")
#
# # ----------------------------
# # Landmarker
# # ----------------------------
# options = vision.HandLandmarkerOptions(
#     base_options=BaseOptions(model_asset_path=MODEL_PATH),
#     running_mode=vision.RunningMode.VIDEO,
#     num_hands=NUM_HANDS
# )
# landmarker = vision.HandLandmarker.create_from_options(options)
#
# # ----------------------------
# # Webcam
# # ----------------------------
# cap = cv2.VideoCapture(0)
# start_time = time.time()
#
# smoothed_xyz     = {}
# gesture_counters = {}
# cursor_x, cursor_y = SCREEN_W / 2, SCREEN_H / 2
# last_click_time  = {}
# last_zoom_time   = {}
#
# def cam_to_screen(nx, ny):
#     sx = (nx - CAM_MARGIN) / (1.0 - 2 * CAM_MARGIN)
#     sy = (ny - CAM_MARGIN) / (1.0 - 2 * CAM_MARGIN)
#     sx = float(np.clip(sx, 0.0, 1.0)) * SCREEN_W
#     sy = float(np.clip(sy, 0.0, 1.0)) * SCREEN_H
#     return sx, sy
#
# def resolve_action(pred):
#     best_idx  = int(np.argmax(pred))
#     best_conf = float(pred[best_idx])
#     action    = LABEL_MAP.get(best_idx, 'NO ACTION')
#     return action, best_conf, best_idx
#
# # ----------------------------
# # Main loop
# # ----------------------------
# while cap.isOpened():
#     success, frame = cap.read()
#     if not success:
#         break
#
#     frame = cv2.flip(frame, 1)
#
#     current_time = time.time()
#     dt = current_time - prev_time
#     prev_time = current_time
#     if dt > 0:
#         fps = 0.9 * fps + 0.1 * (1.0 / dt)
#
#     rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
#     timestamp_ms = int((time.time() - start_time) * 1000)
#     results = landmarker.detect_for_video(mp_image, timestamp_ms)
#
#     h, w, _ = frame.shape
#
#     # Draw control region
#     mx = int(CAM_MARGIN * w)
#     my = int(CAM_MARGIN * h)
#     cv2.rectangle(frame, (mx, my), (w - mx, h - my), (60, 60, 60), 1)
#     cv2.putText(frame, "control zone", (mx + 4, my - 6),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.38, (60, 60, 60), 1)
#
#     if results.hand_landmarks:
#         for hand_id, hand in enumerate(results.hand_landmarks):
#
#             # Initialise state for new hand
#             if hand_id not in smoothed_xyz:
#                 smoothed_xyz[hand_id]    = [(lm.x, lm.y, lm.z) for lm in hand]
#                 last_click_time[hand_id] = 0.0
#                 last_zoom_time[hand_id]  = 0.0
#
#             # Smooth landmarks
#             smoothed_points = []
#             new_xyz = []
#             for i, lm in enumerate(hand):
#                 px, py, pz = smoothed_xyz[hand_id][i]
#                 sx = SMOOTHING_ALPHA * lm.x + (1 - SMOOTHING_ALPHA) * px
#                 sy = SMOOTHING_ALPHA * lm.y + (1 - SMOOTHING_ALPHA) * py
#                 sz = SMOOTHING_ALPHA * lm.z + (1 - SMOOTHING_ALPHA) * pz
#                 smoothed_xyz[hand_id][i] = (sx, sy, sz)
#                 smoothed_points.append((int(sx * w), int(sy * h)))
#                 new_xyz.append((sx, sy, sz))
#
#             # Draw skeleton
#             for x, y in smoothed_points:
#                 cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)
#             for s, e in HAND_CONNECTIONS:
#                 cv2.line(frame, smoothed_points[s], smoothed_points[e], (255, 0, 0), 2)
#
#             # Normalise for model (wrist-relative + scale-invariant)
#             wrist_x, wrist_y, wrist_z = new_xyz[0]
#             centered   = [(x - wrist_x, y - wrist_y, z - wrist_z) for x, y, z in new_xyz]
#             all_vals   = [v for pt in centered for v in pt]
#             scale      = max(abs(v) for v in all_vals) or 1.0
#             normalized = [(x / scale, y / scale, z / scale) for x, y, z in centered]
#
#             landmark_input = np.array(
#                 [v for pt in normalized for v in pt], dtype=np.float32
#             ).reshape(1, 63)
#
#             pred = gesture_model.predict(landmark_input, verbose=0)[0]
#             action, confidence, raw_idx = resolve_action(pred)
#
#             # Console debug
#             score_str = "  ".join(
#                 f"{LABEL_MAP.get(i,'?')}:{pred[i]:.2f}" for i in range(len(pred))
#             )
#             print(f"Hand {hand_id}: {score_str}  → {action} ({confidence:.0%})")
#
#             # Debounce
#             if hand_id not in gesture_counters:
#                 gesture_counters[hand_id] = {'name': action, 'count': 0, 'confirmed': action}
#
#             if gesture_counters[hand_id]['name'] == action:
#                 gesture_counters[hand_id]['count'] += 1
#             else:
#                 gesture_counters[hand_id]['name'] = action
#                 gesture_counters[hand_id]['count'] = 1
#
#             if gesture_counters[hand_id]['count'] >= DEBOUNCE_FRAMES:
#                 gesture_counters[hand_id]['confirmed'] = action
#
#             confirmed = gesture_counters[hand_id]['confirmed']
#
#             # Label colour
#             if confirmed in CURSOR_FREEZE_ACTIONS:
#                 label_color = (255, 200, 0)   # cyan-ish — action mode, cursor frozen
#             elif confidence >= CONFIDENCE_THRESHOLD:
#                 label_color = (0, 0, 255)      # red — confident MOVE
#             else:
#                 label_color = (0, 165, 255)    # orange — uncertain
#
#             prefix = "" if confidence >= CONFIDENCE_THRESHOLD else "~"
#
#             # ----------------------------
#             # Act — only hand 0 drives the cursor
#             # ----------------------------
#             tip_x, tip_y         = new_xyz[8][0], new_xyz[8][1]
#             target_sx, target_sy = cam_to_screen(tip_x, tip_y)
#             now = time.time()
#
#             if confirmed == 'MOVE':
#                 # Only MOVE updates the cursor position
#                 cursor_x = CURSOR_SMOOTH * target_sx + (1 - CURSOR_SMOOTH) * cursor_x
#                 cursor_y = CURSOR_SMOOTH * target_sy + (1 - CURSOR_SMOOTH) * cursor_y
#                 if hand_id == 0:
#                     pyautogui.moveTo(int(cursor_x), int(cursor_y))
#
#             elif confirmed == 'LEFT CLICK':
#                 # Cursor stays exactly where it was — no moveTo called
#                 if hand_id == 0 and (now - last_click_time[hand_id]) > CLICK_COOLDOWN:
#                     pyautogui.click(button='left')
#                     last_click_time[hand_id] = now
#                     print("  → LEFT CLICK fired")
#
#             elif confirmed == 'RIGHT CLICK':
#                 if hand_id == 0 and (now - last_click_time[hand_id]) > CLICK_COOLDOWN:
#                     pyautogui.click(button='right')
#                     last_click_time[hand_id] = now
#                     print("  → RIGHT CLICK fired")
#
#             elif confirmed == 'ZOOM IN':
#                 if (now - last_zoom_time[hand_id]) > ZOOM_COOLDOWN:
#                     pyautogui.hotkey('ctrl', '+')
#                     last_zoom_time[hand_id] = now
#                     print("  → ZOOM IN fired")
#
#             elif confirmed == 'ZOOM OUT':
#                 if (now - last_zoom_time[hand_id]) > ZOOM_COOLDOWN:
#                     pyautogui.hotkey('ctrl', '-')
#                     last_zoom_time[hand_id] = now
#                     print("  → ZOOM OUT fired")
#
#             # HUD label — show FROZEN tag for non-MOVE actions
#             frozen_tag = "  [CURSOR FROZEN]" if confirmed in CURSOR_FREEZE_ACTIONS else ""
#             cv2.putText(frame,
#                         f"Hand {hand_id}: {prefix}{confirmed} ({confidence:.0%}){frozen_tag}",
#                         (10, 55 + hand_id * 30),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.6, label_color, 2, cv2.LINE_AA)
#
#             # Cursor indicator on fingertip
#             tip_px = smoothed_points[8]
#             if hand_id == 0 and confirmed == 'MOVE':
#                 cv2.circle(frame, tip_px, 10, (0, 255, 255), 2)
#                 cv2.putText(frame, "cursor", (tip_px[0] + 12, tip_px[1]),
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
#             elif hand_id == 0 and confirmed in CURSOR_FREEZE_ACTIONS:
#                 cv2.circle(frame, tip_px, 10, (255, 200, 0), 2)
#                 cv2.putText(frame, "frozen", (tip_px[0] + 12, tip_px[1]),
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 200, 0), 1)
#
#     # Clean up gone hands
#     detected_ids = set(range(len(results.hand_landmarks))) if results.hand_landmarks else set()
#     for old_id in list(smoothed_xyz.keys()):
#         if old_id not in detected_ids:
#             del smoothed_xyz[old_id]
#     for old_id in list(gesture_counters.keys()):
#         if old_id not in detected_ids:
#             del gesture_counters[old_id]
#
#     # FPS
#     cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 255, 100), 2, cv2.LINE_AA)
#     cv2.putText(frame, "Q = quit", (w - 90, 30),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1, cv2.LINE_AA)
#
#     # Legend
#     legend = [
#         "MOVE        = index finger pointing  -->  cursor follows",
#         "LEFT CLICK  = fist                  -->  cursor frozen, click fires",
#         "RIGHT CLICK = thumbs down           -->  cursor frozen, click fires",
#         "ZOOM IN     = rock sign             -->  cursor frozen, Ctrl+ fires",
#         "ZOOM OUT    = peace sign            -->  cursor frozen, Ctrl- fires",
#     ]
#     for i, line in enumerate(legend):
#         cv2.putText(frame, line,
#                     (10, h - 15 - (len(legend) - 1 - i) * 17),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.36, (140, 140, 140), 1, cv2.LINE_AA)
#
#     cv2.imshow("Hand Gesture Control", frame)
#     if cv2.waitKey(1) & 0xFF == ord("q"):
#         break
#
# cap.release()
# cv2.destroyAllWindows()
# landmarker.close()

import cv2
import time
import threading
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions
import numpy as np
from tensorflow.keras.models import load_model
import pyautogui

pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0

MODEL_PATH        = "hand_landmarker.task"
GESTURE_MODEL_PATH = "landmark_gesture_model.h5"

# ----------------------------
# Config
# ----------------------------
prev_time = time.time()
fps       = 0.0
NUM_HANDS = 2

HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20)
]

SMOOTHING_ALPHA      = 0.3
DEBOUNCE_FRAMES      = 5
CONFIDENCE_THRESHOLD = 0.75

# ----------------------------
# Cursor control config
# ----------------------------
SCREEN_W, SCREEN_H = pyautogui.size()
CAM_MARGIN     = 0.15
CURSOR_SMOOTH  = 0.4
CLICK_COOLDOWN = 2.0
ZOOM_COOLDOWN  = 1.2

LABEL_MAP = {
    0: 'MOVE',
    1: 'LEFT CLICK',
    2: 'RIGHT CLICK',
    3: 'ZOOM IN',
    4: 'ZOOM OUT',
}
CURSOR_FREEZE_ACTIONS = {'LEFT CLICK', 'RIGHT CLICK', 'ZOOM IN', 'ZOOM OUT'}

# Pre-build smoothing complement so it isn't recalculated every landmark every frame
_ALPHA     = SMOOTHING_ALPHA
_ONE_MINUS = 1.0 - SMOOTHING_ALPHA

# ----------------------------
# Load model — call model(x) directly instead of .predict()
# .predict() has significant per-call Python overhead; __call__ skips it
# ----------------------------
gesture_model    = load_model(GESTURE_MODEL_PATH)
num_classes      = gesture_model.output_shape[-1]

# Warm up the model so the first real call isn't slow
_dummy = np.zeros((1, 63), dtype=np.float32)
gesture_model(_dummy, training=False)

print(f"Model loaded — {num_classes} output classes")
print(f"Label map: {LABEL_MAP}")

# ----------------------------
# Landmarker
# ----------------------------
options = vision.HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=vision.RunningMode.VIDEO,
    num_hands=NUM_HANDS
)
landmarker = vision.HandLandmarker.create_from_options(options)

# ----------------------------
# Webcam — request smaller frame if camera supports it (speeds up capture + flip)
# ----------------------------
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)   # don't queue stale frames

start_time = time.time()

smoothed_xyz     = {}
gesture_counters = {}
cursor_x, cursor_y = SCREEN_W / 2, SCREEN_H / 2
last_click_time  = {}
last_zoom_time   = {}

# ----------------------------
# Threaded prediction state
# One background thread per hand runs inference so the main loop never blocks.
# ----------------------------
_pred_lock    = threading.Lock()
_pred_results = {}   # hand_id -> (action, confidence)
_pred_threads = {}   # hand_id -> Thread

def _run_prediction(hand_id, landmark_input):
    """Runs in a background thread. Writes result to _pred_results."""
    raw = gesture_model(landmark_input, training=False).numpy()[0]
    best_idx  = int(np.argmax(raw))
    best_conf = float(raw[best_idx])
    action    = LABEL_MAP.get(best_idx, 'NO ACTION')
    with _pred_lock:
        _pred_results[hand_id] = (action, best_conf, raw)

# ----------------------------
# Helpers
# ----------------------------
def cam_to_screen(nx, ny):
    sx = float(np.clip((nx - CAM_MARGIN) / (1.0 - 2 * CAM_MARGIN), 0.0, 1.0)) * SCREEN_W
    sy = float(np.clip((ny - CAM_MARGIN) / (1.0 - 2 * CAM_MARGIN), 0.0, 1.0)) * SCREEN_H
    return sx, sy

def normalize_landmarks(new_xyz):
    """Vectorized wrist-relative + scale normalization using numpy."""
    pts    = np.array(new_xyz, dtype=np.float32)   # (21, 3)
    pts   -= pts[0]                                  # subtract wrist
    scale  = np.max(np.abs(pts)) or 1.0
    pts   /= scale
    return pts.reshape(1, 63)

# ----------------------------
# Main loop
# ----------------------------
# Frame counter for debug print throttle
_frame_count = 0

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    _frame_count += 1

    current_time = time.time()
    dt = current_time - prev_time
    prev_time = current_time
    if dt > 0:
        fps = 0.9 * fps + 0.1 * (1.0 / dt)

    # Convert once; reuse for both MediaPipe and display
    rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    timestamp_ms = int((time.time() - start_time) * 1000)
    results  = landmarker.detect_for_video(mp_image, timestamp_ms)

    h, w, _ = frame.shape

    # Draw control region (reuse cached pixel margins)
    mx = int(CAM_MARGIN * w)
    my = int(CAM_MARGIN * h)
    cv2.rectangle(frame, (mx, my), (w - mx, h - my), (60, 60, 60), 1)
    cv2.putText(frame, "control zone", (mx + 4, my - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (60, 60, 60), 1)

    if results.hand_landmarks:
        for hand_id, hand in enumerate(results.hand_landmarks):

            # Init state
            if hand_id not in smoothed_xyz:
                smoothed_xyz[hand_id]    = np.array([[lm.x, lm.y, lm.z] for lm in hand], dtype=np.float32)
                last_click_time[hand_id] = 0.0
                last_zoom_time[hand_id]  = 0.0
                with _pred_lock:
                    _pred_results[hand_id] = ('MOVE', 1.0, None)

            # ----------------------------
            # Vectorized landmark smoothing (numpy, not Python loops)
            # ----------------------------
            raw_pts = np.array([[lm.x, lm.y, lm.z] for lm in hand], dtype=np.float32)
            smoothed_xyz[hand_id] = _ALPHA * raw_pts + _ONE_MINUS * smoothed_xyz[hand_id]
            sxyz = smoothed_xyz[hand_id]                          # (21, 3)
            smoothed_points = (sxyz[:, :2] * [w, h]).astype(int)  # (21, 2) pixel coords

            # Draw skeleton
            for pt in smoothed_points:
                cv2.circle(frame, tuple(pt), 4, (0, 255, 0), -1)
            for s, e in HAND_CONNECTIONS:
                cv2.line(frame, tuple(smoothed_points[s]), tuple(smoothed_points[e]), (255, 0, 0), 2)

            # ----------------------------
            # Kick off background prediction if no thread is running for this hand
            # Uses last known result until the new one arrives — zero blocking
            # ----------------------------
            thread_busy = hand_id in _pred_threads and _pred_threads[hand_id].is_alive()
            if not thread_busy:
                landmark_input = normalize_landmarks(sxyz.tolist())
                t = threading.Thread(
                    target=_run_prediction,
                    args=(hand_id, landmark_input),
                    daemon=True
                )
                _pred_threads[hand_id] = t
                t.start()

            # Read latest prediction result (non-blocking)
            with _pred_lock:
                action, confidence, raw_pred = _pred_results.get(
                    hand_id, ('MOVE', 1.0, None)
                )

            # Throttle console output to every 15 frames (~2x/sec at 30fps)
            if _frame_count % 15 == 0 and raw_pred is not None:
                score_str = "  ".join(
                    f"{LABEL_MAP.get(i,'?')}:{raw_pred[i]:.2f}" for i in range(len(raw_pred))
                )
                print(f"Hand {hand_id}: {score_str}  → {action} ({confidence:.0%})")

            # ----------------------------
            # Debounce
            # ----------------------------
            if hand_id not in gesture_counters:
                gesture_counters[hand_id] = {'name': action, 'count': 0, 'confirmed': action}

            if gesture_counters[hand_id]['name'] == action:
                gesture_counters[hand_id]['count'] += 1
            else:
                gesture_counters[hand_id]['name'] = action
                gesture_counters[hand_id]['count'] = 1

            if gesture_counters[hand_id]['count'] >= DEBOUNCE_FRAMES:
                gesture_counters[hand_id]['confirmed'] = action

            confirmed = gesture_counters[hand_id]['confirmed']

            # Label colour
            if confirmed in CURSOR_FREEZE_ACTIONS:
                label_color = (255, 200, 0)
            elif confidence >= CONFIDENCE_THRESHOLD:
                label_color = (0, 0, 255)
            else:
                label_color = (0, 165, 255)

            prefix = "" if confidence >= CONFIDENCE_THRESHOLD else "~"

            # ----------------------------
            # Act — only hand 0 drives the cursor
            # ----------------------------
            tip_x, tip_y         = float(sxyz[8, 0]), float(sxyz[8, 1])
            target_sx, target_sy = cam_to_screen(tip_x, tip_y)
            now = time.time()

            if confirmed == 'MOVE':
                cursor_x = CURSOR_SMOOTH * target_sx + (1 - CURSOR_SMOOTH) * cursor_x
                cursor_y = CURSOR_SMOOTH * target_sy + (1 - CURSOR_SMOOTH) * cursor_y
                if hand_id == 0:
                    pyautogui.moveTo(int(cursor_x), int(cursor_y))

            elif confirmed == 'LEFT CLICK':
                if hand_id == 0 and (now - last_click_time[hand_id]) > CLICK_COOLDOWN:
                    pyautogui.click(button='left')
                    last_click_time[hand_id] = now
                    print("  → LEFT CLICK fired")

            elif confirmed == 'RIGHT CLICK':
                if hand_id == 0 and (now - last_click_time[hand_id]) > CLICK_COOLDOWN:
                    pyautogui.click(button='right')
                    last_click_time[hand_id] = now
                    print("  → RIGHT CLICK fired")

            elif confirmed == 'ZOOM IN':
                if (now - last_zoom_time[hand_id]) > ZOOM_COOLDOWN:
                    pyautogui.hotkey('ctrl', '+')
                    last_zoom_time[hand_id] = now
                    print("  → ZOOM IN fired")

            elif confirmed == 'ZOOM OUT':
                if (now - last_zoom_time[hand_id]) > ZOOM_COOLDOWN:
                    pyautogui.hotkey('ctrl', '-')
                    last_zoom_time[hand_id] = now
                    print("  → ZOOM OUT fired")

            # HUD
            frozen_tag = "  [CURSOR FROZEN]" if confirmed in CURSOR_FREEZE_ACTIONS else ""
            cv2.putText(frame,
                        f"Hand {hand_id}: {prefix}{confirmed} ({confidence:.0%}){frozen_tag}",
                        (10, 55 + hand_id * 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, label_color, 2, cv2.LINE_AA)

            tip_px = tuple(smoothed_points[8])
            if hand_id == 0 and confirmed == 'MOVE':
                cv2.circle(frame, tip_px, 10, (0, 255, 255), 2)
                cv2.putText(frame, "cursor", (tip_px[0] + 12, tip_px[1]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
            elif hand_id == 0 and confirmed in CURSOR_FREEZE_ACTIONS:
                cv2.circle(frame, tip_px, 10, (255, 200, 0), 2)
                cv2.putText(frame, "frozen", (tip_px[0] + 12, tip_px[1]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 200, 0), 1)

    # Clean up gone hands
    detected_ids = set(range(len(results.hand_landmarks))) if results.hand_landmarks else set()
    for old_id in list(smoothed_xyz.keys()):
        if old_id not in detected_ids:
            del smoothed_xyz[old_id]
    for old_id in list(gesture_counters.keys()):
        if old_id not in detected_ids:
            del gesture_counters[old_id]

    # FPS
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 255, 100), 2, cv2.LINE_AA)
    cv2.putText(frame, "Q = quit", (w - 90, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1, cv2.LINE_AA)

    legend = [
        "MOVE        = index finger pointing  -->  cursor follows",
        "LEFT CLICK  = thumb-to-side          -->  cursor frozen, click fires",
        "RIGHT CLICK = rock sign              -->  cursor frozen, click fires",
        "ZOOM IN     = okay sign              -->  cursor frozen, Ctrl+ fires",
        "ZOOM OUT    = fist sign              -->  cursor frozen, Ctrl- fires",
    ]
    for i, line in enumerate(legend):
        cv2.putText(frame, line,
                    (10, h - 15 - (len(legend) - 1 - i) * 17),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, (140, 140, 140), 1, cv2.LINE_AA)

    cv2.imshow("Hand Gesture Control", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
landmarker.close()
