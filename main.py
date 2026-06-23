"""
main.py — Multi-Person Limb Gesture Control
─────────────────────────────────────────────────────────────────────────────
Pipeline each frame:
  1. HandLandmarker  → every frame   (cursor needs to be responsive)
  2. FaceLandmarker  → every 3 frames (face moves slowly; results cached)
  3. PoseLandmarker  → every 2 frames (body moves slowly; results cached)
  4. Face recogniser → cosine match → authentication temperature
  5. Gesture model   → background inference thread (never blocks main loop)
  6. Cursor control  → registered user only; bystanders blocked

Skeleton colours:
  GREEN = registered user   RED = bystander / unknown

Run face_register.py first to register your face.
"""

import argparse
import cv2
import time
import queue
import threading
import os
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions
from keras.models import load_model
import pyautogui

from face_recognizer import FaceRecognizer, UNKNOWN_LABEL

pyautogui.FAILSAFE = False
pyautogui.PAUSE    = 0
smallView = False

# ── Model paths ───────────────────────────────────────────────────────────────
HAND_TASK_PATH     = "hand_landmarker.task"
FACE_TASK_PATH     = "face_landmarker.task"
POSE_TASK_PATH     = "pose_landmarker_lite.task"
GESTURE_MODEL_PATH = "landmark_gesture_model.h5"

# ── Config ────────────────────────────────────────────────────────────────────
NUM_HANDS            = 2
FACE_DETECT_INTERVAL = 3    # run face detection every N frames
POSE_DETECT_INTERVAL = 2    # run pose detection every N frames

FINGER_CHAINS = [
    [0,1,2,3,4], [0,5,6,7,8], [0,9,10,11,12], [0,13,14,15,16], [0,17,18,19,20]
]

SMOOTHING_ALPHA = np.float32(0.3)
ONE_MINUS_ALPHA = np.float32(0.7)
DEBOUNCE_FRAMES = 5

# ── Cursor config ─────────────────────────────────────────────────────────────
SCREEN_W, SCREEN_H = pyautogui.size()
CAM_MARGIN          = 0.15
_CAM_SCALE          = 1.0 / (1.0 - 2 * CAM_MARGIN)
CURSOR_SMOOTH       = 0.35
CLICK_COOLDOWN      = 2.0
ZOOM_COOLDOWN       = 1.2

# ── Authentication temperature ────────────────────────────────────────────────
TEMP_RISE       = 0.08 * FACE_DETECT_INTERVAL   # compensate for skipped frames
TEMP_FALL       = 0.04 * FACE_DETECT_INTERVAL
TEMP_ACTIVATE   = 0.60
TEMP_DEACTIVATE = 0.25
GRACE_SECONDS   = 10.0

auth_temp      = 0.0
user_active    = False
last_seen_time = 0.0

# ── Pose skeleton connections ─────────────────────────────────────────────────
POSE_CONNECTIONS = np.array([
    (0,1),(1,2),(2,3),(3,7),(0,4),(4,5),(5,6),(6,8),
    (11,12),
    (11,13),(13,15),(12,14),(14,16),
    (15,17),(15,19),(15,21),(16,18),(16,20),(16,22),
    (11,23),(12,24),(23,24),
    (23,25),(25,27),(27,29),(27,31),(29,31),
    (24,26),(26,28),(28,30),(28,32),(30,32),
], dtype=np.int32)

POSE_VIS_THRESH = 0.4

LABEL_MAP = {
    0: 'MOVE', 1: 'LEFT CLICK', 2: 'RIGHT CLICK', 3: 'ZOOM IN', 4: 'ZOOM OUT',
}
CURSOR_FREEZE_ACTIONS = {'LEFT CLICK', 'RIGHT CLICK', 'ZOOM IN', 'ZOOM OUT'}

# ── Load gesture model ────────────────────────────────────────────────────────
gesture_model = load_model(GESTURE_MODEL_PATH)
gesture_model(np.zeros((1, 63), dtype=np.float32), training=False)
print(f"Gesture model loaded — {gesture_model.output_shape[-1]} classes")

# ── Parse logged-in user from login_system.py ────────────────────────────────
_arg_parser = argparse.ArgumentParser()
_arg_parser.add_argument("--user", type=str, default=None,
                         help="Username of the logged-in user (set by login_system.py)")
_args      = _arg_parser.parse_args()
ACTIVE_USER = _args.user
if ACTIVE_USER:
    print(f"Active session for user: {ACTIVE_USER}")
else:
    print("WARNING: no --user passed. Run login_system.py instead of main.py directly.")

# ── Face recognizer ───────────────────────────────────────────────────────────
face_rec       = FaceRecognizer(active_user=ACTIVE_USER)
num_registered = len(face_rec.face_db)

# ── MediaPipe landmarkers ─────────────────────────────────────────────────────
hand_landmarker = vision.HandLandmarker.create_from_options(
    vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=HAND_TASK_PATH),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=NUM_HANDS,
    )
)

face_landmarker = None
if os.path.exists(FACE_TASK_PATH):
    face_landmarker = vision.FaceLandmarker.create_from_options(
        vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=FACE_TASK_PATH),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=4,
            min_face_detection_confidence=0.45,
            min_face_presence_confidence=0.45,
            min_tracking_confidence=0.45,
        )
    )
    print("Face landmarker: ready")
else:
    print(f"WARNING: '{FACE_TASK_PATH}' not found — face recognition disabled.")

pose_landmarker = None
if os.path.exists(POSE_TASK_PATH):
    pose_landmarker = vision.PoseLandmarker.create_from_options(
        vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=POSE_TASK_PATH),
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

# ── Camera ───────────────────────────────────────────────────────────────────
try:
    from picamera2 import Picamera2
    _HAS_PICAMERA2 = True
except ImportError:
    _HAS_PICAMERA2 = False

if _HAS_PICAMERA2:
    _picam = Picamera2()
    _picam.configure(_picam.create_video_configuration(
        main={"size": (640, 480), "format": "RGB888"}))
    _picam.start()

    class _RPiCap:
        def isOpened(self): return True
        def read(self):     return True, _picam.capture_array()
        def release(self):  _picam.stop()

    cap = _RPiCap()
    print("Camera: RPi CSI (picamera2)")
else:
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS,          30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)
    print("Camera: USB webcam (cv2)")

start_time = time.perf_counter()

# ── Runtime state ─────────────────────────────────────────────────────────────
smoothed_xyz     = {}
gesture_counters = {}
cursor_x, cursor_y = SCREEN_W / 2.0, SCREEN_H / 2.0
last_click_time  = {}
last_zoom_time   = {}

# Cached detection results (updated on their respective intervals)
face_nose_pos   = None
face_size       = None
recognised_user = UNKNOWN_LABEL
recog_score     = 0.0
limb_mode       = False
num_faces       = 0
cached_face_lms = []

cached_pose_lms = []
num_people      = 0

_chain_bufs = [np.zeros((len(c), 1, 2), dtype=np.int32) for c in FINGER_CHAINS]

fps       = 0.0
prev_time = time.perf_counter()
_frame_n  = 0

# ── Gesture inference worker ──────────────────────────────────────────────────
class InferenceWorker:
    def __init__(self):
        self._q      = queue.Queue(maxsize=1)
        self._result = ('MOVE', 1.0)
        self._lock   = threading.Lock()
        threading.Thread(target=self._run, daemon=True).start()

    def submit(self, x):
        try: self._q.put_nowait(x)
        except queue.Full: pass

    def result(self):
        with self._lock: return self._result

    def _run(self):
        while True:
            data = self._q.get()
            raw  = gesture_model(data, training=False).numpy()[0]
            idx  = int(np.argmax(raw))
            conf = float(raw[idx])
            act  = LABEL_MAP.get(idx, 'NO ACTION') if conf >= 0.75 else 'NO ACTION'
            with self._lock: self._result = (act, conf)

workers = {i: InferenceWorker() for i in range(NUM_HANDS)}

# ── Helpers ───────────────────────────────────────────────────────────────────
def absolute_to_screen(nx, ny):
    sx = float(np.clip((nx - CAM_MARGIN) * _CAM_SCALE, 0.0, 1.0)) * SCREEN_W
    sy = float(np.clip((ny - CAM_MARGIN) * _CAM_SCALE, 0.0, 1.0)) * SCREEN_H
    return sx, sy

def draw_pose_skeleton(frame, pose_lms, color, w, h, label=None, scores=None):
    lm_arr = np.array([[lm.x, lm.y] for lm in pose_lms], dtype=np.float32)
    vis    = np.array([lm.visibility for lm in pose_lms], dtype=np.float32)
    pts    = (lm_arr * (w, h)).astype(np.int32)

    for a, b in POSE_CONNECTIONS:
        if vis[a] >= POSE_VIS_THRESH and vis[b] >= POSE_VIS_THRESH:
            cv2.line(frame, tuple(pts[a]), tuple(pts[b]), color, 2)

    # Synthetic neck: midpoint of shoulders connected up to nose
    if vis[11] >= POSE_VIS_THRESH and vis[12] >= POSE_VIS_THRESH:
        neck = ((pts[11].astype(np.float32) + pts[12].astype(np.float32)) / 2).astype(np.int32)
        cv2.circle(frame, tuple(neck), 4, color, -1)
        if vis[0] >= POSE_VIS_THRESH:
            cv2.line(frame, tuple(neck), tuple(pts[0]), color, 2)

    for pt in pts[vis >= POSE_VIS_THRESH]:
        cv2.circle(frame, tuple(pt), 4, color, -1)

    if label is not None and vis[0] >= POSE_VIS_THRESH:
        lx, ly = int(pts[0][0]) - 20, max(int(pts[0][1]) - 18, 10)
        cv2.putText(frame, label, (lx, ly),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
        if scores:
            cv2.putText(frame, scores, (lx, ly + 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)

# ── Main loop ─────────────────────────────────────────────────────────────────
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    _frame_n += 1
    frame = cv2.flip(frame, 1)  # mirror horizontally

    now_t     = time.perf_counter()
    dt        = now_t - prev_time
    prev_time = now_t
    if dt > 0:
        fps = 0.9 * fps + 0.1 / dt

    h, w  = frame.shape[:2]
    rgb_c = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)   # contiguous RGB for MediaPipe
    ts_ms = int((now_t - start_time) * 1000)

    # ── Face detection (every FACE_DETECT_INTERVAL frames) ───────────────────
    if face_landmarker is not None and _frame_n % FACE_DETECT_INTERVAL == 0:
        f_result        = face_landmarker.detect_for_video(
            mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_c), ts_ms)
        cached_face_lms = f_result.face_landmarks
        num_faces       = len(cached_face_lms)

        face_nose_pos   = None
        face_size       = None
        recognised_user = UNKNOWN_LABEL
        recog_score     = 0.0

        for fi, face_lms in enumerate(cached_face_lms):
            user, score = face_rec.recognise(face_lms)
            if fi == 0:
                recognised_user = user
                recog_score     = score
                face_nose_pos   = face_rec.get_nose_tip(face_lms)
                face_size       = face_rec.get_face_size(face_lms)

                if user != UNKNOWN_LABEL:
                    auth_temp      = min(1.0, auth_temp + TEMP_RISE)
                    last_seen_time = now_t
                else:
                    auth_temp = max(0.0, auth_temp - TEMP_FALL)

                if not user_active:
                    if auth_temp >= TEMP_ACTIVATE:
                        user_active = True
                else:
                    if auth_temp < TEMP_DEACTIVATE and (now_t - last_seen_time) > GRACE_SECONDS:
                        user_active = False

                limb_mode = user_active

    # Draw cached face boxes
    for fi, face_lms in enumerate(cached_face_lms):
        nose   = face_rec.get_nose_tip(face_lms)
        pts_px = np.array([[int(lm.x*w), int(lm.y*h)] for lm in face_lms], dtype=np.int32)
        xs, ys  = pts_px[:,0], pts_px[:,1]
        box_col = (0,220,0) if (fi==0 and recognised_user != UNKNOWN_LABEL) else (0,0,200)
        cv2.rectangle(frame,
                      (max(xs.min()-10,0), max(ys.min()-10,0)),
                      (min(xs.max()+10,w), min(ys.max()+10,h)),
                      box_col, 2)
        cv2.circle(frame, (int(nose[0]*w), int(nose[1]*h)), 5, (0,255,255), -1)

    # ── Pose detection (every POSE_DETECT_INTERVAL frames) ───────────────────
    if pose_landmarker is not None and _frame_n % POSE_DETECT_INTERVAL == 0:
        p_result        = pose_landmarker.detect_for_video(
            mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_c), ts_ms)
        cached_pose_lms = p_result.pose_landmarks
        num_people      = len(cached_pose_lms)

    # Draw cached pose skeletons
    for pi, pose_lms in enumerate(cached_pose_lms):
        p_nose = np.array([pose_lms[0].x, pose_lms[0].y], dtype=np.float32)
        is_reg = (user_active and face_nose_pos is not None and
                  float(np.linalg.norm(p_nose - face_nose_pos)) < 0.18)
        skel_col   = (0,220,0) if is_reg else (0,0,200)
        above_txt  = "YOU" if is_reg else f"P{pi}:UNK"
        scores_txt = f"pca={recog_score:.3f}  geo={face_size:.3f}" if (is_reg and face_size) else None
        draw_pose_skeleton(frame, pose_lms, skel_col, w, h, above_txt, scores_txt)

    # ── Hand detection (every frame) ─────────────────────────────────────────
    h_result = hand_landmarker.detect_for_video(
        mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_c), ts_ms)

    # Control zone
    mx, my   = int(CAM_MARGIN*w), int(CAM_MARGIN*h)
    zone_col = (0,200,80) if limb_mode else (60,60,60)
    cv2.rectangle(frame, (mx,my), (w-mx,h-my), zone_col, 1)
    cv2.putText(frame, "control zone", (mx+4,my-5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.33, zone_col, 1)

    if limb_mode and face_nose_pos is not None:
        nx_px, ny_px = int(face_nose_pos[0]*w), int(face_nose_pos[1]*h)
        cv2.line(frame, (nx_px-15,ny_px), (nx_px+15,ny_px), (0,255,255), 1)
        cv2.line(frame, (nx_px,ny_px-15), (nx_px,ny_px+15), (0,255,255), 1)

    hand_action_strs = []

    if h_result.hand_landmarks:
        for hand_id, hand in enumerate(h_result.hand_landmarks):
            wrist = np.array([hand[0].x, hand[0].y], dtype=np.float32)
            hand_is_user = (user_active and face_nose_pos is not None and
                            float(np.linalg.norm(wrist - face_nose_pos)) < 0.70)

            if hand_id not in smoothed_xyz:
                smoothed_xyz[hand_id]    = np.array([[lm.x,lm.y,lm.z] for lm in hand], dtype=np.float32)
                last_click_time[hand_id] = 0.0
                last_zoom_time[hand_id]  = 0.0

            raw_pts = np.array([[lm.x,lm.y,lm.z] for lm in hand], dtype=np.float32)
            sxyz    = SMOOTHING_ALPHA * raw_pts + ONE_MINUS_ALPHA * smoothed_xyz[hand_id]
            smoothed_xyz[hand_id] = sxyz
            sp = (sxyz[:,:2] * (w,h)).astype(np.int32)

            line_col = (255,0,0) if hand_is_user else (0,0,200)
            dot_col  = (0,255,0) if hand_is_user else (0,0,200)
            for ci, chain in enumerate(FINGER_CHAINS):
                _chain_bufs[ci][:,0,:] = sp[chain]
                cv2.polylines(frame, [_chain_bufs[ci]], False, line_col, 2)
            for pt in sp:
                cv2.circle(frame, (int(pt[0]),int(pt[1])), 4, dot_col, -1)

            pts_n  = sxyz - sxyz[0]
            scale  = np.max(np.abs(pts_n)) or 1.0
            pts_n /= scale
            workers[hand_id].submit(pts_n.reshape(1,63).astype(np.float32))
            action, confidence = workers[hand_id].result()

            if hand_id not in gesture_counters:
                gesture_counters[hand_id] = {'name': action, 'count': 0, 'confirmed': action}
            gc = gesture_counters[hand_id]
            if gc['name'] == action: gc['count'] += 1
            else: gc['name'] = action; gc['count'] = 1
            if gc['count'] >= DEBOUNCE_FRAMES: gc['confirmed'] = action
            confirmed = gc['confirmed']

            if not hand_is_user:             label_color = (0,0,200)
            elif confirmed == 'NO ACTION':   label_color = (120,120,120)
            elif confirmed in CURSOR_FREEZE_ACTIONS: label_color = (255,200,0)
            elif confidence >= 0.75:         label_color = (0,200,0)
            else:                            label_color = (0,165,255)

            user_tag = recognised_user[:3].upper() if recognised_user != UNKNOWN_LABEL else 'UNK'
            hud_str  = f"H{hand_id}[{user_tag if hand_is_user else 'UNK'}]: {confirmed} ({confidence:.0%})"
            if not hand_is_user: hud_str += "  [blocked]"
            hand_action_strs.append((hud_str, label_color))

            tip_px = (int(sp[8,0]), int(sp[8,1]))

            if hand_is_user and hand_id == 0:
                tip_xy = sxyz[8,:2]
                now    = time.perf_counter()

                if confirmed == 'MOVE':
                    tgt_x, tgt_y = absolute_to_screen(float(tip_xy[0]), float(tip_xy[1]))
                    cursor_x += CURSOR_SMOOTH * (tgt_x - cursor_x)
                    cursor_y += CURSOR_SMOOTH * (tgt_y - cursor_y)
                    pyautogui.moveTo(int(cursor_x), int(cursor_y))
                    cv2.circle(frame, tip_px, 10, (0,255,255), 2)
                    if limb_mode and face_nose_pos is not None:
                        cv2.line(frame, (int(face_nose_pos[0]*w), int(face_nose_pos[1]*h)),
                                 tip_px, (0,255,255), 1)

                elif confirmed == 'LEFT CLICK':
                    if (now - last_click_time[hand_id]) > CLICK_COOLDOWN:
                        pyautogui.click(button='left')
                        last_click_time[hand_id] = now

                elif confirmed == 'RIGHT CLICK':
                    if (now - last_click_time[hand_id]) > CLICK_COOLDOWN:
                        pyautogui.click(button='right')
                        last_click_time[hand_id] = now

                elif confirmed == 'ZOOM IN':
                    if (now - last_zoom_time[hand_id]) > ZOOM_COOLDOWN:
                        pyautogui.hotkey('ctrl', '+')
                        last_zoom_time[hand_id] = now

                elif confirmed == 'ZOOM OUT':
                    if (now - last_zoom_time[hand_id]) > ZOOM_COOLDOWN:
                        pyautogui.hotkey('ctrl', '-')
                        last_zoom_time[hand_id] = now

            elif not hand_is_user:
                cv2.circle(frame, tip_px, 10, (0,0,200), 2)
                cv2.putText(frame, "blocked", (tip_px[0]+6, tip_px[1]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,200), 1)

    detected = set(range(len(h_result.hand_landmarks))) if h_result.hand_landmarks else set()
    for oid in list(smoothed_xyz):
        if oid not in detected: del smoothed_xyz[oid]
    for oid in list(gesture_counters):
        if oid not in detected: del gesture_counters[oid]

    # ── HUD ───────────────────────────────────────────────────────────────────
    cv2.putText(frame, f"FPS:{fps:.0f}  People:{num_people}  Faces:{num_faces}",
                (10,22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (100,255,100), 2, cv2.LINE_AA)

    for i, (hstr, hcol) in enumerate(hand_action_strs):
        cv2.putText(frame, hstr, (10, 50+i*26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, hcol, 2, cv2.LINE_AA)

    cv2.putText(frame, "Q=quit", (w-70,22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180,180,180), 1)

    legend = [
        "GREEN = registered user  |  RED = bystander (blocked)",
        "LEFT=fist  RIGHT=thumbs-down  ZOOM IN=rock  ZOOM OUT=peace",
    ]
    for i, line in enumerate(legend):
        cv2.putText(frame, line, (6, h-8-(len(legend)-1-i)*15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.30, (120,120,120), 1)

    temp_pct        = int(auth_temp * 100)
    grace_remaining = max(0.0, GRACE_SECONDS - (now_t - last_seen_time))
    if user_active:
        banner_txt = (f"REGISTERED USER ACTIVE  temp={temp_pct}%"
                      if auth_temp >= TEMP_ACTIVATE
                      else f"REGISTERED USER ACTIVE  temp={temp_pct}%  grace {grace_remaining:.1f}s")
        banner_col = (0,220,0)
    else:
        banner_txt = ("NO USERS REGISTERED — run face_register.py"
                      if num_registered == 0
                      else f"Users: {num_registered} registered  |  temp={temp_pct}%  (show face)")
        banner_col = (0,140,200)

    cv2.rectangle(frame, (0,h-48), (w,h-20), (20,20,20), -1)
    cv2.putText(frame, banner_txt, (10,h-27),
                cv2.FONT_HERSHEY_SIMPLEX, 0.60, banner_col, 2, cv2.LINE_AA)

    small = cv2.resize(frame, (100, 100))
    if smallView:
        cv2.imshow("Multi-Person Limb Gesture Control", small)
    else:
        cv2.imshow("Multi-Person Limb Gesture Control", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
hand_landmarker.close()
if face_landmarker: face_landmarker.close()
if pose_landmarker: pose_landmarker.close()
