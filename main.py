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

This file is intentionally thin — it wires together the classes in
camera_manager.py, landmark_pipeline.py, auth_manager.py, gesture_engine.py,
cursor_controller.py, hud_renderer.py, ptz_controller.py, and
face_recognizer.py. All of the actual detection/control/drawing logic
lives in those modules.
"""

import argparse
import cv2
import time
import numpy as np
import mediapipe as mp

from face_recognizer import UNKNOWN_LABEL
from ptz_controller import PTZController
from camera_manager import CameraManager
from landmark_pipeline import LandmarkPipeline
from auth_manager import AuthManager
from gesture_engine import GestureEngine, FINGER_CHAINS, CURSOR_FREEZE_ACTIONS
from cursor_controller import CursorController
from hud_renderer import HUDRenderer
from Zoom_webcam import ZoomWebcamController

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
POSE_VIS_THRESH      = 0.4
ENABLE_AUTO_ZOOM    = True # digital zoom/crop that frames the recognised user's body
                            # (display-only — toggle live with the 'z' key)

# ── Parse logged-in user from login_system.py ────────────────────────────────
_arg_parser = argparse.ArgumentParser()
_arg_parser.add_argument("--user", type=str, default=None,
                         help="Username of the logged-in user (set by login_system.py)")
_args       = _arg_parser.parse_args()
ACTIVE_USER = _args.user
if ACTIVE_USER:
    print(f"Active session for user: {ACTIVE_USER}")
else:
    print("WARNING: no --user passed. Run login_system.py instead of main.py directly.")

# ── Core components ───────────────────────────────────────────────────────────
auth       = AuthManager(active_user=ACTIVE_USER, face_detect_interval=FACE_DETECT_INTERVAL)
ptz        = PTZController(active_user=ACTIVE_USER)
gestures   = GestureEngine(GESTURE_MODEL_PATH, num_hands=NUM_HANDS)
cursor     = CursorController()
hud        = HUDRenderer(num_registered=auth.num_registered)
auto_zoom  = ZoomWebcamController(enabled=ENABLE_AUTO_ZOOM,
                                  control_zone_margin=CursorController.CAM_MARGIN)
landmarks  = LandmarkPipeline(
    HAND_TASK_PATH, FACE_TASK_PATH, POSE_TASK_PATH,
    num_hands=NUM_HANDS,
    face_detect_interval=FACE_DETECT_INTERVAL,
    pose_detect_interval=POSE_DETECT_INTERVAL,
)
cam = CameraManager(width=640, height=480, fps=30)

start_time = time.perf_counter()
fps        = 0.0
prev_time  = time.perf_counter()
frame_n    = 0

# ── Main loop ─────────────────────────────────────────────────────────────────
while cam.is_opened():
    success, frame = cam.read()
    if not success:
        break

    frame_n += 1
    frame = cv2.flip(frame, 1)  # mirror horizontally

    now_t     = time.perf_counter()
    dt        = now_t - prev_time
    prev_time = now_t
    if dt > 0:
        fps = 0.9 * fps + 0.1 / dt

    h, w  = frame.shape[:2]
    rgb_c = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)   # contiguous RGB for MediaPipe
    ts_ms = int((now_t - start_time) * 1000)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_c)

    # ── Detection (hand every frame; face/pose on their own interval) ────────
    h_result, cached_face_lms, cached_pose_lms, face_updated = landmarks.detect(
        mp_image, frame_n, ts_ms)
    num_faces  = len(cached_face_lms)
    num_people = len(cached_pose_lms)

    # ── Authentication (only recompute when face detection actually ran) ────
    if face_updated:
        auth.update(cached_face_lms, now_t)

    # ── Auto zoom (display-only; tracks only the recognised user's body) ────
    tracked_pose = None
    if auth.is_registered_face_visible and cached_pose_lms:
        best_dist = None
        for pose_lms in cached_pose_lms:
            p_nose = np.array([pose_lms[0].x, pose_lms[0].y], dtype=np.float32)
            dist   = float(np.linalg.norm(p_nose - auth.face_nose_pos))
            if dist < 0.18 and (best_dist is None or dist < best_dist):
                best_dist    = dist
                tracked_pose = pose_lms
    auto_zoom.update(tracked_pose, w, h)

    # ── PTZ tracking (drives the physical pan/tilt hardware) ─────────────────
    ptz.update(
        nose_pos=auth.face_nose_pos,
        recognised_user=auth.recognised_user,
        is_registered_face_visible=auth.is_registered_face_visible,
        pose_landmarks=cached_pose_lms,
    )

    # ── Crop/zoom the CLEAN camera frame FIRST, before any HUD is drawn ─────
    # Every draw_* call below then draws fresh at native resolution on this
    # already-zoomed frame, using zoom=auto_zoom to remap each element's
    # raw-frame coordinate to where it belongs post-crop. This keeps text
    # and thin lines crisp under magnification (rather than drawing HUD on
    # the raw frame and blurrily stretching the whole composited image),
    # and is what makes the control-zone box, skeletons, and labels track
    # correctly with the zoom.
    display_frame = auto_zoom.apply(frame)

    # ── Draw cached face boxes / pose skeletons ──────────────────────────────
    hud.draw_face_boxes(display_frame, auth.face_rec, cached_face_lms,
                         auth.recognised_user, UNKNOWN_LABEL, w, h, zoom=auto_zoom)
    hud.draw_pose_skeletons(display_frame, cached_pose_lms, auth.user_active,
                             auth.face_nose_pos, auth.recog_score, auth.face_size, w, h,
                             zoom=auto_zoom)

    # ── Control zone + crosshair ──────────────────────────────────────────────
    hud.draw_control_zone(display_frame, auth.limb_mode, CursorController.CAM_MARGIN,
                           w, h, zoom=auto_zoom)
    hud.draw_crosshair(display_frame, auth.limb_mode, auth.face_nose_pos, w, h, zoom=auto_zoom)

    hand_action_strs = []

    if h_result.hand_landmarks:
        for hand_id, hand in enumerate(h_result.hand_landmarks):
            wrist_raw = np.array([hand[0].x, hand[0].y], dtype=np.float32)
            raw_pts   = np.array([[lm.x, lm.y, lm.z] for lm in hand], dtype=np.float32)

            sxyz = gestures.smooth_hand(hand_id, raw_pts)
            sp   = (sxyz[:, :2] * (w, h)).astype(np.int32)
            wrist = wrist_raw

            hand_is_user = (auth.user_active and auth.face_nose_pos is not None and
                            float(np.linalg.norm(wrist - auth.face_nose_pos)) < 0.70)

            line_col = (255, 0, 0) if hand_is_user else (0, 0, 200)
            dot_col  = (0, 255, 0) if hand_is_user else (0, 0, 200)
            hud.draw_hand(display_frame, FINGER_CHAINS, sp, line_col, dot_col, zoom=auto_zoom)

            confirmed, confidence = gestures.classify(hand_id, sxyz)

            if not hand_is_user:                     label_color = (0, 0, 200)
            elif confirmed == 'NO ACTION':           label_color = (120, 120, 120)
            elif confirmed in CURSOR_FREEZE_ACTIONS: label_color = (255, 200, 0)
            elif confidence >= 0.75:                 label_color = (0, 200, 0)
            else:                                    label_color = (0, 165, 255)

            user_tag = auth.recognised_user[:3].upper() if auth.recognised_user != UNKNOWN_LABEL else 'UNK'
            hud_str  = f"H{hand_id}[{user_tag if hand_is_user else 'UNK'}]: {confirmed} ({confidence:.0%})"
            if not hand_is_user:
                hud_str += "  [blocked]"
            hand_action_strs.append((hud_str, label_color))

            tip_px = (int(sp[8, 0]), int(sp[8, 1]))

            if hand_is_user and hand_id == 0:
                tip_xy = sxyz[8, :2]
                cursor.handle_action(confirmed, hand_id, tip_xy)
                if confirmed == 'MOVE':
                    hud.draw_move_indicator(display_frame, tip_px, auth.limb_mode,
                                             auth.face_nose_pos, w, h, zoom=auto_zoom)

            elif not hand_is_user:
                hud.draw_blocked_hand(display_frame, tip_px, zoom=auto_zoom)

    detected = set(range(len(h_result.hand_landmarks))) if h_result.hand_landmarks else set()
    gestures.forget_stale_hands(detected)

    # ── HUD (fixed-position chrome — drawn on the final frame either way) ────
    hud.draw_status_hud(display_frame, fps, num_people, num_faces, hand_action_strs, w)
    hud.draw_auth_banner(display_frame, auth.auth_temp, auth.user_active,
                          auth.grace_remaining(now_t), AuthManager.TEMP_ACTIVATE, w, h)
    ptz.draw_debug_hud(display_frame)
    display_frame = auto_zoom.draw_debug(display_frame)

    small = cv2.resize(display_frame, (100, 100))
    if smallView:
        cv2.imshow("Multi-Person Limb Gesture Control", small)
    else:
        cv2.imshow("Multi-Person Limb Gesture Control", display_frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif key == ord("z"):
        state = auto_zoom.toggle()
        print(f"Auto zoom {'enabled' if state else 'disabled'}")

cam.release()
cv2.destroyAllWindows()
landmarks.close()
ptz.close()