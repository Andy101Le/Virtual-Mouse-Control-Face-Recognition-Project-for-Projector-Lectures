"""
gesture_session.py
───────────────────
Runs the whole detection pipeline on a background thread and exposes it
to the web server: preview frames out, control commands in.

This is the piece that used to be main.py's `while` loop. It keeps all
the same components (CameraManager, LandmarkPipeline, AuthManager,
GestureEngine, HUDRenderer, ZoomWebcamController, PTZController) and the
same ordering — crop first, then draw HUD zoom-aware on top — so the
preview in the browser looks exactly like the old desktop window.

Two things are different from main.py:

  1. The cursor goes out over Bluetooth HID (BTCursorController) rather
     than pyautogui. Nothing else in the pipeline changes, because
     BTCursorController deliberately mirrors CursorController's
     interface.

  2. Face registration happens here too, driven from the web. The old
     Tkinter flow needed a keyboard (hold SPACE) and its own camera
     handle. Two processes can't both own the Pi camera, so registration
     has to run inside this same loop: the web UI flips
     `capture_request` on, the loop harvests embeddings from the frames
     it's already computing, and reports progress back over Socket.IO.
"""

import base64
import logging
import threading
import time

import cv2
import numpy as np
import mediapipe as mp

from camera_manager import CameraManager
from landmark_pipeline import LandmarkPipeline
from auth_manager import AuthManager
from gesture_engine import GestureEngine, FINGER_CHAINS, CURSOR_FREEZE_ACTIONS
from hud_renderer import HUDRenderer
from zoom_webcam import ZoomWebcamController
from autofocus_controller import AutoFocusController
from bt_cursor_controller import BTCursorController
from face_recognizer import UNKNOWN_LABEL, FaceRecognizer

log = logging.getLogger(__name__)

HAND_TASK_PATH     = "hand_landmarker.task"
FACE_TASK_PATH     = "face_landmarker.task"
POSE_TASK_PATH     = "pose_landmarker_lite.task"
GESTURE_MODEL_PATH = "landmark_gesture_model.h5"

NUM_HANDS            = 2
FACE_DETECT_INTERVAL = 3
POSE_DETECT_INTERVAL = 2

FACE_SAMPLES_NEEDED = 60
FACE_SAMPLE_DELAY   = 0.05

PREVIEW_QUALITY = 70    # JPEG quality for the browser preview
PREVIEW_FPS     = 15    # cap preview rate; detection still runs full speed


class GestureSession:
    def __init__(self, db, hid_device, ptz, socketio, active_user=None):
        self.db       = db
        self.hid      = hid_device
        self.ptz      = ptz
        self.socketio = socketio

        self.active_user = active_user
        self._running    = False
        self._thread     = None
        self._lock       = threading.Lock()

        # Web-driven face registration state
        self.capture_request  = None    # username being registered, or None
        self._capture_samples = []
        self._capture_last_t  = 0.0

        # Components are built lazily on start() so the web server can
        # boot (and show a useful error page) even if a model file is
        # missing or the camera is busy.
        self.cam       = None
        self.landmarks = None
        self.auth      = None
        self.gestures  = None
        self.cursor    = None
        self.hud       = None
        self.zoom      = None
        self.autofocus = None

        self.last_error = None
        self.fps        = 0.0

    # ── Lifecycle ───────────────────────────────────────────────────────────
    def start(self):
        if self._running:
            return
        self._running = True
        self._thread  = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
        if self.autofocus:
            self.autofocus.close()
        if self.cam:
            self.cam.release()
        if self.landmarks:
            self.landmarks.close()

    def set_active_user(self, username):
        """
        Called on login. Rebuilds the recogniser so only this user's face
        counts as a match — bystanders stay blocked even if they're
        registered in the DB.
        """
        with self._lock:
            if username == self.active_user:
                return   # already tracking this user — cheap to call on every
                         # dashboard load, so a restored session re-syncs too
            self.active_user = username
            if self.auth is not None:
                self.auth = AuthManager(active_user=username,
                                        face_detect_interval=FACE_DETECT_INTERVAL)
                self.hud.num_registered = self.auth.num_registered
        # Tell the PTZ controller who to follow. Without this its
        # active_user stays None, recognised_user == None is never true, and
        # the camera never leaves the LOST state — i.e. it never tracks.
        if self.ptz is not None:
            self.ptz.set_active_user(username)

    # ── Web-driven face registration ────────────────────────────────────────
    def begin_face_capture(self, username):
        with self._lock:
            self.capture_request  = username
            self._capture_samples = []
            self._capture_last_t  = 0.0
        log.info("Face capture started for '%s'", username)

    def cancel_face_capture(self):
        with self._lock:
            self.capture_request  = None
            self._capture_samples = []

    def _handle_face_capture(self, cached_face_lms, now_t):
        """Harvest an embedding from this frame if registration is active."""
        with self._lock:
            username = self.capture_request
        if not username:
            return

        if not cached_face_lms:
            self.socketio.emit("face_capture_progress", {
                "collected": len(self._capture_samples),
                "needed":    FACE_SAMPLES_NEEDED,
                "detected":  False,
            })
            return

        if (now_t - self._capture_last_t) < FACE_SAMPLE_DELAY:
            return
        self._capture_last_t = now_t

        emb = FaceRecognizer.extract_embedding(cached_face_lms[0])
        self._capture_samples.append(emb)
        n = len(self._capture_samples)

        self.socketio.emit("face_capture_progress", {
            "collected": n,
            "needed":    FACE_SAMPLES_NEEDED,
            "detected":  True,
        })

        if n >= FACE_SAMPLES_NEEDED:
            mean_emb  = np.mean(self._capture_samples, axis=0)
            mean_emb /= (np.linalg.norm(mean_emb) or 1.0)   # L2 for cosine sim
            self.db.save_face_embedding(username, mean_emb)

            with self._lock:
                self.capture_request  = None
                self._capture_samples = []

            # Reload so the newly-registered face is recognised immediately,
            # without needing a restart.
            self.auth = AuthManager(active_user=self.active_user,
                                    face_detect_interval=FACE_DETECT_INTERVAL)
            self.hud.num_registered = self.auth.num_registered

            log.info("Face registered for '%s'", username)
            self.socketio.emit("face_capture_done", {"username": username})

    # ── Main loop ───────────────────────────────────────────────────────────
    def _build(self):
        self.auth      = AuthManager(active_user=self.active_user,
                                     face_detect_interval=FACE_DETECT_INTERVAL)
        self.gestures  = GestureEngine(GESTURE_MODEL_PATH, num_hands=NUM_HANDS)
        self.cursor    = BTCursorController(self.hid)
        self.hud       = HUDRenderer(num_registered=self.auth.num_registered)
        self.zoom      = ZoomWebcamController(
            enabled=True,
            control_zone_margin=BTCursorController.CAM_MARGIN)
        self.landmarks = LandmarkPipeline(
            HAND_TASK_PATH, FACE_TASK_PATH, POSE_TASK_PATH,
            num_hands=NUM_HANDS,
            face_detect_interval=FACE_DETECT_INTERVAL,
            pose_detect_interval=POSE_DETECT_INTERVAL,
        )
        self.cam = CameraManager(width=640, height=480, fps=30)

        # Continuous ("digital-camera style") autofocus, sharing the PTZ's
        # Focuser + I2C lock. start() kicks off the one-time full-range sweep
        # on its own thread; from then on it just watches sharpness (fed by
        # report_frame below) and hunts directionally only when the image
        # goes soft. If there's no focus hardware / it errors, the loop keeps
        # running — autofocus is best-effort, never fatal.
        try:
            self.autofocus = AutoFocusController(self.ptz.focuser,
                                                 self.ptz.io_lock)
            self.autofocus.start()
        except Exception:
            log.exception("Autofocus unavailable — continuing without it")
            self.autofocus = None

    def _run(self):
        try:
            self._build()
        except Exception as e:
            self.last_error = str(e)
            log.exception("Gesture session failed to start")
            self.socketio.emit("session_error", {"error": str(e)})
            self._running = False
            return

        start_time     = time.perf_counter()
        prev_time      = start_time
        last_preview_t = 0.0
        frame_n        = 0

        while self._running:
            ok, frame = self.cam.read()
            if not ok:
                time.sleep(0.01)
                continue

            frame_n += 1
            frame = cv2.flip(frame, 1)

            # Hand the raw frame to autofocus (cheap: it just measures a
            # centre-ROI sharpness scalar; all focus motor I/O is on its
            # own thread).
            if self.autofocus is not None:
                self.autofocus.report_frame(frame)

            now_t     = time.perf_counter()
            dt        = now_t - prev_time
            prev_time = now_t
            if dt > 0:
                self.fps = 0.9 * self.fps + 0.1 / dt

            h, w  = frame.shape[:2]
            rgb_c = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            ts_ms = int((now_t - start_time) * 1000)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_c)

            (h_result, cached_face_lms,
             cached_pose_lms, face_updated) = self.landmarks.detect(
                mp_image, frame_n, ts_ms)

            if face_updated:
                self.auth.update(cached_face_lms, now_t)
                self._handle_face_capture(cached_face_lms, now_t)

            # PTZ auto-tracking. In manual mode the controller is disabled
            # (see ptz_manual.ManualPTZ), so these calls become no-ops
            # rather than fighting the web UI's D-pad.
            self.ptz.update(
                nose_pos=self.auth.face_nose_pos,
                recognised_user=self.auth.recognised_user,
                is_registered_face_visible=self.auth.is_registered_face_visible,
                pose_landmarks=cached_pose_lms,
            )

            # Pick the recognised user's pose for auto-zoom tracking
            tracked_pose = None
            if self.auth.is_registered_face_visible and cached_pose_lms:
                best = None
                for pose_lms in cached_pose_lms:
                    p_nose = np.array([pose_lms[0].x, pose_lms[0].y], dtype=np.float32)
                    d = float(np.linalg.norm(p_nose - self.auth.face_nose_pos))
                    if d < 0.18 and (best is None or d < best):
                        best, tracked_pose = d, pose_lms
            self.zoom.update(tracked_pose, w, h)

            # Crop FIRST on the clean frame, then draw HUD zoom-aware on top
            display = self.zoom.apply(frame)

            self.hud.draw_face_boxes(display, self.auth.face_rec, cached_face_lms,
                                     self.auth.recognised_user, UNKNOWN_LABEL,
                                     w, h, zoom=self.zoom)
            self.hud.draw_pose_skeletons(display, cached_pose_lms,
                                         self.auth.user_active,
                                         self.auth.face_nose_pos,
                                         self.auth.recog_score,
                                         self.auth.face_size, w, h, zoom=self.zoom)
            self.hud.draw_control_zone(display, self.auth.limb_mode,
                                       BTCursorController.CAM_MARGIN,
                                       w, h, zoom=self.zoom)
            self.hud.draw_crosshair(display, self.auth.limb_mode,
                                    self.auth.face_nose_pos, w, h, zoom=self.zoom)

            hand_action_strs = []
            if h_result.hand_landmarks:
                for hand_id, hand in enumerate(h_result.hand_landmarks):
                    wrist   = np.array([hand[0].x, hand[0].y], dtype=np.float32)
                    raw_pts = np.array([[lm.x, lm.y, lm.z] for lm in hand],
                                       dtype=np.float32)
                    sxyz = self.gestures.smooth_hand(hand_id, raw_pts)
                    sp   = (sxyz[:, :2] * (w, h)).astype(np.int32)

                    hand_is_user = (self.auth.user_active and
                                    self.auth.face_nose_pos is not None and
                                    float(np.linalg.norm(
                                        wrist - self.auth.face_nose_pos)) < 0.70)

                    line_col = (255, 0, 0) if hand_is_user else (0, 0, 200)
                    dot_col  = (0, 255, 0) if hand_is_user else (0, 0, 200)
                    self.hud.draw_hand(display, FINGER_CHAINS, sp,
                                       line_col, dot_col, zoom=self.zoom)

                    confirmed, confidence = self.gestures.classify(hand_id, sxyz)

                    if not hand_is_user:                     col = (0, 0, 200)
                    elif confirmed == 'NO ACTION':           col = (120, 120, 120)
                    elif confirmed in CURSOR_FREEZE_ACTIONS: col = (255, 200, 0)
                    elif confidence >= 0.75:                 col = (0, 200, 0)
                    else:                                    col = (0, 165, 255)

                    tag = (self.auth.recognised_user[:3].upper()
                           if self.auth.recognised_user != UNKNOWN_LABEL else 'UNK')
                    s = f"H{hand_id}[{tag if hand_is_user else 'UNK'}]: {confirmed} ({confidence:.0%})"
                    if not hand_is_user:
                        s += "  [blocked]"
                    hand_action_strs.append((s, col))

                    tip_px = (int(sp[8, 0]), int(sp[8, 1]))
                    if hand_is_user and hand_id == 0:
                        self.cursor.handle_action(confirmed, hand_id, sxyz[8, :2])
                        if confirmed == 'MOVE':
                            self.hud.draw_move_indicator(
                                display, tip_px, self.auth.limb_mode,
                                self.auth.face_nose_pos, w, h, zoom=self.zoom)
                    elif not hand_is_user:
                        self.hud.draw_blocked_hand(display, tip_px, zoom=self.zoom)

            detected = (set(range(len(h_result.hand_landmarks)))
                        if h_result.hand_landmarks else set())
            self.gestures.forget_stale_hands(detected)

            self.hud.draw_status_hud(display, self.fps, len(cached_pose_lms),
                                     len(cached_face_lms), hand_action_strs, w)
            self.hud.draw_auth_banner(display, self.auth.auth_temp,
                                      self.auth.user_active,
                                      self.auth.grace_remaining(now_t),
                                      AuthManager.TEMP_ACTIVATE, w, h)
            self.ptz.draw_debug_hud(display)
            display = self.zoom.draw_debug(display)

            # ── Push preview to the browser (rate-limited) ──────────────────
            if (now_t - last_preview_t) >= (1.0 / PREVIEW_FPS):
                last_preview_t = now_t
                ok_enc, buf = cv2.imencode(
                    ".jpg", display,
                    [int(cv2.IMWRITE_JPEG_QUALITY), PREVIEW_QUALITY])
                if ok_enc:
                    self.socketio.emit("preview_frame", {
                        "image": base64.b64encode(buf).decode("ascii"),
                        "telemetry": self.telemetry(),
                    })

        log.info("Gesture loop exited")

    # ── Status for the web UI ───────────────────────────────────────────────
    def telemetry(self):
        return {
            "fps":            round(self.fps, 1),
            "zoom":           round(self.zoom.zoom, 2) if self.zoom else 1.0,
            "zoom_mode":      self.zoom.last_mode if self.zoom else None,
            "zoom_enabled":   self.zoom.enabled if self.zoom else False,
            "user_active":    self.auth.user_active if self.auth else False,
            "recognised":     self.auth.recognised_user if self.auth else None,
            "auth_temp":      round(self.auth.auth_temp, 2) if self.auth else 0.0,
            "hid_connected":  self.hid.connected,
            "hid_peer":       self.hid.peer_address,
            "ptz_state":      self.ptz.get_state() if self.ptz else None,
            "focus":          self.autofocus.focus_position if self.autofocus else None,
            "focus_mode":     self.autofocus.mode if self.autofocus else None,
            "focus_state":    self.autofocus.state if self.autofocus else None,
        }

    # ── Digital zoom controls (the web UI's zoom slider) ────────────────────
    def set_zoom_enabled(self, enabled):
        if self.zoom:
            self.zoom.enabled = bool(enabled)
        return self.zoom.enabled if self.zoom else False

    def set_zoom_cap(self, max_zoom):
        """
        Raise/lower the absolute digital zoom ceiling from the web slider.
        The arm-reach control-zone cap still applies on top of this, but it
        is now distance-adaptive (relaxes toward ZoomWebcamController.
        FAR_MAX_ZOOM as the subject moves away), so raising this ceiling
        DOES take visible effect at range — it governs how tightly a
        distant presenter can be framed.
        """
        if self.zoom:
            self.zoom.MAX_ZOOM = float(max_zoom)
        return max_zoom

    # ── Autofocus controls (the web UI's focus panel) ───────────────────────
    def set_focus_auto(self, enabled):
        """Toggle continuous autofocus. Off = hold focus / manual control."""
        if self.autofocus:
            return self.autofocus.set_auto(bool(enabled))
        return False

    def set_focus_manual(self, value):
        """Move focus to an absolute value and switch to manual control."""
        if self.autofocus:
            self.autofocus.set_manual_focus(int(value))
        return value

    def trigger_refocus(self):
        """Force a one-shot autofocus hunt right now."""
        if self.autofocus:
            self.autofocus.trigger_refocus()

    def focus_status(self):
        return self.autofocus.status() if self.autofocus else None
