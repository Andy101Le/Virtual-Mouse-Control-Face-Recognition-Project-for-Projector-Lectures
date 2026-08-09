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
import os
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
from face_embedder import FaceEmbedder, AsyncFaceEmbedder
from control_toggle import ControlToggle, is_toggle_pose, is_peace_sign

log = logging.getLogger(__name__)

HAND_TASK_PATH     = "hand_landmarker.task"
FACE_TASK_PATH     = "face_landmarker.task"
POSE_TASK_PATH     = "pose_landmarker_lite.task"
GESTURE_MODEL_PATH = "landmark_gesture_model.h5"

NUM_HANDS            = 2
FACE_DETECT_INTERVAL = 3
# Pose is the most expensive detector in the loop (~37 ms/call, vs 39 ms for
# the hand landmarker that runs every frame), and it feeds only slow, smoothed
# consumers: the PTZ's coarse fallback when the face is lost, the auto-zoom
# framing, and the detection crop's anchor. All three read the CACHED result
# every frame, so their smoothing rates are unchanged by detecting less often
# — only the anchor's freshness degrades, by at most 200 ms. Rate rather than
# frame interval; see LandmarkPipeline.
POSE_DETECT_HZ = 5.0

FACE_SAMPLES_NEEDED = 60
FACE_SAMPLE_DELAY   = 0.05

PREVIEW_QUALITY = 70    # JPEG quality for the browser preview
PREVIEW_FPS     = 15    # cap preview rate; detection still runs full speed

# ── Long-range capture geometry ─────────────────────────────────────────────
# Capture at 4x the old pixel count (same 4:3 aspect, so every normalized
# coordinate, the control-zone margin, and the digital-zoom math behave
# exactly as before), but never feed MediaPipe more than DETECT_W of width:
# while someone is tracked, detection runs on a native-resolution crop
# around them (a "detector telephoto" — ~2x the pixels on a distant face
# for the same inference cost); with nobody tracked it runs on the full
# frame downscaled to DETECT_W, which is identical to the old 640x480 path.
#
# The extra resolution is not free: the added preview downscale, detection
# crop and larger autofocus ROI measure out at roughly 5-8 ms per frame on a
# Pi 5, against a 33 ms budget at 30 fps. That buys ~2x the pixels on a
# distant face, which is the whole point — but if the frame rate drops
# further than you're willing to trade, set CAPTURE_HIRES=0 to fall straight
# back to the old 640x480 single-resolution path (which also disables the
# detection crop, since at 640 wide there are no spare pixels to crop into).
# Watch the fps figure in the dashboard telemetry to decide.
_HIRES = os.environ.get("CAPTURE_HIRES", "1") != "0"

CAPTURE_W, CAPTURE_H = (1280, 960) if _HIRES else (640, 480)
DETECT_W             = 640
PREVIEW_W            = 640    # browser preview width — keeps JPEG/socket
                              # load the same as the old full-frame preview

# The crop shrinks as the face shrinks: full frame at/above
# CROP_FULL_FACE_SIZE (close range — hands at the frame edges must stay
# detectable), down to CROP_MIN_SCALE of the frame for a distant subject.
CROP_FULL_FACE_SIZE = 0.20
CROP_MIN_SCALE      = 0.50
ROI_SMOOTH          = 0.35   # EMA on the crop centre so it glides, not jumps
ROI_LOST_SECONDS    = 3.0    # nobody seen this long -> full-frame search

# A crop is a blindfold: whatever falls outside it cannot be detected, so it
# cannot correct the crop either. Two escapes from that trap:
#
#   - If the active user hasn't been RECOGNISED for this long while a crop is
#     active, the crop is assumed to be pointed at the wrong person (a
#     bystander's pose can hold the anchor indefinitely — their pose keeps
#     "something seen" true, so ROI_LOST_SECONDS never fires). Drop to the
#     full frame and re-search.
#   - After such a reset, stay full-frame for a beat before cropping again,
#     otherwise the next frame just re-locks onto the same bystander.
ROI_UNRECOGNISED_SECONDS = 4.0
ROI_RESEARCH_SECONDS     = 1.5

# How far past the control zone's half-width a hand may sit and still count
# as the tracked user's. Slack above 1.0 because the zone is deliberately
# sized for a COMFORTABLE extension, and a hand raised above the head or
# stretched past it is still plainly theirs.
HAND_OWNERSHIP_SLACK = 1.6

# Pose head landmarks (MediaPipe pose: 0 nose, 2/5 eyes, 7/8 ears) and the
# factor converting their span into the face-mesh bbox diagonal that
# get_face_size() reports. Only the ratio's rough scale matters: it exists so
# the crop keeps tightening as someone walks away and their face stops being
# detectable at all — without it the last measured face size sticks forever
# and the crop never engages in exactly the walk-away case it's built for.
POSE_HEAD_IDS         = (0, 2, 5, 7, 8)
POSE_FACE_SIZE_RATIO  = 1.7
POSE_HEAD_VIS_THRESH  = 0.35
# Fallback when even the head landmarks are too weak to measure: shrink the
# estimate each face interval so the crop still tightens rather than freezing.
ROI_FACE_SIZE_DECAY   = 0.90


class GestureSession:
    def __init__(self, db, hid_device, ptz, socketio, active_user=None):
        self.db       = db
        self.hid      = hid_device
        self.ptz      = ptz
        self.socketio = socketio

        self.active_user = active_user
        self._running    = False
        self._suspended  = False
        self._thread     = None
        self._lock       = threading.Lock()

        # Web-driven face registration state
        self.capture_request  = None    # username being registered, or None
        self._capture_samples = []
        self._capture_last_t  = 0.0

        # Tracked-crop ("detector telephoto") state — where in the full
        # frame the person of interest is, and how big their face is.
        self._roi_center    = None    # (x, y) normalized, or None
        self._roi_face_size = None    # EMA of normalized face bbox diagonal
        self._roi_last_seen = 0.0
        self._roi_recognised_t = 0.0  # last time the ACTIVE user was recognised
        self._roi_search_until = 0.0  # stay full-frame until this time

        # Last good body scale for the cursor control zone, held across the
        # frames where face detection drops out.
        self._zone_anchor = None
        self._zone_size   = None

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
        self.control_toggle = ControlToggle(enabled=True)
        self.embedder  = None    # sync SFace, for enrollment
        self.identifier = None   # async SFace worker, for live recognition

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
        if self.identifier:
            self.identifier.close()
        if self.cam:
            self.cam.release()
        if self.landmarks:
            self.landmarks.close()

    def set_suspended(self, suspended):
        """
        True = idle the whole pipeline: frames are still read (so the
        camera buffer stays fresh and resume is instant) but nothing is
        processed — no detection, no face tracking, no PTZ following, no
        HID cursor, no preview. Driven by the web server's viewer
        presence: suspended once nobody with a logged-in dashboard tab is
        connected, resumed the moment one connects.
        """
        if suspended == self._suspended:
            return
        self._suspended = suspended
        log.info("Gesture pipeline %s",
                 "SUSPENDED — no logged-in viewer" if suspended else "resumed")

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

    def _handle_face_capture(self, cached_face_lms, frame, now_t):
        """Harvest an embedding from this frame if registration is active."""
        with self._lock:
            username = self.capture_request
        if not username:
            return

        if self.embedder is None:
            # Nothing usable can be stored without the recognition model, and
            # writing a placeholder would mark the user "registered" while
            # leaving them permanently unrecognisable.
            self.socketio.emit("face_capture_error", {
                "error": "Face recognition model missing — run "
                         "setup/get_models.sh, then restart."})
            with self._lock:
                self.capture_request  = None
                self._capture_samples = []
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

        # Enrollment embeds inline rather than through the async worker: it
        # runs a handful of times during a deliberate registration flow, and
        # a sample must correspond to exactly the frame it was taken from.
        emb = self.embedder.embed(frame, cached_face_lms[0])
        if emb is None:
            self.socketio.emit("face_capture_progress", {
                "collected": len(self._capture_samples),
                "needed":    FACE_SAMPLES_NEEDED,
                "detected":  False,
            })
            return
        self._capture_samples.append(emb)
        n = len(self._capture_samples)

        self.socketio.emit("face_capture_progress", {
            "collected": n,
            "needed":    FACE_SAMPLES_NEEDED,
            "detected":  True,
        })

        if n >= FACE_SAMPLES_NEEDED:
            # Keep several diverse samples rather than one averaged point —
            # a face varies with pose and expression, and max-similarity over
            # a spread of samples tolerates that far better than distance to
            # a mean that represents no actual view of the face.
            samples = FaceRecognizer.pack_samples(self._capture_samples)
            if samples is None:
                self.socketio.emit("face_capture_error", {
                    "error": "No usable face samples captured — try again."})
                with self._lock:
                    self.capture_request  = None
                    self._capture_samples = []
                return
            self.db.save_face_embedding(username, samples)

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

    # ── Zoom/autofocus coordination (PTZ worker-thread callbacks) ───────────
    def _on_zoom_step(self, _zoom_value):
        if self.autofocus is not None:
            self.autofocus.suppress(1.2)

    def _on_zoom_settled(self, _units_moved):
        # trigger_refocus() deliberately bypasses the manual-focus guard,
        # because its original caller was the user pressing "Focus now". Wired
        # to an automatic hardware event that bypass becomes a bug: every zoom
        # step would rack the lens away from the position the user parked with
        # the slider, while the UI still reported MANUAL. Zoom moving is a
        # reason to re-focus only if focus is ours to drive.
        if self.autofocus is not None and self.autofocus.auto_enabled:
            self.autofocus.trigger_refocus()

    # ── Tracked-crop ("detector telephoto") helpers ─────────────────────────
    def _update_roi(self, cached_pose_lms, now_t):
        """
        Keep the crop centre on the person of interest. Priority: the
        recognised user's face, then any detected face, then the most
        visible pose (bodies stay detectable far beyond faces — this is
        what lets a distant user be ACQUIRED cold: pose aims the crop,
        the crop makes their small face resolvable, recognition takes it
        from there). This picks where detection LOOKS — it grants nothing:
        identity still comes only from the recognition gate.
        """
        anchor = None
        if self.auth.face_nose_pos is not None:
            anchor = (float(self.auth.face_nose_pos[0]),
                      float(self.auth.face_nose_pos[1]))
            if self.auth.face_size:
                self._blend_face_size(float(self.auth.face_size))
            if self.auth.is_registered_face_visible:
                # Only an actual recognition clears the bystander watchdog.
                # "A face is present" is not evidence it's the right face.
                self._roi_recognised_t = now_t
        elif cached_pose_lms:
            # Prefer the pose closest to where the anchor already is, rather
            # than the most visible one. A bystander walking in front will
            # usually out-score the tracked user on visibility, and picking
            # by visibility hands them the crop; picking by continuity keeps
            # it on the person we were already following.
            best = None
            for pose_lms in cached_pose_lms:
                if not len(pose_lms) or pose_lms[0].visibility <= 0.4:
                    continue
                cand = (float(pose_lms[0].x), float(pose_lms[0].y))
                if self._roi_center is None:
                    key = -pose_lms[0].visibility     # cold start: most visible
                else:
                    key = ((cand[0] - self._roi_center[0]) ** 2 +
                           (cand[1] - self._roi_center[1]) ** 2)
                if best is None or key < best[0]:
                    best, anchor = (key, cand), cand
                    head = self._pose_face_size(pose_lms)
            if anchor is not None:
                # The face is no longer detectable but the body still is —
                # size the crop from the head span so it keeps tightening as
                # they recede. Without this the last good face size sticks
                # and the crop never engages while someone walks away.
                self._blend_face_size(head if head else
                                      (self._roi_face_size or CROP_FULL_FACE_SIZE)
                                      * ROI_FACE_SIZE_DECAY)

        if anchor is not None:
            if self._roi_center is None:
                self._roi_center = anchor
            else:
                a = ROI_SMOOTH
                self._roi_center = (a * anchor[0] + (1 - a) * self._roi_center[0],
                                    a * anchor[1] + (1 - a) * self._roi_center[1])
            self._roi_last_seen = now_t
        elif (self._roi_center is not None and
              (now_t - self._roi_last_seen) > ROI_LOST_SECONDS):
            self._roi_center    = None   # back to full-frame searching
            self._roi_face_size = None

        # Bystander watchdog. A crop that has been up for a while without the
        # active user being recognised inside it is, by definition, looking in
        # the wrong place — and it can never discover that on its own, because
        # the user is outside the only region being searched.
        if (self._roi_center is not None and
                (now_t - self._roi_recognised_t) > ROI_UNRECOGNISED_SECONDS):
            self._roi_center      = None
            self._roi_face_size   = None
            self._roi_recognised_t = now_t          # restart the watchdog
            self._roi_search_until = now_t + ROI_RESEARCH_SECONDS

    def _blend_face_size(self, value):
        """EMA the face-size estimate the crop is scaled from."""
        self._roi_face_size = (value if self._roi_face_size is None else
                               ROI_SMOOTH * value +
                               (1 - ROI_SMOOTH) * self._roi_face_size)

    @staticmethod
    def _pose_face_size(pose_lms):
        """Face size estimated from pose head landmarks, on the same scale
        get_face_size() reports, or None if the head isn't visible enough."""
        pts = [(lm.x, lm.y) for i in POSE_HEAD_IDS
               for lm in (pose_lms[i],) if lm.visibility > POSE_HEAD_VIS_THRESH]
        if len(pts) < 2:
            return None
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        span = ((max(xs) - min(xs)) ** 2 + (max(ys) - min(ys)) ** 2) ** 0.5
        return span * POSE_FACE_SIZE_RATIO if span > 1e-4 else None

    def _detect_window(self, w, h, now_t=0.0):
        """
        The full-frame pixel rect detection should run on this frame:
        (x0, y0, cw, ch), or None for the whole frame. Same aspect ratio
        as the frame, so LandmarkPipeline's remap uses one uniform scale
        and hand/face geometry is preserved exactly.
        """
        if now_t < self._roi_search_until:
            # Forced full-frame re-search after the bystander watchdog fired.
            return None
        if self._roi_center is None or not _HIRES:
            # At 640x480 capture the frame IS the detector's native input —
            # cropping into it would hand MediaPipe fewer pixels, not more,
            # so the telephoto trick only makes sense above DETECT_W.
            return None
        size  = self._roi_face_size
        # No face size yet (pose-only anchor) means the face is too small
        # to detect — use the tightest crop to give it the most pixels.
        scale = (size / CROP_FULL_FACE_SIZE) if size else CROP_MIN_SCALE
        scale = min(max(scale, CROP_MIN_SCALE), 1.0)
        if scale >= 0.999:
            return None
        cw = int(round(w * scale))
        ch = int(round(h * scale))
        x0 = int(round(self._roi_center[0] * w - cw / 2))
        y0 = int(round(self._roi_center[1] * h - ch / 2))
        x0 = max(0, min(x0, w - cw))
        y0 = max(0, min(y0, h - ch))
        return (x0, y0, cw, ch)

    # ── Main loop ───────────────────────────────────────────────────────────
    def _build(self):
        self.auth      = AuthManager(active_user=self.active_user,
                                     face_detect_interval=FACE_DETECT_INTERVAL)
        self.gestures  = GestureEngine(GESTURE_MODEL_PATH)
        self.cursor    = BTCursorController(self.hid)
        self.hud       = HUDRenderer(num_registered=self.auth.num_registered)
        self.zoom      = ZoomWebcamController(
            enabled=True,
            control_zone_margin=BTCursorController.CAM_MARGIN)
        self.landmarks = LandmarkPipeline(
            HAND_TASK_PATH, FACE_TASK_PATH, POSE_TASK_PATH,
            num_hands=NUM_HANDS,
            face_detect_interval=FACE_DETECT_INTERVAL,
            pose_detect_hz=POSE_DETECT_HZ,
        )
        self.cam = CameraManager(width=CAPTURE_W, height=CAPTURE_H, fps=30)

        # Face identity. One embedding costs ~56 ms, so live recognition runs
        # on a worker thread and the loop consumes whatever it last published;
        # enrollment uses the synchronous one, where a sample must match the
        # exact frame it came from. A missing model is not fatal, but it does
        # mean nobody can be recognised — fail loudly and keep the pipeline
        # up rather than authenticating people we cannot actually identify.
        try:
            self.embedder   = FaceEmbedder()
            self.identifier = AsyncFaceEmbedder()
            log.info("Face recognition (SFace) ready")
        except FileNotFoundError as e:
            self.embedder = self.identifier = None
            log.error("%s — face recognition DISABLED; nobody will be "
                      "recognised until the model is installed.", e)
        except Exception:
            self.embedder = self.identifier = None
            log.exception("Face recognition unavailable — continuing without it")

        if self.auth.face_rec.stale_users:
            log.warning("Re-registration required for: %s",
                        ", ".join(self.auth.face_rec.stale_users))

        # Continuous ("digital-camera style") autofocus, sharing the PTZ's
        # Focuser + I2C lock. start() kicks off the one-time full-range sweep
        # on its own thread; from then on it just watches sharpness (fed by
        # report_frame below) and hunts directionally only when the image
        # goes soft. If there's no focus hardware / it errors, the loop keeps
        # running — autofocus is best-effort, never fatal.
        self.autofocus = None
        if self.ptz.focuser is None:
            # PTZController already warned on the command line and disabled
            # itself (no I2C hardware) — there is no focus motor to drive
            # either, so skip starting a worker thread that would just
            # crash on its first hardware access.
            log.info("No PTZ hardware — autofocus unavailable")
        else:
            try:
                self.autofocus = AutoFocusController(self.ptz.focuser,
                                                     self.ptz.io_lock)
                self.autofocus.start()
            except Exception:
                log.exception("Autofocus unavailable — continuing without it")
                self.autofocus = None

        # Optical zoom <-> autofocus coordination: zoom moves shift the
        # focus plane, so while the zoom motor steps, AF must not chase the
        # transient blur (suppress), and once it settles it should re-find
        # the peak deliberately (one forced hunt, which bypasses suppress).
        self.ptz.on_zoom_step    = self._on_zoom_step
        self.ptz.on_zoom_settled = self._on_zoom_settled

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

            if self._suspended:
                # Nobody logged-in is watching — drop the frame unprocessed.
                # Reading (above) still happens every pass so the sensor
                # pipeline doesn't back up and resume shows a live image
                # immediately, but no detection/tracking/cursor runs.
                time.sleep(0.05)
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
            ts_ms = int((now_t - start_time) * 1000)

            # Choose what MediaPipe sees this frame: a native-resolution
            # crop around the tracked person (transform maps its landmarks
            # back to full-frame coordinates inside LandmarkPipeline), or
            # the whole frame downscaled to DETECT_W when nobody's tracked.
            win = self._detect_window(w, h, now_t)
            if win is not None:
                x0, y0, cw, ch = win
                det       = frame[y0:y0 + ch, x0:x0 + cw]
                transform = (x0 / w, y0 / h, cw / w)
            else:
                det       = frame
                transform = None
            if det.shape[1] > DETECT_W:
                dh  = max(1, int(round(det.shape[0] * DETECT_W / det.shape[1])))
                det = cv2.resize(det, (DETECT_W, dh),
                                 interpolation=cv2.INTER_AREA)
            rgb_c    = cv2.cvtColor(det, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_c)

            (h_result, cached_face_lms,
             cached_pose_lms, face_updated) = self.landmarks.detect(
                mp_image, frame_n, ts_ms, transform=transform)

            if face_updated:
                # Identity runs off-thread. Submit this frame's faces, then
                # consume whatever the worker last finished — typically one
                # or two face-intervals old, which the auth temperature
                # debounce absorbs without noticing. Landmarks are already
                # remapped to full-frame coords, so the FULL frame is the
                # right image to align against, not the detection crop.
                embeddings = None
                if self.identifier is not None:
                    self.identifier.submit(frame, cached_face_lms,
                                           anchor=self._roi_center,
                                           token=frame_n)
                    published = self.identifier.result()
                    embeddings = published[1] if published else None
                self.auth.update(cached_face_lms, now_t, embeddings=embeddings)
                self._handle_face_capture(cached_face_lms, frame, now_t)

            # Follow the person with the detection crop, and point the
            # autofocus sharpness window at them too.
            self._update_roi(cached_pose_lms, now_t)
            if self.autofocus is not None:
                self.autofocus.set_roi_center(self._roi_center)

            # PTZ auto-tracking. In manual mode the controller is disabled
            # (see ptz_manual.ManualPTZ), so these calls become no-ops
            # rather than fighting the web UI's D-pad.
            self.ptz.update(
                nose_pos=self.auth.face_nose_pos,
                recognised_user=self.auth.recognised_user,
                is_registered_face_visible=self.auth.is_registered_face_visible,
                pose_landmarks=cached_pose_lms,
                face_size=self.auth.face_size,
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
            # Everything visual happens at preview resolution: the full-res
            # frame exists for detection crops and autofocus only. The
            # digital zoom + every HUD overlay work in normalized coords,
            # so they land identically on the smaller canvas — and JPEG/
            # socket cost stays what it was at 640x480 capture.
            pw = min(PREVIEW_W, w)
            ph = int(round(h * pw / w))
            preview = (cv2.resize(frame, (pw, ph),
                                  interpolation=cv2.INTER_AREA)
                       if w > pw else frame)

            # Optical zoom gets first refusal on any magnification, because
            # it costs no image quality; the digital crop only adds what the
            # lens has run out of travel to provide. Feeding the motor's
            # state in here is what keeps the two loops from both chasing
            # the same "person looks small" error at once.
            self.zoom.set_optical(self.ptz.get_zoom_state() if self.ptz else None)
            self.zoom.update(tracked_pose, pw, ph)

            # Size the cursor control zone to the user's apparent reach. A
            # fixed zone asked for the same absolute hand sweep at every
            # distance, so a fully extended arm at range only ever reached
            # the middle of the host screen.
            # Face detection flickers at range, and face_size is None on any
            # frame where nothing was found. Reverting to the fallback zone on
            # each of those would drag the mapping — and the cursor with it —
            # every time a distant face dropped for a beat. So hold the last
            # good measurement for as long as the user counts as active;
            # AuthManager's grace period already decides when they're really
            # gone, and this reuses that judgement rather than inventing one.
            if self.auth.face_nose_pos is not None and self.auth.face_size:
                self._zone_anchor = self.auth.face_nose_pos
                self._zone_size   = self.auth.face_size
            if self.auth.user_active and self._zone_anchor is not None:
                # Hand the detection bounds over too: the crop follows the
                # face, so without this the zone can extend below what
                # MediaPipe is looking at and the bottom of the host screen
                # maps to hand positions that are never detected.
                if win is not None:
                    dx0, dy0, dcw, dch = win
                    det_bounds = (dx0 / w, dy0 / h,
                                  (dx0 + dcw) / w, (dy0 + dch) / h)
                else:
                    det_bounds = None
                self.cursor.set_control_zone(self._zone_anchor, self._zone_size,
                                             detect_window=det_bounds)
            else:
                self._zone_anchor = self._zone_size = None
                self.cursor.set_control_zone(None, None)
            # Keep the display crop honest about what has to stay visible.
            self.zoom.set_control_zone(self.cursor.zone)

            # Crop FIRST on the clean frame, then draw HUD zoom-aware on top
            display = self.zoom.apply(preview)

            self.hud.draw_face_boxes(display, self.auth.face_rec, cached_face_lms,
                                     self.auth.recognised_user, UNKNOWN_LABEL,
                                     pw, ph, zoom=self.zoom)
            self.hud.draw_pose_skeletons(display, cached_pose_lms,
                                         self.auth.user_active,
                                         self.auth.face_nose_pos,
                                         self.auth.recog_score,
                                         self.auth.face_size, pw, ph, zoom=self.zoom)
            self.hud.draw_control_zone(display, self.auth.limb_mode,
                                       self.cursor.zone,
                                       pw, ph, zoom=self.zoom)
            self.hud.draw_crosshair(display, self.auth.limb_mode,
                                    self.auth.face_nose_pos, pw, ph, zoom=self.zoom)

            hand_action_strs = []
            # Both of the user's hands act independently (e.g. left clicks
            # while right moves), but only ONE hand may drive the cursor per
            # frame — first MOVE hand in detection order claims it, so two
            # open palms don't yank the pointer back and forth.
            move_claimed = False
            user_hand_pts = []
            if h_result.hand_landmarks:
                for hand_id, hand in enumerate(h_result.hand_landmarks):
                    wrist   = np.array([hand[0].x, hand[0].y], dtype=np.float32)
                    raw_pts = np.array([[lm.x, lm.y, lm.z] for lm in hand],
                                       dtype=np.float32)
                    sxyz = self.gestures.smooth_hand(hand_id, raw_pts)
                    sp   = (sxyz[:, :2] * (pw, ph)).astype(np.int32)

                    # "Is this hand the tracked user's?" scales with how big
                    # they appear, for the same reason the control zone does.
                    # A fixed 0.70 normalized radius was most of the room at
                    # 20 ft — wide enough to adopt a bystander's hand — while
                    # being comparatively tight up close. Reach plus a margin
                    # is the honest bound: a hand further from you than your
                    # own arm can stretch is not yours.
                    own_radius = self.cursor.reach_radius * HAND_OWNERSHIP_SLACK
                    hand_is_user = (self.auth.user_active and
                                    self.auth.face_nose_pos is not None and
                                    float(np.linalg.norm(
                                        wrist - self.auth.face_nose_pos)) < own_radius)

                    if hand_is_user:
                        # Collected for the open-palm master switch below,
                        # which must only ever answer to the tracked user.
                        user_hand_pts.append(sxyz)

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
                    # A hand making a switch gesture is talking to the master
                    # switch, not steering — see control_toggle.is_toggle_pose.
                    talking_to_switch = hand_is_user and is_toggle_pose(sxyz)

                    # Peace sign = middle click, EXCEPT while the master
                    # switch is using it as its own reset step. Note the
                    # click is dispatched here rather than through the Keras
                    # model, which has no middle-click class; the pose is
                    # read geometrically like the switch gestures are.
                    peace_for_click = (hand_is_user
                                       and self.control_toggle.enabled
                                       and not self.control_toggle.owns_peace_sign
                                       and is_peace_sign(sxyz))
                    if self.cursor.peace_click(hand_id, peace_for_click, now_t):
                        hand_action_strs.append(
                            (f"H{hand_id}: MIDDLE CLICK", (0, 200, 0)))

                    if (hand_is_user and self.control_toggle.enabled
                            and not talking_to_switch):
                        if confirmed == 'MOVE':
                            if not move_claimed:
                                move_claimed = True
                                self.cursor.handle_action(confirmed, hand_id,
                                                          sxyz[8, :2])
                                self.hud.draw_move_indicator(
                                    display, tip_px, self.auth.limb_mode,
                                    self.auth.face_nose_pos, pw, ph, zoom=self.zoom)
                        else:
                            # Clicks/scrolls dispatch from ANY user hand —
                            # per-hand cooldowns in the cursor controller
                            # keep them from double-firing.
                            self.cursor.handle_action(confirmed, hand_id,
                                                      sxyz[8, :2])
                    elif not hand_is_user:
                        self.hud.draw_blocked_hand(display, tip_px, zoom=self.zoom)

            # Master switch: an open palm held by the tracked user toggles all
            # control output. Evaluated AFTER the hand loop so it sees every
            # hand this frame, and outside the enabled check so the gesture
            # that switches control back ON is still watched while it's off.
            if self.control_toggle.update(user_hand_pts, now_t,
                                          nose_pos=self.auth.face_nose_pos,
                                          face_size=self.auth.face_size):
                state = "ENABLED" if self.control_toggle.enabled else "DISABLED"
                log.info("Controls %s by open-palm gesture", state)
                self.socketio.emit("controls_toggled",
                                   self.control_toggle.status())

            detected = (set(range(len(h_result.hand_landmarks)))
                        if h_result.hand_landmarks else set())
            self.gestures.forget_stale_hands(detected)

            self.hud.draw_status_hud(display, self.fps, len(cached_pose_lms),
                                     len(cached_face_lms), hand_action_strs, pw)
            self.hud.draw_auth_banner(display, self.auth.auth_temp,
                                      self.auth.user_active,
                                      self.auth.grace_remaining(now_t),
                                      AuthManager.TEMP_ACTIVATE, pw, ph)
            self.hud.draw_control_switch(display, self.control_toggle, pw, ph)
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
            "optical_zoom":   (self.ptz.get_telemetry().get("zoom")
                               if self.ptz else None),
            # What the viewer actually sees: optical x digital. Reporting
            # only one of the two understates the framing whenever the
            # handoff has the lens carrying most of the magnification.
            "total_mag":      round(self.zoom.total_mag, 2) if self.zoom else 1.0,
            **self.control_toggle.status(),
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
