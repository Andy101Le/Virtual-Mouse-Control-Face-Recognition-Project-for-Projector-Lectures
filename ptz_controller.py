"""
ptz_controller.py
──────────────────
Autonomous pan / tilt controller for the Arducam PTZ (OV5647) hardware
used in the Virtual Mouse project.

Wraps the existing Focuser.py register interface in a background-threaded
controller so it never blocks main.py's camera loop. Designed to be fed
per-frame face- AND pose-tracking data that main.py already computes —
no extra MediaPipe inference happens in here.

Usage in main.py (minimal):

    from ptz_controller import PTZController

    ptz = PTZController(active_user=ACTIVE_USER)
    ...
    # inside the main loop, after face/pose detection updates each frame:
    ptz.update(
        nose_pos=face_nose_pos,
        recognised_user=recognised_user,
        is_registered_face_visible=(user_active and recognised_user != UNKNOWN_LABEL),
        pose_landmarks=cached_pose_lms,   # list of pose results this frame
    )
    ...
    # to draw a debug HUD (motor positions, last action, state) on `frame`:
    ptz.draw_debug_hud(frame)
    ...
    # on shutdown:
    ptz.close()

Behavior — three states:
  - LOCKED_FACE: face detected & recognised as active_user this frame.
    Pans/tilts using face nose position (fine control).
  - SEARCHING_BY_POSE: face recognition just failed/disappeared, but a
    pose skeleton is visible near where the face was last confirmed
    (proximity check against the last known face position — this is
    what keeps the system from "finding" a stranger's body). Pose
    detection works at much greater range than face detection, so this
    state uses the pose's nose as a coarse proxy, and hands back to
    LOCKED_FACE automatically once the face becomes detectable again.
  - LOST: neither face nor a plausible nearby pose is visible. Holds
    the last pan/tilt position — no scanning, no recentring — and only
    re-engages when the same active_user's face reappears.

  Pan/tilt commands track an internally-held COMMANDED target rather
  than re-reading the hardware register every tick. This matters
  because the PTZ's physical motors take real time to travel — reading
  Focuser.get() faster than the motor can move returns a stale
  position, which previously caused the controller to re-issue the
  same command repeatedly and then overshoot once the real position
  caught up. Now the controller always reasons from where it last TOLD
  the motor to go.

  All hardware I/O happens on a single background worker thread via a
  work queue, so slow I2C polling never stalls the vision pipeline.
"""

import threading
import queue
import time

from Focuser import Focuser


class _Targets:
    """
    Tracks where we last COMMANDED the hardware to go, separate from
    where the hardware physically is right now. The PTZ's motors take
    real time to travel; re-reading Focuser.get() faster than that
    causes the controller to see a stale position and re-issue (or
    overshoot) commands. Reasoning from our own commanded targets
    avoids that class of bug entirely.
    """
    def __init__(self, motor_x, motor_y, zoom):
        self.motor_x = motor_x
        self.motor_y = motor_y
        self.zoom    = zoom


# State machine states
STATE_LOCKED_FACE     = "LOCKED_FACE"
STATE_SEARCHING_POSE  = "SEARCHING_BY_POSE"
STATE_LOST            = "LOST"


class PTZController:
    # ── Tunables: center-region tolerance ────────────────────────────────────
    # The user's nose must drift outside this normalized distance from dead
    # center (0.5, 0.5) — independently on each axis — before the camera
    # reacts at all. Below this, the subject is considered "centered enough"
    # and the motors don't move.
    #
    # Tightened from 0.19154: that wide band existed to swallow the old
    # SG90s' coarse, sloppy steps so they wouldn't hunt around the middle.
    # The MG92B lands accurately on the commanded degree, the GPIO-driven
    # SG92R takes sub-degree commands, and the gains ramp smoothly from the
    # band edge — so a tighter band now just holds the face closer to dead
    # center instead of dithering. Raise back toward 0.19 if it ever hunts.
    CENTER_TOLERANCE     = 0.10

    # ── Tunables: position smoothing ─────────────────────────────────────────
    # EMA applied to the incoming nose position itself, BEFORE computing
    # pan/tilt error — this is what actually removes the "snappy"/jerky feel,
    # since a single noisy detection (or a quick head turn) no longer
    # immediately yields a full-strength motor command; it has to persist
    # across a few frames to move the smoothed position enough to matter.
    # Lower = smoother but slower to follow real movement; higher = snappier.
    POSITION_SMOOTH_ALPHA = 0.45

    # ── Tunables: pan/tilt (fine, face-driven) ──────────────────────────────
    # Gains apply to the error BEYOND the center tolerance, so the response
    # ramps smoothly from zero at the band edge instead of kicking in at
    # full strength the instant the face crosses the tolerance. Magnitudes
    # restored to the original 26/22 — the 6.5/6.0 crawl-tuning dated from
    # when the servo connectors were miswired and made tracking visibly lag
    # a moving subject once the wiring was fixed.
    #
    # Per-axis step ceilings replace the old shared MAX_STEP_PER_TICK, sized
    # to what each servo can actually do: the metal-geared MG92B on pan is
    # fast and accurate enough to execute 3°/tick (~60°/s at the 20 Hz tick
    # rate) cleanly, and the SG92R on tilt is GPIO-driven and accepts
    # fractional degrees, so its ceiling needn't be a whole number. Near the
    # band edge both axes get genuinely sub-degree steps — pan banks them in
    # a fractional accumulator until a whole degree is owed (Arducam-board
    # limit), tilt commands the fraction directly.
    PAN_GAIN             = 26.0
    TILT_GAIN            = 22.0
    PAN_MAX_STEP         = 3.0    # °/tick ceiling — pan, MG92B via the board
    TILT_MAX_STEP        = 2.5    # °/tick ceiling — tilt, SG92R via GPIO

    # ── Tunables: pose-fallback coarse search ───────────────────────────────
    POSE_PROXIMITY_THRESH = 0.22    # max normalized distance between pose
                                     # nose and the LAST KNOWN face position
                                     # to trust this pose as "probably them"
    POSE_PAN_GAIN          = 24.0   # restored alongside PAN_GAIN/TILT_GAIN;
    POSE_TILT_GAIN         = 18.0   # same beyond-the-deadzone application,
                                     # kept slightly gentler than the face
                                     # gains, as before
    POSE_VIS_THRESH        = 0.4    # match main.py's POSE_VIS_THRESH

    # Lengthened from 1.0 s for long-range tracking: at distance the face
    # detector flickers (small, marginal detections) and pose sometimes
    # drops a beat too — a longer grace keeps the anchor alive through
    # those gaps instead of resetting to LOST and zooming out.
    LOST_AFTER_SECONDS    = 2.5     # face AND pose both absent this long -> LOST

    UPDATE_MIN_INTERVAL   = 0.05

    # ── Tunables: optical auto-zoom (OPT_ZOOM) ──────────────────────────────
    # The lens has a motorised zoom the project never used. While face-locked
    # the controller servos it to hold the face's normalized size (bbox
    # diagonal, as AuthManager.face_size reports it) inside a target band —
    # subject walks away, face shrinks below the band, camera zooms in; and
    # vice versa. On LOST it returns to the wide end so re-acquisition sees
    # the whole room. This is the main range-doubling lever: 2x optical
    # magnification puts 2x the pixels on the face at 2x the distance.
    FACE_TARGET_LO   = 0.10   # zoom IN while face size below this
    FACE_TARGET_HI   = 0.16   # zoom OUT while face size above this
    FACE_SIZE_SMOOTH = 0.30   # EMA on face size fed to the zoom decision

    ZOOM_STEP          = 500    # register units per zoom nudge while tracking
    ZOOM_WIDE_STEP     = 1500   # faster steps when returning to wide on LOST
    ZOOM_STEP_INTERVAL = 0.20   # min seconds between zoom writes — keeps the
                                # shared I2C bus responsive for pan/tilt
    ZOOM_DWELL_TICKS   = 4      # face must sit outside the band this many
                                # consecutive snapshots IN THE SAME DIRECTION
                                # before zoom moves (rides out single noisy
                                # size readings — see _auto_zoom_track)
    ZOOM_SETTLE_S      = 0.6    # zoom idle this long after moving -> fire
                                # on_zoom_settled (autofocus re-hunts then)

    # True = register MAX_VALUE (20000) is the telephoto end. Flip this if
    # test_zoom_sweep.py shows your lens is tightest at the LOW end.
    ZOOM_TELE_AT_MAX = True

    # Estimated optical magnification at full tele relative to full wide.
    # Used to scale pan/tilt gains and step ceilings down as the FOV
    # narrows — without this, gains tuned at wide angle overshoot and
    # oscillate when zoomed in. Doesn't need to be exact; if tracking
    # oscillates only when zoomed, raise it, and if zoomed tracking feels
    # sluggish, lower it.
    ZOOM_FOV_MAG = 3.0

    def __init__(self, active_user=None, i2c_bus=1, enabled=True, verbose=True,
                 tilt_gpio_pin=None, enable_zoom=True):
        self.active_user = active_user
        self.enabled     = enabled
        self.verbose     = verbose
        self.zoom_enabled = enable_zoom

        # Autofocus coordination hooks (set by GestureSession). Zoom moves
        # shift the focus plane, so the AF controller must know when zoom is
        # moving (suppress its scene-change hunts) and when it has settled
        # (kick one deliberate re-hunt).
        self.on_zoom_step    = None   # called as on_zoom_step(zoom_value)
        self.on_zoom_settled = None   # called as on_zoom_settled(total_moved)

        try:
            if tilt_gpio_pin is not None:
                # The Arducam board's tilt channel is dead (it ACKs
                # OPT_MOTOR_Y writes but never drives the header) — route
                # tilt pulses out a Pi GPIO pin instead. Pan/zoom/focus
                # still go to the board.
                from gpio_tilt import GpioTiltFocuser
                self._focuser = GpioTiltFocuser(i2c_bus, tilt_pin=tilt_gpio_pin)
                print(f"[PTZController] tilt servo driven from GPIO{tilt_gpio_pin} "
                      f"(Arducam tilt channel bypassed)")
            else:
                self._focuser = Focuser(i2c_bus)
        except Exception as e:
            # No PTZ board on this I2C bus (missing smbus module, or no
            # device answering at the expected address) — a Pi running
            # without the PTZ hardware attached is a normal, supported
            # setup, not a fatal error. Warn on the command line and run
            # with PTZ tracking/nudging disabled instead of letting the
            # exception take the whole process down.
            print(f"[PTZController] WARNING: PTZ hardware not detected on "
                  f"i2c bus {i2c_bus} ({e}). Disabling PTZ tracking and "
                  f"manual controls; the rest of the app continues normally.")
            self._focuser = None
            self.enabled  = False

        # GPIO-driven tilt (GpioTiltFocuser) accepts fractional degrees —
        # the SG92R just follows whatever pulse width it's given. The
        # Arducam board only takes whole degrees, hence the capability flag.
        self._frac_tilt = bool(getattr(self._focuser, "FRACTIONAL_TILT", False))

        # Pan/tilt and focus are different registers but share ONE I2C bus and
        # busy flag. The autofocus controller drives OPT_FOCUS from its own
        # thread; this lock serialises every bus transaction so the two never
        # collide mid-read. Exposed via .io_lock so AutoFocusController can
        # take the same one.
        self._io_lock = threading.Lock()

        # State machine
        self._state            = STATE_LOST
        self._last_seen_time   = 0.0   # last time face OR pose was usable
        self._last_known_face_pos = None  # (x, y) — used for pose proximity gating

        # EMA-smoothed tracking position (x, y), normalized 0-1 — fed into
        # pan/tilt error calculation instead of the raw per-frame detection,
        # so a single noisy reading can't yield a full-strength motor
        # command. Reset to None whenever tracking is lost, so re-acquiring
        # the subject snaps straight to their real position instead of
        # slowly smoothing in from wherever the camera last pointed.
        self._smoothed_pos = None

        # Fractional-degree carry for pan/tilt. The Arducam board only takes
        # whole degrees, so per-tick steps below 1° accumulate here until
        # they add up to a full degree — gentle proportional commands still
        # move the camera instead of rounding to zero forever.
        self._accum_x = 0.0
        self._accum_y = 0.0

        # Auto-zoom state
        self._smoothed_face_size = None
        self._zoom_dwell         = 0     # consecutive same-direction snapshots
        self._zoom_dwell_dir     = 0     # +1 = wants tele, -1 = wants wide
        self._zoom_last_step_t   = 0.0
        self._zoom_net_units     = 0     # SIGNED units moved since the last
                                         # settle event — see
                                         # _auto_zoom_check_settled
        self._zoom_last_move_t   = 0.0

        # Commanded targets (NOT re-read from hardware every tick — see _Targets)
        self._targets = None  # populated from real hardware on first use

        # Telemetry
        self._telemetry_lock = threading.Lock()
        self._telemetry = {
            "state": STATE_LOST,
            "motor_x": None, "motor_y": None, "zoom": None,
            "last_action": "idle",
            "last_pan_dir": "-", "last_tilt_dir": "-",
            "pan_at_limit": False, "tilt_at_limit": False,
        }

        # Threading / work queue
        self._q       = queue.Queue(maxsize=1)
        self._stop    = threading.Event()
        self._worker  = threading.Thread(target=self._run, daemon=True)
        self._last_update_t = 0.0

        self._worker.start()
        self._seed_targets()

    def _seed_targets(self):
        """Read real hardware state once at startup to initialize our
        commanded-target tracking, then never read it again during normal
        operation — see _Targets docstring for why.

        Clamps the read values to Focuser's documented valid ranges. This
        matters if the hardware register was left in an out-of-range state
        by a previous run — Focuser.get() does NOT clamp on reads, only
        Focuser.set() clamps on writes, so a bad value can persist in the
        register indefinitely until something starts fresh from a sane
        number.
        """
        zoom_lo, zoom_hi = self._zoom_range()
        # The WIDE end is whichever end isn't telephoto — not simply the low
        # register value. With ZOOM_TELE_AT_MAX=False (which web_server tells
        # you to set if your lens is tightest at the low end) parking at
        # zoom_lo would mean parking at full TELE, making _zoom_mag() report
        # 3x with the lens actually wide and inflating every derived figure.
        zoom_wide = zoom_lo if self.ZOOM_TELE_AT_MAX else zoom_hi
        if self._focuser is None:
            # No hardware — safe-ish placeholders, zoom at the wide end so
            # _zoom_frac() reads 0.0 and _zoom_mag() stays 1.0, leaving the
            # pan/tilt gains unscaled exactly as they were pre-zoom.
            self._targets = _Targets(90, 90, zoom_wide)
            return
        try:
            with self._io_lock:
                mx_raw = self._focuser.get(Focuser.OPT_MOTOR_X)
                my_raw = self._focuser.get(Focuser.OPT_MOTOR_Y)
                mz_raw = self._focuser.get(Focuser.OPT_ZOOM)

            mx = self._clamp(mx_raw, Focuser.opts[Focuser.OPT_MOTOR_X]["MIN_VALUE"],
                              Focuser.opts[Focuser.OPT_MOTOR_X]["MAX_VALUE"])
            my = self._clamp(my_raw, Focuser.opts[Focuser.OPT_MOTOR_Y]["MIN_VALUE"],
                              Focuser.opts[Focuser.OPT_MOTOR_Y]["MAX_VALUE"])
            mz = self._clamp(mz_raw, zoom_lo, zoom_hi)

            if (mx, my) != (mx_raw, my_raw):
                print(f"[PTZController] WARNING: hardware register(s) were out of "
                      f"valid range at startup (raw motor_x={mx_raw}, motor_y={my_raw}) "
                      f"— clamping to (motor_x={mx}, motor_y={my}) and re-writing "
                      f"corrected values to hardware now.")
                with self._io_lock:
                    self._focuser.set(Focuser.OPT_MOTOR_X, mx, 0)
                    self._focuser.set(Focuser.OPT_MOTOR_Y, my, 0)

            self._targets = _Targets(mx, my, mz)
            with self._telemetry_lock:
                self._telemetry["motor_x"] = mx
                self._telemetry["motor_y"] = my
                self._telemetry["zoom"]    = mz
        except Exception as e:
            print(f"[PTZController] Could not read initial hardware state: {e}")
            # Safe-ish fallbacks: mid-range pan/tilt, wide-end zoom.
            self._targets = _Targets(90, 90, zoom_wide)

    # ── Public API ──────────────────────────────────────────────────────────
    @property
    def focuser(self):
        """The shared Focuser instance (so autofocus can drive OPT_FOCUS)."""
        return self._focuser

    @property
    def io_lock(self):
        """Shared I2C-bus lock — take this around any Focuser access made
        from another thread so it can't collide with pan/tilt writes."""
        return self._io_lock

    def update(self, nose_pos, recognised_user, is_registered_face_visible,
               pose_landmarks=None, face_size=None):
        """
        Call once per main-loop iteration (cheap — just queues a snapshot
        for the background worker; never blocks).

        nose_pos:    np.array([x, y]) normalized 0-1, or None
        recognised_user: str, the username matched this frame (or UNKNOWN_LABEL)
        is_registered_face_visible: bool — True only when the active_user's
            own face is currently recognised & authenticated this frame
        pose_landmarks: main.py's `cached_pose_lms` — a list of pose results
            (each a list of 33 landmarks with .x/.y/.visibility), or None/[].
            Used as a coarse fallback when face detection can't resolve the
            user at a distance. Optional — omit it and the controller just
            won't have a pose fallback (face-only behavior).
        face_size: normalized face bbox diagonal (AuthManager.face_size), or
            None. Drives the optical auto-zoom: while face-locked, zoom
            servos to keep this inside [FACE_TARGET_LO, FACE_TARGET_HI].
        """
        if not self.enabled:
            return

        now = time.perf_counter()
        if now - self._last_update_t < self.UPDATE_MIN_INTERVAL:
            return
        self._last_update_t = now

        snapshot = {
            "nose_pos": None if nose_pos is None else (float(nose_pos[0]), float(nose_pos[1])),
            "recognised_user": recognised_user,
            "is_target_visible": bool(is_registered_face_visible and
                                       recognised_user == self.active_user),
            "pose_landmarks": pose_landmarks or [],
            "face_size": None if face_size is None else float(face_size),
            "t": now,
        }

        try:
            self._q.put_nowait(snapshot)
        except queue.Full:
            try:
                self._q.get_nowait()
            except queue.Empty:
                pass
            try:
                self._q.put_nowait(snapshot)
            except queue.Full:
                pass

    def set_enabled(self, enabled: bool):
        if enabled and not self.enabled:
            # Resuming auto after manual driving — drop any stale sub-degree
            # carry so the first correction is computed fresh.
            self._accum_x = self._accum_y = 0.0
        self.enabled = enabled

    def set_active_user(self, username):
        """
        Tell the controller WHOSE face counts as the tracking target. Until
        this is set, is_target_visible can never be true (recognised_user is
        compared against self.active_user), so the camera would sit in LOST
        and never move. Called on login so the PTZ follows the person who
        just signed in.

        Clears the tracking anchors so a user switch re-acquires from the new
        person's real position rather than smoothing in from the old one.
        """
        self.active_user          = username
        self._last_known_face_pos = None
        self._smoothed_pos        = None
        self._accum_x = self._accum_y = 0.0
        self._smoothed_face_size  = None
        self._zoom_dwell          = 0
        self._zoom_dwell_dir      = 0

    def get_state(self) -> str:
        return self._state

    def get_telemetry(self) -> dict:
        """Snapshot of current motor/state for display or logging."""
        with self._telemetry_lock:
            return dict(self._telemetry)

    def draw_debug_hud(self, frame, origin=(10, 110), line_height=20,
                        color=(0, 255, 255)):
        """
        Draws live PTZ telemetry directly onto `frame` (in place) using
        cv2 — state, motor X/Y positions + direction, and a red warning
        line if any axis is pinned at its physical limit (camera can't
        move further that way). Call once per main-loop iteration, after
        ptz.update(...), right before cv2.imshow(...).
        """
        import cv2  # local import — keeps this module importable even
                    # in contexts where cv2 may not be available.
        t = self.get_telemetry()
        lines = [
            (f"PTZ state={t['state']}  last={t['last_action']}", color),
            (f"motorX={t['motor_x']}  dir={t['last_pan_dir']}   "
             f"motorY={t['motor_y']}  dir={t['last_tilt_dir']}", color),
        ]
        limits_hit = [name for name, key in
                      (("PAN", "pan_at_limit"), ("TILT", "tilt_at_limit"))
                      if t.get(key)]
        if limits_hit:
            lines.append((f"!!! AT PHYSICAL LIMIT: {', '.join(limits_hit)} !!!", (0, 0, 255)))

        x, y = origin
        for i, (line, line_color) in enumerate(lines):
            cv2.putText(frame, line, (x, y + i * line_height),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, line_color, 1, cv2.LINE_AA)

    def close(self):
        self._stop.set()
        try:
            self._q.put_nowait(None)
        except queue.Full:
            pass
        self._worker.join(timeout=2.0)
        # GpioTiltFocuser needs its pulse train stopped; the plain I2C
        # Focuser has no close() and needs none.
        close_focuser = getattr(self._focuser, "close", None)
        if callable(close_focuser):
            close_focuser()

    # ── Background worker ─────────────────────────────────────────────────
    def _run(self):
        while not self._stop.is_set():
            try:
                snap = self._q.get(timeout=0.5)
            except queue.Empty:
                continue
            if snap is None or self._stop.is_set():
                break
            self._process(snap)

    def _process(self, snap):
        now = snap["t"]

        if snap["is_target_visible"] and snap["nose_pos"] is not None:
            # ── Face confirmed: fine-grained tracking ───────────────────────
            self._state = STATE_LOCKED_FACE
            self._last_seen_time = now
            self._last_known_face_pos = snap["nose_pos"]
            smoothed = self._smooth_position(snap["nose_pos"])
            self._track_pan_tilt_fine(smoothed)
            self._auto_zoom_track(snap["face_size"], now)

        else:
            # Face not confirmed this frame — see if a pose plausibly
            # belongs to the user we just lost, using proximity to their
            # last known face position as the identity proxy.
            nose_xy = self._find_plausible_pose(snap["pose_landmarks"])

            if nose_xy is not None:
                self._state = STATE_SEARCHING_POSE
                self._last_seen_time = now
                self._last_known_face_pos = nose_xy  # keep proximity anchor fresh
                smoothed = self._smooth_position(nose_xy)
                self._track_pan_tilt_coarse(smoothed)
            else:
                # Nothing plausible. Drop to LOST only after a short grace
                # period — a single missed frame shouldn't reset the anchor.
                if now - self._last_seen_time > self.LOST_AFTER_SECONDS:
                    self._state = STATE_LOST
                    self._last_known_face_pos = None
                    self._smoothed_pos = None  # forget stale position so the
                                                 # next acquisition snaps to
                                                 # the subject's real spot
                                                 # instead of smoothing in
                                                 # from a stale location
                    self._accum_x = self._accum_y = 0.0  # and stale sub-degree carry
                    self._smoothed_face_size = None
                if self._state == STATE_LOST:
                    # Wide FOV finds a re-appearing subject far faster than a
                    # zoomed-in one — return to the wide end while LOST.
                    self._auto_zoom_wide(now)

        self._auto_zoom_check_settled(now)

        with self._telemetry_lock:
            self._telemetry["state"] = self._state

    def _smooth_position(self, raw_xy):
        """
        EMA-smooths a raw (x, y) detection against the last smoothed
        position. First call after a reset (self._smoothed_pos is None)
        snaps straight to the raw value — no slow catch-up the moment
        someone is freshly (re)acquired.
        """
        if self._smoothed_pos is None:
            self._smoothed_pos = (float(raw_xy[0]), float(raw_xy[1]))
        else:
            a = self.POSITION_SMOOTH_ALPHA
            px, py = self._smoothed_pos
            self._smoothed_pos = (a * raw_xy[0] + (1 - a) * px,
                                   a * raw_xy[1] + (1 - a) * py)
        return self._smoothed_pos

    def _find_plausible_pose(self, pose_landmarks_list):
        """
        Returns the nose (x, y) for the pose most likely to be the
        active_user, or None if nothing plausible is visible.

        "Plausible" = a pose whose nose landmark is within
        POSE_PROXIMITY_THRESH of the last position we confirmed via face
        recognition. This is the identity-continuity guard: it means the
        coarse pose-search can only ever continue tracking someone we
        JUST had a confirmed face-lock on, never pick a stranger out of
        a crowd from a cold start.
        """
        if not pose_landmarks_list or self._last_known_face_pos is None:
            return None

        anchor = self._last_known_face_pos
        best = None
        best_dist = self.POSE_PROXIMITY_THRESH

        for pose_lms in pose_landmarks_list:
            if len(pose_lms) <= 0:
                continue
            nose = pose_lms[0]
            if nose.visibility < self.POSE_VIS_THRESH:
                continue

            dx = nose.x - anchor[0]
            dy = nose.y - anchor[1]
            dist = (dx * dx + dy * dy) ** 0.5
            if dist < best_dist:
                best = (float(nose.x), float(nose.y))
                best_dist = dist

        return best

    # ── Optical auto-zoom ───────────────────────────────────────────────────
    def _zoom_range(self):
        # ptz_manual._opt_bounds exists because trimmed copies of Focuser.py
        # in the wild define only OPT_FOCUS. _seed_targets runs from
        # __init__, which web_server executes at module scope, so an
        # unguarded KeyError here would stop the whole server from importing
        # rather than just costing us the zoom channel.
        o = Focuser.opts.get(Focuser.OPT_ZOOM)
        if not o:
            return 3000, 20000    # documented Arducam range
        return o["MIN_VALUE"], o["MAX_VALUE"]

    def _zoom_frac(self):
        """How far toward telephoto we are: 0.0 = full wide, 1.0 = full tele."""
        lo, hi = self._zoom_range()
        f = (self._targets.zoom - lo) / float(hi - lo)
        return f if self.ZOOM_TELE_AT_MAX else 1.0 - f

    def _zoom_mag(self):
        """Estimated optical magnification at the current zoom position,
        relative to full wide. Pan/tilt gains and step ceilings divide by
        this: the same normalized on-screen error corresponds to a smaller
        angular error once the FOV has narrowed, so unscaled gains overshoot
        and oscillate exactly when zoomed onto a distant subject."""
        return 1.0 + self._zoom_frac() * (self.ZOOM_FOV_MAG - 1.0)

    def get_zoom_state(self):
        """
        What the digital zoom needs to know to stay out of the optical
        zoom's way. Safe to call from the vision thread every frame — it
        only reads commanded-target bookkeeping, never touches I2C.

        mag:      current optical magnification relative to full wide.
        headroom: 0..1, how much of the optical range is still unused
                  toward telephoto. 0.0 means the motor is at its tele
                  stop and any further magnification must come from the
                  digital crop.
        moving:   True while the motor is mid-travel. The digital zoom
                  freezes then, so the picture doesn't pump as the two
                  loops chase the same error at different speeds.

        With optical zoom disabled or no hardware present this reports a
        motor pinned at full wide with no headroom, which makes the
        digital zoom behave exactly as it did before optical existed.
        """
        if self._targets is None:
            return {"mag": 1.0, "headroom": 0.0, "moving": False}
        if not (self.zoom_enabled and self.enabled):
            # Nothing is going to drive the motor — auto-zoom is off, or the
            # web UI took manual control (ManualPTZ.set_mode disables the
            # whole controller). The lens can still be parked at telephoto,
            # by the manual slider or by a previous run, so report the real
            # magnification; but report NO headroom, or the digital zoom
            # would sit forever waiting on a motor that will never step.
            return {"mag": self._zoom_mag(), "headroom": 0.0, "moving": False}

        # Headroom is about INTENT, not mechanical travel. The optical loop
        # stops as soon as the face is inside [FACE_TARGET_LO, FACE_TARGET_HI]
        # and will not move again however much travel remains — so reporting
        # raw remaining travel would park the digital zoom at 1.0 forever and
        # silently replace its framing target with the much looser optical
        # one. Headroom is therefore nonzero only when the motor both wants
        # to go further toward tele AND still can.
        wants_tele = (self._smoothed_face_size is not None and
                      self._smoothed_face_size < self.FACE_TARGET_LO)
        travel_left = max(0.0, 1.0 - self._zoom_frac())
        return {
            "mag":      self._zoom_mag(),
            "headroom": travel_left if wants_tele else 0.0,
            # perf_counter, not time(): _zoom_last_move_t is stamped from
            # the snapshot clock in update(), which is perf_counter-based.
            "moving":   (time.perf_counter() - self._zoom_last_move_t)
                        < self.ZOOM_SETTLE_S,
        }

    def _auto_zoom_track(self, face_size, now):
        """Face-locked: hold the face's normalized size inside the target
        band by nudging the zoom motor. Runs on the worker thread."""
        if not self.zoom_enabled or face_size is None:
            return

        if self._smoothed_face_size is None:
            self._smoothed_face_size = face_size
        else:
            a = self.FACE_SIZE_SMOOTH
            self._smoothed_face_size = (a * face_size +
                                        (1 - a) * self._smoothed_face_size)

        fs = self._smoothed_face_size
        if fs < self.FACE_TARGET_LO:
            zoom_in = True
        elif fs > self.FACE_TARGET_HI:
            zoom_in = False
        else:
            self._zoom_dwell = 0
            self._zoom_dwell_dir = 0
            return

        # Require the face to sit outside the band for several consecutive
        # snapshots IN THE SAME DIRECTION — one glitchy size reading must not
        # pump the zoom motor.
        #
        # The direction check is the whole point and it was missing. The
        # counter used to increment for any out-of-band reading whatever its
        # sign, so three ticks of "face too small" followed by ONE bogus
        # oversized reading fired a zoom OUT: a single noisy sample spent the
        # credit the opposite direction had built up, which is precisely the
        # failure the dwell exists to prevent. At range that is the common
        # case, not a corner case — a barely-resolved face reports an
        # unreliable bbox — and the resulting in/out hunting kept re-triggering
        # autofocus, blurring the image and making recognition worse still.
        direction = 1 if zoom_in else -1
        if direction != self._zoom_dwell_dir:
            self._zoom_dwell     = 0
            self._zoom_dwell_dir = direction
        self._zoom_dwell += 1
        if self._zoom_dwell < self.ZOOM_DWELL_TICKS:
            return

        tele_sign = 1 if self.ZOOM_TELE_AT_MAX else -1
        delta = self.ZOOM_STEP * tele_sign * (1 if zoom_in else -1)
        self._zoom_write_delta(delta, now)

    def _auto_zoom_wide(self, now):
        """LOST: step back toward the wide end (chunked, not one long travel,
        so the shared I2C bus never blocks for the whole journey)."""
        if not self.zoom_enabled:
            return
        lo, hi = self._zoom_range()
        wide = lo if self.ZOOM_TELE_AT_MAX else hi
        delta = self._clamp(wide - self._targets.zoom,
                            -self.ZOOM_WIDE_STEP, self.ZOOM_WIDE_STEP)
        self._zoom_write_delta(delta, now)

    def _zoom_write_delta(self, register_delta, now):
        if register_delta == 0:
            return
        if now - self._zoom_last_step_t < self.ZOOM_STEP_INTERVAL:
            return
        lo, hi = self._zoom_range()
        new = int(self._clamp(self._targets.zoom + register_delta, lo, hi))
        if new == self._targets.zoom:
            return
        try:
            with self._io_lock:
                # flag=0 → don't block for motor travel on THIS call. Note
                # this defers the wait rather than avoiding it: Focuser.set
                # calls waitingForFree() before writing, so the next bus op
                # pays for this travel — and since pan/tilt runs on this same
                # worker thread and holds the same _io_lock, that cost lands
                # on pan/tilt. During a zoom ramp the tracking rate therefore
                # drops below UPDATE_MIN_INTERVAL and queued snapshots get
                # dropped (queue maxsize=1). Degraded tracking while zooming
                # is the accepted trade; the alternative is blocking here for
                # the full travel, which is strictly worse.
                self._focuser.set(Focuser.OPT_ZOOM, new, 0)
        except Exception as e:
            print(f"[PTZController] zoom write failed: {e}")
            return

        self._log(f"ZOOM {self._targets.zoom} -> {new}  "
                  f"(face_size={self._smoothed_face_size and round(self._smoothed_face_size, 3)}, "
                  f"mag~{self._zoom_mag():.2f}x)  [{self._state}]")
        self._targets.zoom     = new
        self._zoom_last_step_t = now
        self._zoom_last_move_t = now
        self._zoom_net_units   += register_delta
        self._zoom_dwell        = 0
        self._zoom_dwell_dir    = 0
        with self._telemetry_lock:
            self._telemetry["zoom"]        = new
            self._telemetry["last_action"] = "zoom"

        cb = self.on_zoom_step
        if cb:
            try:
                cb(new)
            except Exception:
                pass

    def _auto_zoom_check_settled(self, now):
        """Fire on_zoom_settled once the motor has been quiet for a beat
        after moving — that's when autofocus should do ONE deliberate
        re-hunt, instead of chasing the focus plane during the ramp."""
        # NET, not total travel. An in-then-out pair leaves the lens exactly
        # where it started, so the focus plane never moved and re-hunting
        # would be pure cost — and while the zoom was hunting, that cost was
        # a blurred image, which degrades the very recognition whose failure
        # caused the hunting in the first place.
        if (self._zoom_net_units and
                (now - self._zoom_last_move_t) >= self.ZOOM_SETTLE_S):
            moved = abs(self._zoom_net_units)
            self._zoom_net_units = 0
            cb = self.on_zoom_settled
            if cb:
                try:
                    cb(moved)
                except Exception:
                    pass

    # ── Pan / tilt — fine (face-driven) ─────────────────────────────────────
    def _track_pan_tilt_fine(self, nose_xy):
        m = self._zoom_mag()
        self._track_pan_tilt(nose_xy, self.PAN_GAIN / m, self.TILT_GAIN / m,
                             self.PAN_MAX_STEP / m, self.TILT_MAX_STEP / m)

    # ── Pan / tilt — coarse (pose-driven, gentler gain) ─────────────────────
    def _track_pan_tilt_coarse(self, nose_xy):
        m = self._zoom_mag()
        self._track_pan_tilt(nose_xy, self.POSE_PAN_GAIN / m,
                             self.POSE_TILT_GAIN / m,
                             self.PAN_MAX_STEP / m, self.TILT_MAX_STEP / m)

    def _track_pan_tilt(self, nose_xy, pan_gain, tilt_gain,
                        pan_max_step, tilt_max_step):
        nx, ny = nose_xy
        err_x = nx - 0.5   # positive => subject is right of center (in the
                            # mirrored preview main.py shows you)
        err_y = ny - 0.5   # positive => subject is below center

        if self.verbose:
            print(f"[PTZController] RAW nose_xy=({nx:.4f}, {ny:.4f})  "
                  f"err_x={err_x:+.4f} err_y={err_y:+.4f}  [{self._state}]")

        pan_dir, tilt_dir = "-", "-"
        mx_lo = Focuser.opts[Focuser.OPT_MOTOR_X]["MIN_VALUE"]
        mx_hi = Focuser.opts[Focuser.OPT_MOTOR_X]["MAX_VALUE"]
        my_lo = Focuser.opts[Focuser.OPT_MOTOR_Y]["MIN_VALUE"]
        my_hi = Focuser.opts[Focuser.OPT_MOTOR_Y]["MAX_VALUE"]

        try:
            if abs(err_x) > self.CENTER_TOLERANCE:
                # Gain acts on the error PAST the deadzone edge, so the step
                # ramps up from zero as the face drifts further off-center
                # instead of jumping to a full-strength command the moment
                # the tolerance is crossed (see the PAN_GAIN comment).
                excess = abs(err_x) - self.CENTER_TOLERANCE
                step = self._clamp((excess if err_x > 0 else -excess) * pan_gain,
                                    -pan_max_step, pan_max_step)
                # Accumulate fractional degrees — the board only accepts
                # whole ones. int() truncates toward zero, so the leftover
                # fraction keeps its sign and carries into the next tick;
                # a whole degree is only written once one has built up.
                self._accum_x += step
                whole = int(self._accum_x)
                if whole != 0:
                    self._accum_x -= whole
                    # Subject moving right (err_x positive) must INCREASE
                    # motor_x to pan the camera right (confirmed against real
                    # hardware).
                    raw_new_val = self._targets.motor_x + whole
                    # Clamp our own commanded target to the real hardware
                    # range (0-180). Focuser.set() also clamps internally, but
                    # only the value IT writes — it never reports the clamped
                    # value back, so without this our internal target keeps
                    # drifting past the physical limit every tick while the
                    # real servo sits pinned at its end stop.
                    new_val = self._clamp(raw_new_val, mx_lo, mx_hi)
                    at_limit = (new_val != raw_new_val)
                    with self._io_lock:
                        self._focuser.set(Focuser.OPT_MOTOR_X, new_val, 0)
                    pan_dir = "RIGHT" if whole > 0 else "LEFT"
                    limit_note = "  *** AT PHYSICAL LIMIT ***" if at_limit else ""
                    self._log(f"PAN  err_x={err_x:+.3f} step={step:+.2f} "
                              f"motor_x {self._targets.motor_x} -> {new_val}  ({pan_dir})  "
                              f"[{self._state}]{limit_note}")
                    self._targets.motor_x = new_val
                    with self._telemetry_lock:
                        self._telemetry["motor_x"] = new_val
                        self._telemetry["last_pan_dir"] = pan_dir
                        self._telemetry["last_action"] = "pan"
                        self._telemetry["pan_at_limit"] = at_limit

            if abs(err_y) > self.CENTER_TOLERANCE:
                # Same deadzone-relative proportional step as the pan axis.
                excess = abs(err_y) - self.CENTER_TOLERANCE
                step = self._clamp((excess if err_y > 0 else -excess) * tilt_gain,
                                    -tilt_max_step, tilt_max_step)
                if self._frac_tilt:
                    # GPIO-driven SG92R: command the fractional step directly.
                    # Rounded to 0.1° — about the real resolution of the
                    # software-PWM pulse train, and it keeps a sub-resolution
                    # step from producing a no-op write every tick.
                    raw_new_val = round(self._targets.motor_y + step, 1)
                else:
                    # Arducam board only takes whole degrees — bank fractions
                    # in the accumulator like the pan axis above.
                    self._accum_y += step
                    whole = int(self._accum_y)
                    raw_new_val = None
                    if whole != 0:
                        self._accum_y -= whole
                        raw_new_val = self._targets.motor_y + whole
                if raw_new_val is not None and raw_new_val != self._targets.motor_y:
                    new_val = self._clamp(raw_new_val, my_lo, my_hi)
                    at_limit = (new_val != raw_new_val)
                    with self._io_lock:
                        self._focuser.set(Focuser.OPT_MOTOR_Y, new_val, 0)
                    tilt_dir = "DOWN" if step > 0 else "UP"
                    limit_note = "  *** AT PHYSICAL LIMIT ***" if at_limit else ""
                    self._log(f"TILT err_y={err_y:+.3f} step={step:+.2f} "
                              f"motor_y {self._targets.motor_y} -> {new_val}  ({tilt_dir})  "
                              f"[{self._state}]{limit_note}")
                    self._targets.motor_y = new_val
                    with self._telemetry_lock:
                        self._telemetry["motor_y"] = new_val
                        self._telemetry["last_tilt_dir"] = tilt_dir
                        self._telemetry["last_action"] = "tilt"
                        self._telemetry["tilt_at_limit"] = at_limit
        except Exception as e:
            print(f"[PTZController] pan/tilt write failed: {e}")

    def _log(self, msg):
        if self.verbose:
            print(f"[PTZController] {msg}")

    @staticmethod
    def _clamp(value, lo, hi):
        return max(lo, min(hi, value))