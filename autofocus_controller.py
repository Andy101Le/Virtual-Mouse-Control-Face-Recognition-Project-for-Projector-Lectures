"""
autofocus_controller.py
────────────────────────
Continuous ("digital-camera style") autofocus for the Arducam
programmable-focus lens, layered on top of the same Focuser the
PTZController already drives.

Two distinct behaviours, matching how a real camera focuses:

  1. STARTUP FULL SWEEP (once). On start() the lens racks through its
     entire focus range, sampling image sharpness at each step, finds
     the global peak, then does a fine local search to settle on it.
     This is the "rack the whole range" pass you only ever want at
     power-on — it's slow and visibly hunts end-to-end, but it
     guarantees a good starting focus with no prior knowledge.

  2. CONTINUOUS AF (while running). After the sweep it just WATCHES
     sharpness. It does NOT constantly wobble the motor. Only when the
     image goes soft — subject moved closer/further, scene changed —
     past a trigger threshold does it kick off a small, DIRECTIONAL
     hill-climb: nudge focus one way; if that sharpens, keep going; if
     it blurs, reverse; converge on the new peak; then go quiet again.
     That's the "it knows which way to focus" behaviour the user asked
     for — no full rack-out/rack-in during normal operation.

     Crucially, a hunt that finds NO improvement in either direction
     re-baselines and stops. Laplacian sharpness depends on scene
     CONTENT as well as focus (a person walking out of frame lowers it
     even when focus is perfect), so "image went soft" alone can't be
     trusted to mean "defocused". Probing both directions and only
     committing to a climb when a probe actually improves is what stops
     the lens hunting every time the scene merely changes.

WHY IT LIVES ON ITS OWN THREAD
  Focus moves + settle waits are slow (I2C round-trips plus real motor
  travel, tens to hundreds of ms per step). Doing them inline would
  stall the vision loop. So all focus hardware I/O runs on a private
  worker thread here. The vision loop's only job is to hand us a fresh
  sharpness number each frame via report_frame(); it never blocks.

WHY IT SHARES A LOCK WITH THE PTZ
  Pan/tilt and focus are different registers but the SAME I2C chip on
  the SAME bus, sharing one busy flag. Two threads driving the bus at
  once corrupts reads. PTZController owns the Focuser and exposes an
  io_lock; we take it around every focus op so pan/tilt and focus never
  touch the bus simultaneously.
"""

import logging
import threading
import time

import cv2

from Focuser import Focuser

log = logging.getLogger(__name__)

MODE_AUTO   = "auto"
MODE_MANUAL = "manual"

STATE_STARTUP = "startup"   # doing the one-time full-range sweep
STATE_IDLE    = "idle"      # in focus, just watching
STATE_HUNTING = "hunting"   # actively hill-climbing back to a peak
STATE_MANUAL  = "manual"    # user is driving focus by hand


class AutoFocusController:
    # ── Focus travel range to sweep (register units) ────────────────────────
    # Focuser.OPT_FOCUS accepts 0..20000; the Arducam reference autofocus
    # never racks past ~18000 (beyond that is mechanical over-travel with no
    # sharper focus to find), so we cap the usable range there too.
    FOCUS_MIN = 0
    FOCUS_MAX = 18000

    # ── Startup full sweep ──────────────────────────────────────────────────
    COARSE_STEP = 600    # register units between coarse samples (whole range)
    FINE_STEP   = 80     # step for the fine local search around the coarse peak
    FINE_SPAN   = 900    # ± span of that fine search

    # ── Continuous AF ───────────────────────────────────────────────────────
    TRIGGER_RATIO   = 0.68   # start a hunt once sharpness falls below this
                             # fraction of the last known in-focus peak
    IMPROVE_RATIO   = 1.03   # a probe/step counts as "sharper" only if it beats
                             # the reference by this factor (noise guard)
    WORSE_RATIO     = 0.97   # during a climb, treat a drop below this * best as
                             # "passed the peak" and stop
    HUNT_STEP       = 140    # register units per hill-climb nudge
    HUNT_MAX_STEPS  = 45     # safety cap on a single hunt
    SOFT_FRAMES     = 5      # consecutive soft frames before a hunt fires
                             # (rides out brief occlusions / motion blur)
    SETTLE_FRAMES   = 2      # fresh frames to average per measurement

    # ── Sharpness ROI (fraction of frame, centred) ──────────────────────────
    # The PTZ keeps the subject roughly centred, so a central window measures
    # focus on the subject and ignores blurry/empty background at the edges.
    ROI_X = (0.28, 0.72)
    ROI_Y = (0.22, 0.78)

    def __init__(self, focuser, io_lock, do_startup_sweep=True):
        self._focuser = focuser
        self._io_lock = io_lock or threading.Lock()
        self._do_startup_sweep = do_startup_sweep

        # Latest sharpness handed in by the vision loop (report_frame).
        self._sample_lock  = threading.Lock()
        self._sharpness_val = None
        self._sample_id     = 0

        # Command flags set from the web thread, consumed by the worker.
        self._cmd_lock       = threading.Lock()
        self._mode           = MODE_AUTO
        self._pending_manual = None    # focus value to jump to, or None
        self._force_hunt     = False

        # Published state (read by telemetry).
        self._focus_pos       = self.FOCUS_MIN
        self._peak_sharpness  = 0.0
        self._last_dir        = 1
        self._state           = STATE_STARTUP

        self._stop       = threading.Event()
        self._worker     = threading.Thread(target=self._run, daemon=True,
                                            name="autofocus")

    # ── Lifecycle ───────────────────────────────────────────────────────────
    def start(self):
        self._worker.start()

    def close(self):
        self._stop.set()
        self._worker.join(timeout=2.0)

    # ── Fed by the vision loop (cheap, no hardware I/O) ──────────────────────
    def report_frame(self, frame):
        """
        Called every vision-loop iteration with the latest camera frame.
        Computes one sharpness scalar over a centre ROI and stores it for
        the worker thread. Does NO hardware I/O and never blocks the loop.
        """
        val = self._sharpness(frame)
        with self._sample_lock:
            self._sharpness_val = val
            self._sample_id += 1

    @classmethod
    def _sharpness(cls, frame):
        h, w = frame.shape[:2]
        x0, x1 = int(w * cls.ROI_X[0]), int(w * cls.ROI_X[1])
        y0, y1 = int(h * cls.ROI_Y[0]), int(h * cls.ROI_Y[1])
        roi = frame[y0:y1, x0:x1]
        if roi.ndim == 3:
            roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        # Variance of the Laplacian: high for crisp edges, low when blurred.
        return float(cv2.Laplacian(roi, cv2.CV_64F).var())

    # ── Web-driven controls ─────────────────────────────────────────────────
    def set_auto(self, enabled):
        """Enable continuous AF, or drop into manual (motor held where it is)."""
        with self._cmd_lock:
            self._mode = MODE_AUTO if enabled else MODE_MANUAL
            if enabled:
                # Re-baseline so re-enabling doesn't instantly trigger a hunt
                # off a stale peak recorded before the user moved the lens.
                self._peak_sharpness = 0.0
        return self._mode == MODE_AUTO

    def set_manual_focus(self, value):
        """Jump focus to an absolute value and switch to manual control."""
        with self._cmd_lock:
            self._mode           = MODE_MANUAL
            self._pending_manual = int(value)

    def trigger_refocus(self):
        """Force a one-shot hunt right now (the 'Focus now' button)."""
        with self._cmd_lock:
            self._force_hunt = True

    # ── Reporting ───────────────────────────────────────────────────────────
    @property
    def mode(self):
        return self._mode

    @property
    def state(self):
        return self._state

    @property
    def focus_position(self):
        return self._focus_pos

    def limits(self):
        return {"min": self.FOCUS_MIN, "max": self.FOCUS_MAX}

    def status(self):
        with self._sample_lock:
            sharp = self._sharpness_val
        return {
            "mode":      self._mode,
            "state":     self._state,
            "focus":     self._focus_pos,
            "sharpness": round(sharp, 1) if sharp is not None else None,
            "limits":    self.limits(),
        }

    # ── Worker thread ────────────────────────────────────────────────────────
    def _run(self):
        if self._do_startup_sweep:
            try:
                self._full_sweep()
            except Exception:
                log.exception("Autofocus startup sweep failed")
        self._state = STATE_IDLE

        soft_count = 0
        while not self._stop.is_set():
            # ── Consume any pending web commands first ──────────────────────
            with self._cmd_lock:
                pending = self._pending_manual
                self._pending_manual = None
                force = self._force_hunt
                self._force_hunt = False
                mode = self._mode

            if pending is not None:
                self._state = STATE_MANUAL
                try:
                    self._set_focus(pending)
                except Exception:
                    log.exception("Manual focus write failed")
                continue

            if mode == MODE_MANUAL and not force:
                self._state = STATE_MANUAL
                time.sleep(0.1)
                continue

            # ── Continuous AF: watch, and hunt only when it goes soft ───────
            s = self._measure(settle_frames=1)
            if s is None:
                time.sleep(0.05)
                continue

            if self._peak_sharpness <= 0:
                self._peak_sharpness = s

            trigger = force or (s < self._peak_sharpness * self.TRIGGER_RATIO)
            if trigger:
                soft_count += 1
            else:
                soft_count = 0
                if s > self._peak_sharpness:   # scene got crisper on its own
                    self._peak_sharpness = s
                self._state = STATE_IDLE

            if force or soft_count >= self.SOFT_FRAMES:
                soft_count = 0
                try:
                    self._hunt()
                except Exception:
                    log.exception("Autofocus hunt failed")

            time.sleep(0.03)

    # ── Focus hardware helper (lock-protected) ──────────────────────────────
    def _clamp_focus(self, value):
        return int(max(self.FOCUS_MIN, min(self.FOCUS_MAX, value)))

    def _set_focus(self, value):
        value = self._clamp_focus(value)
        # flag=1 → Focuser waits for the motor to actually settle before
        # returning, so the next sharpness sample reflects THIS focus, not a
        # position the lens is still travelling toward.
        with self._io_lock:
            self._focuser.set(Focuser.OPT_FOCUS, value)
        self._focus_pos = value
        return value

    def _measure(self, settle_frames=None):
        """
        Wait for `settle_frames` fresh sharpness samples from the vision loop
        (i.e. captured after the most recent focus move) and return their
        average, or None if none arrive in time.
        """
        want = settle_frames or self.SETTLE_FRAMES
        vals = []
        with self._sample_lock:
            last_id = self._sample_id
        deadline = time.monotonic() + 1.5
        while len(vals) < want and not self._stop.is_set():
            if time.monotonic() > deadline:
                break
            with self._sample_lock:
                cur_id, cur_val = self._sample_id, self._sharpness_val
            if cur_id != last_id and cur_val is not None:
                last_id = cur_id
                vals.append(cur_val)
            else:
                time.sleep(0.005)
        if not vals:
            return None
        return sum(vals) / len(vals)

    # ── Startup: full-range rack + fine settle ──────────────────────────────
    def _full_sweep(self):
        self._state = STATE_STARTUP
        log.info("Autofocus: startup full-range sweep")
        best_val, best_pos = -1.0, self._clamp_focus(self._focus_pos)

        # Park at the near end and let exposure/frames settle first.
        self._set_focus(self.FOCUS_MIN)
        self._measure(settle_frames=self.SETTLE_FRAMES)

        pos = self.FOCUS_MIN
        while pos <= self.FOCUS_MAX and not self._stop.is_set():
            self._set_focus(pos)
            s = self._measure()
            if s is not None and s > best_val:
                best_val, best_pos = s, pos
            pos += self.COARSE_STEP

        # Fine local search bracketing the coarse peak.
        lo = self._clamp_focus(best_pos - self.FINE_SPAN)
        hi = self._clamp_focus(best_pos + self.FINE_SPAN)
        pos = lo
        while pos <= hi and not self._stop.is_set():
            self._set_focus(pos)
            s = self._measure()
            if s is not None and s > best_val:
                best_val, best_pos = s, pos
            pos += self.FINE_STEP

        self._set_focus(best_pos)
        self._peak_sharpness = max(best_val, 0.0)
        log.info("Autofocus: sweep done — focus=%d sharpness=%.1f",
                 best_pos, best_val)

    # ── Runtime: small directional hill-climb ────────────────────────────────
    def _hunt(self):
        """
        Local, directional refocus. Probe one small step each way; if neither
        improves on where we are, the softness was scene content (not
        defocus) — re-baseline and bail. Otherwise climb the winning
        direction until sharpness stops rising.
        """
        self._state = STATE_HUNTING
        base_pos = self._focus_pos
        base_s   = self._measure()
        if base_s is None:
            self._state = STATE_IDLE
            return

        direction = self._last_dir or 1
        probe_pos = base_pos
        probe_s   = base_s
        improved  = False

        for _ in range(2):   # try preferred direction, then the other
            cand = self._clamp_focus(base_pos + direction * self.HUNT_STEP)
            if cand != base_pos:
                self._set_focus(cand)
                s = self._measure()
                if s is not None and s > base_s * self.IMPROVE_RATIO:
                    improved, probe_pos, probe_s = True, cand, s
                    break
            direction = -direction

        if not improved:
            # Already at the focus peak; treat the softness as a scene/content
            # change and adopt the current sharpness as the new reference so we
            # don't keep re-triggering on it.
            self._set_focus(base_pos)
            self._peak_sharpness = max(base_s, self._peak_sharpness * 0.9)
            self._last_dir = direction
            self._state = STATE_IDLE
            log.info("Autofocus: no focus gain either way — scene change, holding")
            return

        # Hill-climb in the winning direction.
        best_pos, best_s = probe_pos, probe_s
        self._last_dir = direction
        steps = 0
        while steps < self.HUNT_MAX_STEPS and not self._stop.is_set():
            steps += 1
            cand = self._clamp_focus(best_pos + direction * self.HUNT_STEP)
            if cand == best_pos:
                break   # hit a travel limit
            self._set_focus(cand)
            s = self._measure()
            if s is None:
                continue
            if s > best_s:
                best_s, best_pos = s, cand
            elif s < best_s * self.WORSE_RATIO:
                break   # clearly past the peak

        self._set_focus(best_pos)
        self._peak_sharpness = best_s
        self._state = STATE_IDLE
        log.info("Autofocus: refocused -> focus=%d sharpness=%.1f",
                 best_pos, best_s)
