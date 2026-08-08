"""
ptz_manual.py
──────────────
Manual pan / tilt / zoom overrides for the web UI's control pad.

Coexisting with auto-tracking is the whole problem here. PTZController
runs a background thread that continuously drives the motors toward the
tracked face. If the web UI wrote to the motors at the same time, the
two would fight and the camera would judder. So:

  - Entering manual mode calls ptz.set_enabled(False), which stops the
    auto-tracker from issuing any further motor commands.
  - Manual moves then write through the SAME Focuser instance the
    tracker owns, and update the tracker's commanded-target bookkeeping
    (_targets) in step.
  - Returning to auto calls ptz.set_enabled(True). Because _targets was
    kept in sync, the tracker resumes from where the camera actually is
    rather than snapping back to a stale target.

That last point is why this class deliberately touches ptz._focuser and
ptz._targets: keeping the tracker's own notion of "where I last
commanded the motors" honest is exactly what prevents a jump on the
auto/manual handoff.
"""

import logging

from Focuser import Focuser

log = logging.getLogger(__name__)

MODE_AUTO   = "auto"
MODE_MANUAL = "manual"


def _opt_bounds(opt, fallback_lo, fallback_hi):
    """
    Arducam's full Focuser defines MIN/MAX for every OPT_*. Some trimmed
    copies of Focuser.py in the wild only define OPT_FOCUS, which would
    make Focuser.opts[OPT_MOTOR_X] raise KeyError. Fall back to the
    documented PTZ ranges rather than crashing the whole web server.
    """
    info = Focuser.opts.get(opt)
    if not info:
        log.warning("Focuser.opts is missing entry for opt 0x%04x — using "
                    "fallback range %d..%d. If pan/tilt behaves oddly, your "
                    "Focuser.py is a trimmed copy; use Arducam's full version.",
                    opt, fallback_lo, fallback_hi)
        return fallback_lo, fallback_hi
    return info["MIN_VALUE"], info["MAX_VALUE"]


class ManualPTZ:
    """Wraps a live PTZController with web-driven manual control."""

    def __init__(self, ptz):
        self.ptz  = ptz
        self.mode = MODE_AUTO

        self.pan_lo,  self.pan_hi  = _opt_bounds(Focuser.OPT_MOTOR_X, 0, 180)
        self.tilt_lo, self.tilt_hi = _opt_bounds(Focuser.OPT_MOTOR_Y, 0, 180)

        # Hardware (optical) zoom is a separate motor from the digital
        # zoom in zoom_webcam.py. Not every PTZ board populates it.
        self.has_hw_zoom = Focuser.OPT_ZOOM in Focuser.opts
        if self.has_hw_zoom:
            self.zoom_lo, self.zoom_hi = _opt_bounds(Focuser.OPT_ZOOM, 0, 1000)
        else:
            self.zoom_lo = self.zoom_hi = 0
            log.info("No hardware zoom on this Focuser — the web zoom slider "
                     "will drive digital zoom (zoom_webcam) only.")

    # ── Mode ────────────────────────────────────────────────────────────────
    def set_mode(self, mode):
        if mode not in (MODE_AUTO, MODE_MANUAL):
            raise ValueError(f"Unknown PTZ mode: {mode}")
        self.mode = mode
        # Auto-tracking must be off while a human is steering, or the two
        # will fight over the motors.
        self.ptz.set_enabled(mode == MODE_AUTO)
        log.info("PTZ mode -> %s", mode)
        return self.mode

    # ── Absolute / relative motion ──────────────────────────────────────────
    def _write(self, opt, value, lo, hi, target_attr=None):
        value = int(max(lo, min(hi, value)))
        try:
            # Share the PTZ's I2C lock: these writes come from the Flask
            # thread and must not collide with the tracker or autofocus.
            with self.ptz.io_lock:
                self.ptz._focuser.set(opt, value, 0)
        except Exception:
            log.exception("Focuser write failed (opt=0x%04x, value=%s)", opt, value)
            return None
        # Keep the auto-tracker's commanded-target bookkeeping in step so
        # it doesn't snap the camera back when auto mode resumes.
        if target_attr is not None:
            setattr(self.ptz._targets, target_attr, value)
        return value

    def set_pan(self, value):
        return self._write(Focuser.OPT_MOTOR_X, value,
                           self.pan_lo, self.pan_hi, "motor_x")

    def set_tilt(self, value):
        return self._write(Focuser.OPT_MOTOR_Y, value,
                           self.tilt_lo, self.tilt_hi, "motor_y")

    def nudge(self, d_pan=0, d_tilt=0):
        """Relative move — what the D-pad buttons call."""
        if d_pan:
            self.set_pan(self.ptz._targets.motor_x + d_pan)
        if d_tilt:
            self.set_tilt(self.ptz._targets.motor_y + d_tilt)
        return self.position()

    def set_hw_zoom(self, value):
        if not self.has_hw_zoom:
            return None
        # Same reason pan/tilt sync their targets: the tracker's optical
        # auto-zoom reasons from _targets.zoom rather than re-reading the
        # register, and _zoom_mag() scales the pan/tilt gains off it. Left
        # stale, a manual zoom would make the tracker think the lens is
        # still wide — it would step from the wrong position and drive
        # pan/tilt with gains meant for a much wider field of view.
        written = self._write(Focuser.OPT_ZOOM, value,
                              self.zoom_lo, self.zoom_hi, "zoom")
        if written is None:
            return None
        # A manual zoom shifts the focus plane exactly as an automatic one
        # does, so it needs the same autofocus coordination the auto path
        # gets — otherwise AF chases the blur across the slider drag. It also
        # has to publish its own telemetry: the auto loop updates the
        # telemetry dict as it steps, and with the tracker disabled during
        # manual mode nothing else will, so the dashboard's optical-zoom
        # readout would sit stale until auto mode resumed.
        with self.ptz._telemetry_lock:
            self.ptz._telemetry["zoom"] = written
        cb = self.ptz.on_zoom_step
        if cb:
            try:
                cb(written)
            except Exception:
                log.exception("on_zoom_step callback failed")
        return written

    def center(self):
        self.set_pan((self.pan_lo + self.pan_hi) // 2)
        self.set_tilt((self.tilt_lo + self.tilt_hi) // 2)
        return self.position()

    # ── Reporting ───────────────────────────────────────────────────────────
    def position(self):
        return {
            "pan":  self.ptz._targets.motor_x,
            "tilt": self.ptz._targets.motor_y,
        }

    def limits(self):
        return {
            "pan":  {"min": self.pan_lo,  "max": self.pan_hi},
            "tilt": {"min": self.tilt_lo, "max": self.tilt_hi},
            "hw_zoom": ({"min": self.zoom_lo, "max": self.zoom_hi}
                        if self.has_hw_zoom else None),
        }

    def status(self):
        st = {
            "mode":     self.mode,
            "position": self.position(),
            "limits":   self.limits(),
        }
        try:
            st["tracker"] = self.ptz.get_telemetry()
        except Exception:
            st["tracker"] = None
        return st
