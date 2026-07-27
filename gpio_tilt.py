"""
gpio_tilt.py
─────────────
Drive the tilt servo from a Raspberry Pi GPIO pin instead of the Arducam
PTZ board.

Why this exists: the board's tilt channel died. Diagnosis (2026-07-22):
the board ACKs OPT_MOTOR_Y writes over I2C and reads every value back
correctly, but its tilt header never drives a servo — three known-good
servos (SG92R, WS-MG90S, and the SG90 cross-check) all sat silent and
motionless on it while the pan channel kept working. Rather than lose
tilt until the board is replaced, the servo's signal wire moves to a Pi
GPIO pin and this wrapper generates the pulses.

GpioTiltFocuser wraps the real Focuser and reroutes ONLY OPT_MOTOR_Y to
a gpiozero AngularServo; every other option (pan, zoom, focus, IR-cut)
passes straight through to the board. PTZController, ManualPTZ and the
autofocus controller all talk through the same Focuser interface, so
swapping this wrapper in at construction time fixes tilt everywhere at
once — none of them can tell the difference.

Wiring (servo lead → Pi 5 40-pin header):
    orange/yellow (signal) → GPIO18  = physical pin 12
    brown/black   (GND)    → physical pin 14 (any ground pin works)
    red           (5V)     → physical pin 2 or 4

Enable by exporting TILT_GPIO_PIN=18 before starting web_server.py, or
test standalone with:  python3 test_ptz_sweep.py tilt --gpio 18

NOTE on Pi 5: gpiozero's default pin factory (lgpio) generates the servo
pulse train in software. That is fine for an SG92R holding a small
camera, but if you ever notice a fine tremor in the image, move to
hardware PWM (dtoverlay=pwm,pin=18,func=2 in /boot/firmware/config.txt
plus a hardware-PWM pin factory) — same wiring, steadier pulses.
"""

from Focuser import Focuser


class GpioTiltFocuser:
    """Drop-in Focuser replacement: tilt on GPIO, everything else I2C."""

    # Unlike the Arducam board (whole degrees only), the GPIO pulse train
    # can hit any angle — PTZController checks this flag and commands tilt
    # in fractional degrees when it's present.
    FRACTIONAL_TILT = True

    # 0.5–2.5 ms maps the SG92R's full 0–180° travel. If the servo buzzes
    # at the very ends of travel, narrow these toward 0.6/2.4 ms.
    _MIN_PULSE_S = 0.5 / 1000
    _MAX_PULSE_S = 2.5 / 1000

    def __init__(self, bus, tilt_pin=18, initial_angle=90):
        self._board = Focuser(bus)

        # Imported here, not at module top: the rest of the app must keep
        # working on setups without gpiozero when the bypass is disabled.
        from gpiozero import AngularServo
        self._servo = AngularServo(
            tilt_pin,
            min_angle=0, max_angle=180,
            min_pulse_width=self._MIN_PULSE_S,
            max_pulse_width=self._MAX_PULSE_S,
            initial_angle=initial_angle,
        )
        self._tilt_angle = int(initial_angle)

    # ── Focuser interface ───────────────────────────────────────────────────
    def get(self, opt, flag=0):
        if opt == Focuser.OPT_MOTOR_Y:
            # The board's register is dead weight now — report the angle we
            # are actually driving on the GPIO pin.
            return self._tilt_angle
        return self._board.get(opt, flag)

    def set(self, opt, value, flag=1):
        if opt == Focuser.OPT_MOTOR_Y:
            info = Focuser.opts[Focuser.OPT_MOTOR_Y]
            # No int() here — fractional angles are the whole point of the
            # GPIO path (see FRACTIONAL_TILT); only clamp to the valid range.
            value = float(max(info["MIN_VALUE"], min(info["MAX_VALUE"], value)))
            self._servo.angle = value
            self._tilt_angle = value
            return
        return self._board.set(opt, value, flag)

    def close(self):
        """Stop the pulse train and release the pin (servo goes limp)."""
        try:
            self._servo.detach()
            self._servo.close()
        except Exception:
            pass

    def __getattr__(self, name):
        # Anything not overridden (reset, isBusy, waitingForFree, opts…)
        # behaves exactly as the real board.
        return getattr(self._board, name)
