"""
latency_log.py
───────────────
Measures how long it actually takes for a hand movement to become a cursor
movement on the paired PC.

The pipeline already reports fps, and fps is the wrong number. Throughput
and latency come apart badly here: every filter in the chain trades one for
the other, and a loop can run at a perfectly respectable frame rate while
the pointer sits a third of a second behind the hand. Nothing in the system
measured that, so every tuning decision about responsiveness was being made
from a guess.

The chain this breaks down, in order:

    sensor exposure ─► driver/buffer ─► our loop ─► detect ─► filters ─► HID

  * `driver_ms`  — sensor timestamp to the frame arriving in read(). Buffer
                   depth and ISP; not something the code controls directly.
  * `detect_ms`  — MediaPipe hand/face/pose for this frame.
  * `pipeline_ms`— read() returning to the HID report going out. Everything
                   we actually own.
  * `total_ms`   — sensor to HID. The number a user feels, minus the
                   exposure window itself and whatever the host adds.

`SensorTimestamp` is CLOCK_BOOTTIME on the Pi, so it is compared against
`time.clock_gettime(CLOCK_BOOTTIME)` rather than perf_counter — mixing the
two clocks would put an arbitrary constant in every row. Where the sensor
timestamp is missing (the USB webcam path), driver_ms is left blank rather
than filled with a fabricated zero.

Enabled by LATENCY_LOG, off by default and free when off:

    LATENCY_LOG=1                 -> logs/latency_<timestamp>.csv
    LATENCY_LOG=/some/path.csv    -> that file

**It must be set in the SERVER's own environment** (the PyCharm run
configuration, not a shell that merely spawns it), and — like the low-light
log — rows only appear once a logged-in dashboard is connected, because the
pipeline suspends itself until then.
"""
import csv
import logging
import os
import time

log = logging.getLogger(__name__)

LOG_HZ = 10.0

_COLUMNS = [
    "t", "frame", "fps",
    "driver_ms", "detect_ms", "pipeline_ms", "total_ms",
    "dt_ms", "action", "moved",
]


def boottime():
    """The clock SensorTimestamp is expressed in."""
    try:
        return time.clock_gettime(time.CLOCK_BOOTTIME)
    except (AttributeError, OSError):
        # Not Linux, or no BOOTTIME — monotonic is the closest thing, and
        # only the sensor-to-read delta is affected.
        return time.monotonic()


class LatencyLog:
    """One CSV row per sampled frame. maybe_log() returns immediately when
    disabled, so this can stay wired into the loop permanently."""

    def __init__(self, path=None, log_hz=LOG_HZ):
        self.enabled = False
        self._fh = None
        self._writer = None
        self._period = 1.0 / log_hz if log_hz > 0 else 0.0
        self._last_t = -1e9
        self.path = None

        # Stripped for the same reason as the low-light log: this is usually
        # typed into an IDE run-configuration field where a trailing space is
        # invisible, and would otherwise be read as a filename.
        env = (os.environ.get("LATENCY_LOG", "") if path is None
               else str(path)).strip()
        if not env or env == "0":
            return

        if env in ("1", "true", "yes"):
            d = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
            os.makedirs(d, exist_ok=True)
            env = os.path.join(d, time.strftime("latency_%Y%m%d_%H%M%S.csv"))

        try:
            self._fh = open(env, "w", newline="", buffering=1)
        except OSError:
            log.exception("Could not open latency log '%s'", env)
            return
        self._writer = csv.writer(self._fh)
        self._writer.writerow(_COLUMNS)
        self.enabled = True
        self.path = env
        log.info("Latency logging to %s", env)
        print(f"Latency logging to {env}")

    def close(self):
        if self._fh is not None:
            self._fh.close()
            self._fh = None
        self.enabled = False

    @staticmethod
    def sensor_age_ms(metadata, read_boottime):
        """
        Milliseconds between the sensor capturing the frame and read()
        handing it to us. None when the camera reports no timestamp.
        """
        if not metadata:
            return None
        ts_ns = metadata.get("SensorTimestamp")
        if not ts_ns:
            return None
        return (read_boottime - ts_ns / 1e9) * 1000.0

    def maybe_log(self, now_t, *, frame_n=0, fps=0.0, dt=0.0,
                  driver_ms=None, detect_t=None, read_t=None,
                  sent_t=None, action=""):
        """
        Sample this frame if it is time to. Rate-limited rather than
        every-frame because the point is a distribution, not a trace, and
        the log should not itself become a latency source.

        read_t/detect_t/sent_t are perf_counter stamps; sent_t is None on
        frames that produced no cursor movement, which are still worth a row
        (they show what the pipeline costs when it is not steering).
        """
        if not self.enabled:
            return
        if (now_t - self._last_t) < self._period:
            return
        self._last_t = now_t

        def ms(a, b):
            return round((b - a) * 1000.0, 2) if (a is not None
                                                  and b is not None) else ""

        pipeline_ms = ms(read_t, sent_t)
        total_ms = ""
        if driver_ms is not None and pipeline_ms != "":
            total_ms = round(driver_ms + pipeline_ms, 2)

        self._writer.writerow([
            round(now_t, 4),
            frame_n,
            round(fps, 1),
            round(driver_ms, 2) if driver_ms is not None else "",
            ms(read_t, detect_t),
            pipeline_ms,
            total_ms,
            round(dt * 1000.0, 2),
            action,
            1 if sent_t is not None else 0,
        ])
