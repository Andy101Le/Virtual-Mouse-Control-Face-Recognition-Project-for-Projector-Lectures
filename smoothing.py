"""
smoothing.py
─────────────
Frame-rate-independent exponential smoothing.

The filters in this pipeline used to be written as a fixed per-frame weight
(`x += 0.35 * (target - x)`). That is a bug in disguise: the lag such a
filter introduces is measured in FRAMES, so its behaviour in seconds — the
only thing a user can feel — depends entirely on how fast the loop happens
to be running. The loop here varies from ~12 to ~30 fps depending on how
many people are in shot and what the detectors are doing, which means the
cursor got laggier exactly when the system was busiest, and every
performance regression silently became a responsiveness regression too.

Expressing the same filter as a time constant fixes both. `tau` is the time
to close ~63% of the gap to the target, in seconds, and it means the same
thing at any frame rate.

Conversion, for reading the old constants:

    tau   = -dt / ln(1 - alpha)
    alpha = 1 - exp(-dt / tau)

so the old cursor filter (alpha 0.35 at ~13 fps) was tau ~= 0.18 s, and the
old landmark filter (alpha 0.30) was tau ~= 0.22 s. Those two sit in series
on the cursor path, which is where most of the input lag was coming from.
"""
import math


class OneEuroFilter:
    """
    Speed-adaptive smoothing (Casiez, Roussel & Vogel, CHI 2012).

    A single time constant cannot serve a pointer. Make it long and slow
    precise movement is beautifully steady while fast movement lags; make it
    short and fast movement is crisp while the pointer shivers whenever you
    hold still trying to hit something small. Both complaints are real and
    they are the same constant pulling in opposite directions.

    One Euro drops the constant and varies the cutoff with how fast the
    input is actually moving: nearly still, it filters hard, so jitter dies;
    moving quickly, it barely filters at all, so nothing lags. That maps
    exactly onto how a pointer gets used — you sweep coarsely, then settle.

    `min_cutoff` sets the behaviour at rest (lower = steadier), `beta` sets
    how eagerly it opens up with speed (higher = less lag when moving). With
    beta = 0 this degenerates to a plain EMA at min_cutoff, which is the
    escape hatch back to fixed-time-constant behaviour.
    """

    def __init__(self, min_cutoff, beta, d_cutoff=1.0):
        self.min_cutoff = float(min_cutoff)
        self.beta       = float(beta)
        self.d_cutoff   = float(d_cutoff)
        self._x     = None   # last filtered value
        self._x_raw = None   # last RAW input, for the derivative
        self._dx    = 0.0    # filtered derivative, in units per second

    @staticmethod
    def _tau(cutoff):
        return 1.0 / (2.0 * math.pi * cutoff) if cutoff > 0 else float("inf")

    def reset(self, value=None):
        """
        Forget history, optionally seeding the output at `value`.

        Dropping `_x_raw` is the important half: the next sample then
        contributes no derivative at all. Without that, resuming after a
        pause — where the input has legitimately moved but is not MOVING —
        reads as enormous speed, opens the cutoff wide, and lets the pointer
        teleport, which is precisely what the caller reset it to avoid.
        """
        self._x     = value
        self._x_raw = None
        self._dx    = 0.0

    def __call__(self, x, dt, beta=None):
        """`beta` overrides the configured value for this sample; pass 0.0 to
        suppress speed adaptation and behave as a plain EMA at min_cutoff."""
        x = float(x)
        if self._x is None or dt <= 0.0:
            self._x = self._x_raw = x
            return self._x

        # Speed is measured between successive INPUTS, not against our own
        # filtered output. Using the output would conflate the filter's own
        # lag with real movement: any large tracking error would read as
        # speed, unfilter itself, and snap — the exact behaviour this filter
        # exists to prevent.
        dx = 0.0 if self._x_raw is None else (x - self._x_raw) / dt
        self._x_raw = x

        # Filter the derivative before it steers the cutoff, or one noisy
        # sample reads as fast movement and unfilters the very jitter this
        # is meant to remove.
        self._dx += ema_alpha(dt, self._tau(self.d_cutoff)) * (dx - self._dx)

        b = self.beta if beta is None else float(beta)
        cutoff = self.min_cutoff + b * abs(self._dx)
        self._x += ema_alpha(dt, self._tau(cutoff)) * (x - self._x)
        return self._x


def ema_alpha(dt, tau):
    """
    Per-sample EMA weight that realises a `tau`-second time constant when
    samples arrive `dt` seconds apart.

    tau <= 0 disables smoothing (follow the input exactly). A non-positive
    dt returns 0.0 — no time has passed, so the estimate cannot have moved;
    this also makes the first frame safe, before any dt is known.
    """
    if tau <= 0.0:
        return 1.0
    if dt <= 0.0:
        return 0.0
    return 1.0 - math.exp(-dt / tau)
