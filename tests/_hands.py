"""
Synthetic MediaPipe hands for the gesture tests.

Shared so importing a helper doesn't drag another test suite's assertions
along as an import side effect.
"""
import numpy as np

# MediaPipe hand layout: 0 wrist, then 4 landmarks per finger.
_CHAINS = {"thumb": (1, 2, 3, 4), "index": (5, 6, 7, 8),
           "middle": (9, 10, 11, 12), "ring": (13, 14, 15, 16),
           "pinky": (17, 18, 19, 20)}
# Splay each finger sideways so they don't sit on top of one another.
_SPLAY = {"thumb": -0.9, "index": -0.3, "middle": 0.0, "ring": 0.3, "pinky": 0.6}

ALL = ("thumb", "index", "middle", "ring", "pinky")
PEACE = ("index", "middle")


def hand(extended, wrist=(0.5, 0.5)):
    """
    A hand pointing 'up' (-y). `extended` names the fingers that are
    straight; the rest curl back toward the palm, which is what makes the
    tip land closer to the wrist than its own PIP joint.
    """
    pts = np.zeros((21, 3), dtype=np.float32)
    pts[0] = (wrist[0], wrist[1], 0.0)
    for name, (a, b, c, d) in _CHAINS.items():
        dx = _SPLAY[name] * 0.05
        out = name in extended
        for k, idx in enumerate((a, b, c, d), start=1):
            if out:
                reach = 0.05 * k                     # straight: tip furthest
            else:
                reach = 0.05 * min(k, 2) - 0.035 * max(0, k - 2)   # curled back
            pts[idx] = (wrist[0] + dx * k / 4.0, wrist[1] - reach, 0.0)
    return pts


def rotate(pts, deg):
    """Rotate a hand about its wrist, to check orientation-invariance."""
    r = np.radians(deg)
    c, s = np.cos(r), np.sin(r)
    out = pts.copy()
    rel = pts[:, :2] - pts[0, :2]
    out[:, 0] = pts[0, 0] + rel[:, 0] * c - rel[:, 1] * s
    out[:, 1] = pts[0, 1] + rel[:, 0] * s + rel[:, 1] * c
    return out
