"""Offline check of the optical->digital zoom handoff. No hardware needed."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from zoom_webcam import ZoomWebcamController

W, H = 640, 480


class LM:
    def __init__(self, x, y, vis=1.0):
        self.x, self.y, self.visibility = x, y, vis


def pose(face_h=0.05):
    """A small (distant) person centred in frame: face at top, shoulders below."""
    lms = [LM(0.5, 0.5, 0.0) for _ in range(33)]
    for i in range(9):                      # FACE_IDS
        lms[i] = LM(0.5, 0.30 + face_h * (i / 8.0), 0.9)
    lms[11] = LM(0.46, 0.30 + face_h * 2.0, 0.9)   # shoulders
    lms[12] = LM(0.54, 0.30 + face_h * 2.0, 0.9)
    return lms


def settle(z, n=200, **opt):
    for _ in range(n):
        z.set_optical(opt or None)
        z.update(pose(), W, H)
    return z.zoom


print("scenario                                    digital zoom")
print("-" * 62)

# 1. No optical zoom at all -> behaves as before, digital does the work.
z = ZoomWebcamController(enabled=True, control_zone_margin=0.15)
base = settle(z)
print(f"no optical (baseline)                       {base:.3f}")
assert base > 1.05, "digital should zoom in on a distant subject when alone"

# 2. Optical has headroom -> digital must NOT zoom in past 1.0.
z = ZoomWebcamController(enabled=True, control_zone_margin=0.15)
held = settle(z, mag=1.0, headroom=1.0, moving=False)
print(f"optical has headroom -> digital held        {held:.3f}")
assert abs(held - 1.0) < 1e-6, f"digital should stay at 1.0, got {held}"

# 3. Optical at its stop -> digital takes over and fills the gap.
z = ZoomWebcamController(enabled=True, control_zone_margin=0.15)
filled = settle(z, mag=3.0, headroom=0.0, moving=False)
print(f"optical maxed -> digital fills the gap      {filled:.3f}")
assert filled > 1.05, "digital should engage once the lens is out of travel"

# 4. Motor mid-travel -> digital frozen even with no headroom left.
z = ZoomWebcamController(enabled=True, control_zone_margin=0.15)
frozen = settle(z, mag=2.0, headroom=0.0, moving=True)
print(f"motor moving -> digital frozen              {frozen:.3f}")
assert abs(frozen - 1.0) < 1e-6, f"digital should freeze mid-travel, got {frozen}"

# 5. Zooming OUT is never held back, even with optical headroom available.
z = ZoomWebcamController(enabled=True, control_zone_margin=0.15)
settle(z, mag=3.0, headroom=0.0, moving=False)   # get digital zoomed in
zoomed_in = z.zoom
for _ in range(300):                              # subject leaves frame
    z.set_optical({"mag": 3.0, "headroom": 1.0, "moving": False})
    z.update(None, W, H)
print(f"zoom-out with headroom  {zoomed_in:.3f} ->        {z.zoom:.3f}")
assert z.zoom < zoomed_in - 0.05, "zoom-out must never be blocked by the handoff"

# 6. total_mag reports the combined figure.
z = ZoomWebcamController(enabled=True, control_zone_margin=0.15)
z.set_optical({"mag": 2.5, "headroom": 0.0, "moving": False})
z.zoom = 1.4
print(f"total_mag = 2.5 opt x 1.4 dig               {z.total_mag:.3f}")
assert abs(z.total_mag - 3.5) < 1e-6

print("-" * 62)
print("all handoff assertions passed")
