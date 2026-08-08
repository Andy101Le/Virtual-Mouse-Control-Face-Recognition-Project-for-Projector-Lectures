# Offline checks

Hardware-free tests for the long-range tracking and face-recognition work.
No camera, no I2C, no PTZ board — run them anywhere:

    python3 tests/test_zoom_handoff.py
    python3 tests/test_detector_telephoto.py
    python3 tests/test_face_recognition.py

They cover the logic that is easy to get silently wrong, not the hardware
and not SFace's own accuracy (which needs real faces to measure).

**`test_zoom_handoff.py`** — the optical/digital zoom split. Digital must
not crop while the lens still has magnification to give losslessly, must
freeze while the motor is travelling, must still be free to zoom *out* at
any time, and must take over once the lens is out of travel.

**`test_detector_telephoto.py`** — the detection crop and landmark remap.
Checks the crop keeps frame aspect ratio, is clamped inside the frame, and
gives the expected 2.0x pixels on a distant face; that `_remap` round-trips
crop-space landmarks back to full-frame coordinates exactly. Also carries
regressions for two bugs found in review: the crop must keep tightening as
someone walks away (its face-size estimate used to freeze once the face
stopped being detectable), and a bystander must not be able to hold the
crop indefinitely with the real user cropped out of frame.

**`test_face_recognition.py`** — the SFace matching path. Pre-SFace
embeddings must be rejected as needing re-enrollment rather than misread;
enrollment samples must survive the pack/save/load round-trip; the
distance-aware threshold ramp must stay monotonic and clamped. Two
regressions here too: a registered bystander must not be able to make the
active user return UNKNOWN, and `AuthManager` must pick the face that
actually matches rather than whichever one MediaPipe returned first.

The face test copies `login_system.db` to a temp directory — it never
writes to the real database.

## Hardware check

`test_zoom_sweep.py` in the repo root is different: it drives the real
lens and needs the server stopped. It exists to answer one question the
software cannot — whether the Arducam zoom channel actually moves the
lens, or only ACKs the I2C write like the dead tilt channel did.
