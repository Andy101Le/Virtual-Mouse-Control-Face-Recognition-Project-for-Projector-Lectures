# Offline checks

Hardware-free tests for the long-range tracking and face-recognition work.
No camera, no I2C, no PTZ board — run them anywhere:

    python3 tests/test_zoom_handoff.py
    python3 tests/test_detector_telephoto.py
    python3 tests/test_face_recognition.py
    python3 tests/test_control_zone.py
    python3 tests/test_control_toggle.py
    python3 tests/test_middle_click.py
    python3 tests/test_gesture_mlp.py
    python3 tests/test_pose_rate.py
    python3 tests/test_scene_light.py

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

**`test_control_zone.py`** — the reach-based cursor mapping. The zone used
to be a fixed box at every distance, so reaching the screen edge meant
sweeping your hand 85% of the way across the camera frame however far away
you stood; at range a fully extended arm reached only the middle 22% of the
screen. The test prints old-vs-new screen coverage per apparent face size
and asserts full extension spans the whole screen at every distance. It
also checks the zone is clamped by shifting rather than squashing (so
sensitivity stays uniform left-to-right), that zone corners map exactly to
screen corners, and that the hand-ownership radius tightens with distance
instead of covering half the room.

**`test_control_toggle.py`** — the open-palm master switch. The pose
detectors are geometric rather than a model class (the Keras model has only
five outputs), so they're testable without a camera: the test builds
synthetic hands and checks open-palm and peace-sign detection hold under
rotation and scale, and reject each other. It then drives the state machine
through arm -> hold -> fire -> peace -> lower -> re-arm. The load-bearing
assertion is that holding the palm up does NOT toggle repeatedly, which is
the whole reason the reset sequence exists. Also checks a hand hanging at
your side is ignored, bystander-only frames never toggle, and the dashboard
override re-arms rather than leaving the switch mid-sequence.

**`test_middle_click.py`** — the peace-sign middle click and, mainly, its
boundary with the master switch. The peace sign has two jobs: normally it
sends a middle click, but the switch borrows it as the reset step of its
toggle sequence. The important case is switching controls back ON, because
afterwards controls are enabled and the reset peace sign would otherwise be
a live middle click — the test drives that whole sequence and asserts zero
clicks, then asserts a peace sign clicks again once the switch has
re-armed. Also covers: a held sign fires exactly once rather than
streaming, a brief flicker of the pose fires nothing, nothing fires while
controls are paused, and a disconnected host doesn't silently swallow a
click by latching it.

**`test_gesture_mlp.py`** — the numpy replacement for the Keras gesture
model. `gesture_mlp.py` re-implements the forward pass by hand, so the only
thing that makes it safe is being numerically identical to what it replaced:
the test compares the two over 800 rows of random and synthetic-hand inputs
and asserts no argmax disagreements and no flips across the 0.75 confidence
gate (worst probability difference measured: 3.7e-06). It also checks the
loader rejects architectures it cannot reproduce rather than quietly
computing something else, and that BatchNorm — which sits *after* the ReLU
here and so cannot be folded into the preceding Dense — is handled
correctly. **This is the one suite that needs TensorFlow.** It skips the
comparison if TF is missing, so re-run it with TF installed after any
retraining.

**`test_pose_rate.py`** — how often the pose landmarker runs, which is
purely a performance decision: it is the most expensive detector in the loop
(~53 ms/call measured under load) and feeds only slow, smoothed consumers.
It used to run every 2nd frame, tying its rate to the loop's frame rate, so
any speed-up elsewhere was immediately spent on more pose detections. It now
runs at a fixed 5 Hz off the frame timestamp. The test asserts the target is
a *ceiling* — a slow loop rounds the rate down, which is the safe direction —
that a faster loop doesn't spend the gain on more detections, and that
falling behind produces one detection rather than a catch-up burst. Measured
saving at the ~12 fps this pipeline achieves: 8.8 ms/frame, rising to
17.6 ms/frame if the loop ever reaches 30.

**`test_scene_light.py`** — the low-light instrumentation. Its job is to
separate two failures that look identical from inside the detector ("no face
found"): a uniformly dim room, which needs more light, and a subject who is
dark only because the projector screen behind them is bright, which needs
different metering and gets no benefit from more exposure. The test builds
both frames synthetically and asserts the subject/scene luminance ratio
separates them (1.07 vs 0.18). It then walks the region-selection fallbacks
— face, then pose head, then tracked crop, then whole frame — because the
measurement has to survive exactly the moment the face stops being
detectable. Also covers clamping rects to the frame, the sample grid being
centred so a side-lit face isn't read from its dark half, a bystander not
capturing the measurement, and the logger's two behaviours: rate-limiting
redundant samples, and forcing a row the instant a face or the auth state is
lost, which is the row worth having.

`_hands.py` holds the synthetic hand builder shared by the gesture tests,
so importing a helper doesn't run another suite's assertions as a side
effect.

The face test copies `login_system.db` to a temp directory — it never
writes to the real database.

## Hardware check

`test_zoom_sweep.py` in the repo root is different: it drives the real
lens and needs the server stopped. It exists to answer one question the
software cannot — whether the Arducam zoom channel actually moves the
lens, or only ACKs the I2C write like the dead tilt channel did.
