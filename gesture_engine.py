"""
gesture_engine.py
──────────────────
Turns raw per-hand MediaPipe landmarks into a debounced gesture label.

Classification runs inline on the camera thread. It used to run on a
background worker per hand, because the Keras model cost ~9 ms per call
and would have blocked the loop; the numpy forward pass in
`gesture_mlp.py` costs ~0.08 ms, so the queue, thread and lock cost more
than the work they were deferring. Running inline also removes the lag
they introduced — the worker held the *previous* frame's result, and
dropped submissions whenever it fell behind, so a gesture took its
debounce window plus however far the worker had slipped to confirm.

Also owns the EMA smoothing applied to raw hand landmarks before they're
drawn or fed to the model, and the debounce counter that requires a
gesture to repeat for DEBOUNCE_FRAMES consecutive frames before it's
"confirmed" (this is what keeps a single misclassified frame from
triggering a click).
"""

import numpy as np

from gesture_mlp import GestureMLP
from smoothing import ema_alpha

LABEL_MAP = {
    0: 'MOVE', 1: 'LEFT CLICK', 2: 'RIGHT CLICK', 3: 'ZOOM IN', 4: 'ZOOM OUT',
}
CURSOR_FREEZE_ACTIONS = {'LEFT CLICK', 'RIGHT CLICK', 'ZOOM IN', 'ZOOM OUT'}

FINGER_CHAINS = [
    [0, 1, 2, 3, 4], [0, 5, 6, 7, 8], [0, 9, 10, 11, 12],
    [0, 13, 14, 15, 16], [0, 17, 18, 19, 20],
]

# Landmark smoothing, as a time constant rather than a per-frame weight —
# see smoothing.py for why. This was alpha=0.3, i.e. tau ~= 0.22 s at the
# ~13 fps the loop actually runs at, and it sat in series with the cursor
# filter for a combined ~0.4 s of input lag.
#
# It is NOT tuned as low as the cursor filter, because this array feeds the
# CLASSIFIER as well as the pointer. Noisier landmarks mean more frames
# where the predicted label flips, and every flip resets the debounce
# counter — so oversharpening here would cost gesture-confirmation latency
# even while it buys pointer latency. Tune LANDMARK_TAU_S up if gestures
# start feeling unreliable. The pointer's own smoothing is separate and
# speed-adaptive now — see BTCursorController.CURSOR_MIN_CUTOFF_HZ.
LANDMARK_TAU_S = 0.12

# Below this the winning class is treated as no gesture at all, so an
# ambiguous pose can't fire a click on its way to somewhere else.
CONFIDENCE_THRESHOLD = 0.75


class GestureEngine:
    # How many consecutive frames a label must repeat before it is trusted.
    #
    # Five frames is ~0.4 s at the loop's real rate — the right price for a
    # click, and far too much for a pointer. The two failures are not
    # remotely symmetric: a spurious MOVE nudges the cursor a few pixels and
    # the next frame corrects it, while a spurious middle click opens
    # browser tabs and a spurious right click opens a context menu. So MOVE
    # confirms fast and everything that actuates keeps the full window.
    DEBOUNCE_FRAMES      = 5
    MOVE_DEBOUNCE_FRAMES = 2

    # 'NO ACTION' deliberately keeps the long window. It is the label that
    # STOPS the cursor, and confirming it quickly would make the pointer
    # stutter every time a frame or two of the move pose was misread.
    def _frames_needed(self, action):
        return (self.MOVE_DEBOUNCE_FRAMES if action == 'MOVE'
                else self.DEBOUNCE_FRAMES)

    def __init__(self, model_path):
        self.gesture_model = GestureMLP(model_path)
        print(f"Gesture model loaded — {self.gesture_model.n_classes} classes")

        self._smoothed_xyz     = {}
        self._gesture_counters = {}

    def smooth_hand(self, hand_id, raw_pts, dt):
        """
        EMA-smooth this hand's raw (21, 3) landmark array, seeding state
        the first time a given hand_id is seen so it doesn't slowly
        smooth in from zero.

        dt: seconds since the previous frame. The smoothing strength is
        derived from it so the filter behaves the same at any frame rate.
        """
        if hand_id not in self._smoothed_xyz:
            self._smoothed_xyz[hand_id] = raw_pts
            return raw_pts
        a = np.float32(ema_alpha(dt, LANDMARK_TAU_S))
        sxyz = a * raw_pts + (np.float32(1.0) - a) * self._smoothed_xyz[hand_id]
        self._smoothed_xyz[hand_id] = sxyz
        return sxyz

    def classify(self, hand_id, smoothed_xyz):
        """Classify this hand's normalised pose and return the debounced
        (confirmed_action, confidence) pair."""
        pts_n  = smoothed_xyz - smoothed_xyz[0]
        scale  = np.max(np.abs(pts_n)) or 1.0
        pts_n /= scale
        # The model was trained on right-hand samples only, so a left hand
        # (an x-mirrored image of the training data) misclassifies. Rather
        # than trust MediaPipe's handedness label (unreliable on partial
        # views), classify the features AND their x-mirrored twin as one
        # batch and keep whichever orientation scores higher — the wrong
        # chirality reads as an unfamiliar pose and scores low, so either
        # physical hand matches the training chirality. Mirroring after
        # normalisation is safe: negating x changes neither the
        # wrist-relative origin nor the max-abs scale.
        mirrored = pts_n * np.float32([-1.0, 1.0, 1.0])
        batch = np.stack([pts_n.reshape(63),
                          mirrored.reshape(63)]).astype(np.float32)

        probs      = self.gesture_model(batch)
        row        = probs[int(np.argmax(np.max(probs, axis=1)))]
        idx        = int(np.argmax(row))
        confidence = float(row[idx])
        action     = (LABEL_MAP.get(idx, 'NO ACTION')
                      if confidence >= CONFIDENCE_THRESHOLD else 'NO ACTION')

        gc = self._gesture_counters.setdefault(
            hand_id, {'name': action, 'count': 0, 'confirmed': action})
        if gc['name'] == action:
            gc['count'] += 1
        else:
            gc['name']  = action
            gc['count'] = 1
        if gc['count'] >= self._frames_needed(action):
            gc['confirmed'] = action

        # `action` is the RAW label for this frame, before debouncing. The
        # caller needs it to stop the cursor the instant a click pose starts
        # forming rather than after it is confirmed: forming a click curls
        # the hand, which drags the fingertip, and for the whole debounce
        # window the confirmed label is still MOVE — so the pointer faithfully
        # followed that drag and slid off whatever you were aiming at. That
        # is invisible on a big target and fatal on a small one.
        return gc['confirmed'], confidence, action

    def forget_stale_hands(self, active_hand_ids):
        """Drop smoothing/debounce state for hands no longer detected."""
        for oid in list(self._smoothed_xyz):
            if oid not in active_hand_ids:
                del self._smoothed_xyz[oid]
        for oid in list(self._gesture_counters):
            if oid not in active_hand_ids:
                del self._gesture_counters[oid]
