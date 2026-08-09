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

LABEL_MAP = {
    0: 'MOVE', 1: 'LEFT CLICK', 2: 'RIGHT CLICK', 3: 'ZOOM IN', 4: 'ZOOM OUT',
}
CURSOR_FREEZE_ACTIONS = {'LEFT CLICK', 'RIGHT CLICK', 'ZOOM IN', 'ZOOM OUT'}

FINGER_CHAINS = [
    [0, 1, 2, 3, 4], [0, 5, 6, 7, 8], [0, 9, 10, 11, 12],
    [0, 13, 14, 15, 16], [0, 17, 18, 19, 20],
]

SMOOTHING_ALPHA = np.float32(0.3)
ONE_MINUS_ALPHA = np.float32(0.7)

# Below this the winning class is treated as no gesture at all, so an
# ambiguous pose can't fire a click on its way to somewhere else.
CONFIDENCE_THRESHOLD = 0.75


class GestureEngine:
    DEBOUNCE_FRAMES = 5

    def __init__(self, model_path):
        self.gesture_model = GestureMLP(model_path)
        print(f"Gesture model loaded — {self.gesture_model.n_classes} classes")

        self._smoothed_xyz     = {}
        self._gesture_counters = {}

    def smooth_hand(self, hand_id, raw_pts):
        """
        EMA-smooth this hand's raw (21, 3) landmark array, seeding state
        the first time a given hand_id is seen so it doesn't slowly
        smooth in from zero.
        """
        if hand_id not in self._smoothed_xyz:
            self._smoothed_xyz[hand_id] = raw_pts
        sxyz = SMOOTHING_ALPHA * raw_pts + ONE_MINUS_ALPHA * self._smoothed_xyz[hand_id]
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
        if gc['count'] >= self.DEBOUNCE_FRAMES:
            gc['confirmed'] = action

        return gc['confirmed'], confidence

    def forget_stale_hands(self, active_hand_ids):
        """Drop smoothing/debounce state for hands no longer detected."""
        for oid in list(self._smoothed_xyz):
            if oid not in active_hand_ids:
                del self._smoothed_xyz[oid]
        for oid in list(self._gesture_counters):
            if oid not in active_hand_ids:
                del self._gesture_counters[oid]
