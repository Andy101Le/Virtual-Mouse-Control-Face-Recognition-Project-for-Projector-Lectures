"""
gesture_engine.py
──────────────────
Turns raw per-hand MediaPipe landmarks into a debounced gesture label.

Each hand gets its own background InferenceWorker so the (relatively
slow) Keras model never blocks the main camera loop — the worker always
holds the most recent classification result, and main.py just asks for
whatever's currently available.

Also owns the EMA smoothing applied to raw hand landmarks before they're
drawn or fed to the model, and the debounce counter that requires a
gesture to repeat for DEBOUNCE_FRAMES consecutive frames before it's
"confirmed" (this is what keeps a single misclassified frame from
triggering a click).
"""

import queue
import threading
import numpy as np
from keras.models import load_model

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


class _InferenceWorker:
    """Runs the Keras gesture model on a background thread for one hand."""

    def __init__(self, model):
        self._model  = model
        self._q      = queue.Queue(maxsize=1)
        self._result = ('MOVE', 1.0)
        self._lock   = threading.Lock()
        threading.Thread(target=self._run, daemon=True).start()

    def submit(self, features):
        try:
            self._q.put_nowait(features)
        except queue.Full:
            pass

    def result(self):
        with self._lock:
            return self._result

    def _run(self):
        while True:
            data = self._q.get()
            raw  = self._model(data, training=False).numpy()[0]
            idx  = int(np.argmax(raw))
            conf = float(raw[idx])
            act  = LABEL_MAP.get(idx, 'NO ACTION') if conf >= 0.75 else 'NO ACTION'
            with self._lock:
                self._result = (act, conf)


class GestureEngine:
    DEBOUNCE_FRAMES = 5

    def __init__(self, model_path, num_hands=2):
        self.gesture_model = load_model(model_path)
        # Warm up so the first real frame isn't slowed by lazy graph tracing
        self.gesture_model(np.zeros((1, 63), dtype=np.float32), training=False)
        print(f"Gesture model loaded — {self.gesture_model.output_shape[-1]} classes")

        self._workers = {i: _InferenceWorker(self.gesture_model) for i in range(num_hands)}

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
        """Submit this hand's normalised pose to its worker and return the
        debounced (confirmed_action, confidence) pair."""
        pts_n  = smoothed_xyz - smoothed_xyz[0]
        scale  = np.max(np.abs(pts_n)) or 1.0
        pts_n /= scale
        self._workers[hand_id].submit(pts_n.reshape(1, 63).astype(np.float32))
        action, confidence = self._workers[hand_id].result()

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
