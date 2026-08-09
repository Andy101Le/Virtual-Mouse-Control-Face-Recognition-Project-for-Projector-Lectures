"""
Numpy gesture MLP vs. Keras.

The numpy forward pass in `gesture_mlp.py` exists only to be faster; if it
is not also numerically identical to the Keras model it replaces, it is a
bug factory. This compares the two over random inputs, over inputs shaped
like real normalised hand landmarks, and over the degenerate cases the live
pipeline can actually produce.

Keras/TensorFlow is imported here and nowhere else in the project. Skips
rather than fails if TF is not installed, so the suite still runs once TF
has been uninstalled — but then it is only checking that the numpy path
loads and behaves sanely, not that it matches. Run it once *with* TF
present after any retraining.
"""

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gesture_mlp import GestureMLP, UnsupportedModel  # noqa: E402

MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'landmark_gesture_model.h5')

# Same tolerance the pipeline cares about: the action is picked by argmax
# and gated on a 0.75 confidence threshold, so anything at 1e-5 is noise.
PROB_TOLERANCE = 1e-5

passed = failed = 0


def check(name, condition, detail=''):
    global passed, failed
    if condition:
        passed += 1
        print(f"  PASS  {name}")
    else:
        failed += 1
        print(f"  FAIL  {name}" + (f" — {detail}" if detail else ''))


def synthetic_hands(n, rng):
    """
    Inputs shaped like what GestureEngine.classify actually submits:
    wrist-relative landmarks divided by their max absolute value, so every
    row has at least one coordinate at ±1 and the first point is the origin.
    """
    pts = rng.normal(0.0, 0.35, size=(n, 21, 3)).astype(np.float32)
    pts -= pts[:, :1, :]
    scale = np.abs(pts).max(axis=(1, 2), keepdims=True)
    scale[scale == 0] = 1.0
    pts /= scale
    return pts.reshape(n, 63)


def main():
    print(f"\nGesture MLP — numpy vs Keras\n{'=' * 60}")

    if not os.path.exists(MODEL_PATH):
        print(f"model not found: {MODEL_PATH}")
        return 1

    net = GestureMLP(MODEL_PATH)
    print(f"loaded {net!r}\n")

    rng = np.random.default_rng(20260809)

    print("shape and sanity")
    check("input width is 63", net.n_features == 63, f"got {net.n_features}")
    check("output width is 5", net.n_classes == 5, f"got {net.n_classes}")

    probe = net(synthetic_hands(64, rng))
    check("output shape is (64, 5)", probe.shape == (64, 5), f"got {probe.shape}")
    check("rows sum to 1", np.allclose(probe.sum(axis=1), 1.0, atol=1e-6),
          f"max deviation {np.abs(probe.sum(axis=1) - 1.0).max():.2e}")
    check("probabilities in [0, 1]", bool((probe >= 0).all() and (probe <= 1).all()))
    check("output is float32", probe.dtype == np.float32, str(probe.dtype))

    single = net(np.zeros(63, dtype=np.float32))
    check("a bare (63,) row is treated as one sample", single.shape == (1, 5),
          f"got {single.shape}")

    caller_array = synthetic_hands(4, rng)
    before = caller_array.copy()
    net(caller_array)
    check("caller's input array is not modified",
          np.array_equal(caller_array, before))

    try:
        net(np.zeros((2, 40), dtype=np.float32))
        check("wrong feature count raises", False, "no exception")
    except ValueError:
        check("wrong feature count raises", True)

    # Degenerate inputs the live path can produce: an all-zero row happens
    # whenever every landmark lands on the wrist (scale falls back to 1.0).
    for label, x in (("all zeros", np.zeros((1, 63), np.float32)),
                     ("all ones", np.ones((1, 63), np.float32)),
                     ("large magnitude", np.full((1, 63), 50.0, np.float32))):
        out = net(x)
        check(f"{label} gives finite probabilities summing to 1",
              bool(np.isfinite(out).all()) and abs(float(out.sum()) - 1.0) < 1e-6,
              str(out))

    print("\nagreement with Keras")
    try:
        os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
        from keras.models import load_model
    except ImportError:
        print("  SKIP  TensorFlow/Keras not installed — cannot compare.")
        print("        Run this suite with TF present after retraining.")
    else:
        keras_model = load_model(MODEL_PATH)
        keras_model(np.zeros((1, 63), dtype=np.float32), training=False)

        cases = {
            'synthetic hands': synthetic_hands(300, rng),
            'uniform noise': rng.uniform(-1, 1, (200, 63)).astype(np.float32),
            'normal noise': rng.normal(0, 3, (200, 63)).astype(np.float32),
            # The real batch is always two rows: the pose and its x-mirrored
            # twin, so check that exact shape too.
            'mirrored pairs': np.concatenate([
                (lambda p: np.stack([p.reshape(63),
                                     (p * np.float32([-1, 1, 1])).reshape(63)]))(
                    synthetic_hands(1, rng).reshape(21, 3))
                for _ in range(50)]).astype(np.float32),
        }

        worst_prob = 0.0
        total_rows = 0
        for label, x in cases.items():
            mine = net(x)
            theirs = keras_model(x, training=False).numpy()
            diff = float(np.abs(mine - theirs).max())
            worst_prob = max(worst_prob, diff)
            disagreements = int((mine.argmax(1) != theirs.argmax(1)).sum())
            total_rows += len(x)
            check(f"{label}: max probability diff < {PROB_TOLERANCE:g}",
                  diff < PROB_TOLERANCE, f"{diff:.3e}")
            check(f"{label}: no argmax disagreements over {len(x)} rows",
                  disagreements == 0, f"{disagreements} rows differ")

            # The pipeline thresholds confidence at 0.75; a row that lands on
            # opposite sides of that line in the two implementations would
            # change behaviour even with a matching argmax.
            gate_flips = int((mine.max(1) >= 0.75).sum() != (theirs.max(1) >= 0.75).sum())
            check(f"{label}: no 0.75-confidence-gate flips", gate_flips == 0)

        print(f"\n  worst probability difference over {total_rows} rows: "
              f"{worst_prob:.3e}")

        print("\nspeed")
        batch = synthetic_hands(2, rng)
        for _ in range(20):
            net(batch)
            keras_model(batch, training=False)

        t0 = time.perf_counter()
        for _ in range(500):
            net(batch)
        numpy_ms = (time.perf_counter() - t0) / 500 * 1000

        t0 = time.perf_counter()
        for _ in range(200):
            keras_model(batch, training=False).numpy()
        keras_ms = (time.perf_counter() - t0) / 200 * 1000

        print(f"  numpy {numpy_ms:.3f} ms   keras {keras_ms:.3f} ms   "
              f"({keras_ms / numpy_ms:.0f}x)")
        check("numpy path is at least 10x faster", numpy_ms * 10 < keras_ms,
              f"numpy {numpy_ms:.3f} ms vs keras {keras_ms:.3f} ms")

    print("\nunsupported architectures are rejected, not mis-run")
    try:
        GestureMLP(os.path.join(os.path.dirname(MODEL_PATH), 'no_such_model.h5'))
        check("a missing file raises", False, "no exception")
    except (OSError, UnsupportedModel):
        check("a missing file raises", True)

    print(f"\n{'=' * 60}\n{passed} passed, {failed} failed\n")
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
