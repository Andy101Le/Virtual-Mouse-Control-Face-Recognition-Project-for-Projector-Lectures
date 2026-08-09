"""
gesture_mlp.py
──────────────
A plain-numpy forward pass for the landmark gesture MLP, reading the
weights straight out of the Keras .h5 file.

The model is 59,397 parameters on 63 inputs. TensorFlow spends about
9.3 ms per call on it, essentially all of that graph-execution overhead —
the arithmetic itself is four small matmuls. Doing them in numpy measures
0.08 ms, a ~110x saving repeated for every hand in every frame.

The ~480 MB of resident memory TensorFlow costs is a *separate* win and
this module does not deliver it on its own: `mediapipe` imports TF too, in
`tasks/python/core/optional_dependencies.py`, purely to borrow a docgen
decorator. That import is wrapped in a try/except and falls back to a no-op,
so the memory only comes back once TensorFlow is actually uninstalled from
the environment — measured 525 MB -> 47 MB when it is. Nothing outside
`tests/test_gesture_mlp.py` imports TF any more, and that test skips its
comparison when TF is missing.

The .h5 stays the single source of truth: retraining and dropping in a new
file needs no export step. Widths and depths are read from the file, so a
retrained model of a different shape still loads. Anything this module
cannot reproduce exactly — a layer type it doesn't implement, an activation
it doesn't know — raises at load time rather than quietly computing
something else.

**BatchNorm is not folded into the preceding Dense.** In this architecture
each Dense carries its own ReLU and the BatchNormalization sits *after* it,
so the two are separated by a nonlinearity and folding them is simply wrong
(an early attempt that did it was fast and disagreed with Keras on 59 of
200 inputs). Each BN is instead collapsed to the single scale/offset pair
it is at inference time, which is exact.
"""

import json

import h5py
import numpy as np

# Keras' default BatchNormalization epsilon, used when a layer config
# somehow omits it.
_DEFAULT_BN_EPSILON = 1e-3


class UnsupportedModel(Exception):
    """The .h5 holds a graph this module cannot reproduce exactly."""


def _relu(x):
    return np.maximum(x, 0.0, out=x)


def _softmax(x):
    # Subtract the row max before exponentiating, as Keras does, so large
    # logits can't overflow.
    x -= np.max(x, axis=-1, keepdims=True)
    np.exp(x, out=x)
    x /= np.sum(x, axis=-1, keepdims=True)
    return x


def _linear(x):
    return x


_ACTIVATIONS = {'relu': _relu, 'softmax': _softmax, 'linear': _linear}

# Layers that do nothing at inference time and can be skipped outright.
# Dropout is identity outside training; InputLayer holds no computation.
_INFERENCE_NOOPS = {'Dropout', 'InputLayer'}


class _DenseOp:
    """out = activation(x @ kernel + bias)"""

    __slots__ = ('kernel', 'bias', 'activation', 'name')

    def __init__(self, name, kernel, bias, activation):
        self.name = name
        self.kernel = kernel
        self.bias = bias
        self.activation = activation

    def __call__(self, x):
        return self.activation(x @ self.kernel + self.bias)


class _BatchNormOp:
    """
    out = x * scale + offset

    At inference BatchNormalization is the fixed affine map
    gamma * (x - moving_mean) / sqrt(moving_var + eps) + beta, so both
    halves collapse into one multiply-add computed once at load time.
    """

    __slots__ = ('scale', 'offset', 'name')

    def __init__(self, name, gamma, beta, mean, var, epsilon):
        self.name = name
        self.scale = (gamma / np.sqrt(var + epsilon)).astype(np.float32)
        self.offset = (beta - mean * self.scale).astype(np.float32)

    def __call__(self, x):
        x *= self.scale
        x += self.offset
        return x


def _read_weights(h5file):
    """
    Map layer name -> {variable name: array}.

    Keras writes these under `model_weights/<layer>/<...>/<variable>`, with
    the middle path varying by save format, so key off the two ends only.
    """
    weights = {}

    def visit(path, obj):
        if not isinstance(obj, h5py.Dataset):
            return
        parts = path.split('/')
        if len(parts) < 3 or parts[0] != 'model_weights':
            return
        weights.setdefault(parts[1], {})[parts[-1]] = np.asarray(
            obj, dtype=np.float32)

    h5file.visititems(visit)
    return weights


def _layer_configs(h5file):
    """The model's layer list, in order, out of the saved JSON config."""
    raw = h5file.attrs.get('model_config')
    if raw is None:
        raise UnsupportedModel("no model_config in the .h5 — not a Keras model file")
    if isinstance(raw, bytes):
        raw = raw.decode('utf-8')
    config = json.loads(raw)
    try:
        return config['config']['layers']
    except (KeyError, TypeError):
        raise UnsupportedModel("model_config has no layer list; expected a Sequential model")


def _require(layer_name, variables, *names):
    missing = [n for n in names if n not in variables]
    if missing:
        raise UnsupportedModel(
            f"layer {layer_name!r} is missing saved weights {missing}")
    return [variables[n] for n in names]


class GestureMLP:
    """
    The gesture classifier as a list of numpy ops.

    Call it with a (N, n_features) float32 array; it returns the (N, n_classes)
    softmax probabilities, exactly as `keras_model(x, training=False)` would.
    """

    def __init__(self, model_path):
        with h5py.File(model_path, 'r') as f:
            layers = _layer_configs(f)
            weights = _read_weights(f)

        self.ops = []
        self.n_features = None
        self.n_classes = None

        for layer in layers:
            kind = layer.get('class_name')
            cfg = layer.get('config', {})
            name = cfg.get('name', kind)

            if kind in _INFERENCE_NOOPS:
                continue

            if kind == 'Dense':
                kernel, bias = _require(name, weights.get(name, {}), 'kernel', 'bias')
                # Newer Keras can serialise this as a dict rather than a
                # name; treat anything that isn't a plain name we know as
                # unsupported instead of guessing.
                activation = cfg.get('activation', 'linear')
                if not isinstance(activation, str) or activation not in _ACTIVATIONS:
                    raise UnsupportedModel(
                        f"layer {name!r} uses activation {activation!r}, which "
                        f"gesture_mlp does not implement")
                if self.n_features is None:
                    self.n_features = int(kernel.shape[0])
                self.n_classes = int(kernel.shape[1])
                self.ops.append(
                    _DenseOp(name, kernel, bias, _ACTIVATIONS[activation]))

            elif kind == 'BatchNormalization':
                axis = cfg.get('axis', -1)
                # Anything but the feature axis would need real broadcasting
                # logic; this model normalises the last axis.
                if axis not in (-1, 1):
                    raise UnsupportedModel(
                        f"layer {name!r} normalises axis {axis}; only the "
                        f"feature axis is supported")
                if not cfg.get('scale', True) or not cfg.get('center', True):
                    raise UnsupportedModel(
                        f"layer {name!r} was saved with scale/center disabled")
                gamma, beta, mean, var = _require(
                    name, weights.get(name, {}),
                    'gamma', 'beta', 'moving_mean', 'moving_variance')
                self.ops.append(_BatchNormOp(
                    name, gamma, beta, mean, var,
                    float(cfg.get('epsilon', _DEFAULT_BN_EPSILON))))

            else:
                raise UnsupportedModel(
                    f"layer {name!r} is a {kind}, which gesture_mlp does not "
                    f"implement — the numpy path would not match Keras")

        if self.n_features is None:
            raise UnsupportedModel("model has no Dense layers")

    def __call__(self, x):
        # Copy once up front: the ops write in place from here on, and the
        # caller's array must not be touched.
        out = np.array(x, dtype=np.float32, copy=True, ndmin=2)
        if out.shape[-1] != self.n_features:
            raise ValueError(
                f"expected {self.n_features} features per row, got {out.shape[-1]}")
        for op in self.ops:
            out = op(out)
        return out

    def __repr__(self):
        return (f"<GestureMLP {self.n_features}→{self.n_classes}, "
                f"{len(self.ops)} ops>")
