"""Batch scoring of LightGBM models, exact by default.

``Predictor`` prepares a model once and scores numpy or pandas data without the GIL. Default float64 predictions
equal ``lightgbm.Booster.predict``'s and ``silvic.Booster.predict``'s bit for bit. ``Model`` is a drop-in for
``lleaves.Model``: ``Model(model_file=...)``, ``compile()`` (records scoring options), ``predict(X, n_jobs)``.

CPU rows are scored in blocks of 32 or 64 with QuickScorer leaf bitvectors. ``kernel`` selects a CPU vector unit;
``available_kernels()`` lists the supported choices. ``device="cuda"`` explicitly selects native CUDA serving on
Linux; ``device="neuron"`` selects an AWS NeuronCore. Both retain the forest on the device and are never selected
automatically. ``precision="float64"`` is the default; CUDA also accepts ``precision="float32"``, while Neuron
requires it explicitly. Float32 mode uses approximate model arithmetic and output, converting float64 inputs to
float32. ``available_devices()`` reports runtime device availability.
"""

import json
import os

from . import _silvic
from .basic import _to_matrix


def available_kernels():
    """The kernels this CPU runs (``portable``, ``neon``, ``avx2``, ``avx512``)."""
    return list(_silvic.available_kernels())


def available_devices():
    """Serving devices available at runtime: ``cpu`` and optionally ``cuda`` or ``neuron``."""
    return list(_silvic.available_serving_devices())


def _model_text(model):
    """Model text from a ``silvic.Booster`` or ``lightgbm.Booster`` (as its ``model_to_string()`` writes it, which
    keeps only the best iteration when there is one), model text, or a path to a model file."""
    if hasattr(model, "model_to_string"):
        return model.model_to_string()
    if isinstance(model, str) and "\n" in model:
        return model
    if isinstance(model, (str, os.PathLike)):
        with open(model) as file:
            return file.read()
    raise TypeError(
        f"expected a Booster, model text or a model file path, got {type(model).__name__}"
    )


class Predictor:
    """A model compiled for batch scoring.

    ``model`` is a ``silvic.Booster``, a ``lightgbm.Booster``, model text, or a path to a model file. On the CPU,
    ``kernel=None`` picks the best available vector unit. ``device="cuda"`` or ``device="neuron"`` selects native
    device serving on Linux without CPU fallback; ``device_id`` is the GPU ordinal or logical NeuronCore index.
    Neither accepts a CPU ``kernel``. ``precision="float32"`` returns approximate float32 predictions with
    float32 thresholds, leaves and accumulation, not a cast of float64 predictions. Neuron requires explicit
    float32 precision. The float64 default remains bitwise exact on CPU and CUDA.
    """

    def __init__(
        self, model, kernel=None, *, device="cpu", device_id=0, precision="float64"
    ):
        self._handle = _silvic.Predictor(
            _model_text(model),
            kernel,
            device=device,
            device_id=device_id,
            precision=precision,
        )
        stored = self._handle.pandas_categorical
        self.pandas_categorical = json.loads(stored) if stored is not None else None

    @property
    def kernel(self):
        return self._handle.kernel

    @property
    def device(self):
        return self._handle.device

    @property
    def precision(self):
        return self._handle.precision

    def predict(self, data, raw_score=False, num_iteration=None, num_threads=1):
        """One prediction per row of ``data`` (numpy, f32 or f64 in C or Fortran order, or pandas, with categories
        mapped as at training). ``raw_score`` skips the objective's transform; ``num_iteration`` ``None`` or ``<= 0``
        uses every tree. On the CPU, ``num_threads`` 1 scores on the calling thread and 0 on every core.
        CUDA and Neuron require the default ``num_threads=1``; they do not use CPU thread-pool controls."""
        matrix, _, _, _ = _to_matrix(data, pandas_categorical=self.pandas_categorical)
        return self._handle.predict(
            matrix,
            raw_score=raw_score,
            num_iteration=num_iteration,
            num_threads=num_threads,
        )

    def num_trees(self):
        return self._handle.num_trees

    def num_feature(self):
        return self._handle.num_features


class Model:
    """``lleaves.Model`` with silvic's predictor: build it from ``model_file`` (or ``model_str``), call ``compile()``
    to record ``raw_score``, then ``predict(data, n_jobs)``. Model preparation happens at construction.
    Float64 predictions equal LightGBM's bit for bit; explicit CUDA or Neuron float32 is approximate."""

    def __init__(
        self,
        model_file=None,
        model_str=None,
        kernel=None,
        *,
        device="cpu",
        device_id=0,
        precision="float64",
    ):
        if (model_file is None) == (model_str is None):
            raise TypeError("pass exactly one of model_file and model_str")
        self._predictor = Predictor(
            model_str if model_str is not None else os.fspath(model_file),
            kernel,
            device=device,
            device_id=device_id,
            precision=precision,
        )
        self._raw_score = False

    @property
    def precision(self):
        return self._predictor.precision

    def compile(self, cache=None, *, raw_score=False, **_options):
        """Records ``raw_score``; lleaves' other compile options (``cache``, ``fblocksize``, ``finline``, ...) have
        nothing to tune here and are accepted and ignored."""
        self._raw_score = raw_score

    def predict(self, data, n_jobs=None):
        """One prediction per row. ``n_jobs=None`` uses every CPU core, or default device execution.
        CUDA and Neuron only accept ``n_jobs=None`` or ``1``."""
        threads = (
            (0 if self._predictor.device == "cpu" else 1) if n_jobs is None else n_jobs
        )
        return self._predictor.predict(
            data, raw_score=self._raw_score, num_threads=threads
        )

    def num_feature(self):
        return self._predictor.num_feature()

    def num_trees(self):
        return self._predictor.num_trees()

    def num_model_per_iteration(self):
        return 1
