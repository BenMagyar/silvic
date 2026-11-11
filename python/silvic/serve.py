"""Fast, exact batch scoring of LightGBM models.

``Predictor`` compiles a model once (milliseconds) and scores numpy or pandas data without the GIL; its predictions
equal ``lightgbm.Booster.predict``'s and ``silvic.Booster.predict``'s bit for bit. ``Model`` is a drop-in for
``lleaves.Model``: ``Model(model_file=...)``, ``compile()`` (nothing to compile ahead of time), ``predict(X, n_jobs)``.

Rows are scored in blocks of 32 or 64 with QuickScorer leaf bitvectors on the CPU's vector unit: ``kernel`` picks it
(``available_kernels()`` lists what this CPU runs; the default is NEON on arm64 and AVX2 on x86-64 when present,
else ``portable``).
"""

import json
import os

from . import _silvic
from .basic import _to_matrix


def available_kernels():
    """The kernels this CPU runs (``portable``, ``neon``, ``avx2``, ``avx512``)."""
    return list(_silvic.available_kernels())


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
    raise TypeError(f"expected a Booster, model text or a model file path, got {type(model).__name__}")


class Predictor:
    """A model compiled for batch scoring.

    ``model`` is a ``silvic.Booster``, a ``lightgbm.Booster``, model text, or a path to a model file. ``kernel``
    ``None`` picks the best kernel this CPU runs.
    """

    def __init__(self, model, kernel=None):
        self._handle = _silvic.Predictor(_model_text(model), kernel)
        stored = self._handle.pandas_categorical
        self.pandas_categorical = json.loads(stored) if stored is not None else None

    @property
    def kernel(self):
        return self._handle.kernel

    def predict(self, data, raw_score=False, num_iteration=None, num_threads=1):
        """One prediction per row of ``data`` (numpy, f32 or f64 in C or Fortran order, or pandas, with categories
        mapped as at training). ``raw_score`` skips the objective's transform; ``num_iteration`` ``None`` or ``<= 0``
        uses every tree; ``num_threads`` 1 scores on the calling thread, 0 on every core."""
        matrix, _, _, _ = _to_matrix(data, pandas_categorical=self.pandas_categorical)
        return self._handle.predict(matrix, raw_score=raw_score, num_iteration=num_iteration, num_threads=num_threads)

    def num_trees(self):
        return self._handle.num_trees

    def num_feature(self):
        return self._handle.num_features


class Model:
    """``lleaves.Model`` with silvic's predictor: build it from ``model_file`` (or ``model_str``), call ``compile()``
    as with lleaves (it only records ``raw_score``: the model compiles in milliseconds when it is built), then
    ``predict(data, n_jobs)``. Predictions equal LightGBM's bit for bit, where lleaves can differ in the last bits."""

    def __init__(self, model_file=None, model_str=None, kernel=None):
        if (model_file is None) == (model_str is None):
            raise TypeError("pass exactly one of model_file and model_str")
        self._predictor = Predictor(model_str if model_str is not None else os.fspath(model_file), kernel)
        self._raw_score = False

    def compile(self, cache=None, *, raw_score=False, **_options):
        """Records ``raw_score``; lleaves' other compile options (``cache``, ``fblocksize``, ``finline``, ...) have
        nothing to tune here and are accepted and ignored."""
        self._raw_score = raw_score

    def predict(self, data, n_jobs=None):
        """One prediction per row of ``data``; ``n_jobs`` ``None`` uses every core, as lleaves does."""
        return self._predictor.predict(data, raw_score=self._raw_score, num_threads=0 if n_jobs is None else n_jobs)

    def num_feature(self):
        return self._predictor.num_feature()

    def num_trees(self):
        return self._predictor.num_trees()

    def num_model_per_iteration(self):
        return 1
