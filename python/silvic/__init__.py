"""silvic: LightGBM-compatible gradient boosting on CPU, Apple Metal, NVIDIA CUDA and AWS Neuron.

The API mirrors LightGBM's Python package (``Dataset``, ``train``, ``Booster``, ``log_evaluation``,
``record_evaluation``, and the scikit-learn estimators ``SilvicRegressor``, ``SilvicClassifier`` and
``SilvicRanker``), and models are LightGBM 3.3.5 ``model.txt`` text that ``lightgbm.Booster`` 3.3.5/4.x loads and
lleaves compiles. ``silvic.ray.train`` trains one model across Ray actors. ``device_type`` in the parameters picks
the device: ``auto`` (CUDA, then Metal, then the CPU), ``gpu``, ``cpu``, ``metal``, ``cuda`` or explicit ``neuron``
on Linux. Neuron is never selected automatically; ``gpu_device_id`` selects its logical NeuronCore index.

Quantized gradient histograms use integer sums independent of reduction order. Objective arithmetic precision can
differ between backends.
"""

from ._silvic import (
    CollectiveError,
    ConfigError,
    DataError,
    DeviceError,
    ModelError,
    SilvicError,
    UnsupportedError,
    __version__,
    available_devices,
)
from .basic import Booster, Dataset
from .callback import CallbackEnv, log_evaluation, record_evaluation
from .engine import train
from .sklearn import SilvicClassifier, SilvicModel, SilvicRanker, SilvicRegressor

__all__ = [
    "Booster",
    "CallbackEnv",
    "CollectiveError",
    "ConfigError",
    "DataError",
    "DeviceError",
    "ModelError",
    "SilvicClassifier",
    "SilvicError",
    "SilvicModel",
    "SilvicRanker",
    "SilvicRegressor",
    "UnsupportedError",
    "Dataset",
    "__version__",
    "available_devices",
    "log_evaluation",
    "record_evaluation",
    "train",
]
