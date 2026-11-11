"""``Dataset`` and ``Booster``, mirroring LightGBM 3.3.5 / 4.x ``lightgbm.basic`` for training and prediction.

A ``Dataset`` keeps its inputs until training needs it, then bins them once (``construct``), as LightGBM's lazy
``Dataset`` does: training parameters reach binning, and a validation set bins with its reference's bin mappers.
numpy input of dtype float32 or float64 is read in place; any other dtype is converted to float32, as LightGBM does.
pandas input follows LightGBM 4.x ``_data_from_pandas``: columns must be numeric, bool or ``category``; categories
become their codes (missing as NaN), unordered ``category`` columns are categorical features under ``'auto'``, and the
category levels are stored on the model's ``pandas_categorical:`` line so prediction maps codes the same way.

A ``Booster`` holds a trained model in LightGBM 3.3.5's text format and predicts from it; ``to_lightgbm()`` hands the
same text to ``lightgbm.Booster``.
"""

import collections
import json
from pathlib import Path

import numpy as np

from . import _params
from . import _silvic

_FLOAT_DTYPES = (np.dtype(np.float32), np.dtype(np.float64))


def _pandas():
    try:
        import pandas
    except ImportError:
        return None
    return pandas


def _is_dataframe(data):
    pandas = _pandas()
    return pandas is not None and isinstance(data, pandas.DataFrame)


def _json_default_with_numpy(value):
    """LightGBM's ``json_default_with_numpy``: numpy scalars and arrays as plain JSON."""
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _dump_pandas_categorical(pandas_categorical):
    """The JSON LightGBM's Python package writes after ``pandas_categorical:``."""
    return json.dumps(pandas_categorical, default=_json_default_with_numpy)


def _check_pandas_dtypes(data):
    """LightGBM ``_check_for_bad_pandas_dtypes``: only numbers and bools reach the core."""
    pandas = _pandas()
    bad = [
        f"{column}: {dtype}"
        for column, dtype in data.dtypes.items()
        if not (
            pandas.api.types.is_integer_dtype(dtype)
            or pandas.api.types.is_float_dtype(dtype)
            or pandas.api.types.is_bool_dtype(dtype)
        )
    ]
    if bad:
        raise ValueError("pandas dtypes must be int, float or bool.\nFields with bad pandas dtypes: " + ", ".join(bad))


def _data_from_pandas(data, feature_name, categorical_feature, pandas_categorical):
    """LightGBM 4.x ``_data_from_pandas``: ``(matrix, feature_name, categorical_feature, pandas_categorical)``."""
    pandas = _pandas()
    if len(data.shape) != 2 or data.shape[0] < 1:
        raise ValueError("Input data must be 2 dimensional and non empty.")
    data = data.copy(deep=False)
    if feature_name == "auto":
        feature_name = [str(column) for column in data.columns]
    cat_cols = [
        column for column, dtype in zip(data.columns, data.dtypes) if isinstance(dtype, pandas.CategoricalDtype)
    ]
    cat_cols_not_ordered = [column for column in cat_cols if not data[column].cat.ordered]
    if pandas_categorical is None:
        pandas_categorical = [list(data[column].cat.categories) for column in cat_cols]
    else:
        if len(cat_cols) != len(pandas_categorical):
            raise ValueError("train and valid dataset categorical_feature do not match.")
        for column, category in zip(cat_cols, pandas_categorical):
            if list(data[column].cat.categories) != list(category):
                data[column] = data[column].cat.set_categories(category)
    if cat_cols:
        data[cat_cols] = data[cat_cols].apply(lambda x: x.cat.codes).replace({-1: np.nan})
    if categorical_feature == "auto":
        categorical_feature = cat_cols_not_ordered
    _check_pandas_dtypes(data)
    # LightGBM appends float32 so that an all-integer frame still converts to a float type.
    target_dtype = np.result_type(*[dtype.type for dtype in data.dtypes], np.float32)
    try:
        matrix = data.to_numpy(dtype=target_dtype, copy=False)
    except ValueError:
        # Nullable dtypes need an explicit NA value.
        matrix = data.to_numpy(dtype=target_dtype, na_value=np.nan)
    return matrix, feature_name, categorical_feature, pandas_categorical


def _matrix_from_numpy(data):
    """A 2-D float32/float64 C- or F-contiguous array; other dtypes become float32 (LightGBM ``__init_from_np2d``)."""
    matrix = np.asarray(data)
    if matrix.ndim != 2:
        raise ValueError(f"data must be 2 dimensional, got {matrix.ndim} dimensions")
    if matrix.dtype not in _FLOAT_DTYPES:
        matrix = matrix.astype(np.float32)
    if not (matrix.flags.c_contiguous or matrix.flags.f_contiguous) or not matrix.flags.aligned:
        matrix = np.ascontiguousarray(matrix)
    return matrix


def _to_matrix(data, feature_name="auto", categorical_feature="auto", pandas_categorical=None):
    """``(matrix, feature_name, categorical_feature, pandas_categorical)`` for numpy or pandas ``data``."""
    if _is_dataframe(data):
        matrix, feature_name, categorical_feature, pandas_categorical = _data_from_pandas(
            data, feature_name, categorical_feature, pandas_categorical
        )
        return _matrix_from_numpy(matrix), feature_name, categorical_feature, pandas_categorical
    if hasattr(data, "tocsr") or hasattr(data, "tocsc"):
        raise TypeError("silvic trains on dense data; convert the sparse matrix with .toarray()")
    if feature_name == "auto":
        feature_name = None
    if categorical_feature == "auto":
        categorical_feature = []
    return _matrix_from_numpy(data), feature_name, categorical_feature, pandas_categorical


def _to_1d(values, dtype, name):
    """A contiguous 1-D array of ``dtype`` from a list, numpy array or pandas Series / one-column DataFrame."""
    if values is None:
        return None
    if _is_dataframe(values):
        if values.shape[1] != 1:
            raise ValueError(f"{name} DataFrame must have exactly one column, got {values.shape[1]}")
        values = values.iloc[:, 0]
    array = np.asarray(values)
    if array.ndim != 1:
        if array.ndim == 2 and 1 in array.shape:
            array = array.reshape(-1)
        else:
            raise ValueError(f"{name} must be 1 dimensional, got shape {array.shape}")
    return np.ascontiguousarray(array, dtype=dtype)


def _feature_names(names, num_features):
    if names is None:
        return None
    names = [str(name) for name in names]
    if len(names) != num_features:
        raise ValueError(f"feature_name has {len(names)} names for {num_features} features")
    # LightGBM's `Dataset::set_feature_names` replaces spaces, which would split the model's `feature_names=` line.
    return [name.replace(" ", "_") for name in names]


def _categorical_indices(categorical_feature, feature_names, num_features):
    """Column indices of ``categorical_feature`` (indices or names, as LightGBM accepts both)."""
    indices = []
    for feature in categorical_feature or []:
        if isinstance(feature, (int, np.integer)):
            index = int(feature)
        elif feature_names is not None and str(feature).replace(" ", "_") in feature_names:
            index = feature_names.index(str(feature).replace(" ", "_"))
        else:
            raise ValueError(f"categorical feature {feature!r} is not a feature name")
        if not 0 <= index < num_features:
            raise ValueError(f"categorical feature index {index} is out of range for {num_features} features")
        indices.append(index)
    return sorted(set(indices))


class Dataset:
    """Training or validation data, binned on first use (LightGBM ``Dataset``).

    Arguments keep LightGBM 4.x's names and positions. ``init_score`` is not supported. ``params`` holds binning
    parameters (``max_bin``, ``bin_construct_sample_cnt``, ...); ``silvic.train`` adds its own before binning.
    """

    def __init__(
        self,
        data,
        label=None,
        reference=None,
        weight=None,
        group=None,
        init_score=None,
        feature_name="auto",
        categorical_feature="auto",
        params=None,
        free_raw_data=True,
    ):
        if init_score is not None:
            raise ValueError("silvic does not support init_score")
        if reference is not None and not isinstance(reference, Dataset):
            raise TypeError("reference must be a silvic.Dataset")
        self.data = data
        self.label = label
        self.reference = reference
        self.weight = weight
        self.group = group
        self.feature_name = feature_name
        self.categorical_feature = categorical_feature
        self.params = dict(params or {})
        self.free_raw_data = free_raw_data
        self.pandas_categorical = None
        self._handle = None

    def construct(self):
        """Bins the data (once); a validation set first constructs its reference."""
        if self._handle is not None:
            return self
        if self.data is None:
            raise ValueError("cannot construct a Dataset whose raw data was freed")
        if self.label is None:
            raise ValueError("label must be set before the Dataset is constructed")
        if self.reference is not None:
            self.reference.construct()
            self._construct_validation()
        else:
            self._construct_training()
        if self.free_raw_data:
            self.data = None
        return self

    def _metadata(self, num_rows):
        label = _to_1d(self.label, np.float32, "label")
        weight = _to_1d(self.weight, np.float32, "weight")
        group = _to_1d(self.group, np.uint32, "group")
        if label.shape[0] != num_rows:
            raise ValueError(f"label has {label.shape[0]} values for {num_rows} rows")
        return label, weight, group

    def _construct_training(self):
        matrix, names, categorical, self.pandas_categorical = _to_matrix(
            self.data, self.feature_name, self.categorical_feature
        )
        num_features = matrix.shape[1]
        names = _feature_names(names, num_features)
        label, weight, group = self._metadata(matrix.shape[0])
        self._handle = _silvic.Dataset(
            matrix,
            label,
            weight=weight,
            group=group,
            feature_names=names,
            categorical_features=_categorical_indices(categorical, names, num_features),
            params_json=_params.to_json(self.params),
        )

    def _construct_validation(self):
        reference = self.reference
        self.pandas_categorical = reference.pandas_categorical
        matrix, _, _, _ = _to_matrix(self.data, pandas_categorical=reference.pandas_categorical)
        if matrix.shape[1] != reference.num_feature():
            raise ValueError(
                f"validation data has {matrix.shape[1]} features, its reference has {reference.num_feature()}"
            )
        label, weight, group = self._metadata(matrix.shape[0])
        self._handle = _silvic.Dataset.construct_with_reference(
            matrix, label, reference._handle, weight=weight, group=group
        )

    def _update_params(self, params):
        """Adds training parameters ahead of binning; they cannot change a constructed Dataset."""
        if self._handle is None and params:
            self.params = _params.merge(self.params, params)
        return self

    def _check_unconstructed(self, what):
        if self._handle is not None:
            raise ValueError(f"cannot set {what} after the Dataset is constructed")

    def create_valid(self, data, label=None, weight=None, group=None, init_score=None, params=None):
        """A validation Dataset binned with this one's bin mappers."""
        return Dataset(
            data, label=label, reference=self, weight=weight, group=group, init_score=init_score, params=params
        )

    def set_reference(self, reference):
        self._check_unconstructed("reference")
        self.reference = reference
        return self

    def set_label(self, label):
        self._check_unconstructed("label")
        self.label = label
        return self

    def set_weight(self, weight):
        self._check_unconstructed("weight")
        self.weight = weight
        return self

    def set_group(self, group):
        self._check_unconstructed("group")
        self.group = group
        return self

    def set_feature_name(self, feature_name):
        if feature_name != "auto":
            self._check_unconstructed("feature_name")
            self.feature_name = feature_name
        return self

    def set_categorical_feature(self, categorical_feature):
        if categorical_feature != "auto":
            self._check_unconstructed("categorical_feature")
            self.categorical_feature = categorical_feature
        return self

    def get_label(self):
        return _to_1d(self.label, np.float32, "label")

    def get_weight(self):
        return _to_1d(self.weight, np.float32, "weight")

    def get_group(self):
        return _to_1d(self.group, np.uint32, "group")

    def get_feature_name(self):
        if self._handle is None:
            raise ValueError("cannot get feature names before the Dataset is constructed")
        return self._handle.feature_names

    def num_data(self):
        if self._handle is None:
            raise ValueError("cannot get num_data before the Dataset is constructed")
        return self._handle.num_rows

    def num_feature(self):
        if self._handle is None:
            raise ValueError("cannot get num_feature before the Dataset is constructed")
        return self._handle.num_features


class Booster:
    """A trained model (LightGBM ``Booster``), from ``silvic.train``, a model file or a model string.

    silvic boosters do not train further, so ``train_set`` is not accepted; ``params`` is kept as ``.params``.
    Pickling stores the model text.
    """

    def __init__(self, params=None, train_set=None, model_file=None, model_str=None):
        if train_set is not None:
            raise ValueError("silvic.Booster does not train; use silvic.train")
        if (model_file is None) == (model_str is None):
            raise TypeError("pass exactly one of model_file and model_str")
        if model_file is not None:
            model_str = Path(model_file).read_text()
        self.params = dict(params or {})
        self.best_iteration = 0
        self.best_score = collections.defaultdict(collections.OrderedDict)
        self._load(model_str)

    def _load(self, model_str):
        self._handle = _silvic.Booster(model_str)
        stored = self._handle.pandas_categorical
        self.pandas_categorical = json.loads(stored) if stored is not None else None

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_handle"] = self.model_to_string()
        state.pop("pandas_categorical")
        state["best_score"] = {name: dict(scores) for name, scores in self.best_score.items()}
        return state

    def __setstate__(self, state):
        model_str = state.pop("_handle")
        best_score = state.pop("best_score")
        self.__dict__.update(state)
        self.best_score = collections.defaultdict(collections.OrderedDict)
        for name, scores in best_score.items():
            self.best_score[name].update(scores)
        self._load(model_str)

    def __copy__(self):
        return self.__deepcopy__(None)

    def __deepcopy__(self, memo):
        clone = Booster.__new__(Booster)
        clone.__setstate__(self.__getstate__())
        return clone

    def predict(self, data, raw_score=False, num_iteration=None):
        """One prediction per row of ``data`` (numpy or pandas). ``raw_score`` skips the objective's transform
        (sigmoid for ``binary``); ``num_iteration`` ``None`` or ``<= 0`` uses every tree."""
        matrix, _, _, _ = _to_matrix(data, pandas_categorical=self.pandas_categorical)
        return self._handle.predict(matrix, raw_score=raw_score, num_iteration=num_iteration)

    def model_to_string(self):
        """The model text, byte for byte what LightGBM 3.3.5's ``Booster.model_to_string()`` writes."""
        return self._handle.model_to_string()

    def save_model(self, filename):
        Path(filename).write_text(self.model_to_string())
        return self

    def num_trees(self):
        return self._handle.num_trees

    def current_iteration(self):
        return self._handle.num_trees

    def num_feature(self):
        return self._handle.num_features

    def feature_name(self):
        return self._handle.feature_names

    def feature_importance(self, importance_type="split"):
        """Split counts (int32) or total split gains (float64) per feature, as LightGBM returns them."""
        importance = self._handle.feature_importance(importance_type)
        if importance_type == "split":
            return importance.astype(np.int32)
        return importance

    def to_lightgbm(self):
        """The same model as a ``lightgbm.Booster``."""
        import lightgbm

        return lightgbm.Booster(model_str=self.model_to_string())
