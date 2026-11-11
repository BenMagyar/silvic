"""scikit-learn estimators mirroring LightGBM 4.x ``LGBMRegressor``, ``LGBMClassifier`` and ``LGBMRanker``.

Constructor parameters keep LightGBM's names and defaults, and ``**kwargs`` pass through as training parameters.
A constructor parameter that ``**kwargs`` also sets under another alias (``subsample`` and ``bagging_fraction``) yields
to the ``**kwargs`` value, so the core never sees both. ``n_jobs=None`` uses every core.

The estimators subclass scikit-learn's ``BaseEstimator`` and mixins when scikit-learn is importable, and work without
it. ``SilvicClassifier`` is binary only (silvic has no multiclass objective); ``class_weight`` becomes sample weights
as in LightGBM. ``init_score``, ``init_model`` and callable objectives or metrics are not supported.
"""

import os

import numpy as np

from . import _params
from .basic import Dataset
from .callback import record_evaluation
from .engine import train

try:
    from sklearn.base import BaseEstimator as _Base
    from sklearn.base import ClassifierMixin as _ClassifierMixin
    from sklearn.base import RegressorMixin as _RegressorMixin
    from sklearn.exceptions import NotFittedError as _NotFittedBase

    class NotFittedError(_NotFittedBase):
        """Raised when an estimator is used before ``fit`` (scikit-learn's ``NotFittedError``)."""

except ImportError:
    _Base = object

    class _ClassifierMixin:
        _estimator_type = "classifier"

    class _RegressorMixin:
        _estimator_type = "regressor"

    class NotFittedError(ValueError, AttributeError):
        """Raised when an estimator is used before ``fit``."""


_INIT_PARAMS = (
    "boosting_type",
    "num_leaves",
    "max_depth",
    "learning_rate",
    "n_estimators",
    "subsample_for_bin",
    "objective",
    "class_weight",
    "min_split_gain",
    "min_child_weight",
    "min_child_samples",
    "subsample",
    "subsample_freq",
    "colsample_bytree",
    "reg_alpha",
    "reg_lambda",
    "random_state",
    "n_jobs",
    "importance_type",
)

# Constructor arguments that are not training parameters.
_NOT_PARAMS = ("n_estimators", "class_weight", "importance_type", "objective", "n_jobs")

# LightGBM 3.3.5's `silent` constructor argument; it only controls logging there.
_IGNORED_KWARGS = ("silent",)


class SilvicModel(_Base):
    """The shared implementation of the estimators (LightGBM ``LGBMModel``)."""

    _default_objective = None

    def __init__(
        self,
        boosting_type="gbdt",
        num_leaves=31,
        max_depth=-1,
        learning_rate=0.1,
        n_estimators=100,
        subsample_for_bin=200000,
        objective=None,
        class_weight=None,
        min_split_gain=0.0,
        min_child_weight=1e-3,
        min_child_samples=20,
        subsample=1.0,
        subsample_freq=0,
        colsample_bytree=1.0,
        reg_alpha=0.0,
        reg_lambda=0.0,
        random_state=None,
        n_jobs=None,
        importance_type="split",
        **kwargs,
    ):
        self.boosting_type = boosting_type
        self.num_leaves = num_leaves
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.subsample_for_bin = subsample_for_bin
        self.objective = objective
        self.class_weight = class_weight
        self.min_split_gain = min_split_gain
        self.min_child_weight = min_child_weight
        self.min_child_samples = min_child_samples
        self.subsample = subsample
        self.subsample_freq = subsample_freq
        self.colsample_bytree = colsample_bytree
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.importance_type = importance_type
        self._other_params = {}
        self.set_params(**kwargs)
        self._booster = None
        self._evals_result = None
        self._n_features = None

    def get_params(self, deep=True):
        params = {name: getattr(self, name) for name in _INIT_PARAMS}
        params.update(self._other_params)
        return params

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
            if key not in _INIT_PARAMS:
                self._other_params[key] = value
        return self

    def __sklearn_is_fitted__(self):
        return self._booster is not None

    def _n_threads(self):
        if self.n_jobs is None:
            return None
        if self.n_jobs < 0:
            # joblib's convention, as scikit-learn and LightGBM read it: -1 is every core, -2 all but one, ...
            return max((os.cpu_count() or 1) + 1 + self.n_jobs, 1)
        return self.n_jobs

    def _training_params(self, eval_metric):
        """LightGBM ``_process_params(stage='fit')`` without the parts silvic does not support."""
        kwargs = {key: value for key, value in self._other_params.items() if key not in _IGNORED_KWARGS}
        objective = self.objective or self._default_objective
        for key in _params.find(kwargs, "objective"):
            objective = kwargs.pop(key)
        if callable(objective):
            raise ValueError("silvic does not support custom objective functions")
        ctor = {name: getattr(self, name) for name in _INIT_PARAMS if name not in _NOT_PARAMS}
        if isinstance(ctor["random_state"], np.random.RandomState):
            ctor["random_state"] = ctor["random_state"].randint(np.iinfo(np.int32).max)
        elif isinstance(ctor["random_state"], np.random.Generator):
            ctor["random_state"] = int(ctor["random_state"].integers(np.iinfo(np.int32).max))
        ctor["num_threads"] = self._n_threads()
        params = _params.merge(ctor, kwargs)
        params["objective"] = objective
        metrics = [eval_metric] if isinstance(eval_metric, str) else list(eval_metric or [])
        if any(callable(metric) for metric in metrics):
            raise ValueError("silvic does not support custom eval_metric functions")
        if metrics:
            configured = []
            for key in _params.find(params, "metric"):
                value = params.pop(key)
                configured = [value] if isinstance(value, str) else list(value)
            params["metric"] = [m for m in metrics if m not in configured] + configured
        return {key: value for key, value in params.items() if value is not None}

    def _fit(
        self,
        X,
        y,
        sample_weight=None,
        init_score=None,
        group=None,
        eval_set=None,
        eval_names=None,
        eval_sample_weight=None,
        eval_init_score=None,
        eval_group=None,
        eval_metric=None,
        feature_name="auto",
        categorical_feature="auto",
        callbacks=None,
        init_model=None,
        params=None,
    ):
        if init_score is not None or eval_init_score is not None:
            raise ValueError("silvic does not support init_score")
        if init_model is not None:
            raise ValueError("silvic does not support init_model")
        params = params if params is not None else self._training_params(eval_metric)
        self._n_features = int(np.shape(X)[1])
        train_set = Dataset(
            X,
            label=y,
            weight=sample_weight,
            group=group,
            feature_name=feature_name,
            categorical_feature=categorical_feature,
            params=params,
        )
        valid_sets = []
        if eval_set is not None:
            if isinstance(eval_set, tuple):
                eval_set = [eval_set]
            if eval_group is not None and len(eval_group) != len(eval_set):
                raise ValueError(
                    f"Length of eval_group ({len(eval_group)}) not equal to length of eval_set ({len(eval_set)})"
                )
            for i, (valid_x, valid_y) in enumerate(eval_set):
                if valid_x is X and valid_y is y:
                    valid_sets.append(train_set)
                    continue
                valid_sets.append(
                    Dataset(
                        valid_x,
                        label=self._encode_labels(valid_y),
                        reference=train_set,
                        weight=_nth(eval_sample_weight, i),
                        group=_nth(eval_group, i),
                        params=params,
                    )
                )
        evals_result = {}
        callbacks = list(callbacks or []) + [record_evaluation(evals_result)]
        self._booster = train(
            params,
            train_set,
            num_boost_round=self.n_estimators,
            valid_sets=valid_sets,
            valid_names=eval_names,
            callbacks=callbacks,
        )
        self._evals_result = evals_result
        return self

    def _encode_labels(self, y):
        return y

    def fit(
        self,
        X,
        y,
        sample_weight=None,
        init_score=None,
        eval_set=None,
        eval_names=None,
        eval_sample_weight=None,
        eval_init_score=None,
        eval_metric=None,
        feature_name="auto",
        categorical_feature="auto",
        callbacks=None,
        init_model=None,
    ):
        """Trains on ``X``, ``y``; ``eval_set`` is a list of ``(X, y)`` validation pairs (LightGBM ``fit``)."""
        return self._fit(
            X,
            y,
            sample_weight=sample_weight,
            init_score=init_score,
            eval_set=eval_set,
            eval_names=eval_names,
            eval_sample_weight=eval_sample_weight,
            eval_init_score=eval_init_score,
            eval_metric=eval_metric,
            feature_name=feature_name,
            categorical_feature=categorical_feature,
            callbacks=callbacks,
            init_model=init_model,
        )

    def predict(self, X, raw_score=False, num_iteration=None):
        if X is not None and np.shape(X)[1] != self.n_features_:
            raise ValueError(
                f"Number of features of the model must match the input. Model n_features_ is {self.n_features_} "
                f"and input n_features is {np.shape(X)[1]}"
            )
        return self.booster_.predict(X, raw_score=raw_score, num_iteration=num_iteration)

    def _check_fitted(self):
        if self._booster is None:
            raise NotFittedError("No booster found. Need to call fit beforehand.")

    @property
    def booster_(self):
        self._check_fitted()
        return self._booster

    @property
    def evals_result_(self):
        self._check_fitted()
        return self._evals_result

    @property
    def best_score_(self):
        self._check_fitted()
        return self._booster.best_score

    @property
    def best_iteration_(self):
        self._check_fitted()
        return self._booster.best_iteration

    @property
    def n_features_(self):
        self._check_fitted()
        return self._n_features

    @property
    def n_features_in_(self):
        self._check_fitted()
        return self._n_features

    @property
    def n_estimators_(self):
        self._check_fitted()
        return self._booster.current_iteration()

    @property
    def objective_(self):
        self._check_fitted()
        return self._booster.params.get("objective")

    @property
    def feature_name_(self):
        self._check_fitted()
        return self._booster.feature_name()

    @property
    def feature_importances_(self):
        self._check_fitted()
        return self._booster.feature_importance(importance_type=self.importance_type)


def _nth(collection, i):
    """LightGBM ``_extract_evaluation_meta_data``: the ``i``-th item of a list or dict, else ``None``."""
    if collection is None:
        return None
    if isinstance(collection, list):
        return collection[i] if len(collection) > i else None
    if isinstance(collection, dict):
        return collection.get(i)
    raise TypeError("eval_sample_weight and eval_group should be dict or list")


class SilvicRegressor(_RegressorMixin, SilvicModel):
    """LightGBM ``LGBMRegressor``: the ``regression`` (L2) objective by default."""

    _default_objective = "regression"


class SilvicClassifier(_ClassifierMixin, SilvicModel):
    """LightGBM ``LGBMClassifier`` for two classes: the ``binary`` objective on labels encoded as ``classes_``."""

    _default_objective = "binary"

    def _encode_labels(self, y):
        y = np.asarray(y).ravel()
        codes = np.searchsorted(self._classes, y)
        unknown = (codes >= len(self._classes)) | (self._classes[np.minimum(codes, len(self._classes) - 1)] != y)
        if unknown.any():
            raise ValueError(f"y contains labels not seen in training: {np.unique(y[unknown])}")
        return codes.astype(np.float32)

    def _class_sample_weight(self, codes, sample_weight):
        """LightGBM's ``class_weight``: scikit-learn ``compute_sample_weight`` folded into ``sample_weight``."""
        if self.class_weight is None:
            return sample_weight
        if self.class_weight == "balanced":
            counts = np.bincount(codes.astype(np.int64), minlength=len(self._classes))
            per_class = len(codes) / (len(self._classes) * counts)
        else:
            per_class = np.array([self.class_weight.get(c, 1.0) for c in self._classes], dtype=np.float64)
        weight = per_class[codes.astype(np.int64)]
        return weight if sample_weight is None else weight * np.asarray(sample_weight)

    def fit(
        self,
        X,
        y,
        sample_weight=None,
        init_score=None,
        eval_set=None,
        eval_names=None,
        eval_sample_weight=None,
        eval_class_weight=None,
        eval_init_score=None,
        eval_metric=None,
        feature_name="auto",
        categorical_feature="auto",
        callbacks=None,
        init_model=None,
    ):
        """Trains a binary classifier; ``y`` holds two distinct labels."""
        if eval_class_weight is not None:
            raise ValueError("silvic does not support eval_class_weight")
        y_array = np.asarray(y).ravel()
        self._classes = np.unique(y_array)
        if len(self._classes) != 2:
            raise ValueError(f"silvic classifiers are binary; y has {len(self._classes)} classes")
        codes = self._encode_labels(y_array)
        if eval_set is not None:
            # LightGBM keeps `(X, y)` recognisable as the training set after encoding, so it is evaluated as
            # `training`.
            pairs = [eval_set] if isinstance(eval_set, tuple) else eval_set
            eval_set = [(vx, codes) if vx is X and vy is y else (vx, vy) for vx, vy in pairs]
        return self._fit(
            X,
            codes,
            sample_weight=self._class_sample_weight(codes, sample_weight),
            init_score=init_score,
            eval_set=eval_set,
            eval_names=eval_names,
            eval_sample_weight=eval_sample_weight,
            eval_init_score=eval_init_score,
            eval_metric=eval_metric,
            feature_name=feature_name,
            categorical_feature=categorical_feature,
            callbacks=callbacks,
            init_model=init_model,
        )

    @property
    def classes_(self):
        self._check_fitted()
        return self._classes

    @property
    def n_classes_(self):
        self._check_fitted()
        return len(self._classes)

    def predict_proba(self, X, raw_score=False, num_iteration=None):
        """``[P(classes_[0]), P(classes_[1])]`` per row, or the raw score with ``raw_score``."""
        result = super().predict(X, raw_score=raw_score, num_iteration=num_iteration)
        if raw_score:
            return result
        return np.vstack((1.0 - result, result)).transpose()

    def predict(self, X, raw_score=False, num_iteration=None):
        """The more probable class per row (``classes_[0]`` on a tie, as LightGBM's argmax picks)."""
        result = self.predict_proba(X, raw_score=raw_score, num_iteration=num_iteration)
        if raw_score:
            return result
        return self._classes[np.argmax(result, axis=1)]


class SilvicRanker(SilvicModel):
    """LightGBM ``LGBMRanker``: the ``lambdarank`` objective on query groups (``group`` holds query sizes)."""

    _default_objective = "lambdarank"

    def fit(
        self,
        X,
        y,
        sample_weight=None,
        init_score=None,
        group=None,
        eval_set=None,
        eval_names=None,
        eval_sample_weight=None,
        eval_init_score=None,
        eval_group=None,
        eval_metric=None,
        eval_at=(1, 2, 3, 4, 5),
        feature_name="auto",
        categorical_feature="auto",
        callbacks=None,
        init_model=None,
    ):
        """Trains a ranker; ``eval_at`` sets the NDCG positions unless ``**kwargs`` sets ``eval_at`` or an alias."""
        if group is None:
            raise ValueError("Should set group for ranking task")
        if eval_set is not None:
            if eval_group is None:
                raise ValueError("Eval_group cannot be None when eval_set is not None")
            if len(eval_group) != len(eval_set):
                raise ValueError("Length of eval_group should be equal to eval_set")
        params = self._training_params(eval_metric)
        if not _params.find(params, "eval_at"):
            params["eval_at"] = list(eval_at)
        return self._fit(
            X,
            y,
            sample_weight=sample_weight,
            init_score=init_score,
            group=group,
            eval_set=eval_set,
            eval_names=eval_names,
            eval_sample_weight=eval_sample_weight,
            eval_init_score=eval_init_score,
            eval_group=eval_group,
            eval_metric=eval_metric,
            feature_name=feature_name,
            categorical_feature=categorical_feature,
            callbacks=callbacks,
            init_model=init_model,
            params=params,
        )
