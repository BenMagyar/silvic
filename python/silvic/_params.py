"""Parameter names and aliases, as LightGBM 3.3.5's ``config.h`` lists them (``_ConfigAliases`` in its Python package).

The Rust core validates every key itself and rejects a canonical name and its alias set together. The Python layer
needs the aliases for the places where LightGBM's Python package resolves them before the core sees them: which of
``num_boost_round`` and ``params['num_iterations']`` wins, merging dataset and training parameters, and dropping a
scikit-learn constructor default the caller overrides with an alias in ``**kwargs``.
"""

import json

import numpy as np

_ALIASES = {
    "objective": ("objective_type", "app", "application", "loss"),
    "boosting": ("boosting_type", "boost"),
    "num_iterations": (
        "num_iteration", "n_iter", "num_tree", "num_trees", "num_round", "num_rounds", "num_boost_round",
        "n_estimators", "max_iter",
    ),
    "learning_rate": ("shrinkage_rate", "eta"),
    "num_leaves": ("num_leaf", "max_leaves", "max_leaf", "max_leaf_nodes"),
    "num_threads": ("num_thread", "nthread", "nthreads", "n_jobs"),
    "device_type": ("device",),
    "seed": ("random_seed", "random_state"),
    "max_depth": (),
    "min_data_in_leaf": ("min_data_per_leaf", "min_data", "min_child_samples", "min_samples_leaf"),
    "min_sum_hessian_in_leaf": ("min_sum_hessian_per_leaf", "min_sum_hessian", "min_hessian", "min_child_weight"),
    "bagging_fraction": ("sub_row", "subsample", "bagging"),
    "bagging_freq": ("subsample_freq",),
    "bagging_seed": ("bagging_fraction_seed",),
    "feature_fraction": ("sub_feature", "colsample_bytree"),
    "lambda_l1": ("reg_alpha", "l1_regularization"),
    "lambda_l2": ("reg_lambda", "lambda", "l2_regularization"),
    "min_gain_to_split": ("min_split_gain",),
    "max_bin": ("max_bins",),
    "bin_construct_sample_cnt": ("subsample_for_bin",),
    "data_random_seed": ("data_seed",),
    "metric": ("metrics", "metric_types"),
    "eval_at": ("ndcg_eval_at", "ndcg_at", "map_eval_at", "map_at"),
    "verbosity": ("verbose",),
    "early_stopping_round": ("early_stopping_rounds", "early_stopping", "n_iter_no_change"),
}


def aliases(name):
    """``name`` and its aliases."""
    return (name, *_ALIASES.get(name, ()))


def canonical(key):
    """The canonical name of ``key`` (``key`` itself when it has no aliases)."""
    for name, names in _ALIASES.items():
        if key == name or key in names:
            return name
    return key


def find(params, name):
    """The keys of ``params`` that set ``name``, in insertion order."""
    names = aliases(name)
    return [key for key in params if key in names]


def merge(base, override):
    """``base`` updated with ``override``, dropping every key of ``base`` that ``override`` sets under any alias."""
    overridden = {canonical(key) for key in override}
    merged = {key: value for key, value in base.items() if canonical(key) not in overridden}
    merged.update(override)
    return merged


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (set, tuple)):
        return list(value)
    raise TypeError(f"parameter value {value!r} of type {type(value).__name__} is not JSON serializable")


def to_json(params):
    """``params`` as the JSON object the core parses; ``None`` values are dropped, as LightGBM drops them."""
    return json.dumps({key: value for key, value in params.items() if value is not None}, default=_json_default)
