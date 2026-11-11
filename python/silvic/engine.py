"""``train``, mirroring LightGBM 3.3.5 ``lightgbm.train`` (whose argument order it keeps) and LightGBM 4.x callbacks.

A number of iterations in ``params`` (``num_iterations`` or any alias) wins over ``num_boost_round``, as in
LightGBM. ``verbose_eval`` keeps LightGBM 3.3.5's meaning: by default the evaluation results print every iteration
unless ``callbacks`` are given; an int prints every that many iterations. A validation set that *is* the training
set is evaluated as ``training`` (or its ``valid_names`` entry), as in LightGBM.

Custom objectives (``fobj``), custom metrics (``feval``), continued training (``init_model``), early stopping and
``learning_rates`` are outside silvic's scope and raise ``ValueError``.
"""

import collections

from . import _params
from . import _silvic
from .basic import Booster, Dataset, _dump_pandas_categorical
from .callback import CallbackEnv, log_evaluation, record_evaluation


def _resolve_num_iterations(params, num_boost_round):
    """Pops every ``num_iterations`` alias from ``params`` (the last one wins) and sets ``num_iterations``."""
    for key in _params.find(params, "num_iterations"):
        num_boost_round = params.pop(key)
    if num_boost_round is None or num_boost_round <= 0:
        raise ValueError("num_boost_round should be greater than zero.")
    params["num_iterations"] = int(num_boost_round)
    return params["num_iterations"]


def _reject_early_stopping(params, early_stopping_rounds=None):
    for key in _params.find(params, "early_stopping_round"):
        value = params.pop(key)
        if value:
            raise ValueError(f"silvic does not support early stopping (`{key}={value}`)")
    if early_stopping_rounds:
        raise ValueError("silvic does not support early stopping (early_stopping_rounds)")


def _is_early_stopping(callback):
    # LightGBM 4.x returns an `_EarlyStoppingCallback`; 3.3.5 returns the closure `early_stopping.<locals>._callback`.
    name = getattr(callback, "__qualname__", "") or type(callback).__qualname__
    return type(callback).__name__ == "_EarlyStoppingCallback" or name.startswith("early_stopping.")


def _prepare_callbacks(callbacks, verbose_eval, evals_result):
    """The callbacks to run after each iteration, sorted by ``order`` as LightGBM sorts them."""
    callbacks = list(callbacks or [])
    for callback in callbacks:
        if _is_early_stopping(callback):
            raise ValueError("silvic does not support early stopping callbacks")
        if getattr(callback, "before_iteration", False):
            raise ValueError("silvic does not support callbacks that run before an iteration (before_iteration)")
    # LightGBM gives callbacks without an `order` a negative one in list order, so they run first.
    ordered = [(getattr(cb, "order", i - len(callbacks)), i, cb) for i, cb in enumerate(callbacks)]
    if verbose_eval == "warn":
        verbose_eval = not callbacks
    if verbose_eval is True:
        verbose_eval = 1
    if verbose_eval and isinstance(verbose_eval, int):
        ordered.append((10, len(ordered), log_evaluation(verbose_eval)))
    if evals_result is not None:
        ordered.append((20, len(ordered), record_evaluation(evals_result)))
    return [cb for _, _, cb in sorted(ordered, key=lambda item: (item[0], item[1]))]


class _IterationHook:
    """The ``_silvic.train`` callback: builds LightGBM's ``CallbackEnv`` and runs the callbacks."""

    def __init__(self, callbacks, params, num_iterations):
        self.callbacks = callbacks
        self.params = params
        self.num_iterations = num_iterations
        self.last_evals = []

    def __call__(self, iteration, evals):
        self.last_evals = evals
        env = CallbackEnv(
            model=None,
            params=self.params,
            iteration=iteration,
            begin_iteration=0,
            end_iteration=self.num_iterations,
            evaluation_result_list=evals,
        )
        for callback in self.callbacks:
            callback(env)
        return False

    def best_score(self):
        best_score = collections.defaultdict(collections.OrderedDict)
        for dataset_name, eval_name, score, _ in self.last_evals:
            best_score[dataset_name][eval_name] = score
        return best_score


def _validation_handles(train_set, valid_sets, valid_names, params):
    """``[(name, _silvic.Dataset)]`` for ``valid_sets``, constructed against ``train_set``."""
    if valid_sets is None:
        return []
    if isinstance(valid_sets, Dataset):
        valid_sets = [valid_sets]
    if isinstance(valid_names, str):
        valid_names = [valid_names]
    handles = []
    for i, valid_set in enumerate(valid_sets):
        if not isinstance(valid_set, Dataset):
            raise TypeError("Training only accepts Dataset object")
        if valid_set is train_set:
            name = valid_names[i] if valid_names is not None and len(valid_names) > i else "training"
            handles.append((name, train_set._handle))
            continue
        if valid_set.reference is None:
            valid_set.set_reference(train_set)
        elif valid_set.reference is not train_set:
            raise ValueError(f"validation set {i} has a reference other than the training set")
        valid_set._update_params(params).construct()
        name = valid_names[i] if valid_names is not None and len(valid_names) > i else f"valid_{i}"
        handles.append((name, valid_set._handle))
    return handles


def train(
    params,
    train_set,
    num_boost_round=100,
    valid_sets=None,
    valid_names=None,
    fobj=None,
    feval=None,
    init_model=None,
    feature_name="auto",
    categorical_feature="auto",
    early_stopping_rounds=None,
    evals_result=None,
    verbose_eval="warn",
    learning_rates=None,
    keep_training_booster=False,
    callbacks=None,
):
    """Trains a model on ``train_set`` and returns its ``Booster`` (LightGBM ``train``).

    ``evals_result`` (3.3.5) and ``record_evaluation`` (4.x) both record the validation history.
    ``keep_training_booster`` has no effect: a silvic ``Booster`` is always a finished model.
    """
    unsupported = (("fobj", fobj), ("feval", feval), ("init_model", init_model), ("learning_rates", learning_rates))
    for name, value in unsupported:
        if value is not None:
            raise ValueError(f"silvic does not support `{name}`")
    if not isinstance(train_set, Dataset):
        raise TypeError("Training only accepts Dataset object")
    params = dict(params or {})
    num_iterations = _resolve_num_iterations(params, num_boost_round)
    _reject_early_stopping(params, early_stopping_rounds)
    train_set._update_params(params).set_feature_name(feature_name).set_categorical_feature(categorical_feature)
    train_set.construct()
    valid = _validation_handles(train_set, valid_sets, valid_names, params)
    hook = _IterationHook(_prepare_callbacks(callbacks, verbose_eval, evals_result), params, num_iterations)
    model_str = _silvic.train(
        _params.to_json(params),
        train_set._handle,
        valid,
        callback=hook,
        pandas_categorical=_dump_pandas_categorical(train_set.pandas_categorical),
    )
    booster = Booster(params=params, model_str=model_str)
    booster.best_score = hook.best_score()
    return booster
