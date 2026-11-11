"""Training callbacks, as LightGBM 4.x ``lightgbm.callback`` defines them.

A callback is called after every tree with a ``CallbackEnv`` whose ``evaluation_result_list`` holds
``(dataset_name, metric_name, value, is_higher_better)`` tuples, in LightGBM's format. silvic computes evaluations
inside the training loop, so ``env.model`` is ``None`` and callbacks that run before an iteration
(``before_iteration = True``, e.g. ``reset_parameter``) or stop early (``early_stopping``) are rejected by ``train``.
Callbacks run in ascending ``order``, as in LightGBM.
"""

import collections

CallbackEnv = collections.namedtuple(
    "CallbackEnv",
    ["model", "params", "iteration", "begin_iteration", "end_iteration", "evaluation_result_list"],
)


def _format_eval_result(value):
    return f"{value[0]}'s {value[1]}: {value[2]:g}"


class _LogEvaluationCallback:
    """Prints ``[iteration]\\t<dataset>'s <metric>: <value>...`` every ``period`` iterations and at the last."""

    order = 10
    before_iteration = False

    def __init__(self, period=1, show_stdv=True):
        self.period = period
        self.show_stdv = show_stdv

    def __call__(self, env):
        if self.period > 0 and env.evaluation_result_list:
            iteration = env.iteration + 1
            if iteration % self.period == 0 or iteration == env.end_iteration:
                result = "\t".join(_format_eval_result(x) for x in env.evaluation_result_list)
                print(f"[{iteration}]\t{result}")


class _RecordEvaluationCallback:
    """Appends every evaluation to ``eval_result[dataset][metric]``; clears ``eval_result`` on the first call."""

    order = 20
    before_iteration = False

    def __init__(self, eval_result):
        if not isinstance(eval_result, dict):
            raise TypeError("eval_result should be a dictionary")
        self.eval_result = eval_result

    def _init(self, env):
        self.eval_result.clear()
        for data_name, eval_name, *_ in env.evaluation_result_list:
            self.eval_result.setdefault(data_name, collections.OrderedDict())
            self.eval_result[data_name].setdefault(eval_name, [])

    def __call__(self, env):
        if env.iteration == env.begin_iteration:
            self._init(env)
        for data_name, eval_name, result, *_ in env.evaluation_result_list:
            self.eval_result[data_name][eval_name].append(result)


def log_evaluation(period=1, show_stdv=True):
    """A callback that prints the evaluation results every ``period`` iterations (LightGBM ``log_evaluation``)."""
    return _LogEvaluationCallback(period=period, show_stdv=show_stdv)


def record_evaluation(eval_result):
    """A callback that records the evaluation history into ``eval_result`` (LightGBM ``record_evaluation``)."""
    return _RecordEvaluationCallback(eval_result)
