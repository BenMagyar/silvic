# /// script
# requires-python = ">=3.10"
# dependencies = ["lightgbm==4.7.0", "numpy"]
# ///
"""Records LightGBM 4.7.0 metric values for fixed scores into cases.json.

Each case's scores enter as the dataset's `init_score`; `Booster.eval_train` before any boosting round evaluates
LightGBM's metrics on exactly those scores. Scores are rounded to f32, the precision silvic keeps them in.
"""

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np

rng = np.random.default_rng(20261107)


def f32(values):
    return [float(v) for v in np.asarray(values, dtype=np.float32)]


def record(name, params, label, scores, weight=None, group=None):
    label, scores = f32(label), f32(scores)
    weight = None if weight is None else f32(weight)
    data = lgb.Dataset(
        rng.normal(size=(len(label), 2)),
        label=label,
        weight=weight,
        group=group,
        init_score=np.asarray(scores, dtype=np.float64),
        params={"verbose": -1},
    )
    booster = lgb.Booster({**params, "verbose": -1, "num_threads": 1}, data)
    expected = {metric: value for _, metric, value, _ in booster.eval_train()}
    return {
        "name": name,
        "objective": params["objective"],
        "sigmoid": params.get("sigmoid", 1.0),
        "metric": params["metric"],
        "eval_at": params.get("eval_at", []),
        "label": label,
        "weight": weight,
        "group": None if group is None else [int(g) for g in group],
        "scores": scores,
        "expected": expected,
    }


def ranking_case(name, weighted):
    group = rng.integers(1, 25, size=40)
    n = int(group.sum())
    label = rng.integers(0, 5, size=n).astype(float)
    label[: group[0]] = 0  # a query whose max DCG is 0 counts as 1
    scores = np.round(rng.normal(size=n), 1)  # one decimal: plenty of ties
    weight = rng.uniform(0.2, 3.0, size=n) if weighted else None
    params = {"objective": "lambdarank", "metric": "ndcg", "eval_at": [1, 3, 5, 10]}
    return record(name, params, label, scores, weight, group)


def binary_case(name, weighted):
    n = 500
    label = (rng.uniform(size=n) < 0.3).astype(float)
    scores = np.round(rng.normal(scale=2.0, size=n) + label, 1)
    scores[:5] = [40.0, -40.0, 40.0, -40.0, 0.0]  # saturated probabilities hit the kEpsilon floor
    label[:4] = [0, 1, 1, 0]
    weight = rng.uniform(0.1, 2.0, size=n) if weighted else None
    params = {"objective": "binary", "sigmoid": 1.7, "metric": ["binary_logloss", "auc", "l2"]}
    return record(name, params, label, scores, weight)


def regression_case(name, weighted):
    n = 300
    label = rng.normal(loc=0.5, size=n)
    scores = label + rng.normal(scale=0.5, size=n)
    weight = rng.uniform(0.1, 2.0, size=n) if weighted else None
    params = {"objective": "regression", "metric": ["l2", "rmse", "auc"]}
    return record(name, params, label, scores, weight)


cases = [
    ranking_case("ndcg", False),
    ranking_case("ndcg_weighted", True),
    binary_case("binary", False),
    binary_case("binary_weighted", True),
    regression_case("regression", False),
    regression_case("regression_weighted", True),
]
Path(__file__).with_name("cases.json").write_text(json.dumps({"lightgbm": lgb.__version__, "cases": cases}) + "\n")
