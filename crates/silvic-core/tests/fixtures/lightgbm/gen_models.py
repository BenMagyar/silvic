#!/usr/bin/env -S uv run --script
# /// script
# requires-python = "==3.12.*"
# dependencies = ["lightgbm==3.3.5", "numpy<2"]
# ///
"""Generates cases.json and <case>.txt: LightGBM 3.3.5 models and LightGBM's predictions for rows that reach every
decision path, which `tests/model_lightgbm.rs` holds `silvic_core::Model` to.

Each <case>.txt is `Booster.model_to_string()` of the booster `lgb.train` returns (LightGBM reloads it from its own
text before returning, so the file is what a reload writes). The cases:
  regression_nan_cat  regression; a feature with NaN (MissingType::NaN), one without (None), a sorted-ctr and a
                      one-hot categorical feature, and a feature split at +-kZeroThreshold;
  zero_missing        regression with zero_as_missing (MissingType::Zero) and NaN in the data;
  binary              binary with sigmoid 0.7, NaN features;
  lambdarank          lambdarank with a custom label_gain, query groups and a categorical feature;
  categorical_wide    a categorical feature with values up to 99, so bitsets span several 32-bit words;
  single_leaf         a constant first tree (num_leaves=1, from training on a constant label), then trees trained
                      on top of it with init_model.

Rows: the first training rows, every feature set to an edge value in turn (NaN, +-0, +-1e-36, +-kZeroThreshold,
+-inf, +-1e300, the smallest subnormal; for categorical features also negative, fractional, unseen, >= 32 and
out-of-int32-range values), and every numerical threshold with its two neighbouring doubles.

Format (one case object per entry of "cases"; one row per line):
  {"lightgbm_version", "generator",
   "cases": [{"name", "model_file", "rows": [[value]], "raw": [value], "predicted": [value],
              "prefix_trees", "raw_prefix": [value], "importance_split": [int], "importance_gain": [value]}]}
  Every float is a string holding Python's repr (shortest round-trip digits, "nan", "inf", "-inf"), so it parses
  back to the same bits. "raw" is predict(raw_score=True), "predicted" is predict() (sigmoid for binary),
  "raw_prefix" is predict(raw_score=True, num_iteration=prefix_trees). Importances are
  Booster.feature_importance("split"/"gain") of the reloaded model.

Usage, with the repo's LightGBM 3.3.5 environment:
  .venv-lgb3/bin/python crates/silvic-core/tests/fixtures/lightgbm/gen_models.py
or with uv, which provides the Python and LightGBM the header pins:
  ./gen_models.py
Writes next to this script. Fixtures are recorded from LightGBM and not regenerated casually: a new file changes what
the byte-for-byte round-trip test holds silvic to. gen_models_v4.py records a subset with LightGBM 4.x.
"""

import json
import math
import os
import sys

import lightgbm as lgb
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EXPECTED_VERSION = "3.3.5"

# LightGBM `kZeroThreshold` is the float 1e-35f.
K_ZERO = float(np.float32(1e-35))
NUMERIC_EDGES = [
    math.nan, 0.0, -0.0, 1e-36, -1e-36, 1e-35, -1e-35, K_ZERO, -K_ZERO, float(np.nextafter(K_ZERO, 1.0)),
    -float(np.nextafter(K_ZERO, 1.0)), math.inf, -math.inf, 1e300, -1e300, 5e-324,
]
CATEGORICAL_EDGES = [
    -1.0, -0.5, 0.5, 2.7, 31.0, 32.0, 33.0, 63.0, 64.0, 65.0, 99.0, 100.0, 500.0, 2.0**31 - 1, 2.0**31, -(2.0**31),
    1e10, -1e10, 2.0**32 + 5,
]
TRAIN_ROWS = 100
MAX_THRESHOLDS = 60

COMMON = {"verbosity": -1, "num_threads": 1, "deterministic": True, "force_row_wise": True, "seed": 7}


def train(params, x, y, rounds, names, categorical=(), group=None, init_model=None):
    data = lgb.Dataset(x, y, group=group, feature_name=names or "auto",
                       categorical_feature=list(categorical) or "auto", free_raw_data=False)
    return lgb.train({**COMMON, **params}, data, num_boost_round=rounds, init_model=init_model)


def regression_nan_cat():
    rng = np.random.default_rng(11)
    n = 3000
    num_nan = rng.normal(size=n)
    num_nan[rng.random(n) < 0.15] = np.nan
    num_dense = rng.uniform(-2, 2, n)
    cat_ctr = rng.integers(0, 10, n).astype(float)
    cat_onehot = rng.integers(0, 3, n).astype(float)
    num_zeros = np.where(rng.random(n) < 0.4, 0.0, rng.normal(size=n))
    cat_effect = rng.normal(size=10) * 2
    y = (2 * np.nan_to_num(num_nan, nan=1.5) + np.sin(3 * num_dense) + cat_effect[cat_ctr.astype(int)]
         + np.array([0.0, 1.0, -1.0])[cat_onehot.astype(int)] + 0.7 * (num_zeros > 0) - 1.5 * (num_zeros < 0)
         + rng.normal(size=n) * 0.1)
    x = np.column_stack([num_nan, num_dense, cat_ctr, cat_onehot, num_zeros])
    params = {"objective": "regression", "num_leaves": 15, "learning_rate": 0.1, "min_data_in_leaf": 10,
              "min_data_per_group": 10, "cat_smooth": 1.0, "cat_l2": 1.0, "max_cat_to_onehot": 4}
    names = ["num_nan", "num_dense", "cat_ctr", "cat_onehot", "num_zeros"]
    return train(params, x, y, 25, names, categorical=[2, 3]), x, [2, 3]


def zero_missing():
    rng = np.random.default_rng(12)
    n = 2000
    x = rng.normal(size=(n, 3))
    x[rng.random((n, 3)) < 0.3] = 0.0
    x[rng.random((n, 3)) < 0.1] = np.nan
    filled = np.nan_to_num(x, nan=0.0)
    y = filled[:, 0] * 2 - (filled[:, 1] == 0) * 1.5 + np.abs(filled[:, 2]) + rng.normal(size=n) * 0.1
    params = {"objective": "regression", "zero_as_missing": True, "num_leaves": 8, "min_data_in_leaf": 10}
    return train(params, x, y, 15, ["a", "b", "c"]), x, []


def binary():
    rng = np.random.default_rng(13)
    n = 3000
    x = rng.normal(size=(n, 4))
    x[rng.random((n, 4)) < 0.1] = np.nan
    logit = np.nan_to_num(x[:, 0], nan=-1.0) * 1.5 - np.nan_to_num(x[:, 1], nan=0.5) + x[:, 2] * x[:, 3]
    y = (rng.random(n) < 1 / (1 + np.exp(-np.nan_to_num(logit)))).astype(float)
    params = {"objective": "binary", "sigmoid": 0.7, "num_leaves": 12, "min_data_in_leaf": 15}
    return train(params, x, y, 30, None), x, []


def lambdarank():
    rng = np.random.default_rng(14)
    sizes = rng.integers(5, 30, 150)
    n = int(sizes.sum())
    x = rng.normal(size=(n, 4))
    x[:, 3] = rng.integers(0, 6, n)
    x[rng.random(n) < 0.05, 0] = np.nan
    score = np.nan_to_num(x[:, 0]) + 0.5 * x[:, 1] - 0.3 * x[:, 2] ** 2 + (x[:, 3] == 2) * 1.0
    y = np.clip(np.round(score + rng.normal(size=n) * 0.5 + 1.5), 0, 4)
    params = {"objective": "lambdarank", "label_gain": [0, 2, 5, 9, 20], "eval_at": [3, 5], "num_leaves": 10,
              "min_data_in_leaf": 10, "min_data_per_group": 10, "cat_smooth": 1.0}
    names = ["q0", "q1", "q2", "qcat"]
    return train(params, x, y, 20, names, categorical=[3], group=sizes), x, [3]


def categorical_wide():
    rng = np.random.default_rng(15)
    n = 6000
    cat = rng.integers(0, 100, n).astype(float)
    num = rng.normal(size=n)
    effect = rng.normal(size=100) * 3
    y = effect[cat.astype(int)] + num + rng.normal(size=n) * 0.1
    x = np.column_stack([cat, num])
    params = {"objective": "regression", "num_leaves": 12, "max_cat_threshold": 64, "min_data_per_group": 5,
              "cat_smooth": 1.0, "cat_l2": 1.0, "min_data_in_leaf": 5}
    return train(params, x, y, 15, ["wide_cat", "num"], categorical=[0]), x, [0]


def single_leaf():
    rng = np.random.default_rng(16)
    n = 1000
    x = rng.normal(size=(n, 2))
    constant = train({"objective": "regression"}, x, np.full(n, 3.0), 5, None)
    y = x[:, 0] - 2 * x[:, 1] + rng.normal(size=n) * 0.1
    return train({"objective": "regression", "num_leaves": 6}, x, y, 6, None, init_model=constant), x, []


CASES = {
    "regression_nan_cat": regression_nan_cat,
    "zero_missing": zero_missing,
    "binary": binary,
    "lambdarank": lambdarank,
    "categorical_wide": categorical_wide,
    "single_leaf": single_leaf,
}


def trees(model_str):
    """Per tree: {key: value} of its lines."""
    out = []
    for block in model_str.split("\nTree=")[1:]:
        lines = block.split("\n\n")[0].split("\n")[1:]
        out.append(dict(line.split("=", 1) for line in lines))
    return out


def numerical_thresholds(model_str):
    """(feature, threshold) of numerical splits in order of first appearance."""
    seen = []
    for tree in trees(model_str):
        for feature, threshold, decision in zip(tree["split_feature"].split(), tree["threshold"].split(),
                                                tree["decision_type"].split()):
            key = (int(feature), float(threshold))
            if int(decision) & 1 == 0 and key not in seen:
                seen.append(key)
    return seen[:MAX_THRESHOLDS]


def edge_rows(x, model_str, categorical):
    rows = [x[i].copy() for i in range(TRAIN_ROWS)]
    rows.append(np.full(x.shape[1], np.nan))
    rows.append(np.zeros(x.shape[1]))
    for feature in range(x.shape[1]):
        edges = NUMERIC_EDGES + (CATEGORICAL_EDGES if feature in categorical else [])
        for value in edges:
            row = x[feature].copy()
            row[feature] = value
            rows.append(row)
    for i, (feature, threshold) in enumerate(numerical_thresholds(model_str)):
        for value in (threshold, np.nextafter(threshold, -np.inf), np.nextafter(threshold, np.inf)):
            row = x[i % TRAIN_ROWS].copy()
            row[feature] = value
            rows.append(row)
    return np.ascontiguousarray(np.array(rows, dtype=np.float64))


def bits(values):
    return np.asarray(values, dtype=np.float64).view(np.uint64)


def check_case(name, model_str):
    """Fails when a case no longer covers what it exists for."""
    parsed = trees(model_str)
    decisions = [int(d) for tree in parsed for d in tree["decision_type"].split()]
    if name == "categorical_wide":
        widths = []
        for tree in parsed:
            if "cat_boundaries" in tree:
                bounds = [int(v) for v in tree["cat_boundaries"].split()]
                widths.extend(b - a for a, b in zip(bounds, bounds[1:]))
        if max(widths, default=0) < 2:
            sys.exit("categorical_wide: no multi-word bitset")
    if name == "single_leaf" and parsed[0]["num_leaves"] != "1":
        sys.exit("single_leaf: the first tree has more than one leaf")
    if name == "regression_nan_cat":
        if not any((d >> 2) & 3 == 2 for d in decisions) or not any(d & 1 for d in decisions):
            sys.exit("regression_nan_cat: missing NaN or categorical splits")
        if f"threshold={-K_ZERO!r}" not in model_str and f" {-K_ZERO!r}" not in model_str:
            sys.exit("regression_nan_cat: no split at -kZeroThreshold")
    if name == "zero_missing" and not any((d >> 2) & 3 == 1 for d in decisions):
        sys.exit("zero_missing: no MissingType::Zero split")


def record(name, build, prefix):
    booster, x, categorical = build()
    model_str = booster.model_to_string()
    check_case(name, model_str)
    reloaded = lgb.Booster(model_str=model_str)
    if reloaded.model_to_string() != model_str:
        sys.exit(f"{name}: LightGBM's own reload changes the model text")
    rows = edge_rows(x, model_str, categorical)
    raw = reloaded.predict(rows, raw_score=True)
    if not np.array_equal(bits(raw), bits(booster.predict(rows, raw_score=True))):
        sys.exit(f"{name}: the reloaded booster predicts differently")
    prefix_trees = max(1, reloaded.num_trees() // 2)
    model_file = f"{prefix}{name}.txt"
    with open(os.path.join(HERE, model_file), "w", newline="") as f:
        f.write(model_str)
    return {
        "name": f"{prefix}{name}",
        "model_file": model_file,
        "rows": [[repr(float(v)) for v in row] for row in rows],
        "raw": [repr(float(v)) for v in raw],
        "predicted": [repr(float(v)) for v in reloaded.predict(rows)],
        "prefix_trees": prefix_trees,
        "raw_prefix": [repr(float(v)) for v in reloaded.predict(rows, raw_score=True, num_iteration=prefix_trees)],
        "importance_split": [int(v) for v in reloaded.feature_importance("split")],
        "importance_gain": [repr(float(v)) for v in reloaded.feature_importance("gain")],
    }


def dump(document, path):
    """JSON with one row or value list per line."""
    lines = ["{", f' "lightgbm_version": {json.dumps(document["lightgbm_version"])},',
             f' "generator": {json.dumps(document["generator"])},', ' "cases": [']
    cases = document["cases"]
    for i, case in enumerate(cases):
        lines.append("  {")
        items = list(case.items())
        for j, (key, value) in enumerate(items):
            comma = "," if j + 1 < len(items) else ""
            if key == "rows":
                lines.append('   "rows": [')
                lines.extend(f"    {json.dumps(row)}" + ("," if k + 1 < len(value) else "")
                             for k, row in enumerate(value))
                lines.append(f"   ]{comma}")
            else:
                lines.append(f"   {json.dumps(key)}: {json.dumps(value)}{comma}")
        lines.append("  }" + ("," if i + 1 < len(cases) else ""))
    lines.extend([" ]", "}"])
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def write_fixtures(names, prefix, cases_file, generator):
    cases = [record(name, CASES[name], prefix) for name in names]
    dump({"lightgbm_version": lgb.__version__, "generator": generator, "cases": cases},
         os.path.join(HERE, cases_file))
    for case in cases:
        print(f"{case['name']}: {len(case['rows'])} rows")


def main():
    if lgb.__version__ != EXPECTED_VERSION:
        sys.exit(f"needs lightgbm {EXPECTED_VERSION}, found {lgb.__version__}")
    write_fixtures(list(CASES), "", "cases.json", "gen_models.py")


if __name__ == "__main__":
    main()
