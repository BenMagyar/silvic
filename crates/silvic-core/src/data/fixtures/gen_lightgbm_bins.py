# /// script
# requires-python = ">=3.10"
# dependencies = ["lightgbm==3.3.5", "numpy<2"]
# ///
"""Records the bins LightGBM 3.3.5 builds for synthetic features, for `data/parity_tests.rs`.

The features come from a hash of (stream, row) and exact float arithmetic, so the Rust test rebuilds the same values
bit for bit. For every case:

- `feature_infos` comes from a one-iteration train on all features (categorical ones declared), under LightGBM's
  default `feature_pre_filter`, which is what marks the constant feature `none`.
- `thresholds` of a numerical feature come from a one-tree train on that feature alone, with the dense rank of the
  value as label, `num_leaves` above the bin count and `min_data_in_leaf=1`: every boundary between two non-empty bins
  then has positive gain, so the tree splits at all of them and its thresholds are the bin upper bounds.

Run: `uv run crates/silvic-core/src/data/fixtures/gen_lightgbm_bins.py` (or `.venv-lgb3/bin/python ...`).
"""

import json
import pathlib

import lightgbm as lgb
import numpy as np

assert lgb.__version__ == "3.3.5", lgb.__version__

OUT = pathlib.Path(__file__).with_name("lightgbm_bins.json")

CASES = [
    {"name": "all_rows", "rows": 150_000, "params": {}},
    {"name": "selection_sample", "rows": 250_000, "params": {}},
    {
        "name": "set_sample_small_max_bin",
        "rows": 60_000,
        "params": {"bin_construct_sample_cnt": 2000, "max_bin": 31, "min_data_in_bin": 5},
    },
]

FEATURES = [
    "normal",
    "zeros_nan",
    "integers",
    "mass_points",
    "negative",
    "tiny",
    "categorical",
    "cat_ties",
    "cat_rare",
    "constant",
]
CATEGORICAL = {"categorical", "cat_ties", "cat_rare"}


def mix32(x):
    x = x.astype(np.uint32)
    x ^= x >> np.uint32(16)
    x *= np.uint32(0x7FEB352D)
    x ^= x >> np.uint32(15)
    x *= np.uint32(0x846CA68B)
    x ^= x >> np.uint32(16)
    return x


def unit(stream, rows):
    h = mix32(mix32(np.arange(rows, dtype=np.uint32)) ^ mix32(np.array([stream], dtype=np.uint32)))
    return (h >> np.uint32(8)).astype(np.float64) / 16777216.0


def feature(name, rows):
    if name == "normal":
        return (((unit(10, rows) + unit(11, rows)) + unit(12, rows)) + unit(13, rows) - 2.0) * 1.5
    if name == "zeros_nan":
        a = unit(20, rows)
        base = ((unit(21, rows) + unit(22, rows)) - 1.0) * 3.0
        return np.where(a < 0.3, 0.0, np.where(a < 0.4, np.nan, base))
    if name == "integers":
        return np.floor(unit(30, rows) * 20.0) - 5.0
    if name == "mass_points":
        a = unit(40, rows)
        return np.where(a < 0.5, 1.0, ((unit(41, rows) * unit(42, rows)) * unit(43, rows)) * 1000.0)
    if name == "negative":
        u = unit(50, rows)
        return -(u * u * 10.0) - 0.5
    if name == "tiny":
        a = unit(60, rows)
        return np.where(a < 0.2, 1e-36, np.where(a < 0.3, -1e-36, unit(61, rows) - 0.5))
    if name == "categorical":
        a = unit(70, rows)
        u = unit(71, rows)
        return np.where(a < 0.05, np.nan, np.mod(np.floor(u * u * 12.0) * 7.0, 12.0) - 1.0)
    if name == "cat_ties":
        return (np.arange(rows) % 6).astype(np.float64)
    if name == "cat_rare":
        a = unit(80, rows)
        rare = np.arange(rows) % 75000 == 7
        return np.where(rare, 40.0, np.where(a < 0.7, 5.0, np.where(a < 0.99, 9.0, 2.0)))
    if name == "constant":
        return np.full(rows, 3.0)
    raise ValueError(name)


def base_params(case):
    return {"verbose": -1, **case["params"]}


def feature_infos(case, matrix):
    label = unit(99, case["rows"])
    params = {**base_params(case), "num_leaves": 4}
    categorical = [i for i, name in enumerate(FEATURES) if name in CATEGORICAL]
    booster = lgb.train(params, lgb.Dataset(matrix, label, categorical_feature=categorical), num_boost_round=1)
    for line in booster.model_to_string().splitlines():
        if line.startswith("feature_infos="):
            return line[len("feature_infos=") :].split(" ")
    raise RuntimeError("no feature_infos")


def thresholds(case, column):
    finite = ~np.isnan(column)
    label = np.full(column.shape, -1.0)
    label[finite] = np.unique(column[finite], return_inverse=True)[1]
    params = {
        **base_params(case),
        "objective": "regression",
        "num_leaves": 300,
        "min_data_in_leaf": 1,
        "min_sum_hessian_in_leaf": 0.0,
        "learning_rate": 1.0,
    }
    booster = lgb.train(params, lgb.Dataset(column.reshape(-1, 1), label), num_boost_round=1)
    found = []
    for line in booster.model_to_string().splitlines():
        if line.startswith("threshold="):
            found.extend(t for t in line[len("threshold=") :].split(" ") if t)
    assert found, case["name"]
    return sorted(found, key=float)


def main():
    out = {"lightgbm": lgb.__version__, "features": FEATURES, "cases": []}
    for case in CASES:
        columns = [feature(name, case["rows"]) for name in FEATURES]
        infos = feature_infos(case, np.column_stack(columns))
        features = []
        for name, column, info in zip(FEATURES, columns, infos):
            entry = {"name": name, "feature_infos": info}
            if name not in CATEGORICAL and name != "constant":
                entry["thresholds"] = thresholds(case, column)
            features.append(entry)
        out["cases"].append({**case, "features": features})
    OUT.write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
