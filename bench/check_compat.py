"""Checks that a silvic-trained lambdarank model loads, predicts and explains in LightGBM 3.3.5 and lleaves 1.1.0.

Run in two steps, because silvic's environment (`.venv`, numpy 2) and LightGBM 3.3.5's (`.venv-lgb3`: lightgbm 3.3.5,
lleaves 1.1.0, numpy<2) cannot share an interpreter:

    .venv/bin/python bench/check_compat.py train --rows 2000000 --out /tmp/silvic-compat
    .venv-lgb3/bin/python bench/check_compat.py check --out /tmp/silvic-compat

`train` fits a lambdarank model on the Metal backend and writes `model.txt`, the probe rows and silvic's raw
predictions. `check` loads the model with LightGBM 3.3.5 (`Booster(model_str=...)`, through a pickle round trip),
compares predictions, computes SHAP contributions with `pred_contrib=True` and checks they sum to the prediction, then
compiles the model with lleaves 1.1.0 and compares again.
"""

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np

PROBE_ROWS = 20_000


def train(args):
    import silvic

    import gen_ltr

    train_set, valid_set = gen_ltr.generate(args.rows)
    params = {
        "objective": "lambdarank",
        "num_leaves": 53,
        "max_depth": 11,
        "learning_rate": 0.05,
        "min_data_in_leaf": 52,
        "min_sum_hessian_in_leaf": 0.0015,
        "lambda_l1": 14.0,
        "lambda_l2": 0.27,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "feature_fraction": 0.9,
        "seed": 123,
        "device_type": "metal",
        "verbose": -1,
    }
    dataset = silvic.Dataset(train_set["X"], train_set["y"], group=train_set["group"],
                             categorical_feature=[int(i) for i in train_set["categorical"]])
    booster = silvic.train(params, dataset, num_boost_round=args.iterations)
    probe = np.ascontiguousarray(valid_set["X"][:PROBE_ROWS], dtype=np.float64)
    args.out.mkdir(parents=True, exist_ok=True)
    booster.save_model(args.out / "model.txt")
    np.save(args.out / "probe.npy", probe)
    np.save(args.out / "silvic_raw.npy", booster.predict(probe, raw_score=True))
    print(f"trained {booster.num_trees()} trees -> {args.out}")


def check(args):
    import lightgbm as lgb
    import lleaves

    model_str = (args.out / "model.txt").read_text()
    probe = np.load(args.out / "probe.npy")
    expected = np.load(args.out / "silvic_raw.npy")

    booster = pickle.loads(pickle.dumps(lgb.Booster(model_str=model_str)))
    predicted = booster.predict(probe, raw_score=True)
    lgb_max_diff = float(np.max(np.abs(predicted - expected)))

    contrib = booster.predict(probe, pred_contrib=True)
    shap_max_diff = float(np.max(np.abs(contrib.sum(axis=1) - predicted)))

    llvm_model = lleaves.Model(model_file=str(args.out / "model.txt"))
    llvm_model.compile(cache=str(args.out / "cache.efl"))
    lleaves_max_diff = float(np.max(np.abs(llvm_model.predict(probe) - expected)))

    print(f"lightgbm {lgb.__version__}: {booster.num_trees()} trees, max |prediction - silvic| = {lgb_max_diff:.3g}")
    print(f"pred_contrib: shape {contrib.shape}, max |sum(contrib) - prediction| = {shap_max_diff:.3g}")
    print(f"lleaves {lleaves.__version__ if hasattr(lleaves, '__version__') else '1.1.0'}: "
          f"max |prediction - silvic| = {lleaves_max_diff:.3g}")
    failed = lgb_max_diff != 0.0 or shap_max_diff > 1e-9 or lleaves_max_diff > 1e-9
    sys.exit(1 if failed else 0)


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    train_parser = sub.add_parser("train")
    train_parser.add_argument("--rows", type=int, default=2_000_000)
    train_parser.add_argument("--out", type=Path, required=True)
    train_parser.add_argument("--iterations", type=int, default=300)
    check_parser = sub.add_parser("check")
    check_parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    train(args) if args.command == "train" else check(args)


if __name__ == "__main__":
    main()
