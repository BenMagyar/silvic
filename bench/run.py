"""Trains the same lambdarank model with LightGBM and silvic and reports wall time and NDCG.

    python bench/run.py --rows 2000000 --iterations 500 --engines lightgbm,silvic-cpu,silvic-metal
    python bench/run.py --data bench/data/ltr_2m --iterations 500 --engines lightgbm,silvic-metal

`--rows` generates the data in memory with `gen_ltr.py` (the same rows `gen_ltr.py --rows N` writes); `--data`
reads a pair it wrote. Each engine trains from numpy arrays already in memory; the reported time covers dataset
construction (binning) and training, which is what a training job pays. NDCG@12 and NDCG@50 are computed here, as
LightGBM's `NDCGMetric` computes them, from each model's predictions by `lightgbm.Booster`, so every engine is scored
by the same code.
"""

import argparse
import json
import time

import lightgbm as lgb
import numpy as np

import gen_ltr

PARAMS = {
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
    "eval_at": [12, 50],
    "seed": 123,
    "verbose": -1,
}


def load(prefix):
    train = np.load(f"{prefix}.train.npz")
    valid = np.load(f"{prefix}.valid.npz")
    return train, valid


def ndcg(valid, model_str, eval_at=(12, 50)):
    """LightGBM's `NDCGMetric`: stable sort by score desc, gain `2^label - 1`, discount `1/log2(2 + i)`, and a query
    without relevant items counts as 1."""
    scores = lgb.Booster(model_str=model_str).predict(valid["X"], raw_score=True)
    labels = valid["y"].astype(np.int64)
    gain = (1 << labels) - 1
    totals = {k: 0.0 for k in eval_at}
    start = 0
    for size in valid["group"]:
        end = start + int(size)
        order = np.argsort(-scores[start:end], kind="stable")
        ideal = np.sort(gain[start:end])[::-1]
        ranked = gain[start:end][order]
        discount = 1.0 / np.log2(np.arange(end - start) + 2.0)
        for k in eval_at:
            max_dcg = float((ideal[:k] * discount[:k]).sum())
            totals[k] += 1.0 if max_dcg <= 0 else float((ranked[:k] * discount[:k]).sum()) / max_dcg
        start = end
    return {f"ndcg@{k}": totals[k] / len(valid["group"]) for k in eval_at}


def run_lightgbm(train, valid, iterations, quantized):
    params = {**PARAMS, "num_iterations": iterations, "num_threads": 0}
    if quantized:
        params.update({"use_quantized_grad": True, "num_grad_quant_bins": 4, "quant_train_renew_leaf": True})
    start = time.perf_counter()
    dtrain = lgb.Dataset(train["X"], train["y"], group=train["group"],
                         categorical_feature=[int(i) for i in train["categorical"]])
    booster = lgb.train(params, dtrain)
    return time.perf_counter() - start, booster.model_to_string()


def run_silvic(train, valid, iterations, device):
    import silvic

    params = {**PARAMS, "num_iterations": iterations, "device_type": device}
    start = time.perf_counter()
    dtrain = silvic.Dataset(train["X"], train["y"], group=train["group"],
                            categorical_feature=[int(i) for i in train["categorical"]])
    booster = silvic.train(params, dtrain)
    return time.perf_counter() - start, booster.model_to_string()


def main():
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--data", help="prefix of the .train.npz/.valid.npz pair gen_ltr.py wrote")
    source.add_argument("--rows", type=int, help="generate this many rows in memory with gen_ltr.py instead")
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--engines", default="lightgbm,lightgbm-quantized,silvic-cpu,silvic-metal")
    args = parser.parse_args()

    train, valid = load(args.data) if args.data else gen_ltr.generate(args.rows)
    rows, cols = train["X"].shape
    print(f"{rows} train rows, {cols} features, {len(train['group'])} queries, {args.iterations} iterations")
    results = []
    for engine in args.engines.split(","):
        if engine == "lightgbm":
            seconds, model = run_lightgbm(train, valid, args.iterations, quantized=False)
        elif engine == "lightgbm-quantized":
            seconds, model = run_lightgbm(train, valid, args.iterations, quantized=True)
        elif engine.startswith("silvic-"):
            seconds, model = run_silvic(train, valid, args.iterations, engine.removeprefix("silvic-"))
        else:
            raise SystemExit(f"unknown engine {engine}")
        metrics = ndcg(valid, model)
        row = {"engine": engine, "seconds": round(seconds, 2), **{k: round(v, 5) for k, v in metrics.items()}}
        print(json.dumps(row), flush=True)
        results.append(row)


if __name__ == "__main__":
    main()
