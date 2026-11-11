# /// script
# requires-python = ">=3.10"
# dependencies = ["lightgbm==4.7.0", "numpy>=1.26"]
# ///
"""Records LightGBM 4.7's validation NDCG@10 on a synthetic lambdarank set for `tests/train_cpu.rs`.

The data comes from the counter-based hash silvic uses for random draws (`quantize::uniform`), so the Rust test
regenerates the same rows bit for bit without a data fixture. Labels use only exact IEEE operations (no
transcendental functions), evaluated in the same order as `train_cpu.rs`.

    .venv/bin/python crates/silvic-core/tests/fixtures/lambdarank_parity.py            # rewrite the fixture
    .venv/bin/python crates/silvic-core/tests/fixtures/lambdarank_parity.py --bench --rows 2000000 --features 140
"""

import argparse
import json
import pathlib
import time

import lightgbm as lgb
import numpy as np

SEED = 20261107
VALID_ROW_OFFSET = 10_000_000
NOISE_STREAM = 999
NAN_STREAM = 1000
CATEGORICAL = 7
CAT_EFFECT = np.array([0.0, 0.8, -0.5, 0.3, 0.0, -0.8, 0.6, 0.1])
FIXTURE = pathlib.Path(__file__).with_name("lambdarank_parity.json")


def mix32(x):
    x = x ^ (x >> np.uint32(16))
    x = x * np.uint32(0x7FEB352D)
    x = x ^ (x >> np.uint32(15))
    x = x * np.uint32(0x846CA68B)
    return x ^ (x >> np.uint32(16))


def uniform(stream, rows):
    """`quantize::uniform(SEED, stream, 0, row)` as float64 (exact float32 values)."""
    lo = (rows & np.uint64(0xFFFFFFFF)).astype(np.uint32)
    hi = (rows >> np.uint64(32)).astype(np.uint32)
    row_hash = mix32(lo ^ mix32(hi))
    h = mix32(np.uint32(SEED) ^ mix32(np.uint32(stream) ^ mix32(np.uint32(0) ^ row_hash)))
    return (h >> np.uint32(8)).astype(np.float64) * (1.0 / 16777216.0)


def generate(first_row, num_rows, num_features, num_queries):
    rows = np.arange(first_row, first_row + num_rows, dtype=np.uint64)
    x = np.empty((num_rows, num_features), dtype=np.float64)
    for j in range(num_features):
        col = uniform(j, rows)
        if j == CATEGORICAL:
            col = np.floor(col * 8.0)
        elif j % 10 == 3:
            col = np.where(uniform(NAN_STREAM + j, rows) < 0.1, np.nan, col)
        x[:, j] = col
    x3 = np.where(np.isnan(x[:, 3]), 0.5, x[:, 3])
    s = 3.0 * x[:, 0]
    s = s + 2.0 * x[:, 1] * x[:, 2]
    s = s - 1.5 * np.abs(x[:, 4] - 0.5)
    s = s + x3
    s = s + CAT_EFFECT[x[:, CATEGORICAL].astype(np.int64)]
    s = s + 0.6 * x[:, 10] * x[:, 11]
    s = s + uniform(NOISE_STREAM, rows)
    label = np.clip(np.floor(s - 1.0), 0.0, 4.0)
    group = np.full(num_queries, num_rows // num_queries, dtype=np.int64)
    group[-1] += num_rows - group.sum()
    return x.astype(np.float32), label.astype(np.float32), group


def params(quantized, iterations, threads):
    return {
        "objective": "lambdarank",
        "metric": "ndcg",
        "eval_at": [10],
        "learning_rate": 0.1,
        "num_leaves": 31,
        "min_data_in_leaf": 20,
        "num_iterations": iterations,
        "max_bin": 255,
        "num_threads": threads,
        "device_type": "cpu",
        "verbose": -1,
        "use_quantized_grad": quantized,
        "num_grad_quant_bins": 4,
        "quant_train_renew_leaf": True,
        "stochastic_rounding": True,
    }


def run(args, quantized):
    x, y, group = generate(0, args.rows, args.features, args.queries)
    vx, vy, vgroup = generate(VALID_ROW_OFFSET, args.valid_rows, args.features, args.valid_queries)
    p = params(quantized, args.iterations, args.threads)
    train = lgb.Dataset(x, y, group=group, categorical_feature=[CATEGORICAL], params={"max_bin": 255}).construct()
    valid = lgb.Dataset(vx, vy, group=vgroup, reference=train, categorical_feature=[CATEGORICAL])
    record = {}
    start = time.perf_counter()
    lgb.train(p, train, valid_sets=[valid], valid_names=["valid"], callbacks=[lgb.record_evaluation(record)])
    seconds = time.perf_counter() - start
    return record["valid"]["ndcg@10"][-1], seconds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=100_000)
    parser.add_argument("--features", type=int, default=50)
    parser.add_argument("--queries", type=int, default=1000)
    parser.add_argument("--valid-rows", type=int, default=20_000)
    parser.add_argument("--valid-queries", type=int, default=200)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--threads", type=int, default=14)
    parser.add_argument("--bench", action="store_true", help="print timings instead of writing the fixture")
    args = parser.parse_args()
    results = {mode: run(args, mode == "quantized") for mode in ("quantized", "float")}
    if args.bench:
        for mode, (ndcg, seconds) in results.items():
            print(f"lightgbm {lgb.__version__} {mode}: {seconds:.2f}s train, valid ndcg@10 {ndcg:.5f}")
        return
    fixture = {
        "lightgbm_version": lgb.__version__,
        "rows": args.rows,
        "features": args.features,
        "queries": args.queries,
        "valid_rows": args.valid_rows,
        "valid_queries": args.valid_queries,
        "iterations": args.iterations,
        "ndcg@10": {mode: ndcg for mode, (ndcg, _) in results.items()},
    }
    FIXTURE.write_text(json.dumps(fixture, indent=2) + "\n")
    print(json.dumps(fixture, indent=2))


if __name__ == "__main__":
    main()
