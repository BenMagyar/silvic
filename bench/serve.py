"""Single-thread latency and throughput of scoring a 2000-tree lambdarank model, 3000 rows per call.

    python bench/serve.py
    python bench/serve.py --engines lightgbm,lleaves,silvic --repeats 500

The first run trains the model with LightGBM (`PARAMS` below: lambdarank, 53 leaves, depth 11) on rows `gen_ltr.py`
generates in memory, and writes it to `--model`; later runs load it. Scoring batches are other `gen_ltr.py` rows
(float32, C order: 136 features, NaN in the float columns, 5 categoricals of at most 20 levels).

Every engine scores on one thread, and its time covers what a caller pays per call, input conversion included:

- `lightgbm`: `lightgbm.Booster.predict(X, num_threads=1)`.
- `lleaves`, `lleaves-noinline`: `lleaves.Model.predict(X, n_jobs=1)` after `compile()` (`finline=False` for the
  second), with the compiled binary cached next to the model.
- `treelite`: tl2cgen 1.x C code from treelite 4.x (the CPU engine Triton's FIL backend wraps), compiled with clang,
  `Predictor(nthread=1).predict(DMatrix(X))`; `treelite-quantize` compiles with `quantize=1`.
- `silvic-<kernel>`: `silvic.serve.Predictor(model, kernel=<kernel>).predict(X, num_threads=1)` for every kernel the
  machine runs (`portable` and the CPU's SIMD kernel).

Each engine's predictions on every batch are compared with LightGBM's (max abs difference); silvic's must be equal
bit for bit, or the script exits non-zero.
"""

import argparse
import hashlib
import importlib.metadata
import json
import os
import sys
import time
from pathlib import Path

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
    "seed": 123,
    "verbose": -1,
}
DATA = Path(__file__).parent / "data"
SCORING_SEED = 11


def train_model(path, rows, trees):
    train, _ = gen_ltr.generate(rows)
    params = {**PARAMS, "num_iterations": trees, "num_threads": 0}
    dataset = lgb.Dataset(train["X"], train["y"], group=train["group"],
                          categorical_feature=[int(i) for i in train["categorical"]])
    start = time.perf_counter()
    booster = lgb.train(params, dataset)
    print(f"trained {booster.num_trees()} trees on {rows} rows in {time.perf_counter() - start:.0f} s", flush=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(path))


def scoring_batches(batch, count):
    """`count` batches of `batch` rows, cut from rows generated with another seed than the training rows."""
    rows, _ = gen_ltr.generate(batch * count + batch, seed=SCORING_SEED)
    X = rows["X"]
    return [np.ascontiguousarray(X[i * batch:(i + 1) * batch]) for i in range(count)]


def cache_path(model_path, tag):
    digest = hashlib.sha256(model_path.read_bytes()).hexdigest()[:12]
    return model_path.with_name(f"{model_path.stem}.{digest}.{tag}")


def lightgbm_engine(model_path):
    booster = lgb.Booster(model_file=str(model_path))
    return lambda X: booster.predict(X, num_threads=1)


def lleaves_engine(model_path, inline):
    import lleaves

    model = lleaves.Model(model_file=str(model_path))
    tag = f"lleaves-{importlib.metadata.version('lleaves')}-{'inline' if inline else 'noinline'}.bin"
    start = time.perf_counter()
    model.compile(cache=str(cache_path(model_path, tag)), finline=inline)
    print(f"  lleaves compile/load: {time.perf_counter() - start:.1f} s", flush=True)
    return lambda X: model.predict(X, n_jobs=1)


def treelite_engine(model_path, quantize):
    import tl2cgen
    import treelite

    library = cache_path(model_path, f"tl2cgen-{tl2cgen.__version__}-q{int(quantize)}.dylib")
    if not library.exists():
        start = time.perf_counter()
        model = treelite.frontend.load_lightgbm_model(str(model_path))
        params = {"parallel_comp": os.cpu_count() * 4, "quantize": int(quantize)}
        tl2cgen.export_lib(model, toolchain="clang", libpath=str(library), params=params, nthread=os.cpu_count())
        print(f"  tl2cgen compile: {time.perf_counter() - start:.0f} s", flush=True)
    predictor = tl2cgen.Predictor(str(library), nthread=1)
    return lambda X: predictor.predict(tl2cgen.DMatrix(X)).reshape(-1)


def silvic_engine(model_path, kernel):
    from silvic import serve

    predictor = serve.Predictor(str(model_path), kernel=kernel)
    return lambda X: predictor.predict(X, num_threads=1)


def engine_for(name, model_path):
    if name == "lightgbm":
        return lightgbm_engine(model_path)
    if name in ("lleaves", "lleaves-noinline"):
        return lleaves_engine(model_path, inline=name == "lleaves")
    if name in ("treelite", "treelite-quantize"):
        return treelite_engine(model_path, quantize=name == "treelite-quantize")
    if name.startswith("silvic-"):
        return silvic_engine(model_path, name.removeprefix("silvic-"))
    raise SystemExit(f"unknown engine {name}")


def default_engines():
    engines = ["lightgbm", "lleaves", "lleaves-noinline", "treelite", "treelite-quantize"]
    try:
        from silvic import serve
    except ImportError:
        return engines
    return engines + [f"silvic-{kernel}" for kernel in serve.available_kernels()]


def time_engine(predict, batches, repeats, warmup):
    for i in range(warmup):
        predict(batches[i % len(batches)])
    latencies = np.empty(repeats)
    for i in range(repeats):
        X = batches[i % len(batches)]
        start = time.perf_counter_ns()
        predict(X)
        latencies[i] = (time.perf_counter_ns() - start) / 1e6
    return latencies


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=DATA / "serve_ltr.txt",
                        help="LightGBM model text; trained and written here when absent")
    parser.add_argument("--train-rows", type=int, default=1_000_000)
    parser.add_argument("--trees", type=int, default=2000)
    parser.add_argument("--batch", type=int, default=3000)
    parser.add_argument("--batches", type=int, default=16, help="distinct batches cycled through")
    parser.add_argument("--repeats", type=int, default=300)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--engines", help="comma-separated; default: every engine installed")
    parser.add_argument("--json", type=Path, help="also write the results here")
    args = parser.parse_args()

    if not args.model.exists():
        train_model(args.model, args.train_rows, args.trees)
    batches = scoring_batches(args.batch, args.batches)
    reference_booster = lgb.Booster(model_file=str(args.model))
    reference = [reference_booster.predict(X, num_threads=0) for X in batches]
    print(f"{reference_booster.num_trees()} trees, {batches[0].shape[1]} features, {args.batch} rows per call, "
          f"{args.repeats} calls, lightgbm {lgb.__version__}", flush=True)

    engines = args.engines.split(",") if args.engines else default_engines()
    results = []
    failed = False
    for name in engines:
        print(f"{name}:", flush=True)
        try:
            predict = engine_for(name, args.model)
        except Exception as error:  # noqa: BLE001 - a missing or broken competitor skips its row, silvic's fails the run
            if name.startswith("silvic-"):
                raise
            print(f"  skipped: {type(error).__name__}: {error}", flush=True)
            continue
        diff = max(float(np.max(np.abs(predict(X) - expected))) for X, expected in zip(batches, reference))
        exact = all(np.array_equal(predict(X), expected) for X, expected in zip(batches, reference))
        latencies = time_engine(predict, batches, args.repeats, args.warmup)
        p50, p99 = np.percentile(latencies, [50, 99])
        row = {
            "engine": name,
            "p50_ms": round(float(p50), 3),
            "p99_ms": round(float(p99), 3),
            "rows_per_s": round(args.batch / (float(latencies.mean()) / 1e3)),
            "max_abs_diff": diff,
            "exact": exact,
        }
        print("  " + json.dumps(row), flush=True)
        results.append(row)
        if name.startswith("silvic-") and not exact:
            print(f"  {name} differs from LightGBM", file=sys.stderr)
            failed = True

    print()
    print("| engine | p50 ms | p99 ms | rows/s | max abs diff vs LightGBM |")
    print("| --- | ---: | ---: | ---: | ---: |")
    for row in results:
        print(f"| {row['engine']} | {row['p50_ms']:.2f} | {row['p99_ms']:.2f} | {row['rows_per_s']:,} | "
              f"{row['max_abs_diff']:.3g} |")
    if args.json:
        args.json.write_text(json.dumps(results, indent=1))
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
