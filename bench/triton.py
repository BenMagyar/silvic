"""Latency of a tritonserver scoring 3000-row x 136-feature FP32 requests with `bench/data/serve_ltr.txt` (the
2000-tree lambdarank model `serve.py` trains), through the silvic backend or Triton's FIL backend.

    tests/triton/run.sh bench
    python bench/triton.py --url localhost:8000 --model-file ~/.cache/silvic-triton/models/ltr/1/model.txt

The server runs in explicit model-control mode with a repository holding the model as `<--model>/1/model.txt`
(`tests/triton/make_repository.py` writes one); the bench reloads `--model` with a config of its own for each row:
`max_batch_size` 4096 without dynamic batching (each request is scored alone), FP32 in, FP64 out for silvic and FP32
out for FIL (which computes in f32), and `instance_group [{ kind: KIND_CPU count: <instances> }]`. Each row sends
`--requests` requests per client over gRPC from `concurrency` clients at once, each waiting for its response before
sending the next, after `--warmup` requests per client. `--backend fil` needs a server with Triton's FIL backend,
such as the `deploy/triton/Dockerfile` image (NGC tritonserver, which ships it). It reports:

- client latency p50 and p99 (`InferenceServerClient.infer` round trips, request encoding included);
- server compute: the mean `compute_infer` time per request from Triton's statistics (the backend's
  `ModelInstanceExecute` scoring, without queueing or tensor copies), and the mean queue time;
- throughput: rows scored per second over the row.

silvic's outputs must equal `lightgbm.Booster.predict` bit for bit, or the script exits non-zero; FIL's max absolute
difference from it is reported. On the colima VM (4 vCPUs) these are not production numbers.
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import gen_ltr  # noqa: E402

ROWS = 3000
FEATURES = 136
SCORING_SEED = 11


def scoring_batches(count):
    """`count` batches of 3000 rows, from rows generated with another seed than the training rows (`serve.py`)."""
    # `generate` keeps 10% of the queries back as its validation split.
    rows, _ = gen_ltr.generate(2 * ROWS * count, seed=SCORING_SEED)
    X = rows["X"]
    return [np.ascontiguousarray(X[i * ROWS : (i + 1) * ROWS]) for i in range(count)]


def config(backend, instances, num_threads):
    output_type = "TYPE_FP64" if backend == "silvic" else "TYPE_FP32"
    parameters = {"num_threads": num_threads} if backend == "silvic" else {"model_type": "lightgbm",
                                                                           "output_class": "false"}
    return {
        "backend": backend,
        "max_batch_size": 4096,
        "input": [{"name": "input__0", "data_type": "TYPE_FP32", "dims": [FEATURES]}],
        "output": [{"name": "output__0", "data_type": output_type, "dims": [1]}],
        "instance_group": [{"kind": "KIND_CPU", "count": instances}],
        "parameters": {key: {"string_value": str(value)} for key, value in parameters.items()},
    }


def compute_stats(http, model):
    (entry,) = http.get_inference_statistics(model)["model_stats"]
    stats = entry["inference_stats"]
    return {key: (int(stats[key]["count"]), int(stats[key]["ns"])) for key in ["success", "compute_infer", "queue"]}


def run_row(args, http, batches, want, concurrency, instances, num_threads):
    import tritonclient.grpc as grpcclient

    http.unload_model(args.model)
    http.load_model(args.model, config=json.dumps(config(args.backend, instances, num_threads)))

    def client_loop(worker, count, latencies):
        connection = grpcclient.InferenceServerClient(args.grpc_url)
        outputs = [grpcclient.InferRequestedOutput("output__0")]
        worst = 0.0
        for index in range(count):
            batch = (worker * 7 + index) % len(batches)
            tensor = grpcclient.InferInput("input__0", [ROWS, FEATURES], "FP32")
            tensor.set_data_from_numpy(batches[batch])
            start = time.perf_counter()
            result = connection.infer(args.model, [tensor], outputs=outputs)
            latencies.append(time.perf_counter() - start)
            got = result.as_numpy("output__0").reshape(-1)
            if args.backend == "silvic" and not np.array_equal(got.view(np.uint64), want[batch].view(np.uint64)):
                raise SystemExit(f"silvic differs from LightGBM on batch {batch}")
            worst = max(worst, float(np.max(np.abs(got - want[batch]))))
        return worst

    with ThreadPoolExecutor(concurrency) as pool:
        list(pool.map(lambda worker: client_loop(worker, args.warmup, []), range(concurrency)))
    before = compute_stats(http, args.model)
    per_client = [[] for _ in range(concurrency)]
    start = time.perf_counter()
    with ThreadPoolExecutor(concurrency) as pool:
        worst = max(pool.map(lambda w: client_loop(w, args.requests, per_client[w]), range(concurrency)))
    elapsed = time.perf_counter() - start
    after = compute_stats(http, args.model)
    latencies = np.array([latency for client in per_client for latency in client]) * 1e3
    delta = {key: (after[key][0] - before[key][0], after[key][1] - before[key][1]) for key in after}
    return {
        "backend": args.backend,
        "concurrency": concurrency,
        "instances": instances,
        "num_threads": num_threads if args.backend == "silvic" else "-",
        "p50_ms": float(np.percentile(latencies, 50)),
        "p99_ms": float(np.percentile(latencies, 99)),
        "compute_ms": delta["compute_infer"][1] / max(delta["compute_infer"][0], 1) / 1e6,
        "queue_ms": delta["queue"][1] / max(delta["queue"][0], 1) / 1e6,
        "rows_per_s": len(latencies) * ROWS / elapsed,
        "max_abs_diff": worst,
        "requests": int(delta["success"][0]),
    }


def parse_rows(text):
    """`concurrency:instances:num_threads,...`."""
    return [tuple(int(part) for part in row.split(":")) for row in text.split(",")]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", default=os.environ.get("TRITON_URL", "localhost:8000"), help="HTTP host:port")
    parser.add_argument("--grpc-url", default=os.environ.get("TRITON_GRPC_URL"), help="default: --url host, 8001")
    parser.add_argument("--backend", default="silvic", choices=["silvic", "fil"])
    parser.add_argument("--model", default="ltr", help="repository model whose 1/model.txt is serve_ltr.txt")
    parser.add_argument("--model-file", type=Path,
                        default=Path(os.environ.get("TRITON_REPOSITORY", "~/.cache/silvic-triton/models")).expanduser()
                        / "ltr" / "1" / "model.txt", help="the model, for LightGBM's reference predictions")
    parser.add_argument("--rows", type=parse_rows, default=parse_rows("1:1:1,4:4:1"),
                        help="concurrency:instances:num_threads per row (num_threads is silvic's; default 1:1:1,4:4:1)")
    parser.add_argument("--requests", type=int, default=200, help="timed requests per client")
    parser.add_argument("--warmup", type=int, default=10, help="untimed requests per client first")
    parser.add_argument("--json", type=Path, help="also write the rows here")
    args = parser.parse_args()
    args.grpc_url = args.grpc_url or f"{args.url.rsplit(':', 1)[0]}:8001"

    import lightgbm as lgb
    import tritonclient.http as httpclient

    http = httpclient.InferenceServerClient(args.url)
    metadata = http.get_server_metadata()
    booster = lgb.Booster(model_file=str(args.model_file))
    batches = scoring_batches(10)
    want = [booster.predict(batch, num_threads=1) for batch in batches]
    print(f"tritonserver {metadata['version']}, {args.backend} backend, {os.cpu_count()} client CPUs; "
          f"{ROWS}-row FP32 requests over gRPC", flush=True)
    print("| backend | concurrency | instances | num_threads | p50 ms | p99 ms | server compute ms | queue ms "
          "| rows/s | max abs diff |")
    print("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    results = []
    for concurrency, instances, num_threads in args.rows:
        row = run_row(args, http, batches, want, concurrency, instances, num_threads)
        results.append(row)
        print(f"| {row['backend']} | {concurrency} | {instances} | {row['num_threads']} | {row['p50_ms']:.2f} "
              f"| {row['p99_ms']:.2f} | {row['compute_ms']:.2f} | {row['queue_ms']:.2f} | {row['rows_per_s']:,.0f} "
              f"| {row['max_abs_diff']:.3g} |", flush=True)
    if args.json:
        args.json.write_text(json.dumps({"server": metadata["version"], "rows": results}, indent=2))


if __name__ == "__main__":
    main()
