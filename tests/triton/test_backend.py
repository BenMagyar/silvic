"""End to end against a live tritonserver with the silvic backend: predictions equal `lightgbm.Booster.predict`'s
bit for bit over HTTP and gRPC, bad requests fail alone, the repository API loads and unloads, and the statistics
endpoint counts what was sent.

Skipped unless `TRITON_URL` (HTTP, `host:port`) is set. The server serves the repository
`make_repository.py` writes, with `--model-control-mode=explicit --load-model=*`; `TRITON_REPOSITORY` is that
repository's path as this process sees it (the reference boosters are read from it), `TRITON_GRPC_URL` the gRPC
endpoint (`TRITON_URL`'s host, port 8001, by default). `tests/triton/run.sh` does all of it in containers:

    tests/triton/run.sh
    TRITON_URL=localhost:8000 TRITON_REPOSITORY=~/.cache/silvic-triton/models pytest tests/triton
"""

import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(not os.environ.get("TRITON_URL"), reason="TRITON_URL is not set")

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parents[1] / "bench")]

HTTP_URL = os.environ.get("TRITON_URL", "")
GRPC_URL = os.environ.get("TRITON_GRPC_URL") or f"{HTTP_URL.rsplit(':', 1)[0]}:8001"
REPOSITORY = Path(os.environ.get("TRITON_REPOSITORY", "~/.cache/silvic-triton/models")).expanduser()
LTR_ROWS_SEED = 11


@pytest.fixture(scope="module")
def clients():
    import tritonclient.grpc as grpcclient
    import tritonclient.http as httpclient

    return {"http": httpclient, "grpc": grpcclient}


def client(clients, protocol):
    module = clients[protocol]
    return module.InferenceServerClient(HTTP_URL if protocol == "http" else GRPC_URL)


def infer(clients, protocol, model, X, input_name="input__0", connection=None):
    module = clients[protocol]
    connection = connection or client(clients, protocol)
    datatype = {np.float32: "FP32", np.float64: "FP64"}[X.dtype.type]
    tensor = module.InferInput(input_name, list(X.shape), datatype)
    tensor.set_data_from_numpy(X)
    result = connection.infer(model, [tensor], outputs=[module.InferRequestedOutput("output__0")])
    return result.as_numpy("output__0")


def booster(model, file_name="model.txt"):
    import lightgbm as lgb

    return lgb.Booster(model_file=str(REPOSITORY / model / "1" / file_name))


def ltr_rows(count, seed=LTR_ROWS_SEED):
    import gen_ltr

    # `generate` keeps 10% of the queries back as its validation split.
    rows, _ = gen_ltr.generate(2 * count + 200, seed=seed)
    return np.ascontiguousarray(rows["X"][:count])


def wide_doubles(X):
    """`X` in f64 with the float columns moved off f32-representable values, so FP64 input is tested as FP64."""
    X = X.astype(np.float64)
    X[:, 90:131] += np.random.default_rng(3).normal(scale=1e-9, size=X[:, 90:131].shape)
    return X


def assert_bitwise(got, want, case):
    got = np.asarray(got).reshape(-1)
    want = np.asarray(want).astype(got.dtype).reshape(-1)
    assert got.shape == want.shape, f"{case}: shape {got.shape} != {want.shape}"
    bits = np.uint64 if got.dtype == np.float64 else np.uint32
    same = np.array_equal(got.view(bits), want.view(bits))
    assert same, f"{case}: {np.count_nonzero(got.view(bits) != want.view(bits))} differ, max |d| {np.nanmax(abs(got - want))}"


def stats(clients, model):
    (entry,) = client(clients, "http").get_inference_statistics(model)["model_stats"]
    return entry


def reload(clients, model, config=None):
    connection = client(clients, "http")
    connection.unload_model(model)
    wait_unloaded(connection, model)
    connection.load_model(model, config=config)


def wait_unloaded(connection, model):
    for _ in range(100):
        if not connection.is_model_ready(model):
            return
        time.sleep(0.05)
    raise AssertionError(f"{model} still ready after unload")


def test_server_is_live_with_every_model(clients):
    connection = client(clients, "http")
    assert connection.is_server_live() and connection.is_server_ready()
    metadata = connection.get_server_metadata()
    print(f"tritonserver {metadata['version']}")
    for model in ["ltr", "ltr_nobatch", "ltr_auto", "ltr_fil", "ltr_optional", "binary", "binary_raw"]:
        assert connection.is_model_ready(model), model
    config = connection.get_model_config("ltr_auto")
    assert [(t["name"], t["data_type"], t["dims"]) for t in config["input"]] == [("input__0", "TYPE_FP32", [136])]
    assert [(t["name"], t["data_type"], t["dims"]) for t in config["output"]] == [("output__0", "TYPE_FP64", [1])]
    assert connection.get_model_config("ltr_fil")["instance_group"][0]["kind"] == "KIND_CPU"


@pytest.mark.parametrize("protocol", ["http", "grpc"])
@pytest.mark.parametrize(
    "model, dtype, file_name",
    [
        ("ltr", np.float32, "model.txt"),
        ("ltr_nobatch", np.float64, "model.txt"),
        ("ltr_auto", np.float32, "booster.txt"),
        ("ltr_fil", np.float32, "model.txt"),
    ],
)
def test_lightgbm_model_matches_booster_predict(clients, protocol, model, dtype, file_name):
    X = ltr_rows(3000)
    X = X if dtype == np.float32 else wide_doubles(X)
    want = booster(model, file_name).predict(X)
    got = infer(clients, protocol, model, X)
    batching = model != "ltr_nobatch"
    assert got.shape == ((3000, 1) if batching else (3000,)), f"{model}: output shape"
    assert got.dtype == (np.float32 if model == "ltr_fil" else np.float64), f"{model}: output type"
    assert_bitwise(got, want, f"{model} {protocol}")


@pytest.mark.parametrize("protocol", ["http", "grpc"])
def test_silvic_binary_model_raw_and_transformed(clients, protocol):
    from make_repository import binary_rows

    X = binary_rows(2000, seed=5)
    reference = booster("binary")
    X32 = X.astype(np.float32)
    assert_bitwise(infer(clients, protocol, "binary", X32), reference.predict(X32), f"binary {protocol}")
    raw = reference.predict(X, raw_score=True, num_iteration=20)
    assert_bitwise(infer(clients, protocol, "binary_raw", X), raw, f"binary_raw {protocol}")


def test_one_row_requests_batch_together(clients):
    """Concurrent 1-row gRPC requests: dynamic batching delivers several per execution, each answered exactly."""
    X = ltr_rows(256)
    want = booster("ltr").predict(X)
    before = stats(clients, "ltr")
    local = threading.local()

    def one(row):
        if not hasattr(local, "connection"):
            local.connection = client(clients, "grpc")
        return infer(clients, "grpc", "ltr", X[row : row + 1], connection=local.connection)

    with ThreadPoolExecutor(32) as pool:
        got = np.concatenate([out.reshape(-1) for out in pool.map(one, range(len(X)))])
    assert_bitwise(got, want, "1-row requests")
    after = stats(clients, "ltr")
    requests = int(after["inference_stats"]["success"]["count"]) - int(before["inference_stats"]["success"]["count"])
    executions = int(after["execution_count"]) - int(before["execution_count"])
    rows = int(after["inference_count"]) - int(before["inference_count"])
    print(f"256 1-row requests in {executions} executions")
    assert (requests, rows) == (256, 256)
    assert executions < requests, "dynamic batching put several requests in one execution"


@pytest.mark.parametrize("protocol", ["http", "grpc"])
def test_concurrent_clients_mixed_sizes(clients, protocol):
    from make_repository import binary_rows

    ltr_X = ltr_rows(4000, seed=21)
    ltr_want = booster("ltr").predict(ltr_X)
    binary_X = binary_rows(4000, seed=6).astype(np.float32)
    binary_want = booster("binary").predict(binary_X)
    sizes = [1, 7, 300, 64, 1, 1000, 2, 255, 256, 3]
    jobs = []
    for worker in range(8):
        for index, size in enumerate(sizes):
            start = (worker * 389 + index * 97) % (4000 - size)
            jobs.append((worker, start, size))

    def run(worker):
        connection = client(clients, protocol)
        for job_worker, start, size in jobs:
            if job_worker != worker:
                continue
            for model, X, want in [("ltr", ltr_X, ltr_want), ("binary", binary_X, binary_want)]:
                got = infer(clients, protocol, model, X[start : start + size], connection=connection)
                assert_bitwise(got, want[start : start + size], f"{model} {protocol} rows {start}+{size}")

    before = stats(clients, "ltr")
    with ThreadPoolExecutor(8) as pool:
        list(pool.map(run, range(8)))
    after = stats(clients, "ltr")
    executions = int(after["execution_count"]) - int(before["execution_count"])
    print(f"{len(jobs)} concurrent ltr requests in {executions} executions")
    assert executions < len(jobs)


@pytest.mark.parametrize("protocol", ["http", "grpc"])
def test_bad_requests_fail_alone(clients, protocol):
    from tritonclient.utils import InferenceServerException

    connection = client(clients, protocol)
    X = ltr_rows(16)
    bad = [
        ("ltr", X.astype(np.float64), "input__0", "FP64"),
        ("ltr", np.ascontiguousarray(X[:, :135]), "input__0", "135"),
        ("ltr", X, "features", "unknown input"),
        ("ltr_nobatch", X, "input__0", "FP32"),
    ]
    for model, rows, name, case in bad:
        with pytest.raises(InferenceServerException):
            infer(clients, protocol, model, rows, input_name=name, connection=connection)
        assert connection.is_server_live() and connection.is_model_ready(model), f"{case}: healthy"
    assert_bitwise(infer(clients, protocol, "ltr", X, connection=connection), booster("ltr").predict(X), "after")


def missing_input_request(clients, protocol, connection):
    """A request to `ltr_optional` without its (optional) input: only the backend can reject it."""
    module = clients[protocol]
    outputs = [module.InferRequestedOutput("output__0")]
    return connection.infer("ltr_optional", [], outputs=outputs)


def test_a_backend_error_spares_its_batch(clients):
    """A request the backend rejects, batched with good ones, gets the error; they get their scores."""
    from tritonclient.utils import InferenceServerException

    reload(clients, "ltr_optional")
    X = ltr_rows(40)
    want = booster("ltr_optional").predict(X)
    barrier = threading.Barrier(8)

    def run(index):
        connection = client(clients, "grpc")
        barrier.wait()
        if index % 4 == 3:
            with pytest.raises(InferenceServerException, match="input__0"):
                missing_input_request(clients, "grpc", connection)
            return None
        return infer(clients, "grpc", "ltr_optional", X[index * 5 : index * 5 + 5], connection=connection)

    with ThreadPoolExecutor(8) as pool:
        results = list(pool.map(run, range(8)))
    for index, got in enumerate(results):
        if got is not None:
            assert_bitwise(got, want[index * 5 : index * 5 + 5], f"request {index}")
    entry = stats(clients, "ltr_optional")
    inference = entry["inference_stats"]
    print(f"8 requests in {entry['execution_count']} executions")
    assert int(inference["success"]["count"]) == 6
    assert int(inference["fail"]["count"]) == 2
    assert int(entry["inference_count"]) == 30
    assert int(entry["execution_count"]) < 8


def test_load_unload_reload(clients):
    from tritonclient.utils import InferenceServerException

    connection = client(clients, "http")
    X = ltr_rows(100)
    want = booster("ltr_auto", "booster.txt").predict(X)
    connection.unload_model("ltr_auto")
    wait_unloaded(connection, "ltr_auto")
    with pytest.raises(InferenceServerException):
        infer(clients, "http", "ltr_auto", X, connection=connection)
    connection.load_model("ltr_auto")
    assert connection.is_model_ready("ltr_auto")
    assert_bitwise(infer(clients, "http", "ltr_auto", X, connection=connection), want, "reloaded")

    base = {"backend": "silvic", "max_batch_size": 4096, "parameters": {"model_file": {"string_value": "booster.txt"}}}
    failures = [
        ({"parameters": {**base["parameters"], "num_trees": {"string_value": "5"}}}, "unknown parameter `num_trees`"),
        # On a host without GPUs Triton refuses KIND_GPU before the backend sees it; with GPUs, the backend does.
        ({"instance_group": [{"kind": "KIND_GPU", "count": 1}]}, "KIND_GPU"),
        ({"parameters": {"model_file": {"string_value": "missing.txt"}}}, "missing.txt"),
        (
            {"input": [{"name": "input__0", "data_type": "TYPE_FP32", "dims": [135]}]},
            "the model reads 136 features",
        ),
    ]
    for change, message in failures:
        with pytest.raises(InferenceServerException) as failure:
            connection.load_model("ltr_auto", config=json.dumps({**base, **change}))
        assert message in str(failure.value), f"{message}: {failure.value}"
        assert connection.is_server_live(), message
    connection.load_model("ltr_auto", config=json.dumps({**base, "parameters": {
        **base["parameters"], "num_iteration": {"string_value": "100"}}}))
    assert_bitwise(
        infer(clients, "http", "ltr_auto", X, connection=connection),
        booster("ltr_auto", "booster.txt").predict(X, num_iteration=100),
        "num_iteration 100",
    )
    # A load without a config does not replace a loaded override config, so unload first.
    reload(clients, "ltr_auto")
    assert_bitwise(infer(clients, "http", "ltr_auto", X, connection=connection), want, "reloaded again")


def test_statistics_count_requests_and_time(clients):
    reload(clients, "ltr_nobatch")
    connection = client(clients, "grpc")
    sizes = [1, 10, 100, 1000, 3000]
    start = time.perf_counter()
    for size in sizes:
        infer(clients, "grpc", "ltr_nobatch", wide_doubles(ltr_rows(size)), connection=connection)
    elapsed_ns = (time.perf_counter() - start) * 1e9
    entry = stats(clients, "ltr_nobatch")
    inference = entry["inference_stats"]
    assert int(inference["success"]["count"]) == len(sizes)
    assert int(inference["fail"]["count"]) == 0
    # Without batching Triton counts one inference per request.
    assert int(entry["inference_count"]) == len(sizes)
    assert int(entry["execution_count"]) == len(sizes)
    compute = int(inference["compute_infer"]["ns"])
    total = int(inference["success"]["ns"])
    print(f"compute_infer {compute / 1e6:.2f} ms, success {total / 1e6:.2f} ms, client {elapsed_ns / 1e6:.2f} ms")
    assert 0 < compute <= total < elapsed_ns, "steady-clock timestamps nest inside the client's wall time"
    for phase in ["queue", "compute_input", "compute_output"]:
        assert 0 <= int(inference[phase]["ns"]) < elapsed_ns, phase
    (batch,) = entry["batch_stats"]
    assert int(batch["batch_size"]) == 1 and int(batch["compute_infer"]["count"]) == len(sizes)
