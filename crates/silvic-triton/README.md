# silvic-triton

A native [Triton Inference Server](https://github.com/triton-inference-server/server) backend, `silvic`
(`libtriton_silvic.so`), that serves LightGBM `model.txt` models (LightGBM 3.3.5 and 4.x, or silvic) with
[`silvic-serve`](../silvic-serve). FP64 outputs equal `lightgbm.Booster.predict`'s bit for bit; FP32 outputs are
their cast. It reads the FIL backend's LightGBM configs, so a FIL model directory serves with `backend: "fil"`
changed to `backend: "silvic"`.

It implements the backend C API of triton-inference-server/core (`tritonbackend.h`), declared by hand from r24.10,
and needs backend API 1.10 (Triton 22.09) or newer: every call it makes exists since 1.0 except
`TRITONBACKEND_BackendAttributeAddPreferredInstanceGroup`, added in 1.10. It has run on tritonserver 2.51.0
(NGC 24.10, backend API 1.19).

## Model repository

```
models/
  ranker/
    config.pbtxt
    1/
      model.txt
```

```protobuf
backend: "silvic"
max_batch_size: 4096
input [{ name: "input__0" data_type: TYPE_FP32 dims: [ 136 ] }]
output [{ name: "output__0" data_type: TYPE_FP64 dims: [ 1 ] }]
instance_group [{ kind: KIND_CPU count: 2 }]
dynamic_batching { max_queue_delay_microseconds: 500 }
parameters [
  { key: "raw_score" value: { string_value: "false" } },
  { key: "num_threads" value: { string_value: "1" } }
]
```

Each model version (`<model>/<version>/`) is loaded and compiled once, in `TRITONBACKEND_ModelInitialize`; its
instances share the compiled `Predictor`.

## I/O

| Config | Input dims | Request shape | Output dims | Response shape |
| --- | --- | --- | --- | --- |
| `max_batch_size > 0` | `[num_features]` | `[rows, num_features]` | `[1]` | `[rows, 1]` |
| `max_batch_size: 0` | `[-1, num_features]` | `[rows, num_features]` | `[-1]` | `[rows]` |

- One input, of any name, `TYPE_FP32` or `TYPE_FP64`, row-major. Its width must be the model's feature count.
  LightGBM reads FP32 as `Booster.predict` reads a float32 array, so FP32 input gives the same predictions too.
- One output, of any name: `TYPE_FP64` holds LightGBM's predictions exactly, `TYPE_FP32` their cast to f32 (as FIL
  clients expect).
- `KIND_CPU` instances; `KIND_AUTO` and `KIND_MODEL` run on the CPU. The backend tells Triton it prefers CPU
  instances (`TRITONBACKEND_GetBackendAttribute`), so a config without `instance_group` gets them on a host with
  GPUs too (not yet run on one). `KIND_GPU` is a load error.
- An input that is one contiguous CPU buffer of the configured type, suitably aligned, is scored where Triton holds
  it; otherwise its buffers are copied (asking Triton for CPU memory).
- silvic-serve scores rows 64 at a time, so a 1-row request alone costs a 64-row block. Requests of fewer than 256
  rows that dynamic batching delivers in one execution are copied into one matrix and scored together; larger ones
  are scored on their own. Every request gets its own response.
- A bad request (wrong type, width or shape, a missing input) gets an error response; the other requests of the
  execution, the instance and the server are unaffected.
- Triton's statistics (`/v2/models/<model>/stats`, and the metrics endpoint) get per-request and per-execution
  timestamps from the same steady clock Triton uses: `compute_input` is the input checks, `compute_infer` the copies
  and scoring, `compute_output` the responses.

## Parameters

Triton's string parameters (`parameters [{ key: "..." value: { string_value: "..." } }]`). A key that is neither
below nor one of FIL's tuning parameters fails the load, naming it.

| Parameter | Default | Meaning |
| --- | --- | --- |
| `model_file` | `model.txt` | The model file in the version directory |
| `raw_score` | `false` | `Booster.predict(raw_score=...)`: margins instead of the objective's transform (the sigmoid for `binary`) |
| `num_iteration` | `0` | `Booster.predict(num_iteration=...)`: score with the first N trees; 0 or less, every tree |
| `kernel` | `Kernel::detect` | `portable`, `neon`, `avx2` or `avx512`; a kernel the CPU does not run fails the load |
| `num_threads` | `1` | Threads per scored matrix: 1 scores on the instance's thread, N on N rayon tasks, 0 on every core |

For parallelism across requests, prefer more instances (`instance_group` `count`) to `num_threads`.

## FIL-compatible names

| FIL backend | silvic backend |
| --- | --- |
| `backend: "fil"` | `backend: "silvic"` |
| `input__0`, `output__0` | The same names when auto-completed; any names when configured |
| `model.txt`, the LightGBM model file name | The same default |
| `model_type: "lightgbm"` | Accepted; any other model type fails the load |
| `is_classifier` (formerly `output_class`) `false` | One score per row, as configured |
| `is_classifier: true` with `predict_proba: true` (binary) | The probability, one value per row as FIL returns for binary models; the model must be binary |
| `is_classifier: true` without `predict_proba` (class IDs) | A load error |
| `threshold`, `storage_type`, `algo`, `layout`, `chunk_size`, `threads_per_tree`, `blocks_per_sm`, `transfer_threshold`, `use_experimental_optimizations`, `xgboost_allow_unknown_field` | Accepted, no effect: they tune FIL's execution or its class-ID output |
| `instance_group [{ kind: KIND_AUTO }]` | CPU instances |
| FP32 output, computed in f32 | FP32 output, computed in f64 and cast |

## Auto-complete

With Triton's auto-complete on (the default; off with `--disable-auto-complete-config`), a config without `input`
or without `output` gets them from the model. With only

```protobuf
backend: "silvic"
max_batch_size: 4096
```

Triton serves `input__0` (`TYPE_FP32`, dims `[num_features]`) and `output__0` (`TYPE_FP64`, dims `[1]`); with
`max_batch_size: 0` (or none), dims `[-1, num_features]` and `[-1]`. A config that gives them is checked against the
model instead.

## Build and run

The production image is NVIDIA's tritonserver with the backend in `/opt/tritonserver/backends/silvic/`
([`deploy/triton/Dockerfile`](../../deploy/triton/Dockerfile)), built from the repository root:

```sh
docker build -f deploy/triton/Dockerfile -t tritonserver-silvic .
docker run --rm -p 8000:8000 -p 8001:8001 -p 8002:8002 -v "$PWD/models":/models tritonserver-silvic \
    tritonserver --model-repository=/models
```

To add the backend to another tritonserver, build `libtriton_silvic.so` on Linux
(`cargo build --release -p silvic-triton`; the aarch64 build needs glibc 2.34) and put it at
`<backend-directory>/silvic/libtriton_silvic.so`.

On macOS the crate builds and links (`-undefined dynamic_lookup`) for `cargo build --workspace`; the C API resolves
only inside a tritonserver process on Linux.

## Tests

- `cargo test -p silvic-triton`: the configuration parsing, auto-complete and validation, the request checks,
  batching and splitting, and output writing, against `Model::predict` on the LightGBM 3.3.5 fixtures of
  `silvic-core`, without Triton (the C API is left out of the test binary).
- `tests/triton/run.sh`: [`tests/triton/test_backend.py`](../../tests/triton/test_backend.py) against a real
  tritonserver. It builds the backend for Linux in a container, builds a 0.9 GB image with the tritonserver 2.51.0
  binary from the `nvidia-pytriton` 0.7.0 wheel ([`tests/triton/Dockerfile`](../../tests/triton/Dockerfile)),
  writes a model repository (`tests/triton/make_repository.py`), and runs pytest in a client container. Without
  `TRITON_URL` the test skips, so it is outside the default pytest run.

The end-to-end test checks, over HTTP and gRPC: outputs equal to `lightgbm.Booster.predict` 4.6.0 bit for bit for
`bench/data/serve_ltr.txt` (LightGBM-trained, 2000 trees) and a silvic-trained binary model with categorical
features and NaN, raw and transformed, FP32 and FP64 inputs, FP32 and FP64 outputs, batching on and off; 1-row
requests and 8 concurrent clients under dynamic batching, with several requests per execution; bad requests failing
alone, including a backend-rejected request batched with good ones; unload, load and reload through the
repository API, and load errors naming the problem; the statistics' request, inference and execution counts and
nested timestamps.

| Platform | Verified |
| --- | --- |
| aarch64 Linux (colima VM on an Apple M4 Pro), tritonserver 2.51.0 from `nvidia-pytriton` 0.7.0 | `tests/triton/run.sh`: all 19 tests pass |
| aarch64 Linux, `nvcr.io/nvidia/tritonserver:24.10-py3` with `deploy/triton/Dockerfile` | The image builds; all 19 tests pass against it |
| x86-64 Linux | `cargo check` only; the AVX2 kernel is tested by `silvic-serve`, not inside Triton |
| A host with GPUs | Not run: `KIND_GPU` refusal and `KIND_AUTO` resolving to CPU instances there are untested |

## Benchmark

`bench/triton.py` (`tests/triton/run.sh bench`): 3000-row x 136-feature FP32 requests over gRPC, one at a time per
client, `bench/data/serve_ltr.txt`, no dynamic batching, on the NGC 24.10 image in a colima VM with 4 vCPUs on an
Apple M4 Pro, clients in the same VM. The VM shares its 4 vCPUs between the server and the clients: these are not
production numbers. Server compute is Triton's mean `compute_infer` per request.

| Backend | Clients | Instances | Threads per request | p50 ms | p99 ms | Server compute ms | Rows/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| silvic | 1 | 1 | 1 | 13.2 | 15.0 | 12.3 | 223,000 |
| silvic | 4 | 4 | 1 | 13.9 | 18.3 | 12.8 | 831,000 |
| silvic | 1 | 1 | 4 | 4.0 | 6.9 | 3.4 | 702,000 |
| FIL | 1 | 1 | OpenMP default | 243 | 275 | 245 | 12,200 |
| FIL | 4 | 4 | OpenMP default | 1870 | 2128 | 1863 | 6,400 |
| FIL, `OMP_NUM_THREADS=1` | 1 | 1 | 1 | 1075 | 1140 | 1080 | 2,800 |
| FIL, `OMP_NUM_THREADS=1` | 4 | 4 | 1 | 2047 | 2160 | 2044 | 5,800 |

FIL runs on the CPU (`KIND_CPU`) with its default parameters and `model_type: lightgbm`, `output_class: false`.
silvic's outputs equal LightGBM's bit for bit in every row; FIL's differ by up to 2.4e-5 (it computes in f32).
