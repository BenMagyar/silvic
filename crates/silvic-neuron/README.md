# silvic-neuron

AWS Neuron training and explicit float32 serving for silvic, through NKI kernels and the dynamically loaded
Neuron Runtime (`libnrt.so`). Targets Trainium, Trainium2 and Inferentia2 on Linux. No CPU fallback and no SDK
needed to build the Rust crate.

## Setup

Install the [Neuron driver and runtime](https://awsdocs-neuron.readthedocs-hosted.com/en/v2.25.0/neuron-runtime/nrt-api-guide.html#requirements)
on the accelerator host. The compiler runs in a separate Linux x86-64 Python 3.11 environment; PyTorch is not required:

```sh
python3.11 -m venv .venv-neuron
.venv-neuron/bin/pip install --extra-index-url https://pip.repos.neuron.amazonaws.com \
    'neuronx-cc==2.21.18209.0+043b1bf7' 'numpy==1.26.4' 'scipy==1.12.0'
export SILVIC_NEURON_PYTHON="$PWD/.venv-neuron/bin/python"
export SILVIC_NEURON_LIBRARY=/opt/aws/neuron/lib/libnrt.so
export NEURON_LOGICAL_NC_CONFIG=1
```

Set these before starting the process. Trainium2 requires logical-core configuration 1, matching the single-core
compiler specialization. The compiler version is pinned because the standalone NEFF adapter uses its internal
NKI compiler API. Compiled kernels are cached by source, tensor shapes, constants, target and compiler
version under `$XDG_CACHE_HOME/silvic/neuron` or `~/.cache/silvic/neuron`; `SILVIC_NEURON_CACHE` overrides the directory.
Missing runtimes, devices and compilers return errors.

## Training

`NeuronBackend::new(ordinal)` implements `silvic_core::Backend`; `None` picks logical NeuronCore 0. Binned rows,
partitions and scores stay on the device. NKI kernels build histograms, stably partition rows, reduce leaf sums and
update scores. Objectives, deterministic quantization, bagging and split search run on the host.

Quantized histograms reduce bounded integer values in exact float32 tiles and fold them in float64 on the host.
Fixed-point leaf sums use four signed radix-2^16 limbs, exact tile reductions and a checked wide host fold rather
than lossy i64-to-f32 conversion. Float histograms remain approximate. Device addresses are restricted to signed
32-bit storage; oversized datasets and models fail explicitly.

```python
import silvic

booster = silvic.train(
    {"objective": "lambdarank", "device_type": "neuron"},
    silvic.Dataset(X, label=y, group=query_sizes),
    num_boost_round=200,
)
```

`auto` and `gpu` keep their existing CUDA/Metal/CPU selection. `gpu_device_id` selects a logical NeuronCore within
`NEURON_RT_VISIBLE_CORES`. For Ray, pass `device_type="neuron"` to `silvic.ray.train` without `use_gpu=True`.
Each worker reserves one `neuron_cores` resource by default; Ray supplies core visibility. A positive whole-number
`resources_per_worker["neuron_cores"]` overrides the reservation. The backend uses one core per worker and the existing
TCP ring for distributed histogram reduction.

## Serving

`NeuronPredictor::new(&model, ordinal)` retains the forest and reusable buffers on the device. `predict` returns
float32 scores; `predict_raw` writes raw scores to a caller's float32 buffer. Inputs may be float32 or float64,
row- or column-major; float64 inputs are converted to float32. Numerical and categorical splits, missing values,
tree prefixes and binary transforms are supported. Trees accumulate in their original order.

```python
import silvic.serve

predictor = silvic.serve.Predictor(booster, device="neuron", precision="float32")
scores = predictor.predict(X_test)
```

Thresholds, leaves, accumulation and output use float32. Predictions are approximate. Neuron
requires explicit float32 precision and default device threading (`num_threads=1`); CPU kernel selection is invalid.
CPU and CUDA serving default to float64. `is_available()` probes the runtime and visible accelerator.

## Tests

```sh
cargo test -p silvic-neuron
SILVIC_TEST_NEURON=1 cargo test -p silvic-neuron -- --nocapture
```

The first command runs host checks; device regressions execute only with the environment variable set on a configured
Neuron host. Python device tests run when `silvic.available_devices()` / `silvic.serve.available_devices()` include
`neuron`.

From the repository root, `"$SILVIC_NEURON_PYTHON" crates/silvic-neuron/tests/kernels.py` runs the compiler/simulator
regressions without hardware: signed cancellation across i64 limits, ordered float32 tree prefixes, and single-core
NEFF tensor contracts on `trn1`, `inf2` and `trn2`.
