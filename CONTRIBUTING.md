# Contributing

## Setup

- Rust: `rust-toolchain.toml` selects stable with rustfmt and clippy; the minimum supported version, the workspace's
  `rust-version`, is 1.94.1.
- Python: two [uv](https://docs.astral.sh/uv/) environments at the repository root. `.venv` builds and tests the
  package; `.venv-lgb3` holds LightGBM 3.3.5 and lleaves 1.1.0, the versions the compatibility checks load models
  in, which have no numpy 2 build:

  ```sh
  uv venv --python 3.12 .venv
  uv pip install --python .venv/bin/python maturin numpy pandas pyarrow "ray[default]==2.58.0" lightgbm lleaves \
      pytest scikit-learn

  uv venv --python 3.12 .venv-lgb3
  # LightGBM 3.3.5 has no arm64 macOS wheel and its CMakeLists predates CMake 4's policy floor; it builds with libomp.
  export CMAKE_POLICY_VERSION_MINIMUM=3.5
  uv pip install --python .venv-lgb3/bin/python setuptools wheel "numpy<2" scipy scikit-learn pandas
  uv pip install --python .venv-lgb3/bin/python --no-build-isolation "lightgbm==3.3.5"
  uv pip install --python .venv-lgb3/bin/python "lleaves==1.1.0" "llvmlite<0.45"
  ```

- CUDA: building needs no CUDA toolkit (cudarc loads the driver and NVRTC at run time). Running needs an NVIDIA
  driver and `libnvrtc` from CUDA 12. The CUDA syntax test needs a clang that accepts `-x cuda` (`SILVIC_CUDA_CLANG`,
  Homebrew's LLVM, or `clang++` on `PATH`).
- AWS Neuron: [`silvic-neuron`](crates/silvic-neuron) builds without the SDK. Execution needs a Neuron accelerator
  host, the Neuron driver/runtime and a separate Python 3.11 environment with the pinned `neuronx-cc` compiler.
  See the crate README for setup.
- [blaze](https://github.com/indexable-inc/blaze) can shorten debug edit-test loops (`blaze test -p silvic-core`), but
  cargo and maturin stay the build: blaze hands `--release` builds to cargo and refuses `lto`, and benchmarks, wheels
  and CI are release builds.

## Checks

The workspace-wide checks CI runs:

```sh
cargo fmt --all --check
cargo clippy --workspace --all-targets -- -D warnings
cargo clippy -p silvic-gpu --features cuda --all-targets -- -D warnings
cargo clippy -p silvic-gpu --features metal --all-targets -- -D warnings   # macOS
RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps
cargo +1.94.1 check --workspace --all-targets --locked
```

Tests:

| Crate | Command | Needs |
| --- | --- | --- |
| `silvic-core` | `cargo test -p silvic-core` | `.venv` and `.venv-lgb3` for the LightGBM/lleaves load check, skipped without them |
| `silvic-net` | `cargo test -p silvic-net` | |
| `silvic-serve` | `cargo test -p silvic-serve` | |
| `silvic-serve` | `ROSETTA_ADVERTISE_AVX=1 SILVIC_SERVE_EXPECT_KERNELS=portable,avx2 cargo test -p silvic-serve --target x86_64-apple-darwin` | Apple silicon with Rosetta 2 and the `x86_64-apple-darwin` target: runs the AVX2 kernel |
| `silvic-triton` | `cargo test -p silvic-triton` | |
| `silvic-triton` | `tests/triton/run.sh` | Docker (about 2 GB of images, and the `silvic-triton-target` and `silvic-triton-cargo` volumes) and `.venv`: builds the backend for Linux and runs `tests/triton/test_backend.py` against tritonserver 2.51.0 |
| `silvic-gpu` | `cargo test -p silvic-gpu --features metal` | a Metal device, skipped without one; clang with CUDA support for `tests/cuda_syntax.rs` |
| `silvic-gpu` | `cargo test -p silvic-gpu --features cuda` | nothing beyond the syntax test runs CUDA yet |
| `silvic-neuron` | `cargo test -p silvic-neuron` | host numeric/validation checks; hardware regressions run only with `SILVIC_TEST_NEURON=1` on a configured Neuron host |
| Python | `.venv/bin/maturin develop --release && .venv/bin/pytest tests/python` | `.venv`; the Ray tests start a local Ray cluster |

Timing tests are ignored by default and print per-operation times in release builds:

```sh
SILVIC_BENCH_ROWS=2000000 cargo test --release -p silvic-core --test train_cpu -- --ignored --nocapture cpu_training_speed
cargo test --release -p silvic-core --lib data::build::tests::binning_throughput -- --ignored --nocapture
cargo test --release -p silvic-core --lib objective -- --ignored --nocapture lambdarank_gradient_speed
```

## Benchmarks

Criterion benches under each crate's `benches/`, run with `cargo bench -p <crate> --bench <name>`:

| Crate | Bench | Measures |
| --- | --- | --- |
| `silvic-gpu` | `histogram` | Metal against `CpuBackend` histograms, 10M rows x 136 features, root and a 10% leaf (`--features metal`) |
| `silvic-net` | `allreduce` | Ring allreduce latency over 127.0.0.1, world sizes 2 to 8, 3 elements and 300 KB |
| `silvic-serve` | `predict` | Single-thread scoring of 3000 rows with a random 2000-tree, 53-leaf forest: the reference walk and every kernel |

End to end, `bench/run.py` trains the same lambdarank model with LightGBM and silvic on the same rows and reports wall
time and NDCG@12/@50; `bench/gen_ltr.py` documents the data. `bench/check_compat.py` trains a model with silvic and
checks it in LightGBM 3.3.5 (predictions and `pred_contrib` SHAP values) and lleaves 1.1.0.
`bench/serve.py` times single-thread scoring of 3000-row batches with a 2000-tree lambdarank model (trained once by
LightGBM on `gen_ltr.py` rows) in LightGBM, lleaves, treelite/tl2cgen and every `silvic.serve` kernel, and fails unless
`silvic.serve` equals LightGBM bit for bit; run the lleaves rows from `.venv-lgb3` (lleaves 1.1.0), whose LLVM build
the `.venv` lleaves cannot use. `bench/triton.py` times 3000-row gRPC requests to a tritonserver serving that model
with the silvic backend (or FIL's, `--backend fil`): client p50/p99 and Triton's per-request compute time.

```sh
.venv/bin/python bench/run.py --rows 2000000 --iterations 500 --engines silvic-metal,silvic-cpu,lightgbm
.venv/bin/python bench/check_compat.py train --rows 2000000 --out /tmp/silvic-compat
.venv-lgb3/bin/python bench/check_compat.py check --out /tmp/silvic-compat
.venv/bin/python bench/serve.py --engines lightgbm,treelite,treelite-quantize,silvic-portable,silvic-neon
.venv-lgb3/bin/python bench/serve.py --engines lightgbm,lleaves,lleaves-noinline
tests/triton/run.sh bench --rows 1:1:1,4:4:1,1:1:4
```

## Fixtures

The parity fixtures are recorded from LightGBM, by uv scripts whose headers pin the version they record:

| Generator | Records | Read by |
| --- | --- | --- |
| `crates/silvic-core/tests/fixtures/lightgbm/gen_models.py` | LightGBM 3.3.5 models, edge-case rows and their predictions | `tests/model_lightgbm.rs` |
| `crates/silvic-core/tests/fixtures/lightgbm/gen_models_v4.py` | LightGBM 4.7 models of three of those cases | `tests/model_lightgbm.rs` |
| `crates/silvic-core/tests/fixtures/metrics/generate.py` | LightGBM 4.7 metric values on fixed scores | `tests/metric_lightgbm.rs` |
| `crates/silvic-core/src/data/fixtures/gen_lightgbm_bins.py` | LightGBM 3.3.5 bin boundaries and `feature_infos` | `src/data/parity_tests.rs` |
| `crates/silvic-core/tests/fixtures/lambdarank_parity.py` | LightGBM 4.7 validation NDCG@10 on hash-generated rows | `tests/train_cpu.rs` |

Their output is deterministic, so a regeneration diff shows only real changes. A fixture that stops matching means
silvic no longer reads, writes or trains what LightGBM does; fix silvic rather than the fixture.

## Docs site

`docs/` is a static site, served as is (GitHub Pages from `/docs`): the overview, a page per crate under `docs/docs/`,
the changelog and its RSS feed. It has no build step; preview it with
`python3 -m http.server --directory docs`. The crate pages describe the same public API as the crate READMEs and the
changelog page follows `CHANGELOG.md`, so a change to either updates both. The overview's figures come from
`bench/run.py`; update them only from a new run. `mark.js` draws the animated mark with WebGL and falls back to a
still shape without it.

## Style

Comments explain why, not what: the LightGBM behaviour being matched (named by its class or function in 3.3.5), an
invariant, a format fact. Rustdoc and the READMEs describe behaviour in the present tense. Functions stay under
clippy's cognitive complexity threshold (a workspace lint, so `-D warnings` fails on it); split a function by phase or
by match arm rather than allowing the lint. Names say what they hold; single letters are for closure parameters, short
loop indices and formulas ported from LightGBM (keeping its names), and clippy's `many_single_char_names` is a
workspace lint. Kernels use only the macro vocabulary both preludes define. Commit messages follow Conventional
Commits, scoped by crate (`feat(silvic-gpu): ...`).
