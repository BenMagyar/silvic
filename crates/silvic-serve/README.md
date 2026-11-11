# silvic-serve

Batch scoring of LightGBM models (`model.txt` from LightGBM 3.3.5 or 4.x, or from silvic), with predictions equal to
LightGBM's bit for bit. `Predictor::new` compiles a `silvic_core::Model` in milliseconds; `predict` scores a dense
matrix (f32 or f64, row- or column-major) on the calling thread or on rayon's pool.

Each tree is QuickScorer's leaf bitvector (Lucchese et al., SIGIR 2015): leaves numbered left to right, and per node
the bits of its left subtree, cleared in the rows that go right there; the lowest bit left is the exit leaf, so a tree
is scored without branching on the data. Rows are scored 64 at a time (32 on the portable kernel), one SIMD byte lane
per row, with each tree's bitvectors byte-sliced across the lanes as in RapidScorer (Ye et al., KDD 2018) and in the
spirit of vQS (Lucchese et al., SIGIR 2016). Before that, each block of rows is quantized to one byte per (feature,
missing-value handling, 254 thresholds): a value's rank among the feature's split thresholds, so every split is a
byte compare. Leaf values are summed per row in tree order, in f64, as `GBDT::PredictRaw` sums them.

| LightGBM 3.3.5 | silvic-serve |
| --- | --- |
| Dense row reader drops `\|v\| <= kZeroThreshold` | Such values read as 0 before ranking |
| `NumericalDecision`, `MissingType::None` (NaN read as 0) | NaN takes the rank of 0 |
| `NumericalDecision`, `MissingType::NaN` / `Zero` | NaN (and 0 for `Zero`) take code 0 or 255: left or right of every threshold, per `default_left` |
| `CategoricalDecision` bitsets | Categories every split treats alike share a class; a split is a 64-byte table of the classes it sends left |
| NaN, negative or unseen category goes right | Such categories have no class and read 0 from every table |
| `GBDT::PredictRaw`, `ConvertOutput` | Leaf values added in tree order in f64; the sigmoid for `binary` |

Trees with more than 256 leaves, a NaN threshold, or a categorical split on a feature with more than 64 distinct
category behaviours are walked node by node with `silvic_core`'s decisions (`Predictor::num_walked_trees` counts
them), so every model `Model::from_lightgbm_string` accepts scores, those trees at the reference walk's speed.

## Kernels

| Kernel | CPUs | Rows per block | Picked by `Kernel::detect` |
| --- | --- | --- | --- |
| `Neon` | aarch64 | 64 | yes |
| `Avx2` | x86-64 with AVX2 (runtime detection) | 64 | yes |
| `Avx512` | x86-64 with AVX-512F and AVX-512BW (runtime detection) | 64 | no: ask for it by name |
| `Portable` | any; plain Rust over u64 words (SWAR) | 32 | when nothing else runs |

Every kernel gives the same predictions. The AVX-512 kernel compiles and passes clippy but has not run on AVX-512
hardware yet, which is why `Kernel::detect` does not pick it: run this crate's tests on such a machine
(`SILVIC_SERVE_EXPECT_KERNELS=portable,avx2,avx512 cargo test -p silvic-serve`) before selecting it.

## Use it

```rust
use silvic_core::data::{DenseData, DenseMatrix, Layout};
use silvic_core::Model;
use silvic_serve::Predictor;

let model = Model::from_lightgbm_string(&std::fs::read_to_string("model.txt")?)?;
let predictor = Predictor::new(&model)?;
let rows = DenseMatrix { data: DenseData::F32(&x), rows: 3000, cols: model.num_features(), layout: Layout::RowMajor };
let scores = predictor.predict(&rows, false, None, 1); // raw_score, num_trees, num_threads
```

From Python, `silvic.serve.Predictor(booster_or_text_or_path).predict(X, num_threads=1)`, or the lleaves drop-in
`silvic.serve.Model(model_file="model.txt")` with `compile()` and `predict(X, n_jobs=1)`.

## Tests

`cargo test -p silvic-serve` compares every kernel the CPU runs, bit for bit:

- `tests/lightgbm_fixtures.rs`: against the predictions LightGBM 3.3.5 and 4.7 recorded for the models and edge rows
  in `silvic-core/tests/fixtures/lightgbm`, all trees and a prefix.
- `tests/random_forests.rs`: against `Model::predict_raw_row` on random forests (every missing type and default
  direction, multi-word category bitsets, 64 classes, more than 64 classes, single-leaf trees, 65 to 300 leaves, NaN
  thresholds, 700 thresholds per feature) and rows at every threshold, its neighbouring doubles, NaN, ±0,
  ±`kZeroThreshold` and ±inf; and against `Model::predict` for f32/f64, row- and column-major input, raw and
  transformed scores, tree prefixes and thread counts.
- `tests/trained_models.rs`: against `Model::predict` on regression, binary and lambdarank models silvic trains with
  NaN and categorical features, and on 120- and 300-leaf trees.

`SILVIC_SERVE_EXPECT_KERNELS` (e.g. `portable,neon`) fails the run unless exactly those kernels ran. The AVX2 kernel
runs on Apple silicon under Rosetta 2, which reports AVX2 only when asked:
`ROSETTA_ADVERTISE_AVX=1 SILVIC_SERVE_EXPECT_KERNELS=portable,avx2 cargo test -p silvic-serve --target x86_64-apple-darwin`.

`cargo bench -p silvic-serve --bench predict` times 3000 rows of a random 2000-tree, 53-leaf forest through the
reference walk and every kernel; `bench/serve.py` compares against LightGBM, lleaves and treelite on a trained model.
