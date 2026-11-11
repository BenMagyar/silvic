# silvic-core

LightGBM-compatible gradient boosting: binning, objectives, leaf-wise tree growth on quantized integer histograms,
and LightGBM 3.3.5 model text. Training runs through a `Backend`: this crate's `CpuBackend`, the Metal and CUDA
backends of [`silvic-gpu`](../silvic-gpu), or the AWS Neuron backend of
[`silvic-neuron`](../silvic-neuron). Histograms reduce across ranks through a `Collective`
(`LocalCollective` in one process, [`silvic-net`](../silvic-net)'s TCP ring across many).

| Module | LightGBM 3.3.5 counterpart |
| --- | --- |
| `config` | `Config` (`config.h`): names, aliases, defaults; unknown or unsupported keys are errors |
| `data` | `BinMapper::FindBin` (`GreedyFindBin`, `FindBinWithZeroAsOneBin`, categorical), `Random::Sample` |
| `objective` | `RegressionL2loss`, `BinaryLogloss`, `LambdarankNDCG` (with its sigmoid table) |
| `metric` | `L2Metric`, `RMSEMetric`, `BinaryLoglossMetric`, `AUCMetric`, `NDCGMetric` |
| `split` | `FeatureHistogram` (numerical with both missing directions, one-hot and sorted categorical) |
| `learner` | `SerialTreeLearner` / `DataParallelTreeLearner` (leaf-wise growth, histogram subtraction) |
| `boosting` | `GBDT::TrainOneIter` (boost from average, bagging, shrinkage, single-leaf stop) |
| `quantize` | LightGBM 4's `GradientDiscretizer` (`use_quantized_grad`), with a counter-based RNG |
| `tree`, `model` | `Tree`, `GBDT::SaveModelToString` / `LoadModelFromString`, `Tree::GetLeaf` |

Quantized histograms are integer sums keyed by global row id (stochastic rounding and bagging draw from a hash of
seed, iteration and row), so a model is the same on every backend that computes the same gradients and on any
number of ranks: allreducing integers is exact in any order. Leaf values are renewed from exact fixed-point sums of
the float gradients (`quant_train_renew_leaf`, on by default).

```rust
use silvic_core::data::{DatasetInput, DenseData, DenseMatrix, Layout};
use silvic_core::{train, BinnedDataset, CpuBackend, LocalCollective, Params};

let params = Params::from_json(r#"{"objective": "lambdarank", "num_leaves": 31, "num_iterations": 100}"#)?;
let input = DatasetInput {
    features: DenseMatrix { data: DenseData::F32(&x), rows, cols, layout: Layout::RowMajor },
    label: &labels,
    weight: None,
    group: Some(&query_sizes),
    feature_names: None,
    categorical_features: vec![],
};
let train_set = BinnedDataset::construct(&input, &params, &mut LocalCollective)?;
let mut backend = CpuBackend::new();
let model = train(&params, &train_set, &[], &mut backend, &mut LocalCollective, &mut |_| Ok(()))?;
std::fs::write("model.txt", model.to_lightgbm_string())?;
```

## Tests

- `tests/model_lightgbm.rs` reads models written by LightGBM 3.3.5 and 4.7 (`tests/fixtures/lightgbm/`), writes
  them back byte for byte and predicts bit for bit what LightGBM predicted; silvic-built models load in LightGBM
  3.3.5 and 4.7 and compile in lleaves 1.1.0 with the same predictions (`check_load.py`, skipped without the venvs).
- `tests/metric_lightgbm.rs` compares every metric with LightGBM 4.7's on the same scores.
- `src/data/parity_tests.rs` rebuilds LightGBM 3.3.5's bin boundaries from the same samples
  (`src/data/fixtures/`).
- `tests/train_cpu.rs` trains every objective, checks that two ranks train the model of one byte for byte, and
  compares lambdarank validation NDCG@10 with LightGBM 4.7's on the same rows (`tests/fixtures/lambdarank_parity.py`).
