# silvic

LightGBM-compatible gradient boosting in Rust, on CPU, Apple Metal and NVIDIA CUDA, in one process or across Ray
actors.

silvic trains the models LightGBM trains (LightGBM's binning, objectives, split search and leaf-wise trees) and writes
LightGBM 3.3.5 model text, so `lightgbm.Booster` 3.3.5 and 4.x, `pred_contrib` SHAP values and lleaves serve its
models unchanged. Its histograms are integer sums of quantized gradients: training is bit-reproducible, and a model
trained by N Ray workers is the model trained on the same rows in one process.

## Crates

- [`silvic-core`](crates/silvic-core): parameters, binning, objectives, metrics, split search, leaf-wise growth, the
  boosting loop, model text and prediction, and the CPU backend.
- [`silvic-gpu`](crates/silvic-gpu): the Metal and CUDA backends, from one kernel source compiled at run time.
- [`silvic-net`](crates/silvic-net): TCP ring collectives for distributed training.
- [`silvic-python`](crates/silvic-python): the `silvic` Python package (LightGBM's Python API) and `silvic.ray`.

[`docs/`](docs) is the documentation site: an overview, a page per crate and the changelog.

## Use it

```python
import silvic

params = {
    "objective": "lambdarank",
    "num_leaves": 63, "learning_rate": 0.05,
    "bagging_fraction": 0.8, "bagging_freq": 5, "feature_fraction": 0.9,
    "device_type": "auto",  # CUDA, then Metal, then the CPU
}
train_set = silvic.Dataset(X, label=y, group=query_sizes, categorical_feature=categorical_columns)
booster = silvic.train(params, train_set, num_boost_round=1500)
booster.save_model("model.txt")      # LightGBM 3.3.5 model text, for lleaves
lgb_booster = booster.to_lightgbm()  # a lightgbm.Booster, to pickle like any other
```

`SilvicRanker`, `SilvicRegressor` and `SilvicClassifier` follow `LGBMRanker`, `LGBMRegressor` and `LGBMClassifier`.

Across Ray actors, from a Ray Dataset:

```python
import ray
import silvic.ray

train_ds = ray.data.read_parquet("s3://bucket/train/")
booster = silvic.ray.train(
    params, train_ds,
    label_column="label", group_column="query_id",
    num_workers=4, use_gpu=True, num_boost_round=1500,
)
```

Rows are hash-split by query group in one streaming pass over the dataset, so each actor holds whole queries in an
order that depends only on the data and `num_workers`. Actors run in a placement group and reduce histograms over a
TCP ring; if one fails, the others are killed and its error is raised.

## Performance

Lambdarank (53 leaves, depth 11, learning rate 0.05, bagging 0.8 every 5 iterations, `feature_fraction` 0.9, L1 14,
L2 0.27) on synthetic learning-to-rank data (`bench/gen_ltr.py`: 136 features, of which 90 are counts, 41 floats with
NaN and 5 categoricals; queries of about 110 rows), on an Apple M4 Pro (14 CPU cores, 20 GPU cores). Times include
binning; LightGBM 4.7.0 runs on every core. `bench/run.py` reproduces them.

| Train rows | Iterations | LightGBM 4.7 CPU | silvic CPU | silvic Metal |
| ---: | ---: | ---: | ---: | ---: |
| 1.8M | 500 | 65.9 s | 49.4 s | 23.0 s |
| 9.0M | 200 | 134.4 s | 97.5 s | 23.2 s |

Validation NDCG@12 on the same runs:

| Train rows | LightGBM 4.7 CPU | silvic CPU | silvic Metal |
| ---: | ---: | ---: | ---: |
| 1.8M | 0.8867 | 0.8920 | 0.8872 |
| 9.0M | 0.8387 | 0.8406 | 0.8415 |

LightGBM 4.7's own quantized training (`use_quantized_grad=true`) aborts on this data with
`Check failed: (best_split_info.left_count) > (0)`: 30% of its queries have no relevant item, so their rows have zero
gradient and hessian. silvic never splits a leaf without hessian and rejects empty children.

## Compatibility

- **Models**: silvic writes what LightGBM 3.3.5's `Booster.model_to_string()` writes, and reads LightGBM 3.x and
  4.x models back byte for byte. LightGBM 3.3.5 and 4.7 load silvic models and predict exactly what silvic predicts;
  `pred_contrib` contributions sum to the prediction; lleaves 1.1.0 compiles them. `bench/check_compat.py` checks
  all three on a lambdarank model.
- **Parameters**: LightGBM 3.3.5's names and aliases. Objectives `regression` (L2), `binary` and `lambdarank`;
  metrics `l2`, `rmse`, `binary_logloss`, `auc`, `ndcg` (`eval_at`); native categoricals, NaN, weights, bagging,
  `feature_fraction`, `max_depth`, L1/L2, `min_data_in_leaf`, `min_sum_hessian_in_leaf`, `min_gain_to_split`. An
  unknown key, or one of LightGBM's features outside this list (multiclass, DART, GOSS, early stopping, monotone
  constraints, linear trees, custom objectives), is an error rather than ignored.
- **Training numerics** differ from LightGBM by design: gradients are quantized by default (`use_quantized_grad`,
  `num_grad_quant_bins` 4), stochastic rounding and bagging draw from a counter-based hash of (seed, iteration, row),
  leaves are renewed from exact gradient sums (`quant_train_renew_leaf` defaults to true, LightGBM's to false), and
  training scores are f32. Models are LightGBM's in kind and accuracy, not bit-identical to LightGBM's trees.
  `use_quantized_grad=false` trains on float histograms.
- **Determinism**: integer histograms reduce exactly in any order, so the same data and parameters give the same
  model on every run and on any number of workers (when `bin_construct_sample_cnt` covers every row, so all worker
  counts bin from the same sample). Quantized L2 models are byte-identical on the CPU and Metal; binary and lambdarank
  gradients differ in the last float bits between the CPU (f64) and Metal (f32).
- **Queries** of any size train: LightGBM's 10000-position discount table is extended by its formula.

## Status

The CUDA backend compiles, never panics on a missing or mismatched driver, and its kernels pass clang's CUDA front
end, but it has not run on an NVIDIA GPU. The Metal and CPU backends, the Ray integration and the LightGBM, SHAP and
lleaves compatibility are tested on Apple silicon.
