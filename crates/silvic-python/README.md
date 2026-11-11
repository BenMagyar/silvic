# silvic-python

The `silvic._silvic` extension module (PyO3, abi3 for CPython 3.10+), built by maturin from the repository's
`pyproject.toml`, and the `silvic` Python package in [`python/silvic`](../../python/silvic) that wraps it in
LightGBM's Python API.

| silvic | LightGBM 3.3.5 / 4.x |
| --- | --- |
| `silvic.Dataset` | `lightgbm.Dataset` (dense numpy and pandas; lazy construction; `reference=`) |
| `silvic.train` | `lightgbm.train` (`init_model` continues bit-exactly; no `fobj`, `feval`, early stopping) |
| `silvic.Booster` | `lightgbm.Booster` (`predict`, `model_to_string`, `save_model`, `feature_importance`, pickling) |
| `silvic.log_evaluation`, `silvic.record_evaluation` | the LightGBM 4.x callbacks |
| `SilvicRegressor`, `SilvicClassifier`, `SilvicRanker` | `LGBMRegressor`, `LGBMClassifier` (binary), `LGBMRanker` |
| `silvic.ray.train` | distributed training over Ray actors, resuming from checkpoints after worker loss (no LightGBM counterpart) |

Models are LightGBM 3.3.5 model text; `Booster.to_lightgbm()` returns `lightgbm.Booster(model_str=...)`.

```python
import silvic

train_set = silvic.Dataset(X, label=y, group=query_sizes)
booster = silvic.train({"objective": "lambdarank", "num_leaves": 31}, train_set, num_boost_round=500)
scores = booster.predict(X_test)
```

## AWS Neuron

Explicit `device_type="neuron"` uses [`silvic-neuron`](../silvic-neuron) for training on Linux; `auto` and `gpu`
retain their existing behavior. Serving requires `silvic.serve.Predictor(model, device="neuron", precision="float32")`;
CPU/CUDA float64 defaults are unchanged. The Neuron compiler and runtime are installed separately as documented
in that crate.

For `silvic.ray.train`, `device_type="neuron"` reserves one `neuron_cores` resource per worker by default.
Use `resources_per_worker` to override the reservation; do not combine Neuron with `use_gpu=True` or GPU resources.

## Tests

`tests/python` compares predictions against LightGBM 4.x loading the same model text, and distributed Ray training
against single-process training on the same rows:

```sh
.venv/bin/maturin develop --release
.venv/bin/pytest tests/python
```
