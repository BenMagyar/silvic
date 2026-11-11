# silvic-python

The `silvic._silvic` extension module (PyO3, abi3 for CPython 3.10+), built by maturin from the repository's
`pyproject.toml`, and the `silvic` Python package in [`python/silvic`](../../python/silvic) that wraps it in
LightGBM's Python API.

| silvic | LightGBM 3.3.5 / 4.x |
| --- | --- |
| `silvic.Dataset` | `lightgbm.Dataset` (dense numpy and pandas; lazy construction; `reference=`) |
| `silvic.train` | `lightgbm.train` (no `fobj`, `feval`, `init_model`, early stopping) |
| `silvic.Booster` | `lightgbm.Booster` (`predict`, `model_to_string`, `save_model`, `feature_importance`, pickling) |
| `silvic.log_evaluation`, `silvic.record_evaluation` | the LightGBM 4.x callbacks |
| `SilvicRegressor`, `SilvicClassifier`, `SilvicRanker` | `LGBMRegressor`, `LGBMClassifier` (binary), `LGBMRanker` |
| `silvic.ray.train` | distributed training over Ray actors (no LightGBM counterpart) |

Models are LightGBM 3.3.5 model text; `Booster.to_lightgbm()` returns `lightgbm.Booster(model_str=...)`.

```python
import silvic

train_set = silvic.Dataset(X, label=y, group=query_sizes)
booster = silvic.train({"objective": "lambdarank", "num_leaves": 31}, train_set, num_boost_round=500)
scores = booster.predict(X_test)
```

## Tests

`tests/python` compares predictions against LightGBM 4.x loading the same model text, and distributed Ray training
against single-process training on the same rows:

```sh
.venv/bin/maturin develop --release
.venv/bin/pytest tests/python
```
