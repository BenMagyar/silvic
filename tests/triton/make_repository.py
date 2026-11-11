"""Writes the Triton model repository the end-to-end test (`test_backend.py`) and `bench/triton.py` serve.

    .venv/bin/python tests/triton/make_repository.py --out ~/.cache/silvic-triton/models

Models, all `backend: "silvic"`:

- `ltr`: `bench/data/serve_ltr.txt` (LightGBM, 2000-tree lambdarank, 136 features), FP32 in, FP64 out, batching with
  dynamic batching (a 20 ms queue delay, so concurrent requests share executions).
- `ltr_nobatch`: the same model, `max_batch_size: 0`, FP64 in, FP64 out.
- `ltr_auto`: only `backend` and `max_batch_size` (auto-complete fills in the I/O), the model as `1/booster.txt` via
  `model_file`.
- `ltr_fil`: the FIL backend's config for the model with only `backend` changed: FP32 out, `KIND_AUTO`, FIL's
  `model_type`, `is_classifier` and `storage_type` parameters.
- `ltr_optional`: the input marked `optional`, so a request without it reaches the backend.
- `binary`: a silvic-trained binary model with categorical features and NaN, FP32 in, FP64 out, FIL's
  `is_classifier` and `predict_proba` (the probability), dynamic batching.
- `binary_raw`: the same model, `max_batch_size: 0`, FP64 in, `raw_score`, `num_iteration: 20`, the portable
  kernel, 2 threads.

`bench/data/serve_ltr.txt` comes from `bench/serve.py`'s first run. The binary model is trained here with silvic
(`.venv`, after `maturin develop`), deterministic for a given silvic.
"""

import argparse
import shutil
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
LTR_MODEL = ROOT / "bench" / "data" / "serve_ltr.txt"
LTR_FEATURES = 136
BINARY_FEATURES = 6
BINARY_CATEGORICAL = [2, 3]


def binary_rows(rows, seed):
    """Rows for the binary model: NaN in two numeric columns, zeros in one, categories 0..9 in two (unseen 10..12
    and negatives when `seed` is not the training seed)."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(rows, BINARY_FEATURES))
    X[rng.random(rows) < 0.1, 0] = np.nan
    X[rng.random(rows) < 0.05, 1] = np.nan
    levels = 10 if seed == 0 else 13
    X[:, 2] = rng.integers(0, levels, size=rows)
    X[:, 3] = rng.integers(-1 if seed else 0, 6, size=rows)
    X[rng.random(rows) < 0.3, 4] = 0.0
    return X


def train_binary(path):
    import silvic

    X = binary_rows(5000, seed=0)
    logit = 0.8 * np.nan_to_num(X[:, 0], nan=1.5) - X[:, 1] ** 2 + (X[:, 2] % 3 == 0) - 0.5 * (X[:, 3] == 2)
    y = (logit + np.random.default_rng(1).normal(size=len(X)) > 0).astype(np.float64)
    params = {"objective": "binary", "num_leaves": 15, "learning_rate": 0.1, "min_data_in_leaf": 20, "verbose": -1}
    dataset = silvic.Dataset(X, label=y, categorical_feature=BINARY_CATEGORICAL)
    booster = silvic.train(params, dataset, num_boost_round=60)
    booster.save_model(str(path))


def tensor(kind, name, data_type, dims, extra=""):
    dims = ", ".join(str(d) for d in dims)
    return f'{kind} [{{ name: "{name}" data_type: {data_type} dims: [ {dims} ]{extra} }}]\n'


def parameters(**values):
    entries = ", ".join(f'{{ key: "{key}" value: {{ string_value: "{value}" }} }}' for key, value in values.items())
    return f"parameters [ {entries} ]\n"


def configs(binary_features):
    batching_ltr = tensor("input", "input__0", "TYPE_FP32", [LTR_FEATURES]) + tensor(
        "output", "output__0", "TYPE_FP64", [1]
    )
    return {
        "ltr": (
            LTR_MODEL,
            "model.txt",
            "max_batch_size: 4096\n"
            + batching_ltr
            + "dynamic_batching { max_queue_delay_microseconds: 20000 }\n"
            + "instance_group [{ kind: KIND_CPU count: 1 }]\n",
        ),
        "ltr_nobatch": (
            LTR_MODEL,
            "model.txt",
            "max_batch_size: 0\n"
            + tensor("input", "input__0", "TYPE_FP64", [-1, LTR_FEATURES])
            + tensor("output", "output__0", "TYPE_FP64", [-1]),
        ),
        "ltr_auto": (LTR_MODEL, "booster.txt", "max_batch_size: 4096\n" + parameters(model_file="booster.txt")),
        "ltr_fil": (
            LTR_MODEL,
            "model.txt",
            "max_batch_size: 32768\n"
            + tensor("input", "input__0", "TYPE_FP32", [LTR_FEATURES])
            + tensor("output", "output__0", "TYPE_FP32", [1])
            + "instance_group [{ kind: KIND_AUTO }]\n"
            + parameters(model_type="lightgbm", is_classifier="false", storage_type="AUTO")
            + "dynamic_batching {}\n",
        ),
        "ltr_optional": (
            LTR_MODEL,
            "model.txt",
            "max_batch_size: 64\n"
            + tensor("input", "input__0", "TYPE_FP32", [LTR_FEATURES], " optional: true")
            + tensor("output", "output__0", "TYPE_FP64", [1])
            + "dynamic_batching { max_queue_delay_microseconds: 20000 }\n",
        ),
        "binary": (
            None,
            "model.txt",
            "max_batch_size: 4096\n"
            + tensor("input", "input__0", "TYPE_FP32", [binary_features])
            + tensor("output", "output__0", "TYPE_FP64", [1])
            + parameters(is_classifier="true", predict_proba="true")
            + "dynamic_batching { max_queue_delay_microseconds: 20000 }\n",
        ),
        "binary_raw": (
            None,
            "model.txt",
            "max_batch_size: 0\n"
            + tensor("input", "input__0", "TYPE_FP64", [-1, binary_features])
            + tensor("output", "output__0", "TYPE_FP64", [-1])
            + parameters(raw_score="true", num_iteration="20", kernel="portable", num_threads="2"),
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if not LTR_MODEL.exists():
        raise SystemExit(f"{LTR_MODEL} is missing: run bench/serve.py once to train it")
    out = args.out.expanduser()
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    binary_model = out / "binary.model.txt"
    train_binary(binary_model)
    for name, (source, file_name, body) in configs(BINARY_FEATURES).items():
        version = out / name / "1"
        version.mkdir(parents=True)
        shutil.copyfile(source or binary_model, version / file_name)
        (out / name / "config.pbtxt").write_text(f'backend: "silvic"\n{body}')
    binary_model.unlink()
    print(f"wrote {', '.join(sorted(p.name for p in out.iterdir()))} to {out}")


if __name__ == "__main__":
    main()
