"""``silvic.serve`` against ``silvic.Booster.predict`` and ``lightgbm.Booster.predict``: predictions must be equal bit
for bit for every kernel, input layout and source of the model."""

import threading

import lightgbm
import numpy as np
import pandas as pd
import pytest
from conftest import NUM_FEATURES, binary_data, ranking_data, regression_data

import silvic
from silvic import serve

ROUNDS = 30
PARAMS = {"num_leaves": 31, "min_data_in_leaf": 5, "verbosity": -1}
CASES = {
    "regression": ({"objective": "regression"}, regression_data),
    "binary": ({"objective": "binary"}, binary_data),
    "lambdarank": ({"objective": "lambdarank"}, ranking_data),
}


def _train(name):
    objective, make = CASES[name]
    data = make()
    X, y = data[0], data[1]
    group = data[2] if len(data) > 2 else None
    booster = silvic.train({**PARAMS, **objective}, silvic.Dataset(X, label=y, group=group), ROUNDS, verbose_eval=False)
    return booster, X


def _layouts(X):
    """The same rows as float64 and float32, in C and Fortran order, with extra NaN and zeros."""
    X = X.copy()
    X[::7, 1] = np.nan
    X[::5, 2] = 0.0
    X[::11, 3] = -0.0
    for dtype in (np.float64, np.float32):
        yield np.ascontiguousarray(X, dtype=dtype)
        yield np.asfortranarray(X, dtype=dtype)


@pytest.mark.parametrize("name", CASES)
def test_matches_booster_for_every_kernel_and_layout(name):
    booster, X = _train(name)
    for kernel in serve.available_kernels():
        predictor = serve.Predictor(booster, kernel=kernel)
        assert predictor.kernel == kernel
        assert predictor.num_trees() == booster.num_trees()
        assert predictor.num_feature() == NUM_FEATURES
        for data in _layouts(X):
            for raw_score in (False, True):
                expected = booster.predict(data, raw_score=raw_score)
                for num_threads in (1, 0, 3):
                    got = predictor.predict(data, raw_score=raw_score, num_threads=num_threads)
                    assert got.dtype == np.float64
                    np.testing.assert_array_equal(got, expected, err_msg=f"{kernel} {data.dtype} raw={raw_score}")


def test_num_iteration_matches_booster():
    booster, X = _train("binary")
    predictor = serve.Predictor(booster)
    for num_iteration in (None, 0, -1, 1, 7, ROUNDS, ROUNDS + 10):
        np.testing.assert_array_equal(
            predictor.predict(X, num_iteration=num_iteration), booster.predict(X, num_iteration=num_iteration)
        )


def test_model_sources_agree(tmp_path):
    booster, X = _train("regression")
    path = tmp_path / "model.txt"
    booster.save_model(path)
    expected = booster.predict(X)
    lgb_booster = lightgbm.Booster(model_file=str(path))
    for source in (booster, booster.model_to_string(), str(path), path, lgb_booster):
        np.testing.assert_array_equal(serve.Predictor(source).predict(X), expected)
    np.testing.assert_array_equal(lgb_booster.predict(X, num_threads=1), expected)


def test_lightgbm_trained_model_matches_lightgbm():
    X, y = regression_data(rows=3000, seed=3)
    X[::9, 4] = np.nan
    categories = np.random.default_rng(3).integers(0, 12, size=len(X))
    X = np.column_stack([X, categories.astype(np.float64)])
    y = y + np.where(categories % 3 == 0, 1.0, -0.5)
    dataset = lightgbm.Dataset(X, y, categorical_feature=[NUM_FEATURES], params={"verbosity": -1})
    lgb_booster = lightgbm.train({**PARAMS, "objective": "regression"}, dataset, num_boost_round=ROUNDS)
    predictor = serve.Predictor(lgb_booster)
    np.testing.assert_array_equal(predictor.predict(X), lgb_booster.predict(X, num_threads=1))
    np.testing.assert_array_equal(predictor.predict(X, raw_score=True), lgb_booster.predict(X, raw_score=True))


def test_pandas_categories_map_as_at_training():
    rng = np.random.default_rng(5)
    rows = 3000
    colour = pd.Categorical(rng.choice(["red", "green", "blue", "teal"], size=rows))
    numeric = rng.normal(size=rows)
    effect = pd.Series(colour).map({"red": 3.0, "green": -1.0, "blue": 0.5, "teal": 0.0}).to_numpy(float)
    frame = pd.DataFrame({"colour": colour, "x": numeric})
    y = effect + numeric + rng.normal(scale=0.1, size=rows)
    booster = silvic.train({**PARAMS, "objective": "regression"}, silvic.Dataset(frame, label=y), ROUNDS, verbose_eval=False)
    predictor = serve.Predictor(booster)
    assert predictor.pandas_categorical == [["blue", "green", "red", "teal"]]
    reordered = frame.assign(colour=frame["colour"].cat.reorder_categories(["teal", "red", "green", "blue"]))
    for data in (frame, reordered):
        np.testing.assert_array_equal(predictor.predict(data), booster.predict(data))
    np.testing.assert_array_equal(predictor.predict(frame), booster.to_lightgbm().predict(frame))
    missing = frame.assign(colour=frame["colour"].where(np.arange(rows) % 4 != 0))
    np.testing.assert_array_equal(predictor.predict(missing), booster.predict(missing))


def test_lleaves_style_model(tmp_path):
    booster, X = _train("binary")
    path = tmp_path / "model.txt"
    booster.save_model(path)
    model = serve.Model(model_file=path)
    assert model.compile() is None
    assert model.num_feature() == NUM_FEATURES
    assert model.num_trees() == ROUNDS
    assert model.num_model_per_iteration() == 1
    np.testing.assert_array_equal(model.predict(X, n_jobs=1), booster.predict(X))
    np.testing.assert_array_equal(model.predict(X), booster.predict(X))
    raw = serve.Model(model_str=booster.model_to_string())
    raw.compile(cache="ignored.bin", raw_score=True, fblocksize=16, finline=False)
    np.testing.assert_array_equal(raw.predict(X, n_jobs=2), booster.predict(X, raw_score=True))
    with pytest.raises(TypeError):
        serve.Model()
    with pytest.raises(TypeError):
        serve.Model(model_file=path, model_str=booster.model_to_string())


def test_errors():
    booster, X = _train("regression")
    predictor = serve.Predictor(booster)
    with pytest.raises(ValueError, match="number of features"):
        predictor.predict(X[:, :-1])
    with pytest.raises(ValueError, match="unknown kernel"):
        serve.Predictor(booster, kernel="sse9")
    with pytest.raises(silvic.ModelError):
        serve.Predictor("tree\nversion=v3\nnum_class=1\n")
    with pytest.raises(TypeError):
        serve.Predictor(42)
    missing = [k for k in ("portable", "neon", "avx2", "avx512") if k not in serve.available_kernels()]
    for kernel in missing:
        with pytest.raises(silvic.UnsupportedError):
            serve.Predictor(booster, kernel=kernel)


def test_threads_score_concurrently_without_the_gil():
    booster, X = _train("lambdarank")
    predictor = serve.Predictor(booster)
    expected = booster.predict(X)
    results = [None] * 8

    def score(index):
        results[index] = predictor.predict(X, num_threads=1)

    threads = [threading.Thread(target=score, args=(i,)) for i in range(len(results))]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    for result in results:
        np.testing.assert_array_equal(result, expected)
