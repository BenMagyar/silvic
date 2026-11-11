"""``silvic.serve`` against ``silvic.Booster.predict`` and ``lightgbm.Booster.predict``: predictions must be equal bit
for bit for every kernel, input layout and source of the model."""

import json
import threading
from pathlib import Path

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


@pytest.fixture(params=["cpu", "cuda"])
def device(request):
    if request.param == "cuda" and "cuda" not in serve.available_devices():
        pytest.skip("CUDA serving is unavailable")
    return request.param


@pytest.fixture(params=["cuda", "neuron"])
def float32_device(request):
    if request.param not in serve.available_devices():
        pytest.skip(f"{request.param} serving is unavailable")
    return request.param


def _train(name):
    objective, make = CASES[name]
    data = make()
    X, y = data[0], data[1]
    group = data[2] if len(data) > 2 else None
    booster = silvic.train(
        {**PARAMS, **objective},
        silvic.Dataset(X, label=y, group=group),
        ROUNDS,
        verbose_eval=False,
    )
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
def test_matches_booster_for_every_kernel_and_layout(name, device):
    booster, X = _train(name)
    kernels = serve.available_kernels() if device == "cpu" else [None]
    for kernel in kernels:
        predictor = serve.Predictor(booster, kernel=kernel, device=device)
        assert predictor.kernel == (kernel or "cuda")
        assert predictor.precision == "float64"
        assert predictor.num_trees() == booster.num_trees()
        assert predictor.num_feature() == NUM_FEATURES
        for data in _layouts(X):
            for raw_score in (False, True):
                expected = booster.predict(data, raw_score=raw_score)
                for num_threads in (1, 0, 3) if device == "cpu" else (1,):
                    got = predictor.predict(
                        data, raw_score=raw_score, num_threads=num_threads
                    )
                    assert got.dtype == np.float64
                    np.testing.assert_array_equal(
                        got, expected, err_msg=f"{kernel} {data.dtype} raw={raw_score}"
                    )


def test_num_iteration_matches_booster(device):
    booster, X = _train("binary")
    predictor = serve.Predictor(booster, device=device)
    for num_iteration in (None, 0, -1, 1, 7, ROUNDS, ROUNDS + 10):
        np.testing.assert_array_equal(
            predictor.predict(X, num_iteration=num_iteration),
            booster.predict(X, num_iteration=num_iteration),
        )


def test_model_sources_agree(tmp_path, device):
    booster, X = _train("regression")
    path = tmp_path / "model.txt"
    booster.save_model(path)
    expected = booster.predict(X)
    lgb_booster = lightgbm.Booster(model_file=str(path))
    for source in (booster, booster.model_to_string(), str(path), path, lgb_booster):
        np.testing.assert_array_equal(
            serve.Predictor(source, device=device).predict(X), expected
        )
    np.testing.assert_array_equal(lgb_booster.predict(X, num_threads=1), expected)


def test_binary_fixture_matches_current_host_lightgbm(device):
    fixtures = (
        Path(__file__).resolve().parents[2]
        / "crates/silvic-core/tests/fixtures/lightgbm"
    )
    case = next(
        case
        for case in json.loads((fixtures / "cases.json").read_text())["cases"]
        if case["name"] == "binary"
    )
    data = np.array(
        [[float(value) for value in row] for row in case["rows"]], dtype=np.float64
    )
    model_path = fixtures / case["model_file"]
    reference = lightgbm.Booster(model_file=str(model_path))
    predictor = serve.Predictor(model_path, device=device)
    for raw_score in (True, False):
        for prefix in (None, case["prefix_trees"]):
            got = predictor.predict(data, raw_score=raw_score, num_iteration=prefix)
            expected = reference.predict(
                data, raw_score=raw_score, num_iteration=prefix, num_threads=1
            )
            np.testing.assert_array_equal(got.view(np.uint64), expected.view(np.uint64))


def test_lightgbm_trained_model_matches_lightgbm(device):
    X, y = regression_data(rows=3000, seed=3)
    X[::9, 4] = np.nan
    categories = np.random.default_rng(3).integers(0, 12, size=len(X))
    X = np.column_stack([X, categories.astype(np.float64)])
    y = y + np.where(categories % 3 == 0, 1.0, -0.5)
    dataset = lightgbm.Dataset(
        X, y, categorical_feature=[NUM_FEATURES], params={"verbosity": -1}
    )
    lgb_booster = lightgbm.train(
        {**PARAMS, "objective": "regression"}, dataset, num_boost_round=ROUNDS
    )
    predictor = serve.Predictor(lgb_booster, device=device)
    np.testing.assert_array_equal(
        predictor.predict(X), lgb_booster.predict(X, num_threads=1)
    )
    np.testing.assert_array_equal(
        predictor.predict(X, raw_score=True), lgb_booster.predict(X, raw_score=True)
    )


def test_pandas_categories_map_as_at_training(device):
    rng = np.random.default_rng(5)
    rows = 3000
    colour = pd.Categorical(rng.choice(["red", "green", "blue", "teal"], size=rows))
    numeric = rng.normal(size=rows)
    effect = (
        pd.Series(colour)
        .map({"red": 3.0, "green": -1.0, "blue": 0.5, "teal": 0.0})
        .to_numpy(float)
    )
    frame = pd.DataFrame({"colour": colour, "x": numeric})
    y = effect + numeric + rng.normal(scale=0.1, size=rows)
    booster = silvic.train(
        {**PARAMS, "objective": "regression"},
        silvic.Dataset(frame, label=y),
        ROUNDS,
        verbose_eval=False,
    )
    predictor = serve.Predictor(booster, device=device)
    assert predictor.pandas_categorical == [["blue", "green", "red", "teal"]]
    reordered = frame.assign(
        colour=frame["colour"].cat.reorder_categories(["teal", "red", "green", "blue"])
    )
    for data in (frame, reordered):
        np.testing.assert_array_equal(predictor.predict(data), booster.predict(data))
    np.testing.assert_array_equal(
        predictor.predict(frame), booster.to_lightgbm().predict(frame)
    )
    missing = frame.assign(colour=frame["colour"].where(np.arange(rows) % 4 != 0))
    np.testing.assert_array_equal(predictor.predict(missing), booster.predict(missing))


def test_lleaves_style_model(tmp_path, device):
    booster, X = _train("binary")
    path = tmp_path / "model.txt"
    booster.save_model(path)
    model = serve.Model(model_file=path, device=device)
    model.compile()
    assert model.num_feature() == NUM_FEATURES
    assert model.num_trees() == ROUNDS
    assert model.num_model_per_iteration() == 1
    np.testing.assert_array_equal(model.predict(X, n_jobs=1), booster.predict(X))
    np.testing.assert_array_equal(model.predict(X), booster.predict(X))
    raw = serve.Model(model_str=booster.model_to_string(), device=device)
    raw.compile(cache="ignored.bin", raw_score=True, fblocksize=16, finline=False)
    np.testing.assert_array_equal(
        raw.predict(X, n_jobs=2 if device == "cpu" else 1),
        booster.predict(X, raw_score=True),
    )
    if device == "cuda":
        with pytest.raises(ValueError):
            model.predict(X, n_jobs=2)
    with pytest.raises(TypeError):
        serve.Model()
    with pytest.raises(TypeError):
        serve.Model(model_file=path, model_str=booster.model_to_string())


def test_errors():
    booster, X = _train("regression")
    predictor = serve.Predictor(booster)
    with pytest.raises(ValueError):
        predictor.predict(X[:, :-1])
    with pytest.raises(ValueError):
        serve.Predictor(booster, kernel="sse9")
    with pytest.raises(silvic.ModelError):
        serve.Predictor("tree\nversion=v3\nnum_class=1\n")
    with pytest.raises(TypeError):
        serve.Predictor(42)
    missing = [
        k
        for k in ("portable", "neon", "avx2", "avx512")
        if k not in serve.available_kernels()
    ]
    for kernel in missing:
        with pytest.raises(silvic.UnsupportedError):
            serve.Predictor(booster, kernel=kernel)


def test_device_selection_errors_without_cpu_fallback():
    booster, X = _train("regression")
    for options in (
        {"device": "metal"},
        {"device": "cpu", "device_id": 1},
        {"device": "cuda", "kernel": "portable"},
        {"device": "neuron", "precision": "float32", "kernel": "portable"},
    ):
        with pytest.raises(ValueError):
            serve.Predictor(booster, **options)
    if "cuda" not in serve.available_devices():
        with pytest.raises((silvic.DeviceError, silvic.UnsupportedError)):
            serve.Predictor(booster, device="cuda")
    else:
        predictor = serve.Predictor(booster, device="cuda")
        for threads in (0, 2):
            with pytest.raises(ValueError):
                predictor.predict(X, num_threads=threads)
    if "neuron" not in serve.available_devices():
        with pytest.raises((silvic.DeviceError, silvic.UnsupportedError)):
            serve.Predictor(booster, device="neuron", precision="float32")


def test_precision_selection_errors():
    booster, _ = _train("regression")
    for options in (
        {"precision": "float32"},
        {"precision": "float16"},
        {"device": "cuda", "precision": "float16"},
        {"device": "neuron"},
        {"device": "neuron", "precision": "float64"},
        {"device": "neuron", "precision": "float16"},
    ):
        with pytest.raises(ValueError):
            serve.Predictor(booster, **options)


@pytest.mark.parametrize("name", CASES)
def test_float32_device_layouts_prefixes_and_objectives(name, float32_device):
    booster, X = _train(name)
    predictor = serve.Predictor(booster, device=float32_device, precision="float32")
    assert predictor.precision == "float32"
    assert predictor.device == predictor.kernel == float32_device
    for threads in (0, 2):
        with pytest.raises(ValueError):
            predictor.predict(X, num_threads=threads)
    for data in _layouts(X):
        for raw_score in (True, False):
            for prefix in (None, 1, 7, ROUNDS + 10):
                got = predictor.predict(data, raw_score=raw_score, num_iteration=prefix)
                expected = booster.predict(
                    data.astype(np.float32), raw_score=raw_score, num_iteration=prefix
                )
                assert got.dtype == np.float32
                np.testing.assert_allclose(got, expected, atol=1e-5, rtol=1e-5)


def test_float32_device_accumulates_in_float32_not_output_cast(float32_device):
    booster, X = _train("regression")
    reference = booster.to_lightgbm()
    for tree in reference.dump_model()["tree_info"]:
        index = tree["tree_index"]
        value = (1e8, 1.0, -1e8)[index] if index < 3 else 0.0
        for leaf in range(tree["num_leaves"]):
            reference.set_leaf_output(index, leaf, value)
    data = np.ascontiguousarray(X[:33], dtype=np.float32)
    exact = serve.Predictor(reference)
    approximate = serve.Predictor(reference, device=float32_device, precision="float32")
    np.testing.assert_array_equal(exact.predict(data), np.ones(33, dtype=np.float64))
    got = approximate.predict(data)
    assert got.dtype == np.float32
    np.testing.assert_array_equal(got, np.zeros(33, dtype=np.float32))


def test_float32_model_wrapper(float32_device):
    booster, X = _train("binary")
    model = serve.Model(
        model_str=booster.model_to_string(), device=float32_device, precision="float32"
    )
    for raw_score in (False, True):
        model.compile(raw_score=raw_score)
        expected = booster.predict(X.astype(np.float32), raw_score=raw_score)
        for n_jobs in (None, 1):
            got = model.predict(X, n_jobs=n_jobs)
            assert got.dtype == np.float32
            np.testing.assert_allclose(got, expected, atol=1e-5, rtol=1e-5)
    for n_jobs in (0, 2):
        with pytest.raises(ValueError):
            model.predict(X, n_jobs=n_jobs)


def test_threads_score_concurrently_without_the_gil(device):
    booster, X = _train("lambdarank")
    predictor = serve.Predictor(booster, device=device)
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
