"""Trains on synthetic data and compares predictions against LightGBM 4.x loading the same model text."""

import pickle

import lightgbm
import numpy as np
import pandas as pd
import pytest
from conftest import NUM_FEATURES, binary_data, ranking_data, regression_data

import silvic

ROUNDS = 20

BASE_PARAMS = {"num_leaves": 15, "min_data_in_leaf": 10, "verbosity": -1}

CASES = {
    "regression": ({"objective": "regression"}, regression_data),
    "binary": ({"objective": "binary"}, binary_data),
    "lambdarank": ({"objective": "lambdarank"}, ranking_data),
}


def _train_case(name, **extra):
    objective, make = CASES[name]
    data = make()
    X, y = data[0], data[1]
    group = data[2] if len(data) > 2 else None
    params = {**BASE_PARAMS, **objective, **extra}
    train_set = silvic.Dataset(X, label=y, group=group)
    booster = silvic.train(params, train_set, num_boost_round=ROUNDS, verbose_eval=False)
    return booster, X, y


def test_regression_learns(regression):
    X, y = regression
    booster = silvic.train(
        {**BASE_PARAMS, "objective": "regression"}, silvic.Dataset(X, label=y), ROUNDS, verbose_eval=False
    )
    assert booster.num_trees() == ROUNDS
    prediction = booster.predict(X)
    assert prediction.shape == (len(y),)
    assert np.mean((prediction - y) ** 2) < 0.5 * np.var(y)


def test_binary_probabilities(binary):
    X, y = binary
    params = {**BASE_PARAMS, "objective": "binary"}
    booster = silvic.train(params, silvic.Dataset(X, label=y), ROUNDS, verbose_eval=False)
    probability = booster.predict(X)
    raw = booster.predict(X, raw_score=True)
    assert np.all((probability > 0) & (probability < 1))
    np.testing.assert_allclose(probability, 1.0 / (1.0 + np.exp(-raw)), rtol=1e-12)
    assert np.mean((probability > 0.5) == (y > 0.5)) > 0.8


def test_lambdarank_ranks(ranking):
    X, y, group = ranking
    evals = {}
    train_set = silvic.Dataset(X, label=y, group=group)
    booster = silvic.train(
        {**BASE_PARAMS, "objective": "lambdarank", "eval_at": [3]},
        train_set,
        ROUNDS,
        valid_sets=[train_set],
        evals_result=evals,
        verbose_eval=False,
    )
    ndcg = evals["training"]["ndcg@3"]
    assert len(ndcg) == ROUNDS
    assert ndcg[-1] > ndcg[0]
    assert np.all(np.isfinite(booster.predict(X)))


@pytest.mark.parametrize("name", sorted(CASES))
def test_predict_equals_lightgbm(name):
    booster, X, _ = _train_case(name, bagging_fraction=0.8, bagging_freq=2, feature_fraction=0.9)
    reference = lightgbm.Booster(model_str=booster.model_to_string())
    assert reference.num_trees() == booster.num_trees()
    np.testing.assert_array_equal(booster.predict(X), reference.predict(X))
    np.testing.assert_array_equal(booster.predict(X, raw_score=True), reference.predict(X, raw_score=True))
    np.testing.assert_array_equal(
        booster.predict(X, num_iteration=5), reference.predict(X, num_iteration=5)
    )
    np.testing.assert_array_equal(booster.to_lightgbm().predict(X), reference.predict(X))


def test_training_is_reproducible():
    first, _, _ = _train_case("lambdarank", bagging_fraction=0.7, bagging_freq=1)
    second, _, _ = _train_case("lambdarank", bagging_fraction=0.7, bagging_freq=1)
    assert first.model_to_string() == second.model_to_string()


def test_pickle_round_trip(tmp_path):
    booster, X, _ = _train_case("binary")
    restored = pickle.loads(pickle.dumps(booster))
    assert restored.model_to_string() == booster.model_to_string()
    np.testing.assert_array_equal(restored.predict(X), booster.predict(X))
    path = tmp_path / "model.txt"
    booster.save_model(path)
    from_file = silvic.Booster(model_file=path)
    assert from_file.model_to_string() == booster.model_to_string()


def test_pandas_categorical():
    rng = np.random.default_rng(5)
    rows = 3000
    colour = pd.Categorical(rng.choice(["red", "green", "blue", "teal"], size=rows))
    numeric = rng.normal(size=rows)
    effect = pd.Series(colour).map({"red": 3.0, "green": -1.0, "blue": 0.5, "teal": 0.0}).to_numpy(float)
    frame = pd.DataFrame({"colour": colour, "x": numeric})
    y = effect + numeric + rng.normal(scale=0.1, size=rows)
    booster = silvic.train(
        {**BASE_PARAMS, "objective": "regression"}, silvic.Dataset(frame, label=y), ROUNDS, verbose_eval=False
    )
    model_str = booster.model_to_string()
    assert model_str.rstrip("\n").splitlines()[-1] == 'pandas_categorical:[["blue", "green", "red", "teal"]]'
    assert booster.feature_name() == ["colour", "x"]
    assert booster.pandas_categorical == [["blue", "green", "red", "teal"]]
    # Categories listed in another order map back to the training codes.
    reordered = frame.assign(colour=frame["colour"].cat.reorder_categories(["teal", "red", "green", "blue"]))
    np.testing.assert_array_equal(booster.predict(reordered), booster.predict(frame))
    np.testing.assert_array_equal(booster.predict(frame), booster.to_lightgbm().predict(frame))
    assert np.mean((booster.predict(frame) - y) ** 2) < 0.1 * np.var(y)


def test_numpy_model_ends_with_null_pandas_categorical(regression):
    X, y = regression
    booster = silvic.train({**BASE_PARAMS}, silvic.Dataset(X, label=y), 2, verbose_eval=False)
    assert booster.model_to_string().endswith("\npandas_categorical:null\n")


def test_sklearn_regressor_and_classifier(regression, binary):
    X, y = regression
    regressor = silvic.SilvicRegressor(n_estimators=ROUNDS, num_leaves=15, verbosity=-1).fit(X, y)
    assert regressor.n_features_in_ == NUM_FEATURES
    assert regressor.feature_importances_.shape == (NUM_FEATURES,)
    assert regressor.feature_importances_.dtype == np.int32
    assert regressor.booster_.num_trees() == ROUNDS
    assert np.mean((regressor.predict(X) - y) ** 2) < 0.5 * np.var(y)

    X, y = binary
    labels = np.where(y > 0.5, "yes", "no")
    classifier = silvic.SilvicClassifier(n_estimators=ROUNDS, verbosity=-1, subsample=0.8, bagging_fraction=0.7)
    classifier.fit(X, labels, eval_set=[(X, labels)])
    assert list(classifier.classes_) == ["no", "yes"]
    proba = classifier.predict_proba(X)
    assert proba.shape == (len(y), 2)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0)
    assert set(classifier.predict(X)) <= {"no", "yes"}
    assert "binary_logloss" in classifier.evals_result_["training"]
    assert classifier.booster_.params["bagging_fraction"] == 0.7
    assert "subsample" not in classifier.booster_.params


def test_sklearn_ranker_with_groups(ranking):
    X, y, group = ranking
    ranker = silvic.SilvicRanker(n_estimators=ROUNDS, num_leaves=15, min_child_samples=10, verbosity=-1)
    ranker.fit(X, y, group=group, eval_set=[(X, y)], eval_group=[group], eval_at=[1, 5], eval_names=["train"])
    assert sorted(ranker.evals_result_["train"]) == ["ndcg@1", "ndcg@5"]
    assert len(ranker.evals_result_["train"]["ndcg@5"]) == ROUNDS
    assert ranker.best_score_["train"]["ndcg@5"] == ranker.evals_result_["train"]["ndcg@5"][-1]
    assert ranker.predict(X).shape == (len(y),)
    reference = lightgbm.Booster(model_str=ranker.booster_.model_to_string())
    np.testing.assert_array_equal(ranker.predict(X), reference.predict(X))


def test_sklearn_clone_keeps_params():
    sklearn_base = pytest.importorskip("sklearn.base")
    ranker = silvic.SilvicRanker(n_estimators=7, lambdarank_truncation_level=10)
    clone = sklearn_base.clone(ranker)
    assert clone.get_params() == ranker.get_params()
    assert clone.get_params()["lambdarank_truncation_level"] == 10


def test_record_evaluation_and_log_evaluation(regression, capsys):
    X, y = regression
    train_set = silvic.Dataset(X[:1500], label=y[:1500])
    valid_set = silvic.Dataset(X[1500:], label=y[1500:], reference=train_set)
    recorded = {}
    silvic.train(
        {**BASE_PARAMS, "metric": ["l2", "rmse"]},
        train_set,
        num_boost_round=4,
        valid_sets=[valid_set],
        valid_names=["holdout"],
        callbacks=[silvic.record_evaluation(recorded), silvic.log_evaluation(2)],
    )
    assert list(recorded) == ["holdout"]
    assert sorted(recorded["holdout"]) == ["l2", "rmse"]
    assert len(recorded["holdout"]["l2"]) == 4
    np.testing.assert_allclose(np.sqrt(recorded["holdout"]["l2"]), recorded["holdout"]["rmse"], rtol=1e-9)
    lines = [line for line in capsys.readouterr().out.splitlines() if line.startswith("[")]
    assert [line.split("\t")[0] for line in lines] == ["[2]", "[4]"]
    assert lines[0].split("\t")[1].startswith("holdout's l2: ")


def test_params_win_over_num_boost_round(regression):
    X, y = regression
    booster = silvic.train({**BASE_PARAMS, "n_estimators": 3}, silvic.Dataset(X, label=y), 50, verbose_eval=False)
    assert booster.num_trees() == 3


def test_unknown_param_raises_value_error(regression):
    X, y = regression
    with pytest.raises(ValueError, match="not_a_param") as caught:
        silvic.train({"not_a_param": 1}, silvic.Dataset(X, label=y), 2, verbose_eval=False)
    assert isinstance(caught.value, silvic.SilvicError)
    assert isinstance(caught.value, silvic.ConfigError)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"fobj": lambda preds, data: (preds, preds)},
        {"feval": lambda preds, data: ("x", 0.0, True)},
        {"init_model": "model.txt"},
        {"early_stopping_rounds": 5},
        {"callbacks": [lightgbm.early_stopping(5)]},
    ],
)
def test_unsupported_training_options_raise(regression, kwargs):
    X, y = regression
    with pytest.raises(ValueError, match="does not support"):
        silvic.train(dict(BASE_PARAMS), silvic.Dataset(X, label=y), 2, verbose_eval=False, **kwargs)


def test_early_stopping_param_raises(regression):
    X, y = regression
    with pytest.raises(ValueError, match="early stopping"):
        silvic.train({**BASE_PARAMS, "early_stopping_round": 3}, silvic.Dataset(X, label=y), 2)


def test_callback_exception_propagates(regression):
    X, y = regression

    class Stop(Exception):
        pass

    def stop_at_three(env):
        if env.iteration == 2:
            raise Stop

    with pytest.raises(Stop):
        silvic.train(dict(BASE_PARAMS), silvic.Dataset(X, label=y), 10, callbacks=[stop_at_three])


def test_available_devices_lists_cpu():
    assert "cpu" in silvic.available_devices()


@pytest.mark.skipif("metal" not in silvic.available_devices(), reason="no Metal device")
def test_metal_matches_cpu():
    # L2 gradients are exact in f32, so Metal trains the CPU's model bit for bit. Binary and lambdarank gradients
    # are f64 formulas on the CPU and f32 on Metal (no f64 there), so those models agree to rounding only.
    metal, X, _ = _train_case("regression", device_type="metal", bagging_fraction=0.8, bagging_freq=1)
    cpu, _, _ = _train_case("regression", device_type="cpu", bagging_fraction=0.8, bagging_freq=1)
    assert metal.model_to_string() == cpu.model_to_string()
    metal, X, _ = _train_case("lambdarank", device_type="metal", bagging_fraction=0.8, bagging_freq=1)
    cpu, _, _ = _train_case("lambdarank", device_type="cpu", bagging_fraction=0.8, bagging_freq=1)
    assert metal.num_trees() == cpu.num_trees() == ROUNDS
    np.testing.assert_allclose(metal.predict(X), cpu.predict(X), rtol=1e-3, atol=1e-4)


def test_missing_device_raises_runtime_error():
    if "cuda" in silvic.available_devices():
        pytest.skip("CUDA is present")
    X, y = regression_data(rows=200)
    with pytest.raises(RuntimeError, match="cuda"):
        silvic.train({**BASE_PARAMS, "device_type": "cuda"}, silvic.Dataset(X, label=y), 2, verbose_eval=False)
