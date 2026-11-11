"""Synthetic datasets shared by the Python tests."""

import numpy as np
import pandas as pd
import pytest

NUM_FEATURES = 8


def regression_data(rows=2000, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(rows, NUM_FEATURES))
    X[rng.random(size=X.shape) < 0.02] = np.nan
    y = np.nan_to_num(X[:, 0]) * 2.0 - np.nan_to_num(X[:, 1]) + 0.5 * np.nan_to_num(X[:, 2]) ** 2
    y += rng.normal(scale=0.1, size=rows)
    return X, y


def binary_data(rows=2000, seed=1):
    X, y = regression_data(rows, seed)
    return X, (y > np.median(y)).astype(np.float64)


def ranking_frame(num_queries=150, seed=2):
    """A lambdarank DataFrame: `qid`, `label` (0-4) and float32 features `f0`.., shuffled across queries."""
    rng = np.random.default_rng(seed)
    sizes = rng.integers(3, 25, size=num_queries)
    qid = np.repeat(np.arange(num_queries) * 7 + 3, sizes)
    rows = len(qid)
    features = rng.normal(size=(rows, NUM_FEATURES)).astype(np.float32)
    relevance = features[:, 0] + 0.5 * features[:, 1] + rng.normal(scale=0.5, size=rows)
    label = np.clip(np.round(relevance + 1.5), 0, 4)
    frame = pd.DataFrame(features, columns=[f"f{i}" for i in range(NUM_FEATURES)])
    frame["qid"] = qid
    frame["label"] = label
    return frame.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def group_sizes(qid):
    """Run lengths of consecutive equal values."""
    qid = np.asarray(qid)
    starts = np.flatnonzero(qid[1:] != qid[:-1]) + 1
    return np.diff(np.concatenate(([0], starts, [len(qid)])))


def ranking_data(num_queries=150, seed=2):
    """``(X, y, group)`` with query rows contiguous."""
    frame = ranking_frame(num_queries, seed).sort_values("qid", kind="stable")
    X = frame[[f"f{i}" for i in range(NUM_FEATURES)]].to_numpy()
    return X, frame["label"].to_numpy(), group_sizes(frame["qid"])


@pytest.fixture
def regression():
    return regression_data()


@pytest.fixture
def binary():
    return binary_data()


@pytest.fixture
def ranking():
    return ranking_data()
