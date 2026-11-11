"""Compares ``silvic.ray.train`` on a local Ray cluster against single-process ``silvic.train`` on the same rows."""

import threading
import time

import numpy as np
import pandas as pd
import pytest
from conftest import NUM_FEATURES, group_sizes, ranking_frame

import silvic

ray = pytest.importorskip("ray")
silvic_ray = pytest.importorskip("silvic.ray")

FEATURES = [f"f{i}" for i in range(NUM_FEATURES)]

PARAMS = {
    "objective": "lambdarank",
    "num_leaves": 15,
    "min_data_in_leaf": 5,
    "bagging_fraction": 0.8,
    "bagging_freq": 2,
    "feature_fraction": 0.9,
    "eval_at": [5],
    # Every row in the bin-mapper sample, so a rank's sample equals the single-process one.
    "bin_construct_sample_cnt": 1_000_000,
    "verbosity": -1,
}

ROUNDS = 15


@pytest.fixture(scope="module")
def local_ray():
    ray.init(num_cpus=6, include_dashboard=False, log_to_driver=False)
    yield
    ray.shutdown()


def _to_ray(frame, blocks=4):
    """``frame`` as a Ray dataset of ``blocks`` pandas blocks."""
    bounds = np.linspace(0, len(frame), blocks + 1).astype(int)
    return ray.data.from_pandas([frame.iloc[a:b] for a, b in zip(bounds[:-1], bounds[1:])])


def _single_process(frame, num_workers, params):
    """``silvic.train`` on the workers' rows concatenated in rank order."""
    parts = silvic_ray.ordered_partitions(frame, num_workers, label_column="label", group_column="qid")
    rows = pd.concat(parts, ignore_index=True)
    assert len(rows) == len(frame)
    train_set = silvic.Dataset(rows[FEATURES], label=rows["label"], group=group_sizes(rows["qid"]))
    return silvic.train(params, train_set, ROUNDS, verbose_eval=False)


def test_ordered_partitions_hold_whole_groups():
    frame = ranking_frame()
    parts = silvic_ray.ordered_partitions(frame, 3, label_column="label", group_column="qid")
    assert sum(len(p) for p in parts) == len(frame)
    seen = [set(p["qid"]) for p in parts]
    assert not (seen[0] & seen[1] or seen[0] & seen[2] or seen[1] & seen[2])
    for part in parts:
        assert part["qid"].is_monotonic_increasing
    # The order depends on the rows, not on how they arrive.
    again = silvic_ray.ordered_partitions(
        frame.sample(frac=1.0, random_state=9), 3, label_column="label", group_column="qid"
    )
    for a, b in zip(parts, again):
        pd.testing.assert_frame_equal(a, b)


@pytest.mark.parametrize("num_workers", [2, 3])
def test_distributed_model_equals_single_process(local_ray, num_workers):
    frame = ranking_frame()
    valid = ranking_frame(num_queries=40, seed=11)
    evals = {}
    distributed = silvic_ray.train(
        PARAMS,
        _to_ray(frame),
        label_column="label",
        group_column="qid",
        valid_datasets={"valid": _to_ray(valid, blocks=2)},
        num_boost_round=ROUNDS,
        num_workers=num_workers,
        evals_result=evals,
        verbose_eval=False,
        timeout_s=120,
    )
    single = _single_process(frame, num_workers, PARAMS)
    assert distributed.model_to_string() == single.model_to_string()
    assert distributed.feature_name() == FEATURES
    assert len(evals["valid"]["ndcg@5"]) == ROUNDS


@pytest.mark.skipif("metal" not in silvic.available_devices(), reason="no Metal device")
def test_metal_workers(local_ray):
    frame = ranking_frame()
    params = {**PARAMS, "device_type": "metal"}
    distributed = silvic_ray.train(
        params,
        _to_ray(frame),
        label_column="label",
        group_column="qid",
        num_boost_round=ROUNDS,
        num_workers=2,
        verbose_eval=False,
        timeout_s=120,
    )
    assert distributed.model_to_string() == _single_process(frame, 2, params).model_to_string()


def test_worker_failure_raises(local_ray):
    frame = ranking_frame()
    outcome = {}

    def run():
        try:
            outcome["booster"] = silvic_ray.train(
                PARAMS,
                _to_ray(frame),
                label_column="label",
                group_column="qid",
                num_boost_round=1_000_000,
                num_workers=2,
                verbose_eval=False,
                timeout_s=60,
            )
        except BaseException as error:  # noqa: BLE001 - the test inspects whatever surfaces
            outcome["error"] = error

    driver = threading.Thread(target=run, daemon=True)
    driver.start()
    deadline = time.monotonic() + 60
    names = []
    while time.monotonic() < deadline and len(names) < 2:
        names = sorted(n for n in ray.util.list_named_actors() if n.startswith("silvic-"))
        time.sleep(0.2)
    assert len(names) == 2, "workers never started"
    # Let training get under way before killing rank 1.
    time.sleep(3)
    ray.kill(ray.get_actor(names[1]))
    driver.join(timeout=60)
    assert not driver.is_alive(), "training hung after a worker died"
    assert "error" in outcome
    assert isinstance(outcome["error"], (ray.exceptions.RayActorError, silvic.CollectiveError))
    deadline = time.monotonic() + 30
    while [n for n in ray.util.list_named_actors() if n.startswith("silvic-")] and time.monotonic() < deadline:
        time.sleep(0.2)
    assert not [n for n in ray.util.list_named_actors() if n.startswith("silvic-")], "workers were not killed"
