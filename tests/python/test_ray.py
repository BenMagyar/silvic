"""Compares ``silvic.ray.train`` on local Ray clusters against single-process ``silvic.train`` on the same rows, and
against itself uninterrupted after workers and nodes are lost mid-training."""

import contextlib
import threading
import time

import lightgbm
import numpy as np
import pandas as pd
import pytest
from conftest import NUM_FEATURES, group_sizes, ranking_frame

import silvic

ray = pytest.importorskip("ray")
silvic_ray = pytest.importorskip("silvic.ray")
cluster_utils = pytest.importorskip("ray.cluster_utils")

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

# Fault tests: enough iterations, slowed down, to fail a worker after the first checkpoints and before the end.
FAULT_ROUNDS = 40
FAULT_PERIOD = 5
FAULT_DELAY_S = 0.05

WORKER_LOST = "[silvic] [Warning] worker lost on "


@pytest.fixture(scope="module", autouse=True)
def _shutdown_ray():
    yield
    if ray.is_initialized():
        ray.shutdown()


@pytest.fixture
def local_ray():
    if not ray.is_initialized():
        ray.init(
            address="local", num_cpus=6, include_dashboard=False, log_to_driver=False
        )
    yield


@pytest.fixture
def spot_cluster():
    """A head node without CPUs and three worker nodes with one ``slot`` each, as spot nodes would hold one worker."""
    if ray.is_initialized():
        ray.shutdown()
    cluster = cluster_utils.Cluster(
        initialize_head=True, head_node_args={"num_cpus": 0}
    )
    nodes = [cluster.add_node(num_cpus=2, resources={"slot": 1}) for _ in range(3)]
    ray.init(address=cluster.address, log_to_driver=False)
    cluster.wait_for_nodes()
    yield cluster, nodes
    ray.shutdown()
    cluster.shutdown()


def _to_ray(frame, blocks=4):
    """``frame`` as a Ray dataset of ``blocks`` pandas blocks."""
    bounds = np.linspace(0, len(frame), blocks + 1).astype(int)
    return ray.data.from_pandas(
        [frame.iloc[a:b] for a, b in zip(bounds[:-1], bounds[1:])]
    )


def _single_process(frame, num_workers, params):
    """``silvic.train`` on the workers' rows concatenated in rank order."""
    parts = silvic_ray.ordered_partitions(
        frame, num_workers, label_column="label", group_column="qid"
    )
    rows = pd.concat(parts, ignore_index=True)
    assert len(rows) == len(frame)
    train_set = silvic.Dataset(
        rows[FEATURES], label=rows["label"], group=group_sizes(rows["qid"])
    )
    return silvic.train(params, train_set, ROUNDS, verbose_eval=False)


def test_ordered_partitions_hold_whole_groups():
    frame = ranking_frame()
    parts = silvic_ray.ordered_partitions(
        frame, 3, label_column="label", group_column="qid"
    )
    assert sum(len(p) for p in parts) == len(frame)
    seen = [set(p["qid"]) for p in parts]
    assert not (seen[0] & seen[1] or seen[0] & seen[2] or seen[1] & seen[2])
    for part in parts:
        assert part["qid"].is_monotonic_increasing
    # The order depends on the rows, not on how they arrive.
    again = silvic_ray.ordered_partitions(
        frame.sample(frac=1.0, random_state=9),
        3,
        label_column="label",
        group_column="qid",
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
    assert (
        distributed.model_to_string()
        == _single_process(frame, 2, params).model_to_string()
    )


def _train(frame, **kwargs):
    options = {
        "label_column": "label",
        "group_column": "qid",
        "num_boost_round": FAULT_ROUNDS,
        "checkpoint_period": FAULT_PERIOD,
        "verbose_eval": False,
        "timeout_s": 60,
        **kwargs,
    }
    return silvic_ray.train(options.pop("params", PARAMS), _to_ray(frame), **options)


def _silvic_actors(part):
    return sorted(
        name
        for name in ray.util.list_named_actors()
        if name.startswith("silvic-") and part in name
    )


def _wait_for_checkpoint(iteration, timeout_s=60):
    """Waits until the running job's checkpoint keeper holds a checkpoint at or after ``iteration``."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        for name in _silvic_actors("-keeper"):
            with contextlib.suppress(Exception):
                if (
                    ray.get(ray.get_actor(name).latest_iteration.remote(), timeout=5)
                    >= iteration
                ):
                    return
        time.sleep(0.05)
    raise AssertionError(f"no checkpoint at iteration {iteration} within {timeout_s} s")


def _wait_for_workers(attempt, count, timeout_s=60):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        names = _silvic_actors(f"-a{attempt}-r")
        if len(names) == count:
            return names
        time.sleep(0.05)
    raise AssertionError(f"attempt {attempt} never started {count} workers")


def _while_training(train, fault):
    """Runs ``train`` on a thread and ``fault`` on this one (Ray's ``Cluster.add_node`` needs the main thread);
    returns what ``train`` returns or raises what it raises."""
    outcome = {}

    def run():
        try:
            outcome["result"] = train()
        except BaseException as error:  # noqa: BLE001 - re-raised on the test thread
            outcome["error"] = error

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    fault()
    thread.join(timeout=180)
    assert not thread.is_alive(), "training hung after the fault"
    if "error" in outcome:
        raise outcome["error"]
    return outcome["result"]


def _assert_no_actors_left():
    deadline = time.monotonic() + 30
    while _silvic_actors("") and time.monotonic() < deadline:
        time.sleep(0.2)
    assert not _silvic_actors(""), "silvic actors outlived the job"


def test_killed_worker_resumes_to_the_uninterrupted_model(
    local_ray, monkeypatch, capsys
):
    frame = ranking_frame()
    valid = {"valid": _to_ray(ranking_frame(num_queries=40, seed=11), blocks=2)}
    expected_evals = {}
    expected = _train(
        frame, num_workers=2, valid_datasets=valid, evals_result=expected_evals
    )
    monkeypatch.setattr(silvic_ray, "_TEST_ITERATION_DELAY_S", FAULT_DELAY_S)

    def kill_rank_one():
        _wait_for_checkpoint(2 * FAULT_PERIOD - 1)
        ray.kill(ray.get_actor(_wait_for_workers(1, 2)[1]))

    evals = {}
    resumed = _while_training(
        lambda: _train(frame, num_workers=2, valid_datasets=valid, evals_result=evals),
        kill_rank_one,
    )
    output = capsys.readouterr().out
    assert WORKER_LOST in output
    assert "with 2 workers (attempt 2 of 4)" in output
    assert resumed.model_to_string() == expected.model_to_string()
    assert evals == expected_evals
    assert len(evals["valid"]["ndcg@5"]) == FAULT_ROUNDS
    _assert_no_actors_left()


def test_failures_past_max_failures_raise(local_ray, monkeypatch, capsys):
    monkeypatch.setattr(silvic_ray, "_TEST_ITERATION_DELAY_S", FAULT_DELAY_S)

    def kill_every_attempt():
        for attempt in (1, 2):
            names = _wait_for_workers(attempt, 2)
            time.sleep(1.0)
            ray.kill(ray.get_actor(names[1]))

    with pytest.raises(
        (ray.exceptions.RayActorError, silvic.CollectiveError)
    ) as caught:
        _while_training(
            lambda: _train(
                ranking_frame(), num_workers=2, num_boost_round=100_000, max_failures=1
            ),
            kill_every_attempt,
        )
    assert "silvic.ray.train gave up after 2 attempts" in getattr(
        caught.value, "__notes__", []
    )
    assert capsys.readouterr().out.count(WORKER_LOST) == 1
    _assert_no_actors_left()


def test_config_errors_are_not_retried(local_ray, capsys):
    with pytest.raises(silvic.ConfigError, match="not_a_param"):
        _train(ranking_frame(), num_workers=2, params={**PARAMS, "not_a_param": 1})
    assert not _silvic_actors("")
    if "cuda" in silvic.available_devices():
        return
    # Every worker raises the missing device; that is the job's fault, not a lost worker.
    with pytest.raises(silvic.DeviceError):
        _train(ranking_frame(), num_workers=2, params={**PARAMS, "device_type": "cuda"})
    assert WORKER_LOST not in capsys.readouterr().out
    _assert_no_actors_left()


@pytest.mark.parametrize("device_key", ["device_type", "device"])
@pytest.mark.parametrize(
    ("use_gpu", "resources"),
    [
        (True, None),
        (True, {"CPU": 1, "GPU": 0}),
        (False, {"CPU": 1, "GPU": 1}),
        (False, {"GPU": 0.5, "neuron_cores": 1}),
    ],
)
def test_neuron_rejects_gpu_requests_before_reading_data(
    device_key, use_gpu, resources
):
    with pytest.raises(
        ValueError, match="cannot be combined with use_gpu or GPU resources"
    ):
        silvic_ray.train(
            {device_key: "neuron"},
            None,
            label_column="label",
            use_gpu=use_gpu,
            resources_per_worker=resources,
        )


@pytest.mark.parametrize(
    "cores", [0, -1, 0.5, 1.5, float("nan"), float("inf"), True, False, "1", None]
)
def test_neuron_rejects_invalid_core_reservations_before_reading_data(cores):
    with pytest.raises(
        ValueError, match="neuron_cores must be a positive whole number"
    ):
        silvic_ray.train(
            {"device_type": "neuron"},
            None,
            label_column="label",
            resources_per_worker={"CPU": 1, "neuron_cores": cores},
        )


@pytest.mark.parametrize(
    "params", [{}, {"device_type": "auto"}, {"device_type": "cpu"}, {"device": "cuda"}]
)
def test_neuron_resources_require_explicit_neuron_training_before_reading_data(params):
    with pytest.raises(
        ValueError, match='neuron_cores resources require explicit device_type="neuron"'
    ):
        silvic_ray.train(
            params,
            None,
            label_column="label",
            resources_per_worker={"CPU": 1, "neuron_cores": 1},
        )


SPOT = {"num_workers": 3, "resources_per_worker": {"CPU": 1, "slot": 1}}


def test_spot_node_loss_shrinks_to_min_workers(spot_cluster, monkeypatch, capsys):
    cluster, nodes = spot_cluster
    monkeypatch.setattr(silvic_ray, "_TEST_ITERATION_DELAY_S", FAULT_DELAY_S)

    def preempt():
        _wait_for_checkpoint(2 * FAULT_PERIOD - 1)
        cluster.remove_node(nodes[2])

    frame = ranking_frame()
    booster = _while_training(
        lambda: _train(frame, **SPOT, min_workers=2, replacement_timeout_s=3), preempt
    )
    output = capsys.readouterr().out
    assert WORKER_LOST in output
    assert "continuing on 2 workers instead of 3" in output
    assert "with 2 workers (attempt 2 of 4)" in output
    assert booster.num_trees() == FAULT_ROUNDS
    prediction = booster.predict(frame[FEATURES])
    assert np.all(np.isfinite(prediction))
    reference = lightgbm.Booster(model_str=booster.model_to_string())
    np.testing.assert_array_equal(prediction, reference.predict(frame[FEATURES]))


def test_spot_node_replacement_resumes_to_the_uninterrupted_model(
    spot_cluster, monkeypatch, capsys
):
    cluster, nodes = spot_cluster
    frame = ranking_frame()
    expected = _train(frame, **SPOT)
    monkeypatch.setattr(silvic_ray, "_TEST_ITERATION_DELAY_S", FAULT_DELAY_S)

    def preempt_and_replace():
        _wait_for_checkpoint(2 * FAULT_PERIOD - 1)
        cluster.remove_node(nodes[2])
        cluster.add_node(num_cpus=2, resources={"slot": 1})

    resumed = _while_training(
        lambda: _train(frame, **SPOT, replacement_timeout_s=60), preempt_and_replace
    )
    output = capsys.readouterr().out
    assert WORKER_LOST in output
    assert "with 3 workers (attempt " in output
    assert resumed.model_to_string() == expected.model_to_string()
