"""Distributed training on Ray: one model trained by ``num_workers`` actors, each holding whole query groups.

``train`` takes ``ray.data.Dataset`` inputs (``ray.data.read_parquet("s3://...")`` and friends) and returns the
``silvic.Booster`` of rank 0. The model equals the one ``silvic.train`` trains in one process on the concatenation of
the workers' rows in rank order (``ordered_partitions`` returns those rows), provided ``bin_construct_sample_cnt`` is
at least the number of rows, so both draw the bin mappers from every row.

Partitioning
------------
Rows go to rank ``hash(group key) % num_workers`` (``pandas.util.hash_array`` over the key's values, normalised to
int64, float64 or str so the hash does not depend on how a block typed them); without a group column the hash covers
every used column of the row. Each rank's rows are then sorted by group key, every feature, the label and the weight.
That order depends only on the data and ``num_workers``, never on block boundaries or task timing, which matters
because silvic is bit-reproducible and lambdarank breaks score ties by row order, as LightGBM does.

The exchange is a map-side hash split on Ray core: the dataset streams through ``iter_internal_ref_bundles`` (one
read of the input), a task per block splits it into ``num_workers`` Arrow tables (``num_returns=num_workers``), and
each actor fetches only its own pieces. No worker reads the full dataset, and every row moves once from the reading
node to its actor. The alternatives in Ray Data 2.58 do not fit as well:

- ``Dataset.repartition(n, keys=[...])`` hash-partitions only under the hash shuffle strategy, numbers partitions by
  Ray's own hash rather than by rank, and does not say which output block is which partition, so the blocks would
  still have to be inspected and re-routed.
- ``groupby(...).map_groups`` shuffles the same way and coalesces groups into blocks with the same addressing
  problem, plus a Python call per query group.
- ``to_arrow_refs`` materialises the whole input in the object store before anything is split.
- ``split``/``streaming_split`` divide blocks, not query groups: a query spanning two blocks would land on two ranks.

Training
--------
Actors run in a placement group (``placement_strategy``, one bundle of ``resources_per_worker`` per actor). Each binds
a TCP listener on its node IP; the driver gathers the addresses and hands every actor the list and a random 64-bit
job token, and the actors bin their data (bin mappers from a sample allgathered over every rank) and train
concurrently, reducing histograms over the TCP ring.

``use_gpu=True`` reserves one Ray GPU per actor, which CUDA training uses; ``device_type`` in ``params`` picks the
device as in ``silvic.train`` (``auto`` takes CUDA, then Metal, then the CPU). Metal needs no Ray GPU resource.
``timeout_s`` bounds the first placement, the TCP rendezvous, and how long a collective waits on a silent peer.

Failures and preemption
-----------------------
Every ``checkpoint_period`` iterations and after the last, rank 0 sends the model so far and the evaluation results
since the previous checkpoint to a checkpoint keeper: a zero-resource actor pinned to the driver's node, so it
outlives worker nodes. The sends are fire-and-forget; the model text travels once per checkpoint.

A worker failure is the loss of a worker: a dead actor or node (``RayActorError``, ``NodeDiedError``,
``WorkerCrashedError``), a lost object (``ObjectLostError``), or the ``CollectiveError`` the surviving ranks raise
when a peer vanishes. On one, the driver kills every actor, takes the latest checkpoint, places ``num_workers``
bundles again (Ray reschedules the bundles of a dead node once capacity returns, e.g. when KubeRay replaces a spot
node), and restarts every rank, which re-loads its pieces, re-bins and trains the remaining trees from the
checkpoint (``silvic.train``'s ``init_model``). Partitioning, binning, bagging and quantization depend only on the
data, the worker count and the absolute iteration, so a resumed job ends with the model an uninterrupted job trains,
byte for byte. Partition pieces are reused across attempts; if a piece is lost and Ray cannot reconstruct it, the
next attempt partitions the dataset again (Ray datasets are lazy and re-readable). Each resume prints
``[silvic] [Warning] worker lost on <node> (<error>); resuming from iteration k with N workers (attempt i of max)``.
After ``max_failures`` resumes, the next worker failure is raised, with a note giving the number of attempts.

Any other error (bad parameters or data, a missing device, an exception in user code) is raised at once, never
retried. When ranks fail differently, the root error is raised: such an error first, then a lost worker, then the
collective errors its peers see as a consequence.

When ``min_workers < num_workers`` and the full group cannot be placed within ``replacement_timeout_s`` after a
failure, the job shrinks one worker at a time (each count waiting ``replacement_timeout_s``) down to
``min_workers``, repartitions the rows for the new count and continues from the checkpoint. The rows then sit on
different ranks than before, so the model continues approximately and no longer equals an uninterrupted
``num_workers`` run; a warning says so. A shrunk job never grows back.
"""

import collections
import dataclasses
import secrets
import time

import numpy as np

try:
    import pandas as pd
    import pyarrow as pa
    import pyarrow.compute as pc

    import ray
    from ray.util.placement_group import placement_group, placement_group_table, remove_placement_group
    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy, PlacementGroupSchedulingStrategy
except ImportError as missing:
    raise ImportError("silvic.ray needs Ray: pip install 'silvic[ray]'") from missing

from . import _params
from . import _silvic
from .basic import Booster
from .engine import _IterationHook, _prepare_callbacks, _reject_early_stopping, _resolve_num_iterations

# Seconds the driver waits, after the first actor failure, for the other actors' errors before choosing the root.
_FAILURE_GRACE_S = 5.0

# FNV-1a's 64-bit prime: mixes per-column hashes into one row hash.
_HASH_MIX = np.uint64(0x100000001B3)

# Seconds rank 0 sleeps after each iteration; tests raise it so they can fail a worker mid-training.
_TEST_ITERATION_DELAY_S = 0.0

# Failures that mean a worker (its process, its node or its objects) is gone, rather than a fault in the job.
_WORKER_LOSS = (
    ray.exceptions.RayActorError,
    ray.exceptions.NodeDiedError,
    ray.exceptions.WorkerCrashedError,
    ray.exceptions.LocalRayletDiedError,
    ray.exceptions.ObjectLostError,
)


@dataclasses.dataclass(frozen=True)
class _Spec:
    """Which columns are what."""

    label_column: str
    group_column: object
    weight_column: object
    feature_columns: tuple
    categorical_features: tuple

    @property
    def columns(self):
        """Every column training reads."""
        extra = [c for c in (self.group_column, self.weight_column) if c is not None]
        return [*self.feature_columns, self.label_column, *extra]

    @property
    def sort_columns(self):
        """The row order on a rank: group key, features, label, weight."""
        head = [self.group_column] if self.group_column is not None else []
        tail = [self.weight_column] if self.weight_column is not None else []
        return [*head, *self.feature_columns, self.label_column, *tail]

    @property
    def categorical_indices(self):
        return [self.feature_columns.index(c) for c in self.categorical_features]


def _make_spec(names, label_column, group_column, weight_column, feature_columns, categorical_features):
    reserved = {label_column, group_column, weight_column} - {None}
    missing = [c for c in reserved if c not in names]
    if missing:
        raise ValueError(f"columns {missing} are not in the dataset (columns: {list(names)})")
    if feature_columns is None:
        feature_columns = [c for c in names if c not in reserved]
    feature_columns = tuple(feature_columns)
    missing = [c for c in feature_columns if c not in names]
    if missing:
        raise ValueError(f"feature columns {missing} are not in the dataset")
    if not feature_columns:
        raise ValueError("there are no feature columns")
    categorical_features = tuple(categorical_features or ())
    unknown = [c for c in categorical_features if c not in feature_columns]
    if unknown:
        raise ValueError(f"categorical features {unknown} are not feature columns")
    return _Spec(label_column, group_column, weight_column, feature_columns, categorical_features)


def _decoded(column):
    """``column`` with dictionary (pandas ``category``) encoding undone."""
    if pa.types.is_dictionary(column.type):
        return pc.cast(column, column.type.value_type)
    return column


def _normalized(table):
    """``table`` with every dictionary column decoded, so blocks of one dataset share a schema."""
    columns = [_decoded(column) for column in table.columns]
    return pa.Table.from_arrays(columns, names=table.column_names)


def _hash_column(column):
    """A per-row uint64 hash that depends on the values only, not on their Arrow width."""
    column = _decoded(column)
    kind = column.type
    if pa.types.is_integer(kind) or pa.types.is_boolean(kind):
        values = pc.cast(column, pa.int64()).to_numpy(zero_copy_only=False)
    elif pa.types.is_floating(kind):
        values = pc.cast(column, pa.float64()).to_numpy(zero_copy_only=False)
    elif pa.types.is_string(kind) or pa.types.is_large_string(kind):
        values = column.to_numpy(zero_copy_only=False).astype(object)
    else:
        raise TypeError(f"cannot partition by a column of type {kind}")
    return pd.util.hash_array(values, categorize=False)


def _row_ranks(table, spec, num_workers):
    if spec.group_column is not None:
        hashes = _hash_column(table.column(spec.group_column))
    else:
        hashes = np.zeros(table.num_rows, dtype=np.uint64)
        for name in spec.columns:
            hashes = (hashes * _HASH_MIX) ^ _hash_column(table.column(name))
    return (hashes % np.uint64(num_workers)).astype(np.int64)


def _split_table(table, spec, num_workers):
    """``table``'s rows per rank (compact copies, so each piece serialises alone)."""
    ranks = _row_ranks(table, spec, num_workers)
    order = np.argsort(ranks, kind="stable")
    offsets = np.concatenate(([0], np.cumsum(np.bincount(ranks, minlength=num_workers))))
    return [table.take(pa.array(order[offsets[r] : offsets[r + 1]])) for r in range(num_workers)]


def _order_table(table, spec):
    """``table`` sorted into the rank's training order (``_Spec.sort_columns``, NaN and nulls last)."""
    table = _normalized(table)
    indices = pc.sort_indices(table, sort_keys=[(name, "ascending") for name in spec.sort_columns])
    return table.take(indices)


def _feature_dtype(table, spec):
    """LightGBM 4.x's pandas rule (``np.result_type`` of the columns and float32), from the Arrow schema."""
    dtypes = []
    for name in spec.feature_columns:
        kind = table.schema.field(name).type
        if not (pa.types.is_integer(kind) or pa.types.is_floating(kind) or pa.types.is_boolean(kind)):
            raise ValueError(f"feature column {name!r} has type {kind}; features must be numeric or bool")
        dtypes.append(np.dtype(kind.to_pandas_dtype()))
    return np.result_type(*dtypes, np.float32)


def _float_column(table, name, dtype):
    """``name`` as a ``dtype`` array with nulls as NaN."""
    return pc.cast(table.column(name), pa.float64()).to_numpy(zero_copy_only=False).astype(dtype, copy=False)


def _group_sizes(table, spec):
    """Run lengths of the group key in the sorted ``table``."""
    keys = _decoded(table.column(spec.group_column))
    if keys.null_count:
        raise ValueError(f"group column {spec.group_column!r} has nulls")
    values = keys.to_numpy(zero_copy_only=False)
    starts = np.flatnonzero(values[1:] != values[:-1]) + 1
    bounds = np.concatenate(([0], starts, [len(values)]))
    return np.diff(bounds).astype(np.uint32)


def _training_arrays(table, spec):
    """``(features, label, weight, group)`` of an ordered table, features column-major."""
    dtype = _feature_dtype(table, spec)
    features = np.empty((table.num_rows, len(spec.feature_columns)), dtype=dtype, order="F")
    for j, name in enumerate(spec.feature_columns):
        features[:, j] = _float_column(table, name, dtype)
    if table.column(spec.label_column).null_count:
        raise ValueError(f"label column {spec.label_column!r} has nulls")
    label = _float_column(table, spec.label_column, np.float32)
    weight = _float_column(table, spec.weight_column, np.float32) if spec.weight_column is not None else None
    group = _group_sizes(table, spec) if spec.group_column is not None else None
    return features, label, weight, group


def ordered_partitions(
    data,
    num_workers,
    *,
    label_column,
    group_column=None,
    weight_column=None,
    feature_columns=None,
    categorical_features=(),
):
    """The rows of the pandas DataFrame ``data`` that each of ``num_workers`` ranks trains on, in training order.

    Concatenated in rank order, they are the single-process input ``silvic.train`` needs to reproduce a
    ``silvic.ray.train`` model bit for bit. Columns are the used columns only, with categories decoded.
    """
    table = pa.Table.from_pandas(data, preserve_index=False)
    spec = _make_spec(
        table.column_names, label_column, group_column, weight_column, feature_columns, categorical_features
    )
    pieces = _split_table(_normalized(table.select(spec.columns)), spec, num_workers)
    return [_order_table(piece, spec).to_pandas() for piece in pieces]


@ray.remote
def _split_block(block, start, end, spec, num_workers):
    from ray.data.block import BlockAccessor

    table = BlockAccessor.for_block(block).to_arrow()
    if start is not None:
        table = table.slice(start, end - start)
    pieces = _split_table(_normalized(table.select(spec.columns)), spec, num_workers)
    return pieces[0] if num_workers == 1 else tuple(pieces)


def _partition(dataset, spec, num_workers):
    """Per rank, the object refs of its pieces of ``dataset``; returns once every split task has finished."""
    missing = [c for c in spec.columns if c not in dataset.schema().names]
    if missing:
        raise ValueError(f"columns {missing} are not in the dataset")
    pieces = [[] for _ in range(num_workers)]
    for bundle in dataset.select_columns(spec.columns).iter_internal_ref_bundles():
        slices = getattr(bundle, "slices", None) or (None,) * len(bundle.block_refs)
        for block_ref, block_slice in zip(bundle.block_refs, slices):
            start, end = (block_slice.start_offset, block_slice.end_offset) if block_slice is not None else (None, None)
            refs = _split_block.options(num_returns=num_workers).remote(block_ref, start, end, spec, num_workers)
            for rank, ref in enumerate([refs] if num_workers == 1 else refs):
                pieces[rank].append(ref)
    # Split tasks hold CPUs the placement group may need; let them finish before reserving it.
    every = [ref for rank_refs in pieces for ref in rank_refs]
    if every:
        ray.wait(every, num_returns=len(every), fetch_local=False)
    return pieces


def _load(pieces, spec, rank):
    tables = [table for table in ray.get(pieces) if table.num_rows]
    if not tables:
        raise ValueError(
            f"worker {rank} received no rows: the data has fewer query groups (or rows) than there are workers"
        )
    return _order_table(pa.concat_tables(tables, promote_options="default"), spec)


class _CheckpointKeeper:
    """The latest checkpoint of a job and the evaluation history up to it.

    Records carry the attempt that sent them: once the driver resumes a new attempt, late records of a killed one are
    dropped, so the history never mixes attempts.
    """

    def __init__(self):
        self.attempt = 1
        self.iteration = -1
        self.model_str = None
        self.history = []

    def record(self, attempt, iteration, model_str, history):
        if attempt != self.attempt or iteration <= self.iteration:
            return
        self.iteration = iteration
        self.model_str = model_str
        self.history.extend(history)

    def latest_iteration(self):
        """The last checkpointed iteration (``-1`` before the first checkpoint)."""
        return self.iteration

    def resume(self, attempt):
        """Starts ``attempt``; returns ``(iteration, model_str, history)`` of the latest checkpoint."""
        self.attempt = attempt
        return self.iteration, self.model_str, list(self.history)


_RemoteKeeper = ray.remote(num_cpus=0, max_restarts=0)(_CheckpointKeeper)


class _RankZeroHook:
    """Rank 0's iteration callback and checkpoint sink: runs the evaluation callbacks, keeps the evaluation history,
    and ships checkpoints with the history since the previous one to the keeper."""

    def __init__(self, hook, job):
        self.hook = hook
        self.keeper = job["keeper"]
        self.attempt = job["attempt"]
        self.delay_s = job["iteration_delay_s"]
        self.history = []
        self.unsent = []

    def __call__(self, iteration, evals):
        entry = (iteration, list(evals))
        self.history.append(entry)
        self.unsent.append(entry)
        self.hook(iteration, evals)
        if self.delay_s:
            time.sleep(self.delay_s)
        return False

    def checkpoint(self, iteration, model_str):
        self.keeper.record.remote(self.attempt, iteration, model_str, self.unsent)
        self.unsent = []


class _Worker:
    """One rank: binds, loads its pieces, connects and trains."""

    def __init__(self, rank, num_workers):
        self.rank = rank
        self.num_workers = num_workers
        self.listener = None

    def bind(self):
        self.listener = _silvic.Listener.bind(ray.util.get_node_ip_address())
        return self.listener.address

    def _datasets(self, job, collective, arrays, valid_arrays):
        spec = job["spec"]
        features, label, weight, group = arrays
        train_set = _silvic.Dataset(
            features,
            label,
            weight=weight,
            group=group,
            feature_names=list(spec.feature_columns),
            categorical_features=spec.categorical_indices,
            params_json=job["params_json"],
            collective=collective,
        )
        valid = []
        for name, (v_features, v_label, v_weight, v_group) in valid_arrays:
            dataset = _silvic.Dataset.construct_with_reference(
                v_features, v_label, train_set, weight=v_weight, group=v_group, collective=collective
            )
            valid.append((name, dataset))
        return train_set, valid

    def train(self, job, train_pieces, valid_pieces):
        """Trains the job's remaining trees; rank 0 returns ``(model_str, history)``, the others ``None``."""
        spec = job["spec"]
        arrays = _training_arrays(_load(train_pieces, spec, self.rank), spec)
        valid_arrays = [
            (name, _training_arrays(_load(pieces, spec, self.rank), spec)) for name, pieces in valid_pieces
        ]
        collective = _silvic.connect(
            self.listener, self.rank, job["addresses"], job["token"], timeout_s=job["timeout_s"]
        )
        train_set, valid = self._datasets(job, collective, arrays, valid_arrays)
        del arrays, valid_arrays
        hook = None
        if self.rank == 0:
            callbacks = _prepare_callbacks(None, job["verbose_eval"], None)
            evaluation = _IterationHook(callbacks, job["params"], job["remaining"], job["begin_iteration"])
            hook = _RankZeroHook(evaluation, job)
        # Ray datasets are tables without pandas `category` columns, which LightGBM records as `[]`, as silvic.train
        # does for such a DataFrame.
        model_str = _silvic.train(
            job["params_json"],
            train_set,
            valid,
            collective=collective,
            callback=hook,
            pandas_categorical="[]",
            init_model=job["init_model"],
            checkpoint=hook.checkpoint if hook is not None else None,
            checkpoint_period=job["checkpoint_period"],
        )
        return (model_str, hook.history) if hook is not None else None


_RemoteWorker = ray.remote(max_restarts=0, max_task_retries=0)(_Worker)


def _worker_resources(resources_per_worker, use_gpu):
    resources = dict(resources_per_worker or {"CPU": 1})
    if use_gpu:
        resources.setdefault("GPU", 1)
    return {key: value for key, value in resources.items() if value}


def _actor_options(resources, pg, rank, name):
    custom = {k: v for k, v in resources.items() if k not in ("CPU", "GPU", "memory")}
    options = {
        "num_cpus": resources.get("CPU", 0),
        "num_gpus": resources.get("GPU", 0),
        "scheduling_strategy": PlacementGroupSchedulingStrategy(
            placement_group=pg, placement_group_bundle_index=rank
        ),
        "name": name,
    }
    if custom:
        options["resources"] = custom
    if "memory" in resources:
        options["memory"] = resources["memory"]
    return options


def _cause(error):
    return getattr(error, "cause", None) or error


def _is_worker_loss(error):
    cause = _cause(error)
    return isinstance(error, _WORKER_LOSS) or isinstance(cause, (*_WORKER_LOSS, _silvic.CollectiveError))


def _root_error(errors):
    """``(rank, error)`` of the failure that caused the others: a fault in the job, then a lost worker, then the
    collective errors the surviving ranks see."""

    def priority(item):
        rank, error = item
        if isinstance(_cause(error), _silvic.CollectiveError):
            return (2, rank)
        if _is_worker_loss(error):
            return (1, rank)
        return (0, rank)

    return min(errors, key=priority)


class _RankFailed(Exception):
    """The root failure of an attempt: ``error`` raised by rank ``rank``."""

    def __init__(self, rank, error):
        super().__init__(rank, error)
        self.rank = rank
        self.error = error


def _gather(refs, timeout_s=None):
    """``ray.get`` of every ref; on the first error raises ``_RankFailed`` with the root of every failure seen."""
    pending = {ref: rank for rank, ref in enumerate(refs)}
    results = [None] * len(refs)
    while pending:
        ready, _ = ray.wait(list(pending), num_returns=1, timeout=timeout_s)
        if not ready:
            raise TimeoutError(f"silvic workers did not answer within {timeout_s} s")
        for ref in ready:
            rank = pending.pop(ref)
            try:
                results[rank] = ray.get(ref)
            except Exception as error:  # noqa: BLE001 - any worker failure ends the attempt
                raise _RankFailed(*_root_error([(rank, error), *_more_errors(pending)])) from None
    return results


def _more_errors(pending):
    """Errors of the ``pending`` refs that finish within the grace period."""
    if not pending:
        return []
    done, _ = ray.wait(list(pending), num_returns=len(pending), timeout=_FAILURE_GRACE_S)
    errors = []
    for ref in done:
        try:
            ray.get(ref)
        except Exception as error:  # noqa: BLE001
            errors.append((pending[ref], error))
    return errors


def _warn(message):
    """Prints ``[silvic] [Warning] <message>`` to stdout, where LightGBM's Python package logs."""
    print(f"[silvic] [Warning] {message}", flush=True)


def _describe(error):
    cause = _cause(error)
    lines = str(cause).strip().splitlines()
    text = f"{type(cause).__name__}: {lines[0] if lines else ''}"
    return text if len(text) <= 200 else text[:197] + "..."


def _bundle_nodes(pg, num_workers):
    """Per bundle, ``<ip> (node <id>)`` of the node it was placed on."""
    bundles = placement_group_table(pg).get("bundles_to_node_id", {})
    addresses = {node["NodeID"]: node["NodeManagerAddress"] for node in ray.nodes()}
    labels = []
    for rank in range(num_workers):
        node_id = bundles.get(rank)
        labels.append(f"{addresses.get(node_id, '?')} (node {str(node_id)[:8]})" if node_id else "an unknown node")
    return labels


def _evals_result(history):
    """LightGBM's ``evals_result`` dict from ``[(iteration, [(dataset, metric, value, higher_is_better)])]``."""
    result = {}
    for _, evals in history:
        for data_name, eval_name, value, _ in evals:
            result.setdefault(data_name, collections.OrderedDict()).setdefault(eval_name, []).append(value)
    return result


@dataclasses.dataclass
class _Checkpoint:
    """Where an attempt starts: ``iteration + 1`` trees of ``model_str`` and the evaluation history up to them."""

    iteration: int = -1
    model_str: object = None
    history: list = dataclasses.field(default_factory=list)

    @property
    def trees(self):
        return self.iteration + 1


class _Job:
    """One ``train`` call: its inputs, the partition pieces per worker count, and the attempts."""

    def __init__(self, spec, datasets, params, num_iterations, resources, options):
        self.spec = spec
        self.train_dataset, self.valid_datasets = datasets
        self.params = params
        self.num_iterations = num_iterations
        self.resources = resources
        self.options = options
        self.job_id = secrets.token_hex(4)
        self.pieces = {}
        self.keeper = None

    def _pieces_for(self, num_workers):
        if num_workers not in self.pieces:
            train_pieces = _partition(self.train_dataset, self.spec, num_workers)
            valid_pieces = [(name, _partition(ds, self.spec, num_workers)) for name, ds in self.valid_datasets]
            self.pieces = {num_workers: (train_pieces, valid_pieces)}
        return self.pieces[num_workers]

    def _place(self, num_workers, timeout_s, min_workers):
        """A ready placement group of ``num_workers`` bundles, shrinking towards ``min_workers`` on timeouts."""
        while True:
            pg = placement_group(
                [self.resources] * num_workers,
                strategy=self.options["placement_strategy"],
                name=f"silvic-{self.job_id}-{secrets.token_hex(2)}",
            )
            ready, _ = ray.wait([pg.ready()], timeout=timeout_s)
            if ready:
                return pg, num_workers
            remove_placement_group(pg)
            if num_workers <= min_workers:
                raise TimeoutError(
                    f"Ray could not place {num_workers} workers with {self.resources} each within {timeout_s} s"
                )
            _warn(f"could not place {num_workers} workers within {timeout_s} s; trying {num_workers - 1}")
            num_workers -= 1

    def _job(self, attempt, addresses, start):
        remaining = self.num_iterations - start.trees
        params = {**self.params, "num_iterations": remaining}
        return {
            "spec": self.spec,
            "addresses": addresses,
            "token": secrets.randbits(64),
            "timeout_s": float(self.options["timeout_s"]),
            "params": self.params,
            "params_json": _params.to_json(params),
            "remaining": remaining,
            "begin_iteration": start.trees,
            "init_model": start.model_str,
            "checkpoint_period": self.options["checkpoint_period"],
            "verbose_eval": self.options["verbose_eval"],
            "keeper": self.keeper,
            "attempt": attempt,
            "iteration_delay_s": _TEST_ITERATION_DELAY_S,
        }

    def _attempt(self, pg, num_workers, attempt, start):
        """Trains on ``pg``; returns rank 0's ``(model_str, history)`` or raises ``_RankFailed``."""
        train_pieces, valid_pieces = self._pieces_for(num_workers)
        workers = [
            _RemoteWorker.options(
                **_actor_options(self.resources, pg, rank, f"silvic-{self.job_id}-a{attempt}-r{rank}")
            ).remote(rank, num_workers)
            for rank in range(num_workers)
        ]
        try:
            addresses = _gather([worker.bind.remote() for worker in workers], self.options["timeout_s"])
            job = self._job(attempt, addresses, start)
            refs = [
                worker.train.remote(job, train_pieces[rank], [(name, p[rank]) for name, p in valid_pieces])
                for rank, worker in enumerate(workers)
            ]
            return _gather(refs)[0]
        finally:
            for worker in workers:
                ray.kill(worker, no_restart=True)

    def _resume(self, failed, nodes, attempt, num_workers):
        """After ``failed``, the checkpoint and placement group of attempt ``attempt``, or the error to raise."""
        cause = _cause(failed.error)
        if isinstance(cause, ray.exceptions.ObjectLostError):
            self.pieces = {}
        start = _Checkpoint(*ray.get(self.keeper.resume.remote(attempt), timeout=self.options["timeout_s"]))
        if start.trees >= self.num_iterations:
            return start, None, num_workers
        pg, placed = self._place(num_workers, self.options["replacement_timeout_s"], self.options["min_workers"])
        if placed < num_workers:
            _warn(
                f"continuing on {placed} workers instead of {num_workers}: rows are repartitioned, so the model "
                f"continues from iteration {start.trees} approximately and will not equal an uninterrupted run"
            )
        _warn(
            f"worker lost on {nodes[failed.rank]} ({_describe(failed.error)}); resuming from iteration "
            f"{start.trees} with {placed} workers (attempt {attempt} of {self.options['max_failures'] + 1})"
        )
        return start, pg, placed

    def run(self, num_workers):
        """Trains to the end, resuming after worker failures; returns ``(model_str, history)``."""
        max_failures = self.options["max_failures"]
        pg, num_workers = self._place(num_workers, self.options["timeout_s"], num_workers)
        start = _Checkpoint()
        attempt = 1
        while True:
            try:
                nodes = _bundle_nodes(pg, num_workers)
                model_str, history = self._attempt(pg, num_workers, attempt, start)
                return model_str, start.history + history
            except _RankFailed as failed:
                if not _is_worker_loss(failed.error):
                    raise failed.error from None
                if attempt > max_failures:
                    _add_note(failed.error, f"silvic.ray.train gave up after {attempt} attempts")
                    raise failed.error from None
                lost = failed
            finally:
                remove_placement_group(pg)
            attempt += 1
            try:
                start, pg, num_workers = self._resume(lost, nodes, attempt, num_workers)
            except TimeoutError as timeout:
                raise timeout from lost.error
            if pg is None:
                return start.model_str, start.history


def _add_note(error, note):
    if hasattr(error, "add_note"):
        error.add_note(note)


def _job_params(params, num_boost_round, resources_per_worker):
    params = dict(params or {})
    num_iterations = _resolve_num_iterations(params, num_boost_round)
    _reject_early_stopping(params)
    cpus = (resources_per_worker or {}).get("CPU")
    if cpus and not _params.find(params, "num_threads"):
        # An explicit CPU reservation bounds the threads, as lightgbm_ray does; without one, every core is used.
        params["num_threads"] = int(cpus)
    # A typo in the parameters fails here, before any data is read or any worker placed.
    _silvic.validate_params(_params.to_json(params))
    return params, num_iterations


def _start_keeper(job_id):
    """The checkpoint keeper, on the driver's node so that losing worker nodes never loses checkpoints."""
    node = NodeAffinitySchedulingStrategy(node_id=ray.get_runtime_context().get_node_id(), soft=False)
    return _RemoteKeeper.options(scheduling_strategy=node, name=f"silvic-{job_id}-keeper").remote()


def train(
    params,
    train_dataset,
    *,
    label_column,
    group_column=None,
    weight_column=None,
    feature_columns=None,
    categorical_features=(),
    valid_datasets=None,
    num_boost_round=100,
    num_workers=1,
    use_gpu=False,
    resources_per_worker=None,
    placement_strategy="PACK",
    evals_result=None,
    verbose_eval="warn",
    timeout_s=1800,
    max_failures=3,
    checkpoint_period=25,
    min_workers=None,
    replacement_timeout_s=300,
):
    """Trains one model over ``num_workers`` Ray actors and returns rank 0's ``silvic.Booster``.

    ``train_dataset`` and the values of ``valid_datasets`` (a dict of name to dataset, evaluated in order) are
    ``ray.data.Dataset``s with the same columns. ``feature_columns`` defaults to every column other than the label,
    group and weight columns; ``categorical_features`` names integer-coded categorical feature columns. ``params``,
    ``num_boost_round``, ``evals_result`` and ``verbose_eval`` mean what they mean in ``silvic.train``; evaluation
    results print from rank 0, and ``evals_result`` holds one continuous history across resumes.
    ``resources_per_worker`` (default ``{"CPU": 1}``) is one placement-group bundle; an explicit ``CPU`` count also
    sets ``num_threads`` unless ``params`` does.

    ``max_failures`` worker failures are resumed from the latest of the checkpoints taken every
    ``checkpoint_period`` iterations; ``min_workers`` (default ``num_workers``: never shrink) is the fewest workers a
    resumed job may continue on when ``num_workers`` cannot be placed within ``replacement_timeout_s``.
    """
    if num_workers < 1:
        raise ValueError(f"num_workers must be at least 1, got {num_workers}")
    min_workers = num_workers if min_workers is None else min_workers
    if not 1 <= min_workers <= num_workers:
        raise ValueError(f"min_workers must be in 1..={num_workers}, got {min_workers}")
    if max_failures < 0 or checkpoint_period < 0:
        raise ValueError("max_failures and checkpoint_period must be non-negative")
    params, num_iterations = _job_params(params, num_boost_round, resources_per_worker)
    spec = _make_spec(
        train_dataset.schema().names,
        label_column,
        group_column,
        weight_column,
        feature_columns,
        categorical_features,
    )
    options = {
        "placement_strategy": placement_strategy,
        "timeout_s": timeout_s,
        "verbose_eval": verbose_eval,
        "max_failures": max_failures,
        "checkpoint_period": checkpoint_period,
        "min_workers": min_workers,
        "replacement_timeout_s": replacement_timeout_s,
    }
    datasets = (train_dataset, list((valid_datasets or {}).items()))
    job = _Job(spec, datasets, params, num_iterations, _worker_resources(resources_per_worker, use_gpu), options)
    job.keeper = _start_keeper(job.job_id)
    try:
        model_str, history = job.run(num_workers)
    finally:
        ray.kill(job.keeper, no_restart=True)
    if evals_result is not None:
        evals_result.clear()
        evals_result.update(_evals_result(history))
    booster = Booster(params=params, model_str=model_str)
    if history:
        for dataset_name, eval_name, score, _ in history[-1][1]:
            booster.best_score[dataset_name][eval_name] = score
    return booster
