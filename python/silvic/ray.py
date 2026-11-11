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
concurrently, reducing histograms over the TCP ring. If any actor fails, the driver kills the rest and raises the
root error: an actor's own exception before a dead actor, and either before the collective errors its peers see as
a consequence.

``use_gpu=True`` reserves one Ray GPU per actor, which CUDA training uses; ``device_type`` in ``params`` picks the
device as in ``silvic.train`` (``auto`` takes CUDA, then Metal, then the CPU). Metal needs no Ray GPU resource.
``timeout_s`` bounds placement, the TCP rendezvous, and how long a collective waits on a silent peer.
"""

import dataclasses
import secrets

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

import ray
from ray.util.placement_group import placement_group, remove_placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from . import _params
from . import _silvic
from .basic import Booster
from .engine import _IterationHook, _prepare_callbacks, _reject_early_stopping, _resolve_num_iterations

# Seconds the driver waits, after the first actor failure, for the other actors' errors before choosing the root.
_FAILURE_GRACE_S = 5.0

# FNV-1a's 64-bit prime: mixes per-column hashes into one row hash.
_HASH_MIX = np.uint64(0x100000001B3)


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
    import pandas as pd

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


class _Worker:
    """One rank: binds, loads its pieces, connects and trains."""

    def __init__(self, rank, num_workers):
        self.rank = rank
        self.num_workers = num_workers
        self.listener = None

    def bind(self):
        self.listener = _silvic.Listener.bind(ray.util.get_node_ip_address())
        return self.listener.address

    def train(self, job, train_pieces, valid_pieces, record):
        """Trains; rank 0 returns ``(model_str, evals_result, best_score)``, the others ``None``."""
        spec = job["spec"]
        features, label, weight, group = _training_arrays(_load(train_pieces, spec, self.rank), spec)
        valid_arrays = [
            (name, _training_arrays(_load(pieces, spec, self.rank), spec)) for name, pieces in valid_pieces
        ]
        collective = _silvic.connect(
            self.listener, self.rank, job["addresses"], job["token"], timeout_s=job["timeout_s"]
        )
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
        del features
        valid = []
        for name, (v_features, v_label, v_weight, v_group) in valid_arrays:
            dataset = _silvic.Dataset.construct_with_reference(
                v_features, v_label, train_set, weight=v_weight, group=v_group, collective=collective
            )
            valid.append((name, dataset))
        del valid_arrays
        evals_result = {}
        hook = None
        if record:
            callbacks = _prepare_callbacks(None, job["verbose_eval"], evals_result)
            hook = _IterationHook(callbacks, job["params"], job["num_iterations"])
        # Ray datasets are tables without pandas `category` columns, which LightGBM records as `[]`, as silvic.train
        # does for such a DataFrame.
        model_str = _silvic.train(
            job["params_json"], train_set, valid, collective=collective, callback=hook, pandas_categorical="[]"
        )
        if not record:
            return None
        return model_str, evals_result, {name: dict(scores) for name, scores in hook.best_score().items()}


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


def _root_error(errors):
    """The failure that caused the others: an actor's own exception, then a dead actor, then collective errors."""

    def priority(item):
        rank, error = item
        cause = _cause(error)
        if isinstance(cause, _silvic.CollectiveError):
            return (2, rank)
        if isinstance(error, ray.exceptions.RayActorError):
            return (1, rank)
        return (0, rank)

    return min(errors, key=priority)[1]


def _gather(refs, timeout_s=None):
    """``ray.get`` of every ref, failing on the first error with the root error of every failure seen."""
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
            except Exception as error:  # noqa: BLE001 - any worker failure ends the job
                raise _root_error([(rank, error), *_more_errors(pending)]) from None
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


def _job_params(params, num_boost_round, resources_per_worker):
    params = dict(params or {})
    num_iterations = _resolve_num_iterations(params, num_boost_round)
    _reject_early_stopping(params)
    cpus = (resources_per_worker or {}).get("CPU")
    if cpus and not _params.find(params, "num_threads"):
        # An explicit CPU reservation bounds the threads, as lightgbm_ray does; without one, every core is used.
        params["num_threads"] = int(cpus)
    return params, num_iterations


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
):
    """Trains one model over ``num_workers`` Ray actors and returns rank 0's ``silvic.Booster``.

    ``train_dataset`` and the values of ``valid_datasets`` (a dict of name to dataset, evaluated in order) are
    ``ray.data.Dataset``s with the same columns. ``feature_columns`` defaults to every column other than the label,
    group and weight columns; ``categorical_features`` names integer-coded categorical feature columns. ``params``,
    ``num_boost_round``, ``evals_result`` and ``verbose_eval`` mean what they mean in ``silvic.train``; evaluation
    results print from rank 0. ``resources_per_worker`` (default ``{"CPU": 1}``) is one placement-group bundle;
    an explicit ``CPU`` count also sets ``num_threads`` unless ``params`` does.
    """
    if num_workers < 1:
        raise ValueError(f"num_workers must be at least 1, got {num_workers}")
    params, num_iterations = _job_params(params, num_boost_round, resources_per_worker)
    spec = _make_spec(
        train_dataset.schema().names,
        label_column,
        group_column,
        weight_column,
        feature_columns,
        categorical_features,
    )
    train_pieces = _partition(train_dataset, spec, num_workers)
    valid_pieces = [(name, _partition(ds, spec, num_workers)) for name, ds in (valid_datasets or {}).items()]
    resources = _worker_resources(resources_per_worker, use_gpu)
    job_id = secrets.token_hex(4)
    pg = placement_group([resources] * num_workers, strategy=placement_strategy, name=f"silvic-{job_id}")
    workers = []
    try:
        ready, _ = ray.wait([pg.ready()], timeout=timeout_s)
        if not ready:
            raise TimeoutError(
                f"Ray could not place {num_workers} workers with {resources} each within {timeout_s} s"
            )
        workers = [
            _RemoteWorker.options(**_actor_options(resources, pg, rank, f"silvic-{job_id}-{rank}")).remote(
                rank, num_workers
            )
            for rank in range(num_workers)
        ]
        addresses = _gather([worker.bind.remote() for worker in workers], timeout_s)
        job = {
            "spec": spec,
            "addresses": addresses,
            "token": secrets.randbits(64),
            "timeout_s": float(timeout_s),
            "params": params,
            "params_json": _params.to_json(params),
            "num_iterations": num_iterations,
            "verbose_eval": verbose_eval,
        }
        refs = [
            worker.train.remote(job, train_pieces[rank], [(name, p[rank]) for name, p in valid_pieces], rank == 0)
            for rank, worker in enumerate(workers)
        ]
        results = _gather(refs)
    finally:
        for worker in workers:
            ray.kill(worker, no_restart=True)
        remove_placement_group(pg)
    model_str, history, best_score = results[0]
    if evals_result is not None:
        evals_result.clear()
        evals_result.update(history)
    booster = Booster(params=params, model_str=model_str)
    for name, scores in best_score.items():
        booster.best_score[name].update(scores)
    return booster
