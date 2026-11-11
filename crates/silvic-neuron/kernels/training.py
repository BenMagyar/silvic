"""Bounded training primitives; all integer reductions stay strictly below 2**24.

Tensor arguments follow Rust TensorSpec order. Metadata is int32[8,1]. Histogram
and leaf sums read [begin,count,column_offset,bin_base,...]. Partition metadata
is [begin,count,column_offset,partition_len,...]. Kernels return new HBM outputs.
"""

import neuronxcc.nki.isa as nisa
import neuronxcc.nki.language as nl
import numpy as np
from neuronxcc import nki


@nki.jit
def _add(a, b):
    return nisa.tensor_tensor(
        a, b, op=np.add, dtype=nl.int32, engine=nisa.gpsimd_engine
    )


@nki.jit
def _subtract(a, b):
    inverse = nl.invert(b)
    one = nl.full(b.shape, 1, dtype=nl.int32)
    return _add(a, _add(inverse, one))


@nki.jit
def _less(a, b):
    # All addresses are in [0,2**31), so the signed difference cannot overflow.
    difference = _subtract(a, b)
    return nl.bitwise_and(nl.right_shift(difference, 31), 1) != 0


@nki.jit
def _select(predicate, a, b):
    mask = nl.where(
        predicate,
        nl.full(a.shape, -1, dtype=nl.int32),
        nl.zeros(a.shape, dtype=nl.int32),
    )
    return nl.bitwise_or(nl.bitwise_and(a, mask), nl.bitwise_and(b, nl.invert(mask)))


@nki.jit
def _meta(meta, field, TILE):
    return nl.load(meta[field : field + 1, 0:1]).broadcast_to((TILE, 1))


@nki.jit
def _transpose(data):
    # Vector transpose preserves all f32 bits, including NaN/Inf; the tensor
    # engine's identity-matmul transpose does not preserve those values.
    P = data.shape[0]
    F = data.shape[1]
    result = nl.ndarray((F, P), dtype=data.dtype, buffer=nl.sbuf)
    for p in nl.static_range((P + 31) // 32):
        for f in nl.static_range((F + 31) // 32):
            pe = min((p + 1) * 32, P)
            fe = min((f + 1) * 32, F)
            result[f * 32 : fe, p * 32 : pe] = nisa.nc_transpose(
                data[p * 32 : pe, f * 32 : fe], engine=nisa.vector_engine
            )
    return result


@nki.jit
def histogram(bins, partition, values, meta, *, ROWS, COLUMNS, TILE, CHUNKS, BINS):
    result = nl.ndarray((CHUNKS, 2, BINS, 1), dtype=nl.float32, buffer=nl.shared_hbm)
    b = nisa.iota(nl.arange(BINS)[None, :], dtype=nl.int32).broadcast_to((TILE, BINS))
    output_bin = nl.arange(BINS)[:, None]
    begin = _meta(meta, 0, TILE)
    count = _meta(meta, 1, TILE)
    column = _meta(meta, 2, TILE)
    base = _meta(meta, 3, TILE)
    target_bins = base.broadcast_to((TILE, BINS)) + b
    last = nl.full((TILE, 1), ROWS - 1, dtype=nl.int32)
    for chunk in nl.affine_range(CHUNKS):
        offset = nisa.iota(chunk * TILE + nl.arange(TILE)[:, None], dtype=nl.int32)
        position = _add(begin, offset)
        position = _select(_less(last, position), last, position)
        row = nl.load(partition[position, 0])
        row = _select(offset < count, row, nl.zeros((TILE, 1), dtype=nl.int32))
        observed = nl.load(bins[_add(column, row), 0])
        hit = (observed.broadcast_to((TILE, BINS)) == target_bins) & (
            offset < count
        ).broadcast_to((TILE, BINS))
        for component in nl.static_range(2):
            component_values = values[component]
            value = nl.load(component_values[row, 0])
            weighted = nl.where(hit, value.broadcast_to((TILE, BINS)), 0.0)
            by_bin = _transpose(weighted)
            sums = nl.sum(by_bin, axis=1, keepdims=True)
            nl.store(result[chunk, component, output_bin, 0], sums)
    return result


@nki.jit
def leaf_sums(partition, limbs, meta, *, ROWS, TILE, CHUNKS, LIMBS):
    result = nl.ndarray((CHUNKS, LIMBS, 1), dtype=nl.float32, buffer=nl.shared_hbm)
    begin = _meta(meta, 0, TILE)
    count = _meta(meta, 1, TILE)
    last = nl.full((TILE, 1), ROWS - 1, dtype=nl.int32)
    for chunk in nl.affine_range(CHUNKS):
        offset = nisa.iota(chunk * TILE + nl.arange(TILE)[:, None], dtype=nl.int32)
        position = _add(begin, offset)
        position = _select(_less(last, position), last, position)
        row = nl.load(partition[position, 0])
        row = _select(offset < count, row, nl.zeros((TILE, 1), dtype=nl.int32))
        for limb in nl.static_range(LIMBS):
            limb_values = limbs[limb]
            digit = nl.load(limb_values[row, 0]).astype(nl.float32)
            digit = nl.where(offset < count, digit, 0.0)
            total = nl.sum(_transpose(digit), axis=1, keepdims=True)
            nl.store(result[chunk, limb, 0], total)
    return result


@nki.jit
def partition_flags(bins, partition, meta, split, *, ROWS, TILE):
    flags = nl.ndarray((ROWS, 1), dtype=nl.int32, buffer=nl.shared_hbm)
    counts = nl.ndarray((ROWS // TILE, 1), dtype=nl.int32, buffer=nl.shared_hbm)
    begin = _meta(meta, 0, TILE)
    end = _add(begin, _meta(meta, 1, TILE))
    column = _meta(meta, 2, TILE)
    zero = nl.zeros((TILE, 1), dtype=nl.int32)
    for block in nl.affine_range(ROWS // TILE):
        position = nisa.iota(block * TILE + nl.arange(TILE)[:, None], dtype=nl.int32)
        row = nl.load(partition[position, 0])
        active = (~_less(position, begin)) & _less(position, end)
        row = _select(active, row, zero)
        bin_id = nl.load(bins[_add(column, row), 0])
        word = nl.load(split[nl.right_shift(bin_id, 5), 0])
        left = nl.bitwise_and(nl.right_shift(word, nl.bitwise_and(bin_id, 31)), 1)
        left = nl.where(active, left, 0)
        nl.store(flags[position, 0], left)
        total = nl.sum(nl.transpose(left.astype(nl.float32)), axis=1, keepdims=True)
        nl.store(counts[block, 0], total.astype(nl.int32))
    return flags, counts


@nki.jit
def partition_scatter(partition, flags, offsets, meta, *, ROWS, TILE):
    result = nl.ndarray((ROWS, 1), dtype=nl.int32, buffer=nl.shared_hbm)
    begin = _meta(meta, 0, TILE)
    end = _add(begin, _meta(meta, 1, TILE))
    ones = nl.full((1, TILE), 1.0, dtype=nl.float32)
    one = nl.full((TILE, 1), 1, dtype=nl.int32)
    for block in nl.affine_range(ROWS // TILE):
        position = nisa.iota(block * TILE + nl.arange(TILE)[:, None], dtype=nl.int32)
        row = nl.load(partition[position, 0])
        left = nl.load(flags[position, 0]).astype(nl.float32)
        active = (~_less(position, begin)) & _less(position, end)
        right = nl.where(active, 1.0 - left, 0.0)
        left_prefix = nisa.tensor_tensor_scan(
            ones,
            _transpose(left),
            initial=0.0,
            op0=np.multiply,
            op1=np.add,
            dtype=nl.float32,
        )
        right_prefix = nisa.tensor_tensor_scan(
            ones,
            _transpose(right),
            initial=0.0,
            op0=np.multiply,
            op1=np.add,
            dtype=nl.float32,
        )
        lp = _transpose(left_prefix).astype(nl.int32)
        rp = _transpose(right_prefix).astype(nl.int32)
        left_base = nl.load(offsets[block : block + 1, 0:1]).broadcast_to((TILE, 1))
        right_base = nl.load(
            offsets[ROWS // TILE + block : ROWS // TILE + block + 1, 0:1]
        ).broadcast_to((TILE, 1))
        ld = _subtract(_add(left_base, lp), one)
        rd = _subtract(_add(right_base, rp), one)
        destination = _select(left != 0, ld, rd)
        destination = _select(active, destination, position)
        nl.store(result[destination, 0], row)
    return result


@nki.jit
def copy_u32(source, *, ROWS, TILE):
    result = nl.ndarray((ROWS, 1), dtype=nl.int32, buffer=nl.shared_hbm)
    r = nl.arange(TILE)[:, None]
    for block in nl.affine_range(ROWS // TILE):
        position = block * TILE + r
        nl.store(result[position, 0], nl.load(source[position, 0]))
    return result


@nki.jit
def copy_f32(source, *, ROWS, TILE):
    result = nl.ndarray((ROWS, 1), dtype=nl.float32, buffer=nl.shared_hbm)
    r = nl.arange(TILE)[:, None]
    for block in nl.affine_range(ROWS // TILE):
        position = block * TILE + r
        nl.store(result[position, 0], nl.load(source[position, 0]))
    return result


@nki.jit
def add_tree(bins, scores, nodes, leaves, meta, *, ROWS, COUNT, TILE, NODES):
    result = nl.ndarray((ROWS, 1), dtype=nl.float32, buffer=nl.shared_hbm)
    depth = _meta(meta, 0, TILE)
    row_stride = nl.full((TILE, 1), ROWS, dtype=nl.int32)
    last_row = nl.full((TILE, 1), COUNT, dtype=nl.int32)
    columns = nodes[0]
    left_children = nodes[1]
    right_children = nodes[2]
    for block in nl.affine_range(ROWS // TILE):
        row = nisa.iota(block * TILE + nl.arange(TILE)[:, None], dtype=nl.int32)
        current = nl.zeros((TILE, 1), dtype=nl.int32)
        for level in nl.sequential_range(NODES):
            node_index = nisa.tensor_copy(
                current[...], dtype=nl.int32, engine=nisa.vector_engine
            )
            column = nl.load(columns[node_index, 0])
            column_offset = nisa.tensor_tensor(
                column,
                row_stride,
                op=np.multiply,
                dtype=nl.int32,
                engine=nisa.gpsimd_engine,
            )
            bin_id = nl.load(bins[_add(column_offset, row), 0])
            word_id = nl.right_shift(bin_id, 5)
            word = nl.zeros((TILE, 1), dtype=nl.int32)
            for k in nl.static_range(8):
                mask_words = nodes[3 + k]
                candidate = nl.load(mask_words[node_index, 0])
                word = _select(word_id == k, candidate, word)
            left = nl.bitwise_and(nl.right_shift(word, nl.bitwise_and(bin_id, 31)), 1)
            lchild = nl.load(left_children[node_index, 0])
            rchild = nl.load(right_children[node_index, 0])
            child = _select(left != 0, lchild, rchild)
            current[...] = _select(depth > level, child, current)
        leaf_index = nisa.tensor_copy(
            current[...], dtype=nl.int32, engine=nisa.vector_engine
        )
        value = nl.load(leaves[leaf_index, 0])
        score = nl.load(scores[row, 0])
        nl.store(result[row, 0], nl.where(_less(row, last_row), score + value, score))
    return result
