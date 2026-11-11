"""Bounded float32 row tiles; forest records remain resident in HBM."""

import neuronxcc.nki.isa as nisa
import neuronxcc.nki.language as nl
import numpy as np
from neuronxcc import nki


@nki.jit
def integer_select(condition, yes, no):
    ones = nl.full(condition.shape, -1, dtype=nl.int32, buffer=nl.sbuf)
    mask = nl.where(condition, ones, 0, dtype=nl.int32)
    return nl.bitwise_or(nl.bitwise_and(yes, mask), nl.bitwise_and(no, nl.invert(mask)))


@nki.jit
def integer_add(lhs, rhs):
    add_rhs = nl.broadcast_to(rhs, shape=lhs.shape)
    return nisa.tensor_tensor(
        lhs, add_rhs, op=np.add, dtype=nl.int32, engine=nisa.gpsimd_engine
    )


@nki.jit
def integer_subtract(lhs, rhs):
    subtract_rhs = nl.broadcast_to(rhs, shape=lhs.shape)
    subtract_ones = nl.full(lhs.shape, 1, dtype=nl.int32, buffer=nl.sbuf)
    negative = nisa.tensor_tensor(
        nl.invert(subtract_rhs),
        subtract_ones,
        op=np.add,
        dtype=nl.int32,
        engine=nisa.gpsimd_engine,
    )
    return nisa.tensor_tensor(
        lhs, negative, op=np.add, dtype=nl.int32, engine=nisa.gpsimd_engine
    )


@nki.jit
def score(input, metadata, values, categories, prefix, RECORDS: int, ROWS: int):
    output = nl.ndarray((1, ROWS), dtype=nl.float32, buffer=nl.shared_hbm)
    p = nl.arange(1)[:, None]
    r = nl.arange(ROWS)[None, :]
    limit = nl.load(prefix[p, 0])
    total = nl.full((1, ROWS), 0.0, dtype=nl.float32, buffer=nl.sbuf)
    selected = nl.full((1, ROWS), 0.0, dtype=nl.float32, buffer=nl.sbuf)
    child = nl.full((1, ROWS), 0, dtype=nl.int32, buffer=nl.sbuf)
    zeros = nl.full((1, ROWS), 0, dtype=nl.int32, buffer=nl.sbuf)
    float_zeros = nl.full((1, ROWS), 0.0, dtype=nl.float32, buffer=nl.sbuf)
    maximum_category = nl.full((1, ROWS), 2147483647, dtype=nl.int32, buffer=nl.sbuf)
    row_stride = nl.full((1, 1), ROWS, dtype=nl.int32, buffer=nl.sbuf)
    for record in nl.sequential_range(RECORDS):
        node = nl.load(metadata[p, record * 9])
        left = nl.load(metadata[p, record * 9 + 1])
        right = nl.load(metadata[p, record * 9 + 2])
        feature = nl.load(metadata[p, record * 9 + 3])
        decision = nl.load(metadata[p, record * 9 + 4])
        category_start = nl.load(metadata[p, record * 9 + 5])
        category_length = nl.load(metadata[p, record * 9 + 6])
        tree = nl.load(metadata[p, record * 9 + 7])
        end = nl.load(metadata[p, record * 9 + 8])
        threshold = nl.load(values[p, record * 3])
        left_value = nl.load(values[p, record * 3 + 1])
        right_value = nl.load(values[p, record * 3 + 2])
        feature_offset = nisa.tensor_tensor(
            feature,
            row_stride,
            op=np.multiply,
            dtype=nl.int32,
            engine=nisa.gpsimd_engine,
        )
        value = nl.load(input[p, feature_offset + r])
        value = nl.where(nl.abs(value) <= 1.0e-35, float_zeros, value)
        nan = nl.not_equal(value, value)
        missing = nl.bitwise_and(nl.right_shift(decision, 2), 3)
        numeric_value = nl.where(nan & (missing != 2), float_zeros, value)
        is_missing = ((missing == 1) & (numeric_value == 0.0)) | ((missing == 2) & nan)
        numeric_left = nl.where(
            is_missing, nl.bitwise_and(decision, 2) != 0, numeric_value <= threshold
        )

        # Only nonnegative, representable inputs reach float-to-int conversion.
        # (-1, 0) truncates to category zero; >= 2**31 saturates to INT_MAX.
        safe_value = nl.where((value >= 0.0) & (value < 2147483648.0), value, 0.0)
        category = nl.copy(nl.floor(safe_value), dtype=nl.int32)
        category = integer_select(value >= 2147483648.0, maximum_category, category)
        word_index = nl.right_shift(category, 5)
        in_range = integer_subtract(word_index, category_length) < 0
        safe_index = integer_select(
            in_range, integer_add(word_index, category_start), zeros
        )
        # Vector-engine transpose is bit-exact and limited to 32x32 on SDK 2.26.
        gather_index = nisa.nc_transpose(safe_index, engine=nisa.vector_engine)
        gathered = nl.load(categories[gather_index, 0])
        word = nisa.nc_transpose(gathered, engine=nisa.vector_engine)
        shift = nl.copy(nl.bitwise_and(category, 31), dtype=nl.uint32)
        bit = nl.bitwise_and(nl.right_shift(word, shift), np.uint32(1))
        categorical_left = (value > -1.0) & (~nan) & in_range & (bit != 0)
        goes_left = nl.where(
            nl.bitwise_and(decision, 1) != 0, categorical_left, numeric_left
        )
        active = nl.bitwise_xor(child, node) == 0
        next_child = integer_select(goes_left, left, right)
        leaf_value = nl.where(goes_left, left_value, right_value)
        selected[...] = nl.where(active & (next_child < 0), leaf_value, selected)
        child[...] = integer_select(active, next_child, child)

        # Exactly one ordered float32 addition per tree, never a reduction over trees.
        added = total + nl.where(integer_subtract(tree, limit) < 0, selected, 0.0)
        total[...] = nl.where(end != 0, added, total)
        child[...] = integer_select(end != 0, zeros, child)
        selected[...] = nl.where(end != 0, float_zeros, selected)
    nl.store(output[p, r], total)
    return output
