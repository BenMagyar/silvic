// Parameter structs, the hash RNG and quantization shared by the kernels. Every struct is mirrored field for field by
// a `#[repr(C)]` Rust struct in `src/params.rs` (plain 4-byte fields, so both sides agree on the layout without
// padding rules). The capacities (HIST_QUANT_BINS, ...) are `#define`d by the backend in front of this file from the
// Rust constants, so the host and the kernels cannot disagree on them.

#if !defined(HIST_QUANT_BINS) || !defined(HIST_FLOAT_BINS) || !defined(REDUCE_THREADS) || !defined(SCAN_THREADS) || \
    !defined(HIST_SIDE_LEFT) || !defined(HIST_SIDE_RIGHT)
#error "the backend defines the kernel capacities before common.h"
#endif

// -- parameter structs -----------------------------------------------------------------------------------------------

struct FillParams {
    uint n;
    uint value;
};

struct GradParams {
    uint n;
    uint has_weight;
    float sigmoid;
    uint pad;
};

struct StatsParams {
    uint n;
    uint pad0;
    uint pad1;
    uint pad2;
};

struct QuantParams {
    uint n;
    uint row_offset_lo;
    uint row_offset_hi;
    float inv_grad_scale;
    float inv_hess_scale;
    int grad_max_q;
    int hess_max_q;
    uint constant_hessian;
    uint stochastic;
    uint seed;
    uint iteration;
    uint pad;
};

// Rows [chunk * block, chunk * (block + 1)) of a bagging pass; `threshold` is `ceil(fraction * 2^24)`, so
// `(draw >> 8) < threshold` is exactly `uniform(...) < fraction` in f64.
struct BagParams {
    uint n;
    uint chunk;
    uint row_offset_lo;
    uint row_offset_hi;
    uint seed;
    uint round;
    uint threshold;
    uint pad;
};

struct ScanParams {
    uint n;
    uint pad0;
    uint pad1;
    uint pad2;
};

// A leaf's rows `partition[begin..begin + count]`, `chunk` rows per threadgroup, split on the byte at `column` of
// the group whose row-major bins start at byte `group_base` with `row_stride` bytes per row.
struct PartitionParams {
    uint begin;
    uint count;
    uint chunk;
    uint row_stride;
    uint group_base_lo;
    uint group_base_hi;
    uint column;
    uint pad;
    uint left_bins[8];
};

struct CopyParams {
    uint begin;
    uint count;
    uint pad0;
    uint pad1;
};

// One histogram work unit: the columns of one feature group whose bins fit one threadgroup's shared histogram.
// `col_map[cols_begin + c]` is the unit-relative bin offset of column `c`'s feature, or -1 when the column is padding
// or belongs to another unit of the group.
struct HistUnit {
    uint words_base_lo;
    uint words_base_hi;
    uint row_words;
    uint bin_begin;
    uint num_bins;
    uint cols_begin;
    uint pad0;
    uint pad1;
};

// `side` 0: the rows `partition[begin..begin + count]`. HIST_SIDE_LEFT / HIST_SIDE_RIGHT: one child of that leaf
// after a partition encoded just before, whose left count the kernel reads from the device (`left_total[0]`), so
// partition and histogram need no host round trip in between.
struct HistParams {
    uint begin;
    uint count;
    uint rows_per_block;
    uint side;
};

struct LeafSumParams {
    float grad_scale;
    float hess_scale;
    uint pad0;
    uint pad1;
};

// Where a feature group's row-major bins start in a dataset's bin buffer.
struct GroupLayout {
    uint base_lo;
    uint base_hi;
    uint row_stride;
    uint pad;
};

// `silvic_core::backend::DeviceNode`. Children >= 0 are nodes, < 0 are ~leaf.
struct TreeNode {
    uint group;
    uint column;
    int left;
    int right;
    uint left_bins[8];
};

struct TreeParams {
    uint n;
    uint num_nodes;
    uint pad0;
    uint pad1;
};

// `queries[query_begin..query_begin + num_queries]` are the training queries this launch handles. The pair sigmoid
// is LightGBM's table (`objective::lambdarank::SigmoidTable`, `sigmoid_last + 1` entries), indexed with its f64
// `min_input` and `idx_factor` split into f32 high and low parts.
struct RankParams {
    uint query_begin;
    uint num_queries;
    uint has_weight;
    uint norm;
    uint truncation_level;
    float sigmoid;
    float sigmoid_min_hi;
    float sigmoid_min_lo;
    float sigmoid_factor_hi;
    float sigmoid_factor_lo;
    uint sigmoid_last;
    uint pad;
};

// -- helpers ---------------------------------------------------------------------------------------------------------

DEVICE_FN uint min_u32(uint a, uint b) {
    return a < b ? a : b;
}

DEVICE_FN float clamp_f32(float v, float lo, float hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

// The larger value, ignoring NaN like Rust's `f32::max`.
DEVICE_FN float max_f32(float a, float b) {
    return (b > a || a != a) ? b : a;
}

DEVICE_FN u64 join_u64(uint lo, uint hi) {
    return ((u64)hi << 32) | (u64)lo;
}

// `value as i32` in Rust (NaN to 0, saturating) followed by `clamp(lo, hi)`, for an integral `value`.
DEVICE_FN int clamp_to_int(float value, int lo, int hi) {
    if (value != value) {
        return 0 < lo ? lo : (0 > hi ? hi : 0);
    }
    if (value <= (float)lo) {
        return lo;
    }
    if (value >= (float)hi) {
        return hi;
    }
    return (int)value;
}

// Bit `bin % 32` of `word`, the word of a 256-bit set holding `bin` (`BinSplit::goes_left`).
DEVICE_FN uint bit_set(uint word, uint bin) {
    return (word >> (bin & 31u)) & 1u;
}

// `p` with `begin` and `count` resolved to the rows the histogram covers.
DEVICE_FN HistParams hist_rows(HistParams p, CGPTR(uint) left_total) {
    if (p.side == HIST_SIDE_LEFT) {
        p.count = left_total[0];
    } else if (p.side == HIST_SIDE_RIGHT) {
        p.begin += left_total[0];
        p.count -= left_total[0];
    }
    return p;
}

// -- hash RNG and quantization: `silvic_core::quantize::{mix32, uniform, quantize_row, in_bag}` -----------------------

DEVICE_FN uint mix32(uint x) {
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}

DEVICE_FN uint hash_draw(uint seed, uint stream, uint iteration, uint row_lo, uint row_hi) {
    uint row_hash = mix32(row_lo ^ mix32(row_hi));
    return mix32(seed ^ mix32(stream ^ mix32(iteration ^ row_hash)));
}

DEVICE_FN float uniform01(uint seed, uint stream, uint iteration, uint row_lo, uint row_hi) {
    return (float)(hash_draw(seed, stream, iteration, row_lo, row_hi) >> 8) * (1.0f / 16777216.0f);
}

DEVICE_FN int quantize_grad(QuantParams p, float grad, uint row_lo, uint row_hi) {
    float q;
    if (p.stochastic != 0u) {
        float u = uniform01(p.seed, 0u, p.iteration, row_lo, row_hi);
        q = grad >= 0.0f ? F32_TRUNC(F32_FMA(grad, p.inv_grad_scale, u))
                         : F32_TRUNC(F32_FMA(grad, p.inv_grad_scale, -u));
    } else {
        q = F32_RINT(grad * p.inv_grad_scale);
    }
    return clamp_to_int(q, -p.grad_max_q, p.grad_max_q);
}

DEVICE_FN int quantize_hess(QuantParams p, float hess, uint row_lo, uint row_hi) {
    if (p.constant_hessian != 0u) {
        return 1;
    }
    float q;
    if (p.stochastic != 0u) {
        float u = uniform01(p.seed, 1u, p.iteration, row_lo, row_hi);
        q = F32_TRUNC(F32_FMA(hess, p.inv_hess_scale, u));
    } else {
        q = F32_RINT(hess * p.inv_hess_scale);
    }
    return clamp_to_int(q, 0, p.hess_max_q);
}

// `quantize::to_fixed`: `(value * scale).round_ties_even() as i64`.
DEVICE_FN i64 to_fixed(float value, float scale) {
    float q = F32_RINT(value * scale);
    return q != q ? (i64)0 : (i64)q;
}

// Inclusive scan of one value per thread over the threadgroup in `buf` (`size` = GSIZE entries); returns the
// exclusive prefix and leaves the total in `total[0]`, which callers read before their next scan. Every thread must
// call it.
DEVICE_FN uint scan_exclusive(SPTR(uint) buf, uint lid, uint size, uint value, SPTR(uint) total) {
    buf[lid] = value;
    BARRIER();
    for (uint offset = 1u; offset < size; offset <<= 1) {
        uint add = lid >= offset ? buf[lid - offset] : 0u;
        BARRIER();
        buf[lid] += add;
        BARRIER();
    }
    uint inclusive = buf[lid];
    if (lid == size - 1u) {
        total[0] = inclusive;
    }
    BARRIER();
    return inclusive - value;
}
