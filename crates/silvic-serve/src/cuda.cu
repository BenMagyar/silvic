using u64 = unsigned long long;
using u32 = unsigned int;

template <typename Real>
struct Node;

template <>
struct Node<double> {
    double threshold;
    u64 cat_start;
    int left;
    int right;
    u32 feature;
    u32 decision;
    u64 cat_len;
};

template <>
struct Node<float> {
    float threshold;
    u32 padding;
    u64 cat_start;
    int left;
    int right;
    u32 feature;
    u32 decision;
    u64 cat_len;
};

struct Tree {
    u64 nodes;
    u64 leaves;
    u64 num_nodes;
};

struct Params {
    u64 rows;
    u64 cols;
    u64 trees;
    u64 column_major;
};

static_assert(sizeof(Node<double>) == 40);
static_assert(sizeof(Node<float>) == 40);
static_assert(sizeof(Tree) == 24);
static_assert(sizeof(Params) == 32);

__device__ double absolute(double value) { return fabs(value); }
__device__ float absolute(float value) { return fabsf(value); }
__device__ u32 category_index(double value) {
    return value >= 2147483647.0 ? 2147483647u : u32(__double2int_rz(value));
}
__device__ u32 category_index(float value) {
    return value >= 2147483648.0f ? 2147483647u : u32(__float2int_rz(value));
}
__device__ double ordered_add(double left, double right) { return __dadd_rn(left, right); }
__device__ float ordered_add(float left, float right) { return __fadd_rn(left, right); }

template <typename Real>
__device__ bool goes_left(const Node<Real>& node, Real value, const u32* categories) {
    constexpr Real zero_threshold = Real(1e-35f);
    if (absolute(value) <= zero_threshold) {
        value = Real(0);
    }
    if (node.decision & 1) {
        // Match saturating, truncating category casts, including (-1, 0) -> 0.
        if (isnan(value) || value <= Real(-1)) {
            return false;
        }
        const u32 category = category_index(value);
        const u64 word = category / 32;
        return word < node.cat_len && ((categories[node.cat_start + word] >> (category % 32)) & 1);
    }
    const u32 missing = (node.decision >> 2) & 3;
    if (isnan(value) && missing != 2) {
        value = Real(0);
    }
    const bool is_missing = (missing == 1 && -zero_threshold <= value && value <= zero_threshold)
        || (missing == 2 && isnan(value));
    return is_missing ? bool(node.decision & 2) : value <= node.threshold;
}

template <typename Input, typename Real>
__device__ void predict_rows(
    const Input* input,
    const Tree* trees,
    const Node<Real>* nodes,
    const Real* leaves,
    const u32* categories,
    Real* output,
    Params params
) {
    __shared__ Real selected[8 * 32];
    const u32 row_lane = threadIdx.x & 31;
    const u32 tree_lane = threadIdx.x >> 5;
    const u64 tree_warps = blockDim.x >> 5;
    const u64 stride = u64(gridDim.x) * 32;
    for (u64 first_row = u64(blockIdx.x) * 32; first_row < params.rows; first_row += stride) {
        const u64 row = first_row + row_lane;
        Real score = Real(0);
        for (u64 first_tree = 0; first_tree < params.trees; first_tree += tree_warps) {
            const u64 t = first_tree + tree_lane;
            if (row < params.rows && t < params.trees) {
                const Tree tree = trees[t];
                int child = tree.num_nodes ? 0 : -1;
                while (child >= 0) {
                    const Node<Real> node = nodes[tree.nodes + u32(child)];
                    const u64 offset = params.column_major ? u64(node.feature) * params.rows + row
                                                           : row * params.cols + node.feature;
                    child = goes_left(node, Real(input[offset]), categories) ? node.left : node.right;
                }
                selected[threadIdx.x] = leaves[tree.leaves + u32(~child)];
            }
            __syncthreads();
            if (tree_lane == 0 && row < params.rows) {
                const u64 remaining = params.trees - first_tree;
                const u32 count = u32(remaining < tree_warps ? remaining : tree_warps);
                // Walks run concurrently; the first warp adds their leaves in the original tree order.
                for (u32 tree = 0; tree < count; ++tree) {
                    score = ordered_add(score, selected[tree * 32 + row_lane]);
                }
            }
            __syncthreads();
        }
        if (tree_lane == 0 && row < params.rows) {
            output[row] = score;
        }
    }
}

extern "C" __global__ void predict_f32(
    const float* input,
    const Tree* trees,
    const Node<double>* nodes,
    const double* leaves,
    const u32* categories,
    double* output,
    Params params
) {
    predict_rows(input, trees, nodes, leaves, categories, output, params);
}

extern "C" __global__ void predict_f64(
    const double* input,
    const Tree* trees,
    const Node<double>* nodes,
    const double* leaves,
    const u32* categories,
    double* output,
    Params params
) {
    predict_rows(input, trees, nodes, leaves, categories, output, params);
}

extern "C" __global__ void predict_f32_f32(
    const float* input,
    const Tree* trees,
    const Node<float>* nodes,
    const float* leaves,
    const u32* categories,
    float* output,
    Params params
) {
    predict_rows(input, trees, nodes, leaves, categories, output, params);
}

extern "C" __global__ void predict_f32_f64(
    const double* input,
    const Tree* trees,
    const Node<float>* nodes,
    const float* leaves,
    const u32* categories,
    float* output,
    Params params
) {
    predict_rows(input, trees, nodes, leaves, categories, output, params);
}
