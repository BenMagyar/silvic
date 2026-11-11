//! The kernels' parameter structs, mirrored field for field from `kernels/common.h`, and the capacities both sides
//! share. Every field is 4 bytes, so `#[repr(C)]` lays the structs out exactly as Metal and CUDA do.

use bytemuck::{Pod, Zeroable};
use silvic_core::data::GROUP_BIN_BUDGET;

/// Bins of one quantized histogram work unit: a whole feature group, one packed i32 per bin (32 KiB of threadgroup
/// memory, Metal's limit).
pub(crate) const HIST_QUANT_BINS: u32 = GROUP_BIN_BUDGET;
/// Bins of one float histogram work unit: separate f32 g and h arrays fill the same 32 KiB.
pub(crate) const HIST_FLOAT_BINS: u32 = GROUP_BIN_BUDGET / 2;
/// Threads of the reduction kernels (`grad_stats`, `leaf_sums`); a power of two.
pub(crate) const REDUCE_THREADS: u32 = 256;
/// Threads of the compaction kernels (`bag_*`, `partition_*`, `scan_blocks`); a power of two.
pub(crate) const SCAN_THREADS: u32 = 256;
/// Largest query each lambdarank kernel holds in threadgroup memory, and the threads it runs with.
pub(crate) const LAMBDARANK_CLASSES: [(u32, u32); 3] = [(64, 64), (256, 256), (1024, 256)];
/// Threads of the widest lambdarank kernel: the size of its reduction array.
pub(crate) const LAMBDARANK_THREADS_MAX: u32 = 256;
/// `HistParams::side`: the leaf's own rows, or its left or right child after a partition encoded just before.
pub(crate) const HIST_SIDE_LEAF: u32 = 0;
pub(crate) const HIST_SIDE_LEFT: u32 = 1;
pub(crate) const HIST_SIDE_RIGHT: u32 = 2;

const COMMON: &str = include_str!("../kernels/common.h");
const KERNELS: &str = include_str!("../kernels/kernels.kern");

/// The complete kernel source for one API: `prelude`, the shared capacities as `#define`s, `common.h` and
/// `kernels.kern`.
pub fn kernel_source(prelude: &str) -> String {
    let defines = [
        ("HIST_QUANT_BINS", HIST_QUANT_BINS),
        ("HIST_FLOAT_BINS", HIST_FLOAT_BINS),
        ("REDUCE_THREADS", REDUCE_THREADS),
        ("SCAN_THREADS", SCAN_THREADS),
        ("LAMBDARANK_CAP_SMALL", LAMBDARANK_CLASSES[0].0),
        ("LAMBDARANK_CAP_MEDIUM", LAMBDARANK_CLASSES[1].0),
        ("LAMBDARANK_CAP_LARGE", LAMBDARANK_CLASSES[2].0),
        ("LAMBDARANK_THREADS_MAX", LAMBDARANK_THREADS_MAX),
        ("HIST_SIDE_LEFT", HIST_SIDE_LEFT),
        ("HIST_SIDE_RIGHT", HIST_SIDE_RIGHT),
    ];
    let mut source = String::with_capacity(prelude.len() + COMMON.len() + KERNELS.len() + 512);
    source.push_str(prelude);
    source.push('\n');
    for (name, value) in defines {
        source.push_str(&format!("#define {name} {value}u\n"));
    }
    source.push_str(COMMON);
    source.push('\n');
    source.push_str(KERNELS);
    source
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default, Pod, Zeroable)]
pub(crate) struct FillParams {
    pub n: u32,
    pub value: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default, Pod, Zeroable)]
pub(crate) struct GradParams {
    pub n: u32,
    pub has_weight: u32,
    pub sigmoid: f32,
    pub pad: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default, Pod, Zeroable)]
pub(crate) struct StatsParams {
    pub n: u32,
    pub pad: [u32; 3],
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default, Pod, Zeroable)]
pub(crate) struct QuantParams {
    pub n: u32,
    pub row_offset_lo: u32,
    pub row_offset_hi: u32,
    pub inv_grad_scale: f32,
    pub inv_hess_scale: f32,
    pub grad_max_q: i32,
    pub hess_max_q: i32,
    pub constant_hessian: u32,
    pub stochastic: u32,
    pub seed: u32,
    pub iteration: u32,
    pub pad: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default, Pod, Zeroable)]
pub(crate) struct BagParams {
    pub n: u32,
    pub chunk: u32,
    pub row_offset_lo: u32,
    pub row_offset_hi: u32,
    pub seed: u32,
    pub round: u32,
    pub threshold: u32,
    pub pad: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default, Pod, Zeroable)]
pub(crate) struct ScanParams {
    pub n: u32,
    pub pad: [u32; 3],
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default, Pod, Zeroable)]
pub(crate) struct PartitionParams {
    pub begin: u32,
    pub count: u32,
    pub chunk: u32,
    pub row_stride: u32,
    pub group_base_lo: u32,
    pub group_base_hi: u32,
    pub column: u32,
    pub pad: u32,
    pub left_bins: [u32; 8],
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default, Pod, Zeroable)]
pub(crate) struct CopyParams {
    pub begin: u32,
    pub count: u32,
    pub pad: [u32; 2],
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default, Pod, Zeroable)]
pub(crate) struct HistUnit {
    pub words_base_lo: u32,
    pub words_base_hi: u32,
    pub row_words: u32,
    pub bin_begin: u32,
    pub num_bins: u32,
    pub cols_begin: u32,
    pub pad: [u32; 2],
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default, Pod, Zeroable)]
pub(crate) struct HistParams {
    pub begin: u32,
    pub count: u32,
    pub rows_per_block: u32,
    pub side: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default, Pod, Zeroable)]
pub(crate) struct LeafSumParams {
    pub grad_scale: f32,
    pub hess_scale: f32,
    pub pad: [u32; 2],
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default, Pod, Zeroable)]
pub(crate) struct GroupLayout {
    pub base_lo: u32,
    pub base_hi: u32,
    pub row_stride: u32,
    pub pad: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default, Pod, Zeroable)]
pub(crate) struct TreeNode {
    pub group: u32,
    pub column: u32,
    pub left: i32,
    pub right: i32,
    pub left_bins: [u32; 8],
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default, Pod, Zeroable)]
pub(crate) struct TreeParams {
    pub n: u32,
    pub num_nodes: u32,
    pub pad: [u32; 2],
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default, Pod, Zeroable)]
pub(crate) struct RankParams {
    pub query_begin: u32,
    pub num_queries: u32,
    pub has_weight: u32,
    pub norm: u32,
    pub truncation_level: u32,
    pub sigmoid: f32,
    pub sigmoid_min_hi: f32,
    pub sigmoid_min_lo: f32,
    pub sigmoid_factor_hi: f32,
    pub sigmoid_factor_lo: f32,
    pub sigmoid_last: u32,
    pub pad: u32,
}

// The sizes of the matching structs in `kernels/common.h`.
const _: () = assert!(std::mem::size_of::<FillParams>() == 8);
const _: () = assert!(std::mem::size_of::<GradParams>() == 16);
const _: () = assert!(std::mem::size_of::<StatsParams>() == 16);
const _: () = assert!(std::mem::size_of::<QuantParams>() == 48);
const _: () = assert!(std::mem::size_of::<BagParams>() == 32);
const _: () = assert!(std::mem::size_of::<ScanParams>() == 16);
const _: () = assert!(std::mem::size_of::<PartitionParams>() == 64);
const _: () = assert!(std::mem::size_of::<CopyParams>() == 16);
const _: () = assert!(std::mem::size_of::<HistUnit>() == 32);
const _: () = assert!(std::mem::size_of::<HistParams>() == 16);
const _: () = assert!(std::mem::size_of::<LeafSumParams>() == 16);
const _: () = assert!(std::mem::size_of::<GroupLayout>() == 16);
const _: () = assert!(std::mem::size_of::<TreeNode>() == 48);
const _: () = assert!(std::mem::size_of::<TreeParams>() == 16);
const _: () = assert!(std::mem::size_of::<RankParams>() == 48);
