//! Block scoring: rows are scored `LANES` at a time, one byte lane per row, as in RapidScorer (Ye et al., KDD 2018):
//! a tree's leaf bitvector is stored byte-sliced, byte `b` of every row's bitvector side by side in one vector, so an
//! op (one node, one byte) is a vector compare of the rows' codes, an OR with the op's kept bits, and an AND into
//! that byte. Every op of a tree runs on every block, with no branch on the data; the leaf is the position of each
//! lane's lowest set bit, and leaf values are added to each row's score in tree order, in f64, as `GBDT::PredictRaw`
//! adds them.
//!
//! Ops on one byte form a chain through memory (load, AND, store), so `compile` orders a tree's ops round-robin
//! over its bytes. The kernels differ only in the vector unit: `portable` is plain Rust over u64 words, `neon`
//! aarch64 NEON, `avx2` and `avx512` x86-64 AVX2 and AVX-512BW, entered only after runtime detection.

use std::ops::Range;

use silvic_core::data::{DenseData, DenseMatrix, Layout};

use crate::compile::{FastTree, Forest, Plan, LEAF_WINDOW, MAX_LEAF_BYTES};

#[cfg(target_arch = "x86_64")]
pub(crate) mod avx2;
#[cfg(target_arch = "x86_64")]
pub(crate) mod avx512;
#[cfg(target_arch = "aarch64")]
pub(crate) mod neon;
pub(crate) mod portable;

/// Leaf bitvectors of one block: byte `b` of every lane's bitvector in `bytes[b]`.
pub(crate) type LeafBytes<const LANES: usize> = [[u8; LANES]; MAX_LEAF_BYTES];

/// A vector unit that scores `LANES` rows at once.
pub(crate) trait BlockKernel<const LANES: usize>: Copy {
    /// XOR-ed into every code the kernel reads (`avx2` biases them to compare as signed bytes).
    const CODE_BIAS: u8 = 0;

    /// Adds each lane's leaf value of every plan to `scores`, in plan order. `codes` holds the block's codes per
    /// slot and `rows` its `LANES` rows of `forest.num_features` values.
    fn score_block(
        self,
        forest: &Forest,
        plans: &[Plan],
        codes: &[[u8; LANES]],
        rows: &[f64],
        scores: &mut [f64; LANES],
    );
}

/// The plan loop every kernel runs: `leaf_positions` leaves each lane's leaf position (left to right) of a fast tree
/// in its second argument; walked trees take each row through `Tree::leaf_index`.
#[inline(always)]
pub(crate) fn score_plans<const LANES: usize>(
    forest: &Forest,
    plans: &[Plan],
    rows: &[f64],
    scores: &mut [f64; LANES],
    mut leaf_positions: impl FnMut(&FastTree, &mut [u8; LANES]),
) {
    let mut positions = [0u8; LANES];
    for plan in plans {
        match plan {
            Plan::Fast(tree) => {
                leaf_positions(tree, &mut positions);
                let window: &[f64; LEAF_WINDOW] = forest.leaf_values[tree.leaves as usize..][..LEAF_WINDOW]
                    .try_into()
                    .expect("a window of leaf values");
                for (score, &position) in scores.iter_mut().zip(&positions) {
                    *score += window[usize::from(position)];
                }
            }
            Plan::Walk(index) => {
                let tree = &forest.walked[*index as usize];
                for (score, row) in scores.iter_mut().zip(rows.chunks_exact(forest.num_features)) {
                    *score += tree.leaf_value[tree.leaf_index(row)];
                }
            }
        }
    }
}

/// `codes` cut to its power-of-two length, and the mask for slot indices: every slot is below the length, so
/// `slot & mask` is the slot, and indexing with it needs no bounds check.
#[inline(always)]
pub(crate) fn masked<const LANES: usize>(codes: &[[u8; LANES]]) -> (&[[u8; LANES]], usize) {
    debug_assert!(codes.len().is_power_of_two());
    let mask = codes.len() - 1;
    (&codes[..=mask], mask)
}

/// Scores rows `first_row..first_row + out.len()` of `matrix` with `plans` into `out` (raw scores).
pub(crate) fn score_rows<K: BlockKernel<LANES>, const LANES: usize>(
    kernel: K,
    forest: &Forest,
    plans: &[Plan],
    matrix: &DenseMatrix<'_>,
    first_row: usize,
    out: &mut [f64],
) {
    let mut rows = vec![0.0f64; LANES * forest.num_features];
    let mut codes = vec![[0u8; LANES]; forest.slots.next_power_of_two()];
    for (block, out) in out.chunks_mut(LANES).enumerate() {
        let start = first_row + block * LANES;
        gather(matrix, start..start + out.len(), forest.num_features, &mut rows);
        forest.encode(&rows, &mut codes, K::CODE_BIAS);
        let mut scores = [0.0f64; LANES];
        kernel.score_block(forest, plans, &codes, &rows, &mut scores);
        out.copy_from_slice(&scores[..out.len()]);
    }
}

/// Copies rows `range` of `matrix` (first `width` columns, as f64) into the rows of `rows`; lanes past the range
/// repeat its last row, so every lane holds a real row.
fn gather(matrix: &DenseMatrix<'_>, range: Range<usize>, width: usize, rows: &mut [f64]) {
    if width == 0 {
        return;
    }
    let last = range.end - 1;
    for (lane, row) in rows.chunks_exact_mut(width).enumerate() {
        let source = (range.start + lane).min(last);
        match (matrix.data, matrix.layout) {
            (DenseData::F64(values), Layout::RowMajor) => {
                row.copy_from_slice(&values[source * matrix.cols..][..width]);
            }
            (DenseData::F32(values), Layout::RowMajor) => {
                for (out, &value) in row.iter_mut().zip(&values[source * matrix.cols..][..width]) {
                    *out = f64::from(value);
                }
            }
            (_, Layout::ColumnMajor) => {
                for (col, out) in row.iter_mut().enumerate() {
                    *out = matrix.get(source, col);
                }
            }
        }
    }
}
