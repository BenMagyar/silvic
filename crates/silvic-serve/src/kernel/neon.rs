//! The block kernel on aarch64 NEON (part of every aarch64 target): 64 lanes as four 128-bit vectors. `vqtbl4q_u8`
//! looks classes up in a categorical split's 64-byte table, and reads 0 for indices past it, so classless categories
//! go right; `vrbitq_u8` and `vclzq_u8` count each byte's trailing zeros for the leaf position.

use std::arch::aarch64::{
    uint8x16_t, vandq_u8, vbslq_u8, vcleq_u8, vclzq_u8, vdupq_n_u8, vld1q_u8, vld1q_u8_x4, vld2q_dup_u8, vorrq_u8,
    vqtbl4q_u8, vrbitq_u8, vst1q_u8, vtstq_u8,
};

use crate::compile::{FastTree, Forest, Plan, MAX_CLASSES, MAX_LEAF_BYTES};
use crate::kernel::{masked, score_plans, BlockKernel, LeafBytes};

pub(crate) const LANES: usize = 64;
const QUADS: usize = LANES / 16;

type Lanes = [uint8x16_t; QUADS];

#[derive(Debug, Clone, Copy)]
pub(crate) struct Neon;

impl BlockKernel<LANES> for Neon {
    fn score_block(
        self,
        forest: &Forest,
        plans: &[Plan],
        codes: &[[u8; LANES]],
        rows: &[f64],
        scores: &mut [f64; LANES],
    ) {
        // SAFETY: every aarch64 target rustc supports has NEON in its baseline features.
        unsafe { score_block(forest, plans, codes, rows, scores) }
    }
}

#[target_feature(enable = "neon")]
fn score_block(forest: &Forest, plans: &[Plan], codes: &[[u8; LANES]], rows: &[f64], scores: &mut [f64; LANES]) {
    let mut bytes = [[0u8; LANES]; MAX_LEAF_BYTES];
    score_plans(forest, plans, rows, scores, |tree, positions| {
        leaf_positions(forest, tree, codes, &mut bytes, positions);
    });
}

#[target_feature(enable = "neon")]
#[inline]
fn load(bytes: &[u8; LANES]) -> Lanes {
    // SAFETY: each 16-byte load reads inside `bytes`, with no alignment requirement.
    std::array::from_fn(|quad| unsafe { vld1q_u8(bytes[16 * quad..].as_ptr()) })
}

#[target_feature(enable = "neon")]
#[inline]
fn store(bytes: &mut [u8; LANES], vector: Lanes) {
    for (quad, part) in vector.into_iter().enumerate() {
        // SAFETY: each 16-byte store writes inside `bytes`, with no alignment requirement.
        unsafe { vst1q_u8(bytes[16 * quad..].as_mut_ptr(), part) }
    }
}

#[target_feature(enable = "neon")]
#[inline]
fn leaf_positions(
    forest: &Forest,
    tree: &FastTree,
    codes: &[[u8; LANES]],
    bytes: &mut LeafBytes<LANES>,
    positions: &mut [u8; LANES],
) {
    let (codes, slot_mask) = masked(codes);
    let used = tree.leaf_bytes as usize;
    for byte in &mut bytes[..used] {
        store(byte, [vdupq_n_u8(u8::MAX); QUADS]);
    }
    for &op in &forest.categorical_ops[tree.categorical_start as usize..tree.categorical_end as usize] {
        let table: &[u8; MAX_CLASSES] = &forest.tables[op.table as usize];
        // SAFETY: `table` is 64 readable bytes, exactly what `vld1q_u8_x4` reads, with no alignment requirement.
        let table = unsafe { vld1q_u8_x4(table.as_ptr()) };
        let classes = load(&codes[op.slot as usize & slot_mask]);
        let keep = vdupq_n_u8(op.keep);
        let byte = &mut bytes[op.byte()];
        let bits = load(byte);
        store(
            byte,
            std::array::from_fn(|q| vandq_u8(bits[q], vorrq_u8(vqtbl4q_u8(table, classes[q]), keep))),
        );
    }
    for op in &forest.numeric_ops[tree.numeric_start as usize..tree.numeric_end as usize] {
        let row_codes = load(&codes[op.slot() & slot_mask]);
        // SAFETY: `threshold_keep` is 2 readable bytes, exactly what `vld2q_dup_u8` reads.
        let broadcast = unsafe { vld2q_dup_u8(op.threshold_keep.as_ptr()) };
        let (threshold, keep) = (broadcast.0, broadcast.1);
        let byte = &mut bytes[op.byte()];
        let bits = load(byte);
        store(
            byte,
            std::array::from_fn(|q| vandq_u8(bits[q], vorrq_u8(vcleq_u8(row_codes[q], threshold), keep))),
        );
    }
    let mut found = [vdupq_n_u8(0); QUADS];
    for (index, byte) in bytes[..used].iter().enumerate().rev() {
        let bits = load(byte);
        // A byte's trailing zeros are below 8, so OR-ing in the byte's first position adds it.
        let base = vdupq_n_u8(index as u8 * 8);
        for (position, bits) in found.iter_mut().zip(bits) {
            *position = vbslq_u8(
                vtstq_u8(bits, bits),
                vorrq_u8(vclzq_u8(vrbitq_u8(bits)), base),
                *position,
            );
        }
    }
    store(positions, found);
}
