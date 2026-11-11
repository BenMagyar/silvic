//! The block kernel on x86-64 AVX-512BW: 64 lanes in one 512-bit vector. Unsigned byte compares produce mask
//! registers, so an op is a compare and a masked move of the cleared byte. Class tables and trailing zeros use the
//! same `vpshufb` lookups as `avx2`, on four 128-bit lanes.

use std::arch::x86_64::{
    __m512i, _mm512_add_epi8, _mm512_adds_epu8, _mm512_and_si512, _mm512_broadcast_i32x4, _mm512_cmpgt_epu8_mask,
    _mm512_loadu_si512, _mm512_mask_min_epu8, _mm512_mask_mov_epi8, _mm512_min_epu8, _mm512_or_si512, _mm512_set1_epi8,
    _mm512_setzero_si512, _mm512_shuffle_epi8, _mm512_srli_epi16, _mm512_storeu_si512, _mm512_test_epi8_mask,
    _mm512_xor_si512, _mm_loadu_si128,
};

use crate::compile::{FastTree, Forest, Plan, MAX_CLASSES, MAX_LEAF_BYTES};
use crate::kernel::{masked, score_plans, BlockKernel, LeafBytes};

pub(crate) const LANES: usize = 64;

/// Proof that the CPU runs AVX-512F and AVX-512BW: only [`Avx512::detect`] makes one.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Avx512(());

impl Avx512 {
    pub(crate) fn detect() -> Option<Avx512> {
        (is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512bw")).then_some(Avx512(()))
    }
}

impl BlockKernel<LANES> for Avx512 {
    fn score_block(
        self,
        forest: &Forest,
        plans: &[Plan],
        codes: &[[u8; LANES]],
        rows: &[f64],
        scores: &mut [f64; LANES],
    ) {
        // SAFETY: `self` exists only after `is_x86_feature_detected!` found AVX-512F and AVX-512BW.
        unsafe { score_block(forest, plans, codes, rows, scores) }
    }
}

#[target_feature(enable = "avx512f,avx512bw")]
fn score_block(forest: &Forest, plans: &[Plan], codes: &[[u8; LANES]], rows: &[f64], scores: &mut [f64; LANES]) {
    let mut bytes = [[0u8; LANES]; MAX_LEAF_BYTES];
    score_plans(forest, plans, rows, scores, |tree, positions| {
        leaf_positions(forest, tree, codes, &mut bytes, positions);
    });
}

#[target_feature(enable = "avx512f,avx512bw")]
#[inline]
fn load(bytes: &[u8; LANES]) -> __m512i {
    // SAFETY: `bytes` is 64 readable bytes, exactly what the unaligned load reads.
    unsafe { _mm512_loadu_si512(bytes.as_ptr().cast()) }
}

#[target_feature(enable = "avx512f,avx512bw")]
#[inline]
fn store(bytes: &mut [u8; LANES], vector: __m512i) {
    // SAFETY: `bytes` is 64 writable bytes, exactly what the unaligned store writes.
    unsafe { _mm512_storeu_si512(bytes.as_mut_ptr().cast(), vector) }
}

/// 0xFF in the lanes whose class `table` sends left (see `avx2` for the index arithmetic).
#[target_feature(enable = "avx512f,avx512bw")]
#[inline]
fn lookup(table: &[u8; MAX_CLASSES], classes: __m512i) -> __m512i {
    let mut left = _mm512_setzero_si512();
    for (chunk, entries) in table.as_chunks::<16>().0.iter().enumerate() {
        // SAFETY: `entries` is 16 readable bytes, exactly what the unaligned load reads.
        let entries = _mm512_broadcast_i32x4(unsafe { _mm_loadu_si128(entries.as_ptr().cast()) });
        let index = _mm512_adds_epu8(
            _mm512_xor_si512(classes, _mm512_set1_epi8((chunk * 16) as i8)),
            _mm512_set1_epi8(0x70),
        );
        left = _mm512_or_si512(left, _mm512_shuffle_epi8(entries, index));
    }
    left
}

/// Trailing zeros of each nibble value, 8 for 0, repeated in the four 128-bit lanes `vpshufb` indexes separately.
const fn nibble_zeros(offset: u8) -> [u8; LANES] {
    let mut table = [0u8; LANES];
    let mut index = 0;
    while index < LANES {
        let nibble = index % 16;
        table[index] = if nibble == 0 {
            8
        } else {
            offset + nibble.trailing_zeros() as u8
        };
        index += 1;
    }
    table
}

const LOW_ZEROS: [u8; LANES] = nibble_zeros(0);
const HIGH_ZEROS: [u8; LANES] = nibble_zeros(4);

#[target_feature(enable = "avx512f,avx512bw")]
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
    let ones = _mm512_set1_epi8(-1);
    for byte in &mut bytes[..used] {
        store(byte, ones);
    }
    for &op in &forest.categorical_ops[tree.categorical_start as usize..tree.categorical_end as usize] {
        let left = lookup(
            &forest.tables[op.table as usize],
            load(&codes[op.slot as usize & slot_mask]),
        );
        let keep = _mm512_set1_epi8(op.keep as i8);
        let byte = &mut bytes[op.byte()];
        store(byte, _mm512_and_si512(load(byte), _mm512_or_si512(left, keep)));
    }
    for &op in &forest.numeric_ops[tree.numeric_start as usize..tree.numeric_end as usize] {
        let right = _mm512_cmpgt_epu8_mask(
            load(&codes[op.slot() & slot_mask]),
            _mm512_set1_epi8(op.threshold() as i8),
        );
        let byte = &mut bytes[op.byte()];
        let bits = load(byte);
        let kept = _mm512_and_si512(bits, _mm512_set1_epi8(op.keep() as i8));
        store(byte, _mm512_mask_mov_epi8(bits, right, kept));
    }
    let low_zeros = load(&LOW_ZEROS);
    let high_zeros = load(&HIGH_ZEROS);
    let nibble = _mm512_set1_epi8(0x0F);
    let mut position = _mm512_set1_epi8(-1);
    for (index, byte) in bytes[..used].iter().enumerate() {
        let bits = load(byte);
        let low = _mm512_shuffle_epi8(low_zeros, _mm512_and_si512(bits, nibble));
        let high = _mm512_shuffle_epi8(high_zeros, _mm512_and_si512(_mm512_srli_epi16::<4>(bits), nibble));
        let candidate = _mm512_add_epi8(_mm512_min_epu8(low, high), _mm512_set1_epi8((index * 8) as i8));
        position = _mm512_mask_min_epu8(position, _mm512_test_epi8_mask(bits, bits), position, candidate);
    }
    store(positions, position);
}
