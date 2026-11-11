//! The block kernel on x86-64 AVX2: 64 lanes as two 256-bit vectors. AVX2 compares bytes only as signed, so this
//! kernel's codes carry a bias of 0x80 ([`BlockKernel::CODE_BIAS`]) and an unsigned `code <= threshold` is one signed
//! compare against the biased `threshold + 1`. `vpshufb` looks up 16 bytes, so a 64-byte class table is four lookups, each
//! indexing with `(class ^ 16k) +sat 0x70` (unbiasing in the same XOR): classes of chunk `k` land on `0x70..=0x7F`
//! and every other class (classless 255 included) on `0x80` or above, which `vpshufb` reads as 0. A byte's trailing
//! zeros are the smaller of two nibble lookups.

use std::arch::x86_64::{
    __m256i, _mm256_add_epi8, _mm256_adds_epu8, _mm256_and_si256, _mm256_broadcastsi128_si256, _mm256_cmpeq_epi8,
    _mm256_cmpgt_epi8, _mm256_loadu_si256, _mm256_min_epu8, _mm256_or_si256, _mm256_set1_epi8, _mm256_setr_epi8,
    _mm256_setzero_si256, _mm256_shuffle_epi8, _mm256_srli_epi16, _mm256_storeu_si256, _mm256_xor_si256,
    _mm_loadu_si128,
};

use crate::compile::{FastTree, Forest, Plan, MAX_CLASSES, MAX_LEAF_BYTES};
use crate::kernel::{masked, score_plans, BlockKernel, LeafBytes};

pub(crate) const LANES: usize = 64;
/// Added (XOR) to every code so signed byte compares order codes as unsigned.
const BIAS: u8 = 0x80;

type Lanes = [__m256i; 2];

/// Proof that the CPU runs AVX2: only [`Avx2::detect`] makes one.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Avx2(());

impl Avx2 {
    pub(crate) fn detect() -> Option<Avx2> {
        is_x86_feature_detected!("avx2").then_some(Avx2(()))
    }
}

impl BlockKernel<LANES> for Avx2 {
    const CODE_BIAS: u8 = BIAS;

    fn score_block(
        self,
        forest: &Forest,
        plans: &[Plan],
        codes: &[[u8; LANES]],
        rows: &[f64],
        scores: &mut [f64; LANES],
    ) {
        // SAFETY: `self` exists only after `is_x86_feature_detected!("avx2")` returned true.
        unsafe { score_block(forest, plans, codes, rows, scores) }
    }
}

#[target_feature(enable = "avx2")]
fn score_block(forest: &Forest, plans: &[Plan], codes: &[[u8; LANES]], rows: &[f64], scores: &mut [f64; LANES]) {
    let mut bytes = [[0u8; LANES]; MAX_LEAF_BYTES];
    score_plans(forest, plans, rows, scores, |tree, positions| {
        leaf_positions(forest, tree, codes, &mut bytes, positions);
    });
}

#[target_feature(enable = "avx2")]
#[inline]
fn load(bytes: &[u8; LANES]) -> Lanes {
    // SAFETY: each unaligned 32-byte load reads inside `bytes`.
    std::array::from_fn(|half| unsafe { _mm256_loadu_si256(bytes[32 * half..].as_ptr().cast()) })
}

#[target_feature(enable = "avx2")]
#[inline]
fn store(bytes: &mut [u8; LANES], vector: Lanes) {
    for (half, part) in vector.into_iter().enumerate() {
        // SAFETY: each unaligned 32-byte store writes inside `bytes`.
        unsafe { _mm256_storeu_si256(bytes[32 * half..].as_mut_ptr().cast(), part) }
    }
}

/// 0xFF in the lanes whose class (biased) `table` sends left.
#[target_feature(enable = "avx2")]
#[inline]
fn lookup(table: &[u8; MAX_CLASSES], classes: __m256i) -> __m256i {
    let mut left = _mm256_setzero_si256();
    for (chunk, entries) in table.as_chunks::<16>().0.iter().enumerate() {
        // SAFETY: `entries` is 16 readable bytes, exactly what the unaligned load reads.
        let entries = _mm256_broadcastsi128_si256(unsafe { _mm_loadu_si128(entries.as_ptr().cast()) });
        let local = _mm256_xor_si256(classes, _mm256_set1_epi8((BIAS ^ (chunk * 16) as u8) as i8));
        let index = _mm256_adds_epu8(local, _mm256_set1_epi8(0x70));
        left = _mm256_or_si256(left, _mm256_shuffle_epi8(entries, index));
    }
    left
}

#[target_feature(enable = "avx2")]
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
        store(byte, [_mm256_set1_epi8(-1); 2]);
    }
    for &op in &forest.categorical_ops[tree.categorical_start as usize..tree.categorical_end as usize] {
        let table = &forest.tables[op.table as usize];
        let classes = load(&codes[op.slot as usize & slot_mask]);
        let keep = _mm256_set1_epi8(op.keep as i8);
        let byte = &mut bytes[op.byte()];
        let bits = load(byte);
        store(
            byte,
            std::array::from_fn(|h| _mm256_and_si256(bits[h], _mm256_or_si256(lookup(table, classes[h]), keep))),
        );
    }
    for &op in &forest.numeric_ops[tree.numeric_start as usize..tree.numeric_end as usize] {
        let row_codes = load(&codes[op.slot() & slot_mask]);
        // Local thresholds stay below 254, so `code <= threshold` is `threshold + 1 > code` without overflow.
        let above = _mm256_set1_epi8(((op.threshold() + 1) ^ BIAS) as i8);
        let keep = _mm256_set1_epi8(op.keep() as i8);
        let byte = &mut bytes[op.byte()];
        let bits = load(byte);
        store(
            byte,
            std::array::from_fn(|h| {
                _mm256_and_si256(bits[h], _mm256_or_si256(_mm256_cmpgt_epi8(above, row_codes[h]), keep))
            }),
        );
    }
    // Trailing zeros of a nibble, 8 for none: the low nibble's count, else 4 plus the high nibble's.
    let low_zeros = _mm256_setr_epi8(
        8, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 8, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0,
    );
    let high_zeros = _mm256_setr_epi8(
        8, 4, 5, 4, 6, 4, 5, 4, 7, 4, 5, 4, 6, 4, 5, 4, 8, 4, 5, 4, 6, 4, 5, 4, 7, 4, 5, 4, 6, 4, 5, 4,
    );
    let nibble = _mm256_set1_epi8(0x0F);
    let zero = _mm256_setzero_si256();
    let mut found = [_mm256_set1_epi8(-1); 2];
    for (index, byte) in bytes[..used].iter().enumerate() {
        let base = _mm256_set1_epi8((index * 8) as i8);
        for (position, bits) in found.iter_mut().zip(load(byte)) {
            let low = _mm256_shuffle_epi8(low_zeros, _mm256_and_si256(bits, nibble));
            let high = _mm256_shuffle_epi8(high_zeros, _mm256_and_si256(_mm256_srli_epi16::<4>(bits), nibble));
            let candidate = _mm256_add_epi8(_mm256_min_epu8(low, high), base);
            // An empty byte's candidate becomes 255, which never wins the minimum over a nonempty byte.
            *position = _mm256_min_epu8(*position, _mm256_or_si256(candidate, _mm256_cmpeq_epi8(bits, zero)));
        }
    }
    store(positions, found);
}
