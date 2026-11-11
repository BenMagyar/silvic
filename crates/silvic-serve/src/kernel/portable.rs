//! The block kernel in plain Rust for any CPU: 32 lanes as four u64 words of eight byte lanes (SWAR), with no
//! intrinsics. Byte compares use the carry-free subtraction from Hacker's Delight (§2-18), and leaf positions come
//! from an 8×8 byte transpose (three swap stages) that turns eight leaf bytes of eight lanes into one bitvector word
//! per lane.

use crate::compile::{FastTree, Forest, Plan, MAX_LEAF_BYTES};
use crate::kernel::{masked, score_plans, BlockKernel};

pub(crate) const LANES: usize = 32;
const WORDS: usize = LANES / 8;
const ONES: u64 = 0x0101_0101_0101_0101;
const LOW_BITS: u64 = 0x7F7F_7F7F_7F7F_7F7F;
const HIGH_BITS: u64 = 0x8080_8080_8080_8080;

#[derive(Debug, Clone, Copy)]
pub(crate) struct Portable;

/// Leaf bitvectors of one block: byte `b` of lane `8 * w + j` in byte `j` of `bits[b][w]`.
type LeafWords = [[u64; WORDS]; MAX_LEAF_BYTES];

impl BlockKernel<LANES> for Portable {
    fn score_block(
        self,
        forest: &Forest,
        plans: &[Plan],
        codes: &[[u8; LANES]],
        rows: &[f64],
        scores: &mut [f64; LANES],
    ) {
        let mut bits = [[0u64; WORDS]; MAX_LEAF_BYTES];
        score_plans(forest, plans, rows, scores, |tree, positions| {
            leaf_positions(forest, tree, codes, &mut bits, positions);
        });
    }
}

/// The lane codes of one slot as words.
#[inline(always)]
fn words(codes: &[u8; LANES]) -> [u64; WORDS] {
    std::array::from_fn(|w| u64::from_le_bytes(codes[8 * w..8 * w + 8].try_into().expect("8 bytes")))
}

/// 0xFF in each byte where `a` is above `b` (unsigned), else 0.
#[inline(always)]
fn greater(a: u64, b: u64) -> u64 {
    // Per byte, `(b | 0x80) - (a & 0x7F)` cannot borrow from the next byte, and its top bit is `b & 0x7F >= a & 0x7F`.
    let low_at_least = (b | HIGH_BITS).wrapping_sub(a & LOW_BITS);
    let at_least = (b & !a) | (!(a ^ b) & low_at_least);
    ((!at_least & HIGH_BITS) >> 7) * 0xFF
}

/// Transposes the 8×8 byte matrix `rows` (byte `j` of `rows[i]` becomes byte `i` of `rows[j]`).
#[inline(always)]
fn transpose(rows: &mut [u64; 8]) {
    for (shift, mask) in [
        (8, 0x00FF_00FF_00FF_00FF_u64),
        (16, 0x0000_FFFF_0000_FFFF),
        (32, 0x0000_0000_FFFF_FFFF),
    ] {
        let step = shift / 8;
        for base in (0..8).filter(|i| i & step == 0) {
            let swap = ((rows[base] >> shift) ^ rows[base + step]) & mask;
            rows[base + step] ^= swap;
            rows[base] ^= swap << shift;
        }
    }
}

#[inline(always)]
fn leaf_positions(
    forest: &Forest,
    tree: &FastTree,
    codes: &[[u8; LANES]],
    bits: &mut LeafWords,
    positions: &mut [u8; LANES],
) {
    let (codes, slot_mask) = masked(codes);
    let used = tree.leaf_bytes as usize;
    bits[..used].fill([u64::MAX; WORDS]);
    for &op in &forest.categorical_ops[tree.categorical_start as usize..tree.categorical_end as usize] {
        let table = &forest.tables[op.table as usize];
        let keep = ONES * u64::from(op.keep);
        let classes = &codes[op.slot as usize & slot_mask];
        for (w, word) in bits[op.byte()].iter_mut().enumerate() {
            let left = classes[8 * w..8 * w + 8]
                .iter()
                .enumerate()
                .fold(0u64, |left, (j, &class)| {
                    left | u64::from(table.get(usize::from(class)).copied().unwrap_or(0)) << (8 * j)
                });
            *word &= left | keep;
        }
    }
    for &op in &forest.numeric_ops[tree.numeric_start as usize..tree.numeric_end as usize] {
        let threshold = ONES * u64::from(op.threshold());
        let keep = ONES * u64::from(op.keep());
        for (word, lane_codes) in bits[op.byte()].iter_mut().zip(words(&codes[op.slot() & slot_mask])) {
            *word &= !greater(lane_codes, threshold) | keep;
        }
    }
    let groups = used.div_ceil(8);
    for w in 0..WORDS {
        let mut found = [0u8; 8];
        for group in (0..groups).rev() {
            let mut lanes: [u64; 8] = std::array::from_fn(|i| {
                let byte = group * 8 + i;
                if byte < used {
                    bits[byte][w]
                } else {
                    0
                }
            });
            transpose(&mut lanes);
            for (position, word) in found.iter_mut().zip(lanes) {
                if word != 0 {
                    *position = (group * 64) as u8 + word.trailing_zeros() as u8;
                }
            }
        }
        positions[8 * w..8 * w + 8].copy_from_slice(&found);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn greater_compares_every_byte_pair() {
        for a in 0..=255u8 {
            for b in 0..=255u8 {
                let got = greater(ONES * u64::from(a), (ONES * u64::from(b)) ^ 0x00FF_0000_0000_0000);
                let expected = |b: u8| if a > b { 0xFFu64 } else { 0 };
                assert_eq!(got & 0xFF, expected(b), "{a} > {b}");
                assert_eq!((got >> 48) & 0xFF, expected(b ^ 0xFF), "{a} > {}", b ^ 0xFF);
            }
        }
    }

    #[test]
    fn transpose_swaps_rows_and_columns() {
        let mut rows: [u64; 8] =
            std::array::from_fn(|i| u64::from_le_bytes(std::array::from_fn(|j| (i * 8 + j) as u8)));
        transpose(&mut rows);
        for (j, row) in rows.iter().enumerate() {
            assert_eq!(
                row.to_le_bytes(),
                std::array::from_fn::<u8, 8, _>(|i| (i * 8 + j) as u8)
            );
        }
    }
}
