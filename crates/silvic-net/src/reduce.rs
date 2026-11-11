//! The elementwise reductions, how a buffer is cut into one chunk per rank, and the fold order both allreduce paths
//! share.
//!
//! Chunk `c` of an `n`-element buffer reduces in the order ring reduce-scatter accumulates it: rank `c`'s values,
//! combined with rank `c + 1`'s, then `c + 2`'s, through rank `c - 1` (ranks mod world), as a left fold. The
//! latency path of LightGBM 3.3.5 `Network::AllreduceByAllGather` gathers every buffer and folds it locally;
//! [`fold_slots`] folds in the ring's order instead, so a buffer's result does not depend on which path its size
//! selects.

use std::ops::Range;

use bytemuck::Pod;

use crate::link::{OP_MAX_F64, OP_SUM_F64, OP_SUM_I64};

/// An elementwise reduction over 8-byte elements.
pub(crate) trait Reduce {
    type Elem: Pod;
    /// Wire op code: frames carry it, so ranks in different collectives fail instead of mixing buffers.
    const OP: u32;
    /// `acc` holds the ranks before `x` in the chunk's fold order. The operand order is part of the contract: it keeps
    /// non-associative results (float sums, NaN payloads, signed zeros) identical on every rank and run.
    fn combine(acc: Self::Elem, x: Self::Elem) -> Self::Elem;
}

pub(crate) struct SumF64;

pub(crate) struct SumI64;

pub(crate) struct MaxF64;

impl Reduce for SumF64 {
    type Elem = f64;
    const OP: u32 = OP_SUM_F64;

    #[inline]
    fn combine(acc: f64, x: f64) -> f64 {
        acc + x
    }
}

impl Reduce for SumI64 {
    type Elem = i64;
    const OP: u32 = OP_SUM_I64;

    /// Wraps on overflow as release builds do, so debug and release ranks agree and no rank panics mid-step.
    #[inline]
    fn combine(acc: i64, x: i64) -> i64 {
        acc.wrapping_add(x)
    }
}

impl Reduce for MaxF64 {
    type Elem = f64;
    const OP: u32 = OP_MAX_F64;

    /// NaN-propagating: a NaN on any rank makes the element NaN on every rank, so bad inputs are not silently
    /// dropped by whichever rank happens to fold last.
    #[inline]
    fn combine(acc: f64, x: f64) -> f64 {
        if x > acc || x.is_nan() {
            x
        } else {
            acc
        }
    }
}

/// LightGBM 3.3.5 `Network::Allreduce` threshold: below 4096 bytes, or with fewer elements than ranks, an allgather
/// of the whole buffer (`world - 1` steps) beats reduce-scatter plus allgather (`2 * (world - 1)` steps).
pub(crate) const GATHER_BELOW_BYTES: usize = 4096;

/// Whether an allreduce of `len` elements of `T` takes the allgather path; a function of the length alone, so every
/// rank of a well-formed call picks the same path.
pub(crate) fn gathers<T>(len: usize, world: usize) -> bool {
    len < world || len * std::mem::size_of::<T>() < GATHER_BELOW_BYTES
}

/// Chunk `chunk` of a `len`-element buffer split over `world` ranks: the first `len % world` chunks hold one element
/// more than the rest, so chunks are contiguous, cover the buffer, and are empty only when `len < world`.
pub(crate) fn chunk_range(len: usize, world: usize, chunk: usize) -> Range<usize> {
    let base = len / world;
    let extra = len % world;
    let start = chunk * base + chunk.min(extra);
    start..start + base + usize::from(chunk < extra)
}

/// Reduces `world` gathered buffers (rank `r`'s at `slots[r * n..(r + 1) * n]`) into `out` (`n` elements), folding
/// each chunk in ring reduce-scatter order.
pub(crate) fn fold_slots<R: Reduce>(slots: &[R::Elem], out: &mut [R::Elem], world: usize) {
    let len = out.len();
    debug_assert_eq!(slots.len(), len * world);
    for chunk in 0..world {
        let range = chunk_range(len, world, chunk);
        let out = &mut out[range.clone()];
        out.copy_from_slice(&slots[chunk * len..][range.clone()]);
        for offset in 1..world {
            let rank = (chunk + offset) % world;
            for (acc, &x) in out.iter_mut().zip(&slots[rank * len..][range.clone()]) {
                *acc = R::combine(*acc, x);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chunks_partition_every_length() {
        for world in 1..=7 {
            for len in 0..=30 {
                let mut next = 0;
                for chunk in 0..world {
                    let range = chunk_range(len, world, chunk);
                    assert_eq!(range.start, next, "world {world} len {len} chunk {chunk}: contiguous");
                    assert!(
                        range.len() >= len / world,
                        "world {world} len {len} chunk {chunk}: balanced"
                    );
                    next = range.end;
                }
                assert_eq!(next, len, "world {world} len {len}: covers the buffer");
            }
        }
    }

    #[test]
    fn fold_follows_ring_order() {
        // Chunk 1 of 3 folds ranks 1, 2, 0: (1e16 + -1e16) + 1 keeps the 1 that rank order (1 + 1e16) + -1e16 loses.
        let world = 3;
        let slots = [0.0, 1.0, 0.0, 0.0, 1e16, 0.0, 0.0, -1e16, 0.0];
        let mut out = [0.0; 3];
        fold_slots::<SumF64>(&slots, &mut out, world);
        assert_eq!(out[1], 1.0, "chunk 1 folds ranks 1, 2, 0");
    }

    #[test]
    fn max_propagates_nan() {
        assert!(MaxF64::combine(f64::NAN, 1.0).is_nan(), "NaN accumulator");
        assert!(MaxF64::combine(1.0, f64::NAN).is_nan(), "NaN operand");
        assert_eq!(MaxF64::combine(-1.0, 2.0), 2.0, "larger operand");
    }
}
