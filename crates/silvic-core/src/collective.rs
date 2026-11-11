//! The collectives distributed training runs on, in the shape of LightGBM's `Network` (allreduce, allgather) but
//! over a transport the caller supplies.
//!
//! Every rank calls the same collectives in the same order with buffers of the same length; a rank that diverges
//! deadlocks or fails the job, as in MPI. In quantized training every histogram value is an integer below 2^53, so
//! `allreduce_sum_f64` is exact in any reduction order and a model trained on N ranks equals one trained on one.

use crate::error::Result;

pub trait Collective: Send {
    fn rank(&self) -> usize;

    fn world_size(&self) -> usize;

    /// Elementwise sum across ranks; every rank ends with the sum.
    fn allreduce_sum_f64(&mut self, data: &mut [f64]) -> Result<()>;

    fn allreduce_sum_i64(&mut self, data: &mut [i64]) -> Result<()>;

    /// Elementwise maximum across ranks.
    fn allreduce_max_f64(&mut self, data: &mut [f64]) -> Result<()>;

    /// Every rank's bytes, indexed by rank, on every rank.
    fn allgather_bytes(&mut self, local: &[u8]) -> Result<Vec<Vec<u8>>>;
}

/// A world of one rank: every collective returns its input.
#[derive(Debug, Default, Clone, Copy)]
pub struct LocalCollective;

impl Collective for LocalCollective {
    fn rank(&self) -> usize {
        0
    }

    fn world_size(&self) -> usize {
        1
    }

    fn allreduce_sum_f64(&mut self, _data: &mut [f64]) -> Result<()> {
        Ok(())
    }

    fn allreduce_sum_i64(&mut self, _data: &mut [i64]) -> Result<()> {
        Ok(())
    }

    fn allreduce_max_f64(&mut self, _data: &mut [f64]) -> Result<()> {
        Ok(())
    }

    fn allgather_bytes(&mut self, local: &[u8]) -> Result<Vec<Vec<u8>>> {
        Ok(vec![local.to_vec()])
    }
}
