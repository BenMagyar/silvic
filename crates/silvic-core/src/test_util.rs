//! In-process collectives for tests: `world` ranks on `world` threads exchanging through shared memory. Reductions
//! run in rank order on every rank, like a deterministic MPI.

use std::sync::{Arc, Barrier, Mutex};

use crate::collective::Collective;
use crate::error::Result;

struct Shared {
    barrier: Barrier,
    slots: Mutex<Vec<Vec<u8>>>,
}

pub struct ThreadCollective {
    rank: usize,
    world: usize,
    shared: Arc<Shared>,
}

impl ThreadCollective {
    /// One collective per rank; move each onto its own thread.
    pub fn group(world: usize) -> Vec<ThreadCollective> {
        let shared = Arc::new(Shared {
            barrier: Barrier::new(world),
            slots: Mutex::new(vec![Vec::new(); world]),
        });
        (0..world)
            .map(|rank| ThreadCollective {
                rank,
                world,
                shared: Arc::clone(&shared),
            })
            .collect()
    }

    fn exchange(&mut self, local: &[u8]) -> Vec<Vec<u8>> {
        self.shared.slots.lock().expect("slots")[self.rank] = local.to_vec();
        self.shared.barrier.wait();
        let all = self.shared.slots.lock().expect("slots").clone();
        // Nobody may overwrite a slot before every rank has read it.
        self.shared.barrier.wait();
        all
    }

    fn reduce<T: Copy, const N: usize>(
        &mut self,
        data: &mut [T],
        to_bytes: fn(T) -> [u8; N],
        from_bytes: fn([u8; N]) -> T,
        combine: fn(T, T) -> T,
    ) {
        let local: Vec<u8> = data.iter().flat_map(|&v| to_bytes(v)).collect();
        let all = self.exchange(&local);
        for (i, value) in data.iter_mut().enumerate() {
            let read = |bytes: &Vec<u8>| from_bytes(bytes[i * N..(i + 1) * N].try_into().expect("element"));
            *value = all
                .iter()
                .skip(1)
                .fold(read(&all[0]), |acc, bytes| combine(acc, read(bytes)));
        }
    }
}

impl Collective for ThreadCollective {
    fn rank(&self) -> usize {
        self.rank
    }

    fn world_size(&self) -> usize {
        self.world
    }

    fn allreduce_sum_f64(&mut self, data: &mut [f64]) -> Result<()> {
        self.reduce(data, f64::to_le_bytes, f64::from_le_bytes, |a, b| a + b);
        Ok(())
    }

    fn allreduce_sum_i64(&mut self, data: &mut [i64]) -> Result<()> {
        self.reduce(data, i64::to_le_bytes, i64::from_le_bytes, |a, b| a + b);
        Ok(())
    }

    fn allreduce_max_f64(&mut self, data: &mut [f64]) -> Result<()> {
        self.reduce(data, f64::to_le_bytes, f64::from_le_bytes, f64::max);
        Ok(())
    }

    fn allgather_bytes(&mut self, local: &[u8]) -> Result<Vec<Vec<u8>>> {
        Ok(self.exchange(local))
    }
}

/// Runs `f(rank, collective)` on `world` threads and returns the results in rank order.
pub fn run_ranks<T: Send>(world: usize, f: impl Fn(usize, &mut ThreadCollective) -> T + Sync) -> Vec<T> {
    let collectives = ThreadCollective::group(world);
    std::thread::scope(|scope| {
        let handles: Vec<_> = collectives
            .into_iter()
            .enumerate()
            .map(|(rank, mut collective)| {
                let f = &f;
                scope.spawn(move || f(rank, &mut collective))
            })
            .collect();
        handles.into_iter().map(|h| h.join().expect("rank thread")).collect()
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ranks_agree_on_sums_and_gathers() {
        let results = run_ranks(3, |rank, collective| {
            let mut sums = vec![rank as f64, 1.0];
            collective.allreduce_sum_f64(&mut sums).expect("sum");
            let mut max = vec![rank as f64];
            collective.allreduce_max_f64(&mut max).expect("max");
            let gathered = collective.allgather_bytes(&[rank as u8; 2]).expect("gather");
            (sums, max, gathered)
        });
        for (sums, max, gathered) in results {
            assert_eq!(sums, vec![3.0, 3.0]);
            assert_eq!(max, vec![2.0]);
            assert_eq!(gathered, vec![vec![0, 0], vec![1, 1], vec![2, 2]]);
        }
    }
}
