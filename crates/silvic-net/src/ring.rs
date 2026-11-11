//! [`TcpCollective`]: ring allreduce and allgather over a rank's two neighbour connections, in the shape of LightGBM
//! 3.3.5 `Network::Allreduce` (`ReduceScatterRing` then `AllgatherRing`, or `AllreduceByAllGather` for small buffers)
//! and `Network::Allgather` (`AllgatherRing`).

use std::ops::Range;
use std::time::Duration;

use bytemuck::Pod;
use silvic_core::{Collective, Result, SilvicError};

use crate::link::{Incoming, Link, OP_ALLGATHER};
use crate::reduce::{chunk_range, fold_slots, gathers, MaxF64, Reduce, SumF64, SumI64};
use crate::rendezvous::{form_ring, Listener};

/// One rank of a TCP ring. The crate docs describe the topology, the reduction order and the failure semantics.
#[derive(Debug)]
pub struct TcpCollective {
    rank: usize,
    world: usize,
    /// `None` in a world of one.
    ring: Option<Ring>,
    /// Collectives called so far, wrapping; frames carry it so ranks that diverge fail.
    seq: u32,
    timeout: Duration,
    /// The first failure; every later call returns it.
    broken: Option<SilvicError>,
}

impl TcpCollective {
    /// Binds an ephemeral port on `host` (an IP or hostname the other ranks can reach, such as Ray's node IP) and
    /// advertises `host:port`. An unspecified host (`0.0.0.0`, `::`) is not routable and fails with a config error;
    /// listen on it with [`TcpCollective::bind_advertised`].
    pub fn bind(host: &str) -> Result<Listener> {
        Listener::bind(host, host)
    }

    /// Binds an ephemeral port on `host` (for instance `0.0.0.0`, every interface) and advertises `node_ip:port`.
    pub fn bind_advertised(host: &str, node_ip: &str) -> Result<Listener> {
        Listener::bind(host, node_ip)
    }

    /// Joins the ring as `rank`, where `addresses` holds every rank's [`Listener::address`] indexed by rank and
    /// `job_token` is shared by the ranks of one job only. Connects to rank `(rank + 1) % world`, retrying until
    /// `timeout`, and accepts rank `(rank - 1 + world) % world`. `timeout` bounds the rendezvous and then becomes
    /// the longest any collective may go without progress ([`TcpCollective::set_timeout`] changes it). A world of one
    /// opens no sockets and drops the listener.
    pub fn connect(
        listener: Listener,
        rank: usize,
        addresses: &[String],
        job_token: u64,
        timeout: Duration,
    ) -> Result<Self> {
        let world = addresses.len();
        let ring = form_ring(listener, rank, addresses, job_token, timeout)?.map(|(next, prev)| Ring {
            link: Link::new(next, prev, rank, world, timeout),
            rank,
            world,
            scratch: Vec::new(),
        });
        Ok(Self {
            rank,
            world,
            ring,
            seq: 0,
            timeout,
            broken: None,
        })
    }

    /// Longest a collective may go without progress before it fails.
    pub fn timeout(&self) -> Duration {
        self.timeout
    }

    /// Sets the inactivity timeout of later collectives; it must exceed the longest compute skew between ranks.
    pub fn set_timeout(&mut self, timeout: Duration) {
        self.timeout = timeout;
        if let Some(ring) = &mut self.ring {
            ring.link.set_timeout(timeout);
        }
    }

    /// The sequence number of the call about to run, or the failure that broke the ring.
    fn begin(&mut self) -> Result<u32> {
        if let Some(error) = &self.broken {
            return Err(error.clone());
        }
        let seq = self.seq;
        self.seq = seq.wrapping_add(1);
        Ok(seq)
    }

    /// Records a failure: shuts the ring down so the neighbours fail too, and keeps the error for later calls.
    fn settle<T>(&mut self, result: Result<T>) -> Result<T> {
        if let Err(error) = &result {
            tracing::error!(rank = self.rank, %error, "collective failed; shutting the ring down");
            if let Some(ring) = &self.ring {
                ring.link.shutdown();
            }
            self.broken = Some(error.clone());
        }
        result
    }

    fn allreduce<R: Reduce>(&mut self, data: &mut [R::Elem]) -> Result<()> {
        let seq = self.begin()?;
        match self.ring.as_mut() {
            // Every rank passes the same length, so an empty buffer is skipped on every rank alike.
            Some(ring) if !data.is_empty() => {
                let result = ring.allreduce::<R>(data, seq);
                self.settle(result)
            }
            _ => Ok(()),
        }
    }
}

impl Collective for TcpCollective {
    fn rank(&self) -> usize {
        self.rank
    }

    fn world_size(&self) -> usize {
        self.world
    }

    fn allreduce_sum_f64(&mut self, data: &mut [f64]) -> Result<()> {
        self.allreduce::<SumF64>(data)
    }

    fn allreduce_sum_i64(&mut self, data: &mut [i64]) -> Result<()> {
        self.allreduce::<SumI64>(data)
    }

    fn allreduce_max_f64(&mut self, data: &mut [f64]) -> Result<()> {
        self.allreduce::<MaxF64>(data)
    }

    fn allgather_bytes(&mut self, local: &[u8]) -> Result<Vec<Vec<u8>>> {
        let seq = self.begin()?;
        match self.ring.as_mut() {
            Some(ring) => {
                let result = ring.allgather_bytes(local, seq);
                self.settle(result)
            }
            None => Ok(vec![local.to_vec()]),
        }
    }
}

/// The connected state of a world of two or more.
#[derive(Debug)]
struct Ring {
    link: Link,
    rank: usize,
    world: usize,
    /// Receive buffer of 8-byte elements reused across calls, so the per-split allreduces do not allocate.
    scratch: Vec<u64>,
}

impl Ring {
    fn allreduce<R: Reduce>(&mut self, data: &mut [R::Elem], seq: u32) -> Result<()> {
        if gathers::<R::Elem>(data.len(), self.world) {
            self.gather_and_fold::<R>(data, seq)
        } else {
            self.reduce_scatter::<R>(data, seq)?;
            self.allgather_chunks::<R>(data, seq)
        }
    }

    /// `AllreduceByAllGather`: every rank's whole buffer travels the ring once (`world - 1` steps instead of
    /// `2 * (world - 1)`), then each rank folds all of them in ring order.
    fn gather_and_fold<R: Reduce>(&mut self, data: &mut [R::Elem], seq: u32) -> Result<()> {
        let (rank, world, len) = (self.rank, self.world, data.len());
        let slots = typed::<R::Elem>(&mut self.scratch, len * world);
        slots[rank * len..][..len].copy_from_slice(data);
        for step in 0..world - 1 {
            let send = (rank + world - step) % world;
            let recv = (rank + world - step - 1) % world;
            let (outgoing, incoming) = split(slots, send * len..(send + 1) * len, recv * len..(recv + 1) * len);
            let incoming = Incoming::Exact(bytemuck::cast_slice_mut(incoming));
            self.link
                .exchange(R::OP, seq, bytemuck::cast_slice(outgoing), incoming)?;
        }
        fold_slots::<R>(slots, data, world);
        Ok(())
    }

    /// `ReduceScatterRing`: at step `s` rank `r` sends chunk `r - s` and folds the incoming partial chunk `r - s - 1`
    /// into its own values, so after `world - 1` steps it holds the full chunk `r + 1`.
    fn reduce_scatter<R: Reduce>(&mut self, data: &mut [R::Elem], seq: u32) -> Result<()> {
        let (rank, world, len) = (self.rank, self.world, data.len());
        // Chunk 0 is the longest.
        let buffer = typed::<R::Elem>(&mut self.scratch, chunk_range(len, world, 0).len());
        for step in 0..world - 1 {
            let send = chunk_range(len, world, (rank + world - step) % world);
            let recv = chunk_range(len, world, (rank + world - step - 1) % world);
            let partial = &mut buffer[..recv.len()];
            let incoming = Incoming::Exact(bytemuck::cast_slice_mut(partial));
            self.link
                .exchange(R::OP, seq, bytemuck::cast_slice(&data[send]), incoming)?;
            for (local, &acc) in data[recv].iter_mut().zip(partial.iter()) {
                *local = R::combine(acc, *local);
            }
        }
        Ok(())
    }

    /// `AllgatherRing` of the reduced chunks: at step `s` rank `r` forwards chunk `r + 1 - s` and receives chunk
    /// `r - s` in place.
    fn allgather_chunks<R: Reduce>(&mut self, data: &mut [R::Elem], seq: u32) -> Result<()> {
        let (rank, world, len) = (self.rank, self.world, data.len());
        for step in 0..world - 1 {
            let send = chunk_range(len, world, (rank + 1 + world - step) % world);
            let recv = chunk_range(len, world, (rank + world - step) % world);
            let (outgoing, incoming) = split(data, send, recv);
            let incoming = Incoming::Exact(bytemuck::cast_slice_mut(incoming));
            self.link
                .exchange(R::OP, seq, bytemuck::cast_slice(outgoing), incoming)?;
        }
        Ok(())
    }

    /// `AllgatherRing` of variable-length blocks: at step `s` rank `r` forwards rank `r - s`'s block; each frame's
    /// header carries the block length.
    fn allgather_bytes(&mut self, local: &[u8], seq: u32) -> Result<Vec<Vec<u8>>> {
        let (rank, world) = (self.rank, self.world);
        let mut blocks = vec![Vec::new(); world];
        blocks[rank] = local.to_vec();
        for step in 0..world - 1 {
            let send = (rank + world - step) % world;
            let recv = (rank + world - step - 1) % world;
            let (outgoing, incoming) = split(&mut blocks, send..send + 1, recv..recv + 1);
            self.link
                .exchange(OP_ALLGATHER, seq, &outgoing[0], Incoming::Any(&mut incoming[0]))?;
        }
        Ok(blocks)
    }
}

/// `scratch` grown to at least `len` elements, viewed as `len` elements of `T`.
fn typed<T: Pod>(scratch: &mut Vec<u64>, len: usize) -> &mut [T] {
    const { assert!(size_of::<T>() == size_of::<u64>() && align_of::<T>() <= align_of::<u64>()) };
    if scratch.len() < len {
        scratch.resize(len, 0);
    }
    bytemuck::cast_slice_mut(&mut scratch[..len])
}

/// The disjoint `read` and `write` ranges of `data`, borrowed at once.
fn split<T>(data: &mut [T], read: Range<usize>, write: Range<usize>) -> (&[T], &mut [T]) {
    if read.end <= write.start {
        let (head, tail) = data.split_at_mut(write.start);
        (&head[read], &mut tail[..write.len()])
    } else {
        debug_assert!(write.end <= read.start, "ranges {read:?} and {write:?} overlap");
        let (head, tail) = data.split_at_mut(read.start);
        (&tail[..read.len()], &mut head[write])
    }
}
