//! TCP ring collectives for distributed silvic training: [`TcpCollective`] implements [`silvic_core::Collective`]
//! (allreduce sum/max, allgather) in the role LightGBM 3.3.5's socket `Linkers` and `Network` play for its
//! data-parallel learner.
//!
//! # Topology
//! Every rank (a Ray actor) calls [`TcpCollective::bind`] on its node IP, or [`TcpCollective::bind_advertised`] to
//! listen on `0.0.0.0`, and hands [`Listener::address`] to the driver. The driver sends the list of addresses,
//! indexed by rank, back to every rank, and each calls [`TcpCollective::connect`] with the job's token. Rank `r`
//! connects to `(r + 1) % world`, retrying until the timeout, and accepts `(r - 1 + world) % world`, so each rank holds
//! two sockets: frames go out to the next rank and come in from the previous one. The handshake (magic and protocol
//! version, byte order, job token, rank, world size) rejects stray connections and ranks of another job. A world of
//! one opens no sockets.
//!
//! # Collectives
//! An allreduce of `n` elements cuts the buffer into `world` contiguous chunks, the first `n % world` one element
//! longer than the rest (chunks are empty when `n < world`). It runs as a ring reduce-scatter (`world - 1` steps, after
//! which rank `r` holds the reduced chunk `r + 1`) and a ring allgather (`world - 1` more), like LightGBM
//! `ReduceScatterRing` and `AllgatherRing`. Below 4096 bytes, or with fewer elements than ranks, it instead circulates
//! every rank's whole buffer once (`world - 1` steps; LightGBM `AllreduceByAllGather`), which halves the round trips
//! of the few-element allreduces each split runs. [`allgather_bytes`](silvic_core::Collective::allgather_bytes) is a
//! ring allgather of variable-length blocks whose lengths travel in the frame headers.
//!
//! Each step sends one frame to the next rank while receiving one from the previous rank, on non-blocking sockets
//! multiplexed with `poll(2)`, so no buffer size can deadlock the ring; a step retries for 50 µs before it sleeps in
//! `poll`, since waking a parked thread costs as much as a hop. Header and payload leave in one vectored send
//! and land with one vectored read in the caller's buffer, and `TCP_NODELAY` sends small frames at once. The frame
//! header (op, call sequence number, length) is checked on receipt, so ranks that call different collectives or pass
//! different lengths fail instead of exchanging garbage.
//!
//! # Order
//! Element `i` of chunk `c` reduces as the left fold `((x[c] + x[c + 1]) + x[c + 2]) + ... + x[c - 1]` over ranks
//! mod `world`, on both paths: the small-buffer path folds the gathered buffers in the ring's order. Results are
//! therefore bit-identical on every rank and across runs for a given world size and length; a float sum can differ in
//! its last bits between world sizes, while integer-valued sums (quantized histograms) and i64 sums are exact in any
//! order. `allreduce_max_f64` propagates NaN: a NaN on any rank makes the element NaN on all of them. i64 sums wrap on
//! overflow.
//!
//! # Failures
//! A socket error, a peer closing its connection, a frame from a diverged rank, or no progress for the timeout
//! becomes [`silvic_core::SilvicError::Collective`] naming the peer rank. The failing rank shuts both of its sockets
//! down, so its next rank reads end-of-stream and fails in turn: a rank that exits or fails takes the whole ring down
//! promptly, and a rank that hangs takes it down after one timeout. (A call whose frames all reached a rank before
//! the failure still completes there; that rank's next call fails.) Every later call on a failed collective returns
//! the same error. After an error the buffer's contents are unspecified.
//!
//! Unix only (Linux, macOS): readiness waits use `poll(2)`.

#[cfg(not(unix))]
compile_error!("silvic-net waits on sockets with poll(2) and builds on unix targets (Linux, macOS) only");

mod link;
mod reduce;
mod rendezvous;
mod ring;
mod sys;

pub use rendezvous::Listener;
pub use ring::TcpCollective;
