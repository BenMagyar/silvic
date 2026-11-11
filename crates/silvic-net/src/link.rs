//! The full-duplex transport under every ring step, the role of LightGBM 3.3.5 `Linkers::SendRecv`: one frame goes to
//! the next rank while one arrives from the previous rank, on non-blocking sockets multiplexed with `poll(2)`.
//!
//! Sending and receiving at once is what keeps a step from deadlocking on large buffers: with blocking writes every
//! rank can sit in `write` waiting for its successor to read, while the successor sits in its own `write`. A frame is
//! a 16-byte little-endian header (op, collective sequence number, payload length) and the payload. Header and
//! payload leave in one vectored send, and an expected frame lands with one vectored read straight in the caller's
//! buffer, so a small frame costs one syscall each way and nothing is copied. The receiver checks the header against
//! its own call, so ranks that diverge (another collective, another length) fail instead of misreading the stream.

use std::fmt;
use std::io::{self, IoSlice, IoSliceMut, Read};
use std::net::{Shutdown, TcpStream};
use std::os::fd::AsRawFd;
use std::thread;
use std::time::{Duration, Instant};

use silvic_core::{Result, SilvicError};

use crate::sys::{self, Interest};

/// Frame header bytes: op (u32), sequence number (u32), payload length (u64), little-endian.
const HEADER_LEN: usize = 16;
/// How long a step retries non-blocking I/O, yielding the CPU between tries, before it sleeps in `poll`. Waking a
/// thread parked in `poll` costs tens of microseconds, as much as a network hop, and the per-split allreduces are
/// latency-bound; yielding rather than spinning lets ranks that share a machine run.
const SPIN_BEFORE_POLL: Duration = Duration::from_micros(50);

pub(crate) const OP_SUM_F64: u32 = 1;
pub(crate) const OP_SUM_I64: u32 = 2;
pub(crate) const OP_MAX_F64: u32 = 3;
pub(crate) const OP_ALLGATHER: u32 = 4;

fn op_name(op: u32) -> &'static str {
    match op {
        OP_SUM_F64 => "allreduce_sum_f64",
        OP_SUM_I64 => "allreduce_sum_i64",
        OP_MAX_F64 => "allreduce_max_f64",
        OP_ALLGATHER => "allgather_bytes",
        _ => "an unknown op",
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Header {
    op: u32,
    seq: u32,
    len: u64,
}

impl Header {
    fn encode(self) -> [u8; HEADER_LEN] {
        let mut bytes = [0; HEADER_LEN];
        bytes[..4].copy_from_slice(&self.op.to_le_bytes());
        bytes[4..8].copy_from_slice(&self.seq.to_le_bytes());
        bytes[8..].copy_from_slice(&self.len.to_le_bytes());
        bytes
    }

    fn decode(bytes: &[u8; HEADER_LEN]) -> Self {
        let [o0, o1, o2, o3, s0, s1, s2, s3, l0, l1, l2, l3, l4, l5, l6, l7] = *bytes;
        Self {
            op: u32::from_le_bytes([o0, o1, o2, o3]),
            seq: u32::from_le_bytes([s0, s1, s2, s3]),
            len: u64::from_le_bytes([l0, l1, l2, l3, l4, l5, l6, l7]),
        }
    }
}

impl fmt::Display for Header {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} #{} with {} bytes", op_name(self.op), self.seq, self.len)
    }
}

/// Where the previous rank's payload goes.
pub(crate) enum Incoming<'a> {
    /// A payload of exactly this buffer's length; any other length means the ranks diverged.
    Exact(&'a mut [u8]),
    /// A payload of any length; the vector is resized to it.
    Any(&'a mut Vec<u8>),
}

/// Why receiving failed, before it is named after the peer.
enum Fault {
    Io(io::Error),
    Closed,
    OutOfStep(String),
}

/// The frame being sent; `sent` counts header and payload bytes together.
struct Outgoing<'a> {
    header: [u8; HEADER_LEN],
    payload: &'a [u8],
    sent: usize,
}

impl<'a> Outgoing<'a> {
    fn new(header: Header, payload: &'a [u8]) -> Self {
        Self {
            header: header.encode(),
            payload,
            sent: 0,
        }
    }

    fn done(&self) -> bool {
        self.sent == HEADER_LEN + self.payload.len()
    }

    /// Sends what the socket takes now; `false` when it takes nothing.
    fn pump(&mut self, stream: &TcpStream) -> io::Result<bool> {
        let slices = [
            IoSlice::new(&self.header[self.sent.min(HEADER_LEN)..]),
            IoSlice::new(&self.payload[self.sent.saturating_sub(HEADER_LEN)..]),
        ];
        match sys::send_vectored(stream, &slices) {
            Ok(0) => Err(io::Error::new(io::ErrorKind::WriteZero, "the socket took no bytes")),
            Ok(sent) => {
                self.sent += sent;
                Ok(true)
            }
            Err(error) if error.kind() == io::ErrorKind::WouldBlock => Ok(false),
            Err(error) if error.kind() == io::ErrorKind::Interrupted => Ok(true),
            Err(error) => Err(error),
        }
    }
}

/// The frame being received; `got` counts header and payload bytes together.
struct Receiver<'a> {
    op: u32,
    seq: u32,
    header: [u8; HEADER_LEN],
    got: usize,
    payload: Incoming<'a>,
    /// Known up front for [`Incoming::Exact`], from the header for [`Incoming::Any`].
    payload_len: Option<usize>,
}

impl<'a> Receiver<'a> {
    fn new(op: u32, seq: u32, payload: Incoming<'a>) -> Self {
        let payload_len = match &payload {
            Incoming::Exact(buf) => Some(buf.len()),
            Incoming::Any(_) => None,
        };
        Self {
            op,
            seq,
            header: [0; HEADER_LEN],
            got: 0,
            payload,
            payload_len,
        }
    }

    fn done(&self) -> bool {
        self.payload_len.is_some_and(|len| self.got == HEADER_LEN + len)
    }

    /// Reads what the socket holds now; `false` when it holds nothing.
    fn pump(&mut self, stream: &TcpStream) -> Result<bool, Fault> {
        let header_pending = self.got < HEADER_LEN;
        let read = if header_pending {
            self.read_header(stream)
        } else {
            self.read_payload(stream)
        };
        match read {
            Ok(0) => Err(Fault::Closed),
            Ok(count) => {
                self.got += count;
                if header_pending && self.got >= HEADER_LEN {
                    self.check_header()?;
                }
                Ok(true)
            }
            Err(error) if error.kind() == io::ErrorKind::WouldBlock => Ok(false),
            Err(error) if error.kind() == io::ErrorKind::Interrupted => Ok(true),
            Err(error) => Err(Fault::Io(error)),
        }
    }

    fn read_header(&mut self, mut stream: &TcpStream) -> io::Result<usize> {
        let header = &mut self.header[self.got..];
        match &mut self.payload {
            // The payload length is known, so reading on past the header cannot run into the next frame.
            Incoming::Exact(buf) => stream.read_vectored(&mut [IoSliceMut::new(header), IoSliceMut::new(buf)]),
            Incoming::Any(_) => stream.read(header),
        }
    }

    fn read_payload(&mut self, mut stream: &TcpStream) -> io::Result<usize> {
        let start = self.got - HEADER_LEN;
        let buf = match &mut self.payload {
            Incoming::Exact(buf) => &mut buf[start..],
            Incoming::Any(vec) => &mut vec[start..],
        };
        stream.read(buf)
    }

    fn check_header(&mut self) -> Result<(), Fault> {
        let got = Header::decode(&self.header);
        let expected_len = self.payload_len.map(|len| len as u64);
        if got.op != self.op || got.seq != self.seq || expected_len.is_some_and(|len| len != got.len) {
            let expected = Header {
                op: self.op,
                seq: self.seq,
                len: expected_len.unwrap_or(got.len),
            };
            return Err(Fault::OutOfStep(format!(
                "it sent {got}, this rank expected {expected}"
            )));
        }
        if let Incoming::Any(vec) = &mut self.payload {
            let len = usize::try_from(got.len)
                .map_err(|_| Fault::OutOfStep(format!("it sent {got}, more than this rank can address")))?;
            vec.clear();
            vec.resize(len, 0);
            self.payload_len = Some(len);
        }
        Ok(())
    }
}

/// The two ring connections of one rank: frames go out to `next` and come in from `prev`.
#[derive(Debug)]
pub(crate) struct Link {
    next: TcpStream,
    prev: TcpStream,
    rank: usize,
    next_rank: usize,
    prev_rank: usize,
    /// Longest a step may go without progress in either direction.
    timeout: Duration,
}

impl Link {
    /// `next` and `prev` must already be non-blocking.
    pub(crate) fn new(next: TcpStream, prev: TcpStream, rank: usize, world: usize, timeout: Duration) -> Self {
        Self {
            next,
            prev,
            rank,
            next_rank: (rank + 1) % world,
            prev_rank: (rank + world - 1) % world,
            timeout,
        }
    }

    pub(crate) fn set_timeout(&mut self, timeout: Duration) {
        self.timeout = timeout;
    }

    /// Sends `outgoing` to the next rank as one frame while receiving the previous rank's frame of the same
    /// collective into `incoming`.
    pub(crate) fn exchange(&mut self, op: u32, seq: u32, outgoing: &[u8], incoming: Incoming<'_>) -> Result<()> {
        let mut tx = Outgoing::new(
            Header {
                op,
                seq,
                len: outgoing.len() as u64,
            },
            outgoing,
        );
        let mut rx = Receiver::new(op, seq, incoming);
        let mut deadline = Instant::now() + self.timeout;
        let mut idle_since = None;
        loop {
            let mut progressed = false;
            if !tx.done() {
                progressed |= tx.pump(&self.next).map_err(|error| self.send_failed(&error))?;
            }
            if !rx.done() {
                progressed |= rx.pump(&self.prev).map_err(|fault| self.receive_failed(fault))?;
            }
            if tx.done() && rx.done() {
                return Ok(());
            }
            let now = Instant::now();
            if progressed {
                deadline = now + self.timeout;
                idle_since = None;
            } else if now >= deadline {
                return Err(self.timed_out(rx.done()));
            } else if now - *idle_since.get_or_insert(now) < SPIN_BEFORE_POLL {
                thread::yield_now();
            } else {
                self.wait(tx.done(), rx.done(), deadline - now)?;
            }
        }
    }

    fn wait(&self, sent: bool, received: bool, timeout: Duration) -> Result<()> {
        let interests = [
            Interest {
                fd: (!sent).then(|| self.next.as_raw_fd()),
                events: sys::WRITABLE,
            },
            Interest {
                fd: (!received).then(|| self.prev.as_raw_fd()),
                events: sys::READABLE,
            },
        ];
        sys::wait(interests, timeout).map_err(|error| {
            self.error(format_args!(
                "waiting on ranks {} and {} failed: {error}",
                self.next_rank, self.prev_rank
            ))
        })
    }

    /// Closes both directions, so the next rank reads end-of-stream and fails in turn; the failure travels the
    /// whole ring instead of every rank waiting out its timeout.
    pub(crate) fn shutdown(&self) {
        // Either socket may already be reset by the peer; there is nothing left to report then.
        let _ = self.next.shutdown(Shutdown::Both);
        let _ = self.prev.shutdown(Shutdown::Both);
    }

    fn error(&self, detail: impl fmt::Display) -> SilvicError {
        SilvicError::collective(format!("rank {}: {detail}", self.rank))
    }

    fn send_failed(&self, error: &io::Error) -> SilvicError {
        self.error(format_args!("sending to rank {} failed: {error}", self.next_rank))
    }

    fn receive_failed(&self, fault: Fault) -> SilvicError {
        let peer = self.prev_rank;
        match fault {
            Fault::Io(error) => self.error(format_args!("receiving from rank {peer} failed: {error}")),
            Fault::Closed => self.error(format_args!("rank {peer} closed the connection (it failed or exited)")),
            Fault::OutOfStep(detail) => self.error(format_args!("rank {peer} is out of step: {detail}")),
        }
    }

    fn timed_out(&self, received: bool) -> SilvicError {
        if received {
            self.error(format_args!(
                "timed out after {:?} waiting for rank {} to take data",
                self.timeout, self.next_rank
            ))
        } else {
            self.error(format_args!(
                "timed out after {:?} waiting for rank {}",
                self.timeout, self.prev_rank
            ))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn header_round_trips() {
        let header = Header {
            op: OP_ALLGATHER,
            seq: 0xdead_beef,
            len: 1 << 40,
        };
        assert_eq!(Header::decode(&header.encode()), header, "allgather header");
    }
}
