//! Rendezvous: each rank binds a [`Listener`], the driver collects every rank's address, and [`form_ring`] connects
//! each rank to the next and accepts the previous one. This is the job of LightGBM 3.3.5 `Linkers::Construct` for its
//! machine list, where every pair of machines connects; a ring needs only neighbours.
//!
//! Each connection opens with a handshake: the connector sends a hello (magic with the protocol version, byte-order
//! mark, rank, world size, job token) and the acceptor answers with a verdict. The acceptor drops connections whose
//! hello does not match (port scanners, ranks of another job, a misnumbered rank) and keeps waiting, so a stray cannot
//! take a rank's place; the connector fails at once when its peer rejects it, since retrying cannot fix a mismatch.
//! Payloads travel in native byte order, so the byte-order mark is written natively: a peer of the other endianness
//! reads it reversed and is rejected.

use std::fmt;
use std::io::{self, Read, Write};
use std::net::{IpAddr, SocketAddr, TcpListener, TcpStream, ToSocketAddrs};
use std::os::fd::AsRawFd;
use std::thread;
use std::time::{Duration, Instant};

use silvic_core::{Result, SilvicError};

use crate::sys::{self, Interest};

/// Opens every hello and verdict; the last byte is the protocol version.
const MAGIC: [u8; 8] = *b"silvicn\x01";
/// Written in native byte order; see the module docs.
const BYTE_ORDER_MARK: u32 = 0x0102_0304;
/// Hello: magic, byte-order mark, rank (u32), world size (u32), reserved (u32), job token (u64).
const HELLO_LEN: usize = 32;
/// Verdict: magic, status (u32), the acceptor's rank (u32).
const VERDICT_LEN: usize = 16;
/// Longest an accepted connection may take to send its hello. A rank sends it right after connecting, so a silent
/// connection is a stray, and it must not hold up the real previous rank queued behind it.
const HELLO_TIMEOUT: Duration = Duration::from_secs(5);
/// Connect retries back off from the first to the second delay, doubling.
const CONNECT_BACKOFF: (Duration, Duration) = (Duration::from_millis(10), Duration::from_millis(250));

/// A bound rank that has not joined its ring yet: the listening socket and the address other ranks reach it at.
#[derive(Debug)]
pub struct Listener {
    socket: TcpListener,
    address: String,
}

impl Listener {
    /// Binds an ephemeral port on `host` and advertises it as `advertised:port`.
    pub(crate) fn bind(host: &str, advertised: &str) -> Result<Self> {
        let socket = TcpListener::bind((host, 0))
            .map_err(|error| SilvicError::collective(format!("binding a port on {host} failed: {error}")))?;
        let port = socket
            .local_addr()
            .map_err(|error| SilvicError::collective(format!("reading the bound port failed: {error}")))?
            .port();
        let address = match advertised.parse::<IpAddr>() {
            Ok(ip) if ip.is_unspecified() => {
                return Err(SilvicError::config(format!(
                    "{advertised} is not an address other ranks can reach; listen on it with \
                     TcpCollective::bind_advertised({advertised:?}, node_ip)"
                )))
            }
            Ok(ip) => SocketAddr::new(ip, port).to_string(),
            Err(_) => format!("{advertised}:{port}"),
        };
        Ok(Self { socket, address })
    }

    /// `ip:port` the other ranks connect to; the driver lists these by rank for [`crate::TcpCollective::connect`].
    pub fn address(&self) -> String {
        self.address.clone()
    }
}

/// One rank's place in the ring, as its handshake states it.
#[derive(Debug, Clone, Copy)]
struct Member {
    rank: usize,
    world: usize,
    token: u64,
}

impl Member {
    fn next(&self) -> usize {
        (self.rank + 1) % self.world
    }

    fn prev(&self) -> usize {
        (self.rank + self.world - 1) % self.world
    }

    fn failure(&self, detail: impl fmt::Display) -> SilvicError {
        SilvicError::collective(format!("rank {}: {detail}", self.rank))
    }

    fn hello(&self) -> [u8; HELLO_LEN] {
        let mut bytes = [0; HELLO_LEN];
        bytes[..8].copy_from_slice(&MAGIC);
        bytes[8..12].copy_from_slice(&BYTE_ORDER_MARK.to_ne_bytes());
        bytes[12..16].copy_from_slice(&(self.rank as u32).to_le_bytes());
        bytes[16..20].copy_from_slice(&(self.world as u32).to_le_bytes());
        bytes[24..].copy_from_slice(&self.token.to_le_bytes());
        bytes
    }

    /// The acceptor's answer to `hello`; checks the token first so another job's rank is named as such.
    fn judge(&self, hello: &Hello) -> Verdict {
        if hello.token != self.token {
            Verdict::WrongToken
        } else if hello.world as usize != self.world {
            Verdict::WrongWorld
        } else if hello.rank as usize != self.prev() {
            Verdict::WrongRank
        } else {
            Verdict::Accepted
        }
    }
}

struct Hello {
    rank: u32,
    world: u32,
    token: u64,
}

impl Hello {
    fn decode(bytes: &[u8; HELLO_LEN]) -> Result<Self, String> {
        if bytes[..8] != MAGIC {
            return Err("not a silvic rank (wrong magic or protocol version)".into());
        }
        if u32_at(bytes, 8, u32::from_ne_bytes) != BYTE_ORDER_MARK {
            return Err("its byte order differs from this rank's".into());
        }
        let token = u64::from_le_bytes(bytes[24..].try_into().expect("8 token bytes"));
        Ok(Self {
            rank: u32_at(bytes, 12, u32::from_le_bytes),
            world: u32_at(bytes, 16, u32::from_le_bytes),
            token,
        })
    }
}

fn u32_at(bytes: &[u8], offset: usize, from_bytes: fn([u8; 4]) -> u32) -> u32 {
    from_bytes(bytes[offset..offset + 4].try_into().expect("4 bytes"))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Verdict {
    Accepted,
    WrongToken,
    WrongWorld,
    WrongRank,
}

impl Verdict {
    const ALL: [Verdict; 4] = [
        Verdict::Accepted,
        Verdict::WrongToken,
        Verdict::WrongWorld,
        Verdict::WrongRank,
    ];

    fn encode(self, rank: usize) -> [u8; VERDICT_LEN] {
        let mut bytes = [0; VERDICT_LEN];
        bytes[..8].copy_from_slice(&MAGIC);
        bytes[8..12].copy_from_slice(&(self as u32).to_le_bytes());
        bytes[12..].copy_from_slice(&(rank as u32).to_le_bytes());
        bytes
    }

    /// The verdict and the acceptor's rank, or `None` when the bytes are not a silvic verdict.
    fn decode(bytes: &[u8; VERDICT_LEN]) -> Option<(Self, u32)> {
        if bytes[..8] != MAGIC {
            return None;
        }
        let status = u32_at(bytes, 8, u32::from_le_bytes);
        let verdict = Self::ALL.into_iter().find(|verdict| *verdict as u32 == status)?;
        Some((verdict, u32_at(bytes, 12, u32::from_le_bytes)))
    }

    fn reason(self) -> &'static str {
        match self {
            Verdict::Accepted => "accepted",
            Verdict::WrongToken => "job token mismatch",
            Verdict::WrongWorld => "world size mismatch",
            Verdict::WrongRank => "rank mismatch",
        }
    }
}

/// Connects rank `rank` to its ring neighbours: the `(next, prev)` streams, non-blocking with `TCP_NODELAY`, or
/// `None` for a world of one, which needs no sockets.
pub(crate) fn form_ring(
    listener: Listener,
    rank: usize,
    addresses: &[String],
    token: u64,
    timeout: Duration,
) -> Result<Option<(TcpStream, TcpStream)>> {
    let world = addresses.len();
    if world == 0 || rank >= world {
        return Err(SilvicError::config(format!(
            "rank {rank} is out of range for {world} addresses"
        )));
    }
    if u32::try_from(world).is_err() {
        return Err(SilvicError::config(format!(
            "world size {world} does not fit the handshake's u32"
        )));
    }
    if world == 1 {
        return Ok(None);
    }
    let member = Member { rank, world, token };
    let deadline = Instant::now() + timeout;
    let next_address = addresses[member.next()].as_str();
    // The hello lands in the kernel buffer before the next rank accepts, so every rank can move on to accepting its
    // previous rank and no rank waits on a cycle of accepts.
    let mut next = connect_next(&member, next_address, deadline)?;
    let prev = accept_prev(&listener.socket, &member, deadline, timeout)?;
    await_verdict(&mut next, &member, next_address, deadline)?;
    for (stream, peer) in [(&next, member.next()), (&prev, member.prev())] {
        configure(stream)
            .map_err(|error| member.failure(format_args!("configuring the rank {peer} socket: {error}")))?;
    }
    tracing::debug!(rank, world, next = member.next(), prev = member.prev(), "ring formed");
    Ok(Some((next, prev)))
}

fn connect_next(member: &Member, address: &str, deadline: Instant) -> Result<TcpStream> {
    let next = member.next();
    let (mut backoff, max_backoff) = CONNECT_BACKOFF;
    let mut stream = loop {
        match try_connect(address, deadline) {
            Ok(stream) => break stream,
            Err(error) if Instant::now() + backoff < deadline => {
                tracing::debug!(rank = member.rank, next, address, %error, "retrying connect");
                thread::sleep(backoff);
                backoff = (backoff * 2).min(max_backoff);
            }
            Err(error) => {
                return Err(member.failure(format_args!(
                    "could not connect to rank {next} at {address} before the timeout: {error}"
                )))
            }
        }
    };
    let sent = stream
        .set_nodelay(true)
        .and_then(|()| stream.set_write_timeout(Some(remaining(deadline))))
        .and_then(|()| stream.write_all(&member.hello()));
    sent.map_err(|error| {
        member.failure(format_args!(
            "sending the handshake to rank {next} at {address}: {error}"
        ))
    })?;
    Ok(stream)
}

fn try_connect(address: &str, deadline: Instant) -> io::Result<TcpStream> {
    let mut last_error = io::Error::new(io::ErrorKind::NotFound, "the address resolves to nothing");
    for socket_address in address.to_socket_addrs()? {
        match TcpStream::connect_timeout(&socket_address, remaining(deadline)) {
            Ok(stream) => return Ok(stream),
            Err(error) => last_error = error,
        }
    }
    Err(last_error)
}

/// Connections the acceptor dropped, for the timeout message.
#[derive(Default)]
struct Rejections {
    count: usize,
    last: Option<(SocketAddr, String)>,
}

impl fmt::Display for Rejections {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.last {
            Some((peer, reason)) => write!(
                f,
                "; rejected {} connection(s), the last from {peer}: {reason}",
                self.count
            ),
            None => Ok(()),
        }
    }
}

fn accept_prev(socket: &TcpListener, member: &Member, deadline: Instant, timeout: Duration) -> Result<TcpStream> {
    let prev = member.prev();
    socket
        .set_nonblocking(true)
        .map_err(|error| member.failure(format_args!("configuring the listener: {error}")))?;
    let mut rejections = Rejections::default();
    loop {
        match socket.accept() {
            Ok((stream, peer)) => match greet(stream, member, deadline) {
                Ok(stream) => return Ok(stream),
                Err(reason) => {
                    tracing::warn!(rank = member.rank, %peer, %reason, "rejected a connection");
                    rejections.count += 1;
                    rejections.last = Some((peer, reason));
                }
            },
            Err(error) if error.kind() == io::ErrorKind::WouldBlock => {
                let now = Instant::now();
                if now >= deadline {
                    return Err(member.failure(format_args!(
                        "timed out after {timeout:?} waiting for rank {prev} to connect{rejections}"
                    )));
                }
                let readable = [Interest {
                    fd: Some(socket.as_raw_fd()),
                    events: sys::READABLE,
                }];
                sys::wait(readable, deadline - now).map_err(|error| {
                    member.failure(format_args!("waiting for rank {prev} to connect failed: {error}"))
                })?;
            }
            // A connection reset before it was accepted is a stray that gave up; keep waiting for the rank.
            Err(error)
                if matches!(
                    error.kind(),
                    io::ErrorKind::Interrupted | io::ErrorKind::ConnectionAborted | io::ErrorKind::ConnectionReset
                ) => {}
            Err(error) => return Err(member.failure(format_args!("accepting rank {prev} failed: {error}"))),
        }
    }
}

/// Reads an accepted connection's hello and answers it; the stream when it is the previous rank of this job.
fn greet(mut stream: TcpStream, member: &Member, deadline: Instant) -> Result<TcpStream, String> {
    let patience = remaining(deadline).min(HELLO_TIMEOUT);
    // Accepted sockets inherit O_NONBLOCK from the listener on BSD-derived systems (macOS), not on Linux.
    stream
        .set_nonblocking(false)
        .and_then(|()| sys::suppress_sigpipe(&stream))
        .and_then(|()| stream.set_read_timeout(Some(patience)))
        .and_then(|()| stream.set_write_timeout(Some(patience)))
        .map_err(|error| format!("configuring the socket failed: {error}"))?;
    let mut bytes = [0; HELLO_LEN];
    stream
        .read_exact(&mut bytes)
        .map_err(|error| format!("no handshake: {error}"))?;
    let hello = Hello::decode(&bytes)?;
    let verdict = member.judge(&hello);
    let answered = stream.write_all(&verdict.encode(member.rank));
    if verdict != Verdict::Accepted {
        // The answer to a rejected peer is best-effort: it is dropped either way.
        return Err(format!(
            "{} (it claims rank {} of {})",
            verdict.reason(),
            hello.rank,
            hello.world
        ));
    }
    answered.map_err(|error| format!("answering the handshake failed: {error}"))?;
    Ok(stream)
}

fn await_verdict(stream: &mut TcpStream, member: &Member, address: &str, deadline: Instant) -> Result<()> {
    let next = member.next();
    let mut bytes = [0; VERDICT_LEN];
    stream
        .set_read_timeout(Some(remaining(deadline)))
        .and_then(|()| stream.read_exact(&mut bytes))
        .map_err(|error| {
            member.failure(format_args!(
                "rank {next} at {address} did not answer the handshake: {error}"
            ))
        })?;
    match Verdict::decode(&bytes) {
        Some((Verdict::Accepted, rank)) if rank as usize == next => Ok(()),
        Some((Verdict::Accepted, rank)) => Err(member.failure(format_args!(
            "the listener at {address} is rank {rank}, not rank {next}"
        ))),
        Some((verdict, _)) => Err(member.failure(format_args!(
            "rank {next} at {address} rejected this rank: {}",
            verdict.reason()
        ))),
        None => Err(member.failure(format_args!("the listener at {address} is not a silvic rank"))),
    }
}

/// Switches a handshaken stream to the collective's mode: non-blocking for `poll`, no socket timeouts (the link
/// keeps its own), `TCP_NODELAY` so small frames leave at once.
fn configure(stream: &TcpStream) -> io::Result<()> {
    stream.set_read_timeout(None)?;
    stream.set_write_timeout(None)?;
    stream.set_nodelay(true)?;
    sys::suppress_sigpipe(stream)?;
    stream.set_nonblocking(true)
}

/// Time left before `deadline`, at least a millisecond: std rejects zero socket timeouts.
fn remaining(deadline: Instant) -> Duration {
    deadline
        .saturating_duration_since(Instant::now())
        .max(Duration::from_millis(1))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn handshake_round_trips() {
        let member = Member {
            rank: 3,
            world: 5,
            token: 0x0123_4567_89ab_cdef,
        };
        let hello = Hello::decode(&member.hello()).expect("hello");
        assert_eq!(
            (hello.rank, hello.world, hello.token),
            (3, 5, member.token),
            "hello fields"
        );
        let acceptor = Member { rank: 4, ..member };
        assert_eq!(acceptor.judge(&hello), Verdict::Accepted, "next rank accepts");
        assert_eq!(
            Member { token: 1, ..acceptor }.judge(&hello),
            Verdict::WrongToken,
            "other job"
        );
        assert_eq!(
            Member { rank: 0, ..member }.judge(&hello),
            Verdict::WrongRank,
            "not the previous rank"
        );
        for verdict in Verdict::ALL {
            assert_eq!(Verdict::decode(&verdict.encode(4)), Some((verdict, 4)), "{verdict:?}");
        }
    }

    #[test]
    fn unspecified_host_needs_an_advertised_ip() {
        let error = Listener::bind("0.0.0.0", "0.0.0.0").expect_err("0.0.0.0 is not routable");
        assert_eq!(error.kind(), "config", "error kind");
        let listener = Listener::bind("0.0.0.0", "10.1.2.3").expect("advertised node IP");
        assert!(
            listener.address().starts_with("10.1.2.3:"),
            "advertised address {}",
            listener.address()
        );
    }

    #[test]
    fn connect_retries_until_the_listener_appears() {
        let address = TcpListener::bind("127.0.0.1:0")
            .and_then(|reserved| reserved.local_addr())
            .expect("free port")
            .to_string();
        let member = Member {
            rank: 0,
            world: 2,
            token: 7,
        };
        let late = {
            let address = address.clone();
            thread::spawn(move || {
                thread::sleep(Duration::from_millis(300));
                let (mut stream, _) = TcpListener::bind(address)
                    .and_then(|listener| listener.accept())
                    .expect("late listener");
                let mut hello = [0; HELLO_LEN];
                stream.read_exact(&mut hello).expect("hello");
                Hello::decode(&hello).expect("valid hello").rank
            })
        };
        let deadline = Instant::now() + Duration::from_secs(10);
        connect_next(&member, &address, deadline).expect("connect after retries");
        assert_eq!(late.join().expect("listener thread"), 0, "hello from rank 0");
    }
}
