//! The socket calls std does not offer: `poll(2)` with a timeout, a vectored send that cannot raise `SIGPIPE`, and
//! `SO_NOSIGPIPE` on accepted sockets.
//!
//! A write to a socket the peer has reset raises `SIGPIPE`, which kills a host process that has not ignored it
//! (Python and Rust binaries ignore it; an embedding C++ program may not). std's `write` passes `MSG_NOSIGNAL` but
//! its `write_vectored` is a plain `writev`, so the vectored path goes through `sendmsg(MSG_NOSIGNAL)` on Linux. Apple
//! platforms have no `MSG_NOSIGNAL`; std sets `SO_NOSIGPIPE` on the sockets it creates but not on accepted ones, so
//! [`suppress_sigpipe`] sets it there.

use std::io::{self, IoSlice};
use std::net::TcpStream;
use std::os::fd::{AsRawFd, RawFd};
use std::time::Duration;

/// Readiness interest for one descriptor; `None` leaves the slot out of the wait.
pub(crate) struct Interest {
    pub fd: Option<RawFd>,
    pub events: libc::c_short,
}

pub(crate) const READABLE: libc::c_short = libc::POLLIN;
pub(crate) const WRITABLE: libc::c_short = libc::POLLOUT;

/// Waits until one of the descriptors is ready or `timeout` passes; an interrupted wait counts as a wakeup, since
/// callers re-check their deadline and retry their I/O either way.
pub(crate) fn wait<const N: usize>(interests: [Interest; N], timeout: Duration) -> io::Result<()> {
    // A negative descriptor is skipped by poll(2), so a finished direction cannot wake the wait with a stale
    // hang-up.
    let mut fds = interests.map(|interest| libc::pollfd {
        fd: interest.fd.unwrap_or(-1),
        events: interest.events,
        revents: 0,
    });
    // poll(2) takes whole milliseconds; rounding up keeps a sub-millisecond remainder from spinning.
    let millis = timeout.as_nanos().div_ceil(1_000_000).min(libc::c_int::MAX as u128) as libc::c_int;
    // SAFETY: `fds` is a live, exclusively borrowed array of `N` pollfd structs for the duration of the call.
    let ready = unsafe { libc::poll(fds.as_mut_ptr(), N as libc::nfds_t, millis) };
    if ready >= 0 {
        return Ok(());
    }
    let error = io::Error::last_os_error();
    if error.kind() == io::ErrorKind::Interrupted {
        Ok(())
    } else {
        Err(error)
    }
}

/// `writev` that reports a reset peer as `EPIPE` instead of raising `SIGPIPE`.
#[cfg(not(target_vendor = "apple"))]
pub(crate) fn send_vectored(stream: &TcpStream, slices: &[IoSlice<'_>]) -> io::Result<usize> {
    // SAFETY: msghdr is plain data; all-zero is a valid empty header (no name, no control data).
    let mut header: libc::msghdr = unsafe { std::mem::zeroed() };
    // IoSlice is ABI-compatible with iovec on unix (std guarantee), and sendmsg only reads the slices.
    header.msg_iov = slices.as_ptr() as *mut libc::iovec;
    header.msg_iovlen = slices.len() as _;
    // SAFETY: `header` points at `slices.len()` valid iovecs that outlive the call; the descriptor is owned by
    // `stream`.
    let sent = unsafe { libc::sendmsg(stream.as_raw_fd(), &header, libc::MSG_NOSIGNAL) };
    if sent < 0 {
        Err(io::Error::last_os_error())
    } else {
        Ok(sent as usize)
    }
}

/// `writev`; [`suppress_sigpipe`] has made a reset peer report `EPIPE`.
#[cfg(target_vendor = "apple")]
pub(crate) fn send_vectored(stream: &TcpStream, slices: &[IoSlice<'_>]) -> io::Result<usize> {
    use std::io::Write;
    let mut writer = stream;
    writer.write_vectored(slices)
}

/// Sets `SO_NOSIGPIPE`, so writes to a reset peer fail with `EPIPE` instead of raising `SIGPIPE`.
#[cfg(target_vendor = "apple")]
pub(crate) fn suppress_sigpipe(stream: &TcpStream) -> io::Result<()> {
    let enabled: libc::c_int = 1;
    // SAFETY: the option value is a live c_int of the advertised size; the descriptor is owned by `stream`.
    let status = unsafe {
        libc::setsockopt(
            stream.as_raw_fd(),
            libc::SOL_SOCKET,
            libc::SO_NOSIGPIPE,
            (&enabled as *const libc::c_int).cast(),
            std::mem::size_of::<libc::c_int>() as libc::socklen_t,
        )
    };
    if status == 0 {
        Ok(())
    } else {
        Err(io::Error::last_os_error())
    }
}

/// Nothing to set: every send on these platforms passes `MSG_NOSIGNAL`.
#[cfg(not(target_vendor = "apple"))]
pub(crate) fn suppress_sigpipe(_stream: &TcpStream) -> io::Result<()> {
    Ok(())
}
