//! Checks `TcpCollective` against in-process references: ranks on threads over 127.0.0.1 for world sizes 1, 2, 3 and
//! 5, buffer lengths at every chunking and path edge, and the failure paths (early exit, hang, wrong token, strays,
//! diverged calls).

use std::io::Write;
use std::net::TcpStream;
use std::ops::Range;
use std::thread;
use std::time::{Duration, Instant};

use silvic_core::{Collective, Result, SilvicError};
use silvic_net::{Listener, TcpCollective};

const TIMEOUT: Duration = Duration::from_secs(30);
const TOKEN: u64 = 0x5111_c1c7_e57a;
const WORLDS: [usize; 4] = [1, 2, 3, 5];
const LARGE: usize = 1_000_003;
/// Time a failure may take to reach every rank on top of the timeout that triggered it.
const SLACK: Duration = Duration::from_secs(2);

/// Binds `world` listeners on 127.0.0.1 and runs `body` on each rank's thread once it has joined the ring.
fn run_world<T: Send>(
    world: usize,
    token: impl Fn(usize) -> u64 + Sync,
    timeout: Duration,
    body: impl Fn(&mut TcpCollective) -> Result<T> + Sync,
) -> Vec<Result<T>> {
    let listeners: Vec<Listener> = (0..world)
        .map(|_| TcpCollective::bind("127.0.0.1").expect("bind"))
        .collect();
    run_listeners(listeners, token, timeout, body)
}

fn run_listeners<T: Send>(
    listeners: Vec<Listener>,
    token: impl Fn(usize) -> u64 + Sync,
    timeout: Duration,
    body: impl Fn(&mut TcpCollective) -> Result<T> + Sync,
) -> Vec<Result<T>> {
    let addresses: Vec<String> = listeners.iter().map(Listener::address).collect();
    thread::scope(|scope| {
        let ranks: Vec<_> = listeners
            .into_iter()
            .enumerate()
            .map(|(rank, listener)| {
                let (addresses, token, body) = (&addresses, &token, &body);
                scope.spawn(move || {
                    let mut ring = TcpCollective::connect(listener, rank, addresses, token(rank), timeout)?;
                    body(&mut ring)
                })
            })
            .collect();
        ranks
            .into_iter()
            .map(|rank| rank.join().expect("rank thread"))
            .collect()
    })
}

fn splitmix(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

fn stream(rank: usize, len: usize, salt: u64) -> impl Iterator<Item = u64> {
    let mut state = (rank as u64 + 1).wrapping_mul(0x2545_f491_4f6c_dd1d) ^ ((len as u64) << 8) ^ salt;
    std::iter::repeat_with(move || splitmix(&mut state)).take(len)
}

/// Floats over 40 binary orders of magnitude, so a sum's bits depend on the order it is taken in.
fn floats(rank: usize, len: usize) -> Vec<f64> {
    stream(rank, len, 1)
        .map(|bits| ((bits >> 11) as f64 / (1u64 << 53) as f64 - 0.5) * 2f64.powi((bits % 41) as i32 - 20))
        .collect()
}

/// Integer-valued floats below 2^40: their sum is exact in any order, like a quantized histogram's.
fn integral(rank: usize, len: usize) -> Vec<f64> {
    stream(rank, len, 2)
        .map(|bits| ((bits >> 24) as i64 - (1 << 39)) as f64)
        .collect()
}

fn integers(rank: usize, len: usize) -> Vec<i64> {
    stream(rank, len, 3).map(|bits| bits as i64).collect()
}

fn bytes(rank: usize, len: usize) -> Vec<u8> {
    stream(rank, len, 4).map(|bits| bits as u8).collect()
}

/// The crate's documented chunking: the first `len % world` chunks hold one element more.
fn chunk_range(len: usize, world: usize, chunk: usize) -> Range<usize> {
    let (base, extra) = (len / world, len % world);
    let start = chunk * base + chunk.min(extra);
    start..start + base + usize::from(chunk < extra)
}

/// The crate's documented order: element `i` of chunk `c` folds ranks `c, c + 1, ..., c - 1`.
fn fold(inputs: &[Vec<f64>], combine: fn(f64, f64) -> f64) -> Vec<f64> {
    let (world, len) = (inputs.len(), inputs[0].len());
    let mut out = vec![0.0; len];
    for chunk in 0..world {
        for i in chunk_range(len, world, chunk) {
            out[i] = (1..world).fold(inputs[chunk][i], |acc, offset| {
                combine(acc, inputs[(chunk + offset) % world][i])
            });
        }
    }
    out
}

fn max_with_nan(acc: f64, x: f64) -> f64 {
    if x > acc || x.is_nan() {
        x
    } else {
        acc
    }
}

fn bits(values: &[f64]) -> Vec<u64> {
    values.iter().map(|value| value.to_bits()).collect()
}

/// 0, 1, `world - 1`, both sides of the 4096-byte path switch, and a large odd length.
fn lengths(world: usize) -> Vec<usize> {
    let mut lengths = vec![0, 1, world - 1, 511, 512, LARGE];
    lengths.dedup();
    lengths
}

/// One rank's results for one length: float sum, integer-valued sum, max, i64 sum.
type Outputs = (Vec<u64>, Vec<u64>, Vec<u64>, Vec<i64>);

fn allreduce_everything(ring: &mut TcpCollective) -> Result<Vec<Outputs>> {
    let rank = ring.rank();
    let mut outputs = Vec::new();
    for len in lengths(ring.world_size()) {
        let mut sum = floats(rank, len);
        ring.allreduce_sum_f64(&mut sum)?;
        let mut exact = integral(rank, len);
        ring.allreduce_sum_f64(&mut exact)?;
        let mut max = floats(rank, len);
        ring.allreduce_max_f64(&mut max)?;
        let mut ints = integers(rank, len);
        ring.allreduce_sum_i64(&mut ints)?;
        outputs.push((bits(&sum), bits(&exact), bits(&max), ints));
    }
    Ok(outputs)
}

fn expected_outputs(world: usize) -> Vec<Outputs> {
    lengths(world)
        .into_iter()
        .map(|len| {
            let floats: Vec<_> = (0..world).map(|rank| floats(rank, len)).collect();
            let exact = (0..world)
                .map(|rank| integral(rank, len))
                .fold(vec![0.0; len], |acc, values| {
                    acc.iter().zip(&values).map(|(a, b)| a + b).collect()
                });
            let ints = (0..world)
                .map(|rank| integers(rank, len))
                .fold(vec![0i64; len], |acc, ints| {
                    acc.iter().zip(&ints).map(|(a, b)| a.wrapping_add(*b)).collect()
                });
            (
                bits(&fold(&floats, |a, b| a + b)),
                bits(&exact),
                bits(&fold(&floats, max_with_nan)),
                ints,
            )
        })
        .collect()
}

#[test]
fn reference_order_is_observable() {
    // Matching the ring-order fold bit for bit only pins the order if another order gives other bits.
    let inputs: Vec<Vec<f64>> = (0..3).map(|rank| floats(rank, LARGE)).collect();
    let rank_order: Vec<f64> = (0..LARGE)
        .map(|i| (1..3).fold(inputs[0][i], |acc, rank| acc + inputs[rank][i]))
        .collect();
    assert!(
        bits(&rank_order) != bits(&fold(&inputs, |a, b| a + b)),
        "the test floats are order-sensitive"
    );
}

#[test]
fn allreduce_matches_reference_and_repeats_bit_identically() {
    for world in WORLDS {
        let expected = expected_outputs(world);
        let first: Vec<Vec<Outputs>> = run_world(world, |_| TOKEN, TIMEOUT, allreduce_everything)
            .into_iter()
            .map(|result| result.expect("allreduce"))
            .collect();
        for (rank, outputs) in first.iter().enumerate() {
            for ((len, got), want) in lengths(world).into_iter().zip(outputs).zip(&expected) {
                let case = format!("world {world} rank {rank} len {len}");
                assert!(got.0 == want.0, "{case}: f64 sum differs from the ring-order fold");
                assert!(got.1 == want.1, "{case}: integer-valued f64 sum is not exact");
                assert!(got.2 == want.2, "{case}: f64 max");
                assert!(got.3 == want.3, "{case}: i64 sum");
            }
        }
        let second: Vec<Vec<Outputs>> = run_world(world, |_| TOKEN, TIMEOUT, allreduce_everything)
            .into_iter()
            .map(|result| result.expect("allreduce"))
            .collect();
        assert!(first == second, "world {world}: a second run differs");
    }
}

#[test]
fn max_propagates_nan_from_any_rank() {
    for world in [2, 3, 5] {
        for len in [3, LARGE] {
            let results = run_world(
                world,
                |_| TOKEN,
                TIMEOUT,
                |ring| {
                    let mut values = floats(ring.rank(), len);
                    if ring.rank() == world - 1 {
                        values[1] = f64::NAN;
                    }
                    ring.allreduce_max_f64(&mut values)?;
                    Ok(values)
                },
            );
            for (rank, values) in results.into_iter().enumerate() {
                let values = values.expect("allreduce_max_f64");
                assert!(values[1].is_nan(), "world {world} len {len} rank {rank}: NaN element");
                assert!(
                    !values[0].is_nan(),
                    "world {world} len {len} rank {rank}: other elements"
                );
            }
        }
    }
}

#[test]
fn allgather_returns_every_block() {
    for world in WORLDS {
        // Block lengths by rank: all empty, then rank r holding `base + r` bytes.
        let mut cases: Vec<(String, Vec<usize>)> = vec![("empty".to_string(), vec![0; world])];
        cases.extend(
            [0, 1, world - 1, LARGE]
                .into_iter()
                .map(|base| (format!("base {base}"), (0..world).map(|rank| base + rank).collect())),
        );
        let results = run_world(
            world,
            |_| TOKEN,
            TIMEOUT,
            |ring| {
                let rank = ring.rank();
                cases
                    .iter()
                    .map(|(_, lens)| ring.allgather_bytes(&bytes(rank, lens[rank])))
                    .collect::<Result<Vec<_>>>()
            },
        );
        for (rank, gathered) in results.into_iter().enumerate() {
            for ((name, lens), blocks) in cases.iter().zip(gathered.expect("allgather_bytes")) {
                let expected: Vec<Vec<u8>> = (0..world).map(|owner| bytes(owner, lens[owner])).collect();
                assert!(blocks == expected, "world {world} rank {rank} {name}: blocks");
            }
        }
    }
}

#[test]
fn world_of_one_is_the_identity_and_opens_no_sockets() {
    let listener = TcpCollective::bind("127.0.0.1").expect("bind");
    let address = listener.address();
    let mut ring =
        TcpCollective::connect(listener, 0, std::slice::from_ref(&address), TOKEN, TIMEOUT).expect("connect");
    assert!(
        TcpStream::connect(&address).is_err(),
        "the listener is closed after a world-of-one connect"
    );
    assert_eq!((ring.rank(), ring.world_size()), (0, 1), "rank and world");
    let mut sum = floats(0, 5);
    ring.allreduce_sum_f64(&mut sum).expect("sum");
    assert_eq!(bits(&sum), bits(&floats(0, 5)), "sum is the input");
    let mut ints = integers(0, 5);
    ring.allreduce_sum_i64(&mut ints).expect("i64 sum");
    assert_eq!(ints, integers(0, 5), "i64 sum is the input");
    assert_eq!(
        ring.allgather_bytes(b"solo").expect("allgather"),
        vec![b"solo".to_vec()],
        "allgather"
    );
}

#[test]
fn early_exit_fails_the_others_within_the_timeout() {
    let timeout = Duration::from_secs(5);
    for len in [3, LARGE] {
        let results = run_world(
            3,
            |_| TOKEN,
            timeout,
            |ring| {
                let mut data = vec![1.0; len];
                ring.allreduce_sum_f64(&mut data)?;
                if ring.rank() == 2 {
                    return Ok(None);
                }
                let start = Instant::now();
                let second = ring.allreduce_sum_f64(&mut data);
                let elapsed = start.elapsed();
                let third = ring.allreduce_sum_f64(&mut data);
                Ok(Some((second, elapsed, third)))
            },
        );
        for (rank, result) in results.into_iter().enumerate().take(2) {
            let (second, elapsed, third) = result.expect("first allreduce").expect("ranks 0 and 1 continue");
            let error = second.expect_err("a peer exited");
            assert_eq!(error.kind(), "collective", "len {len} rank {rank}: {error}");
            assert!(elapsed < timeout, "len {len} rank {rank}: failed after {elapsed:?}");
            assert_eq!(
                third,
                Err(error),
                "len {len} rank {rank}: later calls return the first failure"
            );
        }
    }
}

#[test]
fn hung_rank_times_out() {
    let timeout = Duration::from_millis(500);
    let results = run_world(
        2,
        |_| TOKEN,
        timeout,
        |ring| {
            if ring.rank() == 1 {
                thread::sleep(timeout * 3);
            }
            let start = Instant::now();
            let mut data = vec![1.0; 4];
            // Rank 1's first call can still complete on the frame rank 0 sent before giving up; its next call cannot.
            let outcome = ring
                .allreduce_sum_f64(&mut data)
                .and_then(|()| ring.allreduce_sum_f64(&mut data));
            Ok((outcome, start.elapsed()))
        },
    );
    let (outcome, elapsed) = results[0].as_ref().expect("rank 0 joined");
    let error = outcome.as_ref().expect_err("rank 1 never called");
    assert!(error.to_string().contains("timed out"), "rank 0: {error}");
    assert!(error.to_string().contains("rank 1"), "rank 0 names the peer: {error}");
    assert!(
        *elapsed >= timeout && *elapsed < timeout + SLACK,
        "rank 0 failed after {elapsed:?}"
    );
    let (outcome, elapsed) = results[1].as_ref().expect("rank 1 joined");
    assert!(outcome.is_err(), "rank 1 finds the ring shut down");
    assert!(*elapsed < timeout, "rank 1 failed after {elapsed:?}");
}

#[test]
fn wrong_token_is_rejected() {
    let timeout = Duration::from_secs(2);
    let start = Instant::now();
    let results = run_world(3, |rank| if rank == 1 { TOKEN + 1 } else { TOKEN }, timeout, |_| Ok(()));
    for (rank, result) in results.into_iter().enumerate() {
        let error: SilvicError = result.expect_err("the ring must not form");
        assert_eq!(error.kind(), "collective", "rank {rank}: {error}");
        assert!(
            error.to_string().contains("token"),
            "rank {rank} names the token: {error}"
        );
    }
    assert!(
        start.elapsed() < timeout + SLACK,
        "rejection took {:?}",
        start.elapsed()
    );
}

#[test]
fn stray_connections_do_not_take_a_rank_s_place() {
    let listeners: Vec<Listener> = (0..2)
        .map(|_| TcpCollective::bind("127.0.0.1").expect("bind"))
        .collect();
    let target = listeners[0].address();
    let mut garbage = TcpStream::connect(&target).expect("stray connect");
    garbage.write_all(&[0xab; 64]).expect("stray write");
    drop(TcpStream::connect(&target).expect("stray connect"));
    let results = run_listeners(
        listeners,
        |_| TOKEN,
        TIMEOUT,
        |ring| {
            let mut data = vec![ring.rank() as f64; 3];
            ring.allreduce_sum_f64(&mut data)?;
            Ok(data)
        },
    );
    for (rank, result) in results.into_iter().enumerate() {
        assert_eq!(
            result.expect("ring forms past strays"),
            vec![1.0; 3],
            "rank {rank}: sum"
        );
    }
    drop(garbage);
}

#[test]
fn diverged_calls_fail_instead_of_mixing_buffers() {
    let results = run_world(
        2,
        |_| TOKEN,
        TIMEOUT,
        |ring| {
            let mut data = vec![1.0; 10 + ring.rank()];
            Ok(ring.allreduce_sum_f64(&mut data))
        },
    );
    for (rank, result) in results.into_iter().enumerate() {
        let error = result.expect("joined").expect_err("lengths differ");
        assert!(
            error.to_string().contains("out of step"),
            "lengths, rank {rank}: {error}"
        );
    }
    let results = run_world(
        2,
        |_| TOKEN,
        TIMEOUT,
        |ring| {
            let mut data = vec![1.0; 10];
            Ok(if ring.rank() == 0 {
                ring.allreduce_sum_f64(&mut data)
            } else {
                ring.allreduce_max_f64(&mut data)
            })
        },
    );
    for (rank, result) in results.into_iter().enumerate() {
        let error = result.expect("joined").expect_err("ops differ");
        assert!(error.to_string().contains("out of step"), "ops, rank {rank}: {error}");
    }
}
