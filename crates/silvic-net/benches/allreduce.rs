//! Latency of `TcpCollective` allreduces between ranks on threads over 127.0.0.1: the ~300 KB histogram allreduce
//! and the few-element count allreduces each tree split runs.

use std::sync::mpsc;
use std::thread;
use std::time::{Duration, Instant};

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use silvic_core::Collective;
use silvic_net::{Listener, TcpCollective};

const TIMEOUT: Duration = Duration::from_secs(30);
const JOB_TOKEN: u64 = 0x5eed_0000_be4c;

/// Rank 0 runs on the bench thread; ranks 1.. loop on their own threads, running as many allreduces as each batch
/// the leader times.
struct World {
    leader: TcpCollective,
    batches: Vec<mpsc::Sender<u64>>,
    followers: Vec<thread::JoinHandle<()>>,
}

impl World {
    fn start(world: usize, len: usize) -> Self {
        let listeners: Vec<Listener> = (0..world)
            .map(|_| TcpCollective::bind("127.0.0.1").expect("bind listener"))
            .collect();
        let addresses: Vec<String> = listeners.iter().map(Listener::address).collect();
        let mut listeners = listeners.into_iter();
        let leader_listener = listeners.next().expect("rank 0 listener");
        let mut batches = Vec::new();
        let mut followers = Vec::new();
        for (index, listener) in listeners.enumerate() {
            let (send, receive) = mpsc::channel::<u64>();
            let addresses = addresses.clone();
            batches.push(send);
            followers.push(thread::spawn(move || {
                let mut ring = TcpCollective::connect(listener, index + 1, &addresses, JOB_TOKEN, TIMEOUT)
                    .expect("follower connect");
                let mut buf = vec![0.0; len];
                while let Ok(iterations) = receive.recv() {
                    for _ in 0..iterations {
                        ring.allreduce_sum_f64(&mut buf).expect("follower allreduce");
                    }
                }
            }));
        }
        let leader =
            TcpCollective::connect(leader_listener, 0, &addresses, JOB_TOKEN, TIMEOUT).expect("leader connect");
        Self {
            leader,
            batches,
            followers,
        }
    }

    fn time(&mut self, buf: &mut [f64], iterations: u64) -> Duration {
        for batch in &self.batches {
            batch.send(iterations).expect("follower alive");
        }
        let start = Instant::now();
        for _ in 0..iterations {
            self.leader.allreduce_sum_f64(buf).expect("leader allreduce");
        }
        start.elapsed()
    }

    fn stop(self) {
        drop(self.batches);
        for follower in self.followers {
            follower.join().expect("follower thread");
        }
    }
}

fn allreduce_sum_f64(c: &mut Criterion) {
    let mut group = c.benchmark_group("allreduce_sum_f64");
    // 3 elements: a split's left count and sums; 37,500 f64 = 300 KB: one histogram of the target workload.
    for (world, len) in [(2, 3), (4, 3), (2, 37_500), (4, 37_500), (8, 37_500)] {
        group.throughput(Throughput::Bytes((len * std::mem::size_of::<f64>()) as u64));
        let mut ring = World::start(world, len);
        let mut buf = vec![0.0; len];
        group.bench_function(BenchmarkId::new(format!("world_{world}"), len), |b| {
            b.iter_custom(|iterations| ring.time(&mut buf, iterations));
        });
        ring.stop();
    }
    group.finish();
}

criterion_group!(benches, allreduce_sum_f64);
criterion_main!(benches);
