# silvic-net

`TcpCollective`: the collectives distributed silvic training runs on (`allreduce_sum_f64`, `allreduce_sum_i64`,
`allreduce_max_f64`, `allgather_bytes`), over a TCP ring. `silvic.ray.train` creates one per Ray actor.

Each rank binds a listener (`TcpCollective::bind`, or `bind_advertised` to listen on `0.0.0.0` and advertise the
node IP), the addresses are exchanged out of band, and `TcpCollective::connect` joins the ring: every rank connects
to the next and accepts from the previous, with a handshake carrying a job token, rank and world size that rejects
stray or mismatched peers. Allreduces are a ring reduce-scatter and allgather (an allgather below 4 KiB); every
chunk folds in a fixed rank order, so results are deterministic for a world size, and exact in any order for the
integer-valued histograms quantized training reduces.

Sockets are non-blocking and polled, so full-duplex ring steps never deadlock on large buffers. Every frame carries
the operation, a sequence number and the length: ranks that call different collectives fail with an "out of step"
error rather than exchanging garbage. A socket error, a closed peer or a timeout fails the collective naming the
peer rank, shuts the rank's sockets so the failure travels around the ring, and fails every later call.

## Tests

`tests/ring.rs` runs world sizes 1, 2, 3 and 5 on threads over 127.0.0.1: sums, maxima and gathers against direct
computation for lengths 0, 1, world - 1, both sides of the 4 KiB switch and 1,000,003; bit-identical repeats; and the
failure paths (an early exit, a hung rank, a wrong token, stray connections, diverging calls).
