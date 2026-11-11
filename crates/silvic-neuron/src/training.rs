use std::sync::Arc;

use rayon::{prelude::*, ThreadPool};
use serde_json::json;
use silvic_core::{
    backend::{BagSpec, BinSplit, DeviceTree, GradientStats, LeafRange, ObjectiveSetup},
    objective,
    quantize::{self, QuantSpec},
    Backend, BinnedDataset, Result, SilvicError,
};

use crate::runtime::{Buffer, DType, Device, Execution, TensorSpec};

const SOURCE: &str = include_str!("../kernels/training.py");
const TILE: usize = 128;
const HIST_BINS: usize = 128;
const CHUNKS: usize = 16;
const LIMBS: usize = 8;
const NODE_FIELDS: usize = 11;

fn error(message: impl Into<Arc<str>>) -> SilvicError {
    SilvicError::device(message)
}
fn install<R: Send>(pool: &Option<Arc<ThreadPool>>, f: impl FnOnce() -> R + Send) -> R {
    match pool {
        Some(pool) => pool.install(f),
        None => f(),
    }
}
fn spec(name: &str, dtype: DType, shape: &[usize]) -> TensorSpec {
    let mut shape = shape.to_vec();
    shape.push(1);
    TensorSpec::new(name, dtype, shape)
}
fn padded(rows: usize) -> Result<usize> {
    rows.max(1)
        .checked_add(TILE - 1)
        .map(|n| n / TILE * TILE)
        .ok_or_else(|| error("Neuron row capacity overflow"))
}
fn zeros(device: &Device, words: usize) -> Result<Buffer> {
    let bytes = words
        .max(1)
        .checked_mul(4)
        .ok_or_else(|| error("Neuron buffer size overflow"))?;
    device.allocate(bytes)
}

struct Dataset {
    rows: usize,
    capacity: usize,
    bins: Buffer,
    scores: Buffer,
    next_scores: Buffer,
    columns: usize,
    group_columns: Vec<usize>,
}

struct Work {
    partition: Buffer,
    values: Buffer,
    limbs: Buffer,
    meta: Buffer,
    split: Buffer,
    counts: Buffer,
    offsets: Buffer,
    partials: Buffer,
    leaf_partials: Buffer,
    histogram: Execution,
    leaf_sum: Execution,
    partition_flags: Execution,
    partition_scatter: Execution,
    copy_partition: Execution,
    host_partials: Vec<f32>,
    host_leaf_partials: Vec<f32>,
    host_counts: Vec<u32>,
    host_offsets: Vec<u32>,
}

struct TreeWork {
    capacity: usize,
    nodes: Buffer,
    leaves: Buffer,
    meta: Buffer,
    executions: Vec<(Execution, Execution)>,
}

struct Loaded {
    datasets: Vec<Dataset>,
    train: BinnedDataset,
    objective: ObjectiveSetup,
    grad: Vec<f32>,
    hess: Vec<f32>,
    host_values: Vec<f32>,
    quant_values: Vec<f32>,
    host_limbs: Vec<i32>,
    host_scores: Vec<f32>,
    bag: Option<(BagSpec, Vec<u32>)>,
    partition_len: usize,
    quantized: bool,
    quant_ready: bool,
    work: Work,
    tree: Option<TreeWork>,
}

/// Neuron-resident histogram, stable partition and binned tree-scoring backend.
/// Objectives, stochastic quantization, bag draws and fixed-point encoding use
/// core's host helpers; no CPU backend is constructed or used.
pub struct NeuronBackend {
    device: Device,
    pool: Option<Arc<ThreadPool>>,
    loaded: Option<Loaded>,
}

impl NeuronBackend {
    pub fn new(ordinal: Option<usize>) -> Result<Self> {
        Ok(Self {
            device: Device::new(ordinal)?,
            pool: None,
            loaded: None,
        })
    }

    fn state(&mut self) -> Result<&mut Loaded> {
        self.loaded
            .as_mut()
            .ok_or_else(|| error("Neuron backend used before load"))
    }
}

fn upload_dataset(device: &Device, data: &BinnedDataset) -> Result<Dataset> {
    if data.num_rows > u32::MAX as usize {
        return Err(error("Neuron supports at most 2^32 - 1 rows per rank"));
    }
    let capacity = padded(data.num_rows)?;
    let mut group_columns = Vec::with_capacity(data.groups.len());
    let mut columns = 0usize;
    for group in &data.groups {
        group_columns.push(columns);
        columns = columns
            .checked_add(group.row_stride as usize)
            .ok_or_else(|| error("Neuron column overflow"))?;
        if group.data.len()
            != data
                .num_rows
                .checked_mul(group.row_stride as usize)
                .ok_or_else(|| error("group size overflow"))?
        {
            return Err(SilvicError::data("feature group does not match its row stride"));
        }
    }
    let words = columns
        .max(1)
        .checked_mul(capacity)
        .ok_or_else(|| error("Neuron dataset size overflow"))?;
    if words > i32::MAX as usize {
        return Err(error("Neuron binned tensor exceeds signed 32-bit gather addressing"));
    }
    let mut bins = vec![0u32; words];
    for (g, group) in data.groups.iter().enumerate() {
        for row in 0..data.num_rows {
            for col in 0..group.row_stride as usize {
                bins[(group_columns[g] + col) * capacity + row] =
                    u32::from(group.data[row * group.row_stride as usize + col]);
            }
        }
    }
    Ok(Dataset {
        rows: data.num_rows,
        capacity,
        bins: device.upload(&bins)?,
        scores: device.upload(&vec![0f32; capacity])?,
        next_scores: zeros(device, capacity)?,
        columns: columns.max(1),
        group_columns,
    })
}

fn bind(
    device: &Device,
    entry: &str,
    constants: serde_json::Value,
    inputs: Vec<TensorSpec>,
    outputs: Vec<TensorSpec>,
    ib: &[Buffer],
    ob: &[Buffer],
) -> Result<Execution> {
    device
        .compile(SOURCE, entry, &constants, &inputs, &outputs)?
        .bind(ib, ob)
}

fn work(device: &Device, data: &Dataset) -> Result<Work> {
    let n = data.capacity;
    let blocks = n / TILE;
    let partition = zeros(device, n)?;
    let next_partition = zeros(device, n)?;
    let values = zeros(device, 2 * n)?;
    let limbs = zeros(device, LIMBS * n)?;
    let meta = device.upload(&[0u32; 8])?;
    let split = device.upload(&[0u32; 8])?;
    let flags = zeros(device, n)?;
    let counts = zeros(device, blocks)?;
    let offsets = zeros(device, 2 * blocks)?;
    let partials = zeros(device, CHUNKS * 2 * HIST_BINS)?;
    let leaf_partials = zeros(device, CHUNKS * LIMBS)?;
    let constants = json!({"ROWS": n, "COLUMNS": data.columns, "TILE": TILE, "CHUNKS": CHUNKS, "BINS": HIST_BINS});
    let histogram = bind(
        device,
        "histogram",
        constants,
        vec![
            spec("bins", DType::I32, &[data.columns * n]),
            spec("partition", DType::I32, &[n]),
            spec("values", DType::F32, &[2, n]),
            spec("meta", DType::I32, &[8]),
        ],
        vec![spec("partials", DType::F32, &[CHUNKS, 2, HIST_BINS])],
        &[data.bins.clone(), partition.clone(), values.clone(), meta.clone()],
        std::slice::from_ref(&partials),
    )?;
    let leaf_sum = bind(
        device,
        "leaf_sums",
        json!({"ROWS":n,"TILE":TILE,"CHUNKS":CHUNKS,"LIMBS":LIMBS}),
        vec![
            spec("partition", DType::I32, &[n]),
            spec("limbs", DType::I32, &[LIMBS, n]),
            spec("meta", DType::I32, &[8]),
        ],
        vec![spec("partials", DType::F32, &[CHUNKS, LIMBS])],
        &[partition.clone(), limbs.clone(), meta.clone()],
        std::slice::from_ref(&leaf_partials),
    )?;
    let partition_flags = bind(
        device,
        "partition_flags",
        json!({"ROWS":n,"TILE":TILE}),
        vec![
            spec("bins", DType::I32, &[data.columns * n]),
            spec("partition", DType::I32, &[n]),
            spec("meta", DType::I32, &[8]),
            spec("split", DType::I32, &[8]),
        ],
        vec![spec("flags", DType::I32, &[n]), spec("counts", DType::I32, &[blocks])],
        &[data.bins.clone(), partition.clone(), meta.clone(), split.clone()],
        &[flags.clone(), counts.clone()],
    )?;
    let partition_scatter = bind(
        device,
        "partition_scatter",
        json!({"ROWS":n,"TILE":TILE}),
        vec![
            spec("partition", DType::I32, &[n]),
            spec("flags", DType::I32, &[n]),
            spec("offsets", DType::I32, &[2 * blocks]),
            spec("meta", DType::I32, &[8]),
        ],
        vec![spec("result", DType::I32, &[n])],
        &[partition.clone(), flags.clone(), offsets.clone(), meta.clone()],
        std::slice::from_ref(&next_partition),
    )?;
    let copy_partition = bind(
        device,
        "copy_u32",
        json!({"ROWS":n,"TILE":TILE}),
        vec![spec("source", DType::I32, &[n])],
        vec![spec("result", DType::I32, &[n])],
        std::slice::from_ref(&next_partition),
        std::slice::from_ref(&partition),
    )?;
    Ok(Work {
        partition,
        values,
        limbs,
        meta,
        split,
        counts,
        offsets,
        partials,
        leaf_partials,
        histogram,
        leaf_sum,
        partition_flags,
        partition_scatter,
        copy_partition,
        host_partials: vec![0.0; CHUNKS * 2 * HIST_BINS],
        host_leaf_partials: vec![0.0; CHUNKS * LIMBS],
        host_counts: vec![0; blocks],
        host_offsets: vec![0; 2 * blocks],
    })
}

fn check_range(state: &Loaded, leaf: LeafRange) -> Result<()> {
    if u64::from(leaf.begin) + u64::from(leaf.count) > state.partition_len as u64 {
        return Err(error("leaf lies outside the active partition"));
    }
    Ok(())
}

// Four radix-2^16 digits encode a signed i64 without a lossy i64-to-f32 cast.
// The top digit is signed. Every 128-row partial is <=128*65535<2^24,
// so every intermediate integer addition in the f32 reduction is exact.
fn encode_fixed(value: i64, out: &mut [i32; 4]) {
    for (k, digit) in out.iter_mut().enumerate() {
        *digit = if k == 3 {
            (value >> 48) as i32
        } else {
            ((value as u64 >> (16 * k)) & 65535) as i32
        };
    }
}
fn fold_fixed(digits: &[f32]) -> Result<i128> {
    let mut value = 0i128;
    for (k, &digit) in digits.iter().enumerate() {
        if !digit.is_finite() || digit.fract() != 0.0 || digit.abs() > (TILE * 65535) as f32 {
            return Err(error("invalid Neuron fixed-point limb reduction"));
        }
        value += (digit as i128) << (16 * k);
    }
    Ok(value)
}

impl Backend for NeuronBackend {
    fn name(&self) -> String {
        format!("{} (neuron; host objectives)", self.device.name())
    }
    fn set_thread_pool(&mut self, pool: Arc<ThreadPool>) {
        self.pool = Some(pool);
    }

    fn load(&mut self, train: &BinnedDataset, valid: &[&BinnedDataset], objective: &ObjectiveSetup) -> Result<()> {
        if train.label.len() != train.num_rows || train.weight.as_ref().is_some_and(|w| w.len() != train.num_rows) {
            return Err(SilvicError::data("training label/weight length differs from row count"));
        }
        if let ObjectiveSetup::LambdaRank { inverse_max_dcg, .. } = objective {
            let q = train
                .query_boundaries
                .as_ref()
                .ok_or_else(|| SilvicError::data("ranking requires query boundaries"))?;
            if q.first() != Some(&0)
                || q.last().copied().map(u64::from) != Some(train.num_rows as u64)
                || q.windows(2).any(|w| w[0] > w[1])
                || inverse_max_dcg.len() + 1 != q.len()
            {
                return Err(SilvicError::data("invalid ranking query boundaries"));
            }
        }
        for (f, meta) in train.features.iter().enumerate().filter(|(_, m)| m.num_bins != 0) {
            let group = train
                .groups
                .get(meta.group as usize)
                .ok_or_else(|| SilvicError::data("feature references missing group"))?;
            if meta.column >= group.row_stride
                || !group.features.contains(&f)
                || meta.num_bins > 256
                || meta
                    .hist_offset
                    .checked_add(meta.num_bins)
                    .is_none_or(|e| e as usize > train.total_bins)
            {
                return Err(SilvicError::data("invalid feature histogram layout"));
            }
            for row in 0..train.num_rows {
                if group
                    .data
                    .get(row * group.row_stride as usize + meta.column as usize)
                    .is_none_or(|&bin| u32::from(bin) >= meta.num_bins)
                {
                    return Err(SilvicError::data("feature bin outside histogram layout"));
                }
            }
        }
        let mut datasets = Vec::with_capacity(valid.len() + 1);
        for data in std::iter::once(train).chain(valid.iter().copied()) {
            if data.features != train.features
                || data.groups.len() != train.groups.len()
                || data.groups.iter().zip(&train.groups).any(|(a, b)| {
                    a.row_stride != b.row_stride
                        || a.features != b.features
                        || a.hist_offset != b.hist_offset
                        || a.total_bins != b.total_bins
                })
            {
                return Err(SilvicError::data("validation set has a different binned layout"));
            }
            datasets.push(upload_dataset(&self.device, data)?);
        }
        let work = work(&self.device, &datasets[0])?;
        let n = datasets[0].capacity;
        let host_train = BinnedDataset {
            num_rows: train.num_rows,
            row_offset: train.row_offset,
            features: train.features.clone(),
            groups: train
                .groups
                .iter()
                .map(|group| silvic_core::data::FeatureGroup {
                    features: group.features.clone(),
                    row_stride: group.row_stride,
                    hist_offset: group.hist_offset,
                    total_bins: group.total_bins,
                    data: Vec::new(),
                })
                .collect(),
            total_bins: train.total_bins,
            label: train.label.clone(),
            weight: train.weight.clone(),
            query_boundaries: train.query_boundaries.clone(),
        };
        self.loaded = Some(Loaded {
            datasets,
            train: host_train,
            objective: objective.clone(),
            grad: vec![0.0; train.num_rows],
            hess: vec![0.0; train.num_rows],
            host_values: vec![0.0; 2 * n],
            quant_values: vec![0.0; 2 * n],
            host_limbs: vec![0; LIMBS * n],
            host_scores: vec![0.0; train.num_rows],
            bag: None,
            partition_len: 0,
            quantized: false,
            quant_ready: false,
            work,
            tree: None,
        });
        Ok(())
    }

    fn init_scores(&mut self, dataset: usize, value: f64) -> Result<()> {
        let data = self
            .state()?
            .datasets
            .get_mut(dataset)
            .ok_or_else(|| error("missing score dataset"))?;
        let values = vec![value as f32; data.capacity];
        data.scores.write(&values)
    }

    fn compute_gradients(&mut self) -> Result<GradientStats> {
        let pool = self.pool.clone();
        let state = self.state()?;
        state.datasets[0].scores.read(&mut state.host_scores)?;
        install(&pool, || {
            objective::gradients(
                &state.objective,
                &state.host_scores,
                &state.train.label,
                state.train.weight.as_deref(),
                state.train.query_boundaries.as_deref(),
                &mut state.grad,
                &mut state.hess,
            )
        });
        let n = state.datasets[0].capacity;
        state.host_values[..state.grad.len()].copy_from_slice(&state.grad);
        state.host_values[n..n + state.hess.len()].copy_from_slice(&state.hess);
        state.work.values.write(&state.host_values)?;
        state.quantized = false;
        state.quant_ready = false;
        Ok(install(&pool, || GradientStats {
            max_abs_grad: state
                .grad
                .par_iter()
                .map(|v| f64::from(v.abs()))
                .reduce(|| 0.0, f64::max),
            max_hess: state
                .hess
                .par_iter()
                .map(|v| f64::from(v.abs()))
                .reduce(|| 0.0, f64::max),
        }))
    }

    fn quantize(&mut self, spec: &QuantSpec) -> Result<()> {
        if spec.grad_max_q < 0 || spec.grad_max_q > 32767 || spec.hess_max_q < 0 || spec.hess_max_q > 32767 {
            return Err(error(
                "quantized gradients/hessians must fit signed 16-bit histogram units",
            ));
        }
        let pool = self.pool.clone();
        let state = self.state()?;
        let n = state.datasets[0].capacity;
        let (g, h) = state.quant_values.split_at_mut(n);
        install(&pool, || {
            g[..state.grad.len()]
                .par_iter_mut()
                .zip(h[..state.hess.len()].par_iter_mut())
                .enumerate()
                .for_each(|(row, (g, h))| {
                    let (qg, qh) = quantize::unpack(quantize::quantize_row(
                        spec,
                        state.grad[row],
                        state.hess[row],
                        state.train.row_offset + row as u64,
                    ));
                    *g = qg as f32;
                    *h = qh as f32;
                })
        });
        state.work.values.write(&state.quant_values)?;
        state.quantized = true;
        state.quant_ready = true;
        Ok(())
    }

    fn reset_partition(&mut self, bag: &BagSpec) -> Result<u32> {
        let pool = self.pool.clone();
        let state = self.state()?;
        let rows = state.train.num_rows as u32;
        if state.bag.as_ref().is_none_or(|(cached, _)| cached != bag) {
            let selected = install(&pool, || {
                (0..rows)
                    .into_par_iter()
                    .filter(|&row| match *bag {
                        BagSpec::All => true,
                        BagSpec::Bernoulli { seed, round, fraction } => {
                            quantize::in_bag(seed, round, state.train.row_offset + u64::from(row), fraction)
                        }
                    })
                    .collect::<Vec<_>>()
            });
            state.bag = Some((*bag, selected));
        }
        let selected = &state.bag.as_ref().expect("bag initialized").1;
        state.work.partition.write(selected)?;
        state.partition_len = selected.len();
        Ok(selected.len() as u32)
    }

    fn histogram(&mut self, leaf: LeafRange, quantized: bool, out: &mut [f64]) -> Result<()> {
        let state = self.state()?;
        check_range(state, leaf)?;
        if out.len()
            != state
                .train
                .total_bins
                .checked_mul(2)
                .ok_or_else(|| error("histogram size overflow"))?
        {
            return Err(error("histogram output has wrong length"));
        }
        if quantized && !state.quant_ready {
            return Err(error("quantized histogram requested before quantize"));
        }
        if quantized != state.quantized {
            state.work.values.write(if quantized {
                &state.quant_values
            } else {
                &state.host_values
            })?;
            state.quantized = quantized;
        }
        out.fill(0.0);
        let n = state.datasets[0].capacity;
        for feature in state.train.features.iter().filter(|f| f.num_bins != 0) {
            let column = state.datasets[0].group_columns[feature.group as usize] + feature.column as usize;
            for bin in (0..feature.num_bins as usize).step_by(HIST_BINS) {
                for start in (0..leaf.count as usize).step_by(CHUNKS * TILE) {
                    let count = (leaf.count as usize - start).min(CHUNKS * TILE);
                    state.work.meta.write(&[
                        (leaf.begin as usize + start) as u32,
                        count as u32,
                        (column * n) as u32,
                        bin as u32,
                        0,
                        0,
                        0,
                        0,
                    ])?;
                    state.work.histogram.execute()?;
                    state.work.partials.read(&mut state.work.host_partials)?;
                    for chunk in 0..CHUNKS {
                        for lane in 0..HIST_BINS.min(feature.num_bins as usize - bin) {
                            for component in 0..2 {
                                let value = state.work.host_partials[(chunk * 2 + component) * HIST_BINS + lane];
                                if quantized
                                    && (!value.is_finite()
                                        || value.fract() != 0.0
                                        || value.abs() > (TILE * 32767) as f32)
                                {
                                    return Err(error("invalid exact Neuron quantized histogram partial"));
                                }
                                // There are <=u32::MAX rows and |unit|<=32767: every
                                // integer prefix sum is <2^47, hence the f64 fold is exact.
                                out[2 * (feature.hist_offset as usize + bin + lane) + component] += f64::from(value);
                            }
                        }
                    }
                }
            }
        }
        Ok(())
    }

    fn partition(&mut self, leaf: LeafRange, split: &BinSplit) -> Result<u32> {
        let state = self.state()?;
        check_range(state, leaf)?;
        let group = state
            .train
            .groups
            .get(split.group as usize)
            .ok_or_else(|| error("split references missing group"))?;
        if split.column >= group.row_stride {
            return Err(error("split column outside group"));
        }
        if leaf.count == 0 {
            return Ok(0);
        }
        let n = state.datasets[0].capacity;
        let column = state.datasets[0].group_columns[split.group as usize] + split.column as usize;
        state.work.meta.write(&[
            leaf.begin,
            leaf.count,
            (column * n) as u32,
            state.partition_len as u32,
            0,
            0,
            0,
            0,
        ])?;
        state.work.split.write(&split.left_bins)?;
        state.work.partition_flags.execute()?;
        state.work.counts.read(&mut state.work.host_counts)?;
        let left: u32 = state.work.host_counts.iter().try_fold(0u32, |s, &c| {
            s.checked_add(c).ok_or_else(|| error("partition count overflow"))
        })?;
        if left > leaf.count {
            return Err(error("invalid Neuron partition count"));
        }
        let blocks = n / TILE;
        let mut l = leaf.begin;
        let mut r = leaf.begin + left;
        for block in 0..blocks {
            state.work.host_offsets[block] = l;
            state.work.host_offsets[blocks + block] = r;
            let begin = (block * TILE).max(leaf.begin as usize);
            let end = ((block + 1) * TILE).min(leaf.begin as usize + leaf.count as usize);
            let count = end.saturating_sub(begin) as u32;
            let lc = state.work.host_counts[block];
            if lc > count {
                return Err(error("invalid Neuron tile partition count"));
            }
            l += lc;
            r += count - lc;
        }
        state.work.offsets.write(&state.work.host_offsets)?;
        state.work.partition_scatter.execute()?;
        state.work.copy_partition.execute()?;
        Ok(left)
    }

    fn leaf_sums(&mut self, leaves: &[LeafRange], grad_scale: f32, hess_scale: f32) -> Result<Vec<[i64; 2]>> {
        let pool = self.pool.clone();
        let state = self.state()?;
        for &leaf in leaves {
            check_range(state, leaf)?;
        }
        let n = state.datasets[0].capacity;
        install(&pool, || {
            state
                .host_limbs
                .par_chunks_mut(4 * n)
                .enumerate()
                .for_each(|(component, out)| {
                    let (values, scale) = if component == 0 {
                        (&state.grad, grad_scale)
                    } else {
                        (&state.hess, hess_scale)
                    };
                    let (lo, rest) = out.split_at_mut(n);
                    let (mid, rest) = rest.split_at_mut(n);
                    let (hi, sign) = rest.split_at_mut(n);
                    lo.par_iter_mut()
                        .zip(mid.par_iter_mut())
                        .zip(hi.par_iter_mut())
                        .zip(sign.par_iter_mut())
                        .zip(values.par_iter())
                        .for_each(|((((lo, mid), hi), sign), &value)| {
                            let mut digits = [0; 4];
                            encode_fixed(quantize::to_fixed(value, scale), &mut digits);
                            [*lo, *mid, *hi, *sign] = digits;
                        });
                })
        });
        state.work.limbs.write(&state.host_limbs)?;
        let mut result = Vec::with_capacity(leaves.len());
        for leaf in leaves {
            let mut sums = [0i128; 2];
            for start in (0..leaf.count as usize).step_by(CHUNKS * TILE) {
                state.work.meta.write(&[
                    (leaf.begin as usize + start) as u32,
                    (leaf.count as usize - start).min(CHUNKS * TILE) as u32,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                ])?;
                state.work.leaf_sum.execute()?;
                state.work.leaf_partials.read(&mut state.work.host_leaf_partials)?;
                for chunk in 0..CHUNKS {
                    for (component, sum) in sums.iter_mut().enumerate() {
                        let offset = chunk * LIMBS + component * 4;
                        *sum += fold_fixed(&state.work.host_leaf_partials[offset..offset + 4])?;
                    }
                }
            }
            result.push([
                i64::try_from(sums[0]).map_err(|_| error("fixed-point gradient sum exceeds i64"))?,
                i64::try_from(sums[1]).map_err(|_| error("fixed-point hessian sum exceeds i64"))?,
            ]);
        }
        Ok(result)
    }

    fn add_tree(&mut self, tree: &DeviceTree) -> Result<()> {
        let device = self.device.clone();
        let state = self.state()?;
        if tree.leaf_values.is_empty() || tree.nodes.len().checked_add(1) != Some(tree.leaf_values.len()) {
            return Err(error("malformed device tree"));
        }
        let count = tree
            .nodes
            .len()
            .checked_add(tree.leaf_values.len())
            .ok_or_else(|| error("Neuron tree size overflow"))?;
        let capacity = count
            .checked_next_power_of_two()
            .ok_or_else(|| error("Neuron tree capacity overflow"))?;
        let words = capacity
            .checked_mul(NODE_FIELDS)
            .filter(|&words| words <= i32::MAX as usize)
            .ok_or_else(|| error("Neuron tree exceeds signed 32-bit gather addressing"))?;
        let mut nodes = vec![0u32; words];
        let mut values = vec![0f32; capacity];
        let internal = tree.nodes.len();
        let mut depths = vec![0usize; internal];
        let mut depth = 0usize;
        for (i, node) in tree.nodes.iter().enumerate() {
            let group = state
                .train
                .groups
                .get(node.split.group as usize)
                .ok_or_else(|| error("tree references missing group"))?;
            if node.split.column >= group.row_stride {
                return Err(error("tree split column outside group"));
            }
            nodes[i] = (state.datasets[0].group_columns[node.split.group as usize] + node.split.column as usize) as u32;
            for (field, child) in [(1, node.left), (2, node.right)] {
                let child = if child >= 0 {
                    let child = child as usize;
                    if child <= i || child >= internal {
                        return Err(error("tree children must follow their parent"));
                    }
                    depths[child] = depths[i] + 1;
                    child
                } else {
                    let leaf = (!child) as usize;
                    if leaf >= tree.leaf_values.len() {
                        return Err(error("invalid tree leaf index"));
                    }
                    depth = depth.max(depths[i] + 1);
                    internal + leaf
                };
                nodes[field * capacity + i] = child as u32;
            }
            for word in 0..8 {
                nodes[(3 + word) * capacity + i] = node.split.left_bins[word];
            }
        }
        for (leaf, &value) in tree.leaf_values.iter().enumerate() {
            let i = internal + leaf;
            nodes[capacity + i] = i as u32;
            nodes[2 * capacity + i] = i as u32;
            values[i] = value;
        }
        if state.tree.as_ref().is_none_or(|work| work.capacity != capacity) {
            let node_buffer = device.upload(&nodes)?;
            let leaf_buffer = device.upload(&values)?;
            let meta = device.upload(&[0u32; 8])?;
            let mut executions = Vec::with_capacity(state.datasets.len());
            for data in &state.datasets {
                let add = bind(
                    &device,
                    "add_tree",
                    json!({"ROWS":data.capacity,"COUNT":data.rows,"TILE":TILE,"NODES":capacity}),
                    vec![
                        spec("bins", DType::I32, &[data.columns * data.capacity]),
                        spec("scores", DType::F32, &[data.capacity]),
                        spec("nodes", DType::I32, &[NODE_FIELDS, capacity]),
                        spec("leaves", DType::F32, &[capacity]),
                        spec("meta", DType::I32, &[8]),
                    ],
                    vec![spec("result", DType::F32, &[data.capacity])],
                    &[
                        data.bins.clone(),
                        data.scores.clone(),
                        node_buffer.clone(),
                        leaf_buffer.clone(),
                        meta.clone(),
                    ],
                    std::slice::from_ref(&data.next_scores),
                )?;
                let copy = bind(
                    &device,
                    "copy_f32",
                    json!({"ROWS":data.capacity,"TILE":TILE}),
                    vec![spec("source", DType::F32, &[data.capacity])],
                    vec![spec("result", DType::F32, &[data.capacity])],
                    std::slice::from_ref(&data.next_scores),
                    std::slice::from_ref(&data.scores),
                )?;
                executions.push((add, copy));
            }
            state.tree = Some(TreeWork {
                capacity,
                nodes: node_buffer,
                leaves: leaf_buffer,
                meta,
                executions,
            });
        }
        let work = state.tree.as_mut().expect("tree workspace initialized");
        work.nodes.write(&nodes)?;
        work.leaves.write(&values)?;
        work.meta.write(&[depth as u32, 0, 0, 0, 0, 0, 0, 0])?;
        for (add, copy) in &mut work.executions {
            add.execute()?;
            copy.execute()?;
        }
        Ok(())
    }

    fn scores(&mut self, dataset: usize) -> Result<Vec<f32>> {
        let data = self
            .state()?
            .datasets
            .get(dataset)
            .ok_or_else(|| error("missing score dataset"))?;
        let mut scores = vec![0f32; data.rows];
        data.scores.read(&mut scores)?;
        Ok(scores)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn signed_limbs_cover_i64_edges() {
        for value in [i64::MIN, i64::MIN + 1, -65537, -65536, -1, 0, 1, 65535, 65536, i64::MAX] {
            let mut digits = [0; 4];
            encode_fixed(value, &mut digits);
            assert_eq!(fold_fixed(&digits.map(|x| x as f32)).unwrap(), i128::from(value));
        }
    }

    #[test]
    fn worst_case_tile_limb_reductions_are_exact() {
        for value in [i64::MIN, i64::MAX, -1, 1, (1i64 << 30) - 1, -(1i64 << 30)] {
            let mut digits = [0; 4];
            encode_fixed(value, &mut digits);
            let sums = digits.map(|digit| {
                let mut sum = 0f32;
                for _ in 0..TILE {
                    sum += digit as f32;
                }
                sum
            });
            assert_eq!(fold_fixed(&sums).unwrap(), i128::from(value) * TILE as i128);
        }
    }

    #[test]
    fn mixed_signed_limb_fold_matches_wide_sum() {
        let values = [i64::MAX, i64::MIN, 65535, -65536, 123456789012345, -123456789012345];
        let mut sums = [0f32; 4];
        let mut exact = 0i128;
        for value in values {
            let mut digits = [0; 4];
            encode_fixed(value, &mut digits);
            for k in 0..4 {
                sums[k] += digits[k] as f32;
            }
            exact += i128::from(value);
        }
        assert_eq!(fold_fixed(&sums).unwrap(), exact);
    }

    fn dataset(rows: usize) -> BinnedDataset {
        use silvic_core::data::{BinMapper, FeatureGroup, FeatureMeta};
        let numerical = BinMapper::Numerical {
            upper_bounds: (0..254)
                .map(|i| i as f64)
                .chain(std::iter::once(f64::INFINITY))
                .collect(),
            nan_bin: true,
            min: 0.0,
            max: 254.0,
        };
        BinnedDataset {
            num_rows: rows,
            row_offset: (1u64 << 32) + 113,
            features: vec![
                FeatureMeta {
                    name: "category".into(),
                    bins: BinMapper::Categorical {
                        categories: (1..256).collect(),
                    },
                    num_bins: 256,
                    group: 0,
                    column: 0,
                    hist_offset: 0,
                },
                FeatureMeta {
                    name: "numeric_nan".into(),
                    bins: numerical,
                    num_bins: 256,
                    group: 0,
                    column: 1,
                    hist_offset: 256,
                },
                FeatureMeta {
                    name: "other_group".into(),
                    bins: BinMapper::Categorical {
                        categories: vec![1, 2, 3, 4],
                    },
                    num_bins: 5,
                    group: 1,
                    column: 0,
                    hist_offset: 512,
                },
                FeatureMeta {
                    name: "trivial".into(),
                    bins: BinMapper::Trivial,
                    num_bins: 0,
                    group: 0,
                    column: 0,
                    hist_offset: 517,
                },
            ],
            groups: vec![
                FeatureGroup {
                    features: vec![0, 1],
                    row_stride: 4,
                    hist_offset: 0,
                    total_bins: 512,
                    data: (0..rows)
                        .flat_map(|r| [(r % 256) as u8, ((r * 13 + 7) % 256) as u8, 0, 0])
                        .collect(),
                },
                FeatureGroup {
                    features: vec![2],
                    row_stride: 4,
                    hist_offset: 512,
                    total_bins: 5,
                    data: (0..rows).flat_map(|r| [(r % 5) as u8, 0, 0, 0]).collect(),
                },
            ],
            total_bins: 517,
            label: (0..rows).map(|r| (r % 3) as f32).collect(),
            weight: Some((0..rows).map(|r| 0.25 + (r % 4) as f32 * 0.25).collect()),
            query_boundaries: Some(
                (0..rows as u32)
                    .step_by(31)
                    .chain(std::iter::once(rows as u32))
                    .collect(),
            ),
        }
    }

    #[test]
    fn device_training_matches_core_across_objectives_and_modes() {
        if std::env::var("SILVIC_TEST_NEURON").as_deref() != Ok("1") {
            eprintln!("skipped Neuron device training regression; set SILVIC_TEST_NEURON=1 on real hardware");
            return;
        }
        let data = dataset(4099);
        let valid = dataset(259);
        let objectives = [
            ObjectiveSetup::Regression,
            ObjectiveSetup::Binary { sigmoid: 1.3 },
            ObjectiveSetup::LambdaRank {
                sigmoid: 1.0,
                norm: true,
                truncation_level: 10,
                label_gain: vec![0.0, 1.0, 3.0],
                inverse_max_dcg: vec![1.0; data.query_boundaries.as_ref().unwrap().len() - 1],
            },
        ];
        for objective in objectives {
            let mut data = data.clone();
            if matches!(&objective, ObjectiveSetup::Regression) {
                data.weight = None;
            }
            let mut neuron = NeuronBackend::new(None).expect("real Neuron device");
            let mut cpu = silvic_core::CpuBackend::new();
            neuron.load(&data, &[&valid], &objective).unwrap();
            cpu.load(&data, &[&valid], &objective).unwrap();
            for backend in [&mut neuron as &mut dyn Backend, &mut cpu as &mut dyn Backend] {
                backend.init_scores(0, 0.375).unwrap();
                backend.init_scores(1, -0.125).unwrap();
            }
            let stats = neuron.compute_gradients().unwrap();
            let expected_stats = cpu.compute_gradients().unwrap();
            assert_eq!(stats.max_abs_grad, expected_stats.max_abs_grad);
            assert_eq!(stats.max_hess, expected_stats.max_hess);
            let constant_hessian = objective::constant_hessian(&objective, data.weight.as_deref());
            let quant = QuantSpec::new(stats.max_abs_grad, stats.max_hess, constant_hessian, 255, true, 71, 4);
            neuron.quantize(&quant).unwrap();
            cpu.quantize(&quant).unwrap();
            let bag = BagSpec::Bernoulli {
                seed: 19,
                round: 3,
                fraction: 0.73,
            };
            let count = neuron.reset_partition(&bag).unwrap();
            assert_eq!(count, cpu.reset_partition(&bag).unwrap());
            let leaf = LeafRange {
                begin: 117,
                count: count - 233,
            };
            let mut actual = vec![0.0; 2 * data.total_bins];
            let mut expected = actual.clone();
            // Alternating modes must retain the previously quantized values.
            for quantized in [true, false, true] {
                neuron.histogram(leaf, quantized, &mut actual).unwrap();
                cpu.histogram(leaf, quantized, &mut expected).unwrap();
                for (i, (&a, &e)) in actual.iter().zip(&expected).enumerate() {
                    if quantized {
                        assert_eq!(a, e, "quantized bin {i}");
                    } else {
                        assert!((a - e).abs() <= 2e-5 * (1.0 + e.abs()), "float bin {i}: {a} vs {e}");
                    }
                }
            }
            let mut left_bins = [0u32; 8];
            left_bins[0] = 1;
            left_bins[7] = 0x8000_0001;
            let split = BinSplit {
                group: 0,
                column: 1,
                left_bins,
            };
            let left = neuron.partition(leaf, &split).unwrap();
            assert_eq!(left, cpu.partition(leaf, &split).unwrap());
            let selected: Vec<u32> = (0..data.num_rows as u32)
                .filter(|&r| quantize::in_bag(19, 3, data.row_offset + u64::from(r), 0.73))
                .collect();
            let range = leaf.begin as usize..(leaf.begin + leaf.count) as usize;
            let mut stable = selected.clone();
            let middle = &selected[range.clone()];
            let reordered: Vec<u32> = middle
                .iter()
                .copied()
                .filter(|&r| split.goes_left(data.bin(1, r as usize)))
                .chain(
                    middle
                        .iter()
                        .copied()
                        .filter(|&r| !split.goes_left(data.bin(1, r as usize))),
                )
                .collect();
            stable[range].copy_from_slice(&reordered);
            let mut actual_rows = vec![0u32; count as usize];
            neuron
                .loaded
                .as_ref()
                .unwrap()
                .work
                .partition
                .read(&mut actual_rows)
                .unwrap();
            assert_eq!(
                actual_rows, stable,
                "partition must be stable and preserve rows outside the leaf"
            );
            let leaves = [
                LeafRange {
                    begin: leaf.begin,
                    count: left,
                },
                LeafRange {
                    begin: leaf.begin + left,
                    count: leaf.count - left,
                },
                LeafRange { begin: count, count: 0 },
            ];
            let gs = quantize::fixed_point_scale(stats.max_abs_grad);
            let hs = quantize::fixed_point_scale(stats.max_hess);
            assert_eq!(
                neuron.leaf_sums(&leaves, gs, hs).unwrap(),
                cpu.leaf_sums(&leaves, gs, hs).unwrap()
            );
            let tree = DeviceTree {
                nodes: vec![silvic_core::backend::DeviceNode {
                    split,
                    left: !0,
                    right: !1,
                }],
                leaf_values: vec![0.25, -0.125],
            };
            for backend in [&mut neuron as &mut dyn Backend, &mut cpu as &mut dyn Backend] {
                backend.add_tree(&tree).unwrap();
            }
            for dataset in 0..2 {
                assert_eq!(neuron.scores(dataset).unwrap(), cpu.scores(dataset).unwrap());
            }
            let constant = DeviceTree {
                nodes: vec![],
                leaf_values: vec![0.0625],
            };
            for backend in [&mut neuron as &mut dyn Backend, &mut cpu as &mut dyn Backend] {
                backend.add_tree(&constant).unwrap();
            }
            for dataset in 0..2 {
                assert_eq!(neuron.scores(dataset).unwrap(), cpu.scores(dataset).unwrap());
            }
            assert!(neuron
                .histogram(LeafRange { begin: count, count: 1 }, true, &mut actual)
                .is_err());
            assert!(neuron
                .partition(
                    leaf,
                    &BinSplit {
                        group: 99,
                        column: 0,
                        left_bins: [0; 8]
                    }
                )
                .is_err());
        }
    }
}
