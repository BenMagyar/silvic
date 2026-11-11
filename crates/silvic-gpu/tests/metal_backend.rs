//! Compares `GpuBackend<MetalDevice>` against `silvic_core::CpuBackend` (and the host objective functions) on random
//! binned data: integer results and the f32 score and L2 gradient updates bit for bit, the other float results to
//! f32 rounding. Skips with a printed reason when the machine has no Metal device.

#![cfg(all(feature = "metal", target_os = "macos"))]

use silvic_core::backend::{BagSpec, BinSplit, DeviceNode, DeviceTree, GradientStats, LeafRange, ObjectiveSetup, Side};
use silvic_core::config::{Metric, Objective};
use silvic_core::data::{DatasetInput, DenseData, DenseMatrix, Layout};
use silvic_core::quantize::{self, QuantSpec};
use silvic_core::{objective, Backend, BinnedDataset, CpuBackend, LocalCollective, Params};
use silvic_gpu::metal::{self, MetalDevice};
use silvic_gpu::GpuBackend;

/// splitmix64: a small deterministic generator, so the test data needs no extra dependency.
struct Rng(u64);

impl Rng {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    fn below(&mut self, n: u64) -> u64 {
        self.next_u64() % n
    }

    fn normal(&mut self) -> f64 {
        let u = self.uniform().max(1e-300);
        let v = self.uniform();
        (-2.0 * u.ln()).sqrt() * (std::f64::consts::TAU * v).cos()
    }
}

/// Bindings box the backend and move it across threads.
#[test]
fn metal_backend_is_send() {
    fn assert_send<T: Send>() {}
    assert_send::<GpuBackend<MetalDevice>>();
}

fn metal_backend(test: &str) -> Option<GpuBackend<MetalDevice>> {
    if !metal::is_available() {
        eprintln!("{test}: skipped, no Metal device on this machine");
        return None;
    }
    Some(metal::backend(None).expect("metal backend"))
}

/// Row-major features of mixed kinds: low-cardinality, continuous with NaN, categorical (with negatives) and
/// continuous, plus labels.
struct RawData {
    features: Vec<f64>,
    rows: usize,
    cols: usize,
    label: Vec<f32>,
    weight: Vec<f32>,
    categorical: Vec<usize>,
}

fn raw_data(rng: &mut Rng, rows: usize, cols: usize) -> RawData {
    let categorical: Vec<usize> = (0..cols).filter(|c| c % 7 == 3).collect();
    let mut features = Vec::with_capacity(rows * cols);
    let mut label = Vec::with_capacity(rows);
    for _ in 0..rows {
        let mut signal = 0.0;
        for col in 0..cols {
            let value = match col % 7 {
                0 => rng.below(10) as f64,
                1 if rng.uniform() < 0.05 => f64::NAN,
                3 => rng.below(18) as f64 - 2.0,
                _ => rng.normal(),
            };
            if col < 6 && !value.is_nan() {
                signal += value * (col as f64 + 1.0) * 0.1;
            }
            features.push(value);
        }
        label.push((signal + 0.3 * rng.normal()) as f32);
    }
    let weight = (0..rows).map(|_| (0.5 + rng.uniform()) as f32).collect();
    RawData {
        features,
        rows,
        cols,
        label,
        weight,
        categorical,
    }
}

fn input<'a>(raw: &'a RawData, label: &'a [f32], weighted: bool, group: Option<&'a [u32]>) -> DatasetInput<'a> {
    DatasetInput {
        features: DenseMatrix {
            data: DenseData::F64(&raw.features),
            rows: raw.rows,
            cols: raw.cols,
            layout: Layout::RowMajor,
        },
        label,
        weight: weighted.then_some(raw.weight.as_slice()),
        group,
        feature_names: None,
        categorical_features: raw.categorical.clone(),
    }
}

fn construct(input: &DatasetInput<'_>, params: &Params) -> BinnedDataset {
    BinnedDataset::construct(input, params, &mut LocalCollective).expect("construct")
}

fn construct_valid(input: &DatasetInput<'_>, reference: &BinnedDataset) -> BinnedDataset {
    BinnedDataset::construct_with_reference(input, reference, &mut LocalCollective).expect("construct valid")
}

fn non_trivial_features(data: &BinnedDataset) -> Vec<usize> {
    (0..data.features.len())
        .filter(|&f| !data.features[f].is_trivial())
        .collect()
}

/// A random split of `feature`: numerical features send bins `0..=t` (and maybe the NaN bin) left, categorical ones
/// a random bin subset.
fn random_split(rng: &mut Rng, data: &BinnedDataset, feature: usize) -> BinSplit {
    let meta = &data.features[feature];
    let mut left_bins = [0u32; 8];
    if meta.is_categorical() {
        for bin in 0..meta.num_bins {
            if rng.below(2) == 1 {
                left_bins[bin as usize / 32] |= 1 << (bin % 32);
            }
        }
    } else {
        let threshold = rng.below(u64::from(meta.num_bins)) as u32;
        for bin in 0..=threshold {
            left_bins[bin as usize / 32] |= 1 << (bin % 32);
        }
    }
    BinSplit {
        group: meta.group,
        column: meta.column,
        left_bins,
    }
}

/// A random tree in LightGBM's node layout: splitting leaf `l` makes node `num_leaves - 1` with `l` on the left and
/// the new leaf `num_leaves` on the right.
fn random_tree(rng: &mut Rng, data: &BinnedDataset, num_leaves: usize) -> DeviceTree {
    let features = non_trivial_features(data);
    let mut nodes: Vec<DeviceNode> = Vec::new();
    let mut leaf_parent: Vec<Option<(usize, bool)>> = vec![None];
    for new_leaf in 1..num_leaves {
        let leaf = rng.below(new_leaf as u64) as usize;
        let feature = features[rng.below(features.len() as u64) as usize];
        let node = nodes.len();
        nodes.push(DeviceNode {
            split: random_split(rng, data, feature),
            left: !(leaf as i32),
            right: !(new_leaf as i32),
        });
        if let Some((parent, is_left)) = leaf_parent[leaf] {
            if is_left {
                nodes[parent].left = node as i32;
            } else {
                nodes[parent].right = node as i32;
            }
        }
        leaf_parent[leaf] = Some((node, true));
        leaf_parent.push(Some((node, false)));
    }
    let leaf_values = (0..num_leaves).map(|_| (rng.normal() * 0.3) as f32).collect();
    DeviceTree { nodes, leaf_values }
}

fn assert_close(case: &str, gpu: &[f32], cpu: &[f32], relative: f32, absolute: f32) {
    assert_eq!(gpu.len(), cpu.len(), "{case}: length");
    let mut worst = (0usize, 0.0f32);
    for (row, (&a, &b)) in gpu.iter().zip(cpu).enumerate() {
        let error = (a - b).abs();
        let allowed = absolute.max(relative * b.abs());
        assert!(error <= allowed, "{case}: row {row}: metal {a}, cpu {b}");
        if error > worst.1 {
            worst = (row, error);
        }
    }
    eprintln!("{case}: largest difference {} at row {}", worst.1, worst.0);
}

/// Float histograms: each bin within `relative` of the sum of the absolute values it adds (the scale of f32
/// summation error) or of the bin itself.
fn assert_float_hist_close(case: &str, gpu: &[f64], cpu: &[f64], scale: &[f64], relative: f64) {
    for (bin, ((&a, &b), &s)) in gpu.iter().zip(cpu).zip(scale).enumerate() {
        let allowed = relative * s.max(b.abs()) + 1e-12;
        assert!(
            (a - b).abs() <= allowed,
            "{case}: value {bin}: metal {a}, cpu {b}, scale {s}"
        );
    }
}

fn hist(backend: &mut dyn Backend, leaf: LeafRange, quantized: bool, total_bins: usize) -> Vec<f64> {
    let mut out = vec![f64::NAN; 2 * total_bins];
    backend.histogram(leaf, quantized, &mut out).expect("histogram");
    out
}

/// Per-bin sums of `|g|` and `|h|` over `rows`, the error scale of a float histogram.
fn abs_hist(data: &BinnedDataset, rows: &[u32], grad: &[f32], hess: &[f32]) -> Vec<f64> {
    let mut out = vec![0.0; 2 * data.total_bins];
    for feature in non_trivial_features(data) {
        let offset = data.features[feature].hist_offset as usize;
        for &row in rows {
            let bin = offset + usize::from(data.bin(feature, row as usize));
            out[2 * bin] += f64::from(grad[row as usize].abs());
            out[2 * bin + 1] += f64::from(hess[row as usize].abs());
        }
    }
    out
}

struct Pair<'a> {
    cpu: CpuBackend<'a>,
    gpu: GpuBackend<MetalDevice>,
    total_bins: usize,
}

impl Pair<'_> {
    /// `partition_and_histogram` (quantized) on both backends; asserts the histograms agree, returns the left counts.
    fn fused(&mut self, case: &str, leaf: LeafRange, split: &BinSplit, side: Side) -> (u32, u32) {
        let mut gpu = vec![f64::NAN; 2 * self.total_bins];
        let mut cpu = vec![f64::NAN; 2 * self.total_bins];
        let left_gpu = self
            .gpu
            .partition_and_histogram(leaf, split, side, true, &mut gpu)
            .expect("metal partition and histogram");
        let left_cpu = self
            .cpu
            .partition_and_histogram(leaf, split, side, true, &mut cpu)
            .expect("cpu partition and histogram");
        assert!(gpu == cpu, "{case}: quantized histogram of the child of {leaf:?}");
        (left_gpu, left_cpu)
    }

    fn hist_equal(&mut self, case: &str, leaf: LeafRange) {
        let gpu = hist(&mut self.gpu, leaf, true, self.total_bins);
        let cpu = hist(&mut self.cpu, leaf, true, self.total_bins);
        assert!(gpu == cpu, "{case}: quantized histogram of {leaf:?}");
    }

    fn quantize(&mut self, spec: &QuantSpec) {
        self.gpu.quantize(spec).expect("metal quantize");
        self.cpu.quantize(spec).expect("cpu quantize");
    }

    fn gradients(&mut self) -> GradientStats {
        let gpu = self.gpu.compute_gradients().expect("metal gradients");
        let cpu = self.cpu.compute_gradients().expect("cpu gradients");
        assert_eq!(gpu, cpu, "gradient maxima");
        gpu
    }
}

fn load_pair<'a>(train: &BinnedDataset, valid: &[&BinnedDataset], setup: &ObjectiveSetup) -> Option<Pair<'a>> {
    let mut gpu = metal_backend("metal_backend")?;
    let mut cpu = CpuBackend::new();
    gpu.load(train, valid, setup).expect("metal load");
    cpu.load(train, valid, setup).expect("cpu load");
    Some(Pair {
        cpu,
        gpu,
        total_bins: train.total_bins,
    })
}

fn regression_data(rng: &mut Rng, rows: usize, cols: usize) -> (BinnedDataset, BinnedDataset) {
    let raw = raw_data(rng, rows, cols);
    let raw_valid = raw_data(rng, rows / 3, cols);
    let params = Params::default();
    let train = construct(&input(&raw, &raw.label, true, None), &params);
    let valid = construct_valid(&input(&raw_valid, &raw_valid.label, true, None), &train);
    (train, valid)
}

#[test]
fn scores_and_l2_gradients_are_bit_exact() {
    let mut rng = Rng(1);
    let (train, valid) = regression_data(&mut rng, 20_000, 23);
    let Some(mut pair) = load_pair(&train, &[&valid], &ObjectiveSetup::Regression) else {
        return;
    };
    for dataset in 0..2 {
        pair.gpu.init_scores(dataset, 0.25).expect("metal init");
        pair.cpu.init_scores(dataset, 0.25).expect("cpu init");
    }
    for (index, leaves) in [1, 2, 7, 31, 53].into_iter().enumerate() {
        let tree = random_tree(&mut rng, &train, leaves);
        pair.gpu.add_tree(&tree).expect("metal add_tree");
        pair.cpu.add_tree(&tree).expect("cpu add_tree");
        for dataset in 0..2 {
            let gpu = pair.gpu.scores(dataset).expect("metal scores");
            let cpu = pair.cpu.scores(dataset).expect("cpu scores");
            assert!(
                gpu == cpu,
                "tree {index} ({leaves} leaves): scores of dataset {dataset}"
            );
        }
    }
    pair.gradients();
    let (grad, hess) = pair.gpu.gradients().expect("metal gradient readback");
    let scores = pair.cpu.scores(0).expect("cpu scores");
    let mut want_grad = vec![0.0; train.num_rows];
    let mut want_hess = vec![0.0; train.num_rows];
    let weight = train.weight.as_deref();
    objective::gradients(
        &ObjectiveSetup::Regression,
        &scores,
        &train.label,
        weight,
        None,
        &mut want_grad,
        &mut want_hess,
    );
    assert!(grad == want_grad, "weighted L2 gradients");
    assert!(hess == want_hess, "weighted L2 hessians");
}

/// Grows a random tree, checking both backends agree on every split and every histogram. Odd steps go through
/// `partition_and_histogram` (the fused Metal path against the CPU's partition-then-histogram), alternating sides.
fn grow_and_compare(
    pair: &mut Pair<'_>,
    rng: &mut Rng,
    train: &BinnedDataset,
    bag_count: u32,
    splits: usize,
) -> Vec<LeafRange> {
    let features = non_trivial_features(train);
    let mut leaves = vec![LeafRange {
        begin: 0,
        count: bag_count,
    }];
    pair.hist_equal("root", leaves[0]);
    for step in 0..splits {
        let index = rng.below(leaves.len() as u64) as usize;
        let leaf = leaves[index];
        let feature = features[rng.below(features.len() as u64) as usize];
        let split = random_split(rng, train, feature);
        let (left_gpu, left_cpu) = if step % 2 == 1 {
            let side = if step % 4 == 1 { Side::Left } else { Side::Right };
            pair.fused(&format!("split {step} fused {side:?}"), leaf, &split, side)
        } else {
            (
                pair.gpu.partition(leaf, &split).expect("metal partition"),
                pair.cpu.partition(leaf, &split).expect("cpu partition"),
            )
        };
        assert_eq!(left_gpu, left_cpu, "split {step}: left count");
        let left = LeafRange {
            begin: leaf.begin,
            count: left_gpu,
        };
        let right = LeafRange {
            begin: leaf.begin + left_gpu,
            count: leaf.count - left_gpu,
        };
        pair.hist_equal(&format!("split {step} left"), left);
        pair.hist_equal(&format!("split {step} right"), right);
        // Histograms of ranges that are not leaves differ unless the partition order is the same.
        for probe in [left.count / 3, left.count / 2 + right.count / 3] {
            pair.hist_equal(
                &format!("split {step} order probe"),
                LeafRange {
                    begin: leaf.begin + probe / 2,
                    count: probe.min(leaf.count - probe / 2),
                },
            );
        }
        leaves[index] = left;
        leaves.push(right);
    }
    leaves
}

#[test]
fn quantized_histograms_partitions_and_leaf_sums_are_bit_exact() {
    let mut rng = Rng(2);
    let (train, valid) = regression_data(&mut rng, 30_000, 41);
    let Some(mut pair) = load_pair(&train, &[&valid], &ObjectiveSetup::Regression) else {
        return;
    };
    for dataset in 0..2 {
        pair.gpu.init_scores(dataset, -0.5).expect("metal init");
        pair.cpu.init_scores(dataset, -0.5).expect("cpu init");
    }
    let tree = random_tree(&mut rng, &train, 15);
    pair.gpu.add_tree(&tree).expect("metal add_tree");
    pair.cpu.add_tree(&tree).expect("cpu add_tree");
    let stats = pair.gradients();
    let cases = [
        (None, 4, true, BagSpec::All),
        (
            None,
            16,
            false,
            BagSpec::Bernoulli {
                seed: 7,
                round: 3,
                fraction: 0.7,
            },
        ),
        (
            Some(1.0),
            32,
            true,
            BagSpec::Bernoulli {
                seed: 11,
                round: 0,
                fraction: 0.35,
            },
        ),
    ];
    for (iteration, (constant, bins, stochastic, bag)) in cases.into_iter().enumerate() {
        let spec = QuantSpec::new(
            stats.max_abs_grad,
            stats.max_hess,
            constant,
            bins,
            stochastic,
            5,
            iteration as u32,
        );
        pair.quantize(&spec);
        let count_gpu = pair.gpu.reset_partition(&bag).expect("metal bag");
        let count_cpu = pair.cpu.reset_partition(&bag).expect("cpu bag");
        assert_eq!(count_gpu, count_cpu, "case {iteration}: bag count");
        let mut leaves = grow_and_compare(&mut pair, &mut rng, &train, count_gpu, 12);
        leaves.push(LeafRange { begin: 0, count: 0 });
        let grad_scale = quantize::fixed_point_scale(stats.max_abs_grad);
        let hess_scale = quantize::fixed_point_scale(stats.max_hess);
        let gpu = pair
            .gpu
            .leaf_sums(&leaves, grad_scale, hess_scale)
            .expect("metal leaf sums");
        let cpu = pair
            .cpu
            .leaf_sums(&leaves, grad_scale, hess_scale)
            .expect("cpu leaf sums");
        assert_eq!(gpu, cpu, "case {iteration}: leaf sums");
    }
}

/// With 16384 quantization bins a packed histogram holds one row per threadgroup and a launch at most
/// `i32::MAX / 16384` rows, so this leaf takes several launches summed on the host.
#[test]
fn fine_quantization_spans_several_launches() {
    let mut rng = Rng(8);
    let (train, valid) = regression_data(&mut rng, 300_000, 5);
    let Some(mut pair) = load_pair(&train, &[&valid], &ObjectiveSetup::Regression) else {
        return;
    };
    pair.gpu.init_scores(0, 0.3).expect("metal init");
    pair.cpu.init_scores(0, 0.3).expect("cpu init");
    let stats = pair.gradients();
    let spec = QuantSpec::new(stats.max_abs_grad, stats.max_hess, None, 16_384, false, 0, 0);
    pair.quantize(&spec);
    let rows = pair.gpu.reset_partition(&BagSpec::All).expect("metal partition reset");
    assert_eq!(
        rows,
        pair.cpu.reset_partition(&BagSpec::All).expect("cpu partition reset")
    );
    pair.hist_equal("every row", LeafRange { begin: 0, count: rows });
    pair.hist_equal(
        "middle rows",
        LeafRange {
            begin: 1_000,
            count: rows - 2_000,
        },
    );
}

#[test]
fn float_histograms_match_to_f32_rounding() {
    let mut rng = Rng(3);
    let (train, valid) = regression_data(&mut rng, 40_000, 37);
    let Some(mut pair) = load_pair(&train, &[&valid], &ObjectiveSetup::Regression) else {
        return;
    };
    pair.gpu.init_scores(0, 0.1).expect("metal init");
    pair.cpu.init_scores(0, 0.1).expect("cpu init");
    pair.gradients();
    let (grad, hess) = pair.gpu.gradients().expect("metal gradient readback");
    let bag = BagSpec::Bernoulli {
        seed: 3,
        round: 1,
        fraction: 0.8,
    };
    let count = pair.gpu.reset_partition(&bag).expect("metal bag");
    assert_eq!(count, pair.cpu.reset_partition(&bag).expect("cpu bag"), "bag count");
    let rows: Vec<u32> = (0..train.num_rows as u32)
        .filter(|&row| quantize::in_bag(3, 1, u64::from(row), 0.8))
        .collect();
    let leaf = LeafRange { begin: 0, count };
    let gpu = hist(&mut pair.gpu, leaf, false, train.total_bins);
    let cpu = hist(&mut pair.cpu, leaf, false, train.total_bins);
    assert_float_hist_close("bagged root", &gpu, &cpu, &abs_hist(&train, &rows, &grad, &hess), 1e-4);
    let small = LeafRange {
        begin: count / 4,
        count: 777,
    };
    let rows_small = &rows[small.begin as usize..(small.begin + small.count) as usize];
    let gpu = hist(&mut pair.gpu, small, false, train.total_bins);
    let cpu = hist(&mut pair.cpu, small, false, train.total_bins);
    assert_float_hist_close(
        "small range",
        &gpu,
        &cpu,
        &abs_hist(&train, rows_small, &grad, &hess),
        1e-4,
    );
    let feature = non_trivial_features(&train)[2];
    let split = random_split(&mut rng, &train, feature);
    let mut gpu = vec![0.0; 2 * train.total_bins];
    let mut cpu = vec![0.0; 2 * train.total_bins];
    let left_gpu = pair
        .gpu
        .partition_and_histogram(leaf, &split, Side::Right, false, &mut gpu)
        .expect("metal partition and histogram");
    let left_cpu = pair
        .cpu
        .partition_and_histogram(leaf, &split, Side::Right, false, &mut cpu)
        .expect("cpu partition and histogram");
    assert_eq!(left_gpu, left_cpu, "fused float split: left count");
    // The partition is stable, so the right child holds the bagged rows going right, in row order.
    let right_rows: Vec<u32> = rows
        .iter()
        .copied()
        .filter(|&row| !split.goes_left(train.bin(feature, row as usize)))
        .collect();
    assert_float_hist_close(
        "fused right child",
        &gpu,
        &cpu,
        &abs_hist(&train, &right_rows, &grad, &hess),
        1e-4,
    );
}

#[test]
fn binary_gradients_match_to_f32_rounding() {
    let mut rng = Rng(4);
    let raw = raw_data(&mut rng, 25_000, 11);
    let label: Vec<f32> = raw.label.iter().map(|&y| if y > 0.2 { 1.0 } else { 0.0 }).collect();
    let train = construct(&input(&raw, &label, true, None), &Params::default());
    let setup = ObjectiveSetup::Binary { sigmoid: 1.3 };
    let Some(mut pair) = load_pair(&train, &[], &setup) else {
        return;
    };
    pair.gpu.init_scores(0, -0.2).expect("metal init");
    pair.cpu.init_scores(0, -0.2).expect("cpu init");
    for _ in 0..3 {
        let tree = random_tree(&mut rng, &train, 20);
        pair.gpu.add_tree(&tree).expect("metal add_tree");
        pair.cpu.add_tree(&tree).expect("cpu add_tree");
    }
    let gpu_stats = pair.gpu.compute_gradients().expect("metal gradients");
    let cpu_stats = pair.cpu.compute_gradients().expect("cpu gradients");
    let (grad, hess) = pair.gpu.gradients().expect("metal gradient readback");
    let scores = pair.cpu.scores(0).expect("cpu scores");
    let mut want_grad = vec![0.0; train.num_rows];
    let mut want_hess = vec![0.0; train.num_rows];
    objective::gradients(
        &setup,
        &scores,
        &train.label,
        train.weight.as_deref(),
        None,
        &mut want_grad,
        &mut want_hess,
    );
    assert_close("binary gradients", &grad, &want_grad, 1e-5, 1e-7);
    assert_close("binary hessians", &hess, &want_hess, 1e-5, 1e-7);
    assert!(
        (gpu_stats.max_abs_grad - cpu_stats.max_abs_grad).abs() <= 1e-5 * cpu_stats.max_abs_grad,
        "max |g|"
    );
    assert!(
        (gpu_stats.max_hess - cpu_stats.max_hess).abs() <= 1e-5 * cpu_stats.max_hess,
        "max |h|"
    );
}

/// Query sizes covering every lambdarank kernel class and the host fallback for queries over 1024 rows.
fn query_sizes(rng: &mut Rng, rows: usize) -> Vec<u32> {
    let mut sizes = vec![1, 2, 64, 65, 256, 257, 1024, 1025, 2100];
    let mut total: usize = sizes.iter().map(|&s| s as usize).sum();
    while total < rows {
        let size = match rng.below(10) {
            0 => 100 + rng.below(400),
            1 => 1 + rng.below(3),
            _ => 5 + rng.below(60),
        } as usize;
        let size = size.min(rows - total);
        sizes.push(size as u32);
        total += size;
    }
    sizes
}

/// Features, noisy integer labels in `0..=4` and query sizes for lambdarank.
fn rank_raw(rng: &mut Rng, rows: usize) -> (RawData, Vec<f32>, Vec<u32>) {
    let raw = raw_data(rng, rows, 9);
    let label: Vec<f32> = raw
        .label
        .iter()
        .map(|&y| {
            ((f64::from(y) + 2.0 * rng.normal()) * 1.2 + 2.0)
                .round()
                .clamp(0.0, 4.0) as f32
        })
        .collect();
    let sizes = query_sizes(rng, rows);
    (raw, label, sizes)
}

fn rank_params() -> Params {
    Params {
        objective: Objective::LambdaRank,
        ..Params::default()
    }
}

fn rank_data(rng: &mut Rng, rows: usize, weighted: bool) -> (BinnedDataset, Params) {
    let (raw, label, sizes) = rank_raw(rng, rows);
    let params = rank_params();
    let train = construct(&input(&raw, &label, weighted, Some(&sizes)), &params);
    (train, params)
}

fn compare_lambdarank(case: &str, rng: &mut Rng, weighted: bool, mutate: impl Fn(&mut Params)) {
    let (train, mut params) = rank_data(rng, 30_000, weighted);
    mutate(&mut params);
    let setup = objective::setup(&params, &train).expect("lambdarank setup");
    let Some(mut pair) = load_pair(&train, &[], &setup) else {
        return;
    };
    pair.gpu.init_scores(0, 0.0).expect("metal init");
    pair.cpu.init_scores(0, 0.0).expect("cpu init");
    // Few leaves: many rows share a score, exercising the tie order (row ascending).
    for leaves in [4, 9, 30] {
        let tree = random_tree(rng, &train, leaves);
        pair.gpu.add_tree(&tree).expect("metal add_tree");
        pair.cpu.add_tree(&tree).expect("cpu add_tree");
    }
    pair.gpu.compute_gradients().expect("metal gradients");
    let (grad, hess) = pair.gpu.gradients().expect("metal gradient readback");
    let scores = pair.cpu.scores(0).expect("cpu scores");
    let mut want_grad = vec![0.0; train.num_rows];
    let mut want_hess = vec![0.0; train.num_rows];
    objective::gradients(
        &setup,
        &scores,
        &train.label,
        train.weight.as_deref(),
        train.query_boundaries.as_deref(),
        &mut want_grad,
        &mut want_hess,
    );
    assert_close(&format!("{case}: lambdas"), &grad, &want_grad, 1e-5, 1e-7);
    assert_close(&format!("{case}: hessians"), &hess, &want_hess, 1e-5, 1e-7);
}

#[test]
fn lambdarank_gradients_match_to_f32_rounding() {
    let mut rng = Rng(5);
    compare_lambdarank("default", &mut rng, false, |_| {});
    compare_lambdarank("weighted, no norm, truncation 10", &mut rng, true, |p| {
        p.lambdarank_norm = false;
        p.lambdarank_truncation_level = 10;
        p.sigmoid = 2.0;
    });
    compare_lambdarank("truncation past every query", &mut rng, false, |p| {
        p.lambdarank_truncation_level = 3000;
    });
}

fn train_with(
    params: &Params,
    train: &BinnedDataset,
    valid: &BinnedDataset,
    backend: &mut dyn Backend,
) -> (String, f64) {
    let mut last_eval = f64::NAN;
    let mut on_iteration = |report: &silvic_core::IterationReport<'_>| {
        if let Some(eval) = report.evals.first() {
            last_eval = eval.value;
        }
        Ok(())
    };
    let model = silvic_core::train(
        params,
        train,
        &[("valid", valid)],
        backend,
        &mut LocalCollective,
        &mut on_iteration,
    )
    .expect("train");
    (model.to_lightgbm_string(), last_eval)
}

#[test]
fn l2_training_gives_the_cpu_model() {
    let mut rng = Rng(6);
    let (train, valid) = regression_data(&mut rng, 20_000, 15);
    let Some(mut gpu) = metal_backend("l2_training_gives_the_cpu_model") else {
        return;
    };
    for (case, quantized, bagging) in [("quantized", true, true), ("float", false, false)] {
        let params = Params {
            num_iterations: 30,
            num_leaves: 31,
            use_quantized_grad: quantized,
            bagging_fraction: if bagging { 0.8 } else { 1.0 },
            bagging_freq: usize::from(bagging) * 2,
            metric: vec![Metric::L2],
            ..Params::default()
        };
        let (cpu_model, cpu_l2) = train_with(&params, &train, &valid, &mut CpuBackend::new());
        let (gpu_model, gpu_l2) = train_with(&params, &train, &valid, &mut gpu);
        assert_eq!(cpu_model.matches("Tree=").count(), 30, "{case}: trees in the cpu model");
        if quantized {
            assert!(gpu_model == cpu_model, "{case}: model text differs");
        } else {
            // f32 atomics sum in arbitrary order: close, not identical.
            assert!(
                (gpu_l2 - cpu_l2).abs() <= 0.01 * cpu_l2,
                "{case}: valid l2 cpu {cpu_l2}, metal {gpu_l2}"
            );
        }
    }
}

#[test]
fn lambdarank_training_matches_cpu_ndcg() {
    let mut rng = Rng(7);
    let (train, params) = rank_data(&mut rng, 30_000, false);
    let (raw_valid, label_valid, sizes_valid) = rank_raw(&mut rng, 10_000);
    let valid = construct_valid(&input(&raw_valid, &label_valid, false, Some(&sizes_valid)), &train);
    let Some(mut gpu) = metal_backend("lambdarank_training_matches_cpu_ndcg") else {
        return;
    };
    let params = Params {
        num_iterations: 40,
        metric: vec![Metric::Ndcg],
        eval_at: vec![10],
        bagging_fraction: 0.8,
        bagging_freq: 3,
        ..params
    };
    let (_, cpu_ndcg) = train_with(&params, &train, &valid, &mut CpuBackend::new());
    let (_, gpu_ndcg) = train_with(&params, &train, &valid, &mut gpu);
    eprintln!("ndcg@10: cpu {cpu_ndcg}, metal {gpu_ndcg}");
    assert!(
        (cpu_ndcg - gpu_ndcg).abs() <= 0.005,
        "ndcg@10: cpu {cpu_ndcg}, metal {gpu_ndcg}"
    );
}

/// Trains `params.num_iterations` trees on Metal from `init`; returns the model and the model at `checkpoint`.
fn train_metal_from(
    params: &Params,
    init: Option<&silvic_core::Model>,
    train: &BinnedDataset,
    valid: &BinnedDataset,
    backend: &mut dyn Backend,
    checkpoint: usize,
) -> (String, Option<silvic_core::Model>) {
    let mut saved = None;
    let mut on_iteration = |report: &silvic_core::IterationReport<'_>| {
        if report.iteration == checkpoint {
            saved = Some(report.model());
        }
        Ok(())
    };
    let model = silvic_core::train_from(
        params,
        init,
        train,
        &[("valid", valid)],
        backend,
        &mut LocalCollective,
        &mut on_iteration,
    )
    .expect("train");
    (model.to_lightgbm_string(), saved)
}

#[test]
fn resumed_training_matches_uninterrupted_training() {
    let mut rng = Rng(9);
    let (l2_train, l2_valid) = regression_data(&mut rng, 20_000, 15);
    let (rank_train, rank_params) = rank_data(&mut rng, 20_000, false);
    let (raw_valid, label_valid, sizes_valid) = rank_raw(&mut rng, 5_000);
    let rank_valid = construct_valid(&input(&raw_valid, &label_valid, false, Some(&sizes_valid)), &rank_train);
    let Some(mut gpu) = metal_backend("resumed_training_matches_uninterrupted_training") else {
        return;
    };
    let randomized = |params: Params| Params {
        num_iterations: 30,
        bagging_fraction: 0.8,
        bagging_freq: 3,
        feature_fraction: 0.8,
        ..params
    };
    let cases = [
        ("l2", randomized(Params::default()), &l2_train, &l2_valid),
        ("lambdarank", randomized(rank_params), &rank_train, &rank_valid),
    ];
    for (case, params, train, valid) in cases {
        let (full, _) = train_metal_from(&params, None, train, valid, &mut gpu, usize::MAX);
        let first_params = Params {
            num_iterations: 12,
            ..params.clone()
        };
        let (_, checkpoint) = train_metal_from(&first_params, None, train, valid, &mut gpu, 11);
        let text = checkpoint.expect("checkpoint").to_lightgbm_string();
        let init = silvic_core::Model::from_lightgbm_string(&text).expect("checkpoint parses");
        let rest_params = Params {
            num_iterations: 18,
            ..params.clone()
        };
        let (resumed, _) = train_metal_from(&rest_params, Some(&init), train, valid, &mut gpu, usize::MAX);
        assert_eq!(full.matches("Tree=").count(), 30, "{case}: trees");
        assert!(
            resumed == full,
            "{case}: resumed model text differs from the uninterrupted one"
        );
        if case == "l2" {
            // L2 trains the same trees on both backends, so the CPU backend resumes the Metal checkpoint to the same
            // model. (Lambdarank gradients only agree to f32 rounding.)
            let (cpu_resumed, _) = train_metal_from(
                &rest_params,
                Some(&init),
                train,
                valid,
                &mut CpuBackend::new(),
                usize::MAX,
            );
            assert!(cpu_resumed == full, "{case}: cpu resumed from the metal checkpoint");
        }
    }
}
