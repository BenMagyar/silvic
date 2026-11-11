//! Histogram construction on 10M rows x 136 features of 255 bins, Metal against `silvic_core::CpuBackend`: the root
//! (every row) and a 10% leaf (rows scattered over the whole range), quantized (the default) and float.

#[cfg(target_os = "macos")]
mod bench {
    use criterion::{Criterion, Throughput};
    use rayon::prelude::*;
    use silvic_core::backend::{BagSpec, BinSplit, LeafRange, ObjectiveSetup};
    use silvic_core::data::{BinMapper, FeatureGroup, FeatureMeta, GROUP_BIN_BUDGET};
    use silvic_core::quantize::QuantSpec;
    use silvic_core::{Backend, BinnedDataset, CpuBackend};
    use silvic_gpu::metal;

    const ROWS: usize = 10_000_000;
    const FEATURES: usize = 136;
    const BINS: u32 = 255;
    /// Bins `0..=LEAF_THRESHOLD` of a uniform feature hold about 10% of the rows.
    const LEAF_THRESHOLD: u32 = 24;

    fn mix(mut x: u64) -> u64 {
        x = (x ^ (x >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        x = (x ^ (x >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        x ^ (x >> 31)
    }

    /// Every fourth feature has 70% of its rows in bin 0 (sparse-ish columns hit one bin hard); the rest are
    /// uniform.
    fn bin_of(row: usize, feature: usize) -> u8 {
        let h = mix(((row as u64) << 8) | feature as u64);
        if feature.is_multiple_of(4) && h % 10 < 7 {
            0
        } else {
            ((h >> 8) % u64::from(BINS)) as u8
        }
    }

    fn feature_meta(feature: usize, group: usize, column: usize, hist_offset: u32) -> FeatureMeta {
        let upper_bounds = (0..BINS)
            .map(|b| if b + 1 == BINS { f64::INFINITY } else { f64::from(b) })
            .collect();
        FeatureMeta {
            name: format!("Column_{feature}"),
            bins: BinMapper::Numerical {
                upper_bounds,
                nan_bin: false,
                min: 0.0,
                max: f64::from(BINS - 1),
            },
            num_bins: BINS,
            group: group as u32,
            column: column as u32,
            hist_offset,
        }
    }

    fn dataset() -> BinnedDataset {
        let per_group = (GROUP_BIN_BUDGET / BINS) as usize;
        let all: Vec<usize> = (0..FEATURES).collect();
        let mut features = Vec::with_capacity(FEATURES);
        let mut groups = Vec::new();
        let mut hist_offset = 0;
        for (group, members) in all.chunks(per_group).enumerate() {
            let stride = members.len().div_ceil(4) * 4;
            let group_offset = hist_offset;
            for (column, &feature) in members.iter().enumerate() {
                features.push(feature_meta(feature, group, column, hist_offset));
                hist_offset += BINS;
            }
            let mut data = vec![0u8; ROWS * stride];
            data.par_chunks_mut(stride).enumerate().for_each(|(row, bins)| {
                for (column, &feature) in members.iter().enumerate() {
                    bins[column] = bin_of(row, feature);
                }
            });
            groups.push(FeatureGroup {
                features: members.to_vec(),
                row_stride: stride as u32,
                hist_offset: group_offset,
                total_bins: members.len() as u32 * BINS,
                data,
            });
        }
        let label = (0..ROWS)
            .into_par_iter()
            .map(|row| (mix(row as u64 ^ 0x5eed) % 1000) as f32 / 500.0 - 1.0)
            .collect();
        BinnedDataset {
            num_rows: ROWS,
            row_offset: 0,
            features,
            groups,
            total_bins: hist_offset as usize,
            label,
            weight: None,
            query_boundaries: None,
        }
    }

    /// Loads `data`, quantizes the L2 gradients of zero scores and splits the root so the left leaf holds ~10% of
    /// the rows; returns (root, leaf).
    fn prepare(backend: &mut dyn Backend, data: &BinnedDataset) -> (LeafRange, LeafRange) {
        backend.load(data, &[], &ObjectiveSetup::Regression).expect("load");
        backend.init_scores(0, 0.0).expect("init scores");
        let stats = backend.compute_gradients().expect("gradients");
        let spec = QuantSpec::new(stats.max_abs_grad, stats.max_hess, Some(1.0), 4, true, 1, 0);
        backend.quantize(&spec).expect("quantize");
        let rows = backend.reset_partition(&BagSpec::All).expect("partition reset");
        let mut left_bins = [0u32; 8];
        for bin in 0..=LEAF_THRESHOLD {
            left_bins[bin as usize / 32] |= 1 << (bin % 32);
        }
        let split = BinSplit {
            group: 0,
            column: 1,
            left_bins,
        };
        let root = LeafRange { begin: 0, count: rows };
        let left = backend.partition(root, &split).expect("partition");
        (root, LeafRange { begin: 0, count: left })
    }

    fn histogram(backend: &mut dyn Backend, leaf: LeafRange, quantized: bool, out: &mut [f64]) {
        backend.histogram(leaf, quantized, out).expect("histogram");
    }

    pub fn histograms(c: &mut Criterion) {
        if !metal::is_available() {
            eprintln!("histogram bench: skipped, no Metal device on this machine");
            return;
        }
        let data = dataset();
        let mut gpu = metal::backend(None).expect("metal backend");
        let mut cpu = CpuBackend::borrowing(&data, &[]);
        let (root, leaf) = prepare(&mut gpu, &data);
        assert_eq!((root, leaf), prepare(&mut cpu, &data), "metal and cpu partitions");
        let mut gpu_out = vec![0.0; 2 * data.total_bins];
        let mut cpu_out = vec![0.0; 2 * data.total_bins];
        for range in [root, leaf] {
            histogram(&mut gpu, range, true, &mut gpu_out);
            histogram(&mut cpu, range, true, &mut cpu_out);
            assert!(gpu_out == cpu_out, "metal and cpu quantized histograms of {range:?}");
        }
        let mut group = c.benchmark_group("histogram");
        group.sample_size(10);
        for (name, range) in [("root", root), ("leaf_10pct", leaf)] {
            group.throughput(Throughput::Elements(u64::from(range.count) * FEATURES as u64));
            for quantized in [true, false] {
                let mode = if quantized { "quantized" } else { "float" };
                group.bench_function(format!("metal/{mode}/{name}"), |b| {
                    b.iter(|| histogram(&mut gpu, range, quantized, &mut gpu_out))
                });
                group.bench_function(format!("cpu/{mode}/{name}"), |b| {
                    b.iter(|| histogram(&mut cpu, range, quantized, &mut cpu_out))
                });
            }
        }
        group.finish();
    }
}

#[cfg(target_os = "macos")]
criterion::criterion_group!(benches, bench::histograms);
#[cfg(target_os = "macos")]
criterion::criterion_main!(benches);

#[cfg(not(target_os = "macos"))]
fn main() {}
