//! Random forests and rows for the equality tests and the benchmark: trees grown leaf-wise like LightGBM's, with
//! every missing-value handling, categorical bitsets, and rows that hit thresholds, their neighbouring doubles, NaN,
//! ±0, ±`kZeroThreshold` and ±inf.

#![allow(dead_code)]

use silvic_core::model::{Model, ModelObjective};
use silvic_core::tree::{MissingType, SplitChildren, Tree};

/// LightGBM `kZeroThreshold`.
pub const ZERO_THRESHOLD: f64 = 1e-35_f32 as f64;

/// splitmix64.
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Rng {
        Rng(seed)
    }

    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in `0..n`.
    pub fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }

    /// Uniform in `[0, 1)`.
    pub fn unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    pub fn chance(&mut self, p: f64) -> bool {
        self.unit() < p
    }

    pub fn normal(&mut self) -> f64 {
        let u = self.unit().max(f64::MIN_POSITIVE);
        (-2.0 * u.ln()).sqrt() * (std::f64::consts::TAU * self.unit()).cos()
    }
}

/// The shape of a random forest.
#[derive(Debug, Clone)]
pub struct ForestSpec {
    pub trees: usize,
    pub leaves: usize,
    pub max_depth: i32,
    pub features: usize,
    /// Features `features - categorical..` split categorically.
    pub categorical: usize,
    /// Categories `0, stride, 2 * stride, ..` (`categories` of them) of a categorical feature; past 32 makes
    /// multi-word bitsets.
    pub categories: i32,
    pub category_stride: i32,
    /// Distinct thresholds per numeric feature.
    pub thresholds: usize,
    /// Chance that a numeric split's threshold is one of the edge values (0, ±kZeroThreshold, ±inf, ...).
    pub edge_thresholds: f64,
    pub objective: ModelObjective,
}

impl Default for ForestSpec {
    fn default() -> ForestSpec {
        ForestSpec {
            trees: 50,
            leaves: 31,
            max_depth: 11,
            features: 12,
            categorical: 2,
            categories: 20,
            category_stride: 1,
            thresholds: 40,
            edge_thresholds: 0.1,
            objective: ModelObjective::Regression,
        }
    }
}

/// Per numeric feature, the threshold grid splits draw from.
pub fn threshold_grid(rng: &mut Rng, spec: &ForestSpec) -> Vec<Vec<f64>> {
    (0..spec.features)
        .map(|_| {
            let scale = 10f64.powi(rng.below(5) as i32 - 2);
            let mut grid: Vec<f64> = (0..spec.thresholds).map(|_| rng.normal() * scale).collect();
            grid.sort_by(f64::total_cmp);
            grid
        })
        .collect()
}

const EDGE_THRESHOLDS: [f64; 10] = [
    0.0,
    -0.0,
    ZERO_THRESHOLD,
    -ZERO_THRESHOLD,
    1e-300,
    -1e-300,
    f64::INFINITY,
    f64::NEG_INFINITY,
    f64::MAX,
    1.0,
];

fn children(rng: &mut Rng) -> SplitChildren {
    SplitChildren {
        left_value: rng.normal() * 0.1,
        right_value: rng.normal() * 0.1,
        left_count: 10,
        right_count: 10,
        left_weight: 1.0,
        right_weight: 1.0,
        gain: 1.0,
    }
}

fn missing_type(rng: &mut Rng) -> MissingType {
    match rng.below(3) {
        0 => MissingType::None,
        1 => MissingType::Zero,
        _ => MissingType::NaN,
    }
}

/// One tree grown leaf-wise: each split takes a random leaf above `max_depth`.
pub fn random_tree(rng: &mut Rng, spec: &ForestSpec, grid: &[Vec<f64>]) -> Tree {
    let mut tree = Tree::new();
    tree.leaf_value[0] = rng.normal() * 0.1;
    let numeric = spec.features - spec.categorical;
    for _ in 1..spec.leaves {
        let open: Vec<usize> = (0..tree.num_leaves)
            .filter(|&l| tree.leaf_depth[l] < spec.max_depth)
            .collect();
        if open.is_empty() {
            break;
        }
        let leaf = open[rng.below(open.len())];
        let feature = rng.below(spec.features);
        if feature >= numeric {
            let mut categories: Vec<i32> = (0..spec.categories)
                .filter(|_| rng.chance(0.4))
                .map(|c| c * spec.category_stride)
                .collect();
            if categories.is_empty() {
                categories.push(rng.below(spec.categories as usize) as i32 * spec.category_stride);
            }
            tree.split_categorical(leaf, feature, &categories, MissingType::NaN, &children(rng));
        } else {
            let threshold = if rng.chance(spec.edge_thresholds) {
                EDGE_THRESHOLDS[rng.below(EDGE_THRESHOLDS.len())]
            } else {
                grid[feature][rng.below(grid[feature].len())]
            };
            let missing = missing_type(rng);
            let default_left = rng.chance(0.5);
            tree.split_numerical(leaf, feature, threshold, default_left, missing, &children(rng));
        }
    }
    tree
}

pub fn model(trees: Vec<Tree>, features: usize, objective: ModelObjective) -> Model {
    Model {
        objective,
        feature_names: (0..features).map(|i| format!("Column_{i}")).collect(),
        feature_infos: vec!["none".to_string(); features],
        trees,
        parameters: Vec::new(),
        pandas_categorical: None,
    }
}

pub fn random_model(rng: &mut Rng, spec: &ForestSpec) -> (Model, Vec<Vec<f64>>) {
    let grid = threshold_grid(rng, spec);
    let trees = (0..spec.trees).map(|_| random_tree(rng, spec, &grid)).collect();
    (model(trees, spec.features, spec.objective.clone()), grid)
}

/// A value of a numeric feature: often a threshold or one of its neighbouring doubles, or an edge value.
fn numeric_value(rng: &mut Rng, grid: &[f64]) -> f64 {
    let threshold = grid[rng.below(grid.len())];
    match rng.below(12) {
        0 => threshold,
        1 => threshold.next_up(),
        2 => threshold.next_down(),
        3 => f64::NAN,
        4 => [0.0, -0.0][rng.below(2)],
        5 => {
            let edge = [ZERO_THRESHOLD, -ZERO_THRESHOLD][rng.below(2)];
            [edge, edge.next_up(), edge.next_down(), 1e-40, -1e-40][rng.below(5)]
        }
        6 => EDGE_THRESHOLDS[rng.below(EDGE_THRESHOLDS.len())],
        7 => [f64::INFINITY, f64::NEG_INFINITY, f64::MAX, f64::MIN, 1e-300][rng.below(5)],
        _ => {
            let (low, high) = (grid[0], grid[grid.len() - 1]);
            low + (high - low) * (rng.unit() * 1.2 - 0.1)
        }
    }
}

/// A value of a categorical feature: a category, or NaN, negative, fractional, unseen or huge.
fn categorical_value(rng: &mut Rng, spec: &ForestSpec) -> f64 {
    let category = f64::from(rng.below(spec.categories as usize) as i32 * spec.category_stride);
    let past = f64::from(spec.categories * spec.category_stride);
    match rng.below(10) {
        0 => f64::NAN,
        1 => [-1.0, -0.5, -0.0, -1e-40, -3.0][rng.below(5)],
        2 => category + [0.75, 1.0, -1.0][rng.below(3)],
        3 => [past, past + 40.0, 1e10, f64::INFINITY, 2_147_483_648.0][rng.below(5)],
        _ => category,
    }
}

/// `rows` rows of `spec.features` values, row after row.
pub fn random_rows(rng: &mut Rng, spec: &ForestSpec, grid: &[Vec<f64>], rows: usize) -> Vec<f64> {
    let numeric = spec.features - spec.categorical;
    let mut values = Vec::with_capacity(rows * spec.features);
    for _ in 0..rows {
        for (feature, thresholds) in grid.iter().enumerate().take(spec.features) {
            values.push(if feature >= numeric {
                categorical_value(rng, spec)
            } else {
                numeric_value(rng, thresholds)
            });
        }
    }
    values
}
