//! [`Predictor`]: a compiled model and the kernel that scores it, with `Model::predict`'s arguments and results.

use std::fmt;
use std::str::FromStr;

use rayon::prelude::*;
use silvic_core::data::DenseMatrix;
use silvic_core::model::{Model, ModelObjective};
use silvic_core::{Result, SilvicError};

use crate::compile::{Forest, Plan};
use crate::kernel::{portable, score_rows};

/// The vector unit a [`Predictor`] scores with. Every kernel gives the same predictions bit for bit.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Kernel {
    /// Plain Rust (no intrinsics), for any CPU: 32 rows per block.
    Portable,
    /// aarch64 NEON: 64 rows per block.
    Neon,
    /// x86-64 AVX2: 64 rows per block.
    Avx2,
    /// x86-64 AVX-512F and AVX-512BW: 64 rows per block. [`Kernel::detect`] does not pick it; ask for it by name.
    Avx512,
}

impl Kernel {
    pub const ALL: [Kernel; 4] = [Kernel::Portable, Kernel::Neon, Kernel::Avx2, Kernel::Avx512];

    /// The kernel [`Predictor::new`] uses: NEON on aarch64, AVX2 on x86-64 CPUs that have it, else portable.
    pub fn detect() -> Kernel {
        [Kernel::Neon, Kernel::Avx2]
            .into_iter()
            .find(|kernel| kernel.is_available())
            .unwrap_or(Kernel::Portable)
    }

    /// Whether this CPU runs the kernel.
    pub fn is_available(self) -> bool {
        match self {
            Kernel::Portable => true,
            Kernel::Neon => cfg!(target_arch = "aarch64"),
            #[cfg(target_arch = "x86_64")]
            Kernel::Avx2 => crate::kernel::avx2::Avx2::detect().is_some(),
            #[cfg(target_arch = "x86_64")]
            Kernel::Avx512 => crate::kernel::avx512::Avx512::detect().is_some(),
            #[cfg(not(target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx512 => false,
        }
    }

    /// Every kernel this CPU runs.
    pub fn available() -> Vec<Kernel> {
        Kernel::ALL.into_iter().filter(|kernel| kernel.is_available()).collect()
    }

    pub fn name(self) -> &'static str {
        match self {
            Kernel::Portable => "portable",
            Kernel::Neon => "neon",
            Kernel::Avx2 => "avx2",
            Kernel::Avx512 => "avx512",
        }
    }
}

impl fmt::Display for Kernel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

impl FromStr for Kernel {
    type Err = SilvicError;

    fn from_str(name: &str) -> Result<Kernel> {
        Kernel::ALL
            .into_iter()
            .find(|kernel| kernel.name() == name)
            .ok_or_else(|| SilvicError::config(format!("unknown kernel `{name}` (portable, neon, avx2, avx512)")))
    }
}

/// A model compiled for batch scoring. Predictions equal [`Model::predict`]'s bit for bit.
#[derive(Debug, Clone)]
pub struct Predictor {
    forest: Forest,
    objective: ModelObjective,
    kernel: Kernel,
}

impl Predictor {
    /// Compiles `model` for the kernel [`Kernel::detect`] picks. Fails on a tree `Tree::GetLeaf` could not walk (a
    /// child or bitset index out of range, a node reached twice, a feature past the model's).
    pub fn new(model: &Model) -> Result<Predictor> {
        Predictor::with_kernel(model, Kernel::detect())
    }

    /// Compiles `model` for `kernel`; fails if this CPU does not run it.
    pub fn with_kernel(model: &Model, kernel: Kernel) -> Result<Predictor> {
        if !kernel.is_available() {
            return Err(SilvicError::unsupported(format!(
                "this CPU does not run the {kernel} kernel"
            )));
        }
        Ok(Predictor {
            forest: Forest::compile(model)?,
            objective: model.objective.clone(),
            kernel,
        })
    }

    pub fn kernel(&self) -> Kernel {
        self.kernel
    }

    pub fn num_trees(&self) -> usize {
        self.forest.plans.len()
    }

    pub fn num_features(&self) -> usize {
        self.forest.num_features
    }

    /// Trees scored by walking them row by row instead of with leaf bitvectors: those with more than 256 leaves, a
    /// NaN threshold, or a categorical split on a feature with more than 64 distinct category behaviours.
    pub fn num_walked_trees(&self) -> usize {
        self.forest.walked.len()
    }

    /// Writes the raw score (sum of leaf values, before the objective's transform) of every row of `rows` to `out`,
    /// on the calling thread, using the first `num_trees` trees (all when `None`).
    ///
    /// # Panics
    /// When `rows` has fewer columns than the model has features, or `out` is not one value per row.
    pub fn predict_raw(&self, rows: &DenseMatrix<'_>, num_trees: Option<usize>, out: &mut [f64]) {
        self.check(rows);
        assert_eq!(out.len(), rows.rows, "one output per row");
        self.score(rows, 0, self.plans(num_trees), out);
    }

    /// Predictions for every row, as [`Model::predict`] makes them: `raw_score = false` applies the objective's
    /// transform. `num_threads` 1 scores on the calling thread; more split the rows over that many tasks on rayon's
    /// global pool, and 0 over all of its threads.
    ///
    /// # Panics
    /// When `rows` has fewer columns than the model has features.
    pub fn predict(
        &self,
        rows: &DenseMatrix<'_>,
        raw_score: bool,
        num_trees: Option<usize>,
        num_threads: usize,
    ) -> Vec<f64> {
        self.check(rows);
        let plans = self.plans(num_trees);
        let mut out = vec![0.0; rows.rows];
        let threads = if num_threads == 0 {
            rayon::current_num_threads()
        } else {
            num_threads
        };
        // Whole blocks per task, so no task pads a block another one also scores.
        let block = 64;
        let per_task = rows.rows.div_ceil(threads.max(1)).div_ceil(block) * block;
        if threads <= 1 || per_task >= rows.rows {
            self.score(rows, 0, plans, &mut out);
        } else {
            out.par_chunks_mut(per_task)
                .enumerate()
                .for_each(|(task, out)| self.score(rows, task * per_task, plans, out));
        }
        if !raw_score {
            for value in &mut out {
                *value = self.convert_output(*value);
            }
        }
        out
    }

    fn check(&self, rows: &DenseMatrix<'_>) {
        assert!(
            rows.cols >= self.num_features(),
            "the matrix has {} columns, the model reads {} features",
            rows.cols,
            self.num_features()
        );
    }

    fn plans(&self, num_trees: Option<usize>) -> &[Plan] {
        let plans = &self.forest.plans;
        &plans[..num_trees.map_or(plans.len(), |n| n.min(plans.len()))]
    }

    fn score(&self, rows: &DenseMatrix<'_>, first_row: usize, plans: &[Plan], out: &mut [f64]) {
        let forest = &self.forest;
        match self.kernel {
            #[cfg(target_arch = "aarch64")]
            Kernel::Neon => score_rows(crate::kernel::neon::Neon, forest, plans, rows, first_row, out),
            #[cfg(target_arch = "x86_64")]
            Kernel::Avx2 => {
                let kernel = crate::kernel::avx2::Avx2::detect().expect("checked when the predictor was built");
                score_rows(kernel, forest, plans, rows, first_row, out);
            }
            #[cfg(target_arch = "x86_64")]
            Kernel::Avx512 => {
                let kernel = crate::kernel::avx512::Avx512::detect().expect("checked when the predictor was built");
                score_rows(kernel, forest, plans, rows, first_row, out);
            }
            _ => score_rows(portable::Portable, forest, plans, rows, first_row, out),
        }
    }

    /// The objective's `ConvertOutput`, as `Model::predict` applies it.
    fn convert_output(&self, raw: f64) -> f64 {
        match self.objective {
            ModelObjective::Binary { sigmoid } => 1.0 / (1.0 + (-sigmoid * raw).exp()),
            ModelObjective::Regression | ModelObjective::LambdaRank => raw,
        }
    }
}
