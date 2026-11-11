//! The boosting loop (LightGBM 3.3.5 `GBDT::Train`/`GBDT::TrainOneIter`, `src/boosting/gbdt.cpp`).
//!
//! Each iteration computes gradients, reduces their maxima over ranks and quantizes them (quantized mode), draws
//! the bag, grows a tree, renews its leaves from exact sums (quantized mode), shrinks it, adds it to every dataset's
//! scores and evaluates the validation sets. As in LightGBM the initial score (`BoostFromAverage`) is added to the
//! scores up front and folded into the first tree as a bias; a first tree that cannot split becomes a constant tree
//! holding it, and any later tree that cannot split ends training without being added.

use std::sync::Arc;

use rayon::ThreadPool;

use crate::backend::{Backend, BagSpec, ObjectiveSetup};
use crate::collective::Collective;
use crate::config::Params;
use crate::data::BinnedDataset;
use crate::error::{Result, SilvicError};
use crate::learner::{GrownTree, Learner, TreeSpec};
use crate::metric::{self, EvalResult};
use crate::model::{Model, ModelObjective};
use crate::objective;
use crate::quantize::QuantSpec;
use crate::split::{self, HistScale, K_EPSILON};
use crate::tree::Tree;

/// What the iteration callback sees after each tree.
#[derive(Debug, Clone, PartialEq)]
pub struct IterationReport<'a> {
    /// Zero-based.
    pub iteration: usize,
    pub evals: &'a [EvalResult],
}

/// Trains `params.num_iterations` trees on `train`, evaluating `valid` after each. Every rank calls this with its own
/// rows; every rank returns the same model. An error from `on_iteration` stops training and is returned.
///
/// With `num_threads > 0`, split finding, metrics and the backend's host work run on a pool of that many threads
/// (handed to the backend through `Backend::set_thread_pool`); otherwise on rayon's global pool.
pub fn train(
    params: &Params,
    train: &BinnedDataset,
    valid: &[(&str, &BinnedDataset)],
    backend: &mut dyn Backend,
    collective: &mut dyn Collective,
    on_iteration: &mut dyn FnMut(&IterationReport<'_>) -> Result<()>,
) -> Result<Model> {
    let pool = thread_pool(params.num_threads)?;
    if let Some(pool) = &pool {
        backend.set_thread_pool(Arc::clone(pool));
    }
    let setup = objective::setup(params, train)?;
    let valid_sets: Vec<&BinnedDataset> = valid.iter().map(|(_, data)| *data).collect();
    backend.load(train, &valid_sets, &setup)?;
    let init_score = install(pool.as_deref(), || {
        objective::init_score(params, train, &mut *collective)
    })?;
    // `GBDT::BoostFromAverage` ignores initial scores within kEpsilon of 0.
    let init_score = if init_score.abs() > K_EPSILON { init_score } else { 0.0 };
    for dataset in 0..=valid.len() {
        backend.init_scores(dataset, init_score)?;
    }
    tracing::debug!(backend = %backend.name(), init_score, "training started");
    let mut trainer = Trainer {
        params,
        train,
        valid,
        learner: Learner::new(params, &train.features, train.total_bins, pool.as_deref()),
        pool: pool.as_deref(),
        constant_hessian: objective::constant_hessian(&setup, train.weight.as_deref()),
        init_score,
        trees: Vec::new(),
    };
    for iteration in 0..params.num_iterations {
        if !trainer.train_one_iter(backend, collective, iteration)? {
            break;
        }
        let evals = trainer.evaluate(backend, collective)?;
        tracing::debug!(
            iteration,
            leaves = trainer.trees.last().map_or(0, |t| t.num_leaves),
            ?evals,
            "tree added"
        );
        on_iteration(&IterationReport {
            iteration,
            evals: &evals,
        })?;
    }
    Ok(trainer.into_model(&setup))
}

fn thread_pool(num_threads: usize) -> Result<Option<Arc<ThreadPool>>> {
    if num_threads == 0 {
        return Ok(None);
    }
    rayon::ThreadPoolBuilder::new()
        .num_threads(num_threads)
        .thread_name(|index| format!("silvic-{index}"))
        .build()
        .map(|pool| Some(Arc::new(pool)))
        .map_err(|e| SilvicError::config(format!("cannot start {num_threads} threads: {e}")))
}

fn install<R: Send>(pool: Option<&ThreadPool>, op: impl FnOnce() -> R + Send) -> R {
    match pool {
        Some(pool) => pool.install(op),
        None => op(),
    }
}

struct Trainer<'a> {
    params: &'a Params,
    train: &'a BinnedDataset,
    valid: &'a [(&'a str, &'a BinnedDataset)],
    learner: Learner<'a>,
    pool: Option<&'a ThreadPool>,
    constant_hessian: Option<f64>,
    init_score: f64,
    trees: Vec<Tree>,
}

impl Trainer<'_> {
    /// `GBDT::TrainOneIter`; returns whether training continues.
    fn train_one_iter(
        &mut self,
        backend: &mut dyn Backend,
        collective: &mut dyn Collective,
        iteration: usize,
    ) -> Result<bool> {
        let params = self.params;
        let stats = backend.compute_gradients()?;
        let mut maxima = [stats.max_abs_grad, stats.max_hess];
        let scale = if params.use_quantized_grad {
            collective.allreduce_max_f64(&mut maxima)?;
            let spec = QuantSpec::new(
                maxima[0],
                maxima[1],
                self.constant_hessian,
                params.num_grad_quant_bins,
                params.stochastic_rounding,
                params.data_random_seed,
                iteration as u32,
            );
            backend.quantize(&spec)?;
            HistScale {
                grad: spec.grad_scale,
                hess: spec.hess_scale,
            }
        } else {
            HistScale::UNIT
        };
        let local_rows = backend.reset_partition(&self.bag(iteration))?;
        let mut root_count = [i64::from(local_rows)];
        collective.allreduce_sum_i64(&mut root_count)?;
        let feature_mask = split::feature_mask(
            &self.train.features,
            params.feature_fraction,
            params.feature_fraction_seed,
            iteration as u32,
        );
        let spec = TreeSpec {
            quantized: params.use_quantized_grad,
            scale,
            feature_mask: &feature_mask,
            root: crate::backend::LeafRange {
                begin: 0,
                count: local_rows,
            },
            root_count: root_count[0],
        };
        let mut grown = self.learner.grow(backend, collective, spec)?;
        if grown.tree.num_leaves == 1 {
            self.single_leaf(iteration);
            return Ok(false);
        }
        if params.use_quantized_grad && params.quant_train_renew_leaf {
            self.learner
                .renew_leaves(backend, collective, &mut grown, maxima[0], maxima[1])?;
        }
        self.add(backend, grown)?;
        Ok(true)
    }

    /// LightGBM bags every `bagging_freq` iterations; the bag of a round is a pure function of the round.
    fn bag(&self, iteration: usize) -> BagSpec {
        let params = self.params;
        if params.bagging_fraction < 1.0 && params.bagging_freq > 0 {
            BagSpec::Bernoulli {
                seed: params.bagging_seed,
                round: (iteration / params.bagging_freq) as u32,
                fraction: params.bagging_fraction,
            }
        } else {
            BagSpec::All
        }
    }

    /// A tree that cannot split: the first one becomes a constant tree with the initial score, as in
    /// `GBDT::TrainOneIter`; a later one is dropped. Either way training ends.
    fn single_leaf(&mut self, iteration: usize) {
        tracing::debug!(iteration, "no split with positive gain; training stops");
        if self.trees.is_empty() {
            let mut tree = Tree::new();
            tree.as_constant(self.init_score);
            self.trees.push(tree);
        }
    }

    /// Shrinkage, the score update (without the bias, already in the scores) and the first tree's bias.
    fn add(&mut self, backend: &mut dyn Backend, mut grown: GrownTree) -> Result<()> {
        grown.tree.shrink(self.params.learning_rate);
        backend.add_tree(&grown.device_tree())?;
        if self.trees.is_empty() && self.init_score != 0.0 {
            grown.tree.add_bias(self.init_score);
        }
        self.trees.push(grown.tree);
        Ok(())
    }

    fn evaluate(&self, backend: &mut dyn Backend, collective: &mut dyn Collective) -> Result<Vec<EvalResult>> {
        let mut evals = Vec::new();
        for (index, (name, data)) in self.valid.iter().enumerate() {
            let scores = backend.scores(index + 1)?;
            let collective = &mut *collective;
            evals.extend(install(self.pool, || {
                metric::evaluate(self.params, name, &scores, data, collective)
            })?);
        }
        Ok(evals)
    }

    fn into_model(self, setup: &ObjectiveSetup) -> Model {
        let objective = match setup {
            ObjectiveSetup::Regression => ModelObjective::Regression,
            ObjectiveSetup::Binary { sigmoid } => ModelObjective::Binary { sigmoid: *sigmoid },
            ObjectiveSetup::LambdaRank { .. } => ModelObjective::LambdaRank,
        };
        Model {
            objective,
            feature_names: self.train.features.iter().map(|f| f.name.clone()).collect(),
            feature_infos: self.train.features.iter().map(|f| f.bins.info_string()).collect(),
            trees: self.trees,
            parameters: self.params.model_parameters(),
            pandas_categorical: None,
        }
    }
}
