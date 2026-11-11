//! Training parameters under LightGBM 3.3.5's names and aliases (`config.h`), parsed from a JSON object the way
//! LightGBM's Python package passes them: numbers or numeric strings, booleans or `"true"`/`"false"`, lists as JSON
//! arrays or comma-separated strings.
//!
//! An unknown key or a known feature silvic does not implement is an error, never silently ignored: a misspelt
//! `bagging_fraction` would otherwise train a different model than the one configured. Keys that only tune how
//! LightGBM itself runs (`force_row_wise`, `deterministic`, ...) are accepted and have no effect.

use serde_json::{Map, Value};

use crate::error::{Result, SilvicError};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Objective {
    /// `regression` (L2).
    Regression,
    Binary,
    LambdaRank,
}

impl Objective {
    /// The name LightGBM writes on the model's `objective=` line, without arguments.
    pub fn name(self) -> &'static str {
        match self {
            Objective::Regression => "regression",
            Objective::Binary => "binary",
            Objective::LambdaRank => "lambdarank",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceType {
    /// The fastest device present: CUDA, then Metal, then the CPU.
    Auto,
    Cpu,
    /// `gpu`: CUDA where present, else Metal.
    Gpu,
    Metal,
    Cuda,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Metric {
    /// `ndcg@k` for every `k` in `eval_at`.
    Ndcg,
    L2,
    Rmse,
    BinaryLogloss,
    Auc,
}

impl Metric {
    pub fn name(self) -> &'static str {
        match self {
            Metric::Ndcg => "ndcg",
            Metric::L2 => "l2",
            Metric::Rmse => "rmse",
            Metric::BinaryLogloss => "binary_logloss",
            Metric::Auc => "auc",
        }
    }

    /// Whether a larger value is better (LightGBM `Metric::factor_to_bigger_better`).
    pub fn higher_is_better(self) -> bool {
        matches!(self, Metric::Ndcg | Metric::Auc)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Params {
    pub objective: Objective,
    pub num_iterations: usize,
    pub learning_rate: f64,
    pub num_leaves: usize,
    /// `<= 0` means no limit.
    pub max_depth: i32,
    pub min_data_in_leaf: usize,
    pub min_sum_hessian_in_leaf: f64,
    pub lambda_l1: f64,
    pub lambda_l2: f64,
    pub min_gain_to_split: f64,
    pub bagging_fraction: f64,
    pub bagging_freq: usize,
    pub bagging_seed: u32,
    pub feature_fraction: f64,
    pub feature_fraction_seed: u32,
    pub max_bin: usize,
    pub min_data_in_bin: usize,
    pub bin_construct_sample_cnt: usize,
    pub data_random_seed: u32,
    pub max_cat_to_onehot: usize,
    pub max_cat_threshold: usize,
    pub cat_l2: f64,
    pub cat_smooth: f64,
    pub min_data_per_group: usize,
    pub boost_from_average: bool,
    pub sigmoid: f64,
    pub label_gain: Vec<f64>,
    pub lambdarank_truncation_level: usize,
    pub lambdarank_norm: bool,
    pub metric: Vec<Metric>,
    pub eval_at: Vec<usize>,
    pub use_quantized_grad: bool,
    pub num_grad_quant_bins: u32,
    pub quant_train_renew_leaf: bool,
    pub stochastic_rounding: bool,
    pub device_type: DeviceType,
    /// `gpu_device_id`: the CUDA ordinal or Metal device index; `-1` is the default device.
    pub gpu_device_id: i32,
    /// `0` uses every core.
    pub num_threads: usize,
    pub verbosity: i32,
}

/// LightGBM's default `label_gain`: `2^i - 1` for `i` in `0..31`.
pub fn default_label_gain() -> Vec<f64> {
    (0..31).map(|i| f64::from((1u32 << i) - 1)).collect()
}

impl Default for Params {
    fn default() -> Self {
        Params {
            objective: Objective::Regression,
            num_iterations: 100,
            learning_rate: 0.1,
            num_leaves: 31,
            max_depth: -1,
            min_data_in_leaf: 20,
            min_sum_hessian_in_leaf: 1e-3,
            lambda_l1: 0.0,
            lambda_l2: 0.0,
            min_gain_to_split: 0.0,
            bagging_fraction: 1.0,
            bagging_freq: 0,
            bagging_seed: 3,
            feature_fraction: 1.0,
            feature_fraction_seed: 2,
            max_bin: 255,
            min_data_in_bin: 3,
            bin_construct_sample_cnt: 200_000,
            data_random_seed: 1,
            max_cat_to_onehot: 4,
            max_cat_threshold: 32,
            cat_l2: 10.0,
            cat_smooth: 10.0,
            min_data_per_group: 100,
            boost_from_average: true,
            sigmoid: 1.0,
            label_gain: default_label_gain(),
            lambdarank_truncation_level: 30,
            lambdarank_norm: true,
            metric: Vec::new(),
            eval_at: vec![1, 2, 3, 4, 5],
            use_quantized_grad: true,
            num_grad_quant_bins: 4,
            quant_train_renew_leaf: true,
            stochastic_rounding: true,
            device_type: DeviceType::Auto,
            gpu_device_id: -1,
            num_threads: 0,
            verbosity: 1,
        }
    }
}

/// Keys accepted for LightGBM compatibility that change nothing in silvic.
const NO_EFFECT_KEYS: &[&str] = &[
    "force_row_wise",
    "force_col_wise",
    "deterministic",
    "histogram_pool_size",
    "feature_pre_filter",
    "first_metric_only",
    "gpu_platform_id",
    "gpu_use_dp",
    "num_gpu",
];

/// LightGBM features outside silvic's scope; setting them to anything but their default is an error.
const UNSUPPORTED_KEYS: &[(&str, &str)] = &[
    ("is_unbalance", "false"),
    ("scale_pos_weight", "1"),
    ("early_stopping_round", "0"),
    ("monotone_constraints", ""),
    ("linear_tree", "false"),
    ("extra_trees", "false"),
    ("path_smooth", "0"),
    ("max_delta_step", "0"),
    ("zero_as_missing", "false"),
    ("num_class", "1"),
];

/// `(canonical, aliases)` from LightGBM 3.3.5 `config.h`.
const ALIASES: &[(&str, &[&str])] = &[
    ("objective", &["objective_type", "app", "application", "loss"]),
    ("boosting", &["boosting_type", "boost"]),
    (
        "num_iterations",
        &[
            "num_iteration",
            "n_iter",
            "num_tree",
            "num_trees",
            "num_round",
            "num_rounds",
            "num_boost_round",
            "n_estimators",
            "max_iter",
        ],
    ),
    ("learning_rate", &["shrinkage_rate", "eta"]),
    ("num_leaves", &["num_leaf", "max_leaves", "max_leaf", "max_leaf_nodes"]),
    ("num_threads", &["num_thread", "nthread", "nthreads", "n_jobs"]),
    ("device_type", &["device"]),
    ("seed", &["random_seed", "random_state"]),
    ("max_depth", &[]),
    (
        "min_data_in_leaf",
        &["min_data_per_leaf", "min_data", "min_child_samples", "min_samples_leaf"],
    ),
    (
        "min_sum_hessian_in_leaf",
        &[
            "min_sum_hessian_per_leaf",
            "min_sum_hessian",
            "min_hessian",
            "min_child_weight",
        ],
    ),
    ("bagging_fraction", &["sub_row", "subsample", "bagging"]),
    ("bagging_freq", &["subsample_freq"]),
    ("bagging_seed", &["bagging_fraction_seed"]),
    ("feature_fraction", &["sub_feature", "colsample_bytree"]),
    ("feature_fraction_seed", &[]),
    ("lambda_l1", &["reg_alpha", "l1_regularization"]),
    ("lambda_l2", &["reg_lambda", "lambda", "l2_regularization"]),
    ("min_gain_to_split", &["min_split_gain"]),
    ("max_bin", &["max_bins"]),
    ("min_data_in_bin", &[]),
    ("bin_construct_sample_cnt", &["subsample_for_bin"]),
    ("data_random_seed", &["data_seed"]),
    ("max_cat_to_onehot", &[]),
    ("max_cat_threshold", &[]),
    ("cat_l2", &[]),
    ("cat_smooth", &[]),
    ("min_data_per_group", &[]),
    ("boost_from_average", &[]),
    ("sigmoid", &[]),
    ("label_gain", &[]),
    ("lambdarank_truncation_level", &[]),
    ("lambdarank_norm", &[]),
    ("metric", &["metrics", "metric_types"]),
    ("eval_at", &["ndcg_eval_at", "ndcg_at", "map_eval_at", "map_at"]),
    ("use_quantized_grad", &[]),
    ("num_grad_quant_bins", &[]),
    ("quant_train_renew_leaf", &[]),
    ("stochastic_rounding", &[]),
    ("gpu_device_id", &[]),
    ("verbosity", &["verbose"]),
];

fn canonical(key: &str) -> Option<&'static str> {
    ALIASES
        .iter()
        .find(|(name, aliases)| *name == key || aliases.contains(&key))
        .map(|(name, _)| *name)
}

impl Params {
    /// Parses a JSON object of LightGBM parameters. A canonical name and one of its aliases both present is an error.
    pub fn from_json(text: &str) -> Result<Params> {
        let value: Value =
            serde_json::from_str(text).map_err(|e| SilvicError::config(format!("parameters are not JSON: {e}")))?;
        match value {
            Value::Object(map) => Params::from_map(&map),
            _ => Err(SilvicError::config("parameters must be a JSON object")),
        }
    }

    pub fn from_map(map: &Map<String, Value>) -> Result<Params> {
        let mut params = Params::default();
        let mut seen: Vec<(&'static str, &str)> = Vec::new();
        let mut seed = None;
        for (key, value) in map {
            if NO_EFFECT_KEYS.contains(&key.as_str()) {
                continue;
            }
            if let Some((_, default)) = UNSUPPORTED_KEYS.iter().find(|(name, _)| name == key) {
                check_unsupported(key, value, default)?;
                continue;
            }
            let name = canonical(key).ok_or_else(|| SilvicError::config(format!("unknown parameter `{key}`")))?;
            if let Some((_, first)) = seen.iter().find(|(seen_name, _)| *seen_name == name) {
                return Err(SilvicError::config(format!("`{key}` and `{first}` both set `{name}`")));
            }
            seen.push((name, key));
            if name == "seed" {
                seed = Some(to_i64(key, value)?);
            } else {
                params.set(name, key, value)?;
            }
        }
        if let Some(seed) = seed {
            params.derive_seeds(seed, &seen);
        }
        params.validate()?;
        Ok(params)
    }

    /// LightGBM `Config::Set`: `seed` draws `data_random_seed`, `bagging_seed` and `feature_fraction_seed` from its
    /// `Random` in this order, unless they are set explicitly.
    fn derive_seeds(&mut self, seed: i64, seen: &[(&'static str, &str)]) {
        let mut rng = LightGbmRandom::new(seed as i32);
        let explicit = |name: &str| seen.iter().any(|(n, _)| *n == name);
        let data = rng.next_short(0, i32::from(i16::MAX)) as u32;
        let bagging = rng.next_short(0, i32::from(i16::MAX)) as u32;
        let _drop = rng.next_short(0, i32::from(i16::MAX));
        let feature = rng.next_short(0, i32::from(i16::MAX)) as u32;
        if !explicit("data_random_seed") {
            self.data_random_seed = data;
        }
        if !explicit("bagging_seed") {
            self.bagging_seed = bagging;
        }
        if !explicit("feature_fraction_seed") {
            self.feature_fraction_seed = feature;
        }
    }

    fn set(&mut self, name: &str, key: &str, value: &Value) -> Result<()> {
        match name {
            "objective" => self.objective = parse_objective(&to_string(key, value)?)?,
            "boosting" => check_boosting(&to_string(key, value)?)?,
            "device_type" => self.device_type = parse_device(&to_string(key, value)?)?,
            "metric" => self.metric = parse_metrics(key, value)?,
            "label_gain" => self.label_gain = to_f64_list(key, value)?,
            "eval_at" => self.eval_at = to_usize_list(key, value)?,
            _ => self.set_scalar(name, key, value)?,
        }
        Ok(())
    }

    fn set_scalar(&mut self, name: &str, key: &str, value: &Value) -> Result<()> {
        match name {
            "num_iterations" => self.num_iterations = to_usize(key, value)?,
            "learning_rate" => self.learning_rate = to_f64(key, value)?,
            "num_leaves" => self.num_leaves = to_usize(key, value)?,
            "max_depth" => self.max_depth = to_i64(key, value)? as i32,
            "min_data_in_leaf" => self.min_data_in_leaf = to_usize(key, value)?,
            "min_sum_hessian_in_leaf" => self.min_sum_hessian_in_leaf = to_f64(key, value)?,
            "lambda_l1" => self.lambda_l1 = to_f64(key, value)?,
            "lambda_l2" => self.lambda_l2 = to_f64(key, value)?,
            "min_gain_to_split" => self.min_gain_to_split = to_f64(key, value)?,
            "bagging_fraction" => self.bagging_fraction = to_f64(key, value)?,
            "bagging_freq" => self.bagging_freq = to_usize(key, value)?,
            "bagging_seed" => self.bagging_seed = to_i64(key, value)? as u32,
            "feature_fraction" => self.feature_fraction = to_f64(key, value)?,
            "feature_fraction_seed" => self.feature_fraction_seed = to_i64(key, value)? as u32,
            "max_bin" => self.max_bin = to_usize(key, value)?,
            "min_data_in_bin" => self.min_data_in_bin = to_usize(key, value)?,
            "bin_construct_sample_cnt" => self.bin_construct_sample_cnt = to_usize(key, value)?,
            "data_random_seed" => self.data_random_seed = to_i64(key, value)? as u32,
            "max_cat_to_onehot" => self.max_cat_to_onehot = to_usize(key, value)?,
            "max_cat_threshold" => self.max_cat_threshold = to_usize(key, value)?,
            "cat_l2" => self.cat_l2 = to_f64(key, value)?,
            "cat_smooth" => self.cat_smooth = to_f64(key, value)?,
            "min_data_per_group" => self.min_data_per_group = to_usize(key, value)?,
            "boost_from_average" => self.boost_from_average = to_bool(key, value)?,
            "sigmoid" => self.sigmoid = to_f64(key, value)?,
            "lambdarank_truncation_level" => self.lambdarank_truncation_level = to_usize(key, value)?,
            "lambdarank_norm" => self.lambdarank_norm = to_bool(key, value)?,
            "use_quantized_grad" => self.use_quantized_grad = to_bool(key, value)?,
            "num_grad_quant_bins" => self.num_grad_quant_bins = to_usize(key, value)? as u32,
            "quant_train_renew_leaf" => self.quant_train_renew_leaf = to_bool(key, value)?,
            "stochastic_rounding" => self.stochastic_rounding = to_bool(key, value)?,
            "gpu_device_id" => self.gpu_device_id = to_i64(key, value)? as i32,
            "num_threads" => self.num_threads = to_i64(key, value)?.max(0) as usize,
            "verbosity" => self.verbosity = to_i64(key, value)? as i32,
            _ => unreachable!("every canonical name in ALIASES has a setter: {name}"),
        }
        Ok(())
    }

    fn validate(&self) -> Result<()> {
        let fail = |msg: String| Err(SilvicError::config(msg));
        if self.num_leaves < 2 || self.num_leaves > 131_072 {
            return fail(format!("num_leaves must be in 2..=131072, got {}", self.num_leaves));
        }
        if self.learning_rate.is_nan() || self.learning_rate <= 0.0 {
            return fail(format!("learning_rate must be positive, got {}", self.learning_rate));
        }
        if !(self.bagging_fraction > 0.0 && self.bagging_fraction <= 1.0) {
            return fail(format!(
                "bagging_fraction must be in (0, 1], got {}",
                self.bagging_fraction
            ));
        }
        if !(self.feature_fraction > 0.0 && self.feature_fraction <= 1.0) {
            return fail(format!(
                "feature_fraction must be in (0, 1], got {}",
                self.feature_fraction
            ));
        }
        if self.max_bin < 2 || self.max_bin > 255 {
            return fail(format!(
                "max_bin must be in 2..=255 (bins are bytes), got {}",
                self.max_bin
            ));
        }
        if self.lambda_l1 < 0.0 || self.lambda_l2 < 0.0 || self.cat_l2 < 0.0 || self.cat_smooth < 0.0 {
            return fail("lambda_l1, lambda_l2, cat_l2 and cat_smooth must be non-negative".into());
        }
        if !(2..=64).contains(&self.num_grad_quant_bins) || !self.num_grad_quant_bins.is_multiple_of(2) {
            return fail(format!(
                "num_grad_quant_bins must be even and in 2..=64, got {}",
                self.num_grad_quant_bins
            ));
        }
        if self.sigmoid.is_nan() || self.sigmoid <= 0.0 {
            return fail(format!("sigmoid must be positive, got {}", self.sigmoid));
        }
        if self.eval_at.contains(&0) {
            return fail("eval_at positions must be positive".into());
        }
        Ok(())
    }

    /// The metrics to evaluate: `metric` when set, else the objective's default (LightGBM `Config::CheckParamConflict`).
    pub fn effective_metrics(&self) -> Vec<Metric> {
        if !self.metric.is_empty() {
            return self.metric.clone();
        }
        match self.objective {
            Objective::Regression => vec![Metric::L2],
            Objective::Binary => vec![Metric::BinaryLogloss],
            Objective::LambdaRank => vec![Metric::Ndcg],
        }
    }

    /// `[key: value]` lines for the model's `parameters:` block, in LightGBM's spelling.
    pub fn model_parameters(&self) -> Vec<(String, String)> {
        let join = |v: &[f64]| v.iter().map(|x| fmt_g(*x)).collect::<Vec<_>>().join(",");
        let mut out = vec![
            ("boosting".into(), "gbdt".into()),
            ("objective".into(), self.objective.name().into()),
            ("num_iterations".into(), self.num_iterations.to_string()),
            ("learning_rate".into(), fmt_g(self.learning_rate)),
            ("num_leaves".into(), self.num_leaves.to_string()),
            ("max_depth".into(), self.max_depth.to_string()),
            ("min_data_in_leaf".into(), self.min_data_in_leaf.to_string()),
            ("min_sum_hessian_in_leaf".into(), fmt_g(self.min_sum_hessian_in_leaf)),
            ("bagging_fraction".into(), fmt_g(self.bagging_fraction)),
            ("bagging_freq".into(), self.bagging_freq.to_string()),
            ("bagging_seed".into(), self.bagging_seed.to_string()),
            ("feature_fraction".into(), fmt_g(self.feature_fraction)),
            ("feature_fraction_seed".into(), self.feature_fraction_seed.to_string()),
            ("lambda_l1".into(), fmt_g(self.lambda_l1)),
            ("lambda_l2".into(), fmt_g(self.lambda_l2)),
            ("min_gain_to_split".into(), fmt_g(self.min_gain_to_split)),
            ("max_cat_threshold".into(), self.max_cat_threshold.to_string()),
            ("cat_l2".into(), fmt_g(self.cat_l2)),
            ("cat_smooth".into(), fmt_g(self.cat_smooth)),
            ("max_cat_to_onehot".into(), self.max_cat_to_onehot.to_string()),
            ("max_bin".into(), self.max_bin.to_string()),
            ("min_data_in_bin".into(), self.min_data_in_bin.to_string()),
            (
                "bin_construct_sample_cnt".into(),
                self.bin_construct_sample_cnt.to_string(),
            ),
            ("data_random_seed".into(), self.data_random_seed.to_string()),
            ("boost_from_average".into(), (self.boost_from_average as u8).to_string()),
            ("use_quantized_grad".into(), (self.use_quantized_grad as u8).to_string()),
            ("num_grad_quant_bins".into(), self.num_grad_quant_bins.to_string()),
            (
                "quant_train_renew_leaf".into(),
                (self.quant_train_renew_leaf as u8).to_string(),
            ),
            (
                "stochastic_rounding".into(),
                (self.stochastic_rounding as u8).to_string(),
            ),
        ];
        if self.objective == Objective::LambdaRank {
            out.push((
                "lambdarank_truncation_level".into(),
                self.lambdarank_truncation_level.to_string(),
            ));
            out.push(("lambdarank_norm".into(), (self.lambdarank_norm as u8).to_string()));
            out.push(("label_gain".into(), join(&self.label_gain)));
        }
        if self.objective != Objective::Regression {
            out.push(("sigmoid".into(), fmt_g(self.sigmoid)));
        }
        out
    }
}

/// C++ `operator<<` on a double with the default precision of 6 significant digits, which LightGBM uses for
/// `parameters:` values and several model fields.
pub fn fmt_g(value: f64) -> String {
    format_g(value, 6)
}

/// C `printf("%.<precision>g")`: the exponent comes from the value rounded to `precision` significant digits; fixed
/// notation when it is in `-4..precision`, trailing zeros removed. Rust's exact decimal expansion rounds half to
/// even as glibc does (`1.015625` -> `1.01562`).
pub fn format_g(value: f64, precision: usize) -> String {
    if value.is_nan() {
        return "nan".into();
    }
    if value.is_infinite() {
        return if value > 0.0 { "inf" } else { "-inf" }.into();
    }
    if value == 0.0 {
        return if value.is_sign_negative() { "-0" } else { "0" }.into();
    }
    let precision = precision.max(1);
    let sci = format!("{:.*e}", precision - 1, value);
    let (mantissa, exp) = sci.split_once('e').expect("scientific notation");
    let exp: i32 = exp.parse().expect("exponent");
    if (-4..precision as i32).contains(&exp) {
        let decimals = (precision as i32 - 1 - exp) as usize;
        trim_fraction(&format!("{value:.decimals$}")).to_string()
    } else {
        let sign = if exp < 0 { '-' } else { '+' };
        format!("{}e{sign}{:02}", trim_fraction(mantissa), exp.abs())
    }
}

fn trim_fraction(digits: &str) -> &str {
    if digits.contains('.') {
        digits.trim_end_matches('0').trim_end_matches('.')
    } else {
        digits
    }
}

/// LightGBM's `Random` (an LCG over `int`), used only to derive seeds from `seed` as LightGBM does.
struct LightGbmRandom {
    x: u32,
}

impl LightGbmRandom {
    fn new(seed: i32) -> Self {
        LightGbmRandom { x: seed as u32 }
    }

    fn rand_int16(&mut self) -> i32 {
        self.x = self.x.wrapping_mul(214_013).wrapping_add(2_531_011);
        ((self.x >> 16) & 0x7FFF) as i32
    }

    fn next_short(&mut self, lower: i32, upper: i32) -> i32 {
        self.rand_int16() % (upper - lower) + lower
    }
}

fn parse_objective(name: &str) -> Result<Objective> {
    match name.to_ascii_lowercase().as_str() {
        "regression" | "regression_l2" | "l2" | "mean_squared_error" | "mse" => Ok(Objective::Regression),
        "binary" => Ok(Objective::Binary),
        "lambdarank" => Ok(Objective::LambdaRank),
        other => Err(SilvicError::unsupported(format!(
            "objective `{other}`; silvic trains regression, binary and lambdarank"
        ))),
    }
}

fn check_boosting(name: &str) -> Result<()> {
    match name.to_ascii_lowercase().as_str() {
        "gbdt" | "gbrt" => Ok(()),
        other => Err(SilvicError::unsupported(format!(
            "boosting `{other}`; silvic trains gbdt"
        ))),
    }
}

fn parse_device(name: &str) -> Result<DeviceType> {
    match name.to_ascii_lowercase().as_str() {
        "auto" => Ok(DeviceType::Auto),
        "cpu" => Ok(DeviceType::Cpu),
        "gpu" => Ok(DeviceType::Gpu),
        "metal" => Ok(DeviceType::Metal),
        "cuda" => Ok(DeviceType::Cuda),
        other => Err(SilvicError::config(format!(
            "device_type `{other}`; expected auto, cpu, gpu, metal or cuda"
        ))),
    }
}

fn parse_metrics(key: &str, value: &Value) -> Result<Vec<Metric>> {
    let names = match value {
        Value::Array(items) => items.iter().map(|v| to_string(key, v)).collect::<Result<Vec<_>>>()?,
        _ => to_string(key, value)?
            .split(',')
            .map(|s| s.trim().to_string())
            .collect(),
    };
    let mut metrics = Vec::new();
    for name in names {
        let metric = match name.to_ascii_lowercase().as_str() {
            "" | "none" | "null" | "custom" | "na" => continue,
            "ndcg" | "lambdarank" | "rank_xendcg" | "xendcg" | "xe_ndcg" | "xe_ndcg_mart" | "xendcg_mart" => {
                Metric::Ndcg
            }
            "l2" | "mean_squared_error" | "mse" | "regression_l2" | "regression" => Metric::L2,
            "rmse" | "l2_root" | "root_mean_squared_error" => Metric::Rmse,
            "binary_logloss" | "binary" => Metric::BinaryLogloss,
            "auc" => Metric::Auc,
            other => return Err(SilvicError::unsupported(format!("metric `{other}`"))),
        };
        if !metrics.contains(&metric) {
            metrics.push(metric);
        }
    }
    Ok(metrics)
}

fn check_unsupported(key: &str, value: &Value, default: &str) -> Result<()> {
    let text = match value {
        Value::Null => return Ok(()),
        Value::String(s) => s.trim().to_string(),
        Value::Array(items) if items.is_empty() => String::new(),
        Value::Bool(b) => b.to_string(),
        other => other.to_string(),
    };
    let is_default = text == default
        || text
            .parse::<f64>()
            .ok()
            .zip(default.parse::<f64>().ok())
            .is_some_and(|(a, b)| a == b)
        || (default == "false" && text == "0");
    if is_default {
        Ok(())
    } else {
        Err(SilvicError::unsupported(format!("parameter `{key}` (set to {text})")))
    }
}

fn to_string(key: &str, value: &Value) -> Result<String> {
    match value {
        Value::String(s) => Ok(s.trim().to_string()),
        Value::Number(n) => Ok(n.to_string()),
        Value::Bool(b) => Ok(b.to_string()),
        _ => Err(SilvicError::config(format!("`{key}` must be a string"))),
    }
}

fn to_f64(key: &str, value: &Value) -> Result<f64> {
    match value {
        Value::Number(n) => n.as_f64(),
        Value::String(s) => s.trim().parse().ok(),
        _ => None,
    }
    .ok_or_else(|| SilvicError::config(format!("`{key}` must be a number, got {value}")))
}

fn to_i64(key: &str, value: &Value) -> Result<i64> {
    let number = to_f64(key, value)?;
    if number.fract() != 0.0 {
        return Err(SilvicError::config(format!("`{key}` must be an integer, got {value}")));
    }
    Ok(number as i64)
}

fn to_usize(key: &str, value: &Value) -> Result<usize> {
    let number = to_i64(key, value)?;
    usize::try_from(number).map_err(|_| SilvicError::config(format!("`{key}` must be non-negative, got {value}")))
}

fn to_bool(key: &str, value: &Value) -> Result<bool> {
    match value {
        Value::Bool(b) => Some(*b),
        Value::Number(n) => n.as_f64().map(|x| x != 0.0),
        Value::String(s) => match s.trim().to_ascii_lowercase().as_str() {
            "true" | "1" => Some(true),
            "false" | "0" => Some(false),
            _ => None,
        },
        _ => None,
    }
    .ok_or_else(|| SilvicError::config(format!("`{key}` must be a boolean, got {value}")))
}

fn list_items(key: &str, value: &Value) -> Result<Vec<Value>> {
    match value {
        Value::Array(items) => Ok(items.clone()),
        Value::String(s) => Ok(s
            .split(',')
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(|s| Value::String(s.to_string()))
            .collect()),
        Value::Number(_) => Ok(vec![value.clone()]),
        _ => Err(SilvicError::config(format!("`{key}` must be a list"))),
    }
}

fn to_f64_list(key: &str, value: &Value) -> Result<Vec<f64>> {
    list_items(key, value)?.iter().map(|v| to_f64(key, v)).collect()
}

fn to_usize_list(key: &str, value: &Value) -> Result<Vec<usize>> {
    list_items(key, value)?.iter().map(|v| to_usize(key, v)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(json: &str) -> Result<Params> {
        Params::from_json(json)
    }

    #[test]
    fn sklearn_style_aliases_parse() {
        let params = parse(
            r#"{"objective": "lambdarank", "n_estimators": 500, "num_leaves": 63, "max_depth": 8,
                "learning_rate": "0.05", "min_child_weight": 0.002, "reg_alpha": 1.5, "reg_lambda": 0.5,
                "subsample": 0.7, "subsample_freq": 3, "colsample_bytree": 0.8, "min_data_in_leaf": 40,
                "label_gain": [0, 1, 3, 7, 15, 40], "verbose": -1}"#,
        )
        .expect("params");
        assert_eq!(params.objective, Objective::LambdaRank);
        assert_eq!(params.num_iterations, 500);
        assert_eq!(params.learning_rate, 0.05);
        assert_eq!(params.min_sum_hessian_in_leaf, 0.002);
        assert_eq!((params.lambda_l1, params.lambda_l2), (1.5, 0.5));
        assert_eq!((params.bagging_fraction, params.bagging_freq), (0.7, 3));
        assert_eq!(params.label_gain, vec![0.0, 1.0, 3.0, 7.0, 15.0, 40.0]);
        assert_eq!(params.effective_metrics(), vec![Metric::Ndcg]);
    }

    #[test]
    fn unknown_and_unsupported_keys_fail() {
        assert_eq!(parse(r#"{"bagging_fractoin": 0.5}"#).unwrap_err().kind(), "config");
        assert_eq!(parse(r#"{"boosting": "dart"}"#).unwrap_err().kind(), "unsupported");
        assert_eq!(
            parse(r#"{"objective": "multiclass"}"#).unwrap_err().kind(),
            "unsupported"
        );
        assert_eq!(
            parse(r#"{"early_stopping_round": 50}"#).unwrap_err().kind(),
            "unsupported"
        );
        assert!(parse(r#"{"early_stopping_round": 0, "is_unbalance": false, "force_row_wise": true}"#).is_ok());
    }

    #[test]
    fn alias_conflicts_fail() {
        let err = parse(r#"{"num_iterations": 10, "n_estimators": 20}"#).unwrap_err();
        assert_eq!(err.kind(), "config");
    }

    #[test]
    fn seed_derives_lightgbm_seeds_unless_explicit() {
        let params = parse(r#"{"seed": 123, "bagging_seed": 7}"#).expect("params");
        assert_eq!(params.bagging_seed, 7);
        let mut rng = LightGbmRandom::new(123);
        let data = rng.next_short(0, 32767) as u32;
        assert_eq!(params.data_random_seed, data);
    }

    #[test]
    fn g_format_matches_cpp_default_stream_precision() {
        assert_eq!(fmt_g(0.1), "0.1");
        assert_eq!(fmt_g(0.0015), "0.0015");
        assert_eq!(fmt_g(1e-3), "0.001");
        assert_eq!(fmt_g(200000.0), "200000");
        assert_eq!(fmt_g(1e-15), "1e-15");
        assert_eq!(fmt_g(1234567.0), "1.23457e+06");
        assert_eq!(fmt_g(0.0), "0");
        assert_eq!(fmt_g(14.0), "14");
        assert_eq!(fmt_g(1.5e-5), "1.5e-05");
        assert_eq!(fmt_g(999_999.5), "1e+06");
        assert_eq!(fmt_g(1.015625), "1.01562");
        assert_eq!(fmt_g(-0.0460104123), "-0.0460104");
        assert_eq!(format_g(-0.1206692845480225, 17), "-0.1206692845480225");
        assert_eq!(format_g(1.0000000180025095e-35, 17), "1.0000000180025095e-35");
        assert_eq!(format_g(f64::MAX, 17), "1.7976931348623157e+308");
    }
}
