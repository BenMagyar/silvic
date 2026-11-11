//! The model configuration: Triton's `inference.ModelConfig` (`model_config.proto`) as the JSON
//! `TRITONBACKEND_ModelConfig` returns it, auto-completed from the model and validated against it.
//!
//! The defaults are the FIL backend's (`fil_backend/docs/model_config.md`): input `input__0`, output `output__0`,
//! model file `model.txt`, and its `model_type`, `is_classifier` (formerly `output_class`) and `predict_proba`
//! parameters, so a FIL LightGBM model directory serves with `backend: "silvic"`. Parameters are Triton's
//! backend-specific string parameters, read as FIL reads them (`"true"`/`"false"`, decimal integers).

use std::path::Path;
use std::str::FromStr;

use serde_json::{json, Map, Value};
use silvic_core::model::ModelObjective;
use silvic_core::Model;
use silvic_serve::Kernel;

use crate::error::{Error, Result};

/// FIL's model file name for `model_type: "lightgbm"`.
pub const DEFAULT_MODEL_FILE: &str = "model.txt";
/// FIL's input tensor name.
pub const DEFAULT_INPUT_NAME: &str = "input__0";
/// FIL's output tensor name.
pub const DEFAULT_OUTPUT_NAME: &str = "output__0";

/// The parameters silvic reads.
const SILVIC_PARAMETERS: [&str; 5] = ["model_file", "raw_score", "num_iteration", "kernel", "num_threads"];
/// FIL parameters with a meaning here: the model format and the kind of output.
const FIL_PARAMETERS: [&str; 4] = ["model_type", "is_classifier", "output_class", "predict_proba"];
/// FIL parameters that tune only FIL's own execution (GPU tree layout, chunking, CPU/GPU transfer, XGBoost parsing),
/// plus `threshold`, which only moves FIL's class-ID output this backend does not produce: accepted, no effect.
const FIL_IGNORED_PARAMETERS: [&str; 10] = [
    "algo",
    "blocks_per_sm",
    "chunk_size",
    "layout",
    "storage_type",
    "threads_per_tree",
    "threshold",
    "transfer_threshold",
    "use_experimental_optimizations",
    "xgboost_allow_unknown_field",
];

/// The tensor element types the backend reads and writes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DataType {
    Fp32,
    Fp64,
}

impl DataType {
    /// From the `data_type` name in the config (`TYPE_FP32`, `TYPE_FP64`).
    pub fn from_config_name(name: &str) -> Option<DataType> {
        match name {
            "TYPE_FP32" => Some(DataType::Fp32),
            "TYPE_FP64" => Some(DataType::Fp64),
            _ => None,
        }
    }

    pub fn config_name(self) -> &'static str {
        match self {
            DataType::Fp32 => "TYPE_FP32",
            DataType::Fp64 => "TYPE_FP64",
        }
    }

    /// Bytes per element.
    pub fn size(self) -> usize {
        match self {
            DataType::Fp32 => 4,
            DataType::Fp64 => 8,
        }
    }
}

/// The backend parameters of a config.
#[derive(Debug, Clone, PartialEq)]
pub struct Parameters {
    /// The model file in the version directory.
    pub model_file: String,
    /// `Booster.predict(raw_score=...)`: margins instead of the objective's transform.
    pub raw_score: bool,
    /// `Booster.predict(num_iteration=...)` as a tree count: every tree when `None` (a `num_iteration` of 0 or less).
    pub num_trees: Option<usize>,
    /// `Kernel::detect` when `None`.
    pub kernel: Option<Kernel>,
    /// Threads per scored matrix: 1 scores on Triton's instance thread, 0 on all of rayon's.
    pub num_threads: usize,
    /// FIL `is_classifier` (or `output_class`).
    pub classifier: bool,
    /// FIL `predict_proba`.
    pub predict_proba: bool,
}

impl Default for Parameters {
    fn default() -> Parameters {
        Parameters {
            model_file: DEFAULT_MODEL_FILE.to_owned(),
            raw_score: false,
            num_trees: None,
            kernel: None,
            num_threads: 1,
            classifier: false,
            predict_proba: false,
        }
    }
}

impl Parameters {
    /// Reads `parameters`; a key neither silvic nor FIL defines is an error naming it.
    pub fn from_config(config: &Value) -> Result<Parameters> {
        let mut parameters = Parameters::default();
        let Some(entries) = config.get("parameters") else {
            return Ok(parameters);
        };
        let entries = entries
            .as_object()
            .ok_or_else(|| Error::config("`parameters` is not a map"))?;
        for (key, entry) in entries {
            let value = entry
                .get("string_value")
                .and_then(Value::as_str)
                .ok_or_else(|| Error::config(format!("parameter `{key}` has no string_value")))?;
            parameters.set(key, value.trim())?;
        }
        Ok(parameters)
    }

    fn set(&mut self, key: &str, value: &str) -> Result<()> {
        match key {
            "model_file" if value.is_empty() => return Err(Error::config("parameter `model_file` is empty")),
            "model_file" => self.model_file = value.to_owned(),
            "raw_score" => self.raw_score = parse_bool(key, value)?,
            "num_iteration" => {
                let iterations: i64 = parse_number(key, value)?;
                self.num_trees = usize::try_from(iterations).ok().filter(|&count| count > 0);
            }
            "kernel" => {
                self.kernel = Some(Kernel::from_str(value).map_err(|_| {
                    Error::config(format!(
                        "parameter `kernel`: unknown kernel `{value}` (portable, neon, avx2, avx512)"
                    ))
                })?)
            }
            "num_threads" => self.num_threads = parse_number(key, value)?,
            "model_type" if value == "lightgbm" => {}
            "model_type" => {
                return Err(Error::unsupported(format!(
                    "parameter `model_type` is `{value}`: silvic serves LightGBM text models (`lightgbm`)"
                )))
            }
            "is_classifier" | "output_class" => self.classifier = parse_bool(key, value)?,
            "predict_proba" => self.predict_proba = parse_bool(key, value)?,
            _ if FIL_IGNORED_PARAMETERS.contains(&key) => {}
            _ => {
                return Err(Error::config(format!(
                    "unknown parameter `{key}` (silvic reads {}; FIL's {} and its tuning parameters are accepted)",
                    SILVIC_PARAMETERS.join(", "),
                    FIL_PARAMETERS.join(", ")
                )))
            }
        }
        Ok(())
    }
}

fn parse_bool(key: &str, value: &str) -> Result<bool> {
    match value {
        "true" => Ok(true),
        "false" => Ok(false),
        _ => Err(Error::config(format!(
            "parameter `{key}` is `{value}`, not true or false"
        ))),
    }
}

fn parse_number<T: FromStr>(key: &str, value: &str) -> Result<T> {
    value
        .parse()
        .map_err(|_| Error::config(format!("parameter `{key}` is `{value}`, not an integer in range")))
}

/// Reads and parses `<model_dir>/<version>/<model_file>`.
pub fn load_model(model_dir: &Path, version: u64, parameters: &Parameters) -> Result<Model> {
    let path = model_dir.join(version.to_string()).join(&parameters.model_file);
    let text = std::fs::read_to_string(&path).map_err(|error| Error::Io {
        path: path.display().to_string().into(),
        message: error.to_string().into(),
    })?;
    Ok(Model::from_lightgbm_string(&text)?)
}

/// Fills in a missing `input` (`input__0`, FP32) and `output` (`output__0`, FP64: LightGBM's predictions exactly)
/// with the dims the config's `max_batch_size` calls for. Returns whether the config changed.
pub fn auto_complete(config: &mut Value, num_features: usize) -> Result<bool> {
    let batching = max_batch_size(config)? > 0;
    let config = config
        .as_object_mut()
        .ok_or_else(|| Error::config("the model configuration is not a JSON object"))?;
    let input_dims = if batching {
        json!([num_features])
    } else {
        json!([-1, num_features])
    };
    let output_dims = if batching { json!([1]) } else { json!([-1]) };
    let mut changed = false;
    for (field, name, data_type, dims) in [
        ("input", DEFAULT_INPUT_NAME, DataType::Fp32, input_dims),
        ("output", DEFAULT_OUTPUT_NAME, DataType::Fp64, output_dims),
    ] {
        if is_empty_list(config, field) {
            let tensor = json!({ "name": name, "data_type": data_type.config_name(), "dims": dims });
            config.insert(field.to_owned(), json!([tensor]));
            changed = true;
        }
    }
    Ok(changed)
}

fn is_empty_list(config: &Map<String, Value>, field: &str) -> bool {
    config
        .get(field)
        .is_none_or(|value| value.as_array().is_some_and(Vec::is_empty))
}

/// The output tensor's shape for a request of `rows` rows.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OutputShape {
    dims: [i64; 2],
    len: usize,
}

impl OutputShape {
    pub fn as_slice(&self) -> &[i64] {
        &self.dims[..self.len]
    }
}

/// A validated configuration: what the instances need to read requests and write responses.
#[derive(Debug, Clone, PartialEq)]
pub struct Settings {
    /// 0: Triton does not batch, and the input's dims carry the row dimension.
    pub max_batch_size: usize,
    pub input_name: String,
    pub input_type: DataType,
    pub output_name: String,
    pub output_type: DataType,
    pub num_features: usize,
    pub raw_score: bool,
    pub num_trees: Option<usize>,
    pub num_threads: usize,
}

impl Settings {
    /// Checks the config's one input and one output against the model: input dims `[num_features]` with batching or
    /// `[-1, num_features]` without, output dims `[1]` or `[-1]`.
    pub fn new(config: &Value, parameters: &Parameters, model: &Model) -> Result<Settings> {
        check_classifier(parameters, model)?;
        let max_batch_size = max_batch_size(config)?;
        let num_features = model.num_features();
        let width = i64::try_from(num_features).map_err(|_| Error::config("the model has too many features"))?;
        let (input_dims, output_dims, batching) = if max_batch_size > 0 {
            (vec![width], vec![1], "max_batch_size > 0")
        } else {
            (vec![-1, width], vec![-1], "max_batch_size 0")
        };
        let input_rule = format!("the model reads {num_features} features, so with {batching} the input dims are");
        let (input_name, input_type) = tensor(config, "input", &input_dims, &input_rule)?;
        let output_rule = format!("silvic writes one score per row, so with {batching} the output dims are");
        let (output_name, output_type) = tensor(config, "output", &output_dims, &output_rule)?;
        Ok(Settings {
            max_batch_size,
            input_name,
            input_type,
            output_name,
            output_type,
            num_features,
            raw_score: parameters.raw_score,
            num_trees: parameters.num_trees,
            num_threads: parameters.num_threads,
        })
    }

    pub fn batching(&self) -> bool {
        self.max_batch_size > 0
    }

    /// `[rows, 1]` with batching (Triton's batch dimension, then the config's `[1]`), `[rows]` without.
    pub fn output_shape(&self, rows: usize) -> OutputShape {
        let rows = i64::try_from(rows).expect("a row count Triton sent fits its int64 shape");
        if self.batching() {
            OutputShape {
                dims: [rows, 1],
                len: 2,
            }
        } else {
            OutputShape {
                dims: [rows, 0],
                len: 1,
            }
        }
    }
}

/// FIL's classifier outputs: with `predict_proba`, a binary model's probability (one value per row, as FIL returns
/// for binary models), which is this backend's transformed output; without it, class IDs, which it does not produce.
fn check_classifier(parameters: &Parameters, model: &Model) -> Result<()> {
    if !parameters.classifier {
        return Ok(());
    }
    if !matches!(model.objective, ModelObjective::Binary { .. }) {
        return Err(Error::config(format!(
            "parameter `is_classifier` is true but the model's objective is {:?}, not binary",
            model.objective
        )));
    }
    if !parameters.predict_proba {
        return Err(Error::unsupported(
            "class IDs (`is_classifier` without `predict_proba`); set `predict_proba: true` for the probability",
        ));
    }
    Ok(())
}

fn max_batch_size(config: &Value) -> Result<usize> {
    match config.get("max_batch_size") {
        None => Ok(0),
        Some(value) => integer(value)
            .and_then(|size| usize::try_from(size).ok())
            .ok_or_else(|| Error::config(format!("max_batch_size {value} is not a non-negative integer"))),
    }
}

/// Triton writes the proto's int64 fields as JSON numbers; protobuf's own JSON mapping writes them as strings.
fn integer(value: &Value) -> Option<i64> {
    value.as_i64().or_else(|| value.as_str()?.parse().ok())
}

/// The one tensor of `field` (`input` or `output`): its name and data type, after checking its dims are
/// `expected_dims`; `rule` says why they must be.
fn tensor(config: &Value, field: &str, expected_dims: &[i64], rule: &str) -> Result<(String, DataType)> {
    let tensors = config
        .get(field)
        .and_then(Value::as_array)
        .map_or(&[][..], Vec::as_slice);
    let [tensor] = tensors else {
        return Err(Error::config(format!(
            "silvic models have one {field}, the configuration has {}",
            tensors.len()
        )));
    };
    let name = tensor
        .get("name")
        .and_then(Value::as_str)
        .filter(|name| !name.is_empty())
        .ok_or_else(|| Error::config(format!("the {field} has no name")))?;
    let type_name = tensor
        .get("data_type")
        .and_then(Value::as_str)
        .unwrap_or("TYPE_INVALID");
    let data_type = DataType::from_config_name(type_name).ok_or_else(|| {
        Error::config(format!(
            "{field} `{name}` is {type_name}; silvic reads and writes TYPE_FP32 or TYPE_FP64"
        ))
    })?;
    if tensor.get("reshape").is_some_and(|reshape| !reshape.is_null()) {
        return Err(Error::config(format!(
            "{field} `{name}` has a reshape, which silvic does not apply"
        )));
    }
    let dims: Option<Vec<i64>> = tensor
        .get("dims")
        .and_then(Value::as_array)
        .map_or(Some(Vec::new()), |dims| dims.iter().map(integer).collect());
    if dims.as_deref() != Some(expected_dims) {
        return Err(Error::config(format!(
            "{field} `{name}` has dims {}; {rule} {expected_dims:?}",
            tensor.get("dims").unwrap_or(&Value::Null),
        )));
    }
    Ok((name.to_owned(), data_type))
}

#[cfg(test)]
mod tests {
    use super::*;

    const BINARY: &str = include_str!("../../silvic-core/tests/fixtures/lightgbm/binary.txt");
    const REGRESSION: &str = include_str!("../../silvic-core/tests/fixtures/lightgbm/regression_nan_cat.txt");

    fn model(text: &str) -> Model {
        Model::from_lightgbm_string(text).expect("fixture model")
    }

    fn params(pairs: &[(&str, &str)]) -> Value {
        let entries: Map<String, Value> = pairs
            .iter()
            .map(|(key, value)| (key.to_string(), json!({ "string_value": value })))
            .collect();
        json!({ "parameters": entries })
    }

    #[test]
    fn parameters_default_to_lightgbm_predict() {
        let parameters = Parameters::from_config(&json!({})).expect("no parameters");
        assert_eq!(parameters, Parameters::default());
        assert_eq!(parameters.model_file, "model.txt");
        assert_eq!(parameters.num_threads, 1);
    }

    #[test]
    fn parameters_parse() {
        let config = params(&[
            ("model_file", "booster.txt"),
            ("raw_score", "true"),
            ("num_iteration", "40"),
            ("kernel", "portable"),
            ("num_threads", "0"),
            ("model_type", "lightgbm"),
            ("storage_type", "AUTO"),
        ]);
        let parameters = Parameters::from_config(&config).expect("parameters");
        assert_eq!(parameters.model_file, "booster.txt");
        assert!(parameters.raw_score);
        assert_eq!(parameters.num_trees, Some(40));
        assert_eq!(parameters.kernel, Some(Kernel::Portable));
        assert_eq!(parameters.num_threads, 0);
        for all in ["0", "-1"] {
            let parameters = Parameters::from_config(&params(&[("num_iteration", all)])).expect("parameters");
            assert_eq!(parameters.num_trees, None, "num_iteration {all}");
        }
    }

    #[test]
    fn bad_parameters_name_themselves() {
        let cases = [
            (vec![("raw_scores", "true")], "unknown parameter `raw_scores`"),
            (vec![("raw_score", "yes")], "parameter `raw_score` is `yes`"),
            (vec![("num_threads", "-2")], "parameter `num_threads` is `-2`"),
            (vec![("kernel", "sse")], "unknown kernel `sse`"),
            (
                vec![("model_type", "xgboost_json")],
                "parameter `model_type` is `xgboost_json`",
            ),
            (vec![("model_file", "")], "parameter `model_file` is empty"),
        ];
        for (pairs, expected) in cases {
            let error = Parameters::from_config(&params(&pairs)).expect_err("rejected");
            assert!(error.to_string().contains(expected), "{expected}: got {error}");
        }
        let error = Parameters::from_config(&json!({ "parameters": { "raw_score": {} } })).expect_err("no value");
        assert!(error.to_string().contains("no string_value"), "{error}");
    }

    #[test]
    fn auto_complete_fills_io_from_the_model() {
        let regression = model(REGRESSION);
        let mut config = json!({ "backend": "silvic", "max_batch_size": 64, "input": [], "output": [] });
        assert!(auto_complete(&mut config, regression.num_features()).expect("completes"));
        assert_eq!(
            config["input"],
            json!([{ "name": "input__0", "data_type": "TYPE_FP32", "dims": [5] }])
        );
        assert_eq!(
            config["output"],
            json!([{ "name": "output__0", "data_type": "TYPE_FP64", "dims": [1] }])
        );
        let settings = Settings::new(&config, &Parameters::default(), &regression).expect("valid");
        assert!(settings.batching());
        assert_eq!(settings.output_shape(3).as_slice(), [3, 1]);

        let mut config = json!({ "backend": "silvic" });
        assert!(auto_complete(&mut config, 5).expect("completes"));
        assert_eq!(config["input"][0]["dims"], json!([-1, 5]));
        assert_eq!(config["output"][0]["dims"], json!([-1]));
        let settings = Settings::new(&config, &Parameters::default(), &regression).expect("valid");
        assert!(!settings.batching());
        assert_eq!(settings.output_shape(3).as_slice(), [3]);

        let unchanged = config.clone();
        assert!(!auto_complete(&mut config, 5).expect("complete already"));
        assert_eq!(config, unchanged);
    }

    #[test]
    fn given_configs_are_checked_against_the_model() {
        let regression = model(REGRESSION);
        let io = |input_type: &str, input_dims: Value, output_dims: Value| {
            json!({
                "max_batch_size": "8",
                "input": [{ "name": "features", "data_type": input_type, "dims": input_dims }],
                "output": [{ "name": "scores", "data_type": "TYPE_FP32", "dims": output_dims }],
            })
        };
        let settings = Settings::new(
            &io("TYPE_FP64", json!(["5"]), json!([1])),
            &Parameters::default(),
            &regression,
        )
        .expect("valid");
        assert_eq!(
            (settings.input_name.as_str(), settings.input_type, settings.output_type),
            ("features", DataType::Fp64, DataType::Fp32)
        );
        let cases = [
            (
                io("TYPE_FP32", json!([4]), json!([1])),
                "has dims [4]; the model reads 5 features",
            ),
            (
                io("TYPE_FP32", json!([-1, 5]), json!([1])),
                "with max_batch_size > 0 the input dims are [5]",
            ),
            (io("TYPE_FP32", json!([5]), json!([2])), "output `scores` has dims [2]"),
            (io("TYPE_INT64", json!([5]), json!([1])), "is TYPE_INT64"),
        ];
        for (config, expected) in cases {
            let error = Settings::new(&config, &Parameters::default(), &regression).expect_err("rejected");
            assert!(error.to_string().contains(expected), "{expected}: got {error}");
        }
        let mut two_inputs = io("TYPE_FP32", json!([5]), json!([1]));
        let second = two_inputs["input"][0].clone();
        two_inputs["input"].as_array_mut().expect("inputs").push(second);
        let error = Settings::new(&two_inputs, &Parameters::default(), &regression).expect_err("two inputs");
        assert!(
            error.to_string().contains("one input, the configuration has 2"),
            "{error}"
        );
    }

    #[test]
    fn fil_classifier_parameters() {
        let binary = model(BINARY);
        let mut config = json!({ "max_batch_size": 4 });
        auto_complete(&mut config, binary.num_features()).expect("completes");
        let proba = Parameters::from_config(&params(&[("is_classifier", "true"), ("predict_proba", "true")]))
            .expect("parameters");
        Settings::new(&config, &proba, &binary).expect("probabilities");
        let class_ids = Parameters::from_config(&params(&[("output_class", "true")])).expect("parameters");
        let error = Settings::new(&config, &class_ids, &binary).expect_err("class ids");
        assert_eq!(error.kind(), "unsupported");
        let error = Settings::new(&config, &proba, &model(REGRESSION)).expect_err("not binary");
        assert!(error.to_string().contains("not binary"), "{error}");
    }

    #[test]
    fn load_model_reads_the_version_directory() {
        let dir = std::env::temp_dir().join(format!("silvic-triton-load-{}", std::process::id()));
        std::fs::create_dir_all(dir.join("3")).expect("version directory");
        std::fs::write(dir.join("3").join("model.txt"), REGRESSION).expect("model file");
        let loaded = load_model(&dir, 3, &Parameters::default()).expect("loads");
        assert_eq!(loaded.num_features(), 5);
        let error = load_model(&dir, 4, &Parameters::default()).expect_err("no version 4");
        assert_eq!(error.kind(), "io");
        assert!(error.to_string().contains("4/model.txt"), "{error}");
        std::fs::remove_dir_all(&dir).expect("cleanup");
    }
}
