//! One `TRITONBACKEND_ModelInstanceExecute` call's requests: each request's input checked against the
//! configuration, the matrices the requests are scored as, and each request's scores written into its output.
//!
//! silvic-serve scores rows in blocks of 64 (32 on the portable kernel), so a 1-row request alone costs a whole
//! block. Requests of fewer than [`GATHER_MAX_ROWS`] rows that dynamic batching delivers together are copied into
//! one matrix and scored as one; a larger request is scored on its own, in place when Triton holds it as one
//! suitably aligned buffer (no copy), else from a copy. Scores are split back per request in request order.

use std::ops::Range;
use std::panic::{self, AssertUnwindSafe};

use silvic_core::data::{DenseData, DenseMatrix, Layout};
use silvic_serve::Predictor;

use crate::config::{DataType, Settings};
use crate::error::{Error, Result};

/// Requests below this many rows share one scored matrix. Gathering copies the rows; below 4 blocks the block a
/// request would pad out costs more than the copy, above it the copy is the larger cost.
pub const GATHER_MAX_ROWS: usize = 256;

/// A request's input tensor as Triton describes it.
#[derive(Debug, Clone)]
pub struct InputTensor<'a> {
    pub name: &'a str,
    /// `None` for a type the backend does not read.
    pub data_type: Option<DataType>,
    pub shape: &'a [i64],
    /// The tensor's bytes, in order, in CPU memory.
    pub buffers: Vec<&'a [u8]>,
}

/// A request input that matches the configuration: `rows` rows of `num_features` values, row-major, in `buffers`.
#[derive(Debug, Clone)]
pub struct RequestInput<'a> {
    pub rows: usize,
    buffers: Vec<&'a [u8]>,
}

/// Checks a request's input against the configured data type and width (Triton has already checked it against the
/// config's dims and `max_batch_size`; the backend checks again rather than trust a buffer's length).
pub fn check_input<'a>(settings: &Settings, tensor: InputTensor<'a>) -> Result<RequestInput<'a>> {
    let name = tensor.name;
    if tensor.data_type != Some(settings.input_type) {
        return Err(Error::request(format!(
            "input `{name}` must be {}",
            settings.input_type.config_name()
        )));
    }
    let rows = match *tensor.shape {
        [rows, width] if usize::try_from(width) == Ok(settings.num_features) => usize::try_from(rows).ok(),
        _ => None,
    }
    .ok_or_else(|| {
        Error::request(format!(
            "input `{name}` has shape {:?}; the model reads [rows, {}]",
            tensor.shape, settings.num_features
        ))
    })?;
    let expected = rows
        .checked_mul(settings.num_features * settings.input_type.size())
        .ok_or_else(|| Error::request(format!("input `{name}` is too large")))?;
    let actual: usize = tensor.buffers.iter().map(|buffer| buffer.len()).sum();
    if actual != expected {
        return Err(Error::request(format!(
            "input `{name}` holds {actual} bytes; {rows} rows of {} features take {expected}",
            settings.num_features
        )));
    }
    Ok(RequestInput {
        rows,
        buffers: tensor.buffers,
    })
}

/// Memory an instance reuses across execute calls for copied inputs. Stored as f64, so an f32 or f64 view of its
/// start is aligned.
#[derive(Debug, Default)]
pub struct Scratch {
    values: Vec<f64>,
}

impl Scratch {
    /// Copies the members' rows, in order, into one matrix's values.
    fn gather<'s>(&'s mut self, data_type: DataType, members: &[&RequestInput<'_>]) -> DenseData<'s> {
        let buffers = || members.iter().flat_map(|member| member.buffers.iter());
        let total: usize = buffers().map(|buffer| buffer.len()).sum();
        self.values.resize(total.div_ceil(8), 0.0);
        let bytes: &mut [u8] = bytemuck::cast_slice_mut(self.values.as_mut_slice());
        let mut start = 0;
        for buffer in buffers() {
            bytes[start..start + buffer.len()].copy_from_slice(buffer);
            start += buffer.len();
        }
        let bytes = &bytemuck::cast_slice::<f64, u8>(&self.values)[..total];
        match data_type {
            DataType::Fp32 => DenseData::F32(bytemuck::cast_slice(bytes)),
            DataType::Fp64 => DenseData::F64(bytemuck::cast_slice(bytes)),
        }
    }
}

/// The values of a lone request without a copy, when they are one buffer aligned for `data_type`.
fn in_place<'a>(data_type: DataType, input: &RequestInput<'a>) -> Option<DenseData<'a>> {
    let [buffer] = input.buffers[..] else {
        return None;
    };
    match data_type {
        DataType::Fp32 => bytemuck::try_cast_slice(buffer).ok().map(DenseData::F32),
        DataType::Fp64 => bytemuck::try_cast_slice(buffer).ok().map(DenseData::F64),
    }
}

/// The scores of one execute call: every request's rows are a range of one scored matrix's scores.
#[derive(Debug)]
pub struct BatchScores {
    matrices: Vec<Result<Vec<f64>>>,
    /// Per request: its matrix and rows there; `None` for a request whose input was rejected.
    placement: Vec<Option<(usize, Range<usize>)>>,
}

impl BatchScores {
    /// The scores of request `index`, or the error its matrix failed with.
    ///
    /// # Panics
    /// When `index` is a request whose input was rejected.
    pub fn request(&self, index: usize) -> Result<&[f64]> {
        let (matrix, rows) = self.placement[index].clone().expect("a request with a checked input");
        match &self.matrices[matrix] {
            Ok(scores) => Ok(&scores[rows]),
            Err(error) => Err(error.clone()),
        }
    }
}

/// Scores every request whose input was accepted: small ones gathered into one matrix, the others one by one.
pub fn score(
    predictor: &Predictor,
    settings: &Settings,
    inputs: &[Result<RequestInput<'_>>],
    scratch: &mut Scratch,
) -> BatchScores {
    let accepted = || {
        inputs
            .iter()
            .enumerate()
            .filter_map(|(index, input)| Some((index, input.as_ref().ok()?)))
    };
    let small: Vec<usize> = accepted()
        .filter(|(_, input)| input.rows < GATHER_MAX_ROWS)
        .map(|(index, _)| index)
        .collect();
    let mut groups: Vec<Vec<usize>> = accepted()
        .filter(|(index, _)| small.len() < 2 || !small.contains(index))
        .map(|(index, _)| vec![index])
        .collect();
    if small.len() >= 2 {
        groups.push(small);
    }
    let mut placement = vec![None; inputs.len()];
    let mut matrices = Vec::with_capacity(groups.len());
    for group in groups {
        let members: Vec<&RequestInput<'_>> = group
            .iter()
            .map(|&index| inputs[index].as_ref().expect("an accepted input"))
            .collect();
        let mut start = 0;
        for (&index, member) in group.iter().zip(&members) {
            placement[index] = Some((matrices.len(), start..start + member.rows));
            start += member.rows;
        }
        matrices.push(score_matrix(predictor, settings, &members, start, scratch));
    }
    BatchScores { matrices, placement }
}

/// Scores the members' rows as one matrix of `rows` rows. A panic in scoring fails these requests only: the
/// predictor is immutable and the scratch is rewritten before every use.
fn score_matrix(
    predictor: &Predictor,
    settings: &Settings,
    members: &[&RequestInput<'_>],
    rows: usize,
    scratch: &mut Scratch,
) -> Result<Vec<f64>> {
    if rows == 0 {
        return Ok(Vec::new());
    }
    let data = match members {
        [only] => in_place(settings.input_type, only),
        _ => None,
    }
    .unwrap_or_else(|| scratch.gather(settings.input_type, members));
    let matrix = DenseMatrix {
        data,
        rows,
        cols: settings.num_features,
        layout: Layout::RowMajor,
    };
    panic::catch_unwind(AssertUnwindSafe(|| {
        predictor.predict(&matrix, settings.raw_score, settings.num_trees, settings.num_threads)
    }))
    .map_err(|payload| Error::panic(&*payload))
}

/// Writes `scores` into an output buffer as f64, or as their cast to f32, in native byte order (Triton's tensor
/// layout). Written bytewise: the buffer Triton hands out has no alignment guarantee.
///
/// # Panics
/// When `out` is not one element per score.
pub fn write_scores(scores: &[f64], data_type: DataType, out: &mut [u8]) {
    assert_eq!(
        out.len(),
        scores.len() * data_type.size(),
        "one output element per score"
    );
    match data_type {
        DataType::Fp64 => {
            for (bytes, score) in out.as_chunks_mut::<8>().0.iter_mut().zip(scores) {
                *bytes = score.to_ne_bytes();
            }
        }
        DataType::Fp32 => {
            for (bytes, score) in out.as_chunks_mut::<4>().0.iter_mut().zip(scores) {
                *bytes = (*score as f32).to_ne_bytes();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;
    use silvic_core::Model;

    use super::*;
    use crate::config::{self, Parameters};

    const REGRESSION: &str = include_str!("../../silvic-core/tests/fixtures/lightgbm/regression_nan_cat.txt");
    const BINARY: &str = include_str!("../../silvic-core/tests/fixtures/lightgbm/binary.txt");

    struct Fixture {
        model: Model,
        predictor: Predictor,
        settings: Settings,
    }

    fn fixture(text: &str, input_type: DataType, raw_score: bool) -> Fixture {
        let model = Model::from_lightgbm_string(text).expect("fixture model");
        let mut config = json!({ "max_batch_size": 4096 });
        config::auto_complete(&mut config, model.num_features()).expect("completes");
        config["input"][0]["data_type"] = json!(input_type.config_name());
        let parameters = Parameters {
            raw_score,
            ..Parameters::default()
        };
        let settings = Settings::new(&config, &parameters, &model).expect("valid");
        let predictor = Predictor::new(&model).expect("compiles");
        Fixture {
            model,
            predictor,
            settings,
        }
    }

    /// Rows with NaN, zeros, negatives (unseen categories) and values a few units around the thresholds.
    fn rows(count: usize, cols: usize, seed: u64) -> Vec<f64> {
        let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
        (0..count * cols)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                match state % 10 {
                    0 => f64::NAN,
                    1 => 0.0,
                    2 => -1.0,
                    _ => (state >> 11) as f64 / (1u64 << 53) as f64 * 8.0 - 2.0,
                }
            })
            .collect()
    }

    fn bytes(values: &[f64], data_type: DataType) -> Vec<u8> {
        match data_type {
            DataType::Fp64 => values.iter().flat_map(|value| value.to_ne_bytes()).collect(),
            DataType::Fp32 => values.iter().flat_map(|value| (*value as f32).to_ne_bytes()).collect(),
        }
    }

    fn expected(fixture: &Fixture, values: &[f64], data_type: DataType) -> Vec<f64> {
        let rows = values.len() / fixture.settings.num_features;
        let narrowed: Vec<f32> = values.iter().map(|&value| value as f32).collect();
        let data = match data_type {
            DataType::Fp64 => DenseData::F64(values),
            DataType::Fp32 => DenseData::F32(&narrowed),
        };
        let matrix = DenseMatrix {
            data,
            rows,
            cols: fixture.settings.num_features,
            layout: Layout::RowMajor,
        };
        fixture.model.predict(&matrix, fixture.settings.raw_score, None)
    }

    fn tensor<'a>(name: &'a str, data_type: DataType, shape: &'a [i64], buffers: Vec<&'a [u8]>) -> InputTensor<'a> {
        InputTensor {
            name,
            data_type: Some(data_type),
            shape,
            buffers,
        }
    }

    #[test]
    fn rejected_inputs_say_why() {
        let fixture = fixture(REGRESSION, DataType::Fp32, false);
        let data = bytes(&rows(2, 5, 1), DataType::Fp32);
        let cases: [(InputTensor<'_>, &str); 5] = [
            (
                tensor("x", DataType::Fp64, &[2, 5], vec![&data]),
                "input `x` must be TYPE_FP32",
            ),
            (
                tensor("x", DataType::Fp32, &[2, 4], vec![&data]),
                "has shape [2, 4]; the model reads [rows, 5]",
            ),
            (tensor("x", DataType::Fp32, &[10], vec![&data]), "has shape [10]"),
            (tensor("x", DataType::Fp32, &[-1, 5], vec![&data]), "has shape [-1, 5]"),
            (
                tensor("x", DataType::Fp32, &[3, 5], vec![&data]),
                "holds 40 bytes; 3 rows of 5 features take 60",
            ),
        ];
        for (input, message) in cases {
            let error = check_input(&fixture.settings, input).expect_err("rejected");
            assert!(error.to_string().contains(message), "{message}: got {error}");
        }
        let unknown = InputTensor {
            data_type: None,
            ..tensor("x", DataType::Fp32, &[2, 5], vec![&data])
        };
        assert!(check_input(&fixture.settings, unknown).is_err(), "an unread type");
    }

    /// Requests of every size and buffer split score as `Model::predict` scores each alone, bit for bit.
    #[test]
    fn batched_requests_score_as_alone() {
        for (text, data_type, raw_score) in [
            (REGRESSION, DataType::Fp32, false),
            (REGRESSION, DataType::Fp64, true),
            (BINARY, DataType::Fp32, false),
            (BINARY, DataType::Fp64, true),
        ] {
            let fixture = fixture(text, data_type, raw_score);
            let cols = fixture.settings.num_features;
            let sizes = [1, 3, 0, 300, 1, 64, 257];
            let values: Vec<Vec<f64>> = sizes
                .iter()
                .enumerate()
                .map(|(seed, &count)| rows(count, cols, seed as u64))
                .collect();
            let encoded: Vec<Vec<u8>> = values.iter().map(|rows| bytes(rows, data_type)).collect();
            // The 300-row request starts one byte past an aligned address, so it is copied; the second 1-row
            // request arrives in two buffers.
            let mut shifted = vec![0_u8; encoded[3].len() + 1];
            shifted[1..].copy_from_slice(&encoded[3]);
            let shapes: Vec<[i64; 2]> = sizes.iter().map(|&count| [count as i64, cols as i64]).collect();
            let split = encoded[4].len() / 2;
            let buffers: Vec<Vec<&[u8]>> = (0..sizes.len())
                .map(|index| match index {
                    4 => vec![&encoded[4][..split], &encoded[4][split..]],
                    3 => vec![&shifted[1..]],
                    _ => vec![&encoded[index][..]],
                })
                .collect();
            let mut inputs: Vec<Result<RequestInput<'_>>> = (0..sizes.len())
                .map(|index| {
                    check_input(
                        &fixture.settings,
                        tensor("x", data_type, &shapes[index], buffers[index].clone()),
                    )
                })
                .collect();
            inputs.insert(2, Err(Error::request("rejected")));
            let mut scratch = Scratch::default();
            let scores = score(&fixture.predictor, &fixture.settings, &inputs, &mut scratch);
            // 300 and 257 rows alone; 1, 3, 0, 1 and 64 together.
            assert_eq!(scores.matrices.len(), 3, "{text:.20}: matrices");
            let requests = (0..inputs.len()).filter(|&index| index != 2);
            for (request, rows) in requests.zip(&values) {
                let got = scores.request(request).expect("scored");
                let want = expected(&fixture, rows, data_type);
                let same = got.len() == want.len() && got.iter().zip(&want).all(|(a, b)| a.to_bits() == b.to_bits());
                assert!(
                    same,
                    "{data_type:?} raw {raw_score} request {request}: {got:?} != {want:?}"
                );
            }
        }
    }

    #[test]
    fn a_failed_matrix_fails_only_its_requests() {
        let mut fixture = fixture(REGRESSION, DataType::Fp64, false);
        // Narrower than the model: the predictor panics on this matrix's rows.
        fixture.settings.num_features = 2;
        let data = bytes(&rows(300, 2, 1), DataType::Fp64);
        let shape = [300, 2];
        let inputs = vec![check_input(
            &fixture.settings,
            tensor("x", DataType::Fp64, &shape, vec![&data]),
        )];
        let scores = score(&fixture.predictor, &fixture.settings, &inputs, &mut Scratch::default());
        let error = scores.request(0).expect_err("panicked");
        assert_eq!(error.kind(), "panic");
        assert!(error.to_string().contains("the model reads 5 features"), "{error}");
    }

    #[test]
    fn scores_write_as_f64_or_f32_at_any_alignment() {
        let scores = [0.1, -2.5e300, f64::NAN, 1.0 / 3.0];
        let mut out = [0_u8; 33];
        write_scores(&scores, DataType::Fp64, &mut out[1..]);
        let back: Vec<f64> = out[1..]
            .as_chunks::<8>()
            .0
            .iter()
            .map(|bytes| f64::from_ne_bytes(*bytes))
            .collect();
        assert!(
            back.iter().zip(&scores).all(|(a, b)| a.to_bits() == b.to_bits()),
            "f64: {back:?}"
        );
        let mut out = [0_u8; 16];
        write_scores(&scores, DataType::Fp32, &mut out);
        let back: Vec<f32> = out
            .as_chunks::<4>()
            .0
            .iter()
            .map(|bytes| f32::from_ne_bytes(*bytes))
            .collect();
        let cast: Vec<f32> = scores.iter().map(|&score| score as f32).collect();
        assert!(
            back.iter().zip(&cast).all(|(a, b)| a.to_bits() == b.to_bits()),
            "f32: {back:?}"
        );
    }
}
