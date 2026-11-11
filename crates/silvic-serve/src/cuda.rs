//! Cooperative row/tree traversal with shared leaf tiles and ordered f64 or opt-in f32 accumulation on CUDA.

use bytemuck::{Pod, Zeroable};
use silvic_core::data::{DenseData, DenseMatrix, Layout};
use silvic_core::model::{Model, ModelObjective};
use silvic_core::tree::CATEGORICAL_MASK;
use silvic_core::{Result, SilvicError};
use silvic_gpu::cuda::CudaDevice;
use silvic_gpu::{Arg, GpuDevice};

use crate::compile::Shape;

type Buffer = <CudaDevice as GpuDevice>::Buffer;

#[derive(Clone, Copy, Pod, Zeroable)]
#[repr(C)]
struct Node {
    threshold: f64,
    cat_start: u64,
    left: i32,
    right: i32,
    feature: u32,
    decision: u32,
    cat_len: u64,
}

#[derive(Clone, Copy, Pod, Zeroable)]
#[repr(C)]
struct NodeF32 {
    threshold: f32,
    padding: u32,
    cat_start: u64,
    left: i32,
    right: i32,
    feature: u32,
    decision: u32,
    cat_len: u64,
}

trait Precision: Pod + Default {
    type Node: Pod + Zeroable;
    const KERNELS: [&'static str; 2];
    fn node(node: Node) -> Self::Node;
    fn leaf(value: f64) -> Self;
    fn transform(value: Self, sigmoid: Self) -> Self;
}

impl Precision for f64 {
    type Node = Node;
    const KERNELS: [&'static str; 2] = ["predict_f32", "predict_f64"];

    fn node(node: Node) -> Node {
        node
    }

    fn leaf(value: f64) -> Self {
        value
    }

    fn transform(value: Self, sigmoid: Self) -> Self {
        1.0 / (1.0 + (-sigmoid * value).exp())
    }
}

impl Precision for f32 {
    type Node = NodeF32;
    const KERNELS: [&'static str; 2] = ["predict_f32_f32", "predict_f32_f64"];

    fn node(node: Node) -> NodeF32 {
        let rounded = node.threshold as f32;
        let threshold = if f64::from(rounded) > node.threshold {
            rounded.next_down()
        } else {
            rounded
        };
        NodeF32 {
            threshold,
            padding: 0,
            cat_start: node.cat_start,
            left: node.left,
            right: node.right,
            feature: node.feature,
            decision: node.decision,
            cat_len: node.cat_len,
        }
    }

    fn leaf(value: f64) -> Self {
        value as f32
    }

    fn transform(value: Self, sigmoid: Self) -> Self {
        1.0f32 / (1.0f32 + (-sigmoid * value).exp())
    }
}

#[derive(Clone, Copy, Pod, Zeroable)]
#[repr(C)]
struct Tree {
    nodes: u64,
    leaves: u64,
    num_nodes: u64,
}

#[derive(Clone, Copy, Pod, Zeroable)]
#[repr(C)]
struct Params {
    rows: u64,
    cols: u64,
    trees: u64,
    column_major: u64,
}

struct PackedForest<P: Precision> {
    trees: Vec<Tree>,
    nodes: Vec<P::Node>,
    leaves: Vec<P>,
    categories: Vec<u32>,
}

impl<P: Precision> PackedForest<P> {
    fn new(model: &Model) -> Result<Self> {
        let mut packed = Self {
            trees: Vec::with_capacity(model.trees.len()),
            nodes: Vec::new(),
            leaves: Vec::new(),
            categories: Vec::new(),
        };
        for (index, tree) in model.trees.iter().enumerate() {
            let shape = Shape::of(tree, model.num_features())
                .map_err(|message| SilvicError::model(format!("tree {index}: {message}")))?;
            let node_start = packed.nodes.len();
            let num_nodes = tree.num_leaves.saturating_sub(1);
            packed.trees.push(Tree {
                nodes: node_start as u64,
                leaves: packed.leaves.len() as u64,
                num_nodes: num_nodes as u64,
            });
            let node_end = node_start
                .checked_add(num_nodes)
                .ok_or_else(|| SilvicError::model("CUDA forest node count overflows usize"))?;
            packed.nodes.resize(node_end, P::Node::zeroed());
            for (node, _, _) in shape.nodes {
                let categorical = tree.num_cat > 0 && tree.decision_type[node] & CATEGORICAL_MASK != 0;
                let (cat_start, cat_len) = if categorical {
                    let index = tree.threshold[node] as usize;
                    let start = tree.cat_boundaries[index] as usize;
                    let end = tree.cat_boundaries[index + 1] as usize;
                    (packed.categories.len() as u64 + start as u64, (end - start) as u64)
                } else {
                    (0, 0)
                };
                packed.nodes[node_start + node] = P::node(Node {
                    threshold: tree.threshold[node],
                    cat_start,
                    left: tree.left_child[node],
                    right: tree.right_child[node],
                    feature: tree.split_feature[node] as u32,
                    decision: (tree.decision_type[node] as u8 & !(CATEGORICAL_MASK as u8)) as u32
                        | u32::from(categorical),
                    cat_len,
                });
            }
            packed
                .leaves
                .extend(tree.leaf_value[..tree.num_leaves.max(1)].iter().copied().map(P::leaf));
            packed.categories.extend_from_slice(&tree.cat_threshold);
        }
        Ok(packed)
    }
}

struct Scratch {
    buffer: Buffer,
    capacity: usize,
}

impl Scratch {
    fn new(device: &CudaDevice) -> Result<Self> {
        Ok(Self {
            buffer: device.alloc(0)?,
            capacity: 0,
        })
    }

    fn reserve(&mut self, device: &CudaDevice, bytes: usize) -> Result<()> {
        if bytes > self.capacity {
            let capacity = bytes.checked_next_power_of_two().unwrap_or(bytes);
            self.buffer = device.alloc(capacity)?;
            self.capacity = capacity;
        }
        Ok(())
    }
}

fn upload<T: Pod>(device: &CudaDevice, values: &[T]) -> Result<Buffer> {
    let bytes = bytemuck::cast_slice(values);
    let buffer = device.alloc(bytes.len())?;
    device.upload(&buffer, 0, bytes)?;
    Ok(buffer)
}

/// Whether the guarded CUDA driver/NVRTC probe finds a device. Never selects a CPU fallback.
pub fn cuda_is_available() -> bool {
    silvic_gpu::cuda::is_available()
}

struct PredictorImpl<P: Precision> {
    device: CudaDevice,
    trees: Buffer,
    nodes: Buffer,
    leaves: Buffer,
    categories: Buffer,
    input: Scratch,
    output: Scratch,
    num_trees: usize,
    num_features: usize,
    objective: ModelObjective,
    precision: std::marker::PhantomData<P>,
}

impl<P: Precision> PredictorImpl<P> {
    fn new(model: &Model, ordinal: Option<usize>) -> Result<Self> {
        let packed = PackedForest::<P>::new(model)?;
        let mut device = CudaDevice::new(ordinal)?;
        device.compile(include_str!("cuda.cu"), &P::KERNELS)?;
        let trees = upload(&device, &packed.trees)?;
        let nodes = upload(&device, &packed.nodes)?;
        let leaves = upload(&device, &packed.leaves)?;
        let categories = upload(&device, &packed.categories)?;
        let input = Scratch::new(&device)?;
        let output = Scratch::new(&device)?;
        device.sync()?;
        Ok(Self {
            device,
            trees,
            nodes,
            leaves,
            categories,
            input,
            output,
            num_trees: model.trees.len(),
            num_features: model.num_features(),
            objective: model.objective.clone(),
            precision: std::marker::PhantomData,
        })
    }

    fn num_trees(&self) -> usize {
        self.num_trees
    }

    fn num_features(&self) -> usize {
        self.num_features
    }

    fn predict_raw(&mut self, rows: &DenseMatrix<'_>, num_trees: Option<usize>, out: &mut [P]) -> Result<()> {
        let elements = self.check(rows)?;
        if out.len() != rows.rows {
            return Err(SilvicError::config("CUDA prediction requires one output per row"));
        }
        if rows.rows == 0 {
            return Ok(());
        }
        let (kernel, input): (usize, &[u8]) = match rows.data {
            DenseData::F32(values) => (0, bytemuck::cast_slice(&values[..elements])),
            DenseData::F64(values) => (1, bytemuck::cast_slice(&values[..elements])),
        };
        self.input.reserve(&self.device, input.len())?;
        self.output.reserve(&self.device, std::mem::size_of_val(out))?;
        self.device.upload(&self.input.buffer, 0, input)?;
        let params = Params {
            rows: rows.rows as u64,
            cols: rows.cols as u64,
            trees: num_trees.unwrap_or(self.num_trees).min(self.num_trees) as u64,
            column_major: u64::from(matches!(rows.layout, Layout::ColumnMajor)),
        };
        let tree_warps = (params.trees.clamp(1, 8) as u32).min(self.device.max_threads_per_group() / 32);
        if tree_warps == 0 {
            return Err(SilvicError::device("CUDA prediction kernel supports no complete warp"));
        }
        let block = tree_warps * 32;
        let blocks = rows.rows.div_ceil(32).min(65_535) as u32;
        self.device.launch(
            kernel,
            [blocks, 1, 1],
            [block, 1, 1],
            &[
                Arg::Buffer(&self.input.buffer),
                Arg::Buffer(&self.trees),
                Arg::Buffer(&self.nodes),
                Arg::Buffer(&self.leaves),
                Arg::Buffer(&self.categories),
                Arg::Buffer(&self.output.buffer),
                Arg::Bytes(bytemuck::bytes_of(&params)),
            ],
        )?;
        self.device
            .download(&self.output.buffer, 0, bytemuck::cast_slice_mut(out))
    }

    fn predict(&mut self, rows: &DenseMatrix<'_>, raw_score: bool, num_trees: Option<usize>) -> Result<Vec<P>> {
        self.check(rows)?;
        let mut out = vec![P::default(); rows.rows];
        self.predict_raw(rows, num_trees, &mut out)?;
        if !raw_score {
            if let ModelObjective::Binary { sigmoid } = self.objective {
                let sigmoid = P::leaf(sigmoid);
                for value in &mut out {
                    *value = P::transform(*value, sigmoid);
                }
            }
        }
        Ok(out)
    }

    fn check(&self, rows: &DenseMatrix<'_>) -> Result<usize> {
        if rows.cols < self.num_features {
            return Err(SilvicError::config(format!(
                "the matrix has {} columns, the model reads {} features",
                rows.cols, self.num_features
            )));
        }
        let elements = rows
            .rows
            .checked_mul(rows.cols)
            .ok_or_else(|| SilvicError::config("CUDA prediction matrix dimensions overflow usize"))?;
        let available = match rows.data {
            DenseData::F32(values) => values.len(),
            DenseData::F64(values) => values.len(),
        };
        if elements > available || rows.rows > isize::MAX as usize / std::mem::size_of::<P>() {
            return Err(SilvicError::config(
                "CUDA prediction matrix storage is too short or output is too large",
            ));
        }
        Ok(elements)
    }
}

macro_rules! predictor_api {
    ($name:ident, $scalar:ty, $description:literal) => {
        #[doc = $description]
        ///
        /// Requires the `cuda` feature and a working CUDA device; errors never fall back to CPU scoring.
        /// The forest stays resident and input/output device buffers are reused on the predictor's stream.
        pub struct $name {
            inner: PredictorImpl<$scalar>,
        }

        impl $name {
            /// Validates and uploads `model` once, opening device `ordinal` (default 0).
            pub fn new(model: &Model, ordinal: Option<usize>) -> Result<Self> {
                Ok(Self {
                    inner: PredictorImpl::new(model, ordinal)?,
                })
            }

            pub fn num_trees(&self) -> usize {
                self.inner.num_trees()
            }

            pub fn num_features(&self) -> usize {
                self.inner.num_features()
            }

            /// Writes one raw score per row, using a clamped tree prefix (`None` means all trees).
            /// Accepts f32/f64 matrices in either layout without host conversion.
            /// Invalid dimensions, short storage and output length mismatches return errors.
            pub fn predict_raw(
                &mut self,
                rows: &DenseMatrix<'_>,
                num_trees: Option<usize>,
                out: &mut [$scalar],
            ) -> Result<()> {
                self.inner.predict_raw(rows, num_trees, out)
            }

            /// Scores rows, applying the objective in the output precision unless `raw_score` is true.
            pub fn predict(
                &mut self,
                rows: &DenseMatrix<'_>,
                raw_score: bool,
                num_trees: Option<usize>,
            ) -> Result<Vec<$scalar>> {
                self.inner.predict(rows, raw_score, num_trees)
            }
        }
    };
}

predictor_api!(CudaPredictor, f64, "An exact f64 CUDA predictor.");
predictor_api!(
    CudaPredictorF32,
    f32,
    "An approximate CUDA predictor with f32 thresholds, leaves, ordered sums and output. \
     Inputs are converted to f32 on the GPU before split decisions. Numerical thresholds round \
     toward negative infinity; categorical metadata retains the original integer indices."
);
