//! LightGBM 3.3.5's model file: `GBDT::SaveModelToString` with `Tree::ToString` for each tree, followed by the
//! `pandas_categorical:` line LightGBM's Python `Booster.model_to_string()` appends, and the reverse
//! (`GBDT::LoadModelFromString`, the `Tree(const char*)` constructor, `_load_pandas_categorical`).
//!
//! Numbers print as LightGBM prints them: `threshold`, `leaf_value` and `leaf_weight` with `{:.17g}`
//! (`ArrayToString<true>`, exact round trip), `split_gain`, `internal_value`, `internal_weight` and `shrinkage` with
//! 6 significant digits (`{:g}` and the stream default). A one-leaf tree writes empty `leaf_weight` and `leaf_count`:
//! LightGBM's string constructor reads only `leaf_value` and `shrinkage` of such a tree, so that is what every
//! reloaded booster writes, and `lgb.train` returns a reloaded booster.
//!
//! Parsing reads LightGBM 3.x and 4.x files. Header lines silvic has no use for (`monotone_constraints`, `version`,
//! ...) are skipped, and `feature_importances:` is recomputed from the trees, as LightGBM does. The `parameters:`
//! block is kept line for line, so writing a parsed 3.3.5 file reproduces it byte for byte. Trees are checked to be
//! well formed (every index in range, every node and leaf reached once from the root) so prediction cannot panic or
//! loop on a corrupt file.

use std::cmp::Reverse;
use std::fmt::{self, Write as _};
use std::iter::Peekable;
use std::str::{FromStr, Lines};

use crate::config::{fmt_g, format_g};
use crate::error::{Result, SilvicError};
use crate::model::{ImportanceType, Model, ModelObjective};
use crate::tree::{Tree, CATEGORICAL_MASK};

/// `kModelVersion`.
const MODEL_VERSION: &str = "v3";
/// Significant digits of `ArrayToString<true>`: enough for any double to read back unchanged.
const HIGH_PRECISION: usize = 17;
/// Significant digits of `ArrayToString<false>` (`{:g}`) and of a default `std::stringstream`.
const DEFAULT_PRECISION: usize = 6;
const PANDAS_CATEGORICAL_KEY: &str = "pandas_categorical:";
const END_OF_TREES: &str = "end of trees";
/// LightGBM objective names (`ParseObjectiveAlias`) that are L2 regression.
const REGRESSION_NAMES: &[&str] = &[
    "regression",
    "regression_l2",
    "l2",
    "mean_squared_error",
    "mse",
    "l2_root",
    "root_mean_squared_error",
    "rmse",
];
const MULTICLASS_NAMES: &[&str] = &["multiclass", "softmax", "multiclassova", "multiclass_ova", "ova", "ovr"];

impl Model {
    /// The model file LightGBM 3.3.5's Python `Booster.model_to_string()` returns, byte for byte.
    pub fn to_lightgbm_string(&self) -> String {
        let trees: Vec<String> = self
            .trees
            .iter()
            .enumerate()
            .map(|(i, tree)| tree_block(i, tree))
            .collect();
        let mut out = String::with_capacity(trees.iter().map(String::len).sum::<usize>() + 8192);
        self.write_header(&mut out, &trees);
        for tree in &trees {
            out.push_str(tree);
        }
        out.push_str(END_OF_TREES);
        out.push('\n');
        self.write_importances(&mut out);
        self.write_parameters(&mut out);
        out.push('\n');
        out.push_str(PANDAS_CATEGORICAL_KEY);
        out.push_str(self.pandas_categorical.as_deref().unwrap_or("null"));
        out.push('\n');
        out
    }

    /// Parses a LightGBM v3 model string (written by silvic or by LightGBM 3.x/4.x).
    pub fn from_lightgbm_string(text: &str) -> Result<Model> {
        let mut lines = text.lines().peekable();
        let header = Header::parse(&mut lines)?;
        let trees = parse_trees(&mut lines, header.feature_names.len())?;
        if let Some(sizes) = header.tree_sizes {
            if sizes != trees.len() {
                return Err(SilvicError::model(format!(
                    "tree_sizes lists {sizes} trees, the file has {}",
                    trees.len()
                )));
            }
        }
        let mut model = Model {
            objective: header.objective,
            feature_names: header.feature_names,
            feature_infos: header.feature_infos,
            trees,
            parameters: Vec::new(),
            pandas_categorical: None,
        };
        read_footer(lines, &mut model)?;
        Ok(model)
    }

    fn write_header(&self, out: &mut String, trees: &[String]) {
        out.push_str("tree\n");
        push_fmt(out, format_args!("version={MODEL_VERSION}\n"));
        out.push_str("num_class=1\nnum_tree_per_iteration=1\nlabel_index=0\n");
        push_fmt(
            out,
            format_args!("max_feature_idx={}\n", self.num_features() as i64 - 1),
        );
        out.push_str("objective=");
        match &self.objective {
            ModelObjective::Regression => out.push_str("regression"),
            ModelObjective::Binary { sigmoid } => push_fmt(out, format_args!("binary sigmoid:{}", fmt_g(*sigmoid))),
            ModelObjective::LambdaRank => out.push_str("lambdarank"),
        }
        out.push('\n');
        push_fmt(out, format_args!("feature_names={}\n", self.feature_names.join(" ")));
        push_fmt(out, format_args!("feature_infos={}\n", self.feature_infos.join(" ")));
        push_list(out, "tree_sizes", trees, |out, tree| {
            push_fmt(out, format_args!("{}", tree.len()))
        });
        out.push('\n');
    }

    /// `feature_importances:`: split counts above zero, stable-sorted by count descending.
    fn write_importances(&self, out: &mut String) {
        let mut counts: Vec<(usize, &str)> = self
            .feature_importance(ImportanceType::Split)
            .iter()
            .zip(&self.feature_names)
            .map(|(&count, name)| (count as usize, name.as_str()))
            .filter(|&(count, _)| count > 0)
            .collect();
        counts.sort_by_key(|&(count, _)| Reverse(count));
        out.push_str("\nfeature_importances:\n");
        for (count, name) in counts {
            push_fmt(out, format_args!("{name}={count}\n"));
        }
    }

    /// LightGBM writes the block only when it has parameters (a trained booster always does).
    fn write_parameters(&self, out: &mut String) {
        if self.parameters.is_empty() {
            return;
        }
        out.push_str("\nparameters:\n");
        for (key, value) in &self.parameters {
            push_fmt(out, format_args!("[{key}: {value}]\n"));
        }
        out.push_str("\nend of parameters\n");
    }
}

/// `Tree=<i>\n`, `Tree::ToString`, and the newline `SaveModelToString` adds: the unit `tree_sizes` measures.
fn tree_block(index: usize, tree: &Tree) -> String {
    let mut out = String::with_capacity(64 * tree.num_leaves + 256);
    push_fmt(&mut out, format_args!("Tree={index}\n"));
    write_tree(&mut out, tree);
    out.push('\n');
    out
}

/// `Tree::ToString`; like `ArrayToString(values, n)`, a list prints at most `n` entries.
fn write_tree(out: &mut String, tree: &Tree) {
    let leaves = tree.num_leaves;
    let nodes = leaves.saturating_sub(1);
    let leaf_stats = if leaves > 1 { leaves } else { 0 };
    push_fmt(out, format_args!("num_leaves={leaves}\nnum_cat={}\n", tree.num_cat));
    push_list(out, "split_feature", head(&tree.split_feature, nodes), push_display);
    push_list(out, "split_gain", head(&tree.split_gain, nodes), |out, &gain| {
        push_g(out, f64::from(gain))
    });
    push_list(out, "threshold", head(&tree.threshold, nodes), push_exact);
    push_list(out, "decision_type", head(&tree.decision_type, nodes), push_display);
    push_list(out, "left_child", head(&tree.left_child, nodes), push_display);
    push_list(out, "right_child", head(&tree.right_child, nodes), push_display);
    push_list(out, "leaf_value", head(&tree.leaf_value, leaves), push_exact);
    push_list(out, "leaf_weight", head(&tree.leaf_weight, leaf_stats), push_exact);
    push_list(out, "leaf_count", head(&tree.leaf_count, leaf_stats), push_display);
    push_list(out, "internal_value", head(&tree.internal_value, nodes), |out, &v| {
        push_g(out, v)
    });
    push_list(out, "internal_weight", head(&tree.internal_weight, nodes), |out, &v| {
        push_g(out, v)
    });
    push_list(out, "internal_count", head(&tree.internal_count, nodes), push_display);
    if tree.num_cat > 0 {
        push_list(
            out,
            "cat_boundaries",
            head(&tree.cat_boundaries, tree.num_cat + 1),
            push_display,
        );
        push_list(out, "cat_threshold", &tree.cat_threshold, push_display);
    }
    out.push_str("is_linear=0\n");
    push_fmt(out, format_args!("shrinkage={}\n\n", fmt_g(tree.shrinkage)));
}

fn head<T>(values: &[T], n: usize) -> &[T] {
    &values[..n.min(values.len())]
}

/// `key=v1 v2 ...\n`.
fn push_list<T>(out: &mut String, key: &str, values: &[T], push: impl Fn(&mut String, &T)) {
    out.push_str(key);
    out.push('=');
    for (i, value) in values.iter().enumerate() {
        if i > 0 {
            out.push(' ');
        }
        push(out, value);
    }
    out.push('\n');
}

fn push_fmt(out: &mut String, args: fmt::Arguments<'_>) {
    out.write_fmt(args).expect("writing to a String cannot fail");
}

fn push_display<T: fmt::Display>(out: &mut String, value: &T) {
    push_fmt(out, format_args!("{value}"));
}

/// `{:.17g}`.
fn push_exact(out: &mut String, value: &f64) {
    out.push_str(&format_g(*value, HIGH_PRECISION));
}

/// `{:g}`.
fn push_g(out: &mut String, value: f64) {
    out.push_str(&format_g(value, DEFAULT_PRECISION));
}

/// The header lines `LoadModelFromString` reads before the first tree.
struct Header {
    objective: ModelObjective,
    feature_names: Vec<String>,
    feature_infos: Vec<String>,
    /// Trees `tree_sizes` lists, when the file has the line.
    tree_sizes: Option<usize>,
}

impl Header {
    fn parse(lines: &mut Peekable<Lines<'_>>) -> Result<Header> {
        let mut entries: Vec<(&str, &str)> = Vec::new();
        while let Some(&line) = lines.peek() {
            if line.starts_with("Tree=") || line == END_OF_TREES {
                break;
            }
            lines.next();
            if !line.is_empty() {
                entries.push(line.split_once('=').unwrap_or((line, "")));
            }
        }
        let get = |key: &str| entries.iter().rev().find(|(k, _)| *k == key).map(|&(_, value)| value);
        let require = |key: &str| get(key).ok_or_else(|| SilvicError::model(format!("no {key} line")));

        let num_class: i32 = parse_number(require("num_class")?, "num_class")?;
        let per_iteration: i32 =
            get("num_tree_per_iteration").map_or(Ok(num_class), |v| parse_number(v, "num_tree_per_iteration"))?;
        if num_class != 1 || per_iteration != 1 {
            return Err(SilvicError::unsupported(format!(
                "multiclass models (num_class={num_class}, num_tree_per_iteration={per_iteration})"
            )));
        }
        if get("average_output").is_some() {
            return Err(SilvicError::unsupported("average_output (random forest) models"));
        }
        let max_feature_idx: i64 = parse_number(require("max_feature_idx")?, "max_feature_idx")?;
        let num_features = max_feature_idx
            .checked_add(1)
            .and_then(|value| usize::try_from(value).ok())
            .ok_or_else(|| SilvicError::model(format!("max_feature_idx={max_feature_idx}")))?;
        let objective = parse_objective(get("objective").ok_or_else(|| {
            SilvicError::unsupported("models without an objective (trained with a custom objective)")
        })?)?;
        Ok(Header {
            objective,
            feature_names: split_words(require("feature_names")?, "feature_names", num_features)?,
            feature_infos: split_words(require("feature_infos")?, "feature_infos", num_features)?,
            tree_sizes: get("tree_sizes").map(|sizes| sizes.split(' ').filter(|s| !s.is_empty()).count()),
        })
    }
}

/// `Common::Split(value, ' ')`, which drops empty pieces.
fn split_words(value: &str, key: &str, expected: usize) -> Result<Vec<String>> {
    let words: Vec<String> = value.split(' ').filter(|s| !s.is_empty()).map(str::to_string).collect();
    if words.len() != expected {
        return Err(SilvicError::model(format!(
            "{key} has {} entries, max_feature_idx says {expected}",
            words.len()
        )));
    }
    Ok(words)
}

/// `ObjectiveFunction::CreateObjectiveFunction` on an `objective=` line, for the objectives silvic predicts.
fn parse_objective(line: &str) -> Result<ModelObjective> {
    let mut tokens = line.split(' ').filter(|s| !s.is_empty());
    let name = tokens.next().unwrap_or("");
    let args: Vec<&str> = tokens.collect();
    if REGRESSION_NAMES.contains(&name) {
        if args.contains(&"sqrt") {
            return Err(SilvicError::unsupported("regression with reg_sqrt"));
        }
        return Ok(ModelObjective::Regression);
    }
    match name {
        "binary" => {
            let sigmoid = args.iter().find_map(|arg| arg.strip_prefix("sigmoid:"));
            match sigmoid.map(|value| parse_number::<f64>(value, "sigmoid")).transpose()? {
                Some(sigmoid) if sigmoid > 0.0 => Ok(ModelObjective::Binary { sigmoid }),
                _ => Err(SilvicError::model(format!(
                    "binary objective needs sigmoid > 0, got `{line}`"
                ))),
            }
        }
        "lambdarank" => Ok(ModelObjective::LambdaRank),
        _ if MULTICLASS_NAMES.contains(&name) => {
            Err(SilvicError::unsupported(format!("multiclass objective `{line}`")))
        }
        _ => Err(SilvicError::unsupported(format!("objective `{line}`"))),
    }
}

/// The `Tree=<i>` blocks, up to the first line that does not start one.
fn parse_trees(lines: &mut Peekable<Lines<'_>>, num_features: usize) -> Result<Vec<Tree>> {
    let mut trees = Vec::new();
    while let Some(&line) = lines.peek() {
        if line.is_empty() {
            lines.next();
            continue;
        }
        if !line.starts_with("Tree=") {
            break;
        }
        lines.next();
        let fields = TreeFields::read(lines);
        let tree = parse_tree(&fields, num_features).map_err(|error| match error {
            SilvicError::Model(message) => SilvicError::model(format!("tree {}: {message}", trees.len())),
            other => other,
        })?;
        trees.push(tree);
    }
    Ok(trees)
}

/// The `key=value` lines of one tree, up to the blank line that ends it.
struct TreeFields<'a> {
    entries: Vec<(&'a str, &'a str)>,
}

impl<'a> TreeFields<'a> {
    fn read(lines: &mut Peekable<Lines<'a>>) -> TreeFields<'a> {
        let mut entries = Vec::new();
        while let Some(line) = lines.next_if(|line| !line.is_empty()) {
            entries.push(line.split_once('=').unwrap_or((line, "")));
        }
        TreeFields { entries }
    }

    fn get(&self, key: &str) -> Option<&'a str> {
        self.entries
            .iter()
            .rev()
            .find(|(k, _)| *k == key)
            .map(|&(_, value)| value)
    }

    fn require(&self, key: &str) -> Result<&'a str> {
        self.get(key)
            .ok_or_else(|| SilvicError::model(format!("no {key} field")))
    }

    /// `StringToArray(value, n)`: exactly `n` numbers.
    fn list<T: FromStr>(&self, key: &str, n: usize) -> Result<Vec<T>> {
        parse_list(self.require(key)?, key, n)
    }

    /// A list LightGBM fills with zeros when the field is absent.
    fn list_or_default<T: FromStr + Default + Clone>(&self, key: &str, n: usize) -> Result<Vec<T>> {
        match self.get(key) {
            Some(value) => parse_list(value, key, n),
            None => Ok(vec![T::default(); n]),
        }
    }
}

fn parse_list<T: FromStr>(value: &str, key: &str, n: usize) -> Result<Vec<T>> {
    let values = value
        .split(' ')
        .filter(|s| !s.is_empty())
        .map(|s| parse_number(s, key))
        .collect::<Result<Vec<T>>>()?;
    if values.len() != n {
        return Err(SilvicError::model(format!(
            "{key} has {} values, expected {n}",
            values.len()
        )));
    }
    Ok(values)
}

fn parse_number<T: FromStr>(text: &str, what: &str) -> Result<T> {
    text.trim()
        .parse()
        .map_err(|_| SilvicError::model(format!("{what}: cannot parse `{text}`")))
}

/// `Tree::Tree(const char*)`, then the training-only `leaf_parent`/`leaf_depth` rebuilt from the child arrays.
fn parse_tree(fields: &TreeFields<'_>, num_features: usize) -> Result<Tree> {
    let num_leaves: usize = parse_number(fields.require("num_leaves")?, "num_leaves")?;
    if num_leaves == 0 {
        return Err(SilvicError::model("num_leaves=0"));
    }
    if fields.get("is_linear").is_some_and(|value| value.trim() != "0") {
        return Err(SilvicError::unsupported("linear trees"));
    }
    let num_cat: usize = parse_number(fields.require("num_cat")?, "num_cat")?;
    let leaf_value = fields.list("leaf_value", num_leaves)?;
    let shrinkage = fields
        .get("shrinkage")
        .map_or(Ok(1.0), |value| parse_number(value, "shrinkage"))?;
    let mut tree = Tree::new();
    tree.leaf_value = leaf_value;
    tree.shrinkage = shrinkage;
    if num_leaves == 1 {
        return Ok(tree);
    }
    let nodes = num_leaves - 1;
    tree.num_leaves = num_leaves;
    tree.num_cat = num_cat;
    tree.left_child = fields.list("left_child", nodes)?;
    tree.right_child = fields.list("right_child", nodes)?;
    tree.split_feature = fields.list("split_feature", nodes)?;
    tree.threshold = fields.list("threshold", nodes)?;
    tree.split_gain = fields.list_or_default("split_gain", nodes)?;
    tree.internal_count = fields.list_or_default("internal_count", nodes)?;
    tree.internal_value = fields.list_or_default("internal_value", nodes)?;
    tree.internal_weight = fields.list_or_default("internal_weight", nodes)?;
    tree.leaf_weight = fields.list_or_default("leaf_weight", num_leaves)?;
    tree.leaf_count = fields.list_or_default("leaf_count", num_leaves)?;
    tree.decision_type = fields.list_or_default("decision_type", nodes)?;
    if num_cat > 0 {
        let boundaries = num_cat
            .checked_add(1)
            .ok_or_else(|| SilvicError::model(format!("num_cat={num_cat}")))?;
        tree.cat_boundaries = fields.list("cat_boundaries", boundaries)?;
        let words = usize::try_from(*tree.cat_boundaries.last().expect("boundaries is nonzero"))
            .map_err(|_| SilvicError::model("negative cat_boundaries"))?;
        tree.cat_threshold = fields.list("cat_threshold", words)?;
    }
    check_splits(&tree, num_features)?;
    link_leaves(&mut tree)?;
    Ok(tree)
}

/// Every split reads a real feature, and every categorical split names a bitset inside `cat_threshold`.
fn check_splits(tree: &Tree, num_features: usize) -> Result<()> {
    if tree.cat_boundaries.windows(2).any(|w| w[0] < 0 || w[0] > w[1]) {
        return Err(SilvicError::model("cat_boundaries are not ascending"));
    }
    for (node, &feature) in tree.split_feature.iter().enumerate() {
        if usize::try_from(feature).map_or(true, |f| f >= num_features) {
            return Err(SilvicError::model(format!("node {node} splits on feature {feature}")));
        }
        let categorical = tree.num_cat > 0 && tree.decision_type[node] & CATEGORICAL_MASK != 0;
        let threshold = tree.threshold[node];
        if categorical && !(threshold >= 0.0 && (threshold as usize) < tree.num_cat) {
            return Err(SilvicError::model(format!(
                "node {node} names categorical bitset {threshold}"
            )));
        }
    }
    Ok(())
}

/// Fills `leaf_parent` and `leaf_depth`, failing unless the child arrays form one tree rooted at node 0. A child node
/// always has a higher index than its parent (`Tree::Split` numbering), which rules out cycles.
fn link_leaves(tree: &mut Tree) -> Result<()> {
    let nodes = tree.num_leaves - 1;
    let mut node_depth: Vec<Option<i32>> = vec![None; nodes];
    node_depth[0] = Some(0);
    tree.leaf_parent = vec![-1; tree.num_leaves];
    tree.leaf_depth = vec![-1; tree.num_leaves];
    for node in 0..nodes {
        let depth = node_depth[node].ok_or_else(|| SilvicError::model(format!("node {node} is unreachable")))?;
        for child in [tree.left_child[node], tree.right_child[node]] {
            let bad = || SilvicError::model(format!("node {node} has an invalid or repeated child {child}"));
            if child >= 0 {
                let child = child as usize;
                if child <= node || child >= nodes || node_depth[child].is_some() {
                    return Err(bad());
                }
                node_depth[child] = Some(depth + 1);
            } else {
                let leaf = !child as usize;
                if leaf >= tree.num_leaves || tree.leaf_depth[leaf] >= 0 {
                    return Err(bad());
                }
                tree.leaf_parent[leaf] = node as i32;
                tree.leaf_depth[leaf] = depth + 1;
            }
        }
    }
    match tree.leaf_depth.iter().position(|&depth| depth < 0) {
        Some(leaf) => Err(SilvicError::model(format!("leaf {leaf} is unreachable"))),
        None => Ok(()),
    }
}

/// Fills `parameters` from the `parameters:` block and `pandas_categorical` from its line (`None` for `null`).
fn read_footer(lines: Peekable<Lines<'_>>, model: &mut Model) -> Result<()> {
    let mut in_parameters = false;
    for line in lines {
        if in_parameters {
            if line == "end of parameters" {
                in_parameters = false;
            } else if !line.is_empty() {
                model.parameters.push(parse_parameter(line)?);
            }
        } else if line == "parameters:" {
            in_parameters = true;
        } else if let Some(json) = line.strip_prefix(PANDAS_CATEGORICAL_KEY) {
            model.pandas_categorical = (json != "null").then(|| json.to_string());
        }
    }
    Ok(())
}

/// `[key: value]`.
fn parse_parameter(line: &str) -> Result<(String, String)> {
    line.strip_prefix('[')
        .and_then(|inner| inner.strip_suffix(']'))
        .and_then(|inner| inner.split_once(": "))
        .map(|(key, value)| (key.to_string(), value.to_string()))
        .ok_or_else(|| SilvicError::model(format!("parameter line `{line}` is not `[key: value]`")))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tree::{MissingType, SplitChildren};

    fn children(left_value: f64, right_value: f64, gain: f32) -> SplitChildren {
        SplitChildren {
            left_value,
            right_value,
            left_count: 30,
            right_count: 70,
            left_weight: 7.5,
            right_weight: 17.25,
            gain,
        }
    }

    fn model(trees: Vec<Tree>) -> Model {
        Model {
            objective: ModelObjective::Binary { sigmoid: 1.0 },
            feature_names: vec!["a".into(), "b".into(), "c".into()],
            feature_infos: vec!["[-1:1]".into(), "1:3:40".into(), "none".into()],
            trees,
            parameters: vec![("boosting".into(), "gbdt".into()), ("label_gain".into(), String::new())],
            pandas_categorical: None,
        }
    }

    #[test]
    fn writes_the_lightgbm_layout() {
        let mut tree = Tree::new();
        tree.split_numerical(0, 0, 0.1, true, MissingType::NaN, &children(-0.5, 0.25, 12.5));
        tree.split_categorical(1, 1, &[1, 40], MissingType::None, &children(0.125, 1.0 / 3.0, 3.0));
        tree.shrink(0.1);
        let text = model(vec![tree, Tree::new()]).to_lightgbm_string();
        let expected_tree = "Tree=0\nnum_leaves=3\nnum_cat=1\nsplit_feature=0 1\nsplit_gain=12.5 3\n\
            threshold=0.10000000000000001 0\ndecision_type=10 1\nleft_child=-1 -2\nright_child=1 -3\n\
            leaf_value=-0.050000000000000003 0.012500000000000001 0.033333333333333333\n\
            leaf_weight=7.5 7.5 17.25\nleaf_count=30 30 70\ninternal_value=0 0.025\ninternal_weight=0 17.25\n\
            internal_count=100 100\ncat_boundaries=0 2\ncat_threshold=2 256\nis_linear=0\nshrinkage=0.1\n\n\n";
        let single_leaf = "Tree=1\nnum_leaves=1\nnum_cat=0\nsplit_feature=\nsplit_gain=\nthreshold=\n\
            decision_type=\nleft_child=\nright_child=\nleaf_value=0\nleaf_weight=\nleaf_count=\n\
            internal_value=\ninternal_weight=\ninternal_count=\nis_linear=0\nshrinkage=1\n\n\n";
        let expected = format!(
            "tree\nversion=v3\nnum_class=1\nnum_tree_per_iteration=1\nlabel_index=0\nmax_feature_idx=2\n\
             objective=binary sigmoid:1\nfeature_names=a b c\nfeature_infos=[-1:1] 1:3:40 none\n\
             tree_sizes={} {}\n\n{expected_tree}{single_leaf}end of trees\n\nfeature_importances:\na=1\nb=1\n\n\
             parameters:\n[boosting: gbdt]\n[label_gain: ]\n\nend of parameters\n\npandas_categorical:null\n",
            expected_tree.len(),
            single_leaf.len()
        );
        assert_eq!(text, expected);
        assert_eq!(
            Model::from_lightgbm_string(&text).expect("parse").to_lightgbm_string(),
            text
        );
    }

    #[test]
    fn importances_sort_stably_by_count() {
        let mut tree = Tree::new();
        tree.split_numerical(0, 2, 0.0, false, MissingType::None, &children(0.0, 0.0, 1.0));
        tree.split_numerical(0, 0, 0.0, false, MissingType::None, &children(0.0, 0.0, 1.0));
        tree.split_numerical(1, 2, 1.0, false, MissingType::None, &children(0.0, 0.0, 1.0));
        tree.split_numerical(2, 1, 1.0, false, MissingType::None, &children(0.0, 0.0, 0.0));
        let text = model(vec![tree]).to_lightgbm_string();
        assert!(
            text.contains("\nfeature_importances:\nc=2\na=1\n\nparameters:"),
            "{text}"
        );
    }

    #[test]
    fn parses_header_variants() {
        let mut parsed = model(vec![Tree::new()]);
        parsed.objective = ModelObjective::Regression;
        parsed.pandas_categorical = Some(r#"[["x", "y"]]"#.into());
        let text = parsed.to_lightgbm_string();
        assert!(text.ends_with("\npandas_categorical:[[\"x\", \"y\"]]\n"));
        assert_eq!(Model::from_lightgbm_string(&text).expect("parse"), parsed);
        let alias = text.replace("objective=regression", "objective=mse\nmonotone_constraints=0 0 0");
        assert_eq!(
            Model::from_lightgbm_string(&alias).expect("alias").objective,
            ModelObjective::Regression
        );
        let ranking = text.replace("objective=regression", "objective=lambdarank");
        assert_eq!(
            Model::from_lightgbm_string(&ranking).expect("lambdarank").objective,
            ModelObjective::LambdaRank
        );
        let binary = text.replace("objective=regression", "objective=binary sigmoid:0.7");
        let binary = Model::from_lightgbm_string(&binary).expect("binary");
        assert_eq!(binary.objective, ModelObjective::Binary { sigmoid: 0.7 });
    }

    #[test]
    fn rejects_unsupported_models() {
        let text = model(vec![Tree::new()]).to_lightgbm_string();
        let objective = "objective=binary sigmoid:1";
        for (from, to) in [
            ("num_class=1", "num_class=3"),
            (objective, "objective=multiclass num_class:3"),
            (objective, "objective=regression_l1"),
            (objective, "objective=regression sqrt"),
            (objective, "objective=rank_xendcg"),
            ("label_index=0", "label_index=0\naverage_output"),
            ("is_linear=0", "is_linear=1"),
        ] {
            let error = Model::from_lightgbm_string(&text.replacen(from, to, 1)).expect_err(to);
            assert!(matches!(error, SilvicError::Unsupported(_)), "{to}: {error}");
        }
        let error = Model::from_lightgbm_string(&text.replace(&format!("{objective}\n"), "")).expect_err("none");
        assert!(matches!(error, SilvicError::Unsupported(_)), "no objective: {error}");
    }

    #[test]
    fn rejects_malformed_trees() {
        let mut tree = Tree::new();
        tree.split_numerical(0, 0, 0.0, true, MissingType::None, &children(1.0, 2.0, 1.0));
        tree.split_numerical(1, 1, 0.0, true, MissingType::None, &children(1.0, 2.0, 1.0));
        let text = model(vec![tree]).to_lightgbm_string();
        for (from, to) in [
            ("left_child=-1 -2", "left_child=-1 0"),
            ("left_child=-1 -2", "left_child=-1 -1"),
            ("right_child=1 -3", "right_child=1 -4"),
            ("split_feature=0 1", "split_feature=0 3"),
            ("leaf_value=", "leaf_value=1 "),
            ("sigmoid:1", "sigmoid:0"),
            ("[boosting: gbdt]", "boosting: gbdt"),
            ("max_feature_idx=2", "max_feature_idx=3"),
            ("max_feature_idx=2", "max_feature_idx=9223372036854775807"),
            ("num_cat=0", "num_cat=18446744073709551615"),
        ] {
            assert!(text.contains(from), "{from}");
            let error = Model::from_lightgbm_string(&text.replacen(from, to, 1)).expect_err(to);
            assert!(matches!(error, SilvicError::Model(_)), "{to}: {error}");
        }
    }
}
