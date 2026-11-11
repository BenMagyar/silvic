use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{LazyLock, Mutex};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use silvic_core::{Result, SilvicError};

use crate::runtime::TensorSpec;

const ADAPTER: &str = include_str!("../compiler/compile.py");
static COMPILE_LOCK: Mutex<()> = Mutex::new(());
static WORK_ID: AtomicU64 = AtomicU64::new(0);
static COMPILER: LazyLock<std::result::Result<(PathBuf, String), String>> =
    LazyLock::new(|| detect_compiler().map_err(|error| error.to_string()));

pub(crate) struct Artifact {
    pub neff: PathBuf,
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
}

#[derive(Serialize)]
struct Request<'a> {
    entry: &'a str,
    target: &'a str,
    constants: &'a serde_json::Value,
    inputs: &'a [TensorSpec],
    outputs: &'a [TensorSpec],
}

#[derive(Deserialize)]
struct Port {
    name: String,
    port: String,
    dtype: String,
    shape: Vec<usize>,
}

#[derive(Deserialize)]
struct Manifest {
    compiler_version: String,
    target: String,
    neff: PathBuf,
    inputs: Vec<Port>,
    outputs: Vec<Port>,
}

fn command_error(output: std::process::Output, operation: &str) -> SilvicError {
    SilvicError::device(format!(
        "Neuron {operation} failed ({}): {}{}",
        output.status,
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    ))
}

fn compiler() -> Result<&'static (PathBuf, String)> {
    COMPILER.as_ref().map_err(|error| SilvicError::device(error.as_str()))
}

fn detect_compiler() -> Result<(PathBuf, String)> {
    let python = PathBuf::from(std::env::var_os("SILVIC_NEURON_PYTHON").unwrap_or_else(|| "python3".into()));
    let output = Command::new(&python)
        .args(["-c", ADAPTER, "--version"])
        .output()
        .map_err(|error| SilvicError::device(format!("start Neuron compiler Python {}: {error}", python.display())))?;
    if !output.status.success() {
        return Err(command_error(output, "compiler version query"));
    }
    let version = String::from_utf8(output.stdout)
        .map_err(|_| SilvicError::device("Neuron compiler returned a non-UTF8 version"))?;
    let version = version.trim();
    if version.is_empty() {
        return Err(SilvicError::device("Neuron compiler returned an empty version"));
    }
    Ok((python, version.to_owned()))
}

fn cache_root() -> Result<PathBuf> {
    if let Some(path) = std::env::var_os("SILVIC_NEURON_CACHE") {
        return Ok(PathBuf::from(path));
    }
    if let Some(path) = std::env::var_os("XDG_CACHE_HOME") {
        return Ok(PathBuf::from(path).join("silvic/neuron"));
    }
    let home = std::env::var_os("HOME")
        .ok_or_else(|| SilvicError::config("set SILVIC_NEURON_CACHE when HOME and XDG_CACHE_HOME are unset"))?;
    Ok(PathBuf::from(home).join(".cache/silvic/neuron"))
}

fn io_error(path: &Path, error: std::io::Error) -> SilvicError {
    SilvicError::device(format!("Neuron compilation cache {}: {error}", path.display()))
}

fn work_directory(root: &Path) -> Result<PathBuf> {
    loop {
        let id = WORK_ID.fetch_add(1, Ordering::Relaxed);
        let path = root.join(format!("build-{}-{id}", std::process::id()));
        match std::fs::create_dir(&path) {
            Ok(()) => return Ok(path),
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(error) => return Err(io_error(&path, error)),
        }
    }
}

fn ports(actual: Vec<Port>, wanted: &[TensorSpec]) -> Result<Vec<String>> {
    if actual.len() != wanted.len() {
        return Err(SilvicError::device(
            "Neuron compiler tensor count differs from requested signature",
        ));
    }
    actual
        .into_iter()
        .zip(wanted)
        .map(|(port, spec)| {
            let dtype = serde_json::to_value(spec.dtype)
                .map_err(|error| SilvicError::device(format!("serialize Neuron dtype: {error}")))?;
            if port.name != spec.name
                || port.shape != spec.shape
                || dtype.as_str() != Some(port.dtype.as_str())
                || port.port.is_empty()
                || port.port.as_bytes().contains(&0)
            {
                return Err(SilvicError::device(format!(
                    "Neuron compiler port differs from tensor {}",
                    spec.name
                )));
            }
            Ok(port.port)
        })
        .collect()
}

fn artifact(path: &Path, request: &Request<'_>, version: &str) -> Result<Artifact> {
    let bytes = std::fs::read(path).map_err(|error| io_error(path, error))?;
    let manifest: Manifest = serde_json::from_slice(&bytes)
        .map_err(|error| SilvicError::device(format!("Neuron compiler manifest {}: {error}", path.display())))?;
    if manifest.compiler_version != version || manifest.target != request.target {
        return Err(SilvicError::device(
            "Neuron executable compiler or target does not match its cache key",
        ));
    }
    let metadata = std::fs::metadata(&manifest.neff).map_err(|error| io_error(&manifest.neff, error))?;
    if !metadata.is_file() || metadata.len() == 0 {
        return Err(SilvicError::device("Neuron executable cache contains no NEFF"));
    }
    Ok(Artifact {
        neff: manifest.neff,
        inputs: ports(manifest.inputs, request.inputs)?,
        outputs: ports(manifest.outputs, request.outputs)?,
    })
}

pub(crate) fn compile(
    source: &str,
    entry: &str,
    constants: &serde_json::Value,
    inputs: &[TensorSpec],
    outputs: &[TensorSpec],
    target: &str,
) -> Result<Artifact> {
    for tensor in inputs.iter().chain(outputs) {
        tensor.bytes()?;
    }
    let _guard = COMPILE_LOCK
        .lock()
        .map_err(|_| SilvicError::device("Neuron compilation lock is poisoned"))?;
    let (python, version) = compiler()?;
    let request = Request {
        entry,
        target,
        constants,
        inputs,
        outputs,
    };
    let request_bytes = serde_json::to_vec(&request)
        .map_err(|error| SilvicError::config(format!("Neuron kernel signature: {error}")))?;
    let mut hash = Sha256::new();
    for part in [
        source.as_bytes(),
        ADAPTER.as_bytes(),
        &request_bytes,
        version.as_bytes(),
        python.as_os_str().as_encoded_bytes(),
    ] {
        hash.update(part.len().to_le_bytes());
        hash.update(part);
    }
    let key = format!("{:x}", hash.finalize());
    let root = cache_root()?.join(key);
    std::fs::create_dir_all(&root).map_err(|error| io_error(&root, error))?;
    let cached = root.join("manifest.json");
    if cached.is_file() {
        return artifact(&cached, &request, version);
    }
    let work = work_directory(&root)?;
    let source_path = work.join("kernel.py");
    let request_path = work.join("request.json");
    std::fs::write(&source_path, source).map_err(|error| io_error(&source_path, error))?;
    std::fs::write(&request_path, request_bytes).map_err(|error| io_error(&request_path, error))?;
    let output = Command::new(python)
        .args(["-c", ADAPTER, "--source"])
        .arg(&source_path)
        .arg("--request")
        .arg(&request_path)
        .arg("--output-directory")
        .arg(&work)
        .output()
        .map_err(|error| SilvicError::device(format!("start Neuron kernel compiler: {error}")))?;
    if !output.status.success() {
        return Err(command_error(output, "kernel compilation"));
    }
    let manifest_path = work.join("manifest.json");
    let result = artifact(&manifest_path, &request, version)?;
    let published = root.join(format!(
        "manifest-{}.tmp",
        work.file_name().expect("work directory").to_string_lossy()
    ));
    std::fs::copy(&manifest_path, &published).map_err(|error| io_error(&published, error))?;
    std::fs::rename(&published, &cached).map_err(|error| io_error(&cached, error))?;
    Ok(result)
}
