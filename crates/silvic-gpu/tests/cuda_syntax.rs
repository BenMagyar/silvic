//! Checks the CUDA kernel source against clang's CUDA frontend, since NVRTC needs an NVIDIA toolchain: the CUDA
//! prelude with a sample kernel that uses every macro, and the full source `kernel_source` builds for NVRTC.
//! `kernels/cuda_syntax/shim.h` stands in for the declarations NVRTC provides implicitly. Also checks that the
//! Metal and CUDA preludes define the same macros.
//!
//! The compiler is `SILVIC_CUDA_CLANG`, else `/opt/homebrew/opt/llvm/bin/clang++`, else `clang++` on `PATH`; the
//! clang checks skip, printing why, when none of them compiles CUDA.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::process::Command;

const HOMEBREW_CLANG: &str = "/opt/homebrew/opt/llvm/bin/clang++";

/// Device-side frontend only: no CUDA installation (`-nocudainc -nocudalib`) and no NVPTX backend needed, which
/// Apple's clang lacks.
const CUDA_FLAGS: &[&str] = &[
    "-x",
    "cuda",
    "--cuda-gpu-arch=sm_80",
    "--cuda-device-only",
    "-nocudainc",
    "-nocudalib",
    "-std=c++17",
    "-fsyntax-only",
];

fn kernels_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("kernels")
}

fn read(path: &Path) -> String {
    std::fs::read_to_string(path).unwrap_or_else(|err| panic!("read {}: {err}", path.display()))
}

fn compiles_cuda(clang: &str) -> bool {
    let probe = Path::new(env!("CARGO_TARGET_TMPDIR")).join("cuda_syntax_probe.cu");
    std::fs::write(
        &probe,
        "extern \"C\" __attribute__((global)) void probe(int* p) { p[0] = 1; }\n",
    )
    .expect("write probe");
    Command::new(clang)
        .args(CUDA_FLAGS)
        .arg(&probe)
        .output()
        .is_ok_and(|out| out.status.success())
}

/// The CUDA-capable clang, or why there is none.
fn find_clang() -> Result<String, String> {
    if let Ok(clang) = std::env::var("SILVIC_CUDA_CLANG") {
        return if compiles_cuda(&clang) {
            Ok(clang)
        } else {
            Err(format!("SILVIC_CUDA_CLANG={clang} does not compile `-x cuda`"))
        };
    }
    let candidate = if Path::new(HOMEBREW_CLANG).exists() {
        HOMEBREW_CLANG.to_string()
    } else if Command::new("clang++")
        .arg("--version")
        .output()
        .is_ok_and(|out| out.status.success())
    {
        "clang++".to_string()
    } else {
        return Err(format!("neither {HOMEBREW_CLANG} nor clang++ on PATH exists"));
    };
    if compiles_cuda(&candidate) {
        Ok(candidate)
    } else {
        Err(format!("{candidate} does not compile `-x cuda`"))
    }
}

/// The files of `parts` concatenated, each starting with a `#line` directive so diagnostics name the file.
fn concat(parts: &[PathBuf]) -> String {
    let mut source = String::new();
    for part in parts {
        source.push_str(&format!("\n#line 1 \"{}\"\n", part.display()));
        source.push_str(&read(part));
    }
    source
}

/// Compiles the shim followed by `source`.
fn check_cuda(case: &str, source: &str) {
    let clang = match find_clang() {
        Ok(clang) => clang,
        Err(reason) => {
            println!("skipping CUDA syntax check {case}: {reason}");
            return;
        }
    };
    let source = read(&kernels_dir().join("cuda_syntax/shim.h")) + source;
    let path = Path::new(env!("CARGO_TARGET_TMPDIR")).join(format!("cuda_syntax_{case}.cu"));
    std::fs::write(&path, source).expect("write source");
    let out = Command::new(&clang)
        .args(CUDA_FLAGS)
        .arg(&path)
        .output()
        .expect("run clang");
    let diagnostics = String::from_utf8_lossy(&out.stderr);
    if !diagnostics.is_empty() {
        println!("{case} diagnostics from {clang}:\n{diagnostics}");
    }
    assert!(
        out.status.success(),
        "{case}: clang rejected the CUDA source:\n{diagnostics}"
    );
}

#[test]
fn sample_kernel_compiles_as_cuda() {
    let dir = kernels_dir();
    check_cuda(
        "sample",
        &concat(&[dir.join("prelude_cuda.h"), dir.join("cuda_syntax/sample.kern")]),
    );
}

#[test]
fn kernels_compile_as_cuda() {
    let prelude = read(&kernels_dir().join("prelude_cuda.h"));
    check_cuda("kernels", &silvic_gpu::kernel_source(&prelude));
}

/// The names of every `#define` in a prelude.
fn defined_macros(path: &Path) -> BTreeSet<String> {
    read(path)
        .lines()
        .filter_map(|line| line.trim_start().strip_prefix("#define"))
        .filter_map(|rest| {
            let name: String = rest
                .trim_start()
                .chars()
                .take_while(|c| c.is_alphanumeric() || *c == '_')
                .collect();
            (!name.is_empty()).then_some(name)
        })
        .collect()
}

#[test]
fn preludes_define_the_same_macros() {
    let dir = kernels_dir();
    let metal = dir.join("prelude_metal.h");
    if !metal.exists() {
        println!("skipping prelude comparison: {} does not exist yet", metal.display());
        return;
    }
    let cuda = defined_macros(&dir.join("prelude_cuda.h"));
    let metal = defined_macros(&metal);
    let only_cuda: Vec<_> = cuda.difference(&metal).collect();
    let only_metal: Vec<_> = metal.difference(&cuda).collect();
    assert!(
        only_cuda.is_empty() && only_metal.is_empty(),
        "prelude macros differ: only in prelude_cuda.h {only_cuda:?}, only in prelude_metal.h {only_metal:?}"
    );
}
