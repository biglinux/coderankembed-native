//! Compile the GPU shaders to SPIR-V with `glslc`, from shaderc.

use std::path::PathBuf;
use std::process::Command;

fn main() {
    let out = PathBuf::from(std::env::var_os("OUT_DIR").expect("cargo sets OUT_DIR"));
    // Output name, source, and the defines it is compiled with.
    let shaders: [(&str, &str, &[&str]); 8] = [
        ("attention", "attention", &[]),
        ("gather", "gather", &[]),
        ("gemm", "gemm", &[]),
        ("gemm_half", "gemm", &["-DHALF_PRODUCTS"]),
        ("gemm_rows", "gemm_rows", &[]),
        ("layer_norm", "layer_norm", &[]),
        ("rope", "rope", &[]),
        ("swiglu", "swiglu", &[]),
    ];
    for (name, source, defines) in shaders {
        let source = format!("shaders/{source}.comp");
        println!("cargo::rerun-if-changed={source}");
        let status = Command::new("glslc")
            .args(["--target-env=vulkan1.2", "-O"])
            .args(defines)
            .args([&source, "-o"])
            .arg(out.join(format!("{name}.spv")))
            .status()
            .unwrap_or_else(|e| panic!("glslc, from shaderc, compiles the GPU shaders: {e}"));
        assert!(status.success(), "glslc failed on {source}: {status}");
    }
}
