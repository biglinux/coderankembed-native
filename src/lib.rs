//! CodeRankEmbed, the code retrieval encoder trained on CoRNStack, in plain
//! Rust: the WordPiece tokenizer and the 12-layer NomicBERT encoder, on the
//! CPU with SIMD kernels chosen at run time or on a Vulkan GPU.
//!
//! [`Encoder::open`] reads the model's published files (`config.json`,
//! `model.safetensors`, `tokenizer.json`) and [`Encoder::embed`] turns texts
//! into 768-float embeddings, the hidden state of the `[CLS]` token as the
//! model's sentence-transformers pooling selects. A query takes the
//! [`QUERY_PREFIX`] the model was trained with; code goes in as it is.
//!
//! - `encoder`: the public API, batching texts and choosing the backend.
//! - `tokenizer`: the WordPiece tokenizer of `tokenizer.json`.
//! - `safetensors`: reading the weights file.
//! - `model`: the configuration, the weights and the encoder on the CPU.
//! - `gemm`: the CPU's matrix products; `math`: its elementwise kernels;
//!   `pool`: its worker threads.
//! - `gpu`: the encoder on a Vulkan GPU, over `vulkan` and `half`.
//! - `fuzzing`: the fuzz targets' bodies, only in `cargo fuzz` builds.

/// `a * b + c` on 8 lanes: one fused instruction with FMA, a multiply and an
/// add without.
#[cfg(target_arch = "x86_64")]
macro_rules! madd8 {
    (fused, $a:expr, $b:expr, $c:expr) => {
        std::arch::x86_64::_mm256_fmadd_ps($a, $b, $c)
    };
    (split, $a:expr, $b:expr, $c:expr) => {
        std::arch::x86_64::_mm256_add_ps(std::arch::x86_64::_mm256_mul_ps($a, $b), $c)
    };
}

/// Define `$name`, running `$body` compiled for AVX2 or AVX when the CPU has
/// them and as plain code otherwise. The body fuses nothing, so all three
/// give the same bits.
macro_rules! dispatch {
    ($(#[$doc:meta])* $vis:vis $name:ident($($arg:ident: $ty:ty),*) $body:block) => {
        $(#[$doc])*
        $vis fn $name($($arg: $ty),*) {
            #[cfg(target_arch = "x86_64")]
            {
                #[target_feature(enable = "avx2,fma")]
                fn avx2($($arg: $ty),*) $body
                #[target_feature(enable = "avx")]
                fn avx($($arg: $ty),*) $body
                match $crate::simd_tier() {
                    // SAFETY: the CPU supports AVX2 and FMA.
                    $crate::TIER_AVX2 => return unsafe { avx2($($arg),*) },
                    // SAFETY: the CPU supports AVX.
                    $crate::TIER_AVX => return unsafe { avx($($arg),*) },
                    _ => {}
                }
            }
            $body
        }
    };
}

mod encoder;
#[cfg(fuzzing)]
#[doc(hidden)]
pub mod fuzzing;
mod gemm;
mod gpu;
mod half;
mod math;
mod model;
mod pool;
mod safetensors;
mod tokenizer;
mod vulkan;

use std::sync::OnceLock;

pub use encoder::{Device, Encoder, Role};

/// Values in an embedding.
pub const DIMENSION: usize = 768;

/// Tokens of the longest sequence the encoder reads, `[CLS]` and `[SEP]`
/// included, and of one forward pass.
pub const MAX_TOKENS: usize = encoder::BATCH_TOKENS;

/// What a natural-language query starts with, as the model was trained.
pub const QUERY_PREFIX: &str = "Represent this query for searching relevant code: ";

/// Why the model could not be loaded or a text not embedded.
#[derive(Debug)]
pub enum Error {
    /// A file could not be read.
    Io(std::path::PathBuf, std::io::Error),
    /// A model file is malformed or describes another model.
    Model(String),
    /// The worker threads could not start.
    Threads(std::io::Error),
    /// The GPU could not be used, or failed during a batch.
    Gpu(String),
    /// A sequence of token ids the model cannot read.
    Input(String),
}

impl Error {
    /// A malformed model file, named by `path` in the message.
    pub(crate) fn model(path: &std::path::Path, msg: impl std::fmt::Display) -> Self {
        Error::Model(format!("{}: {msg}", path.display()))
    }
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::Io(path, e) => write!(f, "{}: {e}", path.display()),
            Error::Model(msg) => write!(f, "model: {msg}"),
            Error::Threads(e) => write!(f, "worker threads: {e}"),
            Error::Gpu(msg) => write!(f, "GPU: {msg}"),
            Error::Input(msg) => write!(f, "input: {msg}"),
        }
    }
}

impl std::error::Error for Error {}

const TIER_AVX: u8 = 1;
const TIER_AVX2: u8 = 2;

/// Instruction set the kernels use: 2 = AVX2 with FMA and F16C, 1 = AVX,
/// 0 = SSE2 on x86-64 and plain Rust elsewhere. Results are the same on every
/// run within a tier; tiers 1 and 0 also give the same bits on x86-64, tier 2
/// does not, since it fuses each multiply-add.
///
/// `CODERANKEMBED_TIER` caps it, to run an older CPU's code on a newer one.
///
/// # Panics
/// When `CODERANKEMBED_TIER` is set to anything but 0, 1 or 2: a run asked
/// to test a tier must not test another.
fn simd_tier() -> u8 {
    static TIER: OnceLock<u8> = OnceLock::new();
    *TIER.get_or_init(|| {
        #[cfg(target_arch = "x86_64")]
        let detected = if std::is_x86_feature_detected!("avx2")
            && std::is_x86_feature_detected!("fma")
            && std::is_x86_feature_detected!("f16c")
        {
            TIER_AVX2
        } else if std::is_x86_feature_detected!("avx") {
            TIER_AVX
        } else {
            0
        };
        #[cfg(not(target_arch = "x86_64"))]
        let detected = 0;
        match std::env::var("CODERANKEMBED_TIER") {
            Err(std::env::VarError::NotPresent) => detected,
            Ok(cap) if let Ok(cap @ 0..=2) = cap.parse::<u8>() => cap.min(detected),
            _ => panic!("CODERANKEMBED_TIER takes 0, 1 or 2"),
        }
    })
}

/// Where the Hugging Face cache keeps `nomic-ai/CodeRankEmbed`, or
/// `CODERANKEMBED_MODEL` when set: the snapshot `refs/main` names, else the
/// only one there is.
pub fn default_model_dir() -> Option<std::path::PathBuf> {
    use std::path::PathBuf;
    let env = |key: &str| {
        std::env::var_os(key)
            .filter(|v| !v.is_empty())
            .map(PathBuf::from)
    };
    if let Some(dir) = env("CODERANKEMBED_MODEL") {
        return Some(dir);
    }
    let hub = env("HF_HUB_CACHE")
        .or_else(|| env("HF_HOME").map(|home| home.join("hub")))
        .or_else(|| env("XDG_CACHE_HOME").map(|cache| cache.join("huggingface/hub")))
        .or_else(|| env("HOME").map(|home| home.join(".cache/huggingface/hub")))?;
    let repo = hub.join("models--nomic-ai--CodeRankEmbed");
    let snapshots = repo.join("snapshots");
    if let Ok(main) = std::fs::read_to_string(repo.join("refs/main")) {
        return Some(snapshots.join(main.trim()));
    }
    let mut found = std::fs::read_dir(&snapshots)
        .ok()?
        .filter_map(|entry| Some(entry.ok()?.path()))
        .filter(|dir| dir.join("model.safetensors").is_file());
    let only = found.next()?;
    found.next().is_none().then_some(only)
}
