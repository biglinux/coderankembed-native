//! Does the encoder still compute what CodeRankEmbed's own PyTorch code
//! computes?
//!
//! `tests/fixtures/reference.json` was written by `tools/make_fixtures.py`
//! from Hugging Face's tokenizer and sentence-transformers running the
//! model in float32. The weights come from the Hugging Face cache or
//! `CODERANKEMBED_MODEL` (see `default_model_dir`): run with
//! `--include-ignored`, and `--skip gpu_` on a machine without a GPU.

use std::assert_matches;
use std::sync::{Mutex, OnceLock};

use coderankembed_native::{DIMENSION, Device, Encoder, QUERY_PREFIX, Role, default_model_dir};
use serde_json::Value;

fn fixtures() -> &'static Value {
    static FIXTURES: OnceLock<Value> = OnceLock::new();
    FIXTURES.get_or_init(|| {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/reference.json");
        serde_json::from_slice(&std::fs::read(path).expect("fixtures")).expect("fixtures are JSON")
    })
}

fn encoder() -> &'static Mutex<Encoder> {
    static ENCODER: OnceLock<Mutex<Encoder>> = OnceLock::new();
    ENCODER.get_or_init(|| {
        let dir = default_model_dir().expect(
            "the CodeRankEmbed weights: download nomic-ai/CodeRankEmbed with the Hugging Face \
             CLI or point CODERANKEMBED_MODEL at its files",
        );
        Mutex::new(Encoder::open(&dir, Device::Cpu, 0).expect("the model loads"))
    })
}

fn ids(value: &Value) -> Vec<u32> {
    value
        .as_array()
        .expect("ids")
        .iter()
        .map(|id| u32::try_from(id.as_u64().expect("an id")).expect("a u32 id"))
        .collect()
}

/// Cosine similarity and largest difference relative to the reference's norm.
fn distance(got: &[f32], want: &[f32]) -> (f64, f64) {
    let norm = |v: &[f32]| v.iter().map(|x| f64::from(*x).powi(2)).sum::<f64>().sqrt();
    let dot: f64 = got
        .iter()
        .zip(want)
        .map(|(a, b)| f64::from(*a) * f64::from(*b))
        .sum();
    let worst = got
        .iter()
        .zip(want)
        .map(|(a, b)| f64::from((a - b).abs()))
        .fold(0.0, f64::max);
    (dot / (norm(got) * norm(want)), worst / norm(want))
}

#[test]
#[ignore = "needs the CodeRankEmbed weights"]
fn tokens_match_hugging_face() {
    let encoder = encoder().lock().expect("encoder");
    let fixtures = fixtures();
    // The length sentence-transformers cuts texts to.
    let max_tokens = fixtures["max_tokens"].as_u64().expect("max_tokens");
    assert_eq!(u64::try_from(encoder.max_tokens()), Ok(max_tokens));
    let cases = fixtures["tokens"].as_array().expect("tokens");
    let embedded = fixtures["embeddings"].as_array().expect("embeddings");
    for case in cases.iter().chain(embedded) {
        let text = case["text"].as_str().expect("text");
        let text = match case["role"].as_str() {
            Some("query") => format!("{QUERY_PREFIX}{text}"),
            _ => text.to_string(),
        };
        assert_eq!(encoder.tokenize(&text), ids(&case["ids"]), "{text:?}");
    }
}

/// The texts of role `name` and their reference embeddings.
fn expected<'a>(cases: &'a [Value], name: &str) -> (Vec<&'a str>, Vec<Vec<f32>>) {
    cases
        .iter()
        .filter(|case| case["role"] == name)
        .map(|case| {
            let embedding = case["embedding"].as_array().expect("embedding");
            let embedding = embedding
                .iter()
                .map(|v| v.as_f64().expect("a float") as f32)
                .collect();
            (case["text"].as_str().expect("text"), embedding)
        })
        .unzip()
}

#[test]
#[ignore = "needs the CodeRankEmbed weights"]
fn embeddings_match_pytorch() {
    let mut encoder = encoder().lock().expect("encoder");
    let cases = fixtures()["embeddings"].as_array().expect("embeddings");
    for (role, name) in [(Role::Query, "query"), (Role::Document, "document")] {
        let (texts, want) = expected(cases, name);
        // Each text alone, then all of them in one batch.
        let mut batches: Vec<(usize, Vec<&str>)> = texts
            .iter()
            .enumerate()
            .map(|(i, &t)| (i, vec![t]))
            .collect();
        batches.push((0, texts.clone()));
        for (first, batch) in batches {
            let got = encoder.embed(&batch, role).expect("the CPU embeds");
            for (n, got) in got.as_chunks::<DIMENSION>().0.iter().enumerate() {
                let (cosine, worst) = distance(got, &want[first + n]);
                assert!(
                    cosine > 0.999_99 && worst < 1e-3,
                    "{name} {} of a batch of {}: cosine {cosine}, largest difference {worst} of the norm",
                    first + n,
                    batch.len()
                );
            }
        }
    }
}

/// The GPU keeps the weights in half floats, and on AMD and NVIDIA multiplies
/// in them too, which moves each embedding by up to about 4e-4 of its norm.
#[test]
#[ignore = "needs the CodeRankEmbed weights and a Vulkan GPU"]
fn gpu_embeddings_match_pytorch() {
    let dir = default_model_dir().expect("the CodeRankEmbed weights");
    let mut encoder = Encoder::open(&dir, Device::Gpu, 0).expect("a GPU");
    let cases = fixtures()["embeddings"].as_array().expect("embeddings");
    for (role, name) in [(Role::Query, "query"), (Role::Document, "document")] {
        let (texts, want) = expected(cases, name);
        let got = encoder.embed(&texts, role).expect("the GPU embeds");
        for (n, got) in got.as_chunks::<DIMENSION>().0.iter().enumerate() {
            let (cosine, worst) = distance(got, &want[n]);
            assert!(
                cosine > 0.999_99 && worst < 1e-3,
                "{name} {n}: cosine {cosine}, largest difference {worst} of the norm"
            );
        }
    }
}

/// Token ids from another tokenizer, such as an indexer's windows, give the
/// same vectors as the texts they came from; ids the model cannot read are
/// refused.
#[test]
#[ignore = "needs the CodeRankEmbed weights"]
fn token_ids_embed_as_their_texts() {
    let mut encoder = encoder().lock().expect("encoder");
    let texts = ["fn main() {}", "def add(a, b):\n    return a + b\n"];
    let ids: Vec<Vec<u32>> = texts.iter().map(|t| encoder.tokenize(t)).collect();
    let ids: Vec<&[u32]> = ids.iter().map(Vec::as_slice).collect();
    let from_ids = encoder.embed_ids(&ids).expect("the CPU embeds");
    let from_texts = encoder
        .embed(&texts, Role::Document)
        .expect("the CPU embeds");
    assert_eq!(from_ids, from_texts);
    let long = vec![101; 8193];
    for refused in [&[][..], &long, &[101, 1 << 20, 102]] {
        assert_matches!(
            encoder.embed_ids(&[refused]),
            Err(coderankembed_native::Error::Input(_))
        );
    }
}

/// More sequences than one GPU batch holds, as a caller's one-token windows
/// give, run in several batches.
#[test]
#[ignore = "needs the CodeRankEmbed weights and a Vulkan GPU"]
fn gpu_many_short_sequences_embed() {
    let dir = default_model_dir().expect("the CodeRankEmbed weights");
    let mut encoder = Encoder::open(&dir, Device::Gpu, 0).expect("a GPU");
    let cls = [101];
    let sequences = vec![&cls[..]; 4097];
    let got = encoder.embed_ids(&sequences).expect("the GPU embeds");
    let rows = got.as_chunks::<DIMENSION>().0;
    assert_eq!(rows.len(), sequences.len());
    // The last batch of one row runs other kernels, which sum in another order.
    let (cosine, _) = distance(&rows[rows.len() - 1], &rows[0]);
    assert!(cosine > 0.999_99, "cosine {cosine}");
    assert!(rows[..4096].iter().all(|row| row == &rows[0]));
}
