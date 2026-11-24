//! Texts in, embeddings out: tokenizing, batching and the encoder.

use std::ops::Range;
use std::path::Path;

use crate::gpu::GpuModel;
use crate::model::{Config, CpuModel, Tensors};
use crate::tokenizer::Tokenizer;
use crate::vulkan::Gpu;
use crate::{DIMENSION, Error, QUERY_PREFIX};

/// What a text is, which decides whether it takes the [`QUERY_PREFIX`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Role {
    /// A natural-language question.
    Query,
    /// Code, or anything else searched.
    Document,
}

/// Where the encoder runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Device {
    /// The CPU, with the SIMD instructions it has.
    Cpu,
    /// The best Vulkan GPU: a discrete one first.
    Gpu,
}

/// Tokens of one batch, and the longest text: enough rows for the products
/// to keep every CPU thread busy, few enough that the activations of a batch
/// stay under 400 MB.
pub(crate) const BATCH_TOKENS: usize = 8192;
/// Sequences of one batch: as many as the shortest texts, `[CLS]` and
/// `[SEP]` alone, take to fill it.
pub(crate) const BATCH_SEQS: usize = BATCH_TOKENS / 2;

#[expect(
    clippy::large_enum_variant,
    reason = "a process holds one encoder; boxing its model saves nothing"
)]
enum Backend {
    Cpu(CpuModel),
    Gpu(GpuModel),
}

/// CodeRankEmbed, loaded and ready to embed.
pub struct Encoder {
    tokenizer: Tokenizer,
    backend: Backend,
    max_tokens: usize,
    /// Token ids below this have an embedding.
    vocab: usize,
}

impl Encoder {
    /// Load the model onto `device` from `dir`, which holds the files of
    /// `nomic-ai/CodeRankEmbed` as published: `config.json`,
    /// `model.safetensors` and `tokenizer.json`. On the CPU the encoder runs
    /// on `threads` threads, or one per CPU the process may use when 0.
    ///
    /// # Errors
    /// When a file cannot be read, is malformed or describes another model,
    /// the worker threads cannot start, or there is no usable GPU for
    /// [`Device::Gpu`].
    pub fn open(dir: &Path, device: Device, threads: usize) -> Result<Self, Error> {
        let config = Config::load(&dir.join("config.json"))?;
        let tokenizer = Tokenizer::load(&dir.join("tokenizer.json"))?;
        if config.hidden != DIMENSION {
            return Err(Error::Model(format!(
                "{} values per embedding, not {DIMENSION}",
                config.hidden
            )));
        }
        if tokenizer.max_id() as usize >= config.vocab {
            return Err(Error::Model(format!(
                "the tokenizer's ids pass the {} embeddings",
                config.vocab
            )));
        }
        let vocab = config.vocab;
        // Before the weights, so that a missing GPU costs no read of them,
        // which a fallback to the CPU would repeat.
        let gpu = match device {
            Device::Gpu => Some(Gpu::open().map_err(Error::Gpu)?),
            Device::Cpu => None,
        };
        let tensors = Tensors::load(config, &dir.join("model.safetensors"))?;
        let backend = match gpu {
            Some(gpu) => Backend::Gpu(GpuModel::new(tensors, gpu)?),
            None => {
                let threads = if threads == 0 {
                    std::thread::available_parallelism().map_or(1, |n| n.get())
                } else {
                    threads
                };
                Backend::Cpu(CpuModel::new(tensors, threads)?)
            }
        };
        Ok(Self {
            tokenizer,
            backend,
            max_tokens: BATCH_TOKENS,
            vocab,
        })
    }

    /// Tokens a text is cut to, `[CLS]` and `[SEP]` included: 8192 unless
    /// set, the length the model's sentence-transformers configuration
    /// gives.
    pub fn max_tokens(&self) -> usize {
        self.max_tokens
    }

    /// Cut texts to `max_tokens` tokens, `[CLS]` and `[SEP]` included.
    ///
    /// # Panics
    /// When `max_tokens` is below 2 or above 8192, the longest text the
    /// model reads.
    pub fn set_max_tokens(&mut self, max_tokens: usize) {
        assert!(
            (2..=BATCH_TOKENS).contains(&max_tokens),
            "{max_tokens} tokens, not 2 to {BATCH_TOKENS}"
        );
        self.max_tokens = max_tokens;
    }

    /// The token ids of `text`, between `[CLS]` and `[SEP]` and cut to
    /// [`Encoder::max_tokens`], as the encoder reads them for a document.
    pub fn tokenize(&self, text: &str) -> Vec<u32> {
        self.tokenizer.encode(text, self.max_tokens)
    }

    /// Tokens of `text` without `[CLS]` and `[SEP]`. Parts of a text split
    /// at whitespace have as many tokens together as the whole.
    pub fn count_tokens(&self, text: &str) -> usize {
        self.tokenizer.count(text)
    }

    /// Where the encoder runs.
    pub fn device(&self) -> Device {
        match self.backend {
            Backend::Cpu(_) => Device::Cpu,
            Backend::Gpu(_) => Device::Gpu,
        }
    }

    /// What the encoder runs on: the GPU's name, or the CPU threads.
    pub fn device_name(&self) -> String {
        match &self.backend {
            Backend::Cpu(model) => format!("the CPU, {} threads", model.threads()),
            Backend::Gpu(model) => model.name().to_string(),
        }
    }

    /// The embeddings of `texts`, [`DIMENSION`] floats each, one after the
    /// other in the order of `texts`. They are not normalized: as the
    /// model's pooling leaves them.
    ///
    /// # Errors
    /// When the GPU fails during a batch; the encoder then refuses every
    /// later batch.
    pub fn embed(&mut self, texts: &[&str], role: Role) -> Result<Vec<f32>, Error> {
        let mut prefixed = String::new();
        let sequences: Vec<Vec<u32>> = texts
            .iter()
            .map(|text| match role {
                Role::Query => {
                    prefixed.clear();
                    prefixed.push_str(QUERY_PREFIX);
                    prefixed.push_str(text);
                    self.tokenizer.encode(&prefixed, self.max_tokens)
                }
                Role::Document => self.tokenizer.encode(text, self.max_tokens),
            })
            .collect();
        let sequences: Vec<&[u32]> = sequences.iter().map(Vec::as_slice).collect();
        self.embed_ids(&sequences)
    }

    /// The embeddings of sequences of token ids, each between `[CLS]` and
    /// `[SEP]` as [`Encoder::tokenize`] gives them (a caller with its own
    /// tokenizer and windows of long texts passes those), [`DIMENSION`]
    /// floats each, one after the other in order, not normalized.
    ///
    /// # Errors
    /// When a sequence is empty or longer than 8192 tokens, or holds an id
    /// with no embedding (an [`Error::Input`]), and when the GPU fails during
    /// a batch, as [`Encoder::embed`].
    pub fn embed_ids(&mut self, sequences: &[&[u32]]) -> Result<Vec<f32>, Error> {
        for (index, tokens) in sequences.iter().enumerate() {
            if !(1..=BATCH_TOKENS).contains(&tokens.len()) {
                return Err(Error::Input(format!(
                    "sequence {index} has {} tokens, not 1 to {BATCH_TOKENS}",
                    tokens.len()
                )));
            }
            if let Some(id) = tokens.iter().find(|&&id| id as usize >= self.vocab) {
                return Err(Error::Input(format!(
                    "sequence {index} holds id {id}, past the {} embeddings",
                    self.vocab
                )));
            }
        }
        let mut out = vec![0.0; sequences.len() * DIMENSION];
        let mut ids = Vec::new();
        let mut seqs: Vec<Range<usize>> = Vec::new();
        let mut first = 0;
        for (index, tokens) in sequences.iter().enumerate() {
            if !ids.is_empty()
                && (ids.len() + tokens.len() > BATCH_TOKENS || seqs.len() == BATCH_SEQS)
            {
                self.run(&ids, &seqs, &mut out[first * DIMENSION..index * DIMENSION])?;
                ids.clear();
                seqs.clear();
                first = index;
            }
            seqs.push(ids.len()..ids.len() + tokens.len());
            ids.extend_from_slice(tokens);
        }
        if !seqs.is_empty() {
            self.run(&ids, &seqs, &mut out[first * DIMENSION..])?;
        }
        Ok(out)
    }

    fn run(&mut self, ids: &[u32], seqs: &[Range<usize>], out: &mut [f32]) -> Result<(), Error> {
        match &mut self.backend {
            Backend::Cpu(model) => {
                model.embed(ids, seqs, out);
                Ok(())
            }
            Backend::Gpu(model) => model.embed(ids, seqs, out),
        }
    }
}
