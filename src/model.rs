//! The NomicBERT encoder behind CodeRankEmbed, on the CPU.
//!
//! A BERT encoder with rotary position embeddings on the queries and keys
//! instead of learned positions, one fused projection for queries, keys and
//! values, a SwiGLU feed-forward (`fc2(fc11(x) * silu(fc12(x)))`), no biases
//! on any projection, and LayerNorm after each residual. Only the variant
//! CodeRankEmbed uses is read; [`Config::parse`] refuses the others.
//!
//! A batch is its sequences' tokens one after the other, without padding:
//! every projection runs over all tokens at once and attention over each
//! sequence alone. The last layer only computes the `[CLS]` rows, which the
//! pooling keeps.

use std::ops::Range;
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Mutex, PoisonError};

use crate::Error;
use crate::gemm::{Matrix, NR, Out, Panels, Row, multiply, multiply_into, panel_rows};
use crate::math::{layer_norm, softmax_rows, swiglu};
use crate::pool::{Pool, Shared};
use crate::safetensors::SafeTensors;

/// What `config.json` says about the encoder's shape.
pub(crate) struct Config {
    pub(crate) hidden: usize,
    pub(crate) heads: usize,
    pub(crate) inner: usize,
    pub(crate) layers: usize,
    pub(crate) vocab: usize,
    pub(crate) eps: f32,
    pub(crate) rotary_base: f32,
}

impl Config {
    pub(crate) fn load(path: &Path) -> Result<Self, Error> {
        let text = std::fs::read(path).map_err(|e| Error::Io(path.to_path_buf(), e))?;
        Self::parse(&text).map_err(|msg| Error::model(path, msg))
    }

    fn parse(text: &[u8]) -> Result<Self, String> {
        let json: serde_json::Value = serde_json::from_slice(text).map_err(|e| e.to_string())?;
        let flag = |key: &str| json.get(key).and_then(serde_json::Value::as_bool);
        let number = |key: &str| json.get(key).and_then(serde_json::Value::as_f64);
        let set = |key: &str| json.get(key).is_some_and(|v| !v.is_null());
        let int = |key: &str| {
            json.get(key)
                .and_then(serde_json::Value::as_u64)
                .and_then(|n| usize::try_from(n).ok())
                .filter(|&n| n > 0)
                .ok_or_else(|| format!("{key} is missing"))
        };
        if json.get("model_type").and_then(|v| v.as_str()) != Some("nomic_bert") {
            return Err("not a NomicBERT model".to_string());
        }
        let unsupported = [
            (
                json.get("activation_function").and_then(|v| v.as_str()) != Some("swiglu"),
                "activation_function",
            ),
            (flag("prenorm") != Some(false), "prenorm"),
            (flag("causal") == Some(true), "causal"),
            (flag("use_rms_norm") == Some(true), "use_rms_norm"),
            (flag("parallel_block") == Some(true), "parallel_block"),
            (flag("qkv_proj_bias") != Some(false), "qkv_proj_bias"),
            (flag("mlp_fc1_bias") != Some(false), "mlp_fc1_bias"),
            (flag("mlp_fc2_bias") != Some(false), "mlp_fc2_bias"),
            (
                number("rotary_emb_fraction") != Some(1.0),
                "rotary_emb_fraction",
            ),
            (
                flag("rotary_emb_interleaved") == Some(true),
                "rotary_emb_interleaved",
            ),
            (set("rotary_scaling_factor"), "rotary_scaling_factor"),
            (set("rotary_emb_scale_base"), "rotary_emb_scale_base"),
            (
                set("moe_every_n_layers") && int("moe_every_n_layers").is_ok(),
                "moe_every_n_layers",
            ),
        ];
        if let Some((_, key)) = unsupported.iter().find(|(differs, _)| *differs) {
            return Err(format!("unsupported NomicBERT variant ({key})"));
        }
        let (hidden, heads) = (int("n_embd")?, int("n_head")?);
        if !hidden.is_multiple_of(heads) || !(hidden / heads).is_multiple_of(2) {
            return Err(format!(
                "{heads} heads do not divide {hidden} channels in even halves"
            ));
        }
        // PyTorch pads the vocabulary to a multiple, as the weights are stored.
        let multiple = int("pad_vocab_size_multiple").unwrap_or(1);
        Ok(Self {
            hidden,
            heads,
            inner: int("n_inner")?,
            layers: int("n_layer")?,
            vocab: int("vocab_size")?
                .checked_next_multiple_of(multiple)
                .ok_or("vocab_size is too large")?,
            eps: number("layer_norm_epsilon").unwrap_or(1e-12) as f32,
            rotary_base: number("rotary_emb_base").unwrap_or(10_000.0) as f32,
        })
    }

    pub(crate) fn head_dim(&self) -> usize {
        self.hidden / self.heads
    }

    /// What attention scales the queries by: one over the square root of a
    /// head's channels.
    pub(crate) fn query_scale(&self) -> f32 {
        1.0 / (self.head_dim() as f32).sqrt()
    }
}

/// A product's weights as stored, `n` rows of `k`: `(weights, n, k)`.
type StoredProduct = (Vec<f32>, usize, usize);

/// A LayerNorm's scale and shift.
pub(crate) struct Norm {
    pub(crate) gamma: Vec<f32>,
    pub(crate) beta: Vec<f32>,
}

/// One layer's weights as stored: projections `out x in`, row-major.
pub(crate) struct LayerTensors {
    pub(crate) qkv: Vec<f32>,
    pub(crate) out: Vec<f32>,
    /// `fc11` then `fc12`: the up and gate projections, one product.
    pub(crate) up_gate: Vec<f32>,
    pub(crate) down: Vec<f32>,
    pub(crate) norm1: Norm,
    pub(crate) norm2: Norm,
}

/// The model's weights as stored, for a backend to lay out its own way.
pub(crate) struct Tensors {
    pub(crate) config: Config,
    /// `vocab` rows of `hidden`, the token type 0 embedding already added.
    pub(crate) embeddings: Vec<f32>,
    pub(crate) emb_norm: Norm,
    pub(crate) layers: Vec<LayerTensors>,
}

impl Tensors {
    pub(crate) fn load(config: Config, path: &Path) -> Result<Self, Error> {
        // The layer count sizes the list of names below: check it against the
        // file before trusting it.
        let listed = SafeTensors::open(path)?.tensor_count();
        if config
            .layers
            .checked_mul(9)
            .and_then(|n| n.checked_add(4))
            .is_none_or(|wanted| wanted > listed)
        {
            return Err(Error::model(
                path,
                format!(
                    "{} layers, more than its {listed} tensors hold",
                    config.layers
                ),
            ));
        }
        let (h, i) = (config.hidden, config.inner);
        // Every tensor, in the order they are taken below.
        let mut wanted: Vec<(String, Vec<usize>)> = vec![
            ("emb_ln.weight".into(), vec![h]),
            ("emb_ln.bias".into(), vec![h]),
            (
                "embeddings.word_embeddings.weight".into(),
                vec![config.vocab, h],
            ),
            ("embeddings.token_type_embeddings.weight".into(), vec![2, h]),
        ];
        for n in 0..config.layers {
            let p = format!("encoder.layers.{n}");
            wanted.extend([
                (format!("{p}.attn.Wqkv.weight"), vec![3 * h, h]),
                (format!("{p}.attn.out_proj.weight"), vec![h, h]),
                (format!("{p}.mlp.fc11.weight"), vec![i, h]),
                (format!("{p}.mlp.fc12.weight"), vec![i, h]),
                (format!("{p}.mlp.fc2.weight"), vec![h, i]),
                (format!("{p}.norm1.weight"), vec![h]),
                (format!("{p}.norm1.bias"), vec![h]),
                (format!("{p}.norm2.weight"), vec![h]),
                (format!("{p}.norm2.bias"), vec![h]),
            ]);
        }
        let mut tensors = read_tensors(path, &wanted)?.into_iter();
        let mut next = || tensors.next().expect("one tensor per name");
        let emb_norm = Norm {
            gamma: next(),
            beta: next(),
        };
        let mut embeddings = next();
        let types = next();
        for row in embeddings.chunks_exact_mut(h) {
            for (v, t) in row.iter_mut().zip(&types[..h]) {
                *v += t;
            }
        }
        let mut layers = Vec::with_capacity(config.layers);
        for _ in 0..config.layers {
            let (qkv, out) = (next(), next());
            let mut up_gate = next();
            up_gate.extend(next());
            let down = next();
            let norm1 = Norm {
                gamma: next(),
                beta: next(),
            };
            let norm2 = Norm {
                gamma: next(),
                beta: next(),
            };
            layers.push(LayerTensors {
                qkv,
                out,
                up_gate,
                down,
                norm1,
                norm2,
            });
        }
        Ok(Self {
            config,
            embeddings,
            emb_norm,
            layers,
        })
    }
}

/// The float tensors `wanted` of the file at `path`, each with its shape, in
/// order. Several threads read them, each through its own handle, largest
/// first: the copies out of the page cache, the page faults of the new
/// memory and the conversions then run side by side.
fn read_tensors(path: &Path, wanted: &[(String, Vec<usize>)]) -> Result<Vec<Vec<f32>>, Error> {
    let mut order: Vec<usize> = (0..wanted.len()).collect();
    order.sort_by_key(|&t| std::cmp::Reverse(wanted[t].1.iter().product::<usize>()));
    let threads = std::thread::available_parallelism()
        .map_or(1, |n| n.get())
        .min(8);
    let taken = AtomicUsize::new(0);
    let read = || -> Result<Vec<(usize, Vec<f32>)>, Error> {
        let mut file = SafeTensors::open(path)?;
        let mut done = Vec::new();
        while let Some(&t) = order.get(taken.fetch_add(1, Ordering::Relaxed)) {
            let (name, shape) = &wanted[t];
            done.push((t, file.floats(name, shape)?));
        }
        Ok(done)
    };
    let mut tensors: Vec<Option<Vec<f32>>> = vec![None; wanted.len()];
    std::thread::scope(|scope| -> Result<(), Error> {
        let workers = (0..threads)
            .map(|_| std::thread::Builder::new().spawn_scoped(scope, read))
            .collect::<Result<Vec<_>, _>>()
            .map_err(Error::Threads)?;
        for worker in workers {
            let done = worker
                .join()
                .unwrap_or_else(|panic| std::panic::resume_unwind(panic))?;
            for (t, values) in done {
                tensors[t] = Some(values);
            }
        }
        Ok(())
    })?;
    Ok(tensors
        .into_iter()
        .map(|t| t.expect("every tensor read"))
        .collect())
}

/// The normalized embeddings of `ids` into `x`, `norm.gamma.len()` floats
/// a token: the first step of the encoder, which both backends take on the
/// CPU.
pub(crate) fn embed_tokens(embeddings: &[f32], norm: &Norm, eps: f32, ids: &[u32], x: &mut [f32]) {
    let h = norm.gamma.len();
    for (row, &id) in x.chunks_exact_mut(h).zip(ids) {
        row.copy_from_slice(&embeddings[id as usize * h..][..h]);
    }
    layer_norm(&mut x[..ids.len() * h], &norm.gamma, &norm.beta, eps);
}

/// `(cos, sin)` of each position below `len` and each of the `half`
/// frequencies, in f32 as the reference computes them.
pub(crate) fn rotary_tables(base: f32, half: usize, len: usize) -> (Vec<f32>, Vec<f32>) {
    let dim = (2 * half) as f32;
    let inv_freq: Vec<f32> = (0..half)
        .map(|i| 1.0 / base.powf((2 * i) as f32 / dim))
        .collect();
    let mut cos = Vec::with_capacity(len * half);
    let mut sin = Vec::with_capacity(len * half);
    for t in 0..len {
        for &f in &inv_freq {
            let angle = t as f32 * f;
            cos.push(angle.cos());
            sin.push(angle.sin());
        }
    }
    (cos, sin)
}

struct Layer {
    qkv: Matrix,
    out: Matrix,
    up_gate: Matrix,
    down: Matrix,
    norm1: Norm,
    norm2: Norm,
}

/// Products packed at once while the model loads.
const PACKING_TASKS: usize = 4;
/// Rows of a product task: 16 groups of the micro-kernel.
const ROW_BLOCK: usize = 96;
/// Query rows of an attention task: few enough that the scores of one stay
/// in L2 for sequences of a few thousand tokens.
const QUERY_BLOCK: usize = 64;

/// The encoder with its weights packed for the CPU.
pub(crate) struct CpuModel {
    config: Config,
    embeddings: Vec<f32>,
    emb_norm: Norm,
    layers: Vec<Layer>,
    pool: Pool,
    cos: Vec<f32>,
    sin: Vec<f32>,
    /// Activations, kept from one batch to the next.
    x: Vec<f32>,
    qkv: Vec<f32>,
    /// Each sequence's keys and values per head, packed for the products.
    packed: Vec<Row>,
    context: Vec<f32>,
    inner: Vec<f32>,
}

impl CpuModel {
    pub(crate) fn new(tensors: Tensors, threads: usize) -> Result<Self, Error> {
        let Tensors {
            config,
            embeddings,
            emb_norm,
            layers,
        } = tensors;
        let (h, i) = (config.hidden, config.inner);
        let mut pool = Pool::new(threads, "coderankembed").map_err(Error::Threads)?;
        // The four products of every layer, packed on the pool's threads;
        // each task frees its weights once packed, so that the stored and
        // the packed weights of all layers are never held at once.
        let mut norms = Vec::with_capacity(layers.len());
        let mut stored = Vec::with_capacity(4 * layers.len());
        for l in layers {
            stored.extend([
                (l.qkv, 3 * h, h),
                (l.out, h, h),
                (l.up_gate, 2 * i, h),
                (l.down, h, i),
            ]);
            norms.push((l.norm1, l.norm2));
        }
        let stored: Vec<Mutex<Option<StoredProduct>>> =
            stored.into_iter().map(|w| Mutex::new(Some(w))).collect();
        let packed: Vec<Mutex<Option<Matrix>>> =
            (0..stored.len()).map(|_| Mutex::new(None)).collect();
        // A few at a time: each task holds a packed matrix of up to 19 MB
        // besides the stored weights, and more of them cost memory for
        // little time.
        let taken = AtomicUsize::new(0);
        pool.run(PACKING_TASKS.min(stored.len()), |_, _| {
            loop {
                let task = taken.fetch_add(1, Ordering::Relaxed);
                let Some(slot) = stored.get(task) else { break };
                let (w, n, k) = slot
                    .lock()
                    .unwrap_or_else(PoisonError::into_inner)
                    .take()
                    .expect("each product packed once");
                let matrix = Matrix::transposed(&w, n, k);
                drop(w);
                *packed[task].lock().unwrap_or_else(PoisonError::into_inner) = Some(matrix);
            }
        });
        let mut packed = packed.into_iter().map(|m| {
            m.into_inner()
                .unwrap_or_else(PoisonError::into_inner)
                .expect("every product packed")
        });
        let layers = norms
            .into_iter()
            .map(|(norm1, norm2)| Layer {
                qkv: packed.next().expect("a packed qkv"),
                out: packed.next().expect("a packed out"),
                up_gate: packed.next().expect("a packed up_gate"),
                down: packed.next().expect("a packed down"),
                norm1,
                norm2,
            })
            .collect();
        Ok(Self {
            config,
            embeddings,
            emb_norm,
            layers,
            pool,
            cos: Vec::new(),
            sin: Vec::new(),
            x: Vec::new(),
            qkv: Vec::new(),
            packed: Vec::new(),
            context: Vec::new(),
            inner: Vec::new(),
        })
    }

    pub(crate) fn threads(&self) -> usize {
        self.pool.threads()
    }

    /// The `[CLS]` hidden state of each sequence `seqs` of `ids`, `hidden`
    /// floats each, into `out`.
    pub(crate) fn embed(&mut self, ids: &[u32], seqs: &[Range<usize>], out: &mut [f32]) {
        let c = &self.config;
        let (h, i, d) = (c.hidden, c.inner, c.head_dim());
        let (t, n) = (ids.len(), seqs.len());
        assert_eq!(out.len(), n * h, "one embedding per sequence");
        let longest = seqs.iter().map(Range::len).max().unwrap_or(0);
        if self.cos.len() < longest * d / 2 {
            (self.cos, self.sin) = rotary_tables(c.rotary_base, d / 2, longest);
        }
        self.x.resize(t * h, 0.0);
        self.qkv.resize(t * 3 * h, 0.0);
        self.context.resize(t * h, 0.0);
        self.inner.resize(t * 2 * i, 0.0);
        embed_tokens(&self.embeddings, &self.emb_norm, c.eps, ids, &mut self.x);
        for (index, layer) in self.layers.iter().enumerate() {
            let last = index + 1 == self.layers.len();
            linear(
                &mut self.pool,
                &self.x,
                h,
                t,
                layer.qkv.panels(),
                &mut self.qkv,
                3 * h,
                false,
            );
            rotate(&mut self.pool, &mut self.qkv, seqs, c, &self.cos, &self.sin);
            attention(
                &mut self.pool,
                &self.qkv,
                seqs,
                last,
                c,
                &mut self.packed,
                &mut self.context,
            );
            // The last layer goes on with one row per sequence: its [CLS].
            let rows = if last {
                for (s, seq) in seqs.iter().enumerate() {
                    self.x
                        .copy_within(seq.start * h..(seq.start + 1) * h, s * h);
                }
                n
            } else {
                t
            };
            let x = &mut self.x[..rows * h];
            linear(
                &mut self.pool,
                &self.context,
                h,
                rows,
                layer.out.panels(),
                x,
                h,
                true,
            );
            layer_norm(x, &layer.norm1.gamma, &layer.norm1.beta, c.eps);
            let inner = &mut self.inner[..rows * 2 * i];
            linear(
                &mut self.pool,
                x,
                h,
                rows,
                layer.up_gate.panels(),
                inner,
                2 * i,
                false,
            );
            swiglu(inner, i);
            linear(
                &mut self.pool,
                inner,
                2 * i,
                rows,
                layer.down.panels(),
                x,
                h,
                true,
            );
            layer_norm(x, &layer.norm2.gamma, &layer.norm2.beta, c.eps);
        }
        out.copy_from_slice(&self.x[..n * h]);
    }
}

/// `out[r] (+)= a[r] w` for the `rows` rows of `a`, `a_stride` and `stride`
/// floats apart, split between the threads in blocks of rows and, when the
/// rows are few, of columns too.
#[expect(
    clippy::too_many_arguments,
    reason = "a product's operands and their layout"
)]
fn linear(
    pool: &mut Pool,
    a: &[f32],
    a_stride: usize,
    rows: usize,
    w: Panels<'_>,
    out: &mut [f32],
    stride: usize,
    accumulate: bool,
) {
    let row_blocks = rows.div_ceil(ROW_BLOCK);
    let panels = w.count();
    // At least four panels a task: below that, the row pointers and the
    // tiles' loads and stores cost more than the products.
    let column_blocks = (2 * pool.threads())
        .div_ceil(row_blocks)
        .clamp(1, panels.div_ceil(4));
    let panels_per_block = panels.div_ceil(column_blocks);
    let buffer = Shared::new(out);
    pool.run(row_blocks * column_blocks, |task, scratch| {
        let (rb, cb) = (task / column_blocks, task % column_blocks);
        let rows = rb * ROW_BLOCK..((rb + 1) * ROW_BLOCK).min(rows);
        let panels = cb * panels_per_block..((cb + 1) * panels_per_block).min(panels);
        let out = Out {
            buffer,
            at: rows.start * stride,
            stride,
        };
        // SAFETY: the tasks cover disjoint blocks of rows and panels.
        unsafe {
            multiply(
                a,
                a_stride,
                rows,
                w,
                panels,
                out,
                accumulate,
                &mut scratch.gemm,
            )
        };
    });
}

/// Rotate the queries and keys of `qkv` by their positions within their
/// sequences, and scale the queries by `1 / sqrt(head_dim)`.
fn rotate(
    pool: &mut Pool,
    qkv: &mut [f32],
    seqs: &[Range<usize>],
    c: &Config,
    cos: &[f32],
    sin: &[f32],
) {
    let (h, d) = (c.hidden, c.head_dim());
    let half = d / 2;
    let scale = c.query_scale();
    let width = 3 * h;
    let tokens = qkv.len() / width;
    let buffer = Shared::new(qkv);
    let blocks = tokens.div_ceil(ROW_BLOCK);
    pool.run(blocks, |block, _| {
        let rows = block * ROW_BLOCK..((block + 1) * ROW_BLOCK).min(tokens);
        // SAFETY: each task rotates its own rows.
        let part = unsafe { buffer.part(rows.start * width, rows.len() * width) };
        for (row, token) in part.chunks_exact_mut(width).zip(rows) {
            let seq = seqs.partition_point(|s| s.end <= token);
            let position = token - seqs[seq].start;
            let (cos, sin) = (
                &cos[position * half..][..half],
                &sin[position * half..][..half],
            );
            let (queries, keys) = row[..2 * h].split_at_mut(h);
            for x in queries.chunks_exact_mut(d).chain(keys.chunks_exact_mut(d)) {
                let (lo, hi) = x.split_at_mut(half);
                for (((a, b), &cos), &sin) in lo.iter_mut().zip(hi.iter_mut()).zip(cos).zip(sin) {
                    (*a, *b) = (*a * cos - *b * sin, *b * cos + *a * sin);
                }
            }
            for v in queries.iter_mut() {
                *v *= scale;
            }
        }
    });
}

/// Rows of one head's packed keys and values, for a sequence of `len`
/// tokens and heads of `d` channels.
fn head_rows(d: usize, len: usize) -> (usize, usize) {
    (panel_rows(d, len), panel_rows(len, d))
}

/// Pack each head's keys and values of every sequence into `packed`, as the
/// right operands of `q k^T` and `p v`: per sequence, per head, the keys
/// then the values. Returns where each sequence's heads start.
fn pack_heads(
    pool: &mut Pool,
    qkv: &[f32],
    seqs: &[Range<usize>],
    c: &Config,
    packed: &mut Vec<Row>,
) -> Vec<usize> {
    let (h, d, heads) = (c.hidden, c.head_dim(), c.heads);
    let width = 3 * h;
    let mut starts = Vec::with_capacity(seqs.len() + 1);
    let mut total = 0;
    for seq in seqs {
        starts.push(total);
        let (keys, values) = head_rows(d, seq.len());
        total += heads * (keys + values);
    }
    packed.resize(total, Row::default());
    let shared = Shared::new(packed);
    pool.run(seqs.len() * heads, |task, _| {
        let (s, head) = (task / heads, task % heads);
        let seq = &seqs[s];
        let len = seq.len();
        let (key_rows, value_rows) = head_rows(d, len);
        // SAFETY: each task packs its own sequence's head.
        let out = unsafe {
            shared.part(
                starts[s] + head * (key_rows + value_rows),
                key_rows + value_rows,
            )
        };
        let (keys, values) = out.split_at_mut(key_rows);
        let (k0, v0) = (h + head * d, 2 * h + head * d);
        for (j, row) in qkv[seq.start * width..seq.end * width]
            .chunks_exact(width)
            .enumerate()
        {
            for (x, &v) in row[k0..k0 + d].iter().enumerate() {
                keys[(j / NR) * d + x].0[j % NR] = v;
            }
            for (x, &v) in row[v0..v0 + d].iter().enumerate() {
                values[(x / NR) * len + j].0[x % NR] = v;
            }
        }
    });
    starts
}

/// Self-attention of every sequence `seqs` of `qkv` into `context`: for all
/// rows, or with `cls_only` for each sequence's first row alone, written to
/// row `s` for sequence `s`.
///
/// Each head's keys and values are packed once (`pack_heads`); the products
/// then run over blocks of query rows.
fn attention(
    pool: &mut Pool,
    qkv: &[f32],
    seqs: &[Range<usize>],
    cls_only: bool,
    c: &Config,
    packed: &mut Vec<Row>,
    context: &mut [f32],
) {
    let (h, d, heads) = (c.hidden, c.head_dim(), c.heads);
    let width = 3 * h;
    let starts = pack_heads(pool, qkv, seqs, c, packed);
    let packed: &[Row] = packed;
    // Tasks: each sequence's heads, each over blocks of its query rows.
    let mut tasks = Vec::new();
    for (s, seq) in seqs.iter().enumerate() {
        let q = if cls_only {
            seq.start..seq.start + 1
        } else {
            seq.clone()
        };
        for head in 0..heads {
            let mut start = q.start;
            while start < q.end {
                let end = (start + QUERY_BLOCK).min(q.end);
                tasks.push((s, head, start..end));
                start = end;
            }
        }
    }
    let buffer = Shared::new(context);
    pool.run(tasks.len(), |task, scratch| {
        let (s, head, ref rows) = tasks[task];
        let len = seqs[s].len();
        let (key_rows, value_rows) = head_rows(d, len);
        let at = starts[s] + head * (key_rows + value_rows);
        let keys = Panels {
            k: d,
            n: len,
            rows: &packed[at..at + key_rows],
        };
        let values = Panels {
            k: len,
            n: d,
            rows: &packed[at + key_rows..at + key_rows + value_rows],
        };
        let m = rows.len();
        scratch.floats.resize(m * len, 0.0);
        let scores = &mut scratch.floats[..m * len];
        multiply_into(
            &qkv[head * d..],
            width,
            rows.clone(),
            keys,
            scores,
            len,
            &mut scratch.gemm,
        );
        softmax_rows(scores, len);
        let first = if cls_only { s } else { rows.start };
        let out = Out {
            buffer,
            at: first * h + head * d,
            stride: h,
        };
        // SAFETY: each task writes its own head's columns of its own rows.
        unsafe {
            multiply(
                scores,
                len,
                0..m,
                values,
                0..values.count(),
                out,
                false,
                &mut scratch.gemm,
            )
        };
    });
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// Smooth pseudo-random values in -0.5..0.5.
    pub(crate) fn values(len: usize, seed: f32) -> Vec<f32> {
        (0..len)
            .map(|i| ((i as f32 * 0.37 + seed).sin()) * 0.5)
            .collect()
    }

    /// A two-layer model of 128 channels in two heads, with fixed
    /// pseudo-random weights that half floats hold exactly.
    pub(crate) fn tiny() -> Tensors {
        let (h, i) = (128, 64);
        let exact =
            |v: Vec<f32>| -> Vec<f32> { v.iter().map(|x| (x * 256.0).round() / 256.0).collect() };
        let vocab = 50;
        let config = Config {
            hidden: h,
            heads: h / 64,
            inner: i,
            layers: 2,
            vocab,
            eps: 1e-12,
            rotary_base: 1000.0,
        };
        let norm = |seed: f32| Norm {
            gamma: values(h, seed).iter().map(|v| 1.0 + v).collect(),
            beta: values(h, seed + 0.5),
        };
        let layers = (0..2)
            .map(|l| {
                let seed = l as f32 * 10.0;
                LayerTensors {
                    // Small enough that attention does not pick one key
                    // alone, so a missing query scale changes the result.
                    qkv: exact(
                        values(3 * h * h, seed + 1.0)
                            .iter()
                            .map(|v| v / 8.0)
                            .collect(),
                    ),
                    out: exact(values(h * h, seed + 2.0)),
                    up_gate: exact(values(2 * i * h, seed + 3.0)),
                    down: exact(values(h * i, seed + 4.0)),
                    norm1: norm(seed + 5.0),
                    norm2: norm(seed + 6.0),
                }
            })
            .collect();
        Tensors {
            config,
            embeddings: values(vocab * h, 7.0).iter().map(|v| v * 4.0).collect(),
            emb_norm: norm(8.0),
            layers,
        }
    }

    /// `x w^T`, `w` being `n` rows of `x`'s width, in f64.
    fn product(x: &[f64], w: &[f32], n: usize) -> Vec<f64> {
        let k = w.len() / n;
        x.chunks(k)
            .flat_map(|row| {
                w.chunks(k).map(|wr| {
                    row.iter()
                        .zip(wr)
                        .map(|(a, b)| a * f64::from(*b))
                        .sum::<f64>()
                })
            })
            .collect()
    }

    fn normalized(x: &mut [f64], norm: &Norm, eps: f32) {
        for row in x.chunks_mut(norm.gamma.len()) {
            let mean = row.iter().sum::<f64>() / row.len() as f64;
            let var = row.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / row.len() as f64;
            let inv = 1.0 / (var + f64::from(eps)).sqrt();
            for ((v, g), b) in row.iter_mut().zip(&norm.gamma).zip(&norm.beta) {
                *v = (*v - mean) * inv * f64::from(*g) + f64::from(*b);
            }
        }
    }

    /// Softmax attention of one sequence's rotated `qkv` rows, every head.
    fn attention(c: &Config, qkv: &[f64]) -> Vec<f64> {
        let (h, d) = (c.hidden, c.head_dim());
        let len = qkv.len() / (3 * h);
        let mut context = vec![0.0; len * h];
        for head in 0..c.heads {
            let at = |row: usize, part: usize| &qkv[row * 3 * h + part * h + head * d..][..d];
            for q in 0..len {
                let scores: Vec<f64> = (0..len)
                    .map(|k| {
                        let dot: f64 = at(q, 0).iter().zip(at(k, 1)).map(|(a, b)| a * b).sum();
                        dot / (d as f64).sqrt()
                    })
                    .collect();
                let max = scores.iter().copied().fold(f64::MIN, f64::max);
                let weights: Vec<f64> = scores.iter().map(|s| (s - max).exp()).collect();
                let total: f64 = weights.iter().sum();
                for (k, w) in weights.iter().enumerate() {
                    for (out, v) in context[q * h + head * d..][..d].iter_mut().zip(at(k, 2)) {
                        *out += w / total * v;
                    }
                }
            }
        }
        context
    }

    /// The `[CLS]` hidden state of one sequence, from NomicBERT's definition
    /// in f64: the oracle of the CPU encoder.
    fn definition(t: &Tensors, ids: &[u32]) -> Vec<f64> {
        let c = &t.config;
        let (h, i, d) = (c.hidden, c.inner, c.head_dim());
        let mut x: Vec<f64> = ids
            .iter()
            .flat_map(|&id| &t.embeddings[id as usize * h..][..h])
            .map(|&v| f64::from(v))
            .collect();
        normalized(&mut x, &t.emb_norm, c.eps);
        for layer in &t.layers {
            let mut qkv = product(&x, &layer.qkv, 3 * h);
            for (pos, row) in qkv.chunks_mut(3 * h).enumerate() {
                for head in row[..2 * h].chunks_mut(d) {
                    for j in 0..d / 2 {
                        let freq = 1.0 / f64::from(c.rotary_base).powf((2 * j) as f64 / d as f64);
                        let (sin, cos) = (pos as f64 * freq).sin_cos();
                        let (a, b) = (head[j], head[j + d / 2]);
                        (head[j], head[j + d / 2]) = (a * cos - b * sin, b * cos + a * sin);
                    }
                }
            }
            let context = attention(c, &qkv);
            for (v, o) in x.iter_mut().zip(product(&context, &layer.out, h)) {
                *v += o;
            }
            normalized(&mut x, &layer.norm1, c.eps);
            let up_gate = product(&x, &layer.up_gate, 2 * i);
            let inner: Vec<f64> = up_gate
                .chunks(2 * i)
                .flat_map(|row| {
                    let (up, gate) = row.split_at(i);
                    up.iter().zip(gate).map(|(u, g)| u * g / (1.0 + (-g).exp()))
                })
                .collect();
            for (v, o) in x.iter_mut().zip(product(&inner, &layer.down, h)) {
                *v += o;
            }
            normalized(&mut x, &layer.norm2, c.eps);
        }
        x.truncate(h);
        x
    }

    /// Three sequences in one batch of more than a row block, one longer
    /// than a query block, against each sequence alone in f64.
    #[test]
    fn cpu_model_matches_definition() {
        let ids: Vec<u32> = (0..130).map(|i| (i * 7 % 50) as u32).collect();
        let seqs = [0..2usize, 2..20, 20..130];
        let tensors = tiny();
        let want: Vec<Vec<f64>> = seqs
            .iter()
            .map(|s| definition(&tensors, &ids[s.clone()]))
            .collect();
        let mut model = CpuModel::new(tensors, 3).unwrap();
        let mut got = vec![0.0; 3 * 128];
        model.embed(&ids, &seqs, &mut got);
        for (got, want) in got.chunks(128).zip(&want) {
            for (g, w) in got.iter().zip(want) {
                assert!((f64::from(*g) - w).abs() < 1e-4, "{g} against {w}");
            }
        }
    }

    #[test]
    fn published_config_reads_its_shape() {
        let model = include_str!("../tests/data/config.json");
        let c = Config::parse(model.as_bytes()).unwrap();
        assert_eq!(
            (c.hidden, c.heads, c.inner, c.layers, c.vocab),
            (768, 12, 3072, 12, 30528)
        );
        assert_eq!((c.eps, c.rotary_base), (1e-12, 1000.0));
    }

    #[test]
    fn each_variant_key_is_refused() {
        let model = include_str!("../tests/data/config.json");
        // Each key as CodeRankEmbed publishes it, then as another variant sets it.
        for (key, published, other) in [
            ("activation_function", "\"swiglu\"", "\"gelu\""),
            ("causal", "false", "true"),
            ("mlp_fc1_bias", "false", "true"),
            ("mlp_fc2_bias", "false", "true"),
            ("parallel_block", "false", "true"),
            ("prenorm", "false", "true"),
            ("qkv_proj_bias", "false", "true"),
            ("rotary_emb_fraction", "1.0", "0.5"),
            ("rotary_emb_interleaved", "false", "true"),
            ("rotary_emb_scale_base", "null", "512"),
            ("rotary_scaling_factor", "null", "2.0"),
            ("use_rms_norm", "false", "true"),
        ] {
            let from = format!("\"{key}\": {published}");
            assert!(model.contains(&from), "{from}");
            let variant = model.replace(&from, &format!("\"{key}\": {other}"));
            let refused = Config::parse(variant.as_bytes()).err();
            assert!(refused.is_some_and(|e| e.contains(key)), "{key}");
        }
        let moe = model.replacen('{', "{\"moe_every_n_layers\": 2,", 1);
        let refused = Config::parse(moe.as_bytes()).err();
        assert!(refused.is_some_and(|e| e.contains("moe_every_n_layers")));
    }

    #[test]
    fn layers_beyond_the_file_are_refused() {
        let dir = std::env::temp_dir().join(format!("coderankembed-layers-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("model.safetensors");
        let mut bytes = 2u64.to_le_bytes().to_vec();
        bytes.extend_from_slice(b"{}");
        std::fs::write(&path, bytes).unwrap();
        let model = include_str!("../tests/data/config.json");
        // Refused by the count, before a name is built: nine times the
        // second passes u64::MAX.
        for layers in [12, 2_049_638_230_412_172_402] {
            let mut config = Config::parse(model.as_bytes()).unwrap();
            config.layers = layers;
            let Err(Error::Model(message)) = Tensors::load(config, &path) else {
                panic!("{layers} layers loaded");
            };
            assert!(message.contains(&format!("{layers} layers")), "{message}");
        }
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn rotary_tables_start_at_the_identity() {
        let (cos, sin) = rotary_tables(1000.0, 2, 3);
        assert_eq!((&cos[..2], &sin[..2]), (&[1.0, 1.0][..], &[0.0, 0.0][..]));
        // Position 1, frequency 1 / 1000^(2/4): angle 1/sqrt(1000).
        assert!((sin[3] - (1.0f32 / 1000f32.sqrt()).sin()).abs() < 1e-7);
    }
}
