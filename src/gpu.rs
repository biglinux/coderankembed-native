//! The encoder on a Vulkan GPU: the weights in half floats, the arithmetic
//! and the activations in floats.
//!
//! For a batch, the token embeddings, computed on the CPU, and a block of
//! 32-bit words (each token's position, the attention tasks and each
//! sequence's `[CLS]` row) go in, the `[CLS]` hidden states come out; its
//! layers run in as few submissions as a driver's job timeout allows.
//! The layers run as in [`CpuModel`](crate::model::CpuModel): over all
//! tokens without padding, the last layer over the `[CLS]` rows alone.

use std::ops::Range;

use crate::Error;
use crate::encoder::{BATCH_SEQS, BATCH_TOKENS};
use crate::half::floats_to_halves;
use crate::model::{Config, Norm, Tensors, embed_tokens, rotary_tables};
use crate::vulkan::{Buffer, Commands, Gpu, Kernel};

/// Positions the rotary tables cover: the longest text the model reads.
const MAX_POSITIONS: usize = BATCH_TOKENS;
/// Query rows of an attention task, the workgroup size of attention.comp.
const QUERY_BLOCK: usize = 64;
/// Words of one attention task.
const TASK_WORDS: usize = 8;
/// The channels of one head attention.comp is compiled for.
const HEAD_DIM: usize = 64;
/// Rows and columns of a gemm.comp workgroup's tile, which the widths of the
/// products are multiples of.
const GEMM_TILE: usize = 128;
/// Products of at most this many rows run gemm_rows.comp, which reads each
/// weight once, over this many columns a workgroup.
const FEW_ROWS: usize = 32;

struct Kernels {
    attention: Kernel,
    gather: Kernel,
    gemm: Kernel,
    gemm_rows: Kernel,
    layer_norm: Kernel,
    rope: Kernel,
    swiglu: Kernel,
}

struct Layer {
    qkv: Buffer,
    out: Buffer,
    up_gate: Buffer,
    down: Buffer,
    norm1: Buffer,
    norm2: Buffer,
}

/// Device buffers sized for a batch of [`BATCH_TOKENS`] tokens.
struct Activations {
    x: Buffer,
    qkv: Buffer,
    context: Buffer,
    inner: Buffer,
    cls: Buffer,
    meta: Buffer,
    /// Host buffers: the embeddings and words going in, the result out.
    input: Buffer,
    meta_input: Buffer,
    output: Buffer,
}

pub(crate) struct GpuModel {
    gpu: Gpu,
    kernels: Kernels,
    config: Config,
    embeddings: Vec<f32>,
    emb_norm: Norm,
    layers: Vec<Layer>,
    /// The rotary cos table, then the sin table.
    tables: Buffer,
    buffers: Activations,
    /// The embeddings of a batch, before they are copied in.
    x: Vec<f32>,
}

/// gemm.comp as `vendor`'s GPUs run it best within the parity tests'
/// bounds: products in half floats on AMD's and NVIDIA's, 1.3 to 1.6 times
/// faster there; in floats elsewhere, Intel's being slower and less exact
/// in half floats.
fn gemm_spirv(vendor: u32) -> &'static [u8] {
    const AMD: u32 = 0x1002;
    const NVIDIA: u32 = 0x10de;
    if vendor == AMD || vendor == NVIDIA {
        include_bytes!(concat!(env!("OUT_DIR"), "/gemm_half.spv"))
    } else {
        include_bytes!(concat!(env!("OUT_DIR"), "/gemm.spv"))
    }
}

fn kernel(gpu: &Gpu, spirv: &[u8], bindings: u32, push_words: u32) -> Result<Kernel, String> {
    gpu.kernel(spirv, bindings, 4 * push_words)
}

/// What a weights buffer holds: a product's weights as half floats, or
/// floats as they are.
enum Upload {
    Halves(Vec<f32>),
    Floats(Vec<f32>),
}

impl Upload {
    fn len(&self) -> usize {
        match self {
            Upload::Halves(values) => 2 * values.len(),
            Upload::Floats(values) => 4 * values.len(),
        }
    }

    fn write(&self, dst: &mut [u8]) {
        match self {
            Upload::Halves(values) => floats_to_halves(values, dst.as_chunks_mut::<2>().0),
            Upload::Floats(values) => {
                for (d, v) in dst.as_chunks_mut::<4>().0.iter_mut().zip(values) {
                    *d = v.to_le_bytes();
                }
            }
        }
    }
}

fn push(words: &[u32]) -> Vec<u8> {
    words.iter().flat_map(|w| w.to_le_bytes()).collect()
}

fn groups(items: usize, per_group: usize) -> u32 {
    u32::try_from(items.div_ceil(per_group)).expect("batches are capped far below u32::MAX groups")
}

fn word(n: usize) -> u32 {
    u32::try_from(n).expect("batches are capped far below u32::MAX")
}

impl GpuModel {
    pub(crate) fn new(tensors: Tensors, gpu: Gpu) -> Result<Self, Error> {
        Self::build(tensors, gpu).map_err(Error::Gpu)
    }

    fn build(tensors: Tensors, mut gpu: Gpu) -> Result<Self, String> {
        let Tensors {
            config,
            embeddings,
            emb_norm,
            layers,
        } = tensors;
        let (h, i) = (config.hidden, config.inner);
        if config.head_dim() != HEAD_DIM
            || !h.is_multiple_of(GEMM_TILE)
            || !(2 * i).is_multiple_of(GEMM_TILE)
            || !i.is_multiple_of(32)
        {
            return Err(format!(
                "the shaders take heads of {HEAD_DIM} channels and widths in multiples of {GEMM_TILE}, \
                 not {h} channels in {} heads and {i} inner",
                config.heads
            ));
        }
        let kernels = Kernels {
            attention: kernel(
                &gpu,
                include_bytes!(concat!(env!("OUT_DIR"), "/attention.spv")),
                3,
                2,
            )?,
            gather: kernel(
                &gpu,
                include_bytes!(concat!(env!("OUT_DIR"), "/gather.spv")),
                3,
                3,
            )?,
            gemm: kernel(&gpu, gemm_spirv(gpu.vendor()), 3, 6)?,
            gemm_rows: kernel(
                &gpu,
                include_bytes!(concat!(env!("OUT_DIR"), "/gemm_rows.spv")),
                3,
                6,
            )?,
            layer_norm: kernel(
                &gpu,
                include_bytes!(concat!(env!("OUT_DIR"), "/layer_norm.spv")),
                2,
                2,
            )?,
            rope: kernel(
                &gpu,
                include_bytes!(concat!(env!("OUT_DIR"), "/rope.spv")),
                3,
                5,
            )?,
            swiglu: kernel(
                &gpu,
                include_bytes!(concat!(env!("OUT_DIR"), "/swiglu.spv")),
                1,
                1,
            )?,
        };
        let (mut cos, sin) = rotary_tables(config.rotary_base, HEAD_DIM / 2, MAX_POSITIONS);
        cos.extend(sin);
        // Per layer the four products' weights in half floats and the two
        // LayerNorms' scale then shift in floats, then the rotary tables.
        // Each is freed once written, so that the floats and the half floats
        // of all the weights are never held at once.
        let layer_count = layers.len();
        let mut uploads = Vec::with_capacity(6 * layer_count + 1);
        for l in layers {
            let norm = |n: Norm| Upload::Floats([n.gamma, n.beta].concat());
            uploads.extend([
                Upload::Halves(l.qkv),
                Upload::Halves(l.out),
                Upload::Halves(l.up_gate),
                Upload::Halves(l.down),
                norm(l.norm1),
                norm(l.norm2),
            ]);
        }
        uploads.push(Upload::Floats(cos));
        let sizes: Vec<usize> = uploads.iter().map(Upload::len).collect();
        let mut uploads: Vec<Option<Upload>> = uploads.into_iter().map(Some).collect();
        let mut buffers = gpu
            .upload_many(&sizes, |i, dst| {
                uploads[i]
                    .take()
                    .expect("each upload written once")
                    .write(dst);
            })?
            .into_iter();
        let mut next = || buffers.next().expect("one buffer per upload");
        let uploaded = (0..layer_count)
            .map(|_| Layer {
                qkv: next(),
                out: next(),
                up_gate: next(),
                down: next(),
                norm1: next(),
                norm2: next(),
            })
            .collect();
        let tables = next();
        let float = |n: usize| gpu.buffer(4 * n);
        let meta_words = meta_words(BATCH_TOKENS, BATCH_SEQS);
        let buffers = Activations {
            x: float(BATCH_TOKENS * h)?,
            qkv: float(BATCH_TOKENS * 3 * h)?,
            context: float(BATCH_TOKENS * h)?,
            inner: float(BATCH_TOKENS * 2 * i)?,
            cls: float(BATCH_SEQS * h)?,
            meta: float(meta_words)?,
            input: gpu.host_buffer(4 * BATCH_TOKENS * h, false)?,
            meta_input: gpu.host_buffer(4 * meta_words, false)?,
            output: gpu.host_buffer(4 * BATCH_SEQS * h, true)?,
        };
        Ok(Self {
            gpu,
            kernels,
            config,
            embeddings,
            emb_norm,
            layers: uploaded,
            tables,
            buffers,
            x: Vec::new(),
        })
    }

    pub(crate) fn name(&self) -> &str {
        self.gpu.name()
    }

    /// The `[CLS]` hidden state of each sequence `seqs` of `ids`, `hidden`
    /// floats each, into `out`.
    pub(crate) fn embed(
        &mut self,
        ids: &[u32],
        seqs: &[Range<usize>],
        out: &mut [f32],
    ) -> Result<(), Error> {
        // The host buffers below panic once a failed wait left the GPU busy.
        self.gpu.usable().map_err(Error::Gpu)?;
        let c = &self.config;
        let h = c.hidden;
        let (t, n) = (ids.len(), seqs.len());
        assert!(
            t <= BATCH_TOKENS && n <= BATCH_SEQS && out.len() == n * h,
            "a batch too large"
        );
        self.x.resize(t * h, 0.0);
        embed_tokens(&self.embeddings, &self.emb_norm, c.eps, ids, &mut self.x);
        let input = self.buffers.input.bytes_mut();
        for (dst, v) in input.as_chunks_mut::<4>().0.iter_mut().zip(&self.x) {
            *dst = v.to_le_bytes();
        }
        let (meta, plan) = batch_words(seqs, t);
        let meta_bytes = 4 * meta.len();
        let meta_input = self.buffers.meta_input.bytes_mut();
        for (dst, w) in meta_input.as_chunks_mut::<4>().0.iter_mut().zip(&meta) {
            *dst = w.to_le_bytes();
        }
        let Self {
            gpu,
            kernels,
            config,
            layers,
            tables,
            buffers,
            ..
        } = self;
        // A driver resets a GPU that runs one job past its timeout (amdgpu's
        // ten seconds), and the desktop waits on the job meanwhile; but each
        // submission also costs a wait. A job runs at most a full batch's
        // worth of token-layers: one layer of 8192 tokens, every layer of a
        // short query.
        let per_submission = (BATCH_TOKENS / t).clamp(1, layers.len());
        for (first, group) in (0..layers.len())
            .step_by(per_submission)
            .zip(layers.chunks(per_submission))
        {
            let last_group = first + group.len() == layers.len();
            gpu.run(|commands| {
                if first == 0 {
                    commands.copy(&buffers.input, 0, &buffers.x, 4 * t * h)?;
                    commands.copy(&buffers.meta_input, 0, &buffers.meta, meta_bytes)?;
                    commands.barrier();
                }
                for (j, layer) in group.iter().enumerate() {
                    let last = last_group && j + 1 == group.len();
                    record(
                        commands, kernels, config, layer, last, tables, buffers, &plan,
                    )?;
                }
                if last_group {
                    commands.copy(&buffers.cls, 0, &buffers.output, 4 * n * h)?;
                }
                Ok(())
            })
            .map_err(Error::Gpu)?;
        }
        for (v, b) in out
            .iter_mut()
            .zip(self.buffers.output.bytes().as_chunks::<4>().0)
        {
            *v = f32::from_le_bytes(*b);
        }
        Ok(())
    }
}

/// The words of a batch: each token's position, then the attention tasks
/// of all rows, then those of the `[CLS]` rows, then the `[CLS]` rows; and
/// where each part starts.
fn batch_words(seqs: &[Range<usize>], tokens: usize) -> (Vec<u32>, Plan) {
    let n = seqs.len();
    let mut meta: Vec<u32> = Vec::with_capacity(meta_words(tokens, n));
    for seq in seqs {
        meta.extend(0..word(seq.len()));
    }
    let all_tasks = meta.len();
    for seq in seqs {
        let mut q = seq.start;
        while q < seq.end {
            let count = QUERY_BLOCK.min(seq.end - q);
            meta.extend([seq.start, seq.len(), q, count, q, 0, 0, 0].map(word));
            q += count;
        }
    }
    let cls_tasks = meta.len();
    for (s, seq) in seqs.iter().enumerate() {
        meta.extend([seq.start, seq.len(), seq.start, 1, s, 0, 0, 0].map(word));
    }
    let cls_rows = meta.len();
    meta.extend(seqs.iter().map(|seq| word(seq.start)));
    let plan = Plan {
        tokens,
        seqs: n,
        all_tasks,
        all_task_count: (cls_tasks - all_tasks) / TASK_WORDS,
        cls_tasks,
        cls_rows,
    };
    (meta, plan)
}

/// The most words [`batch_words`] gives for `tokens` tokens in `seqs`
/// sequences: a sequence's tasks are its rows in blocks of `QUERY_BLOCK`,
/// one more for its last block and one for its `[CLS]` row.
fn meta_words(tokens: usize, seqs: usize) -> usize {
    tokens + (tokens / QUERY_BLOCK + 2 * seqs) * TASK_WORDS + seqs
}

/// Where a batch's words lie in the meta buffer, and its sizes.
struct Plan {
    tokens: usize,
    seqs: usize,
    all_tasks: usize,
    all_task_count: usize,
    cls_tasks: usize,
    cls_rows: usize,
}

/// `out[rows] (+)= a[rows] w^T` with `gemm.comp`, or `gemm_rows.comp` for
/// a few rows.
struct Product<'a> {
    a: &'a Buffer,
    a_stride: usize,
    w: &'a Buffer,
    k: usize,
    n: usize,
    out: &'a Buffer,
    out_stride: usize,
    accumulate: bool,
}

fn gemm(c: &mut Commands<'_>, k: &Kernels, rows: usize, p: &Product<'_>) -> Result<(), String> {
    let words = [
        rows,
        p.n,
        p.k,
        p.a_stride,
        p.out_stride,
        usize::from(p.accumulate),
    ]
    .map(word);
    let (kernel, workgroups) = if rows <= FEW_ROWS {
        (&k.gemm_rows, [groups(p.n, FEW_ROWS), 1, 1])
    } else {
        (
            &k.gemm,
            [groups(p.n, GEMM_TILE), groups(rows, GEMM_TILE), 1],
        )
    };
    c.dispatch(kernel, &[p.a, p.w, p.out], &push(&words), workgroups)?;
    c.barrier();
    Ok(())
}

fn layer_norm(
    c: &mut Commands<'_>,
    kernel: &Kernel,
    x: &Buffer,
    norm: &Buffer,
    rows: usize,
    config: &Config,
) -> Result<(), String> {
    let words = [word(config.hidden), config.eps.to_bits()];
    c.dispatch(kernel, &[x, norm], &push(&words), [word(rows), 1, 1])?;
    c.barrier();
    Ok(())
}

/// `up * silu(gate)` over `rows` rows of `inner`, up then gate.
fn swiglu(
    c: &mut Commands<'_>,
    kernel: &Kernel,
    inner: &Buffer,
    rows: usize,
    width: usize,
) -> Result<(), String> {
    let words = [word(width)];
    // A workgroup row per row: a full batch's rows times `width` / 256 pass
    // the 65535 workgroups Vulkan guarantees on one axis.
    c.dispatch(
        kernel,
        &[inner],
        &push(&words),
        [groups(width, 256), word(rows), 1],
    )?;
    c.barrier();
    Ok(())
}

/// Record one layer; the last goes on with the `[CLS]` rows alone.
#[expect(
    clippy::too_many_arguments,
    reason = "a layer's kernels, weights and buffers"
)]
fn record(
    c: &mut Commands<'_>,
    k: &Kernels,
    config: &Config,
    layer: &Layer,
    last: bool,
    tables: &Buffer,
    b: &Activations,
    plan: &Plan,
) -> Result<(), String> {
    let (h, i, heads) = (config.hidden, config.inner, config.heads);
    let (t, n) = (plan.tokens, plan.seqs);
    let scale = config.query_scale();
    gemm(
        c,
        k,
        t,
        &Product {
            a: &b.x,
            a_stride: h,
            w: &layer.qkv,
            k: h,
            n: 3 * h,
            out: &b.qkv,
            out_stride: 3 * h,
            accumulate: false,
        },
    )?;
    let rope = [
        word(t),
        word(h),
        word(HEAD_DIM / 2),
        word(MAX_POSITIONS * HEAD_DIM / 2),
        scale.to_bits(),
    ];
    c.dispatch(
        &k.rope,
        &[&b.qkv, &b.meta, tables],
        &push(&rope),
        [groups(t * h, 256), 1, 1],
    )?;
    c.barrier();
    let (rows, x, tasks, task_count) = if last {
        let gather = [word(n), word(h), word(plan.cls_rows)];
        c.dispatch(
            &k.gather,
            &[&b.x, &b.meta, &b.cls],
            &push(&gather),
            [groups(n * h, 256), 1, 1],
        )?;
        (n, &b.cls, plan.cls_tasks, n)
    } else {
        (t, &b.x, plan.all_tasks, plan.all_task_count)
    };
    let attention = [word(h), word(tasks)];
    c.dispatch(
        &k.attention,
        &[&b.qkv, &b.meta, &b.context],
        &push(&attention),
        [word(task_count), word(heads), 1],
    )?;
    c.barrier();
    gemm(
        c,
        k,
        rows,
        &Product {
            a: &b.context,
            a_stride: h,
            w: &layer.out,
            k: h,
            n: h,
            out: x,
            out_stride: h,
            accumulate: true,
        },
    )?;
    layer_norm(c, &k.layer_norm, x, &layer.norm1, rows, config)?;
    gemm(
        c,
        k,
        rows,
        &Product {
            a: x,
            a_stride: h,
            w: &layer.up_gate,
            k: h,
            n: 2 * i,
            out: &b.inner,
            out_stride: 2 * i,
            accumulate: false,
        },
    )?;
    swiglu(c, &k.swiglu, &b.inner, rows, i)?;
    gemm(
        c,
        k,
        rows,
        &Product {
            a: &b.inner,
            a_stride: 2 * i,
            w: &layer.down,
            k: i,
            n: h,
            out: x,
            out_stride: h,
            accumulate: true,
        },
    )?;
    layer_norm(c, &k.layer_norm, x, &layer.norm2, rows, config)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gemm::{GemmScratch, Matrix, multiply_into};
    use crate::math::{layer_norm as cpu_layer_norm, swiglu as cpu_swiglu};
    use crate::model::tests::{tiny, values};

    fn bytes_of(values: &[f32]) -> Vec<u8> {
        values.iter().flat_map(|v| v.to_le_bytes()).collect()
    }

    /// A device buffer holding `bytes`.
    fn upload(gpu: &mut Gpu, bytes: &[u8]) -> Buffer {
        let mut buffers = gpu
            .upload_many(&[bytes.len()], |_, dst| dst.copy_from_slice(bytes))
            .expect("the upload runs");
        buffers.remove(0)
    }

    fn halves_of(values: &[f32]) -> Vec<u8> {
        let mut bytes = vec![0u8; 2 * values.len()];
        Upload::Halves(values.to_vec()).write(&mut bytes);
        bytes
    }

    fn read(gpu: &mut Gpu, buffer: &Buffer, len: usize) -> Vec<f32> {
        let host = gpu.host_buffer(4 * len, true).expect("a host buffer");
        gpu.run(|c| c.copy(buffer, 0, &host, 4 * len))
            .expect("the copy runs");
        host.bytes()
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b))
            .collect()
    }

    fn largest_gap(a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(a.len(), b.len());
        a.iter()
            .zip(b)
            .map(|(x, y)| {
                assert!(x.is_finite() && y.is_finite(), "{x} against {y}");
                (x - y).abs()
            })
            .fold(0.0, f32::max)
    }

    #[test]
    #[ignore = "needs a Vulkan GPU"]
    fn gpu_gemm_matches_cpu() {
        let mut gpu = Gpu::open().expect("a GPU");
        let rows_spirv = include_bytes!(concat!(env!("OUT_DIR"), "/gemm_rows.spv"));
        let tiled = kernel(&gpu, gemm_spirv(gpu.vendor()), 3, 6).unwrap();
        let few = kernel(&gpu, rows_spirv, 3, 6).unwrap();
        // The last has a k below one stage of gemm_rows.comp.
        for (m, kk, n) in [
            (1, 768, 128),
            (21, 3072, 768),
            (170, 64, 256),
            (32, 96, 128),
        ] {
            let a = values(m * kk, 0.3);
            // Weights that half floats hold exactly.
            let w: Vec<f32> = values(n * kk, 1.1)
                .iter()
                .map(|v| (v * 256.0).round() / 256.0)
                .collect();
            let start = values(m * n, 2.0);
            let mut want = vec![0.0; m * n];
            multiply_into(
                &a,
                kk,
                0..m,
                Matrix::transposed(&w, n, kk).panels(),
                &mut want,
                n,
                &mut GemmScratch::default(),
            );
            for (v, s) in want.iter_mut().zip(&start) {
                *v += s;
            }
            let mut runs = vec![(
                "gemm",
                &tiled,
                [groups(n, GEMM_TILE), groups(m, GEMM_TILE), 1],
            )];
            if m <= FEW_ROWS {
                runs.push(("gemm_rows", &few, [groups(n, FEW_ROWS), 1, 1]));
            }
            for (name, k, workgroups) in runs {
                let ab = upload(&mut gpu, &bytes_of(&a));
                let wb = upload(&mut gpu, &halves_of(&w));
                let ob = upload(&mut gpu, &bytes_of(&start));
                let words = [m, n, kk, kk, n, 1].map(word);
                gpu.run(|c| c.dispatch(k, &[&ab, &wb, &ob], &push(&words), workgroups))
                    .unwrap();
                // gemm_rows.comp sums floats in another order than the CPU;
                // gemm.comp in half floats is 3.4e-4 of the largest value off
                // at most here.
                let tolerance = if name == "gemm" { 1e-3 } else { 1e-5 };
                let largest = want.iter().fold(0.0f32, |l, v| l.max(v.abs()));
                let gap = largest_gap(&read(&mut gpu, &ob, m * n), &want);
                assert!(
                    gap < tolerance * largest,
                    "{name} {m}x{kk}x{n}: {gap} of {largest}"
                );
            }
        }
    }

    #[test]
    #[ignore = "needs a Vulkan GPU"]
    fn gpu_norm_and_swiglu_match_cpu() {
        let mut gpu = Gpu::open().expect("a GPU");
        let ln = kernel(
            &gpu,
            include_bytes!(concat!(env!("OUT_DIR"), "/layer_norm.spv")),
            2,
            2,
        )
        .unwrap();
        let sw = kernel(
            &gpu,
            include_bytes!(concat!(env!("OUT_DIR"), "/swiglu.spv")),
            1,
            1,
        )
        .unwrap();
        let (rows, width) = (5, 768);
        let x: Vec<f32> = values(rows * width, 0.4)
            .iter()
            .map(|v| v * 9.0 + 3.0)
            .collect();
        let norm = Norm {
            gamma: values(width, 1.0),
            beta: values(width, 2.0),
        };
        let xb = upload(&mut gpu, &bytes_of(&x));
        let nb = upload(
            &mut gpu,
            &bytes_of(&[norm.gamma.clone(), norm.beta.clone()].concat()),
        );
        gpu.run(|c| {
            c.dispatch(
                &ln,
                &[&xb, &nb],
                &push(&[word(width), 1e-12f32.to_bits()]),
                [word(rows), 1, 1],
            )
        })
        .unwrap();
        let mut want = x;
        cpu_layer_norm(&mut want, &norm.gamma, &norm.beta, 1e-12);
        let gap = largest_gap(&read(&mut gpu, &xb, rows * width), &want);
        assert!(gap < 1e-4, "layer norm: {gap}");
        let inner = 3072;
        let y: Vec<f32> = values(rows * 2 * inner, 0.8)
            .iter()
            .map(|v| v * 12.0)
            .collect();
        let yb = upload(&mut gpu, &bytes_of(&y));
        // A cap below rows * inner / 256 workgroups, as the 65535 Vulkan
        // guarantees is below a full batch's.
        gpu.cap_groups(16);
        gpu.run(|c| swiglu(c, &sw, &yb, rows, inner)).unwrap();
        let mut want = y;
        cpu_swiglu(&mut want, inner);
        let gap = largest_gap(&read(&mut gpu, &yb, rows * 2 * inner), &want);
        assert!(gap < 1e-4, "swiglu: {gap}");
    }

    /// Attention from its definition in f64, over two sequences, one shorter
    /// than a tile of keys. The scores are small, so that keys past a
    /// sequence would weigh as much as those in it.
    #[test]
    #[ignore = "needs a Vulkan GPU"]
    fn gpu_attention_matches_definition() {
        let mut gpu = Gpu::open().expect("a GPU");
        let at = kernel(
            &gpu,
            include_bytes!(concat!(env!("OUT_DIR"), "/attention.spv")),
            3,
            2,
        )
        .unwrap();
        let (h, heads, d) = (768, 12, 64);
        let seqs = [0..3usize, 3..103];
        let t = 103;
        let qkv: Vec<f32> = values(t * 3 * h, 0.9).iter().map(|v| v * 0.2).collect();
        let mut meta: Vec<u32> = Vec::new();
        for seq in &seqs {
            let mut q = seq.start;
            while q < seq.end {
                let count = QUERY_BLOCK.min(seq.end - q);
                meta.extend([seq.start, seq.len(), q, count, q, 0, 0, 0].map(word));
                q += count;
            }
        }
        let tasks = meta.len() / TASK_WORDS;
        let qb = upload(&mut gpu, &bytes_of(&qkv));
        let words: Vec<u8> = meta.iter().flat_map(|w| w.to_le_bytes()).collect();
        let mb = upload(&mut gpu, &words);
        let ob = gpu.buffer(4 * t * h).unwrap();
        gpu.run(|c| {
            c.dispatch(
                &at,
                &[&qb, &mb, &ob],
                &push(&[word(h), 0]),
                [word(tasks), word(heads), 1],
            )
        })
        .unwrap();
        let got = read(&mut gpu, &ob, t * h);
        let mut want = vec![0.0f32; t * h];
        for seq in &seqs {
            for head in 0..heads {
                for i in seq.clone() {
                    let q = &qkv[i * 3 * h + head * d..][..d];
                    let scores: Vec<f64> = seq
                        .clone()
                        .map(|j| {
                            let k = &qkv[j * 3 * h + h + head * d..][..d];
                            q.iter()
                                .zip(k)
                                .map(|(a, b)| f64::from(*a) * f64::from(*b))
                                .sum()
                        })
                        .collect();
                    let max = scores.iter().copied().fold(f64::MIN, f64::max);
                    let e: Vec<f64> = scores.iter().map(|s| (s - max).exp()).collect();
                    let total: f64 = e.iter().sum();
                    for x in 0..d {
                        let v: f64 = seq
                            .clone()
                            .zip(&e)
                            .map(|(j, p)| p * f64::from(qkv[j * 3 * h + 2 * h + head * d + x]))
                            .sum();
                        want[i * h + head * d + x] = (v / total) as f32;
                    }
                }
            }
        }
        let gap = largest_gap(&got, &want);
        assert!(gap < 1e-4, "attention: {gap}");
    }

    /// The products' half floats move the hidden states by about 1e-3; a
    /// buffer or task in the wrong place moves them by their own size.
    #[test]
    #[ignore = "needs a Vulkan GPU"]
    fn gpu_model_matches_cpu() {
        let ids: Vec<u32> = (0..90).map(|i| (i * 7 % 50) as u32).collect();
        let seqs = [0..2usize, 2..20, 20..90];
        let mut cpu = crate::model::CpuModel::new(tiny(), 4).unwrap();
        let mut gpu = GpuModel::new(tiny(), Gpu::open().expect("a GPU")).unwrap();
        let mut want = vec![0.0; 3 * 128];
        let mut got = vec![0.0; 3 * 128];
        cpu.embed(&ids, &seqs, &mut want);
        gpu.embed(&ids, &seqs, &mut got).unwrap();
        let gap = largest_gap(&got, &want);
        assert!(gap < 1e-2, "{gap}: {:?} vs {:?}", &got[..4], &want[..4]);
    }
}
