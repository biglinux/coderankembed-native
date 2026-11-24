//! Matrix products: every linear layer, and the two products of attention.
//!
//! The right operand is a `k x n` [`Matrix`], packed into panels of 16
//! columns, `[n / 16][k][16]`, so the 16 weights one step of the micro-kernel
//! needs are one contiguous, aligned run of 64 bytes. The left operand is
//! never copied: the kernel reads each row where it lies, through a pointer
//! per row, as XNNPACK's indirect convolution does (see `NOTICE`).
//!
//! [`multiply`] covers a range of rows and a range of panels on the calling
//! thread, so a layer can be split between threads by rows, by columns or
//! both. The reduction runs in blocks of `KC` values: a panel block meets
//! every six-row group of the range, and the rows stay in L2 across panels.
//! The order of the sums depends only on `k`, never on the ranges or on the
//! thread.

use std::ops::Range;

use crate::pool::Shared;

mod kernel;

use kernel::{KernelCall, kernel};

/// Columns of a weight panel and of a micro-tile.
pub(crate) const NR: usize = 16;
/// Rows of a micro-tile.
pub(crate) const MR: usize = 6;
/// Length of one block of the reduction. Each block loads and stores every
/// output tile once, which costs more than a panel block outgrowing L1.
const KC: usize = 512;

/// One step of a panel: 16 weights, aligned so that a vector load of them
/// never straddles two cache lines (Sandy Bridge pays for every split load).
#[derive(Clone, Copy, Default)]
#[repr(C, align(64))]
pub(crate) struct Row(pub(crate) [f32; NR]);

/// Rows a packed `k x n` matrix takes: `[n / 16][k]`, the last panel's
/// columns past `n` unused.
pub(crate) fn panel_rows(k: usize, n: usize) -> usize {
    n.div_ceil(NR) * k
}

/// The packed right operand of a product, wherever its rows are kept.
#[derive(Clone, Copy)]
pub(crate) struct Panels<'a> {
    pub(crate) k: usize,
    pub(crate) n: usize,
    pub(crate) rows: &'a [Row],
}

impl Panels<'_> {
    pub(crate) fn count(&self) -> usize {
        self.n.div_ceil(NR)
    }
}

/// A layer's weights, packed once at load.
pub(crate) struct Matrix {
    k: usize,
    n: usize,
    rows: Vec<Row>,
}

impl Matrix {
    /// The transpose of `w`, `n` rows of `k` values: the weight of a PyTorch
    /// `Linear(k, n)`, so that the product computes `x W^T`.
    pub(crate) fn transposed(w: &[f32], n: usize, k: usize) -> Self {
        assert_eq!(w.len(), k * n, "weights are not {n} x {k}");
        let mut rows = vec![Row::default(); panel_rows(k, n)];
        for (j, column) in w.chunks_exact(k).enumerate() {
            for (x, &v) in column.iter().enumerate() {
                rows[(j / NR) * k + x].0[j % NR] = v;
            }
        }
        Self { k, n, rows }
    }

    pub(crate) fn panels(&self) -> Panels<'_> {
        Panels {
            k: self.k,
            n: self.n,
            rows: &self.rows,
        }
    }
}

/// Row pointers of one reduction block.
#[derive(Default)]
pub(crate) struct GemmScratch {
    ptrs: Vec<*const f32>,
}

// SAFETY: the pointers are only meaningful during the `multiply` call that
// writes them; a scratch carries no live borrow between calls.
unsafe impl Send for GemmScratch {}

/// Where [`multiply`] writes: row `r` of the product, column `j`, is element
/// `at + (r - rows.start) * stride + j` of `buffer`.
#[derive(Clone, Copy)]
pub(crate) struct Out<'a> {
    pub(crate) buffer: Shared<'a>,
    pub(crate) at: usize,
    pub(crate) stride: usize,
}

/// The columns of panels `panels` of `a[rows] * w`, row `r` of `a` being
/// `a[r * a_stride..][..w.k]`, written to `out`, or with `accumulate`, added
/// to what `out` holds.
///
/// # Safety
/// No other live reference may reach the elements of `out.buffer` this call
/// writes: the rows `rows` and the columns of `panels`.
#[expect(
    clippy::too_many_arguments,
    reason = "a product's operands and their layout"
)]
pub(crate) unsafe fn multiply(
    a: &[f32],
    a_stride: usize,
    rows: Range<usize>,
    w: Panels<'_>,
    panels: Range<usize>,
    out: Out<'_>,
    accumulate: bool,
    scratch: &mut GemmScratch,
) {
    let m = rows.len();
    if m == 0 || panels.is_empty() {
        return;
    }
    assert!(
        (rows.end - 1) * a_stride + w.k <= a.len(),
        "rows out of bounds"
    );
    assert!(
        panels.end <= w.count() && w.rows.len() >= panel_rows(w.k, w.n),
        "panels out of bounds"
    );
    let width = (panels.end * NR).min(w.n);
    assert!(
        out.stride >= width && out.at + (m - 1) * out.stride + width <= out.buffer.len(),
        "output too small"
    );
    let base = out.buffer.ptr(out.at);
    let groups = m.div_ceil(MR);
    let kernel = kernel();
    let mut tile = [0.0f32; MR * NR];
    let mut c0 = 0;
    while c0 < w.k {
        let c1 = (c0 + KC).min(w.k);
        let load = accumulate || c0 > 0;
        scratch.ptrs.clear();
        for g in 0..groups {
            for i in 0..MR {
                let row = rows.start + (g * MR + i).min(m - 1);
                scratch.ptrs.push(a[row * a_stride + c0..].as_ptr());
            }
        }
        for p in panels.clone() {
            let b = w.rows[p * w.k + c0..].as_ptr().cast::<f32>();
            let cols = (w.n - p * NR).min(NR);
            // A full panel writes its full row groups in place, in one call.
            let full = if cols == NR { m / MR } else { 0 };
            if full > 0 {
                let call = KernelCall {
                    depth: c1 - c0,
                    a: scratch.ptrs.as_ptr(),
                    b,
                    load,
                    groups: full,
                };
                // SAFETY: rows 0..full*6 and columns p*16..p*16+16 are inside
                // `out` (asserted above with cols == 16) and the caller's
                // alone; each pointer reaches `w.k - c0` floats of its row
                // (asserted above), and `b` holds `w.k - c0` rows of this
                // panel.
                unsafe { kernel(&call, base.add(p * NR), out.stride) };
            }
            // The rest one 6 x 16 group at a time, through a tile.
            for g in full..groups {
                let valid = (m - g * MR).min(MR);
                // SAFETY: rows below `m` and columns p*16..p*16+cols are inside
                // `out` (asserted above) and the caller's alone.
                let at = |i: usize| unsafe { base.add((g * MR + i) * out.stride + p * NR) };
                if load {
                    for i in 0..valid {
                        // SAFETY: see `at`.
                        unsafe {
                            std::ptr::copy_nonoverlapping(at(i), tile[i * NR..].as_mut_ptr(), cols)
                        };
                    }
                }
                let call = KernelCall {
                    depth: c1 - c0,
                    a: scratch.ptrs[g * MR..].as_ptr(),
                    b,
                    load,
                    groups: 1,
                };
                // SAFETY: as above, with the tile as the group's 6 x 16 outputs.
                unsafe { kernel(&call, tile.as_mut_ptr(), NR) };
                for i in 0..valid {
                    // SAFETY: see `at`.
                    unsafe { std::ptr::copy_nonoverlapping(tile[i * NR..].as_ptr(), at(i), cols) };
                }
            }
        }
        c0 = c1;
    }
}

/// [`multiply`] over all of `w` into `out`, its rows `stride` apart.
pub(crate) fn multiply_into(
    a: &[f32],
    a_stride: usize,
    rows: Range<usize>,
    w: Panels<'_>,
    out: &mut [f32],
    stride: usize,
    scratch: &mut GemmScratch,
) {
    let out = Out {
        buffer: Shared::new(out),
        at: 0,
        stride,
    };
    // SAFETY: `out` borrows its buffer mutably for this call.
    unsafe { multiply(a, a_stride, rows, w, 0..w.count(), out, false, scratch) };
}

#[cfg(test)]
mod tests {
    use super::kernel::*;
    use super::*;

    fn values(len: usize, seed: f32) -> Vec<f32> {
        (0..len)
            .map(|i| ((i as f32 * 0.37 + seed).sin()) * 0.5)
            .collect()
    }

    /// `a w^T` in f64, `w` being `n` rows of `k`: the oracle for every path.
    fn reference(a: &[f32], m: usize, k: usize, n: usize, w: &[f32]) -> Vec<f64> {
        let mut out = vec![0.0; m * n];
        for r in 0..m {
            for j in 0..n {
                out[r * n + j] = (0..k)
                    .map(|x| f64::from(a[r * k + x]) * f64::from(w[j * k + x]))
                    .sum();
            }
        }
        out
    }

    fn close(got: &[f32], want: &[f64], stride: usize, n: usize) {
        for (r, row) in want.chunks(n).enumerate() {
            for (j, &v) in row.iter().enumerate() {
                let g = f64::from(got[r * stride + j]);
                assert!(
                    (g - v).abs() <= 1e-4 * (1.0 + v.abs()),
                    "row {r} col {j}: {g} vs {v}"
                );
            }
        }
    }

    /// Each kernel this CPU can run, against the portable one, on two tiles
    /// with loading.
    #[test]
    fn kernels_agree_on_a_tile() {
        let depth = 37;
        let a = values(2 * MR * depth, 0.2);
        let b = values(depth * NR, 0.6);
        let ptrs: Vec<*const f32> = (0..2 * MR).map(|i| a[i * depth..].as_ptr()).collect();
        let call = KernelCall {
            depth,
            a: ptrs.as_ptr(),
            b: b.as_ptr(),
            load: true,
            groups: 2,
        };
        let start = values(2 * MR * NR, 2.4);
        let run = |kernel: Kernel| {
            let mut c = start.clone();
            // SAFETY: the tiles, pointers and weights above have the sizes
            // `KernelCall` describes.
            unsafe { kernel(&call, c.as_mut_ptr(), NR) };
            c
        };
        let want = run(kernel_portable);
        let mut kernels: Vec<Kernel> = Vec::new();
        #[cfg(target_arch = "x86_64")]
        {
            kernels.push(kernel_sse);
            if crate::simd_tier() >= crate::TIER_AVX {
                kernels.push(kernel_avx);
            }
            if crate::simd_tier() >= crate::TIER_AVX2 {
                kernels.push(kernel_avx2);
            }
        }
        for kernel in kernels {
            for (g, w) in run(kernel).iter().zip(&want) {
                assert!((g - w).abs() <= 1e-5 * (1.0 + w.abs()), "{g} vs {w}");
            }
        }
    }

    #[test]
    fn products_match_reference() {
        // Sizes cover partial row groups, partial panels and several k blocks.
        for &(m, k, n) in &[(1, 3, 1), (7, 40, 17), (13, 1100, 48), (6, 512, 16)] {
            let a = values(m * k, 0.3);
            let w = values(k * n, 1.1);
            let matrix = Matrix::transposed(&w, n, k);
            let stride = n + 3;
            let mut out = vec![f32::NAN; (m - 1) * stride + n];
            multiply_into(
                &a,
                k,
                0..m,
                matrix.panels(),
                &mut out,
                stride,
                &mut GemmScratch::default(),
            );
            close(&out, &reference(&a, m, k, n, &w), stride, n);
        }
    }

    #[test]
    fn ranges_accumulate_onto_output() {
        let (m, k, n) = (20, 70, 40);
        let a = values(m * k, 0.7);
        let w = values(k * n, -0.4);
        let matrix = Matrix::transposed(&w, n, k);
        let mut out = vec![1.0f32; m * n];
        let mut scratch = GemmScratch::default();
        let buffer = Shared::new(&mut out);
        for rows in [0..5, 5..17, 17..20] {
            for panels in [0..1, 1..3] {
                let out = Out {
                    buffer,
                    at: rows.start * n,
                    stride: n,
                };
                // SAFETY: each call writes its own rows and panels.
                unsafe {
                    multiply(
                        &a,
                        k,
                        rows.clone(),
                        matrix.panels(),
                        panels,
                        out,
                        true,
                        &mut scratch,
                    )
                };
            }
        }
        let want: Vec<f64> = reference(&a, m, k, n, &w).iter().map(|v| v + 1.0).collect();
        close(&out, &want, n, n);
    }
}
