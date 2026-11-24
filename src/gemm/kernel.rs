//! The micro-kernels of [`multiply`](super::multiply): one 6 x 16 tile of
//! the product per row group, in plain Rust, SSE2, AVX and AVX2.

use super::{MR, NR};

/// Micro-tiles: for each row group, `c[6 x 16] = (load ? c : 0) + sum over
/// the reduction block of a * b`.
pub(super) struct KernelCall {
    /// Length of the reduction block.
    pub(super) depth: usize,
    /// Six row pointers per group, each to `depth` readable floats.
    pub(super) a: *const *const f32,
    /// `depth` rows of 16 weights.
    pub(super) b: *const f32,
    pub(super) load: bool,
    /// Consecutive row groups, each a tile: the pointers of group g start at
    /// `a + g * 6`, its rows at `c + g * 6 * ldc`. One call for many saves the
    /// call and setup per tile, which counts in short products.
    pub(super) groups: usize,
}

pub(super) type Kernel = unsafe fn(&KernelCall, *mut f32, usize);

/// The fastest kernel this CPU runs.
#[cfg(target_arch = "x86_64")]
pub(super) fn kernel() -> Kernel {
    match crate::simd_tier() {
        crate::TIER_AVX2 => kernel_avx2,
        crate::TIER_AVX => kernel_avx,
        _ => kernel_sse,
    }
}

#[cfg(not(target_arch = "x86_64"))]
pub(super) fn kernel() -> Kernel {
    kernel_portable
}

/// The kernel in plain Rust, for other architectures and as the oracle of the
/// vector kernels.
///
/// # Safety
/// `call` must describe readable memory as [`KernelCall`] says, and `c` must
/// be writable for 6 rows of 16 floats `ldc` apart per group.
#[cfg_attr(
    target_arch = "x86_64",
    allow(
        dead_code,
        reason = "x86-64 runs the SSE kernel instead; the tests compare against this one"
    )
)]
pub(super) unsafe fn kernel_portable(call: &KernelCall, c: *mut f32, ldc: usize) {
    // SAFETY: all reads and writes stay within the ranges the caller promises.
    unsafe {
        for group in 0..call.groups {
            let c = c.add(group * MR * ldc);
            let rows = std::slice::from_raw_parts(call.a.add(group * MR), MR);
            let mut acc = [[0.0f32; NR]; MR];
            if call.load {
                for (i, row) in acc.iter_mut().enumerate() {
                    row.copy_from_slice(std::slice::from_raw_parts(c.add(i * ldc), NR));
                }
            }
            for x in 0..call.depth {
                let w = std::slice::from_raw_parts(call.b.add(x * NR), NR);
                for (row, &p) in acc.iter_mut().zip(rows) {
                    let a = *p.add(x);
                    for (v, wv) in row.iter_mut().zip(w) {
                        *v += a * wv;
                    }
                }
            }
            for (i, row) in acc.iter().enumerate() {
                std::slice::from_raw_parts_mut(c.add(i * ldc), NR).copy_from_slice(row);
            }
        }
    }
}

/// [`kernel_portable`] in 8-lane registers: twelve accumulators, two weight
/// vectors and one broadcast per row and step; `$fma` is `fused` or `split`.
///
/// The rows are written out instead of looped over: Sandy Bridge runs the
/// schedule LLVM gives the loop slower.
#[cfg(target_arch = "x86_64")]
macro_rules! vector_kernel {
    ($(#[$doc:meta])* $name:ident, $features:literal, $fma:ident) => {
        $(#[$doc])*
        ///
        /// # Safety
        /// As [`kernel_portable`], and the CPU must have the features enabled
        /// here.
        #[target_feature(enable = $features)]
        pub(super) unsafe fn $name(call: &KernelCall, c: *mut f32, ldc: usize) {
            use std::arch::x86_64::*;
            // SAFETY: all reads and writes stay within the ranges the caller
            // promises.
            unsafe {
                for group in 0..call.groups {
                    let c = c.add(group * MR * ldc);
                    let p = call.a.add(group * MR);
                    let mut acc = [[_mm256_setzero_ps(); 2]; MR];
                    if call.load {
                        for (i, row) in acc.iter_mut().enumerate() {
                            row[0] = _mm256_loadu_ps(c.add(i * ldc));
                            row[1] = _mm256_loadu_ps(c.add(i * ldc + 8));
                        }
                    }
                    let (p0, p1, p2) = (*p, *p.add(1), *p.add(2));
                    let (p3, p4, p5) = (*p.add(3), *p.add(4), *p.add(5));
                    let mut b = call.b;
                    for x in 0..call.depth {
                        let w0 = _mm256_loadu_ps(b);
                        let w1 = _mm256_loadu_ps(b.add(8));
                        let a = _mm256_broadcast_ss(&*p0.add(x));
                        acc[0][0] = madd8!($fma, a, w0, acc[0][0]);
                        acc[0][1] = madd8!($fma, a, w1, acc[0][1]);
                        let a = _mm256_broadcast_ss(&*p1.add(x));
                        acc[1][0] = madd8!($fma, a, w0, acc[1][0]);
                        acc[1][1] = madd8!($fma, a, w1, acc[1][1]);
                        let a = _mm256_broadcast_ss(&*p2.add(x));
                        acc[2][0] = madd8!($fma, a, w0, acc[2][0]);
                        acc[2][1] = madd8!($fma, a, w1, acc[2][1]);
                        let a = _mm256_broadcast_ss(&*p3.add(x));
                        acc[3][0] = madd8!($fma, a, w0, acc[3][0]);
                        acc[3][1] = madd8!($fma, a, w1, acc[3][1]);
                        let a = _mm256_broadcast_ss(&*p4.add(x));
                        acc[4][0] = madd8!($fma, a, w0, acc[4][0]);
                        acc[4][1] = madd8!($fma, a, w1, acc[4][1]);
                        let a = _mm256_broadcast_ss(&*p5.add(x));
                        acc[5][0] = madd8!($fma, a, w0, acc[5][0]);
                        acc[5][1] = madd8!($fma, a, w1, acc[5][1]);
                        b = b.add(NR);
                    }
                    for (i, row) in acc.iter().enumerate() {
                        _mm256_storeu_ps(c.add(i * ldc), row[0]);
                        _mm256_storeu_ps(c.add(i * ldc + 8), row[1]);
                    }
                }
            }
        }
    };
}

/// [`kernel_portable`] in SSE registers, which every x86-64 CPU has: the
/// columns as two halves of 6 x 8, each with twelve 4-wide accumulators, two
/// weight vectors and one broadcast per row and step.
///
/// # Safety
/// As [`kernel_portable`].
#[cfg(target_arch = "x86_64")]
pub(super) unsafe fn kernel_sse(call: &KernelCall, c: *mut f32, ldc: usize) {
    use std::arch::x86_64::*;
    // SAFETY: all reads and writes stay within the ranges the caller promises;
    // SSE2 is part of x86-64.
    unsafe {
        for group in 0..call.groups {
            let c = c.add(group * MR * ldc);
            let p = call.a.add(group * MR);
            let rows = [*p, *p.add(1), *p.add(2), *p.add(3), *p.add(4), *p.add(5)];
            for half in [0, 8] {
                let mut acc = [[_mm_setzero_ps(); 2]; MR];
                if call.load {
                    for (i, row) in acc.iter_mut().enumerate() {
                        row[0] = _mm_loadu_ps(c.add(i * ldc + half));
                        row[1] = _mm_loadu_ps(c.add(i * ldc + half + 4));
                    }
                }
                let mut b = call.b.add(half);
                for x in 0..call.depth {
                    let w0 = _mm_loadu_ps(b);
                    let w1 = _mm_loadu_ps(b.add(4));
                    for (row, &ptr) in acc.iter_mut().zip(&rows) {
                        let a = _mm_set1_ps(*ptr.add(x));
                        row[0] = _mm_add_ps(_mm_mul_ps(a, w0), row[0]);
                        row[1] = _mm_add_ps(_mm_mul_ps(a, w1), row[1]);
                    }
                    b = b.add(NR);
                }
                for (i, row) in acc.iter().enumerate() {
                    _mm_storeu_ps(c.add(i * ldc + half), row[0]);
                    _mm_storeu_ps(c.add(i * ldc + half + 4), row[1]);
                }
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
vector_kernel!(
    /// The AVX2 kernel, one fused multiply-add per product.
    kernel_avx2,
    "avx2,fma",
    fused
);
#[cfg(target_arch = "x86_64")]
vector_kernel!(
    /// The AVX kernel: a multiply and an add per product, which Sandy and Ivy
    /// Bridge run on separate ports.
    kernel_avx,
    "avx",
    split
);
