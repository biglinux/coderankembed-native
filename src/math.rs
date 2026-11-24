//! The elementwise functions and row normalizations of the encoder.
//!
//! Each body is written once as plain arithmetic and compiled three times:
//! inside an AVX2 and an AVX function, where LLVM vectorizes it 8 wide, and
//! as is. No copy fuses multiplies and adds the source keeps apart, so all
//! three give the same bits. The exponential follows XNNPACK's f32 "p5"
//! kernels (see `NOTICE`): 2^n times a degree-5 polynomial, with `n` rounded
//! by a magic bias.

const EXP_MAGIC_BIAS: f32 = f32::from_bits(0x4b40_007f);
const LOG2E: f32 = f32::from_bits(0x3fb8_aa3b);
const MINUS_LN2_HI: f32 = f32::from_bits(0xbf31_7200);
const MINUS_LN2_LO: f32 = f32::from_bits(0xb5bf_be8e);
const EXP_C5: f32 = f32::from_bits(0x3c07_cfce);
const EXP_C4: f32 = f32::from_bits(0x3d2b_9d0d);
const EXP_C3: f32 = f32::from_bits(0x3e2a_ad40);
const EXP_C2: f32 = f32::from_bits(0x3eff_fee3);
const EXP_C1: f32 = f32::from_bits(0x3f7f_fffb);
/// Below this, exp(z) is denormal and taken as 0.
const EXP_CUTOFF: f32 = f32::from_bits(0xc2ae_ac4f);

/// `exp(z)` for `z <= 0`.
#[inline(always)]
fn exp_neg(z: f32) -> f32 {
    let n = z * LOG2E + EXP_MAGIC_BIAS;
    let s = f32::from_bits(n.to_bits() << 23);
    let n = n - EXP_MAGIC_BIAS;
    let t = n * MINUS_LN2_LO + (n * MINUS_LN2_HI + z);
    let p = (((EXP_C5 * t + EXP_C4) * t + EXP_C3) * t + EXP_C2) * t + EXP_C1;
    let e = (t * s) * p + s;
    if z < EXP_CUTOFF { 0.0 } else { e }
}

/// `1 / (1 + exp(-x))`, from `exp(-|x|)` so neither tail cancels.
#[inline(always)]
fn sigmoid_one(x: f32) -> f32 {
    let e = exp_neg(-x.abs());
    let s = e / (e + 1.0);
    if x > 0.0 { 1.0 - s } else { s }
}

/// Sum in eight interleaved lanes, which LLVM may keep in one vector.
#[inline(always)]
fn sum(values: &[f32]) -> f32 {
    let mut lanes = [0.0f32; 8];
    let (chunks, rest) = values.as_chunks::<8>();
    for chunk in chunks {
        for (l, v) in lanes.iter_mut().zip(chunk) {
            *l += v;
        }
    }
    let mut total = ((lanes[0] + lanes[4]) + (lanes[1] + lanes[5]))
        + ((lanes[2] + lanes[6]) + (lanes[3] + lanes[7]));
    for v in rest {
        total += v;
    }
    total
}

#[inline(always)]
fn layer_norm_body(x: &mut [f32], gamma: &[f32], beta: &[f32], eps: f32) {
    let n = x.len() as f32;
    let mean = sum(x) / n;
    let mut squares = [0.0f32; 8];
    let (chunks, rest) = x.as_chunks::<8>();
    for chunk in chunks {
        for (s, v) in squares.iter_mut().zip(chunk) {
            let d = v - mean;
            *s += d * d;
        }
    }
    let mut var = sum(&squares);
    for v in rest {
        let d = v - mean;
        var += d * d;
    }
    let inv = 1.0 / (var / n + eps).sqrt();
    for ((v, g), b) in x.iter_mut().zip(gamma).zip(beta) {
        *v = (*v - mean) * inv * g + b;
    }
}

#[inline(always)]
fn softmax_body(x: &mut [f32]) {
    let max = x.iter().fold(f32::NEG_INFINITY, |m, &v| m.max(v));
    for v in x.iter_mut() {
        *v = exp_neg(*v - max);
    }
    let inv = 1.0 / sum(x);
    for v in x.iter_mut() {
        *v *= inv;
    }
}

dispatch! {
    /// Softmax of each `width`-long row of `x`.
    pub(crate) softmax_rows(x: &mut [f32], width: usize) {
        for row in x.chunks_exact_mut(width) {
            softmax_body(row);
        }
    }
}

dispatch! {
    /// The gated unit of a SwiGLU feed-forward over rows of `2 * width`
    /// values, `up` then `gate`: `up * silu(gate)` into the first `width` of
    /// each row.
    pub(crate) swiglu(x: &mut [f32], width: usize) {
        for row in x.chunks_exact_mut(2 * width) {
            let (up, gate) = row.split_at_mut(width);
            for (u, &g) in up.iter_mut().zip(gate.iter()) {
                *u *= g * sigmoid_one(g);
            }
        }
    }
}

dispatch! {
    /// Layer normalization of each `gamma.len()`-long row of `x`.
    pub(crate) layer_norm(x: &mut [f32], gamma: &[f32], beta: &[f32], eps: f32) {
        assert!(
            !gamma.is_empty()
                && x.len().is_multiple_of(gamma.len())
                && beta.len() == gamma.len(),
            "rows of {} values with {} betas cannot hold {}",
            gamma.len(),
            beta.len(),
            x.len()
        );
        for row in x.chunks_exact_mut(gamma.len()) {
            layer_norm_body(row, gamma, beta, eps);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn swiglu_matches_definition() {
        let width = 4001;
        let mut x: Vec<f32> = (0..2 * width)
            .map(|i| {
                if i < width {
                    1.5
                } else {
                    (i - width) as f32 / 20.0 - 100.0
                }
            })
            .collect();
        let want: Vec<f64> = (0..width)
            .map(|i| {
                let g = f64::from(x[width + i]);
                1.5 * g / (1.0 + (-g).exp())
            })
            .collect();
        swiglu(&mut x, width);
        for (g, w) in x.iter().zip(&want) {
            assert!(
                (f64::from(*g) - w).abs() <= 4.0 * f64::from(f32::EPSILON) * (1.0 + w.abs()),
                "{g} vs {w}"
            );
        }
    }

    #[test]
    fn softmax_matches_definition() {
        let mut x: Vec<f32> = (0..98)
            .map(|i| (i as f32 * 0.7).sin() * 30.0 - 100.0 * f32::from(u8::from(i % 13 == 0)))
            .collect();
        let want: Vec<f64> = x
            .chunks(49)
            .flat_map(|row| {
                let max = row.iter().fold(f32::MIN, |m, &v| m.max(v));
                let e: Vec<f64> = row.iter().map(|&v| f64::from(v - max).exp()).collect();
                let total: f64 = e.iter().sum();
                e.into_iter().map(move |v| v / total)
            })
            .collect();
        softmax_rows(&mut x, 49);
        for (g, w) in x.iter().zip(&want) {
            assert!((f64::from(*g) - w).abs() < 1e-6, "{g} vs {w}");
        }
    }

    #[test]
    fn layer_norm_matches_definition() {
        let c = 37;
        let mut x: Vec<f32> = (0..3 * c)
            .map(|i| (i as f32 * 1.3).cos() * 4.0 + 2.0)
            .collect();
        let gamma: Vec<f32> = (0..c).map(|i| 1.0 + i as f32 / 50.0).collect();
        let beta: Vec<f32> = (0..c).map(|i| i as f32 / 10.0).collect();
        let want: Vec<f64> = x
            .chunks(c)
            .flat_map(|row| {
                let mean = row.iter().map(|&v| f64::from(v)).sum::<f64>() / c as f64;
                let var = row
                    .iter()
                    .map(|&v| (f64::from(v) - mean).powi(2))
                    .sum::<f64>()
                    / c as f64;
                let row = row.to_vec();
                (0..c).map(move |j| (f64::from(row[j]) - mean) / (var + 1e-12).sqrt())
            })
            .collect();
        layer_norm(&mut x, &gamma, &beta, 1e-12);
        for (i, (g, w)) in x.iter().zip(&want).enumerate() {
            let w = w * f64::from(gamma[i % c]) + f64::from(beta[i % c]);
            assert!((f64::from(*g) - w).abs() < 1e-5, "{g} vs {w}");
        }
    }
}
