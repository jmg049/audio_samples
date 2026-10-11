//! Element-wise logarithms of `f32` slices that vectorise.
//!
//! The scalar `f32::ln` / `f32::log10` call the C library once per element
//! and cannot vectorise. These kernels evaluate the logarithm of each `f32`
//! value in `f64` with straight-line code (exponent and mantissa split, then
//! an odd series in `s = (m - 1) / (m + 1)` with a relative error below
//! 1e-15) and round the result to `f32` once. The result is the correctly
//! rounded `f32` logarithm except when the exact value lies within about
//! 1e-15 (relative) of a rounding boundary, so it differs from the C
//! library's by at most one unit in the last place, and only rarely.
//!
//! Zero, negative, subnormal, infinite and NaN inputs are left to the scalar
//! functions, so the special values are those of the standard library.
//!
//! On x86-64 with AVX2 the same code is compiled with AVX2 enabled (chosen at
//! run time). FMA is not enabled, so the results are identical on every
//! machine.
//!
//! `f64` slices use the standard library's scalar functions unchanged,
//! except [`log2_f64`], which uses the msun `log2` kernel of
//! [`crate::utils::fast_math`] (within 1 ulp, not always bit-identical to the
//! C library).

use std::any::TypeId;

use crate::AudioFloat;

const LN_2: f64 = std::f64::consts::LN_2;
const LOG10_E: f64 = std::f64::consts::LOG10_E;
const LOG10_2: f64 = std::f64::consts::LOG10_2;

/// `ln(x)` of a positive normal finite `f32` (as bits), in `f64`.
#[inline(always)]
fn ln_f64_of_f32_bits(bits: u32) -> (f64, f64) {
    // x = 2^e * m with m in [1, 2); fold m into [sqrt(1/2), sqrt(2)).
    let mut e = ((bits >> 23) & 0xff) as i32 - 127;
    let mant = bits & 0x007f_ffff;
    // m > sqrt(2) <=> mantissa bits > 0x3504f3 (sqrt(2) = 1.0110101000001001111001101...)
    let big = mant > 0x0035_04f3;
    let m_bits = mant | if big { 0x3f00_0000 } else { 0x3f80_0000 };
    e += i32::from(big);
    let m = f64::from(f32::from_bits(m_bits));
    // ln(m) = 2 atanh(s), s = (m - 1) / (m + 1), |s| <= 0.1716.
    let s = (m - 1.0) / (m + 1.0);
    let z = s * s;
    let p = 1.0 / 3.0
        + z * (1.0 / 5.0
            + z * (1.0 / 7.0
                + z * (1.0 / 9.0
                    + z * (1.0 / 11.0 + z * (1.0 / 13.0 + z * (1.0 / 15.0 + z * (1.0 / 17.0)))))));
    let ln_m = 2.0 * s + 2.0 * s * z * p;
    (f64::from(e), ln_m)
}

/// Whether `bits` is a positive normal finite `f32`.
#[inline(always)]
fn positive_normal(bits: u32) -> bool {
    // exponent field in 1..=254 and sign bit clear
    bits.wrapping_sub(0x0080_0000) < 0x7f00_0000
}

/// Applies `fast` (valid for positive normal values) to the chunks of `v`
/// that hold only positive normal values and `slow` element-wise elsewhere.
#[inline(always)]
fn by_chunks(v: &mut [f32], fast: impl Fn(u32) -> f32, slow: impl Fn(f32) -> f32) {
    for chunk in v.chunks_mut(64) {
        if chunk
            .iter()
            .fold(true, |a, x| a & positive_normal(x.to_bits()))
        {
            for x in chunk.iter_mut() {
                *x = fast(x.to_bits());
            }
        } else {
            for x in chunk.iter_mut() {
                *x = if positive_normal(x.to_bits()) {
                    fast(x.to_bits())
                } else {
                    slow(*x)
                };
            }
        }
    }
}

#[inline(always)]
fn ln_f32_generic(v: &mut [f32]) {
    by_chunks(
        v,
        |bits| {
            let (e, ln_m) = ln_f64_of_f32_bits(bits);
            (e * LN_2 + ln_m) as f32
        },
        f32::ln,
    );
}

#[inline(always)]
fn log10_f32_generic(v: &mut [f32]) {
    by_chunks(
        v,
        |bits| {
            let (e, ln_m) = ln_f64_of_f32_bits(bits);
            (e * LOG10_2 + ln_m * LOG10_E) as f32
        },
        f32::log10,
    );
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
fn ln_f32_avx2(v: &mut [f32]) {
    ln_f32_generic(v);
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
fn log10_f32_avx2(v: &mut [f32]) {
    log10_f32_generic(v);
}

#[cfg(target_arch = "x86_64")]
fn has_avx2() -> bool {
    std::arch::is_x86_feature_detected!("avx2")
}

/// `x = ln(x)` for every element of an `f32` slice.
pub(crate) fn ln_f32(v: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    if has_avx2() {
        // SAFETY: AVX2 is available (checked at run time).
        unsafe { ln_f32_avx2(v) };
        return;
    }
    ln_f32_generic(v);
}

/// `x = log10(x)` for every element of an `f32` slice.
pub(crate) fn log10_f32(v: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    if has_avx2() {
        // SAFETY: AVX2 is available (checked at run time).
        unsafe { log10_f32_avx2(v) };
        return;
    }
    log10_f32_generic(v);
}

#[inline(always)]
fn log2_f64_generic(v: &mut [f64]) {
    use crate::utils::fast_math::{log2_kernel, positive_normal_f64};
    for chunk in v.chunks_mut(64) {
        if chunk.iter().fold(true, |a, &x| a & positive_normal_f64(x)) {
            for x in chunk.iter_mut() {
                *x = log2_kernel(*x);
            }
        } else {
            for x in chunk.iter_mut() {
                *x = if positive_normal_f64(*x) {
                    log2_kernel(*x)
                } else {
                    x.log2()
                };
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
fn log2_f64_avx2(v: &mut [f64]) {
    log2_f64_generic(v);
}

/// `x = log2(x)` for every element of an `f64` slice, within 1 ulp.
///
/// Positive normal finite values use the vectorising msun kernel; zero,
/// negative, subnormal, infinite and NaN values the standard library (so
/// the special values are its own).
pub(crate) fn log2_f64(v: &mut [f64]) {
    #[cfg(target_arch = "x86_64")]
    if has_avx2() {
        // SAFETY: AVX2 is available (checked at run time).
        unsafe { log2_f64_avx2(v) };
        return;
    }
    log2_f64_generic(v);
}

/// Matrix product `out = w · x` with `x` given as rows: `out` holds
/// `w.len() / rows.len()` rows of `width` values (row stride `out_stride`,
/// starting at `out[0]`), and `out[q][t] = sum_m (w[q][m] * rows[m][t])`,
/// summed over `m` in ascending order starting from `-0.0`, i.e. exactly
/// `(0..n).map(|m| w[q][m] * rows[m][t]).sum()`.
///
/// Register-blocked over 4 output rows and 16 columns, vectorised across
/// columns; the order of operations of every output value is the same in
/// the blocked and the remainder paths.
pub(crate) fn matmul_rows<F: AudioFloat>(
    out: &mut [F],
    out_stride: usize,
    width: usize,
    w: &[F],
    rows: &[&[F]],
) {
    #[cfg(target_arch = "x86_64")]
    if has_avx2() {
        // SAFETY: AVX2 is available (checked at run time).
        unsafe { matmul_rows_avx2(out, out_stride, width, w, rows) };
        return;
    }
    matmul_rows_generic(out, out_stride, width, w, rows);
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
fn matmul_rows_avx2<F: AudioFloat>(
    out: &mut [F],
    out_stride: usize,
    width: usize,
    w: &[F],
    rows: &[&[F]],
) {
    matmul_rows_generic(out, out_stride, width, w, rows);
}

#[inline(always)]
fn matmul_rows_generic<F: AudioFloat>(
    out: &mut [F],
    out_stride: usize,
    width: usize,
    w: &[F],
    rows: &[&[F]],
) {
    // 4 x 2 vectors of accumulators per block with 256-bit vectors.
    if size_of::<F>() == 4 {
        matmul_rows_tb::<F, 16>(out, out_stride, width, w, rows);
    } else {
        matmul_rows_tb::<F, 8>(out, out_stride, width, w, rows);
    }
}

#[inline(always)]
fn matmul_rows_tb<F: AudioFloat, const TB: usize>(
    out: &mut [F],
    out_stride: usize,
    width: usize,
    w: &[F],
    rows: &[&[F]],
) {
    const QB: usize = 4;
    let n = rows.len();
    let n_q = w.len() / n.max(1);
    // The last partial block of columns, zero-padded to TB columns.
    let full = width / TB * TB;
    let rem = width - full;
    let pad: Vec<F> = if rem > 0 {
        let mut p = vec![F::zero(); n * TB];
        for (m, r) in rows.iter().enumerate() {
            p[m * TB..m * TB + rem].copy_from_slice(&r[full..width]);
        }
        p
    } else {
        Vec::new()
    };
    let pad_rows: Vec<&[F]> = pad.chunks(TB).collect();
    let mut wb: Vec<[F; QB]> = vec![[F::zero(); QB]; n];
    let mut q = 0;
    while q < n_q {
        let qb = QB.min(n_q - q);
        // The block's weights by column of `w`; rows past the end get weight
        // 0 and are not stored.
        for (m, wm) in wb.iter_mut().enumerate() {
            for (qi, v) in wm.iter_mut().enumerate() {
                *v = if qi < qb {
                    w[(q + qi) * n + m]
                } else {
                    F::zero()
                };
            }
        }
        let mut t = 0;
        while t < full {
            let a = block::<F, QB, TB>(rows, t, &wb);
            store_block(out, out_stride, q, qb, t, TB, &a);
            t += TB;
        }
        if rem > 0 {
            let a = block::<F, QB, TB>(&pad_rows, 0, &wb);
            store_block(out, out_stride, q, qb, full, rem, &a);
        }
        q += qb;
    }
}

/// `4 x TB` block of `w · x` from columns `off..off + TB` of `src`.
#[inline(always)]
fn block<F: AudioFloat, const QB: usize, const TB: usize>(
    src: &[&[F]],
    off: usize,
    wb: &[[F; QB]],
) -> [[F; TB]; QB] {
    const { assert!(QB == 4) };
    let z = -F::zero();
    let (mut a0, mut a1, mut a2, mut a3) = ([z; TB], [z; TB], [z; TB], [z; TB]);
    for (r, wm) in src.iter().zip(wb) {
        let x = &r[off..off + TB];
        let (w0, w1, w2, w3) = (wm[0], wm[1], wm[2], wm[3]);
        for (j, &xj) in x.iter().enumerate() {
            a0[j] += w0 * xj;
            a1[j] += w1 * xj;
            a2[j] += w2 * xj;
            a3[j] += w3 * xj;
        }
    }
    std::array::from_fn(|qi| match qi {
        0 => a0,
        1 => a1,
        2 => a2,
        _ => a3,
    })
}

#[inline(always)]
fn store_block<F: AudioFloat, const QB: usize, const TB: usize>(
    out: &mut [F],
    out_stride: usize,
    q: usize,
    qb: usize,
    t: usize,
    len: usize,
    a: &[[F; TB]; QB],
) {
    for (qi, row) in a.iter().enumerate().take(qb) {
        let o = (q + qi) * out_stride + t;
        out[o..o + len].copy_from_slice(&row[..len]);
    }
}

/// `x = log10(max(amin, x))` element-wise, returning the maximum of the
/// results (NaN-ignoring, `-inf` for an empty slice): the vectorised
/// logarithm for `f32`, the standard library's `log10` for `f64`.
pub(crate) fn log10_max_in_place<F: AudioFloat>(v: &mut [F], amin: F) -> F {
    match as_f32(v) {
        Some(v32) => {
            let amin32 = amin.to_f32().expect("f32 input has an f32 amin");
            for x in v32.iter_mut() {
                *x = amin32.max(*x);
            }
            log10_f32(v32);
            F::of(f64::from(max_f32(v32)))
        }
        None => {
            let mut m = F::neg_infinity();
            for x in v.iter_mut() {
                *x = amin.max(*x).log10();
                m = m.max(*x);
            }
            m
        }
    }
}

/// Maximum of an `f32` slice (NaN-ignoring, `-inf` when empty), eight
/// running maxima so that it vectorises.
fn max_f32(v: &[f32]) -> f32 {
    let mut acc = [f32::NEG_INFINITY; 8];
    let mut it = v.chunks_exact(8);
    for c in &mut it {
        for (a, &x) in acc.iter_mut().zip(c) {
            *a = a.max(x);
        }
    }
    let m = it
        .remainder()
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, f32::max);
    acc.iter().copied().fold(m, f32::max)
}

/// The slice as `&mut [f32]` when `F` is `f32`.
fn as_f32<F: AudioFloat>(v: &mut [F]) -> Option<&mut [f32]> {
    if TypeId::of::<F>() == TypeId::of::<f32>() {
        // SAFETY: F is f32 (same type, so same size, alignment and validity).
        Some(unsafe { std::slice::from_raw_parts_mut(v.as_mut_ptr().cast::<f32>(), v.len()) })
    } else {
        None
    }
}

/// `x = ln(x)` element-wise: the vectorised kernel for `f32`, the standard
/// library's `ln` for `f64`.
pub(crate) fn ln_in_place<F: AudioFloat>(v: &mut [F]) {
    match as_f32(v) {
        Some(v) => ln_f32(v),
        None => {
            for x in v {
                *x = x.ln();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ulps(a: f32, b: f32) -> u32 {
        if a == b || (a.is_nan() && b.is_nan()) {
            return 0;
        }
        (a.to_bits() as i64 - b.to_bits() as i64).unsigned_abs() as u32
    }

    #[test]
    fn ln_and_log10_within_one_ulp_of_correct_rounding() {
        // Every 97th f32 bit pattern of the positive normal range plus the
        // specials.
        let mut xs: Vec<f32> = (0x0080_0000u32..0x7f80_0000)
            .step_by(97)
            .map(f32::from_bits)
            .collect();
        xs.extend([
            0.0,
            -0.0,
            -1.0,
            1.0,
            2.0,
            10.0,
            0.5,
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
            1e-40,
            f32::MIN_POSITIVE,
            f32::MAX,
            std::f32::consts::SQRT_2,
        ]);
        let mut ln = xs.clone();
        ln_f32(&mut ln);
        let mut lg = xs.clone();
        log10_f32(&mut lg);
        let (mut max_ln, mut max_lg) = (0, 0);
        for ((&x, &a), &b) in xs.iter().zip(&ln).zip(&lg) {
            // f64 reference, correctly rounded to f32 in all but tie-adjacent cases
            let r_ln = f64::from(x).ln() as f32;
            let r_lg = f64::from(x).log10() as f32;
            max_ln = max_ln.max(ulps(a, r_ln));
            max_lg = max_lg.max(ulps(b, r_lg));
            if !(x > 0.0 && x.is_finite()) || x < f32::MIN_POSITIVE {
                assert!(ulps(a, x.ln()) == 0, "ln special {x}: {a}");
                assert!(ulps(b, x.log10()) == 0, "log10 special {x}: {b}");
            }
        }
        assert!(
            max_ln <= 1 && max_lg <= 1,
            "ln {max_ln} ulp, log10 {max_lg} ulp"
        );
        assert_eq!(ln[xs.len() - 11], 0.0); // ln(1)
        assert_eq!(lg[xs.len() - 9], 1.0); // log10(10)
    }

    fn check_matmul<F: AudioFloat + Bits>() {
        for &(n_q, n, width) in &[
            (1, 1, 1),
            (3, 5, 7),
            (4, 128, 16),
            (20, 128, 431),
            (7, 3, 33),
        ] {
            let w: Vec<F> = (0..n_q * n)
                .map(|i| F::of(((i * 37) % 11) as f64 * 0.37 - 1.3))
                .collect();
            let x: Vec<Vec<F>> = (0..n)
                .map(|m| {
                    (0..width)
                        .map(|t| F::of(((m * 13 + t * 7) % 17) as f64 * 0.11 - 0.9))
                        .collect()
                })
                .collect();
            let rows: Vec<&[F]> = x.iter().map(Vec::as_slice).collect();
            let stride = width + 3;
            let mut out = vec![F::nan(); n_q * stride];
            matmul_rows(&mut out, stride, width, &w, &rows);
            for q in 0..n_q {
                for t in 0..width {
                    let want: F = (0..n).map(|m| w[q * n + m] * x[m][t]).sum();
                    assert_eq!(out[q * stride + t].to_bits_u64(), want.to_bits_u64());
                }
            }
        }
    }

    trait Bits {
        fn to_bits_u64(self) -> u64;
    }
    impl Bits for f32 {
        fn to_bits_u64(self) -> u64 {
            u64::from(self.to_bits())
        }
    }
    impl Bits for f64 {
        fn to_bits_u64(self) -> u64 {
            self.to_bits()
        }
    }

    #[test]
    fn matmul_rows_equals_the_ordered_sum_bitwise() {
        check_matmul_t::<f32>();
        check_matmul_t::<f64>();
    }

    fn check_matmul_t<F: AudioFloat + Bits>() {
        check_matmul::<F>();
    }
}
