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

/// Polynomial in `z²` for `atan(z) / z` on `(0, 1]`, relative error about 6e-11, a thousandth of
/// the unit in the last place of an `f32` result.
const ATAN_COEFFS: [f64; 12] = [
    0.999_999_999_941_231,
    -0.333_333_315_711_632_5,
    0.199_999_120_502_528_02,
    -0.142_839_807_478_299_39,
    0.110_932_412_013_734_95,
    -0.089_802_389_268_321_9,
    0.072_457_179_657_229_77,
    -0.054_279_102_391_517_856,
    0.034_209_602_643_339_526,
    -0.016_111_676_035_663_52,
    0.004_850_931_474_355_162,
    -0.000_684_791_995_686_280_5,
];

/// `atan2(y, x)` and `sqrt(x² + y²)` of one pair, with no branch and no call into libm. The result
/// is NaN for a pair this does not cover, for [`atan2_norm_f32`] to redo.
#[inline(always)]
fn atan2_norm_pair(y: f32, x: f32) -> (f32, f32) {
    let (x, y) = (f64::from(x), f64::from(y));
    let (ax, ay) = (x.abs(), y.abs());
    let steep = ay > ax;
    let (n, d) = if steep { (ax, ay) } else { (ay, ax) };
    let z = n / d;
    let w = z * z;
    let mut p = ATAN_COEFFS[11];
    for c in ATAN_COEFFS[..11].iter().rev() {
        p = p * w + *c;
    }
    let p = p * z;
    let r = if steep {
        std::f64::consts::FRAC_PI_2 - p
    } else {
        p
    };
    let r = if x.is_sign_negative() {
        std::f64::consts::PI - r
    } else {
        r
    };
    // The squares of `f32` values neither overflow nor underflow in `f64`.
    (r.copysign(y) as f32, (x * x + y * y).sqrt() as f32)
}

#[inline(always)]
fn atan2_norm_f32_generic(y: &[f32], x: &[f32], angle: &mut [f32], norm: &mut [f32]) {
    for (((a, m), &yi), &xi) in angle.iter_mut().zip(norm.iter_mut()).zip(y).zip(x) {
        (*a, *m) = atan2_norm_pair(yi, xi);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
fn atan2_norm_f32_avx2(y: &[f32], x: &[f32], angle: &mut [f32], norm: &mut [f32]) {
    atan2_norm_f32_generic(y, x, angle, norm);
}

/// `angle[i] = atan2(y[i], x[i])` and `norm[i] = hypot(x[i], y[i])`, vectorised.
///
/// The usual path uses only addition, multiplication, division and square root in `f64`, which
/// IEEE 754 defines exactly and Rust never contracts, and rounds once, so the result does not
/// depend on the system's libm or on the vector width. Pairs outside that path (zero over zero,
/// two infinities, NaN) go through `f64::atan2` and `f32::hypot`, whose answers there are exact.
pub(crate) fn atan2_norm_f32(y: &[f32], x: &[f32], angle: &mut [f32], norm: &mut [f32]) {
    assert!(y.len() == x.len() && y.len() == angle.len() && y.len() == norm.len());
    #[cfg(target_arch = "x86_64")]
    if has_avx2() {
        // SAFETY: AVX2 is available (checked at run time).
        unsafe { atan2_norm_f32_avx2(y, x, angle, norm) };
    } else {
        atan2_norm_f32_generic(y, x, angle, norm);
    }
    #[cfg(not(target_arch = "x86_64"))]
    atan2_norm_f32_generic(y, x, angle, norm);
    let redo = angle
        .iter()
        .zip(norm.iter())
        .fold(false, |any, (a, m)| any | a.is_nan() | m.is_nan());
    if redo {
        for (((a, m), &yi), &xi) in angle.iter_mut().zip(norm.iter_mut()).zip(y).zip(x) {
            if a.is_nan() {
                *a = f64::from(yi).atan2(f64::from(xi)) as f32;
            }
            if m.is_nan() {
                *m = xi.hypot(yi);
            }
        }
    }
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
/// results (NaN-ignoring, `-inf` for an empty slice), with the vectorised
/// logarithms ([`log10_f32`], [`log10_f64`]).
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
            for x in v.iter_mut() {
                *x = amin.max(*x);
            }
            log10_in_place(v);
            let mut acc = [F::neg_infinity(); 8];
            let (blocks, rem) = v.as_chunks::<8>();
            for c in blocks {
                for (a, &x) in acc.iter_mut().zip(c) {
                    *a = a.max(x);
                }
            }
            let m = rem.iter().copied().fold(F::neg_infinity(), F::max);
            acc.iter().copied().fold(m, F::max)
        }
    }
}

/// Maximum of an `f32` slice (NaN-ignoring, `-inf` when empty), eight
/// running maxima so that it vectorises.
fn max_f32(v: &[f32]) -> f32 {
    let mut acc = [f32::NEG_INFINITY; 8];
    let (blocks, rem) = v.as_chunks::<8>();
    for c in blocks {
        for (a, &x) in acc.iter_mut().zip(c) {
            *a = a.max(x);
        }
    }
    let m = rem.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    acc.iter().copied().fold(m, f32::max)
}

/// `10^y` of an `f32` value with `|y| <= 300`, in `f64` (relative error
/// below 1e-15): `2^k * e^(r ln 2)` with `t = y log2(10) = k + r`.
#[inline(always)]
fn exp10_f64_of_f32(y: f32) -> f64 {
    const LOG2_10: f64 = std::f64::consts::LOG2_10;
    const LN_2: f64 = std::f64::consts::LN_2;
    // Round to nearest by the 1.5 * 2^52 shift; its low bits hold k.
    const SHIFT: f64 = 6_755_399_441_055_744.0;
    let t = f64::from(y) * LOG2_10;
    let shifted = t + SHIFT;
    let k = shifted - SHIFT;
    let r = t - k;
    let u = r * LN_2;
    // e^u, |u| <= 0.35: Taylor series to u^13 / 13!.
    let mut p = 1.0 / 6_227_020_800.0;
    p = 1.0 / 479_001_600.0 + u * p;
    p = 1.0 / 39_916_800.0 + u * p;
    p = 1.0 / 3_628_800.0 + u * p;
    p = 1.0 / 362_880.0 + u * p;
    p = 1.0 / 40_320.0 + u * p;
    p = 1.0 / 5_040.0 + u * p;
    p = 1.0 / 720.0 + u * p;
    p = 1.0 / 120.0 + u * p;
    p = 1.0 / 24.0 + u * p;
    p = 1.0 / 6.0 + u * p;
    p = 0.5 + u * p;
    p = 1.0 + u * p;
    p = 1.0 + u * p;
    let k_bits = shifted.to_bits().wrapping_sub(SHIFT.to_bits());
    let scale = f64::from_bits(k_bits.wrapping_add(1023) << 52);
    p * scale
}

/// `x = 10^x` for every element of an `f32` slice: evaluated in `f64` and
/// rounded to `f32` once (the correctly rounded value except within about
/// 1e-15 of a rounding boundary); NaN and `|x| > 300` go to `f32::powf`.
pub(crate) fn exp10_f32(v: &mut [f32]) {
    #[inline(always)]
    fn generic(v: &mut [f32]) {
        for chunk in v.chunks_mut(64) {
            let fast = chunk.iter().fold(true, |a, &y| a & (y.abs() <= 300.0));
            if fast {
                for y in chunk.iter_mut() {
                    *y = exp10_f64_of_f32(*y) as f32;
                }
            } else {
                for y in chunk.iter_mut() {
                    *y = if y.abs() <= 300.0 {
                        exp10_f64_of_f32(*y) as f32
                    } else {
                        10f32.powf(*y)
                    };
                }
            }
        }
    }
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    fn avx2(v: &mut [f32]) {
        generic(v);
    }
    #[cfg(target_arch = "x86_64")]
    if has_avx2() {
        // SAFETY: AVX2 is available (checked at run time).
        unsafe { avx2(v) };
        return;
    }
    generic(v);
}

// -----------------------------------------------------------------------------
// f64 logarithm in double-double

/// `ln(1 + j / 64)` for `j = -19..=27` (index `j + 19`) as `hi + lo`, exact
/// to about 2^-106 (computed with 60-digit decimal arithmetic); padded to 64
/// entries so that a masked index needs no bounds check.
#[rustfmt::skip]
static LN_C_HI: [f64; 64] = [
    f64::from_bits(0xBFD68AC83E9C6A14),
    f64::from_bits(0xBFD522AE0738A3D8),
    f64::from_bits(0xBFD3C25277333184),
    f64::from_bits(0xBFD269621134DB92),
    f64::from_bits(0xBFD1178E8227E47C),
    f64::from_bits(0xBFCF991C6CB3B379),
    f64::from_bits(0xBFCD1037F2655E7B),
    f64::from_bits(0xBFCA93ED3C8AD9E3),
    f64::from_bits(0xBFC823C16551A3C2),
    f64::from_bits(0xBFC5BF406B543DB2),
    f64::from_bits(0xBFC365FCB0159016),
    f64::from_bits(0xBFC1178E8227E47C),
    f64::from_bits(0xBFBDA727638446A2),
    f64::from_bits(0xBFB9335E5D594989),
    f64::from_bits(0xBFB4D3115D207EAC),
    f64::from_bits(0xBFB08598B59E3A07),
    f64::from_bits(0xBFA894AA149FB343),
    f64::from_bits(0xBFA0415D89E74444),
    f64::from_bits(0xBF90205658935847),
    f64::from_bits(0x0000000000000000),
    f64::from_bits(0x3F8FC0A8B0FC03E4),
    f64::from_bits(0x3F9F829B0E783300),
    f64::from_bits(0x3FA77458F632DCFC),
    f64::from_bits(0x3FAF0A30C01162A6),
    f64::from_bits(0x3FB341D7961BD1D1),
    f64::from_bits(0x3FB6F0D28AE56B4C),
    f64::from_bits(0x3FBA926D3A4AD563),
    f64::from_bits(0x3FBE27076E2AF2E6),
    f64::from_bits(0x3FC0D77E7CD08E59),
    f64::from_bits(0x3FC29552F81FF523),
    f64::from_bits(0x3FC44D2B6CCB7D1E),
    f64::from_bits(0x3FC5FF3070A793D4),
    f64::from_bits(0x3FC7AB890210D909),
    f64::from_bits(0x3FC9525A9CF456B4),
    f64::from_bits(0x3FCAF3C94E80BFF3),
    f64::from_bits(0x3FCC8FF7C79A9A22),
    f64::from_bits(0x3FCE27076E2AF2E6),
    f64::from_bits(0x3FCFB9186D5E3E2B),
    f64::from_bits(0x3FD0A324E27390E3),
    f64::from_bits(0x3FD1675CABABA60E),
    f64::from_bits(0x3FD22941FBCF7966),
    f64::from_bits(0x3FD2E8E2BAE11D31),
    f64::from_bits(0x3FD3A64C556945EA),
    f64::from_bits(0x3FD4618BC21C5EC2),
    f64::from_bits(0x3FD51AAD872DF82D),
    f64::from_bits(0x3FD5D1BDBF5809CA),
    f64::from_bits(0x3FD686C81E9B14AF),
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
];

/// Low parts of [`LN_C_HI`].
#[rustfmt::skip]
static LN_C_LO: [f64; 64] = [
    f64::from_bits(0xBC5A64EADD740178),
    f64::from_bits(0x3C68F7E9B38A6979),
    f64::from_bits(0x3C72AD27E50A8EC6),
    f64::from_bits(0xBC7E0EFADD9DB02B),
    f64::from_bits(0x3C60E63A5F01C691),
    f64::from_bits(0xBC6F665066F980A2),
    f64::from_bits(0xBC660629242471A2),
    f64::from_bits(0xBC6BCAFA9DE97203),
    f64::from_bits(0x3C61232CE70BE781),
    f64::from_bits(0x3C21F5B44C0DF7E7),
    f64::from_bits(0xBC57D411A5B944AD),
    f64::from_bits(0x3C50E63A5F01C691),
    f64::from_bits(0xBC5401FA71733019),
    f64::from_bits(0x3C5478A85704CCB7),
    f64::from_bits(0xBC5769F42C7842CC),
    f64::from_bits(0x3C5DD7009902BF32),
    f64::from_bits(0xBC3A8BE97660A23D),
    f64::from_bits(0xBC4C05CF1D753622),
    f64::from_bits(0xBC327C8E8416E71F),
    f64::from_bits(0x0000000000000000),
    f64::from_bits(0xBC183092C59642A1),
    f64::from_bits(0x3C333E3F04F1EF23),
    f64::from_bits(0x3C418D3CA87B9296),
    f64::from_bits(0x3C485F325C5BBACD),
    f64::from_bits(0xBC5B599F227BECBB),
    f64::from_bits(0xBC5906D99184B992),
    f64::from_bits(0x3C5942F48AA70EA9),
    f64::from_bits(0xBC361578001E0162),
    f64::from_bits(0x3C69A5DC5E9030AC),
    f64::from_bits(0x3C6301771C407DBF),
    f64::from_bits(0x3C69F4F6543E1F88),
    f64::from_bits(0xBC5BC60EFAFC6F6E),
    f64::from_bits(0x3C4BE36B2D6A0608),
    f64::from_bits(0x3C6D904C1D4E2E26),
    f64::from_bits(0xBC5398CFF3641985),
    f64::from_bits(0xBC64F689F8434012),
    f64::from_bits(0xBC461578001E0162),
    f64::from_bits(0xBC6CAAAE64F21ACB),
    f64::from_bits(0x3C77DCFDE8061C03),
    f64::from_bits(0x3C2CE63EAB883717),
    f64::from_bits(0xBC776F5EB09628AF),
    f64::from_bits(0xBC78F4CDB95EBDF9),
    f64::from_bits(0xBC6C68651945F97C),
    f64::from_bits(0x3C7F42DECDECCF1D),
    f64::from_bits(0x3C43927AC19F55E3),
    f64::from_bits(0x3C74236383DC7FE1),
    f64::from_bits(0xBC6DDEA0F7F58E3D),
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
];

const LN2_HI: f64 = f64::from_bits(0x3FE6_2E42_FEE0_0000); // 32 significant bits
const LN2_LO: f64 = f64::from_bits(0x3DEA_39EF_3579_3C76);
const LOG10E_HI: f64 = f64::from_bits(0x3FDB_CB7B_1526_E50E);
const LOG10E_LO: f64 = f64::from_bits(0x3C69_5355_BAAA_FAD3);
/// 1.5 * 2^52: adding and subtracting it rounds to the nearest integer.
const ROUND_SHIFT: f64 = 6_755_399_441_055_744.0;

/// Dekker's split of `a` into a 26-bit high part and the rest.
#[inline(always)]
fn split(a: f64) -> (f64, f64) {
    let c = 134_217_729.0 * a;
    let hi = c - (c - a);
    (hi, a - hi)
}

/// `a + b` as `s + e` exactly (Knuth).
#[inline(always)]
fn two_sum(a: f64, b: f64) -> (f64, f64) {
    let s = a + b;
    let bb = s - a;
    (s, (a - (s - bb)) + (b - bb))
}

/// `ln(x)` as a double-double `hi + lo` (relative error about 2^-70) for a
/// positive normal finite `x`.
///
/// `x = 2^e m` with `m` in `[sqrt(1/2), sqrt(2))`, `m = c (1 + r)` with
/// `c = 1 + j/64` (so `m - c` is exact and `|r| <= 1/128`), and
/// `ln(x) = e ln 2 + ln(c) + ln(1 + r)` with `ln(c)` from a table and
/// `ln(1 + r)` from its series, the leading terms carried exactly.
#[inline(always)]
fn ln_dd<const FMA: bool>(x: f64) -> (f64, f64) {
    let bits = x.to_bits();
    let mant = bits & 0x000F_FFFF_FFFF_FFFF;
    // m > sqrt(2) <=> mantissa bits > those of sqrt(2) = 0x3FF6A09E667F3BCD
    let big = mant > 0x0006_A09E_667F_3BCD;
    let m = f64::from_bits(
        mant | if big {
            0x3FE0_0000_0000_0000
        } else {
            0x3FF0_0000_0000_0000
        },
    );
    // The exponent as f64 without an integer conversion: 2^52 + biased
    // exponent, minus 2^52 + 1023.
    let biased = f64::from_bits(0x4330_0000_0000_0000 | ((bits >> 52) + u64::from(big)));
    let e = biased - (4_503_599_627_370_496.0 + 1023.0);
    let shifted = (m - 1.0) * 64.0 + ROUND_SHIFT;
    let jf = shifted - ROUND_SHIFT;
    let idx = (shifted.to_bits().wrapping_add(19) & 63) as usize;
    let c = 1.0 + jf * (1.0 / 64.0);
    let u = m - c;
    // r = u / c as r_hi + r_lo (c has at most 7 significant bits, so the
    // products of c with the halves of r_hi are exact).
    // r_hi is u / c to within a few ulp; r_lo carries the exact remainder
    // (u - r_hi c) / c, so r_hi + r_lo = u / c to about 2^-100.
    let inv_c = 1.0 / c;
    let r_hi = u * inv_c;
    // The residual u - r_hi c and the rounding error of r_hi^2 are exact
    // (Dekker's products, or one fused multiply-add each: the same values),
    // so both paths give identical results.
    let s_hi = r_hi * r_hi;
    let (r_lo, s_lo) = if FMA {
        ((-r_hi).mul_add(c, u) * inv_c, r_hi.mul_add(r_hi, -s_hi))
    } else {
        let (rh, rl) = split(r_hi);
        (
            ((u - rh * c) - rl * c) * inv_c,
            ((rh * rh - s_hi) + 2.0 * (rh * rl)) + rl * rl,
        )
    };
    let r = r_hi;
    let tail = r
        * s_hi
        * (1.0 / 3.0
            - r * (1.0 / 4.0
                - r * (1.0 / 5.0
                    - r * (1.0 / 6.0
                        - r * (1.0 / 7.0
                            - r * (1.0 / 8.0
                                - r * (1.0 / 9.0 - r * (1.0 / 10.0 - r * (1.0 / 11.0)))))))));
    // ln(1 + r) = r - r^2/2 + tail, with r = r_hi + r_lo and
    // r^2 / 2 = s_hi / 2 + s_lo / 2 + r_hi r_lo + ...
    let small = r_lo - r_hi * r_lo - 0.5 * s_lo + tail;
    let (lc_hi, lc_lo) = (LN_C_HI[idx], LN_C_LO[idx]);
    let (h1, l1) = two_sum(e * LN2_HI, lc_hi);
    let (h2, l2) = two_sum(h1, r_hi);
    let (h3, l3) = two_sum(h2, -0.5 * s_hi);
    let lo = l1 + l2 + l3 + (e * LN2_LO + lc_lo + small);
    let hi = h3 + lo;
    (hi, lo - (hi - h3))
}

/// Whether `bits` is a positive normal finite `f64`.
#[inline(always)]
fn positive_normal_f64(bits: u64) -> bool {
    bits.wrapping_sub(0x0010_0000_0000_0000) < 0x7FE0_0000_0000_0000
}

/// `log10(x)` of a positive normal finite `f64`: the double-double natural
/// logarithm times `log10(e)` in double-double, rounded once.
#[inline(always)]
fn log10_f64_normal<const FMA: bool>(x: f64) -> f64 {
    // log10(e)_hi split into 26-bit halves (constant).
    const E_H: f64 = f64::from_bits(0x3FDB_CB7B_1800_0000);
    const E_L: f64 = LOG10E_HI - E_H;
    let (h, l) = ln_dd::<FMA>(x);
    let p = h * LOG10E_HI;
    let pe = if FMA {
        h.mul_add(LOG10E_HI, -p)
    } else {
        let (hh, hl) = split(h);
        ((hh * E_H - p) + hh * E_L + hl * E_H) + hl * E_L
    };
    p + (pe + (h * LOG10E_LO + l * LOG10E_HI))
}

#[inline(always)]
fn log10_f64_generic<const FMA: bool>(v: &mut [f64]) {
    for chunk in v.chunks_mut(64) {
        if chunk
            .iter()
            .fold(true, |a, x| a & positive_normal_f64(x.to_bits()))
        {
            for x in chunk.iter_mut() {
                *x = log10_f64_normal::<FMA>(*x);
            }
        } else {
            for x in chunk.iter_mut() {
                *x = if positive_normal_f64(x.to_bits()) {
                    log10_f64_normal::<FMA>(*x)
                } else {
                    x.log10()
                };
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
fn log10_f64_avx2_fma(v: &mut [f64]) {
    log10_f64_generic::<true>(v);
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
fn log10_f64_avx2(v: &mut [f64]) {
    log10_f64_generic::<false>(v);
}

/// `x = log10(x)` for every element of an `f64` slice, vectorised.
///
/// The result is the correctly rounded `log10(x)` except when the exact value
/// lies within about 2^-70 (relative) of a rounding boundary; it differs from
/// the C library's `log10` (which is not correctly rounded) by at most one
/// unit in the last place. Zero, negative, subnormal, infinite and NaN inputs
/// go to the standard library, so the special values are unchanged.
pub(crate) fn log10_f64(v: &mut [f64]) {
    #[cfg(target_arch = "x86_64")]
    if has_avx2() {
        if std::arch::is_x86_feature_detected!("fma") {
            // SAFETY: AVX2 and FMA are available (checked at run time).
            unsafe { log10_f64_avx2_fma(v) };
        } else {
            // SAFETY: AVX2 is available (checked at run time).
            unsafe { log10_f64_avx2(v) };
        }
        return;
    }
    log10_f64_generic::<false>(v);
}

/// `x = log10(x)` element-wise for `f32` or `f64` slices (the vectorised
/// kernels above).
pub(crate) fn log10_in_place<F: AudioFloat>(v: &mut [F]) {
    if let Some(v32) = as_f32(v) {
        log10_f32(v32);
        return;
    }
    if TypeId::of::<F>() == TypeId::of::<f64>() {
        // SAFETY: F is f64 (same type, so same size, alignment and validity).
        let v64 = unsafe { std::slice::from_raw_parts_mut(v.as_mut_ptr().cast::<f64>(), v.len()) };
        log10_f64(v64);
        return;
    }
    for x in v {
        *x = x.log10();
    }
}

/// The slice as `&mut [f32]` when `F` is `f32`.
pub(crate) fn as_f32<F: AudioFloat>(v: &mut [F]) -> Option<&mut [f32]> {
    if TypeId::of::<F>() == TypeId::of::<f32>() {
        // SAFETY: F is f32 (same type, so same size, alignment and validity).
        Some(unsafe { std::slice::from_raw_parts_mut(v.as_mut_ptr().cast::<f32>(), v.len()) })
    } else {
        None
    }
}

/// [`as_f32`] of a shared slice.
pub(crate) fn as_f32_slice<F: AudioFloat>(v: &[F]) -> Option<&[f32]> {
    if TypeId::of::<F>() == TypeId::of::<f32>() {
        // SAFETY: F is f32 (same type, so same size, alignment and validity).
        Some(unsafe { std::slice::from_raw_parts(v.as_ptr().cast::<f32>(), v.len()) })
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

    fn ulp_distance(a: f32, b: f32) -> i64 {
        let key = |v: f32| {
            let bits = v.to_bits() as i64;
            if v.is_sign_negative() {
                (1i64 << 31) - bits
            } else {
                bits
            }
        };
        (key(a) - key(b)).abs()
    }

    #[test]
    fn atan2_norm_is_close_to_the_f64_result() {
        let mut state = 0x9e37_79b9_7f4a_7c15_u64;
        let mut next = || {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1);
            let bits = (state >> 32) as u32;
            let exp = (bits >> 8) % 254;
            f32::from_bits((bits & 0x8000_0000) | (exp << 23) | (bits.rotate_left(7) & 0x7f_ffff))
        };
        let n = 2_000_000;
        let (y, x): (Vec<f32>, Vec<f32>) = (0..n).map(|_| (next(), next())).unzip();
        let (mut a, mut m) = (vec![0.0; n], vec![0.0; n]);
        atan2_norm_f32(&y, &x, &mut a, &mut m);
        let (mut worst, mut worst_norm) = (0, 0);
        for i in 0..n {
            let (yi, xi) = (f64::from(y[i]), f64::from(x[i]));
            worst = worst.max(ulp_distance(a[i], yi.atan2(xi) as f32));
            worst_norm = worst_norm.max(ulp_distance(m[i], yi.hypot(xi) as f32));
        }
        assert!(worst <= 1, "angle {worst} ulps");
        assert!(worst_norm <= 1, "norm {worst_norm} ulps");
    }

    #[test]
    fn atan2_norm_special_values_match_f64() {
        let specials = [
            0.0f32,
            -0.0,
            1.0,
            -1.0,
            f32::MIN_POSITIVE,
            1e-30,
            -1e-30,
            1e30,
            -1e30,
            f32::MAX,
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
        ];
        for &yi in &specials {
            for &xi in &specials {
                let (mut a, mut m) = ([0.0f32; 16], [0.0f32; 16]);
                atan2_norm_f32(&[yi; 16], &[xi; 16], &mut a, &mut m);
                let want = f64::from(yi).atan2(f64::from(xi)) as f32;
                let want_norm = xi.hypot(yi);
                for (got, got_norm) in a.iter().zip(&m) {
                    assert!(
                        ulp_distance(*got, want) <= 1 || (got.is_nan() && want.is_nan()),
                        "atan2({yi}, {xi}) = {got}, want {want}"
                    );
                    assert!(
                        ulp_distance(*got_norm, want_norm) <= 1
                            || (got_norm.is_nan() && want_norm.is_nan()),
                        "hypot({xi}, {yi}) = {got_norm}, want {want_norm}"
                    );
                }
            }
        }
    }

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

    #[test]
    fn exp10_f32_within_one_ulp() {
        let mut ys: Vec<f32> = (0..2_000_000).map(|i| -120.0 + i as f32 * 1.6e-4).collect();
        ys.extend([
            0.0,
            -0.0,
            1.0,
            2.0,
            -1.0,
            38.0,
            39.0,
            -45.0,
            -46.0,
            300.0,
            301.0,
            -301.0,
            f32::NAN,
            f32::INFINITY,
            f32::NEG_INFINITY,
        ]);
        let mut v = ys.clone();
        exp10_f32(&mut v);
        let mut worst = 0;
        for (&y, &got) in ys.iter().zip(&v) {
            let want = 10f64.powf(f64::from(y)) as f32;
            if y.is_nan() {
                assert!(got.is_nan());
                continue;
            }
            worst = worst.max(ulps(got, want));
        }
        assert!(worst <= 1, "{worst} ulp");
        assert_eq!(v[ys.len() - 15], 1.0);
        assert_eq!(v[ys.len() - 12], 100.0);
    }

    fn ulps64(a: f64, b: f64) -> u64 {
        if a == b || (a.is_nan() && b.is_nan()) {
            return 0;
        }
        (a.to_bits() as i128 - b.to_bits() as i128).unsigned_abs() as u64
    }

    #[test]
    fn log10_f64_close_to_the_c_library() {
        let mut st = 0x9e37_79b9_7f4a_7c15u64;
        let mut xs = Vec::new();
        for n in 0..3_000_000u64 {
            st = st
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            xs.push(match n % 4 {
                0 => f64::from_bits(st >> 1),
                1 => f64::from_bits((st >> 12) | 0x3fe0_0000_0000_0000),
                2 => 1.0 + ((st >> 11) as f64 / (1u64 << 53) as f64 - 0.5) * 1e-6,
                _ => (st >> 11) as f64 * 1e-10,
            });
        }
        xs.extend([
            0.0,
            -0.0,
            -1.0,
            1.0,
            10.0,
            1e-300,
            5e-324,
            2.2e-308,
            f64::MAX,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::NAN,
            0.1,
            1000.0,
            1e22,
            1e-22,
            std::f64::consts::SQRT_2,
        ]);
        let mut v = xs.clone();
        log10_f64(&mut v);
        if let Ok(path) = std::env::var("AUDIO_SAMPLES_DUMP_LOG10") {
            let mut out = String::new();
            for (x, y) in xs
                .iter()
                .zip(&v)
                .filter(|(x, y)| ulps64(**y, x.log10()) > 0)
                .take(400_000)
            {
                out.push_str(&format!(
                    "{:016x} {:016x} {:016x}\n",
                    x.to_bits(),
                    y.to_bits(),
                    x.log10().to_bits()
                ));
            }
            std::fs::write(path, out).unwrap();
        }
        let mut worst = 0;
        for (&x, &got) in xs.iter().zip(&v) {
            let want = x.log10();
            if !(x > 0.0 && x.is_finite()) || x < f64::MIN_POSITIVE {
                assert_eq!(got.to_bits(), want.to_bits(), "special {x:e}");
            }
            worst = worst.max(ulps64(got, want));
        }
        // glibc's log10 is within about 2 ulp of the exact value just around
        // x = 1 and within 1 ulp elsewhere; every result above that differs
        // from it was checked against 50-digit decimal arithmetic to be the
        // correctly rounded value.
        assert!(worst <= 2, "{worst} ulp");
        for k in -22..=22 {
            let x = 10f64.powi(k);
            if x == format!("1e{k}").parse::<f64>().unwrap() && k >= 0 {
                let mut y = [x];
                log10_f64(&mut y);
                assert_eq!(y[0], f64::from(k), "1e{k}");
            }
        }
    }

    #[test]
    fn log10_f64_fused_and_unfused_paths_agree() {
        let mut st = 0x2545_f491_4f6c_dd1du64;
        let xs: Vec<f64> = (0..1_000_000)
            .map(|n| {
                st = st
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                if n % 2 == 0 {
                    f64::from_bits((st >> 2) | 0x0010_0000_0000_0000)
                } else {
                    f64::from_bits((st >> 12) | 0x3fe0_0000_0000_0000)
                }
            })
            .collect();
        let mut a = xs.clone();
        log10_f64_generic::<false>(&mut a);
        let mut b = xs.clone();
        log10_f64_generic::<true>(&mut b);
        for ((x, p), q) in xs.iter().zip(&a).zip(&b) {
            assert_eq!(p.to_bits(), q.to_bits(), "{x:e}");
        }
    }
}
