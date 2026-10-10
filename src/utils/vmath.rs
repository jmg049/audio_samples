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
//! `f64` slices use the standard library's scalar functions unchanged.

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

/// `x = log10(x)` element-wise: the vectorised kernel for `f32`, the
/// standard library's `log10` for `f64`.
#[allow(dead_code)] // used by the dB conversions
pub(crate) fn log10_in_place<F: AudioFloat>(v: &mut [F]) {
    match as_f32(v) {
        Some(v) => log10_f32(v),
        None => {
            for x in v {
                *x = x.log10();
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
}
