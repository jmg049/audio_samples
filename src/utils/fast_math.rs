//! A branch-free `sin` kernel for loops that should vectorise.
//!
//! It follows FreeBSD's msun (fdlibm) algorithm and coefficients: the
//! result is within 1 ulp of the exact value, like the C library's, but is
//! not always bit-identical to it. Every step is plain arithmetic and bit
//! manipulation (no table lookups, branches or calls), so the compiler can
//! vectorise a loop over it. Callers check their arguments against the
//! valid range and fall back to the standard library outside it.

/// Largest `|x|` accepted by [`sin_kernel`]: `2^20 · π/2`, so that the
/// quadrant `n` fits in 20 bits and `n · PIO2_1` is exact.
pub(crate) const SIN_MAX_ARG: f64 = 1_647_099.0;

const fn f(bits: u64) -> f64 {
    f64::from_bits(bits)
}

// e_rem_pio2.c: 2/π and π/2 split into 33-bit pieces with their tails.
const INVPIO2: f64 = f(0x3fe4_5f30_6dc9_c883);
const PIO2_1: f64 = f(0x3ff9_21fb_5440_0000);
const PIO2_1T: f64 = f(0x3dd0_b461_1a62_6331);
const PIO2_2: f64 = f(0x3dd0_b461_1a60_0000);
const PIO2_2T: f64 = f(0x3ba3_198a_2e03_7073);
const PIO2_3: f64 = f(0x3ba3_198a_2e00_0000);
const PIO2_3T: f64 = f(0x397b_839a_2520_49c1);
// k_sin.c
const S1: f64 = f(0xbfc5_5555_5555_5549);
const S2: f64 = f(0x3f81_1111_1110_f8a6);
const S3: f64 = f(0xbf2a_01a0_19c1_61d5);
const S4: f64 = f(0x3ec7_1de3_57b1_fe7d);
const S5: f64 = f(0xbe5a_e5e6_8a2b_9ceb);
const S6: f64 = f(0x3de5_d93a_5acf_d57c);
// k_cos.c
const C1: f64 = f(0x3fa5_5555_5555_554c);
const C2: f64 = f(0xbf56_c16c_16c1_5177);
const C3: f64 = f(0x3efa_01a0_19cb_1590);
const C4: f64 = f(0xbe92_7e4f_809c_52ad);
const C5: f64 = f(0x3e21_ee9e_bdb4_b1c4);
const C6: f64 = f(0xbda8_fae9_be88_38d4);

/// `1.5 · 2^52`: adding and subtracting it rounds a value below `2^51` in
/// magnitude to the nearest integer (ties to even), and the low bits of the
/// sum's representation hold that integer.
const ROUNDER: f64 = 6_755_399_441_055_744.0;

/// `sin(x)` for `|x| <= SIN_MAX_ARG` (and finite), within 1 ulp.
///
/// The argument is reduced by `π/2` as fdlibm's `__ieee754_rem_pio2` does
/// for medium arguments (up to three Cody–Waite rounds, 151 bits of `π/2`)
/// to `y0 + y1` in about `[-π/4, π/4]`, then the sine or cosine kernel of
/// the quadrant is applied.
#[inline(always)]
pub(crate) fn sin_kernel(x: f64) -> f64 {
    let shifted = x * INVPIO2 + ROUNDER;
    let n = shifted.to_bits() & 3;
    let fnq = shifted - ROUNDER;
    let exponent = |v: f64| ((v.to_bits() >> 52) & 0x7ff) as i64;
    let j = exponent(x);
    // fdlibm's medium-size reduction, with its three rounds evaluated and
    // the result selected by the same cancellation tests.
    let r1 = x - fnq * PIO2_1;
    let w1 = fnq * PIO2_1T;
    let y01 = r1 - w1;
    let t = r1;
    let w = fnq * PIO2_2;
    let r2 = t - w;
    let w2 = fnq * PIO2_2T - ((t - r2) - w);
    let y02 = r2 - w2;
    let t = r2;
    let w = fnq * PIO2_3;
    let r3 = t - w;
    let w3 = fnq * PIO2_3T - ((t - r3) - w);
    let y03 = r3 - w3;
    let second = j - exponent(y01) > 16;
    let third = second && j - exponent(y02) > 49;
    let (r, w, y0) = if third {
        (r3, w3, y03)
    } else if second {
        (r2, w2, y02)
    } else {
        (r1, w1, y01)
    };
    let y1 = (r - y0) - w;

    let z = y0 * y0;
    let zz = z * z;
    // __kernel_sin(y0, y1, 1)
    let rs = S2 + z * (S3 + z * S4) + z * zz * (S5 + z * S6);
    let v = z * y0;
    let ks = y0 - ((z * (0.5 * y1 - v * rs) - y1) - v * S1);
    // __kernel_cos(y0, y1)
    let rc = z * (C1 + z * (C2 + z * C3)) + zz * zz * (C4 + z * (C5 + z * C6));
    let hz = 0.5 * z;
    let wc = 1.0 - hz;
    let kc = wc + (((1.0 - wc) - hz) + (z * rc - y0 * y1));

    let base = if n & 1 == 0 { ks } else { kc };
    if n & 2 == 0 { base } else { -base }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ulp_diff(a: f64, b: f64) -> f64 {
        let ulp = f64::EPSILON * b.abs().max(f64::MIN_POSITIVE);
        (a - b).abs() / ulp
    }

    #[test]
    fn sin_kernel_is_within_one_ulp() {
        // Dense near zero and every quadrant boundary, sparse up to the limit.
        let mut worst = 0.0_f64;
        let mut x = -SIN_MAX_ARG;
        while x <= SIN_MAX_ARG {
            for v in [x, x * 1e-6, x * 1e-3, x * 1e-12] {
                worst = worst.max(ulp_diff(sin_kernel(v), v.sin()));
            }
            x += 13.371;
        }
        for k in -1000_i32..1000 {
            let c = f64::from(k) * std::f64::consts::FRAC_PI_2;
            for d in [-1e-9, 0.0, 1e-9, 0.3, 0.785] {
                let v = c + d;
                worst = worst.max(ulp_diff(sin_kernel(v), v.sin()));
            }
        }
        assert!(worst <= 1.0, "worst error {worst} ulp");
        assert_eq!(sin_kernel(0.0), 0.0);
    }
}
