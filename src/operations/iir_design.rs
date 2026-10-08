//! IIR filter design core: analog prototypes, frequency transforms, the
//! bilinear transform and zero-pole-gain → second-order-sections conversion.
//!
//! This module is a faithful port of the SciPy `scipy.signal` design pipeline
//! (`iirfilter` → `buttap`/`cheb1ap`/`cheb2ap`/`besselap`/`ellipap` →
//! `lp2{lp,hp,bp,bs}_zpk` → `bilinear_zpk` → `zpk2sos`). It deliberately follows
//! the same algorithms, the same internal frequency normalisation (`fs = 2`)
//! and the same section pairing/ordering rules so that designed coefficients
//! agree with SciPy to within a few ULPs. That agreement is enforced by the
//! golden-fixture tests in `tests/filter_reference.rs`.
//!
//! Everything here works on `f64` and is private to the crate; the public
//! entry points are [`crate::operations::iir_filtering::SosFilter`] and the
//! [`crate::operations::traits::AudioIirFiltering`] trait.

use std::f64::consts::PI;

use num_complex::Complex;

use crate::operations::types::{BesselNorm, FilterResponse, IirFilterDesign, IirFilterType};
use crate::{AudioSampleError, AudioSampleResult, ParameterError};

type C64 = Complex<f64>;

/// One second-order section in SciPy layout: `[b0, b1, b2, a0, a1, a2]`.
pub(crate) type Sos = [f64; 6];

/// Highest supported filter order.
///
/// The SOS pipeline is numerically sound well beyond this, but the Bessel
/// root finder and the elliptic special functions are only verified against
/// SciPy up to this order (see the reference fixtures).
pub(crate) const MAX_IIR_ORDER: usize = 24;

/// Analog or digital zero-pole-gain description.
#[derive(Debug, Clone)]
pub(crate) struct Zpk {
    pub(crate) zeros: Vec<C64>,
    pub(crate) poles: Vec<C64>,
    pub(crate) gain: f64,
}

fn prod_neg(roots: &[C64]) -> C64 {
    roots.iter().fold(C64::new(1.0, 0.0), |acc, &r| acc * (-r))
}

/// SciPy `m = arange(-N+1, N, 2)`.
fn symmetric_indices(n: usize) -> impl Iterator<Item = f64> {
    let n = n as i64;
    (0..n).map(move |i| (-n + 1 + 2 * i) as f64)
}

// ============================================================================
// Analog low-pass prototypes (cutoff 1 rad/s)
// ============================================================================

/// `scipy.signal.buttap`.
pub(crate) fn buttap(n: usize) -> Zpk {
    let nf = n as f64;
    let poles = symmetric_indices(n)
        .map(|m| -(C64::new(0.0, PI * m / (2.0 * nf))).exp())
        .collect();
    Zpk {
        zeros: Vec::new(),
        poles,
        gain: 1.0,
    }
}

/// `scipy.signal.cheb1ap`.
pub(crate) fn cheb1ap(n: usize, rp: f64) -> Zpk {
    let nf = n as f64;
    let eps = (10f64.powf(0.1 * rp) - 1.0).sqrt();
    let mu = 1.0 / nf * (1.0 / eps).asinh();
    let poles: Vec<C64> = symmetric_indices(n)
        .map(|m| {
            let theta = PI * m / (2.0 * nf);
            -(C64::new(mu, theta)).sinh()
        })
        .collect();
    let mut gain = prod_neg(&poles).re;
    if n.is_multiple_of(2) {
        gain /= eps.mul_add(eps, 1.0).sqrt();
    }
    Zpk {
        zeros: Vec::new(),
        poles,
        gain,
    }
}

/// `scipy.signal.cheb2ap`.
pub(crate) fn cheb2ap(n: usize, rs: f64) -> Zpk {
    let nf = n as f64;
    let de = 1.0 / (10f64.powf(0.1 * rs) - 1.0).sqrt();
    let mu = (1.0 / de).asinh() / nf;

    // Zeros: m = arange(-N+1, N, 2) with the centre (m = 0) removed for odd N.
    let zeros: Vec<C64> = symmetric_indices(n)
        .filter(|&m| m != 0.0)
        .map(|m| -(C64::new(0.0, 1.0) / (m * PI / (2.0 * nf)).sin()).conj())
        .collect();

    let poles: Vec<C64> = symmetric_indices(n)
        .map(|m| {
            let p = -(C64::new(0.0, PI * m / (2.0 * nf))).exp();
            let p = C64::new(mu.sinh() * p.re, mu.cosh() * p.im);
            C64::new(1.0, 0.0) / p
        })
        .collect();

    let gain = (prod_neg(&poles) / prod_neg(&zeros)).re;
    Zpk { zeros, poles, gain }
}

/// Coefficients of the reverse Bessel polynomial θ_N, lowest degree first:
/// `a_k = (2N - k)! / (2^(N-k) · k! · (N-k)!)`.
fn reverse_bessel_coeffs(n: usize) -> Vec<f64> {
    // a_N = 1 and a_{k-1} / a_k = (2N - k + 1) · k / (2 (N - k + 1)).
    let mut a = vec![0.0; n + 1];
    a[n] = 1.0;
    for k in (1..=n).rev() {
        let kf = k as f64;
        let nf = n as f64;
        a[k - 1] = a[k] * (2.0 * nf - kf + 1.0) * kf / (nf - kf + 1.0) / 2.0;
    }
    a
}

// ---------------------------------------------------------------------------
// Bessel polynomial zeros
// ---------------------------------------------------------------------------
//
// The zeros of the ordinary Bessel polynomial y_N are badly conditioned with
// respect to plain floating-point evaluation (the recurrence cancels by ~1e12
// near the zeros at N = 24). Following SciPy's `_bessel_zeros`, we start from
// Campos & Calderón's asymptotic approximations and refine with Aberth-Ehrlich
// iterations, but evaluate y_N in double-double arithmetic (~32 significant
// digits) so the refined zeros are accurate to f64 precision.

/// Double-double number `hi + lo`.
#[derive(Clone, Copy)]
struct Dd {
    hi: f64,
    lo: f64,
}

impl Dd {
    const fn new(v: f64) -> Self {
        Self { hi: v, lo: 0.0 }
    }

    fn quick_two_sum(a: f64, b: f64) -> Self {
        let s = a + b;
        Self {
            hi: s,
            lo: b - (s - a),
        }
    }

    fn add(self, o: Self) -> Self {
        let s = self.hi + o.hi;
        let bb = s - self.hi;
        let err = (self.hi - (s - bb)) + (o.hi - bb);
        Self::quick_two_sum(s, err + self.lo + o.lo)
    }

    fn neg(self) -> Self {
        Self {
            hi: -self.hi,
            lo: -self.lo,
        }
    }

    fn mul_f64(self, c: f64) -> Self {
        let p = self.hi * c;
        let err = self.hi.mul_add(c, -p);
        Self::quick_two_sum(p, err + self.lo * c)
    }
}

/// Complex double-double.
#[derive(Clone, Copy)]
struct CDd {
    re: Dd,
    im: Dd,
}

impl CDd {
    const fn from_c64(z: C64) -> Self {
        Self {
            re: Dd::new(z.re),
            im: Dd::new(z.im),
        }
    }

    fn add(self, o: Self) -> Self {
        Self {
            re: self.re.add(o.re),
            im: self.im.add(o.im),
        }
    }

    fn mul_c64(self, z: C64) -> Self {
        Self {
            re: self.re.mul_f64(z.re).add(self.im.mul_f64(z.im).neg()),
            im: self.re.mul_f64(z.im).add(self.im.mul_f64(z.re)),
        }
    }

    fn mul_f64(self, c: f64) -> Self {
        Self {
            re: self.re.mul_f64(c),
            im: self.im.mul_f64(c),
        }
    }

    fn to_c64(self) -> C64 {
        C64::new(self.re.hi + self.re.lo, self.im.hi + self.im.lo)
    }
}

/// Ordinary Bessel polynomial `y_N(x)` (double-double) and its derivative
/// (double) by the recurrence `y_n = (2n - 1)·x·y_{n-1} + y_{n-2}`.
fn bessel_poly_eval(n: usize, x: C64) -> (C64, C64) {
    let one = C64::new(1.0, 0.0);
    if n == 0 {
        return (one, C64::new(0.0, 0.0));
    }
    let mut y_prev = CDd::from_c64(one);
    // y_1 = 1 + x, formed exactly in double-double (rounding it in f64 would
    // be amplified by the recurrence).
    let mut y = CDd {
        re: Dd::new(1.0).add(Dd::new(x.re)),
        im: Dd::new(x.im),
    };
    let (mut d_prev, mut d) = (C64::new(0.0, 0.0), one);
    for k in 2..=n {
        let c = (2 * k - 1) as f64;
        let y_next = y.mul_c64(x).mul_f64(c).add(y_prev);
        let d_next = c * (y.to_c64() + x * d) + d_prev;
        y_prev = y;
        y = y_next;
        d_prev = d;
        d = d_next;
    }
    (y.to_c64(), d)
}

/// Campos & Calderón (2011) starting points for the zeros of `y_N`
/// (SciPy `_campos_zeros`).
fn campos_zeros(n: usize) -> Vec<C64> {
    if n == 1 {
        return vec![C64::new(-1.0, 0.0)];
    }
    let nf = n as f64;
    let poly = |c: &[f64]| c.iter().rev().fold(0.0, |acc, &v| acc * nf + v);
    let s = poly(&[0.0, 0.0, 2.0, 0.0, -3.0, 1.0]);
    let b3 = poly(&[16.0, -8.0]) / s;
    let b2 = poly(&[-24.0, -12.0, 12.0]) / s;
    let b1 = poly(&[8.0, 24.0, -12.0, -2.0]) / s;
    let b0 = poly(&[0.0, -6.0, 0.0, 5.0, -1.0]) / s;
    let r = poly(&[0.0, 0.0, 2.0, 1.0]);
    let a1 = poly(&[-6.0, -6.0]) / r;
    let a2 = 6.0 / r;
    (1..=n)
        .map(|k| {
            let k = k as f64;
            let x = a1 * k + a2 * k * k;
            let y = b0 + b1 * k + b2 * k * k + b3 * k * k * k;
            C64::new(x, y)
        })
        .collect()
}

/// Zeros of the ordinary Bessel polynomial `y_N` (SciPy `_bessel_zeros`).
fn bessel_zeros(n: usize) -> Vec<C64> {
    let one = C64::new(1.0, 0.0);
    let mut x = campos_zeros(n);
    for _ in 0..200 {
        let mut max_step = 0.0f64;
        for i in 0..n {
            let (f, fp) = bessel_poly_eval(n, x[i]);
            if f.norm() == 0.0 {
                continue;
            }
            let ratio = f / fp;
            let sum = (0..n)
                .filter(|&j| j != i)
                .fold(C64::new(0.0, 0.0), |acc, j| acc + one / (x[i] - x[j]));
            let step = ratio / (one - ratio * sum);
            x[i] -= step;
            max_step = max_step.max(step.norm() / x[i].norm());
        }
        if max_step < 1e-17 {
            break;
        }
    }
    for xi in &mut x {
        for _ in 0..3 {
            let (f, fp) = bessel_poly_eval(n, *xi);
            if fp.norm() == 0.0 {
                break;
            }
            *xi -= f / fp;
        }
    }
    enforce_conjugate_symmetry(x)
}

/// Roots of the reverse Bessel polynomial θ_N (the delay-normalised Bessel
/// poles): the reciprocals of the zeros of `y_N`.
fn reverse_bessel_roots(n: usize) -> Vec<C64> {
    bessel_zeros(n)
        .into_iter()
        .map(|x| C64::new(1.0, 0.0) / x)
        .collect()
}

/// Make a root set of a real polynomial exactly conjugate-symmetric.
///
/// Roots with a negligible imaginary part become real; every root in the
/// upper half-plane is paired with the closest lower-half-plane root, and the
/// pair is replaced by the average `(p + conj(q)) / 2` and its conjugate.
fn enforce_conjugate_symmetry(roots: Vec<C64>) -> Vec<C64> {
    let mut real = Vec::new();
    let mut upper = Vec::new();
    let mut lower = Vec::new();
    for r in roots {
        if r.im.abs() <= 1e-10 * r.norm().max(1.0) {
            real.push(C64::new(r.re, 0.0));
        } else if r.im > 0.0 {
            upper.push(r);
        } else {
            lower.push(r);
        }
    }
    let mut out = real;
    for p in upper {
        let (idx, _) = lower
            .iter()
            .enumerate()
            .map(|(i, q)| (i, (p - q.conj()).norm()))
            .min_by(|a, b| a.1.total_cmp(&b.1))
            .unwrap_or((usize::MAX, 0.0));
        let avg = if idx == usize::MAX {
            p
        } else {
            let q = lower.swap_remove(idx);
            (p + q.conj()) / 2.0
        };
        out.push(avg);
        out.push(avg.conj());
    }
    out
}

/// `scipy.signal.besselap`.
pub(crate) fn besselap(n: usize, norm: BesselNorm) -> Zpk {
    let coeffs = reverse_bessel_coeffs(n);
    // Poles normalised for unit group delay at DC.
    let a_last = coeffs[0];
    let mut poles = reverse_bessel_roots(n);

    let gain = match norm {
        BesselNorm::Delay => a_last,
        BesselNorm::Magnitude => {
            // Find w with |H(jw)| = 1/sqrt(2) (monotone decreasing).
            let mag = |w: f64| -> f64 {
                let s = C64::new(0.0, w);
                let den = poles
                    .iter()
                    .fold(C64::new(1.0, 0.0), |acc, &p| acc * (s - p));
                (a_last / den).norm()
            };
            let target = std::f64::consts::FRAC_1_SQRT_2;
            let (mut lo, mut hi) = (1e-6f64, 1e3f64);
            for _ in 0..200 {
                let mid = 0.5 * (lo + hi);
                if mag(mid) > target {
                    lo = mid;
                } else {
                    hi = mid;
                }
            }
            let norm_factor = 0.5 * (lo + hi);
            for p in &mut poles {
                *p /= norm_factor;
            }
            norm_factor.powi(-(n as i32)) * a_last
        }
        BesselNorm::Phase => {
            let scale = 10f64.powf(-a_last.log10() / n as f64);
            for p in &mut poles {
                *p *= scale;
            }
            1.0
        }
    };
    Zpk {
        zeros: Vec::new(),
        poles,
        gain,
    }
}

// ---------------------------------------------------------------------------
// Elliptic special functions (ports of the Cephes routines SciPy uses)
// ---------------------------------------------------------------------------

/// Arithmetic-geometric mean.
fn agm(mut a: f64, mut b: f64) -> f64 {
    for _ in 0..64 {
        if (a - b).abs() <= f64::EPSILON * a {
            break;
        }
        let an = 0.5 * (a + b);
        b = (a * b).sqrt();
        a = an;
    }
    0.5 * (a + b)
}

/// Complete elliptic integral of the first kind, parameter `m` (SciPy `ellipk`).
fn ellipk(m: f64) -> f64 {
    if m >= 1.0 {
        return f64::INFINITY;
    }
    PI / (2.0 * agm(1.0, (1.0 - m).sqrt()))
}

/// `K(1 - p)` evaluated without cancellation (SciPy `ellipkm1`).
fn ellipkm1(p: f64) -> f64 {
    if p <= 0.0 {
        return f64::INFINITY;
    }
    PI / (2.0 * agm(1.0, p.sqrt()))
}

/// Jacobi elliptic functions `(sn, cn, dn)` (Cephes `ellpj`, as used by SciPy).
fn ellipj(u: f64, m: f64) -> (f64, f64, f64) {
    const MACHEP: f64 = 1.110_223_024_625_156_5e-16;
    if m < 1.0e-9 {
        let t = u.sin();
        let b = u.cos();
        let ai = 0.25 * m * (u - t * b);
        return (t - ai * b, b + ai * t, 1.0 - 0.5 * m * t * t);
    }
    if m >= 0.999_999_999_9 {
        let mut ai = 0.25 * (1.0 - m);
        let b = u.cosh();
        let t = u.tanh();
        let phi = 1.0 / b;
        let twon = b * u.sinh();
        let sn = t + ai * (twon - u) / (b * b);
        ai *= t * phi;
        let cn = phi - ai * (twon - u);
        let dn = phi + ai * (twon + u);
        return (sn, cn, dn);
    }
    let mut a = [0.0f64; 9];
    let mut c = [0.0f64; 9];
    a[0] = 1.0;
    let mut b = (1.0 - m).sqrt();
    c[0] = m.sqrt();
    let mut twon = 1.0;
    let mut i = 0usize;
    while (c[i] / a[i]).abs() > MACHEP {
        if i > 7 {
            break;
        }
        let ai = a[i];
        i += 1;
        c[i] = (ai - b) / 2.0;
        let t = (ai * b).sqrt();
        a[i] = (ai + b) / 2.0;
        b = t;
        twon *= 2.0;
    }
    let mut phi = twon * a[i] * u;
    let mut prev;
    loop {
        let t = c[i] * phi.sin() / a[i];
        prev = phi;
        phi = (t.asin() + phi) / 2.0;
        i -= 1;
        if i == 0 {
            break;
        }
    }
    let sn = phi.sin();
    let cn = phi.cos();
    let dn = cn / (phi - prev).cos();
    (sn, cn, dn)
}

/// Solve the degree equation for the elliptic modulus (SciPy `_ellipdeg`).
fn ellipdeg(n: usize, m1: f64) -> f64 {
    const MMAX: i32 = 7;
    let k1 = ellipk(m1);
    let k1p = ellipkm1(m1);
    let q1 = (-PI * k1p / k1).exp();
    let q = q1.powf(1.0 / n as f64);
    let num: f64 = (0..=MMAX).map(|m| q.powi(m * (m + 1))).sum();
    let den: f64 = 1.0 + 2.0 * (1..=MMAX + 1).map(|m| q.powi(m * m)).sum::<f64>();
    16.0 * q * (num / den).powi(4)
}

fn complex_asin(z: C64) -> C64 {
    let i = C64::new(0.0, 1.0);
    -i * (i * z + (C64::new(1.0, 0.0) - z * z).sqrt()).ln()
}

/// Inverse Jacobian elliptic `sn` for complex argument (SciPy `_arc_jac_sn`).
fn arc_jac_sn(w: C64, m: f64) -> C64 {
    let complement = |kx: C64| ((C64::new(1.0, 0.0) - kx) * (C64::new(1.0, 0.0) + kx)).sqrt();
    let k = m.sqrt();
    let mut ks = vec![k];
    let mut niter = 0;
    while *ks.last().unwrap_or(&0.0) != 0.0 && niter < 32 {
        let k_ = ks[ks.len() - 1];
        let k_p = complement(C64::new(k_, 0.0)).re;
        ks.push((1.0 - k_p) / (1.0 + k_p));
        niter += 1;
    }
    let big_k = ks[1..].iter().fold(1.0, |acc, &v| acc * (1.0 + v)) * PI / 2.0;
    let mut wn = w;
    for idx in 0..ks.len() - 1 {
        let kn = ks[idx];
        let knext = ks[idx + 1];
        wn = 2.0 * wn / ((1.0 + knext) * (C64::new(1.0, 0.0) + complement(kn * wn)));
    }
    let u = 2.0 / PI * complex_asin(wn);
    big_k * u
}

/// Real inverse Jacobian elliptic `sc` with complementary modulus.
fn arc_jac_sc1(w: f64, m: f64) -> f64 {
    arc_jac_sn(C64::new(0.0, w), m).im
}

/// `scipy.signal.ellipap`.
pub(crate) fn ellipap(n: usize, rp: f64, rs: f64) -> AudioSampleResult<Zpk> {
    const EPSILON: f64 = 2e-16;
    if n == 1 {
        let p = -(1.0 / (10f64.powf(0.1 * rp) - 1.0)).sqrt();
        return Ok(Zpk {
            zeros: Vec::new(),
            poles: vec![C64::new(p, 0.0)],
            gain: -p,
        });
    }
    let eps_sq = 10f64.powf(0.1 * rp) - 1.0;
    let eps = eps_sq.sqrt();
    let ck1_sq = eps_sq / (10f64.powf(0.1 * rs) - 1.0);
    if ck1_sq == 0.0 {
        return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
            "stopband_attenuation",
            "cannot design an elliptic filter with the given passband ripple and stopband attenuation",
        )));
    }
    let k1 = ellipk(ck1_sq);
    let m = ellipdeg(n, ck1_sq);
    let capk = ellipk(m);

    let js: Vec<f64> = ((1 - n % 2)..n).step_by(2).map(|j| j as f64).collect();
    let scd: Vec<(f64, f64, f64)> = js.iter().map(|&j| ellipj(j * capk / n as f64, m)).collect();

    let mut zeros: Vec<C64> = scd
        .iter()
        .filter(|(s, _, _)| s.abs() > EPSILON)
        .map(|(s, _, _)| C64::new(0.0, 1.0 / (m.sqrt() * s)))
        .collect();
    let conj: Vec<C64> = zeros.iter().map(|z| z.conj()).collect();
    zeros.extend(conj);

    let r = arc_jac_sc1(1.0 / eps, ck1_sq);
    let v0 = capk * r / (n as f64 * k1);
    let (sv, cv, dv) = ellipj(v0, 1.0 - m);

    let mut poles: Vec<C64> = scd
        .iter()
        .map(|&(s, c, d)| -C64::new(c * d * sv * cv, s * dv) / (1.0 - (d * sv).powi(2)))
        .collect();
    if n % 2 == 1 {
        let norm = poles.iter().map(|p| p.norm_sqr()).sum::<f64>().sqrt();
        let newp: Vec<C64> = poles
            .iter()
            .filter(|p| p.im.abs() > EPSILON * norm)
            .map(|p| p.conj())
            .collect();
        poles.extend(newp);
    } else {
        let conj: Vec<C64> = poles.iter().map(|p| p.conj()).collect();
        poles.extend(conj);
    }

    let mut gain = (prod_neg(&poles) / prod_neg(&zeros)).re;
    if n.is_multiple_of(2) {
        gain /= (1.0 + eps_sq).sqrt();
    }
    Ok(Zpk { zeros, poles, gain })
}

// ============================================================================
// Analog frequency transforms and the bilinear transform
// ============================================================================

fn relative_degree(z: &Zpk) -> usize {
    z.poles.len().saturating_sub(z.zeros.len())
}

/// `scipy.signal.lp2lp_zpk`.
pub(crate) fn lp2lp_zpk(z: &Zpk, wo: f64) -> Zpk {
    let degree = relative_degree(z) as i32;
    Zpk {
        zeros: z.zeros.iter().map(|&r| r * wo).collect(),
        poles: z.poles.iter().map(|&r| r * wo).collect(),
        gain: z.gain * wo.powi(degree),
    }
}

/// `scipy.signal.lp2hp_zpk`.
pub(crate) fn lp2hp_zpk(z: &Zpk, wo: f64) -> Zpk {
    let degree = relative_degree(z);
    let mut zeros: Vec<C64> = z.zeros.iter().map(|&r| wo / r).collect();
    zeros.extend(std::iter::repeat_n(C64::new(0.0, 0.0), degree));
    Zpk {
        zeros,
        poles: z.poles.iter().map(|&r| wo / r).collect(),
        gain: z.gain * (prod_neg(&z.zeros) / prod_neg(&z.poles)).re,
    }
}

fn split_band(roots: &[C64], wo: f64) -> Vec<C64> {
    let mut out = Vec::with_capacity(roots.len() * 2);
    out.extend(roots.iter().map(|&r| r + (r * r - wo * wo).sqrt()));
    out.extend(roots.iter().map(|&r| r - (r * r - wo * wo).sqrt()));
    out
}

/// `scipy.signal.lp2bp_zpk`.
pub(crate) fn lp2bp_zpk(z: &Zpk, wo: f64, bw: f64) -> Zpk {
    let degree = relative_degree(z) as i32;
    let z_lp: Vec<C64> = z.zeros.iter().map(|&r| r * (bw / 2.0)).collect();
    let p_lp: Vec<C64> = z.poles.iter().map(|&r| r * (bw / 2.0)).collect();
    let mut zeros = split_band(&z_lp, wo);
    zeros.extend(std::iter::repeat_n(C64::new(0.0, 0.0), degree as usize));
    Zpk {
        zeros,
        poles: split_band(&p_lp, wo),
        gain: z.gain * bw.powi(degree),
    }
}

/// `scipy.signal.lp2bs_zpk`.
pub(crate) fn lp2bs_zpk(z: &Zpk, wo: f64, bw: f64) -> Zpk {
    let degree = relative_degree(z);
    let z_hp: Vec<C64> = z.zeros.iter().map(|&r| (bw / 2.0) / r).collect();
    let p_hp: Vec<C64> = z.poles.iter().map(|&r| (bw / 2.0) / r).collect();
    let mut zeros = split_band(&z_hp, wo);
    zeros.extend(std::iter::repeat_n(C64::new(0.0, wo), degree));
    zeros.extend(std::iter::repeat_n(C64::new(0.0, -wo), degree));
    Zpk {
        zeros,
        poles: split_band(&p_hp, wo),
        gain: z.gain * (prod_neg(&z.zeros) / prod_neg(&z.poles)).re,
    }
}

/// `scipy.signal.bilinear_zpk`.
pub(crate) fn bilinear_zpk(z: &Zpk, fs: f64) -> Zpk {
    let degree = relative_degree(z);
    let fs2 = 2.0 * fs;
    let map = |r: &C64| (fs2 + r) / (fs2 - r);
    let mut zeros: Vec<C64> = z.zeros.iter().map(map).collect();
    zeros.extend(std::iter::repeat_n(C64::new(-1.0, 0.0), degree));
    let num = z
        .zeros
        .iter()
        .fold(C64::new(1.0, 0.0), |acc, &r| acc * (fs2 - r));
    let den = z
        .poles
        .iter()
        .fold(C64::new(1.0, 0.0), |acc, &r| acc * (fs2 - r));
    Zpk {
        zeros,
        poles: z.poles.iter().map(map).collect(),
        gain: z.gain * (num / den).re,
    }
}

// ============================================================================
// zpk2sos (SciPy pairing = 'nearest')
// ============================================================================

/// Split roots into conjugate-pair representatives (positive imaginary part)
/// and purely real roots, as SciPy's `_cplxreal`.
fn cplxreal(roots: &[C64]) -> AudioSampleResult<(Vec<C64>, Vec<f64>)> {
    if roots.is_empty() {
        return Ok((Vec::new(), Vec::new()));
    }
    let tol = 100.0 * f64::EPSILON;
    let mut z: Vec<C64> = roots.to_vec();
    // lexsort((abs(imag), real)): primary key real, secondary |imag|.
    z.sort_by(|a, b| {
        a.re.total_cmp(&b.re)
            .then_with(|| a.im.abs().total_cmp(&b.im.abs()))
    });
    let is_real = |r: &C64| r.im.abs() <= tol * r.norm();
    let zr: Vec<f64> = z.iter().filter(|r| is_real(r)).map(|r| r.re).collect();
    if zr.len() == z.len() {
        return Ok((Vec::new(), zr));
    }
    let zc_all: Vec<C64> = z.into_iter().filter(|r| !is_real(r)).collect();
    let mut zp: Vec<C64> = zc_all.iter().copied().filter(|r| r.im > 0.0).collect();
    let mut zn: Vec<C64> = zc_all.iter().copied().filter(|r| r.im < 0.0).collect();
    if zp.len() != zn.len() {
        return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
            "filter_design",
            "internal error: complex root without a matching conjugate",
        )));
    }
    // Runs of (approximately) equal real part are sorted by |imag|.
    let mut start = 0;
    while start < zp.len() {
        let mut stop = start + 1;
        while stop < zp.len() && (zp[stop].re - zp[stop - 1].re) <= tol * zp[stop - 1].norm() {
            stop += 1;
        }
        if stop - start > 1 {
            zp[start..stop].sort_by(|a, b| a.im.abs().total_cmp(&b.im.abs()));
            zn[start..stop].sort_by(|a, b| a.im.abs().total_cmp(&b.im.abs()));
        }
        start = stop;
    }
    for (p, n) in zp.iter().zip(zn.iter()) {
        if (p - n.conj()).norm() > tol * n.norm() {
            return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
                "filter_design",
                "internal error: complex root without a matching conjugate",
            )));
        }
    }
    let zc = zp
        .iter()
        .zip(zn.iter())
        .map(|(p, n)| (p + n.conj()) / 2.0)
        .collect();
    Ok((zc, zr))
}

fn is_real(r: &C64) -> bool {
    r.im == 0.0
}

/// Index of the root in `fro` nearest to `to`, restricted to real / complex /
/// any roots (SciPy `_nearest_real_complex_idx`). Ties resolve to the lowest
/// index, as a stable `argsort` would.
fn nearest_idx(fro: &[C64], to: C64, which: Which) -> usize {
    let wanted = |r: &C64| match which {
        Which::Any => true,
        Which::Real => is_real(r),
        Which::Complex => !is_real(r),
    };
    let mut best: Option<(usize, f64)> = None;
    for (i, r) in fro.iter().enumerate() {
        if !wanted(r) {
            continue;
        }
        let d = (r - to).norm();
        if best.is_none_or(|(_, bd)| d < bd) {
            best = Some((i, d));
        }
    }
    // SciPy would raise if no root of the requested kind exists; fall back to
    // the nearest root of any kind (cannot happen for valid designs).
    best.map_or_else(|| nearest_idx(fro, to, Which::Any), |(i, _)| i)
}

#[derive(Clone, Copy)]
enum Which {
    Real,
    Complex,
    Any,
}

/// Coefficients (highest degree first, real part) of the monic polynomial
/// with the given (at most two) roots, as `numpy.poly`; returns the
/// coefficient array and its length.
fn poly2(roots: &[C64]) -> ([f64; 3], usize) {
    match roots {
        [] => ([1.0, 0.0, 0.0], 1),
        [r] => ([1.0, -r.re, 0.0], 2),
        [r1, r2] => ([1.0, (-*r1 - *r2).re, (*r1 * *r2).re], 3),
        _ => unreachable!("second-order sections have at most two roots"),
    }
}

fn single_zpksos(z: &[C64], p: &[C64]) -> Sos {
    let (b, nb) = poly2(z);
    let (a, na) = poly2(p);
    let mut sos = [0.0; 6];
    sos[3 - nb..3].copy_from_slice(&b[..nb]);
    sos[6 - na..6].copy_from_slice(&a[..na]);
    sos
}

/// `scipy.signal.zpk2sos(z, p, k, pairing='nearest')` for digital filters.
pub(crate) fn zpk2sos(zpk: &Zpk) -> AudioSampleResult<Vec<Sos>> {
    let mut z = zpk.zeros.clone();
    let mut p = zpk.poles.clone();
    if z.is_empty() && p.is_empty() {
        return Ok(vec![[zpk.gain, 0.0, 0.0, 1.0, 0.0, 0.0]]);
    }
    let zero = C64::new(0.0, 0.0);
    if z.len() < p.len() {
        z.extend(std::iter::repeat_n(zero, p.len() - z.len()));
    }
    if p.len() < z.len() {
        p.extend(std::iter::repeat_n(zero, z.len() - p.len()));
    }
    let n_sections = p.len().div_ceil(2);
    if p.len() % 2 == 1 {
        p.push(zero);
        z.push(zero);
    }

    let to_vec = |(c, r): (Vec<C64>, Vec<f64>)| -> Vec<C64> {
        let mut v = c;
        v.extend(r.into_iter().map(|x| C64::new(x, 0.0)));
        v
    };
    let mut z = to_vec(cplxreal(&z)?);
    let mut p = to_vec(cplxreal(&p)?);

    // Index of the root closest to the unit circle (optionally real only);
    // ties resolve to the lowest index, like `np.argmin`.
    let idx_worst = |v: &[C64], real_only: bool| -> usize {
        let mut best = usize::MAX;
        let mut best_val = f64::INFINITY;
        for (i, r) in v.iter().enumerate() {
            if real_only && !is_real(r) {
                continue;
            }
            let val = (1.0 - r.norm()).abs();
            if val < best_val {
                best = i;
                best_val = val;
            }
        }
        best
    };

    let mut sos = vec![[0.0f64; 6]; n_sections];
    for si in (0..n_sections).rev() {
        let p1_idx = idx_worst(&p, false);
        let p1 = p.remove(p1_idx);
        let n_real_p = p.iter().filter(|r| is_real(r)).count();
        let n_real_z = z.iter().filter(|r| is_real(r)).count();

        if is_real(&p1) && n_real_p == 0 {
            // Special case (1): last remaining real pole.
            let z1_idx = nearest_idx(&z, p1, Which::Real);
            let z1 = z.remove(z1_idx);
            sos[si] = single_zpksos(&[z1, zero], &[p1, zero]);
        } else if p.len() + 1 == z.len() && !is_real(&p1) && n_real_p == 1 && n_real_z == 1 {
            // Special case (2): one real pole and one real zero left; must
            // pair with a complex zero.
            let z1_idx = nearest_idx(&z, p1, Which::Complex);
            let z1 = z.remove(z1_idx);
            sos[si] = single_zpksos(&[z1, z1.conj()], &[p1, p1.conj()]);
        } else {
            let p2 = if is_real(&p1) {
                let p2_idx = idx_worst(&p, true);
                p.remove(p2_idx)
            } else {
                p1.conj()
            };
            if z.is_empty() {
                sos[si] = single_zpksos(&[], &[p1, p2]);
            } else {
                let z1_idx = nearest_idx(&z, p1, Which::Any);
                let z1 = z.remove(z1_idx);
                if !is_real(&z1) {
                    sos[si] = single_zpksos(&[z1, z1.conj()], &[p1, p2]);
                } else if !z.is_empty() {
                    let z2_idx = nearest_idx(&z, p1, Which::Real);
                    let z2 = z.remove(z2_idx);
                    sos[si] = single_zpksos(&[z1, z2], &[p1, p2]);
                } else {
                    sos[si] = single_zpksos(&[z1], &[p1, p2]);
                }
            }
        }
    }
    for c in &mut sos[0][..3] {
        *c *= zpk.gain;
    }
    Ok(sos)
}

// ============================================================================
// Top-level design (SciPy `iirfilter(..., output='sos', fs=fs)`)
// ============================================================================

/// Critical frequencies, in Hz.
#[derive(Debug, Clone, Copy)]
pub(crate) enum Wn {
    Single(f64),
    Band(f64, f64),
}

/// Design a digital IIR filter as SOS, exactly like
/// `scipy.signal.iirfilter(N, Wn, btype, ftype, output='sos', fs=fs)`.
pub(crate) fn iirfilter_sos(
    proto: &Zpk,
    response: FilterResponse,
    wn: Wn,
    sample_rate: f64,
) -> AudioSampleResult<Vec<Sos>> {
    // SciPy normalises to Nyquist = 1 and then designs with fs = 2.
    const FS: f64 = 2.0;
    let warp = |f_hz: f64| {
        let w = 2.0 * f_hz / sample_rate;
        2.0 * FS * (PI * w / FS).tan()
    };
    let analog = match (response, wn) {
        (FilterResponse::LowPass, Wn::Single(f)) => lp2lp_zpk(proto, warp(f)),
        (FilterResponse::HighPass, Wn::Single(f)) => lp2hp_zpk(proto, warp(f)),
        (FilterResponse::BandPass, Wn::Band(lo, hi)) => {
            let (w0, w1) = (warp(lo), warp(hi));
            lp2bp_zpk(proto, (w0 * w1).sqrt(), w1 - w0)
        }
        (FilterResponse::BandStop, Wn::Band(lo, hi)) => {
            let (w0, w1) = (warp(lo), warp(hi));
            lp2bs_zpk(proto, (w0 * w1).sqrt(), w1 - w0)
        }
        _ => {
            return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
                "response",
                "critical frequencies do not match the response type",
            )));
        }
    };
    zpk2sos(&bilinear_zpk(&analog, FS))
}

fn param_err(name: &str, msg: impl Into<String>) -> AudioSampleError {
    AudioSampleError::Parameter(ParameterError::invalid_value(name, msg.into()))
}

/// Validate an [`IirFilterDesign`] against `sample_rate` and design it.
pub(crate) fn design_sos(
    design: &IirFilterDesign,
    sample_rate: f64,
) -> AudioSampleResult<Vec<Sos>> {
    if !(sample_rate.is_finite() && sample_rate > 0.0) {
        return Err(param_err(
            "sample_rate",
            "sample rate must be positive and finite",
        ));
    }
    let nyquist = sample_rate / 2.0;
    let order = design.order.get();
    if order > MAX_IIR_ORDER {
        return Err(param_err(
            "order",
            format!("filter order must be ≤ {MAX_IIR_ORDER} (got {order})"),
        ));
    }

    let in_range = |f: f64| f.is_finite() && f > 0.0 && f < nyquist;
    let wn = match design.response {
        FilterResponse::LowPass | FilterResponse::HighPass => {
            let f = design.cutoff_frequency.ok_or_else(|| {
                param_err(
                    "cutoff_frequency",
                    "cutoff frequency required for low-pass/high-pass filters",
                )
            })?;
            if !in_range(f) {
                return Err(param_err(
                    "cutoff_frequency",
                    format!("cutoff frequency must be in (0, {nyquist}) Hz (Nyquist), got {f}"),
                ));
            }
            Wn::Single(f)
        }
        FilterResponse::BandPass | FilterResponse::BandStop => {
            let lo = design.low_frequency.ok_or_else(|| {
                param_err("low_frequency", "low frequency required for band filters")
            })?;
            let hi = design.high_frequency.ok_or_else(|| {
                param_err("high_frequency", "high frequency required for band filters")
            })?;
            if !in_range(lo) || !in_range(hi) || lo >= hi {
                return Err(param_err(
                    "frequency_range",
                    format!(
                        "band edges must satisfy 0 < low < high < {nyquist} Hz (Nyquist), got [{lo}, {hi}]"
                    ),
                ));
            }
            Wn::Band(lo, hi)
        }
    };

    let positive = |name: &str, v: Option<f64>, what: &str| -> AudioSampleResult<f64> {
        let v =
            v.ok_or_else(|| param_err(name, format!("{what} required for this filter type")))?;
        if !(v.is_finite() && v > 0.0) {
            return Err(param_err(
                name,
                format!("{what} must be positive and finite, got {v}"),
            ));
        }
        Ok(v)
    };

    let proto = match design.filter_type {
        IirFilterType::Butterworth => buttap(order),
        IirFilterType::ChebyshevI => cheb1ap(
            order,
            positive(
                "passband_ripple",
                design.passband_ripple,
                "passband ripple (dB)",
            )?,
        ),
        IirFilterType::ChebyshevII => cheb2ap(
            order,
            positive(
                "stopband_attenuation",
                design.stopband_attenuation,
                "stopband attenuation (dB)",
            )?,
        ),
        IirFilterType::Bessel => besselap(order, design.bessel_norm),
        IirFilterType::Elliptic => {
            let rp = positive(
                "passband_ripple",
                design.passband_ripple,
                "passband ripple (dB)",
            )?;
            let rs = positive(
                "stopband_attenuation",
                design.stopband_attenuation,
                "stopband attenuation (dB)",
            )?;
            if rs <= rp {
                return Err(param_err(
                    "stopband_attenuation",
                    format!(
                        "stopband attenuation ({rs} dB) must exceed the passband ripple ({rp} dB)"
                    ),
                ));
            }
            ellipap(order, rp, rs)?
        }
        #[allow(unreachable_patterns)]
        other => {
            return Err(AudioSampleError::unsupported(
                "design_iir_filter",
                format!("filter type {other:?} is not implemented"),
            ));
        }
    };

    iirfilter_sos(&proto, design.response, wn, sample_rate)
}

// ============================================================================
// Analysis helpers
// ============================================================================

/// Complex frequency response of an SOS cascade at `freqs_hz`
/// (SciPy `sosfreqz(sos, worN=freqs, fs=fs)`).
pub(crate) fn sosfreqz(sos: &[Sos], freqs_hz: &[f64], sample_rate: f64) -> Vec<C64> {
    freqs_hz
        .iter()
        .map(|&f| {
            let w = 2.0 * PI * f / sample_rate;
            let zi1 = C64::new(w.cos(), -w.sin());
            let zi2 = zi1 * zi1;
            sos.iter().fold(C64::new(1.0, 0.0), |acc, s| {
                let num = s[0] + zi1 * s[1] + zi2 * s[2];
                let den = s[3] + zi1 * s[4] + zi2 * s[5];
                acc * num / den
            })
        })
        .collect()
}

/// Steady-state initial conditions for a unit step, per section, for the
/// transposed direct form II (SciPy `sosfilt_zi`). Coefficients must be
/// normalised so that `a0 == 1`.
pub(crate) fn sosfilt_zi(sos: &[Sos]) -> Vec<[f64; 2]> {
    let mut scale = 1.0;
    sos.iter()
        .map(|s| {
            let (b0, b1, b2, a1, a2) = (
                s[0] / s[3],
                s[1] / s[3],
                s[2] / s[3],
                s[4] / s[3],
                s[5] / s[3],
            );
            // lfilter_zi for a biquad: solve (I - A^T) zi = B with
            // A the companion matrix of a, B = b[1:] - a[1:] * b0.
            //   zi0 = (B0 + B1 - ... ) closed form below.
            let bb1 = b1 - a1 * b0;
            let bb2 = b2 - a2 * b0;
            let det = 1.0 + a1 + a2;
            let zi0 = (bb1 + bb2) / det;
            let zi1 = bb2 - a2 * zi0;
            let zi = [scale * zi0, scale * zi1];
            scale *= (b0 + b1 + b2) / (1.0 + a1 + a2);
            zi
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn butter4_lowpass_matches_scipy_coefficients() {
        // scipy.signal.butter(4, 1000, fs=48000, output='sos')
        let expected: [Sos; 2] = [
            [
                1.555_172_178_089_176e-5,
                3.110_344_356_178_352e-5,
                1.555_172_178_089_176e-5,
                1.0,
                -1.769_504_348_512_836_8,
                0.784_773_331_782_562_9,
            ],
            [
                1.0,
                2.0,
                1.0,
                1.0,
                -1.888_555_953_889_046_4,
                0.904_852_228_768_567_7,
            ],
        ];
        let sos = iirfilter_sos(
            &buttap(4),
            FilterResponse::LowPass,
            Wn::Single(1000.0),
            48000.0,
        )
        .unwrap();
        for (a, b) in sos.iter().zip(expected.iter()) {
            for (x, y) in a.iter().zip(b.iter()) {
                assert!((x - y).abs() <= 1e-12 * y.abs().max(1.0), "{a:?} vs {b:?}");
            }
        }
    }

    #[test]
    fn sosfilt_zi_gives_constant_step_response() {
        let sos = iirfilter_sos(
            &cheb1ap(5, 1.0),
            FilterResponse::LowPass,
            Wn::Single(3000.0),
            44100.0,
        )
        .unwrap();
        let zi = sosfilt_zi(&sos);
        // Filtering a constant 1.0 from these initial conditions must yield the
        // DC gain immediately with no transient.
        let dc = sosfreqz(&sos, &[0.0], 44100.0)[0].re;
        let mut state = zi;
        for _ in 0..10 {
            let mut x = 1.0;
            for (s, st) in sos.iter().zip(state.iter_mut()) {
                let y = s[0] * x + st[0];
                st[0] = s[1] * x - s[4] * y + st[1];
                st[1] = s[2] * x - s[5] * y;
                x = y;
            }
            assert!((x - dc).abs() < 1e-12, "{x} vs {dc}");
        }
    }
}
