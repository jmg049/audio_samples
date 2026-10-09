//! Small numerical helpers shared by the explainers.
//!
//! These exist so that explanations can *measure* properties (window
//! side-lobe levels, pole radii, overlap-add ripple) instead of quoting
//! textbook values. They are deliberately simple and dependency-free: a
//! radix-2 FFT, a quadratic root solver and a few loops.

use std::f64::consts::PI;

use num_complex::Complex;

type C64 = Complex<f64>;

#[cfg(feature = "transforms")]
/// In-place iterative radix-2 FFT (decimation in time).
///
/// `buf.len()` must be a power of two.
fn fft_in_place(buf: &mut [C64]) {
    let n = buf.len();
    debug_assert!(n.is_power_of_two());
    // Bit-reversal permutation.
    let mut j = 0usize;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j |= bit;
        if i < j {
            buf.swap(i, j);
        }
    }
    // Butterflies.
    let mut len = 2;
    while len <= n {
        let ang = -2.0 * PI / len as f64;
        let w_len = C64::new(ang.cos(), ang.sin());
        for start in (0..n).step_by(len) {
            let mut w = C64::new(1.0, 0.0);
            for k in 0..len / 2 {
                let u = buf[start + k];
                let v = buf[start + k + len / 2] * w;
                buf[start + k] = u + v;
                buf[start + k + len / 2] = u - v;
                w *= w_len;
            }
        }
        len <<= 1;
    }
}

#[cfg(feature = "transforms")]
/// Magnitude of the zero-padded DFT of `x` (bins `0..=pad/2`).
fn padded_magnitude(x: &[f64], pad: usize) -> Vec<f64> {
    let mut buf = vec![C64::new(0.0, 0.0); pad];
    for (b, &v) in buf.iter_mut().zip(x) {
        b.re = v;
    }
    fft_in_place(&mut buf);
    buf[..=pad / 2].iter().map(|c| c.norm()).collect()
}

/// |DTFT| of `x` at a normalised frequency `cycles_per_sample`.
pub(crate) fn dtft_magnitude(x: &[f64], cycles_per_sample: f64) -> f64 {
    let w = 2.0 * PI * cycles_per_sample;
    x.iter()
        .enumerate()
        .fold(C64::new(0.0, 0.0), |acc, (n, &v)| {
            acc + C64::from_polar(v, -w * n as f64)
        })
        .norm()
}

#[cfg(feature = "transforms")]
/// Frequency-domain figures of merit of a window, all measured from its
/// coefficients.
#[derive(Debug, Clone, Copy)]
pub(crate) struct WindowMetrics {
    /// Window length `N`.
    pub n: usize,
    /// Coherent gain `Σw / N`.
    pub coherent_gain: f64,
    /// Equivalent noise bandwidth in bins, `N Σw² / (Σw)²`.
    pub enbw_bins: f64,
    /// Loss for a tone half-way between bins, in dB (positive number).
    pub scalloping_loss_db: f64,
    /// Highest side lobe relative to the main-lobe peak, in dB (negative),
    /// if the spectrum has a first null.
    pub peak_side_lobe_db: Option<f64>,
    /// Full main-lobe width at -3.01 dB (half power), in bins.
    pub width_3db_bins: Option<f64>,
    /// Full main-lobe width at -6.02 dB (half amplitude), in bins.
    pub width_6db_bins: Option<f64>,
    /// Distance from the peak to the first null, in bins.
    pub first_null_bins: Option<f64>,
    /// `w[n] == w[N-1-n]` for all `n`.
    pub symmetric: bool,
    /// `w[n] == w[N-n]` for `n = 1..N-1` (DFT-even) and not symmetric.
    pub periodic: bool,
}

#[cfg(feature = "transforms")]
/// Measures [`WindowMetrics`] for `w` using a zero-padded FFT (at least 32
/// points per bin) and parabolic refinement of the side-lobe peak.
pub(crate) fn window_metrics(w: &[f64]) -> WindowMetrics {
    let n = w.len().max(1);
    let sum: f64 = w.iter().sum();
    let sum_sq: f64 = w.iter().map(|v| v * v).sum();
    let coherent_gain = sum / n as f64;
    let enbw_bins = n as f64 * sum_sq / (sum * sum);

    let peak = sum.abs();
    let half_bin = dtft_magnitude(w, 0.5 / n as f64);
    let scalloping_loss_db = -20.0 * (half_bin / peak).log10();

    let tol = 1e-12 * w.iter().fold(0.0f64, |m, v| m.max(v.abs())).max(1e-300);
    let symmetric = (0..n / 2).all(|i| (w[i] - w[n - 1 - i]).abs() <= tol);
    let periodic = !symmetric && n > 2 && (1..n / 2).all(|i| (w[i] - w[n - i]).abs() <= tol);

    // Zero-padded spectrum: `ppb` points per DFT bin.
    let pad = (32 * n).next_power_of_two().clamp(1024, 1 << 21);
    let ppb = pad as f64 / n as f64;
    let mag = padded_magnitude(w, pad);
    let db: Vec<f64> = mag
        .iter()
        .map(|m| 20.0 * (m / peak).max(1e-300).log10())
        .collect();

    // Crossing of a level (first index where db < level), linearly interpolated.
    let crossing = |level: f64| -> Option<f64> {
        let k = db.iter().position(|&d| d < level)?;
        if k == 0 {
            return None;
        }
        let (d0, d1) = (db[k - 1], db[k]);
        let frac = (d0 - level) / (d0 - d1);
        Some(((k - 1) as f64 + frac) / ppb)
    };
    let half_power = -10.0 * 2f64.log10();
    let half_amp = -20.0 * 2f64.log10();
    let width_3db_bins = crossing(half_power).map(|x| 2.0 * x);
    let width_6db_bins = crossing(half_amp).map(|x| 2.0 * x);

    // First local minimum of the magnitude after the main lobe.
    let first_null = (1..mag.len() - 1).find(|&k| mag[k] <= mag[k - 1] && mag[k] <= mag[k + 1]);
    let first_null_bins = first_null.map(|k| k as f64 / ppb);
    let peak_side_lobe_db = first_null.and_then(|k0| {
        let (k_max, _) = db
            .iter()
            .enumerate()
            .skip(k0)
            .max_by(|a, b| a.1.total_cmp(b.1))?;
        if k_max == 0 || k_max + 1 >= db.len() {
            return Some(db[k_max]);
        }
        // Parabolic interpolation of the peak in dB.
        let (a, b, c) = (db[k_max - 1], db[k_max], db[k_max + 1]);
        let denom = a - 2.0 * b + c;
        Some(if denom.abs() > 0.0 {
            b - 0.125 * (a - c) * (a - c) / denom
        } else {
            b
        })
    });

    WindowMetrics {
        n,
        coherent_gain,
        enbw_bins,
        scalloping_loss_db,
        peak_side_lobe_db,
        width_3db_bins,
        width_6db_bins,
        first_null_bins,
        symmetric,
        periodic,
    }
}

#[cfg(feature = "transforms")]
/// Result of an overlap-add check for a window and hop.
#[derive(Debug, Clone, Copy)]
pub(crate) struct OverlapCheck {
    /// Mean of `Σ_m w[n - mH]` in steady state.
    pub cola_mean: f64,
    /// `(max - min) / mean` of `Σ_m w[n - mH]` in steady state.
    pub cola_ripple: f64,
    /// Minimum of `Σ_m w²[n - mH]` in steady state.
    pub nola_min: f64,
}

#[cfg(feature = "transforms")]
/// Evaluates the constant-overlap-add (COLA) and nonzero-overlap-add (NOLA)
/// sums of `w` at hop `hop` over one hop period in steady state.
pub(crate) fn overlap_check(w: &[f64], hop: usize) -> OverlapCheck {
    let n = w.len();
    let hop = hop.max(1);
    let mut sums = Vec::with_capacity(hop);
    let mut sq_sums = Vec::with_capacity(hop);
    // Positions n0..n0+hop, far enough from the start that every frame that
    // can cover them exists.
    let n0 = n.div_ceil(hop) * hop + n;
    for pos in n0..n0 + hop {
        let mut s = 0.0;
        let mut s2 = 0.0;
        // Frames start at m*hop; frame covers [m*hop, m*hop + n).
        let m_hi = pos / hop;
        let m_lo = (pos + 1).saturating_sub(n).div_ceil(hop);
        for m in m_lo..=m_hi {
            let idx = pos - m * hop;
            if idx < n {
                s += w[idx];
                s2 += w[idx] * w[idx];
            }
        }
        sums.push(s);
        sq_sums.push(s2);
    }
    let mean = sums.iter().sum::<f64>() / sums.len() as f64;
    let (lo, hi) = sums
        .iter()
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), &v| {
            (lo.min(v), hi.max(v))
        });
    let nola_min = sq_sums.iter().copied().fold(f64::INFINITY, f64::min);
    OverlapCheck {
        cola_mean: mean,
        cola_ripple: if mean.abs() > 0.0 {
            (hi - lo) / mean.abs()
        } else {
            f64::INFINITY
        },
        nola_min,
    }
}

/// Roots of `c0·z² + c1·z + c2 = 0`, i.e. the zeros (or poles) of one
/// second-order section `c0 + c1 z⁻¹ + c2 z⁻²`. Degenerate sections
/// (`c2 == 0`, a first-order section stored in SOS form) return a single
/// root; the cancelling root at the origin is omitted.
pub(crate) fn section_roots(c0: f64, c1: f64, c2: f64) -> Vec<C64> {
    if c2 == 0.0 {
        if c0 == 0.0 || c1 == 0.0 {
            return if c0 == 0.0 {
                Vec::new()
            } else {
                vec![C64::new(0.0, 0.0)]
            };
        }
        return vec![C64::new(-c1 / c0, 0.0)];
    }
    if c0 == 0.0 {
        return vec![C64::new(-c2 / c1, 0.0)];
    }
    let disc = c1 * c1 - 4.0 * c0 * c2;
    let two_a = 2.0 * c0;
    if disc >= 0.0 {
        // Numerically stable form: avoid subtracting nearly equal numbers.
        let sign = if c1 >= 0.0 { 1.0 } else { -1.0 };
        let q = -0.5 * (c1 + sign * disc.sqrt());
        vec![C64::new(q / c0, 0.0), C64::new(c2 / q, 0.0)]
    } else {
        let re = -c1 / two_a;
        let im = (-disc).sqrt() / two_a;
        vec![C64::new(re, im.abs()), C64::new(re, -im.abs())]
    }
}

/// Formats a complex number compactly, e.g. `0.8123 ± 0.2201j` style parts.
pub(crate) fn fmt_complex(z: C64) -> String {
    use super::doc::fmt_num;
    if z.im.abs() < 1e-12 {
        fmt_num(z.re)
    } else if z.im > 0.0 {
        format!("{} + {}j", fmt_num(z.re), fmt_num(z.im))
    } else {
        format!("{} - {}j", fmt_num(z.re), fmt_num(-z.im))
    }
}

/// Converts a linear magnitude to dB, mapping 0 to -inf (callers decide how
/// to display it).
pub(crate) fn mag_db(m: f64) -> f64 {
    20.0 * m.log10()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(feature = "transforms")]
    fn hann_symmetric(n: usize) -> Vec<f64> {
        (0..n)
            .map(|i| 0.5 - 0.5 * (2.0 * PI * i as f64 / (n - 1) as f64).cos())
            .collect()
    }

    #[cfg(feature = "transforms")]
    #[test]
    fn fft_matches_direct_dft() {
        let x: Vec<f64> = (0..16).map(|i| ((i * 7) % 5) as f64 - 2.0).collect();
        let fast = padded_magnitude(&x, 16);
        for (k, m) in fast.iter().enumerate() {
            let direct = dtft_magnitude(&x, k as f64 / 16.0);
            assert!((m - direct).abs() < 1e-9, "bin {k}: {m} vs {direct}");
        }
    }

    #[cfg(feature = "transforms")]
    #[test]
    fn rectangular_window_metrics() {
        let m = window_metrics(&[1.0; 256]);
        assert!((m.coherent_gain - 1.0).abs() < 1e-12);
        assert!((m.enbw_bins - 1.0).abs() < 1e-12);
        // Textbook: -13.26 dB peak side lobe, 3.92 dB scalloping loss,
        // 0.89-bin 3 dB width, 1.21-bin 6 dB width, first null at 1 bin.
        assert!((m.peak_side_lobe_db.unwrap() + 13.26).abs() < 0.05, "{m:?}");
        assert!((m.scalloping_loss_db - 3.92).abs() < 0.01);
        assert!((m.width_3db_bins.unwrap() - 0.886).abs() < 0.01);
        assert!((m.width_6db_bins.unwrap() - 1.207).abs() < 0.01);
        assert!((m.first_null_bins.unwrap() - 1.0).abs() < 0.05);
        assert!(m.symmetric);
    }

    #[cfg(feature = "transforms")]
    #[test]
    fn hann_window_metrics() {
        let n = 1024;
        let m = window_metrics(&hann_symmetric(n));
        // Symmetric Hann: CG = (N-1)/(2N), ENBW = 1.5 N/(N-1) exactly.
        let nf = n as f64;
        assert!((m.coherent_gain - (nf - 1.0) / (2.0 * nf)).abs() < 1e-12);
        assert!((m.enbw_bins - 1.5 * nf / (nf - 1.0)).abs() < 1e-9);
        assert!((m.peak_side_lobe_db.unwrap() + 31.47).abs() < 0.1, "{m:?}");
        assert!((m.scalloping_loss_db - 1.42).abs() < 0.01);
        assert!((m.width_3db_bins.unwrap() - 1.44).abs() < 0.01);
        assert!((m.width_6db_bins.unwrap() - 2.00).abs() < 0.01);
        assert!(m.symmetric && !m.periodic);
    }

    #[cfg(feature = "transforms")]
    #[test]
    fn periodic_hann_is_detected() {
        let n = 64;
        let w: Vec<f64> = (0..n)
            .map(|i| 0.5 - 0.5 * (2.0 * PI * i as f64 / n as f64).cos())
            .collect();
        let m = window_metrics(&w);
        assert!(m.periodic && !m.symmetric);
    }

    #[cfg(feature = "transforms")]
    #[test]
    fn periodic_hann_is_cola_at_half_overlap() {
        let n = 512;
        let w: Vec<f64> = (0..n)
            .map(|i| 0.5 - 0.5 * (2.0 * PI * i as f64 / n as f64).cos())
            .collect();
        let c = overlap_check(&w, n / 2);
        assert!(c.cola_ripple < 1e-12);
        assert!((c.cola_mean - 1.0).abs() < 1e-12);
        // A rectangular window at hop = N tiles the time axis exactly.
        let tiled = overlap_check(&[1.0; 8], 8);
        assert!(tiled.cola_ripple < 1e-12 && (tiled.nola_min - 1.0).abs() < 1e-12);
    }

    #[test]
    fn section_roots_cover_all_cases() {
        // (z - 0.5)(z - 0.25) = z² - 0.75 z + 0.125
        let mut r = section_roots(1.0, -0.75, 0.125);
        r.sort_by(|a, b| a.re.total_cmp(&b.re));
        assert!((r[0].re - 0.25).abs() < 1e-12 && (r[1].re - 0.5).abs() < 1e-12);
        // Complex pair with radius 0.9.
        let rad: f64 = 0.9;
        let th: f64 = 0.3;
        let r = section_roots(1.0, -2.0 * rad * th.cos(), rad * rad);
        assert!(r.iter().all(|z| (z.norm() - 0.9).abs() < 1e-12));
        // First-order section.
        let r = section_roots(1.0, -0.5, 0.0);
        assert_eq!(r.len(), 1);
        assert!((r[0].re - 0.5).abs() < 1e-12);
    }
}
