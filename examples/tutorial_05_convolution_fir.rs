//! Tutorial 05: Convolution and FIR filters.
//!
//! Accompanies the guide chapter `audio_samples::guide::c05_convolution_fir`
//! (`documentation/guide/05_convolution_fir.md`).
//!
//! What it teaches:
//! 1. the convolution sum, and the two lengths the crate returns: the full
//!    convolution (`convolve`, N + M − 1) and the valid part (`apply_filter`,
//!    N − M + 1);
//! 2. the impulse response;
//! 3. the moving-average frequency response: closed-form Dirichlet kernel
//!    versus gains measured by filtering sines;
//! 4. linear phase for symmetric taps and the (M − 1)/2 group delay,
//!    measured with a delayed impulse and a pulse centroid;
//! 5. a windowed-sinc low-pass designed from scratch (the crate has no FIR
//!    designer) and verified.
//!
//! Every number printed is also checked with `assert!`.
//!
//! Run with:
//!
//! ```text
//! cargo run --release --example tutorial_05_convolution_fir --features educational,full_no_plotting
//! ```

use std::f64::consts::PI;
use std::num::NonZeroUsize;

use audio_samples::educational::topics;
use audio_samples::{AudioProcessing, AudioSamples, AudioTransforms};
use ndarray::Array1;
use non_empty_slice::NonEmptySlice;
use num_complex::Complex;
use spectrograms::{WindowType, make_window};

fn mono(v: Vec<f64>, fs: u32) -> AudioSamples<'static, f64> {
    let sr = std::num::NonZeroU32::new(fs).expect("fs > 0");
    AudioSamples::new_mono(Array1::from(v), sr).expect("non-empty")
}

/// Full convolution y[n] = Σ_j h[j] x[n − j], n = 0 … N + M − 2.
fn convolve_full(x: &[f64], h: &[f64]) -> Vec<f64> {
    let mut y = vec![0.0; x.len() + h.len() - 1];
    for (n, yn) in y.iter_mut().enumerate() {
        for (j, &hj) in h.iter().enumerate() {
            if n >= j && n - j < x.len() {
                *yn += hj * x[n - j];
            }
        }
    }
    y
}

/// H(e^{jω}) = Σ h[j] e^{−jωj}.
fn dtft(h: &[f64], omega: f64) -> Complex<f64> {
    h.iter()
        .enumerate()
        .map(|(j, &c)| Complex::from_polar(c, -omega * j as f64))
        .sum()
}

/// Group delay τ(ω) = Re{ Σ j h[j] e^{−jωj} / H(e^{jω}) }.
fn group_delay(h: &[f64], omega: f64) -> f64 {
    let num: Complex<f64> = h
        .iter()
        .enumerate()
        .map(|(j, &c)| Complex::from_polar(j as f64 * c, -omega * j as f64))
        .sum();
    (num / dtft(h, omega)).re
}

/// Steady-state gain of `h` at `f` Hz: filter a sine with `apply_filter` and
/// project the valid output (exactly `fs` samples = 1 s) on sin and cos.
fn measured_gain(h: &[f64], f: f64, fs: u32) -> f64 {
    let m = h.len();
    let len = fs as usize + m - 1;
    let x: Vec<f64> = (0..len)
        .map(|n| (2.0 * PI * f * n as f64 / f64::from(fs)).sin())
        .collect();
    let y = mono(x, fs)
        .apply_filter(NonEmptySlice::new(h).expect("non-empty"))
        .expect("filter");
    let ys = y.as_slice().expect("contiguous");
    assert_eq!(ys.len(), fs as usize);
    let (mut s, mut c) = (0.0, 0.0);
    for (i, &v) in ys.iter().enumerate() {
        let phi = 2.0 * PI * f * i as f64 / f64::from(fs);
        s += v * phi.sin();
        c += v * phi.cos();
    }
    2.0 / ys.len() as f64 * (s * s + c * c).sqrt()
}

/// Windowed-sinc low-pass: ideal impulse response 2 f_c/f_s · sinc(2 f_c/f_s (n − (M−1)/2)),
/// times a Hamming window, normalised to unit DC gain.
fn windowed_sinc(m: usize, fc: f64, fs: f64) -> Vec<f64> {
    let w = make_window::<f64>(WindowType::Hamming, NonZeroUsize::new(m).expect("m > 0"));
    let wc = 2.0 * fc / fs; // cutoff as a fraction of Nyquist
    let centre = (m - 1) as f64 / 2.0;
    let mut h: Vec<f64> = (0..m)
        .map(|n| {
            let t = n as f64 - centre;
            let sinc = if t == 0.0 {
                1.0
            } else {
                (PI * wc * t).sin() / (PI * wc * t)
            };
            wc * sinc * w[n]
        })
        .collect();
    let dc: f64 = h.iter().sum();
    h.iter_mut().for_each(|v| *v /= dc);
    h
}

fn db(x: f64) -> f64 {
    20.0 * x.log10()
}

fn main() {
    println!("=== Tutorial 05: convolution and FIR filters ===\n");

    // ------------------------------------------------------------------
    // 1. Full versus valid convolution.
    // ------------------------------------------------------------------
    println!("1. Lengths: N = 1000 input samples, M = 31 taps");
    let x: Vec<f64> = (0..1000)
        .map(|n| (0.05 * n as f64).sin() + 0.3 * (0.71 * n as f64).cos())
        .collect();
    let h = windowed_sinc(31, 2000.0, 16_000.0);
    let full_lib = mono(x.clone(), 16_000)
        .convolve(&mono(h.clone(), 16_000))
        .expect("convolve");
    let valid_lib = mono(x.clone(), 16_000)
        .apply_filter(NonEmptySlice::new(&h).expect("non-empty"))
        .expect("filter");
    let full = convolve_full(&x, &h);
    let fl = full_lib.as_slice().expect("contiguous");
    let vl = valid_lib.as_slice().expect("contiguous");
    println!(
        "   convolve: {} samples (N + M - 1 = 1030); apply_filter: {} samples (N - M + 1 = 970)",
        fl.len(),
        vl.len()
    );
    assert_eq!(fl.len(), 1030);
    assert_eq!(vl.len(), 970);
    let err_full = full
        .iter()
        .zip(fl)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f64::max);
    // The valid part is full[M-1 .. N].
    let err_valid = full[30..1000]
        .iter()
        .zip(vl)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f64::max);
    println!("   direct sum vs convolve (FFT): {err_full:.1e}; vs apply_filter: {err_valid:.1e}");
    // FFT convolution rounds differently from the direct sum: ~1e-15 per term.
    assert!(err_full < 1e-12);
    assert!(err_valid < 1e-12);

    // ------------------------------------------------------------------
    // 2. Impulse response.
    // ------------------------------------------------------------------
    let mut delta = vec![0.0; 64];
    delta[0] = 1.0;
    let ir = mono(delta, 16_000)
        .convolve(&mono(h.clone(), 16_000))
        .expect("convolve");
    let ir_err = ir.as_slice().expect("contiguous")[..31]
        .iter()
        .zip(&h)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f64::max);
    println!("\n2. Impulse in, taps out: max |y[n] - h[n]| = {ir_err:.1e}");
    assert!(ir_err < 1e-12);

    // ------------------------------------------------------------------
    // 3. Moving average: Dirichlet kernel.
    // ------------------------------------------------------------------
    println!("\n3. 8-tap moving average at 8 kHz: |H(f)| = |sin(pi f M/fs) / (M sin(pi f/fs))|");
    let ma = vec![1.0 / 8.0; 8];
    let fs_ma = 8000u32;
    for f in [250.0, 500.0, 750.0, 1000.0, 1500.0, 2000.0, 3000.0] {
        let x = PI * f / f64::from(fs_ma);
        let closed = ((8.0 * x).sin() / (8.0 * x.sin())).abs();
        let measured = measured_gain(&ma, f, fs_ma);
        println!("   {f:>6} Hz: closed form {closed:.6}, measured {measured:.6}");
        // Integer number of cycles in exactly 1 s: projection is exact up to rounding.
        assert!((closed - measured).abs() < 1e-9);
        if f % 1000.0 == 0.0 {
            assert!(measured < 1e-9, "null at multiples of fs/M = 1000 Hz");
        }
    }

    // ------------------------------------------------------------------
    // 4. Linear phase and group delay.
    // ------------------------------------------------------------------
    println!("\n4. Linear phase and group delay");
    let lp = windowed_sinc(63, 2000.0, 16_000.0);
    for f in [100.0, 500.0, 1000.0, 1500.0] {
        let omega = 2.0 * PI * f / 16_000.0;
        let tau = group_delay(&lp, omega);
        assert!((tau - 31.0).abs() < 1e-9);
    }
    println!("   symmetric 63 taps: group delay 31 samples at 100, 500, 1000, 1500 Hz");
    let skew = [0.5, 0.3, 0.2];
    let (gd_dc, gd_ny) = (group_delay(&skew, 0.0), group_delay(&skew, PI));
    println!(
        "   taps [0.5, 0.3, 0.2]: group delay {gd_dc:.2} samples at DC, {gd_ny:.2} at Nyquist"
    );
    assert!((gd_dc - 0.7).abs() < 1e-12);
    assert!((gd_ny - 0.25).abs() < 1e-12);
    let skew_doc = topics::fir(&skew, 16_000.0);
    assert!(skew_doc.quantity("Group delay").is_none());
    assert!((skew_doc.quantity("Group delay at DC").expect("q").value - 0.7).abs() < 1e-12);

    // Delayed impulse at sample 100 -> output peak at 100 + 31.
    let mut d = vec![0.0; 400];
    d[100] = 1.0;
    let y_full = convolve_full(&d, &lp);
    let peak = (0..y_full.len())
        .max_by(|&a, &b| y_full[a].total_cmp(&y_full[b]))
        .expect("non-empty");
    let y_valid = mono(d, 16_000)
        .apply_filter(NonEmptySlice::new(&lp).expect("non-empty"))
        .expect("filter");
    let yv = y_valid.as_slice().expect("contiguous");
    let peak_valid = (0..yv.len())
        .max_by(|&a, &b| yv[a].total_cmp(&yv[b]))
        .expect("non-empty");
    println!(
        "   impulse at n = 100: full-convolution peak at n = {peak}; apply_filter peak at index {peak_valid} (= {peak} - (M-1))"
    );
    assert_eq!(peak, 131);
    assert_eq!(peak_valid, 131 - 62);

    // Centroid of a Gaussian pulse shifts by exactly (M-1)/2 for symmetric taps.
    let pulse: Vec<f64> = (0..400)
        .map(|n| (-((n as f64 - 150.0) / 12.0).powi(2) / 2.0).exp())
        .collect();
    let centroid = |v: &[f64]| {
        v.iter().enumerate().map(|(n, a)| n as f64 * a).sum::<f64>() / v.iter().sum::<f64>()
    };
    let shift = centroid(&convolve_full(&pulse, &lp)) - centroid(&pulse);
    println!("   Gaussian pulse centroid moves by {shift:.6} samples");
    assert!((shift - 31.0).abs() < 1e-9);

    // ------------------------------------------------------------------
    // 5. Windowed-sinc low-pass, verified.
    // ------------------------------------------------------------------
    println!("\n5. Windowed-sinc low-pass: M = 63, fc = 2 kHz, fs = 16 kHz, Hamming");
    let fs = 16_000u32;
    let mut rows = Vec::new();
    for f in [0.0, 500.0, 1000.0, 1500.0, 2000.0, 3000.0, 4000.0, 6000.0] {
        let g = if f == 0.0 {
            lp.iter().sum()
        } else {
            measured_gain(&lp, f, fs)
        };
        let closed = dtft(&lp, 2.0 * PI * f / f64::from(fs)).norm();
        println!("   {f:>6} Hz: |H| = {g:.5} ({:>7.2} dB)", db(g));
        assert!((g - closed).abs() < 1e-9);
        rows.push((f, g));
    }
    let gain = |f: f64| rows.iter().find(|r| r.0 == f).expect("row").1;
    assert!((gain(0.0) - 1.0).abs() < 1e-12);
    // Passband ripple of a Hamming design is about ±0.02 dB.
    for f in [500.0, 1000.0, 1500.0] {
        assert!(db(gain(f)).abs() < 0.05, "passband at {f} Hz");
    }
    // A windowed sinc passes half the amplitude at its cutoff (-6 dB).
    assert!((gain(2000.0) - 0.5).abs() < 0.01);
    // Stop band: measured below -59 dB from 3 kHz up (Hamming side lobes are
    // nominally -53 dB; the sinc taper pushes them lower here).
    for f in [3000.0, 4000.0, 6000.0] {
        assert!(db(gain(f)) < -59.0, "stop band at {f} Hz");
    }

    let doc = topics::fir(&lp, 16_000.0);
    assert_eq!(doc.quantity("Group delay").expect("q").value, 31.0);
    assert!((doc.quantity("DC gain").expect("q").value - 1.0).abs() < 1e-12);
    assert!(doc.has_warning_containing("type I"));

    // Long filters (>= 160 taps) take the FFT path inside apply_filter; same result.
    let long = windowed_sinc(201, 2000.0, 16_000.0);
    let y_fft = mono(x.clone(), 16_000)
        .apply_filter(NonEmptySlice::new(&long).expect("non-empty"))
        .expect("filter");
    let direct = convolve_full(&x, &long);
    let e = direct[200..1000]
        .iter()
        .zip(y_fft.as_slice().expect("contiguous"))
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f64::max);
    println!("   201 taps through apply_filter (FFT path) vs direct sum: {e:.1e}");
    assert!(e < 1e-12);
    assert!(topics::fir(&long, 16_000.0).has_warning_containing("FFT convolution"));

    // Magnitude of the response at the bins of a zero-padded rfft of the taps
    // equals |H| at k fs / N: the frequency response *is* the DFT of h.
    let n_fft = 1024;
    let spec = mono(lp.clone(), fs)
        .rfft(NonZeroUsize::new(n_fft).expect("n > 0"))
        .expect("rfft");
    let k = 128; // 128 * 16000 / 1024 = 2000 Hz
    assert!((spec[[0, k]] - gain(2000.0)).abs() < 1e-9);

    println!("\n6. topics::fir for the 63-tap design\n");
    println!("{doc}");
    println!("All assertions passed.");
}
