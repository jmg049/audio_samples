//! Tutorial 06: IIR filters: design, poles, zeros and stability.
//!
//! Accompanies the guide chapter `audio_samples::guide::c06_iir_filters`.
//!
//! What this lab shows, with every number checked by an `assert!`:
//!
//! 1. A 4th-order Butterworth low-pass (1 kHz at 16 kHz) designed by the
//!    library has exactly -3.01 dB at its cutoff and follows the closed-form
//!    digital Butterworth magnitude.
//! 2. The same filter re-derived from first principles (analogue poles on a
//!    circle, pre-warping, bilinear transform) has the same poles, gain and
//!    frequency response as the library design.
//! 3. Running the second-order sections' difference equations by hand gives the
//!    same output as `apply_iir_filter`.
//! 4. Every pole lies inside the unit circle; the margin shrinks as the cutoff
//!    becomes a small fraction of the sample rate.
//! 5. Chebyshev type I trades a 1 dB pass-band ripple for a steeper roll-off.
//! 6. Forward-backward filtering (`filtfilt`) has zero phase and squared gain
//!    (-6.02 dB at the cutoff).
//!
//! Run with:
//! `cargo run --release --example tutorial_06_iir_filters --features educational,full_no_plotting`

use std::f64::consts::PI;
use std::time::Duration;

use audio_samples::educational::ExplainFor;
use audio_samples::operations::types::{FilterResponse, IirFilterDesign};
use audio_samples::{AudioIirFiltering, AudioSampleResult, impulse, nzu, sample_rate, sine_wave};
use num_complex::Complex64;

const FS: f64 = 16_000.0;
const FC: f64 = 1_000.0;

fn db(x: f64) -> f64 {
    20.0 * x.log10()
}

/// Roots of `c0 z^2 + c1 z + c2`: the poles (or zeros) of one section, since
/// `c0 + c1 z^-1 + c2 z^-2 = 0` is the same equation multiplied by `z^-2`.
fn quadratic_roots(c0: f64, c1: f64, c2: f64) -> Vec<Complex64> {
    if c2 == 0.0 {
        // First-order section: c0 z + c1 = 0 (or no finite root at all).
        return if c1 == 0.0 {
            vec![]
        } else {
            vec![Complex64::new(-c1 / c0, 0.0)]
        };
    }
    let disc = Complex64::new(c1 * c1 - 4.0 * c0 * c2, 0.0).sqrt();
    vec![(-c1 + disc) / (2.0 * c0), (-c1 - disc) / (2.0 * c0)]
}

/// Poles of every section of an SOS cascade (rows `[b0,b1,b2,a0,a1,a2]`).
fn sos_poles(sos: &[[f64; 6]]) -> Vec<Complex64> {
    sos.iter()
        .flat_map(|s| quadratic_roots(s[3], s[4], s[5]))
        .collect()
}

/// Digital Butterworth low-pass from first principles: returns (poles, gain).
/// All N zeros are at z = -1.
fn butterworth_from_scratch(order: usize, fc: f64, fs: f64) -> (Vec<Complex64>, f64) {
    let k = 2.0 * fs; // bilinear constant K = 2 f_s (any K gives the same H(z))
    let omega_c = k * (PI * fc / fs).tan(); // pre-warped analogue cutoff, rad/s
    let n = order as f64;
    let poles: Vec<Complex64> = (0..order)
        .map(|i| {
            // Left-half-plane poles on a circle of radius Omega_c.
            let theta = PI * (2.0 * i as f64 + n + 1.0) / (2.0 * n);
            let s = Complex64::from_polar(omega_c, theta);
            (k + s) / (k - s) // bilinear map s -> z
        })
        .collect();
    // Choose the gain so that H(z = 1) = 1 (unity gain at DC):
    // H(1) = g 2^N / prod(1 - p_k).
    let den: Complex64 = poles.iter().map(|p| Complex64::new(1.0, 0.0) - p).product();
    let gain = den.re / 2f64.powi(order as i32);
    (poles, gain)
}

/// |H(e^{jw})| of a filter with all zeros at z = -1.
fn zpk_mag(poles: &[Complex64], gain: f64, f: f64, fs: f64) -> f64 {
    let z = Complex64::from_polar(1.0, 2.0 * PI * f / fs);
    let num: Complex64 = poles.iter().map(|_| z + 1.0).product();
    let den: Complex64 = poles.iter().map(|p| z - p).product();
    (gain * num / den).norm()
}

/// Direct-form I difference equation of a cascade of biquads.
fn run_sos_by_hand(sos: &[[f64; 6]], x: &[f64]) -> Vec<f64> {
    let mut y = x.to_vec();
    for s in sos {
        let (b0, b1, b2, a0, a1, a2) = (s[0], s[1], s[2], s[3], s[4], s[5]);
        let (mut x1, mut x2, mut y1, mut y2) = (0.0, 0.0, 0.0, 0.0);
        for v in &mut y {
            let xn = *v;
            let yn = (b0 * xn + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2) / a0;
            x2 = x1;
            x1 = xn;
            y2 = y1;
            y1 = yn;
            *v = yn;
        }
    }
    y
}

fn main() -> AudioSampleResult<()> {
    println!("== Tutorial 06: IIR filters ==\n");

    // ---------------------------------------------------------------- 1
    let design = IirFilterDesign::butterworth_lowpass(nzu!(4), FC);
    let sos = design.sos_coefficients(FS)?;
    println!("1. Butterworth low-pass, N = 4, f_c = {FC} Hz, f_s = {FS} Hz");
    println!(
        "   {} second-order sections [b0 b1 b2 a0 a1 a2]:",
        sos.len()
    );
    for s in &sos {
        println!(
            "   [{:+.6e} {:+.6e} {:+.6e} {:.1} {:+.6} {:+.6}]",
            s[0], s[1], s[2], s[3], s[4], s[5]
        );
    }
    assert_eq!(sos.len(), 2);
    let probes = [0.0, FC, 2.0 * FC, 4.0 * FC];
    let (mag, _) = design.frequency_response(&probes, FS)?;
    println!(
        "   |H| at 0, fc, 2fc, 4fc: {:.3} dB, {:.4} dB, {:.2} dB, {:.2} dB",
        db(mag[0]),
        db(mag[1]),
        db(mag[2]),
        db(mag[3])
    );
    // 1/sqrt(2) is exact for a Butterworth at fc; tolerances allow for rounding.
    assert!(db(mag[0]).abs() < 1e-9);
    assert!((db(mag[1]) + 3.0103).abs() < 1e-4);

    // Closed form: |H|^2 = 1 / (1 + (tan(pi f/fs) / tan(pi fc/fs))^(2N)).
    for (&f, &m) in probes.iter().zip(&mag) {
        let r = (PI * f / FS).tan() / (PI * FC / FS).tan();
        let exact = 1.0 / (1.0 + r.powi(8)).sqrt();
        assert!((m - exact).abs() < 1e-12, "f = {f}: {m} vs {exact}");
    }
    // At 2 fc the digital filter is 25.5 dB down; the analogue prototype
    // would be 10 log10(1 + 2^8) = 24.1 dB down.
    assert!((db(mag[2]) + 25.5).abs() < 0.05);
    println!("   matches |H|^2 = 1/(1 + (tan(pi f/fs)/tan(pi fc/fs))^8) to 1e-12\n");

    // ---------------------------------------------------------------- 2
    println!("2. Re-deriving the design: analogue poles -> pre-warp -> bilinear transform");
    let (my_poles, my_gain) = butterworth_from_scratch(4, FC, FS);
    let lib_poles = sos_poles(&sos);
    assert_eq!(lib_poles.len(), 4);
    for p in &my_poles {
        let nearest = lib_poles
            .iter()
            .map(|q| (p - q).norm())
            .fold(f64::INFINITY, f64::min);
        println!(
            "   pole {:+.6} {:+.6}j  |p| = {:.6}  (distance to library pole {:.1e})",
            p.re,
            p.im,
            p.norm(),
            nearest
        );
        // Both are O(1) numbers from a few dozen flops; 1e-12 is ample.
        assert!(nearest < 1e-12);
    }
    let lib_gain: f64 = sos.iter().map(|s| s[0]).product();
    println!("   overall gain: mine {my_gain:.6e}, library product of b0 {lib_gain:.6e}");
    assert!((my_gain - lib_gain).abs() / lib_gain < 1e-10);
    for f in [100.0, 900.0, 1_000.0, 1_500.0, 3_000.0, 7_000.0] {
        let (m, _) = design.frequency_response(&[f], FS)?;
        assert!((zpk_mag(&my_poles, my_gain, f, FS) - m[0]).abs() < 1e-12);
    }
    println!("   zero-pole-gain response agrees with the library to 1e-12\n");

    // ---------------------------------------------------------------- 3
    println!("3. The difference equation, by hand");
    let imp = impulse::<f64>(Duration::from_millis(20), sample_rate!(16000), 1.0, 0.0);
    let lib_out = imp.apply_iir_filter(&design)?;
    let x = imp.as_slice().expect("mono, contiguous").to_vec();
    let mine = run_sos_by_hand(&sos, &x);
    let lib = lib_out.as_slice().expect("mono, contiguous");
    let max_err = mine
        .iter()
        .zip(lib)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f64::max);
    println!("   impulse response h[0..4] = {:.6?}", &mine[..4]);
    println!("   max |hand - library| = {max_err:.2e}");
    // Same arithmetic in a different association order: 1e-12 covers rounding.
    assert!(max_err < 1e-12);
    // The impulse response sums to the DC gain, 1 (it has decayed by 20 ms).
    let dc: f64 = mine.iter().sum();
    println!("   sum of h[n] over 20 ms = {dc:.6} (DC gain)\n");
    assert!((dc - 1.0).abs() < 1e-6);

    // ---------------------------------------------------------------- 4
    println!("4. Stability: pole radius versus normalised cutoff (N = 4)");
    let mut last = 0.0;
    for fc in [4_000.0, 1_000.0, 250.0, 50.0] {
        let r = sos_poles(&IirFilterDesign::butterworth_lowpass(nzu!(4), fc).sos_coefficients(FS)?)
            .iter()
            .map(|p| p.norm())
            .fold(0.0, f64::max);
        println!(
            "   fc = {fc:>6} Hz  (fc/fs = {:.5})  max |p| = {r:.6}",
            fc / FS
        );
        assert!(r < 1.0 && r > last);
        last = r;
    }
    let doc = imp.explain(&IirFilterDesign::butterworth_lowpass(nzu!(8), 10.0));
    let margin = doc.quantity("Stability margin").unwrap().value;
    println!("   order 8 at 10 Hz: stability margin 1 - max|p| = {margin:.2e}");
    assert!(margin > 0.0 && margin < 1e-3);
    assert!(doc.has_warning_containing("very sensitive to rounding"));
    println!("   -> the explainer warns: poles within 1e-3 of the unit circle\n");

    // ---------------------------------------------------------------- 5
    println!("5. Butterworth versus Chebyshev type I (N = 4, 1 dB ripple)");
    let cheb = IirFilterDesign::chebyshev_i(FilterResponse::LowPass, nzu!(4), FC, 1.0);
    let grid: Vec<f64> = (0..=1000).map(|i| FC * i as f64 / 1000.0).collect();
    let (mb, _) = design.frequency_response(&grid, FS)?;
    let (mc, _) = cheb.frequency_response(&grid, FS)?;
    let (b_max, b_min) = (db(mb[0]), db(*mb.last().unwrap()));
    let c_max = mc.iter().map(|&m| db(m)).fold(f64::MIN, f64::max);
    let c_min = mc.iter().map(|&m| db(m)).fold(f64::MAX, f64::min);
    println!(
        "   pass band 0..fc: Butterworth {b_min:.3}..{b_max:.3} dB, Chebyshev {c_min:.3}..{c_max:.3} dB"
    );
    assert!(mb.windows(2).all(|w| w[1] <= w[0] + 1e-12)); // monotonic (up to rounding near DC)
    // Even order: 0 dB at the ripple peaks, -1 dB at DC and at fc.
    assert!(c_max.abs() < 1e-3 && (c_min + 1.0).abs() < 1e-6);
    let (b2, _) = design.frequency_response(&[2.0 * FC], FS)?;
    let (c2, _) = cheb.frequency_response(&[2.0 * FC], FS)?;
    println!(
        "   at 2 fc: Butterworth {:.2} dB, Chebyshev {:.2} dB",
        db(b2[0]),
        db(c2[0])
    );
    // About 10 dB more attenuation one octave above the cutoff.
    assert!((db(c2[0]) + 35.5).abs() < 0.1);
    println!();

    // ---------------------------------------------------------------- 6
    println!("6. Zero-phase filtering with filtfilt");
    let tone = sine_wave::<f64>(FC, Duration::from_millis(500), sample_rate!(16000), 1.0);
    let once = tone.apply_iir_filter(&design)?;
    let twice = tone.filtfilt(&design)?;
    let t = tone.as_slice().unwrap();
    let o = once.as_slice().unwrap();
    let w = twice.as_slice().unwrap();
    // Measure amplitude and lag in the middle 250 ms (away from edge transients).
    let mid = 2_000..6_000;
    let amp = |v: &[f64]| v[mid.clone()].iter().fold(0.0f64, |a, x| a.max(x.abs()));
    let corr = |v: &[f64], lag: usize| -> f64 { mid.clone().map(|n| v[n] * t[n - lag]).sum() };
    let best_lag = |v: &[f64]| {
        (0..16)
            .max_by(|&a, &b| corr(v, a).total_cmp(&corr(v, b)))
            .unwrap()
    };
    println!(
        "   1 kHz tone: sosfilt gain {:.3} dB, lag {} samples; filtfilt gain {:.3} dB, lag {} samples",
        db(amp(o)),
        best_lag(o),
        db(amp(w)),
        best_lag(w)
    );
    // Peak-picking a sampled sine can underestimate its amplitude by up to
    // 1 - cos(pi fc/fs) = 1.9 % (0.17 dB); 0.2 dB covers it.
    assert!((db(amp(o)) + 3.01).abs() < 0.2);
    assert!((db(amp(w)) + 6.02).abs() < 0.2);
    assert_eq!(best_lag(w), 0);
    assert_eq!(best_lag(o), 8); // half a period: -180 degrees at fc for N = 4

    println!("\n== Explain: what the library says about this design ==\n");
    let doc = tone.explain(&design);
    println!("{doc}");
    let g = doc.quantity("Measured gain at the cutoff").unwrap().value;
    assert!((g + 3.0103).abs() < 1e-3);
    println!("All checks passed.");
    Ok(())
}
