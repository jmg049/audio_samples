//! Tutorial 03: The DFT, windows and leakage.
//!
//! Accompanies the guide chapter `audio_samples::guide::c03_dft_windows`
//! (`documentation/guide/03_dft_windows.md`).
//!
//! What it teaches:
//! 1. the DFT definition and sign convention (a naive DFT loop agrees with
//!    `fft`, and a sine at bin k gives `X[k] = -j A N / 2`);
//! 2. that `rfft` returns magnitudes `|X[k]|`, not complex values (unlike numpy);
//! 3. bin frequencies `f_k = k f_s / N`;
//! 4. leakage and scalloping of an off-bin tone, with and without a window;
//! 5. window figures (coherent gain, ENBW, scalloping loss, peak side-lobe
//!    level) measured from the coefficients and compared with
//!    `topics::window`;
//! 6. that zero padding refines the frequency grid but does not resolve two
//!    close tones; a longer signal does.
//!
//! Every number printed is also checked with `assert!`.
//!
//! Run with:
//!
//! ```text
//! cargo run --release --example tutorial_03_dft_windows --features educational,full_no_plotting
//! ```

use std::f64::consts::PI;
use std::num::NonZeroUsize;
use std::time::Duration;

use audio_samples::educational::{Explain, ExplainContext, topics};
use audio_samples::{AudioProcessing, AudioSamples, AudioTransforms, sample_rate, sine_wave};
use ndarray::Array1;
use num_complex::Complex;
use spectrograms::{WindowType, make_window};

fn nz(n: usize) -> NonZeroUsize {
    NonZeroUsize::new(n).expect("n > 0")
}

fn mono(v: Vec<f64>) -> AudioSamples<'static, f64> {
    AudioSamples::new_mono(Array1::from(v), sample_rate!(16000)).expect("non-empty")
}

/// The DFT written out: X[k] = Σ x[n] e^{-j 2π k n / N}, k = 0..=N/2.
fn naive_dft(x: &[f64]) -> Vec<Complex<f64>> {
    let n = x.len();
    (0..=n / 2)
        .map(|k| {
            x.iter()
                .enumerate()
                .map(|(i, &v)| {
                    let phi = -2.0 * PI * (k * i) as f64 / n as f64;
                    Complex::new(v * phi.cos(), v * phi.sin())
                })
                .sum()
        })
        .collect()
}

struct Figures {
    cg: f64,
    enbw: f64,
    scalloping_db: f64,
    psl_db: f64,
}

/// Window figures computed from the coefficients; the spectrum is
/// zero-padded 16x so that half a bin is exactly 8 fine-grid points.
fn figures(w: &[f64]) -> Figures {
    let n = w.len();
    let sum: f64 = w.iter().sum();
    let sum2: f64 = w.iter().map(|v| v * v).sum();
    let pad = 16;
    let spec = mono(w.to_vec())
        .rfft(nz(pad * n))
        .expect("rfft")
        .row(0)
        .to_vec();
    let w0 = spec[0];
    // First null: first local minimum of |W|.
    let null = (1..spec.len() - 1)
        .find(|&i| spec[i] <= spec[i - 1] && spec[i] <= spec[i + 1])
        .expect("a null exists");
    let psl = spec[null..].iter().copied().fold(0.0, f64::max);
    Figures {
        cg: sum / n as f64,
        enbw: n as f64 * sum2 / (sum * sum),
        scalloping_db: -20.0 * (spec[pad / 2] / w0).log10(),
        psl_db: 20.0 * (psl / w0).log10(),
    }
}

/// Number of local maxima above half the largest value, between bins lo..hi.
fn count_peaks(mag: &[f64], lo: usize, hi: usize) -> usize {
    let top = mag[lo..hi].iter().copied().fold(0.0, f64::max);
    (lo.max(1)..hi)
        .filter(|&i| mag[i] > mag[i - 1] && mag[i] >= mag[i + 1] && mag[i] > 0.5 * top)
        .count()
}

fn main() {
    println!("=== Tutorial 03: the DFT, windows and leakage ===\n");
    let fs = 16_000.0;

    // ------------------------------------------------------------------
    // 1. Definition and sign convention.
    // ------------------------------------------------------------------
    println!("1. A naive DFT loop versus the library's fft");
    let n = 64;
    // A sine at bin 4 plus a cosine at bin 9 plus DC.
    let x: Vec<f64> = (0..n)
        .map(|i| {
            let t = i as f64 / n as f64;
            0.25 + (2.0 * PI * 4.0 * t).sin() + 0.5 * (2.0 * PI * 9.0 * t).cos()
        })
        .collect();
    let lib = mono(x.clone()).fft(nz(n)).expect("fft").row(0).to_vec();
    let ours = naive_dft(&x);
    let max_err = lib
        .iter()
        .zip(&ours)
        .map(|(a, b)| (a - b).norm())
        .fold(0.0, f64::max);
    println!(
        "   max |X_lib - X_naive| over {} bins = {max_err:.2e}",
        lib.len()
    );
    // O(N) additions of O(1) terms in f64: errors of order 1e-13 at most.
    assert!(max_err < 1e-12);
    println!(
        "   X[0] = {:.3}, X[4] = {:.3}, X[9] = {:.3}",
        lib[0], lib[4], lib[9]
    );
    // DC: N * 0.25 = 16. A sine at bin k: -j A N / 2 = -32j. A cosine: A N / 2 = 16.
    assert!((lib[0] - Complex::new(16.0, 0.0)).norm() < 1e-9);
    assert!((lib[4] - Complex::new(0.0, -32.0)).norm() < 1e-9);
    assert!((lib[9] - Complex::new(16.0, 0.0)).norm() < 1e-9);

    // rfft returns |X[k]|, not the complex values.
    let mags = mono(x).rfft(nz(n)).expect("rfft").row(0).to_vec();
    let max_mag_err = mags
        .iter()
        .zip(&lib)
        .map(|(m, c)| (m - c.norm()).abs())
        .fold(0.0, f64::max);
    assert!(max_mag_err < 1e-12);
    println!("   rfft(x)[4] = {:.3} (a real magnitude, |X[4]|)", mags[4]);

    // ------------------------------------------------------------------
    // 2. Bin frequencies.
    // ------------------------------------------------------------------
    println!("\n2. Bin frequencies for N = 512 at 16 kHz");
    let n = 512;
    let df = fs / n as f64;
    let dft_doc = topics::dft(n, fs);
    assert_eq!(dft_doc.quantity("Bin spacing").expect("q").value, 31.25);
    assert_eq!(
        dft_doc.quantity("Non-redundant bins").expect("q").value,
        257.0
    );
    println!(
        "   Δf = fs/N = {df} Hz; bin 32 is {} Hz; 257 bins up to {} Hz",
        32.0 * df,
        256.0 * df
    );
    assert_eq!(df, 31.25);

    // ------------------------------------------------------------------
    // 3. Leakage of an off-bin tone.
    // ------------------------------------------------------------------
    println!("\n3. Leakage: a 1000 Hz (bin 32) and a 1015.625 Hz (bin 32.5) tone, N = 512");
    let dur = Duration::from_millis(32); // exactly 512 samples
    let hann = make_window::<f64>(WindowType::Hanning, nz(n));
    let rect = make_window::<f64>(WindowType::Rectangular, nz(n));
    let spectrum = |f: f64, w: &non_empty_slice::NonEmptySlice<f64>| -> Vec<f64> {
        let tone = sine_wave::<f64>(f, dur, sample_rate!(16000), 1.0);
        assert_eq!(tone.samples_per_channel().get(), n);
        tone.apply_window(w)
            .expect("window")
            .rfft(nz(n))
            .expect("rfft")
            .row(0)
            .to_vec()
    };
    for (name, w, wt) in [
        ("rectangular", &rect, WindowType::Rectangular),
        ("Hann       ", &hann, WindowType::Hanning),
    ] {
        let on = spectrum(1000.0, w);
        let off = spectrum(1015.625, w);
        let peak_on = on.iter().copied().fold(0.0, f64::max);
        let peak_off = off.iter().copied().fold(0.0, f64::max);
        let scallop = 20.0 * (peak_on / peak_off).log10();
        // Leakage 20 bins away from the off-bin tone (bin 52 vs. the peak).
        let far = 20.0 * (off[52] / peak_off).log10();
        let predicted = topics::window(&wt, n, fs)
            .quantity("Scalloping loss")
            .expect("q")
            .value;
        println!(
            "   {name}: peak drop {scallop:.2} dB (window figure {predicted:.2} dB), level 19.5 bins away {far:.1} dB"
        );
        // The window figure assumes a lone complex exponential; a real sine
        // also has a negative-frequency image whose leakage adds to the peak
        // bins (strongest for the rectangular window), hence 0.1 dB.
        assert!((scallop - predicted).abs() < 0.1);
        match wt {
            WindowType::Rectangular => {
                assert!((predicted - 3.92).abs() < 0.01);
                assert!(far > -37.0);
            }
            _ => {
                assert!((predicted - 1.42).abs() < 0.01);
                assert!(far < -80.0);
            }
        }
    }

    // ------------------------------------------------------------------
    // 4. Window figures.
    // ------------------------------------------------------------------
    println!("\n4. Window figures at N = 512 (measured here | topics::window)");
    println!("   window       CG             ENBW (bins)    scalloping (dB)  PSL (dB)");
    for (name, wt, psl_expected) in [
        ("rectangular", WindowType::Rectangular, -13.26),
        ("Hann", WindowType::Hanning, -31.47),
        ("Hamming", WindowType::Hamming, -42.68),
        ("Blackman", WindowType::Blackman, -58.11),
    ] {
        let w = make_window::<f64>(wt.clone(), nz(n));
        let f = figures(&w);
        let doc = topics::window(&wt, n, fs);
        let q = |name: &str| doc.quantity(name).expect("quantity").value;
        println!(
            "   {name:<11}  {:.4}|{:.4}  {:.4}|{:.4}  {:.3}|{:.3}     {:.2}|{:.2}",
            f.cg,
            q("Coherent gain"),
            f.enbw,
            q("Equivalent noise bandwidth"),
            f.scalloping_db,
            q("Scalloping loss"),
            f.psl_db,
            q("Peak side-lobe level")
        );
        assert!((f.cg - q("Coherent gain")).abs() < 1e-12);
        assert!((f.enbw - q("Equivalent noise bandwidth")).abs() < 1e-12);
        // Scalloping and PSL are read off a zero-padded FFT both here and in the
        // library; the grids may differ, so they agree to 0.05 dB.
        assert!((f.scalloping_db - q("Scalloping loss")).abs() < 0.05);
        assert!((f.psl_db - q("Peak side-lobe level")).abs() < 0.05);
        assert!(
            (f.psl_db - psl_expected).abs() < 0.1,
            "{name}: {}",
            f.psl_db
        );
    }

    // The library's Hann is symmetric: w[0] = w[N-1] = 0, denominator N - 1.
    assert_eq!(hann[0], 0.0);
    assert!(hann[n - 1].abs() < 1e-15);
    let periodic_mid = 0.5 - 0.5 * (2.0 * PI * 256.0 / n as f64).cos();
    println!(
        "   symmetric Hann: w[0] = {}, w[511] = {:.1e}, w[256] = {:.6} (periodic Hann would give {periodic_mid})",
        hann[0],
        hann[n - 1],
        hann[256]
    );
    assert!(hann[256] < 1.0 && periodic_mid == 1.0);
    let hann_doc = WindowType::Hanning.explain_with(&ExplainContext::new().with_n_fft(n));
    assert!(hann_doc.has_warning_containing("symmetric"));

    // ------------------------------------------------------------------
    // 5. Zero padding is not resolution.
    // ------------------------------------------------------------------
    println!("\n5. Tones at 1000 and 1020 Hz (20 Hz apart)");
    let two_tones = |dur: Duration| {
        let a = sine_wave::<f64>(1000.0, dur, sample_rate!(16000), 1.0);
        let b = sine_wave::<f64>(1020.0, dur, sample_rate!(16000), 1.0);
        let v: Vec<f64> = a
            .as_slice()
            .expect("contiguous")
            .iter()
            .zip(b.as_slice().expect("contiguous"))
            .map(|(x, y)| x + y)
            .collect();
        v
    };
    let analyse = |v: Vec<f64>, n_fft: usize| {
        let len = v.len();
        let w = make_window::<f64>(WindowType::Hanning, nz(len));
        let mag = mono(v)
            .apply_window(&w)
            .expect("window")
            .rfft(nz(n_fft))
            .expect("rfft")
            .row(0)
            .to_vec();
        let bin = fs / n_fft as f64;
        count_peaks(&mag, (900.0 / bin) as usize, (1120.0 / bin) as usize)
    };
    let short = two_tones(Duration::from_millis(32)); // 512 samples
    let long = two_tones(Duration::from_millis(256)); // 4096 samples
    let p_short = analyse(short.clone(), 512);
    let p_padded = analyse(short, 8192);
    let p_long = analyse(long, 8192);
    println!("   512 samples, N = 512:              {p_short} peak(s) (Δf = 31.25 Hz)");
    println!(
        "   512 samples zero-padded to 8192:   {p_padded} peak(s) (grid 1.95 Hz, same main lobe)"
    );
    println!("   4096 samples, N = 8192:            {p_long} peak(s) (main lobe 8x narrower)");
    assert_eq!(p_short, 1);
    assert_eq!(p_padded, 1);
    assert_eq!(p_long, 2);

    // ------------------------------------------------------------------
    // 6. Ask the library to explain.
    // ------------------------------------------------------------------
    println!("\n6. WindowType::Hanning explained at N = 512, 16 kHz\n");
    let doc =
        WindowType::Hanning.explain_with(&ExplainContext::new().with_n_fft(n).with_sample_rate(fs));
    let enbw_hz = doc.quantity("ENBW (Hz)").expect("q").value;
    assert!((enbw_hz - figures(&hann).enbw * df).abs() < 1e-9);
    println!("{doc}");
    println!("All assertions passed.");
}
