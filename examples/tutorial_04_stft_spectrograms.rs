//! Tutorial 04: STFT and spectrograms.
//!
//! Accompanies the guide chapter `audio_samples::guide::c04_stft_spectrograms`
//! (`documentation/guide/04_stft_spectrograms.md`).
//!
//! What it teaches:
//! 1. the STFT as the crate computes it (centring by ⌊N/2⌋ zeros, symmetric
//!    Hann window), re-derived for one frame with a naive DFT;
//! 2. the frame-count formula, checked against `stft()`;
//! 3. inversion with `istft` (weighted overlap-add) to round-off accuracy;
//! 4. the time-frequency trade-off, measured: a long window resolves two
//!    close tones, a short one localises a click;
//! 5. linear, log-frequency and mel spectrograms, and the *absolute* dB floor;
//! 6. `StftParams::explain_with` and its pitch-range warnings.
//!
//! Every number printed is also checked with `assert!`.
//!
//! Run with:
//!
//! ```text
//! cargo run --release --example tutorial_04_stft_spectrograms --features educational,full_no_plotting
//! ```

use std::f64::consts::PI;
use std::num::NonZeroUsize;
use std::time::Duration;

use audio_samples::educational::{Explain, ExplainContext, ExplainFor, WarningLevel};
use audio_samples::utils::generation::{impulse, silence};
use audio_samples::{AudioSamples, AudioTransforms, nzu, sample_rate, sine_wave, white_noise};
use ndarray::Array1;
use num_complex::Complex;
use spectrograms::{
    LogHzParams, LogParams, MelParams, SpectrogramParams, StftParams, WindowType, make_window,
};

fn nz(n: usize) -> NonZeroUsize {
    NonZeroUsize::new(n).expect("n > 0")
}

/// 1 + ⌊(L + 2p − N) / H⌋ frames, p = ⌊N/2⌋ with centring, 0 without.
fn frame_count(l: usize, n: usize, h: usize, centre: bool) -> usize {
    let p = if centre { n / 2 } else { 0 };
    1 + (l + 2 * p - n) / h
}

/// Number of local maxima above half the largest value in `v[lo..hi]`.
fn count_peaks(v: &[f64], lo: usize, hi: usize) -> usize {
    let top = v[lo..hi].iter().copied().fold(0.0, f64::max);
    (lo..hi)
        .filter(|&i| v[i] > v[i - 1] && v[i] >= v[i + 1] && v[i] > 0.5 * top)
        .count()
}

fn main() {
    println!("=== Tutorial 04: STFT and spectrograms ===\n");
    let fs = 16_000.0;
    let sr = sample_rate!(16000);
    let one_s = Duration::from_secs(1);

    // ------------------------------------------------------------------
    // 1. Frame count and one frame by hand.
    // ------------------------------------------------------------------
    println!("1. STFT of 1 s of noise at 16 kHz, N = 512, H = 128, Hann");
    let x = white_noise::<f64>(one_s, sr, 0.5, Some(7));
    let xs = x.as_slice().expect("contiguous").to_vec();
    let l = xs.len();
    let (n, h) = (512, 128);
    let params = StftParams::new(nzu!(512), nzu!(128), WindowType::Hanning, true).expect("params");
    let st = x.stft(&params).expect("stft");
    let frames = st.data.ncols();
    println!(
        "   centre = true : {frames} frames (formula: {}), {} bins",
        frame_count(l, n, h, true),
        st.data.nrows()
    );
    assert_eq!(frames, frame_count(l, n, h, true));
    assert_eq!(frames, 126);
    assert_eq!(st.data.nrows(), n / 2 + 1);
    let p_nc = StftParams::new(nzu!(512), nzu!(128), WindowType::Hanning, false).expect("params");
    let frames_nc = x.stft(&p_nc).expect("stft").data.ncols();
    println!(
        "   centre = false: {frames_nc} frames (formula: {})",
        frame_count(l, n, h, false)
    );
    assert_eq!(frames_nc, frame_count(l, n, h, false));
    assert_eq!(frames_nc, 122);

    // Frame m by hand: pad ⌊N/2⌋ zeros each side, take N samples from mH,
    // multiply by the symmetric Hann window, take the DFT.
    let m = 10;
    let pad = n / 2;
    let w: Vec<f64> = (0..n)
        .map(|i| 0.5 - 0.5 * (2.0 * PI * i as f64 / (n - 1) as f64).cos())
        .collect();
    let lib_w = make_window::<f64>(WindowType::Hanning, nz(n));
    let w_err = w
        .iter()
        .zip(lib_w.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f64::max);
    assert!(w_err < 1e-15);
    let frame: Vec<f64> = (0..n)
        .map(|i| {
            let idx = (m * h + i) as isize - pad as isize;
            if idx >= 0 && (idx as usize) < l {
                xs[idx as usize] * w[i]
            } else {
                0.0
            }
        })
        .collect();
    let mut max_err: f64 = 0.0;
    for k in 0..=n / 2 {
        let xk: Complex<f64> = frame
            .iter()
            .enumerate()
            .map(|(i, &v)| Complex::from_polar(v, -2.0 * PI * (k * i % n) as f64 / n as f64))
            .sum();
        max_err = max_err.max((xk - st.data[[k, m]]).norm());
    }
    println!("   frame {m} by hand vs stft(): max |difference| = {max_err:.2e}");
    // 512-term sums of O(0.5) values: rounding error of order 1e-13.
    assert!(max_err < 1e-10);

    // ------------------------------------------------------------------
    // 2. Inversion.
    // ------------------------------------------------------------------
    let y = AudioSamples::<f64>::istft(st).expect("istft");
    let ys = y.as_slice().expect("contiguous");
    let expected_len = (frames - 1) * h + n - 2 * (n / 2);
    let rec_err = xs
        .iter()
        .zip(ys)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f64::max);
    println!(
        "\n2. istft: {} samples (formula (M-1)H + N - 2⌊N/2⌋ = {expected_len}), max error {rec_err:.2e}",
        ys.len()
    );
    assert_eq!(ys.len(), expected_len);
    assert_eq!(ys.len(), 16_000);
    assert!(rec_err < 1e-12);

    // ------------------------------------------------------------------
    // 3. The time-frequency trade-off.
    // ------------------------------------------------------------------
    println!("\n3. Time-frequency trade-off (hop 32 samples = 2 ms)");
    let a = sine_wave::<f64>(1000.0, one_s, sr, 0.5);
    let b = sine_wave::<f64>(1050.0, one_s, sr, 0.5);
    let tones: Vec<f64> = a
        .as_slice()
        .expect("contiguous")
        .iter()
        .zip(b.as_slice().expect("contiguous"))
        .map(|(p, q)| p + q)
        .collect();
    let tones = AudioSamples::new_mono(Array1::from(tones), sr).expect("non-empty");
    let click = impulse::<f64>(one_s, sr, 1.0, 0.5);
    let mut spreads = Vec::new();
    for n_fft in [256usize, 2048] {
        let p = StftParams::new(nz(n_fft), nzu!(32), WindowType::Hanning, true).expect("params");
        // Frequency: average magnitude over frames, count peaks near 1 kHz.
        let s = tones.stft(&p).expect("stft");
        let avg: Vec<f64> = (0..s.data.nrows())
            .map(|k| s.data.row(k).iter().map(|c| c.norm()).sum::<f64>())
            .collect();
        let bin = fs / n_fft as f64;
        let peaks = count_peaks(&avg, (800.0 / bin) as usize, (1250.0 / bin) as usize);
        // Time: frame energy of the click; width above half the maximum.
        let c = click.stft(&p).expect("stft");
        let energy: Vec<f64> = (0..c.data.ncols())
            .map(|j| c.data.column(j).iter().map(|v| v.norm_sqr()).sum::<f64>())
            .collect();
        let top = energy.iter().copied().fold(0.0, f64::max);
        let width_ms =
            energy.iter().filter(|&&e| e > 0.5 * top).count() as f64 * 32.0 / fs * 1000.0;
        println!(
            "   N = {n_fft:>4}: Δf = {bin:>6.2} Hz, window {:>5.1} ms -> {peaks} peak(s) for 1000+1050 Hz; click spread {width_ms:.0} ms",
            n_fft as f64 / fs * 1000.0
        );
        spreads.push((peaks, width_ms));
    }
    assert_eq!(spreads[0].0, 1);
    assert_eq!(spreads[1].0, 2);
    // The click's energy envelope is Σ w²: its half-power width is a fixed
    // fraction of N, so it scales with N (8x here, to within one hop).
    let ratio = spreads[1].1 / spreads[0].1;
    println!("   click spread ratio = {ratio:.2} (window ratio 8)");
    assert!((ratio - 8.0).abs() < 0.6);

    // ------------------------------------------------------------------
    // 4. Spectrogram types and the dB floor.
    // ------------------------------------------------------------------
    println!("\n4. Linear / log-Hz / mel spectrograms of a full-scale 1 kHz sine");
    let sine = sine_wave::<f64>(1000.0, one_s, sr, 1.0);
    let sp = SpectrogramParams::new(params.clone(), fs).expect("params");
    let db = LogParams::new(-80.0).expect("db");
    let lin = sine.linear_db_spectrogram(&sp, &db).expect("lin");
    let loghz = sine
        .loghz_power_spectrogram(&sp, &LogHzParams::new(nzu!(64), 80.0, 8000.0).expect("p"))
        .expect("loghz");
    let mel_p = MelParams::new(nzu!(40), 0.0, 8000.0).expect("mel");
    let mel = sine.mel_power_spectrogram(&sp, &mel_p).expect("mel");
    println!(
        "   rows: linear {}, log-Hz {}, mel {}; frames {}",
        lin.data().nrows(),
        loghz.data().nrows(),
        mel.data().nrows(),
        lin.data().ncols()
    );
    assert_eq!(lin.data().nrows(), 257);
    assert_eq!(loghz.data().nrows(), 64);
    assert_eq!(mel.data().nrows(), 40);

    // Absolute dB: a bin-centred sine of amplitude A peaks at A Σw / 2.
    let sum_w: f64 = w.iter().sum();
    let predicted_db = 20.0 * (sum_w / 2.0).log10();
    let peak_db = lin.data().iter().copied().fold(f64::MIN, f64::max);
    println!(
        "   linear dB peak = {peak_db:.2} dB re 1 (predicted 20 log10(Σw/2) = {predicted_db:.2} dB), above 0 dB"
    );
    assert!((peak_db - predicted_db).abs() < 0.01);
    assert!((predicted_db - 42.13).abs() < 0.01);
    let quiet = silence::<f64>(Duration::from_millis(250), sr);
    let q_db = quiet.linear_db_spectrogram(&sp, &db).expect("lin");
    let all_floor = q_db.data().iter().all(|&v| (v + 80.0).abs() < 1e-9);
    println!("   silence: every bin equals the floor, -80 dB re 1: {all_floor}");
    assert!(all_floor);

    // The mel band with the most energy is centred near 1 kHz.
    let mel_freqs = mel.frequencies();
    let band_energy: Vec<f64> = (0..40).map(|b| mel.data().row(b).sum()).collect();
    let best = (0..40)
        .max_by(|&i, &j| band_energy[i].total_cmp(&band_energy[j]))
        .expect("non-empty");
    let spacing = mel_freqs[best + 1] - mel_freqs[best];
    println!(
        "   strongest mel band: {best} centred at {:.0} Hz (neighbour spacing {spacing:.0} Hz)",
        mel_freqs[best]
    );
    assert!((mel_freqs[best] - 1000.0).abs() < spacing);

    // ------------------------------------------------------------------
    // 5. Ask the library to explain.
    // ------------------------------------------------------------------
    println!("\n5. StftParams explained for a speech signal with f0 in 80-400 Hz");
    let speech_like = sine_wave::<f64>(120.0, one_s, sr, 0.5);
    for (n_fft, cautions) in [(1024usize, 0usize), (256, 1), (128, 2)] {
        let p =
            StftParams::new(nz(n_fft), nz(n_fft / 4), WindowType::Hanning, true).expect("params");
        let ctx = ExplainContext::for_audio(&speech_like).with_f0_range(80.0, 400.0);
        let doc = p.explain_with(&ctx);
        let short = doc.has_warning_containing("shorter than two periods");
        let coarse = doc.has_warning_containing("coarser than the lowest expected f0");
        println!(
            "   N = {n_fft:>4}: Δf = {:>5.1} Hz, frames = {}, short-window warning {short}, coarse-bin warning {coarse}",
            doc.quantity("Frequency resolution").expect("q").value,
            doc.quantity("Frames").expect("q").value
        );
        assert_eq!(usize::from(short) + usize::from(coarse), cautions);
        assert_eq!(doc.has_warning(WarningLevel::Caution), cautions > 0);
        assert_eq!(
            doc.quantity("Frames").expect("q").value as usize,
            frame_count(16_000, n_fft, n_fft / 4, true)
        );
    }
    let doc = speech_like.explain(&params);
    assert_eq!(
        doc.quantity("Frequency resolution").expect("q").value,
        31.25
    );
    assert_eq!(doc.quantity("Frames").expect("q").value, 126.0);
    println!("\n{doc}");
    println!("All assertions passed.");
}
