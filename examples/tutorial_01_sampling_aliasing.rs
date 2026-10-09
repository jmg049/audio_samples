//! Tutorial 01: Sampling and aliasing.
//!
//! Accompanies the guide chapter `audio_samples::guide::c01_sampling_aliasing`
//! (`documentation/guide/01_sampling_aliasing.md`).
//!
//! What it teaches:
//! 1. the folding formula `f_a = |f - k f_s|`, `k = round(f / f_s)`;
//! 2. that an 11 kHz tone generated at 48 kHz and *naively* decimated to
//!    16 kHz (keep every third sample) reappears at 5 kHz, while a proper
//!    anti-aliased resampler removes it;
//! 3. that a naive (hard-switched) square wave aliases and a band-limited one
//!    does not, measured from the spectrum.
//!
//! Every number printed is also checked with `assert!`.
//!
//! Run with:
//!
//! ```text
//! cargo run --release --example tutorial_01_sampling_aliasing --features educational,full_no_plotting
//! ```

use std::num::NonZeroUsize;
use std::time::Duration;

use audio_samples::educational::{WarningLevel, topics};
use audio_samples::operations::types::ResamplingQuality;
use audio_samples::utils::generation::{square_wave, square_wave_bandlimited};
use audio_samples::{AudioSamples, AudioTransforms, resample, sample_rate, sine_wave};

/// Folded (alias) frequency of a tone `f` sampled at `fs`.
fn fold(f: f64, fs: f64) -> f64 {
    (f - (f / fs).round() * fs).abs()
}

/// One-sided magnitude spectrum of a mono signal, zero-padded to `n_fft`.
fn magnitude_spectrum(audio: &AudioSamples<'_, f64>, n_fft: usize) -> Vec<f64> {
    let n = NonZeroUsize::new(n_fft).expect("n_fft > 0");
    audio.rfft(n).expect("rfft").row(0).to_vec()
}

/// Index of the largest element.
fn argmax(v: &[f64]) -> usize {
    v.iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(i, _)| i)
        .expect("non-empty")
}

fn db(x: f64) -> f64 {
    20.0 * x.log10()
}

fn main() {
    println!("=== Tutorial 01: sampling and aliasing ===\n");

    // ------------------------------------------------------------------
    // 1. The folding formula, by hand and from the library.
    // ------------------------------------------------------------------
    println!("1. Folding formula f_a = |f - round(f/fs) fs|");
    let cases = [
        (1_000.0, 16_000.0, 1_000.0),
        (7_000.0, 8_000.0, 1_000.0),
        (11_000.0, 16_000.0, 5_000.0),
        (17_000.0, 16_000.0, 1_000.0),
        (25_000.0, 16_000.0, 7_000.0),
    ];
    for (f, fs, expected) in cases {
        let by_hand = fold(f, fs);
        let doc = topics::aliasing(f, fs);
        let lib = doc.quantity("Alias frequency").expect("quantity").value;
        println!(
            "   {f:>7} Hz sampled at {fs:>6} Hz appears at {by_hand:>6} Hz (library: {lib} Hz)"
        );
        assert_eq!(by_hand, expected);
        assert_eq!(lib, expected);
        // A Caution warning is raised exactly when the tone is above Nyquist.
        assert_eq!(doc.has_warning(WarningLevel::Caution), f > fs / 2.0);
    }

    // ------------------------------------------------------------------
    // 2. Naive decimation versus proper resampling.
    // ------------------------------------------------------------------
    println!("\n2. An 11 kHz tone at 48 kHz, reduced to 16 kHz");
    let fs_lo = 16_000.0;
    let f_tone = 11_000.0;
    let hi = sine_wave::<f64>(f_tone, Duration::from_millis(250), sample_rate!(48000), 0.5);
    let hi_s = hi.as_slice().expect("contiguous");
    assert_eq!(hi_s.len(), 12_000);

    // (a) Naive decimation in plain Rust: keep every third sample.
    let naive: Vec<f64> = hi_s.iter().step_by(3).copied().collect();
    assert_eq!(naive.len(), 4_000);
    let naive_audio =
        AudioSamples::new_mono(ndarray::Array1::from(naive.clone()), sample_rate!(16000))
            .expect("non-empty");

    // (b) The library's Fast tier: an integer down-ratio keeps every 3rd sample too.
    let fast = resample(&hi, sample_rate!(16000), ResamplingQuality::Fast).expect("resample");
    let fast_s = fast.as_slice().expect("contiguous");
    // The resampler works in f32, so agreement is to f32 precision (~6e-8 relative).
    let max_diff = naive
        .iter()
        .zip(fast_s)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f64::max);
    println!("   Fast tier vs naive decimation: max |difference| = {max_diff:.2e}");
    assert_eq!(fast_s.len(), naive.len());
    assert!(max_diff < 1e-6);

    // (c) The library's High tier: windowed-sinc with an anti-aliasing low-pass.
    let high = resample(&hi, sample_rate!(16000), ResamplingQuality::High).expect("resample");
    assert_eq!(high.samples_per_channel().get(), 4_000);

    // 4000 samples at 16 kHz: a 4000-point DFT has 4 Hz bins, so 5 kHz is bin 1250.
    let n_fft = 4_000;
    let bin_hz = fs_lo / n_fft as f64;
    let spec_naive = magnitude_spectrum(&naive_audio, n_fft);
    let spec_high = magnitude_spectrum(&high, n_fft);
    let peak_bin = argmax(&spec_naive);
    let peak_hz = peak_bin as f64 * bin_hz;
    // A bin-centred sine of amplitude A gives |X| = A N / 2.
    let expected_mag = 0.5 * n_fft as f64 / 2.0;
    println!(
        "   naive: peak at {peak_hz} Hz, |X| = {:.1} (A N/2 = {expected_mag})",
        spec_naive[peak_bin]
    );
    assert_eq!(peak_hz, fold(f_tone, fs_lo));
    assert_eq!(peak_hz, 5_000.0);
    assert!((spec_naive[peak_bin] - expected_mag).abs() < 1e-6 * expected_mag);

    // Rejection by the proper resampler: compare the naive alias with the
    // *largest* bin anywhere in the High-tier spectrum (a conservative figure).
    let high_max = spec_high.iter().copied().fold(0.0, f64::max);
    let rejection_db = db(spec_naive[peak_bin] / high_max);
    println!("   High tier: largest spectral peak is {rejection_db:.1} dB below the naive alias");
    assert!(rejection_db > 70.0);
    println!("   (an 11 kHz tone cannot exist in a 16 kHz signal: it should vanish, not move)");

    // ------------------------------------------------------------------
    // 3. Naive versus band-limited square wave.
    // ------------------------------------------------------------------
    println!("\n3. A 700 Hz square wave at 16 kHz, 1 s (1 Hz bins)");
    let f0 = 700.0;
    let fs = 16_000.0;
    let one_s = Duration::from_secs(1);
    let naive_sq = square_wave::<f64>(f0, one_s, sample_rate!(16000), 1.0);
    let bl_sq = square_wave_bandlimited::<f64>(f0, one_s, sample_rate!(16000), 1.0);
    let n = 16_000;
    // 700 whole periods fit in the block, so every harmonic and every alias
    // falls exactly on an integer-Hz bin: no window is needed.
    let s_naive = magnitude_spectrum(&naive_sq, n);
    let s_bl = magnitude_spectrum(&bl_sq, n);

    let is_odd_harmonic = |k: usize| k.is_multiple_of(700) && (k / 700) % 2 == 1;
    let aliased_fraction = |s: &[f64]| {
        let total: f64 = s.iter().map(|m| m * m).sum();
        let alias: f64 = s
            .iter()
            .enumerate()
            .filter(|(k, _)| !is_odd_harmonic(*k))
            .map(|(_, m)| m * m)
            .sum();
        alias / total
    };
    let frac_naive_db = 10.0 * aliased_fraction(&s_naive).log10();
    let frac_bl_db = 10.0 * aliased_fraction(&s_bl).log10();
    println!(
        "   power outside the odd harmonics: naive {frac_naive_db:.2} dB, band-limited {frac_bl_db:.1} dB (re total)"
    );
    assert!((frac_naive_db + 14.77).abs() < 0.01);
    assert!(frac_bl_db < -200.0);

    // The 13th harmonic (9100 Hz) is the first above Nyquist; it folds to 6900 Hz.
    let alias_hz = fold(13.0 * f0, fs);
    let alias_bin = alias_hz as usize;
    assert_eq!(alias_hz, 6_900.0);
    let rel_db = db(s_naive[alias_bin] / s_naive[700]);
    println!(
        "   13th harmonic 9100 Hz folds to {alias_hz} Hz, level {rel_db:.1} dB re fundamental"
    );
    // The ideal 13th harmonic is 1/13 of the fundamental (-22.3 dB). The naive
    // samples carry aliases of every higher harmonic too, so the agreement is
    // close but not exact (measured: within 0.3 dB).
    assert!((rel_db - db(1.0 / 13.0)).abs() < 0.3);
    // The band-limited version has nothing there (below 1e-9 of the fundamental).
    assert!(s_bl[alias_bin] / s_bl[700] < 1e-9);

    // ------------------------------------------------------------------
    // 4. Ask the library to explain.
    // ------------------------------------------------------------------
    println!("\n4. topics::sampling(16 kHz) and topics::aliasing(11 kHz, 16 kHz)\n");
    let s = topics::sampling(16_000.0);
    assert_eq!(s.quantity("Nyquist frequency").expect("q").value, 8_000.0);
    assert_eq!(s.quantity("Sampling period").expect("q").value, 0.0625);
    println!("{s}");
    let a = topics::aliasing(11_000.0, 16_000.0);
    assert!(a.has_warning_containing("will alias to 5000 Hz"));
    println!("{a}");
    println!("All assertions passed.");
}
