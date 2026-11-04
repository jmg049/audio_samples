#![cfg(all(feature = "statistics", feature = "transforms", feature = "processing"))]
//! Regression tests for statistics, signal comparison and sample-rate detection.

use std::f64::consts::PI;
use std::num::NonZeroUsize;

use audio_samples::operations::types::ChannelReduction;
use audio_samples::utils::comparison::align_signals;
use audio_samples::utils::detection::{
    analyze_spectrum_for_cutoff, detect_fundamental_frequency, detect_sample_rate,
};
use audio_samples::{
    AudioProcessing, AudioSamples, AudioStatistics, fft_alignment_lag, sample_rate,
};
use ndarray::{Array1, Array2, array};
use non_empty_slice::NonEmptySlice;

fn nz(n: usize) -> NonZeroUsize {
    NonZeroUsize::new(n).unwrap()
}

fn sine(freq: f64, fs: f64, n: usize, amp: f64) -> Vec<f64> {
    (0..n)
        .map(|i| amp * (2.0 * PI * freq * i as f64 / fs).sin())
        .collect()
}

/// Deterministic uniform noise in [-1, 1) (64-bit LCG).
fn noise(n: usize, seed: u64) -> Vec<f64> {
    let mut state = seed;
    (0..n)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((state >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
        })
        .collect()
}

// the autocorrelation was divided by the FFT size twice.
#[test]
fn autocorrelation_is_unbiased_estimate() {
    let audio = AudioSamples::new_mono(array![1.0f64, 2.0, 3.0, 4.0], sample_rate!(8000)).unwrap();
    let r = audio.autocorrelation(nz(3)).unwrap();
    let expected = [30.0 / 4.0, 20.0 / 3.0, 11.0 / 2.0, 4.0];
    for (k, (&got, &want)) in r.iter().zip(expected.iter()).enumerate() {
        assert!((got - want).abs() < 1e-9, "lag {k}: got {got}, want {want}");
    }

    // r[0] of a sine of amplitude A is its mean square, A^2/2.
    let a = 0.8;
    let x = sine(440.0, 16000.0, 16000, a);
    let audio = AudioSamples::new_mono(Array1::from(x), sample_rate!(16000)).unwrap();
    let r0 = audio.autocorrelation(nz(1)).unwrap()[0];
    assert!(
        ((r0 - a * a / 2.0) / (a * a / 2.0)).abs() < 1e-3,
        "r0 = {r0}"
    );
}

// cross-correlation accumulated in the sample type (i16 overflow).
#[test]
fn cross_correlation_i16_accumulates_in_f64() {
    let x: Vec<i16> = sine(440.0, 16000.0, 16000, 0.5 * 32767.0)
        .iter()
        .map(|v| v.round() as i16)
        .collect();
    let mean_sq = x.iter().map(|&v| f64::from(v) * f64::from(v)).sum::<f64>() / x.len() as f64;
    let audio = AudioSamples::new_mono(Array1::from(x), sample_rate!(16000)).unwrap();
    let r = audio.cross_correlation(&audio, nz(2)).unwrap();
    assert!((r[0] - mean_sq).abs() / mean_sq < 1e-12, "r0 = {}", r[0]);

    // Convention: r[l] = (1/count) sum self[i] * other[i + l]; a peak at l = D
    // means `other` is `self` delayed by D samples.
    let base = noise(400, 7);
    let mut delayed = vec![0.0; 7];
    delayed.extend_from_slice(&base[..400 - 7]);
    let a = AudioSamples::new_mono(Array1::from(base), sample_rate!(8000)).unwrap();
    let b = AudioSamples::new_mono(Array1::from(delayed), sample_rate!(8000)).unwrap();
    let r = a.cross_correlation(&b, nz(20)).unwrap();
    let argmax = r
        .iter()
        .enumerate()
        .max_by(|p, q| p.1.partial_cmp(q.1).unwrap())
        .unwrap()
        .0;
    assert_eq!(argmax, 7);
}

// mean/variance summed in the sample type (overflow, panic).
#[test]
fn mean_and_variance_do_not_overflow_integer_types() {
    let audio =
        AudioSamples::new_mono(Array1::from(vec![1000i16; 40_000]), sample_rate!(8000)).unwrap();
    assert_eq!(audio.mean(), 1000.0);
    assert_eq!(audio.variance(), 0.0);

    let audio = AudioSamples::new_mono(Array1::from(vec![200u8; 200]), sample_rate!(8000)).unwrap();
    assert_eq!(audio.mean(), 200.0);

    let audio =
        AudioSamples::new_mono(array![30000i16, 30000, 30000, 30000], sample_rate!(8000)).unwrap();
    assert_eq!(audio.mean(), 30000.0);
}

// DC removal on i16 used the overflowed mean.
#[test]
fn remove_dc_offset_i16_constant_gives_zero() {
    let mut audio =
        AudioSamples::new_mono(array![30000i16, 30000, 30000, 30000], sample_rate!(8000)).unwrap();
    audio.remove_dc_offset_in_place().unwrap();
    for i in 0..4 {
        assert_eq!(audio[i], 0, "sample {i}");
    }
}

// multi-channel variance was the inter-channel spread (0 for L = R).
#[test]
fn variance_multichannel_is_population_variance_of_all_samples() {
    let x = sine(440.0, 16000.0, 16000, 0.5);
    let mut data = Array2::<f64>::zeros((2, x.len()));
    for (i, &v) in x.iter().enumerate() {
        data[[0, i]] = v;
        data[[1, i]] = v;
    }
    let stereo = AudioSamples::new_multi_channel(data, sample_rate!(16000)).unwrap();
    let var = stereo.variance();
    assert!((var - 0.125).abs() < 1e-3, "variance = {var}");
    assert!((stereo.std_dev() - var.sqrt()).abs() < 1e-15);
}

// u8 is offset-binary, so zero crossings must be counted about 128.
#[test]
fn zero_crossings_u8_counted_about_mid_scale() {
    let x: Vec<u8> = sine(440.0, 44100.0, 44100, 0.5)
        .iter()
        .map(|v| (128.0 + 127.0 * v).round() as u8)
        .collect();
    let audio = AudioSamples::new_mono(Array1::from(x), sample_rate!(44100)).unwrap();
    let zc = audio.zero_crossings();
    assert!((zc as i64 - 880).abs() <= 2, "zero crossings = {zc}");
    // zero_crossing_rate is crossings per second (here the duration is 1 s).
    assert!((audio.zero_crossing_rate() - zc as f64).abs() < 1e-9);
}

// pin the documented sign convention of fft_alignment_lag.
#[test]
fn fft_alignment_lag_sign_convention() {
    let reference = noise(1000, 3);
    let mut query = vec![0.0; 25];
    query.extend_from_slice(&reference[..1000 - 25]);
    let r = AudioSamples::new_mono(Array1::from(reference), sample_rate!(8000)).unwrap();
    let q = AudioSamples::new_mono(Array1::from(query), sample_rate!(8000)).unwrap();
    // `query` delayed by 25 samples -> negative lag.
    assert_eq!(fft_alignment_lag(&r, &q, 100), Some(-25));
    // Swapping the roles: the query now leads -> positive lag.
    assert_eq!(fft_alignment_lag(&q, &r, 100), Some(25));
}

// bin frequency must be k * fs / N, also for odd N.
#[test]
fn spectral_frequencies_exact_for_odd_length() {
    for &n in &[9usize, 15, 101] {
        let fs = 1000.0;
        let x: Vec<f64> = (0..n)
            .map(|i| (2.0 * PI * 2.0 * i as f64 / n as f64).cos())
            .collect();
        let f_true = 2.0 * fs / n as f64;
        let audio = AudioSamples::new_mono(Array1::from(x), sample_rate!(1000)).unwrap();
        let c = audio.spectral_centroid(ChannelReduction::Error).unwrap();
        assert!((c - f_true).abs() < 1e-6, "N={n}: centroid {c} vs {f_true}");
        let r = audio
            .spectral_rolloff(0.5, ChannelReduction::Error)
            .unwrap();
        assert!((r - f_true).abs() < 1e-6, "N={n}: rolloff {r} vs {f_true}");
        let bw = audio.spectral_bandwidth(ChannelReduction::Error).unwrap();
        assert!(bw < 1e-3, "N={n}: bandwidth {bw}");
    }
}

// align_signals never applied a lag between the two signals.
#[test]
fn align_signals_finds_delay() {
    let signal = noise(2000, 11);
    let mut reference = vec![0.0; 50];
    reference.extend_from_slice(&signal[..2000 - 50]);
    let s = AudioSamples::new_mono(Array1::from(signal), sample_rate!(8000)).unwrap();
    let r = AudioSamples::new_mono(Array1::from(reference.clone()), sample_rate!(8000)).unwrap();
    let (aligned, offset) = align_signals(&r, &s).unwrap();
    assert_eq!(offset, 50);
    for i in 50..2000 {
        assert_eq!(aligned[i], reference[i]);
    }
}

// sub-harmonic picks and amplitude-dependent `None`.
#[test]
fn fundamental_frequency_sines_within_one_percent() {
    for &f in &[440.0, 220.0] {
        for &a in &[0.1, 0.8] {
            let x = sine(f, 44100.0, 22050, a);
            let audio = AudioSamples::new_mono(Array1::from(x), sample_rate!(44100)).unwrap();
            let est = detect_fundamental_frequency(&audio)
                .unwrap()
                .unwrap_or_else(|| panic!("no estimate for {f} Hz at amplitude {a}"));
            assert!(
                (est - f).abs() / f < 0.01,
                "{f} Hz at amplitude {a}: estimated {est} Hz"
            );
        }
    }
}

// candidate Nyquist frequencies were mapped to half their bin.
#[test]
fn spectrum_cutoff_maps_frequency_to_correct_bin() {
    let len = 4097; // n_fft = 8192, nyquist 8 kHz (fs = 16 kHz)
    let flat_below = |cut: f64| -> Vec<f64> {
        (0..len)
            .map(|i| {
                if i as f64 / (len - 1) as f64 * 8000.0 < cut {
                    1.0
                } else {
                    1e-6
                }
            })
            .collect()
    };
    let spec = flat_below(4000.0);
    let r = analyze_spectrum_for_cutoff(NonEmptySlice::from_slice(&spec).unwrap(), 8000.0);
    assert_eq!(r, Some(8000));
    let spec = flat_below(2000.0);
    let r = analyze_spectrum_for_cutoff(NonEmptySlice::from_slice(&spec).unwrap(), 8000.0);
    assert_eq!(r, None);
}

// detect_sample_rate documented Ok(None) when nothing is found but
// returned Err.
#[test]
fn detect_sample_rate_returns_none_without_cutoff() {
    let x = noise(8192, 5);
    let audio = AudioSamples::new_mono(Array1::from(x), sample_rate!(16000)).unwrap();
    assert_eq!(detect_sample_rate(&audio).unwrap(), None);
}
