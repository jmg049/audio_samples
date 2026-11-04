//! Regression tests for processing, editing, generation and resampling: each
//! result is checked against the operation's mathematical definition.
#![cfg(all(
    feature = "processing",
    feature = "editing",
    feature = "random-generation",
    feature = "iir-filtering",
    feature = "resampling"
))]

use std::num::NonZeroUsize;
use std::time::Duration;

use audio_samples::operations::types::{
    FadeCurve, NoiseColor, PerturbationConfig, PerturbationMethod, ResamplingQuality,
};
use audio_samples::utils::generation::am_signal;
use audio_samples::{
    AudioEditing, AudioIirFiltering, AudioProcessing, AudioSamples, AudioTypeConversion,
    NormalizationConfig, resample, sample_rate,
};
use ndarray::{Array1, array};
use non_empty_slice::NonEmptyVec;

fn mono_i16(v: &[i16]) -> AudioSamples<'static, i16> {
    AudioSamples::new_mono(Array1::from_vec(v.to_vec()), sample_rate!(44100)).unwrap()
}

fn to_vec<T: audio_samples::StandardSample>(a: &AudioSamples<'_, T>) -> Vec<T> {
    a.as_mono().unwrap().iter().copied().collect()
}

fn assert_close_i16(got: &[i16], want: &[i32], tol: i32) {
    assert_eq!(got.len(), want.len());
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        assert!(
            (i32::from(g) - w).abs() <= tol,
            "sample {i}: got {g}, want {w} (±{tol}); full output {got:?}"
        );
    }
}

// ---------------------------------------------------------------------------
// integer gains, peak normalisation and fades
// ---------------------------------------------------------------------------

#[test]
fn scale_i16_uses_fractional_gain() {
    let a = mono_i16(&[10000, -20000, 3000]);
    assert_close_i16(&to_vec(&a.scale(0.5)), &[5000, -10000, 1500], 1);
    assert_close_i16(&to_vec(&a.scale(0.99)), &[9900, -19800, 2970], 1);
    assert_close_i16(&to_vec(&a.scale(1.5)), &[15000, -30000, 4500], 1);
    assert_close_i16(&to_vec(&a.scale(-0.5)), &[-5000, 10000, -1500], 1);
    // Saturates rather than wrapping.
    assert_close_i16(&to_vec(&a.scale(4.0)), &[32767, -32768, 12000], 1);
}

#[test]
fn peak_normalise_i16_reaches_full_scale() {
    for start in [8000i16, 20000] {
        let a = mono_i16(&[start, -start / 2, 100]);
        let out = to_vec(&a.normalize(NormalizationConfig::peak(i16::MAX)).unwrap());
        assert!((i32::from(out[0]) - 32767).abs() <= 1, "{start}: {out:?}");
    }
}

#[test]
fn fade_in_i16_matches_f64_fade() {
    let n = 200;
    let ones_i16 =
        AudioSamples::new_mono(Array1::from_elem(n, 30000i16), sample_rate!(100)).unwrap();
    let ones_f64: AudioSamples<'static, f64> = ones_i16.to_format::<f64>();
    for curve in [
        FadeCurve::Linear,
        FadeCurve::Exponential,
        FadeCurve::Logarithmic,
        FadeCurve::SmoothStep,
    ] {
        let fi = to_vec(&ones_i16.fade_in(1.0, curve).unwrap());
        let ff = ones_f64.fade_in(1.0, curve).unwrap().to_format::<i16>();
        let ff = to_vec(&ff);
        let want: Vec<i32> = ff.iter().map(|&v| i32::from(v)).collect();
        assert_close_i16(&fi, &want, 1);
        // The fade really happens (not all zeros, not unchanged).
        assert_eq!(fi[0], 0);
        assert!(fi[50] > 1000 && fi[50] < 30000, "{curve:?}: {}", fi[50]);

        let fo = to_vec(&ones_i16.fade_out(1.0, curve).unwrap());
        let fo_f = to_vec(&ones_f64.fade_out(1.0, curve).unwrap().to_format::<i16>());
        let want: Vec<i32> = fo_f.iter().map(|&v| i32::from(v)).collect();
        assert_close_i16(&fo, &want, 1);
    }
}

#[test]
fn mix_i16_weights_are_fractional() {
    let a = mono_i16(&[10000, -10000]);
    let b = mono_i16(&[20000, 0]);
    let sources = NonEmptyVec::new(vec![a, b]).unwrap();
    let mixed = AudioSamples::mix(&sources, None).unwrap();
    assert_close_i16(&to_vec(&mixed), &[15000, -5000], 1);
}

// ---------------------------------------------------------------------------
// integer min-max normalisation
// ---------------------------------------------------------------------------

#[test]
fn min_max_i16() {
    let a = mono_i16(&[-20000, 0, 20000]);
    let out = to_vec(
        &a.normalize(NormalizationConfig::min_max(-1000, 1000))
            .unwrap(),
    );
    assert_eq!(out, vec![-1000, 0, 1000]);

    let b = mono_i16(&[-1000, 0, 1000]);
    let out = to_vec(
        &b.normalize(NormalizationConfig::min_max(i16::MIN, i16::MAX))
            .unwrap(),
    );
    assert_eq!(out[0], i16::MIN);
    assert_eq!(out[2], i16::MAX);
    assert!(out[1].abs() <= 1, "{out:?}");
}

// ---------------------------------------------------------------------------
// median normalisation subtracts the true median
// ---------------------------------------------------------------------------

#[test]
fn median_is_order_statistic() {
    let a = AudioSamples::new_mono(array![5.0f64, 9.0, 1.0, 7.0], sample_rate!(8000)).unwrap();
    let out = to_vec(&a.normalize(NormalizationConfig::median()).unwrap());
    assert_eq!(out, vec![-1.0, 3.0, -5.0, 1.0]);

    let b = AudioSamples::new_mono(array![100.0f64, 0.0, 1.0], sample_rate!(8000)).unwrap();
    let out = to_vec(&b.normalize(NormalizationConfig::median()).unwrap());
    assert_eq!(out, vec![99.0, -1.0, 0.0]);
}

// ---------------------------------------------------------------------------
// mu is a curve parameter, not a sample value
// ---------------------------------------------------------------------------

#[test]
fn mu_law_i16_matches_f64() {
    let x = mono_i16(&[16384, 1000, -16384, -3000]);
    let got = to_vec(&x.mu_compress(255).unwrap());
    let want_f = x
        .to_format::<f64>()
        .mu_compress(255.0)
        .unwrap()
        .to_format::<i16>();
    let want: Vec<i32> = to_vec(&want_f).iter().map(|&v| i32::from(v)).collect();
    assert_close_i16(&got, &want, 1);
    // Sign is preserved (it once gave -32768 for positive inputs).
    assert!(
        got[0] > 0 && got[1] > 0 && got[2] < 0 && got[3] < 0,
        "{got:?}"
    );

    let y = mono_i16(&[20000, -20000]);
    let got = to_vec(&y.mu_expand(255).unwrap());
    let want_f = y
        .to_format::<f64>()
        .mu_expand(255.0)
        .unwrap()
        .to_format::<i16>();
    let want: Vec<i32> = to_vec(&want_f).iter().map(|&v| i32::from(v)).collect();
    assert_close_i16(&got, &want, 1);
}

// ---------------------------------------------------------------------------
// GaussianNoise perturbation
// ---------------------------------------------------------------------------

fn sine(n: usize, fs: u32, f: f64, a: f64) -> AudioSamples<'static, f64> {
    let v: Vec<f64> = (0..n)
        .map(|i| a * (2.0 * std::f64::consts::PI * f * i as f64 / f64::from(fs)).sin())
        .collect();
    AudioSamples::new_mono(Array1::from_vec(v), std::num::NonZeroU32::new(fs).unwrap()).unwrap()
}

fn added_noise(color: NoiseColor, snr_db: f64, seed: u64) -> (Vec<f64>, Vec<f64>) {
    let clean = sine(4 * 44100, 44100, 440.0, 0.5);
    let cfg = PerturbationConfig::with_seed(
        PerturbationMethod::GaussianNoise {
            target_snr_db: snr_db,
            noise_color: color,
        },
        seed,
    );
    let noisy = clean.perturb(&cfg).unwrap();
    let c = to_vec(&clean);
    let y = to_vec(&noisy);
    let noise: Vec<f64> = y.iter().zip(&c).map(|(a, b)| a - b).collect();
    (c, noise)
}

fn power(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>() / v.len() as f64
}

fn band_power(noise: &[f64], lo: f64, hi: f64) -> f64 {
    let a = AudioSamples::new_mono(Array1::from_vec(noise.to_vec()), sample_rate!(44100)).unwrap();
    let f = a
        .butterworth_bandpass(NonZeroUsize::new(4).unwrap(), lo, hi)
        .unwrap();
    let v = to_vec(&f);
    power(&v[4410..]) // skip the filter start-up
}

#[test]
fn gaussian_noise_snr_kurtosis_and_colour() {
    for color in [NoiseColor::White, NoiseColor::Pink, NoiseColor::Brown] {
        for snr in [0.0, 20.0, 40.0] {
            let (c, n) = added_noise(color, snr, 7);
            let measured = 10.0 * (power(&c) / power(&n)).log10();
            assert!(
                (measured - snr).abs() < 0.5,
                "{color:?} target {snr} dB measured {measured} dB"
            );
        }
    }

    // White noise is Gaussian: excess kurtosis ~ 0 (uniform would be -1.2).
    let (_, n) = added_noise(NoiseColor::White, 20.0, 11);
    let m2 = power(&n);
    let m4 = n.iter().map(|x| x.powi(4)).sum::<f64>() / n.len() as f64;
    let excess = m4 / (m2 * m2) - 3.0;
    assert!(excess.abs() < 0.1, "excess kurtosis {excess}");

    // Spectral colour: power in [2k, 4k] vs [500, 1k] (two octaves apart,
    // each band one octave wide). White: +6 dB, pink: 0 dB, brown: -6 dB.
    for (color, want) in [
        (NoiseColor::White, 6.02),
        (NoiseColor::Pink, 0.0),
        (NoiseColor::Brown, -6.02),
    ] {
        let (_, n) = added_noise(color, 0.0, 3);
        let ratio = 10.0 * (band_power(&n, 2000.0, 4000.0) / band_power(&n, 500.0, 1000.0)).log10();
        assert!(
            (ratio - want).abs() < 1.5,
            "{color:?}: high/low band ratio {ratio} dB, want {want} dB"
        );
    }

    // Seeded runs are reproducible.
    let (_, a) = added_noise(NoiseColor::Brown, 10.0, 5);
    let (_, b) = added_noise(NoiseColor::Brown, 10.0, 5);
    assert_eq!(a, b);
}

// ---------------------------------------------------------------------------
// standard AM definition
// ---------------------------------------------------------------------------

/// Amplitude of the component at `f` (integer number of cycles in the buffer).
fn tone_amplitude(x: &[f64], fs: f64, f: f64) -> f64 {
    let (mut re, mut im) = (0.0, 0.0);
    for (i, &v) in x.iter().enumerate() {
        let ph = 2.0 * std::f64::consts::PI * f * i as f64 / fs;
        re += v * ph.cos();
        im += v * ph.sin();
    }
    2.0 * (re * re + im * im).sqrt() / x.len() as f64
}

#[test]
fn am_signal_is_standard_am() {
    let fs = 8000.0;
    let a = 0.8;
    for m in [0.0, 0.5, 1.0] {
        let x = am_signal::<f64>(
            1000.0,
            100.0,
            m,
            Duration::from_secs(1),
            sample_rate!(8000),
            a,
        );
        let v = to_vec(&x);
        let carrier = tone_amplitude(&v, fs, 1000.0);
        let lower = tone_amplitude(&v, fs, 900.0);
        let upper = tone_amplitude(&v, fs, 1100.0);
        assert!(
            (carrier - a / (1.0 + m)).abs() < 1e-6,
            "m={m}: carrier {carrier}"
        );
        let side = a * m / (2.0 * (1.0 + m));
        assert!((lower - side).abs() < 1e-6, "m={m}: lower {lower}");
        assert!((upper - side).abs() < 1e-6, "m={m}: upper {upper}");
        let peak = v.iter().fold(0.0f64, |p, s| p.max(s.abs()));
        assert!(peak <= a + 1e-12 && peak > 0.99 * a, "m={m}: peak {peak}");
    }
}

// ---------------------------------------------------------------------------
// sinc resampling is time-aligned and keeps the tail
// ---------------------------------------------------------------------------

fn argmax_abs(v: &[f32]) -> usize {
    (0..v.len())
        .max_by(|&a, &b| v[a].abs().total_cmp(&v[b].abs()))
        .unwrap()
}

#[test]
fn impulse_lands_at_scaled_time() {
    let n_in = 20000usize;
    let rates: [(u32, u32); 4] = [
        (44100, 48000),
        (48000, 44100),
        (44100, 22050),
        (16000, 44100),
    ];
    for quality in [ResamplingQuality::Medium, ResamplingQuality::High] {
        for (fs_in, fs_out) in rates {
            for &pos in &[10000usize, n_in - 50] {
                let mut x = Array1::<f32>::zeros(n_in);
                x[pos] = 1.0;
                let a =
                    AudioSamples::new_mono(x, std::num::NonZeroU32::new(fs_in).unwrap()).unwrap();
                let y = resample(&a, std::num::NonZeroU32::new(fs_out).unwrap(), quality).unwrap();
                let v = to_vec(&y);
                assert_eq!(v.len(), (n_in * fs_out as usize).div_ceil(fs_in as usize));
                let want = (pos as f64 * f64::from(fs_out) / f64::from(fs_in)).round() as i64;
                let got = argmax_abs(&v) as i64;
                assert!(
                    (got - want).abs() <= 1,
                    "{quality:?} {fs_in}->{fs_out} impulse at {pos}: peak {got}, want {want}"
                );
                assert!(
                    v[got as usize] > 0.3,
                    "{quality:?} {fs_in}->{fs_out}: peak amplitude {}",
                    v[got as usize]
                );
            }
        }
    }
}

#[test]
fn sine_matches_analytic_after_resampling() {
    let f = 440.0;
    let x: Vec<f32> = (0..44100)
        .map(|i| (2.0 * std::f64::consts::PI * f * i as f64 / 44100.0).sin() as f32)
        .collect();
    let a = AudioSamples::new_mono(Array1::from_vec(x), sample_rate!(44100)).unwrap();
    for quality in [ResamplingQuality::Medium, ResamplingQuality::High] {
        let y = to_vec(&resample(&a, sample_rate!(48000), quality).unwrap());
        let reference: Vec<f64> = (0..y.len())
            .map(|m| (2.0 * std::f64::consts::PI * f * m as f64 / 48000.0).sin())
            .collect();
        // Cross-correlate over a window away from the edges.
        let (lo, hi) = (2000usize, y.len() - 2000);
        let best_lag = (-20i64..=20)
            .max_by(|&p, &q| {
                let c = |lag: i64| -> f64 {
                    (lo..hi)
                        .map(|m| f64::from(y[m]) * reference[(m as i64 + lag) as usize])
                        .sum()
                };
                c(p).total_cmp(&c(q))
            })
            .unwrap();
        assert_eq!(best_lag, 0, "{quality:?}: best lag {best_lag}");
        // Fractional residual delay from the quadrature correlation:
        // y ≈ sin(ω m − φ), delay = φ/ω output samples. The delay removed is
        // a whole number of frames, so up to half a sample remains (before
        // the delay fix it was ~69 / ~139 samples).
        let w = 2.0 * std::f64::consts::PI * f / 48000.0;
        let (mut s_acc, mut c_acc) = (0.0, 0.0);
        for (m, &ym) in y.iter().enumerate().take(hi).skip(lo) {
            s_acc += f64::from(ym) * (w * m as f64).sin();
            c_acc += f64::from(ym) * (w * m as f64).cos();
        }
        let delay = (-c_acc).atan2(s_acc) / w;
        assert!(
            delay.abs() < 0.6,
            "{quality:?}: residual delay {delay} samples"
        );
        // One output sample of 440 Hz at 48 kHz is a phase step of 0.058 rad.
        let max_err = (lo..hi)
            .map(|m| (f64::from(y[m]) - reference[m]).abs())
            .fold(0.0, f64::max);
        assert!(max_err < 0.035, "{quality:?}: max error {max_err}");
    }
}
