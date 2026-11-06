//! Regression tests for Welch PSD scaling, the HPSS bin count, the range of
//! `frequency_to_note`, Levinson-Durbin and the integer-input unit consistency
//! of the spectrogram methods.
#![cfg(all(
    feature = "transforms",
    feature = "decomposition",
    feature = "opus-codec"
))]

use audio_samples::codecs::opus::{levinson_durbin, lpc_analysis};
use audio_samples::operations::hpss::HpssConfig;
use audio_samples::utils::audio_math::frequency_to_note;
use audio_samples::{AudioSamples, AudioTransforms, nzu, sample_rate};
use ndarray::Array1;
use spectrograms::{SpectrogramParams, StftParams, WindowType};

/// Deterministic uniform noise in [-1, 1): 64-bit LCG (Knuth MMIX constants),
/// top 53 bits of the state. Reproduced bit-for-bit in Python to generate the
/// SciPy reference values below.
fn lcg_noise(n: usize) -> Vec<f64> {
    let mut state: u64 = 0x2545_F491_4F6C_DD1D;
    (0..n)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((state >> 11) as f64 / (1u64 << 53) as f64).mul_add(2.0, -1.0)
        })
        .collect()
}

fn rel_err(a: f64, b: f64) -> f64 {
    (a - b).abs() / b.abs()
}

// ── Welch PSD density scaling ─────────────────────────────────────────

/// Reference values generated with SciPy 1.18.1:
///
/// ```python
/// x = lcg_noise(8192)  # same generator as above
/// f, P = scipy.signal.welch(x, fs=1000.0, window=scipy.signal.windows.hann(N, sym=True),
///                           nperseg=N, noverlap=N - N//2, detrend=False, scaling='density')
/// ```
///
/// for N = 256 (noverlap 128) and N = 255 (hop floor(0.5·255) = 127, noverlap 128).
#[test]
fn psd_matches_scipy_welch_density() {
    let x = lcg_noise(8192);
    let audio = AudioSamples::new_mono(Array1::from(x), sample_rate!(1000)).unwrap();

    // (window_size, [(bin, scipy density)], scipy ∫P df)
    type Case = (usize, [(usize, f64); 4], f64);
    let cases: [Case; 2] = [
        (
            256,
            [
                (0, 0.000_263_698_825_332_907_8),
                (1, 0.000_526_980_232_954_885_2),
                (64, 0.000_745_215_310_866_470_5),
                (128, 0.000_280_892_186_876_878_1),
            ],
            0.330_076_974_806_427_3,
        ),
        (
            255,
            [
                (0, 0.000_269_018_708_816_698_7),
                (1, 0.000_519_208_607_549_311_5),
                (64, 0.000_653_250_093_088_274_6),
                (127, 0.000_647_809_506_839_111_1),
            ],
            0.331_114_768_178_404_46,
        ),
    ];

    for (n, bins, integral) in cases {
        let psd = audio
            .power_spectral_density(std::num::NonZeroUsize::new(n).unwrap(), 0.5)
            .unwrap();
        assert_eq!(psd.density().len(), n / 2 + 1);
        for (k, want) in bins {
            let got = psd.density()[k];
            assert!(
                rel_err(got, want) < 1e-9,
                "N={n} bin {k}: got {got:e}, scipy {want:e}"
            );
        }
        let df = 1000.0 / n as f64;
        let got_integral: f64 = psd.density().iter().sum::<f64>() * df;
        assert!(
            rel_err(got_integral, integral) < 1e-9,
            "N={n}: integral {got_integral} vs scipy {integral}"
        );
    }
}

/// Parseval: for white noise, ∫ PSD df ≈ mean square of the signal.
#[test]
fn psd_integral_equals_mean_square_for_white_noise() {
    let x = lcg_noise(1 << 15);
    let mean_square = x.iter().map(|v| v * v).sum::<f64>() / x.len() as f64;
    let fs = 8000.0;
    let audio = AudioSamples::new_mono(Array1::from(x), sample_rate!(8000)).unwrap();
    let psd = audio.power_spectral_density(nzu!(512), 0.5).unwrap();
    let df = fs / 512.0;
    let integral: f64 = psd.density().iter().sum::<f64>() * df;
    assert!(
        rel_err(integral, mean_square) < 0.03,
        "∫PSD df = {integral}, mean square = {mean_square}"
    );
}

#[test]
fn psd_rejects_single_sample_window() {
    let audio = AudioSamples::new_mono(Array1::from(lcg_noise(64)), sample_rate!(1000)).unwrap();
    assert!(audio.power_spectral_density(nzu!(1), 0.0).is_err());
}

// ── Integer input: spectrogram methods use audio-aware scaling like `stft` ────

/// `linear_power_spectrogram` (and the other spectrogram methods) used a raw numeric
/// cast, so an `i16` signal gave values 32768² times larger than `stft` of the same
/// signal (and `u8` input gained a huge DC component). All spectral methods now use
/// the same audio-aware conversion to [-1, 1].
#[test]
fn integer_input_spectrogram_matches_stft_power() {
    let x: Vec<i16> = lcg_noise(4096)
        .iter()
        .map(|v| (v * 16_000.0) as i16)
        .collect();
    let audio = AudioSamples::new_mono(Array1::from(x), sample_rate!(16000)).unwrap();
    let stft_params = StftParams::new(nzu!(256), nzu!(64), WindowType::Hanning, true).unwrap();
    let stft = audio.stft(&stft_params).unwrap();
    let params = SpectrogramParams::new(stft_params, 16_000.0).unwrap();
    let spec = audio.linear_power_spectrogram(&params).unwrap();
    let power = spec.data();
    assert_eq!(power.dim(), stft.data.dim());
    let max_power = power.iter().cloned().fold(0.0_f64, f64::max);
    for (p, z) in power.iter().zip(stft.data.iter()) {
        assert!(
            (p - z.norm_sqr()).abs() <= 1e-9 * max_power,
            "power {p} vs |stft|² {}",
            z.norm_sqr()
        );
    }
}

// ── HPSS bin count ─────────────────────────────────────────────────────

#[test]
fn hpss_num_freq_bins_matches_stft_rows() {
    for n_fft in [5usize, 8, 9] {
        let stft_params = StftParams::new(
            std::num::NonZeroUsize::new(n_fft).unwrap(),
            nzu!(2),
            WindowType::Hanning,
            true,
        )
        .unwrap();
        let cfg = HpssConfig::new(stft_params.clone(), 3, 3, 0.0);
        let audio =
            AudioSamples::new_mono(Array1::from(lcg_noise(64)), sample_rate!(1000)).unwrap();
        let rows = audio.stft(&stft_params).unwrap().data.nrows();
        assert_eq!(cfg.num_freq_bins().get(), rows, "n_fft = {n_fft}");
        assert_eq!(rows, n_fft / 2 + 1);
    }
}

// ── frequency_to_note range ───────────────────────────────────────────

#[test]
fn frequency_to_note_rejects_out_of_range_midi() {
    // 5 Hz is MIDI ≈ -8.5: below MIDI 0, an error rather than "C-1".
    assert!(frequency_to_note(5.0).is_err());
    // 20 kHz is MIDI ≈ 135.1: above 127.
    assert!(frequency_to_note(20_000.0).is_err());
    // Range edges: MIDI 0 (8.18 Hz) and MIDI 127 (12543.85 Hz) are accepted.
    assert_eq!(frequency_to_note(8.1758).unwrap().0, "C-1");
    assert_eq!(frequency_to_note(12_543.85).unwrap().0, "G9");
    let (note, cents) = frequency_to_note(440.0).unwrap();
    assert_eq!(note, "A4");
    assert!(cents.abs() < 1e-9);
}

// ── Levinson-Durbin returns the final-order solution ───────────────────

/// AR(2) process x[n] = 1.3·x[n−1] − 0.8·x[n−2] + e[n], i.e. A(z) = 1 − 1.3 z⁻¹ + 0.8 z⁻²,
/// so the crate's convention (e[n] = x[n] + Σ a[k] x[n−1−k]) expects a = [−1.3, 0.8].
#[test]
fn levinson_durbin_recovers_exact_ar2() {
    let (phi1, phi2) = (1.3_f64, -0.8_f64);
    let rho1 = phi1 / (1.0 - phi2);
    let rho2 = phi1.mul_add(rho1, phi2);
    let lpc = levinson_durbin(&[1.0, rho1, rho2], 2).unwrap();
    assert_eq!(lpc.coeffs.len(), 2);
    assert!(
        (f64::from(lpc.coeffs[0]) + 1.3).abs() < 1e-4,
        "{:?}",
        lpc.coeffs
    );
    assert!(
        (f64::from(lpc.coeffs[1]) - 0.8).abs() < 1e-4,
        "{:?}",
        lpc.coeffs
    );

    // Order 3 on an exact AR(2) autocorrelation: third coefficient ≈ 0 and the
    // first two still match (not the order-2 solution padded with 0, nor
    // [k1, 0] at order 2).
    let rho3 = phi1.mul_add(rho2, phi2 * rho1);
    let lpc3 = levinson_durbin(&[1.0, rho1, rho2, rho3], 3).unwrap();
    assert!(
        (f64::from(lpc3.coeffs[0]) + 1.3).abs() < 1e-3,
        "{:?}",
        lpc3.coeffs
    );
    assert!(
        (f64::from(lpc3.coeffs[1]) - 0.8).abs() < 1e-3,
        "{:?}",
        lpc3.coeffs
    );
    assert!(f64::from(lpc3.coeffs[2]).abs() < 1e-3, "{:?}", lpc3.coeffs);
}

#[test]
fn lpc_analysis_recovers_ar2_process() {
    let e = lcg_noise(20_000);
    let mut x = vec![0.0f64; e.len()];
    for n in 0..e.len() {
        let x1 = if n >= 1 { x[n - 1] } else { 0.0 };
        let x2 = if n >= 2 { x[n - 2] } else { 0.0 };
        x[n] = 1.3f64.mul_add(x1, (-0.8f64).mul_add(x2, e[n]));
    }
    let x32: Vec<f32> = x.iter().map(|&v| v as f32).collect();
    let lpc = lpc_analysis(&x32, 2);
    assert!(
        (f64::from(lpc.coeffs[0]) + 1.3).abs() < 0.05,
        "{:?}",
        lpc.coeffs
    );
    assert!(
        (f64::from(lpc.coeffs[1]) - 0.8).abs() < 0.05,
        "{:?}",
        lpc.coeffs
    );
}
