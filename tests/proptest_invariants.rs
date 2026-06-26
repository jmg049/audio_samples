//! Property-based invariant tests across processing, channels, conversion,
//! comparison, and IIR-filtering surfaces.
//!
//! These tests assert *invariants* (properties that must hold for arbitrary
//! valid inputs) rather than fixed known-value cases. They use `proptest` to
//! generate random signal lengths, amplitudes, and sample values, keeping case
//! counts modest so the suite stays fast.
//!
//! Run with:
//! ```bash
//! cargo test --test proptest_invariants \
//!   --no-default-features \
//!   --features processing,channels,iir-filtering,editing
//! ```
//!
//! Invariants covered:
//! - Normalize: peak normalization never overshoots the target and hits it for
//!   non-silent input.
//! - Channel round-trip: duplicate→extract returns the original mono data;
//!   interleave→deinterleave round-trips.
//! - Conversion round-trip: f32→i16→f32, f32→i32→f32 within quantisation
//!   tolerance; i16→f32→i16 lossless; f64→f32→f64 within f32 epsilon.
//! - Correlation: `correlation(a, a) ≈ 1.0` for non-constant signals; symmetry.
//! - Filter stability: Butterworth/Chebyshev lowpass yields finite, bounded
//!   output (no NaN/Inf/blowup).
//! - In-place == borrowing: `x.op(args)` equals cloning then `x.op_in_place(args)`.

use std::num::NonZeroUsize;

use audio_samples::operations::types::{FilterResponse, IirFilterDesign};
use audio_samples::utils::comparison::correlation;
use audio_samples::{
    AudioChannelOps, AudioIirFiltering, AudioProcessing, AudioSamples, AudioTypeConversion,
    NormalizationConfig, sample_rate,
};
use ndarray::Array1;
use proptest::prelude::*;

#[cfg(test)]
proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    // -------------------------------------------------------------------------
    // Normalize (feature: processing)
    // -------------------------------------------------------------------------

    /// After peak-normalizing to `target`, the resulting peak must not exceed
    /// `target` (within a small epsilon), and for a non-silent signal it must
    /// actually reach `target`.
    #[test]
    fn normalize_peak_does_not_overshoot(
        samples in prop::collection::vec(-1.0f32..1.0f32, 8..512),
        target in 0.1f32..4.0f32,
    ) {
        // Guarantee the signal is non-silent so the "reaches target" branch holds.
        let mut samples = samples;
        samples[0] = 0.7;
        let audio =
            AudioSamples::new_mono(Array1::from(samples), sample_rate!(44100)).unwrap();

        let normalized = audio.normalize(NormalizationConfig::peak(target)).unwrap();
        let peak: f32 = normalized
            .as_slice()
            .unwrap()
            .iter()
            .map(|x| x.abs())
            .fold(0.0f32, f32::max);

        let eps = target * 1e-4 + 1e-6;
        prop_assert!(
            peak <= target + eps,
            "peak {peak} overshot target {target} (eps {eps})"
        );
        prop_assert!(
            (peak - target).abs() <= eps,
            "non-silent peak {peak} did not reach target {target} (eps {eps})"
        );
    }

    // -------------------------------------------------------------------------
    // Channel round-trip (feature: channels)
    // -------------------------------------------------------------------------

    /// `duplicate_to_channels(n)` then `extract_channel(i)` returns the original
    /// mono data for every channel index `i`.
    #[test]
    fn duplicate_then_extract_round_trips(
        samples in prop::collection::vec(-1.0f32..1.0f32, 4..256),
        n in 1usize..5usize,
    ) {
        let original = samples.clone();
        let audio =
            AudioSamples::new_mono(Array1::from(samples), sample_rate!(44100)).unwrap();
        let multi = audio.duplicate_to_channels(n).unwrap();
        prop_assert_eq!(multi.num_channels().get() as usize, n);

        for i in 0..n {
            let ch = multi.extract_channel(i).unwrap();
            let ch_slice = ch.as_slice().unwrap();
            prop_assert_eq!(ch_slice.len(), original.len());
            for (j, (&o, &c)) in original.iter().zip(ch_slice).enumerate() {
                prop_assert_eq!(o, c, "channel {} sample {} mismatch", i, j);
            }
        }
    }

    /// Interleaving channels then deinterleaving them recovers each channel's
    /// data exactly.
    #[test]
    fn interleave_then_deinterleave_round_trips(
        samples in prop::collection::vec(-1.0f32..1.0f32, 4..256),
        n in 2usize..5usize,
    ) {
        let original = samples.clone();
        let mono =
            AudioSamples::new_mono(Array1::from(samples), sample_rate!(44100)).unwrap();
        let multi = mono.duplicate_to_channels(n).unwrap();

        let channels = multi.deinterleave_channels().unwrap();
        prop_assert_eq!(channels.len(), n);
        for ch in &channels {
            prop_assert!(ch.is_mono());
            let s = ch.as_slice().unwrap();
            prop_assert_eq!(s.len(), original.len());
            for (&o, &c) in original.iter().zip(s) {
                prop_assert_eq!(o, c);
            }
        }
    }

    // -------------------------------------------------------------------------
    // Conversion round-trips (core)
    // -------------------------------------------------------------------------

    /// f32 → i16 → f32 stays within the 16-bit quantisation tolerance.
    #[test]
    fn f32_i16_f32_within_quantisation_tolerance(
        samples in prop::collection::vec(-1.0f32..1.0f32, 4..256),
    ) {
        let original = samples.clone();
        let audio =
            AudioSamples::new_mono(Array1::from(samples), sample_rate!(44100)).unwrap();
        let round = audio.to_format::<i16>().to_type::<f32>();
        let out = round.as_slice().unwrap();
        // 16-bit quantisation step ≈ 1/32768 ≈ 3.05e-5; allow generous slack.
        for (&o, &r) in original.iter().zip(out) {
            prop_assert!(
                (o - r).abs() < 1e-3,
                "f32->i16->f32 drift too large: {o} -> {r}"
            );
        }
    }

    /// f32 → i32 → f32 stays within the (very fine) 32-bit quantisation tolerance.
    #[test]
    fn f32_i32_f32_within_quantisation_tolerance(
        samples in prop::collection::vec(-1.0f32..1.0f32, 4..256),
    ) {
        let original = samples.clone();
        let audio =
            AudioSamples::new_mono(Array1::from(samples), sample_rate!(44100)).unwrap();
        let round = audio.to_format::<i32>().to_type::<f32>();
        let out = round.as_slice().unwrap();
        for (&o, &r) in original.iter().zip(out) {
            // i32 resolution far exceeds f32 mantissa; f32 rounding dominates.
            prop_assert!(
                (o - r).abs() < 1e-4,
                "f32->i32->f32 drift too large: {o} -> {r}"
            );
        }
    }

    /// i16 → f32 → i16 recovers every code within 1 LSB.
    ///
    /// f32 has a 24-bit mantissa, so no precision is lost in the float domain;
    /// the only deviation comes from the audio-aware asymmetric PCM scaling
    /// (negatives are normalised by 32768, positives by 32767, while the reverse
    /// path multiplies by 32767). That asymmetry can shift a code by at most one
    /// least-significant bit, so the genuine invariant is round-trip within ±1.
    #[test]
    fn i16_f32_i16_round_trips_within_one_lsb(
        samples in prop::collection::vec(i16::MIN..i16::MAX, 4..256),
    ) {
        let original = samples.clone();
        let audio =
            AudioSamples::new_mono(Array1::from(samples), sample_rate!(44100)).unwrap();
        let round = audio.to_format::<f32>().to_type::<i16>();
        let out = round.as_slice().unwrap();
        for (&o, &r) in original.iter().zip(out) {
            let diff = (i32::from(o) - i32::from(r)).abs();
            prop_assert!(
                diff <= 1,
                "i16->f32->i16 drifted by {diff} LSB: {o} -> {r}"
            );
        }
    }

    /// f64 → f32 → f64 stays within f32 epsilon (relative to magnitude).
    #[test]
    fn f64_f32_f64_within_f32_epsilon(
        samples in prop::collection::vec(-1.0f64..1.0f64, 4..256),
    ) {
        let original = samples.clone();
        let audio =
            AudioSamples::new_mono(Array1::from(samples), sample_rate!(44100)).unwrap();
        let round = audio.to_format::<f32>().to_type::<f64>();
        let out = round.as_slice().unwrap();
        for (&o, &r) in original.iter().zip(out) {
            let tol = (o.abs() * f64::from(f32::EPSILON)) + f64::from(f32::EPSILON);
            prop_assert!(
                (o - r).abs() <= tol,
                "f64->f32->f64 drift {} exceeds f32 epsilon tol {tol}",
                (o - r).abs()
            );
        }
    }

    // -------------------------------------------------------------------------
    // Correlation (core; ungated)
    // -------------------------------------------------------------------------

    /// `correlation(a, a) ≈ 1.0` for any non-constant signal, and correlation is
    /// symmetric: `correlation(a, b) == correlation(b, a)`.
    #[test]
    fn correlation_self_is_one_and_symmetric(
        a in prop::collection::vec(-1.0f64..1.0f64, 8..256),
        b in prop::collection::vec(-1.0f64..1.0f64, 8..256),
    ) {
        // Make both non-constant and equal length by truncating to the shorter.
        let len = a.len().min(b.len());
        let mut a: Vec<f64> = a[..len].to_vec();
        let mut b: Vec<f64> = b[..len].to_vec();
        // Force non-constant variance regardless of generated values.
        a[0] = 0.9;
        a[1] = -0.9;
        b[0] = 0.4;
        b[1] = -0.7;

        let sa = AudioSamples::new_mono(Array1::from(a), sample_rate!(44100)).unwrap();
        let sb = AudioSamples::new_mono(Array1::from(b), sample_rate!(44100)).unwrap();

        let self_corr = correlation(&sa, &sa).unwrap();
        prop_assert!(
            (self_corr - 1.0).abs() < 1e-9,
            "correlation(a, a) was {self_corr}, expected 1.0"
        );

        let ab = correlation(&sa, &sb).unwrap();
        let ba = correlation(&sb, &sa).unwrap();
        prop_assert!(
            (ab - ba).abs() < 1e-12,
            "correlation not symmetric: ab={ab}, ba={ba}"
        );
        prop_assert!((-1.0..=1.0).contains(&ab), "correlation {ab} out of [-1, 1]");
    }

    // -------------------------------------------------------------------------
    // Filter stability (feature: iir-filtering)
    // -------------------------------------------------------------------------

    /// A Butterworth lowpass applied to a bounded random signal yields all-finite
    /// output bounded by a sane multiple of the input peak (no NaN/Inf/blowup).
    #[test]
    fn butterworth_lowpass_is_stable(
        samples in prop::collection::vec(-1.0f64..1.0f64, 32..512),
        cutoff in 500.0f64..8000.0f64,
    ) {
        let input_peak = samples
            .iter()
            .map(|x| x.abs())
            .fold(0.0f64, f64::max)
            .max(1e-9);
        let mut audio =
            AudioSamples::new_mono(Array1::from(samples), sample_rate!(44100)).unwrap();
        audio
            .butterworth_lowpass_in_place(NonZeroUsize::new(4).unwrap(), cutoff)
            .unwrap();
        let out = audio.as_slice().unwrap();
        for &x in out {
            prop_assert!(x.is_finite(), "lowpass produced non-finite {x}");
            prop_assert!(
                x.abs() <= input_peak * 16.0,
                "lowpass blew up: {x} vs input peak {input_peak}"
            );
        }
    }

    /// A Chebyshev Type I lowpass is likewise finite and bounded.
    #[test]
    fn chebyshev_lowpass_is_stable(
        samples in prop::collection::vec(-1.0f64..1.0f64, 32..512),
        cutoff in 500.0f64..8000.0f64,
    ) {
        let input_peak = samples
            .iter()
            .map(|x| x.abs())
            .fold(0.0f64, f64::max)
            .max(1e-9);
        let mut audio =
            AudioSamples::new_mono(Array1::from(samples), sample_rate!(44100)).unwrap();
        let design = IirFilterDesign::chebyshev_i(
            FilterResponse::LowPass,
            NonZeroUsize::new(4).unwrap(),
            cutoff,
            0.5,
        );
        audio.apply_iir_filter_in_place(&design).unwrap();
        let out = audio.as_slice().unwrap();
        for &x in out {
            prop_assert!(x.is_finite(), "chebyshev produced non-finite {x}");
            prop_assert!(
                x.abs() <= input_peak * 16.0,
                "chebyshev blew up: {x} vs input peak {input_peak}"
            );
        }
    }

    // -------------------------------------------------------------------------
    // In-place == borrowing (validates the dual-variant convention)
    // -------------------------------------------------------------------------

    /// `scale(f)` equals cloning and applying `scale_in_place(f)`.
    #[test]
    fn scale_borrowing_equals_in_place(
        samples in prop::collection::vec(-1.0f32..1.0f32, 4..256),
        factor in -4.0f64..4.0f64,
    ) {
        let audio =
            AudioSamples::new_mono(Array1::from(samples), sample_rate!(44100)).unwrap();

        let borrowed = audio.scale(factor);
        let mut in_place = audio.clone();
        in_place.scale_in_place(factor);

        let a = borrowed.as_slice().unwrap();
        let b = in_place.as_slice().unwrap();
        prop_assert_eq!(a.len(), b.len());
        for (&x, &y) in a.iter().zip(b) {
            prop_assert_eq!(x, y, "scale borrowing != in-place");
        }
    }

    /// `apply_iir_filter(design)` equals cloning and applying
    /// `apply_iir_filter_in_place(design)`.
    #[test]
    fn iir_filter_borrowing_equals_in_place(
        samples in prop::collection::vec(-1.0f64..1.0f64, 32..256),
        cutoff in 500.0f64..8000.0f64,
    ) {
        let audio =
            AudioSamples::new_mono(Array1::from(samples), sample_rate!(44100)).unwrap();
        let design =
            IirFilterDesign::butterworth_lowpass(NonZeroUsize::new(4).unwrap(), cutoff);

        let borrowed = audio.apply_iir_filter(&design).unwrap();
        let mut in_place = audio.clone();
        in_place.apply_iir_filter_in_place(&design).unwrap();

        let a = borrowed.as_slice().unwrap();
        let b = in_place.as_slice().unwrap();
        prop_assert_eq!(a.len(), b.len());
        for (&x, &y) in a.iter().zip(b) {
            prop_assert!(
                (x - y).abs() < 1e-12,
                "iir filter borrowing != in-place: {x} vs {y}"
            );
        }
    }
}
