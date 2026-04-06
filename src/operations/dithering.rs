//! Dithering and bit-depth reduction operations for [`AudioSamples`].
//!
//! This module implements the [`AudioDithering`] trait, providing TPDF dithering
//! with optional noise shaping and in-place bit-depth reduction (`requantize`).
//!
//! Dithering improves the perceived quality of audio that will be reduced to a
//! lower bit depth by replacing deterministic quantisation distortion with
//! spectrally controlled stochastic noise that is less audible.  The recommended
//! workflow is to call [`dither`][AudioDithering::dither] immediately before
//! [`requantize`][AudioDithering::requantize].
//!
//! The dither noise is generated using the `rand` crate, which is enabled
//! transitively by the `dithering` feature flag via `random-generation`.
//!
//! # Noise amplitude
//!
//! For integer sample types (`u8`, `i16`, `I24`, `i32`) the noise amplitude is
//! set to 1 LSB of the native representation (i.e. 1 / MAX in the normalised
//! `[−1, 1]` domain).  For floating-point types (`f32`, `f64`) a fixed
//! 24-bit-equivalent noise floor (≈ 2⁻²³) is used so that the dither is
//! effective when combined with a subsequent `requantize(≤24)` call but
//! inaudible at full float precision.
//!
//! # Noise shaping
//!
//! - [`NoiseShape::Flat`]: pure TPDF — two independent uniform random values are
//!   subtracted to produce a triangular distribution.
//! - [`NoiseShape::FWeighted`]: first-order high-pass shaped TPDF — the raw noise
//!   is filtered with the transfer function `H(z) = 1 − 0.5·z⁻¹`, concentrating
//!   noise energy towards higher frequencies where the ear is less sensitive.
//!   State is maintained per channel so that multi-channel audio is shaped
//!   independently.
//!
//! [`AudioSamples`]: crate::AudioSamples

use crate::operations::traits::AudioDithering;
use crate::operations::types::NoiseShape;
use crate::repr::AudioData;
use crate::traits::StandardSample;
use crate::{AudioSampleError, AudioSampleResult, AudioSamples, ParameterError};

use ndarray::Axis;

impl<T> AudioDithering for AudioSamples<'_, T>
where
    T: StandardSample,
{
    fn dither(mut self, shape: NoiseShape) -> Self {
        // Compute 1-LSB amplitude in the normalised [−1, 1] domain.
        //
        // For integer types the raw MAX cast to f64 gives the number of
        // representable positive levels (e.g. 32 767 for i16), so one LSB
        // normalised = 1 / MAX.
        //
        // For float types (f32::MAX or f64::MAX ≫ 1) the concept of "1 LSB"
        // depends on the value; we use an approximation of 2⁻²³ (~24-bit
        // noise floor) as a practical default.
        let max_raw: f64 = T::MAX.cast_into();
        let lsb_norm: f64 = if max_raw > 1.0 {
            1.0 / max_raw
        } else {
            // ~24-bit equivalent for floating-point containers (2^-23)
            1.192_093_0e-7_f64
        };

        match shape {
            NoiseShape::Flat => {
                // Pure TPDF: u1 − u2 gives triangular distribution on (−1, 1).
                // Uses `mapv_inplace` for cache-friendly traversal.
                self.data.mapv_inplace(|sample| {
                    let tpdf: f64 = rand::random::<f64>() - rand::random::<f64>();
                    let s: f64 = sample.convert_to();
                    T::convert_from(tpdf.mul_add(lsb_norm, s))
                });
            }
            NoiseShape::FWeighted => {
                // First-order high-pass shaped TPDF.
                //
                // Each channel maintains independent feedback state so that
                // inter-channel correlation does not introduce spectral artefacts.
                // Transfer function: H(z) = 1 − 0.5·z⁻¹ (single-pole HP).
                let apply_fweighted = |iter: &mut dyn Iterator<Item = &mut T>| {
                    let mut prev: f64 = 0.0;
                    for sample in iter {
                        let tpdf: f64 = rand::random::<f64>() - rand::random::<f64>();
                        let shaped = 0.5_f64.mul_add(-prev, tpdf);
                        prev = tpdf;
                        let s: f64 = (*sample).convert_to();
                        *sample = T::convert_from(shaped.mul_add(lsb_norm, s));
                    }
                };

                match &mut self.data {
                    AudioData::Mono(arr) => {
                        apply_fweighted(&mut arr.iter_mut() as &mut dyn Iterator<Item = &mut T>);
                    }
                    AudioData::Multi(arr) => {
                        for mut channel in arr.axis_iter_mut(Axis(0)) {
                            apply_fweighted(
                                &mut channel.iter_mut() as &mut dyn Iterator<Item = &mut T>,
                            );
                        }
                    }
                }
            }
        }

        self
    }

    fn requantize(mut self, bits: u32) -> AudioSampleResult<Self> {
        if bits == 0 || bits > 32 {
            return Err(AudioSampleError::Parameter(ParameterError::out_of_range(
                "bits",
                bits.to_string(),
                "1",
                "32",
                "bit depth must be in the range [1, 32]",
            )));
        }

        // Quantise each sample to a grid of 2^(bits−1) levels per polarity.
        //
        // All arithmetic is performed in the normalised f64 domain:
        //   1. sample.convert_to() → f64 in [−1, 1] (or wider for float types)
        //   2. snap to nearest grid point
        //   3. T::convert_from(quantized_f64) → back to T
        let levels = (1u64 << (bits - 1)) as f64;
        self.data.mapv_inplace(|sample| {
            let s: f64 = sample.convert_to();
            let quantized = (s * levels).round() / levels;
            T::convert_from(quantized)
        });

        Ok(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::operations::types::NoiseShape;
    use crate::sample_rate;
    use ndarray::array;

    #[test]
    fn test_dither_flat_changes_samples() {
        // After flat dithering, samples should differ slightly from originals.
        // With a large enough buffer the probability of no change is negligible.
        let data: ndarray::Array1<f32> = ndarray::Array1::from_elem(256, 0.5);
        let audio = AudioSamples::new_mono(data, sample_rate!(44100)).unwrap();
        let dithered = audio.dither(NoiseShape::Flat);
        // At least one sample should have been perturbed; check via indexing
        let arr = dithered.as_mono().unwrap();
        assert!(arr.iter().any(|&s| (s - 0.5f32).abs() > 0.0));
    }

    #[test]
    fn test_dither_fweighted_changes_samples() {
        let data: ndarray::Array1<f32> = ndarray::Array1::from_elem(256, 0.5);
        let audio = AudioSamples::new_mono(data, sample_rate!(44100)).unwrap();
        let dithered = audio.dither(NoiseShape::FWeighted);
        let arr = dithered.as_mono().unwrap();
        assert!(arr.iter().any(|&s| (s - 0.5f32).abs() > 0.0));
    }

    #[test]
    fn test_dither_preserves_approximate_level() {
        // Dithering should add only tiny noise; the mean should remain close.
        let data: ndarray::Array1<f32> = ndarray::Array1::from_elem(1024, 0.5);
        let audio = AudioSamples::new_mono(data, sample_rate!(44100)).unwrap();
        let dithered = audio.dither(NoiseShape::Flat);
        let arr = dithered.as_mono().unwrap();
        let mean: f32 = arr.iter().sum::<f32>() / 1024.0;
        assert!((mean - 0.5).abs() < 0.001, "Mean shifted too much: {mean}");
    }

    #[test]
    fn test_requantize_8bit_exact() {
        // 0.5 * 128 = 64.0 → 64 / 128 = 0.5 (exact grid point)
        let data = array![0.5f32, -0.5, 0.25, -0.25];
        let audio = AudioSamples::new_mono(data, sample_rate!(44100)).unwrap();
        let rq = audio.requantize(8).unwrap();
        // 0.5 is a grid point for 8-bit quantization
        assert!((rq[0] - 0.5f32).abs() < 1e-5);
        // -0.5 is also a grid point
        assert!((rq[1] - (-0.5f32)).abs() < 1e-5);
    }

    #[test]
    fn test_requantize_reduces_precision() {
        // 0.501 should be rounded to the nearest 8-bit grid point ≈ 0.5
        let data = array![0.501f32, -0.001];
        let audio = AudioSamples::new_mono(data, sample_rate!(44100)).unwrap();
        let rq = audio.requantize(8).unwrap();
        // 0.501 * 128 = 64.128 → round to 64 → /128 = 0.5
        assert!((rq[0] - 0.5f32).abs() < 0.01);
    }

    #[test]
    fn test_requantize_error_bits_zero() {
        let data = array![0.5f32, -0.5];
        let audio = AudioSamples::new_mono(data, sample_rate!(44100)).unwrap();
        assert!(audio.requantize(0).is_err());
    }

    #[test]
    fn test_requantize_error_bits_too_large() {
        let data = array![0.5f32, -0.5];
        let audio = AudioSamples::new_mono(data, sample_rate!(44100)).unwrap();
        assert!(audio.requantize(33).is_err());
    }

    #[test]
    fn test_requantize_32bit_is_identity_for_float() {
        // 32-bit quantization of a float value should be essentially identity
        // since float precision exceeds 32-bit integer precision
        let data = array![0.123_456_789_f32, -0.987_654_3];
        let audio = AudioSamples::new_mono(data.clone(), sample_rate!(44100)).unwrap();
        let rq = audio.requantize(32).unwrap();
        // levels = 2^31, so step = 1/2^31 ≈ 4.6e-10 — well below f32 precision
        assert!((rq[0] - data[0]).abs() < 1e-5);
    }

    #[test]
    fn test_dither_multichannel() {
        // Multi-channel audio: each channel should receive independent dithering.
        let data = ndarray::array![[0.5f32, 0.5, 0.5], [0.3f32, 0.3, 0.3]];
        let audio = AudioSamples::new_multi_channel(data.into(), sample_rate!(44100)).unwrap();
        let dithered = audio.dither(NoiseShape::Flat);
        // The audio should still be multi-channel with the same shape
        assert!(dithered.is_multi_channel());
    }
}
