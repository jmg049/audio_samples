//! [`AudioPerceptualAnalysis`] trait definition.

use spectrograms::WindowType;

use crate::{AudioSampleResult, AudioSamples, StandardSample};

use super::{BandLayout, PerceptualAnalysisResult, PsychoacousticConfig};

/// Psychoacoustic analysis operations on [`AudioSamples`].
///
/// This trait exposes the [`analyse_psychoacoustic`](AudioPerceptualAnalysis::analyse_psychoacoustic)
/// method, which drives the full psychoacoustic model pipeline:
///
/// 1. Compute MDCT coefficients of the signal.
/// 2. Derive per-bin energies from the MDCT power spectrum.
/// 3. Map bins to perceptual bands via [`BandLayout`].
/// 4. Compute masking thresholds and signal-to-mask ratios using the
///    psychoacoustic model in [`PsychoacousticConfig`].
///
/// The result contains the raw MDCT coefficients (useful for downstream
/// quantization), the per-bin energies, and the per-band [`BandMetrics`] that
/// summarise audibility, masking, and allowed quantization noise.
///
/// ## Usage
///
/// Bring the trait into scope and call the method on any mono [`AudioSamples`]:
///
/// ```rust
/// # #[cfg(feature = "psychoacoustic")] {
/// use audio_samples::{
///     AudioPerceptualAnalysis, BandLayout, PsychoacousticConfig,
///     sine_wave, sample_rate,
/// };
/// use spectrograms::WindowType;
/// use non_empty_slice::NonEmptySlice;
/// use std::num::NonZeroUsize;
/// use std::time::Duration;
///
/// let signal = sine_wave::<f32>(440.0, Duration::from_millis(100), sample_rate!(44100), 0.5);
///
/// let n_bins = NonZeroUsize::new(1024).unwrap();
/// let layout = BandLayout::bark(NonZeroUsize::new(24).unwrap(), 44100.0, n_bins);
///
/// // Build a minimal config (24 uniform weights, standard masking parameters).
/// let weights: Vec<f32> = vec![1.0; 24];
/// let weights_slice = NonEmptySlice::from_slice(&weights).unwrap();
/// let config = PsychoacousticConfig::new(
///     -60.0,  // noise_floor dB
///     14.5,   // masking_gain dB (tonal)
///     5.5,    // noise_masking_gain dB
///     25.0,   // upward_spread dB/Bark
///     6.0,    // downward_spread dB/Bark
///     weights_slice,
///     1e-10,  // epsilon
/// );
///
/// let result = signal.analyse_psychoacoustic(WindowType::Hanning, &layout, &config).unwrap();
/// assert_eq!(result.band_metrics.len().get(), 24);
/// # }
/// ```
///
/// [`BandMetrics`]: super::BandMetrics
pub trait AudioPerceptualAnalysis {
    /// Runs the psychoacoustic analysis pipeline on this signal.
    ///
    /// Requires **mono** input. Multi-channel signals must be mixed down or a
    /// single channel extracted before calling this method.
    ///
    /// The MDCT window size is chosen automatically as the largest power-of-two
    /// ≤ `min(2048, signal_length)` that is at least 4 samples.
    ///
    /// # Arguments
    /// - `window` – Window function applied before the MDCT.
    ///   [`WindowType::Hanning`] is a reasonable default. Note that only the
    ///   sine window (`MdctParams::sine_window`) gives perfect reconstruction,
    ///   but other windows may better suit the analysis goal.
    /// - `band_layout` – Frequency-band partitioning to analyse.
    ///   Use [`BandLayout::bark`] or [`BandLayout::mel`] for standard presets.
    /// - `config` – Psychoacoustic model parameters. The number of perceptual
    ///   weights in `config` must equal `band_layout.len()`.
    ///
    /// # Returns
    /// A [`PerceptualAnalysisResult`] containing:
    /// - `coefficients` — flattened MDCT coefficients (shape: `n_coefficients × n_frames`).
    /// - `bin_energies` — average power per MDCT bin across frames.
    /// - `band_metrics` — per-band energy, masking threshold, SMR, importance,
    ///   and allowed noise.
    ///
    /// # Errors
    /// - [`AudioSampleError::Parameter`] if the signal is not mono, is too short
    ///   (< 4 samples), or if `config` is incompatible with `band_layout`.
    fn analyse_psychoacoustic(
        &self,
        window: WindowType,
        band_layout: &BandLayout,
        config: &PsychoacousticConfig,
    ) -> AudioSampleResult<PerceptualAnalysisResult>;
}

impl<T> AudioPerceptualAnalysis for AudioSamples<'_, T>
where
    T: StandardSample,
{
    #[inline]
    fn analyse_psychoacoustic(
        &self,
        window: WindowType,
        band_layout: &BandLayout,
        config: &PsychoacousticConfig,
    ) -> AudioSampleResult<PerceptualAnalysisResult> {
        super::analyse_signal(self, window, band_layout, config)
    }
}
