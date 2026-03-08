//! Parametric equalizer for precise frequency shaping.
//!
//! This module implements parametric EQ processing: peak/notch filters, shelving
//! filters, and pass/stop filters organised into bands. A [`ParametricEq`] configuration
//! holds any number of [`EqBand`]s that are applied in sequence.
//!
//! Parametric EQ is the standard tool for correcting frequency imbalances, shaping
//! tonal character, and removing unwanted resonances. Each band gives independent
//! control over centre frequency, gain, and bandwidth (Q), making it far more
//! flexible than fixed-band graphic equalisers.
//!
//! All EQ operations are accessed through the [`AudioParametricEq`] trait. Build a
//! [`ParametricEq`] using its constructors or add bands one at a time, then pass it
//! to [`apply_parametric_eq`][AudioParametricEq::apply_parametric_eq]. Convenience
//! methods such as [`apply_peak_filter`][AudioParametricEq::apply_peak_filter] and
//! [`apply_low_shelf`][AudioParametricEq::apply_low_shelf] handle single-band use
//! cases without constructing a full [`ParametricEq`].
//!
//! # Example
//!
//! ```
//! use audio_samples::{AudioSamples, AudioParametricEq, sample_rate};
//! use audio_samples::operations::types::{ParametricEq, EqBand};
//! use ndarray::Array1;
//!
//! let data = Array1::from_elem(1024, 0.5f32);
//! let mut audio = AudioSamples::new_mono(data, sample_rate!(44100)).unwrap();
//!
//! let mut eq = ParametricEq::new();
//! eq.add_band(EqBand::peak(1000.0, 3.0, 2.0));        // +3 dB at 1 kHz
//! eq.add_band(EqBand::low_shelf(100.0, -2.0, 0.707)); // -2 dB shelf below 100 Hz
//! audio.apply_parametric_eq(&eq).unwrap();
//! ```

use num_traits::FloatConst;

use crate::operations::iir_filtering::IirFilter;
use crate::operations::traits::AudioParametricEq;
use crate::operations::types::{EqBand, EqBandType, ParametricEq};
use crate::traits::StandardSample;
use crate::utils::audio_math::db_to_amplitude as db_to_linear;
use crate::{AudioData, AudioSampleError, LayoutError, ParameterError};
use crate::{AudioSampleResult, AudioSamples, AudioTypeConversion, ConvertTo};

impl<T> AudioParametricEq for AudioSamples<'_, T>
where
    T: StandardSample,
{
    /// Applies a multi-band parametric EQ to the signal.
    ///
    /// Each enabled band in `eq` is applied in sequence using the RBJ biquad filter
    /// coefficients appropriate for its type. An optional output gain is applied after
    /// all bands. If the EQ is bypassed, the signal is returned unchanged.
    ///
    /// # Arguments
    ///
    /// - `eq` – The parametric EQ configuration. All enabled bands are validated before
    ///   processing begins.
    ///
    /// # Returns
    ///
    /// `Ok(())` on success. The audio is modified in place.
    ///
    /// # Errors
    ///
    /// Returns [crate::AudioSampleError::Parameter] if any enabled band fails validation
    /// (e.g. frequency above the Nyquist limit, Q factor ≤ 0, or gain out of range).
    ///
    /// # Example
    ///
    /// ```
    /// use audio_samples::{AudioSamples, AudioParametricEq, sample_rate};
    /// use audio_samples::operations::types::{ParametricEq, EqBand};
    /// use ndarray::Array1;
    ///
    /// let data = Array1::from_elem(512, 0.5f32);
    /// let mut audio = AudioSamples::new_mono(data, sample_rate!(44100)).unwrap();
    ///
    /// let mut eq = ParametricEq::new();
    /// eq.add_band(EqBand::peak(1000.0, 3.0, 2.0));
    /// audio.apply_parametric_eq(&eq).unwrap();
    /// ```
    #[inline]
    fn apply_parametric_eq(&mut self, eq: &ParametricEq) -> AudioSampleResult<()> {
        let sample_rate = self.sample_rate_hz();
        if eq.is_bypassed() {
            return Ok(());
        }

        // Validate the EQ configuration
        let eq = eq.clone().validate(sample_rate)?;

        // Apply each enabled band in sequence
        for band in &eq.bands {
            if band.is_enabled() {
                self.apply_eq_band(band)?;
            }
        }

        // Apply output gain if present
        if eq.output_gain_db != 0.0 {
            let output_gain_linear = db_to_linear(eq.output_gain_db);
            self.apply_linear_gain(output_gain_linear);
        }

        Ok(())
    }

    /// Applies a single EQ band filter to the signal.
    ///
    /// Designs and applies one biquad filter for the given [`EqBand`] using RBJ cookbook
    /// formulas. All seven band types are supported: `Peak`, `LowShelf`, `HighShelf`,
    /// `LowPass`, `HighPass`, `BandPass`, and `BandStop`. Multi-channel audio is
    /// processed per channel with filter state reset between channels.
    ///
    /// If the band is disabled, the signal is returned unchanged.
    ///
    /// # Arguments
    ///
    /// - `band` – The EQ band to apply, including its type, frequency, gain, and Q factor.
    ///
    /// # Returns
    ///
    /// `Ok(())` on success. The audio is modified in place.
    ///
    /// # Errors
    ///
    /// - [crate::AudioSampleError::Parameter] if band validation fails (frequency above Nyquist,
    ///   Q ≤ 0, or gain out of range).
    /// - [crate::AudioSampleError::Layout] if the underlying sample buffer is non-contiguous.
    ///
    /// # Example
    ///
    /// ```
    /// use audio_samples::{AudioSamples, AudioParametricEq, sample_rate};
    /// use audio_samples::operations::types::EqBand;
    /// use ndarray::Array1;
    ///
    /// let data = Array1::from_elem(512, 0.5f32);
    /// let mut audio = AudioSamples::new_mono(data, sample_rate!(44100)).unwrap();
    /// // Boost at 2 kHz by 4 dB with Q of 1.5
    /// audio.apply_eq_band(&EqBand::peak(2000.0, 4.0, 1.5)).unwrap();
    /// ```
    #[inline]
    fn apply_eq_band(&mut self, band: &EqBand) -> AudioSampleResult<()> {
        let sample_rate = self.sample_rate_hz();
        if !band.is_enabled() {
            return Ok(());
        }

        // Validate band parameters
        band.validate(sample_rate)?;

        // Design the filter based on band type
        let (b_coeffs, a_coeffs) = design_eq_band_filter(band, sample_rate);
        let mut filter = IirFilter::new(b_coeffs, a_coeffs);

        // Apply filter to audio data
        match &mut self.data {
            AudioData::Mono(_) => {
                let mut working_samples = self.as_float();
                let Some(mono_self) = self.as_mono_mut() else {
                    return Err(AudioSampleError::Layout(LayoutError::NonContiguous {
                        operation: "parametric EQ".to_string(),
                        layout_type: "non-contiguous mono samples".to_string(),
                    }));
                };

                let working_samples = working_samples.as_mono_mut().ok_or_else(|| {
                    AudioSampleError::Parameter(ParameterError::invalid_value(
                        "audio_format",
                        "Failed to get mono data. Underlying data is not mono.",
                    ))
                })?;

                let working_samples = working_samples.as_slice_mut();
                filter.process_samples_in_place(working_samples);

                for (i, output) in working_samples.iter_mut().enumerate() {
                    mono_self[i] = (*output).convert_to();
                }
            }
            AudioData::Multi(samples) => {
                let num_channels = samples.nrows().get();
                // Process each channel independently
                for channel in 0..num_channels {
                    let mut working_samples = self.as_float();

                    let Some(multi_self) = self.as_multi_channel_mut() else {
                        return Err(AudioSampleError::Layout(LayoutError::NonContiguous {
                            operation: "parametric EQ".to_string(),
                            layout_type: "non-contiguous multi-channel samples".to_string(),
                        }));
                    };
                    let working_samples = working_samples.as_multi_channel_mut() .ok_or_else(|| AudioSampleError::Parameter(ParameterError::invalid_value(
                        "audio_format",
                        "Failed to get multi-channel data. Underlying data is not multi-channel."
                    )))?;
                    let working_samples = working_samples.as_slice_mut().ok_or_else(|| {
                        AudioSampleError::Layout(LayoutError::NonContiguous {
                            operation: "parametric EQ".to_string(),
                            layout_type: "non-contiguous multi-channel samples".to_string(),
                        })
                    })?;

                    filter.process_samples_in_place(working_samples);

                    for (i, output) in working_samples.iter().enumerate() {
                        multi_self[[channel, i]] = (*output).convert_to();
                    }

                    // Reset filter state for next channel
                    filter.reset();
                }
            }
        }

        Ok(())
    }

    /// Applies a peak (or notch) filter at the specified centre frequency.
    ///
    /// A peak filter boosts or cuts a band of frequencies centred around `frequency`.
    /// Positive `gain_db` creates a boost (peak); negative creates a cut (notch). The
    /// Q factor controls bandwidth: higher Q values produce a narrower effect.
    ///
    /// # Arguments
    ///
    /// - `frequency` – Centre frequency in Hz. Must be in `(0, Nyquist)`.
    /// - `gain_db` – Gain in dB. Positive boosts, negative cuts.
    /// - `q_factor` – Quality factor controlling bandwidth. Must be > 0.
    ///
    /// # Returns
    ///
    /// `Ok(())` on success. The audio is modified in place.
    ///
    /// # Errors
    ///
    /// Returns [crate::AudioSampleError::Parameter] if `frequency`, `gain_db`, or `q_factor`
    /// fail band validation.
    ///
    /// # Example
    ///
    /// ```
    /// use audio_samples::{AudioSamples, AudioParametricEq, sample_rate};
    /// use ndarray::Array1;
    ///
    /// let data = Array1::from_elem(512, 0.5f32);
    /// let mut audio = AudioSamples::new_mono(data, sample_rate!(44100)).unwrap();
    /// // Boost at 880 Hz by 6 dB with Q of 2.0
    /// audio.apply_peak_filter(880.0, 6.0, 2.0).unwrap();
    /// ```
    #[inline]
    fn apply_peak_filter(
        &mut self,
        frequency: f64,
        gain_db: f64,
        q_factor: f64,
    ) -> AudioSampleResult<()> {
        let band = EqBand::peak(frequency, gain_db, q_factor);
        self.apply_eq_band(&band)
    }

    /// Applies a low shelf filter that boosts or cuts frequencies below `frequency`.
    ///
    /// All frequencies below the corner frequency receive approximately `gain_db` of
    /// boost or cut, with a transition region controlled by `q_factor`. The shelf levels
    /// off smoothly as frequency approaches zero.
    ///
    /// # Arguments
    ///
    /// - `frequency` – Corner (shelf) frequency in Hz. Must be in `(0, Nyquist)`.
    /// - `gain_db` – Shelf gain in dB. Positive boosts, negative cuts.
    /// - `q_factor` – Shelf slope control. `0.707` gives a maximally flat shelf.
    ///   Must be > 0.
    ///
    /// # Returns
    ///
    /// `Ok(())` on success. The audio is modified in place.
    ///
    /// # Errors
    ///
    /// Returns [crate::AudioSampleError::Parameter] if any parameter fails band validation.
    ///
    /// # Example
    ///
    /// ```
    /// use audio_samples::{AudioSamples, AudioParametricEq, sample_rate};
    /// use ndarray::Array1;
    ///
    /// let data = Array1::from_elem(512, 0.5f32);
    /// let mut audio = AudioSamples::new_mono(data, sample_rate!(44100)).unwrap();
    /// // Cut -3 dB below 200 Hz
    /// audio.apply_low_shelf(200.0, -3.0, 0.707).unwrap();
    /// ```
    #[inline]
    fn apply_low_shelf(
        &mut self,
        frequency: f64,
        gain_db: f64,
        q_factor: f64,
    ) -> AudioSampleResult<()> {
        let band = EqBand::low_shelf(frequency, gain_db, q_factor);
        self.apply_eq_band(&band)
    }

    /// Applies a high shelf filter that boosts or cuts frequencies above `frequency`.
    ///
    /// All frequencies above the corner frequency receive approximately `gain_db` of
    /// boost or cut, with a transition region controlled by `q_factor`. The shelf levels
    /// off smoothly as frequency approaches the Nyquist limit.
    ///
    /// # Arguments
    ///
    /// - `frequency` – Corner (shelf) frequency in Hz. Must be in `(0, Nyquist)`.
    /// - `gain_db` – Shelf gain in dB. Positive boosts, negative cuts.
    /// - `q_factor` – Shelf slope control. `0.707` gives a maximally flat shelf.
    ///   Must be > 0.
    ///
    /// # Returns
    ///
    /// `Ok(())` on success. The audio is modified in place.
    ///
    /// # Errors
    ///
    /// Returns [crate::AudioSampleError::Parameter] if any parameter fails band validation.
    ///
    /// # Example
    ///
    /// ```
    /// use audio_samples::{AudioSamples, AudioParametricEq, sample_rate};
    /// use ndarray::Array1;
    ///
    /// let data = Array1::from_elem(512, 0.5f32);
    /// let mut audio = AudioSamples::new_mono(data, sample_rate!(44100)).unwrap();
    /// // Boost +4 dB above 8 kHz
    /// audio.apply_high_shelf(8000.0, 4.0, 0.707).unwrap();
    /// ```
    #[inline]
    fn apply_high_shelf(
        &mut self,
        frequency: f64,
        gain_db: f64,
        q_factor: f64,
    ) -> AudioSampleResult<()> {
        let band = EqBand::high_shelf(frequency, gain_db, q_factor);
        self.apply_eq_band(&band)
    }

    /// Applies a three-band EQ (low shelf, mid peak, high shelf) in a single call.
    ///
    /// Constructs a [`ParametricEq`] with three bands and applies it:
    /// - A low shelf affecting frequencies below `low_freq`.
    /// - A peak filter centred at `mid_freq`.
    /// - A high shelf affecting frequencies above `high_freq`.
    ///
    /// This mirrors the EQ section found on most mixers and channel strips.
    ///
    /// # Arguments
    ///
    /// - `low_freq` – Low shelf corner frequency in Hz.
    /// - `low_gain` – Low shelf gain in dB.
    /// - `mid_freq` – Mid peak centre frequency in Hz.
    /// - `mid_gain` – Mid peak gain in dB.
    /// - `mid_q` – Mid peak Q factor.
    /// - `high_freq` – High shelf corner frequency in Hz.
    /// - `high_gain` – High shelf gain in dB.
    ///
    /// # Returns
    ///
    /// `Ok(())` on success. The audio is modified in place.
    ///
    /// # Errors
    ///
    /// Returns [crate::AudioSampleError::Parameter] if any band fails validation
    /// (e.g. frequency above Nyquist or Q ≤ 0).
    ///
    /// # Example
    ///
    /// ```
    /// use audio_samples::{AudioSamples, AudioParametricEq, sample_rate};
    /// use ndarray::Array1;
    ///
    /// let data = Array1::from_elem(512, 0.5f32);
    /// let mut audio = AudioSamples::new_mono(data, sample_rate!(44100)).unwrap();
    /// // Low shelf -2 dB at 200 Hz, mid peak +3 dB at 1 kHz (Q=2), high shelf +1 dB at 4 kHz
    /// audio.apply_three_band_eq(200.0, -2.0, 1000.0, 3.0, 2.0, 4000.0, 1.0).unwrap();
    /// ```
    #[inline]
    fn apply_three_band_eq(
        &mut self,
        low_freq: f64,
        low_gain: f64,
        mid_freq: f64,
        mid_gain: f64,
        mid_q: f64,
        high_freq: f64,
        high_gain: f64,
    ) -> AudioSampleResult<()> {
        let eq = ParametricEq::three_band(
            low_freq, low_gain, mid_freq, mid_gain, mid_q, high_freq, high_gain,
        );
        self.apply_parametric_eq(&eq)
    }

    /// Computes the combined magnitude and phase response of a parametric EQ.
    ///
    /// Evaluates each enabled band's biquad filter at every frequency in `frequencies`
    /// and combines the results: magnitudes are multiplied and phases are summed. The
    /// EQ's output gain is applied to the combined magnitude. Disabled bands are skipped.
    ///
    /// This method does not modify the audio signal; it is purely analytical.
    ///
    /// # Arguments
    ///
    /// - `eq` – The parametric EQ whose frequency response to compute.
    /// - `frequencies` – Frequencies in Hz at which to evaluate the response. An empty
    ///   slice returns empty vectors.
    ///
    /// # Returns
    ///
    /// A tuple `(magnitudes, phases)` where:
    /// - `magnitudes` – Linear magnitude response at each frequency (1.0 = unity gain).
    /// - `phases` – Phase response in radians at each frequency.
    ///
    /// Both vectors have the same length as `frequencies`.
    ///
    /// # Errors
    ///
    /// Returns [crate::AudioSampleError::Parameter] if any enabled band fails to design
    /// a filter (e.g. frequency above the Nyquist limit).
    ///
    /// # Example
    ///
    /// ```
    /// use audio_samples::{AudioSamples, AudioParametricEq, sample_rate};
    /// use audio_samples::operations::types::{ParametricEq, EqBand};
    /// use ndarray::Array1;
    ///
    /// let audio = AudioSamples::new_mono(
    ///     Array1::from_elem(512, 0.5f32), sample_rate!(44100),
    /// ).unwrap();
    ///
    /// let mut eq = ParametricEq::new();
    /// eq.add_band(EqBand::peak(1000.0, 6.0, 2.0));
    ///
    /// let freqs = [100.0_f64, 500.0, 1000.0, 5000.0];
    /// let (magnitudes, phases) = audio.eq_frequency_response(&eq, &freqs).unwrap();
    /// assert_eq!(magnitudes.len(), 4);
    /// // At the peak frequency, magnitude should be boosted above unity
    /// assert!(magnitudes[2] > 1.0);
    /// ```
    #[inline]
    fn eq_frequency_response(
        &self,
        eq: &ParametricEq,
        frequencies: &[f64],
    ) -> AudioSampleResult<(Vec<f64>, Vec<f64>)> {
        let mut combined_magnitude = vec![1.0; frequencies.len()];
        let mut combined_phase = vec![0.0; frequencies.len()];
        let sample_rate = self.sample_rate().get();
        for band in &eq.bands {
            if !band.is_enabled() {
                continue;
            }

            let (b_coeffs, a_coeffs) = design_eq_band_filter(band, f64::from(sample_rate));
            let filter = IirFilter::new(b_coeffs, a_coeffs);
            let (magnitude, phase) = filter.frequency_response(frequencies, f64::from(sample_rate));

            // Combine responses (multiply magnitudes, add phases)
            for i in 0..frequencies.len() {
                combined_magnitude[i] *= magnitude[i];
                combined_phase[i] += phase[i];
            }
        }

        // Apply output gain
        if eq.output_gain_db != 0.0 {
            let output_gain_linear = db_to_linear(eq.output_gain_db);
            for magnitude in &mut combined_magnitude {
                *magnitude *= output_gain_linear;
            }
        }

        Ok((combined_magnitude, combined_phase))
    }
}

/// Design a filter for a parametric EQ band.
fn design_eq_band_filter(band: &EqBand, sample_rate: f64) -> (Vec<f64>, Vec<f64>) {
    match band.band_type {
        EqBandType::Peak => {
            design_peak_filter(band.frequency, band.gain_db, band.q_factor, sample_rate)
        }
        EqBandType::LowShelf => {
            design_low_shelf_filter(band.frequency, band.gain_db, band.q_factor, sample_rate)
        }
        EqBandType::HighShelf => {
            design_high_shelf_filter(band.frequency, band.gain_db, band.q_factor, sample_rate)
        }
        EqBandType::LowPass => design_lowpass_filter(band.frequency, band.q_factor, sample_rate),
        EqBandType::HighPass => design_highpass_filter(band.frequency, band.q_factor, sample_rate),
        EqBandType::BandPass => design_bandpass_filter(band.frequency, band.q_factor, sample_rate),
        EqBandType::BandStop => design_bandstop_filter(band.frequency, band.q_factor, sample_rate),
    }
}

/// Design a peak/notch filter using the RBJ (Robert Bristow-Johnson) cookbook formulas.
fn design_peak_filter(
    frequency: f64,
    gain_db: f64,
    q_factor: f64,
    sample_rate: f64,
) -> (Vec<f64>, Vec<f64>) {
    let a = 10.04f64.powf(gain_db / 40.0); // sqrt of linear gain
    let omega = 2.0 * std::f64::consts::PI * frequency / sample_rate;
    let (sin_omega, cos_omega) = omega.sin_cos();
    let alpha = sin_omega / (2.0 * q_factor);

    // RBJ peak filter coefficients
    let b0 = 1.0 + alpha * a;
    let b1 = -2.0 * cos_omega;
    let b2 = 1.0 - alpha * a;
    let a0 = 1.0 + alpha / a;
    let a1 = -2.0 * cos_omega;
    let a2 = 1.0 - alpha / a;

    // Normalize by a0
    let b_coeffs = vec![b0 / a0, b1 / a0, b2 / a0];
    let a_coeffs = vec![1.0, a1 / a0, a2 / a0];

    (b_coeffs, a_coeffs)
}

/// Design a low shelf filter using the RBJ cookbook formulas.
fn design_low_shelf_filter(
    frequency: f64,
    gain_db: f64,
    q_factor: f64,
    sample_rate: f64,
) -> (Vec<f64>, Vec<f64>) {
    let a = 10.0f64.powf(gain_db / 40.0); // sqrt of linear gain
    let omega = 2.0 * std::f64::consts::PI * frequency / sample_rate;
    let (sin_omega, cos_omega) = omega.sin_cos();
    let alpha = sin_omega / (2.0 * q_factor);
    let sqrt_2a = (2.0 * a).sqrt();

    // RBJ low shelf filter coefficients
    let b0 = a * ((a - 1.0).mul_add(-cos_omega, a + 1.0) + sqrt_2a * alpha);
    let b1 = 2.0 * a * (a + 1.0).mul_add(-cos_omega, a - 1.0);
    let b2 = a * ((a - 1.0).mul_add(-cos_omega, a + 1.0) - sqrt_2a * alpha);
    let a0 = (a - 1.0).mul_add(cos_omega, a + 1.0) + sqrt_2a * alpha;
    let a1 = -2.0 * (a + 1.0).mul_add(cos_omega, a - 1.0);
    let a2 = (a - 1.0).mul_add(cos_omega, a + 1.0) - sqrt_2a * alpha;

    // Normalize by a0
    let b_coeffs = vec![b0 / a0, b1 / a0, b2 / a0];
    let a_coeffs = vec![1.0, a1 / a0, a2 / a0];

    (b_coeffs, a_coeffs)
}

/// Design a high shelf filter using the RBJ cookbook formulas.
fn design_high_shelf_filter(
    frequency: f64,
    gain_db: f64,
    q_factor: f64,
    sample_rate: f64,
) -> (Vec<f64>, Vec<f64>) {
    let a = 10.0f64.powf(gain_db / 40.0); // sqrt of linear gain
    let omega = 2.0 * f64::PI() * frequency / sample_rate;
    let (sin_omega, cos_omega) = omega.sin_cos();
    let alpha = sin_omega / (2.0 * q_factor);
    let sqrt_2a = (2.0 * a).sqrt();

    // RBJ high shelf filter coefficients
    let b0 = a * ((a - 1.0).mul_add(cos_omega, a + 1.0) + sqrt_2a * alpha);
    let b1 = -2.0 * a * (a + 1.0).mul_add(cos_omega, a - 1.0);
    let b2 = a * ((a - 1.0).mul_add(cos_omega, a + 1.0) - sqrt_2a * alpha);
    let a0 = (a - 1.0).mul_add(-cos_omega, a + 1.0) + sqrt_2a * alpha;
    let a1 = 2.0 * (a + 1.0).mul_add(-cos_omega, a - 1.0);
    let a2 = (a - 1.0).mul_add(-cos_omega, a + 1.0) - sqrt_2a * alpha;

    // Normalize by a0
    let b_coeffs = vec![b0 / a0, b1 / a0, b2 / a0];
    let a_coeffs = vec![1.0, a1 / a0, a2 / a0];

    (b_coeffs, a_coeffs)
}

/// Design a low-pass filter using the RBJ cookbook formulas.
fn design_lowpass_filter(frequency: f64, q_factor: f64, sample_rate: f64) -> (Vec<f64>, Vec<f64>) {
    let omega = 2.0 * std::f64::consts::PI * frequency / sample_rate;
    let (sin_omega, cos_omega) = omega.sin_cos();
    let alpha = sin_omega / (2.0 * q_factor);

    // RBJ low-pass filter coefficients
    let b0 = (1.0 - cos_omega) / 2.0;
    let b1 = 1.0 - cos_omega;
    let b2 = (1.0 - cos_omega) / 2.0;
    let a0 = 1.0 + alpha;
    let a1 = -2.0 * cos_omega;
    let a2 = 1.0 - alpha;

    // Normalize by a0
    let b_coeffs = vec![b0 / a0, b1 / a0, b2 / a0];
    let a_coeffs = vec![1.0, a1 / a0, a2 / a0];

    (b_coeffs, a_coeffs)
}

/// Design a high-pass filter using the RBJ cookbook formulas.
fn design_highpass_filter(frequency: f64, q_factor: f64, sample_rate: f64) -> (Vec<f64>, Vec<f64>) {
    let omega = 2.0 * std::f64::consts::PI * frequency / sample_rate;
    let (sin_omega, cos_omega) = omega.sin_cos();
    let alpha = sin_omega / (2.0 * q_factor);

    // RBJ high-pass filter coefficients
    let b0 = f64::midpoint(1.0, cos_omega);
    let b1 = -(1.0 + cos_omega);
    let b2 = f64::midpoint(1.0, cos_omega);
    let a0 = 1.0 + alpha;
    let a1 = -2.0 * cos_omega;
    let a2 = 1.0 - alpha;

    // Normalize by a0
    let b_coeffs = vec![b0 / a0, b1 / a0, b2 / a0];
    let a_coeffs = vec![1.0, a1 / a0, a2 / a0];

    (b_coeffs, a_coeffs)
}

/// Design a band-pass filter using the RBJ cookbook formulas.
fn design_bandpass_filter(frequency: f64, q_factor: f64, sample_rate: f64) -> (Vec<f64>, Vec<f64>) {
    let omega = 2.0 * f64::PI() * frequency / sample_rate;
    let (sin_omega, cos_omega) = omega.sin_cos();

    let alpha = sin_omega / (2.0 * q_factor);

    // RBJ band-pass filter coefficients
    let b0 = alpha;
    let b1 = 0.0;
    let b2 = -alpha;
    let a0 = 1.0 + alpha;
    let a1 = -2.0 * cos_omega;
    let a2 = 1.0 - alpha;

    // Normalize by a0
    let b_coeffs = vec![b0 / a0, b1 / a0, b2 / a0];
    let a_coeffs = vec![1.0, a1 / a0, a2 / a0];

    (b_coeffs, a_coeffs)
}

/// Design a band-stop (notch) filter using the RBJ cookbook formulas.
fn design_bandstop_filter(frequency: f64, q_factor: f64, sample_rate: f64) -> (Vec<f64>, Vec<f64>) {
    let omega = 2.0 * std::f64::consts::PI * frequency / sample_rate;
    let (sin_omega, cos_omega) = omega.sin_cos();

    let alpha = sin_omega / (2.0 * q_factor);

    // RBJ band-stop filter coefficients
    let b0 = 1.0;
    let b1 = -2.0 * cos_omega;
    let b2 = 1.0;
    let a0 = 1.0 + alpha;
    let a1 = -2.0 * cos_omega;
    let a2 = 1.0 - alpha;

    // Normalize by a0
    let b_coeffs = vec![b0 / a0, b1 / a0, b2 / a0];
    let a_coeffs = vec![1.0, a1 / a0, a2 / a0];

    (b_coeffs, a_coeffs)
}

impl<T> AudioSamples<'_, T>
where
    T: StandardSample,
{
    /// Apply a linear gain to all samples.
    fn apply_linear_gain(&mut self, gain: f64) {
        self.apply(|x| {
            let x_f: f64 = T::cast_into(x);
            let y_f = x_f * gain;
            T::cast_from(y_f)
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::operations::traits::AudioParametricEq;
    use crate::sample_rate;
    use crate::utils::audio_math::amplitude_to_db as linear_to_db;
    use non_empty_slice::{NonEmptyVec, non_empty_vec};
    use std::f64::consts::PI;

    #[test]
    fn test_peak_filter() {
        // Create a test signal with multiple frequency components
        let sample_rate = 44100.0;
        let duration = 0.1;
        let samples_count = (sample_rate * duration) as usize;

        let mut samples = Vec::new();
        for i in 0..samples_count {
            let t = i as f64 / sample_rate;
            // Mix of frequencies: 440Hz, 880Hz, 1760Hz
            let value = (2.0 * PI * 440.0 * t).sin()
                + (2.0 * PI * 880.0 * t).sin()
                + (2.0 * PI * 1760.0 * t).sin();
            samples.push(value as f32);
        }
        let samples = NonEmptyVec::new(samples).unwrap();
        let mut audio: AudioSamples<'_, f32> =
            AudioSamples::from_mono_vec(samples, sample_rate!(44100));

        // Apply peak filter at 880Hz with +6dB gain
        let result = audio.apply_peak_filter(880.0, 6.0, 2.0);
        assert!(result.is_ok());
    }

    #[test]
    fn test_low_shelf_filter() {
        let sample_rate = 44100.0;
        let duration = 0.1;
        let samples_count = (sample_rate * duration) as usize;

        let mut samples = Vec::new();
        for i in 0..samples_count {
            let t = i as f64 / sample_rate;
            // Mix of low and high frequencies
            let value = (2.0 * PI * 100.0 * t).sin()
                + (2.0 * PI * 1000.0 * t).sin()
                + (2.0 * PI * 5000.0 * t).sin();
            samples.push(value as f32);
        }
        let samples = NonEmptyVec::new(samples).unwrap();
        let mut audio: AudioSamples<'_, f32> =
            AudioSamples::from_mono_vec(samples, sample_rate!(44100));

        // Apply low shelf filter at 500Hz with -3dB gain
        let result = audio.apply_low_shelf(500.0, -3.0, 0.707);
        assert!(result.is_ok());
    }

    #[test]
    fn test_high_shelf_filter() {
        let sample_rate = 44100.0;
        let duration = 0.1;
        let samples_count = (sample_rate * duration) as usize;

        let mut samples = Vec::new();
        for i in 0..samples_count {
            let t = i as f64 / sample_rate;
            // Mix of low and high frequencies
            let value = (2.0 * PI * 100.0 * t).sin()
                + (2.0 * PI * 1000.0 * t).sin()
                + (2.0 * PI * 5000.0 * t).sin();
            samples.push(value as f32);
        }
        let samples = NonEmptyVec::new(samples).unwrap();
        let mut audio: AudioSamples<'_, f32> =
            AudioSamples::from_mono_vec(samples, sample_rate!(44100));

        // Apply high shelf filter at 2000Hz with +4dB gain
        let result = audio.apply_high_shelf(2000.0, 4.0, 0.707);
        assert!(result.is_ok());

        // High frequencies should be boosted
    }

    #[test]
    fn test_three_band_eq() {
        let sample_rate = 44100.0;
        let duration = 0.1;
        let samples_count = (sample_rate * duration) as usize;

        let mut samples = Vec::new();
        for i in 0..samples_count {
            let t = i as f64 / sample_rate;
            // Wide frequency range
            let value = (2.0 * PI * 100.0 * t).sin()
                + (2.0 * PI * 1000.0 * t).sin()
                + (2.0 * PI * 5000.0 * t).sin();
            samples.push(value as f32);
        }
        let samples = NonEmptyVec::new(samples).unwrap();
        let mut audio: AudioSamples<'_, f32> =
            AudioSamples::from_mono_vec(samples, sample_rate!(44100));

        // Apply 3-band EQ: low shelf at 200Hz (-2dB), mid peak at 1kHz (+3dB), high shelf at 4kHz (+1dB)
        let result = audio.apply_three_band_eq(200.0, -2.0, 1000.0, 3.0, 2.0, 4000.0, 1.0);
        assert!(result.is_ok());

        // Should apply all three bands
    }

    #[test]
    fn test_parametric_eq_configuration() {
        let mut audio: AudioSamples<'_, f32> =
            AudioSamples::from_mono_vec(non_empty_vec![1.0f32, 0.0, -1.0], sample_rate!(44100));

        let mut eq = ParametricEq::new();
        eq.add_band(EqBand::peak(1000.0, 3.0, 2.0));
        eq.add_band(EqBand::low_shelf(100.0, -2.0, 0.707));
        eq.set_output_gain(1.0);

        let result = audio.apply_parametric_eq(&eq);
        assert!(result.is_ok());

        // Check EQ configuration
        assert_eq!(eq.band_count(), 2);
        assert_eq!(eq.output_gain_db, 1.0);
        assert!(!eq.is_bypassed());
    }

    #[test]
    fn test_eq_band_validation() {
        let sample_rate = 44100.0;

        // Test valid band
        let valid_band = EqBand::peak(1000.0, 3.0, 2.0);
        assert!(valid_band.validate(sample_rate).is_ok());

        // Test invalid frequency (too high)
        let invalid_band = EqBand::peak(sample_rate, 3.0, 2.0);
        assert!(invalid_band.validate(sample_rate).is_err());

        // Test invalid Q factor
        let invalid_band = EqBand::peak(1000.0, 3.0, 0.0);
        assert!(invalid_band.validate(sample_rate).is_err());

        // Test extreme gain
        let extreme_band = EqBand::peak(1000.0, 50.0, 2.0);
        assert!(extreme_band.validate(sample_rate).is_err());
    }

    #[test]
    fn test_eq_band_enable_disable() {
        let mut band = EqBand::peak(1000.0, 3.0, 2.0);

        assert!(band.is_enabled());

        band.set_enabled(false);
        assert!(!band.is_enabled());

        band.set_enabled(true);
        assert!(band.is_enabled());
    }

    #[test]
    fn test_parametric_eq_bypass() {
        let mut audio: AudioSamples<'_, f32> =
            AudioSamples::from_mono_vec(non_empty_vec![1.0f32, 0.5, -0.5], sample_rate!(44100));
        let original_samples = audio.data.clone();

        let mut eq = ParametricEq::new();
        eq.add_band(EqBand::peak(1000.0, 10.0, 2.0)); // Large gain
        eq.set_bypassed(true);

        let result = audio.apply_parametric_eq(&eq);
        assert!(result.is_ok());

        // Audio should be unchanged when bypassed
        match (&audio.data, &original_samples) {
            (AudioData::Mono(new), AudioData::Mono(orig)) => {
                assert_eq!(new, orig);
            }
            _ => panic!("Expected mono audio"),
        }
    }

    #[test]
    fn test_db_linear_conversion() {
        assert!((db_to_linear(0.0_f64) - 1.0).abs() < 1e-10);
        assert!((db_to_linear(20.0_f64) - 10.0).abs() < 1e-10);
        assert!((db_to_linear(-20.0_f64) - 0.1).abs() < 1e-10);

        assert!((linear_to_db(1.0_f64) - 0.0).abs() < 1e-10);
        assert!((linear_to_db(10.0_f64) - 20.0).abs() < 1e-10);
        assert!((linear_to_db(0.1_f64) - (-20.0)).abs() < 1e-10);
    }

    #[test]
    fn test_five_band_eq() {
        let eq = ParametricEq::five_band();
        assert_eq!(eq.band_count(), 5);

        // Check default frequencies
        assert_eq!(eq.get_band(0).unwrap().frequency, 100.0);
        assert_eq!(eq.get_band(1).unwrap().frequency, 300.0);
        assert_eq!(eq.get_band(2).unwrap().frequency, 1000.0);
        assert_eq!(eq.get_band(3).unwrap().frequency, 3000.0);
        assert_eq!(eq.get_band(4).unwrap().frequency, 8000.0);
    }
}
