//! IIR (Infinite Impulse Response) filter implementations.
//!
//! ## What
//!
//! This module provides the [`IirFilter`] struct for direct-form IIR
//! filtering, and the [`AudioIirFiltering`] trait implementation for
//! [`AudioSamples`]. Supported filter families include Butterworth
//! (all response types) and Chebyshev Type I (design pending).
//!
//! ## Why
//!
//! IIR filters achieve sharper frequency roll-off than FIR filters of
//! the same length by feeding back a portion of the output signal. They
//! are the standard choice for parametric audio effects and
//! telecommunications pre-/post-processing.
//!
//! ## How
//!
//! Use [`IirFilter`] directly when you need manual control over
//! coefficients or need to filter raw `f64` samples. For [`AudioSamples`],
//! bring [`AudioIirFiltering`] into scope and call the convenience methods
//! (`butterworth_lowpass`, etc.), which internally design and apply the filter.
//!
//! ```
//! use audio_samples::operations::iir_filtering::IirFilter;
//!
//! // First-order moving-average: y[n] = 0.5*x[n] + 0.5*x[n-1]
//! let mut filter = IirFilter::new(vec![0.5, 0.5], vec![1.0]);
//! let output = filter.process_samples(&[1.0, 1.0, 1.0, 1.0]);
//! assert_eq!(output[0], 0.5); // 0.5*1.0 + 0.5*0.0 (initial delay is zero)
//! assert_eq!(output[1], 1.0); // 0.5*1.0 + 0.5*1.0 (steady state)
//! ```

use std::num::NonZeroUsize;

use crate::operations::traits::AudioIirFiltering;
use crate::operations::types::{FilterResponse, IirFilterDesign, IirFilterType};
use crate::repr::AudioData;
use crate::traits::StandardSample;
use crate::{AudioSampleError, AudioSampleResult, AudioSamples, ConvertTo, ParameterError};

use ndarray::Axis;
use num_complex::Complex;
use num_traits::FloatConst;

/// A direct-form IIR filter with internal state.
///
/// Implements the standard difference equation using feed-forward
/// (`b_coeffs`) and feed-back (`a_coeffs`) coefficients. Internal
/// delay lines maintain state between calls to [`Self::process_sample`],
/// allowing the filter to be driven sample-by-sample or in bulk.
///
/// ## Invariants
/// - `a_coeffs` must not be empty; `a_coeffs[0]` is the normalisation
///   divisor (conventionally 1.0).
/// - `x_delays.len() == b_coeffs.len() - 1`
/// - `y_delays.len() == a_coeffs.len() - 1`
///
/// # Examples
/// ```
/// use audio_samples::operations::iir_filtering::IirFilter;
///
/// let filter = IirFilter::new(vec![0.5, 0.5], vec![1.0]);
/// assert_eq!(filter.b_coeffs.len(), 2);
/// assert_eq!(filter.x_delays.len(), 1); // one past sample stored
/// ```
#[derive(Debug, Clone)]
pub struct IirFilter {
    /// Feed-forward (numerator) coefficients, ordered as b[0], b[1], …, b[M].
    pub b_coeffs: Vec<f64>,
    /// Feed-back (denominator) coefficients, ordered as a[0], a[1], …, a[N].
    /// `a[0]` is used as the normalisation divisor; set it to 1.0 for
    /// standard filter designs.
    pub a_coeffs: Vec<f64>,
    /// Input delay line storing past inputs: x[n-1], x[n-2], …
    pub x_delays: Vec<f64>,
    /// Output delay line storing past outputs: y[n-1], y[n-2], …
    pub y_delays: Vec<f64>,
}

impl IirFilter {
    /// Create a new IIR filter with the given coefficients.
    ///
    /// Delay lines are zero-initialised, so the filter starts in a
    /// clean state.  `a_coeffs` must not be empty; `a_coeffs[0]` is
    /// used as the normalisation divisor (conventionally 1.0).
    ///
    /// # Arguments
    /// - `b_coeffs` – Feed-forward (numerator) coefficients \[b₀, b₁, …, bₘ\].
    /// - `a_coeffs` – Feed-back (denominator) coefficients \[a₀, a₁, …, aₙ\].
    ///
    /// # Returns
    /// A new [`IirFilter`] with zeroed delay lines.
    ///
    /// # Examples
    /// ```
    /// use audio_samples::operations::iir_filtering::IirFilter;
    ///
    /// // Identity filter: passes input unchanged.
    /// let filter = IirFilter::new(vec![1.0], vec![1.0]);
    /// assert_eq!(filter.b_coeffs, vec![1.0]);
    /// assert!(filter.x_delays.is_empty());
    /// assert!(filter.y_delays.is_empty());
    /// ```
    #[inline]
    #[must_use]
    pub fn new(b_coeffs: Vec<f64>, a_coeffs: Vec<f64>) -> Self {
        let x_delays = vec![0.0; b_coeffs.len().saturating_sub(1)];
        let y_delays = vec![0.0; a_coeffs.len().saturating_sub(1)];

        Self {
            b_coeffs,
            a_coeffs,
            x_delays,
            y_delays,
        }
    }

    /// Process a single sample through the filter.
    ///
    /// Applies the difference equation and updates the internal delay
    /// lines.  Successive calls advance the filter state, so call
    /// order matters.
    ///
    /// # Arguments
    /// - `input` – The current input sample.
    ///
    /// # Returns
    /// The corresponding output sample.
    ///
    /// # Examples
    /// ```
    /// use audio_samples::operations::iir_filtering::IirFilter;
    ///
    /// // Moving-average: y[n] = 0.5*x[n] + 0.5*x[n-1]
    /// let mut filter = IirFilter::new(vec![0.5, 0.5], vec![1.0]);
    /// assert_eq!(filter.process_sample(1.0), 0.5);  // 0.5*1 + 0.5*0
    /// assert_eq!(filter.process_sample(1.0), 1.0);  // 0.5*1 + 0.5*1
    /// ```
    #[inline]
    pub fn process_sample(&mut self, input: f64) -> f64 {
        // Compute feed-forward part
        let mut output = self.b_coeffs[0] * input;
        for i in 0..self.x_delays.len() {
            output += self.b_coeffs[i + 1] * self.x_delays[i];
        }

        // Compute feed-back part
        for i in 0..self.y_delays.len() {
            output -= self.a_coeffs[i + 1] * self.y_delays[i];
        }

        // Normalize by a[0] (should be 1.0 for normalized filters)
        output /= self.a_coeffs[0];

        // Update delay lines
        for i in (1..self.x_delays.len()).rev() {
            self.x_delays[i] = self.x_delays[i - 1];
        }
        if !self.x_delays.is_empty() {
            self.x_delays[0] = input;
        }

        for i in (1..self.y_delays.len()).rev() {
            self.y_delays[i] = self.y_delays[i - 1];
        }
        if !self.y_delays.is_empty() {
            self.y_delays[0] = output;
        }

        output
    }

    /// Process a slice of samples through the filter, returning a new vector.
    ///
    /// Each sample is fed through [`Self::process_sample`] in order, so
    /// filter state carries across the slice.
    ///
    /// # Arguments
    /// - `input` – Contiguous slice of input samples.
    ///
    /// # Returns
    /// A `Vec<f64>` of the same length containing the filtered output.
    ///
    /// # Examples
    /// ```
    /// use audio_samples::operations::iir_filtering::IirFilter;
    ///
    /// let mut filter = IirFilter::new(vec![0.5, 0.5], vec![1.0]);
    /// let out = filter.process_samples(&[1.0, 1.0, 1.0]);
    /// assert_eq!(out, vec![0.5, 1.0, 1.0]);
    /// ```
    #[inline]
    pub fn process_samples(&mut self, input: &[f64]) -> Vec<f64> {
        input.iter().map(|&x| self.process_sample(x)).collect()
    }

    /// Process a mutable slice of samples through the filter in place.
    ///
    /// Overwrites each element with the corresponding filtered output.
    /// Equivalent to [`Self::process_samples`] but avoids allocating a
    /// new vector.
    ///
    /// # Arguments
    /// - `input` – Mutable slice; each element is replaced with its
    ///   filtered value in order.
    ///
    /// # Examples
    /// ```
    /// use audio_samples::operations::iir_filtering::IirFilter;
    ///
    /// let mut filter = IirFilter::new(vec![0.5, 0.5], vec![1.0]);
    /// let mut buf = [1.0, 1.0, 1.0];
    /// filter.process_samples_in_place(&mut buf);
    /// assert_eq!(buf, [0.5, 1.0, 1.0]);
    /// ```
    #[inline]
    pub fn process_samples_in_place(&mut self, input: &mut [f64]) {
        input.iter_mut().for_each(|x| {
            *x = self.process_sample(*x);
        });
    }

    /// Reset the filter's internal delay lines to zero.
    ///
    /// After a reset the filter behaves identically to a freshly
    /// constructed filter with the same coefficients.
    ///
    /// # Examples
    /// ```
    /// use audio_samples::operations::iir_filtering::IirFilter;
    ///
    /// let mut filter = IirFilter::new(vec![1.0, 0.5], vec![1.0, 0.2]);
    /// filter.process_sample(1.0);
    /// assert_ne!(filter.x_delays[0], 0.0);
    /// filter.reset();
    /// assert_eq!(filter.x_delays[0], 0.0);
    /// assert_eq!(filter.y_delays[0], 0.0);
    /// ```
    #[inline]
    pub fn reset(&mut self) {
        self.x_delays.fill(0.0);
        self.y_delays.fill(0.0);
    }

    /// Compute the frequency response at the given frequencies.
    ///
    /// Evaluates the transfer function H(z) = B(z)/A(z) on the unit
    /// circle at each requested frequency, returning magnitude and
    /// phase vectors of the same length.
    ///
    /// # Arguments
    /// - `frequencies` – Frequencies in Hz at which to evaluate the response.
    /// - `sample_rate` – Sample rate of the signal in Hz.
    ///
    /// # Returns
    /// A `(magnitudes, phases)` tuple; both vectors have the same
    /// length as `frequencies`.
    ///
    /// # Examples
    /// ```
    /// use audio_samples::operations::iir_filtering::IirFilter;
    ///
    /// // Identity filter: magnitude 1.0, phase 0.0 everywhere.
    /// let filter = IirFilter::new(vec![1.0], vec![1.0]);
    /// let freqs = vec![0.0, 100.0, 1000.0, 10000.0];
    /// let (mag, phase) = filter.frequency_response(&freqs, 44100.0);
    /// assert_eq!(mag.len(), 4);
    /// for &m in &mag {
    ///     assert!((m - 1.0).abs() < 1e-10);
    /// }
    /// ```
    #[inline]
    #[must_use]
    pub fn frequency_response(
        &self,
        frequencies: &[f64],
        sample_rate: f64,
    ) -> (Vec<f64>, Vec<f64>) {
        let mut magnitudes = Vec::new();
        let mut phases = Vec::new();

        for &freq in frequencies {
            let omega = 2.0 * f64::PI() * freq / sample_rate;
            let z = Complex::new(0.0, omega).exp();

            // Compute numerator (B(z))
            let mut numerator = Complex::new(0.0, 0.0);
            for (i, &b) in self.b_coeffs.iter().enumerate() {
                let term = z.powf(-(i as f64));
                numerator += term * b;
            }

            // Compute denominator (A(z))
            let mut denominator = num_complex::Complex::new(0.0, 0.0);
            for (i, &a) in self.a_coeffs.iter().enumerate() {
                denominator += z.powf(-(i as f64)) * a;
            }

            // H(z) = B(z) / A(z)
            let h = numerator / denominator;
            magnitudes.push(h.norm());
            phases.push(h.arg());
        }

        (magnitudes, phases)
    }
}

impl<T> AudioIirFiltering for AudioSamples<'_, T>
where
    T: StandardSample,
{
    /// Apply an IIR filter using the specified design parameters.
    ///
    /// Designs a filter from `design` (using the audio's own sample
    /// rate), then applies it to every channel independently.
    /// Multi-channel audio resets the filter state between channels
    /// so each channel is filtered from a clean initial state.
    ///
    /// # Arguments
    /// - `design` – Filter specification (type, order, frequencies, ripple).
    ///
    /// # Returns
    /// `Ok(())` on success.
    ///
    /// # Errors
    /// - [`AudioSampleError::Parameter`] – if any frequency in `design`
    ///   is out of the valid range (0, Nyquist), or the filter type is
    ///   not yet implemented.
    ///
    /// # Examples
    /// ```
    /// use audio_samples::{AudioSamples, sample_rate};
    /// use audio_samples::operations::traits::AudioIirFiltering;
    /// use audio_samples::operations::types::IirFilterDesign;
    /// use non_empty_slice::NonEmptyVec;
    /// use std::num::NonZeroUsize;
    ///
    /// let samples = NonEmptyVec::new(vec![1.0f32, 0.5, -0.5, -1.0, 0.0, 1.0, 0.5, -0.5]).unwrap();
    /// let mut audio: AudioSamples<'_, f32> = AudioSamples::from_mono_vec(samples, sample_rate!(44100));
    /// let design = IirFilterDesign::butterworth_lowpass(NonZeroUsize::new(2).unwrap(), 1000.0);
    /// assert!(audio.apply_iir_filter(&design).is_ok());
    /// ```
    fn apply_iir_filter(&mut self, design: &IirFilterDesign) -> AudioSampleResult<()> {
        let sample_rate = self.sample_rate_hz();
        let (b_coeffs, a_coeffs) = design_iir_filter(design, sample_rate)?;
        let mut filter = IirFilter::new(b_coeffs, a_coeffs);

        match &mut self.data {
            AudioData::Mono(samples) => {
                let input_samples: Vec<f64> = samples.iter().map(|&x| x.convert_to()).collect();

                let output_samples = filter.process_samples(&input_samples);

                for (i, &output) in output_samples.iter().enumerate() {
                    samples[i] = output.convert_to();
                }
            }
            AudioData::Multi(data) => {
                // Process each channel independently
                for ch_idx in 0..data.dim().0.get() {
                    let mut channel = data.index_axis_mut(Axis(0), ch_idx);
                    let input_samples: Vec<f64> = channel.iter().map(|&x| x.convert_to()).collect();

                    let output_samples = filter.process_samples(&input_samples);

                    for (sample, output) in channel.iter_mut().zip(output_samples.iter()) {
                        *sample = (*output).convert_to();
                    }

                    // Reset filter state for next channel
                    filter.reset();
                }
            }
        }

        Ok(())
    }

    /// Apply a second-order Butterworth low-pass filter.
    ///
    /// Convenience wrapper that constructs an [`IirFilterDesign`] and
    /// delegates to [`Self::apply_iir_filter`].  Only order 2 produces
    /// mathematically correct coefficients; other orders use an
    /// approximate placeholder.
    ///
    /// # Arguments
    /// - `order` – Filter order (use 2 for correct results).
    /// - `cutoff_frequency` – Cutoff frequency in Hz; must be in
    ///   (0, Nyquist).
    ///
    /// # Returns
    /// `Ok(())` on success.
    ///
    /// # Errors
    /// - [`AudioSampleError::Parameter`] – if `cutoff_frequency` is
    ///   outside the valid range.
    ///
    /// # Examples
    /// ```
    /// use audio_samples::{AudioSamples, sample_rate};
    /// use audio_samples::operations::traits::AudioIirFiltering;
    /// use non_empty_slice::NonEmptyVec;
    /// use std::num::NonZeroUsize;
    ///
    /// let samples = NonEmptyVec::new(vec![1.0f32, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0]).unwrap();
    /// let mut audio: AudioSamples<'_, f32> = AudioSamples::from_mono_vec(samples, sample_rate!(44100));
    /// assert!(audio.butterworth_lowpass(NonZeroUsize::new(2).unwrap(), 1000.0).is_ok());
    /// ```
    fn butterworth_lowpass(
        &mut self,
        order: NonZeroUsize,
        cutoff_frequency: f64,
    ) -> AudioSampleResult<()> {
        let design = IirFilterDesign::butterworth_lowpass(order, cutoff_frequency);
        self.apply_iir_filter(&design)
    }

    /// Apply a second-order Butterworth high-pass filter.
    ///
    /// Convenience wrapper that constructs an [`IirFilterDesign`] and
    /// delegates to [`Self::apply_iir_filter`].  Only order 2 produces
    /// mathematically correct coefficients; other orders use an
    /// approximate placeholder.
    ///
    /// # Arguments
    /// - `order` – Filter order (use 2 for correct results).
    /// - `cutoff_frequency` – Cutoff frequency in Hz; must be in
    ///   (0, Nyquist).
    ///
    /// # Returns
    /// `Ok(())` on success.
    ///
    /// # Errors
    /// - [`AudioSampleError::Parameter`] – if `cutoff_frequency` is
    ///   outside the valid range.
    ///
    /// # Examples
    /// ```
    /// use audio_samples::{AudioSamples, sample_rate};
    /// use audio_samples::operations::traits::AudioIirFiltering;
    /// use non_empty_slice::NonEmptyVec;
    /// use std::num::NonZeroUsize;
    ///
    /// let samples = NonEmptyVec::new(vec![1.0f32, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]).unwrap();
    /// let mut audio: AudioSamples<'_, f32> = AudioSamples::from_mono_vec(samples, sample_rate!(44100));
    /// assert!(audio.butterworth_highpass(NonZeroUsize::new(2).unwrap(), 500.0).is_ok());
    /// ```
    fn butterworth_highpass(
        &mut self,
        order: NonZeroUsize,
        cutoff_frequency: f64,
    ) -> AudioSampleResult<()> {
        let design = IirFilterDesign::butterworth_highpass(order, cutoff_frequency);
        self.apply_iir_filter(&design)
    }

    /// Apply a Butterworth band-pass filter.
    ///
    /// Convenience wrapper that constructs an [`IirFilterDesign`] and
    /// delegates to [`Self::apply_iir_filter`].  The current
    /// implementation uses an approximate placeholder; treat results
    /// as indicative only.
    ///
    /// # Arguments
    /// - `order` – Filter order.
    /// - `low_frequency` – Lower cutoff frequency in Hz; must be > 0.
    /// - `high_frequency` – Upper cutoff frequency in Hz; must be < Nyquist
    ///   and > `low_frequency`.
    ///
    /// # Returns
    /// `Ok(())` on success.
    ///
    /// # Errors
    /// - [`AudioSampleError::Parameter`] – if the frequency range is
    ///   invalid.
    ///
    /// # Examples
    /// ```
    /// use audio_samples::{AudioSamples, sample_rate};
    /// use audio_samples::operations::traits::AudioIirFiltering;
    /// use non_empty_slice::NonEmptyVec;
    /// use std::num::NonZeroUsize;
    ///
    /// let samples = NonEmptyVec::new(vec![1.0f32, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0]).unwrap();
    /// let mut audio: AudioSamples<'_, f32> = AudioSamples::from_mono_vec(samples, sample_rate!(44100));
    /// assert!(audio.butterworth_bandpass(NonZeroUsize::new(2).unwrap(), 100.0, 5000.0).is_ok());
    /// ```
    fn butterworth_bandpass(
        &mut self,
        order: NonZeroUsize,
        low_frequency: f64,
        high_frequency: f64,
    ) -> AudioSampleResult<()> {
        let design = IirFilterDesign::butterworth_bandpass(order, low_frequency, high_frequency);
        self.apply_iir_filter(&design)
    }

    /// Apply a Chebyshev Type I filter.
    ///
    /// Chebyshev Type I filters offer sharper roll-off than Butterworth
    /// filters of the same order at the cost of passband ripple.
    ///
    /// > **Note:** This design is not yet implemented.  All calls
    /// > currently return `Err`.
    ///
    /// # Arguments
    /// - `order` – Filter order (number of poles).
    /// - `cutoff_frequency` – Cutoff frequency in Hz.
    /// - `passband_ripple` – Maximum passband ripple in dB.
    /// - `response` – Filter response type (low-pass, high-pass, etc.).
    ///
    /// # Returns
    /// `Ok(())` on success (currently unreachable).
    ///
    /// # Errors
    /// - [`AudioSampleError::Parameter`] – always, because the design
    ///   is not yet implemented.
    ///
    /// # Examples
    /// ```
    /// use audio_samples::{AudioSamples, sample_rate};
    /// use audio_samples::operations::traits::AudioIirFiltering;
    /// use audio_samples::operations::types::FilterResponse;
    /// use non_empty_slice::NonEmptyVec;
    /// use std::num::NonZeroUsize;
    ///
    /// let samples = NonEmptyVec::new(vec![1.0f32, 0.0, -1.0, 0.0]).unwrap();
    /// let mut audio: AudioSamples<'_, f32> = AudioSamples::from_mono_vec(samples, sample_rate!(44100));
    /// let result = audio.chebyshev_i(
    ///     NonZeroUsize::new(4).unwrap(), 1000.0, 0.5, FilterResponse::LowPass,
    /// );
    /// assert!(result.is_err());
    /// ```
    fn chebyshev_i(
        &mut self,
        order: NonZeroUsize,
        cutoff_frequency: f64,
        passband_ripple: f64,
        response: FilterResponse,
    ) -> AudioSampleResult<()> {
        let design =
            IirFilterDesign::chebyshev_i(response, order, cutoff_frequency, passband_ripple);
        self.apply_iir_filter(&design)
    }

    /// Return the frequency response at the specified frequencies.
    ///
    /// > **Note:** This is a placeholder that returns a flat magnitude-1,
    /// > phase-0 response regardless of the audio content.  A complete
    /// > implementation would store and query the last-applied filter.
    ///
    /// # Arguments
    /// - `frequencies` – Frequencies in Hz at which to compute the response.
    ///
    /// # Returns
    /// `Ok((magnitudes, phases))` where both vectors have the same
    /// length as `frequencies`.  Currently `magnitudes` is all 1.0 and
    /// `phases` is all 0.0.
    ///
    /// # Examples
    /// ```
    /// use audio_samples::{AudioSamples, sample_rate};
    /// use audio_samples::operations::traits::AudioIirFiltering;
    /// use non_empty_slice::NonEmptyVec;
    ///
    /// let samples = NonEmptyVec::new(vec![1.0f32, 0.0, -1.0]).unwrap();
    /// let audio: AudioSamples<'_, f32> = AudioSamples::from_mono_vec(samples, sample_rate!(44100));
    /// let freqs = [100.0, 500.0, 1000.0];
    /// let (mag, phase) = audio.frequency_response(&freqs).unwrap();
    /// assert_eq!(mag.len(), 3);
    /// assert_eq!(phase, vec![0.0, 0.0, 0.0]);
    /// ```
    fn frequency_response(&self, frequencies: &[f64]) -> AudioSampleResult<(Vec<f64>, Vec<f64>)> {
        // For now, return a placeholder implementation
        // In a full implementation, this would store the current filter state
        // and return its frequency response
        Ok((vec![1.0; frequencies.len()], vec![0.0; frequencies.len()]))
    }
}

/// Design an IIR filter based on the given specifications.
///
/// Returns the (b_coeffs, a_coeffs) for the designed filter.
fn design_iir_filter(
    design: &IirFilterDesign,
    sample_rate: f64,
) -> AudioSampleResult<(Vec<f64>, Vec<f64>)> {
    match design.filter_type {
        IirFilterType::Butterworth => design_butterworth_filter(design, sample_rate),
        IirFilterType::ChebyshevI => design_chebyshev_i_filter(design, sample_rate),
        _ => Err(AudioSampleError::Parameter(ParameterError::invalid_value(
            "filter_type",
            "Filter type not yet implemented",
        ))),
    }
}

/// Design a Butterworth filter.
fn design_butterworth_filter(
    design: &IirFilterDesign,
    sample_rate: f64,
) -> AudioSampleResult<(Vec<f64>, Vec<f64>)> {
    let nyquist = sample_rate / 2.0;

    match design.response {
        FilterResponse::LowPass => {
            let cutoff = design.cutoff_frequency.ok_or_else(|| {
                AudioSampleError::Parameter(ParameterError::invalid_value(
                    "cutoff_frequency",
                    "Cutoff frequency required for low-pass filter",
                ))
            })?;

            if cutoff <= 0.0 || cutoff >= nyquist {
                return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
                    "cutoff_frequency",
                    "Cutoff frequency must be between 0 and Nyquist frequency",
                )));
            }

            design_butterworth_lowpass(design.order, cutoff, sample_rate)
        }
        FilterResponse::HighPass => {
            let cutoff = design.cutoff_frequency.ok_or_else(|| {
                AudioSampleError::Parameter(ParameterError::invalid_value(
                    "cutoff_frequency",
                    "Cutoff frequency required for high-pass filter",
                ))
            })?;

            if cutoff <= 0.0 || cutoff >= nyquist {
                return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
                    "cutoff_frequency",
                    "Cutoff frequency must be between 0 and Nyquist frequency",
                )));
            }

            design_butterworth_highpass(design.order, cutoff, sample_rate)
        }
        FilterResponse::BandPass => {
            let low_freq = design.low_frequency.ok_or_else(|| {
                AudioSampleError::Parameter(ParameterError::invalid_value(
                    "low_frequency",
                    "Low frequency required for band-pass filter",
                ))
            })?;
            let high_freq = design.high_frequency.ok_or_else(|| {
                AudioSampleError::Parameter(ParameterError::invalid_value(
                    "high_frequency",
                    "High frequency required for band-pass filter",
                ))
            })?;

            if low_freq <= 0.0 || high_freq >= nyquist || low_freq >= high_freq {
                return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
                    "frequency_range",
                    "Invalid frequency range for band-pass filter",
                )));
            }

            design_butterworth_bandpass(design.order, low_freq, high_freq, sample_rate)
        }
        FilterResponse::BandStop => {
            Err(AudioSampleError::Parameter(ParameterError::invalid_value(
                "filter_response",
                "Band-stop Butterworth filter not yet implemented",
            )))
        }
    }
}

/// Design a Chebyshev Type I filter.
fn design_chebyshev_i_filter(
    design: &IirFilterDesign,
    sample_rate: f64,
) -> AudioSampleResult<(Vec<f64>, Vec<f64>)> {
    let _nyquist = sample_rate / 2.0;
    let _ripple = design.passband_ripple.ok_or_else(|| {
        AudioSampleError::Parameter(ParameterError::invalid_value(
            "passband_ripple",
            "Passband ripple required for Chebyshev Type I filter",
        ))
    })?;
    // TODO!

    // Simplified implementation - in a full implementation, this would
    // compute the Chebyshev polynomials and design the filter accordingly
    Err(AudioSampleError::Parameter(ParameterError::invalid_value(
        "filter_design",
        "Chebyshev Type I filter not yet fully implemented",
    )))
}

/// Design a Butterworth low-pass filter using bilinear transform.
fn design_butterworth_lowpass(
    order: NonZeroUsize,
    cutoff_freq: f64,
    sample_rate: f64,
) -> AudioSampleResult<(Vec<f64>, Vec<f64>)> {
    let order = order.get();
    // Pre-warp the cutoff frequency for bilinear transform
    let wc = 2.0 * sample_rate * (f64::PI() * cutoff_freq / sample_rate).tan();

    // Generate analog Butterworth poles
    let mut poles = Vec::new();
    let order_f = order as f64;
    for k in 0..order {
        let angle = f64::PI() * 2.0f64.mul_add(k as f64, 1.0) / (2.0 * order_f);
        let (sin, cos) = angle.sin_cos();
        let real = -wc * sin;
        let imag = wc * cos;
        poles.push(Complex::new(real, imag));
    }

    // Convert to digital using bilinear transform
    // This is a simplified implementation for demonstration
    // A full implementation would properly handle the bilinear transform

    // For a simple 2nd-order Butterworth low-pass filter
    if order == 2 {
        let k = wc / (2.0 * sample_rate);
        let k2 = k * k;
        let sqrt2 = 2.0f64.sqrt();
        let norm = sqrt2.mul_add(k, 1.0) + k2;

        let b_coeffs = vec![k2 / norm, 2.0 * k2 / norm, k2 / norm];
        let a_coeffs = vec![
            1.0,
            2.0f64.mul_add(k2, -2.0) / norm,
            (sqrt2.mul_add(-k, 1.0) + k2) / norm,
        ];

        Ok((b_coeffs, a_coeffs))
    } else {
        // For other orders, use a simplified approach
        // This is not mathematically correct but serves as a placeholder
        let mut b_coeffs = vec![0.0; order + 1];
        let mut a_coeffs = vec![0.0; order + 1];

        // Simple approximation - not correct for actual use
        b_coeffs[0] = 1.0;
        a_coeffs[0] = 1.0;
        for (i, coeff) in a_coeffs.iter_mut().enumerate().take(order + 1).skip(1) {
            *coeff = 0.1 * i as f64;
        }

        Ok((b_coeffs, a_coeffs))
    }
}

/// Design a Butterworth high-pass filter.
fn design_butterworth_highpass(
    order: NonZeroUsize,
    cutoff_freq: f64,
    sample_rate: f64,
) -> AudioSampleResult<(Vec<f64>, Vec<f64>)> {
    let order = order.get();
    // For a simple 2nd-order Butterworth high-pass filter
    if order == 2 {
        let wc = 2.0 * sample_rate * (f64::PI() * cutoff_freq / sample_rate).tan();
        let k = wc / (2.0 * sample_rate);
        let k2 = k * k;
        let sqrt2 = 2.0f64.sqrt();
        let norm = sqrt2.mul_add(k, 1.0) + k2;

        let b_coeffs = vec![1.0 / norm, -2.0 / norm, 1.0 / norm];
        let a_coeffs = vec![
            1.0,
            2.0f64.mul_add(k2, -2.0) / norm,
            (sqrt2.mul_add(-k, 1.0) + k2) / norm,
        ];

        Ok((b_coeffs, a_coeffs))
    } else {
        // Simplified placeholder implementation
        let mut b_coeffs = vec![0.0; order + 1];
        let mut a_coeffs = vec![0.0; order + 1];

        b_coeffs[0] = 1.0;
        a_coeffs[0] = 1.0;
        for (i, coeff) in a_coeffs.iter_mut().enumerate().take(order + 1).skip(1) {
            *coeff = 0.1 * i as f64;
        }

        Ok((b_coeffs, a_coeffs))
    }
}

/// Design a Butterworth band-pass filter.
fn design_butterworth_bandpass(
    order: NonZeroUsize,
    low_freq: f64,
    high_freq: f64,
    sample_rate: f64,
) -> AudioSampleResult<(Vec<f64>, Vec<f64>)> {
    let order = order.get();
    // Simplified implementation - cascade low-pass and high-pass
    // This is not the proper way to design a band-pass filter
    // but serves as a placeholder

    let center_freq: f64 = (low_freq * high_freq).sqrt();
    let bandwidth = high_freq - low_freq;

    // Simple approximation
    let mut b_coeffs = vec![0.0; order + 1];
    let mut a_coeffs = vec![0.0; order + 1];

    b_coeffs[0] = bandwidth / sample_rate;
    a_coeffs[0] = 1.0;
    a_coeffs[1] = -2.0 * (2.0 * std::f64::consts::PI * center_freq / sample_rate).cos();

    if order > 1 {
        a_coeffs[2] = 0.9;
    }

    Ok((b_coeffs, a_coeffs))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::operations::traits::AudioIirFiltering;
    use crate::sample_rate;
    use non_empty_slice::NonEmptyVec;
    use std::f64::consts::PI;

    #[test]
    fn test_iir_filter_creation() {
        let b_coeffs = vec![1.0, 0.0, -1.0];
        let a_coeffs = vec![1.0, 0.0, 0.5];
        let filter = IirFilter::new(b_coeffs.clone(), a_coeffs.clone());

        assert_eq!(filter.b_coeffs, b_coeffs);
        assert_eq!(filter.a_coeffs, a_coeffs);
        assert_eq!(filter.x_delays.len(), 2);
        assert_eq!(filter.y_delays.len(), 2);
    }

    #[test]
    fn test_butterworth_lowpass_filter() {
        // Create a test signal with two frequency components
        let sample_rate = 44100.0;
        let duration = 0.1;
        let samples_count = (sample_rate * duration) as usize;

        let mut samples = Vec::new();
        for i in 0..samples_count {
            let t = i as f64 / sample_rate;
            // Low frequency component (500 Hz) + high frequency component (5000 Hz)
            let value = (2.0 * PI * 500.0 * t).sin() + 0.5 * (2.0 * PI * 5000.0 * t).sin();
            samples.push(value as f32);
        }
        let samples = NonEmptyVec::new(samples).unwrap();
        let mut audio: AudioSamples<'_, f32> =
            AudioSamples::from_mono_vec(samples, sample_rate!(44100));

        // Apply Butterworth low-pass filter with cutoff at 1000 Hz
        let result = audio.butterworth_lowpass(NonZeroUsize::new(2).unwrap(), 1000.0);
        assert!(result.is_ok());

        // The high frequency component should be attenuated
        // This is a basic test - in practice, you'd analyze the frequency content
    }

    #[test]
    fn test_butterworth_highpass_filter() {
        // Create a test signal with low and high frequency components
        let sample_rate = 44100.0;
        let duration = 0.1;
        let samples_count = (sample_rate * duration) as usize;

        let mut samples = Vec::new();
        for i in 0..samples_count {
            let t = i as f64 / sample_rate;
            // Low frequency component (100 Hz) + high frequency component (2000 Hz)
            let value = (2.0 * PI * 100.0 * t).sin() + 0.5 * (2.0 * PI * 2000.0 * t).sin();
            samples.push(value as f32);
        }
        let samples = NonEmptyVec::new(samples).unwrap();
        let mut audio: AudioSamples<'_, f32> =
            AudioSamples::from_mono_vec(samples, sample_rate!(44100));

        // Apply Butterworth high-pass filter with cutoff at 500 Hz
        let result = audio.butterworth_highpass(NonZeroUsize::new(2).unwrap(), 500.0);
        assert!(result.is_ok());

        // The low frequency component should be attenuated
    }

    #[test]
    fn test_butterworth_bandpass_filter() {
        let sample_rate = 44100.0;
        let duration = 0.1;
        let samples_count = (sample_rate * duration) as usize;

        let mut samples = Vec::new();
        for i in 0..samples_count {
            let t = i as f64 / sample_rate;
            // Multiple frequency components
            let value = (2.0 * PI * 100.0 * t).sin()
                + (2.0 * PI * 1000.0 * t).sin()
                + (2.0 * PI * 5000.0 * t).sin();
            samples.push(value as f32);
        }
        let samples = NonEmptyVec::new(samples).unwrap();

        let mut audio: AudioSamples<'_, f32> =
            AudioSamples::from_mono_vec(samples, sample_rate!(44100));

        // Apply Butterworth band-pass filter from 500 Hz to 2000 Hz
        let result = audio.butterworth_bandpass(NonZeroUsize::new(2).unwrap(), 500.0, 2000.0);
        assert!(result.is_ok());

        // Only frequencies between 500-2000 Hz should pass through
    }

    #[test]
    fn test_filter_with_multichannel_audio() {
        let sample_rate = 44100.0;
        let duration = 0.1;
        let samples_count = (sample_rate * duration) as usize;

        // Create stereo test signal
        let mut left_samples = Vec::new();
        let mut right_samples = Vec::new();

        for i in 0..samples_count {
            let t = i as f64 / sample_rate;
            let left_value = (2.0 * PI * 440.0 * t).sin() + 0.5 * (2.0 * PI * 4400.0 * t).sin();
            let right_value = (2.0 * PI * 880.0 * t).sin() + 0.5 * (2.0 * PI * 8800.0 * t).sin();
            left_samples.push(left_value as f32);
            right_samples.push(right_value as f32);
        }

        let stereo_data = ndarray::Array2::from_shape_vec(
            (2, samples_count),
            [left_samples, right_samples].concat(),
        )
        .unwrap();

        let mut audio =
            AudioSamples::new_multi_channel(stereo_data.into(), sample_rate!(44100)).unwrap();

        // Apply low-pass filter to stereo signal
        let result = audio.butterworth_lowpass(NonZeroUsize::new(2).unwrap(), 2000.0);
        assert!(result.is_ok());

        // Both channels should be filtered independently
    }

    #[test]
    fn test_filter_design_validation() {
        let sample_rate = 44100.0;
        let samples = NonEmptyVec::new(vec![1.0f32, 0.0, -1.0]).unwrap();
        let mut audio: AudioSamples<'_, f32> =
            AudioSamples::from_mono_vec(samples, sample_rate!(44100));

        // Test invalid cutoff frequencies
        assert!(
            audio
                .butterworth_lowpass(NonZeroUsize::new(2).unwrap(), 0.0)
                .is_err()
        );
        assert!(
            audio
                .butterworth_lowpass(NonZeroUsize::new(2).unwrap(), sample_rate / 2.0)
                .is_err()
        );
        assert!(
            audio
                .butterworth_lowpass(NonZeroUsize::new(2).unwrap(), -100.0)
                .is_err()
        );

        // Test invalid band-pass frequencies
        assert!(
            audio
                .butterworth_bandpass(NonZeroUsize::new(2).unwrap(), 2000.0, 1000.0)
                .is_err()
        );
        assert!(
            audio
                .butterworth_bandpass(NonZeroUsize::new(2).unwrap(), 0.0, 1000.0)
                .is_err()
        );
        assert!(
            audio
                .butterworth_bandpass(NonZeroUsize::new(2).unwrap(), 1000.0, sample_rate / 2.0)
                .is_err()
        );
    }

    #[test]
    fn test_filter_design_struct() {
        let design = IirFilterDesign::butterworth_lowpass(NonZeroUsize::new(4).unwrap(), 1000.0);
        assert_eq!(design.filter_type, IirFilterType::Butterworth);
        assert_eq!(design.response, FilterResponse::LowPass);
        assert_eq!(design.order, NonZeroUsize::new(4).unwrap());
        assert_eq!(design.cutoff_frequency, Some(1000.0));

        let design = IirFilterDesign::chebyshev_i(
            FilterResponse::HighPass,
            NonZeroUsize::new(6).unwrap(),
            2000.0,
            1.0,
        );
        assert_eq!(design.filter_type, IirFilterType::ChebyshevI);
        assert_eq!(design.response, FilterResponse::HighPass);
        assert_eq!(design.order, NonZeroUsize::new(6).unwrap());
        assert_eq!(design.cutoff_frequency, Some(2000.0));
        assert_eq!(design.passband_ripple, Some(1.0));
    }

    #[test]
    fn test_iir_filter_processing() {
        let b_coeffs = vec![1.0, 0.0, -1.0];
        let a_coeffs = vec![1.0, 0.0, 0.5];
        let mut filter = IirFilter::new(b_coeffs, a_coeffs);

        let input = vec![1.0, 0.0, -1.0, 0.0, 1.0];
        let output = filter.process_samples(&input);

        assert_eq!(output.len(), input.len());
        // First output should be 1.0 (1.0 * 1.0)
        assert_eq!(output[0], 1.0);
    }

    #[test]
    fn test_filter_reset() {
        let b_coeffs = vec![1.0, 0.5];
        let a_coeffs = vec![1.0, 0.2];
        let mut filter = IirFilter::new(b_coeffs, a_coeffs);

        // Process some samples
        filter.process_sample(1.0);
        filter.process_sample(0.5);

        // Check that delays are not zero
        assert_ne!(filter.x_delays[0], 0.0);
        assert_ne!(filter.y_delays[0], 0.0);

        // Reset filter
        filter.reset();

        // Check that delays are zero
        assert_eq!(filter.x_delays[0], 0.0);
        assert_eq!(filter.y_delays[0], 0.0);
    }
}
