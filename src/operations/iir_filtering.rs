//! IIR (Infinite Impulse Response) filter implementations.
//!
//! This module provides the [`IirFilter`] struct for direct-form IIR
//! filtering, the [`SosFilter`] struct for second-order sections cascade,
//! and the [`AudioIirFiltering`] trait implementation for [`AudioSamples`].
//! Supported filter families include Butterworth, Chebyshev Type I, and
//! Chebyshev Type II (inverse Chebyshev), each across all four response types
//! (low-pass, high-pass, band-pass, band-stop). Band-pass and band-stop use the
//! analog low-pass→band transforms applied to the analog prototype before the
//! bilinear transform, so the band edges and notch are placed correctly.
//!
//! IIR filters achieve sharper frequency roll-off than FIR filters of
//! the same length by feeding back a portion of the output signal. They
//! are the standard choice for parametric audio effects and
//! telecommunications pre-/post-processing.
//!
//! ## Filter Architectures
//!
//! - **Direct Form** ([`IirFilter`]): Used for low-order filters (order ≤ 2).
//!   Numerically stable and efficient for simple biquad sections.
//! - **Second-Order Sections** ([`SosFilter`]): Used automatically for
//!   high-order filters (order > 2). Cascades multiple biquad sections
//!   for numerical stability. This is the industry standard approach
//!   used in SciPy, MATLAB, and professional DSP tools.
//!
//! ## Stability Considerations
//!
//! Filter orders are limited to ≤ 12 for numerical stability. Higher orders
//! may exhibit coefficient quantization errors and unstable poles near the
//! unit circle. For extreme filtering requirements, consider cascading
//! multiple lower-order filters instead.
//!
//! ## Usage
//!
//! Use [`IirFilter`] or [`SosFilter`] directly when you need manual control
//! over coefficients or need to filter raw `f64` samples. For [`AudioSamples`],
//! bring [`AudioIirFiltering`] into scope and call the convenience methods
//! (`butterworth_lowpass`, `chebyshev_i`, etc.), which internally design
//! and apply the appropriate filter representation.
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

// `IirFilterDesign` is used by the design dispatcher; the streaming
// `to_sos` / `from_design` API is defined further down in this module.
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
#[non_exhaustive]
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
        for x in input.iter_mut() {
            *x = self.process_sample(*x);
        }
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
            // z^-1 = reciprocal of z on the unit circle. Successive
            // multiplication by `z_inv` replaces the per-coefficient
            // `z.powf(-i)` call, giving the same z^-i terms iteratively.
            let z_inv = z.inv();

            // Compute numerator (B(z))
            let mut numerator = Complex::new(0.0, 0.0);
            let mut z_pow = Complex::new(1.0, 0.0); // z^0
            for &b in &self.b_coeffs {
                numerator += z_pow * b;
                z_pow *= z_inv;
            }

            // Compute denominator (A(z))
            let mut denominator = num_complex::Complex::new(0.0, 0.0);
            let mut z_pow = Complex::new(1.0, 0.0); // z^0
            for &a in &self.a_coeffs {
                denominator += z_pow * a;
                z_pow *= z_inv;
            }

            // H(z) = B(z) / A(z)
            let h = numerator / denominator;
            magnitudes.push(h.norm());
            phases.push(h.arg());
        }

        (magnitudes, phases)
    }
}

// ============================================================================
// Private Biquad — fast inner type used by apply_iir_filter
// ============================================================================

/// Fixed-size second-order (biquad) section using Direct Form II Transposed.
///
/// All coefficients are pre-normalised by a0 at construction so the hot loop
/// contains **no division and no heap allocation**.  State fits in two f64
/// slots on the stack.
///
/// Direct Form II Transposed recurrence:
/// ```text
///   y    = b0·x + s0
///   s0'  = b1·x − a1·y + s1
///   s1'  = b2·x − a2·y
/// ```
#[derive(Debug, Clone, Copy)]
struct Biquad {
    b0: f64,
    b1: f64,
    b2: f64,
    a1: f64,
    a2: f64,
    s0: f64,
    s1: f64,
}

impl Biquad {
    fn from_coeffs(b_coeffs: &[f64], a_coeffs: &[f64]) -> Self {
        let a0 = a_coeffs.first().copied().unwrap_or(1.0);
        let b0 = b_coeffs.first().copied().unwrap_or(0.0) / a0;
        let b1 = b_coeffs.get(1).copied().unwrap_or(0.0) / a0;
        let b2 = b_coeffs.get(2).copied().unwrap_or(0.0) / a0;
        let a1 = a_coeffs.get(1).copied().unwrap_or(0.0) / a0;
        let a2 = a_coeffs.get(2).copied().unwrap_or(0.0) / a0;
        Self {
            b0,
            b1,
            b2,
            a1,
            a2,
            s0: 0.0,
            s1: 0.0,
        }
    }

    #[inline(always)]
    fn process(&mut self, x: f64) -> f64 {
        let y = self.b0 * x + self.s0;
        self.s0 = self.b1 * x - self.a1 * y + self.s1;
        self.s1 = self.b2 * x - self.a2 * y;
        y
    }

    #[inline(always)]
    fn reset(&mut self) {
        self.s0 = 0.0;
        self.s1 = 0.0;
    }

    /// Recover normalised `(b, a)` coefficient vectors (a0 = 1) for this biquad.
    ///
    /// Used to rebuild a stateful [`IirFilter`] section from a designed
    /// (already-normalised) [`Biquad`] when assembling a streaming
    /// [`SosFilter`]. The coefficients are identical up to the a0 = 1
    /// normalisation the biquad already applied at construction.
    fn to_coeffs(self) -> (Vec<f64>, Vec<f64>) {
        (
            vec![self.b0, self.b1, self.b2],
            vec![1.0, self.a1, self.a2],
        )
    }
}

/// Build a stateful [`SosFilter`] (cascade of [`IirFilter`] sections) from an
/// internal [`FilterRepresentation`].
///
/// The biquads carried by the representation are already normalised by a0, so
/// each section is reconstructed as an order-2 [`IirFilter`] with a0 = 1. The
/// resulting cascade is mathematically identical to the fast biquad path but
/// exposes a public, resettable streaming API.
fn sos_from_representation(repr: &FilterRepresentation) -> SosFilter {
    let sections = match repr {
        FilterRepresentation::Single(bq) => {
            let (b, a) = bq.to_coeffs();
            vec![IirFilter::new(b, a)]
        }
        FilterRepresentation::Cascade(bqs) => bqs
            .iter()
            .map(|bq| {
                let (b, a) = bq.to_coeffs();
                IirFilter::new(b, a)
            })
            .collect(),
    };
    SosFilter::new(sections)
}

/// Zero-phase (forward-backward) filtering of one channel of `f64` samples,
/// in place, using `sos`.
///
/// Algorithm (SciPy `filtfilt` style):
/// 1. Extend the signal at both ends with odd (point-)reflection padding of
///    length `padlen ≈ 3·(2·n_sections)` so the edges of interest are not
///    contaminated by the filter's start-up transient.
/// 2. Filter the padded signal forward, reset the cascade state, reverse,
///    filter forward again (= backward over the original orientation), reverse
///    back. The two passes cancel the phase and square the magnitude.
/// 3. Strip the padding, leaving an output the same length as the input.
///
/// Odd reflection about the first sample maps `x[k]` to `2·x[0] − x[k]`
/// (and symmetrically at the tail about the last sample), which continues the
/// signal smoothly without introducing a discontinuity or a DC step.
///
/// For very short signals (`len ≤ padlen`) the padding is clamped to
/// `len − 1`; a single-sample signal is filtered without padding.
fn filtfilt_slice(sos: &mut SosFilter, signal: &mut [f64]) {
    let len = signal.len();
    if len == 0 {
        return;
    }

    // Padding length: 3 * (number of biquad coefficients along one side). Each
    // second-order section contributes 2 to the effective order, so 2*n_sections
    // is the total filter order; tripling it gives ample transient room. This
    // matches scipy.signal.sosfiltfilt's default `padlen = 3 * (2 * n_sections)`.
    let n_sections = sos.sections.len().max(1);
    let mut padlen = 3 * (2 * n_sections);
    if padlen >= len {
        // Not enough samples for the default; clamp so we never reflect past
        // the available data.
        padlen = len.saturating_sub(1);
    }

    // Build the odd-reflected, padded buffer:
    //   left  pad: 2*x[0]   - x[padlen..0]      (reversed)
    //   body:      x[0..len]
    //   right pad: 2*x[len-1] - x[len-2 .. len-2-padlen] (reversed)
    let mut padded = Vec::with_capacity(len + 2 * padlen);
    let x0 = signal[0];
    for k in (1..=padlen).rev() {
        padded.push(2.0 * x0 - signal[k]);
    }
    padded.extend_from_slice(signal);
    let xn = signal[len - 1];
    for k in 1..=padlen {
        padded.push(2.0 * xn - signal[len - 1 - k]);
    }

    // Forward pass.
    sos.reset();
    sos.process_block(&mut padded);

    // Backward pass: reverse, filter forward (reset state first), reverse back.
    padded.reverse();
    sos.reset();
    sos.process_block(&mut padded);
    padded.reverse();

    // Strip the padding back to the original length.
    signal.copy_from_slice(&padded[padlen..padlen + len]);
}

/// A second-order sections (SOS) cascade filter.
///
/// Represents an IIR filter as a cascade of second-order (biquad) sections.
/// This architecture is numerically stable for high-order filters (order > 4),
/// where direct-form implementations suffer from coefficient quantization
/// and floating-point errors.
///
/// Each section is a standard biquad filter that processes the output of
/// the previous section. The cascade approach is the industry standard for
/// high-order IIR filters (used in SciPy, MATLAB, etc.).
///
/// # Invariants
/// - `sections` must not be empty.
/// - Each section is a valid second-order [`IirFilter`] (order ≤ 2).
///
/// # Examples
/// ```
/// use audio_samples::operations::iir_filtering::{IirFilter, SosFilter};
///
/// // Two cascaded first-order sections
/// let section1 = IirFilter::new(vec![0.5, 0.5], vec![1.0]);
/// let section2 = IirFilter::new(vec![0.3, 0.7], vec![1.0]);
/// let mut sos = SosFilter::new(vec![section1, section2]);
///
/// let output = sos.process_sample(1.0);
/// assert!(output > 0.0);
/// ```
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct SosFilter {
    /// The cascade of biquad sections, processed in order.
    pub sections: Vec<IirFilter>,
}

impl SosFilter {
    /// Create a new SOS filter from a vector of biquad sections.
    ///
    /// The sections are applied in order: input → section[0] → section[1] → ... → output.
    ///
    /// # Arguments
    /// - `sections` – Vector of second-order [`IirFilter`] sections.
    ///
    /// # Returns
    /// A new [`SosFilter`] with the given sections.
    ///
    /// # Examples
    /// ```
    /// use audio_samples::operations::iir_filtering::{IirFilter, SosFilter};
    ///
    /// let section1 = IirFilter::new(vec![1.0, 0.0, 0.0], vec![1.0, 0.0, 0.0]);
    /// let section2 = IirFilter::new(vec![1.0, 0.0, 0.0], vec![1.0, 0.0, 0.0]);
    /// let sos = SosFilter::new(vec![section1, section2]);
    /// assert_eq!(sos.sections.len(), 2);
    /// ```
    #[inline]
    #[must_use]
    pub const fn new(sections: Vec<IirFilter>) -> Self {
        Self { sections }
    }

    /// Process a single sample through the cascade.
    ///
    /// Feeds the input through each section in order, passing the
    /// output of section[i] as the input to section[i+1].
    ///
    /// # Arguments
    /// - `input` – The current input sample.
    ///
    /// # Returns
    /// The final output sample after cascading through all sections.
    ///
    /// # Examples
    /// ```
    /// use audio_samples::operations::iir_filtering::{IirFilter, SosFilter};
    ///
    /// let section = IirFilter::new(vec![0.5, 0.5], vec![1.0]);
    /// let mut sos = SosFilter::new(vec![section]);
    /// assert_eq!(sos.process_sample(1.0), 0.5);
    /// assert_eq!(sos.process_sample(1.0), 1.0);
    /// ```
    #[inline]
    pub fn process_sample(&mut self, input: f64) -> f64 {
        let mut signal = input;
        for section in &mut self.sections {
            signal = section.process_sample(signal);
        }
        signal
    }

    /// Process a slice of samples through the cascade, returning a new vector.
    ///
    /// Equivalent to calling [`Self::process_sample`] for each input sample
    /// in order. Filter state carries across the slice and all sections.
    ///
    /// # Arguments
    /// - `input` – Contiguous slice of input samples.
    ///
    /// # Returns
    /// A `Vec<f64>` of the same length containing the filtered output.
    ///
    /// # Examples
    /// ```
    /// use audio_samples::operations::iir_filtering::{IirFilter, SosFilter};
    ///
    /// let section = IirFilter::new(vec![0.5, 0.5], vec![1.0]);
    /// let mut sos = SosFilter::new(vec![section]);
    /// let out = sos.process_samples(&[1.0, 1.0, 1.0]);
    /// assert_eq!(out, vec![0.5, 1.0, 1.0]);
    /// ```
    #[inline]
    pub fn process_samples(&mut self, input: &[f64]) -> Vec<f64> {
        input.iter().map(|&x| self.process_sample(x)).collect()
    }

    /// Process a mutable slice of samples through the cascade in place.
    ///
    /// Overwrites each element with the corresponding filtered output.
    /// Equivalent to [`Self::process_samples`] but avoids allocating.
    ///
    /// # Arguments
    /// - `input` – Mutable slice; each element is replaced with its
    ///   filtered value in order.
    ///
    /// # Examples
    /// ```
    /// use audio_samples::operations::iir_filtering::{IirFilter, SosFilter};
    ///
    /// let section = IirFilter::new(vec![0.5, 0.5], vec![1.0]);
    /// let mut sos = SosFilter::new(vec![section]);
    /// let mut buf = [1.0, 1.0, 1.0];
    /// sos.process_samples_in_place(&mut buf);
    /// assert_eq!(buf, [0.5, 1.0, 1.0]);
    /// ```
    #[inline]
    pub fn process_samples_in_place(&mut self, input: &mut [f64]) {
        for x in input.iter_mut() {
            *x = self.process_sample(*x);
        }
    }

    /// Reset all sections' delay lines to zero.
    ///
    /// After a reset the cascade behaves identically to a freshly
    /// constructed filter with the same section coefficients.
    ///
    /// # Examples
    /// ```
    /// use audio_samples::operations::iir_filtering::{IirFilter, SosFilter};
    ///
    /// let section = IirFilter::new(vec![1.0, 0.5], vec![1.0, 0.2]);
    /// let mut sos = SosFilter::new(vec![section]);
    /// sos.process_sample(1.0);
    /// assert_ne!(sos.sections[0].x_delays[0], 0.0);
    /// sos.reset();
    /// assert_eq!(sos.sections[0].x_delays[0], 0.0);
    /// ```
    #[inline]
    pub fn reset(&mut self) {
        for section in &mut self.sections {
            section.reset();
        }
    }

    /// Build an [`SosFilter`] from a filter design, for design-once streaming.
    ///
    /// Designs the filter once (via the same dispatcher used by
    /// [`AudioIirFiltering::apply_iir_filter_in_place`]) and returns a stateful
    /// cascade whose [`Self::process_sample`] / [`Self::process_block`] retain
    /// state across calls. This lets real-time / block processing avoid
    /// redesigning the filter for every block.
    ///
    /// # Arguments
    /// - `design` – Filter specification (type, order, frequencies, ripple).
    /// - `sample_rate` – Sample rate of the signal in Hz.
    ///
    /// # Returns
    /// A freshly-reset [`SosFilter`] implementing the requested design.
    ///
    /// # Errors
    /// - [crate::AudioSampleError::Parameter] – if the design is invalid
    ///   (out-of-range frequency, unsupported response, order > 12, …).
    ///
    /// # Examples
    /// ```
    /// use audio_samples::operations::iir_filtering::SosFilter;
    /// use audio_samples::operations::types::IirFilterDesign;
    /// use std::num::NonZeroUsize;
    ///
    /// let design = IirFilterDesign::butterworth_lowpass(NonZeroUsize::new(4).unwrap(), 1000.0);
    /// let mut sos = SosFilter::from_design(&design, 44100.0).unwrap();
    /// let y = sos.process_sample(1.0);
    /// assert!(y.is_finite());
    /// ```
    #[inline]
    pub fn from_design(
        design: &IirFilterDesign,
        sample_rate: f64,
    ) -> AudioSampleResult<SosFilter> {
        let repr = design_iir_filter(design, sample_rate)?;
        Ok(sos_from_representation(&repr))
    }

    /// Process a block of samples in place, retaining filter state across calls.
    ///
    /// Each element of `block` is replaced with its filtered value, in order,
    /// flowing through every section. Because the internal delay lines persist,
    /// calling `process_block` on consecutive halves of a signal yields exactly
    /// the same result as calling it once on the whole signal — the key
    /// property for streaming / real-time block processing.
    ///
    /// This is an alias of [`Self::process_samples_in_place`] with a name that
    /// signals the design-once / stream-many usage.
    ///
    /// # Arguments
    /// - `block` – Mutable slice; each element is replaced with its filtered value.
    ///
    /// # Examples
    /// ```
    /// use audio_samples::operations::iir_filtering::{IirFilter, SosFilter};
    ///
    /// let section = IirFilter::new(vec![0.5, 0.5], vec![1.0]);
    /// let mut sos = SosFilter::new(vec![section]);
    /// let mut buf = [1.0, 1.0, 1.0];
    /// sos.process_block(&mut buf);
    /// assert_eq!(buf, [0.5, 1.0, 1.0]);
    /// ```
    #[inline]
    pub fn process_block(&mut self, block: &mut [f64]) {
        self.process_samples_in_place(block);
    }

    /// Compute the frequency response at the given frequencies.
    ///
    /// Evaluates the combined transfer function of all cascaded sections.
    /// The magnitude response is the product of all section magnitudes;
    /// the phase response is the sum of all section phases.
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
    /// use audio_samples::operations::iir_filtering::{IirFilter, SosFilter};
    ///
    /// // Identity sections: total magnitude 1.0, phase 0.0
    /// let section1 = IirFilter::new(vec![1.0], vec![1.0]);
    /// let section2 = IirFilter::new(vec![1.0], vec![1.0]);
    /// let sos = SosFilter::new(vec![section1, section2]);
    /// let (mag, phase) = sos.frequency_response(&[100.0, 1000.0], 44100.0);
    /// assert_eq!(mag.len(), 2);
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
        let mut combined_magnitudes = vec![1.0; frequencies.len()];
        let mut combined_phases = vec![0.0; frequencies.len()];

        for section in &self.sections {
            let (mag, phase) = section.frequency_response(frequencies, sample_rate);
            for i in 0..frequencies.len() {
                combined_magnitudes[i] *= mag[i];
                combined_phases[i] += phase[i];
            }
        }

        (combined_magnitudes, combined_phases)
    }
}

// ============================================================================
// Helper functions for SOS filter design
// ============================================================================

/// Apply the bilinear transform to convert an analog pole to a digital pole.
///
/// Uses the transformation: z = (1 + s/(2fs)) / (1 - s/(2fs))
/// where s is the analog pole and fs is the sample rate.
///
/// # Arguments
/// - `s_pole` – Complex analog pole in the s-plane.
/// - `sample_rate` – Sample rate in Hz.
///
/// # Returns
/// The corresponding digital pole in the z-plane.
fn bilinear_transform_pole(s_pole: Complex<f64>, sample_rate: f64) -> Complex<f64> {
    let two_fs = 2.0 * sample_rate;
    let numerator = Complex::new(1.0, 0.0) + s_pole / two_fs;
    let denominator = Complex::new(1.0, 0.0) - s_pole / two_fs;
    numerator / denominator
}

/// Pair complex conjugate poles for biquad construction.
///
/// Groups poles into conjugate pairs: (p, p*). For odd-order filters,
/// the real pole is paired with itself (producing a first-order section
/// represented as a biquad with a duplicate pole).
///
/// # Arguments
/// - `poles` – Vector of complex poles (may include conjugate pairs and real poles).
///
/// # Returns
/// Vector of pole pairs: each element is `(pole1, pole2)` where `pole2 = pole1.conj()`
/// for conjugate pairs, or `pole2 = pole1` for real poles.
fn pair_poles(poles: Vec<Complex<f64>>) -> Vec<(Complex<f64>, Complex<f64>)> {
    let mut pairs = Vec::new();
    let mut used = vec![false; poles.len()];

    for i in 0..poles.len() {
        if used[i] {
            continue;
        }

        // Check if this is a real pole (imaginary part ≈ 0)
        if poles[i].im.abs() < 1e-10 {
            // Real pole: pair with itself
            pairs.push((poles[i], poles[i]));
            used[i] = true;
        } else {
            // Complex pole: find its conjugate
            let mut found_conjugate = false;
            for j in (i + 1)..poles.len() {
                if used[j] {
                    continue;
                }
                // Check if poles[j] is the conjugate of poles[i]
                let diff = (poles[i].conj() - poles[j]).norm();
                if diff < 1e-10 {
                    pairs.push((poles[i], poles[j]));
                    used[i] = true;
                    used[j] = true;
                    found_conjugate = true;
                    break;
                }
            }
            if !found_conjugate {
                // No conjugate found, pair with itself (shouldn't happen for properly designed filters)
                pairs.push((poles[i], poles[i]));
                used[i] = true;
            }
        }
    }

    pairs
}

/// Construct biquad coefficients from a pair of poles and zeros.
///
/// Creates a second-order filter section from two poles and two zeros.
/// For real poles/zeros, pass the same value twice. The DC gain is
/// normalized to the specified value.
///
/// # Arguments
/// - `p1` – First pole.
/// - `p2` – Second pole (or same as `p1` for first-order section).
/// - `z1` – First zero.
/// - `z2` – Second zero (or same as `z1` for first-order section).
/// - `gain` – DC gain for normalization.
///
/// # Returns
/// `(b_coeffs, a_coeffs)` vectors of length 3 representing the biquad.
fn biquad_from_poles_zeros(
    p1: Complex<f64>,
    p2: Complex<f64>,
    z1: Complex<f64>,
    z2: Complex<f64>,
    gain: f64,
) -> (Vec<f64>, Vec<f64>) {
    // Denominator: (z - p1)(z - p2) = z^2 - (p1 + p2)z + p1*p2
    // Expanded: a0 = 1, a1 = -(p1 + p2), a2 = p1*p2
    let a = vec![1.0, -(p1 + p2).re, (p1 * p2).re];

    // Numerator: (z - z1)(z - z2) = z^2 - (z1 + z2)z + z1*z2
    // Expanded: b0 = 1, b1 = -(z1 + z2), b2 = z1*z2
    let b = vec![gain, gain * (-(z1 + z2)).re, gain * (z1 * z2).re];

    (b, a)
}

// ============================================================================
// Analog zero-pole-gain (ZPK) prototype infrastructure
// ============================================================================
//
// To produce correct band-pass / band-stop responses and Chebyshev Type II
// (which has finite transmission zeros), the simple "pole-only, zeros pinned at
// ±1" path is not sufficient. Instead we follow the standard SciPy/MATLAB
// pipeline:
//
//   1. Build a *normalised* analog low-pass prototype (cutoff ω = 1 rad/s) as a
//      set of zeros, poles and a real gain in the s-plane.
//   2. Apply an analog frequency transform (LP→LP, LP→HP, LP→BP, LP→BS) that
//      maps the prototype onto the desired band, in the s-plane.
//   3. Bilinear-transform every zero and pole into the z-plane and fold the
//      analog gain through the bilinear Jacobian.
//   4. Pair conjugate roots into biquad second-order sections.
//
// This keeps zeros first-class throughout, which is exactly what BP/BS and
// Chebyshev II require.

/// Analog zero-pole-gain description in the s-plane.
#[derive(Debug, Clone)]
struct AnalogZpk {
    zeros: Vec<Complex<f64>>,
    poles: Vec<Complex<f64>>,
    gain: f64,
}

/// Normalised (ω_c = 1) analog Butterworth low-pass prototype.
///
/// No finite zeros; `order` poles equally spaced on the unit circle in the left
/// half-plane; unit gain.
fn butterworth_prototype(order: usize) -> AnalogZpk {
    let mut poles = Vec::with_capacity(order);
    let n = order as f64;
    for k in 0..order {
        // Angles π/2 · (2k+1)/N offset into the left half-plane.
        let theta = f64::PI() * (2.0f64.mul_add(k as f64, 1.0)) / (2.0 * n) + f64::PI() / 2.0;
        poles.push(Complex::new(theta.cos(), theta.sin()));
    }
    AnalogZpk {
        zeros: Vec::new(),
        poles,
        gain: 1.0,
    }
}

/// Normalised analog Chebyshev Type I low-pass prototype (ω_c = 1).
///
/// `ripple_db` is the passband ripple. No finite zeros; poles on an ellipse.
/// The gain is set so the prototype matches the standard Cheby-I DC behaviour
/// (0 dB at DC for even order, peak-ripple reference for odd order).
fn chebyshev1_prototype(order: usize, ripple_db: f64) -> AnalogZpk {
    let epsilon = (10.0_f64.powf(ripple_db / 10.0) - 1.0).sqrt();
    let n = order as f64;
    let mu = (1.0 / epsilon).asinh() / n;

    let mut poles = Vec::with_capacity(order);
    for k in 0..order {
        let theta = f64::PI() * (2.0f64.mul_add(k as f64, 1.0)) / (2.0 * n);
        // Left-half-plane Chebyshev poles.
        let re = -mu.sinh() * theta.sin();
        let im = mu.cosh() * theta.cos();
        poles.push(Complex::new(re, im));
    }

    // Gain so that the prototype transfer function has the canonical Cheby-I
    // normalisation: product(-poles) gives the leading coefficient; for even
    // order divide by sqrt(1+ε²) to land the DC value on the ripple bound.
    let mut gain = poles.iter().fold(Complex::new(1.0, 0.0), |acc, &p| acc * (-p));
    let mut gain_re = gain.re;
    if order.is_multiple_of(2) {
        gain_re /= (1.0 + epsilon * epsilon).sqrt();
    }
    gain = Complex::new(gain_re, 0.0);

    AnalogZpk {
        zeros: Vec::new(),
        poles,
        gain: gain.re,
    }
}

/// Normalised analog Chebyshev Type II (inverse Chebyshev) low-pass prototype.
///
/// Here `ω = 1` is the *stopband* edge: the response first reaches the
/// requested `stopband_db` attenuation at ω = 1 and is equiripple beyond it,
/// while the passband is maximally flat. The prototype has `order` poles and,
/// for even order, `order` finite imaginary-axis zeros (odd order drops one
/// zero to ∞, leaving `order−1`). Built by reciprocal-mapping the Chebyshev I
/// poles and placing zeros at `j / cos((2k−1)π/2N)`.
fn chebyshev2_prototype(order: usize, stopband_db: f64) -> AnalogZpk {
    let n = order as f64;
    // ε for Type II is defined from the stopband attenuation:
    //   ε = 1 / sqrt(10^(As/10) − 1)
    let epsilon = 1.0 / (10.0_f64.powf(stopband_db / 10.0) - 1.0).sqrt();
    let mu = (1.0 / epsilon).asinh() / n;

    // Poles: reciprocal of the Chebyshev I ellipse poles (inverse Chebyshev).
    let mut poles = Vec::with_capacity(order);
    for k in 0..order {
        let theta = f64::PI() * (2.0f64.mul_add(k as f64, 1.0)) / (2.0 * n);
        let cheb1_re = -mu.sinh() * theta.sin();
        let cheb1_im = mu.cosh() * theta.cos();
        let cheb1 = Complex::new(cheb1_re, cheb1_im);
        // Reciprocal map s -> 1/s places poles for the inverse filter and
        // keeps the stopband edge at ω = 1.
        poles.push(Complex::new(1.0, 0.0) / cheb1);
    }

    // Zeros: on the imaginary axis at ± j / cos((2k+1)π/2N), for k = 0..N-1.
    // For odd k giving cos ≈ 0 (only when order is odd, the middle term) the
    // zero goes to infinity and is dropped.
    let mut zeros = Vec::with_capacity(order);
    for k in 0..order {
        let c = (f64::PI() * (2.0f64.mul_add(k as f64, 1.0)) / (2.0 * n)).cos();
        if c.abs() < 1e-12 {
            continue; // zero at infinity (odd order centre term)
        }
        zeros.push(Complex::new(0.0, 1.0 / c));
    }

    // Gain so the prototype passband (ω → 0) settles at unity:
    //   H(0) = gain · Π(−z) / Π(−p)  ⇒  gain = Π(−p) / Π(−z).
    let prod_poles = poles.iter().fold(Complex::new(1.0, 0.0), |acc, &p| acc * (-p));
    let prod_zeros = zeros.iter().fold(Complex::new(1.0, 0.0), |acc, &z| acc * (-z));
    let gain = (prod_poles / prod_zeros).re;

    AnalogZpk { zeros, poles, gain }
}

/// Analog low-pass → low-pass scaling: s → s / ω0.
///
/// Scales every root by ω0 and adjusts the gain by the degree deficit so the
/// transfer function stays consistent (each missing finite zero contributes a
/// factor of ω0 to the gain).
fn lp_to_lp(zpk: &AnalogZpk, omega0: f64) -> AnalogZpk {
    let zeros: Vec<_> = zpk.zeros.iter().map(|&z| z * omega0).collect();
    let poles: Vec<_> = zpk.poles.iter().map(|&p| p * omega0).collect();
    let degree = zpk.poles.len() as i32 - zpk.zeros.len() as i32;
    let gain = zpk.gain * omega0.powi(degree);
    AnalogZpk { zeros, poles, gain }
}

/// Analog low-pass → high-pass: s → ω0 / s.
///
/// Maps poles/zeros to ω0/root and adds zeros at the origin to fill the degree
/// deficit, matching SciPy's `lp2hp_zpk`.
fn lp_to_hp(zpk: &AnalogZpk, omega0: f64) -> AnalogZpk {
    let degree = zpk.poles.len() as i32 - zpk.zeros.len() as i32;

    let mut zeros: Vec<_> = zpk
        .zeros
        .iter()
        .map(|&z| Complex::new(omega0, 0.0) / z)
        .collect();
    let poles: Vec<_> = zpk
        .poles
        .iter()
        .map(|&p| Complex::new(omega0, 0.0) / p)
        .collect();
    // Fill the missing zeros (that were at infinity) with zeros at the origin.
    for _ in 0..degree {
        zeros.push(Complex::new(0.0, 0.0));
    }

    // Gain: H_hp(s) = H_lp(ω0/s); the gain folds through as
    //   k · Π(−z_lp) / Π(−p_lp)  (real).
    let prod_z = zpk
        .zeros
        .iter()
        .fold(Complex::new(1.0, 0.0), |acc, &z| acc * (-z));
    let prod_p = zpk
        .poles
        .iter()
        .fold(Complex::new(1.0, 0.0), |acc, &p| acc * (-p));
    let gain = zpk.gain * (prod_z / prod_p).re;

    AnalogZpk { zeros, poles, gain }
}

/// Analog low-pass → band-pass: s → (s² + ω0²) / (s·BW).
///
/// `omega0` is the geometric-centre frequency, `bw` the bandwidth, both in
/// rad/s. Each prototype root splits into two; the gain is scaled by BW^degree;
/// `degree` zeros are added at the origin. Mirrors SciPy's `lp2bp_zpk`.
fn lp_to_bp(zpk: &AnalogZpk, omega0: f64, bw: f64) -> AnalogZpk {
    let degree = zpk.poles.len() as i32 - zpk.zeros.len() as i32;

    // Scale roots by BW/2, then each splits via s = scaled ± sqrt(scaled² − ω0²).
    let split = |root: Complex<f64>| -> (Complex<f64>, Complex<f64>) {
        let scaled = root * (bw / 2.0);
        let disc = (scaled * scaled - omega0 * omega0).sqrt();
        (scaled + disc, scaled - disc)
    };

    let mut zeros = Vec::with_capacity(zpk.zeros.len() * 2 + degree as usize);
    for &z in &zpk.zeros {
        let (a, b) = split(z);
        zeros.push(a);
        zeros.push(b);
    }
    // Missing finite zeros map to the origin.
    for _ in 0..degree {
        zeros.push(Complex::new(0.0, 0.0));
    }

    let mut poles = Vec::with_capacity(zpk.poles.len() * 2);
    for &p in &zpk.poles {
        let (a, b) = split(p);
        poles.push(a);
        poles.push(b);
    }

    let gain = zpk.gain * bw.powi(degree);
    AnalogZpk { zeros, poles, gain }
}

/// Analog low-pass → band-stop: s → (s·BW) / (s² + ω0²).
///
/// `omega0` is the geometric centre, `bw` the bandwidth. Each root inverts and
/// splits; the deficit is filled with `degree` conjugate pairs of zeros at
/// ±jω0 (the notch). Mirrors SciPy's `lp2bs_zpk`.
fn lp_to_bs(zpk: &AnalogZpk, omega0: f64, bw: f64) -> AnalogZpk {
    let degree = zpk.poles.len() as i32 - zpk.zeros.len() as i32;

    // Invert then split: root -> (BW/2)/root ± sqrt(((BW/2)/root)² − ω0²).
    let split = |root: Complex<f64>| -> (Complex<f64>, Complex<f64>) {
        let inv = Complex::new(bw / 2.0, 0.0) / root;
        let disc = (inv * inv - omega0 * omega0).sqrt();
        (inv + disc, inv - disc)
    };

    let mut zeros = Vec::with_capacity(zpk.zeros.len() * 2 + (degree as usize) * 2);
    for &z in &zpk.zeros {
        let (a, b) = split(z);
        zeros.push(a);
        zeros.push(b);
    }
    // Missing finite zeros map to ±jω0 (the band-stop notch).
    for _ in 0..degree {
        zeros.push(Complex::new(0.0, omega0));
        zeros.push(Complex::new(0.0, -omega0));
    }

    let mut poles = Vec::with_capacity(zpk.poles.len() * 2);
    for &p in &zpk.poles {
        let (a, b) = split(p);
        poles.push(a);
        poles.push(b);
    }

    // Gain folds through as k · Π(−z_lp)/Π(−p_lp) (real).
    let prod_z = zpk
        .zeros
        .iter()
        .fold(Complex::new(1.0, 0.0), |acc, &z| acc * (-z));
    let prod_p = zpk
        .poles
        .iter()
        .fold(Complex::new(1.0, 0.0), |acc, &p| acc * (-p));
    let gain = zpk.gain * (prod_z / prod_p).re;

    AnalogZpk { zeros, poles, gain }
}

/// Bilinear-transform an analog ZPK into a digital ZPK.
///
/// Uses z = (1 + s/(2fs)) / (1 − s/(2fs)). Finite zeros and poles map through
/// [`bilinear_transform_pole`]; the analog gain folds through the bilinear
/// Jacobian, and zeros missing relative to poles map to z = −1.
fn bilinear_zpk(zpk: &AnalogZpk, sample_rate: f64) -> AnalogZpk {
    let fs2 = 2.0 * sample_rate;
    let degree = zpk.poles.len() as i32 - zpk.zeros.len() as i32;

    let zeros_z: Vec<_> = zpk
        .zeros
        .iter()
        .map(|&z| bilinear_transform_pole(z, sample_rate))
        .collect();
    let poles_z: Vec<_> = zpk
        .poles
        .iter()
        .map(|&p| bilinear_transform_pole(p, sample_rate))
        .collect();

    // Gain correction: k_z = k_s · Π(2fs − z_s) / Π(2fs − p_s)  (real for
    // conjugate-symmetric root sets).
    let num = zpk
        .zeros
        .iter()
        .fold(Complex::new(1.0, 0.0), |acc, &z| acc * (Complex::new(fs2, 0.0) - z));
    let den = zpk
        .poles
        .iter()
        .fold(Complex::new(1.0, 0.0), |acc, &p| acc * (Complex::new(fs2, 0.0) - p));
    let gain = zpk.gain * (num / den).re;

    // Append the missing zeros at z = −1.
    let mut zeros = zeros_z;
    for _ in 0..degree {
        zeros.push(Complex::new(-1.0, 0.0));
    }

    AnalogZpk {
        zeros,
        poles: poles_z,
        gain,
    }
}

/// Pair complex-conjugate roots (zeros or poles) into ordered pairs.
///
/// Like [`pair_poles`] but generic over roots that may be zeros. Real roots are
/// paired with another available real root when possible, otherwise duplicated.
fn pair_roots(roots: &[Complex<f64>]) -> Vec<(Complex<f64>, Complex<f64>)> {
    let mut pairs = Vec::new();
    let mut used = vec![false; roots.len()];

    for i in 0..roots.len() {
        if used[i] {
            continue;
        }
        if roots[i].im.abs() < 1e-9 {
            // Real root: try to pair with another unused real root.
            used[i] = true;
            let mut partner = None;
            for j in (i + 1)..roots.len() {
                if !used[j] && roots[j].im.abs() < 1e-9 {
                    partner = Some(j);
                    break;
                }
            }
            match partner {
                Some(j) => {
                    used[j] = true;
                    pairs.push((roots[i], roots[j]));
                }
                None => pairs.push((roots[i], Complex::new(0.0, 0.0))),
            }
        } else {
            // Complex root: find its conjugate.
            used[i] = true;
            let mut found = false;
            for j in (i + 1)..roots.len() {
                if used[j] {
                    continue;
                }
                if (roots[i].conj() - roots[j]).norm() < 1e-7 {
                    used[j] = true;
                    pairs.push((roots[i], roots[j]));
                    found = true;
                    break;
                }
            }
            if !found {
                pairs.push((roots[i], roots[i].conj()));
            }
        }
    }
    pairs
}

/// Reference frequency (Hz) at which a response type should be normalised to
/// unit gain, given the digital design parameters.
#[derive(Clone, Copy)]
enum GainRef {
    Dc,
    Nyquist,
    Hz(f64),
    /// Do not renormalise: trust the gain carried through the bilinear
    /// transform. Used by elliptic filters, whose passband is equiripple and
    /// whose DC/edge value is intentionally offset by the ripple bound, so
    /// pinning |H| = 1 at any single reference would shift the whole passband.
    Prototype,
}

/// Assemble a digital ZPK into an [`SosFilter`], normalising the overall gain to
/// 1.0 at the supplied reference frequency.
///
/// Zeros and poles are paired independently and zipped into biquad sections;
/// any leftover unpaired set (when counts differ) is padded so each section is
/// a valid second-order block. The cascade gain is applied to the first
/// section, then corrected so |H| = 1 at `gain_ref`.
fn zpk_to_sos(zpk: &AnalogZpk, sample_rate: f64, gain_ref: GainRef) -> SosFilter {
    let pole_pairs = pair_roots(&zpk.poles);
    let zero_pairs = pair_roots(&zpk.zeros);

    let n_sections = pole_pairs.len().max(zero_pairs.len()).max(1);
    let mut sections = Vec::with_capacity(n_sections);

    for i in 0..n_sections {
        let (p1, p2) = pole_pairs
            .get(i)
            .copied()
            .unwrap_or((Complex::new(0.0, 0.0), Complex::new(0.0, 0.0)));
        let (z1, z2) = zero_pairs
            .get(i)
            .copied()
            .unwrap_or((Complex::new(0.0, 0.0), Complex::new(0.0, 0.0)));
        let (b, a) = biquad_from_poles_zeros(p1, p2, z1, z2, 1.0);
        sections.push(IirFilter::new(b, a));
    }

    // Apply the overall gain to the first section's numerator.
    for coeff in &mut sections[0].b_coeffs {
        *coeff *= zpk.gain;
    }

    let mut sos = SosFilter::new(sections);

    // Normalise to unit gain at the reference frequency, unless the prototype
    // gain is already calibrated (elliptic) and must be preserved.
    let ref_freq = match gain_ref {
        GainRef::Dc => 0.0,
        GainRef::Nyquist => sample_rate / 2.0,
        GainRef::Hz(f) => f,
        GainRef::Prototype => return sos,
    };
    let (mag, _) = sos.frequency_response(&[ref_freq], sample_rate);
    if mag[0].abs() > 1e-12 {
        let correction = 1.0 / mag[0];
        for coeff in &mut sos.sections[0].b_coeffs {
            *coeff *= correction;
        }
    }

    sos
}

/// Pre-warp a frequency (Hz) for the bilinear transform, returning rad/s.
fn prewarp(freq_hz: f64, sample_rate: f64) -> f64 {
    2.0 * sample_rate * (f64::PI() * freq_hz / sample_rate).tan()
}

/// Design an SOS filter from a normalised analog prototype and a target
/// response, doing all frequency transforms in the analog domain before the
/// bilinear transform.
fn design_from_prototype(
    proto: &AnalogZpk,
    response: FilterResponse,
    sample_rate: f64,
    cutoff: Option<f64>,
    low: Option<f64>,
    high: Option<f64>,
) -> SosFilter {
    design_from_prototype_with_gain(proto, response, sample_rate, cutoff, low, high, None)
}

/// As [`design_from_prototype`], but with an optional [`GainRef`] override.
///
/// When `gain_override` is `Some`, that reference is used instead of the
/// response-type default. Elliptic filters pass [`GainRef::Prototype`] so the
/// carefully calibrated equiripple gain survives unchanged.
fn design_from_prototype_with_gain(
    proto: &AnalogZpk,
    response: FilterResponse,
    sample_rate: f64,
    cutoff: Option<f64>,
    low: Option<f64>,
    high: Option<f64>,
    gain_override: Option<GainRef>,
) -> SosFilter {
    match response {
        FilterResponse::LowPass => {
            let wc = prewarp(cutoff.expect("lowpass needs cutoff"), sample_rate);
            let analog = lp_to_lp(proto, wc);
            let digital = bilinear_zpk(&analog, sample_rate);
            zpk_to_sos(&digital, sample_rate, gain_override.unwrap_or(GainRef::Dc))
        }
        FilterResponse::HighPass => {
            let wc = prewarp(cutoff.expect("highpass needs cutoff"), sample_rate);
            let analog = lp_to_hp(proto, wc);
            let digital = bilinear_zpk(&analog, sample_rate);
            zpk_to_sos(
                &digital,
                sample_rate,
                gain_override.unwrap_or(GainRef::Nyquist),
            )
        }
        FilterResponse::BandPass => {
            let lo = low.expect("bandpass needs low");
            let hi = high.expect("bandpass needs high");
            let w_lo = prewarp(lo, sample_rate);
            let w_hi = prewarp(hi, sample_rate);
            let omega0 = (w_lo * w_hi).sqrt();
            let bw = w_hi - w_lo;
            let analog = lp_to_bp(proto, omega0, bw);
            let digital = bilinear_zpk(&analog, sample_rate);
            // Normalise at the geometric-centre frequency (Hz).
            let center_hz = (lo * hi).sqrt();
            zpk_to_sos(
                &digital,
                sample_rate,
                gain_override.unwrap_or(GainRef::Hz(center_hz)),
            )
        }
        FilterResponse::BandStop => {
            let lo = low.expect("bandstop needs low");
            let hi = high.expect("bandstop needs high");
            let w_lo = prewarp(lo, sample_rate);
            let w_hi = prewarp(hi, sample_rate);
            let omega0 = (w_lo * w_hi).sqrt();
            let bw = w_hi - w_lo;
            let analog = lp_to_bs(proto, omega0, bw);
            let digital = bilinear_zpk(&analog, sample_rate);
            // Band-stop passes DC, so normalise there.
            zpk_to_sos(&digital, sample_rate, gain_override.unwrap_or(GainRef::Dc))
        }
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
    /// - [crate::AudioSampleError::Parameter] – if any frequency in `design`
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
    /// assert!(audio.apply_iir_filter_in_place(&design).is_ok());
    /// ```
    #[inline]
    fn apply_iir_filter_in_place(&mut self, design: &IirFilterDesign) -> AudioSampleResult<()> {
        let sample_rate = self.sample_rate_hz();
        let filter_repr = design_iir_filter(design, sample_rate)?;

        // Process sample-by-sample. The IIR difference equation is inherently
        // sequential (each output depends on previous outputs), so batching
        // gains nothing. We use Direct Form II Transposed biquad sections:
        // stack-allocated state, pre-normalised coefficients, no division.
        match filter_repr {
            FilterRepresentation::Single(mut bq) => match self.data_mut() {
                AudioData::Mono(samples) => {
                    for sample in samples.iter_mut() {
                        let x: f64 = (*sample).convert_to();
                        *sample = bq.process(x).convert_to();
                    }
                }
                AudioData::Multi(data) => {
                    for ch_idx in 0..data.dim().0.get() {
                        let mut channel = data.index_axis_mut(Axis(0), ch_idx);
                        for sample in channel.iter_mut() {
                            let x: f64 = (*sample).convert_to();
                            *sample = bq.process(x).convert_to();
                        }
                        bq.reset();
                    }
                }
            },
            FilterRepresentation::Cascade(mut bqs) => match self.data_mut() {
                AudioData::Mono(samples) => {
                    // Per-sample-all-sections: each sample flows through every biquad
                    // while staying in registers, so the buffer is touched only once.
                    // (A section-major variant that streams the whole buffer per section
                    // was measured ~1.8x SLOWER — it adds a read+write memory pass per
                    // section, and the biquad recurrence is sequential so register
                    // residency of the coefficients buys nothing.)
                    for sample in samples.iter_mut() {
                        let mut x: f64 = (*sample).convert_to();
                        for bq in bqs.iter_mut() {
                            x = bq.process(x);
                        }
                        *sample = x.convert_to();
                    }
                }
                AudioData::Multi(data) => {
                    for ch_idx in 0..data.dim().0.get() {
                        let mut channel = data.index_axis_mut(Axis(0), ch_idx);
                        for sample in channel.iter_mut() {
                            let mut x: f64 = (*sample).convert_to();
                            for bq in bqs.iter_mut() {
                                x = bq.process(x);
                            }
                            *sample = x.convert_to();
                        }
                        for bq in bqs.iter_mut() {
                            bq.reset();
                        }
                    }
                }
            },
        }

        Ok(())
    }

    /// Apply zero-phase (forward-backward) IIR filtering in place.
    ///
    /// See [`AudioIirFiltering::filtfilt_in_place`] for the full description.
    #[inline]
    fn filtfilt_in_place(&mut self, design: &IirFilterDesign) -> AudioSampleResult<()> {
        let sample_rate = self.sample_rate_hz();
        // Design once; clone the (reset) cascade per channel so each channel
        // starts from a clean state, mirroring the per-channel reset used by
        // `apply_iir_filter_in_place` and `apply_eq_band_in_place`.
        let mut sos = SosFilter::from_design(design, sample_rate)?;

        match self.data_mut() {
            AudioData::Mono(samples) => {
                let mut buf: Vec<f64> = samples.iter().map(|&s| s.convert_to()).collect();
                filtfilt_slice(&mut sos, &mut buf);
                for (dst, &v) in samples.iter_mut().zip(buf.iter()) {
                    *dst = v.convert_to();
                }
            }
            AudioData::Multi(data) => {
                for ch_idx in 0..data.dim().0.get() {
                    let mut channel = data.index_axis_mut(Axis(0), ch_idx);
                    let mut buf: Vec<f64> =
                        channel.iter().map(|&s| s.convert_to()).collect();
                    filtfilt_slice(&mut sos.clone(), &mut buf);
                    for (dst, &v) in channel.iter_mut().zip(buf.iter()) {
                        *dst = v.convert_to();
                    }
                }
            }
        }

        Ok(())
    }

    /// Apply a Butterworth low-pass filter.
    ///
    /// Convenience wrapper that constructs an [`IirFilterDesign`] and
    /// delegates to [`Self::apply_iir_filter`]. Supports arbitrary filter
    /// orders up to 12. Order 2 uses direct-form implementation; higher
    /// orders automatically use a numerically stable second-order sections
    /// (SOS) cascade.
    ///
    /// # Arguments
    /// - `order` – Filter order (number of poles); must be ≤ 12.
    /// - `cutoff_frequency` – Cutoff frequency in Hz; must be in
    ///   (0, Nyquist).
    ///
    /// # Returns
    /// `Ok(())` on success.
    ///
    /// # Errors
    /// - [crate::AudioSampleError::Parameter] – if `cutoff_frequency` is
    ///   outside the valid range, or if `order` > 12.
    ///
    /// # Examples
    /// ```
    /// use audio_samples::{AudioSamples, sample_rate};
    /// use audio_samples::operations::traits::AudioIirFiltering;
    /// use non_empty_slice::NonEmptyVec;
    /// use std::num::NonZeroUsize;
    ///
    /// // Order-2 Butterworth lowpass
    /// let samples = NonEmptyVec::new(vec![1.0f32, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0]).unwrap();
    /// let mut audio: AudioSamples<'_, f32> = AudioSamples::from_mono_vec(samples, sample_rate!(44100));
    /// assert!(audio.butterworth_lowpass_in_place(NonZeroUsize::new(2).unwrap(), 1000.0).is_ok());
    ///
    /// // High-order (8th-order) Butterworth lowpass
    /// let samples2 = NonEmptyVec::new(vec![1.0f32; 200]).unwrap();
    /// let mut audio2: AudioSamples<'_, f32> = AudioSamples::from_mono_vec(samples2, sample_rate!(44100));
    /// assert!(audio2.butterworth_lowpass_in_place(NonZeroUsize::new(8).unwrap(), 2000.0).is_ok());
    /// ```
    #[inline]
    fn butterworth_lowpass_in_place(
        &mut self,
        order: NonZeroUsize,
        cutoff_frequency: f64,
    ) -> AudioSampleResult<()> {
        let design = IirFilterDesign::butterworth_lowpass(order, cutoff_frequency);
        self.apply_iir_filter_in_place(&design)
    }

    /// Apply a Butterworth high-pass filter.
    ///
    /// Convenience wrapper that constructs an [`IirFilterDesign`] and
    /// delegates to [`Self::apply_iir_filter`]. Supports arbitrary filter
    /// orders up to 12. Order 2 uses direct-form implementation; higher
    /// orders automatically use a numerically stable second-order sections
    /// (SOS) cascade.
    ///
    /// # Arguments
    /// - `order` – Filter order (number of poles); must be ≤ 12.
    /// - `cutoff_frequency` – Cutoff frequency in Hz; must be in
    ///   (0, Nyquist).
    ///
    /// # Returns
    /// `Ok(())` on success.
    ///
    /// # Errors
    /// - [crate::AudioSampleError::Parameter] – if `cutoff_frequency` is
    ///   outside the valid range, or if `order` > 12.
    ///
    /// # Examples
    /// ```
    /// use audio_samples::{AudioSamples, sample_rate};
    /// use audio_samples::operations::traits::AudioIirFiltering;
    /// use non_empty_slice::NonEmptyVec;
    /// use std::num::NonZeroUsize;
    ///
    /// // Order-2 Butterworth highpass
    /// let samples = NonEmptyVec::new(vec![1.0f32, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]).unwrap();
    /// let mut audio: AudioSamples<'_, f32> = AudioSamples::from_mono_vec(samples, sample_rate!(44100));
    /// assert!(audio.butterworth_highpass_in_place(NonZeroUsize::new(2).unwrap(), 500.0).is_ok());
    ///
    /// // High-order (6th-order) Butterworth highpass
    /// let samples2 = NonEmptyVec::new(vec![1.0f32; 200]).unwrap();
    /// let mut audio2: AudioSamples<'_, f32> = AudioSamples::from_mono_vec(samples2, sample_rate!(44100));
    /// assert!(audio2.butterworth_highpass_in_place(NonZeroUsize::new(6).unwrap(), 1500.0).is_ok());
    /// ```
    #[inline]
    fn butterworth_highpass_in_place(
        &mut self,
        order: NonZeroUsize,
        cutoff_frequency: f64,
    ) -> AudioSampleResult<()> {
        let design = IirFilterDesign::butterworth_highpass(order, cutoff_frequency);
        self.apply_iir_filter_in_place(&design)
    }

    /// Apply a Butterworth band-pass filter.
    ///
    /// Convenience wrapper that constructs an [`IirFilterDesign`] and
    /// delegates to [`Self::apply_iir_filter`]. The bandpass filter is
    /// implemented by cascading lowpass and highpass sections, effectively
    /// doubling the filter order. Supports orders up to 12 (per section).
    ///
    /// # Arguments
    /// - `order` – Filter order per section; must be ≤ 12. The effective
    ///   total order is approximately 2 × `order`.
    /// - `low_frequency` – Lower cutoff frequency in Hz; must be > 0.
    /// - `high_frequency` – Upper cutoff frequency in Hz; must be < Nyquist
    ///   and > `low_frequency`.
    ///
    /// # Returns
    /// `Ok(())` on success.
    ///
    /// # Errors
    /// - [crate::AudioSampleError::Parameter] – if the frequency range is
    ///   invalid, or if `order` > 12.
    ///
    /// # Examples
    /// ```
    /// use audio_samples::{AudioSamples, sample_rate};
    /// use audio_samples::operations::traits::AudioIirFiltering;
    /// use non_empty_slice::NonEmptyVec;
    /// use std::num::NonZeroUsize;
    ///
    /// // Order-2 bandpass (effective order ≈ 4)
    /// let samples = NonEmptyVec::new(vec![1.0f32, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0]).unwrap();
    /// let mut audio: AudioSamples<'_, f32> = AudioSamples::from_mono_vec(samples, sample_rate!(44100));
    /// assert!(audio.butterworth_bandpass_in_place(NonZeroUsize::new(2).unwrap(), 100.0, 5000.0).is_ok());
    ///
    /// // High-order (4th-order per section) bandpass
    /// let samples2 = NonEmptyVec::new(vec![1.0f32; 200]).unwrap();
    /// let mut audio2: AudioSamples<'_, f32> = AudioSamples::from_mono_vec(samples2, sample_rate!(44100));
    /// assert!(audio2.butterworth_bandpass_in_place(NonZeroUsize::new(4).unwrap(), 800.0, 1200.0).is_ok());
    /// ```
    #[inline]
    fn butterworth_bandpass_in_place(
        &mut self,
        order: NonZeroUsize,
        low_frequency: f64,
        high_frequency: f64,
    ) -> AudioSampleResult<()> {
        let design = IirFilterDesign::butterworth_bandpass(order, low_frequency, high_frequency);
        self.apply_iir_filter_in_place(&design)
    }

    /// Apply a Chebyshev Type I filter.
    ///
    /// Chebyshev Type I filters offer sharper roll-off than Butterworth
    /// filters of the same order at the cost of passband ripple. The
    /// `passband_ripple` parameter controls the amount of ripple in dB
    /// (typically 0.01 to 5.0 dB). Smaller ripple values approach
    /// Butterworth response; larger values give steeper roll-off.
    ///
    /// All filter orders use a numerically stable second-order sections
    /// (SOS) cascade. Orders are limited to ≤ 12 for stability.
    ///
    /// # Arguments
    /// - `order` – Filter order (number of poles); must be ≤ 12.
    /// - `cutoff_frequency` – Cutoff frequency in Hz; must be in
    ///   (0, 0.95 × Nyquist).
    /// - `passband_ripple` – Maximum passband ripple in dB; must be
    ///   in (0, 5.0]. Typical values: 0.5 dB (moderate ripple),
    ///   1.0 dB (aggressive roll-off).
    /// - `response` – Filter response type: [`FilterResponse::LowPass`],
    ///   [`FilterResponse::HighPass`], or [`FilterResponse::BandPass`].
    ///
    /// # Returns
    /// `Ok(())` on success.
    ///
    /// # Errors
    /// - [crate::AudioSampleError::Parameter] – if `cutoff_frequency` is out of range,
    ///   `passband_ripple` is invalid, `order` > 12, or `response` is
    ///   [`FilterResponse::BandStop`] (not yet implemented).
    ///
    /// # Examples
    /// ```
    /// use audio_samples::{AudioSamples, sample_rate};
    /// use audio_samples::operations::traits::AudioIirFiltering;
    /// use audio_samples::operations::types::FilterResponse;
    /// use non_empty_slice::NonEmptyVec;
    /// use std::num::NonZeroUsize;
    ///
    /// // 4th-order Chebyshev Type I lowpass with 0.5 dB ripple
    /// let samples = NonEmptyVec::new(vec![1.0f32, 0.0, -1.0, 0.0, 1.0, 0.0, -1.0, 0.0]).unwrap();
    /// let mut audio: AudioSamples<'_, f32> = AudioSamples::from_mono_vec(samples, sample_rate!(44100));
    /// assert!(audio.chebyshev_i_in_place(
    ///     NonZeroUsize::new(4).unwrap(),
    ///     1000.0,
    ///     0.5,
    ///     FilterResponse::LowPass,
    /// ).is_ok());
    ///
    /// // 6th-order Chebyshev Type I highpass with 1.0 dB ripple
    /// let samples2 = NonEmptyVec::new(vec![1.0f32; 200]).unwrap();
    /// let mut audio2: AudioSamples<'_, f32> = AudioSamples::from_mono_vec(samples2, sample_rate!(44100));
    /// assert!(audio2.chebyshev_i_in_place(
    ///     NonZeroUsize::new(6).unwrap(),
    ///     2000.0,
    ///     1.0,
    ///     FilterResponse::HighPass,
    /// ).is_ok());
    /// ```
    #[inline]
    fn chebyshev_i_in_place(
        &mut self,
        order: NonZeroUsize,
        cutoff_frequency: f64,
        passband_ripple: f64,
        response: FilterResponse,
    ) -> AudioSampleResult<()> {
        let design =
            IirFilterDesign::chebyshev_i(response, order, cutoff_frequency, passband_ripple);
        self.apply_iir_filter_in_place(&design)
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
    #[inline]
    fn frequency_response(&self, frequencies: &[f64]) -> AudioSampleResult<(Vec<f64>, Vec<f64>)> {
        // For now, return a placeholder implementation
        // In a full implementation, this would store the current filter state
        // and return its frequency response
        Ok((vec![1.0; frequencies.len()], vec![0.0; frequencies.len()]))
    }
}

// ============================================================================
// Butterworth SOS filter design
// ============================================================================

/// Generate analog Butterworth poles on a circle in the s-plane.
///
/// Butterworth poles are evenly distributed on a circle of radius `omega_c`
/// in the left half of the s-plane. For order N, poles are at angles
/// θk = π(1/2 + (2k+1)/(2N)) for k = 0, 1, ..., N-1.
/// This ensures θk ∈ (π/2, 3π/2), placing all poles in the left half-plane.
///
/// # Arguments
/// - `order` – Filter order (number of poles).
/// - `omega_c` – Pre-warped cutoff frequency (radians/second).
///
/// # Returns
/// Vector of N complex analog poles.
fn butterworth_analog_poles(order: usize, omega_c: f64) -> Vec<Complex<f64>> {
    let mut poles = Vec::with_capacity(order);
    let order_f = order as f64;

    for k in 0..order {
        // θk = π(1/2 + (2k+1)/(2N))
        let theta = f64::PI() * (0.5 + 2.0f64.mul_add(k as f64, 1.0) / (2.0 * order_f));
        let real = omega_c * theta.cos(); // cos(θ) < 0 for θ ∈ (π/2, 3π/2)
        let imag = omega_c * theta.sin(); // sin(θ) varies
        poles.push(Complex::new(real, imag));
    }

    poles
}

/// Design a Butterworth lowpass filter as a second-order sections cascade.
///
/// Uses analog pole generation, bilinear transform, and pole pairing to
/// construct a numerically stable SOS representation. The DC gain is
/// normalized to 1.0.
///
/// # Arguments
/// - `order` – Filter order (number of poles).
/// - `cutoff` – Cutoff frequency in Hz.
/// - `sample_rate` – Sample rate in Hz.
///
/// # Returns
/// A [`SosFilter`] implementing the Butterworth lowpass response.
fn design_butterworth_lowpass_sos(order: usize, cutoff: f64, sample_rate: f64) -> SosFilter {
    // Pre-warp the cutoff frequency for bilinear transform
    let omega_c = 2.0 * sample_rate * (f64::PI() * cutoff / sample_rate).tan();

    // Generate analog poles
    let analog_poles = butterworth_analog_poles(order, omega_c);

    // Convert to digital poles using bilinear transform
    let digital_poles: Vec<Complex<f64>> = analog_poles
        .iter()
        .map(|&p| bilinear_transform_pole(p, sample_rate))
        .collect();

    // Pair poles for biquad construction
    let pole_pairs = pair_poles(digital_poles);

    // Construct biquad sections
    let mut sections = Vec::new();
    for (p1, p2) in pole_pairs {
        // Lowpass: zeros at z = -1 (Nyquist frequency)
        let z1 = Complex::new(-1.0, 0.0);
        let z2 = Complex::new(-1.0, 0.0);

        // Temporary gain (will normalize later)
        let (b, a) = biquad_from_poles_zeros(p1, p2, z1, z2, 1.0);
        sections.push(IirFilter::new(b, a));
    }

    // Normalize DC gain to 1.0
    // Evaluate response at DC (freq = 0 Hz)
    let mut sos = SosFilter::new(sections);
    let (dc_response, _) = sos.frequency_response(&[0.0], sample_rate);
    let dc_gain = dc_response[0];

    // Apply gain correction to first section
    if dc_gain.abs() > 1e-10 {
        let gain_correction = 1.0 / dc_gain;
        for coeff in &mut sos.sections[0].b_coeffs {
            *coeff *= gain_correction;
        }
    }

    sos
}

/// Transform a lowpass SOS filter to highpass using spectral inversion.
///
/// Applies the transformation z → -z, which flips the frequency response
/// around Nyquist/2. This is done by negating the odd-indexed b and a
/// coefficients in each biquad section.
///
/// # Arguments
/// - `lowpass` – A lowpass [`SosFilter`] to transform.
///
/// # Returns
/// A highpass [`SosFilter`] with the same order and cutoff.
fn lowpass_to_highpass_sos(mut lowpass: SosFilter) -> SosFilter {
    for section in &mut lowpass.sections {
        // Negate odd-indexed coefficients (indices 1, 3, 5, ...)
        for i in (1..section.b_coeffs.len()).step_by(2) {
            section.b_coeffs[i] = -section.b_coeffs[i];
        }
        for i in (1..section.a_coeffs.len()).step_by(2) {
            section.a_coeffs[i] = -section.a_coeffs[i];
        }
    }
    lowpass
}

/// Design a Butterworth highpass filter as a second-order sections cascade.
///
/// Designs a lowpass filter and applies spectral inversion to convert it
/// to highpass. The gain at Nyquist is normalized to 1.0.
///
/// # Arguments
/// - `order` – Filter order (number of poles).
/// - `cutoff` – Cutoff frequency in Hz.
/// - `sample_rate` – Sample rate in Hz.
///
/// # Returns
/// A [`SosFilter`] implementing the Butterworth highpass response.
fn design_butterworth_highpass_sos(order: usize, cutoff: f64, sample_rate: f64) -> SosFilter {
    let lowpass = design_butterworth_lowpass_sos(order, cutoff, sample_rate);
    let mut highpass = lowpass_to_highpass_sos(lowpass);

    // Normalize Nyquist gain to 1.0
    let nyquist = sample_rate / 2.0;
    let (nyq_response, _) = highpass.frequency_response(&[nyquist], sample_rate);
    let nyq_gain = nyq_response[0];

    if nyq_gain.abs() > 1e-10 {
        let gain_correction = 1.0 / nyq_gain;
        for coeff in &mut highpass.sections[0].b_coeffs {
            *coeff *= gain_correction;
        }
    }

    highpass
}

// Butterworth band-pass / band-stop are now produced by the analog-prototype
// pipeline (`butterworth_prototype` → `lp_to_bp`/`lp_to_bs` → `bilinear_zpk`)
// in `design_butterworth_filter`, replacing the earlier cascaded-LP+HP
// approximation.

// ============================================================================
// Chebyshev Type I SOS filter design
// ============================================================================

/// Compute Chebyshev Type I filter parameters from passband ripple.
///
/// Calculates epsilon (ripple parameter) and ellipse semi-axes (a, b)
/// for Chebyshev pole placement.
///
/// # Arguments
/// - `order` – Filter order (number of poles).
/// - `ripple_db` – Passband ripple in dB (typically 0.01 to 5.0).
///
/// # Returns
/// `(epsilon, a, b)` where epsilon is the ripple parameter,
/// and (a, b) are the ellipse semi-axes for pole placement.
fn chebyshev1_params(order: usize, ripple_db: f64) -> (f64, f64, f64) {
    // ε = sqrt(10^(Rp/10) - 1)
    let epsilon = (10.0_f64.powf(ripple_db / 10.0) - 1.0).sqrt();

    // Semi-axes of the ellipse:
    // a = sinh(asinh(1/ε) / N)
    // b = cosh(asinh(1/ε) / N)
    let asinh_inv_eps = (1.0 / epsilon + (1.0 / epsilon).mul_add(1.0 / epsilon, 1.0).sqrt()).ln();
    let order_f = order as f64;

    let a = (asinh_inv_eps / order_f).sinh();
    let b = (asinh_inv_eps / order_f).cosh();

    (epsilon, a, b)
}

/// Generate analog Chebyshev Type I poles on an ellipse in the s-plane.
///
/// Chebyshev Type I poles lie on an ellipse (not a circle like Butterworth).
/// The ellipse shape is determined by the passband ripple parameter.
/// For order N, poles are at angles θk = π(1/2 + (2k+1)/(2N)) for k = 0...N-1.
///
/// # Arguments
/// - `order` – Filter order (number of poles).
/// - `omega_c` – Pre-warped cutoff frequency (radians/second).
/// - `ripple_db` – Passband ripple in dB.
///
/// # Returns
/// Vector of N complex analog poles.
fn chebyshev1_analog_poles(order: usize, omega_c: f64, ripple_db: f64) -> Vec<Complex<f64>> {
    let (_epsilon, a, b) = chebyshev1_params(order, ripple_db);

    let mut poles = Vec::with_capacity(order);
    let order_f = order as f64;

    for k in 0..order {
        // θk = π(1/2 + (2k+1)/(2N))
        let theta = f64::PI() * (0.5 + 2.0f64.mul_add(k as f64, 1.0) / (2.0 * order_f));

        // Poles on ellipse: s_k = a*ωc*cos(θ) + j*b*ωc*sin(θ)
        // (cos(θ) < 0 for θ ∈ (π/2, 3π/2), so real part is negative)
        let real = a * omega_c * theta.cos();
        let imag = b * omega_c * theta.sin();
        poles.push(Complex::new(real, imag));
    }

    poles
}

/// Design a Chebyshev Type I lowpass filter as a second-order sections cascade.
///
/// Uses elliptical pole generation, bilinear transform, and pole pairing.
/// The passband ripple is specified in dB (typically 0.01 to 5.0 dB).
/// The DC gain is normalized to 1.0.
///
/// # Arguments
/// - `order` – Filter order (number of poles).
/// - `cutoff` – Cutoff frequency in Hz (-3dB point for ripple = 0).
/// - `ripple_db` – Passband ripple in dB.
/// - `sample_rate` – Sample rate in Hz.
///
/// # Returns
/// A [`SosFilter`] implementing the Chebyshev Type I lowpass response.
fn design_chebyshev1_lowpass_sos(
    order: usize,
    cutoff: f64,
    ripple_db: f64,
    sample_rate: f64,
) -> SosFilter {
    // Pre-warp the cutoff frequency
    let omega_c = 2.0 * sample_rate * (f64::PI() * cutoff / sample_rate).tan();

    // Generate analog poles on ellipse
    let analog_poles = chebyshev1_analog_poles(order, omega_c, ripple_db);

    // Convert to digital poles using bilinear transform
    let digital_poles: Vec<Complex<f64>> = analog_poles
        .iter()
        .map(|&p| bilinear_transform_pole(p, sample_rate))
        .collect();

    // Pair poles for biquad construction
    let pole_pairs = pair_poles(digital_poles);

    // Construct biquad sections
    let mut sections = Vec::new();
    for (p1, p2) in pole_pairs {
        // Lowpass: zeros at z = -1 (Nyquist frequency)
        let z1 = Complex::new(-1.0, 0.0);
        let z2 = Complex::new(-1.0, 0.0);

        let (b, a) = biquad_from_poles_zeros(p1, p2, z1, z2, 1.0);
        sections.push(IirFilter::new(b, a));
    }

    // Normalize DC gain to 1.0
    let mut sos = SosFilter::new(sections);
    let (dc_response, _) = sos.frequency_response(&[0.0], sample_rate);
    let dc_gain = dc_response[0];

    if dc_gain.abs() > 1e-10 {
        let gain_correction = 1.0 / dc_gain;
        for coeff in &mut sos.sections[0].b_coeffs {
            *coeff *= gain_correction;
        }
    }

    sos
}

/// Design a Chebyshev Type I highpass filter as a second-order sections cascade.
///
/// Designs a lowpass filter and applies spectral inversion to convert to highpass.
///
/// # Arguments
/// - `order` – Filter order (number of poles).
/// - `cutoff` – Cutoff frequency in Hz.
/// - `ripple_db` – Passband ripple in dB.
/// - `sample_rate` – Sample rate in Hz.
///
/// # Returns
/// A [`SosFilter`] implementing the Chebyshev Type I highpass response.
fn design_chebyshev1_highpass_sos(
    order: usize,
    cutoff: f64,
    ripple_db: f64,
    sample_rate: f64,
) -> SosFilter {
    let lowpass = design_chebyshev1_lowpass_sos(order, cutoff, ripple_db, sample_rate);
    let mut highpass = lowpass_to_highpass_sos(lowpass);

    // Normalize Nyquist gain
    let nyquist = sample_rate / 2.0;
    let (nyq_response, _) = highpass.frequency_response(&[nyquist], sample_rate);
    let nyq_gain = nyq_response[0];

    if nyq_gain.abs() > 1e-10 {
        let gain_correction = 1.0 / nyq_gain;
        for coeff in &mut highpass.sections[0].b_coeffs {
            *coeff *= gain_correction;
        }
    }

    highpass
}

// Chebyshev Type I band-pass / band-stop are now produced by the
// analog-prototype pipeline (`chebyshev1_prototype` → `lp_to_bp`/`lp_to_bs` →
// `bilinear_zpk`) in `design_chebyshev_i_filter`, replacing the earlier
// cascaded-LP+HP approximation.

// ============================================================================
// Bessel (Bessel–Thomson) analog prototype
// ============================================================================
//
// The Bessel low-pass prototype is all-pole: its poles are the roots of the
// reverse Bessel polynomial θ_N(s) of degree N. We compute those roots
// numerically (Durand–Kerner), then frequency-normalise so the magnitude
// response is −3 dB at ω = 1 rad/s. This matches the default of older SciPy
// and `scipy.signal.bessel(..., norm='mag')`. The poles agree with SciPy to
// better than 1e-12 for orders ≤ 8 (validated in tests).

/// Reverse Bessel polynomial coefficients θ_N(x) = Σ a_k x^k, returned
/// highest-degree-first (a_N, a_{N-1}, …, a_0) for the root finder.
///
/// a_k = (2N − k)! / (2^(N−k) · k! · (N−k)!).
fn reverse_bessel_coeffs(order: usize) -> Vec<f64> {
    // Compute factorials as f64; orders are small (≤ 12) so 24! fits in f64
    // exactly enough for these well-conditioned roots.
    fn factorial(n: usize) -> f64 {
        (1..=n).fold(1.0_f64, |acc, i| acc * i as f64)
    }
    let n = order;
    let mut coeffs = Vec::with_capacity(n + 1);
    for k in 0..=n {
        let a_k =
            factorial(2 * n - k) / (2.0_f64.powi((n - k) as i32) * factorial(k) * factorial(n - k));
        coeffs.push(a_k);
    }
    coeffs.reverse(); // highest degree first
    coeffs
}

/// Find all complex roots of a polynomial via the Durand–Kerner (Weierstrass)
/// iteration. `coeffs` are highest-degree-first; the polynomial is made monic
/// internally. Robust for the well-conditioned Bessel polynomials used here.
fn durand_kerner(coeffs: &[f64]) -> Vec<Complex<f64>> {
    let lead = coeffs[0];
    let monic: Vec<Complex<f64>> = coeffs.iter().map(|&c| Complex::new(c / lead, 0.0)).collect();
    let n = monic.len() - 1;
    if n == 0 {
        return Vec::new();
    }

    // Horner evaluation of the monic polynomial.
    let poly = |x: Complex<f64>| -> Complex<f64> {
        let mut r = monic[0];
        for &a in &monic[1..] {
            r = r * x + a;
        }
        r
    };

    // Spread initial guesses around the complex plane to avoid degeneracy.
    let seed = Complex::new(0.4, 0.9);
    let mut roots: Vec<Complex<f64>> = (0..n).map(|k| seed.powu(k as u32)).collect();

    for _ in 0..1000 {
        let mut max_delta = 0.0_f64;
        let current = roots.clone();
        for i in 0..n {
            let mut den = Complex::new(1.0, 0.0);
            for j in 0..n {
                if j != i {
                    den *= current[i] - current[j];
                }
            }
            let delta = poly(current[i]) / den;
            roots[i] = current[i] - delta;
            max_delta = max_delta.max(delta.norm());
        }
        if max_delta < 1e-15 {
            break;
        }
    }
    roots
}

/// Magnitude of an all-pole transfer function H(s) = gain / Π(s − p) at s = jω.
fn allpole_mag(poles: &[Complex<f64>], gain: f64, omega: f64) -> f64 {
    let s = Complex::new(0.0, omega);
    let mut val = Complex::new(gain, 0.0);
    for &p in poles {
        val /= s - p;
    }
    val.norm()
}

/// Normalised analog Bessel low-pass prototype (−3 dB at ω = 1, all-pole).
///
/// Poles are the reverse-Bessel-polynomial roots scaled so that the magnitude
/// drops to 1/√2 at ω = 1 rad/s (SciPy `norm='mag'`). No finite zeros.
fn bessel_prototype(order: usize) -> AnalogZpk {
    let coeffs = reverse_bessel_coeffs(order);
    let raw_poles = durand_kerner(&coeffs);

    // Gain so H(0) = 1: gain = Π(−p).
    let gain0 = raw_poles
        .iter()
        .fold(Complex::new(1.0, 0.0), |acc, &p| acc * (-p))
        .re;

    // Locate the −3 dB frequency by bisection (magnitude is monotone-decreasing
    // for the Bessel low-pass), then scale every pole by 1/ω₀.
    let target = 1.0 / 2.0_f64.sqrt();
    let mut lo = 1e-3_f64;
    let mut hi = 100.0_f64;
    for _ in 0..200 {
        let mid = 0.5 * (lo + hi);
        if allpole_mag(&raw_poles, gain0, mid) > target {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    let omega0 = 0.5 * (lo + hi);

    let poles: Vec<Complex<f64>> = raw_poles.iter().map(|&p| p / omega0).collect();
    // Unit-gain prototype at DC: gain = Π(−p) of the scaled poles.
    let gain = poles
        .iter()
        .fold(Complex::new(1.0, 0.0), |acc, &p| acc * (-p))
        .re;

    AnalogZpk {
        zeros: Vec::new(),
        poles,
        gain,
    }
}

// ============================================================================
// Elliptic (Cauer) analog prototype
// ============================================================================
//
// Equiripple in BOTH passband and stopband. Implemented after Orfanidis'
// "Lecture Notes on Elliptic Filter Design" (the same algorithm SciPy's
// `ellipap` uses): the special functions K(m), the Jacobi functions sn/cn/dn,
// the modular degree equation and the inverse Jacobi sn are all evaluated via
// the AGM / Landen transformation. The resulting zeros, poles and gain match
// `scipy.signal.ellip(..., analog=True)` to ≈1e-12 (validated in tests).
//
// Throughout, the modulus convention is m = k² (as in SciPy), so e.g.
// `ellipk(m)` is the complete elliptic integral K with parameter m.

/// Complete elliptic integral of the first kind K(m), m = k², via AGM.
fn ellipk(m: f64) -> f64 {
    if m >= 1.0 {
        return f64::INFINITY;
    }
    let (mut a, mut b) = (1.0_f64, (1.0 - m).sqrt());
    for _ in 0..60 {
        if (a - b).abs() <= 1e-16 * a.abs() {
            break;
        }
        let an = 0.5 * (a + b);
        let bn = (a * b).sqrt();
        a = an;
        b = bn;
    }
    f64::PI() / (2.0 * a)
}

/// Jacobi elliptic functions (sn, cn, dn) for real argument `u`, modulus
/// `m = k²`, via the descending-Landen / AGM scale sequence.
fn ellipj(u: f64, m: f64) -> (f64, f64, f64) {
    if m == 0.0 {
        return (u.sin(), u.cos(), 1.0);
    }
    let mut a = vec![1.0_f64];
    let mut c = vec![m.sqrt()];
    let mut b = (1.0 - m).sqrt();
    let mut n = 0usize;
    while c[n].abs() > 1e-15 && n < 40 {
        let ai = a[n];
        let bi = b;
        a.push(0.5 * (ai + bi));
        c.push(0.5 * (ai - bi));
        b = (ai * bi).sqrt();
        n += 1;
    }
    let mut phi = (2.0_f64).powi(n as i32) * a[n] * u;
    for i in (1..=n).rev() {
        let arg = (c[i] / a[i] * phi.sin()).clamp(-1.0, 1.0);
        phi = 0.5 * (phi + arg.asin());
    }
    let sn = phi.sin();
    let cn = phi.cos();
    let dn = (1.0 - m * sn * sn).max(0.0).sqrt();
    (sn, cn, dn)
}

/// Solve the modular degree equation n·K(m)/K'(m) = K(m₁)/K'(m₁) for m, using
/// the nome series (Orfanidis Eq. 49). `m1` is the squared selectivity ck1².
fn ellipdeg(n: usize, m1: f64) -> f64 {
    const MMAX: usize = 7;
    let k1 = ellipk(m1);
    let k1p = ellipk(1.0 - m1);
    let q1 = (-f64::PI() * k1p / k1).exp();
    let q = q1.powf(1.0 / n as f64);

    let mut num = 0.0_f64;
    for i in 0..=MMAX {
        num += q.powi((i * (i + 1)) as i32);
    }
    let mut den = 1.0_f64;
    for i in 1..=(MMAX + 1) {
        den += 2.0 * q.powi((i * i) as i32);
    }
    16.0 * q * (num / den).powi(4)
}

/// Inverse Jacobi sn for complex argument `w`, modulus `m = k²`, via the
/// ascending Landen transformation (Orfanidis Eq. 56).
fn arc_jac_sn(w: Complex<f64>, m: f64) -> Complex<f64> {
    // (1 - kx²)^(1/2) for complex kx.
    let complement = |kx: Complex<f64>| ((Complex::new(1.0, 0.0) - kx) * (Complex::new(1.0, 0.0) + kx)).sqrt();

    let mut ks = vec![m.sqrt()];
    let mut k_ = m.sqrt();
    let mut n = 0usize;
    while k_ != 0.0 && n < 30 {
        let kp = complement(Complex::new(k_, 0.0)).re;
        k_ = (1.0 - kp) / (1.0 + kp);
        ks.push(k_);
        n += 1;
    }
    // K = Π(1 + k_i) · π/2 over the descended moduli (excluding the seed).
    let big_k: f64 = ks[1..].iter().fold(1.0, |acc, &kv| acc * (1.0 + kv)) * f64::PI() / 2.0;

    let mut wn = w;
    for idx in 0..(ks.len() - 1) {
        let kn = ks[idx];
        let knext = ks[idx + 1];
        wn = (2.0 * wn)
            / ((1.0 + knext) * (Complex::new(1.0, 0.0) + complement(Complex::new(kn, 0.0) * wn)));
    }
    // u = (2/π)·asin(w_last); z = K·u.
    let u = (2.0 / f64::PI()) * complex_asin(wn);
    Complex::new(big_k, 0.0) * u
}

/// Complex arcsine, asin(z) = −i·ln(i·z + sqrt(1 − z²)).
fn complex_asin(z: Complex<f64>) -> Complex<f64> {
    let i = Complex::new(0.0, 1.0);
    let one = Complex::new(1.0, 0.0);
    let root = (one - z * z).sqrt();
    -i * (i * z + root).ln()
}

/// Real inverse Jacobi sc with complementary modulus: solve w = sc(z, 1−m)
/// for real z. Uses sc(z, m) = −i·sn(i·z, 1−m) ⇒ z = Im(arc_jac_sn(i·w, m)).
fn arc_jac_sc1(w: f64, m: f64) -> f64 {
    arc_jac_sn(Complex::new(0.0, w), m).im
}

/// Normalised analog elliptic (Cauer) low-pass prototype (passband edge at
/// ω = 1), following Orfanidis / SciPy `ellipap`.
///
/// `rp` is the passband ripple in dB; `rs` the stopband attenuation in dB.
/// Returns finite imaginary-axis zeros, left-half-plane poles and a real gain.
fn elliptic_prototype(order: usize, rp: f64, rs: f64) -> AnalogZpk {
    let n = order;
    let eps_sq = (10.0_f64.powf(0.1 * rp)) - 1.0; // ε² = 10^(rp/10) − 1
    let eps = eps_sq.sqrt();
    let ck1_sq = eps_sq / (10.0_f64.powf(0.1 * rs) - 1.0); // selectivity²

    let m = ellipdeg(n, ck1_sq);
    let capk = ellipk(m);

    // j runs over 1−(N mod 2), …, N−1 stepping by 2 (the prototype indices).
    let j_start = 1 - (n % 2);
    let mut js = Vec::new();
    let mut jv = j_start as i64;
    while (jv as usize) < n {
        js.push(jv);
        jv += 2;
    }

    // Finite transmission zeros on the imaginary axis: z = j / (√m · sn).
    let mut zeros = Vec::new();
    for &jval in &js {
        let (s, _c, _d) = ellipj(jval as f64 * capk / n as f64, m);
        if s.abs() > 2e-16 {
            let z = 1.0 / (m.sqrt() * s);
            zeros.push(Complex::new(0.0, z));
        }
    }
    let conj_zeros: Vec<Complex<f64>> = zeros.iter().map(|z| z.conj()).collect();
    zeros.extend(conj_zeros);

    // Pole placement: shift the prototype by v0 along the imaginary axis.
    let r = arc_jac_sc1(1.0 / eps, ck1_sq);
    let v0 = capk * r / (n as f64 * ellipk(ck1_sq));
    let (sv, cv, dv) = ellipj(v0, 1.0 - m);

    let mut poles = Vec::new();
    for &jval in &js {
        let (s, c, d) = ellipj(jval as f64 * capk / n as f64, m);
        // p = −(c·d·sv·cv + j·s·dv) / (1 − (d·sv)²)
        let num = Complex::new(c * d * sv * cv, s * dv);
        let den = 1.0 - (d * sv) * (d * sv);
        poles.push(-num / den);
    }
    if n % 2 == 1 {
        // Odd order keeps a real pole; mirror only the genuinely complex ones.
        let new_poles: Vec<Complex<f64>> = poles
            .iter()
            .filter(|p| p.im.abs() > 1e-12)
            .map(|p| p.conj())
            .collect();
        poles.extend(new_poles);
    } else {
        let conj_poles: Vec<Complex<f64>> = poles.iter().map(|p| p.conj()).collect();
        poles.extend(conj_poles);
    }

    // Gain k = Re(Π(−p) / Π(−z)); even order is divided by √(1+ε²) so the
    // DC value lands on the −rp ripple bound.
    let prod_p = poles
        .iter()
        .fold(Complex::new(1.0, 0.0), |acc, &p| acc * (-p));
    let prod_z = if zeros.is_empty() {
        Complex::new(1.0, 0.0)
    } else {
        zeros.iter().fold(Complex::new(1.0, 0.0), |acc, &z| acc * (-z))
    };
    let mut gain = (prod_p / prod_z).re;
    if n.is_multiple_of(2) {
        gain /= (1.0 + eps_sq).sqrt();
    }

    AnalogZpk { zeros, poles, gain }
}

// ============================================================================
// IIR filter design dispatcher
// ============================================================================

/// Internal representation of a designed IIR filter.
///
/// Both variants are biquad-based (Direct Form II Transposed).  The `Single`
/// variant is used when the filter has exactly one section (order ≤ 2); the
/// `Cascade` variant is used for higher-order SOS filters.
#[derive(Debug, Clone)]
enum FilterRepresentation {
    /// A single biquad section (order ≤ 2).
    Single(Biquad),
    /// A cascade of biquad sections (SOS, order > 2).
    Cascade(Vec<Biquad>),
}

/// Design an IIR filter based on the given specifications.
///
/// Returns a [`FilterRepresentation`] which may be either direct-form
/// (for order ≤ 2) or SOS cascade (for order > 2).
fn design_iir_filter(
    design: &IirFilterDesign,
    sample_rate: f64,
) -> AudioSampleResult<FilterRepresentation> {
    match design.filter_type {
        IirFilterType::Butterworth => design_butterworth_filter(design, sample_rate),
        IirFilterType::ChebyshevI => design_chebyshev_i_filter(design, sample_rate),
        IirFilterType::ChebyshevII => design_chebyshev_ii_filter(design, sample_rate),
        IirFilterType::Bessel => design_bessel_filter(design, sample_rate),
        IirFilterType::Elliptic => design_elliptic_filter(design, sample_rate),
        // `IirFilterType` is `#[non_exhaustive]`; future variants fall through
        // here until a dedicated design path is wired up.
        #[allow(unreachable_patterns)]
        other => Err(AudioSampleError::unsupported(
            "design_iir_filter",
            format!(
                "filter type {other:?} is not yet implemented; use Butterworth, ChebyshevI, ChebyshevII, Bessel or Elliptic"
            ),
        )),
    }
}

/// Design a Butterworth filter.
fn design_butterworth_filter(
    design: &IirFilterDesign,
    sample_rate: f64,
) -> AudioSampleResult<FilterRepresentation> {
    let nyquist = sample_rate / 2.0;
    let order = design.order.get();

    // Validate order (limit to 12 for stability)
    if order > 12 {
        return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
            "order",
            "Filter order must be ≤ 12 for numerical stability",
        )));
    }

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

            // Single biquad for order 2, cascade for higher orders
            if order == 2 {
                let (b, a) = design_butterworth_lowpass(design.order, cutoff, sample_rate);
                Ok(FilterRepresentation::Single(Biquad::from_coeffs(&b, &a)))
            } else {
                let sos = design_butterworth_lowpass_sos(order, cutoff, sample_rate);
                Ok(FilterRepresentation::Cascade(
                    sos.sections
                        .iter()
                        .map(|s| Biquad::from_coeffs(&s.b_coeffs, &s.a_coeffs))
                        .collect(),
                ))
            }
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

            // Single biquad for order 2, cascade for higher orders
            if order == 2 {
                let (b, a) = design_butterworth_highpass(design.order, cutoff, sample_rate);
                Ok(FilterRepresentation::Single(Biquad::from_coeffs(&b, &a)))
            } else {
                let sos = design_butterworth_highpass_sos(order, cutoff, sample_rate);
                Ok(FilterRepresentation::Cascade(
                    sos.sections
                        .iter()
                        .map(|s| Biquad::from_coeffs(&s.b_coeffs, &s.a_coeffs))
                        .collect(),
                ))
            }
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

            let proto = butterworth_prototype(order);
            let sos = design_from_prototype(
                &proto,
                FilterResponse::BandPass,
                sample_rate,
                None,
                Some(low_freq),
                Some(high_freq),
            );
            Ok(sos_to_cascade(&sos))
        }
        FilterResponse::BandStop => {
            let low_freq = design.low_frequency.ok_or_else(|| {
                AudioSampleError::Parameter(ParameterError::invalid_value(
                    "low_frequency",
                    "Low frequency required for band-stop filter",
                ))
            })?;
            let high_freq = design.high_frequency.ok_or_else(|| {
                AudioSampleError::Parameter(ParameterError::invalid_value(
                    "high_frequency",
                    "High frequency required for band-stop filter",
                ))
            })?;

            if low_freq <= 0.0 || high_freq >= nyquist || low_freq >= high_freq {
                return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
                    "frequency_range",
                    "Invalid frequency range for band-stop filter",
                )));
            }

            let proto = butterworth_prototype(order);
            let sos = design_from_prototype(
                &proto,
                FilterResponse::BandStop,
                sample_rate,
                None,
                Some(low_freq),
                Some(high_freq),
            );
            Ok(sos_to_cascade(&sos))
        }
    }
}

impl IirFilterDesign {
    /// Design this filter as a stateful second-order-sections cascade.
    ///
    /// Convenience wrapper around [`SosFilter::from_design`]; designs the
    /// filter once for the given `sample_rate` so the returned [`SosFilter`]
    /// can stream consecutive blocks without redesigning.
    ///
    /// # Arguments
    /// - `sample_rate` – Sample rate of the signal in Hz.
    ///
    /// # Returns
    /// A freshly-reset [`SosFilter`] implementing this design.
    ///
    /// # Errors
    /// - [crate::AudioSampleError::Parameter] – if the design is invalid.
    ///
    /// # Examples
    /// ```
    /// use audio_samples::operations::types::IirFilterDesign;
    /// use std::num::NonZeroUsize;
    ///
    /// let design = IirFilterDesign::butterworth_lowpass(NonZeroUsize::new(4).unwrap(), 1000.0);
    /// let sos = design.to_sos(44100.0).unwrap();
    /// assert!(!sos.sections.is_empty());
    /// ```
    #[inline]
    pub fn to_sos(&self, sample_rate: f64) -> AudioSampleResult<SosFilter> {
        SosFilter::from_design(self, sample_rate)
    }
}

/// Convert an [`SosFilter`] into a [`FilterRepresentation::Cascade`] of fast
/// biquads.
fn sos_to_cascade(sos: &SosFilter) -> FilterRepresentation {
    FilterRepresentation::Cascade(
        sos.sections
            .iter()
            .map(|s| Biquad::from_coeffs(&s.b_coeffs, &s.a_coeffs))
            .collect(),
    )
}

/// Design a Chebyshev Type I filter.
fn design_chebyshev_i_filter(
    design: &IirFilterDesign,
    sample_rate: f64,
) -> AudioSampleResult<FilterRepresentation> {
    let nyquist = sample_rate / 2.0;
    let order = design.order.get();

    let ripple = design.passband_ripple.ok_or_else(|| {
        AudioSampleError::Parameter(ParameterError::invalid_value(
            "passband_ripple",
            "Passband ripple required for Chebyshev Type I filter",
        ))
    })?;

    // Validate ripple parameter (0.01 to 5.0 dB is typical range)
    if ripple <= 0.0 || ripple > 5.0 {
        return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
            "passband_ripple",
            "Passband ripple must be between 0.01 and 5.0 dB",
        )));
    }

    // Validate order (limit to 12 for stability)
    if order > 12 {
        return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
            "order",
            "Filter order must be ≤ 12 for numerical stability",
        )));
    }

    match design.response {
        FilterResponse::LowPass => {
            let cutoff = design.cutoff_frequency.ok_or_else(|| {
                AudioSampleError::Parameter(ParameterError::invalid_value(
                    "cutoff_frequency",
                    "Cutoff frequency required for low-pass filter",
                ))
            })?;

            if cutoff <= 0.0 || cutoff >= nyquist * 0.95 {
                return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
                    "cutoff_frequency",
                    "Cutoff frequency must be between 0 and 0.95 * Nyquist frequency",
                )));
            }

            let sos = design_chebyshev1_lowpass_sos(order, cutoff, ripple, sample_rate);
            Ok(FilterRepresentation::Cascade(
                sos.sections
                    .iter()
                    .map(|s| Biquad::from_coeffs(&s.b_coeffs, &s.a_coeffs))
                    .collect(),
            ))
        }
        FilterResponse::HighPass => {
            let cutoff = design.cutoff_frequency.ok_or_else(|| {
                AudioSampleError::Parameter(ParameterError::invalid_value(
                    "cutoff_frequency",
                    "Cutoff frequency required for high-pass filter",
                ))
            })?;

            if cutoff <= 0.0 || cutoff >= nyquist * 0.95 {
                return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
                    "cutoff_frequency",
                    "Cutoff frequency must be between 0 and 0.95 * Nyquist frequency",
                )));
            }

            let sos = design_chebyshev1_highpass_sos(order, cutoff, ripple, sample_rate);
            Ok(FilterRepresentation::Cascade(
                sos.sections
                    .iter()
                    .map(|s| Biquad::from_coeffs(&s.b_coeffs, &s.a_coeffs))
                    .collect(),
            ))
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

            if low_freq <= 0.0 || high_freq >= nyquist * 0.95 || low_freq >= high_freq {
                return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
                    "frequency_range",
                    "Invalid frequency range for band-pass filter",
                )));
            }

            let proto = chebyshev1_prototype(order, ripple);
            let sos = design_from_prototype(
                &proto,
                FilterResponse::BandPass,
                sample_rate,
                None,
                Some(low_freq),
                Some(high_freq),
            );
            Ok(sos_to_cascade(&sos))
        }
        FilterResponse::BandStop => {
            let low_freq = design.low_frequency.ok_or_else(|| {
                AudioSampleError::Parameter(ParameterError::invalid_value(
                    "low_frequency",
                    "Low frequency required for band-stop filter",
                ))
            })?;
            let high_freq = design.high_frequency.ok_or_else(|| {
                AudioSampleError::Parameter(ParameterError::invalid_value(
                    "high_frequency",
                    "High frequency required for band-stop filter",
                ))
            })?;

            if low_freq <= 0.0 || high_freq >= nyquist * 0.95 || low_freq >= high_freq {
                return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
                    "frequency_range",
                    "Invalid frequency range for band-stop filter",
                )));
            }

            let proto = chebyshev1_prototype(order, ripple);
            let sos = design_from_prototype(
                &proto,
                FilterResponse::BandStop,
                sample_rate,
                None,
                Some(low_freq),
                Some(high_freq),
            );
            Ok(sos_to_cascade(&sos))
        }
    }
}

/// Design a Chebyshev Type II (inverse Chebyshev) filter.
///
/// Uses the analog inverse-Chebyshev prototype (maximally flat passband,
/// equiripple stopband) transformed via the standard analog frequency
/// transforms and bilinear transform. The `stopband_attenuation` field
/// (dB) sets the stopband floor; for low/high-pass `cutoff_frequency` is the
/// stopband edge, for band responses `low/high_frequency` are the band edges.
fn design_chebyshev_ii_filter(
    design: &IirFilterDesign,
    sample_rate: f64,
) -> AudioSampleResult<FilterRepresentation> {
    let nyquist = sample_rate / 2.0;
    let order = design.order.get();

    let stopband_db = design.stopband_attenuation.ok_or_else(|| {
        AudioSampleError::Parameter(ParameterError::invalid_value(
            "stopband_attenuation",
            "Stopband attenuation required for Chebyshev Type II filter",
        ))
    })?;

    // Sensible stopband attenuation range.
    if stopband_db <= 0.0 || stopband_db > 200.0 {
        return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
            "stopband_attenuation",
            "Stopband attenuation must be between 0 and 200 dB",
        )));
    }

    if order > 12 {
        return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
            "order",
            "Filter order must be ≤ 12 for numerical stability",
        )));
    }

    let proto = chebyshev2_prototype(order, stopband_db);

    match design.response {
        FilterResponse::LowPass | FilterResponse::HighPass => {
            let cutoff = design.cutoff_frequency.ok_or_else(|| {
                AudioSampleError::Parameter(ParameterError::invalid_value(
                    "cutoff_frequency",
                    "Cutoff frequency required for low/high-pass filter",
                ))
            })?;
            if cutoff <= 0.0 || cutoff >= nyquist * 0.95 {
                return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
                    "cutoff_frequency",
                    "Cutoff frequency must be between 0 and 0.95 * Nyquist frequency",
                )));
            }
            let sos = design_from_prototype(
                &proto,
                design.response,
                sample_rate,
                Some(cutoff),
                None,
                None,
            );
            Ok(sos_to_cascade(&sos))
        }
        FilterResponse::BandPass | FilterResponse::BandStop => {
            let low_freq = design.low_frequency.ok_or_else(|| {
                AudioSampleError::Parameter(ParameterError::invalid_value(
                    "low_frequency",
                    "Low frequency required for band filter",
                ))
            })?;
            let high_freq = design.high_frequency.ok_or_else(|| {
                AudioSampleError::Parameter(ParameterError::invalid_value(
                    "high_frequency",
                    "High frequency required for band filter",
                ))
            })?;
            if low_freq <= 0.0 || high_freq >= nyquist * 0.95 || low_freq >= high_freq {
                return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
                    "frequency_range",
                    "Invalid frequency range for band filter",
                )));
            }
            let sos = design_from_prototype(
                &proto,
                design.response,
                sample_rate,
                None,
                Some(low_freq),
                Some(high_freq),
            );
            Ok(sos_to_cascade(&sos))
        }
    }
}

/// Design a Bessel (Bessel–Thomson) filter.
///
/// All-pole maximally-flat-group-delay prototype, normalised so the magnitude
/// is −3 dB at the cutoff. Low/high-pass use `cutoff_frequency`; band-pass /
/// band-stop use `low_frequency`/`high_frequency`.
fn design_bessel_filter(
    design: &IirFilterDesign,
    sample_rate: f64,
) -> AudioSampleResult<FilterRepresentation> {
    let nyquist = sample_rate / 2.0;
    let order = design.order.get();

    if order > 12 {
        return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
            "order",
            "Filter order must be ≤ 12 for numerical stability",
        )));
    }

    let proto = bessel_prototype(order);

    match design.response {
        FilterResponse::LowPass | FilterResponse::HighPass => {
            let cutoff = design.cutoff_frequency.ok_or_else(|| {
                AudioSampleError::Parameter(ParameterError::invalid_value(
                    "cutoff_frequency",
                    "Cutoff frequency required for low/high-pass filter",
                ))
            })?;
            if cutoff <= 0.0 || cutoff >= nyquist {
                return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
                    "cutoff_frequency",
                    "Cutoff frequency must be between 0 and Nyquist frequency",
                )));
            }
            let sos = design_from_prototype(
                &proto,
                design.response,
                sample_rate,
                Some(cutoff),
                None,
                None,
            );
            Ok(sos_to_cascade(&sos))
        }
        FilterResponse::BandPass | FilterResponse::BandStop => {
            let low_freq = design.low_frequency.ok_or_else(|| {
                AudioSampleError::Parameter(ParameterError::invalid_value(
                    "low_frequency",
                    "Low frequency required for band filter",
                ))
            })?;
            let high_freq = design.high_frequency.ok_or_else(|| {
                AudioSampleError::Parameter(ParameterError::invalid_value(
                    "high_frequency",
                    "High frequency required for band filter",
                ))
            })?;
            if low_freq <= 0.0 || high_freq >= nyquist || low_freq >= high_freq {
                return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
                    "frequency_range",
                    "Invalid frequency range for band filter",
                )));
            }
            let sos = design_from_prototype(
                &proto,
                design.response,
                sample_rate,
                None,
                Some(low_freq),
                Some(high_freq),
            );
            Ok(sos_to_cascade(&sos))
        }
    }
}

/// Design an elliptic (Cauer) filter.
///
/// Equiripple in both passband and stopband. `passband_ripple` (dB) sets the
/// passband ripple; `stopband_attenuation` (dB) the stopband floor. Low/high-
/// pass use `cutoff_frequency` (the passband edge); band responses use
/// `low_frequency`/`high_frequency`.
fn design_elliptic_filter(
    design: &IirFilterDesign,
    sample_rate: f64,
) -> AudioSampleResult<FilterRepresentation> {
    let nyquist = sample_rate / 2.0;
    let order = design.order.get();

    let rp = design.passband_ripple.ok_or_else(|| {
        AudioSampleError::Parameter(ParameterError::invalid_value(
            "passband_ripple",
            "Passband ripple required for elliptic filter",
        ))
    })?;
    let rs = design.stopband_attenuation.ok_or_else(|| {
        AudioSampleError::Parameter(ParameterError::invalid_value(
            "stopband_attenuation",
            "Stopband attenuation required for elliptic filter",
        ))
    })?;

    if rp <= 0.0 || rp > 10.0 {
        return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
            "passband_ripple",
            "Passband ripple must be between 0 and 10 dB",
        )));
    }
    if rs <= rp || rs > 200.0 {
        return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
            "stopband_attenuation",
            "Stopband attenuation must exceed passband ripple and be ≤ 200 dB",
        )));
    }
    if order > 12 {
        return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
            "order",
            "Filter order must be ≤ 12 for numerical stability",
        )));
    }

    let proto = elliptic_prototype(order, rp, rs);

    match design.response {
        FilterResponse::LowPass | FilterResponse::HighPass => {
            let cutoff = design.cutoff_frequency.ok_or_else(|| {
                AudioSampleError::Parameter(ParameterError::invalid_value(
                    "cutoff_frequency",
                    "Cutoff frequency required for low/high-pass filter",
                ))
            })?;
            if cutoff <= 0.0 || cutoff >= nyquist * 0.95 {
                return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
                    "cutoff_frequency",
                    "Cutoff frequency must be between 0 and 0.95 * Nyquist frequency",
                )));
            }
            let sos = design_from_prototype_with_gain(
                &proto,
                design.response,
                sample_rate,
                Some(cutoff),
                None,
                None,
                Some(GainRef::Prototype),
            );
            Ok(sos_to_cascade(&sos))
        }
        FilterResponse::BandPass | FilterResponse::BandStop => {
            let low_freq = design.low_frequency.ok_or_else(|| {
                AudioSampleError::Parameter(ParameterError::invalid_value(
                    "low_frequency",
                    "Low frequency required for band filter",
                ))
            })?;
            let high_freq = design.high_frequency.ok_or_else(|| {
                AudioSampleError::Parameter(ParameterError::invalid_value(
                    "high_frequency",
                    "High frequency required for band filter",
                ))
            })?;
            if low_freq <= 0.0 || high_freq >= nyquist * 0.95 || low_freq >= high_freq {
                return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
                    "frequency_range",
                    "Invalid frequency range for band filter",
                )));
            }
            let sos = design_from_prototype_with_gain(
                &proto,
                design.response,
                sample_rate,
                None,
                Some(low_freq),
                Some(high_freq),
                Some(GainRef::Prototype),
            );
            Ok(sos_to_cascade(&sos))
        }
    }
}

/// Design a 2nd-order Butterworth low-pass filter using bilinear transform.
///
/// This is a direct-form implementation for order-2 filters only.
/// For higher orders, use [`design_butterworth_lowpass_sos`] instead.
///
/// # Panics
/// Panics if order ≠ 2 (caller should use SOS for other orders).
fn design_butterworth_lowpass(
    order: NonZeroUsize,
    cutoff_freq: f64,
    sample_rate: f64,
) -> (Vec<f64>, Vec<f64>) {
    let order = order.get();
    debug_assert_eq!(
        order, 2,
        "design_butterworth_lowpass only supports order 2; use SOS for higher orders"
    );

    // Pre-warp the cutoff frequency for bilinear transform
    let wc = 2.0 * sample_rate * (f64::PI() * cutoff_freq / sample_rate).tan();
    let k = wc / (2.0 * sample_rate);
    let k2 = k * k;
    let sqrt2 = 2.0f64.sqrt();
    let norm = sqrt2.mul_add(k, 1.0) + k2;

    // Numerator coefficients (zeros at z = -1)
    let b_coeffs = vec![k2 / norm, 2.0 * k2 / norm, k2 / norm];

    // Denominator coefficients (normalized Butterworth poles)
    let a_coeffs = vec![
        1.0,
        2.0f64.mul_add(k2, -2.0) / norm,
        (sqrt2.mul_add(-k, 1.0) + k2) / norm,
    ];

    (b_coeffs, a_coeffs)
}

/// Design a 2nd-order Butterworth high-pass filter using bilinear transform.
///
/// This is a direct-form implementation for order-2 filters only.
/// For higher orders, use [`design_butterworth_highpass_sos`] instead.
///
/// # Panics
/// Panics if order ≠ 2 (caller should use SOS for other orders).
fn design_butterworth_highpass(
    order: NonZeroUsize,
    cutoff_freq: f64,
    sample_rate: f64,
) -> (Vec<f64>, Vec<f64>) {
    let order = order.get();
    debug_assert_eq!(
        order, 2,
        "design_butterworth_highpass only supports order 2; use SOS for higher orders"
    );

    // Pre-warp the cutoff frequency for bilinear transform
    let wc = 2.0 * sample_rate * (f64::PI() * cutoff_freq / sample_rate).tan();
    let k = wc / (2.0 * sample_rate);
    let k2 = k * k;
    let sqrt2 = 2.0f64.sqrt();
    let norm = sqrt2.mul_add(k, 1.0) + k2;

    // Numerator coefficients (zeros at z = 1, DC)
    let b_coeffs = vec![1.0 / norm, -2.0 / norm, 1.0 / norm];

    // Denominator coefficients (same poles as lowpass)
    let a_coeffs = vec![
        1.0,
        2.0f64.mul_add(k2, -2.0) / norm,
        (sqrt2.mul_add(-k, 1.0) + k2) / norm,
    ];

    (b_coeffs, a_coeffs)
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
        let result = audio.butterworth_lowpass_in_place(NonZeroUsize::new(2).unwrap(), 1000.0);
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
        let result = audio.butterworth_highpass_in_place(NonZeroUsize::new(2).unwrap(), 500.0);
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
        let result = audio.butterworth_bandpass_in_place(NonZeroUsize::new(2).unwrap(), 500.0, 2000.0);
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
            AudioSamples::new_multi_channel(stereo_data, sample_rate!(44100)).unwrap();

        // Apply low-pass filter to stereo signal
        let result = audio.butterworth_lowpass_in_place(NonZeroUsize::new(2).unwrap(), 2000.0);
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
                .butterworth_lowpass_in_place(NonZeroUsize::new(2).unwrap(), 0.0)
                .is_err()
        );
        assert!(
            audio
                .butterworth_lowpass_in_place(NonZeroUsize::new(2).unwrap(), sample_rate / 2.0)
                .is_err()
        );
        assert!(
            audio
                .butterworth_lowpass_in_place(NonZeroUsize::new(2).unwrap(), -100.0)
                .is_err()
        );

        // Test invalid band-pass frequencies
        assert!(
            audio
                .butterworth_bandpass_in_place(NonZeroUsize::new(2).unwrap(), 2000.0, 1000.0)
                .is_err()
        );
        assert!(
            audio
                .butterworth_bandpass_in_place(NonZeroUsize::new(2).unwrap(), 0.0, 1000.0)
                .is_err()
        );
        assert!(
            audio
                .butterworth_bandpass_in_place(NonZeroUsize::new(2).unwrap(), 1000.0, sample_rate / 2.0)
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

    // ========================================================================
    // SOS and high-order filter tests
    // ========================================================================

    #[test]
    fn test_butterworth_poles_order_6() {
        // Test that 6th-order Butterworth generates 6 poles on a circle
        let order = 6;
        let omega_c = 1000.0; // Arbitrary cutoff

        let poles = super::butterworth_analog_poles(order, omega_c);

        assert_eq!(poles.len(), 6);

        // All poles should have magnitude ≈ omega_c (on circle)
        for pole in &poles {
            let magnitude = pole.norm();
            assert!(
                (magnitude - omega_c).abs() < 1e-6,
                "Pole magnitude should be ≈ {}, got {}",
                omega_c,
                magnitude
            );
        }

        // All poles should be in left half-plane (negative real part)
        for pole in &poles {
            assert!(pole.re < 0.0, "Pole should be in left half-plane");
        }
    }

    #[test]
    fn test_bilinear_transform() {
        // Test that bilinear transform maps left-half s-plane to inside unit circle
        let sample_rate = 44100.0;
        let s_pole = Complex::new(-1000.0, 2000.0); // Stable analog pole

        let z_pole = super::bilinear_transform_pole(s_pole, sample_rate);

        // Digital pole should be inside unit circle (magnitude < 1)
        assert!(
            z_pole.norm() < 1.0,
            "Digital pole should be inside unit circle, got magnitude {}",
            z_pole.norm()
        );
    }

    #[test]
    fn test_pole_pairing() {
        // Test conjugate pole pairing
        let p1 = Complex::new(-0.5, 0.3);
        let p2 = Complex::new(-0.5, -0.3); // Conjugate of p1
        let p3 = Complex::new(-0.7, 0.0); // Real pole

        let poles = vec![p1, p2, p3];
        let pairs = super::pair_poles(poles);

        assert_eq!(pairs.len(), 2, "Should have 2 pairs (1 conjugate + 1 real)");

        // Find the conjugate pair
        let conjugate_pair = pairs.iter().find(|(a, b)| (a.conj() - *b).norm() < 1e-10);
        assert!(conjugate_pair.is_some(), "Should find conjugate pair");

        // Find the real pair (paired with itself)
        let real_pair = pairs
            .iter()
            .find(|(a, b)| a.im.abs() < 1e-10 && (*a - *b).norm() < 1e-10);
        assert!(
            real_pair.is_some(),
            "Should find real pole paired with itself"
        );
    }

    #[test]
    fn test_biquad_coefficients() {
        // Test biquad construction from simple poles/zeros
        let p1 = Complex::new(0.5, 0.0);
        let p2 = Complex::new(0.5, 0.0);
        let z1 = Complex::new(-1.0, 0.0);
        let z2 = Complex::new(-1.0, 0.0);
        let gain = 1.0;

        let (b, a) = super::biquad_from_poles_zeros(p1, p2, z1, z2, gain);

        // Should have 3 coefficients each
        assert_eq!(b.len(), 3);
        assert_eq!(a.len(), 3);

        // a[0] should be 1.0 (normalized)
        assert_eq!(a[0], 1.0);
    }

    #[test]
    fn test_sos_cascade() {
        // Test SOS filter cascade processing
        use super::SosFilter;

        let section1 = IirFilter::new(vec![0.5, 0.5], vec![1.0]);
        let section2 = IirFilter::new(vec![0.3, 0.7], vec![1.0]);

        let mut sos = SosFilter::new(vec![section1, section2]);

        // Process a simple signal
        let input = vec![1.0, 0.0, 0.0];
        let output = sos.process_samples(&input);

        assert_eq!(output.len(), 3);
        // Output should be non-zero after cascading
        assert!(output[0].abs() > 0.0);
    }

    #[test]
    fn test_sos_reset() {
        use super::SosFilter;

        let section1 = IirFilter::new(vec![1.0, 0.5], vec![1.0, 0.2]);
        let section2 = IirFilter::new(vec![1.0, 0.3], vec![1.0, 0.1]);

        let mut sos = SosFilter::new(vec![section1, section2]);

        // Process a sample
        sos.process_sample(1.0);

        // Delays should be non-zero
        assert_ne!(sos.sections[0].x_delays[0], 0.0);

        // Reset
        sos.reset();

        // All delays should be zero
        for section in &sos.sections {
            for &delay in &section.x_delays {
                assert_eq!(delay, 0.0);
            }
            for &delay in &section.y_delays {
                assert_eq!(delay, 0.0);
            }
        }
    }

    #[test]
    fn test_butterworth_order_6() {
        // Test 6th-order Butterworth lowpass filter
        let sample_rate = 44100.0;
        let duration = 0.1;
        let samples_count = (sample_rate * duration) as usize;

        let mut samples = Vec::new();
        for i in 0..samples_count {
            let t = i as f64 / sample_rate;
            // Low frequency (500 Hz) + high frequency (8000 Hz)
            let value = (2.0 * PI * 500.0 * t).sin() + 0.5 * (2.0 * PI * 8000.0 * t).sin();
            samples.push(value as f32);
        }
        let samples = NonEmptyVec::new(samples).unwrap();
        let mut audio: AudioSamples<'_, f32> =
            AudioSamples::from_mono_vec(samples, sample_rate!(44100));

        // Apply 6th-order Butterworth lowpass at 2000 Hz
        let result = audio.butterworth_lowpass_in_place(NonZeroUsize::new(6).unwrap(), 2000.0);
        assert!(result.is_ok(), "6th-order Butterworth should work");
    }

    #[test]
    fn test_butterworth_order_8() {
        // Test 8th-order Butterworth highpass filter
        let sample_rate = 44100.0;
        let duration = 0.1;
        let samples_count = (sample_rate * duration) as usize;

        let mut samples = Vec::new();
        for i in 0..samples_count {
            let t = i as f64 / sample_rate;
            let value = (2.0 * PI * 100.0 * t).sin() + 0.5 * (2.0 * PI * 5000.0 * t).sin();
            samples.push(value as f32);
        }
        let samples = NonEmptyVec::new(samples).unwrap();
        let mut audio: AudioSamples<'_, f32> =
            AudioSamples::from_mono_vec(samples, sample_rate!(44100));

        // Apply 8th-order Butterworth highpass at 1000 Hz
        let result = audio.butterworth_highpass_in_place(NonZeroUsize::new(8).unwrap(), 1000.0);
        assert!(result.is_ok(), "8th-order Butterworth should work");
    }

    #[test]
    fn test_butterworth_order_10() {
        // Test 10th-order Butterworth bandpass filter
        let sample_rate = 44100.0;
        let duration = 0.05;
        let samples_count = (sample_rate * duration) as usize;

        let mut samples = vec![0.0f32; samples_count];
        for (i, s) in samples.iter_mut().enumerate() {
            let t = i as f64 / sample_rate;
            *s = (2.0 * PI * 1000.0 * t).sin() as f32;
        }
        let samples = NonEmptyVec::new(samples).unwrap();
        let mut audio: AudioSamples<'_, f32> =
            AudioSamples::from_mono_vec(samples, sample_rate!(44100));

        // Apply 10th-order Butterworth bandpass
        let result = audio.butterworth_bandpass_in_place(NonZeroUsize::new(10).unwrap(), 800.0, 1200.0);
        assert!(
            result.is_ok(),
            "10th-order Butterworth bandpass should work"
        );
    }

    #[test]
    fn test_chebyshev_basic() {
        // Test that Chebyshev Type I no longer errors
        let samples = NonEmptyVec::new(vec![1.0f32, 0.0, -1.0, 0.0, 1.0, 0.0, -1.0, 0.0]).unwrap();
        let mut audio: AudioSamples<'_, f32> =
            AudioSamples::from_mono_vec(samples, sample_rate!(44100));

        // This should now work (not error)
        let result = audio.chebyshev_i_in_place(
            NonZeroUsize::new(4).unwrap(),
            1000.0,
            0.5,
            FilterResponse::LowPass,
        );
        assert!(result.is_ok(), "Chebyshev Type I should be implemented");
    }

    #[test]
    fn test_chebyshev_ripple_validation() {
        let samples = NonEmptyVec::new(vec![1.0f32; 100]).unwrap();
        let mut audio: AudioSamples<'_, f32> =
            AudioSamples::from_mono_vec(samples, sample_rate!(44100));

        // Test invalid ripple (too large)
        let result = audio.chebyshev_i_in_place(
            NonZeroUsize::new(4).unwrap(),
            1000.0,
            10.0, // Invalid: > 5.0 dB
            FilterResponse::LowPass,
        );
        assert!(result.is_err(), "Should reject ripple > 5.0 dB");

        // Test invalid ripple (zero or negative)
        let samples2 = NonEmptyVec::new(vec![1.0f32; 100]).unwrap();
        let mut audio2: AudioSamples<'_, f32> =
            AudioSamples::from_mono_vec(samples2, sample_rate!(44100));
        let result2 = audio2.chebyshev_i_in_place(
            NonZeroUsize::new(4).unwrap(),
            1000.0,
            0.0,
            FilterResponse::LowPass,
        );
        assert!(result2.is_err(), "Should reject ripple ≤ 0");
    }

    #[test]
    fn test_frequency_response_dc_gain() {
        // Test that lowpass filter design has DC gain ≈ 1.0
        let sample_rate = 44100.0;
        let order = 6;
        let cutoff = 1000.0;

        // Design a 6th-order Butterworth lowpass filter
        let sos = super::design_butterworth_lowpass_sos(order, cutoff, sample_rate);

        // Check frequency response at DC (0 Hz)
        let (mag, _phase) = sos.frequency_response(&[0.0], sample_rate);

        // DC gain should be normalized to 1.0
        assert!(
            (mag[0] - 1.0).abs() < 0.01,
            "DC gain should be ≈ 1.0, got {}",
            mag[0]
        );
    }

    #[test]
    fn test_sos_frequency_response() {
        use super::SosFilter;

        // Create a simple SOS filter and check frequency response
        let section = IirFilter::new(vec![1.0, 0.0, 0.0], vec![1.0, 0.0, 0.0]);
        let sos = SosFilter::new(vec![section]);

        let (mag, phase) = sos.frequency_response(&[100.0, 1000.0, 10000.0], 44100.0);

        assert_eq!(mag.len(), 3);
        assert_eq!(phase.len(), 3);

        // Identity filter should have magnitude 1.0 everywhere
        for &m in &mag {
            assert!(
                (m - 1.0).abs() < 1e-10,
                "Identity filter magnitude should be 1.0"
            );
        }
    }

    #[test]
    fn test_multichannel_sos() {
        // Test that SOS filters work correctly with multi-channel audio
        let sample_rate = 44100.0;
        let duration = 0.05;
        let samples_count = (sample_rate * duration) as usize;

        let mut left_samples = Vec::new();
        let mut right_samples = Vec::new();

        for i in 0..samples_count {
            let t = i as f64 / sample_rate;
            left_samples.push((2.0 * PI * 440.0 * t).sin() as f32);
            right_samples.push((2.0 * PI * 880.0 * t).sin() as f32);
        }

        let stereo_data = ndarray::Array2::from_shape_vec(
            (2, samples_count),
            [left_samples, right_samples].concat(),
        )
        .unwrap();

        let mut audio =
            AudioSamples::new_multi_channel(stereo_data, sample_rate!(44100)).unwrap();

        // Apply 6th-order filter to stereo
        let result = audio.butterworth_lowpass_in_place(NonZeroUsize::new(6).unwrap(), 2000.0);
        assert!(result.is_ok(), "SOS should work with multi-channel audio");
    }

    #[test]
    fn test_order_limit() {
        // Test that order > 12 is rejected
        let samples = NonEmptyVec::new(vec![1.0f32; 100]).unwrap();
        let mut audio: AudioSamples<'_, f32> =
            AudioSamples::from_mono_vec(samples, sample_rate!(44100));

        let result = audio.butterworth_lowpass_in_place(NonZeroUsize::new(14).unwrap(), 1000.0);
        assert!(result.is_err(), "Should reject order > 12");
    }

    #[test]
    fn test_chebyshev_highpass() {
        // Test Chebyshev Type I highpass
        let samples = NonEmptyVec::new(vec![1.0f32; 200]).unwrap();
        let mut audio: AudioSamples<'_, f32> =
            AudioSamples::from_mono_vec(samples, sample_rate!(44100));

        let result = audio.chebyshev_i_in_place(
            NonZeroUsize::new(4).unwrap(),
            2000.0,
            0.5,
            FilterResponse::HighPass,
        );
        assert!(result.is_ok(), "Chebyshev highpass should work");
    }

    #[test]
    fn test_chebyshev_bandpass() {
        // Test Chebyshev Type I bandpass
        let samples = NonEmptyVec::new(vec![1.0f32; 200]).unwrap();
        let mut audio: AudioSamples<'_, f32> =
            AudioSamples::from_mono_vec(samples, sample_rate!(44100));

        let result = audio.chebyshev_i_in_place(
            NonZeroUsize::new(4).unwrap(),
            1000.0,
            1.0,
            FilterResponse::BandPass,
        );
        // Note: This uses the design.cutoff_frequency field, but bandpass needs low/high
        // The current API design has a limitation here - we'd need to use IirFilterDesign::chebyshev_i_bandpass
        // For now, this will error because cutoff_frequency is used instead of low/high_frequency
        // This is a known API limitation, not a bug in our implementation
        assert!(
            result.is_err(),
            "Current API doesn't support Chebyshev bandpass via chebyshev_i method"
        );
    }

    #[test]
    fn test_butterworth_lowpass_dual_variant() {
        // The non-mutating variant must leave the original untouched and
        // produce the same result as clone + in-place.
        let samples = NonEmptyVec::new(vec![1.0f32, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0]).unwrap();
        let original: AudioSamples<'_, f32> =
            AudioSamples::from_mono_vec(samples, sample_rate!(44100));

        let order = NonZeroUsize::new(2).unwrap();

        // Non-mutating: returns a new copy, leaves `original` unchanged.
        let filtered = original.butterworth_lowpass(order, 1000.0).unwrap();

        // In-place on a clone.
        let mut in_place = original.clone();
        in_place.butterworth_lowpass_in_place(order, 1000.0).unwrap();

        assert_eq!(
            filtered.as_slice().unwrap(),
            in_place.as_slice().unwrap(),
            "non-mutating and in-place variants must produce equal results"
        );

        // The original must be untouched by the non-mutating call.
        let pristine: AudioSamples<'_, f32> = AudioSamples::from_mono_vec(
            NonEmptyVec::new(vec![1.0f32, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0]).unwrap(),
            sample_rate!(44100),
        );
        assert_eq!(
            original.as_slice().unwrap(),
            pristine.as_slice().unwrap(),
            "non-mutating variant must not modify the original"
        );
    }

    // ========================================================================
    // Validated frequency-response tests for the new designs
    //
    // Reference dB values cross-checked against SciPy:
    //   signal.cheby2 / signal.butter / signal.cheby1 with output='sos',
    //   evaluated via signal.sosfreqz at fs = 44100. Concrete checkpoints are
    //   noted inline next to each assertion.
    // ========================================================================

    const FS: f64 = 44100.0;

    fn db(mag: f64) -> f64 {
        20.0 * mag.max(1e-12).log10()
    }

    /// Convenience: magnitude (dB) of an SOS at a single frequency.
    fn mag_db(sos: &SosFilter, freq: f64) -> f64 {
        let (m, _) = sos.frequency_response(&[freq], FS);
        db(m[0])
    }

    // ---- Chebyshev Type II low-pass ----------------------------------------

    #[test]
    fn test_cheby2_lowpass_response() {
        // order 4, stopband edge 2000 Hz, As = 40 dB.
        // SciPy reference: 0 Hz ~ 0 dB, 2000 Hz = -40 dB, >=2000 Hz <= -40 dB,
        // 1000 Hz ~ -3.04 dB.
        let proto = super::chebyshev2_prototype(4, 40.0);
        let sos = super::design_from_prototype(
            &proto,
            FilterResponse::LowPass,
            FS,
            Some(2000.0),
            None,
            None,
        );

        // Maximally flat passband: ~0 dB within 1 dB at DC and well below edge.
        assert!(mag_db(&sos, 0.0).abs() < 1.0, "DC: {}", mag_db(&sos, 0.0));
        assert!(
            mag_db(&sos, 100.0).abs() < 1.0,
            "100Hz: {}",
            mag_db(&sos, 100.0)
        );
        assert!(
            mag_db(&sos, 500.0).abs() < 1.0,
            "500Hz: {}",
            mag_db(&sos, 500.0)
        );

        // Stopband: at and beyond the stopband edge attenuation >= 40 dB
        // (equiripple touches exactly -40, allow tiny numerical slack).
        assert!(
            mag_db(&sos, 2000.0) <= -40.0 + 0.5,
            "2000Hz stopband: {}",
            mag_db(&sos, 2000.0)
        );
        assert!(
            mag_db(&sos, 2500.0) <= -40.0 + 0.5,
            "2500Hz stopband: {}",
            mag_db(&sos, 2500.0)
        );
        assert!(
            mag_db(&sos, 4000.0) <= -40.0 + 0.5,
            "4000Hz stopband: {}",
            mag_db(&sos, 4000.0)
        );

        // Transition: 1000 Hz ~ -3 dB (SciPy: -3.04). Within 1 dB.
        assert!(
            (mag_db(&sos, 1000.0) - (-3.04)).abs() < 1.0,
            "1000Hz transition: {}",
            mag_db(&sos, 1000.0)
        );
    }

    #[test]
    fn test_cheby2_lowpass_odd_order() {
        // Odd order drops one zero to infinity. order 3, edge 2000, As 40.
        // SciPy reference: DC ~0 dB, 2000 Hz = -40 dB, 4000 Hz ~ -40 dB.
        let proto = super::chebyshev2_prototype(3, 40.0);
        let sos = super::design_from_prototype(
            &proto,
            FilterResponse::LowPass,
            FS,
            Some(2000.0),
            None,
            None,
        );
        assert!(mag_db(&sos, 0.0).abs() < 1.0, "DC: {}", mag_db(&sos, 0.0));
        assert!(
            mag_db(&sos, 2000.0) <= -40.0 + 0.5,
            "2000Hz: {}",
            mag_db(&sos, 2000.0)
        );
        assert!(
            mag_db(&sos, 4000.0) <= -40.0 + 0.5,
            "4000Hz: {}",
            mag_db(&sos, 4000.0)
        );
    }

    // ---- Chebyshev Type II high-pass ---------------------------------------

    #[test]
    fn test_cheby2_highpass_response() {
        // order 4, stopband edge 2000 Hz, As = 40 dB.
        // SciPy reference: 2000 Hz = -40 dB, 100 Hz = -40.2 dB (stopband),
        // 10000 Hz ~ 0 dB.
        let proto = super::chebyshev2_prototype(4, 40.0);
        let sos = super::design_from_prototype(
            &proto,
            FilterResponse::HighPass,
            FS,
            Some(2000.0),
            None,
            None,
        );

        // Passband (well above edge): flat ~0 dB.
        assert!(
            mag_db(&sos, 10000.0).abs() < 1.0,
            "10kHz: {}",
            mag_db(&sos, 10000.0)
        );
        assert!(
            mag_db(&sos, 5000.0).abs() < 1.0,
            "5kHz: {}",
            mag_db(&sos, 5000.0)
        );

        // Stopband: at and below edge attenuation >= 40 dB.
        assert!(
            mag_db(&sos, 2000.0) <= -40.0 + 0.5,
            "2000Hz: {}",
            mag_db(&sos, 2000.0)
        );
        assert!(
            mag_db(&sos, 1000.0) <= -40.0 + 0.5,
            "1000Hz: {}",
            mag_db(&sos, 1000.0)
        );
        assert!(
            mag_db(&sos, 100.0) <= -40.0 + 0.5,
            "100Hz: {}",
            mag_db(&sos, 100.0)
        );
    }

    // ---- Butterworth band-pass (proper analog LP->BP transform) ------------

    #[test]
    fn test_butterworth_bandpass_response() {
        // order 4, band [500, 2000] Hz. Geometric centre = 1000 Hz.
        // SciPy reference: centre 0 dB, edges -3.01 dB, 100 Hz -65 dB,
        // 8000 Hz -61 dB.
        let proto = super::butterworth_prototype(4);
        let sos = super::design_from_prototype(
            &proto,
            FilterResponse::BandPass,
            FS,
            None,
            Some(500.0),
            Some(2000.0),
        );
        let center = (500.0_f64 * 2000.0).sqrt(); // 1000 Hz

        // ~0 dB at the geometric centre.
        assert!(
            mag_db(&sos, center).abs() < 0.5,
            "center {center}Hz: {}",
            mag_db(&sos, center)
        );

        // Edges ~ -3 dB.
        assert!(
            (mag_db(&sos, 500.0) - (-3.01)).abs() < 1.0,
            "500Hz edge: {}",
            mag_db(&sos, 500.0)
        );
        assert!(
            (mag_db(&sos, 2000.0) - (-3.01)).abs() < 1.0,
            "2000Hz edge: {}",
            mag_db(&sos, 2000.0)
        );

        // Strong attenuation outside the band.
        assert!(
            mag_db(&sos, 100.0) <= -20.0,
            "100Hz: {}",
            mag_db(&sos, 100.0)
        );
        assert!(
            mag_db(&sos, 8000.0) <= -20.0,
            "8000Hz: {}",
            mag_db(&sos, 8000.0)
        );
    }

    // ---- Butterworth band-stop (analog LP->BS transform) -------------------

    #[test]
    fn test_butterworth_bandstop_response() {
        // order 4, band [500, 2000] Hz, notch centred at 1000 Hz.
        // SciPy reference: 1000 Hz extremely deep (< -200 dB), edges -3 dB,
        // DC and 15 kHz ~ 0 dB.
        let proto = super::butterworth_prototype(4);
        let sos = super::design_from_prototype(
            &proto,
            FilterResponse::BandStop,
            FS,
            None,
            Some(500.0),
            Some(2000.0),
        );
        let center = (500.0_f64 * 2000.0).sqrt();

        // Deep notch at centre.
        assert!(
            mag_db(&sos, center) <= -40.0,
            "notch center: {}",
            mag_db(&sos, center)
        );

        // ~0 dB well outside the notch.
        assert!(mag_db(&sos, 50.0).abs() < 1.0, "50Hz: {}", mag_db(&sos, 50.0));
        assert!(
            mag_db(&sos, 100.0).abs() < 1.0,
            "100Hz: {}",
            mag_db(&sos, 100.0)
        );
        assert!(
            mag_db(&sos, 15000.0).abs() < 1.0,
            "15kHz: {}",
            mag_db(&sos, 15000.0)
        );

        // Edges ~ -3 dB.
        assert!(
            (mag_db(&sos, 500.0) - (-3.01)).abs() < 1.0,
            "500Hz edge: {}",
            mag_db(&sos, 500.0)
        );
    }

    // ---- Chebyshev I band-pass (proper analog LP->BP transform) ------------

    #[test]
    fn test_cheby1_bandpass_response() {
        // order 4, band [500, 2000] Hz, rp = 0.5 dB. Centre 1000 Hz.
        // SciPy reference: passband sits within [-0.5, 0] dB across [500,2000],
        // 100 Hz -74 dB, 5000 Hz -49 dB.
        let proto = super::chebyshev1_prototype(4, 0.5);
        let sos = super::design_from_prototype(
            &proto,
            FilterResponse::BandPass,
            FS,
            None,
            Some(500.0),
            Some(2000.0),
        );
        let center = (500.0_f64 * 2000.0).sqrt();

        // Within the passband, gain stays within the ripple bound of 0 dB.
        for &f in &[500.0, center, 1000.0, 2000.0] {
            let g = mag_db(&sos, f);
            assert!(
                (-0.5 - 0.2..=0.1).contains(&g),
                "passband {f}Hz gain {g} out of ripple bound"
            );
        }

        // Strong rejection outside.
        assert!(
            mag_db(&sos, 100.0) <= -30.0,
            "100Hz: {}",
            mag_db(&sos, 100.0)
        );
        assert!(
            mag_db(&sos, 5000.0) <= -30.0,
            "5000Hz: {}",
            mag_db(&sos, 5000.0)
        );
    }

    // ---- Chebyshev I band-stop ---------------------------------------------

    #[test]
    fn test_cheby1_bandstop_response() {
        // order 4, band [500, 2000] Hz, rp = 0.5 dB. Notch at 1000 Hz.
        let proto = super::chebyshev1_prototype(4, 0.5);
        let sos = super::design_from_prototype(
            &proto,
            FilterResponse::BandStop,
            FS,
            None,
            Some(500.0),
            Some(2000.0),
        );
        let center = (500.0_f64 * 2000.0).sqrt();

        assert!(
            mag_db(&sos, center) <= -40.0,
            "notch center: {}",
            mag_db(&sos, center)
        );
        // Passband on both sides within the ripple bound.
        assert!(mag_db(&sos, 50.0) >= -0.7, "50Hz: {}", mag_db(&sos, 50.0));
        assert!(
            mag_db(&sos, 15000.0) >= -0.7,
            "15kHz: {}",
            mag_db(&sos, 15000.0)
        );
    }

    // ---- Chebyshev II band-stop --------------------------------------------

    #[test]
    fn test_cheby2_bandstop_response() {
        // order 4, band [500, 2000] Hz, As = 40 dB. Notch at 1000 Hz.
        // SciPy reference: across [500,2000] attenuation = -40 dB, DC and
        // 15 kHz ~ 0 dB.
        let proto = super::chebyshev2_prototype(4, 40.0);
        let sos = super::design_from_prototype(
            &proto,
            FilterResponse::BandStop,
            FS,
            None,
            Some(500.0),
            Some(2000.0),
        );
        let center = (500.0_f64 * 2000.0).sqrt();

        // Stopband floor across the band >= 40 dB attenuation.
        assert!(
            mag_db(&sos, center) <= -40.0 + 0.5,
            "center: {}",
            mag_db(&sos, center)
        );
        assert!(
            mag_db(&sos, 500.0) <= -40.0 + 0.5,
            "500Hz: {}",
            mag_db(&sos, 500.0)
        );
        assert!(
            mag_db(&sos, 2000.0) <= -40.0 + 0.5,
            "2000Hz: {}",
            mag_db(&sos, 2000.0)
        );

        // Maximally flat passband ~0 dB outside the band.
        assert!(mag_db(&sos, 50.0).abs() < 1.0, "50Hz: {}", mag_db(&sos, 50.0));
        assert!(
            mag_db(&sos, 15000.0).abs() < 1.0,
            "15kHz: {}",
            mag_db(&sos, 15000.0)
        );
    }

    // ---- End-to-end dispatcher wiring (apply_iir_filter) -------------------

    #[test]
    fn test_cheby2_dispatcher_wired() {
        // Chebyshev II lowpass via the public apply_iir_filter path.
        let samples = NonEmptyVec::new(vec![1.0f32; 256]).unwrap();
        let audio: AudioSamples<'_, f32> =
            AudioSamples::from_mono_vec(samples, sample_rate!(44100));
        let design = IirFilterDesign::chebyshev_ii(
            FilterResponse::LowPass,
            NonZeroUsize::new(4).unwrap(),
            2000.0,
            40.0,
        );
        assert!(audio.apply_iir_filter(&design).is_ok());

        // And in-place band-stop.
        let samples2 = NonEmptyVec::new(vec![1.0f32; 256]).unwrap();
        let mut audio2: AudioSamples<'_, f32> =
            AudioSamples::from_mono_vec(samples2, sample_rate!(44100));
        let bs = IirFilterDesign::chebyshev_ii_band(
            FilterResponse::BandStop,
            NonZeroUsize::new(4).unwrap(),
            500.0,
            2000.0,
            40.0,
        );
        assert!(audio2.apply_iir_filter_in_place(&bs).is_ok());
    }

    #[test]
    fn test_butterworth_bandstop_dispatcher_wired() {
        // Butterworth band-stop now supported (was previously unsupported).
        let samples = NonEmptyVec::new(vec![1.0f32; 256]).unwrap();
        let mut audio: AudioSamples<'_, f32> =
            AudioSamples::from_mono_vec(samples, sample_rate!(44100));
        let mut bs = IirFilterDesign::butterworth_bandpass(
            NonZeroUsize::new(4).unwrap(),
            500.0,
            2000.0,
        );
        bs.response = FilterResponse::BandStop;
        assert!(audio.apply_iir_filter_in_place(&bs).is_ok());
    }

    // ========================================================================
    // Bessel filter tests
    //
    // Reference dB values cross-checked against
    //   scipy.signal.bessel(N, Wn, btype, output='sos', fs=44100, norm='mag')
    // evaluated with signal.sosfreqz. Checkpoints noted inline.
    // ========================================================================

    /// The analog Bessel prototype poles must match scipy's `norm='mag'`
    /// prototype (−3 dB at ω = 1) to high precision for orders 2,4,6,8.
    #[test]
    fn test_bessel_prototype_poles_match_scipy() {
        // scipy.signal.bessel(N, 1, analog=True, output='zpk', norm='mag')
        // poles, sorted by (imag, real).
        let order4 = [
            Complex::new(-0.995208764350272, -1.257105739454664),
            Complex::new(-1.370067830551442, -0.410249717493752),
            Complex::new(-1.370067830551442, 0.410249717493752),
            Complex::new(-0.995208764350272, 1.257105739454664),
        ];
        let proto = super::bessel_prototype(4);
        let mut got = proto.poles.clone();
        got.sort_by(|a, b| {
            a.im.partial_cmp(&b.im)
                .unwrap()
                .then(a.re.partial_cmp(&b.re).unwrap())
        });
        assert_eq!(got.len(), 4);
        for (g, e) in got.iter().zip(order4.iter()) {
            assert!(
                (g - e).norm() < 1e-9,
                "Bessel N=4 pole mismatch: got {g}, expected {e}"
            );
        }
        assert!(proto.zeros.is_empty(), "Bessel prototype is all-pole");

        // Order 2 prototype magnitude must be exactly −3 dB at ω = 1.
        let proto2 = super::bessel_prototype(2);
        let mag1 = super::allpole_mag(&proto2.poles, proto2.gain, 1.0);
        assert!(
            (db(mag1) - (-3.0103)).abs() < 1e-3,
            "Bessel −3 dB normalization off: {} dB",
            db(mag1)
        );
    }

    #[test]
    fn test_bessel_lowpass_response() {
        // order 4, cutoff 1000 Hz. SciPy: 0 Hz 0 dB, 100 −0.028, 500 −0.703,
        // 1000 −3.01, 2000 −13.53, 4000 −35.30. Monotonic, no ripple.
        let proto = super::bessel_prototype(4);
        let sos = super::design_from_prototype(
            &proto,
            FilterResponse::LowPass,
            FS,
            Some(1000.0),
            None,
            None,
        );
        assert!(mag_db(&sos, 0.0).abs() < 0.01, "DC: {}", mag_db(&sos, 0.0));
        assert!(
            (mag_db(&sos, 500.0) - (-0.703)).abs() < 0.1,
            "500Hz: {}",
            mag_db(&sos, 500.0)
        );
        assert!(
            (mag_db(&sos, 1000.0) - (-3.01)).abs() < 0.1,
            "cutoff −3 dB: {}",
            mag_db(&sos, 1000.0)
        );
        assert!(
            (mag_db(&sos, 2000.0) - (-13.533)).abs() < 0.3,
            "2000Hz: {}",
            mag_db(&sos, 2000.0)
        );
        assert!(
            (mag_db(&sos, 4000.0) - (-35.295)).abs() < 0.5,
            "4000Hz: {}",
            mag_db(&sos, 4000.0)
        );

        // Monotonic roll-off (no ripple): magnitude strictly decreasing.
        let mut prev = mag_db(&sos, 0.0);
        for f in [50.0, 200.0, 500.0, 800.0, 1000.0, 1500.0, 2000.0, 3000.0] {
            let cur = mag_db(&sos, f);
            assert!(cur <= prev + 1e-6, "non-monotonic at {f}Hz: {cur} > {prev}");
            prev = cur;
        }
    }

    #[test]
    fn test_bessel_lowpass_flat_group_delay() {
        // The defining Bessel property: approximately flat group delay across
        // the passband. We estimate group delay by finite-differencing the
        // unwrapped phase of the SOS response.
        let proto = super::bessel_prototype(4);
        let sos = super::design_from_prototype(
            &proto,
            FilterResponse::LowPass,
            FS,
            Some(1000.0),
            None,
            None,
        );

        // group delay τ(f) = −dφ/dω, ω = 2πf/FS. Sample finely in passband.
        let freqs: Vec<f64> = (1..=40).map(|i| i as f64 * 20.0).collect(); // 20..800 Hz
        let (_, phases) = sos.frequency_response(&freqs, FS);
        // unwrap phase
        let mut unwrapped = phases;
        for i in 1..unwrapped.len() {
            let mut d = unwrapped[i] - unwrapped[i - 1];
            while d > PI {
                d -= 2.0 * PI;
            }
            while d < -PI {
                d += 2.0 * PI;
            }
            unwrapped[i] = unwrapped[i - 1] + d;
        }
        let dw = 2.0 * PI * 20.0 / FS;
        let mut gds = Vec::new();
        for i in 1..unwrapped.len() {
            gds.push(-(unwrapped[i] - unwrapped[i - 1]) / dw);
        }
        let mean: f64 = gds.iter().sum::<f64>() / gds.len() as f64;
        // scipy group delay is ~14.8 samples and very flat across passband.
        for &g in &gds {
            assert!(
                (g - mean).abs() / mean < 0.05,
                "group delay not flat: {g} vs mean {mean}"
            );
        }
        assert!(
            (mean - 14.8).abs() < 1.0,
            "group delay mean {mean} (expected ~14.8 samples)"
        );
    }

    #[test]
    fn test_bessel_highpass_response() {
        // order 4, cutoff 1000 Hz. SciPy: Nyquist 0 dB, 5000 −0.10,
        // 2000 −0.70, 1000 −3.01, 500 −13.44, 100 −65.7.
        let proto = super::bessel_prototype(4);
        let sos = super::design_from_prototype(
            &proto,
            FilterResponse::HighPass,
            FS,
            Some(1000.0),
            None,
            None,
        );
        assert!(
            mag_db(&sos, 22050.0).abs() < 0.05,
            "Nyquist: {}",
            mag_db(&sos, 22050.0)
        );
        assert!(
            (mag_db(&sos, 2000.0) - (-0.698)).abs() < 0.1,
            "2000Hz: {}",
            mag_db(&sos, 2000.0)
        );
        assert!(
            (mag_db(&sos, 1000.0) - (-3.01)).abs() < 0.1,
            "cutoff: {}",
            mag_db(&sos, 1000.0)
        );
        assert!(
            (mag_db(&sos, 500.0) - (-13.437)).abs() < 0.4,
            "500Hz: {}",
            mag_db(&sos, 500.0)
        );
    }

    #[test]
    fn test_bessel_bandpass_response() {
        // order 4, band [500,2000], centre 1000 Hz. SciPy: centre ~0 dB,
        // 100 −51 dB, 8000 −47 dB, 250 −19.5 dB.
        let proto = super::bessel_prototype(4);
        let sos = super::design_from_prototype(
            &proto,
            FilterResponse::BandPass,
            FS,
            None,
            Some(500.0),
            Some(2000.0),
        );
        let center = (500.0_f64 * 2000.0).sqrt();
        assert!(
            mag_db(&sos, center).abs() < 0.5,
            "centre: {}",
            mag_db(&sos, center)
        );
        assert!(
            mag_db(&sos, 100.0) <= -30.0,
            "100Hz: {}",
            mag_db(&sos, 100.0)
        );
        assert!(
            mag_db(&sos, 8000.0) <= -30.0,
            "8000Hz: {}",
            mag_db(&sos, 8000.0)
        );
    }

    #[test]
    fn test_bessel_bandstop_response() {
        // order 4, band [500,2000], notch at 1000 Hz. SciPy: notch very deep,
        // DC/15 kHz ~0 dB.
        let proto = super::bessel_prototype(4);
        let sos = super::design_from_prototype(
            &proto,
            FilterResponse::BandStop,
            FS,
            None,
            Some(500.0),
            Some(2000.0),
        );
        let center = (500.0_f64 * 2000.0).sqrt();
        assert!(
            mag_db(&sos, center) <= -40.0,
            "notch: {}",
            mag_db(&sos, center)
        );
        assert!(mag_db(&sos, 50.0).abs() < 0.5, "50Hz: {}", mag_db(&sos, 50.0));
        assert!(
            mag_db(&sos, 15000.0).abs() < 0.5,
            "15kHz: {}",
            mag_db(&sos, 15000.0)
        );
    }

    #[test]
    fn test_bessel_dispatcher_wired() {
        let samples = NonEmptyVec::new(vec![1.0f32; 256]).unwrap();
        let mut audio: AudioSamples<'_, f32> =
            AudioSamples::from_mono_vec(samples, sample_rate!(44100));
        let design =
            IirFilterDesign::bessel(FilterResponse::LowPass, NonZeroUsize::new(4).unwrap(), 1000.0);
        assert!(audio.apply_iir_filter_in_place(&design).is_ok());

        let samples2 = NonEmptyVec::new(vec![1.0f32; 256]).unwrap();
        let mut audio2: AudioSamples<'_, f32> =
            AudioSamples::from_mono_vec(samples2, sample_rate!(44100));
        let bp = IirFilterDesign::bessel_band(
            FilterResponse::BandPass,
            NonZeroUsize::new(4).unwrap(),
            500.0,
            2000.0,
        );
        assert!(audio2.apply_iir_filter_in_place(&bp).is_ok());
    }

    // ========================================================================
    // Elliptic (Cauer) filter tests
    //
    // Reference dB values cross-checked against
    //   scipy.signal.ellip(N, rp, rs, Wn, btype, output='sos', fs=44100)
    // evaluated with signal.sosfreqz. Checkpoints noted inline.
    // ========================================================================

    /// The analog elliptic prototype zeros/poles/gain must match SciPy's
    /// `ellipap` to ≈1e-9 for both even and odd orders.
    #[test]
    fn test_elliptic_prototype_matches_scipy() {
        // scipy.signal.ellip(4, 1, 40, 1, analog=True, output='zpk').
        let poles_ref = [
            Complex::new(-0.1052812646, -0.9937108112),
            Complex::new(-0.3642905959, -0.4786027676),
            Complex::new(-0.3642905959, 0.4786027676),
            Complex::new(-0.1052812646, 0.9937108112),
        ];
        let zeros_ref = [
            Complex::new(0.0, -3.5252874330),
            Complex::new(0.0, -1.6095504012),
            Complex::new(0.0, 1.6095504012),
            Complex::new(0.0, 3.5252874330),
        ];
        let k_ref = 1.000000000000e-02;

        let proto = super::elliptic_prototype(4, 1.0, 40.0);
        let mut p = proto.poles.clone();
        p.sort_by(|a, b| {
            a.im.partial_cmp(&b.im)
                .unwrap()
                .then(a.re.partial_cmp(&b.re).unwrap())
        });
        let mut z = proto.zeros.clone();
        z.sort_by(|a, b| {
            a.im.partial_cmp(&b.im)
                .unwrap()
                .then(a.re.partial_cmp(&b.re).unwrap())
        });
        for (g, e) in p.iter().zip(poles_ref.iter()) {
            assert!((g - e).norm() < 1e-8, "pole mismatch got {g} exp {e}");
        }
        for (g, e) in z.iter().zip(zeros_ref.iter()) {
            assert!((g - e).norm() < 1e-7, "zero mismatch got {g} exp {e}");
        }
        assert!(
            (proto.gain - k_ref).abs() < 1e-9,
            "gain mismatch got {} exp {k_ref}",
            proto.gain
        );

        // Odd order N=3, rp=0.5, rs=60: pole/zero counts and a real pole.
        let proto3 = super::elliptic_prototype(3, 0.5, 60.0);
        assert_eq!(proto3.poles.len(), 3, "N=3 has 3 poles");
        assert_eq!(proto3.zeros.len(), 2, "N=3 drops one zero to infinity");
        assert!(
            proto3.poles.iter().any(|p| p.im.abs() < 1e-9),
            "odd order should have a real pole"
        );
    }

    #[test]
    fn test_elliptic_lowpass_response() {
        // order 4, cutoff 2000 Hz, rp=1, rs=40. SciPy: DC −1.0, 500 −0.435,
        // 1000 −0.085, 1500 −0.998, 1900 −0.013, 2000 −1.0, 2500 −18.82,
        // 4000 −40.03, 8000 −49.2.
        let proto = super::elliptic_prototype(4, 1.0, 40.0);
        let sos = super::design_from_prototype_with_gain(
            &proto,
            FilterResponse::LowPass,
            FS,
            Some(2000.0),
            None,
            None,
            Some(super::GainRef::Prototype),
        );
        // Passband ripple stays within [−1, 0] dB (allow tiny slack).
        for f in [0.0, 500.0, 1000.0, 1500.0, 1900.0, 2000.0] {
            let g = mag_db(&sos, f);
            assert!(
                (-1.0 - 0.1..=0.05).contains(&g),
                "passband {f}Hz gain {g} outside ripple bound"
            );
        }
        // Equiripple: the passband touches the −1 dB bound more than once.
        let touches = [0.0_f64, 1500.0, 2000.0]
            .iter()
            .filter(|&&f| (mag_db(&sos, f) - (-1.0)).abs() < 0.05)
            .count();
        assert!(touches >= 2, "expected multiple −1 dB touches, got {touches}");
        // Equiripple peaks (touch 0 dB) between the −1 dB dips.
        assert!(
            mag_db(&sos, 1000.0) >= -0.2,
            "passband peak 1000Hz: {}",
            mag_db(&sos, 1000.0)
        );
        assert!(
            mag_db(&sos, 1900.0) >= -0.2,
            "passband peak 1900Hz: {}",
            mag_db(&sos, 1900.0)
        );

        // Stopband: ≥40 dB at and beyond the stopband edge.
        assert!(
            mag_db(&sos, 4000.0) <= -40.0 + 0.5,
            "4000Hz: {}",
            mag_db(&sos, 4000.0)
        );
        assert!(
            mag_db(&sos, 8000.0) <= -40.0 + 0.5,
            "8000Hz: {}",
            mag_db(&sos, 8000.0)
        );
        // Cross-check exact SciPy transition value.
        assert!(
            (mag_db(&sos, 2500.0) - (-18.819)).abs() < 0.5,
            "2500Hz: {}",
            mag_db(&sos, 2500.0)
        );
    }

    #[test]
    fn test_elliptic_highpass_response() {
        // order 4, cutoff 2000 Hz, rp=1, rs=40. SciPy: Nyquist −1.0,
        // 4000 −0.066, 2000 −1.0, 1500 −24.6, 1000 −40.0, 500 −53.6.
        let proto = super::elliptic_prototype(4, 1.0, 40.0);
        let sos = super::design_from_prototype_with_gain(
            &proto,
            FilterResponse::HighPass,
            FS,
            Some(2000.0),
            None,
            None,
            Some(super::GainRef::Prototype),
        );
        for f in [4000.0, 8000.0, 22000.0] {
            let g = mag_db(&sos, f);
            assert!((-1.1..=0.05).contains(&g), "passband {f}Hz gain {g}");
        }
        assert!(
            (mag_db(&sos, 2000.0) - (-1.0)).abs() < 0.1,
            "edge 2000Hz: {}",
            mag_db(&sos, 2000.0)
        );
        assert!(
            mag_db(&sos, 1000.0) <= -40.0 + 0.5,
            "1000Hz stopband: {}",
            mag_db(&sos, 1000.0)
        );
        assert!(
            mag_db(&sos, 500.0) <= -40.0 + 0.5,
            "500Hz stopband: {}",
            mag_db(&sos, 500.0)
        );
    }

    #[test]
    fn test_elliptic_bandpass_response() {
        // order 4, band [500,2000], rp=1, rs=40. SciPy: band edges and centre
        // within ripple bound; 100 −43, 8000 −44, 250 −43 dB.
        let proto = super::elliptic_prototype(4, 1.0, 40.0);
        let sos = super::design_from_prototype_with_gain(
            &proto,
            FilterResponse::BandPass,
            FS,
            None,
            Some(500.0),
            Some(2000.0),
            Some(super::GainRef::Prototype),
        );
        let center = (500.0_f64 * 2000.0).sqrt();
        for f in [500.0, center, 2000.0] {
            let g = mag_db(&sos, f);
            assert!((-1.0 - 0.2..=0.1).contains(&g), "passband {f}Hz gain {g}");
        }
        assert!(
            mag_db(&sos, 100.0) <= -40.0 + 0.5,
            "100Hz: {}",
            mag_db(&sos, 100.0)
        );
        assert!(
            mag_db(&sos, 8000.0) <= -40.0 + 0.5,
            "8000Hz: {}",
            mag_db(&sos, 8000.0)
        );
    }

    #[test]
    fn test_elliptic_bandstop_response() {
        // order 4, band [500,2000], rp=1, rs=40. SciPy: notch band ≥40 dB,
        // DC/15 kHz within ripple bound.
        let proto = super::elliptic_prototype(4, 1.0, 40.0);
        let sos = super::design_from_prototype_with_gain(
            &proto,
            FilterResponse::BandStop,
            FS,
            None,
            Some(500.0),
            Some(2000.0),
            Some(super::GainRef::Prototype),
        );
        let center = (500.0_f64 * 2000.0).sqrt();
        assert!(
            mag_db(&sos, center) <= -40.0 + 0.5,
            "notch centre: {}",
            mag_db(&sos, center)
        );
        assert!(
            mag_db(&sos, 500.0) <= -1.0 + 0.1,
            "edge 500Hz: {}",
            mag_db(&sos, 500.0)
        );
        assert!(
            mag_db(&sos, 2000.0) <= -1.0 + 0.1,
            "edge 2000Hz: {}",
            mag_db(&sos, 2000.0)
        );
        // Passband on both sides within the ripple bound.
        assert!(mag_db(&sos, 50.0) >= -1.1, "50Hz: {}", mag_db(&sos, 50.0));
        assert!(
            mag_db(&sos, 15000.0) >= -1.1,
            "15kHz: {}",
            mag_db(&sos, 15000.0)
        );
    }

    #[test]
    fn test_elliptic_dispatcher_wired() {
        let samples = NonEmptyVec::new(vec![1.0f32; 256]).unwrap();
        let mut audio: AudioSamples<'_, f32> =
            AudioSamples::from_mono_vec(samples, sample_rate!(44100));
        let design = IirFilterDesign::elliptic(
            FilterResponse::LowPass,
            NonZeroUsize::new(4).unwrap(),
            2000.0,
            1.0,
            40.0,
        );
        assert!(audio.apply_iir_filter_in_place(&design).is_ok());

        let samples2 = NonEmptyVec::new(vec![1.0f32; 256]).unwrap();
        let mut audio2: AudioSamples<'_, f32> =
            AudioSamples::from_mono_vec(samples2, sample_rate!(44100));
        let bs = IirFilterDesign::elliptic_band(
            FilterResponse::BandStop,
            NonZeroUsize::new(4).unwrap(),
            500.0,
            2000.0,
            1.0,
            40.0,
        );
        assert!(audio2.apply_iir_filter_in_place(&bs).is_ok());
    }

    #[test]
    fn test_elliptic_validation_errors() {
        let samples = NonEmptyVec::new(vec![1.0f32; 64]).unwrap();
        let mut audio: AudioSamples<'_, f32> =
            AudioSamples::from_mono_vec(samples, sample_rate!(44100));
        // rs must exceed rp.
        let bad = IirFilterDesign::elliptic(
            FilterResponse::LowPass,
            NonZeroUsize::new(4).unwrap(),
            2000.0,
            5.0,
            3.0,
        );
        assert!(audio.apply_iir_filter_in_place(&bad).is_err());
    }

    // ========================================================================
    // Streaming (design-once) SosFilter + zero-phase filtfilt
    // ========================================================================

    /// Streaming continuity: filtering a signal in two consecutive
    /// `process_block` halves on a single stateful SosFilter must equal
    /// filtering the whole signal in one pass.
    #[test]
    fn test_sos_streaming_block_continuity() {
        let design =
            IirFilterDesign::butterworth_lowpass(NonZeroUsize::new(6).unwrap(), 1500.0);

        // Build a non-trivial signal.
        let n = 512;
        let signal: Vec<f64> = (0..n)
            .map(|i| {
                let t = i as f64 / FS;
                (2.0 * PI * 300.0 * t).sin() + 0.4 * (2.0 * PI * 4000.0 * t).sin()
            })
            .collect();

        // One-pass reference.
        let mut sos_whole = SosFilter::from_design(&design, FS).unwrap();
        let mut whole = signal.clone();
        sos_whole.process_block(&mut whole);

        // Two consecutive halves on one streaming filter (state retained).
        let mut sos_stream = design.to_sos(FS).unwrap();
        let mut streamed = signal.clone();
        let mid = n / 2;
        let (first, second) = streamed.split_at_mut(mid);
        sos_stream.process_block(first);
        sos_stream.process_block(second);

        for (i, (a, b)) in whole.iter().zip(streamed.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-12,
                "stream discontinuity at {i}: whole={a}, streamed={b}"
            );
        }

        // reset() restores the freshly-constructed state.
        sos_stream.reset();
        let mut after_reset = signal;
        sos_stream.process_block(&mut after_reset);
        for (a, b) in whole.iter().zip(after_reset.iter()) {
            assert!((a - b).abs() < 1e-12, "reset did not restore clean state");
        }
    }

    /// Zero phase: a symmetric input stays symmetric after filtfilt, and the
    /// cross-correlation peak with the input is at lag 0. Contrast with the
    /// single-pass filter, which group-delays the signal (peak at lag > 0).
    #[test]
    fn test_filtfilt_zero_phase_symmetry() {
        let design =
            IirFilterDesign::butterworth_lowpass(NonZeroUsize::new(4).unwrap(), 2000.0);

        // Symmetric input: a centred triangular burst on a zero background.
        let n = 401usize;
        let center = n / 2;
        let mut sig = vec![0.0f64; n];
        for (i, s) in sig.iter_mut().enumerate() {
            let d = (i as isize - center as isize).abs();
            if d <= 20 {
                *s = 20.0 - d as f64;
            }
        }
        // sig is exactly symmetric about `center`.

        // --- filtfilt: must remain symmetric ---
        let samples = NonEmptyVec::new(sig.iter().map(|&v| v as f32).collect()).unwrap();
        let mut audio: AudioSamples<'_, f32> =
            AudioSamples::from_mono_vec(samples, sample_rate!(44100));
        audio.filtfilt_in_place(&design).unwrap();
        let ff: Vec<f64> = audio
            .as_slice()
            .unwrap()
            .iter()
            .map(|&v| v as f64)
            .collect();

        // Symmetry: ff[center - k] ≈ ff[center + k].
        let mut max_asym = 0.0f64;
        for k in 0..=center.min(n - 1 - center) {
            max_asym = max_asym.max((ff[center - k] - ff[center + k]).abs());
        }
        let peak = ff.iter().cloned().fold(0.0f64, |a, b| a.max(b.abs()));
        assert!(
            max_asym < 1e-4 * peak.max(1e-9),
            "filtfilt broke symmetry: max_asym={max_asym}, peak={peak}"
        );

        // Cross-correlation peak between input and filtfilt output at lag 0.
        let best_lag_ff = best_xcorr_lag(&sig, &ff, 40);
        assert_eq!(best_lag_ff, 0, "filtfilt should have zero group delay");

        // --- single-pass: DOES delay (peak at positive lag) ---
        let samples2 = NonEmptyVec::new(sig.iter().map(|&v| v as f32).collect()).unwrap();
        let mut audio2: AudioSamples<'_, f32> =
            AudioSamples::from_mono_vec(samples2, sample_rate!(44100));
        audio2.apply_iir_filter_in_place(&design).unwrap();
        let sp: Vec<f64> = audio2
            .as_slice()
            .unwrap()
            .iter()
            .map(|&v| v as f64)
            .collect();
        let best_lag_sp = best_xcorr_lag(&sig, &sp, 40);
        assert!(
            best_lag_sp > 0,
            "single-pass filter should be group-delayed, got lag {best_lag_sp}"
        );
    }

    /// Magnitude: the effective filtfilt magnitude is the SQUARE of the
    /// single-pass magnitude (≈ 2x the single-pass attenuation in dB), at both
    /// a passband and a stopband frequency.
    #[test]
    fn test_filtfilt_magnitude_is_squared() {
        let design =
            IirFilterDesign::butterworth_lowpass(NonZeroUsize::new(4).unwrap(), 2000.0);
        let sos = SosFilter::from_design(&design, FS).unwrap();

        // Pass a pure tone through filtfilt and measure the steady-state
        // amplitude ratio; compare to the squared single-pass |H|.
        for &freq in &[500.0f64, 6000.0] {
            let n = 8192usize;
            let sig: Vec<f64> = (0..n)
                .map(|i| (2.0 * PI * freq * (i as f64) / FS).sin())
                .collect();

            let samples =
                NonEmptyVec::new(sig.iter().map(|&v| v as f32).collect()).unwrap();
            let mut audio: AudioSamples<'_, f32> =
                AudioSamples::from_mono_vec(samples, sample_rate!(44100));
            audio.filtfilt_in_place(&design).unwrap();
            let out: Vec<f64> = audio
                .as_slice()
                .unwrap()
                .iter()
                .map(|&v| v as f64)
                .collect();

            // Steady-state amplitude over the central region (avoid any edges).
            let lo = n / 4;
            let hi = 3 * n / 4;
            let amp_out = out[lo..hi].iter().cloned().fold(0.0f64, |a, b| a.max(b.abs()));
            let measured_db = db(amp_out); // input amplitude is 1.0

            // Single-pass magnitude (linear) -> squared -> dB.
            let (m, _) = sos.frequency_response(&[freq], FS);
            let single_db = db(m[0]);
            let expected_db = 2.0 * single_db;

            assert!(
                (measured_db - expected_db).abs() < 1.0,
                "freq {freq}: filtfilt {measured_db:.2} dB vs expected square {expected_db:.2} dB (single {single_db:.2})"
            );
        }
    }

    /// Cross-correlation lag (in `-max_lag..=max_lag`) at which `b` best aligns
    /// with `a`; positive lag means `b` is delayed relative to `a`.
    fn best_xcorr_lag(a: &[f64], b: &[f64], max_lag: isize) -> isize {
        let n = a.len() as isize;
        let mut best_lag = 0isize;
        let mut best_val = f64::NEG_INFINITY;
        for lag in -max_lag..=max_lag {
            let mut acc = 0.0;
            // j = i + lag: positive `lag` means `b` is shifted later (delayed)
            // relative to `a`, so a group-delayed `b` peaks at positive lag.
            for i in 0..n {
                let j = i + lag;
                if j >= 0 && j < n {
                    acc += a[i as usize] * b[j as usize];
                }
            }
            if acc > best_val {
                best_val = acc;
                best_lag = lag;
            }
        }
        best_lag
    }
}
