//! IIR (Infinite Impulse Response) filter design and application.
//!
//! ## What
//!
//! This module provides:
//!
//! - [`IirFilter`] - a stateful direct-form filter for arbitrary `(b, a)`
//!   coefficients (the equivalent of `scipy.signal.lfilter`).
//! - [`SosFilter`] - a stateful cascade of second-order sections, the
//!   numerically robust representation used for every designed filter
//!   (the equivalent of `scipy.signal.sosfilt`).
//! - The [`AudioIirFiltering`] implementation for [`AudioSamples`], which
//!   designs and applies Butterworth, Chebyshev Type I, Chebyshev Type II,
//!   Bessel and elliptic filters in low-pass, high-pass, band-pass and
//!   band-stop form, plus zero-phase forward-backward filtering.
//!
//! ## Why
//!
//! IIR filters reach a given roll-off with far fewer operations than FIR
//! filters, which makes them the standard tool for tone shaping, rumble and
//! hiss removal, band limiting and crossover design.
//!
//! ## How
//!
//! Designs follow the classic analog-prototype pipeline exactly as SciPy
//! implements it: normalised analog prototype → analog frequency transform →
//! bilinear transform (with frequency pre-warping) → pairing into second-order
//! sections. The coefficients produced for a given [`IirFilterDesign`] agree
//! with `scipy.signal.iirfilter(..., output='sos', fs=fs)` to within a few
//! ULPs, and filtering agrees with `scipy.signal.sosfilt` / `sosfiltfilt`.
//! This is verified by the golden-fixture tests in `tests/filter_reference.rs`,
//! which also cross-check against independent C/C++ implementations
//! (DSPFilters, liquid-dsp).
//!
//! | `IirFilterDesign` | SciPy equivalent |
//! |---|---|
//! | `butterworth_lowpass(N, fc)` | `butter(N, fc, 'lowpass', fs=fs, output='sos')` |
//! | `chebyshev_i(resp, N, fc, rp)` | `cheby1(N, rp, fc, resp, fs=fs, output='sos')` |
//! | `chebyshev_ii(resp, N, fc, rs)` | `cheby2(N, rs, fc, resp, fs=fs, output='sos')` |
//! | `elliptic(resp, N, fc, rp, rs)` | `ellip(N, rp, rs, fc, resp, fs=fs, output='sos')` |
//! | `bessel(resp, N, fc)` | `bessel(N, fc, resp, norm='mag', fs=fs, output='sos')` |
//!
//! Integer sample types are converted to `f64` for filtering and converted
//! back afterwards; results that exceed full scale are clamped.
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

use crate::operations::iir_design::{self, Sos};
use crate::operations::traits::AudioIirFiltering;
use crate::operations::types::{FilterResponse, IirFilterDesign};

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
    /// Feed-forward (numerator) coefficients, ordered as `b[0]`, `b[1]`, …, `b[M]`.
    pub b_coeffs: Vec<f64>,
    /// Feed-back (denominator) coefficients, ordered as `a[0]`, `a[1]`, …, `a[N]`.
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
    /// - `b_coeffs` - Feed-forward (numerator) coefficients \[b₀, b₁, …, bₘ\].
    /// - `a_coeffs` - Feed-back (denominator) coefficients \[a₀, a₁, …, aₙ\].
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
    /// - `input` - The current input sample.
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
    /// - `input` - Contiguous slice of input samples.
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
    /// - `input` - Mutable slice; each element is replaced with its
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

    /// Process a block in place; bit-identical to calling
    /// [`Self::process_sample`] for each element, but with an allocation-free,
    /// register-resident fast path for sections of order ≤ 2.
    #[inline]
    pub(crate) fn process_biquad_block(&mut self, block: &mut [f64]) {
        if self.b_coeffs.len() != 3 || self.a_coeffs.len() != 3 {
            for x in block.iter_mut() {
                *x = self.process_sample(*x);
            }
            return;
        }
        let (b0, b1, b2) = (self.b_coeffs[0], self.b_coeffs[1], self.b_coeffs[2]);
        let (a0, a1, a2) = (self.a_coeffs[0], self.a_coeffs[1], self.a_coeffs[2]);
        let (mut x1, mut x2) = (self.x_delays[0], self.x_delays[1]);
        let (mut y1, mut y2) = (self.y_delays[0], self.y_delays[1]);
        for v in block.iter_mut() {
            let x = *v;
            let mut y = b0 * x;
            y += b1 * x1;
            y += b2 * x2;
            y -= a1 * y1;
            y -= a2 * y2;
            y /= a0;
            x2 = x1;
            x1 = x;
            y2 = y1;
            y1 = y;
            *v = y;
        }
        self.x_delays[0] = x1;
        self.x_delays[1] = x2;
        self.y_delays[0] = y1;
        self.y_delays[1] = y2;
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
    /// - `frequencies` - Frequencies in Hz at which to evaluate the response.
    /// - `sample_rate` - Sample rate of the signal in Hz.
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
// Transposed direct form II biquad (the `scipy.signal.sosfilt` kernel)
// ============================================================================

/// Second-order section in transposed direct form II with `a0 == 1`.
///
/// The recurrence and its operation order are identical to SciPy's
/// `sosfilt` inner loop:
/// ```text
///   y    = b0·x + s0
///   s0'  = b1·x - a1·y + s1
///   s1'  = b2·x - a2·y
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
    fn from_sos(s: &Sos) -> Self {
        let a0 = s[3];
        Self {
            b0: s[0] / a0,
            b1: s[1] / a0,
            b2: s[2] / a0,
            a1: s[4] / a0,
            a2: s[5] / a0,
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
}

/// Run `signal` through a TDF-II cascade in place, starting from (and
/// leaving the final state in) `bqs`.
#[inline]
fn sosfilt_slice(bqs: &mut [Biquad], signal: &mut [f64]) {
    match bqs {
        [] => {}
        [b] => {
            let mut bq = *b;
            for x in signal.iter_mut() {
                *x = bq.process(*x);
            }
            *b = bq;
        }
        [b0, b1] => {
            let (mut q0, mut q1) = (*b0, *b1);
            for x in signal.iter_mut() {
                *x = q1.process(q0.process(*x));
            }
            *b0 = q0;
            *b1 = q1;
        }
        _ => {
            // Process in L1-sized blocks, section-major within a block: each
            // section's coefficients and state stay in registers while it
            // sweeps the block, and the block stays cache resident.
            const BLOCK: usize = 256;
            for chunk in signal.chunks_mut(BLOCK) {
                for b in bqs.iter_mut() {
                    let mut bq = *b;
                    for x in chunk.iter_mut() {
                        *x = bq.process(*x);
                    }
                    *b = bq;
                }
            }
        }
    }
}

fn biquads(sos: &[Sos]) -> Vec<Biquad> {
    sos.iter().map(Biquad::from_sos).collect()
}

/// Zero-phase forward-backward filtering of `signal` in place, matching
/// `scipy.signal.sosfiltfilt(sos, x, padtype='odd')`.
///
/// - The signal is extended at both ends by odd reflection of
///   `padlen = 3 · (2·n_sections + 1 - min(#b2==0, #a2==0))` samples.
/// - Each pass starts from the steady-state initial conditions
///   ([`iir_design::sosfilt_zi`]) scaled by the first sample of the data it
///   filters, which suppresses the start-up transient.
///
/// SciPy rejects signals with `len ≤ padlen`; here the padding is reduced to
/// `len - 1` instead so short signals are still filtered.
pub(crate) fn sosfiltfilt_slice(sos: &[Sos], signal: &mut [f64]) {
    let len = signal.len();
    if len == 0 || sos.is_empty() {
        return;
    }
    let n_sections = sos.len();
    let zb2 = sos.iter().filter(|s| s[2] == 0.0).count();
    let za2 = sos.iter().filter(|s| s[5] == 0.0).count();
    let ntaps = 2 * n_sections + 1 - zb2.min(za2);
    let edge = (3 * ntaps).min(len - 1);

    let mut ext = Vec::with_capacity(len + 2 * edge);
    let x0 = signal[0];
    let xn = signal[len - 1];
    ext.extend((1..=edge).rev().map(|k| 2.0 * x0 - signal[k]));
    ext.extend_from_slice(signal);
    ext.extend((1..=edge).map(|k| 2.0 * xn - signal[len - 1 - k]));

    let zi = iir_design::sosfilt_zi(sos);
    let mut bqs = biquads(sos);

    let set_state = |bqs: &mut [Biquad], scale: f64| {
        for (b, z) in bqs.iter_mut().zip(zi.iter()) {
            b.s0 = z[0] * scale;
            b.s1 = z[1] * scale;
        }
    };

    set_state(&mut bqs, ext[0]);
    sosfilt_slice(&mut bqs, &mut ext);
    ext.reverse();
    set_state(&mut bqs, ext[0]);
    sosfilt_slice(&mut bqs, &mut ext);
    ext.reverse();

    signal.copy_from_slice(&ext[edge..edge + len]);
}

// ============================================================================
// SosFilter (public streaming cascade)
// ============================================================================

/// A second-order sections (SOS) cascade filter.
///
/// Represents an IIR filter as a cascade of second-order (biquad) sections.
/// This architecture is numerically stable for high-order filters, where
/// direct-form implementations suffer from coefficient quantisation and
/// floating-point errors. It is the representation used by SciPy
/// (`output='sos'`), MATLAB and most professional DSP tools.
///
/// Each section is an [`IirFilter`] that processes the output of the previous
/// section, and keeps its own delay-line state between calls, so an
/// `SosFilter` can be fed a stream block by block.
///
/// Use [`SosFilter::from_design`] to design a filter, or
/// [`SosFilter::from_sos_coefficients`] to load coefficients produced
/// elsewhere (for example `scipy.signal.butter(..., output='sos')`).
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
    /// The sections are applied in order: input → `section[0]` → `section[1]` → ... → output.
    ///
    /// # Arguments
    /// - `sections` - Vector of second-order [`IirFilter`] sections.
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

    /// Build a filter from second-order-section coefficients in SciPy layout.
    ///
    /// Each row is `[b0, b1, b2, a0, a1, a2]`, exactly the layout of the array
    /// returned by `scipy.signal.butter(..., output='sos')` and friends, and of
    /// MATLAB's `sos` matrices. Rows are normalised so that `a0 == 1`.
    ///
    /// # Arguments
    /// - `sos` - One row per section; must not be empty.
    ///
    /// # Returns
    /// A freshly-reset [`SosFilter`].
    ///
    /// # Errors
    /// - [crate::AudioSampleError::Parameter] - if `sos` is empty, any
    ///   coefficient is non-finite, or any `a0` is zero.
    ///
    /// # Examples
    /// ```
    /// use audio_samples::operations::iir_filtering::SosFilter;
    ///
    /// // scipy.signal.butter(2, 1000, fs=48000, output='sos')
    /// let sos = SosFilter::from_sos_coefficients(&[[
    ///     0.00391612666054737, 0.00783225332109474, 0.00391612666054737,
    ///     1.0, -1.815341082704568, 0.8310055893467575,
    /// ]]).unwrap();
    /// assert_eq!(sos.sections.len(), 1);
    /// ```
    pub fn from_sos_coefficients(sos: &[[f64; 6]]) -> AudioSampleResult<Self> {
        if sos.is_empty() {
            return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
                "sos",
                "at least one second-order section is required",
            )));
        }
        let mut sections = Vec::with_capacity(sos.len());
        for row in sos {
            if row.iter().any(|v| !v.is_finite()) || row[3] == 0.0 {
                return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
                    "sos",
                    "coefficients must be finite and a0 must be non-zero",
                )));
            }
            let a0 = row[3];
            sections.push(IirFilter::new(
                vec![row[0] / a0, row[1] / a0, row[2] / a0],
                vec![1.0, row[4] / a0, row[5] / a0],
            ));
        }
        Ok(Self { sections })
    }

    /// Return the coefficients as rows of `[b0, b1, b2, a0, a1, a2]`.
    ///
    /// This is the SciPy / MATLAB `sos` layout, suitable for
    /// `scipy.signal.sosfilt` or `sosfreqz`. Sections with fewer than three
    /// coefficients are zero-padded.
    ///
    /// # Examples
    /// ```
    /// use audio_samples::operations::iir_filtering::SosFilter;
    /// use audio_samples::operations::types::IirFilterDesign;
    /// use std::num::NonZeroUsize;
    ///
    /// let design = IirFilterDesign::butterworth_lowpass(NonZeroUsize::new(4).unwrap(), 1000.0);
    /// let sos = SosFilter::from_design(&design, 48000.0).unwrap();
    /// let rows = sos.sos_coefficients();
    /// assert_eq!(rows.len(), 2);
    /// assert_eq!(rows[0][3], 1.0);
    /// ```
    #[must_use]
    pub fn sos_coefficients(&self) -> Vec<[f64; 6]> {
        self.sections
            .iter()
            .map(|s| {
                let g = |v: &[f64], i: usize| v.get(i).copied().unwrap_or(0.0);
                [
                    g(&s.b_coeffs, 0),
                    g(&s.b_coeffs, 1),
                    g(&s.b_coeffs, 2),
                    g(&s.a_coeffs, 0),
                    g(&s.a_coeffs, 1),
                    g(&s.a_coeffs, 2),
                ]
            })
            .collect()
    }

    /// Process a single sample through the cascade.
    ///
    /// Feeds the input through each section in order, passing the
    /// output of `section[i]` as the input to `section[i+1]`.
    ///
    /// # Arguments
    /// - `input` - The current input sample.
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
    /// - `input` - Contiguous slice of input samples.
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
        let mut out = input.to_vec();
        self.process_samples_in_place(&mut out);
        out
    }

    /// Process a mutable slice of samples through the cascade in place.
    ///
    /// Overwrites each element with the corresponding filtered output.
    /// Equivalent to [`Self::process_samples`] but avoids allocating.
    ///
    /// Second-order sections run through an allocation-free fast path; the
    /// result is identical to calling [`Self::process_sample`] per sample.
    ///
    /// # Arguments
    /// - `input` - Mutable slice; each element is replaced with its
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
        const BLOCK: usize = 256;
        for chunk in input.chunks_mut(BLOCK) {
            for section in &mut self.sections {
                section.process_biquad_block(chunk);
            }
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
    /// Designs the filter once (via the same pipeline used by
    /// [`AudioIirFiltering::apply_iir_filter_in_place`]) and returns a stateful
    /// cascade whose [`Self::process_sample`] / [`Self::process_block`] retain
    /// state across calls. This lets real-time / block processing avoid
    /// redesigning the filter for every block.
    ///
    /// The coefficients equal those of the matching SciPy design (see the
    /// [module documentation](self)).
    ///
    /// # Arguments
    /// - `design` - Filter specification (type, order, frequencies, ripple).
    /// - `sample_rate` - Sample rate of the signal in Hz.
    ///
    /// # Returns
    /// A freshly-reset [`SosFilter`] implementing the requested design.
    ///
    /// # Errors
    /// - [crate::AudioSampleError::Parameter] - if the design is invalid
    ///   (frequency outside `(0, Nyquist)`, missing or non-positive ripple /
    ///   attenuation, order above the supported maximum, …).
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
    pub fn from_design(design: &IirFilterDesign, sample_rate: f64) -> AudioSampleResult<SosFilter> {
        let sos = iir_design::design_sos(design, sample_rate)?;
        Self::from_sos_coefficients(&sos)
    }

    /// Process a block of samples in place, retaining filter state across calls.
    ///
    /// Each element of `block` is replaced with its filtered value, in order,
    /// flowing through every section. Because the internal delay lines persist,
    /// calling `process_block` on consecutive halves of a signal yields exactly
    /// the same result as calling it once on the whole signal - the key
    /// property for streaming / real-time block processing.
    ///
    /// This is an alias of [`Self::process_samples_in_place`] with a name that
    /// signals the design-once / stream-many usage.
    ///
    /// # Arguments
    /// - `block` - Mutable slice; each element is replaced with its filtered value.
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

    /// Zero-phase forward-backward filtering of a whole signal, in place.
    ///
    /// Equivalent to `scipy.signal.sosfiltfilt(sos, x)` (odd padding,
    /// steady-state initial conditions). The magnitude response is squared and
    /// the phase response cancels. The filter's own streaming state is neither
    /// used nor modified.
    ///
    /// Signals no longer than SciPy's default pad length are padded with
    /// `len - 1` samples instead of raising an error.
    ///
    /// # Arguments
    /// - `signal` - The whole signal; replaced by the filtered result.
    ///
    /// # Examples
    /// ```
    /// use audio_samples::operations::iir_filtering::SosFilter;
    /// use audio_samples::operations::types::IirFilterDesign;
    /// use std::num::NonZeroUsize;
    ///
    /// let design = IirFilterDesign::butterworth_lowpass(NonZeroUsize::new(4).unwrap(), 1000.0);
    /// let sos = SosFilter::from_design(&design, 48000.0).unwrap();
    /// let mut x = vec![1.0; 256];
    /// sos.filtfilt_in_place(&mut x);
    /// // A constant passes a unity-DC-gain low-pass untouched.
    /// assert!(x.iter().all(|v| (v - 1.0).abs() < 1e-9));
    /// ```
    pub fn filtfilt_in_place(&self, signal: &mut [f64]) {
        sosfiltfilt_slice(&self.sos_coefficients(), signal);
    }

    /// Compute the frequency response at the given frequencies.
    ///
    /// Evaluates the combined transfer function of all cascaded sections.
    /// The magnitude response is the product of all section magnitudes;
    /// the phase response is the sum of all section phases (so it is not
    /// wrapped to `(-π, π]`).
    ///
    /// # Arguments
    /// - `frequencies` - Frequencies in Hz at which to evaluate the response.
    /// - `sample_rate` - Sample rate of the signal in Hz.
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

impl IirFilterDesign {
    /// Design this filter as a stateful second-order-sections cascade.
    ///
    /// Convenience wrapper around [`SosFilter::from_design`]; designs the
    /// filter once for the given `sample_rate` so the returned [`SosFilter`]
    /// can stream consecutive blocks without redesigning.
    ///
    /// # Arguments
    /// - `sample_rate` - Sample rate of the signal in Hz.
    ///
    /// # Returns
    /// A freshly-reset [`SosFilter`] implementing this design.
    ///
    /// # Errors
    /// - [crate::AudioSampleError::Parameter] - if the design is invalid.
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

    /// Design this filter and return its SOS coefficients in SciPy layout.
    ///
    /// Each row is `[b0, b1, b2, a0, a1, a2]` with `a0 == 1`; the array is
    /// what `scipy.signal.iirfilter(..., output='sos', fs=sample_rate)` returns
    /// for the same specification.
    ///
    /// # Errors
    /// - [crate::AudioSampleError::Parameter] - if the design is invalid.
    ///
    /// # Examples
    /// ```
    /// use audio_samples::operations::types::IirFilterDesign;
    /// use std::num::NonZeroUsize;
    ///
    /// // Same as scipy.signal.butter(3, 1000, fs=48000, output='sos'):
    /// // a third-order filter has one second-order and one first-order section.
    /// let rows = IirFilterDesign::butterworth_lowpass(NonZeroUsize::new(3).unwrap(), 1000.0)
    ///     .sos_coefficients(48000.0)
    ///     .unwrap();
    /// assert_eq!(rows.len(), 2);
    /// ```
    pub fn sos_coefficients(&self, sample_rate: f64) -> AudioSampleResult<Vec<[f64; 6]>> {
        iir_design::design_sos(self, sample_rate)
    }

    /// Evaluate the frequency response of this design.
    ///
    /// # Arguments
    /// - `frequencies` - Frequencies in Hz.
    /// - `sample_rate` - Sample rate in Hz.
    ///
    /// # Returns
    /// `(magnitudes, phases)`: linear magnitude and phase in radians wrapped
    /// to `(-π, π]`, one entry per frequency (SciPy `sosfreqz`).
    ///
    /// # Errors
    /// - [crate::AudioSampleError::Parameter] - if the design is invalid.
    ///
    /// # Examples
    /// ```
    /// use audio_samples::operations::types::IirFilterDesign;
    /// use std::num::NonZeroUsize;
    ///
    /// let design = IirFilterDesign::butterworth_lowpass(NonZeroUsize::new(4).unwrap(), 1000.0);
    /// let (mag, _phase) = design.frequency_response(&[0.0, 1000.0], 48000.0).unwrap();
    /// assert!((mag[0] - 1.0).abs() < 1e-12);
    /// // Butterworth: -3.01 dB at the cutoff.
    /// assert!((mag[1] - std::f64::consts::FRAC_1_SQRT_2).abs() < 1e-12);
    /// ```
    pub fn frequency_response(
        &self,
        frequencies: &[f64],
        sample_rate: f64,
    ) -> AudioSampleResult<(Vec<f64>, Vec<f64>)> {
        let sos = iir_design::design_sos(self, sample_rate)?;
        let h = iir_design::sosfreqz(&sos, frequencies, sample_rate);
        Ok((
            h.iter().map(|c| c.norm()).collect(),
            h.iter().map(|c| c.arg()).collect(),
        ))
    }
}

// ============================================================================
// AudioSamples integration
// ============================================================================

/// Apply `f` to every channel of `audio` as a contiguous `f64` buffer, writing
/// the result back (converted, and clamped for integer types).
fn for_each_channel_f64<T, F>(audio: &mut AudioSamples<'_, T>, mut f: F)
where
    T: StandardSample,
    F: FnMut(&mut [f64]),
{
    let mut buf: Vec<f64> = Vec::new();
    match audio.data_mut() {
        AudioData::Mono(samples) => {
            buf.clear();
            buf.extend(samples.iter().map(|&s| -> f64 { s.convert_to() }));
            f(&mut buf);
            for (dst, &v) in samples.iter_mut().zip(buf.iter()) {
                *dst = v.convert_to();
            }
        }
        AudioData::Multi(data) => {
            for ch_idx in 0..data.dim().0.get() {
                let mut channel = data.index_axis_mut(Axis(0), ch_idx);
                buf.clear();
                buf.extend(channel.iter().map(|&s| -> f64 { s.convert_to() }));
                f(&mut buf);
                for (dst, &v) in channel.iter_mut().zip(buf.iter()) {
                    *dst = v.convert_to();
                }
            }
        }
    }
}

/// Filter every channel of `audio` with `sos` (zero initial state per channel).
pub(crate) fn apply_sos_to_audio<T: StandardSample>(audio: &mut AudioSamples<'_, T>, sos: &[Sos]) {
    let template = biquads(sos);
    for_each_channel_f64(audio, |buf| {
        let mut bqs = template.clone();
        sosfilt_slice(&mut bqs, buf);
    });
}

impl<T> AudioIirFiltering for AudioSamples<'_, T>
where
    T: StandardSample,
{
    /// Apply an IIR filter using the specified design parameters.
    ///
    /// Designs a filter from `design` (using the audio's own sample
    /// rate), then applies it to every channel independently, each starting
    /// from zero state. Equivalent to `scipy.signal.sosfilt(sos, x)` with the
    /// SOS coefficients SciPy would design for the same specification.
    ///
    /// # Arguments
    /// - `design` - Filter specification (type, order, frequencies, ripple).
    ///
    /// # Returns
    /// `Ok(())` on success.
    ///
    /// # Errors
    /// - [crate::AudioSampleError::Parameter] - if any frequency in `design`
    ///   is out of the valid range (0, Nyquist), a required ripple /
    ///   attenuation is missing, or the order exceeds the supported maximum.
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
        let sos = iir_design::design_sos(design, self.sample_rate_hz())?;
        apply_sos_to_audio(self, &sos);
        Ok(())
    }

    /// Apply zero-phase (forward-backward) IIR filtering in place.
    ///
    /// See [`AudioIirFiltering::filtfilt_in_place`] for the full description.
    #[inline]
    fn filtfilt_in_place(&mut self, design: &IirFilterDesign) -> AudioSampleResult<()> {
        let sos = iir_design::design_sos(design, self.sample_rate_hz())?;
        for_each_channel_f64(self, |buf| sosfiltfilt_slice(&sos, buf));
        Ok(())
    }

    /// Apply a Butterworth low-pass filter.
    ///
    /// Convenience wrapper that constructs an [`IirFilterDesign`] and
    /// delegates to [`Self::apply_iir_filter_in_place`]. Equivalent to
    /// `scipy.signal.sosfilt(scipy.signal.butter(order, cutoff, 'lowpass',
    /// fs=fs, output='sos'), x)`.
    ///
    /// # Arguments
    /// - `order` - Filter order (number of poles).
    /// - `cutoff_frequency` - -3 dB frequency in Hz; must be in (0, Nyquist).
    ///
    /// # Returns
    /// `Ok(())` on success.
    ///
    /// # Errors
    /// - [crate::AudioSampleError::Parameter] - if `cutoff_frequency` is
    ///   outside the valid range, or if `order` exceeds the supported maximum.
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
    /// delegates to [`Self::apply_iir_filter_in_place`]. Equivalent to
    /// `scipy.signal.sosfilt(scipy.signal.butter(order, cutoff, 'highpass',
    /// fs=fs, output='sos'), x)`.
    ///
    /// # Arguments
    /// - `order` - Filter order (number of poles).
    /// - `cutoff_frequency` - -3 dB frequency in Hz; must be in (0, Nyquist).
    ///
    /// # Returns
    /// `Ok(())` on success.
    ///
    /// # Errors
    /// - [crate::AudioSampleError::Parameter] - if `cutoff_frequency` is
    ///   outside the valid range, or if `order` exceeds the supported maximum.
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
    /// delegates to [`Self::apply_iir_filter_in_place`]. The band-pass is
    /// derived from an `order`-pole low-pass prototype by the analog
    /// low-pass→band-pass transform, so the digital filter has `2 × order`
    /// poles. Equivalent to `scipy.signal.butter(order, [low, high],
    /// 'bandpass', fs=fs, output='sos')`.
    ///
    /// # Arguments
    /// - `order` - Order of the low-pass prototype.
    /// - `low_frequency` - Lower -3 dB edge in Hz; must be > 0.
    /// - `high_frequency` - Upper -3 dB edge in Hz; must be < Nyquist
    ///   and > `low_frequency`.
    ///
    /// # Returns
    /// `Ok(())` on success.
    ///
    /// # Errors
    /// - [crate::AudioSampleError::Parameter] - if the frequency range is
    ///   invalid, or if `order` exceeds the supported maximum.
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
    /// assert!(audio.butterworth_bandpass_in_place(NonZeroUsize::new(2).unwrap(), 100.0, 5000.0).is_ok());
    ///
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

    /// Apply a Chebyshev Type I low-pass or high-pass filter.
    ///
    /// Chebyshev Type I filters offer sharper roll-off than Butterworth
    /// filters of the same order at the cost of passband ripple. The passband
    /// ripples between 0 dB and `-passband_ripple` dB; at
    /// `cutoff_frequency` the response is exactly `-passband_ripple` dB.
    /// Equivalent to `scipy.signal.cheby1(order, passband_ripple, cutoff,
    /// response, fs=fs, output='sos')`.
    ///
    /// For band-pass / band-stop use [`IirFilterDesign::chebyshev_i_band`]
    /// with [`Self::apply_iir_filter_in_place`].
    ///
    /// # Arguments
    /// - `order` - Filter order (number of poles).
    /// - `cutoff_frequency` - Passband edge in Hz; must be in (0, Nyquist).
    /// - `passband_ripple` - Maximum passband ripple in dB; must be positive.
    ///   Typical values: 0.1-1 dB.
    /// - `response` - [`FilterResponse::LowPass`] or [`FilterResponse::HighPass`].
    ///
    /// # Returns
    /// `Ok(())` on success.
    ///
    /// # Errors
    /// - [crate::AudioSampleError::Parameter] - if `cutoff_frequency` is out of
    ///   range, `passband_ripple` is not positive, the order exceeds the
    ///   supported maximum, or `response` is a band response (which needs two
    ///   edge frequencies).
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
    /// Deprecated placeholder: audio samples do not carry a filter, so this
    /// always returns a flat magnitude-1, phase-0 response. Use
    /// [`IirFilterDesign::frequency_response`] or
    /// [`SosFilter::frequency_response`] instead.
    ///
    /// # Arguments
    /// - `frequencies` - Frequencies in Hz at which to compute the response.
    ///
    /// # Returns
    /// `Ok((magnitudes, phases))` where both vectors have the same length as
    /// `frequencies`; `magnitudes` is all 1.0 and `phases` is all 0.0.
    #[inline]
    #[allow(deprecated)]
    fn frequency_response(&self, frequencies: &[f64]) -> AudioSampleResult<(Vec<f64>, Vec<f64>)> {
        Ok((vec![1.0; frequencies.len()], vec![0.0; frequencies.len()]))
    }
}

#[cfg(test)]
mod tests {
    use crate::operations::types::IirFilterType;

    // ---- test helpers bridging the old prototype-level API -----------------
    use crate::operations::iir_design::{self, Zpk};
    use crate::operations::types::BesselNorm;

    struct Proto {
        ty: IirFilterType,
        order: usize,
        rp: Option<f64>,
        rs: Option<f64>,
        poles: Vec<Complex<f64>>,
        zeros: Vec<Complex<f64>>,
        gain: f64,
    }

    fn mk(ty: IirFilterType, order: usize, rp: Option<f64>, rs: Option<f64>, z: Zpk) -> Proto {
        Proto {
            ty,
            order,
            rp,
            rs,
            poles: z.poles,
            zeros: z.zeros,
            gain: z.gain,
        }
    }
    fn butterworth_prototype(n: usize) -> Proto {
        mk(
            IirFilterType::Butterworth,
            n,
            None,
            None,
            iir_design::buttap(n),
        )
    }
    fn chebyshev1_prototype(n: usize, rp: f64) -> Proto {
        mk(
            IirFilterType::ChebyshevI,
            n,
            Some(rp),
            None,
            iir_design::cheb1ap(n, rp),
        )
    }
    fn chebyshev2_prototype(n: usize, rs: f64) -> Proto {
        mk(
            IirFilterType::ChebyshevII,
            n,
            None,
            Some(rs),
            iir_design::cheb2ap(n, rs),
        )
    }
    fn bessel_prototype(n: usize) -> Proto {
        mk(
            IirFilterType::Bessel,
            n,
            None,
            None,
            iir_design::besselap(n, BesselNorm::Magnitude),
        )
    }
    fn elliptic_prototype(n: usize, rp: f64, rs: f64) -> Proto {
        mk(
            IirFilterType::Elliptic,
            n,
            Some(rp),
            Some(rs),
            iir_design::ellipap(n, rp, rs).unwrap(),
        )
    }
    #[allow(dead_code)]
    enum GainRef {
        Prototype,
    }
    fn design_from_prototype_with_gain(
        p: &Proto,
        response: FilterResponse,
        fs: f64,
        cutoff: Option<f64>,
        low: Option<f64>,
        high: Option<f64>,
        _gain: Option<GainRef>,
    ) -> SosFilter {
        design_from_prototype(p, response, fs, cutoff, low, high)
    }
    fn design_from_prototype(
        p: &Proto,
        response: FilterResponse,
        fs: f64,
        cutoff: Option<f64>,
        low: Option<f64>,
        high: Option<f64>,
    ) -> SosFilter {
        let order = NonZeroUsize::new(p.order).unwrap();
        let mut d = IirFilterDesign::butterworth_lowpass(order, cutoff.unwrap_or(1.0));
        d.filter_type = p.ty;
        d.response = response;
        d.cutoff_frequency = cutoff;
        d.low_frequency = low;
        d.high_frequency = high;
        d.passband_ripple = p.rp;
        d.stopband_attenuation = p.rs;
        SosFilter::from_design(&d, fs).unwrap()
    }
    fn allpole_mag(poles: &[Complex<f64>], gain: f64, omega: f64) -> f64 {
        let s = Complex::new(0.0, omega);
        poles
            .iter()
            .fold(Complex::new(gain, 0.0), |acc, &p| acc / (s - p))
            .norm()
    }
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
        let result =
            audio.butterworth_bandpass_in_place(NonZeroUsize::new(2).unwrap(), 500.0, 2000.0);
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

        let mut audio = AudioSamples::new_multi_channel(stereo_data, sample_rate!(44100)).unwrap();

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
                .butterworth_bandpass_in_place(
                    NonZeroUsize::new(2).unwrap(),
                    1000.0,
                    sample_rate / 2.0
                )
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
        let result =
            audio.butterworth_bandpass_in_place(NonZeroUsize::new(10).unwrap(), 800.0, 1200.0);
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
            10.0, // Large but valid (SciPy accepts any positive ripple)
            FilterResponse::LowPass,
        );
        assert!(result.is_ok(), "Large ripple is valid: {result:?}");

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

        let mut audio = AudioSamples::new_multi_channel(stereo_data, sample_rate!(44100)).unwrap();

        // Apply 6th-order filter to stereo
        let result = audio.butterworth_lowpass_in_place(NonZeroUsize::new(6).unwrap(), 2000.0);
        assert!(result.is_ok(), "SOS should work with multi-channel audio");
    }

    #[test]
    fn test_order_limit() {
        // Test that order > MAX_IIR_ORDER is rejected
        let samples = NonEmptyVec::new(vec![1.0f32; 100]).unwrap();
        let mut audio: AudioSamples<'_, f32> =
            AudioSamples::from_mono_vec(samples, sample_rate!(44100));

        let result = audio.butterworth_lowpass_in_place(
            NonZeroUsize::new(iir_design::MAX_IIR_ORDER + 1).unwrap(),
            1000.0,
        );
        assert!(result.is_err(), "Should reject order > MAX_IIR_ORDER");
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
        let samples =
            NonEmptyVec::new(vec![1.0f32, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0]).unwrap();
        let original: AudioSamples<'_, f32> =
            AudioSamples::from_mono_vec(samples, sample_rate!(44100));

        let order = NonZeroUsize::new(2).unwrap();

        // Non-mutating: returns a new copy, leaves `original` unchanged.
        let filtered = original.butterworth_lowpass(order, 1000.0).unwrap();

        // In-place on a clone.
        let mut in_place = original.clone();
        in_place
            .butterworth_lowpass_in_place(order, 1000.0)
            .unwrap();

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
        let proto = chebyshev2_prototype(4, 40.0);
        let sos = design_from_prototype(
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
        let proto = chebyshev2_prototype(3, 40.0);
        let sos = design_from_prototype(
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
        let proto = chebyshev2_prototype(4, 40.0);
        let sos = design_from_prototype(
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
        let proto = butterworth_prototype(4);
        let sos = design_from_prototype(
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
        let proto = butterworth_prototype(4);
        let sos = design_from_prototype(
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
        assert!(
            mag_db(&sos, 50.0).abs() < 1.0,
            "50Hz: {}",
            mag_db(&sos, 50.0)
        );
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
        let proto = chebyshev1_prototype(4, 0.5);
        let sos = design_from_prototype(
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
        let proto = chebyshev1_prototype(4, 0.5);
        let sos = design_from_prototype(
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
        let proto = chebyshev2_prototype(4, 40.0);
        let sos = design_from_prototype(
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
        assert!(
            mag_db(&sos, 50.0).abs() < 1.0,
            "50Hz: {}",
            mag_db(&sos, 50.0)
        );
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
        let mut bs =
            IirFilterDesign::butterworth_bandpass(NonZeroUsize::new(4).unwrap(), 500.0, 2000.0);
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
    /// prototype (-3 dB at ω = 1) to high precision for orders 2,4,6,8.
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
        let proto = bessel_prototype(4);
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

        // Order 2 prototype magnitude must be exactly -3 dB at ω = 1.
        let proto2 = bessel_prototype(2);
        let mag1 = allpole_mag(&proto2.poles, proto2.gain, 1.0);
        assert!(
            (db(mag1) - (-3.0103)).abs() < 1e-3,
            "Bessel -3 dB normalization off: {} dB",
            db(mag1)
        );
    }

    #[test]
    fn test_bessel_lowpass_response() {
        // order 4, cutoff 1000 Hz. SciPy: 0 Hz 0 dB, 100 -0.028, 500 -0.703,
        // 1000 -3.01, 2000 -13.53, 4000 -35.30. Monotonic, no ripple.
        let proto = bessel_prototype(4);
        let sos = design_from_prototype(
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
            "cutoff -3 dB: {}",
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
        let proto = bessel_prototype(4);
        let sos = design_from_prototype(
            &proto,
            FilterResponse::LowPass,
            FS,
            Some(1000.0),
            None,
            None,
        );

        // group delay τ(f) = -dφ/dω, ω = 2πf/FS. Sample finely in passband.
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
        // order 4, cutoff 1000 Hz. SciPy: Nyquist 0 dB, 5000 -0.10,
        // 2000 -0.70, 1000 -3.01, 500 -13.44, 100 -65.7.
        let proto = bessel_prototype(4);
        let sos = design_from_prototype(
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
        // 100 -51 dB, 8000 -47 dB, 250 -19.5 dB.
        let proto = bessel_prototype(4);
        let sos = design_from_prototype(
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
        let proto = bessel_prototype(4);
        let sos = design_from_prototype(
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
        assert!(
            mag_db(&sos, 50.0).abs() < 0.5,
            "50Hz: {}",
            mag_db(&sos, 50.0)
        );
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
        let design = IirFilterDesign::bessel(
            FilterResponse::LowPass,
            NonZeroUsize::new(4).unwrap(),
            1000.0,
        );
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

        let proto = elliptic_prototype(4, 1.0, 40.0);
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
        let proto3 = elliptic_prototype(3, 0.5, 60.0);
        assert_eq!(proto3.poles.len(), 3, "N=3 has 3 poles");
        assert_eq!(proto3.zeros.len(), 2, "N=3 drops one zero to infinity");
        assert!(
            proto3.poles.iter().any(|p| p.im.abs() < 1e-9),
            "odd order should have a real pole"
        );
    }

    #[test]
    fn test_elliptic_lowpass_response() {
        // order 4, cutoff 2000 Hz, rp=1, rs=40. SciPy: DC -1.0, 500 -0.435,
        // 1000 -0.085, 1500 -0.998, 1900 -0.013, 2000 -1.0, 2500 -18.82,
        // 4000 -40.03, 8000 -49.2.
        let proto = elliptic_prototype(4, 1.0, 40.0);
        let sos = design_from_prototype_with_gain(
            &proto,
            FilterResponse::LowPass,
            FS,
            Some(2000.0),
            None,
            None,
            Some(GainRef::Prototype),
        );
        // Passband ripple stays within [-1, 0] dB (allow tiny slack).
        for f in [0.0, 500.0, 1000.0, 1500.0, 1900.0, 2000.0] {
            let g = mag_db(&sos, f);
            assert!(
                (-1.0 - 0.1..=0.05).contains(&g),
                "passband {f}Hz gain {g} outside ripple bound"
            );
        }
        // Equiripple: the passband touches the -1 dB bound more than once.
        let touches = [0.0_f64, 1500.0, 2000.0]
            .iter()
            .filter(|&&f| (mag_db(&sos, f) - (-1.0)).abs() < 0.05)
            .count();
        assert!(
            touches >= 2,
            "expected multiple -1 dB touches, got {touches}"
        );
        // Equiripple peaks (touch 0 dB) between the -1 dB dips.
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
        // order 4, cutoff 2000 Hz, rp=1, rs=40. SciPy: Nyquist -1.0,
        // 4000 -0.066, 2000 -1.0, 1500 -24.6, 1000 -40.0, 500 -53.6.
        let proto = elliptic_prototype(4, 1.0, 40.0);
        let sos = design_from_prototype_with_gain(
            &proto,
            FilterResponse::HighPass,
            FS,
            Some(2000.0),
            None,
            None,
            Some(GainRef::Prototype),
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
        // within ripple bound; 100 -43, 8000 -44, 250 -43 dB.
        let proto = elliptic_prototype(4, 1.0, 40.0);
        let sos = design_from_prototype_with_gain(
            &proto,
            FilterResponse::BandPass,
            FS,
            None,
            Some(500.0),
            Some(2000.0),
            Some(GainRef::Prototype),
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
        let proto = elliptic_prototype(4, 1.0, 40.0);
        let sos = design_from_prototype_with_gain(
            &proto,
            FilterResponse::BandStop,
            FS,
            None,
            Some(500.0),
            Some(2000.0),
            Some(GainRef::Prototype),
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
        let design = IirFilterDesign::butterworth_lowpass(NonZeroUsize::new(6).unwrap(), 1500.0);

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
        let design = IirFilterDesign::butterworth_lowpass(NonZeroUsize::new(4).unwrap(), 2000.0);

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
        let design = IirFilterDesign::butterworth_lowpass(NonZeroUsize::new(4).unwrap(), 2000.0);
        let sos = SosFilter::from_design(&design, FS).unwrap();

        // Pass a pure tone through filtfilt and measure the steady-state
        // amplitude ratio; compare to the squared single-pass |H|.
        for &freq in &[500.0f64, 6000.0] {
            let n = 8192usize;
            let sig: Vec<f64> = (0..n)
                .map(|i| (2.0 * PI * freq * (i as f64) / FS).sin())
                .collect();

            let samples = NonEmptyVec::new(sig.iter().map(|&v| v as f32).collect()).unwrap();
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
            let amp_out = out[lo..hi]
                .iter()
                .cloned()
                .fold(0.0f64, |a, b| a.max(b.abs()));
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
