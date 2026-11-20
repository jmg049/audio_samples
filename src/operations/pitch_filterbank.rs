//! Multirate semitone filterbank and the IIR time-frequency representation.
//!
//! # What
//!
//! - [`mr_frequencies`]: the centre frequency and sample rate of each semitone from MIDI
//!   24 to 108.
//! - [`semitone_filterbank`]: one band-pass filter per semitone, of minimum order for a
//!   specified ripple and attenuation.
//! - [`iirt`] and [`AudioPitchFilterbank::iirt`]: the energy of each band per frame.
//!
//! # Why
//!
//! A constant-Q analysis with FFTs trades time resolution at low frequencies for
//! frequency resolution. Running narrow IIR filters at a low sample rate gives a pitch
//! spectrogram with the same window length for every band, which suits chord recognition
//! and music synchronisation.
//!
//! # How
//!
//! Low bands run on a version of the signal resampled to a low rate, where the filter
//! poles are far enough from the unit circle to be stable. Each band is filtered forwards
//! and backwards, and its squared output is summed over each frame and scaled back to the
//! original rate. Filters are second-order sections unless asked for otherwise. Filtering
//! runs in `f64`; the output has the audio's float type.
//!
//! ```
//! use audio_samples::operations::pitch_filterbank::{AudioPitchFilterbank, IirtConfig};
//! use audio_samples::sample_rate;
//! use audio_samples::utils::generation::sine_wave;
//! use std::time::Duration;
//!
//! let audio = sine_wave::<f32>(440.0, Duration::from_millis(500), sample_rate!(22050), 0.5);
//! let power = audio.iirt(&IirtConfig::new()).unwrap();
//! assert_eq!(power.dim().1, 85);
//! // MIDI 69 is band 45.
//! let band = (0..85)
//!     .max_by(|&a, &b| power[[0, a, 5]].partial_cmp(&power[[0, b, 5]]).unwrap())
//!     .unwrap();
//! assert_eq!(band, 45);
//! ```
//!
//! Reference: M. Müller, *Information Retrieval for Music and Motion*, Springer, 2007,
//! chapter 3.

use std::num::NonZeroUsize;

use ndarray::{Array2, Array3, ArrayView2, Axis};

use crate::audio_float::finite_float_rows;
use crate::operations::iir_design::{
    Wn, bandpass_order, buttap, cheb1ap, cheb2ap, ellipap, iirfilter_sos,
};
use crate::operations::iir_filtering::sosfiltfilt_slice;
use crate::operations::types::{FilterResponse, IirFilterType, ResamplingQuality};
use crate::utils::array_ops::{PadMode, pad_center};
use crate::utils::audio_math::midi_to_hz;
use crate::{AudioFloat, AudioSampleResult, AudioSamples, ParameterError, StandardSample};

fn invalid<T>(parameter: &str, reason: impl Into<String>) -> AudioSampleResult<T> {
    Err(ParameterError::invalid_value(parameter, reason.into()).into())
}

fn of<F: AudioFloat>(v: f64) -> F {
    <F as num_traits::NumCast>::from(v).unwrap_or_else(F::nan)
}

/// Centre frequencies and sample rates of the semitones from MIDI 24 to 108.
///
/// MIDI 24 to 59 run at 882 Hz, 60 to 93 at 4410 Hz and 94 to 108 at 22050 Hz.
///
/// # Arguments
/// - `tuning`: deviation from A440 in fractions of a semitone.
///
/// # Returns
/// Two arrays of length 85: the centre frequencies and the sample rates, in Hz.
///
/// # Examples
/// ```
/// use audio_samples::operations::pitch_filterbank::mr_frequencies;
///
/// let (freqs, rates) = mr_frequencies(0.0);
/// assert_eq!(freqs.len(), 85);
/// assert!((freqs[45] - 440.0).abs() < 1e-9);
/// assert_eq!(rates[0], 882.0);
/// assert_eq!(rates[84], 22050.0);
/// ```
#[must_use]
pub fn mr_frequencies(tuning: f64) -> ([f64; 85], [f64; 85]) {
    let mut freqs = [0.0; 85];
    let mut rates = [0.0; 85];
    for i in 0..85 {
        freqs[i] = midi_to_hz(24.0 + tuning + i as f64);
        rates[i] = match i {
            0..=35 => 882.0,
            36..=69 => 4410.0,
            _ => 22050.0,
        };
    }
    (freqs, rates)
}

/// Layout of a designed filter.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FilterLayout {
    /// Second-order sections. Numerically robust.
    #[default]
    Sos,
    /// Numerator and denominator polynomials. High orders lose precision.
    Ba,
}

/// One band's filter.
#[derive(Debug, Clone, PartialEq)]
pub enum BandFilter {
    /// Sections `[b0, b1, b2, a0, a1, a2]`.
    Sos(Vec<[f64; 6]>),
    /// Numerator and denominator polynomials, highest power first.
    Ba {
        /// Numerator.
        b: Vec<f64>,
        /// Denominator.
        a: Vec<f64>,
    },
}

/// Parameters of each band of [`semitone_filterbank`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MultirateFilterConfig {
    /// Centre frequency over bandwidth. The stop band is twice as wide as the pass band.
    /// Default 25.
    pub q: f64,
    /// Largest attenuation in the pass band, in dB. Default 1.
    pub passband_ripple: f64,
    /// Smallest attenuation in the stop band, in dB. Default 50.
    pub stopband_attenuation: f64,
    /// The filter family. Default elliptic.
    pub filter_type: IirFilterType,
}

impl MultirateFilterConfig {
    /// Q of 25, 1 dB ripple, 50 dB attenuation, elliptic.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            q: 25.0,
            passband_ripple: 1.0,
            stopband_attenuation: 50.0,
            filter_type: IirFilterType::Elliptic,
        }
    }

    /// Settings matching librosa 1.0's defaults; currently the same as [`new`](Self::new).
    #[must_use]
    pub const fn librosa() -> Self {
        Self::new()
    }

    /// Checks the configuration.
    ///
    /// # Errors
    /// [`ParameterError`] if a value is not positive and finite, or the attenuation does
    /// not exceed the ripple.
    pub fn validate(&self) -> AudioSampleResult<()> {
        for (name, v) in [
            ("q", self.q),
            ("passband_ripple", self.passband_ripple),
            ("stopband_attenuation", self.stopband_attenuation),
        ] {
            if !(v.is_finite() && v > 0.0) {
                return invalid(name, "must be positive and finite");
            }
        }
        if self.stopband_attenuation <= self.passband_ripple {
            return invalid("stopband_attenuation", "must exceed the ripple");
        }
        Ok(())
    }
}

impl Default for MultirateFilterConfig {
    fn default() -> Self {
        Self::new()
    }
}

fn design_band(
    center: f64,
    rate: f64,
    config: &MultirateFilterConfig,
) -> AudioSampleResult<Vec<[f64; 6]>> {
    let bw = center / config.q;
    let (order, low, high) = bandpass_order(
        config.filter_type,
        (center - 0.5 * bw, center + 0.5 * bw),
        (center - bw, center + bw),
        config.passband_ripple,
        config.stopband_attenuation,
        rate,
    )?;
    if order == 0 || order > 64 {
        return invalid("q", format!("the specification needs order {order}"));
    }
    let proto = match config.filter_type {
        IirFilterType::Butterworth => buttap(order),
        IirFilterType::ChebyshevI => cheb1ap(order, config.passband_ripple),
        IirFilterType::ChebyshevII => cheb2ap(order, config.stopband_attenuation),
        _ => ellipap(order, config.passband_ripple, config.stopband_attenuation)?,
    };
    iirfilter_sos(&proto, FilterResponse::BandPass, Wn::Band(low, high), rate)
}

fn poly_multiply(a: &[f64], b: &[f64]) -> Vec<f64> {
    let mut out = vec![0.0; a.len() + b.len() - 1];
    for (i, x) in a.iter().enumerate() {
        for (j, y) in b.iter().enumerate() {
            out[i + j] += x * y;
        }
    }
    out
}

/// Designs one band-pass filter per centre frequency.
///
/// # Arguments
/// - `center_freqs`: centre frequencies in Hz.
/// - `sample_rates`: the sample rate of each band in Hz; same length.
/// - `config`: bandwidth, ripple, attenuation and family.
/// - `layout`: the layout of the result.
///
/// # Returns
/// One filter per band.
///
/// # Errors
/// [`ParameterError`] if the lengths differ, the configuration is invalid or a band's
/// edges lie outside `(0, rate / 2)`.
///
/// # Examples
/// ```
/// use audio_samples::operations::pitch_filterbank::{
///     BandFilter, FilterLayout, MultirateFilterConfig, mr_frequencies, semitone_filterbank,
/// };
///
/// let (f, r) = mr_frequencies(0.0);
/// let bank = semitone_filterbank(&f, &r, &MultirateFilterConfig::new(), FilterLayout::Sos).unwrap();
/// assert_eq!(bank.len(), 85);
/// assert!(matches!(bank[0], BandFilter::Sos(_)));
/// ```
pub fn semitone_filterbank(
    center_freqs: &[f64],
    sample_rates: &[f64],
    config: &MultirateFilterConfig,
    layout: FilterLayout,
) -> AudioSampleResult<Vec<BandFilter>> {
    config.validate()?;
    if center_freqs.len() != sample_rates.len() {
        return invalid("center_freqs", "must have one sample rate per band");
    }
    center_freqs
        .iter()
        .zip(sample_rates)
        .map(|(&c, &r)| {
            let sos = design_band(c, r, config)?;
            Ok(match layout {
                FilterLayout::Sos => BandFilter::Sos(sos),
                FilterLayout::Ba => {
                    let (mut b, mut a) = (vec![1.0], vec![1.0]);
                    for s in &sos {
                        b = poly_multiply(&b, &s[0..3]);
                        a = poly_multiply(&a, &s[3..6]);
                    }
                    BandFilter::Ba { b, a }
                }
            })
        })
        .collect()
}

/// Parameters of [`iirt`].
#[derive(Debug, Clone, PartialEq)]
pub struct IirtConfig {
    /// Frame length in samples of the original rate. Default 2048.
    pub win_length: NonZeroUsize,
    /// Samples between frames; `None` is `win_length / 4`. Default `None`.
    pub hop_length: Option<NonZeroUsize>,
    /// Pad so that frame `t` is centred at `t * hop_length`. Default true.
    pub center: bool,
    /// Deviation from A440 in fractions of a semitone. Default 0.
    pub tuning: f64,
    /// How centring pads the signal. Default zeros.
    pub pad_mode: PadMode<f64>,
    /// The filter layout. Default second-order sections.
    pub layout: FilterLayout,
    /// The resampler quality. Default high.
    pub quality: ResamplingQuality,
    /// Bandwidth, ripple, attenuation and family.
    pub filter: MultirateFilterConfig,
    /// Centre frequencies and sample rates in place of [`mr_frequencies`]. Default `None`.
    pub bands: Option<(Vec<f64>, Vec<f64>)>,
}

impl IirtConfig {
    /// A frame of 2048 samples, centred with zeros, the default filterbank and
    /// high-quality resampling.
    #[must_use]
    pub fn new() -> Self {
        Self {
            win_length: crate::nzu!(2048),
            hop_length: None,
            center: true,
            tuning: 0.0,
            pad_mode: PadMode::Constant(0.0),
            layout: FilterLayout::Sos,
            quality: ResamplingQuality::High,
            filter: MultirateFilterConfig::new(),
            bands: None,
        }
    }

    /// Settings matching librosa 1.0's defaults; currently the same as [`new`](Self::new).
    #[must_use]
    pub fn librosa() -> Self {
        Self::new()
    }

    /// Sets the frame length.
    #[must_use]
    pub fn with_win_length(mut self, win_length: NonZeroUsize) -> Self {
        self.win_length = win_length;
        self
    }

    /// Sets the hop.
    #[must_use]
    pub fn with_hop_length(mut self, hop_length: Option<NonZeroUsize>) -> Self {
        self.hop_length = hop_length;
        self
    }

    /// Sets the centring.
    #[must_use]
    pub fn with_center(mut self, center: bool) -> Self {
        self.center = center;
        self
    }

    /// Sets the tuning.
    #[must_use]
    pub fn with_tuning(mut self, tuning: f64) -> Self {
        self.tuning = tuning;
        self
    }

    /// Sets the pad mode.
    #[must_use]
    pub fn with_pad_mode(mut self, pad_mode: PadMode<f64>) -> Self {
        self.pad_mode = pad_mode;
        self
    }

    /// Sets the filter layout.
    #[must_use]
    pub fn with_layout(mut self, layout: FilterLayout) -> Self {
        self.layout = layout;
        self
    }

    /// Sets the resampler quality.
    #[must_use]
    pub fn with_quality(mut self, quality: ResamplingQuality) -> Self {
        self.quality = quality;
        self
    }

    /// Sets the band filters' parameters.
    #[must_use]
    pub fn with_filter(mut self, filter: MultirateFilterConfig) -> Self {
        self.filter = filter;
        self
    }

    /// Sets the bands.
    #[must_use]
    pub fn with_bands(mut self, bands: Option<(Vec<f64>, Vec<f64>)>) -> Self {
        self.bands = bands;
        self
    }
}

impl Default for IirtConfig {
    fn default() -> Self {
        Self::new()
    }
}

fn round_half_even(v: f64) -> f64 {
    let r = v.round();
    if (v - v.trunc()).abs() == 0.5 && r % 2.0 != 0.0 {
        r - v.signum()
    } else {
        r
    }
}

/// `n` steps of direct form II transposed, with the state in `z`.
fn lfilter(b: &[f64], a: &[f64], x: &mut [f64], z: &mut [f64]) {
    let order = z.len();
    for v in x.iter_mut() {
        let input = *v;
        let output = b[0] * input + if order > 0 { z[0] } else { 0.0 };
        for i in 0..order {
            let next = if i + 1 < order { z[i + 1] } else { 0.0 };
            z[i] = b[i + 1] * input + next - a[i + 1] * output;
        }
        *v = output;
    }
}

/// The steady state of [`lfilter`] for a unit step: the companion system solved by
/// Gaussian elimination with partial pivoting.
fn lfilter_zi(b: &[f64], a: &[f64]) -> Vec<f64> {
    let n = a.len() - 1;
    let mut m = vec![vec![0.0; n + 1]; n];
    // (I - A) zi = B, where A is the companion matrix of the state update.
    for i in 0..n {
        m[i][i] = 1.0;
        if i + 1 < n {
            m[i][i + 1] = -1.0;
        }
        m[i][0] += a[i + 1];
        m[i][n] = b[i + 1] - a[i + 1] * b[0];
    }
    for p in 0..n {
        let pivot = (p..n)
            .max_by(|&r, &s| m[r][p].abs().total_cmp(&m[s][p].abs()))
            .unwrap_or(p);
        m.swap(p, pivot);
        let d = m[p][p];
        if d == 0.0 {
            continue;
        }
        for r in p + 1..n {
            let f = m[r][p] / d;
            if f != 0.0 {
                let (above, below) = m.split_at_mut(r);
                for (x, y) in below[0][p..=n].iter_mut().zip(&above[p][p..=n]) {
                    *x -= f * y;
                }
            }
        }
    }
    let mut zi = vec![0.0; n];
    for r in (0..n).rev() {
        let mut s = m[r][n];
        for (a, z) in m[r].iter().zip(zi.iter()).skip(r + 1) {
            s -= a * z;
        }
        zi[r] = if m[r][r] == 0.0 { 0.0 } else { s / m[r][r] };
    }
    zi
}

/// Zero-phase filtering with polynomials: odd reflection of `3 * max(len(a), len(b))`
/// samples at each end, and steady-state initial conditions for each pass.
fn filtfilt_ba(b: &[f64], a: &[f64], signal: &mut [f64]) -> AudioSampleResult<()> {
    let len = signal.len();
    let edge = 3 * a.len().max(b.len());
    if len <= edge {
        return invalid(
            "y",
            format!("{len} samples are too few for this band's filter"),
        );
    }
    let norm = a[0];
    let b: Vec<f64> = b.iter().map(|v| v / norm).collect();
    let a: Vec<f64> = a.iter().map(|v| v / norm).collect();
    let (x0, xn) = (signal[0], signal[len - 1]);
    let mut ext = Vec::with_capacity(len + 2 * edge);
    ext.extend((1..=edge).rev().map(|k| 2.0 * x0 - signal[k]));
    ext.extend_from_slice(signal);
    ext.extend((1..=edge).map(|k| 2.0 * xn - signal[len - 1 - k]));
    let zi = lfilter_zi(&b, &a);
    let mut z: Vec<f64> = zi.iter().map(|v| v * ext[0]).collect();
    lfilter(&b, &a, &mut ext, &mut z);
    ext.reverse();
    let mut z: Vec<f64> = zi.iter().map(|v| v * ext[0]).collect();
    lfilter(&b, &a, &mut ext, &mut z);
    ext.reverse();
    signal.copy_from_slice(&ext[edge..edge + len]);
    Ok(())
}

/// The energy of each semitone band per frame, from `(channels, samples)` audio.
///
/// # Arguments
/// - `y`: the signal, finite.
/// - `sample_rate`: its rate in Hz.
/// - `config`: frames, padding and bands.
///
/// # Returns
/// `(channels, bands, frames)`, in the signal's float type.
///
/// # Errors
/// [`ParameterError`] if the signal is too short for one frame or for a band's filter, or
/// the configuration is invalid.
///
/// # Examples
/// See the [module documentation](self).
pub fn iirt<F: AudioFloat>(
    y: ArrayView2<'_, F>,
    sample_rate: u32,
    config: &IirtConfig,
) -> AudioSampleResult<Array3<F>> {
    if y.iter().any(|v| !v.is_finite()) {
        return invalid("y", "must be finite");
    }
    config.filter.validate()?;
    let win = config.win_length.get();
    let hop = config
        .hop_length
        .map_or((win / 4).max(1), NonZeroUsize::get);
    let padded: Array2<F>;
    let y = if config.center {
        let mode = match config.pad_mode {
            PadMode::Constant(v) => PadMode::Constant(of::<F>(v)),
            PadMode::Edge => PadMode::Edge,
            PadMode::Reflect => PadMode::Reflect,
            PadMode::Symmetric => PadMode::Symmetric,
            PadMode::Wrap => PadMode::Wrap,
        };
        padded = pad_center(&y, y.ncols() + 2 * (win / 2), Axis(1), mode)?;
        padded.view()
    } else {
        y
    };
    if y.ncols() < win {
        return invalid(
            "y",
            format!("{} samples are too few for a frame of {win}", y.ncols()),
        );
    }
    let (freqs, rates) = config.bands.clone().unwrap_or_else(|| {
        let (f, r) = mr_frequencies(config.tuning);
        (f.to_vec(), r.to_vec())
    });
    let bank = semitone_filterbank(&freqs, &rates, &config.filter, config.layout)?;
    let n_frames = 1 + (y.ncols() - win) / hop;
    let mut out = Array3::<F>::zeros((y.nrows(), freqs.len(), n_frames));

    let mut distinct: Vec<f64> = rates.clone();
    distinct.sort_by(f64::total_cmp);
    distinct.dedup();
    let mut buffer: Vec<f64> = Vec::new();
    for &rate in &distinct {
        if rate.fract() != 0.0 || rate <= 0.0 || rate > f64::from(u32::MAX) {
            return invalid(
                "sample_rates",
                format!("{rate} is not a positive integer rate"),
            );
        }
        let resampled: Array2<F>;
        let rows = if rate as u32 == sample_rate {
            y.view()
        } else {
            resampled = crate::resampling::resample_float_rows(
                y.view(),
                sample_rate,
                rate as u32,
                config.quality,
                None,
            )?;
            resampled.view()
        };
        let factor = f64::from(sample_rate) / rate;
        let hop_at_rate = hop as f64 / factor;
        let win_at_rate = round_half_even(win as f64 / factor) as usize;
        for (band, (filter, _)) in bank
            .iter()
            .zip(&rates)
            .enumerate()
            .filter(|(_, (_, r))| **r == rate)
        {
            for (channel, row) in rows.rows().into_iter().enumerate() {
                buffer.clear();
                buffer.extend(row.iter().map(|v| v.to_f64().unwrap_or(f64::NAN)));
                match filter {
                    BandFilter::Sos(sos) => sosfiltfilt_slice(sos, &mut buffer),
                    BandFilter::Ba { b, a } => filtfilt_ba(b, a, &mut buffer)?,
                }
                let starts = |len: usize| -> usize {
                    (len as f64 - win_at_rate as f64)
                        .max(0.0)
                        .div_euclid(hop_at_rate) as usize
                        + usize::from(
                            (len as f64 - win_at_rate as f64).rem_euclid(hop_at_rate) > 0.0,
                        )
                };
                if starts(buffer.len()) < n_frames {
                    let min_length = (n_frames as f64 * hop_at_rate).ceil() as usize + win_at_rate;
                    buffer.resize(min_length, 0.0);
                }
                for t in 0..n_frames {
                    let start = round_half_even(t as f64 * hop_at_rate) as usize;
                    let end = (start + win_at_rate).min(buffer.len());
                    let energy: f64 = buffer[start.min(end)..end].iter().map(|v| v * v).sum();
                    out[[channel, band, t]] = of::<F>(factor * energy);
                }
            }
        }
    }
    Ok(out)
}

/// The IIR time-frequency representation of audio.
pub trait AudioPitchFilterbank {
    /// The audio's float type.
    type Float: AudioFloat;

    /// The energy of each semitone band per frame.
    ///
    /// # Arguments
    /// - `config`: frames, padding and bands.
    ///
    /// # Returns
    /// `(channels, bands, frames)`.
    ///
    /// # Errors
    /// As for [`iirt`].
    fn iirt(&self, config: &IirtConfig) -> AudioSampleResult<Array3<Self::Float>>;
}

impl<T: StandardSample> AudioPitchFilterbank for AudioSamples<'_, T> {
    type Float = T::Float;

    fn iirt(&self, config: &IirtConfig) -> AudioSampleResult<Array3<T::Float>> {
        let rows = finite_float_rows(self)?;
        iirt(rows.view(), self.sample_rate().get(), config)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bank_rejects_bad_specifications() {
        let cfg = MultirateFilterConfig {
            q: 0.0,
            ..MultirateFilterConfig::new()
        };
        assert!(semitone_filterbank(&[100.0], &[882.0], &cfg, FilterLayout::Sos).is_err());
        let ok = MultirateFilterConfig::new();
        assert!(semitone_filterbank(&[100.0, 200.0], &[882.0], &ok, FilterLayout::Sos).is_err());
        assert!(semitone_filterbank(&[800.0], &[882.0], &ok, FilterLayout::Sos).is_err());
    }

    #[test]
    fn bands_select_the_right_frequency() {
        let sr = 4410.0;
        let t: Vec<f64> = (0..4410)
            .map(|i| (2.0 * std::f64::consts::PI * 440.0 * f64::from(i) / sr).sin())
            .collect();
        let bank = semitone_filterbank(
            &[440.0, 880.0],
            &[sr, sr],
            &MultirateFilterConfig::new(),
            FilterLayout::Sos,
        )
        .unwrap();
        let energy = |f: &BandFilter| {
            let BandFilter::Sos(sos) = f else {
                unreachable!("asked for sections")
            };
            let mut b = t.clone();
            sosfiltfilt_slice(sos, &mut b);
            b.iter().map(|v| v * v).sum::<f64>()
        };
        assert!(energy(&bank[0]) > 1000.0 * energy(&bank[1]));
    }

    #[test]
    fn polynomial_form_agrees_with_sections() {
        let sr = 22050.0;
        let mut a: Vec<f64> = (0..3000).map(|i| (f64::from(i) * 0.37).sin()).collect();
        let mut b = a.clone();
        let cfg = MultirateFilterConfig::new();
        let sos = semitone_filterbank(&[5000.0], &[sr], &cfg, FilterLayout::Sos).unwrap();
        let ba = semitone_filterbank(&[5000.0], &[sr], &cfg, FilterLayout::Ba).unwrap();
        let (BandFilter::Sos(s), BandFilter::Ba { b: num, a: den }) = (&sos[0], &ba[0]) else {
            unreachable!("asked for each layout")
        };
        sosfiltfilt_slice(s, &mut a);
        filtfilt_ba(num, den, &mut b).unwrap();
        // The two pad differently, so compare away from the ends.
        for (x, y) in a[500..2500].iter().zip(&b[500..2500]) {
            assert!((x - y).abs() < 1e-4, "{x} {y}");
        }
    }

    #[test]
    fn rejects_short_and_non_finite_signals() {
        let y = Array2::<f64>::zeros((1, 100));
        let cfg = IirtConfig::new().with_center(false);
        assert!(iirt(y.view(), 22050, &cfg).is_err());
        let mut n = Array2::<f64>::zeros((1, 3000));
        n[[0, 5]] = f64::NAN;
        assert!(iirt(n.view(), 22050, &IirtConfig::new()).is_err());
    }
}
