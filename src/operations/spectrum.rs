//! librosa-compatible spectral analysis: STFT / ISTFT, dB scaling, mel
//! spectrograms, MFCCs and frame-wise feature utilities.
//!
//! ## What
//!
//! This module provides the [`AudioSpectrum`] trait and the free functions it is
//! built from. Together they reproduce librosa 1.0's core spectral pipeline:
//!
//! | librosa | here |
//! |---|---|
//! | `librosa.stft` / `librosa.istft` | [`AudioSpectrum::stft_matrix`] / [`AudioSpectrum::istft_matrix`] ([`stft`], [`istft`]) |
//! | `numpy.abs(librosa.stft(y)) ** power` | [`AudioSpectrum::spectrogram_matrix`] |
//! | `librosa.feature.melspectrogram` | [`AudioSpectrum::melspectrogram`] ([`melspectrogram_from_spectrogram`]) |
//! | `librosa.feature.mfcc` | [`AudioSpectrum::mfcc_frames`] ([`mfcc_from_mel_db`]) |
//! | `librosa.power_to_db` / `amplitude_to_db` / `db_to_power` / `db_to_amplitude` | [`power_to_db`], [`amplitude_to_db`], [`db_to_power`], [`db_to_amplitude`] |
//! | `librosa.feature.delta` / `stack_memory` | [`delta`], [`stack_memory`] |
//! | `librosa.util.frame` | [`frame`] |
//!
//! Windows and mel filterbanks live in [`crate::operations::filterbanks`];
//! unit conversions (mel scales, frames/samples/time) in
//! [`crate::utils::audio_math`].
//!
//! ## Why
//!
//! The spectrogram methods of [`AudioTransforms`](crate::AudioTransforms) delegate
//! to the `spectrograms` crate, whose conventions differ from librosa's
//! (symmetric windows, an absolute dB floor, an unnormalised DCT, mono only).
//! Code ported from Python, and results compared with published librosa
//! features, need librosa's exact conventions. Every config struct here
//! defaults to librosa's defaults, so `X::default()` gives librosa-equivalent
//! output (to floating-point round-off).
//!
//! ## How
//!
//! Bring [`AudioSpectrum`] into scope and call it on any [`AudioSamples`].
//! All outputs are `f64` (or `Complex<f64>`) arrays shaped
//! `(channels, n_features, n_frames)`; mono input gives one channel. Input samples
//! are converted to `f64` with the crate's audio-aware conversion.
//!
//! ```rust
//! use audio_samples::operations::spectrum::{AudioSpectrum, MfccConfig, StftConfig};
//! use audio_samples::{AudioSamples, sample_rate, sine_wave};
//! use std::time::Duration;
//!
//! let audio = sine_wave::<f32>(440.0, Duration::from_millis(500), sample_rate!(22050), 0.5);
//!
//! // librosa.stft(y): (1, 1025, 22) complex.
//! let stft = audio.stft_matrix(&StftConfig::default()).unwrap();
//! assert_eq!(stft.dim(), (1, 1025, 22));
//!
//! // librosa.istft(D, length=len(y)) reconstructs the signal.
//! let n = audio.samples_per_channel();
//! let y = AudioSamples::<f32>::istft_matrix(&stft, &StftConfig::default(), sample_rate!(22050), Some(n))
//!     .unwrap();
//! assert_eq!(y.samples_per_channel(), n);
//!
//! // librosa.feature.mfcc(y=y, sr=22050): (1, 20, 22).
//! let mfcc = audio.mfcc_frames(&MfccConfig::default()).unwrap();
//! assert_eq!(mfcc.dim(), (1, 20, 22));
//! ```

use std::cell::RefCell;
use std::num::NonZeroUsize;
use std::sync::Arc;

use ndarray::{
    Array, Array2, Array3, ArrayBase, ArrayView2, ArrayView3, Axis, Data, Dimension, Slice, Zip,
};
use num_complex::Complex;
use realfft::{ComplexToReal, RealFftPlanner, RealToComplex};

use crate::operations::filterbanks::{
    MelFilterbankConfig, Window, mel_filterbank, padded_window, window_sumsquare_from_squared,
};
use crate::repr::SampleRate;
use crate::{
    AudioSampleError, AudioSampleResult, AudioSamples, AudioTypeConversion, ParameterError,
    StandardSample,
};

thread_local! {
    /// Per-thread real-FFT planner. realfft caches plans (twiddles) by length, so
    /// repeated transforms of the same size reuse them; plans are fetched once per
    /// call and every frame is processed with caller-owned scratch buffers.
    static REAL_FFT_PLANNER: RefCell<RealFftPlanner<f64>> = RefCell::new(RealFftPlanner::new());
}

pub(crate) fn forward_plan(n: usize) -> Arc<dyn RealToComplex<f64>> {
    REAL_FFT_PLANNER.with(|p| p.borrow_mut().plan_fft_forward(n))
}

fn inverse_plan(n: usize) -> Arc<dyn ComplexToReal<f64>> {
    REAL_FFT_PLANNER.with(|p| p.borrow_mut().plan_fft_inverse(n))
}

/// Frames per block in [`stft`] (see the comment there).
const STFT_BLOCK: usize = 16;

fn param_err(parameter: &str, reason: impl ToString) -> AudioSampleError {
    AudioSampleError::Parameter(ParameterError::invalid_value(parameter, reason))
}

// =============================================================================
// STFT configuration
// =============================================================================

/// How the signal is extended at both ends when frames are centred
/// (librosa's `pad_mode`, applied with `numpy.pad`).
///
/// # Examples
///
/// ```rust
/// use audio_samples::operations::spectrum::StftPadMode;
///
/// assert_eq!(StftPadMode::default(), StftPadMode::Constant);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum StftPadMode {
    /// Zero padding (`numpy.pad(mode="constant")`). librosa 1.0's default.
    #[default]
    Constant,
    /// Mirror without repeating the edge sample: `d c b | a b c d | c b a`
    /// (`mode="reflect"`, librosa's default before 0.10).
    Reflect,
    /// Repeat the edge sample: `a a a | a b c d | d d d` (`mode="edge"`).
    Edge,
    /// Mirror including the edge sample: `c b a | a b c d | d c b` (`mode="symmetric"`).
    Symmetric,
}

/// Configuration of a short-time Fourier transform, mirroring `librosa.stft`.
///
/// ## Purpose
///
/// Describes how a signal is cut into overlapping, windowed frames before each
/// frame is Fourier transformed, and therefore also how the frames are
/// recombined by the inverse transform.
///
/// ## Intended usage
///
/// Pass to [`AudioSpectrum::stft_matrix`], [`AudioSpectrum::istft_matrix`],
/// [`AudioSpectrum::spectrogram_matrix`], or embed in a
/// [`MelSpectrogramConfig`]. `Default` reproduces librosa's defaults:
/// `n_fft = 2048`, `hop_length = win_length / 4`, `win_length = n_fft`, a
/// periodic Hann window, centred frames and zero padding.
///
/// ## Invariants
///
/// Checked by [`StftConfig::validate`]: `win_length <= n_fft`, the effective hop
/// is at least 1 sample, and the window is valid for `win_length` samples.
///
/// # Examples
///
/// ```rust
/// use audio_samples::nzu;
/// use audio_samples::operations::filterbanks::Window;
/// use audio_samples::operations::spectrum::{StftConfig, StftPadMode};
///
/// let cfg = StftConfig::new()
///     .with_n_fft(nzu!(1024))
///     .with_hop_length(Some(nzu!(256)))
///     .with_window(Window::Hamming)
///     .with_pad_mode(StftPadMode::Reflect);
/// assert_eq!(cfg.effective_win_length(), 1024);
/// assert_eq!(cfg.effective_hop_length().unwrap().get(), 256);
/// assert_eq!(cfg.n_bins(), 513);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct StftConfig {
    /// FFT length in samples (frame length after zero-padding the window). Sets the
    /// frequency resolution: `n_fft / 2 + 1` bins spaced `sample_rate / n_fft` Hz apart.
    /// librosa default: 2048.
    pub n_fft: NonZeroUsize,
    /// Samples between the starts of successive frames. `None` means
    /// `win_length / 4` (librosa's default; 512 for the default `n_fft`). Smaller
    /// hops give more frames (finer time resolution) at a higher cost.
    pub hop_length: Option<NonZeroUsize>,
    /// Length of the window in samples, `<= n_fft`. `None` means `n_fft`. A shorter
    /// window is zero-padded symmetrically to `n_fft` (librosa's `pad_center`),
    /// trading frequency resolution for time resolution.
    pub win_length: Option<NonZeroUsize>,
    /// Analysis window, computed in its periodic form (SciPy `fftbins=True`) as
    /// librosa does. librosa default: Hann.
    pub window: Window,
    /// If `true` (librosa's default) the signal is padded by `n_fft / 2` samples at
    /// both ends so that frame `t` is centred on sample `t * hop_length`. If
    /// `false`, frame `t` starts at sample `t * hop_length` and the signal must be
    /// at least `n_fft` samples long.
    pub center: bool,
    /// How the signal is extended when `center` is `true`. librosa 1.0 default:
    /// [`StftPadMode::Constant`] (zeros).
    pub pad_mode: StftPadMode,
}

impl Default for StftConfig {
    fn default() -> Self {
        Self {
            n_fft: crate::nzu!(2048),
            hop_length: None,
            win_length: None,
            window: Window::Hann,
            center: true,
            pad_mode: StftPadMode::Constant,
        }
    }
}

impl StftConfig {
    /// Returns the librosa-default configuration (same as [`Default`]).
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the FFT length.
    #[must_use]
    pub fn with_n_fft(mut self, n_fft: NonZeroUsize) -> Self {
        self.n_fft = n_fft;
        self
    }

    /// Sets the hop length (`None` = `win_length / 4`).
    #[must_use]
    pub fn with_hop_length(mut self, hop_length: Option<NonZeroUsize>) -> Self {
        self.hop_length = hop_length;
        self
    }

    /// Sets the window length (`None` = `n_fft`).
    #[must_use]
    pub fn with_win_length(mut self, win_length: Option<NonZeroUsize>) -> Self {
        self.win_length = win_length;
        self
    }

    /// Sets the analysis window.
    #[must_use]
    pub fn with_window(mut self, window: Window) -> Self {
        self.window = window;
        self
    }

    /// Sets whether frames are centred.
    #[must_use]
    pub fn with_center(mut self, center: bool) -> Self {
        self.center = center;
        self
    }

    /// Sets the padding mode used when frames are centred.
    #[must_use]
    pub fn with_pad_mode(mut self, pad_mode: StftPadMode) -> Self {
        self.pad_mode = pad_mode;
        self
    }

    /// The window length in samples (`win_length`, or `n_fft` when `None`).
    #[must_use]
    pub fn effective_win_length(&self) -> usize {
        self.win_length.map_or(self.n_fft.get(), NonZeroUsize::get)
    }

    /// The hop length in samples (`hop_length`, or `win_length / 4` when `None`).
    ///
    /// # Errors
    ///
    /// [`ParameterError`] if `hop_length` is `None` and `win_length / 4` is 0 (a
    /// window shorter than 4 samples needs an explicit hop).
    pub fn effective_hop_length(&self) -> AudioSampleResult<NonZeroUsize> {
        match self.hop_length {
            Some(h) => Ok(h),
            None => NonZeroUsize::new(self.effective_win_length() / 4).ok_or_else(|| {
                param_err(
                    "hop_length",
                    "win_length / 4 is 0; set hop_length explicitly for windows shorter than 4 samples",
                )
            }),
        }
    }

    /// Number of frequency bins of the STFT, `n_fft / 2 + 1`.
    #[must_use]
    pub fn n_bins(&self) -> usize {
        self.n_fft.get() / 2 + 1
    }

    /// Checks the configuration.
    ///
    /// # Errors
    ///
    /// [`ParameterError`] if `win_length > n_fft`, the effective hop is 0, or the
    /// window is invalid for `win_length` samples.
    pub fn validate(&self) -> AudioSampleResult<()> {
        self.effective_hop_length()?;
        padded_window(&self.window, self.win_length, self.n_fft).map(|_| ())
    }

    /// The periodic window zero-padded to `n_fft`, plus the hop length.
    fn analysis_window(&self) -> AudioSampleResult<(Vec<f64>, usize)> {
        let hop = self.effective_hop_length()?.get();
        let w = padded_window(&self.window, self.win_length, self.n_fft)?;
        Ok((w.to_vec(), hop))
    }
}

// =============================================================================
// STFT / ISTFT on raw arrays
// =============================================================================

/// Source index into a signal of length `len` for padded position `i` (may be
/// negative or `>= len`), following `numpy.pad` semantics for each mode.
/// Returns `None` for constant (zero) padding outside the signal.
fn pad_source_index(i: isize, len: usize, mode: StftPadMode) -> Option<usize> {
    let n = len as isize;
    if (0..n).contains(&i) {
        return Some(i as usize);
    }
    match mode {
        StftPadMode::Constant => None,
        StftPadMode::Edge => Some(if i < 0 { 0 } else { len - 1 }),
        StftPadMode::Reflect => {
            if len == 1 {
                return Some(0);
            }
            let period = 2 * (n - 1);
            let m = i.rem_euclid(period);
            Some(if m < n { m } else { period - m } as usize)
        }
        StftPadMode::Symmetric => {
            let period = 2 * n;
            let m = i.rem_euclid(period);
            Some(if m < n { m } else { period - 1 - m } as usize)
        }
    }
}

/// Pads `x` by `pad` samples at both ends with `mode`.
pub(crate) fn pad_signal(x: &[f64], pad: usize, mode: StftPadMode) -> Vec<f64> {
    let mut out = vec![0.0; x.len() + 2 * pad];
    out[pad..pad + x.len()].copy_from_slice(x);
    if mode != StftPadMode::Constant {
        for p in (0..pad).chain(pad + x.len()..out.len()) {
            if let Some(src) = pad_source_index(p as isize - pad as isize, x.len(), mode) {
                out[p] = x[src];
            }
        }
    }
    out
}

/// Computes the STFT of every row of `signal` (shape `(channels, samples)`),
/// equivalent to `librosa.stft`.
///
/// This is the array-level function behind [`AudioSpectrum::stft_matrix`]; use it
/// when the signal is not held in an [`AudioSamples`].
///
/// # Arguments
///
/// * `signal` - `(channels, samples)` array of samples (at least one sample).
/// * `config` - STFT parameters.
///
/// # Returns
///
/// A complex array of shape `(channels, n_fft / 2 + 1, n_frames)` where
/// `n_frames = 1 + (len + 2 * (n_fft / 2) - n_fft) / hop` when centred and
/// `1 + (len - n_fft) / hop` otherwise. Frame `t`, bin `k` is
/// `sum_n w[n] x[t * hop + n - pad] exp(-2 pi i k n / n_fft)`.
///
/// # Errors
///
/// - [`ParameterError`] if the configuration is invalid ([`StftConfig::validate`]),
///   the signal is empty, or `center` is `false` and the signal is shorter than `n_fft`.
///
/// # Examples
///
/// ```rust
/// use audio_samples::nzu;
/// use audio_samples::operations::spectrum::{StftConfig, stft};
/// use ndarray::Array2;
///
/// let x = Array2::from_shape_fn((2, 4096), |(c, n)| ((n * (c + 1)) as f64 * 0.01).sin());
/// let s = stft(x.view(), &StftConfig::new().with_n_fft(nzu!(512))).unwrap();
/// assert_eq!(s.dim(), (2, 257, 33));
/// ```
///
/// # References
///
/// J. B. Allen, "Short term spectral analysis, synthesis, and modification by
/// discrete Fourier transform," *IEEE Trans. ASSP* 25(3), 1977; librosa `core.stft`.
pub fn stft(
    signal: ArrayView2<'_, f64>,
    config: &StftConfig,
) -> AudioSampleResult<Array3<Complex<f64>>> {
    let (window, hop) = config.analysis_window()?;
    let n_fft = config.n_fft.get();
    let (n_channels, len) = signal.dim();
    if len == 0 || n_channels == 0 {
        return Err(param_err(
            "signal",
            "cannot compute the STFT of an empty signal",
        ));
    }
    let pad = if config.center { n_fft / 2 } else { 0 };
    let padded_len = len + 2 * pad;
    if padded_len < n_fft {
        return Err(param_err(
            "n_fft",
            format!(
                "n_fft={n_fft} is too large for uncentred analysis of a signal of length {len}"
            ),
        ));
    }
    let n_frames = 1 + (padded_len - n_fft) / hop;
    let n_bins = n_fft / 2 + 1;

    let plan = forward_plan(n_fft);
    let mut scratch = plan.make_scratch_vec();
    let mut frame_in = vec![0.0; n_fft];
    // Frames are transformed in blocks of `STFT_BLOCK` into a small frame-major
    // buffer, then scattered into the bin-major output: each bin row receives a
    // contiguous run of `STFT_BLOCK` frames, and the block stays in cache (a
    // whole-signal transpose is several times slower than the FFTs).
    let mut block = vec![Complex::new(0.0, 0.0); STFT_BLOCK * n_bins];
    let mut out = Array3::<Complex<f64>>::uninit((n_channels, n_bins, n_frames));
    let mut row_buf = Vec::new();

    for (c, row) in signal.outer_iter().enumerate() {
        let x: &[f64] = if let Some(s) = row.as_slice() {
            s
        } else {
            row_buf.clear();
            row_buf.extend(row.iter().copied());
            &row_buf
        };
        let padded;
        let xp: &[f64] = if pad > 0 {
            padded = pad_signal(x, pad, config.pad_mode);
            &padded
        } else {
            x
        };
        let mut out_c = out.index_axis_mut(Axis(0), c);
        for t0 in (0..n_frames).step_by(STFT_BLOCK) {
            let nb = STFT_BLOCK.min(n_frames - t0);
            for (j, spec) in block.chunks_exact_mut(n_bins).take(nb).enumerate() {
                let t = t0 + j;
                let seg = &xp[t * hop..t * hop + n_fft];
                for ((dst, &s), &w) in frame_in.iter_mut().zip(seg).zip(&window) {
                    *dst = w * s;
                }
                plan.process_with_scratch(&mut frame_in, spec, &mut scratch)
                    .map_err(|e| {
                        AudioSampleError::Processing(crate::ProcessingError::algorithm_failure(
                            "stft",
                            e.to_string(),
                        ))
                    })?;
            }
            for (k, mut dst) in out_c.outer_iter_mut().enumerate() {
                for (j, d) in dst
                    .slice_mut(ndarray::s![t0..t0 + nb])
                    .iter_mut()
                    .enumerate()
                {
                    d.write(block[j * n_bins + k]);
                }
            }
        }
    }
    // SAFETY: every element was written above: channels `c`, bins `k` and
    // frames `t0..t0 + nb` for consecutive blocks covering `0..n_frames`.
    Ok(unsafe { out.assume_init() })
}

/// Reconstructs signals from an STFT matrix, equivalent to `librosa.istft`.
///
/// Each frame is inverse-FFT'd, windowed with the same (periodic, zero-padded)
/// window and overlap-added; the sum is divided by the window's sum-squared envelope
/// ([`window_sumsquare`](crate::operations::filterbanks::window_sumsquare)) wherever
/// that envelope exceeds the smallest positive `f64`. With centred frames the
/// `n_fft / 2` padding is trimmed. This is the least-squares inverse of [`stft`]
/// (Griffin and Lim), so `istft(stft(x)) == x` up to round-off.
///
/// # Arguments
///
/// * `stft_matrix` - `(channels, n_fft / 2 + 1, n_frames)` complex array.
/// * `config` - The STFT parameters used for analysis. `n_fft` must agree with the
///   number of frequency bins. `pad_mode` is not used.
/// * `length` - If `Some(n)`, the output is exactly `n` samples per channel
///   (zero-padded or truncated, and frames not needed for `n` samples are skipped),
///   which recovers the original length. If `None`, the length is
///   `n_fft + hop * (n_frames - 1)`, minus `2 * (n_fft / 2)` when centred.
///
/// # Returns
///
/// A `(channels, length)` array of samples.
///
/// # Errors
///
/// [`ParameterError`] if the configuration is invalid, the bin count does not match
/// `n_fft`, there are no frames, or the resulting length is 0.
///
/// # Differences from librosa
///
/// librosa infers `n_fft` from the bin count when not given; here `config.n_fft` is
/// authoritative and a mismatch is an error. `length = Some(0)` is rejected (librosa
/// treats `length=0` like `None`).
///
/// # Examples
///
/// ```rust
/// use audio_samples::nzu;
/// use audio_samples::operations::spectrum::{StftConfig, istft, stft};
/// use ndarray::Array2;
///
/// let x = Array2::from_shape_fn((1, 3000), |(_, n)| (n as f64 * 0.05).sin());
/// let cfg = StftConfig::new().with_n_fft(nzu!(256));
/// let y = istft(stft(x.view(), &cfg).unwrap().view(), &cfg, Some(3000)).unwrap();
/// let err = (&y - &x).iter().fold(0.0f64, |m, d| m.max(d.abs()));
/// assert!(err < 1e-12);
/// ```
///
/// # References
///
/// D. W. Griffin and J. S. Lim, "Signal estimation from modified short-time Fourier
/// transform," *IEEE Trans. ASSP* 32(2), 1984; librosa `core.istft`.
pub fn istft(
    stft_matrix: ArrayView3<'_, Complex<f64>>,
    config: &StftConfig,
    length: Option<usize>,
) -> AudioSampleResult<Array2<f64>> {
    let (window, hop) = config.analysis_window()?;
    let n_fft = config.n_fft.get();
    let (n_channels, n_bins, total_frames) = stft_matrix.dim();
    if n_bins != n_fft / 2 + 1 {
        return Err(AudioSampleError::Parameter(
            ParameterError::invalid_configuration(
                "istft",
                format!(
                    "STFT matrix has {n_bins} frequency bins but n_fft={n_fft} implies {}",
                    n_fft / 2 + 1
                ),
            ),
        ));
    }
    if total_frames == 0 || n_channels == 0 {
        return Err(param_err(
            "stft_matrix",
            "the STFT matrix has no frames or no channels",
        ));
    }
    if length == Some(0) {
        return Err(param_err(
            "length",
            "the requested output length must be positive",
        ));
    }
    let n_frames = match length {
        Some(len) => {
            let padded = if config.center {
                len + 2 * (n_fft / 2)
            } else {
                len
            };
            total_frames.min(padded.div_ceil(hop))
        }
        None => total_frames,
    };
    let full_len = n_fft + hop * (n_frames - 1);
    let start = if config.center { n_fft / 2 } else { 0 };
    let out_len = match length {
        Some(len) => len,
        None => full_len - 2 * start,
    };
    if out_len == 0 {
        return Err(param_err(
            "stft_matrix",
            "the inverse STFT would be empty (a single centred frame with even n_fft)",
        ));
    }

    let win_sq: Vec<f64> = window.iter().map(|w| w * w).collect();
    let wss = window_sumsquare_from_squared(&win_sq, n_frames, hop);
    let scale = 1.0 / n_fft as f64;
    let plan = inverse_plan(n_fft);
    let mut scratch = plan.make_scratch_vec();
    let mut spec = vec![Complex::new(0.0, 0.0); n_bins];
    let mut frame_out = vec![0.0; n_fft];
    let mut ybuf = vec![0.0; full_len];
    let mut out = Array2::<f64>::zeros((n_channels, out_len));

    for (c, chan) in stft_matrix.outer_iter().enumerate() {
        ybuf.iter_mut().for_each(|v| *v = 0.0);
        for t in 0..n_frames {
            for (dst, src) in spec.iter_mut().zip(chan.column(t)) {
                *dst = *src;
            }
            // numpy's irfft ignores the imaginary parts of the DC and Nyquist bins.
            spec[0].im = 0.0;
            if n_fft.is_multiple_of(2) {
                spec[n_bins - 1].im = 0.0;
            }
            plan.process_with_scratch(&mut spec, &mut frame_out, &mut scratch)
                .map_err(|e| {
                    AudioSampleError::Processing(crate::ProcessingError::algorithm_failure(
                        "istft",
                        e.to_string(),
                    ))
                })?;
            let dst = &mut ybuf[t * hop..t * hop + n_fft];
            for ((y, &f), &w) in dst.iter_mut().zip(&frame_out).zip(&window) {
                *y += w * (f * scale);
            }
        }
        let mut row = out.index_axis_mut(Axis(0), c);
        for (i, y) in row.iter_mut().enumerate() {
            let src = start + i;
            if src < full_len {
                let norm = wss[src];
                *y = if norm > f64::MIN_POSITIVE {
                    ybuf[src] / norm
                } else {
                    ybuf[src]
                };
            }
        }
    }
    Ok(out)
}

/// Splits a 1-D signal into overlapping frames, equivalent to
/// `librosa.util.frame(x, frame_length=frame_length, hop_length=hop_length)`.
///
/// # Arguments
///
/// * `x` - The signal.
/// * `frame_length` - Samples per frame.
/// * `hop_length` - Samples between the starts of successive frames.
///
/// # Returns
///
/// A `(frame_length, n_frames)` array whose column `t` is
/// `x[t * hop_length .. t * hop_length + frame_length]`, with
/// `n_frames = 1 + (x.len() - frame_length) / hop_length`. Trailing samples that do
/// not fill a whole frame are dropped (no padding).
///
/// # Errors
///
/// [`ParameterError`] if `x` is shorter than `frame_length`.
///
/// # Examples
///
/// ```rust
/// use audio_samples::nzu;
/// use audio_samples::operations::spectrum::frame;
///
/// let x: Vec<f64> = (0..10).map(f64::from).collect();
/// let f = frame(&x, nzu!(4), nzu!(3)).unwrap();
/// assert_eq!(f.dim(), (4, 3));
/// assert_eq!(f.column(1).to_vec(), vec![3.0, 4.0, 5.0, 6.0]);
/// ```
pub fn frame(
    x: &[f64],
    frame_length: NonZeroUsize,
    hop_length: NonZeroUsize,
) -> AudioSampleResult<Array2<f64>> {
    let (fl, hop) = (frame_length.get(), hop_length.get());
    if x.len() < fl {
        return Err(param_err(
            "frame_length",
            format!("input is too short (n={}) for frame_length={fl}", x.len()),
        ));
    }
    let n_frames = 1 + (x.len() - fl) / hop;
    Ok(Array2::from_shape_fn((fl, n_frames), |(i, t)| {
        x[t * hop + i]
    }))
}

// =============================================================================
// dB scaling
// =============================================================================

/// Reference value that dB conversions are relative to (librosa's `ref`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DbReference {
    /// A fixed reference; its absolute value is used (librosa `ref=<float>`).
    /// librosa's default is `Value(1.0)`, i.e. absolute dB.
    Value(f64),
    /// The maximum of the input over the reduction axes (librosa `ref=numpy.max`),
    /// so the loudest element maps to 0 dB.
    Max,
}

/// Which elements share a reference maximum and a `top_db` floor (librosa's `axes`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum DbAxes {
    /// librosa's `axes="auto"` (the default): the last two axes for arrays with two or
    /// more dimensions, so each channel of a `(channels, features, frames)` array is
    /// referenced and floored independently; the whole array otherwise.
    #[default]
    Auto,
    /// The whole array (librosa `axes=None`).
    All,
}

/// Configuration of a power/amplitude-to-dB conversion, mirroring librosa's
/// `power_to_db` / `amplitude_to_db` arguments.
///
/// ## Purpose
///
/// Controls the reference level (0 dB), the smallest value considered before
/// taking the logarithm, and the dynamic-range floor.
///
/// ## Intended usage
///
/// [`DbConfig::default`] (= [`DbConfig::power`]) is librosa's `power_to_db`
/// default; use [`DbConfig::amplitude`] for `amplitude_to_db`'s defaults (its
/// `amin` is `1e-5`).
///
/// ## Invariants
///
/// Checked by [`DbConfig::validate`]: `amin` finite and positive, `top_db`
/// finite and non-negative, a `Value` reference finite.
///
/// # Examples
///
/// ```rust
/// use audio_samples::operations::spectrum::{DbConfig, DbReference};
///
/// let cfg = DbConfig::power().with_reference(DbReference::Max).with_top_db(None);
/// assert_eq!(cfg.amin, 1e-10);
/// assert_eq!(DbConfig::amplitude().amin, 1e-5);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct DbConfig {
    /// The 0 dB reference. librosa default: `Value(1.0)`.
    pub reference: DbReference,
    /// Values (and the reference) are clamped below at `amin` before the logarithm,
    /// avoiding `log(0)`. In the units of the input: `1e-10` for power (librosa
    /// `power_to_db`), `1e-5` for amplitude (`amplitude_to_db`). Must be positive.
    pub amin: f64,
    /// If `Some(t)`, the output is floored at `max - t` dB, where `max` is taken over
    /// the same axes as the reference ([`DbAxes`]). librosa default: `Some(80.0)`.
    pub top_db: Option<f64>,
    /// Reduction axes for [`DbReference::Max`] and `top_db`. librosa default: `Auto`.
    pub axes: DbAxes,
}

impl Default for DbConfig {
    fn default() -> Self {
        Self::power()
    }
}

impl DbConfig {
    /// librosa `power_to_db` defaults: `ref=1.0`, `amin=1e-10`, `top_db=80`, `axes="auto"`.
    #[must_use]
    pub fn power() -> Self {
        Self {
            reference: DbReference::Value(1.0),
            amin: 1e-10,
            top_db: Some(80.0),
            axes: DbAxes::Auto,
        }
    }

    /// librosa `amplitude_to_db` defaults: `ref=1.0`, `amin=1e-5`, `top_db=80`, `axes="auto"`.
    #[must_use]
    pub fn amplitude() -> Self {
        Self {
            amin: 1e-5,
            ..Self::power()
        }
    }

    /// Same as [`DbConfig::power`].
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the reference.
    #[must_use]
    pub fn with_reference(mut self, reference: DbReference) -> Self {
        self.reference = reference;
        self
    }

    /// Sets `amin`.
    #[must_use]
    pub fn with_amin(mut self, amin: f64) -> Self {
        self.amin = amin;
        self
    }

    /// Sets the dynamic-range floor (`None` disables it).
    #[must_use]
    pub fn with_top_db(mut self, top_db: Option<f64>) -> Self {
        self.top_db = top_db;
        self
    }

    /// Sets the reduction axes.
    #[must_use]
    pub fn with_axes(mut self, axes: DbAxes) -> Self {
        self.axes = axes;
        self
    }

    /// Checks the configuration.
    ///
    /// # Errors
    ///
    /// [`ParameterError`] if `amin` is not finite and positive, `top_db` is negative or
    /// not finite, or a `Value` reference is not finite.
    pub fn validate(&self) -> AudioSampleResult<()> {
        if !(self.amin.is_finite() && self.amin > 0.0) {
            return Err(AudioSampleError::Parameter(ParameterError::out_of_range(
                "amin",
                self.amin,
                "> 0",
                "finite",
                "amin must be strictly positive",
            )));
        }
        if let Some(t) = self.top_db {
            if !(t.is_finite() && t >= 0.0) {
                return Err(AudioSampleError::Parameter(ParameterError::out_of_range(
                    "top_db",
                    t,
                    0,
                    "finite",
                    "top_db must be non-negative",
                )));
            }
        }
        if let DbReference::Value(r) = self.reference {
            if !r.is_finite() {
                return Err(param_err("reference", "reference value must be finite"));
            }
        }
        Ok(())
    }
}

/// Size of each independent block for `axes` in a standard-layout array of `shape`.
fn db_block_size(shape: &[usize], axes: DbAxes) -> usize {
    let total: usize = shape.iter().product();
    match axes {
        DbAxes::Auto if shape.len() >= 2 => shape[shape.len() - 2] * shape[shape.len() - 1],
        _ => total,
    }
}

/// Shared implementation of `power_to_db` on already-"powered" values.
///
/// `values` are the power values in logical order; `refs` gives, per block, the
/// reference value (already squared for amplitude input).
fn power_to_db_core<D: Dimension>(
    shape: D,
    values: Vec<f64>,
    ref_for_block: impl Fn(&[f64]) -> f64,
    amin: f64,
    top_db: Option<f64>,
    axes: DbAxes,
) -> Array<f64, D> {
    let block = db_block_size(shape.slice(), axes).max(1);
    let mut out = values;
    for chunk in out.chunks_mut(block) {
        let ref_value = ref_for_block(chunk);
        let ref_db = 10.0 * amin.max(ref_value).log10();
        for v in chunk.iter_mut() {
            *v = 10.0 * amin.max(*v).log10() - ref_db;
        }
        if let Some(t) = top_db {
            let max = chunk.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let floor = max - t;
            for v in chunk.iter_mut() {
                *v = v.max(floor);
            }
        }
    }
    Array::from_shape_vec(shape, out).expect("shape matches the number of elements")
}

/// Maximum of a block (`NaN`-ignoring, `-inf` when empty).
fn block_max(chunk: &[f64]) -> f64 {
    chunk.iter().copied().fold(f64::NEG_INFINITY, f64::max)
}

/// Converts a power spectrogram (or any power-like array) to decibels, equivalent to
/// `librosa.power_to_db`.
///
/// Computes `10 log10(max(amin, S)) - 10 log10(max(amin, |ref|))`, then, if
/// `top_db` is set, floors the result at `max - top_db` where the maximum is taken
/// per block of [`DbAxes`] (per channel for `(channels, features, frames)` input
/// with the default `Auto`).
///
/// # Arguments
///
/// * `s` - Non-negative power values, any dimensionality (negative values are clamped
///   to `amin`, as in librosa).
/// * `config` - Reference, `amin`, `top_db` and axes. Use [`DbConfig::default`] for
///   librosa's defaults.
///
/// # Returns
///
/// An array of the same shape (standard layout) in dB.
///
/// # Errors
///
/// [`ParameterError`] from [`DbConfig::validate`].
///
/// # Examples
///
/// ```rust
/// use audio_samples::operations::spectrum::{DbConfig, DbReference, power_to_db};
/// use ndarray::array;
///
/// let s = array![[1.0, 0.1], [0.01, 0.0]];
/// let db = power_to_db(&s, &DbConfig::default()).unwrap();
/// assert_eq!(db, array![[0.0, -10.0], [-20.0, -80.0]]); // floored at max - 80
///
/// let rel = power_to_db(&s, &DbConfig::default().with_reference(DbReference::Max).with_top_db(None))
///     .unwrap();
/// assert_eq!(rel[[1, 1]], -100.0); // amin = 1e-10
/// ```
pub fn power_to_db<S, D>(s: &ArrayBase<S, D>, config: &DbConfig) -> AudioSampleResult<Array<f64, D>>
where
    S: Data<Elem = f64>,
    D: Dimension,
{
    config.validate()?;
    let values: Vec<f64> = s.iter().copied().collect();
    let reference = config.reference;
    Ok(power_to_db_core(
        s.raw_dim(),
        values,
        |chunk| match reference {
            DbReference::Value(r) => r.abs(),
            DbReference::Max => block_max(chunk),
        },
        config.amin,
        config.top_db,
        config.axes,
    ))
}

/// Converts an amplitude (magnitude) spectrogram to decibels, equivalent to
/// `librosa.amplitude_to_db`.
///
/// Equal to [`power_to_db`] applied to `|S|^2` with reference `ref^2` and
/// `amin^2`, i.e. `20 log10(max(amin, |S|) / max(amin, |ref|))` with the same
/// `top_db` floor.
///
/// # Arguments
///
/// * `s` - Amplitudes (absolute values are taken), any dimensionality. For complex
///   STFT values pass `stft.mapv(|z| z.norm())`.
/// * `config` - Use [`DbConfig::amplitude`] for librosa's defaults (`amin = 1e-5`);
///   [`DbConfig::default`] has the *power* default `amin = 1e-10`.
///
/// # Returns
///
/// An array of the same shape in dB.
///
/// # Errors
///
/// [`ParameterError`] from [`DbConfig::validate`].
///
/// # Examples
///
/// ```rust
/// use audio_samples::operations::spectrum::{DbConfig, amplitude_to_db};
/// use ndarray::array;
///
/// let db = amplitude_to_db(&array![1.0, 0.1, -0.01], &DbConfig::amplitude()).unwrap();
/// assert!((db[1] + 20.0).abs() < 1e-12);
/// assert!((db[2] + 40.0).abs() < 1e-12);
/// ```
pub fn amplitude_to_db<S, D>(
    s: &ArrayBase<S, D>,
    config: &DbConfig,
) -> AudioSampleResult<Array<f64, D>>
where
    S: Data<Elem = f64>,
    D: Dimension,
{
    config.validate()?;
    let power: Vec<f64> = s.iter().map(|x| x.abs() * x.abs()).collect();
    let reference = config.reference;
    Ok(power_to_db_core(
        s.raw_dim(),
        power,
        |chunk| match reference {
            DbReference::Value(r) => r.abs() * r.abs(),
            // max(|S|)^2 == max(|S|^2): squaring is monotonic and rounds identically.
            DbReference::Max => block_max(chunk),
        },
        config.amin * config.amin,
        config.top_db,
        config.axes,
    ))
}

/// Converts decibels back to power, equivalent to `librosa.db_to_power(S_db, ref=reference)`:
/// `reference * 10^(S_db / 10)`.
///
/// This inverts [`power_to_db`] for values above the `amin` / `top_db` floors.
///
/// # Examples
///
/// ```rust
/// use audio_samples::operations::spectrum::db_to_power;
/// use ndarray::array;
///
/// let p = db_to_power(&array![0.0, -10.0, 20.0], 1.0);
/// assert!((p[1] - 0.1).abs() < 1e-15);
/// assert!((p[2] - 100.0).abs() < 1e-12);
/// ```
#[must_use]
pub fn db_to_power<S, D>(s_db: &ArrayBase<S, D>, reference: f64) -> Array<f64, D>
where
    S: Data<Elem = f64>,
    D: Dimension,
{
    let v: Vec<f64> = s_db
        .iter()
        .map(|&x| reference * 10f64.powf(x * 0.1))
        .collect();
    Array::from_shape_vec(s_db.raw_dim(), v).expect("shape matches the number of elements")
}

/// Converts decibels back to amplitude, equivalent to
/// `librosa.db_to_amplitude(S_db, ref=reference)`: `sqrt(reference^2 * 10^(S_db / 10))`.
///
/// # Examples
///
/// ```rust
/// use audio_samples::operations::spectrum::db_to_amplitude;
/// use ndarray::array;
///
/// let a = db_to_amplitude(&array![-20.0, 0.0], 1.0);
/// assert!((a[0] - 0.1).abs() < 1e-15);
/// assert_eq!(a[1], 1.0);
/// ```
#[must_use]
pub fn db_to_amplitude<S, D>(s_db: &ArrayBase<S, D>, reference: f64) -> Array<f64, D>
where
    S: Data<Elem = f64>,
    D: Dimension,
{
    let r2 = reference * reference;
    let v: Vec<f64> = s_db
        .iter()
        .map(|&x| (r2 * 10f64.powf(x * 0.1)).sqrt())
        .collect();
    Array::from_shape_vec(s_db.raw_dim(), v).expect("shape matches the number of elements")
}

// =============================================================================
// Mel spectrogram and MFCC configuration
// =============================================================================

/// Configuration of a mel spectrogram, mirroring `librosa.feature.melspectrogram`.
///
/// ## Purpose
///
/// Combines the STFT parameters, the mel filterbank and the exponent applied to
/// the STFT magnitude before mel pooling.
///
/// ## Intended usage
///
/// Pass to [`AudioSpectrum::melspectrogram`] or embed in an [`MfccConfig`].
/// `Default` reproduces librosa's defaults: `n_fft = 2048`, `hop_length = 512`
/// (librosa's melspectrogram fixes 512 rather than `win_length / 4`), periodic
/// Hann window, centred with zero padding, 128 Slaney-scale bands from 0 Hz to
/// Nyquist with Slaney normalisation, `power = 2` (a power spectrogram).
///
/// ## Invariants
///
/// Checked by [`MelSpectrogramConfig::validate`]: the STFT and filterbank configs are
/// valid and `power` is finite and positive.
///
/// # Examples
///
/// ```rust
/// use audio_samples::nzu;
/// use audio_samples::operations::spectrum::MelSpectrogramConfig;
///
/// let cfg = MelSpectrogramConfig::default();
/// assert_eq!(cfg.stft.hop_length, Some(nzu!(512)));
/// assert_eq!(cfg.power, 2.0);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct MelSpectrogramConfig {
    /// STFT parameters. librosa defaults, with `hop_length = Some(512)`.
    pub stft: StftConfig,
    /// Mel filterbank parameters. librosa defaults.
    pub mel: MelFilterbankConfig,
    /// Exponent applied to the STFT magnitude before the mel projection: `2.0` for
    /// power (librosa's default), `1.0` for magnitude. Must be finite and positive.
    pub power: f64,
}

impl Default for MelSpectrogramConfig {
    fn default() -> Self {
        Self {
            stft: StftConfig::default().with_hop_length(Some(crate::nzu!(512))),
            mel: MelFilterbankConfig::default(),
            power: 2.0,
        }
    }
}

impl MelSpectrogramConfig {
    /// Returns the librosa-default configuration (same as [`Default`]).
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the STFT parameters.
    #[must_use]
    pub fn with_stft(mut self, stft: StftConfig) -> Self {
        self.stft = stft;
        self
    }

    /// Sets the mel filterbank parameters.
    #[must_use]
    pub fn with_mel(mut self, mel: MelFilterbankConfig) -> Self {
        self.mel = mel;
        self
    }

    /// Sets the magnitude exponent.
    #[must_use]
    pub fn with_power(mut self, power: f64) -> Self {
        self.power = power;
        self
    }

    /// Checks the configuration for a given sample rate.
    ///
    /// # Errors
    ///
    /// [`ParameterError`] if the STFT or filterbank configuration is invalid, or
    /// `power` is not finite and positive.
    pub fn validate(&self, sample_rate: f64) -> AudioSampleResult<()> {
        validate_power(self.power)?;
        self.stft.validate()?;
        self.mel.validate(sample_rate)
    }
}

fn validate_power(power: f64) -> AudioSampleResult<()> {
    if power.is_finite() && power > 0.0 {
        Ok(())
    } else {
        Err(AudioSampleError::Parameter(ParameterError::out_of_range(
            "power",
            power,
            "> 0",
            "finite",
            "the spectrogram exponent must be finite and positive",
        )))
    }
}

/// The DCT type used to compute cepstral coefficients (SciPy `scipy.fft.dct` `type`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum DctType {
    /// DCT-I (requires at least 2 mel bands).
    One,
    /// DCT-II, the standard MFCC transform (librosa default).
    #[default]
    Two,
    /// DCT-III (the inverse of an unnormalised DCT-II up to scale).
    Three,
}

/// Normalisation of the DCT (SciPy `scipy.fft.dct` `norm`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum DctNorm {
    /// No normalisation (SciPy `norm=None` / `"backward"`). DCT-II coefficients are then
    /// `2 sum_n x[n] cos(pi k (2n + 1) / (2N))`.
    Unnormalized,
    /// Orthonormal DCT (SciPy / librosa `norm="ortho"`, the default).
    #[default]
    Ortho,
}

/// Configuration of MFCC extraction, mirroring `librosa.feature.mfcc`.
///
/// ## Purpose
///
/// Describes the full MFCC pipeline: mel power spectrogram, conversion to dB,
/// DCT along the mel axis, truncation to `n_mfcc` coefficients and optional
/// liftering.
///
/// ## Intended usage
///
/// Pass to [`AudioSpectrum::mfcc_frames`] (from audio) or [`mfcc_from_mel_db`]
/// (from a precomputed log-mel spectrogram, like librosa's `mfcc(S=...)`).
/// `Default` reproduces librosa's defaults: 20 coefficients from a 128-band
/// Slaney mel power spectrogram (librosa `mel_norm="slaney"`), `power_to_db`
/// with `ref=1`, `amin=1e-10`, `top_db=80`, orthonormal DCT-II, no liftering.
///
/// ## Invariants
///
/// Checked by [`MfccConfig::validate`]: `n_mfcc <= n_mels`, DCT-I needs at least two
/// mel bands, `lifter` is finite and non-negative, and the nested configs are valid.
///
/// # Examples
///
/// ```rust
/// use audio_samples::nzu;
/// use audio_samples::operations::spectrum::{DctNorm, DctType, MfccConfig};
///
/// let htk_like = MfccConfig::new()
///     .with_n_mfcc(nzu!(13))
///     .with_lifter(22.0)
///     .with_dct_type(DctType::Two)
///     .with_dct_norm(DctNorm::Ortho);
/// assert_eq!(htk_like.n_mfcc.get(), 13);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct MfccConfig {
    /// Mel spectrogram parameters (librosa's `**kwargs` and `mel_norm`). librosa defaults.
    pub mel_spectrogram: MelSpectrogramConfig,
    /// Conversion of the mel power spectrogram to dB. librosa uses `power_to_db`
    /// defaults ([`DbConfig::power`]).
    pub db: DbConfig,
    /// Number of cepstral coefficients kept (the first `n_mfcc` DCT outputs).
    /// librosa default: 20. Must not exceed the number of mel bands.
    pub n_mfcc: NonZeroUsize,
    /// DCT type. librosa default: [`DctType::Two`].
    pub dct_type: DctType,
    /// DCT normalisation. librosa default: [`DctNorm::Ortho`].
    pub dct_norm: DctNorm,
    /// Cepstral liftering coefficient `L`: if positive, coefficient `k` (0-based) is
    /// scaled by `1 + (L / 2) sin(pi (k + 1) / L)`, de-emphasising high quefrencies
    /// less than the low ones. `0` disables liftering (librosa default).
    pub lifter: f64,
}

impl Default for MfccConfig {
    fn default() -> Self {
        Self {
            mel_spectrogram: MelSpectrogramConfig::default(),
            db: DbConfig::power(),
            n_mfcc: crate::nzu!(20),
            dct_type: DctType::Two,
            dct_norm: DctNorm::Ortho,
            lifter: 0.0,
        }
    }
}

impl MfccConfig {
    /// Returns the librosa-default configuration (same as [`Default`]).
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the mel spectrogram parameters.
    #[must_use]
    pub fn with_mel_spectrogram(mut self, mel_spectrogram: MelSpectrogramConfig) -> Self {
        self.mel_spectrogram = mel_spectrogram;
        self
    }

    /// Sets the dB conversion.
    #[must_use]
    pub fn with_db(mut self, db: DbConfig) -> Self {
        self.db = db;
        self
    }

    /// Sets the number of coefficients.
    #[must_use]
    pub fn with_n_mfcc(mut self, n_mfcc: NonZeroUsize) -> Self {
        self.n_mfcc = n_mfcc;
        self
    }

    /// Sets the DCT type.
    #[must_use]
    pub fn with_dct_type(mut self, dct_type: DctType) -> Self {
        self.dct_type = dct_type;
        self
    }

    /// Sets the DCT normalisation.
    #[must_use]
    pub fn with_dct_norm(mut self, dct_norm: DctNorm) -> Self {
        self.dct_norm = dct_norm;
        self
    }

    /// Sets the lifter coefficient (`0` disables liftering).
    #[must_use]
    pub fn with_lifter(mut self, lifter: f64) -> Self {
        self.lifter = lifter;
        self
    }

    /// Checks the DCT / lifter part of the configuration for `n_mels` input bands.
    fn validate_cepstrum(&self, n_mels: usize) -> AudioSampleResult<()> {
        if !(self.lifter.is_finite() && self.lifter >= 0.0) {
            return Err(AudioSampleError::Parameter(ParameterError::out_of_range(
                "lifter",
                self.lifter,
                0,
                "finite",
                "the MFCC lifter must be a finite, non-negative number",
            )));
        }
        if self.n_mfcc.get() > n_mels {
            return Err(AudioSampleError::Parameter(ParameterError::out_of_range(
                "n_mfcc",
                self.n_mfcc.get(),
                1,
                n_mels,
                "cannot keep more cepstral coefficients than there are mel bands",
            )));
        }
        if self.dct_type == DctType::One && n_mels < 2 {
            return Err(param_err("dct_type", "DCT-I needs at least two mel bands"));
        }
        Ok(())
    }

    /// Checks the configuration for a given sample rate.
    ///
    /// # Errors
    ///
    /// [`ParameterError`] if any nested configuration is invalid, `n_mfcc` exceeds the
    /// number of mel bands, DCT-I is used with a single band, or `lifter` is negative or
    /// not finite.
    pub fn validate(&self, sample_rate: f64) -> AudioSampleResult<()> {
        self.mel_spectrogram.validate(sample_rate)?;
        self.db.validate()?;
        self.validate_cepstrum(self.mel_spectrogram.mel.n_mels.get())
    }
}

/// `cos(pi * m / denom)` with `m` reduced modulo `2 * denom` first, for accuracy.
fn cos_pi_ratio(m: usize, denom: usize) -> f64 {
    let m = m % (2 * denom);
    (std::f64::consts::PI * m as f64 / denom as f64).cos()
}

/// The first `n_out` rows of the `n_in`-point DCT matrix (SciPy conventions).
fn dct_matrix(n_out: usize, n_in: usize, dct_type: DctType, norm: DctNorm) -> Array2<f64> {
    let nf = n_in as f64;
    let ortho = norm == DctNorm::Ortho;
    Array2::from_shape_fn((n_out, n_in), |(k, n)| match dct_type {
        DctType::Two => {
            let c = 2.0 * cos_pi_ratio(k * (2 * n + 1), 2 * n_in);
            if ortho {
                let f = if k == 0 {
                    (1.0 / (4.0 * nf)).sqrt()
                } else {
                    (1.0 / (2.0 * nf)).sqrt()
                };
                c * f
            } else {
                c
            }
        }
        DctType::Three => {
            let c = if n == 0 {
                1.0
            } else {
                cos_pi_ratio(n * (2 * k + 1), 2 * n_in)
            };
            if ortho {
                c * if n == 0 {
                    1.0 / nf.sqrt()
                } else {
                    (2.0 / nf).sqrt()
                }
            } else if n == 0 {
                c
            } else {
                2.0 * c
            }
        }
        DctType::One => {
            let edge_n = n == 0 || n == n_in - 1;
            let c = if edge_n { 1.0 } else { 2.0 } * cos_pi_ratio(k * n, n_in - 1);
            if ortho {
                let v = if edge_n {
                    std::f64::consts::SQRT_2
                } else {
                    1.0
                };
                let w = if k == 0 || k == n_in - 1 {
                    std::f64::consts::FRAC_1_SQRT_2
                } else {
                    1.0
                };
                (1.0 / (2.0 * (nf - 1.0))).sqrt() * w * v * c
            } else {
                c
            }
        }
    })
}

/// Computes MFCCs from a log-power (dB) mel spectrogram, equivalent to
/// `librosa.feature.mfcc(S=mel_db, n_mfcc=..., dct_type=..., norm=..., lifter=...)`.
///
/// Applies the DCT along the second-to-last axis (the mel axis), keeps the first
/// `n_mfcc` coefficients and applies the lifter. Only the `n_mfcc`, `dct_type`,
/// `dct_norm` and `lifter` fields of `config` are used.
///
/// # Arguments
///
/// * `mel_db` - Log-power mel spectrogram, e.g. `(n_mels, n_frames)` or
///   `(channels, n_mels, n_frames)` (at least 2-D), typically from [`power_to_db`]
///   of [`AudioSpectrum::melspectrogram`].
/// * `config` - MFCC parameters.
///
/// # Returns
///
/// An array with the same shape as `mel_db` except that the mel axis has `n_mfcc`
/// entries.
///
/// # Errors
///
/// [`ParameterError`] if `mel_db` has fewer than two dimensions, an empty mel axis, or
/// the DCT/lifter parameters are invalid for its number of bands.
///
/// # Differences from librosa
///
/// librosa silently returns fewer than `n_mfcc` rows when `n_mfcc > n_mels`; this is an
/// error here.
///
/// # Examples
///
/// ```rust
/// use audio_samples::nzu;
/// use audio_samples::operations::spectrum::{MfccConfig, mfcc_from_mel_db};
/// use ndarray::Array2;
///
/// // A flat log-mel spectrum has energy only in c0.
/// let s = Array2::from_elem((40, 5), -20.0);
/// let m = mfcc_from_mel_db(&s, &MfccConfig::new().with_n_mfcc(nzu!(13))).unwrap();
/// assert_eq!(m.dim(), (13, 5));
/// assert!((m[[0, 0]] + 20.0 * 40f64.sqrt()).abs() < 1e-10);
/// assert!(m[[1, 0]].abs() < 1e-10);
/// ```
///
/// # References
///
/// S. Davis and P. Mermelstein, "Comparison of parametric representations for
/// monosyllabic word recognition in continuously spoken sentences," *IEEE Trans. ASSP*
/// 28(4), 1980; librosa `feature.mfcc`.
pub fn mfcc_from_mel_db<S, D>(
    mel_db: &ArrayBase<S, D>,
    config: &MfccConfig,
) -> AudioSampleResult<Array<f64, D>>
where
    S: Data<Elem = f64>,
    D: Dimension,
{
    let ndim = mel_db.ndim();
    if ndim < 2 {
        return Err(param_err(
            "mel_db",
            "expected at least two dimensions (mel bands, frames)",
        ));
    }
    let ax = Axis(ndim - 2);
    let n_mels = mel_db.len_of(ax);
    if n_mels == 0 {
        return Err(param_err("mel_db", "the mel axis is empty"));
    }
    config.validate_cepstrum(n_mels)?;
    let n_mfcc = config.n_mfcc.get();
    let mut basis = dct_matrix(n_mfcc, n_mels, config.dct_type, config.dct_norm);
    if config.lifter > 0.0 {
        let l = config.lifter;
        for (k, mut row) in basis.outer_iter_mut().enumerate() {
            let li = (std::f64::consts::PI * (k + 1) as f64 / l).sin();
            let g = 1.0 + (l / 2.0) * li;
            row.mapv_inplace(|b| b * g);
        }
    }
    let mut shape = mel_db.raw_dim();
    shape[ndim - 2] = n_mfcc;
    let mut out = Array::<f64, D>::zeros(shape);
    let mut lane_buf = vec![0.0; n_mels];
    Zip::from(out.lanes_mut(ax))
        .and(mel_db.lanes(ax))
        .for_each(|mut o, x| {
            for (dst, &v) in lane_buf.iter_mut().zip(x.iter()) {
                *dst = v;
            }
            for (y, row) in o.iter_mut().zip(basis.outer_iter()) {
                *y = row.iter().zip(&lane_buf).map(|(b, v)| b * v).sum();
            }
        });
    Ok(out)
}

/// Projects a (power or magnitude) spectrogram onto mel bands, equivalent to
/// `librosa.feature.melspectrogram(S=spectrogram, sr=sample_rate, n_fft=n_fft, ...)`.
///
/// # Arguments
///
/// * `spectrogram` - `(channels, n_fft / 2 + 1, n_frames)` non-negative spectrogram,
///   e.g. from [`AudioSpectrum::spectrogram_matrix`].
/// * `sample_rate` - Sampling rate of the analysed signal in Hz.
/// * `n_fft` - FFT length used to compute `spectrogram`.
/// * `mel` - Mel filterbank parameters.
///
/// # Returns
///
/// A `(channels, n_mels, n_frames)` mel spectrogram.
///
/// # Errors
///
/// [`ParameterError`] if the filterbank configuration is invalid or the bin count of
/// `spectrogram` does not match `n_fft`.
///
/// # Examples
///
/// ```rust
/// use audio_samples::nzu;
/// use audio_samples::operations::filterbanks::MelFilterbankConfig;
/// use audio_samples::operations::spectrum::melspectrogram_from_spectrogram;
/// use ndarray::Array3;
///
/// let s = Array3::<f64>::ones((1, 257, 4));
/// let m = melspectrogram_from_spectrogram(
///     s.view(), 16000.0, nzu!(512), &MelFilterbankConfig::new().with_n_mels(nzu!(40)),
/// ).unwrap();
/// assert_eq!(m.dim(), (1, 40, 4));
/// ```
pub fn melspectrogram_from_spectrogram(
    spectrogram: ArrayView3<'_, f64>,
    sample_rate: f64,
    n_fft: NonZeroUsize,
    mel: &MelFilterbankConfig,
) -> AudioSampleResult<Array3<f64>> {
    let (n_channels, n_bins, n_frames) = spectrogram.dim();
    if n_bins != n_fft.get() / 2 + 1 {
        return Err(AudioSampleError::Parameter(
            ParameterError::invalid_configuration(
                "melspectrogram",
                format!(
                    "spectrogram has {n_bins} frequency bins but n_fft={} implies {}",
                    n_fft.get(),
                    n_fft.get() / 2 + 1
                ),
            ),
        ));
    }
    let basis = mel_filterbank(sample_rate, n_fft, mel)?;
    let mut out = Array3::<f64>::zeros((n_channels, basis.nrows(), n_frames));
    for (c, s) in spectrogram.outer_iter().enumerate() {
        banded_dot_into(&basis, s, out.index_axis_mut(Axis(0), c));
    }
    Ok(out)
}

/// `out = basis · s` for a filterbank whose rows are each non-zero over one
/// contiguous range of columns (mel and similar triangular filterbanks).
///
/// Only the non-zero range of every row is visited, so the cost is
/// proportional to the number of non-zero filter weights rather than to
/// `rows × columns`; each output row is accumulated as a sum of scaled
/// spectrogram rows (contiguous, vectorisable). For finite input the result
/// equals the dense product up to floating-point summation order.
pub(crate) fn banded_dot_into(
    basis: &Array2<f64>,
    s: ArrayView2<'_, f64>,
    mut out: ndarray::ArrayViewMut2<'_, f64>,
) {
    let s = s.as_standard_layout();
    let n_frames = s.ncols();
    for (filter, mut out_row) in basis.outer_iter().zip(out.outer_iter_mut()) {
        out_row.fill(0.0);
        let Some(lo) = filter.iter().position(|&w| w != 0.0) else {
            continue;
        };
        let hi = filter.len() - filter.iter().rev().position(|&w| w != 0.0).unwrap_or(0);
        let acc = out_row
            .as_slice_mut()
            .expect("rows of a standard-layout array");
        for k in lo..hi {
            let w = filter[k];
            let src = s.row(k);
            let src = src.as_slice().expect("rows of a standard-layout array");
            for (a, &v) in acc[..n_frames].iter_mut().zip(src) {
                *a += w * v;
            }
        }
    }
}

/// `|z|^power`.
///
/// Computed from `re² + im²` rather than numpy's `hypot`-based `abs(z)`
/// (the libm `hypot` call dominated spectrogram computation); the two agree to
/// within about one ulp. STFT values are far from the range where the squares
/// could overflow or underflow.
fn magnitude_power(z: Complex<f64>, power: f64) -> f64 {
    let sq = z.norm_sqr();
    if power == 2.0 {
        sq
    } else if power == 1.0 {
        sq.sqrt()
    } else {
        sq.sqrt().powf(power)
    }
}

// =============================================================================
// delta / stack_memory
// =============================================================================

/// How [`delta`] handles the edges of the data (SciPy `savgol_filter` `mode`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum DeltaMode {
    /// Fit a polynomial to the first and last `width` samples and differentiate it
    /// at the edge points (librosa default). Requires `width <= ` the axis length.
    #[default]
    Interp,
    /// Repeat the edge sample.
    Nearest,
    /// Mirror without repeating the edge sample (SciPy `"mirror"`).
    Mirror,
    /// Pad with zeros.
    Constant,
    /// Wrap around periodically.
    Wrap,
}

/// Configuration of [`delta`], mirroring `librosa.feature.delta`.
///
/// ## Purpose
///
/// Controls the Savitzky-Golay derivative estimate: the window width, the
/// derivative order, the axis it runs along and the edge treatment.
///
/// ## Intended usage
///
/// `Default` is librosa's default: `width = 9`, first derivative, along the last
/// axis (frames), `Interp` edges. For delta-deltas use `order = 2`.
///
/// ## Invariants
///
/// `width` is odd and `>= 3`, `1 <= order < width`, `axis` addresses an existing
/// axis (negative values count from the end), and with `Interp` the axis has at
/// least `width` entries. Checked when [`delta`] runs.
///
/// # Examples
///
/// ```rust
/// use audio_samples::operations::spectrum::DeltaConfig;
///
/// let dd = DeltaConfig::new().with_order(2).with_width(5);
/// assert_eq!(dd.axis, -1);
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeltaConfig {
    /// Number of frames in the local fit (odd, >= 3). Wider windows smooth more.
    /// librosa default: 9.
    pub width: usize,
    /// Derivative order (1 = delta, 2 = delta-delta). Also the fitted polynomial
    /// order, as in librosa. librosa default: 1.
    pub order: usize,
    /// Axis along which to differentiate; negative values count from the end.
    /// librosa default: -1 (frames).
    pub axis: isize,
    /// Edge handling. librosa default: [`DeltaMode::Interp`].
    pub mode: DeltaMode,
}

impl Default for DeltaConfig {
    fn default() -> Self {
        Self {
            width: 9,
            order: 1,
            axis: -1,
            mode: DeltaMode::Interp,
        }
    }
}

impl DeltaConfig {
    /// Returns the librosa-default configuration (same as [`Default`]).
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the window width.
    #[must_use]
    pub fn with_width(mut self, width: usize) -> Self {
        self.width = width;
        self
    }

    /// Sets the derivative order.
    #[must_use]
    pub fn with_order(mut self, order: usize) -> Self {
        self.order = order;
        self
    }

    /// Sets the axis.
    #[must_use]
    pub fn with_axis(mut self, axis: isize) -> Self {
        self.axis = axis;
        self
    }

    /// Sets the edge mode.
    #[must_use]
    pub fn with_mode(mut self, mode: DeltaMode) -> Self {
        self.mode = mode;
        self
    }

    /// Checks the configuration against data with `ndim` dimensions and returns the
    /// resolved (non-negative) axis.
    ///
    /// # Errors
    ///
    /// [`ParameterError`] if `width` is even or below 3, `order` is 0 or not below
    /// `width`, or `axis` is out of range.
    pub fn validate(&self, ndim: usize) -> AudioSampleResult<usize> {
        if self.width < 3 || self.width.is_multiple_of(2) {
            return Err(param_err("width", "width must be an odd integer >= 3"));
        }
        if self.order == 0 || self.order >= self.width {
            return Err(AudioSampleError::Parameter(ParameterError::out_of_range(
                "order",
                self.order,
                1,
                self.width - 1,
                "the derivative order must be positive and smaller than width",
            )));
        }
        resolve_axis(self.axis, ndim)
    }
}

fn resolve_axis(axis: isize, ndim: usize) -> AudioSampleResult<usize> {
    let n = ndim as isize;
    let a = if axis < 0 { axis + n } else { axis };
    if (0..n).contains(&a) {
        Ok(a as usize)
    } else {
        Err(AudioSampleError::Parameter(ParameterError::out_of_range(
            "axis",
            axis,
            -n,
            n - 1,
            "axis out of range for the data",
        )))
    }
}

/// Solves the `(p+1) x (p+1)` system `g a = rhs` for every column of `rhs`
/// (Gaussian elimination with partial pivoting; `g` is a small, well-conditioned
/// Gram matrix).
fn solve_small(mut g: Vec<Vec<f64>>, mut rhs: Vec<Vec<f64>>) -> Vec<Vec<f64>> {
    let n = g.len();
    for col in 0..n {
        let pivot = (col..n)
            .max_by(|&a, &b| g[a][col].abs().total_cmp(&g[b][col].abs()))
            .unwrap_or(col);
        g.swap(col, pivot);
        rhs.swap(col, pivot);
        for r in col + 1..n {
            let f = g[r][col] / g[col][col];
            let (g_top, g_bottom) = g.split_at_mut(r);
            for (x, y) in g_bottom[0][col..].iter_mut().zip(&g_top[col][col..]) {
                *x -= f * y;
            }
            let (top, bottom) = rhs.split_at_mut(r);
            for (x, y) in bottom[0].iter_mut().zip(&top[col]) {
                *x -= f * y;
            }
        }
    }
    for col in (0..n).rev() {
        let (top, bottom) = rhs.split_at_mut(col + 1);
        let target = &mut top[col];
        for (f, row) in g[col][col + 1..].iter().zip(bottom.iter()) {
            for (x, y) in target.iter_mut().zip(row) {
                *x -= f * y;
            }
        }
        let d = g[col][col];
        target.iter_mut().for_each(|x| *x /= d);
    }
    rhs
}

/// Savitzky-Golay helper: the least-squares polynomial fit over a window of `width`
/// samples, in the scaled coordinate `u = (j - h) / h`.
struct SavGol {
    width: usize,
    half: usize,
    order: usize,
    /// `(order + 1) x width` pseudo-inverse: polynomial coefficients `a = pinv * x`.
    pinv: Vec<Vec<f64>>,
    /// Centre derivative weights (SciPy `savgol_coeffs(..., use="dot")`).
    centre: Vec<f64>,
}

impl SavGol {
    fn new(width: usize, order: usize) -> Self {
        let half = width / 2;
        let h = half as f64;
        let p = order; // polyorder == deriv, as in librosa.feature.delta
        let u: Vec<f64> = (0..width).map(|j| (j as f64 - h) / h).collect();
        let pow = |x: f64, q: usize| x.powi(q as i32);
        let gram: Vec<Vec<f64>> = (0..=p)
            .map(|a| {
                (0..=p)
                    .map(|b| u.iter().map(|&x| pow(x, a + b)).sum())
                    .collect()
            })
            .collect();
        let vt: Vec<Vec<f64>> = (0..=p)
            .map(|a| u.iter().map(|&x| pow(x, a)).collect())
            .collect();
        let pinv = solve_small(gram, vt);
        let scale = factorial(order) / h.powi(order as i32);
        let centre = pinv[order].iter().map(|v| v * scale).collect();
        Self {
            width,
            half,
            order,
            pinv,
            centre,
        }
    }

    /// The `order`-th derivative at window position `pos` (0-based) of the polynomial
    /// fitted to `window` (length `width`).
    fn edge_derivative(&self, window: &[f64], pos: usize) -> f64 {
        let h = self.half as f64;
        let u0 = (pos as f64 - h) / h;
        let d = self.order;
        let mut acc = 0.0;
        for q in d..self.pinv.len() {
            let a_q: f64 = self.pinv[q].iter().zip(window).map(|(p, x)| p * x).sum();
            acc += falling_factorial(q, d) * a_q * u0.powi((q - d) as i32);
        }
        acc / h.powi(d as i32)
    }
}

fn factorial(n: usize) -> f64 {
    (1..=n).map(|k| k as f64).product()
}

/// `q! / (q - d)!`
fn falling_factorial(q: usize, d: usize) -> f64 {
    ((q - d + 1)..=q).map(|k| k as f64).product()
}

/// Extended-signal index for the non-interp delta modes (`None` = zero).
fn delta_source_index(i: isize, len: usize, mode: DeltaMode) -> Option<usize> {
    let n = len as isize;
    if (0..n).contains(&i) {
        return Some(i as usize);
    }
    match mode {
        DeltaMode::Constant | DeltaMode::Interp => None,
        DeltaMode::Nearest => Some(if i < 0 { 0 } else { len - 1 }),
        DeltaMode::Wrap => Some(i.rem_euclid(n) as usize),
        DeltaMode::Mirror => pad_source_index(i, len, StftPadMode::Reflect),
    }
}

/// Estimates local derivatives of a feature sequence with a Savitzky-Golay filter,
/// equivalent to `librosa.feature.delta` (`scipy.signal.savgol_filter(data, width,
/// deriv=order, polyorder=order, mode=mode)`).
///
/// # Arguments
///
/// * `data` - Feature array of any dimensionality, e.g. MFCCs `(channels, n_mfcc, frames)`.
/// * `config` - Window width, order, axis and edge mode.
///
/// # Returns
///
/// An array of the same shape holding the `order`-th derivative estimate along the
/// chosen axis.
///
/// # Errors
///
/// [`ParameterError`] if the configuration is invalid for `data`, or `mode` is
/// [`DeltaMode::Interp`] and the axis is shorter than `width`.
///
/// # Examples
///
/// ```rust
/// use audio_samples::operations::spectrum::{DeltaConfig, delta};
/// use ndarray::Array2;
///
/// // The derivative of a linear ramp is its slope everywhere (also at the edges).
/// let ramp = Array2::from_shape_fn((2, 20), |(_, t)| 3.0 * t as f64);
/// let d = delta(&ramp, &DeltaConfig::default()).unwrap();
/// assert!(d.iter().all(|v| (v - 3.0).abs() < 1e-10));
/// ```
///
/// # References
///
/// A. Savitzky and M. J. E. Golay, "Smoothing and differentiation of data by
/// simplified least squares procedures," *Analytical Chemistry* 36(8), 1964;
/// librosa `feature.delta`.
pub fn delta<S, D>(data: &ArrayBase<S, D>, config: &DeltaConfig) -> AudioSampleResult<Array<f64, D>>
where
    S: Data<Elem = f64>,
    D: Dimension,
{
    let ax = config.validate(data.ndim())?;
    let len = data.len_of(Axis(ax));
    if config.mode == DeltaMode::Interp && config.width > len {
        return Err(param_err(
            "width",
            format!(
                "when mode is Interp, width={} cannot exceed the axis length {len}",
                config.width
            ),
        ));
    }
    let sg = SavGol::new(config.width, config.order);
    let (w, h) = (sg.width, sg.half);
    let mut out = Array::<f64, D>::zeros(data.raw_dim());
    let mut x = vec![0.0; len];
    Zip::from(out.lanes_mut(Axis(ax)))
        .and(data.lanes(Axis(ax)))
        .for_each(|mut o, lane| {
            for (dst, &v) in x.iter_mut().zip(lane.iter()) {
                *dst = v;
            }
            for (i, y) in o.iter_mut().enumerate() {
                let interior = i >= h && i + h < len;
                *y = if interior {
                    sg.centre
                        .iter()
                        .zip(&x[i - h..i - h + w])
                        .map(|(c, v)| c * v)
                        .sum()
                } else if config.mode == DeltaMode::Interp {
                    if i < h {
                        sg.edge_derivative(&x[..w], i)
                    } else {
                        sg.edge_derivative(&x[len - w..], i + w - len)
                    }
                } else {
                    sg.centre
                        .iter()
                        .enumerate()
                        .map(|(j, c)| {
                            let src = i as isize + j as isize - h as isize;
                            delta_source_index(src, len, config.mode).map_or(0.0, |s| c * x[s])
                        })
                        .sum()
                };
            }
        });
    Ok(out)
}

/// Stacks time-delayed copies of a feature array, equivalent to
/// `librosa.feature.stack_memory(data, n_steps=n_steps, delay=delay)` (zero padding).
///
/// Block `k` (rows `k * d .. (k + 1) * d` of the second-to-last axis, where `d` is its
/// original length) holds the data delayed by `k * delay` frames along the last axis:
/// `out[.., k * d + r, t] = data[.., r, t - k * delay]`, or `0` where that index is
/// outside the data. A negative `delay` looks into the future instead.
///
/// # Arguments
///
/// * `data` - At least 2-D array `(..., features, frames)`. Reshape 1-D input to
///   `(1, frames)` first.
/// * `n_steps` - Number of copies (librosa default 2).
/// * `delay` - Frame delay between copies (librosa default 1). Must not be 0.
///
/// # Returns
///
/// An array of shape `(..., features * n_steps, frames)`.
///
/// # Errors
///
/// [`ParameterError`] if `delay` is 0, `data` has fewer than two dimensions, or no frames.
///
/// # Examples
///
/// ```rust
/// use audio_samples::nzu;
/// use audio_samples::operations::spectrum::stack_memory;
/// use ndarray::array;
///
/// let x = array![[1.0, 2.0, 3.0]];
/// let s = stack_memory(&x, nzu!(2), 1).unwrap();
/// assert_eq!(s, array![[1.0, 2.0, 3.0], [0.0, 1.0, 2.0]]);
/// ```
pub fn stack_memory<S, D>(
    data: &ArrayBase<S, D>,
    n_steps: NonZeroUsize,
    delay: isize,
) -> AudioSampleResult<Array<f64, D>>
where
    S: Data<Elem = f64>,
    D: Dimension,
{
    if delay == 0 {
        return Err(param_err("delay", "delay must be a non-zero integer"));
    }
    let ndim = data.ndim();
    if ndim < 2 {
        return Err(param_err(
            "data",
            "expected at least two dimensions (features, frames); reshape 1-D input to (1, frames)",
        ));
    }
    let (feat_ax, time_ax) = (Axis(ndim - 2), Axis(ndim - 1));
    let t = data.len_of(time_ax);
    if t == 0 {
        return Err(param_err(
            "data",
            "cannot stack memory of data with no frames",
        ));
    }
    let d = data.len_of(feat_ax);
    let mut shape = data.raw_dim();
    shape[ndim - 2] = d * n_steps.get();
    let mut out = Array::<f64, D>::zeros(shape);
    for k in 0..n_steps.get() {
        let shift = (k as isize).saturating_mul(delay);
        let s = shift.unsigned_abs();
        if s >= t {
            continue;
        }
        let (dst, src) = if shift >= 0 {
            (Slice::from(s..t), Slice::from(0..t - s))
        } else {
            (Slice::from(0..t - s), Slice::from(s..t))
        };
        let mut block = out.slice_axis_mut(feat_ax, Slice::from(k * d..(k + 1) * d));
        block
            .slice_axis_mut(time_ax, dst)
            .assign(&data.slice_axis(time_ax, src));
    }
    Ok(out)
}

// =============================================================================
// The AudioSpectrum trait
// =============================================================================

/// librosa-compatible spectral analysis on [`AudioSamples`].
///
/// Every method works on all channels: outputs are shaped
/// `(channels, n_features, n_frames)` (mono gives one channel), always `f64` /
/// `Complex<f64>`. Each config's `Default` reproduces librosa 1.0's defaults.
/// See the [module documentation](self) for the librosa correspondence.
///
/// The existing [`AudioTransforms`](crate::AudioTransforms) spectrogram methods are
/// unchanged; they follow the `spectrograms` crate's conventions instead.
pub trait AudioSpectrum {
    /// Computes the complex STFT of every channel, equivalent to `librosa.stft`.
    ///
    /// # Arguments
    ///
    /// * `config` - STFT parameters ([`StftConfig::default`] = librosa defaults).
    ///
    /// # Returns
    ///
    /// A `(channels, n_fft / 2 + 1, n_frames)` complex array; see [`stft`].
    ///
    /// # Errors
    ///
    /// [`ParameterError`] if the configuration is invalid, or `center` is `false` and the
    /// signal is shorter than `n_fft`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use audio_samples::operations::spectrum::{AudioSpectrum, StftConfig};
    /// use audio_samples::{AudioSamples, sample_rate};
    /// use ndarray::Array2;
    ///
    /// let stereo = AudioSamples::new_multi_channel(Array2::<f32>::zeros((2, 22050)), sample_rate!(22050))
    ///     .unwrap();
    /// let s = stereo.stft_matrix(&StftConfig::default()).unwrap();
    /// assert_eq!(s.dim(), (2, 1025, 44));
    /// ```
    fn stft_matrix(&self, config: &StftConfig) -> AudioSampleResult<Array3<Complex<f64>>>;

    /// Inverts an STFT matrix to audio, equivalent to `librosa.istft`.
    ///
    /// # Arguments
    ///
    /// * `stft` - `(channels, n_fft / 2 + 1, n_frames)` complex array.
    /// * `config` - The STFT parameters used for analysis.
    /// * `sample_rate` - Sample rate of the returned audio.
    /// * `length` - Exact output length in samples per channel (pass the original
    ///   length to undo centring and padding exactly); `None` keeps every frame.
    ///
    /// # Returns
    ///
    /// Mono `f64` audio for a one-channel matrix, multi-channel audio otherwise.
    ///
    /// # Errors
    ///
    /// See [`istft`].
    ///
    /// # Examples
    ///
    /// ```rust
    /// use audio_samples::operations::spectrum::{AudioSpectrum, StftConfig};
    /// use audio_samples::{AudioSamples, sample_rate, sine_wave};
    /// use std::time::Duration;
    ///
    /// let x = sine_wave::<f64>(220.0, Duration::from_millis(200), sample_rate!(22050), 0.5);
    /// let cfg = StftConfig::default();
    /// let s = x.stft_matrix(&cfg).unwrap();
    /// let y = AudioSamples::<f64>::istft_matrix(&s, &cfg, sample_rate!(22050), Some(x.samples_per_channel()))
    ///     .unwrap();
    /// let (a, b) = (x.as_mono().unwrap().as_view(), y.as_mono().unwrap().as_view());
    /// assert!(a.iter().zip(b.iter()).all(|(p, q)| (p - q).abs() < 1e-12));
    /// ```
    fn istft_matrix(
        stft: &Array3<Complex<f64>>,
        config: &StftConfig,
        sample_rate: SampleRate,
        length: Option<NonZeroUsize>,
    ) -> AudioSampleResult<AudioSamples<'static, f64>>
    where
        Self: Sized;

    /// Computes `|STFT|^power` for every channel, equivalent to
    /// `numpy.abs(librosa.stft(y, ...)) ** power`.
    ///
    /// # Arguments
    ///
    /// * `config` - STFT parameters.
    /// * `power` - Exponent: `1.0` for magnitude, `2.0` for power. Finite and positive.
    ///
    /// # Returns
    ///
    /// A `(channels, n_fft / 2 + 1, n_frames)` array.
    ///
    /// # Errors
    ///
    /// [`ParameterError`] if `power` is not finite and positive, or as for
    /// [`stft_matrix`](Self::stft_matrix).
    ///
    /// # Examples
    ///
    /// ```rust
    /// use audio_samples::operations::spectrum::{AudioSpectrum, StftConfig};
    /// use audio_samples::{sample_rate, sine_wave};
    /// use std::time::Duration;
    ///
    /// let x = sine_wave::<f64>(1000.0, Duration::from_millis(100), sample_rate!(16000), 1.0);
    /// let p = x.spectrogram_matrix(&StftConfig::default(), 2.0).unwrap();
    /// assert_eq!(p.dim().1, 1025);
    /// assert!(p.iter().all(|&v| v >= 0.0));
    /// ```
    fn spectrogram_matrix(&self, config: &StftConfig, power: f64)
    -> AudioSampleResult<Array3<f64>>;

    /// Computes a mel spectrogram, equivalent to `librosa.feature.melspectrogram`.
    ///
    /// # Arguments
    ///
    /// * `config` - STFT, filterbank and exponent ([`MelSpectrogramConfig::default`] =
    ///   librosa defaults: power 2, 128 Slaney bands, hop 512).
    ///
    /// # Returns
    ///
    /// A `(channels, n_mels, n_frames)` array.
    ///
    /// # Errors
    ///
    /// [`ParameterError`] if the configuration is invalid for the signal's sample rate.
    ///
    /// # Differences from librosa
    ///
    /// The filterbank is `f64`; librosa's default builds it in `float32`, which changes
    /// results by about 1e-7 relative (pass `dtype=numpy.float64` to librosa for
    /// round-off agreement).
    ///
    /// # Examples
    ///
    /// ```rust
    /// use audio_samples::operations::spectrum::{AudioSpectrum, MelSpectrogramConfig};
    /// use audio_samples::{sample_rate, sine_wave};
    /// use std::time::Duration;
    ///
    /// let x = sine_wave::<f32>(440.0, Duration::from_secs(1), sample_rate!(22050), 0.5);
    /// let m = x.melspectrogram(&MelSpectrogramConfig::default()).unwrap();
    /// assert_eq!(m.dim(), (1, 128, 44));
    /// ```
    fn melspectrogram(&self, config: &MelSpectrogramConfig) -> AudioSampleResult<Array3<f64>>;

    /// Computes MFCCs, equivalent to `librosa.feature.mfcc(y=..., sr=...)`.
    ///
    /// The pipeline is [`melspectrogram`](Self::melspectrogram), then [`power_to_db`]
    /// with `config.db` (per channel, as librosa's `axes="auto"`), then
    /// [`mfcc_from_mel_db`].
    ///
    /// # Arguments
    ///
    /// * `config` - MFCC parameters ([`MfccConfig::default`] = librosa defaults).
    ///
    /// # Returns
    ///
    /// A `(channels, n_mfcc, n_frames)` array.
    ///
    /// # Errors
    ///
    /// [`ParameterError`] if the configuration is invalid for the signal's sample rate.
    ///
    /// # Differences from librosa
    ///
    /// As for [`melspectrogram`](Self::melspectrogram) (the mel basis is `f64`). This is a
    /// different method from [`AudioTransforms::mfcc`](crate::AudioTransforms::mfcc),
    /// which keeps the `spectrograms` crate's HTK-like conventions.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use audio_samples::nzu;
    /// use audio_samples::operations::spectrum::{AudioSpectrum, MfccConfig};
    /// use audio_samples::{sample_rate, sine_wave};
    /// use std::time::Duration;
    ///
    /// let x = sine_wave::<f32>(440.0, Duration::from_secs(1), sample_rate!(22050), 0.5);
    /// let m = x.mfcc_frames(&MfccConfig::new().with_n_mfcc(nzu!(13)).with_lifter(22.0)).unwrap();
    /// assert_eq!(m.dim(), (1, 13, 44));
    /// ```
    fn mfcc_frames(&self, config: &MfccConfig) -> AudioSampleResult<Array3<f64>>;
}

/// The audio as a `(channels, samples)` `f64` array.
fn audio_f64_rows<T: StandardSample>(audio: &AudioSamples<'_, T>) -> Array2<f64> {
    let w = audio.to_format::<f64>();
    if let Some(m) = w.as_mono() {
        m.as_view().insert_axis(Axis(0)).to_owned()
    } else if let Some(m) = w.as_multi_channel() {
        m.as_view().to_owned()
    } else {
        unreachable!("audio is either mono or multi-channel")
    }
}

impl<T> AudioSpectrum for AudioSamples<'_, T>
where
    T: StandardSample,
{
    fn stft_matrix(&self, config: &StftConfig) -> AudioSampleResult<Array3<Complex<f64>>> {
        stft(audio_f64_rows(self).view(), config)
    }

    fn istft_matrix(
        stft: &Array3<Complex<f64>>,
        config: &StftConfig,
        sample_rate: SampleRate,
        length: Option<NonZeroUsize>,
    ) -> AudioSampleResult<AudioSamples<'static, f64>> {
        let y = istft(stft.view(), config, length.map(NonZeroUsize::get))?;
        if y.nrows() == 1 {
            AudioSamples::new_mono(y.index_axis_move(Axis(0), 0), sample_rate)
        } else {
            AudioSamples::new_multi_channel(y, sample_rate)
        }
    }

    fn spectrogram_matrix(
        &self,
        config: &StftConfig,
        power: f64,
    ) -> AudioSampleResult<Array3<f64>> {
        validate_power(power)?;
        let s = self.stft_matrix(config)?;
        Ok(s.mapv(|z| magnitude_power(z, power)))
    }

    fn melspectrogram(&self, config: &MelSpectrogramConfig) -> AudioSampleResult<Array3<f64>> {
        let sr = self.sample_rate_hz();
        config.validate(sr)?;
        let s = self.spectrogram_matrix(&config.stft, config.power)?;
        melspectrogram_from_spectrogram(s.view(), sr, config.stft.n_fft, &config.mel)
    }

    fn mfcc_frames(&self, config: &MfccConfig) -> AudioSampleResult<Array3<f64>> {
        config.validate(self.sample_rate_hz())?;
        let mel = self.melspectrogram(&config.mel_spectrogram)?;
        let mel_db = power_to_db(&mel, &config.db)?;
        mfcc_from_mel_db(&mel_db, config)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sample_rate;
    use ndarray::{Array1, array};

    fn noise(n: usize, seed: u64) -> Vec<f64> {
        // Small deterministic LCG; good enough for round-trip tests.
        let mut s = seed;
        (0..n)
            .map(|_| {
                s = s
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                ((s >> 11) as f64 / (1u64 << 53) as f64) - 0.5
            })
            .collect()
    }

    #[test]
    fn stft_frame_counts_match_librosa_formula() {
        for &len in &[1usize, 100, 2047, 2048, 2049, 5000] {
            let x = Array2::from_shape_vec((1, len), noise(len, 1)).unwrap();
            let s = stft(x.view(), &StftConfig::default()).unwrap();
            assert_eq!(s.dim(), (1, 1025, 1 + len / 512), "len {len}");
        }
        let x = Array2::from_shape_vec((1, 5000), noise(5000, 2)).unwrap();
        let s = stft(x.view(), &StftConfig::default().with_center(false)).unwrap();
        assert_eq!(s.dim().2, 1 + (5000 - 2048) / 512);
        let short = Array2::from_shape_vec((1, 100), noise(100, 3)).unwrap();
        assert!(stft(short.view(), &StftConfig::default().with_center(false)).is_err());
    }

    #[test]
    fn stft_of_dc_has_energy_only_in_bin_zero() {
        let x = Array2::from_elem((1, 4096), 1.0);
        let s = stft(x.view(), &StftConfig::default().with_center(false)).unwrap();
        // Periodic Hann sums to n_fft / 2.
        assert!((s[[0, 0, 0]].re - 1024.0).abs() < 1e-9);
        assert!(s[[0, 1, 0]].norm() > 0.0);
        assert!(s[[0, 5, 0]].norm() < 1e-9);
    }

    #[test]
    fn istft_round_trip_all_pad_modes_and_windows() {
        let len = 7000;
        let x = Array2::from_shape_vec((2, len), noise(2 * len, 4)).unwrap();
        let configs = [
            StftConfig::default(),
            StftConfig::default().with_pad_mode(StftPadMode::Reflect),
            StftConfig::default()
                .with_n_fft(nzu!(1024))
                .with_win_length(Some(nzu!(600))),
            StftConfig::default()
                .with_n_fft(nzu!(1023))
                .with_hop_length(Some(nzu!(200))),
            StftConfig::default()
                .with_window(Window::Hamming)
                .with_center(false),
        ];
        for cfg in &configs {
            let s = stft(x.view(), cfg).unwrap();
            let y = istft(s.view(), cfg, Some(len)).unwrap();
            assert_eq!(y.dim(), (2, len));
            // Uncentred analysis cannot reconstruct the first/last partial frames.
            let (lo, hi) = if cfg.center {
                (0, len)
            } else {
                (1024, len - 2048)
            };
            for c in 0..2 {
                for i in lo..hi {
                    assert!((y[[c, i]] - x[[c, i]]).abs() < 1e-10, "{cfg:?} c{c} i{i}");
                }
            }
        }
    }

    #[test]
    fn istft_length_handling() {
        let x = Array2::from_shape_vec((1, 5000), noise(5000, 5)).unwrap();
        let cfg = StftConfig::default();
        let s = stft(x.view(), &cfg).unwrap();
        let y = istft(s.view(), &cfg, None).unwrap();
        assert_eq!(y.ncols(), 2048 + 512 * (s.dim().2 - 1) - 2048);
        let longer = istft(s.view(), &cfg, Some(6000)).unwrap();
        assert_eq!(longer.ncols(), 6000);
        assert!(
            longer
                .slice(ndarray::s![0, 5632..])
                .iter()
                .all(|&v| v == 0.0)
        );
        assert!(istft(s.view(), &cfg, Some(0)).is_err());
        assert!(istft(s.view(), &cfg.with_n_fft(nzu!(1024)), None).is_err());
    }

    #[test]
    fn pad_modes_match_numpy() {
        let x = [1.0, 2.0, 3.0];
        assert_eq!(
            pad_signal(&x, 4, StftPadMode::Constant),
            vec![0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 0.0]
        );
        assert_eq!(
            pad_signal(&x, 4, StftPadMode::Edge),
            vec![1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 3.0, 3.0, 3.0, 3.0, 3.0]
        );
        // numpy.pad([1, 2, 3], 4, mode="reflect") = [1 2 3 2 1 2 3 2 1 2 3]
        assert_eq!(
            pad_signal(&x, 4, StftPadMode::Reflect),
            vec![1.0, 2.0, 3.0, 2.0, 1.0, 2.0, 3.0, 2.0, 1.0, 2.0, 3.0]
        );
        // numpy.pad([1, 2, 3], 4, mode="symmetric") = [3 3 2 1 1 2 3 3 2 1 1]
        assert_eq!(
            pad_signal(&x, 4, StftPadMode::Symmetric),
            vec![3.0, 3.0, 2.0, 1.0, 1.0, 2.0, 3.0, 3.0, 2.0, 1.0, 1.0]
        );
        assert_eq!(pad_signal(&[5.0], 2, StftPadMode::Reflect), vec![5.0; 5]);
    }

    #[test]
    fn trait_methods_mono_and_stereo() {
        let mono =
            AudioSamples::new_mono(Array1::from(noise(8000, 6)), sample_rate!(16000)).unwrap();
        let cfg = StftConfig::default().with_n_fft(nzu!(512));
        let s = mono.stft_matrix(&cfg).unwrap();
        let y = AudioSamples::<f64>::istft_matrix(&s, &cfg, sample_rate!(16000), Some(nzu!(8000)))
            .unwrap();
        assert_eq!(y.num_channels().get(), 1);
        let stereo = AudioSamples::new_multi_channel(
            Array2::from_shape_vec((2, 8000), noise(16000, 7)).unwrap(),
            sample_rate!(16000),
        )
        .unwrap();
        let s2 = stereo.stft_matrix(&cfg).unwrap();
        assert_eq!(s2.dim().0, 2);
        let y2 = AudioSamples::<f64>::istft_matrix(&s2, &cfg, sample_rate!(16000), None).unwrap();
        assert_eq!(y2.num_channels().get(), 2);

        let mel = stereo
            .melspectrogram(
                &MelSpectrogramConfig::default()
                    .with_mel(MelFilterbankConfig::new().with_n_mels(nzu!(40))),
            )
            .unwrap();
        assert_eq!(mel.dim(), (2, 40, 16));
        let mfcc = stereo
            .mfcc_frames(
                &MfccConfig::default().with_mel_spectrogram(
                    MelSpectrogramConfig::default()
                        .with_mel(MelFilterbankConfig::new().with_n_mels(nzu!(40))),
                ),
            )
            .unwrap();
        assert_eq!(mfcc.dim(), (2, 20, 16));
        assert!(mono.spectrogram_matrix(&cfg, 0.0).is_err());
        assert!(
            mono.mfcc_frames(&MfccConfig::default().with_n_mfcc(nzu!(200)))
                .is_err()
        );
    }

    #[test]
    fn power_to_db_top_db_is_per_channel_with_auto_axes() {
        let s = array![[[1.0, 1e-12]], [[1e-6, 1e-12]]];
        let db = power_to_db(&s, &DbConfig::default()).unwrap();
        assert_eq!(db[[0, 0, 1]], -80.0);
        // Channel 1: max is -60 dB, so 1e-12 (clamped to amin = 1e-10, -100 dB) is not floored.
        assert!((db[[1, 0, 1]] + 100.0).abs() < 1e-9);
        let all = power_to_db(&s, &DbConfig::default().with_axes(DbAxes::All)).unwrap();
        assert_eq!(all[[1, 0, 1]], -80.0);
        assert!(power_to_db(&s, &DbConfig::default().with_amin(0.0)).is_err());
        assert!(power_to_db(&s, &DbConfig::default().with_top_db(Some(-1.0))).is_err());
    }

    #[test]
    fn db_round_trips() {
        let p = array![1e-3, 0.5, 2.0];
        let back = db_to_power(
            &power_to_db(&p, &DbConfig::default().with_top_db(None)).unwrap(),
            1.0,
        );
        for (a, b) in back.iter().zip(&p) {
            assert!((a / b - 1.0).abs() < 1e-12);
        }
        let a = array![1e-3, 0.5, 2.0];
        let back = db_to_amplitude(
            &amplitude_to_db(&a, &DbConfig::amplitude().with_top_db(None)).unwrap(),
            1.0,
        );
        for (x, y) in back.iter().zip(&a) {
            assert!((x / y - 1.0).abs() < 1e-12);
        }
    }

    #[test]
    fn dct_matrices_are_orthonormal_with_ortho_norm() {
        for t in [DctType::Two, DctType::Three, DctType::One] {
            let m = dct_matrix(8, 8, t, DctNorm::Ortho);
            let i = m.dot(&m.t());
            for r in 0..8 {
                for c in 0..8 {
                    let e = if r == c { 1.0 } else { 0.0 };
                    assert!((i[[r, c]] - e).abs() < 1e-12, "{t:?}");
                }
            }
        }
    }

    #[test]
    fn delta_of_polynomials() {
        let quad = Array2::from_shape_fn((1, 15), |(_, t)| (t as f64).powi(2));
        let d1 = delta(&quad, &DeltaConfig::new().with_width(5)).unwrap();
        // First derivative of t^2 with a linear fit is exact in the interior (2t).
        for t in 2..13 {
            assert!((d1[[0, t]] - 2.0 * t as f64).abs() < 1e-9);
        }
        let d2 = delta(&quad, &DeltaConfig::new().with_order(2)).unwrap();
        assert!(d2.iter().all(|v| (v - 2.0).abs() < 1e-9));
        assert!(delta(&quad, &DeltaConfig::new().with_width(4)).is_err());
        assert!(delta(&quad, &DeltaConfig::new().with_width(17)).is_err());
        assert!(
            delta(
                &quad,
                &DeltaConfig::new()
                    .with_width(17)
                    .with_mode(DeltaMode::Nearest)
            )
            .is_ok()
        );
        assert!(delta(&quad, &DeltaConfig::new().with_axis(2)).is_err());
        // axis 0 has a single entry; nearest mode makes the derivative zero.
        let d0 = delta(
            &quad,
            &DeltaConfig::new()
                .with_axis(0)
                .with_width(3)
                .with_mode(DeltaMode::Nearest),
        )
        .unwrap();
        assert!(d0.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn stack_memory_negative_delay_and_large_shift() {
        let x = array![[1.0, 2.0, 3.0]];
        let s = stack_memory(&x, nzu!(3), -1).unwrap();
        assert_eq!(s, array![[1.0, 2.0, 3.0], [2.0, 3.0, 0.0], [3.0, 0.0, 0.0]]);
        let big = stack_memory(&x, nzu!(2), 10).unwrap();
        assert_eq!(big.row(1).to_vec(), vec![0.0; 3]);
        assert!(stack_memory(&x, nzu!(2), 0).is_err());
        assert!(stack_memory(&array![1.0, 2.0], nzu!(2), 1).is_err());
    }

    #[test]
    fn frame_matches_librosa_count() {
        let x: Vec<f64> = (0..5000).map(f64::from).collect();
        let f = frame(&x, nzu!(2048), nzu!(512)).unwrap();
        assert_eq!(f.dim(), (2048, 1 + (5000 - 2048) / 512));
        assert!(frame(&x[..10], nzu!(11), nzu!(1)).is_err());
    }
}
