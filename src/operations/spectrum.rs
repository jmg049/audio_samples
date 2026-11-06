//! Spectral analysis: STFT / ISTFT, dB scaling, mel spectrograms, MFCCs and
//! frame-wise feature utilities.
//!
//! ## What
//!
//! This module provides the [`AudioSpectrum`] trait and the free functions it is
//! built from:
//!
//! | Operation | Items |
//! |---|---|
//! | Complex STFT and its inverse | [`AudioSpectrum::stft_matrix`] / [`AudioSpectrum::istft_matrix`] ([`stft`], [`istft`]) |
//! | Magnitude spectrogram raised to a power | [`AudioSpectrum::spectrogram_matrix`] |
//! | Mel spectrogram | [`AudioSpectrum::melspectrogram`] ([`melspectrogram_from_spectrogram`]) |
//! | MFCC | [`AudioSpectrum::mfcc_frames`] ([`mfcc_from_mel_db`]) |
//! | Power and amplitude to dB and back | [`power_to_db`], [`amplitude_to_db`], [`db_to_power`], [`db_to_amplitude`] |
//! | Derivatives and time-delay stacking | [`delta`], [`stack_memory`] |
//! | Overlapping frames of a signal | [`frame`] |
//!
//! Windows and mel filterbanks live in [`crate::operations::filterbanks`];
//! unit conversions (mel scales, frames/samples/time) in
//! [`crate::utils::audio_math`].
//!
//! ## Why
//!
//! The spectrogram methods of [`AudioTransforms`](crate::AudioTransforms) delegate
//! to the `spectrograms` crate, whose conventions differ (symmetric windows, an
//! absolute dB floor, an unnormalised DCT, mono only). This module uses periodic
//! windows, dB relative to a configurable reference with an optional `top_db`
//! floor, an orthonormal DCT, and any number of channels. Every config struct
//! has a `librosa()` constructor ([`DbConfig`] has `librosa_power()` and
//! `librosa_amplitude()`); only the [`MfccConfig`] and [`DbConfig`] defaults
//! differ from them, in having no 80 dB floor.
//!
//! ## How
//!
//! Bring [`AudioSpectrum`] into scope and call it on any [`AudioSamples`].
//! Outputs are arrays shaped `(channels, n_features, n_frames)`; mono input gives
//! one channel.
//!
//! The output precision follows the input: the element type is the sample
//! type's [`AudioSample::Float`](crate::AudioSample::Float), the smallest float
//! type that holds every input sample exactly. `u8`, `i16`, `I24` and `f32`
//! audio gives `f32` (or `Complex<f32>`); `i32` and `f64` audio gives `f64`.
//! Samples are converted with the crate's audio-aware conversion and the whole
//! computation runs in that type, so `f32` analysis uses half the memory of
//! `f64`. Parameters in the config structs stay `f64` and are rounded to the
//! output type once; filter tables (windows, mel bases, DCT matrices) are
//! computed in `f64` and rounded once. [`AudioSpectrum::istft_matrix`] returns
//! audio of the receiver's sample type.
//!
//! The free functions on bare arrays ([`stft`], [`power_to_db`], [`delta`],
//! ...) are generic over the element type ([`AudioFloat`]: `f32` or `f64`) and
//! return arrays of the same type.
//!
//! ```rust
//! use audio_samples::operations::spectrum::{AudioSpectrum, MfccConfig, StftConfig};
//! use audio_samples::{AudioSamples, sample_rate, sine_wave};
//! use std::time::Duration;
//!
//! let audio = sine_wave::<f32>(440.0, Duration::from_millis(500), sample_rate!(22050), 0.5);
//!
//! // (channels, bins, frames) complex. f32 audio gives Complex<f32>.
//! let stft = audio.stft_matrix(&StftConfig::default()).unwrap();
//! assert_eq!(stft.dim(), (1, 1025, 22));
//! let _: &ndarray::Array3<num_complex::Complex<f32>> = &stft;
//!
//! // The inverse with the original length reconstructs the signal.
//! let n = audio.samples_per_channel();
//! let y = AudioSamples::<f32>::istft_matrix(&stft, &StftConfig::default(), sample_rate!(22050), Some(n))
//!     .unwrap();
//! assert_eq!(y.samples_per_channel(), n);
//!
//! // 20 coefficients per frame: (1, 20, 22).
//! let mfcc = audio.mfcc_frames(&MfccConfig::default()).unwrap();
//! assert_eq!(mfcc.dim(), (1, 20, 22));
//! ```

use std::num::NonZeroUsize;
use std::sync::Arc;

use ndarray::{
    Array, Array2, Array3, ArrayBase, ArrayView2, ArrayView3, ArrayViewMutD, Axis, Data, Dimension,
    Slice, Zip,
};
use num_complex::Complex;
use realfft::{ComplexToReal, RealToComplex};

use crate::audio_float::{finite_float_rows, float_rows_to_audio};
use crate::operations::filterbanks::{
    MelFilterbankConfig, Window, mel_filterbank_bands, padded_window, window_sumsquare_from_squared,
};
use crate::repr::SampleRate;
use crate::utils::{MAX_SIGNAL_ELEMENTS, check_derived_len};
use crate::{
    AudioFloat, AudioSampleError, AudioSampleResult, AudioSamples, ParameterError, StandardSample,
};

/// The forward real-FFT plan of length `n` from this thread's planner for `F`.
///
/// realfft caches plans (twiddle factors) by length, so repeated transforms of
/// the same size reuse them. Callers fetch a plan once per call and process
/// every frame with their own scratch buffers.
pub(crate) fn forward_plan<F: AudioFloat>(n: usize) -> Arc<dyn RealToComplex<F>> {
    F::with_real_fft_planner(|p| p.plan_fft_forward(n))
}

/// The inverse real-FFT plan of length `n` from this thread's planner for `F`.
fn inverse_plan<F: AudioFloat>(n: usize) -> Arc<dyn ComplexToReal<F>> {
    F::with_real_fft_planner(|p| p.plan_fft_inverse(n))
}

/// Frames per block in [`stft`] (see the comment there).
const STFT_BLOCK: usize = 16;

fn param_err(parameter: &str, reason: impl ToString) -> AudioSampleError {
    AudioSampleError::Parameter(ParameterError::invalid_value(parameter, reason))
}

// =============================================================================
// STFT configuration
// =============================================================================

/// How the signal is extended at both ends when frames are centred.
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
    /// Zero padding. The default.
    #[default]
    Constant,
    /// Mirror without repeating the edge sample: `d c b | a b c d | c b a`.
    Reflect,
    /// Repeat the edge sample: `a a a | a b c d | d d d`.
    Edge,
    /// Mirror including the edge sample: `c b a | a b c d | d c b`.
    Symmetric,
}

/// Configuration of a short-time Fourier transform.
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
/// [`MelSpectrogramConfig`]. `Default` is
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
    /// Default: 2048.
    pub n_fft: NonZeroUsize,
    /// Samples between the starts of successive frames. `None` means
    /// `win_length / 4` (the default; 512 for the default `n_fft`). Smaller
    /// hops give more frames (finer time resolution) at a higher cost.
    pub hop_length: Option<NonZeroUsize>,
    /// Length of the window in samples, `<= n_fft`. `None` means `n_fft`. A shorter
    /// window is zero-padded symmetrically to `n_fft`, trading frequency
    /// resolution for time resolution.
    pub win_length: Option<NonZeroUsize>,
    /// Analysis window, computed in its periodic form. Default: Hann.
    pub window: Window,
    /// If `true` (the default) the signal is padded by `n_fft / 2` samples at
    /// both ends so that frame `t` is centred on sample `t * hop_length`. If
    /// `false`, frame `t` starts at sample `t * hop_length` and the signal must be
    /// at least `n_fft` samples long.
    pub center: bool,
    /// How the signal is extended when `center` is `true`. Default:
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
    /// Returns the default configuration (same as [`Default`]).
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Settings matching librosa 1.0's defaults; currently the same as [`Default`].
    #[must_use]
    pub fn librosa() -> Self {
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

    /// The periodic window zero-padded to `n_fft` (computed in `f64`, rounded
    /// to `F`), plus the hop length.
    fn analysis_window<F: AudioFloat>(&self) -> AudioSampleResult<(Vec<F>, usize)> {
        let hop = self.effective_hop_length()?.get();
        let w = padded_window(&self.window, self.win_length, self.n_fft)?;
        Ok((w.iter().map(|&v| F::of(v)).collect(), hop))
    }
}

// =============================================================================
// STFT / ISTFT on raw arrays
// =============================================================================

/// Source index into a signal of length `len` for padded position `i` (may be
/// negative or `>= len`). The mirror modes repeat periodically, so padding longer
/// than the signal keeps reflecting.
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

/// Pads `x` by `pad` samples at both ends with `mode` (the reference for the
/// index-based padding of [`padded_frame_into`]).
#[cfg(test)]
pub(crate) fn pad_signal<F: AudioFloat>(x: &[F], pad: usize, mode: StftPadMode) -> Vec<F> {
    let mut out = vec![F::zero(); x.len() + 2 * pad];
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

/// Computes the STFT of every row of `signal` (shape `(channels, samples)`).
///
/// This is the array-level function behind [`AudioSpectrum::stft_matrix`]; use it
/// when the signal is not held in an [`AudioSamples`].
///
/// The transform runs in the element type `F` of `signal` (`f32` or `f64`); the
/// window is computed in `f64` and rounded to `F` once.
///
/// # Arguments
///
/// * `signal` - `(channels, samples)` array of samples (at least one sample).
/// * `config` - STFT parameters.
///
/// # Returns
///
/// A complex array of element type `Complex<F>` and shape `(channels, n_fft / 2 + 1, n_frames)` where
/// `n_frames = 1 + (len + 2 * (n_fft / 2) - n_fft) / hop` when centred and
/// `1 + (len - n_fft) / hop` otherwise. Frame `t`, bin `k` is
/// `sum_n w[n] x[t * hop + n - pad] exp(-2 pi i k n / n_fft)`.
///
/// # Errors
///
/// - [`ParameterError`] if the configuration is invalid ([`StftConfig::validate`]),
///   the signal is empty, or `center` is `false` and the signal is shorter than `n_fft`.
/// - [`ParameterError`] if `n_fft` exceeds 2^31 or the STFT matrix would exceed
///   2^31 elements.
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
/// discrete Fourier transform," *IEEE Trans. ASSP* 25(3), 1977.
pub fn stft<F: AudioFloat>(
    signal: ArrayView2<'_, F>,
    config: &StftConfig,
) -> AudioSampleResult<Array3<Complex<F>>> {
    stft_map(signal, config, |z| z)
}

/// The number of frames [`stft`] gives for `samples` samples per channel.
///
/// # Arguments
///
/// * `samples` - Samples per channel.
/// * `config` - STFT parameters.
///
/// # Returns
///
/// `1 + (samples + 2 * (n_fft / 2) - n_fft) / hop` when centred and
/// `1 + (samples - n_fft) / hop` otherwise.
///
/// # Errors
///
/// [`ParameterError`] if the hop length is invalid, or `center` is `false` and the signal
/// is shorter than `n_fft`.
///
/// # Examples
///
/// ```rust
/// use audio_samples::nzu;
/// use audio_samples::operations::spectrum::{StftConfig, stft_n_frames};
///
/// let frames = stft_n_frames(4096, &StftConfig::new().with_n_fft(nzu!(512))).unwrap();
/// assert_eq!(frames, 33);
/// ```
pub fn stft_n_frames(samples: usize, config: &StftConfig) -> AudioSampleResult<usize> {
    let hop = config.effective_hop_length()?.get();
    let n_fft = config.n_fft.get();
    let padded_len = samples + if config.center { 2 * (n_fft / 2) } else { 0 };
    if padded_len < n_fft {
        return Err(param_err(
            "n_fft",
            format!(
                "n_fft={n_fft} is too large for uncentred analysis of a signal of length {samples}"
            ),
        ));
    }
    Ok(1 + (padded_len - n_fft) / hop)
}

/// [`stft`] written into a given array, which needs no memory for the matrix.
///
/// # Arguments
///
/// * `signal` - `(channels, samples)` array of samples (at least one sample).
/// * `config` - STFT parameters.
/// * `out` - Of shape `(channels, n_fft / 2 + 1, n_frames)`, in any memory layout, with
///   `n_frames` from [`stft_n_frames`].
///
/// # Errors
///
/// As [`stft`], and [`ParameterError`] if `out` has another shape.
///
/// # Examples
///
/// ```rust
/// use audio_samples::nzu;
/// use audio_samples::operations::spectrum::{StftConfig, stft, stft_into};
/// use ndarray::{Array2, Array3};
/// use num_complex::Complex;
///
/// let x = Array2::from_shape_fn((2, 4096), |(c, n)| ((n * (c + 1)) as f64 * 0.01).sin());
/// let config = StftConfig::new().with_n_fft(nzu!(512));
/// let mut out = Array3::<Complex<f64>>::zeros((2, 257, 33));
/// stft_into(x.view(), &config, out.view_mut()).unwrap();
/// assert_eq!(out, stft(x.view(), &config).unwrap());
/// ```
pub fn stft_into<F: AudioFloat>(
    signal: ArrayView2<'_, F>,
    config: &StftConfig,
    mut out: ndarray::ArrayViewMut3<'_, Complex<F>>,
) -> AudioSampleResult<()> {
    let frames = StftFrames::new(signal, config)?;
    let (n_channels, n_bins, n_frames) = (signal.nrows(), frames.n_bins, frames.n_frames);
    if out.dim() != (n_channels, n_bins, n_frames) {
        return Err(param_err(
            "out",
            format!(
                "has shape {:?}, expected {:?}",
                out.dim(),
                (n_channels, n_bins, n_frames)
            ),
        ));
    }
    frames.for_each_block(signal, config, |c, t0, nb, block| {
        for (k, mut dst) in out.index_axis_mut(Axis(0), c).outer_iter_mut().enumerate() {
            let mut dst = dst.slice_mut(ndarray::s![t0..t0 + nb]);
            for (j, d) in dst.iter_mut().enumerate() {
                *d = block[j * n_bins + k];
            }
        }
    })
}

/// Fills `frame` with the windowed padded frame that starts at signal index
/// `start` (negative or running past the end where the frame overlaps the
/// padding), with the padding of `mode`.
#[inline]
pub(crate) fn padded_frame_into<F: AudioFloat>(
    x: &[F],
    start: isize,
    window: Option<&[F]>,
    mode: StftPadMode,
    frame: &mut [F],
) {
    let n = frame.len();
    if start >= 0 && start as usize + n <= x.len() {
        let seg = &x[start as usize..start as usize + n];
        match window {
            Some(w) => {
                for ((dst, &s), &wv) in frame.iter_mut().zip(seg).zip(w) {
                    *dst = wv * s;
                }
            }
            None => frame.copy_from_slice(seg),
        }
        return;
    }
    for (i, dst) in frame.iter_mut().enumerate() {
        let s = pad_source_index(start + i as isize, x.len(), mode).map_or(F::zero(), |j| x[j]);
        *dst = match window {
            Some(w) => w[i] * s,
            None => s,
        };
    }
}

/// The sum of squares of every frame of a padded signal: for frame `t`,
/// `sum_{k < frame_length} xp[t * hop + k]^2` where `xp` is `x` padded by
/// `pad` samples at both ends with `mode`, summed in order of increasing `k`
/// starting from zero (the order of a sequential sum).
///
/// Frames are processed in tiles. The squared samples of a tile are written
/// once into a small `(hop, width)` buffer whose row `r` holds the samples
/// at offsets `q * hop + r`; frame `t` then reads its `k`-th square from row
/// `k % hop` at column `t + k / hop`, so the inner loop runs over a
/// contiguous run of frames and vectorises across frames while every frame
/// keeps its sequential summation order. Neither the padded signal nor the
/// framed matrix is materialised.
///
/// `n_frames` frames are returned; every frame must lie within the padded
/// signal (`(n_frames - 1) * hop + frame_length <= x.len() + 2 * pad`).
///
/// On x86-64 CPUs with AVX2 the loops are compiled for it (the same
/// operations in the same order, so the same results).
#[cfg(any(feature = "feature-extraction", feature = "effects", test))]
pub(crate) fn frame_square_sums<F: AudioFloat>(
    x: &[F],
    frame_length: usize,
    hop: usize,
    pad: usize,
    mode: StftPadMode,
    n_frames: usize,
) -> Vec<F> {
    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("avx2") {
        // SAFETY: the CPU supports AVX2 (checked just above).
        return unsafe { frame_square_sums_avx2(x, frame_length, hop, pad, mode, n_frames) };
    }
    frame_square_sums_body(x, frame_length, hop, pad, mode, n_frames)
}

/// [`frame_square_sums`] compiled for AVX2.
///
/// # Safety
/// The CPU must support AVX2.
#[cfg(all(
    target_arch = "x86_64",
    any(feature = "feature-extraction", feature = "effects", test)
))]
#[target_feature(enable = "avx2")]
unsafe fn frame_square_sums_avx2<F: AudioFloat>(
    x: &[F],
    frame_length: usize,
    hop: usize,
    pad: usize,
    mode: StftPadMode,
    n_frames: usize,
) -> Vec<F> {
    frame_square_sums_body(x, frame_length, hop, pad, mode, n_frames)
}

#[cfg(any(feature = "feature-extraction", feature = "effects", test))]
#[inline(always)]
fn frame_square_sums_body<F: AudioFloat>(
    x: &[F],
    frame_length: usize,
    hop: usize,
    pad: usize,
    mode: StftPadMode,
    n_frames: usize,
) -> Vec<F> {
    let fl = frame_length;
    let padded_len = x.len() + 2 * pad;
    debug_assert!(n_frames == 0 || (n_frames - 1) * hop + fl <= padded_len);
    let square_at = |i: usize| -> F {
        let s = i as isize - pad as isize;
        let v = if s >= 0 && (s as usize) < x.len() {
            x[s as usize]
        } else if i < padded_len {
            pad_source_index(s, x.len(), mode).map_or(F::zero(), |j| x[j])
        } else {
            F::zero()
        };
        v * v
    };
    let mut out = vec![F::zero(); n_frames];
    if n_frames == 0 {
        return out;
    }
    if hop > fl {
        // Frames do not overlap: sum each one directly.
        for (t, o) in out.iter_mut().enumerate() {
            let mut acc = F::zero();
            for k in 0..fl {
                acc += square_at(t * hop + k);
            }
            *o = acc;
        }
        return out;
    }
    let tile = (32_768 / hop).clamp(1, 64);
    let spread = (fl - 1) / hop; // frame t reads columns t..=t + spread
    let width = tile + spread;
    let mut m = vec![F::zero(); hop * width];
    let mut acc = vec![F::zero(); tile];
    for t0 in (0..n_frames).step_by(tile) {
        let nb = tile.min(n_frames - t0);
        // Squares of xp[(t0 + c) * hop + r] into m[r * width + c].
        for c in 0..nb + spread {
            let base = (t0 + c) * hop;
            let s0 = base as isize - pad as isize;
            if s0 >= 0 && s0 as usize + hop <= x.len() {
                let seg = &x[s0 as usize..s0 as usize + hop];
                for (r, &v) in seg.iter().enumerate() {
                    m[r * width + c] = v * v;
                }
            } else {
                for r in 0..hop {
                    m[r * width + c] = square_at(base + r);
                }
            }
        }
        let acc = &mut acc[..nb];
        acc.fill(F::zero());
        let mut k = 0;
        'outer: for off in 0..=spread {
            for r in 0..hop {
                if k == fl {
                    break 'outer;
                }
                let row = &m[r * width + off..r * width + off + nb];
                for (a, &v) in acc.iter_mut().zip(row) {
                    *a += v;
                }
                k += 1;
            }
        }
        out[t0..t0 + nb].copy_from_slice(acc);
    }
    out
}

/// [`stft`] with `map` applied to every STFT value as it is written, so that
/// a derived quantity (a magnitude, a power) is computed without holding the
/// complex matrix.
///
/// The padding is not materialised: frames that overlap it are assembled
/// from the signal by index, the others read the signal directly, so the
/// memory beyond the output is a few frames.
pub(crate) fn stft_map<F: AudioFloat, O>(
    signal: ArrayView2<'_, F>,
    config: &StftConfig,
    map: impl Fn(Complex<F>) -> O,
) -> AudioSampleResult<Array3<O>> {
    let frames = StftFrames::new(signal, config)?;
    let (n_channels, n_bins, n_frames) = (signal.nrows(), frames.n_bins, frames.n_frames);
    let mut out = Array3::<O>::uninit((n_channels, n_bins, n_frames));
    frames.for_each_block(signal, config, |c, t0, nb, block| {
        let mut out_c = out.index_axis_mut(Axis(0), c);
        for (k, mut dst) in out_c.outer_iter_mut().enumerate() {
            let dst = dst
                .as_slice_mut()
                .expect("rows of a new standard-layout array are contiguous");
            for (j, d) in dst[t0..t0 + nb].iter_mut().enumerate() {
                d.write(map(block[j * n_bins + k]));
            }
        }
    })?;
    // SAFETY: every element was written above: channels `c`, bins `k` and
    // frames `t0..t0 + nb` for consecutive blocks covering `0..n_frames`.
    Ok(unsafe { out.assume_init() })
}

/// The layout of an STFT: the analysis window and the frame and bin counts.
struct StftFrames<F> {
    window: Vec<F>,
    hop: usize,
    pad: usize,
    n_bins: usize,
    n_frames: usize,
}

impl<F: AudioFloat> StftFrames<F> {
    /// Validates the signal and configuration.
    fn new(signal: ArrayView2<'_, F>, config: &StftConfig) -> AudioSampleResult<Self> {
        let (window, hop) = config.analysis_window::<F>()?;
        let n_fft = config.n_fft.get();
        let (n_channels, len) = signal.dim();
        if len == 0 || n_channels == 0 {
            return Err(param_err(
                "signal",
                "cannot compute the STFT of an empty signal",
            ));
        }
        let pad = if config.center { n_fft / 2 } else { 0 };
        let n_frames = stft_n_frames(len, config)?;
        let n_bins = n_fft / 2 + 1;
        check_derived_len(
            "n_fft",
            "the STFT matrix",
            n_channels as f64 * n_bins as f64 * n_frames as f64,
            MAX_SIGNAL_ELEMENTS,
        )?;
        Ok(Self {
            window,
            hop,
            pad,
            n_bins,
            n_frames,
        })
    }

    /// Transforms the frames of every channel in blocks of up to `STFT_BLOCK`
    /// and calls `sink(channel, first frame, frames in the block, block)`. The
    /// block is frame-major, `n_bins` values per frame, and stays in cache, so
    /// the sink can move it to its destination without a whole-signal
    /// transpose, which is several times slower than the FFTs.
    fn for_each_block(
        &self,
        signal: ArrayView2<'_, F>,
        config: &StftConfig,
        mut sink: impl FnMut(usize, usize, usize, &[Complex<F>]),
    ) -> AudioSampleResult<()> {
        let n_bins = self.n_bins;
        let plan = forward_plan::<F>(config.n_fft.get());
        let mut scratch = plan.make_scratch_vec();
        let mut frame_in = vec![F::zero(); config.n_fft.get()];
        let mut block = vec![Complex::new(F::zero(), F::zero()); STFT_BLOCK * n_bins];
        let mut row_buf = Vec::new();
        for (c, row) in signal.outer_iter().enumerate() {
            let x: &[F] = if let Some(s) = row.as_slice() {
                s
            } else {
                row_buf.clear();
                row_buf.extend(row.iter().copied());
                &row_buf
            };
            for t0 in (0..self.n_frames).step_by(STFT_BLOCK) {
                let nb = STFT_BLOCK.min(self.n_frames - t0);
                for (j, spec) in block.chunks_exact_mut(n_bins).take(nb).enumerate() {
                    let start = ((t0 + j) * self.hop) as isize - self.pad as isize;
                    padded_frame_into(x, start, Some(&self.window), config.pad_mode, &mut frame_in);
                    plan.process_with_scratch(&mut frame_in, spec, &mut scratch)
                        .map_err(|e| {
                            AudioSampleError::Processing(crate::ProcessingError::algorithm_failure(
                                "stft",
                                e.to_string(),
                            ))
                        })?;
                }
                sink(c, t0, nb, &block);
            }
        }
        Ok(())
    }
}

/// Reconstructs signals from an STFT matrix.
///
/// Each frame is inverse-FFT'd, windowed with the same (periodic, zero-padded)
/// window and overlap-added; the sum is divided by the window's sum-squared envelope
/// ([`window_sumsquare`](crate::operations::filterbanks::window_sumsquare)) wherever
/// that envelope exceeds the smallest positive normal value of `F`. The
/// computation runs in the element type `F` of `stft_matrix`. With centred frames
/// the `n_fft / 2` padding is trimmed. This is the least-squares inverse of [`stft`]
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
/// A `(channels, length)` array of samples of type `F`.
///
/// # Errors
///
/// [`ParameterError`] if the configuration is invalid, the bin count does not match
/// `n_fft`, there are no frames, the resulting length is 0, or the output would
/// exceed 2^31 samples.
///
/// `config.n_fft` is authoritative: it is not inferred from the bin count, and a
/// mismatch is an error. `length = Some(0)` is rejected.
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
/// transform," *IEEE Trans. ASSP* 32(2), 1984.
pub fn istft<F: AudioFloat>(
    stft_matrix: ArrayView3<'_, Complex<F>>,
    config: &StftConfig,
    length: Option<usize>,
) -> AudioSampleResult<Array2<F>> {
    let (window, hop) = config.analysis_window::<F>()?;
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
    if let Some(len) = length {
        check_derived_len(
            "length",
            "the inverse STFT",
            n_channels as f64 * len as f64,
            MAX_SIGNAL_ELEMENTS,
        )?;
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
    check_derived_len(
        "hop_length",
        "the inverse STFT",
        n_channels as f64 * (n_fft as f64 + hop as f64 * (n_frames - 1) as f64),
        MAX_SIGNAL_ELEMENTS,
    )?;
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

    let win_sq: Vec<F> = window.iter().map(|&w| w * w).collect();
    let wss = window_sumsquare_from_squared(&win_sq, n_frames, hop);
    let scale = F::one() / F::of_usize(n_fft);
    let plan = inverse_plan::<F>(n_fft);
    let mut scratch = plan.make_scratch_vec();
    let mut spec = vec![Complex::new(F::zero(), F::zero()); n_bins];
    let mut frame_out = vec![F::zero(); n_fft];
    let mut ybuf = vec![F::zero(); full_len];
    let mut out = Array2::<F>::zeros((n_channels, out_len));

    for (c, chan) in stft_matrix.outer_iter().enumerate() {
        ybuf.iter_mut().for_each(|v| *v = F::zero());
        for t in 0..n_frames {
            for (dst, src) in spec.iter_mut().zip(chan.column(t)) {
                *dst = *src;
            }
            // The imaginary parts of the DC and (for even n_fft) Nyquist bins are
            // ignored, as a real inverse FFT requires.
            spec[0].im = F::zero();
            if n_fft.is_multiple_of(2) {
                spec[n_bins - 1].im = F::zero();
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
                *y = if norm > F::min_positive_value() {
                    ybuf[src] / norm
                } else {
                    ybuf[src]
                };
            }
        }
    }
    Ok(out)
}

/// Splits a 1-D signal into overlapping frames.
///
/// # Arguments
///
/// * `x` - The signal (`f32` or `f64`).
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
/// [`ParameterError`] if `x` is shorter than `frame_length`, or the frame matrix
/// would exceed 2^31 elements.
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
pub fn frame<F: AudioFloat>(
    x: &[F],
    frame_length: NonZeroUsize,
    hop_length: NonZeroUsize,
) -> AudioSampleResult<Array2<F>> {
    let (fl, hop) = (frame_length.get(), hop_length.get());
    if x.len() < fl {
        return Err(param_err(
            "frame_length",
            format!("input is too short (n={}) for frame_length={fl}", x.len()),
        ));
    }
    let n_frames = 1 + (x.len() - fl) / hop;
    check_derived_len(
        "frame_length",
        "the frame matrix",
        fl as f64 * n_frames as f64,
        MAX_SIGNAL_ELEMENTS,
    )?;
    // Row-major output, filled in blocks of 16 frames: each row of a block
    // reads 16 sequential streams of x and writes 16 contiguous values. The
    // allocation is not zeroed first (a zeroed one is fresh pages that fault
    // on the first write).
    const B: usize = 16;
    let mut out = Array2::<F>::uninit((fl, n_frames));
    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("avx") {
        use std::any::TypeId;
        let o = out.as_slice_mut().expect("standard layout");
        if TypeId::of::<F>() == TypeId::of::<f32>() {
            // SAFETY: F is f32 (checked), so the slices have f32 layout;
            // AVX is available (checked).
            unsafe {
                let xs = std::slice::from_raw_parts(x.as_ptr().cast::<f32>(), x.len());
                frame_copy_avx_f32(xs, fl, hop, n_frames, o.as_mut_ptr().cast::<f32>());
                return Ok(out.assume_init());
            }
        }
        if TypeId::of::<F>() == TypeId::of::<f64>() {
            // SAFETY: as above, for f64.
            unsafe {
                let xs = std::slice::from_raw_parts(x.as_ptr().cast::<f64>(), x.len());
                frame_copy_avx_f64(xs, fl, hop, n_frames, o.as_mut_ptr().cast::<f64>());
                return Ok(out.assume_init());
            }
        }
    }
    {
        let o = out.as_slice_mut().expect("standard layout");
        let empty: &[F] = &[];
        for tb in (0..n_frames).step_by(B) {
            let te = (tb + B).min(n_frames);
            // The frames of the block, each exactly `fl` long.
            let mut cols = [empty; B];
            for (c, t) in cols.iter_mut().zip(tb..te) {
                *c = &x[t * hop..t * hop + fl];
            }
            let cols = &cols[..te - tb];
            // Tiles of R rows: one read of R consecutive samples per frame.
            const R: usize = 8;
            let mut rows = o.chunks_exact_mut(n_frames);
            let mut i = 0;
            while i + R <= fl {
                let mut tile: [&mut [std::mem::MaybeUninit<F>]; R] =
                    std::array::from_fn(|_| rows.next().expect("fl rows"));
                for (k, c) in cols.iter().enumerate() {
                    let vals = &c[i..i + R];
                    for (row, &v) in tile.iter_mut().zip(vals) {
                        row[tb + k] = std::mem::MaybeUninit::new(v);
                    }
                }
                i += R;
            }
            for (row, i) in rows.zip(i..fl) {
                for (v, c) in row[tb..te].iter_mut().zip(cols) {
                    *v = std::mem::MaybeUninit::new(c[i]);
                }
            }
        }
    }
    // SAFETY: every element was written above (each block of frames, every
    // row).
    Ok(unsafe { out.assume_init() })
}

/// The row-major `(fl, n)` frame matrix of `x` (frame `t` starts at
/// `t * hop`) written to `out`, transposing 8 x 8 blocks of frames with AVX.
///
/// # Safety
///
/// AVX must be available; `x` must hold `(n - 1) * hop + fl` samples and
/// `out` must be valid for `fl * n` writes.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
unsafe fn frame_copy_avx_f32(x: &[f32], fl: usize, hop: usize, n: usize, out: *mut f32) {
    use std::arch::x86_64::{
        _mm256_loadu_ps, _mm256_permute2f128_ps, _mm256_shuffle_ps, _mm256_storeu_ps,
        _mm256_unpackhi_ps, _mm256_unpacklo_ps,
    };
    debug_assert!(n == 0 || x.len() >= (n - 1) * hop + fl);
    let p = x.as_ptr();
    let mut t = 0;
    while t + 8 <= n {
        let mut i = 0;
        while i + 8 <= fl {
            // SAFETY: frames t..t + 8 cover samples i..i + 8 (i + 8 <= fl),
            // and rows i..i + 8, columns t..t + 8 are inside `out`.
            unsafe {
                let r = |k: usize| _mm256_loadu_ps(p.add((t + k) * hop + i));
                let (r0, r1, r2, r3) = (r(0), r(1), r(2), r(3));
                let (r4, r5, r6, r7) = (r(4), r(5), r(6), r(7));
                let (t0, t1) = (_mm256_unpacklo_ps(r0, r1), _mm256_unpackhi_ps(r0, r1));
                let (t2, t3) = (_mm256_unpacklo_ps(r2, r3), _mm256_unpackhi_ps(r2, r3));
                let (t4, t5) = (_mm256_unpacklo_ps(r4, r5), _mm256_unpackhi_ps(r4, r5));
                let (t6, t7) = (_mm256_unpacklo_ps(r6, r7), _mm256_unpackhi_ps(r6, r7));
                let s0 = _mm256_shuffle_ps::<0x44>(t0, t2);
                let s1 = _mm256_shuffle_ps::<0xee>(t0, t2);
                let s2 = _mm256_shuffle_ps::<0x44>(t1, t3);
                let s3 = _mm256_shuffle_ps::<0xee>(t1, t3);
                let s4 = _mm256_shuffle_ps::<0x44>(t4, t6);
                let s5 = _mm256_shuffle_ps::<0xee>(t4, t6);
                let s6 = _mm256_shuffle_ps::<0x44>(t5, t7);
                let s7 = _mm256_shuffle_ps::<0xee>(t5, t7);
                let rows = [
                    _mm256_permute2f128_ps::<0x20>(s0, s4),
                    _mm256_permute2f128_ps::<0x20>(s1, s5),
                    _mm256_permute2f128_ps::<0x20>(s2, s6),
                    _mm256_permute2f128_ps::<0x20>(s3, s7),
                    _mm256_permute2f128_ps::<0x31>(s0, s4),
                    _mm256_permute2f128_ps::<0x31>(s1, s5),
                    _mm256_permute2f128_ps::<0x31>(s2, s6),
                    _mm256_permute2f128_ps::<0x31>(s3, s7),
                ];
                for (j, v) in rows.into_iter().enumerate() {
                    _mm256_storeu_ps(out.add((i + j) * n + t), v);
                }
            }
            i += 8;
        }
        for i in i..fl {
            for k in 0..8 {
                // SAFETY: inside `out` (row i < fl, column t + k < n).
                unsafe { *out.add(i * n + t + k) = x[(t + k) * hop + i] };
            }
        }
        t += 8;
    }
    for i in 0..fl {
        for tt in t..n {
            // SAFETY: inside `out` (row i < fl, column tt < n).
            unsafe { *out.add(i * n + tt) = x[tt * hop + i] };
        }
    }
}

/// [`frame_copy_avx_f32`] for `f64`, transposing 4 x 4 blocks.
///
/// # Safety
///
/// As [`frame_copy_avx_f32`].
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
unsafe fn frame_copy_avx_f64(x: &[f64], fl: usize, hop: usize, n: usize, out: *mut f64) {
    use std::arch::x86_64::{
        _mm256_loadu_pd, _mm256_permute2f128_pd, _mm256_storeu_pd, _mm256_unpackhi_pd,
        _mm256_unpacklo_pd,
    };
    debug_assert!(n == 0 || x.len() >= (n - 1) * hop + fl);
    let p = x.as_ptr();
    let mut t = 0;
    while t + 4 <= n {
        let mut i = 0;
        while i + 4 <= fl {
            // SAFETY: frames t..t + 4 cover samples i..i + 4 (i + 4 <= fl),
            // and rows i..i + 4, columns t..t + 4 are inside `out`.
            unsafe {
                let r = |k: usize| _mm256_loadu_pd(p.add((t + k) * hop + i));
                let (r0, r1, r2, r3) = (r(0), r(1), r(2), r(3));
                let (t0, t1) = (_mm256_unpacklo_pd(r0, r1), _mm256_unpackhi_pd(r0, r1));
                let (t2, t3) = (_mm256_unpacklo_pd(r2, r3), _mm256_unpackhi_pd(r2, r3));
                let rows = [
                    _mm256_permute2f128_pd::<0x20>(t0, t2),
                    _mm256_permute2f128_pd::<0x20>(t1, t3),
                    _mm256_permute2f128_pd::<0x31>(t0, t2),
                    _mm256_permute2f128_pd::<0x31>(t1, t3),
                ];
                for (j, v) in rows.into_iter().enumerate() {
                    _mm256_storeu_pd(out.add((i + j) * n + t), v);
                }
            }
            i += 4;
        }
        for i in i..fl {
            for k in 0..4 {
                // SAFETY: inside `out` (row i < fl, column t + k < n).
                unsafe { *out.add(i * n + t + k) = x[(t + k) * hop + i] };
            }
        }
        t += 4;
    }
    for i in 0..fl {
        for tt in t..n {
            // SAFETY: inside `out` (row i < fl, column tt < n).
            unsafe { *out.add(i * n + tt) = x[tt * hop + i] };
        }
    }
}

/// Overlapping frames of a signal as a strided view, without copying.
///
/// The same frames as [`frame`] (a `(frame_length, n_frames)` matrix whose
/// column `t` is `x[t * hop_length .. t * hop_length + frame_length]`), as a
/// read-only view into `x` with strides `(1, hop_length)`. Use [`frame`] for an
/// owned copy.
///
/// # Arguments
///
/// * `x` - The signal.
/// * `frame_length` - Samples per frame.
/// * `hop_length` - Samples between the starts of successive frames.
///
/// # Errors
///
/// [`ParameterError`] if `x` is shorter than `frame_length`.
///
/// # Examples
///
/// ```rust
/// use audio_samples::nzu;
/// use audio_samples::operations::spectrum::{frame, frame_view};
///
/// let x: Vec<f64> = (0..10).map(f64::from).collect();
/// let v = frame_view(&x, nzu!(4), nzu!(3)).unwrap();
/// assert_eq!(v.dim(), (4, 3));
/// assert_eq!(v.column(1).to_vec(), vec![3.0, 4.0, 5.0, 6.0]);
/// assert_eq!(v, frame(&x, nzu!(4), nzu!(3)).unwrap());
/// ```
pub fn frame_view<T>(
    x: &[T],
    frame_length: NonZeroUsize,
    hop_length: NonZeroUsize,
) -> AudioSampleResult<ndarray::ArrayView2<'_, T>> {
    use ndarray::ShapeBuilder;
    let (fl, hop) = (frame_length.get(), hop_length.get());
    if x.len() < fl {
        return Err(param_err(
            "frame_length",
            format!("input is too short (n={}) for frame_length={fl}", x.len()),
        ));
    }
    let n_frames = 1 + (x.len() - fl) / hop;
    // The last element read is x[(n_frames - 1) * hop + fl - 1] < x.len(), and
    // read-only views may overlap.
    Ok(
        ndarray::ArrayView2::from_shape((fl, n_frames).strides((1, hop)), x)
            .expect("the frames lie inside x"),
    )
}

// =============================================================================
// dB scaling
// =============================================================================

/// Reference value that dB conversions are relative to.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DbReference {
    /// A fixed reference; its absolute value is used. The default configurations
    /// use `Value(1.0)`, i.e. absolute dB.
    Value(f64),
    /// The maximum of the input over the reduction axes, so the loudest element
    /// maps to 0 dB.
    Max,
}

/// Which elements share a reference maximum and a `top_db` floor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum DbAxes {
    /// The default: the last two axes for arrays with two or
    /// more dimensions, so each channel of a `(channels, features, frames)` array is
    /// referenced and floored independently; the whole array otherwise.
    #[default]
    Auto,
    /// The whole array.
    All,
}

/// Configuration of a power/amplitude-to-dB conversion ([`power_to_db`],
/// [`amplitude_to_db`]).
///
/// ## Purpose
///
/// Controls the reference level (0 dB), the smallest value considered before
/// taking the logarithm, and the dynamic-range floor.
///
/// ## Intended usage
///
/// [`DbConfig::default`] (= [`DbConfig::power`]) converts power values to dB
/// with no dynamic-range floor; use [`DbConfig::amplitude`] for amplitudes (its
/// `amin` is `1e-5`). Neither clips the output: set `top_db` to request a floor.
///
/// [`DbConfig::librosa_power`] and [`DbConfig::librosa_amplitude`] floor the
/// output at 80 dB below the maximum.
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
    /// The 0 dB reference. Default: `Value(1.0)`.
    pub reference: DbReference,
    /// Values (and the reference) are clamped below at `amin` before the logarithm,
    /// avoiding `log(0)`. In the units of the input: `1e-10` for power
    /// ([`DbConfig::power`]), `1e-5` for amplitude ([`DbConfig::amplitude`]). Must be
    /// positive.
    pub amin: f64,
    /// If `Some(t)`, the output is floored at `max - t` dB, where `max` is taken over
    /// the same axes as the reference ([`DbAxes`]). `None` (the default) applies no
    /// floor, so values below `max - t` are kept.
    pub top_db: Option<f64>,
    /// Reduction axes for [`DbReference::Max`] and `top_db`. Default: `Auto`.
    pub axes: DbAxes,
}

impl Default for DbConfig {
    fn default() -> Self {
        Self::power()
    }
}

impl DbConfig {
    /// Power to dB with no floor: `reference = Value(1.0)`, `amin = 1e-10`,
    /// `top_db = None`, `axes = Auto`.
    #[must_use]
    pub fn power() -> Self {
        Self {
            reference: DbReference::Value(1.0),
            amin: 1e-10,
            top_db: None,
            axes: DbAxes::Auto,
        }
    }

    /// Amplitude to dB with no floor: as [`DbConfig::power`] with `amin = 1e-5`.
    #[must_use]
    pub fn amplitude() -> Self {
        Self {
            amin: 1e-5,
            ..Self::power()
        }
    }

    /// Settings matching librosa 1.0's `power_to_db` defaults: [`DbConfig::power`]
    /// with `top_db = Some(80.0)`.
    #[must_use]
    pub fn librosa_power() -> Self {
        Self {
            top_db: Some(80.0),
            ..Self::power()
        }
    }

    /// Settings matching librosa 1.0's `amplitude_to_db` defaults:
    /// [`DbConfig::amplitude`] with `top_db = Some(80.0)`.
    #[must_use]
    pub fn librosa_amplitude() -> Self {
        Self {
            top_db: Some(80.0),
            ..Self::amplitude()
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

/// Number of leading axes whose indices select independent blocks for `axes`
/// (`Auto` on 2-D or more: one block per index of the leading axes; otherwise
/// the whole array is one block).
fn db_block_axes(ndim: usize, axes: DbAxes) -> usize {
    match axes {
        DbAxes::Auto if ndim >= 2 => ndim - 2,
        _ => 0,
    }
}

/// Calls `f` on every block of `values` (see [`db_block_axes`]): each block as
/// one slice in memory order when it is contiguous (C or Fortran order),
/// otherwise as a copied-out buffer that is written back afterwards.
fn for_each_db_block<F: AudioFloat>(
    mut values: ArrayViewMutD<'_, F>,
    lead: usize,
    f: &mut impl FnMut(&mut [F]),
) {
    if lead > 0 {
        for sub in values.outer_iter_mut() {
            for_each_db_block(sub, lead - 1, f);
        }
        return;
    }
    if let Some(slice) = values.as_slice_memory_order_mut() {
        f(slice);
    } else {
        let mut buf: Vec<F> = values.iter().copied().collect();
        f(&mut buf);
        for (dst, src) in values.iter_mut().zip(buf) {
            *dst = src;
        }
    }
}

/// Shared implementation of `power_to_db` on already-"powered" values.
///
/// `values` holds the power values (in the memory layout of the input, so
/// Fortran-order spectrograms are processed in memory order); `ref_for_block`
/// gives, per block, the reference value (already squared for amplitude
/// input). Each output element depends only on its input element and on its
/// block's maximum, so the processing order does not change the result.
fn power_to_db_core<F: AudioFloat, D: Dimension>(
    mut values: Array<F, D>,
    ref_for_block: impl Fn(&[F]) -> F,
    amin: F,
    top_db: Option<F>,
    axes: DbAxes,
) -> Array<F, D> {
    let ten = F::of(10.0);
    let lead = db_block_axes(values.ndim(), axes);
    for_each_db_block(
        values.view_mut().into_dyn(),
        lead,
        &mut |chunk: &mut [F]| {
            let ref_value = ref_for_block(chunk);
            let ref_db = ten * amin.max(ref_value).log10();
            // l = log10(max(amin, v)) (vectorised for f32, see `utils::vmath`)
            // and its maximum, then ten * l - ref_db and the top_db floor in
            // one pass. `l -> ten * l - ref_db` is non-decreasing in floating
            // point, so the dB maximum is `ten * max(l) - ref_db`.
            let l_max = crate::utils::vmath::log10_max_in_place(chunk, amin);
            match top_db {
                Some(t) => {
                    let floor = (ten * l_max - ref_db) - t;
                    for v in chunk.iter_mut() {
                        *v = (ten * *v - ref_db).max(floor);
                    }
                }
                None => {
                    for v in chunk.iter_mut() {
                        *v = ten * *v - ref_db;
                    }
                }
            }
        },
    );
    values
}

/// Maximum of a block (`NaN`-ignoring, `-inf` when empty). Eight running
/// maxima, so that the loop vectorises; the result is the same value as a
/// sequential fold (only the sign of a zero maximum may differ, which no
/// caller distinguishes).
fn block_max<F: AudioFloat>(chunk: &[F]) -> F {
    let mut acc = [F::neg_infinity(); 8];
    let (blocks, rem) = chunk.as_chunks::<8>();
    for c in blocks {
        for (a, &v) in acc.iter_mut().zip(c) {
            *a = a.max(v);
        }
    }
    let m = rem.iter().copied().fold(F::neg_infinity(), F::max);
    acc.iter().copied().fold(m, F::max)
}

/// Converts a power spectrogram (or any power-like array) to decibels.
///
/// Computes `10 log10(max(amin, S)) - 10 log10(max(amin, |ref|))`, then, if
/// `top_db` is set, floors the result at `max - top_db` where the maximum is taken
/// per block of [`DbAxes`] (per channel for `(channels, features, frames)` input
/// with the default `Auto`).
///
/// # Arguments
///
/// * `s` - Non-negative power values, any dimensionality (negative values are clamped
///   to `amin`).
/// * `config` - Reference, `amin`, `top_db` and axes; [`DbConfig::default`] has no
///   floor.
///
/// # Returns
///
/// An array of the same shape (standard layout) in dB, in the element type `F`
/// of `s` (`f32` or `f64`). The `f64` parameters of `config` are rounded to `F`.
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
/// // The default applies no floor: 0.0 is clamped to amin = 1e-10, i.e. -100 dB.
/// let db = power_to_db(&s, &DbConfig::default()).unwrap();
/// assert_eq!(db, array![[0.0, -10.0], [-20.0, -100.0]]);
///
/// // Floor the output at max - 80 dB.
/// let floored = power_to_db(&s, &DbConfig::librosa_power()).unwrap();
/// assert_eq!(floored, array![[0.0, -10.0], [-20.0, -80.0]]);
///
/// let rel = power_to_db(&s, &DbConfig::default().with_reference(DbReference::Max).with_top_db(None))
///     .unwrap();
/// assert_eq!(rel[[1, 1]], -100.0); // amin = 1e-10
/// ```
pub fn power_to_db<F, S, D>(
    s: &ArrayBase<S, D>,
    config: &DbConfig,
) -> AudioSampleResult<Array<F, D>>
where
    F: AudioFloat,
    S: Data<Elem = F>,
    D: Dimension,
{
    power_to_db_owned(s.to_owned(), config)
}

/// [`power_to_db`] that converts `s` in place.
fn power_to_db_owned<F: AudioFloat, D: Dimension>(
    s: Array<F, D>,
    config: &DbConfig,
) -> AudioSampleResult<Array<F, D>> {
    config.validate()?;
    let reference = config.reference;
    Ok(power_to_db_core(
        s,
        |chunk| match reference {
            DbReference::Value(r) => F::of(r).abs(),
            DbReference::Max => block_max(chunk),
        },
        F::of(config.amin),
        config.top_db.map(F::of),
        config.axes,
    ))
}

/// Converts an amplitude (magnitude) spectrogram to decibels.
///
/// Equal to [`power_to_db`] applied to `|S|^2` with reference `ref^2` and
/// `amin^2`, i.e. `20 log10(max(amin, |S|) / max(amin, |ref|))` with the same
/// `top_db` floor.
///
/// # Arguments
///
/// * `s` - Amplitudes (absolute values are taken), any dimensionality. For complex
///   STFT values pass `stft.mapv(|z| z.norm())`.
/// * `config` - Use [`DbConfig::amplitude`] for the amplitude default `amin = 1e-5`;
///   [`DbConfig::default`] has the *power* default `amin = 1e-10`.
///
/// # Returns
///
/// An array of the same shape in dB, in the element type `F` of `s`.
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
/// let db = amplitude_to_db(&array![1.0_f64, 0.1, -0.01], &DbConfig::amplitude()).unwrap();
/// assert!((db[1] + 20.0).abs() < 1e-12);
/// assert!((db[2] + 40.0).abs() < 1e-12);
/// ```
pub fn amplitude_to_db<F, S, D>(
    s: &ArrayBase<S, D>,
    config: &DbConfig,
) -> AudioSampleResult<Array<F, D>>
where
    F: AudioFloat,
    S: Data<Elem = F>,
    D: Dimension,
{
    config.validate()?;
    let reference = config.reference;
    Ok(power_to_db_core(
        s.map(|x| x.abs() * x.abs()),
        |chunk| match reference {
            DbReference::Value(r) => {
                let r = F::of(r).abs();
                r * r
            }
            // max(|S|)^2 == max(|S|^2): squaring is monotonic and rounds identically.
            DbReference::Max => block_max(chunk),
        },
        F::of(config.amin * config.amin),
        config.top_db.map(F::of),
        config.axes,
    ))
}

/// Converts decibels back to power: `reference * 10^(S_db / 10)`.
///
/// This inverts [`power_to_db`] for values above the `amin` / `top_db` floors.
/// The result has the element type `F` of `s_db`; `reference` is rounded to `F`.
///
/// # Examples
///
/// ```rust
/// use audio_samples::operations::spectrum::db_to_power;
/// use ndarray::array;
///
/// let p = db_to_power(&array![0.0_f64, -10.0, 20.0], 1.0);
/// assert!((p[1] - 0.1).abs() < 1e-15);
/// assert!((p[2] - 100.0).abs() < 1e-12);
///
/// // f32 input gives f32 output.
/// let p32 = db_to_power(&array![-10.0_f32], 1.0);
/// assert!((p32[0] - 0.1_f32).abs() < 1e-7);
/// ```
#[must_use]
pub fn db_to_power<F, S, D>(s_db: &ArrayBase<S, D>, reference: f64) -> Array<F, D>
where
    F: AudioFloat,
    S: Data<Elem = F>,
    D: Dimension,
{
    let (reference, ten, tenth) = (F::of(reference), F::of(10.0), F::of(0.1));
    if std::any::TypeId::of::<F>() != std::any::TypeId::of::<f32>() {
        return s_db.mapv(|x| reference * ten.powf(x * tenth));
    }
    let mut out = s_db.mapv(|x| x * tenth);
    if let Some(v) = out
        .as_slice_memory_order_mut()
        .and_then(crate::utils::vmath::as_f32)
    {
        // f32: the vectorised 10^y (see `utils::vmath`).
        crate::utils::vmath::exp10_f32(v);
        out.mapv_inplace(|p| reference * p);
        return out;
    }
    out.mapv_inplace(|y| reference * ten.powf(y));
    out
}

/// Converts decibels back to amplitude: `sqrt(reference^2 * 10^(S_db / 10))`.
/// The result has the element type `F` of `s_db`; `reference^2` is rounded to `F`.
///
/// # Examples
///
/// ```rust
/// use audio_samples::operations::spectrum::db_to_amplitude;
/// use ndarray::array;
///
/// let a = db_to_amplitude(&array![-20.0_f64, 0.0], 1.0);
/// assert!((a[0] - 0.1).abs() < 1e-15);
/// assert_eq!(a[1], 1.0);
/// ```
#[must_use]
pub fn db_to_amplitude<F, S, D>(s_db: &ArrayBase<S, D>, reference: f64) -> Array<F, D>
where
    F: AudioFloat,
    S: Data<Elem = F>,
    D: Dimension,
{
    let (r2, ten, tenth) = (F::of(reference * reference), F::of(10.0), F::of(0.1));
    if std::any::TypeId::of::<F>() != std::any::TypeId::of::<f32>() {
        return s_db.mapv(|x| (r2 * ten.powf(x * tenth)).sqrt());
    }
    let mut out = s_db.mapv(|x| x * tenth);
    if let Some(v) = out
        .as_slice_memory_order_mut()
        .and_then(crate::utils::vmath::as_f32)
    {
        crate::utils::vmath::exp10_f32(v);
        out.mapv_inplace(|p| (r2 * p).sqrt());
        return out;
    }
    out.mapv_inplace(|y| (r2 * ten.powf(y)).sqrt());
    out
}

// =============================================================================
// Mel spectrogram and MFCC configuration
// =============================================================================

/// Configuration of a mel spectrogram.
///
/// ## Purpose
///
/// Combines the STFT parameters, the mel filterbank and the exponent applied to
/// the STFT magnitude before mel pooling.
///
/// ## Intended usage
///
/// Pass to [`AudioSpectrum::melspectrogram`] or embed in an [`MfccConfig`].
/// `Default` is `n_fft = 2048`, `hop_length = 512` (fixed, rather than
/// `win_length / 4`), periodic
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
    /// STFT parameters. Default: [`StftConfig::default`] with `hop_length = Some(512)`.
    pub stft: StftConfig,
    /// Mel filterbank parameters. Default: [`MelFilterbankConfig::default`].
    pub mel: MelFilterbankConfig,
    /// Exponent applied to the STFT magnitude before the mel projection: `2.0` for
    /// power (the default), `1.0` for magnitude. Must be finite and positive.
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
    /// Returns the default configuration (same as [`Default`]).
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Settings matching librosa 1.0's defaults; currently the same as [`Default`].
    #[must_use]
    pub fn librosa() -> Self {
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

/// The DCT type used to compute cepstral coefficients.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum DctType {
    /// DCT-I (requires at least 2 mel bands).
    One,
    /// DCT-II, the standard MFCC transform (the default).
    #[default]
    Two,
    /// DCT-III (the inverse of an unnormalised DCT-II up to scale).
    Three,
}

/// Normalisation of the DCT.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum DctNorm {
    /// No normalisation. DCT-II coefficients are then
    /// `2 sum_n x[n] cos(pi k (2n + 1) / (2N))`.
    Unnormalized,
    /// Orthonormal DCT (the default).
    #[default]
    Ortho,
}

/// Configuration of MFCC extraction.
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
/// (from a precomputed log-mel spectrogram).
/// `Default` is 20 coefficients from a 128-band Slaney mel power spectrogram
/// with Slaney-normalised filters, [`power_to_db`] with `reference = 1`,
/// `amin = 1e-10` and no floor, orthonormal DCT-II, no liftering.
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
    /// Mel spectrogram parameters. Default: [`MelSpectrogramConfig::default`].
    pub mel_spectrogram: MelSpectrogramConfig,
    /// Conversion of the mel power spectrogram to dB. Default: [`DbConfig::power`]
    /// (no dynamic-range floor). [`DbConfig::librosa_power`], used by
    /// [`MfccConfig::librosa`], floors the mel spectrogram at 80 dB below its
    /// maximum.
    pub db: DbConfig,
    /// Number of cepstral coefficients kept (the first `n_mfcc` DCT outputs).
    /// Default: 20. Must not exceed the number of mel bands.
    pub n_mfcc: NonZeroUsize,
    /// DCT type. Default: [`DctType::Two`].
    pub dct_type: DctType,
    /// DCT normalisation. Default: [`DctNorm::Ortho`].
    pub dct_norm: DctNorm,
    /// Cepstral liftering coefficient `L`: if positive, coefficient `k` (0-based) is
    /// scaled by `1 + (L / 2) sin(pi (k + 1) / L)`, de-emphasising high quefrencies
    /// less than the low ones. `0` disables liftering (the default).
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
    /// Returns the default configuration (same as [`Default`]). It differs from
    /// [`MfccConfig::librosa`] only in `db`, which applies no dynamic-range floor.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Settings matching librosa 1.0's defaults, including the 80 dB floor of the
    /// mel spectrogram (`db = DbConfig::librosa_power()`).
    #[must_use]
    pub fn librosa() -> Self {
        Self {
            mel_spectrogram: MelSpectrogramConfig::librosa(),
            db: DbConfig::librosa_power(),
            ..Self::default()
        }
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

/// The first `n_out` rows of the `n_in`-point DCT matrix, scaled as [`DctNorm`]
/// describes.
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

/// Computes MFCCs from a log-power (dB) mel spectrogram.
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
/// entries, in the element type `F` of `mel_db`. The DCT matrix (with the lifter
/// applied) is computed in `f64` and rounded to `F` once.
///
/// # Errors
///
/// [`ParameterError`] if `mel_db` has fewer than two dimensions, an empty mel axis, or
/// the DCT/lifter parameters are invalid for its number of bands (including `n_mfcc`
/// greater than the number of mel bands).
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
/// 28(4), 1980.
pub fn mfcc_from_mel_db<F, S, D>(
    mel_db: &ArrayBase<S, D>,
    config: &MfccConfig,
) -> AudioSampleResult<Array<F, D>>
where
    F: AudioFloat,
    S: Data<Elem = F>,
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
    let basis = cepstrum_basis(
        n_mfcc,
        n_mels,
        config.dct_type,
        config.dct_norm,
        config.lifter,
    );
    let basis: Vec<F> = basis.iter().map(|&b| F::of(b)).collect();
    let mut shape = mel_db.raw_dim();
    shape[ndim - 2] = n_mfcc;
    let mut out = Array::<F, D>::zeros(shape);
    mfcc_blocks(
        mel_db.view().into_dyn(),
        out.view_mut().into_dyn(),
        &basis,
        n_mels,
    );
    Ok(out)
}

/// The `(n_mfcc, n_mels)` DCT matrix with the lifter applied, row-major, in
/// `f64` (computed once per thread and parameter set).
fn cepstrum_basis(
    n_mfcc: usize,
    n_mels: usize,
    dct_type: DctType,
    norm: DctNorm,
    lifter: f64,
) -> Arc<Vec<f64>> {
    type Key = (usize, usize, DctType, DctNorm, u64);
    thread_local! {
        static CACHE: std::cell::RefCell<Vec<(Key, Arc<Vec<f64>>)>> =
            const { std::cell::RefCell::new(Vec::new()) };
    }
    let key: Key = (n_mfcc, n_mels, dct_type, norm, lifter.to_bits());
    if let Some(b) = CACHE.with(|c| {
        c.borrow()
            .iter()
            .find(|(k, _)| *k == key)
            .map(|(_, b)| b.clone())
    }) {
        return b;
    }
    let mut basis = dct_matrix(n_mfcc, n_mels, dct_type, norm);
    if lifter > 0.0 {
        let l = lifter;
        for (k, mut row) in basis.outer_iter_mut().enumerate() {
            let li = (std::f64::consts::PI * (k + 1) as f64 / l).sin();
            let g = 1.0 + (l / 2.0) * li;
            row.mapv_inplace(|b| b * g);
        }
    }
    let b = Arc::new(basis.iter().copied().collect::<Vec<f64>>());
    CACHE.with(|c| {
        let mut c = c.borrow_mut();
        if c.len() >= 8 {
            c.remove(0);
        }
        c.push((key, b.clone()));
    });
    b
}

/// `out = basis · x` over the last two axes of every block of the leading axes.
fn mfcc_blocks<F: AudioFloat>(
    x: ndarray::ArrayViewD<'_, F>,
    mut out: ArrayViewMutD<'_, F>,
    basis: &[F],
    n_mels: usize,
) {
    if x.ndim() > 2 {
        for (xs, os) in x.outer_iter().zip(out.outer_iter_mut()) {
            mfcc_blocks(xs, os, basis, n_mels);
        }
        return;
    }
    let x = x.into_dimensionality::<ndarray::Ix2>().expect("two axes");
    let mut out = out.into_dimensionality::<ndarray::Ix2>().expect("two axes");
    let w_all = x.ncols();
    let out_slice = out.as_slice_mut().expect("a fresh standard-layout array");
    crate::utils::layout::for_each_row_tile(x, |t0, tile| {
        let w = tile.ncols();
        let rows: Vec<&[F]> = (0..n_mels)
            .map(|m| crate::utils::layout::row(&tile, m))
            .collect();
        crate::utils::vmath::matmul_rows(&mut out_slice[t0..], w_all, w, basis, &rows);
    });
}

/// Projects a (power or magnitude) spectrogram onto mel bands.
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
/// A `(channels, n_mels, n_frames)` mel spectrogram in the element type `F` of
/// `spectrogram`. The filterbank is computed in `f64` and rounded to `F` once.
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
pub fn melspectrogram_from_spectrogram<F: AudioFloat>(
    spectrogram: ArrayView3<'_, F>,
    sample_rate: f64,
    n_fft: NonZeroUsize,
    mel: &MelFilterbankConfig,
) -> AudioSampleResult<Array3<F>> {
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
    let basis = mel_bands_cached::<F>(sample_rate, n_fft, mel)?;
    let mut out = Array3::<F>::zeros((n_channels, basis.len(), n_frames));
    for (c, s) in spectrogram.outer_iter().enumerate() {
        banded_dot_into(&basis, s, out.index_axis_mut(Axis(0), c));
    }
    Ok(out)
}

/// Mel filter bands, `(first column, weights)` per filter.
type MelBands<F> = Vec<(usize, Vec<F>)>;

/// [`mel_filterbank_bands`] for `F`, kept per thread for the last few
/// parameter sets (the bands depend only on the parameters).
fn mel_bands_cached<F: AudioFloat>(
    sample_rate: f64,
    n_fft: NonZeroUsize,
    mel: &MelFilterbankConfig,
) -> AudioSampleResult<Arc<MelBands<F>>> {
    use std::any::{Any, TypeId};
    type Entry = (
        TypeId,
        u64,
        usize,
        MelFilterbankConfig,
        Arc<dyn Any + Send + Sync>,
    );
    thread_local! {
        static CACHE: std::cell::RefCell<Vec<Entry>> = const { std::cell::RefCell::new(Vec::new()) };
    }
    let (tid, sr, n) = (TypeId::of::<F>(), sample_rate.to_bits(), n_fft.get());
    let hit = CACHE.with(|c| {
        c.borrow()
            .iter()
            .find(|e| e.0 == tid && e.1 == sr && e.2 == n && e.3 == *mel)
            .map(|e| e.4.clone())
    });
    if let Some(b) = hit {
        return Ok(b
            .downcast::<MelBands<F>>()
            .expect("entries are keyed by their type"));
    }
    let bands = Arc::new(mel_filterbank_bands::<F>(sample_rate, n_fft, mel)?);
    CACHE.with(|c| {
        let mut c = c.borrow_mut();
        if c.len() >= 8 {
            c.remove(0);
        }
        c.push((
            tid,
            sr,
            n,
            mel.clone(),
            bands.clone() as Arc<dyn Any + Send + Sync>,
        ));
    });
    Ok(bands)
}

/// `out = basis · s` for a filterbank whose rows are each non-zero over one
/// contiguous range of columns (mel and similar triangular filterbanks),
/// given as `(first column, weights)` per row ([`mel_filterbank_bands`]).
///
/// Only the non-zero range of every row is visited, so the cost is
/// proportional to the number of non-zero filter weights rather than to
/// `rows × columns`; each output row is accumulated as a sum of scaled
/// spectrogram rows (contiguous, vectorisable). For finite input the result
/// equals the dense product up to floating-point summation order.
pub(crate) fn banded_dot_into<F: AudioFloat>(
    basis: &[(usize, Vec<F>)],
    s: ArrayView2<'_, F>,
    mut out: ndarray::ArrayViewMut2<'_, F>,
) {
    out.fill(F::zero());
    // Row tiles of `s` (C-order input in place, other layouts copied one
    // tile at a time); each output value is accumulated over its filter's
    // bins in ascending order, as before.
    crate::utils::layout::for_each_row_tile(s, |t0, tile| {
        let w = tile.ncols();
        for ((lo, weights), mut out_row) in basis.iter().zip(out.outer_iter_mut()) {
            let acc = out_row
                .as_slice_mut()
                .expect("rows of a standard-layout array");
            let acc = &mut acc[t0..t0 + w];
            for (k, &wk) in weights.iter().enumerate() {
                for (a, &v) in acc.iter_mut().zip(crate::utils::layout::row(&tile, lo + k)) {
                    *a += wk * v;
                }
            }
        }
    });
}

/// `|z|^power`.
///
/// Computed from `re² + im²` rather than a `hypot`-based `abs(z)` (the libm
/// `hypot` call dominated spectrogram computation); the two agree to
/// within about one ulp. STFT values are far from the range where the squares
/// could overflow or underflow.
fn magnitude_power<F: AudioFloat>(z: Complex<F>, power: f64) -> F {
    let sq = z.norm_sqr();
    if power == 2.0 {
        sq
    } else if power == 1.0 {
        sq.sqrt()
    } else {
        sq.sqrt().powf(F::of(power))
    }
}

// =============================================================================
// delta / stack_memory
// =============================================================================

/// How [`delta`] handles the edges of the data.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum DeltaMode {
    /// Fit a polynomial to the first and last `width` samples and differentiate it
    /// at the edge points (the default). Requires `width <= ` the axis length.
    #[default]
    Interp,
    /// Repeat the edge sample.
    Nearest,
    /// Mirror without repeating the edge sample.
    Mirror,
    /// Pad with zeros.
    Constant,
    /// Wrap around periodically.
    Wrap,
}

/// Configuration of [`delta`].
///
/// ## Purpose
///
/// Controls the Savitzky-Golay derivative estimate: the window width, the
/// derivative order, the axis it runs along and the edge treatment.
///
/// ## Intended usage
///
/// `Default` is `width = 9`, first derivative, along the last
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
    /// Default: 9.
    pub width: usize,
    /// Derivative order (1 = delta, 2 = delta-delta). Also the fitted polynomial
    /// order. Default: 1.
    pub order: usize,
    /// Axis along which to differentiate; negative values count from the end.
    /// Default: -1 (frames).
    pub axis: isize,
    /// Edge handling. Default: [`DeltaMode::Interp`].
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
    /// Returns the default configuration (same as [`Default`]).
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Settings matching librosa 1.0's defaults; currently the same as [`Default`].
    #[must_use]
    pub fn librosa() -> Self {
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
/// samples, in the scaled coordinate `u = (j - h) / h`. The coefficients are
/// computed in `f64` and rounded to `F` once.
struct SavGol<F> {
    width: usize,
    half: usize,
    order: usize,
    /// `(order + 1) x width` pseudo-inverse: polynomial coefficients `a = pinv * x`.
    pinv: Vec<Vec<F>>,
    /// Centre derivative weights, applied as a dot product with the window.
    centre: Vec<F>,
}

impl<F: AudioFloat> SavGol<F> {
    fn new(width: usize, order: usize) -> Self {
        let half = width / 2;
        let h = half as f64;
        let p = order; // polynomial order equals the derivative order
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
        let centre = pinv[order].iter().map(|v| F::of(v * scale)).collect();
        let pinv = pinv
            .iter()
            .map(|row| row.iter().map(|&v| F::of(v)).collect())
            .collect();
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
    fn edge_derivative(&self, window: &[F], pos: usize) -> F {
        let h = self.half as f64;
        let u0 = F::of((pos as f64 - h) / h);
        let d = self.order;
        let mut acc = F::zero();
        for q in d..self.pinv.len() {
            let a_q: F = self.pinv[q].iter().zip(window).map(|(&p, &x)| p * x).sum();
            acc += F::of(falling_factorial(q, d)) * a_q * u0.powi((q - d) as i32);
        }
        acc / F::of(h.powi(d as i32))
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

/// Estimates local derivatives of a feature sequence with a Savitzky-Golay filter
/// of length `width` whose polynomial order and derivative order are both `order`.
///
/// # Arguments
///
/// * `data` - Feature array of any dimensionality, e.g. MFCCs `(channels, n_mfcc, frames)`.
/// * `config` - Window width, order, axis and edge mode.
///
/// # Returns
///
/// An array of the same shape holding the `order`-th derivative estimate along the
/// chosen axis, in the element type `F` of `data`. The filter coefficients are
/// computed in `f64` and rounded to `F` once.
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
/// simplified least squares procedures," *Analytical Chemistry* 36(8), 1964.
pub fn delta<F, S, D>(
    data: &ArrayBase<S, D>,
    config: &DeltaConfig,
) -> AudioSampleResult<Array<F, D>>
where
    F: AudioFloat,
    S: Data<Elem = F>,
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
    let sg = SavGol::<F>::new(config.width, config.order);
    let (w, h) = (sg.width, sg.half);
    let mut out = Array::<F, D>::zeros(data.raw_dim());
    let mut x = vec![F::zero(); len];
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
                        .map(|(&c, &v)| c * v)
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
                            delta_source_index(src, len, config.mode)
                                .map_or(F::zero(), |s| *c * x[s])
                        })
                        .sum()
                };
            }
        });
    Ok(out)
}

/// Stacks time-delayed copies of a feature array, padding with zeros.
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
/// * `n_steps` - Number of copies (commonly 2).
/// * `delay` - Frame delay between copies (commonly 1). Must not be 0.
///
/// # Returns
///
/// An array of shape `(..., features * n_steps, frames)` and element type `F`.
///
/// # Errors
///
/// [`ParameterError`] if `delay` is 0, `data` has fewer than two dimensions or no frames,
/// or the output would exceed 2^31 elements.
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
pub fn stack_memory<F, S, D>(
    data: &ArrayBase<S, D>,
    n_steps: NonZeroUsize,
    delay: isize,
) -> AudioSampleResult<Array<F, D>>
where
    F: AudioFloat,
    S: Data<Elem = F>,
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
    check_derived_len(
        "n_steps",
        "the stacked memory",
        data.len() as f64 * n_steps.get() as f64,
        MAX_SIGNAL_ELEMENTS,
    )?;
    let mut shape = data.raw_dim();
    shape[ndim - 2] = d * n_steps.get();
    let mut out = Array::<F, D>::zeros(shape);
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

/// Spectral analysis on [`AudioSamples`].
///
/// Every method works on all channels: outputs are shaped
/// `(channels, n_features, n_frames)` (mono gives one channel). Outputs have the
/// element type [`Float`](Self::Float) (or `Complex<Self::Float>`): `f32` for
/// `u8`, `i16`, `I24` and `f32` audio, `f64` for `i32` and `f64` audio (see
/// [`AudioSample::Float`](crate::AudioSample::Float)). The computation runs in
/// that type. See the [module documentation](self) for the operations and their
/// free-function counterparts.
///
/// The existing [`AudioTransforms`](crate::AudioTransforms) spectrogram methods are
/// unchanged; they follow the `spectrograms` crate's conventions instead.
pub trait AudioSpectrum {
    /// The float type of the outputs and of the computation: the sample type's
    /// [`AudioSample::Float`](crate::AudioSample::Float).
    type Float: AudioFloat;

    /// The sample type of the audio. [`istft_matrix`](Self::istft_matrix)
    /// returns audio of this type.
    type Sample: StandardSample;

    /// Computes the complex STFT of every channel.
    ///
    /// # Arguments
    ///
    /// * `config` - STFT parameters (see [`StftConfig::default`]).
    ///
    /// # Returns
    ///
    /// A `(channels, n_fft / 2 + 1, n_frames)` complex array; see [`stft`].
    ///
    /// # Errors
    ///
    /// [`ParameterError`] if the audio contains a NaN or infinite sample.
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
    fn stft_matrix(&self, config: &StftConfig) -> AudioSampleResult<Array3<Complex<Self::Float>>>;

    /// [`AudioSpectrum::stft_matrix`] written into a given array, which needs no memory
    /// for the matrix.
    ///
    /// # Arguments
    ///
    /// * `config` - STFT parameters.
    /// * `out` - Of shape `(channels, n_fft / 2 + 1, n_frames)`, in any memory layout, with
    ///   `n_frames` from [`stft_n_frames`].
    ///
    /// # Errors
    ///
    /// As [`AudioSpectrum::stft_matrix`], and [`ParameterError`] if `out` has another shape.
    fn stft_matrix_into(
        &self,
        config: &StftConfig,
        out: ndarray::ArrayViewMut3<'_, Complex<Self::Float>>,
    ) -> AudioSampleResult<()>;

    /// Inverts an STFT matrix to audio.
    ///
    /// The inverse transform runs in [`Float`](Self::Float), and the result is
    /// converted to the sample type [`Sample`](Self::Sample) with the crate's
    /// audio-aware conversion (integer types round and saturate).
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
    /// Mono audio of type [`Sample`](Self::Sample) for a one-channel matrix,
    /// multi-channel audio otherwise.
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
        stft: &Array3<Complex<Self::Float>>,
        config: &StftConfig,
        sample_rate: SampleRate,
        length: Option<NonZeroUsize>,
    ) -> AudioSampleResult<AudioSamples<'static, Self::Sample>>
    where
        Self: Sized;

    /// Computes `|STFT|^power` for every channel.
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
    /// [`ParameterError`] if the audio contains a NaN or infinite sample.
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
    fn spectrogram_matrix(
        &self,
        config: &StftConfig,
        power: f64,
    ) -> AudioSampleResult<Array3<Self::Float>>;

    /// Computes a mel spectrogram.
    ///
    /// # Arguments
    ///
    /// * `config` - STFT, filterbank and exponent ([`MelSpectrogramConfig::default`]:
    ///   power 2, 128 Slaney bands, hop 512).
    ///
    /// # Returns
    ///
    /// A `(channels, n_mels, n_frames)` array.
    ///
    /// # Errors
    ///
    /// [`ParameterError`] if the audio contains a NaN or infinite sample.
    ///
    /// [`ParameterError`] if the configuration is invalid for the signal's sample rate.
    ///
    /// # Precision
    ///
    /// The filterbank is computed in `f64` and rounded to [`Float`](Self::Float).
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
    fn melspectrogram(
        &self,
        config: &MelSpectrogramConfig,
    ) -> AudioSampleResult<Array3<Self::Float>>;

    /// Computes MFCCs.
    ///
    /// The pipeline is [`melspectrogram`](Self::melspectrogram), then [`power_to_db`]
    /// with `config.db` (per channel with the default [`DbAxes::Auto`]), then
    /// [`mfcc_from_mel_db`].
    ///
    /// # Arguments
    ///
    /// * `config` - MFCC parameters (see [`MfccConfig::default`]).
    ///
    /// # Returns
    ///
    /// A `(channels, n_mfcc, n_frames)` array.
    ///
    /// # Errors
    ///
    /// [`ParameterError`] if the audio contains a NaN or infinite sample.
    ///
    /// [`ParameterError`] if the configuration is invalid for the signal's sample rate.
    ///
    /// # Precision
    ///
    /// As for [`melspectrogram`](Self::melspectrogram) (the mel basis). This is a
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
    fn mfcc_frames(&self, config: &MfccConfig) -> AudioSampleResult<Array3<Self::Float>>;
}

/// `|STFT|^power` of `(channels, samples)` rows, computed in `F`.
///
/// # Errors
/// As for [`AudioSpectrum::spectrogram_matrix`] (without the finiteness check).
pub(crate) fn spectrogram_rows<F: AudioFloat>(
    rows: ArrayView2<'_, F>,
    config: &StftConfig,
    power: f64,
) -> AudioSampleResult<Array3<F>> {
    validate_power(power)?;
    if power == 2.0 {
        stft_map(rows, config, |z| z.norm_sqr())
    } else if power == 1.0 {
        stft_map(rows, config, |z| z.norm_sqr().sqrt())
    } else {
        stft_map(rows, config, |z| magnitude_power(z, power))
    }
}

/// The mel spectrogram of `(channels, samples)` rows sampled at
/// `sample_rate` Hz, computed in `F`.
///
/// # Errors
/// As for [`AudioSpectrum::melspectrogram`] (without the finiteness check).
pub(crate) fn melspectrogram_rows<F: AudioFloat>(
    rows: ArrayView2<'_, F>,
    sample_rate: f64,
    config: &MelSpectrogramConfig,
) -> AudioSampleResult<Array3<F>> {
    config.validate(sample_rate)?;
    validate_power(config.power)?;
    let power = config.power;
    if power == 2.0 {
        mel_of_blocks(rows, sample_rate, config, |z| z.norm_sqr())
    } else if power == 1.0 {
        mel_of_blocks(rows, sample_rate, config, |z| z.norm_sqr().sqrt())
    } else {
        mel_of_blocks(rows, sample_rate, config, |z| magnitude_power(z, power))
    }
}

/// The mel spectrogram, projected one block of frames at a time, so that the
/// whole `(bins, frames)` spectrogram is never held in memory: the memory touched
/// beyond the output is a block. Each output value adds its filter's bins in
/// ascending order, as [`banded_dot_into`] does, so the result is the same.
fn mel_of_blocks<F: AudioFloat>(
    rows: ArrayView2<'_, F>,
    sample_rate: f64,
    config: &MelSpectrogramConfig,
    map: impl Fn(Complex<F>) -> F,
) -> AudioSampleResult<Array3<F>> {
    let frames = StftFrames::new(rows, &config.stft)?;
    let (n_bins, n_frames) = (frames.n_bins, frames.n_frames);
    let basis = mel_bands_cached::<F>(sample_rate, config.stft.n_fft, &config.mel)?;
    let mut out = Array3::<F>::zeros((rows.nrows(), basis.len(), n_frames));
    let mut values = vec![F::zero(); STFT_BLOCK * n_bins];
    frames.for_each_block(rows, &config.stft, |c, t0, nb, block| {
        // Bin-major, so that each filter weight scales a contiguous run of frames.
        for (k, bin) in values
            .as_chunks_mut::<STFT_BLOCK>()
            .0
            .iter_mut()
            .enumerate()
        {
            for (j, v) in bin[..nb].iter_mut().enumerate() {
                *v = map(block[j * n_bins + k]);
            }
        }
        let mut out_c = out.index_axis_mut(Axis(0), c);
        for ((lo, weights), mut out_row) in basis.iter().zip(out_c.outer_iter_mut()) {
            let acc = out_row
                .as_slice_mut()
                .expect("rows of a standard-layout array");
            let acc = &mut acc[t0..t0 + nb];
            for (i, &wk) in weights.iter().enumerate() {
                let bin = &values[(lo + i) * STFT_BLOCK..(lo + i) * STFT_BLOCK + nb];
                for (a, &v) in acc.iter_mut().zip(bin) {
                    *a += wk * v;
                }
            }
        }
    })?;
    Ok(out)
}

/// The MFCCs of `(channels, samples)` rows sampled at `sample_rate` Hz,
/// computed in `F`.
///
/// # Errors
/// As for [`AudioSpectrum::mfcc_frames`] (without the finiteness check).
pub(crate) fn mfcc_rows<F: AudioFloat>(
    rows: ArrayView2<'_, F>,
    sample_rate: f64,
    config: &MfccConfig,
) -> AudioSampleResult<Array3<F>> {
    config.validate(sample_rate)?;
    let mel = melspectrogram_rows(rows, sample_rate, &config.mel_spectrogram)?;
    let mel_db = power_to_db_owned(mel, &config.db)?;
    mfcc_from_mel_db(&mel_db, config)
}

impl<T> AudioSpectrum for AudioSamples<'_, T>
where
    T: StandardSample,
{
    type Float = T::Float;
    type Sample = T;

    fn stft_matrix(&self, config: &StftConfig) -> AudioSampleResult<Array3<Complex<T::Float>>> {
        stft(finite_float_rows(self)?.view(), config)
    }

    fn stft_matrix_into(
        &self,
        config: &StftConfig,
        out: ndarray::ArrayViewMut3<'_, Complex<T::Float>>,
    ) -> AudioSampleResult<()> {
        stft_into(finite_float_rows(self)?.view(), config, out)
    }

    fn istft_matrix(
        stft: &Array3<Complex<T::Float>>,
        config: &StftConfig,
        sample_rate: SampleRate,
        length: Option<NonZeroUsize>,
    ) -> AudioSampleResult<AudioSamples<'static, T>> {
        let y = istft(stft.view(), config, length.map(NonZeroUsize::get))?;
        float_rows_to_audio::<T>(y, sample_rate, "istft")
    }

    fn spectrogram_matrix(
        &self,
        config: &StftConfig,
        power: f64,
    ) -> AudioSampleResult<Array3<T::Float>> {
        validate_power(power)?;
        spectrogram_rows(finite_float_rows(self)?.view(), config, power)
    }

    fn melspectrogram(&self, config: &MelSpectrogramConfig) -> AudioSampleResult<Array3<T::Float>> {
        let sr = self.sample_rate_hz();
        config.validate(sr)?;
        melspectrogram_rows(finite_float_rows(self)?.view(), sr, config)
    }

    fn mfcc_frames(&self, config: &MfccConfig) -> AudioSampleResult<Array3<T::Float>> {
        let sr = self.sample_rate_hz();
        config.validate(sr)?;
        mfcc_rows(finite_float_rows(self)?.view(), sr, config)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Projecting block by block gives the same bits as projecting the whole
    /// spectrogram, for every power path, block boundary and channel count.
    #[test]
    fn mel_of_blocks_matches_the_whole_spectrogram() {
        let rows = Array2::from_shape_fn((2, 12_345), |(c, n)| {
            ((n as f32 * (0.01 + 0.03 * c as f32)).sin() + (n as f32 * 0.37).cos()) * 0.4
        });
        for power in [2.0, 1.0, 1.5] {
            let mut config = MelSpectrogramConfig::default();
            config.power = power;
            config.stft = config.stft.with_n_fft(crate::nzu!(512));
            config.mel = config.mel.with_n_mels(crate::nzu!(40));
            let direct = melspectrogram_rows(rows.view(), 16_000.0, &config).unwrap();
            let s = spectrogram_rows(rows.view(), &config.stft, power).unwrap();
            let whole =
                melspectrogram_from_spectrogram(s.view(), 16_000.0, config.stft.n_fft, &config.mel)
                    .unwrap();
            assert_eq!(direct.dim(), whole.dim());
            assert!(
                direct
                    .iter()
                    .zip(whole.iter())
                    .all(|(a, b)| a.to_bits() == b.to_bits()),
                "power {power}"
            );
        }
    }

    #[test]
    fn frame_copy_matches_view() {
        let x: Vec<f64> = (0..1000).map(|i| (i as f64 * 0.37).sin()).collect();
        for (fl, hop) in [
            (1, 1),
            (7, 3),
            (64, 16),
            (100, 1),
            (1000, 5),
            (33, 40),
            (17, 9),
            (8, 8),
            (12, 1),
        ] {
            let (fl, hop) = (
                NonZeroUsize::new(fl).unwrap(),
                NonZeroUsize::new(hop).unwrap(),
            );
            let c = frame(&x, fl, hop).unwrap();
            assert!(c.is_standard_layout());
            assert_eq!(c, frame_view(&x, fl, hop).unwrap());
            let x32: Vec<f32> = x.iter().map(|&v| v as f32).collect();
            let c32 = frame(&x32, fl, hop).unwrap();
            assert_eq!(c32, frame_view(&x32, fl, hop).unwrap());
        }
    }
    use crate::{AudioTypeConversion, sample_rate};
    use ndarray::{Array1, array};

    #[test]
    fn extreme_sizes_are_parameter_errors() {
        let x = Array2::<f64>::zeros((1, 4096));
        let huge = StftConfig::new().with_n_fft(NonZeroUsize::new(1 << 40).unwrap());
        assert!(matches!(
            stft(x.view(), &huge),
            Err(AudioSampleError::Parameter(_))
        ));
        let cfg = StftConfig::new().with_n_fft(crate::nzu!(512));
        let d = stft(x.view(), &cfg).unwrap();
        assert!(matches!(
            istft(d.view(), &cfg, Some(usize::MAX)),
            Err(AudioSampleError::Parameter(_))
        ));
        let wide_hop = cfg.with_hop_length(NonZeroUsize::new(usize::MAX / 2));
        assert!(matches!(
            istft(d.view(), &wide_hop, None),
            Err(AudioSampleError::Parameter(_))
        ));
        // 50_000 x 50_001 frames of a 100_000-sample signal: 2.5e9 elements.
        let long = vec![0.0; 100_000];
        assert!(matches!(
            frame(&long, crate::nzu!(50_000), crate::nzu!(1)),
            Err(AudioSampleError::Parameter(_))
        ));
        assert!(matches!(
            stack_memory(
                &Array2::<f64>::zeros((4, 100)),
                NonZeroUsize::new(1 << 40).unwrap(),
                1
            ),
            Err(AudioSampleError::Parameter(_))
        ));
    }

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
        let x = Array2::from_elem((1, 4096), 1.0_f64);
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
    fn frame_square_sums_match_sequential_sums_of_the_padded_signal() {
        let x: Vec<f64> = noise(1000, 3);
        let modes = [
            StftPadMode::Constant,
            StftPadMode::Edge,
            StftPadMode::Reflect,
            StftPadMode::Symmetric,
        ];
        for mode in modes {
            for (fl, hop, pad) in [
                (64, 16, 32),
                (50, 7, 25),
                (33, 33, 0),
                (10, 40, 5),
                (1, 1, 0),
            ] {
                let xp = pad_signal(&x, pad, mode);
                let n_frames = 1 + (xp.len() - fl) / hop;
                let want: Vec<f64> = (0..n_frames)
                    .map(|t| xp[t * hop..t * hop + fl].iter().map(|&v| v * v).sum())
                    .collect();
                let got = frame_square_sums(&x, fl, hop, pad, mode, n_frames);
                assert_eq!(got, want, "{mode:?} fl={fl} hop={hop} pad={pad}");
            }
        }
    }

    #[test]
    fn stft_without_padding_copy_matches_the_padded_frames() {
        let x = Array2::from_shape_vec((1, 300), noise(300, 9)).unwrap();
        for mode in [
            StftPadMode::Constant,
            StftPadMode::Edge,
            StftPadMode::Reflect,
        ] {
            let cfg = StftConfig::new()
                .with_n_fft(crate::nzu!(128))
                .with_hop_length(Some(crate::nzu!(32)))
                .with_pad_mode(mode);
            let s = stft(x.view(), &cfg).unwrap();
            let xp = pad_signal(x.row(0).as_slice().unwrap(), 64, mode);
            let (w, _) = cfg.analysis_window::<f64>().unwrap();
            let plan = forward_plan::<f64>(128);
            for t in 0..s.dim().2 {
                let mut frame: Vec<f64> = xp[t * 32..t * 32 + 128]
                    .iter()
                    .zip(&w)
                    .map(|(&v, &wv)| wv * v)
                    .collect();
                let mut spec = plan.make_output_vec();
                plan.process(&mut frame, &mut spec).unwrap();
                for (k, z) in spec.iter().enumerate() {
                    assert_eq!(s[[0, k, t]], *z);
                }
            }
        }
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
        // Padding longer than the signal keeps reflecting, without the edge sample.
        assert_eq!(
            pad_signal(&x, 4, StftPadMode::Reflect),
            vec![1.0, 2.0, 3.0, 2.0, 1.0, 2.0, 3.0, 2.0, 1.0, 2.0, 3.0]
        );
        // Padding longer than the signal keeps reflecting, with the edge sample.
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
        let s = array![[[1.0_f64, 1e-12]], [[1e-6, 1e-12]]];
        let db = power_to_db(&s, &DbConfig::librosa_power()).unwrap();
        assert_eq!(db[[0, 0, 1]], -80.0);
        // Channel 1: max is -60 dB, so 1e-12 (clamped to amin = 1e-10, -100 dB) is not floored.
        assert!((db[[1, 0, 1]] + 100.0).abs() < 1e-9);
        let all = power_to_db(&s, &DbConfig::librosa_power().with_axes(DbAxes::All)).unwrap();
        assert_eq!(all[[1, 0, 1]], -80.0);
        assert!(power_to_db(&s, &DbConfig::default().with_amin(0.0)).is_err());
        assert!(power_to_db(&s, &DbConfig::default().with_top_db(Some(-1.0))).is_err());
    }

    #[test]
    fn db_round_trips() {
        let p = array![1e-3_f64, 0.5, 2.0];
        let back = db_to_power(
            &power_to_db(&p, &DbConfig::default().with_top_db(None)).unwrap(),
            1.0,
        );
        for (a, b) in back.iter().zip(&p) {
            assert!((a / b - 1.0).abs() < 1e-12);
        }
        let a = array![1e-3_f64, 0.5, 2.0];
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

    /// A 1 s test tone plus noise at 22.05 kHz, in `f64`.
    fn tone_and_noise() -> AudioSamples<'static, f64> {
        let x: Vec<f64> = noise(22050, 11)
            .iter()
            .enumerate()
            .map(|(i, v)| 0.5 * (0.05 * i as f64).sin() + 0.4 * v)
            .collect();
        AudioSamples::new_mono(Array1::from(x), sample_rate!(22050)).unwrap()
    }

    /// Largest `|a - b|` over the elements, divided by the largest `|b|`.
    fn rel_err_to_peak(a: impl Iterator<Item = f64>, b: impl Iterator<Item = f64> + Clone) -> f64 {
        let peak = b.clone().fold(0.0f64, |m, v| m.max(v.abs()));
        let err = a.zip(b).fold(0.0f64, |m, (p, q)| m.max((p - q).abs()));
        err / peak
    }

    #[test]
    fn output_float_type_follows_sample_type() {
        let audio = tone_and_noise();
        let i16_audio = audio.to_format::<i16>();
        let i32_audio = audio.to_format::<i32>();
        let mel_cfg = MelSpectrogramConfig::default();
        let m16: Array3<f32> = i16_audio.melspectrogram(&mel_cfg).unwrap();
        let m32: Array3<f64> = i32_audio.melspectrogram(&mel_cfg).unwrap();
        assert_eq!(m16.dim(), m32.dim());
        let s16: Array3<Complex<f32>> = i16_audio.stft_matrix(&StftConfig::default()).unwrap();
        let s32: Array3<Complex<f64>> = i32_audio.stft_matrix(&StftConfig::default()).unwrap();
        assert_eq!(s16.dim(), s32.dim());
        let c: Array3<f32> = audio
            .to_format::<f32>()
            .mfcc_frames(&MfccConfig::default())
            .unwrap();
        assert_eq!(c.dim(), (1, 20, 44));
    }

    #[test]
    fn f32_path_agrees_with_f64_path() {
        // Each f32 result is compared with the f64 path on the same input samples
        // (i16 and f32 samples are exact in f64). The measured worst case is
        // 3.2 * f32::EPSILON of the peak (stft ~1.1e-7, mel ~2.4e-7, mfcc
        // ~3.8e-7). The tolerance of 8 * f32::EPSILON (9.5e-7) leaves headroom
        // below the O(log2(n_fft) * eps) = 11 * eps round-off bound of a
        // 2048-point FFT; a computation that lost precision (for example a mel
        // basis or window rounded to f32 twice) would exceed it.
        let tol = 8.0 * f64::from(f32::EPSILON);
        let audio = tone_and_noise();
        let stft_cfg = StftConfig::default();
        let mel_cfg = MelSpectrogramConfig::default();
        let mfcc_cfg = MfccConfig::default();
        let check = |name: &str,
                     s32: Array3<Complex<f32>>,
                     m32: Array3<f32>,
                     c32: Array3<f32>,
                     reference: AudioSamples<'static, f64>| {
            let s64 = reference.stft_matrix(&stft_cfg).unwrap();
            let e = rel_err_to_peak(
                s32.iter()
                    .map(|z| Complex::new(f64::from(z.re), f64::from(z.im)).norm()),
                s64.iter().map(|z| z.norm()),
            );
            // Magnitudes only bound part of the error; also compare the parts.
            let e_parts = rel_err_to_peak(
                s32.iter().flat_map(|z| [f64::from(z.re), f64::from(z.im)]),
                s64.iter().flat_map(|z| [z.re, z.im]),
            );
            assert!(e < tol && e_parts < tol, "{name} stft: {e:e} / {e_parts:e}");
            let m64 = reference.melspectrogram(&mel_cfg).unwrap();
            let e = rel_err_to_peak(m32.iter().map(|&v| f64::from(v)), m64.iter().copied());
            assert!(e < tol, "{name} melspectrogram: {e:e}");
            let c64 = reference.mfcc_frames(&mfcc_cfg).unwrap();
            let e = rel_err_to_peak(c32.iter().map(|&v| f64::from(v)), c64.iter().copied());
            assert!(e < tol, "{name} mfcc: {e:e}");
        };
        let a16 = audio.to_format::<i16>();
        check(
            "i16",
            a16.stft_matrix(&stft_cfg).unwrap(),
            a16.melspectrogram(&mel_cfg).unwrap(),
            a16.mfcc_frames(&mfcc_cfg).unwrap(),
            a16.to_format::<f64>(),
        );
        let a32 = audio.to_format::<f32>();
        check(
            "f32",
            a32.stft_matrix(&stft_cfg).unwrap(),
            a32.melspectrogram(&mel_cfg).unwrap(),
            a32.mfcc_frames(&mfcc_cfg).unwrap(),
            a32.to_format::<f64>(),
        );
    }

    #[test]
    fn istft_matrix_returns_the_receiver_sample_type() {
        let a16 = tone_and_noise().to_format::<i16>();
        let cfg = StftConfig::default();
        let len = Some(nzu!(22050));
        let s = a16.stft_matrix(&cfg).unwrap();
        let y: AudioSamples<'static, i16> =
            AudioSamples::<i16>::istft_matrix(&s, &cfg, sample_rate!(22050), len).unwrap();
        // The f64 path re-quantised to i16 must agree within one LSB.
        let s64 = a16.to_format::<f64>().stft_matrix(&cfg).unwrap();
        let y64 = AudioSamples::<f64>::istft_matrix(&s64, &cfg, sample_rate!(22050), len)
            .unwrap()
            .to_format::<i16>();
        let (y, y64) = (
            y.as_mono().unwrap().as_view(),
            y64.as_mono().unwrap().as_view(),
        );
        assert_eq!(y.len(), 22050);
        for (&p, &q) in y.iter().zip(y64.iter()) {
            assert!((i32::from(p) - i32::from(q)).abs() <= 1, "{p} vs {q}");
        }
    }
}
