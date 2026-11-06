//! Spectral analysis and frequency-domain transformations for [`AudioSamples`].
//!
//! ## What
//!
//! This module implements the [`AudioTransforms`] trait: the DFT, the short-time
//! Fourier transform (STFT) and its inverse, spectrograms on linear, log-Hz, mel,
//! gammatone and constant-Q axes, MFCCs, chroma, a Welch power spectral density
//! estimate, FFT convolution, and magnitude/phase decomposition. The numerics
//! (FFT, windows, filterbanks) come from the [`spectrograms`] crate (2.1.x).
//!
//! ## Why
//!
//! Frequency-domain analysis is central to audio processing: spectrograms,
//! MFCCs, chromagrams and pitch detection all begin with a transform into the
//! spectral domain. The conventions below are stated precisely because they
//! differ in several places from librosa/SciPy defaults, and values computed
//! here are only comparable with other tools when those conventions match.
//!
//! ## How
//!
//! All methods are available on any [`AudioSamples<T>`] (`u8`, `i16`, `I24`,
//! `i32`, `f32`, `f64`). Samples are first converted to `f64` with the
//! audio-aware conversion (integers scaled to [-1, 1], `u8` offset-binary
//! 128 ↦ 0), so the values below are in units of full scale. Most methods
//! require mono input; [`fft`](AudioTransforms::fft) transforms every channel
//! independently.
//!
//! ### Discrete Fourier transform
//!
//! For a frame x\[n\], n = 0 … N − 1 (N = `n_fft`), the forward DFT is
//!
//! ```text
//! X[k] = Σ_{n=0}^{N-1} x[n] · e^{-j 2π k n / N},     k = 0 … ⌊N/2⌋
//! ```
//!
//! with a **negative exponent and no normalisation** (no 1/N or 1/√N; same as
//! `numpy.fft.rfft`). Only the ⌊N/2⌋ + 1 non-negative-frequency bins of the real
//! signal's spectrum are returned. Bin k is at frequency f_k = k·f_s/N Hz
//! (f_s = sample rate), so the bin spacing is f_s/N and the last bin is the
//! Nyquist frequency f_s/2 when N is even. Signals shorter than N are
//! zero-padded; longer signals are rejected. A sinusoid of amplitude A on a bin
//! centre gives |X\[k\]| = A·Σw/2 (A·N/2 with no window, ≈ A·N/4 with Hann).
//!
//! ### Short-time Fourier transform
//!
//! With window w\[n\] of length N and hop H (samples),
//!
//! ```text
//! X[k, m] = Σ_{n=0}^{N-1} x_p[n + mH] · w[n] · e^{-j 2π k n / N}
//! ```
//!
//! - **Window.** `WindowType::Hanning` is the Hann window in its **symmetric**
//!   form, w\[n\] = 0.5 − 0.5·cos(2πn/(N − 1)); Hamming and Blackman are symmetric
//!   too. librosa and `scipy.signal.stft/spectrogram/welch` default to the
//!   **periodic** Hann (denominator N), so STFT values differ from theirs by
//!   about 0.3 % at N = 512 unless the same window is used.
//! - **Centring.** With `centre = true`, x_p is x zero-padded by ⌊N/2⌋ samples
//!   on both sides (librosa `center=True, pad_mode='constant'`, not `'reflect'`),
//!   so frame m is centred on sample mH. With `centre = false`, x_p = x and frame
//!   m starts at sample mH.
//! - **Frames.** The number of frames is 1 + ⌊(L + 2⌊N/2⌋ − N)/H⌋ with centring
//!   and 1 + ⌊(L − N)/H⌋ without (L = signal length). The reported frame time is
//!   t_m = mH/f_s seconds in both cases.
//! - **Inverse.** [`istft`](AudioTransforms::istft) uses windowed overlap-add
//!   normalised by Σ_m w²[n − mH] (the least-squares inverse), which reconstructs
//!   x exactly wherever that sum is non-zero.
//!
//! ### Spectrogram amplitude scales
//!
//! From the STFT, the `Power` scale is P[k, m] = |X[k, m]|² (units of full
//! scale², no division by N), the `Magnitude` scale is √P = |X[k, m]|, and the
//! `Decibels` scale is
//!
//! ```text
//! P_dB = 10 · log10( max(P, a_min) ),    a_min = 10^(floor_db / 10)
//! ```
//!
//! i.e. dB re 1 (an absolute reference, not the peak). `LogParams::new(-80.0)`
//! therefore means a_min = 10⁻⁸ in power, **not** "80 dB below the maximum":
//! there is no `top_db`, and loud bins can exceed 0 dB. librosa's
//! `power_to_db(S, ref=np.max, top_db=80)` instead measures relative to the peak;
//! subtract the maximum yourself to reproduce it.
//!
//! ### Mel spectrogram
//!
//! Mel bands use the **Slaney** mel scale (librosa's default, `htk=False`):
//!
//! ```text
//! m(f) = 3f / 200                              for f < 1000 Hz
//! m(f) = 15 + 27 · ln(f / 1000) / ln(6.4)      for f ≥ 1000 Hz
//! ```
//!
//! `n_mels + 2` points are spaced uniformly in m between `f_min` and `f_max`;
//! band b is a triangle in Hz rising from point b to 1 at point b + 1 and
//! falling to 0 at point b + 2. The filterbank W is applied to the **power**
//! spectrogram, M[b, m] = Σ_k W[b, k]·P[k, m]. `MelParams::new` applies **no**
//! filter normalisation; `MelParams::with_norm(.., MelNorm::Slaney)` gives
//! librosa's default area normalisation. The `Magnitude` mel spectrogram is
//! √M (not Σ_k W[b, k]·|X[k, m]|), and the `Decibels` one is 10·log10(max(M, a_min)).
//! Note: `crate::utils::audio_math::hz_to_mel` and `mel_frequencies` use the
//! HTK formula 2595·log10(1 + f/700), a different mel scale; do not use them to
//! label these bands.
//!
//! ### MFCC
//!
//! [`mfcc`](AudioTransforms::mfcc) computes, per frame:
//!
//! 1. the power mel spectrogram M\[b\] with `n_mels` bands from 0 Hz to f_s/2
//!    (Slaney scale, no filter normalisation; the range is fixed);
//! 2. L\[b\] = 10·log10(max(M\[b\], 10⁻⁸)) (dB, floor −80 dB);
//! 3. an **unnormalised** DCT-II, c\[q\] = Σ_{b=0}^{B-1} L\[b\]·cos(π q (b + ½)/B),
//!    B = `n_mels`, keeping q = 0 … `n_mfcc` − 1;
//! 4. the HTK sinusoidal lifter c\[q\] ← c\[q\]·(1 + (L/2)·sin(π q / L)) with
//!    **L = 22 by default** (`MfccParams::with_lifter(0)` disables it);
//! 5. if `with_c0(false)`, c\[0\] is dropped, leaving `n_mfcc − 1` rows.
//!
//! librosa's `feature.mfcc` uses an orthonormal DCT (`norm='ortho'`), no lifter
//! and `power_to_db(..., top_db=80)`, so its values differ by a factor of
//! √(B/2) for q ≥ 1 (√B for c0) and by the lifter weights.
//!
//! ### Other representations
//!
//! - **Log-Hz:** `n_bins` log-spaced frequencies; each output bin linearly
//!   interpolates the two nearest STFT bins.
//! - **Gammatone:** with the default `ErbSpacing::Linear`, `spectrograms` 2.1.3
//!   spaces the centre frequencies uniformly in ERB **bandwidth**
//!   24.7·(4.37 f/1000 + 1), which is linear in Hz, not on the ERB-number (Cam)
//!   scale (an upstream bug; see [`gammatone_spectrogram`](AudioTransforms::gammatone_spectrogram)).
//! - **CQT:** bin k has centre f_k = f_min·2^{k/b} (b bins per octave) and
//!   constant quality factor Q = f_k/Δf_k = 1/(2^{1/b} − 1) ≈ 16.8 for b = 12.
//! - **PSD:** Welch's method with density scaling, see
//!   [`power_spectral_density`](AudioTransforms::power_spectral_density).
//!
//! ### Known limitations of `spectrograms` 2.1.3
//!
//! - `WindowType::Kaiser { beta }` is wrong for `beta > 3.75` (an error in the
//!   modified Bessel function I₀); do not use it in teaching material.
//! - `MelSpectrogram::frequencies()` assumes the mel range is 0 Hz to Nyquist and
//!   ignores `MelParams::f_min`/`f_max`, so the axis is wrong for any other range
//!   (the crate's plotting code rebuilds the correct axis).
//! - A window of length 1 (`n_fft = 1`) gives NaN for Hann, Hamming and Blackman;
//!   use `n_fft ≥ 2`.
//!
//! ### Available operations
//!
//! | Method | Description | Multi-channel? |
//! |--------|-------------|:--------------:|
//! | [`fft`](AudioTransforms::fft) | Full-signal one-sided DFT | Yes |
//! | [`stft`](AudioTransforms::stft) / [`istft`](AudioTransforms::istft) | Forward / inverse STFT | No |
//! | [`linear_spectrogram`](AudioTransforms::linear_spectrogram) | Linearly-spaced spectrogram | No |
//! | [`log_frequency_spectrogram`](AudioTransforms::log_frequency_spectrogram) | Log-Hz spectrogram | No |
//! | [`mel_spectrogram`](AudioTransforms::mel_spectrogram) | Mel-scaled spectrogram | No |
//! | [`mfcc`](AudioTransforms::mfcc) | Mel-Frequency Cepstral Coefficients | No |
//! | [`chromagram`](AudioTransforms::chromagram) | Chroma (pitch-class) features | No |
//! | [`gammatone_spectrogram`](AudioTransforms::gammatone_spectrogram) | Gammatone (auditory) spectrogram | No |
//! | [`constant_q_transform`](AudioTransforms::constant_q_transform) | Constant-Q Transform | No |
//! | [`cqt_spectrogram`](AudioTransforms::cqt_spectrogram) | CQT spectrogram | No |
//! | [`power_spectral_density`](AudioTransforms::power_spectral_density) | Welch PSD estimate | No |
//! | [`magphase`](AudioTransforms::magphase) | Magnitude / phase decomposition | - |
//!
//! ## Example
//!
//! ```rust
//! use audio_samples::{AudioSamples, AudioTransforms, sample_rate};
//! use ndarray::array;
//! use std::num::NonZeroUsize;
//!
//! // x = [1, 0, -1, 0] is a cosine at f_s/4: all its energy is in bin k = 1.
//! let data  = array![1.0f32, 0.0, -1.0, 0.0];
//! let audio = AudioSamples::new_mono(data, sample_rate!(44100)).unwrap();
//!
//! let spectrum = audio.fft(NonZeroUsize::new(4).unwrap()).unwrap();
//! assert_eq!(spectrum.shape(), &[1, 3]); // one channel, N/2 + 1 bins
//! assert!((spectrum[[0, 1]].re - 2.0).abs() < 1e-12); // unnormalised: X[1] = N/2
//! assert!(spectrum[[0, 0]].norm() < 1e-12 && spectrum[[0, 2]].norm() < 1e-12);
//! ```
//!
//! ## Error Handling
//!
//! All operations return [`crate::AudioSampleResult`]. Mono-only methods
//! return [`crate::AudioSampleError::Layout`]
//! ([`LayoutError::ChannelCountUnsupported`])
//! when given multi-channel input. Failures originating inside [`spectrograms`]
//! are propagated as [`crate::AudioSampleError::Spectrogram`].
//!
//! ## See Also
//!
//! - [`AudioSamples`]: The core audio data type.
//! - [`AudioTransforms`]: The trait defining all spectral operations.
//! - [`crate::utils::audio_math::fft_frequencies`]: the bin frequencies k·f_s/N.
//!
//! [`AudioTransforms`]: crate::operations::traits::AudioTransforms
use std::num::{NonZeroU32, NonZeroUsize};

use crate::operations::traits::AudioTransforms;
use crate::{
    AudioSampleError, AudioSampleResult, AudioSamples, AudioTypeConversion, ChannelRequirement,
    LayoutError, ParameterError, StandardSample,
};

use ndarray::{Array1, Array2, ArrayView2, Axis};
use non_empty_slice::NonEmptySlice;
use num_complex::Complex;
use spectrograms::{
    AmpScaleSpec, ChromaParams, Chromagram, CqtParams, CqtResult, CqtSpectrogram, FftPlanner,
    Gammatone, GammatoneParams, GammatoneSpectrogram, LinearHz, LinearSpectrogram, LogHz,
    LogHzParams, LogHzSpectrogram, LogParams, MelParams, MelSpectrogram, Mfcc, MfccParams,
    Spectrogram, SpectrogramParams, StftParams, StftPlan, StftResult, WindowType, fft_convolve,
    fft_deconvolve,
};

use std::cell::RefCell;

thread_local! {
    /// Thread-local [`FftPlanner`] reused across all FFT-based transforms in
    /// this module. `spectrograms::FftPlanner` wraps a real-FFT planner that
    /// memoizes plans (twiddles, scratch) by size internally, so reconstructing
    /// it per call discarded that cache. Keeping one planner per thread lets
    /// repeated same-size transforms reuse cached plans. The planner is `!Sync`
    /// (it owns a real-FFT planner), so `thread_local` is the correct scope.
    static FFT_PLANNER: RefCell<FftPlanner> = RefCell::new(FftPlanner::new());
}

/// Runs `f` with the thread-local cached [`FftPlanner`] borrowed mutably.
///
/// The borrow is released as soon as `f` returns, so callers must finish all
/// planner use inside the closure.
#[inline]
fn with_fft_planner<R>(f: impl FnOnce(&mut FftPlanner) -> R) -> R {
    FFT_PLANNER.with(|p| f(&mut p.borrow_mut()))
}

/// Power spectral density estimate produced by [`AudioTransforms::power_spectral_density`].
///
/// Pairs a frequency axis with the estimated power-per-Hz at each bin. The two
/// slices are always the same length: `frequencies()[i]` is the centre frequency
/// of bin `i` in Hz, and `density()[i]` is the estimated power spectral density
/// at that frequency.
#[derive(Clone, Debug, PartialEq)]
pub struct Psd {
    frequencies: Vec<f64>,
    density: Vec<f64>,
}

impl Psd {
    /// Constructs a [`Psd`] from a frequency axis and a matching density vector.
    #[inline]
    pub(crate) fn new(frequencies: Vec<f64>, density: Vec<f64>) -> Self {
        Self {
            frequencies,
            density,
        }
    }

    /// The frequency axis, in Hz. Same length as [`density`](Self::density).
    #[inline]
    #[must_use]
    pub fn frequencies(&self) -> &[f64] {
        &self.frequencies
    }

    /// The estimated one-sided power spectral density at each frequency bin, in
    /// (full-scale units)² per Hz. Same length as [`frequencies`](Self::frequencies).
    #[inline]
    #[must_use]
    pub fn density(&self) -> &[f64] {
        &self.density
    }

    /// Consumes the estimate, returning the `(frequencies, density)` vectors.
    #[inline]
    #[must_use]
    pub fn into_parts(self) -> (Vec<f64>, Vec<f64>) {
        (self.frequencies, self.density)
    }
}

impl<T> AudioTransforms for AudioSamples<'_, T>
where
    T: StandardSample,
{
    /// Computes the Fast Fourier Transform of the audio signal.
    ///
    /// Each channel is transformed independently. Each row holds the
    /// non-negative-frequency half of the unnormalised DFT,
    /// X\[k\] = Σ_n x\[n\]·e^{−j2πkn/N} for k = 0 … ⌊N/2⌋ (N = `n_fft`); bin k is at
    /// k·f_s/N Hz (see [`crate::utils::audio_math::fft_frequencies`]). No window
    /// is applied. Like `numpy.fft.rfft`, there is no 1/N scaling.
    ///
    /// # Arguments
    /// - `n_fft` - FFT length in samples. If longer than the signal the input
    ///   is zero-padded internally; a signal longer than `n_fft` is an error.
    ///
    /// # Returns
    /// An `Array2<Complex<f64>>` of shape `(channels, n_fft / 2 + 1)`.
    ///
    /// # Errors
    /// Returns an error if the signal is longer than `n_fft` or the FFT
    /// computation fails.
    ///
    /// # Examples
    /// ```
    /// use audio_samples::{AudioSamples, AudioTransforms, sample_rate};
    /// use ndarray::array;
    /// use std::num::NonZeroUsize;
    ///
    /// let data  = array![1.0f32, 0.0, -1.0, 0.0];
    /// let audio = AudioSamples::new_mono(data, sample_rate!(44100)).unwrap();
    /// let spectrum = audio.fft(NonZeroUsize::new(4).unwrap()).unwrap();
    /// assert_eq!(spectrum.shape()[0], 1); // one row per channel
    /// ```
    #[inline]
    fn fft(&self, n_fft: NonZeroUsize) -> AudioSampleResult<Array2<Complex<f64>>> {
        let working_samples = self.to_format::<f64>();

        let mut channel_ffts: Vec<Array1<Complex<f64>>> =
            Vec::with_capacity(self.num_channels().get() as usize);

        with_fft_planner(|fft_planner| -> AudioSampleResult<()> {
            for ch in working_samples.channels() {
                let ch_slice = ch
                    .as_slice()
                    .expect("Can always get a mono channel as a slice");
                // safety: ch_slice is non-empty since audio has samples
                let working_ch_slice = unsafe { NonEmptySlice::new_unchecked(ch_slice) };
                let channel_fft: Array1<Complex<f64>> = fft_planner.fft(working_ch_slice, n_fft)?;
                channel_ffts.push(channel_fft);
            }
            Ok(())
        })?;

        ndarray::stack(
            Axis(0),
            &channel_ffts.iter().map(|a| a.view()).collect::<Vec<_>>(),
        )
        .map_err(Into::into)
    }

    /// Computes the Short-Time Fourier Transform (STFT) of a mono signal.
    ///
    /// Computes X\[k, m\] = Σ_n x_p\[n + mH\]·w\[n\]·e^{−j2πkn/N} (N = `n_fft`,
    /// H = `hop_size`), an unnormalised one-sided DFT of each windowed frame
    /// (⌊N/2⌋ + 1 bins × frames). `WindowType::Hanning` is the **symmetric** Hann
    /// window 0.5 − 0.5·cos(2πn/(N − 1)), not the periodic Hann used by default in
    /// librosa and SciPy. With `centre = true` the signal is zero-padded by ⌊N/2⌋
    /// samples on both sides, so frame m is centred on sample mH. The result equals
    /// `librosa.stft(x, n_fft=N, hop_length=H, window=scipy.signal.get_window('hann',
    /// N, fftbins=False), pad_mode='constant')`. See the module docs for the frame
    /// count. The returned [`StftResult`] carries both the complex matrix and the
    /// parameters required by [`istft`] for reconstruction.
    ///
    /// # Arguments
    /// - `params` - STFT configuration (FFT size, hop size, window function,
    ///   and centering behaviour). Field-level constraints are documented on
    ///   [`StftParams`].
    ///
    /// # Returns
    /// An [`StftResult`] whose `data` has shape `(n_fft / 2 + 1, n_frames)`.
    ///
    /// # Errors
    /// - [`crate::AudioSampleError::Layout`] if the signal is multi-channel.
    /// - Errors from the underlying [`spectrograms`] STFT computation.
    ///
    /// [`istft`]: Self::istft
    ///
    /// # Examples
    /// ```
    /// use audio_samples::{AudioSamples, AudioTransforms, nzu, sample_rate, sine_wave};
    /// use spectrograms::{StftParams, WindowType};
    /// use std::time::Duration;
    ///
    /// let audio = sine_wave::<f64>(440.0, Duration::from_millis(100), sample_rate!(44100), 0.8);
    /// let params = StftParams::new(nzu!(1024), nzu!(256), WindowType::Hanning, true).unwrap();
    /// let result = audio.stft(&params).unwrap();
    /// assert!(result.data.nrows() > 0); // frequency bins
    /// ```
    #[inline]
    fn stft(&self, params: &StftParams) -> AudioSampleResult<StftResult> {
        if self.is_multi_channel() {
            return Err(AudioSampleError::Layout(
                LayoutError::channel_count_unsupported(
                    "stft",
                    ChannelRequirement::Mono,
                    self.num_channels().get(),
                ),
            ));
        }

        let samples = self.to_format::<f64>();
        let samples_slice = samples
            .as_slice()
            .expect("Safe since we have ensured mono audio samples");
        // safety: samples_slice is non-empty since audio has samples
        let samples_slice = unsafe { NonEmptySlice::new_unchecked(samples_slice) };
        let spect_params = SpectrogramParams::new(params.to_owned(), self.sample_rate_hz())?;
        let mut stft_plan = StftPlan::new(&spect_params)?;

        stft_plan
            .compute(samples_slice, &spect_params)
            .map_err(Into::into)
    }

    /// Reconstructs a time-domain signal from an [`StftResult`].
    ///
    /// Synthesis is windowed overlap-add normalised by Σ_m w²[n − mH] (the
    /// least-squares inverse STFT), using the window and hop stored in the
    /// [`StftResult`]; reconstruction is exact wherever that sum is non-zero.
    /// The output is a mono signal at the sample rate recorded during the
    /// forward transform. Its length is (n_frames − 1)·H + N − 2⌊N/2⌋ with
    /// centring, which can be up to `hop − 1` samples shorter than the original;
    /// trim or pad to the original length if needed.
    ///
    /// # Arguments
    /// - `stft` - the [`StftResult`] produced by a prior call to [`stft`].
    ///
    /// # Returns
    /// A mono [`AudioSamples`] at the original sample rate.
    ///
    /// # Errors
    /// Returns an error if the reconstruction fails (e.g. mismatched
    /// parameters inside the [`StftResult`]).
    ///
    /// [`stft`]: Self::stft
    ///
    /// # Examples
    /// ```
    /// use audio_samples::{AudioSamples, AudioTransforms, nzu, sample_rate, sine_wave};
    /// use spectrograms::{StftParams, WindowType};
    /// use std::time::Duration;
    ///
    /// let audio = sine_wave::<f64>(440.0, Duration::from_millis(100), sample_rate!(44100), 0.8);
    /// let params = StftParams::new(nzu!(1024), nzu!(256), WindowType::Hanning, true).unwrap();
    /// let stft_result = audio.stft(&params).unwrap();
    /// let reconstructed = AudioSamples::<f64>::istft(stft_result).unwrap();
    /// assert!(reconstructed.samples_per_channel().get() > 0);
    /// ```
    #[inline]
    fn istft(stft: StftResult) -> AudioSampleResult<AudioSamples<'static, T>> {
        let sample_rate = stft.sample_rate;
        // safety: sample_rate is non-zero since it was validated during STFT computation on the ``spectrograms`` side
        let sample_rate = unsafe { NonZeroU32::new_unchecked(sample_rate as u32) };
        let data = &stft.data;
        let n_fft = stft.params.n_fft();
        let hop_size = stft.params.hop_size();
        let window = stft.params.window();
        let centre = stft.params.centre();
        let data = spectrograms::istft(data, n_fft, hop_size, window, centre)?;

        let audio = AudioSamples::from_mono_vec::<f64>(data, sample_rate);
        Ok(audio)
    }

    /// Computes a linearly-spaced spectrogram.
    ///
    /// Rows are the STFT bins (bin k at k·f_s/N Hz). `Power` gives |X|², `Magnitude`
    /// gives |X|, and `Decibels` gives 10·log10(max(|X|², 10^(floor_db/10))) re 1
    /// (an absolute floor, not a range below the peak; see the module docs).
    /// Prefer the typed convenience methods -
    /// [`linear_magnitude_spectrogram`](crate::operations::AudioTransforms::linear_magnitude_spectrogram), [`linear_power_spectrogram`](crate::operations::AudioTransforms::linear_power_spectrogram), or
    /// [`linear_db_spectrogram`](crate::operations::AudioTransforms::linear_db_spectrogram) - for the most common amplitude scales.
    ///
    /// # Arguments
    /// - `params` - spectrogram parameters (window, hop, FFT size).
    /// - `db` - required when `AmpScale` is `Decibels`; ignored otherwise.
    ///   Its `floor_db` is an absolute power floor in dB re 1.
    ///
    /// # Returns
    /// A `Spectrogram<LinearHz, AmpScale>`.
    ///
    /// # Errors
    /// - [`crate::AudioSampleError::Layout`] if the signal is multi-channel.
    /// - Errors from the underlying spectrogram computation.
    ///
    /// # Examples
    /// ```
    /// use audio_samples::{AudioSamples, AudioTransforms, nzu, sample_rate, sine_wave};
    /// use spectrograms::{Magnitude, SpectrogramParams, StftParams, WindowType};
    /// use std::time::Duration;
    ///
    /// let audio = sine_wave::<f64>(440.0, Duration::from_millis(100), sample_rate!(44100), 0.8);
    /// let stft = StftParams::new(nzu!(1024), nzu!(256), WindowType::Hanning, true).unwrap();
    /// let params = SpectrogramParams::new(stft, audio.sample_rate_hz()).unwrap();
    /// let spect = audio.linear_spectrogram::<Magnitude>(&params, None).unwrap();
    /// assert!(spect.data().nrows() > 0);
    /// ```
    #[inline]
    fn linear_spectrogram<AmpScale>(
        &self,
        params: &SpectrogramParams,
        db: Option<&LogParams>,
    ) -> AudioSampleResult<Spectrogram<LinearHz, AmpScale>>
    where
        AmpScale: AmpScaleSpec + 'static,
    {
        if self.is_multi_channel() {
            return Err(AudioSampleError::Layout(
                LayoutError::channel_count_unsupported(
                    "linear_frequency_spectrogram",
                    ChannelRequirement::Mono,
                    self.num_channels().get(),
                ),
            ));
        }
        let samples = self.as_f64();
        let samples_slice = samples.as_slice().expect("Safe since we have ensured mono");
        // safety: Guaranteed non-empty since audio is mono and has samples
        let samples_slice = unsafe { NonEmptySlice::new_unchecked(samples_slice) };
        LinearSpectrogram::<AmpScale>::compute(samples_slice, params, db).map_err(Into::into)
    }

    /// Computes a log-frequency-spaced spectrogram.
    ///
    /// Prefer the typed convenience methods -
    /// [`loghz_power_spectrogram`](crate::operations::AudioTransforms::loghz_power_spectrogram), [`loghz_magnitude_spectrogram`](crate::operations::AudioTransforms::loghz_magnitude_spectrogram), or
    /// [`loghz_db_spectrogram`](crate::operations::AudioTransforms::loghz_db_spectrogram) - for the most common amplitude scales.
    ///
    /// # Arguments
    /// - `params` - spectrogram parameters (window, hop, FFT size).
    /// - `loghz` - log-Hz frequency-axis configuration (min/max
    ///   frequency, number of bins). Field-level constraints are
    ///   documented on [`LogHzParams`].
    /// - `db` - required when `AmpScale` is `Decibels`; ignored otherwise.
    ///
    /// # Returns
    /// A `Spectrogram<LogHz, AmpScale>`.
    ///
    /// # Errors
    /// - [`crate::AudioSampleError::Layout`] if the signal is multi-channel.
    /// - Errors from the underlying spectrogram computation.
    ///
    /// # Examples
    /// ```
    /// use audio_samples::{AudioSamples, AudioTransforms, nzu, sample_rate, sine_wave};
    /// use spectrograms::{LogHzParams, Magnitude, SpectrogramParams, StftParams, WindowType};
    /// use std::time::Duration;
    ///
    /// let audio = sine_wave::<f64>(440.0, Duration::from_millis(100), sample_rate!(44100), 0.8);
    /// let stft = StftParams::new(nzu!(1024), nzu!(256), WindowType::Hanning, true).unwrap();
    /// let params = SpectrogramParams::new(stft, audio.sample_rate_hz()).unwrap();
    /// let loghz = LogHzParams::new(nzu!(64), 80.0, 8_000.0).unwrap();
    /// let spect = audio.log_frequency_spectrogram::<Magnitude>(&params, &loghz, None).unwrap();
    /// assert!(spect.data().nrows() > 0);
    /// ```
    #[inline]
    fn log_frequency_spectrogram<AmpScale>(
        &self,
        params: &SpectrogramParams,
        loghz: &LogHzParams,
        db: Option<&LogParams>,
    ) -> AudioSampleResult<Spectrogram<LogHz, AmpScale>>
    where
        AmpScale: AmpScaleSpec,
    {
        if self.is_multi_channel() {
            return Err(AudioSampleError::Layout(
                LayoutError::channel_count_unsupported(
                    "log_frequency_spectrogram",
                    ChannelRequirement::Mono,
                    self.num_channels().get(),
                ),
            ));
        }
        let samples: AudioSamples<'static, f64> = self.as_f64();
        let samples_slice = samples.as_slice().expect("Safe since we have ensured mono");
        // safety: Guaranteed non-empty since audio is mono and has samples
        let samples_slice = unsafe { NonEmptySlice::new_unchecked(samples_slice) };
        LogHzSpectrogram::<AmpScale>::compute(samples_slice, params, loghz, db).map_err(Into::into)
    }

    /// Computes a mel-scaled spectrogram.
    ///
    /// The mel scale approximates human pitch perception by compressing
    /// high frequencies relative to low ones. Bands use the **Slaney** mel scale
    /// (`librosa.hz_to_mel(htk=False)`) with triangular filters in Hz, applied to the
    /// **power** spectrogram |X|². `MelParams::new` applies no filter normalisation
    /// (librosa's default is `norm='slaney'`; use `MelParams::with_norm(..,
    /// MelNorm::Slaney)` to match it). The `Magnitude` output is the square root of
    /// mel power, not a mel-weighted sum of magnitudes; `Decibels` is
    /// 10·log10(max(mel power, 10^(floor_db/10))) re 1. See the module docs for the
    /// formulas.
    ///
    /// Note: in `spectrograms` 2.1.3 the result's `frequencies()` axis is only
    /// correct when `f_min = 0` and `f_max` = Nyquist; for other ranges compute the
    /// band centres yourself (the inner `n_mels` of `n_mels + 2` points equally
    /// spaced on the Slaney mel scale between `f_min` and `f_max`).
    ///
    /// Prefer the typed
    /// convenience methods - [`mel_mag_spectrogram`](crate::operations::AudioTransforms::mel_mag_spectrogram),
    /// [`mel_power_spectrogram`](crate::operations::AudioTransforms::mel_power_spectrogram), or [`mel_db_spectrogram`](crate::operations::AudioTransforms::mel_db_spectrogram) - for the
    /// most common amplitude scales.
    ///
    /// # Arguments
    /// - `params` - spectrogram parameters (window, hop, FFT size).
    /// - `mel` - mel filter-bank configuration (number of bands,
    ///   frequency range). Field-level constraints are documented on
    ///   [`MelParams`].
    /// - `db` - required when `AmpScale` is `Decibels`; ignored otherwise.
    ///
    /// # Returns
    /// A [`MelSpectrogram<AmpScale>`].
    ///
    /// # Errors
    /// - [`crate::AudioSampleError::Layout`] if the signal is multi-channel.
    /// - Errors from the underlying spectrogram computation.
    ///
    /// ## See Also
    /// - [Mel scale - Wikipedia](https://en.wikipedia.org/wiki/Mel_scale)
    ///
    /// # Examples
    /// ```
    /// use audio_samples::{AudioSamples, AudioTransforms, nzu, sample_rate, sine_wave};
    /// use spectrograms::{Magnitude, MelParams, SpectrogramParams, StftParams, WindowType};
    /// use std::time::Duration;
    ///
    /// let audio = sine_wave::<f64>(440.0, Duration::from_millis(100), sample_rate!(44100), 0.8);
    /// let stft = StftParams::new(nzu!(1024), nzu!(256), WindowType::Hanning, true).unwrap();
    /// let params = SpectrogramParams::new(stft, audio.sample_rate_hz()).unwrap();
    /// let mel = MelParams::new(nzu!(40), 0.0, 8_000.0).unwrap();
    /// let spect = audio.mel_spectrogram::<Magnitude>(&params, &mel, None).unwrap();
    /// assert!(spect.data().nrows() > 0);
    /// ```
    #[inline]
    fn mel_spectrogram<AmpScale>(
        &self,
        params: &SpectrogramParams,
        mel: &MelParams,
        db: Option<&LogParams>, // only used when AmpScale = Decibels
    ) -> AudioSampleResult<MelSpectrogram<AmpScale>>
    where
        AmpScale: AmpScaleSpec,
    {
        if self.is_multi_channel() {
            return Err(AudioSampleError::Layout(
                LayoutError::channel_count_unsupported(
                    "mel_spectrogram",
                    ChannelRequirement::Mono,
                    self.num_channels().get(),
                ),
            ));
        }
        let samples = self.as_f64();
        let samples_slice = samples.as_slice().expect("Safe since we have ensured mono");
        // safety: Guaranteed non-empty since audio is mono and has samples
        let samples_slice = unsafe { NonEmptySlice::new_unchecked(samples_slice) };
        MelSpectrogram::<AmpScale>::compute(samples_slice, params, mel, db).map_err(Into::into)
    }

    /// Computes Mel-Frequency Cepstral Coefficients (MFCCs).
    ///
    /// MFCCs are a compact spectral representation widely used in speech
    /// recognition and audio classification. The pipeline is: power mel
    /// spectrogram (Slaney scale, `n_mels` bands from 0 Hz to f_s/2, no filter
    /// normalisation) → 10·log10(max(·, 10⁻⁸)) → DCT-II **without** orthonormal
    /// scaling, c\[q\] = Σ_b L\[b\]·cos(πq(b + ½)/`n_mels`), keeping the first
    /// `n_mfcc` coefficients → HTK sinusoidal lifter 1 + (L/2)·sin(πq/L) with
    /// **L = 22 by default** (`MfccParams::with_lifter(0)` disables it).
    /// `with_c0(false)` drops c0 and returns `n_mfcc − 1` coefficients.
    ///
    /// Note: values therefore differ from `librosa.feature.mfcc` (orthonormal DCT,
    /// no lifter, `top_db = 80`) by a factor √(`n_mels`/2) for q ≥ 1 (√`n_mels` for
    /// c0) and by the lifter weights.
    ///
    /// # Arguments
    /// - `stft_params` - STFT configuration used for the underlying
    ///   spectrogram.
    /// - `n_mels` - number of mel filter-bank bands.
    /// - `mfcc_params` - MFCC-specific configuration (number of
    ///   coefficients, etc.). Field-level constraints are documented on
    ///   [`MfccParams`].
    ///
    /// # Returns
    /// An [`Mfcc`] whose `data` has shape `(n_mfcc, n_frames)` (or
    /// `(n_mfcc − 1, n_frames)` without c0).
    ///
    /// # Errors
    /// - [`crate::AudioSampleError::Layout`] if the signal is multi-channel.
    /// - Errors from the underlying computation, including `n_mfcc > n_mels`.
    ///
    /// ## See Also
    /// - [MFCC - Wikipedia](https://en.wikipedia.org/wiki/Mel-frequency_cepstral_coefficients)
    ///
    /// # Examples
    /// ```
    /// use audio_samples::{AudioSamples, AudioTransforms, nzu, sample_rate, sine_wave};
    /// use spectrograms::{MfccParams, StftParams, WindowType};
    /// use std::time::Duration;
    ///
    /// let audio = sine_wave::<f64>(440.0, Duration::from_millis(200), sample_rate!(44100), 0.8);
    /// let stft = StftParams::new(nzu!(1024), nzu!(256), WindowType::Hanning, true).unwrap();
    /// let mfcc_params = MfccParams::speech_standard();
    /// let result = audio.mfcc(&stft, nzu!(40), &mfcc_params).unwrap();
    /// assert!(result.data.nrows() > 0); // MFCC coefficients
    /// ```
    #[inline]
    fn mfcc(
        &self,
        stft_params: &StftParams,
        n_mels: NonZeroUsize,
        mfcc_params: &MfccParams,
    ) -> AudioSampleResult<Mfcc> {
        if self.is_multi_channel() {
            return Err(AudioSampleError::Layout(
                LayoutError::channel_count_unsupported(
                    "mfcc",
                    ChannelRequirement::Mono,
                    self.num_channels().get(),
                ),
            ));
        }

        let samples = self.as_f64();
        let samples_slice = samples.as_slice().expect("Safe since we have ensured mono");
        // safety: Guaranteed non-empty since audio is mono and has samples
        let samples_slice = unsafe { NonEmptySlice::new_unchecked(samples_slice) };
        let sample_rate_f = self.sample_rate_hz();
        spectrograms::mfcc(
            samples_slice,
            stft_params,
            sample_rate_f,
            n_mels,
            mfcc_params,
        )
        .map_err(Into::into)
    }

    /// Computes chromagram (pitch-class energy) features.
    ///
    /// A chromagram projects the spectrum onto the twelve pitch classes
    /// (C, C♯, D, … , B), collapsing octave differences.  The result
    /// is useful for harmonic and key detection.
    ///
    /// # Arguments
    /// - `stft_params` - STFT configuration used for the underlying
    ///   spectrogram.
    /// - `cfg` - chromagram configuration (tuning, normalization,
    ///   etc.). Field-level constraints are documented on
    ///   [`ChromaParams`].
    ///
    /// # Returns
    /// A [`Chromagram`].
    ///
    /// # Errors
    /// - [`crate::AudioSampleError::Layout`] if the signal is multi-channel.
    /// - Errors from the underlying computation.
    ///
    /// # Examples
    /// ```
    /// use audio_samples::{AudioSamples, AudioTransforms, nzu, sample_rate, sine_wave};
    /// use spectrograms::{ChromaParams, StftParams, WindowType};
    /// use std::time::Duration;
    ///
    /// let audio = sine_wave::<f64>(440.0, Duration::from_millis(200), sample_rate!(44100), 0.8);
    /// let stft = StftParams::new(nzu!(1024), nzu!(256), WindowType::Hanning, true).unwrap();
    /// let chroma_params = ChromaParams::music_standard();
    /// let result = audio.chromagram(&stft, &chroma_params).unwrap();
    /// assert_eq!(result.data.nrows(), 12); // twelve pitch classes (C through B)
    /// ```
    #[inline]
    fn chromagram(
        &self,
        stft_params: &StftParams,
        cfg: &ChromaParams,
    ) -> AudioSampleResult<Chromagram> {
        if self.is_multi_channel() {
            return Err(AudioSampleError::Layout(
                LayoutError::channel_count_unsupported(
                    "chromagram",
                    ChannelRequirement::Mono,
                    self.num_channels().get(),
                ),
            ));
        }

        let samples = self.as_f64();
        let samples_slice = samples.as_slice().expect("Safe since we have ensured mono");
        // safety: Guaranteed non-empty since audio is mono and has samples
        let samples_slice = unsafe { NonEmptySlice::new_unchecked(samples_slice) };
        let sample_rate_f = self.sample_rate_hz();
        spectrograms::chromagram(samples_slice, stft_params, sample_rate_f, cfg).map_err(Into::into)
    }

    /// Computes a gammatone-filtered spectrogram.
    ///
    /// Gammatone filters model the bandpass response of the human
    /// cochlea. Note: with the default `ErbSpacing::Linear`, `spectrograms` 2.1.3
    /// spaces the centre frequencies uniformly in ERB **bandwidth**
    /// 24.7·(4.37 f/1000 + 1), which is linear in Hz, **not** on the ERB-number
    /// (Cam) scale 21.4·log10(1 + 0.00437 f) (an upstream bug). For auditory
    /// (ERB-rate) spacing use
    /// `GammatoneParams::new(..)?.with_spacing(ErbSpacing::AppleTr35)` (Slaney /
    /// Patterson spacing, uniform in ln(f + 228.7), equivalent to the
    /// Glasberg & Moore ERB-number scale). Prefer the
    /// typed convenience methods -
    /// [`gammatone_magnitude_spectrogram`](crate::operations::AudioTransforms::gammatone_magnitude_spectrogram),
    /// [`gammatone_power_spectrogram`](crate::operations::AudioTransforms::gammatone_power_spectrogram), or
    /// [`gammatone_db_spectrogram`](crate::operations::AudioTransforms::gammatone_db_spectrogram) - for the most common amplitude
    /// scales.
    ///
    /// # Arguments
    /// - `params` - spectrogram parameters (window, hop, FFT size).
    /// - `gammatone_params` - gammatone filter-bank configuration
    ///   (number of bands, frequency range). Field-level constraints are
    ///   documented on [`GammatoneParams`].
    /// - `db` - required when `AmpScale` is `Decibels`; ignored otherwise.
    ///
    /// # Returns
    /// A `Spectrogram<Gammatone, AmpScale>`.
    ///
    /// # Errors
    /// - [`crate::AudioSampleError::Layout`] if the signal is multi-channel.
    /// - Errors from the underlying computation.
    ///
    /// ## See Also
    /// - [Gammatone filter - Wikipedia](https://en.wikipedia.org/wiki/Gammatone)
    ///
    /// # Examples
    /// ```
    /// use audio_samples::{AudioSamples, AudioTransforms, nzu, sample_rate, sine_wave};
    /// use spectrograms::{GammatoneParams, Magnitude, SpectrogramParams, StftParams, WindowType};
    /// use std::time::Duration;
    ///
    /// let audio = sine_wave::<f64>(440.0, Duration::from_millis(100), sample_rate!(44100), 0.8);
    /// let stft = StftParams::new(nzu!(1024), nzu!(256), WindowType::Hanning, true).unwrap();
    /// let params = SpectrogramParams::new(stft, audio.sample_rate_hz()).unwrap();
    /// let gammatone = GammatoneParams::new(nzu!(32), 80.0, 8_000.0).unwrap();
    /// let spect = audio.gammatone_spectrogram::<Magnitude>(&params, &gammatone, None).unwrap();
    /// assert!(spect.data().nrows() > 0);
    /// ```
    #[inline]
    fn gammatone_spectrogram<AmpScale>(
        &self,
        params: &SpectrogramParams,
        gammatone_params: &GammatoneParams,
        db: Option<&LogParams>,
    ) -> AudioSampleResult<Spectrogram<Gammatone, AmpScale>>
    where
        AmpScale: AmpScaleSpec,
    {
        if self.is_multi_channel() {
            return Err(AudioSampleError::Layout(
                LayoutError::channel_count_unsupported(
                    "gammatone_spectrogram",
                    ChannelRequirement::Mono,
                    self.num_channels().get(),
                ),
            ));
        }

        let samples = self.as_f64();
        let samples_slice = samples.as_slice().expect("Safe since we have ensured mono");
        // safety: Guaranteed non-empty since audio is mono and has samples
        let samples_slice = unsafe { NonEmptySlice::new_unchecked(samples_slice) };
        GammatoneSpectrogram::<AmpScale>::compute(samples_slice, params, gammatone_params, db)
            .map_err(Into::into)
    }

    /// Computes the Constant-Q Transform (CQT) of a mono signal.
    ///
    /// Bin k has centre frequency f_k = f_min·2^{k/b} (b = bins per octave;
    /// geometric spacing with ratio 2^{1/b}) and a constant quality factor
    /// Q = f_k/Δf_k = 1/(2^{1/b} − 1) (≈ 16.8 for b = 12), so its frequency
    /// resolution follows the musical scale, which suits pitch and harmonic
    /// analysis. The kernel for bin k is about Q·f_s/f_k samples long, so low
    /// bins have long kernels (good frequency, poor time resolution).
    ///
    /// Note (`spectrograms` 2.1.3): frames are `min(signal_len, 16384)` samples
    /// long and are not centred, so a signal shorter than 16384 samples yields a
    /// single frame whatever `hop_size` is; kernels longer than the frame are
    /// truncated, which lowers the effective Q at low frequencies.
    ///
    /// # Arguments
    /// - `params` - CQT parameters (frequency range, bins per octave,
    ///   etc.). Field-level constraints are documented on [`CqtParams`].
    /// - `hop_size` - hop length in samples between successive frames (only
    ///   relevant for signals longer than the 16384-sample frame).
    ///
    /// # Returns
    /// A [`CqtResult`] containing the CQT matrix.
    ///
    /// # Errors
    /// - [`crate::AudioSampleError::Layout`] if the signal is multi-channel.
    /// - Errors from the underlying CQT computation.
    ///
    /// ## See Also
    /// - [Constant-Q transform - Wikipedia](https://en.wikipedia.org/wiki/Constant-Q_transform)
    ///
    /// # Examples
    /// ```
    /// use audio_samples::{AudioSamples, AudioTransforms, nzu, sample_rate, sine_wave};
    /// use spectrograms::CqtParams;
    /// use std::time::Duration;
    ///
    /// let audio = sine_wave::<f64>(440.0, Duration::from_millis(200), sample_rate!(44100), 0.8);
    /// // 12 bins per octave, 7 octaves, starting at 32.7 Hz (C1)
    /// let cqt_params = CqtParams::new(nzu!(12), nzu!(7), 32.7).unwrap();
    /// let result = audio.constant_q_transform(&cqt_params, nzu!(256)).unwrap();
    /// assert_eq!(result.data.nrows(), 84); // 12 bins x 7 octaves
    /// // A 200 ms signal is shorter than the 16384-sample frame: one frame.
    /// assert_eq!(result.data.ncols(), 1);
    /// ```
    #[inline]
    fn constant_q_transform(
        &self,
        params: &CqtParams,
        hop_size: NonZeroUsize,
    ) -> AudioSampleResult<CqtResult> {
        if self.is_multi_channel() {
            return Err(AudioSampleError::Layout(
                LayoutError::channel_count_unsupported(
                    "constant_q_transform",
                    ChannelRequirement::Mono,
                    self.num_channels().get(),
                ),
            ));
        }

        let samples = self.as_f64();
        let samples_slice = samples.as_slice().expect("Safe since we have ensured mono");
        // safety: Guaranteed non-empty since audio is mono and has samples
        let samples_slice = unsafe { NonEmptySlice::new_unchecked(samples_slice) };
        spectrograms::cqt(
            samples_slice,
            f64::from(self.sample_rate().get()),
            params,
            hop_size,
        )
        .map_err(Into::into)
    }

    /// Computes a CQT-based spectrogram.
    ///
    /// Applies the CQT to the signal and returns the result as a typed
    /// spectrogram.  Prefer the typed convenience methods -
    /// [`cqt_magnitude_spectrogram`](crate::operations::AudioTransforms::cqt_magnitude_spectrogram), [`cqt_power_spectrogram`](crate::operations::AudioTransforms::cqt_power_spectrogram), or
    /// [`cqt_db_spectrogram`](crate::operations::AudioTransforms::cqt_db_spectrogram) - for the most common amplitude scales.
    ///
    /// # Arguments
    /// - `params` - spectrogram parameters (window, hop, FFT size).
    /// - `cqt` - CQT parameters (frequency range, bins per octave,
    ///   etc.). Field-level constraints are documented on [`CqtParams`].
    /// - `db` - required when `AmpScale` is `Decibels`; ignored otherwise.
    ///
    /// # Returns
    /// A [`CqtSpectrogram<AmpScale>`].
    ///
    /// # Errors
    /// - [`crate::AudioSampleError::Layout`] if the signal is multi-channel.
    /// - Errors from the underlying computation.
    ///
    /// # Examples
    /// ```
    /// use audio_samples::{AudioSamples, AudioTransforms, nzu, sample_rate, sine_wave};
    /// use spectrograms::{CqtParams, Magnitude, SpectrogramParams, StftParams, WindowType};
    /// use std::time::Duration;
    ///
    /// let audio = sine_wave::<f64>(440.0, Duration::from_millis(200), sample_rate!(44100), 0.8);
    /// let stft = StftParams::new(nzu!(1024), nzu!(256), WindowType::Hanning, true).unwrap();
    /// let params = SpectrogramParams::new(stft, audio.sample_rate_hz()).unwrap();
    /// let cqt = CqtParams::new(nzu!(12), nzu!(7), 32.7).unwrap();
    /// let spect = audio.cqt_spectrogram::<Magnitude>(&params, &cqt, None).unwrap();
    /// assert!(spect.data().nrows() > 0);
    /// ```
    #[inline]
    fn cqt_spectrogram<AmpScale>(
        &self,
        params: &SpectrogramParams,
        cqt: &CqtParams,
        db: Option<&LogParams>,
    ) -> AudioSampleResult<CqtSpectrogram<AmpScale>>
    where
        AmpScale: AmpScaleSpec,
    {
        if self.is_multi_channel() {
            return Err(AudioSampleError::Layout(
                LayoutError::channel_count_unsupported(
                    "cqt",
                    ChannelRequirement::Mono,
                    self.num_channels().get(),
                ),
            ));
        }

        let working_samples = self.as_f64();
        let working_samples_slice = working_samples
            .as_slice()
            .expect("Safe since we have ensured mono");

        // safety: working_samples_slice is non-empty since audio is mono and has samples
        let working_samples_slice = unsafe { NonEmptySlice::new_unchecked(working_samples_slice) };

        CqtSpectrogram::<AmpScale>::compute(working_samples_slice, params, cqt, db)
            .map_err(Into::into)
    }

    /// Estimates the power spectral density using Welch's method.
    ///
    /// The signal is split into segments of N = `window_size` samples with hop
    /// H = max(1, ⌊(1 − `overlap`)·N⌋); incomplete trailing segments are dropped.
    /// Each segment is multiplied by the **symmetric** Hann window w\[n\]
    /// (`WindowType::Hanning`), with no detrending, and transformed with an
    /// N-point DFT X_s\[k\]. The one-sided density estimate is
    ///
    /// ```text
    /// P[k] = c_k · mean_s |X_s[k]|² / (f_s · Σ_n w[n]²),   c_k = 1 at DC and Nyquist, 2 otherwise
    /// ```
    ///
    /// in units² per Hz (units of full scale), at f_k = k·f_s/N. This matches
    /// `scipy.signal.welch(x, fs, window=scipy.signal.windows.hann(N, sym=True),
    /// nperseg=N, noverlap=N − H, detrend=False, scaling='density')`, and
    /// Σ_k P\[k\]·f_s/N ≈ the mean square of the signal (Parseval). Note: SciPy's
    /// defaults differ (periodic Hann, `detrend='constant'`).
    ///
    /// # Arguments
    /// - `window_size` - segment length N in samples. Must be at least 2 and must
    ///   not exceed the signal length.
    /// - `overlap` - fractional overlap between adjacent segments, in
    ///   the range `[0, 1)`.
    ///
    /// # Returns
    /// A [`Psd`] whose [`frequencies`](Psd::frequencies) and
    /// [`density`](Psd::density) slices both have length ⌊N/2⌋ + 1.
    /// `frequencies[k]` = k·f_s/N in Hz; `density[k]` is P\[k\] in units²/Hz.
    ///
    /// # Errors
    /// - [`crate::AudioSampleError::Layout`] if the signal is multi-channel.
    /// - [`crate::AudioSampleError::Parameter`] if `overlap` is outside `[0, 1)`,
    ///   or if `window_size` is 1 or exceeds the signal length.
    ///
    /// ## See Also
    /// - [Welch's method - Wikipedia](https://en.wikipedia.org/wiki/Welch%27s_method)
    ///
    /// # Examples
    /// ```
    /// use audio_samples::{AudioSamples, AudioTransforms, nzu, sample_rate, sine_wave};
    /// use std::time::Duration;
    ///
    /// let audio = sine_wave::<f64>(440.0, Duration::from_millis(200), sample_rate!(44100), 0.8);
    /// let psd = audio.power_spectral_density(nzu!(1024), 0.5).unwrap();
    /// assert_eq!(psd.frequencies().len(), psd.density().len());
    /// assert!(!psd.frequencies().is_empty());
    /// ```
    #[inline]
    fn power_spectral_density(
        &self,
        window_size: NonZeroUsize,
        overlap: f64,
    ) -> AudioSampleResult<Psd> {
        if self.is_multi_channel() {
            return Err(AudioSampleError::Layout(
                LayoutError::channel_count_unsupported(
                    "power_spectral_density",
                    ChannelRequirement::Mono,
                    self.num_channels().get(),
                ),
            ));
        }
        if !(0.0..1.0).contains(&overlap) {
            return Err(AudioSampleError::Parameter(ParameterError::out_of_range(
                "overlap",
                overlap.to_string(),
                "0.0",
                "1.0",
                "overlap must be in [0, 1)",
            )));
        }
        if self.samples_per_channel() < window_size {
            return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
                "window_size",
                "window_size must not exceed the signal length",
            )));
        }
        if window_size.get() < 2 {
            // A 1-point symmetric Hann window is 0/0 = NaN.
            return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
                "window_size",
                "window_size must be at least 2",
            )));
        }

        let samples = self.to_format::<f64>();
        let signal = samples
            .as_slice()
            .expect("Safe since we have ensured mono audio samples");

        let win = window_size.get();
        let hop = ((1.0 - overlap) * win as f64).floor().max(1.0) as usize;
        let sample_rate = self.sample_rate_hz();

        let mut sum: Vec<f64> = Vec::new();
        let mut segment_count = 0u64;

        with_fft_planner(|planner| -> AudioSampleResult<()> {
            let mut start = 0;
            while start + win <= signal.len() {
                let segment = &signal[start..start + win];
                // safety: segment length == win == window_size >= 1
                let segment_slice = unsafe { NonEmptySlice::new_unchecked(segment) };
                let power = planner.power_spectrum(
                    segment_slice,
                    window_size,
                    Some(WindowType::Hanning),
                )?;

                if sum.is_empty() {
                    sum = power.to_vec();
                } else {
                    for (acc, &val) in sum.iter_mut().zip(power.iter()) {
                        *acc += val;
                    }
                }
                segment_count += 1;
                start += hop;
            }
            Ok(())
        })?;

        // Average the periodograms, then apply density scaling (SciPy
        // `welch(scaling='density')`): divide by fs·Σw², and double every bin
        // except DC and (for even `win`) Nyquist to fold in negative frequencies.
        let count = segment_count as f64;
        let window = spectrograms::make_window::<f64>(WindowType::Hanning, window_size);
        let window_power: f64 = window.iter().map(|w| w * w).sum();
        let scale = 1.0 / (count * sample_rate * window_power);
        let n_bins = sum.len();
        for (k, val) in sum.iter_mut().enumerate() {
            let is_nyquist = win.is_multiple_of(2) && k == n_bins - 1;
            let one_sided = if k == 0 || is_nyquist { 1.0 } else { 2.0 };
            *val *= one_sided * scale;
        }

        // Build frequency axis
        let frequencies: Vec<f64> = (0..sum.len())
            .map(|i| i as f64 * sample_rate / win as f64)
            .collect();

        Ok(Psd::new(frequencies, sum))
    }

    /// Shorthand for [`linear_spectrogram`](crate::operations::AudioTransforms::linear_spectrogram) with `Magnitude` amplitude scale.
    ///
    /// # Examples
    /// ```
    /// use audio_samples::{AudioSamples, AudioTransforms, nzu, sample_rate, sine_wave};
    /// use spectrograms::{SpectrogramParams, StftParams, WindowType};
    /// use std::time::Duration;
    ///
    /// let audio = sine_wave::<f64>(440.0, Duration::from_millis(100), sample_rate!(44100), 0.8);
    /// let stft = StftParams::new(nzu!(1024), nzu!(256), WindowType::Hanning, true).unwrap();
    /// let params = SpectrogramParams::new(stft, audio.sample_rate_hz()).unwrap();
    /// let spect = audio.linear_magnitude_spectrogram(&params).unwrap();
    /// assert!(spect.data().nrows() > 0);
    /// ```
    #[inline]
    fn linear_magnitude_spectrogram(
        &self,
        params: &SpectrogramParams,
    ) -> AudioSampleResult<spectrograms::LinearMagnitudeSpectrogram> {
        self.linear_spectrogram::<spectrograms::Magnitude>(params, None)
    }

    /// Shorthand for [`linear_spectrogram`](crate::operations::AudioTransforms::linear_spectrogram) with `Power` amplitude scale.
    ///
    /// # Examples
    /// ```
    /// use audio_samples::{AudioSamples, AudioTransforms, nzu, sample_rate, sine_wave};
    /// use spectrograms::{SpectrogramParams, StftParams, WindowType};
    /// use std::time::Duration;
    ///
    /// let audio = sine_wave::<f64>(440.0, Duration::from_millis(100), sample_rate!(44100), 0.8);
    /// let stft = StftParams::new(nzu!(1024), nzu!(256), WindowType::Hanning, true).unwrap();
    /// let params = SpectrogramParams::new(stft, audio.sample_rate_hz()).unwrap();
    /// let spect = audio.linear_power_spectrogram(&params).unwrap();
    /// assert!(spect.data().nrows() > 0);
    /// ```
    #[inline]
    fn linear_power_spectrogram(
        &self,
        params: &SpectrogramParams,
    ) -> AudioSampleResult<spectrograms::LinearPowerSpectrogram> {
        self.linear_spectrogram::<spectrograms::Power>(params, None)
    }

    /// Shorthand for [`linear_spectrogram`](crate::operations::AudioTransforms::linear_spectrogram) with `Decibels` amplitude scale.
    ///
    /// Values are 10·log10(max(P, 10^(floor_db/10))) of the power P, in dB re 1:
    /// `floor_db` is an absolute floor, not a range below the peak (no `top_db`).
    ///
    /// # Examples
    /// ```
    /// use audio_samples::{AudioSamples, AudioTransforms, nzu, sample_rate, sine_wave};
    /// use spectrograms::{LogParams, SpectrogramParams, StftParams, WindowType};
    /// use std::time::Duration;
    ///
    /// let audio = sine_wave::<f64>(440.0, Duration::from_millis(100), sample_rate!(44100), 0.8);
    /// let stft = StftParams::new(nzu!(1024), nzu!(256), WindowType::Hanning, true).unwrap();
    /// let params = SpectrogramParams::new(stft, audio.sample_rate_hz()).unwrap();
    /// let db = LogParams::new(-80.0).unwrap();
    /// let spect = audio.linear_db_spectrogram(&params, &db).unwrap();
    /// assert!(spect.data().nrows() > 0);
    /// ```
    #[inline]
    fn linear_db_spectrogram(
        &self,
        params: &SpectrogramParams,
        db: &LogParams,
    ) -> AudioSampleResult<spectrograms::LinearDbSpectrogram> {
        self.linear_spectrogram::<spectrograms::Decibels>(params, Some(db))
    }

    /// Shorthand for [`log_frequency_spectrogram`](crate::operations::AudioTransforms::log_frequency_spectrogram) with `Power` amplitude scale.
    ///
    /// # Examples
    /// ```
    /// use audio_samples::{AudioSamples, AudioTransforms, nzu, sample_rate, sine_wave};
    /// use spectrograms::{LogHzParams, SpectrogramParams, StftParams, WindowType};
    /// use std::time::Duration;
    ///
    /// let audio = sine_wave::<f64>(440.0, Duration::from_millis(100), sample_rate!(44100), 0.8);
    /// let stft = StftParams::new(nzu!(1024), nzu!(256), WindowType::Hanning, true).unwrap();
    /// let params = SpectrogramParams::new(stft, audio.sample_rate_hz()).unwrap();
    /// let loghz = LogHzParams::new(nzu!(64), 80.0, 8_000.0).unwrap();
    /// let spect = audio.loghz_power_spectrogram(&params, &loghz).unwrap();
    /// assert!(spect.data().nrows() > 0);
    /// ```
    #[inline]
    fn loghz_power_spectrogram(
        &self,
        params: &SpectrogramParams,
        loghz: &LogHzParams,
    ) -> AudioSampleResult<spectrograms::LogHzPowerSpectrogram> {
        self.log_frequency_spectrogram::<spectrograms::Power>(params, loghz, None)
    }

    /// Shorthand for [`log_frequency_spectrogram`](crate::operations::AudioTransforms::log_frequency_spectrogram) with `Magnitude` amplitude scale.
    ///
    /// # Examples
    /// ```
    /// use audio_samples::{AudioSamples, AudioTransforms, nzu, sample_rate, sine_wave};
    /// use spectrograms::{LogHzParams, SpectrogramParams, StftParams, WindowType};
    /// use std::time::Duration;
    ///
    /// let audio = sine_wave::<f64>(440.0, Duration::from_millis(100), sample_rate!(44100), 0.8);
    /// let stft = StftParams::new(nzu!(1024), nzu!(256), WindowType::Hanning, true).unwrap();
    /// let params = SpectrogramParams::new(stft, audio.sample_rate_hz()).unwrap();
    /// let loghz = LogHzParams::new(nzu!(64), 80.0, 8_000.0).unwrap();
    /// let spect = audio.loghz_magnitude_spectrogram(&params, &loghz).unwrap();
    /// assert!(spect.data().nrows() > 0);
    /// ```
    #[inline]
    fn loghz_magnitude_spectrogram(
        &self,
        params: &SpectrogramParams,
        loghz: &LogHzParams,
    ) -> AudioSampleResult<spectrograms::LogHzMagnitudeSpectrogram> {
        self.log_frequency_spectrogram::<spectrograms::Magnitude>(params, loghz, None)
    }

    /// Shorthand for [`log_frequency_spectrogram`](crate::operations::AudioTransforms::log_frequency_spectrogram) with `Decibels` amplitude scale.
    ///
    /// Values are 10·log10(max(P, 10^(floor_db/10))) of the power P, in dB re 1:
    /// `floor_db` is an absolute floor, not a range below the peak (no `top_db`).
    ///
    /// # Examples
    /// ```
    /// use audio_samples::{AudioSamples, AudioTransforms, nzu, sample_rate, sine_wave};
    /// use spectrograms::{LogHzParams, LogParams, SpectrogramParams, StftParams, WindowType};
    /// use std::time::Duration;
    ///
    /// let audio = sine_wave::<f64>(440.0, Duration::from_millis(100), sample_rate!(44100), 0.8);
    /// let stft = StftParams::new(nzu!(1024), nzu!(256), WindowType::Hanning, true).unwrap();
    /// let params = SpectrogramParams::new(stft, audio.sample_rate_hz()).unwrap();
    /// let loghz = LogHzParams::new(nzu!(64), 80.0, 8_000.0).unwrap();
    /// let db = LogParams::new(-80.0).unwrap();
    /// let spect = audio.loghz_db_spectrogram(&params, &loghz, &db).unwrap();
    /// assert!(spect.data().nrows() > 0);
    /// ```
    #[inline]
    fn loghz_db_spectrogram(
        &self,
        params: &SpectrogramParams,
        loghz: &LogHzParams,
        db: &LogParams,
    ) -> AudioSampleResult<spectrograms::LogHzDbSpectrogram> {
        self.log_frequency_spectrogram::<spectrograms::Decibels>(params, loghz, Some(db))
    }

    /// Shorthand for [`mel_spectrogram`](crate::operations::AudioTransforms::mel_spectrogram) with `Magnitude` amplitude scale.
    ///
    /// # Examples
    /// ```
    /// use audio_samples::{AudioSamples, AudioTransforms, nzu, sample_rate, sine_wave};
    /// use spectrograms::{MelParams, SpectrogramParams, StftParams, WindowType};
    /// use std::time::Duration;
    ///
    /// let audio = sine_wave::<f64>(440.0, Duration::from_millis(100), sample_rate!(44100), 0.8);
    /// let stft = StftParams::new(nzu!(1024), nzu!(256), WindowType::Hanning, true).unwrap();
    /// let params = SpectrogramParams::new(stft, audio.sample_rate_hz()).unwrap();
    /// let mel = MelParams::new(nzu!(40), 0.0, 8_000.0).unwrap();
    /// let spect = audio.mel_mag_spectrogram(&params, &mel).unwrap();
    /// assert!(spect.data().nrows() > 0);
    /// ```
    #[inline]
    fn mel_mag_spectrogram(
        &self,
        params: &SpectrogramParams,
        mel: &MelParams,
    ) -> AudioSampleResult<spectrograms::MelMagnitudeSpectrogram> {
        self.mel_spectrogram(params, mel, None)
    }

    /// Shorthand for [`mel_spectrogram`](crate::operations::AudioTransforms::mel_spectrogram) with `Decibels` amplitude scale.
    ///
    /// Values are 10·log10(max(P, 10^(floor_db/10))) of the power P, in dB re 1:
    /// `floor_db` is an absolute floor, not a range below the peak (no `top_db`).
    ///
    /// # Examples
    /// ```
    /// use audio_samples::{AudioSamples, AudioTransforms, nzu, sample_rate, sine_wave};
    /// use spectrograms::{LogParams, MelParams, SpectrogramParams, StftParams, WindowType};
    /// use std::time::Duration;
    ///
    /// let audio = sine_wave::<f64>(440.0, Duration::from_millis(100), sample_rate!(44100), 0.8);
    /// let stft = StftParams::new(nzu!(1024), nzu!(256), WindowType::Hanning, true).unwrap();
    /// let params = SpectrogramParams::new(stft, audio.sample_rate_hz()).unwrap();
    /// let mel = MelParams::new(nzu!(40), 0.0, 8_000.0).unwrap();
    /// let db = LogParams::new(-80.0).unwrap();
    /// let spect = audio.mel_db_spectrogram(&params, &mel, &db).unwrap();
    /// assert!(spect.data().nrows() > 0);
    /// ```
    #[inline]
    fn mel_db_spectrogram(
        &self,
        params: &SpectrogramParams,
        mel: &MelParams,
        db: &LogParams,
    ) -> AudioSampleResult<spectrograms::LogMelSpectrogram> {
        self.mel_spectrogram(params, mel, Some(db))
    }

    /// Shorthand for [`mel_spectrogram`](crate::operations::AudioTransforms::mel_spectrogram) with `Power` amplitude scale.
    ///
    /// # Examples
    /// ```
    /// use audio_samples::{AudioSamples, AudioTransforms, nzu, sample_rate, sine_wave};
    /// use spectrograms::{MelParams, SpectrogramParams, StftParams, WindowType};
    /// use std::time::Duration;
    ///
    /// let audio = sine_wave::<f64>(440.0, Duration::from_millis(100), sample_rate!(44100), 0.8);
    /// let stft = StftParams::new(nzu!(1024), nzu!(256), WindowType::Hanning, true).unwrap();
    /// let params = SpectrogramParams::new(stft, audio.sample_rate_hz()).unwrap();
    /// let mel = MelParams::new(nzu!(40), 0.0, 8_000.0).unwrap();
    /// let spect = audio.mel_power_spectrogram(&params, &mel).unwrap();
    /// assert!(spect.data().nrows() > 0);
    /// ```
    #[inline]
    fn mel_power_spectrogram(
        &self,
        params: &SpectrogramParams,
        mel: &MelParams,
    ) -> AudioSampleResult<spectrograms::MelPowerSpectrogram> {
        self.mel_spectrogram(params, mel, None)
    }

    /// Shorthand for [`gammatone_spectrogram`](crate::operations::AudioTransforms::gammatone_spectrogram) with `Magnitude` amplitude scale.
    ///
    /// # Examples
    /// ```
    /// use audio_samples::{AudioSamples, AudioTransforms, nzu, sample_rate, sine_wave};
    /// use spectrograms::{GammatoneParams, SpectrogramParams, StftParams, WindowType};
    /// use std::time::Duration;
    ///
    /// let audio = sine_wave::<f64>(440.0, Duration::from_millis(100), sample_rate!(44100), 0.8);
    /// let stft = StftParams::new(nzu!(1024), nzu!(256), WindowType::Hanning, true).unwrap();
    /// let params = SpectrogramParams::new(stft, audio.sample_rate_hz()).unwrap();
    /// let gammatone = GammatoneParams::new(nzu!(32), 80.0, 8_000.0).unwrap();
    /// let spect = audio.gammatone_magnitude_spectrogram(&params, &gammatone).unwrap();
    /// assert!(spect.data().nrows() > 0);
    /// ```
    #[inline]
    fn gammatone_magnitude_spectrogram(
        &self,
        params: &SpectrogramParams,
        gammatone_params: &GammatoneParams,
    ) -> AudioSampleResult<spectrograms::GammatoneMagnitudeSpectrogram> {
        self.gammatone_spectrogram::<spectrograms::Magnitude>(params, gammatone_params, None)
    }

    /// Shorthand for [`gammatone_spectrogram`](crate::operations::AudioTransforms::gammatone_spectrogram) with `Power` amplitude scale.
    ///
    /// # Examples
    /// ```
    /// use audio_samples::{AudioSamples, AudioTransforms, nzu, sample_rate, sine_wave};
    /// use spectrograms::{GammatoneParams, SpectrogramParams, StftParams, WindowType};
    /// use std::time::Duration;
    ///
    /// let audio = sine_wave::<f64>(440.0, Duration::from_millis(100), sample_rate!(44100), 0.8);
    /// let stft = StftParams::new(nzu!(1024), nzu!(256), WindowType::Hanning, true).unwrap();
    /// let params = SpectrogramParams::new(stft, audio.sample_rate_hz()).unwrap();
    /// let gammatone = GammatoneParams::new(nzu!(32), 80.0, 8_000.0).unwrap();
    /// let spect = audio.gammatone_power_spectrogram(&params, &gammatone).unwrap();
    /// assert!(spect.data().nrows() > 0);
    /// ```
    #[inline]
    fn gammatone_power_spectrogram(
        &self,
        params: &SpectrogramParams,
        gammatone_params: &GammatoneParams,
    ) -> AudioSampleResult<spectrograms::GammatonePowerSpectrogram> {
        self.gammatone_spectrogram::<spectrograms::Power>(params, gammatone_params, None)
    }

    /// Shorthand for [`gammatone_spectrogram`](crate::operations::AudioTransforms::gammatone_spectrogram) with `Decibels` amplitude scale.
    ///
    /// Values are 10·log10(max(P, 10^(floor_db/10))) of the power P, in dB re 1:
    /// `floor_db` is an absolute floor, not a range below the peak (no `top_db`).
    ///
    /// # Examples
    /// ```
    /// use audio_samples::{AudioSamples, AudioTransforms, nzu, sample_rate, sine_wave};
    /// use spectrograms::{GammatoneParams, LogParams, SpectrogramParams, StftParams, WindowType};
    /// use std::time::Duration;
    ///
    /// let audio = sine_wave::<f64>(440.0, Duration::from_millis(100), sample_rate!(44100), 0.8);
    /// let stft = StftParams::new(nzu!(1024), nzu!(256), WindowType::Hanning, true).unwrap();
    /// let params = SpectrogramParams::new(stft, audio.sample_rate_hz()).unwrap();
    /// let gammatone = GammatoneParams::new(nzu!(32), 80.0, 8_000.0).unwrap();
    /// let db = LogParams::new(-80.0).unwrap();
    /// let spect = audio.gammatone_db_spectrogram(&params, &gammatone, &db).unwrap();
    /// assert!(spect.data().nrows() > 0);
    /// ```
    #[inline]
    fn gammatone_db_spectrogram(
        &self,
        params: &SpectrogramParams,
        gammatone_params: &GammatoneParams,
        db: &LogParams,
    ) -> AudioSampleResult<spectrograms::GammatoneDbSpectrogram> {
        self.gammatone_spectrogram::<spectrograms::Decibels>(params, gammatone_params, Some(db))
    }

    /// Shorthand for [`cqt_spectrogram`](crate::operations::AudioTransforms::cqt_spectrogram) with `Magnitude` amplitude scale.
    ///
    /// # Examples
    /// ```
    /// use audio_samples::{AudioSamples, AudioTransforms, nzu, sample_rate, sine_wave};
    /// use spectrograms::{CqtParams, SpectrogramParams, StftParams, WindowType};
    /// use std::time::Duration;
    ///
    /// let audio = sine_wave::<f64>(440.0, Duration::from_millis(200), sample_rate!(44100), 0.8);
    /// let stft = StftParams::new(nzu!(1024), nzu!(256), WindowType::Hanning, true).unwrap();
    /// let params = SpectrogramParams::new(stft, audio.sample_rate_hz()).unwrap();
    /// let cqt = CqtParams::new(nzu!(12), nzu!(7), 32.7).unwrap();
    /// let spect = audio.cqt_magnitude_spectrogram(&params, &cqt).unwrap();
    /// assert!(spect.data().nrows() > 0);
    /// ```
    #[inline]
    fn cqt_magnitude_spectrogram(
        &self,
        params: &SpectrogramParams,
        cqt: &CqtParams,
    ) -> AudioSampleResult<spectrograms::CqtMagnitudeSpectrogram> {
        self.cqt_spectrogram::<spectrograms::Magnitude>(params, cqt, None)
    }

    /// Shorthand for [`cqt_spectrogram`](crate::operations::AudioTransforms::cqt_spectrogram) with `Power` amplitude scale.
    ///
    /// # Examples
    /// ```
    /// use audio_samples::{AudioSamples, AudioTransforms, nzu, sample_rate, sine_wave};
    /// use spectrograms::{CqtParams, SpectrogramParams, StftParams, WindowType};
    /// use std::time::Duration;
    ///
    /// let audio = sine_wave::<f64>(440.0, Duration::from_millis(200), sample_rate!(44100), 0.8);
    /// let stft = StftParams::new(nzu!(1024), nzu!(256), WindowType::Hanning, true).unwrap();
    /// let params = SpectrogramParams::new(stft, audio.sample_rate_hz()).unwrap();
    /// let cqt = CqtParams::new(nzu!(12), nzu!(7), 32.7).unwrap();
    /// let spect = audio.cqt_power_spectrogram(&params, &cqt).unwrap();
    /// assert!(spect.data().nrows() > 0);
    /// ```
    #[inline]
    fn cqt_power_spectrogram(
        &self,
        params: &SpectrogramParams,
        cqt: &CqtParams,
    ) -> AudioSampleResult<spectrograms::CqtPowerSpectrogram> {
        self.cqt_spectrogram::<spectrograms::Power>(params, cqt, None)
    }

    /// Shorthand for [`cqt_spectrogram`](crate::operations::AudioTransforms::cqt_spectrogram) with `Decibels` amplitude scale.
    ///
    /// Values are 10·log10(max(P, 10^(floor_db/10))) of the power P, in dB re 1:
    /// `floor_db` is an absolute floor, not a range below the peak (no `top_db`).
    ///
    /// # Examples
    /// ```
    /// use audio_samples::{AudioSamples, AudioTransforms, nzu, sample_rate, sine_wave};
    /// use spectrograms::{CqtParams, LogParams, SpectrogramParams, StftParams, WindowType};
    /// use std::time::Duration;
    ///
    /// let audio = sine_wave::<f64>(440.0, Duration::from_millis(200), sample_rate!(44100), 0.8);
    /// let stft = StftParams::new(nzu!(1024), nzu!(256), WindowType::Hanning, true).unwrap();
    /// let params = SpectrogramParams::new(stft, audio.sample_rate_hz()).unwrap();
    /// let cqt = CqtParams::new(nzu!(12), nzu!(7), 32.7).unwrap();
    /// let db = LogParams::new(-80.0).unwrap();
    /// let spect = audio.cqt_db_spectrogram(&params, &cqt, &db).unwrap();
    /// assert!(spect.data().nrows() > 0);
    /// ```
    #[inline]
    fn cqt_db_spectrogram(
        &self,
        params: &SpectrogramParams,
        cqt: &CqtParams,
        db: &LogParams,
    ) -> AudioSampleResult<spectrograms::CqtDbSpectrogram> {
        self.cqt_spectrogram::<spectrograms::Decibels>(params, cqt, Some(db))
    }

    /// Linear convolution of this signal with `other` via FFT.
    ///
    /// See [`AudioTransforms::convolve`](crate::operations::AudioTransforms::convolve).
    ///
    /// # Errors
    /// - [`crate::AudioSampleError::Layout`] if either signal is multi-channel.
    /// - Errors propagated from the underlying FFT.
    #[inline]
    fn convolve(&self, other: &Self) -> AudioSampleResult<AudioSamples<'static, Self::Sample>> {
        let result_f64 = convolve_or_deconvolve_mono(self, other, |a, b| Ok(fft_convolve(a, b)?))?;
        from_f64_mono::<T>(result_f64, self.sample_rate())
    }

    /// Regularised spectral-division deconvolution.
    ///
    /// See [`AudioTransforms::deconvolve`](crate::operations::AudioTransforms::deconvolve).
    ///
    /// # Errors
    /// - [`crate::AudioSampleError::Layout`] if either signal is multi-channel.
    /// - Errors propagated from the underlying FFT.
    #[inline]
    fn deconvolve(
        &self,
        denominator: &Self,
        regularization: f64,
    ) -> AudioSampleResult<AudioSamples<'static, Self::Sample>> {
        let result_f64 = convolve_or_deconvolve_mono(self, denominator, |a, b| {
            Ok(fft_deconvolve(a, b, regularization)?)
        })?;
        from_f64_mono::<T>(result_f64, self.sample_rate())
    }

    /// Decomposes a complex spectrogram into magnitude and phase.
    ///
    /// Given a complex matrix `D`, returns `(S, P)` with S = |D|^p (p = `power`)
    /// and P = D/|D| (unit-magnitude phase factors), so that `D = S^(1/p) · P`
    /// elementwise (`D = S · P` for the default p = 1).
    /// Bins where the magnitude is zero are assigned a phase of `1 + 0i`.
    ///
    /// # Arguments
    /// - `complex_spect` - the complex STFT or FFT matrix.
    /// - `power` - exponent applied to the magnitude values.  `None`
    ///   defaults to 1 (raw magnitude).
    ///
    /// # Returns
    /// `(magnitude, phase)` - the magnitude matrix (real-valued) and
    /// the phase matrix (complex unit-magnitude).
    ///
    /// # Examples
    /// ```
    /// use audio_samples::{AudioSamples, AudioTransforms, nzu, sample_rate, sine_wave};
    /// use std::time::Duration;
    ///
    /// // Use a short signal so n_fft (1024) is larger than the signal length (441 samples).
    /// let audio = sine_wave::<f64>(440.0, Duration::from_millis(10), sample_rate!(44100), 0.8);
    /// let spectrum = audio.fft(nzu!(1024)).unwrap();
    /// let (mag, phase) = AudioSamples::<f64>::magphase(&spectrum, None);
    /// assert_eq!(mag.shape(), phase.shape());
    /// ```
    #[inline]
    fn magphase(
        complex_spect: &Array2<Complex<f64>>,
        power: Option<NonZeroUsize>,
    ) -> (Array2<f64>, Array2<Complex<f64>>) {
        magphase(complex_spect.view(), power)
    }
}

/// Splits a complex spectrogram into magnitude and phase, equivalent to
/// `librosa.magphase(D, power=power)`.
///
/// This is [`AudioTransforms::magphase`] on a view, so callers holding a
/// borrowed or Fortran-order matrix (a numpy array, for example) need no copy.
/// Both outputs have the memory layout of the input when it is contiguous.
///
/// # Arguments
///
/// * `complex_spect` - Complex spectrogram.
/// * `power` - Exponent of the magnitude (`None` means 1).
///
/// # Returns
///
/// `(magnitude ** power, phase)` with `phase = D / |D|`, and `1 + 0i` where
/// `|D| == 0`.
///
/// # Examples
///
/// ```rust
/// use audio_samples::operations::transforms::magphase;
/// use ndarray::array;
/// use num_complex::Complex;
///
/// let d = array![[Complex::new(3.0, 4.0), Complex::new(0.0, 0.0)]];
/// let (mag, phase) = magphase(d.view(), None);
/// assert_eq!(mag, array![[5.0, 0.0]]);
/// assert_eq!(phase, array![[Complex::new(0.6, 0.8), Complex::new(1.0, 0.0)]]);
/// ```
#[must_use]
pub fn magphase<F: num_traits::Float>(
    complex_spect: ArrayView2<'_, Complex<F>>,
    power: Option<NonZeroUsize>,
) -> (Array2<F>, Array2<Complex<F>>) {
    // One fused pass in the memory order of the input (outputs in the same
    // layout, so a Fortran-order spectrogram is not traversed with strides).
    // |z| = sqrt(re² + im²) (within 1 ulp of `hypot`, and branch-free so the
    // loop vectorises) wherever re² + im² is a normal finite number; the rare
    // other elements (overflow, or so small that the squares underflow) are
    // redone with `hypot` afterwards. The phase is `D / m + 0` for `m != 0`
    // and `D / 1 + 1` for `m == 0` (librosa's `zeros_to_ones` trick), where
    // `x + 0.0` turns `-0.0` into `+0.0`.
    use ndarray::ShapeBuilder;
    let dim = complex_spect.dim();
    let fortran = !complex_spect.is_standard_layout() && complex_spect.t().is_standard_layout();
    let (zero, one) = (F::zero(), F::one());
    let mut mag = Array2::<F>::zeros(dim.set_f(fortran));
    let mut phase = Array2::<Complex<F>>::zeros(dim.set_f(fortran));
    let split = |z: &Complex<F>, m: &mut F, p: &mut Complex<F>| {
        let s = z.re * z.re + z.im * z.im;
        let ok = s >= F::min_positive_value() && s <= F::max_value();
        let mv = if ok { s.sqrt() } else { F::nan() };
        let (m_nz, zo) = if mv == zero { (one, one) } else { (mv, zero) };
        *m = mv;
        *p = Complex {
            re: z.re / m_nz + zo,
            im: z.im / m_nz,
        };
    };
    ndarray::Zip::from(&mut mag)
        .and(&mut phase)
        .and(&complex_spect)
        .for_each(|m, p, z| split(z, m, p));
    // Elements whose squares overflow or underflow (or a NaN input): hypot.
    if mag.iter().any(|m| m.is_nan()) {
        ndarray::Zip::from(&mut mag)
            .and(&mut phase)
            .and(&complex_spect)
            .for_each(|m, p, z| {
                if m.is_nan() {
                    let mv = z.norm();
                    let (m_nz, zo) = if mv == zero { (one, one) } else { (mv, zero) };
                    *m = mv;
                    *p = Complex {
                        re: z.re / m_nz + zo,
                        im: z.im / m_nz,
                    };
                }
            });
    }

    // Raise magnitude to the given power (`powf(1.0)` is the identity).
    if let Some(p) = power.filter(|p| p.get() != 1) {
        let p = F::from(p.get()).unwrap_or_else(F::infinity);
        mag.mapv_inplace(|x| x.powf(p));
    }

    (mag, phase)
}

/// [`magphase`] with any exponent, equivalent to `librosa.magphase(D, power=power)`
/// for a fractional or negative `power`.
///
/// # Arguments
///
/// * `complex_spect` - Complex spectrogram.
/// * `power` - Exponent of the magnitude.
///
/// # Returns
///
/// `(magnitude ** power, phase)` with `phase = D / |D|`, and `1 + 0i` where
/// `|D| == 0`. A zero magnitude with a negative `power` gives infinity.
///
/// # Examples
///
/// ```rust
/// use audio_samples::operations::transforms::magphase_powf;
/// use ndarray::array;
/// use num_complex::Complex;
///
/// let d = array![[Complex::new(3.0, 4.0), Complex::new(0.0, 0.0)]];
/// let (mag, phase) = magphase_powf(d.view(), 0.5);
/// assert_eq!(mag, array![[5.0_f64.sqrt(), 0.0]]);
/// assert_eq!(phase, array![[Complex::new(0.6, 0.8), Complex::new(1.0, 0.0)]]);
/// ```
#[must_use]
pub fn magphase_powf<F: num_traits::Float>(
    complex_spect: ArrayView2<'_, Complex<F>>,
    power: F,
) -> (Array2<F>, Array2<Complex<F>>) {
    let (mut mag, phase) = magphase(complex_spect, None);
    let half = F::from(0.5).unwrap_or_else(F::nan);
    if power == half {
        mag.mapv_inplace(F::sqrt);
    } else if power != F::one() {
        mag.mapv_inplace(|x| x.powf(power));
    }
    (mag, phase)
}

/// Extracts the single mono channel of `audio` as an owned `Vec<f64>`, erroring
/// on multi-channel input.
fn mono_f64<T: StandardSample>(audio: &AudioSamples<'_, T>) -> AudioSampleResult<Vec<f64>> {
    if audio.num_channels().get() != 1 {
        return Err(AudioSampleError::Layout(
            LayoutError::channel_count_unsupported(
                "convolve/deconvolve (mono only)",
                ChannelRequirement::Mono,
                audio.num_channels().get(),
            ),
        ));
    }
    let working = audio.to_format::<f64>();
    let ch = working.as_slice().expect("validated single channel");
    Ok(ch.to_vec())
}

/// Runs an FFT convolution-style op on the mono channels of `a` and `b`.
fn convolve_or_deconvolve_mono<T, F>(
    a: &AudioSamples<'_, T>,
    b: &AudioSamples<'_, T>,
    op: F,
) -> AudioSampleResult<Vec<f64>>
where
    T: StandardSample,
    F: FnOnce(
        &non_empty_slice::NonEmptySlice<f64>,
        &non_empty_slice::NonEmptySlice<f64>,
    ) -> AudioSampleResult<non_empty_slice::NonEmptyVec<f64>>,
{
    let a_vec = mono_f64(a)?;
    let b_vec = mono_f64(b)?;
    // AudioSamples::new_mono rejects empty data, so this error path is unreachable for well-formed inputs.
    let a_ne = non_empty_slice::NonEmptySlice::new(a_vec.as_slice()).ok_or_else(|| {
        AudioSampleError::Parameter(ParameterError::invalid_value("self", "empty signal"))
    })?;
    let b_ne = non_empty_slice::NonEmptySlice::new(b_vec.as_slice()).ok_or_else(|| {
        AudioSampleError::Parameter(ParameterError::invalid_value("other", "empty signal"))
    })?;
    Ok(op(a_ne, b_ne)?.into_vec())
}

/// Wraps a `Vec<f64>` back into a mono `AudioSamples<T>` at `sample_rate`.
fn from_f64_mono<T: StandardSample>(
    data: Vec<f64>,
    sample_rate: std::num::NonZeroU32,
) -> AudioSampleResult<AudioSamples<'static, T>> {
    use non_empty_slice::NonEmptyVec;
    let ne_vec = NonEmptyVec::new(data).map_err(|_| {
        AudioSampleError::Parameter(ParameterError::invalid_value("data", "empty result"))
    })?;
    Ok(AudioSamples::<T>::from_mono_vec::<f64>(ne_vec, sample_rate))
}

#[cfg(test)]
mod convolution_tests {
    use crate::{AudioSampleError, AudioSamples, AudioTransforms, LayoutError, sample_rate};
    use ndarray::array;

    #[test]
    fn convolve_with_unit_impulse_is_identity() {
        let signal =
            AudioSamples::new_mono(array![1.0_f64, 2.0, 3.0], sample_rate!(48000)).unwrap();
        let impulse = AudioSamples::new_mono(array![1.0_f64], sample_rate!(48000)).unwrap();
        let out = signal.convolve(&impulse).unwrap();
        assert_eq!(out.sample_rate().get(), 48000);
        let ch = out.as_slice().unwrap();
        assert!((ch[0] - 1.0).abs() < 1e-9);
        assert!((ch[1] - 2.0).abs() < 1e-9);
        assert!((ch[2] - 3.0).abs() < 1e-9);
    }

    #[test]
    fn deconvolve_inverts_convolve() {
        let excitation = AudioSamples::new_mono(
            array![1.0_f64, 0.7, -0.3, 0.2, 0.9, -0.5, 0.1, 0.4],
            sample_rate!(48000),
        )
        .unwrap();
        let system =
            AudioSamples::new_mono(array![0.0_f64, 0.0, 1.0, 0.5], sample_rate!(48000)).unwrap();
        let recorded = excitation.convolve(&system).unwrap();
        let recovered = recorded.deconvolve(&excitation, 0.0).unwrap();
        let r = recovered.as_slice().unwrap();
        for (i, want) in [0.0, 0.0, 1.0, 0.5].iter().enumerate() {
            assert!(
                (r[i] - want).abs() < 1e-6,
                "tap {i}: got {}, want {want}",
                r[i]
            );
        }
    }

    #[test]
    fn multichannel_input_is_rejected() {
        // 2 channels, 4 samples each
        let stereo = AudioSamples::new_multi_channel(
            array![[1.0_f64, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]],
            sample_rate!(48000),
        )
        .unwrap();
        let mono = AudioSamples::new_mono(array![1.0_f64], sample_rate!(48000)).unwrap();
        let err = stereo.convolve(&mono).unwrap_err();
        assert!(
            matches!(
                err,
                AudioSampleError::Layout(LayoutError::ChannelCountUnsupported { .. })
            ),
            "expected Layout(ChannelCountUnsupported), got: {err:?}"
        );
    }
}

#[cfg(test)]
mod psd_tests {
    use super::Psd;

    #[test]
    fn psd_accessors_and_into_parts_round_trip() {
        let psd = Psd::new(vec![0.0, 10.0, 20.0], vec![1.0, 0.5, 0.25]);
        assert_eq!(psd.frequencies(), &[0.0, 10.0, 20.0]);
        assert_eq!(psd.density(), &[1.0, 0.5, 0.25]);

        let (freqs, density) = psd.into_parts();
        assert_eq!(freqs, vec![0.0, 10.0, 20.0]);
        assert_eq!(density, vec![1.0, 0.5, 0.25]);
    }
}

#[cfg(test)]
mod magphase_tests {
    use super::magphase;
    use ndarray::Array2;
    use num_complex::Complex;

    /// The fused pass agrees with `hypot` to 1 ulp, takes the `hypot` path
    /// for squares that overflow or underflow, follows librosa at zero, and
    /// gives the same result for either memory order.
    #[test]
    fn magphase_edge_cases_and_layouts() {
        let vals = [
            Complex::new(3.0, 4.0),
            Complex::new(0.0, 0.0),
            Complex::new(-0.0, 0.0),
            Complex::new(1e200, -3e200),
            Complex::new(1e-200, 2e-200),
            Complex::new(-1.5, 1e-300),
            Complex::new(f64::NAN, 1.0),
            Complex::new(0.1, -0.7),
        ];
        let d = Array2::from_shape_fn((4, 6), |(i, j)| {
            vals[(i * 6 + j) % vals.len()] * (1.0 + j as f64)
        });
        let (m, p) = magphase(d.view(), None);
        for ((z, mv), pv) in d.iter().zip(&m).zip(&p) {
            let h = z.norm();
            if h.is_nan() {
                assert!(mv.is_nan());
                continue;
            }
            assert!((mv - h).abs() <= h * f64::EPSILON, "{z}: {mv} vs {h}");
            if h == 0.0 {
                assert_eq!(*pv, Complex::new(z.re + 1.0, z.im));
            } else {
                assert!((pv.norm() - 1.0).abs() < 4.0 * f64::EPSILON, "{z}: {pv}");
            }
        }
        // Fortran-order input gives the same values.
        let f = d.t().as_standard_layout().reversed_axes().to_owned();
        assert!(!f.is_standard_layout());
        let (mf, pf) = magphase(f.view(), None);
        assert!(
            m.iter()
                .zip(&mf)
                .all(|(a, b)| a == b || (a.is_nan() && b.is_nan()))
        );
        assert!(
            p.iter()
                .zip(&pf)
                .all(|(a, b)| a == b || (a.re.is_nan() && b.re.is_nan()))
        );
    }
}
