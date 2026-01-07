//! Spectral analysis and frequency-domain transformations for [`AudioSamples`].
//!
//! ## What
//!
//! This module implements the [`AudioTransforms`] trait, providing FFT-based
//! spectral analysis and spectrogram computation for audio data. All spectral
//! operations delegate to the [`spectrograms`] crate.
//!
//! ## Why
//!
//! Frequency-domain analysis is central to audio processing --- spectrograms,
//! MFCCs, chromagrams, and pitch detection all begin with a transform into the
//! spectral domain.
//!
//! ## How
//!
//! All operations are available on any [`AudioSamples<T>`] where `T` is a
//! supported sample type (`u8`, `i16`, `I24`, `i32`, `f32`, `f64`).
//! Most spectral methods require mono input; multi-channel signals must be
//! mixed or channel-selected before use. The one exception is
//! [`fft`](AudioTransforms::fft), which transforms every channel
//! independently.
//!
//! ### Available operations
//!
//! | Method | Description | Multi-channel? |
//! |--------|-------------|:--------------:|
//! | [`fft`](AudioTransforms::fft) | Full-signal FFT | Yes |
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
//! | [`magphase`](AudioTransforms::magphase) | Magnitude / phase decomposition | — |
//!
//! ## Example
//!
//! ```rust
//! use audio_samples::{AudioSamples, AudioTransforms, sample_rate};
//! use ndarray::array;
//! use std::num::NonZeroUsize;
//!
//! let data  = array![1.0f32, 0.0, -1.0, 0.0];
//! let audio = AudioSamples::new_mono(data, sample_rate!(44100)).unwrap();
//!
//! let spectrum = audio.fft(NonZeroUsize::new(4).unwrap()).unwrap();
//! assert_eq!(spectrum.shape()[0], 1); // one row per channel
//! ```
//!
//! ## Error Handling
//!
//! All operations return [`crate::AudioSampleResult`]. Mono-only methods
//! return [`crate::AudioSampleError::Layout`] or
//! [`crate::AudioSampleError::Parameter`] when given multi-channel input.
//! Failures originating inside [`spectrograms`] are propagated as
//! [`crate::AudioSampleError::Processing`].
//!
//! ## See Also
//!
//! - [`AudioSamples`]: The core audio data type.
//! - [`AudioTransforms`]: The trait defining all spectral operations.
//!
//! [`AudioTransforms`]: crate::operations::traits::AudioTransforms
use std::num::{NonZeroU32, NonZeroUsize};

use crate::operations::traits::AudioTransforms;
use crate::{
    AudioSampleError, AudioSampleResult, AudioSamples, AudioTypeConversion, LayoutError,
    ParameterError, StandardSample,
};

use ndarray::{Array1, Array2, Axis};
use non_empty_slice::NonEmptySlice;
use num_complex::Complex;
use spectrograms::{
    AmpScaleSpec, ChromaParams, Chromagram, CqtParams, CqtResult, CqtSpectrogram, FftPlanner,
    Gammatone, GammatoneParams, GammatoneSpectrogram, LinearHz, LinearSpectrogram, LogHz,
    LogHzParams, LogHzSpectrogram, LogParams, MelParams, MelSpectrogram, Mfcc, MfccParams,
    Spectrogram, SpectrogramParams, StftParams, StftPlan, StftResult, WindowType,
};

impl<T> AudioTransforms for AudioSamples<'_, T>
where
    T: StandardSample,
{
    /// Computes the Fast Fourier Transform of the audio signal.
    ///
    /// Each channel is transformed independently. The output has one row per
    /// channel containing the complex spectral bins.
    ///
    /// # Arguments
    /// - `n_fft` — FFT length in samples. If longer than the signal the input
    ///   is zero-padded internally.
    ///
    /// # Returns
    /// An `Array2<Complex<f64>>` where each row is the FFT of the
    /// corresponding channel.
    ///
    /// # Errors
    /// Returns an error if the FFT computation fails.
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
    fn fft(&self, n_fft: NonZeroUsize) -> AudioSampleResult<Array2<Complex<f64>>> {
        let working_samples = self.to_format::<f64>();

        let mut fft_planner = FftPlanner::new();

        let mut channel_ffts: Vec<Array1<Complex<f64>>> =
            Vec::with_capacity(self.num_channels().get() as usize);

        for ch in working_samples.channels() {
            let ch_slice = ch
                .as_slice()
                .expect("Can always get a mono channel as a slice");
            // safety: ch_slice is non-empty since audio has samples
            let working_ch_slice = unsafe { NonEmptySlice::new_unchecked(ch_slice) };
            let channel_fft: Array1<Complex<f64>> = fft_planner.fft(working_ch_slice, n_fft)?;
            channel_ffts.push(channel_fft);
        }

        ndarray::stack(
            Axis(0),
            &channel_ffts.iter().map(|a| a.view()).collect::<Vec<_>>(),
        )
        .map_err(Into::into)
    }

    /// Computes the Short-Time Fourier Transform (STFT) of a mono signal.
    ///
    /// The signal is divided into overlapping, windowed frames and each frame
    /// is transformed to the frequency domain. The returned [`StftResult`]
    /// carries both the complex matrix and the parameters required by
    /// [`istft`] for reconstruction.
    ///
    /// # Arguments
    /// - `params` — STFT configuration (FFT size, hop size, window function,
    ///   and centering behaviour). Field-level constraints are documented on
    ///   [`StftParams`].
    ///
    /// # Returns
    /// An [`StftResult`] containing the complex STFT matrix and metadata.
    ///
    /// # Errors
    /// - [`crate::AudioSampleError::Parameter`] if the signal is multi-channel.
    /// - Errors from the underlying [`spectrograms`] STFT computation.
    ///
    /// [`istft`]: Self::istft
    fn stft(&self, params: &StftParams) -> AudioSampleResult<StftResult> {
        if self.is_multi_channel() {
            return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
                "self",
                "STFT is only supported for mono audio samples",
            )));
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
    /// Uses overlap-add synthesis with the window and hop parameters stored
    /// in the [`StftResult`]. The output is a mono signal at the sample rate
    /// that was recorded during the forward transform.
    ///
    /// # Arguments
    /// - `stft` — the [`StftResult`] produced by a prior call to [`stft`].
    ///
    /// # Returns
    /// A mono [`AudioSamples`] at the original sample rate.
    ///
    /// # Errors
    /// Returns an error if the reconstruction fails (e.g. mismatched
    /// parameters inside the [`StftResult`]).
    ///
    /// [`stft`]: Self::stft
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
    /// Prefer the typed convenience methods —
    /// [`linear_magnitude_spectrogram`], [`linear_power_spectrogram`], or
    /// [`linear_db_spectrogram`] — for the most common amplitude scales.
    ///
    /// # Arguments
    /// - `params` — spectrogram parameters (window, hop, FFT size).
    /// - `db` — required when `AmpScale` is `Decibels`; ignored otherwise.
    ///
    /// # Returns
    /// A `Spectrogram<LinearHz, AmpScale>`.
    ///
    /// # Errors
    /// - [`crate::AudioSampleError::Layout`] if the signal is multi-channel.
    /// - Errors from the underlying spectrogram computation.
    fn linear_spectrogram<AmpScale>(
        &self,
        params: &SpectrogramParams,
        db: Option<&LogParams>,
    ) -> AudioSampleResult<Spectrogram<LinearHz, AmpScale>>
    where
        AmpScale: AmpScaleSpec + 'static,
    {
        if self.is_multi_channel() {
            return Err(AudioSampleError::Layout(LayoutError::invalid_operation(
                "linear_frequency_spectrogram",
                "Linear frequency spectrogram is only supported for mono audio samples",
            )));
        }
        let samples = self.cast_as_f64();
        let samples_slice = samples.as_slice().expect("Safe since we have ensured mono");
        // safety: Guaranteed non-empty since audio is mono and has samples
        let samples_slice = unsafe { NonEmptySlice::new_unchecked(samples_slice) };
        LinearSpectrogram::<AmpScale>::compute(samples_slice, params, db).map_err(Into::into)
    }

    /// Computes a log-frequency-spaced spectrogram.
    ///
    /// Prefer the typed convenience methods —
    /// [`loghz_power_spectrogram`], [`loghz_magnitude_spectrogram`], or
    /// [`loghz_db_spectrogram`] — for the most common amplitude scales.
    ///
    /// # Arguments
    /// - `params` — spectrogram parameters (window, hop, FFT size).
    /// - `loghz` — log-Hz frequency-axis configuration (min/max
    ///   frequency, number of bins). Field-level constraints are
    ///   documented on [`LogHzParams`].
    /// - `db` — required when `AmpScale` is `Decibels`; ignored otherwise.
    ///
    /// # Returns
    /// A `Spectrogram<LogHz, AmpScale>`.
    ///
    /// # Errors
    /// - [`crate::AudioSampleError::Layout`] if the signal is multi-channel.
    /// - Errors from the underlying spectrogram computation.
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
            return Err(AudioSampleError::Layout(LayoutError::invalid_operation(
                "log_frequency_spectrogram",
                "LogHz spectrogram is only supported for mono audio samples",
            )));
        }
        let samples: AudioSamples<'static, f64> = self.cast_as_f64();
        let samples_slice = samples.as_slice().expect("Safe since we have ensured mono");
        // safety: Guaranteed non-empty since audio is mono and has samples
        let samples_slice = unsafe { NonEmptySlice::new_unchecked(samples_slice) };
        LogHzSpectrogram::<AmpScale>::compute(samples_slice, params, loghz, db).map_err(Into::into)
    }

    /// Computes a mel-scaled spectrogram.
    ///
    /// The mel scale approximates human auditory perception by compressing
    /// high frequencies relative to low ones.  Prefer the typed
    /// convenience methods — [`mel_mag_spectrogram`],
    /// [`mel_power_spectrogram`], or [`mel_db_spectrogram`] — for the
    /// most common amplitude scales.
    ///
    /// # Arguments
    /// - `params` — spectrogram parameters (window, hop, FFT size).
    /// - `mel` — mel filter-bank configuration (number of bands,
    ///   frequency range). Field-level constraints are documented on
    ///   [`MelParams`].
    /// - `db` — required when `AmpScale` is `Decibels`; ignored otherwise.
    ///
    /// # Returns
    /// A [`MelSpectrogram<AmpScale>`].
    ///
    /// # Errors
    /// - [`crate::AudioSampleError::Layout`] if the signal is multi-channel.
    /// - Errors from the underlying spectrogram computation.
    ///
    /// ## See Also
    /// - [Mel scale — Wikipedia](https://en.wikipedia.org/wiki/Mel_scale)
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
            return Err(AudioSampleError::Layout(LayoutError::invalid_operation(
                "mel_spectrogram",
                "Mel spectrogram is only supported for mono audio samples",
            )));
        }
        let samples = self.cast_as_f64();
        let samples_slice = samples.as_slice().expect("Safe since we have ensured mono");
        // safety: Guaranteed non-empty since audio is mono and has samples
        let samples_slice = unsafe { NonEmptySlice::new_unchecked(samples_slice) };
        MelSpectrogram::<AmpScale>::compute(samples_slice, params, mel, db).map_err(Into::into)
    }

    /// Computes Mel-Frequency Cepstral Coefficients (MFCCs).
    ///
    /// MFCCs are a compact spectral representation widely used in speech
    /// recognition and audio classification.  They are derived by
    /// applying a DCT to log-mel filter-bank energies.
    ///
    /// # Arguments
    /// - `stft_params` — STFT configuration used for the underlying
    ///   spectrogram.
    /// - `n_mels` — number of mel filter-bank bands.
    /// - `mfcc_params` — MFCC-specific configuration (number of
    ///   coefficients, etc.). Field-level constraints are documented on
    ///   [`MfccParams`].
    ///
    /// # Returns
    /// An [`Mfcc`] containing the MFCC matrix.
    ///
    /// # Errors
    /// - [`crate::AudioSampleError::Layout`] if the signal is multi-channel.
    /// - Errors from the underlying computation.
    ///
    /// ## See Also
    /// - [MFCC — Wikipedia](https://en.wikipedia.org/wiki/Mel-frequency_cepstral_coefficients)
    fn mfcc(
        &self,
        stft_params: &StftParams,
        n_mels: NonZeroUsize,
        mfcc_params: &MfccParams,
    ) -> AudioSampleResult<Mfcc> {
        if self.is_multi_channel() {
            return Err(AudioSampleError::Layout(LayoutError::invalid_operation(
                "mfcc",
                "MFCC is only supported for mono audio samples",
            )));
        }

        let samples = self.cast_as_f64();
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
    /// - `stft_params` — STFT configuration used for the underlying
    ///   spectrogram.
    /// - `cfg` — chromagram configuration (tuning, normalization,
    ///   etc.). Field-level constraints are documented on
    ///   [`ChromaParams`].
    ///
    /// # Returns
    /// A [`Chromagram`].
    ///
    /// # Errors
    /// - [`crate::AudioSampleError::Layout`] if the signal is multi-channel.
    /// - Errors from the underlying computation.
    fn chromagram(
        &self,
        stft_params: &StftParams,
        cfg: &ChromaParams,
    ) -> AudioSampleResult<Chromagram> {
        if self.is_multi_channel() {
            return Err(AudioSampleError::Layout(LayoutError::invalid_operation(
                "chromagram",
                "Chromagram is only supported for mono audio samples",
            )));
        }

        let samples = self.cast_as_f64();
        let samples_slice = samples.as_slice().expect("Safe since we have ensured mono");
        // safety: Guaranteed non-empty since audio is mono and has samples
        let samples_slice = unsafe { NonEmptySlice::new_unchecked(samples_slice) };
        let sample_rate_f = self.sample_rate_hz();
        spectrograms::chromagram(samples_slice, stft_params, sample_rate_f, cfg).map_err(Into::into)
    }

    /// Computes a gammatone-filtered spectrogram.
    ///
    /// Gammatone filters model the bandpass response of the human
    /// cochlea.  The filter centre frequencies are spaced according to
    /// the ERB (Equivalent Rectangular Bandwidth) scale.  Prefer the
    /// typed convenience methods —
    /// [`gammatone_magnitude_spectrogram`],
    /// [`gammatone_power_spectrogram`], or
    /// [`gammatone_db_spectrogram`] — for the most common amplitude
    /// scales.
    ///
    /// # Arguments
    /// - `params` — spectrogram parameters (window, hop, FFT size).
    /// - `gammatone_params` — gammatone filter-bank configuration
    ///   (number of bands, frequency range). Field-level constraints are
    ///   documented on [`GammatoneParams`].
    /// - `db` — required when `AmpScale` is `Decibels`; ignored otherwise.
    ///
    /// # Returns
    /// A `Spectrogram<Gammatone, AmpScale>`.
    ///
    /// # Errors
    /// - [`crate::AudioSampleError::Layout`] if the signal is multi-channel.
    /// - Errors from the underlying computation.
    ///
    /// ## See Also
    /// - [Gammatone filter — Wikipedia](https://en.wikipedia.org/wiki/Gammatone)
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
            return Err(AudioSampleError::Layout(LayoutError::invalid_operation(
                "chromagram",
                "Chromagram is only supported for mono audio samples",
            )));
        }

        let samples = self.cast_as_f64();
        let samples_slice = samples.as_slice().expect("Safe since we have ensured mono");
        // safety: Guaranteed non-empty since audio is mono and has samples
        let samples_slice = unsafe { NonEmptySlice::new_unchecked(samples_slice) };
        GammatoneSpectrogram::<AmpScale>::compute(samples_slice, params, gammatone_params, db)
            .map_err(Into::into)
    }

    /// Computes the Constant-Q Transform (CQT) of a mono signal.
    ///
    /// The CQT uses a bank of bandpass filters whose centre frequencies
    /// are spaced logarithmically with a constant ratio
    /// Q = f / Δf.  This gives it the same frequency resolution as the
    /// musical scale, making it preferred for pitch and harmonic
    /// analysis.
    ///
    /// # Arguments
    /// - `params` — CQT parameters (frequency range, bins per octave,
    ///   etc.). Field-level constraints are documented on [`CqtParams`].
    /// - `hop_size` — hop length in samples between successive frames.
    ///
    /// # Returns
    /// A [`CqtResult`] containing the CQT matrix.
    ///
    /// # Errors
    /// - [`crate::AudioSampleError::Layout`] if the signal is multi-channel.
    /// - Errors from the underlying CQT computation.
    ///
    /// ## See Also
    /// - [Constant-Q transform — Wikipedia](https://en.wikipedia.org/wiki/Constant-Q_transform)
    fn constant_q_transform(
        &self,
        params: &CqtParams,
        hop_size: NonZeroUsize,
    ) -> AudioSampleResult<CqtResult> {
        if self.is_multi_channel() {
            return Err(AudioSampleError::Layout(LayoutError::invalid_operation(
                "chromagram",
                "Chromagram is only supported for mono audio samples",
            )));
        }

        let samples = self.cast_as_f64();
        let samples_slice = samples.as_slice().expect("Safe since we have ensured mono");
        // safety: Guaranteed non-empty since audio is mono and has samples
        let samples_slice = unsafe { NonEmptySlice::new_unchecked(samples_slice) };
        spectrograms::cqt(
            samples_slice,
            f64::from(self.sample_rate.get()),
            params,
            hop_size,
        )
        .map_err(Into::into)
    }

    /// Computes a CQT-based spectrogram.
    ///
    /// Applies the CQT to the signal and returns the result as a typed
    /// spectrogram.  Prefer the typed convenience methods —
    /// [`cqt_magnitude_spectrogram`], [`cqt_power_spectrogram`], or
    /// [`cqt_db_spectrogram`] — for the most common amplitude scales.
    ///
    /// # Arguments
    /// - `params` — spectrogram parameters (window, hop, FFT size).
    /// - `cqt` — CQT parameters (frequency range, bins per octave,
    ///   etc.). Field-level constraints are documented on [`CqtParams`].
    /// - `db` — required when `AmpScale` is `Decibels`; ignored otherwise.
    ///
    /// # Returns
    /// A [`CqtSpectrogram<AmpScale>`].
    ///
    /// # Errors
    /// - [`crate::AudioSampleError::Layout`] if the signal is multi-channel.
    /// - Errors from the underlying computation.
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
            return Err(AudioSampleError::Layout(LayoutError::invalid_operation(
                "cqt",
                "CqtSpectrogram is only supported for mono audio samples",
            )));
        }

        let working_samples = self.as_float();
        let working_samples_slice = working_samples
            .as_slice()
            .expect("Safe since we have ensured mono");

        // safety : working_samples_slice is non-empty since audio is mono and has samples
        let working_samples_slice = unsafe { NonEmptySlice::new_unchecked(working_samples_slice) };

        CqtSpectrogram::<AmpScale>::compute(working_samples_slice, params, cqt, db)
            .map_err(Into::into)
    }

    /// Estimates the power spectral density using Welch's method.
    ///
    /// The signal is split into overlapping segments; each is windowed
    /// with a Hanning window and FFT'd, and the resulting periodograms
    /// are averaged.  The final values are normalised to power per Hz.
    ///
    /// # Arguments
    /// - `window_size` — length of each segment in samples.  Must not
    ///   exceed the signal length.
    /// - `overlap` — fractional overlap between adjacent segments, in
    ///   the range `[0, 1)`.
    ///
    /// # Returns
    /// A pair `(frequencies, psd)` of equal length.  `frequencies[i]` is
    /// the centre frequency of bin `i` in Hz; `psd[i]` is the estimated
    /// power spectral density at that frequency.
    ///
    /// # Errors
    /// - [`crate::AudioSampleError::Parameter`] if the signal is
    ///   multi-channel, if `overlap` is outside `[0, 1)`, or if
    ///   `window_size` exceeds the signal length.
    ///
    /// ## See Also
    /// - [Welch's method — Wikipedia](https://en.wikipedia.org/wiki/Welch%27s_method)
    fn power_spectral_density(
        &self,
        window_size: NonZeroUsize,
        overlap: f64,
    ) -> AudioSampleResult<(Vec<f64>, Vec<f64>)> {
        if self.is_multi_channel() {
            return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
                "self",
                "PSD is only supported for mono audio samples",
            )));
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

        let samples = self.to_format::<f64>();
        let signal = samples
            .as_slice()
            .expect("Safe since we have ensured mono audio samples");

        let win = window_size.get();
        let hop = ((1.0 - overlap) * win as f64).floor().max(1.0) as usize;
        let sample_rate = self.sample_rate_hz();

        let mut planner = FftPlanner::new();
        let mut sum: Vec<f64> = Vec::new();
        let mut segment_count = 0u64;

        let mut start = 0;
        while start + win <= signal.len() {
            let segment = &signal[start..start + win];
            // safety: segment length == win == window_size >= 1
            let segment_slice = unsafe { NonEmptySlice::new_unchecked(segment) };
            let power =
                planner.power_spectrum(segment_slice, window_size, Some(WindowType::Hanning))?;

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

        // Average
        let count = segment_count as f64;
        for val in &mut sum {
            *val /= count;
        }

        // Convert power to power spectral density (power / Hz)
        let freq_resolution = sample_rate / win as f64;
        for val in &mut sum {
            *val /= freq_resolution;
        }

        // Build frequency axis
        let frequencies: Vec<f64> = (0..sum.len())
            .map(|i| i as f64 * sample_rate / win as f64)
            .collect();

        Ok((frequencies, sum))
    }

    /// Shorthand for [`linear_spectrogram`] with `Magnitude` amplitude scale.
    fn linear_magnitude_spectrogram(
        &self,
        params: &SpectrogramParams,
    ) -> AudioSampleResult<spectrograms::LinearMagnitudeSpectrogram> {
        self.linear_spectrogram::<spectrograms::Magnitude>(params, None)
    }

    /// Shorthand for [`linear_spectrogram`] with `Power` amplitude scale.
    fn linear_power_spectrogram(
        &self,
        params: &SpectrogramParams,
    ) -> AudioSampleResult<spectrograms::LinearPowerSpectrogram> {
        self.linear_spectrogram::<spectrograms::Power>(params, None)
    }

    /// Shorthand for [`linear_spectrogram`] with `Decibels` amplitude scale.
    fn linear_db_spectrogram(
        &self,
        params: &SpectrogramParams,
        db: &LogParams,
    ) -> AudioSampleResult<spectrograms::LinearDbSpectrogram> {
        self.linear_spectrogram::<spectrograms::Decibels>(params, Some(db))
    }

    /// Shorthand for [`log_frequency_spectrogram`] with `Power` amplitude scale.
    fn loghz_power_spectrogram(
        &self,
        params: &SpectrogramParams,
        loghz: &LogHzParams,
    ) -> AudioSampleResult<spectrograms::LogHzPowerSpectrogram> {
        self.log_frequency_spectrogram::<spectrograms::Power>(params, loghz, None)
    }

    /// Shorthand for [`log_frequency_spectrogram`] with `Magnitude` amplitude scale.
    fn loghz_magnitude_spectrogram(
        &self,
        params: &SpectrogramParams,
        loghz: &LogHzParams,
    ) -> AudioSampleResult<spectrograms::LogHzMagnitudeSpectrogram> {
        self.log_frequency_spectrogram::<spectrograms::Magnitude>(params, loghz, None)
    }

    /// Shorthand for [`log_frequency_spectrogram`] with `Decibels` amplitude scale.
    fn loghz_db_spectrogram(
        &self,
        params: &SpectrogramParams,
        loghz: &LogHzParams,
        db: &LogParams,
    ) -> AudioSampleResult<spectrograms::LogHzDbSpectrogram> {
        self.log_frequency_spectrogram::<spectrograms::Decibels>(params, loghz, Some(db))
    }

    /// Shorthand for [`mel_spectrogram`] with `Magnitude` amplitude scale.
    fn mel_mag_spectrogram(
        &self,
        params: &SpectrogramParams,
        mel: &MelParams,
    ) -> AudioSampleResult<spectrograms::MelMagnitudeSpectrogram> {
        self.mel_spectrogram(params, mel, None)
    }

    /// Shorthand for [`mel_spectrogram`] with `Decibels` amplitude scale.
    fn mel_db_spectrogram(
        &self,
        params: &SpectrogramParams,
        mel: &MelParams,
        db: &LogParams,
    ) -> AudioSampleResult<spectrograms::LogMelSpectrogram> {
        self.mel_spectrogram(params, mel, Some(db))
    }

    /// Shorthand for [`mel_spectrogram`] with `Power` amplitude scale.
    fn mel_power_spectrogram(
        &self,
        params: &SpectrogramParams,
        mel: &MelParams,
    ) -> AudioSampleResult<spectrograms::MelPowerSpectrogram> {
        self.mel_spectrogram(params, mel, None)
    }

    /// Shorthand for [`gammatone_spectrogram`] with `Magnitude` amplitude scale.
    fn gammatone_magnitude_spectrogram(
        &self,
        params: &SpectrogramParams,
        gammatone_params: &GammatoneParams,
    ) -> AudioSampleResult<spectrograms::GammatoneMagnitudeSpectrogram> {
        self.gammatone_spectrogram::<spectrograms::Magnitude>(params, gammatone_params, None)
    }

    /// Shorthand for [`gammatone_spectrogram`] with `Power` amplitude scale.
    fn gammatone_power_spectrogram(
        &self,
        params: &SpectrogramParams,
        gammatone_params: &GammatoneParams,
    ) -> AudioSampleResult<spectrograms::GammatonePowerSpectrogram> {
        self.gammatone_spectrogram::<spectrograms::Power>(params, gammatone_params, None)
    }

    /// Shorthand for [`gammatone_spectrogram`] with `Decibels` amplitude scale.
    fn gammatone_db_spectrogram(
        &self,
        params: &SpectrogramParams,
        gammatone_params: &GammatoneParams,
        db: &LogParams,
    ) -> AudioSampleResult<spectrograms::GammatoneDbSpectrogram> {
        self.gammatone_spectrogram::<spectrograms::Decibels>(params, gammatone_params, Some(db))
    }

    /// Shorthand for [`cqt_spectrogram`] with `Magnitude` amplitude scale.
    fn cqt_magnitude_spectrogram(
        &self,
        params: &SpectrogramParams,
        cqt: &CqtParams,
    ) -> AudioSampleResult<spectrograms::CqtMagnitudeSpectrogram> {
        self.cqt_spectrogram::<spectrograms::Magnitude>(params, cqt, None)
    }

    /// Shorthand for [`cqt_spectrogram`] with `Power` amplitude scale.
    fn cqt_power_spectrogram(
        &self,
        params: &SpectrogramParams,
        cqt: &CqtParams,
    ) -> AudioSampleResult<spectrograms::CqtPowerSpectrogram> {
        self.cqt_spectrogram::<spectrograms::Power>(params, cqt, None)
    }

    /// Shorthand for [`cqt_spectrogram`] with `Decibels` amplitude scale.
    fn cqt_db_spectrogram(
        &self,
        params: &SpectrogramParams,
        cqt: &CqtParams,
        db: &LogParams,
    ) -> AudioSampleResult<spectrograms::CqtDbSpectrogram> {
        self.cqt_spectrogram::<spectrograms::Decibels>(params, cqt, Some(db))
    }

    /// Decomposes a complex spectrogram into magnitude and phase.
    ///
    /// Given a complex matrix `D`, returns `(S, P)` such that
    /// `D = S * P` elementwise, where `S` contains magnitudes raised to
    /// `power` and `P` contains unit-magnitude complex phase factors.
    /// Bins where the magnitude is zero are assigned a phase of `1 + 0i`.
    ///
    /// # Arguments
    /// - `complex_spect` — the complex STFT or FFT matrix.
    /// - `power` — exponent applied to the magnitude values.  `None`
    ///   defaults to 1 (raw magnitude).
    ///
    /// # Returns
    /// `(magnitude, phase)` — the magnitude matrix (real-valued) and
    /// the phase matrix (complex unit-magnitude).
    fn magphase(
        complex_spect: &Array2<Complex<f64>>,
        power: Option<NonZeroUsize>,
    ) -> (Array2<f64>, Array2<Complex<f64>>) {
        // Magnitude: elementwise absolute value

        let mut mag = complex_spect.mapv(num_complex::Complex::norm);

        // zeros_to_ones: 1.0 where mag == 0, else 0.0
        let zeros_to_ones = mag.mapv(|x| if x == 0.0 { 1.0 } else { 0.0 });

        // mag_nonzero = mag + zeros_to_ones
        let mag_nonzero = &mag + &zeros_to_ones;

        // Compute phase = D / mag_nonzero, but handle zeros separately
        let mut phase = complex_spect.clone();

        let power = match power {
            Some(p) => p.get() as f64,
            None => 1.0,
        };

        // Perform elementwise division for real and imaginary parts
        ndarray::Zip::from(&mut phase)
            .and(&mag_nonzero)
            .and(&zeros_to_ones)
            .for_each(|p, &m_nz, &z| {
                let div = Complex {
                    re: p.re / m_nz + z, // add 1.0 if originally zero
                    im: p.im / m_nz,
                };
                *p = div;
            });

        // Raise magnitude to the given power
        mag.mapv_inplace(|x| x.powf(power));

        (mag, phase)
    }
}
