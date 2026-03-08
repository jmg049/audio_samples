//! Pitch detection and fundamental frequency analysis.
//!
//! This module implements algorithms for detecting the fundamental frequency
//! (pitch) of audio signals, tracking pitch over time, and analysing harmonic
//! structure. It also provides musical key estimation via chromagram analysis.
//!
//! No single algorithm is optimal for all audio types. This module exposes the
//! YIN algorithm (accurate for voiced speech and musical instruments) and an
//! autocorrelation method (simpler and faster, effective for clean tones), so
//! callers can choose the right tool for their signal.
//!
//! Use [`AudioPitchAnalysis::detect_pitch_yin`] for robust single-frame pitch
//! detection or [`AudioPitchAnalysis::track_pitch`] for a time-varying pitch
//! contour. Harmonic analysis and key estimation use the `spectrograms` crate
//! internally for FFT computation. All methods operate on mono signals only.
//!
//! # Example
//!
//! ```
//! use audio_samples::operations::traits::AudioPitchAnalysis;
//! use audio_samples::{sample_rate, sine_wave};
//! use std::time::Duration;
//!
//! // Generate a 440 Hz sine wave at 44100 Hz for 0.1 s.
//! let hz = 440.0f64;
//! let audio = sine_wave::<f64>(hz, Duration::from_millis(100), sample_rate!(44100), 1.0);
//!
//! let pitch = audio.detect_pitch_yin(0.1, 80.0, 1000.0).unwrap();
//! assert!(pitch.is_some());
//! let detected_hz = pitch.unwrap();
//! assert!((detected_hz - hz).abs() < 10.0, "detected {detected_hz:.1} Hz, expected {hz} Hz");
//! ```

use std::num::NonZeroUsize;

use crate::operations::traits::AudioPitchAnalysis;
use crate::operations::types::PitchDetectionMethod;
use crate::{
    AudioSampleError, AudioSampleResult, AudioSamples, AudioTypeConversion, ParameterError,
    StandardSample,
};

use non_empty_slice::NonEmptySlice;
use spectrograms::{ChromaParams, SpectrogramPlanner, StftParams, WindowType};

impl<T> AudioPitchAnalysis for AudioSamples<'_, T>
where
    T: StandardSample,
    Self: AudioTypeConversion<Sample = T>,
{
    /// Detects the fundamental frequency using the YIN pitch detection algorithm.
    ///
    /// YIN computes a cumulative mean normalised difference function (CMND) and
    /// finds the first lag below `threshold`, which corresponds to the fundamental
    /// period. Lower thresholds are stricter and reduce false detections; values
    /// in `[0.1, 0.2]` are typical for musical audio.
    ///
    /// # Arguments
    ///
    /// - `threshold` – Confidence threshold in `[0.0, 1.0]`.
    /// - `min_frequency` – Minimum detectable frequency in Hz (> 0).
    /// - `max_frequency` – Maximum detectable frequency in Hz (> `min_frequency`).
    ///
    /// # Returns
    ///
    /// - `Some(frequency_hz)` – Estimated fundamental frequency.
    /// - `None` – Signal is too short, silent, or no pitch was detected.
    ///
    /// # Errors
    ///
    /// Returns [crate::AudioSampleError::Unsupported] for multi-channel audio.
    /// Returns [crate::AudioSampleError::Parameter] if `threshold ∉ [0.0, 1.0]`,
    /// `min_frequency ≤ 0.0`, or `max_frequency ≤ min_frequency`.
    ///
    /// # Example
    ///
    /// ```
    /// use audio_samples::operations::traits::AudioPitchAnalysis;
    /// use audio_samples::{sample_rate, sine_wave};
    /// use std::time::Duration;
    ///
    /// let hz = 440.0f64;
    /// let audio = sine_wave::<f64>(hz, Duration::from_millis(100), sample_rate!(44100), 1.0);
    ///
    /// let pitch = audio.detect_pitch_yin(0.1, 80.0, 1000.0).unwrap();
    /// assert!(pitch.is_some());
    /// assert!((pitch.unwrap() - hz).abs() < 10.0);
    /// ```
    #[inline]
    fn detect_pitch_yin(
        &self,
        threshold: f64,
        min_frequency: f64,
        max_frequency: f64,
    ) -> AudioSampleResult<Option<f64>> {
        if self.is_multi_channel() {
            return Err(AudioSampleError::unsupported(
                "YIN pitch detection is only supported for mono audio samples",
            ));
        }

        if !(0.0..=1.0).contains(&threshold) {
            return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
                "threshold",
                "Threshold must be between 0.0 and 1.0",
            )));
        }

        if min_frequency <= 0.0 || max_frequency <= min_frequency {
            return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
                "frequency_range",
                "Invalid frequency range",
            )));
        }

        let samples = self.to_format::<f64>();
        let samples_slice = samples
            .as_slice()
            .expect("Same to unwrap since mono samples always return a contiguous slice of memory");
        let sample_rate_f = self.sample_rate_hz();

        // Calculate tau (lag) limits based on frequency constraints
        let min_tau = (sample_rate_f / max_frequency) as usize;
        let max_tau = (sample_rate_f / min_frequency) as usize;
        if max_tau >= samples_slice.len() / 2 {
            return Ok(None);
        }
        // Safety: samples is guaranteed non-empty by construction
        let samples = non_empty_slice::non_empty_slice!(samples_slice);
        let result = yin_pitch_detection(samples, min_tau, max_tau, threshold);

        Ok(result.map(|tau| sample_rate_f / tau))
    }

    /// Detects the fundamental frequency using autocorrelation.
    ///
    /// Finds the lag with maximum autocorrelation within the range implied by
    /// `[min_frequency, max_frequency]` and converts it to a frequency. This
    /// method is fast and effective for clean, periodic signals but less robust
    /// than YIN on noisy or voiced speech.
    ///
    /// # Arguments
    ///
    /// - `min_frequency` – Minimum detectable frequency in Hz (> 1.0).
    /// - `max_frequency` – Maximum detectable frequency in Hz (> `min_frequency`).
    ///
    /// # Returns
    ///
    /// - `Some(frequency_hz)` – Estimated fundamental frequency.
    /// - `None` – Signal is too short, silent, or unpitched.
    ///
    /// # Errors
    ///
    /// Returns [crate::AudioSampleError::Unsupported] for multi-channel audio.
    /// Returns [crate::AudioSampleError::Parameter] if `min_frequency ≤ 1.0` or
    /// `max_frequency ≤ min_frequency`.
    ///
    /// # Example
    ///
    /// ```
    /// use audio_samples::operations::traits::AudioPitchAnalysis;
    /// use audio_samples::{sample_rate, sine_wave};
    /// use std::time::Duration;
    ///
    /// let hz = 220.0f64;
    /// let audio = sine_wave::<f32>(hz, Duration::from_millis(100), sample_rate!(44100), 1.0);
    ///
    /// let pitch = audio.detect_pitch_autocorr(80.0, 1000.0).unwrap();
    /// assert!(pitch.is_some());
    /// assert!((pitch.unwrap() - hz).abs() < 15.0);
    /// ```
    #[inline]
    fn detect_pitch_autocorr(
        &self,
        min_frequency: f64,
        max_frequency: f64,
    ) -> AudioSampleResult<Option<f64>> {
        if self.is_multi_channel() {
            return Err(AudioSampleError::unsupported(
                "Autocorrelation pitch detection is only supported for mono audio samples",
            ));
        }

        if min_frequency <= 1.0 || max_frequency <= min_frequency {
            return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
                "frequency_range",
                format!("Invalid frequency range {min_frequency} - {max_frequency}"),
            )));
        }

        let samples = self.to_format::<f64>();
        let samples_slice = samples
            .as_slice()
            .expect("Same to unwrap since mono samples always return a contiguous slice of memory");

        let sample_rate = self.sample_rate_hz();

        // Calculate tau (lag) limits based on frequency constraints
        let min_tau = (sample_rate / max_frequency) as usize;
        let max_tau = (sample_rate / min_frequency) as usize;

        if max_tau >= samples.len().get() / 2 {
            return Ok(None);
        }
        // safety: samples is guaranteed non-empty by construction
        let samples = unsafe { NonEmptySlice::new_unchecked(samples_slice) };

        let result = autocorr_pitch_detection(samples, min_tau, max_tau);

        Ok(result.map(|tau| sample_rate / tau))
    }

    /// Tracks pitch over time by applying pitch detection to successive windows.
    ///
    /// The signal is split into overlapping frames of `window_size` samples,
    /// advancing by `hop_size` each step. Each frame is analysed independently
    /// using `method`. Frames shorter than `window_size / 2` at the signal end
    /// are discarded. Only [`PitchDetectionMethod::Yin`] and
    /// [`PitchDetectionMethod::Autocorrelation`] are implemented; other variants
    /// log a warning and return `None` for that frame.
    ///
    /// # Arguments
    ///
    /// - `window_size` – Analysis window length in samples; must be ≤ signal length.
    /// - `hop_size` – Step between successive windows in samples; must be < `window_size`.
    /// - `method` – Pitch detection algorithm to use per frame.
    /// - `threshold` – YIN confidence threshold; ignored for autocorrelation.
    /// - `min_frequency` – Minimum detectable frequency in Hz.
    /// - `max_frequency` – Maximum detectable frequency in Hz.
    ///
    /// # Returns
    ///
    /// A `Vec<(f64, Option<f64>)>` of `(time_seconds, frequency_hz)` pairs in
    /// time order. `frequency_hz` is `None` when no pitch was found in that frame.
    ///
    /// # Errors
    ///
    /// Returns [crate::AudioSampleError::Unsupported] for multi-channel audio.
    /// Returns [crate::AudioSampleError::Parameter] if `window_size > signal length`
    /// or `hop_size >= window_size`.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use std::num::NonZeroUsize;
    /// use audio_samples::operations::traits::AudioPitchAnalysis;
    /// use audio_samples::operations::types::PitchDetectionMethod;
    /// use audio_samples::{sample_rate, sine_wave};
    /// use std::time::Duration;
    ///
    /// let hz = 440.0f64;
    /// let audio = sine_wave::<f32>(hz, Duration::from_millis(500), sample_rate!(44100), 1.0);
    ///
    /// let track = audio.track_pitch(
    ///     NonZeroUsize::new(2048).unwrap(),
    ///     NonZeroUsize::new(512).unwrap(),
    ///     PitchDetectionMethod::Yin,
    ///     0.1,
    ///     80.0,
    ///     1000.0,
    /// ).unwrap();
    ///
    /// assert!(!track.is_empty());
    /// let detected: Vec<f64> = track.iter().filter_map(|&(_, f)| f).collect();
    /// let avg = detected.iter().sum::<f64>() / detected.len() as f64;
    /// assert!((avg - hz).abs() < 20.0);
    /// ```
    #[inline]
    fn track_pitch(
        &self,
        window_size: NonZeroUsize,
        hop_size: NonZeroUsize,
        method: PitchDetectionMethod,
        threshold: f64,
        min_frequency: f64,
        max_frequency: f64,
    ) -> AudioSampleResult<Vec<(f64, Option<f64>)>> {
        if self.is_multi_channel() {
            return Err(AudioSampleError::unsupported(
                "Pitch tracking is only supported for mono audio samples",
            ));
        }

        if window_size.get() <= hop_size.get() {
            return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
                "window_hop_size",
                "Window size must be greater than or equal to hop size",
            )));
        }

        if self.len() < window_size {
            return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
                "window_size",
                "Window size cannot be larger than the total number of samples",
            )));
        }

        let window_size = window_size.get();
        let hop_size = hop_size.get();
        let samples = self.as_f64();
        let samples_slice = samples
            .as_slice()
            .expect("Same to unwrap since mono samples always return a contiguous slice of memory");

        let sample_rate = self.sample_rate_hz();
        let total_results = (0..samples_slice.len()).step_by(hop_size).count();
        let mut results = Vec::with_capacity(total_results);
        // Process each windowed segment
        // Guaranteed at least one iteration since samples is non-empty and window_size <= samples.len()
        for start in (0..samples_slice.len()).step_by(hop_size) {
            let end = (start + window_size).min(samples_slice.len());
            if end - start < window_size / 2 {
                break; // Skip windows that are too short
            }

            let window = &samples_slice[start..end];
            // safety: window is guaranteed non-empty since end - start >= window_size / 2 > 0
            let window = unsafe { NonEmptySlice::new_unchecked(window) };
            let time_seconds = start as f64 / sample_rate;

            // Create temporary AudioSamples for this window
            let window_data = window.to_non_empty_vec();
            let window_audio =
                AudioSamples::from_mono_vec(window_data, self.sample_rate).into_owned();
            let frequency: Option<f64> = match method {
                PitchDetectionMethod::Yin => {
                    window_audio.detect_pitch_yin(threshold, min_frequency, max_frequency)?
                }
                PitchDetectionMethod::Autocorrelation => {
                    window_audio.detect_pitch_autocorr(min_frequency, max_frequency)?
                }
                _ => {
                    eprintln!("Pitch detection method {method:?} not implemented yet");
                    None
                } // Other methods not implemented yet
            };

            results.push((time_seconds, frequency));
        }

        Ok(results)
    }

    /// Computes the harmonic-to-noise ratio (HNR) in decibels.
    ///
    /// HNR measures how much of the signal's energy comes from periodic
    /// (harmonic) components versus aperiodic (noise) components. A high HNR
    /// indicates a clean, voiced tone; a low or negative HNR indicates
    /// noise-dominated content. Power spectrum computation is delegated to the
    /// `spectrograms` crate. Harmonic bins are those within one
    /// frequency-resolution unit of a multiple of `fundamental_freq`.
    ///
    /// # Arguments
    ///
    /// - `fundamental_freq` – Known fundamental frequency in Hz (> 0).
    /// - `num_harmonics` – Number of harmonics to accumulate into harmonic power.
    /// - `n_fft` – FFT size. Defaults to the signal length when `None`.
    /// - `window_type` – Window function applied before FFT. Defaults to Hanning when `None`.
    ///
    /// # Returns
    ///
    /// HNR in dB. Returns `f64::INFINITY` when noise power is zero (purely
    /// harmonic signal).
    ///
    /// # Errors
    ///
    /// Returns [crate::AudioSampleError::Unsupported] for multi-channel audio.
    /// Returns [crate::AudioSampleError::Parameter] if `fundamental_freq ≤ 0.0`.
    ///
    /// # Example
    ///
    /// ```
    /// use std::num::NonZeroUsize;
    /// use audio_samples::operations::traits::AudioPitchAnalysis;
    /// use audio_samples::{sample_rate, sine_wave};
    /// use std::time::Duration;
    ///
    /// // A pure sine wave is maximally harmonic.
    /// let hz = 440.0f64;
    /// let audio = sine_wave::<f32>(hz, Duration::from_millis(100), sample_rate!(44100), 1.0);
    ///
    /// let hnr = audio
    ///     .harmonic_to_noise_ratio(hz, NonZeroUsize::new(5).unwrap(), None, None)
    ///     .unwrap();
    /// assert!(hnr > 0.0, "pure sine should have positive HNR, got {hnr:.1} dB");
    /// ```
    #[inline]
    fn harmonic_to_noise_ratio(
        &self,
        fundamental_freq: f64,
        num_harmonics: NonZeroUsize,
        n_fft: Option<NonZeroUsize>,
        window_type: Option<WindowType>,
    ) -> AudioSampleResult<f64> {
        if self.is_multi_channel() {
            return Err(AudioSampleError::Unsupported(
                "Harmonic-to-noise ratio analysis is only supported for mono audio samples"
                    .to_string(),
            ));
        }

        if fundamental_freq <= 0.0 {
            return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
                "harmonics_config",
                "Invalid fundamental frequency",
            )));
        }
        let sample_rate = self.sample_rate_hz();

        let samples = self.as_f64();
        let samples_slice = samples
            .as_slice()
            .expect("Same to unwrap since mono samples always return a contiguous slice of memory");
        // safety: samples is guaranteed non-empty by construction
        let samples_slice = unsafe { NonEmptySlice::new_unchecked(samples_slice) };
        let planner = SpectrogramPlanner::new();
        let n_fft = n_fft.map_or(samples.len(), |n| n);

        // Compute power spectrum
        let spectrum = planner.compute_power_spectrum(
            samples_slice,
            n_fft,
            window_type.unwrap_or_default(),
        )?;
        // frequency resolution should be retrievable from ``spectrograms`` crate
        let freq_resolution = sample_rate / n_fft.get() as f64;

        // Calculate harmonic and noise power
        let mut harmonic_power = 0.0;
        let mut total_power = 0.0;

        for (i, &power) in spectrum.iter().enumerate() {
            let freq = i as f64 * freq_resolution;
            total_power += power;

            // Check if this frequency is close to any harmonic
            for harmonic in 1..=num_harmonics.get() {
                let harmonic_freq = fundamental_freq * harmonic as f64;
                if (freq - harmonic_freq).abs() < freq_resolution {
                    harmonic_power += power;
                    break;
                }
            }
        }

        let noise_power = total_power - harmonic_power;
        if noise_power <= 0.0 {
            return Ok(f64::INFINITY);
        }

        Ok(10.0 * (harmonic_power / noise_power).log10())
    }

    /// Analyses the harmonic content relative to a known fundamental frequency.
    ///
    /// Computes the power spectrum of the signal and extracts the peak power
    /// within a `tolerance`-relative frequency band around each harmonic of
    /// `fundamental_freq`. All magnitudes are normalised so that the fundamental
    /// (index 0) equals 1.0; higher indices contain the relative magnitudes of
    /// the 2nd, 3rd, … harmonics.
    ///
    /// # Arguments
    ///
    /// - `fundamental_freq` – Fundamental frequency in Hz (> 0).
    /// - `num_harmonics` – Number of harmonics to extract, including the fundamental.
    /// - `tolerance` – Fractional bandwidth around each harmonic to search, in `[0.0, 1.0]`.
    /// - `n_fft` – FFT size. Defaults to the signal length when `None`.
    /// - `window_type` – Window function applied before FFT. Defaults to Hanning when `None`.
    ///
    /// # Returns
    ///
    /// A `Vec<f64>` of length `num_harmonics`. Index 0 is always 1.0 after
    /// normalisation (unless the fundamental has zero power, in which case all
    /// values are the raw magnitudes).
    ///
    /// # Errors
    ///
    /// Returns [crate::AudioSampleError::Unsupported] for multi-channel audio.
    /// Returns [crate::AudioSampleError::Parameter] if `fundamental_freq ≤ 0.0` or
    /// `tolerance ∉ [0.0, 1.0]`.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use std::num::NonZeroUsize;
    /// use audio_samples::operations::traits::AudioPitchAnalysis;
    /// use audio_samples::{sample_rate, sawtooth_wave};
    /// use std::time::Duration;
    ///
    /// // Sawtooth wave at 220 Hz — rich in harmonics.
    /// let hz = 220.0f64;
    /// let audio = sawtooth_wave::<f32>(hz, Duration::from_millis(500), sample_rate!(44100), 1.0);
    ///
    /// let harmonics = audio
    ///     .harmonic_analysis(hz, NonZeroUsize::new(5).unwrap(), 0.1, None, None)
    ///     .unwrap();
    /// assert_eq!(harmonics.len(), 5);
    /// assert!((harmonics[0] - 1.0).abs() < 0.1, "fundamental should be normalised to 1.0");
    /// ```
    #[inline]
    fn harmonic_analysis(
        &self,
        fundamental_freq: f64,
        num_harmonics: NonZeroUsize,
        tolerance: f64,
        n_fft: Option<NonZeroUsize>,
        window_type: Option<WindowType>,
    ) -> AudioSampleResult<Vec<f64>> {
        if self.is_multi_channel() {
            return Err(AudioSampleError::Unsupported(
                "Harmonic analysis is only supported for mono audio samples".to_string(),
            ));
        }

        if fundamental_freq <= 0.0 {
            return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
                "harmonics_config",
                "Invalid fundamental frequency",
            )));
        }

        if !(0.0..=1.0).contains(&tolerance) {
            return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
                "tolerance",
                "Tolerance must be between 0.0 and 1.0",
            )));
        }

        let samples = self.as_f64();
        let samples_slice = samples
            .as_slice()
            .expect("Same to unwrap since mono samples always return a contiguous slice of memory");

        // safety: samples is guaranteed non-empty by construction
        let samples_slice = unsafe { NonEmptySlice::new_unchecked(samples_slice) };

        let sample_rate = self.sample_rate_hz();
        let planner = SpectrogramPlanner::new();
        let n_fft = n_fft.map_or(samples.len(), |n| n);
        // Compute power spectrum
        let spectrum = planner.compute_power_spectrum(
            samples_slice,
            n_fft,
            window_type.unwrap_or_default(),
        )?;
        // frequency resolution should be retrievable from ``spectrograms`` crate
        let freq_resolution = sample_rate / n_fft.get() as f64;

        let mut harmonic_magnitudes = vec![0.0; num_harmonics.get()];

        // Find magnitude at each harmonic frequency
        for harmonic in 1..=num_harmonics.get() {
            let harmonic_freq = fundamental_freq * harmonic as f64;
            let target_bin = (harmonic_freq / freq_resolution).round() as usize;
            // Search within tolerance range
            let tolerance_bins = (tolerance * harmonic_freq / freq_resolution) as usize;
            let start_bin = target_bin.saturating_sub(tolerance_bins);
            let end_bin = (target_bin + tolerance_bins).min(spectrum.len().get() - 1);

            // Find maximum magnitude in the tolerance range
            let max_magnitude = spectrum[start_bin..=end_bin]
                .iter()
                .fold(0.0, |acc: f64, &x| acc.max(x));

            harmonic_magnitudes[harmonic - 1] = max_magnitude;
        }

        // Normalize relative to fundamental
        if harmonic_magnitudes[0] > 0.0 {
            let fundamental_magnitude = harmonic_magnitudes[0];
            for magnitude in &mut harmonic_magnitudes {
                *magnitude /= fundamental_magnitude;
            }
        }

        Ok(harmonic_magnitudes)
    }

    /// Estimates the musical key of the audio using chromagram analysis.
    ///
    /// Computes a chromagram via the `spectrograms` crate and accumulates chroma
    /// energy across all time frames. The average chroma vector is compared
    /// against Krumhansl-Schmuckler major and minor key profiles via Pearson
    /// correlation to identify the best-matching key.
    ///
    /// # Arguments
    ///
    /// - `stft_params` – STFT parameters controlling frame size and hop for
    ///   chromagram computation.
    ///
    /// # Returns
    ///
    /// A `(key_index, confidence)` tuple where:
    /// - `key_index` is in `0..=11` for major keys (C=0, C♯=1, …, B=11) and
    ///   `12..=23` for minor keys (Cm=12, C♯m=13, …, Bm=23).
    /// - `confidence` is in `[0.0, 1.0]`; higher values indicate a stronger match.
    ///
    /// # Errors
    ///
    /// Propagates any error returned by the `spectrograms` crate during STFT
    /// or chromagram computation.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use std::num::NonZeroUsize;
    /// use audio_samples::operations::traits::AudioPitchAnalysis;
    /// use audio_samples::{sample_rate, sine_wave};
    /// use spectrograms::{StftParams, WindowType};
    /// use std::time::Duration;
    ///
    /// # let audio = sine_wave::<f32>(440.0, Duration::from_secs(1), sample_rate!(44100), 1.0);
    /// let params = StftParams::new(
    ///     NonZeroUsize::new(2048).unwrap(),
    ///     NonZeroUsize::new(512).unwrap(),
    ///     WindowType::Hanning,
    ///     true,
    /// ).unwrap();
    /// let (key, confidence) = audio.estimate_key(&params).unwrap();
    /// assert!(key < 24);
    /// assert!((0.0..=1.0).contains(&confidence));
    /// ```
    #[inline]
    fn estimate_key(&self, stft_params: &StftParams) -> AudioSampleResult<(usize, f64)> {
        let samples = self.to_format::<f64>();

        let samples_slice = samples
            .as_slice()
            .expect("Same to unwrap since mono samples always return a contiguous slice of memory");
        // safety: samples is guaranteed non-empty by construction
        let samples_slice = unsafe { NonEmptySlice::new_unchecked(samples_slice) };
        let sample_rate_f = self.sample_rate_hz();
        // Krumhansl-Schmuckler key profiles
        let major_profile: [f64; 12] = [
            6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88,
        ];
        let minor_profile: [f64; 12] = [
            6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17,
        ];
        let chroma_params = ChromaParams::music_standard();
        let chromagram =
            spectrograms::chromagram(samples_slice, stft_params, sample_rate_f, &chroma_params)?;

        // Accumulate chroma features across all time frames
        let mut chroma_sum = vec![0.0; 12];
        for frame_idx in 0..chromagram.n_frames().get() {
            for (pitch_class, value) in chroma_sum.iter_mut().enumerate().take(12) {
                *value += chromagram.data[[pitch_class, frame_idx]];
            }
        }

        // Average chroma features
        let num_frames = chromagram.n_frames().get() as f64;
        for value in &mut chroma_sum {
            *value /= num_frames;
        }

        // Normalize chroma vector
        let chroma_sum_total: f64 = chroma_sum.iter().sum();
        if chroma_sum_total > 0.0 {
            for value in &mut chroma_sum {
                *value /= chroma_sum_total;
            }
        }

        // Calculate correlation with all 24 keys
        let mut best_correlation = -1.0;
        let mut best_key = 0;
        let mut is_major = true;

        // Test all major keys
        for tonic in 0..12 {
            let correlation = calculate_correlation(&chroma_sum, &major_profile, tonic);
            if correlation > best_correlation {
                best_correlation = correlation;
                best_key = tonic;
                is_major = true;
            }
        }

        // Test all minor keys
        for tonic in 0..12 {
            let correlation = calculate_correlation(&chroma_sum, &minor_profile, tonic);
            if correlation > best_correlation {
                best_correlation = correlation;
                best_key = tonic;
                is_major = false;
            }
        }

        // Encode key: 0-11 for major keys, 12-23 for minor keys
        let encoded_key = if is_major { best_key } else { best_key + 12 };

        // Convert correlation to confidence (normalize to 0-1 range)
        let confidence = f64::midpoint(best_correlation, 1.0);

        Ok((encoded_key, confidence))
    }
}

/// Computes Pearson correlation between a chroma vector and a rotated key profile.
///
/// `tonic` rotates `profile` so that position 0 corresponds to the tonic pitch class.
fn calculate_correlation(chroma: &[f64], profile: &[f64], tonic: usize) -> f64 {
    debug_assert_eq!(chroma.len(), 12);
    debug_assert_eq!(profile.len(), 12);

    // Rotate profile to match tonic
    let mut rotated_profile = [0.0; 12];
    for i in 0..12 {
        rotated_profile[i] = profile[(i + tonic) % 12];
    }

    // Calculate Pearson correlation coefficient
    let chroma_mean: f64 = chroma.iter().sum::<f64>() / 12.0;
    let profile_mean: f64 = rotated_profile.iter().sum::<f64>() / 12.0;

    let mut numerator = 0.0;
    let mut chroma_variance = 0.0;
    let mut profile_variance = 0.0;

    for i in 0..12 {
        let chroma_dev = chroma[i] - chroma_mean;
        let profile_dev = rotated_profile[i] - profile_mean;

        numerator += chroma_dev * profile_dev;
        chroma_variance += chroma_dev * chroma_dev;
        profile_variance += profile_dev * profile_dev;
    }

    let denominator = (chroma_variance * profile_variance).sqrt();
    if denominator > 0.0 {
        numerator / denominator
    } else {
        0.0
    }
}

/// Low-level YIN pitch detection that returns the fundamental period in samples.
///
/// Computes the difference function `d(τ) = Σ(x_j − x_{j+τ})²` and its
/// cumulative mean normalised form (CMND). Returns the first lag τ in
/// `[min_tau, max_tau]` whose CMND value falls below `threshold`, refined
/// by a local neighbourhood search. The caller converts τ to Hz with
/// `sample_rate / τ`.
///
/// This is the low-level primitive used by
/// [`AudioPitchAnalysis::detect_pitch_yin`].
///
/// # Arguments
///
/// - `samples` – Non-empty slice of audio samples.
/// - `min_tau` – Smallest lag to search (inclusive), in samples. Set to
///   `(sample_rate / max_frequency) as usize`.
/// - `max_tau` – Largest lag to search (inclusive), in samples. Set to
///   `(sample_rate / min_frequency) as usize`.
/// - `threshold` – Maximum CMND value accepted as a valid pitch. Typical
///   values are in `[0.1, 0.2]`.
///
/// # Returns
///
/// `Some(tau)` — period in samples; convert to Hz with `sample_rate / tau`.
/// `None` — no CMND value below `threshold`, or `max_tau ≥ samples.len() / 2`.
///
/// # Example
///
/// ```
/// use audio_samples::operations::pitch_analysis::yin_pitch_detection;
/// use audio_samples::{sample_rate, sine_wave};
/// use non_empty_slice::NonEmptySlice;
/// use std::time::Duration;
///
/// let sr = 44100.0f64;
/// let hz = 440.0f64;
/// let audio = sine_wave::<f64>(hz, Duration::from_millis(100), sample_rate!(44100), 1.0);
/// let data = audio.as_slice().unwrap();
/// let samples = NonEmptySlice::from_slice(data).unwrap();
///
/// let min_tau = (sr / 1000.0) as usize; // 44 samples → 1000 Hz upper bound
/// let max_tau = (sr / 80.0) as usize;   // 551 samples → 80 Hz lower bound
///
/// let tau = yin_pitch_detection(samples, min_tau, max_tau, 0.1);
/// assert!(tau.is_some());
/// if let Some(t) = tau {
///     let detected_hz = sr / t;
///     assert!((detected_hz - hz).abs() < 10.0);
/// }
/// ```
#[inline]
#[must_use]
pub fn yin_pitch_detection(
    samples: &NonEmptySlice<f64>,
    min_tau: usize,
    max_tau: usize,
    threshold: f64,
) -> Option<f64> {
    let n = samples.len().get();
    if max_tau >= n / 2 {
        return None;
    }

    // Step 1: Compute difference function
    let mut diff_fn = vec![0.0; max_tau + 1];
    for tau in min_tau..=max_tau {
        let mut sum = 0.0;
        for j in 0..(n - tau) {
            let delta = samples[j] - samples[j + tau];
            sum += delta * delta;
        }
        diff_fn[tau] = sum;
    }

    // Step 2: Compute cumulative mean normalized difference
    let mut cmnd = vec![1.0; max_tau + 1];
    let mut running_sum = 0.0;

    for tau in 1..=max_tau {
        running_sum += diff_fn[tau];
        if running_sum > 0.0 {
            cmnd[tau] = diff_fn[tau] / (running_sum / tau as f64);
        }
    }

    // Step 3: Find the first minimum below threshold
    for tau in min_tau..=max_tau {
        if cmnd[tau] < threshold {
            // Find the actual minimum around this tau
            let mut min_tau = tau;
            let mut min_val = cmnd[tau];

            // Search in a small neighborhood
            let start = tau.saturating_sub(5).max(min_tau);
            let end = (tau + 5).min(max_tau);

            for (t, &val) in cmnd.iter().enumerate().skip(start).take(end - start + 1) {
                if val < min_val {
                    min_val = val;
                    min_tau = t;
                }
            }

            return Some(min_tau as f64);
        }
    }

    None
}

/// Low-level autocorrelation pitch detection that returns the fundamental period in samples.
///
/// Finds the lag in `[min_tau, max_tau]` with maximum autocorrelation and
/// returns it as the estimated period. The caller converts to frequency with
/// `sample_rate / tau`. Unlike YIN, this does not normalise the correlation,
/// making it fast but more susceptible to noise.
///
/// This is the low-level primitive used by
/// [`AudioPitchAnalysis::detect_pitch_autocorr`].
///
/// # Arguments
///
/// - `samples` – Slice of audio samples.
/// - `min_tau` – Smallest lag to search (inclusive), in samples. Set to
///   `(sample_rate / max_frequency) as usize`.
/// - `max_tau` – Largest lag to search (inclusive), in samples. Set to
///   `(sample_rate / min_frequency) as usize`.
///
/// # Returns
///
/// `Some(tau)` — lag in samples with highest autocorrelation; convert to Hz
/// with `sample_rate / tau`.
/// `None` — no positive autocorrelation found, or `max_tau ≥ samples.len() / 2`.
///
/// # Example
///
/// ```
/// use audio_samples::operations::pitch_analysis::autocorr_pitch_detection;
/// use audio_samples::{sample_rate, sine_wave};
/// use std::time::Duration;
///
/// let sr = 44100.0f64;
/// let hz = 220.0f64;
/// let audio = sine_wave::<f64>(hz, Duration::from_millis(100), sample_rate!(44100), 1.0);
/// let data = audio.as_slice().unwrap();
///
/// let min_tau = (sr / 1000.0) as usize;
/// let max_tau = (sr / 80.0) as usize;
///
/// let tau = autocorr_pitch_detection(data, min_tau, max_tau);
/// assert!(tau.is_some());
/// if let Some(t) = tau {
///     let detected_hz = sr / t;
///     assert!((detected_hz - hz).abs() < 15.0);
/// }
/// ```
#[inline]
#[must_use]
pub fn autocorr_pitch_detection(samples: &[f64], min_tau: usize, max_tau: usize) -> Option<f64> {
    let n = samples.len();
    if max_tau >= n / 2 {
        return None;
    }

    let mut max_corr = 0.0;
    let mut best_tau = 0;

    for tau in min_tau..=max_tau {
        let mut corr = 0.0;
        for i in 0..(n - tau) {
            corr += samples[i] * samples[i + tau];
        }

        if corr > max_corr {
            max_corr = corr;
            best_tau = tau;
        }
    }

    if best_tau > 0 {
        Some(best_tau as f64)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::operations::traits::AudioPitchAnalysis;
    use crate::sample_rate;
    use ndarray::Array1;
    use non_empty_slice::{NonEmptyVec, non_empty_vec};
    use std::f64::consts::PI;

    #[test]
    fn test_pitch_detection_yin() {
        // Create a simple sine wave at 440 Hz
        let sample_rate = 44100;
        let duration = 1.0; // 1 second
        let frequency = 440.0; // A4
        let samples_count = (sample_rate as f64 * duration) as usize;

        let mut samples = Vec::new();
        for i in 0..samples_count {
            let t = i as f64 / sample_rate as f64;
            let value = (2.0 * PI * frequency * t).sin();
            samples.push(value as f64);
        }
        let samples = NonEmptyVec::new(samples).unwrap();
        let audio: AudioSamples<'_, f64> =
            AudioSamples::from_mono_vec(samples, sample_rate!(44100));

        // Test YIN pitch detection
        let detected_pitch = audio.detect_pitch_yin(0.1, 80.0, 1000.0).unwrap();

        // Should detect approximately 440 Hz
        assert!(detected_pitch.is_some());
        let pitch: f64 = detected_pitch.unwrap();
        assert!((pitch - 440.0).abs() < 10.0); // Allow 10 Hz tolerance
    }

    #[test]
    fn test_pitch_detection_autocorr() {
        // Create a simple sine wave at 220 Hz
        let sample_rate = 44100;
        let duration = 1.0; // 1 second
        let frequency = 220.0; // A3
        let samples_count = (sample_rate as f64 * duration) as usize;

        let mut samples = Vec::new();
        for i in 0..samples_count {
            let t = i as f64 / sample_rate as f64;
            let value = (2.0 * PI * frequency * t).sin();
            samples.push(value as f32);
        }
        let samples = NonEmptyVec::new(samples).unwrap();
        let audio: AudioSamples<'_, f32> =
            AudioSamples::from_mono_vec(samples, sample_rate!(44100));

        // Test autocorrelation pitch detection
        let detected_pitch = audio.detect_pitch_autocorr(80.0, 1000.0).unwrap();

        // Should detect approximately 220 Hz
        assert!(detected_pitch.is_some());
        let pitch: f64 = detected_pitch.unwrap();
        assert!((pitch - 220.0).abs() < 10.0); // Allow 10 Hz tolerance
    }

    #[test]
    fn test_pitch_tracking() {
        // Create a simple sine wave at 440 Hz
        let sample_rate = 44100;
        let duration = 0.5; // 0.5 seconds
        let frequency = 440.0; // A4
        let samples_count = (sample_rate as f64 * duration) as usize;

        let mut samples = Vec::new();
        for i in 0..samples_count {
            let t = i as f64 / sample_rate as f64;
            let value = (2.0 * PI * frequency * t).sin();
            samples.push(value as f32);
        }
        let samples = NonEmptyVec::new(samples).unwrap();
        let audio: AudioSamples<'_, f32> =
            AudioSamples::from_mono_vec(samples, sample_rate!(44100));

        // Test pitch tracking
        let window_size = NonZeroUsize::new(2048).unwrap();
        let hop_size = NonZeroUsize::new(512).unwrap();
        let pitch_track = audio
            .track_pitch(
                window_size,
                hop_size,
                PitchDetectionMethod::Yin,
                0.1,
                80.0,
                1000.0,
            )
            .unwrap();

        // Should have multiple pitch estimates
        assert!(!pitch_track.is_empty());

        // Most should detect around 440 Hz
        let detected_pitches: Vec<f64> =
            pitch_track.iter().filter_map(|(_, pitch)| *pitch).collect();

        assert!(!detected_pitches.is_empty());

        // Average should be close to 440 Hz
        let avg_pitch = detected_pitches.iter().sum::<f64>() / detected_pitches.len() as f64;
        assert!((avg_pitch - 440.0).abs() < 20.0); // Allow 20 Hz tolerance for windowed analysis
    }

    #[test]
    fn test_harmonic_analysis() {
        // Create a sawtooth wave (rich in harmonics) at 220 Hz
        let sample_rate = 44100;
        let duration = 1.0; // 1 second
        let frequency = 220.0; // A3
        let samples_count = (sample_rate as f64 * duration) as usize;

        let mut samples = Vec::new();
        for i in 0..samples_count {
            let t = i as f64 / sample_rate as f64;
            // Sawtooth wave approximation using first few harmonics
            let mut value = 0.0;
            for harmonic in 1..10 {
                value += (2.0 * PI * frequency * harmonic as f64 * t).sin() / harmonic as f64;
            }
            samples.push(value as f32);
        }
        let samples = NonEmptyVec::new(samples).unwrap();
        let audio: AudioSamples<'_, f32> =
            AudioSamples::from_mono_vec(samples, sample_rate!(44100));

        // Test harmonic analysis
        let harmonics = audio
            .harmonic_analysis(frequency, NonZeroUsize::new(5).unwrap(), 0.1, None, None)
            .unwrap();

        // Should have 5 harmonic components
        assert_eq!(harmonics.len(), 5);

        // Fundamental should be normalized to 1.0
        assert!((harmonics[0] - 1.0).abs() < 0.1);

        // Higher harmonics should be present but weaker
        for i in 1..harmonics.len() {
            assert!(harmonics[i] > 0.0);
            assert!(harmonics[i] <= harmonics[0]); // Should be weaker than fundamental
        }
    }

    #[test]
    fn test_silence_detection() {
        // Test with silence (should return None)
        let audio = AudioSamples::new_mono(Array1::<f32>::zeros(44100).into(), sample_rate!(44100))
            .unwrap();

        let detected_pitch = audio.detect_pitch_yin(0.1, 80.0, 1000.0).unwrap();
        assert!(detected_pitch.is_none());

        let detected_pitch_autocorr = audio.detect_pitch_autocorr(80.0, 1000.0).unwrap();
        assert!(detected_pitch_autocorr.is_none());
    }

    #[test]
    fn test_noise_robustness() {
        // Create a noisy sine wave at 440 Hz
        let sample_rate = 44100;
        let duration = 1.0; // 1 second
        let frequency = 440.0; // A4
        let samples_count = (sample_rate as f64 * duration) as usize;

        let mut samples = Vec::new();
        for i in 0..samples_count {
            let t = i as f64 / sample_rate as f64;
            let sine_value = (2.0 * PI * frequency * t).sin();
            // Add some noise
            let noise = (i as f64 * 0.1).sin() * 0.1; // Small amount of noise
            let value = sine_value + noise;
            samples.push(value as f32);
        }

        let samples = NonEmptyVec::new(samples).unwrap();
        let audio: AudioSamples<'_, f32> =
            AudioSamples::from_mono_vec(samples, sample_rate!(44100));

        // YIN should be more robust to noise
        let detected_pitch = audio.detect_pitch_yin(0.1, 80.0, 1000.0).unwrap();

        // Should still detect approximately 440 Hz despite noise
        assert!(detected_pitch.is_some());
        let pitch: f64 = detected_pitch.unwrap();
        assert!((pitch - 440.0).abs() < 20.0); // Allow larger tolerance for noisy signal
    }

    #[test]
    fn test_parameter_validation() {
        let audio: AudioSamples<'_, f32> =
            AudioSamples::from_mono_vec(non_empty_vec![1.0f32, 2.0, 3.0], sample_rate!(44100));

        // Test invalid threshold
        assert!(audio.detect_pitch_yin(-0.1, 80.0, 1000.0).is_err());
        assert!(audio.detect_pitch_yin(1.5, 80.0, 1000.0).is_err());

        // Test invalid frequency range
        assert!(audio.detect_pitch_yin(0.1, -80.0, 1000.0).is_err());
        assert!(audio.detect_pitch_yin(0.1, 1000.0, 800.0).is_err());

        // Test invalid harmonic analysis parameters
        assert!(
            audio
                .harmonic_analysis(-440.0, NonZeroUsize::new(5).unwrap(), 0.1, None, None)
                .is_err()
        );
        assert!(
            audio
                .harmonic_analysis(440.0, NonZeroUsize::new(5).unwrap(), -0.1, None, None)
                .is_err()
        );
        assert!(
            audio
                .harmonic_analysis(440.0, NonZeroUsize::new(5).unwrap(), 1.5, None, None)
                .is_err()
        );
    }
}
