//! Pitch of a mono buffer: one fundamental-frequency estimate for the whole signal
//! ([`AudioPitchAnalysis::detect_pitch_yin`],
//! [`AudioPitchAnalysis::detect_pitch_autocorr`]) or a frame-by-frame contour
//! ([`AudioPitchAnalysis::track_pitch`]), plus a harmonic-to-noise ratio, harmonic
//! magnitudes and key estimation. For frame-wise YIN and pYIN on centred frames,
//! with any number of channels and voicing probabilities, use
//! [`operations::pitch_tracking`](crate::operations::pitch_tracking); see
//! [Choosing between overlapping APIs](crate::operations#choosing-between-overlapping-apis).
//!
//! ```rust
//! use audio_samples::{AudioPitchAnalysis, sample_rate, sine_wave};
//! use std::time::Duration;
//!
//! let hz = 440.0f64;
//! let audio = sine_wave::<f64>(hz, Duration::from_millis(100), sample_rate!(44100), 1.0);
//!
//! // Search 80-1000 Hz: lags 44..=551 samples, so N = 4410 > 2 · 551.
//! let pitch = audio.detect_pitch_yin(0.1, 80.0, 1000.0).unwrap().unwrap();
//! // Parabolic interpolation gives sub-sample periods: well within 1 Hz here.
//! assert!((pitch - hz).abs() < 1.0, "detected {pitch:.2} Hz, expected {hz} Hz");
//! ```
//!
//! ## Concepts
//!
//! A periodic signal with period T₀ seconds repeats every τ₀ = T₀·f_s samples,
//! and f₀ = f_s/τ₀. Both time-domain detectors search for τ₀ by comparing the
//! signal with a copy of itself delayed by a lag τ. Autocorrelation
//! ([`autocorr_pitch_detection`]) is the simplest comparison; YIN
//! ([`yin_pitch_detection`], which documents the algorithm) is a normalised
//! difference that is far less prone to octave errors and to the bias from the
//! shrinking overlap at long lags.
//!
//! ## Conventions
//!
//! - **Search range.** A range `[f_min, f_max]` in Hz becomes the lag range
//!   `τ_min = ⌊f_s/f_max⌋ … τ_max = ⌊f_s/f_min⌋` samples. Every detector needs
//!   `τ_max < ⌊N/2⌋`: the signal or frame must be longer than two periods of the
//!   lowest frequency searched (N ≥ 2(⌊f_s/f_min⌋ + 1)); otherwise the result is
//!   `None`. For example, f_min = 80 Hz at f_s = 16 kHz needs N ≥ 402 samples
//!   (about 25 ms).
//! - [`AudioPitchAnalysis::detect_pitch_yin`] analyses the whole signal as one
//!   frame and returns `None` for unvoiced input.
//! - Every method needs mono input. The harmonic analysis, HNR and key estimation
//!   use the `spectrograms` crate for their FFTs.

use std::num::NonZeroUsize;

use crate::operations::traits::AudioPitchAnalysis;
use crate::operations::types::PitchDetectionMethod;
use crate::{
    AudioSampleError, AudioSampleResult, AudioSamples, AudioTypeConversion, ChannelRequirement,
    LayoutError, ParameterError, StandardSample,
};

use non_empty_slice::NonEmptySlice;
use spectrograms::{ChromaParams, SpectrogramPlanner, StftParams, WindowType};

/// A single frame of a [`PitchContour`].
///
/// `frequency` is `None` for unvoiced frames (no pitch detected).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PitchFrame {
    /// Frame onset time, in seconds from the start of the signal.
    pub time: f64,
    /// Detected fundamental frequency in Hz, or `None` if the frame is unvoiced.
    pub frequency: Option<f64>,
}

/// A time-ordered pitch track produced by [`AudioPitchAnalysis::track_pitch`].
///
/// Each [`PitchFrame`] pairs a frame onset time with the pitch detected in that
/// frame (or `None` when no pitch was found).
#[derive(Clone, Debug, PartialEq)]
pub struct PitchContour {
    frames: Vec<PitchFrame>,
}

impl PitchContour {
    /// Constructs a [`PitchContour`] from its frames.
    #[inline]
    pub(crate) fn new(frames: Vec<PitchFrame>) -> Self {
        Self { frames }
    }

    /// All frames in time order, voiced and unvoiced alike.
    #[inline]
    #[must_use]
    pub fn frames(&self) -> &[PitchFrame] {
        &self.frames
    }

    /// Iterates over voiced frames as `(time_seconds, frequency_hz)` pairs,
    /// skipping unvoiced frames.
    #[inline]
    pub fn voiced_frames(&self) -> impl Iterator<Item = (f64, f64)> + '_ {
        self.frames
            .iter()
            .filter_map(|frame| frame.frequency.map(|hz| (frame.time, hz)))
    }

    /// The mean of all voiced frequencies, or `None` when no frame is voiced.
    #[inline]
    #[must_use]
    pub fn mean_pitch(&self) -> Option<f64> {
        let mut count = 0usize;
        let mut sum = 0.0;
        for (_, hz) in self.voiced_frames() {
            sum += hz;
            count += 1;
        }
        if count == 0 {
            None
        } else {
            Some(sum / count as f64)
        }
    }

    /// The total number of frames (voiced and unvoiced).
    #[inline]
    #[must_use]
    pub fn len(&self) -> usize {
        self.frames.len()
    }

    /// Returns `true` when the contour contains no frames.
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.frames.is_empty()
    }
}

/// One of the twelve pitch classes of the chromatic scale.
///
/// Pitch class `C` has index 0, ascending chromatically to `B` at index 11.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum PitchClass {
    /// C
    C,
    /// C♯ / D♭
    CSharp,
    /// D
    D,
    /// D♯ / E♭
    DSharp,
    /// E
    E,
    /// F
    F,
    /// F♯ / G♭
    FSharp,
    /// G
    G,
    /// G♯ / A♭
    GSharp,
    /// A
    A,
    /// A♯ / B♭
    ASharp,
    /// B
    B,
}

impl PitchClass {
    /// The twelve pitch classes in chromatic order, from `C` (index 0) to `B` (index 11).
    pub const ALL: [Self; 12] = [
        Self::C,
        Self::CSharp,
        Self::D,
        Self::DSharp,
        Self::E,
        Self::F,
        Self::FSharp,
        Self::G,
        Self::GSharp,
        Self::A,
        Self::ASharp,
        Self::B,
    ];

    /// Maps a chromatic index `0..=11` to a pitch class (`0` = C, `11` = B).
    ///
    /// # Errors
    ///
    /// Returns [`AudioSampleError::Parameter`] if `index >= 12`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use audio_samples::operations::pitch_analysis::PitchClass;
    ///
    /// assert_eq!(PitchClass::from_index(9).unwrap(), PitchClass::A);
    /// assert!(PitchClass::from_index(12).is_err());
    /// ```
    #[inline]
    pub fn from_index(index: u8) -> AudioSampleResult<Self> {
        Self::ALL.get(usize::from(index)).copied().ok_or_else(|| {
            AudioSampleError::Parameter(ParameterError::out_of_range(
                "index",
                index,
                0,
                11,
                "a pitch class index is 0 (C) to 11 (B)",
            ))
        })
    }

    /// Returns the chromatic index of this pitch class (`C` = 0, `B` = 11).
    #[inline]
    #[must_use]
    pub fn to_index(self) -> u8 {
        match self {
            Self::C => 0,
            Self::CSharp => 1,
            Self::D => 2,
            Self::DSharp => 3,
            Self::E => 4,
            Self::F => 5,
            Self::FSharp => 6,
            Self::G => 7,
            Self::GSharp => 8,
            Self::A => 9,
            Self::ASharp => 10,
            Self::B => 11,
        }
    }
}

impl std::fmt::Display for PitchClass {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let name = match self {
            Self::C => "C",
            Self::CSharp => "C#",
            Self::D => "D",
            Self::DSharp => "D#",
            Self::E => "E",
            Self::F => "F",
            Self::FSharp => "F#",
            Self::G => "G",
            Self::GSharp => "G#",
            Self::A => "A",
            Self::ASharp => "A#",
            Self::B => "B",
        };
        f.write_str(name)
    }
}

/// The mode (tonality) of an estimated [`Key`].
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Mode {
    /// Major mode.
    Major,
    /// Minor mode.
    Minor,
}

/// A musical key estimate produced by [`AudioPitchAnalysis::estimate_key`].
#[derive(Clone, Copy, Debug)]
pub struct Key {
    /// The tonic pitch class of the key.
    pub tonic: PitchClass,
    /// The mode (major or minor) of the key.
    pub mode: Mode,
    /// Match confidence, in `[0.0, 1.0]`; higher values indicate a stronger match.
    pub confidence: f64,
}

impl<T> AudioPitchAnalysis for AudioSamples<'_, T>
where
    T: StandardSample,
    Self: AudioTypeConversion<Sample = T>,
{
    /// Detects the fundamental frequency using the YIN pitch detection algorithm.
    ///
    /// The whole signal is analysed as one frame. YIN computes the cumulative
    /// mean normalised difference d′(τ) (see the [module docs](self)), takes the
    /// first lag in the search range whose d′ falls below `threshold`, descends
    /// to the bottom of that dip and refines it by parabolic interpolation, so
    /// the period (and hence the frequency) is not restricted to whole samples.
    /// Lower thresholds are stricter (fewer false detections, more `None`);
    /// values in `[0.1, 0.2]` are typical.
    ///
    /// # Arguments
    ///
    /// - `threshold` - Absolute threshold on d′ (dimensionless), in `[0.0, 1.0]`.
    /// - `min_frequency` - Lowest frequency searched, in Hz (> 0). Sets the
    ///   longest lag `⌊f_s/min_frequency⌋`; the signal must be longer than
    ///   twice that lag.
    /// - `max_frequency` - Highest frequency searched, in Hz (> `min_frequency`).
    ///   Sets the shortest lag `⌊f_s/max_frequency⌋`. Values above Nyquist are
    ///   not rejected.
    ///
    /// # Returns
    ///
    /// - `Some(frequency_hz)` - Estimated fundamental frequency `f_s/τ̂` in Hz.
    /// - `None` - Signal of at most `2·⌊f_s/min_frequency⌋ + 1` samples,
    ///   silent, or no lag with d′ below `threshold` (unvoiced).
    ///
    /// # Errors
    ///
    /// Returns [crate::AudioSampleError::Layout] for multi-channel audio.
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
    /// assert!((pitch.unwrap() - hz).abs() < 1.0);
    /// ```
    #[inline]
    fn detect_pitch_yin(
        &self,
        threshold: f64,
        min_frequency: f64,
        max_frequency: f64,
    ) -> AudioSampleResult<Option<f64>> {
        if self.is_multi_channel() {
            return Err(AudioSampleError::Layout(
                LayoutError::channel_count_unsupported(
                    "detect_pitch_yin",
                    ChannelRequirement::Mono,
                    self.num_channels().get(),
                ),
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
            .expect("mono samples are always a contiguous slice");
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
    /// The whole signal is analysed as one frame. Finds the integer lag τ̂ that
    /// maximises the raw autocorrelation `r(τ) = Σ x[n]·x[n+τ]` within the lag
    /// range implied by `[min_frequency, max_frequency]` and returns `f_s/τ̂`.
    /// There is no sub-sample interpolation, so the result lies on the grid
    /// `f_s/k` (spacing ≈ f₀²/f_s). Fast and effective for clean, periodic
    /// signals, but less robust than YIN on noisy signals or speech.
    ///
    /// # Arguments
    ///
    /// - `min_frequency` - Lowest frequency searched, in Hz (> 1.0).
    /// - `max_frequency` - Highest frequency searched, in Hz (> `min_frequency`).
    ///
    /// # Returns
    ///
    /// - `Some(frequency_hz)` - Estimated fundamental frequency in Hz.
    /// - `None` - Signal of at most `2·⌊f_s/min_frequency⌋ + 1` samples,
    ///   silent, or no positive autocorrelation in the lag range.
    ///
    /// # Errors
    ///
    /// Returns [crate::AudioSampleError::Layout] for multi-channel audio.
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
            return Err(AudioSampleError::Layout(
                LayoutError::channel_count_unsupported(
                    "detect_pitch_autocorr",
                    ChannelRequirement::Mono,
                    self.num_channels().get(),
                ),
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
            .expect("mono samples are always a contiguous slice");

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
    /// The signal is split into frames of `window_size` samples starting at
    /// `0, hop_size, 2·hop_size, …` (no padding, no window function). Each frame
    /// is analysed independently with the same algorithm as
    /// [`detect_pitch_yin`](AudioPitchAnalysis::detect_pitch_yin) or
    /// [`detect_pitch_autocorr`](AudioPitchAnalysis::detect_pitch_autocorr).
    /// The final, truncated frames are kept while they are at least
    /// `window_size / 2` samples long. A frame is unvoiced (`None`) when it is
    /// at most `2·⌊f_s/min_frequency⌋ + 1` samples long, so choose
    /// `window_size` > 2 periods of the lowest f₀ of interest. Only
    /// [`PitchDetectionMethod::Yin`] and [`PitchDetectionMethod::Autocorrelation`]
    /// are implemented; other variants return an error.
    ///
    /// # Arguments
    ///
    /// - `window_size` - Analysis frame length in samples; must be ≤ signal length.
    /// - `hop_size` - Step between successive frames in samples; must be < `window_size`.
    /// - `method` - Pitch detection algorithm to use per frame.
    /// - `threshold` - YIN threshold on d′; ignored for autocorrelation.
    /// - `min_frequency` - Lowest frequency searched, in Hz.
    /// - `max_frequency` - Highest frequency searched, in Hz.
    ///
    /// # Returns
    ///
    /// A [`PitchContour`] of [`PitchFrame`]s in time order. `PitchFrame::time`
    /// is the frame **start** time `k·hop_size/f_s` in seconds (not the frame
    /// centre); `PitchFrame::frequency` is `None` when no pitch was found.
    ///
    /// # Errors
    ///
    /// Returns [crate::AudioSampleError::Layout] for multi-channel audio.
    /// Returns [crate::AudioSampleError::Parameter] if `window_size > signal length`,
    /// `hop_size >= window_size`, the threshold or frequency range is invalid, or
    /// `method` is [`PitchDetectionMethod::Cepstrum`] or
    /// [`PitchDetectionMethod::HarmonicProduct`] (not implemented).
    ///
    /// # Example
    ///
    /// ```
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
    /// let avg = track.mean_pitch().unwrap();
    /// assert!((avg - hz).abs() < 1.0);
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
    ) -> AudioSampleResult<PitchContour> {
        if self.is_multi_channel() {
            return Err(AudioSampleError::Layout(
                LayoutError::channel_count_unsupported(
                    "track_pitch",
                    ChannelRequirement::Mono,
                    self.num_channels().get(),
                ),
            ));
        }

        if window_size.get() <= hop_size.get() {
            return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
                "window_hop_size",
                "Window size must be greater than hop size",
            )));
        }

        if self.len() < window_size {
            return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
                "window_size",
                "Window size cannot be larger than the total number of samples",
            )));
        }

        // Unsupported methods are rejected once, before the analysis loop.
        if !matches!(
            method,
            PitchDetectionMethod::Yin | PitchDetectionMethod::Autocorrelation
        ) {
            return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
                "method",
                format!("Pitch detection method {method:?} is not supported by track_pitch"),
            )));
        }

        // Validate the per-frame detector parameters once, up front. They are
        // constant across frames, so this gives the same errors as checking them
        // in every `detect_pitch_*` call and lets the loop call the low-level
        // detectors directly.
        match method {
            PitchDetectionMethod::Yin => {
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
            }
            PitchDetectionMethod::Autocorrelation
                if (min_frequency <= 1.0 || max_frequency <= min_frequency) =>
            {
                return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
                    "frequency_range",
                    format!("Invalid frequency range {min_frequency} - {max_frequency}"),
                )));
            }
            // Unsupported methods are rejected before this point.
            _ => {}
        }

        let window_size = window_size.get();
        let hop_size = hop_size.get();
        // The signal is converted to f64 once; the low-level detectors run on
        // sub-slices of that buffer.
        let samples = self.as_f64();
        let samples_slice = samples
            .as_slice()
            .expect("mono samples are always a contiguous slice");

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
            let time_seconds = start as f64 / sample_rate;

            // The low-level detector runs on the window sub-slice with the tau
            // bounds and the `max_tau >= window_len / 2` early-out of
            // `detect_pitch_yin` / `detect_pitch_autocorr`, so each frame gives
            // exactly the result of calling them on it.
            let window_len = window.len();
            let frequency: Option<f64> = match method {
                PitchDetectionMethod::Yin => {
                    let min_tau = (sample_rate / max_frequency) as usize;
                    let max_tau = (sample_rate / min_frequency) as usize;
                    if max_tau >= window_len / 2 {
                        None
                    } else {
                        // safety: window is guaranteed non-empty since
                        // end - start >= window_size / 2 > 0
                        let window = unsafe { NonEmptySlice::new_unchecked(window) };
                        yin_pitch_detection(window, min_tau, max_tau, threshold)
                            .map(|tau| sample_rate / tau)
                    }
                }
                PitchDetectionMethod::Autocorrelation => {
                    let min_tau = (sample_rate / max_frequency) as usize;
                    let max_tau = (sample_rate / min_frequency) as usize;
                    if max_tau >= window_len / 2 {
                        None
                    } else {
                        autocorr_pitch_detection(window, min_tau, max_tau)
                            .map(|tau| sample_rate / tau)
                    }
                }
                // Unsupported methods are rejected before this loop, so this arm is
                // unreachable in practice; return `None` defensively without logging.
                _ => None,
            };

            results.push(PitchFrame {
                time: time_seconds,
                frequency,
            });
        }

        Ok(PitchContour::new(results))
    }

    /// Computes a spectral harmonic-to-noise ratio (HNR) in decibels.
    ///
    /// One windowed power spectrum P\[k\] of the whole signal is computed
    /// (bin spacing Δf = f_s/`n_fft` Hz). Bin k (frequency k·Δf, DC included)
    /// counts as *harmonic* when |k·Δf − h·f₀| < Δf for some
    /// h = 1..=`num_harmonics`, i.e. only the ±1-bin neighbourhood of each
    /// harmonic; every other bin counts as *noise*:
    ///
    /// ```text
    /// HNR = 10·log10( Σ_harmonic P[k] / Σ_noise P[k] )   dB
    /// ```
    ///
    /// Window leakage outside that neighbourhood (the Hann main lobe alone is
    /// ±2 bins wide) is counted as noise, so the estimate is biased low and
    /// even a pure tone gives a small finite value that depends on where the
    /// tone falls relative to the bins: a 1 s, 440 Hz sine at 44.1 kHz
    /// (`n_fft` = signal length, Hann) lies exactly on a bin, only that bin
    /// counts as harmonic and HNR = 10·log10(0.25/0.125) ≈ 3.01 dB; off-bin
    /// tones give about 13 dB. Compare values computed with the same `n_fft`
    /// and window only. This is not the autocorrelation HNR of Boersma (1993)
    /// used by Praat.
    ///
    /// # Arguments
    ///
    /// - `fundamental_freq` - Known fundamental frequency in Hz (> 0).
    /// - `num_harmonics` - Number of harmonics to accumulate into harmonic power.
    /// - `n_fft` - FFT size. Defaults to the signal length when `None`.
    /// - `window_type` - Window function applied before FFT. Defaults to Hanning when `None`.
    ///
    /// # Returns
    ///
    /// HNR in dB. Returns `f64::INFINITY` only when every bin outside the
    /// harmonic neighbourhoods is exactly zero.
    ///
    /// # Errors
    ///
    /// Returns [crate::AudioSampleError::Layout] for multi-channel audio.
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
    /// // A pure sine: the result is finite, not infinite, because window
    /// // leakage around 440 Hz is partly counted as noise.
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
            return Err(AudioSampleError::Layout(
                LayoutError::channel_count_unsupported(
                    "harmonic_to_noise_ratio",
                    ChannelRequirement::Mono,
                    self.num_channels().get(),
                ),
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
            .expect("mono samples are always a contiguous slice");
        // safety: samples is guaranteed non-empty by construction
        let samples_slice = unsafe { NonEmptySlice::new_unchecked(samples_slice) };
        let planner = SpectrogramPlanner::new();
        let n_fft = n_fft.unwrap_or(samples.len());

        // Compute power spectrum
        let spectrum = planner.compute_power_spectrum(
            samples_slice,
            n_fft,
            window_type.unwrap_or_default(),
        )?;
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
    /// - `fundamental_freq` - Fundamental frequency in Hz (> 0).
    /// - `num_harmonics` - Number of harmonics to extract, including the fundamental.
    /// - `tolerance` - Fractional bandwidth around each harmonic to search, in `[0.0, 1.0]`.
    /// - `n_fft` - FFT size. Defaults to the signal length when `None`.
    /// - `window_type` - Window function applied before FFT. Defaults to Hanning when `None`.
    ///
    /// # Returns
    ///
    /// A `Vec<f64>` of length `num_harmonics`. Index 0 is always 1.0 after
    /// normalisation (unless the fundamental has zero power, in which case all
    /// values are the raw magnitudes).
    ///
    /// # Errors
    ///
    /// Returns [crate::AudioSampleError::Layout] for multi-channel audio.
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
    /// // Sawtooth wave at 220 Hz - rich in harmonics.
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
            return Err(AudioSampleError::Layout(
                LayoutError::channel_count_unsupported(
                    "harmonic_analysis",
                    ChannelRequirement::Mono,
                    self.num_channels().get(),
                ),
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
            .expect("mono samples are always a contiguous slice");

        // safety: samples is guaranteed non-empty by construction
        let samples_slice = unsafe { NonEmptySlice::new_unchecked(samples_slice) };

        let sample_rate = self.sample_rate_hz();
        let planner = SpectrogramPlanner::new();
        let n_fft = n_fft.unwrap_or(samples.len());
        // Compute power spectrum
        let spectrum = planner.compute_power_spectrum(
            samples_slice,
            n_fft,
            window_type.unwrap_or_default(),
        )?;
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
    /// Computes a chromagram via the `spectrograms` crate and averages it over
    /// all frames into a 12-element chroma vector c (index 0 = C). For each of
    /// the 24 keys the Krumhansl-Kessler major or minor profile p (index 0 =
    /// tonic) is rotated so the tonic weight lands on the key's pitch class t,
    /// `p_t[i] = p[(i − t) mod 12]`, and the Pearson correlation ρ(c, p_t) is
    /// computed (Krumhansl-Schmuckler algorithm). The key with the largest ρ is
    /// returned and `confidence = (ρ + 1)/2`.
    ///
    /// # Arguments
    ///
    /// - `stft_params` - STFT parameters controlling frame size and hop for
    ///   chromagram computation.
    ///
    /// # Returns
    ///
    /// A [`Key`](crate::Key) with the estimated `tonic`
    /// ([`PitchClass`](crate::PitchClass)), `mode`
    /// ([`Mode`](crate::Mode)), and a `confidence` in `[0.0, 1.0]`; higher
    /// confidence values indicate a stronger match.
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
    /// let key = audio.estimate_key(&params).unwrap();
    /// assert!((0.0..=1.0).contains(&key.confidence));
    /// ```
    #[inline]
    fn estimate_key(&self, stft_params: &StftParams) -> AudioSampleResult<Key> {
        let samples = self.to_format::<f64>();

        let samples_slice = samples
            .as_slice()
            .expect("mono samples are always a contiguous slice");
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

        // Convert correlation to confidence (normalize to 0-1 range)
        let confidence = f64::midpoint(best_correlation, 1.0);

        Ok(Key {
            tonic: PitchClass::ALL[best_key],
            mode: if is_major { Mode::Major } else { Mode::Minor },
            confidence,
        })
    }
}

/// Computes Pearson correlation between a chroma vector and a rotated key profile.
///
/// `profile[0]` is the weight of the tonic; the profile is rotated so that its
/// tonic weight lands on chroma index `tonic` (`rotated[i] = profile[(i - tonic) mod 12]`).
fn calculate_correlation(chroma: &[f64], profile: &[f64], tonic: usize) -> f64 {
    debug_assert_eq!(chroma.len(), 12);
    debug_assert_eq!(profile.len(), 12);

    // Rotate the profile so that its tonic (index 0) sits at chroma index `tonic`.
    let mut rotated_profile = [0.0; 12];
    for i in 0..12 {
        rotated_profile[i] = profile[(i + 12 - tonic) % 12];
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
/// The period is fractional (refined by parabolic interpolation); convert it to
/// Hz with `sample_rate / τ`. This is the primitive behind
/// [`AudioPitchAnalysis::detect_pitch_yin`] and
/// [`AudioPitchAnalysis::track_pitch`] with [`PitchDetectionMethod::Yin`].
///
/// # Algorithm
///
/// Steps 2 to 5 of A. de Cheveigné and H. Kawahara, "YIN, a fundamental
/// frequency estimator for speech and music", JASA 111(4):1917-1930, 2002, with
/// the departures listed below. x\[j\] are the samples (`n` of them) and τ is a
/// lag in samples.
///
/// 1. Difference function over every overlapping pair of samples:
///
///    ```text
///    d(τ) = Σ_{j=0}^{n−1−τ} (x[j] − x[j+τ])²,   τ = 1 … max_tau
///    ```
///
/// 2. Cumulative mean normalised difference (CMND, dimensionless):
///
///    ```text
///    d′(0) = 1,   d′(τ) = d(τ) / ( (1/τ) · Σ_{j=1}^{τ} d(j) )
///    ```
///
///    The running mean always starts at τ = 1; `[min_tau, max_tau]` only
///    restricts the search.
/// 3. Absolute threshold: take the smallest τ in `[min_tau, max_tau]` with
///    d′(τ) < `threshold`, then move to the bottom of that dip (keep increasing τ
///    while d′(τ+1) < d′(τ)). If no lag dips below the threshold, return `None`.
/// 4. Parabolic interpolation through (τ−1, τ, τ+1):
///
///    ```text
///    τ̂ = τ + ½·(d′(τ−1) − d′(τ+1)) / (d′(τ−1) − 2·d′(τ) + d′(τ+1))
///    ```
///
///    There is no shift when τ is at either end of `[min_tau, max_tau]` or when
///    the vertex lies more than one lag away from τ.
///
/// Departures from the paper: (a) the paper's step 5 interpolates the raw
/// difference function d(τ) around the selected lag (interpolating d′ is
/// slightly biased), whereas this code fits the parabola to d′; (b) the paper's
/// step 6 ("best local estimate") is not implemented, and when no lag dips below
/// the threshold the paper takes the global minimum of d′ rather than declaring
/// the frame unvoiced; (c) the paper sums d(τ) over a fixed integration window
/// starting at the beginning of the frame, whereas this code sums over all
/// n − τ overlapping pairs. A fixed window measures the pitch of the frame's
/// early part, which biases the estimate when the pitch moves within the frame
/// (about 9 cents on a 1.5 octave/s glide with 2048-sample frames at
/// 22.05 kHz); the full overlap keeps the analysed span centred on the frame.
///
/// # Arguments
///
/// - `samples` - Non-empty slice of audio samples.
/// - `min_tau` - Smallest lag to search (inclusive), in samples. Set to
///   `(sample_rate / max_frequency) as usize`.
/// - `max_tau` - Largest lag to search (inclusive), in samples. Set to
///   `(sample_rate / min_frequency) as usize`.
/// - `threshold` - Maximum CMND value accepted as a valid pitch. Typical
///   values are in `[0.1, 0.2]`.
///
/// # Returns
///
/// `Some(tau)` - period in samples (fractional, after parabolic interpolation);
/// convert to Hz with `sample_rate / tau`.
/// `None` - no CMND value below `threshold`, `min_tau > max_tau`, or
/// `max_tau ≥ samples.len() / 2`.
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
///     // True period 44100/440 = 100.23 samples; t is fractional.
///     assert!((t - sr / hz).abs() < 0.1);
///     let detected_hz = sr / t;
///     assert!((detected_hz - hz).abs() < 1.0);
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

    // Lag 0 is never a period; searching it would divide by a zero running mean.
    let min_tau = min_tau.max(1);
    if min_tau > max_tau {
        return None;
    }

    // Step 1: difference function d(τ) = Σ_{j < n-τ} (x_j - x_{j+τ})² for EVERY lag
    // τ in 1..=max_tau. The cumulative mean in step 2 runs over all lags from 1, so
    // the lags below `min_tau` must be computed too (they are just not searched).
    let mut diff_fn = vec![0.0; max_tau + 1];
    for (tau, d) in diff_fn.iter_mut().enumerate().skip(1) {
        *d = samples[..n - tau]
            .iter()
            .zip(&samples[tau..])
            .map(|(&a, &b)| (a - b) * (a - b))
            .sum();
    }

    // Step 2: cumulative mean normalised difference d'(τ) = d(τ) / ((1/τ) Σ_{k=1..τ} d(k))
    // (de Cheveigné & Kawahara 2002, eq. 8).
    let mut cmnd = vec![1.0; max_tau + 1];
    let mut running_sum = 0.0;
    for tau in 1..=max_tau {
        running_sum += diff_fn[tau];
        if running_sum > 0.0 {
            cmnd[tau] = diff_fn[tau] * tau as f64 / running_sum;
        }
    }

    // Step 3: absolute threshold. Take the first lag in [min_tau, max_tau] whose
    // d' falls below `threshold`, then follow that dip down to its local minimum
    // (the bottom of the trough, however wide it is).
    let mut tau = (min_tau..=max_tau).find(|&t| cmnd[t] < threshold)?;
    while tau < max_tau && cmnd[tau + 1] < cmnd[tau] {
        tau += 1;
    }

    // Step 4: parabolic interpolation of the trough (no shift at the ends of the
    // search range, and none when the vertex lies more than one lag away).
    let shift = if tau > min_tau && tau < max_tau {
        let (prev, mid, next) = (cmnd[tau - 1], cmnd[tau], cmnd[tau + 1]);
        let a = next + prev - 2.0 * mid;
        let b = (next - prev) / 2.0;
        if b.abs() >= a.abs() { 0.0 } else { -b / a }
    } else {
        0.0
    };

    Some(tau as f64 + shift)
}

/// Low-level autocorrelation pitch detection that returns the fundamental period in samples.
///
/// Finds the integer lag in `[min_tau, max_tau]` that maximises the raw
/// autocorrelation `r(τ) = Σ_{n=0}^{N−1−τ} x[n]·x[n+τ]` (no normalisation, no
/// mean removal, no interpolation) and returns it as the estimated period.
/// The caller converts to frequency with `sample_rate / tau`. Unlike YIN, the
/// correlation is not normalised, making it fast but more susceptible to noise
/// and biased towards short lags, because the overlap shrinks as τ grows. The
/// lag is an integer, so near f₀ the frequency grid has a spacing of about
/// f₀²/f_s (12 Hz at 440 Hz and 16 kHz).
///
/// This is the low-level primitive used by
/// [`AudioPitchAnalysis::detect_pitch_autocorr`].
///
/// # Arguments
///
/// - `samples` - Slice of audio samples.
/// - `min_tau` - Smallest lag to search (inclusive), in samples. Set to
///   `(sample_rate / max_frequency) as usize`.
/// - `max_tau` - Largest lag to search (inclusive), in samples. Set to
///   `(sample_rate / min_frequency) as usize`.
///
/// # Returns
///
/// `Some(tau)` - integer lag in samples (as `f64`) with highest autocorrelation; convert to Hz
/// with `sample_rate / tau`.
/// `None` - no positive autocorrelation found, or `max_tau ≥ samples.len() / 2`.
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

    /// A 5-harmonic tone with amplitudes 0.6/k.
    fn harmonic_tone(f0: f64, sr: f64, seconds: f64) -> Vec<f64> {
        let n = (sr * seconds) as usize;
        (0..n)
            .map(|i| {
                let t = i as f64 / sr;
                (1..=5)
                    .map(|k| 0.6 / k as f64 * (2.0 * PI * f0 * k as f64 * t).sin())
                    .sum()
            })
            .collect()
    }

    fn cents(f: f64, reference: f64) -> f64 {
        1200.0 * (f / reference).log2()
    }

    fn mono_f64(samples: Vec<f64>, sr: u32) -> AudioSamples<'static, f64> {
        AudioSamples::new_mono(
            Array1::from(samples),
            std::num::NonZeroU32::new(sr).unwrap(),
        )
        .unwrap()
    }

    /// YIN must follow the first sub-threshold dip down to its local minimum. An
    /// arg-min over a +-5-lag window around the first sub-threshold lag stops on
    /// the dip's flank for low pitches: 82.41 Hz at 22.05 kHz comes out as
    /// 84.16 Hz (+36 cents) and at 44.1 kHz as 84.97 Hz (+53 cents).
    #[test]
    fn yin_follows_dip_to_local_minimum() {
        for sr in [22_050u32, 44_100] {
            let f0 = 82.41;
            let audio = mono_f64(harmonic_tone(f0, f64::from(sr), 0.1), sr);
            let hz = audio.detect_pitch_yin(0.1, 50.0, 2000.0).unwrap().unwrap();
            assert!(
                cents(hz, f0).abs() < 1.0,
                "sr {sr}: detected {hz} Hz for {f0} Hz ({:+.2} cents)",
                cents(hz, f0)
            );
        }
    }

    /// The period is refined by parabolic interpolation, so it is not an integer
    /// number of samples: an integer period would give exactly 441 Hz (+3.9 cents)
    /// for 440 Hz at 22.05 kHz, and 980 Hz (-13.7 cents) for 987.77 Hz at 44.1 kHz.
    #[test]
    fn yin_parabolic_interpolation_gives_fractional_period() {
        for (sr, f0) in [(22_050u32, 440.0), (44_100, 987.77)] {
            let audio = mono_f64(harmonic_tone(f0, f64::from(sr), 0.1), sr);
            let hz = audio.detect_pitch_yin(0.1, 50.0, 2000.0).unwrap().unwrap();
            assert!(
                cents(hz, f0).abs() < 1.0,
                "sr {sr}: detected {hz} Hz for {f0} Hz ({:+.2} cents)",
                cents(hz, f0)
            );
        }

        // Clean sine with a period of exactly 100.25 samples.
        let period = 100.25;
        let samples: Vec<f64> = (0..4096)
            .map(|i| (2.0 * PI * i as f64 / period).sin())
            .collect();
        let samples_slice: &[f64] = &samples;
        let slice = non_empty_slice::non_empty_slice!(samples_slice);
        let tau = yin_pitch_detection(slice, 20, 400, 0.1).unwrap();
        assert!(
            (tau - period).abs() < 0.05,
            "period {tau}, expected {period}"
        );
    }

    /// The cumulative mean in the CMND runs over ALL lags from 1. Summing zeros
    /// below `min_tau` (derived from `max_frequency`) would shrink the denominator
    /// and inflate the CMND whenever `max_frequency` is low. On this noisy 110 Hz
    /// tone the CMND minimum is below the 0.2 threshold, so `max_frequency = 300`
    /// must find the pitch.
    #[test]
    fn yin_cmnd_does_not_depend_on_max_frequency() {
        let sr = 44_100u32;
        let mut samples = harmonic_tone(110.0, f64::from(sr), 0.1);
        // Deterministic, roughly Gaussian noise (sum of 4 uniforms from an LCG).
        let mut state: u64 = 0x2545_F491_4F6C_DD1D;
        let mut uniform = || {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (state >> 11) as f64 / (1u64 << 53) as f64 - 0.5
        };
        for x in &mut samples {
            let g = (uniform() + uniform() + uniform() + uniform()) * 3.0f64.sqrt();
            *x += 0.25 * g;
        }
        let audio = mono_f64(samples, sr);

        let wide = audio.detect_pitch_yin(0.2, 60.0, 2000.0).unwrap();
        let narrow = audio.detect_pitch_yin(0.2, 60.0, 300.0).unwrap();
        let (wide, narrow) = (wide.expect("voiced"), narrow.expect("voiced"));
        assert!(
            (wide - narrow).abs() < 1e-9,
            "fmax 2000 -> {wide} Hz, fmax 300 -> {narrow} Hz"
        );
        // Heavy noise moves the estimate by a few tens of cents; it must stay on 110 Hz.
        assert!(cents(wide, 110.0).abs() < 50.0, "detected {wide} Hz");
    }

    /// End-to-end: `track_pitch` with YIN on an exponential 110 -> 880 Hz glide
    /// (3 harmonics). The median and maximum frame errors must stay within 4 / 8 cents
    /// at 22.05 kHz and 2 / 5 cents at 44.1 kHz.
    #[test]
    fn yin_track_pitch_glide_accuracy() {
        for (sr, max_median, max_max) in [(22_050u32, 4.0, 8.0), (44_100, 2.0, 5.0)] {
            let srf = f64::from(sr);
            let n = 2 * sr as usize;
            let mut phase = 0.0;
            let samples: Vec<f64> = (0..n)
                .map(|i| {
                    let f = 110.0 * (1.5 * i as f64 / srf).exp2();
                    phase += 2.0 * PI * f / srf;
                    (1..=3)
                        .map(|k| 0.6 / k as f64 * (k as f64 * phase).sin())
                        .sum()
                })
                .collect();
            let track = mono_f64(samples, sr)
                .track_pitch(
                    NonZeroUsize::new(2048).unwrap(),
                    NonZeroUsize::new(512).unwrap(),
                    PitchDetectionMethod::Yin,
                    0.1,
                    65.0,
                    2093.0,
                )
                .unwrap();
            let mut errors: Vec<f64> = track
                .frames()
                .iter()
                // full-length frames only (trailing partial windows shrink)
                .filter(|fr| fr.time * srf + 2048.0 <= n as f64)
                .map(|fr| {
                    let truth = 110.0 * (1.5 * (fr.time + 1024.0 / srf)).exp2();
                    cents(fr.frequency.expect("voiced"), truth).abs()
                })
                .collect();
            errors.sort_by(f64::total_cmp);
            let (median, max) = (errors[errors.len() / 2], errors[errors.len() - 1]);
            assert!(
                median < max_median && max < max_max,
                "sr {sr}: median {median:.2} cents, max {max:.2} cents"
            );
        }
    }

    /// `calculate_correlation(chroma, profile, t)` must be 1 when the chroma IS the
    /// profile transposed to tonic `t`. The rotation `profile[(i + t) % 12]` would
    /// transpose by `-t` instead and report every key other than C as its mirror
    /// image (D major as A# major, E major as G# major).
    #[test]
    fn key_profile_rotation_puts_tonic_at_chroma_index() {
        let major = [
            6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88,
        ];
        for tonic in 0..12 {
            let chroma: Vec<f64> = (0..12).map(|i| major[(i + 12 - tonic) % 12]).collect();
            let r = calculate_correlation(&chroma, &major, tonic);
            assert!((r - 1.0).abs() < 1e-12, "tonic {tonic}: correlation {r}");
        }
    }

    /// I-IV-V-I cadences of 3-harmonic major triads in D and E major.
    #[test]
    fn estimate_key_d_and_e_major_cadences() {
        let sr = 22_050u32;
        let chord = |midis: &[i32]| -> Vec<f64> {
            let n = (0.5 * f64::from(sr)) as usize;
            (0..n)
                .map(|i| {
                    let t = i as f64 / f64::from(sr);
                    midis
                        .iter()
                        .map(|&m| {
                            let f0 = 440.0 * ((f64::from(m) - 69.0) / 12.0).exp2();
                            (1..=3)
                                .map(|k| 0.6 / k as f64 * (2.0 * PI * f0 * k as f64 * t).sin())
                                .sum::<f64>()
                        })
                        .sum()
                })
                .collect()
        };
        let stft = StftParams::new(
            NonZeroUsize::new(4096).unwrap(),
            NonZeroUsize::new(1024).unwrap(),
            WindowType::Hanning,
            true,
        )
        .unwrap();
        for (root, expected) in [(62, PitchClass::D), (64, PitchClass::E)] {
            let mut samples = Vec::new();
            for triad in [[0, 4, 7], [5, 9, 12], [7, 11, 14], [0, 4, 7]] {
                let midis: Vec<i32> = triad.iter().map(|i| root + i).collect();
                samples.extend(chord(&midis));
            }
            let key = mono_f64(samples, sr).estimate_key(&stft).unwrap();
            assert_eq!(key.tonic, expected, "root {root}: got {key:?}");
            assert_eq!(key.mode, Mode::Major, "root {root}: got {key:?}");
        }
    }

    /// `track_pitch` rejects unsupported pitch detection methods with a single
    /// up-front `Err`, before the analysis loop.
    #[test]
    fn track_pitch_unsupported_method_errors_up_front() {
        let sample_rate = 44100;
        let samples: Vec<f64> = (0..8192)
            .map(|i| (2.0 * PI * 220.0 * i as f64 / sample_rate as f64).sin())
            .collect();
        let samples = NonEmptyVec::new(samples).unwrap();
        let audio: AudioSamples<'_, f64> =
            AudioSamples::from_mono_vec(samples, sample_rate!(44100));

        let result = audio.track_pitch(
            std::num::NonZeroUsize::new(2048).unwrap(),
            std::num::NonZeroUsize::new(512).unwrap(),
            PitchDetectionMethod::Cepstrum,
            0.1,
            80.0,
            1000.0,
        );
        assert!(
            result.is_err(),
            "unsupported method must return Err, got {result:?}"
        );

        // A supported method succeeds on the same input.
        let ok = audio.track_pitch(
            std::num::NonZeroUsize::new(2048).unwrap(),
            std::num::NonZeroUsize::new(512).unwrap(),
            PitchDetectionMethod::Yin,
            0.1,
            80.0,
            1000.0,
        );
        assert!(ok.is_ok(), "supported method must succeed");
    }

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
        assert!(pitch_track.voiced_frames().next().is_some());

        // Average should be close to 440 Hz
        let avg_pitch = pitch_track.mean_pitch().unwrap();
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
        let audio =
            AudioSamples::new_mono(Array1::<f32>::zeros(44100), sample_rate!(44100)).unwrap();

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

    #[test]
    fn pitch_contour_mean_pitch_averages_voiced_frames_only() {
        let contour = PitchContour::new(vec![
            PitchFrame {
                time: 0.0,
                frequency: Some(100.0),
            },
            PitchFrame {
                time: 0.1,
                frequency: None,
            },
            PitchFrame {
                time: 0.2,
                frequency: Some(300.0),
            },
        ]);

        assert_eq!(contour.len(), 3);
        assert!(!contour.is_empty());
        // mean of voiced frequencies: (100 + 300) / 2 = 200
        assert_eq!(contour.mean_pitch(), Some(200.0));
        let voiced: Vec<(f64, f64)> = contour.voiced_frames().collect();
        assert_eq!(voiced, vec![(0.0, 100.0), (0.2, 300.0)]);
    }

    #[test]
    fn pitch_contour_mean_pitch_is_none_when_all_unvoiced() {
        let contour = PitchContour::new(vec![PitchFrame {
            time: 0.0,
            frequency: None,
        }]);
        assert_eq!(contour.mean_pitch(), None);

        let empty = PitchContour::new(Vec::new());
        assert!(empty.is_empty());
        assert_eq!(empty.mean_pitch(), None);
    }

    #[test]
    fn key_pitch_class_index_round_trips_and_displays() {
        for idx in 0u8..12 {
            let pc = PitchClass::from_index(idx).unwrap();
            assert_eq!(pc.to_index(), idx);
        }
        assert!(PitchClass::from_index(12).is_err());
        assert_eq!(PitchClass::C.to_string(), "C");
        assert_eq!(PitchClass::CSharp.to_string(), "C#");
        assert_eq!(PitchClass::B.to_string(), "B");

        // A combined key index: pc = idx % 12, mode = Major if idx < 12 else Minor.
        let major_idx = 4usize; // E major
        assert_eq!(
            PitchClass::from_index((major_idx % 12) as u8).unwrap(),
            PitchClass::E
        );
        assert_eq!(
            if major_idx < 12 {
                Mode::Major
            } else {
                Mode::Minor
            },
            Mode::Major
        );

        let minor_idx = 21usize; // 21 - 12 = 9 -> A minor
        assert_eq!(
            PitchClass::from_index((minor_idx % 12) as u8).unwrap(),
            PitchClass::A
        );
        assert_eq!(
            if minor_idx < 12 {
                Mode::Major
            } else {
                Mode::Minor
            },
            Mode::Minor
        );
    }
}
