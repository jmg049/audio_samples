//! Beat tracking operations for [`AudioSamples`].
//!
//! ## What
//!
//! This module implements the [`AudioBeatTracking`] trait and provides
//! supporting types ([`BeatTrackingConfig`], [`BeatTrackingData`]) and
//! lower-level helpers ([`onset_strength_envelope`], [`track_beats_core`])
//! for tempo-aware beat detection.
//!
//! ## Why
//!
//! Rhythmic analysis — tempo estimation, beat alignment, and
//! synchronisation — is central to music production, DJ software, and
//! audio feature extraction.  Encapsulating the detection pipeline
//! behind a single trait keeps callers isolated from the onset
//! detection internals.
//!
//! ## How
//!
//! Build a [`BeatTrackingConfig`] with the target tempo and onset
//! detection parameters, then call [`AudioBeatTracking::detect_beats`]
//! on any [`AudioSamples`] value.  The lower-level [`track_beats_core`]
//! function is available when you already have a pre-computed onset
//! envelope.
//!
//! ```
//! use audio_samples::operations::beat::track_beats_core;
//! use non_empty_slice::NonEmptyVec;
//! use std::num::NonZeroUsize;
//!
//! let onset = NonEmptyVec::new(vec![0.0f64; 100]).unwrap();
//! let beats = track_beats_core(
//!     &onset,
//!     120.0,
//!     44100.0,
//!     NonZeroUsize::new(512).unwrap(),
//!     None,
//! ).unwrap();
//! assert!(!beats.is_empty());
//! ```

use std::num::NonZeroUsize;

use non_empty_slice::{NonEmptySlice, NonEmptyVec};

use crate::{
    AudioOnsetDetection, AudioSampleError, AudioSampleResult, AudioSamples, ParameterError,
    operations::{onset::OnsetDetectionConfig, traits::AudioBeatTracking},
    traits::StandardSample,
};

/// Results of a beat tracking analysis.
///
/// Holds the estimated tempo and the detected beat timestamps returned
/// by [`AudioBeatTracking::detect_beats`].
///
/// # Ordering
///
/// The elements of `beat_times` are **not** sorted chronologically.
/// The first element is the timestamp of the global onset peak
/// (highest-energy onset in the signal).  Forward beats follow in
/// causal order; backward beats are appended afterwards in reverse
/// temporal order.  Sort `beat_times` if you need chronological
/// order.
#[derive(Clone, Debug)]
pub struct BeatTrackingData {
    /// Estimated tempo in beats per minute.
    pub tempo_bpm: f64,
    /// Beat timestamps in seconds, in detection order.
    ///
    /// The first element is the global onset peak; forward beats
    /// follow in causal order; backward beats are appended last in
    /// reverse temporal order.  See the struct-level documentation.
    pub beat_times: Vec<f64>,
    /// The configuration used to produce this result.
    pub config: BeatTrackingConfig,
}

impl BeatTrackingData {
    /// Create a `BeatTrackingData` from pre-computed components.
    ///
    /// # Arguments
    /// - `tempo_bpm` – Estimated tempo in beats per minute.
    /// - `beat_times` – Beat timestamps in seconds, in detection order.
    /// - `config` – The [`BeatTrackingConfig`] used to produce this
    ///   result.
    ///
    /// # Returns
    /// A new [`BeatTrackingData`].
    #[inline]
    pub fn new(tempo_bpm: f64, beat_times: Vec<f64>, config: BeatTrackingConfig) -> Self {
        Self {
            tempo_bpm,
            beat_times,
            config,
        }
    }
}

impl core::fmt::Display for BeatTrackingData {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        writeln!(f, "Estimated Tempo: {:.2} BPM", self.tempo_bpm)?;
        writeln!(f, "Detected Beats (s):")?;
        for &time in &self.beat_times {
            writeln!(f, "{:.3}", time)?;
        }
        Ok(())
    }
}

/// Configuration for beat detection.
///
/// Controls the target tempo, timing tolerance, and the underlying
/// onset detection pipeline.  Pass to
/// [`AudioBeatTracking::detect_beats`].
///
/// # Invariants
/// - `tempo_bpm` must be positive (> 0).
/// - `tolerance`, when `Some`, should be positive and smaller than
///   the inter-beat interval.
#[derive(Clone, Debug)]
pub struct BeatTrackingConfig {
    /// Target tempo in beats per minute; must be > 0.
    pub tempo_bpm: f64,
    /// Beat timing tolerance in seconds.
    ///
    /// The beat tracker searches for a local onset peak within
    /// ±`tolerance` seconds of each expected beat position.
    /// When `None`, defaults to 10 % of the inter-beat interval.
    pub tolerance: Option<f64>,
    /// Configuration forwarded to the onset detection pipeline.
    pub onset_config: OnsetDetectionConfig,
}

impl BeatTrackingConfig {
    /// Create a new [`BeatTrackingConfig`].
    ///
    /// # Arguments
    /// - `tempo_bpm` – Target tempo in beats per minute; must be > 0.
    /// - `tolerance` – Beat timing tolerance in seconds.  When `None`,
    ///   the tracker defaults to 10 % of the inter-beat interval.
    /// - `onset_config` – Configuration for the onset detection
    ///   pipeline.
    ///
    /// # Returns
    /// A new [`BeatTrackingConfig`].
    #[inline]
    pub fn new(tempo_bpm: f64, tolerance: Option<f64>, onset_config: OnsetDetectionConfig) -> Self {
        Self {
            tempo_bpm,
            tolerance,
            onset_config,
        }
    }
}

impl<'a, T> AudioBeatTracking for AudioSamples<'a, T>
where
    T: StandardSample,
{
    /// Detect beat positions in the audio signal at the target tempo.
    ///
    /// Computes an onset strength envelope from the signal using the
    /// onset detection configuration in `config`, then locates beat
    /// frames by walking forward and backward from the global onset
    /// peak in steps of one inter-beat interval.
    ///
    /// # Arguments
    /// - `config` – Beat tracking configuration: target tempo,
    ///   optional timing tolerance, and onset detection parameters.
    ///
    /// # Returns
    /// A [`BeatTrackingData`] containing the target tempo and the
    /// detected beat timestamps in seconds.  Beat times are in
    /// detection order (global peak first, then forward beats, then
    /// backward beats in reverse); sort `beat_times` for
    /// chronological order.
    ///
    /// # Errors
    /// - [`AudioSampleError::Parameter`] – if `config.tempo_bpm` is
    ///   ≤ 0 or if the inter-beat interval is too small relative to
    ///   the hop size.
    ///
    /// # Examples
    /// ```no_run
    /// use audio_samples::{AudioSamples, sample_rate};
    /// use audio_samples::operations::beat::{BeatTrackingConfig, BeatTrackingData};
    /// use audio_samples::operations::traits::AudioBeatTracking;
    ///
    /// # fn example(audio: AudioSamples<'_, f32>, config: BeatTrackingConfig) {
    /// let result = audio.detect_beats(&config).unwrap();
    /// println!("Tempo: {:.1} BPM", result.tempo_bpm);
    /// for &t in &result.beat_times {
    ///     println!("  beat at {:.3} s", t);
    /// }
    /// # }
    /// ```
    fn detect_beats(&self, config: &BeatTrackingConfig) -> AudioSampleResult<BeatTrackingData> {
        let sr = self.sample_rate_hz();

        // Decide channel strategy explicitly
        let onset = onset_strength_envelope(self, &config.onset_config, None)?;

        let beats = track_beats_core(
            &onset,
            config.tempo_bpm,
            sr,
            config.onset_config.hop_size,
            config.tolerance,
        )?;

        Ok(BeatTrackingData {
            tempo_bpm: config.tempo_bpm,
            beat_times: beats,
            config: config.clone(),
        })
    }
}

/// Compute a smoothed, log-compressed onset strength envelope.
///
/// Runs the onset detection function on `audio`, applies a symmetric
/// moving-average smoothing window of `config.window_size` frames
/// (defaulting to 3 when `None`), then maps each smoothed value
/// through `ln(1 + μ × x)` to compress the dynamic range.
///
/// # Arguments
/// - `audio` – Input audio signal.
/// - `config` – Onset detection configuration.
/// - `log_compression` – Compression factor μ in `ln(1 + μ × x)`.
///   Defaults to `0.5` when `None`.
///
/// # Returns
/// A non-empty vector of onset strength values, one per analysis
/// frame.
///
/// # Errors
/// - Propagates any error from the underlying onset detection
///   function.
pub fn onset_strength_envelope<T>(
    audio: &AudioSamples<'_, T>,
    config: &OnsetDetectionConfig,
    log_compression: Option<f64>,
) -> AudioSampleResult<NonEmptyVec<f64>>
where
    T: StandardSample,
{
    let (_times, odf) = audio.onset_detection_function(config)?;
    let odf = odf.to_vec();
    // Simple moving average smoothing
    let window = config.window_size.unwrap_or(crate::nzu!(3)).get();
    let mut smoothed = vec![0.0; odf.len()];
    for (i, _) in odf.iter().enumerate() {
        let start = i.saturating_sub(window);
        let end: usize = (i + window + 1).min(odf.len());
        let acc: f64 = odf
            .iter()
            .skip(start)
            .take(end - start)
            .fold(0.0, |acc, x| acc + *x);
        smoothed[i] = acc / (end - start) as f64;
    }

    let compression = log_compression.unwrap_or(0.5);

    let env: Vec<f64> = smoothed
        .iter()
        .map(|&x| (1.0 + compression * x).ln())
        .collect();

    // safety: odf is non-empty, so env is non-empty
    let env = unsafe { NonEmptyVec::new_unchecked(env) };
    Ok(env)
}

#[inline(always)]
fn peak_index(slice: &[f64]) -> usize {
    let mut best_i = 0usize;
    let mut best_v = f64::NEG_INFINITY;

    // Manual loop beats iterator chains for branch predictability and inlining
    for i in 0..slice.len() {
        let v = unsafe { *slice.get_unchecked(i) };
        if v > best_v {
            best_v = v;
            best_i = i;
        }
    }

    best_i
}

/// Core beat tracking kernel operating on a pre-computed onset envelope.
///
/// Finds the global peak of `onset`, then walks forward and backward
/// from that peak in steps of one inter-beat interval.  At each
/// expected beat position a local peak search within a tolerance
/// window selects the actual frame.
///
/// Beat times are returned in detection order: the first element is
/// the global onset peak; forward beats follow in causal order;
/// backward beats are appended last in reverse temporal order.
/// Sort the result if chronological order is needed.
///
/// # Arguments
/// - `onset` – Pre-computed onset strength envelope, one value per
///   analysis frame.
/// - `tempo_bpm` – Target tempo in beats per minute; must be > 0.
/// - `sample_rate` – Sample rate of the original audio in Hz.
/// - `hop_size` – Number of audio samples per analysis frame.
/// - `tolerance_seconds` – Half-width of the local search window in
///   seconds.  When `None`, defaults to 10 % of the inter-beat
///   interval (minimum 1 frame).
///
/// # Returns
/// Beat timestamps in seconds, in detection order.
///
/// # Errors
/// - [`AudioSampleError::Parameter`] – if `tempo_bpm` is ≤ 0 or if
///   the inter-beat interval in frames is ≤ 0.
///
/// # Examples
/// ```
/// use audio_samples::operations::beat::track_beats_core;
/// use non_empty_slice::NonEmptyVec;
/// use std::num::NonZeroUsize;
///
/// let onset = NonEmptyVec::new(vec![0.0f64; 100]).unwrap();
/// let beats = track_beats_core(
///     &onset,
///     120.0,
///     44100.0,
///     NonZeroUsize::new(512).unwrap(),
///     None,
/// ).unwrap();
/// assert!(!beats.is_empty());
/// ```
#[inline]
pub fn track_beats_core(
    onset: &NonEmptySlice<f64>,
    tempo_bpm: f64,
    sample_rate: f64,
    hop_size: NonZeroUsize,
    tolerance_seconds: Option<f64>,
) -> AudioSampleResult<Vec<f64>> {
    if tempo_bpm <= 0.0 {
        return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
            "tempo_bpm",
            tempo_bpm,
        )));
    }

    // --- Timing model ---
    let hop_time = hop_size.get() as f64 / sample_rate;
    let ibi_seconds = 60.0 / tempo_bpm;

    let ibi_frames = (ibi_seconds / hop_time).round() as isize;
    if ibi_frames <= 0 {
        return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
            "tempo_bpm",
            "Inter-beat interval too small",
        )));
    }

    let tol_frames = tolerance_seconds
        .map(|t| (t / hop_time).round() as isize)
        .unwrap_or((ibi_frames as f64 * 0.1).round() as isize)
        .max(1);

    let len = onset.len().get() as isize;

    // --- Starting peak ---
    let mut start = 0isize;
    let mut best_v = f64::NEG_INFINITY;
    for i in 0..onset.len().get() {
        let v = unsafe { *onset.get_unchecked(i) };
        if v > best_v {
            best_v = v;
            start = i as isize;
        }
    }

    // Conservative capacity estimate
    let est_beats = (len / ibi_frames).max(1) as usize;
    let mut beat_frames = Vec::with_capacity(est_beats);
    beat_frames.push(start);

    // --- Forward tracking ---
    let mut idx = start;
    while idx + ibi_frames < len {
        let target = idx + ibi_frames;

        let lo = (target - tol_frames).max(0) as usize;
        let hi = (target + tol_frames).min(len) as usize;

        let rel = peak_index(&onset[lo..hi]) as isize;
        idx = lo as isize + rel;
        beat_frames.push(idx);
    }

    // --- Backward tracking ---
    let mut idx = start;
    while idx - ibi_frames > 0 {
        let target = idx - ibi_frames;

        let lo = (target - tol_frames).max(0) as usize;
        let hi = (target + tol_frames).min(len) as usize;

        let rel = peak_index(&onset[lo..hi]) as isize;
        idx = lo as isize + rel;
        beat_frames.push(idx);
    }

    // --- Convert to seconds ---
    let mut times = Vec::with_capacity(beat_frames.len());
    for f in beat_frames {
        times.push(f as f64 * hop_time);
    }

    Ok(times)
}

#[cfg(test)]
mod tests {
    use super::*;
    use non_empty_slice::NonEmptyVec;
    use proptest::prelude::*;

    fn synthetic_onset(len: usize) -> NonEmptyVec<f64> {
        // Simple periodic peaks with noise
        let mut v = vec![0.0; len];
        for i in (0..len).step_by(20.max(1)) {
            v[i] = 1.0;
        }
        let v = NonEmptyVec::new(v).unwrap();
        v
    }

    proptest! {
        #[test]
        fn beat_times_are_finite_and_non_negative(
            len in 64usize..2048,
            tempo in 40.0f64..240.0,
            sr in 8_000.0f64..96_000.0,
            hop in 1usize..2048,
        ) {
            let onset = synthetic_onset(len);
            let hop = NonZeroUsize::new(hop).unwrap();
            let beats = track_beats_core(
                &onset,
                tempo,
                sr,
                hop,
                None,
            ).unwrap();

            for &t in &beats {
                prop_assert!(t.is_finite());
                prop_assert!(t >= 0.0);
            }
        }

        #[test]
        fn beat_times_within_signal_bounds(
            len in 64usize..4096,
            tempo in 40.0f64..240.0,
            sr in 8_000.0f64..96_000.0,
            hop in 1usize..1024,
        ) {
            let onset = synthetic_onset(len);
            let duration = (len as f64 * hop as f64) / sr;
            let hop = NonZeroUsize::new(hop).unwrap();
            let beats = track_beats_core(
                &onset,
                tempo,
                sr,
                hop,
                None,
            ).unwrap();

            for &t in &beats {
                prop_assert!(t <= duration + 1e-6);
            }
        }

        #[test]
        fn first_beat_is_global_peak(
            len in 128usize..2048,
            tempo in 60.0f64..180.0,
            sr in 16_000.0f64..48_000.0,
            hop in 1usize..1024,
        ) {
            let hop = NonZeroUsize::new(hop).unwrap();
            let onset = vec![0.0; len];
            let mut onset = NonEmptyVec::new(onset).unwrap();
            let peak_idx = len / 3;
            onset[peak_idx] = 10.0;

            let beats = track_beats_core(
                &onset,
                tempo,
                sr,
                hop,
                None,
            ).unwrap();

            let first_frame = (beats[0] * sr / hop.get() as f64).round() as usize;
            prop_assert_eq!(first_frame, peak_idx);
        }

        #[test]
        fn insertion_order_preserves_forward_then_backward_structure(
            len in 256usize..4096,
            tempo in 60.0f64..180.0,
            sr in 16_000.0f64..48_000.0,
            hop in 1usize..512,
        ) {
            let onset = synthetic_onset(len);
            let hop = NonZeroUsize::new(hop).unwrap();
            let beats = track_beats_core(
                &onset,
                tempo,
                sr,
                hop,
                None,
            ).unwrap();

            if beats.len() > 1usize && beats.len() >= 3 {
                // First forward step should move forward in time
                prop_assert!(beats[1] >= beats[0] || beats.len() == 1);

                // Last element should be <= first element if backward beats exist
                let last = beats[beats.len() - 1];
                prop_assert!(last <= beats[0] || beats.len() <= 2);
            }
        }
    }
}
