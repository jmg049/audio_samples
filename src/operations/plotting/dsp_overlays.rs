//! DSP feature overlays for plots
//!
//! This module provides functions to compute DSP features over sliding windows
//! for visualization as plot overlays.

use std::num::NonZeroUsize;

use crate::{AudioSamples, StandardSample};

/// Computes RMS (Root Mean Square) amplitude over sliding windows.
///
/// Divides the input signal into overlapping frames and computes the RMS value for each frame.
/// RMS is a measure of signal power and correlates with perceived loudness. Useful for
/// visualizing amplitude envelopes on waveform plots.
///
/// # Arguments
/// * `audio` — Input audio signal (mono or multi-channel)
/// * `window_size` — Window size in samples. Typical values: 2048-8192 for music, 512-2048
///   for speech.
/// * `hop_size` — Hop size (stride) in samples between successive windows. Smaller hop sizes
///   produce smoother curves at the cost of more computation.
///
/// # Returns
/// A tuple `(time_points, rms_values)` where:
/// - `time_points` are the center times of each window in seconds
/// - `rms_values` are the RMS amplitudes (same scale as input samples, typically 0.0-1.0 for
///   normalized audio)
///
/// # Example
/// ```rust,no_run
/// use audio_samples::{AudioSamples, sample_rate, nzu};
/// use audio_samples::operations::plotting::dsp_overlays;
///
/// let audio = AudioSamples::new_mono(ndarray::Array1::from_elem(44100, 0.0f32), sample_rate!(44100))?;
/// let (times, rms) = dsp_overlays::compute_windowed_rms(&audio, nzu!(2048), nzu!(512));
/// # Ok::<(), audio_samples::AudioSampleError>(())
/// ```
#[inline]
#[must_use]
pub fn compute_windowed_rms<'b, T>(
    audio: &'b AudioSamples<'b, T>,
    window_size: NonZeroUsize,
    hop_size: NonZeroUsize,
) -> (Vec<f64>, Vec<f64>)
where
    T: StandardSample,
{
    use crate::operations::traits::AudioStatistics;

    let sample_rate = f64::from(audio.sample_rate().get());
    let mut time_points = Vec::new();
    let mut rms_values = Vec::new();

    for (idx, window) in audio.windows(window_size, hop_size).enumerate() {
        let center_sample = idx * hop_size.get() + window_size.get() / 2;
        let time = center_sample as f64 / sample_rate;
        let rms = window.rms();

        time_points.push(time);
        rms_values.push(rms);
    }

    (time_points, rms_values)
}

/// Computes peak (maximum absolute amplitude) over sliding windows.
///
/// Divides the input signal into overlapping frames and finds the maximum absolute sample
/// value in each frame. Peak values indicate the loudest instantaneous amplitude and are
/// useful for visualizing dynamic range and clipping risk.
///
/// # Arguments
/// * `audio` — Input audio signal (mono or multi-channel)
/// * `window_size` — Window size in samples. Typical values: 2048-8192 for music, 512-2048
///   for speech.
/// * `hop_size` — Hop size (stride) in samples between successive windows. Smaller hop sizes
///   produce smoother curves at the cost of more computation.
///
/// # Returns
/// A tuple `(time_points, peak_values)` where:
/// - `time_points` are the center times of each window in seconds
/// - `peak_values` are the maximum absolute amplitudes (same scale as input samples, typically
///   0.0-1.0 for normalized audio)
///
/// # Example
/// ```rust,no_run
/// use audio_samples::{AudioSamples, sample_rate, nzu};
/// use audio_samples::operations::plotting::dsp_overlays;
///
/// let audio = AudioSamples::new_mono(ndarray::Array1::from_elem(44100, 0.0f32), sample_rate!(44100))?;
/// let (times, peaks) = dsp_overlays::compute_windowed_peak(&audio, nzu!(2048), nzu!(512));
/// # Ok::<(), audio_samples::AudioSampleError>(())
/// ```
#[inline]
#[must_use]
pub fn compute_windowed_peak<'a, 'iter, T>(
    audio: &'iter AudioSamples<'a, T>,
    window_size: NonZeroUsize,
    hop_size: NonZeroUsize,
) -> (Vec<f64>, Vec<f64>)
where
    T: StandardSample,
    'iter: 'a,
{
    use crate::operations::traits::AudioStatistics;

    let sample_rate = f64::from(audio.sample_rate().get());
    let mut time_points = Vec::new();
    let mut peak_values = Vec::new();

    for (idx, window) in audio.windows(window_size, hop_size).enumerate() {
        let center_sample = idx * hop_size.get() + window_size.get() / 2;
        let time = center_sample as f64 / sample_rate;
        let peak = window.peak();

        time_points.push(time);
        peak_values.push(peak.to_f64().unwrap_or(0.0));
    }

    (time_points, peak_values)
}

/// Computes zero-crossing rate (ZCR) over sliding windows.
///
/// Divides the input signal into overlapping frames and counts how many times the signal
/// crosses the zero amplitude line in each frame. ZCR is a simple measure of signal noisiness
/// and spectral content: higher ZCR correlates with noisier, more broadband signals (e.g.,
/// fricatives in speech), while lower ZCR indicates tonal, pitched content.
///
/// # Arguments
/// * `audio` — Input audio signal (mono or multi-channel)
/// * `window_size` — Window size in samples. Typical values: 2048-8192 for music, 512-2048
///   for speech.
/// * `hop_size` — Hop size (stride) in samples between successive windows. Smaller hop sizes
///   produce smoother curves at the cost of more computation.
///
/// # Returns
/// A tuple `(time_points, zcr_values)` where:
/// - `time_points` are the center times of each window in seconds
/// - `zcr_values` are the zero-crossing rates (counts per sample, typically 0.0-0.5)
///
/// # Example
/// ```rust,no_run
/// use audio_samples::{AudioSamples, sample_rate, nzu};
/// use audio_samples::operations::plotting::dsp_overlays;
///
/// let audio = AudioSamples::new_mono(ndarray::Array1::from_elem(44100, 0.0f32), sample_rate!(44100))?;
/// let (times, zcr) = dsp_overlays::compute_windowed_zcr(&audio, nzu!(2048), nzu!(512));
/// # Ok::<(), audio_samples::AudioSampleError>(())
/// ```
#[inline]
#[must_use]
pub fn compute_windowed_zcr<'a, T>(
    audio: &'a AudioSamples<'a, T>,
    window_size: NonZeroUsize,
    hop_size: NonZeroUsize,
) -> (Vec<f64>, Vec<f64>)
where
    T: StandardSample,
{
    use crate::operations::traits::AudioStatistics;

    let sample_rate = f64::from(audio.sample_rate().get());
    let mut time_points = Vec::new();
    let mut zcr_values = Vec::new();

    for (idx, window) in audio.windows(window_size, hop_size).enumerate() {
        let center_sample = idx * hop_size.get() + window_size.get() / 2;
        let time = center_sample as f64 / sample_rate;
        let zcr = window.zero_crossing_rate();

        time_points.push(time);
        zcr_values.push(zcr);
    }

    (time_points, zcr_values)
}

/// Computes signal energy over sliding windows.
///
/// Divides the input signal into overlapping frames and computes the energy (RMS squared,
/// proportional to power) for each frame. Energy is a measure of signal intensity and is
/// useful for detecting voiced vs. unvoiced segments, onset detection, and dynamic analysis.
///
/// # Arguments
/// * `audio` — Input audio signal (mono or multi-channel)
/// * `window_size` — Window size in samples. Typical values: 2048-8192 for music, 512-2048
///   for speech.
/// * `hop_size` — Hop size (stride) in samples between successive windows. Smaller hop sizes
///   produce smoother curves at the cost of more computation.
///
/// # Returns
/// A tuple `(time_points, energy_values)` where:
/// - `time_points` are the center times of each window in seconds
/// - `energy_values` are the squared RMS values (power, typically 0.0-1.0 for normalized audio)
///
/// # Example
/// ```rust,no_run
/// use audio_samples::{AudioSamples, sample_rate, nzu};
/// use audio_samples::operations::plotting::dsp_overlays;
///
/// let audio = AudioSamples::new_mono(ndarray::Array1::from_elem(44100, 0.0f32), sample_rate!(44100))?;
/// let (times, energy) = dsp_overlays::compute_windowed_energy(&audio, nzu!(2048), nzu!(512));
/// # Ok::<(), audio_samples::AudioSampleError>(())
/// ```
#[inline]
#[must_use]
pub fn compute_windowed_energy<'a, T>(
    audio: &'a AudioSamples<'a, T>,
    window_size: NonZeroUsize,
    hop_size: NonZeroUsize,
) -> (Vec<f64>, Vec<f64>)
where
    T: StandardSample,
{
    use crate::operations::traits::AudioStatistics;

    let sample_rate = f64::from(audio.sample_rate().get());
    let mut time_points = Vec::new();
    let mut energy_values = Vec::new();

    for (idx, window) in audio.windows(window_size, hop_size).enumerate() {
        let center_sample = idx * hop_size.get() + window_size.get() / 2;
        let time = center_sample as f64 / sample_rate;

        // Compute energy as RMS squared (proportional to power)
        let rms = window.rms();
        let energy = rms * rms;

        time_points.push(time);
        energy_values.push(energy);
    }

    (time_points, energy_values)
}

/// Computes spectral centroid over sliding windows.
///
/// Divides the input signal into overlapping frames, computes the FFT for each frame, and
/// calculates the spectral centroid (the "center of mass" of the spectrum). The centroid
/// correlates with perceived brightness: higher centroid values indicate more high-frequency
/// content, while lower values indicate more bass-heavy signals.
///
/// # Arguments
/// * `audio` — Input audio signal (mono or multi-channel)
/// * `window_size` — Window size in samples. Must be a power of 2 for FFT efficiency.
///   Typical values: 2048-8192.
/// * `hop_size` — Hop size (stride) in samples between successive windows. Smaller hop sizes
///   produce smoother curves at the cost of more computation.
///
/// # Returns
/// A tuple `(time_points, centroid_values)` where:
/// - `time_points` are the center times of each window in seconds
/// - `centroid_values` are the spectral centroid frequencies in Hz
///
/// Windows where the spectral centroid cannot be computed (e.g., silent frames) are skipped.
///
/// # Example
/// ```rust,no_run
/// use audio_samples::{AudioSamples, sample_rate, nzu};
/// use audio_samples::operations::plotting::dsp_overlays;
///
/// let audio = AudioSamples::new_mono(ndarray::Array1::from_elem(44100, 0.0f32), sample_rate!(44100))?;
/// let (times, centroids) = dsp_overlays::compute_windowed_spectral_centroid(
///     &audio, nzu!(2048), nzu!(512)
/// );
/// # Ok::<(), audio_samples::AudioSampleError>(())
/// ```
#[inline]
#[must_use]
pub fn compute_windowed_spectral_centroid<'a, T>(
    audio: &'a AudioSamples<'a, T>,
    window_size: NonZeroUsize,
    hop_size: NonZeroUsize,
) -> (Vec<f64>, Vec<f64>)
where
    T: StandardSample,
{
    use crate::operations::traits::AudioStatistics;

    let sample_rate = audio.sample_rate_hz();
    let mut time_points = Vec::new();
    let mut centroid_values = Vec::new();

    for (idx, window) in audio.windows(window_size, hop_size).enumerate() {
        let center_sample = idx * hop_size.get() + window_size.get() / 2;
        let time = center_sample as f64 / sample_rate;

        // Compute spectral centroid for this window
        if let Ok(centroid) = window.spectral_centroid() {
            time_points.push(time);
            centroid_values.push(centroid);
        }
    }

    (time_points, centroid_values)
}

/// Computes spectral rolloff frequency over sliding windows.
///
/// Divides the input signal into overlapping frames, computes the FFT for each frame, and
/// calculates the rolloff frequency: the frequency below which a specified percentage (e.g., 85%)
/// of the spectral energy is contained. Rolloff is useful for distinguishing tonal vs. noisy
/// content and for detecting the high-frequency "cutoff" of the signal.
///
/// # Arguments
/// * `audio` — Input audio signal (mono or multi-channel)
/// * `window_size` — Window size in samples. Must be a power of 2 for FFT efficiency.
///   Typical values: 2048-8192.
/// * `hop_size` — Hop size (stride) in samples between successive windows. Smaller hop sizes
///   produce smoother curves at the cost of more computation.
/// * `rolloff_percent` — Rolloff threshold as a fraction (e.g., 0.85 for 85%). Typical values:
///   0.85-0.95.
///
/// # Returns
/// A tuple `(time_points, rolloff_values)` where:
/// - `time_points` are the center times of each window in seconds
/// - `rolloff_values` are the rolloff frequencies in Hz
///
/// Windows where the spectral rolloff cannot be computed (e.g., silent frames) are skipped.
///
/// # Example
/// ```rust,no_run
/// use audio_samples::{AudioSamples, sample_rate, nzu};
/// use audio_samples::operations::plotting::dsp_overlays;
///
/// let audio = AudioSamples::new_mono(ndarray::Array1::from_elem(44100, 0.0f32), sample_rate!(44100))?;
/// let (times, rolloff) = dsp_overlays::compute_windowed_spectral_rolloff(
///     &audio, nzu!(2048), nzu!(512), 0.85
/// );
/// # Ok::<(), audio_samples::AudioSampleError>(())
/// ```
#[inline]
#[must_use]
pub fn compute_windowed_spectral_rolloff<'a, 'iter, T>(
    audio: &'iter AudioSamples<'a, T>,
    window_size: NonZeroUsize,
    hop_size: NonZeroUsize,
    rolloff_percent: f64,
) -> (Vec<f64>, Vec<f64>)
where
    T: StandardSample,
    'iter: 'a,
{
    use crate::operations::traits::AudioStatistics;

    let sample_rate = audio.sample_rate_hz();
    let mut time_points = Vec::new();
    let mut rolloff_values = Vec::new();

    for (idx, window) in audio.windows(window_size, hop_size).enumerate() {
        let center_sample = idx * hop_size.get() + window_size.get() / 2;
        let time = center_sample as f64 / sample_rate;

        // Compute spectral rolloff for this window
        if let Ok(rolloff) = window.spectral_rolloff(rolloff_percent) {
            time_points.push(time);
            rolloff_values.push(rolloff);
        }
    }

    (time_points, rolloff_values)
}
