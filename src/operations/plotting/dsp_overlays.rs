//! DSP feature overlays for plots
//!
//! This module provides functions to compute DSP features over sliding windows
//! for visualization as plot overlays.

use std::num::NonZeroUsize;

use crate::{AudioSamples, StandardSample};

/// Compute RMS (Root Mean Square) over sliding windows
///
/// # Arguments
/// * `audio` - Input audio
/// * `window_size` - Window size in samples
/// * `hop_size` - Hop size in samples
///
/// # Returns
/// Tuple of (time_points, rms_values) suitable for plotting
pub fn compute_windowed_rms<'b, T>(
    audio: &'b AudioSamples<'b, T>,
    window_size: usize,
    hop_size: usize,
) -> (Vec<f64>, Vec<f64>)
where
    T: StandardSample,
{
    use crate::operations::traits::AudioStatistics;
    use std::num::NonZeroUsize;

    let sample_rate = audio.sample_rate().get() as f64;
    let mut time_points = Vec::new();
    let mut rms_values = Vec::new();

    let win_size = NonZeroUsize::new(window_size).unwrap();
    let hop = NonZeroUsize::new(hop_size).unwrap();

    for (idx, window) in audio.windows(win_size, hop).enumerate() {
        let center_sample = idx * hop_size + window_size / 2;
        let time = center_sample as f64 / sample_rate;
        let rms = window.rms();

        time_points.push(time);
        rms_values.push(rms);
    }

    (time_points, rms_values)
}

/// Compute peak values over sliding windows
pub fn compute_windowed_peak<'a, 'iter, T>(
    audio: &'iter AudioSamples<'a, T>,
    window_size: usize,
    hop_size: usize,
) -> (Vec<f64>, Vec<f64>)
where
    T: StandardSample,
    'iter: 'a,
{
    use crate::operations::traits::AudioStatistics;
    use std::num::NonZeroUsize;

    let sample_rate = audio.sample_rate().get() as f64;
    let mut time_points = Vec::new();
    let mut peak_values = Vec::new();

    let win_size = NonZeroUsize::new(window_size).unwrap();
    let hop = NonZeroUsize::new(hop_size).unwrap();

    for (idx, window) in audio.windows(win_size, hop).enumerate() {
        let center_sample = idx * hop_size + window_size / 2;
        let time = center_sample as f64 / sample_rate;
        let peak = window.peak();

        time_points.push(time);
        peak_values.push(peak.to_f64().unwrap_or(0.0));
    }

    (time_points, peak_values)
}

/// Compute zero-crossing rate over sliding windows
pub fn compute_windowed_zcr<'a, T>(
    audio: &'a AudioSamples<'a, T>,
    window_size: usize,
    hop_size: usize,
) -> (Vec<f64>, Vec<f64>)
where
    T: StandardSample,
{
    use crate::operations::traits::AudioStatistics;
    use std::num::NonZeroUsize;

    let sample_rate = audio.sample_rate().get() as f64;
    let mut time_points = Vec::new();
    let mut zcr_values = Vec::new();

    let win_size = NonZeroUsize::new(window_size).unwrap();
    let hop = NonZeroUsize::new(hop_size).unwrap();

    for (idx, window) in audio.windows(win_size, hop).enumerate() {
        let center_sample = idx * hop_size + window_size / 2;
        let time = center_sample as f64 / sample_rate;
        let zcr = window.zero_crossing_rate();

        time_points.push(time);
        zcr_values.push(zcr);
    }

    (time_points, zcr_values)
}

/// Compute energy over sliding windows
pub fn compute_windowed_energy<'a, T>(
    audio: &'a AudioSamples<'a, T>,
    window_size: usize,
    hop_size: usize,
) -> (Vec<f64>, Vec<f64>)
where
    T: StandardSample,
{
    use crate::operations::traits::AudioStatistics;
    use std::num::NonZeroUsize;

    let sample_rate = audio.sample_rate().get() as f64;
    let mut time_points = Vec::new();
    let mut energy_values = Vec::new();

    let win_size = NonZeroUsize::new(window_size).unwrap();
    let hop = NonZeroUsize::new(hop_size).unwrap();

    for (idx, window) in audio.windows(win_size, hop).enumerate() {
        let center_sample = idx * hop_size + window_size / 2;
        let time = center_sample as f64 / sample_rate;

        // Compute energy as RMS squared (proportional to power)
        let rms = window.rms();
        let energy = rms * rms;

        time_points.push(time);
        energy_values.push(energy);
    }

    (time_points, energy_values)
}

/// Compute spectral centroid over sliding windows
///
/// # Arguments
/// * `audio` - Input audio
/// * `window_size` - Window size in samples
/// * `hop_size` - Hop size in samples
///
/// # Returns
/// Tuple of (time_points, centroid_values) where centroid is in Hz
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

/// Compute spectral rolloff over sliding windows
///
/// # Arguments
/// * `audio` - Input audio
/// * `window_size` - Window size in samples
/// * `hop_size` - Hop size in samples
/// * `rolloff_percent` - Rolloff percentage (typically 0.85 for 85%)
///
/// # Returns
/// Tuple of (time_points, rolloff_values) where rolloff is in Hz
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
