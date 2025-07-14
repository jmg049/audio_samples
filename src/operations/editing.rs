//! Time-domain editing operations for AudioSamples.
//!
//! This module implements the AudioEditing trait, providing comprehensive
//! time-domain audio editing operations including cutting, pasting, mixing,
//! and envelope operations using efficient ndarray operations.

use super::traits::AudioEditing;
use super::types::FadeCurve;
use crate::repr::AudioData;
use crate::{AudioSample, AudioSampleError, AudioSampleResult, AudioSamples};
use ndarray::{Array1, Array2, Axis, concatenate, s};
use num_traits::{Float, FromPrimitive, ToPrimitive, Zero};

/// Helper function to convert seconds to samples
fn seconds_to_samples(seconds: f64, sample_rate: u32) -> usize {
    (seconds * sample_rate as f64) as usize
}

/// Validates time bounds for trim operations
fn validate_time_bounds(start: f64, end: f64, duration: f64) -> AudioSampleResult<()> {
    if start < 0.0 {
        return Err(AudioSampleError::InvalidParameter(format!(
            "Start time cannot be negative: {}",
            start
        )));
    }
    if end <= start {
        return Err(AudioSampleError::InvalidParameter(format!(
            "End time ({}) must be greater than start time ({})",
            end, start
        )));
    }
    if end > duration {
        return Err(AudioSampleError::InvalidParameter(format!(
            "End time ({}) exceeds audio duration ({})",
            end, duration
        )));
    }
    Ok(())
}

/// Applies fade curve transformation to a position value [0.0, 1.0]
fn apply_fade_curve(curve: &FadeCurve, position: f64) -> f64 {
    match curve {
        FadeCurve::Linear => position,
        FadeCurve::Exponential => position * position,
        FadeCurve::Logarithmic => {
            if position <= 0.0 {
                0.0
            } else {
                (1.0 + position).ln() / 2.0_f64.ln()
            }
        }
        FadeCurve::SmoothStep => position * position * (3.0 - 2.0 * position),
        FadeCurve::Custom(func) => func(position),
    }
}

impl<T: AudioSample + ToPrimitive + FromPrimitive + Zero + Float + Copy> AudioEditing<T>
    for AudioSamples<T>
{
    /// Reverses the order of audio samples.
    fn reverse(&self) -> Self
    where
        Self: Sized,
    {
        match &self.data {
            AudioData::Mono(arr) => {
                // Reverse the 1D array using ndarray's reverse slicing
                let reversed = arr.slice(s![..;-1]).to_owned();
                AudioSamples::new_mono(reversed, self.sample_rate())
            }
            AudioData::MultiChannel(arr) => {
                // Reverse along the time axis (axis 1)
                let reversed = arr.slice(s![.., ..;-1]).to_owned();
                AudioSamples::new_multi_channel(reversed, self.sample_rate())
            }
        }
    }

    /// Reverses the order of audio samples in place.
    /// This method modifies the original audio samples.
    fn reverse_in_place(&mut self) -> AudioSampleResult<()>
    where
        Self: Sized,
    {
        match &mut self.data {
            AudioData::Mono(arr) => {
                // Reverse the 1D array in place
                arr.swap_axes(0, 0); // No-op, just to indicate in-place operation
                arr.as_slice_mut().unwrap().reverse();
            }
            AudioData::MultiChannel(arr) => {
                // Reverse along the time axis (axis 1)
                for mut channel in arr.axis_iter_mut(Axis(0)) {
                    channel.as_slice_mut().unwrap().reverse();
                }
            }
        }
        Ok(())
    }

    /// Extracts a segment of audio between start and end times.
    fn trim(&self, start_seconds: f64, end_seconds: f64) -> AudioSampleResult<Self>
    where
        Self: Sized,
    {
        let duration = self.duration_seconds();
        validate_time_bounds(start_seconds, end_seconds, duration)?;

        let start_sample = seconds_to_samples(start_seconds, self.sample_rate());
        let end_sample = seconds_to_samples(end_seconds, self.sample_rate());

        match &self.data {
            AudioData::Mono(arr) => {
                if end_sample > arr.len() {
                    return Err(AudioSampleError::InvalidParameter(format!(
                        "End sample {} exceeds audio length {}",
                        end_sample,
                        arr.len()
                    )));
                }
                let trimmed = arr.slice(s![start_sample..end_sample]).to_owned();
                Ok(AudioSamples::new_mono(trimmed, self.sample_rate()))
            }
            AudioData::MultiChannel(arr) => {
                if end_sample > arr.ncols() {
                    return Err(AudioSampleError::InvalidParameter(format!(
                        "End sample {} exceeds audio length {}",
                        end_sample,
                        arr.ncols()
                    )));
                }
                let trimmed = arr.slice(s![.., start_sample..end_sample]).to_owned();
                Ok(AudioSamples::new_multi_channel(trimmed, self.sample_rate()))
            }
        }
    }

    /// Adds padding/silence to the beginning and/or end of the audio.
    fn pad(
        &self,
        pad_start_seconds: f64,
        pad_end_seconds: f64,
        pad_value: T,
    ) -> AudioSampleResult<Self>
    where
        Self: Sized,
    {
        if pad_start_seconds < 0.0 || pad_end_seconds < 0.0 {
            return Err(AudioSampleError::InvalidParameter(
                "Padding durations cannot be negative".to_string(),
            ));
        }

        let start_samples = seconds_to_samples(pad_start_seconds, self.sample_rate());
        let end_samples = seconds_to_samples(pad_end_seconds, self.sample_rate());

        match &self.data {
            AudioData::Mono(arr) => {
                let total_length = start_samples + arr.len() + end_samples;
                let mut padded = Array1::from_elem(total_length, pad_value);

                // Copy original data to the middle
                padded
                    .slice_mut(s![start_samples..start_samples + arr.len()])
                    .assign(arr);

                Ok(AudioSamples::new_mono(padded, self.sample_rate()))
            }
            AudioData::MultiChannel(arr) => {
                let total_length = start_samples + arr.ncols() + end_samples;
                let mut padded = Array2::from_elem((arr.nrows(), total_length), pad_value);

                // Copy original data to the middle
                padded
                    .slice_mut(s![.., start_samples..start_samples + arr.ncols()])
                    .assign(arr);

                Ok(AudioSamples::new_multi_channel(padded, self.sample_rate()))
            }
        }
    }

    /// Splits audio into segments of specified duration.
    fn split(&self, segment_duration_seconds: f64) -> AudioSampleResult<Vec<Self>>
    where
        Self: Sized,
    {
        if segment_duration_seconds <= 0.0 {
            return Err(AudioSampleError::InvalidParameter(
                "Segment duration must be positive".to_string(),
            ));
        }

        let segment_samples = seconds_to_samples(segment_duration_seconds, self.sample_rate());
        let total_samples = self.samples_per_channel();

        if segment_samples > total_samples {
            return Err(AudioSampleError::InvalidParameter(
                "Segment duration exceeds audio length".to_string(),
            ));
        }

        let mut segments = Vec::new();
        let mut start = 0;

        while start < total_samples {
            let end = (start + segment_samples).min(total_samples);

            match &self.data {
                AudioData::Mono(arr) => {
                    let segment = arr.slice(s![start..end]).to_owned();
                    segments.push(AudioSamples::new_mono(segment, self.sample_rate()));
                }
                AudioData::MultiChannel(arr) => {
                    let segment = arr.slice(s![.., start..end]).to_owned();
                    segments.push(AudioSamples::new_multi_channel(segment, self.sample_rate()));
                }
            }

            start += segment_samples;
        }

        Ok(segments)
    }

    /// Concatenates multiple audio segments into one.
    fn concatenate(segments: &[Self]) -> AudioSampleResult<Self>
    where
        Self: Sized,
    {
        if segments.is_empty() {
            return Err(AudioSampleError::InvalidParameter(
                "Cannot concatenate empty segment list".to_string(),
            ));
        }

        // Validate all segments have the same sample rate and channel count
        let first = &segments[0];
        for segment in segments.iter().skip(1) {
            if segment.sample_rate() != first.sample_rate() {
                return Err(AudioSampleError::InvalidParameter(
                    "All segments must have the same sample rate".to_string(),
                ));
            }
            if segment.channels() != first.channels() {
                return Err(AudioSampleError::InvalidParameter(
                    "All segments must have the same number of channels".to_string(),
                ));
            }
        }

        match &first.data {
            AudioData::Mono(_) => {
                let arrays: Vec<_> = segments
                    .iter()
                    .map(|s| {
                        if let AudioData::Mono(arr) = &s.data {
                            arr.view()
                        } else {
                            unreachable!()
                        }
                    })
                    .collect();

                let concatenated = concatenate(Axis(0), &arrays).map_err(|e| {
                    AudioSampleError::InvalidParameter(format!("Concatenation failed: {}", e))
                })?;

                Ok(AudioSamples::new_mono(concatenated, first.sample_rate()))
            }
            AudioData::MultiChannel(_) => {
                let arrays: Vec<_> = segments
                    .iter()
                    .map(|s| {
                        if let AudioData::MultiChannel(arr) = &s.data {
                            arr.view()
                        } else {
                            unreachable!()
                        }
                    })
                    .collect();

                let concatenated = concatenate(Axis(1), &arrays).map_err(|e| {
                    AudioSampleError::InvalidParameter(format!("Concatenation failed: {}", e))
                })?;

                Ok(AudioSamples::new_multi_channel(
                    concatenated,
                    first.sample_rate(),
                ))
            }
        }
    }

    /// Mixes multiple audio sources together.
    fn mix(sources: &[Self], weights: Option<&[f64]>) -> AudioSampleResult<Self>
    where
        Self: Sized,
    {
        if sources.is_empty() {
            return Err(AudioSampleError::InvalidParameter(
                "Cannot mix empty source list".to_string(),
            ));
        }

        // Validate all sources have the same properties
        let first = &sources[0];
        for source in sources.iter().skip(1) {
            if source.sample_rate() != first.sample_rate() {
                return Err(AudioSampleError::InvalidParameter(
                    "All sources must have the same sample rate".to_string(),
                ));
            }
            if source.channels() != first.channels() {
                return Err(AudioSampleError::InvalidParameter(
                    "All sources must have the same number of channels".to_string(),
                ));
            }
            if source.samples_per_channel() != first.samples_per_channel() {
                return Err(AudioSampleError::InvalidParameter(
                    "All sources must have the same length".to_string(),
                ));
            }
        }

        // Validate weights if provided
        let mix_weights = if let Some(w) = weights {
            if w.len() != sources.len() {
                return Err(AudioSampleError::InvalidParameter(
                    "Number of weights must match number of sources".to_string(),
                ));
            }
            w
        } else {
            // Equal weights
            &vec![1.0 / sources.len() as f64; sources.len()]
        };

        match &first.data {
            AudioData::Mono(_) => {
                let mut result = first.clone();
                if let AudioData::Mono(result_arr) = &mut result.data {
                    // Start with first source * weight
                    if let AudioData::Mono(_first_arr) = &first.data {
                        let weight = T::from(mix_weights[0]).unwrap_or(T::zero());
                        result_arr.mapv_inplace(|x| x * weight);
                    }

                    // Add remaining sources
                    for (i, source) in sources.iter().skip(1).enumerate() {
                        if let AudioData::Mono(source_arr) = &source.data {
                            let weight = T::from(mix_weights[i + 1]).unwrap_or(T::zero());
                            for (r, s) in result_arr.iter_mut().zip(source_arr.iter()) {
                                *r = *r + *s * weight;
                            }
                        }
                    }
                }
                Ok(result)
            }
            AudioData::MultiChannel(_) => {
                let mut result = first.clone();
                if let AudioData::MultiChannel(result_arr) = &mut result.data {
                    // Start with first source * weight
                    if let AudioData::MultiChannel(_first_arr) = &first.data {
                        let weight = T::from(mix_weights[0]).unwrap_or(T::zero());
                        result_arr.mapv_inplace(|x| x * weight);
                    }

                    // Add remaining sources
                    for (i, source) in sources.iter().skip(1).enumerate() {
                        if let AudioData::MultiChannel(source_arr) = &source.data {
                            let weight = T::from(mix_weights[i + 1]).unwrap_or(T::zero());
                            for (r, s) in result_arr.iter_mut().zip(source_arr.iter()) {
                                *r = *r + *s * weight;
                            }
                        }
                    }
                }
                Ok(result)
            }
        }
    }

    /// Applies fade-in envelope over specified duration.
    fn fade_in(&mut self, duration_seconds: f64, curve: FadeCurve) -> AudioSampleResult<()> {
        if duration_seconds <= 0.0 {
            return Err(AudioSampleError::InvalidParameter(
                "Fade duration must be positive".to_string(),
            ));
        }

        let fade_samples = seconds_to_samples(duration_seconds, self.sample_rate());
        let total_samples = self.samples_per_channel();
        let actual_fade_samples = fade_samples.min(total_samples);

        match &mut self.data {
            AudioData::Mono(arr) => {
                for i in 0..actual_fade_samples {
                    let position = i as f64 / actual_fade_samples as f64;
                    let gain = apply_fade_curve(&curve, position);
                    let gain_t = T::from(gain).unwrap_or(T::zero());
                    arr[i] = arr[i] * gain_t;
                }
            }
            AudioData::MultiChannel(arr) => {
                for i in 0..actual_fade_samples {
                    let position = i as f64 / actual_fade_samples as f64;
                    let gain = apply_fade_curve(&curve, position);
                    let gain_t = T::from(gain).unwrap_or(T::zero());
                    for channel in 0..arr.nrows() {
                        arr[[channel, i]] = arr[[channel, i]] * gain_t;
                    }
                }
            }
        }

        Ok(())
    }

    /// Applies fade-out envelope over specified duration.
    fn fade_out(&mut self, duration_seconds: f64, curve: FadeCurve) -> AudioSampleResult<()> {
        if duration_seconds <= 0.0 {
            return Err(AudioSampleError::InvalidParameter(
                "Fade duration must be positive".to_string(),
            ));
        }

        let fade_samples = seconds_to_samples(duration_seconds, self.sample_rate());
        let total_samples = self.samples_per_channel();
        let actual_fade_samples = fade_samples.min(total_samples);
        let start_sample = total_samples - actual_fade_samples;

        match &mut self.data {
            AudioData::Mono(arr) => {
                for i in 0..actual_fade_samples {
                    let position = 1.0 - (i as f64 / actual_fade_samples as f64);
                    let gain = apply_fade_curve(&curve, position);
                    let gain_t = T::from(gain).unwrap_or(T::zero());
                    arr[start_sample + i] = arr[start_sample + i] * gain_t;
                }
            }
            AudioData::MultiChannel(arr) => {
                for i in 0..actual_fade_samples {
                    let position = 1.0 - (i as f64 / actual_fade_samples as f64);
                    let gain = apply_fade_curve(&curve, position);
                    let gain_t = T::from(gain).unwrap_or(T::zero());
                    for channel in 0..arr.nrows() {
                        arr[[channel, start_sample + i]] =
                            arr[[channel, start_sample + i]] * gain_t;
                    }
                }
            }
        }

        Ok(())
    }

    /// Repeats the audio signal a specified number of times.
    fn repeat(&self, count: usize) -> AudioSampleResult<Self>
    where
        Self: Sized,
    {
        if count == 0 {
            return Err(AudioSampleError::InvalidParameter(
                "Repeat count must be greater than 0".to_string(),
            ));
        }

        if count == 1 {
            return Ok(self.clone());
        }

        match &self.data {
            AudioData::Mono(arr) => {
                let mut repeated = Array1::zeros(arr.len() * count);
                for i in 0..count {
                    let start = i * arr.len();
                    let end = start + arr.len();
                    repeated.slice_mut(s![start..end]).assign(arr);
                }
                Ok(AudioSamples::new_mono(repeated, self.sample_rate()))
            }
            AudioData::MultiChannel(arr) => {
                let mut repeated = Array2::zeros((arr.nrows(), arr.ncols() * count));
                for i in 0..count {
                    let start = i * arr.ncols();
                    let end = start + arr.ncols();
                    repeated.slice_mut(s![.., start..end]).assign(arr);
                }
                Ok(AudioSamples::new_multi_channel(
                    repeated,
                    self.sample_rate(),
                ))
            }
        }
    }

    /// Crops audio to remove silence from beginning and end.
    fn trim_silence(&self, threshold: T) -> AudioSampleResult<Self>
    where
        Self: Sized,
    {
        match &self.data {
            AudioData::Mono(arr) => {
                // Find first non-silent sample
                let start = arr.iter().position(|&x| x.abs() > threshold).unwrap_or(0);

                // Find last non-silent sample
                let end = arr
                    .iter()
                    .rposition(|&x| x.abs() > threshold)
                    .map(|pos| pos + 1)
                    .unwrap_or(arr.len());

                if start >= end {
                    // All samples are below threshold, return minimal audio
                    let minimal = Array1::from_elem(1, T::zero());
                    return Ok(AudioSamples::new_mono(minimal, self.sample_rate()));
                }

                let trimmed = arr.slice(s![start..end]).to_owned();
                Ok(AudioSamples::new_mono(trimmed, self.sample_rate()))
            }
            AudioData::MultiChannel(arr) => {
                // Find first non-silent frame (any channel above threshold)
                let start = (0..arr.ncols())
                    .find(|&col| arr.column(col).iter().any(|&x| x.abs() > threshold))
                    .unwrap_or(0);

                // Find last non-silent frame
                let end = (0..arr.ncols())
                    .rev()
                    .find(|&col| arr.column(col).iter().any(|&x| x.abs() > threshold))
                    .map(|col| col + 1)
                    .unwrap_or(arr.ncols());

                if start >= end {
                    // All samples are below threshold, return minimal audio
                    let minimal = Array2::from_elem((arr.nrows(), 1), T::zero());
                    return Ok(AudioSamples::new_multi_channel(minimal, self.sample_rate()));
                }

                let trimmed = arr.slice(s![.., start..end]).to_owned();
                Ok(AudioSamples::new_multi_channel(trimmed, self.sample_rate()))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::AudioSamples;
    use ndarray::Array1;
    // use approx::assert_abs_diff_eq;

    #[test]
    fn test_reverse_mono_audio() {
        let samples = Array1::from(vec![1.0f32, 2.0, 3.0, 4.0, 5.0]);
        let audio = AudioSamples::new_mono(samples, 44100);

        let reversed = audio.reverse();

        if let AudioData::Mono(arr) = &reversed.data {
            let expected = vec![5.0, 4.0, 3.0, 2.0, 1.0];
            // Convert to vec for comparison since array might not be contiguous
            let actual: Vec<f32> = arr.iter().cloned().collect();
            assert_eq!(actual, expected);
        } else {
            panic!("Expected mono data");
        }
    }

    #[test]
    fn test_trim_with_time_bounds() {
        let samples = Array1::from(vec![1.0f32; 44100]); // 1 second at 44.1kHz
        let audio = AudioSamples::new_mono(samples, 44100);

        let trimmed = audio.trim(0.25, 0.75).unwrap();

        // Should be 0.5 seconds = 22050 samples
        assert_eq!(trimmed.samples_per_channel(), 22050);
    }

    #[test]
    fn test_pad_with_silence() {
        let samples = Array1::from(vec![1.0f32; 1000]);
        let audio = AudioSamples::new_mono(samples, 44100);

        let padded = audio.pad(0.1, 0.1, 0.0).unwrap(); // 0.1s = 4410 samples each side

        assert_eq!(padded.samples_per_channel(), 1000 + 4410 + 4410);

        if let AudioData::Mono(arr) = &padded.data {
            // Check padding is zeros
            assert_eq!(arr[0], 0.0);
            assert_eq!(arr[arr.len() - 1], 0.0);
            // Check original data is preserved
            assert_eq!(arr[4410], 1.0);
        }
    }

    #[test]
    fn test_split_into_segments() {
        let samples = Array1::from(vec![1.0f32; 8820]); // 0.2 seconds at 44.1kHz
        let audio = AudioSamples::new_mono(samples, 44100);

        let segments = audio.split(0.05).unwrap(); // Split into 0.05s segments

        assert_eq!(segments.len(), 4); // 0.2s / 0.05s = 4 segments
        assert_eq!(segments[0].samples_per_channel(), 2205); // 0.05s * 44100
    }

    #[test]
    fn test_concatenate_segments() {
        let samples1 = Array1::from(vec![1.0f32; 1000]);
        let samples2 = Array1::from(vec![2.0f32; 1000]);
        let audio1 = AudioSamples::new_mono(samples1, 44100);
        let audio2 = AudioSamples::new_mono(samples2, 44100);

        let concatenated = AudioSamples::concatenate(&[audio1, audio2]).unwrap();

        assert_eq!(concatenated.samples_per_channel(), 2000);

        if let AudioData::Mono(arr) = &concatenated.data {
            assert_eq!(arr[500], 1.0); // First segment
            assert_eq!(arr[1500], 2.0); // Second segment
        }
    }

    #[test]
    fn test_mix_two_sources() {
        let samples1 = Array1::from(vec![1.0f32; 1000]);
        let samples2 = Array1::from(vec![2.0f32; 1000]);
        let audio1 = AudioSamples::new_mono(samples1, 44100);
        let audio2 = AudioSamples::new_mono(samples2, 44100);

        let mixed = AudioSamples::mix(&[audio1, audio2], None).unwrap();

        if let AudioData::Mono(arr) = &mixed.data {
            // Equal weighting: (1.0 + 2.0) / 2 = 1.5
            assert!((arr[0] - 1.5).abs() < 1e-6);
        }
    }

    #[test]
    fn test_fade_operations() {
        let samples = Array1::from(vec![1.0f32; 1000]);
        let mut audio = AudioSamples::new_mono(samples, 44100);

        // Test fade in
        audio.fade_in(0.01, FadeCurve::Linear).unwrap(); // 0.01s fade = 441 samples

        if let AudioData::Mono(arr) = &audio.data {
            assert_eq!(arr[0], 0.0); // Should start at 0
            assert!(arr[220] > 0.0 && arr[220] < 1.0); // Should be partially faded (middle of fade)
            // The fade applies to indices 0..441, so arr[440] should be close to but not exactly 1.0
            assert!(arr[440] > 0.99); // Should be nearly full at end of fade
        }

        // Reset and test fade out
        let samples = Array1::from(vec![1.0f32; 1000]);
        let mut audio = AudioSamples::new_mono(samples, 44100);

        audio.fade_out(0.01, FadeCurve::Linear).unwrap();

        if let AudioData::Mono(arr) = &audio.data {
            // The last sample should be very close to 0 but not exactly 0 due to discrete sampling
            assert!(arr[arr.len() - 1] < 0.01); // Should end very close to 0
            let fade_start = arr.len() - 441;
            assert!(arr[fade_start + 220] > 0.0 && arr[fade_start + 220] < 1.0);
            // Should be partially faded
        }
    }

    #[test]
    fn test_repeat_audio() {
        let samples = Array1::from(vec![1.0f32, 2.0]);
        let audio = AudioSamples::new_mono(samples, 44100);

        let repeated = audio.repeat(3).unwrap();

        assert_eq!(repeated.samples_per_channel(), 6); // 2 * 3

        if let AudioData::Mono(arr) = &repeated.data {
            let expected = vec![1.0, 2.0, 1.0, 2.0, 1.0, 2.0];
            assert_eq!(arr.as_slice().unwrap(), &expected);
        }
    }

    #[test]
    fn test_trim_silence() {
        let mut samples = vec![0.0f32; 1000];
        // Add some signal in the middle
        for i in 400..600 {
            samples[i] = 1.0;
        }
        let audio = AudioSamples::new_mono(Array1::from(samples), 44100);

        let trimmed = audio.trim_silence(0.1).unwrap();

        // Should trim silent portions at start and end
        assert_eq!(trimmed.samples_per_channel(), 200); // Samples 400-599

        if let AudioData::Mono(arr) = &trimmed.data {
            assert!(arr.iter().all(|&x| x == 1.0));
        }
    }

    #[test]
    fn test_multi_source_mixing_with_weights() {
        let samples1 = Array1::from(vec![1.0f32; 100]);
        let samples2 = Array1::from(vec![2.0f32; 100]);
        let samples3 = Array1::from(vec![3.0f32; 100]);
        let audio1 = AudioSamples::new_mono(samples1, 44100);
        let audio2 = AudioSamples::new_mono(samples2, 44100);
        let audio3 = AudioSamples::new_mono(samples3, 44100);

        let weights = vec![0.5, 0.3, 0.2];
        let mixed = AudioSamples::mix(&[audio1, audio2, audio3], Some(&weights)).unwrap();

        if let AudioData::Mono(arr) = &mixed.data {
            // 1.0*0.5 + 2.0*0.3 + 3.0*0.2 = 0.5 + 0.6 + 0.6 = 1.7
            assert!((arr[0] - 1.7).abs() < 1e-6);
        }
    }
}
