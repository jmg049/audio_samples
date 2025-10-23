//! Time-domain editing operations for AudioSamples.
//!
//! This module implements the AudioEditing trait, providing comprehensive
//! time-domain audio editing operations including cutting, pasting, mixing,
//! and envelope operations using efficient ndarray operations.

use crate::operations::traits::AudioEditing;
use crate::operations::types::{
    FadeCurve, NoiseColor, PadSide, PerturbationConfig, PerturbationMethod,
};
use crate::repr::AudioData;
use crate::seconds_to_samples;
use crate::utils::{pink_noise, white_noise};
use crate::{
    AudioSample, AudioSampleError, AudioSampleResult, AudioSamples, AudioTypeConversion, ConvertTo,
    I24, iterators::AudioSampleIterators,
};
use ndarray::{Array1, Array2, Axis, s};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

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

impl<T: AudioSample> AudioEditing<T> for AudioSamples<T>
where
    i16: ConvertTo<T>,
    I24: ConvertTo<T>,
    i32: ConvertTo<T>,
    f32: ConvertTo<T>,
    f64: ConvertTo<T>,
    AudioSamples<T>: AudioTypeConversion<T>,
{
    /// Reverses the order of audio samples.
    fn reverse(&self) -> AudioSamples<T>
    where
        Self: Sized,
    {
        match &self.data {
            AudioData::Mono(arr) => {
                // Reverse the 1D array using ndarray's reverse slicing
                let reversed = arr.slice(s![..;-1]).to_owned();
                AudioSamples::new_mono(reversed.into(), self.sample_rate())
            }
            AudioData::MultiChannel(arr) => {
                // Reverse along the time axis (axis 1)
                let reversed = arr.slice(s![.., ..;-1]).to_owned();
                AudioSamples::new_multi_channel(reversed.into(), self.sample_rate())
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
                return {
                    arr.as_slice_mut()
                        .ok_or(AudioSampleError::ArrayLayoutError {
                            message: "Mono samples must be contiguous".to_string(),
                        })?
                        .reverse();
                    Ok(())
                };
            }
            AudioData::MultiChannel(arr) => {
                // Reverse along the time axis (axis 1)
                for mut channel in arr.axis_iter_mut(Axis(0)) {
                    channel
                        .as_slice_mut()
                        .ok_or(AudioSampleError::ArrayLayoutError {
                            message: "Multi-channel samples must be contiguous".to_string(),
                        })?
                        .reverse();
                }
            }
        }
        Ok(())
    }

    /// Extracts a segment of audio between start and end times.
    fn trim(&self, start_seconds: f64, end_seconds: f64) -> AudioSampleResult<AudioSamples<T>>
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
                Ok(AudioSamples::new_mono(trimmed.into(), self.sample_rate()))
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
                Ok(AudioSamples::new_multi_channel(
                    trimmed.into(),
                    self.sample_rate(),
                ))
            }
        }
    }

    /// Adds padding/silence to the beginning and/or end of the audio.
    fn pad(
        &self,
        pad_start_seconds: f64,
        pad_end_seconds: f64,
        pad_value: T,
    ) -> AudioSampleResult<AudioSamples<T>>
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

                Ok(AudioSamples::new_mono(padded.into(), self.sample_rate()))
            }
            AudioData::MultiChannel(arr) => {
                let total_length = start_samples + arr.ncols() + end_samples;
                let mut padded = Array2::from_elem((arr.nrows(), total_length), pad_value);

                // Copy original data to the middle
                padded
                    .slice_mut(s![.., start_samples..start_samples + arr.ncols()])
                    .assign(arr);

                Ok(AudioSamples::new_multi_channel(
                    padded.into(),
                    self.sample_rate(),
                ))
            }
        }
    }

    fn pad_to_duration(
        &self,
        target_duration_seconds: f64,
        pad_value: T,
        pad_side: PadSide,
    ) -> AudioSampleResult<AudioSamples<T>>
    where
        Self: Sized,
    {
        let current_duration = self.duration_seconds();

        if target_duration_seconds <= current_duration {
            return Ok(self.clone());
        }

        let total_target_samples = seconds_to_samples(target_duration_seconds, self.sample_rate());
        let current_samples = self.samples_per_channel();
        let total_padding_samples = total_target_samples - current_samples;

        let (pad_start_samples, pad_end_samples) = match pad_side {
            PadSide::Left => (total_padding_samples, 0),
            PadSide::Right => (0, total_padding_samples),
        };

        let padded = self.pad(
            pad_start_samples as f64 / self.sample_rate() as f64,
            pad_end_samples as f64 / self.sample_rate() as f64,
            pad_value,
        )?;

        Ok(padded)
    }

    /// Splits audio into segments of specified duration.
    fn split(&self, segment_duration_seconds: f64) -> AudioSampleResult<Vec<AudioSamples<T>>>
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
                    segments.push(AudioSamples::new_mono(segment.into(), self.sample_rate()));
                }
                AudioData::MultiChannel(arr) => {
                    let segment = arr.slice(s![.., start..end]).to_owned();
                    segments.push(AudioSamples::new_multi_channel(
                        segment.into(),
                        self.sample_rate(),
                    ));
                }
            }

            start += segment_samples;
        }

        Ok(segments)
    }

    /// Concatenates multiple audio segments into one.
    fn concatenate(segments: &[Self]) -> AudioSampleResult<AudioSamples<T>>
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
            if segment.num_channels() != first.num_channels() {
                return Err(AudioSampleError::InvalidParameter(
                    "All segments must have the same number of channels".to_string(),
                ));
            }
        }

        match &first.data {
            AudioData::Mono(_) => {
                let mut all_samples = Vec::new();
                for segment in segments.iter() {
                    let segment = segment.as_mono().ok_or(AudioSampleError::InvalidInput {
                        msg: "Expected mono audio data".to_string(),
                    })?;
                    all_samples.extend_from_slice(segment.as_slice().ok_or(
                        AudioSampleError::InvalidInput {
                            msg: "Mono samples must be contiguous".to_string(),
                        },
                    )?);
                }

                let concatenated = Array1::from_vec(all_samples);
                Ok(AudioSamples::new_mono(
                    concatenated.into(),
                    first.sample_rate(),
                ))
            }
            AudioData::MultiChannel(_) => {
                let num_channels = first.num_channels();
                let mut total_samples = 0;

                // Calculate total length
                for segment in segments.iter() {
                    total_samples += segment.samples_per_channel();
                }

                // Create concatenated array
                let mut concatenated_data = Vec::with_capacity(num_channels * total_samples);

                // For each channel, concatenate all segments
                for (channel_idx, _) in first.channels().enumerate() {
                    for segment in segments.iter() {
                        let segment_multi =
                            segment
                                .as_multi_channel()
                                .ok_or(AudioSampleError::InvalidInput {
                                    msg: "Expected multi-channel audio data".to_string(),
                                })?;
                        let channel_data = segment_multi.row(channel_idx);
                        concatenated_data.extend_from_slice(channel_data.as_slice().ok_or(
                            AudioSampleError::InvalidInput {
                                msg: "Multi-channel samples must be contiguous".to_string(),
                            },
                        )?);
                    }
                }

                let concatenated =
                    Array2::from_shape_vec((num_channels, total_samples), concatenated_data)
                        .map_err(|e| {
                            AudioSampleError::InvalidParameter(format!("Array shape error: {}", e))
                        })?;

                Ok(AudioSamples::new_multi_channel(
                    concatenated.into(),
                    first.sample_rate(),
                ))
            }
        }
    }

    fn stack(sources: &[Self]) -> AudioSampleResult<AudioSamples<T>>
    where
        Self: Sized,
    {
        if sources.is_empty() {
            return Err(AudioSampleError::InvalidParameter(
                "Cannot stack empty source list".to_string(),
            ));
        }

        if sources.len() == 1 {
            return Ok(sources[0].clone());
        }

        // Validate all sources have the same sample rate and length
        let first: AudioSamples<T> = sources[0].clone();
        if first.is_multi_channel() {
            return Err(AudioSampleError::InvalidParameter(
                "Stacking is only supported for mono sources".to_string(),
            ));
        }

        for (idx, source) in sources.iter().enumerate().skip(1) {
            if source.is_multi_channel() {
                return Err(AudioSampleError::InvalidParameter(format!(
                    "Stacking is only supported for mono sources. Audio at index {} is not mono",
                    idx
                )));
            }

            if source.sample_rate() != first.sample_rate() {
                return Err(AudioSampleError::InvalidParameter(
                    "All sources must have the same sample rate".to_string(),
                ));
            }
            if source.samples_per_channel() != first.samples_per_channel() {
                return Err(AudioSampleError::InvalidParameter(
                    "All sources must have the same length".to_string(),
                ));
            }
        }

        // Create stacked array

        let num_sources = sources.len();
        let num_samples = first.samples_per_channel();
        let mut stacked = Array2::zeros((num_sources, num_samples));
        for (i, source) in sources.iter().enumerate() {
            if let AudioData::Mono(arr) = &source.data {
                stacked.slice_mut(s![i, ..]).assign(&arr.view());
            }
        }

        Ok(AudioSamples::new_multi_channel(
            stacked.into(),
            first.sample_rate(),
        ))
    }

    /// Mixes multiple audio sources together.
    fn mix(sources: &[Self], weights: Option<&[f64]>) -> AudioSampleResult<AudioSamples<T>>
    where
        Self: Sized,
    {
        if sources.is_empty() {
            return Err(AudioSampleError::InvalidParameter(
                "Cannot mix empty source list".to_string(),
            ));
        }

        // Validate all sources have the same properties
        let first: AudioSamples<T> = sources[0].clone();

        for source in sources.iter().skip(1) {
            if source.sample_rate() != first.sample_rate() {
                return Err(AudioSampleError::InvalidParameter(
                    "All sources must have the same sample rate".to_string(),
                ));
            }
            if source.num_channels() != first.num_channels() {
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
                    if let AudioData::Mono(_first_arr) = &first.data {
                        let weight: T = T::convert_from(mix_weights[0])?;
                        result_arr.mapv_inplace(|x| x * weight);
                    }

                    // Add remaining sources
                    for (i, source) in sources.iter().skip(1).enumerate() {
                        if let AudioData::Mono(source_arr) = &source.data {
                            let weight: T = T::convert_from(mix_weights[i + 1])?;
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
                        let weight: T = T::convert_from(mix_weights[0])?;
                        result_arr.mapv_inplace(|x| x * weight);
                    }

                    // Add remaining sources
                    for (i, source) in sources.iter().skip(1).enumerate() {
                        if let AudioData::MultiChannel(source_arr) = &source.data {
                            let weight: T = T::convert_from(mix_weights[i + 1])?;
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
                    let gain_t: T = T::convert_from(gain)?;
                    arr[i] = arr[i] * gain_t;
                }
            }
            AudioData::MultiChannel(arr) => {
                for i in 0..actual_fade_samples {
                    let position = i as f64 / actual_fade_samples as f64;
                    let gain = apply_fade_curve(&curve, position);
                    let gain_t: T = T::convert_from(gain)?;

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
                    let gain_t: T = T::convert_from(gain)?;

                    arr[start_sample + i] = arr[start_sample + i] * gain_t;
                }
            }
            AudioData::MultiChannel(arr) => {
                for i in 0..actual_fade_samples {
                    let position = 1.0 - (i as f64 / actual_fade_samples as f64);
                    let gain = apply_fade_curve(&curve, position);
                    let gain_t: T = T::convert_from(gain)?;
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
    fn repeat(&self, count: usize) -> AudioSampleResult<AudioSamples<T>>
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
                Ok(AudioSamples::new_mono(repeated.into(), self.sample_rate()))
            }
            AudioData::MultiChannel(arr) => {
                let mut repeated = Array2::zeros((arr.nrows(), arr.ncols() * count));
                for i in 0..count {
                    let start = i * arr.ncols();
                    let end = start + arr.ncols();
                    repeated.slice_mut(s![.., start..end]).assign(arr);
                }
                Ok(AudioSamples::new_multi_channel(
                    repeated.into(),
                    self.sample_rate(),
                ))
            }
        }
    }

    /// Crops audio to remove silence from beginning and end.
    fn trim_silence(&self, threshold: T) -> AudioSampleResult<AudioSamples<T>>
    where
        Self: Sized,
    {
        let threshold: f32 = threshold.convert_to()?;
        match &self.data {
            AudioData::Mono(arr) => {
                // Find first non-silent sample
                let mut start = 0;

                for (idx, sample) in arr.iter().enumerate() {
                    match sample.convert_to() {
                        Ok(value) => {
                            let value: f32 = value;
                            if value.abs() > threshold {
                                start = idx; // Include this sample
                                break;
                            }
                        }
                        Err(_) => {
                            return Err(AudioSampleError::ConversionError(
                                sample.to_string(),
                                "T".to_string(),
                                "f32".to_string(),
                                "Failed to convert sample to f32".to_string(),
                            ));
                        }
                    }
                }

                // Find last non-silent sample
                for (idx, sample) in arr.iter().enumerate().rev() {
                    match sample.convert_to() {
                        Ok(value) => {
                            let value: f32 = value;
                            if value.abs() > threshold {
                                return Ok(AudioSamples::new_mono(
                                    arr.slice(s![start..=idx]).to_owned().into(),
                                    self.sample_rate(),
                                ));
                            }
                        }
                        Err(_) => {
                            return Err(AudioSampleError::ConversionError(
                                sample.to_string(),
                                "T".to_string(),
                                "f32".to_string(),
                                "Failed to convert sample to f32".to_string(),
                            ));
                        }
                    }
                }

                // If we reach here, all samples are below threshold, so return minimal audio
                Ok(AudioSamples::zeros_mono(arr.len(), self.sample_rate()))
            }
            AudioData::MultiChannel(arr) => {
                // Find first non-silent frame (any channel above threshold)
                let mut start = 0;

                for (idx, col) in arr.axis_iter(Axis(1)).enumerate() {
                    if col.iter().any(|&x| {
                        match x.convert_to() {
                            Ok(value) => {
                                let value: f32 = value;
                                value.abs() > threshold
                            }
                            Err(_) => false, // If conversion fails, treat as silent
                        }
                    }) {
                        start = idx; // Include this frame
                        break;
                    }
                }

                // Find last non-silent frame
                for (idx, col) in arr.axis_iter(Axis(1)).enumerate().rev() {
                    if col.iter().any(|&x| {
                        match x.convert_to() {
                            Ok(value) => {
                                let value: f32 = value;
                                value.abs() > threshold
                            }
                            Err(_) => false, // If conversion fails, treat as silent
                        }
                    }) {
                        return Ok(AudioSamples::new_multi_channel(
                            arr.slice(s![.., start..=idx]).to_owned().into(),
                            self.sample_rate(),
                        ));
                    }
                }

                // If we reach here, all frames are below threshold, so return minimal audio
                Ok(AudioSamples::zeros_multi(
                    arr.nrows(),
                    arr.len(),
                    self.sample_rate(),
                ))
            }
        }
    }

    /// Applies perturbation to audio samples for data augmentation.
    fn perturb(&self, config: &PerturbationConfig) -> AudioSampleResult<AudioSamples<T>> {
        let mut owned = self.clone();
        owned.perturb_(config)?;
        Ok(owned)
    }

    /// Applies perturbation to audio samples in place.
    fn perturb_(&mut self, config: &PerturbationConfig) -> AudioSampleResult<()> {
        config
            .validate(self.sample_rate() as f64)
            .map_err(|e| AudioSampleError::InvalidParameter(e))?;

        // Apply perturbation based on seed or use thread-local randomness
        if let Some(seed) = config.seed {
            let mut rng = StdRng::seed_from_u64(seed);
            apply_perturbation_with_rng(self, &config.method, &mut rng)
        } else {
            let mut rng = rand::rng();
            apply_perturbation_with_rng(self, &config.method, &mut rng)
        }
    }
}
/// Apply perturbation with a given RNG
fn apply_perturbation_with_rng<T: AudioSample, R: Rng>(
    audio: &mut AudioSamples<T>,
    method: &PerturbationMethod,
    rng: &mut R,
) -> AudioSampleResult<()>
where
    i16: ConvertTo<T>,
    I24: ConvertTo<T>,
    i32: ConvertTo<T>,
    f32: ConvertTo<T>,
    f64: ConvertTo<T>,
    T: ConvertTo<f64>,
    for<'a> AudioSamples<T>: AudioTypeConversion<T>,
{
    match method {
        PerturbationMethod::GaussianNoise {
            target_snr_db,
            noise_color,
        } => apply_gaussian_noise_(audio, *target_snr_db, *noise_color, rng),
        PerturbationMethod::RandomGain {
            min_gain_db,
            max_gain_db,
        } => apply_random_gain_(audio, *min_gain_db, *max_gain_db, rng),
        PerturbationMethod::HighPassFilter {
            cutoff_hz,
            slope_db_per_octave,
        } => apply_high_pass_filter_(audio, *cutoff_hz, *slope_db_per_octave),
        PerturbationMethod::PitchShift {
            semitones,
            preserve_formants,
        } => apply_pitch_shift_(audio, *semitones, *preserve_formants),
    }
}

/// Helper function to generate brown noise
fn generate_brown_noise<T: AudioSample, R: Rng>(
    duration: f64,
    sample_rate: u32,
    amplitude: f64,
    rng: &mut R,
) -> AudioSampleResult<Array1<T>>
where
    i16: ConvertTo<T>,
    I24: ConvertTo<T>,
    i32: ConvertTo<T>,
    f32: ConvertTo<T>,
    f64: ConvertTo<T>,
{
    let num_samples = (duration * sample_rate as f64) as usize;
    let mut samples = Vec::with_capacity(num_samples);
    let mut brown_state = 0.0;

    for _ in 0..num_samples {
        let white = (rng.random_range(0.0..1.0) - 0.5) * 2.0;
        brown_state = (brown_state + white * 0.02_f64).clamp(-1.0, 1.0);
        let sample = amplitude * brown_state;
        samples.push(sample.convert_to()?);
    }

    Ok(Array1::from_vec(samples))
}

/// Calculate RMS (Root Mean Square) of audio samples
fn calculate_rms<T: AudioSample>(audio: &AudioSamples<T>) -> AudioSampleResult<f64>
where
    T: ConvertTo<f64>,
{
    let mut sum_squares = 0.0;
    let mut total_samples = 0;

    match &audio.data {
        AudioData::Mono(arr) => {
            for sample in arr.iter() {
                let sample_f64: f64 = sample.convert_to()?;
                sum_squares += sample_f64 * sample_f64;
                total_samples += 1;
            }
        }
        AudioData::MultiChannel(arr) => {
            for sample in arr.iter() {
                let sample_f64: f64 = sample.convert_to()?;
                sum_squares += sample_f64 * sample_f64;
                total_samples += 1;
            }
        }
    }

    if total_samples > 0 {
        Ok((sum_squares / total_samples as f64).sqrt())
    } else {
        Ok(0.0)
    }
}

/// Apply Gaussian noise to audio samples
fn apply_gaussian_noise_<T: AudioSample, R: Rng>(
    audio: &mut AudioSamples<T>,
    target_snr_db: f64,
    noise_color: NoiseColor,
    rng: &mut R,
) -> AudioSampleResult<()>
where
    i16: ConvertTo<T>,
    I24: ConvertTo<T>,
    i32: ConvertTo<T>,
    f32: ConvertTo<T>,
    f64: ConvertTo<T>,
    T: ConvertTo<f64>,
    for<'a> AudioSamples<T>: AudioTypeConversion<T>,
{
    // Calculate signal RMS
    let signal_rms = calculate_rms(audio)?;
    if signal_rms == 0.0 {
        return Ok(()); // No signal, no noise needed
    }

    // Calculate target noise RMS from SNR
    let target_noise_rms = signal_rms / 10.0_f64.powf(target_snr_db / 20.0);

    let duration = audio.duration_seconds();

    // Generate noise based on color
    let noise_audio: AudioSamples<T> = match noise_color {
        NoiseColor::White => {
            let mut noise = white_noise(duration, audio.sample_rate(), target_noise_rms)?;
            // Use custom RNG for deterministic results
            apply_custom_noise_to_audio(&mut noise, target_noise_rms, rng)?;
            noise
        }
        NoiseColor::Pink => {
            let mut noise = pink_noise(duration, audio.sample_rate(), target_noise_rms)?;
            apply_custom_noise_to_audio(&mut noise, target_noise_rms, rng)?;
            noise
        }
        NoiseColor::Brown => match &audio.data {
            AudioData::Mono(_) => {
                let noise_array =
                    generate_brown_noise(duration, audio.sample_rate(), target_noise_rms, rng)?;
                AudioSamples::new_mono(noise_array.into(), audio.sample_rate())
            }
            AudioData::MultiChannel(arr) => {
                let mut noise_arrays = Vec::new();
                for _ in 0..arr.nrows() {
                    let noise_array =
                        generate_brown_noise(duration, audio.sample_rate(), target_noise_rms, rng)?;
                    noise_arrays.push(noise_array);
                }
                let noise_2d = Array2::from_shape_vec(
                    (arr.nrows(), noise_arrays[0].len()),
                    noise_arrays
                        .into_iter()
                        .flat_map(|arr| arr.into_iter())
                        .collect(),
                )
                .unwrap();
                AudioSamples::new_multi_channel(noise_2d.into(), audio.sample_rate())
            }
        },
    };

    // Add noise to signal
    match (&mut audio.data, &noise_audio.data) {
        (AudioData::Mono(signal), AudioData::Mono(noise)) => {
            for (s, n) in signal.iter_mut().zip(noise.iter()) {
                *s = *s + *n
            }
        }
        (AudioData::MultiChannel(signal), AudioData::MultiChannel(noise)) => {
            for (s, n) in signal.iter_mut().zip(noise.iter()) {
                *s = *s + *n
            }
        }
        _ => {
            return Err(AudioSampleError::InvalidInput {
                msg: "Channel mismatch between signal and noise".to_string(),
            });
        }
    }

    Ok(())
}

/// Helper to apply custom noise generation using provided RNG
fn apply_custom_noise_to_audio<T: AudioSample, R: Rng>(
    noise: &mut AudioSamples<T>,
    amplitude: f64,
    rng: &mut R,
) -> AudioSampleResult<()>
where
    f64: ConvertTo<T>,
{
    match &mut noise.data {
        AudioData::Mono(arr) => {
            for sample in arr.iter_mut() {
                let random_value = (rng.random_range(0.0..1.0) - 0.5) * 2.0;
                *sample = (amplitude * random_value).convert_to()?;
            }
        }
        AudioData::MultiChannel(arr) => {
            for sample in arr.iter_mut() {
                let random_value = (rng.random_range(0.0..1.0) - 0.5) * 2.0;
                *sample = (amplitude * random_value).convert_to()?;
            }
        }
    }
    Ok(())
}

/// Apply random gain to audio samples
fn apply_random_gain_<T: AudioSample, R: Rng>(
    audio: &mut AudioSamples<T>,
    min_gain_db: f64,
    max_gain_db: f64,
    rng: &mut R,
) -> AudioSampleResult<()>
where
    T: ConvertTo<f64>,
    f64: ConvertTo<T>,
{
    let gain_db = rng.random_range(min_gain_db..=max_gain_db);
    let gain_linear = 10.0_f64.powf(gain_db / 20.0);

    match &mut audio.data {
        AudioData::Mono(arr) => {
            for sample in arr.iter_mut() {
                let sample_f64: f64 = sample.convert_to()?;
                *sample = (sample_f64 * gain_linear).convert_to()?;
            }
        }
        AudioData::MultiChannel(arr) => {
            for sample in arr.iter_mut() {
                let sample_f64: f64 = sample.convert_to()?;
                *sample = (sample_f64 * gain_linear).convert_to()?;
            }
        }
    }

    Ok(())
}

/// Apply high-pass filter to audio samples (simple implementation)
fn apply_high_pass_filter_<T: AudioSample>(
    audio: &mut AudioSamples<T>,
    cutoff_hz: f64,
    _slope_db_per_octave: Option<f64>,
) -> AudioSampleResult<()>
where
    T: ConvertTo<f64>,
    f64: ConvertTo<T>,
{
    let sample_rate = audio.sample_rate() as f64;
    let rc = 1.0 / (2.0 * std::f64::consts::PI * cutoff_hz);
    let dt = 1.0 / sample_rate;
    let alpha = rc / (rc + dt);

    match &mut audio.data {
        AudioData::Mono(arr) => {
            if arr.len() > 0 {
                let mut prev_input: f64 = arr[0].convert_to()?;
                let mut prev_output = 0.0;

                for sample in arr.iter_mut() {
                    let current_input: f64 = sample.convert_to()?;
                    let output = alpha * (prev_output + current_input - prev_input);
                    *sample = output.convert_to()?;

                    prev_input = current_input;
                    prev_output = output;
                }
            }
        }
        AudioData::MultiChannel(arr) => {
            for mut channel in arr.axis_iter_mut(Axis(0)) {
                if channel.len() > 0 {
                    let mut prev_input: f64 = channel[0].convert_to()?;
                    let mut prev_output = 0.0;

                    for sample in channel.iter_mut() {
                        let current_input: f64 = sample.convert_to()?;
                        let output = alpha * (prev_output + current_input - prev_input);
                        *sample = output.convert_to()?;

                        prev_input = current_input;
                        prev_output = output;
                    }
                }
            }
        }
    }

    Ok(())
}

/// Apply pitch shift to audio samples (basic implementation)
fn apply_pitch_shift_<T: AudioSample>(
    audio: &mut AudioSamples<T>,
    semitones: f64,
    _preserve_formants: bool,
) -> AudioSampleResult<()>
where
    T: ConvertTo<f64>,
    f64: ConvertTo<T>,
{
    if semitones == 0.0 {
        return Ok(());
    }

    let pitch_ratio = 2.0_f64.powf(semitones / 12.0);

    // Simple time-domain pitch shifting using interpolation
    match &mut audio.data {
        AudioData::Mono(arr) => {
            let original_data: Vec<f64> = arr
                .iter()
                .map(|&x| x.convert_to())
                .collect::<Result<Vec<_>, _>>()?;

            for (i, sample) in arr.iter_mut().enumerate() {
                let source_index = i as f64 * pitch_ratio;
                let index_floor = source_index.floor() as usize;
                let index_frac = source_index - source_index.floor();

                if index_floor < original_data.len() {
                    let interpolated = if index_floor + 1 < original_data.len() {
                        original_data[index_floor] * (1.0 - index_frac)
                            + original_data[index_floor + 1] * index_frac
                    } else {
                        original_data[index_floor]
                    };
                    *sample = interpolated.convert_to()?;
                } else {
                    *sample = T::convert_from(0.0)?;
                }
            }
        }
        AudioData::MultiChannel(arr) => {
            let original_data: Vec<Vec<f64>> = arr
                .outer_iter()
                .map(|channel| {
                    channel
                        .iter()
                        .map(|&x| x.convert_to())
                        .collect::<Result<Vec<_>, _>>()
                })
                .collect::<Result<Vec<_>, _>>()?;

            for (ch_idx, mut channel) in arr.axis_iter_mut(Axis(0)).enumerate() {
                for (i, sample) in channel.iter_mut().enumerate() {
                    let source_index = i as f64 * pitch_ratio;
                    let index_floor = source_index.floor() as usize;
                    let index_frac = source_index - source_index.floor();

                    if index_floor < original_data[ch_idx].len() {
                        let interpolated = if index_floor + 1 < original_data[ch_idx].len() {
                            original_data[ch_idx][index_floor] * (1.0 - index_frac)
                                + original_data[ch_idx][index_floor + 1] * index_frac
                        } else {
                            original_data[ch_idx][index_floor]
                        };
                        *sample = interpolated.convert_to()?;
                    } else {
                        *sample = T::convert_from(0.0)?;
                    }
                }
            }
        }
    }

    Ok(())
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
        let audio = AudioSamples::new_mono(samples.into(), 44100);

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
        let audio = AudioSamples::new_mono(samples.into(), 44100);

        let trimmed = audio.trim(0.25, 0.75).unwrap();

        // Should be 0.5 seconds = 22050 samples
        assert_eq!(trimmed.samples_per_channel(), 22050);
    }

    #[test]
    fn test_pad_with_silence() {
        let samples = Array1::from(vec![1.0f32; 1000]);
        let audio = AudioSamples::new_mono(samples.into(), 44100);

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
        let audio = AudioSamples::new_mono(samples.into(), 44100);

        let segments = audio.split(0.05).unwrap(); // Split into 0.05s segments

        assert_eq!(segments.len(), 4); // 0.2s / 0.05s = 4 segments
        assert_eq!(segments[0].samples_per_channel(), 2205); // 0.05s * 44100
    }

    #[test]
    fn test_concatenate_segments() {
        let samples1 = Array1::from(vec![1.0f32; 1000]);
        let samples2 = Array1::from(vec![2.0f32; 1000]);
        let audio1 = AudioSamples::new_mono(samples1.into(), 44100);
        let audio2 = AudioSamples::new_mono(samples2.into(), 44100);

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
        let audio1 = AudioSamples::new_mono(samples1.into(), 44100);
        let audio2 = AudioSamples::new_mono(samples2.into(), 44100);

        let mixed = AudioSamples::mix(&[audio1, audio2], None).unwrap();

        if let AudioData::Mono(arr) = &mixed.data {
            // Equal weighting: (1.0 + 2.0) / 2 = 1.5
            assert!((arr[0] - 1.5).abs() < 1e-6);
        }
    }

    #[test]
    fn test_fade_operations() {
        let samples = Array1::from(vec![1.0f32; 1000]);
        let mut audio = AudioSamples::new_mono(samples.into(), 44100);

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
        let mut audio = AudioSamples::new_mono(samples.into(), 44100);

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
        let audio = AudioSamples::new_mono(samples.into(), 44100);

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
        let audio = AudioSamples::new_mono(Array1::from(samples).into(), 44100);

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
        let audio1 = AudioSamples::new_mono(samples1.into(), 44100);
        let audio2 = AudioSamples::new_mono(samples2.into(), 44100);
        let audio3 = AudioSamples::new_mono(samples3.into(), 44100);

        let weights = vec![0.5, 0.3, 0.2];
        let mixed = AudioSamples::mix(&[audio1, audio2, audio3], Some(&weights)).unwrap();

        if let AudioData::Mono(arr) = &mixed.data {
            // 1.0*0.5 + 2.0*0.3 + 3.0*0.2 = 0.5 + 0.6 + 0.6 = 1.7
            assert!((arr[0] - 1.7).abs() < 1e-6);
        }
    }

    #[test]
    fn test_perturbation_gaussian_noise() {
        use crate::operations::types::*;

        let samples = Array1::from(vec![1.0f32; 1000]);
        let audio = AudioSamples::new_mono(samples.into(), 44100);

        let config = PerturbationConfig::with_seed(
            PerturbationMethod::gaussian_noise(20.0, NoiseColor::White),
            12345,
        );

        let noisy_audio = audio.perturb(&config).unwrap();

        // The original signal should be different from the noisy signal
        if let (AudioData::Mono(original), AudioData::Mono(noisy)) =
            (&audio.data, &noisy_audio.data)
        {
            assert_ne!(original[0], noisy[0]);
            assert_eq!(original.len(), noisy.len());
        }
    }

    #[test]
    fn test_perturbation_random_gain() {
        use crate::operations::types::*;

        let samples = Array1::from(vec![1.0f32; 100]);
        let mut audio = AudioSamples::new_mono(samples.into(), 44100);

        let config =
            PerturbationConfig::with_seed(PerturbationMethod::random_gain(-3.0, 3.0), 54321);

        let original_sample = if let AudioData::Mono(arr) = &audio.data {
            arr[0]
        } else {
            panic!("Expected mono audio");
        };

        audio.perturb_(&config).unwrap();

        let gained_sample = if let AudioData::Mono(arr) = &audio.data {
            arr[0]
        } else {
            panic!("Expected mono audio");
        };

        // Sample should be modified by gain
        assert_ne!(original_sample, gained_sample);
    }

    #[test]
    fn test_perturbation_high_pass_filter() {
        use crate::operations::types::*;

        let samples = Array1::from(vec![1.0f32; 100]);
        let mut audio = AudioSamples::new_mono(samples.into(), 44100);

        let config = PerturbationConfig::new(PerturbationMethod::high_pass_filter(80.0));

        audio.perturb_(&config).unwrap();

        // After high-pass filtering, the signal should be modified
        // This is more of a smoke test to ensure no crashes
        if let AudioData::Mono(arr) = &audio.data {
            assert_eq!(arr.len(), 100);
        }
    }

    #[test]
    fn test_perturbation_deterministic() {
        let samples = Array1::from(vec![1.0f32; 100]);
        let audio = AudioSamples::new_mono(samples.into(), 44100);

        let config = PerturbationConfig::with_seed(PerturbationMethod::random_gain(-1.0, 1.0), 42);

        let result1 = audio.perturb(&config).unwrap();
        let result2 = audio.perturb(&config).unwrap();

        // Results should be identical due to fixed seed
        if let (AudioData::Mono(arr1), AudioData::Mono(arr2)) = (&result1.data, &result2.data) {
            for (a, b) in arr1.iter().zip(arr2.iter()) {
                assert_eq!(a, b);
            }
        }
    }

    #[test]
    fn test_perturbation_validation() {
        let samples = Array1::from(vec![1.0f32; 100]);
        let mut audio = AudioSamples::new_mono(samples.into(), 44100);

        // Test invalid high-pass filter cutoff
        let invalid_config = PerturbationConfig::new(
            PerturbationMethod::high_pass_filter(50000.0), // Above Nyquist frequency
        );

        let result = audio.perturb_(&invalid_config);
        assert!(result.is_err());
    }
}
