//! Time-domain editing operations for AudioSamples.
//!
//! This module implements the AudioEditing trait, providing comprehensive
//! time-domain audio editing operations including cutting, pasting, mixing,
//! and envelope operations using efficient ndarray operations.

#[cfg(feature = "random-generation")]
use crate::brown_noise;
use crate::operations::traits::AudioEditing;
use crate::operations::types::{
    FadeCurve, NoiseColor, PadSide, PerturbationConfig, PerturbationMethod,
};
use crate::repr::AudioData;
use crate::seconds_to_samples;
use crate::{
    AudioSample, AudioSampleError, AudioSampleResult, AudioSamples, AudioStatistics,
    AudioTypeConversion, ConvertTo, I24, LayoutError, ParameterError, ProcessingError, RealFloat,
    samples_to_seconds, to_precision,
};
#[cfg(feature = "random-generation")]
use crate::{pink_noise, white_noise};
use ndarray::{Array1, Array2, Axis, s};
#[cfg(feature = "random-generation")]
use rand::distr::StandardUniform;
#[cfg(feature = "random-generation")]
use rand::rngs::StdRng;
#[cfg(feature = "random-generation")]
use rand::{Rng, SeedableRng};

/// Validates time bounds for trim operations
fn validate_time_bounds<F: RealFloat>(start: F, end: F, duration: F) -> AudioSampleResult<()> {
    if start < F::zero() {
        return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
            "parameter",
            format!("Start time cannot be negative: {}", start),
        )));
    }
    if end <= start {
        return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
            "parameter",
            format!(
                "End time ({}) must be greater than start time ({})",
                end, start
            ),
        )));
    }
    if end > duration {
        return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
            "parameter",
            format!("End time ({}) exceeds audio duration ({})", end, duration),
        )));
    }
    Ok(())
}

/// Applies fade curve transformation to a position value [0.0, 1.0]
fn apply_fade_curve<F: RealFloat>(curve: &FadeCurve<F>, position: F) -> F {
    match curve {
        FadeCurve::Linear => position,
        FadeCurve::Exponential => position * position,
        FadeCurve::Logarithmic => {
            if position <= F::zero() {
                F::zero()
            } else {
                (F::one() + position).ln() / to_precision::<F, _>(2.0).ln()
            }
        }
        FadeCurve::SmoothStep => {
            position * position * (to_precision::<F, _>(3.0) - to_precision::<F, _>(2.0) * position)
        }
        FadeCurve::Custom(func) => func(position),
    }
}

impl<'a, T: AudioSample> AudioEditing<'a, T> for AudioSamples<'a, T>
where
    i16: ConvertTo<T>,
    I24: ConvertTo<T>,
    i32: ConvertTo<T>,
    f32: ConvertTo<T>,
    f64: ConvertTo<T>,
    AudioSamples<'a, T>: AudioTypeConversion<'a, T>,
{
    /// Reverses the order of audio samples.
    fn reverse<'b>(&self) -> AudioSamples<'b, T>
    where
        Self: Sized,
    {
        match &self.data {
            AudioData::Mono(arr) => {
                // Reverse the 1D array using ndarray's reverse slicing
                let reversed = arr.slice(s![..;-1]).to_owned();
                AudioSamples::new_mono(reversed, self.sample_rate())
            }
            AudioData::Multi(arr) => {
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
                return {
                    arr.as_slice_mut().reverse();
                    Ok(())
                };
            }
            AudioData::Multi(arr) => {
                // Reverse along the time axis (axis 1)
                for mut channel in arr.axis_iter_mut(Axis(0)) {
                    channel
                        .as_slice_mut()
                        .ok_or(AudioSampleError::Layout(LayoutError::NonContiguous {
                            operation: "array access".to_string(),
                            layout_type: "Multi-channel samples must be contiguous".to_string(),
                        }))?
                        .reverse();
                }
            }
        }
        Ok(())
    }

    /// Extracts a segment of audio between start and end times.
    fn trim<'b, F: RealFloat>(
        &self,
        start_seconds: F,
        end_seconds: F,
    ) -> AudioSampleResult<AudioSamples<'b, T>> {
        let duration: F = self.duration_seconds();
        validate_time_bounds(start_seconds, end_seconds, duration)?;

        let start_sample = seconds_to_samples(start_seconds, self.sample_rate());
        let end_sample = seconds_to_samples(end_seconds, self.sample_rate());

        match &self.data {
            AudioData::Mono(arr) => {
                if end_sample > arr.len() {
                    return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
                        "parameter",
                        format!(
                            "End sample {} exceeds audio length {}",
                            end_sample,
                            arr.len()
                        ),
                    )));
                }
                let trimmed = arr.slice(s![start_sample..end_sample]).to_owned();
                Ok(AudioSamples::new_mono(trimmed, self.sample_rate()))
            }
            AudioData::Multi(arr) => {
                if end_sample > arr.ncols() {
                    return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
                        "parameter",
                        format!(
                            "End sample {} exceeds audio length {}",
                            end_sample,
                            arr.ncols()
                        ),
                    )));
                }
                let trimmed = arr.slice(s![.., start_sample..end_sample]).to_owned();
                Ok(AudioSamples::new_multi_channel(trimmed, self.sample_rate()))
            }
        }
    }

    /// Adds padding/silence to the beginning and/or end of the audio.
    fn pad<'b, F: RealFloat>(
        &self,
        pad_start_seconds: F,
        pad_end_seconds: F,
        pad_value: T,
    ) -> AudioSampleResult<AudioSamples<'b, T>> {
        if pad_start_seconds < F::zero() || pad_end_seconds < F::zero() {
            return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
                "padding_durations",
                "Padding durations cannot be negative",
            )));
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
                    .assign(&arr.view());

                Ok(AudioSamples::new_mono(padded, self.sample_rate()))
            }
            AudioData::Multi(arr) => {
                let total_length = start_samples + arr.ncols() + end_samples;
                let mut padded = Array2::from_elem((arr.nrows(), total_length), pad_value);

                // Copy original data to the middle
                padded
                    .slice_mut(s![.., start_samples..start_samples + arr.ncols()])
                    .assign(&arr.view());

                Ok(AudioSamples::new_multi_channel(padded, self.sample_rate()))
            }
        }
    }

    fn pad_samples_right<'b>(
        &self,
        target_num_samples: usize,
        pad_value: T,
    ) -> AudioSampleResult<AudioSamples<'b, T>> {
        let current_num_samples = self.samples_per_channel();
        if target_num_samples <= current_num_samples {
            return Ok(self.clone().into_owned());
        }

        let target_num_samples_seconds: f64 =
            samples_to_seconds(target_num_samples, self.sample_rate);

        self.pad_to_duration(target_num_samples_seconds, pad_value, PadSide::Right)
    }

    fn pad_to_duration<'b, F: RealFloat>(
        &self,
        target_duration_seconds: F,
        pad_value: T,
        pad_side: PadSide,
    ) -> AudioSampleResult<AudioSamples<'b, T>> {
        let current_duration = self.duration_seconds();

        if target_duration_seconds <= current_duration {
            return Ok(self.clone().into_owned());
        }

        let total_target_samples = seconds_to_samples(target_duration_seconds, self.sample_rate());
        let current_samples = self.samples_per_channel();
        let total_padding_samples = total_target_samples - current_samples;

        let (pad_start_samples, pad_end_samples) = match pad_side {
            PadSide::Left => (total_padding_samples, 0),
            PadSide::Right => (0, total_padding_samples),
        };

        let padded = self.pad(
            to_precision::<F, _>(pad_start_samples) / to_precision::<F, _>(self.sample_rate),
            to_precision::<F, _>(pad_end_samples) / to_precision::<F, _>(self.sample_rate),
            pad_value,
        )?;

        Ok(padded)
    }

    /// Splits audio into segments of specified duration.
    fn split<F: RealFloat>(
        &self,
        segment_duration_seconds: F,
    ) -> AudioSampleResult<Vec<AudioSamples<'static, T>>> {
        if segment_duration_seconds <= F::zero() {
            return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
                "parameter",
                "Segment duration must be positive",
            )));
        }

        let segment_samples = seconds_to_samples(segment_duration_seconds, self.sample_rate());
        let total_samples = self.samples_per_channel();

        if segment_samples > total_samples {
            return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
                "parameter",
                "Segment duration exceeds audio length",
            )));
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
                AudioData::Multi(arr) => {
                    let segment = arr.slice(s![.., start..end]).to_owned();
                    segments.push(AudioSamples::new_multi_channel(segment, self.sample_rate()));
                }
            }

            start += segment_samples;
        }

        Ok(segments)
    }

    /// Concatenates multiple audio segments into one.
    fn concatenate<'b>(segments: &'b [AudioSamples<'b, T>]) -> AudioSampleResult<AudioSamples<'b, T>>
    where
        'b : 'a,
        Self: Sized,
    {
        if segments.is_empty() {
            return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
                "parameter",
                "Cannot concatenate empty segment list",
            )));
        }

        // Validate all segments have the same sample rate and channel count
        let first_sample_rate = segments[0].sample_rate();
        let first_num_channels = segments[0].num_channels();
        for segment in segments.iter().skip(1) {
            if segment.sample_rate() != first_sample_rate {
                return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
                    "parameter",
                    "All segments must have the same sample rate",
                )));
            }
            if segment.num_channels() != first_num_channels {
                return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
                    "parameter",
                    "All segments must have the same number of channels",
                )));
            }
        }

        let first_sample_rate = segments[0].sample_rate();
        let first_is_mono = segments[0].is_mono();

        match first_is_mono {
            true => {
                let mut all_samples = Vec::new();
                for segment in segments.iter() {
                    let owned_segment = segment.clone().into_owned();
                    
                    let segment = owned_segment.as_mono().ok_or(AudioSampleError::Parameter(
                        ParameterError::invalid_value("input", "Expected mono audio data"),
                    ))?;
                    all_samples.extend_from_slice(segment.as_slice().ok_or(
                        AudioSampleError::Parameter(ParameterError::invalid_value(
                            "input",
                            "Mono samples must be contiguous",
                        )),
                    )?);
                }

                let concatenated = Array1::from_vec(all_samples);
                Ok(AudioSamples::new_mono(concatenated, first_sample_rate))
            }
            false => {
                let num_channels = segments[0].num_channels();
                let mut total_samples = 0;

                // Calculate total length
                for segment in segments.iter() {
                    total_samples += segment.samples_per_channel();
                }

                // Create concatenated array
                let mut concatenated_data: Vec<T> =
                    Vec::with_capacity(num_channels * total_samples);

                // For each channel, concatenate all segments
                for channel_idx in 0..num_channels {
                    for segment in segments.iter() {
                        
                        let owned_segment = segment.clone().into_owned();
                        
                        let segment_multi =
                            owned_segment
                                .as_multi_channel()
                                .ok_or(AudioSampleError::Parameter(
                                    ParameterError::invalid_value(
                                        "input",
                                        "Expected multi-channel audio data",
                                    ),
                                ))?;
                        let channel_data = segment_multi.row(channel_idx);
                        concatenated_data.extend_from_slice(channel_data.as_slice().ok_or(
                            AudioSampleError::Parameter(ParameterError::invalid_value(
                                "input",
                                "Multi-channel samples must be contiguous",
                            )),
                        )?);
                    }
                }

                let concatenated =
                    Array2::from_shape_vec((num_channels, total_samples), concatenated_data)
                        .map_err(|e| {
                            AudioSampleError::Parameter(ParameterError::invalid_value(
                                "parameter",
                                format!("Array shape error: {}", e),
                            ))
                        })?;

                Ok(AudioSamples::new_multi_channel(
                    concatenated,
                    first_sample_rate,
                ))
            }
        }
    }



    fn stack(sources: &[Self]) -> AudioSampleResult<AudioSamples<'static, T>> {
        if sources.is_empty() {
            return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
                "parameter",
                "Cannot stack empty source list",
            )));
        }

        if sources.len() == 1 {
            return Ok(sources[0].clone().into_owned());
        }

        // Validate all sources have the same sample rate and length
        let first: AudioSamples<T> = sources[0].clone();
        if first.is_multi_channel() {
            return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
                "parameter",
                "Stacking is only supported for mono sources",
            )));
        }

        for (idx, source) in sources.iter().enumerate().skip(1) {
            if source.is_multi_channel() {
                return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
                    "parameter",
                    format!(
                        "Stacking is only supported for mono sources. Audio at index {} is not mono",
                        idx
                    ),
                )));
            }

            if source.sample_rate() != first.sample_rate() {
                return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
                    "parameter",
                    "All sources must have the same sample rate",
                )));
            }
            if source.samples_per_channel() != first.samples_per_channel() {
                return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
                    "parameter",
                    "All sources must have the same length",
                )));
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
            stacked,
            first.sample_rate(),
        ))
    }

    /// Mixes multiple audio sources together.
    fn mix<F>(
        sources: &[Self],
        weights: Option<&[F]>,
    ) -> AudioSampleResult<AudioSamples<'static, T>>
    where
        F: RealFloat + ConvertTo<T>,
        T: ConvertTo<F>,
        Self: Sized,
    {
        if sources.is_empty() {
            return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
                "parameter",
                "Cannot mix empty source list",
            )));
        }

        // Validate all sources have the same properties
        let first: AudioSamples<T> = sources[0].clone();

        for source in sources.iter().skip(1) {
            if source.sample_rate() != first.sample_rate() {
                return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
                    "parameter",
                    "All sources must have the same sample rate",
                )));
            }
            if source.num_channels() != first.num_channels() {
                return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
                    "parameter",
                    "All sources must have the same number of channels",
                )));
            }
            if source.samples_per_channel() != first.samples_per_channel() {
                return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
                    "parameter",
                    "All sources must have the same length",
                )));
            }
        }

        // Validate weights if provided
        let mix_weights = if let Some(w) = weights {
            if w.len() != sources.len() {
                return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
                    "parameter",
                    "Number of weights must match number of sources",
                )));
            }
            w
        } else {
            // Equal weights
            &vec![F::one() / to_precision::<F, _>(sources.len()); sources.len()]
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
                                *r += *s * weight;
                            }
                        }
                    }
                }
                Ok(result.into_owned())
            }
            AudioData::Multi(_) => {
                let mut result = first.clone();

                if let AudioData::Multi(result_arr) = &mut result.data {
                    // Start with first source * weight
                    if let AudioData::Multi(_first_arr) = &first.data {
                        let weight: T = T::convert_from(mix_weights[0])?;
                        result_arr.mapv_inplace(|x| x * weight);
                    }

                    // Add remaining sources
                    for (i, source) in sources.iter().skip(1).enumerate() {
                        if let AudioData::Multi(source_arr) = &source.data {
                            let weight: T = T::convert_from(mix_weights[i + 1])?;
                            for (r, s) in result_arr.iter_mut().zip(source_arr.iter()) {
                                *r += *s * weight;
                            }
                        }
                    }
                }
                Ok(result.into_owned())
            }
        }
    }

    /// Applies fade-in envelope over specified duration.
    fn fade_in<F>(&mut self, duration_seconds: F, curve: FadeCurve<F>) -> AudioSampleResult<()>
    where
        F: RealFloat + ConvertTo<T>,
        T: ConvertTo<F>,
    {
        if duration_seconds <= F::zero() {
            return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
                "parameter",
                "Fade duration must be positive",
            )));
        }

        let fade_samples = seconds_to_samples(duration_seconds, self.sample_rate());
        let total_samples = self.samples_per_channel();
        let actual_fade_samples = fade_samples.min(total_samples);

        match &mut self.data {
            AudioData::Mono(arr) => {
                for i in 0..actual_fade_samples {
                    let position =
                        to_precision::<F, _>(i) / to_precision::<F, _>(actual_fade_samples);
                    let gain = apply_fade_curve(&curve, position);
                    let gain_t: T = T::convert_from(gain)?;
                    arr[i] *= gain_t;
                }
            }
            AudioData::Multi(arr) => {
                for i in 0..actual_fade_samples {
                    let position =
                        to_precision::<F, _>(i) / to_precision::<F, _>(actual_fade_samples);
                    let gain = apply_fade_curve(&curve, position);
                    let gain_t: T = T::convert_from(gain)?;

                    for channel in 0..arr.nrows() {
                        arr[[channel, i]] *= gain_t;
                    }
                }
            }
        }

        Ok(())
    }

    /// Applies fade-out envelope over specified duration.
    fn fade_out<F>(&mut self, duration_seconds: F, curve: FadeCurve<F>) -> AudioSampleResult<()>
    where
        F: RealFloat + ConvertTo<T>,
        T: ConvertTo<F>,
    {
        if duration_seconds <= F::zero() {
            return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
                "parameter",
                "Fade duration must be positive",
            )));
        }

        let fade_samples = seconds_to_samples(duration_seconds, self.sample_rate());
        let total_samples = self.samples_per_channel();
        let actual_fade_samples = fade_samples.min(total_samples);
        let start_sample = total_samples - actual_fade_samples;

        match &mut self.data {
            AudioData::Mono(arr) => {
                for i in 0..actual_fade_samples {
                    let position = F::one()
                        - (to_precision::<F, _>(i) / to_precision::<F, _>(actual_fade_samples));
                    let gain = apply_fade_curve(&curve, position);
                    let gain_t: T = T::convert_from(gain)?;

                    arr[start_sample + i] *= gain_t;
                }
            }
            AudioData::Multi(arr) => {
                for i in 0..actual_fade_samples {
                    let position = F::one()
                        - (to_precision::<F, _>(i) / to_precision::<F, _>(actual_fade_samples));
                    let gain = apply_fade_curve(&curve, position);
                    let gain_t: T = T::convert_from(gain)?;
                    for channel in 0..arr.nrows() {
                        arr[[channel, start_sample + i]] *= gain_t;
                    }
                }
            }
        }

        Ok(())
    }

    /// Repeats the audio signal a specified number of times.
    fn repeat(&self, count: usize) -> AudioSampleResult<AudioSamples<'static, T>> {
        if count == 0 {
            return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
                "parameter",
                "Repeat count must be greater than 0",
            )));
        }

        if count == 1 {
            return Ok(self.clone().into_owned());
        }

        match &self.data {
            AudioData::Mono(arr) => {
                let mut repeated = Array1::zeros(arr.len() * count);
                for i in 0..count {
                    let start = i * arr.len();
                    let end = start + arr.len();
                    repeated.slice_mut(s![start..end]).assign(&arr.view());
                }
                Ok(AudioSamples::new_mono(repeated, self.sample_rate()))
            }
            AudioData::Multi(arr) => {
                let mut repeated = Array2::zeros((arr.nrows(), arr.ncols() * count));
                for i in 0..count {
                    let start = i * arr.ncols();
                    let end = start + arr.ncols();
                    repeated.slice_mut(s![.., start..end]).assign(&arr.view());
                }
                Ok(AudioSamples::new_multi_channel(
                    repeated,
                    self.sample_rate(),
                ))
            }
        }
    }

    /// Crops audio to remove silence from the beginning and end using a dB threshold.
    fn trim_silence<F>(&self, threshold_db: F) -> AudioSampleResult<AudioSamples<'static, T>>
    where
        F: RealFloat + ConvertTo<T>,
        T: ConvertTo<F>,
    {
        // Convert dB threshold to linear amplitude
        // dB = 20 * log10(x)  =>  x = 10^(dB / 20)
        let threshold_lin: F =
            to_precision::<F, _>(10.0).powf(threshold_db / to_precision::<F, _>(20.0));

        match &self.data {
            AudioData::Mono(arr) => {
                // Find first non-silent sample
                let mut start = 0usize;
                let mut found_start = false;

                for (idx, sample) in arr.iter().enumerate() {
                    let value: F = sample.convert_to()?;
                    if value.abs() > threshold_lin {
                        start = idx;
                        found_start = true;
                        break;
                    }
                }

                // If no non-silent sample was found, return silence
                if !found_start {
                    return Ok(AudioSamples::zeros_mono(arr.len(), self.sample_rate()));
                }

                // Find last non-silent sample
                let mut end = arr.len() - 1;
                for (idx, sample) in arr.iter().enumerate().rev() {
                    let value: F = sample.convert_to()?;
                    if value.abs() > threshold_lin {
                        end = idx;
                        break;
                    }
                }

                Ok(AudioSamples::new_mono(
                    arr.slice(s![start..=end]).to_owned(),
                    self.sample_rate(),
                ))
            }

            AudioData::Multi(arr) => {
                let n_frames = arr.ncols();

                // Find first non-silent frame
                let mut start = 0usize;
                let mut found_start = false;
                for idx in 0..n_frames {
                    let col = arr.column(idx);
                    let is_silent = col.iter().all(|&x| {
                        x.convert_to()
                            .map(|v: F| v.abs() <= threshold_lin)
                            .unwrap_or(true)
                    });
                    if !is_silent {
                        start = idx;
                        found_start = true;
                        break;
                    }
                }

                if !found_start {
                    return Ok(AudioSamples::zeros_multi(
                        arr.nrows(),
                        arr.len(),
                        self.sample_rate(),
                    ));
                }

                // Find last non-silent frame
                let mut end = n_frames - 1;
                for idx in (0..n_frames).rev() {
                    let col = arr.column(idx);
                    let is_silent = col.iter().all(|&x| {
                        x.convert_to()
                            .map(|v: F| v.abs() <= threshold_lin)
                            .unwrap_or(true)
                    });
                    if !is_silent {
                        end = idx;
                        break;
                    }
                }

                Ok(AudioSamples::new_multi_channel(
                    arr.slice(s![.., start..=end]).to_owned(),
                    self.sample_rate(),
                ))
            }
        }
    }

    /// Applies perturbation to audio samples for data augmentation.
    #[cfg(feature = "random-generation")]
    fn perturb<'b, F>(
        &self,
        config: &PerturbationConfig<F>,
    ) -> AudioSampleResult<AudioSamples<'b, T>>
    where
        F: RealFloat + ConvertTo<T>,
        T: ConvertTo<F>,
        StandardUniform: rand::prelude::Distribution<F>,
    {
        let mut owned = self.clone();
        owned.perturb_(config)?;
        Ok(owned.into_owned())
    }

    /// Applies perturbation to audio samples in place.
    #[cfg(feature = "random-generation")]
    fn perturb_<F>(&mut self, config: &PerturbationConfig<F>) -> AudioSampleResult<()>
    where
        F: RealFloat + ConvertTo<T>,
        T: ConvertTo<F>,
        StandardUniform: rand::prelude::Distribution<F>,
    {
        config.validate(to_precision(self.sample_rate))?;
        // Apply perturbation based on seed or use thread-local randomness
        if let Some(seed) = config.seed {
            let mut rng = StdRng::seed_from_u64(seed);
            apply_perturbation_with_rng(self, &config.method, &mut rng)
        } else {
            let mut rng = rand::rng();
            apply_perturbation_with_rng(self, &config.method, &mut rng)
        }
    }

    fn trim_all_silence<F>(
        &self,
        threshold_db: F,
        min_silence_duration_seconds: F,
    ) -> AudioSampleResult<AudioSamples<'static, T>>
    where
        F: RealFloat + ConvertTo<T>,
        T: ConvertTo<F>,
    {
        let threshold_lin: F =
            to_precision::<F, _>(10.0).powf(threshold_db / to_precision::<F, _>(20.0));
        let sr = self.sample_rate();
        let min_silence_samples = (min_silence_duration_seconds * to_precision::<F, _>(sr))
            .round()
            .to_usize()
            .expect("Error converting min_silence_duration_seconds to samples");

        match &self.data {
            AudioData::Mono(arr) => {
                let mut segments: Vec<(usize, usize)> = Vec::new();
                let mut in_silence = true;
                let mut silence_start = 0usize;
                let mut segment_start = 0usize;

                for (i, sample) in arr.iter().enumerate() {
                    let value: F = sample.convert_to()?;
                    let is_silent = value.abs() <= threshold_lin;

                    if in_silence && !is_silent {
                        // Entering a non-silent region
                        segment_start = i;
                        in_silence = false;
                    } else if !in_silence && is_silent {
                        // Entering silence
                        silence_start = i;
                        in_silence = true;
                    } else if in_silence {
                        // If silence has lasted long enough, finalise previous segment
                        if i - silence_start >= min_silence_samples && segment_start < silence_start
                        {
                            segments.push((segment_start, silence_start));
                        }
                    }
                }

                // Handle trailing non-silent region
                if !in_silence && segment_start < arr.len() {
                    segments.push((segment_start, arr.len()));
                }

                // Concatenate non-silent segments
                let total_len: usize = segments.iter().map(|(s, e)| e - s).sum();
                let mut result = Array1::<T>::zeros(total_len);
                let mut offset = 0usize;

                for (s, e) in segments {
                    let len = e - s;
                    result
                        .slice_mut(s![offset..offset + len])
                        .assign(&arr.slice(s![s..e]));
                    offset += len;
                }

                Ok(AudioSamples::new_mono(result, sr))
            }

            AudioData::Multi(arr) => {
                let n_channels = arr.nrows();
                let n_frames = arr.ncols();
                let mut segments: Vec<(usize, usize)> = Vec::new();
                let mut in_silence = true;
                let mut silence_start = 0usize;
                let mut segment_start = 0usize;

                for i in 0..n_frames {
                    let is_silent = arr.column(i).iter().all(|&x| {
                        x.convert_to()
                            .map(|v: F| v.abs() <= threshold_lin)
                            .unwrap_or(true)
                    });

                    if in_silence && !is_silent {
                        segment_start = i;
                        in_silence = false;
                    } else if !in_silence && is_silent {
                        silence_start = i;
                        in_silence = true;
                    } else if in_silence
                        && i - silence_start >= min_silence_samples
                        && segment_start < silence_start
                    {
                        segments.push((segment_start, silence_start));
                    }
                }

                if !in_silence && segment_start < n_frames {
                    segments.push((segment_start, n_frames));
                }

                let total_len: usize = segments.iter().map(|(s, e)| e - s).sum();
                let mut result = ndarray::Array2::<T>::zeros((n_channels, total_len));
                let mut offset = 0usize;

                for (s, e) in segments {
                    let len = e - s;
                    result
                        .slice_mut(s![.., offset..offset + len])
                        .assign(&arr.slice(s![.., s..e]));
                    offset += len;
                }

                Ok(AudioSamples::new_multi_channel(result, sr))
            }
        }
    }
    
    fn concatenate_owned<'b>(
        segments: Vec<AudioSamples<'_, T>>,
    ) -> AudioSampleResult<AudioSamples<'b, T>> 
        where Self: Sized {
                    if segments.is_empty() {
            return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
                "parameter",
                "Cannot concatenate empty segment list",
            )));
        }

        // Validate all segments have the same sample rate and channel count
        let first_sample_rate = segments[0].sample_rate();
        let first_num_channels = segments[0].num_channels();
        for segment in segments.iter().skip(1) {
            if segment.sample_rate() != first_sample_rate {
                return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
                    "parameter",
                    "All segments must have the same sample rate",
                )));
            }
            if segment.num_channels() != first_num_channels {
                return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
                    "parameter",
                    "All segments must have the same number of channels",
                )));
            }
        }

        let first_sample_rate = segments[0].sample_rate();
        let first_is_mono = segments[0].is_mono();

        match first_is_mono {
            true => {
                let mut all_samples = Vec::new();
                for segment in segments.iter() {
                    let owned_segment = segment.clone().into_owned();
                    
                    let segment = owned_segment.as_mono().ok_or(AudioSampleError::Parameter(
                        ParameterError::invalid_value("input", "Expected mono audio data"),
                    ))?;
                    all_samples.extend_from_slice(segment.as_slice().ok_or(
                        AudioSampleError::Parameter(ParameterError::invalid_value(
                            "input",
                            "Mono samples must be contiguous",
                        )),
                    )?);
                }

                let concatenated = Array1::from_vec(all_samples);
                Ok(AudioSamples::new_mono(concatenated, first_sample_rate))
            }
            false => {
                let num_channels = segments[0].num_channels();
                let mut total_samples = 0;

                // Calculate total length
                for segment in segments.iter() {
                    total_samples += segment.samples_per_channel();
                }

                // Create concatenated array
                let mut concatenated_data: Vec<T> =
                    Vec::with_capacity(num_channels * total_samples);

                // For each channel, concatenate all segments
                for channel_idx in 0..num_channels {
                    for segment in segments.iter() {
                        
                        let owned_segment = segment.clone().into_owned();
                        
                        let segment_multi =
                            owned_segment
                                .as_multi_channel()
                                .ok_or(AudioSampleError::Parameter(
                                    ParameterError::invalid_value(
                                        "input",
                                        "Expected multi-channel audio data",
                                    ),
                                ))?;
                        let channel_data = segment_multi.row(channel_idx);
                        concatenated_data.extend_from_slice(channel_data.as_slice().ok_or(
                            AudioSampleError::Parameter(ParameterError::invalid_value(
                                "input",
                                "Multi-channel samples must be contiguous",
                            )),
                        )?);
                    }
                }

                let concatenated =
                    Array2::from_shape_vec((num_channels, total_samples), concatenated_data)
                        .map_err(|e| {
                            AudioSampleError::Parameter(ParameterError::invalid_value(
                                "parameter",
                                format!("Array shape error: {}", e),
                            ))
                        })?;

                Ok(AudioSamples::new_multi_channel(
                    concatenated,
                    first_sample_rate,
                ))
            }
        }
    }
}

/// Apply perturbation with a given RNG
#[cfg(feature = "random-generation")]
fn apply_perturbation_with_rng<T, R, F>(
    audio: &mut AudioSamples<T>,
    method: &PerturbationMethod<F>,
    rng: &mut R,
) -> AudioSampleResult<()>
where
    T: AudioSample + ConvertTo<F>,
    R: Rng,
    F: RealFloat + ConvertTo<T>,
    i16: ConvertTo<T>,
    I24: ConvertTo<T>,
    i32: ConvertTo<T>,
    f32: ConvertTo<T>,
    f64: ConvertTo<T>,
    for<'a> AudioSamples<'a, T>: AudioTypeConversion<'a, T>,
    StandardUniform: rand::distr::Distribution<F>,
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

/// Calculate RMS (Root Mean Square) of audio samples
fn calculate_rms<T, F>(audio: &AudioSamples<T>) -> Option<F>
where
    T: AudioSample + ConvertTo<F>,
    F: RealFloat,
    i16: ConvertTo<T>,
    I24: ConvertTo<T>,
    i32: ConvertTo<T>,
    f32: ConvertTo<T>,
    f64: ConvertTo<T>,
    for<'a> AudioSamples<'a, T>: AudioStatistics<'a, T>,
{
    audio.rms()
}

/// Apply Gaussian noise to audio samples
#[cfg(feature = "random-generation")]
fn apply_gaussian_noise_<T, R, F>(
    audio: &mut AudioSamples<T>,
    target_snr_db: F,
    noise_color: NoiseColor,
    rng: &mut R,
) -> AudioSampleResult<()>
where
    T: AudioSample + ConvertTo<F>,
    R: Rng,
    F: RealFloat + ConvertTo<T>,
    i16: ConvertTo<T>,
    I24: ConvertTo<T>,
    i32: ConvertTo<T>,
    f32: ConvertTo<T>,
    f64: ConvertTo<T>,
    for<'a> AudioSamples<'a, T>: AudioTypeConversion<'a, T>,
    StandardUniform: rand::distr::Distribution<F>,
{
    // Calculate signal RMS
    let signal_rms: F = match calculate_rms(audio) {
        Some(rms) => rms,
        None => {
            return Err(AudioSampleError::Processing(
                ProcessingError::algorithm_failure("algorithm", "Failed to calculate signal RMS"),
            ));
        }
    };

    // Calculate target noise RMS from SNR
    let target_noise_rms =
        signal_rms / to_precision::<F, _>(10.0).powf(target_snr_db / to_precision::<F, _>(20.0));

    let duration = audio.duration_seconds();

    // Generate noise based on color
    let noise_audio: AudioSamples<T> = match noise_color {
        NoiseColor::White => {
            let mut noise = white_noise(duration, audio.sample_rate, target_noise_rms);
            // Use custom RNG for deterministic results
            apply_custom_noise_to_audio(&mut noise, target_noise_rms, rng)?;
            noise
        }
        NoiseColor::Pink => {
            let mut noise = pink_noise(duration, audio.sample_rate, target_noise_rms);
            apply_custom_noise_to_audio(&mut noise, target_noise_rms, rng)?;
            noise
        }
        NoiseColor::Brown => match &audio.data {
            AudioData::Mono(_) => brown_noise(
                duration,
                audio.sample_rate(),
                to_precision::<F, _>(0.02),
                target_noise_rms,
            )?,
            AudioData::Multi(arr) => {
                let mut noise_arrays = Vec::new();
                for _ in 0..arr.nrows() {
                    let noise = brown_noise(
                        duration,
                        audio.sample_rate(),
                        to_precision::<F, _>(0.02),
                        target_noise_rms,
                    )?;
                    noise_arrays.push(noise);
                }
                AudioSamples::stack(&noise_arrays)?
            }
        },
    };

    // Add noise to signal
    match (&mut audio.data, &noise_audio.data) {
        (AudioData::Mono(signal), AudioData::Mono(noise)) => {
            for (s, n) in signal.iter_mut().zip(noise.iter()) {
                *s += *n
            }
        }
        (AudioData::Multi(signal), AudioData::Multi(noise)) => {
            for (s, n) in signal.iter_mut().zip(noise.iter()) {
                *s += *n
            }
        }
        _ => {
            return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
                "input",
                "Channel mismatch between signal and noise",
            )));
        }
    }

    Ok(())
}

/// Helper to apply custom noise generation using provided RNG
#[cfg(feature = "random-generation")]
fn apply_custom_noise_to_audio<T, R, F>(
    noise: &mut AudioSamples<T>,
    amplitude: F,
    rng: &mut R,
) -> AudioSampleResult<()>
where
    T: AudioSample + ConvertTo<F>,
    R: Rng,
    F: RealFloat + ConvertTo<T>,
{
    match &mut noise.data {
        AudioData::Mono(arr) => {
            for sample in arr.iter_mut() {
                let random_value: f64 = (rng.random_range(0.0..1.0) - 0.5) * 2.0;
                let random_value: F = to_precision::<F, _>(random_value);
                *sample = (amplitude * random_value).convert_to()?;
            }
        }
        AudioData::Multi(arr) => {
            for sample in arr.iter_mut() {
                let random_value = (rng.random_range(0.0..1.0) - 0.5) * 2.0;
                let random_value: F = to_precision::<F, _>(random_value);
                *sample = (amplitude * random_value).convert_to()?;
            }
        }
    }
    Ok(())
}

/// Apply random gain to audio samples
#[cfg(feature = "random-generation")]
fn apply_random_gain_<T, R, F>(
    audio: &mut AudioSamples<T>,
    min_gain_db: F,
    max_gain_db: F,
    rng: &mut R,
) -> AudioSampleResult<()>
where
    T: AudioSample + ConvertTo<F>,
    R: Rng,
    F: RealFloat + ConvertTo<T>,
{
    let gain_db = rng.random_range(
        min_gain_db
            .to_f64()
            .expect("Float conversion should not fail")
            ..=max_gain_db
                .to_f64()
                .expect("Float conversion should not fail"),
    );
    let gain_db: F = to_precision::<F, _>(gain_db);
    let gain_linear: F =
        to_precision::<F, _>(10.0).powf(to_precision::<F, _>(gain_db) / to_precision::<F, _>(20.0));
    // 10.0_f64.powf(gain_db / 20.0);

    match &mut audio.data {
        AudioData::Mono(arr) => {
            for sample in arr.iter_mut() {
                let sample_f: F = sample.convert_to()?;
                *sample = (sample_f * gain_linear).convert_to()?;
            }
        }
        AudioData::Multi(arr) => {
            for sample in arr.iter_mut() {
                let sample_f: F = sample.convert_to()?;
                *sample = (sample_f * gain_linear).convert_to()?;
            }
        }
    }

    Ok(())
}

/// Apply high-pass filter to audio samples (simple implementation)
fn apply_high_pass_filter_<T, F>(
    audio: &mut AudioSamples<T>,
    cutoff_hz: F,
    _slope_db_per_octave: Option<F>,
) -> AudioSampleResult<()>
where
    T: AudioSample + ConvertTo<F>,
    F: RealFloat + ConvertTo<T>,
{
    let sample_rate = to_precision::<F, _>(audio.sample_rate());
    let rc = F::one() / (to_precision::<F, _>(2.0) * F::PI() * cutoff_hz);
    let dt = F::one() / sample_rate;
    let alpha = rc / (rc + dt);

    match &mut audio.data {
        AudioData::Mono(arr) => {
            if !arr.is_empty() {
                let mut prev_input: F = arr[0].convert_to()?;
                let mut prev_output = F::zero();

                for sample in arr.iter_mut() {
                    let current_input: F = sample.convert_to()?;
                    let output = alpha * (prev_output + current_input - prev_input);
                    *sample = output.convert_to()?;

                    prev_input = current_input;
                    prev_output = output;
                }
            }
        }
        AudioData::Multi(arr) => {
            for mut channel in arr.axis_iter_mut(Axis(0)) {
                if !channel.is_empty() {
                    let mut prev_input: F = channel[0].convert_to()?;
                    let mut prev_output = F::zero();

                    for sample in channel.iter_mut() {
                        let current_input: F = sample.convert_to()?;
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
fn apply_pitch_shift_<T, F>(
    audio: &mut AudioSamples<T>,
    semitones: F,
    _preserve_formants: bool,
) -> AudioSampleResult<()>
where
    T: AudioSample + ConvertTo<F>,
    F: RealFloat + ConvertTo<T>,
{
    if semitones == F::zero() {
        return Ok(());
    }

    let pitch_ratio = to_precision::<F, _>(2.0).powf(semitones / to_precision::<F, _>(12.0));

    // Simple time-domain pitch shifting using interpolation
    match &mut audio.data {
        AudioData::Mono(arr) => {
            let original_data: Vec<F> = arr
                .iter()
                .map(|&x| x.convert_to())
                .collect::<Result<Vec<_>, _>>()?;

            for (i, sample) in arr.iter_mut().enumerate() {
                let source_index = to_precision::<F, _>(i) * pitch_ratio;
                let index_floor = source_index
                    .floor()
                    .to_usize()
                    .expect("Float conversion to usize should not fail");
                let index_frac = source_index - source_index.floor();

                if index_floor < original_data.len() {
                    let interpolated = if index_floor + 1 < original_data.len() {
                        original_data[index_floor] * (F::one() - index_frac)
                            + original_data[index_floor + 1] * index_frac
                    } else {
                        original_data[index_floor]
                    };
                    *sample = interpolated.convert_to()?;
                } else {
                    *sample = T::convert_from(F::zero())?;
                }
            }
        }
        AudioData::Multi(arr) => {
            let original_data: Vec<Vec<F>> = arr
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
                    let source_index = to_precision::<F, _>(i) * pitch_ratio;
                    let index_floor = source_index
                        .floor()
                        .to_usize()
                        .expect("Float conversion to usize should not fail");
                    let index_frac = source_index - source_index.floor();

                    if index_floor < original_data[ch_idx].len() {
                        let interpolated = if index_floor + 1 < original_data[ch_idx].len() {
                            original_data[ch_idx][index_floor] * (F::one() - index_frac)
                                + original_data[ch_idx][index_floor + 1] * index_frac
                        } else {
                            original_data[ch_idx][index_floor]
                        };
                        *sample = interpolated.convert_to()?;
                    } else {
                        *sample = T::convert_from(F::zero())?;
                    }
                }
            }
        }
    }

    Ok(())
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
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
        let audio= &[audio1, audio2];
        let concatenated = AudioSamples::concatenate(audio).unwrap();

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

        let mixed = AudioSamples::mix::<f32>(&[audio1, audio2], None).unwrap();

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

        let trimmed = audio.trim_silence(-10.0).unwrap();

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
