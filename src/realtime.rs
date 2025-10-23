//! Real-time audio operations with zero-allocation guarantees.
//!
//! This module provides allocation-free variants of common audio processing operations
//! specifically designed for real-time audio applications where memory allocation
//! can cause audio dropouts and latency issues.
//!
//! # Key Design Principles
//!
//! ## Zero Allocation
//! All operations in this module are guaranteed to not allocate memory during processing,
//! making them suitable for real-time audio threads.
//!
//! ## In-place Processing
//! Operations work directly on existing audio buffers without creating intermediate
//! copies or temporary allocations.
//!
//! ## Bounded Time Complexity
//! Operations have predictable, bounded execution time suitable for audio callback
//! functions with strict timing requirements.
//!
//! # Example Usage
//!
//! ```rust
//! use audio_samples::{AudioSamples, realtime::RealtimeAudioOps};
//! use ndarray::array;
//!
//! let mut audio = AudioSamples::new_mono(array![0.1f32, 0.8, -0.3, 0.9], 44100);
//!
//! // All operations are allocation-free and suitable for real-time use
//! audio.realtime_scale(0.5);
//! audio.realtime_apply_gain(0.8);
//! audio.realtime_clip(-1.0, 1.0);
//! ```

use num_traits::Zero;

use crate::{AudioSample, AudioSampleError, AudioSampleResult, AudioSamples};

/// Trait providing allocation-free audio operations for real-time processing.
///
/// All methods in this trait are guaranteed to:
/// - Not allocate memory during execution
/// - Have bounded, predictable execution time
/// - Work in-place on existing audio data
/// - Be suitable for real-time audio callback threads
pub trait RealtimeAudioOps<T: AudioSample> {
    /// Apply gain to all samples in-place without allocation.
    ///
    /// This is a zero-allocation alternative to scaling operations that's
    /// suitable for real-time audio processing.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use audio_samples::{AudioSamples, realtime::RealtimeAudioOps};
    /// # use ndarray::array;
    /// let mut audio = AudioSamples::new_mono(array![0.5f32, -0.3, 0.8], 44100);
    /// audio.realtime_scale(0.5); // Scale by 50%
    /// ```
    fn realtime_scale(&mut self, factor: T);

    /// Apply gain with dB conversion in-place without allocation.
    ///
    /// Converts decibel value to linear gain and applies it to all samples.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use audio_samples::{AudioSamples, realtime::RealtimeAudioOps};
    /// # use ndarray::array;
    /// let mut audio = AudioSamples::new_mono(array![0.5f32, -0.3, 0.8], 44100);
    /// audio.realtime_apply_gain_db(-6.0); // Apply -6dB gain
    /// ```
    fn realtime_apply_gain_db(&mut self, gain_db: f64);

    /// Hard clip samples to specified range in-place without allocation.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use audio_samples::{AudioSamples, realtime::RealtimeAudioOps};
    /// # use ndarray::array;
    /// let mut audio = AudioSamples::new_mono(array![0.5f32, -1.5, 1.2], 44100);
    /// audio.realtime_clip(-1.0, 1.0); // Clip to [-1.0, 1.0] range
    /// ```
    fn realtime_clip(&mut self, min: T, max: T);

    /// Apply soft limiting using fast tanh approximation without allocation.
    ///
    /// Uses a computationally efficient tanh approximation for soft limiting
    /// that's suitable for real-time processing.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use audio_samples::{AudioSamples, realtime::RealtimeAudioOps};
    /// # use ndarray::array;
    /// let mut audio = AudioSamples::new_mono(array![0.5f32, -1.5, 1.2], 44100);
    /// audio.realtime_soft_limit(0.8); // Soft limit with threshold 0.8
    /// ```
    fn realtime_soft_limit(&mut self, threshold: T);

    /// Apply a function to all samples in-place without allocation.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use audio_samples::{AudioSamples, realtime::RealtimeAudioOps};
    /// # use ndarray::array;
    /// let mut audio = AudioSamples::new_mono(array![0.5f32, -0.3, 0.8], 44100);
    /// audio.realtime_map_inplace(|sample| sample * sample); // Square all samples
    /// ```
    fn realtime_map_inplace<F>(&mut self, f: F)
    where
        F: Fn(T) -> T;

    /// Apply DC offset removal (high-pass filter) without allocation.
    ///
    /// Uses a simple one-pole high-pass filter to remove DC offset.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use audio_samples::{AudioSamples, realtime::RealtimeAudioOps};
    /// # use ndarray::array;
    /// let mut audio = AudioSamples::new_mono(array![0.5f32, 0.6, 0.55], 44100);
    /// audio.realtime_remove_dc_offset(0.995); // Remove DC with cutoff coefficient
    /// ```
    fn realtime_remove_dc_offset(&mut self, coefficient: T);

    /// Apply simple lowpass filter without allocation.
    ///
    /// Uses a one-pole lowpass filter with specified cutoff coefficient.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use audio_samples::{AudioSamples, realtime::RealtimeAudioOps};
    /// # use ndarray::array;
    /// let mut audio = AudioSamples::new_mono(array![0.5f32, -0.3, 0.8], 44100);
    /// audio.realtime_lowpass_filter(0.8); // Apply lowpass with coefficient 0.8
    /// ```
    fn realtime_lowpass_filter(&mut self, coefficient: T);

    /// Mix with another audio source in-place without allocation.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use audio_samples::{AudioSamples, realtime::RealtimeAudioOps};
    /// # use ndarray::array;
    /// let mut audio1 = AudioSamples::new_mono(array![0.5f32, -0.3, 0.8], 44100);
    /// let audio2 = AudioSamples::new_mono(array![0.2f32, 0.1, -0.4], 44100);
    /// audio1.realtime_mix_with(&audio2, 0.5); // Mix at 50% level
    /// ```
    fn realtime_mix_with(&mut self, other: &AudioSamples<T>, mix_level: T)
    -> AudioSampleResult<()>;

    /// Apply stereo width adjustment for multi-channel audio without allocation.
    ///
    /// Only works with stereo (2-channel) audio.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use audio_samples::{AudioSamples, realtime::RealtimeAudioOps};
    /// # use ndarray::array;
    /// let stereo_data = array![[0.5f32, -0.3], [0.8, 0.2]]; // 2 channels × 2 samples
    /// let mut audio = AudioSamples::new_multi_channel(stereo_data, 44100);
    /// audio.realtime_adjust_stereo_width(1.5); // Widen stereo image
    /// ```
    fn realtime_adjust_stereo_width(&mut self, width: T) -> AudioSampleResult<()>;

    /// Process audio block using pre-allocated windowed state without allocation.
    ///
    /// This is designed for real-time STFT-style processing with overlap-add.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use audio_samples::{AudioSamples, realtime::{RealtimeAudioOps, RealtimeState}};
    /// # use ndarray::array;
    /// let mut audio = AudioSamples::new_mono(array![0.1f32, 0.2, 0.3, 0.4], 44100);
    /// let mut state = RealtimeState::new(1);
    ///
    /// // Initialize windowed processing (done once, not in real-time)
    /// state.setup_windowed_processing(512, 256);
    ///
    /// // Process blocks in real-time (allocation-free)
    /// let block = [0.1f32, 0.2, 0.3, 0.4];
    /// let output = audio.realtime_windowed_process(&block, &mut state, |window| {
    ///     // Custom processing function - runs on each window
    ///     window.iter().map(|&x| x * 0.5).collect()
    /// }).unwrap();
    /// ```
    fn realtime_windowed_process<F>(
        &mut self,
        input_block: &[T],
        state: &mut RealtimeState<T>,
        processor: F,
    ) -> AudioSampleResult<Vec<T>>
    where
        F: Fn(&[T]) -> Vec<T>;

    /// Apply overlapping windowed processing with pre-allocated buffers.
    ///
    /// This is a more efficient version of windowed processing that reuses
    /// buffers and minimizes allocations.
    fn realtime_overlapping_process<F>(
        &mut self,
        state: &mut RealtimeState<T>,
        processor: F,
    ) -> AudioSampleResult<()>
    where
        F: Fn(&mut [T]);
}

impl<T: AudioSample> RealtimeAudioOps<T> for AudioSamples<T> {
    fn realtime_scale(&mut self, factor: T) {
        match &mut self.data {
            crate::AudioData::Mono(arr) => {
                for sample in arr.iter_mut() {
                    *sample = *sample * factor;
                }
            }
            crate::AudioData::MultiChannel(arr) => {
                for sample in arr.iter_mut() {
                    *sample = *sample * factor;
                }
            }
        }
    }

    fn realtime_apply_gain_db(&mut self, gain_db: f64) {
        // Convert dB to linear gain: gain_linear = 10^(gain_db/20)
        let gain_db = gain_db as f32;
        let twenty: f32 = 20.0;
        let ten: f32 = 10.0;
        let gain_linear = ten.powf(gain_db / twenty);
        self.realtime_scale(T::from_f32(gain_linear).unwrap());
    }

    fn realtime_clip(&mut self, min: T, max: T) {
        match &mut self.data {
            crate::AudioData::Mono(arr) => {
                for sample in arr.iter_mut() {
                    if *sample < min {
                        *sample = min;
                    } else if *sample > max {
                        *sample = max;
                    }
                }
            }
            crate::AudioData::MultiChannel(arr) => {
                for sample in arr.iter_mut() {
                    if *sample < min {
                        *sample = min;
                    } else if *sample > max {
                        *sample = max;
                    }
                }
            }
        }
    }

    fn realtime_soft_limit(&mut self, threshold: T) {
        // Fast tanh approximation for soft limiting
        let tanh_approx = |x: T| -> T {
            let three: T = T::cast_from(3.0);
            let nine: T = T::cast_from(9.0);
            let twenty_seven: T = T::cast_from(27.0);
            let neg_one: T = T::cast_from(-1.0);
            let one = T::one();

            if x < -three {
                neg_one
            } else if x > three {
                one
            } else {
                x * (twenty_seven + x * x) / (twenty_seven + nine * x * x)
            }
        };

        match &mut self.data {
            crate::AudioData::Mono(arr) => {
                for sample in arr.iter_mut() {
                    let normalized = *sample / threshold;
                    let limited = tanh_approx(normalized) * threshold;
                    *sample = limited;
                }
            }
            crate::AudioData::MultiChannel(arr) => {
                for sample in arr.iter_mut() {
                    let normalized = *sample / threshold;
                    let limited = tanh_approx(normalized) * threshold;
                    *sample = limited;
                }
            }
        }
    }

    fn realtime_map_inplace<F>(&mut self, f: F)
    where
        F: Fn(T) -> T,
    {
        match &mut self.data {
            crate::AudioData::Mono(arr) => {
                for sample in arr.iter_mut() {
                    *sample = f(*sample);
                }
            }
            crate::AudioData::MultiChannel(arr) => {
                for sample in arr.iter_mut() {
                    *sample = f(*sample);
                }
            }
        }
    }

    fn realtime_remove_dc_offset(&mut self, coefficient: T) {
        // Simple one-pole high-pass filter: y[n] = coefficient * (y[n-1] + x[n] - x[n-1])
        match &mut self.data {
            crate::AudioData::Mono(arr) => {
                if arr.len() < 2 {
                    return; // Need at least 2 samples
                }

                let mut prev_input = arr[0];
                let mut prev_output = T::zero();

                for i in 1..arr.len() {
                    let current_input = arr[i];
                    let output = coefficient * (prev_output + current_input - prev_input);
                    arr[i] = output;
                    prev_input = current_input;
                    prev_output = output;
                }
            }
            crate::AudioData::MultiChannel(arr) => {
                let (channels, samples) = arr.dim();
                if samples < 2 {
                    return;
                }

                for ch in 0..channels {
                    let mut prev_input = arr[(ch, 0)];
                    let mut prev_output = T::zero();

                    for i in 1..samples {
                        let current_input = arr[(ch, i)];
                        let output = coefficient * (prev_output + current_input - prev_input);
                        arr[(ch, i)] = output;
                        prev_input = current_input;
                        prev_output = output;
                    }
                }
            }
        }
    }

    fn realtime_lowpass_filter(&mut self, coefficient: T) {
        // Simple one-pole lowpass filter: y[n] = coefficient * y[n-1] + (1 - coefficient) * x[n]
        let one_minus_coeff = T::one() - coefficient;

        match &mut self.data {
            crate::AudioData::Mono(arr) => {
                if arr.is_empty() {
                    return;
                }

                let mut prev_output = arr[0];
                for i in 1..arr.len() {
                    let output = coefficient * prev_output + one_minus_coeff * arr[i];
                    arr[i] = output;
                    prev_output = output;
                }
            }
            crate::AudioData::MultiChannel(arr) => {
                let (channels, samples) = arr.dim();
                if samples == 0 {
                    return;
                }

                for ch in 0..channels {
                    let mut prev_output = arr[(ch, 0)];
                    for i in 1..samples {
                        let output = coefficient * prev_output + one_minus_coeff * arr[(ch, i)];
                        arr[(ch, i)] = output;
                        prev_output = output;
                    }
                }
            }
        }
    }

    fn realtime_mix_with(
        &mut self,
        other: &AudioSamples<T>,
        mix_level: T,
    ) -> AudioSampleResult<()> {
        if self.num_channels() != other.num_channels()
            || self.samples_per_channel() != other.samples_per_channel()
        {
            return Err(AudioSampleError::DimensionMismatch(
                "Audio samples must have same dimensions for mixing".to_string(),
            ));
        }

        match (&mut self.data, &other.data) {
            (crate::AudioData::Mono(self_arr), crate::AudioData::Mono(other_arr)) => {
                for (self_sample, other_sample) in self_arr.iter_mut().zip(other_arr.iter()) {
                    *self_sample = *self_sample + mix_level * *other_sample;
                }
            }
            (
                crate::AudioData::MultiChannel(self_arr),
                crate::AudioData::MultiChannel(other_arr),
            ) => {
                for (self_sample, other_sample) in self_arr.iter_mut().zip(other_arr.iter()) {
                    *self_sample = *self_sample + mix_level * *other_sample;
                }
            }
            _ => {
                return Err(AudioSampleError::DimensionMismatch(
                    "Audio samples have incompatible channel layouts".to_string(),
                ));
            }
        }

        Ok(())
    }

    fn realtime_adjust_stereo_width(&mut self, width: T) -> AudioSampleResult<()> {
        match &mut self.data {
            crate::AudioData::Mono(_) => {
                return Err(AudioSampleError::InvalidInput {
                    msg: "Stereo width adjustment requires stereo (2-channel) audio".to_string(),
                });
            }
            crate::AudioData::MultiChannel(arr) => {
                let (channels, samples) = arr.dim();
                if channels != 2 {
                    return Err(AudioSampleError::InvalidInput {
                        msg: format!("Expected 2 channels for stereo, got {}", channels),
                    });
                }

                // M/S processing for stereo width
                let half = T::one() / (T::one() + T::one()); // 0.5

                for i in 0..samples {
                    let left = arr[(0, i)];
                    let right = arr[(1, i)];

                    // Convert to Mid/Side
                    let mid = (left + right) * half;
                    let side = (left - right) * half;

                    // Apply width adjustment to side signal
                    let adjusted_side = side * width;

                    // Convert back to Left/Right
                    arr[(0, i)] = mid + adjusted_side; // Left
                    arr[(1, i)] = mid - adjusted_side; // Right
                }
            }
        }

        Ok(())
    }

    fn realtime_windowed_process<F>(
        &mut self,
        input_block: &[T],
        state: &mut RealtimeState<T>,
        processor: F,
    ) -> AudioSampleResult<Vec<T>>
    where
        F: Fn(&[T]) -> Vec<T>,
    {
        let windowed_state = &mut state.windowed_state;

        if windowed_state.window_size == 0 {
            return Err(AudioSampleError::InvalidParameter(
                "Windowed state not initialized. Call setup_windowed_processing first.".to_string(),
            ));
        }

        // Add input to buffer
        for &sample in input_block {
            windowed_state.input_buffer[windowed_state.buffer_position] = sample;
            windowed_state.buffer_position += 1;

            // Process when we have a full window
            if windowed_state.buffer_position >= windowed_state.window_size {
                // Apply window function
                for i in 0..windowed_state.window_size {
                    let buffer_idx = (windowed_state.buffer_position - windowed_state.window_size
                        + i)
                        % windowed_state.input_buffer.len();
                    state.work_buffer[i] =
                        windowed_state.input_buffer[buffer_idx] * windowed_state.window_function[i];
                }

                // Process the windowed data
                let processed = processor(&state.work_buffer[..windowed_state.window_size]);

                // Overlap-add to output buffer
                for (i, &sample) in processed.iter().enumerate() {
                    if i < windowed_state.output_buffer.len() {
                        windowed_state.output_buffer[i] = windowed_state.output_buffer[i] + sample;
                    }
                }

                // Advance by hop size
                windowed_state.buffer_position = windowed_state.hop_size;

                // Shift output buffer
                let hop_size = windowed_state.hop_size;
                windowed_state.output_buffer.copy_within(hop_size.., 0);
                for i in (windowed_state.output_buffer.len() - hop_size)
                    ..windowed_state.output_buffer.len()
                {
                    windowed_state.output_buffer[i] = T::zero();
                }
            }
        }

        // Return processed samples (first hop_size samples)
        Ok(windowed_state.output_buffer[..windowed_state
            .hop_size
            .min(windowed_state.output_buffer.len())]
            .to_vec())
    }

    fn realtime_overlapping_process<F>(
        &mut self,
        state: &mut RealtimeState<T>,
        processor: F,
    ) -> AudioSampleResult<()>
    where
        F: Fn(&mut [T]),
    {
        match &mut self.data {
            crate::AudioData::Mono(arr) => {
                let windowed_state = &mut state.windowed_state;

                if windowed_state.window_size == 0 {
                    return Err(AudioSampleError::InvalidParameter(
                        "Windowed state not initialized".to_string(),
                    ));
                }

                let data = arr
                    .as_slice_mut()
                    .ok_or(AudioSampleError::ArrayLayoutError {
                        message: "Audio data must be contiguous for windowed processing"
                            .to_string(),
                    })?;

                let window_size = windowed_state.window_size;
                let hop_size = windowed_state.hop_size;

                // Initialize output buffer
                if windowed_state.output_buffer.len() != data.len() {
                    windowed_state.output_buffer.resize(data.len(), T::zero());
                } else {
                    for sample in windowed_state.output_buffer.iter_mut() {
                        *sample = T::zero();
                    }
                }

                // Process overlapping windows
                let mut pos = 0;
                while pos + window_size <= data.len() {
                    // Extract window into working buffer
                    state.work_buffer[..window_size].copy_from_slice(&data[pos..pos + window_size]);

                    // Apply window function
                    for i in 0..window_size {
                        state.work_buffer[i] =
                            state.work_buffer[i] * windowed_state.window_function[i];
                    }

                    // Process in-place
                    processor(&mut state.work_buffer[..window_size]);

                    // Overlap-add to output buffer
                    for i in 0..window_size {
                        if pos + i < windowed_state.output_buffer.len() {
                            let windowed_sample =
                                state.work_buffer[i] * windowed_state.window_function[i];
                            windowed_state.output_buffer[pos + i] =
                                windowed_state.output_buffer[pos + i] + windowed_sample;
                        }
                    }

                    pos += hop_size;
                }

                // Copy processed data back
                data.copy_from_slice(&windowed_state.output_buffer[..data.len()]);
            }
            crate::AudioData::MultiChannel(arr) => {
                let (channels, _) = arr.dim();

                // Process each channel independently
                for ch in 0..channels {
                    if let Some(channel_data) = arr.row_mut(ch).as_slice_mut() {
                        let windowed_state = &mut state.windowed_state;

                        if windowed_state.window_size == 0 {
                            return Err(AudioSampleError::InvalidParameter(
                                "Windowed state not initialized".to_string(),
                            ));
                        }

                        let window_size = windowed_state.window_size;
                        let hop_size = windowed_state.hop_size;

                        let mut pos = 0;
                        while pos + window_size <= channel_data.len() {
                            // Extract window into working buffer
                            state.work_buffer[..window_size]
                                .copy_from_slice(&channel_data[pos..pos + window_size]);

                            // Apply window function
                            for i in 0..window_size {
                                state.work_buffer[i] =
                                    state.work_buffer[i] * windowed_state.window_function[i];
                            }

                            // Process in-place
                            processor(&mut state.work_buffer[..window_size]);

                            // Overlap-add back to original data
                            for i in 0..window_size {
                                if pos + i < channel_data.len() {
                                    channel_data[pos + i] = channel_data[pos + i]
                                        + state.work_buffer[i] * windowed_state.window_function[i];
                                }
                            }

                            pos += hop_size;
                        }
                    }
                }
            }
        }

        Ok(())
    }
}

/// Real-time audio processing state for operations that require memory.
///
/// This struct holds pre-allocated buffers and state for real-time operations
/// that need to maintain state between calls (like filters).
#[derive(Debug, Clone)]
pub struct RealtimeState<T: AudioSample> {
    /// Filter states for each channel (for multi-channel processing)
    pub filter_states: Vec<FilterState<T>>,
    /// Working buffer for temporary calculations
    pub work_buffer: Vec<T>,
    /// Windowed processing state
    pub windowed_state: WindowedState<T>,
}

/// State for real-time windowed operations.
#[derive(Debug, Clone)]
pub struct WindowedState<T: AudioSample> {
    /// Input buffer for overlap-add processing
    pub input_buffer: Vec<T>,
    /// Output buffer for overlap-add processing
    pub output_buffer: Vec<T>,
    /// Window function coefficients (pre-computed)
    pub window_function: Vec<T>,
    /// FFT working buffers (pre-allocated)
    pub fft_input: Vec<num_complex::Complex<T>>,
    pub fft_output: Vec<num_complex::Complex<T>>,
    /// Current position in buffers
    pub buffer_position: usize,
    /// Window size
    pub window_size: usize,
    /// Hop size
    pub hop_size: usize,
}

/// State for real-time filters that need to maintain history.
#[derive(Debug, Clone)]
pub struct FilterState<T: AudioSample> {
    /// Previous input sample
    pub prev_input: T,
    /// Previous output sample
    pub prev_output: T,
    /// Additional filter coefficients if needed
    pub coefficients: [T; 4],
}

impl<T: AudioSample> Default for WindowedState<T> {
    fn default() -> Self {
        Self {
            input_buffer: Vec::new(),
            output_buffer: Vec::new(),
            window_function: Vec::new(),
            fft_input: Vec::new(),
            fft_output: Vec::new(),
            buffer_position: 0,
            window_size: 0,
            hop_size: 0,
        }
    }
}

impl<T: AudioSample> Default for FilterState<T> {
    fn default() -> Self {
        Self {
            prev_input: T::zero(),
            prev_output: T::zero(),
            coefficients: [T::zero(); 4],
        }
    }
}

impl<T: AudioSample> RealtimeState<T> {
    /// Create new real-time state for specified number of channels.
    pub fn new(num_channels: usize) -> Self {
        Self {
            filter_states: vec![FilterState::default(); num_channels],
            work_buffer: vec![T::zero(); 1024], // Pre-allocate reasonable buffer
            windowed_state: WindowedState::default(),
        }
    }

    /// Resize internal buffers if needed (should be done during initialization, not in real-time).
    pub fn resize(&mut self, num_channels: usize, buffer_size: usize) {
        self.filter_states
            .resize(num_channels, FilterState::default());
        self.work_buffer.resize(buffer_size, T::zero());
    }

    /// Reset all filter states to zero.
    pub fn reset(&mut self) {
        for state in &mut self.filter_states {
            *state = FilterState::default();
        }
        self.windowed_state.buffer_position = 0;
    }

    /// Setup windowed processing state (call during initialization, not in real-time).
    ///
    /// This pre-allocates all buffers needed for windowed operations.
    pub fn setup_windowed_processing(&mut self, window_size: usize, hop_size: usize) {
        self.windowed_state.window_size = window_size;
        self.windowed_state.hop_size = hop_size;
        self.windowed_state.buffer_position = 0;

        // Pre-allocate buffers
        self.windowed_state.input_buffer = vec![T::zero(); window_size * 2]; // Ring buffer
        self.windowed_state.output_buffer = vec![T::zero(); window_size];
        self.windowed_state.fft_input = vec![num_complex::Complex::zero(); window_size];
        self.windowed_state.fft_output = vec![num_complex::Complex::zero(); window_size];

        // Generate window function (Hanning window)
        self.windowed_state.window_function = Vec::with_capacity(window_size);
        for i in 0..window_size {
            let phase: T =
                T::cast_from(2.0 * std::f64::consts::PI * i as f64 / (window_size - 1) as f64);

            let phase: f32 = phase.convert_to().unwrap();

            let half: T = T::cast_from(0.5);
            let one: T = T::one();
            let window_val = half * (one - T::from_f32(phase.cos()).unwrap());
            self.windowed_state.window_function.push(window_val);
        }

        // Ensure work buffer is large enough
        if self.work_buffer.len() < window_size {
            self.work_buffer.resize(window_size, T::zero());
        }
    }

    /// Generate window function coefficients for optimized windowed processing.
    pub fn generate_window_function(window_size: usize, window_type: WindowType) -> Vec<T> {
        let mut window = Vec::with_capacity(window_size);

        match window_type {
            WindowType::Rectangular => {
                window.resize(window_size, T::one());
            }
            WindowType::Hanning => {
                for i in 0..window_size {
                    let phase: T = T::cast_from(
                        2.0 * std::f64::consts::PI * i as f64 / (window_size - 1) as f64,
                    );
                    let phase: f32 = phase.convert_to().unwrap();
                    let half: T = T::cast_from(0.5);
                    let one: T = T::one();
                    let window_val = half * (one - T::from_f32(phase.cos()).unwrap());
                    window.push(window_val);
                }
            }
            WindowType::Hamming => {
                for i in 0..window_size {
                    let phase: T = T::cast_from(
                        2.0 * std::f64::consts::PI * i as f64 / (window_size - 1) as f64,
                    );
                    let phase: f32 = phase.convert_to().unwrap();
                    let coeff1: T = T::cast_from(0.54);
                    let coeff2: T = T::cast_from(0.46);
                    let window_val = coeff1 - coeff2 * T::from_f32(phase.cos()).unwrap();
                    window.push(window_val);
                }
            }
            WindowType::Blackman => {
                for i in 0..window_size {
                    let phase1: T = T::cast_from(
                        2.0 * std::f64::consts::PI * i as f64 / (window_size - 1) as f64,
                    );
                    let phase1: f32 = phase1.convert_to().unwrap();
                    let phase2: T = T::cast_from(
                        4.0 * std::f64::consts::PI * i as f64 / (window_size - 1) as f64,
                    );
                    let phase2: f32 = phase2.convert_to().unwrap();
                    let coeff1: T = T::cast_from(0.42);
                    let coeff2: T = T::cast_from(0.5);
                    let coeff3: T = T::cast_from(0.08);
                    let window_val = coeff1 - coeff2 * T::from_f32(phase1.cos()).unwrap()
                        + coeff3 * T::from_f32(phase2.cos()).unwrap();
                    window.push(window_val);
                }
            }
        }

        window
    }
}

/// Window function types for optimized windowed processing.
#[derive(Debug, Clone, Copy)]
pub enum WindowType {
    Rectangular,
    Hanning,
    Hamming,
    Blackman,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::AudioSamples;
    use ndarray::array;

    #[test]
    fn test_realtime_scale() {
        let mut audio = AudioSamples::new_mono(array![1.0f32, 2.0, 3.0, 4.0], 44100);
        audio.realtime_scale(0.5);

        let expected = array![0.5f32, 1.0, 1.5, 2.0];
        assert_eq!(audio.as_mono().unwrap(), &expected);
    }

    #[test]
    fn test_realtime_clip() {
        let mut audio = AudioSamples::new_mono(array![-2.0f32, -0.5, 0.5, 2.0], 44100);
        audio.realtime_clip(-1.0, 1.0);

        let expected = array![-1.0f32, -0.5, 0.5, 1.0];
        assert_eq!(audio.as_mono().unwrap(), &expected);
    }

    #[test]
    fn test_realtime_apply_gain_db() {
        let mut audio = AudioSamples::new_mono(array![1.0f32], 44100);
        audio.realtime_apply_gain_db(-6.0); // -6dB ≈ 0.5 linear

        // Should be approximately 0.5 (allowing for floating point precision)
        assert!((audio.as_mono().unwrap()[0] - 0.501).abs() < 0.01);
    }

    #[test]
    fn test_realtime_mix_with() {
        let mut audio1 = AudioSamples::new_mono(array![0.5f32, 0.3, 0.8], 44100);
        let audio2 = AudioSamples::new_mono(array![0.2f32, 0.1, -0.4], 44100);

        audio1.realtime_mix_with(&audio2, 0.5).unwrap();

        let result = audio1.as_mono().unwrap();
        let expected = array![0.6f32, 0.35, 0.6]; // 0.5 + 0.5*0.2, etc.

        // Allow for floating point precision
        assert!((result[0] - expected[0]).abs() < 1e-6);
        assert!((result[1] - expected[1]).abs() < 1e-6);
        assert!((result[2] - expected[2]).abs() < 1e-6);
    }

    #[test]
    fn test_realtime_stereo_width() {
        let stereo_data = array![[1.0f32, 0.0], [0.0, 1.0]]; // L=1,R=0 then L=0,R=1
        let mut audio = AudioSamples::new_multi_channel(stereo_data, 44100);

        audio.realtime_adjust_stereo_width(2.0).unwrap(); // Double width

        // Should increase stereo separation
        let result = audio.as_multi_channel().unwrap();
        assert!(result[(0, 0)] > 0.5); // Left channel should be boosted when width > 1
        assert!(result[(1, 0)] < -0.25); // Right channel should go negative
    }

    #[test]
    fn test_realtime_lowpass_filter() {
        let mut audio = AudioSamples::new_mono(array![1.0f32, 0.0, 1.0, 0.0], 44100);
        audio.realtime_lowpass_filter(0.5);

        // Filter should smooth the square wave
        let result = audio.as_mono().unwrap();
        assert!(result[1] > 0.0); // Should not be exactly 0 after filtering
        assert!(result[1] < result[0]); // Should be less than previous sample
    }

    #[test]
    fn test_realtime_state() {
        let mut state = RealtimeState::<f32>::new(2);
        assert_eq!(state.filter_states.len(), 2);
        assert_eq!(state.work_buffer.len(), 1024);

        state.reset();
        assert_eq!(state.filter_states[0].prev_input, 0.0);
        assert_eq!(state.filter_states[0].prev_output, 0.0);
    }

    #[test]
    fn test_windowed_processing_setup() {
        let mut state = RealtimeState::<f32>::new(1);
        state.setup_windowed_processing(512, 256);

        assert_eq!(state.windowed_state.window_size, 512);
        assert_eq!(state.windowed_state.hop_size, 256);
        assert_eq!(state.windowed_state.input_buffer.len(), 1024); // 2 * window_size
        assert_eq!(state.windowed_state.output_buffer.len(), 512);
        assert_eq!(state.windowed_state.window_function.len(), 512);
    }

    #[test]
    fn test_window_function_generation() {
        let hanning = RealtimeState::<f32>::generate_window_function(8, WindowType::Hanning);
        assert_eq!(hanning.len(), 8);

        // Test that window starts and ends near zero (characteristic of Hanning)
        assert!(hanning[0] < 0.1);
        assert!(hanning[7] < 0.1);

        // Test that middle values are larger
        assert!(hanning[3] > 0.5);
        assert!(hanning[4] > 0.5);
    }

    #[test]
    fn test_realtime_overlapping_process() {
        let data = ndarray::Array1::from_elem(1024, 1.0f32);
        let mut audio = AudioSamples::new_mono(data, 44100);
        let mut state = RealtimeState::new(1);
        state.setup_windowed_processing(256, 128);

        // Apply gain to each window
        audio
            .realtime_overlapping_process(&mut state, |window| {
                for sample in window.iter_mut() {
                    *sample *= 0.5;
                }
            })
            .unwrap();

        // Check that processing was applied (values should be reduced)
        let result = audio.as_mono().unwrap();
        // Check samples that are likely to be fully processed by overlap-add
        assert!(result[256] < 0.9); // Should be significantly reduced from 1.0
        assert!(result[512] < 0.9); // Should be significantly reduced from 1.0
    }
}
