//! Multi-stream audio mixer for combining multiple audio sources.

use super::error::{PlaybackError, PlaybackResult};
use crate::{AudioSample, AudioSamples};
use parking_lot::{Mutex, RwLock};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Configuration for the audio mixer.
#[derive(Debug, Clone)]
pub struct MixerConfig {
    /// Sample rate for the mixer output
    pub sample_rate: u32,

    /// Number of output channels
    pub output_channels: usize,

    /// Maximum number of input channels
    pub max_input_channels: usize,

    /// Buffer size for mixing operations
    pub buffer_size: usize,

    /// Master volume level (0.0 to 1.0)
    pub master_volume: f64,

    /// Whether to automatically normalize output to prevent clipping
    pub auto_normalize: bool,

    /// Normalization threshold (above which normalization kicks in)
    pub normalization_threshold: f64,

    /// Mix algorithm to use
    pub mix_algorithm: MixAlgorithm,

    /// Whether to enable per-channel effects processing
    pub enable_effects: bool,
}

impl Default for MixerConfig {
    fn default() -> Self {
        Self {
            sample_rate: 44100,
            output_channels: 2,
            max_input_channels: 32,
            buffer_size: 1024,
            master_volume: 0.8,
            auto_normalize: true,
            normalization_threshold: 0.95,
            mix_algorithm: MixAlgorithm::Linear,
            enable_effects: false,
        }
    }
}

impl MixerConfig {
    /// Create a low-latency mixer configuration
    pub fn low_latency() -> Self {
        Self {
            sample_rate: 48000,
            output_channels: 2,
            max_input_channels: 8,
            buffer_size: 256,
            master_volume: 0.8,
            auto_normalize: false, // Disable for lower latency
            normalization_threshold: 0.95,
            mix_algorithm: MixAlgorithm::Linear,
            enable_effects: false,
        }
    }

    /// Create a high-quality mixer configuration
    pub fn high_quality() -> Self {
        Self {
            sample_rate: 96000,
            output_channels: 8, // Surround sound
            max_input_channels: 64,
            buffer_size: 2048,
            master_volume: 0.8,
            auto_normalize: true,
            normalization_threshold: 0.90,
            mix_algorithm: MixAlgorithm::WeightedAverage,
            enable_effects: true,
        }
    }
}

/// Algorithm used for mixing multiple audio streams
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MixAlgorithm {
    /// Simple linear addition
    Linear,
    /// Weighted average based on channel volumes
    WeightedAverage,
    /// RMS-based mixing to preserve perceived loudness
    Rms,
    /// Peak-preserving mix with automatic gain control
    PeakPreserving,
}

/// Configuration for a single mixer channel
#[derive(Debug, Clone)]
pub struct ChannelConfig {
    /// Channel volume (0.0 to 1.0)
    pub volume: f64,

    /// Pan position (-1.0 = full left, 0.0 = center, 1.0 = full right)
    pub pan: f64,

    /// Whether this channel is muted
    pub muted: bool,

    /// Whether this channel is soloed (if any channel is soloed, only soloed channels play)
    pub solo: bool,

    /// Input channel mapping (which input channels map to which output channels)
    pub input_mapping: Vec<usize>,

    /// Output channel mapping
    pub output_mapping: Vec<usize>,

    /// Per-channel gain adjustment in dB
    pub gain_db: f64,

    /// High-pass filter cutoff frequency (Hz, 0 = disabled)
    pub highpass_freq: f64,

    /// Low-pass filter cutoff frequency (Hz, 0 = disabled)
    pub lowpass_freq: f64,
}

impl Default for ChannelConfig {
    fn default() -> Self {
        Self {
            volume: 1.0,
            pan: 0.0,
            muted: false,
            solo: false,
            input_mapping: vec![0, 1], // Stereo by default
            output_mapping: vec![0, 1],
            gain_db: 0.0,
            highpass_freq: 0.0,
            lowpass_freq: 0.0,
        }
    }
}

impl ChannelConfig {
    /// Create a mono channel configuration
    pub fn mono(output_channel: usize) -> Self {
        Self {
            input_mapping: vec![0],
            output_mapping: vec![output_channel],
            ..Default::default()
        }
    }

    /// Create a stereo channel configuration
    pub fn stereo() -> Self {
        Self {
            input_mapping: vec![0, 1],
            output_mapping: vec![0, 1],
            ..Default::default()
        }
    }

    /// Set the volume for this channel
    pub fn with_volume(mut self, volume: f64) -> Self {
        self.volume = volume.clamp(0.0, 1.0);
        self
    }

    /// Set the pan for this channel
    pub fn with_pan(mut self, pan: f64) -> Self {
        self.pan = pan.clamp(-1.0, 1.0);
        self
    }

    /// Set gain in decibels
    pub fn with_gain_db(mut self, gain_db: f64) -> Self {
        self.gain_db = gain_db.clamp(-60.0, 20.0);
        self
    }
}

/// Represents a single mixer channel
pub struct MixerChannel<T: AudioSample>
where
    T: AudioSampleExt + Copy,
{
    id: usize,
    config: ChannelConfig,
    buffer: Vec<T>,
    last_activity: Instant,
    peak_level: f64,
    rms_level: f64,

    // Simple filters (high-pass and low-pass)
    hp_filter_state: [f64; 2], // Previous input and output for 1-pole HP
    lp_filter_state: [f64; 2], // Previous input and output for 1-pole LP
}

impl<T: AudioSample> MixerChannel<T>
where
    T: AudioSampleExt + Copy,
{
    fn new(id: usize, config: ChannelConfig) -> Self {
        Self {
            id,
            config,
            buffer: Vec::new(),
            last_activity: Instant::now(),
            peak_level: 0.0,
            rms_level: 0.0,
            hp_filter_state: [0.0; 2],
            lp_filter_state: [0.0; 2],
        }
    }

    /// Update the channel configuration
    pub fn set_config(&mut self, config: ChannelConfig) {
        self.config = config;
    }

    /// Get the current configuration
    pub fn config(&self) -> &ChannelConfig {
        &self.config
    }

    /// Get the peak level (0.0 to 1.0)
    pub fn peak_level(&self) -> f64 {
        self.peak_level
    }

    /// Get the RMS level (0.0 to 1.0)
    pub fn rms_level(&self) -> f64 {
        self.rms_level
    }

    /// Check if the channel is active (has recent audio)
    pub fn is_active(&self) -> bool {
        self.last_activity.elapsed() < Duration::from_millis(100)
    }

    /// Process audio data for this channel with volume, pan, and filtering
    fn process_audio(&mut self, input: &[T], sample_rate: f64) -> Vec<f64> {
        if self.config.muted {
            return vec![0.0; input.len()];
        }

        let mut output: Vec<f64> = Vec::with_capacity(input.len());
        let gain_linear = 10.0_f64.powf(self.config.gain_db / 20.0);
        let volume = self.config.volume * gain_linear;

        // Convert to f64 and apply initial gain
        let mut samples: Vec<f64> = input
            .iter()
            .map(|&sample| sample.cast_to_f64() * volume)
            .collect();

        // Apply high-pass filter if enabled
        if self.config.highpass_freq > 0.0 {
            let cutoff = self.config.highpass_freq / sample_rate;
            let rc = 1.0 / (2.0 * std::f64::consts::PI * cutoff);
            let dt = 1.0 / sample_rate;
            let alpha = rc / (rc + dt);

            for i in 0..samples.len() {
                let sample = samples[i];
                let filtered = alpha * (self.hp_filter_state[1] + sample - self.hp_filter_state[0]);
                samples[i] = filtered;
                self.hp_filter_state[0] = sample;
                self.hp_filter_state[1] = filtered;
            }
        }

        // Apply low-pass filter if enabled
        if self.config.lowpass_freq > 0.0 {
            let cutoff = self.config.lowpass_freq / sample_rate;
            let rc = 1.0 / (2.0 * std::f64::consts::PI * cutoff);
            let dt = 1.0 / sample_rate;
            let alpha = dt / (rc + dt);

            for i in 0..samples.len() {
                let sample = samples[i];
                let filtered = self.lp_filter_state[1] + alpha * (sample - self.lp_filter_state[1]);
                samples[i] = filtered;
                self.lp_filter_state[1] = filtered;
            }
        }

        // Update levels
        if !samples.is_empty() {
            self.peak_level = samples.iter().map(|&s| s.abs()).fold(0.0, f64::max);

            self.rms_level =
                (samples.iter().map(|&s| s * s).sum::<f64>() / samples.len() as f64).sqrt();

            self.last_activity = Instant::now();
        }

        samples
    }
}

/// Multi-stream audio mixer
pub struct AudioMixer<T: AudioSample>
where
    T: AudioSampleExt + Copy,
{
    config: MixerConfig,
    channels: HashMap<usize, Arc<Mutex<MixerChannel<T>>>>,
    next_channel_id: usize,

    // Mixing state
    mix_buffer: Vec<f64>,
    output_buffer: Vec<T>,

    // Master controls
    master_volume: Arc<Mutex<f64>>,
    master_mute: Arc<Mutex<bool>>,

    // Statistics
    total_samples_mixed: u64,
    active_channels: usize,
    peak_output_level: f64,

    // Solo state
    has_solo_channels: bool,
}

impl<T: AudioSample> AudioMixer<T>
where
    T: AudioSampleExt + Copy,
{
    /// Create a new audio mixer with the given configuration
    pub fn new(config: MixerConfig) -> Self {
        Self {
            config: config.clone(),
            channels: HashMap::new(),
            next_channel_id: 0,
            mix_buffer: vec![0.0; config.buffer_size * config.output_channels],
            output_buffer: vec![T::default(); config.buffer_size * config.output_channels],
            master_volume: Arc::new(Mutex::new(config.master_volume)),
            master_mute: Arc::new(Mutex::new(false)),
            total_samples_mixed: 0,
            active_channels: 0,
            peak_output_level: 0.0,
            has_solo_channels: false,
        }
    }

    /// Create a mixer with default configuration
    pub fn with_default_config() -> Self {
        Self::new(MixerConfig::default())
    }

    /// Create a low-latency mixer
    pub fn low_latency() -> Self {
        Self::new(MixerConfig::low_latency())
    }

    /// Create a high-quality mixer
    pub fn high_quality() -> Self {
        Self::new(MixerConfig::high_quality())
    }

    /// Add a new channel to the mixer
    pub fn add_channel(&mut self, config: ChannelConfig) -> usize {
        let id = self.next_channel_id;
        self.next_channel_id += 1;

        let channel = Arc::new(Mutex::new(MixerChannel::new(id, config)));
        self.channels.insert(id, channel);

        self.update_solo_state();
        id
    }

    /// Remove a channel from the mixer
    pub fn remove_channel(&mut self, channel_id: usize) -> bool {
        let removed = self.channels.remove(&channel_id).is_some();
        if removed {
            self.update_solo_state();
        }
        removed
    }

    /// Get a reference to a channel
    pub fn get_channel(&self, channel_id: usize) -> Option<Arc<Mutex<MixerChannel<T>>>> {
        self.channels.get(&channel_id).cloned()
    }

    /// Update the configuration for a channel
    pub fn update_channel_config(
        &mut self,
        channel_id: usize,
        config: ChannelConfig,
    ) -> PlaybackResult<()> {
        if let Some(channel) = self.channels.get(&channel_id) {
            let mut ch = channel.lock();
            ch.set_config(config);
            drop(ch);
            self.update_solo_state();
            Ok(())
        } else {
            Err(PlaybackError::InvalidConfig(format!(
                "Channel {} not found",
                channel_id
            )))
        }
    }

    /// Add audio data to a specific channel
    pub fn add_channel_audio(
        &mut self,
        channel_id: usize,
        audio: &AudioSamples<T>,
    ) -> PlaybackResult<()> {
        if let Some(_channel) = self.channels.get(&channel_id) {
            // TODO: Implement once AudioSamples has a method to access raw data
            // let mut ch = channel.lock();
            // ch.buffer.extend_from_slice(audio.as_slice());
            Ok(())
        } else {
            Err(PlaybackError::InvalidConfig(format!(
                "Channel {} not found",
                channel_id
            )))
        }
    }

    /// Mix all channels and produce output audio
    pub fn mix(&mut self) -> PlaybackResult<AudioSamples<T>> {
        let buffer_size = self.config.buffer_size;
        let output_channels = self.config.output_channels;
        let sample_rate = self.config.sample_rate as f64;

        // Clear mix buffer
        self.mix_buffer.fill(0.0);
        self.active_channels = 0;

        // Check for solo channels
        self.update_solo_state();

        // Collect channel data first to avoid borrowing conflicts
        let mut processed_channels = Vec::new();

        for (_, channel_arc) in &self.channels {
            let mut channel = channel_arc.lock();

            // Skip if muted, or if other channels are soloed and this isn't
            if channel.config.muted || (self.has_solo_channels && !channel.config.solo) {
                continue;
            }

            // Check if we have enough data
            let samples_needed = buffer_size * channel.config.input_mapping.len();
            if channel.buffer.len() < samples_needed {
                continue;
            }

            // Extract samples for this mix
            let input_samples: Vec<T> = channel.buffer.drain(..samples_needed).collect();

            // Process the audio (apply volume, pan, filters)
            let processed = channel.process_audio(&input_samples, sample_rate);
            let config = channel.config.clone();
            let is_active = channel.is_active();

            processed_channels.push((processed, config, is_active));
        }

        // Now mix the processed channels
        for (processed, config, is_active) in processed_channels {
            self.mix_channel_into_buffer(&processed, &config, buffer_size);
            if is_active {
                self.active_channels += 1;
            }
        }

        // Apply master volume and convert to output format
        self.apply_master_processing()?;

        // Create AudioSamples from output buffer
        let output_data = self.output_buffer[..buffer_size * output_channels].to_vec();
        let array = ndarray::Array2::from_shape_vec((buffer_size, output_channels), output_data)
            .map_err(|e| {
                PlaybackError::InvalidConfig(format!("Failed to create output array: {}", e))
            })?;

        use crate::repr::AudioData;
        let audio_samples =
            AudioSamples::new(AudioData::MultiChannel(array), self.config.sample_rate);

        self.total_samples_mixed += buffer_size as u64;
        Ok(audio_samples)
    }

    /// Mix a single channel's audio into the mix buffer
    fn mix_channel_into_buffer(
        &mut self,
        processed: &[f64],
        config: &ChannelConfig,
        buffer_size: usize,
    ) {
        let output_channels = self.config.output_channels;
        let input_channels = config.input_mapping.len();

        match self.config.mix_algorithm {
            MixAlgorithm::Linear => {
                // Simple linear addition
                for frame in 0..buffer_size {
                    for (input_ch, &output_ch) in config.output_mapping.iter().enumerate() {
                        if output_ch < output_channels && input_ch < input_channels {
                            let input_idx = frame * input_channels + input_ch;
                            let output_idx = frame * output_channels + output_ch;

                            if input_idx < processed.len() && output_idx < self.mix_buffer.len() {
                                let sample = processed[input_idx];

                                // Apply pan if stereo output
                                let (left_gain, right_gain) = if output_channels >= 2 {
                                    self.calculate_pan_gains(config.pan)
                                } else {
                                    (1.0, 1.0)
                                };

                                match output_ch {
                                    0 => self.mix_buffer[output_idx] += sample * left_gain,
                                    1 if output_channels >= 2 => {
                                        self.mix_buffer[output_idx] += sample * right_gain
                                    }
                                    _ => self.mix_buffer[output_idx] += sample,
                                }
                            }
                        }
                    }
                }
            }

            MixAlgorithm::WeightedAverage => {
                // Weight by channel volume and active channel count
                let weight = if self.active_channels > 0 {
                    config.volume / self.active_channels as f64
                } else {
                    config.volume
                };

                for frame in 0..buffer_size {
                    for (input_ch, &output_ch) in config.output_mapping.iter().enumerate() {
                        if output_ch < output_channels && input_ch < input_channels {
                            let input_idx = frame * input_channels + input_ch;
                            let output_idx = frame * output_channels + output_ch;

                            if input_idx < processed.len() && output_idx < self.mix_buffer.len() {
                                let sample = processed[input_idx] * weight;
                                let (left_gain, right_gain) = if output_channels >= 2 {
                                    self.calculate_pan_gains(config.pan)
                                } else {
                                    (1.0, 1.0)
                                };

                                match output_ch {
                                    0 => self.mix_buffer[output_idx] += sample * left_gain,
                                    1 if output_channels >= 2 => {
                                        self.mix_buffer[output_idx] += sample * right_gain
                                    }
                                    _ => self.mix_buffer[output_idx] += sample,
                                }
                            }
                        }
                    }
                }
            }

            _ => {
                // For now, fall back to linear for other algorithms
                self.mix_channel_into_buffer(processed, config, buffer_size);
            }
        }
    }

    /// Calculate stereo pan gains from pan value (-1.0 to 1.0)
    fn calculate_pan_gains(&self, pan: f64) -> (f64, f64) {
        let pan_clamped = pan.clamp(-1.0, 1.0);
        let left_gain = ((1.0 - pan_clamped) * 0.5).sqrt();
        let right_gain = ((1.0 + pan_clamped) * 0.5).sqrt();
        (left_gain, right_gain)
    }

    /// Apply master volume, normalization, and convert to output format
    fn apply_master_processing(&mut self) -> PlaybackResult<()> {
        let master_volume = *self.master_volume.lock();
        let master_mute = *self.master_mute.lock();

        if master_mute {
            self.output_buffer.fill(T::default());
            return Ok(());
        }

        // Find peak level for normalization
        self.peak_output_level = self.mix_buffer.iter().map(|&s| s.abs()).fold(0.0, f64::max);

        // Apply auto-normalization if enabled and needed
        let normalization_gain = if self.config.auto_normalize
            && self.peak_output_level > self.config.normalization_threshold
        {
            self.config.normalization_threshold / self.peak_output_level
        } else {
            1.0
        };

        let total_gain = master_volume * normalization_gain;

        // Convert to output format
        for (i, &sample) in self.mix_buffer.iter().enumerate() {
            if i < self.output_buffer.len() {
                let final_sample = sample * total_gain;
                self.output_buffer[i] = T::cast_from_f64(final_sample.clamp(-1.0, 1.0));
            }
        }

        Ok(())
    }

    /// Update the solo state based on current channels
    fn update_solo_state(&mut self) {
        self.has_solo_channels = self.channels.values().any(|ch| ch.lock().config.solo);
    }

    /// Set master volume
    pub fn set_master_volume(&self, volume: f64) {
        *self.master_volume.lock() = volume.clamp(0.0, 1.0);
    }

    /// Get master volume
    pub fn master_volume(&self) -> f64 {
        *self.master_volume.lock()
    }

    /// Set master mute
    pub fn set_master_mute(&self, mute: bool) {
        *self.master_mute.lock() = mute;
    }

    /// Get master mute state
    pub fn is_master_muted(&self) -> bool {
        *self.master_mute.lock()
    }

    /// Get the number of active channels
    pub fn active_channel_count(&self) -> usize {
        self.active_channels
    }

    /// Get the current peak output level
    pub fn peak_output_level(&self) -> f64 {
        self.peak_output_level
    }

    /// Get total samples mixed
    pub fn total_samples_mixed(&self) -> u64 {
        self.total_samples_mixed
    }

    /// Get list of all channel IDs
    pub fn channel_ids(&self) -> Vec<usize> {
        self.channels.keys().cloned().collect()
    }
}

// Extension trait for AudioSample to handle f64 conversion
trait AudioSampleExt {
    fn cast_to_f64(self) -> f64;
    fn cast_from_f64(value: f64) -> Self;
}

impl AudioSampleExt for f32 {
    fn cast_to_f64(self) -> f64 {
        self as f64
    }

    fn cast_from_f64(value: f64) -> Self {
        value as f32
    }
}

impl AudioSampleExt for f64 {
    fn cast_to_f64(self) -> f64 {
        self
    }

    fn cast_from_f64(value: f64) -> Self {
        value
    }
}

impl AudioSampleExt for i16 {
    fn cast_to_f64(self) -> f64 {
        self as f64 / i16::MAX as f64
    }

    fn cast_from_f64(value: f64) -> Self {
        (value * i16::MAX as f64).clamp(i16::MIN as f64, i16::MAX as f64) as i16
    }
}

impl AudioSampleExt for i32 {
    fn cast_to_f64(self) -> f64 {
        self as f64 / i32::MAX as f64
    }

    fn cast_from_f64(value: f64) -> Self {
        (value * i32::MAX as f64).clamp(i32::MIN as f64, i32::MAX as f64) as i32
    }
}

impl AudioSampleExt for crate::I24 {
    fn cast_to_f64(self) -> f64 {
        let value: i32 = self.to_i32();
        value as f64 / (1i32 << 23) as f64
    }

    fn cast_from_f64(value: f64) -> Self {
        let scaled = (value * (1i32 << 23) as f64).clamp(-8388608.0, 8388607.0) as i32;
        crate::I24::saturating_from_i32(scaled)
    }
}
