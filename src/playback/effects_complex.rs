//! Real-time audio effects processing engine.
//!
//! This module provides a real-time effects processing engine that leverages
//! the comprehensive audio operations available in the `operations` module.
//! Effects are applied using the existing AudioSamples processing infrastructure
//! rather than implementing separate processing logic.

use super::error::{PlaybackError, PlaybackResult};
use crate::operations::*;
use crate::{AudioSample, AudioSamples};
use parking_lot::Mutex;
use std::collections::HashMap;
use std::sync::Arc;

/// Configuration for the effects engine
#[derive(Debug, Clone)]
pub struct EffectsConfig {
    /// Sample rate for effects processing
    pub sample_rate: u32,

    /// Number of channels to process
    pub channels: usize,

    /// Buffer size for processing
    pub buffer_size: usize,

    /// Whether to enable parallel processing of effects
    pub parallel_processing: bool,

    /// Maximum number of effects per chain
    pub max_effects_per_chain: usize,

    /// Whether to enable wet/dry mixing
    pub enable_wet_dry_mix: bool,
}

impl Default for EffectsConfig {
    fn default() -> Self {
        Self {
            sample_rate: 44100,
            channels: 2,
            buffer_size: 1024,
            parallel_processing: false, // Disable by default for deterministic processing
            max_effects_per_chain: 16,
            enable_wet_dry_mix: true,
        }
    }
}

/// Base trait for all audio effects that leverages the operations infrastructure.
///
/// Effects are applied using AudioSamples instances and the comprehensive
/// operations available in the operations module, providing consistent
/// behavior and leveraging optimized implementations.
pub trait AudioEffect<T: AudioSample>: Send + Sync
where
    T: AudioSample,
{
    /// Apply the effect to AudioSamples using the operations infrastructure.
    ///
    /// This method leverages the existing audio operations for consistent
    /// behavior and performance optimization.
    fn apply(&mut self, audio: AudioSamples<T>) -> PlaybackResult<AudioSamples<T>>;

    /// Reset the effect state
    fn reset(&mut self);

    /// Get the effect name
    fn name(&self) -> &str;

    /// Get the latency introduced by this effect in samples
    fn latency_samples(&self) -> usize {
        0
    }

    /// Check if the effect is enabled
    fn is_enabled(&self) -> bool {
        true
    }

    /// Enable or disable the effect
    fn set_enabled(&mut self, enabled: bool);

    /// Get the wet/dry mix level (0.0 = all dry, 1.0 = all wet)
    fn wet_level(&self) -> f64 {
        1.0
    }

    /// Set the wet/dry mix level
    fn set_wet_level(&mut self, level: f64);

    /// Apply wet/dry mixing to the processed audio.
    ///
    /// This is a default implementation that mixes the dry (original) and
    /// wet (processed) signals according to the wet_level setting.
    fn apply_wet_dry_mix(
        &self,
        dry: &AudioSamples<T>,
        wet: AudioSamples<T>,
    ) -> PlaybackResult<AudioSamples<T>> {
        if !self.is_enabled() {
            return Ok(dry.clone());
        }

        let wet_level = self.wet_level();
        if wet_level >= 1.0 {
            return Ok(wet);
        }
        if wet_level <= 0.0 {
            return Ok(dry.clone());
        }

        // For now, just return the wet signal - proper mixing would require trait bounds
        // TODO: Implement proper wet/dry mixing when trait bounds are resolved
        Ok(wet)
    }
}

/// Gain effect that leverages the AudioProcessing operations.
pub struct GainEffect {
    name: String,
    gain_db: f64,
    enabled: bool,
    wet_level: f64,
}

impl GainEffect {
    pub fn new(gain_db: f64) -> Self {
        Self {
            name: "Gain".to_string(),
            gain_db,
            enabled: true,
            wet_level: 1.0,
        }
    }

    pub fn set_gain_db(&mut self, gain_db: f64) {
        self.gain_db = gain_db;
    }

    pub fn gain_db(&self) -> f64 {
        self.gain_db
    }
}

impl<T: AudioSample> AudioEffect<T> for GainEffect
where
    T: AudioSample + Copy + Into<f64> + From<f64>,
{
    fn apply(&mut self, mut audio: AudioSamples<T>) -> PlaybackResult<AudioSamples<T>> {
        if !self.enabled {
            return Ok(audio);
        }

        // Convert dB to linear scale factor and apply directly
        let gain_linear = 10.0_f64.powf(self.gain_db / 20.0);
        let dry = audio.clone();

        // Apply gain to all samples using the apply method
        let mut audio = audio;
        audio
            .apply(|sample| {
                let sample_f64: f64 = sample.into();
                T::from(sample_f64 * gain_linear)
            })
            .map_err(|e| {
                PlaybackError::EffectsProcessing(format!("Gain processing failed: {}", e))
            })?;

        // Apply wet/dry mixing
        self.apply_wet_dry_mix(&dry, audio)
    }

    fn reset(&mut self) {
        // Gain effect has no internal state to reset
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn is_enabled(&self) -> bool {
        self.enabled
    }

    fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    fn wet_level(&self) -> f64 {
        self.wet_level
    }

    fn set_wet_level(&mut self, level: f64) {
        self.wet_level = level.clamp(0.0, 1.0);
    }
}

/// Compressor effect that leverages the AudioDynamicRange operations.
pub struct CompressorEffect {
    name: String,
    config: CompressorConfig,
    enabled: bool,
    wet_level: f64,
}

impl CompressorEffect {
    pub fn new(config: CompressorConfig) -> Self {
        Self {
            name: "Compressor".to_string(),
            config,
            enabled: true,
            wet_level: 1.0,
        }
    }

    pub fn vocal() -> Self {
        Self::new(CompressorConfig::vocal())
    }

    pub fn drum() -> Self {
        Self::new(CompressorConfig::drum())
    }

    pub fn bus() -> Self {
        Self::new(CompressorConfig::bus())
    }

    pub fn set_threshold_db(&mut self, threshold_db: f64) {
        self.config.threshold_db = threshold_db;
    }

    pub fn set_ratio(&mut self, ratio: f64) {
        self.config.ratio = ratio;
    }

    pub fn set_attack_ms(&mut self, attack_ms: f64) {
        self.config.attack_ms = attack_ms;
    }

    pub fn set_release_ms(&mut self, release_ms: f64) {
        self.config.release_ms = release_ms;
    }

    pub fn config(&self) -> &CompressorConfig {
        &self.config
    }
}

impl<T: AudioSample> AudioEffect<T> for CompressorEffect
where
    T: AudioSample + Copy + Into<f64> + From<f64>,
{
    fn apply(&mut self, mut audio: AudioSamples<T>) -> PlaybackResult<AudioSamples<T>> {
        if !self.enabled {
            return Ok(audio);
        }

        let dry = audio.clone();
        // For now, just pass through - real compression would be implemented here
        // using the operations module when trait bounds are properly resolved

        // Apply wet/dry mixing
        self.apply_wet_dry_mix(&dry, audio)
    }

    fn reset(&mut self) {
        // Compressor state is managed internally by the operations
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn is_enabled(&self) -> bool {
        self.enabled
    }

    fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    fn wet_level(&self) -> f64 {
        self.wet_level
    }

    fn set_wet_level(&mut self, level: f64) {
        self.wet_level = level.clamp(0.0, 1.0);
    }
}

/// Low-pass filter effect that leverages the AudioProcessing operations.
pub struct LowPassEffect {
    name: String,
    cutoff_hz: f64,
    sample_rate: u32,
    enabled: bool,
    wet_level: f64,
}

impl LowPassEffect {
    pub fn new(cutoff_hz: f64, sample_rate: u32) -> Self {
        Self {
            name: "Low Pass Filter".to_string(),
            cutoff_hz,
            sample_rate,
            enabled: true,
            wet_level: 1.0,
        }
    }

    pub fn set_cutoff_hz(&mut self, cutoff_hz: f64) {
        self.cutoff_hz = cutoff_hz;
    }

    pub fn cutoff_hz(&self) -> f64 {
        self.cutoff_hz
    }
}

impl<T: AudioSample> AudioEffect<T> for LowPassEffect
where
    T: AudioSample + Copy + Into<f64> + From<f64>,
{
    fn apply(&mut self, mut audio: AudioSamples<T>) -> PlaybackResult<AudioSamples<T>> {
        if !self.enabled {
            return Ok(audio);
        }

        let dry = audio.clone();
        // For now, just pass through - real filtering would be implemented here
        // using the operations module when trait bounds are properly resolved

        // Apply wet/dry mixing
        self.apply_wet_dry_mix(&dry, audio)
    }

    fn reset(&mut self) {
        // Filter state is managed internally by the operations
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn is_enabled(&self) -> bool {
        self.enabled
    }

    fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    fn wet_level(&self) -> f64 {
        self.wet_level
    }

    fn set_wet_level(&mut self, level: f64) {
        self.wet_level = level.clamp(0.0, 1.0);
    }
}

/// Simple delay/echo effect
pub struct DelayEffect {
    name: String,
    delay_buffer: Vec<f64>,
    delay_samples: usize,
    write_index: usize,
    feedback: f64,
    wet_level: f64,
    enabled: bool,
}

impl DelayEffect {
    pub fn new(delay_ms: f64, sample_rate: u32, feedback: f64) -> Self {
        let delay_samples = (delay_ms * sample_rate as f64 / 1000.0) as usize;
        let buffer_size = delay_samples.max(1);

        Self {
            name: "Delay".to_string(),
            delay_buffer: vec![0.0; buffer_size],
            delay_samples,
            write_index: 0,
            feedback: feedback.clamp(0.0, 0.95),
            wet_level: 0.3,
            enabled: true,
        }
    }

    pub fn set_delay_ms(&mut self, delay_ms: f64, sample_rate: u32) {
        let new_delay_samples = (delay_ms * sample_rate as f64 / 1000.0) as usize;
        if new_delay_samples != self.delay_samples {
            self.delay_samples = new_delay_samples;
            let new_buffer_size = new_delay_samples.max(1);
            self.delay_buffer.resize(new_buffer_size, 0.0);
            self.write_index = 0; // Reset to avoid index issues
        }
    }

    pub fn set_feedback(&mut self, feedback: f64) {
        self.feedback = feedback.clamp(0.0, 0.95);
    }

    pub fn feedback(&self) -> f64 {
        self.feedback
    }
}

impl<T: AudioSample> AudioEffect<T> for DelayEffect
where
    T: Into<f64> + From<f64>,
{
    fn apply(&mut self, audio: AudioSamples<T>) -> PlaybackResult<AudioSamples<T>> {
        if !self.enabled {
            return Ok(audio);
        }

        // For now, just pass through - proper delay would need access to raw sample data
        // This can be implemented later when we have proper methods for accessing sample data
        Ok(audio)
    }

    fn reset(&mut self) {
        self.delay_buffer.fill(0.0);
        self.write_index = 0;
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn latency_samples(&self) -> usize {
        0 // Delay doesn't add processing latency, just decorrelation
    }

    fn is_enabled(&self) -> bool {
        self.enabled
    }

    fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    fn wet_level(&self) -> f64 {
        self.wet_level
    }

    fn set_wet_level(&mut self, level: f64) {
        self.wet_level = level.clamp(0.0, 1.0);
    }
}

/// Effect chain that processes multiple effects in sequence
pub struct EffectChain<T: AudioSample> {
    effects: Vec<Box<dyn AudioEffect<T>>>,
    temp_buffers: Vec<Vec<T>>,
    bypass: bool,
}

impl<T: AudioSample> EffectChain<T> {
    pub fn new() -> Self {
        Self {
            effects: Vec::new(),
            temp_buffers: Vec::new(),
            bypass: false,
        }
    }

    /// Add an effect to the end of the chain
    pub fn add_effect(&mut self, effect: Box<dyn AudioEffect<T>>) {
        self.effects.push(effect);
        // Add a temporary buffer for this effect
        self.temp_buffers.push(Vec::new());
    }

    /// Remove an effect by index
    pub fn remove_effect(&mut self, index: usize) -> Option<Box<dyn AudioEffect<T>>> {
        if index < self.effects.len() {
            self.temp_buffers.remove(index);
            Some(self.effects.remove(index))
        } else {
            None
        }
    }

    /// Get a mutable reference to an effect by index
    pub fn get_effect_mut(&mut self, index: usize) -> Option<&mut Box<dyn AudioEffect<T>>> {
        self.effects.get_mut(index)
    }

    /// Get the number of effects in the chain
    pub fn effect_count(&self) -> usize {
        self.effects.len()
    }

    /// Set bypass state
    pub fn set_bypass(&mut self, bypass: bool) {
        self.bypass = bypass;
    }

    /// Check if bypassed
    pub fn is_bypassed(&self) -> bool {
        self.bypass
    }

    /// Process audio through the effect chain
    pub fn process(&mut self, input_audio: AudioSamples<T>) -> PlaybackResult<AudioSamples<T>> {
        if self.bypass || self.effects.is_empty() {
            return Ok(input_audio);
        }

        let mut current_audio = input_audio;

        // Process through each effect
        for effect in self.effects.iter_mut() {
            current_audio = effect.apply(current_audio)?;
        }

        Ok(current_audio)
    }

    /// Reset all effects in the chain
    pub fn reset(&mut self) {
        for effect in &mut self.effects {
            effect.reset();
        }
    }

    /// Get total latency of the chain
    pub fn total_latency_samples(&self) -> usize {
        self.effects.iter().map(|e| e.latency_samples()).sum()
    }
}

/// Multi-channel effects processor
pub struct EffectsEngine<T: AudioSample> {
    config: EffectsConfig,
    chains: HashMap<usize, EffectChain<T>>,
    next_chain_id: usize,
    temp_channel_buffers: Vec<Vec<T>>,
}

impl<T: AudioSample> EffectsEngine<T> {
    /// Create a new effects engine
    pub fn new(config: EffectsConfig) -> Self {
        Self {
            temp_channel_buffers: vec![Vec::new(); config.channels],
            config,
            chains: HashMap::new(),
            next_chain_id: 0,
        }
    }

    /// Create an effects engine with default configuration
    pub fn with_default_config() -> Self {
        Self::new(EffectsConfig::default())
    }

    /// Create a new effect chain
    pub fn create_chain(&mut self) -> usize {
        let id = self.next_chain_id;
        self.next_chain_id += 1;

        let chain = EffectChain::new();
        self.chains.insert(id, chain);
        id
    }

    /// Remove an effect chain
    pub fn remove_chain(&mut self, chain_id: usize) -> bool {
        self.chains.remove(&chain_id).is_some()
    }

    /// Get a mutable reference to an effect chain
    pub fn get_chain_mut(&mut self, chain_id: usize) -> Option<&mut EffectChain<T>> {
        self.chains.get_mut(&chain_id)
    }

    /// Process audio through a specific effect chain
    pub fn process_chain(
        &mut self,
        chain_id: usize,
        audio: AudioSamples<T>,
    ) -> PlaybackResult<AudioSamples<T>> {
        if let Some(chain) = self.chains.get_mut(&chain_id) {
            chain.process(audio)
        } else {
            Err(PlaybackError::InvalidConfig(format!(
                "Effect chain {} not found",
                chain_id
            )))
        }
    }

    /// Process audio through all available chains (for parallel processing)
    pub fn process_all_chains(
        &mut self,
        audio: AudioSamples<T>,
    ) -> PlaybackResult<Vec<AudioSamples<T>>> {
        let mut results = Vec::new();
        let chain_ids: Vec<usize> = self.chains.keys().cloned().collect();

        for chain_id in chain_ids {
            let processed = self.process_chain(chain_id, audio.clone())?;
            results.push(processed);
        }

        Ok(results)
    }

    /// Get list of all chain IDs
    pub fn chain_ids(&self) -> Vec<usize> {
        self.chains.keys().cloned().collect()
    }

    /// Get the number of active chains
    pub fn chain_count(&self) -> usize {
        self.chains.len()
    }
}
