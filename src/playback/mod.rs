//! Advanced audio playback functionality.
//!
//! This module provides comprehensive real-time audio playback capabilities including:
//!
//! - **Device Management**: Automatic discovery and configuration of audio devices
//! - **Multi-Channel Mixing**: Professional-grade audio mixing with per-channel controls  
//! - **Effects Processing**: Plugin-style real-time audio effects
//! - **Transport Controls**: Full playback control (play, pause, stop, seek)
//! - **Error Recovery**: Robust error handling with automatic recovery strategies
//!
//! # Architecture Overview
//!
//! The playback system is built around several core components:
//!
//! - [`AudioPlayer`] - High-level audio player with transport controls
//! - [`DeviceManager`] - Audio device discovery and management
//! - [`AudioMixer`] - Multi-channel audio mixing console
//! - [`EffectsEngine`] - Real-time audio effects processing
//! - [`PlaybackError`] - Comprehensive error handling
//!
//! # Sample Type Support
//!
//! All playback components are generic over sample types and support:
//! - `i16` - 16-bit signed integers (most common)  
//! - `I24` - 24-bit signed integers (professional audio)
//! - `i32` - 32-bit signed integers
//! - `f32` - 32-bit floats (normalized -1.0 to 1.0)
//! - `f64` - 64-bit floats (highest precision)
//!
//! # Quick Start Examples
//!
//! ## Basic Playback
//! ```rust,ignore
//! use audio_samples::playback::{AudioPlayer, PlaybackConfig};
//! use audio_samples::AudioSamples;
//!
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! // Create a player with default configuration
//! let mut player = AudioPlayer::<f32>::new()?;
//!
//! // Load some audio data
//! let audio = AudioSamples::from_file("example.wav")?;
//! player.load_audio(audio).await?;
//!
//! // Start playback
//! player.play().await?;
//!
//! // Control playback
//! player.set_volume(0.8);
//! player.pause().await?;
//! player.seek(std::time::Duration::from_secs(30)).await?;
//! player.play().await?;
//! # Ok(())
//! # }
//! ```
//!
//! ## Multi-Channel Mixing
//! ```rust,ignore  
//! use audio_samples::playback::{AudioMixer, MixerConfig, ChannelConfig};
//!
//! # fn example() -> Result<(), Box<dyn std::error::Error>> {
//! let config = MixerConfig::new(48000, 1024, 2);
//! let mut mixer = AudioMixer::<f32>::new(config)?;
//!
//! // Add channels with different configurations
//! let channel_config = ChannelConfig {
//!     volume: 0.8,
//!     pan: -0.3,  // Slightly left
//!     solo: false,
//!     muted: false,
//!     ..Default::default()
//! };
//!
//! let channel_id = mixer.add_channel(channel_config)?;
//!
//! // Mix all channels
//! let mixed_output = mixer.mix()?;
//! # Ok(())
//! # }
//! ```
//!
//! ## Effects Processing
//! ```rust,ignore
//! use audio_samples::playback::{EffectsEngine, EffectsConfig};
//! use audio_samples::playback::{GainEffect, ComplexAudioEffect};
//!
//! # fn example() -> Result<(), Box<dyn std::error::Error>> {
//! let config = EffectsConfig::default();
//! let mut engine = EffectsEngine::<f32>::new(config);
//!
//! // Create an effect chain  
//! let chain_id = engine.create_chain();
//! let chain = engine.get_chain_mut(chain_id).unwrap();
//!
//! // Add effects to the chain
//! let gain_effect = Box::new(GainEffect::new(-6.0)); // -6dB gain
//! chain.add_effect(gain_effect);
//!
//! // Process audio through the effect chain
//! let processed = engine.process_chain(chain_id, audio_samples)?;
//! # Ok(())
//! # }
//! ```
//!
//! # Integration with Operations
//!
//! The playback system integrates seamlessly with the [`operations`](crate::operations) module:
//!
//! ```rust,ignore
//! use audio_samples::operations::*;
//!
//! # fn example() -> Result<(), Box<dyn std::error::Error>> {
//! // Apply operations before playback
//! let mut audio = AudioSamples::from_file("input.wav")?;
//!
//! // Normalize and apply EQ
//! audio.normalize(-1.0, 1.0, NormalizationMethod::Peak)?;
//! audio.parametric_eq(&eq_bands, 48000.0)?;
//!
//! // Load into player
//! player.load_audio(audio).await?;
//! # Ok(())
//! # }
//! ```
//!
//! # Feature Gates
//!
//! - `playback` - Enables core playback functionality (requires CPAL)
//! - `streaming` - Enables integration with streaming sources
//! - `realtime` - Enables both playback and streaming with real-time processing
//!
//! See [`PLAYBACK_ARCHITECTURE.md`](../../../PLAYBACK_ARCHITECTURE.md) for detailed
//! architecture documentation.

// Core playback infrastructure modules
pub mod devices;
pub mod effects;
pub mod effects_complex;
pub mod error;
pub mod mixer;
pub mod player;
pub mod traits;

pub mod simple_player;
pub mod streaming_player;

// Re-export player types for convenience
pub use simple_player::{SimpleAudioPlayer, SimplePlaybackState};

#[cfg(all(feature = "streaming", feature = "playback"))]
pub use streaming_player::{StreamingPlayer, StreamingPlayerState};

// Re-export advanced playback components
pub use devices::{DeviceHandle, DeviceManager};
pub use effects::{
    AudioEffect as SimpleAudioEffect, EffectChain as SimpleEffectChain,
    EffectsEngine as SimpleEffectsEngine,
};
pub use effects_complex::{
    AudioEffect as ComplexAudioEffect, EffectChain as ComplexEffectChain,
    EffectsEngine as ComplexEffectsEngine,
};
pub use error::{PlaybackError, PlaybackResult};
pub use mixer::{AudioMixer, ChannelConfig, MixAlgorithm, MixerConfig};
pub use player::{AudioPlayer, PlaybackConfig};
pub use traits::{
    AudioDevice, AudioFormatSpec, DeviceInfo, DeviceType, PlaybackController, PlaybackMetrics,
    PlaybackSink, PlaybackState, SampleFormat,
};
