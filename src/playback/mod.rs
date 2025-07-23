//! Audio playback infrastructure for real-time audio output.
//!
//! This module provides comprehensive playback capabilities including:
//! - Cross-platform audio device abstraction
//! - Real-time audio streaming to output devices
//! - Multi-stream mixing and routing
//! - Low-latency playback with minimal overhead
//! - Transport controls (play, pause, stop, seek)
//!
//! # Features
//!
//! - **Cross-Platform**: Works on Windows, macOS, and Linux via cpal
//! - **Low Latency**: Optimized for real-time audio applications
//! - **Device Management**: Automatic device discovery and configuration
//! - **Format Negotiation**: Automatic format conversion to match device capabilities
//! - **Robust Error Handling**: Graceful handling of device disconnections and format changes
//!
//! # Example
//!
//! ```rust,ignore
//! use audio_samples::playback::*;
//! use audio_samples::AudioSamples;
//!
//! async fn play_audio() -> Result<(), PlaybackError> {
//!     let mut player = AudioPlayer::new()?;
//!     let audio = AudioSamples::new_mono(samples, 44100);
//!     
//!     player.play(audio).await?;
//!     player.wait_for_completion().await?;
//!     
//!     Ok(())
//! }
//! ```

#[cfg(feature = "playback")]
pub mod traits;

#[cfg(feature = "playback")]
pub mod devices;

#[cfg(feature = "playback")]
pub mod mixer;

#[cfg(feature = "playback")]
pub mod effects;

#[cfg(feature = "playback")]
pub mod error;

#[cfg(feature = "playback")]
pub mod player;

// Re-export main types for convenience
#[cfg(feature = "playback")]
pub use error::PlaybackError;

#[cfg(feature = "playback")]
pub use traits::{AudioDevice, PlaybackController, PlaybackSink};

#[cfg(feature = "playback")]
pub use player::{AudioPlayer, PlaybackConfig, PlaybackState};

#[cfg(feature = "playback")]
pub use devices::{DeviceInfo, DeviceManager, DeviceType};

#[cfg(feature = "playback")]
pub use mixer::{AudioMixer, MixerChannel, MixerConfig};
