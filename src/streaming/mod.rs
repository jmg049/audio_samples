//! Audio streaming infrastructure for real-time audio processing.
//!
//! This module provides comprehensive streaming capabilities including:
//! - Multiple audio sources (TCP, UDP, files, generators)
//! - Adaptive buffering and flow control
//! - Real-time processing pipelines
//! - Error recovery and stream management
//!
//! # Features
//!
//! - **Low Latency**: Optimized for real-time audio applications
//! - **Multiple Sources**: Support for network streams, files, and generated audio
//! - **Robust Error Handling**: Automatic recovery from network issues
//! - **Flexible Buffering**: Adaptive buffer sizes based on stream characteristics
//!
//! # Example
//!
//! ```rust,ignore
//! use audio_samples::streaming::*;
//!
//! async fn stream_audio() -> Result<(), StreamError> {
//!     let mut source = TcpStreamSource::connect("localhost:8080").await?;
//!     let mut stream = AudioStream::new(source, StreamConfig::default());
//!     
//!     while let Some(chunk) = stream.next_chunk().await? {
//!         // Process audio chunk in real-time
//!         process_audio(chunk);
//!     }
//!     
//!     Ok(())
//! }
//! ```

#[cfg(feature = "streaming")]
pub mod traits;

#[cfg(feature = "streaming")]
pub mod sources;

#[cfg(feature = "streaming")]
pub mod buffers;

#[cfg(feature = "streaming")]
pub mod error;

#[cfg(feature = "streaming")]
pub mod stream;

// Re-export main types for convenience
#[cfg(feature = "streaming")]
pub use error::StreamError;

#[cfg(feature = "streaming")]
pub use traits::{AudioSource, StreamProcessor, StreamSink};

#[cfg(feature = "streaming")]
pub use stream::{AudioStream, StreamConfig, StreamState};

#[cfg(feature = "streaming")]
pub use sources::{
    file::FileStreamSource, generator::GeneratorSource, tcp::TcpStreamSource, udp::UdpStreamSource,
};

#[cfg(feature = "streaming")]
pub use buffers::{BufferConfig, CircularBuffer, StreamBuffer};
