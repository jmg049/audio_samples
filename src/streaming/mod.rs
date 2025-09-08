//! Audio streaming infrastructure for real-time audio processing.
//!
//! This module provides comprehensive streaming capabilities including:
//! - Multiple audio sources (TCP, UDP, files, generators)
//! - Adaptive buffering and flow control
//! - Real-time processing pipelines
//! - Error recovery and stream management
//!
//! # Overview
//!
//! The streaming module enables real-time audio processing with:
//! - **Audio Sources**: Generate or receive audio data (TCP, UDP, file, signal generators)  
//! - **Audio Processing**: Transform audio data in real-time
//! - **Buffering**: Handle timing differences between producers and consumers
//! - **Error Recovery**: Graceful handling of network issues and buffer problems
//!
//! # Quick Start
//!
//! ```rust
//! use audio_samples::streaming::{
//!     sources::generator::GeneratorSource,
//!     traits::AudioSource,
//! };
//!
//! # tokio_test::block_on(async {
//! // Create a sine wave generator
//! let mut generator = GeneratorSource::<f32>::sine(440.0, 48000, 2);
//!
//! // Generate audio data
//! if let Ok(Some(audio_chunk)) = generator.next_chunk().await {
//!     println!("Generated {} samples at {} Hz",
//!         audio_chunk.samples_per_channel(),
//!         audio_chunk.sample_rate());
//!     println!("Format info: {:?}", generator.format_info());
//! }
//! # });
//! ```
//!
//! # Architecture
//!
//! ## Audio Sources
//!
//! Audio sources implement the [`AudioSource`] trait to provide audio data:
//!
//! - [`GeneratorSource`] - Generates test signals (sine, noise, chirp, etc.)
//! - [`TcpStreamSource`] - Receives audio over TCP connections  
//! - [`UdpStreamSource`] - Receives audio over UDP with optional multicast
//! - [`FileStreamSource`] - Streams audio from large files
//!
//! ## Buffering System
//!
//! The buffering system handles real-time constraints:
//!
//! - [`CircularBuffer`] - Lock-free circular buffer for audio samples
//! - [`StreamBuffer`] - High-level streaming buffer with watermark management
//! - Adaptive buffering to handle network jitter and processing delays
//!
//! ## Error Handling
//!
//! The [`StreamError`] type provides comprehensive error classification:
//!
//! - **Recoverable errors**: Buffer under/overruns, network timeouts
//! - **Fatal errors**: Format mismatches, invalid configurations
//! - Automatic retry logic for recoverable errors
//!
//! # Examples
//!
//! ## Signal Generation
//!
//! ```rust
//! use audio_samples::streaming::sources::generator::{GeneratorSource, SignalType};
//! use std::time::Duration;
//!
//! # tokio_test::block_on(async {
//! // Create different signal types
//! let sine = GeneratorSource::<f32>::sine(440.0, 48000, 2);
//! let noise = GeneratorSource::<f32>::white_noise(48000, 2);  
//! let silence = GeneratorSource::<f32>::silence(48000, 2);
//!
//! // Generate a 1-second chirp from 440Hz to 880Hz
//! let chirp = GeneratorSource::<f32>::chirp(440.0, 880.0, Duration::from_secs(1), 48000, 1);
//! # });
//! ```
//!
//! [`AudioSource`]: traits::AudioSource
//! [`GeneratorSource`]: sources::generator::GeneratorSource
//! [`TcpStreamSource`]: sources::tcp::TcpStreamSource
//! [`UdpStreamSource`]: sources::udp::UdpStreamSource
//! [`FileStreamSource`]: sources::file::FileStreamSource
//! [`CircularBuffer`]: buffers::CircularBuffer
//! [`StreamBuffer`]: buffers::StreamBuffer
//! [`StreamError`]: error::StreamError
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

pub mod buffers;
pub mod error;
pub mod sources;
pub mod stream;
pub mod traits;

// Re-export main types for convenience
pub use buffers::{BufferConfig, CircularBuffer, StreamBuffer};
pub use error::StreamError;
pub use sources::{
    file::FileStreamSource, generator::GeneratorSource, tcp::TcpStreamSource, udp::UdpStreamSource,
};
pub use stream::{AudioStream, StreamState};
pub use traits::{AudioSource, StreamConfig, StreamProcessor, StreamSink};

#[cfg(test)]
pub mod tests;
