//! Unified streaming and playback API.
//!
//! This module provides a high-level API for streaming audio sources
//! directly to playback devices with comprehensive lifecycle management.

use crate::{
    AudioSample, ConvertTo,
    playback::{StreamingPlayer, StreamingPlayerState},
    streaming::{sources::GeneratorSource, traits::AudioSource},
};
use std::time::Duration;

/// High-level streaming playback controller
pub struct StreamingPlayback<T: AudioSample> {
    player: StreamingPlayer<T>,
}

impl<T: AudioSample> StreamingPlayback<T>
where
    T: Clone + Default + Send + Sync + 'static,
    f32: From<T>,
    f64: ConvertTo<T>,
{
    /// Create a new streaming playback instance
    pub fn new() -> Result<Self, Box<dyn std::error::Error>> {
        Ok(Self {
            player: StreamingPlayer::new()?,
        })
    }

    /// Play from a generator source
    pub async fn play_generator(
        &mut self,
        generator: GeneratorSource<T>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        self.player.connect_and_play(generator).await
    }

    /// Play from any audio source
    pub async fn play_source<S>(&mut self, source: S) -> Result<(), Box<dyn std::error::Error>>
    where
        S: AudioSource<T> + Send + 'static,
    {
        self.player.connect_and_play(source).await
    }

    /// Pause playback
    pub fn pause(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        self.player.pause()
    }

    /// Resume playback
    pub fn resume(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        self.player.resume()
    }

    /// Stop playback and streaming
    pub async fn stop(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        self.player.stop().await
    }

    /// Set volume (0.0 to 1.0)
    pub fn set_volume(&mut self, volume: f32) {
        self.player.set_volume(volume);
    }

    /// Get current volume
    pub fn volume(&self) -> f32 {
        self.player.volume()
    }

    /// Get current state
    pub fn state(&self) -> StreamingPlayerState {
        self.player.state()
    }

    /// Check if currently playing
    pub fn is_playing(&self) -> bool {
        self.player.is_playing()
    }

    /// Get buffer level (0.0 to 1.0)
    pub fn buffer_level(&self) -> f64 {
        self.player.buffer_level()
    }

    /// Wait for playback to complete (for finite streams)
    pub async fn wait_for_completion(&self) -> Result<(), Box<dyn std::error::Error>> {
        while matches!(
            self.state(),
            StreamingPlayerState::Playing | StreamingPlayerState::Connecting
        ) {
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
        Ok(())
    }
}

/// Convenience functions for common use cases
impl StreamingPlayback<f32> {
    /// Play a sine wave
    pub async fn play_sine_wave(
        &mut self,
        frequency: f64,
        sample_rate: usize,
        channels: usize,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let generator = GeneratorSource::<f32>::sine(frequency, sample_rate, channels);
        self.play_generator(generator).await
    }

    /// Play white noise
    pub async fn play_white_noise(
        &mut self,
        sample_rate: usize,
        channels: usize,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let generator = GeneratorSource::<f32>::white_noise(sample_rate, channels);
        self.play_generator(generator).await
    }

    /// Play silence
    pub async fn play_silence(
        &mut self,
        sample_rate: usize,
        channels: usize,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let generator = GeneratorSource::<f32>::silence(sample_rate, channels);
        self.play_generator(generator).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[tokio::test]
    async fn test_streaming_playback_creation() {
        let result = StreamingPlayback::<f32>::new();
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_sine_wave_playback() {
        let mut playback = StreamingPlayback::<f32>::new().unwrap();

        // Start sine wave playback
        let result = playback.play_sine_wave(440.0, 44100, 2).await;
        assert!(result.is_ok());

        // Should be playing
        assert!(playback.is_playing());

        // Wait a bit then stop
        tokio::time::sleep(Duration::from_millis(100)).await;
        let stop_result = playback.stop().await;
        assert!(stop_result.is_ok());
    }
}
