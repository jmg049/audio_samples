//! Streaming audio player that connects audio sources to real-time playback.
//!
//! This module is only available when both streaming and playback features are enabled.

#[cfg(all(feature = "streaming", feature = "playback"))]
use crate::{
    AudioSample, AudioSamples,
    streaming::{StreamError, traits::AudioSource},
};
#[cfg(all(feature = "streaming", feature = "playback"))]
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicBool, AtomicUsize, Ordering},
};
#[cfg(all(feature = "streaming", feature = "playback"))]
use std::time::Duration;
#[cfg(all(feature = "streaming", feature = "playback"))]
use tokio::{sync::mpsc, task::JoinHandle};

#[cfg(all(feature = "streaming", feature = "playback"))]
use cpal::{
    Device, Stream, StreamConfig,
    traits::{DeviceTrait, HostTrait, StreamTrait},
};

/// State of the streaming player
#[cfg(all(feature = "streaming", feature = "playback"))]
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum StreamingPlayerState {
    Idle,
    Connecting,
    Playing,
    Paused,
    Stopped,
    Error,
}

/// A player that can stream audio from any AudioSource in real-time
#[cfg(all(feature = "streaming", feature = "playback"))]
pub struct StreamingPlayer<T: AudioSample> {
    // Audio output
    _device: Device,
    stream: Option<Stream>,

    // State management
    state: Arc<Mutex<StreamingPlayerState>>,
    volume: Arc<Mutex<f32>>,

    // Streaming infrastructure
    audio_buffer: Arc<Mutex<Vec<T>>>,
    buffer_position: Arc<AtomicUsize>,
    buffer_write_position: Arc<AtomicUsize>,
    is_buffer_empty: Arc<AtomicBool>,

    // Communication with streaming task
    chunk_sender: Option<mpsc::UnboundedSender<AudioSamples<T>>>,
    streaming_task: Option<JoinHandle<()>>,
    buffer_task: Option<JoinHandle<()>>,
    stop_signal: Arc<AtomicBool>,

    // Configuration
    sample_rate: u32,
    channels: u16,
    buffer_size: usize,
}

#[cfg(all(feature = "streaming", feature = "playback"))]
impl<T: AudioSample> StreamingPlayer<T>
where
    T: Clone + Default + Send + Sync + 'static,
    f32: From<T>,
{
    /// Create a new streaming player
    pub fn new() -> Result<Self, Box<dyn std::error::Error>> {
        let host = cpal::default_host();
        let device = host
            .default_output_device()
            .ok_or("No output device available")?;

        Ok(Self {
            _device: device,
            stream: None,
            state: Arc::new(Mutex::new(StreamingPlayerState::Idle)),
            volume: Arc::new(Mutex::new(0.8)),
            audio_buffer: Arc::new(Mutex::new(Vec::with_capacity(48000 * 4))), // 4 seconds at 48kHz mono
            buffer_position: Arc::new(AtomicUsize::new(0)),
            buffer_write_position: Arc::new(AtomicUsize::new(0)),
            is_buffer_empty: Arc::new(AtomicBool::new(true)),
            chunk_sender: None,
            streaming_task: None,
            buffer_task: None,
            stop_signal: Arc::new(AtomicBool::new(false)),
            sample_rate: 44100,
            channels: 2,
            buffer_size: 48000 * 4, // 4 seconds
        })
    }

    /// Connect to an audio source and start streaming playback
    pub async fn connect_and_play<S>(
        &mut self,
        mut source: S,
    ) -> Result<(), Box<dyn std::error::Error>>
    where
        S: AudioSource<T> + Send + 'static,
    {
        // Set state to connecting
        *self.state.lock().unwrap() = StreamingPlayerState::Connecting;

        // Get format info from source
        let format_info = source.format_info();
        self.sample_rate = format_info.sample_rate as u32;
        self.channels = format_info.channels as u16;

        // Create communication channel for audio chunks
        let (chunk_tx, mut chunk_rx) = mpsc::unbounded_channel::<AudioSamples<T>>();
        let chunk_tx_clone = chunk_tx.clone();
        self.chunk_sender = Some(chunk_tx);

        // Reset buffer state
        self.audio_buffer.lock().unwrap().clear();
        self.buffer_position.store(0, Ordering::Relaxed);
        self.buffer_write_position.store(0, Ordering::Relaxed);
        self.is_buffer_empty.store(true, Ordering::Relaxed);
        self.stop_signal.store(false, Ordering::Relaxed);

        // Clone shared state for the streaming task
        let stop_signal = Arc::clone(&self.stop_signal);
        let state = Arc::clone(&self.state);

        // Start streaming task
        let streaming_handle = tokio::spawn(async move {
            while !stop_signal.load(Ordering::Relaxed) {
                match source.next_chunk().await {
                    Ok(Some(chunk)) => {
                        // Send chunk to buffer management task
                        if chunk_tx_clone.send(chunk).is_err() {
                            break; // Receiver dropped, exit
                        }
                    }
                    Ok(None) => {
                        // Stream ended normally
                        break;
                    }
                    Err(StreamError::Buffer { operation: _, .. }) => {
                        // Recoverable buffer error, continue
                        tokio::time::sleep(Duration::from_millis(10)).await;
                        continue;
                    }
                    Err(_) => {
                        // Non-recoverable error
                        *state.lock().unwrap() = StreamingPlayerState::Error;
                        break;
                    }
                }
            }
        });

        self.streaming_task = Some(streaming_handle);

        // Start buffer management task
        let buffer_handle = {
            let audio_buffer = Arc::clone(&self.audio_buffer);
            let buffer_write_position = Arc::clone(&self.buffer_write_position);
            let is_buffer_empty = Arc::clone(&self.is_buffer_empty);
            let stop_signal = Arc::clone(&self.stop_signal);
            let buffer_size = self.buffer_size;

            tokio::spawn(async move {
                while let Some(chunk) = chunk_rx.recv().await {
                    if stop_signal.load(Ordering::Relaxed) {
                        break;
                    }

                    // Add chunk to circular buffer
                    let samples = Self::extract_interleaved_samples(&chunk);
                    let mut buffer = audio_buffer.lock().unwrap();

                    // Ensure buffer has enough space
                    if buffer.len() < buffer_size {
                        buffer.resize(buffer_size, T::default());
                    }

                    let write_pos = buffer_write_position.load(Ordering::Relaxed);

                    // Write samples to buffer (circular)
                    for (i, &sample) in samples.iter().enumerate() {
                        let pos = (write_pos + i) % buffer_size;
                        buffer[pos] = sample;
                    }

                    // Update write position
                    let new_write_pos = (write_pos + samples.len()) % buffer_size;
                    buffer_write_position.store(new_write_pos, Ordering::Relaxed);
                    is_buffer_empty.store(false, Ordering::Relaxed);
                }
            })
        };

        // Store the buffer handle so the task stays alive
        self.buffer_task = Some(buffer_handle);

        // Create and start the audio stream
        self.create_stream()?;

        if let Some(ref stream) = self.stream {
            stream.play()?;
        }

        *self.state.lock().unwrap() = StreamingPlayerState::Playing;

        Ok(())
    }

    /// Extract interleaved samples from AudioSamples
    fn extract_interleaved_samples(audio: &AudioSamples<T>) -> Vec<T> {
        use crate::repr::AudioData;

        match &audio.data {
            AudioData::Mono(arr) => arr.to_vec(),
            AudioData::MultiChannel(arr) => {
                let mut interleaved = Vec::with_capacity(arr.len());
                let samples_per_channel = arr.ncols();
                let num_channels = arr.nrows();

                for sample_idx in 0..samples_per_channel {
                    for channel_idx in 0..num_channels {
                        interleaved.push(arr[(channel_idx, sample_idx)]);
                    }
                }
                interleaved
            }
        }
    }

    /// Pause playback
    pub fn pause(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        if let Some(ref stream) = self.stream {
            stream.pause()?;
        }
        *self.state.lock().unwrap() = StreamingPlayerState::Paused;
        Ok(())
    }

    /// Resume playback
    pub fn resume(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        if let Some(ref stream) = self.stream {
            stream.play()?;
        }
        *self.state.lock().unwrap() = StreamingPlayerState::Playing;
        Ok(())
    }

    /// Stop playback and streaming
    pub async fn stop(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        // Signal streaming task to stop
        self.stop_signal.store(true, Ordering::Relaxed);

        // Stop audio stream
        if let Some(ref stream) = self.stream {
            stream.pause()?;
        }

        // Wait for streaming task to finish
        if let Some(handle) = self.streaming_task.take() {
            handle.abort();
        }

        // Wait for buffer task to finish
        if let Some(handle) = self.buffer_task.take() {
            handle.abort();
        }

        // Clear communication
        self.chunk_sender = None;

        // Clear buffer
        self.audio_buffer.lock().unwrap().clear();
        self.buffer_position.store(0, Ordering::Relaxed);
        self.buffer_write_position.store(0, Ordering::Relaxed);
        self.is_buffer_empty.store(true, Ordering::Relaxed);

        *self.state.lock().unwrap() = StreamingPlayerState::Stopped;

        Ok(())
    }

    /// Set volume (0.0 to 1.0)
    pub fn set_volume(&mut self, volume: f32) {
        *self.volume.lock().unwrap() = volume.clamp(0.0, 1.0);
    }

    /// Get current volume
    pub fn volume(&self) -> f32 {
        *self.volume.lock().unwrap()
    }

    /// Get current state
    pub fn state(&self) -> StreamingPlayerState {
        *self.state.lock().unwrap()
    }

    /// Check if currently playing
    pub fn is_playing(&self) -> bool {
        matches!(self.state(), StreamingPlayerState::Playing)
    }

    /// Get buffer level (0.0 to 1.0)
    pub fn buffer_level(&self) -> f64 {
        if self.is_buffer_empty.load(Ordering::Relaxed) {
            return 0.0;
        }

        let read_pos = self.buffer_position.load(Ordering::Relaxed);
        let write_pos = self.buffer_write_position.load(Ordering::Relaxed);

        let available_samples = if write_pos >= read_pos {
            write_pos - read_pos
        } else {
            self.buffer_size - read_pos + write_pos
        };

        available_samples as f64 / self.buffer_size as f64
    }

    fn create_stream(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        let host = cpal::default_host();
        let device = host
            .default_output_device()
            .ok_or("No output device available")?;

        let config = StreamConfig {
            channels: self.channels,
            sample_rate: cpal::SampleRate(self.sample_rate),
            buffer_size: cpal::BufferSize::Default,
        };

        // Clone shared state for audio callback
        let audio_buffer = Arc::clone(&self.audio_buffer);
        let buffer_position = Arc::clone(&self.buffer_position);
        let volume = Arc::clone(&self.volume);
        let state = Arc::clone(&self.state);
        let is_buffer_empty = Arc::clone(&self.is_buffer_empty);
        let buffer_size = self.buffer_size;

        let stream = device.build_output_stream(
            &config,
            move |data: &mut [f32], _: &cpal::OutputCallbackInfo| {
                let current_state = *state.lock().unwrap();

                if current_state != StreamingPlayerState::Playing {
                    // Fill with silence when not playing
                    data.fill(0.0);
                    return;
                }

                if is_buffer_empty.load(Ordering::Relaxed) {
                    // No data available yet, fill with silence
                    data.fill(0.0);
                    return;
                }

                let buffer = audio_buffer.lock().unwrap();
                let vol = *volume.lock().unwrap();
                let read_pos = buffer_position.load(Ordering::Relaxed);

                if buffer.is_empty() {
                    data.fill(0.0);
                    return;
                }

                for (i, output_sample) in data.iter_mut().enumerate() {
                    let sample_index = (read_pos + i) % buffer_size;

                    if sample_index < buffer.len() {
                        let audio_sample = buffer[sample_index].clone();
                        *output_sample = f32::from(audio_sample) * vol;
                    } else {
                        *output_sample = 0.0;
                    }
                }

                // Update read position
                let new_read_pos = (read_pos + data.len()) % buffer_size;
                buffer_position.store(new_read_pos, Ordering::Relaxed);
            },
            |err| {
                eprintln!("Streaming audio error: {}", err);
            },
            None,
        )?;

        self.stream = Some(stream);
        Ok(())
    }
}

// Ensure proper cleanup on drop
#[cfg(all(feature = "streaming", feature = "playback"))]
impl<T: AudioSample> Drop for StreamingPlayer<T> {
    fn drop(&mut self) {
        // Signal stop
        self.stop_signal.store(true, Ordering::Relaxed);

        // Abort streaming task if it exists
        if let Some(handle) = self.streaming_task.take() {
            handle.abort();
        }

        // Abort buffer task if it exists
        if let Some(handle) = self.buffer_task.take() {
            handle.abort();
        }
    }
}
