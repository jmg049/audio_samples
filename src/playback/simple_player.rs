//! Simple, working audio player implementation

use crate::{AudioSample, AudioSamples};
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicBool, AtomicUsize, Ordering},
};
use std::time::Duration;

use cpal::{
    Device, Stream, StreamConfig,
    traits::{DeviceTrait, HostTrait, StreamTrait},
};

/// Simple playback state
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SimplePlaybackState {
    Stopped,
    Playing,
    Paused,
}

/// Simple audio player that just works
pub struct SimpleAudioPlayer<T: AudioSample> {
    _device: Device,

    stream: Option<Stream>,

    // Shared audio data
    audio_data: Arc<Mutex<Vec<T>>>,

    // Playback state
    state: Arc<Mutex<SimplePlaybackState>>,
    position: Arc<AtomicUsize>,
    volume: Arc<Mutex<f32>>,
    loop_enabled: Arc<AtomicBool>,

    // Configuration
    sample_rate: u32,
    channels: u16,
}

impl<T: AudioSample> SimpleAudioPlayer<T>
where
    T: Clone + Default + Send + Sync + 'static,
    f32: From<T>,
{
    /// Create a new simple audio player

    pub fn new() -> Result<Self, Box<dyn std::error::Error>> {
        let host = cpal::default_host();
        let device = host
            .default_output_device()
            .ok_or("No output device available")?;

        Ok(Self {
            _device: device,
            stream: None,
            audio_data: Arc::new(Mutex::new(Vec::new())),
            state: Arc::new(Mutex::new(SimplePlaybackState::Stopped)),
            position: Arc::new(AtomicUsize::new(0)),
            volume: Arc::new(Mutex::new(0.8)),
            loop_enabled: Arc::new(AtomicBool::new(false)),
            sample_rate: 44100,
            channels: 2,
        })
    }

    /// Load audio data for playback
    pub fn load_audio(&mut self, audio: AudioSamples<T>) -> Result<(), Box<dyn std::error::Error>> {
        use crate::repr::AudioData;

        // Extract samples from AudioSamples and convert to interleaved format
        let samples = match &audio.data {
            AudioData::Mono(arr) => {
                // For mono, just clone the array
                arr.to_vec()
            }
            AudioData::MultiChannel(arr) => {
                // For multi-channel, interleave the samples
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
        };

        *self.audio_data.lock().unwrap() = samples;
        self.position.store(0, Ordering::Relaxed);

        // Update configuration from audio
        self.sample_rate = audio.sample_rate();
        self.channels = audio.num_channels() as u16;

        Ok(())
    }

    /// Start playback

    pub fn play(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        if self.stream.is_none() {
            self.create_stream()?;
        }

        if let Some(ref stream) = self.stream {
            stream.play()?;
        }

        *self.state.lock().unwrap() = SimplePlaybackState::Playing;
        Ok(())
    }

    #[cfg(not(feature = "playback"))]
    pub fn play(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        *self.state.lock().unwrap() = SimplePlaybackState::Playing;
        println!("Playing audio (feature disabled)");
        Ok(())
    }

    /// Pause playback

    pub fn pause(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        if let Some(ref stream) = self.stream {
            stream.pause()?;
        }
        *self.state.lock().unwrap() = SimplePlaybackState::Paused;
        Ok(())
    }

    #[cfg(not(feature = "playback"))]
    pub fn pause(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        *self.state.lock().unwrap() = SimplePlaybackState::Paused;
        println!("Paused audio (feature disabled)");
        Ok(())
    }

    /// Stop playback

    pub fn stop(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        if let Some(ref stream) = self.stream {
            stream.pause()?; // CPAL doesn't have stop, just pause
        }
        *self.state.lock().unwrap() = SimplePlaybackState::Stopped;
        self.position.store(0, Ordering::Relaxed);
        Ok(())
    }

    #[cfg(not(feature = "playback"))]
    pub fn stop(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        *self.state.lock().unwrap() = SimplePlaybackState::Stopped;
        self.position.store(0, Ordering::Relaxed);
        println!("Stopped audio (feature disabled)");
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

    /// Enable/disable looping
    pub fn set_loop(&mut self, enable: bool) {
        self.loop_enabled.store(enable, Ordering::Relaxed);
    }

    /// Check if looping is enabled
    pub fn is_loop_enabled(&self) -> bool {
        self.loop_enabled.load(Ordering::Relaxed)
    }

    /// Get current playback state
    pub fn state(&self) -> SimplePlaybackState {
        *self.state.lock().unwrap()
    }

    /// Get current position in samples
    pub fn position_samples(&self) -> usize {
        self.position.load(Ordering::Relaxed)
    }

    /// Get current position as duration
    pub fn position(&self) -> Duration {
        let pos = self.position_samples();
        let samples_per_second = self.sample_rate as f64 * self.channels as f64;
        Duration::from_secs_f64(pos as f64 / samples_per_second)
    }

    /// Seek to position (basic implementation)
    pub fn seek(&mut self, position: Duration) -> Result<(), Box<dyn std::error::Error>> {
        let samples_per_second = self.sample_rate as f64 * self.channels as f64;
        let target_sample = (position.as_secs_f64() * samples_per_second) as usize;

        let audio_len = self.audio_data.lock().unwrap().len();
        let clamped_position = target_sample.min(audio_len);

        self.position.store(clamped_position, Ordering::Relaxed);
        Ok(())
    }

    /// Check if currently playing
    pub fn is_playing(&self) -> bool {
        matches!(self.state(), SimplePlaybackState::Playing)
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

        // Clone Arc references for the callback
        let audio_data = Arc::clone(&self.audio_data);
        let position = Arc::clone(&self.position);
        let volume = Arc::clone(&self.volume);
        let state = Arc::clone(&self.state);
        let loop_enabled = Arc::clone(&self.loop_enabled);

        let stream = device.build_output_stream(
            &config,
            move |data: &mut [f32], _: &cpal::OutputCallbackInfo| {
                let current_state = *state.lock().unwrap();

                if current_state != SimplePlaybackState::Playing {
                    // Fill with silence when not playing
                    data.fill(0.0);
                    return;
                }

                let audio_samples = audio_data.lock().unwrap();
                let vol = *volume.lock().unwrap();
                let pos = position.load(Ordering::Relaxed);
                let is_looping = loop_enabled.load(Ordering::Relaxed);

                for (i, output_sample) in data.iter_mut().enumerate() {
                    let sample_index = pos + i;

                    if sample_index < audio_samples.len() {
                        let audio_sample = audio_samples[sample_index].clone();
                        *output_sample = f32::from(audio_sample) * vol;
                    } else if is_looping && !audio_samples.is_empty() {
                        // Loop back to beginning
                        let loop_index = sample_index % audio_samples.len();
                        let audio_sample = audio_samples[loop_index].clone();
                        *output_sample = f32::from(audio_sample) * vol;
                    } else {
                        // End of audio, fill with silence
                        *output_sample = 0.0;
                    }
                }

                // Update position
                let new_pos = if is_looping && !audio_samples.is_empty() {
                    (pos + data.len()) % audio_samples.len()
                } else {
                    pos + data.len()
                };
                position.store(new_pos, Ordering::Relaxed);

                // Stop if we've reached the end and not looping
                if !is_looping && new_pos >= audio_samples.len() {
                    *state.lock().unwrap() = SimplePlaybackState::Stopped;
                }
            },
            |err| {
                eprintln!("Audio stream error: {}", err);
            },
            None,
        )?;

        self.stream = Some(stream);
        Ok(())
    }
}
