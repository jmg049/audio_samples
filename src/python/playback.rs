//! Python bindings for audio playback functionality.
//!
//! This module provides Python access to audio playback capabilities including:
//! - Simple audio player with transport controls
//! - Volume control and looping
//! - Playback state management
//! - Cross-platform audio output

use crate::{
    playback::{SimpleAudioPlayer, SimplePlaybackState},
    python::{AudioSamplesData, PyAudioSamples, utils::map_error},
};
use pyo3::prelude::*;
use std::time::Duration;

/// Python wrapper for SimplePlaybackState
#[pyclass(name = "PlaybackState", module = "audio_samples.playback")]
#[derive(Clone, Copy)]
pub struct PyPlaybackState {
    inner: SimplePlaybackState,
}

#[pymethods]
impl PyPlaybackState {
    /// Stopped state
    #[classattr]
    const STOPPED: PyPlaybackState = PyPlaybackState {
        inner: SimplePlaybackState::Stopped,
    };

    /// Playing state
    #[classattr]
    const PLAYING: PyPlaybackState = PyPlaybackState {
        inner: SimplePlaybackState::Playing,
    };

    /// Paused state
    #[classattr]
    const PAUSED: PyPlaybackState = PyPlaybackState {
        inner: SimplePlaybackState::Paused,
    };

    /// Check if currently stopped
    fn is_stopped(&self) -> bool {
        matches!(self.inner, SimplePlaybackState::Stopped)
    }

    /// Check if currently playing
    fn is_playing(&self) -> bool {
        matches!(self.inner, SimplePlaybackState::Playing)
    }

    /// Check if currently paused
    fn is_paused(&self) -> bool {
        matches!(self.inner, SimplePlaybackState::Paused)
    }

    fn __repr__(&self) -> String {
        match self.inner {
            SimplePlaybackState::Stopped => "PlaybackState.STOPPED".to_string(),
            SimplePlaybackState::Playing => "PlaybackState.PLAYING".to_string(),
            SimplePlaybackState::Paused => "PlaybackState.PAUSED".to_string(),
        }
    }

    fn __str__(&self) -> String {
        match self.inner {
            SimplePlaybackState::Stopped => "Stopped".to_string(),
            SimplePlaybackState::Playing => "Playing".to_string(),
            SimplePlaybackState::Paused => "Paused".to_string(),
        }
    }

    fn __eq__(&self, other: &PyPlaybackState) -> bool {
        self.inner == other.inner
    }
}

/// Python wrapper for SimpleAudioPlayer
#[pyclass(name = "AudioPlayer", module = "audio_samples.playback")]
pub struct PyAudioPlayer {
    // We'll use f32 as the default sample type for simplicity
    inner: SimpleAudioPlayer<f32>,
}

#[pymethods]
impl PyAudioPlayer {
    /// Create a new audio player
    #[new]
    fn new() -> PyResult<Self> {
        match SimpleAudioPlayer::<f32>::new() {
            Ok(player) => Ok(Self { inner: player }),
            Err(e) => Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Failed to create audio player: {}",
                e
            ))),
        }
    }

    /// Load audio data for playback
    ///
    /// # Arguments
    /// * `audio` - AudioSamples object containing audio data
    ///
    /// # Examples
    /// ```python
    /// import audio_samples as aus
    /// import numpy as np
    ///
    /// # Create some audio data
    /// data = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 44100))
    /// audio = aus.from_numpy(data, sample_rate=44100)
    ///
    /// # Load into player
    /// player = aus.playback.AudioPlayer()
    /// player.load_audio(audio)
    /// ```
    fn load_audio(&mut self, audio: &PyAudioSamples) -> PyResult<()> {
        // Convert to f32 AudioSamples if needed
        let f32_audio = match &audio.data() {
            AudioSamplesData::F32(audio_f32) => audio_f32.clone(),
            AudioSamplesData::F64(audio_f64) => {
                use crate::operations::AudioTypeConversion;
                audio_f64.as_f32().map_err(map_error)?
            }
            AudioSamplesData::I32(audio_i32) => {
                use crate::operations::AudioTypeConversion;
                audio_i32.as_f32().map_err(map_error)?
            }
            AudioSamplesData::I16(audio_i16) => {
                use crate::operations::AudioTypeConversion;
                audio_i16.as_f32().map_err(map_error)?
            }
            AudioSamplesData::I24(audio_i24) => {
                use crate::operations::AudioTypeConversion;
                audio_i24.as_f32().map_err(map_error)?
            }
        };

        self.inner.load_audio(f32_audio).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Failed to load audio: {}",
                e
            ))
        })
    }

    /// Start playback
    ///
    /// # Examples
    /// ```python
    /// player.play()
    /// ```
    fn play(&mut self) -> PyResult<()> {
        self.inner.play().map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Failed to start playback: {}",
                e
            ))
        })
    }

    /// Pause playback
    ///
    /// # Examples
    /// ```python
    /// player.pause()
    /// ```
    fn pause(&mut self) -> PyResult<()> {
        self.inner.pause().map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Failed to pause playback: {}",
                e
            ))
        })
    }

    /// Stop playback and reset position to beginning
    ///
    /// # Examples
    /// ```python
    /// player.stop()
    /// ```
    fn stop(&mut self) -> PyResult<()> {
        self.inner.stop().map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Failed to stop playback: {}",
                e
            ))
        })
    }

    /// Set playback volume (0.0 to 1.0)
    ///
    /// # Arguments
    /// * `volume` - Volume level from 0.0 (silent) to 1.0 (full volume)
    ///
    /// # Examples
    /// ```python
    /// player.set_volume(0.5)  # 50% volume
    /// ```
    fn set_volume(&mut self, volume: f32) -> PyResult<()> {
        if volume < 0.0 || volume > 1.0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Volume must be between 0.0 and 1.0",
            ));
        }
        self.inner.set_volume(volume);
        Ok(())
    }

    /// Get current volume level (0.0 to 1.0)
    ///
    /// # Returns
    /// Current volume level
    ///
    /// # Examples
    /// ```python
    /// current_volume = player.volume()
    /// ```
    fn volume(&self) -> f32 {
        self.inner.volume()
    }

    /// Enable or disable looping
    ///
    /// # Arguments
    /// * `enable` - Whether to enable looping
    ///
    /// # Examples
    /// ```python
    /// player.set_loop(True)   # Enable looping
    /// player.set_loop(False)  # Disable looping
    /// ```
    fn set_loop(&mut self, enable: bool) {
        self.inner.set_loop(enable);
    }

    /// Check if looping is enabled
    ///
    /// # Returns
    /// True if looping is enabled, False otherwise
    ///
    /// # Examples
    /// ```python
    /// if player.is_loop_enabled():
    ///     print("Audio will loop")
    /// ```
    fn is_loop_enabled(&self) -> bool {
        self.inner.is_loop_enabled()
    }

    /// Get current playback state
    ///
    /// # Returns
    /// Current playback state (STOPPED, PLAYING, or PAUSED)
    ///
    /// # Examples
    /// ```python
    /// state = player.state()
    /// if state.is_playing():
    ///     print("Currently playing")
    /// ```
    fn state(&self) -> PyPlaybackState {
        PyPlaybackState {
            inner: self.inner.state(),
        }
    }

    /// Get current position in samples
    ///
    /// # Returns
    /// Current playback position in samples
    ///
    /// # Examples
    /// ```python
    /// position_samples = player.position_samples()
    /// ```
    fn position_samples(&self) -> usize {
        self.inner.position_samples()
    }

    /// Get current position as duration in seconds
    ///
    /// # Returns
    /// Current playback position in seconds
    ///
    /// # Examples
    /// ```python
    /// position_sec = player.position()
    /// print(f"Position: {position_sec:.2f} seconds")
    /// ```
    fn position(&self) -> f64 {
        self.inner.position().as_secs_f64()
    }

    /// Seek to a specific position
    ///
    /// # Arguments
    /// * `position_sec` - Position to seek to in seconds
    ///
    /// # Examples
    /// ```python
    /// player.seek(30.0)  # Seek to 30 seconds
    /// ```
    fn seek(&mut self, position_sec: f64) -> PyResult<()> {
        if position_sec < 0.0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Position must be non-negative",
            ));
        }

        let duration = Duration::from_secs_f64(position_sec);
        self.inner.seek(duration).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Failed to seek: {}", e))
        })
    }

    /// Check if currently playing
    ///
    /// # Returns
    /// True if currently playing, False otherwise
    ///
    /// # Examples
    /// ```python
    /// if player.is_playing():
    ///     print("Audio is playing")
    /// ```
    fn is_playing(&self) -> bool {
        self.inner.is_playing()
    }

    /// Convenience method to play audio from start
    ///
    /// # Examples
    /// ```python
    /// player.play_from_start()  # Stop, reset, and play
    /// ```
    fn play_from_start(&mut self) -> PyResult<()> {
        self.stop()?;
        self.play()
    }

    /// Convenience method to toggle play/pause
    ///
    /// # Examples
    /// ```python
    /// player.toggle_playback()  # Play if stopped/paused, pause if playing
    /// ```
    fn toggle_playback(&mut self) -> PyResult<()> {
        match self.inner.state() {
            SimplePlaybackState::Playing => self.pause(),
            SimplePlaybackState::Stopped | SimplePlaybackState::Paused => self.play(),
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "AudioPlayer(state={}, volume={:.2}, loop={})",
            match self.inner.state() {
                SimplePlaybackState::Stopped => "STOPPED",
                SimplePlaybackState::Playing => "PLAYING",
                SimplePlaybackState::Paused => "PAUSED",
            },
            self.inner.volume(),
            self.inner.is_loop_enabled()
        )
    }

    fn __str__(&self) -> String {
        format!(
            "AudioPlayer: {} (pos: {:.2}s, vol: {:.0}%)",
            match self.inner.state() {
                SimplePlaybackState::Stopped => "Stopped",
                SimplePlaybackState::Playing => "Playing",
                SimplePlaybackState::Paused => "Paused",
            },
            self.inner.position().as_secs_f64(),
            self.inner.volume() * 100.0
        )
    }

    // Context manager support
    fn __enter__(&mut self, py: Python) -> PyResult<Py<Self>> {
        // Return self for context manager protocol
        Ok(Py::new(py, self.clone())?)
    }

    fn __exit__(
        &mut self,
        _exc_type: Option<&Bound<PyAny>>,
        _exc_val: Option<&Bound<PyAny>>,
        _exc_tb: Option<&Bound<PyAny>>,
    ) -> PyResult<()> {
        // Stop playback when exiting context
        let _ = self.stop(); // Ignore errors during cleanup
        Ok(())
    }
}

// Manual Clone implementation since SimpleAudioPlayer doesn't implement Clone
impl Clone for PyAudioPlayer {
    fn clone(&self) -> Self {
        // For cloning, create a new player with same settings
        // Note: This won't copy the loaded audio or playback state
        match SimpleAudioPlayer::<f32>::new() {
            Ok(mut new_player) => {
                new_player.set_volume(self.inner.volume());
                new_player.set_loop(self.inner.is_loop_enabled());
                Self { inner: new_player }
            }
            Err(_) => {
                // If creation fails, create a dummy player
                // This should rarely happen in practice
                Self {
                    inner: SimpleAudioPlayer::<f32>::new()
                        .unwrap_or_else(|_| panic!("Failed to create audio player during clone")),
                }
            }
        }
    }
}

/// Create a new audio player
///
/// # Returns
/// A new AudioPlayer instance
///
/// # Examples
/// ```python
/// import audio_samples.playback as playback
///
/// player = playback.create_player()
/// ```
#[pyfunction(name = "create_player")]
pub fn create_player() -> PyResult<PyAudioPlayer> {
    PyAudioPlayer::new()
}

/// Convenience function to play audio with default settings
///
/// # Arguments
/// * `audio` - AudioSamples object to play
/// * `volume` - Volume level (0.0 to 1.0, default: 0.8)
/// * `loop_audio` - Whether to loop the audio (default: False)
///
/// # Returns
/// AudioPlayer instance that is already playing
///
/// # Examples
/// ```python
/// import audio_samples as aus
/// import numpy as np
///
/// # Create audio
/// data = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 44100))
/// audio = aus.from_numpy(data, sample_rate=44100)
///
/// # Play with convenience function
/// player = aus.playback.play_audio(audio, volume=0.5, loop_audio=True)
/// ```
#[pyfunction(name = "play_audio")]
#[pyo3(signature = (audio, *, volume=0.8, loop_audio=false))]
pub fn play_audio(
    audio: &PyAudioSamples,
    volume: f32,
    loop_audio: bool,
) -> PyResult<PyAudioPlayer> {
    let mut player = PyAudioPlayer::new()?;
    player.set_volume(volume)?;
    player.set_loop(loop_audio);
    player.load_audio(audio)?;
    player.play()?;
    Ok(player)
}

/// Register playback module with Python
pub fn register_module(_py: Python, m: &Bound<PyModule>) -> PyResult<()> {
    // Classes
    m.add_class::<PyPlaybackState>()?;
    m.add_class::<PyAudioPlayer>()?;

    // Factory functions
    m.add_function(pyo3::wrap_pyfunction!(create_player, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(play_audio, m)?)?;

    // Module constants for convenience
    m.add("STOPPED", PyPlaybackState::STOPPED)?;
    m.add("PLAYING", PyPlaybackState::PLAYING)?;
    m.add("PAUSED", PyPlaybackState::PAUSED)?;

    Ok(())
}
