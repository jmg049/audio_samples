//! AudioDynamicRange trait implementation for Python bindings.
//!
//! This module provides Python access to professional-grade dynamic range
//! processing operations including compression, limiting, side-chain processing,
//! and envelope following for audio production and mastering applications.

use super::{PyAudioSamples, utils::*};
use crate::operations::{
    AudioDynamicRange,
    types::{CompressorConfig, LimiterConfig, EnvelopeFollowerConfig, KneeType},
};
use pyo3::prelude::*;

impl PyAudioSamples {
    /// Apply dynamic range compression to the audio signal.
    ///
    /// Compression reduces the dynamic range by attenuating signals above
    /// a threshold, useful for controlling peaks and achieving more consistent
    /// loudness levels in audio production.
    ///
    /// # Arguments
    /// * `threshold` - Threshold in dB above which compression is applied (default: -20.0)
    /// * `ratio` - Compression ratio (1.0 = no compression, higher = more compression, default: 4.0)
    /// * `attack` - Attack time in seconds (default: 0.003)
    /// * `release` - Release time in seconds (default: 0.1)
    /// * `knee` - Knee type ('hard' or 'soft', default: 'soft')
    /// * `knee_width` - Knee width in dB for soft knee (default: 2.0)
    /// * `makeup_gain` - Makeup gain in dB to compensate for level reduction (default: 0.0)
    ///
    /// # Returns
    /// New AudioSamples object with compression applied
    ///
    /// # Examples
    /// ```python
    /// import audio_samples as aus
    /// import numpy as np
    ///
    /// # Create audio with dynamic range
    /// t = np.linspace(0, 1, 44100)
    /// signal = np.sin(2 * np.pi * 440 * t) * np.random.uniform(0.1, 1.0, 44100)
    /// audio = aus.from_numpy(signal, sample_rate=44100)
    ///
    /// # Apply gentle compression
    /// compressed = audio.compress(threshold=-15, ratio=3.0, attack=0.005, release=0.05)
    ///
    /// # Heavy compression for limiting dynamic range
    /// limited = audio.compress(threshold=-10, ratio=10.0, knee='hard')
    ///
    /// # Vintage-style compression with slow attack
    /// vintage = audio.compress(threshold=-18, ratio=2.5, attack=0.1, release=0.3)
    /// ```
    pub(crate) fn compress_impl(
        &self,
        threshold: Option<f64>,
        ratio: Option<f64>,
        attack: Option<f64>,
        release: Option<f64>,
        knee: Option<&str>,
        knee_width: Option<f64>,
        makeup_gain: Option<f64>,
    ) -> PyResult<PyAudioSamples> {
        let knee_type = if let Some(knee_str) = knee {
            validate_string_param("knee", knee_str, &["hard", "soft"])?;
            match knee_str {
                "hard" => KneeType::Hard,
                "soft" => KneeType::Soft,
                _ => unreachable!(),
            }
        } else {
            KneeType::Soft
        };

        let config = CompressorConfig {
            threshold: threshold.unwrap_or(-20.0),
            ratio: ratio.unwrap_or(4.0),
            attack: attack.unwrap_or(0.003),
            release: release.unwrap_or(0.1),
            knee: knee_type,
            knee_width: knee_width.unwrap_or(2.0),
            makeup_gain: makeup_gain.unwrap_or(0.0),
        };

        let compressed = self
            .with_inner(|inner| inner.compress(&config))
            .map_err(map_error)?;
        Ok(PyAudioSamples::from_inner(compressed))
    }

    /// Apply dynamic range compression in-place.
    ///
    /// Same as compress() but modifies the current AudioSamples object
    /// instead of returning a new one.
    ///
    /// # Arguments
    /// Same as compress()
    ///
    /// # Examples
    /// ```python
    /// # Modify audio directly
    /// audio.compress_(threshold=-12, ratio=4.0, attack=0.01)
    /// ```
    pub(crate) fn compress_in_place_impl(
        &mut self,
        threshold: Option<f64>,
        ratio: Option<f64>,
        attack: Option<f64>,
        release: Option<f64>,
        knee: Option<&str>,
        knee_width: Option<f64>,
        makeup_gain: Option<f64>,
    ) -> PyResult<()> {
        let knee_type = if let Some(knee_str) = knee {
            validate_string_param("knee", knee_str, &["hard", "soft"])?;
            match knee_str {
                "hard" => KneeType::Hard,
                "soft" => KneeType::Soft,
                _ => unreachable!(),
            }
        } else {
            KneeType::Soft
        };

        let config = CompressorConfig {
            threshold: threshold.unwrap_or(-20.0),
            ratio: ratio.unwrap_or(4.0),
            attack: attack.unwrap_or(0.003),
            release: release.unwrap_or(0.1),
            knee: knee_type,
            knee_width: knee_width.unwrap_or(2.0),
            makeup_gain: makeup_gain.unwrap_or(0.0),
        };

        self.with_inner_mut(|inner| {
            inner.compress_in_place(&config)?;
            Ok(())
        })
        .map_err(map_error)
    }

    /// Apply peak limiting to prevent signal from exceeding a threshold.
    ///
    /// A limiter is essentially a compressor with a very high ratio and
    /// fast attack time. This implementation includes lookahead processing
    /// to prevent artifacts and maintain transparency.
    ///
    /// # Arguments
    /// * `threshold` - Maximum peak level in dB (default: -0.1)
    /// * `lookahead` - Lookahead time in seconds for artifact-free limiting (default: 0.005)
    /// * `release` - Release time in seconds (default: 0.05)
    /// * `knee_width` - Soft knee width in dB (default: 0.5)
    ///
    /// # Returns
    /// New AudioSamples object with limiting applied
    ///
    /// # Examples
    /// ```python
    /// # Prevent clipping with transparent limiting
    /// limited = audio.limit(threshold=-0.3, lookahead=0.005)
    ///
    /// # Aggressive limiting for maximum loudness
    /// brick_wall = audio.limit(threshold=-0.1, lookahead=0.01, release=0.01)
    ///
    /// # Gentle limiting with longer release
    /// smooth_limit = audio.limit(threshold=-1.0, release=0.2)
    /// ```
    pub(crate) fn limit_impl(
        &self,
        threshold: Option<f64>,
        lookahead: Option<f64>,
        release: Option<f64>,
        knee_width: Option<f64>,
    ) -> PyResult<PyAudioSamples> {
        let config = LimiterConfig {
            threshold: threshold.unwrap_or(-0.1),
            lookahead: lookahead.unwrap_or(0.005),
            release: release.unwrap_or(0.05),
            knee_width: knee_width.unwrap_or(0.5),
        };

        let limited = self
            .with_inner(|inner| inner.limit(&config))
            .map_err(map_error)?;
        Ok(PyAudioSamples::from_inner(limited))
    }

    /// Apply peak limiting in-place.
    ///
    /// Same as limit() but modifies the current AudioSamples object
    /// instead of returning a new one.
    ///
    /// # Arguments
    /// Same as limit()
    ///
    /// # Examples
    /// ```python
    /// # Apply limiting directly to audio
    /// audio.limit_(threshold=-0.5, lookahead=0.01)
    /// ```
    pub(crate) fn limit_in_place_impl(
        &mut self,
        threshold: Option<f64>,
        lookahead: Option<f64>,
        release: Option<f64>,
        knee_width: Option<f64>,
    ) -> PyResult<()> {
        let config = LimiterConfig {
            threshold: threshold.unwrap_or(-0.1),
            lookahead: lookahead.unwrap_or(0.005),
            release: release.unwrap_or(0.05),
            knee_width: knee_width.unwrap_or(0.5),
        };

        self.with_inner_mut(|inner| {
            inner.limit_in_place(&config)?;
            Ok(())
        })
        .map_err(map_error)
    }

    /// Apply side-chain compression using another audio signal as the control source.
    ///
    /// Side-chain compression triggers compression based on the level of
    /// a separate control signal rather than the input signal itself.
    /// Commonly used for ducking effects and rhythmic pumping.
    ///
    /// # Arguments
    /// * `sidechain_signal` - AudioSamples object to use as the control signal
    /// * `threshold` - Threshold in dB for compression (default: -20.0)
    /// * `ratio` - Compression ratio (default: 4.0)
    /// * `attack` - Attack time in seconds (default: 0.001)
    /// * `release` - Release time in seconds (default: 0.1)
    /// * `knee` - Knee type ('hard' or 'soft', default: 'soft')
    /// * `knee_width` - Knee width in dB for soft knee (default: 2.0)
    ///
    /// # Returns
    /// New AudioSamples object with side-chain compression applied
    ///
    /// # Examples
    /// ```python
    /// # Create kick drum pattern for side-chain trigger
    /// kick_pattern = create_kick_pattern()  # Your kick drum signal
    /// 
    /// # Apply side-chain compression to bass using kick as trigger
    /// ducked_bass = bass_audio.sidechain_compress(
    ///     sidechain_signal=kick_pattern,
    ///     threshold=-25,
    ///     ratio=8.0,
    ///     attack=0.001,
    ///     release=0.15
    /// )
    ///
    /// # Pumping effect with short release
    /// pumping = synth_pad.sidechain_compress(
    ///     sidechain_signal=kick_pattern,
    ///     threshold=-30,
    ///     ratio=6.0,
    ///     release=0.05
    /// )
    /// ```
    pub(crate) fn sidechain_compress_impl(
        &self,
        sidechain_signal: &PyAudioSamples,
        threshold: Option<f64>,
        ratio: Option<f64>,
        attack: Option<f64>,
        release: Option<f64>,
        knee: Option<&str>,
        knee_width: Option<f64>,
    ) -> PyResult<PyAudioSamples> {
        // Validate sample rate and channel compatibility
        if self.sample_rate() != sidechain_signal.sample_rate() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Sample rates must match for side-chain compression",
            ));
        }

        let knee_type = if let Some(knee_str) = knee {
            validate_string_param("knee", knee_str, &["hard", "soft"])?;
            match knee_str {
                "hard" => KneeType::Hard,
                "soft" => KneeType::Soft,
                _ => unreachable!(),
            }
        } else {
            KneeType::Soft
        };

        let config = CompressorConfig {
            threshold: threshold.unwrap_or(-20.0),
            ratio: ratio.unwrap_or(4.0),
            attack: attack.unwrap_or(0.001),
            release: release.unwrap_or(0.1),
            knee: knee_type,
            knee_width: knee_width.unwrap_or(2.0),
            makeup_gain: 0.0, // No makeup gain for side-chain
        };

        // Convert sidechain signal to compatible format
        let sidechain_f64 = sidechain_signal.as_f64().map_err(map_error)?;
        
        let compressed = self
            .with_inner(|inner| inner.sidechain_compress(&sidechain_f64, &config))
            .map_err(map_error)?;
        Ok(PyAudioSamples::from_inner(compressed))
    }

    /// Compute envelope following to extract the amplitude envelope of the signal.
    ///
    /// An envelope follower tracks the amplitude contour of an audio signal,
    /// useful for analysis, control signals, and dynamic processing.
    ///
    /// # Arguments
    /// * `attack` - Attack time in seconds for envelope rise (default: 0.001)
    /// * `release` - Release time in seconds for envelope decay (default: 0.1)
    /// * `mode` - Envelope detection mode ('peak' or 'rms', default: 'rms')
    ///
    /// # Returns
    /// NumPy array containing the envelope signal
    ///
    /// # Examples
    /// ```python
    /// # Extract RMS envelope for level analysis
    /// rms_envelope = audio.envelope_follower(attack=0.01, release=0.1, mode='rms')
    ///
    /// # Fast peak tracking for transient detection
    /// peak_envelope = audio.envelope_follower(attack=0.0001, release=0.05, mode='peak')
    ///
    /// # Slow envelope for musical dynamics
    /// musical_envelope = audio.envelope_follower(attack=0.1, release=0.5, mode='rms')
    ///
    /// # Use envelope for visualization
    /// import matplotlib.pyplot as plt
    /// time = np.linspace(0, len(audio) / audio.sample_rate, len(audio))
    /// plt.plot(time, audio.to_numpy(), alpha=0.5, label='Audio')
    /// plt.plot(time, rms_envelope, label='RMS Envelope')
    /// plt.legend()
    /// ```
    pub(crate) fn envelope_follower_impl(
        &self,
        py: Python,
        attack: Option<f64>,
        release: Option<f64>,
        mode: Option<&str>,
    ) -> PyResult<PyObject> {
        let detection_mode = if let Some(mode_str) = mode {
            validate_string_param("mode", mode_str, &["peak", "rms"])?;
            mode_str
        } else {
            "rms"
        };

        let config = EnvelopeFollowerConfig {
            attack: attack.unwrap_or(0.001),
            release: release.unwrap_or(0.1),
            mode: match detection_mode {
                "peak" => crate::operations::types::EnvelopeMode::Peak,
                "rms" => crate::operations::types::EnvelopeMode::RMS,
                _ => unreachable!(),
            },
        };

        let envelope = self
            .with_inner(|inner| inner.envelope_follower(&config))
            .map_err(map_error)?;

        array1_to_numpy(py, ndarray::Array1::from(envelope))
    }

    /// Get the gain reduction curve applied by the compressor for analysis.
    ///
    /// Returns the amount of gain reduction (in dB) applied at each sample,
    /// useful for analyzing compressor behavior and creating visual feedback.
    ///
    /// # Arguments
    /// * `threshold` - Compression threshold in dB (default: -20.0)
    /// * `ratio` - Compression ratio (default: 4.0)
    /// * `knee` - Knee type ('hard' or 'soft', default: 'soft')
    /// * `knee_width` - Knee width in dB for soft knee (default: 2.0)
    ///
    /// # Returns
    /// NumPy array containing gain reduction values in dB (negative values)
    ///
    /// # Examples
    /// ```python
    /// # Analyze compressor behavior
    /// gain_reduction = audio.compressor_gain_reduction(
    ///     threshold=-15, ratio=4.0, knee='soft'
    /// )
    ///
    /// # Visualize compression activity
    /// import matplotlib.pyplot as plt
    /// time = np.linspace(0, len(audio) / audio.sample_rate, len(gain_reduction))
    /// plt.plot(time, gain_reduction)
    /// plt.ylabel('Gain Reduction (dB)')
    /// plt.xlabel('Time (s)')
    /// plt.title('Compressor Activity')
    /// ```
    pub(crate) fn compressor_gain_reduction_impl(
        &self,
        py: Python,
        threshold: Option<f64>,
        ratio: Option<f64>,
        knee: Option<&str>,
        knee_width: Option<f64>,
    ) -> PyResult<PyObject> {
        let knee_type = if let Some(knee_str) = knee {
            validate_string_param("knee", knee_str, &["hard", "soft"])?;
            match knee_str {
                "hard" => KneeType::Hard,
                "soft" => KneeType::Soft,
                _ => unreachable!(),
            }
        } else {
            KneeType::Soft
        };

        let config = CompressorConfig {
            threshold: threshold.unwrap_or(-20.0),
            ratio: ratio.unwrap_or(4.0),
            attack: 0.0, // Not used for gain reduction analysis
            release: 0.0, // Not used for gain reduction analysis
            knee: knee_type,
            knee_width: knee_width.unwrap_or(2.0),
            makeup_gain: 0.0,
        };

        let gain_reduction = self
            .with_inner(|inner| inner.compressor_gain_reduction(&config))
            .map_err(map_error)?;

        array1_to_numpy(py, ndarray::Array1::from(gain_reduction))
    }
}