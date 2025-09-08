//! AudioDynamicRange trait implementation for Python bindings.
//!
//! This module provides Python access to professional-grade dynamic range
//! processing operations including compression, limiting, side-chain processing,
//! and envelope following for audio production and mastering applications.

use super::{PyAudioSamples, utils::*};
use crate::operations::{
    AudioDynamicRange,
    types::{CompressorConfig, DynamicRangeMethod, KneeType, LimiterConfig},
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
        // Parse knee type
        let knee_type = match knee.unwrap_or("soft") {
            "hard" => KneeType::Hard,
            "soft" => KneeType::Soft,
            _ => {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "Invalid knee type. Must be 'hard' or 'soft'",
                ));
            }
        };

        // Create compressor configuration
        let config = CompressorConfig {
            threshold_db: threshold.unwrap_or(-20.0),
            ratio: ratio.unwrap_or(4.0),
            attack_ms: attack.unwrap_or(0.003) * 1000.0, // Convert to ms
            release_ms: release.unwrap_or(0.1) * 1000.0, // Convert to ms
            makeup_gain_db: makeup_gain.unwrap_or(0.0),
            knee_type,
            knee_width_db: knee_width.unwrap_or(2.0),
            detection_method: DynamicRangeMethod::Rms,
            side_chain: crate::operations::types::SideChainConfig::disabled(),
            lookahead_ms: 0.0,
        };

        // Apply compression
        let mut result = self.copy();
        let sample_rate = result.sample_rate() as f64;
        result
            .mutate_inner(|inner| inner.apply_compressor(&config, sample_rate))
            .map_err(map_error)?;
        Ok(result)
    }

    /// Apply limiting to the audio signal.
    ///
    /// Prevents the signal from exceeding the specified ceiling level.
    /// Limiting is typically used as the final stage in mastering to
    /// prevent clipping and maximize loudness.
    ///
    /// # Arguments
    /// * `ceiling` - Maximum output level in dB (default: -0.1)
    /// * `release` - Release time in seconds (default: 0.05)
    /// * `lookahead` - Lookahead time in seconds (default: 0.005)
    ///
    /// # Returns
    /// New AudioSamples object with limiting applied
    pub(crate) fn limit_impl(
        &self,
        ceiling: Option<f64>,
        release: Option<f64>,
        lookahead: Option<f64>,
    ) -> PyResult<PyAudioSamples> {
        // Create limiter configuration
        let config = LimiterConfig {
            ceiling_db: ceiling.unwrap_or(-0.1),
            attack_ms: 0.1,                               // Default attack time
            release_ms: release.unwrap_or(0.05) * 1000.0, // Convert to ms
            knee_type: KneeType::Hard,
            knee_width_db: 0.1,
            detection_method: DynamicRangeMethod::Peak,
            side_chain: crate::operations::types::SideChainConfig::disabled(),
            lookahead_ms: lookahead.unwrap_or(0.005) * 1000.0, // Convert to ms
            isp_limiting: true,
        };

        // Apply limiting
        let mut result = self.copy();
        let sample_rate = result.sample_rate() as f64;
        result
            .mutate_inner(|inner| inner.apply_limiter(&config, sample_rate))
            .map_err(map_error)?;
        Ok(result)
    }

    /// Apply dynamic range compression in-place.
    ///
    /// Same as compress() but modifies the current AudioSamples object.
    ///
    /// # Arguments
    /// Same as compress()
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
        // Parse knee type
        let knee_type = match knee.unwrap_or("soft") {
            "hard" => KneeType::Hard,
            "soft" => KneeType::Soft,
            _ => {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "Invalid knee type. Must be 'hard' or 'soft'",
                ));
            }
        };

        // Create compressor configuration
        let config = CompressorConfig {
            threshold_db: threshold.unwrap_or(-20.0),
            ratio: ratio.unwrap_or(4.0),
            attack_ms: attack.unwrap_or(0.003) * 1000.0, // Convert to ms
            release_ms: release.unwrap_or(0.1) * 1000.0, // Convert to ms
            makeup_gain_db: makeup_gain.unwrap_or(0.0),
            knee_type,
            knee_width_db: knee_width.unwrap_or(2.0),
            detection_method: DynamicRangeMethod::Rms,
            side_chain: crate::operations::types::SideChainConfig::disabled(),
            lookahead_ms: 0.0,
        };

        // Apply compression in-place
        let sample_rate = self.sample_rate() as f64;
        self.mutate_inner(|inner| inner.apply_compressor(&config, sample_rate))
            .map_err(map_error)
    }

    /// Apply limiting in-place.
    ///
    /// Same as limit() but modifies the current AudioSamples object.
    ///
    /// # Arguments
    /// Same as limit()
    pub(crate) fn limit_in_place_impl(
        &mut self,
        ceiling: Option<f64>,
        release: Option<f64>,
        lookahead: Option<f64>,
    ) -> PyResult<()> {
        // Create limiter configuration
        let config = LimiterConfig {
            ceiling_db: ceiling.unwrap_or(-0.1),
            attack_ms: 0.1,                               // Default attack time
            release_ms: release.unwrap_or(0.05) * 1000.0, // Convert to ms
            knee_type: KneeType::Hard,
            knee_width_db: 0.1,
            detection_method: DynamicRangeMethod::Peak,
            side_chain: crate::operations::types::SideChainConfig::disabled(),
            lookahead_ms: lookahead.unwrap_or(0.005) * 1000.0, // Convert to ms
            isp_limiting: true,
        };

        // Apply limiting in-place
        let sample_rate = self.sample_rate() as f64;
        self.mutate_inner(|inner| inner.apply_limiter(&config, sample_rate))
            .map_err(map_error)
    }
}
