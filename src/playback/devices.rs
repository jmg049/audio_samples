//! Audio device management and enumeration using CPAL.

use super::{
    error::{PlaybackError, PlaybackResult},
    traits::{AudioDevice, AudioFormatSpec, DeviceInfo, DeviceType, SampleFormat},
};
use crate::AudioSample;
use parking_lot::{Mutex, RwLock};
use std::sync::Arc;
use std::time::Duration;

use cpal::{
    Device, Host, StreamConfig, SupportedStreamConfig,
    traits::{DeviceTrait, HostTrait},
};

/// Manages audio output devices and their capabilities.
///
/// The DeviceManager provides a unified interface for discovering,
/// querying, and managing audio output devices across different platforms.
pub struct DeviceManager {
    host: Host,

    #[cfg(not(feature = "playback"))]
    host: (),

    cached_devices: Arc<RwLock<Vec<DeviceHandle>>>,
    default_output_device: Arc<Mutex<Option<DeviceHandle>>>,
}

impl DeviceManager {
    /// Create a new device manager instance.
    pub fn new() -> PlaybackResult<Self> {
        {
            let host = cpal::default_host();
            let manager = Self {
                host,
                cached_devices: Arc::new(RwLock::new(Vec::new())),
                default_output_device: Arc::new(Mutex::new(None)),
            };

            // Initialize device cache
            let mut manager = manager;
            manager.refresh_devices()?;
            Ok(manager)
        }

        #[cfg(not(feature = "playback"))]
        {
            Err(PlaybackError::FeatureNotEnabled {
                feature: "playback".to_string(),
                operation: "create device manager".to_string(),
            })
        }
    }

    /// Get the default output device.
    pub fn default_output_device(&self) -> PlaybackResult<Option<DeviceHandle>> {
        {
            let cached_default = self.default_output_device.lock().clone();
            if let Some(device) = cached_default {
                return Ok(Some(device));
            }

            if let Some(device) = self.host.default_output_device() {
                let handle = DeviceHandle::new(device)?;
                *self.default_output_device.lock() = Some(handle.clone());
                Ok(Some(handle))
            } else {
                Ok(None)
            }
        }

        #[cfg(not(feature = "playback"))]
        {
            Err(PlaybackError::FeatureNotEnabled {
                feature: "playback".to_string(),
                operation: "get default device".to_string(),
            })
        }
    }

    /// Get all available output devices.
    pub fn output_devices(&mut self) -> PlaybackResult<Vec<DeviceHandle>> {
        {
            let cached = self.cached_devices.read().clone();
            if !cached.is_empty() {
                return Ok(cached);
            }

            // Refresh and return
            drop(cached); // Release read lock
            self.refresh_devices()?;
            Ok(self.cached_devices.read().clone())
        }

        #[cfg(not(feature = "playback"))]
        {
            Err(PlaybackError::FeatureNotEnabled {
                feature: "playback".to_string(),
                operation: "list output devices".to_string(),
            })
        }
    }

    /// Find a device by name.
    pub fn find_device_by_name(&mut self, name: &str) -> PlaybackResult<Option<DeviceHandle>> {
        let devices = self.output_devices()?;
        Ok(devices
            .into_iter()
            .find(|device| device.info().name == name))
    }

    /// Find the best device for the given format.
    pub fn find_best_device(
        &mut self,
        desired_format: &AudioFormatSpec,
    ) -> PlaybackResult<Option<DeviceHandle>> {
        let devices = self.output_devices()?;

        // First try to find an exact match
        for device in &devices {
            if device.supports_format(desired_format) {
                return Ok(Some(device.clone()));
            }
        }

        // If no exact match, find the most compatible device
        let mut best_device = None;
        let mut best_score = 0;

        for device in &devices {
            let score = device.compatibility_score(desired_format);
            if score > best_score {
                best_score = score;
                best_device = Some(device.clone());
            }
        }

        Ok(best_device)
    }

    /// Refresh the device cache.
    pub fn refresh_devices(&mut self) -> PlaybackResult<()> {
        {
            let mut devices = Vec::new();

            for device in
                self.host
                    .output_devices()
                    .map_err(|e| PlaybackError::DeviceEnumeration {
                        source: Box::new(e),
                    })?
            {
                match DeviceHandle::new(device) {
                    Ok(handle) => devices.push(handle),
                    Err(e) => {
                        // Log warning but continue with other devices
                        eprintln!("Warning: Failed to create handle for device: {}", e);
                    }
                }
            }

            *self.cached_devices.write() = devices;

            // Clear cached default device to force refresh
            *self.default_output_device.lock() = None;

            Ok(())
        }

        #[cfg(not(feature = "playback"))]
        {
            Err(PlaybackError::FeatureNotEnabled {
                feature: "playback".to_string(),
                operation: "refresh devices".to_string(),
            })
        }
    }

    /// Get the number of available devices.
    pub fn device_count(&mut self) -> PlaybackResult<usize> {
        Ok(self.output_devices()?.len())
    }

    /// Check if any devices are available.
    pub fn has_devices(&mut self) -> PlaybackResult<bool> {
        Ok(self.device_count()? > 0)
    }
}

impl Default for DeviceManager {
    fn default() -> Self {
        Self::new().expect("Failed to create default DeviceManager")
    }
}

/// A handle to an audio output device.
#[derive(Clone)]
pub struct DeviceHandle {
    device: Arc<Device>,

    #[cfg(not(feature = "playback"))]
    device: Arc<()>,

    info: DeviceInfo,
    supported_formats: Vec<AudioFormatSpec>,
    default_format: AudioFormatSpec,
}

impl DeviceHandle {
    /// Create a new device handle.

    pub fn new(device: Device) -> PlaybackResult<Self> {
        let name = device.name().map_err(|e| PlaybackError::DeviceQuery {
            operation: "get name".to_string(),
            source: Box::new(e),
        })?;

        // Get supported output configs
        let supported_configs: Vec<_> = device
            .supported_output_configs()
            .map_err(|e| PlaybackError::DeviceQuery {
                operation: "get supported configs".to_string(),
                source: Box::new(e),
            })?
            .collect();

        let default_config =
            device
                .default_output_config()
                .map_err(|e| PlaybackError::DeviceQuery {
                    operation: "get default config".to_string(),
                    source: Box::new(e),
                })?;

        // Convert CPAL configs to our format specs
        let mut supported_formats = Vec::new();
        let mut max_channels = 0;
        let mut supported_sample_rates = Vec::new();

        for config in &supported_configs {
            let sample_format = match config.sample_format() {
                cpal::SampleFormat::I16 => SampleFormat::I16,
                cpal::SampleFormat::I32 => SampleFormat::I32,
                cpal::SampleFormat::F32 => SampleFormat::F32,
                cpal::SampleFormat::F64 => SampleFormat::F64,
                _ => continue, // Skip unsupported formats
            };

            let channels = config.channels() as usize;
            max_channels = max_channels.max(channels);

            // Add sample rates from the range
            let min_rate = config.min_sample_rate().0;
            let max_rate = config.max_sample_rate().0;

            // Add common sample rates within the range
            for &rate in &[
                8000, 11025, 16000, 22050, 44100, 48000, 88200, 96000, 176400, 192000,
            ] {
                if rate >= min_rate && rate <= max_rate {
                    supported_sample_rates.push(rate);
                    supported_formats.push(AudioFormatSpec::new(rate, channels, sample_format));
                }
            }
        }

        // Remove duplicates and sort
        supported_sample_rates.sort_unstable();
        supported_sample_rates.dedup();

        // Create default format from CPAL default config
        let default_format = AudioFormatSpec::new(
            default_config.sample_rate().0,
            default_config.channels() as usize,
            match default_config.sample_format() {
                cpal::SampleFormat::I16 => SampleFormat::I16,
                cpal::SampleFormat::I32 => SampleFormat::I32,
                cpal::SampleFormat::F32 => SampleFormat::F32,
                cpal::SampleFormat::F64 => SampleFormat::F64,
                _ => SampleFormat::F32, // Default fallback
            },
        );

        // Detect device type (simplified heuristic)
        let device_type = if name.to_lowercase().contains("usb") {
            DeviceType::Usb
        } else if name.to_lowercase().contains("bluetooth") || name.to_lowercase().contains("bt") {
            DeviceType::Bluetooth
        } else if name.to_lowercase().contains("built-in")
            || name.to_lowercase().contains("internal")
        {
            DeviceType::BuiltIn
        } else if name.to_lowercase().contains("virtual")
            || name.to_lowercase().contains("loopback")
        {
            DeviceType::Virtual
        } else if name.to_lowercase().contains("interface") || name.to_lowercase().contains("audio")
        {
            DeviceType::AudioInterface
        } else {
            DeviceType::Unknown
        };

        let info = DeviceInfo {
            id: format!("{:p}", &device as *const _), // Use pointer as ID
            name: name.clone(),
            device_type,
            is_default: false, // Will be set by DeviceManager if needed
            max_channels,
            supported_sample_rates,
            native_format: Some(default_format.clone()),
        };

        Ok(Self {
            device: Arc::new(device),
            info,
            supported_formats,
            default_format,
        })
    }

    #[cfg(not(feature = "playback"))]
    pub fn new(_: ()) -> PlaybackResult<Self> {
        Err(PlaybackError::FeatureNotEnabled {
            feature: "playback".to_string(),
            operation: "create device handle".to_string(),
        })
    }

    /// Get device information.
    pub fn info(&self) -> &DeviceInfo {
        &self.info
    }

    /// Get the underlying CPAL device.

    pub fn cpal_device(&self) -> &Device {
        &self.device
    }

    /// Calculate a compatibility score for a desired format (0-100).
    fn compatibility_score(&self, desired: &AudioFormatSpec) -> i32 {
        let mut score = 0;

        // Check if exact format is supported
        if self.supports_format(desired) {
            return 100;
        }

        // Score based on sample rate compatibility
        let closest_rate = self
            .info
            .supported_sample_rates
            .iter()
            .min_by_key(|&&rate| (rate as i32 - desired.sample_rate as i32).abs())
            .copied()
            .unwrap_or(44100);

        let rate_diff = (closest_rate as i32 - desired.sample_rate as i32).abs();
        score += (50 - (rate_diff / 1000).min(50)) as i32;

        // Score based on channel compatibility
        if self.info.max_channels >= desired.channels {
            score += 25;
        } else {
            score += (self.info.max_channels as i32 * 25 / desired.channels as i32).max(0);
        }

        // Score based on format compatibility
        if self
            .supported_formats
            .iter()
            .any(|f| f.sample_format == desired.sample_format)
        {
            score += 25;
        } else {
            // Prefer float formats for conversion flexibility
            match desired.sample_format {
                SampleFormat::F32 | SampleFormat::F64 => {
                    if self
                        .supported_formats
                        .iter()
                        .any(|f| f.sample_format.is_float())
                    {
                        score += 15;
                    } else {
                        score += 5;
                    }
                }
                _ => {
                    if self
                        .supported_formats
                        .iter()
                        .any(|f| !f.sample_format.is_float())
                    {
                        score += 10;
                    }
                }
            }
        }

        score
    }
}

impl AudioDevice for DeviceHandle {
    fn device_info(&self) -> DeviceInfo {
        self.info.clone()
    }

    fn supported_formats(&self) -> PlaybackResult<Vec<AudioFormatSpec>> {
        Ok(self.supported_formats.clone())
    }

    fn default_format(&self) -> PlaybackResult<AudioFormatSpec> {
        Ok(self.default_format.clone())
    }

    fn supports_format(&self, format: &AudioFormatSpec) -> bool {
        self.supported_formats
            .iter()
            .any(|f| f.is_compatible(format))
    }

    fn preferred_buffer_size(&self) -> Option<usize> {
        // Use a reasonable default based on sample rate
        match self.default_format.sample_rate {
            rate if rate >= 88200 => Some(2048), // High sample rates
            rate if rate >= 44100 => Some(1024), // Standard rates
            _ => Some(512),                      // Lower sample rates
        }
    }

    fn buffer_size_range(&self) -> (usize, usize) {
        // Conservative range that should work on most devices
        (64, 8192)
    }

    fn output_latency(&self) -> Duration {
        // Estimate based on buffer size and sample rate
        let buffer_size = self.preferred_buffer_size().unwrap_or(1024);
        let sample_rate = self.default_format.sample_rate as f64;
        Duration::from_secs_f64(buffer_size as f64 / sample_rate)
    }

    fn is_available(&self) -> bool {
        {
            // Try to query device name as a simple availability check
            self.device.name().is_ok()
        }

        #[cfg(not(feature = "playback"))]
        {
            false
        }
    }
}
