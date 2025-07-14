//! AudioProcessing trait implementation for Python bindings.
//!
//! This module provides both in-place and functional audio processing operations
//! following pandas-style conventions. Methods ending with '_' modify the object
//! in-place, while methods without '_' return new objects.

use super::{PyAudioSamples, utils::*};
use crate::operations::{AudioProcessing, NormalizationMethod};
use pyo3::prelude::*;

impl PyAudioSamples {
    /// Parse normalization method from string
    pub(crate) fn parse_normalization_method_impl(method: &str) -> PyResult<NormalizationMethod> {
        match method.to_lowercase().as_str() {
            "minmax" => Ok(NormalizationMethod::MinMax),
            "zscore" | "z_score" => Ok(NormalizationMethod::ZScore),
            "mean" => Ok(NormalizationMethod::Mean),
            "median" => Ok(NormalizationMethod::Median),
            "peak" => Ok(NormalizationMethod::Peak),
            _ => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Invalid normalization method: '{}'. Valid options: 'minmax', 'zscore', 'mean', 'median', 'peak'",
                method
            ))),
        }
    }
}

impl PyAudioSamples {
    // ==========================================
    // IN-PLACE OPERATIONS (modify self)
    // ==========================================

    /// Normalize audio samples in-place using the specified method.
    ///
    /// # Arguments
    /// * `method` - Normalization method ('minmax', 'zscore', 'mean', 'median', 'peak')
    /// * `min_val` - Target minimum value (only for 'minmax' method)
    /// * `max_val` - Target maximum value (only for 'minmax' method)
    ///
    /// # Examples
    /// ```python
    /// import numpy as np
    /// import audio_samples as aus
    ///
    /// audio = aus.from_numpy(np.random.normal(0, 2, 44100), sample_rate=44100)
    ///
    /// # Min-max normalization to [-1, 1] range
    /// audio.normalize_(method='minmax', min_val=-1.0, max_val=1.0)
    ///
    /// # Z-score normalization (zero mean, unit variance)
    /// audio.normalize_(method='zscore')
    ///
    /// # Peak normalization (scale by peak value)
    /// audio.normalize_(method='peak')
    /// ```
    pub(crate) fn normalize_inplace_impl(
        &mut self,
        method: &str,
        min_val: f64,
        max_val: f64,
    ) -> PyResult<()> {
        let norm_method = Self::parse_normalization_method_impl(method)?;
        self.mutate_inner(|inner| inner.normalize(min_val, max_val, norm_method))
            .map_err(map_error)
    }

    /// Scale all audio samples by a constant factor in-place.
    ///
    /// # Arguments
    /// * `factor` - Scaling factor (1.0 = no change, 0.5 = half volume, 2.0 = double volume)
    ///
    /// # Examples
    /// ```python
    /// # Reduce volume by half
    /// audio.scale_(0.5)
    ///
    /// # Double the volume (careful of clipping!)
    /// audio.scale_(2.0)
    /// ```
    pub(crate) fn scale_inplace_impl(&mut self, factor: f64) -> PyResult<()> {
        self.mutate_inner(|inner| inner.scale(factor))
            .map_err(map_error)
    }

    /// Apply a windowing function to the audio samples in-place.
    ///
    /// # Arguments
    /// * `window` - NumPy array with window coefficients (must match audio length)
    ///
    /// # Examples
    /// ```python
    /// import numpy as np
    ///
    /// # Apply Hanning window
    /// window = np.hanning(len(audio))
    /// audio.apply_window_(window)
    ///
    /// # Apply custom window
    /// custom_window = np.exp(-np.linspace(0, 5, len(audio)))  # Exponential decay
    /// audio.apply_window_(custom_window)
    /// ```
    pub(crate) fn apply_window_inplace_impl(&mut self, window: &Bound<PyAny>) -> PyResult<()> {
        // Convert numpy array to Vec<f64>
        let window_array = if let Ok(array_f64) = window.extract::<numpy::PyReadonlyArray1<f64>>() {
            array_f64.as_array().to_vec()
        } else if let Ok(array_f32) = window.extract::<numpy::PyReadonlyArray1<f32>>() {
            array_f32.as_array().iter().map(|&x| x as f64).collect()
        } else {
            return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                "Window must be a 1D numpy array of f32 or f64",
            ));
        };

        self.mutate_inner(|inner| inner.apply_window(&window_array))
            .map_err(map_error)
    }

    /// Apply a digital filter to the audio samples in-place.
    ///
    /// # Arguments
    /// * `coeffs` - NumPy array with filter coefficients
    /// * `mode` - Convolution mode ('full', 'same', 'valid')
    ///
    /// # Examples
    /// ```python
    /// # Simple moving average filter
    /// moving_avg = np.ones(5) / 5
    /// audio.apply_filter_(moving_avg, mode='same')
    ///
    /// # High-pass filter coefficients
    /// high_pass = np.array([1, -1])  # Simple difference filter
    /// audio.apply_filter_(high_pass, mode='same')
    /// ```
    pub(crate) fn apply_filter_inplace_impl(
        &mut self,
        coeffs: &Bound<PyAny>,
        mode: &str,
    ) -> PyResult<()> {
        validate_string_param("mode", mode, &["full", "same", "valid"])?;

        // Convert numpy array to Vec<f64>
        let filter_coeffs = if let Ok(array_f64) = coeffs.extract::<numpy::PyReadonlyArray1<f64>>()
        {
            array_f64.as_array().to_vec()
        } else if let Ok(array_f32) = coeffs.extract::<numpy::PyReadonlyArray1<f32>>() {
            array_f32.as_array().iter().map(|&x| x as f64).collect()
        } else {
            return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                "Filter coefficients must be a 1D numpy array of f32 or f64",
            ));
        };

        self.mutate_inner(|inner| inner.apply_filter(&filter_coeffs))
            .map_err(map_error)
    }

    /// Apply μ-law compression to the audio samples in-place.
    ///
    /// # Arguments
    /// * `mu` - μ-law parameter (default: 255 for standard μ-law)
    ///
    /// # Examples
    /// ```python
    /// # Standard μ-law compression (telecommunications)
    /// audio.mu_compress_(mu=255)
    ///
    /// # Lighter compression
    /// audio.mu_compress_(mu=100)
    /// ```
    pub(crate) fn mu_compress_inplace_impl(&mut self, mu: f64) -> PyResult<()> {
        self.mutate_inner(|inner| inner.mu_compress(mu))
            .map_err(map_error)
    }

    /// Apply μ-law expansion (decompression) to the audio samples in-place.
    ///
    /// # Arguments
    /// * `mu` - μ-law parameter (should match compression parameter)
    ///
    /// # Examples
    /// ```python
    /// # Expand previously compressed audio
    /// audio.mu_expand_(mu=255)
    /// ```
    pub(crate) fn mu_expand_inplace_impl(&mut self, mu: f64) -> PyResult<()> {
        self.mutate_inner(|inner| inner.mu_expand(mu))
            .map_err(map_error)
    }

    /// Apply a low-pass filter in-place.
    ///
    /// # Arguments
    /// * `cutoff_hz` - Cutoff frequency in Hz
    ///
    /// # Examples
    /// ```python
    /// # Remove frequencies above 1kHz
    /// audio.low_pass_filter_(cutoff_hz=1000)
    /// ```
    pub(crate) fn low_pass_filter_inplace_impl(&mut self, cutoff_hz: f64) -> PyResult<()> {
        self.mutate_inner(|inner| inner.low_pass_filter(cutoff_hz))
            .map_err(map_error)
    }

    /// Apply a high-pass filter in-place.
    ///
    /// # Arguments
    /// * `cutoff_hz` - Cutoff frequency in Hz
    ///
    /// # Examples
    /// ```python
    /// # Remove frequencies below 100Hz (remove rumble)
    /// audio.high_pass_filter_(cutoff_hz=100)
    /// ```
    pub(crate) fn high_pass_filter_inplace_impl(&mut self, cutoff_hz: f64) -> PyResult<()> {
        self.mutate_inner(|inner| inner.high_pass_filter(cutoff_hz))
            .map_err(map_error)
    }

    /// Apply a band-pass filter in-place.
    ///
    /// # Arguments
    /// * `low_hz` - Low cutoff frequency in Hz
    /// * `high_hz` - High cutoff frequency in Hz
    ///
    /// # Examples
    /// ```python
    /// # Keep only frequencies between 300Hz and 3kHz (voice range)
    /// audio.band_pass_filter_(low_hz=300, high_hz=3000)
    /// ```
    pub(crate) fn band_pass_filter_inplace_impl(
        &mut self,
        low_hz: f64,
        high_hz: f64,
    ) -> PyResult<()> {
        if low_hz >= high_hz {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Low frequency must be less than high frequency",
            ));
        }
        self.mutate_inner(|inner| inner.band_pass_filter(low_hz, high_hz))
            .map_err(map_error)
    }

    /// Remove DC offset by subtracting the mean value in-place.
    ///
    /// # Examples
    /// ```python
    /// # Remove any DC bias from the signal
    /// audio.remove_dc_offset_()
    /// ```
    pub(crate) fn remove_dc_offset_inplace_impl(&mut self) -> PyResult<()> {
        self.mutate_inner(|inner| inner.remove_dc_offset())
            .map_err(map_error)
    }

    /// Clip audio samples to the specified range in-place.
    ///
    /// # Arguments
    /// * `min_val` - Minimum allowed value
    /// * `max_val` - Maximum allowed value
    ///
    /// # Examples
    /// ```python
    /// # Clip to prevent severe distortion
    /// audio.clip_(min_val=-1.0, max_val=1.0)
    ///
    /// # Soft clipping
    /// audio.clip_(min_val=-0.8, max_val=0.8)
    /// ```
    pub(crate) fn clip_inplace_impl(&mut self, min_val: f64, max_val: f64) -> PyResult<()> {
        if min_val >= max_val {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "min_val must be less than max_val",
            ));
        }
        self.mutate_inner(|inner| inner.clip(min_val, max_val))
            .map_err(map_error)
    }

    // ==========================================
    // FUNCTIONAL OPERATIONS (return new objects)
    // ==========================================

    /// Normalize audio samples using the specified method (functional version).
    ///
    /// Returns a new AudioSamples object, leaving the original unchanged.
    ///
    /// # Arguments
    /// * `method` - Normalization method ('minmax', 'zscore', 'mean', 'median', 'peak')
    /// * `min_val` - Target minimum value (only for 'minmax' method)
    /// * `max_val` - Target maximum value (only for 'minmax' method)
    ///
    /// # Returns
    /// New AudioSamples object with normalized data
    ///
    /// # Examples
    /// ```python
    /// # Create normalized copy (original unchanged)
    /// normalized = audio.normalize(method='zscore')
    /// peak_normalized = audio.normalize(method='peak')
    /// ```
    pub(crate) fn normalize_impl(
        &self,
        method: &str,
        min_val: f64,
        max_val: f64,
    ) -> PyResult<PyAudioSamples> {
        let mut result = self.copy();
        result.normalize_inplace_impl(method, min_val, max_val)?;
        Ok(result)
    }

    /// Scale all audio samples by a constant factor (functional version).
    ///
    /// # Arguments
    /// * `factor` - Scaling factor
    ///
    /// # Returns
    /// New AudioSamples object with scaled data
    ///
    /// # Examples
    /// ```python
    /// # Create scaled copy (original unchanged)
    /// quieter = audio.scale(0.5)
    /// louder = audio.scale(2.0)
    /// ```
    pub(crate) fn scale_functional_impl(&self, factor: f64) -> PyResult<PyAudioSamples> {
        let mut result = self.copy();
        result.scale_inplace_impl(factor)?;
        Ok(result)
    }

    /// Apply a windowing function (functional version).
    ///
    /// # Arguments
    /// * `window` - NumPy array with window coefficients
    ///
    /// # Returns
    /// New AudioSamples object with windowed data
    ///
    /// # Examples
    /// ```python
    /// # Create windowed copy
    /// windowed = audio.apply_window(np.hanning(len(audio)))
    /// ```
    pub(crate) fn apply_window_functional_impl(
        &self,
        window: &Bound<PyAny>,
    ) -> PyResult<PyAudioSamples> {
        let mut result = self.copy();
        result.apply_window_inplace_impl(window)?;
        Ok(result)
    }

    /// Apply a digital filter (functional version).
    ///
    /// # Arguments
    /// * `coeffs` - NumPy array with filter coefficients
    /// * `mode` - Convolution mode ('full', 'same', 'valid')
    ///
    /// # Returns
    /// New AudioSamples object with filtered data
    ///
    /// # Examples
    /// ```python
    /// # Create filtered copy
    /// filtered = audio.apply_filter(np.ones(5) / 5, mode='same')
    /// ```
    pub(crate) fn apply_filter_impl(
        &self,
        coeffs: &Bound<PyAny>,
        mode: &str,
    ) -> PyResult<PyAudioSamples> {
        let mut result = self.copy();
        result.apply_filter_inplace_impl(coeffs, mode)?;
        Ok(result)
    }

    /// Apply μ-law compression (functional version).
    ///
    /// # Arguments
    /// * `mu` - μ-law parameter
    ///
    /// # Returns
    /// New AudioSamples object with compressed data
    ///
    /// # Examples
    /// ```python
    /// compressed = audio.mu_compress(mu=255)
    /// ```
    pub(crate) fn mu_compress_impl(&self, mu: f64) -> PyResult<PyAudioSamples> {
        let mut result = self.copy();
        result.mu_compress_inplace_impl(mu)?;
        Ok(result)
    }

    /// Apply μ-law expansion (functional version).
    ///
    /// # Arguments
    /// * `mu` - μ-law parameter
    ///
    /// # Returns
    /// New AudioSamples object with expanded data
    ///
    /// # Examples
    /// ```python
    /// expanded = compressed_audio.mu_expand(mu=255)
    /// ```
    pub(crate) fn mu_expand_impl(&self, mu: f64) -> PyResult<PyAudioSamples> {
        let mut result = self.copy();
        result.mu_expand_inplace_impl(mu)?;
        Ok(result)
    }

    /// Apply a low-pass filter (functional version).
    ///
    /// # Arguments
    /// * `cutoff_hz` - Cutoff frequency in Hz
    ///
    /// # Returns
    /// New AudioSamples object with filtered data
    ///
    /// # Examples
    /// ```python
    /// low_passed = audio.low_pass_filter(cutoff_hz=1000)
    /// ```
    pub(crate) fn low_pass_filter_functional_impl(
        &self,
        cutoff_hz: f64,
    ) -> PyResult<PyAudioSamples> {
        let mut result = self.copy();
        result.low_pass_filter_inplace_impl(cutoff_hz)?;
        Ok(result)
    }

    /// Apply a high-pass filter (functional version).
    ///
    /// # Arguments
    /// * `cutoff_hz` - Cutoff frequency in Hz
    ///
    /// # Returns
    /// New AudioSamples object with filtered data
    ///
    /// # Examples
    /// ```python
    /// high_passed = audio.high_pass_filter(cutoff_hz=100)
    /// ```
    pub(crate) fn high_pass_filter_functional_impl(
        &self,
        cutoff_hz: f64,
    ) -> PyResult<PyAudioSamples> {
        let mut result = self.copy();
        result.high_pass_filter_inplace_impl(cutoff_hz)?;
        Ok(result)
    }

    /// Apply a band-pass filter (functional version).
    ///
    /// # Arguments
    /// * `low_hz` - Low cutoff frequency in Hz
    /// * `high_hz` - High cutoff frequency in Hz
    ///
    /// # Returns
    /// New AudioSamples object with filtered data
    ///
    /// # Examples
    /// ```python
    /// band_passed = audio.band_pass_filter(low_hz=300, high_hz=3000)
    /// ```
    pub(crate) fn band_pass_filter_functional_impl(
        &self,
        low_hz: f64,
        high_hz: f64,
    ) -> PyResult<PyAudioSamples> {
        let mut result = self.copy();
        result.band_pass_filter_inplace_impl(low_hz, high_hz)?;
        Ok(result)
    }

    /// Remove DC offset (functional version).
    ///
    /// # Returns
    /// New AudioSamples object with DC offset removed
    ///
    /// # Examples
    /// ```python
    /// dc_free = audio.remove_dc_offset()
    /// ```
    pub(crate) fn remove_dc_offset_functional_impl(&self) -> PyResult<PyAudioSamples> {
        let mut result = self.copy();
        result.remove_dc_offset_inplace_impl()?;
        Ok(result)
    }

    /// Clip audio samples to the specified range (functional version).
    ///
    /// # Arguments
    /// * `min_val` - Minimum allowed value
    /// * `max_val` - Maximum allowed value
    ///
    /// # Returns
    /// New AudioSamples object with clipped data
    ///
    /// # Examples
    /// ```python
    /// clipped = audio.clip(min_val=-1.0, max_val=1.0)
    /// ```
    pub(crate) fn clip_functional_impl(
        &self,
        min_val: f64,
        max_val: f64,
    ) -> PyResult<PyAudioSamples> {
        let mut result = self.copy();
        result.clip_inplace_impl(min_val, max_val)?;
        Ok(result)
    }
}
