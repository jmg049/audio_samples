//! Python bindings for the audio_samples library.
//!
//! This module provides a comprehensive Python interface to the Rust audio processing
//! capabilities, designed with a pandas-like API that offers both in-place and functional
//! operations for maximum flexibility and performance.
//!
//! ## Key Design Features
//!
//! - **Dual Operation Modes**: Methods ending with `_` modify in-place, others return new objects
//! - **Rich Default Arguments**: Pythonic defaults for all optional parameters
//! - **Explicit Naming**: All PyO3 decorators use explicit `name` parameters
//! - **Zero-copy Where Possible**: Efficient memory sharing with NumPy arrays
//! - **Type Safety**: Rust's type system ensures safe operations across the Python boundary

use crate::{AudioSamples, I24, operations::AudioTypeConversion};
use numpy::PyArrayDescr;
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

/// Internal enum to store AudioSamples of different types while preserving
/// the original dtype. This allows the Python bindings to maintain memory
/// efficiency and performance benefits of different sample formats.
#[derive(Debug, Clone)]
pub(crate) enum AudioSamplesData {
    /// 16-bit signed integer samples (CD quality, memory efficient)
    I16(AudioSamples<i16>),
    /// 24-bit signed integer samples (professional audio)
    I24(AudioSamples<I24>),
    /// 32-bit signed integer samples (high precision integer)
    I32(AudioSamples<i32>),
    /// 32-bit floating-point samples (good precision/memory balance)
    F32(AudioSamples<f32>),
    /// 64-bit floating-point samples (maximum precision)
    F64(AudioSamples<f64>),
}

// Import submodules
mod conversions;
mod editing;
mod processing;
mod statistics;
mod transforms;
mod utils;

// Re-export utility functions
use utils::{convert_audio_samples_to_numpy, convert_numpy_to_audio_samples};

/// Python wrapper for AudioSamples with efficient numpy integration.
///
/// This class provides a pandas-like API for audio processing operations,
/// with clear distinction between in-place and functional methods.
///
/// # Operation Modes
/// - **In-place methods**: End with `_` (e.g., `normalize_()`, `scale_()`)
/// - **Functional methods**: Return new objects (e.g., `normalize()`, `scale()`)
///
/// # Examples
/// ```python
/// import audio_samples as aus
/// import numpy as np
///
/// # Create from numpy array
/// audio = aus.from_numpy(np.random.randn(44100), sample_rate=44100)
///
/// # In-place operations (modify original)
/// audio.normalize_(method='minmax')  # audio is modified
/// audio.scale_(0.5)                  # audio is modified
///
/// # Functional operations (return new objects)
/// normalized = audio.normalize(method='zscore')  # audio unchanged
/// scaled = audio.scale(2.0)                      # audio unchanged
/// ```
#[pyclass(name = "AudioSamples", module = "audio_samples")]
#[derive(Clone)]
pub struct PyAudioSamples {
    /// Internal Rust AudioSamples with preserved dtype for memory efficiency
    pub(crate) data: AudioSamplesData,
}

#[pymethods]
impl PyAudioSamples {
    /// Create a new AudioSamples object.
    ///
    /// # Arguments
    /// * `data` - NumPy array containing audio samples
    /// * `sample_rate` - Sample rate in Hz
    /// * `channels` - Number of channels (auto-detected from array shape if None)
    ///
    /// # Examples
    /// ```python
    /// # Mono audio
    /// audio = AudioSamples(np.random.randn(44100), sample_rate=44100)
    ///
    /// # Stereo audio (2 channels, 44100 samples each)
    /// stereo_data = np.random.randn(2, 44100)
    /// audio = AudioSamples(stereo_data, sample_rate=44100, channels=2)
    /// ```
    #[new]
    #[pyo3(signature = (data, sample_rate, *, channels=None))]
    fn new(data: &Bound<PyAny>, sample_rate: u32, channels: Option<usize>) -> PyResult<Self> {
        let data = convert_numpy_to_audio_samples(data, sample_rate, channels)?;
        Ok(PyAudioSamples { data })
    }

    /// Sample rate in Hz.
    #[getter]
    fn sample_rate(&self) -> u32 {
        match &self.data {
            AudioSamplesData::I16(inner) => inner.sample_rate(),
            AudioSamplesData::I24(inner) => inner.sample_rate(),
            AudioSamplesData::I32(inner) => inner.sample_rate(),
            AudioSamplesData::F32(inner) => inner.sample_rate(),
            AudioSamplesData::F64(inner) => inner.sample_rate(),
        }
    }

    /// Number of audio channels.
    #[getter]
    fn channels(&self) -> usize {
        match &self.data {
            AudioSamplesData::I16(inner) => inner.channels(),
            AudioSamplesData::I24(inner) => inner.channels(),
            AudioSamplesData::I32(inner) => inner.channels(),
            AudioSamplesData::F32(inner) => inner.channels(),
            AudioSamplesData::F64(inner) => inner.channels(),
        }
    }

    /// Number of samples per channel.
    #[getter]
    fn length(&self) -> usize {
        match &self.data {
            AudioSamplesData::I16(inner) => inner.samples_per_channel(),
            AudioSamplesData::I24(inner) => inner.samples_per_channel(),
            AudioSamplesData::I32(inner) => inner.samples_per_channel(),
            AudioSamplesData::F32(inner) => inner.samples_per_channel(),
            AudioSamplesData::F64(inner) => inner.samples_per_channel(),
        }
    }

    /// Duration in seconds.
    #[getter]
    fn duration(&self) -> f64 {
        match &self.data {
            AudioSamplesData::I16(inner) => inner.duration_seconds(),
            AudioSamplesData::I24(inner) => inner.duration_seconds(),
            AudioSamplesData::I32(inner) => inner.duration_seconds(),
            AudioSamplesData::F32(inner) => inner.duration_seconds(),
            AudioSamplesData::F64(inner) => inner.duration_seconds(),
        }
    }

    /// Total number of samples across all channels.
    #[getter]
    fn size(&self) -> usize {
        match &self.data {
            AudioSamplesData::I16(inner) => inner.total_samples(),
            AudioSamplesData::I24(inner) => inner.total_samples(),
            AudioSamplesData::I32(inner) => inner.total_samples(),
            AudioSamplesData::F32(inner) => inner.total_samples(),
            AudioSamplesData::F64(inner) => inner.total_samples(),
        }
    }

    /// Shape of the audio data as (channels, samples_per_channel).
    #[getter]
    fn shape(&self) -> (usize, usize) {
        match &self.data {
            AudioSamplesData::I16(inner) => (inner.channels(), inner.samples_per_channel()),
            AudioSamplesData::I24(inner) => (inner.channels(), inner.samples_per_channel()),
            AudioSamplesData::I32(inner) => (inner.channels(), inner.samples_per_channel()),
            AudioSamplesData::F32(inner) => (inner.channels(), inner.samples_per_channel()),
            AudioSamplesData::F64(inner) => (inner.channels(), inner.samples_per_channel()),
        }
    }

    /// Get the current data type of the audio samples.
    #[getter]
    fn dtype<'py>(&'py self, py: Python<'py>) -> PyResult<Bound<'py, PyArrayDescr>> {
        // Convert the internal data type to a NumPy dtype
        let dtype: Bound<'_, PyArrayDescr> = match &self.data {
            AudioSamplesData::I16(_) => numpy::PyArrayDescr::new(py, "int16"),
            AudioSamplesData::I24(_) => numpy::PyArrayDescr::new(py, "int24"),
            AudioSamplesData::I32(_) => numpy::PyArrayDescr::new(py, "int32"),
            AudioSamplesData::F32(_) => numpy::PyArrayDescr::new(py, "float32"),
            AudioSamplesData::F64(_) => numpy::PyArrayDescr::new(py, "float64"),
        }?;
        Ok(dtype)
    }

    fn sample_type<'py>(&'py self, py: Python<'py>) -> PyResult<Bound<'py, PyArrayDescr>> {
        // Convert the internal data type to a NumPy dtype
        let dtype: Bound<'_, PyArrayDescr> = match &self.data {
            AudioSamplesData::I16(_) => numpy::PyArrayDescr::new(py, "int16"),
            AudioSamplesData::I24(_) => numpy::PyArrayDescr::new(py, "int24"),
            AudioSamplesData::I32(_) => numpy::PyArrayDescr::new(py, "int32"),
            AudioSamplesData::F32(_) => numpy::PyArrayDescr::new(py, "float32"),
            AudioSamplesData::F64(_) => numpy::PyArrayDescr::new(py, "float64"),
        }?;
        Ok(dtype)
    }

    const fn bytes_per_sample(&self) -> usize {
        match &self.data {
            AudioSamplesData::I16(_) => std::mem::size_of::<i16>(),
            AudioSamplesData::I24(_) => std::mem::size_of::<I24>(),
            AudioSamplesData::I32(_) => std::mem::size_of::<i32>(),
            AudioSamplesData::F32(_) => std::mem::size_of::<f32>(),
            AudioSamplesData::F64(_) => std::mem::size_of::<f64>(),
        }
    }

    /// Convert to NumPy array.
    ///
    /// # Arguments
    /// * `dtype` - Target NumPy dtype (numpy.dtype or string like 'f32', 'f64', 'i16', 'i32')
    /// * `copy` - Force a copy even if memory layout allows sharing
    ///
    /// # Returns
    /// NumPy array with shape (channels, samples) for multi-channel or (samples,) for mono
    ///
    /// # Examples
    /// ```python
    /// # Get as float64 (default)
    /// numpy_array = audio.to_numpy()
    ///
    /// # Convert using numpy dtypes (preferred)
    /// numpy_f32 = audio.to_numpy(dtype=np.float32)
    /// numpy_i16 = audio.to_numpy(dtype=np.int16)
    ///
    /// # Convert using string dtypes (convenience)
    /// numpy_f32 = audio.to_numpy(dtype='f32')
    ///
    /// # Force a copy
    /// numpy_copy = audio.to_numpy(copy=True)
    /// ```
    #[pyo3(signature = (*, dtype=None, copy=false))]
    fn to_numpy(&self, py: Python, dtype: Option<&Bound<PyAny>>, copy: bool) -> PyResult<PyObject> {
        let dtype_str = if let Some(dt) = dtype {
            // Try to extract numpy dtype first, fall back to string
            if let Ok(numpy_dtype) = dt.downcast::<numpy::PyArrayDescr>() {
                Some(crate::python::conversions::get_target_type(numpy_dtype)?)
            } else if let Ok(dtype_str) = dt.extract::<String>() {
                crate::python::utils::validate_string_param(
                    "dtype",
                    &dtype_str,
                    &["f64", "f32", "i32", "i16"],
                )?;
                Some(match dtype_str.as_str() {
                    "f64" => crate::python::conversions::TargetType::F64,
                    "f32" => crate::python::conversions::TargetType::F32,
                    "i32" => crate::python::conversions::TargetType::I32,
                    "i16" => crate::python::conversions::TargetType::I16,
                    _ => unreachable!(),
                })
            } else {
                return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                    "dtype must be a numpy dtype or string ('f64', 'f32', 'i32', 'i16')",
                ));
            }
        } else {
            None
        };

        convert_audio_samples_to_numpy(py, &self.data, dtype_str, copy)
    }

    /// String representation showing key properties.
    fn __repr__(&self, py: Python<'_>) -> String {
        format!(
            "AudioSamples(channels={}, length={}, sample_rate={}, duration={:.3}s, dtype={})",
            self.channels(),
            self.length(),
            self.sample_rate(),
            self.duration(),
            self.dtype(py)
                .unwrap_or(PyArrayDescr::new(py, "object").unwrap())
        )
    }

    /// Brief string representation.
    fn __str__(&self, py: Python<'_>) -> String {
        format!(
            "AudioSamples({} ch, {:.3}s @ {}Hz, {})",
            self.channels(),
            self.duration(),
            self.sample_rate(),
            self.dtype(py)
                .unwrap_or(PyArrayDescr::new(py, "object").unwrap())
        )
    }

    /// Length for Python len() function.
    fn __len__(&self) -> usize {
        self.length()
    }

    /// Create a copy of the AudioSamples object.
    #[pyo3(name = "copy")]
    fn copy(&self) -> Self {
        PyAudioSamples {
            data: self.data.clone(),
        }
    }

    // Audio Statistics Methods
    /// Return the peak (maximum absolute value) in the audio samples.
    #[pyo3(name = "peak")]
    fn peak(&self) -> f64 {
        self.peak_impl()
    }

    /// Return the minimum value in the audio samples.
    #[pyo3(name = "min")]
    fn min(&self) -> f64 {
        self.min_impl()
    }

    /// Return the maximum value in the audio samples.
    #[pyo3(name = "max")]
    fn max(&self) -> f64 {
        self.max_impl()
    }

    /// Compute the Root Mean Square (RMS) of the audio samples.
    #[pyo3(name = "rms")]
    fn rms(&self) -> f64 {
        self.rms_impl()
    }

    // Audio Type Conversion Methods
    /// Convert to different sample type with explicit dtype specification.
    #[pyo3(signature = (dtype, *, copy=true))]
    fn astype(&self, py: Python, dtype: &Bound<PyAny>, copy: bool) -> PyResult<PyObject> {
        self.astype_impl(py, dtype, copy)
    }

    /// Convert to the highest precision floating-point format (f64).
    #[pyo3(name = "to_f64")]
    fn to_f64(&self) -> PyResult<PyAudioSamples> {
        self.to_f64_impl()
    }

    /// Convert to single precision floating-point format (f32).
    #[pyo3(name = "to_f32")]
    fn to_f32(&self) -> PyResult<PyAudioSamples> {
        self.to_f32_impl()
    }

    /// Convert to 32-bit integer format.
    #[pyo3(name = "to_i32")]
    fn to_i32(&self) -> PyResult<PyAudioSamples> {
        self.to_i32_impl()
    }

    /// Convert to 16-bit integer format (CD quality).
    #[pyo3(name = "to_i16")]
    fn to_i16(&self) -> PyResult<PyAudioSamples> {
        self.to_i16_impl()
    }

    // Additional Statistics Methods
    /// Compute the statistical variance of the audio samples.
    #[pyo3(name = "variance")]
    fn variance(&self) -> f64 {
        self.variance_impl()
    }

    /// Compute the standard deviation of the audio samples.
    #[pyo3(name = "std_dev")]
    fn std_dev(&self) -> f64 {
        self.std_dev_impl()
    }

    /// Count the number of zero crossings in the audio signal.
    #[pyo3(name = "zero_crossings")]
    fn zero_crossings(&self) -> usize {
        self.zero_crossings_impl()
    }

    /// Compute the zero crossing rate (crossings per second).
    #[pyo3(name = "zero_crossing_rate")]
    fn zero_crossing_rate(&self) -> f64 {
        self.zero_crossing_rate_impl()
    }

    /// Compute the autocorrelation function up to max_lag samples.
    #[pyo3(signature = (*, max_lag=None, normalize=true))]
    fn autocorrelation(
        &self,
        py: Python,
        max_lag: Option<usize>,
        normalize: bool,
    ) -> PyResult<PyObject> {
        self.autocorrelation_impl(py, max_lag, normalize)
    }

    /// Compute cross-correlation with another audio signal.
    #[pyo3(signature = (other, *, max_lag=None, normalize=true))]
    fn cross_correlation(
        &self,
        py: Python,
        other: &PyAudioSamples,
        max_lag: Option<usize>,
        normalize: bool,
    ) -> PyResult<PyObject> {
        self.cross_correlation_impl(py, other, max_lag, normalize)
    }

    /// Compute the spectral centroid (brightness measure).
    #[pyo3(name = "spectral_centroid")]
    fn spectral_centroid(&self) -> PyResult<f64> {
        self.spectral_centroid_impl()
    }

    /// Compute spectral rolloff frequency.
    #[pyo3(signature = (*, rolloff_percent=0.85))]
    fn spectral_rolloff(&self, rolloff_percent: f64) -> PyResult<f64> {
        self.spectral_rolloff_impl(rolloff_percent)
    }

    // Additional Conversion Methods
    /// Check if the audio data can be represented without loss in the target type.
    #[pyo3(signature = (target_dtype, *, tolerance=1e-10))]
    fn can_convert_lossless(&self, target_dtype: &str, tolerance: f64) -> PyResult<bool> {
        self.can_convert_lossless_impl(target_dtype, tolerance)
    }

    /// Get information about precision loss for a given conversion.
    #[pyo3(name = "conversion_info")]
    fn conversion_info(&self, py: Python, target_dtype: &str) -> PyResult<PyObject> {
        self.conversion_info_impl(py, target_dtype)
    }

    /// Create a view of the audio data as a different type without conversion.
    #[pyo3(name = "view_as")]
    fn view_as(&self, target_dtype: &str) -> PyResult<PyAudioSamples> {
        self.view_as_impl(target_dtype)
    }

    /// Test method to verify methods can be added.
    #[pyo3(name = "test_method")]
    fn test_method(&self) -> f64 {
        42.0
    }

    // Audio Editing Methods
    /// Reverse the order of audio samples.
    #[pyo3(name = "reverse")]
    fn reverse(&self) -> PyResult<PyAudioSamples> {
        self.reverse_impl()
    }
    /// Reverse the order of audio samples in-place.
    #[pyo3(name = "reverse_")]
    fn reverse_in_place(&mut self) -> PyResult<()> {
        self.reverse_inplace_impl()
    }

    /// Extract a segment of audio between start and end times.
    #[pyo3(signature = (start_sec, end_sec, *, copy=true))]
    fn trim(&self, start_sec: f64, end_sec: f64, copy: bool) -> PyResult<PyAudioSamples> {
        self.trim_impl(start_sec, end_sec, copy)
    }

    /// Add padding/silence to the beginning and/or end of the audio.
    #[pyo3(signature = (start_sec, end_sec, *, value=0.0))]
    fn pad(&self, start_sec: f64, end_sec: f64, value: f64) -> PyResult<PyAudioSamples> {
        self.pad_impl(start_sec, end_sec, value)
    }

    /// Split audio into segments of specified duration.
    #[pyo3(signature = (segment_duration, *, overlap=0.0))]
    fn split(&self, segment_duration: f64, overlap: f64) -> PyResult<Vec<PyAudioSamples>> {
        self.split_impl(segment_duration, overlap)
    }

    /// Repeat the audio signal a specified number of times.
    #[pyo3(name = "repeat")]
    fn repeat(&self, count: usize) -> PyResult<PyAudioSamples> {
        self.repeat_impl(count)
    }

    /// Remove silence from the beginning and end of the audio.
    #[pyo3(signature = (threshold, *, min_duration=0.0))]
    fn trim_silence(&self, threshold: f64, min_duration: f64) -> PyResult<PyAudioSamples> {
        self.trim_silence_impl(threshold, min_duration)
    }

    // Fade Methods - Dual Mode (in-place and functional)
    /// Apply fade-in envelope in-place.
    #[pyo3(signature = (duration, *, curve="linear"))]
    fn fade_in_(&mut self, duration: f64, curve: &str) -> PyResult<()> {
        self.fade_in_inplace_impl(duration, curve)
    }

    /// Apply fade-in envelope (functional version).
    #[pyo3(signature = (duration, *, curve="linear"))]
    fn fade_in(&self, duration: f64, curve: &str) -> PyResult<PyAudioSamples> {
        self.fade_in_impl(duration, curve)
    }

    /// Apply fade-out envelope in-place.
    #[pyo3(signature = (duration, *, curve="linear"))]
    fn fade_out_(&mut self, duration: f64, curve: &str) -> PyResult<()> {
        self.fade_out_inplace_impl(duration, curve)
    }

    /// Apply fade-out envelope (functional version).
    #[pyo3(signature = (duration, *, curve="linear"))]
    fn fade_out(&self, duration: f64, curve: &str) -> PyResult<PyAudioSamples> {
        self.fade_out_impl(duration, curve)
    }

    // Static Methods for Audio Editing
    /// Concatenate multiple audio segments into one.
    #[staticmethod]
    #[pyo3(signature = (segments, *, axis=0))]
    fn concatenate(segments: &Bound<PyAny>, axis: i32) -> PyResult<PyAudioSamples> {
        // Extract list of PyAudioSamples from Python list
        let py_list = segments.downcast::<pyo3::types::PyList>()?;
        let mut segment_copies = Vec::new();
        for item in py_list.iter() {
            let audio_sample: PyRef<PyAudioSamples> = item.extract()?;
            segment_copies.push((*audio_sample).clone());
        }
        let segment_refs: Vec<&PyAudioSamples> = segment_copies.iter().collect();
        PyAudioSamples::concatenate_impl(segment_refs, axis)
    }

    /// Mix multiple audio sources together.
    #[staticmethod]
    #[pyo3(signature = (sources, *, weights=None, normalize=false))]
    fn mix(
        sources: &Bound<PyAny>,
        weights: Option<Vec<f64>>,
        normalize: bool,
    ) -> PyResult<PyAudioSamples> {
        // Extract list of PyAudioSamples from Python list
        let py_list = sources.downcast::<pyo3::types::PyList>()?;
        let mut source_copies = Vec::new();
        for item in py_list.iter() {
            let audio_sample: PyRef<PyAudioSamples> = item.extract()?;
            source_copies.push((*audio_sample).clone());
        }
        let source_refs: Vec<&PyAudioSamples> = source_copies.iter().collect();
        PyAudioSamples::mix_impl(source_refs, weights, normalize)
    }

    /// Cross-fade between two audio sources.
    #[staticmethod]
    #[pyo3(signature = (audio1, audio2, duration, *, curve="linear", offset=0.0))]
    fn crossfade(
        audio1: &PyAudioSamples,
        audio2: &PyAudioSamples,
        duration: f64,
        curve: &str,
        offset: f64,
    ) -> PyResult<PyAudioSamples> {
        PyAudioSamples::crossfade_impl(audio1, audio2, duration, curve, offset)
    }

    // Audio Processing Methods - In-place operations (modify self)
    /// Normalize audio samples in-place using the specified method.
    #[pyo3(signature = (method, *, min_val=-1.0, max_val=1.0), name = "normalize_")]
    fn normalize_(&mut self, method: &str, min_val: f64, max_val: f64) -> PyResult<()> {
        self.normalize_inplace_impl(method, min_val, max_val)
    }

    /// Scale all audio samples by a constant factor in-place.
    #[pyo3(name = "scale_")]
    fn scale_(&mut self, factor: f64) -> PyResult<()> {
        self.scale_inplace_impl(factor)
    }

    /// Apply a windowing function to the audio samples in-place.
    #[pyo3(name = "apply_window_")]
    fn apply_window_(&mut self, window: &Bound<PyAny>) -> PyResult<()> {
        self.apply_window_inplace_impl(window)
    }

    /// Apply a digital filter to the audio samples in-place.
    #[pyo3(signature = (coeffs, *, mode="same"), name = "apply_filter_")]
    fn apply_filter_(&mut self, coeffs: &Bound<PyAny>, mode: &str) -> PyResult<()> {
        self.apply_filter_inplace_impl(coeffs, mode)
    }

    /// Apply μ-law compression to the audio samples in-place.
    #[pyo3(signature = (*, mu=255.0), name = "mu_compress_")]
    fn mu_compress_(&mut self, mu: f64) -> PyResult<()> {
        self.mu_compress_inplace_impl(mu)
    }

    /// Apply μ-law expansion (decompression) to the audio samples in-place.
    #[pyo3(signature = (*, mu=255.0), name = "mu_expand_")]
    fn mu_expand_(&mut self, mu: f64) -> PyResult<()> {
        self.mu_expand_inplace_impl(mu)
    }

    /// Apply a low-pass filter in-place.
    #[pyo3(name = "low_pass_filter_")]
    fn low_pass_filter_(&mut self, cutoff_hz: f64) -> PyResult<()> {
        self.low_pass_filter_inplace_impl(cutoff_hz)
    }

    /// Apply a high-pass filter in-place.
    #[pyo3(name = "high_pass_filter_")]
    fn high_pass_filter_(&mut self, cutoff_hz: f64) -> PyResult<()> {
        self.high_pass_filter_inplace_impl(cutoff_hz)
    }

    /// Apply a band-pass filter in-place.
    #[pyo3(name = "band_pass_filter_")]
    fn band_pass_filter_(&mut self, low_hz: f64, high_hz: f64) -> PyResult<()> {
        self.band_pass_filter_inplace_impl(low_hz, high_hz)
    }

    /// Remove DC offset by subtracting the mean value in-place.
    #[pyo3(name = "remove_dc_offset_")]
    fn remove_dc_offset_(&mut self) -> PyResult<()> {
        self.remove_dc_offset_inplace_impl()
    }

    /// Clip audio samples to the specified range in-place.
    #[pyo3(name = "clip_")]
    fn clip_(&mut self, min_val: f64, max_val: f64) -> PyResult<()> {
        self.clip_inplace_impl(min_val, max_val)
    }

    // Audio Processing Methods - Functional operations (return new objects)
    /// Normalize audio samples using the specified method (functional version).
    #[pyo3(signature = (method, *, min_val=-1.0, max_val=1.0), name = "normalize")]
    fn normalize(&self, method: &str, min_val: f64, max_val: f64) -> PyResult<PyAudioSamples> {
        self.normalize_impl(method, min_val, max_val)
    }

    /// Scale all audio samples by a constant factor (functional version).
    #[pyo3(name = "scale")]
    fn scale(&self, factor: f64) -> PyResult<PyAudioSamples> {
        self.scale_functional_impl(factor)
    }

    /// Apply a windowing function (functional version).
    #[pyo3(name = "apply_window")]
    fn apply_window(&self, window: &Bound<PyAny>) -> PyResult<PyAudioSamples> {
        self.apply_window_functional_impl(window)
    }

    /// Apply a digital filter (functional version).
    #[pyo3(signature = (coeffs, *, mode="same"), name = "apply_filter")]
    fn apply_filter(&self, coeffs: &Bound<PyAny>, mode: &str) -> PyResult<PyAudioSamples> {
        self.apply_filter_impl(coeffs, mode)
    }

    /// Apply μ-law compression (functional version).
    #[pyo3(signature = (*, mu=255.0), name = "mu_compress")]
    fn mu_compress(&self, mu: f64) -> PyResult<PyAudioSamples> {
        self.mu_compress_impl(mu)
    }

    /// Apply μ-law expansion (functional version).
    #[pyo3(signature = (*, mu=255.0), name = "mu_expand")]
    fn mu_expand(&self, mu: f64) -> PyResult<PyAudioSamples> {
        self.mu_expand_impl(mu)
    }

    /// Apply a low-pass filter (functional version).
    #[pyo3(name = "low_pass_filter")]
    fn low_pass_filter(&self, cutoff_hz: f64) -> PyResult<PyAudioSamples> {
        self.low_pass_filter_functional_impl(cutoff_hz)
    }

    /// Apply a high-pass filter (functional version).
    #[pyo3(name = "high_pass_filter")]
    fn high_pass_filter(&self, cutoff_hz: f64) -> PyResult<PyAudioSamples> {
        self.high_pass_filter_functional_impl(cutoff_hz)
    }

    /// Apply a band-pass filter (functional version).
    #[pyo3(name = "band_pass_filter")]
    fn band_pass_filter(&self, low_hz: f64, high_hz: f64) -> PyResult<PyAudioSamples> {
        self.band_pass_filter_functional_impl(low_hz, high_hz)
    }

    /// Remove DC offset (functional version).
    #[pyo3(name = "remove_dc_offset")]
    fn remove_dc_offset(&self) -> PyResult<PyAudioSamples> {
        self.remove_dc_offset_functional_impl()
    }

    /// Clip audio samples to the specified range (functional version).
    #[pyo3(name = "clip")]
    fn clip(&self, min_val: f64, max_val: f64) -> PyResult<PyAudioSamples> {
        self.clip_functional_impl(min_val, max_val)
    }

    // Audio Transform Methods - Frequency domain analysis and spectral transformations
    /// Compute the Fast Fourier Transform of the audio samples.
    #[pyo3(signature = (*, n=None, axis=0), name = "fft")]
    fn fft(&self, py: Python, n: Option<usize>, axis: i32) -> PyResult<PyObject> {
        self.fft_impl(py, n, axis)
    }

    /// Compute the inverse Fast Fourier Transform from frequency domain.
    #[pyo3(name = "ifft")]
    fn ifft(&self, spectrum: &Bound<PyAny>) -> PyResult<PyAudioSamples> {
        self.ifft_impl(spectrum)
    }

    /// Compute the Short-Time Fourier Transform (STFT).
    #[pyo3(signature = (*, window_size=1024, hop_size=None, window="hanning"), name = "stft")]
    fn stft(
        &self,
        py: Python,
        window_size: usize,
        hop_size: Option<usize>,
        window: &str,
    ) -> PyResult<PyObject> {
        self.stft_impl(py, window_size, hop_size, window)
    }

    /// Compute the inverse Short-Time Fourier Transform (ISTFT).
    #[pyo3(signature = (stft_data, *, hop_size=512, window="hanning"), name = "istft")]
    fn istft(
        &self,
        stft_data: &Bound<PyAny>,
        hop_size: usize,
        window: &str,
    ) -> PyResult<PyAudioSamples> {
        self.istft_impl(stft_data, hop_size, window)
    }

    /// Compute a spectrogram of the audio signal.
    #[pyo3(signature = (*, window_size=1024, hop_size=None, window="hanning", scale="linear", power=2.0), name = "spectrogram")]
    fn spectrogram(
        &self,
        py: Python,
        window_size: usize,
        hop_size: Option<usize>,
        window: &str,
        scale: &str,
        power: f64,
    ) -> PyResult<PyObject> {
        self.spectrogram_impl(py, window_size, hop_size, window, scale, power)
    }

    /// Compute a mel-scale spectrogram.
    #[pyo3(signature = (*, n_mels=128, fmin=0.0, fmax=None, window_size=None, hop_size=None), name = "mel_spectrogram")]
    fn mel_spectrogram(
        &self,
        py: Python,
        n_mels: Option<usize>,
        fmin: Option<f64>,
        fmax: Option<f64>,
        window_size: Option<usize>,
        hop_size: Option<usize>,
    ) -> PyResult<PyObject> {
        self.mel_spectrogram_impl(py, n_mels, fmin, fmax, window_size, hop_size)
    }

    /// Compute Mel-Frequency Cepstral Coefficients (MFCCs).
    #[pyo3(signature = (*, n_mfcc=13, n_mels=128, fmin=0.0, fmax=None, window_size=None, hop_size=None), name = "mfcc")]
    fn mfcc(
        &self,
        py: Python,
        n_mfcc: Option<usize>,
        n_mels: Option<usize>,
        fmin: Option<f64>,
        fmax: Option<f64>,
        window_size: Option<usize>,
        hop_size: Option<usize>,
    ) -> PyResult<PyObject> {
        self.mfcc_impl(py, n_mfcc, n_mels, fmin, fmax, window_size, hop_size)
    }

    /// Compute a gammatone filter bank spectrogram.
    #[pyo3(signature = (*, n_filters=64, fmin=50.0, fmax=None, window_size=None, hop_size=None), name = "gammatone_spectrogram")]
    fn gammatone_spectrogram(
        &self,
        py: Python,
        n_filters: Option<usize>,
        fmin: Option<f64>,
        fmax: Option<f64>,
        window_size: Option<usize>,
        hop_size: Option<usize>,
    ) -> PyResult<PyObject> {
        self.gammatone_spectrogram_impl(py, n_filters, fmin, fmax, window_size, hop_size)
    }

    /// Compute a chromagram (pitch class profile).
    #[pyo3(signature = (*, n_chroma=12, window_size=1024, hop_size=None), name = "chroma")]
    fn chroma(
        &self,
        py: Python,
        n_chroma: Option<usize>,
        window_size: Option<usize>,
        hop_size: Option<usize>,
    ) -> PyResult<PyObject> {
        self.chroma_impl(py, n_chroma, window_size, hop_size)
    }

    /// Compute the power spectral density using Welch's method.
    #[pyo3(signature = (*, window_size=1024, overlap=0.5, window="hanning"), name = "power_spectral_density")]
    fn power_spectral_density(
        &self,
        py: Python,
        window_size: Option<usize>,
        overlap: Option<f64>,
        window: Option<&str>,
    ) -> PyResult<PyObject> {
        self.power_spectral_density_impl(py, window_size, overlap, window)
    }
}

/// Factory Functions
/// ================

/// Create AudioSamples from a NumPy array.
///
/// # Arguments
/// * `array` - NumPy array containing audio data
/// * `sample_rate` - Sample rate in Hz
/// * `copy` - Whether to copy the data (default: True for safety)
///
/// # Examples
/// ```python
/// import numpy as np
/// import audio_samples as aus
///
/// # From 1D array (mono)
/// mono_data = np.random.randn(44100)
/// audio = aus.from_numpy(mono_data, sample_rate=44100)
///
/// # From 2D array (multi-channel)
/// stereo_data = np.random.randn(2, 44100)  # 2 channels, 44100 samples
/// audio = aus.from_numpy(stereo_data, sample_rate=44100)
/// ```
#[pyfunction(name = "from_numpy")]
#[pyo3(signature = (array, sample_rate, *, copy=true))]
fn from_numpy(array: &Bound<PyAny>, sample_rate: u32, copy: bool) -> PyResult<PyAudioSamples> {
    let data = convert_numpy_to_audio_samples(array, sample_rate, None)?;
    Ok(PyAudioSamples { data })
}

/// Create AudioSamples filled with zeros.
///
/// # Arguments
/// * `length` - Number of samples per channel
/// * `sample_rate` - Sample rate in Hz
/// * `channels` - Number of channels (default: 1)
/// * `dtype` - Data type (numpy.dtype or string like 'f32', 'f64', 'i16', 'i32', default: 'f64')
///
/// # Examples
/// ```python
/// import numpy as np
/// import audio_samples as aus
///
/// # Mono silence (default f64)
/// silence = aus.zeros(44100, sample_rate=44100)
///
/// # Stereo silence with explicit numpy dtype
/// stereo_silence = aus.zeros(44100, sample_rate=44100, channels=2, dtype=np.float32)
///
/// # Using string dtype for convenience
/// int16_silence = aus.zeros(44100, sample_rate=44100, dtype='i16')
/// ```
#[pyfunction(name = "zeros")]
#[pyo3(signature = (length, sample_rate, *, channels=1, dtype=None))]
fn zeros(
    _py: Python,
    length: usize,
    sample_rate: u32,
    channels: usize,
    dtype: Option<&Bound<PyAny>>,
) -> PyResult<PyAudioSamples> {
    // Parse dtype parameter (numpy dtype or string, default to 'f64')
    let target_type = if let Some(dt) = dtype {
        if let Ok(numpy_dtype) = dt.downcast::<numpy::PyArrayDescr>() {
            crate::python::conversions::get_target_type(numpy_dtype)?
        } else if let Ok(dtype_str) = dt.extract::<String>() {
            utils::validate_string_param("dtype", &dtype_str, &["f64", "f32", "i32", "i16"])?;
            match dtype_str.as_str() {
                "f64" => crate::python::conversions::TargetType::F64,
                "f32" => crate::python::conversions::TargetType::F32,
                "i32" => crate::python::conversions::TargetType::I32,
                "i16" => crate::python::conversions::TargetType::I16,
                _ => unreachable!(),
            }
        } else {
            return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                "dtype must be a numpy dtype or string ('f64', 'f32', 'i32', 'i16')",
            ));
        }
    } else {
        crate::python::conversions::TargetType::F64
    };

    // Create zeros array in the target type
    let data = match target_type {
        crate::python::conversions::TargetType::F64 => {
            let inner = if channels == 1 {
                AudioSamples::<f64>::zeros_mono(length, sample_rate)
            } else {
                AudioSamples::<f64>::zeros_multi(channels, length, sample_rate)
            };
            crate::python::AudioSamplesData::F64(inner)
        }
        crate::python::conversions::TargetType::F32 => {
            let inner_f64 = if channels == 1 {
                AudioSamples::<f64>::zeros_mono(length, sample_rate)
            } else {
                AudioSamples::<f64>::zeros_multi(channels, length, sample_rate)
            };
            let inner_f32 = inner_f64.to_f32().map_err(utils::map_error)?;
            crate::python::AudioSamplesData::F32(inner_f32)
        }
        crate::python::conversions::TargetType::I32 => {
            let inner_f64 = if channels == 1 {
                AudioSamples::<f64>::zeros_mono(length, sample_rate)
            } else {
                AudioSamples::<f64>::zeros_multi(channels, length, sample_rate)
            };
            let inner_i32 = inner_f64.to_i32().map_err(utils::map_error)?;
            crate::python::AudioSamplesData::I32(inner_i32)
        }
        crate::python::conversions::TargetType::I16 => {
            let inner_f64 = if channels == 1 {
                AudioSamples::<f64>::zeros_mono(length, sample_rate)
            } else {
                AudioSamples::<f64>::zeros_multi(channels, length, sample_rate)
            };
            let inner_i16 = inner_f64.to_i16().map_err(utils::map_error)?;
            crate::python::AudioSamplesData::I16(inner_i16)
        }
    };

    Ok(PyAudioSamples { data })
}

/// Create AudioSamples filled with ones.
///
/// # Arguments
/// * `length` - Number of samples per channel
/// * `sample_rate` - Sample rate in Hz
/// * `channels` - Number of channels (default: 1)
/// * `dtype` - Data type (numpy.dtype or string like 'f32', 'f64', 'i16', 'i32', default: 'f64')
///
/// # Examples
/// ```python
/// import numpy as np
/// import audio_samples as aus
///
/// # Mono ones (default f64)
/// ones_audio = aus.ones(1000, sample_rate=44100)
///
/// # Stereo ones with explicit numpy dtype
/// stereo_ones = aus.ones(1000, sample_rate=44100, channels=2, dtype=np.float32)
///
/// # Using string dtype for convenience
/// int16_ones = aus.ones(1000, sample_rate=44100, dtype='i16')
/// ```
#[pyfunction(name = "ones")]
#[pyo3(signature = (length, sample_rate, *, channels=1, dtype=None))]
fn ones(
    _py: Python,
    length: usize,
    sample_rate: u32,
    channels: usize,
    dtype: Option<&Bound<PyAny>>,
) -> PyResult<PyAudioSamples> {
    // Parse dtype parameter (numpy dtype or string, default to 'f64')
    let target_type = if let Some(dt) = dtype {
        if let Ok(numpy_dtype) = dt.downcast::<numpy::PyArrayDescr>() {
            crate::python::conversions::get_target_type(numpy_dtype)?
        } else if let Ok(dtype_str) = dt.extract::<String>() {
            utils::validate_string_param("dtype", &dtype_str, &["f64", "f32", "i32", "i16"])?;
            match dtype_str.as_str() {
                "f64" => crate::python::conversions::TargetType::F64,
                "f32" => crate::python::conversions::TargetType::F32,
                "i32" => crate::python::conversions::TargetType::I32,
                "i16" => crate::python::conversions::TargetType::I16,
                _ => unreachable!(),
            }
        } else {
            return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                "dtype must be a numpy dtype or string ('f64', 'f32', 'i32', 'i16')",
            ));
        }
    } else {
        crate::python::conversions::TargetType::F64
    };

    // Create ones array in the target type
    let data = match target_type {
        crate::python::conversions::TargetType::F64 => {
            let mut inner = if channels == 1 {
                AudioSamples::<f64>::zeros_mono(length, sample_rate)
            } else {
                AudioSamples::<f64>::zeros_multi(channels, length, sample_rate)
            };
            // Fill with ones
            match inner.channels() {
                1 => {
                    if let Some(mono_data) = inner.as_mono_mut() {
                        mono_data.fill(1.0);
                    }
                }
                _ => {
                    if let Some(multi_data) = inner.as_multi_channel_mut() {
                        multi_data.fill(1.0);
                    }
                }
            }
            crate::python::AudioSamplesData::F64(inner)
        }
        crate::python::conversions::TargetType::F32 => {
            let mut inner_f64 = if channels == 1 {
                AudioSamples::<f64>::zeros_mono(length, sample_rate)
            } else {
                AudioSamples::<f64>::zeros_multi(channels, length, sample_rate)
            };
            // Fill with ones
            match inner_f64.channels() {
                1 => {
                    if let Some(mono_data) = inner_f64.as_mono_mut() {
                        mono_data.fill(1.0);
                    }
                }
                _ => {
                    if let Some(multi_data) = inner_f64.as_multi_channel_mut() {
                        multi_data.fill(1.0);
                    }
                }
            }
            let inner_f32 = inner_f64.to_f32().map_err(utils::map_error)?;
            crate::python::AudioSamplesData::F32(inner_f32)
        }
        crate::python::conversions::TargetType::I32 => {
            let mut inner_f64 = if channels == 1 {
                AudioSamples::<f64>::zeros_mono(length, sample_rate)
            } else {
                AudioSamples::<f64>::zeros_multi(channels, length, sample_rate)
            };
            // Fill with ones
            match inner_f64.channels() {
                1 => {
                    if let Some(mono_data) = inner_f64.as_mono_mut() {
                        mono_data.fill(1.0);
                    }
                }
                _ => {
                    if let Some(multi_data) = inner_f64.as_multi_channel_mut() {
                        multi_data.fill(1.0);
                    }
                }
            }
            let inner_i32 = inner_f64.to_i32().map_err(utils::map_error)?;
            crate::python::AudioSamplesData::I32(inner_i32)
        }
        crate::python::conversions::TargetType::I16 => {
            let mut inner_f64 = if channels == 1 {
                AudioSamples::<f64>::zeros_mono(length, sample_rate)
            } else {
                AudioSamples::<f64>::zeros_multi(channels, length, sample_rate)
            };
            // Fill with ones
            match inner_f64.channels() {
                1 => {
                    if let Some(mono_data) = inner_f64.as_mono_mut() {
                        mono_data.fill(1.0);
                    }
                }
                _ => {
                    if let Some(multi_data) = inner_f64.as_multi_channel_mut() {
                        multi_data.fill(1.0);
                    }
                }
            }
            let inner_i16 = inner_f64.to_i16().map_err(utils::map_error)?;
            crate::python::AudioSamplesData::I16(inner_i16)
        }
    };

    Ok(PyAudioSamples { data })
}

/// Module Registration
/// ==================

/// Register the audio_samples Python module.
///
/// This function is called from the main lib.rs to set up the Python module
/// with all classes, functions, and constants.
pub fn register_module(_py: Python, m: &Bound<PyModule>) -> PyResult<()> {
    // Main class
    m.add_class::<PyAudioSamples>()?;

    // Factory functions
    m.add_function(wrap_pyfunction!(from_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(zeros, m)?)?;
    m.add_function(wrap_pyfunction!(ones, m)?)?;

    // Module constants
    m.add("__version__", "0.1.0")?;
    m.add("SUPPORTED_DTYPES", vec!["i16", "I24", "i32", "f32", "f64"])?;

    // Documentation
    m.add(
        "__doc__",
        "High-performance audio processing library with pandas-like API",
    )?;

    Ok(())
}

/// Implementation of trait methods via delegation to separate modules
impl PyAudioSamples {
    /// Get a reference to the inner AudioSamplesData for use in submodules
    pub(crate) fn data(&self) -> &crate::python::AudioSamplesData {
        &self.data
    }

    /// Create a new PyAudioSamples from AudioSamplesData
    pub(crate) fn from_data(data: crate::python::AudioSamplesData) -> Self {
        Self { data }
    }

    // Helper methods for backward compatibility in submodules
    /// Create a new PyAudioSamples from an AudioSamples<f64> (for compatibility)
    pub(crate) fn from_inner(inner: AudioSamples<f64>) -> Self {
        Self {
            data: crate::python::AudioSamplesData::F64(inner),
        }
    }

    /// Helper method: convert to f64 AudioSamples for operations that need it
    pub(crate) fn as_f64(&self) -> Result<AudioSamples<f64>, crate::AudioSampleError> {
        use crate::operations::AudioTypeConversion;
        match &self.data {
            AudioSamplesData::F64(audio) => Ok(audio.clone()),
            AudioSamplesData::F32(audio) => audio.to_f64(),
            AudioSamplesData::I32(audio) => audio.to_f64(),
            AudioSamplesData::I16(audio) => audio.to_f64(),
            AudioSamplesData::I24(audio) => audio.to_f64(),
        }
    }

    /// Helper method: apply operation to inner data and wrap result
    pub(crate) fn with_inner<F, T>(&self, f: F) -> Result<T, crate::AudioSampleError>
    where
        F: Fn(&AudioSamples<f64>) -> Result<T, crate::AudioSampleError>,
    {
        let f64_audio = self.as_f64()?;
        f(&f64_audio)
    }

    /// Helper method: apply operation and return new PyAudioSamples
    pub(crate) fn map_inner<F>(&self, f: F) -> Result<PyAudioSamples, crate::AudioSampleError>
    where
        F: Fn(&AudioSamples<f64>) -> Result<AudioSamples<f64>, crate::AudioSampleError>,
    {
        let f64_audio = self.as_f64()?;
        let result = f(&f64_audio)?;
        Ok(PyAudioSamples::from_inner(result))
    }

    /// Helper method: apply mutable operation in-place (converts to f64 if needed)
    pub(crate) fn mutate_inner<F>(&mut self, f: F) -> Result<(), crate::AudioSampleError>
    where
        F: Fn(&mut AudioSamples<f64>) -> Result<(), crate::AudioSampleError>,
    {
        // Convert to f64 for operation
        let mut f64_audio = self.as_f64()?;
        f(&mut f64_audio)?;
        self.data = AudioSamplesData::F64(f64_audio);
        Ok(())
    }

    // Statistics and conversion method implementations are defined in their respective submodules
}
