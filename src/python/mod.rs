//! Python bindings for the audio_samples library.
//!
//! This module provides a comprehensive Python interface to the Rust audio processing
//! capabilities, designed with a pandas-like API that offers both in-place and functional
//! operations for maximum flexibility and performance.
//!
//! # Architecture Overview
//!
//! The Python bindings wrap the core Rust [`AudioSamples`] struct in a Python-friendly
//! [`PyAudioSamples`] class while preserving the high-performance, type-safe processing
//! capabilities of the underlying Rust implementation.
//!
//! ## Core Components
//!
//! - **[`PyAudioSamples`]** - Main Python class wrapping audio data and operations
//! - **[`AudioSamplesData`]** - Internal enum preserving original sample types (i16, f32, f64, etc.)
//! - **Factory Functions** - [`from_numpy()`], [`zeros()`], [`ones()`] for convenient object creation
//! - **Submodules** - Feature-gated [`streaming`] and [`playback`] modules for real-time processing
//!
//! # Key Design Features
//!
//! ## Dual Operation Modes
//! Following pandas conventions, methods come in two flavors:
//! - **In-place methods**: End with `_` and modify the object directly (e.g., `normalize_()`, `scale_()`)
//! - **Functional methods**: Return new objects without modifying the original (e.g., `normalize()`, `scale()`)
//!
//! ## Type Preservation
//! The bindings maintain the original sample type (i16, f32, f64, etc.) throughout operations
//! to preserve memory efficiency and numerical precision:
//!
//! ```python
//! import numpy as np
//! import audio_samples as aus
//!
//! # Create int16 audio (memory efficient)
//! audio_i16 = aus.from_numpy(np.random.randint(-32768, 32767, 44100, dtype=np.int16), 44100)
//! print(f"Original dtype: {audio_i16.dtype}")  # int16
//!
//! # Operations preserve type when possible
//! scaled = audio_i16.scale(0.5)
//! print(f"After scaling: {scaled.dtype}")  # Still int16
//! ```
//!
//! ## Rich Default Arguments
//! All functions provide Pythonic defaults to minimize boilerplate:
//!
//! ```python
//! # Minimal required arguments
//! audio.normalize()  # Uses method='peak', min_val=-1.0, max_val=1.0
//! audio.butterworth_filter(1000.0)  # Uses order=4, filter_type='low_pass'
//! audio.resample(48000)  # Uses quality='medium'
//! ```
//!
//! ## Zero-Copy Integration with NumPy
//! Efficient memory sharing with NumPy arrays when memory layouts are compatible:
//!
//! ```python
//! # Zero-copy conversion when possible
//! numpy_array = audio.numpy(copy=False)  # May share memory
//!
//! # Always get a copy when needed
//! numpy_copy = audio.numpy(copy=True)   # Guaranteed independent copy
//! ```
//!
//! # Quick Start Guide
//!
//! ## Basic Audio Processing Workflow
//!
//! ```python
//! import numpy as np
//! import audio_samples as aus
//!
//! # Create audio from NumPy array
//! data = np.random.randn(44100)  # 1 second of noise at 44.1kHz
//! audio = aus.from_numpy(data, sample_rate=44100)
//!
//! print(f"Audio info: {audio}")
//! # Output: AudioSamples(1 ch, 1.000s @ 44100Hz, float64)
//!
//! # Get basic statistics
//! print(f"RMS: {audio.rms():.3f}")
//! print(f"Peak: {audio.peak():.3f}")
//! print(f"Duration: {audio.duration:.3f}s")
//! ```
//!
//! ## In-Place vs Functional Operations
//!
//! ```python
//! import audio_samples as aus
//! import numpy as np
//!
//! # Create test audio
//! audio = aus.from_numpy(np.random.randn(44100), 44100)
//! original_peak = audio.peak()
//!
//! # Functional operation - returns new object
//! normalized = audio.normalize(method='peak')
//! print(f"Original peak: {audio.peak():.3f}")      # Unchanged
//! print(f"Normalized peak: {normalized.peak():.3f}")  # 1.0
//!
//! # In-place operation - modifies original
//! audio.normalize_(method='peak')
//! print(f"After in-place: {audio.peak():.3f}")    # Now 1.0
//! ```
//!
//! ## Advanced Processing Pipeline
//!
//! ```python
//! import audio_samples as aus
//! import numpy as np
//!
//! # Create stereo test signal
//! left = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 44100))   # 440Hz sine
//! right = np.sin(2 * np.pi * 880 * np.linspace(0, 1, 44100))  # 880Hz sine
//! stereo_data = np.vstack([left, right])
//! audio = aus.from_numpy(stereo_data, 44100)
//!
//! # Processing chain
//! processed = (audio
//!     .normalize(method='peak')           # Normalize to [-1, 1]
//!     .butterworth_filter(2000.0, filter_type='low_pass', order=6)  # Low-pass filter
//!     .fade_in(0.1)                      # 100ms fade-in
//!     .fade_out(0.1)                     # 100ms fade-out
//!     .scale(0.8))                       # Reduce volume to -1.9dB
//!
//! print(f"Processed: {processed}")
//! ```
//!
//! ## Type Conversion and Format Handling
//!
//! ```python
//! # Create high-precision audio
//! audio_f64 = aus.zeros(44100, 44100, dtype='f64')
//!
//! # Convert between formats
//! audio_f32 = audio_f64.to_f32()    # Convert to single precision
//! audio_i16 = audio_f64.to_i16()    # Convert to 16-bit integer (CD quality)
//!
//! # Check conversion quality
//! can_convert = audio_f64.can_convert_lossless('i16')
//! conversion_info = audio_f64.conversion_info('i16')
//! print(f"Can convert losslessly: {can_convert}")
//! print(f"Conversion info: {conversion_info}")
//! ```
//!
//! # Real-Time Processing
//!
//! When built with the appropriate features, real-time streaming and playback are available:
//!
//! ## Streaming Audio (Requires `streaming` feature)
//!
//! ```python
//! import audio_samples.streaming as streaming
//!
//! # Generate real-time audio stream
//! generator = streaming.GeneratorSource.sine(440.0, 44100, 2)  # 440Hz stereo
//!
//! # Process streaming audio
//! while True:
//!     chunk = await generator.next_chunk()
//!     if chunk is None:
//!         break
//!     
//!     # Process chunk
//!     processed_chunk = chunk.normalize().scale(0.8)
//!     # ... send to playback or network
//! ```
//!
//! ## Audio Playback (Requires `playback` feature)
//!
//! ```python
//! import audio_samples.playback as playback
//!
//! # Create simple player
//! player = playback.SimpleAudioPlayer()
//!
//! # Load and play audio
//! await player.load_audio(audio)
//! await player.play()
//!
//! # Control playback
//! player.set_volume(0.8)
//! await player.pause()
//! await player.seek(30.0)  # Seek to 30 seconds
//! await player.resume()
//! ```
//!
//! # Integration with Scientific Python Ecosystem
//!
//! The bindings integrate seamlessly with popular Python scientific libraries:
//!
//! ## NumPy Integration
//!
//! ```python
//! import numpy as np
//! import audio_samples as aus
//!
//! # Seamless conversion to/from NumPy
//! np_data = np.random.randn(2, 44100)  # Stereo audio
//! audio = aus.from_numpy(np_data, 44100)
//!
//! # Get NumPy view (zero-copy when possible)
//! np_view = audio.numpy(copy=False)
//!
//! # Use NumPy operations alongside audio_samples
//! windowed = audio.numpy() * np.hanning(44100)
//! windowed_audio = aus.from_numpy(windowed, 44100)
//! ```
//!
//! ## SciPy Integration
//!
//! ```python
//! import scipy.signal
//! import audio_samples as aus
//!
//! # Use SciPy filters with audio_samples
//! b, a = scipy.signal.butter(4, 0.2)  # Design filter
//! audio_np = audio.numpy()
//! filtered_np = scipy.signal.lfilter(b, a, audio_np)
//! filtered_audio = aus.from_numpy(filtered_np, audio.sample_rate)
//!
//! # Compare with built-in filtering
//! builtin_filtered = audio.butterworth_filter(0.2 * audio.sample_rate / 2)
//! ```
//!
//! # Performance Considerations
//!
//! ## Memory Efficiency
//! - Choose appropriate data types: `i16` for memory efficiency, `f64` for precision
//! - Use in-place operations (`method_()`) to avoid temporary object creation
//! - Prefer `copy=False` in `to_numpy()` when the array won't be modified
//!
//! ## Processing Speed
//! - Rust implementations are typically 2-10x faster than pure Python equivalents
//! - Batch operations are more efficient than sample-by-sample processing
//! - Multi-channel operations are optimized for parallel processing when possible
//!
//! ## Type Preservation
//! ```python
//! # Good: preserves original precision
//! result = audio.normalize().scale(0.8).clip(-1.0, 1.0)
//!
//! # Less efficient: unnecessary conversions
//! result = audio.to_f64().normalize().scale(0.8).clip(-1.0, 1.0).to_f32()
//! ```
//!
//! See the individual function and class documentation for detailed API reference.

use crate::operations::types::ResamplingQuality;
use crate::operations::{MonoConversionMethod, StereoConversionMethod};
use crate::{AudioChannelOps, AudioProcessing, AudioSampleError};
use crate::{AudioData, AudioSamples, I24, operations::AudioTypeConversion};
use numpy::{PyArrayDescr, PyArrayMethods};
use pyo3::exceptions::{PyIndexError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
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
mod dynamic_range;
mod editing;
mod generation;
mod iir_filtering;
mod parametric_eq;
mod processing;
mod statistics;
mod transforms;
mod utils;
mod utils_bindings;

// Import streaming and playback modules (feature-gated)
pub mod playback;
pub mod streaming;

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
            AudioSamplesData::I16(inner) => inner.num_channels(),
            AudioSamplesData::I24(inner) => inner.num_channels(),
            AudioSamplesData::I32(inner) => inner.num_channels(),
            AudioSamplesData::F32(inner) => inner.num_channels(),
            AudioSamplesData::F64(inner) => inner.num_channels(),
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
            AudioSamplesData::I16(inner) => (inner.num_channels(), inner.samples_per_channel()),
            AudioSamplesData::I24(inner) => (inner.num_channels(), inner.samples_per_channel()),
            AudioSamplesData::I32(inner) => (inner.num_channels(), inner.samples_per_channel()),
            AudioSamplesData::F32(inner) => (inner.num_channels(), inner.samples_per_channel()),
            AudioSamplesData::F64(inner) => (inner.num_channels(), inner.samples_per_channel()),
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
    /// * `dtype` - Target NumPy dtype (numpy.dtype or string like 'f32', 'f64', 'i16', 'i32') or None for original type
    /// * `copy` - Force a copy even if memory layout allows sharing
    ///
    /// # Returns
    /// NumPy array with shape (channels, samples) for multi-channel or (samples,) for mono
    ///
    /// # Examples
    /// ```python
    /// # Get as float64 (default)
    /// numpy_array = audio.numpy()
    ///
    /// # Convert using numpy dtypes (preferred)
    /// numpy_f32 = audio.numpy(dtype=np.float32)
    /// numpy_i16 = audio.numpy(dtype=np.int16)
    ///
    /// # Convert using string dtypes (convenience)
    /// numpy_f32 = audio.numpy(dtype='f32')
    ///
    /// # Force a copy
    /// numpy_copy = audio.numpy(copy=True)
    /// ```
    #[pyo3(signature = (*, dtype=None, copy=false))]
    fn numpy(&self, py: Python, dtype: Option<&Bound<PyAny>>, copy: bool) -> PyResult<Py<PyAny>> {
        let dtype_str = if let Some(dt) = dtype {
            // Try to extract numpy dtype first, fall back to string
            if let Ok(numpy_dtype) = dt.downcast::<numpy::PyArrayDescr>() {
                Some(conversions::get_target_type(numpy_dtype)?)
            } else if let Ok(dtype_str) = dt.extract::<String>() {
                utils::validate_string_param(
                    "dtype",
                    &dtype_str,
                    &[
                        "f64", "f32", "i32", "i16", "float64", "float32", "int32", "int16",
                    ],
                )?;
                Some(match dtype_str.as_str() {
                    "f64" | "float64" => conversions::TargetType::F64,
                    "f32" | "float32" => conversions::TargetType::F32,
                    "i32" | "int32" => conversions::TargetType::I32,
                    "i16" | "int16" => conversions::TargetType::I16,
                    _ => unreachable!(),
                })
            } else {
                return Err(PyErr::new::<PyTypeError, _>(
                    "dtype must be a numpy dtype or string ('f64', 'f32', 'i32', 'i16', 'float64', 'float32', 'int32', 'int16')",
                ));
            }
        } else {
            None
        };

        convert_audio_samples_to_numpy(py, &self.data, dtype_str, copy)
    }

    /// Length for Python len() function.
    fn __len__(&self) -> usize {
        self.length()
    }

    fn __copy__(&self) -> Self {
        self.copy()
    }

    fn __getitem__(&self, py: Python, idx: &Bound<PyAny>) -> PyResult<PyAudioSamples> {
        self.getitem_impl(py, idx)
    }

    fn __setitem__(
        &mut self,
        _py: Python,
        idx: &Bound<PyAny>,
        value: &Bound<PyAny>,
    ) -> PyResult<()> {
        self.setitem_impl(idx, value)
    }

    /// Create a copy of the AudioSamples object.
    #[pyo3(name = "copy")]
    fn copy(&self) -> Self {
        PyAudioSamples {
            data: self.data.clone(),
        }
    }

    /// Enhanced string representation showing metadata and sample values.
    ///
    /// Provides detailed information including actual sample values, similar to NumPy arrays.
    /// For large arrays, shows truncated values with ellipsis.
    fn __repr__(&self) -> String {
        self.repr_impl(true) // Detailed representation
    }

    /// Concise string representation showing metadata and sample preview.
    ///
    /// Provides a more compact view while still showing sample values.
    fn __str__(&self) -> String {
        self.repr_impl(false) // Concise representation  
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
    fn astype(&self, py: Python, dtype: &Bound<PyAny>, copy: bool) -> PyResult<Py<PyAny>> {
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
    ) -> PyResult<Py<PyAny>> {
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
    ) -> PyResult<Py<PyAny>> {
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
    fn conversion_info(&self, py: Python, target_dtype: &str) -> PyResult<Py<PyAny>> {
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
        let py_list = segments.downcast::<PyList>()?;
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
        let py_list = sources.downcast::<PyList>()?;
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

    // Dynamic Range Processing Methods - In-place operations
    /// Apply dynamic range compression in-place.
    #[pyo3(signature = (*, threshold=-20.0, ratio=4.0, attack=0.003, release=0.1, knee="hard", knee_width=2.0, makeup_gain=0.0), name = "compress_")]
    fn compress_(
        &mut self,
        threshold: Option<f64>,
        ratio: Option<f64>,
        attack: Option<f64>,
        release: Option<f64>,
        knee: Option<&str>,
        knee_width: Option<f64>,
        makeup_gain: Option<f64>,
    ) -> PyResult<()> {
        self.compress_in_place_impl(
            threshold,
            ratio,
            attack,
            release,
            knee,
            knee_width,
            makeup_gain,
        )
    }

    /// Apply brick-wall limiting in-place.
    #[pyo3(signature = (*, ceiling=-0.1, release=0.05, lookahead=0.005), name = "limit_")]
    fn limit_(
        &mut self,
        ceiling: Option<f64>,
        release: Option<f64>,
        lookahead: Option<f64>,
    ) -> PyResult<()> {
        self.limit_in_place_impl(ceiling, release, lookahead)
    }

    // IIR Filtering Methods - In-place operations
    /// Apply Butterworth IIR filter in-place.
    #[pyo3(signature = (cutoff, *, order=4, filter_type="low_pass", high_cutoff=None), name = "butterworth_filter_")]
    fn butterworth_filter_(
        &mut self,
        cutoff: f64,
        order: Option<usize>,
        filter_type: Option<&str>,
        high_cutoff: Option<f64>,
    ) -> PyResult<()> {
        self.butterworth_filter_in_place_impl(cutoff, order, filter_type, high_cutoff)
    }

    /// Apply Chebyshev Type I IIR filter in-place.
    #[pyo3(signature = (cutoff, *, order=4, ripple=0.1, filter_type="low_pass", high_cutoff=None), name = "chebyshev_filter_")]
    fn chebyshev_filter_(
        &mut self,
        cutoff: f64,
        order: Option<usize>,
        ripple: Option<f64>,
        filter_type: Option<&str>,
        high_cutoff: Option<f64>,
    ) -> PyResult<()> {
        self.chebyshev_filter_in_place_impl(cutoff, order, filter_type, ripple, high_cutoff)
    }

    // Parametric EQ Methods - In-place operations
    /// Apply parametric equalization in-place.
    #[pyo3(signature = (bands, *, output_gain=0.0), name = "parametric_eq_")]
    fn parametric_eq_(&mut self, bands: &Bound<PyAny>, output_gain: Option<f64>) -> PyResult<()> {
        self.parametric_eq_in_place_impl(bands, output_gain)
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

    // Dynamic Range Processing Methods - Functional operations
    /// Apply dynamic range compression (functional version).
    #[pyo3(signature = (*, threshold=-20.0, ratio=4.0, attack=0.003, release=0.1, knee="hard", knee_width=2.0, makeup_gain=0.0), name = "compress")]
    fn compress(
        &self,
        threshold: Option<f64>,
        ratio: Option<f64>,
        attack: Option<f64>,
        release: Option<f64>,
        knee: Option<&str>,
        knee_width: Option<f64>,
        makeup_gain: Option<f64>,
    ) -> PyResult<PyAudioSamples> {
        self.compress_impl(
            threshold,
            ratio,
            attack,
            release,
            knee,
            knee_width,
            makeup_gain,
        )
    }

    /// Apply brick-wall limiting (functional version).
    #[pyo3(signature = (*, ceiling=-0.1, release=0.05, lookahead=0.005), name = "limit")]
    fn limit(
        &self,
        ceiling: Option<f64>,
        release: Option<f64>,
        lookahead: Option<f64>,
    ) -> PyResult<PyAudioSamples> {
        self.limit_impl(ceiling, release, lookahead)
    }

    // IIR Filtering Methods - Functional operations
    /// Apply Butterworth IIR filter (functional version).
    #[pyo3(signature = (cutoff, *, order=4, filter_type="low_pass", high_cutoff=None), name = "butterworth_filter")]
    fn butterworth_filter(
        &self,
        cutoff: f64,
        order: Option<usize>,
        filter_type: Option<&str>,
        high_cutoff: Option<f64>,
    ) -> PyResult<PyAudioSamples> {
        self.butterworth_filter_impl(cutoff, order, filter_type, high_cutoff)
    }

    /// Apply Chebyshev Type I IIR filter (functional version).
    #[pyo3(signature = (cutoff, *, order=4, ripple=0.1, filter_type="low_pass", high_cutoff=None), name = "chebyshev_filter")]
    fn chebyshev_filter(
        &self,
        cutoff: f64,
        order: Option<usize>,
        ripple: Option<f64>,
        filter_type: Option<&str>,
        high_cutoff: Option<f64>,
    ) -> PyResult<PyAudioSamples> {
        self.chebyshev_filter_impl(cutoff, order, filter_type, ripple, high_cutoff)
    }

    // Parametric EQ Methods - Functional operations
    /// Apply parametric equalization (functional version).
    #[pyo3(signature = (bands, *, output_gain=0.0), name = "parametric_eq")]
    fn parametric_eq(
        &self,
        bands: &Bound<PyAny>,
        output_gain: Option<f64>,
    ) -> PyResult<PyAudioSamples> {
        self.parametric_eq_impl(bands, output_gain)
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
    fn fft(&self, py: Python, n: Option<usize>, axis: i32) -> PyResult<Py<PyAny>> {
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
    ) -> PyResult<Py<PyAny>> {
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
    ) -> PyResult<Py<PyAny>> {
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
    ) -> PyResult<Py<PyAny>> {
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
    ) -> PyResult<Py<PyAny>> {
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
    ) -> PyResult<Py<PyAny>> {
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
    ) -> PyResult<Py<PyAny>> {
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
    ) -> PyResult<Py<PyAny>> {
        self.power_spectral_density_impl(py, window_size, overlap, window)
    }

    // =====================
    // Dynamic Range Methods - TEMPORARILY DISABLED
    // =====================
    // These methods will be re-enabled once the corresponding Rust traits
    // and types are properly implemented.

    // ====================
    // IIR Filtering Methods - TEMPORARILY DISABLED
    // ====================
    // These methods will be re-enabled once the corresponding Rust traits
    // and types are properly implemented.

    // =======================
    // Parametric EQ Methods - TEMPORARILY DISABLED
    // =======================
    // These methods will be re-enabled once the corresponding Rust traits
    // and types are properly implemented.

    // ========================
    // Resampling Methods
    // ========================

    /// Resample audio to a target sample rate.
    ///
    /// # Arguments
    /// * `target_sample_rate` - Target sample rate in Hz
    /// * `quality` - Resampling quality ('fast', 'medium', 'high')
    ///
    /// # Examples
    /// ```python
    /// import audio_samples as aus
    /// import numpy as np
    ///
    /// # Resample from 44.1kHz to 48kHz
    /// audio = aus.from_numpy(np.random.randn(44100), sample_rate=44100)
    /// resampled = audio.resample(48000, quality='high')
    /// print(f"New sample rate: {resampled.sample_rate}")
    /// ```
    #[pyo3(signature = (target_sample_rate, *, quality="medium"), name = "resample")]
    fn resample(&self, target_sample_rate: u32, quality: &str) -> PyResult<PyAudioSamples> {
        self.resample_impl(target_sample_rate, quality)
    }

    /// Resample audio by a specific ratio.
    ///
    /// # Arguments
    /// * `ratio` - Resampling ratio (output_rate / input_rate)
    /// * `quality` - Resampling quality ('fast', 'medium', 'high')
    ///
    /// # Examples
    /// ```python
    /// # Upsample by 2x (double the sample rate)
    /// upsampled = audio.resample_by_ratio(2.0, quality='high')
    ///
    /// # Downsample by half
    /// downsampled = audio.resample_by_ratio(0.5, quality='medium')
    /// ```
    #[pyo3(signature = (ratio, *, quality="medium"), name = "resample_by_ratio")]
    fn resample_by_ratio(&self, ratio: f64, quality: &str) -> PyResult<PyAudioSamples> {
        self.resample_by_ratio_impl(ratio, quality)
    }

    // ========================
    // Enhanced Channel Methods
    // ========================

    /// Pan stereo audio left or right.
    #[pyo3(signature = (position,), name = "pan")]
    fn pan(&self, position: f64) -> PyResult<PyAudioSamples> {
        self.pan_impl(position)
    }

    /// Pan stereo audio left or right in-place.
    #[pyo3(signature = (position,), name = "pan_")]
    fn pan_in_place(&mut self, position: f64) -> PyResult<()> {
        self.pan_in_place_impl(position)
    }

    /// Adjust stereo balance between left and right channels.
    #[pyo3(signature = (balance,), name = "balance")]
    fn balance(&self, balance: f64) -> PyResult<PyAudioSamples> {
        self.balance_impl(balance)
    }

    /// Adjust stereo balance between left and right channels in-place.
    #[pyo3(signature = (balance,), name = "balance_")]
    fn balance_in_place(&mut self, balance: f64) -> PyResult<()> {
        self.balance_in_place_impl(balance)
    }

    /// Convert to mono by mixing all channels with equal weights.
    #[pyo3(signature = (), name = "to_mono")]
    fn to_mono(&self) -> PyResult<PyAudioSamples> {
        self.to_mono_impl()
    }

    /// Convert mono audio to stereo by duplicating the channel.
    #[pyo3(signature = (), name = "to_stereo")]
    fn to_stereo(&self) -> PyResult<PyAudioSamples> {
        self.to_stereo_impl()
    }

    /// Extract a specific channel as mono audio.
    #[pyo3(signature = (channel_index,), name = "extract_channel")]
    fn extract_channel(&self, channel_index: usize) -> PyResult<PyAudioSamples> {
        self.extract_channel_impl(channel_index)
    }

    /// Swap left and right channels (for stereo audio).
    #[pyo3(signature = (), name = "swap_channels")]
    fn swap_channels(&self) -> PyResult<PyAudioSamples> {
        self.swap_channels_impl()
    }

    /// Swap left and right channels in-place (for stereo audio).
    #[pyo3(signature = (), name = "swap_channels_")]
    fn swap_channels_in_place(&mut self) -> PyResult<()> {
        self.swap_channels_in_place_impl()
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
    let _ = copy; // TODO: Implement copy parameter
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
            conversions::get_target_type(numpy_dtype)?
        } else if let Ok(dtype_str) = dt.extract::<String>() {
            utils::validate_string_param("dtype", &dtype_str, &["f64", "f32", "i32", "i16"])?;
            match dtype_str.as_str() {
                "f64" => conversions::TargetType::F64,
                "f32" => conversions::TargetType::F32,
                "i32" => conversions::TargetType::I32,
                "i16" => conversions::TargetType::I16,
                _ => unreachable!(),
            }
        } else {
            return Err(PyErr::new::<PyTypeError, _>(
                "dtype must be a numpy dtype or string ('f64', 'f32', 'i32', 'i16')",
            ));
        }
    } else {
        conversions::TargetType::F64
    };

    // Create zeros array in the target type
    let data = match target_type {
        conversions::TargetType::F64 => {
            let inner = if channels == 1 {
                AudioSamples::<f64>::zeros_mono(length, sample_rate)
            } else {
                AudioSamples::<f64>::zeros_multi(channels, length, sample_rate)
            };
            AudioSamplesData::F64(inner)
        }
        conversions::TargetType::F32 => {
            let inner_f64 = if channels == 1 {
                AudioSamples::<f64>::zeros_mono(length, sample_rate)
            } else {
                AudioSamples::<f64>::zeros_multi(channels, length, sample_rate)
            };
            let inner_f32 = inner_f64.as_f32().map_err(utils::map_error)?;
            AudioSamplesData::F32(inner_f32)
        }
        conversions::TargetType::I32 => {
            let inner_f64 = if channels == 1 {
                AudioSamples::<f64>::zeros_mono(length, sample_rate)
            } else {
                AudioSamples::<f64>::zeros_multi(channels, length, sample_rate)
            };
            let inner_i32 = inner_f64.as_i32().map_err(utils::map_error)?;
            AudioSamplesData::I32(inner_i32)
        }
        conversions::TargetType::I16 => {
            let inner_f64 = if channels == 1 {
                AudioSamples::<f64>::zeros_mono(length, sample_rate)
            } else {
                AudioSamples::<f64>::zeros_multi(channels, length, sample_rate)
            };
            let inner_i16 = inner_f64.as_i16().map_err(utils::map_error)?;
            AudioSamplesData::I16(inner_i16)
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
            conversions::get_target_type(numpy_dtype)?
        } else if let Ok(dtype_str) = dt.extract::<String>() {
            utils::validate_string_param(
                "dtype",
                &dtype_str,
                &[
                    "f64", "f32", "i32", "i16", "float64", "float32", "int32", "int16",
                ],
            )?;
            match dtype_str.as_str() {
                "f64" | "float64" => conversions::TargetType::F64,
                "f32" | "float32" => conversions::TargetType::F32,
                "i32" | "int32" => conversions::TargetType::I32,
                "i16" | "int16" => conversions::TargetType::I16,
                _ => unreachable!("If invalid, then it would have errored above"),
            }
        } else {
            return Err(PyErr::new::<PyTypeError, _>(
                "dtype must be a numpy dtype or string ('f64', 'f32', 'i32', 'i16', 'float64', 'float32', 'int32', 'int16')",
            ));
        }
    } else {
        conversions::TargetType::F64
    };

    // Create ones array in the target type
    let data = match target_type {
        conversions::TargetType::F64 => {
            let mut inner = if channels == 1 {
                AudioSamples::<f64>::zeros_mono(length, sample_rate)
            } else {
                AudioSamples::<f64>::zeros_multi(channels, length, sample_rate)
            };
            // Fill with ones
            match inner.num_channels() {
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
            AudioSamplesData::F64(inner)
        }
        conversions::TargetType::F32 => {
            let mut inner_f64 = if channels == 1 {
                AudioSamples::<f64>::zeros_mono(length, sample_rate)
            } else {
                AudioSamples::<f64>::zeros_multi(channels, length, sample_rate)
            };
            // Fill with ones
            match inner_f64.num_channels() {
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
            let inner_f32 = inner_f64.as_f32().map_err(utils::map_error)?;
            AudioSamplesData::F32(inner_f32)
        }
        conversions::TargetType::I32 => {
            let mut inner_f64 = if channels == 1 {
                AudioSamples::<f64>::zeros_mono(length, sample_rate)
            } else {
                AudioSamples::<f64>::zeros_multi(channels, length, sample_rate)
            };
            // Fill with ones
            match inner_f64.num_channels() {
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
            let inner_i32 = inner_f64.as_i32().map_err(utils::map_error)?;
            AudioSamplesData::I32(inner_i32)
        }
        conversions::TargetType::I16 => {
            let mut inner_f64 = if channels == 1 {
                AudioSamples::<f64>::zeros_mono(length, sample_rate)
            } else {
                AudioSamples::<f64>::zeros_multi(channels, length, sample_rate)
            };
            // Fill with ones
            match inner_f64.num_channels() {
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
            let inner_i16 = inner_f64.as_i16().map_err(utils::map_error)?;
            AudioSamplesData::I16(inner_i16)
        }
    };

    Ok(PyAudioSamples { data })
}

/// Get information about the audio_samples library.
///
/// # Returns
/// A dictionary with library information including version, features, and capabilities.
#[pyfunction(name = "info")]
fn library_info(py: Python) -> PyResult<Py<PyAny>> {
    let info = PyDict::new(py);
    info.set_item("version", "0.1.0")?;
    info.set_item("supported_dtypes", vec!["i16", "I24", "i32", "f32", "f64"])?;

    let mut features = Vec::new();
    features.push("streaming");
    features.push("playback");
    info.set_item("features", features)?;
    info.set_item(
        "description",
        "High-performance audio processing library which puts Audio first.",
    )?;

    Ok(info.into())
}

/// List all available audio processing operations categorized by type.
#[pyfunction(name = "available_operations")]
fn available_operations(py: Python) -> PyResult<Py<PyAny>> {
    use PyDict;

    let ops = PyDict::new(py);

    ops.set_item(
        "statistics",
        vec![
            "peak",
            "min",
            "max",
            "rms",
            "variance",
            "std_dev",
            "zero_crossings",
            "zero_crossing_rate",
            "spectral_centroid",
            "spectral_rolloff",
        ],
    )?;

    ops.set_item(
        "processing",
        vec![
            "normalize",
            "scale",
            "apply_window",
            "apply_filter",
            "mu_compress",
            "mu_expand",
            "clip",
            "remove_dc_offset",
        ],
    )?;

    ops.set_item(
        "filtering",
        vec!["low_pass_filter", "high_pass_filter", "band_pass_filter"],
    )?;

    ops.set_item("resampling", vec!["resample", "resample_by_ratio"])?;

    ops.set_item(
        "editing",
        vec![
            "reverse",
            "trim",
            "pad",
            "split",
            "repeat",
            "trim_silence",
            "fade_in",
            "fade_out",
            "concatenate",
            "mix",
            "crossfade",
        ],
    )?;

    ops.set_item(
        "transforms",
        vec![
            "fft",
            "ifft",
            "stft",
            "istft",
            "spectrogram",
            "mel_spectrogram",
            "mfcc",
            "gammatone_spectrogram",
            "chroma",
            "power_spectral_density",
        ],
    )?;

    ops.set_item(
        "channels",
        vec![
            "pan",
            "balance",
            "to_mono",
            "to_stereo",
            "extract_channel",
            "swap_channels",
        ],
    )?;

    ops.set_item(
        "conversions",
        vec!["astype", "to_f64", "to_f32", "to_i32", "to_i16", "to_numpy"],
    )?;

    ops.set_item(
        "generation",
        vec![
            "sine_wave",
            "cosine_wave",
            "white_noise",
            "pink_noise",
            "square_wave",
            "sawtooth_wave",
            "triangle_wave",
            "chirp",
            "impulse",
            "silence",
        ],
    )?;

    ops.set_item(
        "detection",
        vec![
            "detect_sample_rate",
            "detect_fundamental_frequency",
            "detect_silence_regions",
            "detect_dynamic_range",
            "detect_clipping",
        ],
    )?;

    ops.set_item(
        "comparison",
        vec!["correlation", "mse", "snr", "align_signals"],
    )?;

    Ok(ops.into())
}

/// Module Registration
/// ==================

/// Register the audio_samples Python module.
///
/// This function is called from the main lib.rs to set up the Python module
/// with all classes, functions, and constants.
pub fn register_module(py: Python, m: &Bound<PyModule>) -> PyResult<()> {
    // Main class
    m.add_class::<PyAudioSamples>()?;

    // Factory functions
    m.add_function(wrap_pyfunction!(from_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(zeros, m)?)?;
    m.add_function(wrap_pyfunction!(ones, m)?)?;
    m.add_function(wrap_pyfunction!(to_tensor, m)?)?;
    m.add_function(wrap_pyfunction!(to_gpu, m)?)?;
    m.add_function(wrap_pyfunction!(from_tensor, m)?)?;

    // Signal generation functions
    generation::register_functions(m)?;

    // Utils functions
    utils_bindings::register_functions(m)?;

    // Helper/info functions
    m.add_function(wrap_pyfunction!(library_info, m)?)?;
    m.add_function(wrap_pyfunction!(available_operations, m)?)?;

    // Register streaming submodule (if feature enabled)
    #[cfg(feature = "streaming")]
    {
        let streaming_module = PyModule::new(py, "streaming")?;
        streaming::register_module(py, &streaming_module)?;
        m.add_submodule(&streaming_module)?;
        // Make the submodule importable by registering it in sys.modules
        py.import("sys")?
            .getattr("modules")?
            .set_item("audio_samples.streaming", &streaming_module)?;
    }

    // Register playback submodule (if feature enabled)
    #[cfg(feature = "playback")]
    {
        let playback_module = PyModule::new(py, "playback")?;
        playback::register_module(py, &playback_module)?;
        m.add_submodule(&playback_module)?;
        // Make the submodule importable by registering it in sys.modules
        py.import("sys")?
            .getattr("modules")?
            .set_item("audio_samples.playback", &playback_module)?;
    }

    // Module constants
    m.add("__version__", "0.1.0")?;
    m.add("SUPPORTED_DTYPES", vec!["i16", "I24", "i32", "f32", "f64"])?;

    // Control what gets exported with __all__
    let mut all_items = vec![
        "AudioSamples",
        "from_numpy",
        "zeros",
        "ones",
        "sine_wave",
        "cosine_wave",
        "white_noise",
        "pink_noise",
        "square_wave",
        "sawtooth_wave",
        "triangle_wave",
        "chirp",
        "impulse",
        "silence",
        "detect_sample_rate",
        "detect_fundamental_frequency",
        "detect_silence_regions",
        "detect_dynamic_range",
        "detect_clipping",
        "correlation",
        "mse",
        "snr",
        "align_signals",
        "info",
        "available_operations",
        "__version__",
        "SUPPORTED_DTYPES",
    ];

    // Add submodules to __all__ if they exist
    #[cfg(feature = "streaming")]
    all_items.push("streaming");

    #[cfg(feature = "playback")]
    all_items.push("playback");

    m.add("__all__", all_items)?;

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
    pub(crate) fn data(&self) -> &AudioSamplesData {
        &self.data
    }

    /// Create a new PyAudioSamples from AudioSamplesData
    pub(crate) fn from_data(data: AudioSamplesData) -> Self {
        Self { data }
    }

    // Helper methods for backward compatibility in submodules
    /// Create a new PyAudioSamples from an AudioSamples<f64> (for compatibility)
    pub(crate) fn from_inner(inner: AudioSamples<f64>) -> Self {
        Self {
            data: AudioSamplesData::F64(inner),
        }
    }

    /// Helper method: convert to f64 AudioSamples for operations that need it
    pub(crate) fn as_f64(&self) -> Result<AudioSamples<f64>, AudioSampleError> {
        use crate::operations::AudioTypeConversion;
        match &self.data {
            AudioSamplesData::F64(audio) => Ok(audio.clone()),
            AudioSamplesData::F32(audio) => audio.as_f64(),
            AudioSamplesData::I32(audio) => audio.as_f64(),
            AudioSamplesData::I16(audio) => audio.as_f64(),
            AudioSamplesData::I24(audio) => audio.as_f64(),
        }
    }

    /// Helper method: apply operation to inner data and wrap result
    pub(crate) fn with_inner<F, T>(&self, f: F) -> Result<T, AudioSampleError>
    where
        F: Fn(&AudioSamples<f64>) -> Result<T, AudioSampleError>,
    {
        let f64_audio = self.as_f64()?;
        f(&f64_audio)
    }

    /// Helper method: apply operation and return new PyAudioSamples
    pub(crate) fn map_inner<F>(&self, f: F) -> Result<PyAudioSamples, AudioSampleError>
    where
        F: Fn(&AudioSamples<f64>) -> Result<AudioSamples<f64>, AudioSampleError>,
    {
        let f64_audio = self.as_f64()?;
        let result = f(&f64_audio)?;
        Ok(PyAudioSamples::from_inner(result))
    }

    /// Helper method: apply mutable operation in-place (converts to f64 if needed)
    pub(crate) fn mutate_inner<F>(&mut self, f: F) -> Result<(), AudioSampleError>
    where
        F: Fn(&mut AudioSamples<f64>) -> Result<(), AudioSampleError>,
    {
        // Convert to f64 for operation
        let mut f64_audio = self.as_f64()?;
        f(&mut f64_audio)?;
        self.data = AudioSamplesData::F64(f64_audio);
        Ok(())
    }

    /// Helper method: apply operation to mutable inner data
    pub(crate) fn with_inner_mut<F, T>(&mut self, f: F) -> Result<T, AudioSampleError>
    where
        F: Fn(&mut AudioSamples<f64>) -> Result<T, AudioSampleError>,
    {
        let mut f64_audio = self.as_f64()?;
        let result = f(&mut f64_audio)?;
        self.data = AudioSamplesData::F64(f64_audio);
        Ok(result)
    }

    // Enhanced Channel Operations Implementation
    /// Pan stereo audio left or right.
    pub(crate) fn pan_impl(&self, position: f64) -> PyResult<PyAudioSamples> {
        use crate::operations::AudioChannelOps;

        if position < -1.0 || position > 1.0 {
            return Err(PyErr::new::<PyValueError, _>(
                "Pan position must be between -1.0 (left) and 1.0 (right)",
            ));
        }

        let mut f64_audio = self.as_f64().map_err(utils::map_error)?;
        f64_audio.pan(position).map_err(utils::map_error)?;
        Ok(PyAudioSamples::from_inner(f64_audio))
    }

    pub(crate) fn pan_in_place_impl(&mut self, position: f64) -> PyResult<()> {
        if position < -1.0 || position > 1.0 {
            return Err(PyErr::new::<PyValueError, _>(
                "Pan position must be between -1.0 (left) and 1.0 (right)",
            ));
        }

        self.with_inner_mut(|inner| {
            inner.pan(position)?;
            Ok(())
        })
        .map_err(utils::map_error)
    }

    /// Adjust stereo balance between left and right channels.
    pub(crate) fn balance_impl(&self, balance: f64) -> PyResult<PyAudioSamples> {
        if balance < -1.0 || balance > 1.0 {
            return Err(PyErr::new::<PyValueError, _>(
                "Balance must be between -1.0 (left) and 1.0 (right)",
            ));
        }

        let mut f64_audio = self.as_f64().map_err(utils::map_error)?;
        f64_audio.balance(balance).map_err(utils::map_error)?;
        Ok(PyAudioSamples::from_inner(f64_audio))
    }

    pub(crate) fn balance_in_place_impl(&mut self, balance: f64) -> PyResult<()> {
        if balance < -1.0 || balance > 1.0 {
            return Err(PyErr::new::<PyValueError, _>(
                "Balance must be between -1.0 (left) and 1.0 (right)",
            ));
        }

        self.with_inner_mut(|inner| {
            inner.balance(balance)?;
            Ok(())
        })
        .map_err(utils::map_error)
    }

    /// Convert to mono by mixing all channels with equal weights.
    pub(crate) fn to_mono_impl(&self) -> PyResult<PyAudioSamples> {
        let mono = self
            .with_inner(|inner| inner.to_mono(MonoConversionMethod::Average))
            .map_err(utils::map_error)?;
        Ok(PyAudioSamples::from_inner(mono))
    }

    /// Convert mono audio to stereo by duplicating the channel.
    pub(crate) fn to_stereo_impl(&self) -> PyResult<PyAudioSamples> {
        let stereo = self
            .with_inner(|inner| inner.to_stereo(StereoConversionMethod::Duplicate))
            .map_err(utils::map_error)?;
        Ok(PyAudioSamples::from_inner(stereo))
    }

    /// Extract a specific channel as mono audio.
    pub(crate) fn extract_channel_impl(&self, channel_index: usize) -> PyResult<PyAudioSamples> {
        if channel_index >= self.channels() {
            return Err(PyErr::new::<PyValueError, _>(format!(
                "Channel index {} out of range (0-{})",
                channel_index,
                self.channels() - 1
            )));
        }

        let channel = self
            .with_inner(|inner| inner.extract_channel(channel_index))
            .map_err(utils::map_error)?;
        Ok(PyAudioSamples::from_inner(channel))
    }

    /// Swap left and right channels (for stereo audio).
    pub(crate) fn swap_channels_impl(&self) -> PyResult<PyAudioSamples> {
        if self.channels() != 2 {
            return Err(PyErr::new::<PyValueError, _>(
                "Channel swapping requires exactly 2 channels (stereo audio)",
            ));
        }

        let mut f64_audio = self.as_f64().map_err(utils::map_error)?;
        f64_audio.swap_channels(0, 1).map_err(utils::map_error)?;
        Ok(PyAudioSamples::from_inner(f64_audio))
    }

    pub(crate) fn swap_channels_in_place_impl(&mut self) -> PyResult<()> {
        use crate::operations::AudioChannelOps;

        if self.channels() != 2 {
            return Err(PyErr::new::<PyValueError, _>(
                "Channel swapping requires exactly 2 channels (stereo audio)",
            ));
        }

        self.with_inner_mut(|inner| {
            inner.swap_channels(0, 1)?;
            Ok(())
        })
        .map_err(utils::map_error)
    }

    // ========================
    // Resampling Implementation Methods
    // ========================

    /// Parse resampling quality from string parameter.
    pub(crate) fn parse_resampling_quality_impl(quality: &str) -> PyResult<ResamplingQuality> {
        match quality.to_lowercase().as_str() {
            "fast" | "low" | "low_quality" => Ok(ResamplingQuality::Fast),
            "medium" | "med" | "medium_quality" => Ok(ResamplingQuality::Medium),
            "high" | "high_quality" | "best" => Ok(ResamplingQuality::High),
            _ => Err(PyErr::new::<PyValueError, _>(format!(
                "Invalid resampling quality: '{}'. Valid options: 'fast', 'medium', 'high' (or aliases like 'high_quality')",
                quality
            ))),
        }
    }

    /// Resample audio to a target sample rate.
    pub(crate) fn resample_impl(
        &self,
        target_sample_rate: u32,
        quality: &str,
    ) -> PyResult<PyAudioSamples> {
        let quality_enum = Self::parse_resampling_quality_impl(quality)?;

        let resampled = self
            .with_inner(|inner| inner.resample(target_sample_rate as usize, quality_enum))
            .map_err(utils::map_error)?;

        Ok(PyAudioSamples::from_inner(resampled))
    }

    /// Resample audio by a specific ratio.
    pub(crate) fn resample_by_ratio_impl(
        &self,
        ratio: f64,
        quality: &str,
    ) -> PyResult<PyAudioSamples> {
        if ratio <= 0.0 {
            return Err(PyErr::new::<PyValueError, _>(format!(
                "Resampling ratio must be positive, got: {}",
                ratio
            )));
        }

        let quality_enum = Self::parse_resampling_quality_impl(quality)?;

        let resampled = self
            .with_inner(|inner| inner.resample_by_ratio(ratio, quality_enum))
            .map_err(utils::map_error)?;

        Ok(PyAudioSamples::from_inner(resampled))
    }

    // ========================
    // Indexing and Slicing Implementation - Python to Rust bridge
    // ========================

    /// Convert Python indexing/slicing to Rust AudioSamples operations.
    ///
    /// Supports:
    /// - Single integer: `audio[42]` -> extract single sample (mono) or error (multi-channel)
    /// - Slice object: `audio[10:50]` -> slice_samples(10..50)
    /// - Tuple: `audio[0, 10:50]` -> channel 0, samples 10-50
    /// - Tuple with slices: `audio[:, 100:200]` -> all channels, samples 100-200
    fn getitem_impl(&self, _py: Python, idx: &Bound<PyAny>) -> PyResult<PyAudioSamples> {
        // Handle single integer index
        if let Ok(index) = idx.extract::<isize>() {
            return self.getitem_single_index(index);
        }

        // Handle slice object
        if let Ok(slice) = idx.downcast::<pyo3::types::PySlice>() {
            return self.getitem_slice(slice);
        }

        // Handle tuple (multi-dimensional indexing)
        if let Ok(tuple) = idx.downcast::<pyo3::types::PyTuple>() {
            return self.getitem_tuple(tuple);
        }

        // Unsupported indexing type
        Err(PyErr::new::<PyTypeError, _>(format!(
            "Unsupported index type: {}. Use int, slice, or tuple of int/slice.",
            idx.get_type().name()?
        )))
    }

    /// Handle single integer indexing: `audio[42]`
    fn getitem_single_index(&self, index: isize) -> PyResult<PyAudioSamples> {
        // Only allow single indexing for mono audio
        if self.channels() != 1 {
            return Err(PyErr::new::<PyValueError, _>(
                "Single index not supported for multi-channel audio. Use audio[channel, sample] or audio[:, sample_range].",
            ));
        }

        let length = self.length() as isize;
        let actual_index = if index < 0 { length + index } else { index };

        if actual_index < 0 || actual_index >= length {
            return Err(PyErr::new::<PyIndexError, _>(format!(
                "Index {} out of bounds for audio with {} samples",
                index, length
            )));
        }

        // Extract single sample as a new AudioSamples instance
        let rust_index = actual_index as usize;
        self.with_inner(|inner| inner.slice_samples(rust_index..rust_index + 1))
            .map_err(utils::map_error)
            .map(PyAudioSamples::from_inner)
    }

    /// Handle slice indexing: `audio[10:50]` or `audio[::2]`
    fn getitem_slice(&self, slice: &Bound<pyo3::types::PySlice>) -> PyResult<PyAudioSamples> {
        let length = self.length();

        // Convert Python slice to Rust range
        let indices = slice.indices(length as isize)?;
        let start = indices.start as usize;
        let stop = indices.stop as usize;
        let step = indices.step;

        if step != 1 {
            return Err(PyErr::new::<PyValueError, _>(
                "Step slicing not yet implemented. Use audio[start:stop] without step.",
            ));
        }

        if start >= stop {
            return Err(PyErr::new::<PyValueError, _>(format!(
                "Invalid slice range: {}:{} results in empty slice",
                start, stop
            )));
        }

        // Use our Rust slice_samples method
        self.with_inner(|inner| inner.slice_samples(start..stop))
            .map_err(utils::map_error)
            .map(PyAudioSamples::from_inner)
    }

    /// Handle tuple indexing: `audio[0, 10:50]` or `audio[:, 100:200]`
    fn getitem_tuple(&self, tuple: &Bound<pyo3::types::PyTuple>) -> PyResult<PyAudioSamples> {
        if tuple.len() != 2 {
            return Err(PyErr::new::<PyValueError, _>(
                "Tuple indexing must have exactly 2 elements: (channel_index, sample_index/slice)",
            ));
        }

        let channel_item = tuple.get_item(0)?;
        let sample_item = tuple.get_item(1)?;

        // Parse channel specification
        let channel_range = self.parse_channel_spec(&channel_item)?;

        // Parse sample specification
        let sample_range = self.parse_sample_spec(&sample_item)?;

        // Apply the slicing using our Rust methods
        match (channel_range, sample_range) {
            (Some((ch_start, ch_end)), Some((s_start, s_end))) => {
                // Both channels and samples specified
                self.with_inner(|inner| inner.slice_both(ch_start..ch_end, s_start..s_end))
                    .map_err(utils::map_error)
                    .map(PyAudioSamples::from_inner)
            }
            (None, Some((s_start, s_end))) => {
                // All channels, specific sample range
                self.with_inner(|inner| inner.slice_samples(s_start..s_end))
                    .map_err(utils::map_error)
                    .map(PyAudioSamples::from_inner)
            }
            (Some((ch_start, ch_end)), None) => {
                // Specific channels, all samples
                self.with_inner(|inner| inner.slice_channels(ch_start..ch_end))
                    .map_err(utils::map_error)
                    .map(PyAudioSamples::from_inner)
            }
            (None, None) => {
                // This would be [:, :] - just return a copy
                Ok(self.copy())
            }
        }
    }

    /// Parse channel specification: int or slice
    /// Returns None for ":" (all channels), Some((start, end)) for specific range
    fn parse_channel_spec(&self, item: &Bound<PyAny>) -> PyResult<Option<(usize, usize)>> {
        // Handle single channel index
        if let Ok(index) = item.extract::<isize>() {
            let num_channels = self.channels() as isize;
            let actual_index = if index < 0 {
                num_channels + index
            } else {
                index
            };

            if actual_index < 0 || actual_index >= num_channels {
                return Err(PyErr::new::<PyIndexError, _>(format!(
                    "Channel index {} out of bounds for audio with {} channels",
                    index, num_channels
                )));
            }

            let ch = actual_index as usize;
            return Ok(Some((ch, ch + 1)));
        }

        // Handle slice
        if let Ok(slice) = item.downcast::<pyo3::types::PySlice>() {
            let num_channels = self.channels();
            let indices = slice.indices(num_channels as isize)?;

            if indices.step != 1 {
                return Err(PyErr::new::<PyValueError, _>(
                    "Step slicing not supported for channels",
                ));
            }

            let start = indices.start as usize;
            let stop = indices.stop as usize;

            if start >= stop {
                return Err(PyErr::new::<PyValueError, _>(format!(
                    "Invalid channel slice: {}:{} results in empty slice",
                    start, stop
                )));
            }

            // Check for full range slice (equivalent to ":")
            if start == 0 && stop == num_channels {
                return Ok(None); // All channels
            }

            return Ok(Some((start, stop)));
        }

        Err(PyErr::new::<PyTypeError, _>(
            "Channel index must be int or slice",
        ))
    }

    /// Parse sample specification: int or slice
    /// Returns None for ":" (all samples), Some((start, end)) for specific range
    fn parse_sample_spec(&self, item: &Bound<PyAny>) -> PyResult<Option<(usize, usize)>> {
        // Handle single sample index
        if let Ok(index) = item.extract::<isize>() {
            let length = self.length() as isize;
            let actual_index = if index < 0 { length + index } else { index };

            if actual_index < 0 || actual_index >= length {
                return Err(PyErr::new::<PyIndexError, _>(format!(
                    "Sample index {} out of bounds for audio with {} samples",
                    index, length
                )));
            }

            let idx = actual_index as usize;
            return Ok(Some((idx, idx + 1)));
        }

        // Handle slice
        if let Ok(slice) = item.downcast::<pyo3::types::PySlice>() {
            let length = self.length();
            let indices = slice.indices(length as isize)?;

            if indices.step != 1 {
                return Err(PyErr::new::<PyValueError, _>(
                    "Step slicing not supported for samples",
                ));
            }

            let start = indices.start as usize;
            let stop = indices.stop as usize;

            if start >= stop {
                return Err(PyErr::new::<PyValueError, _>(format!(
                    "Invalid sample slice: {}:{} results in empty slice",
                    start, stop
                )));
            }

            // Check for full range slice (equivalent to ":")
            if start == 0 && stop == length {
                return Ok(None); // All samples
            }

            return Ok(Some((start, stop)));
        }

        Err(PyErr::new::<PyTypeError, _>(
            "Sample index must be int or slice",
        ))
    }

    // ========================
    // Assignment (setitem) Implementation
    // ========================

    /// Implementation for Python __setitem__ operations.
    ///
    /// Supports assignment patterns like:
    /// - audio[42] = value         # Single sample (mono only)
    /// - audio[10:50] = values     # Sample range
    /// - audio[:, 100:200] = array # All channels, sample range
    /// - audio[0, :] = channel     # Single channel, all samples
    /// - audio[0, 10:50] = values  # Single channel, sample range
    fn setitem_impl(&mut self, idx: &Bound<PyAny>, value: &Bound<PyAny>) -> PyResult<()> {
        // Handle single integer index
        if let Ok(index) = idx.extract::<isize>() {
            return self.setitem_single_index(index, value);
        }

        // Handle slice object
        if let Ok(slice) = idx.downcast::<pyo3::types::PySlice>() {
            return self.setitem_slice(slice, value);
        }

        // Handle tuple (multi-dimensional assignment)
        if let Ok(tuple) = idx.downcast::<pyo3::types::PyTuple>() {
            return self.setitem_tuple(tuple, value);
        }

        // Unsupported indexing type
        Err(PyErr::new::<PyTypeError, _>(format!(
            "Unsupported index type: {}. Use int, slice, or tuple of int/slice.",
            idx.get_type().name()?
        )))
    }

    /// Handle single integer assignment: `audio[42] = value`
    fn setitem_single_index(&mut self, index: isize, value: &Bound<PyAny>) -> PyResult<()> {
        // Only allow single indexing for mono audio
        if self.channels() != 1 {
            return Err(PyErr::new::<PyValueError, _>(
                "Single index assignment not supported for multi-channel audio. Use audio[channel, sample] or audio[:, sample_range].",
            ));
        }

        let length = self.length() as isize;
        let actual_index = if index < 0 { length + index } else { index };

        if actual_index < 0 || actual_index >= length {
            return Err(PyErr::new::<PyIndexError, _>(format!(
                "Index {} out of bounds for audio with {} samples",
                index, length
            )));
        }

        // Extract the scalar value
        let sample_value = self.extract_scalar_value(value)?;
        let rust_index = actual_index as usize;

        // Assign to the appropriate sample type
        self.set_single_sample(0, rust_index, sample_value)
    }

    /// Handle slice assignment: `audio[10:50] = values`
    fn setitem_slice(
        &mut self,
        slice: &Bound<pyo3::types::PySlice>,
        value: &Bound<PyAny>,
    ) -> PyResult<()> {
        let length = self.length();

        // Convert Python slice to Rust range
        let indices = slice.indices(length as isize)?;
        let start = indices.start as usize;
        let stop = indices.stop as usize;
        let step = indices.step;

        if step != 1 {
            return Err(PyErr::new::<PyValueError, _>(
                "Step assignment not yet implemented. Use audio[start:stop] without step.",
            ));
        }

        if start >= stop {
            return Err(PyErr::new::<PyValueError, _>(format!(
                "Invalid slice range: {}:{} results in empty slice",
                start, stop
            )));
        }

        let range_length = stop - start;

        // Extract values array from Python
        let values = self.extract_values_array(value, Some(range_length))?;

        // Assign to all channels for the given sample range
        self.set_sample_range_all_channels(start, stop, &values)
    }

    /// Handle tuple assignment: `audio[0, 10:50] = values` or `audio[:, 100:200] = array`
    fn setitem_tuple(
        &mut self,
        tuple: &Bound<pyo3::types::PyTuple>,
        value: &Bound<PyAny>,
    ) -> PyResult<()> {
        if tuple.len() != 2 {
            return Err(PyErr::new::<PyValueError, _>(
                "Tuple indexing must have exactly 2 elements: (channel_index, sample_index/slice)",
            ));
        }

        let channel_item = tuple.get_item(0)?;
        let sample_item = tuple.get_item(1)?;

        // Parse channel specification
        let channel_range = self.parse_channel_spec(&channel_item)?;

        // Parse sample specification
        let sample_range = self.parse_sample_spec(&sample_item)?;

        // Apply the assignment based on the specifications
        match (channel_range, sample_range) {
            (Some((ch_start, ch_end)), Some((s_start, s_end))) => {
                // Both channels and samples specified: audio[0, 10:50] = values
                let range_length = s_end - s_start;
                let channel_count = ch_end - ch_start;

                let values =
                    self.extract_values_array(value, Some(range_length * channel_count))?;
                self.set_sample_range_specific_channels(ch_start, ch_end, s_start, s_end, &values)
            }
            (None, Some((s_start, s_end))) => {
                // All channels, specific sample range: audio[:, 100:200] = array
                let range_length = s_end - s_start;
                let values =
                    self.extract_values_array(value, Some(range_length * self.channels()))?;
                self.set_sample_range_all_channels(s_start, s_end, &values)
            }
            (Some((ch_start, ch_end)), None) => {
                // Specific channels, all samples: audio[0:2, :] = channel_data
                let channel_count = ch_end - ch_start;
                let values =
                    self.extract_values_array(value, Some(self.length() * channel_count))?;
                self.set_all_samples_specific_channels(ch_start, ch_end, &values)
            }
            (None, None) => {
                // This would be [:, :] - assign to entire array
                let total_elements = self.length() * self.channels();
                let values = self.extract_values_array(value, Some(total_elements))?;
                self.set_all_samples_all_channels(&values)
            }
        }
    }

    /// Extract scalar value from Python object
    fn extract_scalar_value(&self, value: &Bound<PyAny>) -> PyResult<f64> {
        if let Ok(val) = value.extract::<f64>() {
            return Ok(val);
        }
        if let Ok(val) = value.extract::<f32>() {
            return Ok(val as f64);
        }
        if let Ok(val) = value.extract::<i64>() {
            return Ok(val as f64);
        }
        if let Ok(val) = value.extract::<i32>() {
            return Ok(val as f64);
        }
        if let Ok(val) = value.extract::<i16>() {
            return Ok(val as f64);
        }

        Err(PyErr::new::<PyTypeError, _>(
            "Value must be a numeric scalar (int or float)",
        ))
    }

    /// Extract array of values from Python object  
    fn extract_values_array(
        &self,
        value: &Bound<PyAny>,
        expected_len: Option<usize>,
    ) -> PyResult<Vec<f64>> {
        // Try numpy array first
        if let Ok(array) = value.downcast::<numpy::PyArrayDyn<f64>>() {
            let values: Vec<f64> = array.readonly().as_array().iter().copied().collect();

            if let Some(expected) = expected_len {
                if values.len() != expected {
                    return Err(PyErr::new::<PyValueError, _>(format!(
                        "Shape mismatch: expected {} values, got {}",
                        expected,
                        values.len()
                    )));
                }
            }

            return Ok(values);
        }

        // Try list/sequence
        if let Ok(seq) = value.downcast::<pyo3::types::PySequence>() {
            let mut values = Vec::new();
            let length = seq.len()?;
            for i in 0..length {
                let item = seq.get_item(i)?;
                let val = self.extract_scalar_value(&item)?;
                values.push(val);
            }

            if let Some(expected) = expected_len {
                if values.len() != expected {
                    return Err(PyErr::new::<PyValueError, _>(format!(
                        "Shape mismatch: expected {} values, got {}",
                        expected,
                        values.len()
                    )));
                }
            }

            return Ok(values);
        }

        // Try single scalar (for broadcasting)
        if let Ok(val) = self.extract_scalar_value(value) {
            let count = expected_len.unwrap_or(1);
            return Ok(vec![val; count]);
        }

        Err(PyErr::new::<PyTypeError, _>(
            "Value must be a scalar, list, or numpy array",
        ))
    }

    /// Set a single sample at the specified channel and index
    fn set_single_sample(&mut self, channel: usize, sample: usize, value: f64) -> PyResult<()> {
        self.with_inner_mut(|inner| match &mut inner.data {
            AudioData::Mono(arr) => {
                arr[sample] = value;
                Ok(())
            }
            AudioData::MultiChannel(arr) => {
                arr[(channel, sample)] = value;
                Ok(())
            }
        })
        .map_err(utils::map_error)
    }

    /// Set a range of samples for all channels
    fn set_sample_range_all_channels(
        &mut self,
        start: usize,
        stop: usize,
        values: &[f64],
    ) -> PyResult<()> {
        let channels = self.channels();
        let range_length = stop - start;

        if values.len() != range_length * channels {
            return Err(PyErr::new::<PyValueError, _>(format!(
                "Shape mismatch: expected {} values ({} samples × {} channels), got {}",
                range_length * channels,
                range_length,
                channels,
                values.len()
            )));
        }

        self.with_inner_mut(|inner| match &mut inner.data {
            AudioData::Mono(arr) => {
                for (i, &val) in values.iter().enumerate() {
                    arr[start + i] = val;
                }
                Ok(())
            }
            AudioData::MultiChannel(arr) => {
                for sample_idx in 0..range_length {
                    for ch in 0..channels {
                        let value_idx = sample_idx * channels + ch;
                        arr[(ch, start + sample_idx)] = values[value_idx];
                    }
                }
                Ok(())
            }
        })
        .map_err(utils::map_error)
    }

    /// Set a range of samples for specific channels
    fn set_sample_range_specific_channels(
        &mut self,
        ch_start: usize,
        ch_end: usize,
        s_start: usize,
        s_end: usize,
        values: &[f64],
    ) -> PyResult<()> {
        let channel_count = ch_end - ch_start;
        let sample_count = s_end - s_start;

        if values.len() != channel_count * sample_count {
            return Err(PyErr::new::<PyValueError, _>(format!(
                "Shape mismatch: expected {} values ({} channels × {} samples), got {}",
                channel_count * sample_count,
                channel_count,
                sample_count,
                values.len()
            )));
        }

        self.with_inner_mut(|inner| {
            match &mut inner.data {
                AudioData::Mono(_) => {
                    if ch_start != 0 || ch_end != 1 {
                        return Err(AudioSampleError::InvalidParameter(
                            "Cannot specify channel range for mono audio".to_string(),
                        ));
                    }
                    // Fall through to single channel case
                }
                AudioData::MultiChannel(_) => {}
            }

            match &mut inner.data {
                AudioData::Mono(arr) => {
                    for (i, &val) in values.iter().enumerate() {
                        arr[s_start + i] = val;
                    }
                    Ok(())
                }
                AudioData::MultiChannel(arr) => {
                    for sample_idx in 0..sample_count {
                        for (ch_idx, ch) in (ch_start..ch_end).enumerate() {
                            let value_idx = sample_idx * channel_count + ch_idx;
                            arr[(ch, s_start + sample_idx)] = values[value_idx];
                        }
                    }
                    Ok(())
                }
            }
        })
        .map_err(utils::map_error)
    }

    /// Set all samples for specific channels
    fn set_all_samples_specific_channels(
        &mut self,
        ch_start: usize,
        ch_end: usize,
        values: &[f64],
    ) -> PyResult<()> {
        let channel_count = ch_end - ch_start;
        let sample_count = self.length();

        if values.len() != channel_count * sample_count {
            return Err(PyErr::new::<PyValueError, _>(format!(
                "Shape mismatch: expected {} values ({} channels × {} samples), got {}",
                channel_count * sample_count,
                channel_count,
                sample_count,
                values.len()
            )));
        }

        self.with_inner_mut(|inner| {
            match &mut inner.data {
                AudioData::Mono(_) => {
                    if ch_start != 0 || ch_end != 1 {
                        return Err(AudioSampleError::InvalidParameter(
                            "Cannot specify channel range for mono audio".to_string(),
                        ));
                    }
                }
                AudioData::MultiChannel(_) => {}
            }

            match &mut inner.data {
                AudioData::Mono(arr) => {
                    for (i, &val) in values.iter().enumerate() {
                        arr[i] = val;
                    }
                    Ok(())
                }
                AudioData::MultiChannel(arr) => {
                    for sample_idx in 0..sample_count {
                        for (ch_idx, ch) in (ch_start..ch_end).enumerate() {
                            let value_idx = sample_idx * channel_count + ch_idx;
                            arr[(ch, sample_idx)] = values[value_idx];
                        }
                    }
                    Ok(())
                }
            }
        })
        .map_err(utils::map_error)
    }

    /// Set all samples for all channels
    fn set_all_samples_all_channels(&mut self, values: &[f64]) -> PyResult<()> {
        let channels = self.channels();
        let length = self.length();

        if values.len() != channels * length {
            return Err(PyErr::new::<PyValueError, _>(format!(
                "Shape mismatch: expected {} values ({} channels × {} samples), got {}",
                channels * length,
                channels,
                length,
                values.len()
            )));
        }

        self.with_inner_mut(|inner| match &mut inner.data {
            AudioData::Mono(arr) => {
                for (i, &val) in values.iter().enumerate() {
                    arr[i] = val;
                }
                Ok(())
            }
            AudioData::MultiChannel(arr) => {
                for sample_idx in 0..length {
                    for ch in 0..channels {
                        let value_idx = sample_idx * channels + ch;
                        arr[(ch, sample_idx)] = values[value_idx];
                    }
                }
                Ok(())
            }
        })
        .map_err(utils::map_error)
    }

    // ========================
    // String Representation - Enhanced display with sample values
    // ========================

    /// Generate string representation showing both metadata and sample values.
    ///
    /// This provides NumPy-style display that shows actual audio data alongside
    /// the essential metadata like shape, sample rate, and data type.
    ///
    /// # Arguments
    /// * `detailed` - If true, shows more verbose metadata (repr-style).
    ///               If false, shows compact metadata (str-style).
    ///

    /// Get the string representation of the sample type.
    /// Enhanced representation method that shows both metadata and sample values
    fn repr_impl(&self, detailed: bool) -> String {
        let channels = self.channels();
        let length = self.length();
        let sample_rate = self.sample_rate();
        let duration = length as f64 / sample_rate as f64;
        let dtype = self.sample_type_string();

        // Format the metadata
        let metadata = if detailed {
            format!(
                "AudioSamples(channels={}, length={}, sample_rate={}, duration={:.3}s, dtype={})",
                channels, length, sample_rate, duration, dtype
            )
        } else {
            format!(
                "AudioSamples({} ch, {:.3}s @ {}Hz, {})",
                channels, duration, sample_rate, dtype
            )
        };

        // Get the sample values formatted as a string
        let sample_values = self.format_sample_values();

        // Combine metadata and values
        format!("{}\n{}", metadata, sample_values)
    }

    /// Format sample values for display, similar to NumPy array representation
    fn format_sample_values(&self) -> String {
        let channels = self.channels();
        let length = self.length();

        if length == 0 {
            return "[]".to_string();
        }

        match channels {
            1 => {
                // Mono audio - format as 1D array
                self.format_mono_values()
            }
            _ => {
                // Multi-channel audio - format as 2D array
                self.format_multi_channel_values()
            }
        }
    }

    /// Format mono audio as a 1D array with intelligent truncation
    fn format_mono_values(&self) -> String {
        const MAX_DISPLAY_SAMPLES: usize = 8;
        let length = self.length();

        if length <= MAX_DISPLAY_SAMPLES {
            // Show all values
            let values: Vec<String> = (0..length)
                .map(|i| self.get_sample_as_string(0, i))
                .collect();
            format!("[{}]", values.join(", "))
        } else {
            // Show first few, ellipsis, then last few
            let show_each_side = MAX_DISPLAY_SAMPLES / 2;
            let mut values = Vec::new();

            // First few samples
            for i in 0..show_each_side {
                values.push(self.get_sample_as_string(0, i));
            }

            values.push("...".to_string());

            // Last few samples
            for i in (length - show_each_side)..length {
                values.push(self.get_sample_as_string(0, i));
            }

            format!("[{}]", values.join(", "))
        }
    }

    /// Format multi-channel audio as a 2D array with intelligent truncation
    fn format_multi_channel_values(&self) -> String {
        const MAX_DISPLAY_CHANNELS: usize = 4;
        const MAX_DISPLAY_SAMPLES: usize = 6;

        let channels = self.channels();
        let length = self.length();

        let mut channel_strings = Vec::new();

        // Determine how many channels to show
        let show_all_channels = channels <= MAX_DISPLAY_CHANNELS;
        let channels_to_show = if show_all_channels {
            channels
        } else {
            MAX_DISPLAY_CHANNELS / 2
        };

        // Format first few channels
        for ch in 0..channels_to_show.min(channels) {
            channel_strings.push(self.format_channel_values(ch, length, MAX_DISPLAY_SAMPLES));
        }

        // Add ellipsis for channels if needed
        if !show_all_channels && channels > MAX_DISPLAY_CHANNELS {
            channel_strings.push(" ...".to_string());

            // Show last few channels
            let start_ch = channels - (MAX_DISPLAY_CHANNELS / 2);
            for ch in start_ch..channels {
                channel_strings.push(self.format_channel_values(ch, length, MAX_DISPLAY_SAMPLES));
            }
        }

        format!("[{}]", channel_strings.join(",\n "))
    }

    /// Format a single channel's values with sample truncation
    fn format_channel_values(&self, channel: usize, length: usize, max_samples: usize) -> String {
        if length <= max_samples {
            // Show all values for this channel
            let values: Vec<String> = (0..length)
                .map(|i| self.get_sample_as_string(channel, i))
                .collect();
            format!("[{}]", values.join(", "))
        } else {
            // Show first few, ellipsis, then last few
            let show_each_side = max_samples / 2;
            let mut values = Vec::new();

            // First few samples
            for i in 0..show_each_side {
                values.push(self.get_sample_as_string(channel, i));
            }

            values.push("...".to_string());

            // Last few samples
            for i in (length - show_each_side)..length {
                values.push(self.get_sample_as_string(channel, i));
            }

            format!("[{}]", values.join(", "))
        }
    }

    /// Get a single sample value as a formatted string
    fn get_sample_as_string(&self, channel: usize, sample: usize) -> String {
        match &self.data {
            AudioSamplesData::F64(audio) => {
                let value = match &audio.data {
                    AudioData::Mono(arr) => arr[sample],
                    AudioData::MultiChannel(arr) => arr[(channel, sample)],
                };
                format!("{:8.6}", value)
            }
            AudioSamplesData::F32(audio) => {
                let value = match &audio.data {
                    AudioData::Mono(arr) => arr[sample],
                    AudioData::MultiChannel(arr) => arr[(channel, sample)],
                };
                format!("{:8.6}", value)
            }
            AudioSamplesData::I32(audio) => {
                let value = match &audio.data {
                    AudioData::Mono(arr) => arr[sample],
                    AudioData::MultiChannel(arr) => arr[(channel, sample)],
                };
                format!("{:10}", value)
            }
            AudioSamplesData::I16(audio) => {
                let value = match &audio.data {
                    AudioData::Mono(arr) => arr[sample],
                    AudioData::MultiChannel(arr) => arr[(channel, sample)],
                };
                format!("{:6}", value)
            }
            AudioSamplesData::I24(audio) => {
                let value = match &audio.data {
                    AudioData::Mono(arr) => arr[sample],
                    AudioData::MultiChannel(arr) => arr[(channel, sample)],
                };
                format!("{:8}", value.to_i32())
            }
        }
    }

    fn sample_type_string(&self) -> &'static str {
        match &self.data {
            AudioSamplesData::F64(_) => "float64",
            AudioSamplesData::F32(_) => "float32",
            AudioSamplesData::I32(_) => "int32",
            AudioSamplesData::I16(_) => "int16",
            AudioSamplesData::I24(_) => "int24",
        }
    }
}

#[pyfunction(name = "to_tensor", signature = (audio, *, device=None, copy=false))]
pub fn to_tensor(
    py: Python,
    audio: &PyAudioSamples,
    device: Option<String>,
    copy: Option<bool>,
) -> PyResult<Py<PyAny>> {
    let numpy_array = audio.numpy(py, None, copy.unwrap_or(false))?;
    let torch = py.import("torch")?;

    let device = device.unwrap_or_else(|| "cpu".to_string());

    if device != "cpu" && !device.starts_with("cuda") {
        return Err(PyErr::new::<PyValueError, _>(
            "Invalid device specified. Use 'cpu' or 'cuda[:N]' where N is the GPU index.",
        ));
    }

    let kwargs = PyDict::new(py);
    kwargs.set_item("device", device.clone())?;
    let tensor = torch.call_method("from_numpy", (numpy_array,), Some(&kwargs))?;
    Ok(tensor.into())
}

#[pyfunction(name = "to_gpu", signature = (audio, *, device = None, copy=false))]
pub fn to_gpu(
    py: Python,
    audio: &PyAudioSamples,
    device: Option<String>,
    copy: Option<bool>,
) -> PyResult<Py<PyAny>> {
    to_tensor(py, audio, Some(device.unwrap_or("cuda".to_string())), copy)
}

#[pyfunction(name = "from_tensor", signature = (tensor, sample_rate, *, copy=false))]
pub fn from_tensor(
    tensor: &Bound<PyAny>,
    sample_rate: u32,
    copy: bool,
) -> PyResult<PyAudioSamples> {
    let array = tensor.call_method0("cpu")?.call_method0("numpy")?;
    from_numpy(&array, sample_rate, copy)
}
