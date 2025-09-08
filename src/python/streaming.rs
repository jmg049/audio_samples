//! Simplified Python bindings for audio streaming functionality.

use crate::{
    python::{AudioSamplesData, PyAudioSamples},
    streaming::{
        sources::GeneratorSource,
        traits::{AudioFormatInfo, AudioSource, StreamConfig},
    },
};
use pyo3::prelude::*;
use std::time::Duration;

/// Python wrapper for AudioFormatInfo
#[pyclass(name = "AudioFormatInfo", module = "audio_samples.streaming")]
#[derive(Clone)]
pub struct PyAudioFormatInfo {
    inner: AudioFormatInfo,
}

#[pymethods]
impl PyAudioFormatInfo {
    /// Create format info for f32 samples
    #[staticmethod]
    fn f32(sample_rate: usize, channels: usize) -> Self {
        Self {
            inner: AudioFormatInfo::f32(sample_rate, channels),
        }
    }

    /// Create format info for i16 samples
    #[staticmethod]
    fn i16(sample_rate: usize, channels: usize) -> Self {
        Self {
            inner: AudioFormatInfo::i16(sample_rate, channels),
        }
    }

    /// Sample rate in Hz
    #[getter]
    fn sample_rate(&self) -> usize {
        self.inner.sample_rate
    }

    /// Number of channels
    #[getter]
    fn channels(&self) -> usize {
        self.inner.channels
    }

    /// Sample format string
    #[getter]
    fn sample_format(&self) -> String {
        self.inner.sample_format.clone()
    }

    /// Bits per sample
    #[getter]
    fn bits_per_sample(&self) -> u8 {
        self.inner.bits_per_sample
    }

    /// Whether samples are signed
    #[getter]
    fn is_signed(&self) -> bool {
        self.inner.is_signed
    }

    /// Whether samples are floating point
    #[getter]
    fn is_float(&self) -> bool {
        self.inner.is_float
    }

    /// Check if this format is compatible with another
    fn is_compatible(&self, other: &PyAudioFormatInfo) -> bool {
        self.inner.is_compatible(&other.inner)
    }

    fn __repr__(&self) -> String {
        format!(
            "AudioFormatInfo(sample_rate={}, channels={}, format='{}', bits={})",
            self.inner.sample_rate,
            self.inner.channels,
            self.inner.sample_format,
            self.inner.bits_per_sample
        )
    }
}

/// Python wrapper for StreamConfig
#[pyclass(name = "StreamConfig", module = "audio_samples.streaming")]
#[derive(Clone)]
pub struct PyStreamConfig {
    inner: StreamConfig,
}

#[pymethods]
impl PyStreamConfig {
    /// Create a new stream configuration
    #[new]
    #[pyo3(signature = (*, buffer_size=1024, max_buffer_size=8192, min_buffer_level=0.25, read_timeout_ms=100, auto_recovery=true, max_recovery_attempts=3))]
    fn new(
        buffer_size: usize,
        max_buffer_size: usize,
        min_buffer_level: f64,
        read_timeout_ms: u64,
        auto_recovery: bool,
        max_recovery_attempts: usize,
    ) -> Self {
        Self {
            inner: StreamConfig {
                buffer_size,
                max_buffer_size,
                min_buffer_level,
                read_timeout: Duration::from_millis(read_timeout_ms),
                auto_recovery,
                max_recovery_attempts,
                preferred_format: None,
            },
        }
    }

    /// Create configuration optimized for low-latency applications
    #[staticmethod]
    fn low_latency() -> Self {
        Self {
            inner: StreamConfig::low_latency(),
        }
    }

    /// Create configuration optimized for high-quality streaming
    #[staticmethod]
    fn high_quality() -> Self {
        Self {
            inner: StreamConfig::high_quality(),
        }
    }

    /// Create configuration for network streaming with error tolerance
    #[staticmethod]
    fn network_streaming() -> Self {
        Self {
            inner: StreamConfig::network_streaming(),
        }
    }

    /// Target buffer size in samples per channel
    #[getter]
    fn buffer_size(&self) -> usize {
        self.inner.buffer_size
    }

    #[setter]
    fn set_buffer_size(&mut self, size: usize) {
        self.inner.buffer_size = size;
    }

    /// Whether to automatically recover from errors
    #[getter]
    fn auto_recovery(&self) -> bool {
        self.inner.auto_recovery
    }

    #[setter]
    fn set_auto_recovery(&mut self, enabled: bool) {
        self.inner.auto_recovery = enabled;
    }

    fn __repr__(&self) -> String {
        format!(
            "StreamConfig(buffer_size={}, auto_recovery={})",
            self.inner.buffer_size, self.inner.auto_recovery
        )
    }
}

/// Python wrapper for GeneratorSource
#[pyclass(name = "GeneratorSource", module = "audio_samples.streaming")]
pub struct PyGeneratorSource {
    inner_f32: Option<GeneratorSource<f32>>,
    sample_format: String,
}

#[pymethods]
impl PyGeneratorSource {
    /// Create a sine wave generator
    #[staticmethod]
    fn sine(frequency: f64, sample_rate: usize, channels: usize) -> PyResult<Self> {
        Ok(Self {
            inner_f32: Some(GeneratorSource::<f32>::sine(
                frequency,
                sample_rate,
                channels,
            )),
            sample_format: "f32".to_string(),
        })
    }

    /// Create a white noise generator
    #[staticmethod]
    fn white_noise(sample_rate: usize, channels: usize) -> PyResult<Self> {
        Ok(Self {
            inner_f32: Some(GeneratorSource::<f32>::white_noise(sample_rate, channels)),
            sample_format: "f32".to_string(),
        })
    }

    /// Create a silence generator
    #[staticmethod]
    fn silence(sample_rate: usize, channels: usize) -> PyResult<Self> {
        Ok(Self {
            inner_f32: Some(GeneratorSource::<f32>::silence(sample_rate, channels)),
            sample_format: "f32".to_string(),
        })
    }

    /// Create a chirp generator
    #[staticmethod]
    fn chirp(
        start_freq: f64,
        end_freq: f64,
        duration_sec: f64,
        sample_rate: usize,
        channels: usize,
    ) -> PyResult<Self> {
        let duration = Duration::from_secs_f64(duration_sec);
        Ok(Self {
            inner_f32: Some(GeneratorSource::<f32>::chirp(
                start_freq,
                end_freq,
                duration,
                sample_rate,
                channels,
            )),
            sample_format: "f32".to_string(),
        })
    }

    /// Generate the next chunk of audio data (simplified sync version)
    fn next_chunk(&mut self) -> PyResult<Option<PyAudioSamples>> {
        if let Some(ref mut source) = self.inner_f32 {
            // For now, we'll create a simple runtime for the async call
            // In production, this would use pyo3_asyncio for proper async support
            use tokio::runtime::Handle;

            let chunk_result = if Handle::try_current().is_ok() {
                // Already in async context
                futures::executor::block_on(source.next_chunk())
            } else {
                // Need to create runtime
                let rt = tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build()
                    .map_err(|e| {
                        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                            "Failed to create tokio runtime: {}",
                            e
                        ))
                    })?;
                rt.block_on(source.next_chunk())
            };

            match chunk_result {
                Ok(Some(chunk)) => {
                    let py_chunk = PyAudioSamples::from_data(AudioSamplesData::F32(chunk));
                    Ok(Some(py_chunk))
                }
                Ok(None) => Ok(None),
                Err(e) => Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                    "Stream error: {}",
                    e
                ))),
            }
        } else {
            Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                "Generator source not properly initialized",
            ))
        }
    }

    /// Get format information for this generator
    fn format_info(&self) -> PyResult<PyAudioFormatInfo> {
        if let Some(ref source) = self.inner_f32 {
            Ok(PyAudioFormatInfo {
                inner: source.format_info(),
            })
        } else {
            Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                "Generator source not properly initialized",
            ))
        }
    }

    /// Check if the source is still active
    fn is_active(&self) -> bool {
        if let Some(ref source) = self.inner_f32 {
            source.is_active()
        } else {
            false
        }
    }

    fn __repr__(&self) -> String {
        format!("GeneratorSource(format='{}')", self.sample_format)
    }
}

/// Factory functions for convenience
#[pyfunction(name = "sine_generator")]
pub fn sine_generator(
    frequency: f64,
    sample_rate: usize,
    channels: usize,
) -> PyResult<PyGeneratorSource> {
    PyGeneratorSource::sine(frequency, sample_rate, channels)
}

#[pyfunction(name = "noise_generator")]
pub fn noise_generator(sample_rate: usize, channels: usize) -> PyResult<PyGeneratorSource> {
    PyGeneratorSource::white_noise(sample_rate, channels)
}

#[pyfunction(name = "silence_generator")]
pub fn silence_generator(sample_rate: usize, channels: usize) -> PyResult<PyGeneratorSource> {
    PyGeneratorSource::silence(sample_rate, channels)
}

#[pyfunction(name = "chirp_generator")]
pub fn chirp_generator(
    start_freq: f64,
    end_freq: f64,
    duration_sec: f64,
    sample_rate: usize,
    channels: usize,
) -> PyResult<PyGeneratorSource> {
    PyGeneratorSource::chirp(start_freq, end_freq, duration_sec, sample_rate, channels)
}

/// Register streaming module with Python
pub fn register_module(_py: Python, m: &Bound<PyModule>) -> PyResult<()> {
    // Configuration classes
    m.add_class::<PyAudioFormatInfo>()?;
    m.add_class::<PyStreamConfig>()?;

    // Source classes
    m.add_class::<PyGeneratorSource>()?;

    // Factory functions
    m.add_function(pyo3::wrap_pyfunction!(sine_generator, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(noise_generator, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(silence_generator, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(chirp_generator, m)?)?;

    Ok(())
}
