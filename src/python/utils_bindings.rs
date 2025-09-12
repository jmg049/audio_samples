//! Python bindings for utility functions.
//!
//! This module provides Python bindings for the utility functions in `src/utils/`,
//! including signal generation, audio detection, and comparison functions.
//! All functions support generic sample types and return PyAudioSamples objects.

use crate::operations::traits::AudioTypeConversion;
use crate::python::{AudioSamplesData, PyAudioSamples, utils as py_utils};
use crate::utils;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use pyo3::wrap_pyfunction;
use numpy;
use ndarray;

// =====================
// Signal Generation Functions
// =====================

/// Generate a sine wave with the specified parameters.
///
/// # Arguments
/// * `frequency` - Frequency of the sine wave in Hz
/// * `duration` - Duration of the signal in seconds
/// * `sample_rate` - Sample rate in Hz
/// * `amplitude` - Amplitude of the sine wave (0.0 to 1.0, default: 1.0)
/// * `dtype` - Target sample type ('f64', 'f32', 'i32', 'i16', default: 'f64')
///
/// # Examples
/// ```python
/// import audio_samples as aus
///
/// # Generate 440Hz sine wave for 1 second at 44.1kHz
/// sine = aus.sine_wave(440.0, 1.0, 44100)
/// print(f"Generated sine wave: {sine}")
///
/// # Generate with specific data type
/// sine_i16 = aus.sine_wave(440.0, 1.0, 44100, amplitude=0.5, dtype='i16')
/// ```
#[pyfunction(name = "sine_wave")]
#[pyo3(signature = (frequency, duration, sample_rate, *, amplitude=1.0, dtype="f64"))]
fn py_sine_wave(
    frequency: f64,
    duration: f64,
    sample_rate: u32,
    amplitude: f64,
    dtype: &str,
) -> PyResult<PyAudioSamples> {
    py_utils::validate_string_param("dtype", dtype, &["f64", "f32", "i32", "i16"])?;

    let data = match dtype {
        "f64" => {
            let audio =
                utils::generation::sine_wave::<f64>(frequency, duration, sample_rate, amplitude)
                    .map_err(py_utils::map_error)?;
            AudioSamplesData::F64(audio)
        }
        "f32" => {
            let audio =
                utils::generation::sine_wave::<f32>(frequency, duration, sample_rate, amplitude)
                    .map_err(py_utils::map_error)?;
            AudioSamplesData::F32(audio)
        }
        "i32" => {
            let audio =
                utils::generation::sine_wave::<i32>(frequency, duration, sample_rate, amplitude)
                    .map_err(py_utils::map_error)?;
            AudioSamplesData::I32(audio)
        }
        "i16" => {
            let audio =
                utils::generation::sine_wave::<i16>(frequency, duration, sample_rate, amplitude)
                    .map_err(py_utils::map_error)?;
            AudioSamplesData::I16(audio)
        }
        _ => unreachable!(),
    };

    Ok(PyAudioSamples::from_data(data))
}

/// Generate a cosine wave with the specified parameters.
///
/// # Arguments
/// * `frequency` - Frequency of the cosine wave in Hz
/// * `duration` - Duration of the signal in seconds  
/// * `sample_rate` - Sample rate in Hz
/// * `amplitude` - Amplitude of the cosine wave (0.0 to 1.0, default: 1.0)
/// * `dtype` - Target sample type ('f64', 'f32', 'i32', 'i16', default: 'f64')
#[pyfunction(name = "cosine_wave")]
#[pyo3(signature = (frequency, duration, sample_rate, *, amplitude=1.0, dtype="f64"))]
fn py_cosine_wave(
    frequency: f64,
    duration: f64,
    sample_rate: u32,
    amplitude: f64,
    dtype: &str,
) -> PyResult<PyAudioSamples> {
    py_utils::validate_string_param("dtype", dtype, &["f64", "f32", "i32", "i16"])?;

    let data = match dtype {
        "f64" => {
            let audio =
                utils::generation::cosine_wave::<f64>(frequency, duration, sample_rate, amplitude)
                    .map_err(py_utils::map_error)?;
            AudioSamplesData::F64(audio)
        }
        "f32" => {
            let audio =
                utils::generation::cosine_wave::<f32>(frequency, duration, sample_rate, amplitude)
                    .map_err(py_utils::map_error)?;
            AudioSamplesData::F32(audio)
        }
        "i32" => {
            let audio =
                utils::generation::cosine_wave::<i32>(frequency, duration, sample_rate, amplitude)
                    .map_err(py_utils::map_error)?;
            AudioSamplesData::I32(audio)
        }
        "i16" => {
            let audio =
                utils::generation::cosine_wave::<i16>(frequency, duration, sample_rate, amplitude)
                    .map_err(py_utils::map_error)?;
            AudioSamplesData::I16(audio)
        }
        _ => unreachable!(),
    };

    Ok(PyAudioSamples::from_data(data))
}

/// Generate white noise.
///
/// # Arguments
/// * `duration` - Duration of the signal in seconds
/// * `sample_rate` - Sample rate in Hz
/// * `amplitude` - Amplitude scaling factor (default: 1.0)
/// * `dtype` - Target sample type ('f64', 'f32', 'i32', 'i16', default: 'f64')
#[pyfunction(name = "white_noise")]
#[pyo3(signature = (duration, sample_rate, *, amplitude=1.0, dtype="f64"))]
fn py_white_noise(
    duration: f64,
    sample_rate: u32,
    amplitude: f64,
    dtype: &str,
) -> PyResult<PyAudioSamples> {
    py_utils::validate_string_param("dtype", dtype, &["f64", "f32", "i32", "i16"])?;

    let data = match dtype {
        "f64" => {
            let audio = utils::generation::white_noise::<f64>(duration, sample_rate, amplitude)
                .map_err(py_utils::map_error)?;
            AudioSamplesData::F64(audio)
        }
        "f32" => {
            let audio = utils::generation::white_noise::<f32>(duration, sample_rate, amplitude)
                .map_err(py_utils::map_error)?;
            AudioSamplesData::F32(audio)
        }
        "i32" => {
            let audio = utils::generation::white_noise::<i32>(duration, sample_rate, amplitude)
                .map_err(py_utils::map_error)?;
            AudioSamplesData::I32(audio)
        }
        "i16" => {
            let audio = utils::generation::white_noise::<i16>(duration, sample_rate, amplitude)
                .map_err(py_utils::map_error)?;
            AudioSamplesData::I16(audio)
        }
        _ => unreachable!(),
    };

    Ok(PyAudioSamples::from_data(data))
}

/// Generate pink noise (1/f noise).
///
/// # Arguments
/// * `duration` - Duration of the signal in seconds
/// * `sample_rate` - Sample rate in Hz
/// * `amplitude` - Amplitude scaling factor (default: 1.0)
/// * `dtype` - Target sample type ('f64', 'f32', 'i32', 'i16', default: 'f64')
#[pyfunction(name = "pink_noise")]
#[pyo3(signature = (duration, sample_rate, *, amplitude=1.0, dtype="f64"))]
fn py_pink_noise(
    duration: f64,
    sample_rate: u32,
    amplitude: f64,
    dtype: &str,
) -> PyResult<PyAudioSamples> {
    py_utils::validate_string_param("dtype", dtype, &["f64", "f32", "i32", "i16"])?;

    let data = match dtype {
        "f64" => {
            let audio = utils::generation::pink_noise::<f64>(duration, sample_rate, amplitude)
                .map_err(py_utils::map_error)?;
            AudioSamplesData::F64(audio)
        }
        "f32" => {
            let audio = utils::generation::pink_noise::<f32>(duration, sample_rate, amplitude)
                .map_err(py_utils::map_error)?;
            AudioSamplesData::F32(audio)
        }
        "i32" => {
            let audio = utils::generation::pink_noise::<i32>(duration, sample_rate, amplitude)
                .map_err(py_utils::map_error)?;
            AudioSamplesData::I32(audio)
        }
        "i16" => {
            let audio = utils::generation::pink_noise::<i16>(duration, sample_rate, amplitude)
                .map_err(py_utils::map_error)?;
            AudioSamplesData::I16(audio)
        }
        _ => unreachable!(),
    };

    Ok(PyAudioSamples::from_data(data))
}

/// Generate a square wave.
///
/// # Arguments
/// * `frequency` - Frequency of the square wave in Hz
/// * `duration` - Duration of the signal in seconds
/// * `sample_rate` - Sample rate in Hz
/// * `amplitude` - Amplitude of the square wave (default: 1.0)
/// * `dtype` - Target sample type ('f64', 'f32', 'i32', 'i16', default: 'f64')
#[pyfunction(name = "square_wave")]
#[pyo3(signature = (frequency, duration, sample_rate, *, amplitude=1.0, dtype="f64"))]
fn py_square_wave(
    frequency: f64,
    duration: f64,
    sample_rate: u32,
    amplitude: f64,
    dtype: &str,
) -> PyResult<PyAudioSamples> {
    py_utils::validate_string_param("dtype", dtype, &["f64", "f32", "i32", "i16"])?;

    let data = match dtype {
        "f64" => {
            let audio =
                utils::generation::square_wave::<f64>(frequency, duration, sample_rate, amplitude)
                    .map_err(py_utils::map_error)?;
            AudioSamplesData::F64(audio)
        }
        "f32" => {
            let audio =
                utils::generation::square_wave::<f32>(frequency, duration, sample_rate, amplitude)
                    .map_err(py_utils::map_error)?;
            AudioSamplesData::F32(audio)
        }
        "i32" => {
            let audio =
                utils::generation::square_wave::<i32>(frequency, duration, sample_rate, amplitude)
                    .map_err(py_utils::map_error)?;
            AudioSamplesData::I32(audio)
        }
        "i16" => {
            let audio =
                utils::generation::square_wave::<i16>(frequency, duration, sample_rate, amplitude)
                    .map_err(py_utils::map_error)?;
            AudioSamplesData::I16(audio)
        }
        _ => unreachable!(),
    };

    Ok(PyAudioSamples::from_data(data))
}

/// Generate a sawtooth wave.
///
/// # Arguments
/// * `frequency` - Frequency of the sawtooth wave in Hz
/// * `duration` - Duration of the signal in seconds
/// * `sample_rate` - Sample rate in Hz
/// * `amplitude` - Amplitude of the sawtooth wave (default: 1.0)
/// * `dtype` - Target sample type ('f64', 'f32', 'i32', 'i16', default: 'f64')
#[pyfunction(name = "sawtooth_wave")]
#[pyo3(signature = (frequency, duration, sample_rate, *, amplitude=1.0, dtype="f64"))]
fn py_sawtooth_wave(
    frequency: f64,
    duration: f64,
    sample_rate: u32,
    amplitude: f64,
    dtype: &str,
) -> PyResult<PyAudioSamples> {
    py_utils::validate_string_param("dtype", dtype, &["f64", "f32", "i32", "i16"])?;

    let data = match dtype {
        "f64" => {
            let audio = utils::generation::sawtooth_wave::<f64>(
                frequency,
                duration,
                sample_rate,
                amplitude,
            )
            .map_err(py_utils::map_error)?;
            AudioSamplesData::F64(audio)
        }
        "f32" => {
            let audio = utils::generation::sawtooth_wave::<f32>(
                frequency,
                duration,
                sample_rate,
                amplitude,
            )
            .map_err(py_utils::map_error)?;
            AudioSamplesData::F32(audio)
        }
        "i32" => {
            let audio = utils::generation::sawtooth_wave::<i32>(
                frequency,
                duration,
                sample_rate,
                amplitude,
            )
            .map_err(py_utils::map_error)?;
            AudioSamplesData::I32(audio)
        }
        "i16" => {
            let audio = utils::generation::sawtooth_wave::<i16>(
                frequency,
                duration,
                sample_rate,
                amplitude,
            )
            .map_err(py_utils::map_error)?;
            AudioSamplesData::I16(audio)
        }
        _ => unreachable!(),
    };

    Ok(PyAudioSamples::from_data(data))
}

/// Generate a triangle wave.
///
/// # Arguments
/// * `frequency` - Frequency of the triangle wave in Hz
/// * `duration` - Duration of the signal in seconds
/// * `sample_rate` - Sample rate in Hz
/// * `amplitude` - Amplitude of the triangle wave (default: 1.0)
/// * `dtype` - Target sample type ('f64', 'f32', 'i32', 'i16', default: 'f64')
#[pyfunction(name = "triangle_wave")]
#[pyo3(signature = (frequency, duration, sample_rate, *, amplitude=1.0, dtype="f64"))]
fn py_triangle_wave(
    frequency: f64,
    duration: f64,
    sample_rate: u32,
    amplitude: f64,
    dtype: &str,
) -> PyResult<PyAudioSamples> {
    py_utils::validate_string_param("dtype", dtype, &["f64", "f32", "i32", "i16"])?;

    let data = match dtype {
        "f64" => {
            let audio = utils::generation::triangle_wave::<f64>(
                frequency,
                duration,
                sample_rate,
                amplitude,
            )
            .map_err(py_utils::map_error)?;
            AudioSamplesData::F64(audio)
        }
        "f32" => {
            let audio = utils::generation::triangle_wave::<f32>(
                frequency,
                duration,
                sample_rate,
                amplitude,
            )
            .map_err(py_utils::map_error)?;
            AudioSamplesData::F32(audio)
        }
        "i32" => {
            let audio = utils::generation::triangle_wave::<i32>(
                frequency,
                duration,
                sample_rate,
                amplitude,
            )
            .map_err(py_utils::map_error)?;
            AudioSamplesData::I32(audio)
        }
        "i16" => {
            let audio = utils::generation::triangle_wave::<i16>(
                frequency,
                duration,
                sample_rate,
                amplitude,
            )
            .map_err(py_utils::map_error)?;
            AudioSamplesData::I16(audio)
        }
        _ => unreachable!(),
    };

    Ok(PyAudioSamples::from_data(data))
}

/// Generate a chirp signal (frequency sweep).
///
/// # Arguments
/// * `f0` - Starting frequency in Hz
/// * `f1` - Ending frequency in Hz  
/// * `duration` - Duration of the signal in seconds
/// * `sample_rate` - Sample rate in Hz
/// * `amplitude` - Amplitude of the chirp (default: 1.0)
/// * `dtype` - Target sample type ('f64', 'f32', 'i32', 'i16', default: 'f64')
#[pyfunction(name = "chirp")]
#[pyo3(signature = (f0, f1, duration, sample_rate, *, amplitude=1.0, dtype="f64"))]
fn py_chirp(
    f0: f64,
    f1: f64,
    duration: f64,
    sample_rate: u32,
    amplitude: f64,
    dtype: &str,
) -> PyResult<PyAudioSamples> {
    py_utils::validate_string_param("dtype", dtype, &["f64", "f32", "i32", "i16"])?;

    let data = match dtype {
        "f64" => {
            let audio = utils::generation::chirp::<f64>(f0, f1, duration, sample_rate, amplitude)
                .map_err(py_utils::map_error)?;
            AudioSamplesData::F64(audio)
        }
        "f32" => {
            let audio = utils::generation::chirp::<f32>(f0, f1, duration, sample_rate, amplitude)
                .map_err(py_utils::map_error)?;
            AudioSamplesData::F32(audio)
        }
        "i32" => {
            let audio = utils::generation::chirp::<i32>(f0, f1, duration, sample_rate, amplitude)
                .map_err(py_utils::map_error)?;
            AudioSamplesData::I32(audio)
        }
        "i16" => {
            let audio = utils::generation::chirp::<i16>(f0, f1, duration, sample_rate, amplitude)
                .map_err(py_utils::map_error)?;
            AudioSamplesData::I16(audio)
        }
        _ => unreachable!(),
    };

    Ok(PyAudioSamples::from_data(data))
}

/// Generate an impulse signal.
///
/// # Arguments
/// * `duration` - Duration of the signal in seconds
/// * `sample_rate` - Sample rate in Hz
/// * `amplitude` - Amplitude of the impulse (default: 1.0)
/// * `delay` - Delay before the impulse in seconds (default: 0.0)
/// * `dtype` - Target sample type ('f64', 'f32', 'i32', 'i16', default: 'f64')
#[pyfunction(name = "impulse")]
#[pyo3(signature = (duration, sample_rate, *, amplitude=1.0, delay=0.0, dtype="f64"))]
fn py_impulse(
    duration: f64,
    sample_rate: u32,
    amplitude: f64,
    delay: f64,
    dtype: &str,
) -> PyResult<PyAudioSamples> {
    py_utils::validate_string_param("dtype", dtype, &["f64", "f32", "i32", "i16"])?;

    let data = match dtype {
        "f64" => {
            let audio = utils::generation::impulse::<f64>(duration, sample_rate, amplitude, delay)
                .map_err(py_utils::map_error)?;
            AudioSamplesData::F64(audio)
        }
        "f32" => {
            let audio = utils::generation::impulse::<f32>(duration, sample_rate, amplitude, delay)
                .map_err(py_utils::map_error)?;
            AudioSamplesData::F32(audio)
        }
        "i32" => {
            let audio = utils::generation::impulse::<i32>(duration, sample_rate, amplitude, delay)
                .map_err(py_utils::map_error)?;
            AudioSamplesData::I32(audio)
        }
        "i16" => {
            let audio = utils::generation::impulse::<i16>(duration, sample_rate, amplitude, delay)
                .map_err(py_utils::map_error)?;
            AudioSamplesData::I16(audio)
        }
        _ => unreachable!(),
    };

    Ok(PyAudioSamples::from_data(data))
}

/// Generate silence.
///
/// # Arguments
/// * `duration` - Duration of the silence in seconds
/// * `sample_rate` - Sample rate in Hz
/// * `dtype` - Target sample type ('f64', 'f32', 'i32', 'i16', default: 'f64')
#[pyfunction(name = "silence")]
#[pyo3(signature = (duration, sample_rate, *, dtype="f64"))]
fn py_silence(duration: f64, sample_rate: u32, dtype: &str) -> PyResult<PyAudioSamples> {
    py_utils::validate_string_param("dtype", dtype, &["f64", "f32", "i32", "i16"])?;

    let data = match dtype {
        "f64" => {
            let audio = utils::generation::silence::<f64>(duration, sample_rate)
                .map_err(py_utils::map_error)?;
            AudioSamplesData::F64(audio)
        }
        "f32" => {
            let audio = utils::generation::silence::<f32>(duration, sample_rate)
                .map_err(py_utils::map_error)?;
            AudioSamplesData::F32(audio)
        }
        "i32" => {
            let audio = utils::generation::silence::<i32>(duration, sample_rate)
                .map_err(py_utils::map_error)?;
            AudioSamplesData::I32(audio)
        }
        "i16" => {
            let audio = utils::generation::silence::<i16>(duration, sample_rate)
                .map_err(py_utils::map_error)?;
            AudioSamplesData::I16(audio)
        }
        _ => unreachable!(),
    };

    Ok(PyAudioSamples::from_data(data))
}

// =====================
// Audio Detection Functions
// =====================

/// Detect the sample rate of audio data by analyzing frequency content.
///
/// # Arguments
/// * `audio` - AudioSamples object to analyze
///
/// # Returns
/// Detected sample rate in Hz, or None if unable to determine
#[pyfunction(name = "detect_sample_rate")]
fn py_detect_sample_rate(audio: &PyAudioSamples) -> PyResult<Option<u32>> {
    match &audio.data {
        AudioSamplesData::F64(inner) => {
            utils::detection::detect_sample_rate(inner).map_err(py_utils::map_error)
        }
        AudioSamplesData::F32(inner) => {
            utils::detection::detect_sample_rate(inner).map_err(py_utils::map_error)
        }
        AudioSamplesData::I32(inner) => {
            utils::detection::detect_sample_rate(inner).map_err(py_utils::map_error)
        }
        AudioSamplesData::I16(inner) => {
            utils::detection::detect_sample_rate(inner).map_err(py_utils::map_error)
        }
        AudioSamplesData::I24(inner) => {
            utils::detection::detect_sample_rate(inner).map_err(py_utils::map_error)
        }
    }
}

/// Detect the fundamental frequency of audio data.
///
/// # Arguments
/// * `audio` - AudioSamples object to analyze
///
/// # Returns
/// Detected fundamental frequency in Hz, or None if unable to determine
#[pyfunction(name = "detect_fundamental_frequency")]
fn py_detect_fundamental_frequency(audio: &PyAudioSamples) -> PyResult<Option<f64>> {
    match &audio.data {
        AudioSamplesData::F64(inner) => {
            utils::detection::detect_fundamental_frequency(inner).map_err(py_utils::map_error)
        }
        AudioSamplesData::F32(inner) => {
            utils::detection::detect_fundamental_frequency(inner).map_err(py_utils::map_error)
        }
        AudioSamplesData::I32(inner) => {
            utils::detection::detect_fundamental_frequency(inner).map_err(py_utils::map_error)
        }
        AudioSamplesData::I16(inner) => {
            utils::detection::detect_fundamental_frequency(inner).map_err(py_utils::map_error)
        }
        AudioSamplesData::I24(inner) => {
            utils::detection::detect_fundamental_frequency(inner).map_err(py_utils::map_error)
        }
    }
}

/// Detect silence regions in audio data.
///
/// # Arguments
/// * `audio` - AudioSamples object to analyze
/// * `threshold` - Silence threshold as amplitude (default: 0.01)
///
/// # Returns
/// List of (start_time, end_time) tuples in seconds for each silence region
#[pyfunction(name = "detect_silence_regions")]
#[pyo3(signature = (audio, *, threshold=0.01))]
fn py_detect_silence_regions(
    py: Python,
    audio: &PyAudioSamples,
    threshold: f64,
) -> PyResult<Py<PyAny>> {
    let regions = match &audio.data {
        AudioSamplesData::F64(inner) => utils::detection::detect_silence_regions(inner, threshold)
            .map_err(py_utils::map_error)?,
        AudioSamplesData::F32(inner) => {
            utils::detection::detect_silence_regions(inner, threshold as f32)
                .map_err(py_utils::map_error)?
        }
        AudioSamplesData::I32(inner) => {
            utils::detection::detect_silence_regions(inner, (threshold * (i32::MAX as f64)) as i32)
                .map_err(py_utils::map_error)?
        }
        AudioSamplesData::I16(inner) => {
            utils::detection::detect_silence_regions(inner, (threshold * (i16::MAX as f64)) as i16)
                .map_err(py_utils::map_error)?
        }
        AudioSamplesData::I24(inner) => {
            // For I24, use a simple approach - convert to f64 first
            let f64_inner = inner.as_f64().map_err(py_utils::map_error)?;
            utils::detection::detect_silence_regions(&f64_inner, threshold)
                .map_err(py_utils::map_error)?
        }
    };

    // Convert Vec<(f64, f64)> to Python list of tuples
    let py_list = PyList::new(py, regions.iter().map(|(start, end)| (*start, *end)))?;
    Ok(py_list.into())
}

/// Detect dynamic range of audio data.
///
/// # Arguments
/// * `audio` - AudioSamples object to analyze
///
/// # Returns
/// Dictionary with dynamic range metrics: 'peak_db', 'rms_db', 'dynamic_range'
#[pyfunction(name = "detect_dynamic_range")]
fn py_detect_dynamic_range(py: Python, audio: &PyAudioSamples) -> PyResult<Py<PyAny>> {
    let (peak_db, rms_db, dynamic_range) = match &audio.data {
        AudioSamplesData::F64(inner) => {
            utils::detection::detect_dynamic_range(inner).map_err(py_utils::map_error)?
        }
        AudioSamplesData::F32(inner) => {
            utils::detection::detect_dynamic_range(inner).map_err(py_utils::map_error)?
        }
        AudioSamplesData::I32(inner) => {
            utils::detection::detect_dynamic_range(inner).map_err(py_utils::map_error)?
        }
        AudioSamplesData::I16(inner) => {
            utils::detection::detect_dynamic_range(inner).map_err(py_utils::map_error)?
        }
        AudioSamplesData::I24(inner) => {
            utils::detection::detect_dynamic_range(inner).map_err(py_utils::map_error)?
        }
    };

    let dict = PyDict::new(py);
    dict.set_item("peak_db", peak_db)?;
    dict.set_item("rms_db", rms_db)?;
    dict.set_item("dynamic_range", dynamic_range)?;

    Ok(dict.into())
}

/// Detect clipping in audio data.
///
/// # Arguments
/// * `audio` - AudioSamples object to analyze
/// * `threshold` - Clipping threshold as fraction of maximum amplitude (default: 0.99)
///
/// # Returns
/// List of (start_time, end_time) tuples for clipped regions
#[pyfunction(name = "detect_clipping")]
#[pyo3(signature = (audio, *, threshold=0.99))]
fn py_detect_clipping(py: Python, audio: &PyAudioSamples, threshold: f64) -> PyResult<Py<PyAny>> {
    let clipped_regions = match &audio.data {
        AudioSamplesData::F64(inner) => {
            utils::detection::detect_clipping(inner, threshold).map_err(py_utils::map_error)?
        }
        AudioSamplesData::F32(inner) => {
            utils::detection::detect_clipping(inner, threshold).map_err(py_utils::map_error)?
        }
        AudioSamplesData::I32(inner) => {
            utils::detection::detect_clipping(inner, threshold).map_err(py_utils::map_error)?
        }
        AudioSamplesData::I16(inner) => {
            utils::detection::detect_clipping(inner, threshold).map_err(py_utils::map_error)?
        }
        AudioSamplesData::I24(inner) => {
            utils::detection::detect_clipping(inner, threshold).map_err(py_utils::map_error)?
        }
    };

    // Convert clipped regions to Python list of tuples
    let regions_list = PyList::new(
        py,
        clipped_regions.iter().map(|(start, end)| (*start, *end)),
    )?;

    Ok(regions_list.into())
}

// =====================
// Audio Comparison Functions
// =====================

/// Compute correlation between two audio signals.
///
/// # Arguments
/// * `a` - First AudioSamples object
/// * `b` - Second AudioSamples object  
///
/// # Returns
/// Correlation coefficient as a float
#[pyfunction(name = "correlation")]
fn py_correlation(a: &PyAudioSamples, b: &PyAudioSamples) -> PyResult<f64> {
    // Convert both to f64 for computation
    let a_f64 = a.as_f64().map_err(py_utils::map_error)?;
    let b_f64 = b.as_f64().map_err(py_utils::map_error)?;

    utils::comparison::correlation(&a_f64, &b_f64).map_err(py_utils::map_error)
}

/// Compute mean squared error between two audio signals.
///
/// # Arguments
/// * `a` - First AudioSamples object (reference)
/// * `b` - Second AudioSamples object (test)
///
/// # Returns
/// Mean squared error value
#[pyfunction(name = "mse")]
fn py_mse(a: &PyAudioSamples, b: &PyAudioSamples) -> PyResult<f64> {
    // Convert both to f64 for computation
    let a_f64 = a.as_f64().map_err(py_utils::map_error)?;
    let b_f64 = b.as_f64().map_err(py_utils::map_error)?;

    utils::comparison::mse(&a_f64, &b_f64).map_err(py_utils::map_error)
}

/// Compute signal-to-noise ratio between signal and noise.
///
/// # Arguments  
/// * `signal` - AudioSamples object representing the signal
/// * `noise` - AudioSamples object representing the noise
///
/// # Returns
/// Signal-to-noise ratio in dB
#[pyfunction(name = "snr")]
fn py_snr(signal: &PyAudioSamples, noise: &PyAudioSamples) -> PyResult<f64> {
    // Convert both to f64 for computation
    let signal_f64 = signal.as_f64().map_err(py_utils::map_error)?;
    let noise_f64 = noise.as_f64().map_err(py_utils::map_error)?;

    utils::comparison::snr(&signal_f64, &noise_f64).map_err(py_utils::map_error)
}

/// Align two audio signals by finding optimal time offset.
///
/// # Arguments
/// * `reference` - Reference AudioSamples object
/// * `target` - Target AudioSamples object to align
///
/// # Returns
/// Dictionary with aligned audio and alignment info: 'aligned_audio', 'shift_samples'
#[pyfunction(name = "align_signals")]
fn py_align_signals(
    py: Python,
    reference: &PyAudioSamples,
    target: &PyAudioSamples,
) -> PyResult<Py<PyAny>> {
    // Convert both to f64 for computation
    let reference_f64 = reference.as_f64().map_err(py_utils::map_error)?;
    let target_f64 = target.as_f64().map_err(py_utils::map_error)?;

    let (aligned_audio, shift_samples) =
        utils::comparison::align_signals(&reference_f64, &target_f64)
            .map_err(py_utils::map_error)?;

    let dict = PyDict::new(py);
    dict.set_item(
        "aligned_audio",
        PyAudioSamples::from_data(AudioSamplesData::F64(aligned_audio)),
    )?;
    dict.set_item("shift_samples", shift_samples)?;

    Ok(dict.into())
}

// =====================
// Librosa Compatibility Functions  
// =====================

/// Convert power spectrogram to decibel (dB) scale.
///
/// Similar to librosa.power_to_db() for converting power spectrograms to dB scale.
///
/// # Arguments  
/// * `S` - NumPy array containing power spectrogram values
/// * `ref_val` - Reference power (default: 1.0)
/// * `amin` - Minimum threshold for avoiding log(0) (default: 1e-10)
/// * `top_db` - Threshold the output at top_db below the maximum (optional)
///
/// # Returns
/// NumPy array with values converted to dB scale
///
/// # Examples
/// ```python
/// import numpy as np
/// import audio_samples as aus
///
/// # Convert power spectrogram to dB
/// S = np.abs(D) ** 2  # Power spectrogram from STFT
/// S_db = aus.power_to_db(S, ref_val=np.max(S))
///
/// # With top_db threshold
/// S_db = aus.power_to_db(S, ref_val=np.max(S), top_db=80.0)
/// ```
#[pyfunction(name = "power_to_db")]  
#[pyo3(signature = (S, *, ref_val=1.0, amin=1e-10, top_db=None))]
fn py_power_to_db(
    py: Python,
    S: &Bound<PyAny>,
    ref_val: f64,
    amin: f64,
    top_db: Option<f64>,
) -> PyResult<Py<PyAny>> {
    // Convert NumPy array to ndarray
    if let Ok(array) = S.extract::<numpy::PyReadonlyArray2<f64>>() {
        let arr = array.as_array();
        let mut result = arr.mapv(|x| 10.0 * (x.max(amin) / ref_val).log10());
        
        // Apply top_db threshold if specified
        if let Some(threshold) = top_db {
            let max_val = result.fold(f64::NEG_INFINITY, |max, &x| x.max(max));
            result.mapv_inplace(|x| x.max(max_val - threshold));
        }
        
        py_utils::array2_to_numpy(py, result)
    } else if let Ok(array) = S.extract::<numpy::PyReadonlyArray1<f64>>() {
        let arr = array.as_array();  
        let mut result = arr.mapv(|x| 10.0 * (x.max(amin) / ref_val).log10());
        
        // Apply top_db threshold if specified
        if let Some(threshold) = top_db {
            let max_val = result.fold(f64::NEG_INFINITY, |max, &x| x.max(max));
            result.mapv_inplace(|x| x.max(max_val - threshold));
        }
        
        py_utils::array1_to_numpy(py, result)
    } else {
        Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
            "Input must be a NumPy array of floats"
        ))
    }
}

/// Convert amplitude spectrogram to decibel (dB) scale.
///
/// Similar to librosa.amplitude_to_db() for converting amplitude spectrograms to dB scale.
///
/// # Arguments
/// * `S` - NumPy array containing amplitude spectrogram values  
/// * `ref_val` - Reference amplitude (default: 1.0)
/// * `amin` - Minimum threshold for avoiding log(0) (default: 1e-5)
/// * `top_db` - Threshold the output at top_db below the maximum (optional)
///
/// # Returns
/// NumPy array with values converted to dB scale
///
/// # Examples
/// ```python
/// import numpy as np
/// import audio_samples as aus
///
/// # Convert amplitude spectrogram to dB
/// D = audio.stft(n_fft=2048, hop_length=512)
/// magnitude = np.abs(D)
/// magnitude_db = aus.amplitude_to_db(magnitude, ref_val=np.max(magnitude))
/// ```
#[pyfunction(name = "amplitude_to_db")]
#[pyo3(signature = (S, *, ref_val=1.0, amin=1e-5, top_db=None))]
fn py_amplitude_to_db(
    py: Python,
    S: &Bound<PyAny>,
    ref_val: f64,
    amin: f64, 
    top_db: Option<f64>,
) -> PyResult<Py<PyAny>> {
    // Convert NumPy array to ndarray
    if let Ok(array) = S.extract::<numpy::PyReadonlyArray2<f64>>() {
        let arr = array.as_array();
        let mut result = arr.mapv(|x| 20.0 * (x.max(amin) / ref_val).log10());
        
        // Apply top_db threshold if specified
        if let Some(threshold) = top_db {
            let max_val = result.fold(f64::NEG_INFINITY, |max, &x| x.max(max));
            result.mapv_inplace(|x| x.max(max_val - threshold));
        }
        
        py_utils::array2_to_numpy(py, result)
    } else if let Ok(array) = S.extract::<numpy::PyReadonlyArray1<f64>>() {
        let arr = array.as_array();
        let mut result = arr.mapv(|x| 20.0 * (x.max(amin) / ref_val).log10());
        
        // Apply top_db threshold if specified  
        if let Some(threshold) = top_db {
            let max_val = result.fold(f64::NEG_INFINITY, |max, &x| x.max(max));
            result.mapv_inplace(|x| x.max(max_val - threshold));
        }
        
        py_utils::array1_to_numpy(py, result)
    } else {
        Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
            "Input must be a NumPy array of floats"
        ))
    }
}

/// Generate frequency axis for FFT results.
///
/// Similar to numpy.fft.fftfreq() and librosa's frequency utilities.
///
/// # Arguments
/// * `n` - Length of the FFT (number of frequency bins)
/// * `sr` - Sample rate in Hz
/// * `n_fft` - FFT window size (optional, defaults to 2*(n-1) for compatibility)
///
/// # Returns
/// NumPy array of frequency values in Hz
///
/// # Examples  
/// ```python
/// import audio_samples as aus
///
/// # Get frequency axis for STFT result
/// D = audio.stft(n_fft=2048, hop_length=512)
/// freqs = aus.fft_frequencies(n=D.shape[0], sr=44100, n_fft=2048)
/// ```
#[pyfunction(name = "fft_frequencies")]
#[pyo3(signature = (n, sr, *, n_fft=None))]
fn py_fft_frequencies(py: Python, n: usize, sr: f64, n_fft: Option<usize>) -> PyResult<Py<PyAny>> {
    let actual_n_fft = n_fft.unwrap_or(2 * (n - 1));
    let frequencies: Vec<f64> = (0..n)
        .map(|i| (i as f64) * sr / (actual_n_fft as f64))
        .collect();
    
    let freq_array = ndarray::Array1::from(frequencies);
    py_utils::array1_to_numpy(py, freq_array)
}

/// Generate time axis for STFT frames.
///
/// Similar to librosa.times_like() for generating time coordinates of STFT frames.
///
/// # Arguments
/// * `n_frames` - Number of time frames
/// * `sr` - Sample rate in Hz  
/// * `hop_length` - Number of samples between frames
/// * `n_fft` - FFT window size (optional, for centering calculation)
///
/// # Returns
/// NumPy array of time values in seconds
///
/// # Examples
/// ```python
/// import audio_samples as aus
///
/// # Get time axis for STFT result
/// D = audio.stft(n_fft=2048, hop_length=512)
/// times = aus.frames_to_time(n_frames=D.shape[1], sr=44100, hop_length=512)
/// ```
#[pyfunction(name = "frames_to_time")]
#[pyo3(signature = (n_frames, sr, hop_length, *, n_fft=None))]
fn py_frames_to_time(py: Python, n_frames: usize, sr: f64, hop_length: usize, n_fft: Option<usize>) -> PyResult<Py<PyAny>> {
    // Center frames if n_fft provided (librosa default behavior)
    let offset = if let Some(fft_size) = n_fft { 
        (fft_size / 2) as f64 
    } else { 
        0.0 
    };
    
    let times: Vec<f64> = (0..n_frames)
        .map(|i| (offset + (i * hop_length) as f64) / sr)
        .collect();
    
    let time_array = ndarray::Array1::from(times);
    py_utils::array1_to_numpy(py, time_array)
}

/// Register all utility functions with the Python module.
pub fn register_functions(m: &Bound<PyModule>) -> PyResult<()> {
    // Signal generation functions
    m.add_function(wrap_pyfunction!(py_sine_wave, m)?)?;
    m.add_function(wrap_pyfunction!(py_cosine_wave, m)?)?;
    m.add_function(wrap_pyfunction!(py_white_noise, m)?)?;
    m.add_function(wrap_pyfunction!(py_pink_noise, m)?)?;
    m.add_function(wrap_pyfunction!(py_square_wave, m)?)?;
    m.add_function(wrap_pyfunction!(py_sawtooth_wave, m)?)?;
    m.add_function(wrap_pyfunction!(py_triangle_wave, m)?)?;
    m.add_function(wrap_pyfunction!(py_chirp, m)?)?;
    m.add_function(wrap_pyfunction!(py_impulse, m)?)?;
    m.add_function(wrap_pyfunction!(py_silence, m)?)?;

    // Detection functions
    m.add_function(wrap_pyfunction!(py_detect_sample_rate, m)?)?;
    m.add_function(wrap_pyfunction!(py_detect_fundamental_frequency, m)?)?;
    m.add_function(wrap_pyfunction!(py_detect_silence_regions, m)?)?;
    m.add_function(wrap_pyfunction!(py_detect_dynamic_range, m)?)?;
    m.add_function(wrap_pyfunction!(py_detect_clipping, m)?)?;

    // Comparison functions
    m.add_function(wrap_pyfunction!(py_correlation, m)?)?;
    m.add_function(wrap_pyfunction!(py_mse, m)?)?;
    m.add_function(wrap_pyfunction!(py_snr, m)?)?;
    m.add_function(wrap_pyfunction!(py_align_signals, m)?)?;

    // Librosa compatibility functions
    m.add_function(wrap_pyfunction!(py_power_to_db, m)?)?;
    m.add_function(wrap_pyfunction!(py_amplitude_to_db, m)?)?;
    m.add_function(wrap_pyfunction!(py_fft_frequencies, m)?)?;
    m.add_function(wrap_pyfunction!(py_frames_to_time, m)?)?;

    Ok(())
}
