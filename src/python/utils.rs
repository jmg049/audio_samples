//! Utility functions for efficient NumPy array conversions and error handling.
//!
//! This module provides the core conversion logic between NumPy arrays and
//! AudioSamples, optimizing for zero-copy operations where possible and
//! providing clear error mappings between Rust and Python.

use crate::{AudioSampleError, AudioSamples};
use ndarray::{Array1, Array2};
use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use pyo3::types::PyList;

/// Convert a Python/NumPy array to AudioSamplesData, preserving the original dtype.
///
/// This function handles both 1D (mono) and 2D (multi-channel) arrays,
/// attempting zero-copy conversion where possible and preserving the original
/// NumPy dtype for memory efficiency and performance.
///
/// # Arguments
/// * `py_array` - Python object that should be a NumPy array
/// * `sample_rate` - Sample rate in Hz
/// * `channels` - Override channel count (auto-detected from shape if None)
///
/// # Returns
/// AudioSamplesData with the converted data and preserved dtype
pub fn convert_numpy_to_audio_samples(
    py_array: &Bound<PyAny>,
    sample_rate: u32,
    channels: Option<usize>,
) -> PyResult<crate::python::AudioSamplesData> {
    use crate::python::AudioSamplesData;

    // let target_type = 

    // Try f64 (preserve as f64)
    if let Ok(array_f64) = py_array.extract::<PyReadonlyArray1<f64>>() {
        let ndarray = array_f64.as_array().to_owned();
        let audio = AudioSamples::new_mono(ndarray, sample_rate);
        return Ok(AudioSamplesData::F64(audio));
    }

    if let Ok(array_f64) = py_array.extract::<PyReadonlyArray2<f64>>() {
        let ndarray = array_f64.as_array().to_owned();
        if let Some(ch) = channels {
            if ch != ndarray.nrows() {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "Channel count mismatch: expected {}, got {}",
                    ch,
                    ndarray.nrows()
                )));
            }
        }
        let audio = AudioSamples::new_multi_channel(ndarray, sample_rate);
        return Ok(AudioSamplesData::F64(audio));
    }

    // Try f32 (preserve as f32)
    if let Ok(array_f32) = py_array.extract::<PyReadonlyArray1<f32>>() {
        let ndarray = array_f32.as_array().to_owned();
        let audio = AudioSamples::new_mono(ndarray, sample_rate);
        return Ok(AudioSamplesData::F32(audio));
    }

    if let Ok(array_f32) = py_array.extract::<PyReadonlyArray2<f32>>() {
        let ndarray = array_f32.as_array().to_owned();
        if let Some(ch) = channels {
            if ch != ndarray.nrows() {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "Channel count mismatch: expected {}, got {}",
                    ch,
                    ndarray.nrows()
                )));
            }
        }
        let audio = AudioSamples::new_multi_channel(ndarray, sample_rate);
        return Ok(AudioSamplesData::F32(audio));
    }

    // Try i32 (preserve as i32)
    if let Ok(array_i32) = py_array.extract::<PyReadonlyArray1<i32>>() {
        let ndarray = array_i32.as_array().to_owned();
        let audio = AudioSamples::new_mono(ndarray, sample_rate);
        return Ok(AudioSamplesData::I32(audio));
    }

    if let Ok(array_i32) = py_array.extract::<PyReadonlyArray2<i32>>() {
        let ndarray = array_i32.as_array().to_owned();
        if let Some(ch) = channels {
            if ch != ndarray.nrows() {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "Channel count mismatch: expected {}, got {}",
                    ch,
                    ndarray.nrows()
                )));
            }
        }
        let audio = AudioSamples::new_multi_channel(ndarray, sample_rate);
        return Ok(AudioSamplesData::I32(audio));
    }

    // Try i16 (preserve as i16)
    if let Ok(array_i16) = py_array.extract::<PyReadonlyArray1<i16>>() {
        let ndarray = array_i16.as_array().to_owned();
        let audio = AudioSamples::new_mono(ndarray, sample_rate);
        return Ok(AudioSamplesData::I16(audio));
    }

    if let Ok(array_i16) = py_array.extract::<PyReadonlyArray2<i16>>() {
        let ndarray = array_i16.as_array().to_owned();
        if let Some(ch) = channels {
            if ch != ndarray.nrows() {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "Channel count mismatch: expected {}, got {}",
                    ch,
                    ndarray.nrows()
                )));
            }
        }
        let audio = AudioSamples::new_multi_channel(ndarray, sample_rate);
        return Ok(AudioSamplesData::I16(audio));
    }

    eprintln!(
        "Failed to convert NumPy array to AudioSamplesData: unsupported dtype or shape - {:?}",
        py_array.get_type().name()
    );

    // If we get here, the array type is not supported
    Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
        "Unsupported array type. Supported types: f64, f32, i32, i16",
    ))
}

/// Convert AudioSamplesData to a Python NumPy array, preserving type when possible.
///
/// This function attempts to share memory when possible and provides
/// efficient type conversion when requested.
///
/// # Arguments
/// * `py` - Python interpreter reference
/// * `audio_data` - AudioSamplesData to convert
/// * `target_type` - Target type (None preserves original type)
/// * `copy` - Force a copy even if sharing is possible
///
/// # Returns
/// PyObject containing the NumPy array
pub fn convert_audio_samples_to_numpy(
    py: Python,
    audio_data: &crate::python::AudioSamplesData,
    target_type: Option<crate::python::conversions::TargetType>,
    copy: bool,
) -> PyResult<PyObject> {
    use crate::python::conversions::TargetType;
    use crate::python::AudioSamplesData;

    // If no target type specified, preserve the original type
    let target = if let Some(t) = target_type {
        t
    } else {
        // Determine target type from current data type
        match audio_data {
            AudioSamplesData::I16(_) => TargetType::I16,
            AudioSamplesData::I24(_) => TargetType::I32, // I24 -> i32 in numpy
            AudioSamplesData::I32(_) => TargetType::I32,
            AudioSamplesData::F32(_) => TargetType::F32,
            AudioSamplesData::F64(_) => TargetType::F64,
        }
    };

    match target {
        TargetType::F64 => convert_to_numpy_f64(py, audio_data, copy),
        TargetType::F32 => convert_to_numpy_f32(py, audio_data, copy),
        TargetType::I32 => convert_to_numpy_i32(py, audio_data, copy),
        TargetType::I16 => convert_to_numpy_i16(py, audio_data, copy),
    }
}

/// Convert to f64 NumPy array (potentially zero-copy)
fn convert_to_numpy_f64(
    py: Python,
    audio_data: &crate::python::AudioSamplesData,
    _force_copy: bool,
) -> PyResult<PyObject> {
    use crate::operations::AudioTypeConversion;
    use crate::python::AudioSamplesData;

    // If already f64, try zero-copy; otherwise convert
    let f64_audio = match audio_data {
        AudioSamplesData::F64(audio) => {
            // Already f64, can potentially share memory
            audio.clone()
        }
        AudioSamplesData::F32(audio) => audio.to_f64().map_err(map_error)?,
        AudioSamplesData::I32(audio) => audio.to_f64().map_err(map_error)?,
        AudioSamplesData::I16(audio) => audio.to_f64().map_err(map_error)?,
        AudioSamplesData::I24(audio) => audio.to_f64().map_err(map_error)?,
    };

    match f64_audio.channels() {
        1 => {
            let mono_data = f64_audio.as_mono().ok_or_else(|| {
                PyErr::new::<pyo3::exceptions::PyValueError, _>("Expected mono audio data")
            })?;
            let copied = mono_data.to_owned();
            Ok(PyArray1::from_owned_array(py, copied).into())
        }
        _ => {
            let multi_data = f64_audio.as_multi_channel().ok_or_else(|| {
                PyErr::new::<pyo3::exceptions::PyValueError, _>("Expected multi-channel audio data")
            })?;
            let copied = multi_data.to_owned();
            Ok(PyArray2::from_owned_array(py, copied).into())
        }
    }
}

/// Convert to f32 NumPy array
fn convert_to_numpy_f32(
    py: Python,
    audio_data: &crate::python::AudioSamplesData,
    _force_copy: bool,
) -> PyResult<PyObject> {
    use crate::operations::AudioTypeConversion;
    use crate::python::AudioSamplesData;

    // If already f32, try zero-copy; otherwise convert
    let f32_audio = match audio_data {
        AudioSamplesData::F32(audio) => {
            // Already f32, can potentially share memory
            audio.clone()
        }
        AudioSamplesData::F64(audio) => audio.to_f32().map_err(map_error)?,
        AudioSamplesData::I32(audio) => audio.to_f32().map_err(map_error)?,
        AudioSamplesData::I16(audio) => audio.to_f32().map_err(map_error)?,
        AudioSamplesData::I24(audio) => audio.to_f32().map_err(map_error)?,
    };

    match f32_audio.channels() {
        1 => {
            let mono_data = f32_audio.as_mono().ok_or_else(|| {
                PyErr::new::<pyo3::exceptions::PyValueError, _>("Expected audio data")
            })?;
            let copied = mono_data.to_owned();
            Ok(PyArray1::from_owned_array(py, copied).into())
        }
        _ => {
            let multi_data = f32_audio.as_multi_channel().ok_or_else(|| {
                PyErr::new::<pyo3::exceptions::PyValueError, _>("Expected audio data")
            })?;
            let copied = multi_data.to_owned();
            Ok(PyArray2::from_owned_array(py, copied).into())
        }
    }
}

/// Convert to i32 NumPy array
fn convert_to_numpy_i32(
    py: Python,
    audio_data: &crate::python::AudioSamplesData,
    _force_copy: bool,
) -> PyResult<PyObject> {
    use crate::operations::AudioTypeConversion;
    use crate::python::AudioSamplesData;

    // If already i32 or I24, try zero-copy; otherwise convert
    let i32_audio = match audio_data {
        AudioSamplesData::I32(audio) => {
            // Already i32, can potentially share memory
            audio.clone()
        }
        AudioSamplesData::I24(audio) => audio.to_i32().map_err(map_error)?,
        AudioSamplesData::F64(audio) => audio.to_i32().map_err(map_error)?,
        AudioSamplesData::F32(audio) => audio.to_i32().map_err(map_error)?,
        AudioSamplesData::I16(audio) => audio.to_i32().map_err(map_error)?,
    };

    match i32_audio.channels() {
        1 => {
            let mono_data = i32_audio.as_mono().ok_or_else(|| {
                PyErr::new::<pyo3::exceptions::PyValueError, _>("Expected audio data")
            })?;
            let copied = mono_data.to_owned();
            Ok(PyArray1::from_owned_array(py, copied).into())
        }
        _ => {
            let multi_data = i32_audio.as_multi_channel().ok_or_else(|| {
                PyErr::new::<pyo3::exceptions::PyValueError, _>("Expected audio data")
            })?;
            let copied = multi_data.to_owned();
            Ok(PyArray2::from_owned_array(py, copied).into())
        }
    }
}

/// Convert to i16 NumPy array
fn convert_to_numpy_i16(
    py: Python,
    audio_data: &crate::python::AudioSamplesData,
    _force_copy: bool,
) -> PyResult<PyObject> {
    use crate::operations::AudioTypeConversion;
    use crate::python::AudioSamplesData;

    // If already i16, try zero-copy; otherwise convert
    let i16_audio = match audio_data {
        AudioSamplesData::I16(audio) => {
            // Already i16, can potentially share memory
            audio.clone()
        }
        AudioSamplesData::F64(audio) => audio.to_i16().map_err(map_error)?,
        AudioSamplesData::F32(audio) => audio.to_i16().map_err(map_error)?,
        AudioSamplesData::I32(audio) => audio.to_i16().map_err(map_error)?,
        AudioSamplesData::I24(audio) => audio.to_i16().map_err(map_error)?,
    };

    match i16_audio.channels() {
        1 => {
            let mono_data = i16_audio.as_mono().ok_or_else(|| {
                PyErr::new::<pyo3::exceptions::PyValueError, _>("Expected audio data")
            })?;
            let copied = mono_data.to_owned();
            Ok(PyArray1::from_owned_array(py, copied).into())
        }
        _ => {
            let multi_data = i16_audio.as_multi_channel().ok_or_else(|| {
                PyErr::new::<pyo3::exceptions::PyValueError, _>("Expected audio data")
            })?;
            let copied = multi_data.to_owned();
            Ok(PyArray2::from_owned_array(py, copied).into())
        }
    }
}

/// Map Rust AudioSampleError to appropriate Python exceptions.
///
/// This provides clear error messages and uses appropriate Python exception types
/// for different kinds of errors that can occur during audio processing.
pub fn map_error(error: AudioSampleError) -> PyErr {
    match error {
        AudioSampleError::ConversionError(value, from_type, to_type, reason) => {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Audio conversion failed: {} ({} -> {}): {}",
                value, from_type, to_type, reason
            ))
        }
        AudioSampleError::InvalidRange(msg) => {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Invalid range: {}", msg))
        }
        AudioSampleError::InvalidParameter(msg) => {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Invalid parameter: {}", msg))
        }
        AudioSampleError::DimensionMismatch(msg) => {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Dimension mismatch: {}", msg))
        }
    }
}

/// Validate that a string parameter is one of the allowed values.
pub fn validate_string_param(param_name: &str, value: &str, allowed: &[&str]) -> PyResult<()> {
    if allowed.contains(&value) {
        Ok(())
    } else {
        Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "Invalid {}: '{}'. Allowed values: {:?}",
            param_name, value, allowed
        )))
    }
}

/// Helper to create a Python list from a Rust vector.
pub fn vec_to_pylist_f64(py: Python, vec: Vec<f64>) -> PyResult<PyObject> {
    let py_list = PyList::new(py, vec)?;
    Ok(py_list.into())
}

/// Helper to create a 2D NumPy array from a Rust Array2.
pub fn array2_to_numpy<T>(py: Python, array: Array2<T>) -> PyResult<PyObject>
where
    T: numpy::Element + Copy,
{
    let py_array = PyArray2::from_owned_array(py, array);
    Ok(py_array.into())
}

/// Helper to create a 1D NumPy array from a Rust Array1.
pub fn array1_to_numpy<T>(py: Python, array: Array1<T>) -> PyResult<PyObject>
where
    T: numpy::Element + Copy,
{
    let py_array = PyArray1::from_owned_array(py, array);
    Ok(py_array.into())
}

/// Parse optional hop size with sensible default.
pub fn parse_hop_size(hop_size: Option<usize>, window_size: usize) -> usize {
    hop_size.unwrap_or(window_size / 4) // Default to 75% overlap
}

/// Parse optional maximum frequency with sensible default based on sample rate.
pub fn parse_fmax(fmax: Option<f64>, sample_rate: u32) -> f64 {
    fmax.unwrap_or(sample_rate as f64 / 2.0) // Nyquist frequency
}
