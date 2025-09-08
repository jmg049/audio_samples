//! Python bindings for audio signal generation utilities.

use crate::python::{utils, PyAudioSamples, AudioSamplesData};
use crate::utils::generation::*;
use crate::AudioSample;
use pyo3::prelude::*;

/// Generate a sine wave with the specified parameters.
///
/// # Arguments
/// * `frequency` - Frequency of the sine wave in Hz
/// * `duration` - Duration of the signal in seconds  
/// * `sample_rate` - Sample rate in Hz
/// * `amplitude` - Amplitude of the sine wave (0.0 to 1.0, default: 1.0)
/// * `dtype` - Data type (numpy.dtype or string like 'f32', 'f64', 'i16', 'i32', default: 'f64')
///
/// # Examples
/// ```python
/// import audio_samples as aus
/// 
/// # Generate 440Hz tone for 1 second
/// sine = aus.sine_wave(440.0, 1.0, 44100)
/// 
/// # Generate with specific amplitude and dtype
/// sine = aus.sine_wave(440.0, 1.0, 44100, amplitude=0.5, dtype='f32')
/// ```
#[pyfunction(name = "sine_wave")]
#[pyo3(signature = (frequency, duration, sample_rate, *, amplitude=1.0, dtype=None))]
fn py_sine_wave(
    _py: Python,
    frequency: f64,
    duration: f64,
    sample_rate: u32,
    amplitude: f64,
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

    // Generate sine wave in the target type
    let data = match target_type {
        crate::python::conversions::TargetType::F64 => {
            let inner = sine_wave::<f64>(frequency, duration, sample_rate, amplitude)
                .map_err(utils::map_error)?;
            AudioSamplesData::F64(inner)
        }
        crate::python::conversions::TargetType::F32 => {
            let inner = sine_wave::<f32>(frequency, duration, sample_rate, amplitude)
                .map_err(utils::map_error)?;
            AudioSamplesData::F32(inner)
        }
        crate::python::conversions::TargetType::I32 => {
            let inner = sine_wave::<i32>(frequency, duration, sample_rate, amplitude)
                .map_err(utils::map_error)?;
            AudioSamplesData::I32(inner)
        }
        crate::python::conversions::TargetType::I16 => {
            let inner = sine_wave::<i16>(frequency, duration, sample_rate, amplitude)
                .map_err(utils::map_error)?;
            AudioSamplesData::I16(inner)
        }
    };

    Ok(PyAudioSamples { data })
}

/// Generate a cosine wave with the specified parameters.
///
/// # Arguments
/// * `frequency` - Frequency of the cosine wave in Hz
/// * `duration` - Duration of the signal in seconds
/// * `sample_rate` - Sample rate in Hz  
/// * `amplitude` - Amplitude of the cosine wave (0.0 to 1.0, default: 1.0)
/// * `dtype` - Data type (numpy.dtype or string like 'f32', 'f64', 'i16', 'i32', default: 'f64')
///
/// # Examples
/// ```python
/// import audio_samples as aus
///
/// # Generate 880Hz cosine for 2 seconds
/// cosine = aus.cosine_wave(880.0, 2.0, 44100)
/// ```
#[pyfunction(name = "cosine_wave")]
#[pyo3(signature = (frequency, duration, sample_rate, *, amplitude=1.0, dtype=None))]
fn py_cosine_wave(
    _py: Python,
    frequency: f64,
    duration: f64,
    sample_rate: u32,
    amplitude: f64,
    dtype: Option<&Bound<PyAny>>,
) -> PyResult<PyAudioSamples> {
    let target_type = parse_dtype_param(dtype)?;

    let data = match target_type {
        crate::python::conversions::TargetType::F64 => {
            let inner = cosine_wave::<f64>(frequency, duration, sample_rate, amplitude)
                .map_err(utils::map_error)?;
            AudioSamplesData::F64(inner)
        }
        crate::python::conversions::TargetType::F32 => {
            let inner = cosine_wave::<f32>(frequency, duration, sample_rate, amplitude)
                .map_err(utils::map_error)?;
            AudioSamplesData::F32(inner)
        }
        crate::python::conversions::TargetType::I32 => {
            let inner = cosine_wave::<i32>(frequency, duration, sample_rate, amplitude)
                .map_err(utils::map_error)?;
            AudioSamplesData::I32(inner)
        }
        crate::python::conversions::TargetType::I16 => {
            let inner = cosine_wave::<i16>(frequency, duration, sample_rate, amplitude)
                .map_err(utils::map_error)?;
            AudioSamplesData::I16(inner)
        }
    };

    Ok(PyAudioSamples { data })
}

/// Generate white noise with the specified parameters.
///
/// White noise has equal energy across all frequencies within the Nyquist range.
///
/// # Arguments
/// * `duration` - Duration of the signal in seconds
/// * `sample_rate` - Sample rate in Hz
/// * `amplitude` - Amplitude of the noise (0.0 to 1.0, default: 1.0)
/// * `dtype` - Data type (numpy.dtype or string like 'f32', 'f64', 'i16', 'i32', default: 'f64')
///
/// # Examples
/// ```python
/// import audio_samples as aus
///
/// # Generate 5 seconds of white noise
/// noise = aus.white_noise(5.0, 44100)
/// ```
#[pyfunction(name = "white_noise")]  
#[pyo3(signature = (duration, sample_rate, *, amplitude=1.0, dtype=None))]
fn py_white_noise(
    _py: Python,
    duration: f64,
    sample_rate: u32,
    amplitude: f64,
    dtype: Option<&Bound<PyAny>>,
) -> PyResult<PyAudioSamples> {
    let target_type = parse_dtype_param(dtype)?;

    let data = match target_type {
        crate::python::conversions::TargetType::F64 => {
            let inner = white_noise::<f64>(duration, sample_rate, amplitude)
                .map_err(utils::map_error)?;
            AudioSamplesData::F64(inner)
        }
        crate::python::conversions::TargetType::F32 => {
            let inner = white_noise::<f32>(duration, sample_rate, amplitude)
                .map_err(utils::map_error)?;
            AudioSamplesData::F32(inner)
        }
        crate::python::conversions::TargetType::I32 => {
            let inner = white_noise::<i32>(duration, sample_rate, amplitude)
                .map_err(utils::map_error)?;
            AudioSamplesData::I32(inner)
        }
        crate::python::conversions::TargetType::I16 => {
            let inner = white_noise::<i16>(duration, sample_rate, amplitude)
                .map_err(utils::map_error)?;
            AudioSamplesData::I16(inner)
        }
    };

    Ok(PyAudioSamples { data })
}

/// Generate pink noise with the specified parameters.
///
/// Pink noise has equal energy per octave, with power decreasing at -3dB per octave.
///
/// # Arguments
/// * `duration` - Duration of the signal in seconds
/// * `sample_rate` - Sample rate in Hz
/// * `amplitude` - Amplitude of the noise (0.0 to 1.0, default: 1.0)
/// * `dtype` - Data type (numpy.dtype or string like 'f32', 'f64', 'i16', 'i32', default: 'f64')
///
/// # Examples
/// ```python
/// import audio_samples as aus
///
/// # Generate 3 seconds of pink noise
/// pink = aus.pink_noise(3.0, 44100)
/// ```
#[pyfunction(name = "pink_noise")]
#[pyo3(signature = (duration, sample_rate, *, amplitude=1.0, dtype=None))]
fn py_pink_noise(
    _py: Python,
    duration: f64,
    sample_rate: u32,
    amplitude: f64,
    dtype: Option<&Bound<PyAny>>,
) -> PyResult<PyAudioSamples> {
    let target_type = parse_dtype_param(dtype)?;

    let data = match target_type {
        crate::python::conversions::TargetType::F64 => {
            let inner = pink_noise::<f64>(duration, sample_rate, amplitude)
                .map_err(utils::map_error)?;
            AudioSamplesData::F64(inner)
        }
        crate::python::conversions::TargetType::F32 => {
            let inner = pink_noise::<f32>(duration, sample_rate, amplitude)
                .map_err(utils::map_error)?;
            AudioSamplesData::F32(inner)
        }
        crate::python::conversions::TargetType::I32 => {
            let inner = pink_noise::<i32>(duration, sample_rate, amplitude)
                .map_err(utils::map_error)?;
            AudioSamplesData::I32(inner)
        }
        crate::python::conversions::TargetType::I16 => {
            let inner = pink_noise::<i16>(duration, sample_rate, amplitude)
                .map_err(utils::map_error)?;
            AudioSamplesData::I16(inner)
        }
    };

    Ok(PyAudioSamples { data })
}

/// Generate a square wave with the specified parameters.
///
/// # Arguments
/// * `frequency` - Frequency of the square wave in Hz
/// * `duration` - Duration of the signal in seconds
/// * `sample_rate` - Sample rate in Hz
/// * `amplitude` - Amplitude of the square wave (0.0 to 1.0, default: 1.0)
/// * `dtype` - Data type (numpy.dtype or string like 'f32', 'f64', 'i16', 'i32', default: 'f64')
///
/// # Examples
/// ```python
/// import audio_samples as aus
///
/// # Generate 1kHz square wave for 0.5 seconds
/// square = aus.square_wave(1000.0, 0.5, 44100)
/// ```
#[pyfunction(name = "square_wave")]
#[pyo3(signature = (frequency, duration, sample_rate, *, amplitude=1.0, dtype=None))]
fn py_square_wave(
    _py: Python,
    frequency: f64,
    duration: f64,
    sample_rate: u32,
    amplitude: f64,
    dtype: Option<&Bound<PyAny>>,
) -> PyResult<PyAudioSamples> {
    let target_type = parse_dtype_param(dtype)?;

    let data = match target_type {
        crate::python::conversions::TargetType::F64 => {
            let inner = square_wave::<f64>(frequency, duration, sample_rate, amplitude)
                .map_err(utils::map_error)?;
            AudioSamplesData::F64(inner)
        }
        crate::python::conversions::TargetType::F32 => {
            let inner = square_wave::<f32>(frequency, duration, sample_rate, amplitude)
                .map_err(utils::map_error)?;
            AudioSamplesData::F32(inner)
        }
        crate::python::conversions::TargetType::I32 => {
            let inner = square_wave::<i32>(frequency, duration, sample_rate, amplitude)
                .map_err(utils::map_error)?;
            AudioSamplesData::I32(inner)
        }
        crate::python::conversions::TargetType::I16 => {
            let inner = square_wave::<i16>(frequency, duration, sample_rate, amplitude)
                .map_err(utils::map_error)?;
            AudioSamplesData::I16(inner)
        }
    };

    Ok(PyAudioSamples { data })
}

/// Generate a sawtooth wave with the specified parameters.
///
/// # Arguments
/// * `frequency` - Frequency of the sawtooth wave in Hz
/// * `duration` - Duration of the signal in seconds
/// * `sample_rate` - Sample rate in Hz
/// * `amplitude` - Amplitude of the sawtooth wave (0.0 to 1.0, default: 1.0)
/// * `dtype` - Data type (numpy.dtype or string like 'f32', 'f64', 'i16', 'i32', default: 'f64')
///
/// # Examples
/// ```python
/// import audio_samples as aus
///
/// # Generate 220Hz sawtooth for 1.5 seconds
/// sawtooth = aus.sawtooth_wave(220.0, 1.5, 44100)
/// ```
#[pyfunction(name = "sawtooth_wave")]
#[pyo3(signature = (frequency, duration, sample_rate, *, amplitude=1.0, dtype=None))]
fn py_sawtooth_wave(
    _py: Python,
    frequency: f64,
    duration: f64,
    sample_rate: u32,
    amplitude: f64,
    dtype: Option<&Bound<PyAny>>,
) -> PyResult<PyAudioSamples> {
    let target_type = parse_dtype_param(dtype)?;

    let data = match target_type {
        crate::python::conversions::TargetType::F64 => {
            let inner = sawtooth_wave::<f64>(frequency, duration, sample_rate, amplitude)
                .map_err(utils::map_error)?;
            AudioSamplesData::F64(inner)
        }
        crate::python::conversions::TargetType::F32 => {
            let inner = sawtooth_wave::<f32>(frequency, duration, sample_rate, amplitude)
                .map_err(utils::map_error)?;
            AudioSamplesData::F32(inner)
        }
        crate::python::conversions::TargetType::I32 => {
            let inner = sawtooth_wave::<i32>(frequency, duration, sample_rate, amplitude)
                .map_err(utils::map_error)?;
            AudioSamplesData::I32(inner)
        }
        crate::python::conversions::TargetType::I16 => {
            let inner = sawtooth_wave::<i16>(frequency, duration, sample_rate, amplitude)
                .map_err(utils::map_error)?;
            AudioSamplesData::I16(inner)
        }
    };

    Ok(PyAudioSamples { data })
}

/// Generate a triangle wave with the specified parameters.
///
/// # Arguments
/// * `frequency` - Frequency of the triangle wave in Hz
/// * `duration` - Duration of the signal in seconds
/// * `sample_rate` - Sample rate in Hz
/// * `amplitude` - Amplitude of the triangle wave (0.0 to 1.0, default: 1.0)
/// * `dtype` - Data type (numpy.dtype or string like 'f32', 'f64', 'i16', 'i32', default: 'f64')
///
/// # Examples
/// ```python
/// import audio_samples as aus
///
/// # Generate 330Hz triangle wave for 2 seconds
/// triangle = aus.triangle_wave(330.0, 2.0, 44100)
/// ```
#[pyfunction(name = "triangle_wave")]
#[pyo3(signature = (frequency, duration, sample_rate, *, amplitude=1.0, dtype=None))]
fn py_triangle_wave(
    _py: Python,
    frequency: f64,
    duration: f64,
    sample_rate: u32,
    amplitude: f64,
    dtype: Option<&Bound<PyAny>>,
) -> PyResult<PyAudioSamples> {
    let target_type = parse_dtype_param(dtype)?;

    let data = match target_type {
        crate::python::conversions::TargetType::F64 => {
            let inner = triangle_wave::<f64>(frequency, duration, sample_rate, amplitude)
                .map_err(utils::map_error)?;
            AudioSamplesData::F64(inner)
        }
        crate::python::conversions::TargetType::F32 => {
            let inner = triangle_wave::<f32>(frequency, duration, sample_rate, amplitude)
                .map_err(utils::map_error)?;
            AudioSamplesData::F32(inner)
        }
        crate::python::conversions::TargetType::I32 => {
            let inner = triangle_wave::<i32>(frequency, duration, sample_rate, amplitude)
                .map_err(utils::map_error)?;
            AudioSamplesData::I32(inner)
        }
        crate::python::conversions::TargetType::I16 => {
            let inner = triangle_wave::<i16>(frequency, duration, sample_rate, amplitude)
                .map_err(utils::map_error)?;
            AudioSamplesData::I16(inner)
        }
    };

    Ok(PyAudioSamples { data })
}

/// Generate a chirp (frequency sweep) signal.
///
/// # Arguments
/// * `start_freq` - Starting frequency in Hz
/// * `end_freq` - Ending frequency in Hz
/// * `duration` - Duration of the signal in seconds
/// * `sample_rate` - Sample rate in Hz
/// * `amplitude` - Amplitude of the chirp (0.0 to 1.0, default: 1.0)
/// * `dtype` - Data type (numpy.dtype or string like 'f32', 'f64', 'i16', 'i32', default: 'f64')
///
/// # Examples
/// ```python
/// import audio_samples as aus
///
/// # Generate frequency sweep from 100Hz to 2000Hz over 3 seconds
/// chirp = aus.chirp(100.0, 2000.0, 3.0, 44100)
/// ```
#[pyfunction(name = "chirp")]
#[pyo3(signature = (start_freq, end_freq, duration, sample_rate, *, amplitude=1.0, dtype=None))]
fn py_chirp(
    _py: Python,
    start_freq: f64,
    end_freq: f64,
    duration: f64,
    sample_rate: u32,
    amplitude: f64,
    dtype: Option<&Bound<PyAny>>,
) -> PyResult<PyAudioSamples> {
    let target_type = parse_dtype_param(dtype)?;

    let data = match target_type {
        crate::python::conversions::TargetType::F64 => {
            let inner = chirp::<f64>(start_freq, end_freq, duration, sample_rate, amplitude)
                .map_err(utils::map_error)?;
            AudioSamplesData::F64(inner)
        }
        crate::python::conversions::TargetType::F32 => {
            let inner = chirp::<f32>(start_freq, end_freq, duration, sample_rate, amplitude)
                .map_err(utils::map_error)?;
            AudioSamplesData::F32(inner)
        }
        crate::python::conversions::TargetType::I32 => {
            let inner = chirp::<i32>(start_freq, end_freq, duration, sample_rate, amplitude)
                .map_err(utils::map_error)?;
            AudioSamplesData::I32(inner)
        }
        crate::python::conversions::TargetType::I16 => {
            let inner = chirp::<i16>(start_freq, end_freq, duration, sample_rate, amplitude)
                .map_err(utils::map_error)?;
            AudioSamplesData::I16(inner)
        }
    };

    Ok(PyAudioSamples { data })
}

/// Generate an impulse (delta function) signal.
///
/// # Arguments
/// * `duration` - Duration of the signal in seconds
/// * `sample_rate` - Sample rate in Hz
/// * `amplitude` - Amplitude of the impulse (0.0 to 1.0, default: 1.0)
/// * `position` - Position of the impulse in seconds (0.0 to duration, default: 0.0)
/// * `dtype` - Data type (numpy.dtype or string like 'f32', 'f64', 'i16', 'i32', default: 'f64')
///
/// # Examples
/// ```python
/// import audio_samples as aus
///
/// # Generate impulse at the beginning
/// impulse = aus.impulse(1.0, 44100)
///
/// # Generate impulse at 0.5 seconds
/// delayed_impulse = aus.impulse(2.0, 44100, position=0.5)
/// ```
#[pyfunction(name = "impulse")]
#[pyo3(signature = (duration, sample_rate, *, amplitude=1.0, position=0.0, dtype=None))]
fn py_impulse(
    _py: Python,
    duration: f64,
    sample_rate: u32,
    amplitude: f64,
    position: f64,
    dtype: Option<&Bound<PyAny>>,
) -> PyResult<PyAudioSamples> {
    let target_type = parse_dtype_param(dtype)?;

    let data = match target_type {
        crate::python::conversions::TargetType::F64 => {
            let inner = impulse::<f64>(duration, sample_rate, amplitude, position)
                .map_err(utils::map_error)?;
            AudioSamplesData::F64(inner)
        }
        crate::python::conversions::TargetType::F32 => {
            let inner = impulse::<f32>(duration, sample_rate, amplitude, position)
                .map_err(utils::map_error)?;
            AudioSamplesData::F32(inner)
        }
        crate::python::conversions::TargetType::I32 => {
            let inner = impulse::<i32>(duration, sample_rate, amplitude, position)
                .map_err(utils::map_error)?;
            AudioSamplesData::I32(inner)
        }
        crate::python::conversions::TargetType::I16 => {
            let inner = impulse::<i16>(duration, sample_rate, amplitude, position)
                .map_err(utils::map_error)?;
            AudioSamplesData::I16(inner)
        }
    };

    Ok(PyAudioSamples { data })
}

/// Generate silence (zeros) with the specified duration.
///
/// # Arguments
/// * `duration` - Duration of the signal in seconds
/// * `sample_rate` - Sample rate in Hz
/// * `dtype` - Data type (numpy.dtype or string like 'f32', 'f64', 'i16', 'i32', default: 'f64')
///
/// # Examples
/// ```python
/// import audio_samples as aus
///
/// # Generate 5 seconds of silence
/// silence = aus.silence(5.0, 44100)
/// ```
#[pyfunction(name = "silence")]
#[pyo3(signature = (duration, sample_rate, *, dtype=None))]
fn py_silence(
    _py: Python,
    duration: f64,
    sample_rate: u32,
    dtype: Option<&Bound<PyAny>>,
) -> PyResult<PyAudioSamples> {
    let target_type = parse_dtype_param(dtype)?;

    let data = match target_type {
        crate::python::conversions::TargetType::F64 => {
            let inner = silence::<f64>(duration, sample_rate)
                .map_err(utils::map_error)?;
            AudioSamplesData::F64(inner)
        }
        crate::python::conversions::TargetType::F32 => {
            let inner = silence::<f32>(duration, sample_rate)
                .map_err(utils::map_error)?;
            AudioSamplesData::F32(inner)
        }
        crate::python::conversions::TargetType::I32 => {
            let inner = silence::<i32>(duration, sample_rate)
                .map_err(utils::map_error)?;
            AudioSamplesData::I32(inner)
        }
        crate::python::conversions::TargetType::I16 => {
            let inner = silence::<i16>(duration, sample_rate)
                .map_err(utils::map_error)?;
            AudioSamplesData::I16(inner)
        }
    };

    Ok(PyAudioSamples { data })
}

/// Helper function to parse dtype parameter
fn parse_dtype_param(dtype: Option<&Bound<PyAny>>) -> PyResult<crate::python::conversions::TargetType> {
    if let Some(dt) = dtype {
        if let Ok(numpy_dtype) = dt.downcast::<numpy::PyArrayDescr>() {
            crate::python::conversions::get_target_type(numpy_dtype)
        } else if let Ok(dtype_str) = dt.extract::<String>() {
            utils::validate_string_param("dtype", &dtype_str, &["f64", "f32", "i32", "i16"])?;
            Ok(match dtype_str.as_str() {
                "f64" => crate::python::conversions::TargetType::F64,
                "f32" => crate::python::conversions::TargetType::F32,
                "i32" => crate::python::conversions::TargetType::I32,
                "i16" => crate::python::conversions::TargetType::I16,
                _ => unreachable!(),
            })
        } else {
            Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                "dtype must be a numpy dtype or string ('f64', 'f32', 'i32', 'i16')",
            ))
        }
    } else {
        Ok(crate::python::conversions::TargetType::F64)
    }
}

/// Register the generation functions in the Python module.
pub fn register_functions(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_function(pyo3::wrap_pyfunction!(py_sine_wave, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(py_cosine_wave, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(py_white_noise, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(py_pink_noise, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(py_square_wave, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(py_sawtooth_wave, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(py_triangle_wave, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(py_chirp, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(py_impulse, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(py_silence, m)?)?;
    
    Ok(())
}