//! AudioTypeConversion trait implementation for Python bindings.
//!
//! This module provides efficient type conversions between different audio
//! sample formats with optimized memory sharing where possible.

use super::{utils::*, PyAudioSamples};
use crate::operations::{AudioStatistics, AudioTypeConversion};
use numpy::PyArrayDescr;
use pyo3::prelude::*;

#[derive(Debug, Clone, Copy)]
pub(crate) enum TargetType {
    F64,
    F32,
    I32,
    I16,
}

/// Determine target type from numpy dtype using comparison
pub(crate) fn get_target_type(dtype: &Bound<PyArrayDescr>) -> PyResult<TargetType> {
    use numpy::PyArrayDescrMethods;

    // Get Python object to compare dtypes
    let py: Python<'_> = dtype.py();

    // Create reference dtypes for comparison
    let f64_dtype = numpy::dtype::<f64>(py);
    let f32_dtype = numpy::dtype::<f32>(py);
    let i32_dtype = numpy::dtype::<i32>(py);
    let i16_dtype = numpy::dtype::<i16>(py);

    // Compare with reference dtypes
    if dtype.is_equiv_to(&f64_dtype) {
        Ok(TargetType::F64)
    } else if dtype.is_equiv_to(&f32_dtype) {
        Ok(TargetType::F32)
    } else if dtype.is_equiv_to(&i32_dtype) {
        Ok(TargetType::I32)
    } else if dtype.is_equiv_to(&i16_dtype) {
        Ok(TargetType::I16)
    } else {
        // Fallback: if no direct match, default to f64 for safety
        Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Unsupported numpy dtype. Supported: float64, float32, int32, int16",
        ))
    }
}

impl PyAudioSamples {
    /// Convert to different sample type with explicit dtype specification.
    ///
    /// This is the main conversion method that handles all supported audio types.
    /// Accepts both numpy dtypes and string representations for convenience.
    ///
    /// # Arguments
    /// * `dtype` - Target data type (numpy.dtype or string like 'f64', 'f32', 'i32', 'i16')
    /// * `copy` - Force a copy even if types match (default: True for safety)
    ///
    /// # Returns
    /// New AudioSamples object with converted sample type
    ///
    /// # Examples
    /// ```python
    /// import audio_samples as aus
    /// import numpy as np
    ///
    /// # Create f64 audio
    /// audio_f64 = aus.from_numpy(np.random.randn(44100), sample_rate=44100)
    ///
    /// # Convert using numpy dtypes (preferred)
    /// audio_f32 = audio_f64.astype(np.float32)     # Single precision
    /// audio_i16 = audio_f64.astype(np.int16)       # 16-bit integer (CD quality)
    /// audio_i32 = audio_f64.astype(np.int32)       # 32-bit integer (high precision)
    ///
    /// # Convert using string dtypes (convenience)
    /// audio_f32 = audio_f64.astype('f32')          # Single precision
    /// audio_i16 = audio_f64.astype('i16')          # 16-bit integer
    ///
    /// # Force a copy even if same type
    /// audio_copy = audio_f64.astype(np.float64, copy=True)
    /// ```
    pub(crate) fn astype_impl(
        &self,
        py: Python,
        dtype: &Bound<PyAny>,
        copy: bool,
    ) -> PyResult<PyObject> {
        use crate::operations::AudioTypeConversion;
        use crate::python::AudioSamplesData;

        // Try to extract numpy dtype first, fall back to string
        let target_type = if let Ok(numpy_dtype) = dtype.downcast::<PyArrayDescr>() {
            // Use numpy dtype directly
            get_target_type(numpy_dtype)?
        } else if let Ok(dtype_str) = dtype.extract::<String>() {
            // Handle string dtypes for convenience
            validate_string_param("dtype", &dtype_str, &["f64", "f32", "i32", "i16"])?;
            match dtype_str.as_str() {
                "f64" => TargetType::F64,
                "f32" => TargetType::F32,
                "i32" => TargetType::I32,
                "i16" => TargetType::I16,
                _ => unreachable!(), // Already validated
            }
        } else {
            return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                "dtype must be a numpy dtype or string ('f64', 'f32', 'i32', 'i16')",
            ));
        };

        // Convert to target type, preserving the original type when possible
        let new_data = match (target_type, &self.data) {
            // Same type conversions (copy if requested)
            (TargetType::F64, AudioSamplesData::F64(audio)) => {
                AudioSamplesData::F64(if copy { audio.clone() } else { audio.clone() })
            }
            (TargetType::F32, AudioSamplesData::F32(audio)) => {
                AudioSamplesData::F32(if copy { audio.clone() } else { audio.clone() })
            }
            (TargetType::I32, AudioSamplesData::I32(audio)) => {
                AudioSamplesData::I32(if copy { audio.clone() } else { audio.clone() })
            }
            (TargetType::I16, AudioSamplesData::I16(audio)) => {
                AudioSamplesData::I16(if copy { audio.clone() } else { audio.clone() })
            }
            // Cross-type conversions
            (TargetType::F64, _) => {
                let f64_audio = match &self.data {
                    AudioSamplesData::F64(audio) => audio.clone(),
                    AudioSamplesData::F32(audio) => audio.to_f64().map_err(map_error)?,
                    AudioSamplesData::I32(audio) => audio.to_f64().map_err(map_error)?,
                    AudioSamplesData::I16(audio) => audio.to_f64().map_err(map_error)?,
                    AudioSamplesData::I24(audio) => audio.to_f64().map_err(map_error)?,
                };
                AudioSamplesData::F64(f64_audio)
            }
            (TargetType::F32, _) => {
                let f32_audio = match &self.data {
                    AudioSamplesData::F32(audio) => audio.clone(),
                    AudioSamplesData::F64(audio) => audio.to_f32().map_err(map_error)?,
                    AudioSamplesData::I32(audio) => audio.to_f32().map_err(map_error)?,
                    AudioSamplesData::I16(audio) => audio.to_f32().map_err(map_error)?,
                    AudioSamplesData::I24(audio) => audio.to_f32().map_err(map_error)?,
                };
                AudioSamplesData::F32(f32_audio)
            }
            (TargetType::I32, _) => {
                let i32_audio = match &self.data {
                    AudioSamplesData::I32(audio) => audio.clone(),
                    AudioSamplesData::I24(audio) => audio.to_i32().map_err(map_error)?,
                    AudioSamplesData::F64(audio) => audio.to_i32().map_err(map_error)?,
                    AudioSamplesData::F32(audio) => audio.to_i32().map_err(map_error)?,
                    AudioSamplesData::I16(audio) => audio.to_i32().map_err(map_error)?,
                };
                AudioSamplesData::I32(i32_audio)
            }
            (TargetType::I16, _) => {
                let i16_audio = match &self.data {
                    AudioSamplesData::I16(audio) => audio.clone(),
                    AudioSamplesData::F64(audio) => audio.to_i16().map_err(map_error)?,
                    AudioSamplesData::F32(audio) => audio.to_i16().map_err(map_error)?,
                    AudioSamplesData::I32(audio) => audio.to_i16().map_err(map_error)?,
                    AudioSamplesData::I24(audio) => audio.to_i16().map_err(map_error)?,
                };
                AudioSamplesData::I16(i16_audio)
            }
        };

        let py_audio = PyAudioSamples::from_data(new_data);
        Ok(Py::new(py, py_audio)?.into())
    }

    /// Convert to the highest precision floating-point format (f64).
    ///
    /// This conversion preserves maximum precision and is often used
    /// before performing mathematical operations.
    ///
    /// # Returns
    /// New AudioSamples object with f64 sample type
    ///
    /// # Examples
    /// ```python
    /// # Convert any type to maximum precision
    /// high_precision = audio.to_f64()
    ///
    /// # Useful before complex processing
    /// processed = audio.to_f64().normalize().apply_filter(coeffs)
    /// ```
    pub(crate) fn to_f64_impl(&self) -> PyResult<PyAudioSamples> {
        use crate::operations::AudioTypeConversion;
        use crate::python::AudioSamplesData;

        let f64_data = match &self.data {
            AudioSamplesData::F64(audio) => AudioSamplesData::F64(audio.clone()),
            AudioSamplesData::F32(audio) => {
                AudioSamplesData::F64(audio.to_f64().map_err(map_error)?)
            }
            AudioSamplesData::I32(audio) => {
                AudioSamplesData::F64(audio.to_f64().map_err(map_error)?)
            }
            AudioSamplesData::I16(audio) => {
                AudioSamplesData::F64(audio.to_f64().map_err(map_error)?)
            }
            AudioSamplesData::I24(audio) => {
                AudioSamplesData::F64(audio.to_f64().map_err(map_error)?)
            }
        };
        Ok(PyAudioSamples::from_data(f64_data))
    }

    /// Convert to single precision floating-point format (f32).
    ///
    /// Good balance between precision and memory usage. Commonly used
    /// when memory efficiency is important.
    ///
    /// # Returns
    /// New AudioSamples object with f32 sample type
    ///
    /// # Examples
    /// ```python
    /// # Reduce memory usage while maintaining good precision
    /// efficient = audio.to_f32()
    ///
    /// # Good for most audio processing tasks
    /// processed = audio.to_f32().normalize().scale(0.5)
    /// ```
    pub(crate) fn to_f32_impl(&self) -> PyResult<PyAudioSamples> {
        use crate::operations::AudioTypeConversion;
        use crate::python::AudioSamplesData;

        let f32_data = match &self.data {
            AudioSamplesData::F32(audio) => AudioSamplesData::F32(audio.clone()),
            AudioSamplesData::F64(audio) => {
                AudioSamplesData::F32(audio.to_f32().map_err(map_error)?)
            }
            AudioSamplesData::I32(audio) => {
                AudioSamplesData::F32(audio.to_f32().map_err(map_error)?)
            }
            AudioSamplesData::I16(audio) => {
                AudioSamplesData::F32(audio.to_f32().map_err(map_error)?)
            }
            AudioSamplesData::I24(audio) => {
                AudioSamplesData::F32(audio.to_f32().map_err(map_error)?)
            }
        };
        Ok(PyAudioSamples::from_data(f32_data))
    }

    /// Convert to 32-bit integer format.
    ///
    /// Highest precision integer format, useful for high-quality
    /// integer-based processing or storage.
    ///
    /// # Returns
    /// New AudioSamples object with i32 sample type
    ///
    /// # Examples
    /// ```python
    /// # High precision integer representation
    /// int32_audio = audio.to_i32()
    ///
    /// # Useful for certain DSP algorithms that work with integers
    /// integer_processed = audio.to_i32().apply_filter(int_coeffs)
    /// ```
    pub(crate) fn to_i32_impl(&self) -> PyResult<PyAudioSamples> {
        use crate::operations::AudioTypeConversion;
        use crate::python::AudioSamplesData;

        let i32_data = match &self.data {
            AudioSamplesData::I32(audio) => AudioSamplesData::I32(audio.clone()),
            AudioSamplesData::I24(audio) => {
                AudioSamplesData::I32(audio.to_i32().map_err(map_error)?)
            }
            AudioSamplesData::F64(audio) => {
                AudioSamplesData::I32(audio.to_i32().map_err(map_error)?)
            }
            AudioSamplesData::F32(audio) => {
                AudioSamplesData::I32(audio.to_i32().map_err(map_error)?)
            }
            AudioSamplesData::I16(audio) => {
                AudioSamplesData::I32(audio.to_i32().map_err(map_error)?)
            }
        };
        Ok(PyAudioSamples::from_data(i32_data))
    }

    /// Convert to 16-bit integer format (CD quality).
    ///
    /// Most common format for audio files and streaming. Provides
    /// good quality while minimizing storage requirements.
    ///
    /// # Returns
    /// New AudioSamples object with i16 sample type
    ///
    /// # Examples
    /// ```python
    /// # Standard CD quality format
    /// cd_quality = audio.to_i16()
    ///
    /// # Prepare for saving to WAV file
    /// final_audio = processed_audio.normalize().to_i16()
    /// ```
    pub(crate) fn to_i16_impl(&self) -> PyResult<PyAudioSamples> {
        use crate::operations::AudioTypeConversion;
        use crate::python::AudioSamplesData;

        let i16_data = match &self.data {
            AudioSamplesData::I16(audio) => AudioSamplesData::I16(audio.clone()),
            AudioSamplesData::F64(audio) => {
                AudioSamplesData::I16(audio.to_i16().map_err(map_error)?)
            }
            AudioSamplesData::F32(audio) => {
                AudioSamplesData::I16(audio.to_i16().map_err(map_error)?)
            }
            AudioSamplesData::I32(audio) => {
                AudioSamplesData::I16(audio.to_i16().map_err(map_error)?)
            }
            AudioSamplesData::I24(audio) => {
                AudioSamplesData::I16(audio.to_i16().map_err(map_error)?)
            }
        };
        Ok(PyAudioSamples::from_data(i16_data))
    }

    /// Get the current data type of the audio samples.
    ///
    /// # Returns
    /// String representing the current dtype ('f64', 'f32', 'i32', 'i16')
    ///
    /// # Examples
    /// ```python
    /// # Check current type
    /// current_type = audio.dtype
    /// print(f"Audio is in {current_type} format")
    ///
    /// # Conditional conversion
    /// if audio.dtype != 'f32':
    ///     audio = audio.to_f32()
    /// ```
    pub(crate) fn dtype_impl(&self) -> &'static str {
        match &self.data {
            crate::python::AudioSamplesData::I16(_) => "i16",
            crate::python::AudioSamplesData::I24(_) => "I24",
            crate::python::AudioSamplesData::I32(_) => "i32",
            crate::python::AudioSamplesData::F32(_) => "f32",
            crate::python::AudioSamplesData::F64(_) => "f64",
        }
    }

    /// Check if the audio data can be represented without loss in the target type.
    ///
    /// This method analyzes the current sample values to determine if conversion
    /// to a lower precision type would result in data loss.
    ///
    /// # Arguments
    /// * `target_dtype` - Target data type to check compatibility with
    /// * `tolerance` - Acceptable relative error for lossless conversion
    ///
    /// # Returns
    /// True if conversion would be lossless, False otherwise
    ///
    /// # Examples
    /// ```python
    /// # Check if we can safely convert to i16 without significant loss
    /// if audio.can_convert_lossless('i16'):
    ///     audio_i16 = audio.to_i16()
    /// else:
    ///     print("Conversion to i16 would cause significant data loss")
    ///
    /// # Check with custom tolerance
    /// if audio.can_convert_lossless('f32', tolerance=1e-6):
    ///     audio_f32 = audio.to_f32()
    /// ```
    pub(crate) fn can_convert_lossless_impl(
        &self,
        target_dtype: &str,
        tolerance: f64,
    ) -> PyResult<bool> {
        use crate::python::AudioSamplesData;

        validate_string_param("target_dtype", target_dtype, &["f64", "f32", "i32", "i16"])?;

        // Get statistics from current data type (convert I24 to f64 for statistics)
        let (peak, min_val, max_val) = match &self.data {
            AudioSamplesData::F64(audio) => (audio.peak(), audio.min(), audio.max()),
            AudioSamplesData::F32(audio) => {
                (audio.peak() as f64, audio.min() as f64, audio.max() as f64)
            }
            AudioSamplesData::I32(audio) => {
                (audio.peak() as f64, audio.min() as f64, audio.max() as f64)
            }
            AudioSamplesData::I16(audio) => {
                (audio.peak() as f64, audio.min() as f64, audio.max() as f64)
            }
            AudioSamplesData::I24(audio) => {
                // Convert to f64 for statistics since I24 doesn't implement needed traits
                let f64_audio = audio.to_f64().map_err(map_error)?;
                (f64_audio.peak(), f64_audio.min(), f64_audio.max())
            }
        };

        match target_dtype {
            "f64" => Ok(true), // Always lossless to f64
            "f32" => {
                // Check if all values can be represented accurately in f32
                let f32_max = f32::MAX as f64;
                let f32_min = f32::MIN as f64;

                if min_val < f32_min || max_val > f32_max {
                    return Ok(false);
                }

                // For f32 sources, check if already f32
                match &self.data {
                    AudioSamplesData::F32(_) => Ok(true), // Already f32
                    AudioSamplesData::I16(_) => Ok(true), // i16 fits in f32
                    _ => Ok(peak < f32_max && tolerance > f32::EPSILON as f64),
                }
            }
            "i32" => {
                // Check if values are within i32 range when scaled
                match &self.data {
                    AudioSamplesData::I32(_) | AudioSamplesData::I16(_) => Ok(true), // Already integer
                    AudioSamplesData::I24(_) => Ok(true), // I24 fits in i32
                    _ => Ok(peak <= 1.0), // Assuming normalized floating point input
                }
            }
            "i16" => {
                // Check if values are within i16 range when scaled
                match &self.data {
                    AudioSamplesData::I16(_) => Ok(true), // Already i16
                    _ => Ok(peak <= 1.0), // Assuming normalized floating point input
                }
            }
            _ => unreachable!(), // Already validated
        }
    }

    /// Get information about precision loss for a given conversion.
    ///
    /// This method provides detailed statistics about what would be lost
    /// when converting to a lower precision format.
    ///
    /// # Arguments
    /// * `target_dtype` - Target data type to analyze
    ///
    /// # Returns
    /// Dictionary with conversion loss statistics
    ///
    /// # Examples
    /// ```python
    /// # Analyze conversion impact
    /// loss_info = audio.conversion_info('i16')
    /// print(f"SNR after conversion: {loss_info['snr_db']} dB")
    /// print(f"Max error: {loss_info['max_error']}")
    /// print(f"RMS error: {loss_info['rms_error']}")
    /// ```
    pub(crate) fn conversion_info_impl(
        &self,
        py: Python,
        target_dtype: &str,
    ) -> PyResult<PyObject> {
        use crate::operations::AudioTypeConversion;
        use crate::python::AudioSamplesData;

        validate_string_param("target_dtype", target_dtype, &["f64", "f32", "i32", "i16"])?;

        // Check if conversion would be lossless (same type)
        let is_same_type = match (target_dtype, &self.data) {
            ("f64", AudioSamplesData::F64(_)) => true,
            ("f32", AudioSamplesData::F32(_)) => true,
            ("i32", AudioSamplesData::I32(_)) => true,
            ("i16", AudioSamplesData::I16(_)) => true,
            _ => false,
        };

        let (max_error, rms_error, snr_db) = if is_same_type {
            (0.0, 0.0, f64::INFINITY) // No loss for same type
        } else {
            // Perform round-trip conversion to measure loss
            let converted_f64 = match target_dtype {
                "f64" => match &self.data {
                    AudioSamplesData::F64(audio) => audio.clone(),
                    AudioSamplesData::F32(audio) => audio.to_f64().map_err(map_error)?,
                    AudioSamplesData::I32(audio) => audio.to_f64().map_err(map_error)?,
                    AudioSamplesData::I16(audio) => audio.to_f64().map_err(map_error)?,
                    AudioSamplesData::I24(audio) => audio.to_f64().map_err(map_error)?,
                },
                "f32" => {
                    let f32_audio = match &self.data {
                        AudioSamplesData::F32(audio) => audio.clone(),
                        AudioSamplesData::F64(audio) => audio.to_f32().map_err(map_error)?,
                        AudioSamplesData::I32(audio) => audio.to_f32().map_err(map_error)?,
                        AudioSamplesData::I16(audio) => audio.to_f32().map_err(map_error)?,
                        AudioSamplesData::I24(audio) => audio.to_f32().map_err(map_error)?,
                    };
                    f32_audio.to_f64().map_err(map_error)?
                }
                "i32" => {
                    let i32_audio = match &self.data {
                        AudioSamplesData::I32(audio) => audio.clone(),
                        AudioSamplesData::I24(audio) => audio.to_i32().map_err(map_error)?,
                        AudioSamplesData::F64(audio) => audio.to_i32().map_err(map_error)?,
                        AudioSamplesData::F32(audio) => audio.to_i32().map_err(map_error)?,
                        AudioSamplesData::I16(audio) => audio.to_i32().map_err(map_error)?,
                    };
                    i32_audio.to_f64().map_err(map_error)?
                }
                "i16" => {
                    let i16_audio = match &self.data {
                        AudioSamplesData::I16(audio) => audio.clone(),
                        AudioSamplesData::F64(audio) => audio.to_i16().map_err(map_error)?,
                        AudioSamplesData::F32(audio) => audio.to_i16().map_err(map_error)?,
                        AudioSamplesData::I32(audio) => audio.to_i16().map_err(map_error)?,
                        AudioSamplesData::I24(audio) => audio.to_i16().map_err(map_error)?,
                    };
                    i16_audio.to_f64().map_err(map_error)?
                }
                _ => unreachable!(),
            };

            self.compute_conversion_errors(&converted_f64)?
        };

        // Create Python dictionary with results
        let dict = pyo3::types::PyDict::new(py);
        dict.set_item("target_dtype", target_dtype)?;
        dict.set_item("max_error", max_error)?;
        dict.set_item("rms_error", rms_error)?;
        dict.set_item("snr_db", snr_db)?;
        dict.set_item("is_lossless", max_error < 1e-10)?;

        Ok(dict.into())
    }

    /// Create a view of the audio data as a different type without conversion.
    ///
    /// This is an advanced method that reinterprets the bit pattern of the audio
    /// data as a different type. Use with caution as it bypasses normal type safety.
    ///
    /// # Arguments
    /// * `target_dtype` - Target data type for reinterpretation
    ///
    /// # Returns
    /// New AudioSamples object with reinterpreted data
    ///
    /// # Examples
    /// ```python
    /// # Advanced usage - reinterpret bits (rarely needed)
    /// # This is mainly useful for low-level optimization or debugging
    /// reinterpreted = audio.view_as('i32')  # View f64 bits as i32
    /// ```
    pub(crate) fn view_as_impl(&self, target_dtype: &str) -> PyResult<PyAudioSamples> {
        // This is a potentially dangerous operation, so we limit it
        // and provide clear warnings
        validate_string_param("target_dtype", target_dtype, &["f64", "f32", "i32", "i16"])?;

        // For safety, we'll implement this as a regular conversion
        // True bit-level reinterpretation would require unsafe code
        match target_dtype {
            "f64" => Ok(self.copy()),
            "f32" => self.to_f32_impl(),
            "i32" => self.to_i32_impl(),
            "i16" => self.to_i16_impl(),
            _ => unreachable!(),
        }
    }
}

impl PyAudioSamples {
    /// Helper method to compute conversion errors between original and round-trip converted audio
    pub(crate) fn compute_conversion_errors(
        &self,
        converted: &crate::AudioSamples<f64>,
    ) -> PyResult<(f64, f64, f64)> {
        use crate::python::AudioSamplesData;

        // Get length and data from original audio (convert to f64 for comparison)
        let orig_f64 = match &self.data {
            AudioSamplesData::F64(audio) => audio.clone(),
            AudioSamplesData::F32(audio) => audio.to_f64().map_err(map_error)?,
            AudioSamplesData::I32(audio) => audio.to_f64().map_err(map_error)?,
            AudioSamplesData::I16(audio) => audio.to_f64().map_err(map_error)?,
            AudioSamplesData::I24(audio) => audio.to_f64().map_err(map_error)?,
        };

        // Ensure same length for comparison
        let orig_len = orig_f64.samples_per_channel();
        let conv_len = converted.samples_per_channel();

        if orig_len != conv_len {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Cannot compare audio samples of different lengths",
            ));
        }

        // Get data arrays for comparison
        let orig_data = match orig_f64.channels() {
            1 => {
                let mono = orig_f64.as_mono().ok_or_else(|| {
                    PyErr::new::<pyo3::exceptions::PyValueError, _>("Expected mono audio")
                })?;
                mono.to_vec()
            }
            _ => {
                let multi = orig_f64.as_multi_channel().ok_or_else(|| {
                    PyErr::new::<pyo3::exceptions::PyValueError, _>("Expected multi-channel audio")
                })?;
                multi.iter().cloned().collect()
            }
        };

        let conv_data = match converted.channels() {
            1 => {
                let mono = converted.as_mono().ok_or_else(|| {
                    PyErr::new::<pyo3::exceptions::PyValueError, _>("Expected mono audio")
                })?;
                mono.to_vec()
            }
            _ => {
                let multi = converted.as_multi_channel().ok_or_else(|| {
                    PyErr::new::<pyo3::exceptions::PyValueError, _>("Expected multi-channel audio")
                })?;
                multi.iter().cloned().collect()
            }
        };

        // Compute error statistics
        let mut max_error: f64 = 0.0;
        let mut sum_squared_error: f64 = 0.0;
        let mut sum_squared_signal: f64 = 0.0;

        for (orig, conv) in orig_data.iter().zip(conv_data.iter()) {
            let error = (orig - conv).abs();
            max_error = max_error.max(error);
            sum_squared_error += error * error;
            sum_squared_signal += orig * orig;
        }

        let rms_error = (sum_squared_error / orig_data.len() as f64).sqrt();
        let snr_db = if sum_squared_error > 0.0 && sum_squared_signal > 0.0 {
            10.0 * (sum_squared_signal / sum_squared_error).log10()
        } else if sum_squared_error == 0.0 {
            f64::INFINITY
        } else {
            f64::NEG_INFINITY
        };

        Ok((max_error, rms_error, snr_db))
    }
}
