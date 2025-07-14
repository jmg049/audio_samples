//! AudioStatistics trait implementation for Python bindings.
//!
//! This module provides Python access to statistical analysis operations
//! for audio data, including RMS, variance, zero crossings, and correlation analysis.

use super::{PyAudioSamples, utils::*};
use crate::operations::{AudioStatistics, AudioTypeConversion};
use pyo3::prelude::*;

impl PyAudioSamples {
    /// Return the peak (maximum absolute value) in the audio samples.
    ///
    /// This is useful for preventing clipping and measuring signal levels.
    pub(crate) fn peak_impl(&self) -> f64 {
        match &self.data {
            crate::python::AudioSamplesData::F64(audio) => audio.peak(),
            crate::python::AudioSamplesData::F32(audio) => audio.peak() as f64,
            crate::python::AudioSamplesData::I32(audio) => audio.peak() as f64,
            crate::python::AudioSamplesData::I16(audio) => audio.peak() as f64,
            crate::python::AudioSamplesData::I24(audio) => {
                // Convert to f64 for statistics since I24 doesn't implement needed traits
                match audio.to_f64() {
                    Ok(f64_audio) => f64_audio.peak(),
                    Err(_) => 0.0, // Fallback value
                }
            }
        }
    }

    /// Return the minimum value in the audio samples.
    ///
    /// # Returns
    /// Minimum sample value as float
    ///
    /// # Examples
    /// ```python
    /// audio = aus.from_numpy(np.array([0.5, -0.8, 0.3, -0.2]), sample_rate=44100)
    /// min_val = audio.min()  # Returns -0.8
    /// ```
    pub(crate) fn min_impl(&self) -> f64 {
        match &self.data {
            crate::python::AudioSamplesData::F64(audio) => audio.min(),
            crate::python::AudioSamplesData::F32(audio) => audio.min() as f64,
            crate::python::AudioSamplesData::I32(audio) => audio.min() as f64,
            crate::python::AudioSamplesData::I16(audio) => audio.min() as f64,
            crate::python::AudioSamplesData::I24(audio) => {
                // Convert to f64 for statistics since I24 doesn't implement needed traits
                match audio.to_f64() {
                    Ok(f64_audio) => f64_audio.min(),
                    Err(_) => 0.0, // Fallback value
                }
            }
        }
    }

    /// Return the maximum value in the audio samples.
    ///
    /// # Returns
    /// Maximum sample value as float
    ///
    /// # Examples
    /// ```python
    /// audio = aus.from_numpy(np.array([0.5, -0.8, 0.3, -0.2]), sample_rate=44100)
    /// max_val = audio.max()  # Returns 0.5
    /// ```
    pub(crate) fn max_impl(&self) -> f64 {
        match &self.data {
            crate::python::AudioSamplesData::F64(audio) => audio.max(),
            crate::python::AudioSamplesData::F32(audio) => audio.max() as f64,
            crate::python::AudioSamplesData::I32(audio) => audio.max() as f64,
            crate::python::AudioSamplesData::I16(audio) => audio.max() as f64,
            crate::python::AudioSamplesData::I24(audio) => {
                // Convert to f64 for statistics since I24 doesn't implement needed traits
                match audio.to_f64() {
                    Ok(f64_audio) => f64_audio.max(),
                    Err(_) => 0.0, // Fallback value
                }
            }
        }
    }

    /// Compute the Root Mean Square (RMS) of the audio samples.
    ///
    /// RMS is useful for measuring average signal power/energy and
    /// provides a perceptually relevant measure of loudness.
    ///
    /// # Returns
    /// RMS value as float
    ///
    /// # Examples
    /// ```python
    /// # Create sine wave
    /// t = np.linspace(0, 1, 44100)
    /// sine = np.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave
    /// audio = aus.from_numpy(sine, sample_rate=44100)
    /// rms = audio.rms()  # Should be approximately 0.707 for unit sine
    /// ```
    pub(crate) fn rms_impl(&self) -> f64 {
        self.with_inner(|inner| Ok(inner.rms()))
            .map_err(map_error)
            .unwrap_or(0.0)
    }

    /// Compute the statistical variance of the audio samples.
    ///
    /// Variance measures the spread of sample values around the mean.
    ///
    /// # Returns
    /// Variance as float
    ///
    /// # Examples
    /// ```python
    /// audio = aus.from_numpy(np.random.normal(0, 0.1, 44100), sample_rate=44100)
    /// var = audio.variance()  # Should be close to 0.01 for std=0.1
    /// ```
    pub(crate) fn variance_impl(&self) -> f64 {
        self.with_inner(|inner| Ok(inner.variance()))
            .map_err(map_error)
            .unwrap_or(0.0)
    }

    /// Compute the standard deviation of the audio samples.
    ///
    /// Standard deviation is the square root of variance and provides
    /// a measure of signal variability in the same units as the samples.
    ///
    /// # Returns
    /// Standard deviation as float
    ///
    /// # Examples
    /// ```python
    /// audio = aus.from_numpy(np.random.normal(0, 0.1, 44100), sample_rate=44100)
    /// std = audio.std_dev()  # Should be close to 0.1
    /// ```
    pub(crate) fn std_dev_impl(&self) -> f64 {
        self.with_inner(|inner| Ok(inner.std_dev()))
            .map_err(map_error)
            .unwrap_or(0.0)
    }

    /// Count the number of zero crossings in the audio signal.
    ///
    /// Zero crossings are useful for pitch detection and signal analysis.
    /// The count represents transitions from positive to negative values or vice versa.
    ///
    /// # Returns
    /// Number of zero crossings as integer
    ///
    /// # Examples
    /// ```python
    /// # Create square wave (many zero crossings)
    /// square = np.sign(np.sin(2 * np.pi * 440 * np.linspace(0, 1, 44100)))
    /// audio = aus.from_numpy(square, sample_rate=44100)
    /// crossings = audio.zero_crossings()  # Should be close to 880 (2 * frequency)
    /// ```
    pub(crate) fn zero_crossings_impl(&self) -> usize {
        self.with_inner(|inner| Ok(inner.zero_crossings()))
            .map_err(map_error)
            .unwrap_or(0)
    }

    /// Compute the zero crossing rate (crossings per second).
    ///
    /// This normalizes the zero crossing count by the signal duration,
    /// making it independent of audio length.
    ///
    /// # Returns
    /// Zero crossing rate in Hz as float
    ///
    /// # Examples
    /// ```python
    /// # 440 Hz sine wave should have ~880 crossings per second
    /// t = np.linspace(0, 1, 44100)
    /// sine = np.sin(2 * np.pi * 440 * t)
    /// audio = aus.from_numpy(sine, sample_rate=44100)
    /// zcr = audio.zero_crossing_rate()  # Should be close to 880
    /// ```
    pub(crate) fn zero_crossing_rate_impl(&self) -> f64 {
        self.with_inner(|inner| Ok(inner.zero_crossing_rate()))
            .map_err(map_error)
            .unwrap_or(0.0)
    }

    /// Compute the autocorrelation function up to max_lag samples.
    ///
    /// Returns a numpy array of correlation values for each lag offset.
    /// Useful for pitch detection and periodicity analysis.
    ///
    /// # Arguments
    /// * `max_lag` - Maximum lag to compute in samples (default: length/4)
    /// * `normalize` - Whether to normalize correlations to [-1, 1] range
    ///
    /// # Returns
    /// NumPy array of autocorrelation values
    ///
    /// # Examples
    /// ```python
    /// # Periodic signal should show peaks at period intervals
    /// t = np.linspace(0, 1, 44100)
    /// periodic = np.sin(2 * np.pi * 440 * t)  # 440 Hz = period of 100 samples at 44.1kHz
    /// audio = aus.from_numpy(periodic, sample_rate=44100)
    /// autocorr = audio.autocorrelation(max_lag=500)
    /// # Should show peak around lag 100
    /// ```
    pub(crate) fn autocorrelation_impl(
        &self,
        py: Python,
        max_lag: Option<usize>,
        normalize: bool,
    ) -> PyResult<PyObject> {
        let actual_max_lag = max_lag.unwrap_or(self.length() / 4);
        let autocorr = self
            .with_inner(|inner| inner.autocorrelation(actual_max_lag))
            .map_err(map_error)?;

        if normalize {
            // Normalize by the zero-lag value (first element)
            let normalized: Vec<f64> = if autocorr.is_empty() || autocorr[0] == 0.0 {
                autocorr
            } else {
                let zero_lag = autocorr[0];
                autocorr.into_iter().map(|x| x / zero_lag).collect()
            };
            array1_to_numpy(py, ndarray::Array1::from(normalized))
        } else {
            array1_to_numpy(py, ndarray::Array1::from(autocorr))
        }
    }

    /// Compute cross-correlation with another audio signal.
    ///
    /// Returns correlation values for each lag offset between the two signals.
    /// Useful for alignment, synchronization, and similarity analysis.
    ///
    /// # Arguments
    /// * `other` - The other AudioSamples object to correlate with
    /// * `max_lag` - Maximum lag to compute in samples (default: min(length1, length2)/4)
    /// * `normalize` - Whether to normalize correlations to [-1, 1] range
    ///
    /// # Returns
    /// NumPy array of cross-correlation values
    ///
    /// # Examples
    /// ```python
    /// # Create original and delayed signals
    /// t = np.linspace(0, 1, 44100)
    /// original = np.sin(2 * np.pi * 440 * t)
    /// delayed = np.concatenate([np.zeros(100), original[:-100]])  # 100 sample delay
    ///
    /// audio1 = aus.from_numpy(original, sample_rate=44100)
    /// audio2 = aus.from_numpy(delayed, sample_rate=44100)
    /// xcorr = audio1.cross_correlation(audio2, max_lag=200)
    /// # Should show peak around lag 100
    /// ```
    pub(crate) fn cross_correlation_impl(
        &self,
        py: Python,
        other: &PyAudioSamples,
        max_lag: Option<usize>,
        normalize: bool,
    ) -> PyResult<PyObject> {
        // Check compatibility
        if self.sample_rate() != other.sample_rate() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Sample rates must match for cross-correlation",
            ));
        }

        if self.channels() != other.channels() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Channel counts must match for cross-correlation",
            ));
        }

        let actual_max_lag = max_lag.unwrap_or(std::cmp::min(self.length(), other.length()) / 4);

        let self_f64 = self.as_f64().map_err(map_error)?;
        let other_f64 = other.as_f64().map_err(map_error)?;
        let xcorr = self_f64
            .cross_correlation(&other_f64, actual_max_lag)
            .map_err(map_error)?;

        if normalize {
            // Normalize by the geometric mean of the zero-lag autocorrelations
            let self_autocorr = self_f64.autocorrelation(0).map_err(map_error)?;
            let other_autocorr = other_f64.autocorrelation(0).map_err(map_error)?;

            if !self_autocorr.is_empty()
                && !other_autocorr.is_empty()
                && self_autocorr[0] > 0.0
                && other_autocorr[0] > 0.0
            {
                let normalization = (self_autocorr[0] * other_autocorr[0]).sqrt();
                let normalized: Vec<f64> = xcorr.into_iter().map(|x| x / normalization).collect();
                array1_to_numpy(py, ndarray::Array1::from(normalized))
            } else {
                array1_to_numpy(py, ndarray::Array1::from(xcorr))
            }
        } else {
            array1_to_numpy(py, ndarray::Array1::from(xcorr))
        }
    }

    /// Compute the spectral centroid (brightness measure).
    ///
    /// The spectral centroid represents the "center of mass" of the spectrum
    /// and is often used as a measure of brightness or timbre.
    ///
    /// # Returns
    /// Spectral centroid in Hz as float
    ///
    /// # Examples
    /// ```python
    /// # High frequency content should have high centroid
    /// t = np.linspace(0, 1, 44100)
    /// high_freq = np.sin(2 * np.pi * 4000 * t)  # 4kHz sine
    /// audio = aus.from_numpy(high_freq, sample_rate=44100)
    /// centroid = audio.spectral_centroid()  # Should be around 4000 Hz
    /// ```
    pub(crate) fn spectral_centroid_impl(&self) -> PyResult<f64> {
        self.with_inner(|inner| inner.spectral_centroid())
            .map_err(map_error)
    }

    /// Compute spectral rolloff frequency.
    ///
    /// The rolloff frequency is the frequency below which a specified percentage
    /// of the total spectral energy is contained.
    ///
    /// # Arguments
    /// * `rolloff_percent` - Percentage of energy (0.0 to 1.0, default: 0.85)
    ///
    /// # Returns
    /// Rolloff frequency in Hz as float
    ///
    /// # Examples
    /// ```python
    /// # White noise should have rolloff around 85% of Nyquist frequency
    /// noise = np.random.normal(0, 0.1, 44100)
    /// audio = aus.from_numpy(noise, sample_rate=44100)
    /// rolloff = audio.spectral_rolloff(rolloff_percent=0.85)
    /// # Should be around 0.85 * 22050 = 18742 Hz for white noise
    /// ```
    pub(crate) fn spectral_rolloff_impl(&self, rolloff_percent: f64) -> PyResult<f64> {
        if rolloff_percent <= 0.0 || rolloff_percent >= 1.0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "rolloff_percent must be between 0.0 and 1.0",
            ));
        }
        self.with_inner(|inner| inner.spectral_rolloff(rolloff_percent))
            .map_err(map_error)
    }
}

/// Additional statistics methods for PyAudioSamples using multiple-pymethods feature
#[pymethods]
impl PyAudioSamples {
    // These methods are already declared in mod.rs, so we don't need to redeclare them here
    // The multiple-pymethods feature will automatically combine them
}
