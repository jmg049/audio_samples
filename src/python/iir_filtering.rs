//! AudioIirFiltering trait implementation for Python bindings.
//!
//! This module provides Python access to Infinite Impulse Response (IIR) digital
//! filter design and application, including Butterworth and Chebyshev filter types
//! with comprehensive frequency response analysis capabilities.

use super::{PyAudioSamples, utils::*};
use crate::operations::{
    AudioIirFiltering,
    types::{IirFilterDesign, FilterType, FilterResponse, ButterworthConfig, ChebyshevConfig},
};
use pyo3::prelude::*;

impl PyAudioSamples {
    /// Apply a Butterworth IIR filter to the audio signal.
    ///
    /// Butterworth filters provide maximally flat frequency response in the passband
    /// with no ripple, making them ideal for general-purpose filtering applications.
    ///
    /// # Arguments
    /// * `cutoff` - Cutoff frequency in Hz
    /// * `order` - Filter order (higher = steeper rolloff, default: 4)
    /// * `filter_type` - Filter type ('lowpass', 'highpass', 'bandpass', 'bandstop', default: 'lowpass')
    /// * `high_cutoff` - High cutoff frequency for bandpass/bandstop filters (Hz)
    ///
    /// # Returns
    /// New AudioSamples object with filtering applied
    ///
    /// # Examples
    /// ```python
    /// import audio_samples as aus
    /// import numpy as np
    ///
    /// # Create test signal with multiple frequencies
    /// t = np.linspace(0, 1, 44100)
    /// signal = (np.sin(2 * np.pi * 440 * t) +  # 440 Hz
    ///           np.sin(2 * np.pi * 1000 * t) +  # 1 kHz
    ///           np.sin(2 * np.pi * 5000 * t))   # 5 kHz
    /// audio = aus.from_numpy(signal, sample_rate=44100)
    ///
    /// # Low-pass filter to remove high frequencies
    /// lowpass = audio.butterworth_filter(cutoff=2000, order=6, filter_type='lowpass')
    ///
    /// # High-pass filter to remove low frequencies
    /// highpass = audio.butterworth_filter(cutoff=800, order=4, filter_type='highpass')
    ///
    /// # Band-pass filter to isolate middle frequencies
    /// bandpass = audio.butterworth_filter(
    ///     cutoff=500, high_cutoff=2000, order=4, filter_type='bandpass'
    /// )
    ///
    /// # Band-stop (notch) filter to remove specific frequency range
    /// notch = audio.butterworth_filter(
    ///     cutoff=950, high_cutoff=1050, order=8, filter_type='bandstop'
    /// )
    /// ```
    pub(crate) fn butterworth_filter_impl(
        &self,
        cutoff: f64,
        order: Option<usize>,
        filter_type: Option<&str>,
        high_cutoff: Option<f64>,
    ) -> PyResult<PyAudioSamples> {
        let filter_type_enum = if let Some(ftype) = filter_type {
            validate_string_param("filter_type", ftype, &["lowpass", "highpass", "bandpass", "bandstop"])?;
            match ftype {
                "lowpass" => FilterType::LowPass,
                "highpass" => FilterType::HighPass,
                "bandpass" => FilterType::BandPass,
                "bandstop" => FilterType::BandStop,
                _ => unreachable!(),
            }
        } else {
            FilterType::LowPass
        };

        // Validate frequency parameters
        let nyquist = self.sample_rate() as f64 / 2.0;
        if cutoff <= 0.0 || cutoff >= nyquist {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                format!("Cutoff frequency must be between 0 and {} Hz (Nyquist)", nyquist)
            ));
        }

        if let Some(high_freq) = high_cutoff {
            if high_freq <= cutoff || high_freq >= nyquist {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    format!("High cutoff must be between {} and {} Hz", cutoff, nyquist)
                ));
            }
        }

        let config = ButterworthConfig {
            cutoff,
            high_cutoff,
            order: order.unwrap_or(4),
            filter_type: filter_type_enum,
        };

        let filtered = self
            .with_inner(|inner| inner.butterworth_filter(&config))
            .map_err(map_error)?;
        Ok(PyAudioSamples::from_inner(filtered))
    }

    /// Apply a Butterworth IIR filter in-place.
    ///
    /// Same as butterworth_filter() but modifies the current AudioSamples object.
    ///
    /// # Arguments
    /// Same as butterworth_filter()
    ///
    /// # Examples
    /// ```python
    /// # Apply low-pass filter directly
    /// audio.butterworth_filter_(cutoff=1500, order=6)
    /// ```
    pub(crate) fn butterworth_filter_in_place_impl(
        &mut self,
        cutoff: f64,
        order: Option<usize>,
        filter_type: Option<&str>,
        high_cutoff: Option<f64>,
    ) -> PyResult<()> {
        let filter_type_enum = if let Some(ftype) = filter_type {
            validate_string_param("filter_type", ftype, &["lowpass", "highpass", "bandpass", "bandstop"])?;
            match ftype {
                "lowpass" => FilterType::LowPass,
                "highpass" => FilterType::HighPass,
                "bandpass" => FilterType::BandPass,
                "bandstop" => FilterType::BandStop,
                _ => unreachable!(),
            }
        } else {
            FilterType::LowPass
        };

        let nyquist = self.sample_rate() as f64 / 2.0;
        if cutoff <= 0.0 || cutoff >= nyquist {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                format!("Cutoff frequency must be between 0 and {} Hz (Nyquist)", nyquist)
            ));
        }

        if let Some(high_freq) = high_cutoff {
            if high_freq <= cutoff || high_freq >= nyquist {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    format!("High cutoff must be between {} and {} Hz", cutoff, nyquist)
                ));
            }
        }

        let config = ButterworthConfig {
            cutoff,
            high_cutoff,
            order: order.unwrap_or(4),
            filter_type: filter_type_enum,
        };

        self.with_inner_mut(|inner| {
            inner.butterworth_filter_in_place(&config)?;
            Ok(())
        })
        .map_err(map_error)
    }

    /// Apply a Chebyshev Type I IIR filter to the audio signal.
    ///
    /// Chebyshev filters provide steeper rolloff than Butterworth filters
    /// but with passband ripple. Type I has ripple in the passband.
    ///
    /// # Arguments
    /// * `cutoff` - Cutoff frequency in Hz
    /// * `order` - Filter order (default: 4)
    /// * `ripple` - Passband ripple in dB (default: 0.5)
    /// * `filter_type` - Filter type ('lowpass', 'highpass', 'bandpass', 'bandstop', default: 'lowpass')
    /// * `high_cutoff` - High cutoff frequency for bandpass/bandstop filters (Hz)
    ///
    /// # Returns
    /// New AudioSamples object with filtering applied
    ///
    /// # Examples
    /// ```python
    /// # Steep low-pass filter with minimal ripple
    /// steep_lowpass = audio.chebyshev_filter(
    ///     cutoff=1000, order=6, ripple=0.1, filter_type='lowpass'
    /// )
    ///
    /// # High-pass with more aggressive rolloff
    /// aggressive_highpass = audio.chebyshev_filter(
    ///     cutoff=300, order=8, ripple=0.5, filter_type='highpass'
    /// )
    ///
    /// # Narrow band-pass with steep sides
    /// narrow_bandpass = audio.chebyshev_filter(
    ///     cutoff=990, high_cutoff=1010, order=10, ripple=0.2, filter_type='bandpass'
    /// )
    /// ```
    pub(crate) fn chebyshev_filter_impl(
        &self,
        cutoff: f64,
        order: Option<usize>,
        ripple: Option<f64>,
        filter_type: Option<&str>,
        high_cutoff: Option<f64>,
    ) -> PyResult<PyAudioSamples> {
        let filter_type_enum = if let Some(ftype) = filter_type {
            validate_string_param("filter_type", ftype, &["lowpass", "highpass", "bandpass", "bandstop"])?;
            match ftype {
                "lowpass" => FilterType::LowPass,
                "highpass" => FilterType::HighPass,
                "bandpass" => FilterType::BandPass,
                "bandstop" => FilterType::BandStop,
                _ => unreachable!(),
            }
        } else {
            FilterType::LowPass
        };

        let nyquist = self.sample_rate() as f64 / 2.0;
        if cutoff <= 0.0 || cutoff >= nyquist {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                format!("Cutoff frequency must be between 0 and {} Hz (Nyquist)", nyquist)
            ));
        }

        if let Some(high_freq) = high_cutoff {
            if high_freq <= cutoff || high_freq >= nyquist {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    format!("High cutoff must be between {} and {} Hz", cutoff, nyquist)
                ));
            }
        }

        let ripple_db = ripple.unwrap_or(0.5);
        if ripple_db < 0.0 || ripple_db > 10.0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Ripple must be between 0.0 and 10.0 dB"
            ));
        }

        let config = ChebyshevConfig {
            cutoff,
            high_cutoff,
            order: order.unwrap_or(4),
            ripple: ripple_db,
            filter_type: filter_type_enum,
        };

        let filtered = self
            .with_inner(|inner| inner.chebyshev_filter(&config))
            .map_err(map_error)?;
        Ok(PyAudioSamples::from_inner(filtered))
    }

    /// Apply a Chebyshev Type I IIR filter in-place.
    ///
    /// Same as chebyshev_filter() but modifies the current AudioSamples object.
    ///
    /// # Arguments
    /// Same as chebyshev_filter()
    ///
    /// # Examples
    /// ```python
    /// # Apply steep high-pass filter directly
    /// audio.chebyshev_filter_(cutoff=200, order=8, ripple=0.3, filter_type='highpass')
    /// ```
    pub(crate) fn chebyshev_filter_in_place_impl(
        &mut self,
        cutoff: f64,
        order: Option<usize>,
        ripple: Option<f64>,
        filter_type: Option<&str>,
        high_cutoff: Option<f64>,
    ) -> PyResult<()> {
        let filter_type_enum = if let Some(ftype) = filter_type {
            validate_string_param("filter_type", ftype, &["lowpass", "highpass", "bandpass", "bandstop"])?;
            match ftype {
                "lowpass" => FilterType::LowPass,
                "highpass" => FilterType::HighPass,
                "bandpass" => FilterType::BandPass,
                "bandstop" => FilterType::BandStop,
                _ => unreachable!(),
            }
        } else {
            FilterType::LowPass
        };

        let nyquist = self.sample_rate() as f64 / 2.0;
        if cutoff <= 0.0 || cutoff >= nyquist {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                format!("Cutoff frequency must be between 0 and {} Hz (Nyquist)", nyquist)
            ));
        }

        if let Some(high_freq) = high_cutoff {
            if high_freq <= cutoff || high_freq >= nyquist {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    format!("High cutoff must be between {} and {} Hz", cutoff, nyquist)
                ));
            }
        }

        let ripple_db = ripple.unwrap_or(0.5);
        if ripple_db < 0.0 || ripple_db > 10.0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Ripple must be between 0.0 and 10.0 dB"
            ));
        }

        let config = ChebyshevConfig {
            cutoff,
            high_cutoff,
            order: order.unwrap_or(4),
            ripple: ripple_db,
            filter_type: filter_type_enum,
        };

        self.with_inner_mut(|inner| {
            inner.chebyshev_filter_in_place(&config)?;
            Ok(())
        })
        .map_err(map_error)
    }

    /// Compute the frequency response of a digital filter.
    ///
    /// Returns the magnitude and phase response of a filter at specified
    /// frequencies, useful for analyzing filter characteristics and designing
    /// complementary filters.
    ///
    /// # Arguments
    /// * `filter_design` - Filter configuration (use create_butterworth_design or create_chebyshev_design)
    /// * `frequencies` - NumPy array of frequencies in Hz to analyze
    /// * `worN` - Number of frequency points (if frequencies not provided, default: 512)
    ///
    /// # Returns
    /// Tuple of (frequencies, magnitude, phase) as NumPy arrays
    ///
    /// # Examples
    /// ```python
    /// # Analyze Butterworth filter response
    /// freqs = np.logspace(1, 4, 1000)  # 10 Hz to 10 kHz
    /// filter_design = audio.create_butterworth_design(cutoff=1000, order=4)
    /// frequencies, magnitude, phase = audio.frequency_response(filter_design, freqs)
    ///
    /// # Plot frequency response
    /// import matplotlib.pyplot as plt
    /// fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    ///
    /// # Magnitude response
    /// ax1.semilogx(frequencies, 20 * np.log10(magnitude))
    /// ax1.set_ylabel('Magnitude (dB)')
    /// ax1.grid(True)
    ///
    /// # Phase response  
    /// ax2.semilogx(frequencies, np.degrees(phase))
    /// ax2.set_ylabel('Phase (degrees)')
    /// ax2.set_xlabel('Frequency (Hz)')
    /// ax2.grid(True)
    ///
    /// # Analyze filter without providing frequencies (auto-generated)
    /// freq_auto, mag_auto, phase_auto = audio.frequency_response(filter_design, worN=1024)
    /// ```
    pub(crate) fn frequency_response_impl(
        &self,
        py: Python,
        filter_design: &IirFilterDesign,
        frequencies: Option<&Bound<PyAny>>,
        worN: Option<usize>,
    ) -> PyResult<PyObject> {
        let freq_vec = if let Some(freq_array) = frequencies {
            // Convert numpy array to Vec<f64>
            if let Ok(array) = freq_array.extract::<numpy::PyReadonlyArray1<f64>>() {
                array.as_array().to_vec()
            } else {
                return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                    "Frequencies must be a 1D numpy array of float64"
                ));
            }
        } else {
            // Generate logarithmic frequency grid from 1 Hz to Nyquist
            let n_points = worN.unwrap_or(512);
            let nyquist = self.sample_rate() as f64 / 2.0;
            let log_start = 1.0_f64.log10();
            let log_end = nyquist.log10();
            (0..n_points)
                .map(|i| {
                    let log_freq = log_start + (log_end - log_start) * i as f64 / (n_points - 1) as f64;
                    10.0_f64.powf(log_freq)
                })
                .collect()
        };

        let response = self
            .with_inner(|inner| inner.frequency_response(filter_design, &freq_vec))
            .map_err(map_error)?;

        // Extract magnitude and phase
        let (frequencies, magnitude, phase): (Vec<f64>, Vec<f64>, Vec<f64>) = response
            .into_iter()
            .map(|resp| (resp.frequency, resp.magnitude, resp.phase))
            .multiunzip();

        // Convert to numpy arrays
        let freq_array = array1_to_numpy(py, ndarray::Array1::from(frequencies))?;
        let mag_array = array1_to_numpy(py, ndarray::Array1::from(magnitude))?;
        let phase_array = array1_to_numpy(py, ndarray::Array1::from(phase))?;

        Ok(pyo3::types::PyTuple::new(py, &[freq_array, mag_array, phase_array])?.into())
    }

    /// Create a Butterworth filter design for frequency response analysis.
    ///
    /// This helper function creates a filter design object that can be used
    /// with frequency_response() for analysis without applying the filter.
    ///
    /// # Arguments
    /// Same as butterworth_filter()
    ///
    /// # Returns
    /// IirFilterDesign object for use with frequency_response()
    ///
    /// # Examples
    /// ```python
    /// # Create filter design for analysis
    /// design = audio.create_butterworth_design(cutoff=1000, order=6, filter_type='lowpass')
    /// freqs, mag, phase = audio.frequency_response(design)
    /// ```
    pub(crate) fn create_butterworth_design_impl(
        &self,
        cutoff: f64,
        order: Option<usize>,
        filter_type: Option<&str>,
        high_cutoff: Option<f64>,
    ) -> PyResult<IirFilterDesign> {
        let filter_type_enum = if let Some(ftype) = filter_type {
            validate_string_param("filter_type", ftype, &["lowpass", "highpass", "bandpass", "bandstop"])?;
            match ftype {
                "lowpass" => FilterType::LowPass,
                "highpass" => FilterType::HighPass,
                "bandpass" => FilterType::BandPass,
                "bandstop" => FilterType::BandStop,
                _ => unreachable!(),
            }
        } else {
            FilterType::LowPass
        };

        let nyquist = self.sample_rate() as f64 / 2.0;
        if cutoff <= 0.0 || cutoff >= nyquist {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                format!("Cutoff frequency must be between 0 and {} Hz (Nyquist)", nyquist)
            ));
        }

        let config = ButterworthConfig {
            cutoff,
            high_cutoff,
            order: order.unwrap_or(4),
            filter_type: filter_type_enum,
        };

        self.with_inner(|inner| inner.create_butterworth_design(&config))
            .map_err(map_error)
    }

    /// Create a Chebyshev filter design for frequency response analysis.
    ///
    /// This helper function creates a filter design object that can be used
    /// with frequency_response() for analysis without applying the filter.
    ///
    /// # Arguments
    /// Same as chebyshev_filter()
    ///
    /// # Returns
    /// IirFilterDesign object for use with frequency_response()
    ///
    /// # Examples
    /// ```python
    /// # Create filter design for analysis
    /// design = audio.create_chebyshev_design(
    ///     cutoff=2000, order=8, ripple=0.5, filter_type='lowpass'
    /// )
    /// freqs, mag, phase = audio.frequency_response(design)
    /// ```
    pub(crate) fn create_chebyshev_design_impl(
        &self,
        cutoff: f64,
        order: Option<usize>,
        ripple: Option<f64>,
        filter_type: Option<&str>,
        high_cutoff: Option<f64>,
    ) -> PyResult<IirFilterDesign> {
        let filter_type_enum = if let Some(ftype) = filter_type {
            validate_string_param("filter_type", ftype, &["lowpass", "highpass", "bandpass", "bandstop"])?;
            match ftype {
                "lowpass" => FilterType::LowPass,
                "highpass" => FilterType::HighPass,
                "bandpass" => FilterType::BandPass,
                "bandstop" => FilterType::BandStop,
                _ => unreachable!(),
            }
        } else {
            FilterType::LowPass
        };

        let nyquist = self.sample_rate() as f64 / 2.0;
        if cutoff <= 0.0 || cutoff >= nyquist {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                format!("Cutoff frequency must be between 0 and {} Hz (Nyquist)", nyquist)
            ));
        }

        let ripple_db = ripple.unwrap_or(0.5);
        if ripple_db < 0.0 || ripple_db > 10.0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Ripple must be between 0.0 and 10.0 dB"
            ));
        }

        let config = ChebyshevConfig {
            cutoff,
            high_cutoff,
            order: order.unwrap_or(4),
            ripple: ripple_db,
            filter_type: filter_type_enum,
        };

        self.with_inner(|inner| inner.create_chebyshev_design(&config))
            .map_err(map_error)
    }

    /// Design and apply a custom IIR filter using direct coefficient specification.
    ///
    /// For advanced users who want to specify filter coefficients directly,
    /// useful for implementing custom filter designs or porting filters from
    /// other systems.
    ///
    /// # Arguments
    /// * `b` - Numerator coefficients (feedforward) as NumPy array
    /// * `a` - Denominator coefficients (feedback) as NumPy array
    /// * `zi` - Initial conditions for filter delay elements (optional)
    ///
    /// # Returns
    /// New AudioSamples object with custom filtering applied
    ///
    /// # Examples
    /// ```python
    /// # Design a simple first-order low-pass filter
    /// # H(z) = (1-α) / (1 - α*z^-1) where α = exp(-2π*fc/fs)
    /// import numpy as np
    /// 
    /// fc = 1000  # Cutoff frequency
    /// fs = audio.sample_rate
    /// alpha = np.exp(-2 * np.pi * fc / fs)
    /// 
    /// b = np.array([1 - alpha])  # Numerator
    /// a = np.array([1, -alpha])  # Denominator
    /// 
    /// filtered = audio.custom_iir_filter(b=b, a=a)
    ///
    /// # Apply with initial conditions
    /// zi = np.zeros(max(len(b), len(a)) - 1)
    /// filtered_ic = audio.custom_iir_filter(b=b, a=a, zi=zi)
    /// ```
    pub(crate) fn custom_iir_filter_impl(
        &self,
        b: &Bound<PyAny>,
        a: &Bound<PyAny>,
        zi: Option<&Bound<PyAny>>,
    ) -> PyResult<PyAudioSamples> {
        // Convert numerator coefficients
        let b_coeffs = if let Ok(array) = b.extract::<numpy::PyReadonlyArray1<f64>>() {
            array.as_array().to_vec()
        } else {
            return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                "Numerator coefficients (b) must be a 1D numpy array of float64"
            ));
        };

        // Convert denominator coefficients
        let a_coeffs = if let Ok(array) = a.extract::<numpy::PyReadonlyArray1<f64>>() {
            array.as_array().to_vec()
        } else {
            return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                "Denominator coefficients (a) must be a 1D numpy array of float64"
            ));
        };

        // Convert initial conditions if provided
        let zi_vec = if let Some(zi_array) = zi {
            if let Ok(array) = zi_array.extract::<numpy::PyReadonlyArray1<f64>>() {
                Some(array.as_array().to_vec())
            } else {
                return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                    "Initial conditions (zi) must be a 1D numpy array of float64"
                ));
            }
        } else {
            None
        };

        let filtered = self
            .with_inner(|inner| inner.custom_iir_filter(&b_coeffs, &a_coeffs, zi_vec.as_deref()))
            .map_err(map_error)?;
        Ok(PyAudioSamples::from_inner(filtered))
    }
}

// Helper trait extensions for multi-unzip functionality
trait MultiUnzip<A, B, C> {
    fn multiunzip(self) -> (Vec<A>, Vec<B>, Vec<C>);
}

impl<A, B, C, I> MultiUnzip<A, B, C> for I
where
    I: Iterator<Item = (A, B, C)>,
{
    fn multiunzip(self) -> (Vec<A>, Vec<B>, Vec<C>) {
        let mut a_vec = Vec::new();
        let mut b_vec = Vec::new();
        let mut c_vec = Vec::new();
        
        for (a, b, c) in self {
            a_vec.push(a);
            b_vec.push(b);
            c_vec.push(c);
        }
        
        (a_vec, b_vec, c_vec)
    }
}