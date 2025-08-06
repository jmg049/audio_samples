//! AudioParametricEq trait implementation for Python bindings.
//!
//! This module provides Python access to professional-grade parametric equalization
//! using the RBJ (Robert Bristow-Johnson) cookbook formulas. Supports multi-band
//! EQ with peak, shelf, and filter band types for precise frequency shaping.

use super::{PyAudioSamples, utils::*};
use crate::operations::{
    AudioParametricEq,
    types::{ParametricEqConfig, EqBand, EqBandType, FilterResponse},
};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

impl PyAudioSamples {
    /// Apply parametric equalization with multiple bands.
    ///
    /// A parametric EQ allows precise control over frequency response using
    /// multiple bands, each with configurable frequency, gain, and Q (bandwidth).
    /// Uses RBJ cookbook formulas for accurate digital filter implementation.
    ///
    /// # Arguments
    /// * `bands` - List of EQ band configurations (dictionaries or tuples)
    /// * `gain` - Global output gain in dB (default: 0.0)
    ///
    /// # Band Configuration
    /// Each band can be specified as:
    /// - Dictionary: `{'type': 'peak', 'freq': 1000, 'gain': 3, 'q': 2}`
    /// - Tuple: `('peak', 1000, 3, 2)` for (type, freq, gain, q)
    ///
    /// Band types:
    /// - 'peak': Boost/cut around center frequency
    /// - 'low_shelf': Boost/cut low frequencies
    /// - 'high_shelf': Boost/cut high frequencies
    /// - 'low_pass': Remove high frequencies
    /// - 'high_pass': Remove low frequencies
    /// - 'notch': Deep cut at center frequency
    /// - 'all_pass': Phase shift without amplitude change
    ///
    /// # Returns
    /// New AudioSamples object with EQ applied
    ///
    /// # Examples
    /// ```python
    /// import audio_samples as aus
    /// import numpy as np
    ///
    /// # Create test audio
    /// t = np.linspace(0, 1, 44100)
    /// signal = np.random.randn(44100) * 0.1  # White noise
    /// audio = aus.from_numpy(signal, sample_rate=44100)
    ///
    /// # Multi-band EQ using dictionaries (preferred)
    /// eq_bands = [
    ///     {'type': 'high_pass', 'freq': 80, 'q': 0.7},           # Remove sub-bass
    ///     {'type': 'low_shelf', 'freq': 200, 'gain': -2, 'q': 0.7},  # Reduce low-mids
    ///     {'type': 'peak', 'freq': 1000, 'gain': 3, 'q': 2},         # Boost presence
    ///     {'type': 'peak', 'freq': 3000, 'gain': -1, 'q': 1},        # Slight cut
    ///     {'type': 'high_shelf', 'freq': 8000, 'gain': 2, 'q': 0.7}, # Brighten highs
    /// ]
    /// equalized = audio.parametric_eq(bands=eq_bands)
    ///
    /// # Vocal EQ chain
    /// vocal_eq = [
    ///     {'type': 'high_pass', 'freq': 100, 'q': 0.7},     # Remove rumble
    ///     {'type': 'peak', 'freq': 200, 'gain': -3, 'q': 1}, # Reduce muddiness
    ///     {'type': 'peak', 'freq': 2500, 'gain': 2, 'q': 1.5}, # Clarity
    ///     {'type': 'peak', 'freq': 5000, 'gain': 1, 'q': 2},   # Presence
    ///     {'type': 'high_shelf', 'freq': 10000, 'gain': 1.5, 'q': 0.7}, # Air
    /// ]
    /// vocal_audio = audio.parametric_eq(bands=vocal_eq)
    ///
    /// # Simple 3-band EQ using tuples
    /// simple_eq = [
    ///     ('low_shelf', 200, -1, 0.7),   # Bass: -1dB
    ///     ('peak', 1000, 2, 1),          # Mids: +2dB  
    ///     ('high_shelf', 5000, 1, 0.7),  # Treble: +1dB
    /// ]
    /// simple_equalized = audio.parametric_eq(bands=simple_eq, gain=-0.5)
    /// ```
    pub(crate) fn parametric_eq_impl(
        &self,
        py: Python,
        bands: &Bound<PyAny>,
        gain: Option<f64>,
    ) -> PyResult<PyAudioSamples> {
        let eq_bands = self.parse_eq_bands(py, bands)?;
        
        let config = ParametricEqConfig {
            bands: eq_bands,
            global_gain: gain.unwrap_or(0.0),
        };

        let equalized = self
            .with_inner(|inner| inner.parametric_eq(&config))
            .map_err(map_error)?;
        Ok(PyAudioSamples::from_inner(equalized))
    }

    /// Apply parametric equalization in-place.
    ///
    /// Same as parametric_eq() but modifies the current AudioSamples object.
    ///
    /// # Arguments
    /// Same as parametric_eq()
    ///
    /// # Examples
    /// ```python
    /// # Apply EQ directly to audio
    /// bands = [
    ///     {'type': 'peak', 'freq': 440, 'gain': -6, 'q': 4},  # Notch out 440Hz
    ///     {'type': 'high_shelf', 'freq': 8000, 'gain': 2, 'q': 0.7},
    /// ]
    /// audio.parametric_eq_(bands=bands)
    /// ```
    pub(crate) fn parametric_eq_in_place_impl(
        &mut self,
        py: Python,
        bands: &Bound<PyAny>,
        gain: Option<f64>,
    ) -> PyResult<()> {
        let eq_bands = self.parse_eq_bands(py, bands)?;
        
        let config = ParametricEqConfig {
            bands: eq_bands,
            global_gain: gain.unwrap_or(0.0),
        };

        self.with_inner_mut(|inner| {
            inner.parametric_eq_in_place(&config)?;
            Ok(())
        })
        .map_err(map_error)
    }

    /// Add a single EQ band to existing parametric EQ.
    ///
    /// Convenience method for adding one band at a time, useful for
    /// interactive EQ building or real-time parameter adjustment.
    ///
    /// # Arguments
    /// * `band_type` - Type of EQ band ('peak', 'low_shelf', 'high_shelf', etc.)
    /// * `frequency` - Center frequency in Hz
    /// * `gain` - Gain in dB (ignored for filters without gain)
    /// * `q` - Q factor (bandwidth control, default: 1.0)
    ///
    /// # Returns
    /// New AudioSamples object with single EQ band applied
    ///
    /// # Examples
    /// ```python
    /// # Build EQ step by step
    /// eq_audio = audio.copy()
    /// eq_audio = eq_audio.add_eq_band('high_pass', frequency=80, q=0.7)
    /// eq_audio = eq_audio.add_eq_band('peak', frequency=1000, gain=3, q=2)
    /// eq_audio = eq_audio.add_eq_band('high_shelf', frequency=8000, gain=2, q=0.7)
    ///
    /// # Quick notch filter
    /// notched = audio.add_eq_band('notch', frequency=60, q=30)  # Remove 60Hz hum
    ///
    /// # Gentle high-pass
    /// high_passed = audio.add_eq_band('high_pass', frequency=120, q=0.5)
    /// ```
    pub(crate) fn add_eq_band_impl(
        &self,
        band_type: &str,
        frequency: f64,
        gain: Option<f64>,
        q: Option<f64>,
    ) -> PyResult<PyAudioSamples> {
        let eq_band = self.create_eq_band(band_type, frequency, gain, q)?;
        
        let config = ParametricEqConfig {
            bands: vec![eq_band],
            global_gain: 0.0,
        };

        let equalized = self
            .with_inner(|inner| inner.parametric_eq(&config))
            .map_err(map_error)?;
        Ok(PyAudioSamples::from_inner(equalized))
    }

    /// Compute the frequency response of the parametric EQ.
    ///
    /// Returns the magnitude and phase response that would be applied
    /// by the specified EQ bands, useful for visualization and analysis.
    ///
    /// # Arguments
    /// * `bands` - List of EQ band configurations
    /// * `frequencies` - NumPy array of frequencies to analyze (Hz)
    /// * `worN` - Number of frequency points if frequencies not provided (default: 512)
    ///
    /// # Returns
    /// Tuple of (frequencies, magnitude, phase) as NumPy arrays
    ///
    /// # Examples
    /// ```python
    /// # Analyze EQ response
    /// eq_bands = [
    ///     {'type': 'low_shelf', 'freq': 200, 'gain': -2, 'q': 0.7},
    ///     {'type': 'peak', 'freq': 1000, 'gain': 3, 'q': 2},
    ///     {'type': 'high_shelf', 'freq': 5000, 'gain': 1, 'q': 0.7},
    /// ]
    /// 
    /// # Get response for specific frequencies
    /// test_freqs = np.logspace(1, 4, 1000)  # 10 Hz to 10 kHz
    /// freqs, magnitude, phase = audio.eq_frequency_response(eq_bands, test_freqs)
    ///
    /// # Plot EQ curve
    /// import matplotlib.pyplot as plt
    /// plt.figure(figsize=(12, 6))
    /// 
    /// plt.subplot(2, 1, 1)
    /// plt.semilogx(freqs, 20 * np.log10(magnitude))
    /// plt.ylabel('Magnitude (dB)')
    /// plt.grid(True)
    /// plt.title('Parametric EQ Frequency Response')
    ///
    /// plt.subplot(2, 1, 2)
    /// plt.semilogx(freqs, np.degrees(phase))
    /// plt.ylabel('Phase (degrees)')
    /// plt.xlabel('Frequency (Hz)')
    /// plt.grid(True)
    ///
    /// # Auto-generate frequency points
    /// freqs_auto, mag_auto, phase_auto = audio.eq_frequency_response(eq_bands, worN=1024)
    /// ```
    pub(crate) fn eq_frequency_response_impl(
        &self,
        py: Python,
        bands: &Bound<PyAny>,
        frequencies: Option<&Bound<PyAny>>,
        worN: Option<usize>,
    ) -> PyResult<PyObject> {
        let eq_bands = self.parse_eq_bands(py, bands)?;
        
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
            // Generate logarithmic frequency grid
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

        let config = ParametricEqConfig {
            bands: eq_bands,
            global_gain: 0.0,
        };

        let response = self
            .with_inner(|inner| inner.eq_frequency_response(&config, &freq_vec))
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

    /// Create a complementary EQ that inverts the frequency response.
    ///
    /// Generates an EQ configuration that applies the opposite effect,
    /// useful for creating matched EQ pairs or undoing previous equalization.
    ///
    /// # Arguments
    /// * `bands` - Original EQ band configurations to invert
    ///
    /// # Returns
    /// New AudioSamples object with inverted EQ applied
    ///
    /// # Examples
    /// ```python
    /// # Original EQ
    /// original_eq = [
    ///     {'type': 'low_shelf', 'freq': 200, 'gain': -2, 'q': 0.7},
    ///     {'type': 'peak', 'freq': 1000, 'gain': 3, 'q': 2},
    ///     {'type': 'high_shelf', 'freq': 5000, 'gain': 1, 'q': 0.7},
    /// ]
    /// 
    /// # Apply original EQ
    /// eq_audio = audio.parametric_eq(bands=original_eq)
    /// 
    /// # Create and apply complementary EQ (should restore original)
    /// restored = eq_audio.complementary_eq(bands=original_eq)
    /// 
    /// # Verify restoration (should be very close to original)
    /// difference = audio.to_numpy() - restored.to_numpy()
    /// print(f"Max difference: {np.max(np.abs(difference))}")
    /// ```
    pub(crate) fn complementary_eq_impl(
        &self,
        py: Python,
        bands: &Bound<PyAny>,
    ) -> PyResult<PyAudioSamples> {
        let original_bands = self.parse_eq_bands(py, bands)?;
        
        // Invert the EQ bands
        let complementary_bands: Vec<EqBand> = original_bands
            .into_iter()
            .map(|band| {
                let mut inverted = band;
                // Invert gain for bands that have gain
                match &mut inverted.band_type {
                    EqBandType::Peak { gain, .. } => *gain = -*gain,
                    EqBandType::LowShelf { gain, .. } => *gain = -*gain,
                    EqBandType::HighShelf { gain, .. } => *gain = -*gain,
                    // Filters without gain remain unchanged
                    _ => {}
                }
                inverted
            })
            .collect();

        let config = ParametricEqConfig {
            bands: complementary_bands,
            global_gain: 0.0,
        };

        let equalized = self
            .with_inner(|inner| inner.parametric_eq(&config))
            .map_err(map_error)?;
        Ok(PyAudioSamples::from_inner(equalized))
    }

    /// Match the EQ of one audio signal to another using spectral analysis.
    ///
    /// Automatically generates an EQ curve that makes the current audio
    /// match the spectral characteristics of the target audio.
    ///
    /// # Arguments
    /// * `target_audio` - AudioSamples object to match
    /// * `num_bands` - Number of EQ bands to use (default: 10)
    /// * `smoothing` - Smoothing factor for spectral analysis (default: 0.3)
    ///
    /// # Returns
    /// New AudioSamples object with matching EQ applied
    ///
    /// # Examples
    /// ```python
    /// # Load reference audio with desired sound
    /// reference = aus.from_file("reference_track.wav")
    /// 
    /// # Match our audio to the reference
    /// matched = audio.match_eq(target_audio=reference, num_bands=12)
    /// 
    /// # Compare spectral content
    /// orig_spectrum = audio.fft()
    /// matched_spectrum = matched.fft()
    /// ref_spectrum = reference.fft()
    /// 
    /// # matched_spectrum should be closer to ref_spectrum than orig_spectrum
    /// ```
    pub(crate) fn match_eq_impl(
        &self,
        target_audio: &PyAudioSamples,
        num_bands: Option<usize>,
        smoothing: Option<f64>,
    ) -> PyResult<PyAudioSamples> {
        // Validate compatibility
        if self.sample_rate() != target_audio.sample_rate() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Sample rates must match for EQ matching"
            ));
        }

        let actual_num_bands = num_bands.unwrap_or(10);
        let actual_smoothing = smoothing.unwrap_or(0.3);

        if actual_smoothing < 0.0 || actual_smoothing > 1.0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Smoothing must be between 0.0 and 1.0"
            ));
        }

        // Convert target to compatible format
        let target_f64 = target_audio.as_f64().map_err(map_error)?;

        let matched = self
            .with_inner(|inner| inner.match_eq(&target_f64, actual_num_bands, actual_smoothing))
            .map_err(map_error)?;
        Ok(PyAudioSamples::from_inner(matched))
    }

    // Helper methods for parsing EQ bands
    fn parse_eq_bands(&self, py: Python, bands: &Bound<PyAny>) -> PyResult<Vec<EqBand>> {
        if let Ok(band_list) = bands.downcast::<PyList>() {
            let mut eq_bands = Vec::new();
            
            for item in band_list.iter() {
                let band = if let Ok(dict) = item.downcast::<PyDict>() {
                    self.parse_eq_band_from_dict(dict)?
                } else if let Ok(tuple) = item.extract::<(String, f64, Option<f64>, Option<f64>)>() {
                    self.create_eq_band(&tuple.0, tuple.1, tuple.2, tuple.3)?
                } else {
                    return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                        "EQ bands must be dictionaries or tuples (type, freq, gain, q)"
                    ));
                };
                eq_bands.push(band);
            }
            
            Ok(eq_bands)
        } else {
            Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                "Bands must be a list of dictionaries or tuples"
            ))
        }
    }

    fn parse_eq_band_from_dict(&self, dict: &Bound<PyDict>) -> PyResult<EqBand> {
        let band_type = dict.get_item("type")?
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>("Band must have 'type' field"))?
            .extract::<String>()?;

        let frequency = dict.get_item("freq")?
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>("Band must have 'freq' field"))?
            .extract::<f64>()?;

        let gain = dict.get_item("gain").and_then(|g| g.ok()).map(|g| g.extract::<f64>()).transpose()?;
        let q = dict.get_item("q").and_then(|q| q.ok()).map(|q| q.extract::<f64>()).transpose()?;

        self.create_eq_band(&band_type, frequency, gain, q)
    }

    fn create_eq_band(&self, band_type: &str, frequency: f64, gain: Option<f64>, q: Option<f64>) -> PyResult<EqBand> {
        validate_string_param(
            "band_type", 
            band_type, 
            &["peak", "low_shelf", "high_shelf", "low_pass", "high_pass", "notch", "all_pass"]
        )?;

        let nyquist = self.sample_rate() as f64 / 2.0;
        if frequency <= 0.0 || frequency >= nyquist {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                format!("Frequency must be between 0 and {} Hz (Nyquist)", nyquist)
            ));
        }

        let band_type_enum = match band_type {
            "peak" => EqBandType::Peak { 
                gain: gain.unwrap_or(0.0),
                q: q.unwrap_or(1.0),
            },
            "low_shelf" => EqBandType::LowShelf { 
                gain: gain.unwrap_or(0.0),
                q: q.unwrap_or(0.7),
            },
            "high_shelf" => EqBandType::HighShelf { 
                gain: gain.unwrap_or(0.0),
                q: q.unwrap_or(0.7),
            },
            "low_pass" => EqBandType::LowPass { 
                q: q.unwrap_or(0.7),
            },
            "high_pass" => EqBandType::HighPass { 
                q: q.unwrap_or(0.7),
            },
            "notch" => EqBandType::Notch { 
                q: q.unwrap_or(10.0),
            },
            "all_pass" => EqBandType::AllPass { 
                q: q.unwrap_or(1.0),
            },
            _ => unreachable!(),
        };

        Ok(EqBand {
            frequency,
            band_type: band_type_enum,
        })
    }
}

// Helper trait for multi-unzip functionality (same as in iir_filtering.rs)
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