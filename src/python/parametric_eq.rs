//! AudioParametricEq trait implementation for Python bindings.
//!
//! This module provides Python access to professional-grade parametric equalization
//! using the RBJ (Robert Bristow-Johnson) cookbook formulas. Supports multi-band
//! EQ with peak, shelf, and filter band types for precise frequency shaping.

use super::{PyAudioSamples, utils::*};
use crate::operations::{
    AudioParametricEq,
    types::{EqBand, EqBandType, ParametricEq},
};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

impl PyAudioSamples {
    /// Apply parametric equalization with multiple bands.
    ///
    /// Parametric EQ allows precise frequency shaping using multiple bands
    /// with various filter types including peaks, shelves, and filters.
    ///
    /// # Arguments
    /// * `bands` - List of band dictionaries with keys:
    ///   - 'type': Band type ('peak', 'low_shelf', 'high_shelf', 'low_pass', 'high_pass')
    ///   - 'frequency': Center/corner frequency in Hz
    ///   - 'gain_db': Gain in dB (ignored for pass filters, default: 0.0)
    ///   - 'q_factor': Quality factor for bandwidth control (default: 1.0)
    ///   - 'enabled': Whether band is active (default: True)
    /// * `output_gain` - Overall output gain in dB (default: 0.0)
    ///
    /// # Returns
    /// New AudioSamples object with parametric EQ applied
    ///
    /// # Examples
    /// ```python
    /// # Apply 3-band EQ
    /// bands = [
    ///     {'type': 'low_shelf', 'frequency': 100, 'gain_db': -2, 'q_factor': 0.707},
    ///     {'type': 'peak', 'frequency': 1000, 'gain_db': 3, 'q_factor': 2.0},
    ///     {'type': 'high_shelf', 'frequency': 8000, 'gain_db': 1, 'q_factor': 0.707}
    /// ]
    /// eq_audio = audio.parametric_eq(bands, output_gain=0.5)
    /// ```
    pub(crate) fn parametric_eq_impl(
        &self,
        bands: &Bound<PyAny>,
        output_gain: Option<f64>,
    ) -> PyResult<PyAudioSamples> {
        // Parse bands from Python input
        let eq = self.parse_parametric_eq(bands, output_gain)?;

        // Apply parametric EQ
        let mut result = self.copy();
        let sample_rate = result.sample_rate() as f64;
        result
            .mutate_inner(|inner| inner.apply_parametric_eq(&eq, sample_rate))
            .map_err(map_error)?;
        Ok(result)
    }

    /// Apply parametric equalization in-place.
    ///
    /// Same as parametric_eq() but modifies the current AudioSamples object.
    ///
    /// # Arguments
    /// Same as parametric_eq()
    pub(crate) fn parametric_eq_in_place_impl(
        &mut self,
        bands: &Bound<PyAny>,
        output_gain: Option<f64>,
    ) -> PyResult<()> {
        // Parse bands from Python input
        let eq = self.parse_parametric_eq(bands, output_gain)?;

        // Apply parametric EQ in-place
        let sample_rate = self.sample_rate() as f64;
        self.mutate_inner(|inner| inner.apply_parametric_eq(&eq, sample_rate))
            .map_err(map_error)
    }

    /// Parse parametric EQ configuration from Python input.
    fn parse_parametric_eq(
        &self,
        bands: &Bound<PyAny>,
        output_gain: Option<f64>,
    ) -> PyResult<ParametricEq> {
        let mut eq = ParametricEq::new();
        eq.set_output_gain(output_gain.unwrap_or(0.0));

        // Handle different input types for bands
        if let Ok(bands_list) = bands.downcast::<PyList>() {
            // List of band dictionaries
            for band_obj in bands_list.iter() {
                let band = self.parse_eq_band(&band_obj)?;
                eq.add_band(band);
            }
        } else if let Ok(band_dict) = bands.downcast::<PyDict>() {
            // Single band dictionary
            let band = self.parse_eq_band_from_dict(band_dict)?;
            eq.add_band(band);
        } else {
            return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                "bands must be a list of dictionaries or a single dictionary",
            ));
        }

        Ok(eq)
    }

    /// Parse a single EQ band from Python object.
    fn parse_eq_band(&self, band_obj: &Bound<PyAny>) -> PyResult<EqBand> {
        if let Ok(band_dict) = band_obj.downcast::<PyDict>() {
            self.parse_eq_band_from_dict(band_dict)
        } else {
            Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                "Each band must be a dictionary",
            ))
        }
    }

    /// Parse EQ band from dictionary.
    fn parse_eq_band_from_dict(&self, band_dict: &Bound<PyDict>) -> PyResult<EqBand> {
        // Extract band type
        let band_type_str = band_dict
            .get_item("type")?
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyKeyError, _>("Missing 'type' key"))?
            .extract::<String>()?;

        let band_type = match band_type_str.as_str() {
            "peak" => EqBandType::Peak,
            "low_shelf" => EqBandType::LowShelf,
            "high_shelf" => EqBandType::HighShelf,
            "low_pass" => EqBandType::LowPass,
            "high_pass" => EqBandType::HighPass,
            "band_pass" => EqBandType::BandPass,
            "band_stop" => EqBandType::BandStop,
            _ => {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "Invalid band type '{}'. Valid types: 'peak', 'low_shelf', 'high_shelf', 'low_pass', 'high_pass', 'band_pass', 'band_stop'",
                    band_type_str
                )));
            }
        };

        // Extract frequency
        let frequency = band_dict
            .get_item("frequency")?
            .ok_or_else(|| {
                PyErr::new::<pyo3::exceptions::PyKeyError, _>("Missing 'frequency' key")
            })?
            .extract::<f64>()?;

        // Extract gain (default 0.0, ignored for pass filters)
        let gain_db = band_dict
            .get_item("gain_db")
            .unwrap_or(None)
            .map(|v| v.extract::<f64>())
            .transpose()?
            .unwrap_or(0.0);

        // Extract Q factor (default 1.0)
        let q_factor = band_dict
            .get_item("q_factor")
            .unwrap_or(None)
            .map(|v| v.extract::<f64>())
            .transpose()?
            .unwrap_or(1.0);

        // Extract enabled flag (default true)
        let enabled = band_dict
            .get_item("enabled")
            .unwrap_or(None)
            .map(|v| v.extract::<bool>())
            .transpose()?
            .unwrap_or(true);

        // Validate parameters
        if frequency <= 0.0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Frequency must be greater than 0",
            ));
        }

        if q_factor <= 0.0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Q factor must be greater than 0",
            ));
        }

        // Create EQ band
        let mut band = EqBand {
            band_type,
            frequency,
            gain_db,
            q_factor,
            enabled,
        };

        band.set_enabled(enabled);
        Ok(band)
    }
}
