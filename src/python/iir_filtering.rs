//! AudioIirFiltering trait implementation for Python bindings.
//!
//! This module provides Python access to Infinite Impulse Response (IIR) digital
//! filter design and application, including Butterworth and Chebyshev filter types
//! with comprehensive frequency response analysis capabilities.

use super::{PyAudioSamples, utils::*};
use crate::operations::{
    AudioIirFiltering,
    types::{FilterResponse, IirFilterDesign, IirFilterType},
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
    pub(crate) fn butterworth_filter_impl(
        &self,
        cutoff: f64,
        order: Option<usize>,
        filter_type: Option<&str>,
        high_cutoff: Option<f64>,
    ) -> PyResult<PyAudioSamples> {
        // Validate cutoff frequency
        let nyquist = self.sample_rate() as f64 / 2.0;
        if cutoff <= 0.0 || cutoff >= nyquist {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Cutoff frequency must be between 0 and {} Hz",
                nyquist
            )));
        }

        // Parse filter type
        let filter_type_str = filter_type.unwrap_or("lowpass");
        validate_string_param(
            "filter_type",
            filter_type_str,
            &["lowpass", "highpass", "bandpass", "bandstop"],
        )?;

        let response = match filter_type_str {
            "lowpass" => FilterResponse::LowPass,
            "highpass" => FilterResponse::HighPass,
            "bandpass" => FilterResponse::BandPass,
            "bandstop" => FilterResponse::BandStop,
            _ => unreachable!(),
        };

        // Validate high_cutoff for band filters
        if matches!(
            response,
            FilterResponse::BandPass | FilterResponse::BandStop
        ) {
            let high_freq = high_cutoff.ok_or_else(|| {
                PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "high_cutoff is required for bandpass and bandstop filters",
                )
            })?;
            if high_freq <= cutoff || high_freq >= nyquist {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "High cutoff must be between {} and {} Hz",
                    cutoff, nyquist
                )));
            }
        }

        // Create filter design
        let design = match response {
            FilterResponse::LowPass => {
                IirFilterDesign::butterworth_lowpass(order.unwrap_or(4), cutoff)
            }
            FilterResponse::HighPass => {
                IirFilterDesign::butterworth_highpass(order.unwrap_or(4), cutoff)
            }
            FilterResponse::BandPass => IirFilterDesign::butterworth_bandpass(
                order.unwrap_or(4),
                cutoff,
                high_cutoff.unwrap(),
            ),
            FilterResponse::BandStop => IirFilterDesign {
                filter_type: IirFilterType::Butterworth,
                response: FilterResponse::BandStop,
                order: order.unwrap_or(4),
                cutoff_frequency: None,
                low_frequency: Some(cutoff),
                high_frequency: high_cutoff,
                passband_ripple: None,
                stopband_attenuation: None,
            },
        };

        // Apply filter
        let mut result = self.copy();
        let sample_rate = result.sample_rate() as f64;
        result
            .mutate_inner(|inner| inner.apply_iir_filter(&design, sample_rate))
            .map_err(map_error)?;
        Ok(result)
    }

    /// Apply a Butterworth IIR filter in-place.
    ///
    /// Same as butterworth_filter() but modifies the current AudioSamples object.
    ///
    /// # Arguments
    /// Same as butterworth_filter()
    pub(crate) fn butterworth_filter_in_place_impl(
        &mut self,
        cutoff: f64,
        order: Option<usize>,
        filter_type: Option<&str>,
        high_cutoff: Option<f64>,
    ) -> PyResult<()> {
        // Use the same logic as the functional version
        let nyquist = self.sample_rate() as f64 / 2.0;
        if cutoff <= 0.0 || cutoff >= nyquist {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Cutoff frequency must be between 0 and {} Hz",
                nyquist
            )));
        }

        let filter_type_str = filter_type.unwrap_or("lowpass");
        validate_string_param(
            "filter_type",
            filter_type_str,
            &["lowpass", "highpass", "bandpass", "bandstop"],
        )?;

        let response = match filter_type_str {
            "lowpass" => FilterResponse::LowPass,
            "highpass" => FilterResponse::HighPass,
            "bandpass" => FilterResponse::BandPass,
            "bandstop" => FilterResponse::BandStop,
            _ => unreachable!(),
        };

        if matches!(
            response,
            FilterResponse::BandPass | FilterResponse::BandStop
        ) {
            let high_freq = high_cutoff.ok_or_else(|| {
                PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "high_cutoff is required for bandpass and bandstop filters",
                )
            })?;
            if high_freq <= cutoff || high_freq >= nyquist {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "High cutoff must be between {} and {} Hz",
                    cutoff, nyquist
                )));
            }
        }

        let design = match response {
            FilterResponse::LowPass => {
                IirFilterDesign::butterworth_lowpass(order.unwrap_or(4), cutoff)
            }
            FilterResponse::HighPass => {
                IirFilterDesign::butterworth_highpass(order.unwrap_or(4), cutoff)
            }
            FilterResponse::BandPass => IirFilterDesign::butterworth_bandpass(
                order.unwrap_or(4),
                cutoff,
                high_cutoff.unwrap(),
            ),
            FilterResponse::BandStop => IirFilterDesign {
                filter_type: IirFilterType::Butterworth,
                response: FilterResponse::BandStop,
                order: order.unwrap_or(4),
                cutoff_frequency: None,
                low_frequency: Some(cutoff),
                high_frequency: high_cutoff,
                passband_ripple: None,
                stopband_attenuation: None,
            },
        };

        // Apply filter in-place
        let sample_rate = self.sample_rate() as f64;
        self.mutate_inner(|inner| inner.apply_iir_filter(&design, sample_rate))
            .map_err(map_error)
    }

    /// Apply a Chebyshev IIR filter to the audio signal.
    ///
    /// Chebyshev filters provide sharper roll-off than Butterworth filters
    /// but introduce ripple in either the passband (Type I) or stopband (Type II).
    ///
    /// # Arguments
    /// * `cutoff` - Cutoff frequency in Hz
    /// * `order` - Filter order (higher = steeper rolloff, default: 4)
    /// * `filter_type` - Filter type ('lowpass', 'highpass', 'bandpass', 'bandstop', default: 'lowpass')
    /// * `ripple` - Passband ripple in dB (default: 1.0)
    /// * `high_cutoff` - High cutoff frequency for bandpass/bandstop filters (Hz)
    ///
    /// # Returns
    /// New AudioSamples object with filtering applied
    pub(crate) fn chebyshev_filter_impl(
        &self,
        cutoff: f64,
        order: Option<usize>,
        filter_type: Option<&str>,
        ripple: Option<f64>,
        high_cutoff: Option<f64>,
    ) -> PyResult<PyAudioSamples> {
        // Validate cutoff frequency
        let nyquist = self.sample_rate() as f64 / 2.0;
        if cutoff <= 0.0 || cutoff >= nyquist {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Cutoff frequency must be between 0 and {} Hz",
                nyquist
            )));
        }

        // Parse filter type
        let filter_type_str = filter_type.unwrap_or("lowpass");
        validate_string_param(
            "filter_type",
            filter_type_str,
            &["lowpass", "highpass", "bandpass", "bandstop"],
        )?;

        let response = match filter_type_str {
            "lowpass" => FilterResponse::LowPass,
            "highpass" => FilterResponse::HighPass,
            "bandpass" => FilterResponse::BandPass,
            "bandstop" => FilterResponse::BandStop,
            _ => unreachable!(),
        };

        // Validate high_cutoff for band filters
        if matches!(
            response,
            FilterResponse::BandPass | FilterResponse::BandStop
        ) {
            let high_freq = high_cutoff.ok_or_else(|| {
                PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "high_cutoff is required for bandpass and bandstop filters",
                )
            })?;
            if high_freq <= cutoff || high_freq >= nyquist {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "High cutoff must be between {} and {} Hz",
                    cutoff, nyquist
                )));
            }
        }

        // Create Chebyshev Type I filter design
        let design = IirFilterDesign::chebyshev_i(
            response,
            order.unwrap_or(4),
            cutoff,
            ripple.unwrap_or(1.0),
        );

        // Apply filter
        let mut result = self.copy();
        let sample_rate = result.sample_rate() as f64;
        result
            .mutate_inner(|inner| inner.apply_iir_filter(&design, sample_rate))
            .map_err(map_error)?;
        Ok(result)
    }

    /// Apply a Chebyshev IIR filter in-place.
    ///
    /// Same as chebyshev_filter() but modifies the current AudioSamples object.
    ///
    /// # Arguments
    /// Same as chebyshev_filter()
    pub(crate) fn chebyshev_filter_in_place_impl(
        &mut self,
        cutoff: f64,
        order: Option<usize>,
        filter_type: Option<&str>,
        ripple: Option<f64>,
        high_cutoff: Option<f64>,
    ) -> PyResult<()> {
        // Use the same logic as the functional version
        let nyquist = self.sample_rate() as f64 / 2.0;
        if cutoff <= 0.0 || cutoff >= nyquist {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Cutoff frequency must be between 0 and {} Hz",
                nyquist
            )));
        }

        let filter_type_str = filter_type.unwrap_or("lowpass");
        validate_string_param(
            "filter_type",
            filter_type_str,
            &["lowpass", "highpass", "bandpass", "bandstop"],
        )?;

        let response = match filter_type_str {
            "lowpass" => FilterResponse::LowPass,
            "highpass" => FilterResponse::HighPass,
            "bandpass" => FilterResponse::BandPass,
            "bandstop" => FilterResponse::BandStop,
            _ => unreachable!(),
        };

        if matches!(
            response,
            FilterResponse::BandPass | FilterResponse::BandStop
        ) {
            let high_freq = high_cutoff.ok_or_else(|| {
                PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "high_cutoff is required for bandpass and bandstop filters",
                )
            })?;
            if high_freq <= cutoff || high_freq >= nyquist {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "High cutoff must be between {} and {} Hz",
                    cutoff, nyquist
                )));
            }
        }

        let design = IirFilterDesign::chebyshev_i(
            response,
            order.unwrap_or(4),
            cutoff,
            ripple.unwrap_or(1.0),
        );

        // Apply filter in-place
        let sample_rate = self.sample_rate() as f64;
        self.mutate_inner(|inner| inner.apply_iir_filter(&design, sample_rate))
            .map_err(map_error)
    }
}
