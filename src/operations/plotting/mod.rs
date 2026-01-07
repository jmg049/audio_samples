pub mod composite;
#[cfg(feature = "plotting")]
pub mod dsp_overlays;
pub mod spectrograms;
pub mod spectrum;
pub mod waveform;

use core::num::NonZeroUsize;
use std::path::Path;

use crate::{
    AudioSampleResult, AudioSamples, StandardSample,
    operations::{
        AudioPlotting, WaveformPlot, create_waveform_plot,
        plotting::{
            spectrograms::{SpectrogramPlot, SpectrogramPlotParams},
            waveform::WaveformPlotParams,
        },
    },
};
pub(crate) const DECIMATE_THRESHOLD: NonZeroUsize = crate::nzu!(25000); // If more than this many samples, apply decimation for plotting

pub trait PlotUtils {
    #[cfg(feature = "html_view")]
    fn show(&self) -> AudioSampleResult<()>;
    fn save<P: AsRef<Path>>(&self, path: P) -> AudioSampleResult<()>;
    fn html(&self) -> AudioSampleResult<String>;
}

/// Applies min-max decimation to waveform data for visual clarity.
///
/// When plotting dense waveforms, displaying every sample can result in visual
/// artifacts where high-frequency oscillations appear as solid blocks. This function
/// downsamples the data while preserving the visual envelope by keeping the minimum
/// and maximum amplitude values within each time bin.
///
/// This is the standard technique used by professional audio software (Audacity,
/// Pro Tools, etc.) for waveform visualization.
///
/// # Arguments
/// * `time_data` - Vector of time values in seconds
/// * `amplitude_data` - Vector of amplitude values
/// * `target_points` - Target number of output points (actual output may be up to 2x this)
///
/// # Returns
/// A tuple of (decimated_time, decimated_amplitude) vectors
///
/// # Algorithm
/// 1. Divide the input data into `target_points/2` bins
/// 2. For each bin, find the sample with minimum and maximum amplitude
/// 3. Output both min and max samples in time order, preserving the envelope
/// 4. This results in approximately `target_points` output samples
fn decimate_waveform(
    time_data: &[f64],
    amplitude_data: &[f64],
    target_points: usize,
) -> (Vec<f64>, Vec<f64>) {
    let n_samples = time_data.len();

    // If we already have fewer samples than target, return as-is
    if n_samples <= target_points {
        return (time_data.to_vec(), amplitude_data.to_vec());
    }

    // Calculate bin size - we'll output 2 points per bin (min and max)
    // So we need target_points/2 bins to get approximately target_points output
    let n_bins = (target_points / 2).max(1);
    let bin_size = n_samples / n_bins;

    let mut decimated_time = Vec::with_capacity(n_bins * 2);
    let mut decimated_amplitude = Vec::with_capacity(n_bins * 2);

    for bin_idx in 0..n_bins {
        let start_idx = bin_idx * bin_size;
        let end_idx = if bin_idx == n_bins - 1 {
            n_samples // Last bin gets all remaining samples
        } else {
            ((bin_idx + 1) * bin_size).min(n_samples)
        };

        if start_idx >= end_idx {
            break;
        }

        // Find min and max in this bin
        let mut min_idx = start_idx;
        let mut max_idx = start_idx;
        let mut min_val = amplitude_data[start_idx];
        let mut max_val = amplitude_data[start_idx];

        for i in start_idx..end_idx {
            let val = amplitude_data[i];
            if val < min_val {
                min_val = val;
                min_idx = i;
            }
            if val > max_val {
                max_val = val;
                max_idx = i;
            }
        }

        // Add min and max in time order (important for proper line rendering)
        if min_idx < max_idx {
            decimated_time.push(time_data[min_idx]);
            decimated_amplitude.push(min_val);
            decimated_time.push(time_data[max_idx]);
            decimated_amplitude.push(max_val);
        } else {
            decimated_time.push(time_data[max_idx]);
            decimated_amplitude.push(max_val);
            decimated_time.push(time_data[min_idx]);
            decimated_amplitude.push(min_val);
        }
    }

    (decimated_time, decimated_amplitude)
}

/// Configure axis for time display with appropriate formatting
///
/// # Arguments
/// * `axis` - The axis to configure
/// * `title` - Optional axis title (will auto-append " (s)")
///
/// # Returns
/// Configured axis with time formatting
pub(crate) fn configure_time_axis(
    mut axis: plotly::layout::Axis,
    title: Option<String>,
) -> plotly::layout::Axis {
    let title_text = if let Some(t) = title {
        if !t.contains('(') {
            format!("{} (s)", t)
        } else {
            t
        }
    } else {
        "Time (s)".to_string()
    };

    axis = axis.title(title_text);

    // Use 2 decimal places for seconds
    axis = axis.tick_format(".2f");

    axis
}

/// Configure axis for frequency display with automatic Hz/kHz switching
///
/// # Arguments
/// * `axis` - The axis to configure
/// * `title` - Optional axis title (will auto-append " (Hz)" or " (kHz)")
/// * `max_freq` - Maximum frequency value to determine if kHz should be used
///
/// # Returns
/// Configured axis with appropriate frequency formatting
pub(crate) fn configure_frequency_axis(
    mut axis: plotly::layout::Axis,
    title: Option<String>,
    max_freq: f64,
) -> plotly::layout::Axis {
    // Use kHz if max frequency is above 1000 Hz
    let use_khz = max_freq > 1000.0;

    let title_text = if let Some(t) = title {
        if !t.contains('(') {
            if use_khz {
                format!("{} (kHz)", t)
            } else {
                format!("{} (Hz)", t)
            }
        } else {
            t
        }
    } else if use_khz {
        "Frequency (kHz)".to_string()
    } else {
        "Frequency (Hz)".to_string()
    };

    axis = axis.title(title_text);

    // Format ticks appropriately
    if use_khz {
        // One decimal place for kHz
        axis = axis.tick_format(".1f");
        // Note: Tick values will need to be divided by 1000 when setting data
    } else {
        // Integer formatting for Hz
        axis = axis.tick_format(".0f");
    }

    axis
}

#[derive(Debug, Clone, Copy)]
pub struct FontSizes {
    pub title: NonZeroUsize,
    pub axis_labels: NonZeroUsize,
    pub ticks: NonZeroUsize,
    pub legend: NonZeroUsize,
    pub super_title: NonZeroUsize,
    pub misc: NonZeroUsize,
}

impl FontSizes {
    #[inline]
    pub fn new(
        title: NonZeroUsize,
        axis_labels: NonZeroUsize,
        ticks: NonZeroUsize,
        legend: NonZeroUsize,
        super_title: NonZeroUsize,
        misc: NonZeroUsize,
    ) -> Self {
        Self {
            title,
            axis_labels,
            ticks,
            legend,
            super_title,
            misc,
        }
    }
}

impl Default for FontSizes {
    fn default() -> Self {
        Self {
            title: crate::nzu!(16),
            axis_labels: crate::nzu!(12),
            ticks: crate::nzu!(10),
            legend: crate::nzu!(12),
            super_title: crate::nzu!(18),
            misc: crate::nzu!(10),
        }
    }
}

/// Common parameters for plotting operations
#[derive(Debug, Clone)]
pub struct PlotParams {
    pub title: Option<String>,
    pub x_label: Option<String>,
    pub y_label: Option<String>,
    pub font_sizes: Option<FontSizes>,
    pub show_legend: bool,
    pub legend_title: Option<String>,
    pub super_title: Option<String>,
    pub grid: bool,
}

impl PlotParams {
    #[inline]
    pub const fn new(
        title: Option<String>,
        x_label: Option<String>,
        y_label: Option<String>,
        font_sizes: Option<FontSizes>,
        show_legend: bool,
        legend_title: Option<String>,
        super_title: Option<String>,
        grid: bool,
    ) -> Self {
        Self {
            title,
            x_label,
            y_label,
            font_sizes,
            show_legend,
            legend_title,
            super_title,
            grid,
        }
    }
}

impl Default for PlotParams {
    fn default() -> Self {
        Self {
            title: None,
            x_label: None,
            y_label: None,
            font_sizes: None,
            show_legend: false,
            legend_title: None,
            super_title: None,
            grid: false,
        }
    }
}

#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum PlotType {
    Waveform(WaveformPlotParams),
    Spectrogram,
}

/// Enum to specify how to manage multiple channels when plotting
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChannelManagementStrategy {
    Average,
    Separate(Layout),
    First,
    Last,
    Overlap,
}

impl Default for ChannelManagementStrategy {
    fn default() -> Self {
        Self::Separate(Layout::default())
    }
}

/// When plotting multiple elements separately, specify the layout
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Layout {
    #[default]
    Vertical,
    Horizontal,
}

impl<T> AudioPlotting for AudioSamples<'_, T>
where
    T: StandardSample,
{
    fn plot_waveform(&self, params: &WaveformPlotParams) -> AudioSampleResult<WaveformPlot> {
        create_waveform_plot(self, params)
    }

    #[cfg(feature = "transforms")]
    fn plot_spectrogram(
        &self,
        params: &SpectrogramPlotParams,
    ) -> AudioSampleResult<SpectrogramPlot> {
        spectrograms::create_spectrogram_plot(self, params)
    }

    #[cfg(not(feature = "transforms"))]
    fn plot_spectrogram(
        &self,
        _params: &SpectrogramPlotParams,
    ) -> AudioSampleResult<SpectrogramPlot> {
        Err(crate::AudioSampleError::Feature(
            crate::FeatureError::NotEnabled {
                feature: "transforms".to_string(),
                operation: "plot spectrograms".to_string(),
            },
        ))
    }

    #[cfg(feature = "transforms")]
    fn plot_magnitude_spectrum(
        &self,
        params: &spectrum::MagnitudeSpectrumParams,
    ) -> AudioSampleResult<spectrum::MagnitudeSpectrumPlot> {
        spectrum::create_magnitude_spectrum_plot(self, params)
    }

    #[cfg(not(feature = "transforms"))]
    fn plot_magnitude_spectrum(
        &self,
        _params: &spectrum::MagnitudeSpectrumParams,
    ) -> AudioSampleResult<spectrum::MagnitudeSpectrumPlot> {
        Err(crate::AudioSampleError::Feature(
            crate::FeatureError::NotEnabled {
                feature: "transforms".to_string(),
                operation: "plot magnitude spectrum".to_string(),
            },
        ))
    }
}
