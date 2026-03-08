//! Audio signal visualization using interactive plots.
//! This module provides interactive HTML-based visualizations of audio signals via the
//! [`plotly`] crate. Three core plot types are supported:
//!
//! - **Waveform plots** — time-domain amplitude traces with optional envelope overlays,
//!   onset markers, beat markers, and DSP feature annotations.
//! - **Spectrogram plots** — time–frequency heatmaps from STFT, mel-scale, or CQT analysis.
//! - **Magnitude spectrum plots** — single-frame frequency-domain views showing per-bin
//!   magnitudes.
//!
//! Visual inspection is essential during audio development: for debugging signal-processing
//! chains, validating onset or beat detection, comparing spectral representations, or
//! generating report figures. This module isolates all plotting concerns from the core
//! audio abstractions, keeping the main API surface clean while providing rich
//! visualisation tools when the `plotting` feature is enabled.
//!
//! Import the [`AudioPlotting`](crate::operations::AudioPlotting) trait and call its methods on any [`AudioSamples`] instance.
//! Each method returns an opaque plot object (`WaveformPlot`, `SpectrogramPlot`,
//! `MagnitudeSpectrumPlot`) that can be rendered to HTML or saved to a file:
//!
//! ```rust
//! use audio_samples::{AudioSamples, sample_rate, sine_wave};
//! use audio_samples::operations::traits::AudioPlotting;
//! use audio_samples::operations::plotting::waveform::WaveformPlotParams;
//! use std::time::Duration;
//!
//! let audio = sine_wave::<f64>(440.0, Duration::from_millis(100), sample_rate!(44100), 0.8);
//! let params = WaveformPlotParams::default();
//! let plot = audio.plot_waveform(&params).unwrap();
//!
//! // Save as interactive HTML
//! // plot.save("output.html").unwrap();
//! ```
//!
//! Plot objects support method chaining for overlays — add RMS envelopes, onset markers,
//! beat markers, or shaded regions after creating the base plot. For spectrogram plots,
//! choose the spectrogram type via [`SpectrogramPlotParams`]; for magnitude spectrum plots,
//! configure FFT size and frequency range via [`MagnitudeSpectrumParams`].
//!
//! [`AudioPlotting`](crate::operations::AudioPlotting): crate::operations::AudioPlotting
//! [`AudioSamples`]: crate::AudioSamples
//! [`SpectrogramPlotParams`]: spectrograms::SpectrogramPlotParams
//! [`MagnitudeSpectrumParams`]: spectrum::MagnitudeSpectrumParams
/// Composite plot layouts combining multiple visualizations.
pub mod composite;
#[cfg(feature = "plotting")]
pub mod dsp_overlays;
/// Spectrogram plotting with various frequency scales and amplitude encodings.
pub mod spectrograms;
/// Magnitude spectrum visualization.
pub mod spectrum;
/// Time-domain waveform plotting with overlay support.
pub mod waveform;

pub use composite::{CompositeLayout, CompositePlot};
pub use spectrograms::{SpectrogramPlot, SpectrogramPlotParams, create_spectrogram_plot};
pub use spectrum::{MagnitudeSpectrumParams, create_magnitude_spectrum_plot};
pub use waveform::{WaveformPlot, WaveformPlotParams};

use core::num::NonZeroUsize;
use std::path::Path;

use crate::{
    AudioSampleResult, AudioSamples, StandardSample,
    operations::{AudioPlotting, create_waveform_plot},
};
pub(crate) const DECIMATE_THRESHOLD: NonZeroUsize = crate::nzu!(25000); // If more than this many samples, apply decimation for plotting

/// Common operations for plot output and rendering.
///
/// # Purpose
///
/// Abstracts the output pipeline for all plot types: generating interactive HTML,
/// saving to disk, and (when `html_view` is enabled) opening in a browser window.
///
/// # Intended Usage
///
/// This trait is implemented by `WaveformPlot`, `SpectrogramPlot`, and
/// `MagnitudeSpectrumPlot`. Users call these methods after creating a plot to
/// render or persist it.
///
/// # Invariants
///
/// The `html()` method always succeeds for valid plots; file I/O failures in
/// `save()` are reported as [`AudioSampleError`].
///
/// [`AudioSampleError`]: crate::AudioSampleError
pub trait PlotUtils {
    /// Generates an interactive HTML string from the plot.
    ///
    /// # Returns
    ///
    /// A `String` containing a complete, self-contained HTML document with
    /// embedded Plotly.js and plot data.
    ///
    /// # Errors
    ///
    /// Returns [`AudioSampleError`] if the internal Plotly rendering fails
    /// (rare; indicates a malformed plot structure).
    ///
    /// [`AudioSampleError`]: crate::AudioSampleError
    fn html(&self) -> AudioSampleResult<String>;

    /// Opens the plot in a browser window using [`html_view`].
    ///
    /// Only available when `feature = "html_view"` is enabled.
    ///
    /// # Returns
    ///
    /// `Ok(())` if the browser was launched successfully.
    ///
    /// # Errors
    ///
    /// Returns [`AudioSampleError`] if the browser cannot be opened or if
    /// the HTML generation fails.
    ///
    /// [`html_view`]: https://docs.rs/html_view
    /// [`AudioSampleError`]: crate::AudioSampleError
    #[cfg(feature = "html_view")]
    fn show(&self) -> AudioSampleResult<()>;

    /// Saves the plot to a file.
    ///
    /// The file extension determines the output format:
    /// - `.html` — interactive HTML (always available).
    /// - `.png`, `.svg`, `.jpeg`, `.jpg`, `.webp` — static images
    ///   (requires `feature = "static-plots"`).
    ///
    /// # Arguments
    ///
    /// - `path` – Output file path. The extension is inspected to determine
    ///   the format.
    ///
    /// # Returns
    ///
    /// `Ok(())` if the file was written successfully.
    ///
    /// # Errors
    ///
    /// - [crate::AudioSampleError::Feature] if the extension requires `static-plots`
    ///   but the feature is not enabled.
    /// - [crate::AudioSampleError::Parameter] if the extension is unsupported.
    /// - [`AudioSampleError`] (unsupported variant) if file I/O fails.
    ///
    /// [crate::AudioSampleError::Feature]: crate::AudioSampleError::Feature
    /// [crate::AudioSampleError::Parameter]: crate::AudioSampleError::Parameter
    /// [`AudioSampleError`]: crate::AudioSampleError
    fn save<P: AsRef<Path>>(&self, path: P) -> AudioSampleResult<()>;
}

/// Applies min-max decimation to waveform data for visual clarity.
///
/// When plotting dense waveforms, displaying every sample can result in visual
/// artifacts where high-frequency oscillations appear as solid blocks. This function
/// downsamples by dividing the input into bins and emitting the minimum and maximum
/// sample from each bin in time order, preserving the visual envelope without
/// storing every point.
///
/// This is the standard technique used by professional audio software (Audacity,
/// Pro Tools, etc.) for waveform visualization.
///
/// # Arguments
///
/// - `time_data` – Time values in seconds.
/// - `amplitude_data` – Corresponding amplitude values.
/// - `target_points` – Target number of output points. The actual output may contain
///   up to `2 × target_points` values (two per bin: min + max).
///
/// # Returns
///
/// A `(decimated_time, decimated_amplitude)` tuple. If `time_data.len() ≤ target_points`,
/// the input is returned unchanged.
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

        for (i, &val) in amplitude_data
            .iter()
            .enumerate()
            .take(end_idx)
            .skip(start_idx)
        {
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

/// Configures an axis for time display with seconds formatting.
///
/// Applies a tick format of `.2f` (two decimal places) and appends `" (s)"` to
/// the title unless the title already contains parentheses.
///
/// # Arguments
///
/// - `axis` – The Plotly axis to configure.
/// - `title` – Optional title string. If `None`, defaults to `"Time (s)"`.
///   If provided without parentheses, `" (s)"` is auto-appended.
///
/// # Returns
///
/// The configured `Axis` with time formatting applied.
pub(crate) fn configure_time_axis(
    mut axis: plotly::layout::Axis,
    title: Option<String>,
) -> plotly::layout::Axis {
    let title_text = title.map_or_else(
        || "Time (s)".to_string(),
        |t| {
            if t.contains('(') {
                t
            } else {
                format!("{t} (s)")
            }
        },
    );

    axis = axis.title(title_text);

    // Use 2 decimal places for seconds
    axis = axis.tick_format(".2f");

    axis
}

/// Configures an axis for frequency display with automatic Hz/kHz switching.
///
/// If `max_freq > 1000.0`, tick values are formatted in kHz with one decimal place
/// and the title is suffixed with `" (kHz)"`. Otherwise, integer Hz formatting is used
/// and the title is suffixed with `" (Hz)"`.
///
/// # Arguments
///
/// - `axis` – The Plotly axis to configure.
/// - `title` – Optional title string. If `None`, defaults to `"Frequency (Hz)"` or
///   `"Frequency (kHz)"` based on `max_freq`. If provided without parentheses, the
///   appropriate unit suffix is auto-appended.
/// - `max_freq` – Maximum frequency value used to decide the unit.
///
/// # Returns
///
/// The configured `Axis` with frequency formatting applied.
///
/// # Note
///
/// When kHz formatting is selected, the **caller** must divide tick values by `1000.0`
/// when setting trace data; this function only adjusts axis display settings.
pub(crate) fn configure_frequency_axis(
    mut axis: plotly::layout::Axis,
    title: Option<String>,
    max_freq: f64,
) -> plotly::layout::Axis {
    // Use kHz if max frequency is above 1000 Hz
    let use_khz = max_freq > 1000.0;

    let title_text = title.map_or_else(
        || {
            if use_khz {
                "Frequency (kHz)".to_string()
            } else {
                "Frequency (Hz)".to_string()
            }
        },
        |t| {
            if t.contains('(') {
                t
            } else if use_khz {
                format!("{t} (kHz)")
            } else {
                format!("{t} (Hz)")
            }
        },
    );
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

/// Font size configuration for plot elements.
///
/// # Purpose
///
/// Encapsulates all font sizes used in a plot — title, axis labels, tick labels,
/// legend text, super-title (for multi-panel plots), and miscellaneous annotations.
/// Provides a single structure for consistent typography across plots.
///
/// # Intended Usage
///
/// Pass a `FontSizes` instance to [`PlotParams`] to override the default sizes.
/// Use [`Default::default()`] for sensible sizes (title: 16, axis labels: 12, etc.).
///
/// # Invariants
///
/// All fields are `NonZeroUsize` — font sizes must be positive integers.
#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub struct FontSizes {
    /// Title font size in points.
    pub title: NonZeroUsize,
    /// Axis label font size in points.
    pub axis_labels: NonZeroUsize,
    /// Tick label font size in points.
    pub ticks: NonZeroUsize,
    /// Legend text font size in points.
    pub legend: NonZeroUsize,
    /// Super-title font size in points (used for multi-panel plots).
    pub super_title: NonZeroUsize,
    /// Miscellaneous text (annotations, etc.) font size in points.
    pub misc: NonZeroUsize,
}

impl FontSizes {
    /// Constructs a `FontSizes` with explicit values for all fields.
    ///
    /// # Arguments
    ///
    /// - `title` – Title font size in points.
    /// - `axis_labels` – Axis label font size in points.
    /// - `ticks` – Tick label font size in points.
    /// - `legend` – Legend text font size in points.
    /// - `super_title` – Super-title font size in points.
    /// - `misc` – Miscellaneous text font size in points.
    ///
    /// # Returns
    ///
    /// A new `FontSizes` instance.
    #[inline]
    #[must_use]
    pub const fn new(
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
    #[inline]
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

/// Common parameters for plotting operations.
///
/// # Purpose
///
/// Aggregates visual styling options shared by all plot types: titles, axis labels,
/// font sizes, legend display, and grid visibility. Plot-specific parameters
/// (e.g. waveform color, spectrogram colormap) are defined in the respective
/// `*PlotParams` structs.
///
/// # Intended Usage
///
/// Embedded as a field in [`WaveformPlotParams`], [`SpectrogramPlotParams`], and
/// [`MagnitudeSpectrumParams`]. Set fields directly or use the builder pattern
/// on the containing struct.
///
/// # Invariants
///
/// All `Option<String>` fields default to `None`; the plot renderer supplies
/// sensible defaults (e.g. `"Time (s)"` for waveform x-axis) when fields are unset.
///
/// [`WaveformPlotParams`]: waveform::WaveformPlotParams
/// [`SpectrogramPlotParams`]: spectrograms::SpectrogramPlotParams
/// [`MagnitudeSpectrumParams`]: spectrum::MagnitudeSpectrumParams
#[derive(Debug, Clone, Default)]
#[non_exhaustive]
pub struct PlotParams {
    /// Main plot title.
    pub title: Option<String>,
    /// X-axis label.
    pub x_label: Option<String>,
    /// Y-axis label.
    pub y_label: Option<String>,
    /// Font size overrides. If `None`, defaults are used.
    pub font_sizes: Option<FontSizes>,
    /// Whether to show the legend.
    pub show_legend: bool,
    /// Legend title text.
    pub legend_title: Option<String>,
    /// Super-title for multi-panel plots.
    pub super_title: Option<String>,
    /// Whether to show grid lines.
    pub grid: bool,
}

impl PlotParams {
    /// Constructs a `PlotParams` with explicit values for all fields.
    ///
    /// # Arguments
    ///
    /// - `title` – Main plot title.
    /// - `x_label` – X-axis label.
    /// - `y_label` – Y-axis label.
    /// - `font_sizes` – Font size configuration. If `None`, defaults are used.
    /// - `show_legend` – Whether to display the legend.
    /// - `legend_title` – Legend title text.
    /// - `super_title` – Super-title for multi-panel plots.
    /// - `grid` – Whether to show grid lines.
    ///
    /// # Returns
    ///
    /// A new `PlotParams` instance.
    #[inline]
    #[must_use]
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

/// Plot type discriminator with associated parameters.
///
/// # Purpose
///
/// Tagged union for composite plotting scenarios where multiple plot types
/// may be combined. Currently supports waveform and spectrogram types.
///
/// # Intended Usage
///
/// Used by composite plot builders that need to distinguish and configure
/// multiple plot types in a single layout.
///
/// # Invariants
///
/// Marked `#[non_exhaustive]` — new plot types may be added in future versions.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum PlotType {
    /// Waveform plot with the given parameters.
    Waveform(Box<WaveformPlotParams>),
    /// Spectrogram plot (parameters are separate).
    Spectrogram,
}

/// Strategy for rendering multi-channel audio in a single plot.
///
/// # Purpose
///
/// Defines how to collapse or lay out multiple channels when plotting: averaging
/// into mono, displaying each channel separately in a grid, selecting a single
/// channel, or overlapping all channels on the same axes.
///
/// # Intended Usage
///
/// Pass this as a field in `*PlotParams` structs. The plot rendering code
/// queries the strategy to determine the subplot layout and trace configuration.
///
/// # Invariants
///
/// The `Separate` variant carries a [`Layout`] value that specifies whether
/// separate subplots are arranged vertically or horizontally.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum ChannelManagementStrategy {
    /// Average all channels into a single mono trace.
    Average,
    /// Display each channel in a separate subplot using the specified layout.
    Separate(Layout),
    /// Display only the first channel.
    First,
    /// Display only the last channel.
    Last,
    /// Overlay all channels on the same axes with distinct colors.
    Overlap,
}

impl Default for ChannelManagementStrategy {
    #[inline]
    fn default() -> Self {
        Self::Separate(Layout::default())
    }
}

/// Subplot grid orientation for multi-element plots.
///
/// # Purpose
///
/// Specifies whether separate subplots are stacked vertically (one column) or
/// arranged horizontally (one row).
///
/// # Intended Usage
///
/// Embedded in [`ChannelManagementStrategy::Separate`]. The default is `Vertical`,
/// which stacks channels top-to-bottom with a shared time axis.
///
/// # Invariants
///
/// This enum is exhaustive — only two layout directions are supported.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum Layout {
    /// Stack subplots vertically (one column, multiple rows).
    #[default]
    Vertical,
    /// Arrange subplots horizontally (one row, multiple columns).
    Horizontal,
}

impl<T> AudioPlotting for AudioSamples<'_, T>
where
    T: StandardSample,
{
    #[inline]
    fn plot_waveform(&self, params: &WaveformPlotParams) -> AudioSampleResult<WaveformPlot> {
        create_waveform_plot(self, params)
    }

    #[cfg(feature = "transforms")]
    #[inline]
    fn plot_spectrogram(
        &self,
        params: &SpectrogramPlotParams,
    ) -> AudioSampleResult<SpectrogramPlot> {
        spectrograms::create_spectrogram_plot(self, params)
    }

    #[cfg(not(feature = "transforms"))]
    #[inline]
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
    #[inline]
    fn plot_magnitude_spectrum(
        &self,
        params: &spectrum::MagnitudeSpectrumParams,
    ) -> AudioSampleResult<spectrum::MagnitudeSpectrumPlot> {
        spectrum::create_magnitude_spectrum_plot(self, params)
    }

    #[cfg(not(feature = "transforms"))]
    #[inline]
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
