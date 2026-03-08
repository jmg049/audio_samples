use super::PlotUtils;
use crate::AudioSampleResult;
use base64::Engine;
use plotly::Plot;
use std::fmt::Write;
use std::path::Path;

/// Spatial arrangement strategy for combining multiple plots in a single HTML document.
///
/// # Purpose
/// Defines how individual plot components should be positioned relative to each other when
/// generating a composite visualization. Controls the flexbox layout direction and grid
/// configuration for the container HTML.
///
/// # Intended Usage
/// Passed to [`CompositePlot::layout`] to specify the desired arrangement before calling
/// [`build()`]. Different layouts are appropriate for different analysis workflows (e.g.,
/// vertical stacking for waveform above spectrogram, horizontal for side-by-side comparison).
///
/// # Invariants
/// - For `Grid { rows, cols }`, both `rows` and `cols` must be greater than zero.
/// - The number of plots added to a `CompositePlot` should match the grid dimensions for
///   optimal appearance, though this is not enforced.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum CompositeLayout {
    /// Stacks plots vertically (top to bottom). Most common for time-aligned visualizations
    /// such as waveform above spectrogram.
    Vertical,
    /// Arranges plots horizontally (left to right). Useful for side-by-side comparison of
    /// different representations of the same signal.
    Horizontal,
    /// Arranges plots in a custom grid with the specified number of rows and columns.
    /// Plots are filled in row-major order (left-to-right, top-to-bottom).
    Grid {
        /// Number of rows in the grid layout.
        rows: usize,
        /// Number of columns in the grid layout.
        cols: usize,
    },
}

/// Defines the interface for plot types that can be composed into multi-plot layouts.
///
/// # Purpose
/// Provides a common abstraction over different plot types (waveforms, spectrograms, spectra)
/// to enable combining them into composite visualizations. Allows access to the underlying
/// Plotly `Plot` object and declares axis-sharing requirements.
///
/// # Intended Usage
/// Implemented by all concrete plot types ([`super::WaveformPlot`], [`super::SpectrogramPlot`],
/// [`super::MagnitudeSpectrumPlot`]). Users interact with this trait primarily through
/// [`CompositePlot::add_plot`], which accepts any type implementing `PlotComponent`.
///
/// # Invariants
/// - `get_plot()` must return a valid Plotly `Plot` instance.
/// - `requires_shared_x_axis()` must accurately reflect whether the plot represents
///   time-domain data that should align with other time-domain plots.
pub trait PlotComponent {
    /// Returns a reference to the underlying Plotly plot.
    ///
    /// Used internally by composite plot generation to extract plot data.
    fn get_plot(&self) -> &Plot;

    /// Returns a mutable reference to the underlying Plotly plot.
    ///
    /// Allows post-construction modifications before composition.
    fn get_plot_mut(&mut self) -> &mut Plot;

    /// Indicates whether this plot should share the X axis with other time-based plots.
    ///
    /// Returns `true` for time-domain plots (waveforms, spectrograms) and `false` for
    /// frequency-domain plots (magnitude spectra). Used by composite plot layout engines
    /// to determine axis alignment strategies.
    fn requires_shared_x_axis(&self) -> bool;
}

/// Multi-plot composition container for arranging multiple visualizations in a single HTML document.
///
/// # Purpose
/// Combines multiple plot components (waveforms, spectrograms, spectra) into a single HTML
/// file with configurable spatial layout. Each plot is rendered as an independent Plotly
/// visualization embedded in an iframe within a flexbox container.
///
/// # Intended Usage
/// Construct via [`CompositePlot::new()`], add plots with [`add_plot`], optionally configure
/// layout with [`layout`], then finalize with [`build()`]. The result can be saved or displayed
/// using the [`PlotUtils`] trait methods.
///
/// # Invariants
/// - At least one plot must be added before calling [`build()`].
/// - All plots are converted to HTML at the time of addition (via [`add_plot`]).
/// - The composite plot only supports HTML output (not static image formats).
///
/// # Implementation Note
/// Due to limitations in `plotly.rs`, this implementation generates separate HTML plots and
/// combines them in an HTML document using iframes and base64 encoding. Future versions may
/// support proper Plotly subplot composition.
pub struct CompositePlot {
    html_plots: Vec<String>,
    layout: CompositeLayout,
}

impl CompositePlot {
    /// Creates a new empty composite plot with vertical layout.
    ///
    /// # Returns
    /// An empty [`CompositePlot`] instance. Use [`add_plot`] to add visualizations.
    ///
    /// # Example
    /// ```rust,no_run
    /// use audio_samples::operations::plotting::CompositePlot;
    ///
    /// let composite = CompositePlot::new();
    /// ```
    #[inline]
    #[must_use]
    pub const fn new() -> Self {
        Self {
            html_plots: Vec::new(),
            layout: CompositeLayout::Vertical,
        }
    }

    /// Adds a plot component to the composition.
    ///
    /// Converts the plot to HTML at the time of addition and stores it for later rendering.
    /// Plots are arranged in the order they are added.
    ///
    /// # Arguments
    /// * `plot` — Any type implementing [`PlotComponent`] (e.g., [`super::WaveformPlot`],
    ///   [`super::SpectrogramPlot`], [`super::MagnitudeSpectrumPlot`]).
    ///
    /// # Returns
    /// Self for method chaining.
    ///
    /// # Example
    /// ```rust,no_run
    /// # use audio_samples::{AudioSamples, sample_rate};
    /// # use audio_samples::operations::traits::AudioPlotting;
    /// # use audio_samples::operations::plotting::{CompositePlot, WaveformPlotParams};
    /// # use audio_samples::operations::plotting::PlotUtils;
    /// # let audio = AudioSamples::new_mono(ndarray::Array1::from_elem(1000, 0.0f32), sample_rate!(44100)).unwrap();
    /// let waveform = audio.plot_waveform(&WaveformPlotParams::default())?;
    /// let composite = CompositePlot::new()
    ///     .add_plot(waveform);
    /// # Ok::<(), audio_samples::AudioSampleError>(())
    /// ```
    #[inline]
    #[must_use]
    pub fn add_plot<P: PlotComponent + 'static>(mut self, plot: P) -> Self {
        // Convert plot to HTML and store it
        self.html_plots.push(plot.get_plot().to_html());
        self
    }

    /// Sets the spatial layout for the composite plot.
    ///
    /// Configures how individual plots are arranged in the final HTML document. Default is
    /// [`CompositeLayout::Vertical`].
    ///
    /// # Arguments
    /// * `layout` — The desired layout strategy (Vertical, Horizontal, or Grid).
    ///
    /// # Returns
    /// Self for method chaining.
    ///
    /// # Example
    /// ```rust,no_run
    /// use audio_samples::operations::plotting::{CompositePlot, CompositeLayout};
    ///
    /// let composite = CompositePlot::new()
    ///     .layout(CompositeLayout::Horizontal);
    /// ```
    #[inline]
    #[must_use]
    pub const fn layout(mut self, layout: CompositeLayout) -> Self {
        self.layout = layout;
        self
    }

    /// Finalizes the composite plot and validates that at least one plot was added.
    ///
    /// This method must be called before saving or displaying the composite plot. It performs
    /// validation to ensure the composite is non-empty.
    ///
    /// # Returns
    /// Self on success.
    ///
    /// # Errors
    /// Returns [crate::AudioSampleError::Parameter] if no plots have been added.
    ///
    /// # Example
    /// ```rust,no_run
    /// # use audio_samples::{AudioSamples, sample_rate};
    /// # use audio_samples::operations::traits::AudioPlotting;
    /// # use audio_samples::operations::plotting::{CompositePlot, WaveformPlotParams};
    /// # use audio_samples::operations::plotting::PlotUtils;
    /// # let audio = AudioSamples::new_mono(ndarray::Array1::from_elem(1000, 0.0f32), sample_rate!(44100)).unwrap();
    /// # let waveform = audio.plot_waveform(&WaveformPlotParams::default())?;
    /// let composite = CompositePlot::new()
    ///     .add_plot(waveform)
    ///     .build()?;
    /// composite.save("composite.html")?;
    /// # Ok::<(), audio_samples::AudioSampleError>(())
    /// ```
    #[inline]
    pub fn build(self) -> AudioSampleResult<Self> {
        if self.html_plots.is_empty() {
            return Err(crate::AudioSampleError::Parameter(
                crate::ParameterError::InvalidValue {
                    parameter: "plots".to_string(),
                    reason: "Cannot build composite plot with no plots".to_string(),
                },
            ));
        }

        Ok(self)
    }

    /// Generate combined HTML with multiple plots arranged according to layout
    #[inline]
    fn generate_composite_html(&self) -> AudioSampleResult<String> {
        let container_style = match self.layout {
            CompositeLayout::Vertical => "flex-direction: column;",
            CompositeLayout::Horizontal => "flex-direction: row;",
            CompositeLayout::Grid { rows: _, cols: _ } => "flex-direction: row; flex-wrap: wrap;",
        };

        let mut html: String = String::with_capacity(4096); // optional heuristic preallocation

        write!(
            html,
            r#"<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Composite Plot</title>
    <style>
        body {{ margin: 0; padding: 0; }}
        .composite-container {{
            display: flex;
            {container_style}
            width: 100%;
            height: 100vh;
        }}
        .plot-item {{
            flex: 1;
            min-height: 0;
            min-width: 0;
        }}
        .plot-item iframe {{
            width: 100%;
            height: 100%;
            border: none;
        }}
    </style>
</head>
<body>
    <div class="composite-container">
"#
        )?;

        for plot_html in &self.html_plots {
            let encoded = base64::engine::general_purpose::STANDARD.encode(plot_html.as_bytes());

            write!(
                html,
                r#"        <div class="plot-item">
            <iframe src="data:text/html;base64,{encoded}"></iframe>
        </div>
"#
            )?;
        }

        html.push_str(
            r"    </div>
</body>
</html>",
        );

        Ok(html)
    }
}

impl Default for CompositePlot {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl PlotUtils for CompositePlot {
    #[inline]
    fn html(&self) -> AudioSampleResult<String> {
        self.generate_composite_html()
    }

    #[cfg(feature = "html_view")]
    #[inline]
    fn show(&self) -> AudioSampleResult<()> {
        let html = self.html()?;
        html_view::show(html).map_err(|e| {
            crate::AudioSampleError::unsupported(format!("Failed to show plot: {}", e))
        })?;
        Ok(())
    }

    #[inline]
    fn save<P: AsRef<Path>>(&self, path: P) -> AudioSampleResult<()> {
        let path = path.as_ref();
        let extension = path.extension().and_then(|s| s.to_str()).unwrap_or("html");

        match extension.to_lowercase().as_str() {
            "html" => {
                let html = self.html()?;
                std::fs::write(path, html).map_err(|e| {
                    crate::AudioSampleError::unsupported(format!("Failed to write HTML file: {e}"))
                })?;
                Ok(())
            }
            _ => Err(crate::AudioSampleError::Parameter(
                crate::ParameterError::InvalidValue {
                    parameter: "file_extension".to_string(),
                    reason: format!("Composite plots only support HTML output. Got: {extension}"),
                },
            )),
        }
    }
}
