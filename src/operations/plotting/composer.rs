//! Plot composition and rendering system.
//!
//! This module provides the PlotComposer for combining multiple plot elements
//! into complex layouts and rendering them to various backends.

use super::core::*;
use super::elements::*;
use crate::ParameterError;
use crate::RealFloat;
use crate::operations::types::WindowType;
use crate::to_precision;
#[cfg(feature = "plotting")]
use plotly::Plot;
use plotly::layout::AxisRange;
#[cfg(feature = "plotting")]
use plotly::layout::{GridPattern, Layout, LayoutGrid, RowOrder};
// #[cfg(feature = "plotting")]
// use plotview_rs::ViewerArgs;
// #[cfg(feature = "plotting")]
// use plotview_rs::ViewerHandle;

/// Main compositor for combining and rendering plot elements
pub struct PlotComposer<F: RealFloat> {
    elements: Vec<Box<dyn PlotElement<F>>>,
    layout: LayoutConfig,
    theme: PlotTheme<F>,
    title: Option<String>,
    size: (u32, u32),
}

impl<F: RealFloat> PlotComposer<F> {
    /// Create a new plot composer
    pub fn new() -> Self {
        Self {
            elements: Vec::new(),
            layout: LayoutConfig::Single,
            theme: PlotTheme::high_quality(),
            title: None,
            size: (1600, 1200), // Much higher default resolution
        }
    }

    /// Add a plot element to the composition
    pub fn add_element(mut self, element: impl PlotElement<F> + 'static) -> Self {
        self.elements.push(Box::new(element));
        self
    }

    /// Set the layout configuration
    pub const fn with_layout(mut self, layout: LayoutConfig) -> Self {
        self.layout = layout;
        self
    }

    /// Set the theme
    pub fn with_theme(mut self, theme: PlotTheme<F>) -> Self {
        self.theme = theme;
        self
    }

    /// Set the overall title
    pub fn with_title(mut self, title: &str) -> Self {
        self.title = Some(title.to_string());
        self
    }

    /// Set the output size
    pub const fn with_size(mut self, size: (u32, u32)) -> Self {
        self.size = size;
        self
    }

    /// Render to an HTML file (interactive)
    pub fn render_to_html(&self, path: &str) -> PlotResult<()> {
        let plot = self.create_plotly_plot()?;
        plot.write_html(path);
        Ok(())
    }

    #[cfg(feature = "static-plots")]
    /// Render to a PNG file (static image)
    pub fn render_to_png(&self, path: &str, width: u32, height: u32) -> PlotResult<()> {
        use plotly_static::{ImageFormat, StaticExporterBuilder};

        let plot = self.create_plotly_plot()?;

        // Convert plot to JSON for plotly_static
        let plot_json = serde_json::to_value(&plot).map_err(|e| {
            crate::AudioSampleError::Parameter(ParameterError::invalid_value(
                "plot_data",
                format!("Failed to serialize plot for PNG export: {}", e),
            ))
        })?;

        // Create static exporter
        let mut exporter = StaticExporterBuilder::default().build().map_err(|e| {
            crate::AudioSampleError::Parameter(ParameterError::invalid_value(
                "exporter",
                format!("Failed to create static exporter: {}", e),
            ))
        })?;

        // Export PNG directly to file
        let path_buf = std::path::Path::new(path);
        exporter
            .write_fig(
                path_buf,
                &plot_json,
                ImageFormat::PNG,
                width as usize,
                height as usize,
                1.0,
            )
            .map_err(|e| {
                crate::AudioSampleError::Parameter(ParameterError::invalid_value(
                    "png_export",
                    format!("Failed to export PNG: {}", e),
                ))
            })?;

        Ok(())
    }
    #[cfg(feature = "static-plots")]
    /// Render to an SVG file (vector graphics)
    pub fn render_to_svg(&self, path: &str, width: u32, height: u32) -> PlotResult<()> {
        use plotly_static::{ImageFormat, StaticExporterBuilder};

        let plot = self.create_plotly_plot()?;

        // Convert plot to JSON for plotly_static
        let plot_json = serde_json::to_value(&plot).map_err(|e| {
            crate::AudioSampleError::Parameter(ParameterError::invalid_value(
                "plot_data",
                format!("Failed to serialize plot for SVG export: {}", e),
            ))
        })?;

        // Create static exporter
        let mut exporter = StaticExporterBuilder::default().build().map_err(|e| {
            crate::AudioSampleError::Parameter(ParameterError::invalid_value(
                "exporter",
                format!("Failed to create static exporter: {}", e),
            ))
        })?;

        // Export to SVG
        let svg_data = exporter
            .write_to_string(
                &plot_json,
                ImageFormat::SVG,
                width as usize,
                height as usize,
                1.0,
            )
            .map_err(|e| {
                crate::AudioSampleError::Parameter(ParameterError::invalid_value(
                    "svg_export",
                    format!("Failed to export SVG: {}", e),
                ))
            })?;

        // Write to file
        std::fs::write(path, svg_data).map_err(|e| {
            crate::AudioSampleError::Parameter(ParameterError::invalid_value(
                "file_path",
                format!("Failed to write SVG file: {}", e),
            ))
        })?;

        Ok(())
    }

    /// Render to a file - automatically detects format from extension
    pub fn render_to_file(&self, path: &str) -> PlotResult<()> {
        let ext = path.split('.').next_back().unwrap_or("");

        #[cfg(feature = "static-plots")]
        {
            match ext {
                "png" => self.render_to_png(path, self.size.0, self.size.1),
                "svg" => self.render_to_svg(path, self.size.0, self.size.1),
                "html" => self.render_to_html(path),
                _ => {
                    // Default to HTML for unknown extensions
                    let html_path = format!("{}.html", path);
                    self.render_to_html(&html_path)
                }
            }
        }
        #[cfg(not(feature = "static-plots"))]
        {
            // If static plots are not enabled, only HTML is supported
            if ext == "html" {
                self.render_to_html(path)
            } else {
                // Default to HTML for unknown extensions
                let html_path = format!("{}.html", path);
                println!(
                    "Static plot rendering not enabled. Defaulting to HTML output at {}",
                    html_path
                );
                println!(
                    "To enable static plot rendering, compile with the 'static-plots' feature."
                );
                self.render_to_html(&html_path)
            }
        }
    }

    /// Show the plot in browser (interactive)
    pub fn show(&self, _blocking: bool) -> PlotResult<()> {
        let _plot = self.create_plotly_plot()?;
        println!(
            "Plot ready to display with title: {}",
            self.get_display_title()
        );
        // TODO: Implement actual viewer when plotview_rs is available
        Ok(())
    }

    /// Gets the display title for the plot, using a default if none is set.
    ///
    /// # Returns
    /// The plot title string, or "Audio Sample Plot" if no title was specified
    pub fn get_display_title(&self) -> String {
        self.title
            .clone()
            .unwrap_or_else(|| "Audio Sample Plot".to_string())
    }

    /// Create a Plotly plot from the composed elements
    fn create_plotly_plot(&self) -> PlotResult<Plot> {
        if self.elements.is_empty() {
            return Err(crate::AudioSampleError::Parameter(
                ParameterError::invalid_value("plot_elements", "No plot elements to render"),
            ));
        }

        let mut plot = Plot::new();

        match self.layout {
            LayoutConfig::Single => self.create_single_panel_plot(&mut plot)?,
            LayoutConfig::VerticalStack => self.create_vertical_stack_plot(&mut plot)?,
            LayoutConfig::HorizontalStack => self.create_horizontal_stack_plot(&mut plot)?,
            LayoutConfig::Grid { rows, cols } => self.create_grid_plot(&mut plot, rows, cols)?,
            LayoutConfig::Custom(_) => {
                return Err(crate::AudioSampleError::Parameter(
                    ParameterError::invalid_value(
                        "layout_config",
                        "Custom layouts not yet implemented",
                    ),
                ));
            }
        }

        // Apply theme and layout
        let layout = self
            .theme
            .to_plotly_layout(self.title.as_deref())
            .width(self.size.0 as usize)
            .height(self.size.1 as usize);

        plot.set_layout(layout);

        Ok(plot)
    }

    /// Create single panel plot with all elements overlaid
    fn create_single_panel_plot(&self, plot: &mut Plot) -> PlotResult<()> {
        // Sort elements by z-order
        let mut indexed_elements: Vec<(usize, &Box<dyn PlotElement<F>>)> =
            self.elements.iter().enumerate().collect();
        indexed_elements.sort_by_key(|(_, elem)| elem.z_order());

        // Add all traces from all elements
        for (_, element) in indexed_elements {
            let traces = element.to_plotly_traces();
            for trace in traces {
                trace.add_to_plot(plot);
            }
        }

        // Set up axes based on combined bounds
        let mut combined_bounds = self.elements[0].data_bounds();
        for element in &self.elements[1..] {
            combined_bounds.expand_to_include(&element.data_bounds());
        }
        combined_bounds = combined_bounds.with_margin(to_precision::<F, _>(0.05));
        let x_axis_range = AxisRange::new(
            combined_bounds
                .x_min
                .to_f64()
                .expect("Float conversion should not fail"),
            combined_bounds
                .x_max
                .to_f64()
                .expect("Float conversion should not fail"),
        );
        let y_axis_range = AxisRange::new(
            combined_bounds
                .y_min
                .to_f64()
                .expect("Float conversion should not fail"),
            combined_bounds
                .y_max
                .to_f64()
                .expect("Float conversion should not fail"),
        );
        let x_axis = self.theme.create_axis("Time (s)").range(x_axis_range);
        let y_axis = self.theme.create_axis("Amplitude").range(y_axis_range);

        plot.set_layout(plot.layout().clone().x_axis(x_axis).y_axis(y_axis));

        Ok(())
    }

    /// Create vertical stack plot with subplots - following plotly.rs book exactly
    fn create_vertical_stack_plot(&self, plot: &mut Plot) -> PlotResult<()> {
        if self.elements.is_empty() {
            return Ok(());
        }

        let num_subplots = self.elements.len();

        // Sort elements by z-order
        let mut indexed_elements: Vec<(usize, &Box<dyn PlotElement<F>>)> =
            self.elements.iter().enumerate().collect();
        indexed_elements.sort_by_key(|(_, elem)| elem.z_order());

        // Add traces with proper axis assignments following the plotly.rs book pattern
        for (subplot_idx, (_, element)) in indexed_elements.iter().enumerate() {
            let traces = element.to_plotly_traces();

            for mut trace in traces {
                if subplot_idx == 0 {
                    // First subplot uses default axes (x, y)
                    trace.add_to_plot(plot);
                } else {
                    // Subsequent subplots use numbered axes (x2, y2), (x3, y3), etc.
                    let x_axis_name = format!("x{}", subplot_idx + 1);
                    let y_axis_name = format!("y{}", subplot_idx + 1);
                    trace = trace.x_axis(&x_axis_name).y_axis(&y_axis_name);
                    trace.add_to_plot(plot);
                }
            }
        }

        // Create the grid layout exactly as shown in plotly.rs book
        let layout = Layout::new()
            .grid(
                LayoutGrid::new()
                    .rows(num_subplots)
                    .columns(1)
                    .pattern(GridPattern::Independent),
            )
            .title(self.title.as_deref().unwrap_or(""))
            .width(self.size.0 as usize)
            .height(self.size.1 as usize);

        plot.set_layout(layout);

        Ok(())
    }

    /// Create horizontal stack plot with subplots
    fn create_horizontal_stack_plot(&self, plot: &mut Plot) -> PlotResult<()> {
        if self.elements.is_empty() {
            return Ok(());
        }

        let num_subplots = self.elements.len();

        // Sort elements by z-order
        let mut indexed_elements: Vec<(usize, &Box<dyn PlotElement<F>>)> =
            self.elements.iter().enumerate().collect();
        indexed_elements.sort_by_key(|(_, elem)| elem.z_order());

        // Add traces with proper axis assignments
        for (subplot_idx, (_, element)) in indexed_elements.iter().enumerate() {
            let traces = element.to_plotly_traces();

            // Assign traces to the appropriate subplot axes
            for mut trace in traces {
                if subplot_idx == 0 {
                    // First subplot uses default axes
                    trace.add_to_plot(plot);
                } else {
                    // Subsequent subplots use numbered axes
                    let x_axis_name = format!("x{}", subplot_idx + 1);
                    let y_axis_name = format!("y{}", subplot_idx + 1);
                    trace = trace.x_axis(&x_axis_name).y_axis(&y_axis_name);
                    trace.add_to_plot(plot);
                }
            }
        }

        // Create layout with grid for horizontal stacking
        let layout = Layout::new().grid(
            LayoutGrid::new()
                .rows(1)
                .columns(num_subplots)
                .pattern(GridPattern::Independent),
        );

        plot.set_layout(layout);

        Ok(())
    }

    /// Create grid layout plot with subplots
    fn create_grid_plot(&self, plot: &mut Plot, rows: usize, cols: usize) -> PlotResult<()> {
        if self.elements.is_empty() {
            return Ok(());
        }

        let max_subplots = rows * cols;
        if self.elements.len() > max_subplots {
            return Err(crate::AudioSampleError::Parameter(
                ParameterError::invalid_value(
                    "grid_elements",
                    format!(
                        "Too many elements ({}) for grid layout ({}x{} = {} max)",
                        self.elements.len(),
                        rows,
                        cols,
                        max_subplots
                    ),
                ),
            ));
        }

        // Sort elements by z-order
        let mut indexed_elements: Vec<(usize, &Box<dyn PlotElement<F>>)> =
            self.elements.iter().enumerate().collect();
        indexed_elements.sort_by_key(|(_, elem)| elem.z_order());

        // Add traces with proper axis assignments
        for (element_idx, (_, element)) in indexed_elements.iter().enumerate() {
            let traces = element.to_plotly_traces();

            // Assign traces to the appropriate subplot axes
            for mut trace in traces {
                if element_idx == 0 {
                    // First subplot uses default axes
                    trace.add_to_plot(plot);
                } else {
                    // Subsequent subplots use numbered axes
                    let x_axis_name = format!("x{}", element_idx + 1);
                    let y_axis_name = format!("y{}", element_idx + 1);
                    trace = trace.x_axis(&x_axis_name).y_axis(&y_axis_name);
                    trace.add_to_plot(plot);
                }
            }
        }

        // Create layout with grid
        let layout = Layout::new().grid(
            LayoutGrid::new()
                .rows(rows)
                .columns(cols)
                .pattern(GridPattern::Independent)
                .row_order(RowOrder::TopToBottom),
        );

        plot.set_layout(layout);

        Ok(())
    }
}

impl<F: RealFloat> Default for PlotComposer<F> {
    fn default() -> Self {
        Self::new()
    }
}

/// Convenience builder functions
impl<F: RealFloat> PlotComposer<F> {
    /// Create a simple waveform plot
    pub fn simple_waveform(time_data: Vec<F>, amplitude_data: Vec<F>) -> Self {
        let waveform = WaveformPlot::<F, F>::new(
            time_data,
            amplitude_data,
            LineStyle::default(),
            PlotMetadata::default(),
        );

        Self::new().add_element(waveform)
    }

    /// Create a waveform with onset overlays
    pub fn waveform_with_onsets(
        time_data: Vec<F>,
        amplitude_data: Vec<F>,
        onset_times: Vec<F>,
        y_range: (F, F),
    ) -> Self {
        let waveform = WaveformPlot::<F, F>::new(
            time_data,
            amplitude_data,
            LineStyle::default(),
            PlotMetadata::default(),
        );

        let onsets = OnsetMarkers::new(onset_times, None, OnsetConfig::default(), y_range);

        Self::new().add_element(waveform).add_element(onsets)
    }

    /// Create a comprehensive audio analysis plot
    pub fn analysis_dashboard(
        time_data: Vec<F>,
        amplitude_data: Vec<F>,
        onset_times: Vec<F>,
        beat_times: Vec<F>,
        tempo_bpm: Option<F>,
    ) -> Self {
        let y_range = (
            amplitude_data.iter().fold(F::infinity(), |a, &b| a.min(b)),
            amplitude_data
                .iter()
                .fold(F::neg_infinity(), |a, &b| a.max(b)),
        );

        let waveform = WaveformPlot::<F, F>::new(
            time_data,
            amplitude_data,
            LineStyle::default(),
            PlotMetadata::default(),
        );

        let onsets = OnsetMarkers::new(onset_times, None, OnsetConfig::default(), y_range);

        let beats = BeatMarkers::new(beat_times, tempo_bpm, BeatConfig::default(), y_range);

        Self::new()
            .add_element(waveform)
            .add_element(onsets)
            .add_element(beats)
            .with_layout(LayoutConfig::Single)
            .with_title("Audio Analysis")
    }
}

/// Helper functions for creating common plot configurations
pub mod presets {
    use super::*;

    /// Create a scientific publication style theme
    pub fn scientific_theme<F: RealFloat>() -> PlotTheme<F> {
        PlotTheme::scientific()
    }

    /// Create a dark theme for presentations
    pub fn dark_theme<F: RealFloat>() -> PlotTheme<F> {
        PlotTheme::dark()
    }

    /// Standard waveform plot configuration
    pub fn waveform_config<F: RealFloat>() -> (LineStyle<F>, PlotMetadata) {
        let style = LineStyle {
            color: "#1f77b4".to_string(), // Professional blue
            width: to_precision::<F, _>(3.0),
            style: LineStyleType::Solid,
        };

        let metadata = PlotMetadata {
            x_label: Some("Time (s)".to_string()),
            y_label: Some("Amplitude".to_string()),
            ..Default::default()
        };

        (style, metadata)
    }

    /// Standard spectrogram configuration
    pub fn spectrogram_config<F: RealFloat>() -> SpectrogramConfig<F> {
        SpectrogramConfig {
            n_fft: 2048,
            window_size: Some(2048),
            hop_length: Some(512),
            window: WindowType::Hanning,
            colormap: ColorPalette::Viridis,
            db_range: (to_precision::<F, _>(-80.0), F::zero()),
            log_freq: false,
            mel_scale: false,
        }
    }
}
