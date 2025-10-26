//! Plot composition and rendering system.
//!
//! This module provides the PlotComposer for combining multiple plot elements
//! into complex layouts and rendering them to various backends.
//!
//! Now powered by Plotly.rs for professional anti-aliased plotting.


use super::core::*;
use super::elements::*;
use plotly::Plot;
use plotview_rs::ViewerArgs;
use plotview_rs::ViewerHandle;
use crate::operations::types::WindowType;
use crate::AudioSampleError;

/// Main compositor for combining and rendering plot elements
pub struct PlotComposer {
    elements: Vec<Box<dyn PlotElement>>,
    layout: LayoutConfig,
    theme: PlotTheme,
    title: Option<String>,
    size: (u32, u32),
}

impl PlotComposer {
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
    pub fn add_element(mut self, element: impl PlotElement + 'static) -> Self {
        self.elements.push(Box::new(element));
        self
    }

    /// Set the layout configuration
    pub fn with_layout(mut self, layout: LayoutConfig) -> Self {
        self.layout = layout;
        self
    }

    /// Set the theme
    pub fn with_theme(mut self, theme: PlotTheme) -> Self {
        self.theme = theme;
        self
    }

    /// Set the overall title
    pub fn with_title(mut self, title: &str) -> Self {
        self.title = Some(title.to_string());
        self
    }

    /// Set the output size
    pub fn with_size(mut self, size: (u32, u32)) -> Self {
        self.size = size;
        self
    }

    /// Render to an HTML file (interactive)
    pub fn render_to_html(&self, path: &str) -> PlotResult<()> {
        let plot = self.create_plotly_plot()?;
        plot.write_html(path);
        Ok(())
    }

    /// Render to a PNG file (static image)
    pub fn render_to_png(&self, path: &str, width: u32, height: u32) -> PlotResult<()> {
        #[cfg(feature = "static-plots")]
        {
            use plotly_static::{StaticExporterBuilder, ImageFormat};

            let plot = self.create_plotly_plot()?;

            // Convert plot to JSON for plotly_static
            let plot_json = serde_json::to_value(&plot)
                .map_err(|e| crate::AudioSampleError::InvalidInput {
                    msg: format!("Failed to serialize plot for PNG export: {}", e),
                })?;

            // Create static exporter
            let mut exporter = StaticExporterBuilder::default()
                .build()
                .map_err(|e| crate::AudioSampleError::InvalidInput {
                    msg: format!("Failed to create static exporter: {}", e),
                })?;

            // Export PNG directly to file
            let path_buf = std::path::Path::new(path);
            exporter.write_fig(path_buf, &plot_json, ImageFormat::PNG, width as usize, height as usize, 1.0)
                .map_err(|e| crate::AudioSampleError::InvalidInput {
                    msg: format!("Failed to export PNG: {}", e),
                })?;

            Ok(())
        }
        #[cfg(not(feature = "static-plots"))]
        {
            let _ = (width, height);
            // Fallback to HTML
            let html_path = path.replace(".png", ".html");
            self.render_to_html(&html_path)?;
            eprintln!("Note: PNG export requires 'static-plots' feature. Created HTML file instead: {}", html_path);
            Ok(())
        }
    }

    /// Render to an SVG file (vector graphics)
    pub fn render_to_svg(&self, path: &str, width: u32, height: u32) -> PlotResult<()> {
        #[cfg(feature = "static-plots")]
        {
            use plotly_static::{StaticExporterBuilder, ImageFormat};

            let plot = self.create_plotly_plot()?;

            // Convert plot to JSON for plotly_static
            let plot_json = serde_json::to_value(&plot)
                .map_err(|e| crate::AudioSampleError::InvalidInput {
                    msg: format!("Failed to serialize plot for SVG export: {}", e),
                })?;

            // Create static exporter
            let mut exporter = StaticExporterBuilder::default()
                .build()
                .map_err(|e| crate::AudioSampleError::InvalidInput {
                    msg: format!("Failed to create static exporter: {}", e),
                })?;

            // Export to SVG
            let svg_data = exporter.write_to_string(&plot_json, ImageFormat::SVG, width as usize, height as usize, 1.0)
                .map_err(|e| crate::AudioSampleError::InvalidInput {
                    msg: format!("Failed to export SVG: {}", e),
                })?;

            // Write to file
            std::fs::write(path, svg_data)
                .map_err(|e| crate::AudioSampleError::InvalidInput {
                    msg: format!("Failed to write SVG file: {}", e),
                })?;

            Ok(())
        }
        #[cfg(not(feature = "static-plots"))]
        {
            let _ = (width, height);
            // Fallback to HTML
            let html_path = path.replace(".svg", ".html");
            self.render_to_html(&html_path)?;
            eprintln!("Note: SVG export requires 'static-plots' feature. Created HTML file instead: {}", html_path);
            Ok(())
        }
    }

    /// Render to a file - automatically detects format from extension
    pub fn render_to_file(&self, path: &str) -> PlotResult<()> {
        if path.ends_with(".png") {
            self.render_to_png(path, self.size.0, self.size.1)
        } else if path.ends_with(".svg") {
            self.render_to_svg(path, self.size.0, self.size.1)
        } else if path.ends_with(".html") {
            self.render_to_html(path)
        } else {
            // Default to HTML for unknown extensions
            let html_path = format!("{}.html", path);
            self.render_to_html(&html_path)
        }
    }

    /// Show the plot in browser (interactive)
    pub fn show(&self, blocking: bool) -> PlotResult<ViewerHandle> {
        let plot = self.create_plotly_plot()?;

        let mut handle = plotview_rs::launch_viewer_managed(
            ViewerArgs::new()
                .content(&plot.to_html())
                .title(&self.get_display_title())
                .size(self.size.0, self.size.1)
                
        ).map_err(|e| AudioSampleError::InvalidInput {
            msg: format!("Failed to launch viewer: {}", e),
        })?;
        println!("Viewer launched with title: {}", self.get_display_title());
        if blocking {
            // Block until the viewer window is closed
            let exit_status = handle.wait().map_err(|e| AudioSampleError::InvalidInput {
                msg: format!("Viewer process error: {}", e),
            })?;

            if !exit_status.success() {
                return Err(AudioSampleError::InvalidInput {
                    msg: format!("Viewer exited with code: {}", exit_status.code().unwrap_or(-1)),
                });
            }
        }
        println!("Viewer handle returned.");
        Ok(handle)
    }

    pub fn get_display_title(&self) -> String {
      self.title.clone().unwrap_or_else(|| "Audio Sample Plot".to_string())
    }

    /// Create a Plotly plot from the composed elements
    fn create_plotly_plot(&self) -> PlotResult<Plot> {
        if self.elements.is_empty() {
            return Err(crate::AudioSampleError::InvalidInput {
                msg: "No plot elements to render".to_string(),
            });
        }

        let mut plot = Plot::new();

        match self.layout {
            LayoutConfig::Single => self.create_single_panel_plot(&mut plot)?,
            LayoutConfig::VerticalStack => self.create_vertical_stack_plot(&mut plot)?,
            LayoutConfig::HorizontalStack => self.create_horizontal_stack_plot(&mut plot)?,
            LayoutConfig::Grid { rows, cols } => self.create_grid_plot(&mut plot, rows, cols)?,
            LayoutConfig::Custom(_) => {
                return Err(crate::AudioSampleError::InvalidInput {
                    msg: "Custom layouts not yet implemented".to_string(),
                });
            }
        }

        // Apply theme and layout
        let layout = self.theme.to_plotly_layout(self.title.as_deref())
            .width(self.size.0 as usize)
            .height(self.size.1 as usize);

        plot.set_layout(layout);

        Ok(plot)
    }

    /// Create single panel plot with all elements overlaid
    fn create_single_panel_plot(&self, plot: &mut Plot) -> PlotResult<()> {
        // Sort elements by z-order
        let mut indexed_elements: Vec<(usize, &Box<dyn PlotElement>)> =
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
        combined_bounds = combined_bounds.with_margin(0.05);

        let x_axis = self.theme.create_axis("Time (s)")
            .range(vec![combined_bounds.x_min, combined_bounds.x_max]);
        let y_axis = self.theme.create_axis("Amplitude")
            .range(vec![combined_bounds.y_min, combined_bounds.y_max]);

        plot.set_layout(plot.layout().clone().x_axis(x_axis).y_axis(y_axis));

        Ok(())
    }

    /// Create vertical stack plot with subplots (simplified for now)
    fn create_vertical_stack_plot(&self, plot: &mut Plot) -> PlotResult<()> {
        // For now, just overlay all traces like single panel
        // TODO: Implement proper subplot functionality
        self.create_single_panel_plot(plot)
    }

    /// Create horizontal stack plot with subplots
    fn create_horizontal_stack_plot(&self, plot: &mut Plot) -> PlotResult<()> {
        // Simplified implementation for now - just overlay all traces
        // In a full implementation, this would use proper horizontal subplots
        self.create_single_panel_plot(plot)
    }

    /// Create grid layout plot with subplots
    fn create_grid_plot(&self, plot: &mut Plot, _rows: usize, _cols: usize) -> PlotResult<()> {
        // Simplified implementation for now - just overlay all traces
        // In a full implementation, this would use proper grid subplots
        self.create_single_panel_plot(plot)
    }
}

impl Default for PlotComposer {
    fn default() -> Self {
        Self::new()
    }
}

/// Convenience builder functions
impl PlotComposer {
    /// Create a simple waveform plot
    pub fn simple_waveform(
        time_data: Vec<f64>,
        amplitude_data: Vec<f64>,
    ) -> Self {
        let waveform = WaveformPlot::<f32>::new(
            time_data,
            amplitude_data,
            LineStyle::default(),
            PlotMetadata::default(),
        );

        Self::new().add_element(waveform)
    }

    /// Create a waveform with onset overlays
    pub fn waveform_with_onsets(
        time_data: Vec<f64>,
        amplitude_data: Vec<f64>,
        onset_times: Vec<f64>,
        y_range: (f64, f64),
    ) -> Self {
        let waveform = WaveformPlot::<f32>::new(
            time_data,
            amplitude_data,
            LineStyle::default(),
            PlotMetadata::default(),
        );

        let onsets = OnsetMarkers::new(
            onset_times,
            None,
            OnsetConfig::default(),
            y_range,
        );

        Self::new()
            .add_element(waveform)
            .add_element(onsets)
    }

    /// Create a comprehensive audio analysis plot
    pub fn analysis_dashboard(
        time_data: Vec<f64>,
        amplitude_data: Vec<f64>,
        onset_times: Vec<f64>,
        beat_times: Vec<f64>,
        tempo_bpm: Option<f64>,
    ) -> Self {
        let y_range = (
            amplitude_data.iter().fold(f64::INFINITY, |a, &b| a.min(b)),
            amplitude_data.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b)),
        );

        let waveform = WaveformPlot::<f32>::new(
            time_data,
            amplitude_data,
            LineStyle::default(),
            PlotMetadata::default(),
        );

        let onsets = OnsetMarkers::new(
            onset_times,
            None,
            OnsetConfig::default(),
            y_range,
        );

        let beats = BeatMarkers::new(
            beat_times,
            tempo_bpm,
            BeatConfig::default(),
            y_range,
        );

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
    pub fn scientific_theme() -> PlotTheme {
        PlotTheme::scientific()
    }

    /// Create a dark theme for presentations
    pub fn dark_theme() -> PlotTheme {
        PlotTheme::dark()
    }

    /// Standard waveform plot configuration
    pub fn waveform_config() -> (LineStyle, PlotMetadata) {
        let style = LineStyle {
            color: "#1f77b4".to_string(), // Professional blue
            width: 3.0,
            style: LineStyleType::Solid,
        };

        let mut metadata = PlotMetadata::default();
        metadata.x_label = Some("Time (s)".to_string());
        metadata.y_label = Some("Amplitude".to_string());

        (style, metadata)
    }

    /// Standard spectrogram configuration
    pub fn spectrogram_config() -> SpectrogramConfig {
        SpectrogramConfig {
            n_fft: 2048,
            window_size: Some(2048),
            hop_length: Some(512),
            window: WindowType::Hanning,
            colormap: ColorPalette::Viridis,
            db_range: (-80.0, 0.0),
            log_freq: false,
            mel_scale: false,
        }
    }
}