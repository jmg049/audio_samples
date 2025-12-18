//! Plot composition and rendering system.
//!
//! This module provides the PlotComposer for combining multiple plot elements
//! into complex layouts and rendering them to various backends.

use std::path::Path;

use super::core::*;
use super::elements::*;
use crate::AudioSampleError;
use crate::ParameterError;
use crate::RealFloat;
use crate::operations::types::WindowType;
use crate::to_precision;
use html_view;
use plotly::Plot;
use plotly::layout::AxisRange;
use plotly::layout::Layout;

#[cfg(feature = "static-plots")]
use plotly_static::{ImageFormat, StaticExporterBuilder};

/// Handle for a non-blocking plot window.
///
/// Returned by `PlotComposer::show_with_handle()` and allows checking
/// if the window was closed or terminating it programmatically.
pub struct PlotHandle {
    inner: html_view::ViewerHandle,
}

impl PlotHandle {
    /// Check if the plot window has been closed (non-blocking).
    ///
    /// Returns `Ok(Some(status))` if closed, `Ok(None)` if still open.
    pub fn try_wait(&mut self) -> PlotResult<Option<html_view::ViewerExitStatus>> {
        self.inner
            .try_wait()
            .map_err(|e| AudioSampleError::Plotting(crate::error::PlottingError::HtmlViewError(e)))
    }

    /// Wait for the plot window to close (blocking).
    ///
    /// Consumes the handle and blocks until the window is closed.
    pub fn wait(self) -> PlotResult<html_view::ViewerExitStatus> {
        self.inner
            .wait()
            .map_err(|e| AudioSampleError::Plotting(crate::error::PlottingError::HtmlViewError(e)))
    }

    /// Terminate the plot window.
    pub fn terminate(&mut self) -> PlotResult<()> {
        self.inner
            .terminate()
            .map_err(|e| AudioSampleError::Plotting(crate::error::PlottingError::HtmlViewError(e)))
    }
}

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
    pub fn render_to_html<P: AsRef<Path>>(&self, path: P, create_parent: bool) -> PlotResult<()> {
        let plot = self.create_plotly_plot()?;

        if create_parent && let Some(parent) = path.as_ref().parent() {
            std::fs::create_dir_all(parent).map_err(|e| {
                AudioSampleError::Parameter(ParameterError::invalid_value(
                    "file_path",
                    format!("Failed to create parent directories: {}", e),
                ))
            })?;
        }

        plot.write_html(path);
        Ok(())
    }

    #[cfg(feature = "static-plots")]
    /// Render to a PNG file (static image)
    pub fn render_to_png<P: AsRef<Path>>(
        &self,
        path: P,
        width: u32,
        height: u32,
    ) -> PlotResult<()> {
        let path = path.as_ref();
        let plot = self.create_plotly_plot()?;

        // Convert plot to JSON for plotly_static
        let plot_json = serde_json::to_value(&plot).map_err(|e| {
            AudioSampleError::Parameter(ParameterError::invalid_value(
                "plot_data",
                format!("Failed to serialize plot for PNG export: {}", e),
            ))
        })?;

        // Create static exporter
        let mut exporter = StaticExporterBuilder::default().build().map_err(|e| {
            AudioSampleError::Parameter(ParameterError::invalid_value(
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
    pub fn render_to_svg<P: AsRef<Path>>(
        &self,
        path: P,
        width: u32,
        height: u32,
    ) -> PlotResult<()> {
        let path = path.as_ref();
        let plot = self.create_plotly_plot()?;

        // Convert plot to JSON for plotly_static
        let plot_json = serde_json::to_value(&plot).map_err(|e| {
            AudioSampleError::Parameter(ParameterError::invalid_value(
                "plot_data",
                format!("Failed to serialize plot for SVG export: {}", e),
            ))
        })?;

        // Create static exporter
        let mut exporter = StaticExporterBuilder::default().build().map_err(|e| {
            AudioSampleError::Parameter(ParameterError::invalid_value(
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
                AudioSampleError::Parameter(ParameterError::invalid_value(
                    "svg_export",
                    format!("Failed to export SVG: {}", e),
                ))
            })?;

        // Write to file
        std::fs::write(path, svg_data).map_err(|e| {
            AudioSampleError::Parameter(ParameterError::invalid_value(
                "file_path",
                format!("Failed to write SVG file: {}", e),
            ))
        })?;

        Ok(())
    }

    /// Render to a file - automatically detects format from extension
    pub fn render_to_file<P: AsRef<Path>>(&self, path: P, create_parent: bool) -> PlotResult<()> {
        let path = path.as_ref();
        let ext = path.extension().and_then(|s| s.to_str()).unwrap_or("");

        // Handle parent directory creation for all formats consistently
        if create_parent && let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| {
                AudioSampleError::Parameter(ParameterError::invalid_value(
                    "file_path",
                    format!("Failed to create parent directories: {}", e),
                ))
            })?;
        }

        #[cfg(feature = "static-plots")]
        {
            match ext {
                "png" => self.render_to_png(path, self.size.0, self.size.1),
                "svg" => self.render_to_svg(path, self.size.0, self.size.1),
                "html" => self.render_to_html(path, false), // Don't duplicate parent creation
                _ => {
                    // Default to HTML for unknown extensions
                    self.render_to_html(path.with_extension("html"), false)
                }
            }
        }
        #[cfg(not(feature = "static-plots"))]
        {
            // If static plots are not enabled, only HTML is supported
            if ext == "html" {
                self.render_to_html(path, false) // Don't duplicate parent creation
            } else {
                // Default to HTML for unknown extensions
                let html_path = path.with_extension("html");
                println!(
                    "Static plot rendering not enabled. Defaulting to HTML output at {}",
                    html_path.display()
                );
                println!(
                    "To enable static plot rendering, compile with the 'static-plots' feature."
                );
                self.render_to_html(&html_path, false) // Don't duplicate parent creation
            }
        }
    }

    /// Show the plot in a window.
    ///
    /// # Arguments
    /// * `blocking` - If `true`, blocks until the window is closed (like matplotlib's `plt.show()`).
    ///                If `false`, returns immediately while the window remains open.
    ///
    /// # Example
    /// ```rust,ignore
    /// // Blocking mode - waits for user to close window
    /// composer.show(true)?;
    ///
    /// // Non-blocking mode - returns immediately
    /// composer.show(false)?;
    /// // ... do other work while window is open
    /// ```
    pub fn show(&self, blocking: bool) -> PlotResult<()> {
        let plot = self.create_plotly_plot()?;
        let html_content = plot.to_html();

        let mut options = html_view::ViewerOptions::inline_html(html_content);
        options.window.title = Some(self.get_display_title());
        options.window.width = Some(self.size.0);
        options.window.height = Some(self.size.1);

        if blocking {
            options.wait = html_view::ViewerWaitMode::Blocking;
        } else {
            options.wait = html_view::ViewerWaitMode::NonBlocking;
        }

        match html_view::open(options) {
            Ok(html_view::ViewerResult::Blocking(_status)) => Ok(()),
            Ok(html_view::ViewerResult::NonBlocking(_handle)) => {
                // In non-blocking mode, we don't wait for the window to close.
                // The handle could be stored to check status later, but for now
                // we just let it go - the window will stay open until the user closes it.
                Ok(())
            }
            Err(e) => Err(AudioSampleError::Plotting(
                crate::error::PlottingError::HtmlViewError(e),
            )),
        }
    }

    /// Show the plot in a window and return a handle for non-blocking control.
    ///
    /// This allows you to check if the window was closed or terminate it programmatically.
    ///
    /// # Returns
    /// A `PlotHandle` that can be used to wait for, poll, or terminate the window.
    ///
    /// # Example
    /// ```rust,ignore
    /// let handle = composer.show_with_handle()?;
    ///
    /// // Do other work...
    ///
    /// // Check if still open (non-blocking)
    /// if handle.try_wait()?.is_some() {
    ///     println!("Window was closed");
    /// }
    ///
    /// // Or wait for it to close (blocking)
    /// handle.wait()?;
    /// ```
    pub fn show_with_handle(&self) -> PlotResult<PlotHandle> {
        let plot = self.create_plotly_plot()?;
        let html_content = plot.to_html();

        let mut options = html_view::ViewerOptions::inline_html(html_content);
        options.window.title = Some(self.get_display_title());
        options.window.width = Some(self.size.0);
        options.window.height = Some(self.size.1);
        options.wait = html_view::ViewerWaitMode::NonBlocking;

        match html_view::open(options) {
            Ok(html_view::ViewerResult::NonBlocking(handle)) => Ok(PlotHandle { inner: handle }),
            Ok(html_view::ViewerResult::Blocking(_)) => {
                // This shouldn't happen since we set NonBlocking mode
                Err(AudioSampleError::Plotting(
                    crate::error::PlottingError::PlotCreation {
                        reason: "Expected non-blocking result but got blocking".to_string(),
                    },
                ))
            }
            Err(e) => Err(AudioSampleError::Plotting(
                crate::error::PlottingError::HtmlViewError(e),
            )),
        }
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

        // Start with themed layout - this preserves theme settings
        let mut layout = self
            .theme
            .to_plotly_layout(self.title.as_deref())
            .width(self.size.0 as usize)
            .height(self.size.1 as usize);

        match self.layout {
            LayoutConfig::Single => self.create_single_panel_plot(&mut plot, &mut layout)?,
            LayoutConfig::VerticalStack => {
                self.create_vertical_stack_plot(&mut plot, &mut layout)?
            }
            LayoutConfig::HorizontalStack => {
                self.create_horizontal_stack_plot(&mut plot, &mut layout)?
            }
            LayoutConfig::Grid { rows, cols } => {
                self.create_grid_plot(&mut plot, &mut layout, rows, cols)?
            }
            LayoutConfig::Custom(_) => {
                return Err(crate::AudioSampleError::Parameter(
                    ParameterError::invalid_value(
                        "layout_config",
                        "Custom layouts not yet implemented",
                    ),
                ));
            }
        }

        plot.set_layout(layout);

        Ok(plot)
    }

    /// Calculate global combined bounds across all elements for shared axes
    fn calculate_global_bounds(&self) -> PlotResult<PlotBounds<F>> {
        if self.elements.is_empty() {
            return Err(crate::AudioSampleError::Parameter(
                ParameterError::invalid_value("plot_elements", "No elements to calculate bounds"),
            ));
        }

        let mut global_bounds = self.elements[0].data_bounds();
        for element in &self.elements[1..] {
            global_bounds.expand_to_include(&element.data_bounds());
        }
        global_bounds = global_bounds.with_margin(to_precision::<F, _>(0.05));
        Ok(global_bounds)
    }

    /// Get axis labels for an element, with fallbacks to defaults
    fn get_axis_labels(&self, _element: &dyn PlotElement<F>) -> (String, String) {
        // Try to get labels from element metadata if available
        // For now, we'll use defaults but this can be extended when metadata access is added
        // TODO: When PlotElement trait has metadata() method, use:
        // if let Some(metadata) = element.metadata() {
        //     let x_label = metadata.x_label.clone().unwrap_or_else(|| "Time (s)".to_string());
        //     let y_label = metadata.y_label.clone().unwrap_or_else(|| "Amplitude".to_string());
        //     (x_label, y_label)
        // } else {
        //     ("Time (s)".to_string(), "Amplitude".to_string())
        // }

        // For now, return sensible defaults - this is better than hard-coding everywhere
        ("Time (s)".to_string(), "Amplitude".to_string())
    }

    /// Create single panel plot with all elements overlaid
    fn create_single_panel_plot(&self, plot: &mut Plot, layout: &mut Layout) -> PlotResult<()> {
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
        let combined_bounds = self.calculate_global_bounds()?;
        let (x_label, y_label) = self.get_axis_labels(self.elements[0].as_ref());
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
        let x_axis = self.theme.create_axis(&x_label).range(x_axis_range);
        let y_axis = self.theme.create_axis(&y_label).range(y_axis_range);

        *layout = layout.clone().x_axis(x_axis).y_axis(y_axis);

        Ok(())
    }

    /// Create vertical stack plot with subplots
    fn create_vertical_stack_plot(&self, plot: &mut Plot, layout: &mut Layout) -> PlotResult<()> {
        let num_subplots = self.elements.len();

        // Use addition order for stacked layouts (not z-order)
        let indexed_elements: Vec<(usize, &Box<dyn PlotElement<F>>)> =
            self.elements.iter().enumerate().collect();

        // Calculate subplot domains for vertical stacking
        let gap = 0.02; // Small gap between subplots
        let available_height = 1.0 - gap * (num_subplots - 1) as f64;
        let subplot_height = available_height / num_subplots as f64;

        // Calculate global x-bounds for shared time axis across all subplots
        let global_bounds = self.calculate_global_bounds()?;

        // Note: layout is already initialized with theme settings by create_plotly_plot

        // Add traces with proper axis assignments and domain configuration
        for (subplot_idx, (_, element)) in indexed_elements.iter().enumerate() {
            let traces = element.to_plotly_traces();

            // Calculate domain coordinates for vertical stacking (top to bottom)
            let y_start = (num_subplots - 1 - subplot_idx) as f64 * (subplot_height + gap);
            let y_end = y_start + subplot_height;

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

            // Configure axes for this subplot
            let element_bounds = element
                .data_bounds()
                .with_margin(to_precision::<F, _>(0.05));
            let (x_label, y_label) = self.get_axis_labels(element.as_ref());

            if subplot_idx == 0 {
                // Configure default axes
                let x_axis = self
                    .theme
                    .create_axis(&x_label)
                    .domain(&[0.0, 1.0]) // Full width
                    .range(vec![
                        global_bounds
                            .x_min
                            .to_f64()
                            .expect("Float conversion should not fail"),
                        global_bounds
                            .x_max
                            .to_f64()
                            .expect("Float conversion should not fail"),
                    ]);
                let y_axis = self
                    .theme
                    .create_axis(&y_label)
                    .domain(&[y_start, y_end])
                    .range(vec![
                        element_bounds
                            .y_min
                            .to_f64()
                            .expect("Float conversion should not fail"),
                        element_bounds
                            .y_max
                            .to_f64()
                            .expect("Float conversion should not fail"),
                    ]);

                *layout = layout.clone().x_axis(x_axis).y_axis(y_axis);
            } else {
                // Configure numbered axes
                let x_axis = self
                    .theme
                    .create_axis(&x_label)
                    .domain(&[0.0, 1.0]) // Full width
                    .range(vec![
                        global_bounds
                            .x_min
                            .to_f64()
                            .expect("Float conversion should not fail"),
                        global_bounds
                            .x_max
                            .to_f64()
                            .expect("Float conversion should not fail"),
                    ]);
                let y_axis = self
                    .theme
                    .create_axis(&y_label)
                    .domain(&[y_start, y_end])
                    .range(vec![
                        element_bounds
                            .y_min
                            .to_f64()
                            .expect("Float conversion should not fail"),
                        element_bounds
                            .y_max
                            .to_f64()
                            .expect("Float conversion should not fail"),
                    ]);

                // Add axes to layout using the specific axis methods
                *layout = match subplot_idx + 1 {
                    2 => layout.clone().x_axis2(x_axis).y_axis2(y_axis),
                    3 => layout.clone().x_axis3(x_axis).y_axis3(y_axis),
                    4 => layout.clone().x_axis4(x_axis).y_axis4(y_axis),
                    5 => layout.clone().x_axis5(x_axis).y_axis5(y_axis),
                    6 => layout.clone().x_axis6(x_axis).y_axis6(y_axis),
                    7 => layout.clone().x_axis7(x_axis).y_axis7(y_axis),
                    8 => layout.clone().x_axis8(x_axis).y_axis8(y_axis),
                    _ => {
                        return Err(crate::AudioSampleError::Parameter(
                            ParameterError::invalid_value(
                                "stack_size",
                                "Vertical stack layout supports maximum 8 subplots",
                            ),
                        ));
                    }
                };
            }
        }

        Ok(())
    }

    /// Create horizontal stack plot with subplots
    fn create_horizontal_stack_plot(&self, plot: &mut Plot, layout: &mut Layout) -> PlotResult<()> {
        let num_subplots = self.elements.len();

        // Use addition order for stacked layouts (not z-order)
        let indexed_elements: Vec<(usize, &Box<dyn PlotElement<F>>)> =
            self.elements.iter().enumerate().collect();

        // Calculate subplot domains
        let gap = 0.02; // Small gap between subplots
        let plot_width = (1.0 - gap * (num_subplots - 1) as f64) / num_subplots as f64;

        // Calculate global x-bounds for shared time axis across all subplots
        let global_bounds = self.calculate_global_bounds()?;

        // Note: layout is already initialized with theme settings by create_plotly_plot

        // Add traces with proper axis assignments and domain configuration
        for (subplot_idx, (_, element)) in indexed_elements.iter().enumerate() {
            let traces = element.to_plotly_traces();

            // Calculate domain coordinates for horizontal stacking (left to right)
            let x_start = subplot_idx as f64 * (plot_width + gap);
            let x_end = x_start + plot_width;

            // Assign traces to the appropriate subplot axes
            for mut trace in traces {
                if subplot_idx == 0 {
                    // First subplot uses default axes (x, y)
                    trace.add_to_plot(plot);
                } else {
                    // Subsequent subplots use numbered axes
                    let x_axis_name = format!("x{}", subplot_idx + 1);
                    let y_axis_name = format!("y{}", subplot_idx + 1);
                    trace = trace.x_axis(&x_axis_name).y_axis(&y_axis_name);
                    trace.add_to_plot(plot);
                }
            }

            // Configure axes for this subplot
            let element_bounds = element
                .data_bounds()
                .with_margin(to_precision::<F, _>(0.05));
            let (x_label, y_label) = self.get_axis_labels(element.as_ref());

            if subplot_idx == 0 {
                // Configure default axes
                let x_axis = self
                    .theme
                    .create_axis(&x_label)
                    .domain(&[x_start, x_end])
                    .range(vec![
                        global_bounds
                            .x_min
                            .to_f64()
                            .expect("Float conversion should not fail"),
                        global_bounds
                            .x_max
                            .to_f64()
                            .expect("Float conversion should not fail"),
                    ]);
                let y_axis = self
                    .theme
                    .create_axis(&y_label)
                    .domain(&[0.0, 1.0]) // Full height
                    .range(vec![
                        element_bounds
                            .y_min
                            .to_f64()
                            .expect("Float conversion should not fail"),
                        element_bounds
                            .y_max
                            .to_f64()
                            .expect("Float conversion should not fail"),
                    ]);

                *layout = layout.clone().x_axis(x_axis).y_axis(y_axis);
            } else {
                // Configure numbered axes
                let x_axis = self
                    .theme
                    .create_axis(&x_label)
                    .domain(&[x_start, x_end])
                    .range(vec![
                        global_bounds
                            .x_min
                            .to_f64()
                            .expect("Float conversion should not fail"),
                        global_bounds
                            .x_max
                            .to_f64()
                            .expect("Float conversion should not fail"),
                    ]);
                let y_axis = self
                    .theme
                    .create_axis(&y_label)
                    .domain(&[0.0, 1.0]) // Full height
                    .range(vec![
                        element_bounds
                            .y_min
                            .to_f64()
                            .expect("Float conversion should not fail"),
                        element_bounds
                            .y_max
                            .to_f64()
                            .expect("Float conversion should not fail"),
                    ]);

                // Add axes to layout using the specific axis methods
                *layout = match subplot_idx + 1 {
                    2 => layout.clone().x_axis2(x_axis).y_axis2(y_axis),
                    3 => layout.clone().x_axis3(x_axis).y_axis3(y_axis),
                    4 => layout.clone().x_axis4(x_axis).y_axis4(y_axis),
                    5 => layout.clone().x_axis5(x_axis).y_axis5(y_axis),
                    6 => layout.clone().x_axis6(x_axis).y_axis6(y_axis),
                    7 => layout.clone().x_axis7(x_axis).y_axis7(y_axis),
                    8 => layout.clone().x_axis8(x_axis).y_axis8(y_axis),
                    _ => {
                        return Err(crate::AudioSampleError::Parameter(
                            ParameterError::invalid_value(
                                "stack_size",
                                "Horizontal stack layout supports maximum 8 subplots",
                            ),
                        ));
                    }
                };
            }
        }

        Ok(())
    }

    /// Create grid layout plot with subplots
    fn create_grid_plot(
        &self,
        plot: &mut Plot,
        layout: &mut Layout,
        rows: usize,
        cols: usize,
    ) -> PlotResult<()> {
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

        // Use addition order for stacked layouts (not z-order)
        let indexed_elements: Vec<(usize, &Box<dyn PlotElement<F>>)> =
            self.elements.iter().enumerate().collect();

        // Calculate subplot domains with gaps
        let gap_x = 0.02; // Small gap between horizontal subplots
        let gap_y = 0.02; // Small gap between vertical subplots

        // Available space after accounting for gaps
        let available_width = 1.0 - gap_x * (cols - 1) as f64;
        let available_height = 1.0 - gap_y * (rows - 1) as f64;

        // Individual subplot dimensions
        let subplot_width = available_width / cols as f64;
        let subplot_height = available_height / rows as f64;

        // Calculate global x-bounds for shared time axis across all subplots
        let global_bounds = self.calculate_global_bounds()?;

        // Note: layout is already initialized with theme settings by create_plotly_plot

        // Add traces and configure axes for each element
        for (element_idx, (_, element)) in indexed_elements.iter().enumerate() {
            let traces = element.to_plotly_traces();

            // Calculate grid position (row, col)
            let row = element_idx / cols;
            let col = element_idx % cols;

            // Calculate domain coordinates
            // X domain: left to right
            let x_start = col as f64 * (subplot_width + gap_x);
            let x_end = x_start + subplot_width;

            // Y domain: top to bottom (plotly uses bottom-to-top, so we invert)
            let y_start = (rows - 1 - row) as f64 * (subplot_height + gap_y);
            let y_end = y_start + subplot_height;

            // Assign traces to the appropriate subplot axes
            for mut trace in traces {
                if element_idx == 0 {
                    // First subplot uses default axes (x, y)
                    trace.add_to_plot(plot);
                } else {
                    // Subsequent subplots use numbered axes
                    let x_axis_name = format!("x{}", element_idx + 1);
                    let y_axis_name = format!("y{}", element_idx + 1);
                    trace = trace.x_axis(&x_axis_name).y_axis(&y_axis_name);
                    trace.add_to_plot(plot);
                }
            }

            // Configure axes for this subplot with domain positioning
            let element_bounds = element
                .data_bounds()
                .with_margin(to_precision::<F, _>(0.05));
            let (x_label, y_label) = self.get_axis_labels(element.as_ref());

            if element_idx == 0 {
                // Configure default axes
                let x_axis = self
                    .theme
                    .create_axis(&x_label)
                    .domain(&[x_start, x_end])
                    .range(vec![
                        global_bounds
                            .x_min
                            .to_f64()
                            .expect("Float conversion should not fail"),
                        global_bounds
                            .x_max
                            .to_f64()
                            .expect("Float conversion should not fail"),
                    ]);
                let y_axis = self
                    .theme
                    .create_axis(&y_label)
                    .domain(&[y_start, y_end])
                    .range(vec![
                        element_bounds
                            .y_min
                            .to_f64()
                            .expect("Float conversion should not fail"),
                        element_bounds
                            .y_max
                            .to_f64()
                            .expect("Float conversion should not fail"),
                    ]);

                *layout = layout.clone().x_axis(x_axis).y_axis(y_axis);
            } else {
                // Configure numbered axes
                let x_axis = self
                    .theme
                    .create_axis(&x_label)
                    .domain(&[x_start, x_end])
                    .range(vec![
                        global_bounds
                            .x_min
                            .to_f64()
                            .expect("Float conversion should not fail"),
                        global_bounds
                            .x_max
                            .to_f64()
                            .expect("Float conversion should not fail"),
                    ]);
                let y_axis = self
                    .theme
                    .create_axis(&y_label)
                    .domain(&[y_start, y_end])
                    .range(vec![
                        element_bounds
                            .y_min
                            .to_f64()
                            .expect("Float conversion should not fail"),
                        element_bounds
                            .y_max
                            .to_f64()
                            .expect("Float conversion should not fail"),
                    ]);

                // Add axes to layout using the specific axis methods
                *layout = match element_idx + 1 {
                    2 => layout.clone().x_axis2(x_axis).y_axis2(y_axis),
                    3 => layout.clone().x_axis3(x_axis).y_axis3(y_axis),
                    4 => layout.clone().x_axis4(x_axis).y_axis4(y_axis),
                    5 => layout.clone().x_axis5(x_axis).y_axis5(y_axis),
                    6 => layout.clone().x_axis6(x_axis).y_axis6(y_axis),
                    7 => layout.clone().x_axis7(x_axis).y_axis7(y_axis),
                    8 => layout.clone().x_axis8(x_axis).y_axis8(y_axis),
                    _ => {
                        return Err(crate::AudioSampleError::Parameter(
                            ParameterError::invalid_value(
                                "grid_size",
                                "Grid layout supports maximum 8 subplots",
                            ),
                        ));
                    }
                };
            }
        }

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

        let beats = BeatMarkers::new(beat_times, tempo_bpm, BeatPlotConfig::default(), y_range);

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
