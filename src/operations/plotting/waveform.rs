use std::path::{Path, PathBuf};

use plotly::common::{AxisSide, Marker, Mode};
use plotly::layout::Annotation;
use plotly::{
    Plot, Scatter,
    layout::{Axis, GridPattern, LayoutGrid, RowOrder, Shape, ShapeLine, ShapeType},
};

use crate::{
    AudioChannelOps, AudioSampleResult, AudioSamples, AudioTypeConversion, StandardSample,
    operations::{
        plotting::{
            ChannelManagementStrategy, DECIMATE_THRESHOLD, PlotParams, PlotUtils,
            composite::PlotComponent, configure_time_axis, decimate_waveform,
        },
        types::MonoConversionMethod,
    },
};

pub struct WaveformPlot {
    _params: WaveformPlotParams, // What parameters created me?
    plot: Plot,                  // the plotly plot
}

impl PlotUtils for WaveformPlot {
    fn html(&self) -> AudioSampleResult<String> {
        Ok(self.plot.to_html())
    }

    #[cfg(feature = "html_view")]
    fn show(&self) -> AudioSampleResult<()> {
        let html = self.html()?;
        html_view::show(html).map_err(|e| {
            crate::AudioSampleError::unsupported(format!("Failed to show plot: {}", e))
        })?;
        Ok(())
    }

    fn save<P: AsRef<Path>>(&self, path: P) -> AudioSampleResult<()> {
        let path = path.as_ref();
        let extension = path.extension().and_then(|s| s.to_str()).unwrap_or("html");

        // TODO! Make above configurable
        match extension.to_lowercase().as_str() {
            "html" => {
                let html = self.html()?;
                std::fs::write(path, html).map_err(|e| {
                    crate::AudioSampleError::unsupported(format!(
                        "Failed to write HTML file: {}",
                        e
                    ))
                })?;
                Ok(())
            }
            #[cfg(feature = "static-plots")]
            "png" | "svg" | "jpeg" | "jpg" | "webp" => {
                use plotly_static::{ImageFormat, StaticExporterBuilder};
                use serde_json::json;
                let mut static_exporter =
                    StaticExporterBuilder::default().build().map_err(|e| {
                        crate::AudioSampleError::unsupported(format!(
                            "Failed to create static exporter: {}",
                            e
                        ))
                    })?;
                let format = match extension {
                    "png" => ImageFormat::PNG,
                    "svg" => ImageFormat::SVG,
                    "jpeg" | "jpg" => ImageFormat::JPEG,
                    "webp" => ImageFormat::WEBP,
                    _ => ImageFormat::PNG,
                };
                let width = 1920;
                let height = 1080;
                let scale = 1.0;
                static_exporter
                    .write_fig(
                        path,
                        &json!(&self.plot.to_json()),
                        format,
                        width,
                        height,
                        scale,
                    )
                    .map_err(|e| {
                        crate::AudioSampleError::unsupported(format!(
                            "Failed to save static image: {}",
                            e
                        ))
                    })?;

                Ok(())
            }
            #[cfg(not(feature = "static-plots"))]
            "png" | "svg" | "jpeg" | "jpg" | "webp" => Err(crate::AudioSampleError::Feature(
                crate::FeatureError::NotEnabled {
                    feature: "static-plots".to_string(),
                    operation: "save plot as static image (PNG/SVG/etc)".to_string(),
                },
            )),
            _ => Err(crate::AudioSampleError::Parameter(
                crate::ParameterError::InvalidValue {
                    parameter: "file_extension".to_string(),
                    reason: format!(
                        "Unsupported file extension: {}. Supported: html, png, svg, jpeg, jpg, webp",
                        extension
                    ),
                },
            )),
        }
    }
}

impl PlotComponent for WaveformPlot {
    fn get_plot(&self) -> &Plot {
        &self.plot
    }

    fn get_plot_mut(&mut self) -> &mut Plot {
        &mut self.plot
    }

    fn requires_shared_x_axis(&self) -> bool {
        true // Waveforms are time-based
    }
}

impl WaveformPlot {
    /// Add a vertical line at the specified time position.
    ///
    /// # Arguments
    /// * `x` - Time position in seconds
    /// * `label` - Optional label for the line
    /// * `color` - Optional color (CSS color string, defaults to "black")
    ///
    /// # Returns
    /// Self for method chaining
    pub fn add_vline(mut self, x: f64, label: Option<&str>, color: Option<&str>) -> Self {
        let color = color.unwrap_or("black").to_string();

        let shape = Shape::new()
            .shape_type(ShapeType::Line)
            .x0(x)
            .x1(x)
            .y0(0)
            .y1(1)
            .y_ref("paper")
            .line(ShapeLine::new().color(color.clone()).width(2.0));

        let mut layout = self.plot.layout().clone();
        layout.add_shape(shape);

        if let Some(label_text) = label {
            let annotation = Annotation::new()
                .x(x)
                .y(1.0)
                .y_ref("paper")
                .text(label_text)
                .show_arrow(false)
                .y_shift(10.0);
            layout.add_annotation(annotation);
        }

        self.plot.set_layout(layout);
        self
    }

    /// Add a horizontal line at the specified amplitude value.
    ///
    /// # Arguments
    /// * `y` - Amplitude value
    /// * `label` - Optional label for the line
    /// * `color` - Optional color (CSS color string, defaults to "black")
    ///
    /// # Returns
    /// Self for method chaining
    pub fn add_hline(mut self, y: f64, label: Option<&str>, color: Option<&str>) -> Self {
        let color = color.unwrap_or("black").to_string();

        let shape = Shape::new()
            .shape_type(ShapeType::Line)
            .x0(0)
            .x1(1)
            .x_ref("paper")
            .y0(y)
            .y1(y)
            .line(ShapeLine::new().color(color.clone()).width(2.0));

        let mut layout = self.plot.layout().clone();
        layout.add_shape(shape);

        if let Some(label_text) = label {
            let annotation = Annotation::new()
                .x(1.0)
                .x_ref("paper")
                .y(y)
                .text(label_text)
                .show_arrow(false)
                .x_shift(10.0);
            layout.add_annotation(annotation);
        }

        self.plot.set_layout(layout);
        self
    }

    /// Add a marker point at the specified position.
    ///
    /// # Arguments
    /// * `x` - Time position in seconds
    /// * `y` - Amplitude value
    /// * `text` - Optional label text
    /// * `_symbol` - Optional marker symbol (currently unused, reserved for future use)
    ///
    /// # Returns
    /// Self for method chaining
    pub fn add_marker(mut self, x: f64, y: f64, text: Option<&str>, _symbol: Option<&str>) -> Self {
        let marker = Marker::new()
            .size(10)
            .color("red")
            .symbol(plotly::common::MarkerSymbol::Circle);

        let mut trace = Scatter::new(vec![x], vec![y])
            .mode(Mode::Markers)
            .marker(marker)
            .show_legend(false);

        if let Some(label_text) = text {
            trace = trace.text_array(vec![label_text.to_string()]);
        }

        self.plot.add_trace(trace);
        self
    }

    /// Add a shaded vertical region between two time points.
    ///
    /// # Arguments
    /// * `x_start` - Start time in seconds
    /// * `x_end` - End time in seconds
    /// * `color` - Optional color (RGBA string, defaults to "rgba(200,200,200,0.2)")
    /// * `opacity` - Optional opacity override (0.0-1.0)
    ///
    /// # Returns
    /// Self for method chaining
    pub fn add_shaded_region(
        mut self,
        x_start: f64,
        x_end: f64,
        color: Option<&str>,
        opacity: Option<f64>,
    ) -> Self {
        let color_str = if let Some(c) = color {
            c.to_string()
        } else {
            format!("rgba(200,200,200,{})", opacity.unwrap_or(0.2))
        };

        let shape = Shape::new()
            .shape_type(ShapeType::Rect)
            .x0(x_start)
            .x1(x_end)
            .y0(0)
            .y1(1)
            .y_ref("paper")
            .fill_color(color_str)
            .line(ShapeLine::new().width(0.0));

        let mut layout = self.plot.layout().clone();
        layout.add_shape(shape);
        self.plot.set_layout(layout);
        self
    }

    /// Add pre-computed RMS envelope overlay
    ///
    /// Use with [`crate::operations::plotting::dsp_overlays::compute_windowed_rms`]
    /// to compute the time points and RMS values from audio samples.
    ///
    /// # Arguments
    /// * `time_points` - Time values for the envelope
    /// * `rms_values` - RMS values at each time point
    /// * `color` - Line color (default: "red")
    /// * `line_width` - Line width in pixels (default: 2.0)
    ///
    /// # Returns
    /// Self for method chaining
    pub fn add_rms_envelope(
        mut self,
        time_points: Vec<f64>,
        rms_values: Vec<f64>,
        color: Option<&str>,
        line_width: Option<f32>,
    ) -> Self {
        let color = color.unwrap_or("red").to_string();
        let width = line_width.unwrap_or(2.0);

        let trace = Scatter::new(time_points, rms_values)
            .mode(Mode::Lines)
            .name("RMS Envelope")
            .line(plotly::common::Line::new().color(color).width(width as f64));

        self.plot.add_trace(trace);
        self
    }

    /// Add pre-computed peak envelope overlay
    ///
    /// Use with [`crate::operations::plotting::dsp_overlays::compute_windowed_peak`]
    /// to compute the time points and peak values from audio samples.
    ///
    /// # Arguments
    /// * `time_points` - Time values for the envelope
    /// * `peak_values` - Peak values at each time point
    /// * `color` - Line color (default: "orange")
    /// * `line_width` - Line width in pixels (default: 2.0)
    ///
    /// # Returns
    /// Self for method chaining
    pub fn add_peak_envelope(
        mut self,
        time_points: Vec<f64>,
        peak_values: Vec<f64>,
        color: Option<&str>,
        line_width: Option<f32>,
    ) -> Self {
        let color = color.unwrap_or("orange").to_string();
        let width = line_width.unwrap_or(2.0);

        let trace = Scatter::new(time_points, peak_values)
            .mode(Mode::Lines)
            .name("Peak Envelope")
            .line(plotly::common::Line::new().color(color).width(width as f64));

        self.plot.add_trace(trace);
        self
    }

    /// Add pre-computed zero-crossing rate (ZCR) overlay
    ///
    /// Use with [`crate::operations::plotting::dsp_overlays::compute_windowed_zcr`]
    /// to compute the time points and ZCR values from audio samples.
    ///
    /// ZCR values use a secondary y-axis (right side) since they are on a different
    /// scale than audio amplitudes.
    ///
    /// # Arguments
    /// * `time_points` - Time values for the ZCR
    /// * `zcr_values` - Zero-crossing rate at each time point
    /// * `color` - Line color (default: "blue")
    /// * `line_width` - Line width in pixels (default: 2.0)
    ///
    /// # Returns
    /// Self for method chaining
    pub fn add_zcr_overlay(
        mut self,
        time_points: Vec<f64>,
        zcr_values: Vec<f64>,
        color: Option<&str>,
        line_width: Option<f32>,
    ) -> Self {
        let color = color.unwrap_or("blue").to_string();
        let width = line_width.unwrap_or(2.0);

        let trace = Scatter::new(time_points, zcr_values)
            .mode(Mode::Lines)
            .name("Zero-Crossing Rate")
            .line(plotly::common::Line::new().color(color).width(width as f64))
            .y_axis("y2"); // Use secondary y-axis

        self.plot.add_trace(trace);

        // Add secondary y-axis to layout
        let mut layout = self.plot.layout().clone();
        let yaxis2 = Axis::new()
            .title("Zero-Crossing Rate")
            .overlaying("y")
            .side(AxisSide::Right);
        layout = layout.y_axis2(yaxis2);
        self.plot.set_layout(layout);

        self
    }

    /// Add pre-computed energy overlay
    ///
    /// Use with [`crate::operations::plotting::dsp_overlays::compute_windowed_energy`]
    /// to compute the time points and energy values from audio samples.
    ///
    /// Energy values use a secondary y-axis (right side) since they are on a different
    /// scale than audio amplitudes.
    ///
    /// # Arguments
    /// * `time_points` - Time values for the energy
    /// * `energy_values` - Energy values at each time point
    /// * `color` - Line color (default: "green")
    /// * `line_width` - Line width in pixels (default: 2.0)
    ///
    /// # Returns
    /// Self for method chaining
    pub fn add_energy_overlay(
        mut self,
        time_points: Vec<f64>,
        energy_values: Vec<f64>,
        color: Option<&str>,
        line_width: Option<f32>,
    ) -> Self {
        let color = color.unwrap_or("green").to_string();
        let width = line_width.unwrap_or(2.0);

        let trace = Scatter::new(time_points, energy_values)
            .mode(Mode::Lines)
            .name("Energy")
            .line(plotly::common::Line::new().color(color).width(width as f64))
            .y_axis("y2"); // Use secondary y-axis

        self.plot.add_trace(trace);

        // Add secondary y-axis to layout
        let mut layout = self.plot.layout().clone();
        let yaxis2 = Axis::new()
            .title("Energy")
            .overlaying("y")
            .side(AxisSide::Right);
        layout = layout.y_axis2(yaxis2);
        self.plot.set_layout(layout);

        self
    }

    /// Add onset markers as vertical lines
    ///
    /// Use with onset detection methods like `detect_onsets()` to mark
    /// detected onset times on the waveform.
    ///
    /// # Arguments
    /// * `onset_times` - Vector of onset times in seconds
    /// * `color` - Optional color for onset markers (default: "red")
    /// * `show_labels` - Whether to show time labels on markers (default: false)
    ///
    /// # Returns
    /// Self for method chaining
    ///
    /// # Example
    /// ```rust,ignore
    /// use audio_samples::operations::AudioOnsetDetection;
    /// use audio_samples::operations::onset::OnsetDetectionConfig;
    ///
    /// let onsets = audio.detect_onsets(&OnsetDetectionConfig::percussive())?;
    /// let plot = audio.plot_waveform(&WaveformPlotParams::default())?
    ///     .add_onset_markers(onsets, Some("red"), false);
    /// ```
    pub fn add_onset_markers(
        mut self,
        onset_times: Vec<f64>,
        color: Option<&str>,
        show_labels: bool,
    ) -> Self {
        let color = color.unwrap_or("red");

        for &time in onset_times.iter() {
            let label = if show_labels {
                Some(format!("{:.3}s", time))
            } else {
                None
            };

            self = self.add_vline(time, label.as_deref(), Some(color));
        }

        self
    }

    /// Add beat markers as vertical lines
    ///
    /// Use with beat tracking methods like `detect_beats()` to mark
    /// detected beat times on the waveform.
    ///
    /// # Arguments
    /// * `beat_times` - Vector of beat times in seconds
    /// * `color` - Optional color for beat markers (default: "blue")
    /// * `show_labels` - Whether to show beat numbers on markers (default: false)
    ///
    /// # Returns
    /// Self for method chaining
    ///
    /// # Example
    /// ```rust,ignore
    /// use audio_samples::operations::AudioBeatTracking;
    /// use audio_samples::operations::beat::BeatTrackingConfig;
    ///
    /// let beats = audio.detect_beats(&BeatTrackingConfig::default())?;
    /// let plot = audio.plot_waveform(&WaveformPlotParams::default())?
    ///     .add_beat_markers(beats.beat_times, Some("blue"), true);
    /// ```
    pub fn add_beat_markers(
        mut self,
        beat_times: Vec<f64>,
        color: Option<&str>,
        show_labels: bool,
    ) -> Self {
        let color = color.unwrap_or("blue");

        for (idx, &time) in beat_times.iter().enumerate() {
            let label = if show_labels {
                Some(format!("{}", idx + 1))
            } else {
                None
            };

            self = self.add_vline(time, label.as_deref(), Some(color));
        }

        self
    }
}

#[derive(Debug, Clone)]
pub struct WaveformPlotParams {
    pub plot_params: PlotParams,
    pub ch_mgmt_strategy: Option<ChannelManagementStrategy>,
    pub color: Option<String>,
    pub line_style: Option<String>,
    pub line_width: Option<f32>,
    pub markers: bool,
    pub save_path: Option<PathBuf>,
}

impl Default for WaveformPlotParams {
    fn default() -> Self {
        Self {
            plot_params: PlotParams::default(),
            ch_mgmt_strategy: None,
            color: None,
            line_style: None,
            line_width: None,
            markers: false,
            save_path: None,
        }
    }
}

/// Builder for WaveformPlotParams
pub struct WaveformPlotParamsBuilder {
    params: WaveformPlotParams,
}

impl WaveformPlotParams {
    /// Create a new builder for WaveformPlotParams
    pub fn builder() -> WaveformPlotParamsBuilder {
        WaveformPlotParamsBuilder {
            params: WaveformPlotParams::default(),
        }
    }
}

impl WaveformPlotParamsBuilder {
    /// Set the plot title
    pub fn title(mut self, title: impl Into<String>) -> Self {
        self.params.plot_params.title = Some(title.into());
        self
    }

    /// Set the x-axis label
    pub fn x_label(mut self, label: impl Into<String>) -> Self {
        self.params.plot_params.x_label = Some(label.into());
        self
    }

    /// Set the y-axis label
    pub fn y_label(mut self, label: impl Into<String>) -> Self {
        self.params.plot_params.y_label = Some(label.into());
        self
    }

    /// Set the waveform color
    pub fn color(mut self, color: impl Into<String>) -> Self {
        self.params.color = Some(color.into());
        self
    }

    /// Set the line width
    pub fn line_width(mut self, width: f32) -> Self {
        self.params.line_width = Some(width);
        self
    }

    /// Enable or disable markers
    pub fn markers(mut self, enabled: bool) -> Self {
        self.params.markers = enabled;
        self
    }

    /// Enable grid
    pub fn grid(mut self, enabled: bool) -> Self {
        self.params.plot_params.grid = enabled;
        self
    }

    /// Set channel management strategy
    pub fn channel_strategy(mut self, strategy: ChannelManagementStrategy) -> Self {
        self.params.ch_mgmt_strategy = Some(strategy);
        self
    }

    /// Build the WaveformPlotParams
    pub fn build(self) -> WaveformPlotParams {
        self.params
    }
}

pub fn create_waveform_plot<T>(
    audio: &AudioSamples<'_, T>,
    params: &WaveformPlotParams,
) -> AudioSampleResult<WaveformPlot>
where
    T: StandardSample,
{
    let audio_f64 = audio.as_float();

    let strategy = params.ch_mgmt_strategy.unwrap_or_default();

    let rows = if audio.is_multi_channel() {
        match strategy {
            ChannelManagementStrategy::Average
            | ChannelManagementStrategy::First
            | ChannelManagementStrategy::Last
            | ChannelManagementStrategy::Overlap => 1,
            ChannelManagementStrategy::Separate(layout) => match layout {
                super::Layout::Vertical => audio.num_channels().get(),
                super::Layout::Horizontal => 1,
            },
        }
    } else {
        1
    } as usize;

    let cols = if audio.is_multi_channel() {
        match strategy {
            ChannelManagementStrategy::Average
            | ChannelManagementStrategy::First
            | ChannelManagementStrategy::Last
            | ChannelManagementStrategy::Overlap => 1,
            ChannelManagementStrategy::Separate(layout) => match layout {
                super::Layout::Vertical => 1,
                super::Layout::Horizontal => audio.num_channels().get(),
            },
        }
    } else {
        1
    } as usize;

    // Configure time axis with proper formatting
    let x_axis = configure_time_axis(
        plotly::layout::Axis::new(),
        params
            .plot_params
            .x_label
            .clone()
            .or_else(|| Some("Time".to_string())),
    );

    // Configure amplitude axis
    let y_axis = plotly::layout::Axis::new().title(
        params
            .plot_params
            .y_label
            .clone()
            .unwrap_or_else(|| "Amplitude".to_string()),
    );

    let plotly_layout = plotly::Layout::new()
        .title(params.plot_params.title.clone().unwrap_or_default())
        .x_axis(x_axis)
        .y_axis(y_axis)
        .show_legend(params.plot_params.show_legend)
        .legend(
            plotly::layout::Legend::new()
                .title(params.plot_params.legend_title.clone().unwrap_or_default()),
        )
        .grid(
            LayoutGrid::new()
                .rows(rows)
                .columns(cols)
                .pattern(GridPattern::Independent)
                .row_order(RowOrder::TopToBottom),
        );

    let mut plot = Plot::new();

    if audio.is_mono() {
        plot_mono_waveform(&audio_f64, params, &mut plot, "Mono", None);
    } else {
        match strategy {
            ChannelManagementStrategy::Average => {
                let avg_audio = audio_f64.to_mono(MonoConversionMethod::Average)?;
                plot_mono_waveform(&avg_audio, params, &mut plot, "Average", None);
            }
            ChannelManagementStrategy::Separate(layout) => {
                for (idx, channel) in audio_f64.channels().enumerate() {
                    let name = format!("Ch{}", idx + 1);
                    let (row, col) = match layout {
                        super::Layout::Vertical => (idx, 0),
                        super::Layout::Horizontal => (0, idx),
                    };
                    let axis_ref = axis_reference(row, col, cols);
                    plot_mono_waveform(&channel, params, &mut plot, &name, Some(axis_ref));
                }
            }
            ChannelManagementStrategy::First => {
                let channel_one = audio_f64.extract_channel(0)?;
                plot_mono_waveform(&channel_one, params, &mut plot, "Ch1", None);
            }
            ChannelManagementStrategy::Last => {
                let last_channel = audio_f64.extract_channel(audio_f64.num_channels().get() - 1)?;
                plot_mono_waveform(&last_channel, params, &mut plot, "Last", None);
            }
            ChannelManagementStrategy::Overlap => {
                for (idx, channel) in audio_f64.channels().enumerate() {
                    let name = format!("Ch{}", idx + 1);
                    plot_mono_waveform(&channel, params, &mut plot, &name, None);
                }
            }
        }
    }

    let mut plotly_layout = plotly_layout;

    if audio.is_multi_channel() {
        if let ChannelManagementStrategy::Separate(layout) = strategy {
            plotly_layout =
                configure_separate_axes(plotly_layout, rows, cols, layout, &params.plot_params);
        }
    }

    plot.set_layout(plotly_layout);

    Ok(WaveformPlot {
        _params: params.clone(),
        plot,
    })
}

fn axis_reference(row: usize, col: usize, cols: usize) -> (String, String) {
    let adjusted_index = row * cols + col;
    (axis_id('x', adjusted_index), axis_id('y', adjusted_index))
}

fn axis_id(prefix: char, index: usize) -> String {
    if index == 0 {
        prefix.to_string()
    } else {
        format!("{}{}", prefix, index + 1)
    }
}

fn configure_separate_axes(
    layout: plotly::Layout,
    rows: usize,
    cols: usize,
    layout_kind: super::Layout,
    plot_params: &PlotParams,
) -> plotly::Layout {
    match layout_kind {
        super::Layout::Vertical => configure_vertical_axes(layout, rows, cols, plot_params),
        super::Layout::Horizontal => configure_horizontal_axes(layout, rows, cols, plot_params),
    }
}

fn configure_vertical_axes(
    mut layout: plotly::Layout,
    rows: usize,
    cols: usize,
    plot_params: &PlotParams,
) -> plotly::Layout {
    if rows == 0 {
        return layout;
    }

    let base_axis_index = (rows - 1) * cols;
    let base_axis_name = axis_id('x', base_axis_index);

    for row in 0..rows {
        for col in 0..cols {
            let axis_index = row * cols + col;
            let axis_x_name = axis_id('x', axis_index);
            let axis_y_name = axis_id('y', axis_index);
            let is_bottom_row = row == rows - 1;

            let mut x_axis = Axis::new().anchor(&axis_y_name).side(AxisSide::Bottom);

            if !is_bottom_row {
                x_axis = x_axis
                    .matches(&base_axis_name)
                    .show_tick_labels(false)
                    .tick_length(0);
            } else if col == 0 {
                if let Some(label) = plot_params.x_label.as_ref() {
                    x_axis = x_axis.title(label.clone());
                }
            }

            layout = assign_x_axis(layout, axis_index, x_axis);

            let mut y_axis = Axis::new().anchor(&axis_x_name);
            if col == 0 && row == 0 {
                if let Some(label) = plot_params.y_label.as_ref() {
                    y_axis = y_axis.title(label.clone());
                }
            }

            layout = assign_y_axis(layout, axis_index, y_axis);
        }
    }

    layout
}

fn configure_horizontal_axes(
    mut layout: plotly::Layout,
    rows: usize,
    cols: usize,
    plot_params: &PlotParams,
) -> plotly::Layout {
    if cols == 0 {
        return layout;
    }

    let base_axis_name = axis_id('y', 0);

    for row in 0..rows {
        for col in 0..cols {
            let axis_index = row * cols + col;
            let axis_x_name = axis_id('x', axis_index);
            let axis_y_name = axis_id('y', axis_index);

            let mut x_axis = Axis::new().anchor(&axis_y_name).side(AxisSide::Bottom);
            if row == 0 {
                if let Some(label) = plot_params.x_label.as_ref() {
                    x_axis = x_axis.title(label.clone());
                }
            }

            layout = assign_x_axis(layout, axis_index, x_axis);

            let mut y_axis = Axis::new().anchor(&axis_x_name);
            if col != 0 {
                y_axis = y_axis
                    .matches(&base_axis_name)
                    .show_tick_labels(false)
                    .tick_length(0);
            } else if row == 0 {
                if let Some(label) = plot_params.y_label.as_ref() {
                    y_axis = y_axis.title(label.clone());
                }
            }

            layout = assign_y_axis(layout, axis_index, y_axis);
        }
    }

    layout
}

fn assign_x_axis(layout: plotly::Layout, index: usize, axis: Axis) -> plotly::Layout {
    match index {
        0 => layout.x_axis(axis),
        1 => layout.x_axis2(axis),
        2 => layout.x_axis3(axis),
        3 => layout.x_axis4(axis),
        4 => layout.x_axis5(axis),
        5 => layout.x_axis6(axis),
        6 => layout.x_axis7(axis),
        7 => layout.x_axis8(axis),
        _ => panic!("Waveform plot supports up to eight subplot x-axes"),
    }
}

fn assign_y_axis(layout: plotly::Layout, index: usize, axis: Axis) -> plotly::Layout {
    match index {
        0 => layout.y_axis(axis),
        1 => layout.y_axis2(axis),
        2 => layout.y_axis3(axis),
        3 => layout.y_axis4(axis),
        4 => layout.y_axis5(axis),
        5 => layout.y_axis6(axis),
        6 => layout.y_axis7(axis),
        7 => layout.y_axis8(axis),
        _ => panic!("Waveform plot supports up to eight subplot y-axes"),
    }
}

fn plot_mono_waveform(
    audio: &AudioSamples<'_, f64>,
    params: &WaveformPlotParams,
    plot: &mut Plot,
    name: &str,
    axis_ref: Option<(String, String)>,
) {
    let plot_mode = match params.markers {
        true => plotly::common::Mode::Markers,
        false => plotly::common::Mode::Lines,
    };
    let time_data = (0..audio.len().get())
        .map(|i| i as f64 / audio.sample_rate_hz())
        .collect::<Vec<_>>();
    let amplitude_data = audio.as_slice().expect("Mono audio is contiguous").to_vec();
    let (time_data, amplitude_data) = if audio.len() > DECIMATE_THRESHOLD {
        decimate_waveform(&time_data, &amplitude_data, DECIMATE_THRESHOLD.get())
    } else {
        (time_data, amplitude_data)
    };
    let mut trace = Scatter::new(time_data, amplitude_data)
        .mode(plot_mode)
        .name(name);
    if let Some((x_axis, y_axis)) = axis_ref {
        trace = trace.x_axis(&x_axis).y_axis(&y_axis);
    }
    plot.add_trace(trace);
}

mod tests {
    #[allow(unused_imports)]
    use super::*;
    #[test]
    fn test_create_waveform_plot() {
        use super::create_waveform_plot;
        let duration = std::time::Duration::from_secs_f64(1.0);
        let audio = crate::stereo_sine_wave::<f64>(440.0, duration, crate::sample_rate!(8000), 0.5);
        let params = super::WaveformPlotParams {
            plot_params: super::PlotParams {
                title: Some("Test Sine Wave".to_string()),
                x_label: Some("Time (s)".to_string()),
                y_label: Some("Amplitude".to_string()),
                show_legend: false,
                legend_title: None,
                font_sizes: None,
                super_title: None,
                grid: true,
            },
            ch_mgmt_strategy: Some(super::ChannelManagementStrategy::Separate(
                crate::operations::plotting::Layout::Vertical,
            )),
            color: None,
            line_style: None,
            line_width: None,
            markers: false,
            save_path: None,
        };
        let plot = create_waveform_plot(&audio, &params).unwrap();

        // Test html() method
        let html = plot.html().unwrap();
        assert!(html.contains("plotly"));

        // Test save() method
        plot.save("test_waveform_plot.html").unwrap();
    }
}
