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

/// Interactive waveform plot with time-domain amplitude traces.
///
/// # Purpose
///
/// Encapsulates a rendered Plotly waveform visualization. Provides methods for
/// saving to disk, generating HTML, and adding overlays (RMS/peak envelopes,
/// onset markers, beat markers, shaded regions).
///
/// # Intended Usage
///
/// Created by [`AudioPlotting::plot_waveform`] or [`create_waveform_plot`].
/// Call [`PlotUtils::save`] or [`PlotUtils::html`] to output the plot, or chain
/// `add_*` methods to annotate the waveform before saving.
///
/// # Invariants
///
/// The internal `Plot` is always valid and can be rendered to HTML. Overlay methods
/// modify the plot in place and return `self` for chaining.
///
/// [`AudioPlotting::plot_waveform`]: crate::operations::AudioPlotting::plot_waveform
/// [`PlotUtils::save`]: super::PlotUtils::save
/// [`PlotUtils::html`]: super::PlotUtils::html
pub struct WaveformPlot {
    _params: WaveformPlotParams, // What parameters created me?
    plot: Plot,                  // the plotly plot
}

impl PlotUtils for WaveformPlot {
    #[inline]
    fn html(&self) -> AudioSampleResult<String> {
        Ok(self.plot.to_html())
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

        // TODO! Make above configurable
        match extension.to_lowercase().as_str() {
            "html" => {
                let html = self.html()?;
                std::fs::write(path, html).map_err(|e| {
                    crate::AudioSampleError::unsupported(format!("Failed to write HTML file: {e}"))
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
                        "Unsupported file extension: {extension}. Supported: html, png, svg, jpeg, jpg, webp"
                    ),
                },
            )),
        }
    }
}

impl PlotComponent for WaveformPlot {
    #[inline]
    fn get_plot(&self) -> &Plot {
        &self.plot
    }

    #[inline]
    fn get_plot_mut(&mut self) -> &mut Plot {
        &mut self.plot
    }

    #[inline]
    fn requires_shared_x_axis(&self) -> bool {
        true // Waveforms are time-based
    }
}

impl WaveformPlot {
    /// Adds a vertical line overlay at the specified time position.
    ///
    /// Renders a vertical line spanning the full amplitude range at the given time. Useful for
    /// marking temporal events, segment boundaries, or alignment points.
    ///
    /// # Arguments
    /// * `x` — Time position in seconds where the vertical line should be drawn
    /// * `label` — Optional text label displayed above the line
    /// * `color` — Optional CSS color string (e.g., "red", "#FF0000"). Defaults to "black".
    ///
    /// # Returns
    /// Self for method chaining.
    ///
    /// # Example
    /// ```rust,no_run
    /// # use audio_samples::{AudioSamples, sample_rate};
    /// # use audio_samples::operations::traits::AudioPlotting;
    /// # use audio_samples::operations::plotting::WaveformPlotParams;
    /// # let audio = AudioSamples::new_mono(ndarray::Array1::from_elem(1000, 0.0f32), sample_rate!(44100)).unwrap();
    /// let plot = audio.plot_waveform(&WaveformPlotParams::default())?
    ///     .add_vline(1.5, Some("Event"), Some("red"));
    /// # Ok::<(), audio_samples::AudioSampleError>(())
    /// ```
    #[inline]
    #[must_use]
    pub fn add_vline(mut self, x: f64, label: Option<&str>, color: Option<&str>) -> Self {
        let color = color.unwrap_or("black").to_string();

        let shape = Shape::new()
            .shape_type(ShapeType::Line)
            .x0(x)
            .x1(x)
            .y0(0)
            .y1(1)
            .y_ref("paper")
            .line(ShapeLine::new().color(color).width(2.0));

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

    /// Adds a horizontal line overlay at the specified amplitude value.
    ///
    /// Renders a horizontal line spanning the full time range at the given amplitude. Useful for
    /// marking reference levels (e.g., zero line, clipping thresholds, RMS levels).
    ///
    /// # Arguments
    /// * `y` — Amplitude value where the horizontal line should be drawn
    /// * `label` — Optional text label displayed to the right of the line
    /// * `color` — Optional CSS color string (e.g., "blue", "#0000FF"). Defaults to "black".
    ///
    /// # Returns
    /// Self for method chaining.
    ///
    /// # Example
    /// ```rust,no_run
    /// # use audio_samples::{AudioSamples, sample_rate};
    /// # use audio_samples::operations::traits::AudioPlotting;
    /// # use audio_samples::operations::plotting::WaveformPlotParams;
    /// # let audio = AudioSamples::new_mono(ndarray::Array1::from_elem(1000, 0.0f32), sample_rate!(44100)).unwrap();
    /// let plot = audio.plot_waveform(&WaveformPlotParams::default())?
    ///     .add_hline(0.5, Some("Threshold"), Some("red"));
    /// # Ok::<(), audio_samples::AudioSampleError>(())
    /// ```
    #[inline]
    #[must_use]
    pub fn add_hline(mut self, y: f64, label: Option<&str>, color: Option<&str>) -> Self {
        let color = color.unwrap_or("black").to_string();

        let shape = Shape::new()
            .shape_type(ShapeType::Line)
            .x0(0)
            .x1(1)
            .x_ref("paper")
            .y0(y)
            .y1(y)
            .line(ShapeLine::new().color(color).width(2.0));

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

    /// Adds a marker point at the specified time-amplitude position.
    ///
    /// Renders a circular marker at the given coordinates. Useful for highlighting specific
    /// sample points, peaks, or feature locations in the waveform.
    ///
    /// # Arguments
    /// * `x` — Time position in seconds
    /// * `y` — Amplitude value
    /// * `text` — Optional text label displayed near the marker
    /// * `_symbol` — Reserved for future use (marker symbol selection). Currently ignored.
    ///
    /// # Returns
    /// Self for method chaining.
    ///
    /// # Example
    /// ```rust,no_run
    /// # use audio_samples::{AudioSamples, sample_rate};
    /// # use audio_samples::operations::traits::AudioPlotting;
    /// # use audio_samples::operations::plotting::WaveformPlotParams;
    /// # let audio = AudioSamples::new_mono(ndarray::Array1::from_elem(1000, 0.0f32), sample_rate!(44100)).unwrap();
    /// let plot = audio.plot_waveform(&WaveformPlotParams::default())?
    ///     .add_marker(1.0, 0.8, Some("Peak"), None);
    /// # Ok::<(), audio_samples::AudioSampleError>(())
    /// ```
    #[inline]
    #[must_use]
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

    /// Adds a shaded vertical region between two time points.
    ///
    /// Renders a semi-transparent rectangular region spanning the full amplitude range between
    /// the specified start and end times. Useful for highlighting temporal segments, analysis
    /// windows, or regions of interest.
    ///
    /// # Arguments
    /// * `x_start` — Start time in seconds
    /// * `x_end` — End time in seconds
    /// * `color` — Optional RGBA color string (e.g., "rgba(255,0,0,0.3)"). If `None`, uses gray
    ///   with the specified opacity.
    /// * `opacity` — Optional opacity override (0.0-1.0). Only used if `color` is `None`.
    ///   Defaults to 0.2.
    ///
    /// # Returns
    /// Self for method chaining.
    ///
    /// # Example
    /// ```rust,no_run
    /// # use audio_samples::{AudioSamples, sample_rate};
    /// # use audio_samples::operations::traits::AudioPlotting;
    /// # use audio_samples::operations::plotting::WaveformPlotParams;
    /// # let audio = AudioSamples::new_mono(ndarray::Array1::from_elem(1000, 0.0f32), sample_rate!(44100)).unwrap();
    /// let plot = audio.plot_waveform(&WaveformPlotParams::default())?
    ///     .add_shaded_region(1.0, 2.0, Some("rgba(255,0,0,0.3)"), None);
    /// # Ok::<(), audio_samples::AudioSampleError>(())
    /// ```
    #[inline]
    #[must_use]
    pub fn add_shaded_region(
        mut self,
        x_start: f64,
        x_end: f64,
        color: Option<&str>,
        opacity: Option<f64>,
    ) -> Self {
        let color_str = color.map_or_else(
            || format!("rgba(200,200,200,{})", opacity.unwrap_or(0.2)),
            std::string::ToString::to_string,
        );

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

    /// Adds a pre-computed RMS envelope overlay to the waveform.
    ///
    /// Renders a line trace showing the RMS (Root Mean Square) amplitude envelope over time.
    /// The RMS envelope provides a smoothed representation of signal power and correlates with
    /// perceived loudness.
    ///
    /// Typically used in conjunction with
    /// [`crate::operations::plotting::dsp_overlays::compute_windowed_rms`] to compute windowed
    /// RMS values over the signal.
    ///
    /// # Arguments
    /// * `time_points` — Time values in seconds for each envelope point
    /// * `rms_values` — RMS amplitude values corresponding to each time point
    /// * `color` — Optional line color. Defaults to "red".
    /// * `line_width` — Optional line width in pixels. Defaults to 2.0.
    ///
    /// # Returns
    /// Self for method chaining.
    ///
    /// # Example
    /// ```rust,no_run
    /// use audio_samples::{AudioSamples, sample_rate, nzu};
    /// use audio_samples::operations::traits::AudioPlotting;
    /// use audio_samples::operations::plotting::{WaveformPlotParams, dsp_overlays};
    ///
    /// let audio = AudioSamples::new_mono(ndarray::Array1::from_elem(44100, 0.0f32), sample_rate!(44100))?;
    /// let (times, rms) = dsp_overlays::compute_windowed_rms(&audio, nzu!(2048), nzu!(512));
    /// let plot = audio.plot_waveform(&WaveformPlotParams::default())?
    ///     .add_rms_envelope(times, rms, Some("red"), Some(2.0));
    /// # Ok::<(), audio_samples::AudioSampleError>(())
    /// ```
    #[inline]
    #[must_use]
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
            .line(
                plotly::common::Line::new()
                    .color(color)
                    .width(f64::from(width)),
            );

        self.plot.add_trace(trace);
        self
    }

    /// Adds a pre-computed peak envelope overlay to the waveform.
    ///
    /// Renders a line trace showing the peak (maximum absolute amplitude) envelope over time.
    /// The peak envelope highlights the loudest instantaneous amplitudes in each window and is
    /// useful for analyzing dynamic range and detecting clipping risk.
    ///
    /// Typically used in conjunction with
    /// [`crate::operations::plotting::dsp_overlays::compute_windowed_peak`] to compute windowed
    /// peak values over the signal.
    ///
    /// # Arguments
    /// * `time_points` — Time values in seconds for each envelope point
    /// * `peak_values` — Peak amplitude values corresponding to each time point
    /// * `color` — Optional line color. Defaults to "orange".
    /// * `line_width` — Optional line width in pixels. Defaults to 2.0.
    ///
    /// # Returns
    /// Self for method chaining.
    ///
    /// # Example
    /// ```rust,no_run
    /// use audio_samples::{AudioSamples, sample_rate, nzu};
    /// use audio_samples::operations::traits::AudioPlotting;
    /// use audio_samples::operations::plotting::{WaveformPlotParams, dsp_overlays};
    ///
    /// let audio = AudioSamples::new_mono(ndarray::Array1::from_elem(44100, 0.0f32), sample_rate!(44100))?;
    /// let (times, peaks) = dsp_overlays::compute_windowed_peak(&audio, nzu!(2048), nzu!(512));
    /// let plot = audio.plot_waveform(&WaveformPlotParams::default())?
    ///     .add_peak_envelope(times, peaks, Some("orange"), Some(2.0));
    /// # Ok::<(), audio_samples::AudioSampleError>(())
    /// ```
    #[inline]
    #[must_use]
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
            .line(
                plotly::common::Line::new()
                    .color(color)
                    .width(f64::from(width)),
            );

        self.plot.add_trace(trace);
        self
    }

    /// Adds a pre-computed zero-crossing rate (ZCR) overlay to the waveform.
    ///
    /// Renders a line trace showing the zero-crossing rate over time. ZCR measures how often
    /// the signal crosses the zero amplitude line and correlates with spectral content: higher
    /// ZCR indicates noisier, more broadband signals, while lower ZCR indicates tonal content.
    ///
    /// ZCR values are plotted on a secondary y-axis (right side) since they are on a different
    /// scale than audio amplitudes.
    ///
    /// Typically used in conjunction with
    /// [`crate::operations::plotting::dsp_overlays::compute_windowed_zcr`] to compute windowed
    /// ZCR values over the signal.
    ///
    /// # Arguments
    /// * `time_points` — Time values in seconds for each ZCR point
    /// * `zcr_values` — Zero-crossing rate values (counts per sample) corresponding to each
    ///   time point
    /// * `color` — Optional line color. Defaults to "blue".
    /// * `line_width` — Optional line width in pixels. Defaults to 2.0.
    ///
    /// # Returns
    /// Self for method chaining.
    ///
    /// # Example
    /// ```rust,no_run
    /// use audio_samples::{AudioSamples, sample_rate, nzu};
    /// use audio_samples::operations::traits::AudioPlotting;
    /// use audio_samples::operations::plotting::{WaveformPlotParams, dsp_overlays};
    ///
    /// let audio = AudioSamples::new_mono(ndarray::Array1::from_elem(44100, 0.0f32), sample_rate!(44100))?;
    /// let (times, zcr) = dsp_overlays::compute_windowed_zcr(&audio, nzu!(2048), nzu!(512));
    /// let plot = audio.plot_waveform(&WaveformPlotParams::default())?
    ///     .add_zcr_overlay(times, zcr, Some("blue"), Some(2.0));
    /// # Ok::<(), audio_samples::AudioSampleError>(())
    /// ```
    #[inline]
    #[must_use]
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
            .line(
                plotly::common::Line::new()
                    .color(color)
                    .width(f64::from(width)),
            )
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

    /// Adds a pre-computed energy overlay to the waveform.
    ///
    /// Renders a line trace showing the signal energy (RMS squared, proportional to power) over
    /// time. Energy is a measure of signal intensity and is useful for detecting voiced vs.
    /// unvoiced segments, onsets, and dynamic changes.
    ///
    /// Energy values are plotted on a secondary y-axis (right side) since they are on a different
    /// scale than audio amplitudes.
    ///
    /// Typically used in conjunction with
    /// [`crate::operations::plotting::dsp_overlays::compute_windowed_energy`] to compute windowed
    /// energy values over the signal.
    ///
    /// # Arguments
    /// * `time_points` — Time values in seconds for each energy point
    /// * `energy_values` — Energy values (squared RMS) corresponding to each time point
    /// * `color` — Optional line color. Defaults to "green".
    /// * `line_width` — Optional line width in pixels. Defaults to 2.0.
    ///
    /// # Returns
    /// Self for method chaining.
    ///
    /// # Example
    /// ```rust,no_run
    /// use audio_samples::{AudioSamples, sample_rate, nzu};
    /// use audio_samples::operations::traits::AudioPlotting;
    /// use audio_samples::operations::plotting::{WaveformPlotParams, dsp_overlays};
    ///
    /// let audio = AudioSamples::new_mono(ndarray::Array1::from_elem(44100, 0.0f32), sample_rate!(44100))?;
    /// let (times, energy) = dsp_overlays::compute_windowed_energy(&audio, nzu!(2048), nzu!(512));
    /// let plot = audio.plot_waveform(&WaveformPlotParams::default())?
    ///     .add_energy_overlay(times, energy, Some("green"), Some(2.0));
    /// # Ok::<(), audio_samples::AudioSampleError>(())
    /// ```
    #[inline]
    #[must_use]
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
            .line(
                plotly::common::Line::new()
                    .color(color)
                    .width(f64::from(width)),
            )
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

    /// Adds onset markers as vertical lines at the specified times.
    ///
    /// Renders vertical lines at each onset time, typically used to visualize the output of
    /// onset detection algorithms. Onsets mark the beginning of transient events such as note
    /// attacks, percussive hits, or significant spectral changes.
    ///
    /// Internally calls [`add_vline`] for each onset time.
    ///
    /// # Arguments
    /// * `onset_times` — Vector of onset times in seconds
    /// * `color` — Optional CSS color for all onset markers. Defaults to "red".
    /// * `show_labels` — If `true`, displays the time value (in seconds) above each marker.
    ///
    /// # Returns
    /// Self for method chaining.
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
    #[inline]
    #[must_use]
    pub fn add_onset_markers(
        mut self,
        onset_times: Vec<f64>,
        color: Option<&str>,
        show_labels: bool,
    ) -> Self {
        let color = color.unwrap_or("red");

        for &time in &onset_times {
            let label = if show_labels {
                Some(format!("{time:.3}s"))
            } else {
                None
            };

            self = self.add_vline(time, label.as_deref(), Some(color));
        }

        self
    }

    /// Adds beat markers as vertical lines at the specified times.
    ///
    /// Renders vertical lines at each beat time, typically used to visualize the output of
    /// beat tracking algorithms. Beats mark the perceptual pulse or rhythm of the signal.
    ///
    /// Internally calls [`add_vline`] for each beat time.
    ///
    /// # Arguments
    /// * `beat_times` — Vector of beat times in seconds
    /// * `color` — Optional CSS color for all beat markers. Defaults to "blue".
    /// * `show_labels` — If `true`, displays the beat number (1, 2, 3, ...) above each marker.
    ///
    /// # Returns
    /// Self for method chaining.
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
    #[inline]
    #[must_use]
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

/// Configuration parameters for waveform plots.
///
/// # Purpose
///
/// Aggregates all visual and layout settings for a waveform plot: common plot
/// parameters (title, labels, grid), channel management strategy, line styling,
/// and optional output path.
///
/// # Intended Usage
///
/// Construct via [`Default::default()`] and field assignment, or use the builder
/// pattern via [`WaveformPlotParams::builder()`]. Pass to
/// [`AudioPlotting::plot_waveform`] to generate a plot.
///
/// # Invariants
///
/// All `Option` fields default to `None`; the plot renderer applies sensible
/// defaults when unset (e.g. automatic color assignment, line width = 1.0).
///
/// [`AudioPlotting::plot_waveform`]: crate::operations::AudioPlotting::plot_waveform
#[derive(Debug, Clone, Default)]
#[non_exhaustive]
pub struct WaveformPlotParams {
    /// Common plot parameters (title, labels, font sizes, etc.).
    pub plot_params: PlotParams,
    /// How to handle multi-channel audio. If `None`, defaults to `Separate(Vertical)`.
    pub ch_mgmt_strategy: Option<ChannelManagementStrategy>,
    /// Line color (CSS color string, e.g. `"blue"` or `"#FF5733"`).
    pub color: Option<String>,
    /// Line style (currently unused, reserved for future use).
    pub line_style: Option<String>,
    /// Line width in pixels.
    pub line_width: Option<f32>,
    /// Whether to draw markers at each sample point.
    pub markers: bool,
    /// Optional file path to auto-save the plot after creation.
    pub save_path: Option<PathBuf>,
}

impl WaveformPlotParams {
    /// Creates a new `WaveformPlotParams` instance with default values.
    ///
    /// All fields are initialized to `None` or default settings. Use the
    /// builder pattern via [`builder()`] to customize parameters fluently.
    ///
    /// # Returns
    ///
    /// A `WaveformPlotParams` instance with default configuration.
    #[inline]
    #[must_use]
    pub fn new(
        plot_params: &PlotParams,
        ch_mgmt_strategy: Option<ChannelManagementStrategy>,
        color: Option<String>,
        line_style: Option<String>,
        line_width: Option<f32>,
        markers: bool,
        save_path: Option<PathBuf>,
    ) -> Self {
        Self {
            plot_params: plot_params.clone(),
            ch_mgmt_strategy,
            color,
            line_style,
            line_width,
            markers,
            save_path,
        }
    }
}

/// Builder for [`WaveformPlotParams`].
///
/// # Purpose
///
/// Provides a fluent interface for constructing [`WaveformPlotParams`] with
/// only the desired fields set.
///
/// # Intended Usage
///
/// Start with [`WaveformPlotParams::builder()`], chain setter methods, and
/// finish with [`build()`][WaveformPlotParamsBuilder::build].
///
/// ```rust
/// use audio_samples::operations::plotting::waveform::WaveformPlotParams;
/// use audio_samples::operations::plotting::ChannelManagementStrategy;
///
/// let params = WaveformPlotParams::builder()
///     .title("My Waveform")
///     .color("blue")
///     .line_width(2.0)
///     .grid(true)
///     .build();
/// ```
pub struct WaveformPlotParamsBuilder {
    params: WaveformPlotParams,
}

impl WaveformPlotParams {
    /// Creates a new builder for constructing [`WaveformPlotParams`].
    ///
    /// # Returns
    ///
    /// A [`WaveformPlotParamsBuilder`] initialized with default values.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use audio_samples::operations::plotting::waveform::WaveformPlotParams;
    ///
    /// let params = WaveformPlotParams::builder()
    ///     .title("My Waveform")
    ///     .grid(true)
    ///     .build();
    /// ```
    #[inline]
    #[must_use]
    pub fn builder() -> WaveformPlotParamsBuilder {
        WaveformPlotParamsBuilder {
            params: Self::default(),
        }
    }
}

impl WaveformPlotParamsBuilder {
    /// Sets the plot title.
    ///
    /// # Arguments
    ///
    /// - `title` – Main title text.
    ///
    /// # Returns
    ///
    /// `Self` for method chaining.
    #[inline]
    #[must_use]
    pub fn title(mut self, title: impl Into<String>) -> Self {
        self.params.plot_params.title = Some(title.into());
        self
    }

    /// Sets the x-axis label.
    ///
    /// # Arguments
    ///
    /// - `label` – X-axis label text (e.g. `"Time"`).
    ///
    /// # Returns
    ///
    /// `Self` for method chaining.
    #[inline]
    #[must_use]
    pub fn x_label(mut self, label: impl Into<String>) -> Self {
        self.params.plot_params.x_label = Some(label.into());
        self
    }

    /// Sets the y-axis label.
    ///
    /// # Arguments
    ///
    /// - `label` – Y-axis label text (e.g. `"Amplitude"`).
    ///
    /// # Returns
    ///
    /// `Self` for method chaining.
    #[inline]
    #[must_use]
    pub fn y_label(mut self, label: impl Into<String>) -> Self {
        self.params.plot_params.y_label = Some(label.into());
        self
    }

    /// Sets the waveform line color.
    ///
    /// # Arguments
    ///
    /// - `color` – CSS color string (e.g. `"blue"`, `"#FF5733"`, `"rgb(255,0,0)"`).
    ///
    /// # Returns
    ///
    /// `Self` for method chaining.
    #[inline]
    #[must_use]
    pub fn color(mut self, color: impl Into<String>) -> Self {
        self.params.color = Some(color.into());
        self
    }

    /// Sets the waveform line width.
    ///
    /// # Arguments
    ///
    /// - `width` – Line width in pixels.
    ///
    /// # Returns
    ///
    /// `Self` for method chaining.
    #[inline]
    #[must_use]
    pub const fn line_width(mut self, width: f32) -> Self {
        self.params.line_width = Some(width);
        self
    }

    /// Enables or disables sample-point markers.
    ///
    /// # Arguments
    ///
    /// - `enabled` – Whether to draw markers at each sample.
    ///
    /// # Returns
    ///
    /// `Self` for method chaining.
    #[inline]
    #[must_use]
    pub const fn markers(mut self, enabled: bool) -> Self {
        self.params.markers = enabled;
        self
    }

    /// Enables or disables grid lines.
    ///
    /// # Arguments
    ///
    /// - `enabled` – Whether to show grid lines.
    ///
    /// # Returns
    ///
    /// `Self` for method chaining.
    #[inline]
    #[must_use]
    pub const fn grid(mut self, enabled: bool) -> Self {
        self.params.plot_params.grid = enabled;
        self
    }

    /// Sets the channel management strategy.
    ///
    /// # Arguments
    ///
    /// - `strategy` – How to handle multi-channel audio (average, separate, overlap, etc.).
    ///
    /// # Returns
    ///
    /// `Self` for method chaining.
    #[inline]
    #[must_use]
    pub const fn channel_strategy(mut self, strategy: ChannelManagementStrategy) -> Self {
        self.params.ch_mgmt_strategy = Some(strategy);
        self
    }

    /// Consumes the builder and returns the configured [`WaveformPlotParams`].
    ///
    /// # Returns
    ///
    /// The constructed `WaveformPlotParams`.
    #[inline]
    #[must_use]
    pub fn build(self) -> WaveformPlotParams {
        self.params
    }
}

/// Creates a waveform plot from audio samples and parameters.
///
/// Converts the audio to `f64`, applies decimation if the sample count exceeds
/// [`DECIMATE_THRESHOLD`], and renders the waveform using the configured channel
/// management strategy.
///
/// # Arguments
///
/// - `audio` – The audio samples to plot.
/// - `params` – Visual and layout parameters.
///
/// # Returns
///
/// A [`WaveformPlot`] that can be saved, rendered to HTML, or annotated with overlays.
///
/// # Errors
///
/// Returns [`AudioSampleError`] if channel extraction fails (e.g. invalid channel
/// index in multi-channel audio when using `First`/`Last` strategies).
///
/// # Examples
///
/// ```rust
/// use audio_samples::{AudioSamples, sample_rate, sine_wave};
/// use audio_samples::operations::plotting::waveform::{WaveformPlotParams, create_waveform_plot};
/// use std::time::Duration;
///
/// let audio = sine_wave::<f64>(440.0, Duration::from_millis(100), sample_rate!(44100), 0.8);
/// let params = WaveformPlotParams::default();
/// let plot = create_waveform_plot(&audio, &params).unwrap();
/// // plot.save("waveform.html").unwrap();
/// ```
///
/// [`DECIMATE_THRESHOLD`]: super::DECIMATE_THRESHOLD
/// [`AudioSampleError`]: crate::AudioSampleError
#[inline]
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
            let x_axis_id = axis_id('x', axis_index);
            let y_axis_id = axis_id('y', axis_index);
            let is_bottom_row = row == rows - 1;

            let mut x_axis = Axis::new().anchor(&y_axis_id).side(AxisSide::Bottom);

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

            let mut y_axis = Axis::new().anchor(&x_axis_id);
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
            let x_axis_id = axis_id('x', axis_index);
            let y_axis_id = axis_id('y', axis_index);

            let mut x_axis = Axis::new().anchor(&y_axis_id).side(AxisSide::Bottom);
            if row == 0 {
                if let Some(label) = plot_params.x_label.as_ref() {
                    x_axis = x_axis.title(label.clone());
                }
            }

            layout = assign_x_axis(layout, axis_index, x_axis);

            let mut y_axis = Axis::new().anchor(&x_axis_id);
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
    let plot_mode = if params.markers {
        plotly::common::Mode::Markers
    } else {
        plotly::common::Mode::Lines
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
