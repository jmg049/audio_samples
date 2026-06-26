//! Lissajous (stereo X-Y) visualization.
//!
//! Scatters channel 0 (x) against channel 1 (y) of a 2-channel signal. This is
//! the standard goniometer / vectorscope view used to inspect stereo width and
//! inter-channel phase correlation: a 45° diagonal indicates a mono-correlated
//! signal, a horizontal line indicates anti-phase, and a circular spread
//! indicates a wide, decorrelated stereo image.

use plotly::common::Mode;
use plotly::layout::Axis;
use plotly::{Layout, Plot, Scatter};
use std::path::Path;

use super::composite::PlotComponent;
use super::{PlotParams, PlotUtils};
use crate::{AudioSampleError, AudioSampleResult, AudioSamples, ParameterError, StandardSample};

/// Configuration parameters for a Lissajous (stereo X-Y) plot.
///
/// # Purpose
/// Encapsulates the settings for the stereo correlation scatter: shared plot
/// styling and an optional cap on the number of plotted points (to keep large
/// signals responsive in the browser).
///
/// # Intended Usage
/// Construct via [`LissajousParams::new`] (or [`Default::default`]) and refine
/// with the `with_*` builder setters before passing to
/// [`create_lissajous_plot`] or the [`crate::operations::AudioPlotting`] trait.
///
/// # Invariants
/// - If `max_points` is `Some(n)`, at most `n` evenly-strided sample pairs are
///   plotted; if `None`, every sample pair is plotted.
#[derive(Debug, Clone, Default)]
#[non_exhaustive]
pub struct LissajousParams {
    /// Base plotting parameters shared across all plot types (title, labels, etc.).
    pub plot_params: PlotParams,
    /// Optional cap on the number of plotted sample pairs. `None` plots all samples.
    pub max_points: Option<usize>,
}

impl LissajousParams {
    /// Creates default Lissajous parameters.
    ///
    /// Plots every sample pair with default styling.
    ///
    /// # Returns
    /// A [`LissajousParams`] with default settings.
    ///
    /// # Example
    /// ```rust,no_run
    /// # use audio_samples::operations::plotting::LissajousParams;
    /// let params = LissajousParams::new();
    /// ```
    #[inline]
    #[must_use]
    pub fn new() -> Self {
        Self {
            plot_params: PlotParams::default(),
            max_points: None,
        }
    }

    /// Sets the shared plot styling parameters (title, axis labels, grid, etc.).
    ///
    /// # Arguments
    /// - `plot_params` – Shared styling parameters.
    ///
    /// # Returns
    /// The updated [`LissajousParams`] for chaining.
    #[inline]
    #[must_use]
    pub fn with_plot_params(mut self, plot_params: PlotParams) -> Self {
        self.plot_params = plot_params;
        self
    }

    /// Caps the number of plotted sample pairs.
    ///
    /// When the signal has more samples than `max_points`, pairs are sampled at
    /// an even stride so the full duration is represented without overloading
    /// the renderer.
    ///
    /// # Arguments
    /// - `max_points` – Maximum number of plotted sample pairs.
    ///
    /// # Returns
    /// The updated [`LissajousParams`] for chaining.
    #[inline]
    #[must_use]
    pub const fn with_max_points(mut self, max_points: usize) -> Self {
        self.max_points = Some(max_points);
        self
    }
}

/// Interactive stereo X-Y (Lissajous) scatter plot.
///
/// # Purpose
/// Encapsulates a rendered Plotly scatter plot of channel 0 (x) against channel
/// 1 (y), used for stereo correlation / phase visualization.
///
/// # Intended Usage
/// Created via [`create_lissajous_plot`] or through the
/// [`crate::operations::AudioPlotting`] trait. Requires exactly two channels.
///
/// # Invariants
/// - The underlying `Plot` contains a single marker trace of (left, right) pairs.
pub struct LissajousPlot {
    _params: LissajousParams,
    plot: Plot,
}

impl PlotUtils for LissajousPlot {
    #[inline]
    fn html(&self) -> AudioSampleResult<String> {
        Ok(self.plot.to_html())
    }

    #[cfg(feature = "html_view")]
    #[inline]
    fn show(&self) -> AudioSampleResult<()> {
        let html = self.html()?;
        html_view::show(html).map_err(|e| {
            crate::AudioSampleError::Processing(crate::ProcessingError::external_dependency(
                "html_view",
                "show",
                e.to_string(),
            ))
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
                std::fs::write(path, html).map_err(|e| crate::AudioSampleError::io("save", &e))?;
                Ok(())
            }
            #[cfg(feature = "static-plots")]
            "png" | "svg" | "jpeg" | "jpg" | "webp" => {
                use plotly_static::{ImageFormat, StaticExporterBuilder};
                use serde_json::json;

                let mut static_exporter =
                    StaticExporterBuilder::default().build().map_err(|e| {
                        crate::AudioSampleError::Processing(
                            crate::ProcessingError::external_dependency(
                                "plotly_static",
                                "save",
                                e.to_string(),
                            ),
                        )
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
                        crate::AudioSampleError::Processing(
                            crate::ProcessingError::external_dependency(
                                "plotly_static",
                                "save",
                                e.to_string(),
                            ),
                        )
                    })
            }
            _ => {
                #[cfg(not(feature = "static-plots"))]
                return Err(crate::AudioSampleError::Feature(
                    crate::FeatureError::NotEnabled {
                        feature: "static-plots".to_string(),
                        operation: format!("save plot as {extension}"),
                    },
                ));

                #[cfg(feature = "static-plots")]
                return Err(crate::AudioSampleError::Parameter(
                    crate::ParameterError::InvalidValue {
                        parameter: "file_extension".to_string(),
                        reason: format!("Unsupported file extension: {}", extension),
                    },
                ));
            }
        }
    }
}

impl PlotComponent for LissajousPlot {
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
        false // X-Y plot has no time axis to share
    }
}

/// Creates an interactive Lissajous (stereo X-Y) plot from a 2-channel signal.
///
/// Scatters channel 0 (x) against channel 1 (y). This is the standard stereo
/// correlation / phase visualization (a goniometer / vectorscope).
///
/// # Arguments
/// * `audio` — The audio data to analyze. Must have **exactly two channels**.
/// * `params` — Configuration controlling the point cap and plot styling.
///
/// # Returns
/// A [`LissajousPlot`] on success, which can be saved or displayed.
///
/// # Errors
/// Returns [`crate::AudioSampleError::Parameter`] if `audio` does not have
/// exactly two channels.
///
/// # Example
/// ```rust,no_run
/// use audio_samples::{AudioSamples, sample_rate};
/// use audio_samples::operations::plotting::{create_lissajous_plot, LissajousParams, PlotUtils};
/// use ndarray::array;
///
/// let audio = AudioSamples::new_multi_channel(
///     array![[0.0f32, 0.5, 1.0], [0.0, 0.5, 1.0]],
///     sample_rate!(44100),
/// )?;
/// let plot = create_lissajous_plot(&audio, &LissajousParams::new())?;
/// let html = plot.html()?;
/// # Ok::<(), audio_samples::AudioSampleError>(())
/// ```
#[inline]
pub fn create_lissajous_plot<T>(
    audio: &AudioSamples<'_, T>,
    params: &LissajousParams,
) -> AudioSampleResult<LissajousPlot>
where
    T: StandardSample,
{
    let n_channels = audio.num_channels().get();
    if n_channels != 2 {
        return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
            "num_channels",
            format!("Lissajous plot requires exactly 2 channels, got {n_channels}"),
        )));
    }

    // Extract both channels as f64 samples. Multi-channel storage is shaped
    // (channels, samples_per_channel), so each row is one channel.
    let multi = audio.as_multi_channel().ok_or_else(|| {
        AudioSampleError::Parameter(ParameterError::invalid_value(
            "audio",
            "expected multi-channel data for a Lissajous plot".to_string(),
        ))
    })?;

    let left = multi.row(0);
    let right = multi.row(1);

    let total = left.len();
    // Even stride so the whole signal is represented under the point cap.
    let stride = match params.max_points {
        Some(cap) if cap > 0 && total > cap => total.div_ceil(cap),
        _ => 1,
    };

    let xs: Vec<f64> = left
        .iter()
        .step_by(stride)
        .map(|v| (*v).convert_to())
        .collect();
    let ys: Vec<f64> = right
        .iter()
        .step_by(stride)
        .map(|v| (*v).convert_to())
        .collect();

    let trace = Scatter::new(xs, ys).mode(Mode::Markers).name("Stereo X-Y");

    let mut plot = Plot::new();
    plot.add_trace(trace);

    let x_label = params
        .plot_params
        .x_label
        .clone()
        .unwrap_or_else(|| "Channel 0 (Left)".to_string());
    let y_label = params
        .plot_params
        .y_label
        .clone()
        .unwrap_or_else(|| "Channel 1 (Right)".to_string());

    let x_axis = Axis::new().title(plotly::common::Title::from(x_label.as_str()));
    let y_axis = Axis::new().title(plotly::common::Title::from(y_label.as_str()));

    let mut layout = Layout::new().x_axis(x_axis).y_axis(y_axis);

    if let Some(ref title) = params.plot_params.title {
        layout = layout.title(plotly::common::Title::from(title.as_str()));
    }

    if params.plot_params.grid {
        let x_axis_with_grid = Axis::new()
            .title(plotly::common::Title::from(x_label.as_str()))
            .grid_color("lightgray");
        let y_axis_with_grid = Axis::new()
            .title(plotly::common::Title::from(y_label.as_str()))
            .grid_color("lightgray");

        layout = layout.x_axis(x_axis_with_grid).y_axis(y_axis_with_grid);
    }

    plot.set_layout(layout);

    Ok(LissajousPlot {
        _params: params.clone(),
        plot,
    })
}

#[cfg(test)]
#[cfg(feature = "plotting")]
mod tests {
    use super::*;
    use crate::sample_rate;
    use ndarray::array;

    #[test]
    fn test_lissajous_renders_nonempty_html() {
        let audio = AudioSamples::new_multi_channel(
            array![
                [0.0f32, 0.25, 0.5, 0.75, 1.0],
                [0.0f32, 0.25, 0.5, 0.75, 1.0]
            ],
            sample_rate!(44100),
        )
        .unwrap();

        let plot = create_lissajous_plot(&audio, &LissajousParams::new()).unwrap();
        let html = plot.html().unwrap();
        assert!(!html.is_empty());
        assert!(
            html.contains("Stereo X-Y"),
            "rendered HTML must contain the trace name"
        );
    }

    #[test]
    fn test_lissajous_max_points_builder_strides() {
        // 10 samples per channel, capped to 3 points -> stride should reduce count.
        let audio = AudioSamples::new_multi_channel(
            array![
                [0.0f32, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                [0.9f32, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]
            ],
            sample_rate!(44100),
        )
        .unwrap();

        let params = LissajousParams::new().with_max_points(3);
        assert_eq!(params.max_points, Some(3));

        let plot = create_lissajous_plot(&audio, &params).unwrap();
        let html = plot.html().unwrap();
        assert!(!html.is_empty());
    }

    #[test]
    fn test_lissajous_errors_on_non_two_channels() {
        // Mono signal must be rejected.
        let mono = AudioSamples::new_mono(array![0.0f32, 0.5, 1.0], sample_rate!(44100)).unwrap();
        assert!(create_lissajous_plot(&mono, &LissajousParams::new()).is_err());

        // Three channels must be rejected too.
        let three = AudioSamples::new_multi_channel(
            array![[0.0f32, 0.5], [0.5f32, 1.0], [1.0f32, 0.0]],
            sample_rate!(44100),
        )
        .unwrap();
        assert!(create_lissajous_plot(&three, &LissajousParams::new()).is_err());
    }
}
