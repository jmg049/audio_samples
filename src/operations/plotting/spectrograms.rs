use plotly::common::{ColorBar, ColorScale, ColorScalePalette, Mode};
use plotly::layout::Annotation;
use plotly::{
    HeatMap, Plot, Scatter,
    layout::{Axis, GridPattern, LayoutGrid, RowOrder, Shape, ShapeLine, ShapeType},
};
use std::path::Path;

#[cfg(feature = "transforms")]
use spectrograms::{GammatoneParams, LogHzParams, LogParams, MelParams, SpectrogramParams};

use crate::{
    AudioChannelOps, AudioSampleResult, AudioSamples, AudioTransforms, AudioTypeConversion,
    StandardSample,
    operations::{
        plotting::{
            ChannelManagementStrategy, Layout, PlotParams, PlotUtils, composite::PlotComponent,
            configure_frequency_axis, configure_time_axis,
        },
        types::MonoConversionMethod,
    },
};

/// Default padding fraction for auto-zooming frequency range (e.g., 0.1 = 10%)
const DEFAULT_FREQ_PADDING: f64 = 0.4; // 40% padding on each side when auto-zooming frequency range

pub struct SpectrogramPlot {
    _params: SpectrogramPlotParams,
    plot: Plot,
}

impl PlotUtils for SpectrogramPlot {
    fn html(&self) -> crate::AudioSampleResult<String> {
        Ok(self.plot.to_html())
    }

    #[cfg(feature = "html_view")]
    fn show(&self) -> crate::AudioSampleResult<()> {
        let html = self.html()?;
        html_view::show(html).map_err(|e| {
            crate::AudioSampleError::unsupported(format!("Failed to show plot: {}", e))
        })?;
        Ok(())
    }

    fn save<P: AsRef<Path>>(&self, path: P) -> crate::AudioSampleResult<()> {
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
                            "Failed to initialize static exporter: {}",
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
                let scale: f64 = 1.0;
                let plot = self.plot.to_json();
                let plot = json!(plot);
                static_exporter
                    .write_fig(path, &plot, format, width, height, scale)
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
                    feature: "plotly_static".to_string(),
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

impl PlotComponent for SpectrogramPlot {
    fn get_plot(&self) -> &Plot {
        &self.plot
    }

    fn get_plot_mut(&mut self) -> &mut Plot {
        &mut self.plot
    }

    fn requires_shared_x_axis(&self) -> bool {
        true // Spectrograms are time-based
    }
}

impl SpectrogramPlot {
    /// Add a vertical line at the specified time position (for marking events).
    ///
    /// # Arguments
    /// * `time` - Time position in seconds
    /// * `label` - Optional label for the line
    ///
    /// # Returns
    /// Self for method chaining
    pub fn add_vline(mut self, time: f64, label: Option<&str>) -> Self {
        let shape = Shape::new()
            .shape_type(ShapeType::Line)
            .x0(time)
            .x1(time)
            .y0(0)
            .y1(1)
            .y_ref("paper")
            .line(
                ShapeLine::new()
                    .color("white".to_string())
                    .width(2.0)
                    .dash(plotly::common::DashType::Dash),
            );

        let mut layout = self.plot.layout().clone();
        layout.add_shape(shape);

        if let Some(label_text) = label {
            let annotation = Annotation::new()
                .x(time)
                .y(1.0)
                .y_ref("paper")
                .text(label_text)
                .show_arrow(false)
                .y_shift(10.0)
                .font(plotly::common::Font::new().color(plotly::color::NamedColor::White));
            layout.add_annotation(annotation);
        }

        self.plot.set_layout(layout);
        self
    }

    /// Add a horizontal line at the specified frequency (for marking frequency bands).
    ///
    /// # Arguments
    /// * `freq` - Frequency in Hz
    /// * `label` - Optional label for the line
    ///
    /// # Returns
    /// Self for method chaining
    pub fn add_hline(mut self, freq: f64, label: Option<&str>) -> Self {
        let shape = Shape::new()
            .shape_type(ShapeType::Line)
            .x0(0)
            .x1(1)
            .x_ref("paper")
            .y0(freq)
            .y1(freq)
            .line(
                ShapeLine::new()
                    .color("white".to_string())
                    .width(2.0)
                    .dash(plotly::common::DashType::Dash),
            );

        let mut layout = self.plot.layout().clone();
        layout.add_shape(shape);

        if let Some(label_text) = label {
            let annotation = Annotation::new()
                .x(1.0)
                .x_ref("paper")
                .y(freq)
                .text(label_text)
                .show_arrow(false)
                .x_shift(10.0)
                .font(plotly::common::Font::new().color(plotly::color::NamedColor::White));
            layout.add_annotation(annotation);
        }

        self.plot.set_layout(layout);
        self
    }

    /// Overlay a continuous contour line (e.g., for pitch/F0 tracking).
    ///
    /// # Arguments
    /// * `times` - Time points in seconds
    /// * `freqs` - Frequency values in Hz
    /// * `label` - Optional label for the contour
    ///
    /// # Returns
    /// Self for method chaining
    ///
    /// # Panics
    /// Panics if `times` and `freqs` have different lengths
    pub fn overlay_contour(mut self, times: &[f64], freqs: &[f64], label: Option<&str>) -> Self {
        assert_eq!(
            times.len(),
            freqs.len(),
            "times and freqs must have the same length"
        );

        // Use white with thicker line for better visibility on spectrograms
        let line = plotly::common::Line::new()
            .color("white".to_string())
            .width(4.0);

        let mut trace = Scatter::new(times.to_vec(), freqs.to_vec())
            .mode(Mode::LinesMarkers) // Add markers for better visibility
            .line(line)
            .marker(
                plotly::common::Marker::new()
                    .size(6)
                    .color("cyan".to_string()),
            )
            .show_legend(label.is_some());

        if let Some(label_text) = label {
            trace = trace.name(label_text);
        }

        self.plot.add_trace(trace);
        self
    }

    /// Add spectral centroid track overlay
    ///
    /// Use with [`crate::operations::plotting::dsp_overlays::compute_windowed_spectral_centroid`]
    /// to compute the spectral centroid over time.
    ///
    /// The centroid represents the "center of mass" of the spectrum and correlates
    /// with perceived brightness of the sound.
    ///
    /// # Arguments
    /// * `times` - Time points in seconds
    /// * `centroid_hz` - Spectral centroid values in Hz
    /// * `label` - Optional label (default: "Spectral Centroid")
    ///
    /// # Returns
    /// Self for method chaining
    ///
    /// # Example
    /// ```rust,ignore
    /// use audio_samples::operations::plotting::dsp_overlays;
    ///
    /// let (times, centroids) = dsp_overlays::compute_windowed_spectral_centroid(
    ///     &audio, 2048, 512
    /// );
    /// let plot = audio.plot_spectrogram(&params)?
    ///     .add_spectral_centroid(times, centroids, None);
    /// ```
    pub fn add_spectral_centroid(
        self,
        times: Vec<f64>,
        centroid_hz: Vec<f64>,
        label: Option<&str>,
    ) -> Self {
        let label = label.unwrap_or("Spectral Centroid");
        self.overlay_contour(&times, &centroid_hz, Some(label))
    }

    /// Add spectral rolloff track overlay
    ///
    /// Use with [`crate::operations::plotting::dsp_overlays::compute_windowed_spectral_rolloff`]
    /// to compute the spectral rolloff over time.
    ///
    /// The rolloff frequency is the point below which a specified percentage (e.g., 85%)
    /// of the spectral energy is contained. It indicates the "cutoff" of the spectrum.
    ///
    /// # Arguments
    /// * `times` - Time points in seconds
    /// * `rolloff_hz` - Spectral rolloff values in Hz
    /// * `label` - Optional label (default: "Spectral Rolloff")
    ///
    /// # Returns
    /// Self for method chaining
    ///
    /// # Example
    /// ```rust,ignore
    /// use audio_samples::operations::plotting::dsp_overlays;
    ///
    /// let (times, rolloff) = dsp_overlays::compute_windowed_spectral_rolloff(
    ///     &audio, 2048, 512, 0.85
    /// );
    /// let plot = audio.plot_spectrogram(&params)?
    ///     .add_spectral_rolloff(times, rolloff, None);
    /// ```
    pub fn add_spectral_rolloff(
        self,
        times: Vec<f64>,
        rolloff_hz: Vec<f64>,
        label: Option<&str>,
    ) -> Self {
        let label = label.unwrap_or("Spectral Rolloff");
        self.overlay_contour(&times, &rolloff_hz, Some(label))
    }
}

#[derive(Debug, Clone)]
pub struct SpectrogramPlotParams {
    pub plot_params: PlotParams,
    pub ch_mgmt_strategy: Option<ChannelManagementStrategy>,
    pub spectrogram_type: SpectrogramType,
    #[cfg(feature = "transforms")]
    pub stft_params: Option<SpectrogramParams>,
    pub colormap: Option<String>,
    pub colorbar_label: Option<String>,
    pub freq_range: Option<(f64, f64)>,
    pub time_range: Option<(f64, f64)>,
    /// Automatically detect and zoom to the frequency range containing significant energy
    /// If true, overrides freq_range with an intelligently detected range
    pub auto_zoom_freq: bool,
    /// Padding around the detected frequency range (as a fraction, e.g., 0.1 = 10%)
    /// Only used when auto_zoom_freq is enabled. Default: 0.1 (10%)
    pub freq_range_padding: Option<f64>,
}

#[derive(Debug, Clone)]
pub enum SpectrogramType {
    #[cfg(feature = "transforms")]
    LinearMagnitude,
    #[cfg(feature = "transforms")]
    LinearPower,
    #[cfg(feature = "transforms")]
    LinearDb { db_params: LogParams },
    #[cfg(feature = "transforms")]
    LogFreqMagnitude { loghz_params: LogHzParams },
    #[cfg(feature = "transforms")]
    LogFreqPower { loghz_params: LogHzParams },
    #[cfg(feature = "transforms")]
    LogFreqDb {
        loghz_params: LogHzParams,
        db_params: LogParams,
    },
    #[cfg(feature = "transforms")]
    MelMagnitude { mel_params: MelParams },
    #[cfg(feature = "transforms")]
    MelPower { mel_params: MelParams },
    #[cfg(feature = "transforms")]
    MelDb {
        mel_params: MelParams,
        db_params: LogParams,
    },
    #[cfg(feature = "transforms")]
    Gammatone { gammatone_params: GammatoneParams },
    #[cfg(feature = "transforms")]
    GammatonePower { gammatone_params: GammatoneParams },
    #[cfg(feature = "transforms")]
    GammatoneDb {
        gammatone_params: GammatoneParams,
        db_params: LogParams,
    },
}

impl SpectrogramPlotParams {
    #[cfg(feature = "transforms")]
    pub fn mel_db() -> Self {
        use std::num::NonZeroUsize;
        Self {
            plot_params: PlotParams::default(),
            ch_mgmt_strategy: None,
            spectrogram_type: SpectrogramType::MelDb {
                mel_params: MelParams::new(NonZeroUsize::new(128).unwrap(), 0.0, 8000.0).unwrap(),
                db_params: LogParams::new(-80.0).unwrap(),
            },
            stft_params: Some(
                SpectrogramParams::builder()
                    .sample_rate(44100.0)
                    .n_fft(crate::nzu!(2048))
                    .hop_size(crate::nzu!(512))
                    .window(spectrograms::WindowType::Hanning)
                    .centre(true)
                    .build()
                    .unwrap(),
            ),
            colormap: Some("Viridis".to_string()),
            colorbar_label: Some("Amplitude (dB)".to_string()),
            freq_range: None,
            time_range: None,
            auto_zoom_freq: true,     // Enable auto-zoom by default
            freq_range_padding: None, // Use default 10%
        }
    }

    #[cfg(feature = "transforms")]
    pub fn linear_magnitude() -> Self {
        Self {
            plot_params: PlotParams::default(),
            ch_mgmt_strategy: None,
            spectrogram_type: SpectrogramType::LinearMagnitude,
            stft_params: Some(
                SpectrogramParams::builder()
                    .sample_rate(44100.0)
                    .n_fft(crate::nzu!(2048))
                    .hop_size(crate::nzu!(512))
                    .window(spectrograms::WindowType::Hanning)
                    .centre(true)
                    .build()
                    .unwrap(),
            ),
            colormap: Some("Viridis".to_string()),
            colorbar_label: Some("Magnitude".to_string()),
            freq_range: None,
            time_range: None,
            auto_zoom_freq: true,     // Enable auto-zoom by default
            freq_range_padding: None, // Use default 10%
        }
    }
}

impl Default for SpectrogramPlotParams {
    fn default() -> Self {
        #[cfg(feature = "transforms")]
        {
            Self::mel_db()
        }
        #[cfg(not(feature = "transforms"))]
        {
            Self {
                plot_params: PlotParams::default(),
                ch_mgmt_strategy: None,
                colormap: Some("Viridis".to_string()),
                colorbar_label: None,
                freq_range: None,
                time_range: None,
                auto_zoom_freq: true,
                freq_range_padding: None,
            }
        }
    }
}

#[cfg(feature = "transforms")]
pub fn create_spectrogram_plot<T>(
    audio: &AudioSamples<'_, T>,
    params: &SpectrogramPlotParams,
) -> AudioSampleResult<SpectrogramPlot>
where
    T: StandardSample,
{
    let audio_f64 = audio.as_float();
    let strategy = params.ch_mgmt_strategy.unwrap_or_default();

    // Determine grid layout based on channel management strategy
    let (rows, cols) = if audio.is_multi_channel() {
        match strategy {
            ChannelManagementStrategy::Average
            | ChannelManagementStrategy::First
            | ChannelManagementStrategy::Last => (1, 1),
            ChannelManagementStrategy::Separate(layout) => match layout {
                Layout::Vertical => (audio.num_channels().get() as usize, 1),
                Layout::Horizontal => (1, audio.num_channels().get() as usize),
            },
            ChannelManagementStrategy::Overlap => {
                return Err(crate::AudioSampleError::Parameter(crate::ParameterError::InvalidValue {
                    parameter: "ch_mgmt_strategy".to_string(),
                    reason: "Overlap strategy not supported for spectrograms. Use Average, First, Last, or Separate.".to_string()
                }));
            }
        }
    } else {
        (1, 1)
    };

    let mut plot = Plot::new();
    let colormap = params
        .colormap
        .clone()
        .unwrap_or_else(|| "Viridis".to_string());
    let colorbar_label = params
        .colorbar_label
        .clone()
        .unwrap_or_else(|| "Amplitude (dB)".to_string());

    // Store first spec_data for auto-zoom detection
    let mut first_spec_data: Option<SpectrogramData> = None;

    // Determine if we should use kHz mode (frequency > 1 kHz)
    // We'll determine this after getting the first spec_data
    let mut use_khz = false;

    // Process based on channel management strategy
    if audio.is_mono() {
        let mut spec_data = compute_spectrogram_data(&audio_f64, params)?;
        if first_spec_data.is_none() {
            first_spec_data = Some(spec_data.clone());
            // Determine kHz mode based on max frequency
            if let Some(max_freq) = spec_data
                .freq_axis
                .iter()
                .cloned()
                .fold(None, |max, x| Some(max.map_or(x, |m: f64| m.max(x))))
            {
                use_khz = max_freq > 1000.0;
            }
        }
        if use_khz {
            spec_data = scale_freq_to_khz(&spec_data);
        }
        add_spectrogram_trace(&mut plot, &spec_data, &colormap, &colorbar_label, None)?;
    } else {
        match strategy {
            ChannelManagementStrategy::Average => {
                let mono_audio = audio_f64.to_mono(MonoConversionMethod::Average)?;
                let mut spec_data = compute_spectrogram_data(&mono_audio, params)?;
                if first_spec_data.is_none() {
                    first_spec_data = Some(spec_data.clone());
                    // Determine kHz mode based on max frequency
                    if let Some(max_freq) = spec_data
                        .freq_axis
                        .iter()
                        .cloned()
                        .fold(None, |max, x| Some(max.map_or(x, |m: f64| m.max(x))))
                    {
                        use_khz = max_freq > 1000.0;
                    }
                }
                if use_khz {
                    spec_data = scale_freq_to_khz(&spec_data);
                }
                add_spectrogram_trace(&mut plot, &spec_data, &colormap, &colorbar_label, None)?;
            }
            ChannelManagementStrategy::First => {
                let mono_audio = audio_f64.to_mono(MonoConversionMethod::Left)?;
                let mut spec_data = compute_spectrogram_data(&mono_audio, params)?;
                if first_spec_data.is_none() {
                    first_spec_data = Some(spec_data.clone());
                    // Determine kHz mode based on max frequency
                    if let Some(max_freq) = spec_data
                        .freq_axis
                        .iter()
                        .cloned()
                        .fold(None, |max, x| Some(max.map_or(x, |m: f64| m.max(x))))
                    {
                        use_khz = max_freq > 1000.0;
                    }
                }
                if use_khz {
                    spec_data = scale_freq_to_khz(&spec_data);
                }
                add_spectrogram_trace(&mut plot, &spec_data, &colormap, &colorbar_label, None)?;
            }
            ChannelManagementStrategy::Last => {
                let mono_audio = audio_f64.to_mono(MonoConversionMethod::Right)?;
                let mut spec_data = compute_spectrogram_data(&mono_audio, params)?;
                if first_spec_data.is_none() {
                    first_spec_data = Some(spec_data.clone());
                    // Determine kHz mode based on max frequency
                    if let Some(max_freq) = spec_data
                        .freq_axis
                        .iter()
                        .cloned()
                        .fold(None, |max, x| Some(max.map_or(x, |m: f64| m.max(x))))
                    {
                        use_khz = max_freq > 1000.0;
                    }
                }
                if use_khz {
                    spec_data = scale_freq_to_khz(&spec_data);
                }
                add_spectrogram_trace(&mut plot, &spec_data, &colormap, &colorbar_label, None)?;
            }
            ChannelManagementStrategy::Separate(layout) => {
                for (idx, channel) in audio_f64.channels().enumerate() {
                    let mut spec_data = compute_spectrogram_data(&channel, params)?;
                    if first_spec_data.is_none() {
                        first_spec_data = Some(spec_data.clone());
                        // Determine kHz mode based on max frequency
                        if let Some(max_freq) = spec_data
                            .freq_axis
                            .iter()
                            .cloned()
                            .fold(None, |max, x| Some(max.map_or(x, |m: f64| m.max(x))))
                        {
                            use_khz = max_freq > 1000.0;
                        }
                    }
                    if use_khz {
                        spec_data = scale_freq_to_khz(&spec_data);
                    }
                    let (row, col) = match layout {
                        Layout::Vertical => (idx, 0),
                        Layout::Horizontal => (0, idx),
                    };
                    let axis_ref = axis_reference(row, col, cols);
                    add_spectrogram_trace(
                        &mut plot,
                        &spec_data,
                        &colormap,
                        &colorbar_label,
                        Some(axis_ref),
                    )?;
                }
            }
            ChannelManagementStrategy::Overlap => unreachable!(),
        }
    }

    // Determine frequency range (auto-zoom or manual)
    let freq_range_unscaled = if params.auto_zoom_freq && params.freq_range.is_none() {
        // Auto-detect frequency range based on energy
        if let Some(ref spec_data) = first_spec_data {
            let padding = params.freq_range_padding.unwrap_or(DEFAULT_FREQ_PADDING); // Default 10%
            Some(detect_frequency_range(spec_data, padding))
        } else {
            None
        }
    } else {
        params.freq_range
    };

    // Keep the original max frequency for axis label determination
    let max_freq_for_axis = freq_range_unscaled
        .map(|(_, max)| max)
        .unwrap_or(audio.sample_rate().get() as f64 / 2.0);

    // Scale freq_range to kHz if needed (for the actual plot range)
    let freq_range = if use_khz {
        freq_range_unscaled.map(|(min, max)| (min / 1000.0, max / 1000.0))
    } else {
        freq_range_unscaled
    };

    // Configure layout
    let mut layout = plotly::Layout::new()
        .title(params.plot_params.title.clone().unwrap_or_default())
        .show_legend(params.plot_params.show_legend)
        .grid(
            LayoutGrid::new()
                .rows(rows)
                .columns(cols)
                .pattern(GridPattern::Independent)
                .row_order(RowOrder::TopToBottom),
        );

    if audio.is_multi_channel() {
        if let ChannelManagementStrategy::Separate(layout_kind) = strategy {
            layout = configure_separate_axes(
                layout,
                rows,
                cols,
                layout_kind,
                &params.plot_params,
                freq_range,
                max_freq_for_axis,
            );
        }
    } else {
        // Single plot - set axis labels and frequency range
        let mut y_axis = configure_frequency_axis(
            Axis::new(),
            params
                .plot_params
                .y_label
                .clone()
                .or_else(|| Some("Frequency".to_string())),
            max_freq_for_axis,
        );

        if let Some((min_freq, max_freq)) = freq_range {
            y_axis = y_axis.range(vec![min_freq, max_freq]);
        }

        let x_axis = configure_time_axis(
            Axis::new(),
            params
                .plot_params
                .x_label
                .clone()
                .or_else(|| Some("Time".to_string())),
        );

        layout = layout.x_axis(x_axis).y_axis(y_axis);
    }

    plot.set_layout(layout);

    Ok(SpectrogramPlot {
        _params: params.clone(),
        plot,
    })
}

#[cfg(feature = "transforms")]
#[derive(Clone)]
struct SpectrogramData {
    time_axis: Vec<f64>,
    freq_axis: Vec<f64>,
    data: Vec<Vec<f64>>,
}

/// Detect the frequency range containing significant energy
///
/// Returns (min_freq, max_freq) with the specified padding
///
/// # Arguments
/// * `spec_data` - The spectrogram data to analyze
/// * `padding_fraction` - Padding to add on each side (e.g., 0.1 = 10%)
fn detect_frequency_range(spec_data: &SpectrogramData, padding_fraction: f64) -> (f64, f64) {
    let n_freq_bins = spec_data.freq_axis.len();
    let n_time_bins = spec_data.data.first().map(|row| row.len()).unwrap_or(0);

    if n_freq_bins == 0 || n_time_bins == 0 {
        return (0.0, spec_data.freq_axis.last().copied().unwrap_or(22050.0));
    }

    // Compute total energy per frequency bin (sum across time)
    let mut freq_energy: Vec<f64> = vec![0.0; n_freq_bins];
    for freq_idx in 0..n_freq_bins {
        for time_idx in 0..n_time_bins {
            let val = spec_data.data[freq_idx][time_idx];
            // For dB values, convert back to linear scale for energy calculation
            let energy = if val < 0.0 {
                // Assume dB scale: dB = 20*log10(amplitude), so amplitude = 10^(dB/20)
                // Energy proportional to amplitude^2 = 10^(dB/10)
                10_f64.powf(val / 10.0)
            } else {
                // Assume linear scale
                val * val
            };
            freq_energy[freq_idx] += energy;
        }
    }

    let total_energy: f64 = freq_energy.iter().sum();
    if total_energy == 0.0 {
        return (0.0, spec_data.freq_axis.last().copied().unwrap_or(22050.0));
    }

    // Find frequency range containing bins with significant energy
    let mut min_freq_idx = 0;
    let mut max_freq_idx = n_freq_bins - 1;

    // Find min frequency (first bin with significant energy)
    for (idx, &energy) in freq_energy.iter().enumerate() {
        if energy > total_energy * 0.001 {
            // More than 0.1% of total energy
            min_freq_idx = idx;
            break;
        }
    }

    // Find max frequency (last bin with significant energy)
    for (idx, &energy) in freq_energy.iter().enumerate().rev() {
        if energy > total_energy * 0.001 {
            max_freq_idx = idx;
            break;
        }
    }

    // Add padding on each side
    let range_size = max_freq_idx.saturating_sub(min_freq_idx);
    let padding = (range_size as f64 * padding_fraction) as usize;
    min_freq_idx = min_freq_idx.saturating_sub(padding);
    max_freq_idx = (max_freq_idx + padding).min(n_freq_bins - 1);

    let min_freq = spec_data.freq_axis[min_freq_idx];
    let max_freq = spec_data.freq_axis[max_freq_idx];

    (min_freq, max_freq)
}

#[cfg(feature = "transforms")]
fn compute_spectrogram_data<T>(
    audio: &AudioSamples<'_, T>,
    params: &SpectrogramPlotParams,
) -> AudioSampleResult<SpectrogramData>
where
    T: StandardSample,
{
    let spec_params = params.stft_params.as_ref().ok_or_else(|| {
        crate::AudioSampleError::Parameter(crate::ParameterError::Missing {
            parameter: "stft_params".to_string(),
        })
    })?;

    // Compute the appropriate spectrogram type
    let (time_axis, freq_axis, data_2d) = match &params.spectrogram_type {
        SpectrogramType::LinearMagnitude => {
            let spec = audio.linear_magnitude_spectrogram(spec_params)?;
            extract_spectrogram_data(spec)
        }
        SpectrogramType::LinearPower => {
            let spec = audio.linear_power_spectrogram(spec_params)?;
            extract_spectrogram_data(spec)
        }
        SpectrogramType::LinearDb { db_params } => {
            let spec = audio.linear_db_spectrogram(spec_params, db_params)?;
            extract_spectrogram_data(spec)
        }
        SpectrogramType::LogFreqMagnitude { loghz_params } => {
            let spec = audio.loghz_magnitude_spectrogram(spec_params, loghz_params)?;
            extract_spectrogram_data(spec)
        }
        SpectrogramType::LogFreqPower { loghz_params } => {
            let spec = audio.loghz_power_spectrogram(spec_params, loghz_params)?;
            extract_spectrogram_data(spec)
        }
        SpectrogramType::LogFreqDb {
            loghz_params,
            db_params,
        } => {
            let spec = audio.loghz_db_spectrogram(spec_params, loghz_params, db_params)?;
            extract_spectrogram_data(spec)
        }
        SpectrogramType::MelMagnitude { mel_params } => {
            let spec = audio.mel_mag_spectrogram(spec_params, mel_params)?;
            extract_spectrogram_data(spec)
        }
        SpectrogramType::MelPower { mel_params } => {
            let spec = audio.mel_power_spectrogram(spec_params, mel_params)?;
            extract_spectrogram_data(spec)
        }
        SpectrogramType::MelDb {
            mel_params,
            db_params,
        } => {
            let spec = audio.mel_db_spectrogram(spec_params, mel_params, db_params)?;
            extract_spectrogram_data(spec)
        }
        SpectrogramType::Gammatone { gammatone_params } => {
            let spec = audio.gammatone_magnitude_spectrogram(spec_params, gammatone_params)?;
            extract_spectrogram_data(spec)
        }
        SpectrogramType::GammatonePower { gammatone_params } => {
            let spec = audio.gammatone_power_spectrogram(spec_params, gammatone_params)?;
            extract_spectrogram_data(spec)
        }
        SpectrogramType::GammatoneDb {
            gammatone_params,
            db_params,
        } => {
            let spec = audio.gammatone_db_spectrogram(spec_params, gammatone_params, db_params)?;
            extract_spectrogram_data(spec)
        }
    };

    // Convert Array2 to Vec<Vec<f64>> for Plotly
    let (n_freq, n_time) = data_2d.dim();
    let mut z_data: Vec<Vec<f64>> = Vec::with_capacity(n_freq);
    for freq_idx in 0..n_freq {
        let mut row: Vec<f64> = Vec::with_capacity(n_time);
        for time_idx in 0..n_time {
            row.push(data_2d[[freq_idx, time_idx]]);
        }
        z_data.push(row);
    }

    Ok(SpectrogramData {
        time_axis,
        freq_axis,
        data: z_data,
    })
}

#[cfg(feature = "transforms")]
fn extract_spectrogram_data<FreqScale, AmpScale>(
    spec: spectrograms::Spectrogram<FreqScale, AmpScale>,
) -> (Vec<f64>, Vec<f64>, ndarray::Array2<f64>)
where
    FreqScale: Copy + Clone + 'static,
    AmpScale: spectrograms::AmpScaleSpec,
{
    let time_axis = spec.times().to_vec();
    let freq_axis = spec.frequencies().to_vec();
    let data = spec.data().clone();

    (time_axis, freq_axis, data)
}

/// Scale frequency values from Hz to kHz
#[cfg(feature = "transforms")]
fn scale_freq_to_khz(spec_data: &SpectrogramData) -> SpectrogramData {
    SpectrogramData {
        time_axis: spec_data.time_axis.clone(),
        freq_axis: spec_data.freq_axis.iter().map(|&f| f / 1000.0).collect(),
        data: spec_data.data.clone(),
    }
}

#[cfg(feature = "transforms")]
fn add_spectrogram_trace(
    plot: &mut Plot,
    spec_data: &SpectrogramData,
    colormap: &str,
    colorbar_label: &str,
    axis_ref: Option<(String, String)>,
) -> AudioSampleResult<()> {
    let colorscale = match colormap {
        // Perceptually uniform (recommended)
        "Viridis" => ColorScale::Palette(ColorScalePalette::Viridis),
        "Cividis" => ColorScale::Palette(ColorScalePalette::Cividis),
        // Sequential
        "Hot" => ColorScale::Palette(ColorScalePalette::Hot),
        "Greys" => ColorScale::Palette(ColorScalePalette::Greys),
        "YlGnBu" => ColorScale::Palette(ColorScalePalette::YlGnBu),
        "Greens" => ColorScale::Palette(ColorScalePalette::Greens),
        "YlOrRd" => ColorScale::Palette(ColorScalePalette::YlOrRd),
        "Reds" => ColorScale::Palette(ColorScalePalette::Reds),
        "Blues" => ColorScale::Palette(ColorScalePalette::Blues),
        // Diverging
        "Bluered" => ColorScale::Palette(ColorScalePalette::Bluered),
        "RdBu" => ColorScale::Palette(ColorScalePalette::RdBu),
        // Misc
        "Jet" => ColorScale::Palette(ColorScalePalette::Jet),
        "Rainbow" => ColorScale::Palette(ColorScalePalette::Rainbow),
        "Portland" => ColorScale::Palette(ColorScalePalette::Portland),
        "Blackbody" => ColorScale::Palette(ColorScalePalette::Blackbody),
        "Earth" => ColorScale::Palette(ColorScalePalette::Earth),
        "Electric" => ColorScale::Palette(ColorScalePalette::Electric),
        "Picnic" => ColorScale::Palette(ColorScalePalette::Picnic),
        _ => ColorScale::Palette(ColorScalePalette::Viridis), // Default
    };

    let mut heatmap = HeatMap::new(
        spec_data.time_axis.clone(),
        spec_data.freq_axis.clone(),
        spec_data.data.clone(),
    )
    .color_scale(colorscale)
    .color_bar(ColorBar::new().title(colorbar_label.to_string()))
    .show_scale(true);

    if let Some((x_axis, y_axis)) = axis_ref {
        heatmap = heatmap.x_axis(&x_axis).y_axis(&y_axis);
    }

    plot.add_trace(heatmap);
    Ok(())
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
    layout_kind: Layout,
    plot_params: &PlotParams,
    freq_range: Option<(f64, f64)>,
    max_freq: f64,
) -> plotly::Layout {
    match layout_kind {
        Layout::Vertical => {
            configure_vertical_axes(layout, rows, cols, plot_params, freq_range, max_freq)
        }
        Layout::Horizontal => {
            configure_horizontal_axes(layout, rows, cols, plot_params, freq_range, max_freq)
        }
    }
}

fn configure_vertical_axes(
    mut layout: plotly::Layout,
    rows: usize,
    cols: usize,
    plot_params: &PlotParams,
    freq_range: Option<(f64, f64)>,
    max_freq: f64,
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

            let mut x_axis = Axis::new().anchor(&axis_y_name);

            if !is_bottom_row {
                x_axis = x_axis
                    .matches(&base_axis_name)
                    .show_tick_labels(false)
                    .tick_length(0);
            } else if col == 0 {
                x_axis = configure_time_axis(
                    x_axis,
                    plot_params
                        .x_label
                        .clone()
                        .or_else(|| Some("Time".to_string())),
                );
            }

            layout = assign_x_axis(layout, axis_index, x_axis);

            let mut y_axis = Axis::new().anchor(&axis_x_name);
            if col == 0 && row == 0 {
                y_axis = configure_frequency_axis(
                    y_axis,
                    plot_params
                        .y_label
                        .clone()
                        .or_else(|| Some("Frequency".to_string())),
                    max_freq,
                );
            }

            // Apply frequency range if specified
            if let Some((min_freq, max_freq)) = freq_range {
                y_axis = y_axis.range(vec![min_freq, max_freq]);
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
    freq_range: Option<(f64, f64)>,
    max_freq: f64,
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

            let mut x_axis = Axis::new().anchor(&axis_y_name);
            if row == 0 {
                x_axis = configure_time_axis(
                    x_axis,
                    plot_params
                        .x_label
                        .clone()
                        .or_else(|| Some("Time".to_string())),
                );
            }

            layout = assign_x_axis(layout, axis_index, x_axis);

            let mut y_axis = Axis::new().anchor(&axis_x_name);
            if col != 0 {
                y_axis = y_axis
                    .matches(&base_axis_name)
                    .show_tick_labels(false)
                    .tick_length(0);
            } else if row == 0 {
                y_axis = configure_frequency_axis(
                    y_axis,
                    plot_params
                        .y_label
                        .clone()
                        .or_else(|| Some("Frequency".to_string())),
                    max_freq,
                );
            }

            // Apply frequency range if specified
            if let Some((min_freq, max_freq)) = freq_range {
                y_axis = y_axis.range(vec![min_freq, max_freq]);
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
        _ => panic!("Spectrogram plot supports up to eight subplot x-axes"),
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
        _ => panic!("Spectrogram plot supports up to eight subplot y-axes"),
    }
}
