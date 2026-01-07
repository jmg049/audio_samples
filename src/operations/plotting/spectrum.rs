use plotly::common::Mode;
use plotly::layout::Axis;
use plotly::{Layout, Plot, Scatter};
use std::path::Path;

use super::composite::PlotComponent;
use super::{PlotParams, PlotUtils};
use crate::operations::traits::AudioTransforms;
use crate::{AudioSampleResult, AudioSamples, StandardSample};

/// Parameters for magnitude spectrum plots
#[derive(Debug, Clone)]
pub struct MagnitudeSpectrumParams {
    pub plot_params: PlotParams,
    pub db_scale: bool,
    pub freq_range: Option<(f64, f64)>, // Optional frequency range to display (min_hz, max_hz)
    pub window_type: Option<spectrograms::WindowType>,
    pub n_fft: Option<usize>,
    pub frame_position: Option<f64>, // Time position in seconds (None = use entire signal)
}

impl MagnitudeSpectrumParams {
    /// Create magnitude spectrum params with dB scale (default)
    pub fn db() -> Self {
        Self {
            plot_params: PlotParams::default(),
            db_scale: true,
            freq_range: None,
            window_type: None,
            n_fft: None,
            frame_position: None,
        }
    }

    /// Create magnitude spectrum params with linear scale
    pub fn linear() -> Self {
        Self {
            plot_params: PlotParams::default(),
            db_scale: false,
            freq_range: None,
            window_type: None,
            n_fft: None,
            frame_position: None,
        }
    }
}

impl Default for MagnitudeSpectrumParams {
    fn default() -> Self {
        Self::db()
    }
}

/// A magnitude spectrum plot showing frequency content
pub struct MagnitudeSpectrumPlot {
    _params: MagnitudeSpectrumParams,
    plot: Plot,
}

impl PlotUtils for MagnitudeSpectrumPlot {
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
                // TODO! Make width/height/scale configurable
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
                    })
            }
            _ => {
                #[cfg(not(feature = "static-plots"))]
                return Err(crate::AudioSampleError::Feature(
                    crate::FeatureError::NotEnabled {
                        feature: "static-plots".to_string(),
                        operation: format!("save plot as {}", extension),
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

impl PlotComponent for MagnitudeSpectrumPlot {
    fn get_plot(&self) -> &Plot {
        &self.plot
    }

    fn get_plot_mut(&mut self) -> &mut Plot {
        &mut self.plot
    }

    fn requires_shared_x_axis(&self) -> bool {
        false // Frequency spectrum doesn't share time axis
    }
}

/// Create a magnitude spectrum plot from audio samples
///
/// # Arguments
/// * `audio` - Input audio samples (mono or multi-channel)
/// * `params` - Plot parameters
///
/// # Returns
/// A MagnitudeSpectrumPlot showing the frequency content
pub fn create_magnitude_spectrum_plot<T>(
    audio: &AudioSamples<'_, T>,
    params: &MagnitudeSpectrumParams,
) -> AudioSampleResult<MagnitudeSpectrumPlot>
where
    T: StandardSample,
{
    use crate::operations::traits::AudioChannelOps;
    use crate::operations::types::MonoConversionMethod;

    // Convert to mono if multi-channel (always clone to own the data for FFT)
    let mono_audio = if audio.num_channels().get() > 1 {
        audio.to_mono(MonoConversionMethod::Average)?
    } else {
        // Clone to own the data
        audio.clone().into_owned()
    };

    // Determine FFT size (must be >= signal length)
    use std::num::NonZeroUsize;
    let signal_len = mono_audio.samples_per_channel().get();
    let n_fft = params.n_fft.unwrap_or_else(|| {
        // Default: use next power of 2 >= signal length
        let mut pow2 = 1;
        while pow2 < signal_len {
            pow2 *= 2;
        }
        pow2 // No cap - must be >= signal length
    });

    // Ensure n_fft >= signal_len
    let n_fft = n_fft.max(signal_len);
    let n_fft_nz = NonZeroUsize::new(n_fft).unwrap();

    // Perform FFT
    let fft_result = mono_audio.fft(n_fft_nz)?;

    // Extract magnitude from complex FFT result (only first channel since we converted to mono)
    let channel_fft = fft_result.row(0);
    let magnitudes: Vec<f64> = channel_fft
        .iter()
        .map(|c| {
            let mag = c.norm() as f64;
            if params.db_scale {
                // Convert to dB scale, avoiding log(0)
                if mag > 1e-10 {
                    20.0 * mag.log10()
                } else {
                    -200.0 // Floor at -200 dB
                }
            } else {
                mag
            }
        })
        .collect();

    // Generate frequency axis (only positive frequencies)
    let sample_rate = audio.sample_rate().get() as f64;
    let n_bins = magnitudes.len();
    let freq_bin = sample_rate / (n_fft as f64);
    let mut frequencies: Vec<f64> = (0..n_bins).map(|i| i as f64 * freq_bin).collect();

    // Apply frequency range filter if specified
    let mut filtered_magnitudes = magnitudes;
    if let Some((min_freq, max_freq)) = params.freq_range {
        let filtered_pairs: Vec<(f64, f64)> = frequencies
            .iter()
            .zip(filtered_magnitudes.iter())
            .filter(|(f, _)| **f >= min_freq && **f <= max_freq)
            .map(|(f, m)| (*f, *m))
            .collect();

        frequencies = filtered_pairs.iter().map(|(f, _)| *f).collect();
        filtered_magnitudes = filtered_pairs.iter().map(|(_, m)| *m).collect();
    }

    // Create plotly trace
    let trace = Scatter::new(frequencies, filtered_magnitudes)
        .mode(Mode::Lines)
        .name("Magnitude");

    let mut plot = Plot::new();
    plot.add_trace(trace);

    // Configure layout
    let x_label = params
        .plot_params
        .x_label
        .clone()
        .unwrap_or_else(|| "Frequency (Hz)".to_string());
    let y_label = params.plot_params.y_label.clone().unwrap_or_else(|| {
        if params.db_scale {
            "Magnitude (dB)".to_string()
        } else {
            "Magnitude".to_string()
        }
    });

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

    Ok(MagnitudeSpectrumPlot {
        _params: params.clone(),
        plot,
    })
}
