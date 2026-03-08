use plotly::common::Mode;
use plotly::layout::Axis;
use plotly::{Layout, Plot, Scatter};
use std::num::NonZeroUsize;
use std::path::Path;

use super::composite::PlotComponent;
use super::{PlotParams, PlotUtils};
use crate::operations::traits::AudioTransforms;
use crate::{AudioSampleResult, AudioSamples, StandardSample};

/// Configuration parameters for magnitude spectrum plot generation.
///
/// # Purpose
/// Encapsulates all settings required to compute and visualize a magnitude spectrum from
/// audio samples, including FFT size, windowing, amplitude scale (linear or dB), and
/// frequency range filtering.
///
/// # Intended Usage
/// Construct via [`MagnitudeSpectrumParams::db()`] or [`linear()`] constructor methods,
/// then modify fields as needed before passing to [`create_magnitude_spectrum_plot`] or
/// the [`crate::AudioPlotting`] trait methods.
///
/// # Invariants
/// - If `n_fft` is `None`, it defaults to the next power of 2 greater than or equal to
///   the signal length.
/// - If `freq_range` is `Some`, only frequency bins within `(min_freq, max_freq)` are
///   displayed.
/// - `frame_position` is currently unused (reserved for future frame-based analysis).
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct MagnitudeSpectrumParams {
    /// Base plotting parameters shared across all plot types (title, labels, legend, etc.).
    pub plot_params: PlotParams,
    /// If `true`, magnitude values are converted to decibels (20 * log10(magnitude)).
    /// If `false`, raw magnitude values are plotted.
    pub db_scale: bool,
    /// Optional frequency range to display as `(min_hz, max_hz)`. If `None`, shows all bins.
    pub freq_range: Option<(f64, f64)>,
    /// Window type for FFT (reserved for future use; currently unused).
    pub window_type: Option<spectrograms::WindowType>,
    /// FFT size. If `None`, defaults to next power of 2 >= signal length.
    pub n_fft: Option<NonZeroUsize>,
    /// Time position in seconds for frame-based analysis (reserved; currently unused).
    /// If `None`, computes spectrum for the entire signal.
    pub frame_position: Option<f64>,
}

impl MagnitudeSpectrumParams {
    /// Creates default parameters for a dB-scaled magnitude spectrum.
    ///
    /// Constructs a parameter set with decibel amplitude scaling, automatic FFT size selection
    /// (next power of 2 >= signal length), and no frequency range filtering. This is the most
    /// common configuration for analyzing frequency content with a perceptually meaningful scale.
    ///
    /// # Returns
    /// A [`MagnitudeSpectrumParams`] instance configured for dB-scale visualization.
    ///
    /// # Example
    /// ```rust,no_run
    /// # use audio_samples::operations::plotting::MagnitudeSpectrumParams;
    /// let params = MagnitudeSpectrumParams::db();
    /// ```
    #[inline]
    #[must_use]
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

    /// Creates default parameters for a linear-scaled magnitude spectrum.
    ///
    /// Constructs a parameter set with linear amplitude scaling, automatic FFT size selection
    /// (next power of 2 >= signal length), and no frequency range filtering. Useful for
    /// inspecting raw magnitude values without logarithmic compression.
    ///
    /// # Returns
    /// A [`MagnitudeSpectrumParams`] instance configured for linear-scale visualization.
    ///
    /// # Example
    /// ```rust,no_run
    /// # use audio_samples::operations::plotting::MagnitudeSpectrumParams;
    /// let params = MagnitudeSpectrumParams::linear();
    /// ```
    #[inline]
    #[must_use]
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
    #[inline]
    fn default() -> Self {
        Self::db()
    }
}

/// Interactive frequency-domain magnitude plot.
///
/// # Purpose
/// Encapsulates a rendered Plotly line plot showing the magnitude spectrum (frequency vs.
/// amplitude) of an audio signal. Provides methods for saving to disk and generating HTML.
///
/// # Intended Usage
/// Created via [`create_magnitude_spectrum_plot`] or through the [`crate::AudioPlotting`]
/// trait. Unlike spectrograms, magnitude spectrum plots show a single frequency snapshot
/// (typically of the entire signal) rather than a time-varying representation.
///
/// # Invariants
/// - The underlying `Plot` contains a single line trace representing the magnitude spectrum.
/// - Frequency axis is always in Hz.
/// - Amplitude axis is either in raw magnitude units or decibels, depending on `db_scale`.
pub struct MagnitudeSpectrumPlot {
    _params: MagnitudeSpectrumParams,
    plot: Plot,
}

impl PlotUtils for MagnitudeSpectrumPlot {
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

impl PlotComponent for MagnitudeSpectrumPlot {
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
        false // Frequency spectrum doesn't share time axis
    }
}

/// Creates an interactive magnitude spectrum plot from audio data.
///
/// Computes the FFT of the input signal (converting multi-channel audio to mono via averaging),
/// extracts magnitude values, optionally converts to dB scale, and renders as a Plotly line plot.
/// Supports frequency range filtering and automatic FFT size selection.
///
/// # Arguments
/// * `audio` — The audio data to analyze. Multi-channel audio is automatically converted to
///   mono by averaging all channels.
/// * `params` — Configuration parameters controlling FFT size, amplitude scale (linear or dB),
///   frequency range, and plot styling.
///
/// # Returns
/// A [`MagnitudeSpectrumPlot`] on success, which can be saved or displayed.
///
/// # Errors
/// Returns an error if:
/// - The FFT computation fails (e.g., insufficient samples, invalid FFT size).
/// - File I/O fails when saving the plot.
///
/// # Example
/// ```rust,no_run
/// use audio_samples::{AudioSamples, sample_rate};
/// use audio_samples::operations::plotting::{create_magnitude_spectrum_plot, MagnitudeSpectrumParams, PlotUtils};
///
/// let audio = AudioSamples::new_mono(ndarray::Array1::from_elem(4096, 0.0f32), sample_rate!(44100))?;
/// let params = MagnitudeSpectrumParams::db();
/// let plot = create_magnitude_spectrum_plot(&audio, &params)?;
/// plot.save("spectrum.html")?;
/// # Ok::<(), audio_samples::AudioSampleError>(())
/// ```
#[inline]
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
    let signal_len = mono_audio.samples_per_channel();
    let n_fft = params.n_fft.unwrap_or_else(|| {
        // Default: use next power of 2 >= signal length
        let mut pow2 = 1;
        while pow2 < signal_len.get() {
            pow2 *= 2;
        }
        // safety: will be at least 1 from above
        unsafe { NonZeroUsize::new_unchecked(pow2) } // No cap - must be >= signal length
    });

    // Ensure n_fft >= signal_len
    let n_fft_nz = n_fft.max(signal_len);

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
    let sample_rate = f64::from(audio.sample_rate().get());
    let n_bins = magnitudes.len();
    let freq_bin = sample_rate / (n_fft.get() as f64);
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
