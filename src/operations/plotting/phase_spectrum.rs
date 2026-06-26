//! Phase spectrum visualization.
//!
//! Mirrors [`spectrum`](crate::operations::plotting::spectrum) but plots the per-bin phase angle
//! (`Complex::arg()`, in radians) on the y-axis instead of magnitude. Phase is
//! optionally unwrapped to remove the ±π discontinuities introduced by the
//! principal-value branch of `atan2`, which is useful for inspecting group
//! delay and phase-linearity of a signal or filter.

use plotly::common::Mode;
use plotly::layout::Axis;
use plotly::{Layout, Plot, Scatter};
use std::num::NonZeroUsize;
use std::path::Path;

use super::composite::PlotComponent;
use super::{PlotParams, PlotUtils};
use crate::operations::traits::AudioTransforms;
use crate::{AudioSampleResult, AudioSamples, StandardSample};

/// Configuration parameters for phase spectrum plot generation.
///
/// # Purpose
/// Encapsulates the settings required to compute and visualize a phase spectrum
/// from audio samples: FFT size, optional phase unwrapping, frequency-range
/// filtering, and shared plot styling.
///
/// # Intended Usage
/// Construct via [`PhaseSpectrumParams::new`] (or [`Default::default`]) and refine
/// with the `with_*` builder setters before passing to
/// [`create_phase_spectrum_plot`] or the [`crate::operations::AudioPlotting`] trait.
///
/// # Invariants
/// - If `n_fft` is `None`, it defaults to the next power of 2 greater than or
///   equal to the signal length.
/// - If `freq_range` is `Some`, only frequency bins within `(min_freq, max_freq)`
///   are displayed.
/// - When `unwrap` is `true`, phase values are unwrapped along the frequency
///   axis so successive bins never jump by more than π.
#[derive(Debug, Clone, Default)]
#[non_exhaustive]
pub struct PhaseSpectrumParams {
    /// Base plotting parameters shared across all plot types (title, labels, etc.).
    pub plot_params: PlotParams,
    /// If `true`, phase values are unwrapped along the frequency axis to remove
    /// ±π discontinuities. If `false`, raw principal-value phase in `(-π, π]` is shown.
    pub unwrap: bool,
    /// Optional frequency range to display as `(min_hz, max_hz)`. If `None`, shows all bins.
    pub freq_range: Option<(f64, f64)>,
    /// FFT size. If `None`, defaults to next power of 2 >= signal length.
    pub n_fft: Option<NonZeroUsize>,
}

impl PhaseSpectrumParams {
    /// Creates default phase-spectrum parameters.
    ///
    /// Uses raw (wrapped) principal-value phase, automatic FFT size selection
    /// (next power of 2 >= signal length), and no frequency-range filtering.
    ///
    /// # Returns
    /// A [`PhaseSpectrumParams`] with default settings.
    ///
    /// # Example
    /// ```rust,no_run
    /// # use audio_samples::operations::plotting::PhaseSpectrumParams;
    /// let params = PhaseSpectrumParams::new();
    /// ```
    #[inline]
    #[must_use]
    pub fn new() -> Self {
        Self {
            plot_params: PlotParams::default(),
            unwrap: false,
            freq_range: None,
            n_fft: None,
        }
    }

    /// Sets the shared plot styling parameters (title, axis labels, grid, etc.).
    ///
    /// # Arguments
    /// - `plot_params` – Shared styling parameters.
    ///
    /// # Returns
    /// The updated [`PhaseSpectrumParams`] for chaining.
    #[inline]
    #[must_use]
    pub fn with_plot_params(mut self, plot_params: PlotParams) -> Self {
        self.plot_params = plot_params;
        self
    }

    /// Enables or disables phase unwrapping along the frequency axis.
    ///
    /// # Arguments
    /// - `unwrap` – `true` to unwrap phase, `false` for raw principal values.
    ///
    /// # Returns
    /// The updated [`PhaseSpectrumParams`] for chaining.
    #[inline]
    #[must_use]
    pub const fn with_unwrap(mut self, unwrap: bool) -> Self {
        self.unwrap = unwrap;
        self
    }

    /// Restricts the displayed frequency range to `(min_hz, max_hz)`.
    ///
    /// # Arguments
    /// - `min_hz` – Lower frequency bound, inclusive.
    /// - `max_hz` – Upper frequency bound, inclusive.
    ///
    /// # Returns
    /// The updated [`PhaseSpectrumParams`] for chaining.
    #[inline]
    #[must_use]
    pub const fn with_freq_range(mut self, min_hz: f64, max_hz: f64) -> Self {
        self.freq_range = Some((min_hz, max_hz));
        self
    }

    /// Sets an explicit FFT size.
    ///
    /// # Arguments
    /// - `n_fft` – FFT length. Clamped up to the signal length at render time.
    ///
    /// # Returns
    /// The updated [`PhaseSpectrumParams`] for chaining.
    #[inline]
    #[must_use]
    pub const fn with_n_fft(mut self, n_fft: NonZeroUsize) -> Self {
        self.n_fft = Some(n_fft);
        self
    }
}

/// Interactive frequency-domain phase plot.
///
/// # Purpose
/// Encapsulates a rendered Plotly line plot showing the phase spectrum
/// (frequency vs. phase angle in radians) of an audio signal.
///
/// # Intended Usage
/// Created via [`create_phase_spectrum_plot`] or through the
/// [`crate::operations::AudioPlotting`] trait.
///
/// # Invariants
/// - The underlying `Plot` contains a single line trace representing the phase spectrum.
/// - The frequency axis is always in Hz.
/// - The phase axis is in radians (wrapped to `(-π, π]` unless unwrapping is enabled).
pub struct PhaseSpectrumPlot {
    _params: PhaseSpectrumParams,
    plot: Plot,
}

impl PlotUtils for PhaseSpectrumPlot {
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

impl PlotComponent for PhaseSpectrumPlot {
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
        false // Frequency spectrum doesn't share a time axis
    }
}

/// Creates an interactive phase spectrum plot from audio data.
///
/// Computes the FFT of the input signal (converting multi-channel audio to mono
/// via averaging), extracts the per-bin phase angle (`Complex::arg()`),
/// optionally unwraps it along the frequency axis, and renders it as a Plotly
/// line plot. Supports frequency-range filtering and automatic FFT size selection.
///
/// # Arguments
/// * `audio` — The audio data to analyze. Multi-channel audio is converted to
///   mono by averaging all channels.
/// * `params` — Configuration controlling FFT size, unwrapping, frequency range,
///   and plot styling.
///
/// # Returns
/// A [`PhaseSpectrumPlot`] on success, which can be saved or displayed.
///
/// # Errors
/// Returns an error if the FFT computation fails (e.g. insufficient samples,
/// invalid FFT size).
///
/// # Example
/// ```rust,no_run
/// use audio_samples::{AudioSamples, sample_rate};
/// use audio_samples::operations::plotting::{create_phase_spectrum_plot, PhaseSpectrumParams, PlotUtils};
///
/// let audio = AudioSamples::new_mono(ndarray::Array1::from_elem(4096, 0.0f32), sample_rate!(44100))?;
/// let params = PhaseSpectrumParams::new().with_unwrap(true);
/// let plot = create_phase_spectrum_plot(&audio, &params)?;
/// let html = plot.html()?;
/// # Ok::<(), audio_samples::AudioSampleError>(())
/// ```
#[inline]
pub fn create_phase_spectrum_plot<T>(
    audio: &AudioSamples<'_, T>,
    params: &PhaseSpectrumParams,
) -> AudioSampleResult<PhaseSpectrumPlot>
where
    T: StandardSample,
{
    use crate::operations::traits::AudioChannelOps;
    use crate::operations::types::MonoConversionMethod;

    // Convert to mono if multi-channel (always own the data for FFT).
    let mono_audio = if audio.num_channels().get() > 1 {
        audio.to_mono(MonoConversionMethod::Average)?
    } else {
        audio.clone().into_owned()
    };

    // Determine FFT size (must be >= signal length).
    let signal_len = mono_audio.samples_per_channel();
    let n_fft = params.n_fft.unwrap_or_else(|| {
        let mut pow2 = 1;
        while pow2 < signal_len.get() {
            pow2 *= 2;
        }
        // safety: pow2 is at least 1 from above
        unsafe { NonZeroUsize::new_unchecked(pow2) }
    });

    // Ensure n_fft >= signal_len.
    let n_fft_nz = n_fft.max(signal_len);

    // Perform FFT.
    let fft_result = mono_audio.fft(n_fft_nz)?;

    // Extract phase from complex FFT result (first channel only since mono).
    let channel_fft = fft_result.row(0);
    let mut phases: Vec<f64> = channel_fft.iter().map(|c| c.arg()).collect();

    // Optionally unwrap the phase along the frequency axis.
    if params.unwrap {
        unwrap_phase(&mut phases);
    }

    // Generate frequency axis (only positive frequencies up to Nyquist as returned by fft).
    let sample_rate = f64::from(audio.sample_rate().get());
    let n_bins = phases.len();
    let freq_bin = sample_rate / (n_fft.get() as f64);
    let mut frequencies: Vec<f64> = (0..n_bins).map(|i| i as f64 * freq_bin).collect();

    // Apply frequency range filter if specified.
    let mut filtered_phases = phases;
    if let Some((min_freq, max_freq)) = params.freq_range {
        let filtered_pairs: Vec<(f64, f64)> = frequencies
            .iter()
            .zip(filtered_phases.iter())
            .filter(|(f, _)| **f >= min_freq && **f <= max_freq)
            .map(|(f, p)| (*f, *p))
            .collect();

        frequencies = filtered_pairs.iter().map(|(f, _)| *f).collect();
        filtered_phases = filtered_pairs.iter().map(|(_, p)| *p).collect();
    }

    // Create plotly trace.
    let trace = Scatter::new(frequencies, filtered_phases)
        .mode(Mode::Lines)
        .name("Phase");

    let mut plot = Plot::new();
    plot.add_trace(trace);

    // Configure layout.
    let x_label = params
        .plot_params
        .x_label
        .clone()
        .unwrap_or_else(|| "Frequency (Hz)".to_string());
    let y_label = params
        .plot_params
        .y_label
        .clone()
        .unwrap_or_else(|| "Phase (radians)".to_string());

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

    Ok(PhaseSpectrumPlot {
        _params: params.clone(),
        plot,
    })
}

/// Unwraps a phase sequence in place so successive samples never jump by more
/// than π. Standard `atan2`-branch unwrapping: whenever the difference between
/// consecutive samples exceeds π in magnitude, a multiple of 2π is added to
/// bring it back within `(-π, π]`.
fn unwrap_phase(phases: &mut [f64]) {
    use std::f64::consts::PI;
    let two_pi = 2.0 * PI;
    let mut correction = 0.0;
    for i in 1..phases.len() {
        let mut delta = (phases[i] + correction) - phases[i - 1];
        // Reduce delta into (-PI, PI] by shifting the running correction.
        while delta > PI {
            correction -= two_pi;
            delta -= two_pi;
        }
        while delta <= -PI {
            correction += two_pi;
            delta += two_pi;
        }
        phases[i] += correction;
    }
}

#[cfg(test)]
#[cfg(feature = "plotting")]
mod tests {
    use super::*;
    use crate::sample_rate;
    use crate::utils::generation::sine_wave;
    use std::time::Duration;

    fn test_signal() -> AudioSamples<'static, f32> {
        sine_wave::<f32>(440.0, Duration::from_millis(50), sample_rate!(44100), 0.8)
    }

    #[test]
    fn test_phase_spectrum_renders_nonempty_html() {
        let audio = test_signal();
        let params = PhaseSpectrumParams::new();
        let plot = create_phase_spectrum_plot(&audio, &params).unwrap();

        let html = plot.html().unwrap();
        assert!(!html.is_empty());
        // The trace name should appear in the rendered plot data.
        assert!(
            html.contains("Phase"),
            "rendered HTML must contain the trace name"
        );
    }

    #[test]
    fn test_phase_spectrum_unwrap_builder() {
        let audio = test_signal();
        // Exercise the builder setters.
        let params = PhaseSpectrumParams::new()
            .with_unwrap(true)
            .with_n_fft(crate::nzu!(4096))
            .with_freq_range(0.0, 5000.0);
        assert!(params.unwrap);
        assert_eq!(params.freq_range, Some((0.0, 5000.0)));

        let plot = create_phase_spectrum_plot(&audio, &params).unwrap();
        let html = plot.html().unwrap();
        assert!(!html.is_empty());
    }

    #[test]
    fn test_unwrap_phase_removes_discontinuity() {
        use std::f64::consts::PI;
        // A linearly increasing phase that wraps once: 0, PI/2, PI, then wraps
        // to ~ -PI/2 (which is +3PI/2 unwrapped).
        let mut phases = vec![0.0, PI / 2.0, PI, -PI / 2.0];
        unwrap_phase(&mut phases);
        // After unwrapping the sequence must be monotonically increasing here.
        assert!((phases[3] - 1.5 * PI).abs() < 1e-9, "got {}", phases[3]);
    }
}
