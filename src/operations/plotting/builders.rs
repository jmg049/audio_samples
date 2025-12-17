//! Builder methods for creating plot elements from AudioSamples.
//!
//! This module provides the implementation of plot element builders that
//! integrate with the AudioSamples type and its analysis capabilities.

use super::core::*;
use super::elements::*;

use crate::CastFrom;
use crate::RealFloat;
use crate::operations::MonoConversionMethod;
use crate::operations::traits::AudioChannelOps;
use crate::operations::traits::AudioStatistics;
#[cfg(feature = "spectral-analysis")]
use crate::operations::traits::AudioTransforms;
#[cfg(feature = "spectral-analysis")]
use crate::operations::types::SpectrogramScale;
#[cfg(any(feature = "fft", feature = "spectral-analysis"))]
use crate::operations::types::WindowType;
use crate::{AudioSample, AudioSampleResult, AudioSamples, ConvertTo, I24, to_precision};
#[cfg(feature = "spectral-analysis")]
use ndarray::Array2;
#[cfg(any(feature = "fft", feature = "spectral-analysis"))]
use num_complex::Complex;
#[cfg(feature = "plotting")]
use plotly::Scatter;
use plotly::common::{Line, Mode};

/// Extension trait for AudioSamples to create plot elements
pub trait AudioPlotBuilders<T: AudioSample> {
    /// Create a waveform plot element
    fn waveform_plot<F>(
        &self,
        style: Option<LineStyle<F>>,
    ) -> AudioSampleResult<WaveformPlot<F, T>>
    where
        F: RealFloat + ConvertTo<T>,
        T: CastFrom<F> + ConvertTo<F>;

    /// Create a waveform plot with custom metadata
    fn waveform_plot_with_metadata<F>(
        &self,
        style: LineStyle<F>,
        metadata: PlotMetadata,
    ) -> AudioSampleResult<WaveformPlot<F, T>>
    where
        F: RealFloat + ConvertTo<T>,
        T: CastFrom<F> + ConvertTo<F>;

    /// Create a spectrogram plot element with function parameters
    #[cfg(feature = "spectral-analysis")]
    fn spectrogram_plot<F>(
        &self,
        n_fft: Option<usize>,
        hop_length: Option<usize>,
        window: Option<WindowType<F>>,
        colormap: Option<ColorPalette>,
        db_range: Option<(F, F)>,
        log_freq: Option<bool>,
    ) -> AudioSampleResult<SpectrogramPlot<F>>
    where
        F: RealFloat + ConvertTo<T>,
        T: ConvertTo<F>;

    /// Create a power spectrum plot element with function parameters
    #[cfg(feature = "spectral-analysis")]
    fn power_spectrum_plot<F>(
        &self,
        n_fft: Option<usize>,
        window: Option<WindowType<F>>,
        db_scale: Option<bool>,
        freq_range: Option<(F, F)>,
        style: Option<LineStyle<F>>,
    ) -> AudioSampleResult<PowerSpectrumPlot<F, T>>
    where
        F: RealFloat + ConvertTo<T>,
        T: ConvertTo<F> + CastFrom<F>;

    /// Create a mel spectrogram plot element with function parameters
    #[cfg(feature = "spectral-analysis")]
    fn mel_spectrogram_plot<F>(
        &self,
        n_mels: Option<usize>,
        n_fft: Option<usize>,
        hop_length: Option<usize>,
        window: Option<WindowType<F>>,
        f_min: Option<F>,
        f_max: Option<F>,
        colormap: Option<ColorPalette>,
        db_range: Option<(F, F)>,
    ) -> AudioSampleResult<SpectrogramPlot<F>>
    where
        F: RealFloat + ConvertTo<T>,
        T: ConvertTo<F>;

    /// Create onset markers overlay with function parameters
    fn onset_markers<F>(
        &self,
        marker_style: Option<MarkerStyle<F>>,
        line_style: Option<LineStyle<F>>,
        show_strength: Option<bool>,
        threshold: Option<F>,
    ) -> AudioSampleResult<OnsetMarkers<F>>
    where
        F: RealFloat + ConvertTo<T>,
        T: ConvertTo<F>;

    /// Create beat markers overlay with function parameters
    fn beat_markers<F>(
        &self,
        marker_style: Option<MarkerStyle<F>>,
        line_style: Option<LineStyle<F>>,
        show_tempo: Option<bool>,
    ) -> AudioSampleResult<BeatMarkers<F>>
    where
        F: RealFloat + ConvertTo<T>,
        T: ConvertTo<F>;

    /// Create a pitch contour overlay with function parameters
    fn pitch_contour<F>(
        &self,
        line_style: Option<LineStyle<F>>,
        show_confidence: Option<bool>,
        freq_range: Option<(F, F)>,
        method: Option<PitchDetectionMethod>,
    ) -> AudioSampleResult<PitchContour<F>>
    where
        F: RealFloat + ConvertTo<T>,
        T: ConvertTo<F>;

    // FFT-based plot methods

    /// Create a complex spectrum plot showing both magnitude and phase
    #[cfg(feature = "spectral-analysis")]
    fn complex_spectrum_plot<F>(
        &self,
        config: Option<ComplexSpectrumConfig<F>>,
    ) -> AudioSampleResult<ComplexSpectrumPlot<F>>
    where
        F: RealFloat + ConvertTo<T>,
        T: ConvertTo<F> + CastFrom<F>;

    /// Create a phase-only spectrum plot
    #[cfg(feature = "spectral-analysis")]
    fn phase_spectrum_plot<F>(
        &self,
        style: Option<LineStyle<F>>,
        phase_mode: Option<PhaseDisplayMode>,
    ) -> AudioSampleResult<PhaseSpectrumPlot<F>>
    where
        F: RealFloat + ConvertTo<T>,
        T: ConvertTo<F> + CastFrom<F>;

    /// Create a peak frequency detection plot
    #[cfg(feature = "spectral-analysis")]
    fn peak_frequencies_plot<F>(
        &self,
        peak_config: Option<PeakDetectionConfig<F>>,
        base_style: Option<LineStyle<F>>,
        peak_marker_style: Option<MarkerStyle<F>>,
    ) -> AudioSampleResult<PeakFrequencyPlot<F>>
    where
        F: RealFloat + ConvertTo<T>,
        T: ConvertTo<F> + CastFrom<F>;

    /// Create a frequency bin tracking plot for monitoring specific frequencies over time
    #[cfg(feature = "spectral-analysis")]
    fn frequency_bins_plot<F>(
        &self,
        config: FrequencyBinConfig<F>,
    ) -> AudioSampleResult<FrequencyBinPlot<F>>
    where
        F: RealFloat;

    /// Create a plot tracking the strongest frequency components over time using STFT
    #[cfg(feature = "spectral-analysis")]
    fn spectral_peaks_over_time_plot<F>(
        &self,
        n_peaks: Option<usize>,
        frequency_range: Option<(F, F)>,
        window_size: Option<usize>,
        hop_size: Option<usize>,
    ) -> AudioSampleResult<FrequencyBinPlot<F>>
    where
        F: RealFloat + ConvertTo<T>,
        T: ConvertTo<F>;

    /// Create a group delay plot for filter analysis
    #[cfg(feature = "spectral-analysis")]
    fn group_delay_plot<F>(
        &self,
        style: Option<LineStyle<F>>,
    ) -> AudioSampleResult<GroupDelayPlot<F>>
    where
        F: RealFloat + ConvertTo<T>,
        T: ConvertTo<F> + CastFrom<F>;

    /// Create an FFT waterfall plot for 3D time-frequency visualization
    #[cfg(feature = "spectral-analysis")]
    fn fft_waterfall_plot<F>(
        &self,
        config: Option<WaterfallConfig<F>>,
    ) -> AudioSampleResult<FftWaterfallPlot<F>>
    where
        F: RealFloat + ConvertTo<T>,
        T: ConvertTo<F>;

    /// Create a window function comparison plot
    #[cfg(feature = "spectral-analysis")]
    fn windowed_spectrum_comparison<F>(
        &self,
        config: Option<WindowComparisonConfig<F>>,
    ) -> AudioSampleResult<WindowComparisonPlot<F>>
    where
        F: RealFloat + ConvertTo<T>,
        T: ConvertTo<F>;

    /// Create an instantaneous frequency plot
    #[cfg(feature = "spectral-analysis")]
    fn instantaneous_frequency_plot<F>(
        &self,
        style: Option<LineStyle<F>>,
    ) -> AudioSampleResult<InstantaneousFrequencyPlot<F>>
    where
        F: RealFloat + ConvertTo<T>,
        T: ConvertTo<F>;
}

// Config structs removed - using function parameters with defaults instead

/// Pitch detection methods
#[derive(Debug, Clone)]
pub enum PitchDetectionMethod {
    /// Autocorrelation-based pitch detection
    Autocorrelation,
    /// YIN algorithm for pitch detection
    Yin,
    /// Cepstrum-based pitch detection
    Cepstrum,
}

/// Pitch contour plot element
#[derive(Debug, Clone)]
pub struct PitchContour<F: RealFloat> {
    time_axis: Vec<F>,
    pitch_values: Vec<F>, // in Hz
    confidence: Option<Vec<F>>,
    line_style: LineStyle<F>,
    show_confidence: bool,
    freq_range: (F, F),
    bounds: PlotBounds<F>,
    metadata: PlotMetadata,
}

impl<F: RealFloat> PitchContour<F> {
    /// Creates a new pitch contour plot element.
    ///
    /// # Arguments
    /// * `time_axis` - Vector of time values in seconds
    /// * `pitch_values` - Vector of pitch values in Hz
    /// * `confidence` - Optional vector of confidence values [0.0, 1.0]
    /// * `line_style` - Line style configuration for the plot
    /// * `show_confidence` - Whether to display confidence indicators
    /// * `freq_range` - Frequency range for the plot (min, max) in Hz
    ///
    /// # Returns
    /// A new `PitchContour` instance ready for rendering
    pub fn new(
        time_axis: Vec<F>,
        pitch_values: Vec<F>,
        confidence: Option<Vec<F>>,
        line_style: LineStyle<F>,
        show_confidence: bool,
        freq_range: (F, F),
    ) -> Self {
        let x_min = time_axis.first().copied().unwrap_or_default();
        let x_max = time_axis.last().copied().unwrap_or_default();
        let y_min = freq_range.0;
        let y_max = freq_range.1;

        let bounds = PlotBounds::new(x_min, x_max, y_min, y_max);

        let metadata = PlotMetadata {
            legend_label: Some("Pitch".to_string()),
            y_label: Some("Frequency (Hz)".to_string()),
            z_order: 8,
            ..Default::default()
        };

        Self {
            time_axis,
            pitch_values,
            confidence,
            line_style,
            show_confidence,
            freq_range,
            bounds,
            metadata,
        }
    }
}

impl<F: RealFloat> PlotElement<F> for PitchContour<F> {
    fn data_bounds(&self) -> PlotBounds<F> {
        self.bounds
    }

    fn metadata(&self) -> &PlotMetadata {
        &self.metadata
    }

    fn to_plotly_traces(&self) -> Vec<PlotTrace<F>> {
        let mut traces = Vec::new();

        // Filter out invalid pitch values and apply frequency range constraints
        let (x_data, y_data): (Vec<F>, Vec<F>) = self
            .time_axis
            .iter()
            .zip(self.pitch_values.iter())
            .filter(|&(_, pitch)| {
                *pitch > F::zero() && *pitch >= self.freq_range.0 && *pitch <= self.freq_range.1
            })
            .map(|(&t, &pitch)| (t, pitch))
            .unzip();

        if !x_data.is_empty() {
            let mut trace = Scatter::new(x_data, y_data)
                .mode(Mode::Lines)
                .line(self.line_style.to_plotly_line());

            if let Some(ref name) = self.metadata.legend_label {
                trace = trace.name(name);
            }

            traces.push(PlotTrace::Scatter(Box::new(*trace)));
        }

        // If show_confidence is enabled and confidence data is available, add confidence visualization
        if self.show_confidence
            && let Some(ref confidence_data) = self.confidence
        {
            let (conf_x_data, conf_y_data): (Vec<F>, Vec<F>) = self
                .time_axis
                .iter()
                .zip(confidence_data.iter())
                .map(|(&t, &conf)| (t, conf))
                .unzip();

            if !conf_x_data.is_empty() {
                let confidence_trace = Scatter::new(conf_x_data, conf_y_data)
                    .mode(Mode::Lines)
                    .line(
                        Line::new()
                            .color(format!("{}80", self.line_style.color)) // Semi-transparent version
                            .width(
                                (self.line_style.width * to_precision::<F, _>(0.5))
                                    .to_f64()
                                    .expect("Should not fail"),
                            ),
                    )
                    .name("Pitch Confidence");

                traces.push(PlotTrace::Scatter(Box::new(*confidence_trace)));
            }
        }

        traces
    }
}

// Implementation with spectral analysis support
#[cfg(feature = "spectral-analysis")]
impl<'a, T: AudioSample> AudioPlotBuilders<T> for AudioSamples<'a, T>
where
    i16: ConvertTo<T>,
    I24: ConvertTo<T>,
    i32: ConvertTo<T>,
    f32: ConvertTo<T>,
    f64: ConvertTo<T>,
    T: ConvertTo<f64>,
    f64: CastFrom<T>,
    for<'b> AudioSamples<'b, T>: AudioStatistics<'b, T> + AudioTransforms<T>,
{
    // todo add automatic support for handling multi-channel audio by plotting per-channel waveforms stacked and also have an option to mixdown to mono before plotting (default for now)
    fn waveform_plot<F>(&self, style: Option<LineStyle<F>>) -> AudioSampleResult<WaveformPlot<F, T>>
    where
        T: CastFrom<F> + ConvertTo<F>,
        F: RealFloat + ConvertTo<T>,
    {
        let sample_rate = to_precision::<F, _>(self.sample_rate.get());
        let samples_per_channel = self.samples_per_channel();

        // Generate time axis
        let time_data: Vec<F> = (0..samples_per_channel)
            .map(|i| to_precision::<F, _>(i) / sample_rate)
            .collect();

        let audio_samples = self.to_mono(MonoConversionMethod::<F>::Average).expect("");

        // Extract amplitude data preserving native sample ranges
        // Instead of normalizing to ±1.0, preserve the actual sample values
        let amplitude_data: Vec<F> = audio_samples
            .map_into(|x| to_precision::<F, _>(x))
            .to_interleaved_vec();
        let style = style.unwrap_or_default();
        let metadata = PlotMetadata {
            x_label: Some("Time (s)".to_string()),
            y_label: Some(format!("Amplitude ({})", T::LABEL)),
            ..Default::default()
        };

        Ok(WaveformPlot::new(
            time_data,
            amplitude_data,
            style,
            metadata,
        ))
    }

    fn waveform_plot_with_metadata<F>(
        &self,
        style: LineStyle<F>,
        metadata: PlotMetadata,
    ) -> AudioSampleResult<WaveformPlot<F, T>>
    where
        T: CastFrom<F> + ConvertTo<F>,
        F: RealFloat + ConvertTo<T>,
    {
        let sample_rate = to_precision::<F, _>(self.sample_rate.get());
        let samples_per_channel = self.samples_per_channel();

        let time_data: Vec<F> = (0..samples_per_channel)
            .map(|i| to_precision::<F, _>(i) / sample_rate)
            .collect();

        let audio_samples = self
            .to_mono(MonoConversionMethod::<F>::Average)?
            .map_into(|x| to_precision::<F, _>(x));
        let amplitude_data: Vec<F> = audio_samples.to_interleaved_vec();

        Ok(WaveformPlot::new(
            time_data,
            amplitude_data,
            style,
            metadata,
        ))
    }

    #[cfg(feature = "spectral-analysis")]
    fn spectrogram_plot<F>(
        &self,
        n_fft: Option<usize>,
        hop_length: Option<usize>,
        window: Option<WindowType<F>>,
        colormap: Option<ColorPalette>,
        db_range: Option<(F, F)>,
        log_freq: Option<bool>,
    ) -> AudioSampleResult<SpectrogramPlot<F>>
    where
        F: RealFloat + ConvertTo<T>,
        T: ConvertTo<F>,
    {
        // Apply defaults
        let n_fft = n_fft.unwrap_or(2048);
        let hop_length = hop_length.unwrap_or(n_fft / 4);
        let window = window.unwrap_or(WindowType::Hanning);
        let colormap = colormap.unwrap_or(ColorPalette::Viridis);
        let db_range = db_range.unwrap_or((to_precision::<F, _>(-80.0), F::zero()));
        let _log_freq = log_freq.unwrap_or(false);

        // Convert to magnitude spectrogram in dB
        let spectrogram_data =
            self.spectrogram(n_fft, hop_length, window, SpectrogramScale::Log, true)?;
        let (rows, cols) = spectrogram_data.dim();

        // Generate time and frequency axes
        let sample_rate = to_precision::<F, _>(self.sample_rate.get());
        let time_axis: Vec<F> = (0..cols)
            .map(|i| to_precision::<F, _>(i) * to_precision::<F, _>(hop_length) / sample_rate)
            .collect();

        let freq_axis: Vec<F> = (0..rows)
            .map(|i| to_precision::<F, _>(i) * sample_rate / to_precision::<F, _>(n_fft))
            .collect();

        let metadata = PlotMetadata {
            x_label: Some("Time (s)".to_string()),
            y_label: Some("Frequency (Hz)".to_string()),
            title: Some("Spectrogram".to_string()),
            ..Default::default()
        };

        // Create a temporary config for the existing SpectrogramPlot::new signature
        let config = SpectrogramConfig {
            n_fft,
            window_size: Some(n_fft),
            hop_length: Some(hop_length),
            window,
            colormap,
            db_range,
            log_freq: _log_freq,
            mel_scale: false,
        };

        Ok(SpectrogramPlot::new(
            spectrogram_data,
            time_axis,
            freq_axis,
            config,
            metadata,
        ))
    }

    #[cfg(feature = "spectral-analysis")]
    fn power_spectrum_plot<F>(
        &self,
        n_fft: Option<usize>,
        window: Option<WindowType<F>>,
        db_scale: Option<bool>,
        freq_range: Option<(F, F)>,
        style: Option<LineStyle<F>>,
    ) -> AudioSampleResult<PowerSpectrumPlot<F, T>>
    where
        F: RealFloat + ConvertTo<T>,
        T: ConvertTo<F> + CastFrom<F>,
    {
        // Apply defaults
        let _n_fft = n_fft.unwrap_or(2048);
        let _window = window.unwrap_or(WindowType::Hanning);
        let db_scale = db_scale.unwrap_or(true);
        let style = style.unwrap_or_else(|| LineStyle {
            color: "#1f77b4".to_string(),
            width: to_precision::<F, _>(3.0),
            style: LineStyleType::Solid,
        });

        // Convert to mono for single plot output (use Plotting trait for per-channel)
        let mono = self.to_mono(MonoConversionMethod::<F>::Average)?;

        // Compute FFT - returns Array2 with shape (1, n_fft) for mono
        let fft_result: Array2<Complex<F>> = mono.fft()?;
        let sample_rate = to_precision::<F, _>(self.sample_rate.get());
        let fft_len = fft_result.ncols();
        let n_bins = fft_len / 2; // Only positive frequencies

        // Generate frequency axis
        let mut frequencies: Vec<F> = (0..n_bins)
            .map(|i| to_precision::<F, _>(i) * sample_rate / to_precision::<F, _>(fft_len))
            .collect();

        // Compute magnitudes from the first (only) row
        let mut magnitudes: Vec<F> = (0..n_bins)
            .map(|i| {
                let complex_val = fft_result[[0, i]];
                let magnitude: F = complex_val.norm();
                if db_scale {
                    if magnitude > F::zero() {
                        to_precision::<F, _>(20.0) * magnitude.log10()
                    } else {
                        to_precision::<F, _>(-120.0)
                    }
                } else {
                    magnitude
                }
            })
            .collect();

        // Apply frequency range filtering if specified
        if let Some((f_min, f_max)) = freq_range {
            let filtered: Vec<(F, F)> = frequencies
                .into_iter()
                .zip(magnitudes)
                .filter(|(f, _)| *f >= f_min && *f <= f_max)
                .collect();
            let (f, m): (Vec<F>, Vec<F>) = filtered.into_iter().unzip();
            frequencies = f;
            magnitudes = m;
        }

        let metadata = PlotMetadata {
            x_label: Some("Frequency (Hz)".to_string()),
            y_label: Some(if db_scale {
                "Magnitude (dB)".to_string()
            } else {
                "Magnitude".to_string()
            }),
            title: Some("Power Spectrum".to_string()),
            ..Default::default()
        };

        Ok(PowerSpectrumPlot::new(
            frequencies,
            magnitudes,
            style,
            metadata,
        ))
    }

    #[cfg(feature = "spectral-analysis")]
    fn mel_spectrogram_plot<F>(
        &self,
        n_mels: Option<usize>,
        n_fft: Option<usize>,
        hop_length: Option<usize>,
        window: Option<WindowType<F>>,
        f_min: Option<F>,
        f_max: Option<F>,
        colormap: Option<ColorPalette>,
        db_range: Option<(F, F)>,
    ) -> AudioSampleResult<SpectrogramPlot<F>>
    where
        F: RealFloat + ConvertTo<T>,
        T: ConvertTo<F>,
    {
        // Apply defaults
        let n_mels = n_mels.unwrap_or(128);
        let n_fft = n_fft.unwrap_or(2048);
        let hop_length = hop_length.unwrap_or(n_fft / 4);
        let window = window.unwrap_or(WindowType::Hanning);
        let f_min = f_min.unwrap_or_default();
        let colormap = colormap.unwrap_or(ColorPalette::Viridis);
        let db_range = db_range.unwrap_or((to_precision::<F, _>(-80.0), F::zero()));

        // Use the AudioTransforms trait method
        let mel_spec_data = self.mel_spectrogram(
            n_mels,
            f_min,
            f_max.unwrap_or(
                to_precision::<F, _>(self.sample_rate.get()) / to_precision::<F, _>(2.0),
            ), // Nyquist frequency as default
            n_fft,
            hop_length,
        )?;

        let (rows, cols) = mel_spec_data.dim();

        // Generate time and mel frequency axes
        let sample_rate = to_precision::<F, _>(self.sample_rate.get());
        let time_axis: Vec<F> = (0..cols)
            .map(|i| to_precision::<F, _>(i) * to_precision::<F, _>(hop_length) / sample_rate)
            .collect();

        let freq_axis: Vec<F> = (0..rows)
            .map(|i| to_precision::<F, _>(i)) // Mel bins are numbered, not frequency
            .collect();

        let metadata = PlotMetadata {
            x_label: Some("Time (s)".to_string()),
            y_label: Some("Mel Frequency".to_string()),
            title: Some("Mel Spectrogram".to_string()),
            ..Default::default()
        };

        // Create a temporary config for the existing SpectrogramPlot::new signature
        let config = SpectrogramConfig {
            n_fft,
            window_size: Some(n_fft),
            hop_length: Some(hop_length),
            window,
            colormap,
            db_range,
            log_freq: false,
            mel_scale: true,
        };

        Ok(SpectrogramPlot::new(
            mel_spec_data,
            time_axis,
            freq_axis,
            config,
            metadata,
        ))
    }

    fn onset_markers<F: RealFloat>(
        &self,
        marker_style: Option<MarkerStyle<F>>,
        line_style: Option<LineStyle<F>>,
        show_strength: Option<bool>,
        threshold: Option<F>,
    ) -> AudioSampleResult<OnsetMarkers<F>>
    where
        F: ConvertTo<T>,
        T: ConvertTo<F>,
    {
        // Apply defaults
        let marker_style = marker_style.unwrap_or_else(|| MarkerStyle {
            color: "#d62728".to_string(), // Red
            size: to_precision::<F, _>(8.0),
            shape: MarkerShape::Triangle,
            fill: true,
        });
        let line_style = line_style.or_else(|| {
            Some(LineStyle {
                color: "#d62728".to_string(), // Red
                width: F::one(),
                style: LineStyleType::Dashed,
            })
        });
        let show_strength = show_strength.unwrap_or(true);
        let detection_threshold = threshold.unwrap_or(to_precision::<F, _>(0.1));

        let peak = self.peak();
        let peak: F = to_precision::<F, _>(peak);
        let y_range = (-peak, peak);

        // Detect onsets using the spectral-analysis onset detection module
        #[cfg(feature = "spectral-analysis")]
        let (onset_times, onset_strengths) = {
            use crate::operations::types::OnsetConfig as DetectionOnsetConfig;

            let mut detection_config = DetectionOnsetConfig::<F>::default();
            detection_config
                .peak_picking
                .adaptive_threshold
                .min_threshold = detection_threshold;

            // Get onset times
            let times = self.detect_onsets(&detection_config).unwrap_or_default();

            // Get onset strengths if requested
            let strengths = if show_strength {
                self.onset_detection_function(&detection_config)
                    .ok()
                    .map(|(_, odf)| odf)
            } else {
                None
            };

            (times, strengths)
        };

        #[cfg(not(feature = "spectral-analysis"))]
        let (onset_times, onset_strengths): (Vec<F>, Option<Vec<F>>) = (Vec::new(), None);

        // Create a temporary config for the existing OnsetMarkers::new signature
        let config = OnsetConfig {
            marker_style,
            line_style,
            show_strength,
            threshold: detection_threshold,
        };

        Ok(OnsetMarkers::new(
            onset_times,
            onset_strengths,
            config,
            y_range,
        ))
    }

    fn beat_markers<F: RealFloat>(
        &self,
        marker_style: Option<MarkerStyle<F>>,
        line_style: Option<LineStyle<F>>,
        show_tempo: Option<bool>,
    ) -> AudioSampleResult<BeatMarkers<F>>
    where
        F: ConvertTo<T>,
        T: ConvertTo<F>,
    {
        // Apply defaults
        let marker_style = marker_style.unwrap_or_else(|| MarkerStyle {
            color: "#2ca02c".to_string(), // Green
            size: to_precision::<F, _>(10.0),
            shape: MarkerShape::Diamond,
            fill: true,
        });
        let line_style = line_style.or_else(|| {
            Some(LineStyle {
                color: "#2ca02c".to_string(), // Green
                width: to_precision::<F, _>(2.0),
                style: LineStyleType::Solid,
            })
        });
        let show_tempo = show_tempo.unwrap_or(true);

        let peak = self.peak();
        let peak: F = to_precision::<F, _>(peak);
        let y_range = (-peak, peak);

        // Detect beats using the beat-detection module
        #[cfg(feature = "beat-detection")]
        let (beat_times, tempo) = {
            use crate::operations::beats::BeatConfig as DetectionBeatConfig;
            use crate::operations::types::OnsetConfig as DetectionOnsetConfig;

            // Use default onset config and a reasonable default tempo estimate
            let onset_config = DetectionOnsetConfig::<F>::default();
            let detection_config = DetectionBeatConfig {
                tempo_bpm: to_precision::<F, _>(120.0), // Default tempo estimate
                onset_config,
                tolerance: Some(to_precision::<F, _>(0.04)),
            };

            match self.detect_beats(&detection_config, None) {
                Ok(tracker) => (tracker.beat_times, Some(tracker.tempo_bpm)),
                Err(_) => (Vec::new(), None),
            }
        };

        #[cfg(not(feature = "beat-detection"))]
        let (beat_times, tempo): (Vec<F>, Option<F>) = (Vec::new(), None);

        // Create a temporary config for the existing BeatMarkers::new signature
        let config = BeatPlotConfig {
            marker_style,
            line_style,
            show_tempo,
        };

        Ok(BeatMarkers::new(beat_times, tempo, config, y_range))
    }

    fn pitch_contour<F: RealFloat>(
        &self,
        line_style: Option<LineStyle<F>>,
        show_confidence: Option<bool>,
        freq_range: Option<(F, F)>,
        method: Option<PitchDetectionMethod>,
    ) -> AudioSampleResult<PitchContour<F>>
    where
        F: ConvertTo<T>,
        T: ConvertTo<F>,
    {
        // Apply defaults
        let line_style = line_style.unwrap_or_else(|| LineStyle {
            color: "#ff00ff".to_string(), // Magenta
            width: to_precision::<F, _>(2.0),
            style: LineStyleType::Solid,
        });
        let show_confidence = show_confidence.unwrap_or(false);
        let freq_range =
            freq_range.unwrap_or((to_precision::<F, _>(80.0), to_precision::<F, _>(2000.0)));
        let method = method.unwrap_or(PitchDetectionMethod::Autocorrelation);

        // Detect pitch using the spectral-analysis pitch detection module
        #[cfg(feature = "spectral-analysis")]
        let (time_axis, pitch_values, confidence) = {
            use crate::operations::traits::AudioPitchAnalysis;
            use crate::operations::types::PitchDetectionMethod as TypesPitchMethod;

            // Convert plotting enum to types enum
            let types_method = match method {
                PitchDetectionMethod::Autocorrelation => TypesPitchMethod::Autocorrelation,
                PitchDetectionMethod::Yin => TypesPitchMethod::Yin,
                PitchDetectionMethod::Cepstrum => TypesPitchMethod::Cepstrum,
            };

            // Use window-based pitch tracking
            let window_size = 2048;
            let hop_size = 512;
            let threshold = to_precision::<F, _>(0.1); // YIN threshold

            match self.track_pitch(
                window_size,
                hop_size,
                types_method,
                threshold,
                freq_range.0,
                freq_range.1,
            ) {
                Ok(results) => {
                    let times: Vec<F> = results.iter().map(|(t, _)| *t).collect();
                    let pitches: Vec<F> = results
                        .iter()
                        .map(|(_, p)| p.unwrap_or(F::zero()))
                        .collect();
                    // For now, no confidence values available from pitch tracking
                    (times, pitches, None::<Vec<F>>)
                }
                Err(_) => (Vec::new(), Vec::new(), None),
            }
        };

        #[cfg(not(feature = "spectral-analysis"))]
        let (time_axis, pitch_values, confidence): (Vec<F>, Vec<F>, Option<Vec<F>>) = {
            let _ = method; // Suppress unused variable warning
            (Vec::new(), Vec::new(), None)
        };

        Ok(PitchContour::new(
            time_axis,
            pitch_values,
            confidence,
            line_style,
            show_confidence,
            freq_range,
        ))
    }

    #[cfg(feature = "fft")]
    fn complex_spectrum_plot<F>(
        &self,
        config: Option<ComplexSpectrumConfig<F>>,
    ) -> AudioSampleResult<ComplexSpectrumPlot<F>>
    where
        F: RealFloat + ConvertTo<T>,
        T: ConvertTo<F> + CastFrom<F>,
    {
        let config = config.unwrap_or_default();

        // Convert to mono for single plot output
        let mono = self.to_mono(MonoConversionMethod::<F>::Average)?;

        // Compute FFT - returns Array2 with shape (1, n_fft) for mono
        let fft_result: Array2<Complex<F>> = mono.fft()?;
        let sample_rate = to_precision::<F, _>(self.sample_rate.get());
        let fft_len = fft_result.ncols();
        let n_bins = fft_len / 2; // Only positive frequencies

        // Generate frequency axis
        let mut frequencies: Vec<F> = (0..n_bins)
            .map(|i| to_precision::<F, _>(i) * sample_rate / to_precision::<F, _>(fft_len))
            .collect();

        // Take only positive frequencies from first row
        let mut complex_values: Vec<Complex<F>> = (0..n_bins).map(|i| fft_result[[0, i]]).collect();

        // Apply frequency range filtering if specified
        if let Some((f_min, f_max)) = config.frequency_range {
            let filtered: Vec<(F, Complex<F>)> = frequencies
                .into_iter()
                .zip(complex_values)
                .filter(|(f, _)| *f >= f_min && *f <= f_max)
                .collect();
            let (f, c): (Vec<F>, Vec<Complex<F>>) = filtered.into_iter().unzip();
            frequencies = f;
            complex_values = c;
        }

        let metadata = PlotMetadata {
            x_label: Some("Frequency (Hz)".to_string()),
            y_label: Some(if config.db_scale {
                "Magnitude (dB)".to_string()
            } else {
                "Magnitude".to_string()
            }),
            title: Some("Complex Spectrum".to_string()),
            ..Default::default()
        };

        Ok(ComplexSpectrumPlot::new(
            frequencies,
            complex_values,
            config,
            metadata,
        ))
    }

    #[cfg(feature = "fft")]
    fn phase_spectrum_plot<F>(
        &self,
        style: Option<LineStyle<F>>,
        phase_mode: Option<PhaseDisplayMode>,
    ) -> AudioSampleResult<PhaseSpectrumPlot<F>>
    where
        F: RealFloat + ConvertTo<T>,
        T: ConvertTo<F> + CastFrom<F>,
    {
        let style = style.unwrap_or_else(|| LineStyle {
            color: "#ff7f0e".to_string(), // Orange
            width: to_precision::<F, _>(2.0),
            style: LineStyleType::Solid,
        });
        let phase_mode = phase_mode.unwrap_or_default();

        // Convert to mono for single plot output
        let mono = self.to_mono(MonoConversionMethod::<F>::Average)?;

        // Compute FFT - returns Array2 with shape (1, n_fft) for mono
        let fft_result: Array2<Complex<F>> = mono.fft()?;
        let sample_rate = to_precision::<F, _>(self.sample_rate.get());
        let fft_len = fft_result.ncols();
        let n_bins = fft_len / 2; // Only positive frequencies

        // Generate frequency axis
        let frequencies: Vec<F> = (0..n_bins)
            .map(|i| to_precision::<F, _>(i) * sample_rate / to_precision::<F, _>(fft_len))
            .collect();

        // Take only positive frequencies from first row
        let complex_values: Vec<Complex<F>> = (0..n_bins).map(|i| fft_result[[0, i]]).collect();

        let metadata = PlotMetadata {
            x_label: Some("Frequency (Hz)".to_string()),
            y_label: Some(match phase_mode {
                PhaseDisplayMode::Degrees | PhaseDisplayMode::UnwrappedDegrees => {
                    "Phase (degrees)".to_string()
                }
                _ => "Phase (radians)".to_string(),
            }),
            title: Some("Phase Spectrum".to_string()),
            legend_label: Some("Phase".to_string()),
            ..Default::default()
        };

        Ok(PhaseSpectrumPlot::new(
            frequencies,
            complex_values,
            style,
            phase_mode,
            metadata,
        ))
    }

    #[cfg(feature = "fft")]
    fn peak_frequencies_plot<F>(
        &self,
        peak_config: Option<PeakDetectionConfig<F>>,
        base_style: Option<LineStyle<F>>,
        peak_marker_style: Option<MarkerStyle<F>>,
    ) -> AudioSampleResult<PeakFrequencyPlot<F>>
    where
        F: RealFloat + ConvertTo<T>,
        T: ConvertTo<F> + CastFrom<F>,
    {
        let peak_config = peak_config.unwrap_or_default();
        let base_style = base_style.unwrap_or_default();
        let peak_marker_style = peak_marker_style.unwrap_or_else(|| MarkerStyle {
            color: "#d62728".to_string(), // Red
            size: to_precision::<F, _>(10.0),
            shape: MarkerShape::Circle,
            fill: true,
        });

        // Convert to mono for single plot output
        let mono = self.to_mono(MonoConversionMethod::<F>::Average)?;

        // Compute FFT - returns Array2 with shape (1, n_fft) for mono
        let fft_result: Array2<Complex<F>> = mono.fft()?;
        let sample_rate = to_precision::<F, _>(self.sample_rate.get());
        let fft_len = fft_result.ncols();
        let n_bins = fft_len / 2; // Only positive frequencies

        // Generate frequency axis
        let frequencies: Vec<F> = (0..n_bins)
            .map(|i| to_precision::<F, _>(i) * sample_rate / to_precision::<F, _>(fft_len))
            .collect();

        // Compute magnitudes in dB from first row
        let magnitudes: Vec<F> = (0..n_bins)
            .map(|i| {
                let complex_val = fft_result[[0, i]];
                let magnitude: F = complex_val.norm();
                if magnitude > F::zero() {
                    to_precision::<F, _>(20.0) * magnitude.log10()
                } else {
                    to_precision::<F, _>(-120.0)
                }
            })
            .collect();

        // Apply frequency range filtering if specified
        let (filtered_frequencies, filtered_magnitudes) =
            if let Some((f_min, f_max)) = peak_config.frequency_range {
                let filtered: Vec<(F, F)> = frequencies
                    .iter()
                    .zip(magnitudes.iter())
                    .filter(|(f, _)| **f >= f_min && **f <= f_max)
                    .map(|(&f, &m)| (f, m))
                    .collect();
                let (f, m): (Vec<F>, Vec<F>) = filtered.into_iter().unzip();
                (f, m)
            } else {
                (frequencies.clone(), magnitudes.clone())
            };

        // Simple peak detection: find local maxima above threshold
        let mut peak_frequencies = Vec::new();
        let mut peak_magnitudes = Vec::new();
        let min_height = peak_config
            .min_height
            .unwrap_or(to_precision::<F, _>(-60.0));
        let min_prominence = peak_config
            .min_prominence
            .unwrap_or(to_precision::<F, _>(3.0));

        for i in 1..filtered_magnitudes.len().saturating_sub(1) {
            let prev = filtered_magnitudes[i - 1];
            let curr = filtered_magnitudes[i];
            let next = filtered_magnitudes[i + 1];

            // Check if it's a local maximum and above threshold
            if curr > prev
                && curr > next
                && curr >= min_height
                && (curr - prev.min(next)) >= min_prominence
            {
                peak_frequencies.push(filtered_frequencies[i]);
                peak_magnitudes.push(curr);
            }
        }

        // Limit number of peaks if specified
        if let Some(max_peaks) = peak_config.max_peaks
            && peak_frequencies.len() > max_peaks
        {
            // Sort by magnitude (descending) and take top max_peaks
            let mut indexed: Vec<(F, F)> =
                peak_frequencies.into_iter().zip(peak_magnitudes).collect();
            indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            indexed.truncate(max_peaks);
            let (f, m): (Vec<F>, Vec<F>) = indexed.into_iter().unzip();
            peak_frequencies = f;
            peak_magnitudes = m;
        }

        let metadata = PlotMetadata {
            x_label: Some("Frequency (Hz)".to_string()),
            y_label: Some("Magnitude (dB)".to_string()),
            title: Some("Peak Frequency Detection".to_string()),
            ..Default::default()
        };

        Ok(PeakFrequencyPlot::new(
            frequencies,
            magnitudes,
            peak_frequencies,
            peak_magnitudes,
            base_style,
            peak_marker_style,
            metadata,
        ))
    }

    #[cfg(feature = "spectral-analysis")]
    fn frequency_bins_plot<F: RealFloat>(
        &self,
        config: FrequencyBinConfig<F>,
    ) -> AudioSampleResult<FrequencyBinPlot<F>> {
        // TODO: Implement STFT-based frequency bin tracking
        // This would require computing overlapping FFTs over time windows

        let sample_rate = to_precision::<F, _>(self.sample_rate.get());
        let samples_per_channel = self.samples_per_channel();

        // Generate time axis based on hop length
        let n_windows = if samples_per_channel > config.window_size {
            (samples_per_channel - config.window_size) / config.hop_length + 1
        } else {
            1 // Minimum one window
        };
        let time_axis: Vec<F> = (0..n_windows)
            .map(|i| {
                to_precision::<F, _>(i) * to_precision::<F, _>(config.hop_length) / sample_rate
            })
            .collect();

        // Placeholder: create empty frequency data for each target frequency
        let frequency_data: Vec<Vec<F>> = config
            .target_frequencies
            .iter()
            .map(|_| vec![F::zero(); time_axis.len()])
            .collect();

        let metadata = PlotMetadata {
            x_label: Some("Time (s)".to_string()),
            y_label: Some(
                if config.show_magnitude {
                    "Magnitude (dB)"
                } else {
                    "Phase"
                }
                .to_string(),
            ),
            title: Some("Frequency Bin Tracking".to_string()),
            ..Default::default()
        };

        Ok(FrequencyBinPlot::new(
            time_axis,
            frequency_data,
            config.target_frequencies,
            config.line_styles,
            metadata,
        ))
    }

    #[cfg(feature = "spectral-analysis")]
    fn spectral_peaks_over_time_plot<F>(
        &self,
        n_peaks: Option<usize>,
        frequency_range: Option<(F, F)>,
        window_size: Option<usize>,
        hop_size: Option<usize>,
    ) -> AudioSampleResult<FrequencyBinPlot<F>>
    where
        F: RealFloat + ConvertTo<T>,
        T: ConvertTo<F>,
    {
        let n_peaks = n_peaks.unwrap_or(5);
        let window_size = window_size.unwrap_or(2048);
        let hop_size = hop_size.unwrap_or(window_size / 4);

        // Use existing STFT implementation
        let stft_data = self.stft(window_size, hop_size, WindowType::Hanning)?;
        let (n_freq_bins, n_time_frames) = stft_data.dim();

        let sample_rate = to_precision::<F, _>(self.sample_rate.get());

        // Generate time and frequency axes
        let time_axis: Vec<F> = (0..n_time_frames)
            .map(|i| to_precision::<F, _>(i) * to_precision::<F, _>(hop_size) / sample_rate)
            .collect();

        let frequencies: Vec<F> = (0..n_freq_bins)
            .map(|i| to_precision::<F, _>(i) * sample_rate / to_precision::<F, _>(window_size))
            .collect();

        // For each time frame, find the top N frequency peaks
        let mut peak_frequencies: Vec<F> = Vec::new();
        let mut magnitude_data: Vec<Vec<F>> = vec![Vec::new(); n_peaks];

        for time_idx in 0..n_time_frames {
            // Get magnitude spectrum for this time frame
            let magnitudes: Vec<F> = (0..n_freq_bins)
                .map(|freq_idx| {
                    let complex_val = stft_data[[freq_idx, time_idx]];
                    let mag: F = complex_val.norm();
                    if mag > F::zero() {
                        to_precision::<F, _>(20.0) * mag.log10()
                    } else {
                        to_precision::<F, _>(-120.0)
                    }
                })
                .collect();

            // Apply frequency range filter if specified
            let filtered_data: Vec<(F, F)> = frequencies
                .iter()
                .zip(magnitudes.iter())
                .enumerate()
                .filter(|&(_, (&freq, _))| {
                    if let Some((f_min, f_max)) = frequency_range {
                        freq >= f_min && freq <= f_max
                    } else {
                        true
                    }
                })
                .map(|(_, (&freq, &mag))| (freq, mag))
                .collect();

            // Sort by magnitude and take top N
            let mut sorted_peaks = filtered_data;
            sorted_peaks.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

            if time_idx == 0 {
                // Initialize peak frequencies from first frame
                peak_frequencies = sorted_peaks
                    .iter()
                    .take(n_peaks)
                    .map(|(freq, _)| *freq)
                    .collect();
            }

            // Store magnitudes for each peak frequency
            for (peak_idx, &target_freq) in peak_frequencies.iter().enumerate() {
                if peak_idx >= n_peaks {
                    break;
                }

                // Find magnitude at or near this frequency
                let magnitude = sorted_peaks
                    .iter()
                    .min_by(|a, b| {
                        let diff_a = (a.0 - target_freq).abs();
                        let diff_b = (b.0 - target_freq).abs();
                        diff_a
                            .partial_cmp(&diff_b)
                            .unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .map(|(_, mag)| *mag)
                    .unwrap_or(to_precision::<F, _>(-120.0));

                magnitude_data[peak_idx].push(magnitude);
            }
        }

        // Generate line styles
        let line_styles: Vec<LineStyle<F>> = (0..n_peaks)
            .map(|i| {
                let palette = ColorPalette::Default;
                LineStyle {
                    color: palette.get_color(i),
                    width: to_precision::<F, _>(2.0),
                    style: LineStyleType::Solid,
                }
            })
            .collect();

        let metadata = PlotMetadata {
            x_label: Some("Time (s)".to_string()),
            y_label: Some("Magnitude (dB)".to_string()),
            title: Some("Spectral Peaks Over Time".to_string()),
            ..Default::default()
        };

        Ok(FrequencyBinPlot::new(
            time_axis,
            magnitude_data,
            peak_frequencies,
            line_styles,
            metadata,
        ))
    }

    #[cfg(feature = "fft")]
    fn group_delay_plot<F>(
        &self,
        style: Option<LineStyle<F>>,
    ) -> AudioSampleResult<GroupDelayPlot<F>>
    where
        F: RealFloat + ConvertTo<T>,
        T: ConvertTo<F> + CastFrom<F>,
    {
        let style = style.unwrap_or_else(|| LineStyle {
            color: "#9467bd".to_string(), // Purple
            width: to_precision::<F, _>(2.0),
            style: LineStyleType::Solid,
        });

        // Convert to mono for single plot output
        let mono = self.to_mono(MonoConversionMethod::<F>::Average)?;

        // Compute FFT - returns Array2 with shape (1, n_fft) for mono
        let fft_result: Array2<Complex<F>> = mono.fft()?;
        let sample_rate = to_precision::<F, _>(self.sample_rate.get());
        let fft_len = fft_result.ncols();
        let n_bins = fft_len / 2; // Only positive frequencies

        // Generate frequency axis
        let frequencies: Vec<F> = (0..n_bins)
            .map(|i| to_precision::<F, _>(i) * sample_rate / to_precision::<F, _>(fft_len))
            .collect();

        // Extract positive frequency complex values from first row
        let complex_values: Vec<Complex<F>> = (0..n_bins).map(|i| fft_result[[0, i]]).collect();

        let metadata = PlotMetadata {
            x_label: Some("Frequency (Hz)".to_string()),
            y_label: Some("Group Delay (samples)".to_string()),
            title: Some("Group Delay".to_string()),
            legend_label: Some("Group Delay".to_string()),
            ..Default::default()
        };

        Ok(GroupDelayPlot::new(
            frequencies,
            complex_values,
            style,
            metadata,
        ))
    }

    #[cfg(feature = "spectral-analysis")]
    fn fft_waterfall_plot<F: RealFloat>(
        &self,
        config: Option<WaterfallConfig<F>>,
    ) -> AudioSampleResult<FftWaterfallPlot<F>> {
        let config = config.unwrap_or_default();

        // TODO: Implement STFT-based waterfall computation
        // This would require computing overlapping FFTs over time windows

        let sample_rate = to_precision::<F, _>(self.sample_rate.get());
        let samples_per_channel = self.samples_per_channel();

        // Generate axes
        let n_windows = if samples_per_channel > config.window_size {
            (samples_per_channel - config.window_size) / config.hop_length + 1
        } else {
            1 // Minimum one window
        };
        let time_axis: Vec<F> = (0..n_windows)
            .map(|i| {
                to_precision::<F, _>(i) * to_precision::<F, _>(config.hop_length) / sample_rate
            })
            .collect();

        let n_freq_bins = config.window_size / 2 + 1;
        let freq_axis: Vec<F> = (0..n_freq_bins)
            .map(|i| {
                to_precision::<F, _>(i) * sample_rate / to_precision::<F, _>(config.window_size)
            })
            .collect();

        // Placeholder: create empty magnitude data
        let magnitude_data = Array2::<F>::zeros((n_freq_bins, n_windows));

        let metadata = PlotMetadata {
            x_label: Some("Time (s)".to_string()),
            y_label: Some("Frequency (Hz)".to_string()),
            title: Some("FFT Waterfall".to_string()),
            ..Default::default()
        };

        Ok(FftWaterfallPlot::new(
            time_axis,
            freq_axis,
            magnitude_data,
            config,
            metadata,
        ))
    }

    #[cfg(feature = "fft")]
    fn windowed_spectrum_comparison<F: RealFloat>(
        &self,
        config: Option<WindowComparisonConfig<F>>,
    ) -> AudioSampleResult<WindowComparisonPlot<F>> {
        let config = config.unwrap_or_default();

        // TODO: Implement window function comparison
        // This would compute FFT of different window functions and compare their frequency responses

        let sample_rate = to_precision::<F, _>(self.sample_rate.get());

        // Generate frequency axis
        let n_bins = config.fft_size / 2 + 1;
        let frequencies: Vec<F> = (0..n_bins)
            .map(|i| to_precision::<F, _>(i) * sample_rate / to_precision::<F, _>(config.fft_size))
            .collect();

        // Placeholder: create responses for each window type
        let window_responses: Vec<(String, Vec<F>)> = config
            .windows
            .iter()
            .map(|window_type| {
                let name = format!("{:?}", window_type);
                let response = vec![F::zero(); frequencies.len()]; // Placeholder
                (name, response)
            })
            .collect();

        // Generate line styles
        let line_styles: Vec<LineStyle<F>> = (0..config.windows.len())
            .map(|i| {
                let palette = ColorPalette::Default;
                LineStyle {
                    color: palette.get_color(i),
                    width: to_precision::<F, _>(2.0),
                    style: LineStyleType::Solid,
                }
            })
            .collect();

        let metadata = PlotMetadata {
            x_label: Some("Frequency (Hz)".to_string()),
            y_label: Some("Magnitude (dB)".to_string()),
            title: Some("Window Function Comparison".to_string()),
            ..Default::default()
        };

        Ok(WindowComparisonPlot::new(
            frequencies,
            window_responses,
            line_styles,
            metadata,
        ))
    }

    #[cfg(feature = "spectral-analysis")]
    fn instantaneous_frequency_plot<F: RealFloat>(
        &self,
        style: Option<LineStyle<F>>,
    ) -> AudioSampleResult<InstantaneousFrequencyPlot<F>> {
        let style = style.unwrap_or_else(|| LineStyle {
            color: "#e377c2".to_string(), // Pink
            width: to_precision::<F, _>(2.0),
            style: LineStyleType::Solid,
        });

        let sample_rate = to_precision::<F, _>(self.sample_rate.get());
        let samples_per_channel = self.samples_per_channel();

        // Generate time axis
        let time_axis: Vec<F> = (0..samples_per_channel)
            .map(|i| to_precision::<F, _>(i) / sample_rate)
            .collect();

        // TODO: Implement proper instantaneous frequency computation
        // This would require computing the analytic signal using Hilbert transform
        // For now, create placeholder complex signal
        let complex_signal: Vec<Complex<F>> = (0..samples_per_channel)
            .map(|_| Complex::new(F::zero(), F::zero()))
            .collect();

        let metadata = PlotMetadata {
            x_label: Some("Time (s)".to_string()),
            y_label: Some("Frequency (Hz)".to_string()),
            title: Some("Instantaneous Frequency".to_string()),
            legend_label: Some("Instantaneous Frequency".to_string()),
            ..Default::default()
        };

        Ok(InstantaneousFrequencyPlot::new(
            time_axis,
            complex_signal,
            sample_rate,
            style,
            metadata,
        ))
    }
}

// Implementation without spectral analysis support (core plotting only)
#[cfg(not(feature = "spectral-analysis"))]
impl<'a, T: AudioSample> AudioPlotBuilders<T> for AudioSamples<'a, T>
where
    i16: ConvertTo<T>,
    I24: ConvertTo<T>,
    i32: ConvertTo<T>,
    f32: ConvertTo<T>,
    f64: ConvertTo<T>,
    T: ConvertTo<f64>,
    f64: CastFrom<T>,
    for<'c> AudioSamples<'c, T>: AudioStatistics<'c, T>,
    for<'c> AudioSamples<'c, T>: AudioStatistics<'c, T>,
{
    fn waveform_plot<F>(&self, style: Option<LineStyle<F>>) -> AudioSampleResult<WaveformPlot<F, T>>
    where
        F: RealFloat + ConvertTo<T>,
        T: CastFrom<F> + ConvertTo<F>,
    {
        let sample_rate = self.sample_rate as f64;
        let samples_per_channel = self.samples_per_channel();

        // Generate time axis
        let time_data: Vec<F> = (0..samples_per_channel)
            .map(|i| to_precision::<F, _>(i as f64) / to_precision::<F, _>(sample_rate))
            .collect();

        let audio_samples = self
            .to_mono(MonoConversionMethod::<f64>::Average)
            .to_mono(MonoConversionMethod::<f64>::Average)
            .unwrap()
            .cast_as_f64()
            .expect("Failed to cast to f64 -- better handling in future");

        // Extract amplitude data preserving native sample ranges
        let amplitude_f64: Vec<f64> = audio_samples.to_interleaved_vec();
        let amplitude_data: Vec<F> = amplitude_f64
            .into_iter()
            .map(to_precision::<F, _>)
            .collect();
        let style = style.unwrap_or_default();
        let mut metadata = PlotMetadata::default();
        metadata.x_label = Some("Time (s)".to_string());
        metadata.y_label = Some(format!("Amplitude ({})", T::LABEL));

        Ok(WaveformPlot::new(
            time_data,
            amplitude_data,
            style,
            metadata,
        ))
    }

    fn waveform_plot_with_metadata<F>(
        &self,
        style: LineStyle<F>,
        style: LineStyle<F>,
        metadata: PlotMetadata,
    ) -> AudioSampleResult<WaveformPlot<F, T>>
    where
        F: RealFloat + ConvertTo<T>,
        T: CastFrom<F> + ConvertTo<F>,
    {
        let sample_rate = self.sample_rate as f64;
        let samples_per_channel = self.samples_per_channel();

        let time_data: Vec<F> = (0..samples_per_channel)
            .map(|i| to_precision::<F, _>(i as f64) / to_precision::<F, _>(sample_rate))
            .collect();

        let audio_samples = self
            .to_mono(MonoConversionMethod::<f64>::Average)
            .to_mono(MonoConversionMethod::<f64>::Average)
            .unwrap()
            .cast_as_f64()
            .expect("Failed to cast to f64 -- better handling in future");
        let amplitude_f64: Vec<f64> = audio_samples.to_interleaved_vec();
        let amplitude_data: Vec<F> = amplitude_f64
            .into_iter()
            .map(to_precision::<F, _>)
            .collect();

        Ok(WaveformPlot::new(
            time_data,
            amplitude_data,
            style,
            metadata,
        ))
    }

    fn onset_markers<F>(
        &self,
        marker_style: Option<MarkerStyle<F>>,
        line_style: Option<LineStyle<F>>,
        marker_style: Option<MarkerStyle<F>>,
        line_style: Option<LineStyle<F>>,
        show_strength: Option<bool>,
        threshold: Option<F>,
    ) -> AudioSampleResult<OnsetMarkers<F>>
    where
        F: RealFloat,
    {
        // Apply defaults
        let marker_style = marker_style.unwrap_or_else(|| MarkerStyle {
            color: "#d62728".to_string(), // Red
            size: to_precision::<F, _>(8.0),
            size: to_precision::<F, _>(8.0),
            shape: MarkerShape::Triangle,
            fill: true,
        });
        let line_style = line_style.or_else(|| {
            Some(LineStyle {
                color: "#d62728".to_string(), // Red
                width: to_precision::<F, _>(1.0),
                width: to_precision::<F, _>(1.0),
                style: LineStyleType::Dashed,
            })
        });
        let show_strength = show_strength.unwrap_or(true);
        let _threshold = threshold.unwrap_or_else(|| to_precision::<F, _>(0.1));
        let _threshold = threshold.unwrap_or_else(|| to_precision::<F, _>(0.1));

        let peak = self.peak();
        let peak_f64: f64 = peak.cast_into();
        let peak_f: F = to_precision::<F, _>(peak_f64);
        let y_range = (-peak_f, peak_f);
        let peak_f: F = to_precision::<F, _>(peak_f64);
        let y_range = (-peak_f, peak_f);

        let config = OnsetConfig {
            marker_style,
            line_style,
            show_strength,
            threshold: _threshold,
        };

        Ok(OnsetMarkers::new(Vec::new(), None, config, y_range))
    }

    fn beat_markers<F>(
        &self,
        marker_style: Option<MarkerStyle<F>>,
        line_style: Option<LineStyle<F>>,
        marker_style: Option<MarkerStyle<F>>,
        line_style: Option<LineStyle<F>>,
        show_tempo: Option<bool>,
    ) -> AudioSampleResult<BeatMarkers<F>>
    where
        F: RealFloat,
    {
        let marker_style = marker_style.unwrap_or_else(|| MarkerStyle {
            color: "#2ca02c".to_string(), // Green
            size: to_precision::<F, _>(10.0),
            shape: MarkerShape::Diamond,
            fill: true,
        });
        let line_style = line_style.or_else(|| {
            Some(LineStyle {
                color: "#2ca02c".to_string(), // Green
                width: to_precision::<F, _>(2.0),
                width: to_precision::<F, _>(2.0),
                style: LineStyleType::Solid,
            })
        });
        let show_tempo = show_tempo.unwrap_or(true);

        let peak = self.peak();
        let peak_f64: f64 = peak.cast_into();
        let peak_f: F = to_precision::<F, _>(peak_f64);
        let y_range = (-peak_f, peak_f);
        let peak_f: F = to_precision::<F, _>(peak_f64);
        let y_range = (-peak_f, peak_f);

        let config = BeatPlotConfig {
            marker_style,
            line_style,
            show_tempo,
        };

        Ok(BeatMarkers::new(Vec::new(), None, config, y_range))
    }

    fn pitch_contour<F>(
        &self,
        line_style: Option<LineStyle<F>>,
        line_style: Option<LineStyle<F>>,
        show_confidence: Option<bool>,
        freq_range: Option<(F, F)>,
        freq_range: Option<(F, F)>,
        _method: Option<PitchDetectionMethod>,
    ) -> AudioSampleResult<PitchContour<F>>
    where
        F: RealFloat,
    {
        let line_style = line_style.unwrap_or_else(|| LineStyle {
            color: "#ff00ff".to_string(), // Magenta
            width: to_precision::<F, _>(2.0),
            width: to_precision::<F, _>(2.0),
            style: LineStyleType::Solid,
        });
        let show_confidence = show_confidence.unwrap_or(false);
        let freq_range =
            freq_range.unwrap_or((to_precision::<F, _>(80.0), to_precision::<F, _>(2000.0)));

        Ok(PitchContour::new(
            Vec::new(),
            Vec::new(),
            None,
            line_style,
            show_confidence,
            freq_range,
        ))
    }
}

/// Creates a frequency domain plot from audio data.
///
/// # Arguments
/// * `data` - Vector of audio sample values
/// * `sample_rate` - Sample rate in Hz
/// * `samples_per_channel` - Number of samples per channel
/// * `style` - Optional line style configuration
///
/// # Returns
/// A `WaveformPlot` configured for frequency domain visualization
///
/// # Note
/// TODO: Add automatic support for multi-channel audio by plotting per-channel
/// waveforms stacked and an option to mixdown to mono before plotting
pub fn freq_plot<F: RealFloat, T: AudioSample>(
    data: Vec<T>,
    sample_rate: u32,
    samples_per_channel: i32,
    style: Option<LineStyle<F>>,
) -> WaveformPlot<F, T> {
    let sample_rate = to_precision::<F, _>(sample_rate);

    // Generate time axis
    let time_data: Vec<F> = (0..samples_per_channel)
        .map(|i| to_precision::<F, _>(i) / sample_rate)
        .collect();

    // Extract amplitude data preserving native sample ranges
    // Instead of normalizing to ±1.0, preserve the actual sample values
    let amplitude_data: Vec<F> = data.into_iter().map(|x| to_precision::<F, _>(x)).collect();
    let style = style.unwrap_or_default();
    let metadata = PlotMetadata {
        x_label: Some("Time (s)".to_string()),
        y_label: Some("Amplitude".to_string()),
        ..Default::default()
    };

    WaveformPlot::new(time_data, amplitude_data, style, metadata)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_line_style_default() {
        let style = LineStyle::<f64>::default();
        assert_eq!(style.width, 3.0); // Default is 3.0 pixels
    }

    #[test]
    fn test_marker_style_default() {
        let style = MarkerStyle::<f64>::default();
        assert_eq!(style.size, 8.0);
        assert!(style.fill);
    }

    #[test]
    fn test_spectrogram_config_default() {
        let config = SpectrogramConfig::<f64>::default();
        assert_eq!(config.n_fft, 2048);
        assert!(!config.log_freq);
    }

    #[test]
    fn test_plot_metadata_default() {
        let metadata = PlotMetadata::default();
        assert!(metadata.x_label.is_none());
        assert!(metadata.y_label.is_none());
        assert!(metadata.title.is_none());
    }

    #[test]
    fn test_waveform_plot_construction() {
        let time_data = vec![0.0, 0.1, 0.2, 0.3];
        let amplitude_data = vec![0.5, -0.5, 0.5, -0.5];
        let style = LineStyle::<f64>::default();
        let metadata = PlotMetadata::default();

        let plot: WaveformPlot<f64, f64> =
            WaveformPlot::new(time_data.clone(), amplitude_data.clone(), style, metadata);

        // Check bounds are properly calculated
        let bounds = plot.data_bounds();
        assert_eq!(bounds.x_min, 0.0);
        assert_eq!(bounds.x_max, 0.3);
    }

    #[test]
    fn test_plot_bounds_calculation() {
        let bounds = PlotBounds::new(0.0_f64, 1.0, -1.0, 1.0);
        assert_eq!(bounds.x_min, 0.0);
        assert_eq!(bounds.x_max, 1.0);
        assert_eq!(bounds.y_min, -1.0);
        assert_eq!(bounds.y_max, 1.0);
    }

    #[test]
    fn test_color_palette_get_color() {
        let palette = ColorPalette::Default;
        let color1 = palette.get_color(0);
        let color2 = palette.get_color(1);
        assert_ne!(color1, color2);
    }

    #[test]
    fn test_pitch_detection_method_variants() {
        let method = PitchDetectionMethod::Autocorrelation;
        match method {
            PitchDetectionMethod::Autocorrelation => {}
            PitchDetectionMethod::Yin => panic!("Wrong variant"),
            PitchDetectionMethod::Cepstrum => panic!("Wrong variant"),
        }
    }

    #[test]
    fn test_onset_config_default() {
        let config = OnsetConfig::<f64>::default();
        assert!(config.show_strength);
        assert!(config.line_style.is_some());
    }

    #[test]
    fn test_beat_config_default() {
        let config = BeatPlotConfig::<f64>::default();
        assert!(config.show_tempo);
        assert!(config.line_style.is_some());
    }

    #[cfg(feature = "spectral-analysis")]
    #[test]
    fn test_complex_spectrum_config_default() {
        let config = ComplexSpectrumConfig::<f64>::default();
        assert!(config.frequency_range.is_none());
    }
}
