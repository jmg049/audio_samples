//! Builder methods for creating plot elements from AudioSamples.
//!
//! This module provides the implementation of plot element builders that
//! integrate with the AudioSamples type and its analysis capabilities.

use super::core::*;
use super::elements::*;
use crate::AudioTypeConversion;
use crate::CastFrom;
use crate::operations::MonoConversionMethod;
use crate::operations::traits::AudioChannelOps;
use crate::operations::traits::{AudioStatistics, AudioTransforms};
use crate::operations::types::SpectrogramScale;
use crate::operations::types::WindowType;
use crate::{AudioSample, AudioSamples, ConvertTo, I24};
use plotly::Scatter;
use plotly::common::{Line, Mode};

/// Extension trait for AudioSamples to create plot elements
pub trait AudioPlotBuilders<T: AudioSample> {
    /// Create a waveform plot element
    fn waveform_plot(&self, style: Option<LineStyle>) -> WaveformPlot<T>;

    /// Create a waveform plot with custom metadata
    fn waveform_plot_with_metadata(
        &self,
        style: LineStyle,
        metadata: PlotMetadata,
    ) -> WaveformPlot<T>;

    /// Create a spectrogram plot element with function parameters
    fn spectrogram_plot(
        &self,
        n_fft: Option<usize>,
        hop_length: Option<usize>,
        window: Option<WindowType>,
        colormap: Option<ColorPalette>,
        db_range: Option<(f64, f64)>,
        log_freq: Option<bool>,
    ) -> PlotResult<SpectrogramPlot>;

    /// Create a power spectrum plot element with function parameters
    fn power_spectrum_plot(
        &self,
        n_fft: Option<usize>,
        window: Option<WindowType>,
        db_scale: Option<bool>,
        freq_range: Option<(f64, f64)>,
        style: Option<LineStyle>,
    ) -> PlotResult<PowerSpectrumPlot<T>>;

    /// Create a mel spectrogram plot element with function parameters
    fn mel_spectrogram_plot(
        &self,
        n_mels: Option<usize>,
        n_fft: Option<usize>,
        hop_length: Option<usize>,
        window: Option<WindowType>,
        f_min: Option<f64>,
        f_max: Option<f64>,
        colormap: Option<ColorPalette>,
        db_range: Option<(f64, f64)>,
    ) -> PlotResult<SpectrogramPlot>;

    /// Create onset markers overlay with function parameters
    fn onset_markers(
        &self,
        marker_style: Option<MarkerStyle>,
        line_style: Option<LineStyle>,
        show_strength: Option<bool>,
        threshold: Option<f64>,
    ) -> PlotResult<OnsetMarkers>;

    /// Create beat markers overlay with function parameters
    fn beat_markers(
        &self,
        marker_style: Option<MarkerStyle>,
        line_style: Option<LineStyle>,
        show_tempo: Option<bool>,
    ) -> PlotResult<BeatMarkers>;

    /// Create a pitch contour overlay with function parameters
    fn pitch_contour(
        &self,
        line_style: Option<LineStyle>,
        show_confidence: Option<bool>,
        freq_range: Option<(f64, f64)>,
        method: Option<PitchDetectionMethod>,
    ) -> PlotResult<PitchContour>;
}

// Config structs removed - using function parameters with defaults instead

/// Pitch detection methods
#[derive(Debug, Clone)]
pub enum PitchDetectionMethod {
    Autocorrelation,
    Yin,
    Cepstrum,
}

/// Pitch contour plot element
#[derive(Debug, Clone)]
pub struct PitchContour {
    time_axis: Vec<f64>,
    pitch_values: Vec<f64>, // in Hz
    confidence: Option<Vec<f64>>,
    line_style: LineStyle,
    show_confidence: bool,
    freq_range: (f64, f64),
    bounds: PlotBounds,
    metadata: PlotMetadata,
}

impl PitchContour {
    pub fn new(
        time_axis: Vec<f64>,
        pitch_values: Vec<f64>,
        confidence: Option<Vec<f64>>,
        line_style: LineStyle,
        show_confidence: bool,
        freq_range: (f64, f64),
    ) -> Self {
        let x_min = time_axis.first().copied().unwrap_or(0.0);
        let x_max = time_axis.last().copied().unwrap_or(1.0);
        let y_min = freq_range.0;
        let y_max = freq_range.1;

        let bounds = PlotBounds::new(x_min, x_max, y_min, y_max);

        let mut metadata = PlotMetadata::default();
        metadata.legend_label = Some("Pitch".to_string());
        metadata.y_label = Some("Frequency (Hz)".to_string());
        metadata.z_order = 8;

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

impl PlotElement for PitchContour {
    fn data_bounds(&self) -> PlotBounds {
        self.bounds
    }

    fn metadata(&self) -> &PlotMetadata {
        &self.metadata
    }

    fn to_plotly_traces(&self) -> Vec<PlotTrace> {
        let mut traces = Vec::new();

        // Filter out invalid pitch values and apply frequency range constraints
        let (x_data, y_data): (Vec<f64>, Vec<f64>) = self
            .time_axis
            .iter()
            .zip(self.pitch_values.iter())
            .filter(|&(_, pitch)| {
                *pitch > 0.0 && *pitch >= self.freq_range.0 && *pitch <= self.freq_range.1
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

            traces.push(PlotTrace::Scatter(*trace));
        }

        // If show_confidence is enabled and confidence data is available, add confidence visualization
        if self.show_confidence {
            if let Some(ref confidence_data) = self.confidence {
                let (conf_x_data, conf_y_data): (Vec<f64>, Vec<f64>) = self
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
                                .width(self.line_style.width * 0.5),
                        )
                        .name("Pitch Confidence");

                    traces.push(PlotTrace::Scatter(*confidence_trace));
                }
            }
        }

        traces
    }
}

impl<T: AudioSample> AudioPlotBuilders<T> for AudioSamples<T>
where
    i16: ConvertTo<T>,
    I24: ConvertTo<T>,
    i32: ConvertTo<T>,
    f32: ConvertTo<T>,
    f64: ConvertTo<T>,
    T: ConvertTo<f64>,
    f64: CastFrom<T>,
    AudioSamples<T>: AudioStatistics<T> + AudioTransforms<T>,
{
    // todo add automatic support for handling multi-channel audio by plotting per-channel waveforms stacked and also have an option to mixdown to mono before plotting (default for now)
    fn waveform_plot(&self, style: Option<LineStyle>) -> WaveformPlot<T> {
        let sample_rate = self.sample_rate() as f64;
        let samples_per_channel = self.samples_per_channel();

        // Generate time axis
        let time_data: Vec<f64> = (0..samples_per_channel)
            .map(|i| i as f64 / sample_rate)
            .collect();

        let audio_samples = self
            .to_mono(MonoConversionMethod::Average)
            .unwrap()
            .cast_as_f64()
            .expect("Failed to cast to f64 -- better handling in future");

        // Extract amplitude data preserving native sample ranges
        // Instead of normalizing to ±1.0, preserve the actual sample values
        let amplitude_data: Vec<f64> = audio_samples.to_interleaved_vec();
        let style = style.unwrap_or_default();
        let mut metadata = PlotMetadata::default();
        metadata.x_label = Some("Time (s)".to_string());

        // Set y-axis label based on sample type
        metadata.y_label = Some(format!("Amplitude ({})", T::LABEL));

        WaveformPlot::new(time_data, amplitude_data, style, metadata)
    }

    fn waveform_plot_with_metadata(
        &self,
        style: LineStyle,
        metadata: PlotMetadata,
    ) -> WaveformPlot<T> {
        let sample_rate = self.sample_rate() as f64;
        let samples_per_channel = self.samples_per_channel();

        let time_data: Vec<f64> = (0..samples_per_channel)
            .map(|i| i as f64 / sample_rate)
            .collect();

        let audio_samples = self
            .to_mono(MonoConversionMethod::Average)
            .unwrap()
            .cast_as_f64()
            .expect("Failed to cast to f64 -- better handling in future");
        let amplitude_data: Vec<f64> = audio_samples.to_interleaved_vec();

        WaveformPlot::new(time_data, amplitude_data, style, metadata)
    }

    fn spectrogram_plot(
        &self,
        n_fft: Option<usize>,
        hop_length: Option<usize>,
        window: Option<WindowType>,
        colormap: Option<ColorPalette>,
        db_range: Option<(f64, f64)>,
        log_freq: Option<bool>,
    ) -> PlotResult<SpectrogramPlot> {
        // Apply defaults
        let n_fft = n_fft.unwrap_or(2048);
        let hop_length = hop_length.unwrap_or(n_fft / 4);
        let window = window.unwrap_or(WindowType::Hanning);
        let colormap = colormap.unwrap_or(ColorPalette::Viridis);
        let db_range = db_range.unwrap_or((-80.0, 0.0));
        let _log_freq = log_freq.unwrap_or(false);

        // Convert to magnitude spectrogram in dB
        let spectrogram_data =
            self.spectrogram(n_fft, hop_length, window, SpectrogramScale::Log, true)?;
        let (rows, cols) = spectrogram_data.dim();

        // Generate time and frequency axes
        let sample_rate = self.sample_rate() as f64;
        let time_axis: Vec<f64> = (0..cols)
            .map(|i| i as f64 * hop_length as f64 / sample_rate)
            .collect();

        let freq_axis: Vec<f64> = (0..rows)
            .map(|i| i as f64 * sample_rate / n_fft as f64)
            .collect();

        let mut metadata = PlotMetadata::default();
        metadata.x_label = Some("Time (s)".to_string());
        metadata.y_label = Some("Frequency (Hz)".to_string());
        metadata.title = Some("Spectrogram".to_string());

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

    fn power_spectrum_plot(
        &self,
        n_fft: Option<usize>,
        window: Option<WindowType>,
        db_scale: Option<bool>,
        freq_range: Option<(f64, f64)>,
        style: Option<LineStyle>,
    ) -> PlotResult<PowerSpectrumPlot<T>> {
        // Apply defaults
        let _n_fft = n_fft.unwrap_or(2048);
        let _window = window.unwrap_or(WindowType::Hanning);
        let db_scale = db_scale.unwrap_or(true);
        let _freq_range = freq_range;
        let style = style.unwrap_or_else(|| LineStyle {
            color: "#1f77b4".to_string(), // Professional blue instead of hardcoded orange
            width: 3.0,
            style: LineStyleType::Solid,
        });

        // Compute FFT
        let fft_result = self.fft()?;
        let sample_rate = self.sample_rate() as f64;

        // Generate frequency axis
        let n_bins = fft_result.len() / 2; // Only positive frequencies
        let frequencies: Vec<f64> = (0..n_bins)
            .map(|i| i as f64 * sample_rate / fft_result.len() as f64)
            .collect();

        // Compute magnitudes
        let magnitudes: Vec<f64> = fft_result[0..n_bins]
            .iter()
            .map(|&complex_val| {
                let magnitude = complex_val.norm();
                if db_scale {
                    if magnitude > 0.0 {
                        20.0 * magnitude.log10()
                    } else {
                        -120.0 // Very low dB value for silence
                    }
                } else {
                    magnitude
                }
            })
            .collect();

        let mut metadata = PlotMetadata::default();
        metadata.x_label = Some("Frequency (Hz)".to_string());
        metadata.y_label = Some(
            if db_scale {
                "Magnitude (dB)"
            } else {
                "Magnitude"
            }
            .to_string(),
        );
        metadata.title = Some("Power Spectrum".to_string());

        Ok(PowerSpectrumPlot::new(
            frequencies,
            magnitudes,
            style,
            metadata,
        ))
    }

    fn mel_spectrogram_plot(
        &self,
        n_mels: Option<usize>,
        n_fft: Option<usize>,
        hop_length: Option<usize>,
        window: Option<WindowType>,
        f_min: Option<f64>,
        f_max: Option<f64>,
        colormap: Option<ColorPalette>,
        db_range: Option<(f64, f64)>,
    ) -> PlotResult<SpectrogramPlot> {
        // Apply defaults
        let n_mels = n_mels.unwrap_or(128);
        let n_fft = n_fft.unwrap_or(2048);
        let hop_length = hop_length.unwrap_or(n_fft / 4);
        let window = window.unwrap_or(WindowType::Hanning);
        let f_min = f_min.unwrap_or(0.0);
        let f_max = f_max;
        let colormap = colormap.unwrap_or(ColorPalette::Viridis);
        let db_range = db_range.unwrap_or((-80.0, 0.0));

        // Use the AudioTransforms trait method
        let mel_spec_data = self.mel_spectrogram(
            n_mels,
            f_min,
            f_max.unwrap_or(self.sample_rate() as f64 / 2.0), // Nyquist frequency as default
            n_fft,
            hop_length,
        )?;

        let (rows, cols) = mel_spec_data.dim();

        // Generate time and mel frequency axes
        let sample_rate = self.sample_rate() as f64;
        let time_axis: Vec<f64> = (0..cols)
            .map(|i| i as f64 * hop_length as f64 / sample_rate)
            .collect();

        let freq_axis: Vec<f64> = (0..rows)
            .map(|i| i as f64) // Mel bins are numbered, not frequency
            .collect();

        let mut metadata = PlotMetadata::default();
        metadata.x_label = Some("Time (s)".to_string());
        metadata.y_label = Some("Mel Frequency".to_string());
        metadata.title = Some("Mel Spectrogram".to_string());

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

    fn onset_markers(
        &self,
        marker_style: Option<MarkerStyle>,
        line_style: Option<LineStyle>,
        show_strength: Option<bool>,
        threshold: Option<f64>,
    ) -> PlotResult<OnsetMarkers> {
        // Apply defaults
        let marker_style = marker_style.unwrap_or_else(|| MarkerStyle {
            color: "#d62728".to_string(), // Red
            size: 8.0,
            shape: MarkerShape::Triangle,
            fill: true,
        });
        let line_style = line_style.or_else(|| {
            Some(LineStyle {
                color: "#d62728".to_string(), // Red
                width: 1.0,
                style: LineStyleType::Dashed,
            })
        });
        let show_strength = show_strength.unwrap_or(true);
        let _threshold = threshold.unwrap_or(0.1);

        // This would integrate with the onset detection module
        // For now, return empty markers
        let peak = self.peak();
        let peak_f64: f64 = peak.cast_into();
        let y_range = (-peak_f64, peak_f64);

        // Create a temporary config for the existing OnsetMarkers::new signature
        let config = OnsetConfig {
            marker_style,
            line_style,
            show_strength,
            threshold: _threshold,
        };

        Ok(OnsetMarkers::new(
            Vec::new(), // Would be populated by onset detection
            None,
            config,
            y_range,
        ))
    }

    fn beat_markers(
        &self,
        marker_style: Option<MarkerStyle>,
        line_style: Option<LineStyle>,
        show_tempo: Option<bool>,
    ) -> PlotResult<BeatMarkers> {
        // Apply defaults
        let marker_style = marker_style.unwrap_or_else(|| MarkerStyle {
            color: "#2ca02c".to_string(), // Green
            size: 10.0,
            shape: MarkerShape::Diamond,
            fill: true,
        });
        let line_style = line_style.or_else(|| {
            Some(LineStyle {
                color: "#2ca02c".to_string(), // Green
                width: 2.0,
                style: LineStyleType::Solid,
            })
        });
        let show_tempo = show_tempo.unwrap_or(true);

        // This would integrate with the beat tracking module
        // For now, return empty markers
        let peak = self.peak();
        let peak_f64: f64 = peak.cast_into();
        let y_range = (-peak_f64, peak_f64);

        // Create a temporary config for the existing BeatMarkers::new signature
        let config = BeatConfig {
            marker_style,
            line_style,
            show_tempo,
        };

        Ok(BeatMarkers::new(
            Vec::new(), // Would be populated by beat tracking
            None,
            config,
            y_range,
        ))
    }

    fn pitch_contour(
        &self,
        line_style: Option<LineStyle>,
        show_confidence: Option<bool>,
        freq_range: Option<(f64, f64)>,
        method: Option<PitchDetectionMethod>,
    ) -> PlotResult<PitchContour> {
        // Apply defaults
        let line_style = line_style.unwrap_or_else(|| LineStyle {
            color: "#ff00ff".to_string(), // Magenta
            width: 2.0,
            style: LineStyleType::Solid,
        });
        let show_confidence = show_confidence.unwrap_or(false);
        let freq_range = freq_range.unwrap_or((80.0, 2000.0));
        let _method = method.unwrap_or(PitchDetectionMethod::Autocorrelation);

        // This would integrate with pitch analysis
        // For now, return empty contour
        Ok(PitchContour::new(
            Vec::new(), // Would be populated by pitch detection
            Vec::new(),
            None,
            line_style,
            show_confidence,
            freq_range,
        ))
    }
}
