//! Individual plot element implementations.
//!
//! This module contains concrete implementations of the PlotElement trait
//! for various types of audio visualizations.

use super::core::*;
use crate::{AudioSample, RealFloat, to_precision};
use ndarray::Array2;
use num_complex::Complex;
#[cfg(feature = "plotting")]
use plotly::common::{Line, Mode};
#[cfg(feature = "plotting")]
use plotly::{HeatMap, Scatter};
use std::marker::PhantomData;

/// Waveform plot element
#[derive(Debug, Clone)]
pub struct WaveformPlot<F: RealFloat, T: AudioSample> {
    data: Vec<(F, F)>, // (time, amplitude)
    style: LineStyle<F>,
    bounds: PlotBounds<F>,
    metadata: PlotMetadata,
    _phantom: PhantomData<T>,
}

impl<F: RealFloat, T: AudioSample> WaveformPlot<F, T> {
    /// Creates a new waveform plot with the given time and amplitude data.
    ///
    /// # Arguments
    /// * `time_data` - Vector of time values in seconds
    /// * `amplitude_data` - Vector of amplitude values
    /// * `style` - Line style configuration for the plot
    /// * `metadata` - Plot metadata including title and labels
    ///
    /// # Returns
    /// A new `WaveformPlot` instance ready for rendering
    pub fn new(
        time_data: Vec<F>,
        amplitude_data: Vec<F>,
        style: LineStyle<F>,
        metadata: PlotMetadata,
    ) -> Self {
        // Combine time and amplitude data
        let data: Vec<(F, F)> = time_data
            .into_iter()
            .zip(amplitude_data.iter().copied())
            .collect();

        // Calculate bounds
        let x_min = data.iter().map(|(t, _)| *t).fold(F::infinity(), F::min);
        let x_max = data.iter().map(|(t, _)| *t).fold(F::neg_infinity(), F::max);
        let y_min = data.iter().map(|(_, a)| *a).fold(F::infinity(), F::min);
        let y_max = data.iter().map(|(_, a)| *a).fold(F::neg_infinity(), F::max);

        let bounds = PlotBounds::new(x_min, x_max, y_min, y_max);

        Self {
            data,
            style,
            bounds,
            metadata,
            _phantom: PhantomData,
        }
    }

    /// Updates the line style for this waveform plot.
    ///
    /// # Arguments
    /// * `style` - New line style to apply
    ///
    /// # Returns
    /// Self with updated style for method chaining
    pub fn with_style(mut self, style: LineStyle<F>) -> Self {
        self.style = style;
        self
    }

    /// Sets the legend label for this waveform plot.
    ///
    /// # Arguments
    /// * `label` - Legend label text
    ///
    /// # Returns
    /// Self with updated legend label for method chaining
    pub fn with_legend_label(mut self, label: String) -> Self {
        self.metadata.legend_label = Some(label);
        self
    }
}

impl<F: RealFloat, T: AudioSample> PlotElement<F> for WaveformPlot<F, T> {
    fn data_bounds(&self) -> PlotBounds<F> {
        self.bounds
    }

    fn metadata(&self) -> &PlotMetadata {
        &self.metadata
    }

    fn to_plotly_traces(&self) -> Vec<PlotTrace<F>> {
        let x_data: Vec<F> = self.data.iter().map(|(x, _)| *x).collect();
        let y_data: Vec<F> = self.data.iter().map(|(_, y)| *y).collect();

        let mut trace = Scatter::new(x_data, y_data)
            .mode(Mode::Lines)
            .line(self.style.to_plotly_line());

        if let Some(ref name) = self.metadata.legend_label {
            trace = trace.name(name);
        }

        vec![PlotTrace::Scatter(Box::new(*trace))]
    }
}

/// Spectrogram plot element
#[derive(Debug, Clone)]
pub struct SpectrogramPlot<F: RealFloat> {
    spectrogram_data: Array2<F>, // Time x Frequency matrix (in dB)
    time_axis: Vec<F>,
    freq_axis: Vec<F>,
    config: SpectrogramConfig<F>,
    bounds: PlotBounds<F>,
    metadata: PlotMetadata,
}

impl<F: RealFloat> SpectrogramPlot<F> {
    /// Creates a new spectrogram plot with the given data.
    ///
    /// # Arguments
    /// * `spectrogram_data` - 2D array of magnitude values in dB (time × frequency)
    /// * `time_axis` - Vector of time values in seconds
    /// * `freq_axis` - Vector of frequency values in Hz
    /// * `config` - Spectrogram visualization configuration
    /// * `metadata` - Plot metadata including title and labels
    ///
    /// # Returns
    /// A new `SpectrogramPlot` instance ready for rendering
    pub fn new(
        spectrogram_data: Array2<F>,
        time_axis: Vec<F>,
        freq_axis: Vec<F>,
        config: SpectrogramConfig<F>,
        metadata: PlotMetadata,
    ) -> Self {
        let x_min = time_axis.first().copied().unwrap_or(F::zero());
        let x_max = time_axis.last().copied().unwrap_or(F::one());
        let y_min = freq_axis.first().copied().unwrap_or(F::zero());
        let y_max = freq_axis.last().copied().unwrap_or(F::one());

        let bounds = PlotBounds::new(x_min, x_max, y_min, y_max);

        Self {
            spectrogram_data,
            time_axis,
            freq_axis,
            config,
            bounds,
            metadata,
        }
    }
}

impl<F: RealFloat> PlotElement<F> for SpectrogramPlot<F> {
    fn data_bounds(&self) -> PlotBounds<F> {
        self.bounds
    }

    fn metadata(&self) -> &PlotMetadata {
        &self.metadata
    }

    fn to_plotly_traces(&self) -> Vec<PlotTrace<F>> {
        // Create a heatmap for the spectrogram
        let (raw_vec, _offset) = self
            .spectrogram_data
            .mapv(|x| x.to_f64().expect("should not fail"))
            .into_raw_vec_and_offset();
        let mut heatmap = HeatMap::new_z(raw_vec)
            .x(self
                .time_axis
                .iter()
                .map(|x| x.to_f64().expect("Should not fail"))
                .collect())
            .y(self
                .freq_axis
                .iter()
                .map(|y| y.to_f64().expect("Should not fail"))
                .collect())
            .color_scale(self.config.colormap.to_plotly_colorscale());

        if let Some(ref title) = self.metadata.title {
            heatmap = heatmap.name(title);
        }

        vec![PlotTrace::HeatMap(Box::new(*heatmap))]
    }
}

/// Onset markers overlay element
#[derive(Debug, Clone)]
pub struct OnsetMarkers<F: RealFloat> {
    onset_times: Vec<F>,
    onset_strengths: Option<Vec<F>>,
    config: OnsetConfig<F>,
    bounds: PlotBounds<F>,
    metadata: PlotMetadata,
}

impl<F: RealFloat> OnsetMarkers<F> {
    /// Creates a new onset markers overlay element.
    ///
    /// # Arguments
    /// * `onset_times` - Vector of onset time positions in seconds
    /// * `onset_strengths` - Optional vector of onset strength values
    /// * `config` - Onset visualization configuration
    /// * `y_range` - Vertical range for the markers (min, max)
    ///
    /// # Returns
    /// A new `OnsetMarkers` instance ready for rendering
    pub fn new(
        onset_times: Vec<F>,
        onset_strengths: Option<Vec<F>>,
        config: OnsetConfig<F>,
        y_range: (F, F),
    ) -> Self {
        let x_min = onset_times.iter().fold(F::infinity(), |a, &b| a.min(b));
        let x_max = onset_times.iter().fold(F::neg_infinity(), |a, &b| a.max(b));

        let bounds = PlotBounds::new(x_min, x_max, y_range.0, y_range.1);

        let metadata = PlotMetadata {
            legend_label: Some("Onsets".to_string()),
            z_order: 10,
            ..Default::default()
        }; // Draw on top

        Self {
            onset_times,
            onset_strengths,
            config,
            bounds,
            metadata,
        }
    }
}

impl<F: RealFloat> PlotElement<F> for OnsetMarkers<F> {
    fn data_bounds(&self) -> PlotBounds<F> {
        self.bounds
    }

    fn metadata(&self) -> &PlotMetadata {
        &self.metadata
    }

    fn to_plotly_traces(&self) -> Vec<PlotTrace<F>> {
        let mut traces: Vec<PlotTrace<F>> = Vec::new();

        // Add vertical lines for onset times
        if self.config.line_style.is_some() {
            for &time in &self.onset_times {
                let line_trace =
                    Scatter::new(vec![time, time], vec![self.bounds.y_min, self.bounds.y_max])
                        .mode(Mode::Lines)
                        .line(Line::new().color(self.config.marker_style.color.clone()))
                        .show_legend(false);

                traces.push(PlotTrace::Scatter(Box::new(*line_trace)));
            }
        }

        // Add markers for onset points
        if !self.onset_times.is_empty() {
            let marker_y_positions: Vec<F> = self
                .onset_times
                .iter()
                .enumerate()
                .map(|(i, _)| {
                    if self.config.show_strength {
                        if let Some(ref strengths) = self.onset_strengths {
                            let strength = *strengths.get(i).unwrap_or(&F::one());
                            self.bounds.y_min + (self.bounds.y_max - self.bounds.y_min) * strength
                        } else {
                            self.bounds.y_max * to_precision::<F, _>(0.9)
                        }
                    } else {
                        self.bounds.y_max * to_precision::<F, _>(0.9)
                    }
                })
                .collect();

            let mut marker_trace = Scatter::new(self.onset_times.clone(), marker_y_positions)
                .mode(Mode::Markers)
                .marker(self.config.marker_style.to_plotly_marker());

            if let Some(ref name) = self.metadata.legend_label {
                marker_trace = marker_trace.name(name);
            }

            traces.push(PlotTrace::Scatter(Box::new(*marker_trace)));
        }

        traces
    }
}

/// Beat markers overlay element
#[derive(Debug, Clone)]
pub struct BeatMarkers<F: RealFloat> {
    beat_times: Vec<F>,
    tempo_bpm: Option<F>,
    config: BeatConfig<F>,
    bounds: PlotBounds<F>,
    metadata: PlotMetadata,
}

impl<F: RealFloat> BeatMarkers<F> {
    /// Creates a new beat markers overlay element.
    ///
    /// # Arguments
    /// * `beat_times` - Vector of beat time positions in seconds
    /// * `tempo_bpm` - Optional tempo in beats per minute
    /// * `config` - Beat visualization configuration
    /// * `y_range` - Vertical range for the markers (min, max)
    ///
    /// # Returns
    /// A new `BeatMarkers` instance ready for rendering
    pub fn new(
        beat_times: Vec<F>,
        tempo_bpm: Option<F>,
        config: BeatConfig<F>,
        y_range: (F, F),
    ) -> Self {
        let x_min = beat_times.iter().fold(F::infinity(), |a, &b| a.min(b));
        let x_max = beat_times.iter().fold(F::neg_infinity(), |a, &b| a.max(b));

        let bounds = PlotBounds::new(x_min, x_max, y_range.0, y_range.1);

        let metadata = PlotMetadata {
            legend_label: if let Some(bpm) = tempo_bpm {
                Some(format!("Beats ({:.1} BPM)", bpm))
            } else {
                Some("Beats".to_string())
            },
            z_order: 9,
            ..Default::default()
        }; // Draw below onsets but above waveform

        Self {
            beat_times,
            tempo_bpm,
            config,
            bounds,
            metadata,
        }
    }
}

impl<F: RealFloat> PlotElement<F> for BeatMarkers<F> {
    fn data_bounds(&self) -> PlotBounds<F> {
        self.bounds
    }

    fn metadata(&self) -> &PlotMetadata {
        &self.metadata
    }

    fn to_plotly_traces(&self) -> Vec<PlotTrace<F>> {
        let mut traces: Vec<PlotTrace<F>> = Vec::new();

        // Add vertical lines for beat times
        if self.config.line_style.is_some() {
            for &time in &self.beat_times {
                let line_trace =
                    Scatter::new(vec![time, time], vec![self.bounds.y_min, self.bounds.y_max])
                        .mode(Mode::Lines)
                        .line(Line::new().color(self.config.marker_style.color.clone()))
                        .show_legend(false);

                traces.push(PlotTrace::Scatter(Box::new(*line_trace)));
            }
        }

        // Add markers for beat points
        if !self.beat_times.is_empty() {
            let marker_y_positions: Vec<F> = self
                .beat_times
                .iter()
                .map(|_| self.bounds.y_max * to_precision::<F, _>(0.8))
                .collect();

            let mut marker_trace = Scatter::new(self.beat_times.clone(), marker_y_positions)
                .mode(Mode::Markers)
                .marker(self.config.marker_style.to_plotly_marker());

            // Use the legend label that includes tempo information if available
            if let Some(ref name) = self.metadata.legend_label {
                marker_trace = marker_trace.name(name);
            } else if let Some(bpm) = self.tempo_bpm {
                marker_trace = marker_trace.name(format!("Beats ({:.1} BPM)", bpm));
            }

            traces.push(PlotTrace::Scatter(Box::new(*marker_trace)));
        }

        traces
    }
}

/// Power spectrum plot element
#[derive(Debug, Clone)]
pub struct PowerSpectrumPlot<F: RealFloat, T: AudioSample> {
    frequencies: Vec<F>,
    magnitudes: Vec<F>, // in dB
    style: LineStyle<F>,
    bounds: PlotBounds<F>,
    metadata: PlotMetadata,
    _phantom: PhantomData<T>,
}

impl<F: RealFloat, T: AudioSample> PowerSpectrumPlot<F, T> {
    /// Creates a new power spectrum plot with the given frequency and magnitude data.
    ///
    /// # Arguments
    /// * `frequencies` - Vector of frequency values in Hz
    /// * `magnitudes` - Vector of magnitude values in dB
    /// * `style` - Line style configuration for the plot
    /// * `metadata` - Plot metadata including title and labels
    ///
    /// # Returns
    /// A new `PowerSpectrumPlot` instance ready for rendering
    pub fn new(
        frequencies: Vec<F>,
        magnitudes: Vec<F>,
        style: LineStyle<F>,
        metadata: PlotMetadata,
    ) -> Self {
        let x_min = frequencies.first().copied().unwrap_or_default();
        let x_max = frequencies.last().copied().unwrap_or_default();
        let y_min = magnitudes.iter().fold(F::infinity(), |a, &b| a.min(b));
        let y_max = magnitudes.iter().fold(F::neg_infinity(), |a, &b| a.max(b));

        let bounds = PlotBounds::new(x_min, x_max, y_min, y_max);

        Self {
            frequencies,
            magnitudes,
            style,
            bounds,
            metadata,
            _phantom: PhantomData,
        }
    }
}

impl<F: RealFloat, T: AudioSample> PlotElement<F> for PowerSpectrumPlot<F, T> {
    fn data_bounds(&self) -> PlotBounds<F> {
        self.bounds
    }

    fn metadata(&self) -> &PlotMetadata {
        &self.metadata
    }

    fn to_plotly_traces(&self) -> Vec<PlotTrace<F>> {
        let mut trace = Scatter::new(self.frequencies.clone(), self.magnitudes.clone())
            .mode(Mode::Lines)
            .line(self.style.to_plotly_line());

        if let Some(ref name) = self.metadata.legend_label {
            trace = trace.name(name);
        }

        vec![PlotTrace::Scatter(Box::new(*trace))]
    }
}

/// Complex spectrum plot element showing both magnitude and phase
#[derive(Debug, Clone)]
pub struct ComplexSpectrumPlot<F: RealFloat> {
    frequencies: Vec<F>,
    complex_values: Vec<Complex<F>>,
    config: ComplexSpectrumConfig<F>,
    bounds: PlotBounds<F>,
    metadata: PlotMetadata,
}

impl<F: RealFloat> ComplexSpectrumPlot<F> {
    /// Creates a new complex spectrum plot with the given frequency and complex data.
    ///
    /// # Arguments
    /// * `frequencies` - Vector of frequency values in Hz
    /// * `complex_values` - Vector of complex frequency domain values
    /// * `config` - Complex spectrum visualization configuration
    /// * `metadata` - Plot metadata including title and labels
    ///
    /// # Returns
    /// A new `ComplexSpectrumPlot` instance ready for rendering
    pub fn new(
        frequencies: Vec<F>,
        complex_values: Vec<Complex<F>>,
        config: ComplexSpectrumConfig<F>,
        metadata: PlotMetadata,
    ) -> Self {
        let x_min = frequencies.first().copied().unwrap_or_default();
        let x_max = frequencies.last().copied().unwrap_or_default();

        // Calculate magnitude and phase bounds
        let magnitudes: Vec<F> = complex_values.iter().map(|c| c.norm()).collect();
        let phases: Vec<F> = complex_values.iter().map(|c| c.arg()).collect();

        let mag_min = magnitudes.iter().fold(F::infinity(), |a, &b| a.min(b));
        let mag_max = magnitudes.iter().fold(F::neg_infinity(), |a, &b| a.max(b));
        let phase_min = phases.iter().fold(F::infinity(), |a, &b| a.min(b));
        let _phase_max = phases.iter().fold(F::neg_infinity(), |a, &b| a.max(b));

        // For split view, we'll need to handle bounds differently
        let (y_min, y_max) = if config.split_view {
            (phase_min, mag_max) // Combined range for subplot layout
        } else {
            // For overlay, normalize phase to magnitude scale
            (mag_min, mag_max)
        };

        let bounds = PlotBounds::new(x_min, x_max, y_min, y_max);

        Self {
            frequencies,
            complex_values,
            config,
            bounds,
            metadata,
        }
    }
}

impl<F: RealFloat> PlotElement<F> for ComplexSpectrumPlot<F> {
    fn data_bounds(&self) -> PlotBounds<F> {
        self.bounds
    }

    fn metadata(&self) -> &PlotMetadata {
        &self.metadata
    }

    fn to_plotly_traces(&self) -> Vec<PlotTrace<F>> {
        let mut traces = Vec::new();

        // Calculate magnitude and phase
        let magnitudes: Vec<F> = self
            .complex_values
            .iter()
            .map(|c| {
                let mag = c.norm();
                if self.config.db_scale && mag > F::zero() {
                    to_precision::<F, _>(20.0) * mag.log10()
                } else {
                    mag
                }
            })
            .collect();

        let phases: Vec<F> = self
            .complex_values
            .iter()
            .map(|c| {
                let phase = c.arg();
                match self.config.phase_mode {
                    PhaseDisplayMode::Wrapped | PhaseDisplayMode::Unwrapped => phase, // TODO: Implement unwrapping
                    PhaseDisplayMode::Degrees | PhaseDisplayMode::UnwrappedDegrees => {
                        phase.to_degrees()
                    } // TODO: Implement unwrapping
                }
            })
            .collect();

        // Magnitude trace
        let mag_trace = Scatter::new(self.frequencies.clone(), magnitudes)
            .mode(Mode::Lines)
            .line(self.config.magnitude_style.to_plotly_line())
            .name("Magnitude");

        traces.push(PlotTrace::Scatter(Box::new(*mag_trace)));

        // Phase trace (if not split view)
        if !self.config.split_view {
            let phase_trace = Scatter::new(self.frequencies.clone(), phases)
                .mode(Mode::Lines)
                .line(self.config.phase_style.to_plotly_line())
                .name("Phase");

            traces.push(PlotTrace::Scatter(Box::new(*phase_trace)));
        }

        traces
    }
}

/// Phase spectrum plot element for phase-only visualization
#[derive(Debug, Clone)]
pub struct PhaseSpectrumPlot<F: RealFloat> {
    frequencies: Vec<F>,
    phases: Vec<F>,
    style: LineStyle<F>,
    bounds: PlotBounds<F>,
    metadata: PlotMetadata,
}

impl<F: RealFloat> PhaseSpectrumPlot<F> {
    /// Creates a new phase spectrum plot with the given frequency and complex data.
    ///
    /// # Arguments
    /// * `frequencies` - Vector of frequency values in Hz
    /// * `complex_values` - Vector of complex frequency domain values
    /// * `style` - Line style configuration for the plot
    /// * `phase_mode` - Phase display mode (wrapped or unwrapped)
    /// * `metadata` - Plot metadata including title and labels
    ///
    /// # Returns
    /// A new `PhaseSpectrumPlot` instance ready for rendering
    pub fn new(
        frequencies: Vec<F>,
        complex_values: Vec<Complex<F>>,
        style: LineStyle<F>,
        phase_mode: PhaseDisplayMode,
        metadata: PlotMetadata,
    ) -> Self {
        let phases: Vec<F> = complex_values
            .iter()
            .map(|c| {
                let phase = c.arg();
                match phase_mode {
                    PhaseDisplayMode::Wrapped | PhaseDisplayMode::Unwrapped => phase, // TODO: Implement unwrapping
                    PhaseDisplayMode::Degrees | PhaseDisplayMode::UnwrappedDegrees => {
                        phase.to_degrees()
                    } // TODO: Implement unwrapping
                }
            })
            .collect();

        let x_min = frequencies.first().copied().unwrap_or_default();
        let x_max = frequencies.last().copied().unwrap_or_default();
        let y_min = phases.iter().fold(F::infinity(), |a, &b| a.min(b));
        let y_max = phases.iter().fold(F::neg_infinity(), |a, &b| a.max(b));

        let bounds = PlotBounds::new(x_min, x_max, y_min, y_max);

        Self {
            frequencies,
            phases,
            style,
            bounds,
            metadata,
        }
    }
}

impl<F: RealFloat> PlotElement<F> for PhaseSpectrumPlot<F> {
    fn data_bounds(&self) -> PlotBounds<F> {
        self.bounds
    }

    fn metadata(&self) -> &PlotMetadata {
        &self.metadata
    }

    fn to_plotly_traces(&self) -> Vec<PlotTrace<F>> {
        let mut trace = Scatter::new(self.frequencies.clone(), self.phases.clone())
            .mode(Mode::Lines)
            .line(self.style.to_plotly_line());

        if let Some(ref name) = self.metadata.legend_label {
            trace = trace.name(name);
        }

        vec![PlotTrace::Scatter(Box::new(*trace))]
    }
}

/// Peak frequency detection plot element
#[derive(Debug, Clone)]
pub struct PeakFrequencyPlot<F: RealFloat> {
    frequencies: Vec<F>,
    magnitudes: Vec<F>,
    peak_frequencies: Vec<F>,
    peak_magnitudes: Vec<F>,
    base_style: LineStyle<F>,
    peak_marker_style: MarkerStyle<F>,
    bounds: PlotBounds<F>,
    metadata: PlotMetadata,
}

impl<F: RealFloat> PeakFrequencyPlot<F> {
    /// Creates a new peak frequency detection plot.
    ///
    /// # Arguments
    /// * `frequencies` - Vector of frequency values in Hz
    /// * `magnitudes` - Vector of magnitude values in dB
    /// * `peak_frequencies` - Vector of detected peak frequency values in Hz
    /// * `peak_magnitudes` - Vector of detected peak magnitude values in dB
    /// * `base_style` - Line style for the base spectrum
    /// * `peak_marker_style` - Marker style for the peak indicators
    /// * `metadata` - Plot metadata including title and labels
    ///
    /// # Returns
    /// A new `PeakFrequencyPlot` instance ready for rendering
    pub fn new(
        frequencies: Vec<F>,
        magnitudes: Vec<F>,
        peak_frequencies: Vec<F>,
        peak_magnitudes: Vec<F>,
        base_style: LineStyle<F>,
        peak_marker_style: MarkerStyle<F>,
        metadata: PlotMetadata,
    ) -> Self {
        let x_min = frequencies.first().copied().unwrap_or_default();
        let x_max = frequencies.last().copied().unwrap_or_default();
        let y_min = magnitudes.iter().fold(F::infinity(), |a, &b| a.min(b));
        let y_max = magnitudes.iter().fold(F::neg_infinity(), |a, &b| a.max(b));

        let bounds = PlotBounds::new(x_min, x_max, y_min, y_max);

        Self {
            frequencies,
            magnitudes,
            peak_frequencies,
            peak_magnitudes,
            base_style,
            peak_marker_style,
            bounds,
            metadata,
        }
    }
}

impl<F: RealFloat> PlotElement<F> for PeakFrequencyPlot<F> {
    fn data_bounds(&self) -> PlotBounds<F> {
        self.bounds
    }

    fn metadata(&self) -> &PlotMetadata {
        &self.metadata
    }

    fn to_plotly_traces(&self) -> Vec<PlotTrace<F>> {
        let mut traces = Vec::new();

        // Base spectrum trace
        let base_trace = Scatter::new(self.frequencies.clone(), self.magnitudes.clone())
            .mode(Mode::Lines)
            .line(self.base_style.to_plotly_line())
            .name("Spectrum");

        traces.push(PlotTrace::Scatter(Box::new(*base_trace)));

        // Peak markers
        if !self.peak_frequencies.is_empty() {
            let peak_trace =
                Scatter::new(self.peak_frequencies.clone(), self.peak_magnitudes.clone())
                    .mode(Mode::Markers)
                    .marker(self.peak_marker_style.to_plotly_marker())
                    .name("Peaks");

            traces.push(PlotTrace::Scatter(Box::new(*peak_trace)));
        }

        traces
    }
}

/// Frequency bin tracking plot element for monitoring specific frequencies over time
#[derive(Debug, Clone)]
pub struct FrequencyBinPlot<F: RealFloat> {
    time_axis: Vec<F>,
    frequency_data: Vec<Vec<F>>, // [frequency][time] data
    target_frequencies: Vec<F>,
    line_styles: Vec<LineStyle<F>>,
    bounds: PlotBounds<F>,
    metadata: PlotMetadata,
}

impl<F: RealFloat> FrequencyBinPlot<F> {
    /// Creates a new frequency bin tracking plot for monitoring specific frequencies over time.
    ///
    /// # Arguments
    /// * `time_axis` - Vector of time values in seconds
    /// * `frequency_data` - Vector of frequency data vectors \[frequency\]\[time\]
    /// * `target_frequencies` - Vector of target frequency values being tracked in Hz
    /// * `line_styles` - Vector of line styles for each frequency trace
    /// * `metadata` - Plot metadata including title and labels
    ///
    /// # Returns
    /// A new `FrequencyBinPlot` instance ready for rendering
    pub fn new(
        time_axis: Vec<F>,
        frequency_data: Vec<Vec<F>>,
        target_frequencies: Vec<F>,
        line_styles: Vec<LineStyle<F>>,
        metadata: PlotMetadata,
    ) -> Self {
        let x_min = time_axis.first().copied().unwrap_or_default();
        let x_max = time_axis.last().copied().unwrap_or_default();

        // Find global y bounds across all frequency data
        let mut y_min = F::infinity();
        let mut y_max = F::neg_infinity();

        for freq_data in &frequency_data {
            for &value in freq_data {
                y_min = y_min.min(value);
                y_max = y_max.max(value);
            }
        }

        let bounds = PlotBounds::new(x_min, x_max, y_min, y_max);

        Self {
            time_axis,
            frequency_data,
            target_frequencies,
            line_styles,
            bounds,
            metadata,
        }
    }
}

impl<F: RealFloat> PlotElement<F> for FrequencyBinPlot<F> {
    fn data_bounds(&self) -> PlotBounds<F> {
        self.bounds
    }

    fn metadata(&self) -> &PlotMetadata {
        &self.metadata
    }

    fn to_plotly_traces(&self) -> Vec<PlotTrace<F>> {
        let mut traces = Vec::new();

        for (i, freq_data) in self.frequency_data.iter().enumerate() {
            let style = self.line_styles.get(i).cloned().unwrap_or_else(|| {
                let palette = ColorPalette::Default;
                LineStyle {
                    color: palette.get_color(i),
                    width: to_precision::<F, _>(2.0),
                    style: LineStyleType::Solid,
                }
            });

            let label = if i < self.target_frequencies.len() {
                format!("{:.1} Hz", self.target_frequencies[i])
            } else {
                format!("Frequency {}", i + 1)
            };

            let trace = Scatter::new(self.time_axis.clone(), freq_data.clone())
                .mode(Mode::Lines)
                .line(style.to_plotly_line())
                .name(&label);

            traces.push(PlotTrace::Scatter(Box::new(*trace)));
        }

        traces
    }
}

/// Group delay plot element for filter analysis
#[derive(Debug, Clone)]
pub struct GroupDelayPlot<F: RealFloat> {
    frequencies: Vec<F>,
    group_delay: Vec<F>,
    style: LineStyle<F>,
    bounds: PlotBounds<F>,
    metadata: PlotMetadata,
}

impl<F: RealFloat> GroupDelayPlot<F> {
    /// Creates a new group delay plot for filter analysis.
    ///
    /// # Arguments
    /// * `frequencies` - Vector of frequency values in Hz
    /// * `complex_values` - Vector of complex frequency response values
    /// * `style` - Line style configuration for the plot
    /// * `metadata` - Plot metadata including title and labels
    ///
    /// # Returns
    /// A new `GroupDelayPlot` instance ready for rendering
    pub fn new(
        frequencies: Vec<F>,
        complex_values: Vec<Complex<F>>,
        style: LineStyle<F>,
        metadata: PlotMetadata,
    ) -> Self {
        // Calculate group delay as negative derivative of phase
        let mut group_delay = Vec::with_capacity(complex_values.len());
        let two_pi = to_precision::<F, _>(2.0) * F::PI();
        for i in 0..complex_values.len() {
            let delay = if i == 0 || i == complex_values.len() - 1 {
                F::zero() // Boundary conditions
            } else {
                let phase_prev = complex_values[i - 1].arg();
                let phase_next = complex_values[i + 1].arg();
                let freq_step = frequencies[i + 1] - frequencies[i - 1];
                -(phase_next - phase_prev) / (two_pi * freq_step)
            };
            group_delay.push(delay);
        }

        let x_min = frequencies.first().copied().unwrap_or_default();
        let x_max = frequencies.last().copied().unwrap_or_default();
        let y_min = group_delay.iter().fold(F::infinity(), |a, &b| a.min(b));
        let y_max = group_delay.iter().fold(F::neg_infinity(), |a, &b| a.max(b));

        let bounds = PlotBounds::new(x_min, x_max, y_min, y_max);

        Self {
            frequencies,
            group_delay,
            style,
            bounds,
            metadata,
        }
    }
}

impl<F: RealFloat> PlotElement<F> for GroupDelayPlot<F> {
    fn data_bounds(&self) -> PlotBounds<F> {
        self.bounds
    }

    fn metadata(&self) -> &PlotMetadata {
        &self.metadata
    }

    fn to_plotly_traces(&self) -> Vec<PlotTrace<F>> {
        let mut trace = Scatter::new(self.frequencies.clone(), self.group_delay.clone())
            .mode(Mode::Lines)
            .line(self.style.to_plotly_line());

        if let Some(ref name) = self.metadata.legend_label {
            trace = trace.name(name);
        }

        vec![PlotTrace::Scatter(Box::new(*trace))]
    }
}

/// FFT Waterfall plot element for 3D time-frequency visualization
#[derive(Debug, Clone)]
pub struct FftWaterfallPlot<F: RealFloat> {
    time_axis: Vec<F>,
    freq_axis: Vec<F>,
    magnitude_data: Array2<F>, // Time x Frequency matrix
    config: WaterfallConfig<F>,
    bounds: PlotBounds<F>,
    metadata: PlotMetadata,
}

impl<F: RealFloat> FftWaterfallPlot<F> {
    /// Creates a new FFT waterfall plot for 3D time-frequency visualization.
    ///
    /// # Arguments
    /// * `time_axis` - Vector of time values in seconds
    /// * `freq_axis` - Vector of frequency values in Hz
    /// * `magnitude_data` - 2D array of magnitude values (time × frequency)
    /// * `config` - Waterfall visualization configuration
    /// * `metadata` - Plot metadata including title and labels
    ///
    /// # Returns
    /// A new `FftWaterfallPlot` instance ready for rendering
    pub fn new(
        time_axis: Vec<F>,
        freq_axis: Vec<F>,
        magnitude_data: Array2<F>,
        config: WaterfallConfig<F>,
        metadata: PlotMetadata,
    ) -> Self {
        let x_min = time_axis.first().copied().unwrap_or_default();
        let x_max = time_axis.last().copied().unwrap_or_default();
        let y_min = freq_axis.first().copied().unwrap_or_default();
        let y_max = freq_axis.last().copied().unwrap_or_default();

        let bounds = PlotBounds::new(x_min, x_max, y_min, y_max);

        Self {
            time_axis,
            freq_axis,
            magnitude_data,
            config,
            bounds,
            metadata,
        }
    }
}

impl<F: RealFloat> PlotElement<F> for FftWaterfallPlot<F> {
    fn data_bounds(&self) -> PlotBounds<F> {
        self.bounds
    }

    fn metadata(&self) -> &PlotMetadata {
        &self.metadata
    }

    fn to_plotly_traces(&self) -> Vec<PlotTrace<F>> {
        // Convert to heatmap representation
        let (raw_vec, _offset) = self
            .magnitude_data
            .mapv(|x| x.to_f64().expect("should not fail"))
            .into_raw_vec_and_offset();
        let mut heatmap: Box<HeatMap<f64, f64, f64>> = HeatMap::new_z(raw_vec)
            .x(self
                .time_axis
                .iter()
                .map(|x| x.to_f64().expect("Should not fail"))
                .collect())
            .y(self
                .freq_axis
                .iter()
                .map(|y| y.to_f64().expect("Should not fail"))
                .collect())
            .color_scale(self.config.colormap.to_plotly_colorscale());

        if let Some(ref title) = self.metadata.title {
            heatmap = heatmap.name(title);
        }

        // convert back to F

        vec![PlotTrace::HeatMap(Box::new(*heatmap))]
    }
}

/// Window function comparison plot element
#[derive(Debug, Clone)]
pub struct WindowComparisonPlot<F: RealFloat> {
    frequencies: Vec<F>,
    window_responses: Vec<(String, Vec<F>)>, // (window_name, response)
    line_styles: Vec<LineStyle<F>>,
    bounds: PlotBounds<F>,
    metadata: PlotMetadata,
}

impl<F: RealFloat> WindowComparisonPlot<F> {
    /// Creates a new window function comparison plot.
    ///
    /// # Arguments
    /// * `frequencies` - Vector of frequency values in Hz
    /// * `window_responses` - Vector of (window_name, response) tuples
    /// * `line_styles` - Vector of line styles for each window function
    /// * `metadata` - Plot metadata including title and labels
    ///
    /// # Returns
    /// A new `WindowComparisonPlot` instance ready for rendering
    pub fn new(
        frequencies: Vec<F>,
        window_responses: Vec<(String, Vec<F>)>,
        line_styles: Vec<LineStyle<F>>,
        metadata: PlotMetadata,
    ) -> Self {
        let x_min = frequencies.first().copied().unwrap_or_default();
        let x_max = frequencies.last().copied().unwrap_or_default();

        // Find global y bounds across all window responses
        let mut y_min = F::infinity();
        let mut y_max = F::neg_infinity();

        for (_, response) in &window_responses {
            for &value in response {
                y_min = y_min.min(value);
                y_max = y_max.max(value);
            }
        }

        let bounds = PlotBounds::new(x_min, x_max, y_min, y_max);

        Self {
            frequencies,
            window_responses,
            line_styles,
            bounds,
            metadata,
        }
    }
}

impl<F: RealFloat> PlotElement<F> for WindowComparisonPlot<F> {
    fn data_bounds(&self) -> PlotBounds<F> {
        self.bounds
    }

    fn metadata(&self) -> &PlotMetadata {
        &self.metadata
    }

    fn to_plotly_traces(&self) -> Vec<PlotTrace<F>> {
        let mut traces = Vec::new();

        for (i, (window_name, response)) in self.window_responses.iter().enumerate() {
            let style = self.line_styles.get(i).cloned().unwrap_or_else(|| {
                let palette = ColorPalette::Default;
                LineStyle {
                    color: palette.get_color(i),
                    width: to_precision::<F, _>(2.0),
                    style: LineStyleType::Solid,
                }
            });

            let trace = Scatter::new(self.frequencies.clone(), response.clone())
                .mode(Mode::Lines)
                .line(style.to_plotly_line())
                .name(window_name);

            traces.push(PlotTrace::Scatter(Box::new(*trace)));
        }

        traces
    }
}

/// Instantaneous frequency plot element for time-varying frequency analysis
#[derive(Debug, Clone)]
pub struct InstantaneousFrequencyPlot<F: RealFloat> {
    time_axis: Vec<F>,
    instantaneous_freq: Vec<F>,
    style: LineStyle<F>,
    bounds: PlotBounds<F>,
    metadata: PlotMetadata,
}

impl<F: RealFloat> InstantaneousFrequencyPlot<F> {
    /// Creates a new instantaneous frequency plot for time-varying frequency analysis.
    ///
    /// # Arguments
    /// * `time_axis` - Vector of time values in seconds
    /// * `complex_signal` - Vector of complex analytical signal values
    /// * `sample_rate` - Sampling rate in Hz
    /// * `style` - Line style configuration for the plot
    /// * `metadata` - Plot metadata including title and labels
    ///
    /// # Returns
    /// A new `InstantaneousFrequencyPlot` instance ready for rendering
    pub fn new(
        time_axis: Vec<F>,
        complex_signal: Vec<Complex<F>>,
        _sample_rate: F,
        style: LineStyle<F>,
        metadata: PlotMetadata,
    ) -> Self {
        // Calculate instantaneous frequency as derivative of phase
        let mut instantaneous_freq = Vec::with_capacity(complex_signal.len());
        let two_pi = to_precision::<F, _>(2.0) * F::PI();
        for i in 0..complex_signal.len() {
            let freq = if i == 0 || i == complex_signal.len() - 1 {
                F::zero() // Boundary conditions
            } else {
                let phase_prev = complex_signal[i - 1].arg();
                let phase_next = complex_signal[i + 1].arg();
                let dt = (time_axis[i + 1] - time_axis[i - 1]) / to_precision::<F, _>(2.0);
                (phase_next - phase_prev) / (two_pi * dt)
            };
            instantaneous_freq.push(freq);
        }

        let x_min = time_axis.first().copied().unwrap_or_default();
        let x_max = time_axis.last().copied().unwrap_or_default();
        let y_min = instantaneous_freq
            .iter()
            .fold(F::infinity(), |a, &b| a.min(b));
        let y_max = instantaneous_freq
            .iter()
            .fold(F::neg_infinity(), |a, &b| a.max(b));

        let bounds = PlotBounds::new(x_min, x_max, y_min, y_max);

        Self {
            time_axis,
            instantaneous_freq,
            style,
            bounds,
            metadata,
        }
    }
}

impl<F: RealFloat> PlotElement<F> for InstantaneousFrequencyPlot<F> {
    fn data_bounds(&self) -> PlotBounds<F> {
        self.bounds
    }

    fn metadata(&self) -> &PlotMetadata {
        &self.metadata
    }

    fn to_plotly_traces(&self) -> Vec<PlotTrace<F>> {
        let mut trace = Scatter::new(self.time_axis.clone(), self.instantaneous_freq.clone())
            .mode(Mode::Lines)
            .line(self.style.to_plotly_line());

        if let Some(ref name) = self.metadata.legend_label {
            trace = trace.name(name);
        }

        vec![PlotTrace::Scatter(Box::new(*trace))]
    }
}
