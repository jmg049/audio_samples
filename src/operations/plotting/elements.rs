//! Individual plot element implementations.
//!
//! This module contains concrete implementations of the PlotElement trait
//! for various types of audio visualizations.

use super::core::*;
use crate::AudioSample;
use ndarray::Array2;
use plotly::common::{Line, Mode};
use plotly::{HeatMap, Scatter};
use std::marker::PhantomData;

/// Waveform plot element
#[derive(Debug, Clone)]
pub struct WaveformPlot<T: AudioSample> {
    data: Vec<(f64, f64)>, // (time, amplitude)
    style: LineStyle,
    bounds: PlotBounds,
    metadata: PlotMetadata,
    _phantom: PhantomData<T>,
}

impl<T: AudioSample> WaveformPlot<T> {
    pub fn new(
        time_data: Vec<f64>,
        amplitude_data: Vec<f64>,
        style: LineStyle,
        metadata: PlotMetadata,
    ) -> Self {
        // Combine time and amplitude data
        let data: Vec<(f64, f64)> = time_data
            .into_iter()
            .zip(amplitude_data.iter().copied())
            .collect();

        // Calculate bounds
        let x_min = data.iter().map(|(t, _)| *t).fold(f64::INFINITY, f64::min);
        let x_max = data
            .iter()
            .map(|(t, _)| *t)
            .fold(f64::NEG_INFINITY, f64::max);
        let y_min = data.iter().map(|(_, a)| *a).fold(f64::INFINITY, f64::min);
        let y_max = data
            .iter()
            .map(|(_, a)| *a)
            .fold(f64::NEG_INFINITY, f64::max);

        let bounds = PlotBounds::new(x_min, x_max, y_min, y_max);

        Self {
            data,
            style,
            bounds,
            metadata,
            _phantom: PhantomData,
        }
    }

    pub fn with_style(mut self, style: LineStyle) -> Self {
        self.style = style;
        self
    }

    pub fn with_legend_label(mut self, label: String) -> Self {
        self.metadata.legend_label = Some(label);
        self
    }
}

impl<T: AudioSample> PlotElement for WaveformPlot<T> {
    fn data_bounds(&self) -> PlotBounds {
        self.bounds
    }

    fn metadata(&self) -> &PlotMetadata {
        &self.metadata
    }

    fn to_plotly_traces(&self) -> Vec<PlotTrace> {
        let x_data: Vec<f64> = self.data.iter().map(|(x, _)| *x).collect();
        let y_data: Vec<f64> = self.data.iter().map(|(_, y)| *y).collect();

        let mut trace = Scatter::new(x_data, y_data)
            .mode(Mode::Lines)
            .line(self.style.to_plotly_line());

        if let Some(ref name) = self.metadata.legend_label {
            trace = trace.name(name);
        }

        vec![PlotTrace::Scatter(*trace)]
    }
}

/// Spectrogram plot element
#[derive(Debug, Clone)]
pub struct SpectrogramPlot {
    spectrogram_data: Array2<f64>, // Time x Frequency matrix (in dB)
    time_axis: Vec<f64>,
    freq_axis: Vec<f64>,
    config: SpectrogramConfig,
    bounds: PlotBounds,
    metadata: PlotMetadata,
}

impl SpectrogramPlot {
    pub fn new(
        spectrogram_data: Array2<f64>,
        time_axis: Vec<f64>,
        freq_axis: Vec<f64>,
        config: SpectrogramConfig,
        metadata: PlotMetadata,
    ) -> Self {
        let x_min = time_axis.first().copied().unwrap_or(0.0);
        let x_max = time_axis.last().copied().unwrap_or(1.0);
        let y_min = freq_axis.first().copied().unwrap_or(0.0);
        let y_max = freq_axis.last().copied().unwrap_or(1.0);

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

impl PlotElement for SpectrogramPlot {
    fn data_bounds(&self) -> PlotBounds {
        self.bounds
    }

    fn metadata(&self) -> &PlotMetadata {
        &self.metadata
    }

    fn to_plotly_traces(&self) -> Vec<PlotTrace> {
        // Create a heatmap for the spectrogram
        let (raw_vec, _offset) = self.spectrogram_data.clone().into_raw_vec_and_offset();
        let mut heatmap = HeatMap::new_z(raw_vec)
            .x(self.time_axis.clone())
            .y(self.freq_axis.clone())
            .color_scale(self.config.colormap.to_plotly_colorscale());

        if let Some(ref title) = self.metadata.title {
            heatmap = heatmap.name(title);
        }

        vec![PlotTrace::HeatMap(*heatmap)]
    }
}

/// Onset markers overlay element
#[derive(Debug, Clone)]
pub struct OnsetMarkers {
    onset_times: Vec<f64>,
    onset_strengths: Option<Vec<f64>>,
    config: OnsetConfig,
    bounds: PlotBounds,
    metadata: PlotMetadata,
}

impl OnsetMarkers {
    pub fn new(
        onset_times: Vec<f64>,
        onset_strengths: Option<Vec<f64>>,
        config: OnsetConfig,
        y_range: (f64, f64),
    ) -> Self {
        let x_min = onset_times.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let x_max = onset_times.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

        let bounds = PlotBounds::new(x_min, x_max, y_range.0, y_range.1);

        let mut metadata = PlotMetadata::default();
        metadata.legend_label = Some("Onsets".to_string());
        metadata.z_order = 10; // Draw on top

        Self {
            onset_times,
            onset_strengths,
            config,
            bounds,
            metadata,
        }
    }
}

impl PlotElement for OnsetMarkers {
    fn data_bounds(&self) -> PlotBounds {
        self.bounds
    }

    fn metadata(&self) -> &PlotMetadata {
        &self.metadata
    }

    fn to_plotly_traces(&self) -> Vec<PlotTrace> {
        let mut traces: Vec<PlotTrace> = Vec::new();

        // Add vertical lines for onset times
        if let Some(_) = &self.config.line_style {
            for &time in &self.onset_times {
                let line_trace =
                    Scatter::new(vec![time, time], vec![self.bounds.y_min, self.bounds.y_max])
                        .mode(Mode::Lines)
                        .line(Line::new().color(self.config.marker_style.color.clone()))
                        .show_legend(false);

                traces.push(PlotTrace::Scatter(*line_trace));
            }
        }

        // Add markers for onset points
        if !self.onset_times.is_empty() {
            let marker_y_positions: Vec<f64> = self
                .onset_times
                .iter()
                .enumerate()
                .map(|(i, _)| {
                    if self.config.show_strength {
                        if let Some(ref strengths) = self.onset_strengths {
                            let strength = strengths.get(i).unwrap_or(&1.0);
                            self.bounds.y_min + (self.bounds.y_max - self.bounds.y_min) * strength
                        } else {
                            self.bounds.y_max * 0.9
                        }
                    } else {
                        self.bounds.y_max * 0.9
                    }
                })
                .collect();

            let mut marker_trace = Scatter::new(self.onset_times.clone(), marker_y_positions)
                .mode(Mode::Markers)
                .marker(self.config.marker_style.to_plotly_marker());

            if let Some(ref name) = self.metadata.legend_label {
                marker_trace = marker_trace.name(name);
            }

            traces.push(PlotTrace::Scatter(*marker_trace));
        }

        traces
    }
}

/// Beat markers overlay element
#[derive(Debug, Clone)]
pub struct BeatMarkers {
    beat_times: Vec<f64>,
    tempo_bpm: Option<f64>,
    config: BeatConfig,
    bounds: PlotBounds,
    metadata: PlotMetadata,
}

impl BeatMarkers {
    pub fn new(
        beat_times: Vec<f64>,
        tempo_bpm: Option<f64>,
        config: BeatConfig,
        y_range: (f64, f64),
    ) -> Self {
        let x_min = beat_times.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let x_max = beat_times.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

        let bounds = PlotBounds::new(x_min, x_max, y_range.0, y_range.1);

        let mut metadata = PlotMetadata::default();
        metadata.legend_label = if let Some(bpm) = tempo_bpm {
            Some(format!("Beats ({:.1} BPM)", bpm))
        } else {
            Some("Beats".to_string())
        };
        metadata.z_order = 9; // Draw below onsets but above waveform

        Self {
            beat_times,
            tempo_bpm,
            config,
            bounds,
            metadata,
        }
    }
}

impl PlotElement for BeatMarkers {
    fn data_bounds(&self) -> PlotBounds {
        self.bounds
    }

    fn metadata(&self) -> &PlotMetadata {
        &self.metadata
    }

    fn to_plotly_traces(&self) -> Vec<PlotTrace> {
        let mut traces: Vec<PlotTrace> = Vec::new();

        // Add vertical lines for beat times
        if let Some(_) = &self.config.line_style {
            for &time in &self.beat_times {
                let line_trace =
                    Scatter::new(vec![time, time], vec![self.bounds.y_min, self.bounds.y_max])
                        .mode(Mode::Lines)
                        .line(Line::new().color(self.config.marker_style.color.clone()))
                        .show_legend(false);

                traces.push(PlotTrace::Scatter(*line_trace));
            }
        }

        // Add markers for beat points
        if !self.beat_times.is_empty() {
            let marker_y_positions: Vec<f64> = self
                .beat_times
                .iter()
                .map(|_| self.bounds.y_max * 0.8)
                .collect();

            let mut marker_trace = Scatter::new(self.beat_times.clone(), marker_y_positions)
                .mode(Mode::Markers)
                .marker(self.config.marker_style.to_plotly_marker());

            // Use the legend label that includes tempo information if available
            if let Some(ref name) = self.metadata.legend_label {
                marker_trace = marker_trace.name(name);
            } else if let Some(bpm) = self.tempo_bpm {
                marker_trace = marker_trace.name(&format!("Beats ({:.1} BPM)", bpm));
            }

            traces.push(PlotTrace::Scatter(*marker_trace));
        }

        traces
    }
}

/// Power spectrum plot element
#[derive(Debug, Clone)]
pub struct PowerSpectrumPlot<T: AudioSample> {
    frequencies: Vec<f64>,
    magnitudes: Vec<f64>, // in dB
    style: LineStyle,
    bounds: PlotBounds,
    metadata: PlotMetadata,
    _phantom: PhantomData<T>,
}

impl<T: AudioSample> PowerSpectrumPlot<T> {
    pub fn new(
        frequencies: Vec<f64>,
        magnitudes: Vec<f64>,
        style: LineStyle,
        metadata: PlotMetadata,
    ) -> Self {
        let x_min = frequencies.first().copied().unwrap_or(0.0);
        let x_max = frequencies.last().copied().unwrap_or(1.0);
        let y_min = magnitudes.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let y_max = magnitudes.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

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

impl<T: AudioSample> PlotElement for PowerSpectrumPlot<T> {
    fn data_bounds(&self) -> PlotBounds {
        self.bounds
    }

    fn metadata(&self) -> &PlotMetadata {
        &self.metadata
    }

    fn to_plotly_traces(&self) -> Vec<PlotTrace> {
        let mut trace = Scatter::new(self.frequencies.clone(), self.magnitudes.clone())
            .mode(Mode::Lines)
            .line(self.style.to_plotly_line());

        if let Some(ref name) = self.metadata.legend_label {
            trace = trace.name(name);
        }

        vec![PlotTrace::Scatter(*trace)]
    }
}
