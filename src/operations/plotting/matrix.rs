//! Interactive plots of feature matrices and simplified waveforms.
//!
//! [`plot_matrix`] draws any matrix as a heatmap whose axes know what they show: seconds,
//! lag, Hz, mel, constant-Q bins, note names, pitch classes, tempo or tonnetz dimensions,
//! with decibel or raw values. [`plot_waveform_bars`] draws a waveform's envelope as bars,
//! and [`plot_waveform_f0`] draws it displaced along a pitch track.
//!
//! ```no_run
//! use audio_samples::operations::plotting::matrix::{AxisKind, MatrixPlotParams, plot_matrix};
//! use audio_samples::operations::plotting::PlotUtils;
//! use ndarray::Array2;
//!
//! let chroma = Array2::<f64>::zeros((12, 100));
//! let params = MatrixPlotParams::new(AxisKind::Time, AxisKind::Chroma);
//! let plot = plot_matrix(chroma.view(), &params).unwrap();
//! plot.save("chroma.html").unwrap();
//! ```

use std::path::Path;

use ndarray::{ArrayView1, ArrayView2};
use plotly::common::{ColorScale, ColorScalePalette, Fill, Mode, Title};
use plotly::layout::{Axis, AxisType, BarMode, Layout};
use plotly::{HeatMap, Plot, Scatter};

use crate::operations::plotting::PlotUtils;
use crate::utils::audio_math::{
    MelScale, cqt_frequencies, key_to_notes, mel_frequencies_with_scale,
};
use crate::{AudioFloat, AudioSampleResult, ParameterError};

fn invalid<T>(parameter: &str, reason: impl Into<String>) -> AudioSampleResult<T> {
    Err(ParameterError::invalid_value(parameter, reason.into()).into())
}

/// What an axis of [`plot_matrix`] shows.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AxisKind {
    /// Column or row numbers.
    #[default]
    Frames,
    /// Seconds.
    Time,
    /// Seconds of lag, wrapped to negative past the middle.
    Lag,
    /// FFT bins in Hz, on a linear scale.
    Linear,
    /// FFT bins in Hz, on a logarithmic scale.
    Log,
    /// Mel bands in Hz.
    Mel,
    /// Constant-Q bins in Hz.
    Cqt,
    /// Constant-Q bins named by note.
    CqtNote,
    /// Pitch classes named in the key.
    Chroma,
    /// Lags as tempo in beats per minute.
    Tempo,
    /// The six tonal-centroid dimensions.
    Tonnetz,
    /// No ticks or label.
    Off,
}

/// How [`plot_matrix`] colours the values.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum ValueScale {
    /// The values as they are.
    #[default]
    Linear,
    /// Decibels of an amplitude against a reference.
    Db(f64),
    /// Decibels of a power against a reference.
    DbPower(f64),
    /// Decibels of an amplitude against the largest value.
    DbFullScale,
    /// Decibels of a power against the largest value.
    DbFullScalePower,
}

/// Parameters of [`plot_matrix`].
#[derive(Debug, Clone, PartialEq)]
pub struct MatrixPlotParams {
    /// Horizontal axis.
    pub x_axis: AxisKind,
    /// Vertical axis.
    pub y_axis: AxisKind,
    /// Value colouring. Default [`ValueScale::Linear`].
    pub value_scale: ValueScale,
    /// Dynamic range of decibel scales. Default 80.
    pub top_db: Option<f64>,
    /// Sample rate behind time and frequency axes. Default 22050.
    pub sample_rate: f64,
    /// Samples between frames. Default 512.
    pub hop_length: usize,
    /// FFT size. Default: `2 * (rows - 1)`.
    pub n_fft: Option<usize>,
    /// Lowest frequency of mel and constant-Q axes.
    pub fmin: Option<f64>,
    /// Highest frequency of mel axes. Default: half the sample rate.
    pub fmax: Option<f64>,
    /// Constant-Q bins per octave. Default 12.
    pub bins_per_octave: usize,
    /// Key for note and chroma names. Default C major.
    pub key: String,
    /// Plot title.
    pub title: Option<String>,
}

impl MatrixPlotParams {
    /// Parameters for the given axes.
    #[must_use]
    pub fn new(x_axis: AxisKind, y_axis: AxisKind) -> Self {
        Self {
            x_axis,
            y_axis,
            value_scale: ValueScale::Linear,
            top_db: Some(80.0),
            sample_rate: 22050.0,
            hop_length: 512,
            n_fft: None,
            fmin: None,
            fmax: None,
            bins_per_octave: 12,
            key: "C:maj".to_owned(),
            title: None,
        }
    }

    /// librosa's display defaults for the given axes; currently equal to [`new`](Self::new).
    #[must_use]
    pub fn librosa(x_axis: AxisKind, y_axis: AxisKind) -> Self {
        Self::new(x_axis, y_axis)
    }
}

/// A rendered plot of a matrix or waveform.
#[derive(Clone)]
pub struct MatrixPlot {
    plot: Plot,
}

impl PlotUtils for MatrixPlot {
    fn html(&self) -> AudioSampleResult<String> {
        Ok(self.plot.to_html())
    }

    #[cfg(feature = "html_view")]
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

    fn save<P: AsRef<Path>>(&self, path: P) -> AudioSampleResult<()> {
        let path = path.as_ref();
        match path.extension().and_then(|s| s.to_str()).unwrap_or("html") {
            "html" => std::fs::write(path, self.html()?)
                .map_err(|e| crate::AudioSampleError::io("save", &e)),
            other => invalid(
                "file_extension",
                format!(
                    "{other}: a matrix plot saves as html; open it in a browser to export an image"
                ),
            ),
        }
    }
}

fn coords(kind: AxisKind, n: usize, p: &MatrixPlotParams) -> AudioSampleResult<Vec<f64>> {
    let sr = p.sample_rate;
    Ok(match kind {
        AxisKind::Frames | AxisKind::Off | AxisKind::Tonnetz => (0..n).map(|i| i as f64).collect(),
        AxisKind::Time | AxisKind::Lag => (0..n)
            .map(|i| i as f64 * p.hop_length as f64 / sr)
            .collect(),
        AxisKind::Linear | AxisKind::Log => {
            let n_fft = p.n_fft.unwrap_or(2 * n.saturating_sub(1)).max(1);
            (0..n).map(|i| i as f64 * sr / n_fft as f64).collect()
        }
        AxisKind::Mel => mel_frequencies_with_scale(
            n,
            p.fmin.unwrap_or(0.0),
            p.fmax.unwrap_or(sr / 2.0),
            MelScale::Slaney,
        ),
        AxisKind::Cqt | AxisKind::CqtNote => {
            let bins = std::num::NonZeroUsize::new(p.bins_per_octave).ok_or_else(|| {
                ParameterError::invalid_value("bins_per_octave", "must be positive".to_string())
            })?;
            cqt_frequencies(n, p.fmin.unwrap_or(32.703_195_662_574_83), bins, 0.0)
        }
        AxisKind::Chroma => (0..n)
            .map(|i| i as f64 * 12.0 / p.bins_per_octave.max(1) as f64)
            .collect(),
        AxisKind::Tempo => (0..n)
            .map(|i| 60.0 * sr / (p.hop_length as f64 * (i as f64).max(0.5)))
            .collect(),
    })
}

fn decorate(
    axis: Axis,
    kind: AxisKind,
    values: &[f64],
    p: &MatrixPlotParams,
) -> AudioSampleResult<Axis> {
    Ok(match kind {
        AxisKind::Off => axis.visible(false),
        AxisKind::Frames => axis.title(Title::with_text("Frames")),
        AxisKind::Time => axis.title(Title::with_text("Time (s)")),
        AxisKind::Lag => {
            let end = values.last().copied().unwrap_or(0.0);
            let picks: Vec<f64> = (0..=8).map(|i| end * f64::from(i) / 8.0).collect();
            let text: Vec<String> = picks
                .iter()
                .map(|v| format!("{:.3}", if *v > 0.5 * end { v - end } else { *v }))
                .collect();
            axis.title(Title::with_text("Lag (s)"))
                .tick_values(picks)
                .tick_text(text)
        }
        AxisKind::Linear => axis.title(Title::with_text("Hz")),
        AxisKind::Log | AxisKind::Mel | AxisKind::Cqt => {
            axis.title(Title::with_text("Hz")).type_(AxisType::Log)
        }
        AxisKind::CqtNote => {
            let names = key_to_notes(&p.key, false, false)?;
            let text: Vec<String> = values
                .iter()
                .map(|&hz| {
                    let midi = (69.0 + 12.0 * (hz / 440.0).log2()).round() as i64;
                    format!(
                        "{}{}",
                        names[midi.rem_euclid(12) as usize],
                        midi.div_euclid(12) - 1
                    )
                })
                .collect();
            axis.title(Title::with_text("Note"))
                .type_(AxisType::Log)
                .tick_values(values.to_vec())
                .tick_text(text)
        }
        AxisKind::Chroma => {
            let names = key_to_notes(&p.key, false, false)?;
            let text: Vec<String> = values
                .iter()
                .map(|&v| names[(v.round() as i64).rem_euclid(12) as usize].clone())
                .collect();
            axis.title(Title::with_text("Pitch class"))
                .tick_values(values.to_vec())
                .tick_text(text)
        }
        AxisKind::Tempo => axis.title(Title::with_text("BPM")).type_(AxisType::Log),
        AxisKind::Tonnetz => axis
            .title(Title::with_text("Tonnetz"))
            .tick_values((0..6).map(f64::from).collect())
            .tick_text(
                ["5x", "5y", "m3x", "m3y", "M3x", "M3y"]
                    .map(String::from)
                    .to_vec(),
            ),
    })
}

fn scale_values(data: &ArrayView2<'_, f64>, p: &MatrixPlotParams) -> Vec<Vec<f64>> {
    let peak = data.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
    let (factor, reference) = match p.value_scale {
        ValueScale::Linear => {
            return data.rows().into_iter().map(|r| r.to_vec()).collect();
        }
        ValueScale::Db(r) => (20.0, r),
        ValueScale::DbPower(r) => (10.0, r),
        ValueScale::DbFullScale => (20.0, peak),
        ValueScale::DbFullScalePower => (10.0, peak),
    };
    let floor = if factor == 10.0 { 1e-10 } else { 1e-5 } * reference.abs().max(1e-300);
    let top = factor * (peak.max(floor) / reference).log10();
    let lowest = p.top_db.map_or(f64::NEG_INFINITY, |t| top - t);
    data.rows()
        .into_iter()
        .map(|r| {
            r.iter()
                .map(|v| (factor * (v.abs().max(floor) / reference).log10()).max(lowest))
                .collect()
        })
        .collect()
}

/// Draws a feature matrix with axes that know what they show.
///
/// # Arguments
/// - `data`: `(rows, columns)`.
/// - `params`: the axes, the value colouring and the sampling parameters.
///
/// # Returns
/// The plot.
///
/// # Errors
/// [`ParameterError`] if `data` is empty, the sample rate is not positive or a key is invalid.
///
/// # Examples
/// See the [module documentation](self).
pub fn plot_matrix<F: AudioFloat>(
    data: ArrayView2<'_, F>,
    params: &MatrixPlotParams,
) -> AudioSampleResult<MatrixPlot> {
    if data.is_empty() {
        return invalid("data", "must not be empty");
    }
    if !(params.sample_rate.is_finite() && params.sample_rate > 0.0) {
        return invalid("sample_rate", "must be positive and finite");
    }
    let (rows, cols) = data.dim();
    let x = coords(params.x_axis, cols, params)?;
    let y = coords(params.y_axis, rows, params)?;
    let as_f64 = data.mapv(|v| v.to_f64().unwrap_or(f64::NAN));
    let values = scale_values(&as_f64.view(), params);
    let palette = if matches!(params.value_scale, ValueScale::Linear)
        && as_f64.iter().any(|v| *v < 0.0)
        && as_f64.iter().any(|v| *v > 0.0)
    {
        ColorScalePalette::RdBu
    } else {
        ColorScalePalette::Viridis
    };
    let trace =
        HeatMap::new(x.clone(), y.clone(), values).color_scale(ColorScale::Palette(palette));
    let mut plot = Plot::new();
    plot.add_trace(trace);
    let mut layout = Layout::new()
        .x_axis(decorate(Axis::new(), params.x_axis, &x, params)?)
        .y_axis(decorate(Axis::new(), params.y_axis, &y, params)?);
    if let Some(t) = &params.title {
        layout = layout.title(Title::with_text(t.clone()));
    }
    plot.set_layout(layout);
    Ok(MatrixPlot { plot })
}

fn block_peaks<F: AudioFloat>(y: ArrayView1<'_, F>, hop: usize) -> Vec<f64> {
    y.to_vec()
        .chunks(hop.max(1))
        .map(|c| {
            c.iter()
                .fold(0.0_f64, |m, v| m.max(v.to_f64().unwrap_or(0.0).abs()))
        })
        .collect()
}

/// Draws a waveform as bars of its envelope, for small figures.
///
/// # Arguments
/// - `y`: the samples.
/// - `sample_rate`: their rate in Hz.
/// - `n_bars`: the number of bars.
/// - `gap_ratio`: the fraction of each bar's slot left empty, in `[0, 1)`.
///
/// # Returns
/// The plot.
///
/// # Errors
/// [`ParameterError`] if `y` is empty, `n_bars` is 0, or a parameter is out of range.
///
/// # Examples
/// ```no_run
/// use audio_samples::operations::plotting::matrix::plot_waveform_bars;
/// use ndarray::Array1;
///
/// let y = Array1::<f64>::zeros(1000);
/// let plot = plot_waveform_bars(y.view(), 22050.0, 50, 0.4).unwrap();
/// ```
pub fn plot_waveform_bars<F: AudioFloat>(
    y: ArrayView1<'_, F>,
    sample_rate: f64,
    n_bars: usize,
    gap_ratio: f64,
) -> AudioSampleResult<MatrixPlot> {
    if y.is_empty() || n_bars == 0 {
        return invalid("n_bars", "needs samples and at least one bar");
    }
    if !(0.0..1.0).contains(&gap_ratio) {
        return invalid("gap_ratio", "must lie in [0, 1)");
    }
    if !(sample_rate.is_finite() && sample_rate > 0.0) {
        return invalid("sample_rate", "must be positive and finite");
    }
    let hop = (y.len() / n_bars).max(1);
    let peaks = block_peaks(y, hop);
    let slot = hop as f64 / sample_rate;
    let centres: Vec<f64> = (0..peaks.len()).map(|i| (i as f64 + 0.5) * slot).collect();
    let lower: Vec<f64> = peaks.iter().map(|p| -p).collect();
    let width = slot * (1.0 - gap_ratio);
    let mut plot = Plot::new();
    plot.add_trace(plotly::Bar::new(centres.clone(), peaks).width(width));
    plot.add_trace(plotly::Bar::new(centres, lower).width(width));
    plot.set_layout(
        Layout::new()
            .bar_mode(BarMode::Relative)
            .x_axis(Axis::new().title(Title::with_text("Time (s)"))),
    );
    Ok(MatrixPlot { plot })
}

/// Draws a waveform riding on its pitch track, with the amplitude displacing the frequency
/// by up to one bin.
///
/// # Arguments
/// - `y`: the samples.
/// - `f0`: the fundamental frequency per frame, in Hz; NaN where unvoiced.
/// - `sample_rate`: the rate of `y` in Hz.
/// - `hop_length`: samples between frames of `f0`.
/// - `bins_per_octave`: the displacement per unit amplitude is one bin.
///
/// # Returns
/// The plot.
///
/// # Errors
/// [`ParameterError`] if an input is empty or a parameter is out of range.
///
/// # Examples
/// ```no_run
/// use audio_samples::operations::plotting::matrix::plot_waveform_f0;
/// use ndarray::{Array1, array};
///
/// let y = Array1::<f64>::zeros(2048);
/// let plot = plot_waveform_f0(y.view(), array![220.0, 220.0].view(), 22050.0, 1024, 12).unwrap();
/// ```
pub fn plot_waveform_f0<F: AudioFloat>(
    y: ArrayView1<'_, F>,
    f0: ArrayView1<'_, f64>,
    sample_rate: f64,
    hop_length: usize,
    bins_per_octave: usize,
) -> AudioSampleResult<MatrixPlot> {
    if y.is_empty() || f0.is_empty() {
        return invalid("y", "needs samples and a pitch track");
    }
    if hop_length == 0 || bins_per_octave == 0 {
        return invalid(
            "hop_length",
            "hop_length and bins_per_octave must be positive",
        );
    }
    if !(sample_rate.is_finite() && sample_rate > 0.0) {
        return invalid("sample_rate", "must be positive and finite");
    }
    let peaks = block_peaks(y, hop_length);
    let largest = peaks
        .iter()
        .fold(0.0_f64, |m, v| m.max(*v))
        .max(f64::MIN_POSITIVE);
    let n = peaks.len().min(f0.len());
    let times: Vec<f64> = (0..n)
        .map(|i| i as f64 * hop_length as f64 / sample_rate)
        .collect();
    let octaves = 1.0 / bins_per_octave as f64;
    let upper: Vec<f64> = (0..n)
        .map(|i| f0[i] * 2f64.powf(peaks[i] / largest * octaves))
        .collect();
    let lower: Vec<f64> = (0..n)
        .map(|i| f0[i] * 2f64.powf(-peaks[i] / largest * octaves))
        .collect();
    let mut plot = Plot::new();
    plot.add_trace(Scatter::new(times.clone(), upper).mode(Mode::Lines));
    plot.add_trace(
        Scatter::new(times, lower)
            .mode(Mode::Lines)
            .fill(Fill::ToNextY),
    );
    plot.set_layout(
        Layout::new()
            .x_axis(Axis::new().title(Title::with_text("Time (s)")))
            .y_axis(
                Axis::new()
                    .title(Title::with_text("Hz"))
                    .type_(AxisType::Log),
            ),
    );
    Ok(MatrixPlot { plot })
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array1, Array2};

    #[test]
    fn every_axis_renders() {
        let data = Array2::from_shape_fn((12, 8), |(i, j)| (i + j) as f64 + 1.0);
        for kind in [
            AxisKind::Frames,
            AxisKind::Time,
            AxisKind::Lag,
            AxisKind::Linear,
            AxisKind::Log,
            AxisKind::Mel,
            AxisKind::Cqt,
            AxisKind::CqtNote,
            AxisKind::Chroma,
            AxisKind::Tempo,
            AxisKind::Tonnetz,
            AxisKind::Off,
        ] {
            let p = MatrixPlotParams::new(AxisKind::Time, kind);
            assert!(
                plot_matrix(data.view(), &p)
                    .unwrap()
                    .html()
                    .unwrap()
                    .contains("heatmap")
            );
        }
    }

    #[test]
    fn decibel_scales_respect_the_range() {
        let data = Array2::from_shape_fn((2, 2), |(i, j)| 10f64.powi(-((i * 2 + j) as i32) * 3));
        let mut p = MatrixPlotParams::new(AxisKind::Frames, AxisKind::Frames);
        p.value_scale = ValueScale::DbFullScale;
        p.top_db = Some(40.0);
        let v = scale_values(&data.view(), &p);
        let flat: Vec<f64> = v.into_iter().flatten().collect();
        assert_eq!(flat[0], 0.0);
        assert!(flat.iter().all(|&x| x >= -40.0));
        p.value_scale = ValueScale::DbPower(1.0);
        assert_eq!(scale_values(&data.view(), &p)[0][0], 0.0);
    }

    #[test]
    fn waveforms_validate() {
        let y = Array1::from_iter((0..1000).map(|i| (f64::from(i) * 0.1).sin()));
        assert!(plot_waveform_bars(y.view(), 1000.0, 10, 0.4).is_ok());
        assert!(plot_waveform_bars(y.view(), 1000.0, 0, 0.4).is_err());
        assert!(plot_waveform_bars(y.view(), 1000.0, 10, 1.0).is_err());
        let f0 = Array1::from_elem(10, 220.0);
        assert!(plot_waveform_f0(y.view(), f0.view(), 1000.0, 100, 12).is_ok());
        assert!(plot_waveform_f0(y.view(), f0.view(), 1000.0, 0, 12).is_err());
        let empty = Array2::<f64>::zeros((0, 0));
        let p = MatrixPlotParams::new(AxisKind::Frames, AxisKind::Frames);
        assert!(plot_matrix(empty.view(), &p).is_err());
    }
}
