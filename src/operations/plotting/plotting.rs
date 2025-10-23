use crate::operations::traits::AudioChannelOps;
use crate::{AudioPlottingUtils, AudioSample, AudioSamples, ConvertTo, I24};
use plotters::prelude::*;

/// Result type for plotting operations
pub type PlotResult<T> = Result<T, Box<dyn std::error::Error>>;

/// Configuration for waveform plots
#[derive(Debug, Clone)]
pub struct WaveformPlotOptions {
    pub figsize: (u32, u32),
    pub title: String,
    pub xlabel: String,
    pub ylabel: String,
    pub wave_color: RGBColor,
    pub background_color: RGBColor,
    pub grid: bool,
    pub grid_color: RGBColor,
    pub font_size: u32,
    pub line_width: u32,
    pub save_path: Option<String>,
}

impl Default for WaveformPlotOptions {
    fn default() -> Self {
        Self {
            figsize: (800, 600),
            title: "Waveform".to_string(),
            xlabel: "Time (s)".to_string(),
            ylabel: "Amplitude".to_string(),
            wave_color: BLUE,
            background_color: WHITE,
            grid: true,
            grid_color: RGBColor(220, 220, 220),
            font_size: 12,
            line_width: 2,
            save_path: None,
        }
    }
}

/// Configuration for spectrogram plots
#[derive(Debug, Clone)]
pub struct SpectrogramPlotOptions {
    pub figsize: (u32, u32),
    pub title: String,
    pub xlabel: String,
    pub ylabel: String,
    pub colormap: String,
    pub background_color: RGBColor,
    pub font_size: u32,
    pub save_path: Option<String>,
    pub n_fft: usize,
    pub hop_length: Option<usize>,
    pub window: String,
    pub log_scale: bool,
    pub db_range: (f64, f64),
}

impl Default for SpectrogramPlotOptions {
    fn default() -> Self {
        Self {
            figsize: (1000, 800),
            title: "Spectrogram".to_string(),
            xlabel: "Time (s)".to_string(),
            ylabel: "Frequency (Hz)".to_string(),
            colormap: "viridis".to_string(),
            background_color: WHITE,
            font_size: 12,
            save_path: None,
            n_fft: 2048,
            hop_length: None,
            window: "hann".to_string(),
            log_scale: true,
            db_range: (-80.0, 0.0),
        }
    }
}

/// Configuration for comparison plots
#[derive(Debug, Clone)]
pub struct ComparisonPlotOptions {
    pub figsize: (u32, u32),
    pub title: String,
    pub xlabel: String,
    pub ylabel: String,
    pub colors: Vec<RGBColor>,
    pub background_color: RGBColor,
    pub grid: bool,
    pub grid_color: RGBColor,
    pub font_size: u32,
    pub line_width: u32,
    pub legend: bool,
    pub save_path: Option<String>,
}

impl Default for ComparisonPlotOptions {
    fn default() -> Self {
        Self {
            figsize: (1000, 700),
            title: "Audio Comparison".to_string(),
            xlabel: "Time (s)".to_string(),
            ylabel: "Amplitude".to_string(),
            colors: vec![BLUE, RED, GREEN, MAGENTA, CYAN],
            background_color: WHITE,
            grid: true,
            grid_color: RGBColor(220, 220, 220),
            font_size: 12,
            line_width: 2,
            legend: true,
            save_path: None,
        }
    }
}

/// Choose a "nice" step in seconds given a rough desired step.
/// Uses 1–2–5 × 10^k progression.
fn nice_step_seconds(rough: f64) -> f64 {
    assert!(rough.is_finite() && rough > 0.0);
    let exp = rough.log10().floor();
    let base = 10f64.powf(exp);
    let mant = rough / base;
    let nice_mant = if mant <= 1.0 {
        1.0
    } else if mant <= 2.0 {
        2.0
    } else if mant <= 5.0 {
        5.0
    } else {
        10.0
    };
    nice_mant * base
}

/// Format time values for axis labels
fn format_time_label(t: f64) -> String {
    if t >= 3600.0 {
        let h = (t / 3600.0) as u32;
        let m = ((t % 3600.0) / 60.0) as u32;
        let s = t % 60.0;
        format!("{}:{:02}:{:06.3}", h, m, s)
    } else if t >= 60.0 {
        let m = (t / 60.0) as u32;
        let s = t % 60.0;
        if t < 600.0 {
            format!("{}:{:06.3}", m, s)
        } else {
            format!("{}:{:05.2}", m, s)
        }
    } else if t < 10.0 {
        format!("{:.3}", t)
    } else {
        format!("{:.2}", t)
    }
}

/// Generate nice time ticks for plotting
fn generate_time_ticks(duration: f64, target_ticks: usize) -> Vec<f64> {
    let clock_steps = [
        0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 15.0, 20.0,
        30.0, 60.0, 120.0, 180.0, 300.0, 600.0, 900.0, 1200.0, 1800.0, 3600.0, 7200.0, 10800.0,
    ];

    let desired_intervals = (target_ticks - 1).max(1);
    let mut best_step = clock_steps[0];
    let mut best_cost = f64::INFINITY;

    for &step in &clock_steps {
        let intervals = (duration / step).ceil() as usize;
        let cost = (intervals as f64 - desired_intervals as f64).abs();
        if cost < best_cost || (cost == best_cost && step < best_step) {
            best_cost = cost;
            best_step = step;
        }
    }

    let n_intervals = (duration / best_step).ceil() as usize;
    (0..=n_intervals).map(|k| k as f64 * best_step).collect()
}

/// Normalize audio data to [-1, 1] range for plotting
fn normalize_for_plotting<T: AudioSample>(data: &[T]) -> Vec<f64>
where
    T: ConvertTo<f64>,
{
    data.iter()
        .map(|&sample| sample.convert_to().unwrap_or(0.0))
        .collect()
}
/// Seconds from 0 to duration with ~target_ticks "nice" spacing (1–2–5).
pub fn time_ticks_seconds<T: AudioSample>(audio: &AudioSamples<T>, target_ticks: usize) -> Vec<f64>
where
    i16: ConvertTo<T>,
    I24: ConvertTo<T>,
    i32: ConvertTo<T>,
    f32: ConvertTo<T>,
    f64: ConvertTo<T>,
{
    let duration = audio.duration_seconds();
    generate_time_ticks(duration, target_ticks)
}

/// Plot a waveform using plotters
pub fn plot_waveform<T: AudioSample>(
    audio: &AudioSamples<T>,
    options: WaveformPlotOptions,
) -> PlotResult<()>
where
    i16: ConvertTo<T>,
    I24: ConvertTo<T>,
    i32: ConvertTo<T>,
    f32: ConvertTo<T>,
    f64: ConvertTo<T>,
    T: ConvertTo<f64>,
{
    let output_path = options.save_path.as_deref().unwrap_or("waveform.png");
    let root = BitMapBackend::new(output_path, options.figsize).into_drawing_area();
    root.fill(&options.background_color)?;

    let num_channels = audio.num_channels();
    let duration = audio.duration_seconds();
    let sample_rate = audio.sample_rate() as f64;

    // For single channel, use the full area; for multi-channel, split vertically
    if num_channels == 1 {
        // Extract channel data
        let channel_data = audio.extract_channel(0).unwrap();
        let samples: Vec<f64> = normalize_for_plotting(channel_data.data.as_slice().unwrap());

        // Create time axis
        let time_axis: Vec<f64> = (0..samples.len()).map(|i| i as f64 / sample_rate).collect();

        // Find amplitude range
        let min_amp = samples.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max_amp = samples.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let amp_range = (max_amp - min_amp).max(0.1);
        let y_margin = amp_range * 0.05;

        let mut chart = ChartBuilder::on(&root)
            .caption(&options.title, ("Arial", options.font_size))
            .margin(10)
            .x_label_area_size(40)
            .y_label_area_size(50)
            .build_cartesian_2d(0.0..duration, (min_amp - y_margin)..(max_amp + y_margin))?;

        if options.grid {
            chart
                .configure_mesh()
                .x_desc(&options.xlabel)
                .y_desc(&options.ylabel)
                .x_label_formatter(&|x| format_time_label(*x))
                .y_label_formatter(&|y| format!("{:.3}", y))
                .draw()?;
        } else {
            chart
                .configure_mesh()
                .x_desc(&options.xlabel)
                .y_desc(&options.ylabel)
                .x_label_formatter(&|x| format_time_label(*x))
                .y_label_formatter(&|y| format!("{:.3}", y))
                .disable_mesh()
                .draw()?;
        }

        // Plot the waveform
        let data_points: Vec<(f64, f64)> = time_axis.into_iter().zip(samples.into_iter()).collect();

        chart.draw_series(LineSeries::new(data_points, &options.wave_color))?;
    } else {
        // Multi-channel plot - for now, plot all channels overlaid
        // Find global amplitude range
        let mut global_min = f64::INFINITY;
        let mut global_max = f64::NEG_INFINITY;

        for ch in 0..num_channels {
            let channel_data = audio.extract_channel(ch).unwrap();
            let samples = normalize_for_plotting(channel_data.data.as_slice().unwrap());
            let min_amp = samples.iter().fold(f64::INFINITY, |a, &b| a.min(b));
            let max_amp = samples.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
            global_min = global_min.min(min_amp);
            global_max = global_max.max(max_amp);
        }

        let amp_range = (global_max - global_min).max(0.1);
        let y_margin = amp_range * 0.05;

        let mut chart = ChartBuilder::on(&root)
            .caption(&options.title, ("Arial", options.font_size))
            .margin(10)
            .x_label_area_size(40)
            .y_label_area_size(50)
            .build_cartesian_2d(
                0.0..duration,
                (global_min - y_margin)..(global_max + y_margin),
            )?;

        if options.grid {
            chart
                .configure_mesh()
                .x_desc(&options.xlabel)
                .y_desc(&options.ylabel)
                .x_label_formatter(&|x| format_time_label(*x))
                .y_label_formatter(&|y| format!("{:.3}", y))
                .draw()?;
        } else {
            chart
                .configure_mesh()
                .x_desc(&options.xlabel)
                .y_desc(&options.ylabel)
                .x_label_formatter(&|x| format_time_label(*x))
                .y_label_formatter(&|y| format!("{:.3}", y))
                .disable_mesh()
                .draw()?;
        }

        // Plot each channel
        let colors = [&RED, &BLUE, &GREEN, &MAGENTA, &CYAN];
        for ch in 0..num_channels {
            let channel_data = audio.extract_channel(ch).unwrap();
            let samples: Vec<f64> = normalize_for_plotting(channel_data.data.as_slice().unwrap());

            let time_axis: Vec<f64> = (0..samples.len()).map(|i| i as f64 / sample_rate).collect();

            let data_points: Vec<(f64, f64)> =
                time_axis.into_iter().zip(samples.into_iter()).collect();

            let color = colors[ch % colors.len()];
            chart
                .draw_series(LineSeries::new(data_points, color))?
                .label(format!("Channel {}", ch + 1))
                .legend(move |(x, y)| PathElement::new(vec![(x, y), (x + 10, y)], color));
        }

        chart.configure_series_labels().draw()?;
    }

    root.present()?;
    Ok(())
}

/// Plot a spectrogram using plotters
pub fn plot_spectrogram<T: AudioSample>(
    _audio: &AudioSamples<T>,
    _options: SpectrogramPlotOptions,
) -> PlotResult<()>
where
    i16: ConvertTo<T>,
    I24: ConvertTo<T>,
    i32: ConvertTo<T>,
    f32: ConvertTo<T>,
    f64: ConvertTo<T>,
{
    // For now, return an error indicating this needs FFT implementation
    Err("Spectrogram plotting requires FFT implementation - not yet available".into())
}

/// Compare multiple audio waveforms
pub fn plot_comparison<T: AudioSample>(
    audio_samples: &[&AudioSamples<T>],
    labels: &[String],
    options: ComparisonPlotOptions,
) -> PlotResult<()>
where
    i16: ConvertTo<T>,
    I24: ConvertTo<T>,
    i32: ConvertTo<T>,
    f32: ConvertTo<T>,
    f64: ConvertTo<T>,
    T: ConvertTo<f64>,
{
    if audio_samples.is_empty() {
        return Err("At least one audio sample is required".into());
    }

    if audio_samples.len() != labels.len() {
        return Err("Number of labels must match number of audio samples".into());
    }

    let output_path = options.save_path.as_deref().unwrap_or("comparison.png");
    let root = BitMapBackend::new(output_path, options.figsize).into_drawing_area();
    root.fill(&options.background_color)?;

    let num_channels = audio_samples[0].num_channels();

    // Find the maximum duration for x-axis
    let max_duration = audio_samples
        .iter()
        .map(|audio| audio.duration_seconds())
        .fold(0.0, f64::max);

    // Find global amplitude range
    let mut global_min = f64::INFINITY;
    let mut global_max = f64::NEG_INFINITY;

    for audio in audio_samples {
        for ch in 0..num_channels.min(audio.num_channels()) {
            let channel_data = audio.extract_channel(ch).unwrap();
            let samples = normalize_for_plotting(channel_data.data.as_slice().unwrap());
            let min_amp = samples.iter().fold(f64::INFINITY, |a, &b| a.min(b));
            let max_amp = samples.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
            global_min = global_min.min(min_amp);
            global_max = global_max.max(max_amp);
        }
    }

    let amp_range = (global_max - global_min).max(0.1);
    let y_margin = amp_range * 0.05;

    let mut chart = ChartBuilder::on(&root)
        .caption(&options.title, ("Arial", options.font_size))
        .margin(10)
        .x_label_area_size(40)
        .y_label_area_size(50)
        .build_cartesian_2d(
            0.0..max_duration,
            (global_min - y_margin)..(global_max + y_margin),
        )?;

    if options.grid {
        chart
            .configure_mesh()
            .x_desc(&options.xlabel)
            .y_desc(&options.ylabel)
            .x_label_formatter(&|x| format_time_label(*x))
            .y_label_formatter(&|y| format!("{:.3}", y))
            .draw()?;
    } else {
        chart
            .configure_mesh()
            .x_desc(&options.xlabel)
            .y_desc(&options.ylabel)
            .x_label_formatter(&|x| format_time_label(*x))
            .y_label_formatter(&|y| format!("{:.3}", y))
            .disable_mesh()
            .draw()?;
    }

    // Plot each audio sample (only first channel for simplicity)
    for (i, audio) in audio_samples.iter().enumerate() {
        let channel_data = audio.extract_channel(0).unwrap();
        let samples = normalize_for_plotting(channel_data.data.as_slice().unwrap());
        let sample_rate = audio.sample_rate() as f64;

        let time_axis: Vec<f64> = (0..samples.len())
            .map(|idx| idx as f64 / sample_rate)
            .collect();

        let data_points: Vec<(f64, f64)> = time_axis.into_iter().zip(samples.into_iter()).collect();

        let color = &options.colors[i % options.colors.len()];

        chart
            .draw_series(LineSeries::new(data_points, color))?
            .label(&labels[i])
            .legend(move |(x, y)| PathElement::new(vec![(x, y), (x + 10, y)], color));
    }

    if options.legend {
        chart.configure_series_labels().draw()?;
    }

    root.present()?;
    Ok(())
}

/// Plot the difference between two audio samples
pub fn plot_difference<T: AudioSample>(
    audio1: &AudioSamples<T>,
    audio2: &AudioSamples<T>,
    labels: (&str, &str),
    options: WaveformPlotOptions,
) -> PlotResult<()>
where
    i16: ConvertTo<T>,
    I24: ConvertTo<T>,
    i32: ConvertTo<T>,
    f32: ConvertTo<T>,
    f64: ConvertTo<T>,
    T: ConvertTo<f64>,
{
    if audio1.num_channels() != audio2.num_channels() {
        return Err("Both audio samples must have the same number of channels".into());
    }

    if audio1.sample_rate() != audio2.sample_rate() {
        return Err("Both audio samples must have the same sample rate".into());
    }

    let output_path = options.save_path.as_deref().unwrap_or("difference.png");
    let root = BitMapBackend::new(output_path, options.figsize).into_drawing_area();
    root.fill(&options.background_color)?;

    let sample_rate = audio1.sample_rate() as f64;

    // Extract channel data (only first channel for simplicity)
    let data1 = audio1.extract_channel(0).unwrap();
    let data2 = audio2.extract_channel(0).unwrap();

    let samples1 = normalize_for_plotting(data1.data.as_slice().unwrap());
    let samples2 = normalize_for_plotting(data2.data.as_slice().unwrap());

    // Compute difference (truncate to shorter length)
    let min_len = samples1.len().min(samples2.len());
    let diff: Vec<f64> = (0..min_len).map(|i| samples1[i] - samples2[i]).collect();

    let duration = min_len as f64 / sample_rate;

    // Find amplitude range for difference
    let min_diff = diff.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let max_diff = diff.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    let diff_range = (max_diff - min_diff).max(0.1);
    let y_margin = diff_range * 0.05;

    let mut chart = ChartBuilder::on(&root)
        .caption(
            format!("Difference: {} - {}", labels.0, labels.1),
            ("Arial", options.font_size),
        )
        .margin(10)
        .x_label_area_size(40)
        .y_label_area_size(50)
        .build_cartesian_2d(0.0..duration, (min_diff - y_margin)..(max_diff + y_margin))?;

    if options.grid {
        chart
            .configure_mesh()
            .x_desc(&options.xlabel)
            .y_desc("Difference")
            .x_label_formatter(&|x| format_time_label(*x))
            .y_label_formatter(&|y| format!("{:.3}", y))
            .draw()?;
    } else {
        chart
            .configure_mesh()
            .x_desc(&options.xlabel)
            .y_desc("Difference")
            .x_label_formatter(&|x| format_time_label(*x))
            .y_label_formatter(&|y| format!("{:.3}", y))
            .disable_mesh()
            .draw()?;
    }

    // Create time axis and plot difference
    let time_axis: Vec<f64> = (0..diff.len()).map(|i| i as f64 / sample_rate).collect();

    let data_points: Vec<(f64, f64)> = time_axis.into_iter().zip(diff.into_iter()).collect();

    chart.draw_series(LineSeries::new(data_points, &options.wave_color))?;

    root.present()?;
    Ok(())
}

impl<T: AudioSample> AudioPlottingUtils<T> for AudioSamples<T>
where
    i16: ConvertTo<T>,
    I24: ConvertTo<T>,
    i32: ConvertTo<T>,
    f32: ConvertTo<T>,
    f64: ConvertTo<T>,
{
    /// Return a time axis (in seconds) with a sensible spacing for plotting.
    /// If `step_seconds` is `Some(dt)`, it is used directly; otherwise, a "nice" dt is chosen
    /// so there are roughly `TARGET_POINTS` points across the clip.
    ///
    /// Invariants:
    /// - spacing is in seconds
    /// - internal stepping uses whole samples (>= 1)
    fn time_axis(&self, step_seconds: Option<f64>) -> Vec<f64> {
        let samples_per_channel = self.samples_per_channel();
        if samples_per_channel == 0 {
            return Vec::new();
        }

        let sr = self.sample_rate() as f64;
        assert!(
            sr.is_finite() && sr > 0.0,
            "sample_rate must be positive and finite"
        );

        let duration = self.duration_seconds(); // should equal samples_per_channel as f64 / sr

        // If the caller didn’t specify a step, pick a "nice" one so we get ~TARGET_POINTS samples.
        const TARGET_POINTS: usize = 20;

        let dt = match step_seconds {
            Some(dt) => dt.max(1.0 / sr), // at least one sample
            None => {
                let rough = duration / TARGET_POINTS as f64;
                nice_step_seconds(rough)
            }
        };

        // Convert to an integer sample stride (at least 1).
        let step_samples = (dt * sr).round().max(1.0) as usize;

        // Number of points we’ll generate (ensure we don’t overflow).
        let n = (samples_per_channel + step_samples - 1) / step_samples;

        // Build the axis using integer math for indices.
        let mut axis = Vec::with_capacity(n + 1);
        let mut idx = 0usize;
        while idx < samples_per_channel {
            axis.push(idx as f64 / sr);
            idx = idx.saturating_add(step_samples);
        }
        // Ensure the last point (duration) is present for clean axes.
        if let Some(&last) = axis.last() {
            if (duration - last) > (0.5 / sr) {
                axis.push(duration);
            }
        }
        axis
    }

    fn frequency_axis(&self) -> Vec<T> {
        todo!()
    }

    fn time_ticks_seconds(&self, target_ticks: usize) -> Vec<f64> {
        time_ticks_seconds(self, target_ticks)
    }

    /// Plot waveform with default options
    fn plot_waveform(&self) -> PlotResult<()> {
        plot_waveform(self, WaveformPlotOptions::default())
    }

    /// Plot waveform with custom options
    fn plot_waveform_with_options(&self, options: WaveformPlotOptions) -> PlotResult<()> {
        plot_waveform(self, options)
    }

    /// Plot spectrogram with default options
    fn plot_spectrogram(&self) -> PlotResult<()> {
        plot_spectrogram(self, SpectrogramPlotOptions::default())
    }

    /// Plot spectrogram with custom options
    fn plot_spectrogram_with_options(&self, options: SpectrogramPlotOptions) -> PlotResult<()> {
        plot_spectrogram(self, options)
    }
}

/// Convenience functions for quick plotting
impl<T: AudioSample> AudioSamples<T>
where
    i16: ConvertTo<T>,
    I24: ConvertTo<T>,
    i32: ConvertTo<T>,
    f32: ConvertTo<T>,
    f64: ConvertTo<T>,
{
    /// Quick waveform plot with default settings
    pub fn quick_plot(&self, output_path: &str) -> PlotResult<()> {
        let mut options = WaveformPlotOptions::default();
        options.save_path = Some(output_path.to_string());
        plot_waveform(self, options)
    }

    /// Quick waveform plot with custom title
    pub fn quick_plot_with_title(&self, output_path: &str, title: &str) -> PlotResult<()> {
        let mut options = WaveformPlotOptions::default();
        options.save_path = Some(output_path.to_string());
        options.title = title.to_string();
        plot_waveform(self, options)
    }

    /// Compare this audio with another
    pub fn compare_with(&self, other: &AudioSamples<T>, output_path: &str) -> PlotResult<()> {
        let mut options = ComparisonPlotOptions::default();
        options.save_path = Some(output_path.to_string());

        let audio_samples = vec![self, other];
        let labels = vec!["Original".to_string(), "Comparison".to_string()];

        plot_comparison(&audio_samples, &labels, options)
    }

    /// Plot difference with another audio sample
    pub fn plot_diff_with(&self, other: &AudioSamples<T>, output_path: &str) -> PlotResult<()> {
        let mut options = WaveformPlotOptions::default();
        options.save_path = Some(output_path.to_string());

        plot_difference(self, other, ("Original", "Modified"), options)
    }
}
