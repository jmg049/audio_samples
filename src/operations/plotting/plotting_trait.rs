//! High-level plotting trait that produces per-channel visualizations.
//!
//! This module provides the `Plotting` trait which creates `PlotComposer` instances
//! with per-channel plots stacked vertically by default. This is the recommended
//! way to visualize audio data.
//!
//! For fine-grained control over individual plot elements, see `AudioPlotBuilders`.

use super::composer::PlotComposer;
use super::core::*;
use super::elements::*;
use crate::operations::types::WindowType;
use crate::{
    AudioSample, AudioSampleResult, AudioSamples, CastFrom, ConvertTo, I24, RealFloat, to_precision,
};

#[cfg(feature = "spectral-analysis")]
use crate::operations::traits::AudioTransforms;
#[cfg(feature = "spectral-analysis")]
use crate::operations::types::SpectrogramScale;

use crate::operations::traits::{AudioChannelOps, AudioStatistics};

/// Configuration for waveform plots.
#[derive(Debug, Clone)]
pub struct WaveformPlotConfig<F: RealFloat> {
    /// Line style for the waveform
    pub style: LineStyle<F>,
    /// Whether to show a shared time axis across channels
    pub shared_time_axis: bool,
}

impl<F: RealFloat> Default for WaveformPlotConfig<F> {
    fn default() -> Self {
        Self {
            style: LineStyle::default(),
            shared_time_axis: true,
        }
    }
}

/// Configuration for spectrum plots.
#[derive(Debug, Clone)]
pub struct SpectrumPlotConfig<F: RealFloat> {
    /// FFT size (default: 2048)
    pub n_fft: usize,
    /// Window function to apply
    pub window: WindowType<F>,
    /// Whether to display in dB scale
    pub db_scale: bool,
    /// Frequency range to display (Hz), None for full range
    pub freq_range: Option<(F, F)>,
    /// Line style for the spectrum
    pub style: LineStyle<F>,
}

impl<F: RealFloat> Default for SpectrumPlotConfig<F> {
    fn default() -> Self {
        Self {
            n_fft: 2048,
            window: WindowType::Hanning,
            db_scale: true,
            freq_range: None,
            style: LineStyle::default(),
        }
    }
}

/// Configuration for spectrogram plots.
/// Wraps the existing SpectrogramConfig with additional display options.
#[derive(Debug, Clone)]
pub struct SpectrogramPlotConfig<F: RealFloat> {
    /// FFT size for frequency analysis
    pub n_fft: usize,
    /// Hop length between frames (defaults to n_fft/4)
    pub hop_length: Option<usize>,
    /// Window function type
    pub window: WindowType<F>,
    /// Color palette for the spectrogram
    pub colormap: ColorPalette,
    /// Dynamic range in dB (min, max)
    pub db_range: (F, F),
    /// Whether to use logarithmic frequency scale
    pub log_freq: bool,
}

impl<F: RealFloat> Default for SpectrogramPlotConfig<F> {
    fn default() -> Self {
        Self {
            n_fft: 2048,
            hop_length: None,
            window: WindowType::Hanning,
            colormap: ColorPalette::Viridis,
            db_range: (to_precision::<F, _>(-80.0), F::zero()),
            log_freq: false,
        }
    }
}

impl<F: RealFloat> SpectrogramPlotConfig<F> {
    /// High resolution spectrogram preset
    pub fn high_resolution() -> Self {
        Self {
            n_fft: 4096,
            hop_length: Some(512),
            window: WindowType::Hanning,
            colormap: ColorPalette::Viridis,
            db_range: (to_precision::<F, _>(-80.0), F::zero()),
            log_freq: false,
        }
    }

    /// Fast spectrogram preset (lower resolution, faster computation)
    pub fn fast() -> Self {
        Self {
            n_fft: 1024,
            hop_length: Some(512),
            window: WindowType::Hanning,
            colormap: ColorPalette::Viridis,
            db_range: (to_precision::<F, _>(-60.0), F::zero()),
            log_freq: false,
        }
    }
}

/// Generate a channel label based on channel index and total channel count.
///
/// For common configurations:
/// - 1 channel: "Mono"
/// - 2 channels: "Left", "Right"
/// - 6 channels (5.1): "Front Left", "Front Right", "Center", "LFE", "Surround Left", "Surround Right"
/// - Other: "Channel 1", "Channel 2", etc.
pub fn channel_label(channel_index: usize, total_channels: usize) -> String {
    match total_channels {
        1 => "Mono".to_string(),
        2 => match channel_index {
            0 => "Left".to_string(),
            1 => "Right".to_string(),
            _ => format!("Channel {}", channel_index + 1),
        },
        6 => match channel_index {
            0 => "Front Left".to_string(),
            1 => "Front Right".to_string(),
            2 => "Center".to_string(),
            3 => "LFE".to_string(),
            4 => "Surround Left".to_string(),
            5 => "Surround Right".to_string(),
            _ => format!("Channel {}", channel_index + 1),
        },
        _ => format!("Channel {}", channel_index + 1),
    }
}

/// High-level plotting trait that produces `PlotComposer` with per-channel visualizations.
///
/// All methods return a `PlotComposer` configured with one subplot per channel,
/// stacked vertically. This allows immediate visualization via `.show()` or
/// further customization.
///
/// # Example
///
/// ```rust,ignore
/// use audio_samples::{AudioSamples, operations::plotting::Plotting};
///
/// let audio = AudioSamples::new_stereo(/* ... */);
///
/// // Plot waveform for all channels
/// audio.plot_waveform(None)?.show(true)?;
///
/// // Plot with custom config
/// let config = WaveformPlotConfig { /* ... */ };
/// audio.plot_waveform(Some(config))?.show(true)?;
/// ```
pub trait Plotting<T: AudioSample> {
    /// Plot waveform for all channels, stacked vertically.
    ///
    /// Each channel gets its own subplot with an appropriate label.
    fn plot_waveform<F>(
        &self,
        config: Option<WaveformPlotConfig<F>>,
    ) -> AudioSampleResult<PlotComposer<F>>
    where
        F: RealFloat + ConvertTo<T>,
        T: CastFrom<F> + ConvertTo<F>;

    /// Plot power spectrum for all channels, stacked vertically.
    #[cfg(feature = "spectral-analysis")]
    fn plot_spectrum<F>(
        &self,
        config: Option<SpectrumPlotConfig<F>>,
    ) -> AudioSampleResult<PlotComposer<F>>
    where
        F: RealFloat + ConvertTo<T>,
        T: ConvertTo<F>;

    /// Plot spectrogram for all channels, stacked vertically.
    #[cfg(feature = "spectral-analysis")]
    fn plot_spectrogram<F>(
        &self,
        config: Option<SpectrogramPlotConfig<F>>,
    ) -> AudioSampleResult<PlotComposer<F>>
    where
        F: RealFloat + ConvertTo<T>,
        T: ConvertTo<F>;
}

// Implementation for AudioSamples
#[cfg(feature = "spectral-analysis")]
impl<'a, T: AudioSample> Plotting<T> for AudioSamples<'a, T>
where
    i16: ConvertTo<T>,
    I24: ConvertTo<T>,
    i32: ConvertTo<T>,
    f32: ConvertTo<T>,
    f64: ConvertTo<T>,
    T: ConvertTo<f64>,
    f64: CastFrom<T>,
    for<'b> AudioSamples<'b, T>: AudioStatistics<'b, T> + AudioTransforms<T> + AudioChannelOps<T>,
{
    fn plot_waveform<F>(
        &self,
        config: Option<WaveformPlotConfig<F>>,
    ) -> AudioSampleResult<PlotComposer<F>>
    where
        F: RealFloat + ConvertTo<T>,
        T: CastFrom<F> + ConvertTo<F>,
    {
        let config = config.unwrap_or_default();
        let sample_rate = to_precision::<F, _>(self.sample_rate.get());
        let samples_per_channel = self.samples_per_channel();
        let n_channels = self.num_channels();

        // Generate time axis (shared across all channels)
        let time_data: Vec<F> = (0..samples_per_channel)
            .map(|i| to_precision::<F, _>(i) / sample_rate)
            .collect();

        let mut composer = PlotComposer::new();

        // Create a waveform plot for each channel
        for ch in 0..n_channels {
            let channel_data = self.extract_channel(ch)?;
            let amplitude_data: Vec<F> = channel_data
                .map_into(|x| to_precision::<F, _>(x))
                .to_interleaved_vec();

            let label = channel_label(ch, n_channels);
            let mut style = config.style.clone();
            // Use different colors for different channels
            style.color = ColorPalette::Default.get_color(ch);

            let metadata = PlotMetadata {
                title: Some(label.clone()),
                x_label: Some("Time (s)".to_string()),
                y_label: Some(format!("Amplitude ({})", T::LABEL)),
                legend_label: Some(label),
                z_order: 0,
            };

            let waveform =
                WaveformPlot::<F, T>::new(time_data.clone(), amplitude_data, style, metadata);
            composer = composer.add_element(waveform);
        }

        // Use vertical stack layout for multiple channels
        if n_channels > 1 {
            composer = composer.with_layout(LayoutConfig::VerticalStack);
        }

        composer = composer.with_title("Waveform");

        Ok(composer)
    }

    #[cfg(feature = "spectral-analysis")]
    fn plot_spectrum<F>(
        &self,
        config: Option<SpectrumPlotConfig<F>>,
    ) -> AudioSampleResult<PlotComposer<F>>
    where
        F: RealFloat + ConvertTo<T>,
        T: ConvertTo<F>,
    {
        let config = config.unwrap_or_default();
        let sample_rate = to_precision::<F, _>(self.sample_rate.get());
        let n_channels = self.num_channels();

        let mut composer = PlotComposer::new();

        // Process each channel separately
        for ch in 0..n_channels {
            let channel_data = self.extract_channel(ch)?;

            // Compute FFT for this channel
            let fft_result = channel_data.fft()?;
            let fft_len = fft_result.ncols();
            let n_bins = fft_len / 2; // Only positive frequencies

            // Generate frequency axis
            let frequencies: Vec<F> = (0..n_bins)
                .map(|i| to_precision::<F, _>(i) * sample_rate / to_precision::<F, _>(fft_len))
                .collect();

            // Extract magnitudes from the first (only) row of the mono channel FFT
            let magnitudes: Vec<F> = (0..n_bins)
                .map(|i| {
                    let complex_val = fft_result[[0, i]];
                    let magnitude: F = complex_val.norm();
                    if config.db_scale {
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
            let (filtered_freqs, filtered_mags) = if let Some((f_min, f_max)) = config.freq_range {
                let filtered: Vec<(F, F)> = frequencies
                    .into_iter()
                    .zip(magnitudes.into_iter())
                    .filter(|(f, _)| *f >= f_min && *f <= f_max)
                    .collect();
                let (f, m): (Vec<F>, Vec<F>) = filtered.into_iter().unzip();
                (f, m)
            } else {
                (frequencies, magnitudes)
            };

            let label = channel_label(ch, n_channels);
            let mut style = config.style.clone();
            style.color = ColorPalette::Default.get_color(ch);

            let metadata = PlotMetadata {
                title: Some(format!("Spectrum - {}", label)),
                x_label: Some("Frequency (Hz)".to_string()),
                y_label: Some(if config.db_scale {
                    "Magnitude (dB)".to_string()
                } else {
                    "Magnitude".to_string()
                }),
                legend_label: Some(label),
                z_order: 0,
            };

            let spectrum =
                PowerSpectrumPlot::<F, T>::new(filtered_freqs, filtered_mags, style, metadata);
            composer = composer.add_element(spectrum);
        }

        if n_channels > 1 {
            composer = composer.with_layout(LayoutConfig::VerticalStack);
        }

        composer = composer.with_title("Power Spectrum");

        Ok(composer)
    }

    #[cfg(feature = "spectral-analysis")]
    fn plot_spectrogram<F>(
        &self,
        config: Option<SpectrogramPlotConfig<F>>,
    ) -> AudioSampleResult<PlotComposer<F>>
    where
        F: RealFloat + ConvertTo<T>,
        T: ConvertTo<F>,
    {
        let config = config.unwrap_or_default();
        let n_fft = config.n_fft;
        let hop_length = config.hop_length.unwrap_or(n_fft / 4);
        let sample_rate = to_precision::<F, _>(self.sample_rate.get());
        let n_channels = self.num_channels();

        let mut composer = PlotComposer::new();

        // Process each channel separately
        for ch in 0..n_channels {
            let channel_data = self.extract_channel(ch)?;

            // Compute spectrogram for this channel
            let spectrogram_data = channel_data.spectrogram(
                n_fft,
                hop_length,
                config.window,
                SpectrogramScale::Log,
                true,
            )?;

            let (n_freq_bins, n_time_frames) = spectrogram_data.dim();

            // Generate time and frequency axes
            let time_axis: Vec<F> = (0..n_time_frames)
                .map(|i| to_precision::<F, _>(i) * to_precision::<F, _>(hop_length) / sample_rate)
                .collect();

            let freq_axis: Vec<F> = (0..n_freq_bins)
                .map(|i| to_precision::<F, _>(i) * sample_rate / to_precision::<F, _>(n_fft))
                .collect();

            let label = channel_label(ch, n_channels);

            let metadata = PlotMetadata {
                title: Some(format!("Spectrogram - {}", label)),
                x_label: Some("Time (s)".to_string()),
                y_label: Some("Frequency (Hz)".to_string()),
                legend_label: Some(label),
                z_order: 0,
            };

            let spec_config = SpectrogramConfig {
                n_fft,
                window_size: Some(n_fft),
                hop_length: Some(hop_length),
                window: config.window,
                colormap: config.colormap.clone(),
                db_range: config.db_range,
                log_freq: config.log_freq,
                mel_scale: false,
            };

            let spectrogram = SpectrogramPlot::new(
                spectrogram_data,
                time_axis,
                freq_axis,
                spec_config,
                metadata,
            );
            composer = composer.add_element(spectrogram);
        }

        if n_channels > 1 {
            composer = composer.with_layout(LayoutConfig::VerticalStack);
        }

        composer = composer.with_title("Spectrogram");

        Ok(composer)
    }
}

// Implementation without spectral-analysis feature
#[cfg(not(feature = "spectral-analysis"))]
impl<'a, T: AudioSample> Plotting<T> for AudioSamples<'a, T>
where
    i16: ConvertTo<T>,
    I24: ConvertTo<T>,
    i32: ConvertTo<T>,
    f32: ConvertTo<T>,
    f64: ConvertTo<T>,
    T: ConvertTo<f64>,
    f64: CastFrom<T>,
    for<'b> AudioSamples<'b, T>: AudioStatistics<'b, T> + AudioChannelOps<T>,
{
    fn plot_waveform<F>(
        &self,
        config: Option<WaveformPlotConfig<F>>,
    ) -> AudioSampleResult<PlotComposer<F>>
    where
        F: RealFloat + ConvertTo<T>,
        T: CastFrom<F> + ConvertTo<F>,
    {
        let config = config.unwrap_or_default();
        let sample_rate = to_precision::<F, _>(self.sample_rate.get());
        let samples_per_channel = self.samples_per_channel();
        let n_channels = self.num_channels();

        // Generate time axis (shared across all channels)
        let time_data: Vec<F> = (0..samples_per_channel)
            .map(|i| to_precision::<F, _>(i) / sample_rate)
            .collect();

        let mut composer = PlotComposer::new();

        // Create a waveform plot for each channel
        for ch in 0..n_channels {
            let channel_data = self.extract_channel(ch)?;
            let amplitude_data: Vec<F> = channel_data
                .map_into(|x| to_precision::<F, _>(x))
                .to_interleaved_vec();

            let label = channel_label(ch, n_channels);
            let mut style = config.style.clone();
            style.color = ColorPalette::Default.get_color(ch);

            let metadata = PlotMetadata {
                title: Some(label.clone()),
                x_label: Some("Time (s)".to_string()),
                y_label: Some(format!("Amplitude ({})", T::LABEL)),
                legend_label: Some(label),
                z_order: 0,
            };

            let waveform =
                WaveformPlot::<F, T>::new(time_data.clone(), amplitude_data, style, metadata);
            composer = composer.add_element(waveform);
        }

        if n_channels > 1 {
            composer = composer.with_layout(LayoutConfig::VerticalStack);
        }

        composer = composer.with_title("Waveform");

        Ok(composer)
    }
}
