//! Integration tests for the `plotting` module.
//!
//! Each plot type is built from a known signal, rendered to an in-memory HTML
//! string via [`PlotUtils::html`], and checked for a non-empty Plotly document.
//! Nothing is written to disk.
//!
//! The `plotting` feature transitively enables `transforms`, so the spectrum,
//! phase-spectrum, and spectrogram plots (which run an FFT) are available
//! whenever this test compiles.
//!
//! Run with:
//! ```bash
//! cargo test --test plotting_integration --features plotting
//! ```

use std::time::Duration;

use audio_samples::operations::AudioPlotting;
use audio_samples::operations::plotting::{
    LissajousParams, PlotUtils, WaveformPlotParams, phase_spectrum::PhaseSpectrumParams,
    spectrograms::SpectrogramPlotParams, spectrum::MagnitudeSpectrumParams,
};
use audio_samples::{AudioSamples, sample_rate, sine_wave, stereo_sine_wave};

fn mono() -> AudioSamples<'static, f64> {
    sine_wave::<f64>(440.0, Duration::from_millis(250), sample_rate!(44100), 0.5)
}

fn stereo() -> AudioSamples<'static, f64> {
    stereo_sine_wave::<f64>(440.0, Duration::from_millis(250), sample_rate!(44100), 0.5)
}

/// Asserts the rendered HTML is a non-empty self-contained Plotly document.
fn assert_plotly_html(html: &str, what: &str) {
    assert!(!html.is_empty(), "{what} HTML was empty");
    assert!(
        html.contains("plotly"),
        "{what} HTML does not contain the expected Plotly marker"
    );
}

#[test]
fn waveform_renders_plotly_html() {
    let html = mono()
        .plot_waveform(&WaveformPlotParams::default())
        .unwrap()
        .html()
        .unwrap();
    assert_plotly_html(&html, "waveform");
}

#[test]
fn magnitude_spectrum_renders_plotly_html() {
    let html = mono()
        .plot_magnitude_spectrum(&MagnitudeSpectrumParams::db())
        .unwrap()
        .html()
        .unwrap();
    assert_plotly_html(&html, "magnitude spectrum");
}

#[test]
fn phase_spectrum_renders_plotly_html() {
    let html = mono()
        .plot_phase_spectrum(&PhaseSpectrumParams::new())
        .unwrap()
        .html()
        .unwrap();
    assert_plotly_html(&html, "phase spectrum");
}

#[test]
fn spectrogram_renders_plotly_html() {
    let html = mono()
        .plot_spectrogram(&SpectrogramPlotParams::mel_db())
        .unwrap()
        .html()
        .unwrap();
    assert_plotly_html(&html, "spectrogram");
}

#[test]
fn lissajous_renders_plotly_html_from_stereo() {
    let html = stereo()
        .plot_lissajous(&LissajousParams::new())
        .unwrap()
        .html()
        .unwrap();
    assert_plotly_html(&html, "lissajous");
}

#[test]
fn lissajous_rejects_mono() {
    // Lissajous requires exactly two channels.
    let result = mono().plot_lissajous(&LissajousParams::new());
    assert!(result.is_err(), "lissajous should reject a mono signal");
}
