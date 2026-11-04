//! Figures for the gallery chapter that no other chapter already has. Each one
//! repeats the calls of the example it illustrates and asserts what it shows.

use std::time::Duration;

use audio_samples::operations::AudioDecomposition;
use audio_samples::operations::hpss::HpssConfig;
use audio_samples::operations::types::ParametricEq;
use audio_samples::utils::generation::{impulse, sine_wave};
use audio_samples::{
    AudioEditing, AudioParametricEq, AudioSamples, AudioStatistics, ToneComponent, compound_tone,
    sample_rate,
};
use guide_figures::*;
use non_empty_slice::{NonEmptySlice, NonEmptyVec};

fn main() -> Result {
    let mut figs = Figures::new("ch15")?;
    hpss(&mut figs)?;
    parametric_eq(&mut figs)?;
    figs.finish()
}

/// Peak envelope per block, so that the SVG stays small.
fn envelope(x: &[f32], fs: f64, block: usize) -> (Vec<(f64, f64)>, Vec<(f64, f64)>) {
    let mut top = Vec::new();
    let mut bottom = Vec::new();
    for (k, chunk) in x.chunks(block).enumerate() {
        let t = (k * block) as f64 / fs;
        let hi = chunk.iter().copied().fold(f32::MIN, f32::max);
        let lo = chunk.iter().copied().fold(f32::MAX, f32::min);
        top.push((t, f64::from(hi)));
        bottom.push((t, f64::from(lo)));
    }
    (top, bottom)
}

/// `examples/hpss.rs`: a 440 Hz tone and an impulse every 0.5 s, separated.
fn hpss(figs: &mut Figures) -> Result {
    let sr = sample_rate!(44100);
    let fs = f64::from(sr.get());
    let duration = Duration::from_secs(3);
    let tone = sine_wave::<f32>(440.0, duration, sr, 0.6);
    let mut impulses = Vec::new();
    let mut t = 0.5;
    while t < 2.9 {
        impulses.push(impulse::<f32>(duration, sr, 0.8, t));
        t += 0.5;
    }
    let impulses = NonEmptyVec::new(impulses).map_err(|_| "no impulses")?;
    let clicks = AudioSamples::mix(&impulses, None)?;
    let sources = NonEmptyVec::new(vec![tone, clicks]).map_err(|_| "no sources")?;
    let mix = AudioSamples::mix(&sources, None)?;
    let (harmonic, percussive) = mix.hpss(&HpssConfig::default())?;

    check(
        "the harmonic part holds the tone's energy",
        harmonic.rms() > 10.0 * percussive.rms(),
    );
    check(
        "the percussive part is impulsive",
        f64::from(percussive.peak()) > 3.0 * percussive.rms(),
    );

    let parts = [
        ("Mix", mix.as_slice().ok_or("mono")?, BLUE),
        ("Harmonic", harmonic.as_slice().ok_or("mono")?, GREEN),
        (
            "Percussive",
            percussive.as_slice().ok_or("mono")?,
            VERMILION,
        ),
    ];
    figs.svg(
        "ch15-hpss",
        "The signal of the hpss example and its two parts from AudioDecomposition::hpss. \
         The tone stays in the harmonic part and the impulses every 0.5 s go to the percussive part. \
         Each curve is the peak envelope in blocks of 200 samples.",
        (720, 520),
        |area| {
            for (panel, (name, x, color)) in rows(area, 3).iter().zip(parts) {
                let (top, bottom) = envelope(x, fs, 200);
                let limit = x.iter().fold(0.0f32, |a, v| a.max(v.abs())).max(0.05);
                let limit = f64::from(limit) * 1.1;
                Chart::new("Time (s)", "Amplitude", (0.0, 3.0), (-limit, limit))
                    .title(name)
                    .add(Series::line("", top, color))
                    .add(Series::line("", bottom, color))
                    .draw(panel)?;
            }
            Ok(())
        },
    )
}

/// `examples/parametric_eq.rs`: the equaliser's response from `eq_frequency_response`.
fn parametric_eq(figs: &mut Figures) -> Result {
    let sr = sample_rate!(44100);
    let components = [
        ToneComponent::new(220.0, 1.0),
        ToneComponent::new(440.0, 0.6),
    ];
    let components = NonEmptySlice::from_slice(&components).ok_or("no components")?;
    let audio = compound_tone::<f64>(components, Duration::from_secs(1), sr);
    let eq = ParametricEq::three_band(120.0, -3.0, 1_000.0, 4.0, 1.2, 8_000.0, 2.0);
    let freqs: Vec<f64> = (0..400)
        .map(|i| 20.0 * 1000f64.powf(f64::from(i) / 399.0))
        .collect();
    let (mag, _) = audio.eq_frequency_response(&eq, &freqs)?;
    let curve: Vec<(f64, f64)> = freqs
        .iter()
        .zip(mag.iter())
        .map(|(f, m)| (*f, 20.0 * m.log10()))
        .collect();

    let at = |target: f64| {
        curve
            .iter()
            .min_by(|a, b| (a.0 - target).abs().total_cmp(&(b.0 - target).abs()))
            .map_or(f64::NAN, |p| p.1)
    };
    assert_close("low shelf well below 120 Hz", at(20.0), -3.0, 0.2);
    assert_close("peak at 1 kHz", at(1000.0), 4.0, 0.3);
    assert_close("high shelf well above 8 kHz", at(20_000.0), 2.0, 0.3);

    figs.svg(
        "ch15-eq",
        "Response of the three-band equaliser of the parametric_eq example, from \
         eq_frequency_response: a -3 dB low shelf at 120 Hz, a +4 dB peak at 1 kHz and a \
         +2 dB high shelf at 8 kHz.",
        SIZE,
        |area| {
            Chart::new("Frequency (Hz)", "Gain (dB)", (20.0, 20_000.0), (-6.0, 8.0))
                .log_x()
                .add(Series::line("", curve.clone(), BLUE))
                .draw(area)
        },
    )
}
