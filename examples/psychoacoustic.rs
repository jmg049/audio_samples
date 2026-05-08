#[cfg(not(feature = "psychoacoustic"))]
fn main() {
    eprintln!("error: This example requires the `psychoacoustic` feature.");
    std::process::exit(1);
}

#[cfg(feature = "psychoacoustic")]
pub fn main() -> audio_samples::AudioSampleResult<()> {
    use std::num::NonZeroUsize;
    use std::time::Duration;

    use audio_samples::{
        AudioPerceptualAnalysis, BandLayout, PsychoacousticConfig, sample_rate, sine_wave,
    };
    use non_empty_slice::NonEmptySlice;
    use spectrograms::WindowType;

    let sr = sample_rate!(44100);

    // A 440 Hz tone (A4) mixed with its second harmonic (A5) at lower amplitude.
    // The 880 Hz component should show up as less important once masked by 440 Hz.
    let fundamental = sine_wave::<f32>(440.0, Duration::from_millis(200), sr, 0.8);
    let harmonic = sine_wave::<f32>(880.0, Duration::from_millis(200), sr, 0.2);
    let signal = fundamental + harmonic;

    println!("=== Psychoacoustic Analysis ===");
    println!(
        "Signal: {} samples @ {} Hz",
        signal.samples_per_channel().get(),
        signal.sample_rate().get()
    );

    // ── Bark-scale layout ──────────────────────────────────────────────────
    // 24 Bark critical bands from 0 Hz to Nyquist, mapped onto 1024 MDCT bins.
    let n_bands = NonZeroUsize::new(24).unwrap();
    let n_bins = NonZeroUsize::new(1024).unwrap();
    let bark_layout = BandLayout::bark(n_bands, 44100.0, n_bins);

    // ── Config ─────────────────────────────────────────────────────────────
    // Uniform perceptual weights; real codec profiles weight mid-range bands higher.
    let weights = vec![1.0_f32; 24];
    let weights_slice = NonEmptySlice::from_slice(&weights).unwrap();
    let config = PsychoacousticConfig::new(
        -60.0, // noise_floor dB
        14.5,  // masking_gain dB  (MPEG-1 reference value)
        0.4,   // masker_exponent  (reserved)
        25.0,  // upward_spread    dB/Bark
        6.0,   // downward_spread  dB/Bark
        weights_slice,
        1e-10, // epsilon
    );

    // ── Analysis ───────────────────────────────────────────────────────────
    let result = signal.analyse_psychoacoustic(WindowType::Hanning, &bark_layout, &config)?;

    println!(
        "\nMDCT coefficients: {} total",
        result.coefficients.len().get()
    );
    println!(
        "Spectral bins:     {} (avg energy per bin)",
        result.bin_energies.len().get()
    );
    println!("Bands analysed:    {}", result.band_metrics.len().get());

    // ── Per-band metrics ───────────────────────────────────────────────────
    println!(
        "\n{:<6} {:<12} {:<10} {:<8} {:<10} {:<10}",
        "Band", "CentreHz", "Energy dB", "SMR dB", "Importance", "AllowNoise"
    );
    println!("{}", "-".repeat(60));

    for (i, metric) in result.band_metrics.as_slice().iter().enumerate() {
        println!(
            "{:<6} {:<12.1} {:<10.2} {:<8.2} {:<10.4} {:<10.2}",
            i,
            metric.band.centre_frequency,
            metric.energy,
            metric.signal_to_mask_ratio,
            metric.importance,
            metric.allowed_noise,
        );
    }

    // ── Top bands by importance ────────────────────────────────────────────
    let mut ranked: Vec<(usize, f32)> = result
        .band_metrics
        .as_slice()
        .iter()
        .enumerate()
        .map(|(i, m)| (i, m.importance))
        .collect();
    ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    println!("\nTop 5 bands by perceptual importance:");
    for (rank, (band_idx, importance)) in ranked.iter().take(5).enumerate() {
        let metric = &result.band_metrics.as_slice()[*band_idx];
        println!(
            "  #{}: band {} ({:.0} Hz) — importance={:.4}, SMR={:.2} dB",
            rank + 1,
            band_idx,
            metric.band.centre_frequency,
            importance,
            metric.signal_to_mask_ratio,
        );
    }

    // ── Mel-scale layout comparison ────────────────────────────────────────
    println!("\n=== Mel-Scale Layout (40 bands) ===");
    let mel_layout = BandLayout::mel(NonZeroUsize::new(40).unwrap(), 44100.0, n_bins);

    let mel_weights = vec![1.0_f32; 40];
    let mel_weights_slice = NonEmptySlice::from_slice(&mel_weights).unwrap();
    let mel_config =
        PsychoacousticConfig::new(-60.0, 14.5, 0.4, 25.0, 6.0, mel_weights_slice, 1e-10);

    let mel_result =
        signal.analyse_psychoacoustic(WindowType::Hanning, &mel_layout, &mel_config)?;
    println!(
        "Mel analysis: {} bands, {} MDCT coefficients",
        mel_result.band_metrics.len().get(),
        mel_result.coefficients.len().get(),
    );

    // Band whose SMR is highest on the Mel layout (most audible / needs most bits).
    let most_audible = mel_result
        .band_metrics
        .as_slice()
        .iter()
        .max_by(|a, b| {
            a.signal_to_mask_ratio
                .partial_cmp(&b.signal_to_mask_ratio)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .unwrap();
    println!(
        "Most audible Mel band: {:.0} Hz — SMR={:.2} dB",
        most_audible.band.centre_frequency, most_audible.signal_to_mask_ratio,
    );

    Ok(())
}
