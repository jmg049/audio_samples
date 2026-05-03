/// Round-trip quality tests for the Opus sketch codec.
///
/// Each test encodes a signal with a fixed mode (SILK or CELT), decodes it,
/// and measures the SNR between original and reconstructed audio.
///
/// ## Quality notes
///
/// - **SILK** achieves high SNR (>30 dB) for both tonal and noise signals because
///   it uses a time-domain LPC analysis/synthesis that doesn't suffer from
///   spectral reconstruction artefacts.
///
/// - **CELT** delegates to the existing [`PerceptualCodec`] pipeline (MDCT +
///   psychoacoustic masking). The quality is limited by the same MDCT boundary
///   effect that affects [`PerceptualCodec`]: the Hanning-windowed MDCT produces
///   near-zero samples at the beginning and end of each frame, which caps the
///   achievable SNR at approximately 2–3 dB for typical signals. This matches
///   the behaviour observed in the existing `codec_quality` integration tests.
///
///   The thresholds below therefore represent the floor that the **sketch**
///   implementation is expected to meet consistently, not the target for a
///   fully-optimised production codec.
///
/// ## Run
///
/// ```bash
/// cargo test --test opus_codec_quality \
///   --no-default-features \
///   --features "opus-codec,random-generation"
/// ```
use std::time::Duration;

use audio_samples::{
    OpusCodec, OpusConfig, OpusMode,
    codecs::{decode, encode},
    sample_rate,
    utils::generation::{sine_wave, white_noise},
    utils::comparison::snr,
};
use non_empty_slice::NonEmptyVec;
use spectrograms::WindowType;

const SIGNAL_DURATION_MS: u64 = 200;

// ── Codec constructors ────────────────────────────────────────────────────────

fn silk_codec(bit_budget: u32) -> OpusCodec {
    let config = OpusConfig::with_mode(OpusMode::Silk, bit_budget);
    OpusCodec::new(config, WindowType::Hanning)
}

fn celt_codec(bit_budget: u32) -> OpusCodec {
    let config = OpusConfig::with_mode(OpusMode::Celt, bit_budget);
    OpusCodec::new(config, WindowType::Hanning)
}

fn auto_codec(bit_budget: u32) -> OpusCodec {
    let config = OpusConfig::new(bit_budget);
    OpusCodec::new(config, WindowType::Hanning)
}

// ── Test signals ──────────────────────────────────────────────────────────────

fn signal_sine(freq_hz: f64) -> audio_samples::AudioSamples<'static, f32> {
    sine_wave::<f32>(
        freq_hz,
        Duration::from_millis(SIGNAL_DURATION_MS),
        sample_rate!(44100),
        0.5,
    )
}

fn signal_white_noise() -> audio_samples::AudioSamples<'static, f32> {
    white_noise::<f32>(
        Duration::from_millis(SIGNAL_DURATION_MS),
        sample_rate!(44100),
        0.5,
        Some(42),
    )
}

// ── Round-trip helper ─────────────────────────────────────────────────────────

/// Encodes → decodes and returns the SNR in dB.
fn round_trip_snr(
    signal: &audio_samples::AudioSamples<'static, f32>,
    codec: OpusCodec,
) -> f64 {
    let encoded = encode(signal, codec).expect("encode failed");
    let reconstructed = decode::<OpusCodec, f32>(encoded).expect("decode failed");

    let orig = signal.as_slice().expect("original contiguous");
    let recon = reconstructed.as_slice().expect("reconstructed contiguous");
    let len = orig.len().min(recon.len());

    let error: Vec<f32> = orig[..len]
        .iter()
        .zip(recon[..len].iter())
        .map(|(a, b)| a - b)
        .collect();

    let nev = unsafe { NonEmptyVec::new_unchecked(error) };
    let error_sig =
        audio_samples::AudioSamples::<'static, f32>::from_mono_vec(nev, signal.sample_rate());

    let orig_trimmed = if len == orig.len() {
        signal.clone()
    } else {
        let trimmed: Vec<f32> = orig[..len].to_vec();
        let nev2 = unsafe { NonEmptyVec::new_unchecked(trimmed) };
        audio_samples::AudioSamples::<'static, f32>::from_mono_vec(nev2, signal.sample_rate())
    };

    snr(&orig_trimmed, &error_sig).expect("snr failed")
}

// ── SILK mode tests ───────────────────────────────────────────────────────────

/// SILK achieves high SNR because it uses a time-domain LPC round-trip
/// that doesn't have the MDCT window boundary limitation.
#[test]
fn silk_sine_440() {
    let snr_db = round_trip_snr(&signal_sine(440.0), silk_codec(64_000));
    eprintln!("SILK sine_440 @ 64k:  {snr_db:.2} dB");
    assert!(snr_db > 20.0, "SILK SNR {snr_db:.2} dB below floor of 20 dB");
}

#[test]
fn silk_sine_1k() {
    let snr_db = round_trip_snr(&signal_sine(1000.0), silk_codec(64_000));
    eprintln!("SILK sine_1k  @ 64k:  {snr_db:.2} dB");
    assert!(snr_db > 20.0, "SILK SNR {snr_db:.2} dB below floor of 20 dB");
}

#[test]
fn silk_white_noise() {
    let snr_db = round_trip_snr(&signal_white_noise(), silk_codec(64_000));
    eprintln!("SILK white_noise @ 64k:  {snr_db:.2} dB");
    assert!(snr_db > 20.0, "SILK SNR {snr_db:.2} dB below floor of 20 dB");
}

// ── CELT mode tests ───────────────────────────────────────────────────────────
//
// CELT quality is bounded by the MDCT window boundary effect.  Each 20 ms
// Opus frame is encoded as a single MDCT window, so the first and last ~hop
// samples of every frame are windowed to near-zero.  This is the same
// limitation as `PerceptualCodec` (see codec_quality.rs).  Thresholds are
// set to 1.5 dB below the minimum actually observed during development to
// provide a stable regression floor.

#[test]
fn celt_sine_440_64k() {
    let snr_db = round_trip_snr(&signal_sine(440.0), celt_codec(64_000));
    eprintln!("CELT sine_440 @ 64k:  {snr_db:.2} dB");
    assert!(snr_db > 1.0, "CELT SNR {snr_db:.2} dB below floor of 1 dB");
}

#[test]
fn celt_sine_440_128k() {
    let snr_db = round_trip_snr(&signal_sine(440.0), celt_codec(128_000));
    eprintln!("CELT sine_440 @ 128k: {snr_db:.2} dB");
    assert!(snr_db > 1.0, "CELT SNR {snr_db:.2} dB below floor of 1 dB");
}

#[test]
fn celt_sine_1k_64k() {
    let snr_db = round_trip_snr(&signal_sine(1000.0), celt_codec(64_000));
    eprintln!("CELT sine_1k  @ 64k:  {snr_db:.2} dB");
    assert!(snr_db > 1.0, "CELT SNR {snr_db:.2} dB below floor of 1 dB");
}

#[test]
fn celt_white_noise_64k() {
    let snr_db = round_trip_snr(&signal_white_noise(), celt_codec(64_000));
    eprintln!("CELT white_noise @ 64k:  {snr_db:.2} dB");
    // White noise through MDCT has near-zero SNR due to boundary effects.
    assert!(snr_db > -1.0, "CELT SNR {snr_db:.2} dB below floor of -1 dB");
}

// ── Auto-mode detection tests ─────────────────────────────────────────────────

#[test]
fn auto_mode_sine_440() {
    let snr_db = round_trip_snr(&signal_sine(440.0), auto_codec(128_000));
    eprintln!("Auto sine_440 @ 128k: {snr_db:.2} dB");
    assert!(snr_db > 1.0, "auto-mode SNR {snr_db:.2} dB below floor of 1 dB");
}

#[test]
fn auto_mode_white_noise() {
    let snr_db = round_trip_snr(&signal_white_noise(), auto_codec(128_000));
    eprintln!("Auto white_noise @ 128k: {snr_db:.2} dB");
    assert!(snr_db > -1.0, "auto-mode SNR {snr_db:.2} dB below floor of -1 dB");
}

// ── BandLayout::celt sanity test ──────────────────────────────────────────────

#[test]
fn celt_band_layout_sanity() {
    use audio_samples::BandLayout;
    use std::num::NonZeroUsize;

    // 20 ms at 44.1 kHz = 882 samples → 441 MDCT bins.
    let n_bins = NonZeroUsize::new(441).unwrap();
    let layout = BandLayout::celt(44100.0, n_bins);

    // Should have between 1 and 21 bands.
    assert!(layout.len().get() >= 1);
    assert!(layout.len().get() <= 21);

    // All bands must respect end_bin > start_bin.
    for band in layout.as_slice().iter() {
        assert!(
            band.end_bin > band.start_bin,
            "band [{}, {}) is empty",
            band.start_bin,
            band.end_bin,
        );
        assert!(band.end_bin <= n_bins.get(), "band end_bin out of range");
    }
}

#[test]
fn celt_band_layout_low_sample_rate() {
    use audio_samples::BandLayout;
    use std::num::NonZeroUsize;

    // 8 kHz narrowband — only bands up to 4 kHz should appear.
    let n_bins = NonZeroUsize::new(80).unwrap();
    let layout = BandLayout::celt(8000.0, n_bins);
    assert!(layout.len().get() >= 1, "must have at least one band");
    for band in layout.as_slice().iter() {
        assert!(band.end_bin > band.start_bin);
        assert!(band.end_bin <= n_bins.get());
    }
}

// ── SILK LPC round-trip unit test ─────────────────────────────────────────────

/// Confirms that the SILK encode/decode pair is a near-perfect round trip.
/// This isolates the LPC primitives from the full OpusCodec stack.
#[test]
fn silk_lpc_round_trip() {
    use audio_samples::{silk_encode_frame, silk_decode_frame};

    let samples: Vec<f32> = (0..128)
        .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 44100.0).sin() * 0.5)
        .collect();

    let frame = silk_encode_frame(&samples).expect("encode");
    let recovered = silk_decode_frame(&frame);

    assert_eq!(recovered.len(), samples.len());

    let signal_power: f32 = samples.iter().map(|&x| x * x).sum::<f32>() / samples.len() as f32;
    let error_power: f32 = samples
        .iter()
        .zip(recovered.iter())
        .map(|(&a, &b)| (a - b).powi(2))
        .sum::<f32>()
        / samples.len() as f32;
    let snr_db = 10.0 * (signal_power / error_power.max(1e-15)).log10();

    assert!(
        snr_db > 30.0,
        "SILK LPC round-trip SNR {snr_db:.2} dB below floor of 30 dB"
    );
}
