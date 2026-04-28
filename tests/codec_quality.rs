/// Round-trip quality tests for the perceptual codec.
///
/// Each test encodes a signal, decodes it, and measures SNR between the
/// original and reconstructed audio. Tests serve as regression guards:
/// any codec change that degrades quality below the established floor fails.
///
/// Run with:
/// ```bash
/// cargo test --test codec_quality \
///   --no-default-features --features "psychoacoustic,random-generation"
/// ```
use std::num::NonZeroUsize;
use std::time::Duration;

use audio_samples::{
    BandLayout, PerceptualCodec, PsychoacousticConfig,
    codecs::{decode, encode},
    sample_rate,
    utils::generation::{sine_wave, white_noise},
    utils::comparison::snr,
};
use non_empty_slice::NonEmptyVec;
use spectrograms::WindowType;

const SR: u32 = 44100;
const N_BANDS: usize = 24;
const N_BINS: usize = 1024;
const WINDOW_SIZE: usize = 2048;
const SIGNAL_DURATION_MS: u64 = 500;

// ── Codec constructors ────────────────────────────────────────────────────────

fn make_codec(bit_budget: u32) -> PerceptualCodec {
    let n_bands = NonZeroUsize::new(N_BANDS).unwrap();
    let n_bins  = NonZeroUsize::new(N_BINS).unwrap();
    let layout  = BandLayout::bark(n_bands, SR as f32, n_bins);
    let weights = PsychoacousticConfig::uniform_weights(n_bands);
    let config  = PsychoacousticConfig::mpeg1(weights.as_non_empty_slice());
    PerceptualCodec::with_window_size(
        layout, config, WindowType::Hanning, bit_budget, 1,
        NonZeroUsize::new(WINDOW_SIZE).unwrap(),
    )
}

// ── Test signals ──────────────────────────────────────────────────────────────

fn signal_sine(freq_hz: f64) -> audio_samples::AudioSamples<'static, f32> {
    sine_wave::<f32>(freq_hz, Duration::from_millis(SIGNAL_DURATION_MS), sample_rate!(44100), 0.5)
}

fn signal_white_noise() -> audio_samples::AudioSamples<'static, f32> {
    white_noise::<f32>(Duration::from_millis(SIGNAL_DURATION_MS), sample_rate!(44100), 0.5, Some(42))
}

fn signal_transient(n_bursts: usize) -> audio_samples::AudioSamples<'static, f32> {
    let burst = sine_wave::<f32>(880.0, Duration::from_millis(50), sample_rate!(44100), 1.0);
    let gap   = audio_samples::silence::<f32>(Duration::from_millis(50), sample_rate!(44100));

    let burst_s = burst.as_slice().expect("contiguous").to_vec();
    let gap_s   = gap.as_slice().expect("contiguous").to_vec();

    let mut flat = Vec::with_capacity((burst_s.len() + gap_s.len()) * n_bursts);
    for _ in 0..n_bursts {
        flat.extend_from_slice(&burst_s);
        flat.extend_from_slice(&gap_s);
    }
    let nev = unsafe { NonEmptyVec::new_unchecked(flat) };
    audio_samples::AudioSamples::<'static, f32>::from_mono_vec(nev, sample_rate!(44100))
}

// ── Core helper ───────────────────────────────────────────────────────────────

/// Encode → decode and return SNR in dB.
///
/// The error signal (original − reconstructed) is computed sample-by-sample;
/// the reconstructed length matches the original because `original_length` is
/// stored in the encoded representation.
fn round_trip_snr(
    signal: &audio_samples::AudioSamples<'static, f32>,
    codec: PerceptualCodec,
) -> f64 {
    let encoded       = encode(signal, codec).expect("encode failed");
    let reconstructed = decode::<PerceptualCodec, f32>(encoded).expect("decode failed");

    let orig  = signal.as_slice().expect("original contiguous");
    let recon = reconstructed.as_slice().expect("reconstructed contiguous");
    let len   = orig.len().min(recon.len());

    let error: Vec<f32> = orig[..len].iter()
        .zip(recon[..len].iter())
        .map(|(a, b)| a - b)
        .collect();

    let nev        = unsafe { NonEmptyVec::new_unchecked(error) };
    let error_sig  = audio_samples::AudioSamples::<'static, f32>::from_mono_vec(nev, signal.sample_rate());

    // Trim original to same length as error signal for SNR computation.
    let orig_trimmed = if len == orig.len() {
        signal.clone()
    } else {
        let trimmed: Vec<f32> = orig[..len].to_vec();
        let nev2 = unsafe { NonEmptyVec::new_unchecked(trimmed) };
        audio_samples::AudioSamples::<'static, f32>::from_mono_vec(nev2, signal.sample_rate())
    };

    snr(&orig_trimmed, &error_sig).expect("snr failed")
}

// ── 440 Hz sine ───────────────────────────────────────────────────────────────

#[test]
fn sine_440_64k() {
    let snr_db = round_trip_snr(&signal_sine(440.0_f64), make_codec(64_000));
    eprintln!("sine_440 @ 64k:  {snr_db:.2} dB");
    assert!(snr_db > 15.0, "SNR {snr_db:.2} dB below floor of 15 dB");
}

#[test]
fn sine_440_128k() {
    let snr_db = round_trip_snr(&signal_sine(440.0_f64), make_codec(128_000));
    eprintln!("sine_440 @ 128k: {snr_db:.2} dB");
    assert!(snr_db > 20.0, "SNR {snr_db:.2} dB below floor of 20 dB");
}

#[test]
fn sine_440_256k() {
    let snr_db = round_trip_snr(&signal_sine(440.0_f64), make_codec(256_000));
    eprintln!("sine_440 @ 256k: {snr_db:.2} dB");
    assert!(snr_db > 25.0, "SNR {snr_db:.2} dB below floor of 25 dB");
}

// ── 1 kHz sine ────────────────────────────────────────────────────────────────

#[test]
fn sine_1k_64k() {
    let snr_db = round_trip_snr(&signal_sine(1000.0_f64), make_codec(64_000));
    eprintln!("sine_1k @ 64k:  {snr_db:.2} dB");
    assert!(snr_db > 15.0, "SNR {snr_db:.2} dB below floor of 15 dB");
}

#[test]
fn sine_1k_128k() {
    let snr_db = round_trip_snr(&signal_sine(1000.0_f64), make_codec(128_000));
    eprintln!("sine_1k @ 128k: {snr_db:.2} dB");
    assert!(snr_db > 20.0, "SNR {snr_db:.2} dB below floor of 20 dB");
}

#[test]
fn sine_1k_256k() {
    let snr_db = round_trip_snr(&signal_sine(1000.0_f64), make_codec(256_000));
    eprintln!("sine_1k @ 256k: {snr_db:.2} dB");
    assert!(snr_db > 25.0, "SNR {snr_db:.2} dB below floor of 25 dB");
}

// ── White noise ───────────────────────────────────────────────────────────────

#[test]
fn white_noise_64k() {
    let snr_db = round_trip_snr(&signal_white_noise(), make_codec(64_000));
    eprintln!("white_noise @ 64k:  {snr_db:.2} dB");
    assert!(snr_db > 3.0, "SNR {snr_db:.2} dB below floor of 3 dB");
}

#[test]
fn white_noise_128k() {
    let snr_db = round_trip_snr(&signal_white_noise(), make_codec(128_000));
    eprintln!("white_noise @ 128k: {snr_db:.2} dB");
    assert!(snr_db > 5.0, "SNR {snr_db:.2} dB below floor of 5 dB");
}

#[test]
fn white_noise_256k() {
    let snr_db = round_trip_snr(&signal_white_noise(), make_codec(256_000));
    eprintln!("white_noise @ 256k: {snr_db:.2} dB");
    assert!(snr_db > 8.0, "SNR {snr_db:.2} dB below floor of 8 dB");
}

// ── Transient signal ──────────────────────────────────────────────────────────

#[test]
fn transient_64k() {
    let snr_db = round_trip_snr(&signal_transient(8), make_codec(64_000));
    eprintln!("transient_8 @ 64k:  {snr_db:.2} dB");
    assert!(snr_db > 10.0, "SNR {snr_db:.2} dB below floor of 10 dB");
}

#[test]
fn transient_128k() {
    let snr_db = round_trip_snr(&signal_transient(8), make_codec(128_000));
    eprintln!("transient_8 @ 128k: {snr_db:.2} dB");
    assert!(snr_db > 15.0, "SNR {snr_db:.2} dB below floor of 15 dB");
}

#[test]
fn transient_256k() {
    let snr_db = round_trip_snr(&signal_transient(8), make_codec(256_000));
    eprintln!("transient_8 @ 256k: {snr_db:.2} dB");
    assert!(snr_db > 20.0, "SNR {snr_db:.2} dB below floor of 20 dB");
}
