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
    BandLayout, PerceptualCodec, PsychoacousticConfig, analyse_signal_with_window_size,
    codecs::{
        decode, encode,
        perceptual::quantization::{
            allocate_bits, max_index_for_word_length, quantize, refine_step_sizes,
        },
    },
    sample_rate,
    utils::comparison::snr,
    utils::generation::{sine_wave, white_noise},
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
    let n_bins = NonZeroUsize::new(N_BINS).unwrap();
    let layout = BandLayout::bark(n_bands, SR as f32, n_bins);
    let weights = PsychoacousticConfig::uniform_weights(n_bands);
    let config = PsychoacousticConfig::mpeg1(weights.as_non_empty_slice());
    PerceptualCodec::with_window_size(
        layout,
        config,
        WindowType::Hanning,
        bit_budget,
        1,
        NonZeroUsize::new(WINDOW_SIZE).unwrap(),
    )
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

fn signal_transient(n_bursts: usize) -> audio_samples::AudioSamples<'static, f32> {
    let burst = sine_wave::<f32>(880.0, Duration::from_millis(50), sample_rate!(44100), 1.0);
    let gap = audio_samples::silence::<f32>(Duration::from_millis(50), sample_rate!(44100));

    let burst_s = burst.as_slice().expect("contiguous").to_vec();
    let gap_s = gap.as_slice().expect("contiguous").to_vec();

    let mut flat = Vec::with_capacity((burst_s.len() + gap_s.len()) * n_bursts);
    for _ in 0..n_bursts {
        flat.extend_from_slice(&burst_s);
        flat.extend_from_slice(&gap_s);
    }
    // SAFETY: callers pass n_bursts >= 1, and each iteration appends the 50 ms
    // burst and gap (2205 samples each at 44100 Hz), so `flat` is non-empty.
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
    let encoded = encode(signal, codec).expect("encode failed");
    let reconstructed = decode::<PerceptualCodec, f32>(encoded).expect("decode failed");

    let orig = signal.as_slice().expect("original contiguous");
    let recon = reconstructed.as_slice().expect("reconstructed contiguous");
    let len = orig.len().min(recon.len());

    let error: Vec<f32> = orig[..len]
        .iter()
        .zip(recon[..len].iter())
        .map(|(a, b)| a - b)
        .collect();

    // SAFETY: len = orig.len().min(recon.len()); both come from a successful
    // encode/decode round-trip of a non-empty signal, so len >= 1 and `error`
    // (built from orig[..len]) is non-empty.
    let nev = unsafe { NonEmptyVec::new_unchecked(error) };
    let error_sig =
        audio_samples::AudioSamples::<'static, f32>::from_mono_vec(nev, signal.sample_rate());

    // Trim original to same length as error signal for SNR computation.
    let orig_trimmed = if len == orig.len() {
        signal.clone()
    } else {
        let trimmed: Vec<f32> = orig[..len].to_vec();
        // SAFETY: len >= 1 (same guarantee as for `error` above), so `trimmed` is non-empty.
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

// ── Multi-tone round-trip ───────────────────────────────────────────────────────

/// Sum of three sinusoids (220 / 440 / 880 Hz), peak-normalised to 0.5.
fn signal_multi_tone() -> audio_samples::AudioSamples<'static, f32> {
    let a = sine_wave::<f32>(
        220.0,
        Duration::from_millis(SIGNAL_DURATION_MS),
        sample_rate!(44100),
        1.0,
    );
    let b = sine_wave::<f32>(
        440.0,
        Duration::from_millis(SIGNAL_DURATION_MS),
        sample_rate!(44100),
        1.0,
    );
    let c = sine_wave::<f32>(
        880.0,
        Duration::from_millis(SIGNAL_DURATION_MS),
        sample_rate!(44100),
        1.0,
    );

    let av = a.as_slice().expect("contiguous");
    let bv = b.as_slice().expect("contiguous");
    let cv = c.as_slice().expect("contiguous");
    let n = av.len().min(bv.len()).min(cv.len());

    let mut mixed: Vec<f32> = (0..n).map(|i| av[i] + bv[i] + cv[i]).collect();
    let peak = mixed
        .iter()
        .fold(0.0_f32, |m, &x| m.max(x.abs()))
        .max(1e-12);
    let scale = 0.5 / peak;
    for x in &mut mixed {
        *x *= scale;
    }

    // SAFETY: n = min of three non-empty signal lengths, so n >= 1 and `mixed`
    // (built from 0..n) is non-empty.
    let nev = unsafe { NonEmptyVec::new_unchecked(mixed) };
    audio_samples::AudioSamples::<'static, f32>::from_mono_vec(nev, sample_rate!(44100))
}

#[test]
fn multi_tone_128k() {
    let snr_db = round_trip_snr(&signal_multi_tone(), make_codec(128_000));
    eprintln!("multi_tone @ 128k: {snr_db:.2} dB");
    assert!(snr_db > 20.0, "SNR {snr_db:.2} dB below floor of 20 dB");
}

#[test]
fn multi_tone_256k() {
    let snr_db = round_trip_snr(&signal_multi_tone(), make_codec(256_000));
    eprintln!("multi_tone @ 256k: {snr_db:.2} dB");
    assert!(snr_db > 25.0, "SNR {snr_db:.2} dB below floor of 25 dB");
}

// ── Quantizer word-length bound (proves the index clamp) ────────────────────────

/// Encoding via the real allocate → refine → quantize path must keep every
/// quantised index inside the band's allocated signed word length. This is the
/// invariant the quantization index clamp enforces: no single coefficient can
/// overrun the bit budget its band was granted.
#[test]
fn quantized_indices_fit_allocated_word_length() {
    let n_bands = NonZeroUsize::new(N_BANDS).unwrap();
    let n_bins = NonZeroUsize::new(N_BINS).unwrap();
    let layout = BandLayout::bark(n_bands, SR as f32, n_bins);
    let weights = PsychoacousticConfig::uniform_weights(n_bands);
    let config = PsychoacousticConfig::mpeg1(weights.as_non_empty_slice());
    let window_size = NonZeroUsize::new(WINDOW_SIZE).unwrap();

    for signal in [
        signal_multi_tone(),
        signal_sine(440.0),
        signal_white_noise(),
    ] {
        let result = analyse_signal_with_window_size(
            &signal,
            WindowType::Hanning,
            Some(window_size),
            &layout,
            &config,
        )
        .expect("analysis failed");

        let mut allocation = allocate_bits(&result.band_metrics, 128_000, 1);
        refine_step_sizes(
            &mut allocation,
            result.coefficients.as_non_empty_slice(),
            result.n_coefficients,
            result.n_frames,
        );
        let quantized = quantize(
            result.coefficients.as_non_empty_slice(),
            result.n_coefficients,
            result.n_frames,
            &allocation,
        );

        let nf = result.n_frames.get();
        let nc = result.n_coefficients.get();
        for alloc in allocation.allocations.iter() {
            let max_index = max_index_for_word_length(alloc.word_length);
            for k in alloc.start_bin..alloc.end_bin.min(nc) {
                for f in 0..nf {
                    let idx = quantized[k * nf + f];
                    assert!(
                        idx.abs() <= max_index,
                        "index {idx} in band [{}, {}) exceeds ±{max_index} \
                         (word_length = {} bits)",
                        alloc.start_bin,
                        alloc.end_bin,
                        alloc.word_length,
                    );
                }
            }
        }
    }
}
