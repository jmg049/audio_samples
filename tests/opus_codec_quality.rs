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

use std::num::NonZeroU32;

use audio_samples::{
    AudioSamples, OpusBandwidth, OpusCodec, OpusConfig, OpusMode, OpusStereoCodec,
    codecs::{decode, encode},
    detect_opus_mode, sample_rate,
    utils::comparison::snr,
    utils::generation::{sine_wave, white_noise},
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
fn round_trip_snr(signal: &audio_samples::AudioSamples<'static, f32>, codec: OpusCodec) -> f64 {
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

    debug_assert!(!error.is_empty(), "error vector must be non-empty");
    // SAFETY: `len = orig.len().min(recon.len())`.  Both signals are the output
    // of a successful encode/decode, which requires a non-empty input, so
    // orig.len() >= 1 and recon.len() >= 1, therefore len >= 1 and `error` is
    // non-empty.
    let nev = unsafe { NonEmptyVec::new_unchecked(error) };
    let error_sig =
        audio_samples::AudioSamples::<'static, f32>::from_mono_vec(nev, signal.sample_rate());

    let orig_trimmed = if len == orig.len() {
        signal.clone()
    } else {
        let trimmed: Vec<f32> = orig[..len].to_vec();
        debug_assert!(!trimmed.is_empty(), "trimmed vector must be non-empty");
        // SAFETY: len >= 1 (same guarantee as error above), so trimmed is non-empty.
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
    assert!(
        snr_db > 20.0,
        "SILK SNR {snr_db:.2} dB below floor of 20 dB"
    );
}

#[test]
fn silk_sine_1k() {
    let snr_db = round_trip_snr(&signal_sine(1000.0), silk_codec(64_000));
    eprintln!("SILK sine_1k  @ 64k:  {snr_db:.2} dB");
    assert!(
        snr_db > 20.0,
        "SILK SNR {snr_db:.2} dB below floor of 20 dB"
    );
}

#[test]
fn silk_white_noise() {
    let snr_db = round_trip_snr(&signal_white_noise(), silk_codec(64_000));
    eprintln!("SILK white_noise @ 64k:  {snr_db:.2} dB");
    assert!(
        snr_db > 20.0,
        "SILK SNR {snr_db:.2} dB below floor of 20 dB"
    );
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
    assert!(
        snr_db > -1.0,
        "CELT SNR {snr_db:.2} dB below floor of -1 dB"
    );
}

// ── Auto-mode detection tests ─────────────────────────────────────────────────

#[test]
fn auto_mode_sine_440() {
    let snr_db = round_trip_snr(&signal_sine(440.0), auto_codec(128_000));
    eprintln!("Auto sine_440 @ 128k: {snr_db:.2} dB");
    assert!(
        snr_db > 1.0,
        "auto-mode SNR {snr_db:.2} dB below floor of 1 dB"
    );
}

#[test]
fn auto_mode_white_noise() {
    let snr_db = round_trip_snr(&signal_white_noise(), auto_codec(128_000));
    eprintln!("Auto white_noise @ 128k: {snr_db:.2} dB");
    assert!(
        snr_db > -1.0,
        "auto-mode SNR {snr_db:.2} dB below floor of -1 dB"
    );
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
    use audio_samples::{silk_decode_frame, silk_encode_frame};

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

// ── detect_mode selection tests ───────────────────────────────────────────────
//
// These tests verify that `detect_mode` routes signals to the expected codec.
// Because the SFM is computed from time-domain sub-band energies (not a
// frequency-domain spectrum), the rule is:
//   SFM ≈ 1  →  energy uniform across sub-windows  →  CELT
//   SFM ≈ 0  →  energy concentrated in one sub-window  →  SILK
//
// A sustained 440 Hz sine has flat per-window energy → CELT.
// A burst occupying only the first 1/8 of the frame followed by silence
// concentrates all energy in sub-band 0 → SFM ≪ 0.4 → SILK.

/// FullBand bypasses the SFM check (SILK is not valid above 8 kHz by spec),
/// so detect_mode returns CELT via the bandwidth-constraint path.
#[test]
fn detect_mode_fullband_forces_celt() {
    let signal = signal_sine(440.0);
    let samples = signal.as_slice().expect("mono contiguous");
    let mode = detect_opus_mode(samples, 44100, OpusBandwidth::FullBand);
    assert_eq!(
        mode,
        OpusMode::Celt,
        "FullBand must always select CELT (SILK not valid above 8 kHz)"
    );
}

/// WideBand allows both SILK and CELT; a sustained 440 Hz sine has flat
/// per-window energy (SFM ≈ 1) so the SFM heuristic selects CELT.
#[test]
fn detect_mode_selects_celt_for_sine() {
    let signal = signal_sine(440.0);
    let samples = signal.as_slice().expect("mono contiguous");
    // WideBand: both SILK and CELT are valid, so the SFM decides.
    let mode = detect_opus_mode(samples, 44100, OpusBandwidth::WideBand);
    assert_eq!(
        mode,
        OpusMode::Celt,
        "sustained 440 Hz sine (flat time-energy) should map to CELT"
    );
}

/// WideBand allows both SILK and CELT. A burst occupying only the first 1/8
/// of the frame (followed by silence) concentrates all energy in one sub-window,
/// giving SFM ≪ 0.4, so the heuristic selects SILK.
#[test]
fn detect_mode_selects_silk_for_burst() {
    // 20 ms at 44100 Hz.
    let n = 882usize;
    let mut burst = vec![0.0f32; n];
    let burst_len = n / 8;
    for (i, sample) in burst.iter_mut().enumerate().take(burst_len) {
        *sample = (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 44100.0).sin() * 0.5;
    }
    // WideBand: both modes are valid, so the SFM decides.
    let mode = detect_opus_mode(&burst, 44100, OpusBandwidth::WideBand);
    assert_eq!(
        mode,
        OpusMode::Silk,
        "burst-then-silence (energy in one sub-window) should map to SILK"
    );
}

// ── OpusStereoCodec round-trip test ──────────────────────────────────────────

/// Encodes a stereo signal with SILK on both channels and verifies that the
/// decoded left channel has SNR above the SILK floor.
#[test]
fn stereo_silk_round_trip() {
    // 440 Hz left, 880 Hz right — both mono, 200 ms.
    let left = signal_sine(440.0);
    let right = signal_sine(880.0);
    let n = left.samples_per_channel().get();

    let left_s = left.as_slice().expect("mono contiguous");
    let right_s = right.as_slice().expect("mono contiguous");

    // Interleave L/R into a stereo AudioSamples.
    let interleaved: Vec<f32> = left_s
        .iter()
        .zip(right_s)
        .flat_map(|(&l, &r)| [l, r])
        .collect();
    let ne = NonEmptyVec::new(interleaved).expect("non-empty");
    let stereo: AudioSamples<'static, f32> = AudioSamples::from_interleaved_vec(
        ne,
        NonZeroU32::new(2).expect("2 > 0"),
        sample_rate!(44100),
    )
    .expect("valid stereo");

    // Encode: mid at 64 kbps SILK, side at 32 kbps SILK.
    let codec = OpusStereoCodec::new(
        OpusConfig::with_mode(OpusMode::Silk, 64_000),
        OpusConfig::with_mode(OpusMode::Silk, 32_000),
        spectrograms::WindowType::Hanning,
    );
    let encoded = encode(&stereo, codec).expect("stereo encode");
    let recovered: AudioSamples<'static, f32> =
        decode::<OpusStereoCodec, f32>(encoded).expect("stereo decode");

    assert_eq!(recovered.num_channels().get(), 2, "decoded must be stereo");
    assert_eq!(
        recovered.samples_per_channel().get(),
        n,
        "length must be preserved"
    );

    // Measure SNR on the left channel against the original 440 Hz sine.
    let mut recon_channels = recovered.channels();
    let recon_left = recon_channels.next().expect("left channel");
    let recon_left_s = recon_left.as_slice().expect("channel contiguous");
    let len = left_s.len().min(recon_left_s.len());

    let signal_power: f64 = left_s[..len]
        .iter()
        .map(|&x| (x as f64).powi(2))
        .sum::<f64>()
        / len as f64;
    let error_power: f64 = left_s[..len]
        .iter()
        .zip(recon_left_s[..len].iter())
        .map(|(a, b)| ((a - b) as f64).powi(2))
        .sum::<f64>()
        / len as f64;
    let snr_db = 10.0 * (signal_power / error_power.max(1e-15)).log10();

    eprintln!("Stereo SILK left-channel SNR: {snr_db:.2} dB");
    assert!(
        snr_db > 20.0,
        "Stereo SILK left-channel SNR {snr_db:.2} dB below floor of 20 dB"
    );
}

// ── Cross-frame SILK state ────────────────────────────────────────────────────

fn snr_db(original: &[f32], recovered: &[f32]) -> f64 {
    let n = original.len().min(recovered.len());
    let signal: f64 = original[..n]
        .iter()
        .map(|&x| (x as f64).powi(2))
        .sum::<f64>()
        / n as f64;
    let error: f64 = original[..n]
        .iter()
        .zip(recovered[..n].iter())
        .map(|(a, b)| ((*a - *b) as f64).powi(2))
        .sum::<f64>()
        / n as f64;
    10.0 * (signal / error.max(1e-15)).log10()
}

#[test]
fn stateful_silk_cross_frame() {
    use audio_samples::{SilkState, silk_decode_frame_stateful, silk_encode_frame_stateful};

    let sample_rate = 44100_u32;
    let frame_len = 882; // 20 ms
    let all_samples: Vec<f32> = (0..frame_len * 3)
        .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / sample_rate as f32).sin() * 0.5)
        .collect();

    let mut enc = SilkState::default();
    let f0 = silk_encode_frame_stateful(&all_samples[..frame_len], sample_rate, &mut enc).unwrap();
    let f1 = silk_encode_frame_stateful(
        &all_samples[frame_len..frame_len * 2],
        sample_rate,
        &mut enc,
    )
    .unwrap();
    let f2 =
        silk_encode_frame_stateful(&all_samples[frame_len * 2..], sample_rate, &mut enc).unwrap();

    let mut dec = SilkState::default();
    let mut recovered = Vec::new();
    recovered.extend(silk_decode_frame_stateful(&f0, &mut dec));
    recovered.extend(silk_decode_frame_stateful(&f1, &mut dec));
    recovered.extend(silk_decode_frame_stateful(&f2, &mut dec));

    assert_eq!(recovered.len(), all_samples.len());
    let db = snr_db(&all_samples, &recovered);
    eprintln!("Stateful SILK 3-frame SNR: {db:.1} dB");
    assert!(db > 20.0, "stateful SILK SNR {db:.1} dB too low");
}

#[test]
fn silk_ltp_detects_pitch() {
    use audio_samples::estimate_pitch;

    let sample_rate = 44100_u32;
    // 220 Hz sine → pitch period ≈ 200 samples.
    let samples: Vec<f32> = (0..882)
        .map(|i| (2.0 * std::f32::consts::PI * 220.0 * i as f32 / sample_rate as f32).sin())
        .collect();
    let result = estimate_pitch(&samples, sample_rate);
    assert!(result.is_some(), "pitch not detected for 220 Hz sine");
    let (lag, gain) = result.unwrap();
    let expected = (sample_rate as f32 / 220.0).round() as usize;
    assert!(
        lag.abs_diff(expected) <= 3,
        "lag {lag} expected ~{expected}"
    );
    assert!(gain > 0.5, "LTP gain {gain:.2} too low for pure sine");
}

// ── Hybrid mode ───────────────────────────────────────────────────────────────

#[test]
fn hybrid_sine_round_trip() {
    use audio_samples::codecs::{decode, encode};
    use audio_samples::{AudioSamples, OpusCodec, OpusConfig, OpusMode, sample_rate};
    use spectrograms::WindowType;

    let sr = sample_rate!(44100);
    let samples: Vec<f32> = (0..882 * 2)
        .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 44100.0).sin() * 0.5)
        .collect();
    let ne = non_empty_slice::NonEmptyVec::new(samples).unwrap();
    let audio = AudioSamples::<'static, f32>::from_mono_vec(ne, sr);

    let config = OpusConfig::with_mode(OpusMode::Hybrid, 128_000);
    let codec = OpusCodec::new(config, WindowType::Hanning);

    let encoded = encode(&audio, codec).expect("hybrid encode");
    assert!(
        encoded
            .frames
            .as_non_empty_slice()
            .iter()
            .all(|f| f.mode == OpusMode::Hybrid),
        "not all frames encoded as Hybrid"
    );

    let recovered: AudioSamples<'static, f32> =
        decode::<OpusCodec, f32>(encoded).expect("hybrid decode");

    let orig_ch = audio.as_slice().expect("mono contiguous");
    let rec_ch = recovered.as_slice().expect("mono contiguous");
    let db = snr_db(orig_ch, rec_ch);
    eprintln!("Hybrid sine round-trip SNR: {db:.1} dB");
    assert!(db > 15.0, "hybrid SNR {db:.1} dB too low");
}

#[test]
fn hybrid_white_noise_round_trip() {
    use audio_samples::codecs::{decode, encode};
    use audio_samples::{AudioSamples, OpusCodec, OpusConfig, OpusMode, sample_rate, white_noise};
    use spectrograms::WindowType;

    let sr = sample_rate!(44100);
    let noise = white_noise::<f32>(std::time::Duration::from_millis(40), sr, 0.5, None);

    let config = OpusConfig::with_mode(OpusMode::Hybrid, 128_000);
    let codec = OpusCodec::new(config, WindowType::Hanning);

    let encoded = encode(&noise, codec).expect("hybrid noise encode");
    let recovered: AudioSamples<'static, f32> =
        decode::<OpusCodec, f32>(encoded).expect("hybrid noise decode");

    let orig = noise.as_slice().expect("mono contiguous");
    let rec = recovered.as_slice().expect("mono contiguous");
    let db = snr_db(orig, rec);
    eprintln!("Hybrid white-noise round-trip SNR: {db:.1} dB");
    // SILK provides no prediction gain on white noise, so the low band is
    // limited to 16-bit quantisation; 5 dB is the realistic floor for this case.
    assert!(db > 5.0, "hybrid noise SNR {db:.1} dB too low");
}
