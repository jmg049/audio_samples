//! Tutorial 14: Psychoacoustics and codecs.
//!
//! Accompanies the guide chapter `audio_samples::guide::c14_psychoacoustics_codecs`.
//!
//! What this lab shows, with every number checked by an `assert!`:
//!
//! 1. dBFS is relative to full scale; dB SPL needs a calibration constant.
//! 2. Perceptual frequency scales as the library implements them: HTK mel
//!    (`utils::audio_math`), Traunmüller Bark and the Glasberg-Moore ERB
//!    *bandwidth* (`codecs::perceptual::bands`), plus the ERB-number scale.
//! 3. Terhardt's absolute threshold of hearing, minimum near 3.3 kHz.
//! 4. Simultaneous masking: the Schroeder et al. (1979) spreading function has
//!    a shallow upper slope (-10 dB/Bark) and a steep lower one (+25 dB/Bark);
//!    the library's preset slopes are the other way round.
//! 5. G.711 mu-law: a 14-bit linear sample becomes an 8-bit code; the SNR is
//!    nearly level-independent, unlike 8-bit linear PCM.
//! 6. The perceptual MDCT codec: SNR rises with the bit budget.
//!
//! Run with:
//! `cargo run --release --example tutorial_14_psychoacoustics_codecs --features educational,full_no_plotting`

use std::f64::consts::PI;
use std::num::NonZeroUsize;
use std::time::Duration;

use audio_samples::codecs::perceptual::bands::{bark_to_hz, hz_to_bark, hz_to_erb};
use audio_samples::codecs::{decode, encode};
use audio_samples::utils::audio_math::{hz_to_mel, mel_to_hz};
use audio_samples::{
    AudioProcessing, AudioSampleResult, AudioSamples, BandLayout, MaskerType, PerceptualCodec,
    PsychoacousticConfig, absolute_threshold_of_hearing, sample_rate, sine_wave,
    spreading_attenuation,
};
use non_empty_slice::NonEmptyVec;
use spectrograms::WindowType;

fn db(x: f64) -> f64 {
    20.0 * x.log10()
}

/// Terhardt (1979) threshold in quiet, dB SPL, f in Hz.
fn terhardt(f: f64) -> f64 {
    let k = f / 1000.0;
    3.64 * k.powf(-0.8) - 6.5 * (-0.6 * (k - 3.3).powi(2)).exp() + 1e-3 * k.powi(4)
}

/// Schroeder, Atal & Hall (1979) spreading function, dB, dz = maskee - masker in Bark.
fn schroeder(dz: f64) -> f64 {
    15.81 + 7.5 * (dz + 0.474) - 17.5 * (1.0 + (dz + 0.474).powi(2)).sqrt()
}

/// ITU-T G.711 mu-law encoder for a 14-bit two's-complement sample.
fn ulaw_encode(x: i32) -> u8 {
    let (mag, sign) = if x < 0 { (-x, 0x80u8) } else { (x, 0x00u8) };
    // Clip to the 14-bit range, add the bias 33 so every segment starts at a power
    // of two, and saturate at 8191 (the top of segment 7).
    let biased = (mag.min(8_159) + 33).min(8_191);
    let seg = (0..8).find(|&s| biased < (64 << s)).unwrap(); // 33..8191 -> segment 0..7
    let mant = ((biased >> (seg + 1)) & 0x0F) as u8;
    !(sign | ((seg as u8) << 4) | mant) // codes are transmitted inverted
}

/// ITU-T G.711 mu-law decoder to a 14-bit sample.
fn ulaw_decode(code: u8) -> i32 {
    let c = !code;
    let seg = i32::from((c >> 4) & 0x07);
    let mant = i32::from(c & 0x0F);
    let mag = (((mant << 1) + 33) << seg) - 33;
    if c & 0x80 != 0 { -mag } else { mag }
}

/// Continuous mu-law characteristic F(x) = sgn(x) ln(1 + mu|x|) / ln(1 + mu).
fn mu_law(x: f64, mu: f64) -> f64 {
    x.signum() * (1.0 + mu * x.abs()).ln() / (1.0 + mu).ln()
}

/// SNR (dB) of a quantiser applied to a sine of peak `level_dbfs`.
fn sine_snr(level_dbfs: f64, q: impl Fn(f64) -> f64) -> f64 {
    let a = 10f64.powf(level_dbfs / 20.0);
    let (mut s, mut e) = (0.0, 0.0);
    for n in 0..8_000 {
        // 8000 samples of a 1 kHz-ish tone at 8 kHz, offset so samples are not periodic.
        let x = a * (2.0 * PI * 0.1234567 * n as f64).sin();
        let y = q(x);
        s += x * x;
        e += (x - y) * (x - y);
    }
    10.0 * (s / e).log10()
}

fn main() -> AudioSampleResult<()> {
    println!("== Tutorial 14: psychoacoustics and codecs ==\n");

    // ---------------------------------------------------------------- 1
    println!("1. dBFS versus dB SPL");
    // A sine of peak 0.5 is -6.02 dBFS (peak) and -9.03 dBFS (RMS).
    let tone = sine_wave::<f64>(1_000.0, Duration::from_secs(1), sample_rate!(16000), 0.5);
    let rms = (tone.as_slice().unwrap().iter().map(|v| v * v).sum::<f64>() / 16_000.0).sqrt();
    println!(
        "   peak 0.5 sine: {:.2} dBFS peak, {:.2} dBFS RMS",
        db(0.5),
        db(rms)
    );
    assert!((db(0.5) + 6.02).abs() < 0.005 && (db(rms) + 9.03).abs() < 0.005);
    // SPL = 20 log10(p / 20 uPa). With a calibration "full-scale sine RMS = 94 dB SPL"
    // (1 Pa), the same tone plays at 94 - 6.02 = 87.98 dB SPL; change the volume knob
    // and the dBFS value stays the same while the SPL changes.
    let spl_of_1pa = db(1.0 / 20e-6);
    println!(
        "   1 Pa = {spl_of_1pa:.2} dB SPL; with a full-scale sine calibrated to 94 dB SPL the tone is {:.2} dB SPL",
        94.0 + db(0.5)
    );
    assert!((spl_of_1pa - 93.98).abs() < 0.005);
    println!();

    // ---------------------------------------------------------------- 2
    println!("2. Perceptual frequency scales at 100 Hz, 1 kHz, 4 kHz");
    for f in [100.0f64, 1_000.0, 4_000.0] {
        let mel = hz_to_mel(f);
        let bark = f64::from(hz_to_bark(f as f32));
        let erb_bw = f64::from(hz_to_erb(f as f32));
        let erb_number = 21.4 * (1.0 + 0.00437 * f).log10();
        println!(
            "   {f:>6} Hz: {mel:>7.1} mel (HTK), {bark:>5.2} Bark, ERB bandwidth {erb_bw:>6.1} Hz, ERB-number {erb_number:>5.2} Cam"
        );
        assert!((mel - 2595.0 * (1.0 + f / 700.0).log10()).abs() < 1e-9);
        assert!((bark - (26.81 * f / (1960.0 + f) - 0.53)).abs() < 1e-4); // f32
        assert!((erb_bw - 24.7 * (4.37 * f / 1000.0 + 1.0)).abs() < 1e-3); // f32
        // Round trips.
        assert!((mel_to_hz(mel) - f).abs() < 1e-9);
        assert!((f64::from(bark_to_hz(bark as f32)) - f).abs() < 0.05);
    }
    assert!((hz_to_mel(1_000.0) - 1_000.0).abs() < 0.02); // HTK: 1000 Hz ~ 1000 mel
    assert!((f64::from(hz_to_bark(1_000.0)) - 8.527).abs() < 1e-3);
    assert!((f64::from(hz_to_erb(1_000.0)) - 132.639).abs() < 1e-3);
    println!();

    // ---------------------------------------------------------------- 3
    println!("3. Absolute threshold of hearing (Terhardt 1979), dB SPL");
    let mut min = (0.0, f64::MAX);
    for f in (20..=16_000).step_by(10) {
        let t = terhardt(f as f64);
        let lib = f64::from(absolute_threshold_of_hearing(f as f32));
        // The library evaluates the same formula in f32 and clamps to [-20, 60].
        assert!(
            (lib - t.clamp(-20.0, 60.0)).abs() < 1e-3,
            "{f} Hz: {lib} vs {t}"
        );
        if t < min.1 {
            min = (f as f64, t);
        }
    }
    for f in [100.0, 1_000.0, 3_300.0, 10_000.0] {
        println!("   {f:>6} Hz: {:>6.2} dB SPL", terhardt(f));
    }
    println!("   minimum {:.2} dB SPL at {} Hz", min.1, min.0);
    assert!((terhardt(100.0) - 22.96).abs() < 0.01);
    assert!((terhardt(1_000.0) - 3.37).abs() < 0.01);
    assert!((min.0 - 3_340.0).abs() <= 30.0 && (min.1 + 5.0).abs() < 0.1);
    println!();

    // ---------------------------------------------------------------- 4
    println!("4. Spreading of simultaneous masking (threshold relative to its peak)");
    // Schroeder's function peaks near dz = 0; its slopes approach 7.5 - 17.5 = -10
    // and 7.5 + 17.5 = +25 dB/Bark far from the masker (secants over 8-10 Bark).
    let up = (schroeder(10.0) - schroeder(8.0)) / 2.0;
    let down = (schroeder(-8.0) - schroeder(-10.0)) / 2.0;
    println!(
        "   Schroeder et al. 1979: {up:+.1} dB/Bark above the masker, {down:+.1} dB/Bark below (rising towards it)"
    );
    assert!((up + 10.0).abs() < 0.3 && (down - 25.0).abs() < 0.3);
    let weights = PsychoacousticConfig::uniform_weights(NonZeroUsize::new(24).unwrap());
    let cfg = PsychoacousticConfig::mpeg1(weights.as_non_empty_slice());
    let att_up = spreading_attenuation(10.0, 12.0, MaskerType::Tonal, &cfg);
    let att_down = spreading_attenuation(10.0, 8.0, MaskerType::Tonal, &cfg);
    println!(
        "   library mpeg1 preset: attenuation 2 Bark above {att_up:.1} dB, 2 Bark below {att_down:.1} dB"
    );
    // 14.5 + 25*2 above, 14.5 + 6*2 below: the preset's upper slope is the steep one.
    assert!((att_up - 64.5).abs() < 1e-4 && (att_down - 26.5).abs() < 1e-4);
    assert!(att_up > att_down); // inverted relative to the psychoacoustic data
    println!();

    // ---------------------------------------------------------------- 5
    println!("5. G.711 mu-law: 14-bit linear -> 8-bit code");
    // Exhaustive check: decode(encode(x)) error never exceeds half a step of the segment.
    for x in -8_159..=8_159 {
        let y = ulaw_decode(ulaw_encode(x));
        let seg = (0..8)
            .find(|&s| (x.abs() + 33).min(8_191) < (64 << s))
            .unwrap();
        assert!((x - y).abs() <= 1 << seg, "x {x} -> {y}");
    }
    assert_eq!(ulaw_encode(0), 0xFF);
    assert_eq!(ulaw_decode(ulaw_encode(8_159)), 8_031);
    // The segmented code tracks the continuous mu = 255 curve: index c (0..127)
    // stays within 2 codes of 127 F(x).
    let mut worst: f64 = 0.0;
    for x in 1..=8_159 {
        let c = f64::from(!ulaw_encode(x) & 0x7F);
        worst = worst.max((c - 127.0 * mu_law(f64::from(x) / 8_159.0, 255.0)).abs());
    }
    println!("   segmented code versus 127 F(x), mu = 255: worst difference {worst:.2} codes");
    assert!(worst < 2.0);
    let ulaw = |x: f64| f64::from(ulaw_decode(ulaw_encode((x * 8_159.0).round() as i32))) / 8_159.0;
    let lin8 = |x: f64| (x * 127.0).round().clamp(-128.0, 127.0) / 127.0;
    println!("   level (dBFS peak)   SNR mu-law   SNR 8-bit linear");
    let mut rows = Vec::new();
    for level in [0.0, -10.0, -20.0, -30.0, -40.0] {
        let (su, sl) = (sine_snr(level, ulaw), sine_snr(level, lin8));
        println!("   {level:>13.0}       {su:>6.1} dB     {sl:>6.1} dB");
        rows.push((su, sl));
    }
    // mu-law: between 34 and 40 dB over the whole 40 dB range of levels ...
    assert!(rows.iter().all(|(su, _)| (34.0..40.5).contains(su)));
    // ... while 8-bit linear starts at 6.02 b + 1.76 = 49.9 dB and loses about
    // 1 dB of SNR per dB of level.
    assert!((rows[0].1 - 49.9).abs() < 0.5);
    assert!((rows[0].1 - rows[4].1 - 40.0).abs() < 1.5);
    assert!(rows[4].0 > rows[4].1 + 20.0);
    // The library's mu_compress is the continuous curve (F(0) = 0) and its
    // mu_expand inverts it exactly.
    let xs = [-0.9, -0.3, -0.01, 0.0, 0.02, 0.5, 1.0];
    let audio: AudioSamples<f64> =
        AudioSamples::from_mono_vec(NonEmptyVec::new(xs.to_vec()).unwrap(), sample_rate!(8000));
    let compressed = audio.mu_compress(255.0)?;
    for (x, y) in xs.iter().zip(compressed.as_slice().unwrap()) {
        assert!((mu_law(*x, 255.0) - y).abs() < 1e-12);
    }
    assert_eq!(compressed.as_slice().unwrap()[3], 0.0);
    let expanded = compressed.mu_expand(255.0)?;
    for (x, y) in xs.iter().zip(expanded.as_slice().unwrap()) {
        assert!((x - y).abs() < 1e-12);
    }
    println!("   library mu_compress equals F(x) and mu_expand inverts it, to 1e-12");
    println!();

    // ---------------------------------------------------------------- 6
    println!(
        "6. Perceptual MDCT codec (24 Bark bands, 2048-sample MDCT), 1 kHz tone, 0.5 s at 44.1 kHz"
    );
    let x = sine_wave::<f32>(
        1_000.0,
        Duration::from_millis(500),
        sample_rate!(44100),
        0.5,
    );
    let mut last = f64::MIN;
    let mut snrs = Vec::new();
    for budget in [32_000u32, 64_000, 128_000, 256_000] {
        let n_bands = NonZeroUsize::new(24).unwrap();
        let layout = BandLayout::bark(n_bands, 44_100.0, NonZeroUsize::new(1024).unwrap());
        let w = PsychoacousticConfig::uniform_weights(n_bands);
        let config = PsychoacousticConfig::mpeg1(w.as_non_empty_slice());
        let codec = PerceptualCodec::with_window_size(
            layout,
            config,
            WindowType::Hanning,
            budget,
            1,
            NonZeroUsize::new(2048).unwrap(),
        );
        let y = decode::<PerceptualCodec, f32>(encode(&x, codec)?)?;
        let (a, b) = (x.as_slice().unwrap(), y.as_slice().unwrap());
        let n = a.len().min(b.len());
        let s: f64 = a[..n].iter().map(|&v| f64::from(v).powi(2)).sum();
        let e: f64 = a[..n]
            .iter()
            .zip(&b[..n])
            .map(|(&p, &q)| f64::from(p - q).powi(2))
            .sum();
        let snr = 10.0 * (s / e).log10();
        println!("   bit budget {budget:>7}: SNR {snr:>6.2} dB");
        assert!(snr > last); // more bits, less quantisation noise
        last = snr;
        snrs.push(snr);
    }
    // About 42 dB at 32 000 bits rising to about 51 dB at 256 000 bits, where it
    // saturates (1 dB tolerance: the figures depend on the bit allocator).
    assert!((snrs[0] - 42.1).abs() < 1.0 && (snrs[3] - 50.9).abs() < 1.0);

    println!("\nAll checks passed.");
    Ok(())
}
