//! The example clips contain what their documentation says, as recovered by the crate's own
//! analysis, and render to the same integer PCM on every platform.

use std::num::NonZeroUsize;
use std::time::Duration;

use audio_samples::operations::pitch_tracking::{AudioPitchTracking, PyinConfig};
use audio_samples::operations::rhythm::{AudioRhythm, BeatTrackConfig, OnsetDetectConfig};
use audio_samples::operations::tonal::{AudioTonal, ChromaStftConfig};
use audio_samples::sample_rate;
use audio_samples::utils::generation::{
    EXAMPLE_NOISY_SNR_DB, EXAMPLE_TEMPO_BPM, ExampleClip, example_clip, example_speech_f0,
};
use audio_samples::{AudioEffects, AudioSamples, HarmonicPercussiveConfig, I24};
use ndarray::Array1;

const SR: u32 = 22050;

fn clip64(clip: ExampleClip, secs: u64) -> AudioSamples<'static, f64> {
    example_clip::<f64>(clip, Duration::from_secs(secs), sample_rate!(22050))
}

fn samples(audio: &AudioSamples<'_, f64>) -> Vec<f64> {
    audio.as_slice().unwrap().to_vec()
}

fn rms(x: &[f64]) -> f64 {
    (x.iter().map(|s| s * s).sum::<f64>() / x.len() as f64).sqrt()
}

/// Least-squares SNR of `estimate` as a scaled copy of `reference`, in dB.
fn snr_db(estimate: &[f64], reference: &[f64]) -> f64 {
    let n = estimate.len().min(reference.len());
    let (estimate, reference) = (&estimate[..n], &reference[..n]);
    let gain = estimate
        .iter()
        .zip(reference)
        .map(|(e, r)| e * r)
        .sum::<f64>()
        / reference.iter().map(|r| r * r).sum::<f64>();
    let (mut signal, mut error) = (0.0, 0.0);
    for (e, r) in estimate.iter().zip(reference) {
        signal += (gain * r) * (gain * r);
        error += (e - gain * r) * (e - gain * r);
    }
    10.0 * (signal / error).log10()
}

/// 64-bit FNV-1a, which the Python test suite computes too.
fn fnv1a(bytes: impl Iterator<Item = u8>) -> u64 {
    bytes.fold(0xcbf2_9ce4_8422_2325, |hash, byte| {
        (hash ^ u64::from(byte)).wrapping_mul(0x0000_0100_0000_01b3)
    })
}

fn pcm16_digest(clip: ExampleClip) -> u64 {
    let audio = example_clip::<i16>(clip, Duration::from_secs(2), sample_rate!(22050));
    fnv1a(
        audio
            .as_slice()
            .unwrap()
            .iter()
            .flat_map(|s| s.to_le_bytes()),
    )
}

/// Digests of 2 s of 16-bit PCM at 22050 Hz. A change here changes every user's clips, so it
/// needs a reason, and it must also be made in the Python test suite.
const PCM16_DIGESTS: [(ExampleClip, u64); 5] = [
    (ExampleClip::Chords, 0x4007_76df_e116_b59f),
    (ExampleClip::DrumLoop, 0x29b7_d1a7_ccad_0015),
    (ExampleClip::ChordsAndDrums, 0xae00_60e6_9853_5762),
    (ExampleClip::Speech, 0x2e0a_b50f_7d7c_1305),
    (ExampleClip::NoisyRecording, 0x7a46_1bdd_55b8_acd0),
];

#[test]
fn pcm_digests_are_stable() {
    for (clip, digest) in PCM16_DIGESTS {
        assert_eq!(
            pcm16_digest(clip),
            digest,
            "{} digest is {:#018x}",
            clip.name(),
            pcm16_digest(clip)
        );
    }
}

#[test]
fn audio_carries_the_requested_rate_duration_and_type() {
    for clip in ExampleClip::ALL {
        let i24 = example_clip::<I24>(clip, Duration::from_millis(500), sample_rate!(16000));
        assert_eq!(i24.sample_rate().get(), 16000);
        assert_eq!(i24.samples_per_channel().get(), 8000);
        assert_eq!(i24.num_channels().get(), 1);
        let f32_clip = example_clip::<f32>(clip, Duration::from_millis(250), sample_rate!(8000));
        assert_eq!(f32_clip.samples_per_channel().get(), 2000);
    }
}

#[test]
#[should_panic(expected = "zero samples")]
fn a_duration_of_no_samples_panics() {
    let _ = example_clip::<f32>(ExampleClip::Chords, Duration::ZERO, sample_rate!(8000));
}

#[test]
fn no_clip_clips_at_any_common_rate() {
    for rate in [
        sample_rate!(8000),
        sample_rate!(16000),
        sample_rate!(22050),
        sample_rate!(44100),
        sample_rate!(48000),
        sample_rate!(96000),
    ] {
        for clip in ExampleClip::ALL {
            let audio = example_clip::<f64>(clip, Duration::from_secs(7), rate);
            let peak = audio
                .as_slice()
                .unwrap()
                .iter()
                .fold(0.0_f64, |m, s| m.max(s.abs()));
            assert!(
                (0.3..0.98).contains(&peak),
                "{} at {rate}: peak {peak}",
                clip.name()
            );
        }
    }
}

#[test]
fn a_shorter_clip_is_a_prefix_of_a_longer_one() {
    for clip in ExampleClip::ALL {
        let short = samples(&clip64(clip, 3));
        let long = samples(&clip64(clip, 7));
        assert_eq!(short[..], long[..short.len()], "{}", clip.name());
    }
}

#[test]
fn drum_loop_has_the_stated_tempo_and_onsets() {
    let drums = clip64(ExampleClip::DrumLoop, 16);
    let beats = drums.beat_track(&BeatTrackConfig::librosa()).unwrap();
    // The estimate falls on the grid of the tempogram, whose spacing is 5 BPM around here.
    assert!(
        (beats.tempo - EXAMPLE_TEMPO_BPM).abs() < 3.0,
        "tempo {}",
        beats.tempo
    );
    let times = beats.beats.times();
    let period = (times[times.len() - 1] - times[0]) / (times.len() - 1) as f64;
    let expected = 60.0 / EXAMPLE_TEMPO_BPM;
    assert!((period - expected).abs() < 0.002, "beat period {period}");
    // Beats land on quarter notes of the stated grid, up to the frame quantisation and the lag
    // of the onset envelope.
    for time in &times {
        let offset = time / expected;
        assert!(
            (offset - offset.round()).abs() * expected < 0.06,
            "beat at {time}"
        );
    }

    let onsets = drums.onset_detect(&OnsetDetectConfig::librosa()).unwrap();
    let eighth = expected / 2.0;
    let onset_times = onsets.times();
    let eighths_in_clip = (16.0 / eighth) as usize;
    assert!(
        onset_times.len().abs_diff(eighths_in_clip) <= 2,
        "{} onsets, {eighths_in_clip} eighths",
        onset_times.len()
    );
    for time in &onset_times {
        let offset = time / eighth;
        assert!(
            (offset - offset.round()).abs() * eighth < 0.06,
            "onset at {time}"
        );
    }
}

#[test]
fn chords_have_the_stated_pitch_classes_and_tuning() {
    let chords = clip64(ExampleClip::Chords, 9);
    let chroma = chords
        .chroma_stft(&ChromaStftConfig::librosa().with_tuning(Some(0.0)))
        .unwrap();
    let frames_per_second = f64::from(SR) / 512.0;
    let bar = 4.0 * 60.0 / EXAMPLE_TEMPO_BPM;
    // Pitch classes with C = 0.
    let expected: [[usize; 3]; 4] = [[0, 4, 7], [2, 7, 11], [0, 4, 9], [0, 5, 9]];
    for (i, want) in expected.iter().enumerate() {
        let start = ((i as f64 * bar + 0.3) * frames_per_second) as usize;
        let end = (((i + 1) as f64 * bar - 0.3) * frames_per_second) as usize;
        let mut mean = [0.0; 12];
        for frame in start..end {
            for (pc, m) in mean.iter_mut().enumerate() {
                *m += chroma[[0, pc, frame]];
            }
        }
        let mut order: Vec<usize> = (0..12).collect();
        order.sort_by(|a, b| mean[*b].total_cmp(&mean[*a]));
        let mut top: Vec<usize> = order[..3].to_vec();
        top.sort_unstable();
        assert_eq!(&top, want, "chord {i}, mean chroma {mean:?}");
    }
}

#[test]
fn the_mix_is_exactly_the_stems_and_hpss_separates_them() {
    let chords = samples(&clip64(ExampleClip::Chords, 9));
    let drums = samples(&clip64(ExampleClip::DrumLoop, 9));
    let mix = clip64(ExampleClip::ChordsAndDrums, 9);
    let mixed = samples(&mix);
    for ((m, c), d) in mixed.iter().zip(&chords).zip(&drums) {
        assert_eq!(*m, 0.5 * c + 0.5 * d);
    }

    let (harmonic, percussive) = mix
        .harmonic_percussive(&HarmonicPercussiveConfig::librosa())
        .unwrap();
    let (harmonic, percussive) = (samples(&harmonic), samples(&percussive));
    let harmonic_snr = snr_db(&harmonic, &chords);
    let percussive_snr = snr_db(&percussive, &drums);
    assert!(harmonic_snr > 10.0, "harmonic part: {harmonic_snr} dB");
    assert!(percussive_snr > 4.0, "percussive part: {percussive_snr} dB");
    // Each part is closer to its own stem than to the other one.
    assert!(harmonic_snr > snr_db(&harmonic, &drums) + 10.0);
    assert!(percussive_snr > snr_db(&percussive, &chords) + 3.0);
}

/// Frames whose analysis window lies fully inside a phrase, with the expected F0.
fn frames_inside_phrases(n_frames: usize) -> Vec<(usize, f64)> {
    let hop = 512.0 / f64::from(SR);
    let half_window = 1024.0 / f64::from(SR);
    (0..n_frames)
        .filter_map(|frame| {
            let t = frame as f64 * hop;
            let inside = example_speech_f0(t - half_window)?;
            let _ = (inside, example_speech_f0(t + half_window)?);
            Some((frame, example_speech_f0(t)?))
        })
        .collect()
}

fn f0_rms_cents(f0: &ndarray::Array2<f64>, voiced: &ndarray::Array2<bool>) -> (f64, f64) {
    let frames = frames_inside_phrases(f0.ncols());
    let (mut sum, mut hits) = (0.0, 0usize);
    for (frame, want) in &frames {
        if voiced[[0, *frame]] {
            let cents = 1200.0 * (f0[[0, *frame]] / want).log2();
            sum += cents * cents;
            hits += 1;
        }
    }
    (
        (sum / hits as f64).sqrt(),
        hits as f64 / frames.len() as f64,
    )
}

#[test]
fn pyin_recovers_the_f0_contour() {
    let result = clip64(ExampleClip::Speech, 7)
        .pyin(&PyinConfig::librosa(60.0, 400.0))
        .unwrap();
    let (cents, coverage) = f0_rms_cents(&result.f0, &result.voiced_flag);
    assert!(cents < 15.0, "{cents} cents RMS");
    assert!(coverage > 0.98, "voiced in {coverage} of the phrase frames");
}

#[test]
fn pyin_recovers_the_f0_contour_through_the_noise() {
    let result = clip64(ExampleClip::NoisyRecording, 7)
        .pyin(&PyinConfig::librosa(60.0, 400.0))
        .unwrap();
    let (cents, coverage) = f0_rms_cents(&result.f0, &result.voiced_flag);
    assert!(cents < 20.0, "{cents} cents RMS");
    assert!(coverage > 0.95, "voiced in {coverage} of the phrase frames");
}

#[test]
fn noisy_recording_has_the_stated_snr_and_hum() {
    for rate in [
        sample_rate!(16000),
        sample_rate!(22050),
        sample_rate!(48000),
    ] {
        let duration = Duration::from_secs(6);
        let speech = example_clip::<f64>(ExampleClip::Speech, duration, rate);
        let noisy = example_clip::<f64>(ExampleClip::NoisyRecording, duration, rate);
        let (speech, noisy) = (samples(&speech), samples(&noisy));
        let noise: Vec<f64> = noisy.iter().zip(&speech).map(|(n, s)| n - s).collect();
        let voiced: Vec<f64> = speech
            .iter()
            .enumerate()
            .filter(|(i, _)| example_speech_f0(*i as f64 / f64::from(rate.get())).is_some())
            .map(|(_, s)| *s)
            .collect();
        let snr = 20.0 * (rms(&voiced) / rms(&noise)).log10();
        assert!(
            (snr - EXAMPLE_NOISY_SNR_DB).abs() < 1.0,
            "at {rate} Hz: {snr} dB"
        );

        // The largest low-frequency component of the noise is the 50 Hz hum.
        let correlate = |hz: f64| {
            let (mut re, mut im) = (0.0, 0.0);
            for (i, n) in noise.iter().enumerate() {
                let phase = std::f64::consts::TAU * hz * i as f64 / f64::from(rate.get());
                re += n * phase.cos();
                im += n * phase.sin();
            }
            (re * re + im * im).sqrt() * 2.0 / noise.len() as f64
        };
        let hum = correlate(50.0);
        assert!(hum > 3.0 * correlate(70.0), "at {rate} Hz: hum {hum}");
        assert!(correlate(100.0) > 3.0 * correlate(120.0));
    }
}

#[test]
fn speech_has_the_stated_formants() {
    // Linear prediction at the centre of each vowel, sampled so that the five formants fit.
    let audio = example_clip::<f64>(
        ExampleClip::Speech,
        Duration::from_secs(4),
        sample_rate!(16000),
    );
    let x = samples(&audio);
    let cases: [(f64, [f64; 3]); 5] = [
        (0.3, [730.0, 1090.0, 2440.0]),
        (0.9, [530.0, 1840.0, 2480.0]),
        (1.5, [270.0, 2290.0, 3010.0]),
        (2.1, [570.0, 840.0, 2410.0]),
        (3.3, [300.0, 870.0, 2240.0]),
    ];
    for (centre, formants) in cases {
        let start = ((centre - 0.1) * 16000.0) as usize;
        let n = 3200;
        let window: Vec<f64> = (0..n)
            .map(|i| {
                let w = 0.5 - 0.5 * (std::f64::consts::TAU * i as f64 / (n - 1) as f64).cos();
                w * x[start + i]
            })
            .collect();
        let segment =
            AudioSamples::new_mono(Array1::from_vec(window), sample_rate!(16000)).unwrap();
        let a = segment
            .preemphasis(0.97)
            .unwrap()
            .lpc(NonZeroUsize::new(14).unwrap())
            .unwrap();
        let peaks = envelope_peaks(a.row(0).as_slice().unwrap(), 16000.0);
        for formant in formants {
            let nearest = peaks
                .iter()
                .map(|peak| (peak - formant).abs())
                .fold(f64::INFINITY, f64::min);
            assert!(
                nearest < 0.1 * formant,
                "formant {formant} Hz at {centre} s, envelope peaks at {peaks:?}"
            );
        }
    }
}

/// Frequencies of the local maxima of the all-pole envelope `1 / |A(f)|`, on a 10 Hz grid.
fn envelope_peaks(a: &[f64], sr: f64) -> Vec<f64> {
    let magnitude = |hz: f64| {
        let w = std::f64::consts::TAU * hz / sr;
        let (mut re, mut im) = (0.0, 0.0);
        for (k, c) in a.iter().enumerate() {
            re += c * (w * k as f64).cos();
            im -= c * (w * k as f64).sin();
        }
        1.0 / (re * re + im * im).sqrt()
    };
    let grid: Vec<f64> = (1..799).map(|i| f64::from(i) * 10.0).collect();
    grid.windows(3)
        .filter(|w| magnitude(w[1]) > magnitude(w[0]) && magnitude(w[1]) > magnitude(w[2]))
        .map(|w| w[1])
        .collect()
}
