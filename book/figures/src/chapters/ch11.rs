//! Figures for chapter 11, short-time energy, zero crossings and VAD.

use std::time::Duration;

use audio_samples::operations::types::{VadConfig, VadMethod};
use audio_samples::utils::generation::{MonoSampleBuilder, ToneComponent};
use audio_samples::{
    AudioSamples, AudioVoiceActivityDetection, compound_tone, nzu, sample_rate, white_noise,
};
use guide_figures::*;
use non_empty_slice::{NonEmptySlice, NonEmptyVec};

fn main() -> Result {
    let mut figs = Figures::new("ch11")?;
    vad(&mut figs)?;
    figs.finish()
}

/// Blank tick labels, for the decision panel whose y values are offsets.
fn no_label(_: f64) -> String {
    String::new()
}

/// The chapter's utterance, energy and combined VAD, and the frame features.
fn vad(figs: &mut Figures) -> Result {
    // The utterance exactly as the chapter builds it.
    let sr = sample_rate!(16000);
    let partials: Vec<ToneComponent> = (1..=20)
        .map(|k| {
            let f = 200.0 * k as f64;
            ToneComponent::new(f, 0.3 / (1.0 + ((f - 700.0) / 200.0).powi(2)))
        })
        .collect();
    let vowel = compound_tone::<f64>(
        NonEmptySlice::from_slice(&partials).ok_or("partials")?,
        Duration::from_millis(400),
        sr,
    );
    let speech = MonoSampleBuilder::<f64>::new(sr)
        .silence(Duration::from_millis(300))
        .add_sample(vowel)
        .white_noise(Duration::from_millis(200), 0.1, Some(1))
        .silence(Duration::from_millis(300))
        .build()?;
    let floor = white_noise::<f64>(Duration::from_millis(1200), sr, 1e-3, Some(2));
    let x: Vec<f64> = speech
        .as_slice()
        .ok_or("mono")?
        .iter()
        .zip(floor.as_slice().ok_or("mono")?)
        .map(|(a, b)| a + b)
        .collect();
    let audio: AudioSamples<f64> =
        AudioSamples::from_mono_vec(NonEmptyVec::new(x.clone()).map_err(|_| "empty")?, sr);

    let cfg = VadConfig::default()
        .with_method(VadMethod::Energy)
        .with_hop_size(nzu!(160))?
        .with_frame_size(nzu!(400))?
        .with_energy_threshold_db(-40.0)
        .with_smooth_frames(nzu!(1))
        .with_min_speech_frames(0)
        .with_min_silence_frames(0)
        .with_hangover_frames(nzu!(1));
    let combined = cfg.with_method(VadMethod::Combined);
    let mask_e = audio.voice_activity_mask(&cfg)?.to_vec();
    let mask_c = audio.voice_activity_mask(&combined)?.to_vec();
    check("120 frames", mask_e.len() == 120);
    let regions_e = audio.speech_regions(&cfg)?;
    let regions_c = audio.speech_regions(&combined)?;
    check(
        format!("energy region {regions_e:?} is (4480, 14960)"),
        regions_e == vec![(4_480, 14_960)],
    );
    check(
        format!("combined region {regions_c:?} is (4640, 11600)"),
        regions_c == vec![(4_640, 11_600)],
    );

    // The chapter computes the frame features by hand (the library has no
    // frame-wise ZCR); the same code, checked below against the library mask.
    let (frame, hop) = (400usize, 160usize);
    let feats: Vec<(f64, f64)> = (0..x.len().div_ceil(hop))
        .map(|m| {
            let fr = &x[m * hop..(m * hop + frame).min(x.len())];
            let rms = (fr.iter().map(|v| v * v).sum::<f64>() / frame as f64).sqrt();
            let signs: Vec<f64> = fr
                .iter()
                .filter(|v| **v != 0.0)
                .map(|v| v.signum())
                .collect();
            let zc = signs.windows(2).filter(|w| w[0] != w[1]).count();
            (rms, zc as f64 / (frame - 1) as f64)
        })
        .collect();
    let to_db = |r: f64| 20.0 * r.log10();
    let (silence, voiced, fricative) = (feats[10], feats[50], feats[78]);
    assert_close("silence level (dBFS)", to_db(silence.0), -64.8, 1.0);
    assert_close("vowel level (dBFS)", to_db(voiced.0), -11.7, 0.1);
    assert_close("fricative level (dBFS)", to_db(fricative.0), -24.8, 0.5);
    assert_close("silence ZCR", silence.1, 0.5, 0.06);
    assert_close("fricative ZCR", fricative.1, 0.5, 0.06);
    check(
        "vowel ZCR in (0.06, 0.11)",
        voiced.1 > 0.06 && voiced.1 < 0.11,
    );
    // Threshold plus one frame of hangover reproduces the library's mask.
    let thr = 10f64.powf(-40.0 / 20.0);
    let mut hold = 0;
    let mine: Vec<bool> = feats
        .iter()
        .map(|&(rms, _)| {
            if rms >= thr {
                hold = 1;
                true
            } else if hold > 0 {
                hold -= 1;
                true
            } else {
                false
            }
        })
        .collect();
    check("hand-made VAD equals the library mask", mine == mask_e);

    let fs = 16_000.0;
    // Waveform as a min/max envelope over 5 ms blocks (one period of the
    // 200 Hz vowel): a vertical stroke per block, 480 points in all.
    let mut wave = Vec::new();
    for (b, chunk) in x.chunks(80).enumerate() {
        let t = (b * 80) as f64 / fs;
        let lo = chunk.iter().copied().fold(f64::INFINITY, f64::min);
        let hi = chunk.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        wave.push((t, lo));
        wave.push((t, hi));
    }
    let peak = x.iter().fold(0.0f64, |m, v| m.max(v.abs()));
    let centre = |m: usize| (m * hop + frame / 2) as f64 / fs;
    let level: Vec<(f64, f64)> = feats
        .iter()
        .enumerate()
        .map(|(m, f)| (centre(m), to_db(f.0)))
        .collect();
    let zcr: Vec<(f64, f64)> = feats
        .iter()
        .enumerate()
        .map(|(m, f)| (centre(m), f.1))
        .collect();
    let steps = |mask: &[bool], base: f64| -> Vec<(f64, f64)> {
        let mut v: Vec<(f64, f64)> = mask
            .iter()
            .enumerate()
            .map(|(m, &a)| ((m * hop) as f64 / fs, base + if a { 1.0 } else { 0.0 }))
            .collect();
        v.push((mask.len() as f64 * hop as f64 / fs, v[v.len() - 1].1));
        v
    };
    let t_end = 1.2;

    figs.svg(
        "ch11-vad",
        "The chapter's utterance (silence, vowel, fricative, silence) in 25 ms frames with a \
         10 ms hop. The frame level crosses the -40 dBFS threshold for the vowel and the \
         fricative; the fricative's zero-crossing rate is outside the 0.02 to 0.30 window, so \
         the combined VAD keeps only the vowel. Decisions from voice_activity_mask.",
        (720, 800),
        |area| {
            let p = rows(area, 4);
            let pk = (peak * 1.15 * 10.0).ceil() / 10.0;
            Chart::new("Time (s)", "Amplitude", (0.0, t_end), (-pk, pk))
                .title("Signal")
                .ticks(12, 5)
                .add(Series::new("", wave.clone(), BLUE, Style::Line(1)))
                .draw(&p[0])?;
            Chart::new("Time (s)", "Frame RMS (dBFS)", (0.0, t_end), (-75.0, 0.0))
                .title("Short-time energy")
                .ticks(12, 5)
                .add(Series::line("", level.clone(), BLUE))
                .hrule(-40.0, "threshold -40 dBFS", VERMILION)
                .draw(&p[1])?;
            Chart::new("Time (s)", "ZCR (per sample)", (0.0, t_end), (0.0, 0.65))
                .title("Zero-crossing rate")
                .ticks(12, 5)
                .add(Series::line("", zcr.clone(), GREEN))
                .hrule(0.30, "upper bound 0.30", VERMILION)
                .hrule(0.02, "lower bound 0.02", VERMILION)
                .draw(&p[2])?;
            Chart::new("Time (s)", "Decision", (0.0, t_end), (-0.3, 3.2))
                .title("VAD decisions (high = speech)")
                .ticks(12, 2)
                .y_format(no_label)
                .add(Series::new(
                    "energy VAD",
                    steps(&mask_e, 1.8),
                    BLUE,
                    Style::Steps(2),
                ))
                .add(Series::new(
                    "combined VAD",
                    steps(&mask_c, 0.0),
                    VERMILION,
                    Style::Steps(2),
                ))
                .legend(LegendPos::UpperRight)
                .draw(&p[3])
        },
    )
}
