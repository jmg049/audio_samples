# 11. Short-time energy, zero crossings and VAD

Before a recogniser, a codec or a diariser does any expensive work, it asks a
cheap question: is anyone speaking in this frame? Voice activity detection
(VAD) answers it from two per-frame features that date back to the 1970s:
short-time energy (speech is louder than the background) and the
zero-crossing rate (voiced sounds cross zero slowly, fricatives and noise
quickly). This chapter computes both by hand on a synthetic utterance and
reproduces the library's VAD decisions exactly.

## Concept

**Framing.** The signal $x[n]$ (samples scaled to $[-1, 1]$) is cut into
frames of $F$ samples that start every $H$ samples; frame $m$ covers
$n = mH, \dots, mH + F - 1$. Typical speech settings are $F = 25$ ms and
$H = 10$ ms, i.e. $F = 400$ and $H = 160$ at $f_s = 16$ kHz (the sample rate in
Hz).

**Short-time energy** is reported as an RMS level in dBFS:

$$r_m = \sqrt{\frac{1}{F} \sum_{i=0}^{F-1} x^2[mH + i]}, \qquad L_m = 20 \log_{10} r_m \ \text{dBFS}$$

where $r_m$ is the frame RMS (dimensionless, full scale = 1) and $L_m$ the
level in decibels relative to full scale. 0 dBFS is an RMS of 1 (a full-scale
DC or square wave), so a full-scale sine sits at $-3.01$ dBFS in this
convention; the AES17 convention (0 dBFS = the RMS of a full-scale sine) reads
3.01 dB higher. Where this guide means a *peak* level, $20\log_{10}$ of the
largest absolute sample, it says "dBFS peak". dBFS has no link to
loudness in the room: the same file can be played at any volume (chapter 14).

**Zero-crossing rate (ZCR)** counts sign changes:

$$z_m = \frac{\#\{\text{sign changes in frame } m\}}{F - 1}$$

in crossings per sample. A sinusoid at $f$ Hz crosses zero $2f$ times per
second, so $z \approx 2f / f_s$; conversely a ZCR of $z$ corresponds to a
"dominant frequency" of about $z f_s / 2$ Hz. White noise changes sign on about
half of all sample pairs, $z \approx 0.5$.

**Three classes of speech frame:**

| class | energy | ZCR |
|---|---|---|
| silence (background) | low | high (noise-like) |
| voiced (vowels, nasals) | high | low (periodic, energy below 1 kHz) |
| unvoiced (fricatives such as /s/) | medium | high (noise-like, energy at 4-8 kHz) |

**Decision rules.** An energy VAD marks frame $m$ as speech if
$L_m \ge T$, where $T$ is a threshold in dBFS, equivalently
$r_m \ge 10^{T/20}$. A combined VAD also requires
$z_{min} \le z_m \le z_{max}$, which rejects low hum and noise-like frames
(and, as we shall see, fricatives).

**Post-processing.** Raw decisions flicker. The library smooths them with a
causal majority vote over $S$ frames, then applies **hangover**: after the last
active frame the next $H_o$ frames stay active, which bridges short pauses
(stop closures, the gap before a final plosive) at the cost of a later
offset. Finally, speech runs shorter than a minimum and silence gaps shorter
than a minimum are removed.

**Two ZCR conventions in the library.** The VAD's ZCR is per frame, in
crossings per sample, and skips exact zeros.
[`AudioStatistics::zero_crossing_rate`](crate::AudioStatistics::zero_crossing_rate)
is for the **whole signal** and is in **crossings per second**; its crossing
rule treats an exact zero as non-positive. There is no frame-wise ZCR function,
so this chapter computes it directly.

## With the library

Build an utterance at 16 kHz with the generators: 300 ms of silence, a 400 ms
"vowel" (harmonics of 200 Hz shaped by a resonance at 700 Hz), a 200 ms
"fricative" (white noise), 300 ms of silence, plus a $-64.8$ dBFS noise floor.
Run an energy VAD with 25 ms frames, 10 ms hop, a $-40$ dBFS threshold and no
smoothing or minimum-run rules so that only the threshold and the hangover act.

```rust
use audio_samples::operations::types::{VadConfig, VadMethod};
use audio_samples::utils::generation::{MonoSampleBuilder, ToneComponent};
use audio_samples::{AudioSamples, AudioVoiceActivityDetection, compound_tone, nzu, sample_rate, white_noise};
use non_empty_slice::{NonEmptySlice, NonEmptyVec};
use std::time::Duration;

let sr = sample_rate!(16000);
let partials: Vec<ToneComponent> = (1..=20)
    .map(|k| {
        let f = 200.0 * k as f64;
        ToneComponent::new(f, 0.3 / (1.0 + ((f - 700.0) / 200.0).powi(2)))
    })
    .collect();
let vowel = compound_tone::<f64>(NonEmptySlice::from_slice(&partials).unwrap(), Duration::from_millis(400), sr);
let speech = MonoSampleBuilder::<f64>::new(sr)
    .silence(Duration::from_millis(300))
    .add_sample(vowel)
    .white_noise(Duration::from_millis(200), 0.1, Some(1)) // the "s"
    .silence(Duration::from_millis(300))
    .build()
    .unwrap();
let floor = white_noise::<f64>(Duration::from_millis(1200), sr, 1e-3, Some(2));
let x: Vec<f64> = speech.as_slice().unwrap().iter().zip(floor.as_slice().unwrap()).map(|(a, b)| a + b).collect();
let audio: AudioSamples<f64> = AudioSamples::from_mono_vec(NonEmptyVec::new(x).unwrap(), sr);

let cfg = VadConfig::default()
    .with_method(VadMethod::Energy)
    .with_hop_size(nzu!(160)).unwrap() // set the hop first: it must stay <= the frame
    .with_frame_size(nzu!(400)).unwrap()
    .with_energy_threshold_db(-40.0)
    .with_smooth_frames(nzu!(1))
    .with_min_speech_frames(0)
    .with_min_silence_frames(0)
    .with_hangover_frames(nzu!(1));
let mask = audio.voice_activity_mask(&cfg).unwrap();
assert_eq!(mask.len().get(), 120); // ceil(19200 / 160) frames (pad_end = true)

// Vowel + fricative form one region, 4480..14960 samples = 0.280-0.935 s
// (the true speech is 0.300-0.900 s).
assert_eq!(audio.speech_regions(&cfg).unwrap(), vec![(4_480, 14_960)]);

// The combined rule (energy AND 0.02 <= ZCR <= 0.30) rejects the noise-like
// fricative and keeps only the vowel (true extent 4800..11200).
let combined = cfg.with_method(VadMethod::Combined);
assert_eq!(audio.speech_regions(&combined).unwrap(), vec![(4_640, 11_600)]);
```

Why those boundaries? The region starts at the first frame that overlaps the
vowel at all: frame 28 starts at sample 4480 and its last 80 samples are
vowel, which is enough to exceed $-40$ dBFS. It ends at the end of the first
inactive frame after the hangover: the last frame touching the fricative is
frame 89, hangover keeps frame 90 active, and frame 91 ends at
$91 \cdot 160 + 400 = 14\,960$. The combined region starts one frame later
because frame 28 is mostly noise floor and therefore has a noise-like ZCR.

## Under the hood

Compute the two features for every frame in plain Rust, check them against
theory, and rebuild the VAD mask with nothing but a threshold and a hangover
counter.

```rust
# use audio_samples::operations::types::{VadConfig, VadMethod};
# use audio_samples::utils::generation::{MonoSampleBuilder, ToneComponent};
# use audio_samples::{AudioSamples, AudioVoiceActivityDetection, compound_tone, nzu, sample_rate, white_noise};
# use non_empty_slice::{NonEmptySlice, NonEmptyVec};
# use std::time::Duration;
# let sr = sample_rate!(16000);
# let partials: Vec<ToneComponent> = (1..=20).map(|k| { let f = 200.0 * k as f64;
#     ToneComponent::new(f, 0.3 / (1.0 + ((f - 700.0) / 200.0).powi(2))) }).collect();
# let vowel = compound_tone::<f64>(NonEmptySlice::from_slice(&partials).unwrap(), Duration::from_millis(400), sr);
# let speech = MonoSampleBuilder::<f64>::new(sr).silence(Duration::from_millis(300)).add_sample(vowel)
#     .white_noise(Duration::from_millis(200), 0.1, Some(1)).silence(Duration::from_millis(300)).build().unwrap();
# let floor = white_noise::<f64>(Duration::from_millis(1200), sr, 1e-3, Some(2));
# let x: Vec<f64> = speech.as_slice().unwrap().iter().zip(floor.as_slice().unwrap()).map(|(a, b)| a + b).collect();
# let audio: AudioSamples<f64> = AudioSamples::from_mono_vec(NonEmptyVec::new(x.clone()).unwrap(), sr);
# let cfg = VadConfig::default().with_method(VadMethod::Energy).with_hop_size(nzu!(160)).unwrap()
#     .with_frame_size(nzu!(400)).unwrap().with_energy_threshold_db(-40.0).with_smooth_frames(nzu!(1))
#     .with_min_speech_frames(0).with_min_silence_frames(0).with_hangover_frames(nzu!(1));
// `x` is the utterance from the previous example (hidden lines rebuild it).
let (frame, hop) = (400usize, 160usize);
let feats: Vec<(f64, f64)> = (0..x.len().div_ceil(hop))
    .map(|m| {
        let fr = &x[m * hop..(m * hop + frame).min(x.len())];
        // RMS: divide by the full frame length (a short last frame counts as zero-padded).
        let rms = (fr.iter().map(|v| v * v).sum::<f64>() / frame as f64).sqrt();
        // ZCR: sign changes between successive non-zero samples, / (F - 1).
        let signs: Vec<f64> = fr.iter().filter(|v| **v != 0.0).map(|v| v.signum()).collect();
        let zc = signs.windows(2).filter(|w| w[0] != w[1]).count();
        (rms, zc as f64 / (frame - 1) as f64)
    })
    .collect();
let db = |r: f64| 20.0 * r.log10();
let (silence, vowel, fricative) = (feats[10], feats[50], feats[78]); // at 0.1, 0.5, 0.78 s

// Silence: the noise floor, RMS 1e-3/sqrt(3) = -64.8 dBFS (random: +-1 dB).
assert!((db(silence.0) + 64.8).abs() < 1.0);
// Vowel: sum of harmonics, RMS sqrt(sum a_k^2 / 2) = -11.7 dBFS. A 25 ms
// frame holds exactly 5 periods of 200 Hz, so the frame RMS matches to 0.05 dB.
let vowel_rms = (1..=20)
    .map(|k| (0.3 / (1.0 + ((200.0 * k as f64 - 700.0) / 200.0).powi(2))).powi(2) / 2.0)
    .sum::<f64>()
    .sqrt();
assert!((db(vowel_rms) + 11.7).abs() < 0.05);
assert!((db(vowel.0) - db(vowel_rms)).abs() < 0.05);
// Fricative: uniform noise of amplitude 0.1, RMS 0.1/sqrt(3) = -24.8 dBFS (+-0.5 dB).
assert!((db(fricative.0) + 24.8).abs() < 0.5);
// ZCR: noise-like frames near 0.5; the vowel near 2 * 780 / 16000 = 0.098,
// set by its strongest partials (600 and 800 Hz).
assert!((silence.1 - 0.5).abs() < 0.06 && (fricative.1 - 0.5).abs() < 0.06);
assert!(vowel.1 > 0.06 && vowel.1 < 0.11);

// The VAD: threshold, then hangover of 1 frame.
let thr = 10f64.powf(-40.0 / 20.0);
let mut hold = 0;
let mine: Vec<bool> = feats
    .iter()
    .map(|&(rms, _)| {
        if rms >= thr { hold = 1; true } else if hold > 0 { hold -= 1; true } else { false }
    })
    .collect();
assert_eq!(mine, audio.voice_activity_mask(&cfg).unwrap().to_vec()); // identical
```

The library's whole-signal
[`zero_crossing_rate`](crate::AudioStatistics::zero_crossing_rate) is simply
[`zero_crossings`](crate::AudioStatistics::zero_crossings) divided by the
duration in seconds:

```rust
use audio_samples::{AudioStatistics, sample_rate, sine_wave};
use std::time::Duration;

// 500 Hz for 1 s: about 2 * 500 = 1000 crossings per second.
let tone = sine_wave::<f64>(500.0, Duration::from_secs(1), sample_rate!(16000), 0.5);
let per_second = tone.zero_crossing_rate();
assert_eq!(per_second, tone.zero_crossings() as f64 / 1.0);
// Within 2 crossings of 1000 (where the first and last samples fall).
assert!((per_second - 1_000.0).abs() <= 2.0);
// In crossings per sample, as the VAD uses: about 2 f / fs = 0.0625.
assert!((per_second / 16_000.0 - 0.0625).abs() < 2e-4);
```

## Ask the library to explain

[`VadConfig`](crate::operations::types::VadConfig) implements
[`Explain`](crate::educational::Explain). Through a signal it converts frames,
hops and hangover into milliseconds and the ZCR window into equivalent tone
frequencies. The default configuration (1024-sample frames) is tuned for
music-rate audio; at 16 kHz the explainer points out that its frames are long
for speech.

```rust
use audio_samples::educational::ExplainFor;
use audio_samples::operations::types::VadConfig;
use audio_samples::{nzu, sample_rate, silence};
use std::time::Duration;

let audio = silence::<f32>(Duration::from_millis(1200), sample_rate!(16000));
let doc = audio.explain(&VadConfig::default());
println!("{doc}");
assert_eq!(doc.quantity("Frame duration").unwrap().value, 64.0); // 1024 / 16 kHz
assert_eq!(doc.quantity("ZCR lower bound as a tone").unwrap().value, 160.0); // 0.02 * 8 kHz
assert_eq!(doc.quantity("ZCR upper bound as a tone").unwrap().value, 2_400.0);
assert!(doc.has_warning_containing("64 ms"));

let speech_cfg = VadConfig::default()
    .with_hop_size(nzu!(160)).unwrap()
    .with_frame_size(nzu!(400)).unwrap();
let doc = audio.explain(&speech_cfg);
assert_eq!(doc.quantity("Frame duration").unwrap().value, 25.0);
assert_eq!(doc.quantity("Hangover").unwrap().value, 20.0); // 2 frames x 10 ms
assert_eq!(doc.quantity("Frames").unwrap().value, 120.0);
```

## Exercises

1. **Levels.** Predict the RMS level in dBFS of (a) a full-scale sine,
   (b) the library's `white_noise` with amplitude 0.1. Check with
   [`AudioStatistics::rms`](crate::AudioStatistics::rms).
2. **Hangover.** With the energy VAD above, predict how far the end of the
   speech region moves when the hangover goes from 1 to 6 frames. Check with
   `speech_regions`.
3. **A loud hum.** A 100 Hz tone peaking at $-10$ dBFS (amplitude 0.316) is
   not speech. Predict its RMS level in this chapter's (RMS) convention and its
   per-sample ZCR, then decide which rule rejects it, energy or combined.

## Answers

1. (a) $20\log_{10}(1/\sqrt 2) = -3.01$ dBFS. (b) Uniform noise on
   $[-a, a)$ has RMS $a/\sqrt 3$, so $20\log_{10}(0.1/\sqrt 3) = -24.8$ dBFS.

```rust
use audio_samples::{AudioStatistics, sample_rate, sine_wave, white_noise};
use std::time::Duration;

let sine = sine_wave::<f64>(1_000.0, Duration::from_secs(1), sample_rate!(16000), 1.0);
assert!((20.0 * sine.rms().log10() + 3.0103).abs() < 1e-3); // whole periods: exact
let noise = white_noise::<f64>(Duration::from_secs(1), sample_rate!(16000), 0.1, Some(7));
// 16000 random samples: the RMS estimate is within about 1 % (0.1 dB).
assert!((20.0 * noise.rms().log10() + 24.77).abs() < 0.1);
```

2. Each extra hangover frame keeps one more 10 ms frame active, so the end
   moves by $5 \times 160 = 800$ samples (50 ms): from 14960 to 15760.

```rust
# use audio_samples::operations::types::{VadConfig, VadMethod};
# use audio_samples::utils::generation::{MonoSampleBuilder, ToneComponent};
# use audio_samples::{AudioSamples, AudioVoiceActivityDetection, compound_tone, nzu, sample_rate, white_noise};
# use non_empty_slice::{NonEmptySlice, NonEmptyVec};
# use std::time::Duration;
# let sr = sample_rate!(16000);
# let partials: Vec<ToneComponent> = (1..=20).map(|k| { let f = 200.0 * k as f64;
#     ToneComponent::new(f, 0.3 / (1.0 + ((f - 700.0) / 200.0).powi(2))) }).collect();
# let vowel = compound_tone::<f64>(NonEmptySlice::from_slice(&partials).unwrap(), Duration::from_millis(400), sr);
# let speech = MonoSampleBuilder::<f64>::new(sr).silence(Duration::from_millis(300)).add_sample(vowel)
#     .white_noise(Duration::from_millis(200), 0.1, Some(1)).silence(Duration::from_millis(300)).build().unwrap();
# let floor = white_noise::<f64>(Duration::from_millis(1200), sr, 1e-3, Some(2));
# let x: Vec<f64> = speech.as_slice().unwrap().iter().zip(floor.as_slice().unwrap()).map(|(a, b)| a + b).collect();
# let audio: AudioSamples<f64> = AudioSamples::from_mono_vec(NonEmptyVec::new(x).unwrap(), sr);
# let cfg = VadConfig::default().with_method(VadMethod::Energy).with_hop_size(nzu!(160)).unwrap()
#     .with_frame_size(nzu!(400)).unwrap().with_energy_threshold_db(-40.0).with_smooth_frames(nzu!(1))
#     .with_min_speech_frames(0).with_min_silence_frames(0);
let end = |h| audio.speech_regions(&cfg.with_hangover_frames(h)).unwrap()[0].1;
assert_eq!(end(nzu!(1)), 14_960);
assert_eq!(end(nzu!(6)), 15_760);
assert_eq!(end(nzu!(6)) - end(nzu!(1)), 5 * 160);
```

3. Its RMS level is $-10 - 3.01 = -13.0$ dBFS (a sine's RMS is its peak over
   $\sqrt 2$). Its ZCR is about $2 \cdot 100 / 16000 = 0.0125$ crossings per
   sample, below $z_{min} = 0.02$, so the combined rule rejects it while the
   energy rule ($-13.0$ dBFS, far above $-40$) accepts it.

```rust
use audio_samples::operations::types::{VadConfig, VadMethod};
use audio_samples::{AudioVoiceActivityDetection, nzu, sample_rate, sine_wave};
use std::time::Duration;

let amp = 10f64.powf(-10.0 / 20.0); // -10 dBFS peak
let hum = sine_wave::<f64>(100.0, Duration::from_millis(500), sample_rate!(16000), amp);
let x = hum.as_slice().unwrap();
// 500 ms holds 50 whole periods, so the RMS is exactly amp / sqrt(2): -13.0 dBFS.
let rms_db = 20.0 * (x.iter().map(|v| v * v).sum::<f64>() / x.len() as f64).sqrt().log10();
assert!((rms_db + 13.01).abs() < 0.01);
let fr = &x[1_000..1_400];
let zc = fr.windows(2).filter(|w| (w[0] > 0.0) != (w[1] > 0.0)).count();
let zcr = zc as f64 / 399.0;
assert!((zcr - 0.0125).abs() < 0.003); // 5 crossings in 400 samples

let cfg = VadConfig::default()
    .with_hop_size(nzu!(160)).unwrap()
    .with_frame_size(nzu!(400)).unwrap()
    .with_energy_threshold_db(-40.0);
let active = |m| hum.voice_activity_mask(&cfg.with_method(m)).unwrap().iter().filter(|&&v| v).count();
assert!(active(VadMethod::Energy) > 0);
assert_eq!(active(VadMethod::Combined), 0);
```

## Common pitfalls

- **Units of ZCR.** The VAD uses crossings per sample (per frame);
  [`zero_crossing_rate`](crate::AudioStatistics::zero_crossing_rate) returns
  crossings per **second** for the whole signal; librosa's
  `zero_crossing_rate` returns a per-frame fraction (crossings / frame length,
  centred frames). Divide by $f_s$ to compare.
- **dBFS conventions.** Here a full-scale sine is $-3.01$ dBFS (RMS
  convention). AES17 would call it 0 dBFS.
- **Default frames are long.** `VadConfig::default()` uses 1024 / 512
  samples (64 / 32 ms at 16 kHz); speech VAD normally uses 20-30 ms frames.
- **Fricatives fail the combined rule.** /s/ and /f/ have speech-like energy
  but noise-like ZCR, so the default ZCR window (0.02-0.30) drops them. Use the
  energy rule with hangover if you need whole words.
- **Smoothing delays onsets.** The majority vote looks only backwards, so
  onsets are reported about $(S - 1)/2$ frames late; hangover delays offsets.
- **Regions are generous.** A region starts at the first frame touching speech
  and ends at the end of the first inactive frame, so it can extend a frame
  beyond the true speech on both sides.

## References

- L. R. Rabiner and M. R. Sambur, "An algorithm for determining the endpoints of isolated utterances", *Bell System Technical Journal* 54(2), 297-315, 1975. The classic energy and zero-crossing detector.
- L. R. Rabiner and R. W. Schafer, *Digital Processing of Speech Signals*, Prentice-Hall, 1978.
- AES17, *AES standard method for digital audio engineering: measurement of digital audio equipment*, Audio Engineering Society (the definition of dBFS as the RMS of a full-scale sine).

## See also

- [`AudioVoiceActivityDetection::voice_activity_mask`](crate::AudioVoiceActivityDetection::voice_activity_mask),
  [`AudioVoiceActivityDetection::speech_regions`](crate::AudioVoiceActivityDetection::speech_regions),
  [`VadConfig`](crate::operations::types::VadConfig),
  [`VadMethod`](crate::operations::types::VadMethod)
- [`AudioStatistics::rms`](crate::AudioStatistics::rms),
  [`AudioStatistics::zero_crossings`](crate::AudioStatistics::zero_crossings),
  [`AudioStatistics::zero_crossing_rate`](crate::AudioStatistics::zero_crossing_rate)
- [`compound_tone`](crate::compound_tone), [`white_noise`](crate::white_noise),
  [`MonoSampleBuilder`](crate::utils::generation::MonoSampleBuilder)
- [`Explain`](crate::educational::Explain), [`ExplainFor`](crate::educational::ExplainFor)
- Previous chapter: [10. Cepstrum, mel and MFCC](10_cepstrum_mel_mfcc.md);
  next chapter: [12. Dynamics](12_dynamics.md)
- Lab: `examples/tutorial_11_energy_vad.rs`
  (`cargo run --release --example tutorial_11_energy_vad --features educational,full_no_plotting`)
