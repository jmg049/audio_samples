# 13. Onsets and tempo

Where does each note or drum hit begin, and how fast is the music? Onset
detection marks the instants at which new sound energy appears; tempo
estimation looks for the period with which those instants repeat. Both are
built on an *onset detection function* (ODF), one number per analysis frame,
and both inherit its quirks, so this chapter is careful about units (samples,
frames, seconds, beats per minute) and honest about where the library's
defaults fall short.

## Concept

**Frames.** The signal is analysed in frames hopped by $H$ samples; frame $t$
is centred at $t H / f_s$ seconds ($f_s$ the sample rate in Hz). At
$H = 512$ and $f_s = 44.1$ kHz a frame lasts $H/f_s = 11.61$ ms and there are
$f_s/H = 86.13$ frames per second.

**Spectral flux.** With $|X_b\[t\]|$ the magnitude of frequency bin $b$ in frame
$t$, the half-wave-rectified flux is

$$o\[t\] = \sum_{b} \max\left(0, \, |X_b\[t\]| - |X_b\[t-1\]|\right), \qquad o\[0\] = 0$$

Only *increases* count, so note offsets do not create onsets. This crate
evaluates it on a constant-Q transform (CQT) whose $B O$ bins (B per octave,
O octaves) start at $f_{min}$ Hz, with an analysis window of
$N_w = \lfloor 4 f_s / f_{min} \rfloor$ samples (four periods of the lowest
bin). The magnitudes are **not** squared, despite the method's name ("energy").
Optionally, values below $\mu$ times a running median of $L$ frames are set to
zero (`adaptive_threshold`, $\mu$ = `adaptive_threshold_multiplier`).

**Peak picking**, as implemented by
[`pick_peaks`](crate::operations::peak_picking::pick_peaks), all in **frames**:
optional pre-emphasis and median filtering; normalisation by the maximum; a
threshold $\theta\[i\] = \mathrm{clamp}(\max_{|j - i| \le w/2} o\[j\] - \delta, \, \theta_{lo}, \theta_{hi})$
(window $w$ frames, $\delta = 0.05$, $\theta_{lo} = 0.01$, $\theta_{hi} = 1$ by default);
strict local maxima above it; finally, peaks are accepted strongest first and
dropped if closer than $s$ frames to an accepted one. A peak at frame $i$ is
reported at $t_i = i H / f_s$ seconds.

**Tempo.** A tempo of $B$ beats per minute is an inter-beat interval of
$60/B$ seconds, i.e. a lag of

$$L = \frac{60 \, f_s}{B \, H} \ \text{frames}, \qquad B = \frac{60 \, f_s}{H \, L}$$

[`estimate_tempo`](crate::operations::AudioBeatTracking::estimate_tempo)
smooths the ODF (7-frame moving average), compresses it ($\ln(1 + 0.5 x)$),
removes the mean and computes $a(L) = \frac{1}{n - L} \sum_i c\[i\] \, c\[i-L\]$
for the integer lags that map into 40-240 BPM. Each lag is weighted by a
log-normal prior centred on 120 BPM, $w(B) = \exp\left(-\frac{1}{2} \log_2^2 (B/120)\right)$,
and the best $a(L) w(B)$ wins. Two consequences: estimates lie on the grid
$60 f_s / (H L)$ (117.45, 120.19, 123.05 BPM around 120), and a periodic
envelope correlates at $2L$ and $3L$ too, so **octave errors** (half or a third
of the tempo) remain possible: the prior only biases the choice.

## With the library

A click track: eight decaying noise bursts, one every 0.5 s (120 BPM). We use
the configuration from the [onset module example](crate::operations::onset):
the default configuration finds nothing here (see below).

```rust
use audio_samples::operations::onset::OnsetDetectionConfig;
use audio_samples::{AudioOnsetDetection, AudioSamples, sample_rate};
use ndarray::Array1;
use std::num::NonZeroUsize;

// Bursts at 0.25, 0.75, 1.25 and 1.75 s; 2 s at 44.1 kHz.
let sr = 44_100usize;
let mut x = vec![0.0f64; 2 * sr];
let mut seed = 0x2545_F491_4F6C_DD1Du64;
for k in 0..4 {
    let start = ((0.25 + 0.5 * k as f64) * sr as f64) as usize;
    for i in 0..256 {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        let r = (seed >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0;
        x[start + i] = 0.9 * (-(i as f64) / 24.0).exp() * r;
    }
}
let audio = AudioSamples::new_mono(Array1::from(x), sample_rate!(44100)).unwrap();

assert!(audio.detect_onsets(&OnsetDetectionConfig::default()).unwrap().is_empty());

let mut config = OnsetDetectionConfig::default();
config.peak_picking.median_filter = false; // keep one-frame peaks
config.peak_picking.pre_emphasis = false;
config.peak_picking.adaptive_threshold.window_size = 11; // frames, about 0.13 s
config.peak_picking.min_peak_separation = NonZeroUsize::new(5).unwrap(); // frames, about 58 ms
let onsets = audio.detect_onsets(&config).unwrap();
println!("{onsets:.4?}");
assert_eq!(onsets.len(), 4);
// Peaks land on frames 19, 62, 105, 148: 43 frames apart (0.4992 s, not 0.5 s).
let frames: Vec<f64> = onsets.iter().map(|t| (t * 44_100.0 / 512.0).round()).collect();
assert_eq!(frames, vec![19.0, 62.0, 105.0, 148.0]);
// Reported times precede the bursts by 29 to 32 ms: the 72.7 ms window
// centred on frame t already contains a burst that starts after its centre.
for (k, t) in onsets.iter().enumerate() {
    let lead = 0.25 + 0.5 * k as f64 - t;
    assert!(lead > 0.029 && lead < 0.032, "lead {lead}");
}
```

The lead grows by 0.77 ms per burst because onset times are quantised to
whole frames: 0.5 s is 43.07 frames, and the detector can only report 43.

**Why the defaults fail.** The default [`PeakPickingConfig`](crate::operations::types::PeakPickingConfig)
sizes (threshold window 1024, separation 512, median length 3) were chosen as
if they counted audio samples, but they count ODF frames. Switching them on one
at a time from the working configuration (the lab, on 8 bursts) gives: the
length-3 median deletes every one-frame peak (0 onsets); a 1024-frame
threshold window, clipped to the signal length, makes the threshold the global
maximum minus 0.05 (1 onset); a 512-frame separation, reduced to a quarter of
the signal (91 frames, 1.06 s), keeps only 3; pre-emphasis changes nothing on
this signal (8). Note also that `threshold`, `min_onset_interval_secs` and
`pre_emphasis` on [`OnsetDetectionConfig`](crate::operations::onset::OnsetDetectionConfig)
itself are not read by `detect_onsets` (the explanation below says so).

## Under the hood

**The ODF.** Compute the CQT magnitude with the same parameters and sum the
rectified frame-to-frame increases; with the median-based zeroing switched off,
the result must equal
[`onset_detection_function`](crate::AudioOnsetDetection::onset_detection_function).
Then re-implement peak picking for this configuration and convert frames to
seconds.

```rust
use audio_samples::operations::onset::OnsetDetectionConfig;
use audio_samples::{AudioOnsetDetection, AudioSamples, AudioTransforms, sample_rate};
use ndarray::Array1;
use spectrograms::{SpectrogramParams, StftParams};
use std::num::NonZeroUsize;

# let sr = 44_100usize;
# let mut x = vec![0.0f64; 2 * sr];
# let mut seed = 0x2545_F491_4F6C_DD1Du64;
# for k in 0..4 {
#     let start = ((0.25 + 0.5 * k as f64) * sr as f64) as usize;
#     for i in 0..256 {
#         seed ^= seed << 13; seed ^= seed >> 7; seed ^= seed << 17;
#         let r = (seed >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0;
#         x[start + i] = 0.9 * (-(i as f64) / 24.0).exp() * r;
#     }
# }
# let audio = AudioSamples::new_mono(Array1::from(x), sample_rate!(44100)).unwrap();
let fs = 44_100.0;
let mut config = OnsetDetectionConfig::default();
config.peak_picking.median_filter = false;
config.peak_picking.pre_emphasis = false;
config.peak_picking.adaptive_threshold.window_size = 11;
config.peak_picking.min_peak_separation = NonZeroUsize::new(5).unwrap();

// 1. Spectral flux on the CQT magnitude.
let mut raw = config.clone();
raw.adaptive_threshold = false;
let (_, odf_lib) = audio.onset_detection_function(&raw).unwrap();
let n_w = raw.effective_window_size(fs); // floor(4 * 44100 / 55) = 3207 samples
assert_eq!(n_w.get(), 3207);
let stft = StftParams::builder()
    .n_fft(n_w)
    .hop_size(raw.hop_size)
    .window(raw.cqt_params.window.clone())
    .centre(true)
    .build()
    .unwrap();
let cqt = audio
    .cqt_magnitude_spectrogram(&SpectrogramParams::new(stft, fs).unwrap(), &raw.cqt_params)
    .unwrap();
let m = cqt.data(); // (bins, frames)
for t in 1..m.ncols() {
    let flux: f64 = (0..m.nrows()).map(|b| (m[[b, t]] - m[[b, t - 1]]).max(0.0)).sum();
    // Same additions in the same order: exact agreement.
    assert_eq!(flux, odf_lib[t]);
}

// 2. Peak picking with the configured threshold (on the ODF detect_onsets uses).
let (_, odf) = audio.onset_detection_function(&config).unwrap();
let max = odf.iter().fold(0.0f64, |a, v| a.max(v.abs()));
let o: Vec<f64> = odf.iter().map(|v| v / max).collect();
let (half, delta, sep) = (11 / 2, 0.05, 5);
let mut candidates: Vec<(usize, f64)> = (1..o.len() - 1)
    .filter(|&i| {
        let local_max = o[i.saturating_sub(half)..(i + half + 1).min(o.len())]
            .iter()
            .fold(f64::NEG_INFINITY, |a, v| a.max(*v));
        o[i] > o[i - 1] && o[i] > o[i + 1] && o[i] > (local_max - delta).clamp(0.01, 1.0)
    })
    .map(|i| (i, o[i]))
    .collect();
candidates.sort_by(|a, b| b.1.total_cmp(&a.1)); // strongest first
let mut peaks: Vec<usize> = Vec::new();
for (i, _) in candidates {
    if peaks.iter().all(|&p| p.abs_diff(i) >= sep) {
        peaks.push(i);
    }
}
peaks.sort_unstable();

// 3. Frames to seconds: t = i H / f_s.
let ours: Vec<f64> = peaks.iter().map(|&i| i as f64 * 512.0 / fs).collect();
assert_eq!(ours, audio.detect_onsets(&config).unwrap());
```

The lab also prints the ODF peak heights for eight identical bursts: 0.3, 0.3,
1.0, 2.0, 3.3, 3.6, 11.5 and 10.4. The constant-Q transform of the external
`spectrograms` crate responds very differently to identical clicks (a factor
of 40 here; the onset module documents this limitation), which is why a
*local* threshold window matters.

**Tempo.** [`onset_strength_envelope`](crate::operations::beat::onset_strength_envelope)
is public, so the autocorrelation and the prior can be written out and
compared with `estimate_tempo`. First at 120 BPM, then at 80 BPM.

```rust
use audio_samples::operations::AudioBeatTracking;
use audio_samples::operations::beat::{BeatTrackingConfig, onset_strength_envelope};
use audio_samples::operations::onset::OnsetDetectionConfig;
use audio_samples::{AudioSamples, sample_rate};
use ndarray::Array1;

let (sr, fs, hop) = (44_100usize, 44_100.0, 512.0);
let clicks = |bpm: f64, seconds: f64| {
    let mut x = vec![0.0f64; (seconds * fs) as usize];
    let mut seed = 0x2545_F491_4F6C_DD1Du64;
    let mut t = 0.1;
    while t < seconds - 0.1 {
        let start = (t * sr as f64) as usize;
        for i in 0..256 {
            seed ^= seed << 13; seed ^= seed >> 7; seed ^= seed << 17;
            let r = (seed >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0;
            x[start + i] = 0.9 * (-(i as f64) / 24.0).exp() * r;
        }
        t += 60.0 / bpm;
    }
    AudioSamples::new_mono(Array1::from(x), sample_rate!(44100)).unwrap()
};

let tempo = |env: &[f64]| -> (f64, usize) {
    let frame_rate = fs / hop; // 86.13 frames per second
    let mean = env.iter().sum::<f64>() / env.len() as f64;
    let c: Vec<f64> = env.iter().map(|v| v - mean).collect();
    let (lo, hi) = ((60.0 * frame_rate / 240.0) as usize, (60.0 * frame_rate / 40.0).ceil() as usize);
    let mut best = (f64::NEG_INFINITY, lo);
    for lag in lo..=hi.min(c.len() - 1) {
        let a = (lag..c.len()).map(|i| c[i] * c[i - lag]).sum::<f64>() / (c.len() - lag) as f64;
        if a <= 0.0 {
            continue;
        }
        let z = (60.0 * frame_rate / lag as f64 / 120.0).log2();
        let score = a * (-0.5 * z * z).exp();
        if score > best.0 {
            best = (score, lag);
        }
    }
    (60.0 * frame_rate / best.1 as f64, best.1)
};

let config = BeatTrackingConfig::new(120.0, None, OnsetDetectionConfig::default());
for (bpm, expected_lag) in [(120.0, 43), (80.0, 129)] {
    let audio = clicks(bpm, 8.5);
    let lib = audio.estimate_tempo(&config).unwrap();
    let env = onset_strength_envelope(&audio, &config.onset_config, None).unwrap();
    let (ours, lag) = tempo(&env);
    println!("{bpm} BPM clicks: estimate {lib:.2} BPM (lag {lag} frames)");
    assert!((ours - lib).abs() < 1e-9);
    assert_eq!(lag, expected_lag);
}
// 120 BPM -> lag 43 -> 120.19 BPM (the nearest grid point).
// 80 BPM  -> lag 129 -> 40.06 BPM: half the tempo.
assert!((60.0 * fs / (hop * 43.0) - 120.19).abs() < 0.005);
assert!((60.0 * fs / (hop * 129.0) - 40.06).abs() < 0.005);
```

Why half? The lab (a ten-click track, slightly different from the twelve-click
track above) prints the ODF at each of its 80 BPM clicks: 0.88, 0.0,
2.6, 0.08, 13.42, 0.71, 0.0, 3.27, 0.03, 8.69. Every other click nearly
vanishes (the CQT response variation again), so the envelope really does
repeat every *two* beats: its autocorrelation at lag 129 (0.00240) is twenty
times that at lag 65 (0.00012), more than the prior (0.29 against 0.84) can
overcome. At 174 BPM the lab finds 58.07 BPM, a third of the tempo (lag 89,
almost exactly $3 \times 29.7$). The estimate is right at 60, 100, 120 and
140 BPM (60.09, 99.38, 120.19, 139.67).

## Ask the library to explain

```rust
use audio_samples::educational::{Explain, ExplainContext};
use audio_samples::operations::onset::OnsetDetectionConfig;
use std::num::NonZeroUsize;

let mut config = OnsetDetectionConfig::default();
config.peak_picking.median_filter = false;
config.peak_picking.pre_emphasis = false;
config.peak_picking.adaptive_threshold.window_size = 11;
config.peak_picking.min_peak_separation = NonZeroUsize::new(5).unwrap();

let doc = config.explain_with(&ExplainContext::new().with_sample_rate(44_100.0));
println!("{doc}");
assert_eq!(doc.quantity("Window size").unwrap().value, 3207.0);
assert!((doc.quantity("Window duration").unwrap().value - 72.72).abs() < 0.005);
assert!((doc.quantity("Time resolution").unwrap().value - 11.61).abs() < 0.005);
assert!((doc.quantity("Minimum peak separation").unwrap().value - 58.05).abs() < 0.005);
assert_eq!(doc.quantity("CQT bins").unwrap().value, 144.0);
assert!(doc.has_warning_containing("counts onset-function frames"));
assert!(doc.has_warning_containing("not read by detect_onsets"));
```

The explanation converts every frame count into milliseconds, flags the
frames-versus-samples confusion of `min_peak_separation`, and lists the
configuration fields that `detect_onsets` ignores.

## Exercises

1. **Units.** You want onsets at least 50 ms apart with $H = 256$ at
   44.1 kHz. What `min_peak_separation` do you set, and what spacing in
   milliseconds does it actually give?
2. **The tempo grid.** At $H = 512$ and 44.1 kHz, which BPM values can
   `estimate_tempo` return between 115 and 125 BPM?
3. **Choosing the hop.** How would halving the hop to 256 samples change the
   grid spacing near 120 BPM? Predict, then compute.

## Answers

**1.** $0.05 \cdot 44100 / 256 = 8.6$, so 9 frames, which is 52.2 ms.

```rust
use audio_samples::educational::{Explain, ExplainContext};
use audio_samples::operations::onset::OnsetDetectionConfig;
use std::num::NonZeroUsize;
let frames = (0.05f64 * 44_100.0 / 256.0).ceil() as usize;
assert_eq!(frames, 9);
let mut c = OnsetDetectionConfig::percussive(); // hop 256
assert_eq!(c.hop_size.get(), 256);
c.peak_picking.min_peak_separation = NonZeroUsize::new(frames).unwrap();
let ms = c.explain_with(&ExplainContext::new().with_sample_rate(44_100.0))
    .quantity("Minimum peak separation").unwrap().value;
assert!((ms - 52.24).abs() < 0.005);
```

**2.** Lags 42, 43 and 44 frames: 123.05, 120.19 and 117.45 BPM. Lag 41
(126.03 BPM) and lag 45 (114.84 BPM) fall just outside.

```rust
let bpm = |lag: f64| 60.0 * 44_100.0 / (512.0 * lag);
let in_range: Vec<f64> = (30..60)
    .map(|l| bpm(l as f64))
    .filter(|b| (115.0..=125.0).contains(b))
    .map(|b| (b * 100.0).round() / 100.0)
    .collect();
assert_eq!(in_range, vec![123.05, 120.19, 117.45]);
```

**3.** The lag doubles (86 frames at 120 BPM) and the spacing between grid
points roughly halves: from 2.73 BPM (lags 43, 44 at $H = 512$) to 1.38 BPM
(lags 86, 87 at $H = 256$). The estimate gets finer, at twice the cost.

```rust
let bpm = |lag: f64, hop: f64| 60.0 * 44_100.0 / (hop * lag);
let coarse = bpm(43.0, 512.0) - bpm(44.0, 512.0);
let fine = bpm(86.0, 256.0) - bpm(87.0, 256.0);
assert!((coarse - 2.73).abs() < 0.005);
assert!((fine - 1.38).abs() < 0.005);
```

## Common pitfalls

- **Frames, not samples.** Every peak-picking length counts ODF frames
  ($H$ samples each). Convert with frames = seconds $\times f_s / H$.
- **The default `OnsetDetectionConfig` finds no onsets on a clean click
  track** (length-3 median, 1024-frame threshold window, 512-frame separation).
  Start from the configuration in this chapter and check it with
  `explain_with`.
- **Reported times are frame centres** and lead the true onset by up to half
  the CQT window (about 30 ms at the default 55 Hz lower edge), and are
  quantised to $H/f_s$.
- **"Energy" flux is magnitude flux** here: magnitudes are not squared.
- **Octave errors in tempo** are expected, not exceptional: check the
  estimate against $2\times$ and $\tfrac{1}{2}\times$ (and $3\times$) before
  trusting it. `BeatTrackingConfig::tempo_bpm` is ignored by
  `estimate_tempo`.
- **librosa differs**: `librosa.onset.onset_strength` uses a mel spectrogram
  in dB with a one-frame lag and `peak_pick` uses separate pre/post windows
  in frames; numbers will not match this crate.

## See also

- [`AudioOnsetDetection::detect_onsets`](crate::AudioOnsetDetection::detect_onsets),
  [`AudioOnsetDetection::onset_detection_function`](crate::AudioOnsetDetection::onset_detection_function),
  [`OnsetDetectionConfig`](crate::operations::onset::OnsetDetectionConfig),
  [`PeakPickingConfig`](crate::operations::types::PeakPickingConfig),
  [`pick_peaks`](crate::operations::peak_picking::pick_peaks).
- [`AudioBeatTracking::estimate_tempo`](crate::operations::AudioBeatTracking::estimate_tempo),
  [`AudioBeatTracking::detect_beats`](crate::operations::AudioBeatTracking::detect_beats),
  [`onset_strength_envelope`](crate::operations::beat::onset_strength_envelope).
- Previous chapter: [`c12_dynamics`](crate::guide::c12_dynamics); next:
  [`c14_psychoacoustics_codecs`](crate::guide::c14_psychoacoustics_codecs).
- Lab: `examples/tutorial_13_onsets_tempo.rs`
  (`cargo run --release --example tutorial_13_onsets_tempo --features educational,full_no_plotting`).
