# 13. Onsets and tempo

Onset detection marks the instants at which new sound energy appears, such as
the start of a note or a drum hit; tempo estimation finds the period with
which those instants repeat. Both are built on an *onset detection function*
(ODF), one number per analysis frame, and both inherit its limitations, so this
chapter states units explicitly (samples, frames, seconds, beats per minute)
and notes where the library's defaults fall short.

## Concept

**Frames.** The signal is analysed in frames hopped by $H$ samples; frame $t$
is centred at $t H / f_s$ seconds ($f_s$ the sample rate in Hz). At
$H = 512$ and $f_s = 44.1$ kHz a frame lasts $H/f_s = 11.61$ ms and there are
$f_s/H = 86.13$ frames per second.

**Spectral flux.** With $|X_b[t]|$ the magnitude of frequency bin $b$ in frame
$t$, the half-wave-rectified flux is

$$o[t] = \sum_{b} \max\left(0, \, |X_b[t]| - |X_b[t-1]|\right), \qquad o[0] = 0$$

Only *increases* count, so note offsets do not create onsets. This crate
evaluates it on a constant-Q transform (CQT) whose $b_o O$ bins ($b_o$ per
octave, $O$ octaves) start at $f_{min}$ Hz, with an analysis window of
$N_w = \lfloor 4 f_s / f_{min} \rfloor$ samples (four periods of the lowest
bin). The magnitudes are **not** squared, despite the method's name ("energy").
Optionally, values below $\mu$ times a running median of $L_{med}$ frames are set to
zero (`adaptive_threshold`, $\mu$ = `adaptive_threshold_multiplier`).

**Peak picking**, as implemented by
[`pick_peaks`](crate::operations::peak_picking::pick_peaks), all in **frames**:
optional pre-emphasis and median filtering; normalisation by the maximum; a
threshold $\theta[i] = \mathrm{clamp}(\max_{|j - i| \le w/2} o[j] - \delta, \, \theta_{lo}, \theta_{hi})$
(window $w$ frames, $\delta = 0.05$, $\theta_{lo} = 0.01$, $\theta_{hi} = 1$ by default);
strict local maxima above it; finally, peaks are accepted strongest first and
dropped if closer than $s$ frames to an accepted one. A peak at frame $i$ is
reported at $t_i = i H / f_s$ seconds.

**Tempo.** A tempo of $\beta$ beats per minute (BPM) is an inter-beat
interval of $60/\beta$ seconds, i.e. a lag of

$$\ell = \frac{60 \, f_s}{\beta \, H} \ \text{frames}, \qquad \beta = \frac{60 \, f_s}{H \, \ell}$$

[`estimate_tempo`](crate::operations::AudioBeatTracking::estimate_tempo)
smooths the ODF (7-frame moving average), compresses it ($\ln(1 + 0.5 x)$),
removes the mean and computes $a(\ell) = \frac{1}{n - \ell} \sum_i c[i] \, c[i-\ell]$
for the integer lags that map into 40-240 BPM. Each lag is weighted by a
log-normal prior centred on 120 BPM, $w(\beta) = \exp\left(-\frac{1}{2} \log_2^2 (\beta/120)\right)$,
and the lag with the largest $a(\ell) w(\beta)$ is chosen. Two consequences: estimates lie on the grid
$60 f_s / (H \ell)$ (117.45, 120.19, 123.05 BPM around 120), and a periodic
envelope correlates at $2\ell$ and $3\ell$ too, so **metrical-level errors**
remain possible: half the tempo (an octave error) or a third of it. The prior
only biases the choice.

## With the library

A click track: four decaying noise bursts, one every 0.5 s (120 BPM). The
default configuration finds all four.

```rust
use audio_samples::operations::onset::OnsetDetectionConfig;
use audio_samples::{AudioOnsetDetection, AudioSamples, sample_rate};
use ndarray::Array1;

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

let config = OnsetDetectionConfig::default();
// Peak-picking lengths count frames: a 9-frame threshold window (104.5 ms at
// hop 512 and 44.1 kHz) and a 3-frame minimum separation (34.8 ms).
let pp = &config.peak_picking;
assert_eq!((pp.adaptive_threshold.window_size, pp.min_peak_separation.get()), (9, 3));
assert!((1000.0f64 * 9.0 * 512.0 / 44_100.0 - 104.5).abs() < 0.05);
assert!((1000.0f64 * 3.0 * 512.0 / 44_100.0 - 34.8).abs() < 0.05);
assert_eq!((pp.adaptive_threshold.delta, pp.adaptive_threshold.min_threshold), (0.07, 0.1));
assert!(pp.pre_emphasis && pp.pre_emphasis_coeff == 0.97 && !pp.median_filter);

let onsets = audio.detect_onsets(&config).unwrap();
println!("{onsets:.4?}");
assert_eq!(onsets.len(), 4);
// Peaks fall on frames 19, 62, 105, 148: 43 frames apart (0.4992 s, not 0.5 s).
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

<!-- figure: ch13-onsets -->

**The defaults, in frames.** The default
[`PeakPickingConfig`](crate::operations::types::PeakPickingConfig) pre-emphasises
the ODF ($o[i] - 0.97\, o[i-1]$), applies no median filter, normalises by the
maximum, and thresholds with a 9-frame window (104.5 ms), $\delta = 0.07$ and a
floor $\theta_{lo} = 0.1$; peaks closer than 3 frames (34.8 ms) are merged.

The floor removes weak onsets. After normalisation, no peak weaker than a
tenth of the strongest one in the clip is reported. On the lab's
eight-burst track the defaults find only 5 onsets: the first three bursts
peak at 0.030, 0.025 and 0.087 of the strongest (the CQT response varies from
click to click; see below). Lowering `adaptive_threshold.min_threshold` to
0.01 finds all 8.

Lengths chosen as if they counted audio samples are far too large in frames.
Starting from the configuration with the 0.01 floor (the lab, on 8 bursts) and
switching on one such setting at a time (a median filter of length 3, a
threshold window of 1024, a separation of 512) shows the effect of each: the
length-3 median deletes every one-frame peak (0 onsets); a 1024-frame threshold window, clipped to the signal length,
makes the threshold the global maximum minus $\delta$ (1 onset); a 512-frame
separation, reduced to a quarter of the signal (91 frames, 1.06 s), keeps only
3. Switching pre-emphasis off changes nothing on this signal (8). The fields
`threshold`, `min_onset_interval_secs` and `pre_emphasis` on
[`OnsetDetectionConfig`](crate::operations::onset::OnsetDetectionConfig)
itself are not read by `detect_onsets` (the explanation below says so).

## How it works

**The ODF.** Compute the CQT magnitude with the same parameters and sum the
rectified frame-to-frame increases; with the median-based zeroing switched off,
the result must equal
[`onset_detection_function`](crate::AudioOnsetDetection::onset_detection_function).
Then re-implement peak picking for the default configuration and convert
frames to seconds.

```rust
use audio_samples::operations::onset::OnsetDetectionConfig;
use audio_samples::{AudioOnsetDetection, AudioSamples, AudioTransforms, sample_rate};
use ndarray::Array1;
use spectrograms::{SpectrogramParams, StftParams};

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
let config = OnsetDetectionConfig::default();

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

// 2. Peak picking with the default settings (on the ODF detect_onsets uses):
//    pre-emphasis, peak normalisation, threshold, local maxima, separation.
let (_, odf) = audio.onset_detection_function(&config).unwrap();
let pre: Vec<f64> = (0..odf.len().get())
    .map(|i| if i == 0 { odf[0] } else { 0.97f64.mul_add(-odf[i - 1], odf[i]) })
    .collect();
let max = pre.iter().fold(0.0f64, |a, v| a.max(v.abs()));
let o: Vec<f64> = pre.iter().map(|v| v / max).collect();
let (half, delta, floor, sep) = (9 / 2, 0.07, 0.1, 3);
let mut candidates: Vec<(usize, f64)> = (1..o.len() - 1)
    .filter(|&i| {
        let local_max = o[i.saturating_sub(half)..(i + half + 1).min(o.len())]
            .iter()
            .fold(f64::NEG_INFINITY, |a, v| a.max(*v));
        o[i] > o[i - 1] && o[i] > o[i + 1] && o[i] > (local_max - delta).clamp(floor, 1.0)
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

<!-- figure: ch13-tempo -->

The halving comes from the ODF. The lab (a ten-click track, slightly different
from the twelve-click track above) prints the ODF at each of its 80 BPM clicks:
0.88, 0.0, 2.6, 0.08, 13.42, 0.71, 0.0, 3.27, 0.03, 8.69. Every other click is
close to zero (the CQT response variation again), so the envelope repeats
every *two* beats: its autocorrelation at lag 129 (0.00240) is twenty times
that at lag 65 (0.00012), a larger ratio than the prior weights (0.29 against
0.84) can reverse. At 174 BPM the lab finds 58.07 BPM, a third of the tempo (lag 89,
almost exactly $3 \times 29.7$). The estimate is correct at 60, 100, 120 and
140 BPM (60.09, 99.38, 120.19, 139.67).

## Ask the library to explain

```rust
use audio_samples::educational::{Explain, ExplainContext};
use audio_samples::operations::onset::OnsetDetectionConfig;

let config = OnsetDetectionConfig::default();
let doc = config.explain_with(&ExplainContext::new().with_sample_rate(44_100.0));
println!("{doc}");
assert_eq!(doc.quantity("Window size").unwrap().value, 3207.0);
assert!((doc.quantity("Window duration").unwrap().value - 72.72).abs() < 0.005);
assert!((doc.quantity("Time resolution").unwrap().value - 11.61).abs() < 0.005);
assert!((doc.quantity("Minimum peak separation").unwrap().value - 34.83).abs() < 0.005);
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
(126.05 BPM) and lag 45 (114.84 BPM) lie outside the range.

```rust
let bpm = |lag: f64| 60.0 * 44_100.0 / (512.0 * lag);
let in_range: Vec<f64> = (30..60)
    .map(|l| bpm(l as f64))
    .filter(|b| (115.0..=125.0).contains(b))
    .map(|b| (b * 100.0).round() / 100.0)
    .collect();
assert_eq!(in_range, vec![123.05, 120.19, 117.45]);
// The neighbours immediately outside the range, and all five values to 0.01 BPM.
let grid: Vec<f64> = (41..=45).map(|l| (bpm(l as f64) * 100.0).round() / 100.0).collect();
assert_eq!(grid, vec![126.05, 123.05, 120.19, 117.45, 114.84]);
```

**3.** The lag doubles (86 frames at 120 BPM) and the spacing between grid
points roughly halves: from 2.73 BPM (lags 43, 44 at $H = 512$) to 1.38 BPM
(lags 86, 87 at $H = 256$). The grid is finer, at twice the computational cost.

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
- **The default floor removes weak onsets.** The ODF is normalised by its
  maximum and the default threshold never drops below 0.1, so an onset whose
  (pre-emphasised) peak is under a tenth of the clip's strongest is not
  reported. Lower `adaptive_threshold.min_threshold` for material with a wide
  dynamic range, and check the result with `explain_with`.
- **Reported times are frame centres** and lead the true onset by up to half
  the CQT window (about 30 ms at the default 55 Hz lower edge), and are
  quantised to $H/f_s$.
- **"Energy" flux is magnitude flux** here: magnitudes are not squared.
- **Octave errors in tempo** are expected, not exceptional: check the
  estimate against $2\times$ and $\tfrac{1}{2}\times$ (and $3\times$) before
  trusting it. `BeatTrackingConfig::tempo_bpm` is ignored by
  `estimate_tempo`.
- **A second onset pipeline**:
  [`AudioRhythm::onset_strength`](crate::AudioRhythm::onset_strength) uses a
  mel spectrogram in dB with a one-frame lag, and
  [`AudioRhythm::onset_detect`](crate::AudioRhythm::onset_detect) picks peaks
  with separate pre/post windows in frames; their numbers will not match the
  functions in this chapter.

## References

- J. P. Bello, L. Daudet, S. Abdallah, C. Duxbury, M. Davies and M. B. Sandler, "A tutorial on onset detection in music signals", *IEEE Transactions on Speech and Audio Processing* 13(5), 1035-1047, 2005.
- S. Dixon, "Onset detection revisited", *Proc. 9th International Conference on Digital Audio Effects (DAFx-06)*, 2006.
- S. Dixon, "Automatic extraction of tempo and beat from expressive performances", *Journal of New Music Research* 30(1), 39-58, 2001.
- D. P. W. Ellis, "Beat tracking by dynamic programming", *Journal of New Music Research* 36(1), 51-60, 2007. The log-Gaussian tempo prior.
- J. C. Brown, "Calculation of a constant Q spectral transform", *Journal of the Acoustical Society of America* 89(1), 425-434, 1991.

## See also

- [`AudioOnsetDetection::detect_onsets`](crate::AudioOnsetDetection::detect_onsets),
  [`AudioOnsetDetection::onset_detection_function`](crate::AudioOnsetDetection::onset_detection_function),
  [`OnsetDetectionConfig`](crate::operations::onset::OnsetDetectionConfig),
  [`PeakPickingConfig`](crate::operations::types::PeakPickingConfig),
  [`pick_peaks`](crate::operations::peak_picking::pick_peaks).
- [`AudioBeatTracking::estimate_tempo`](crate::operations::AudioBeatTracking::estimate_tempo),
  [`AudioBeatTracking::detect_beats`](crate::operations::AudioBeatTracking::detect_beats),
  [`onset_strength_envelope`](crate::operations::beat::onset_strength_envelope).
- Previous chapter: [`c12_dynamics`](12_dynamics.md); next:
  [`c14_psychoacoustics_codecs`](14_psychoacoustics_codecs.md).
- Lab: `examples/tutorial_13_onsets_tempo.rs`
  (`cargo run --release --example tutorial_13_onsets_tempo --features educational,full_no_plotting`).
