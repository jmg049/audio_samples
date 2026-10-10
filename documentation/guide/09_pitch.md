# 09. Pitch: autocorrelation and YIN

The previous chapter recovered the vocal-tract filter; this one recovers the
source. The fundamental frequency $f_0$ of voiced speech carries intonation,
stress, speaker identity and, in tone languages, meaning. Pitch estimators look
for the period of the waveform rather than for a spectral peak, which is why
they still work when the fundamental itself is missing (as it is on a
telephone line, which cuts everything below about 300 Hz).

## Concept

A signal is periodic with period $T_0$ seconds if $x(t) = x(t + T_0)$; its
fundamental frequency is $f_0 = 1/T_0$ Hz. In samples the period is the lag
$\tau_0 = f_s / f_0$, where $f_s$ is the sample rate in Hz; in general
$\tau_0$ is not an integer.

**Autocorrelation.** The (unnormalised) autocorrelation of an $N$-sample frame
$x$ at lag $\tau$ (samples) is

$$r(\tau) = \sum_{i=0}^{N-1-\tau} x\[i\] \\, x\[i+\tau\]$$

It peaks at multiples of the period. The estimator searches
$\tau_{min} = \lfloor f_s / f_{max} \rfloor \le \tau \le \tau_{max} = \lfloor f_s / f_{min} \rfloor$
($f_{min}$, $f_{max}$ the search range in Hz) and reports $\hat{f}_0 = f_s / \hat{\tau}$.
Because $\hat{\tau}$ is an integer, the answers lie on the grid $f_s/\tau$, whose
spacing near $f_0$ is

$$\Delta f = \frac{f_s}{\tau} - \frac{f_s}{\tau + 1} = \frac{f_s}{\tau(\tau+1)} \approx \frac{f_0^2}{f_s}$$

in Hz. The true $f_0$ can fall anywhere between two grid points, so the
worst-case error is half a step, $\Delta f / 2$. At 16 kHz the step is tiny
for a 110 Hz voice ($\Delta f \approx 0.76$ Hz, error at most 0.38 Hz) but not
for higher pitches: $\Delta f \approx 12.0$ Hz at 440 Hz (error at most 6.0 Hz)
and $\Delta f \approx 46.8$ Hz at 880 Hz (error at most 23.4 Hz).

**YIN** (de Cheveigné and Kawahara, 2002) replaces the product by a squared
difference, normalises it, and interpolates:

$$d(\tau) = \sum_{j=0}^{N-1-\tau} \left( x\[j\] - x\[j+\tau\] \right)^2$$

$$d'(0) = 1, \qquad d'(\tau) = \frac{d(\tau)}{\frac{1}{\tau} \sum_{i=1}^{\tau} d(i)}$$

where $d(\tau)$ (squared signal units) is the difference function over all
$N - \tau$ overlapping pairs of samples and $d'(\tau)$ (dimensionless) the *cumulative-mean-normalised
difference* (CMND). $d'$ starts at 1, stays near 1 for lags shorter than the
period and dips towards 0 at the period. The estimate is the first lag with
$d'(\tau) < \theta$ ($\theta$ the threshold, typically 0.1 to 0.2), moved forward
to the bottom of that dip, and refined by fitting a parabola through
$d'(\tau_0 - 1), d'(\tau_0), d'(\tau_0 + 1)$ (this is what the crate and
librosa do; see the note below). With $a = d'(\tau_0+1) + d'(\tau_0-1) - 2d'(\tau_0)$
and $b = (d'(\tau_0+1) - d'(\tau_0-1))/2$ the vertex lies at $\tau_0 - b/a$:

$$\hat{\tau} = \tau_0 + \frac{1}{2} \\, \frac{d'(\tau_0-1) - d'(\tau_0+1)}{d'(\tau_0-1) - 2 d'(\tau_0) + d'(\tau_0+1)}, \qquad \hat{f}_0 = \frac{f_s}{\hat{\tau}}$$

so the period is fractional and the integer grid disappears. Following
librosa, the shift is skipped (left at 0) when $\tau_0$ is at either end of the
search range or when $|b| \ge |a|$ (the vertex would lie more than one lag
away). If no lag dips below $\theta$ the crate declares the frame unvoiced.

*How this differs from the paper.* De Cheveigné and Kawahara's step 5
(parabolic interpolation) fits the parabola to the **raw** difference function
$d(\tau)$, not to $d'$, because interpolating $d'$ is slightly biased; their
step 6 ("best local estimate") then re-runs the search around each frame's
estimate. And when no lag dips below $\theta$, the paper takes the global
minimum of $d'$ instead of giving up. The crate, like librosa's `yin`,
interpolates $d'$ and has no step 6; unlike librosa it reports such frames as
unvoiced. Finally, the paper and librosa sum $d(\tau)$ over a fixed window of
$W$ samples starting at the beginning of the frame; the crate sums over all
$N - \tau$ overlapping pairs. A fixed window measures the pitch of the frame's
early part, which matters when the pitch moves within the frame (about 9 cents
on a 1.5 octave/s glide with 2048-sample frames at 22.05 kHz).

**Octave errors.** A signal with period $\tau_0$ also repeats at $2\tau_0$,
$3\tau_0$, …; and if its odd harmonics are weak it *nearly* repeats at
$\tau_0/2$. Picking $2\tau_0$ halves the estimate (a sub-octave error);
picking $\tau_0/2$ doubles it. YIN's "first dip below $\theta$" rule guards
against the first kind; the second is controlled by $\theta$.

**Frame length.** The difference function needs lags up to $\tau_{max}$ plus a
window, so both detectors in this crate require $N > 2\tau_{max} + 1$,
i.e. at least $N_{min} = 2(\tau_{max} + 1)$ samples, a little more than two periods
of $f_{min}$.

## With the library

[`detect_pitch_autocorr`](crate::AudioPitchAnalysis::detect_pitch_autocorr) and
[`detect_pitch_yin`](crate::AudioPitchAnalysis::detect_pitch_yin) analyse the
whole signal as one frame and return `Ok(None)` when no pitch is found.

```rust
use audio_samples::{AudioPitchAnalysis, sample_rate, sine_wave};
use std::time::Duration;

let fs = 16_000.0;
let x = sine_wave::<f64>(440.0, Duration::from_millis(50), sample_rate!(16000), 0.8);

let ac = x.detect_pitch_autocorr(80.0, 1000.0).unwrap().unwrap();
let yin = x.detect_pitch_yin(0.1, 80.0, 1000.0).unwrap().unwrap();
println!("autocorrelation {ac:.2} Hz, YIN {yin:.3} Hz");

// The true period is 16000/440 = 36.36 samples. Autocorrelation must pick an
// integer lag (36), so it reports 444.44 Hz: off by 4.4 Hz, within half of
// the 12-Hz grid step f0^2/f_s.
assert_eq!(ac, fs / 36.0);
assert!((ac - 440.0).abs() <= 0.5 * (fs / 36.0 - fs / 37.0));

// The grid steps and worst-case errors quoted in the Concept section: the
// step between the two integer lags that bracket fs / f0, and half of it.
let step = |f0: f64| {
    let tau = (fs / f0).floor();
    fs / tau - fs / (tau + 1.0)
};
for (f0, df, worst) in [(110.0, 0.76, 0.38), (440.0, 12.0, 6.0), (880.0, 46.8, 23.4)] {
    assert!((step(f0) - df).abs() < 0.05, "{f0}: {}", step(f0));
    assert!((step(f0) / 2.0 - worst).abs() < 0.05);
    assert!((step(f0) - f0 * f0 / fs).abs() / step(f0) < 0.1); // close to f0^2/fs
}
// YIN interpolates the period and lands within 0.1 Hz (440.0986 Hz).
assert!((yin - 440.0986).abs() < 5e-5);
assert!((yin - 440.0).abs() < 0.1);
```

The lab repeats this for 110, 220, 440 and 880 Hz: the autocorrelation errors
are +0.34, -0.82, +4.44 and +8.89 Hz, always within half a grid step; YIN's
are +0.0006, +0.0124, +0.0986 and +1.2354 Hz.

## Under the hood

Both estimators are short enough to write out. We implement them exactly as
the formulas above and compare with the library on the same 440 Hz frame.

```rust
use audio_samples::{AudioPitchAnalysis, sample_rate, sine_wave};
use std::time::Duration;

let fs = 16_000.0;
let audio = sine_wave::<f64>(440.0, Duration::from_millis(50), sample_rate!(16000), 0.8);
let x = audio.as_slice().unwrap();
let n = x.len();
let (min_tau, max_tau) = ((fs / 1000.0) as usize, (fs / 80.0) as usize); // 16, 200

// Autocorrelation: argmax of r(tau) over the lag range.
let r = |tau: usize| (0..n - tau).map(|i| x[i] * x[i + tau]).sum::<f64>();
let best = (min_tau..=max_tau).max_by(|&a, &b| r(a).total_cmp(&r(b))).unwrap();
assert_eq!(best, 36);
assert_eq!(fs / best as f64, audio.detect_pitch_autocorr(80.0, 1000.0).unwrap().unwrap());

// YIN step 1: difference function over all n - tau overlapping pairs.
let d: Vec<f64> = (0..=max_tau)
    .map(|tau| (0..n - tau).map(|j| (x[j] - x[j + tau]).powi(2)).sum())
    .collect();
// Step 2: cumulative-mean-normalised difference d'.
let mut cmnd = vec![1.0; max_tau + 1];
let mut running = 0.0;
for tau in 1..=max_tau {
    running += d[tau];
    cmnd[tau] = d[tau] * tau as f64 / running;
}
// Step 3: first lag below theta, then down to the bottom of the dip.
let theta = 0.1;
let mut tau = (min_tau..=max_tau).find(|&t| cmnd[t] < theta).unwrap();
assert_eq!(tau, 34); // d'(34) = 0.0756 is already below 0.1 ...
assert!((cmnd[34] - 0.0756).abs() < 1e-4 && (cmnd[36] - 0.0019).abs() < 1e-4);
while tau < max_tau && cmnd[tau + 1] < cmnd[tau] {
    tau += 1;
}
assert_eq!(tau, 36); // ... but the dip bottoms out at 36 (d' = 0.0019)
// Step 4: parabolic interpolation, here on d' (as librosa does; the paper's
// step 5 interpolates the raw d instead), skipped at the ends of the range or
// when |b| >= |a|.
let a = cmnd[tau + 1] + cmnd[tau - 1] - 2.0 * cmnd[tau];
let b = (cmnd[tau + 1] - cmnd[tau - 1]) / 2.0;
let shift = if tau > min_tau && tau < max_tau && b.abs() < a.abs() { -b / a } else { 0.0 };
assert!((shift - 0.3555).abs() < 1e-4); // the true period is 36.3636 samples
let tau_hat = tau as f64 + shift;
let f_hat = fs / tau_hat;
println!("d'(34..=38) = {:.4?}; period {tau_hat:.4} samples, f0 {f_hat:.4} Hz", &cmnd[34..=38]);

let lib = audio.detect_pitch_yin(theta, 80.0, 1000.0).unwrap().unwrap();
// Same operations in the same order, so agreement is to rounding error.
assert!((f_hat - lib).abs() < 1e-9, "{f_hat} vs {lib}");
```

**Octave errors and the missing fundamental.** First, harmonics 2 to 6 of
200 Hz with equal amplitudes and nothing at 200 Hz itself. Then a signal whose
odd harmonics (1, 3, 5) are 20 dB weaker than its even ones: its true period is
still 80 samples (5 ms), but it is almost periodic at 40 samples.

```rust
use audio_samples::{AudioPitchAnalysis, AudioSamples, AudioTransforms, sample_rate};
use ndarray::Array1;
use std::f64::consts::PI;

let fs = 16_000.0;
let n = 1600; // 0.1 s: 20 periods of 200 Hz, so FFT bins land on harmonics
let complex = |harmonics: &[(f64, f64)]| {
    let x: Vec<f64> = (0..n)
        .map(|i| harmonics.iter().map(|&(h, a)| a * (2.0 * PI * h * 200.0 * i as f64 / fs).cos()).sum())
        .collect();
    AudioSamples::new_mono(Array1::from(x), sample_rate!(16000)).unwrap()
};

// 1. Missing fundamental: no energy at 200 Hz, yet both detectors find it.
let missing = complex(&[(2.0, 1.0), (3.0, 1.0), (4.0, 1.0), (5.0, 1.0), (6.0, 1.0)]);
let mag = missing.rfft(std::num::NonZeroUsize::new(n).unwrap()).unwrap();
assert!(mag[[0, 20]] < 1e-9 * mag[[0, 40]]); // bin 20 = 200 Hz is empty
assert!((missing.detect_pitch_yin(0.1, 80.0, 500.0).unwrap().unwrap() - 200.0).abs() < 0.1);
assert_eq!(missing.detect_pitch_autocorr(80.0, 500.0).unwrap().unwrap(), 200.0);

// 2. Weak odd harmonics: an octave error, controlled by the YIN threshold.
let weak = complex(&[(1.0, 0.1), (2.0, 1.0), (3.0, 0.1), (4.0, 1.0), (5.0, 0.1), (6.0, 1.0)]);
let at = |theta: f64| weak.detect_pitch_yin(theta, 80.0, 500.0).unwrap().unwrap();
assert!((at(0.1) - 400.0).abs() < 1.0); // octave error: half the period
assert!((at(0.01) - 200.0).abs() < 0.1); // strict threshold: the true period
// Autocorrelation is biased towards short lags (fewer terms in the sum at
// long lags) and also doubles the pitch.
assert_eq!(weak.detect_pitch_autocorr(80.0, 500.0).unwrap().unwrap(), 400.0);
```

At the half period (lag 40) the CMND has a deep dip of its own, and any
threshold above the bottom of that dip accepts the half period (exercise 3 asks
how deep it is). A strict threshold fixes this signal but rejects more real,
slightly aperiodic speech frames: that is the trade-off the threshold controls.

**Tracking a glide.** [`track_pitch`](crate::AudioPitchAnalysis::track_pitch)
cuts the signal into frames of `window_size` samples every `hop_size` samples
(no window function, no padding) and runs the detector on each. Each
[`PitchFrame`](crate::PitchFrame)'s `time` is the frame **start**; for a
linear glide the fairest reference is the frequency at the frame centre.

```rust
use audio_samples::operations::types::PitchDetectionMethod;
use audio_samples::{AudioPitchAnalysis, chirp, nzu, sample_rate};
use std::time::Duration;

let fs = 16_000.0;
// 0.5 s linear glide from 100 to 200 Hz: f(t) = 100 + 200 t.
let glide = chirp::<f64>(100.0, 200.0, Duration::from_millis(500), sample_rate!(16000), 0.8);

// 512 samples (32 ms) is more than the 402 needed to search down to 80 Hz.
let track = glide
    .track_pitch(nzu!(512), nzu!(128), PitchDetectionMethod::Yin, 0.1, 80.0, 400.0)
    .unwrap();
let mut worst: f64 = 0.0;
for frame in track.frames().iter().filter(|f| f.time + 512.0 / fs <= 0.5) {
    let truth = 100.0 + 200.0 * (frame.time + 256.0 / fs);
    let f = frame.frequency.expect("every full frame is voiced");
    worst = worst.max((f - truth).abs());
}
println!("worst error {worst:.2} Hz over {} frames", track.len());
assert!(worst < 1.0);
assert!((worst - 0.12).abs() < 0.005);

// 256 samples (16 ms) cannot hold two periods of 80 Hz: every frame is unvoiced.
let short = glide
    .track_pitch(nzu!(256), nzu!(64), PitchDetectionMethod::Yin, 0.1, 80.0, 400.0)
    .unwrap();
assert!(short.frames().iter().all(|f| f.frequency.is_none()));
assert_eq!(short.mean_pitch(), None);
```

The lab tracks a 1 s glide from 100 to 300 Hz: with 64 ms and with 32 ms
frames every full frame is voiced and the worst error is well below 1 Hz
(0.24 and 0.12 Hz); with 16 ms frames none is.

## Ask the library to explain

```rust
use audio_samples::educational::{WarningLevel, topics};

// YIN searching 80-400 Hz at 16 kHz.
let doc = topics::yin(0.1, 80.0, 400.0, 16_000.0, Some(256));
println!("{doc}");
assert_eq!(doc.quantity("Longest lag").unwrap().value, 200.0);
assert_eq!(doc.quantity("Minimum frame length").unwrap().value, 402.0);
assert!(doc.has_warning(WarningLevel::Problem)); // 256 < 402
assert!(!topics::yin(0.1, 80.0, 400.0, 16_000.0, Some(512)).has_warning(WarningLevel::Problem));

// Autocorrelation searching 80-1000 Hz: the grid step at the top of the range.
let doc = topics::autocorrelation_pitch(80.0, 1000.0, 16_000.0);
assert_eq!(doc.quantity("Shortest lag").unwrap().value, 16.0);
let step = doc.quantity("Pitch step at f_max").unwrap().value;
assert!((step - (1000.0 - 16_000.0 / 17.0)).abs() < 1e-9); // 58.8 Hz
assert!(doc.has_warning_containing("octave errors"));
```

The YIN explanation prints the four formulas of this chapter and the minimum
frame length; the autocorrelation one quotes the integer-lag pitch steps and
warns that a search range spanning more than an octave allows octave errors.

## Exercises

1. **Grid step.** Near 1 kHz at $f_s = 16$ kHz, how far apart are neighbouring
   autocorrelation estimates? Compare with $f_0^2/f_s$.
2. **Low voices.** A bass singer goes down to 60 Hz. What is the minimum YIN
   frame length at 16 kHz? Predict what `track_pitch` returns on a 70 Hz tone
   with 512-sample frames and $f_{min} = 60$ Hz, then check.
3. **Choosing the threshold.** For the weak-odd-harmonics signal above,
   compute $d'(40)$ yourself and predict the YIN estimate with $\theta = 0.021$
   and $\theta = 0.019$.

## Answers

**1.** Lags 16 and 17 give 1000 Hz and 941.18 Hz: a step of 58.8 Hz, against
$f_0^2/f_s = 62.5$ Hz (the approximation $\tau(\tau+1) \approx \tau^2$ is coarse
for short lags).

```rust
use audio_samples::educational::topics;
let step: f64 = 16_000.0 / 16.0 - 16_000.0 / 17.0;
assert!((step - 58.82).abs() < 0.01);
assert_eq!(1000.0f64 * 1000.0 / 16_000.0, 62.5);
let doc = topics::autocorrelation_pitch(80.0, 1000.0, 16_000.0);
assert!((doc.quantity("Pitch step at f_max").unwrap().value - step).abs() < 1e-9);
```

**2.** $\tau_{max} = \lfloor 16000/60 \rfloor = 266$, so $N_{min} = 2 \cdot 267 = 534$
samples (33.4 ms). With 512-sample frames every frame is unvoiced, even though
512 samples hold more than two 70 Hz periods (2 × 228.6 = 457): the frame must
cover the whole *search* range, not just the actual pitch.

```rust
use audio_samples::educational::topics;
use audio_samples::operations::types::PitchDetectionMethod;
use audio_samples::{AudioPitchAnalysis, nzu, sample_rate, sine_wave};
use std::time::Duration;

let doc = topics::yin(0.1, 60.0, 400.0, 16_000.0, Some(512));
assert_eq!(doc.quantity("Minimum frame length").unwrap().value, 534.0);
assert!(doc.has_warning_containing("shorter than the 534 samples"));

let low = sine_wave::<f64>(70.0, Duration::from_millis(300), sample_rate!(16000), 0.8);
let track = low
    .track_pitch(nzu!(512), nzu!(128), PitchDetectionMethod::Yin, 0.1, 60.0, 400.0)
    .unwrap();
assert!(track.frames().iter().all(|f| f.frequency.is_none()));
let track = low
    .track_pitch(nzu!(1024), nzu!(256), PitchDetectionMethod::Yin, 0.1, 60.0, 400.0)
    .unwrap();
assert!((track.mean_pitch().unwrap() - 70.0).abs() < 0.5);
```

**3.** $d'(40) = 0.0196$, the deepest point of the half-period dip, so
$\theta = 0.021$ accepts lag 40 (400 Hz) and $\theta = 0.019$ rejects it and
continues to the true period (200 Hz).

```rust
use audio_samples::{AudioPitchAnalysis, AudioSamples, sample_rate};
use ndarray::Array1;
use std::f64::consts::PI;

let fs = 16_000.0;
let amps = [(1.0, 0.1), (2.0, 1.0), (3.0, 0.1), (4.0, 1.0), (5.0, 0.1), (6.0, 1.0)];
let x: Vec<f64> = (0..1600)
    .map(|i| amps.iter().map(|&(h, a): &(f64, f64)| a * (2.0 * PI * h * 200.0 * i as f64 / fs).cos()).sum())
    .collect();

// d'(tau) for the library's lag range (80-500 Hz: lags 32..=200), with d
// summed over all N - tau overlapping pairs as in the Concept section.
let max_tau = 200;
let d: Vec<f64> = (0..=max_tau)
    .map(|tau| (0..x.len() - tau).map(|j| (x[j] - x[j + tau]).powi(2)).sum())
    .collect();
let cmnd = |tau: usize| d[tau] * tau as f64 / d[1..=tau].iter().sum::<f64>();
assert!((cmnd(40) - 0.0196).abs() < 5e-5);
assert!((32..40).all(|t| cmnd(t) > cmnd(40))); // 40 is the bottom of the dip

let audio = AudioSamples::new_mono(Array1::from(x), sample_rate!(16000)).unwrap();
let f = |theta: f64| audio.detect_pitch_yin(theta, 80.0, 500.0).unwrap().unwrap();
assert!((f(0.021) - 400.0).abs() < 1.0);
assert!((f(0.019) - 200.0).abs() < 0.1);
```

## Common pitfalls

- **Frame length is set by $f_{min}$**, not by the pitch you expect: frames of
  at most $2\lfloor f_s/f_{min} \rfloor + 1$ samples always return `None`.
- **`PitchFrame::time` is the frame start**, not its centre (librosa's
  `pyin` reports centred frames).
- **Not librosa's `yin`, and not exactly the paper.** Like librosa, the crate
  interpolates $d'$ (the paper interpolates $d$) and has no "best local
  estimate" step. Unlike librosa, a frame with no dip below $\theta$ is
  unvoiced, while `librosa.yin` (and the paper) fall back to the global
  minimum and always return a value. The minimum frame length rule also
  differs.
- **The autocorrelation estimate is quantised** to $f_s/\tau$; use YIN (or
  interpolate the peak yourself) when you need better than $f_0^2/f_s$.
- **Octave errors are a property of the signal**, not a bug: narrow
  $\[f_{min}, f_{max}\]$ when you know the speaker, and lower $\theta$ when
  harmonics alternate in strength.
- `detect_pitch_autocorr` requires $f_{min} > 1$ Hz; both methods reject
  multi-channel audio.

## References

- A. de Cheveigné and H. Kawahara, "YIN, a fundamental frequency estimator for speech and music", *Journal of the Acoustical Society of America* 111(4), 1917-1930, 2002.
- L. R. Rabiner, "On the use of autocorrelation analysis for pitch detection", *IEEE Transactions on Acoustics, Speech, and Signal Processing* 25(1), 24-33, 1977.
- M. Mauch and S. Dixon, "pYIN: a fundamental frequency estimator using probabilistic threshold distributions", *Proc. IEEE ICASSP*, 2014. The probabilistic extension used by librosa's `pyin`.

## See also

- [`AudioPitchAnalysis::detect_pitch_yin`](crate::AudioPitchAnalysis::detect_pitch_yin),
  [`AudioPitchAnalysis::detect_pitch_autocorr`](crate::AudioPitchAnalysis::detect_pitch_autocorr),
  [`AudioPitchAnalysis::track_pitch`](crate::AudioPitchAnalysis::track_pitch),
  [`PitchContour`](crate::PitchContour), [`PitchFrame`](crate::PitchFrame).
- Low-level routines: [`yin_pitch_detection`](crate::operations::pitch_analysis::yin_pitch_detection),
  [`autocorr_pitch_detection`](crate::operations::pitch_analysis::autocorr_pitch_detection).
- Explanations: [`topics::yin`](crate::educational::topics::yin),
  [`topics::autocorrelation_pitch`](crate::educational::topics::autocorrelation_pitch).
- Previous chapter: [`c08_lpc_formants`](crate::guide::c08_lpc_formants); next:
  [`c10_cepstrum_mel_mfcc`](crate::guide::c10_cepstrum_mel_mfcc), which finds the
  pitch a third way, in the cepstrum.
- Lab: `examples/tutorial_09_pitch.rs`
  (`cargo run --release --example tutorial_09_pitch --features educational,full_no_plotting`).
