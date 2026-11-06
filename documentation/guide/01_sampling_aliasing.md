# 01. Sampling and aliasing

A digital audio signal is a sequence of samples taken at a fixed rate: 8 kHz
for narrow-band telephone speech, 16 kHz for wide-band speech and most speech
recognisers, 44.1 or 48 kHz for music. The sample rate determines which
frequencies the samples can represent. Frequencies above that limit are not
removed by sampling: they appear in the sampled signal as lower frequencies.
This chapter shows where they appear, why, and how to prevent it.

## Concept

**Sampling** reads an analogue signal every $T$ seconds:

$$x[n] = x_a(nT), \qquad T = 1 / f_s$$

where $x_a(t)$ is the analogue signal (amplitude in units of full scale) at
time $t$ in seconds, $n$ is the integer sample index, $T$ is the sampling
period in seconds and $f_s$ is the sample rate in Hz (samples per second).

**Why aliasing happens.** For any integer $m$,

$$\cos(2\pi f n T) = \cos(2\pi (f + m f_s) n T)$$

because the extra phase $2\pi m f_s n T = 2\pi m n$ is a whole number of
turns. Here $f$ is a tone frequency in Hz. Since $\cos$ is even, $-f$ gives the
same samples too, so the tones $f$, $f + m f_s$ and $m f_s - f$ are
indistinguishable once sampled (up to a reversal of phase for the
$m f_s - f$ family: a sine comes back with its sign flipped).

**Nyquist frequency and the sampling theorem.** The highest frequency that is
represented unambiguously is the Nyquist frequency

$$f_N = f_s / 2$$

in Hz. If $x_a$ contains no energy at or above $f_N$, it is recovered exactly
from its samples by sinc interpolation (Whittaker-Shannon):

$$x_a(t) = \sum_n x[n] \, \mathrm{sinc}\!\left(\frac{t - nT}{T}\right), \qquad \mathrm{sinc}(u) = \frac{\sin(\pi u)}{\pi u}$$

where $u$ is a dimensionless time offset in sampling periods.

**Folding formula.** A tone at $f$ Hz appears after sampling at

$$f_a = |f - k f_s|, \qquad k = \mathrm{round}(f / f_s)$$

where $f_a$ is the alias frequency in Hz (always between 0 and $f_N$) and $k$
is the nearest integer multiple of the sample rate. Equivalently, the frequency
axis is folded at $0$, $f_N$, $f_s$, $3f_N$, ..., which maps every frequency
into $[0, f_N]$.

<!-- figure: ch01-folding -->

Aliasing occurs in three places:

* **acquisition**: an analogue anti-aliasing filter must remove energy above
  $f_N$ before the converter samples;
* **downsampling**: keeping every $M$-th sample is sampling again at $f_s/M$,
  so content between the new and old Nyquist frequencies must be filtered out
  first;
* **synthesis**: a waveform with jumps (a hard square wave) has harmonics at
  every odd multiple $h f_0$ of its fundamental $f_0$, with amplitudes
  $4A/(\pi h)$ for peak amplitude $A$. Computing it sample by sample samples
  those harmonics too, and the ones above $f_N$ fold back as inharmonic tones.

## With the library

Generate an 11 kHz tone at 48 kHz, then reduce it to 16 kHz (new Nyquist
frequency 8 kHz) in two ways: naively, by keeping every third sample (written
by hand below), and with the library's resampler at each of its three quality
tiers. The folding formula predicts $|11000 - 1 \cdot 16000| = 5000$ Hz.

```rust
use audio_samples::educational::topics;
use audio_samples::operations::types::ResamplingQuality;
use audio_samples::{AudioSamples, AudioTransforms, nzu, resample, sample_rate, sine_wave};
use std::time::Duration;

// The library's folding calculation.
let doc = topics::aliasing(11_000.0, 16_000.0);
assert_eq!(doc.quantity("Alias frequency").unwrap().value, 5_000.0);

// 250 ms of an 11 kHz tone, amplitude 0.5, at 48 kHz: 12 000 samples.
let hi = sine_wave::<f64>(11_000.0, Duration::from_millis(250), sample_rate!(48000), 0.5);

// Naive decimation: keep every third sample. 4000 samples at 16 kHz.
let naive: Vec<f64> = hi.as_slice().unwrap().iter().step_by(3).copied().collect();
let naive = AudioSamples::new_mono(ndarray::Array1::from(naive), sample_rate!(16000)).unwrap();

// A 4000-point DFT of 4000 samples at 16 kHz has 4 Hz bins.
let spec = naive.rfft(nzu!(4000)).unwrap();
let row = spec.row(0);
let peak = (0..row.len()).max_by(|&a, &b| row[a].total_cmp(&row[b])).unwrap();
assert_eq!(peak as f64 * 4.0, 5_000.0); // the alias, exactly where predicted
// A bin-centred sine of amplitude A gives |X| = A N / 2 = 1000: full strength.
assert!((row[peak] - 1000.0).abs() < 1e-6);

// Every resampler tier low-pass filters below 8 kHz before it decimates.
// Measure what is left at 5 kHz in the steady state: the middle 100 ms of
// the output (samples 1200..2800) holds exactly 500 periods of 5 kHz, so one
// DFT bin gives its amplitude, compared with the 0.5 the naive decimator
// delivers there.
let db = |v: f64| 20.0 * v.log10();
let amplitude = |x: &[f64], f: f64| {
    let (re, im) = x.iter().enumerate().fold((0.0, 0.0), |(re, im), (n, &v)| {
        let ph = 2.0 * std::f64::consts::PI * f * n as f64 / 16_000.0;
        (re + v * ph.cos(), im - v * ph.sin())
    });
    2.0 * (re * re + im * im).sqrt() / x.len() as f64
};
let mut steady_db = Vec::new();
for q in [ResamplingQuality::Fast, ResamplingQuality::Medium, ResamplingQuality::High] {
    let y = resample(&hi, sample_rate!(16000), q).unwrap();
    assert_eq!(y.samples_per_channel().get(), 4_000);
    steady_db.push(db(amplitude(&y.as_slice().unwrap()[1_200..2_800], 5_000.0) / 0.5));
}
println!("steady-state alias, Fast/Medium/High: {steady_db:.1?} dB");
// Gone: for this 1 : 3 ratio the resampler works block by block in the
// frequency domain and drops everything above the new Nyquist frequency
// instead of folding the filter's stopband back, so what is left is f64
// round-off, more than 180 dB down.
assert!(steady_db.iter().all(|&d| d < -180.0));

// Over the whole 250 ms the residue is larger, and not because of aliasing:
// the tone starts and stops abruptly, and those edges are broadband clicks
// that the filter passes below 8 kHz. For High, the 5 kHz bin of the full
// output is 78.4 dB below the naive alias, and the largest bin anywhere
// (7652 Hz, in the filter's transition band between 0.913 x 8 kHz = 7304 Hz
// and 8 kHz) only 74.5 dB below it.
let high = resample(&hi, sample_rate!(16000), ResamplingQuality::High).unwrap();
let high_spec = high.rfft(nzu!(4000)).unwrap();
let hrow = high_spec.row(0);
let bin_5k_db = db(row[peak] / hrow[1250]);
let top = (0..hrow.len()).max_by(|&a, &b| hrow[a].total_cmp(&hrow[b])).unwrap();
let largest_db = db(row[peak] / hrow[top]);
println!("5 kHz bin {bin_5k_db:.1} dB, largest bin {largest_db:.1} dB at {} Hz", top * 4);
assert!((bin_5k_db - 78.4).abs() < 0.1);
assert_eq!(top * 4, 7_652);
assert!(top * 4 > 7_304 && top * 4 < 8_000);
assert!((largest_db - 74.5).abs() < 0.1);
// The click is visible in the samples: the first 50 output samples reach 0.12,
// while the middle of the output is below 1e-7.
let hs = high.as_slice().unwrap();
let max_abs = |x: &[f64]| x.iter().fold(0.0f64, |m, v| m.max(v.abs()));
assert!((max_abs(&hs[..50]) - 0.12).abs() < 0.005);
assert!(max_abs(&hs[1_200..2_800]) < 1e-7);
```

The naive version did not remove the tone; it *moved* it to 5 kHz at full
strength, where it is indistinguishable from a genuine 5 kHz component. Every
tier of the resampler removes it instead: in the steady state the residue is
more than 180 dB down, floating-point round-off, for every tier (for this
1 : 3 ratio the resampler drops everything above the new Nyquist frequency in
the frequency domain, so its filter's stopband does not fold back). Measure
over the whole signal and `High`'s figure drops to 78.4 dB at 5 kHz (74.5 dB
for the largest bin, at 7652 Hz): the residue there is the broadband click of a tone switched
on and off abruptly, not aliasing. Always measure rejection away from the ends
of a test signal, as chapter 07 does.

The same applies when a waveform is synthesised. A naive square wave
([`square_wave`](crate::utils::generation::square_wave)) switches between
$\pm A$; the band-limited version
([`square_wave_bandlimited`](crate::utils::generation::square_wave_bandlimited))
adds only the odd harmonics below $f_N$. With $f_0 = 700$ Hz and $f_s = 16$ kHz,
one second holds exactly 700 periods, so every harmonic and every alias falls
on an integer-Hz bin of a 16 000-point DFT, and the power that is *not* at an
odd harmonic of 700 Hz can be summed directly:

```rust
use audio_samples::utils::generation::{square_wave, square_wave_bandlimited};
use audio_samples::{AudioTransforms, nzu, sample_rate};
use std::time::Duration;

let one_s = Duration::from_secs(1);
let naive = square_wave::<f64>(700.0, one_s, sample_rate!(16000), 1.0);
let clean = square_wave_bandlimited::<f64>(700.0, one_s, sample_rate!(16000), 1.0);
let s_naive = naive.rfft(nzu!(16000)).unwrap().row(0).to_vec(); // 1 Hz bins
let s_clean = clean.rfft(nzu!(16000)).unwrap().row(0).to_vec();

let odd_harmonic = |k: usize| k.is_multiple_of(700) && (k / 700) % 2 == 1;
let stray_db = |s: &[f64]| {
    let total: f64 = s.iter().map(|m| m * m).sum();
    let stray: f64 = s.iter().enumerate()
        .filter(|(k, _)| !odd_harmonic(*k))
        .map(|(_, m)| m * m).sum();
    10.0 * (stray / total).log10()
};
// About 3.3 % of the naive wave's power is aliased (-14.77 dB) ...
assert!((stray_db(&s_naive) + 14.77).abs() < 0.01);
// ... and essentially none of the band-limited wave's.
assert!(stray_db(&s_clean) < -200.0);

// The 13th harmonic (9100 Hz) is the first above 8 kHz: it folds to 6900 Hz,
// close to its ideal level of 1/13 of the fundamental (-22.3 dB).
let rel_db = 20.0 * (s_naive[6900] / s_naive[700]).log10();
assert!((rel_db - 20.0 * (1.0f64 / 13.0).log10()).abs() < 0.3);
assert!(s_clean[6900] / s_clean[700] < 1e-9);
```

<!-- figure: ch01-square-spectra -->

## How it works

The folding formula is one line of code. Check it against the library over a sweep
of frequencies, and check the identity behind it: the samples of an 11 kHz sine
at 16 kHz are exactly those of a 5 kHz sine with inverted sign, because
$\sin(2\pi \cdot 11000\, n/16000) = \sin(2\pi(11000 - 16000)\, n/16000) = -\sin(2\pi \cdot 5000\, n / 16000)$.

<!-- figure: ch01-samples -->

```rust
use audio_samples::educational::topics;
use std::f64::consts::PI;

fn fold(f: f64, fs: f64) -> f64 {
    (f - (f / fs).round() * fs).abs()
}

// Every 250 Hz from 0 to 50 kHz at 16 kHz agrees with the library exactly
// (both are a handful of exact operations on multiples of 250).
for i in 0..=200 {
    let f = 250.0 * f64::from(i);
    let lib = topics::aliasing(f, 16_000.0).quantity("Alias frequency").unwrap().value;
    assert_eq!(fold(f, 16_000.0), lib);
    assert!(lib <= 8_000.0);
}

// Identical samples, sign inverted. Tolerance: sin of arguments up to ~2e4 rad
// in f64 carries ~1e-12 of rounding.
let fs = 16_000.0;
for n in 0..1000 {
    let t = f64::from(n) / fs;
    let a = (2.0 * PI * 11_000.0 * t).sin();
    let b = -(2.0 * PI * 5_000.0 * t).sin();
    assert!((a - b).abs() < 1e-9);
}
```

The sampling theorem says a band-limited signal can be recovered *between* its
samples. Reconstruct a 1 kHz tone sampled at 8 kHz halfway between two samples
with a (truncated) sinc sum:

```rust
use std::f64::consts::PI;

let fs = 8_000.0;
let f = 1_000.0;
let x: Vec<f64> = (0..8000).map(|n| (2.0 * PI * f * n as f64 / fs).sin()).collect();
let sinc = |u: f64| if u == 0.0 { 1.0 } else { (PI * u).sin() / (PI * u) };

// Evaluate at t = 4000.5 T, in the middle of the 1 s block.
let u0 = 4000.5;
let estimate: f64 = x.iter().enumerate().map(|(n, &v)| v * sinc(u0 - n as f64)).sum();
let truth = (2.0 * PI * f * u0 / fs).sin();
// The sum is truncated to 8000 terms; the neglected tails decay like 1/(pi u),
// so an error of order 1e-4 is expected.
assert!((estimate - truth).abs() < 1e-3);
```

## Ask the library to explain

[`topics::sampling`](crate::educational::topics::sampling) and
[`topics::aliasing`](crate::educational::topics::aliasing) return an
[`ExplainDoc`](crate::educational::ExplainDoc) with the formulas, the numbers
for your case, a table of tones that share the same samples, and a warning when
a tone is above the Nyquist frequency.

```rust
use audio_samples::educational::{WarningLevel, topics};

let s = topics::sampling(16_000.0);
assert_eq!(s.quantity("Nyquist frequency").unwrap().value, 8_000.0);
assert_eq!(s.quantity("Sampling period").unwrap().value, 0.0625); // ms
// 8 kHz is below the 20 kHz limit of hearing: an informational note.
assert!(s.has_warning(WarningLevel::Info));

let a = topics::aliasing(25_000.0, 16_000.0);
assert_eq!(a.quantity("Alias frequency").unwrap().value, 7_000.0);
assert!(a.has_warning(WarningLevel::Caution));
assert!(a.has_warning_containing("will alias to 7000 Hz"));
println!("{a}"); // formulas, symbols, the table of look-alike tones, the warning
```

## Exercises

1. A telephone line samples at 8 kHz. A 7.5 kHz whistle leaks in before the
   converter. At what frequency does it appear after sampling?
2. You halve the rate of a 44.1 kHz recording by keeping every second sample.
   It contains a 15 kHz tone. Predict its alias frequency, then measure it.
3. In the naive 700 Hz square wave above, there is energy at 1300 Hz, which is
   not a harmonic of 700 Hz. Which harmonic folded there?

## Answers

```rust
use audio_samples::utils::generation::{square_wave, square_wave_bandlimited};
use audio_samples::{AudioSamples, AudioTransforms, nzu, sample_rate, sine_wave};
use std::time::Duration;

fn fold(f: f64, fs: f64) -> f64 {
    (f - (f / fs).round() * fs).abs()
}

// 1. |7500 - 1 * 8000| = 500 Hz.
assert_eq!(fold(7_500.0, 8_000.0), 500.0);

// 2. New rate 22 050 Hz: |15000 - 22050| = 7050 Hz. Measure: 0.2 s at 44.1 kHz
//    is 8820 samples; keeping every second leaves 4410 samples at 22 050 Hz,
//    so a 4410-point DFT has 5 Hz bins and 7050 Hz is bin 1410.
assert_eq!(fold(15_000.0, 22_050.0), 7_050.0);
let x = sine_wave::<f64>(15_000.0, Duration::from_millis(200), sample_rate!(44100), 1.0);
let dec: Vec<f64> = x.as_slice().unwrap().iter().step_by(2).copied().collect();
assert_eq!(dec.len(), 4410);
let dec = AudioSamples::new_mono(ndarray::Array1::from(dec), sample_rate!(22050)).unwrap();
let mag = dec.rfft(nzu!(4410)).unwrap().row(0).to_vec();
let peak = (0..mag.len()).max_by(|&a, &b| mag[a].total_cmp(&mag[b])).unwrap();
assert_eq!(peak as f64 * 5.0, 7_050.0);

// 3. Search the odd harmonics h whose alias is 1300 Hz: the 21st (14 700 Hz,
//    folded about 16 kHz). The band-limited wave stops at the 11th harmonic
//    (7700 Hz), so it has nothing at 1300 Hz.
let h = (1..200).step_by(2).find(|&h| fold(700.0 * f64::from(h), 16_000.0) == 1_300.0);
assert_eq!(h, Some(21));
let one_s = Duration::from_secs(1);
let naive = square_wave::<f64>(700.0, one_s, sample_rate!(16000), 1.0)
    .rfft(nzu!(16000)).unwrap().row(0).to_vec();
let clean = square_wave_bandlimited::<f64>(700.0, one_s, sample_rate!(16000), 1.0)
    .rfft(nzu!(16000)).unwrap().row(0).to_vec();
assert!(naive[1300] / naive[700] > 1e-3);  // clearly present
assert!(clean[1300] / clean[700] < 1e-9);  // absent
```

## Common pitfalls

* Keeping every $M$-th sample is not resampling: it aliases. Every
  [`ResamplingQuality`](crate::operations::types::ResamplingQuality) filters
  first, but the default, `Fast`, uses a short filter whose pass band ends at
  0.8 times the new Nyquist frequency; use `Medium` or `High` when the top of
  the band matters (chapter 07).
* [`square_wave`](crate::utils::generation::square_wave),
  [`sawtooth_wave`](crate::utils::generation::sawtooth_wave) and
  [`triangle_wave`](crate::utils::generation::triangle_wave) are naive and
  alias; use the `*_bandlimited` versions for test signals.
* A tone exactly at $f_N$ is ambiguous: a sine sampled there at zero phase is
  all zeros.
* An alias may come back with inverted phase (the 11 kHz example above
  returns as $-\sin$ at 5 kHz).
* [`rfft`](crate::AudioTransforms::rfft) returns magnitudes and needs
  `n_fft` at least the signal length (chapter 03).

## References

* C. E. Shannon, "Communication in the presence of noise", *Proceedings of the IRE* 37(1), 10-21, 1949.
* A. V. Oppenheim and R. W. Schafer, *Discrete-Time Signal Processing*, 3rd ed., Pearson, 2010. Chapter 4 covers sampling, aliasing and reconstruction.

## See also

* [`topics::sampling`](crate::educational::topics::sampling),
  [`topics::aliasing`](crate::educational::topics::aliasing),
  [`topics::resampling`](crate::educational::topics::resampling)
* [`resample`](crate::resample),
  [`ResamplingQuality`](crate::operations::types::ResamplingQuality),
  [`AudioTransforms::rfft`](crate::AudioTransforms::rfft)
* [`sine_wave`](crate::utils::generation::sine_wave),
  [`square_wave`](crate::utils::generation::square_wave),
  [`square_wave_bandlimited`](crate::utils::generation::square_wave_bandlimited)
* Next: [02. Quantisation and dither](02_quantisation_dither.md).
  Resampling in depth: [07. Resampling](07_resampling.md).
* Lab: `examples/tutorial_01_sampling_aliasing.rs`
  (`cargo run --release --example tutorial_01_sampling_aliasing --features educational,full_no_plotting`).
