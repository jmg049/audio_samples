# 12. Dynamics: compressors and limiters

Speech varies by 30 dB or more between a whispered consonant and a shouted
vowel, and broadcast, telephony and hearing aids all reduce that range so the
quiet parts stay audible and the loud parts do not clip. A compressor has
three parts: a level detector with attack and release
*ballistics*, a *static gain computer* in decibels, and a multiplier. This
chapter measures each part of the library's implementation and reimplements
the whole processor in about a dozen lines.

## Concept

**Detector and ballistics.** The detector turns the input $x[n]$ into a
non-negative value $d[n]$: $|x[n]|$ for a peak detector, or the RMS over the
last 10 ms. A one-pole smoother then produces the envelope

$$e[n] = a\, e[n-1] + (1 - a)\, d[n], \qquad a = e^{-1/(\tau f_s)}$$

where $f_s$ is the sample rate in Hz and $\tau$ the time constant in seconds:
the attack constant $\tau_A$ while $d[n] > e[n-1]$ (the level is rising) and
the release constant $\tau_R$ otherwise. After a step from $d_0$ to $d_1$ the
envelope follows

$$e(t) = d_1 + (d_0 - d_1)\, e^{-t/\tau}$$

so it covers $1 - e^{-1} = 63.2\%$ of the step after one time constant, and its
10-90 % rise time is $\tau \ln 9 \approx 2.2\,\tau$. (Equipment makers quote
attack times in either convention, so numbers are not comparable across
products.)

**Static gain computer (dB domain).** With envelope level
$L = 20\log_{10} e$ (dBFS), threshold $T$ (dBFS), ratio $R \ge 1$
(dimensionless) and knee width $W$ (dB), the hard-knee compressor's output
level is

$$Y = L \ \text{ for } L \le T, \qquad Y = T + (L - T)/R \ \text{ for } L > T$$

so the gain reduction is $r = L - Y = (1 - 1/R)\max(L - T, 0)$ dB. The
quadratic **soft knee** (Giannoulis, Massberg and Reiss, *JAES* 2012) blends
the two lines over $[T - W/2, T + W/2]$:

$$r = (1 - 1/R)\,\frac{(L - T + W/2)^2}{2W} \quad \text{for } |L - T| \le W/2$$

It has slope 1 at the lower edge and $1/R$ at the upper edge, so the curve has
no corner, and at $L = T$ it already reduces by $(1 - 1/R)W/8$ dB.

**Gain.** The output is $y[n] = x[n]\, 10^{-r[n]/20}\, 10^{M/20}$, where $M$ is
the make-up gain in dB. Working in dB makes the ratio a slope and the
threshold an offset.

**Limiter.** A limiter is the compressor with $R \to \infty$ and the ceiling
$C$ (dBFS) in place of $T$: $r = \max(L - C, 0)$. Because the gain follows a
*smoothed* envelope, the output can still exceed $C$ at the start of a
transient.

**Lookahead.** Delay the audio by $N_L$ samples and apply to sample $n$ the
largest reduction computed for samples $n, \dots, n + N_L$. The reduction is
then in place before the transient arrives. If the envelope starts from zero
and the step is to $d_1$, the reduction applied to the first sample of the step
is the one computed $N_L$ samples later, from an envelope of
$d_1(1 - a^{N_L + 1})$, so the overshoot is

$$\Delta = -20 \log_{10}\left(1 - a^{N_L + 1}\right) \ \text{dB}$$

which is about $0.06$ dB for $N_L = 5\,\tau_A f_s$. (The library processes
whole buffers, so it compensates the delay: output index $n$ stays aligned
with input index $n$.)

## With the library

The static curve is available directly from
[`calculate_compression_gain`](crate::operations::dynamic_range::calculate_compression_gain)
(a reduction in dB) and from
[`AudioDynamicRange::get_compression_curve`](crate::AudioDynamicRange::get_compression_curve)
(output levels, make-up gain included).

```rust
use audio_samples::operations::dynamic_range::calculate_compression_gain;
use audio_samples::operations::types::KneeType;
use audio_samples::{AudioDynamicRange, AudioSamples, CompressorConfig, sample_rate};
use non_empty_slice::{NonEmptySlice, NonEmptyVec};

// Hard knee, T = -20 dBFS, R = 4: -8 dBFS is 12 dB over, so the output is -20 + 12/4 = -17.
let r = calculate_compression_gain(-8.0, -20.0, 4.0, KneeType::Hard, 0.0);
assert!((r - 9.0).abs() < 1e-12);

// Soft knee W = 10 dB: no reduction at T - W/2, (1 - 1/R) W / 8 = 0.9375 dB at T,
// and back on the hard-knee line at T + W/2 (3.75 dB).
let soft = |l| calculate_compression_gain(l, -20.0, 4.0, KneeType::Soft, 10.0);
assert_eq!(soft(-25.0), 0.0);
assert!((soft(-20.0) - 0.9375).abs() < 1e-12);
assert!((soft(-15.0) - 3.75).abs() < 1e-12);
// The output curve's slope is 1 at the lower knee edge and 1/R at the upper one.
let y = |l: f64| l - soft(l);
let d = 1e-6;
assert!(((y(-25.0 + d) - y(-25.0 - d)) / (2.0 * d) - 1.0).abs() < 1e-5);
assert!(((y(-15.0 + d) - y(-15.0 - d)) / (2.0 * d) - 0.25).abs() < 1e-5);

// The same curve as output levels through the trait (here with a hard knee).
let mut cfg = CompressorConfig::new();
cfg.threshold_db = -20.0;
cfg.ratio = 4.0;
cfg.knee_type = KneeType::Hard;
cfg.knee_width_db = 0.0;
cfg.makeup_gain_db = 0.0;
let audio: AudioSamples<f64> = AudioSamples::from_mono_vec(NonEmptyVec::new(vec![0.5f64; 16]).unwrap(), sample_rate!(48000));
let curve = audio.get_compression_curve(&cfg, NonEmptySlice::new(&[-30.0, -20.0, -8.0]).unwrap()).unwrap();
assert_eq!(curve, vec![-30.0, -20.0, -17.0]);
```

<!-- figure: ch12-static-curve -->

The ballistics are visible in the response to a step. Apply a DC level of $-40$ dBFS for 100 ms,
then 0 dBFS for 200 ms (peak detector, attack 5 ms, release 50 ms, hard knee
at $-20$ dBFS, 4:1), and read the per-sample reduction with
[`get_gain_reduction`](crate::AudioDynamicRange::get_gain_reduction). Above the
threshold the hard-knee curve can be inverted,
$L = T + r / (1 - 1/R)$, which recovers the envelope.

```rust
use audio_samples::operations::types::{DynamicRangeMethod, KneeType};
use audio_samples::{AudioDynamicRange, AudioSamples, CompressorConfig, sample_rate};
use non_empty_slice::NonEmptyVec;

let fs: f64 = 48_000.0;
let x: Vec<f64> = std::iter::repeat_n(0.01, 4_800).chain(std::iter::repeat_n(1.0, 9_600)).collect();
let audio: AudioSamples<f64> = AudioSamples::from_mono_vec(NonEmptyVec::new(x).unwrap(), sample_rate!(48000));
let mut cfg = CompressorConfig::new();
// Set four fields at once (a destructuring assignment): threshold -20 dBFS,
// ratio 4:1, hard knee, knee width 0 dB.
(cfg.threshold_db, cfg.ratio, cfg.knee_type, cfg.knee_width_db) = (-20.0, 4.0, KneeType::Hard, 0.0);
(cfg.attack_ms, cfg.release_ms, cfg.detection_method) = (5.0, 50.0, DynamicRangeMethod::Peak);

let gr = audio.get_gain_reduction(&cfg).unwrap();
let env = |n: usize| 10f64.powf((-20.0 + gr[n] / 0.75) / 20.0); // valid while gr > 0
let reach = |level: f64| (4_800..).find(|&n| gr[n] > 0.0 && env(n) >= level).unwrap();
let (lo, hi) = (0.01, 1.0);
// Sample 4800 is the first at the new level and already one update of the recursion.
let t63_ms = (reach(lo + (1.0 - (-1f64).exp()) * (hi - lo)) - 4_800 + 1) as f64 / fs * 1e3;
let rise_ms = (reach(lo + 0.9 * (hi - lo)) - reach(lo + 0.1 * (hi - lo))) as f64 / fs * 1e3;
// One sample is 0.021 ms, so both are within 0.03 ms of theory.
assert!((t63_ms - 5.0).abs() < 0.03); // tau_A
assert!((rise_ms - 5.0 * 9f64.ln()).abs() < 0.03); // 10.99 ms = 2.2 tau_A
// Settled: the envelope is 0 dBFS, 20 dB over the threshold, so 15 dB of reduction.
assert!((gr[14_000] - 15.0).abs() < 1e-6);
```

<!-- figure: ch12-attack -->

## How it works

The whole compressor is an envelope follower, the gain computer and a
multiplication. Written out, it matches
[`apply_compressor`](crate::AudioDynamicRange::apply_compressor) sample for
sample, here with a 6 dB soft knee on a step up and back down (which also
tests the release).

```rust
use audio_samples::operations::types::{DynamicRangeMethod, KneeType};
use audio_samples::{AudioDynamicRange, AudioSamples, CompressorConfig, sample_rate};
use non_empty_slice::NonEmptyVec;

let fs: f64 = 48_000.0;
let x: Vec<f64> = [(0.01, 4_800), (1.0, 9_600), (0.01, 14_400)]
    .iter()
    .flat_map(|&(v, n)| std::iter::repeat_n(v, n))
    .collect();
let audio: AudioSamples<f64> = AudioSamples::from_mono_vec(NonEmptyVec::new(x.clone()).unwrap(), sample_rate!(48000));
let (t, ratio, w) = (-20.0, 4.0, 6.0);
let mut cfg = CompressorConfig::new();
(cfg.threshold_db, cfg.ratio, cfg.knee_type, cfg.knee_width_db) = (t, ratio, KneeType::Soft, w);
(cfg.attack_ms, cfg.release_ms, cfg.detection_method) = (5.0, 50.0, DynamicRangeMethod::Peak);
(cfg.makeup_gain_db, cfg.lookahead_ms) = (0.0, 0.0);
let lib = audio.apply_compressor(&cfg).unwrap();

let (a_att, a_rel) = ((-1.0 / (0.005 * fs)).exp(), (-1.0 / (0.050 * fs)).exp());
let reduction = |l: f64| {
    let o = l - t;
    let slope = 1.0 - 1.0 / ratio;
    if 2.0 * o.abs() <= w { slope * (o + w / 2.0).powi(2) / (2.0 * w) } else { slope * o.max(0.0) }
};
let mut e = 0.0f64;
let mine: Vec<f64> = x
    .iter()
    .map(|&v| {
        let d = v.abs(); // peak detector
        let a = if d > e { a_att } else { a_rel };
        e = a * e + (1.0 - a) * d;
        let level = if e > 0.0 { 20.0 * e.log10() } else { -80.0 };
        v * 10f64.powf(-reduction(level) / 20.0)
    })
    .collect();
let err = mine.iter().zip(lib.as_slice().unwrap()).map(|(p, q)| (p - q).abs()).fold(0.0, f64::max);
// Same operations in the same order, so only last-bit differences remain.
assert!(err < 1e-12);
```

<!-- figure: ch12-compressor-step -->

The lookahead formula can be checked the same way: a limiter with a $-6$ dBFS
ceiling and a 1 ms attack, on a step from silence to 0 dBFS. Without
lookahead the first sample of the step passes without reduction (the envelope is still
at $1 - a = 0.02$, i.e. $-33.7$ dBFS), a 6 dB overshoot. With 5 ms of
lookahead ($N_L = 240$ samples) the overshoot is
$-20\log_{10}(1 - a^{241}) = 0.0575$ dB.

```rust
use audio_samples::operations::types::{DynamicRangeMethod, KneeType};
use audio_samples::{AudioDynamicRange, AudioSamples, LimiterConfig, sample_rate};
use non_empty_slice::NonEmptyVec;

let fs: f64 = 48_000.0;
let x: Vec<f64> = std::iter::repeat_n(0.0, 960).chain(std::iter::repeat_n(1.0, 2_400)).collect();
let audio: AudioSamples<f64> = AudioSamples::from_mono_vec(NonEmptyVec::new(x).unwrap(), sample_rate!(48000));
let peak_db = |lookahead_ms: f64| {
    // Arguments: ceiling -6 dBFS, attack 1 ms, release 100 ms, hard knee of
    // width 0 dB, peak detection, the lookahead in ms, and `false` for the
    // inter-sample-peak option (left off here).
    let lim = LimiterConfig::new(-6.0, 1.0, 100.0, KneeType::Hard, 0.0, DynamicRangeMethod::Peak, lookahead_ms, false);
    let y = audio.apply_limiter(&lim).unwrap();
    20.0 * y.as_slice().unwrap().iter().fold(0.0f64, |m, v| m.max(v.abs())).log10()
};
assert!(peak_db(0.0).abs() < 1e-9); // the step passes at 0 dBFS: 6 dB over the ceiling
let a: f64 = (-1.0 / (0.001 * fs)).exp();
let predicted = -20.0 * (1.0 - a.powi(241)).log10();
assert!((predicted - 0.0575).abs() < 1e-3);
assert!((peak_db(5.0) - (-6.0 + predicted)).abs() < 1e-9);
```

<!-- figure: ch12-lookahead -->

## Ask the library to explain

[`CompressorConfig`](crate::CompressorConfig) and
[`LimiterConfig`](crate::LimiterConfig) implement
[`Explain`](crate::educational::Explain). With a sample rate (from a signal)
the explanation includes the smoothing coefficients, the 10-90 % times, a
table of the static curve, and a note when there is no lookahead.

```rust
use audio_samples::educational::ExplainFor;
use audio_samples::operations::types::{DynamicRangeMethod, KneeType};
use audio_samples::{CompressorConfig, LimiterConfig, sample_rate, silence};
use std::time::Duration;

let audio = silence::<f32>(Duration::from_millis(10), sample_rate!(48000));
let mut cfg = CompressorConfig::new();
(cfg.threshold_db, cfg.ratio, cfg.knee_type, cfg.knee_width_db) = (-20.0, 4.0, KneeType::Hard, 0.0);
(cfg.attack_ms, cfg.release_ms, cfg.detection_method) = (5.0, 50.0, DynamicRangeMethod::Peak);
let doc = audio.explain(&cfg);
println!("{doc}");
assert!((doc.quantity("Attack 10-90% rise time").unwrap().value - 10.986).abs() < 1e-3);
assert!((doc.quantity("Attack coefficient").unwrap().value - (-1.0f64 / 240.0).exp()).abs() < 1e-12);
assert!((doc.quantity("Gain reduction at 0 dBFS").unwrap().value - 15.0).abs() < 1e-9);
assert!(doc.has_warning_containing("No lookahead"));

let lim = audio.explain(&LimiterConfig::transparent());
println!("{lim}");
assert!(lim.quantity("Gain reduction for a 0 dBFS envelope").is_some());
```

## Exercises

1. **Read the curve.** With $T = -18$ dBFS and $R = 3$ (hard knee), predict
   the output level for a $-2$ dBFS input, then check with
   `get_compression_curve`.
2. **Knee at the threshold.** For a 6 dB soft knee and $R = 3$, predict the
   reduction exactly at $L = T$.
3. **Attack in samples.** At 16 kHz with $\tau_A = 10$ ms, predict $a$ and how
   many samples an
   [`EnvelopeFollower`](crate::operations::dynamic_range::EnvelopeFollower)
   needs to reach 63.2 % of a step from 0 to 1.
4. **Enough lookahead?** For the limiter above, how much lookahead (in
   samples) keeps the overshoot below 0.01 dB?

## Answers

1. $16$ dB over the threshold becomes $16/3 = 5.33$ dB over it:
   $-18 + 5.33 = -12.67$ dBFS.

```rust
use audio_samples::operations::types::KneeType;
use audio_samples::{AudioDynamicRange, AudioSamples, CompressorConfig, sample_rate};
use non_empty_slice::{NonEmptySlice, NonEmptyVec};

let mut cfg = CompressorConfig::new();
(cfg.threshold_db, cfg.ratio, cfg.knee_type, cfg.makeup_gain_db) = (-18.0, 3.0, KneeType::Hard, 0.0);
let audio: AudioSamples<f64> = AudioSamples::from_mono_vec(NonEmptyVec::new(vec![0.0f64; 4]).unwrap(), sample_rate!(16000));
let y = audio.get_compression_curve(&cfg, NonEmptySlice::new(&[-2.0]).unwrap()).unwrap();
assert!((y[0] - (-18.0 + 16.0 / 3.0)).abs() < 1e-12);
assert!((y[0] + 12.67).abs() < 0.005);
```

2. $(1 - 1/3) \cdot 6 / 8 = 0.5$ dB.

```rust
use audio_samples::operations::dynamic_range::calculate_compression_gain;
use audio_samples::operations::types::KneeType;

let r = calculate_compression_gain(-18.0, -18.0, 3.0, KneeType::Soft, 6.0);
assert!((r - 0.5).abs() < 1e-12);
```

3. $a = e^{-1/160} = 0.99377$; after $k$ updates the envelope is $1 - a^k$,
   which first reaches $1 - e^{-1}$ at $k = 160$ samples (10 ms).

```rust
use audio_samples::operations::dynamic_range::EnvelopeFollower;
use audio_samples::operations::types::DynamicRangeMethod;

let a = (-1.0f64 / 160.0).exp();
assert!((a - 0.99377).abs() < 1e-5);
let mut f = EnvelopeFollower::new(10.0, 100.0, 16_000.0, DynamicRangeMethod::Peak);
let k = (1..).find(|_| f.process(1.0) >= 1.0 - (-1f64).exp()).unwrap();
assert_eq!(k, 160);
```

4. Solve $-20\log_{10}(1 - a^{N_L+1}) < 0.01$: $a^{N_L+1} < 1 - 10^{-0.0005}$,
   so $N_L + 1 > \ln(1 - 10^{-0.0005}) / \ln a = 324.8$, i.e. $N_L \ge 324$ samples
   (6.8 ms, about $6.8\,\tau_A$).

```rust
let a: f64 = (-1.0 / 48.0f64).exp(); // 1 ms at 48 kHz
let overshoot = |n: i32| -20.0 * (1.0 - a.powi(n + 1)).log10();
let n_min = (0..2_000).find(|&n| overshoot(n) < 0.01).unwrap();
assert_eq!(n_min, 324);
assert!(overshoot(323) >= 0.01);
```

## Common pitfalls

- **Attack and release are time constants**, not "time to full reduction":
  63.2 % after $\tau$, 10-90 % in $2.2\,\tau$. Hardware datasheets use either
  convention.
- **Detector choice shifts the threshold.** An RMS detector reads a sine
  3.01 dB below its peak, so the same threshold compresses less than with
  `Peak` (the default `CompressorConfig` uses RMS).
- **Threshold reference.** Thresholds are in dBFS of the *envelope*, not of individual
  samples or of loudness (LUFS).
- **A limiter without lookahead is not a brick wall.** The start of every
  transient passes; even with lookahead a small overshoot remains, and only
  sample peaks are detected (`isp_limiting` is not implemented).
- **Stereo is not linked.** Multi-channel audio is compressed per channel,
  which can shift the stereo image.
- `get_gain_reduction` reports the detector-and-curve reduction without
  lookahead; `apply_compressor` additionally applies the lookahead max-hold.

## References

- D. Giannoulis, M. Massberg and J. D. Reiss, "Digital dynamic range compressor design: a tutorial and analysis", *Journal of the Audio Engineering Society* 60(6), 399-408, 2012. The soft knee used here; the paper also compares detector topologies (where and how the envelope is smoothed).
- U. Zölzer (ed.), *DAFX: Digital Audio Effects*, 2nd ed., Wiley, 2011.

## See also

- [`AudioDynamicRange::apply_compressor`](crate::AudioDynamicRange::apply_compressor),
  [`AudioDynamicRange::apply_limiter`](crate::AudioDynamicRange::apply_limiter),
  [`AudioDynamicRange::get_compression_curve`](crate::AudioDynamicRange::get_compression_curve),
  [`AudioDynamicRange::get_gain_reduction`](crate::AudioDynamicRange::get_gain_reduction)
- [`CompressorConfig`](crate::CompressorConfig), [`LimiterConfig`](crate::LimiterConfig),
  [`calculate_compression_gain`](crate::operations::dynamic_range::calculate_compression_gain),
  [`calculate_limiting_gain`](crate::operations::dynamic_range::calculate_limiting_gain),
  [`EnvelopeFollower`](crate::operations::dynamic_range::EnvelopeFollower)
- The full derivation is in the [`dynamic_range`](crate::operations::dynamic_range) module docs.
- [`Explain`](crate::educational::Explain), [`ExplainFor`](crate::educational::ExplainFor)
- Previous chapter: [11. Short-time energy, zero crossings and VAD](11_energy_vad.md);
  next chapter: [13. Onsets and tempo](13_onsets_tempo.md)
- Lab: `examples/tutorial_12_dynamics.rs`
  (`cargo run --release --example tutorial_12_dynamics --features educational,full_no_plotting`)
