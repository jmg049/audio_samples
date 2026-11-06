# 07. Resampling

Speech models are trained at 16 kHz, CDs are 44.1 kHz, video is 48 kHz:
most audio pipelines convert sample rates at some stage. Without adequate
filtering, downsampling folds high frequencies back into the band as aliases
and upsampling leaves spectral images. This chapter measures both effects on the
library's three quality tiers, compares them with naive decimation, and
rebuilds a windowed-sinc decimator by hand.

## Concept

**Ratio.** Converting from $f_{in}$ to $f_{out}$ (both in Hz) has ratio

$$r = f_{out} / f_{in}$$

(dimensionless). The output can only represent frequencies below its Nyquist
frequency $f_{out}/2$.

**Downsampling and aliasing.** Keeping every $q$-th sample ($r = 1/q$) is
sampling the signal again at $f_{out}$. A component at $f$ Hz above
$f_{out}/2$ reappears at

$$f_a = |f - k f_{out}|$$

where $k$ is the integer that brings $f_a$ into $[0, f_{out}/2]$ (chapter 01).
Once folded it is indistinguishable from real content, so a resampler must
**low-pass filter before decimating**, with a cutoff at or below
$\min(f_{in}, f_{out})/2$.

**Upsampling and imaging.** Raising the rate by $L$ ($r = L$) can be modelled
as inserting $L - 1$ zeros between samples (which leaves copies, *images*, of
the spectrum at $k f_{in} \pm f$) followed by a low-pass *interpolation*
filter that removes them.

**Ideal band-limited interpolation.** A band-limited signal is fully described
by its samples, and its value at any time $t$ (seconds) is

$$x(t) = \sum_n x[n]\, \mathrm{sinc}(f_{in} t - n), \quad \mathrm{sinc}(u) = \frac{\sin(\pi u)}{\pi u}$$

A practical resampler truncates and windows the sinc, and scales it so that its
cutoff is $f_c = \min(f_{in}, f_{out})/2$ (slightly less, to leave room for the
transition band). The same formula then performs both interpolation and
anti-aliasing.

**Linear interpolation** uses a triangular kernel instead of a sinc. For
integer upsampling by $L = 3$ it equals zero-stuffing followed by the FIR
$[1, 2, 3, 2, 1]/3$, whose gain at frequency $f$ (Hz) is

$$G(f) = \frac{3 + 4\cos\omega + 2\cos 2\omega}{9}, \quad \omega = 2\pi f / f_{out}$$

where the $1/9$ combines two factors: zero-stuffing scales the wanted
(baseband) component by $1/L = 1/3$, and the kernel's DC gain of $9/3 = 3$
restores it, so $G = H/3$ with $H$ the kernel's response and $G(0) = 1$. $G$ is
not flat in the pass band and not zero at the images.

**Alias (or image) rejection** is the level of the unwanted component
relative to the input component that produced it:

$$R = 20 \log_{10}(A_{alias} / A_{in}) \ \text{dB}$$

where $A$ is a sinusoid's amplitude. 0 dB means no rejection.

**The library's tiers** ([`ResamplingQuality`](crate::operations::types::ResamplingQuality)).
Every tier except `Fast` upsampling is a Kaiser-windowed sinc low-pass whose
cut-off is set relative to the lower Nyquist frequency
$f_N = \min(f_{in}, f_{out})/2$; the stop band starts at $f_N$:

- `Fast` (the default): when downsampling, a short filter with its pass band
  to $0.80 f_N$ and 70 dB of stop-band attenuation; when upsampling, linear
  interpolation (no anti-imaging filter).
- `Medium`: pass band to $0.90 f_N$, 100 dB.
- `High`: pass band to $0.913 f_N$, 125 dB (close to soxr's `soxr_hq`).

The filter is computed in the buffer's float type: `f32` for `f32`, `i16`,
`I24` and `u8` buffers, `f64` for `f64` and `i32` ones.

<!-- figure: ch07-tier-responses -->

**Time alignment.** Output sample $m$ is the band-limited reconstruction of
the input at the exact input position $m / r = m f_{in} / f_{out}$: the sinc
is centred on that position, so the filter is symmetric (linear phase) and
adds **no delay**. Input before the start and after the end counts as zeros,
and the output has exactly $\lceil N r \rceil$ samples for $N$ input samples.
The streaming [`StreamingResampler`](crate::StreamingResampler) holds back the
filter's look-ahead and emits the same samples as the whole-buffer
[`resample`](crate::resample), chunk by chunk.

## With the library

Mix a 1 kHz tone (to keep) with a 10 kHz tone (to remove), both at amplitude
0.5, and convert 48 kHz to 16 kHz by naive decimation (keep every third
sample, written by hand) and with each tier. The folding formula puts the
10 kHz alias at
$|10\,000 - 16\,000| = 6\,000$ Hz. Measure each component with one DFT bin
over 1600 output samples (whole periods of both 1 kHz and 6 kHz, so there is
no leakage), and measure the time shift of the output from the phase of the
1 kHz tone.

```rust
use audio_samples::operations::types::ResamplingQuality;
use audio_samples::{AudioSamples, resample, sample_rate, sine_wave};
use non_empty_slice::NonEmptyVec;
use std::f64::consts::PI;
use std::time::Duration;

/// DFT bin of the component at f Hz (one bin; exact for whole periods).
fn bin(x: &[f64], f: f64, fs: f64) -> (f64, f64) {
    x.iter().enumerate().fold((0.0, 0.0), |(re, im), (n, &v)| {
        let ph = 2.0 * PI * f * n as f64 / fs;
        (re + v * ph.cos(), im - v * ph.sin())
    })
}
fn amplitude(x: &[f64], f: f64, fs: f64) -> f64 {
    let (re, im) = bin(x, f, fs);
    2.0 * (re * re + im * im).sqrt() / x.len() as f64
}
fn phase(x: &[f64], f: f64, fs: f64) -> f64 {
    let (re, im) = bin(x, f, fs);
    im.atan2(re)
}

let dur = Duration::from_millis(250);
let a = sine_wave::<f64>(1_000.0, dur, sample_rate!(48000), 0.5);
let b = sine_wave::<f64>(10_000.0, dur, sample_rate!(48000), 0.5);
let x: Vec<f64> = a.as_slice().unwrap().iter().zip(b.as_slice().unwrap()).map(|(p, q)| p + q).collect();
let input: AudioSamples<f64> = AudioSamples::from_mono_vec(NonEmptyVec::new(x.clone()).unwrap(), sample_rate!(48000));

let mid = 1_200..2_800; // middle 100 ms of the 16 kHz output
let db = |v: f64| 20.0 * v.log10();
let ideal: Vec<f64> = (0..4_000).map(|n| 0.5 * (2.0 * PI * 1_000.0 * n as f64 / 16_000.0).sin()).collect();
let ideal_phase = phase(&ideal[mid.clone()], 1_000.0, 16_000.0);

// Naive decimation: the alias is at full level, 0 dB.
let naive: Vec<f64> = x.iter().step_by(3).copied().collect();
assert!(db(amplitude(&naive[mid.clone()], 6_000.0, 16_000.0) / 0.5).abs() < 1e-9);

let mut results = Vec::new();
for q in [ResamplingQuality::Fast, ResamplingQuality::Medium, ResamplingQuality::High] {
    let y = resample(&input, sample_rate!(16000), q).unwrap();
    assert_eq!(y.samples_per_channel().get(), 4_000); // ceil(12000 / 3)
    let y = y.as_slice().unwrap();
    let pass_db = db(amplitude(&y[mid.clone()], 1_000.0, 16_000.0) / 0.5);
    let alias_db = db(amplitude(&y[mid.clone()], 6_000.0, 16_000.0) / 0.5);
    // Shift in output frames: phase difference / (2 pi f / fs).
    let shift = (phase(&y[mid.clone()], 1_000.0, 16_000.0) - ideal_phase) / (2.0 * PI / 16.0);
    results.push((pass_db, alias_db, shift));
}
// Every tier passes 1 kHz unchanged (better than 0.001 dB) ...
assert!(results.iter().all(|(p, _, _)| p.abs() < 1e-3));
// ... removes the alias: for this 1 : 3 ratio the resampler drops everything
// above the new Nyquist frequency in the frequency domain, so what is left is
// f64 round-off, more than 180 dB down ...
assert!(results.iter().all(|&(_, a, _)| a < -180.0));
// ... and adds no delay: the shift is zero to rounding error.
assert!(results.iter().all(|(_, _, s)| s.abs() < 1e-9));
```

<!-- figure: ch07-alias-suppression -->

An impulse shows the same time alignment: one at input sample 3000
(48 kHz) peaks at output sample $3000/3 = 1000$ (16 kHz) for every tier, and
the streaming resampler, fed in chunks, produces the same output.

```rust
use audio_samples::operations::types::ResamplingQuality;
use audio_samples::{AudioSamples, StreamingResampler, channels, resample, sample_rate};
use non_empty_slice::NonEmptyVec;

let mut x = vec![0.0f64; 12_000];
x[3_000] = 1.0;
let input: AudioSamples<f64> = AudioSamples::from_mono_vec(NonEmptyVec::new(x.clone()).unwrap(), sample_rate!(48000));
for q in [ResamplingQuality::Fast, ResamplingQuality::Medium, ResamplingQuality::High] {
    let y = resample(&input, sample_rate!(16000), q).unwrap();
    let y = y.as_slice().unwrap();
    let peak = (0..y.len()).max_by(|&a, &b| y[a].abs().total_cmp(&y[b].abs())).unwrap();
    assert_eq!(peak, 1_000);
    // Symmetric filter: the response is symmetric about the peak.
    assert!((y[peak - 1] - y[peak + 1]).abs() < 1e-12);

    let mut stream = StreamingResampler::<f64>::new(sample_rate!(48000), sample_rate!(16000), channels!(1), q).unwrap();
    let mut streamed = Vec::new();
    for chunk in x.chunks(1_000) {
        let piece = AudioSamples::from_mono_vec(NonEmptyVec::new(chunk.to_vec()).unwrap(), sample_rate!(48000));
        if let Some(out) = stream.process_chunk(&piece).unwrap() {
            streamed.extend_from_slice(out.as_slice().unwrap());
        }
    }
    if let Some(out) = stream.finish().unwrap() {
        streamed.extend_from_slice(out.as_slice().unwrap());
    }
    assert_eq!(streamed.len(), y.len());
    assert!(streamed.iter().zip(y).all(|(a, b)| (a - b).abs() < 1e-12));
}
```

## How it works

**Naive decimation is plain sampling.** With $r = 1/3$, keeping every
third sample returns $x[3m]$, and
$\sin(2\pi\, 10\,000\, m / 16\,000) = -\sin(2\pi\, 6\,000\, m / 16\,000)$ for
every integer $m$: on the new grid the 10 kHz tone has exactly the samples of
a 6 kHz tone. No filter applied afterwards can separate the two, which is why
every tier filters first.

```rust
use audio_samples::{sample_rate, sine_wave};
use std::f64::consts::PI;
use std::time::Duration;

let x = sine_wave::<f64>(10_000.0, Duration::from_millis(100), sample_rate!(48000), 0.5);
let x = x.as_slice().unwrap();
let y: Vec<f64> = x.iter().step_by(3).copied().collect(); // 1600 samples at 16 kHz
assert_eq!(y.len(), 1_600);
for (m, &v) in y.iter().enumerate() {
    let six_khz = -0.5 * (2.0 * PI * 6_000.0 * m as f64 / 16_000.0).sin();
    // Tolerance: sin of arguments up to ~6e3 rad carries ~1e-12 of rounding.
    assert!((v - six_khz).abs() < 1e-9);
}
```

**A windowed-sinc decimator, written out.** Low-pass with a 193-tap
Blackman-windowed sinc whose cutoff is $0.95 \times 8$ kHz $= 7.6$ kHz, then
keep every third sample. The filter is centred (no delay), so the 1 kHz output
matches the ideal 16 kHz sine, and the 6 kHz alias is 107.5 dB down: the
filter's stopband, folded back. The library drops that part of the spectrum
for this ratio, so its tiers leave only round-off. Since the library
adds no delay either, its `High` output and this one agree sample by sample.

```rust
use audio_samples::operations::types::ResamplingQuality;
use audio_samples::{AudioSamples, resample, sample_rate};
use non_empty_slice::NonEmptyVec;
use std::f64::consts::PI;

let (fs_in, fs_out, half) = (48_000.0, 16_000.0, 96usize);
// Windowed-sinc low-pass h[k], k = 0..=2*half, cutoff fc (as a fraction of fs_in).
let fc = 0.95 * (fs_out / 2.0) / fs_in;
let m = 2 * half;
let h: Vec<f64> = (0..=m)
    .map(|i| {
        let t = i as f64 - half as f64;
        let sinc = if t == 0.0 { 2.0 * fc } else { (2.0 * PI * fc * t).sin() / (PI * t) };
        let u = i as f64 / m as f64;
        sinc * (0.42 - 0.5 * (2.0 * PI * u).cos() + 0.08 * (4.0 * PI * u).cos()) // Blackman
    })
    .collect();

// Input: 0.5 sin(1 kHz) + 0.5 sin(10 kHz), 250 ms at 48 kHz.
let x: Vec<f64> = (0..12_000)
    .map(|n| {
        let t = n as f64 / fs_in;
        0.5 * (2.0 * PI * 1_000.0 * t).sin() + 0.5 * (2.0 * PI * 10_000.0 * t).sin()
    })
    .collect();
// y[m] = sum_k h[k] x[3m + half - k]  (filter centred on the kept sample).
let y: Vec<f64> = (0..4_000)
    .map(|o| {
        (0..h.len())
            .filter_map(|k| x.get((3 * o + half).checked_sub(k)?).map(|v| h[k] * v))
            .sum()
    })
    .collect();

let amp = |f: f64| {
    let (re, im) = (1_200..2_800).fold((0.0, 0.0), |(re, im), n| {
        let ph = 2.0 * PI * f * n as f64 / fs_out;
        (re + y[n] * ph.cos(), im - y[n] * ph.sin())
    });
    2.0 * (re * re + im * im).sqrt() / 1_600.0
};
assert!((20.0 * (amp(1_000.0) / 0.5).log10()).abs() < 1e-3); // pass band intact
assert!((20.0 * (amp(6_000.0) / 0.5).log10() + 107.5).abs() < 0.1); // alias 107.5 dB down
// Matches the ideal 16 kHz sine to 1e-5 (window ripple in the pass band).
for n in 1_200..2_800 {
    let ideal = 0.5 * (2.0 * PI * 1_000.0 * n as f64 / fs_out).sin();
    assert!((y[n] - ideal).abs() < 1e-5);
}
// And the library's High tier, also centred, gives the same samples to 1e-5.
let input: AudioSamples<f64> = AudioSamples::from_mono_vec(NonEmptyVec::new(x).unwrap(), sample_rate!(48000));
let lib = resample(&input, sample_rate!(16000), ResamplingQuality::High).unwrap();
let lib = lib.as_slice().unwrap();
assert!((1_200..2_800).all(|n| (lib[n] - y[n]).abs() < 1e-5));
```

**Linear interpolation is a poor interpolation filter.** For a 6 kHz tone
upsampled from 16 kHz to 48 kHz with `Fast`, the formula for $G(f)$ predicts
an attenuation of the tone to $G(6\,000) = -3.77$ dB and an image at
$16\,000 - 6\,000 = 10\,000$ Hz at $G(10\,000) = -11.8$ dB.

```rust
use audio_samples::operations::types::ResamplingQuality;
use audio_samples::{resample, sample_rate, sine_wave};
use std::f64::consts::PI;
use std::time::Duration;

let fs = 48_000.0;
let g = |f: f64| {
    let w = 2.0 * PI * f / fs;
    20.0 * ((3.0 + 4.0 * w.cos() + 2.0 * (2.0 * w).cos()) / 9.0).abs().log10()
};
let x = sine_wave::<f64>(6_000.0, Duration::from_millis(250), sample_rate!(16000), 0.5);
let y = resample(&x, sample_rate!(48000), ResamplingQuality::Fast).unwrap();
let y = &y.as_slice().unwrap()[3_600..8_400]; // 100 ms: whole periods of 6 and 10 kHz
let level = |f: f64| {
    let (re, im) = y.iter().enumerate().fold((0.0, 0.0), |(re, im), (n, &v)| {
        let ph = 2.0 * PI * f * n as f64 / fs;
        (re + v * ph.cos(), im - v * ph.sin())
    });
    20.0 * (2.0 * (re * re + im * im).sqrt() / y.len() as f64 / 0.5).log10()
};
// Measured equals predicted to 1e-6 dB (the resampler computes in f64).
assert!((level(6_000.0) - g(6_000.0)).abs() < 1e-6);
assert!((level(10_000.0) - g(10_000.0)).abs() < 1e-6);
assert!((g(6_000.0) + 3.77).abs() < 0.01 && (g(10_000.0) + 11.8).abs() < 0.05);
```

The sinc tiers pass the 6 kHz tone unchanged (to 0.0001 dB) and remove the
image: for 1 : 3 they set every bin above 8 kHz to zero before the inverse
transform, so it is f64 round-off, more than 180 dB down:

```rust
use audio_samples::operations::types::ResamplingQuality;
use audio_samples::{resample, sample_rate, sine_wave};
use std::f64::consts::PI;
use std::time::Duration;

let x = sine_wave::<f64>(6_000.0, Duration::from_millis(250), sample_rate!(16000), 0.5);
for q in [ResamplingQuality::Medium, ResamplingQuality::High] {
    let y = resample(&x, sample_rate!(48000), q).unwrap();
    let y = &y.as_slice().unwrap()[3_600..8_400];
    let level = |f: f64| {
        let (re, im) = y.iter().enumerate().fold((0.0, 0.0), |(re, im), (n, &v)| {
            let ph = 2.0 * PI * f * n as f64 / 48_000.0;
            (re + v * ph.cos(), im - v * ph.sin())
        });
        20.0 * (2.0 * (re * re + im * im).sqrt() / y.len() as f64 / 0.5).log10()
    };
    assert!(level(6_000.0).abs() < 1e-4);
    assert!(level(10_000.0) < -180.0);
}
```

<!-- figure: ch07-upsampling-images -->

## Ask the library to explain

[`topics::resampling`](crate::educational::topics::resampling) summarises a
conversion: ratio, both Nyquist frequencies, the pass-band edge of the sinc
tiers, a table of the three tiers, and a warning for downsampling.

```rust
use audio_samples::educational::topics;

let doc = topics::resampling(48_000.0, 16_000.0);
println!("{doc}");
assert!((doc.quantity("Ratio").unwrap().value - 1.0 / 3.0).abs() < 1e-12);
assert_eq!(doc.quantity("Output Nyquist").unwrap().value, 8_000.0);
// Medium's pass band ends at 0.90 x 8 kHz.
assert!((doc.quantity("Sinc pass-band edge").unwrap().value - 7_200.0).abs() < 1e-9);
assert!(doc.has_warning_containing("Downsampling"));
// Upsampling discards nothing, so there is no such warning.
assert!(!topics::resampling(16_000.0, 48_000.0).has_warning_containing("Downsampling"));
```

## Exercises

1. **Alias of a 13 kHz tone.** A 13 kHz tone is converted from 48 kHz to 16 kHz
   by keeping every third sample. Predict the output frequency, then confirm
   with [`topics::aliasing`](crate::educational::topics::aliasing) and by
   measuring. Measure the residue that each tier of `resample` leaves.
2. **Droop at 2 kHz.** Predict the gain of `Fast` upsampling 16 kHz → 48 kHz
   at 2 kHz from $G(f)$, then measure it.
3. **Alignment at a non-integer ratio.** Convert a 1 kHz tone from 44.1 kHz to
   16 kHz ($r = 160/441$) with `High`. Determine whether the output is still
   time-aligned, and bound the shift by comparing with the ideal 16 kHz sine.

## Answers

1. $|13\,000 - 16\,000| = 3\,000$ Hz, at full level after naive decimation.
   Every tier removes it: measured over the middle 50 ms, `Fast` leaves
   $-156$ dB and `Medium` and `High` less than $-230$ dB (round-off). Over the whole
   100 ms all three read about $-74$ dB, set by the clicks at the ends of the
   tone (chapter 01).

```rust
use audio_samples::educational::topics;
use audio_samples::operations::types::ResamplingQuality;
use audio_samples::{resample, sample_rate, sine_wave};
use std::f64::consts::PI;
use std::time::Duration;

let amp = |y: &[f64]| {
    let (re, im) = y.iter().enumerate().fold((0.0, 0.0), |(re, im), (n, &v)| {
        let ph = 2.0 * PI * 3_000.0 * n as f64 / 16_000.0;
        (re + v * ph.cos(), im - v * ph.sin())
    });
    2.0 * (re * re + im * im).sqrt() / y.len() as f64
};
assert_eq!(topics::aliasing(13_000.0, 16_000.0).quantity("Alias frequency").unwrap().value, 3_000.0);
let x = sine_wave::<f64>(13_000.0, Duration::from_millis(100), sample_rate!(48000), 0.5);
let naive: Vec<f64> = x.as_slice().unwrap().iter().step_by(3).copied().collect();
assert_eq!(naive.len(), 1_600); // 300 whole periods of 3 kHz
assert!((amp(&naive) - 0.5).abs() < 1e-9);

for (q, mid_db) in [
    (ResamplingQuality::Fast, -150.0),
    (ResamplingQuality::Medium, -220.0),
    (ResamplingQuality::High, -220.0),
] {
    let y = resample(&x, sample_rate!(16000), q).unwrap();
    let y = y.as_slice().unwrap();
    let db = |a: f64| 20.0 * (a / 0.5).log10();
    assert!(db(amp(&y[400..1_200])) < mid_db); // middle 50 ms: 150 periods
    assert!((db(amp(y)) + 74.0).abs() < 0.5); // whole signal: the edge clicks
}
```

2. $\omega = 2\pi \cdot 2000/48000 = \pi/12$, so
   $G = (3 + 4\cos(\pi/12) + 2\cos(\pi/6))/9 = 0.955$, i.e. $-0.40$ dB.

```rust
use audio_samples::operations::types::ResamplingQuality;
use audio_samples::{resample, sample_rate, sine_wave};
use std::f64::consts::PI;
use std::time::Duration;

let w = PI / 12.0;
let predicted = 20.0 * ((3.0 + 4.0 * w.cos() + 2.0 * (2.0 * w).cos()) / 9.0).log10();
assert!((predicted + 0.40).abs() < 0.005);

let x = sine_wave::<f64>(2_000.0, Duration::from_millis(250), sample_rate!(16000), 0.5);
let y = resample(&x, sample_rate!(48000), ResamplingQuality::Fast).unwrap();
let y = &y.as_slice().unwrap()[3_600..8_400];
let (re, im) = y.iter().enumerate().fold((0.0, 0.0), |(re, im), (n, &v)| {
    let ph = 2.0 * PI * 2_000.0 * n as f64 / 48_000.0;
    (re + v * ph.cos(), im - v * ph.sin())
});
let measured = 20.0 * (2.0 * (re * re + im * im).sqrt() / y.len() as f64 / 0.5).log10();
assert!((measured - predicted).abs() < 0.01);
```

3. It is. Output sample $m$ is evaluated at the exact input position
   $441 m / 160$, so there is no delay to remove and nothing is rounded to
   whole frames. In the middle 100 ms the output differs from the ideal
   16 kHz sine by less than $10^{-7}$; a shift of $\tau$ output frames would
   cause an error of up to $0.5 \cdot (2\pi/16) \cdot \tau \approx 0.196\,\tau$, so
   $|\tau| < 5.1 \times 10^{-7}$ frames.

```rust
use audio_samples::operations::types::ResamplingQuality;
use audio_samples::{resample, sample_rate, sine_wave};
use std::f64::consts::PI;
use std::time::Duration;

let x = sine_wave::<f64>(1_000.0, Duration::from_millis(250), sample_rate!(44100), 0.5);
let y = resample(&x, sample_rate!(16000), ResamplingQuality::High).unwrap();
let y = y.as_slice().unwrap();
assert_eq!(y.len(), 4_000); // ceil(11025 * 160 / 441)
let err = (800..2_400)
    .map(|n| (y[n] - 0.5 * (2.0 * PI * 1_000.0 * n as f64 / 16_000.0).sin()).abs())
    .fold(0.0, f64::max);
assert!(err < 1e-7);
let slope = 0.5 * 2.0 * PI / 16.0; // largest error per frame of shift
assert!((slope - 0.196).abs() < 0.001);
assert!(1e-7 / slope < 5.1e-7);
```

## Common pitfalls

- **Naive decimation is not resampling.** Keeping every $q$-th sample folds
  everything above the new Nyquist frequency back at full level; every
  `ResamplingQuality` filters first.
- **`Fast` trades bandwidth and rejection for speed.** Its downsampling
  filter starts to roll off at $0.80 f_N$ and is designed for 70 dB; when
  upsampling it is linear interpolation, which droops the pass band and leaves
  images. Use `Medium` or `High` for analysis or listening.
- **Measure rejection away from the ends.** The input is treated as zero
  outside the buffer, so a signal that starts or stops abruptly produces
  broadband clicks that no anti-aliasing filter removes.
- **SciPy differs in defaults.** `scipy.signal.resample` uses the FFT
  (assuming a periodic signal); `resample_poly` uses a Kaiser-windowed FIR.
  Expect small differences in edge behaviour and alignment, not in the pass
  band.
- **Output length** is $\lceil N r \rceil$ frames; durations that are not whole
  numbers of output samples round up.
- **Streaming versus offline.** [`StreamingResampler`](crate::StreamingResampler)
  gives the same, time-aligned samples as [`resample`](crate::resample); it
  holds back the filter's look-ahead until more input (or
  [`finish`](crate::StreamingResampler::finish)) arrives.

## References

- R. E. Crochiere and L. R. Rabiner, *Multirate Digital Signal Processing*, Prentice-Hall, 1983.
- A. V. Oppenheim and R. W. Schafer, *Discrete-Time Signal Processing*, 3rd ed., Pearson, 2010. Sampling-rate conversion.

## See also

- [`resample`](crate::resample), [`resample_by_ratio`](crate::resample_by_ratio),
  [`ResamplingQuality`](crate::operations::types::ResamplingQuality),
  [`StreamingResampler`](crate::StreamingResampler)
- [`topics::resampling`](crate::educational::topics::resampling),
  [`topics::aliasing`](crate::educational::topics::aliasing)
- Previous chapter: [06. IIR filters](06_iir_filters.md);
  next chapter: [08. LPC, the source-filter model and formants](08_lpc_formants.md);
  aliasing basics: [01. Sampling and aliasing](01_sampling_aliasing.md)
- Lab: `examples/tutorial_07_resampling.rs`
  (`cargo run --release --example tutorial_07_resampling --features educational,full_no_plotting`)
