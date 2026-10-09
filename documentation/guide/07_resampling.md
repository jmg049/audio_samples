# 07. Resampling

Speech models are trained at 16 kHz, CDs are 44.1 kHz, video is 48 kHz:
almost every audio pipeline converts sample rates somewhere. Done carelessly,
downsampling folds high frequencies back into the band as aliases and
upsampling leaves spectral images. This chapter measures both effects on the
library's three quality tiers and rebuilds a windowed-sinc decimator by hand.

## Concept

**Ratio.** Converting from $f_{in}$ to $f_{out}$ (both in Hz) has ratio

$$r = f_{out} / f_{in}$$

(dimensionless). The output can only represent frequencies below its Nyquist
frequency $f_{out}/2$.

**Downsampling and aliasing.** Keeping every $q$-th sample ($r = 1/q$) is
sampling the signal again at $f_{out}$. A component at $f$ Hz above
$f_{out}/2$ reappears at

$$f_a = |f - k f_{out}|$$

where $k$ is the integer that brings $f_a$ into $\[0, f_{out}/2\]$ (chapter 01).
Once folded it is indistinguishable from real content, so a resampler must
**low-pass filter before decimating**, with a cutoff at or below
$\min(f_{in}, f_{out})/2$.

**Upsampling and imaging.** Raising the rate by $L$ ($r = L$) can be modelled
as inserting $L - 1$ zeros between samples (which leaves copies, *images*, of
the spectrum at $k f_{in} \pm f$) followed by a low-pass *interpolation*
filter that removes them.

**Ideal band-limited interpolation.** A band-limited signal is fully described
by its samples, and its value at any time $t$ (seconds) is

$$x(t) = \sum_n x\[n\]\, \mathrm{sinc}(f_{in} t - n), \quad \mathrm{sinc}(u) = \frac{\sin(\pi u)}{\pi u}$$

A practical resampler truncates and windows the sinc, and scales it so that its
cutoff is $f_c = \min(f_{in}, f_{out})/2$ (slightly less, to leave room for the
transition band). Then one formula does both jobs: interpolation and
anti-aliasing.

**Linear interpolation** uses a triangular kernel instead of a sinc. For
integer upsampling by $L = 3$ it equals zero-stuffing followed by the FIR
$\[1, 2, 3, 2, 1\]/3$, whose gain at frequency $f$ (Hz) is

$$G(f) = \frac{3 + 4\cos\omega + 2\cos 2\omega}{9}, \quad \omega = 2\pi f / f_{out}$$

where the extra $1/3$ undoes the gain of 3 that the kernel has at DC. $G$ is
not flat in the pass band and not zero at the images.

**Alias (or image) rejection** is the level of the unwanted component
relative to the input component that produced it:

$$R = 20 \log_{10}(A_{alias} / A_{in}) \ \text{dB}$$

where $A$ is a sinusoid's amplitude. 0 dB means no rejection at all.

**The library's tiers** ([`ResamplingQuality`](crate::operations::types::ResamplingQuality)):

- `Fast`: **no anti-aliasing filter.** Integer down-ratios keep every $q$-th
  sample; other ratios use linear interpolation.
- `Medium`: windowed sinc (rubato), 128 taps, Hann² window, cutoff
  $0.95 \times$ the lower Nyquist frequency.
- `High`: windowed sinc, 256 taps, Blackman-Harris² window, cubic
  interpolation of the kernel table, same cutoff.

**Time alignment.** A centred sinc filter delays the signal by half its
length. The whole-buffer [`resample`](crate::resample) removes that delay in
whole output frames, $\mathrm{round}(L_{sinc}\, r / 2 - 1)$ where $L_{sinc}$ is
the number of sinc taps, so input sample $n$ lands at output position $n r$ to
within half an output frame. (The streaming
[`StreamingResampler`](crate::StreamingResampler) cannot look ahead; it reports
its delay with `output_delay`.)

## With the library

Mix a 1 kHz tone (to keep) with a 10 kHz tone (to remove), both at amplitude
0.5, and convert 48 kHz to 16 kHz with each tier. The 10 kHz tone should fold
to $|10\,000 - 16\,000| = 6\,000$ Hz. Measure each component with one DFT bin
over 1600 output samples (whole periods of both 1 kHz and 6 kHz, so there is
no leakage).

```rust
use audio_samples::operations::types::ResamplingQuality;
use audio_samples::{AudioSamples, resample, sample_rate, sine_wave};
use non_empty_slice::NonEmptyVec;
use std::f64::consts::PI;
use std::time::Duration;

/// Amplitude of the component at f Hz (one DFT bin; exact for whole periods).
fn amplitude(x: &[f64], f: f64, fs: f64) -> f64 {
    let (re, im) = x.iter().enumerate().fold((0.0, 0.0), |(re, im), (n, &v)| {
        let ph = 2.0 * PI * f * n as f64 / fs;
        (re + v * ph.cos(), im - v * ph.sin())
    });
    2.0 * (re * re + im * im).sqrt() / x.len() as f64
}

let dur = Duration::from_millis(250);
let a = sine_wave::<f64>(1_000.0, dur, sample_rate!(48000), 0.5);
let b = sine_wave::<f64>(10_000.0, dur, sample_rate!(48000), 0.5);
let x: Vec<f64> = a.as_slice().unwrap().iter().zip(b.as_slice().unwrap()).map(|(p, q)| p + q).collect();
let input: AudioSamples<f64> = AudioSamples::from_mono_vec(NonEmptyVec::new(x).unwrap(), sample_rate!(48000));

let mid = 1_200..2_800; // middle 100 ms of the 16 kHz output
let mut results = Vec::new();
for q in [ResamplingQuality::Fast, ResamplingQuality::Medium, ResamplingQuality::High] {
    let y = resample(&input, sample_rate!(16000), q).unwrap();
    assert_eq!(y.samples_per_channel().get(), 4_000); // ceil(12000 / 3)
    let y = y.as_slice().unwrap();
    let pass_db = 20.0 * (amplitude(&y[mid.clone()], 1_000.0, 16_000.0) / 0.5).log10();
    let alias_db = 20.0 * (amplitude(&y[mid.clone()], 6_000.0, 16_000.0) / 0.5).log10();
    results.push((pass_db, alias_db));
}
// Every tier passes 1 kHz unchanged (better than 0.001 dB).
assert!(results.iter().all(|(p, _)| p.abs() < 1e-3));
// Fast: the alias arrives at full level, 0 dB.
assert!(results[0].1.abs() < 0.01);
// Medium rejects it by about 102 dB, High by about 142 dB (close to the
// f32 noise floor; the resampler computes in f32). 1 dB tolerances.
assert!((results[1].1 + 101.8).abs() < 1.0);
assert!((results[2].1 + 141.6).abs() < 1.0);
```

The sinc tiers are also time-aligned: an impulse at input sample 3000
(48 kHz) peaks at output sample $3000/3 = 1000$ (16 kHz).

```rust
use audio_samples::operations::types::ResamplingQuality;
use audio_samples::{AudioSamples, resample, sample_rate};
use non_empty_slice::NonEmptyVec;

let mut x = vec![0.0f64; 12_000];
x[3_000] = 1.0;
let input: AudioSamples<f64> = AudioSamples::from_mono_vec(NonEmptyVec::new(x).unwrap(), sample_rate!(48000));
for q in [ResamplingQuality::Medium, ResamplingQuality::High] {
    let y = resample(&input, sample_rate!(16000), q).unwrap();
    let y = y.as_slice().unwrap();
    let peak = (0..y.len()).max_by(|&a, &b| y[a].abs().total_cmp(&y[b].abs())).unwrap();
    assert_eq!(peak, 1_000);
}
```

The alignment is exact only to within half an output frame. For this
conversion the removed delay is $\mathrm{round}(256/3/2 - 1) = 42$ frames
against an exact $41.667$ for High, so the output *leads* by about one third of
an output frame (one input sample); the lab measures $+0.335$ frames from the
phase of the 1 kHz tone (and $-0.331$ for Medium, predicted $-1/3$).

## Under the hood

**`Fast` is plain decimation.** With $r = 1/3$ it returns $x\[3m\]$, and
$\sin(2\pi\, 10\,000\, m / 16\,000) = -\sin(2\pi\, 6\,000\, m / 16\,000)$ for
every integer $m$: the 10 kHz tone *is* a 6 kHz tone on the new grid.

```rust
use audio_samples::operations::types::ResamplingQuality;
use audio_samples::{resample, sample_rate, sine_wave};
use std::f64::consts::PI;
use std::time::Duration;

let x = sine_wave::<f64>(10_000.0, Duration::from_millis(100), sample_rate!(48000), 0.5);
let y = resample(&x, sample_rate!(16000), ResamplingQuality::Fast).unwrap();
let (x, y) = (x.as_slice().unwrap(), y.as_slice().unwrap());
for m in 0..y.len() {
    assert!((y[m] - x[3 * m]).abs() < 1e-6); // the resampler works in f32
    let six_khz = -0.5 * (2.0 * PI * 6_000.0 * m as f64 / 16_000.0).sin();
    assert!((y[m] - six_khz).abs() < 1e-6);
}
```

**A windowed-sinc decimator, written out.** Low-pass with a 193-tap
Blackman-windowed sinc whose cutoff is $0.95 \times 8$ kHz $= 7.6$ kHz, then
keep every third sample. The filter is centred (no delay), so the 1 kHz output
matches the ideal 16 kHz sine, and the 6 kHz alias is more than 100 dB down,
in the same class as the library's `Medium` tier.

```rust
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
assert!(20.0 * (amp(6_000.0) / 0.5).log10() < -100.0); // alias > 100 dB down
// Matches the ideal 16 kHz sine to 1e-5 (window ripple in the pass band).
for n in 1_200..2_800 {
    let ideal = 0.5 * (2.0 * PI * 1_000.0 * n as f64 / fs_out).sin();
    assert!((y[n] - ideal).abs() < 1e-5);
}
```

**Linear interpolation is a poor interpolation filter.** Upsampling a 6 kHz
tone from 16 kHz to 48 kHz with `Fast` should, by the formula for $G(f)$, droop
the tone to $G(6\,000) = -3.77$ dB and leave an image at
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
// Measured equals predicted to 0.01 dB (the resampler computes in f32).
assert!((level(6_000.0) - g(6_000.0)).abs() < 0.01);
assert!((level(10_000.0) - g(10_000.0)).abs() < 0.01);
assert!((g(6_000.0) + 3.77).abs() < 0.01 && (g(10_000.0) + 11.8).abs() < 0.05);
```

The sinc tiers, in contrast, pass the 6 kHz tone flat and push the image below
$-95$ dB (the lab prints $-98$ dB for Medium and $-145$ dB for High).

## Ask the library to explain

[`topics::resampling`](crate::educational::topics::resampling) summarises a
conversion: ratio, both Nyquist frequencies, the pass-band edge of the sinc
tiers, and a warning when you downsample.

```rust
use audio_samples::educational::topics;

let doc = topics::resampling(48_000.0, 16_000.0);
println!("{doc}");
assert!((doc.quantity("Ratio").unwrap().value - 1.0 / 3.0).abs() < 1e-12);
assert_eq!(doc.quantity("Output Nyquist").unwrap().value, 8_000.0);
// 0.95 x 8 kHz; the 0.95 is stored as f32, hence the 1e-3 Hz tolerance.
assert!((doc.quantity("Sinc pass-band edge").unwrap().value - 7_600.0).abs() < 1e-3);
assert!(doc.has_warning_containing("Downsampling"));
// Upsampling discards nothing, so there is no such warning.
assert!(!topics::resampling(16_000.0, 48_000.0).has_warning_containing("Downsampling"));
```

## Exercises

1. **Where does 13 kHz go?** A 13 kHz tone is converted from 48 kHz to 16 kHz
   with `Fast`. Predict the output frequency, then confirm with
   [`topics::aliasing`](crate::educational::topics::aliasing) and by measuring.
2. **Droop at 2 kHz.** Predict the gain of `Fast` upsampling 16 kHz → 48 kHz
   at 2 kHz from $G(f)$, then measure it.
3. **When is the alignment exact?** For 48 kHz → 24 kHz with `High`
   ($L_{sinc} = 256$), compute $L_{sinc} r/2 - 1$. Is the removed delay exact?
   Measure the residual shift of a 1 kHz tone.

## Answers

1. $|13\,000 - 16\,000| = 3\,000$ Hz, at full level because `Fast` has no
   anti-aliasing filter.

```rust
use audio_samples::educational::topics;
use audio_samples::operations::types::ResamplingQuality;
use audio_samples::{resample, sample_rate, sine_wave};
use std::f64::consts::PI;
use std::time::Duration;

assert_eq!(topics::aliasing(13_000.0, 16_000.0).quantity("Alias frequency").unwrap().value, 3_000.0);
let x = sine_wave::<f64>(13_000.0, Duration::from_millis(100), sample_rate!(48000), 0.5);
let y = resample(&x, sample_rate!(16000), ResamplingQuality::Fast).unwrap();
let y = y.as_slice().unwrap(); // 1600 samples = 300 whole periods of 3 kHz
let (re, im) = y.iter().enumerate().fold((0.0, 0.0), |(re, im), (n, &v)| {
    let ph = 2.0 * PI * 3_000.0 * n as f64 / 16_000.0;
    (re + v * ph.cos(), im - v * ph.sin())
});
let amp = 2.0 * (re * re + im * im).sqrt() / y.len() as f64;
assert!((amp - 0.5).abs() < 1e-6);
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

3. $256 \cdot 0.5 / 2 - 1 = 63$ exactly, so rounding loses nothing and the
   output is aligned to within the filter's tiny phase error (the measured
   residual is below 0.01 output frames).

```rust
use audio_samples::operations::types::ResamplingQuality;
use audio_samples::{resample, sample_rate, sine_wave};
use std::f64::consts::PI;
use std::time::Duration;

let exact: f64 = 256.0 * 0.5 / 2.0 - 1.0;
assert_eq!(exact.round() - exact, 0.0);

let x = sine_wave::<f64>(1_000.0, Duration::from_millis(250), sample_rate!(48000), 0.5);
let y = resample(&x, sample_rate!(24000), ResamplingQuality::High).unwrap();
let y = y.as_slice().unwrap();
// Compare with the ideal 24 kHz sine in the middle 100 ms. A shift of tau
// output frames would cause an error of up to 0.5 * (2 pi / 24) * tau, so an
// error below 1e-3 means tau < 0.008 frames.
let err = (1_200..3_600)
    .map(|n| (y[n] - 0.5 * (2.0 * PI * 1_000.0 * n as f64 / 24_000.0).sin()).abs())
    .fold(0.0, f64::max);
assert!(err < 1e-3);
```

## Common pitfalls

- **`Fast` is not "slightly worse"; it has no anti-aliasing at all** when
  downsampling. Use `Medium` or `High` whenever the input may contain energy
  above the new Nyquist frequency.
- **Resampling is done in `f32`**, so rejection bottoms out around
  $-140$ dB, whatever the sample type of your buffer.
- **librosa / SciPy differ in defaults.** `librosa.resample` uses soxr by
  default and `scipy.signal.resample` uses the FFT (assuming a periodic
  signal); `resample_poly` uses a Kaiser-windowed FIR. Expect small differences
  in edge behaviour and alignment, not in the pass band.
- **Output length** is $\lceil N r \rceil$ frames; durations that are not whole
  numbers of output samples round up.
- **Streaming versus offline.** [`StreamingResampler`](crate::StreamingResampler)
  output lags by `output_delay()` frames; the whole-buffer functions remove
  the delay for you.

## See also

- [`resample`](crate::resample), [`resample_by_ratio`](crate::resample_by_ratio),
  [`ResamplingQuality`](crate::operations::types::ResamplingQuality),
  [`StreamingResampler`](crate::StreamingResampler)
- [`topics::resampling`](crate::educational::topics::resampling),
  [`topics::aliasing`](crate::educational::topics::aliasing)
- Previous chapter: [06. IIR filters](crate::guide::c06_iir_filters);
  next chapter: [08. LPC, the source-filter model and formants](crate::guide::c08_lpc_formants);
  aliasing basics: [01. Sampling and aliasing](crate::guide::c01_sampling_aliasing)
- Lab: `examples/tutorial_07_resampling.rs`
  (`cargo run --release --example tutorial_07_resampling --features educational,full_no_plotting`)
