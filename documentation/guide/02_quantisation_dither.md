# 02. Quantisation and dither

Sampling (chapter 01) makes time discrete; quantisation makes amplitude
discrete. A 16-bit file can store only 65 536 different values, so every sample
is rounded, and the rounding error behaves like added noise. The level of that
noise, and whether it is heard as noise or as distortion, determine how many
bits a recording needs and how to reduce bit depth without damaging quiet passages
such as a fading voice or a reverb tail.

## Concept

**Uniform quantisation** maps a sample $x$ on the normalised scale
$[-1, 1]$ to an integer code $q$ and back. This crate uses an *asymmetric*
mapping (see [`conversions`](crate::conversions)) so that both $-1$ and $+1$
are reachable and the integer round trip is exact:

$$q = \mathrm{round}(x \cdot 2^{b-1}) \;\; (x < 0), \qquad q = \mathrm{round}\big(x \cdot (2^{b-1} - 1)\big) \;\; (x \ge 0)$$

where $b$ is the word length in bits, $x$ is clamped to $[-1, 1]$ and
$\mathrm{round}$ rounds half away from zero. Converting back divides by the same
factors. The **step size** (one least-significant bit, LSB, in units of full
scale) is therefore

$$\Delta = 2^{1-b} \;\; \text{(below zero)}, \qquad \Delta_+ = \frac{1}{2^{b-1} - 1} \;\; \text{(above zero)}$$

For $b = 16$: $\Delta = 1/32768$ and $\Delta_+ = 1/32767$, a difference of
0.003 %.

**Quantisation noise.** If the signal is large and varied compared with one
step, the rounding error
$e = \hat{x} - x$ is spread uniformly over one step, $[-\Delta/2, \Delta/2]$,
so its power is

$$\sigma_e^2 = \frac{\Delta^2}{12}$$

in units of full scale squared, where $\hat{x}$ is the quantised value.

**Levels in dB.** This guide quotes powers in **dB re 1**, i.e. relative to
full scale squared ($10\log_{10}$ of a mean square, so 0 dB is the power of a
full-scale DC level or square wave), and RMS levels as $20\log_{10}$ of the RMS
in **dBFS** with the same reference, so a full-scale sine (power $1/2$) is at
$-3.01$ dBFS. The AES17 standard instead defines 0 dBFS as the RMS of a
full-scale sine, so AES17 figures are 3.01 dB higher. A level read from the
largest absolute sample is called **dBFS peak**: a sine of amplitude 0.1 is at
$-20$ dBFS peak and $-23.01$ dBFS RMS.

**SQNR of a full-scale sine.** A sine of amplitude 1 has power $1/2$. With
$\Delta = 2/2^b$:

$$\mathrm{SQNR} = 10 \log_{10} \frac{1/2}{\Delta^2/12} = 10\log_{10}\!\left(\tfrac{3}{2} \, 4^{b}\right) \approx 6.02\,b + 1.76 \ \mathrm{dB}$$

Each extra bit adds about 6 dB. A signal peaking $L$ dB below full scale loses
$L$ dB of SQNR.

**Dither.** For a quiet signal only a few steps tall, the error is *not*
uniform noise: it is a deterministic function of the signal, so it repeats with
the signal and appears as harmonics (distortion). **TPDF dither** adds, before
rounding, a random value $d = u_1 + u_2$ where $u_1, u_2$ are independent and
uniform on $[-\Delta/2, \Delta/2)$; $d$ has a triangular density on
$[-\Delta, \Delta]$ and power $\Delta^2/6$. The total error power becomes

$$\sigma^2_{e,\mathrm{TPDF}} = \frac{\Delta^2}{12} + \frac{\Delta^2}{6} = \frac{\Delta^2}{4}$$

i.e. $10\log_{10} 3 \approx 4.77$ dB more noise, but the error's mean and
variance no longer depend on the signal: the distortion is replaced by a steady
hiss.
This crate applies **no** dither anywhere, so the chapter implements it.

## With the library

The crate's convention, its step sizes and the SQNR at 16 bits through the
`i16` conversion ([`to_format`](crate::AudioTypeConversion::to_format)),
measured with [`comparison::snr`](crate::utils::comparison::snr):

```rust
use audio_samples::educational::topics;
use audio_samples::utils::comparison::snr;
use audio_samples::{AudioSamples, AudioTypeConversion, sample_rate, sine_wave};
use ndarray::array;
use std::time::Duration;

// The asymmetric mapping: -1 -> -32768, +1 -> 32767.
let edge = AudioSamples::new_mono(array![-1.0f64, -0.5, 0.0, 0.5, 1.0], sample_rate!(16000)).unwrap();
let codes = edge.to_format::<i16>();
assert_eq!(codes.as_slice().unwrap(), &[-32768, -16384, 0, 16384, 32767]);

let doc = topics::quantisation(16);
assert_eq!(doc.quantity("Quantisation levels").unwrap().value, 65_536.0);
assert_eq!(doc.quantity("Step size (negative half)").unwrap().value, 1.0 / 32768.0);
assert!((doc.quantity("Step size (positive half)").unwrap().value - 1.0 / 32767.0).abs() < 1e-18);

// Full-scale 997 Hz sine at 48 kHz, 1 s, rounded to i16 and back.
let fs = sample_rate!(48000);
let x = sine_wave::<f64>(997.0, Duration::from_secs(1), fs, 1.0);
let back = x.to_format::<i16>().to_format::<f64>();
let err: Vec<f64> = back.as_slice().unwrap().iter()
    .zip(x.as_slice().unwrap()).map(|(q, v)| q - v).collect();
let err = AudioSamples::new_mono(ndarray::Array1::from(err), fs).unwrap();
let measured = snr(&x, &err).unwrap();
let predicted = doc.quantity("SQNR (full-scale sine)").unwrap().value;
assert!((predicted - 98.09).abs() < 0.01);
assert!((measured - predicted).abs() < 0.05); // measured: 98.10 dB
```

The quiet case: a 997 Hz sine of amplitude 2.5 LSB at 16 kHz, rounded to
`i16` with and without TPDF dither. The dither is built from the crate's
[`white_noise`](crate::utils::generation::white_noise), which is uniform on
$[-A, A)$, so $A = \Delta/2$:

```rust
use audio_samples::{AudioSamples, AudioTransforms, AudioTypeConversion, nzu, sample_rate,
                    sine_wave, white_noise};
use std::time::Duration;

let fs = sample_rate!(16000);
let one_s = Duration::from_secs(1);
let lsb = 1.0 / 32768.0;
let x = sine_wave::<f64>(997.0, one_s, fs, 2.5 * lsb);
let xs = x.as_slice().unwrap().to_vec();

let u1 = white_noise::<f64>(one_s, fs, lsb / 2.0, Some(1));
let u2 = white_noise::<f64>(one_s, fs, lsb / 2.0, Some(2));
let dither: Vec<f64> = u1.as_slice().unwrap().iter()
    .zip(u2.as_slice().unwrap()).map(|(a, b)| a + b).collect();

let to_i16_and_back = |v: Vec<f64>| -> Vec<f64> {
    AudioSamples::new_mono(ndarray::Array1::from(v), fs).unwrap()
        .to_format::<i16>().to_format::<f64>().as_slice().unwrap().to_vec()
};
let plain = to_i16_and_back(xs.clone());
let dith = to_i16_and_back(xs.iter().zip(&dither).map(|(a, d)| a + d).collect());
let e_plain: Vec<f64> = plain.iter().zip(&xs).map(|(y, v)| y - v).collect();
let e_dith: Vec<f64> = dith.iter().zip(&xs).map(|(y, v)| y - v).collect();

let power = |v: &[f64]| v.iter().map(|a| a * a).sum::<f64>() / v.len() as f64;
// Dithered error power is Δ²/4 = 3 x (Δ²/12). Measured: 3.02 x, against
// 1.23 x without dither (a deterministic error need not be uniform).
assert!((power(&e_dith) / (lsb * lsb / 12.0) - 3.0).abs() < 0.1);
assert!((power(&e_plain) / (lsb * lsb / 12.0) - 1.23).abs() < 0.01);

// Distortion: the error's 3rd-harmonic bin (2991 Hz) against its mean bin power.
let spur_db = |e: &[f64]| {
    let a = AudioSamples::new_mono(ndarray::Array1::from(e.to_vec()), fs).unwrap();
    let m = a.rfft(nzu!(16000)).unwrap().row(0).to_vec(); // 1 Hz bins
    let mean = m.iter().map(|v| v * v).sum::<f64>() / m.len() as f64;
    10.0 * (m[2991] * m[2991] / mean).log10()
};
assert!(spur_db(&e_plain) > 30.0); // measured +31.9 dB: a clear harmonic
assert!(spur_db(&e_dith) < 3.0);   // measured -4.5 dB: noise only

// Correlation between error and signal: -0.516 undithered, about 0 dithered.
let corr = |e: &[f64]| {
    let num: f64 = e.iter().zip(&xs).map(|(a, b)| a * b).sum();
    num / ((power(e) * power(&xs)).sqrt() * xs.len() as f64)
};
assert!(corr(&e_plain) < -0.5);
assert!(corr(&e_dith).abs() < 0.03);
```

<!-- figure: ch02-staircase -->

Dither raised the error power to $3 \times \Delta^2/12$ (from $1.23 \times$
without dither, so by a factor of about 2.45 here), and the result is still
better: the undithered error has a correlation of about $-0.5$ with the signal
and is concentrated at harmonics of 997 Hz, which is heard as distortion, while
the dithered error has a flat spectrum with no harmonics.

<!-- figure: ch02-dither-spectra -->

## How it works

Write the requantiser in plain Rust with the same convention, check that it is
*bit-identical* to the crate's 16-bit and 8-bit (`u8`, offset binary) round
trips, then measure SQNR at several word lengths against $6.02\,b + 1.76$.

```rust
use audio_samples::{AudioTypeConversion, sample_rate, sine_wave};
use std::time::Duration;

fn requantise(x: f64, b: u32) -> f64 {
    let half = 2f64.powi(b as i32 - 1);
    let x = x.clamp(-1.0, 1.0);
    if x < 0.0 { (x * half).round() / half } else { (x * (half - 1.0)).round() / (half - 1.0) }
}

let x = sine_wave::<f64>(997.0, Duration::from_secs(1), sample_rate!(48000), 1.0);
let xs = x.as_slice().unwrap();

// Same as the crate, sample for sample (to the last bit, up to division rounding).
let via_i16 = x.to_format::<i16>().to_format::<f64>();
let via_u8 = x.to_format::<u8>().to_format::<f64>();
for ((&v, &a), &b) in xs.iter().zip(via_i16.as_slice().unwrap()).zip(via_u8.as_slice().unwrap()) {
    assert!((a - requantise(v, 16)).abs() < 1e-15);
    assert!((b - requantise(v, 8)).abs() < 1e-15);
}

let p_signal = xs.iter().map(|v| v * v).sum::<f64>() / xs.len() as f64;
// (bits, measured SQNR in dB to 0.01, as printed by the lab)
for (b, expected) in [(4u32, 25.62), (8, 49.98), (12, 74.03), (16, 98.10)] {
    let p_err = xs.iter().map(|&v| (requantise(v, b) - v).powi(2)).sum::<f64>() / xs.len() as f64;
    let sqnr = 10.0 * (p_signal / p_err).log10();
    assert!((sqnr - expected).abs() < 0.01);
    // The rule of thumb: within 0.25 dB at 4 and 8 bits (the error is less
    // uniform when the steps are coarse), within 0.05 dB from 12 bits up.
    let rule = 6.02 * f64::from(b) + 1.76;
    assert!((sqnr - rule).abs() < if b >= 12 { 0.05 } else { 0.25 });
}

// Effect of the asymmetry at 16 bits: the steps differ by a factor
// 32768/32767 (0.003 %); averaging the two halves' noise powers:
let eps = 32768.0f64 / 32767.0 - 1.0;
assert!((eps - 3.05e-5).abs() < 1e-7);
let noise_ratio = (1.0 + (1.0 + eps).powi(2)) / 2.0;
assert!((noise_ratio - 1.0 - 3.05e-5).abs() < 1e-7);
assert!(10.0 * noise_ratio.log10() < 1.4e-4); // about 0.0001 dB
```

The asymmetric steps have a negligible effect: at 16 bits the positive-half step is larger
by a factor $32768/32767$, which changes the noise power by about $3\times10^{-5}$
(0.0001 dB), as the last lines check.

## Ask the library to explain

[`topics::quantisation`](crate::educational::topics::quantisation) states the
convention, the step sizes, the noise level, the SQNR and the SQNR with TPDF
dither:

```rust
use audio_samples::educational::topics;

let doc = topics::quantisation(16);
let sqnr = doc.quantity("SQNR (full-scale sine)").unwrap().value;
let rule = doc.quantity("Rule of thumb").unwrap().value;
let dithered = doc.quantity("SQNR with TPDF dither").unwrap().value;
assert!((sqnr - 98.09).abs() < 0.01 && (rule - 98.09).abs() < 0.01);
assert!((sqnr - dithered - 4.77).abs() < 0.01); // 10 log10 3
// Noise power Δ²/12 in dB re 1 (full scale squared) ...
let noise = doc.quantity("Noise level").unwrap().value;
assert!((noise - (-101.10)).abs() < 0.01);
// ... which is -98.09 dBFS in the AES17 convention (full-scale sine = 0 dBFS),
// 3.01 dB higher. In AES17 terms SQNR = -(noise level in dBFS).
let aes17 = doc.quantity("Noise level (AES17 dBFS)").unwrap().value;
assert!((aes17 - (-98.09)).abs() < 0.01);
assert!((aes17 - noise - 10.0 * 2f64.log10()).abs() < 1e-9);
assert!((sqnr + aes17).abs() < 1e-9);
assert!(doc.has_warning_containing("i16"));
println!("{doc}");
```

## Exercises

1. How many bits are needed for an SQNR of at least 90 dB on a full-scale
   sine? Predict from the rule, then confirm by measurement at that word length
   and one bit fewer.
2. A sine peaks at $-20$ dBFS peak (amplitude 0.1) and is stored as `i16`.
   Predict its SQNR, then measure it.
3. Predict how much TPDF dither lowers the SQNR of a loud sine (amplitude 0.9)
   quantised to 8 bits, and measure it.

## Answers

```rust
use audio_samples::utils::comparison::snr;
use audio_samples::{AudioSamples, AudioTypeConversion, sample_rate, sine_wave, white_noise};
use std::time::Duration;

fn requantise(x: f64, b: u32) -> f64 {
    let half = 2f64.powi(b as i32 - 1);
    let x = x.clamp(-1.0, 1.0);
    if x < 0.0 { (x * half).round() / half } else { (x * (half - 1.0)).round() / (half - 1.0) }
}
fn sqnr(x: &[f64], y: &[f64]) -> f64 {
    let ps: f64 = x.iter().map(|v| v * v).sum();
    let pn: f64 = x.iter().zip(y).map(|(a, b)| (b - a).powi(2)).sum();
    10.0 * (ps / pn).log10()
}
let one_s = Duration::from_secs(1);
let fs = sample_rate!(48000);

// 1. b >= (90 - 1.76) / 6.02 = 14.66, so 15 bits.
let b = ((90.0 - 1.76) / 6.02f64).ceil() as u32;
assert_eq!(b, 15);
let x = sine_wave::<f64>(997.0, one_s, fs, 1.0);
let xs = x.as_slice().unwrap();
let at = |b: u32| sqnr(xs, &xs.iter().map(|&v| requantise(v, b)).collect::<Vec<_>>());
assert!(at(15) >= 90.0 && at(14) < 90.0);

// 2. 98.09 - 20 = 78.09 dB.
let quiet = sine_wave::<f64>(997.0, one_s, fs, 0.1);
let back = quiet.to_format::<i16>().to_format::<f64>();
let err: Vec<f64> = back.as_slice().unwrap().iter()
    .zip(quiet.as_slice().unwrap()).map(|(q, v)| q - v).collect();
let err = AudioSamples::new_mono(ndarray::Array1::from(err), fs).unwrap();
let measured = snr(&quiet, &err).unwrap();
assert!((measured - 78.09).abs() < 0.1);
// Its levels: -20 dBFS peak, -23.01 dBFS RMS (whole periods: 997 in 1 s).
let q = quiet.as_slice().unwrap();
let peak_db = 20.0 * q.iter().fold(0.0f64, |a, v| a.max(v.abs())).log10();
let rms_db = 10.0 * (q.iter().map(|v| v * v).sum::<f64>() / q.len() as f64).log10();
assert!((peak_db + 20.0).abs() < 0.01 && (rms_db + 23.01).abs() < 0.01);

// 3. 10 log10(3) = 4.77 dB. One step at 8 bits is 1/128.
let lsb = 1.0 / 128.0;
let loud = sine_wave::<f64>(997.0, one_s, fs, 0.9);
let ls = loud.as_slice().unwrap();
let u1 = white_noise::<f64>(one_s, fs, lsb / 2.0, Some(11));
let u2 = white_noise::<f64>(one_s, fs, lsb / 2.0, Some(12));
let plain: Vec<f64> = ls.iter().map(|&v| requantise(v, 8)).collect();
let dith: Vec<f64> = ls.iter().zip(u1.as_slice().unwrap()).zip(u2.as_slice().unwrap())
    .map(|((&v, a), b)| requantise(v + a + b, 8)).collect();
let penalty = sqnr(ls, &plain) - sqnr(ls, &dith);
// Random dither over 48 000 samples: the noise power estimate is good to ~1 %.
assert!((penalty - 10.0 * 3f64.log10()).abs() < 0.2);
```

## Common pitfalls

* The crate maps integers to floats **asymmetrically** (divide by
  $2^{b-1}$ below zero, by $2^{b-1}-1$ above). Many other audio libraries
  divide by $2^{b-1}$ on both sides, so their positive samples differ from
  this crate's by up to about $3\times10^{-5}$.
* Signed integer-to-integer narrowing (for example `i32` to `i16`) *shifts
  bits*: it truncates towards $-\infty$ (a $-\tfrac12$ LSB bias) rather than
  rounding. Conversions to `u8` are the exception: they rescale and round to
  nearest. Convert through `f64` to round in every case.
* No conversion applies dither; add it before reducing the bit depth of quiet
  material.
* $6.02\,b + 1.76$ dB is for a **full-scale sine**. Speech peaks well below
  full scale and has a high crest factor; its SQNR is much lower.
* `white_noise` is *uniform*, not Gaussian: amplitude $A$ gives RMS
  $A/\sqrt{3}$.

## References

* B. Widrow and I. Kollár, *Quantization Noise: Roundoff Error in Digital Computation, Signal Processing, Control, and Communications*, Cambridge University Press, 2008.
* S. P. Lipshitz, R. A. Wannamaker and J. Vanderkooy, "Quantization and dither: a theoretical survey", *Journal of the Audio Engineering Society* 40(5), 355-375, 1992.
* AES17, *AES standard method for digital audio engineering: measurement of digital audio equipment*, Audio Engineering Society (the definition of dBFS as the RMS of a full-scale sine).

## See also

* [`topics::quantisation`](crate::educational::topics::quantisation)
* [`AudioTypeConversion::to_format`](crate::AudioTypeConversion::to_format),
  the [`conversions`](crate::conversions) module (scaling convention)
* [`comparison::snr`](crate::utils::comparison::snr),
  [`white_noise`](crate::utils::generation::white_noise)
* Previous: [01. Sampling and aliasing](01_sampling_aliasing.md).
  Next: [03. The DFT, windows and leakage](03_dft_windows.md).
* Lab: `examples/tutorial_02_quantisation_dither.rs`
  (`cargo run --release --example tutorial_02_quantisation_dither --features educational,full_no_plotting`).
