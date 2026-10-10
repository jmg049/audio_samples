# 05. Convolution and FIR filters

A finite impulse response (FIR) filter replaces each sample by a weighted sum
of the most recent input samples. That one operation, convolution, smooths
noisy envelopes, removes hum, band-limits a signal before downsampling,
simulates a room from its impulse response, and implements the pre-emphasis
step of every speech front end. This chapter defines convolution precisely,
shows the two output lengths the crate returns, and connects a filter's taps to
its frequency response and delay, ending with a low-pass filter designed from
scratch.

## Concept

**Convolution.** For an input $x\[n\]$ of $N$ samples and taps $h\[m\]$,
$m = 0 \dots M-1$:

$$y\[n\] = \sum_{m=0}^{M-1} h\[m\] \\, x\[n - m\]$$

where $n$ is the output sample index, $m$ the tap index, $M$ the number of taps
and $x\[\cdot\] = 0$
outside the signal. The **full** convolution has $N + M - 1$ samples
($n = 0 \dots N + M - 2$). The **valid** part, where every term of the sum uses
a real input sample, has $N - M + 1$ samples ($n = M-1 \dots N-1$).

**Impulse response.** Feeding the unit impulse $\delta\[n\]$ (1 at $n = 0$,
0 elsewhere) returns $y\[n\] = h\[n\]$: the taps *are* the impulse response.

**Frequency response.** A complex exponential at angular frequency
$\omega = 2\pi f / f_s$ (radians per sample, $f$ in Hz, $f_s$ the sample rate
in Hz) comes out multiplied by

$$H(e^{j\omega}) = \sum_{m=0}^{M-1} h\[m\] \\, e^{-j\omega m}$$

where $j = \sqrt{-1}$ (as in chapter 03; no formula in this chapter uses $j$ as an
index). So a sine of amplitude $A$ leaves with amplitude $A\\,|H|$, shifted by the phase
$\arg H$. For the $M$-tap moving average $h\[m\] = 1/M$ this sums to the
**Dirichlet kernel**

$$|H(f)| = \left| \frac{\sin(\pi f M / f_s)}{M \sin(\pi f / f_s)} \right|$$

with nulls at every multiple of $f_s / M$ Hz (except 0).

**Linear phase and group delay.** The group delay

$$\tau_g(\omega) = -\frac{d \arg H(e^{j\omega})}{d\omega} = \mathrm{Re}\left\\{ \frac{\sum_m m\\,h\[m\]\\,e^{-j\omega m}}{H(e^{j\omega})} \right\\}$$

is the delay in samples of a narrow-band signal at $\omega$. If the taps are
symmetric, $h\[m\] = h\[M-1-m\]$, the phase is exactly $-\omega (M-1)/2$ (plus
jumps of $\pi$ where $H$ changes sign), so **every frequency is delayed by the
same $(M-1)/2$ samples** and waveform shapes are preserved. Antisymmetric
taps, $h\[m\] = -h\[M-1-m\]$, give the same constant delay with an extra
$90°$ phase shift. Other asymmetric taps delay different frequencies by
different amounts.

The four **linear-phase types** combine the symmetry with the parity of $M$.
Each symmetry forces zeros of $H(z)$ at $z = 1$ (DC) or $z = -1$ (Nyquist),
which rules out some filter shapes:

| Type | Taps | Length $M$ | Forced zeros | Cannot be |
|---|---|---|---|---|
| I | symmetric | odd | none | (any shape is possible) |
| II | symmetric | even | $z = -1$ (Nyquist) | high-pass, band-stop |
| III | antisymmetric | odd | $z = 1$ and $z = -1$ | low-pass, high-pass |
| IV | antisymmetric | even | $z = 1$ (DC) | low-pass |

Types III and IV suit differentiators and Hilbert transformers.

**Windowed-sinc design.** The ideal low-pass with cutoff $f_c$ Hz has the
infinitely long impulse response
$h_{\mathrm{ideal}}\[n\] = 2 (f_c/f_s)\\, \mathrm{sinc}\big(2 (f_c / f_s)\\, n\big)$,
with $\mathrm{sinc}(u) = \sin(\pi u)/(\pi u)$. Truncate it to $M$ taps centred
on $(M-1)/2$, multiply by a window $w\[n\]$ (Hamming here) to tame the ripples,
and scale so that $\sum_m h\[m\] = 1$ (unit gain at DC):

$$h\[n\] = c \\; 2\frac{f_c}{f_s}\\, \mathrm{sinc}\\!\left(2 \frac{f_c}{f_s}\left(n - \frac{M-1}{2}\right)\right) w\[n\]$$

where $c$ is the normalising constant. The response passes through about 0.5
($-6$ dB) at $f_c$, and the transition band narrows as $M$ grows. A Hamming
design gives a stop-band attenuation of about 53 dB whatever $M$ is (the lab
finds its first stop-band lobe at $-52.2$ dB); do not confuse this with the
Hamming *window's* own peak side lobe, $-43$ dB (chapter 03).

## With the library

[`AudioTransforms::convolve`](crate::AudioTransforms::convolve) returns the
full convolution; [`AudioProcessing::apply_filter`](crate::AudioProcessing::apply_filter)
returns the valid part (what `numpy.convolve(x, h, 'valid')` gives).

```rust
use audio_samples::educational::topics;
use audio_samples::{AudioProcessing, AudioSamples, AudioTransforms, sample_rate};
use non_empty_slice::NonEmptySlice;

let mono = |v: Vec<f64>| AudioSamples::new_mono(ndarray::Array1::from(v), sample_rate!(16000)).unwrap();
let x: Vec<f64> = (0..1000).map(|n| (0.05 * n as f64).sin()).collect();
let h = vec![1.0 / 31.0; 31]; // 31-tap moving average

let full = mono(x.clone()).convolve(&mono(h.clone())).unwrap();
let valid = mono(x.clone()).apply_filter(NonEmptySlice::new(&h).unwrap()).unwrap();
assert_eq!(full.samples_per_channel().get(), 1030);  // N + M - 1
assert_eq!(valid.samples_per_channel().get(), 970);  // N - M + 1
// The valid part is the full convolution from index M - 1 to N - 1.
let (f, v) = (full.as_slice().unwrap(), valid.as_slice().unwrap());
assert!(f[30..1000].iter().zip(v).all(|(a, b)| (a - b).abs() < 1e-12));

// An impulse in gives the taps out.
let mut delta = vec![0.0; 64];
delta[0] = 1.0;
let ir = mono(delta).convolve(&mono(h.clone())).unwrap();
assert!(ir.as_slice().unwrap()[..31].iter().zip(&h).all(|(a, b)| (a - b).abs() < 1e-12));

// The explainer classifies the taps and reports their delay.
let doc = topics::fir(&h, 16_000.0);
assert_eq!(doc.quantity("Group delay").unwrap().value, 15.0); // (31 - 1) / 2
assert!((doc.quantity("DC gain").unwrap().value - 1.0).abs() < 1e-12);
```

The moving-average response, measured by filtering sines. One second of output
($f_s$ samples) holds a whole number of cycles of each test tone, so projecting
the output on $\sin$ and $\cos$ recovers its amplitude exactly:

```rust
use audio_samples::{AudioProcessing, AudioSamples};
use non_empty_slice::NonEmptySlice;
use std::f64::consts::PI;

fn measured_gain(h: &[f64], f: f64, fs: f64) -> f64 {
    let len = fs as usize + h.len() - 1; // so the valid output is exactly fs samples
    let x: Vec<f64> = (0..len).map(|n| (2.0 * PI * f * n as f64 / fs).sin()).collect();
    let sr = std::num::NonZeroU32::new(fs as u32).unwrap();
    let y = AudioSamples::new_mono(ndarray::Array1::from(x), sr).unwrap()
        .apply_filter(NonEmptySlice::new(h).unwrap()).unwrap();
    let (mut s, mut c) = (0.0, 0.0);
    for (i, &v) in y.as_slice().unwrap().iter().enumerate() {
        let phi = 2.0 * PI * f * i as f64 / fs;
        s += v * phi.sin();
        c += v * phi.cos();
    }
    2.0 / fs * (s * s + c * c).sqrt()
}

let ma = vec![1.0 / 8.0; 8];
let fs = 8_000.0;
for (f, expected) in [(250.0, 0.901764), (500.0, 0.640729), (750.0, 0.304489), (1500.0, 0.224994)] {
    let u = PI * f / fs;
    let dirichlet = ((8.0 * u).sin() / (8.0 * u.sin())).abs();
    let measured = measured_gain(&ma, f, fs);
    assert!((measured - dirichlet).abs() < 1e-9); // exact up to rounding
    assert!((measured - expected).abs() < 1e-6);
}
// Nulls at multiples of fs / M = 1000 Hz.
for f in [1000.0, 2000.0, 3000.0] {
    assert!(measured_gain(&ma, f, fs) < 1e-9);
}
```

## Under the hood

The convolution sum, a group-delay calculation, and the delay seen in the time
domain, all for a 63-tap windowed-sinc low-pass ($f_c = 2$ kHz,
$f_s = 16$ kHz) designed here, since the crate has no FIR designer:

```rust
use audio_samples::{AudioProcessing, AudioSamples, sample_rate};
use non_empty_slice::NonEmptySlice;
use num_complex::Complex;
use spectrograms::{WindowType, make_window};
use std::f64::consts::PI;
use std::num::NonZeroUsize;

fn windowed_sinc(m: usize, fc: f64, fs: f64) -> Vec<f64> {
    let w = make_window::<f64>(WindowType::Hamming, NonZeroUsize::new(m).unwrap());
    let wc = 2.0 * fc / fs;
    let centre = (m - 1) as f64 / 2.0;
    let mut h: Vec<f64> = (0..m).map(|n| {
        let t = n as f64 - centre;
        let sinc = if t == 0.0 { 1.0 } else { (PI * wc * t).sin() / (PI * wc * t) };
        wc * sinc * w[n]
    }).collect();
    let dc: f64 = h.iter().sum();
    h.iter_mut().for_each(|v| *v /= dc);
    h
}
fn convolve_full(x: &[f64], h: &[f64]) -> Vec<f64> {
    let mut y = vec![0.0; x.len() + h.len() - 1];
    for (n, yn) in y.iter_mut().enumerate() {
        for (j, &hj) in h.iter().enumerate() {
            if n >= j && n - j < x.len() { *yn += hj * x[n - j]; }
        }
    }
    y
}
fn group_delay(h: &[f64], omega: f64) -> f64 {
    let num: Complex<f64> = h.iter().enumerate()
        .map(|(j, &c)| Complex::from_polar(j as f64 * c, -omega * j as f64)).sum();
    let den: Complex<f64> = h.iter().enumerate()
        .map(|(j, &c)| Complex::from_polar(c, -omega * j as f64)).sum();
    (num / den).re
}

let lp = windowed_sinc(63, 2000.0, 16_000.0);
assert!((0..31).all(|j| (lp[j] - lp[62 - j]).abs() < 1e-15)); // symmetric (to rounding)

// Direct sum versus apply_filter: the same numbers up to the last bits of
// rounding (below 160 taps apply_filter also sums directly, in another order).
let x: Vec<f64> = (0..1000).map(|n| (0.05 * n as f64).sin() + 0.3 * (0.71 * n as f64).cos()).collect();
let full = convolve_full(&x, &lp);
let y = AudioSamples::new_mono(ndarray::Array1::from(x), sample_rate!(16000)).unwrap()
    .apply_filter(NonEmptySlice::new(&lp).unwrap()).unwrap();
assert!(full[62..1000].iter().zip(y.as_slice().unwrap()).all(|(a, b)| (a - b).abs() < 1e-12));

// Linear phase: 31 samples of delay at every passband frequency ...
for f in [100.0, 500.0, 1000.0, 1500.0] {
    assert!((group_delay(&lp, 2.0 * PI * f / 16_000.0) - 31.0).abs() < 1e-9);
}
// ... but not for asymmetric taps: 0.7 samples at DC, 0.25 at Nyquist.
let skew = [0.5, 0.3, 0.2];
assert!((group_delay(&skew, 0.0) - 0.7).abs() < 1e-12);
assert!((group_delay(&skew, PI) - 0.25).abs() < 1e-12);

// Measured with a delayed impulse: impulse at n = 100 -> peak at 100 + 31.
let mut d = vec![0.0; 400];
d[100] = 1.0;
let resp = convolve_full(&d, &lp);
let peak = (0..resp.len()).max_by(|&a, &b| resp[a].total_cmp(&resp[b])).unwrap();
assert_eq!(peak, 131);
// apply_filter's output index i is time i + M - 1, so the same peak is at 131 - 62.
let yv = AudioSamples::new_mono(ndarray::Array1::from(d), sample_rate!(16000)).unwrap()
    .apply_filter(NonEmptySlice::new(&lp).unwrap()).unwrap();
let yv = yv.as_slice().unwrap();
assert_eq!((0..yv.len()).max_by(|&a, &b| yv[a].total_cmp(&yv[b])).unwrap(), 69);

// And with a smooth pulse: its centroid moves by exactly (M - 1)/2 = 31.
let pulse: Vec<f64> = (0..400).map(|n| (-((n as f64 - 150.0) / 12.0).powi(2) / 2.0).exp()).collect();
let centroid = |v: &[f64]| v.iter().enumerate().map(|(n, a)| n as f64 * a).sum::<f64>() / v.iter().sum::<f64>();
assert!((centroid(&convolve_full(&pulse, &lp)) - centroid(&pulse) - 31.0).abs() < 1e-9);

// The magnitude response, evaluated from the taps (the DTFT).
let gain = |f: f64| -> f64 {
    let om = 2.0 * PI * f / 16_000.0;
    lp.iter().enumerate().map(|(j, &c)| Complex::from_polar(c, -om * j as f64)).sum::<Complex<f64>>().norm()
};
let db = |g: f64| 20.0 * g.log10();
for f in [500.0, 1000.0, 1500.0] {
    assert!(db(gain(f)).abs() < 0.05);      // pass band flat to 0.05 dB (max 0.03 dB)
}
assert!((gain(2000.0) - 0.5005).abs() < 1e-4); // -6.01 dB at the cutoff
for f in [3000.0, 4000.0, 6000.0] {
    assert!(db(gain(f)) < -59.0);           // stop band below -59 dB from 3 kHz
}
```

Long filters take a different path: from 160 taps up, `apply_filter` convolves
by FFT. The lab checks that a 201-tap design agrees with the direct sum to
better than $10^{-12}$.

## Ask the library to explain

[`topics::fir`](crate::educational::topics::fir) classifies the taps
(linear-phase types I-IV), reports the group delay, DC and Nyquist gains and a
small response table, and notes when `apply_filter` will use FFT convolution:

```rust
use audio_samples::educational::topics;

let ma = topics::fir(&[0.2; 5], 16_000.0);
assert_eq!(ma.quantity("Taps").unwrap().value, 5.0);
assert_eq!(ma.quantity("Group delay").unwrap().value, 2.0);                         // samples
assert!((ma.quantity("Group delay (time)").unwrap().value - 0.125).abs() < 1e-12); // ms
assert!(ma.has_warning_containing("type I"));

// Even length and symmetric: type II, forced zero at Nyquist.
let t2 = topics::fir(&[0.25; 4], 16_000.0);
assert!(t2.has_warning_containing("type II"));
assert!(t2.quantity("Nyquist gain").unwrap().value.abs() < 1e-12);

// Asymmetric taps: no constant group delay, only its value at DC.
let skew = topics::fir(&[0.5, 0.3, 0.2], 16_000.0);
assert!(skew.quantity("Group delay").is_none());
assert!((skew.quantity("Group delay at DC").unwrap().value - 0.7).abs() < 1e-12);

// Antisymmetric taps: type III (odd length) and type IV (even length, zero at DC).
assert!(topics::fir(&[1.0, 0.0, -1.0], 16_000.0).has_warning_containing("type III"));
let t4 = topics::fir(&[1.0, -1.0], 16_000.0);
assert!(t4.has_warning_containing("type IV") && t4.quantity("DC gain").unwrap().value == 0.0);

let long = topics::fir(&[1.0 / 200.0; 200], 16_000.0);
assert!(long.has_warning_containing("FFT convolution"));
println!("{ma}");
```

## Exercises

1. Where are the nulls of a 5-tap moving average at $f_s = 10$ kHz? Check two
   of them by filtering sines.
2. You low-pass 1 s of 16 kHz audio with the 63-tap filter above using
   `apply_filter`. How many samples come out, and which input sample is output
   sample $i$ aligned with? Check with a 100 Hz sine.
3. *Spectral inversion*: $h_{hp}\[n\] = \delta\[n - (M-1)/2\] - h_{lp}\[n\]$ turns
   the low-pass into a high-pass. Predict its gain at DC, at the 2 kHz cutoff
   and at 6 kHz, then measure.

## Answers

```rust
use audio_samples::educational::topics;
use audio_samples::{AudioProcessing, AudioSamples};
use non_empty_slice::NonEmptySlice;
use num_complex::Complex;
use spectrograms::{WindowType, make_window};
use std::f64::consts::PI;
use std::num::NonZeroUsize;

fn windowed_sinc(m: usize, fc: f64, fs: f64) -> Vec<f64> {
    let w = make_window::<f64>(WindowType::Hamming, NonZeroUsize::new(m).unwrap());
    let wc = 2.0 * fc / fs;
    let centre = (m - 1) as f64 / 2.0;
    let mut h: Vec<f64> = (0..m).map(|n| {
        let t = n as f64 - centre;
        let sinc = if t == 0.0 { 1.0 } else { (PI * wc * t).sin() / (PI * wc * t) };
        wc * sinc * w[n]
    }).collect();
    let dc: f64 = h.iter().sum();
    h.iter_mut().for_each(|v| *v /= dc);
    h
}
fn filter(x: Vec<f64>, h: &[f64], fs: u32) -> Vec<f64> {
    let sr = std::num::NonZeroU32::new(fs).unwrap();
    AudioSamples::new_mono(ndarray::Array1::from(x), sr).unwrap()
        .apply_filter(NonEmptySlice::new(h).unwrap()).unwrap().as_slice().unwrap().to_vec()
}
let sine = |f: f64, fs: f64, len: usize| -> Vec<f64> {
    (0..len).map(|n| (2.0 * PI * f * n as f64 / fs).sin()).collect()
};
let gain = |h: &[f64], f: f64, fs: f64| -> f64 {
    let om = 2.0 * PI * f / fs;
    h.iter().enumerate().map(|(j, &c)| Complex::from_polar(c, -om * j as f64)).sum::<Complex<f64>>().norm()
};

// 1. Multiples of fs / M = 2000 Hz: 2000 and 4000 Hz (below Nyquist).
let ma = [0.2; 5];
for f in [2000.0, 4000.0] {
    let y = filter(sine(f, 10_000.0, 10_004), &ma, 10_000); // 10 000 outputs
    assert!(y.iter().map(|v| v.abs()).fold(0.0, f64::max) < 1e-12);
}
assert!(gain(&ma, 1000.0, 10_000.0) > 0.5); // but not in between

// 2. 16 000 - 62 = 15 938 samples. Output i is the causal output at time
//    i + 62, which lags the input by 31: so y[i] lines up with x[i + 31].
let lp = windowed_sinc(63, 2000.0, 16_000.0);
let x = sine(100.0, 16_000.0, 16_000);
let y = filter(x.clone(), &lp, 16_000);
assert_eq!(y.len(), 15_938);
// At 100 Hz the gain is 1.0012 (pass-band ripple), so the aligned samples
// agree to about 1.2e-3 of the amplitude.
assert!((gain(&lp, 100.0, 16_000.0) - 1.0012).abs() < 1e-4);
assert!(y.iter().enumerate().all(|(i, v)| (v - x[i + 31]).abs() < 2e-3));

// 3. DC: 1 - 1 = 0. Cutoff: |1 - 0.5| = 0.5 (both terms share the same linear
//    phase, so the magnitudes subtract). 6 kHz: 1 - (almost 0) = 1.
let mut hp: Vec<f64> = lp.iter().map(|v| -v).collect();
hp[31] += 1.0;
assert!(gain(&hp, 0.0, 16_000.0) < 1e-12);
assert!((gain(&hp, 2000.0, 16_000.0) - 0.5).abs() < 0.01);
assert!((gain(&hp, 6000.0, 16_000.0) - 1.0).abs() < 1e-3);
let doc = topics::fir(&hp, 16_000.0);
assert!(doc.has_warning_containing("type I") && doc.quantity("Group delay").unwrap().value == 31.0);
```

## Common pitfalls

* `apply_filter` returns the **valid** part: $M - 1$ samples shorter, and
  output $i$ belongs to input time $i + M - 1$. It is not delay-compensated;
  a linear-phase filter still lags by $(M-1)/2$ on top. `convolve` returns the
  **full** convolution.
* For integer sample types, `apply_filter` interprets the *taps* on the
  normalised scale too (an `i16` tap of 16384 means about 0.5); keep taps in a
  float signal.
* `convolve` needs both signals mono and of the same sample type.
* Even-length symmetric filters (type II) always have a zero at Nyquist and
  cannot be high-pass; spectral inversion needs an odd length.
* A windowed sinc is $-6$ dB (not $-3$ dB) at its design cutoff.
* SciPy's `lfilter(h, 1, x)` returns the full-length causal output; the crate's
  `apply_filter` equals `lfilter(h, 1, x)[M-1:]`. Compare like with like.

## References

* A. V. Oppenheim and R. W. Schafer, *Discrete-Time Signal Processing*, 3rd ed., Pearson, 2010. Linear-phase FIR types and window design.
* F. J. Harris, "On the use of windows for harmonic analysis with the discrete Fourier transform", *Proceedings of the IEEE* 66(1), 51-83, 1978.

## See also

* [`AudioProcessing::apply_filter`](crate::AudioProcessing::apply_filter),
  [`AudioTransforms::convolve`](crate::AudioTransforms::convolve),
  [`impulse`](crate::utils::generation::impulse)
* [`topics::fir`](crate::educational::topics::fir)
* Previous: [04. STFT and spectrograms](crate::guide::c04_stft_spectrograms).
  Next: [06. IIR filters](crate::guide::c06_iir_filters).
* Lab: `examples/tutorial_05_convolution_fir.rs`
  (`cargo run --release --example tutorial_05_convolution_fir --features educational,full_no_plotting`).
