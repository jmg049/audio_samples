# 03. The DFT, windows and leakage

The discrete Fourier transform (DFT) converts a block of samples into
amplitudes and phases at evenly spaced frequencies. Most of the later chapters
of this guide (spectrograms, mel features, pitch, filtering by FFT) build on it.
This chapter covers exactly what it returns, why a pure tone often spreads over
many bins (leakage), and what a window can and cannot do about that.

## Concept

**Definition.** For a block of $N$ samples $x[n]$, $n = 0, \dots, N-1$:

$$X[k] = \sum_{n=0}^{N-1} x[n] \, e^{-j 2\pi k n / N}$$

where $k$ is the integer bin index, $j = \sqrt{-1}$ and $X[k]$ is a complex
number in the units of $x$ (here: full scale) times samples. The exponent is
**negative** and there is **no** $1/N$ factor (the same as `numpy.fft.fft`).

**Bin frequencies.** Bin $k$ is at

$$f_k = \frac{k f_s}{N}, \qquad \Delta f = \frac{f_s}{N}$$

where $f_s$ is the sample rate in Hz and $\Delta f$ the bin spacing in Hz. For
real $x$, $X[N-k] = \overline{X[k]}$, so only bins $0 \dots \lfloor N/2\rfloor$
carry information; this crate returns exactly those $\lfloor N/2\rfloor + 1$.

**A tone on a bin.** For $x[n] = A\cos(2\pi k_0 n/N)$ with $0 < k_0 < N/2$,
$X[k_0] = A N/2$ (real; at $k_0 = 0$ or $N/2$ it is $AN$);
for $A\sin(\cdot)$, $X[k_0] = -jAN/2$. With a window $w[n]$ applied first,
the peak becomes $A \sum_n w[n] / 2$.

**Leakage.** The DFT treats the block as one period of a periodic signal. If
the tone does not complete a whole number of cycles in the block, the implied
periodic signal has a jump at the block edge, and the energy spreads over all
bins. Multiplying by a window that tapers to zero removes the jump, at the cost
of a wider main lobe. A window $w[n]$ is summarised by:

$$\mathrm{CG} = \frac{1}{N}\sum_n w[n], \qquad \mathrm{ENBW} = \frac{N \sum_n w^2[n]}{\left(\sum_n w[n]\right)^2}$$

where the coherent gain CG (dimensionless) scales the peak of a bin-centred
tone relative to no window, and the equivalent noise bandwidth ENBW (in bins)
is the bandwidth of broadband noise each bin collects. Two more figures are read off
the window's spectrum $W(f)$ (with $f$ in bins): the **scalloping loss**
$-20\log_{10}|W(1/2)/W(0)|$ in dB, the worst-case peak drop for a tone halfway
between bins, and the **peak side-lobe level** (PSL), the highest lobe outside
the main lobe in dB relative to $W(0)$.

This crate's `WindowType::Hanning` is the **symmetric** Hann window

$$w[n] = 0.5 - 0.5\cos\!\left(\frac{2\pi n}{N-1}\right)$$

(denominator $N-1$, so $w[0] = w[N-1] = 0$), not the periodic one (denominator
$N$) that SciPy uses by default for spectra.

**Zero padding** (appending zeros before the DFT) samples the same spectrum on
a finer grid. It does **not** narrow the main lobe; only a longer stretch of
signal does.

## With the library

[`fft`](crate::AudioTransforms::fft) returns the complex non-negative-frequency
bins, [`rfft`](crate::AudioTransforms::rfft) their **magnitudes** (numpy's
`rfft` returns complex values):

```rust
use audio_samples::{AudioSamples, AudioTransforms, nzu, sample_rate};
use num_complex::Complex;
use std::f64::consts::PI;

// 64 samples: 0.25 (DC) + sin at bin 4 + 0.5 cos at bin 9.
let x: Vec<f64> = (0..64).map(|i| {
    let t = i as f64 / 64.0;
    0.25 + (2.0 * PI * 4.0 * t).sin() + 0.5 * (2.0 * PI * 9.0 * t).cos()
}).collect();
let audio = AudioSamples::new_mono(ndarray::Array1::from(x), sample_rate!(16000)).unwrap();

let spec = audio.fft(nzu!(64)).unwrap();
assert_eq!(spec.ncols(), 33); // N/2 + 1
let close = |a: Complex<f64>, b: Complex<f64>| (a - b).norm() < 1e-9;
assert!(close(spec[[0, 0]], Complex::new(16.0, 0.0)));   // N x 0.25
assert!(close(spec[[0, 4]], Complex::new(0.0, -32.0)));  // -j A N / 2: negative exponent
assert!(close(spec[[0, 9]], Complex::new(16.0, 0.0)));   // 0.5 N / 2

let mags = audio.rfft(nzu!(64)).unwrap();
assert!((mags[[0, 4]] - 32.0).abs() < 1e-9); // a real number: |X[4]|
```

Measuring leakage: at $f_s = 16$ kHz and $N = 512$, $\Delta f = 31.25$ Hz, so
1000 Hz is bin 32 and 1015.625 Hz is bin 32.5, the worst case. Window the tone
with [`apply_window_f64`](crate::AudioProcessing::apply_window_f64) and the
coefficients from `spectrograms::make_window`:

```rust
use audio_samples::{AudioProcessing, AudioTransforms, nzu, sample_rate, sine_wave};
use spectrograms::{WindowType, make_window};
use std::time::Duration;

let spectrum = |f: f64, wt: WindowType| -> Vec<f64> {
    let w = make_window::<f64>(wt, nzu!(512));
    sine_wave::<f64>(f, Duration::from_millis(32), sample_rate!(16000), 1.0) // 512 samples
        .apply_window_f64(&w).unwrap()
        .rfft(nzu!(512)).unwrap().row(0).to_vec()
};
let db = |r: f64| 20.0 * r.log10();
let peak = |s: &[f64]| s.iter().copied().fold(0.0, f64::max);

// Rectangular (no window): the peak drops by 3.86 dB off-bin, and 19.5 bins
// away the leakage is 34 dB down.
let (on, off) = (spectrum(1000.0, WindowType::Rectangular), spectrum(1015.625, WindowType::Rectangular));
assert!((db(peak(&on) / peak(&off)) - 3.86).abs() < 0.01);
assert!((db(off[52] / peak(&off)) + 34.0).abs() < 0.1);

// Hann: the peak drops by 1.42 dB, and 19.5 bins away the leakage is 86 dB down.
let (on, off) = (spectrum(1000.0, WindowType::Hanning), spectrum(1015.625, WindowType::Hanning));
assert!((db(peak(&on) / peak(&off)) - 1.42).abs() < 0.01);
assert!((db(off[52] / peak(&off)) + 86.0).abs() < 0.1);
```

(The rectangular drop of 3.86 dB is slightly less than the window's 3.92 dB
scalloping loss because a real sine also has a negative-frequency image whose
leakage adds to the bins near the peak.)

<!-- figure: ch03-leakage -->

Zero padding versus a longer signal: two tones 20 Hz apart, Hann window.

```rust
use audio_samples::{AudioProcessing, AudioSamples, AudioTransforms, sample_rate, sine_wave};
use spectrograms::{WindowType, make_window};
use std::num::NonZeroUsize;
use std::time::Duration;

let peaks_near_1khz = |dur: Duration, n_fft: usize| -> usize {
    let a = sine_wave::<f64>(1000.0, dur, sample_rate!(16000), 1.0);
    let b = sine_wave::<f64>(1020.0, dur, sample_rate!(16000), 1.0);
    let v: Vec<f64> = a.as_slice().unwrap().iter().zip(b.as_slice().unwrap()).map(|(p, q)| p + q).collect();
    let w = make_window::<f64>(WindowType::Hanning, NonZeroUsize::new(v.len()).unwrap());
    let m = AudioSamples::new_mono(ndarray::Array1::from(v), sample_rate!(16000)).unwrap()
        .apply_window_f64(&w).unwrap()
        .rfft(NonZeroUsize::new(n_fft).unwrap()).unwrap().row(0).to_vec();
    let bin = 16_000.0 / n_fft as f64;
    let (lo, hi) = ((900.0 / bin) as usize, (1120.0 / bin) as usize);
    let top = m[lo..hi].iter().copied().fold(0.0, f64::max);
    (lo..hi).filter(|&i| m[i] > m[i - 1] && m[i] >= m[i + 1] && m[i] > 0.5 * top).count()
};
assert_eq!(peaks_near_1khz(Duration::from_millis(32), 512), 1);   // 512 samples
assert_eq!(peaks_near_1khz(Duration::from_millis(32), 8192), 1);  // zero-padded: still 1
assert_eq!(peaks_near_1khz(Duration::from_millis(256), 8192), 2); // 4096 samples: resolved
```

<!-- figure: ch03-zero-padding -->

## How it works

The code below re-derives three results: the DFT itself, the symmetric-Hann figures in closed
form, and the rectangular window's scalloping loss.

```rust
use audio_samples::{AudioSamples, AudioTransforms, nzu, sample_rate};
use audio_samples::educational::topics;
use num_complex::Complex;
use spectrograms::{WindowType, make_window};
use std::f64::consts::PI;

// 1. The DFT as a double loop, against the library's FFT.
let x: Vec<f64> = (0..64).map(|i| ((i * 7 % 13) as f64 - 6.0) / 6.0).collect();
let naive: Vec<Complex<f64>> = (0..=32).map(|k| {
    x.iter().enumerate()
        .map(|(n, &v)| Complex::from_polar(v, -2.0 * PI * (k * n) as f64 / 64.0))
        .sum()
}).collect();
let lib = AudioSamples::new_mono(ndarray::Array1::from(x), sample_rate!(16000)).unwrap()
    .fft(nzu!(64)).unwrap();
// 64-term sums of O(1) values: rounding of order 1e-14.
for k in 0..=32 {
    assert!((lib[[0, k]] - naive[k]).norm() < 1e-12);
}

// 2. Symmetric Hann: sum w = (N-1)/2 and sum w^2 = 3(N-1)/8, so
//    CG = (N-1)/(2N) and ENBW = 1.5 N/(N-1). For N = 512: 0.4990 and 1.5029.
let n = 512.0;
let w = make_window::<f64>(WindowType::Hanning, nzu!(512));
let cg = w.iter().sum::<f64>() / n;
let enbw = n * w.iter().map(|v| v * v).sum::<f64>() / w.iter().sum::<f64>().powi(2);
assert!((cg - (n - 1.0) / (2.0 * n)).abs() < 1e-12);
assert!((enbw - 1.5 * n / (n - 1.0)).abs() < 1e-12);
assert!((cg - 0.4990).abs() < 1e-4 && (enbw - 1.5029).abs() < 1e-4);
assert_eq!(w[0], 0.0);                       // symmetric: zero at both ends,
assert!(w[511].abs() < 1e-15);
assert!((w[256] - 0.999991).abs() < 1e-6);   // and no sample exactly at 1.0

// The library's explainer measures the same numbers from the coefficients.
let doc = topics::window(&WindowType::Hanning, 512, 16_000.0);
assert!((doc.quantity("Coherent gain").unwrap().value - cg).abs() < 1e-12);
assert!((doc.quantity("Equivalent noise bandwidth").unwrap().value - enbw).abs() < 1e-12);

// 3. Rectangular window: |W(f)| = |sin(pi f) / sin(pi f / N)|, so half a bin
//    off, |W(1/2)/W(0)| = 1 / (N sin(pi / 2N)), i.e. 3.92 dB.
let ratio: f64 = 1.0 / (n * (PI / (2.0 * n)).sin());
let loss = -20.0 * ratio.log10();
assert!((loss - 3.92).abs() < 0.01);
let rect = topics::window(&WindowType::Rectangular, 512, 16_000.0);
assert!((rect.quantity("Scalloping loss").unwrap().value - loss).abs() < 0.01);
```

## Ask the library to explain

[`topics::dft`](crate::educational::topics::dft) explains bin spacing and the
mirror symmetry; [`topics::window`](crate::educational::topics::window) and
`WindowType::explain_with` measure the window figures from the actual
coefficients the STFT uses:

```rust
use audio_samples::educational::{Explain, ExplainContext, topics};
use spectrograms::WindowType;

let dft = topics::dft(512, 16_000.0);
assert_eq!(dft.quantity("Bin spacing").unwrap().value, 31.25);
assert_eq!(dft.quantity("Non-redundant bins").unwrap().value, 257.0);

// (window, coherent gain, ENBW in bins, peak side-lobe level in dB), N = 512
for (wt, cg, enbw, psl) in [
    (WindowType::Rectangular, 1.0, 1.0, -13.26),
    (WindowType::Hanning, 0.4990, 1.5029, -31.47),
    (WindowType::Hamming, 0.5391, 1.3647, -42.67),
    (WindowType::Blackman, 0.4192, 1.7301, -58.11),
] {
    let doc = wt.explain_with(&ExplainContext::new().with_n_fft(512).with_sample_rate(16_000.0));
    assert!((doc.quantity("Coherent gain").unwrap().value - cg).abs() < 1e-4);
    assert!((doc.quantity("Equivalent noise bandwidth").unwrap().value - enbw).abs() < 1e-4);
    assert!((doc.quantity("Peak side-lobe level").unwrap().value - psl).abs() < 0.01);
}
let hann = WindowType::Hanning.explain_with(&ExplainContext::new().with_n_fft(512));
assert!(hann.has_warning_containing("symmetric"));
println!("{hann}");
```

The table shows the standard trade-off: lower side lobes (less leakage from
loud components into quiet ones) come with a wider main lobe and more noise per
bin.

<!-- figure: ch03-windows -->

## Exercises

1. At $f_s = 44.1$ kHz with $N = 4096$, which bin is nearest to 440 Hz, and
   what is its centre frequency? Check that a Hann-windowed 440 Hz tone peaks
   there.
2. A sine of amplitude 0.5 lies exactly on a bin, Hann window, $N = 1024$.
   Predict $|X[k]|$ at the peak, then measure.
3. With a Hann window (main lobe about 2 bins wide at $-6$ dB), what is the
   smallest $N$ that should separate two tones 10 Hz apart at 16 kHz? Check
   that a power of two above it separates 1000 Hz and 1010 Hz, and that
   $N = 1024$ does not.

## Answers

```rust
use audio_samples::{AudioProcessing, AudioSamples, AudioTransforms, sample_rate, sine_wave};
use spectrograms::{WindowType, make_window};
use std::num::NonZeroUsize;
use std::time::Duration;

let nz = |n: usize| NonZeroUsize::new(n).unwrap();
let hann_mag = |v: Vec<f64>, fs: u32, n_fft: usize| -> Vec<f64> {
    let w = make_window::<f64>(WindowType::Hanning, nz(v.len()));
    AudioSamples::new_mono(ndarray::Array1::from(v), std::num::NonZeroU32::new(fs).unwrap()).unwrap()
        .apply_window_f64(&w).unwrap().rfft(nz(n_fft)).unwrap().row(0).to_vec()
};
let argmax = |m: &[f64]| (0..m.len()).max_by(|&a, &b| m[a].total_cmp(&m[b])).unwrap();

// 1. k = round(440 * 4096 / 44100) = round(40.87) = 41, at 41 * 44100 / 4096 = 441.43 Hz.
let k = (440.0f64 * 4096.0 / 44_100.0).round() as usize;
assert_eq!(k, 41);
assert!((k as f64 * 44_100.0 / 4096.0 - 441.43).abs() < 0.01);
let tone: Vec<f64> = (0..4096).map(|n| (2.0 * std::f64::consts::PI * 440.0 * n as f64 / 44_100.0).sin()).collect();
assert_eq!(argmax(&hann_mag(tone, 44_100, 4096)), 41);

// 2. A * sum(w) / 2 = 0.5 * 511.5 / 2 = 127.875 (sum of a symmetric Hann is (N-1)/2).
let x = sine_wave::<f64>(1000.0, Duration::from_millis(64), sample_rate!(16000), 0.5); // bin 64
let m = hann_mag(x.as_slice().unwrap().to_vec(), 16_000, 1024);
// The tone's negative-frequency image leaks a little into bin 64: allow 1e-3.
assert!((m[64] - 127.875).abs() < 1e-3);

// 3. -6 dB main-lobe width, about 2 fs / N for Hann, <= 10 Hz  =>  N >= 3200;
//    the next power of two is 4096.
let n_min = 2.0 * 16_000.0 / 10.0;
assert_eq!(n_min, 3200.0);
let count = |n: usize| -> usize {
    let v: Vec<f64> = (0..n).map(|i| {
        let t = i as f64 / 16_000.0;
        (2.0 * std::f64::consts::PI * 1000.0 * t).sin() + (2.0 * std::f64::consts::PI * 1010.0 * t).sin()
    }).collect();
    let m = hann_mag(v, 16_000, n);
    let bin = 16_000.0 / n as f64;
    let (lo, hi) = ((950.0 / bin) as usize, (1060.0 / bin) as usize);
    let top = m[lo..hi].iter().copied().fold(0.0, f64::max);
    (lo..hi).filter(|&i| m[i] > m[i - 1] && m[i] >= m[i + 1] && m[i] > 0.5 * top).count()
};
assert_eq!(count(4096), 2);
assert_eq!(count(1024), 1);
```

## Common pitfalls

* [`rfft`](crate::AudioTransforms::rfft) returns **magnitudes**, not numpy's
  complex `rfft`; use [`fft`](crate::AudioTransforms::fft) for phase.
* Neither transform normalises: a full-scale bin-centred sine gives $N/2$, not 1
  or 0.5. Divide by $\sum w / 2$ to read amplitudes.
* `n_fft` must be **at least** the signal length (shorter signals are
  zero-padded; longer ones are an error, not truncated).
* `WindowType::Hanning` is the **symmetric** Hann; SciPy's
  `get_window('hann', N)` uses the periodic one. Values differ by a fraction
  of a percent; use `WindowType::Custom` to match it exactly.
* Zero padding interpolates the spectrum onto a finer grid; it does not improve
  resolution.
* The Kaiser window in `spectrograms` 2.1.3 is wrong for $\beta > 3.75$; avoid
  it for teaching material.

## References

* F. J. Harris, "On the use of windows for harmonic analysis with the discrete Fourier transform", *Proceedings of the IEEE* 66(1), 51-83, 1978. The source of the window figures (coherent gain, ENBW, scalloping loss, side-lobe level) used here.
* J. W. Cooley and J. W. Tukey, "An algorithm for the machine calculation of complex Fourier series", *Mathematics of Computation* 19(90), 297-301, 1965.
* A. V. Oppenheim and R. W. Schafer, *Discrete-Time Signal Processing*, 3rd ed., Pearson, 2010.

## See also

* [`AudioTransforms::fft`](crate::AudioTransforms::fft),
  [`AudioTransforms::rfft`](crate::AudioTransforms::rfft),
  [`AudioProcessing::apply_window_f64`](crate::AudioProcessing::apply_window_f64),
  [`fft_frequencies`](crate::utils::audio_math::fft_frequencies)
* [`topics::dft`](crate::educational::topics::dft),
  [`topics::window`](crate::educational::topics::window)
* The [`transforms`](crate::operations::transforms) module docs (conventions)
* Previous: [02. Quantisation and dither](02_quantisation_dither.md).
  Next: [04. STFT and spectrograms](04_stft_spectrograms.md).
* Lab: `examples/tutorial_03_dft_windows.rs`
  (`cargo run --release --example tutorial_03_dft_windows --features educational,full_no_plotting`).
