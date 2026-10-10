# 04. STFT and spectrograms

Speech and music change over time: a vowel lasts tens of milliseconds, a
plosive burst a few. One DFT of a whole utterance (chapter 03) mixes all of
that together. The short-time Fourier transform (STFT) takes a DFT of short,
overlapping, windowed frames instead, giving a picture of frequency against
time: the spectrogram. This chapter pins down exactly how the crate frames the
signal, why you cannot have fine time and fine frequency resolution at once,
how the transform is inverted, and what the dB scale of a spectrogram means.

## Concept

**Definition (as implemented).** With frame length (and FFT size) $N$, hop
$H$ and window $w[n]$:

$$X_m[k] = \sum_{n=0}^{N-1} x[mH + n - p] \, w[n] \, e^{-j 2\pi k n / N}$$

where $m$ is the frame index, $k = 0 \dots \lfloor N/2 \rfloor$ the bin index,
$p$ the padding in samples and $x[\cdot] = 0$ outside the signal. With
`centre = true`, $p = \lfloor N/2 \rfloor$, so frame $m$ is *centred* on sample
$mH$; with `centre = false`, $p = 0$ and frame $m$ *starts* at sample $mH$. The
window is the symmetric Hann of chapter 03.

**Number of frames** for a signal of $L$ samples:

$$M = 1 + \left\lfloor \frac{L + 2p - N}{H} \right\rfloor$$

**Resolution.** Bins are $\Delta f = f_s / N$ Hz apart and frames
$\Delta t = H / f_s$ seconds apart, with $f_s$ the sample rate in Hz. But the
real time resolution is set by the window *length* $N/f_s$, and the real
frequency resolution by the window's main lobe (about $2\Delta f$ for Hann at
$-6$ dB). Their product is fixed by the window shape: halving one doubles the
other. A smaller hop gives more frames, not sharper ones.

**Inverse (weighted overlap-add).** Inverse-DFT each frame to $y_m[n]$,
multiply by the window again and overlap-add, dividing by the summed squared
windows:

$$\hat{x}[n] = \frac{\sum_m w[n - mH] \, y_m[n - mH]}{\sum_m w^2[n - mH]}$$

This is exact wherever the denominator is non-zero, for any window and hop,
provided the STFT has not been modified. Two overlap conditions are worth
naming. **NOLA** (nonzero overlap-add): the summed *squared* windows
$\sum_m w^2[n - mH]$ are never zero, which is all the inverse above needs.
**COLA** (constant overlap-add): the plain summed windows $\sum_m w[n - mH]$
are the same constant for every $n$, which plain overlap-add without the
division needs. WOLA works even for windows that are not COLA. If the STFT is
modified before inversion (masking, a phase vocoder), the frames are no longer
consistent and the same formula gives the least-squares estimate of Griffin
and Lim (1984) rather than an exact inverse. With centring, the output has
$(M-1)H + N - 2\lfloor N/2\rfloor$ samples.

**Spectrogram scales.** Power $P_m[k] = |X_m[k]|^2$ (no division by $N$),
magnitude $|X_m[k]|$, and decibels

$$P_{\mathrm{dB}} = 10 \log_{10} \max\!\left(P, \; 10^{\,\mathrm{floor}/10}\right)$$

in dB **re 1** (an absolute reference), where floor is the `LogParams` value in
dB. `LogParams::new(-80.0)` clamps at power $10^{-8}$; it is **not** librosa's
`top_db` (80 dB below the peak), and a loud bin can be far above 0 dB. A
log-frequency spectrogram resamples the bins onto log-spaced frequencies; a
mel spectrogram sums power through triangular filters on the Slaney mel scale
(chapter 10).

## With the library

```rust
use audio_samples::{AudioSamples, AudioTransforms, nzu, sample_rate, white_noise};
use spectrograms::{StftParams, WindowType};
use std::time::Duration;

let x = white_noise::<f64>(Duration::from_secs(1), sample_rate!(16000), 0.5, Some(7));
let l = x.samples_per_channel().get(); // 16 000
let frames = |n: usize, h: usize, centre: bool| {
    let p = if centre { n / 2 } else { 0 };
    1 + (l + 2 * p - n) / h
};

let params = StftParams::new(nzu!(512), nzu!(128), WindowType::Hanning, true).unwrap();
let st = x.stft(&params).unwrap();
assert_eq!(st.data.nrows(), 257);                 // N/2 + 1 bins
assert_eq!(st.data.ncols(), frames(512, 128, true));
assert_eq!(st.data.ncols(), 126);

let no_centre = StftParams::new(nzu!(512), nzu!(128), WindowType::Hanning, false).unwrap();
assert_eq!(x.stft(&no_centre).unwrap().data.ncols(), 122);

// Inverse: 125 * 128 + 512 - 512 = 16 000 samples, equal to the input to round-off.
let y = AudioSamples::<f64>::istft(st).unwrap();
assert_eq!(y.samples_per_channel().get(), 16_000);
let err = x.as_slice().unwrap().iter().zip(y.as_slice().unwrap())
    .map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
assert!(err < 1e-12); // measured: about 3e-16
```

The trade-off, measured. Two tones 50 Hz apart, and a click, analysed with a
16 ms and a 128 ms window (hop 2 ms in both):

```rust
use audio_samples::utils::generation::impulse;
use audio_samples::{AudioSamples, AudioTransforms, nzu, sample_rate, sine_wave};
use spectrograms::{StftParams, WindowType};
use std::num::NonZeroUsize;
use std::time::Duration;

let half_s = Duration::from_millis(500);
let a = sine_wave::<f64>(1000.0, half_s, sample_rate!(16000), 0.5);
let b = sine_wave::<f64>(1050.0, half_s, sample_rate!(16000), 0.5);
let tones: Vec<f64> = a.as_slice().unwrap().iter().zip(b.as_slice().unwrap()).map(|(p, q)| p + q).collect();
let tones = AudioSamples::new_mono(ndarray::Array1::from(tones), sample_rate!(16000)).unwrap();
let click = impulse::<f64>(half_s, sample_rate!(16000), 1.0, 0.25);

let analyse = |n: usize| -> (usize, f64) {
    let p = StftParams::new(NonZeroUsize::new(n).unwrap(), nzu!(32), WindowType::Hanning, true).unwrap();
    // Frequency: summed magnitude per bin; count peaks between 800 and 1250 Hz.
    let s = tones.stft(&p).unwrap();
    let avg: Vec<f64> = (0..s.data.nrows()).map(|k| s.data.row(k).iter().map(|c| c.norm()).sum()).collect();
    let bin = 16_000.0 / n as f64;
    let (lo, hi) = ((800.0 / bin) as usize, (1250.0 / bin) as usize);
    let top = avg[lo..hi].iter().copied().fold(0.0, f64::max);
    let peaks = (lo..hi).filter(|&i| avg[i] > avg[i - 1] && avg[i] >= avg[i + 1] && avg[i] > 0.5 * top).count();
    // Time: frames where the click's energy exceeds half its maximum, in ms.
    let c = click.stft(&p).unwrap();
    let e: Vec<f64> = (0..c.data.ncols()).map(|j| c.data.column(j).iter().map(|v| v.norm_sqr()).sum()).collect();
    let emax = e.iter().copied().fold(0.0, f64::max);
    let spread_ms = e.iter().filter(|&&v| v > 0.5 * emax).count() as f64 * 2.0;
    (peaks, spread_ms)
};
let (peaks_short, spread_short) = analyse(256);   // 16 ms window, Δf = 62.5 Hz
let (peaks_long, spread_long) = analyse(2048);    // 128 ms window, Δf = 7.8 Hz
assert_eq!((peaks_short, peaks_long), (1, 2));    // only the long window resolves 50 Hz
assert_eq!((spread_short, spread_long), (6.0, 46.0)); // ... and smears the click 8x more
```

Spectrogram flavours and the absolute dB floor. A full-scale 1 kHz sine is
bin-centred for $N = 512$ at 16 kHz (bin 32), so its peak power is
$(\sum w / 2)^2$ and its dB value is $20\log_{10}(255.5/2) = 42.13$ dB re 1:
**above** 0 dB.

```rust
use audio_samples::utils::generation::silence;
use audio_samples::{AudioTransforms, nzu, sample_rate, sine_wave};
use spectrograms::{LogHzParams, LogParams, MelParams, SpectrogramParams, StftParams, WindowType};
use std::time::Duration;

let sine = sine_wave::<f64>(1000.0, Duration::from_millis(500), sample_rate!(16000), 1.0);
let stft = StftParams::new(nzu!(512), nzu!(128), WindowType::Hanning, true).unwrap();
let sp = SpectrogramParams::new(stft, 16_000.0).unwrap();
let floor = LogParams::new(-80.0).unwrap();

let lin = sine.linear_db_spectrogram(&sp, &floor).unwrap();
let loghz = sine.loghz_power_spectrogram(&sp, &LogHzParams::new(nzu!(64), 80.0, 8000.0).unwrap()).unwrap();
let mel = sine.mel_power_spectrogram(&sp, &MelParams::new(nzu!(40), 0.0, 8000.0).unwrap()).unwrap();
assert_eq!((lin.data().nrows(), loghz.data().nrows(), mel.data().nrows()), (257, 64, 40));

let peak_db = lin.data().iter().copied().fold(f64::MIN, f64::max);
assert!((peak_db - 42.13).abs() < 0.01);

// Silence sits exactly at the floor: -80 dB re 1, whatever else is in the file.
let quiet = silence::<f64>(Duration::from_millis(250), sample_rate!(16000));
let q = quiet.linear_db_spectrogram(&sp, &floor).unwrap();
assert!(q.data().iter().all(|&v| (v + 80.0).abs() < 1e-9));

// The mel band with the most energy is the one centred nearest 1 kHz (1031 Hz).
// (frequencies() is correct here because the range is 0 Hz to Nyquist.)
let f = mel.frequencies();
let best = (0..40).max_by(|&i, &j| mel.data().row(i).sum().total_cmp(&mel.data().row(j).sum())).unwrap();
assert!((f[best] - 1031.0).abs() < 1.0);
```

## Under the hood

Rebuild one frame from the definition (pad, slice, window, DFT), check the
power spectrogram is $|X|^2$, and measure the overlap-add sums that `istft`
relies on:

```rust
use audio_samples::educational::Explain;
use audio_samples::{AudioTransforms, nzu, sample_rate, white_noise};
use num_complex::Complex;
use spectrograms::{SpectrogramParams, StftParams, WindowType};
use std::f64::consts::PI;
use std::time::Duration;

let x = white_noise::<f64>(Duration::from_millis(250), sample_rate!(16000), 0.5, Some(3));
let xs = x.as_slice().unwrap();
let (n, h, m) = (512usize, 128usize, 10usize);
let params = StftParams::new(nzu!(512), nzu!(128), WindowType::Hanning, true).unwrap();
let st = x.stft(&params).unwrap();

// Symmetric Hann, denominator N - 1.
let w: Vec<f64> = (0..n).map(|i| 0.5 - 0.5 * (2.0 * PI * i as f64 / (n - 1) as f64).cos()).collect();
// Frame m covers padded samples mH .. mH + N, i.e. original samples mH - N/2 ...
let frame: Vec<f64> = (0..n).map(|i| {
    let idx = (m * h + i) as isize - (n / 2) as isize;
    if idx >= 0 && (idx as usize) < xs.len() { xs[idx as usize] * w[i] } else { 0.0 }
}).collect();
for k in [0, 1, 37, 128, 256] {
    let xk: Complex<f64> = frame.iter().enumerate()
        .map(|(i, &v)| Complex::from_polar(v, -2.0 * PI * ((k * i) % n) as f64 / n as f64)).sum();
    // 512-term sums of O(0.5) values: rounding of order 1e-13.
    assert!((xk - st.data[[k, m]]).norm() < 1e-10);
}

// Power spectrogram = |X|^2, element by element.
let sp = SpectrogramParams::new(params.clone(), 16_000.0).unwrap();
let pow = x.linear_power_spectrogram(&sp).unwrap();
assert!((pow.data()[[37, m]] - st.data[[37, m]].norm_sqr()).abs() < 1e-9);

// Overlap-add sums in the steady state (away from the ends).
let sum_w: Vec<f64> = (0..h).map(|r| (0..n / h).map(|q| w[r + q * h]).sum()).collect();
let sum_w2: Vec<f64> = (0..h).map(|r| (0..n / h).map(|q| w[r + q * h].powi(2)).sum()).collect();
let mean = sum_w.iter().sum::<f64>() / h as f64;
let (lo, hi) = sum_w.iter().fold((f64::MAX, f64::MIN), |(a, b), &v| (a.min(v), b.max(v)));
let ripple = (hi - lo) / mean;
// The symmetric Hann at 75 % overlap is *almost* COLA (sum ~ 2), not exactly.
assert!((mean - 2.0).abs() < 0.01 && ripple > 0.0 && ripple < 0.01);
// NOLA holds comfortably, so istft is exact anyway.
assert!(sum_w2.iter().copied().fold(f64::MAX, f64::min) > 1.0);
let doc = params.explain();
assert!((doc.quantity("Overlap-add ripple").unwrap().value - ripple).abs() < 1e-6);
```

## Ask the library to explain

`StftParams` implements [`Explain`](crate::educational::Explain). Through a
signal ([`ExplainFor`](crate::educational::ExplainFor)) it knows the sample
rate and length; add the expected pitch range and it warns when the window is
too short for the lowest voice:

```rust
use audio_samples::educational::{Explain, ExplainContext, ExplainFor, WarningLevel};
use audio_samples::{nzu, sample_rate, sine_wave};
use spectrograms::{StftParams, WindowType};
use std::num::NonZeroUsize;
use std::time::Duration;

let speech_like = sine_wave::<f64>(120.0, Duration::from_secs(1), sample_rate!(16000), 0.5);
let params = StftParams::new(nzu!(512), nzu!(128), WindowType::Hanning, true).unwrap();
let doc = speech_like.explain(&params);
assert_eq!(doc.quantity("Frequency resolution").unwrap().value, 31.25); // Hz
assert_eq!(doc.quantity("Frame period").unwrap().value, 8.0);           // ms
assert_eq!(doc.quantity("Frames").unwrap().value, 126.0);

// f0 between 80 and 400 Hz: two periods of 80 Hz last 25 ms.
for (n, short, coarse) in [(1024, false, false), (256, true, false), (128, true, true)] {
    let p = StftParams::new(NonZeroUsize::new(n).unwrap(), NonZeroUsize::new(n / 4).unwrap(),
                            WindowType::Hanning, true).unwrap();
    let ctx = ExplainContext::for_audio(&speech_like).with_f0_range(80.0, 400.0);
    let d = p.explain_with(&ctx);
    // 16 ms (N = 256) and 8 ms (N = 128) windows are shorter than 25 ms ...
    assert_eq!(d.has_warning_containing("shorter than two periods"), short);
    // ... and at N = 128 the 125 Hz bins are coarser than 80 Hz.
    assert_eq!(d.has_warning_containing("coarser than the lowest expected f0"), coarse);
    assert_eq!(d.has_warning(WarningLevel::Caution), short || coarse);
}
println!("{doc}");
```

## Exercises

1. How many frames does `stft` produce for 2.5 s of audio at 22.05 kHz with
   $N = 2048$, $H = 512$ and centring? Predict, then check.
2. At 16 kHz you want one frame every 10 ms. Choose $H$, then the smallest
   power-of-two $N$ for each of two requirements: (a) a **bin spacing**
   $f_s/N$ of at most 20 Hz; (b) a Hann **main lobe** ($-6$ dB width about
   $2 f_s/N$, the real resolution) of at most 20 Hz. Confirm (a) with the
   explainer.
3. A signal is 0.5 s of a full-scale 1 kHz sine followed by 0.5 s of silence.
   With `LogParams::new(-100.0)`, what value do the bins of the last frame
   take, and how far below the spectrogram's peak is that? (librosa's
   `power_to_db(S, ref=np.max, top_db=None)` reports this distance; with its
   default `top_db=80` it would clip it to $-80$ dB.)

## Answers

```rust
use audio_samples::educational::ExplainContext;
use audio_samples::educational::Explain;
use audio_samples::{AudioSamples, AudioTransforms, nzu, sample_rate, sine_wave};
use spectrograms::{LogParams, SpectrogramParams, StftParams, WindowType};
use std::time::Duration;

// 1. L = 55 125; 1 + floor((55 125 + 2*1024 - 2048) / 512) = 1 + 107 = 108.
let l = 55_125usize;
assert_eq!(1 + (l + 2 * 1024 - 2048) / 512, 108);
let x = sine_wave::<f64>(440.0, Duration::from_millis(2500), sample_rate!(22050), 0.5);
assert_eq!(x.samples_per_channel().get(), l);
let p = StftParams::new(nzu!(2048), nzu!(512), WindowType::Hanning, true).unwrap();
assert_eq!(x.stft(&p).unwrap().data.ncols(), 108);

// 2. H = 0.010 * 16 000 = 160.
//    (a) bin spacing: N >= 16 000 / 20 = 800, next power of two 1024.
//    (b) main lobe:   2 * 16 000 / N <= 20, so N >= 1600, next power of two 2048.
let h = (0.010f64 * 16_000.0) as usize;
let n = ((16_000.0f64 / 20.0).ceil() as usize).next_power_of_two();
let n_lobe = ((2.0 * 16_000.0f64 / 20.0).ceil() as usize).next_power_of_two();
assert_eq!((h, n, n_lobe), (160, 1024, 2048));
let p = StftParams::new(nzu!(1024), nzu!(160), WindowType::Hanning, true).unwrap();
let d = p.explain_with(&ExplainContext::new().with_sample_rate(16_000.0));
assert_eq!(d.quantity("Frame period").unwrap().value, 10.0);
assert_eq!(d.quantity("Frequency resolution").unwrap().value, 15.625); // bin spacing
// (b) The explainer's measured -6 dB main-lobe width: too wide at N = 1024,
//     narrow enough at N = 2048 (about 2 bins in both cases).
let lobe = |n: usize| {
    let p = StftParams::new(std::num::NonZeroUsize::new(n).unwrap(), nzu!(160), WindowType::Hanning, true).unwrap();
    p.explain_with(&ExplainContext::new().with_sample_rate(16_000.0))
        .quantity("Main-lobe width (-6 dB)").unwrap().value
};
assert!(lobe(1024) > 20.0 && lobe(2048) <= 20.0);

// 3. -100 dB re 1, which is 100 + 42.13 = 142.13 dB below the peak. (librosa
//    gets the same with top_db=None because its default amin = 1e-10 is the
//    same -100 dB floor.)
let mut v = sine_wave::<f64>(1000.0, Duration::from_millis(500), sample_rate!(16000), 1.0)
    .as_slice().unwrap().to_vec();
v.extend(std::iter::repeat_n(0.0, 8000));
let a = AudioSamples::new_mono(ndarray::Array1::from(v), sample_rate!(16000)).unwrap();
let p = StftParams::new(nzu!(512), nzu!(128), WindowType::Hanning, true).unwrap();
let sp = SpectrogramParams::new(p, 16_000.0).unwrap();
let s = a.linear_db_spectrogram(&sp, &LogParams::new(-100.0).unwrap()).unwrap();
let last = s.data().ncols() - 1;
assert!(s.data().column(last).iter().all(|&db| (db + 100.0).abs() < 1e-9));
let peak = s.data().iter().copied().fold(f64::MIN, f64::max);
assert!((s.data()[[0, last]] - peak + 142.13).abs() < 0.01);
```

## Common pitfalls

* Centring pads with **zeros**, as librosa has done by default since 0.10
  (`pad_mode='constant'`). Older librosa (< 0.10) and `torch.stft` default to
  reflection, so their first and last frames differ.
* The window is the **symmetric** Hann; librosa and SciPy use the periodic
  one. Against current librosa this, not the padding, is the real difference:
  values differ slightly unless you pass the symmetric window yourself.
* dB values are **re 1** with an absolute floor; there is no `top_db`.
  Subtract the maximum yourself to get "dB below peak".
* Power is $|X|^2$ with no $1/N$ or window normalisation: a full-scale sine
  reads about $+42$ dB at $N = 512$, and the number grows with $N$.
* `istft` can return up to $H - 1$ samples fewer than the input; trim or pad.
* The mel spectrogram's `frequencies()` axis is only right for
  $f_\mathrm{min} = 0$ and $f_\mathrm{max} = f_s/2$ in `spectrograms` 2.1.3.
* A smaller hop does not sharpen anything; only $N$ (and the window shape)
  sets resolution.

## References

* J. B. Allen and L. R. Rabiner, "A unified approach to short-time Fourier analysis and synthesis", *Proceedings of the IEEE* 65(11), 1558-1564, 1977.
* D. W. Griffin and J. S. Lim, "Signal estimation from modified short-time Fourier transform", *IEEE Transactions on Acoustics, Speech, and Signal Processing* 32(2), 236-243, 1984.
* F. J. Harris, "On the use of windows for harmonic analysis with the discrete Fourier transform", *Proceedings of the IEEE* 66(1), 51-83, 1978.
* B. McFee et al., "librosa: audio and music signal analysis in Python", *Proceedings of the 14th Python in Science Conference*, 2015.

## See also

* [`AudioTransforms::stft`](crate::AudioTransforms::stft),
  [`AudioTransforms::istft`](crate::AudioTransforms::istft),
  [`AudioTransforms::linear_db_spectrogram`](crate::AudioTransforms::linear_db_spectrogram),
  [`AudioTransforms::loghz_power_spectrogram`](crate::AudioTransforms::loghz_power_spectrogram),
  [`AudioTransforms::mel_power_spectrogram`](crate::AudioTransforms::mel_power_spectrogram)
* The [`transforms`](crate::operations::transforms) module docs (framing,
  dB and mel conventions)
* [`Explain`](crate::educational::Explain),
  [`ExplainContext::with_f0_range`](crate::educational::ExplainContext::with_f0_range),
  [`topics::window`](crate::educational::topics::window)
* Previous: [03. The DFT, windows and leakage](03_dft_windows.md).
  Next: [05. Convolution and FIR filters](05_convolution_fir.md).
  Mel features in depth: [10. Cepstrum, mel and MFCC](10_cepstrum_mel_mfcc.md).
* Lab: `examples/tutorial_04_stft_spectrograms.rs`
  (`cargo run --release --example tutorial_04_stft_spectrograms --features educational,full_no_plotting`).
