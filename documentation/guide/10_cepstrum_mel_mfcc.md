# 10. Cepstrum, mel and MFCC

Speech recognisers rarely use raw spectra. They warp frequency to approximate
the ear's frequency resolution (the mel scale), take logarithms to turn the source-filter product
into a sum, and keep a dozen smooth "cepstral" coefficients that describe the
vocal-tract envelope while discarding pitch. This chapter builds that
pipeline, mel-frequency cepstral coefficients (MFCCs), from the real
cepstrum upwards, and checks each step against the library.

## Concept

**Real cepstrum.** For an $N$-point frame with DFT $X[k]$,

$$c[q] = \frac{1}{N} \sum_{k=0}^{N-1} \log |X[k]| \, e^{j 2 \pi k q / N}$$

where $q$ is the *quefrency* in samples ($q/f_s$ seconds, $f_s$ the sample
rate in Hz) and $\log$ is the natural logarithm. Because $|X|$ is real and
even, $c[q]$ is real and even. Taking logs turns the source-filter product
$|X| = |E| \cdot |H|$ into a sum, $\log|X| = \log|E| + \log|H|$: the smooth
vocal-tract envelope $|H|$ occupies the low quefrencies, and a periodic source
with period $\tau_0$ samples produces a peak at $q = \tau_0$, so
$\hat{f}_0 = f_s / q_{peak}$. The crate has **no cepstrum function**; we compute
it from [`AudioTransforms::fft`](crate::AudioTransforms::fft) ourselves.

**Mel scale.** Two conventions are in use, and this crate uses both:

- HTK (O'Shaughnessy): $m = 2595 \log_{10}(1 + f/700)$, inverse
  $f = 700 (10^{m/2595} - 1)$. Used by
  [`hz_to_mel`](crate::utils::audio_math::hz_to_mel),
  [`mel_to_hz`](crate::utils::audio_math::mel_to_hz) and
  [`mel_frequencies`](crate::utils::audio_math::mel_frequencies).
- Slaney (Auditory Toolbox): $m = 3f/200$ for
  $f < 1000$ Hz and $m = 15 + 27 \ln(f/1000) / \ln 6.4$ above. Used by the mel
  spectrogram and therefore by MFCCs.

Here $f$ is in Hz and $m$ in mels; the two scales have very different units
(1000 Hz is 1000 HTK mels but 15 Slaney mels), so never mix them.

**Mel filterbank.** $M$ triangular filters with edges equally spaced in mels:
$f_i = m^{-1}(m(f_{min}) + i \, \Delta m)$, $\Delta m = (m(f_{max}) - m(f_{min}))/(M+1)$,
$i = 0, \dots, M+1$. Filter $i$ (with neighbours $f_{i-1} < f_i < f_{i+1}$, in Hz)
weights FFT bin $k$ at frequency $k f_s / N_{fft}$ by

$$H_i(f) = \max\left(0, \min\left(\frac{f - f_{i-1}}{f_i - f_{i-1}}, \frac{f_{i+1} - f}{f_{i+1} - f_i}\right)\right)$$

and the mel power in band $i$ is $P_i = \sum_k H_i(k f_s/N_{fft}) \, |X[k]|^2$.

**MFCC, as this library computes it.** Per frame, with $M$ mel bands from 0 Hz
to $f_s/2$ (no filter normalisation):

$$L_i = 10 \log_{10} \max(P_i, 10^{-8}), \qquad c_k = \sum_{i=0}^{M-1} L_i \cos\left(\frac{\pi k (i + \frac{1}{2})}{M}\right), \qquad \tilde{c}_k = \left(1 + \frac{D}{2} \sin\frac{\pi k}{D}\right) c_k$$

where $L_i$ is the log mel energy in dB with an absolute floor of -80 dB,
$c_k$ the **unnormalised** DCT-II ($k = 0, \dots, K-1$, $K$ = `n_mfcc`) and
$\tilde{c}_k$ the liftered coefficient with lifter parameter $D = 22$ by
default (0 disables it). The orthonormal DCT-II gives $c_0/\sqrt{M}$ and
$c_k \sqrt{2/M}$, so with the lifter off this library's values are larger than
the orthonormal ones by $\sqrt{M}$ for $c_0$ and $\sqrt{M/2}$ for $k \ge 1$.

## With the library

```rust
use audio_samples::{AudioSamples, AudioTransforms, nzu, sample_rate, sine_wave};
use spectrograms::{MelParams, MfccParams, SpectrogramParams, StftParams, WindowType};
use std::time::Duration;

let fs = 16_000.0;
let audio = sine_wave::<f64>(440.0, Duration::from_millis(250), sample_rate!(16000), 0.5);

// 32 ms frames every 10 ms, the usual speech setting.
let stft = StftParams::new(nzu!(512), nzu!(160), WindowType::Hanning, true).unwrap();
let mel = MelParams::new(nzu!(40), 0.0, fs / 2.0).unwrap();
let mel_power = audio
    .mel_power_spectrogram(&SpectrogramParams::new(stft.clone(), fs).unwrap(), &mel)
    .unwrap();
let mfcc = audio.mfcc(&stft, nzu!(40), &MfccParams::speech_standard()).unwrap();

// 4000 samples, hop 160, centred frames: 1 + 4000/160 = 26 frames.
assert_eq!(mel_power.data().dim(), (40, 26));
assert_eq!(mfcc.data.dim(), (13, 26)); // 13 coefficients, c0 included, lifter 22
// The tone falls in the band whose centre is nearest 440 Hz. Below 1 kHz the
// Slaney scale is linear, so band i (0-based) is centred at (i + 1) * 73.57 Hz.
let frame = 13;
let row = |b: usize| mel_power.data()[[b, frame]];
let loudest = (0..40).max_by(|&a, &b| row(a).total_cmp(&row(b))).unwrap();
let centre = (loudest + 1) as f64 * (200.0 / 3.0) * (15.0 + 27.0 * 8f64.ln() / 6.4f64.ln()) / 41.0;
println!("loudest mel band: {loudest}, centred at {centre:.1} Hz");
assert_eq!(loudest, 5);
assert!((centre - 441.4).abs() < 0.05);
```

## How it works

**The real cepstrum and cepstral pitch.** A vowel-like frame (the formant
filter of chapter 8 excited every 100 samples, $f_0 = 160$ Hz), Hann-windowed,
1024 samples. The one-sided FFT holds bins $0 \dots N/2$; the even symmetry of
$\log|X|$ folds the inverse DFT onto those bins.

```rust
use audio_samples::{AudioSamples, AudioTransforms, sample_rate};
use ndarray::Array1;
use std::f64::consts::PI;

let fs = 16_000.0;
// Vowel: impulse train (period 100 samples) through three formant resonators.
let mut a = vec![1.0f64];
for &(f, b) in &[(700.0, 80.0), (1220.0, 90.0), (2600.0f64, 120.0f64)] {
    let (r, th) = ((-PI * b / fs).exp(), 2.0 * PI * f / fs);
    let s = [1.0, -2.0 * r * th.cos(), r * r];
    let mut o = vec![0.0; a.len() + 2];
    for (i, ai) in a.iter().enumerate() { for (j, sj) in s.iter().enumerate() { o[i + j] += ai * sj; } }
    a = o;
}
let mut y = vec![0.0f64; 2600];
for i in 0..y.len() {
    let e = if i % 100 == 0 { 1.0 } else { 0.0 };
    y[i] = e - (1..a.len()).filter(|&k| k <= i).map(|k| a[k] * y[i - k]).sum::<f64>();
}

let n = 1024;
let frame: Vec<f64> = (0..n)
    .map(|i| (0.5 - 0.5 * (2.0 * PI * i as f64 / (n - 1) as f64).cos()) * y[1500 + i])
    .collect();
let audio = AudioSamples::new_mono(Array1::from(frame), sample_rate!(16000)).unwrap();
let x = audio.fft(std::num::NonZeroUsize::new(n).unwrap()).unwrap(); // bins 0..=512
let logmag: Vec<f64> = (0..=n / 2).map(|k| x[[0, k]].norm().max(1e-12).ln()).collect();

// c[q] = (1/N) [ L0 + L_{N/2} cos(pi q) + 2 sum_{k=1}^{N/2-1} L_k cos(2 pi k q / N) ]
let ceps = |q: usize| {
    let mut s = logmag[0] + logmag[n / 2] * (PI * q as f64).cos();
    for k in 1..n / 2 {
        s += 2.0 * logmag[k] * (2.0 * PI * (k * q) as f64 / n as f64).cos();
    }
    s / n as f64
};
// Search quefrencies for f0 between 60 and 400 Hz.
let (q_lo, q_hi) = ((fs / 400.0) as usize, (fs / 60.0) as usize); // 40, 266
let q_peak = (q_lo..=q_hi).max_by(|&p, &q| ceps(p).total_cmp(&ceps(q))).unwrap();
println!("cepstral peak at q = {q_peak} samples -> f0 = {} Hz", fs / q_peak as f64);
assert_eq!(q_peak, 100);
assert_eq!(fs / q_peak as f64, 160.0);
```

The lab also prints the low quefrencies ($c[1] = 2.32$, $c[2] = 0.47$),
which describe the envelope, and shows that the pitch peak (1.55) is well
above every other value in the search range (at most 0.38).

<!-- figure: ch10-cepstrum -->

**The two mel scales**, side by side, and the HTK round trip.

```rust
use audio_samples::utils::audio_math::{hz_to_mel, mel_to_hz};

let slaney = |f: f64| if f < 1000.0 { 3.0 * f / 200.0 } else { 15.0 + 27.0 * (f / 1000.0).ln() / 6.4f64.ln() };
let slaney_inv = |m: f64| if m < 15.0 { 200.0 * m / 3.0 } else { 1000.0 * (6.4f64.ln() * (m - 15.0) / 27.0).exp() };

for f in [0.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0] {
    println!("{f:6.0} Hz: HTK {:8.2} mel, Slaney {:6.3} mel", hz_to_mel(f), slaney(f));
    assert!((mel_to_hz(hz_to_mel(f)) - f).abs() < 1e-9); // HTK round trip
    assert!((slaney_inv(slaney(f)) - f).abs() < 1e-9);  // Slaney round trip
}
assert!((hz_to_mel(1000.0) - 999.99).abs() < 0.005);
assert!((hz_to_mel(4000.0) - 2146.06).abs() < 0.005);
assert_eq!(slaney(1000.0), 15.0);
assert!((slaney(4000.0) - 35.164).abs() < 5e-4);
```

**The filterbank.** Rebuild the 40 triangles on the Slaney scale and apply
them to the library's own power spectrogram; the result must be the library's
mel power spectrogram.

```rust
use audio_samples::{AudioTransforms, nzu, sample_rate, sine_wave};
use spectrograms::{MelParams, SpectrogramParams, StftParams, WindowType};
use std::time::Duration;

let fs = 16_000.0;
let audio = sine_wave::<f64>(440.0, Duration::from_millis(250), sample_rate!(16000), 0.5);
let sp = SpectrogramParams::new(StftParams::new(nzu!(512), nzu!(160), WindowType::Hanning, true).unwrap(), fs).unwrap();
let m = 40;
let lin = audio.linear_power_spectrogram(&sp).unwrap();
let mel = audio.mel_power_spectrogram(&sp, &MelParams::new(nzu!(40), 0.0, fs / 2.0).unwrap()).unwrap();
let (lin, mel) = (lin.data(), mel.data());

let slaney = |f: f64| if f < 1000.0 { 3.0 * f / 200.0 } else { 15.0 + 27.0 * (f / 1000.0).ln() / 6.4f64.ln() };
let slaney_inv = |m: f64| if m < 15.0 { 200.0 * m / 3.0 } else { 1000.0 * (6.4f64.ln() * (m - 15.0) / 27.0).exp() };
let step = (slaney(fs / 2.0) - slaney(0.0)) / (m + 1) as f64;
let edges: Vec<f64> = (0..m + 2).map(|i| slaney_inv(i as f64 * step)).collect();

let mut worst = 0.0f64;
for t in 0..mel.ncols() {
    for i in 0..m {
        let (l, c, r) = (edges[i], edges[i + 1], edges[i + 2]);
        let p: f64 = (0..lin.nrows())
            .map(|k| {
                let f = k as f64 * fs / 512.0;
                ((f - l) / (c - l)).min((r - f) / (r - c)).clamp(0.0, 1.0) * lin[[k, t]]
            })
            .sum();
        worst = worst.max((p - mel[[i, t]]).abs() / mel[[i, t]].max(1e-300));
    }
}
// Same products summed in a different order: rounding-level agreement.
assert!(worst < 1e-10, "{worst}");
```

<!-- figure: ch10-filterbank -->

**MFCC, re-derived.** From the library's mel power spectrogram: the dB floor,
the unnormalised DCT-II and the lifter, compared with
[`AudioTransforms::mfcc`](crate::AudioTransforms::mfcc); then the factor
relative to the orthonormal DCT.

```rust
use audio_samples::{AudioTransforms, nzu, sample_rate, sine_wave};
use spectrograms::{MelParams, MfccParams, SpectrogramParams, StftParams, WindowType};
use std::f64::consts::PI;
use std::time::Duration;

let fs = 16_000.0;
let audio = sine_wave::<f64>(440.0, Duration::from_millis(250), sample_rate!(16000), 0.5);
let stft = StftParams::new(nzu!(512), nzu!(160), WindowType::Hanning, true).unwrap();
let m = 40;
let mel = audio
    .mel_power_spectrogram(&SpectrogramParams::new(stft.clone(), fs).unwrap(), &MelParams::new(nzu!(40), 0.0, fs / 2.0).unwrap())
    .unwrap();
let params = MfccParams::speech_standard(); // 13 coefficients, c0 kept, lifter 22
let lib = audio.mfcc(&stft, nzu!(40), &params).unwrap();
let lib_plain = audio.mfcc(&stft, nzu!(40), &params.with_lifter(0)).unwrap();

let dct = |l: &[f64], k: usize| -> f64 {
    l.iter().enumerate().map(|(i, li)| li * (PI * k as f64 * (i as f64 + 0.5) / m as f64).cos()).sum()
};
let mut worst = 0.0f64;
for t in 0..mel.data().ncols() {
    let l: Vec<f64> = (0..m).map(|i| 10.0 * mel.data()[[i, t]].max(1e-8).log10()).collect();
    for k in 0..13 {
        let lifter = 1.0 + 11.0 * (PI * k as f64 / 22.0).sin();
        worst = worst.max((dct(&l, k) * lifter - lib.data[[k, t]]).abs());
        // Orthonormal DCT-II, no lifter.
        let ortho = dct(&l, k) * if k == 0 { (1.0 / m as f64).sqrt() } else { (2.0 / m as f64).sqrt() };
        let factor = if k == 0 { (m as f64).sqrt() } else { (m as f64 / 2.0).sqrt() };
        assert!((lib_plain.data[[k, t]] - factor * ortho).abs() < 1e-9 * lib_plain.data[[k, t]].abs().max(1.0));
    }
}
// The liftered MFCC values here reach about 2300 in magnitude (2298), so
// 1e-8 absolute is rounding level.
let largest = lib.data.iter().fold(0.0f64, |a, v| a.max(v.abs()));
assert!((largest - 2298.06).abs() < 0.01);
assert!(worst < 1e-8, "{worst}");
```

So with 40 bands and the lifter off, this crate's $c_0$ is $\sqrt{40} = 6.32$
times the orthonormal value and $c_k$ ($k \ge 1$) $\sqrt{20} = 4.47$ times.

<!-- figure: ch10-mfcc -->

## Ask the library to explain

[`MelParams`](spectrograms::MelParams) and [`MfccParams`](spectrograms::MfccParams)
implement [`Explain`](crate::educational::Explain). Given the FFT size, the
mel explanation counts filters that contain no FFT bin: with 128 bands and
`n_fft = 256` at 16 kHz the bins are 62.5 Hz apart but the lowest triangles are
only 46.8 Hz wide. The code below checks the prediction against a computed
spectrogram.

```rust
use audio_samples::educational::{Explain, ExplainContext, WarningLevel, topics};
use audio_samples::{AudioTransforms, nzu, sample_rate, white_noise};
use spectrograms::{MelParams, MfccParams, SpectrogramParams, StftParams, WindowType};
use std::time::Duration;

let mel = MelParams::new(nzu!(128), 0.0, 8000.0).unwrap();
let doc = mel.explain_with(&ExplainContext::new().with_sample_rate(16_000.0).with_n_fft(256));
println!("{doc}");
let empty = doc.quantity("Empty filters").unwrap().value as usize;
assert_eq!(empty, 13);
assert!(doc.has_warning_containing("contain no FFT bin"));
assert!((doc.quantity("Narrowest band width").unwrap().value - 46.77).abs() < 0.01);

// The computed spectrogram of white noise has exactly that many all-zero bands.
let noise = white_noise::<f64>(Duration::from_millis(250), sample_rate!(16000), 0.5, Some(7));
let sp = SpectrogramParams::new(StftParams::new(nzu!(256), nzu!(128), WindowType::Hanning, true).unwrap(), 16_000.0).unwrap();
let s = noise.mel_power_spectrogram(&sp, &mel).unwrap();
let zero_rows = s.data().rows().into_iter().filter(|r| r.iter().all(|&v| v == 0.0)).count();
assert_eq!(zero_rows, empty);

// MFCC: the lifter weights 1 + 11 sin(pi k / 22) peak at k = 11.
let mdoc = MfccParams::speech_standard().explain_with(&ExplainContext::new().with_n_mels(40));
assert_eq!(mdoc.quantity("Largest lifter weight").unwrap().value, 12.0);
// A lifter shorter than the number of coefficients gives negative weights.
let bad = MfccParams::speech_standard().with_lifter(10).explain_with(&ExplainContext::new().with_n_mels(40));
assert!(bad.has_warning(WarningLevel::Caution));

// Cepstrum: the longest quefrency of a 1024-point frame is 512 samples, so f0 > 31.25 Hz.
let cdoc = topics::cepstrum(1024, 16_000.0);
assert_eq!(cdoc.quantity("Lowest detectable f0").unwrap().value, 31.25);
assert!(cdoc.has_warning_containing("no stand-alone cepstrum"));
```

## Exercises

1. **Quefrency.** A 250 Hz voice is analysed with 1024-point frames at
   16 kHz. At which quefrency (samples and ms) do you expect the cepstral
   peak? Check with a synthetic vowel.
2. **Units.** Predict `hz_to_mel(700.0)` and the Slaney mel value of 700 Hz.
3. **Empty bands.** For 128 mel bands at 16 kHz, what is the smallest
   power-of-two `n_fft` with no empty filter?
4. **Comparing with the orthonormal DCT.** With 40 mel bands and the default
   lifter, by what total factor does this crate's $c_3$ exceed the orthonormal,
   unliftered $c_3$ (same log mel input)?

## Answers

**1.** $q = f_s/f_0 = 64$ samples, 4 ms.

```rust
use audio_samples::{AudioSamples, AudioTransforms, sample_rate};
use ndarray::Array1;
use std::f64::consts::PI;
let fs = 16_000.0;
let mut a = vec![1.0f64];
for &(f, b) in &[(700.0, 80.0), (1220.0, 90.0), (2600.0f64, 120.0f64)] {
    let (r, th) = ((-PI * b / fs).exp(), 2.0 * PI * f / fs);
    let s = [1.0, -2.0 * r * th.cos(), r * r];
    let mut o = vec![0.0; a.len() + 2];
    for (i, ai) in a.iter().enumerate() { for (j, sj) in s.iter().enumerate() { o[i + j] += ai * sj; } }
    a = o;
}
let mut y = vec![0.0f64; 2600];
for i in 0..y.len() {
    let e = if i % 64 == 0 { 1.0 } else { 0.0 }; // 250 Hz
    y[i] = e - (1..a.len()).filter(|&k| k <= i).map(|k| a[k] * y[i - k]).sum::<f64>();
}
let n = 1024;
let frame: Vec<f64> = (0..n).map(|i| (0.5 - 0.5 * (2.0 * PI * i as f64 / (n - 1) as f64).cos()) * y[1500 + i]).collect();
let x = AudioSamples::new_mono(Array1::from(frame), sample_rate!(16000)).unwrap()
    .fft(std::num::NonZeroUsize::new(n).unwrap()).unwrap();
let lm: Vec<f64> = (0..=n / 2).map(|k| x[[0, k]].norm().max(1e-12).ln()).collect();
let ceps = |q: usize| (lm[0] + lm[n / 2] * (PI * q as f64).cos()
    + (1..n / 2).map(|k| 2.0 * lm[k] * (2.0 * PI * (k * q) as f64 / n as f64).cos()).sum::<f64>()) / n as f64;
let q_peak = (40..=266).max_by(|&p, &q| ceps(p).total_cmp(&ceps(q))).unwrap();
assert_eq!(q_peak, 64);
assert_eq!(1000.0 * q_peak as f64 / fs, 4.0);
```

**2.** HTK: $2595 \log_{10} 2 = 781.17$ mels. Slaney: $3 \cdot 700/200 = 10.5$ mels.

```rust
use audio_samples::utils::audio_math::hz_to_mel;
assert!((hz_to_mel(700.0) - 781.17).abs() < 0.005);
assert_eq!(3.0 * 700.0 / 200.0, 10.5);
```

**3.** 512: the bins are then 31.25 Hz apart, narrower than the 46.8 Hz
lowest triangle, so every triangle contains at least one bin.

```rust
use audio_samples::educational::{Explain, ExplainContext};
use audio_samples::nzu;
use spectrograms::MelParams;
let mel = MelParams::new(nzu!(128), 0.0, 8000.0).unwrap();
let empty = |n_fft: usize| {
    mel.explain_with(&ExplainContext::new().with_sample_rate(16_000.0).with_n_fft(n_fft))
        .quantity("Empty filters").unwrap().value
};
assert!(empty(128) > empty(256)); // coarser bins, more empty bands
assert!(empty(256) > 0.0);
assert_eq!(empty(512), 0.0);
```

**4.** $\sqrt{40/2} \cdot (1 + 11 \sin(3\pi/22)) = 4.472 \times 5.570 = 24.91$.

```rust
let factor = (40.0f64 / 2.0).sqrt() * (1.0 + 11.0 * (3.0 * std::f64::consts::PI / 22.0).sin());
assert!((factor - 24.91).abs() < 0.005);
```

## Common pitfalls

- **Two mel scales.** `audio_math::{hz_to_mel, mel_to_hz, mel_frequencies}`
  are HTK; the mel spectrogram and MFCCs are Slaney. Do not label mel
  spectrogram bands with `mel_frequencies`.
- **No filter normalisation by default.** `MelParams::new` uses
  `MelNorm::None` (peak 1 per triangle), not Slaney's equal-area
  normalisation. `MelParams::with_norm` selects another.
- **Unnormalised DCT and lifter 22** (with $k$ counted from 0): MFCC values are
  much larger than orthonormal-DCT values. Use `with_lifter(0)` and divide by
  $\sqrt{M}$ / $\sqrt{M/2}$ to compare.
- **Absolute log floor.** The dB step clamps at $10^{-8}$ (-80 dB re 1), not
  80 dB below the frame's peak, so a quiet signal's MFCCs depend on its level.
- **Symmetric Hann.** `WindowType::Hanning` is the symmetric window; SciPy
  uses the periodic one by default, so spectra differ slightly.
- **Empty mel bands** appear when bands are narrower than the FFT bin spacing:
  ask `MelParams::explain_with` with the FFT size before choosing `n_mels`.
- **Cepstral pitch needs several periods in the frame** and fails on breathy
  or weak voicing; YIN (chapter 9) is usually more robust.

## References

- S. B. Davis and P. Mermelstein, "Comparison of parametric representations for monosyllabic word recognition in continuously spoken sentences", *IEEE Transactions on Acoustics, Speech, and Signal Processing* 28(4), 357-366, 1980. The MFCC.
- M. Slaney, *Auditory Toolbox, Version 2*, Technical Report 1998-010, Interval Research Corporation, 1998. The Slaney mel scale and filterbank.
- S. Young et al., *The HTK Book*, Cambridge University Engineering Department. The HTK mel formula and the sinusoidal lifter.
- D. O'Shaughnessy, *Speech Communication: Human and Machine*, Addison-Wesley, 1987. The $2595 \log_{10}(1 + f/700)$ mel formula.
- A. M. Noll, "Cepstrum pitch determination", *Journal of the Acoustical Society of America* 41(2), 293-309, 1967.

## See also

- [`AudioTransforms::mfcc`](crate::AudioTransforms::mfcc),
  [`AudioTransforms::mel_power_spectrogram`](crate::AudioTransforms::mel_power_spectrogram),
  [`AudioTransforms::linear_power_spectrogram`](crate::AudioTransforms::linear_power_spectrogram),
  [`AudioTransforms::fft`](crate::AudioTransforms::fft).
- [`MelParams`](spectrograms::MelParams), [`MfccParams`](spectrograms::MfccParams).
- [`hz_to_mel`](crate::utils::audio_math::hz_to_mel),
  [`mel_to_hz`](crate::utils::audio_math::mel_to_hz) (HTK).
- [`topics::cepstrum`](crate::educational::topics::cepstrum).
- Previous chapter: [`c09_pitch`](09_pitch.md); next:
  [`c11_energy_vad`](11_energy_vad.md).
- Lab: `examples/tutorial_10_cepstrum_mel_mfcc.rs`
  (`cargo run --release --example tutorial_10_cepstrum_mel_mfcc --features educational,full_no_plotting`).
