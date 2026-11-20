# 06. IIR filters: design, poles, zeros and stability

A telephone channel band-limits speech, a notch filter removes 50 Hz mains
hum, a de-emphasis filter $1/(1 - \alpha z^{-1})$ undoes the pre-emphasis of
chapter 08 after synthesis: all are recursive (infinite impulse response, IIR)
filters. (Pre-emphasis itself, $1 - \alpha z^{-1}$, has no feedback: it is an
FIR filter.) IIR filters achieve a steep roll-off with few coefficients, but
their output is fed back into their input, so they can ring, distort phase and
become unstable. This chapter designs a Butterworth filter the
way the library does, finds its poles, checks stability and the frequency
response numerically, and ends the "With the library" section with a 50 Hz hum
notch.

## Concept

**Difference equation.** An IIR filter computes each output sample from the
current and past inputs *and* past outputs:

$$y[n] = \sum_{k=0}^{M} b_k\, x[n-k] - \sum_{k=1}^{N} a_k\, y[n-k]$$

where $x[n]$ and $y[n]$ are the input and output samples (dimensionless),
$b_k$ are the feed-forward coefficients, $a_k$ the feedback coefficients
(with $a_0 = 1$), and $M$, $N$ the numerator and denominator orders.

**Transfer function.** Taking the z-transform turns delays into powers of
$z^{-1}$:

$$H(z) = \frac{B(z)}{A(z)} = \frac{b_0 + b_1 z^{-1} + \dots + b_M z^{-M}}{1 + a_1 z^{-1} + \dots + a_N z^{-N}}$$

where $z$ is a complex variable. The roots of $B$ are the **zeros** and the
roots of $A$ are the **poles**. The frequency response is $H$ on the unit
circle, $H(e^{j\omega})$ with $\omega = 2\pi f / f_s$, where $f$ is the
frequency in Hz and $f_s$ the sample rate in Hz.

**Stability.** A causal filter is bounded-input bounded-output stable if and
only if every pole satisfies $|p_i| < 1$ (the poles of $H(z)$ in lowest terms,
after cancelling any factor common to $B$ and $A$). A pole at radius $r$ contributes a
term $r^n$ to the impulse response, so the closer $r$ is to 1, the longer the
filter rings.

**Butterworth prototype.** The analogue Butterworth low-pass of order $N$ has
the maximally flat magnitude

$$|H_a(j\Omega)|^2 = \frac{1}{1 + (\Omega / \Omega_c)^{2N}}$$

where $\Omega$ is the analogue angular frequency in rad/s and $\Omega_c$ the
cutoff in rad/s. At $\Omega = \Omega_c$ the power is halved:
$10 \log_{10}(1/2) = -3.01$ dB. Its $N$ poles lie on a circle of radius
$\Omega_c$ in the left half of the s-plane:

$$s_k = \Omega_c\, e^{j\pi(2k + N + 1)/(2N)}, \quad k = 0, \dots, N-1$$

**Bilinear transform with pre-warping.** The digital filter is obtained by
substituting

$$s = K \frac{1 - z^{-1}}{1 + z^{-1}}$$

where $K$ is a positive constant (rad/s). The usual choice is $K = 2 f_s$;
the library designs with $f_s$ normalised to 2 (so $K = 4$), and every choice
of $K$ gives the same $H(z)$ once the cutoff is pre-warped. The substitution maps the
whole $j\Omega$ axis onto the unit circle, with $\Omega = K \tan(\omega / 2)$,
so frequencies are compressed near Nyquist. To place the digital cutoff exactly
at $f_c$ (Hz) the analogue cutoff is **pre-warped**:

$$\Omega_c = K \tan(\pi f_c / f_s)$$

Each analogue pole maps to $p = (K + s)/(K - s)$, and the $N$ zeros at
$s = \infty$ map to $z = -1$ (Nyquist). Combining the two formulas gives a
closed form for the digital Butterworth magnitude:

$$|H(e^{j\omega})|^2 = \frac{1}{1 + \left(\tan(\pi f / f_s) / \tan(\pi f_c / f_s)\right)^{2N}}$$

**Second-order sections (SOS).** A high-order $A(z)$ is numerically
ill-conditioned: small coefficient errors move the poles a long way. The library therefore factors
$H(z)$ into a cascade of biquads,

$$H(z) = \prod_{i=1}^{S} \frac{b_{0i} + b_{1i} z^{-1} + b_{2i} z^{-2}}{1 + a_{1i} z^{-1} + a_{2i} z^{-2}}$$

where $S = \lceil N/2 \rceil$ is the number of sections and each row is
stored as a row `[b0, b1, b2, a0, a1, a2]` with `a0 = 1`.

**Chebyshev type I** gives a steeper roll-off than Butterworth at the cost of
ripple in the pass band. Its analogue magnitude is

$$|H_a(j\Omega)|^2 = \frac{1}{1 + \varepsilon^2 T_N^2(\Omega / \Omega_c)}, \qquad \varepsilon^2 = 10^{r_p/10} - 1$$

where $r_p$ is the pass-band ripple in dB, $\varepsilon$ the (dimensionless)
ripple factor and $T_N$ the Chebyshev polynomial of order $N$:
$T_N(x) = \cos(N \arccos x)$ for $|x| \le 1$, which oscillates between $-1$ and
$1$, and $T_N(x) = \cosh(N \,\mathrm{arccosh}\, x)$ for $x > 1$, which grows
like $2^{N-1}x^N$. So the pass band ripples between 0 and $-r_p$ dB, and at
$\Omega_c$ ($T_N(1) = 1$) the gain is $-r_p$ dB rather than $-3.01$ dB. At DC,
$T_N(0) = \cos(N\pi/2)$ is $\pm 1$ for even $N$ (gain $-r_p$ dB) and 0 for odd
$N$ (gain 0 dB). As for the Butterworth, the bilinear transform with
pre-warping turns $\Omega/\Omega_c$ into $\tan(\pi f/f_s)/\tan(\pi f_c/f_s)$.

**Zero-phase filtering.** Running the filter forwards and then backwards
(`filtfilt`) gives the overall response $|H|^2$ with zero phase: no delay,
twice the attenuation in dB (so $-6.02$ dB at a Butterworth cutoff), and a
non-causal filter (it needs the whole signal).

The library's design pipeline (`src/operations/iir_design.rs`) runs these
steps in order: analogue prototype → frequency transform (`lp2lp_zpk` and its
high-pass and band variants) → `bilinear_zpk` → `zpk2sos`, and
`tests/filter_reference.rs` checks its coefficients and responses against
golden reference fixtures.

## With the library

Design a 4th-order Butterworth low-pass at 1 kHz for 16 kHz audio, inspect the
sections, and measure the response with
[`IirFilterDesign::frequency_response`](crate::operations::types::IirFilterDesign::frequency_response).

```rust
use audio_samples::nzu;
use audio_samples::operations::types::IirFilterDesign;
use std::f64::consts::PI;

let (fs, fc) = (16_000.0, 1_000.0);
let design = IirFilterDesign::butterworth_lowpass(nzu!(4), fc);
let sos = design.sos_coefficients(fs).unwrap();
assert_eq!(sos.len(), 2); // N = 4 -> two biquads

let probes = [0.0, fc, 2.0 * fc, 4.0 * fc];
let (mag, _phase) = design.frequency_response(&probes, fs).unwrap();
let db: Vec<f64> = mag.iter().map(|m| 20.0 * m.log10()).collect();
// 0 dB at DC and -3.0103 dB at the cutoff (1e-4 dB allows for rounding).
assert!(db[0].abs() < 1e-9);
assert!((db[1] + 3.0103).abs() < 1e-4);
// One octave up the filter is 25.5 dB down, two octaves up 56.1 dB down.
assert!((db[2] + 25.5).abs() < 0.05);
assert!((db[3] + 56.1).abs() < 0.05);

// The closed-form digital Butterworth magnitude matches to 1e-12.
for (&f, &m) in probes.iter().zip(&mag) {
    let r = (PI * f / fs).tan() / (PI * fc / fs).tan();
    assert!((m - 1.0 / (1.0 + r.powi(8)).sqrt()).abs() < 1e-12);
}
```

One octave above the cutoff the digital filter attenuates more (25.5 dB) than
the analogue prototype ($10\log_{10}(1 + 2^8) = 24.1$ dB), because the
bilinear transform compresses the frequency axis towards Nyquist.

<!-- figure: ch06-magnitude -->

Apply it to a signal with
[`AudioIirFiltering::apply_iir_filter`](crate::AudioIirFiltering::apply_iir_filter)
(causal) or
[`AudioIirFiltering::filtfilt`](crate::AudioIirFiltering::filtfilt)
(zero phase). For a 1 kHz tone the causal filter attenuates
by 3.01 dB and shifts the tone by half a period, 8 samples (a 4th-order
Butterworth has $-180°$ of phase at its cutoff); `filtfilt` attenuates by
6.02 dB and does not delay. Half a period is the **phase delay**
$-\arg H / \omega$. The **group delay** of chapter 05, the delay of the envelope
of a narrow-band signal, is larger there: 9.66 samples. The two differ because
IIR phase is not linear in frequency.

```rust
use audio_samples::operations::types::IirFilterDesign;
use audio_samples::{AudioIirFiltering, nzu, sample_rate, sine_wave};
use std::time::Duration;

let design = IirFilterDesign::butterworth_lowpass(nzu!(4), 1_000.0);
let tone = sine_wave::<f64>(1_000.0, Duration::from_millis(500), sample_rate!(16000), 1.0);
let once = tone.apply_iir_filter(&design).unwrap();
let twice = tone.filtfilt(&design).unwrap();
let (t, o, w) = (
    tone.as_slice().unwrap(),
    once.as_slice().unwrap(),
    twice.as_slice().unwrap(),
);

// Measure the middle of the signal, away from start-up and end transients.
let mid = 2_000..6_000;
let amp_db = |v: &[f64]| 20.0 * v[mid.clone()].iter().fold(0.0f64, |a, x| a.max(x.abs())).log10();
let corr = |v: &[f64], lag: usize| -> f64 { mid.clone().map(|n| v[n] * t[n - lag]).sum() };
let lag = |v: &[f64]| (0..16).max_by(|&a, &b| corr(v, a).total_cmp(&corr(v, b))).unwrap();

// Peak-picking a sampled sine can miss the true peak by 1 - cos(pi fc/fs)
// = 1.9 % (0.17 dB), so 0.2 dB is the tolerance.
assert!((amp_db(o) + 3.01).abs() < 0.2);
assert!((amp_db(w) + 6.02).abs() < 0.2);
assert_eq!(lag(o), 8); // half of the 16-sample period: -180 degrees
assert_eq!(lag(w), 0); // zero phase

// Group delay at 1 kHz, -d(arg H)/d(omega), by a central difference on the
// design's response. The ratio H(f + d) / H(f - d) avoids phase wrapping;
// a 0.01 Hz step makes the difference error negligible (< 1e-6 samples).
let (fs, d) = (16_000.0, 0.01);
let (m, ph) = design.frequency_response(&[1_000.0 - d, 1_000.0 + d], fs).unwrap();
let h = |i: usize| num_complex::Complex64::from_polar(m[i], ph[i]);
let group_delay = -(h(1) / h(0)).arg() / (2.0 * std::f64::consts::PI * 2.0 * d / fs);
assert!((group_delay - 9.66).abs() < 0.01); // samples, versus a phase delay of 8
```

<!-- figure: ch06-phase -->

**A 50 Hz hum notch.** Mains hum is a narrow-band tone; removing it requires a
filter with a zero exactly on the unit circle at $\omega_0 = 2\pi f_0/f_s$ and
poles just inside it at the same angle, so that the gain returns to unity
within a few hertz of $f_0$. That is the band-stop biquad of the RBJ *Audio EQ Cookbook*
(Bristow-Johnson), available as
[`EqBand::band_stop`](crate::operations::types::EqBand::band_stop):

$$H(z) = \frac{1 - 2\cos\omega_0\, z^{-1} + z^{-2}}{(1 + \alpha) - 2\cos\omega_0\, z^{-1} + (1 - \alpha) z^{-2}}, \qquad \alpha = \frac{\sin \omega_0}{2Q}$$

where $f_0$ is the notch frequency in Hz and $Q$ (dimensionless) is $f_0$
divided by the $-3$ dB bandwidth in Hz. The poles have radius
$r = \sqrt{(1-\alpha)/(1+\alpha)}$, and the closer $r$ is to 1, the narrower the
notch and the longer it takes to settle (time constant about $1/(1-r)$
samples).

```rust
use audio_samples::operations::types::EqBand;
use audio_samples::{AudioParametricEq, AudioSamples, sample_rate, sine_wave};
use num_complex::Complex64;
use std::f64::consts::PI;
use std::time::Duration;

let (fs, f0, q) = (16_000.0, 50.0, 10.0);

// The cookbook coefficients, evaluated on the unit circle.
let w0 = 2.0 * PI * f0 / fs;
let alpha = w0.sin() / (2.0 * q);
let (b, a) = ([1.0, -2.0 * w0.cos(), 1.0], [1.0 + alpha, -2.0 * w0.cos(), 1.0 - alpha]);
let gain = |f: f64| {
    let z1 = Complex64::from_polar(1.0, -2.0 * PI * f / fs); // z^-1
    ((b[0] + b[1] * z1 + b[2] * z1 * z1) / (a[0] + a[1] * z1 + a[2] * z1 * z1)).norm()
};
// An exact zero: |H(f0)| is 0 up to rounding (a 1e-16 numerator divided by a
// denominator of only 2 alpha sin(w0) = 4e-5, hence 1e-9).
assert!(gain(f0) < 1e-9);
// The -3 dB edges (|H|^2 = 1/2), found by bisection on each side of f0.
let edge = |mut lo: f64, mut hi: f64, rising: bool| {
    for _ in 0..60 {
        let mid = 0.5 * (lo + hi);
        if (gain(mid).powi(2) < 0.5) == rising { lo = mid } else { hi = mid }
    }
    0.5 * (lo + hi)
};
let bandwidth = edge(f0, 2.0 * f0, true) - edge(1.0, f0, false);
assert!((bandwidth - f0 / q).abs() < 0.01); // 5 Hz wide, as Q = 10 specifies
let r = ((1.0 - alpha) / (1.0 + alpha)).sqrt();
assert!((1.0 / (1.0 - r) - 1019.0).abs() < 1.0); // about 1019 samples = 64 ms to settle

// Apply it to 50 Hz hum plus a 1 kHz tone (in place of speech), 2 s long.
let dur = Duration::from_secs(2);
let hum = sine_wave::<f64>(50.0, dur, sample_rate!(16000), 0.5);
let tone = sine_wave::<f64>(1_000.0, dur, sample_rate!(16000), 0.5);
let mix: Vec<f64> = hum.as_slice().unwrap().iter().zip(tone.as_slice().unwrap()).map(|(p, q)| p + q).collect();
let mix = AudioSamples::new_mono(ndarray::Array1::from(mix), sample_rate!(16000)).unwrap();
let out = mix.apply_eq_band(&EqBand::band_stop(f0, q)).unwrap();

// Measure in the second half, after the notch has settled (16 time constants).
// 16 000 samples hold whole periods of both tones, so one DFT bin each is exact.
let tail = &out.as_slice().unwrap()[16_000..];
let amplitude = |f: f64| {
    let (re, im) = tail.iter().enumerate().fold((0.0, 0.0), |(re, im), (n, &v)| {
        let ph = 2.0 * PI * f * n as f64 / fs;
        (re + v * ph.cos(), im - v * ph.sin())
    });
    2.0 * (re * re + im * im).sqrt() / tail.len() as f64
};
let hum_db = 20.0 * (amplitude(50.0) / 0.5).log10();
let tone_db = 20.0 * (amplitude(1_000.0) / 0.5).log10();
// Measured: the hum is 160.5 dB down (what is left is the start-up
// transient, still decaying), the tone loses 0.0001 dB.
assert!(hum_db < -150.0);
assert!(tone_db.abs() < 1e-3);
```

Once settled, the notch removes the hum by more than 150 dB (160.5 dB here)
and leaves the 1 kHz tone unchanged to within 0.001 dB. A peaking cut
([`EqBand::peak`](crate::operations::types::EqBand::peak) with a negative
gain) has no zero on the unit circle, so it attenuates 50 Hz only by the
specified gain.

<!-- figure: ch06-notch -->

## How it works

The code below builds the same filter from the formulas in the Concept section
alone: place the analogue poles on a circle, pre-warp, apply the bilinear
map, fix the gain so that $H(1) = 1$, and compare with the library's poles
(the roots of each section's denominator) and response.

```rust
use audio_samples::nzu;
use audio_samples::operations::types::IirFilterDesign;
use num_complex::Complex64;
use std::f64::consts::PI;

let (fs, fc, order) = (16_000.0, 1_000.0, 4usize);

// 1. Analogue poles, pre-warped cutoff, bilinear map (K = 2 fs).
let k = 2.0 * fs;
let omega_c = k * (PI * fc / fs).tan();
let n = order as f64;
let poles: Vec<Complex64> = (0..order)
    .map(|i| {
        let s = Complex64::from_polar(omega_c, PI * (2.0 * i as f64 + n + 1.0) / (2.0 * n));
        (k + s) / (k - s)
    })
    .collect();
// 2. Zeros all at z = -1; choose the gain g so that H(1) = g 2^N / prod(1 - p) = 1.
let gain = poles.iter().map(|p| 1.0 - p).product::<Complex64>().re / 2f64.powi(order as i32);

// 3. The library's poles: roots of a0 z^2 + a1 z + a2 for every section.
let sos = IirFilterDesign::butterworth_lowpass(nzu!(4), fc).sos_coefficients(fs).unwrap();
let lib_poles: Vec<Complex64> = sos
    .iter()
    .flat_map(|s| {
        let d = Complex64::new(s[4] * s[4] - 4.0 * s[3] * s[5], 0.0).sqrt();
        [(-s[4] + d) / (2.0 * s[3]), (-s[4] - d) / (2.0 * s[3])]
    })
    .collect();
for p in &poles {
    // Every pole is inside the unit circle ...
    assert!(p.norm() < 1.0);
    // ... and matches a library pole. Both come from a few dozen flops on
    // O(1) numbers, so 1e-12 is a loose bound on rounding.
    assert!(lib_poles.iter().any(|q| (p - q).norm() < 1e-12));
}
// The overall gain is the product of the sections' b0.
let lib_gain: f64 = sos.iter().map(|s| s[0]).product();
assert!((gain - lib_gain).abs() / lib_gain < 1e-10);

// 4. Evaluate H(e^{jw}) = g (z+1)^N / prod(z - p) and compare.
let design = IirFilterDesign::butterworth_lowpass(nzu!(4), fc);
for f in [100.0, 1_000.0, 3_000.0, 7_000.0] {
    let z = Complex64::from_polar(1.0, 2.0 * PI * f / fs);
    let h = gain * (z + 1.0).powu(order as u32) / poles.iter().map(|p| z - p).product::<Complex64>();
    let (m, _) = design.frequency_response(&[f], fs).unwrap();
    assert!((h.norm() - m[0]).abs() < 1e-12);
}
```

<!-- figure: ch06-pole-zero -->

The difference equation itself is a five-term loop per section. Running it on
an impulse reproduces the library's filtered output to within rounding:

```rust
use audio_samples::operations::types::IirFilterDesign;
use audio_samples::{AudioIirFiltering, impulse, nzu, sample_rate};
use std::time::Duration;

let design = IirFilterDesign::butterworth_lowpass(nzu!(4), 1_000.0);
let sos = design.sos_coefficients(16_000.0).unwrap();
let x = impulse::<f64>(Duration::from_millis(20), sample_rate!(16000), 1.0, 0.0);
let lib = x.apply_iir_filter(&design).unwrap();

// Direct form I: y[n] = b0 x[n] + b1 x[n-1] + b2 x[n-2] - a1 y[n-1] - a2 y[n-2].
let mut y = x.as_slice().unwrap().to_vec();
for s in &sos {
    let (mut x1, mut x2, mut y1, mut y2) = (0.0, 0.0, 0.0, 0.0);
    for v in &mut y {
        let yn = (s[0] * *v + s[1] * x1 + s[2] * x2 - s[4] * y1 - s[5] * y2) / s[3];
        (x2, x1, y2, y1) = (x1, *v, y1, yn);
        *v = yn;
    }
}
// The library uses a different (transposed) form, so the rounding differs
// slightly; 1e-12 is far below anything audible.
let err = y.iter().zip(lib.as_slice().unwrap()).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
assert!(err < 1e-12);
// The impulse response sums to the DC gain, 1 (it has decayed within 20 ms).
assert!((y.iter().sum::<f64>() - 1.0).abs() < 1e-6);
```

## Ask the library to explain

[`IirFilterDesign`](crate::operations::types::IirFilterDesign) implements
[`Explain`](crate::educational::Explain). Explaining it through a signal
supplies the sample rate, so the explainer can design the filter, tabulate the
poles and zeros of every section, and measure the gain at the cutoff.

```rust
use audio_samples::educational::ExplainFor;
use audio_samples::operations::types::IirFilterDesign;
use audio_samples::{nzu, sample_rate, sine_wave};
use std::time::Duration;

let audio = sine_wave::<f64>(440.0, Duration::from_millis(50), sample_rate!(16000), 0.5);
let doc = audio.explain(&IirFilterDesign::butterworth_lowpass(nzu!(4), 1_000.0));
println!("{doc}");
assert_eq!(doc.quantity("Sections").unwrap().value, 2.0);
assert!((doc.quantity("Measured gain at the cutoff").unwrap().value + 3.0103).abs() < 1e-3);
// Largest pole radius 0.862856 (the pair nearest the cutoff).
assert!((doc.quantity("Largest pole radius").unwrap().value - 0.862856).abs() < 1e-6);

// A low cutoff relative to fs moves the poles towards z = 1. At 10 Hz with
// order 8 the stability margin drops below 1e-3 and the explainer warns.
let doc = audio.explain(&IirFilterDesign::butterworth_lowpass(nzu!(8), 10.0));
let margin = doc.quantity("Stability margin").unwrap().value;
assert!(margin > 0.0 && margin < 1e-3);
assert!(doc.has_warning_containing("very sensitive to rounding"));
```

## Exercises

1. **Higher order, same cutoff gain.** Predict the gain of an 8th-order
   Butterworth low-pass (1 kHz, 16 kHz) at 1 kHz and at 2 kHz, then check with
   `frequency_response`.
2. **Chebyshev versus Butterworth.** For a 4th-order Chebyshev I low-pass with
   1 dB ripple at 1 kHz (16 kHz), use the Chebyshev magnitude formula of the
   Concept section to predict the gain at 1 kHz, the largest and smallest gain
   between 0 and 1 kHz, and the attenuation at 2 kHz. Compare the attenuation
   at 2 kHz with the Butterworth's.
3. **Zeros of a high-pass.** The low-pass zeros are at $z = -1$
   (Nyquist). Predict where the zeros of a 4th-order Butterworth *high-pass*
   lie, then find them from `sos_coefficients`.
4. **filtfilt doubles the decibels.** Predict the attenuation of a 2 kHz tone
   by `filtfilt` with the 4th-order, 1 kHz Butterworth, then measure it.

## Answers

1. The cutoff gain is $-3.01$ dB for every order (this is the definition of
   $f_c$). At 2 kHz the closed form gives
   $10\log_{10}(1 + 2.0824^{16}) = 51.0$ dB of attenuation, about twice the
   4th-order figure.

```rust
use audio_samples::nzu;
use audio_samples::operations::types::IirFilterDesign;
use std::f64::consts::PI;

let (fs, fc) = (16_000.0, 1_000.0);
let (mag, _) = IirFilterDesign::butterworth_lowpass(nzu!(8), fc)
    .frequency_response(&[fc, 2.0 * fc], fs)
    .unwrap();
assert!((20.0 * mag[0].log10() + 3.0103).abs() < 1e-4);
let ratio = (PI * 2.0 * fc / fs).tan() / (PI * fc / fs).tan(); // 2.0824
let predicted = -10.0 * (1.0 + ratio.powi(16)).log10();
assert!((20.0 * mag[1].log10() - predicted).abs() < 1e-9);
assert!((predicted + 51.0).abs() < 0.05);
```

2. At $f_c$ the Chebyshev gain is $-r_p = -1$ dB. With an even order the gain
   at DC is also $-1$ dB ($T_4(0) = 1$) and the ripple peaks ($T_4 = 0$) reach
   0 dB. At 2 kHz, $x = \tan(\pi/8)/\tan(\pi/16) = 2.0824$,
   $T_4(x) = \cosh(4\,\mathrm{arccosh}\,x) = 116.7$ and
   $\varepsilon^2 = 10^{0.1} - 1 = 0.2589$, so the attenuation is
   $10\log_{10}(1 + \varepsilon^2 T_4^2) = 35.5$ dB, 10 dB more than the
   Butterworth's 25.5 dB.

```rust
use audio_samples::nzu;
use audio_samples::operations::types::{FilterResponse, IirFilterDesign};

let cheb = IirFilterDesign::chebyshev_i(FilterResponse::LowPass, nzu!(4), 1_000.0, 1.0);
let grid: Vec<f64> = (0..=1000).map(|i| i as f64).collect(); // 0..1000 Hz, 1 Hz steps
let (m, _) = cheb.frequency_response(&grid, 16_000.0).unwrap();
let db: Vec<f64> = m.iter().map(|v| 20.0 * v.log10()).collect();
assert!((db[1000] + 1.0).abs() < 1e-6); // -1 dB at fc
assert!((db[0] + 1.0).abs() < 1e-6); // -1 dB at DC (even order)
let max = db.iter().cloned().fold(f64::MIN, f64::max);
assert!(max.abs() < 1e-3); // ripple peaks reach 0 dB (to the 1 Hz grid)
// The formula's prediction at 2 kHz, and the measurement.
let x = (std::f64::consts::PI / 8.0).tan() / (std::f64::consts::PI / 16.0).tan();
let t4 = (4.0 * x.acosh()).cosh();
let eps2 = 10f64.powf(0.1) - 1.0;
let predicted = 10.0 * (1.0 + eps2 * t4 * t4).log10();
assert!((x - 2.0824).abs() < 1e-4 && (t4 - 116.7).abs() < 0.05 && (eps2 - 0.2589).abs() < 1e-4);
assert!((predicted - 35.5).abs() < 0.05);
let (m2, _) = cheb.frequency_response(&[2_000.0], 16_000.0).unwrap();
assert!((20.0 * m2[0].log10() + predicted).abs() < 1e-6); // the design obeys the formula
```

3. A high-pass must block DC, so its zeros move to $z = +1$ (all four of
   them); the bilinear transform maps the analogue zeros at $s = 0$ there.

```rust
use audio_samples::nzu;
use audio_samples::operations::types::IirFilterDesign;

let sos = IirFilterDesign::butterworth_highpass(nzu!(4), 1_000.0).sos_coefficients(16_000.0).unwrap();
let mut zeros = 0;
for s in &sos {
    // b0 z^2 + b1 z + b2 = b0 (z - 1)^2 means b1 = -2 b0 and b2 = b0.
    assert!((s[1] + 2.0 * s[0]).abs() < 1e-12 && (s[2] - s[0]).abs() < 1e-12);
    zeros += 2;
}
assert_eq!(zeros, 4);
```

4. The forward pass attenuates by 25.5 dB and so does the backward pass:
   51.0 dB in total ($|H|^2$).

```rust
use audio_samples::operations::types::IirFilterDesign;
use audio_samples::{AudioIirFiltering, nzu, sample_rate, sine_wave};
use std::time::Duration;

let design = IirFilterDesign::butterworth_lowpass(nzu!(4), 1_000.0);
let tone = sine_wave::<f64>(2_000.0, Duration::from_millis(500), sample_rate!(16000), 1.0);
let out = tone.filtfilt(&design).unwrap();
let peak = out.as_slice().unwrap()[2_000..6_000].iter().fold(0.0f64, |a, x| a.max(x.abs()));
// 2 kHz is 8 samples per period, so the sampled peak can miss the true
// peak by 1 - cos(pi/8) = 7.6 % (0.69 dB) unless a sample falls on it; the
// tone starts at phase 0, so samples fall on the peaks and 0.1 dB is enough.
assert!((20.0 * peak.log10() + 51.0).abs() < 0.1);
```

## Common pitfalls

- **Cutoff conventions.** Here the cutoff is the $-3.01$ dB
  point for Butterworth, the $-r_p$ point for Chebyshev I and the start of the
  stop band for Chebyshev II. Some textbooks quote the Chebyshev $-3$ dB
  frequency instead.
- **Frequencies are in Hz.** Some design routines expect the cutoff as a
  fraction of Nyquist; this library always takes Hz and a sample rate.
- **Never cascade a high-order filter as one polynomial.** Use the SOS rows.
  Each row is `[b0, b1, b2, a0, a1, a2]` with the overall gain folded into the
  sections; tools that keep a separate gain `g` normalise differently.
- **`filtfilt` doubles the attenuation in dB** and is non-causal; use it for
  offline analysis, not real-time processing.
- **Low cutoffs at high sample rates** put poles close to $z = 1$, which causes
  long ringing and sensitivity to rounding. Decimate first, or lower the order.
- The trait method `AudioIirFiltering::frequency_response` is deprecated (it
  returns a flat response); use
  [`IirFilterDesign::frequency_response`](crate::operations::types::IirFilterDesign::frequency_response)
  or [`SosFilter::frequency_response`](crate::SosFilter::frequency_response).

## References

- A. V. Oppenheim and R. W. Schafer, *Discrete-Time Signal Processing*, 3rd ed., Pearson, 2010. The bilinear transform, Butterworth and Chebyshev design.
- R. Bristow-Johnson, *Cookbook formulae for audio equalizer biquad filter coefficients* (the "Audio EQ Cookbook"). The notch (band-stop) biquad used above.
- P. Virtanen et al., "SciPy 1.0: fundamental algorithms for scientific computing in Python", *Nature Methods* 17, 261-272, 2020. Describes the `iirfilter` design pipeline (prototype, frequency transform, bilinear transform, second-order sections) used in this chapter.

## See also

- [`IirFilterDesign`](crate::operations::types::IirFilterDesign),
  [`IirFilterDesign::sos_coefficients`](crate::operations::types::IirFilterDesign::sos_coefficients),
  [`IirFilterDesign::frequency_response`](crate::operations::types::IirFilterDesign::frequency_response)
- [`AudioIirFiltering::apply_iir_filter`](crate::AudioIirFiltering::apply_iir_filter),
  [`AudioIirFiltering::filtfilt`](crate::AudioIirFiltering::filtfilt),
  [`SosFilter`](crate::SosFilter) (streaming, block by block)
- [`EqBand::band_stop`](crate::operations::types::EqBand::band_stop),
  [`EqBand::peak`](crate::operations::types::EqBand::peak),
  [`AudioParametricEq::apply_eq_band`](crate::AudioParametricEq::apply_eq_band)
  (RBJ cookbook biquads)
- [`Explain`](crate::educational::Explain), [`ExplainFor`](crate::educational::ExplainFor)
- Previous chapter: [05. Convolution and FIR filters](05_convolution_fir.md);
  next chapter: [07. Resampling](07_resampling.md)
- Lab: `examples/tutorial_06_iir_filters.rs`
  (`cargo run --release --example tutorial_06_iir_filters --features educational,full_no_plotting`)
