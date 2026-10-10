# 08. LPC, the source-filter model and formants

A vowel is a buzz shaped by a tube. The vocal folds produce a periodic train of
air pulses (the *source*), and the throat, mouth and lips act as a resonant
filter whose peaks, the *formants*, decide which vowel we hear. Linear
predictive coding (LPC) fits an all-pole filter to a short frame of speech, so
it hands us the filter half of that model: its roots are the formants, and a
quantised form of its coefficients (line spectral frequencies) is what speech
codecs such as SILK transmit.

## Concept

**Source-filter model.** A voiced frame is modelled as

$$X(z) = E(z) \\, H(z), \qquad H(z) = \frac{G}{A(z)}, \qquad A(z) = 1 + \sum_{k=1}^{p} a_k z^{-k}$$

where $E(z)$ is the excitation (here an impulse train with one pulse every
$1/f_0$ seconds, $f_0$ the fundamental frequency in Hz), $H(z)$ the vocal-tract
filter, $G$ a gain, $p$ the model order and $a_k$ the (dimensionless) predictor
coefficients. The usual choice of gain is $G = \sqrt{E_p}$, the square root of
the order-$p$ prediction-error energy defined below, so that the model driven
by a unit-energy excitation has the frame's residual energy. This crate writes $A(z)$ with a **plus** sign, as MATLAB and
`librosa.lpc` do; many textbooks predict $\hat{x}\[n\] = \sum_k \alpha_k x\[n-k\]$
instead, so $\alpha_k = -a_k$.

**Formants as poles.** One formant at frequency $F$ (Hz) with 3 dB bandwidth $B$
(Hz) is a pair of complex-conjugate poles at radius $r$ and angle $\theta$:

$$r \approx e^{-\pi B / f_s}, \qquad \theta = \frac{2 \pi F}{f_s}$$

where $f_s$ is the sample rate in Hz and $\theta$ is in radians per sample. The
radius formula is a narrow-band approximation: it holds well for an isolated
pole pair with $B \ll f_s$ and $F$ away from 0 and $f_s/2$, and this chapter uses
it as the definition of $B$ when going both ways. Each
pole pair contributes a second-order factor $1 - 2 r \cos\theta \\, z^{-1} + r^2 z^{-2}$
to $A(z)$. Inverting: a root $z_i$ of $A(z)$ with $\mathrm{Im}\\, z_i > 0$ gives

$$F_i = \frac{f_s \arg z_i}{2\pi}, \qquad B_i = -\frac{f_s \ln |z_i|}{\pi}$$

**Linear prediction.** The prediction error (residual) is
$e\[n\] = x\[n\] + \sum_{k=1}^{p} a_k x\[n-k\]$. Minimising its energy over a
windowed frame gives the *normal* (Yule-Walker) equations

$$\sum_{k=1}^{p} a_k R\[|i-k|\] = -R\[i\], \qquad i = 1, \dots, p$$

with the biased autocorrelation $R\[m\] = \sum_{n=0}^{N-1-m} x_w\[n\] \\, x_w\[n+m\]$ of the
windowed frame $x_w\[n\] = w\[n\] x\[n\]$ ($N$ samples, $w$ a Hamming window).
(The biased estimator is usually written with a factor $1/N$; it scales every
$R\[m\]$ equally and cancels from the solution, so the crate omits it.) The
matrix is Toeplitz, so the **Levinson-Durbin** recursion solves it in $O(p^2)$
operations, one order at a time ($E_0 = R\[0\]$, $m = 1, \dots, p$):

$$k_m = -\frac{R\[m\] + \sum_{i=1}^{m-1} a_i^{(m-1)} R\[m-i\]}{E_{m-1}}, \qquad a_i^{(m)} = a_i^{(m-1)} + k_m a_{m-i}^{(m-1)}, \qquad a_m^{(m)} = k_m, \qquad E_m = (1 - k_m^2) E_{m-1}$$

where $k_m$ is the reflection coefficient ($|k_m| < 1$ for a stable filter) and
$E_m$ the prediction-error energy of the order-$m$ predictor.

**Choosing p.** Each formant needs two poles and there is roughly one formant
per kilohertz of bandwidth; two more poles absorb the spectral tilt of the
source and lip radiation. Hence the rule of thumb $p \approx f_s/1000 + 2$
($f_s$ in Hz), 18 at 16 kHz.

**Pre-emphasis.** Real glottal pulses fall off at about -12 dB per octave, and
radiation from the lips adds about +6 dB per octave, so voiced speech has a net
tilt of about -6 dB per octave: high formants are weak and the fit spends poles
on the tilt. The first-order high-pass (FIR) $y\[n\] = x\[n\] - \alpha x\[n-1\]$
with $\alpha \approx 0.97$ adds about +6 dB per octave and so flattens that net
tilt before analysis. (Exercise 3 uses a source with the full -12 dB per octave
and no radiation term, so a -6 dB per octave tilt remains after pre-emphasis.)

The crate's LPC routines live in the speech-codec module and are exported from
the crate root when the `opus-codec` feature is on (it is part of
`full_no_plotting`): [`lpc_analysis`](crate::lpc_analysis) (Hamming window,
autocorrelation, Levinson-Durbin) and [`levinson_durbin`](crate::levinson_durbin).
There is no first-class LPC or formant API outside the codec feature, and no
root finder, so this chapter writes the formant step itself.

## With the library

We synthesise an /a/-like vowel at 16 kHz: formants (700, 80), (1220, 90) and
(2600, 120) Hz as (F, B), excited by an impulse train at $f_0 = 120$ Hz, and
analyse a 512-sample (32 ms) frame taken after the filter has settled.

```rust
use audio_samples::{educational::topics, lpc_analysis, lpc_residual};
use std::f64::consts::PI;

let fs = 16_000.0;
let formants = [(700.0, 80.0), (1220.0, 90.0), (2600.0, 120.0)];

// A(z) as the product of one second-order section per formant.
let mut a_true = vec![1.0f64];
for &(f, b) in &formants {
    let (r, theta) = ((-PI * b / fs).exp(), 2.0 * PI * f / fs);
    let sec = [1.0, -2.0 * r * theta.cos(), r * r];
    let mut out = vec![0.0; a_true.len() + 2];
    for (i, ai) in a_true.iter().enumerate() {
        for (j, sj) in sec.iter().enumerate() {
            out[i + j] += ai * sj;
        }
    }
    a_true = out;
}
assert_eq!(a_true.len(), 7); // three pole pairs: order 6

// Source: one impulse every round(16000/120) = 133 samples. Filter: 1/A(z).
let n = 2400;
let mut y = vec![0.0f64; n];
for i in 0..n {
    let e = if i % 133 == 0 { 1.0 } else { 0.0 };
    y[i] = e - (1..7).filter(|&k| k <= i).map(|k| a_true[k] * y[i - k]).sum::<f64>();
}
let frame: Vec<f32> = y[1600..2112].iter().map(|&v| v as f32).collect();

let p = topics::lpc(18, fs).quantity("Rule-of-thumb order").unwrap().value as usize;
assert_eq!(p, 18);
let lpc = lpc_analysis(&frame, p);
assert_eq!(lpc.coeffs.len(), 18); // the order-p solution: a_1 ... a_18
println!("a_1..a_3 = {:?}, E_p = {:.2}", &lpc.coeffs[..3], lpc.prediction_error);

// Inverse filtering with A(z) leaves the residual e[n]: little more than the
// pulses of the source. The prediction gain compares the energies.
let e = lpc_residual(&frame, &lpc);
let energy = |v: &[f32]| v.iter().map(|&s| f64::from(s).powi(2)).sum::<f64>();
let gain_db = 10.0 * (energy(&frame) / energy(&e)).log10();
println!("prediction gain {gain_db:.1} dB");
assert!((gain_db - 13.8).abs() < 0.05, "prediction gain {gain_db}");
```

`lpc_analysis` returns an [`LpcCoefficients`](crate::LpcCoefficients) holding
$a_1, \dots, a_p$ (as `f32`, index 0 is $a_1$) and the final prediction-error
energy $E_p$. Filtering the frame with $A(z)$ ([`lpc_residual`](crate::lpc_residual))
removes 13.8 dB of its energy: what is left is essentially the excitation
pulses, which a predictor built from the previous 18 samples cannot foresee.

## Under the hood

**Levinson-Durbin, written out.** We rebuild the windowed autocorrelation,
apply the same tiny diagonal loading the library uses ($R\[0\] \leftarrow (1 + 10^{-5}) R\[0\]$),
run the recursion and compare. We also check the defining property: the
coefficients satisfy all $p$ normal equations (an order-$(p-1)$ answer would not).

```rust
# use audio_samples::{levinson_durbin, lpc_analysis};
# use std::f64::consts::PI;
# let fs = 16_000.0;
# let mut a_true = vec![1.0f64];
# for &(f, b) in &[(700.0, 80.0), (1220.0, 90.0), (2600.0f64, 120.0f64)] {
#     let (r, theta) = ((-PI * b / fs).exp(), 2.0 * PI * f / fs);
#     let sec = [1.0, -2.0 * r * theta.cos(), r * r];
#     let mut out = vec![0.0; a_true.len() + 2];
#     for (i, ai) in a_true.iter().enumerate() { for (j, sj) in sec.iter().enumerate() { out[i + j] += ai * sj; } }
#     a_true = out;
# }
# let mut y = vec![0.0f64; 2400];
# for i in 0..2400 {
#     let e = if i % 133 == 0 { 1.0 } else { 0.0 };
#     y[i] = e - (1..7).filter(|&k| k <= i).map(|k| a_true[k] * y[i - k]).sum::<f64>();
# }
# let frame: Vec<f32> = y[1600..2112].iter().map(|&v| v as f32).collect();
let p = 18;
let n = frame.len();
// Hamming-windowed frame (the library rounds it to f32 before correlating).
let xw: Vec<f64> = (0..n)
    .map(|i| {
        let w = 0.54 - 0.46 * (2.0 * PI * i as f64 / (n - 1) as f64).cos();
        f64::from((w * f64::from(frame[i])) as f32)
    })
    .collect();
let r_raw: Vec<f64> = (0..=p)
    .map(|m| (0..n - m).map(|i| xw[i] * xw[i + m]).sum())
    .collect();
let mut r = r_raw.clone();
r[0] *= 1.0 + 1e-5;

// Levinson-Durbin for A(z) = 1 + sum a_k z^-k.
let mut a = vec![0.0f64; p + 1];
let mut err = r[0];
for m in 1..=p {
    let k = -(r[m] + (1..m).map(|i| a[i] * r[m - i]).sum::<f64>()) / err;
    let prev = a.clone();
    for i in 1..m {
        a[i] = prev[i] + k * prev[m - i];
    }
    a[m] = k;
    err *= 1.0 - k * k;
}

let lib = lpc_analysis(&frame, p);
let largest = a.iter().fold(0.0f64, |m, v| m.max(v.abs()));
let diff = (1..=p).map(|k| (a[k] - f64::from(lib.coeffs[k - 1])).abs()).fold(0.0, f64::max);
// The library stores f32 coefficients (relative precision 6e-8), so 1e-6 of
// the largest coefficient is a safe bound.
assert!(diff < 1e-6 * largest, "difference {diff}");
assert!((lib.prediction_error - err).abs() < 1e-9 * err);
// The public recursion on our autocorrelation agrees as well.
let ld = levinson_durbin(&r_raw, p).unwrap();
assert!((1..=p).all(|k| (a[k] - f64::from(ld.coeffs[k - 1])).abs() < 1e-6 * largest));

// Order-p check: every normal equation holds to f32 precision.
let worst = (1..=p)
    .map(|i| ((1..=p).map(|k| f64::from(lib.coeffs[k - 1]) * r[i.abs_diff(k)]).sum::<f64>() + r[i]).abs() / r[0])
    .fold(0.0, f64::max);
assert!(worst < 1e-6, "residual {worst}");
```

**From roots to formants.** $A(z)$ times $z^p$ is the polynomial
$z^p + a_1 z^{p-1} + \dots + a_p$. The Durand-Kerner (Weierstrass) iteration
updates all $p$ root estimates at once,
$z_i \leftarrow z_i - P(z_i) / \prod_{j \ne i} (z_i - z_j)$, and converges for
polynomials with distinct roots such as these. Each root in the upper half
plane is then converted with the $F_i$, $B_i$ formulas above. LPC also places
broad poles (bandwidths of a kilohertz or more) that shape the overall slope;
we keep only resonances narrower than 400 Hz.

```rust
# use audio_samples::lpc_analysis;
# use std::f64::consts::PI;
use num_complex::Complex64;
# let fs = 16_000.0;
# let formants = [(700.0, 80.0), (1220.0, 90.0), (2600.0f64, 120.0f64)];
# let mut a_true = vec![1.0f64];
# for &(f, b) in &formants {
#     let (r, theta) = ((-PI * b / fs).exp(), 2.0 * PI * f / fs);
#     let sec = [1.0, -2.0 * r * theta.cos(), r * r];
#     let mut out = vec![0.0; a_true.len() + 2];
#     for (i, ai) in a_true.iter().enumerate() { for (j, sj) in sec.iter().enumerate() { out[i + j] += ai * sj; } }
#     a_true = out;
# }
# let mut y = vec![0.0f64; 2400];
# for i in 0..2400 {
#     let e = if i % 133 == 0 { 1.0 } else { 0.0 };
#     y[i] = e - (1..7).filter(|&k| k <= i).map(|k| a_true[k] * y[i - k]).sum::<f64>();
# }
# let frame: Vec<f32> = y[1600..2112].iter().map(|&v| v as f32).collect();

/// Roots of z^p + c[1] z^(p-1) + ... + c[p] (c[0] = 1), Durand-Kerner.
fn poly_roots(c: &[f64]) -> Vec<Complex64> {
    let p = c.len() - 1;
    let eval = |z: Complex64| c.iter().fold(Complex64::new(0.0, 0.0), |acc, &ci| acc * z + ci);
    let mut z: Vec<Complex64> = (0..p)
        .map(|i| Complex64::from_polar(0.9, 2.0 * PI * i as f64 / p as f64 + 0.4))
        .collect();
    for _ in 0..5000 {
        let mut biggest = 0.0f64;
        for i in 0..p {
            let den = (0..p).filter(|&j| j != i).fold(Complex64::new(1.0, 0.0), |d, j| d * (z[i] - z[j]));
            let step = eval(z[i]) / den;
            z[i] -= step;
            biggest = biggest.max(step.norm());
        }
        if biggest < 1e-14 {
            break;
        }
    }
    z
}

let lpc = lpc_analysis(&frame, 18);
let mut a = vec![1.0];
a.extend(lpc.coeffs.iter().map(|&v| f64::from(v)));
let roots = poly_roots(&a);
assert!(roots.iter().all(|z| z.norm() < 1.0)); // all poles inside the unit circle

let mut resonances: Vec<(f64, f64)> = roots
    .iter()
    .filter(|z| z.im > 0.0)
    .map(|z| (z.arg() * fs / (2.0 * PI), -z.norm().ln() * fs / PI))
    .filter(|&(_, b)| b < 400.0)
    .collect();
resonances.sort_by(|x, y| x.0.total_cmp(&y.0));
println!("narrow resonances (F, B): {resonances:.1?}");
assert_eq!(resonances.len(), 3);
for (&(f_est, _), &(f_true, _)) in resonances.iter().zip(&formants) {
    // A periodic source samples the envelope only at multiples of 120 Hz,
    // which biases autocorrelation LPC by a few hertz; 20 Hz (under 3 % of
    // F1) is a safe bound.
    assert!((f_est - f_true).abs() < 20.0, "{f_est} vs {f_true}");
}
let errors: Vec<f64> = resonances.iter().zip(&formants).map(|(e, t)| (e.0 - t.0).round()).collect();
assert_eq!(errors, vec![11.0, -9.0, -3.0]);
```

**The LPC envelope and the FFT.** $|G/A(e^{j\omega})|$ is a smooth curve; the
FFT of the frame is a comb of harmonics at multiples of $f_0$. If LPC has done
its job, the harmonic peaks sit on the envelope up to a constant gain. The lab
measures this for harmonics 1 to 33 (up to 3960 Hz): the peaks scatter about
the shifted envelope with a standard deviation of 1.06 dB (asserted below
1.5 dB in the lab, which also writes the curves to a CSV file).

## Ask the library to explain

```rust
use audio_samples::educational::{WarningLevel, topics};

let doc = topics::lpc(18, 16_000.0);
println!("{doc}");
assert_eq!(doc.quantity("Rule-of-thumb order").unwrap().value, 18.0);
assert_eq!(doc.quantity("Resonances representable").unwrap().value, 9.0);
assert!(doc.has_warning_containing("silently lowers the order"));

// An order-4 model is far below the rule of thumb.
assert!(topics::lpc(4, 16_000.0).has_warning(WarningLevel::Caution));
```

The explanation prints the Levinson-Durbin formulas in the crate's sign
convention and warns that `lpc_analysis` quietly lowers the order to half the
frame length for very short frames.

## Exercises

1. **Pole radius.** Predict the pole radius of the 700 Hz formant
   ($B = 80$ Hz) at $f_s = 16$ kHz. What radius gives the same bandwidth at
   $f_s = 8$ kHz?
2. **How many poles?** For the frame above, try orders 4, 6, 8, 10 and 12.
   Predict the smallest order that recovers all three formants (within 150 Hz,
   with $B < 400$ Hz), then check. Why is the answer larger than $2 \times 3$?
3. **Pre-emphasis.** Pass the impulse train through two real poles at
   $z = 0.95$ (a glottal-like tilt) before the vocal tract. At order 18, is the
   2600 Hz formant still found without pre-emphasis? With
   $y\[n\] = x\[n\] - 0.97 x\[n-1\]$?
4. **Rule of thumb.** What order does $f_s/1000 + 2$ suggest at 8 kHz and at
   44.1 kHz?

## Answers

**1.** $r = e^{-\pi \cdot 80/16000} = 0.98441$; at 8 kHz, $r = e^{-\pi \cdot 80/8000} = 0.96907$
(halving $f_s$ doubles $B/f_s$, so the pole moves away from the unit circle).

```rust
let r16 = (-std::f64::consts::PI * 80.0 / 16_000.0).exp();
let r8 = (-std::f64::consts::PI * 80.0 / 8_000.0).exp();
assert!((r16 - 0.98441).abs() < 5e-6);
assert!((r8 - 0.96907).abs() < 5e-6);
```

**2.** Order 10. Orders 4 and 6 find no narrow resonance and order 8 finds only
two. Six poles would suffice in the limit of a long, noise-excited frame, but here
the Hamming window, the short frame and the harmonic excitation leave some
energy that LPC can only absorb with extra (broad) poles, exactly the slack the
"+2" in the rule of thumb provides.

```rust
# use audio_samples::lpc_analysis;
# use num_complex::Complex64;
# use std::f64::consts::PI;
# let fs = 16_000.0;
# let formants = [(700.0, 80.0), (1220.0, 90.0), (2600.0f64, 120.0f64)];
# let mut a_true = vec![1.0f64];
# for &(f, b) in &formants {
#     let (r, theta) = ((-PI * b / fs).exp(), 2.0 * PI * f / fs);
#     let sec = [1.0, -2.0 * r * theta.cos(), r * r];
#     let mut out = vec![0.0; a_true.len() + 2];
#     for (i, ai) in a_true.iter().enumerate() { for (j, sj) in sec.iter().enumerate() { out[i + j] += ai * sj; } }
#     a_true = out;
# }
# let mut y = vec![0.0f64; 2400];
# for i in 0..2400 {
#     let e = if i % 133 == 0 { 1.0 } else { 0.0 };
#     y[i] = e - (1..7).filter(|&k| k <= i).map(|k| a_true[k] * y[i - k]).sum::<f64>();
# }
# let frame: Vec<f32> = y[1600..2112].iter().map(|&v| v as f32).collect();
# fn poly_roots(c: &[f64]) -> Vec<Complex64> {
#     let p = c.len() - 1;
#     let eval = |z: Complex64| c.iter().fold(Complex64::new(0.0, 0.0), |acc, &ci| acc * z + ci);
#     let mut z: Vec<Complex64> = (0..p).map(|i| Complex64::from_polar(0.9, 2.0 * PI * i as f64 / p as f64 + 0.4)).collect();
#     for _ in 0..5000 {
#         let mut biggest = 0.0f64;
#         for i in 0..p {
#             let den = (0..p).filter(|&j| j != i).fold(Complex64::new(1.0, 0.0), |d, j| d * (z[i] - z[j]));
#             let step = eval(z[i]) / den;
#             z[i] -= step;
#             biggest = biggest.max(step.norm());
#         }
#         if biggest < 1e-14 { break; }
#     }
#     z
# }
let found = |p: usize| -> usize {
    let mut a = vec![1.0];
    a.extend(lpc_analysis(&frame, p).coeffs.iter().map(|&v| f64::from(v)));
    let res: Vec<(f64, f64)> = poly_roots(&a)
        .iter()
        .filter(|z| z.im > 0.0)
        .map(|z| (z.arg() * fs / (2.0 * PI), -z.norm().ln() * fs / PI))
        .filter(|&(_, b)| b < 400.0)
        .collect();
    formants.iter().filter(|&&(f, _)| res.iter().any(|&(fe, _)| (fe - f).abs() < 150.0)).count()
};
let counts: Vec<usize> = [4, 6, 8, 10, 12].iter().map(|&p| found(p)).collect();
assert_eq!(counts, vec![0, 0, 2, 3, 3]);
```

**3.** Without pre-emphasis the pole nearest 2600 Hz moves to 2701 Hz and
broadens to $B \approx 729$ Hz: the fit spends its accuracy on the strong low
frequencies and F3 is lost. After pre-emphasis all three formants are back
within 30 Hz (709, 1219 and 2577 Hz).

```rust
# use audio_samples::lpc_analysis;
# use num_complex::Complex64;
# use std::f64::consts::PI;
# let fs = 16_000.0;
# let formants = [(700.0, 80.0), (1220.0, 90.0), (2600.0f64, 120.0f64)];
# let mut a_true = vec![1.0f64];
# for &(f, b) in &formants {
#     let (r, theta) = ((-PI * b / fs).exp(), 2.0 * PI * f / fs);
#     let sec = [1.0, -2.0 * r * theta.cos(), r * r];
#     let mut out = vec![0.0; a_true.len() + 2];
#     for (i, ai) in a_true.iter().enumerate() { for (j, sj) in sec.iter().enumerate() { out[i + j] += ai * sj; } }
#     a_true = out;
# }
# fn poly_roots(c: &[f64]) -> Vec<Complex64> {
#     let p = c.len() - 1;
#     let eval = |z: Complex64| c.iter().fold(Complex64::new(0.0, 0.0), |acc, &ci| acc * z + ci);
#     let mut z: Vec<Complex64> = (0..p).map(|i| Complex64::from_polar(0.9, 2.0 * PI * i as f64 / p as f64 + 0.4)).collect();
#     for _ in 0..5000 {
#         let mut biggest = 0.0f64;
#         for i in 0..p {
#             let den = (0..p).filter(|&j| j != i).fold(Complex64::new(1.0, 0.0), |d, j| d * (z[i] - z[j]));
#             let step = eval(z[i]) / den;
#             z[i] -= step;
#             biggest = biggest.max(step.norm());
#         }
#         if biggest < 1e-14 { break; }
#     }
#     z
# }
# let resonances = |frame: &[f32]| -> Vec<(f64, f64)> {
#     let mut a = vec![1.0];
#     a.extend(lpc_analysis(frame, 18).coeffs.iter().map(|&v| f64::from(v)));
#     poly_roots(&a).iter().filter(|z| z.im > 0.0)
#         .map(|z| (z.arg() * fs / (2.0 * PI), -z.norm().ln() * fs / PI)).collect()
# };
// Glottal-like source: impulse train through 1/(1 - 0.95 z^-1)^2, then 1/A(z).
let n = 2400;
let (mut g1, mut g2, mut y) = (vec![0.0f64; n], vec![0.0f64; n], vec![0.0f64; n]);
for i in 0..n {
    let e = if i % 133 == 0 { 1.0 } else { 0.0 };
    g1[i] = e + if i > 0 { 0.95 * g1[i - 1] } else { 0.0 };
    g2[i] = g1[i] + if i > 0 { 0.95 * g2[i - 1] } else { 0.0 };
    y[i] = g2[i] - (1..7).filter(|&k| k <= i).map(|k| a_true[k] * y[i - k]).sum::<f64>();
}
let raw: Vec<f32> = y[1600..2112].iter().map(|&v| v as f32).collect();
let pre: Vec<f32> = (1600..2112).map(|i| (y[i] - 0.97 * y[i - 1]) as f32).collect();

let nearest = |res: &[(f64, f64)], f: f64| {
    *res.iter().min_by(|x, y| (x.0 - f).abs().total_cmp(&(y.0 - f).abs())).unwrap()
};
let (f3_raw, b3_raw) = nearest(&resonances(&raw), 2600.0);
assert!(b3_raw > 400.0); // F3 is smeared out without pre-emphasis
assert_eq!((f3_raw.round(), b3_raw.round()), (2701.0, 729.0));
let with_pre = resonances(&pre);
let mut estimates = Vec::new();
for &(f, _) in &formants {
    let (fe, be) = nearest(&with_pre, f);
    assert!((fe - f).abs() < 30.0 && be < 400.0, "{f} -> {fe} (B {be})");
    estimates.push(fe.round());
}
assert_eq!(estimates, vec![709.0, 1219.0, 2577.0]);
```

**4.** 10 at 8 kHz (telephone speech; SILK narrowband also uses 10) and
$44.1 + 2 = 46.1$, so about 46, at 44.1 kHz, which is why speech is usually
downsampled before LPC.

```rust
use audio_samples::educational::topics;
let q = |fs: f64| topics::lpc(10, fs).quantity("Rule-of-thumb order").unwrap().value;
assert_eq!(q(8_000.0), 10.0);
assert!((q(44_100.0) - 46.1).abs() < 1e-9);
```

## Common pitfalls

- **Sign convention.** The crate (like MATLAB and librosa) stores $A(z)$
  coefficients, $e\[n\] = x\[n\] + \sum a_k x\[n-k\]$; the "predictor" form uses
  $-a_k$.
- **Not librosa's algorithm.** `librosa.lpc` uses Burg's method on the raw
  frame; `lpc_analysis` applies a Hamming window, the autocorrelation method
  and a $10^{-5}$ diagonal loading, so the numbers differ slightly.
- **`f32` in, `f32` out.** `lpc_analysis` takes `&[f32]` and returns `f32`
  coefficients; compare with a tolerance near $10^{-6}$ relative, not exactly.
- **Short frames.** The order is silently lowered to `frame.len() / 2`.
- **Not every pole is a formant.** Filter roots by bandwidth (and frequency)
  before calling them formants, and expect estimates biased towards harmonics
  for high-pitched voices.
- **Pre-emphasis changes the gain and the residual**, so undo it (with
  $1/(1 - \alpha z^{-1})$) if you resynthesise.

## References

- J. Makhoul, "Linear prediction: a tutorial review", *Proceedings of the IEEE* 63(4), 561-580, 1975.
- N. Levinson, "The Wiener RMS (root mean square) error criterion in filter design and prediction", *Journal of Mathematics and Physics* 25, 261-278, 1947; J. Durbin, "The fitting of time-series models", *Revue de l'Institut International de Statistique* 28(3), 233-244, 1960.
- G. Fant, *Acoustic Theory of Speech Production*, Mouton, 1960. The source-filter model.
- L. R. Rabiner and R. W. Schafer, *Digital Processing of Speech Signals*, Prentice-Hall, 1978.
- J.-M. Valin, K. Vos and T. Terriberry, *Definition of the Opus Audio Codec*, RFC 6716, IETF, 2012. SILK's LPC orders and line spectral frequencies.

## See also

- [`lpc_analysis`](crate::lpc_analysis), [`levinson_durbin`](crate::levinson_durbin),
  [`LpcCoefficients`](crate::LpcCoefficients), [`lpc_residual`](crate::lpc_residual),
  [`lpc_synthesis`](crate::lpc_synthesis) (feature `opus-codec`).
- [`topics::lpc`](crate::educational::topics::lpc) for the formulas evaluated
  at your settings.
- [`AudioTransforms::rfft`](crate::AudioTransforms::rfft) for the magnitude
  spectrum used in the lab.
- Previous chapter: [`c07_resampling`](crate::guide::c07_resampling); next:
  [`c09_pitch`](crate::guide::c09_pitch), which estimates the *source* half of the
  model.
- Lab: `examples/tutorial_08_lpc_formants.rs`
  (`cargo run --release --example tutorial_08_lpc_formants --features educational,full_no_plotting`).
