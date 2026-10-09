//! Tutorial 08 lab: linear prediction (LPC), the source-filter model and formants.
//!
//! Accompanies the guide chapter `audio_samples::guide::c08_lpc_formants`
//! (`documentation/guide/08_lpc_formants.md`).
//!
//! What it does:
//!
//! 1. Synthesises a vowel-like sound with the source-filter model: an impulse
//!    train at f0 = 120 Hz (the glottal source) through an all-pole filter whose
//!    three resonances (formants) are known exactly.
//! 2. Estimates LPC coefficients with the crate's `lpc_analysis` (feature
//!    `opus-codec`) and with a Levinson-Durbin recursion written out here, and
//!    checks that they agree and solve the order-p normal equations.
//! 3. Finds the roots of A(z) with a small Durand-Kerner root finder and turns
//!    the complex pole pairs back into formant frequencies and bandwidths.
//! 4. Compares the LPC spectral envelope with the FFT of the frame and writes
//!    both to a CSV file in the system temp directory.
//! 5. Shows what pre-emphasis does when the source has a spectral tilt.
//!
//! Run with:
//!
//! ```text
//! cargo run --release --example tutorial_08_lpc_formants --features educational,full_no_plotting
//! ```

use std::f64::consts::PI;
use std::io::Write;
use std::num::NonZeroUsize;

use audio_samples::educational::{WarningLevel, topics};
use audio_samples::{AudioSamples, AudioTransforms, levinson_durbin, lpc_analysis, sample_rate};
use ndarray::Array1;
use num_complex::Complex64;

const FS: f64 = 16_000.0;
const F0: f64 = 120.0;
/// (formant frequency F in Hz, bandwidth B in Hz) of an /a/-like vowel.
const FORMANTS: [(f64, f64); 3] = [(700.0, 80.0), (1220.0, 90.0), (2600.0, 120.0)];
const FRAME_START: usize = 1600; // skip the first 100 ms (filter start-up)
const FRAME_LEN: usize = 512; // 32 ms at 16 kHz, about 3.8 pitch periods

/// Coefficients [1, a1, ..., ap] of A(z) = prod_i (1 - 2 r_i cos(theta_i) z^-1 + r_i^2 z^-2).
fn allpole_from_formants(formants: &[(f64, f64)], fs: f64) -> Vec<f64> {
    let mut a = vec![1.0];
    for &(f, b) in formants {
        let r = (-PI * b / fs).exp();
        let theta = 2.0 * PI * f / fs;
        a = poly_mul(&a, &[1.0, -2.0 * r * theta.cos(), r * r]);
    }
    a
}

fn poly_mul(a: &[f64], b: &[f64]) -> Vec<f64> {
    let mut out = vec![0.0; a.len() + b.len() - 1];
    for (i, &ai) in a.iter().enumerate() {
        for (j, &bj) in b.iter().enumerate() {
            out[i + j] += ai * bj;
        }
    }
    out
}

/// y[n] = e[n] - sum_{k=1}^{p} a_k y[n-k] (a[0] = 1 is skipped).
fn allpole_filter(a: &[f64], e: &[f64]) -> Vec<f64> {
    let mut y = vec![0.0; e.len()];
    for n in 0..e.len() {
        let mut acc = e[n];
        for k in 1..a.len().min(n + 1) {
            acc -= a[k] * y[n - k];
        }
        y[n] = acc;
    }
    y
}

/// Impulse train with one unit impulse per period of `f0` (rounded to samples).
fn impulse_train(f0: f64, fs: f64, n: usize) -> Vec<f64> {
    let period = (fs / f0).round() as usize;
    (0..n)
        .map(|i| if i % period == 0 { 1.0 } else { 0.0 })
        .collect()
}

/// Biased autocorrelation of the Hamming-windowed frame, as `lpc_analysis` computes it.
fn hamming_autocorr(x: &[f32], max_lag: usize) -> Vec<f64> {
    let n = x.len();
    let w: Vec<f64> = (0..n)
        .map(|i| {
            let wi = 0.54 - 0.46 * (2.0 * PI * i as f64 / (n - 1) as f64).cos();
            // The library rounds the windowed frame to f32 before correlating.
            f64::from((wi * f64::from(x[i])) as f32)
        })
        .collect();
    (0..=max_lag)
        .map(|m| (0..n - m).map(|i| w[i] * w[i + m]).sum())
        .collect()
}

/// Levinson-Durbin for e[n] = x[n] + sum a_k x[n-k]. Returns (a_1..a_p, E_p).
fn my_levinson(r: &[f64], p: usize) -> (Vec<f64>, f64) {
    let mut a = vec![0.0; p + 1]; // a[0] = 1 is implicit
    let mut err = r[0];
    for m in 1..=p {
        let acc = r[m] + (1..m).map(|i| a[i] * r[m - i]).sum::<f64>();
        let k = -acc / err;
        let prev = a.clone();
        for i in 1..m {
            a[i] = prev[i] + k * prev[m - i];
        }
        a[m] = k;
        err *= 1.0 - k * k;
    }
    (a[1..].to_vec(), err)
}

/// All roots of c[0] z^p + c[1] z^(p-1) + ... + c[p] (c[0] = 1) by Durand-Kerner iteration.
fn poly_roots(c: &[f64]) -> Vec<Complex64> {
    let p = c.len() - 1;
    let eval = |z: Complex64| {
        c.iter()
            .fold(Complex64::new(0.0, 0.0), |acc, &ci| acc * z + ci)
    };
    let mut z: Vec<Complex64> = (0..p)
        .map(|i| Complex64::from_polar(0.9, 2.0 * PI * i as f64 / p as f64 + 0.4))
        .collect();
    for _ in 0..5000 {
        let mut biggest_step: f64 = 0.0;
        for i in 0..p {
            let mut den = Complex64::new(1.0, 0.0);
            for j in 0..p {
                if j != i {
                    den *= z[i] - z[j];
                }
            }
            let step = eval(z[i]) / den;
            z[i] -= step;
            biggest_step = biggest_step.max(step.norm());
        }
        if biggest_step < 1e-14 {
            break;
        }
    }
    z
}

/// (frequency Hz, bandwidth Hz) of each pole in the upper half plane, sorted by frequency.
fn resonances(roots: &[Complex64], fs: f64) -> Vec<(f64, f64)> {
    let mut out: Vec<(f64, f64)> = roots
        .iter()
        .filter(|z| z.im > 1e-9)
        .map(|z| (z.arg() * fs / (2.0 * PI), -z.norm().ln() * fs / PI))
        .collect();
    out.sort_by(|x, y| x.0.total_cmp(&y.0));
    out
}

/// 20 log10 |1/A(e^{jw})| in dB at frequency `f` Hz.
fn envelope_db(a_full: &[f64], f: f64, fs: f64) -> f64 {
    let w = 2.0 * PI * f / fs;
    let s = a_full
        .iter()
        .enumerate()
        .fold(Complex64::new(0.0, 0.0), |acc, (k, &ak)| {
            acc + ak * Complex64::from_polar(1.0, -w * k as f64)
        });
    -20.0 * s.norm().log10()
}

/// For each true formant, the closest narrow resonance (B < 400 Hz) within 150 Hz.
fn match_formants(found: &[(f64, f64)], truth: &[(f64, f64)]) -> Vec<Option<(f64, f64)>> {
    truth
        .iter()
        .map(|&(f, _)| {
            found
                .iter()
                .filter(|r| r.1 < 400.0 && r.0 > 90.0)
                .min_by(|x, y| (x.0 - f).abs().total_cmp(&(y.0 - f).abs()))
                .copied()
                .filter(|r| (r.0 - f).abs() < 150.0)
        })
        .collect()
}

fn lpc_formants(frame: &[f32], order: usize) -> Vec<Option<(f64, f64)>> {
    let lpc = lpc_analysis(frame, order);
    let mut a = vec![1.0];
    a.extend(lpc.coeffs.iter().map(|&v| f64::from(v)));
    match_formants(&resonances(&poly_roots(&a), FS), &FORMANTS)
}

fn coeffs_to_poly(coeffs: &[f32]) -> Vec<f64> {
    let mut a = vec![1.0];
    a.extend(coeffs.iter().map(|&v| f64::from(v)));
    a
}

fn take_frame(x: &[f64]) -> Vec<f32> {
    x[FRAME_START..FRAME_START + FRAME_LEN]
        .iter()
        .map(|&v| v as f32)
        .collect()
}

fn main() {
    println!("== Tutorial 08: LPC, the source-filter model and formants ==\n");

    // ------------------------------------------------------------ 1. Synthesis
    let a_true = allpole_from_formants(&FORMANTS, FS);
    println!("1. Source-filter synthesis at f_s = {FS} Hz, f0 = {F0} Hz");
    for &(f, b) in &FORMANTS {
        let r = (-PI * b / FS).exp();
        println!(
            "   formant F = {f:6.0} Hz, B = {b:4.0} Hz  ->  pole radius r = {r:.5}, angle = {:.4} rad",
            2.0 * PI * f / FS
        );
    }
    let r1 = (-PI * 80.0 / FS).exp();
    assert!((r1 - 0.98441).abs() < 5e-6, "r(B = 80 Hz) = {r1}");

    let n_total = FRAME_START + FRAME_LEN + 256;
    let source = impulse_train(F0, FS, n_total);
    let vowel = allpole_filter(&a_true, &source);
    let frame = take_frame(&vowel);
    println!(
        "   analysis frame: samples {FRAME_START}..{} ({} ms, {:.1} pitch periods)",
        FRAME_START + FRAME_LEN,
        1000.0 * FRAME_LEN as f64 / FS,
        FRAME_LEN as f64 * F0 / FS
    );

    // ------------------------------------------------------------ 2. LPC
    let order = (FS / 1000.0) as usize + 2;
    let doc = topics::lpc(order, FS);
    let rule = doc.quantity("Rule-of-thumb order").expect("quantity").value;
    assert_eq!(rule, 18.0);
    println!("\n2. LPC of order p = {order} (rule of thumb f_s/1000 + 2 = {rule})");

    let lib = lpc_analysis(&frame, order);
    assert_eq!(
        lib.coeffs.len(),
        order,
        "lpc_analysis returns p coefficients"
    );

    let r_raw = hamming_autocorr(&frame, order);
    let mut r = r_raw.clone();
    r[0] *= 1.0 + 1e-5; // the library's diagonal loading
    let (mine, err_mine) = my_levinson(&r, order);
    let max_abs = mine.iter().fold(0.0f64, |m, v| m.max(v.abs()));
    let max_diff = mine
        .iter()
        .zip(&lib.coeffs)
        .map(|(m, l)| (m - f64::from(*l)).abs())
        .fold(0.0, f64::max);
    println!(
        "   lpc_analysis vs Levinson-Durbin written out: max |difference| = {max_diff:.1e} (largest |a_k| = {max_abs:.2})"
    );
    println!(
        "   prediction-error energy E_p: library {:.4}, ours {err_mine:.4}",
        lib.prediction_error
    );
    // The library stores f32 coefficients (relative precision 6e-8), so the
    // two agree to about 1e-7 of the largest coefficient; 1e-6 leaves margin.
    assert!(max_diff < 1e-6 * max_abs.max(1.0), "max diff {max_diff}");
    assert!((lib.prediction_error - err_mine).abs() < 1e-9 * err_mine);

    // The public recursion applied to our autocorrelation agrees as well.
    let ld = levinson_durbin(&r_raw, order).expect("non-silent frame");
    let ld_diff = mine
        .iter()
        .zip(&ld.coeffs)
        .map(|(m, l)| (m - f64::from(*l)).abs())
        .fold(0.0, f64::max);
    assert!(
        ld_diff < 1e-6 * max_abs.max(1.0),
        "levinson_durbin diff {ld_diff}"
    );

    // Order-p check: sum_k a_k R[|i-k|] = -R[i] for i = 1..p.
    let normal_residual = |a: &[f64]| {
        (1..=order)
            .map(|i| {
                let lhs: f64 = (1..=order).map(|k| a[k - 1] * r[i.abs_diff(k)]).sum();
                (lhs + r[i]).abs() / r[0]
            })
            .fold(0.0, f64::max)
    };
    let lib_f64: Vec<f64> = lib.coeffs.iter().map(|&v| f64::from(v)).collect();
    let res_lib = normal_residual(&lib_f64);
    let (mut short, _) = my_levinson(&r, order - 1);
    short.push(0.0);
    let res_short = normal_residual(&short);
    println!(
        "   worst normal-equation residual / R[0]: library {res_lib:.1e}; an order-{} solution would give {res_short:.1e}",
        order - 1
    );
    assert!(res_lib < 1e-6, "not the order-p solution: {res_lib}");
    assert!(res_short > 100.0 * res_lib);

    // ------------------------------------------------------------ 3. Roots
    let a_lpc = coeffs_to_poly(&lib.coeffs);
    let roots = poly_roots(&a_lpc);
    let worst_residual = roots
        .iter()
        .map(|&z| {
            a_lpc
                .iter()
                .fold(Complex64::new(0.0, 0.0), |acc, &c| acc * z + c)
                .norm()
        })
        .fold(0.0, f64::max);
    let max_radius = roots.iter().map(|z| z.norm()).fold(0.0, f64::max);
    println!(
        "\n3. Durand-Kerner: {} roots of A(z), worst |A(z_i)| = {worst_residual:.1e}, largest |z_i| = {max_radius:.4} (inside the unit circle: stable)",
        roots.len()
    );
    assert!(worst_residual < 1e-10);
    assert!(max_radius < 1.0);
    let res = resonances(&roots, FS);
    println!("   all pole pairs (F, B) in Hz:");
    for (f, b) in &res {
        println!("     F = {f:7.1}  B = {b:7.1}");
    }
    let est = match_formants(&res, &FORMANTS);
    println!("   true formant -> narrow LPC resonance (B < 400 Hz):");
    for (&(f, b), e) in FORMANTS.iter().zip(&est) {
        let (fe, be) = e.expect("formant recovered");
        println!(
            "     F = {f:6.0} Hz (B = {b:3.0})  ->  {fe:7.1} Hz (B = {be:5.1}), error {:+5.1} Hz",
            fe - f
        );
        // A periodic source samples the envelope only at harmonics (every
        // 120 Hz), which biases autocorrelation LPC; 20 Hz is under 3 % of F1.
        assert!((fe - f).abs() < 20.0, "formant {f} estimated at {fe}");
    }

    // ------------------------------------------------------------ 4. Envelope
    let windowed: Vec<f64> = frame
        .iter()
        .enumerate()
        .map(|(i, &x)| {
            let w = 0.54 - 0.46 * (2.0 * PI * i as f64 / (FRAME_LEN - 1) as f64).cos();
            w * f64::from(x)
        })
        .collect();
    let audio = AudioSamples::new_mono(Array1::from(windowed), sample_rate!(16000)).expect("frame");
    let n_fft = 1024;
    let mag = audio
        .rfft(NonZeroUsize::new(n_fft).expect("n_fft > 0"))
        .expect("fft");
    let df = FS / n_fft as f64;
    // Harmonic peaks of |X| (largest bin within +-2 bins of h f0) minus the LPC envelope.
    let diffs: Vec<f64> = (1..=33)
        .map(|h| {
            let f = f64::from(h) * F0;
            let k = (f / df).round() as usize;
            let kk = (k - 2..=k + 2)
                .max_by(|&i, &j| mag[[0, i]].total_cmp(&mag[[0, j]]))
                .expect("bins");
            20.0 * mag[[0, kk]].log10() - envelope_db(&a_lpc, f, FS)
        })
        .collect();
    let mean = diffs.iter().sum::<f64>() / diffs.len() as f64;
    let sd = (diffs.iter().map(|d| (d - mean).powi(2)).sum::<f64>() / diffs.len() as f64).sqrt();
    println!(
        "\n4. Harmonic peaks of |FFT| (h f0, h = 1..33, up to 3960 Hz) vs LPC envelope 20 log10|1/A|:"
    );
    println!(
        "   constant gain offset {mean:.1} dB, scatter about it {sd:.2} dB (standard deviation)"
    );
    assert!(sd < 1.5, "harmonics should sit on the envelope, sd {sd}");
    assert!((sd - 1.06).abs() < 0.005, "value quoted in the guide: {sd}");

    let path = std::env::temp_dir().join("tutorial_08_envelope.csv");
    let mut file = std::fs::File::create(&path).expect("create CSV");
    writeln!(file, "freq_hz,fft_db,lpc_envelope_db,true_envelope_db").expect("write");
    for k in 0..mag.ncols() {
        let f = k as f64 * df;
        writeln!(
            file,
            "{f:.2},{:.3},{:.3},{:.3}",
            20.0 * mag[[0, k]].max(1e-12).log10(),
            envelope_db(&a_lpc, f, FS) + mean,
            envelope_db(&a_true, f, FS) + mean
        )
        .expect("write");
    }
    println!(
        "   wrote {} (columns: freq_hz, fft_db, lpc_envelope_db, true_envelope_db)",
        path.display()
    );

    // ------------------------------------------------------------ 5. Order
    println!("\n5. Model order (formants recovered within 150 Hz with B < 400 Hz):");
    let mut smallest = None;
    for p in [4usize, 6, 8, 10, 12, 18, 24] {
        let found = lpc_formants(&frame, p);
        let n = found.iter().filter(|e| e.is_some()).count();
        println!("   p = {p:2}: {n} of 3");
        if n == 3 && smallest.is_none() {
            smallest = Some(p);
        }
    }
    println!("   smallest order (from those tried) that recovers all three: {smallest:?}");
    assert_eq!(smallest, Some(10));
    let low = topics::lpc(4, FS);
    assert!(low.has_warning(WarningLevel::Caution));
    println!("   topics::lpc(4, {FS}) warns: well below the rule of thumb");

    // ------------------------------------------------------------ 6. Pre-emphasis
    // A glottal-like source: the impulse train through two real poles at 0.95
    // (about -12 dB/octave above 130 Hz).
    let glottal = allpole_filter(&poly_mul(&[1.0, -0.95], &[1.0, -0.95]), &source);
    let tilted = allpole_filter(&a_true, &glottal);
    let raw = take_frame(&tilted);
    let pre: Vec<f32> = (FRAME_START..FRAME_START + FRAME_LEN)
        .map(|n| (tilted[n] - 0.97 * tilted[n - 1]) as f32)
        .collect();
    let raw_res = resonances(
        &poly_roots(&coeffs_to_poly(&lpc_analysis(&raw, order).coeffs)),
        FS,
    );
    let raw_f3 = raw_res
        .iter()
        .min_by(|x, y| (x.0 - 2600.0).abs().total_cmp(&(y.0 - 2600.0).abs()))
        .copied()
        .expect("poles");
    let pre_found = lpc_formants(&pre, order);
    println!("\n6. Tilted source (two poles at z = 0.95), order {order}:");
    println!(
        "   without pre-emphasis the pole nearest F3 is at {:.0} Hz with B = {:.0} Hz (too broad to count as a formant)",
        raw_f3.0, raw_f3.1
    );
    assert!(raw_f3.1 > 400.0);
    print!("   with pre-emphasis y[n] = x[n] - 0.97 x[n-1]:");
    for (&(f, _), e) in FORMANTS.iter().zip(&pre_found) {
        let (fe, _) = e.expect("recovered after pre-emphasis");
        print!("  {f:.0} -> {fe:.0} Hz");
        assert!((fe - f).abs() < 30.0);
    }
    println!();

    // ------------------------------------------------------------ 7. Explain
    println!("\n7. What the library says (topics::lpc({order}, {FS})):\n\n{doc}");
}
