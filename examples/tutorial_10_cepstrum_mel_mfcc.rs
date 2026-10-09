//! Tutorial 10 lab: the real cepstrum, the mel scale and MFCCs.
//!
//! Accompanies the guide chapter `audio_samples::guide::c10_cepstrum_mel_mfcc`
//! (`documentation/guide/10_cepstrum_mel_mfcc.md`).
//!
//! Run with:
//!
//! ```text
//! cargo run --release --example tutorial_10_cepstrum_mel_mfcc --features educational,full_no_plotting
//! ```

use std::f64::consts::PI;
use std::num::NonZeroUsize;

use audio_samples::educational::{Explain, ExplainContext, WarningLevel, topics};
use audio_samples::utils::audio_math::{hz_to_mel, mel_to_hz};
use audio_samples::{AudioSamples, AudioTransforms, nzu, sample_rate};
use ndarray::Array1;
use spectrograms::{MelParams, MfccParams, SpectrogramParams, StftParams, WindowType};

const FS: f64 = 16_000.0;

fn slaney_hz_to_mel(f: f64) -> f64 {
    if f < 1000.0 {
        3.0 * f / 200.0
    } else {
        15.0 + 27.0 * (f / 1000.0).ln() / 6.4f64.ln()
    }
}

fn slaney_mel_to_hz(m: f64) -> f64 {
    if m < 15.0 {
        200.0 * m / 3.0
    } else {
        1000.0 * (6.4f64.ln() * (m - 15.0) / 27.0).exp()
    }
}

/// Vowel-like test signal: impulse train at f0 through three formant resonators.
fn vowel(f0_period: usize, n: usize) -> Vec<f64> {
    let mut a = vec![1.0f64];
    for &(f, b) in &[(700.0, 80.0), (1220.0, 90.0), (2600.0f64, 120.0f64)] {
        let (r, th) = ((-PI * b / FS).exp(), 2.0 * PI * f / FS);
        let s = [1.0, -2.0 * r * th.cos(), r * r];
        let mut o = vec![0.0; a.len() + 2];
        for (i, ai) in a.iter().enumerate() {
            for (j, sj) in s.iter().enumerate() {
                o[i + j] += ai * sj;
            }
        }
        a = o;
    }
    let mut y = vec![0.0; n];
    for i in 0..n {
        let e = if i % f0_period == 0 { 1.0 } else { 0.0 };
        y[i] = e
            - (1..a.len())
                .filter(|&k| k <= i)
                .map(|k| a[k] * y[i - k])
                .sum::<f64>();
    }
    y
}

fn mono(x: Vec<f64>) -> AudioSamples<'static, f64> {
    AudioSamples::new_mono(Array1::from(x), sample_rate!(16000)).expect("non-empty")
}

/// Real cepstrum of `frame` (length N): c[q] = (1/N) sum_k log|X[k]| e^{j 2 pi k q / N},
/// computed from the library's one-sided FFT using the even symmetry of log|X|.
fn real_cepstrum(frame: Vec<f64>) -> Vec<f64> {
    let n = frame.len();
    let spec = mono(frame)
        .fft(NonZeroUsize::new(n).expect("n > 0"))
        .expect("fft");
    let logmag: Vec<f64> = (0..=n / 2)
        .map(|k| spec[[0, k]].norm().max(1e-12).ln())
        .collect();
    (0..=n / 2)
        .map(|q| {
            let mut s = logmag[0] + logmag[n / 2] * (PI * q as f64).cos();
            for (k, lk) in logmag.iter().enumerate().take(n / 2).skip(1) {
                s += 2.0 * lk * (2.0 * PI * (k * q) as f64 / n as f64).cos();
            }
            s / n as f64
        })
        .collect()
}

fn hann(n: usize, i: usize) -> f64 {
    0.5 - 0.5 * (2.0 * PI * i as f64 / (n - 1) as f64).cos()
}

/// Unnormalised DCT-II, k = 0..k_out.
fn dct2(l: &[f64], k_out: usize) -> Vec<f64> {
    let m = l.len() as f64;
    (0..k_out)
        .map(|k| {
            l.iter()
                .enumerate()
                .map(|(i, li)| li * (PI * k as f64 * (i as f64 + 0.5) / m).cos())
                .sum()
        })
        .collect()
}

fn main() {
    println!("== Tutorial 10: cepstrum, mel and MFCC ==\n");

    // ------------------------------------------------------------ 1. Cepstrum
    let n = 1024;
    let period = 100; // f0 = 160 Hz
    let y = vowel(period, 3000);
    let frame: Vec<f64> = (0..n).map(|i| hann(n, i) * y[1500 + i]).collect();
    let ceps = real_cepstrum(frame);
    let (qmin, qmax) = ((FS / 400.0) as usize, (FS / 60.0) as usize);
    let qpeak = (qmin..=qmax)
        .max_by(|&a, &b| ceps[a].total_cmp(&ceps[b]))
        .expect("range");
    println!(
        "1. Real cepstrum of a 1024-sample Hann-windowed vowel frame (f0 = 160 Hz, period 100 samples)"
    );
    println!(
        "   low quefrencies (vocal tract): c[1..=3] = {:.3}, {:.3}, {:.3}",
        ceps[1], ceps[2], ceps[3]
    );
    println!(
        "   largest peak for q in {qmin}..={qmax} (400 Hz down to 60 Hz): q = {qpeak} samples = {:.2} ms -> f0 = {:.1} Hz",
        1000.0 * qpeak as f64 / FS,
        FS / qpeak as f64
    );
    assert_eq!(qpeak, 100);
    let second = (qmin..=qmax)
        .filter(|&q| q.abs_diff(qpeak) > 5)
        .map(|q| ceps[q])
        .fold(f64::NEG_INFINITY, f64::max);
    println!(
        "   peak height {:.3}; next highest value away from it {second:.3}",
        ceps[qpeak]
    );
    // Values quoted in the guide.
    assert!((ceps[1] - 2.32).abs() < 0.005 && (ceps[2] - 0.47).abs() < 0.005);
    assert!((ceps[qpeak] - 1.55).abs() < 0.005 && (second - 0.38).abs() < 0.005);
    let doc = topics::cepstrum(n, FS);
    let f0_min = doc
        .quantity("Lowest detectable f0")
        .expect("quantity")
        .value;
    assert_eq!(f0_min, 31.25);
    assert!(doc.has_warning_containing("no stand-alone cepstrum"));
    println!("   topics::cepstrum({n}, {FS}): lowest detectable f0 2 f_s/N = {f0_min} Hz");

    // ------------------------------------------------------------ 2. Mel scales
    println!("\n2. Two mel scales: audio_math (HTK) and the mel spectrogram (Slaney)");
    println!("   f (Hz) | HTK 2595 log10(1 + f/700) | Slaney | HTK round trip error (Hz)");
    for f in [0.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0] {
        let rt = (mel_to_hz(hz_to_mel(f)) - f).abs();
        println!(
            "   {f:6.0} | {:25.2} | {:6.3} | {rt:.1e}",
            hz_to_mel(f),
            slaney_hz_to_mel(f)
        );
        assert!(rt < 1e-9, "HTK round trip at {f} Hz: {rt}");
        assert!((slaney_mel_to_hz(slaney_hz_to_mel(f)) - f).abs() < 1e-9);
    }
    assert!((hz_to_mel(1000.0) - 999.99).abs() < 0.005);
    assert!((hz_to_mel(4000.0) - 2146.06).abs() < 0.005);
    assert_eq!(slaney_hz_to_mel(1000.0), 15.0);
    assert!((slaney_hz_to_mel(4000.0) - 35.164).abs() < 5e-4);

    // ------------------------------------------------------------ 3. Filterbank
    println!("\n3. Mel filterbank re-built and applied to the library's power spectrogram");
    let audio = mono(y);
    let stft = StftParams::new(nzu!(512), nzu!(160), WindowType::Hanning, true).expect("stft");
    let sp = SpectrogramParams::new(stft.clone(), FS).expect("params");
    let n_mels = 40;
    let mel = MelParams::new(NonZeroUsize::new(n_mels).expect("nz"), 0.0, FS / 2.0).expect("mel");
    let lin = audio
        .linear_power_spectrogram(&sp)
        .expect("power spectrogram");
    let melp = audio
        .mel_power_spectrogram(&sp, &mel)
        .expect("mel spectrogram");
    let (lin, melp) = (lin.data(), melp.data());
    let (m_lo, m_hi) = (slaney_hz_to_mel(0.0), slaney_hz_to_mel(FS / 2.0));
    let edges: Vec<f64> = (0..n_mels + 2)
        .map(|i| slaney_mel_to_hz(m_lo + (m_hi - m_lo) * i as f64 / (n_mels + 1) as f64))
        .collect();
    let df = FS / 512.0;
    let mut worst = 0.0f64;
    for t in 0..melp.ncols() {
        for m in 0..n_mels {
            let (l, c, r) = (edges[m], edges[m + 1], edges[m + 2]);
            let acc: f64 = (0..lin.nrows())
                .map(|k| {
                    let f = k as f64 * df;
                    ((f - l) / (c - l)).min((r - f) / (r - c)).clamp(0.0, 1.0) * lin[[k, t]]
                })
                .sum();
            worst = worst.max((acc - melp[[m, t]]).abs() / melp[[m, t]].abs().max(1e-300));
        }
    }
    println!(
        "   {} frames x {n_mels} bands; worst relative difference {worst:.1e}",
        melp.ncols()
    );
    // Same sums of products in a different order: agreement to ~1e-13.
    assert!(worst < 1e-10);
    let explained = mel.explain_with(&ExplainContext::new().with_sample_rate(FS).with_n_fft(512));
    let lowest = explained.quantity("Lowest band centre").expect("q").value;
    println!(
        "   lowest band centre {lowest:.3} Hz (explain) vs {:.3} Hz (ours)",
        edges[1]
    );
    assert!((lowest - edges[1]).abs() < 1e-9);

    // ------------------------------------------------------------ 4. MFCC
    println!("\n4. MFCC = DCT-II(10 log10 max(P_mel, 1e-8)) with lifter 22, re-derived");
    let params = MfccParams::speech_standard();
    let mfcc = audio
        .mfcc(&stft, NonZeroUsize::new(n_mels).expect("nz"), &params)
        .expect("mfcc");
    let mut worst = 0.0f64;
    for t in 0..melp.ncols() {
        let l: Vec<f64> = (0..n_mels)
            .map(|i| 10.0 * melp[[i, t]].max(1e-8).log10())
            .collect();
        for (k, c) in dct2(&l, 13).into_iter().enumerate() {
            let w = 1.0 + 11.0 * (PI * k as f64 / 22.0).sin();
            worst = worst.max((c * w - mfcc.data[[k, t]]).abs());
        }
    }
    println!(
        "   {} coefficients x {} frames; worst |difference| {worst:.1e} (values up to {:.0})",
        mfcc.data.nrows(),
        mfcc.data.ncols(),
        mfcc.data.iter().fold(0.0f64, |m, v| m.max(v.abs()))
    );
    assert!(worst < 1e-8);

    // Relation to librosa's orthonormal DCT (no lifter).
    let plain = audio
        .mfcc(
            &stft,
            NonZeroUsize::new(n_mels).expect("nz"),
            &params.with_lifter(0),
        )
        .expect("mfcc");
    let t = 10;
    let l: Vec<f64> = (0..n_mels)
        .map(|i| 10.0 * melp[[i, t]].max(1e-8).log10())
        .collect();
    let raw = dct2(&l, 13);
    let ortho: Vec<f64> = raw
        .iter()
        .enumerate()
        .map(|(k, c)| {
            let s = if k == 0 { 1.0 } else { 2.0 };
            c * (s / n_mels as f64).sqrt()
        })
        .collect();
    let r0 = plain.data[[0, t]] / ortho[0];
    let r5 = plain.data[[5, t]] / ortho[5];
    println!(
        "   lifter 0: library / orthonormal = {r0:.4} for c0 (sqrt(M) = {:.4}), {r5:.4} for c5 (sqrt(M/2) = {:.4})",
        (n_mels as f64).sqrt(),
        (n_mels as f64 / 2.0).sqrt()
    );
    assert!((r0 - (n_mels as f64).sqrt()).abs() < 1e-9);
    assert!((r5 - (n_mels as f64 / 2.0).sqrt()).abs() < 1e-9);

    // ------------------------------------------------------------ 5. Explain
    println!("\n5. Empty mel bands: 128 bands, n_fft = 256 at 16 kHz");
    let m128 = MelParams::new(nzu!(128), 0.0, 8000.0).expect("mel");
    let ctx = ExplainContext::new().with_sample_rate(FS).with_n_fft(256);
    let doc = m128.explain_with(&ctx);
    let empty = doc.quantity("Empty filters").expect("q").value as usize;
    assert_eq!(empty, 13);
    assert!(doc.has_warning(WarningLevel::Caution));
    let sp256 = SpectrogramParams::new(
        StftParams::new(nzu!(256), nzu!(128), WindowType::Hanning, true).expect("stft"),
        FS,
    )
    .expect("params");
    let m = audio.mel_power_spectrogram(&sp256, &m128).expect("mel");
    let zero_rows = m
        .data()
        .rows()
        .into_iter()
        .filter(|row| row.iter().all(|&v| v == 0.0))
        .count();
    println!(
        "   explain predicts {empty} empty filters; the computed spectrogram has {zero_rows} all-zero rows"
    );
    assert_eq!(zero_rows, empty);

    let mdoc = params.explain_with(&ExplainContext::new().with_n_mels(n_mels));
    assert_eq!(
        mdoc.quantity("Largest lifter weight").expect("q").value,
        12.0
    );
    let bad = MfccParams::speech_standard()
        .with_lifter(10)
        .explain_with(&ExplainContext::new().with_n_mels(n_mels));
    assert!(bad.has_warning(WarningLevel::Caution)); // negative lifter weights
    println!("\n{mdoc}");
    println!("{doc}");
}
