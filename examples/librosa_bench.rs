//! Wall-clock benchmark of the librosa-compatible operations.
//!
//! Times each operation on deterministic test signals and prints one JSON
//! object (median / min / max milliseconds and repetition count per
//! operation). It is the Rust half of a side-by-side comparison with
//! librosa: the companion Python script times the equivalent librosa calls
//! with identical parameters on the identical signal (same formula; use
//! `--dump DIR` to write the signals as little-endian `f64` files and check).
//!
//! Every operation here runs on one thread (none of them is parallel by
//! design). Each is run twice to warm up (thread-local FFT plans, filter
//! tables and window caches are built then), then repeatedly until at least
//! `--min-time` seconds (default 1.0) and `--min-reps` repetitions (default
//! 5) have elapsed, at most `--max-reps` (default 50); the median is reported.
//!
//! ```sh
//! cargo run --release --example librosa_bench --features full_no_plotting
//! cargo run --release --example librosa_bench --features full_no_plotting -- --only cqt,stft_matrix
//! ```

use std::hint::black_box;
use std::num::NonZeroUsize;
use std::time::Instant;

use audio_samples::operations::constant_q::{ChromaCqtConfig, CqtConfig};
use audio_samples::operations::effects::{
    HarmonicPercussiveConfig, PitchShiftConfig, SilenceConfig,
};
use audio_samples::operations::features::{
    RmsConfig, SpectralCentroidConfig, SpectralContrastConfig, ZeroCrossingRateConfig,
};
use audio_samples::operations::pitch_tracking::{PyinConfig, YinConfig};
use audio_samples::operations::rhythm::{
    BeatTrackConfig, OnsetDetectConfig, OnsetStrengthConfig, TempoConfig,
};
use audio_samples::operations::sequence::{DtwConfig, dtw, viterbi};
use audio_samples::operations::spectrum::{
    DbConfig, MelSpectrogramConfig, MfccConfig, StftConfig, power_to_db,
};
use audio_samples::operations::tonal::ChromaStftConfig;
use audio_samples::operations::types::ResamplingQuality;
use audio_samples::{
    AudioConstantQ, AudioEffects, AudioFeatures, AudioPitchTracking, AudioRhythm, AudioSamples,
    AudioSpectrum, AudioTonal, resample, resample_by_ratio, sample_rate,
};
use ndarray::{Array1, Array2};

/// 64-bit LCG (Knuth MMIX constants); uniform in `[0, 1)` from the top 53 bits.
struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// Deterministic test signal: a sequence of decaying harmonic notes (one
/// every 0.5 s, pitch walking in fifths over two octaves from 110 Hz) plus
/// low-level noise, with 0.5 s of digital silence at each end.
fn test_signal(sr: u32, seconds: f64) -> Vec<f64> {
    let n = (f64::from(sr) * seconds).round() as usize;
    let mut rng = Lcg(12345);
    let tau = std::f64::consts::TAU;
    (0..n)
        .map(|i| {
            let noise = 2.0 * rng.next() - 1.0;
            let t = i as f64 / f64::from(sr);
            if t < 0.5 || t >= seconds - 0.5 {
                return 0.0;
            }
            let seg = (t / 0.5).floor();
            let f = 110.0 * 2f64.powf(((seg as u64 * 7) % 24) as f64 / 12.0);
            let env = (-3.0 * (t - seg * 0.5)).exp();
            env * (0.5 * (tau * f * t).sin()
                + 0.25 * (tau * 2.0 * f * t).sin()
                + 0.12 * (tau * 3.0 * f * t).sin())
                + 0.02 * noise
        })
        .collect()
}

/// `rows x cols` matrix of LCG uniforms (row-major).
fn random_matrix(rows: usize, cols: usize, seed: u64) -> Array2<f64> {
    let mut rng = Lcg(seed);
    Array2::from_shape_fn((rows, cols), |_| rng.next())
}

struct Opts {
    only: Option<Vec<String>>,
    min_time: f64,
    min_reps: usize,
    max_reps: usize,
    dump: Option<String>,
}

fn parse_args() -> Opts {
    let mut opts = Opts {
        only: None,
        min_time: 1.0,
        min_reps: 5,
        max_reps: 50,
        dump: None,
    };
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        let mut val = || args.next().expect("missing value for argument");
        match a.as_str() {
            "--only" => opts.only = Some(val().split(',').map(str::to_owned).collect()),
            "--min-time" => opts.min_time = val().parse().expect("--min-time"),
            "--min-reps" => opts.min_reps = val().parse().expect("--min-reps"),
            "--max-reps" => opts.max_reps = val().parse().expect("--max-reps"),
            "--dump" => opts.dump = Some(val()),
            other => panic!("unknown argument {other}"),
        }
    }
    opts
}

struct Bench {
    opts: Opts,
    results: Vec<(String, f64, f64, f64, usize)>,
}

impl Bench {
    fn run<R>(&mut self, name: &str, mut f: impl FnMut() -> R) {
        if let Some(only) = &self.opts.only
            && !only.iter().any(|o| name.starts_with(o.as_str()))
        {
            return;
        }
        for _ in 0..2 {
            black_box(f());
        }
        let mut times = Vec::new();
        let start = Instant::now();
        while times.len() < self.opts.max_reps
            && (times.len() < self.opts.min_reps
                || start.elapsed().as_secs_f64() < self.opts.min_time)
        {
            let t = Instant::now();
            black_box(f());
            times.push(t.elapsed().as_secs_f64() * 1e3);
        }
        times.sort_by(f64::total_cmp);
        let n = times.len();
        let median = if n % 2 == 1 {
            times[n / 2]
        } else {
            0.5 * (times[n / 2 - 1] + times[n / 2])
        };
        eprintln!("{name:<32} {median:10.3} ms  (n={n})");
        self.results
            .push((name.to_owned(), median, times[0], times[n - 1], n));
    }
}

fn main() {
    let opts = parse_args();
    let x22 = test_signal(22050, 30.0);
    let x22_5s = test_signal(22050, 5.0);
    let x44 = test_signal(44100, 30.0);
    if let Some(dir) = &opts.dump {
        for (name, x) in [
            ("sig_22050_30s", &x22),
            ("sig_22050_5s", &x22_5s),
            ("sig_44100_30s", &x44),
        ] {
            let bytes: Vec<u8> = x.iter().flat_map(|v| v.to_le_bytes()).collect();
            std::fs::write(format!("{dir}/{name}.f64"), bytes).expect("dump");
        }
    }
    let y = AudioSamples::new_mono(Array1::from(x22), sample_rate!(22050)).unwrap();
    let y5 = AudioSamples::new_mono(Array1::from(x22_5s), sample_rate!(22050)).unwrap();
    let y44 = AudioSamples::new_mono(Array1::from(x44), sample_rate!(44100)).unwrap();

    let mut b = Bench {
        opts,
        results: Vec::new(),
    };

    // --- spectrum -------------------------------------------------------
    let stft_cfg = StftConfig::default();
    b.run("stft_matrix", || y.stft_matrix(&stft_cfg).unwrap());
    let d = y.stft_matrix(&stft_cfg).unwrap();
    let n = NonZeroUsize::new(y.samples_per_channel().get());
    b.run("istft_matrix", || {
        AudioSamples::<f64>::istft_matrix(&d, &stft_cfg, sample_rate!(22050), n).unwrap()
    });
    let mel_cfg = MelSpectrogramConfig::default();
    b.run("melspectrogram", || y.melspectrogram(&mel_cfg).unwrap());
    let mfcc_cfg = MfccConfig::librosa();
    b.run("mfcc_frames", || y.mfcc_frames(&mfcc_cfg).unwrap());
    let mel = y.melspectrogram(&mel_cfg).unwrap();
    let db_cfg = DbConfig::librosa_power();
    b.run("power_to_db", || power_to_db(&mel, &db_cfg).unwrap());

    // --- tonal / constant-Q ---------------------------------------------
    let chroma_cfg = ChromaStftConfig::default();
    b.run("chroma_stft", || y.chroma_stft(&chroma_cfg).unwrap());
    let cqt_cfg = CqtConfig::default();
    b.run("cqt", || y.cqt(&cqt_cfg).unwrap());
    let chroma_cqt_cfg = ChromaCqtConfig::default();
    b.run("chroma_cqt", || y.chroma_cqt(&chroma_cqt_cfg).unwrap());

    // --- features -------------------------------------------------------
    let c = SpectralCentroidConfig::default();
    b.run("spectral_centroid_frames", || {
        y.spectral_centroid_frames(&c).unwrap()
    });
    let c = SpectralContrastConfig::default();
    b.run("spectral_contrast_frames", || {
        y.spectral_contrast_frames(&c).unwrap()
    });
    let c = RmsConfig::default();
    b.run("rms_frames", || y.rms_frames(&c).unwrap());
    let c = ZeroCrossingRateConfig::default();
    b.run("zero_crossing_rate_frames", || {
        y.zero_crossing_rate_frames(&c).unwrap()
    });

    // --- pitch ----------------------------------------------------------
    let c = YinConfig::default();
    b.run("yin", || y.yin(&c).unwrap());
    let c = PyinConfig::default();
    b.run("pyin_5s", || y5.pyin(&c).unwrap());

    // --- effects --------------------------------------------------------
    b.run("time_stretch_1.25", || {
        y.time_stretch(1.25, &stft_cfg).unwrap()
    });
    let c = PitchShiftConfig::librosa();
    b.run("pitch_shift_+4", || y.pitch_shift(4.0, &c).unwrap());
    let c = HarmonicPercussiveConfig::librosa();
    b.run("hpss", || y.harmonic_percussive(&c).unwrap());
    let c = SilenceConfig::librosa();
    b.run("trim_db", || y.trim_db(&c).unwrap());
    b.run("split_db", || y.split_db(&c).unwrap());
    b.run("preemphasis", || y.preemphasis(0.97).unwrap());

    // --- rhythm ---------------------------------------------------------
    let c = OnsetStrengthConfig::default();
    b.run("onset_strength", || y.onset_strength(&c).unwrap());
    let c = OnsetDetectConfig::default();
    b.run("onset_detect", || y.onset_detect(&c).unwrap());
    let c = TempoConfig::default();
    b.run("tempo", || y.tempo(&c).unwrap());
    let c = BeatTrackConfig::default();
    b.run("beat_track", || y.beat_track(&c).unwrap());

    // --- resampling -----------------------------------------------------
    for (q, qn) in [
        (ResamplingQuality::Fast, "fast"),
        (ResamplingQuality::Medium, "medium"),
        (ResamplingQuality::High, "high"),
    ] {
        b.run(&format!("resample_22050_16000_{qn}"), || {
            resample(&y, sample_rate!(16000), q).unwrap()
        });
        b.run(&format!("resample_44100_48000_{qn}"), || {
            resample(&y44, sample_rate!(48000), q).unwrap()
        });
        b.run(&format!("resample_44100_22050_{qn}"), || {
            resample(&y44, sample_rate!(22050), q).unwrap()
        });
    }
    // Irrational ratio (the pitch_shift path): 2^(-4/12).
    let ratio = 2f64.powf(-4.0 / 12.0);
    b.run("resample_by_ratio_2^-1/3_high", || {
        resample_by_ratio(&y, ratio, ResamplingQuality::High).unwrap()
    });

    // --- sequence -------------------------------------------------------
    let xa = random_matrix(12, 500, 1);
    let xb = random_matrix(12, 500, 2);
    let dc = DtwConfig::default();
    b.run("dtw_12x500", || dtw(xa.view(), xb.view(), &dc).unwrap());
    let n_states = 100;
    let prob = random_matrix(n_states, 2000, 3);
    let mut trans = Array2::from_shape_fn((n_states, n_states), |(i, j)| {
        (-(i.abs_diff(j) as f64) / 3.0).exp()
    });
    for mut row in trans.rows_mut() {
        let s = row.sum();
        row /= s;
    }
    b.run("viterbi_100x2000", || {
        viterbi(prob.view(), trans.view(), None, None).unwrap()
    });

    // --- JSON -----------------------------------------------------------
    let mut out = String::from("{\n  \"library\": \"audio_samples\",\n  \"results\": {\n");
    for (i, (name, med, min, max, n)) in b.results.iter().enumerate() {
        let sep = if i + 1 == b.results.len() { "" } else { "," };
        out.push_str(&format!(
            "    \"{name}\": {{\"median_ms\": {med:.4}, \"min_ms\": {min:.4}, \"max_ms\": {max:.4}, \"reps\": {n}}}{sep}\n"
        ));
    }
    out.push_str("  }\n}");
    println!("{out}");
}
