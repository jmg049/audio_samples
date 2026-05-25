//! Shared body for the `AudioTransforms` bench targets
//! (`bench_trans_walltime`, `_instructions`, `_cycles`, `_cache_misses`,
//! `_branch_misses`).
//!
//! Each per-measurement bench-target file under `bench_suite/benches/`
//! includes this module via `#[path = "../benches_shared/trans.rs"]` and
//! invokes [`bench_all`] with its `Criterion<M>` instance.
//!
//! Conventions: see `bench_suite/METHODOLOGY.md`.
//! Function inventory: see `bench_suite/CATALOG.md` section `Trans`
//! (rows Trans-001 .. Trans-029).
//!
//! ## Cost-axis strategy
//!
//! Most ops in this trait have three or four cost-driving axes —
//! signal length, `n_fft`, `hop_length`, and a filterbank size (`n_mels`,
//! `n_bins`, etc.). Sweeping the cross-product blindly would balloon
//! the suite to hours per row. Instead each row picks a focused subset:
//!
//! - One-shot ops (`fft`, `rfft`): sweep `n_fft ∈ {512, 2048, 8192}` and
//!   constrain `n_fft × len ≤ 2³²` so we never run an 8192-point FFT on
//!   a 1 M sample buffer.
//! - STFT-driven ops (Trans-003 onwards): use a single `n_fft = 2048`,
//!   `hop = n_fft/4 = 512`, `WindowType::Hanning`. `stft_forward` also
//!   sweeps `n_fft ∈ {512, 2048, 8192}`.
//! - Filterbank-driven ops (mel/loghz/gammatone/cqt/mfcc/chroma): cap the
//!   length sweep at 262_144 to keep wall-clock per point under 30 s.

use criterion::measurement::Measurement;
use criterion::{BatchSize, BenchmarkId, Criterion, Throughput};
use std::hint::black_box;
use std::num::NonZeroUsize;

use audio_samples::{AudioTransforms, I24, nzu};

use bench_suite_common::{
    CHANNELS_DEFAULT, DTYPES_DEFAULT, LENGTH_SWEEP_NO_XXXL, ParamLabel, SampleSizePolicy,
    fixture_a440, sample_size_for,
};

use spectrograms::{
    ChromaParams, CqtParams, GammatoneParams, LogHzParams, LogParams, MelParams, MfccParams,
    SpectrogramParams, StftParams, WindowType,
};

// ===========================================================================
// Length sweeps
// ===========================================================================

/// Lengths used for STFT-based and filterbank ops. Capped at 262_144 so the
/// largest call still finishes in well under 30 s per point.
const LENGTH_SWEEP_TRANS_FILTERBANK: &[usize] = &[
    1024, 4096, 16_384, 65_536, 262_144,
];

/// Lengths used for STFT-based ops without a heavy filterbank pass.
/// Capped at 1_048_576 (same as `LENGTH_SWEEP_NO_XXXL`).
const LENGTH_SWEEP_TRANS_STFT: &[usize] = LENGTH_SWEEP_NO_XXXL;

/// `n_fft` sweep used by one-shot FFT ops. Three power-of-two anchors.
const NFFT_SWEEP: &[usize] = &[512, 2048, 8192];

const DEFAULT_N_FFT: usize = 2048;
const DEFAULT_HOP: usize = 512; // n_fft / 4
const BENCH_SAMPLE_RATE_HZ: f64 = 44_100.0;

#[inline]
fn default_window() -> WindowType {
    WindowType::Hanning
}

// ===========================================================================
// Top-level entry point — wrappers call into this with their typed Criterion.
// ===========================================================================

pub fn bench_all<M: Measurement>(c: &mut Criterion<M>) {
    bench_trans_001_fft(c);
    bench_trans_002_rfft(c);
    bench_trans_003_stft(c);
    bench_trans_004_istft(c);
    bench_trans_005_linear_spectrogram(c);
    bench_trans_006_linear_magnitude_spectrogram(c);
    bench_trans_007_linear_power_spectrogram(c);
    bench_trans_008_linear_db_spectrogram(c);
    bench_trans_009_log_frequency_spectrogram(c);
    bench_trans_010_loghz_power_spectrogram(c);
    bench_trans_011_loghz_magnitude_spectrogram(c);
    bench_trans_012_loghz_db_spectrogram(c);
    bench_trans_013_mel_spectrogram(c);
    bench_trans_014_mel_mag_spectrogram(c);
    bench_trans_015_mel_db_spectrogram(c);
    bench_trans_016_mel_power_spectrogram(c);
    bench_trans_017_mfcc(c);
    bench_trans_018_chromagram(c);
    bench_trans_019_power_spectral_density(c);
    bench_trans_020_gammatone_spectrogram(c);
    bench_trans_021_gammatone_magnitude_spectrogram(c);
    bench_trans_022_gammatone_power_spectrogram(c);
    bench_trans_023_gammatone_db_spectrogram(c);
    bench_trans_024_constant_q_transform(c);
    bench_trans_025_cqt_spectrogram(c);
    bench_trans_026_cqt_magnitude_spectrogram(c);
    bench_trans_027_cqt_power_spectrogram(c);
    bench_trans_028_cqt_db_spectrogram(c);
    bench_trans_029_magphase(c);
}

// ===========================================================================
// Param-builders
// ===========================================================================

fn make_stft_params(n_fft: usize, hop: usize) -> StftParams {
    StftParams::new(
        NonZeroUsize::new(n_fft).expect("n_fft > 0"),
        NonZeroUsize::new(hop).expect("hop > 0"),
        default_window(),
        true,
    )
    .expect("valid StftParams")
}

fn make_spec_params(n_fft: usize, hop: usize) -> SpectrogramParams {
    SpectrogramParams::new(make_stft_params(n_fft, hop), BENCH_SAMPLE_RATE_HZ)
        .expect("valid SpectrogramParams")
}

fn default_log_params() -> LogParams {
    LogParams::new(-80.0).expect("valid LogParams")
}

fn default_mel_params() -> MelParams {
    MelParams::new(nzu!(40), 0.0, 8_000.0).expect("valid MelParams")
}

fn default_loghz_params() -> LogHzParams {
    LogHzParams::new(nzu!(64), 80.0, 8_000.0).expect("valid LogHzParams")
}

fn default_gammatone_params() -> GammatoneParams {
    GammatoneParams::new(nzu!(32), 80.0, 8_000.0).expect("valid GammatoneParams")
}

fn default_cqt_params() -> CqtParams {
    // 12 bins/octave × 7 octaves starting at 32.7 Hz (C1) — matches the
    // doc-example in the trait.
    CqtParams::new(nzu!(12), nzu!(7), 32.7).expect("valid CqtParams")
}

fn default_chroma_params() -> ChromaParams {
    ChromaParams::music_standard()
}

fn default_mfcc_params() -> MfccParams {
    MfccParams::speech_standard()
}

// ===========================================================================
// Dispatchers
//
// Every transforms method takes `&self` (or is a static method), so we use
// `iter_batched_ref` with `BatchSize::LargeInput`. Static methods (`istft`,
// `magphase`) need bespoke setup because they don't operate on `AudioSamples`
// directly.
// ===========================================================================

/// Bench a `&self` transforms method on a 440 Hz fixture, swept across the
/// four default dtypes. `body` is invoked as `|audio: &mut AudioSamples<T>|
/// -> _` and its result is `black_box`'d.
macro_rules! dispatch_trans_unary {
    ($group:expr, $id:expr, $dt:expr, $n:expr, $ch:expr, |$audio:ident| $body:expr) => {{
        let id = $id;
        let n: usize = $n;
        let ch: usize = $ch;
        match $dt {
            "i16" => $group.bench_with_input(id, &(n, ch), |b, &(n, ch)| {
                b.iter_batched_ref(
                    || fixture_a440::<i16>(n, ch),
                    |$audio| {
                        black_box($body);
                    },
                    BatchSize::LargeInput,
                );
            }),
            "I24" => $group.bench_with_input(id, &(n, ch), |b, &(n, ch)| {
                b.iter_batched_ref(
                    || fixture_a440::<I24>(n, ch),
                    |$audio| {
                        black_box($body);
                    },
                    BatchSize::LargeInput,
                );
            }),
            "i32" => $group.bench_with_input(id, &(n, ch), |b, &(n, ch)| {
                b.iter_batched_ref(
                    || fixture_a440::<i32>(n, ch),
                    |$audio| {
                        black_box($body);
                    },
                    BatchSize::LargeInput,
                );
            }),
            "f32" => $group.bench_with_input(id, &(n, ch), |b, &(n, ch)| {
                b.iter_batched_ref(
                    || fixture_a440::<f32>(n, ch),
                    |$audio| {
                        black_box($body);
                    },
                    BatchSize::LargeInput,
                );
            }),
            _ => unreachable!("DTYPES_DEFAULT changed without updating dispatcher"),
        }
    }};
}

fn label<M: Measurement>(
    group: &mut criterion::BenchmarkGroup<'_, M>,
    catalog_id: &str,
    label_str: String,
    n: usize,
    ch: usize,
) -> BenchmarkId {
    group.throughput(Throughput::Elements((n * ch) as u64));
    BenchmarkId::new(catalog_id, label_str)
}

// ===========================================================================
// Trans-001 fft — sweep len × n_fft × dtype × channels. Cap n_fft × len ≤ 2^33
// so the 8192-point FFT on a 1 M-sample buffer is skipped (it would be ~14 s
// per call on commodity hardware).
// ===========================================================================

fn bench_trans_001_fft<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("trans");
    for (i, &len) in LENGTH_SWEEP_NO_XXXL.iter().enumerate() {
        group.sample_size(sample_size_for(SampleSizePolicy::NoFast, i));
        for &n_fft in NFFT_SWEEP {
            // Cap (n_fft * len) to keep the largest combination tractable.
            if (n_fft as u64) * (len as u64) > (1u64 << 32) {
                continue;
            }
            let n_fft_nz = NonZeroUsize::new(n_fft).unwrap();
            for &ch in CHANNELS_DEFAULT {
                for &dt in DTYPES_DEFAULT {
                    let lbl = ParamLabel::new()
                        .with("len", len)
                        .with("dt", dt)
                        .with("ch", ch)
                        .with("n_fft", n_fft)
                        .build();
                    let id = label(&mut group, "Trans-001_fft", lbl, len, ch);
                    dispatch_trans_unary!(group, id, dt, len, ch, |a| a.fft(n_fft_nz).ok());
                }
            }
        }
    }
    group.finish();
}

// ===========================================================================
// Trans-002 rfft — default impl: fft + mapv(norm). Same axes as fft.
// ===========================================================================

fn bench_trans_002_rfft<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("trans");
    for (i, &len) in LENGTH_SWEEP_NO_XXXL.iter().enumerate() {
        group.sample_size(sample_size_for(SampleSizePolicy::NoFast, i));
        for &n_fft in NFFT_SWEEP {
            if (n_fft as u64) * (len as u64) > (1u64 << 32) {
                continue;
            }
            let n_fft_nz = NonZeroUsize::new(n_fft).unwrap();
            for &ch in CHANNELS_DEFAULT {
                for &dt in DTYPES_DEFAULT {
                    let lbl = ParamLabel::new()
                        .with("len", len)
                        .with("dt", dt)
                        .with("ch", ch)
                        .with("n_fft", n_fft)
                        .build();
                    let id = label(&mut group, "Trans-002_rfft", lbl, len, ch);
                    dispatch_trans_unary!(group, id, dt, len, ch, |a| a.rfft(n_fft_nz).ok());
                }
            }
        }
    }
    group.finish();
}

// ===========================================================================
// Trans-003 stft — mono only. Sweep n_fft to expose FFT-size sensitivity;
// hop fixed at n_fft/4.
// ===========================================================================

fn bench_trans_003_stft<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("trans");
    let ch = 1;
    for (i, &len) in LENGTH_SWEEP_TRANS_STFT.iter().enumerate() {
        group.sample_size(sample_size_for(SampleSizePolicy::NoFast, i));
        for &n_fft in NFFT_SWEEP {
            if n_fft > len {
                continue;
            }
            let params = make_stft_params(n_fft, n_fft / 4);
            for &dt in DTYPES_DEFAULT {
                let lbl = ParamLabel::new()
                    .with("len", len)
                    .with("dt", dt)
                    .with("ch", ch)
                    .with("n_fft", n_fft)
                    .with("hop", n_fft / 4)
                    .build();
                let id = label(&mut group, "Trans-003_stft", lbl, len, ch);
                dispatch_trans_unary!(group, id, dt, len, ch, |a| a.stft(&params).ok());
            }
        }
    }
    group.finish();
}

// ===========================================================================
// Trans-004 istft — static method consuming `StftResult` by value. Build a
// fresh StftResult per iter via iter_batched (no `_ref`). Mono only.
// ===========================================================================

fn bench_trans_004_istft<M: Measurement>(c: &mut Criterion<M>) {
    use audio_samples::AudioSamples;
    let mut group = c.benchmark_group("trans");
    let ch = 1;
    let dt = "f32"; // istft output type doesn't change cost; one dtype is enough.
    let n_fft = DEFAULT_N_FFT;
    let hop = DEFAULT_HOP;
    let params = make_stft_params(n_fft, hop);
    for (i, &len) in LENGTH_SWEEP_TRANS_STFT.iter().enumerate() {
        group.sample_size(sample_size_for(SampleSizePolicy::NoFast, i));
        let lbl = ParamLabel::new()
            .with("len", len)
            .with("dt", dt)
            .with("ch", ch)
            .with("n_fft", n_fft)
            .with("hop", hop)
            .build();
        let id = label(&mut group, "Trans-004_istft", lbl, len, ch);
        let params_ref = &params;
        group.bench_with_input(id, &len, |b, &len| {
            b.iter_batched(
                || {
                    let audio = fixture_a440::<f32>(len, ch);
                    audio.stft(params_ref).expect("stft setup")
                },
                |stft_result| {
                    black_box(AudioSamples::<f32>::istft(stft_result).ok());
                },
                BatchSize::LargeInput,
            );
        });
    }
    group.finish();
}

// ===========================================================================
// Trans-005 linear_spectrogram — generic over AmpScale; we bench through the
// `Magnitude` instantiation (identical cost regime to the shorthand methods).
// Sweep length only — n_fft fixed at DEFAULT_N_FFT.
// ===========================================================================

fn bench_trans_005_linear_spectrogram<M: Measurement>(c: &mut Criterion<M>) {
    use spectrograms::Magnitude;
    let mut group = c.benchmark_group("trans");
    let ch = 1;
    let params = make_spec_params(DEFAULT_N_FFT, DEFAULT_HOP);
    for (i, &len) in LENGTH_SWEEP_TRANS_STFT.iter().enumerate() {
        group.sample_size(sample_size_for(SampleSizePolicy::NoFast, i));
        for &dt in DTYPES_DEFAULT {
            let lbl = ParamLabel::new()
                .with("len", len)
                .with("dt", dt)
                .with("ch", ch)
                .with("n_fft", DEFAULT_N_FFT)
                .with("hop", DEFAULT_HOP)
                .build();
            let id = label(&mut group, "Trans-005_linear_spectrogram", lbl, len, ch);
            let params_ref = &params;
            dispatch_trans_unary!(group, id, dt, len, ch, |a| {
                a.linear_spectrogram::<Magnitude>(params_ref, None).ok()
            });
        }
    }
    group.finish();
}

// ===========================================================================
// Trans-006 linear_magnitude_spectrogram — shorthand.
// ===========================================================================

fn bench_trans_006_linear_magnitude_spectrogram<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("trans");
    let ch = 1;
    let params = make_spec_params(DEFAULT_N_FFT, DEFAULT_HOP);
    for (i, &len) in LENGTH_SWEEP_TRANS_STFT.iter().enumerate() {
        group.sample_size(sample_size_for(SampleSizePolicy::NoFast, i));
        for &dt in DTYPES_DEFAULT {
            let lbl = ParamLabel::new()
                .with("len", len)
                .with("dt", dt)
                .with("ch", ch)
                .with("n_fft", DEFAULT_N_FFT)
                .with("hop", DEFAULT_HOP)
                .build();
            let id = label(
                &mut group,
                "Trans-006_linear_magnitude_spectrogram",
                lbl,
                len,
                ch,
            );
            let params_ref = &params;
            dispatch_trans_unary!(group, id, dt, len, ch, |a| {
                a.linear_magnitude_spectrogram(params_ref).ok()
            });
        }
    }
    group.finish();
}

// ===========================================================================
// Trans-007 linear_power_spectrogram — shorthand.
// ===========================================================================

fn bench_trans_007_linear_power_spectrogram<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("trans");
    let ch = 1;
    let params = make_spec_params(DEFAULT_N_FFT, DEFAULT_HOP);
    for (i, &len) in LENGTH_SWEEP_TRANS_STFT.iter().enumerate() {
        group.sample_size(sample_size_for(SampleSizePolicy::NoFast, i));
        for &dt in DTYPES_DEFAULT {
            let lbl = ParamLabel::new()
                .with("len", len)
                .with("dt", dt)
                .with("ch", ch)
                .with("n_fft", DEFAULT_N_FFT)
                .with("hop", DEFAULT_HOP)
                .build();
            let id = label(
                &mut group,
                "Trans-007_linear_power_spectrogram",
                lbl,
                len,
                ch,
            );
            let params_ref = &params;
            dispatch_trans_unary!(group, id, dt, len, ch, |a| {
                a.linear_power_spectrogram(params_ref).ok()
            });
        }
    }
    group.finish();
}

// ===========================================================================
// Trans-008 linear_db_spectrogram — shorthand; also needs `LogParams`.
// ===========================================================================

fn bench_trans_008_linear_db_spectrogram<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("trans");
    let ch = 1;
    let params = make_spec_params(DEFAULT_N_FFT, DEFAULT_HOP);
    let db = default_log_params();
    for (i, &len) in LENGTH_SWEEP_TRANS_STFT.iter().enumerate() {
        group.sample_size(sample_size_for(SampleSizePolicy::NoFast, i));
        for &dt in DTYPES_DEFAULT {
            let lbl = ParamLabel::new()
                .with("len", len)
                .with("dt", dt)
                .with("ch", ch)
                .with("n_fft", DEFAULT_N_FFT)
                .with("hop", DEFAULT_HOP)
                .build();
            let id = label(&mut group, "Trans-008_linear_db_spectrogram", lbl, len, ch);
            let params_ref = &params;
            let db_ref = &db;
            dispatch_trans_unary!(group, id, dt, len, ch, |a| {
                a.linear_db_spectrogram(params_ref, db_ref).ok()
            });
        }
    }
    group.finish();
}

// ===========================================================================
// Trans-009 log_frequency_spectrogram — generic; bench via Magnitude.
// ===========================================================================

fn bench_trans_009_log_frequency_spectrogram<M: Measurement>(c: &mut Criterion<M>) {
    use spectrograms::Magnitude;
    let mut group = c.benchmark_group("trans");
    let ch = 1;
    let params = make_spec_params(DEFAULT_N_FFT, DEFAULT_HOP);
    let loghz = default_loghz_params();
    for (i, &len) in LENGTH_SWEEP_TRANS_FILTERBANK.iter().enumerate() {
        group.sample_size(sample_size_for(SampleSizePolicy::NoFast, i));
        for &dt in DTYPES_DEFAULT {
            let lbl = ParamLabel::new()
                .with("len", len)
                .with("dt", dt)
                .with("ch", ch)
                .with("n_fft", DEFAULT_N_FFT)
                .with("hop", DEFAULT_HOP)
                .with("n_bins", 64usize)
                .build();
            let id = label(
                &mut group,
                "Trans-009_log_frequency_spectrogram",
                lbl,
                len,
                ch,
            );
            let params_ref = &params;
            let loghz_ref = &loghz;
            dispatch_trans_unary!(group, id, dt, len, ch, |a| {
                a.log_frequency_spectrogram::<Magnitude>(params_ref, loghz_ref, None)
                    .ok()
            });
        }
    }
    group.finish();
}

// ===========================================================================
// Trans-010 loghz_power_spectrogram — shorthand.
// ===========================================================================

fn bench_trans_010_loghz_power_spectrogram<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("trans");
    let ch = 1;
    let params = make_spec_params(DEFAULT_N_FFT, DEFAULT_HOP);
    let loghz = default_loghz_params();
    for (i, &len) in LENGTH_SWEEP_TRANS_FILTERBANK.iter().enumerate() {
        group.sample_size(sample_size_for(SampleSizePolicy::NoFast, i));
        for &dt in DTYPES_DEFAULT {
            let lbl = ParamLabel::new()
                .with("len", len)
                .with("dt", dt)
                .with("ch", ch)
                .with("n_fft", DEFAULT_N_FFT)
                .with("hop", DEFAULT_HOP)
                .with("n_bins", 64usize)
                .build();
            let id = label(
                &mut group,
                "Trans-010_loghz_power_spectrogram",
                lbl,
                len,
                ch,
            );
            let params_ref = &params;
            let loghz_ref = &loghz;
            dispatch_trans_unary!(group, id, dt, len, ch, |a| {
                a.loghz_power_spectrogram(params_ref, loghz_ref).ok()
            });
        }
    }
    group.finish();
}

// ===========================================================================
// Trans-011 loghz_magnitude_spectrogram — shorthand.
// ===========================================================================

fn bench_trans_011_loghz_magnitude_spectrogram<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("trans");
    let ch = 1;
    let params = make_spec_params(DEFAULT_N_FFT, DEFAULT_HOP);
    let loghz = default_loghz_params();
    for (i, &len) in LENGTH_SWEEP_TRANS_FILTERBANK.iter().enumerate() {
        group.sample_size(sample_size_for(SampleSizePolicy::NoFast, i));
        for &dt in DTYPES_DEFAULT {
            let lbl = ParamLabel::new()
                .with("len", len)
                .with("dt", dt)
                .with("ch", ch)
                .with("n_fft", DEFAULT_N_FFT)
                .with("hop", DEFAULT_HOP)
                .with("n_bins", 64usize)
                .build();
            let id = label(
                &mut group,
                "Trans-011_loghz_magnitude_spectrogram",
                lbl,
                len,
                ch,
            );
            let params_ref = &params;
            let loghz_ref = &loghz;
            dispatch_trans_unary!(group, id, dt, len, ch, |a| {
                a.loghz_magnitude_spectrogram(params_ref, loghz_ref).ok()
            });
        }
    }
    group.finish();
}

// ===========================================================================
// Trans-012 loghz_db_spectrogram — shorthand.
// ===========================================================================

fn bench_trans_012_loghz_db_spectrogram<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("trans");
    let ch = 1;
    let params = make_spec_params(DEFAULT_N_FFT, DEFAULT_HOP);
    let loghz = default_loghz_params();
    let db = default_log_params();
    for (i, &len) in LENGTH_SWEEP_TRANS_FILTERBANK.iter().enumerate() {
        group.sample_size(sample_size_for(SampleSizePolicy::NoFast, i));
        for &dt in DTYPES_DEFAULT {
            let lbl = ParamLabel::new()
                .with("len", len)
                .with("dt", dt)
                .with("ch", ch)
                .with("n_fft", DEFAULT_N_FFT)
                .with("hop", DEFAULT_HOP)
                .with("n_bins", 64usize)
                .build();
            let id = label(&mut group, "Trans-012_loghz_db_spectrogram", lbl, len, ch);
            let params_ref = &params;
            let loghz_ref = &loghz;
            let db_ref = &db;
            dispatch_trans_unary!(group, id, dt, len, ch, |a| {
                a.loghz_db_spectrogram(params_ref, loghz_ref, db_ref).ok()
            });
        }
    }
    group.finish();
}

// ===========================================================================
// Trans-013 mel_spectrogram — generic; bench via Magnitude. Filterbank tier.
// ===========================================================================

fn bench_trans_013_mel_spectrogram<M: Measurement>(c: &mut Criterion<M>) {
    use spectrograms::Magnitude;
    let mut group = c.benchmark_group("trans");
    let ch = 1;
    let params = make_spec_params(DEFAULT_N_FFT, DEFAULT_HOP);
    let mel = default_mel_params();
    for (i, &len) in LENGTH_SWEEP_TRANS_FILTERBANK.iter().enumerate() {
        group.sample_size(sample_size_for(SampleSizePolicy::NoFast, i));
        for &dt in DTYPES_DEFAULT {
            let lbl = ParamLabel::new()
                .with("len", len)
                .with("dt", dt)
                .with("ch", ch)
                .with("n_fft", DEFAULT_N_FFT)
                .with("hop", DEFAULT_HOP)
                .with("n_mels", 40usize)
                .build();
            let id = label(&mut group, "Trans-013_mel_spectrogram", lbl, len, ch);
            let params_ref = &params;
            let mel_ref = &mel;
            dispatch_trans_unary!(group, id, dt, len, ch, |a| {
                a.mel_spectrogram::<Magnitude>(params_ref, mel_ref, None)
                    .ok()
            });
        }
    }
    group.finish();
}

// ===========================================================================
// Trans-014 mel_mag_spectrogram — shorthand.
// ===========================================================================

fn bench_trans_014_mel_mag_spectrogram<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("trans");
    let ch = 1;
    let params = make_spec_params(DEFAULT_N_FFT, DEFAULT_HOP);
    let mel = default_mel_params();
    for (i, &len) in LENGTH_SWEEP_TRANS_FILTERBANK.iter().enumerate() {
        group.sample_size(sample_size_for(SampleSizePolicy::NoFast, i));
        for &dt in DTYPES_DEFAULT {
            let lbl = ParamLabel::new()
                .with("len", len)
                .with("dt", dt)
                .with("ch", ch)
                .with("n_fft", DEFAULT_N_FFT)
                .with("hop", DEFAULT_HOP)
                .with("n_mels", 40usize)
                .build();
            let id = label(&mut group, "Trans-014_mel_mag_spectrogram", lbl, len, ch);
            let params_ref = &params;
            let mel_ref = &mel;
            dispatch_trans_unary!(group, id, dt, len, ch, |a| {
                a.mel_mag_spectrogram(params_ref, mel_ref).ok()
            });
        }
    }
    group.finish();
}

// ===========================================================================
// Trans-015 mel_db_spectrogram — shorthand; needs LogParams.
// ===========================================================================

fn bench_trans_015_mel_db_spectrogram<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("trans");
    let ch = 1;
    let params = make_spec_params(DEFAULT_N_FFT, DEFAULT_HOP);
    let mel = default_mel_params();
    let db = default_log_params();
    for (i, &len) in LENGTH_SWEEP_TRANS_FILTERBANK.iter().enumerate() {
        group.sample_size(sample_size_for(SampleSizePolicy::NoFast, i));
        for &dt in DTYPES_DEFAULT {
            let lbl = ParamLabel::new()
                .with("len", len)
                .with("dt", dt)
                .with("ch", ch)
                .with("n_fft", DEFAULT_N_FFT)
                .with("hop", DEFAULT_HOP)
                .with("n_mels", 40usize)
                .build();
            let id = label(&mut group, "Trans-015_mel_db_spectrogram", lbl, len, ch);
            let params_ref = &params;
            let mel_ref = &mel;
            let db_ref = &db;
            dispatch_trans_unary!(group, id, dt, len, ch, |a| {
                a.mel_db_spectrogram(params_ref, mel_ref, db_ref).ok()
            });
        }
    }
    group.finish();
}

// ===========================================================================
// Trans-016 mel_power_spectrogram — shorthand.
// ===========================================================================

fn bench_trans_016_mel_power_spectrogram<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("trans");
    let ch = 1;
    let params = make_spec_params(DEFAULT_N_FFT, DEFAULT_HOP);
    let mel = default_mel_params();
    for (i, &len) in LENGTH_SWEEP_TRANS_FILTERBANK.iter().enumerate() {
        group.sample_size(sample_size_for(SampleSizePolicy::NoFast, i));
        for &dt in DTYPES_DEFAULT {
            let lbl = ParamLabel::new()
                .with("len", len)
                .with("dt", dt)
                .with("ch", ch)
                .with("n_fft", DEFAULT_N_FFT)
                .with("hop", DEFAULT_HOP)
                .with("n_mels", 40usize)
                .build();
            let id = label(&mut group, "Trans-016_mel_power_spectrogram", lbl, len, ch);
            let params_ref = &params;
            let mel_ref = &mel;
            dispatch_trans_unary!(group, id, dt, len, ch, |a| {
                a.mel_power_spectrogram(params_ref, mel_ref).ok()
            });
        }
    }
    group.finish();
}

// ===========================================================================
// Trans-017 mfcc — mel + DCT. Filterbank tier.
// ===========================================================================

fn bench_trans_017_mfcc<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("trans");
    let ch = 1;
    let stft = make_stft_params(DEFAULT_N_FFT, DEFAULT_HOP);
    let mfcc_params = default_mfcc_params();
    let n_mels = nzu!(40);
    for (i, &len) in LENGTH_SWEEP_TRANS_FILTERBANK.iter().enumerate() {
        group.sample_size(sample_size_for(SampleSizePolicy::NoFast, i));
        for &dt in DTYPES_DEFAULT {
            let lbl = ParamLabel::new()
                .with("len", len)
                .with("dt", dt)
                .with("ch", ch)
                .with("n_fft", DEFAULT_N_FFT)
                .with("hop", DEFAULT_HOP)
                .with("n_mels", 40usize)
                .build();
            let id = label(&mut group, "Trans-017_mfcc", lbl, len, ch);
            let stft_ref = &stft;
            let mfcc_ref = &mfcc_params;
            dispatch_trans_unary!(group, id, dt, len, ch, |a| {
                a.mfcc(stft_ref, n_mels, mfcc_ref).ok()
            });
        }
    }
    group.finish();
}

// ===========================================================================
// Trans-018 chromagram — pitch-class energy. Filterbank tier.
// ===========================================================================

fn bench_trans_018_chromagram<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("trans");
    let ch = 1;
    let stft = make_stft_params(DEFAULT_N_FFT, DEFAULT_HOP);
    let chroma = default_chroma_params();
    for (i, &len) in LENGTH_SWEEP_TRANS_FILTERBANK.iter().enumerate() {
        group.sample_size(sample_size_for(SampleSizePolicy::NoFast, i));
        for &dt in DTYPES_DEFAULT {
            let lbl = ParamLabel::new()
                .with("len", len)
                .with("dt", dt)
                .with("ch", ch)
                .with("n_fft", DEFAULT_N_FFT)
                .with("hop", DEFAULT_HOP)
                .with("n_chroma", 12usize)
                .build();
            let id = label(&mut group, "Trans-018_chromagram", lbl, len, ch);
            let stft_ref = &stft;
            let chroma_ref = &chroma;
            dispatch_trans_unary!(group, id, dt, len, ch, |a| {
                a.chromagram(stft_ref, chroma_ref).ok()
            });
        }
    }
    group.finish();
}

// ===========================================================================
// Trans-019 power_spectral_density — Welch PSD. Mono only, no filterbank;
// use the STFT length sweep with a fixed window_size.
// ===========================================================================

fn bench_trans_019_power_spectral_density<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("trans");
    let ch = 1;
    let window_size_usize = 1024usize;
    let window_size = NonZeroUsize::new(window_size_usize).unwrap();
    let overlap = 0.5_f64;
    for (i, &len) in LENGTH_SWEEP_TRANS_STFT.iter().enumerate() {
        if len < window_size_usize {
            continue;
        }
        group.sample_size(sample_size_for(SampleSizePolicy::NoFast, i));
        for &dt in DTYPES_DEFAULT {
            let lbl = ParamLabel::new()
                .with("len", len)
                .with("dt", dt)
                .with("ch", ch)
                .with("window_size", window_size_usize)
                .with("overlap", format!("{overlap:.2}"))
                .build();
            let id = label(&mut group, "Trans-019_power_spectral_density", lbl, len, ch);
            dispatch_trans_unary!(group, id, dt, len, ch, |a| {
                a.power_spectral_density(window_size, overlap).ok()
            });
        }
    }
    group.finish();
}

// ===========================================================================
// Trans-020 gammatone_spectrogram — gammatone filterbank. Filterbank tier.
// ===========================================================================

fn bench_trans_020_gammatone_spectrogram<M: Measurement>(c: &mut Criterion<M>) {
    use spectrograms::Magnitude;
    let mut group = c.benchmark_group("trans");
    let ch = 1;
    let params = make_spec_params(DEFAULT_N_FFT, DEFAULT_HOP);
    let gam = default_gammatone_params();
    for (i, &len) in LENGTH_SWEEP_TRANS_FILTERBANK.iter().enumerate() {
        group.sample_size(sample_size_for(SampleSizePolicy::NoFast, i));
        for &dt in DTYPES_DEFAULT {
            let lbl = ParamLabel::new()
                .with("len", len)
                .with("dt", dt)
                .with("ch", ch)
                .with("n_fft", DEFAULT_N_FFT)
                .with("hop", DEFAULT_HOP)
                .with("n_bands", 32usize)
                .build();
            let id = label(&mut group, "Trans-020_gammatone_spectrogram", lbl, len, ch);
            let params_ref = &params;
            let gam_ref = &gam;
            dispatch_trans_unary!(group, id, dt, len, ch, |a| {
                a.gammatone_spectrogram::<Magnitude>(params_ref, gam_ref, None)
                    .ok()
            });
        }
    }
    group.finish();
}

// ===========================================================================
// Trans-021 gammatone_magnitude_spectrogram — shorthand.
// ===========================================================================

fn bench_trans_021_gammatone_magnitude_spectrogram<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("trans");
    let ch = 1;
    let params = make_spec_params(DEFAULT_N_FFT, DEFAULT_HOP);
    let gam = default_gammatone_params();
    for (i, &len) in LENGTH_SWEEP_TRANS_FILTERBANK.iter().enumerate() {
        group.sample_size(sample_size_for(SampleSizePolicy::NoFast, i));
        for &dt in DTYPES_DEFAULT {
            let lbl = ParamLabel::new()
                .with("len", len)
                .with("dt", dt)
                .with("ch", ch)
                .with("n_fft", DEFAULT_N_FFT)
                .with("hop", DEFAULT_HOP)
                .with("n_bands", 32usize)
                .build();
            let id = label(
                &mut group,
                "Trans-021_gammatone_magnitude_spectrogram",
                lbl,
                len,
                ch,
            );
            let params_ref = &params;
            let gam_ref = &gam;
            dispatch_trans_unary!(group, id, dt, len, ch, |a| {
                a.gammatone_magnitude_spectrogram(params_ref, gam_ref).ok()
            });
        }
    }
    group.finish();
}

// ===========================================================================
// Trans-022 gammatone_power_spectrogram — shorthand.
// ===========================================================================

fn bench_trans_022_gammatone_power_spectrogram<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("trans");
    let ch = 1;
    let params = make_spec_params(DEFAULT_N_FFT, DEFAULT_HOP);
    let gam = default_gammatone_params();
    for (i, &len) in LENGTH_SWEEP_TRANS_FILTERBANK.iter().enumerate() {
        group.sample_size(sample_size_for(SampleSizePolicy::NoFast, i));
        for &dt in DTYPES_DEFAULT {
            let lbl = ParamLabel::new()
                .with("len", len)
                .with("dt", dt)
                .with("ch", ch)
                .with("n_fft", DEFAULT_N_FFT)
                .with("hop", DEFAULT_HOP)
                .with("n_bands", 32usize)
                .build();
            let id = label(
                &mut group,
                "Trans-022_gammatone_power_spectrogram",
                lbl,
                len,
                ch,
            );
            let params_ref = &params;
            let gam_ref = &gam;
            dispatch_trans_unary!(group, id, dt, len, ch, |a| {
                a.gammatone_power_spectrogram(params_ref, gam_ref).ok()
            });
        }
    }
    group.finish();
}

// ===========================================================================
// Trans-023 gammatone_db_spectrogram — shorthand; needs LogParams.
// ===========================================================================

fn bench_trans_023_gammatone_db_spectrogram<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("trans");
    let ch = 1;
    let params = make_spec_params(DEFAULT_N_FFT, DEFAULT_HOP);
    let gam = default_gammatone_params();
    let db = default_log_params();
    for (i, &len) in LENGTH_SWEEP_TRANS_FILTERBANK.iter().enumerate() {
        group.sample_size(sample_size_for(SampleSizePolicy::NoFast, i));
        for &dt in DTYPES_DEFAULT {
            let lbl = ParamLabel::new()
                .with("len", len)
                .with("dt", dt)
                .with("ch", ch)
                .with("n_fft", DEFAULT_N_FFT)
                .with("hop", DEFAULT_HOP)
                .with("n_bands", 32usize)
                .build();
            let id = label(
                &mut group,
                "Trans-023_gammatone_db_spectrogram",
                lbl,
                len,
                ch,
            );
            let params_ref = &params;
            let gam_ref = &gam;
            let db_ref = &db;
            dispatch_trans_unary!(group, id, dt, len, ch, |a| {
                a.gammatone_db_spectrogram(params_ref, gam_ref, db_ref).ok()
            });
        }
    }
    group.finish();
}

// ===========================================================================
// Trans-024 constant_q_transform — sparse-kernel CQT. Filterbank tier.
// ===========================================================================

fn bench_trans_024_constant_q_transform<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("trans");
    let ch = 1;
    let cqt = default_cqt_params();
    let hop = nzu!(256);
    for (i, &len) in LENGTH_SWEEP_TRANS_FILTERBANK.iter().enumerate() {
        group.sample_size(sample_size_for(SampleSizePolicy::NoFast, i));
        for &dt in DTYPES_DEFAULT {
            let lbl = ParamLabel::new()
                .with("len", len)
                .with("dt", dt)
                .with("ch", ch)
                .with("hop", 256usize)
                .with("n_bins", 12 * 7usize)
                .with("bins_per_octave", 12usize)
                .build();
            let id = label(&mut group, "Trans-024_constant_q_transform", lbl, len, ch);
            let cqt_ref = &cqt;
            dispatch_trans_unary!(group, id, dt, len, ch, |a| {
                a.constant_q_transform(cqt_ref, hop).ok()
            });
        }
    }
    group.finish();
}

// ===========================================================================
// Trans-025 cqt_spectrogram — generic; bench via Magnitude.
// ===========================================================================

fn bench_trans_025_cqt_spectrogram<M: Measurement>(c: &mut Criterion<M>) {
    use spectrograms::Magnitude;
    let mut group = c.benchmark_group("trans");
    let ch = 1;
    let params = make_spec_params(DEFAULT_N_FFT, DEFAULT_HOP);
    let cqt = default_cqt_params();
    for (i, &len) in LENGTH_SWEEP_TRANS_FILTERBANK.iter().enumerate() {
        group.sample_size(sample_size_for(SampleSizePolicy::NoFast, i));
        for &dt in DTYPES_DEFAULT {
            let lbl = ParamLabel::new()
                .with("len", len)
                .with("dt", dt)
                .with("ch", ch)
                .with("n_fft", DEFAULT_N_FFT)
                .with("hop", DEFAULT_HOP)
                .with("n_bins", 12 * 7usize)
                .with("bins_per_octave", 12usize)
                .build();
            let id = label(&mut group, "Trans-025_cqt_spectrogram", lbl, len, ch);
            let params_ref = &params;
            let cqt_ref = &cqt;
            dispatch_trans_unary!(group, id, dt, len, ch, |a| {
                a.cqt_spectrogram::<Magnitude>(params_ref, cqt_ref, None)
                    .ok()
            });
        }
    }
    group.finish();
}

// ===========================================================================
// Trans-026 cqt_magnitude_spectrogram — shorthand.
// ===========================================================================

fn bench_trans_026_cqt_magnitude_spectrogram<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("trans");
    let ch = 1;
    let params = make_spec_params(DEFAULT_N_FFT, DEFAULT_HOP);
    let cqt = default_cqt_params();
    for (i, &len) in LENGTH_SWEEP_TRANS_FILTERBANK.iter().enumerate() {
        group.sample_size(sample_size_for(SampleSizePolicy::NoFast, i));
        for &dt in DTYPES_DEFAULT {
            let lbl = ParamLabel::new()
                .with("len", len)
                .with("dt", dt)
                .with("ch", ch)
                .with("n_fft", DEFAULT_N_FFT)
                .with("hop", DEFAULT_HOP)
                .with("n_bins", 12 * 7usize)
                .build();
            let id = label(
                &mut group,
                "Trans-026_cqt_magnitude_spectrogram",
                lbl,
                len,
                ch,
            );
            let params_ref = &params;
            let cqt_ref = &cqt;
            dispatch_trans_unary!(group, id, dt, len, ch, |a| {
                a.cqt_magnitude_spectrogram(params_ref, cqt_ref).ok()
            });
        }
    }
    group.finish();
}

// ===========================================================================
// Trans-027 cqt_power_spectrogram — shorthand.
// ===========================================================================

fn bench_trans_027_cqt_power_spectrogram<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("trans");
    let ch = 1;
    let params = make_spec_params(DEFAULT_N_FFT, DEFAULT_HOP);
    let cqt = default_cqt_params();
    for (i, &len) in LENGTH_SWEEP_TRANS_FILTERBANK.iter().enumerate() {
        group.sample_size(sample_size_for(SampleSizePolicy::NoFast, i));
        for &dt in DTYPES_DEFAULT {
            let lbl = ParamLabel::new()
                .with("len", len)
                .with("dt", dt)
                .with("ch", ch)
                .with("n_fft", DEFAULT_N_FFT)
                .with("hop", DEFAULT_HOP)
                .with("n_bins", 12 * 7usize)
                .build();
            let id = label(&mut group, "Trans-027_cqt_power_spectrogram", lbl, len, ch);
            let params_ref = &params;
            let cqt_ref = &cqt;
            dispatch_trans_unary!(group, id, dt, len, ch, |a| {
                a.cqt_power_spectrogram(params_ref, cqt_ref).ok()
            });
        }
    }
    group.finish();
}

// ===========================================================================
// Trans-028 cqt_db_spectrogram — shorthand; needs LogParams.
// ===========================================================================

fn bench_trans_028_cqt_db_spectrogram<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("trans");
    let ch = 1;
    let params = make_spec_params(DEFAULT_N_FFT, DEFAULT_HOP);
    let cqt = default_cqt_params();
    let db = default_log_params();
    for (i, &len) in LENGTH_SWEEP_TRANS_FILTERBANK.iter().enumerate() {
        group.sample_size(sample_size_for(SampleSizePolicy::NoFast, i));
        for &dt in DTYPES_DEFAULT {
            let lbl = ParamLabel::new()
                .with("len", len)
                .with("dt", dt)
                .with("ch", ch)
                .with("n_fft", DEFAULT_N_FFT)
                .with("hop", DEFAULT_HOP)
                .with("n_bins", 12 * 7usize)
                .build();
            let id = label(&mut group, "Trans-028_cqt_db_spectrogram", lbl, len, ch);
            let params_ref = &params;
            let cqt_ref = &cqt;
            let db_ref = &db;
            dispatch_trans_unary!(group, id, dt, len, ch, |a| {
                a.cqt_db_spectrogram(params_ref, cqt_ref, db_ref).ok()
            });
        }
    }
    group.finish();
}

// ===========================================================================
// Trans-029 magphase — static helper that decomposes a complex spectrogram
// into magnitude and phase. Pre-compute one fixed-size FFT in setup and time
// only the magphase pass. Mono, fixed n_fft. Length-driven via the FFT size
// of the input matrix (here a single channel × n_fft column).
// ===========================================================================

fn bench_trans_029_magphase<M: Measurement>(c: &mut Criterion<M>) {
    use audio_samples::AudioSamples;
    let mut group = c.benchmark_group("trans");
    let ch = 1;
    let dt = "f32";
    // Sweep n_fft only — this is the matrix dimension that drives magphase
    // cost. Use the FFT signal length sweep (one column per channel) to mirror
    // a realistic post-FFT use case.
    for (i, &n_fft) in [512usize, 2048, 8192, 32_768, 131_072, 524_288]
        .iter()
        .enumerate()
    {
        // Sample-size tier: treat as NoFast across the board.
        let policy_idx = i.min(7);
        group.sample_size(sample_size_for(SampleSizePolicy::NoFast, policy_idx));
        let n_fft_nz = NonZeroUsize::new(n_fft).unwrap();
        // Use a fixture exactly the size of n_fft so there's no padding.
        let len = n_fft;
        let lbl = ParamLabel::new()
            .with("len", len)
            .with("dt", dt)
            .with("ch", ch)
            .with("n_fft", n_fft)
            .build();
        let id = label(&mut group, "Trans-029_magphase", lbl, n_fft, ch);
        group.bench_with_input(id, &n_fft, |b, &n_fft| {
            b.iter_batched_ref(
                || {
                    let audio = fixture_a440::<f32>(n_fft, ch);
                    audio.fft(n_fft_nz).expect("fft setup")
                },
                |spec| {
                    black_box(AudioSamples::<f32>::magphase(spec, None));
                },
                BatchSize::LargeInput,
            );
        });
    }
    group.finish();
}
