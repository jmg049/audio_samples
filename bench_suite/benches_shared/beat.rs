//! Shared body for the `AudioBeatTracking` bench targets
//! (`bench_beat_walltime`, `_instructions`, `_cycles`, `_cache_misses`,
//! `_branch_misses`).
//!
//! Each per-measurement bench-target file under `bench_suite/benches/`
//! includes this module via `#[path = "../benches_shared/beat.rs"]` and
//! invokes [`bench_all`] with its `Criterion<M>` instance.
//!
//! Conventions: see `bench_suite/METHODOLOGY.md`.
//! Function inventory: see `bench_suite/CATALOG.md` section `Beat`
//! (lines 351-361).
//!
//! ## Parameter axes
//!
//! The catalog row lists `n_fft, hop` as the cost axes for Beat-002.
//! `OnsetDetectionConfig` is CQT-based and exposes those parameters as
//! `window_size` (the analysis window in samples; `n_fft`-equivalent) and
//! `hop_size`.  The sweep `n_fft ∈ {1024, 2048}` × `hop ∈ {256, 512}`
//! covers the realtime/musical defaults.

use criterion::measurement::Measurement;
use criterion::{BatchSize, BenchmarkId, Criterion, Throughput};
use std::hint::black_box;
use std::num::NonZeroUsize;

use audio_samples::I24;
use audio_samples::operations::beat::{
    BeatTrackingConfig, onset_strength_envelope, track_beats_core,
};
use audio_samples::operations::onset::OnsetDetectionConfig;
use audio_samples::operations::traits::AudioBeatTracking;
use non_empty_slice::NonEmptyVec;

use bench_suite_common::{
    BENCH_SAMPLE_RATE_HZ, CHANNELS_DEFAULT, DTYPES_DEFAULT, LENGTH_SWEEP_NO_XXXL,
    ParamLabel, SampleSizePolicy, fixture_a440, sample_size_for, seeded_rng,
};
use rand::Rng;

// ===========================================================================
// Top-level entry point — wrappers call into this with their typed Criterion.
// ===========================================================================

pub fn bench_all<M: Measurement>(c: &mut Criterion<M>) {
    bench_beat_001_detect_beats(c);
    bench_beat_002_onset_strength_envelope(c);
    bench_beat_003_track_beats_core(c);
}

// ===========================================================================
// Dispatch macro — DTYPES_DEFAULT-typed expansion for a unary 440 Hz fixture.
// `onset` benches inject a fresh OnsetDetectionConfig per dispatch.
// ===========================================================================

macro_rules! dispatch_unary_sine {
    ($group:expr, $id:expr, $dt:expr, $n:expr, $ch:expr, |$audio:ident| $body:expr) => {{
        let id = $id;
        let n = $n;
        let ch = $ch;
        match $dt {
            "i16" => $group.bench_with_input(id, &(n, ch), |b, &(n, ch)| {
                b.iter_batched_ref(
                    || fixture_a440::<i16>(n, ch),
                    |$audio| { black_box($body); },
                    BatchSize::LargeInput,
                );
            }),
            "I24" => $group.bench_with_input(id, &(n, ch), |b, &(n, ch)| {
                b.iter_batched_ref(
                    || fixture_a440::<I24>(n, ch),
                    |$audio| { black_box($body); },
                    BatchSize::LargeInput,
                );
            }),
            "i32" => $group.bench_with_input(id, &(n, ch), |b, &(n, ch)| {
                b.iter_batched_ref(
                    || fixture_a440::<i32>(n, ch),
                    |$audio| { black_box($body); },
                    BatchSize::LargeInput,
                );
            }),
            "f32" => $group.bench_with_input(id, &(n, ch), |b, &(n, ch)| {
                b.iter_batched_ref(
                    || fixture_a440::<f32>(n, ch),
                    |$audio| { black_box($body); },
                    BatchSize::LargeInput,
                );
            }),
            _ => unreachable!("DTYPES_DEFAULT changed without updating dispatcher"),
        }
    }};
}

/// Build an [`OnsetDetectionConfig`] derived from the `musical()` preset with
/// `window_size` and `hop_size` overridden.  Returns the configured value
/// (the underlying type is `#[non_exhaustive]` but fields are public).
fn cfg_with_window_hop(window: usize, hop: usize) -> OnsetDetectionConfig {
    let mut cfg = OnsetDetectionConfig::musical();
    cfg.window_size = Some(NonZeroUsize::new(window).expect("window > 0"));
    cfg.hop_size = NonZeroUsize::new(hop).expect("hop > 0");
    cfg
}

// ===========================================================================
// Beat-001 detect_beats — end-to-end pipeline (ODF + DP step).
// NoFast tier, lengths capped at XXL (FFT-cost prohibitive).
// ===========================================================================

fn bench_beat_001_detect_beats<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("beat");
    for (i, &len) in LENGTH_SWEEP_NO_XXXL.iter().enumerate() {
        group.sample_size(sample_size_for(SampleSizePolicy::NoFast, i));
        // Skip lengths smaller than the largest analysis window we use —
        // the underlying ODF requires at least one full window.
        if len < 2048 {
            continue;
        }
        for &n_fft in &[1024usize, 2048usize] {
            // Skip combinations where the window is larger than the signal.
            if n_fft > len {
                continue;
            }
            let hop = n_fft / 4;
            let onset_cfg = cfg_with_window_hop(n_fft, hop);
            let beat_cfg = BeatTrackingConfig::new(120.0, None, onset_cfg);
            for &ch in CHANNELS_DEFAULT {
                for &dt in DTYPES_DEFAULT {
                    group.throughput(Throughput::Elements((len * ch) as u64));
                    let label = ParamLabel::new()
                        .with("len", len).with("dt", dt).with("ch", ch)
                        .with("n_fft", n_fft).with("hop", hop).build();
                    let id = BenchmarkId::new("Beat-001_detect_beats", label);
                    dispatch_unary_sine!(
                        group, id, dt, len, ch, |a| a.detect_beats(&beat_cfg).ok()
                    );
                }
            }
        }
    }
    group.finish();
}

// ===========================================================================
// Beat-002 onset_strength_envelope — ODF helper.
// NoFast tier, LENGTH_SWEEP_NO_XXXL (FFT-dominated).
// ===========================================================================

fn bench_beat_002_onset_strength_envelope<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("beat");
    for (i, &len) in LENGTH_SWEEP_NO_XXXL.iter().enumerate() {
        group.sample_size(sample_size_for(SampleSizePolicy::NoFast, i));
        if len < 2048 {
            continue;
        }
        for &n_fft in &[1024usize, 2048usize] {
            if n_fft > len {
                continue;
            }
            let hop = n_fft / 4;
            let cfg = cfg_with_window_hop(n_fft, hop);
            for &ch in CHANNELS_DEFAULT {
                for &dt in DTYPES_DEFAULT {
                    group.throughput(Throughput::Elements((len * ch) as u64));
                    let label = ParamLabel::new()
                        .with("len", len).with("dt", dt).with("ch", ch)
                        .with("n_fft", n_fft).with("hop", hop).build();
                    let id = BenchmarkId::new("Beat-002_onset_strength_envelope", label);
                    dispatch_unary_sine!(
                        group, id, dt, len, ch,
                        |a| onset_strength_envelope(a, &cfg, None).ok()
                    );
                }
            }
        }
    }
    group.finish();
}

// ===========================================================================
// Beat-003 track_beats_core — DP step over a synthetic onset envelope.
// Cost scales with `n_frames` and is independent of dtype/channels.
// ===========================================================================

fn bench_beat_003_track_beats_core<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("beat");
    // Use NO_XXXL frame counts (sweep is over n_frames, not raw samples).
    // The DP walks the envelope once per beat — cheap even at large n_frames.
    let hop_size = NonZeroUsize::new(512).unwrap();
    let sr = BENCH_SAMPLE_RATE_HZ as f64;
    for (i, &n_frames) in LENGTH_SWEEP_NO_XXXL.iter().enumerate() {
        group.sample_size(sample_size_for(SampleSizePolicy::NoFast, i));
        for &tempo_bpm in &[90.0_f64, 120.0, 180.0] {
            group.throughput(Throughput::Elements(n_frames as u64));
            let label = ParamLabel::new()
                .with("n_frames", n_frames)
                .with("tempo_bpm", format!("{tempo_bpm:.0}"))
                .build();
            let id = BenchmarkId::new("Beat-003_track_beats_core", label);
            group.bench_with_input(id, &(n_frames, tempo_bpm), |b, &(n, t)| {
                b.iter_batched_ref(
                    || {
                        use rand::RngExt;
                        let mut rng = seeded_rng();
                        let mut v = vec![0.0_f64; n];
                        // Sparse periodic peaks plus a small noise floor — gives
                        // the DP something to peak-pick at every step.
                        for i in 0..n {
                            v[i] = rng.random_range(0.0..0.05);
                        }
                        for i in (0..n).step_by(20.max(1)) {
                            v[i] = 1.0;
                        }
                        NonEmptyVec::new(v).expect("n > 0 guaranteed by sweep")
                    },
                    |onset| {
                        black_box(
                            track_beats_core(onset, t, sr, hop_size, None).ok()
                        );
                    },
                    BatchSize::LargeInput,
                );
            });
        }
    }
    group.finish();
}
