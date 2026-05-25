//! Shared body for the `AudioPlotting` bench targets
//! (`bench_plot_walltime`, `_instructions`, `_cycles`, `_cache_misses`,
//! `_branch_misses`).
//!
//! Each per-measurement bench-target file under `bench_suite/benches/`
//! includes this module via `#[path = "../benches_shared/plot.rs"]` and
//! invokes [`bench_all`] with its `Criterion<M>` instance.
//!
//! Conventions: see `bench_suite/METHODOLOGY.md`.
//! Function inventory: see `bench_suite/CATALOG.md` section `Plot`
//! (lines 401-415).
//!
//! Notes (per CATALOG Open Q 14):
//! - Plot construction is cheap relative to the actual rendering pipeline
//!   (Plotly HTML emission, base64 encoding) which dominates wall-time when
//!   exercised. The catalog rows only cover construction, so we black-box
//!   the returned plot object and do not render/serialize.
//! - `AudioPlotting` requires `AudioTransforms` as a supertrait, so the
//!   `plotting` feature pulls in `transforms` transitively. The bench
//!   targets still need to declare `plotting` (and indirectly `transforms`
//!   via that dependency) in `required-features`.
//! - We exercise a single dtype (f32) — plot construction does not depend
//!   on the input sample type beyond the implicit `as_float()` conversion
//!   the impl performs.

use criterion::measurement::Measurement;
use criterion::{BatchSize, BenchmarkId, Criterion, Throughput};
use std::hint::black_box;

use audio_samples::operations::plotting::{
    MagnitudeSpectrumParams, SpectrogramPlotParams, WaveformPlotParams,
};
use audio_samples::operations::traits::AudioPlotting;

use bench_suite_common::{
    LENGTH_SWEEP_NO_XXXL, ParamLabel, SampleSizePolicy, fixture_a440, sample_size_for,
};

// ===========================================================================
// Top-level entry point — wrappers call into this with their typed Criterion.
// ===========================================================================

pub fn bench_all<M: Measurement>(c: &mut Criterion<M>) {
    bench_plot_001_plot_waveform(c);
    bench_plot_002_plot_spectrogram(c);
    bench_plot_003_plot_magnitude_spectrum(c);
}

// ===========================================================================
// Plot-001 plot_waveform — NoFast tier, mono f32 only. Construction cost
// only; the returned `WaveformPlot` is `black_box`'d (rendering / HTML
// generation is out of scope per CATALOG Open Q 14).
// ===========================================================================

fn bench_plot_001_plot_waveform<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("plot");
    let ch = 1;
    let dt = "f32";
    for (i, &len) in LENGTH_SWEEP_NO_XXXL.iter().enumerate() {
        group.sample_size(sample_size_for(SampleSizePolicy::NoFast, i));
        group.throughput(Throughput::Elements((len * ch) as u64));
        let label = ParamLabel::new()
            .with("len", len)
            .with("dt", dt)
            .with("ch", ch)
            .build();
        let id = BenchmarkId::new("Plot-001_plot_waveform", label);
        group.bench_with_input(id, &(len, ch), |b, &(n, ch)| {
            let params = WaveformPlotParams::default();
            b.iter_batched_ref(
                || fixture_a440::<f32>(n, ch),
                |audio| {
                    black_box(audio.plot_waveform(&params).ok());
                },
                BatchSize::LargeInput,
            );
        });
    }
    group.finish();
}

// ===========================================================================
// Plot-002 plot_spectrogram — NoFast tier, mono f32 only. Uses the default
// `mel_db` preset (n_fft = 2048, hop = 512, Hanning, 128 mel bands). The
// underlying STFT + mel projection dominates cost at moderate-to-large
// lengths; constructing the Plotly heatmap object is comparatively cheap.
// ===========================================================================

fn bench_plot_002_plot_spectrogram<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("plot");
    let ch = 1;
    let dt = "f32";
    // Spectrogram needs at least one frame; skip lengths shorter than the
    // default n_fft (2048).
    for (i, &len) in LENGTH_SWEEP_NO_XXXL.iter().enumerate() {
        if len < 2048 {
            continue;
        }
        group.sample_size(sample_size_for(SampleSizePolicy::NoFast, i));
        group.throughput(Throughput::Elements((len * ch) as u64));
        let label = ParamLabel::new()
            .with("len", len)
            .with("dt", dt)
            .with("ch", ch)
            .with("preset", "mel_db")
            .build();
        let id = BenchmarkId::new("Plot-002_plot_spectrogram", label);
        group.bench_with_input(id, &(len, ch), |b, &(n, ch)| {
            let params = SpectrogramPlotParams::mel_db();
            b.iter_batched_ref(
                || fixture_a440::<f32>(n, ch),
                |audio| {
                    black_box(audio.plot_spectrogram(&params).ok());
                },
                BatchSize::LargeInput,
            );
        });
    }
    group.finish();
}

// ===========================================================================
// Plot-003 plot_magnitude_spectrum — NoFast tier, mono f32 only. FFT-driven;
// `n_fft = None` defaults to next power of 2 >= signal length so cost grows
// as `O(n log n)`. We use the default `db()` preset (db_scale = true).
// ===========================================================================

fn bench_plot_003_plot_magnitude_spectrum<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("plot");
    let ch = 1;
    let dt = "f32";
    for (i, &len) in LENGTH_SWEEP_NO_XXXL.iter().enumerate() {
        group.sample_size(sample_size_for(SampleSizePolicy::NoFast, i));
        group.throughput(Throughput::Elements((len * ch) as u64));
        let label = ParamLabel::new()
            .with("len", len)
            .with("dt", dt)
            .with("ch", ch)
            .with("db_scale", "true")
            .build();
        let id = BenchmarkId::new("Plot-003_plot_magnitude_spectrum", label);
        group.bench_with_input(id, &(len, ch), |b, &(n, ch)| {
            let params = MagnitudeSpectrumParams::db();
            b.iter_batched_ref(
                || fixture_a440::<f32>(n, ch),
                |audio| {
                    black_box(audio.plot_magnitude_spectrum(&params).ok());
                },
                BatchSize::LargeInput,
            );
        });
    }
    group.finish();
}
