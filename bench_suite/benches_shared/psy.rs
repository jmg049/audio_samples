//! Shared body for the `psychoacoustic` bench targets
//! (`bench_psy_walltime`, `_instructions`, `_cycles`, `_cache_misses`,
//! `_branch_misses`).
//!
//! Each per-measurement bench-target file under `bench_suite/benches/`
//! includes this module via `#[path = "../benches_shared/psy.rs"]` and
//! invokes [`bench_all`] with its `Criterion<M>` instance.
//!
//! Conventions: see `bench_suite/METHODOLOGY.md`.
//! Function inventory: see `bench_suite/CATALOG.md` section `Psy`
//! (lines 417-487).
//!
//! ## Notes on coverage
//!
//! Many `Psy.*` symbols are free functions that operate on `f32` slices
//! rather than on [`AudioSamples<T>`]. For those rows, we build a small
//! inline RNG-driven Vec<f32> fixture and bench against `&slice`; dtype
//! is fixed at f32 (no `DTYPES_DEFAULT` sweep). Trait methods that
//! actually consume [`AudioSamples<T>`] (`analyse_psychoacoustic`,
//! `PerceptualCodec::encode`, `StereoPerceptualCodec::encode`) do sweep
//! `DTYPES_DEFAULT` even though the implementations internally convert
//! to f32 — the conversion cost is part of the realistic call cost.
//!
//! The CATALOG header mentions `BandLayout::{log_hz, linear,
//! equivalent_rectangular_bandwidth, gammatone}` but only `bark`, `mel`,
//! and `erb` exist as of this revision. The non-existent variants are
//! skipped (see report).

#![allow(clippy::too_many_lines)]

use std::num::NonZeroUsize;

use criterion::measurement::Measurement;
use criterion::{BatchSize, BenchmarkId, Criterion, Throughput};
use std::hint::black_box;

use non_empty_slice::NonEmptySlice;
use spectrograms::WindowType;

use audio_samples::{
    AudioCodec, AudioPerceptualAnalysis, BandLayout, I24, PerceptualCodec, PsychoacousticConfig,
    StereoPerceptualCodec,
    codecs::perceptual::{
        analyse_signal, analyse_signal_with_window_size,
        bands::{
            bark_to_hz, erb_to_hz, hz_to_bark, hz_to_erb, hz_to_mel, mel_to_hz, scale_band_layout,
        },
        masking::{
            MaskerType, absolute_threshold_of_hearing, apply_temporal_masking,
            classify_masker_types, compute_band_metrics, compute_smr, detect_transient_windows,
            spreading_attenuation,
        },
        quantization::{
            allocate_bits, dequantize, dequantize_band, quantize, quantize_band,
            step_size_from_allowed_noise,
        },
        reconstruct_signal,
        stereo::{mid_side_decode, mid_side_encode},
    },
    codecs::{decode as codec_decode, encode as codec_encode},
};

use bench_suite_common::{
    DTYPES_DEFAULT, LENGTH_SWEEP_NO_XXXL, ParamLabel, SampleSizePolicy, fixture_a440,
    sample_size_for, seeded_rng,
};

// ===========================================================================
// Sample rate baseline used by every fixture in this section.
// ===========================================================================

const SR: f32 = 44_100.0;

// ===========================================================================
// Helpers — RNG-driven f32 slice fixtures for free-function benches.
// ===========================================================================

/// Build a deterministic Vec<f32> of length `n` filled with values in
/// `[-amp, amp]` from the seeded bench RNG. Used for inputs to free
/// functions that take `&[f32]` slices.
fn random_f32_vec(n: usize, amp: f32) -> Vec<f32> {
    use rand::RngExt;
    let mut rng = seeded_rng();
    (0..n).map(|_| rng.random_range(-amp..amp)).collect()
}

/// Build a deterministic Vec<i32> of length `n` in `[-amp_abs, amp_abs]`.
fn random_i32_vec(n: usize, amp_abs: i32) -> Vec<i32> {
    use rand::RngExt;
    let mut rng = seeded_rng();
    (0..n).map(|_| rng.random_range(-amp_abs..amp_abs)).collect()
}

/// Build a band-energy-dB vector ramping linearly from `start_db` to `end_db`,
/// with a small per-band random perturbation so adjacent bands differ.
fn band_energy_db_vec(n_bands: usize, start_db: f32, end_db: f32) -> Vec<f32> {
    use rand::RngExt;
    let mut rng = seeded_rng();
    (0..n_bands)
        .map(|i| {
            let t = if n_bands > 1 {
                i as f32 / (n_bands - 1) as f32
            } else {
                0.0
            };
            start_db + (end_db - start_db) * t + rng.random_range(-3.0..3.0)
        })
        .collect()
}

/// Build a [`BandLayout`] of the given variant. `"bark"`/`"mel"`/`"erb"`.
fn make_layout(variant: &str, n_bands: usize, n_bins: usize) -> BandLayout {
    let nb = NonZeroUsize::new(n_bands).expect("n_bands > 0");
    let nbi = NonZeroUsize::new(n_bins).expect("n_bins > 0");
    match variant {
        "bark" => BandLayout::bark(nb, SR, nbi),
        "mel" => BandLayout::mel(nb, SR, nbi),
        "erb" => BandLayout::erb(nb, SR, nbi),
        _ => unreachable!("unknown band layout variant: {variant}"),
    }
}

/// Build a uniform-weights [`PsychoacousticConfig`] sized for `n_bands`.
fn make_mpeg1_config(n_bands: usize) -> PsychoacousticConfig {
    let nb = NonZeroUsize::new(n_bands).expect("n_bands > 0");
    let weights = PsychoacousticConfig::uniform_weights(nb);
    let weights_slice = weights.as_non_empty_slice();
    PsychoacousticConfig::mpeg1(weights_slice)
}

/// Build a fresh `PerceptualCodec` (no window switching) with `n_bands` bands.
fn make_perceptual_codec(n_bands: usize, bit_budget: u32) -> PerceptualCodec {
    let nb = NonZeroUsize::new(n_bands).expect("n_bands > 0");
    let n_bins = NonZeroUsize::new(1024).expect("n_bins > 0");
    let layout = BandLayout::bark(nb, SR, n_bins);
    let weights = PsychoacousticConfig::uniform_weights(nb);
    let config = PsychoacousticConfig::mpeg1(weights.as_non_empty_slice());
    PerceptualCodec::new(layout, config, WindowType::Hanning, bit_budget, 1)
}

/// Build a `PerceptualCodec` with window switching enabled.
#[cfg_attr(not(feature = "parallel"), allow(dead_code))]
fn make_perceptual_codec_switching(n_bands: usize, bit_budget: u32) -> PerceptualCodec {
    let nb = NonZeroUsize::new(n_bands).expect("n_bands > 0");
    let n_bins = NonZeroUsize::new(1024).expect("n_bins > 0");
    let layout = BandLayout::bark(nb, SR, n_bins);
    let weights = PsychoacousticConfig::uniform_weights(nb);
    let config = PsychoacousticConfig::mpeg1(weights.as_non_empty_slice());
    PerceptualCodec::with_window_switching(
        layout,
        config,
        WindowType::Hanning,
        bit_budget,
        1,
        NonZeroUsize::new(2048).expect("nz"),
        NonZeroUsize::new(256).expect("nz"),
        8.0,
    )
}

/// Build a fresh `StereoPerceptualCodec`.
fn make_stereo_codec(n_bands: usize, mid_budget: u32, side_budget: u32) -> StereoPerceptualCodec {
    let nb = NonZeroUsize::new(n_bands).expect("n_bands > 0");
    let n_bins = NonZeroUsize::new(1024).expect("n_bins > 0");
    let layout = BandLayout::bark(nb, SR, n_bins);
    let weights = PsychoacousticConfig::uniform_weights(nb);
    let config = PsychoacousticConfig::mpeg1(weights.as_non_empty_slice());
    StereoPerceptualCodec::new(
        layout,
        config,
        WindowType::Hanning,
        mid_budget,
        side_budget,
        1,
    )
}

/// `n_bands` parameter sweep for band-count-dependent ops.
const N_BANDS_SWEEP: &[usize] = &[8, 24, 64];

// ===========================================================================
// Dispatch macros for `AudioSamples<T>`-shaped trait benches.
// ===========================================================================

macro_rules! dispatch_analyse_sine {
    ($group:expr, $id:expr, $dt:expr, $n:expr, $window:expr, $layout:expr, $config:expr) => {{
        let id = $id;
        let n = $n;
        // Bind once at this scope, then take references inside each match arm so
        // the four arm closures borrow disjointly (each arm runs in its own scope).
        let window_owned = $window;
        let layout_owned = $layout;
        let config_owned = $config;
        match $dt {
            "i16" => {
                let window_ref = &window_owned;
                let layout_ref = &layout_owned;
                let config_ref = &config_owned;
                $group.bench_with_input(id, &n, |b, &n| {
                    b.iter_batched_ref(
                        || fixture_a440::<i16>(n, 1),
                        |a| {
                            black_box(a.analyse_psychoacoustic(window_ref.clone(), layout_ref, config_ref).ok());
                        },
                        BatchSize::LargeInput,
                    );
                });
            }
            "I24" => {
                let window_ref = &window_owned;
                let layout_ref = &layout_owned;
                let config_ref = &config_owned;
                $group.bench_with_input(id, &n, |b, &n| {
                    b.iter_batched_ref(
                        || fixture_a440::<I24>(n, 1),
                        |a| {
                            black_box(a.analyse_psychoacoustic(window_ref.clone(), layout_ref, config_ref).ok());
                        },
                        BatchSize::LargeInput,
                    );
                });
            }
            "i32" => {
                let window_ref = &window_owned;
                let layout_ref = &layout_owned;
                let config_ref = &config_owned;
                $group.bench_with_input(id, &n, |b, &n| {
                    b.iter_batched_ref(
                        || fixture_a440::<i32>(n, 1),
                        |a| {
                            black_box(a.analyse_psychoacoustic(window_ref.clone(), layout_ref, config_ref).ok());
                        },
                        BatchSize::LargeInput,
                    );
                });
            }
            "f32" => {
                let window_ref = &window_owned;
                let layout_ref = &layout_owned;
                let config_ref = &config_owned;
                $group.bench_with_input(id, &n, |b, &n| {
                    b.iter_batched_ref(
                        || fixture_a440::<f32>(n, 1),
                        |a| {
                            black_box(a.analyse_psychoacoustic(window_ref.clone(), layout_ref, config_ref).ok());
                        },
                        BatchSize::LargeInput,
                    );
                });
            }
            _ => unreachable!("DTYPES_DEFAULT changed without updating dispatcher"),
        }
    }};
}

// `PerceptualCodec::encode` consumes `self`. Build a fresh codec each iter via
// `iter_batched` (not `_ref`); the audio fixture is the second tuple element.
macro_rules! dispatch_codec_encode_sine {
    ($group:expr, $id:expr, $dt:expr, $n:expr, $codec_fn:expr) => {{
        let id = $id;
        let n = $n;
        let codec_fn = $codec_fn;
        match $dt {
            "i16" => $group.bench_with_input(id, &n, |b, &n| {
                b.iter_batched(
                    || (codec_fn(), fixture_a440::<i16>(n, 1)),
                    |(codec, audio)| {
                        black_box(codec.encode(&audio).ok());
                    },
                    BatchSize::LargeInput,
                );
            }),
            "I24" => $group.bench_with_input(id, &n, |b, &n| {
                b.iter_batched(
                    || (codec_fn(), fixture_a440::<I24>(n, 1)),
                    |(codec, audio)| {
                        black_box(codec.encode(&audio).ok());
                    },
                    BatchSize::LargeInput,
                );
            }),
            "i32" => $group.bench_with_input(id, &n, |b, &n| {
                b.iter_batched(
                    || (codec_fn(), fixture_a440::<i32>(n, 1)),
                    |(codec, audio)| {
                        black_box(codec.encode(&audio).ok());
                    },
                    BatchSize::LargeInput,
                );
            }),
            "f32" => $group.bench_with_input(id, &n, |b, &n| {
                b.iter_batched(
                    || (codec_fn(), fixture_a440::<f32>(n, 1)),
                    |(codec, audio)| {
                        black_box(codec.encode(&audio).ok());
                    },
                    BatchSize::LargeInput,
                );
            }),
            _ => unreachable!("DTYPES_DEFAULT changed without updating dispatcher"),
        }
    }};
}

// `StereoPerceptualCodec::encode` — same shape, stereo fixture.
macro_rules! dispatch_stereo_codec_encode_sine {
    ($group:expr, $id:expr, $dt:expr, $n:expr, $codec_fn:expr) => {{
        let id = $id;
        let n = $n;
        let codec_fn = $codec_fn;
        match $dt {
            "i16" => $group.bench_with_input(id, &n, |b, &n| {
                b.iter_batched(
                    || (codec_fn(), fixture_a440::<i16>(n, 2)),
                    |(codec, audio)| {
                        black_box(codec.encode(&audio).ok());
                    },
                    BatchSize::LargeInput,
                );
            }),
            "I24" => $group.bench_with_input(id, &n, |b, &n| {
                b.iter_batched(
                    || (codec_fn(), fixture_a440::<I24>(n, 2)),
                    |(codec, audio)| {
                        black_box(codec.encode(&audio).ok());
                    },
                    BatchSize::LargeInput,
                );
            }),
            "i32" => $group.bench_with_input(id, &n, |b, &n| {
                b.iter_batched(
                    || (codec_fn(), fixture_a440::<i32>(n, 2)),
                    |(codec, audio)| {
                        black_box(codec.encode(&audio).ok());
                    },
                    BatchSize::LargeInput,
                );
            }),
            "f32" => $group.bench_with_input(id, &n, |b, &n| {
                b.iter_batched(
                    || (codec_fn(), fixture_a440::<f32>(n, 2)),
                    |(codec, audio)| {
                        black_box(codec.encode(&audio).ok());
                    },
                    BatchSize::LargeInput,
                );
            }),
            _ => unreachable!("DTYPES_DEFAULT changed without updating dispatcher"),
        }
    }};
}

// ===========================================================================
// Top-level entry point
// ===========================================================================

pub fn bench_all<M: Measurement>(c: &mut Criterion<M>) {
    // Psy.Trait
    bench_psy_001_analyse_psychoacoustic(c);
    // Psy.Codec
    bench_psy_002_perceptual_codec_encode(c);
    #[cfg(feature = "parallel")]
    bench_psy_002p_perceptual_codec_encode_switching_parallel(c);
    bench_psy_003_perceptual_codec_decode(c);
    bench_psy_004_stereo_codec_encode(c);
    bench_psy_005_stereo_codec_decode(c);
    bench_psy_006_codec_encode_free_fn(c);
    bench_psy_007_codec_decode_free_fn(c);
    bench_psy_008_analyse_signal(c);
    bench_psy_009_analyse_signal_with_window_size(c);
    bench_psy_010_reconstruct_signal(c);
    // Psy.Bands
    bench_psy_011_hz_to_bark(c);
    bench_psy_012_bark_to_hz(c);
    bench_psy_013_hz_to_mel(c);
    bench_psy_014_mel_to_hz(c);
    bench_psy_015_hz_to_erb(c);
    bench_psy_016_erb_to_hz(c);
    bench_psy_017_scale_band_layout(c);
    bench_psy_018_bandlayout_bark(c);
    bench_psy_019_bandlayout_mel(c);
    bench_psy_020_bandlayout_erb(c);
    // Psy.Masking
    bench_psy_021_classify_masker_types(c);
    bench_psy_022_absolute_threshold_of_hearing(c);
    bench_psy_023_spreading_attenuation(c);
    bench_psy_024_compute_band_metrics(c);
    bench_psy_025_apply_temporal_masking(c);
    bench_psy_026_detect_transient_windows(c);
    bench_psy_027_compute_smr(c);
    // Psy.Quantization
    bench_psy_028_step_size_from_allowed_noise(c);
    bench_psy_029_allocate_bits(c);
    bench_psy_030_quantize_band(c);
    bench_psy_031_dequantize_band(c);
    bench_psy_032_quantize(c);
    bench_psy_033_dequantize(c);
    // Psy.Stereo
    bench_psy_034_mid_side_encode(c);
    bench_psy_035_mid_side_decode(c);
}

// ===========================================================================
// Psy-001 — `AudioPerceptualAnalysis::analyse_psychoacoustic`
//
// MDCT-driven; mono only. Length-driven. Sweep DTYPES_DEFAULT (conversion to
// f32 happens internally, captured by the measurement).
// ===========================================================================

fn bench_psy_001_analyse_psychoacoustic<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("psy");
    let layout = make_layout("bark", 24, 1024);
    let config = make_mpeg1_config(24);
    let ch = 1; // mono required
    for (i, &len) in LENGTH_SWEEP_NO_XXXL.iter().enumerate() {
        group.sample_size(sample_size_for(SampleSizePolicy::NoFast, i));
        for &dt in DTYPES_DEFAULT {
            group.throughput(Throughput::Elements((len * ch) as u64));
            let label = ParamLabel::new()
                .with("len", len).with("dt", dt).with("ch", ch)
                .with("n_bands", 24usize).with("layout", "bark")
                .build();
            let id = BenchmarkId::new("Psy-001_analyse_psychoacoustic", label);
            dispatch_analyse_sine!(group, id, dt, len, WindowType::Hanning, &layout, &config);
        }
    }
    group.finish();
}

// ===========================================================================
// Psy-002 — `PerceptualCodec::encode` (no window switching).
// ===========================================================================

fn bench_psy_002_perceptual_codec_encode<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("psy");
    let ch = 1;
    for (i, &len) in LENGTH_SWEEP_NO_XXXL.iter().enumerate() {
        group.sample_size(sample_size_for(SampleSizePolicy::NoFast, i));
        for &dt in DTYPES_DEFAULT {
            group.throughput(Throughput::Elements((len * ch) as u64));
            let label = ParamLabel::new()
                .with("len", len).with("dt", dt).with("ch", ch)
                .with("n_bands", 24usize).with("switching", "off")
                .build();
            let id = BenchmarkId::new("Psy-002_PerceptualCodec_encode", label);
            dispatch_codec_encode_sine!(group, id, dt, len, || make_perceptual_codec(24, 128_000));
        }
    }
    group.finish();
}

// Variant gated on `parallel` — window-switching path that uses rayon when on.
// Bench bodies are otherwise identical; the `switching` axis distinguishes them.
#[cfg(feature = "parallel")]
fn bench_psy_002p_perceptual_codec_encode_switching_parallel<M: Measurement>(
    c: &mut Criterion<M>,
) {
    let mut group = c.benchmark_group("psy");
    let ch = 1;
    for (i, &len) in LENGTH_SWEEP_NO_XXXL.iter().enumerate() {
        group.sample_size(sample_size_for(SampleSizePolicy::NoFast, i));
        for &dt in DTYPES_DEFAULT {
            group.throughput(Throughput::Elements((len * ch) as u64));
            let label = ParamLabel::new()
                .with("len", len).with("dt", dt).with("ch", ch)
                .with("n_bands", 24usize).with("switching", "on_parallel")
                .build();
            let id = BenchmarkId::new("Psy-002_PerceptualCodec_encode", label);
            dispatch_codec_encode_sine!(
                group,
                id,
                dt,
                len,
                || make_perceptual_codec_switching(24, 128_000)
            );
        }
    }
    group.finish();
}

// ===========================================================================
// Psy-003 — `PerceptualCodec::decode` (static, consumes `PerceptualEncodedAudio`).
// Pre-encode once per setup; iter_batched clones the encoded payload per call.
// ===========================================================================

fn bench_psy_003_perceptual_codec_decode<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("psy");
    let ch = 1;
    let dt = "f32";
    for (i, &len) in LENGTH_SWEEP_NO_XXXL.iter().enumerate() {
        group.sample_size(sample_size_for(SampleSizePolicy::NoFast, i));
        // Pre-build the encoded payload for this length.
        let codec = make_perceptual_codec(24, 128_000);
        let audio = fixture_a440::<f32>(len, 1);
        let encoded = match codec.encode(&audio) {
            Ok(e) => e,
            Err(_) => continue,
        };
        group.throughput(Throughput::Elements((len * ch) as u64));
        let label = ParamLabel::new()
            .with("len", len).with("dt", dt).with("ch", ch)
            .with("n_bands", 24usize)
            .build();
        let id = BenchmarkId::new("Psy-003_PerceptualCodec_decode", label);
        group.bench_with_input(id, &encoded, |b, encoded| {
            b.iter_batched(
                || encoded.clone(),
                |enc| {
                    black_box(<PerceptualCodec as AudioCodec>::decode::<f32>(enc).ok());
                },
                BatchSize::LargeInput,
            );
        });
    }
    group.finish();
}

// ===========================================================================
// Psy-004 — `StereoPerceptualCodec::encode`. M/S front-end + 2× mono encode.
// ===========================================================================

fn bench_psy_004_stereo_codec_encode<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("psy");
    let ch = 2;
    for (i, &len) in LENGTH_SWEEP_NO_XXXL.iter().enumerate() {
        group.sample_size(sample_size_for(SampleSizePolicy::NoFast, i));
        for &dt in DTYPES_DEFAULT {
            group.throughput(Throughput::Elements((len * ch) as u64));
            let label = ParamLabel::new()
                .with("len", len).with("dt", dt).with("ch", ch)
                .with("n_bands", 24usize)
                .build();
            let id = BenchmarkId::new("Psy-004_StereoPerceptualCodec_encode", label);
            dispatch_stereo_codec_encode_sine!(
                group,
                id,
                dt,
                len,
                || make_stereo_codec(24, 128_000, 64_000)
            );
        }
    }
    group.finish();
}

// ===========================================================================
// Psy-005 — `StereoPerceptualCodec::decode`.
// ===========================================================================

fn bench_psy_005_stereo_codec_decode<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("psy");
    let ch = 2;
    let dt = "f32";
    for (i, &len) in LENGTH_SWEEP_NO_XXXL.iter().enumerate() {
        group.sample_size(sample_size_for(SampleSizePolicy::NoFast, i));
        let codec = make_stereo_codec(24, 128_000, 64_000);
        let audio = fixture_a440::<f32>(len, 2);
        let encoded = match codec.encode(&audio) {
            Ok(e) => e,
            Err(_) => continue,
        };
        group.throughput(Throughput::Elements((len * ch) as u64));
        let label = ParamLabel::new()
            .with("len", len).with("dt", dt).with("ch", ch)
            .with("n_bands", 24usize)
            .build();
        let id = BenchmarkId::new("Psy-005_StereoPerceptualCodec_decode", label);
        group.bench_with_input(id, &encoded, |b, encoded| {
            b.iter_batched(
                || encoded.clone(),
                |enc| {
                    black_box(
                        <StereoPerceptualCodec as AudioCodec>::decode::<f32>(enc).ok(),
                    );
                },
                BatchSize::LargeInput,
            );
        });
    }
    group.finish();
}

// ===========================================================================
// Psy-006 — `codec::encode` free function (thin wrapper over `codec.encode`).
// Bench overhead vs the trait dispatch path.
// ===========================================================================

fn bench_psy_006_codec_encode_free_fn<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("psy");
    let ch = 1;
    for (i, &len) in LENGTH_SWEEP_NO_XXXL.iter().enumerate() {
        group.sample_size(sample_size_for(SampleSizePolicy::NoFast, i));
        for &dt in DTYPES_DEFAULT {
            group.throughput(Throughput::Elements((len * ch) as u64));
            let label = ParamLabel::new()
                .with("len", len).with("dt", dt).with("ch", ch)
                .with("n_bands", 24usize)
                .build();
            let id = BenchmarkId::new("Psy-006_codec_encode_free_fn", label);
            // Reproduce dispatch_codec_encode_sine inline, but call the free fn.
            match dt {
                "i16" => group.bench_with_input(id, &len, |b, &n| {
                    b.iter_batched(
                        || (make_perceptual_codec(24, 128_000), fixture_a440::<i16>(n, 1)),
                        |(codec, audio)| {
                            black_box(codec_encode(&audio, codec).ok());
                        },
                        BatchSize::LargeInput,
                    );
                }),
                "I24" => group.bench_with_input(id, &len, |b, &n| {
                    b.iter_batched(
                        || (make_perceptual_codec(24, 128_000), fixture_a440::<I24>(n, 1)),
                        |(codec, audio)| {
                            black_box(codec_encode(&audio, codec).ok());
                        },
                        BatchSize::LargeInput,
                    );
                }),
                "i32" => group.bench_with_input(id, &len, |b, &n| {
                    b.iter_batched(
                        || (make_perceptual_codec(24, 128_000), fixture_a440::<i32>(n, 1)),
                        |(codec, audio)| {
                            black_box(codec_encode(&audio, codec).ok());
                        },
                        BatchSize::LargeInput,
                    );
                }),
                "f32" => group.bench_with_input(id, &len, |b, &n| {
                    b.iter_batched(
                        || (make_perceptual_codec(24, 128_000), fixture_a440::<f32>(n, 1)),
                        |(codec, audio)| {
                            black_box(codec_encode(&audio, codec).ok());
                        },
                        BatchSize::LargeInput,
                    );
                }),
                _ => unreachable!(),
            };
        }
    }
    group.finish();
}

// ===========================================================================
// Psy-007 — `codec::decode` free function.
// ===========================================================================

fn bench_psy_007_codec_decode_free_fn<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("psy");
    let ch = 1;
    let dt = "f32";
    for (i, &len) in LENGTH_SWEEP_NO_XXXL.iter().enumerate() {
        group.sample_size(sample_size_for(SampleSizePolicy::NoFast, i));
        let codec = make_perceptual_codec(24, 128_000);
        let audio = fixture_a440::<f32>(len, 1);
        let encoded = match codec.encode(&audio) {
            Ok(e) => e,
            Err(_) => continue,
        };
        group.throughput(Throughput::Elements((len * ch) as u64));
        let label = ParamLabel::new()
            .with("len", len).with("dt", dt).with("ch", ch)
            .with("n_bands", 24usize)
            .build();
        let id = BenchmarkId::new("Psy-007_codec_decode_free_fn", label);
        group.bench_with_input(id, &encoded, |b, encoded| {
            b.iter_batched(
                || encoded.clone(),
                |enc| {
                    black_box(codec_decode::<PerceptualCodec, f32>(enc).ok());
                },
                BatchSize::LargeInput,
            );
        });
    }
    group.finish();
}

// ===========================================================================
// Psy-008 — `analyse_signal` free function (auto window size).
// ===========================================================================

fn bench_psy_008_analyse_signal<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("psy");
    let layout = make_layout("bark", 24, 1024);
    let config = make_mpeg1_config(24);
    let ch = 1;
    for (i, &len) in LENGTH_SWEEP_NO_XXXL.iter().enumerate() {
        group.sample_size(sample_size_for(SampleSizePolicy::NoFast, i));
        for &dt in DTYPES_DEFAULT {
            group.throughput(Throughput::Elements((len * ch) as u64));
            let label = ParamLabel::new()
                .with("len", len).with("dt", dt).with("ch", ch)
                .with("n_bands", 24usize)
                .build();
            let id = BenchmarkId::new("Psy-008_analyse_signal", label);
            match dt {
                "i16" => group.bench_with_input(id, &len, |b, &n| {
                    b.iter_batched_ref(
                        || fixture_a440::<i16>(n, 1),
                        |a| {
                            black_box(analyse_signal(a, WindowType::Hanning, &layout, &config).ok());
                        },
                        BatchSize::LargeInput,
                    );
                }),
                "I24" => group.bench_with_input(id, &len, |b, &n| {
                    b.iter_batched_ref(
                        || fixture_a440::<I24>(n, 1),
                        |a| {
                            black_box(analyse_signal(a, WindowType::Hanning, &layout, &config).ok());
                        },
                        BatchSize::LargeInput,
                    );
                }),
                "i32" => group.bench_with_input(id, &len, |b, &n| {
                    b.iter_batched_ref(
                        || fixture_a440::<i32>(n, 1),
                        |a| {
                            black_box(analyse_signal(a, WindowType::Hanning, &layout, &config).ok());
                        },
                        BatchSize::LargeInput,
                    );
                }),
                "f32" => group.bench_with_input(id, &len, |b, &n| {
                    b.iter_batched_ref(
                        || fixture_a440::<f32>(n, 1),
                        |a| {
                            black_box(analyse_signal(a, WindowType::Hanning, &layout, &config).ok());
                        },
                        BatchSize::LargeInput,
                    );
                }),
                _ => unreachable!(),
            };
        }
    }
    group.finish();
}

// ===========================================================================
// Psy-009 — `analyse_signal_with_window_size` (explicit window size).
// Sweep window_size axis: {512, 2048}.
// ===========================================================================

fn bench_psy_009_analyse_signal_with_window_size<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("psy");
    let config = make_mpeg1_config(24);
    let ch = 1;
    let dt = "f32";
    for (i, &len) in LENGTH_SWEEP_NO_XXXL.iter().enumerate() {
        group.sample_size(sample_size_for(SampleSizePolicy::NoFast, i));
        for &ws in &[512usize, 2048] {
            // Window size cannot exceed length; skip combos that would error.
            if ws > len {
                continue;
            }
            let ws_nz = NonZeroUsize::new(ws).expect("ws > 0");
            // n_bins = ws / 2; rebuild layout so the band layout matches the
            // bins for this window size.
            let n_bins = NonZeroUsize::new(ws / 2).expect("ws/2 > 0");
            let layout = BandLayout::bark(
                NonZeroUsize::new(24).expect("nz"),
                SR,
                n_bins,
            );
            group.throughput(Throughput::Elements((len * ch) as u64));
            let label = ParamLabel::new()
                .with("len", len).with("dt", dt).with("ch", ch)
                .with("n_bands", 24usize).with("window_size", ws)
                .build();
            let id = BenchmarkId::new("Psy-009_analyse_signal_with_window_size", label);
            group.bench_with_input(id, &len, |b, &n| {
                b.iter_batched_ref(
                    || fixture_a440::<f32>(n, 1),
                    |a| {
                        black_box(
                            analyse_signal_with_window_size(
                                a,
                                WindowType::Hanning,
                                Some(ws_nz),
                                &layout,
                                &config,
                            )
                            .ok(),
                        );
                    },
                    BatchSize::LargeInput,
                );
            });
        }
    }
    group.finish();
}

// ===========================================================================
// Psy-010 — `reconstruct_signal` (IMDCT + OLA). Pre-analyse so we have a real
// `PerceptualAnalysisResult` to feed back through. Length-driven via the
// underlying signal length; dtype fixed at f32.
// ===========================================================================

fn bench_psy_010_reconstruct_signal<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("psy");
    let layout = make_layout("bark", 24, 1024);
    let config = make_mpeg1_config(24);
    let ch = 1;
    let dt = "f32";
    for (i, &len) in LENGTH_SWEEP_NO_XXXL.iter().enumerate() {
        group.sample_size(sample_size_for(SampleSizePolicy::NoFast, i));
        let audio = fixture_a440::<f32>(len, 1);
        let result = match analyse_signal(&audio, WindowType::Hanning, &layout, &config) {
            Ok(r) => r,
            Err(_) => continue,
        };
        group.throughput(Throughput::Elements((len * ch) as u64));
        let label = ParamLabel::new()
            .with("len", len).with("dt", dt).with("ch", ch)
            .with("n_bands", 24usize)
            .with("n_coefficients", result.n_coefficients.get())
            .with("n_frames", result.n_frames.get())
            .build();
        let id = BenchmarkId::new("Psy-010_reconstruct_signal", label);
        let n_coefficients = result.n_coefficients;
        let n_frames = result.n_frames;
        let original_length = result.original_length;
        let sample_rate = result.sample_rate;
        let mdct_params = result.mdct_params.clone();
        let coefficients = result.coefficients.clone();
        group.bench_with_input(id, &(), |b, _| {
            b.iter(|| {
                black_box(
                    reconstruct_signal(
                        &coefficients,
                        n_coefficients,
                        n_frames,
                        &mdct_params,
                        Some(original_length),
                        sample_rate,
                    )
                    .ok(),
                );
            });
        });
    }
    group.finish();
}

// ===========================================================================
// Psy-011 / 012 / 013 / 014 / 015 / 016 — scalar Hz<->scale conversions.
//
// Single-call functions; per-iter cost is ~nanoseconds. Bench a single point
// each — there is no length/dtype axis to sweep.
// ===========================================================================

fn bench_psy_011_hz_to_bark<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("psy");
    group.sample_size(sample_size_for(SampleSizePolicy::AlwaysDefault, 0));
    let label = ParamLabel::new().with("scalar", "true").build();
    let id = BenchmarkId::new("Psy-011_hz_to_bark", label);
    group.bench_function(id, |b| {
        b.iter(|| {
            black_box(hz_to_bark(black_box(1000.0_f32)));
        });
    });
    group.finish();
}

fn bench_psy_012_bark_to_hz<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("psy");
    group.sample_size(sample_size_for(SampleSizePolicy::AlwaysDefault, 0));
    let label = ParamLabel::new().with("scalar", "true").build();
    let id = BenchmarkId::new("Psy-012_bark_to_hz", label);
    group.bench_function(id, |b| {
        b.iter(|| {
            black_box(bark_to_hz(black_box(8.5_f32)));
        });
    });
    group.finish();
}

fn bench_psy_013_hz_to_mel<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("psy");
    group.sample_size(sample_size_for(SampleSizePolicy::AlwaysDefault, 0));
    let label = ParamLabel::new().with("scalar", "true").build();
    let id = BenchmarkId::new("Psy-013_bands_hz_to_mel", label);
    group.bench_function(id, |b| {
        b.iter(|| {
            black_box(hz_to_mel(black_box(1000.0_f32)));
        });
    });
    group.finish();
}

fn bench_psy_014_mel_to_hz<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("psy");
    group.sample_size(sample_size_for(SampleSizePolicy::AlwaysDefault, 0));
    let label = ParamLabel::new().with("scalar", "true").build();
    let id = BenchmarkId::new("Psy-014_bands_mel_to_hz", label);
    group.bench_function(id, |b| {
        b.iter(|| {
            black_box(mel_to_hz(black_box(1000.0_f32)));
        });
    });
    group.finish();
}

fn bench_psy_015_hz_to_erb<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("psy");
    group.sample_size(sample_size_for(SampleSizePolicy::AlwaysDefault, 0));
    let label = ParamLabel::new().with("scalar", "true").build();
    let id = BenchmarkId::new("Psy-015_hz_to_erb", label);
    group.bench_function(id, |b| {
        b.iter(|| {
            black_box(hz_to_erb(black_box(1000.0_f32)));
        });
    });
    group.finish();
}

fn bench_psy_016_erb_to_hz<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("psy");
    group.sample_size(sample_size_for(SampleSizePolicy::AlwaysDefault, 0));
    let label = ParamLabel::new().with("scalar", "true").build();
    let id = BenchmarkId::new("Psy-016_erb_to_hz", label);
    group.bench_function(id, |b| {
        b.iter(|| {
            black_box(erb_to_hz(black_box(200.0_f32)));
        });
    });
    group.finish();
}

// ===========================================================================
// Psy-017 — `scale_band_layout`. Cost ∝ n_bands. Sweep `n_bands` and bin ratio.
// ===========================================================================

fn bench_psy_017_scale_band_layout<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("psy");
    group.sample_size(sample_size_for(SampleSizePolicy::AlwaysDefault, 0));
    for &n_bands in N_BANDS_SWEEP {
        let layout = make_layout("bark", n_bands, 1024);
        for &(from_n_bins, to_n_bins) in &[(1024usize, 128usize), (1024, 512), (1024, 2048)] {
            let from = NonZeroUsize::new(from_n_bins).expect("nz");
            let to = NonZeroUsize::new(to_n_bins).expect("nz");
            let label = ParamLabel::new()
                .with("n_bands", n_bands)
                .with("from_n_bins", from_n_bins)
                .with("to_n_bins", to_n_bins)
                .build();
            let id = BenchmarkId::new("Psy-017_scale_band_layout", label);
            group.throughput(Throughput::Elements(n_bands as u64));
            group.bench_with_input(id, &(), |b, _| {
                b.iter(|| {
                    black_box(scale_band_layout(&layout, from, to));
                });
            });
        }
    }
    group.finish();
}

// ===========================================================================
// Psy-018 / 019 / 020 — `BandLayout::{bark, mel, erb}` constructors.
// Sweep `n_bands` and `n_bins`.
// ===========================================================================

fn bench_band_layout_ctor<M: Measurement, F>(
    c: &mut Criterion<M>,
    catalog_id: &'static str,
    variant_label: &'static str,
    build: F,
) where
    F: Fn(NonZeroUsize, f32, NonZeroUsize) -> BandLayout,
{
    let mut group = c.benchmark_group("psy");
    group.sample_size(sample_size_for(SampleSizePolicy::AlwaysDefault, 0));
    for &n_bands in N_BANDS_SWEEP {
        for &n_bins in &[256usize, 1024, 4096] {
            let nb = NonZeroUsize::new(n_bands).expect("nz");
            let nbi = NonZeroUsize::new(n_bins).expect("nz");
            let label = ParamLabel::new()
                .with("n_bands", n_bands)
                .with("n_bins", n_bins)
                .with("layout", variant_label)
                .build();
            let id = BenchmarkId::new(catalog_id, label);
            group.throughput(Throughput::Elements(n_bands as u64));
            group.bench_with_input(id, &(), |b, _| {
                b.iter(|| {
                    black_box(build(nb, SR, nbi));
                });
            });
        }
    }
    group.finish();
}

fn bench_psy_018_bandlayout_bark<M: Measurement>(c: &mut Criterion<M>) {
    bench_band_layout_ctor(c, "Psy-018_BandLayout_bark", "bark", BandLayout::bark);
}

fn bench_psy_019_bandlayout_mel<M: Measurement>(c: &mut Criterion<M>) {
    bench_band_layout_ctor(c, "Psy-019_BandLayout_mel", "mel", BandLayout::mel);
}

fn bench_psy_020_bandlayout_erb<M: Measurement>(c: &mut Criterion<M>) {
    bench_band_layout_ctor(c, "Psy-020_BandLayout_erb", "erb", BandLayout::erb);
}

// ===========================================================================
// Psy-021 — `classify_masker_types`. Sweep n_bands; input is band_energy_db
// (a Vec<f32> of length n_bands).
// ===========================================================================

fn bench_psy_021_classify_masker_types<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("psy");
    group.sample_size(sample_size_for(SampleSizePolicy::AlwaysDefault, 0));
    for &n_bands in N_BANDS_SWEEP {
        let energies = band_energy_db_vec(n_bands, -40.0, 20.0);
        let label = ParamLabel::new().with("n_bands", n_bands).build();
        let id = BenchmarkId::new("Psy-021_classify_masker_types", label);
        group.throughput(Throughput::Elements(n_bands as u64));
        group.bench_with_input(id, &energies, |b, e| {
            b.iter(|| {
                black_box(classify_masker_types(e.as_slice(), 7.0));
            });
        });
    }
    group.finish();
}

// ===========================================================================
// Psy-022 — `absolute_threshold_of_hearing`. Scalar fn, single point.
// ===========================================================================

fn bench_psy_022_absolute_threshold_of_hearing<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("psy");
    group.sample_size(sample_size_for(SampleSizePolicy::AlwaysDefault, 0));
    let label = ParamLabel::new().with("scalar", "true").build();
    let id = BenchmarkId::new("Psy-022_absolute_threshold_of_hearing", label);
    group.bench_function(id, |b| {
        b.iter(|| {
            black_box(absolute_threshold_of_hearing(black_box(1000.0_f32)));
        });
    });
    group.finish();
}

// ===========================================================================
// Psy-023 — `spreading_attenuation`. Scalar inner kernel. Single point.
// ===========================================================================

fn bench_psy_023_spreading_attenuation<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("psy");
    group.sample_size(sample_size_for(SampleSizePolicy::AlwaysDefault, 0));
    let config = make_mpeg1_config(24);
    let label = ParamLabel::new().with("scalar", "true").build();
    let id = BenchmarkId::new("Psy-023_spreading_attenuation", label);
    group.bench_function(id, |b| {
        b.iter(|| {
            black_box(spreading_attenuation(
                black_box(8.0_f32),
                black_box(10.0_f32),
                MaskerType::Tonal,
                &config,
            ));
        });
    });
    group.finish();
}

// ===========================================================================
// Psy-024 — `compute_band_metrics`. Sweep n_bins and n_bands (and layout
// variant). Cost is O(n_bins) (band aggregation) + O(n_bands²) (spreading).
// ===========================================================================

fn bench_psy_024_compute_band_metrics<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("psy");
    group.sample_size(sample_size_for(SampleSizePolicy::AlwaysDefault, 0));
    for &n_bands in N_BANDS_SWEEP {
        for &n_bins in &[256usize, 1024, 4096] {
            for &layout_kind in &["bark", "mel", "erb"] {
                let layout = make_layout(layout_kind, n_bands, n_bins);
                let config = make_mpeg1_config(n_bands);
                let energies = random_f32_vec(n_bins, 1.0);
                let label = ParamLabel::new()
                    .with("n_bands", n_bands)
                    .with("n_bins", n_bins)
                    .with("layout", layout_kind)
                    .build();
                let id = BenchmarkId::new("Psy-024_compute_band_metrics", label);
                group.throughput(Throughput::Elements(n_bins as u64));
                group.bench_with_input(id, &energies, |b, e| {
                    b.iter(|| {
                        black_box(compute_band_metrics(
                            e.as_slice(),
                            &layout,
                            &config,
                            n_bins,
                        ));
                    });
                });
            }
        }
    }
    group.finish();
}

// ===========================================================================
// Psy-025 — `apply_temporal_masking`. Pre-build a Vec<BandMetrics> by running
// `compute_band_metrics` once per frame; sweep n_frames and n_bands.
// ===========================================================================

fn bench_psy_025_apply_temporal_masking<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("psy");
    group.sample_size(sample_size_for(SampleSizePolicy::AlwaysDefault, 0));
    for &n_bands in N_BANDS_SWEEP {
        let n_bins = 1024usize;
        let layout = make_layout("bark", n_bands, n_bins);
        let config = make_mpeg1_config(n_bands);
        for &n_frames in &[16usize, 64, 256] {
            // Synthesize per-frame metrics with varying energies.
            let frame_metrics: Vec<_> = (0..n_frames)
                .map(|f| {
                    let amp = 0.5 + 0.5 * ((f as f32 * 0.1).sin());
                    let mut e = random_f32_vec(n_bins, amp);
                    // Slight bias by frame so per-frame metrics differ.
                    for v in e.iter_mut() {
                        *v *= amp;
                    }
                    compute_band_metrics(&e, &layout, &config, n_bins)
                })
                .collect();
            let label = ParamLabel::new()
                .with("n_bands", n_bands)
                .with("n_frames", n_frames)
                .build();
            let id = BenchmarkId::new("Psy-025_apply_temporal_masking", label);
            group.throughput(Throughput::Elements((n_bands * n_frames) as u64));
            group.bench_with_input(id, &frame_metrics, |b, frames| {
                b.iter(|| {
                    black_box(apply_temporal_masking(frames, &config, 23.2, 0.15, 3.0));
                });
            });
        }
    }
    group.finish();
}

// ===========================================================================
// Psy-026 — `detect_transient_windows`. Length-driven; mono f32 only.
// ===========================================================================

fn bench_psy_026_detect_transient_windows<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("psy");
    let dt = "f32";
    let ch = 1;
    let window_size = NonZeroUsize::new(2048).expect("nz");
    let hop_size = NonZeroUsize::new(1024).expect("nz");
    for (i, &len) in LENGTH_SWEEP_NO_XXXL.iter().enumerate() {
        // Need at least one window's worth of samples.
        if len < window_size.get() {
            continue;
        }
        group.sample_size(sample_size_for(SampleSizePolicy::NoFast, i));
        let samples_vec = random_f32_vec(len, 0.5);
        group.throughput(Throughput::Elements((len * ch) as u64));
        let label = ParamLabel::new()
            .with("len", len).with("dt", dt).with("ch", ch)
            .with("window_size", window_size.get())
            .with("hop_size", hop_size.get())
            .build();
        let id = BenchmarkId::new("Psy-026_detect_transient_windows", label);
        group.bench_with_input(id, &samples_vec, |b, v| {
            b.iter(|| {
                let s = NonEmptySlice::from_slice(v.as_slice()).expect("non-empty");
                black_box(detect_transient_windows(s, window_size, hop_size, 8.0));
            });
        });
    }
    group.finish();
}

// ===========================================================================
// Psy-027 — `compute_smr`. Scalar subtract. Single point.
// ===========================================================================

fn bench_psy_027_compute_smr<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("psy");
    group.sample_size(sample_size_for(SampleSizePolicy::AlwaysDefault, 0));
    let label = ParamLabel::new().with("scalar", "true").build();
    let id = BenchmarkId::new("Psy-027_compute_smr", label);
    group.bench_function(id, |b| {
        b.iter(|| {
            black_box(compute_smr(black_box(20.0_f32), black_box(-5.0_f32)));
        });
    });
    group.finish();
}

// ===========================================================================
// Psy-028 — `step_size_from_allowed_noise`. Closed-form scalar. Single point.
// ===========================================================================

fn bench_psy_028_step_size_from_allowed_noise<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("psy");
    group.sample_size(sample_size_for(SampleSizePolicy::AlwaysDefault, 0));
    let label = ParamLabel::new().with("scalar", "true").build();
    let id = BenchmarkId::new("Psy-028_step_size_from_allowed_noise", label);
    group.bench_function(id, |b| {
        b.iter(|| {
            black_box(step_size_from_allowed_noise(black_box(-20.0_f32)));
        });
    });
    group.finish();
}

// ===========================================================================
// Psy-029 — `allocate_bits`. Sweep n_bands and bit_budget.
// ===========================================================================

fn bench_psy_029_allocate_bits<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("psy");
    group.sample_size(sample_size_for(SampleSizePolicy::AlwaysDefault, 0));
    for &n_bands in N_BANDS_SWEEP {
        // Build BandMetrics by running compute_band_metrics once.
        let n_bins = 1024usize;
        let layout = make_layout("bark", n_bands, n_bins);
        let config = make_mpeg1_config(n_bands);
        let energies = random_f32_vec(n_bins, 1.0);
        let metrics = compute_band_metrics(&energies, &layout, &config, n_bins);
        for &budget in &[32_000u32, 128_000, 512_000] {
            let label = ParamLabel::new()
                .with("n_bands", n_bands)
                .with("bit_budget", budget)
                .build();
            let id = BenchmarkId::new("Psy-029_allocate_bits", label);
            group.throughput(Throughput::Elements(n_bands as u64));
            group.bench_with_input(id, &metrics, |b, m| {
                b.iter(|| {
                    black_box(allocate_bits(m, budget, 1));
                });
            });
        }
    }
    group.finish();
}

// ===========================================================================
// Psy-030 — `quantize_band`. Length-swept over per-band coefficient count.
// ===========================================================================

fn bench_psy_030_quantize_band<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("psy");
    // Bench length here is "n_coeffs in band". Use a modest sweep — bands are
    // small (tens to thousands of bins).
    for (i, &n_coeffs) in [64usize, 256, 1024, 4096, 16_384].iter().enumerate() {
        let policy = if i <= 2 {
            SampleSizePolicy::FastSmall
        } else {
            SampleSizePolicy::NoFast
        };
        // map to length-index for tier selection
        group.sample_size(sample_size_for(policy, i.min(7)));
        let coeffs = random_f32_vec(n_coeffs, 1.0);
        let step = 0.001_f32;
        let label = ParamLabel::new()
            .with("n_coeffs", n_coeffs)
            .with("step", format!("{step:.4}"))
            .build();
        let id = BenchmarkId::new("Psy-030_quantize_band", label);
        group.throughput(Throughput::Elements(n_coeffs as u64));
        group.bench_with_input(id, &coeffs, |b, c| {
            b.iter(|| {
                black_box(quantize_band(c.as_slice(), step));
            });
        });
    }
    group.finish();
}

// ===========================================================================
// Psy-031 — `dequantize_band`.
// ===========================================================================

fn bench_psy_031_dequantize_band<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("psy");
    for (i, &n_coeffs) in [64usize, 256, 1024, 4096, 16_384].iter().enumerate() {
        let policy = if i <= 2 {
            SampleSizePolicy::FastSmall
        } else {
            SampleSizePolicy::NoFast
        };
        group.sample_size(sample_size_for(policy, i.min(7)));
        let q = random_i32_vec(n_coeffs, 1000);
        let step = 0.001_f32;
        let label = ParamLabel::new()
            .with("n_coeffs", n_coeffs)
            .with("step", format!("{step:.4}"))
            .build();
        let id = BenchmarkId::new("Psy-031_dequantize_band", label);
        group.throughput(Throughput::Elements(n_coeffs as u64));
        group.bench_with_input(id, &q, |b, q| {
            b.iter(|| {
                black_box(dequantize_band(q.as_slice(), step));
            });
        });
    }
    group.finish();
}

// ===========================================================================
// Psy-032 — `quantize` (whole-frame). Sweep (n_coefficients × n_frames) and
// n_bands.
// ===========================================================================

fn bench_psy_032_quantize<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("psy");
    for &n_bands in N_BANDS_SWEEP {
        let n_bins = 1024usize;
        let layout = make_layout("bark", n_bands, n_bins);
        let config = make_mpeg1_config(n_bands);
        // Random bin energies → metrics → allocation.
        let energies = random_f32_vec(n_bins, 1.0);
        let metrics = compute_band_metrics(&energies, &layout, &config, n_bins);
        let allocation = allocate_bits(&metrics, 128_000, 1);
        for (i, &n_frames) in [4usize, 16, 64, 256].iter().enumerate() {
            let policy = if i <= 1 {
                SampleSizePolicy::FastSmall
            } else {
                SampleSizePolicy::NoFast
            };
            group.sample_size(sample_size_for(policy, i.min(7)));
            let total = n_bins * n_frames;
            let coeffs = random_f32_vec(total, 1.0);
            // SAFETY: total > 0 (n_bins > 0, n_frames > 0).
            let coeffs_ne = NonEmptySlice::from_slice(&coeffs).expect("non-empty");
            let label = ParamLabel::new()
                .with("n_bands", n_bands)
                .with("n_coefficients", n_bins)
                .with("n_frames", n_frames)
                .build();
            let id = BenchmarkId::new("Psy-032_quantize", label);
            group.throughput(Throughput::Elements(total as u64));
            let nc = NonZeroUsize::new(n_bins).expect("nz");
            let nf = NonZeroUsize::new(n_frames).expect("nz");
            group.bench_with_input(id, &(), |b, _| {
                b.iter(|| {
                    black_box(quantize(coeffs_ne, nc, nf, &allocation));
                });
            });
        }
    }
    group.finish();
}

// ===========================================================================
// Psy-033 — `dequantize`.
// ===========================================================================

fn bench_psy_033_dequantize<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("psy");
    for &n_bands in N_BANDS_SWEEP {
        let n_bins = 1024usize;
        let layout = make_layout("bark", n_bands, n_bins);
        let config = make_mpeg1_config(n_bands);
        let energies = random_f32_vec(n_bins, 1.0);
        let metrics = compute_band_metrics(&energies, &layout, &config, n_bins);
        let allocation = allocate_bits(&metrics, 128_000, 1);
        for (i, &n_frames) in [4usize, 16, 64, 256].iter().enumerate() {
            let policy = if i <= 1 {
                SampleSizePolicy::FastSmall
            } else {
                SampleSizePolicy::NoFast
            };
            group.sample_size(sample_size_for(policy, i.min(7)));
            let total = n_bins * n_frames;
            let q = random_i32_vec(total, 1000);
            let q_ne = NonEmptySlice::from_slice(&q).expect("non-empty");
            let label = ParamLabel::new()
                .with("n_bands", n_bands)
                .with("n_coefficients", n_bins)
                .with("n_frames", n_frames)
                .build();
            let id = BenchmarkId::new("Psy-033_dequantize", label);
            group.throughput(Throughput::Elements(total as u64));
            let nc = NonZeroUsize::new(n_bins).expect("nz");
            let nf = NonZeroUsize::new(n_frames).expect("nz");
            group.bench_with_input(id, &(), |b, _| {
                b.iter(|| {
                    black_box(dequantize(q_ne, nc, nf, &allocation));
                });
            });
        }
    }
    group.finish();
}

// ===========================================================================
// Psy-034 — `mid_side_encode` over f32 L/R slices. Length-swept; dtype fixed.
// ===========================================================================

fn bench_psy_034_mid_side_encode<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("psy");
    let dt = "f32";
    for (i, &len) in LENGTH_SWEEP_NO_XXXL.iter().enumerate() {
        // FastSmall-ish at short lengths (sub-µs trivial sweep), NoFast otherwise.
        let policy = if i <= 2 {
            SampleSizePolicy::FastSmall
        } else {
            SampleSizePolicy::NoFast
        };
        group.sample_size(sample_size_for(policy, i));
        let left = random_f32_vec(len, 0.5);
        let right = random_f32_vec(len, 0.5);
        let label = ParamLabel::new()
            .with("len", len)
            .with("dt", dt)
            .with("ch", 2usize)
            .build();
        let id = BenchmarkId::new("Psy-034_mid_side_encode", label);
        // Two output slices of length `len` each.
        group.throughput(Throughput::Elements((len * 2) as u64));
        group.bench_with_input(id, &(left, right), |b, lr| {
            b.iter(|| {
                let (l, r) = lr;
                black_box(mid_side_encode(l.as_slice(), r.as_slice()));
            });
        });
    }
    group.finish();
}

// ===========================================================================
// Psy-035 — `mid_side_decode`. Same shape; mid/side instead of L/R.
// ===========================================================================

fn bench_psy_035_mid_side_decode<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("psy");
    let dt = "f32";
    for (i, &len) in LENGTH_SWEEP_NO_XXXL.iter().enumerate() {
        let policy = if i <= 2 {
            SampleSizePolicy::FastSmall
        } else {
            SampleSizePolicy::NoFast
        };
        group.sample_size(sample_size_for(policy, i));
        let mid = random_f32_vec(len, 0.5);
        let side = random_f32_vec(len, 0.5);
        let label = ParamLabel::new()
            .with("len", len)
            .with("dt", dt)
            .with("ch", 2usize)
            .build();
        let id = BenchmarkId::new("Psy-035_mid_side_decode", label);
        group.throughput(Throughput::Elements((len * 2) as u64));
        group.bench_with_input(id, &(mid, side), |b, ms| {
            b.iter(|| {
                let (m, s) = ms;
                black_box(mid_side_decode(m.as_slice(), s.as_slice()));
            });
        });
    }
    group.finish();
}

