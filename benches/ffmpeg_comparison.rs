//! FFmpeg CLI vs `audio_samples` library benchmark suite.
//!
//! ## What is being measured
//!
//! Each benchmark group contains paired measurements:
//!
//! - `audio_samples/*` — in-process Rust operations on pre-built `AudioSamples`
//!   objects (no I/O, no process overhead).
//! - `ffmpeg/*` — `std::process::Command` shell-out including process fork/exec,
//!   WAV bytes written to stdin, filter/codec processing, and pipe teardown.
//!
//! ## Framing
//!
//! `audio_samples` operates entirely in-memory within the calling program.
//! FFmpeg is measured as the realistic "shell-out" cost — what any program pays
//! when it uses FFmpeg as a pipeline stage rather than embedding a library.
//! These are not pure algorithm comparisons; they measure two different
//! integration strategies for the same audio transformation.
//!
//! ## Running
//!
//! ```bash
//! cargo bench --bench ffmpeg_comparison \
//!   --no-default-features \
//!   --features "resampling,iir-filtering,channels,processing,statistics,editing"
//! ```
//!
//! Run a single group only (e.g. resampling):
//!
//! ```bash
//! cargo bench --bench ffmpeg_comparison \
//!   --no-default-features \
//!   --features "resampling,iir-filtering,channels,processing,statistics,editing" \
//!   -- resampling
//! ```

use audio_samples::{
    AudioChannelOps, AudioIirFiltering, AudioSamples, AudioStatistics, AudioTypeConversion,
    StandardSample,
    operations::{ResamplingQuality, types::MonoConversionMethod},
    resample, sample_rate, sine_wave, stereo_sine_wave,
};
use audio_samples_io::traits::{AudioStreamWrite, AudioStreamWriter};
use audio_samples_io::wav::StreamedWavWriter;
use criterion::{BatchSize, BenchmarkId, Criterion, criterion_group, criterion_main};
use std::hint::black_box;
use std::io::Write;
use std::num::NonZeroUsize;
use std::process::{Command, Stdio};
use std::time::Duration;

// ---------------------------------------------------------------------------
// WAV encoding helper (uses audio_samples_io)
// ---------------------------------------------------------------------------

/// Encode any `AudioSamples<T>` to an in-memory WAV byte vector.
///
/// Multi-channel audio is automatically interleaved by `StreamedWavWriter`.
/// The output is always written as 32-bit IEEE float.
fn audio_to_wav_bytes<T>(audio: &AudioSamples<T>) -> Vec<u8>
where
    T: StandardSample + 'static,
{
    let channels = audio.num_channels().get() as u16;
    let sample_rate = audio.sample_rate().get();

    let mut buffer = Vec::new();
    {
        let cursor = std::io::Cursor::new(&mut buffer);
        let mut writer = StreamedWavWriter::new_f32(cursor, channels, sample_rate)
            .expect("StreamedWavWriter creation failed");
        writer.write_frames(audio).expect("write_frames failed");
        writer.finalize().expect("finalize failed");
    }
    buffer
}

// ---------------------------------------------------------------------------
// FFmpeg helpers
// ---------------------------------------------------------------------------

/// Returns `true` if `ffmpeg` is on PATH and responds to `ffmpeg -version`.
fn ffmpeg_available() -> bool {
    Command::new("ffmpeg")
        .arg("-version")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

/// Print the installed FFmpeg version once at benchmark startup.
fn print_ffmpeg_version() {
    match Command::new("ffmpeg").arg("-version").output() {
        Ok(out) => {
            let text = String::from_utf8_lossy(&out.stdout);
            let first = text.lines().next().unwrap_or("(unknown)");
            println!("FFmpeg: {first}");
        }
        Err(_) => println!("FFmpeg: not found on PATH — FFmpeg benchmarks will be skipped"),
    }
}

/// Spawn `ffmpeg`, pipe `wav_bytes` to stdin, apply `extra_args`, and wait.
///
/// Both stdout and stderr are discarded. Returns the exit status so callers
/// can wrap it in `black_box`.
fn run_ffmpeg_stdin(wav_bytes: &[u8], extra_args: &[&str]) -> std::process::ExitStatus {
    let mut child = Command::new("ffmpeg")
        .arg("-y")
        .arg("-i")
        .arg("pipe:0")
        .args(extra_args)
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .expect("failed to spawn ffmpeg — is it installed and on PATH?");

    child
        .stdin
        .as_mut()
        .expect("stdin pipe was requested")
        .write_all(wav_bytes)
        .expect("failed to write WAV bytes to ffmpeg stdin");

    child.wait().expect("ffmpeg process did not complete")
}

// ---------------------------------------------------------------------------
// 1. Resampling
// ---------------------------------------------------------------------------

fn bench_resampling(c: &mut Criterion) {
    let have_ffmpeg = ffmpeg_available();

    // (label, duration_secs, dst_hz, quality)
    let cases: &[(&str, u64, u32, ResamplingQuality)] = &[
        ("1s_44100_to_16000_fast", 1, 16000, ResamplingQuality::Fast),
        (
            "1s_44100_to_16000_medium",
            1,
            16000,
            ResamplingQuality::Medium,
        ),
        ("1s_44100_to_16000_high", 1, 16000, ResamplingQuality::High),
        ("1s_44100_to_22050_fast", 1, 22050, ResamplingQuality::Fast),
        ("1s_44100_to_22050_high", 1, 22050, ResamplingQuality::High),
        ("1s_44100_to_48000_fast", 1, 48000, ResamplingQuality::Fast),
        ("1s_44100_to_48000_high", 1, 48000, ResamplingQuality::High),
        (
            "10s_44100_to_16000_fast",
            10,
            16000,
            ResamplingQuality::Fast,
        ),
        (
            "10s_44100_to_16000_high",
            10,
            16000,
            ResamplingQuality::High,
        ),
        (
            "10s_44100_to_48000_high",
            10,
            48000,
            ResamplingQuality::High,
        ),
    ];

    let mut group = c.benchmark_group("resampling");

    for &(label, dur_secs, dst_hz, quality) in cases {
        let audio = sine_wave::<f32>(
            440.0,
            Duration::from_secs(dur_secs),
            sample_rate!(44100),
            0.8,
        );
        let dst_sr = std::num::NonZeroU32::new(dst_hz).unwrap();

        // audio_samples side
        group.bench_with_input(
            BenchmarkId::new("audio_samples", label),
            &(dst_sr, quality),
            |b, &(dst, q)| {
                b.iter(|| {
                    let result = resample(black_box(&audio), black_box(dst), black_box(q))
                        .expect("resample should not fail in benchmark");
                    black_box(result);
                });
            },
        );

        // FFmpeg side
        if have_ffmpeg {
            let wav_bytes = audio_to_wav_bytes(&audio);
            let dst_str = dst_hz.to_string();

            group.bench_with_input(BenchmarkId::new("ffmpeg", label), label, |b, _| {
                b.iter(|| {
                    let status = run_ffmpeg_stdin(
                        black_box(&wav_bytes),
                        &["-ar", &dst_str, "-f", "wav", "pipe:1"],
                    );
                    black_box(status);
                });
            });
        } else {
            eprintln!("resampling/{label}: skipping FFmpeg (not found)");
        }
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// 2. IIR Filtering
// ---------------------------------------------------------------------------

fn bench_iir_filtering(c: &mut Criterion) {
    let have_ffmpeg = ffmpeg_available();

    // (label, duration_secs, filter_type, cutoff_hz, high_hz_for_bandpass, order)
    let cases: &[(&str, u64, &str, f64, f64, usize)] = &[
        ("1s_lowpass_1000hz_order4", 1, "lowpass", 1000.0, 0.0, 4),
        ("1s_highpass_1000hz_order4", 1, "highpass", 1000.0, 0.0, 4),
        (
            "1s_bandpass_500_2000hz_order4",
            1,
            "bandpass",
            500.0,
            2000.0,
            4,
        ),
        ("10s_lowpass_1000hz_order4", 10, "lowpass", 1000.0, 0.0, 4),
        ("10s_highpass_1000hz_order4", 10, "highpass", 1000.0, 0.0, 4),
    ];

    let mut group = c.benchmark_group("iir_filtering");

    for &(label, dur_secs, filter_type, cutoff_hz, high_hz, order) in cases {
        let base_audio = sine_wave::<f32>(
            440.0,
            Duration::from_secs(dur_secs),
            sample_rate!(44100),
            0.8,
        );
        let order_nz = NonZeroUsize::new(order).unwrap();

        // audio_samples side — butterworth_* mutates in place; clone per batch
        group.bench_with_input(BenchmarkId::new("audio_samples", label), label, |b, _| {
            b.iter_batched(
                || base_audio.clone(),
                |mut audio| {
                    match filter_type {
                        "lowpass" => audio.butterworth_lowpass(order_nz, cutoff_hz),
                        "highpass" => audio.butterworth_highpass(order_nz, cutoff_hz),
                        "bandpass" => audio.butterworth_bandpass(order_nz, cutoff_hz, high_hz),
                        _ => unreachable!(),
                    }
                    .expect("IIR filter should not fail in benchmark");
                    black_box(audio);
                },
                BatchSize::SmallInput,
            );
        });

        // FFmpeg side
        if have_ffmpeg {
            let wav_bytes = audio_to_wav_bytes(&base_audio);

            let ffmpeg_filter = match filter_type {
                "lowpass" => format!("lowpass=f={cutoff_hz:.0}"),
                "highpass" => format!("highpass=f={cutoff_hz:.0}"),
                "bandpass" => format!(
                    "bandpass=f={:.0}:width_type=h:width={:.0}",
                    (cutoff_hz + high_hz) / 2.0,
                    high_hz - cutoff_hz,
                ),
                _ => unreachable!(),
            };

            group.bench_with_input(BenchmarkId::new("ffmpeg", label), label, |b, _| {
                b.iter(|| {
                    let status = run_ffmpeg_stdin(
                        black_box(&wav_bytes),
                        &["-af", &ffmpeg_filter, "-f", "wav", "pipe:1"],
                    );
                    black_box(status);
                });
            });
        } else {
            eprintln!("iir_filtering/{label}: skipping FFmpeg (not found)");
        }
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// 3. Volume / Loudness Analysis
// ---------------------------------------------------------------------------

fn bench_volume_analysis(c: &mut Criterion) {
    let have_ffmpeg = ffmpeg_available();

    let cases: &[(&str, u64)] = &[("1s_mono", 1), ("10s_mono", 10)];

    let mut group = c.benchmark_group("volume_analysis");

    for &(label, dur_secs) in cases {
        let audio = sine_wave::<f32>(
            440.0,
            Duration::from_secs(dur_secs),
            sample_rate!(44100),
            0.5,
        );

        // audio_samples: rms() and peak() are infallible — no clone needed
        group.bench_with_input(BenchmarkId::new("audio_samples", label), label, |b, _| {
            b.iter(|| {
                let rms = audio.rms();
                let peak = audio.peak();
                black_box((rms, peak));
            });
        });

        if have_ffmpeg {
            let wav_bytes = audio_to_wav_bytes(&audio);

            // volumedetect is a sink filter — must use `-f null /dev/null`,
            // NOT `-f wav pipe:1` (which would error because there is no audio output).
            group.bench_with_input(BenchmarkId::new("ffmpeg", label), label, |b, _| {
                b.iter(|| {
                    let status = run_ffmpeg_stdin(
                        black_box(&wav_bytes),
                        &["-af", "volumedetect", "-f", "null", "/dev/null"],
                    );
                    black_box(status);
                });
            });
        } else {
            eprintln!("volume_analysis/{label}: skipping FFmpeg (not found)");
        }
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// 4. Channel Mixing (Stereo → Mono)
// ---------------------------------------------------------------------------

fn bench_channel_mixing(c: &mut Criterion) {
    let have_ffmpeg = ffmpeg_available();

    let cases: &[(&str, u64)] = &[("1s_stereo_to_mono", 1), ("10s_stereo_to_mono", 10)];

    let mut group = c.benchmark_group("channel_mixing");

    for &(label, dur_secs) in cases {
        let stereo = stereo_sine_wave::<f32>(
            440.0,
            Duration::from_secs(dur_secs),
            sample_rate!(44100),
            0.8,
        );

        // audio_samples: to_mono returns a new owned AudioSamples — no mutation
        group.bench_with_input(BenchmarkId::new("audio_samples", label), label, |b, _| {
            b.iter(|| {
                let mono = stereo
                    .to_mono(black_box(MonoConversionMethod::Average))
                    .expect("to_mono should not fail in benchmark");
                black_box(mono);
            });
        });

        if have_ffmpeg {
            // audio_to_wav_bytes handles interleaving internally via StreamedWavWriter
            let wav_bytes = audio_to_wav_bytes(&stereo);

            group.bench_with_input(BenchmarkId::new("ffmpeg", label), label, |b, _| {
                b.iter(|| {
                    let status = run_ffmpeg_stdin(
                        black_box(&wav_bytes),
                        &["-ac", "1", "-f", "wav", "pipe:1"],
                    );
                    black_box(status);
                });
            });
        } else {
            eprintln!("channel_mixing/{label}: skipping FFmpeg (not found)");
        }
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// 5. Sample Format Conversion
// ---------------------------------------------------------------------------

fn bench_format_conversion(c: &mut Criterion) {
    let have_ffmpeg = ffmpeg_available();

    let cases: &[(&str, u64)] = &[("1s_f32_to_i16", 1), ("10s_f32_to_i16", 10)];

    let mut group = c.benchmark_group("format_conversion");

    for &(label, dur_secs) in cases {
        let audio_f32 = sine_wave::<f32>(
            440.0,
            Duration::from_secs(dur_secs),
            sample_rate!(44100),
            0.8,
        );

        // audio_samples: to_format borrows self and returns a new owned AudioSamples<i16>
        group.bench_with_input(BenchmarkId::new("audio_samples", label), label, |b, _| {
            b.iter(|| {
                let converted = audio_f32.to_format::<i16>();
                black_box(converted);
            });
        });

        if have_ffmpeg {
            let wav_bytes = audio_to_wav_bytes(&audio_f32);

            group.bench_with_input(BenchmarkId::new("ffmpeg", label), label, |b, _| {
                b.iter(|| {
                    let status = run_ffmpeg_stdin(
                        black_box(&wav_bytes),
                        &["-sample_fmt", "s16", "-f", "wav", "pipe:1"],
                    );
                    black_box(status);
                });
            });
        } else {
            eprintln!("format_conversion/{label}: skipping FFmpeg (not found)");
        }
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

fn run_all(c: &mut Criterion) {
    print_ffmpeg_version();
    bench_resampling(c);
    bench_iir_filtering(c);
    bench_volume_analysis(c);
    bench_channel_mixing(c);
    bench_format_conversion(c);
}

criterion_group!(benches, run_all);
criterion_main!(benches);
