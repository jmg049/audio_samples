# Learning guide

A short, runnable course in speech and audio processing built on `audio_samples`.

| # | Chapter | Rust lab |
|---|---|---|
| 01 | [Sampling and aliasing](crate::guide::c01_sampling_aliasing) | `cargo run --release --example tutorial_01_sampling_aliasing --features educational,full_no_plotting` |
| 02 | [Quantisation and dither](crate::guide::c02_quantisation_dither) | `cargo run --release --example tutorial_02_quantisation_dither --features educational,full_no_plotting` |
| 03 | [The DFT, windows and leakage](crate::guide::c03_dft_windows) | `cargo run --release --example tutorial_03_dft_windows --features educational,full_no_plotting` |
| 04 | [STFT and spectrograms](crate::guide::c04_stft_spectrograms) | `cargo run --release --example tutorial_04_stft_spectrograms --features educational,full_no_plotting` |
| 05 | [Convolution and FIR filters](crate::guide::c05_convolution_fir) | `cargo run --release --example tutorial_05_convolution_fir --features educational,full_no_plotting` |
| 06 | [IIR filters: design, poles, zeros and stability](crate::guide::c06_iir_filters) | `cargo run --release --example tutorial_06_iir_filters --features educational,full_no_plotting` |
| 07 | [Resampling](crate::guide::c07_resampling) | `cargo run --release --example tutorial_07_resampling --features educational,full_no_plotting` |
| 08 | [LPC, the source-filter model and formants](crate::guide::c08_lpc_formants) | `cargo run --release --example tutorial_08_lpc_formants --features educational,full_no_plotting` |
| 09 | [Pitch: autocorrelation and YIN](crate::guide::c09_pitch) | `cargo run --release --example tutorial_09_pitch --features educational,full_no_plotting` |
| 10 | [Cepstrum, mel and MFCC](crate::guide::c10_cepstrum_mel_mfcc) | `cargo run --release --example tutorial_10_cepstrum_mel_mfcc --features educational,full_no_plotting` |
| 11 | [Short-time energy, zero crossings and VAD](crate::guide::c11_energy_vad) | `cargo run --release --example tutorial_11_energy_vad --features educational,full_no_plotting` |
| 12 | [Dynamics: compressors and limiters](crate::guide::c12_dynamics) | `cargo run --release --example tutorial_12_dynamics --features educational,full_no_plotting` |
| 13 | [Onsets and tempo](crate::guide::c13_onsets_tempo) | `cargo run --release --example tutorial_13_onsets_tempo --features educational,full_no_plotting` |
| 14 | [Psychoacoustics and codecs](crate::guide::c14_psychoacoustics_codecs) | `cargo run --release --example tutorial_14_psychoacoustics_codecs --features educational,full_no_plotting` |

See also [For instructors](crate::guide::for_instructors).

## Reading the guide

The chapter links in the table above are rustdoc links: they work in the
rendered API documentation (`cargo doc --open --features
educational,full_no_plotting`, then the `audio_samples::guide` module). If you
are browsing the source repository instead, the chapters are the Markdown files
in `documentation/guide/`:
[01](01_sampling_aliasing.md), [02](02_quantisation_dither.md),
[03](03_dft_windows.md), [04](04_stft_spectrograms.md),
[05](05_convolution_fir.md), [06](06_iir_filters.md),
[07](07_resampling.md), [08](08_lpc_formants.md), [09](09_pitch.md),
[10](10_cepstrum_mel_mfcc.md), [11](11_energy_vad.md),
[12](12_dynamics.md), [13](13_onsets_tempo.md),
[14](14_psychoacoustics_codecs.md) and
[For instructors](FOR_INSTRUCTORS.md). These relative links only work in the
repository; in rustdoc, use the table. In the raw Markdown, code lines that
start with `# ` are boilerplate that rustdoc hides, and `\[`, `\]` inside
formulas are escaped brackets.

The labs are the files `examples/tutorial_NN_<topic>.rs`; run them from a
checkout of this repository with the commands in the table. A matching Python
notebook for each chapter is in the Python bindings' documentation
(`docs/source/tutorials/`).

## Setting up

Every code block in the guide is a doctest of this crate. To compile the
snippets in a crate of your own, add these dependencies to its `Cargo.toml`
(the versions are the ones this crate itself builds against):

```toml
[dependencies]
audio_samples = { version = "2", features = ["educational", "full_no_plotting"] }
spectrograms = "2.1.3"    # StftParams, WindowType, MelParams, LogParams, ...
ndarray = "0.17"          # Array1 and the array! macro
num-complex = "0.4"       # complex numbers in the "Under the hood" sections
non-empty-slice = "0.6"   # NonEmptySlice / NonEmptyVec arguments
```

`audio_samples` does not re-export these crates, so they must be listed
explicitly, with compatible versions: the `StftParams` you pass to `stft`, for
example, must come from the same `spectrograms` release that `audio_samples`
uses. The guide and the `educational` feature are new; if the `audio_samples`
release on crates.io does not have them yet, use a path dependency on a
checkout of this repository instead
(`audio_samples = { path = "../audio_samples", features = [...] }`).

The guide's signals are all generated in code. To analyse your own
recordings, read WAV or FLAC files with the companion crate
[`audio_samples_io`](https://github.com/jmg049/audio_samples_io), whose
`read` function returns an `AudioSamples` value.
