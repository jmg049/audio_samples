# Learning guide

A short, runnable course in speech and audio processing built on `audio_samples`.

| # | Chapter | Rust lab |
|---|---|---|
| 01 | [Sampling and aliasing](01_sampling_aliasing.md) | `cargo run --release --example tutorial_01_sampling_aliasing --features educational,full_no_plotting` |
| 02 | [Quantisation and dither](02_quantisation_dither.md) | `cargo run --release --example tutorial_02_quantisation_dither --features educational,full_no_plotting` |
| 03 | [The DFT, windows and leakage](03_dft_windows.md) | `cargo run --release --example tutorial_03_dft_windows --features educational,full_no_plotting` |
| 04 | [STFT and spectrograms](04_stft_spectrograms.md) | `cargo run --release --example tutorial_04_stft_spectrograms --features educational,full_no_plotting` |
| 05 | [Convolution and FIR filters](05_convolution_fir.md) | `cargo run --release --example tutorial_05_convolution_fir --features educational,full_no_plotting` |
| 06 | [IIR filters: design, poles, zeros and stability](06_iir_filters.md) | `cargo run --release --example tutorial_06_iir_filters --features educational,full_no_plotting` |
| 07 | [Resampling](07_resampling.md) | `cargo run --release --example tutorial_07_resampling --features educational,full_no_plotting` |
| 08 | [LPC, the source-filter model and formants](08_lpc_formants.md) | `cargo run --release --example tutorial_08_lpc_formants --features educational,full_no_plotting` |
| 09 | [Pitch: autocorrelation and YIN](09_pitch.md) | `cargo run --release --example tutorial_09_pitch --features educational,full_no_plotting` |
| 10 | [Cepstrum, mel and MFCC](10_cepstrum_mel_mfcc.md) | `cargo run --release --example tutorial_10_cepstrum_mel_mfcc --features educational,full_no_plotting` |
| 11 | [Short-time energy, zero crossings and VAD](11_energy_vad.md) | `cargo run --release --example tutorial_11_energy_vad --features educational,full_no_plotting` |
| 12 | [Dynamics: compressors and limiters](12_dynamics.md) | `cargo run --release --example tutorial_12_dynamics --features educational,full_no_plotting` |
| 13 | [Onsets and tempo](13_onsets_tempo.md) | `cargo run --release --example tutorial_13_onsets_tempo --features educational,full_no_plotting` |
| 14 | [Psychoacoustics and codecs](14_psychoacoustics_codecs.md) | `cargo run --release --example tutorial_14_psychoacoustics_codecs --features educational,full_no_plotting` |

See also [For instructors](FOR_INSTRUCTORS.md).

## Reading the guide

The chapters are the Markdown files in `documentation/guide/`, with standard
LaTeX maths (`$...$` inline, `$$...$$` display). They are published three ways
from the same files:

* in the API documentation, as the `audio_samples::guide` module
  (`cargo doc --open --features educational,full_no_plotting`), where every code
  block is a doctest;
* as an [illustrated book](https://jmg049.github.io/audio_samples/) with figures computed by the library
  (`book/README.md` in the repository says how to build it);
* as plain Markdown, readable in any Markdown viewer with maths support.

In the raw Markdown, code lines that start with `# ` are setup lines that
rustdoc hides.

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
audio_samples = { version = "3", features = ["educational", "full_no_plotting"] }
spectrograms = "2.1.3"    # StftParams, WindowType, MelParams, LogParams, ...
ndarray = "0.17"          # Array1 and the array! macro
num-complex = "0.4"       # complex numbers in the "How it works" sections
non-empty-slice = "0.6"   # NonEmptySlice / NonEmptyVec arguments
```

`audio_samples` does not re-export these crates, so they must be listed
explicitly, with compatible versions: the `StftParams` you pass to `stft`, for
example, must come from the same `spectrograms` release that `audio_samples`
uses. If the `audio_samples` release on crates.io does not include the guide
and the `educational` feature, use a path dependency on a checkout of this
repository instead
(`audio_samples = { path = "../audio_samples", features = [...] }`).

The guide's signals are all generated in code. To analyse your own
recordings, read WAV or FLAC files with the companion crate
[`audio_samples_io`](https://github.com/jmg049/audio_samples_io), whose
`read` function returns an `AudioSamples` value.
