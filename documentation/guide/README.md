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
