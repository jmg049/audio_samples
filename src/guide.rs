//! Learning guide: a short course in speech and audio processing.
//!
//! Each chapter is a page of this module. Its code blocks are doctests, so every
//! number the text gives is computed and checked, and each chapter has a runnable
//! lab, `examples/tutorial_NN_<topic>.rs`. The same chapters, with figures, are the
//! [illustrated book](https://jmg049.github.io/audio_samples/).
//!
//! The maths renders with KaTeX on docs.rs. For a local build, pass the header:
//! `RUSTDOCFLAGS="--html-in-header documentation/katex-header.html" cargo doc --features educational,full_no_plotting --no-deps`.
#![doc = include_str!(concat!(env!("OUT_DIR"), "/guide/README.md"))]

#[doc = include_str!(concat!(env!("OUT_DIR"), "/guide/01_sampling_aliasing.md"))]
pub mod c01_sampling_aliasing {}

#[doc = include_str!(concat!(env!("OUT_DIR"), "/guide/02_quantisation_dither.md"))]
pub mod c02_quantisation_dither {}

#[doc = include_str!(concat!(env!("OUT_DIR"), "/guide/03_dft_windows.md"))]
pub mod c03_dft_windows {}

#[doc = include_str!(concat!(env!("OUT_DIR"), "/guide/04_stft_spectrograms.md"))]
pub mod c04_stft_spectrograms {}

#[doc = include_str!(concat!(env!("OUT_DIR"), "/guide/05_convolution_fir.md"))]
pub mod c05_convolution_fir {}

#[doc = include_str!(concat!(env!("OUT_DIR"), "/guide/06_iir_filters.md"))]
pub mod c06_iir_filters {}

#[doc = include_str!(concat!(env!("OUT_DIR"), "/guide/07_resampling.md"))]
pub mod c07_resampling {}

#[doc = include_str!(concat!(env!("OUT_DIR"), "/guide/08_lpc_formants.md"))]
pub mod c08_lpc_formants {}

#[doc = include_str!(concat!(env!("OUT_DIR"), "/guide/09_pitch.md"))]
pub mod c09_pitch {}

#[doc = include_str!(concat!(env!("OUT_DIR"), "/guide/10_cepstrum_mel_mfcc.md"))]
pub mod c10_cepstrum_mel_mfcc {}

#[doc = include_str!(concat!(env!("OUT_DIR"), "/guide/11_energy_vad.md"))]
pub mod c11_energy_vad {}

#[doc = include_str!(concat!(env!("OUT_DIR"), "/guide/12_dynamics.md"))]
pub mod c12_dynamics {}

#[doc = include_str!(concat!(env!("OUT_DIR"), "/guide/13_onsets_tempo.md"))]
pub mod c13_onsets_tempo {}

#[doc = include_str!(concat!(env!("OUT_DIR"), "/guide/14_psychoacoustics_codecs.md"))]
pub mod c14_psychoacoustics_codecs {}

#[doc = include_str!(concat!(env!("OUT_DIR"), "/guide/FOR_INSTRUCTORS.md"))]
pub mod for_instructors {}
