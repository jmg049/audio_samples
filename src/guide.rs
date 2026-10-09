//! Learning guide: a short course in speech and audio processing.
//!
//! Each chapter is a module that holds only documentation. Its code blocks are
//! doctests, so every number the text claims is computed and checked by
//! `cargo test --doc --features educational,full_no_plotting`. Every chapter has a
//! matching runnable lab, `examples/tutorial_NN_<topic>.rs`.
//!
//! Built only with the `educational` and `full_no_plotting` features (both are
//! enabled on docs.rs).
#![doc = include_str!("../documentation/guide/README.md")]

#[doc = include_str!("../documentation/guide/01_sampling_aliasing.md")]
pub mod c01_sampling_aliasing {}

#[doc = include_str!("../documentation/guide/02_quantisation_dither.md")]
pub mod c02_quantisation_dither {}

#[doc = include_str!("../documentation/guide/03_dft_windows.md")]
pub mod c03_dft_windows {}

#[doc = include_str!("../documentation/guide/04_stft_spectrograms.md")]
pub mod c04_stft_spectrograms {}

#[doc = include_str!("../documentation/guide/05_convolution_fir.md")]
pub mod c05_convolution_fir {}

#[doc = include_str!("../documentation/guide/06_iir_filters.md")]
pub mod c06_iir_filters {}

#[doc = include_str!("../documentation/guide/07_resampling.md")]
pub mod c07_resampling {}

#[doc = include_str!("../documentation/guide/08_lpc_formants.md")]
pub mod c08_lpc_formants {}

#[doc = include_str!("../documentation/guide/09_pitch.md")]
pub mod c09_pitch {}

#[doc = include_str!("../documentation/guide/10_cepstrum_mel_mfcc.md")]
pub mod c10_cepstrum_mel_mfcc {}

#[doc = include_str!("../documentation/guide/11_energy_vad.md")]
pub mod c11_energy_vad {}

#[doc = include_str!("../documentation/guide/12_dynamics.md")]
pub mod c12_dynamics {}

#[doc = include_str!("../documentation/guide/13_onsets_tempo.md")]
pub mod c13_onsets_tempo {}

#[doc = include_str!("../documentation/guide/14_psychoacoustics_codecs.md")]
pub mod c14_psychoacoustics_codecs {}

#[doc = include_str!("../documentation/guide/FOR_INSTRUCTORS.md")]
pub mod for_instructors {}
