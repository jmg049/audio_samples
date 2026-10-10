//! Learning guide: a short course in speech and audio processing.
//!
//! Each chapter is a module that holds only documentation. Its code blocks are
//! doctests, so every number the text claims is computed and checked by
//! `cargo test --doc --features educational,full_no_plotting`. Every chapter has a
//! matching runnable lab, `examples/tutorial_NN_<topic>.rs`.
//!
//! Built only with the `educational` and `full_no_plotting` features (both are
//! enabled on docs.rs).
//!
//! The chapters are the Markdown files in `documentation/guide/`, written with
//! plain LaTeX maths so that they also render in Markdown viewers and in the
//! book (`book/`). `build.rs` escapes the maths for rustdoc's Markdown parser
//! and turns the links between chapter files into intra-doc links; the pages
//! below are those generated copies.
//!
//! The maths (`$…$` inline, `$$…$$` display) is rendered by KaTeX, loaded from
//! `documentation/katex-header.html` (docs.rs passes it automatically). To get
//! the same rendering locally, build the docs with:
//!
//! ```text
//! RUSTDOCFLAGS="--html-in-header documentation/katex-header.html" cargo doc --features educational,full_no_plotting --no-deps
//! ```
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
