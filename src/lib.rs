//! # AudioSamples
//!
//! A typed audio processing library for Rust that represents audio as a typed
//! object that keeps its sample rate, channel layout and other invariants, rather
//! than as an unstructured numeric buffer.
//!
//! `audio_samples` provides a single central type, [`AudioSamples<T>`], that pairs raw
//! PCM data (backed by [`ndarray`](https://docs.rs/ndarray)) with essential metadata:
//! sample rate, channel count, and memory layout. Every audio processing operation in the
//! library is defined as a trait method on this type, ensuring that metadata travels with
//! the data throughout a processing pipeline.
//!
//! Low-level audio APIs in Rust typically expose bare slices or `Vec<f32>` buffers,
//! leaving metadata management to the caller. This encourages subtle bugs such as
//! mismatched sample rates after resampling, or interleaved/non-interleaved confusion
//! when passing buffers between components. `audio_samples` eliminates these error
//! classes by encoding invariants directly into the type.
//!
//! Start by creating an [`AudioSamples<T>`] from an ndarray or from one of the
//! built-in signal generators (see [`utils::generation`]), then chain trait methods
//! for any further processing. Feature flags keep the dependency footprint small -
//! only enable what your application needs.
//!
//! ## Operation Conventions
//!
//! Every transforming operation comes in two variants. The canonical, unsuffixed
//! name borrows (`op(&self, ..) -> AudioSampleResult<Self>`) and returns a new,
//! owned, modified copy, leaving the receiver untouched; the `op_in_place`
//! counterpart (`op_in_place(&mut self, ..) -> AudioSampleResult<()>`) mutates the
//! receiver in place. Because the borrowing variants return owned values, they
//! compose into pipelines without consuming a binding you still need. A handful of
//! operations that cannot fail (such as `scale`) return `Self` directly rather than
//! a `Result`.
//!
//! [`AudioSamples<T>`] keeps its fields private; reach the underlying buffer and
//! metadata through accessors - [`data()`](AudioSamples::data),
//! [`data_mut()`](AudioSamples::data_mut),
//! [`into_data()`](AudioSamples::into_data),
//! [`into_data_borrowed()`](AudioSamples::into_data_borrowed), and
//! [`sample_rate()`](AudioSamples::sample_rate). Channel **indices** are plain
//! `usize`; channel **counts** use the dedicated `ChannelCount` type. Multi-channel
//! analysis operations (spectral centroid, rolloff, and the rest of the spectral
//! suite) take a [`ChannelReduction`] policy that selects how channels are folded
//! down to a single value (`Error`, `First`, `Average`, or `Channel(i)`). Several
//! analysis operations return structured results rather than bare tuples - for
#![cfg_attr(
    feature = "transforms",
    doc = r#" example `power_spectral_density` yields a [`Psd`],"#
)]
#![cfg_attr(
    not(feature = "transforms"),
    doc = r#" example `power_spectral_density` yields a `Psd` (requires the `transforms` feature),"#
)]
#![cfg_attr(
    feature = "pitch-analysis",
    doc = r#" `track_pitch` a [`PitchContour`], and `estimate_key` a [`Key`]."#
)]
#![cfg_attr(
    not(feature = "pitch-analysis"),
    doc = r#" `track_pitch` a `PitchContour`, and `estimate_key` a `Key` (both require the `pitch-analysis` feature)."#
)]
//!
//! ## Installation
//!
//! ```bash
//! cargo add audio_samples
//! ```
//!
//! ## Features
//!
//! The library uses a modular feature flag system. Only enable what you need.
//!
//! | Feature | Description |
//! |---|---|
//! | `statistics` | Peak, RMS, variance, zero-crossings, and other statistical measures |
//! | `processing` | Normalise, scale, clip, DC offset removal, and other sample-level operations (implies `statistics`) |
//! | `editing` | Time-domain editing: trim, pad, reverse, concatenate, fade (implies `statistics`, `random-generation`) |
//! | `channels` | Channel operations: mono↔stereo conversion, channel extraction, interleave/deinterleave |
//! | `transforms` | Spectrogram and frequency-domain transforms via the `spectrograms` crate; STFT/ISTFT, mel spectrogram and MFCC (`AudioSpectrum`); with `resampling`, CQT/VQT (`AudioConstantQ`) |
//! | `iir-filtering` | IIR filter design and application (Butterworth, Chebyshev I/II, Elliptic/Cauer, Bessel; low/high/band-pass and band-stop), zero-phase `filtfilt`, and a design-once streaming `SosFilter` |
//! | `parametric-eq` | Multi-band parametric equaliser (implies `iir-filtering`) |
//! | `dynamic-range` | Compression, limiting, gating, and expansion with side-chain support, configured via `CompressorConfig`/`LimiterConfig`/`GateConfig`/`ExpanderConfig` |
//! | `envelopes` | Amplitude, RMS, and analytical envelope followers (implies `dynamic-range`, `editing`, `random-generation`) |
//! | `peak-picking` | Onset strength curve peak picking |
//! | `onset-detection` | Onset detection (implies `transforms`, `peak-picking`, `processing`) |
//! | `beat-tracking` | Beat tracking and tempo estimation (`estimate_tempo`); onset, tempo and beat analysis (`AudioRhythm`) |
//! | `decomposition` | Audio source decomposition (implies `onset-detection`) |
//! | `pitch-analysis` | YIN and autocorrelation pitch detection; frame-wise YIN and pYIN (`AudioPitchTracking`) (implies `transforms`, `sequence`) |
//! | `feature-extraction` | Frame-wise spectral features (`AudioFeatures`) and chroma / tuning (`AudioTonal`) (implies `transforms`) |
//! | `effects` | Time stretch, pitch shift, emphasis, dB trim/split, HPSS (`AudioEffects`) (implies `transforms`, `resampling`, `editing`, `sequence`) |
//! | `sequence` | DTW, Viterbi, recurrence matrices and `sync` (`operations::sequence`) |
//! | `vad` | Voice activity detection |
//! | `psychoacoustic` | Psychoacoustic analysis: Bark/Mel band layouts, ATH, masking thresholds, SMR (implies `transforms`) |
//! | `resampling` | Delay-free windowed-sinc sample-rate conversion (whole-buffer and streaming) |
//! | `plotting` | Signal plotting via `plotly` |
//! | `fixed-size-audio` | Stack-allocated fixed-size audio buffers |
//! | `random-generation` | Noise generators and stochastic signal sources (implies `rand`) |
//! | `full` | Enables all of the above |
//!
//! See `Cargo.toml` for the complete dependency graph.
//!
//! ### Modules by feature
//!
//! The default build (`bare-bones`) contains [`AudioSamples`], the conversions, the
//! iterators, [`utils`], [`operations::traits`] and [`operations::types`]. Each feature
//! below adds the modules listed next to it. A module name is a link when this
//! documentation was built with its feature, and plain text otherwise. In a checkout
//! of the repository, `cargo docs` builds and opens the documentation with every
//! feature that builds offline (an alias defined in `.cargo/config.toml`).
//!
//! | Feature | Modules |
//! |---|---|
#![cfg_attr(
    feature = "statistics",
    doc = r#" | `statistics` | [`operations::statistics`] |"#
)]
#![cfg_attr(
    not(feature = "statistics"),
    doc = r#" | `statistics` | `operations::statistics` |"#
)]
#![cfg_attr(
    feature = "processing",
    doc = r#" | `processing` | [`operations::processing`] |"#
)]
#![cfg_attr(
    not(feature = "processing"),
    doc = r#" | `processing` | `operations::processing` |"#
)]
#![cfg_attr(
    feature = "transforms",
    doc = r#" | `transforms` | [`operations::transforms`], [`operations::spectrum`], [`operations::filterbanks`] |"#
)]
#![cfg_attr(
    not(feature = "transforms"),
    doc = r#" | `transforms` | `operations::transforms`, `operations::spectrum`, `operations::filterbanks` |"#
)]
#![cfg_attr(
    all(feature = "transforms", feature = "resampling"),
    doc = r#" | `transforms` + `resampling` | [`operations::constant_q`] |"#
)]
#![cfg_attr(
    not(all(feature = "transforms", feature = "resampling")),
    doc = r#" | `transforms` + `resampling` | `operations::constant_q` |"#
)]
#![cfg_attr(
    feature = "editing",
    doc = r#" | `editing` | [`operations::editing`] |"#
)]
#![cfg_attr(
    not(feature = "editing"),
    doc = r#" | `editing` | `operations::editing` |"#
)]
#![cfg_attr(
    feature = "channels",
    doc = r#" | `channels` | [`operations::channels`] |"#
)]
#![cfg_attr(
    not(feature = "channels"),
    doc = r#" | `channels` | `operations::channels` |"#
)]
#![cfg_attr(
    feature = "dynamic-range",
    doc = r#" | `dynamic-range` | [`operations::dynamic_range`] |"#
)]
#![cfg_attr(
    not(feature = "dynamic-range"),
    doc = r#" | `dynamic-range` | `operations::dynamic_range` |"#
)]
#![cfg_attr(
    feature = "envelopes",
    doc = r#" | `envelopes` | [`operations::envelopes`] |"#
)]
#![cfg_attr(
    not(feature = "envelopes"),
    doc = r#" | `envelopes` | `operations::envelopes` |"#
)]
#![cfg_attr(
    feature = "iir-filtering",
    doc = r#" | `iir-filtering` | [`operations::iir_filtering`] |"#
)]
#![cfg_attr(
    not(feature = "iir-filtering"),
    doc = r#" | `iir-filtering` | `operations::iir_filtering` |"#
)]
#![cfg_attr(
    feature = "parametric-eq",
    doc = r#" | `parametric-eq` | [`operations::parametric_eq`] |"#
)]
#![cfg_attr(
    not(feature = "parametric-eq"),
    doc = r#" | `parametric-eq` | `operations::parametric_eq` |"#
)]
#![cfg_attr(
    feature = "peak-picking",
    doc = r#" | `peak-picking` | [`operations::peak_picking`] |"#
)]
#![cfg_attr(
    not(feature = "peak-picking"),
    doc = r#" | `peak-picking` | `operations::peak_picking` |"#
)]
#![cfg_attr(
    feature = "onset-detection",
    doc = r#" | `onset-detection` | [`operations::onset`] |"#
)]
#![cfg_attr(
    not(feature = "onset-detection"),
    doc = r#" | `onset-detection` | `operations::onset` |"#
)]
#![cfg_attr(
    feature = "beat-tracking",
    doc = r#" | `beat-tracking` | [`operations::beat`], [`operations::rhythm`] |"#
)]
#![cfg_attr(
    not(feature = "beat-tracking"),
    doc = r#" | `beat-tracking` | `operations::beat`, `operations::rhythm` |"#
)]
#![cfg_attr(
    feature = "decomposition",
    doc = r#" | `decomposition` | [`operations::hpss`] |"#
)]
#![cfg_attr(
    not(feature = "decomposition"),
    doc = r#" | `decomposition` | `operations::hpss` |"#
)]
#![cfg_attr(
    feature = "pitch-analysis",
    doc = r#" | `pitch-analysis` | [`operations::pitch_analysis`], [`operations::pitch_tracking`] |"#
)]
#![cfg_attr(
    not(feature = "pitch-analysis"),
    doc = r#" | `pitch-analysis` | `operations::pitch_analysis`, `operations::pitch_tracking` |"#
)]
#![cfg_attr(
    feature = "feature-extraction",
    doc = r#" | `feature-extraction` | [`operations::features`], [`operations::tonal`] |"#
)]
#![cfg_attr(
    not(feature = "feature-extraction"),
    doc = r#" | `feature-extraction` | `operations::features`, `operations::tonal` |"#
)]
#![cfg_attr(
    feature = "effects",
    doc = r#" | `effects` | [`operations::effects`] |"#
)]
#![cfg_attr(
    not(feature = "effects"),
    doc = r#" | `effects` | `operations::effects` |"#
)]
#![cfg_attr(
    feature = "sequence",
    doc = r#" | `sequence` | [`operations::sequence`] |"#
)]
#![cfg_attr(
    not(feature = "sequence"),
    doc = r#" | `sequence` | `operations::sequence` |"#
)]
#![cfg_attr(feature = "linalg", doc = r#" | `linalg` | [`operations::linalg`] |"#)]
#![cfg_attr(
    not(feature = "linalg"),
    doc = r#" | `linalg` | `operations::linalg` |"#
)]
#![cfg_attr(
    feature = "feature-inversion",
    doc = r#" | `feature-inversion` | [`operations::inverse`] |"#
)]
#![cfg_attr(
    not(feature = "feature-inversion"),
    doc = r#" | `feature-inversion` | `operations::inverse` |"#
)]
#![cfg_attr(
    feature = "factorization",
    doc = r#" | `factorization` | [`operations::factorization`] |"#
)]
#![cfg_attr(
    not(feature = "factorization"),
    doc = r#" | `factorization` | `operations::factorization` |"#
)]
#![cfg_attr(
    feature = "pitch-filterbank",
    doc = r#" | `pitch-filterbank` | [`operations::pitch_filterbank`] |"#
)]
#![cfg_attr(
    not(feature = "pitch-filterbank"),
    doc = r#" | `pitch-filterbank` | `operations::pitch_filterbank` |"#
)]
#![cfg_attr(feature = "mellin", doc = r#" | `mellin` | [`operations::mellin`] |"#)]
#![cfg_attr(
    not(feature = "mellin"),
    doc = r#" | `mellin` | `operations::mellin` |"#
)]
#![cfg_attr(
    feature = "segmentation",
    doc = r#" | `segmentation` | [`operations::segment`] |"#
)]
#![cfg_attr(
    not(feature = "segmentation"),
    doc = r#" | `segmentation` | `operations::segment` |"#
)]
#![cfg_attr(feature = "vad", doc = r#" | `vad` | [`operations::vad`] |"#)]
#![cfg_attr(not(feature = "vad"), doc = r#" | `vad` | `operations::vad` |"#)]
#![cfg_attr(
    feature = "plotting",
    doc = r#" | `plotting` | [`operations::plotting`] |"#
)]
#![cfg_attr(
    not(feature = "plotting"),
    doc = r#" | `plotting` | `operations::plotting` |"#
)]
#![cfg_attr(feature = "resampling", doc = r#" | `resampling` | [`resampling`] |"#)]
#![cfg_attr(
    not(feature = "resampling"),
    doc = r#" | `resampling` | `resampling` |"#
)]
#![cfg_attr(
    feature = "psychoacoustic",
    doc = r#" | `psychoacoustic` | [`codecs`], [`codecs::perceptual`] |"#
)]
#![cfg_attr(
    not(feature = "psychoacoustic"),
    doc = r#" | `psychoacoustic` | `codecs`, `codecs::perceptual` |"#
)]
#![cfg_attr(
    feature = "opus-codec",
    doc = r#" | `opus-codec` | [`codecs::opus`] |"#
)]
#![cfg_attr(
    not(feature = "opus-codec"),
    doc = r#" | `opus-codec` | `codecs::opus` |"#
)]
#![cfg_attr(
    feature = "fixed-size-audio",
    doc = r#" | `fixed-size-audio` | [`fixed_audio`] |"#
)]
#![cfg_attr(
    not(feature = "fixed-size-audio"),
    doc = r#" | `fixed-size-audio` | `fixed_audio` |"#
)]
#![cfg_attr(
    feature = "educational",
    doc = r#" | `educational` | [`educational`] |"#
)]
#![cfg_attr(
    not(feature = "educational"),
    doc = r#" | `educational` | `educational` |"#
)]
#![cfg_attr(
    all(feature = "educational", feature = "full_no_plotting"),
    doc = r#" | `educational` + `full_no_plotting` | [`guide`] |"#
)]
#![cfg_attr(
    not(all(feature = "educational", feature = "full_no_plotting")),
    doc = r#" | `educational` + `full_no_plotting` | `guide` |"#
)]
#![cfg_attr(
    feature = "full_no_plotting",
    doc = r#" | `full_no_plotting` | [`librosa_migration`], [`moving_from_librosa_rust`] |"#
)]
#![cfg_attr(
    not(feature = "full_no_plotting"),
    doc = r#" | `full_no_plotting` | `librosa_migration`, `moving_from_librosa_rust` |"#
)]
//!
//! ## Migration guides
//!
#![cfg_attr(
    feature = "full_no_plotting",
    doc = r#" The [`librosa_migration`] and [`moving_from_librosa_rust`] modules (built with"#
)]
#![cfg_attr(
    not(feature = "full_no_plotting"),
    doc = r#" The `librosa_migration` and `moving_from_librosa_rust` modules (built with"#
)]
//! `full_no_plotting` or `full`; also `documentation/LIBROSA_MIGRATION.md` and
//! `documentation/MOVING_FROM_LIBROSA_RUST.md`) map Python librosa functions to their
//! counterparts in this crate and list the differences.
//!
//! ## Error Handling
//!
//! All fallible operations return [`AudioSampleResult<T>`], which is an alias for
//! `Result<T, AudioSampleError>`. Errors are structured so that the variant indicates
//! the category of failure and the inner type provides detail.
//!
//! Every error implements [`miette::Diagnostic`] as well as [`std::error::Error`].
//! Each variant carries a stable, namespaced **code**
//! (`audio_samples::<domain>::<variant>`) for programmatic matching and an
//! actionable **help** hint. Failures that arise from parsing user-supplied text
//! (note names, enum `FromStr`) additionally carry a source span pointing a caret
//! at the offending character. Enable the `fancy` feature to render the coloured
//! caret-underline output (see `examples/error_diagnostics.rs`); library consumers
//! who do not opt in get plain [`Display`](std::fmt::Display) output and do not compile the graphical
//! dependencies.
//!
//! ```rust
//! use audio_samples::{AudioSampleError, AudioSampleResult, ParameterError};
//!
//! let audio_result: AudioSampleResult<()> = Err(AudioSampleError::Parameter(
//!     ParameterError::invalid_value("window_size", "must be > 0"),
//! ));
//!
//! match audio_result {
//!     Ok(()) => {}
//!     Err(AudioSampleError::Conversion(err)) => eprintln!("Conversion failed: {err}"),
//!     Err(AudioSampleError::Parameter(err)) => eprintln!("Invalid parameter: {err}"),
//!     Err(other_err) => eprintln!("Other error: {other_err}"),
//! }
//! ```
//!
//! ## Full Examples
//!
//! Examples are in the repository (see `examples/`) and in the crate-level docs on
//! <https://docs.rs/audio_samples>.
//!
//! ## Quick Start
//!
//! ### Creating Audio Data
//!
//! ```rust
//! use audio_samples::{AudioSamples, sample_rate};
//! use ndarray::array;
//!
//! // Create mono audio
//! let data = array![0.1f32, 0.5, -0.3, 0.8, -0.2];
//! let audio = AudioSamples::new_mono(data, sample_rate!(44100)).unwrap();
//!
//! // Create stereo audio (channels × samples)
//! let stereo_data = array![
//!     [0.1f32, 0.5, -0.3],  // Left channel
//!     [0.8f32, -0.2, 0.4]   // Right channel
//! ];
//! let stereo_audio = AudioSamples::new_multi_channel(stereo_data, sample_rate!(44100)).unwrap();
//! ```
//!
//! ### Basic Statistics
//!
//! Requires the `statistics` feature.
//!
//! ```rust,ignore
//! use audio_samples::{AudioStatistics, sine_wave, sample_rate};
//! use std::time::Duration;
//!
//! // Generate a 440 Hz sine wave at 44.1 kHz sample rate, amplitude 1.0
//! let audio = sine_wave::<f32>(440.0, Duration::from_secs_f32(1.0), sample_rate!(44100), 1.0);
//!
//! let peak           = audio.peak();
//! let min            = audio.min_sample();
//! let max            = audio.max_sample();
//! let mean           = audio.mean();
//! let rms            = audio.rms();
//! let variance       = audio.variance();
//! let zero_crossings = audio.zero_crossings();
//! ```
//!
//! ### Processing Operations
//!
//! Requires the `processing` feature.
//!
//! ```rust,ignore
//! use audio_samples::{AudioProcessing, NormalizationConfig, AudioSamples, sample_rate};
//! use ndarray::array;
//!
//! let data = array![0.1f32, 0.5, -0.3, 0.8, -0.2];
//! let audio = AudioSamples::new_mono(data, sample_rate!(44100)).unwrap();
//!
//! // Method chaining: each borrowing operation returns a NEW owned value, so the
//! // results compose directly. `scale` is infallible (returns `Self`, no `?`),
//! // while `normalize`/`remove_dc_offset` are fallible. Use the `*_in_place`
//! // variants instead when you want to mutate `audio` rather than build a copy.
//! let audio = audio
//!     .normalize(NormalizationConfig::peak(1.0))
//!     .unwrap()
//!     .scale(0.5)
//!     .remove_dc_offset()
//!     .unwrap();
//! ```
//!
//! ### Type Conversions
//!
//! ```rust
//! use audio_samples::{AudioSamples, AudioTypeConversion, sample_rate};
//! use ndarray::array;
//!
//! let audio_f32 = AudioSamples::new_mono(array![1.0f32, 2.0, 3.0], sample_rate!(44100)).unwrap();
//! let audio_i16 = audio_f32.clone().to_type::<i16>();
//! let audio_f64 = audio_f32.to_type::<f64>();
//! ```
//!
//! ### Iterating Over Audio Data
//!
//! ```rust
//! use audio_samples::{AudioSampleIterators, AudioSamples, sample_rate};
//! use ndarray::array;
//!
//! let audio = AudioSamples::new_mono(
//!     array![1.0f32, 2.0, 3.0, 4.0],
//!     sample_rate!(44100),
//! ).unwrap();
//!
//! // Iterate by frames (one sample per channel per time step)
//! for frame in audio.frames() {
//!     println!("Frame: {:?}", frame);
//! }
//!
//! // Iterate by channels
//! for channel in audio.channels() {
//!     println!("Channel: {:?}", channel);
//! }
//! ```
//!
//! ### Psychoacoustic Analysis
//!
//! Requires the `psychoacoustic` feature.
//!
//! ```rust,ignore
//! use audio_samples::{
//!     AudioPerceptualAnalysis, BandLayout, PsychoacousticConfig, sine_wave, sample_rate,
//! };
//! use non_empty_slice::NonEmptySlice;
//! use spectrograms::WindowType;
//! use std::num::NonZeroUsize;
//! use std::time::Duration;
//!
//! let signal = sine_wave::<f32>(440.0, Duration::from_millis(200), sample_rate!(44100), 0.8);
//!
//! // 24 Bark critical bands mapped onto 1024 MDCT bins.
//! let layout = BandLayout::bark(
//!     NonZeroUsize::new(24).unwrap(),
//!     44100.0,
//!     NonZeroUsize::new(1024).unwrap(),
//! );
//!
//! let weights = vec![1.0_f32; 24];
//! let config = PsychoacousticConfig::new(
//!     -60.0, 14.5, 5.5, 25.0, 6.0,
//!     NonEmptySlice::from_slice(&weights).unwrap(),
//!     1e-10,
//! );
//!
//! let result = signal
//!     .analyse_psychoacoustic(WindowType::Hanning, &layout, &config)
//!     .unwrap();
//!
//! // Bands with positive SMR are audible above the masking threshold.
//! for metric in result.band_metrics.as_slice().iter() {
//!     if metric.signal_to_mask_ratio > 0.0 {
//!         println!(
//!             "{:.0} Hz - SMR {:.1} dB (importance {:.2})",
//!             metric.band.centre_frequency,
//!             metric.signal_to_mask_ratio,
//!             metric.importance,
//!         );
//!     }
//! }
//! ```
//!
//! ## Core Type System
//!
//! ### Supported Sample Types
//!
//! | Type | Width | Notes |
//! |---|---|---|
//! | `u8`  | 8-bit unsigned  | Mid-scale (128) represents silence |
//! | `i16` | 16-bit signed   | CD-quality audio |
//! | [`I24`] | 24-bit signed | From the `i24` crate |
//! | `i32` | 32-bit signed   | High-dynamic-range integer audio |
//! | `f32` | 32-bit float    | Most DSP operations use this type |
//! | `f64` | 64-bit float    | High-precision processing |
//!
//! ### Type System Traits
//!
//! - **[`AudioSample`]** - Core trait all sample types implement. Provides constants
//!   (`MAX`, `MIN`, `BITS`, `BYTES`, `LABEL`) and low-level byte operations.
//!
//! - **[`ConvertTo<T>`]** / **[`ConvertFrom<T>`]** - Audio-aware conversions with
//!   correct scaling between bit depths (e.g. `i16 → f32` normalises to `[-1.0, 1.0]`).
//!
//! - **[`CastFrom<T>`]** / **[`CastInto<T>`]** - Raw numeric casts without scaling.
//!   Use these when you need the raw integer value as a float for computation,
//!   not as an audio amplitude.
//!
//! - **[`AudioTypeConversion`]** - High-level conversion methods on [`AudioSamples<T>`]
//!   such as `as_f32()`, `as_i16()`, and `as_type::<U>()`.
//!
//! ## Signal Generation
//!
//! The [`utils::generation`] module provides functions for creating test and
//! reference signals: [`sine_wave`], [`cosine_wave`], [`square_wave`],
//! [`triangle_wave`], [`sawtooth_wave`], [`chirp`], [`exponential_chirp`] (also
//! covering logarithmic sweeps), the band-limited oscillators
//! [`square_wave_bandlimited`], [`sawtooth_wave_bandlimited`], and
//! [`triangle_wave_bandlimited`], [`fm_signal`], [`impulse`], [`silence`],
//! [`compound_tone`], and [`am_signal`]. With the `random-generation` feature,
//! `white_noise`, `pink_noise`, and `brown_noise` are also available.
//!
//! ```rust
//! use audio_samples::{sine_wave, sample_rate};
//! use audio_samples::utils::comparison;
//! use std::time::Duration;
//!
//! let a = sine_wave::<f32>(440.0, Duration::from_secs(1), sample_rate!(44100), 1.0);
//! let b = sine_wave::<f32>(440.0, Duration::from_secs(1), sample_rate!(44100), 1.0);
//! let corr: f64 = comparison::correlation(&a, &b).unwrap();
//! assert!(corr > 0.99);
//! ```
//!
//! ## Documentation
//!
//! Full API documentation is available at [docs.rs/audio_samples](https://docs.rs/audio_samples).
//!
//! ## License
//!
//! MIT
//!
//! ## Contributing
//!
//! Contributions are welcome. Please open an issue or pull request on
//! [GitHub](https://github.com/jmg049/audio_samples).

/// Creates a `NonZeroUsize` from a compile-time constant with zero-check.
///
/// This macro creates a `NonZeroUsize` with a compile-time assertion that the value is
/// non-zero, making it safe to use `new_unchecked` internally. If the value is zero, the
/// code will fail to compile.
///
/// # Example
///
/// ```
/// # use audio_samples::nzu;
/// let size = nzu!(1024);
/// assert_eq!(size.get(), 1024);
/// ```
///
/// This will fail to compile:
/// ```compile_fail
/// # use audio_samples::nzu;
/// let invalid = nzu!(0); // Compile error: assertion failed
/// ```
#[macro_export]
macro_rules! nzu {
    ($rate:expr) => {{
        const RATE: usize = $rate;
        const { assert!(RATE > 0, "non zero usize must be greater than 0") };
        // SAFETY: We just asserted RATE > 0 at compile time
        #[allow(unused_unsafe)]
        unsafe {
            ::core::num::NonZeroUsize::new_unchecked(RATE)
        }
    }};
}

/// Codec infrastructure: the [`AudioCodec`] trait, `encode`/`decode` free functions,
/// and the [`PerceptualCodec`] implementation (requires `feature = "psychoacoustic"`).
#[cfg(feature = "psychoacoustic")]
pub mod codecs;
pub mod conversions;
pub mod iterators;
pub mod operations;
#[cfg(feature = "resampling")]
pub mod resampling;
/// Core trait definitions for audio sample types and operations.
pub mod traits;
pub mod utils;

mod audio_float;
mod error;
/// Fixed-size audio buffer types whose geometry is encoded in the type system.
#[cfg(feature = "fixed-size-audio")]
pub mod fixed_audio;
mod repr;

pub mod simd_conversions;

/// Educational / explainable layer: step-by-step operation explanations with
/// before/after waveforms and term-maths formula rendering.
///
/// Enable with `--features educational`. See [`educational::open_explanation_document`]
/// for the primary entry point.
#[cfg(feature = "educational")]
pub mod educational;

/// Documentation only: the librosa to audio_samples migration guide.
///
/// This module contains no items. It renders `documentation/LIBROSA_MIGRATION.md`
/// and runs its examples as doctests. It exists only when building documentation
/// or doctests with `full_no_plotting` (or `full`) enabled.
#[cfg(all(feature = "full_no_plotting", any(doc, doctest)))]
pub mod librosa_migration {
    #![doc = include_str!("../documentation/LIBROSA_MIGRATION.md")]
}

/// Documentation only: porting a librosa pipeline to Rust.
///
/// This module contains no items. It renders
/// `documentation/MOVING_FROM_LIBROSA_RUST.md` and runs its examples as doctests.
#[cfg(all(feature = "full_no_plotting", any(doc, doctest)))]
pub mod moving_from_librosa_rust {
    #![doc = include_str!("../documentation/MOVING_FROM_LIBROSA_RUST.md")]
}

/// Learning guide: a short course in speech and audio processing, with one
/// chapter per syllabus topic. Every chapter's code is a doctest and has a
/// matching runnable lab in `examples/tutorial_NN_*.rs`.
#[cfg(all(feature = "educational", feature = "full_no_plotting"))]
pub mod guide;

pub use crate::audio_float::AudioFloat;
pub use crate::error::{
    AudioSampleError, AudioSampleResult, ChannelRequirement, ConversionError, EnumParseError,
    FeatureError, LayoutError, NoteParseError, ParameterError, ProcessingError,
};
pub use crate::repr::AudioData;
pub use crate::repr::StereoAudioSamples;
pub use crate::repr::{AudioSamples, SampleType};
pub use crate::traits::{
    AudioSample, AudioTypeConversion, CastFrom, CastInto, ConvertFrom, ConvertTo, StandardSample,
};

pub use crate::iterators::{AudioSampleIterators, ChannelIterator, FrameIterator, PaddingMode};

#[cfg(feature = "editing")]
pub use crate::iterators::{WindowIterator, WindowRefIterator, WindowView};

pub use crate::utils::generation::{
    ExampleClip, ToneComponent, am_signal, chirp, compound_tone, cosine_wave, example_clip,
    exponential_chirp, fm_signal, impulse, sawtooth_wave, sawtooth_wave_bandlimited, silence,
    sine_wave, square_wave, square_wave_bandlimited, triangle_wave, triangle_wave_bandlimited,
};

#[cfg(feature = "channels")]
pub use crate::utils::generation::{
    multichannel_compound_tone, stereo_chirp, stereo_silence, stereo_sine_wave,
};

#[cfg(feature = "statistics")]
pub use crate::operations::AudioStatistics;

pub use crate::operations::types::ChannelReduction;

#[cfg(all(feature = "transforms", feature = "statistics"))]
pub use crate::operations::statistics::fft_alignment_lag;

#[cfg(feature = "processing")]
pub use crate::operations::AudioProcessing;

#[cfg(feature = "processing")]
pub use crate::operations::processing::{mu_law_decode, mu_law_encode};

#[cfg(feature = "processing")]
pub use crate::operations::types::NormalizationConfig;

#[cfg(any(feature = "processing", feature = "peak-picking"))]
pub use crate::operations::types::NormalizationMethod;

#[cfg(feature = "channels")]
pub use crate::operations::AudioChannelOps;

#[cfg(feature = "editing")]
pub use crate::operations::AudioEditing;

#[cfg(feature = "transforms")]
pub use crate::operations::AudioTransforms;

#[cfg(feature = "transforms")]
pub use crate::operations::transforms::Psd;

#[cfg(feature = "transforms")]
pub use crate::operations::AudioSpectrum;

#[cfg(feature = "transforms")]
pub use crate::operations::filterbanks::{MelFilterNorm, MelFilterbankConfig, Window};

#[cfg(feature = "transforms")]
pub use crate::operations::spectrum::{
    DbAxes, DbConfig, DbReference, DctNorm, DctType, DeltaConfig, DeltaMode, MelSpectrogramConfig,
    MfccConfig, StftConfig, StftPadMode,
};

#[cfg(feature = "effects")]
pub use crate::operations::AudioEffects;

#[cfg(feature = "effects")]
pub use crate::operations::effects::{
    ChannelAggregate, ClickPositions, ClicksConfig, GriffinLimConfig, GriffinLimInit,
    HarmonicPercussiveConfig, HpssMaskConfig, NnAggregate, PitchShiftConfig, SilenceConfig,
};

#[cfg(all(feature = "transforms", feature = "resampling"))]
pub use crate::operations::AudioConstantQ;

#[cfg(all(feature = "transforms", feature = "resampling"))]
pub use crate::operations::constant_q::{CqtConfig, VqtConfig, WaveletConfig};

#[cfg(all(
    feature = "transforms",
    feature = "resampling",
    feature = "feature-extraction"
))]
pub use crate::operations::constant_q::{
    ChromaCensConfig, ChromaCqtConfig, ChromaVqtConfig, CqtMode, TonnetzConfig,
};

pub use crate::utils::audio_math::MelScale;

#[cfg(feature = "feature-extraction")]
pub use crate::operations::AudioTonal;

#[cfg(feature = "feature-extraction")]
pub use crate::operations::tonal::{
    ChromaFilterbankConfig, ChromaStftConfig, NormalizeConfig, PiptrackConfig, PiptrackReference,
    TuningConfig,
};

#[cfg(feature = "feature-extraction")]
pub use crate::operations::AudioFeatures;

#[cfg(feature = "feature-extraction")]
pub use crate::operations::features::{
    FrequencyWeighting, PcenConfig, PolyFeaturesConfig, ReassignRefPower, ReassignedSpectrogram,
    ReassignedSpectrogramConfig, RmsConfig, SpectralBandwidthConfig, SpectralCentroidConfig,
    SpectralContrastConfig, SpectralFlatnessConfig, SpectralRolloffConfig, ZeroCrossingRateConfig,
    ZeroCrossingReference,
};

#[cfg(feature = "fixed-size-audio")]
pub use crate::fixed_audio::{FixedSizeAudioSamples, FixedSizeMultiChannelAudioSamples};

#[cfg(feature = "onset-detection")]
pub use crate::operations::traits::AudioOnsetDetection;

#[cfg(feature = "beat-tracking")]
pub use crate::operations::AudioRhythm;

#[cfg(feature = "beat-tracking")]
pub use crate::operations::rhythm::{
    Aggregate, BeatTrack, BeatTrackConfig, FourierTempogramConfig, FrameEvents,
    HybridTempogramConfig, OnsetDetectConfig, OnsetStrengthConfig, PeakPickConfig, PeakPickMethod,
    PlpConfig, TempoConfig, TempoLogPrior, TempogramConfig, TempogramRatioConfig,
};

#[cfg(feature = "decomposition")]
pub use crate::operations::traits::AudioDecomposition;

#[cfg(feature = "dynamic-range")]
pub use crate::operations::traits::AudioDynamicRange;

#[cfg(feature = "dynamic-range")]
pub use crate::operations::types::{
    CompressorConfig, ExpanderConfig, GateConfig, LimiterConfig, SideChainConfig,
};

#[cfg(feature = "iir-filtering")]
pub use crate::operations::traits::AudioIirFiltering;

#[cfg(feature = "iir-filtering")]
pub use crate::operations::iir_filtering::{IirFilter, SosFilter};

#[cfg(feature = "parametric-eq")]
pub use crate::operations::traits::AudioParametricEq;

#[cfg(feature = "parametric-eq")]
pub use crate::operations::types::ThreeBandEqConfig;

#[cfg(feature = "pitch-analysis")]
pub use crate::operations::traits::AudioPitchAnalysis;

#[cfg(feature = "pitch-analysis")]
pub use crate::operations::pitch_analysis::{Key, Mode, PitchClass, PitchContour, PitchFrame};

#[cfg(feature = "pitch-analysis")]
pub use crate::operations::pitch_tracking::{
    AudioPitchTracking, PyinConfig, PyinResult, UnvoicedFill, YinConfig,
};

#[cfg(feature = "vad")]
pub use crate::operations::traits::AudioVoiceActivityDetection;

#[cfg(feature = "psychoacoustic")]
pub use crate::codecs::perceptual::{
    AudioCodec, AudioPerceptualAnalysis, Band, BandLayout, BandMetric, BandMetrics, EncodedSegment,
    PerceptualAnalysisResult, PerceptualCodec, PerceptualEncodedAudio, PsychoacousticConfig,
    StereoPerceptualCodec, StereoPerceptualEncodedAudio, analyse_signal_with_window_size,
    apply_temporal_masking, detect_transient_windows, reconstruct_signal,
};

#[cfg(feature = "psychoacoustic")]
pub use crate::codecs::perceptual::bands::scale_band_layout;

#[cfg(feature = "psychoacoustic")]
pub use crate::codecs::perceptual::masking::{
    MaskerType, absolute_threshold_of_hearing, classify_masker_types, compute_smr,
    spreading_attenuation,
};

#[cfg(feature = "psychoacoustic")]
pub use crate::codecs::perceptual::stereo::{mid_side_decode, mid_side_encode};

#[cfg(feature = "psychoacoustic")]
pub use crate::codecs::perceptual::quantization::{
    BandAllocation, BitAllocationResult, allocate_bits, dequantize, dequantize_band,
    max_index_for_word_length, quantize, quantize_band, step_size_from_allowed_noise,
};

#[cfg(feature = "psychoacoustic")]
pub use crate::codecs::perceptual::codec::{decode as codec_decode, encode as codec_encode};

#[cfg(feature = "opus-codec")]
pub use crate::codecs::opus::{
    CeltEncodedFrame, HYBRID_CROSSOVER_HZ, HybridEncodedFrame, LpcCoefficients, OpusBandwidth,
    OpusCodec, OpusConfig, OpusEncodedAudio, OpusEncodedFrame, OpusFrameData, OpusMode,
    OpusStereoCodec, OpusStereoEncodedAudio, SILK_LPC_ORDER, SilkEncodedFrame, SilkState,
    celt_decode_frame, celt_encode_frame, detect_mode as detect_opus_mode, estimate_pitch,
    hybrid_decode_frame, hybrid_encode_frame, levinson_durbin, lpc_analysis, lpc_residual,
    lpc_residual_stateful, lpc_synthesis, lpc_synthesis_stateful, ltp_residual, ltp_synthesis,
    silk_decode_frame, silk_decode_frame_stateful, silk_encode_frame, silk_encode_frame_stateful,
};

#[cfg(feature = "random-generation")]
pub use crate::utils::generation::{brown_noise, pink_noise, white_noise};

#[cfg(feature = "educational")]
pub use explainable::{ExplainMode, Explainable, Explaining, Explanation};

#[cfg(feature = "educational")]
pub use crate::educational::{Explain, ExplainFor};

#[cfg(all(feature = "educational", feature = "processing"))]
pub use crate::operations::traits::AudioProcessingExt;

pub use i24::{I24, PackedStruct}; // Re-export I24 type that has the AudioSample implementation

use ndarray::{Array1, Array2};
#[cfg(feature = "resampling")]
pub use resampling::{StreamingResampler, resample, resample_by_ratio, resample_scaled};

/// A tagged result that is either a 1-D mono array or a 2-D multi-channel array.
///
/// ## Purpose
///
/// Several operations on [`AudioSamples<T>`] produce output whose dimensionality
/// mirrors the input: a mono input yields a 1-D result; a multi-channel input yields
/// a 2-D result. `NdResult` captures both possibilities in a single return type,
/// letting callers branch on the actual dimensionality at runtime rather than
/// discarding the channel structure.
///
/// ## Intended Usage
///
/// Pattern-match on the variants directly, or use [`into_array1`](NdResult::into_array1) /
/// [`into_array2`](NdResult::into_array2) when you already know the expected shape.
/// Prefer the safe helpers over the `_unchecked` variants unless performance is
/// critical and the shape has already been verified.
///
/// ## Invariants
///
/// - The [`Mono`](NdResult::Mono) variant always holds a 1-D array with at least one element.
/// - The [`MultiChannel`](NdResult::MultiChannel) variant always holds a 2-D array with at
///   least one channel and at least one sample per channel.
///
/// ## Assumptions
///
/// Callers must not construct `NdResult` directly with empty arrays.
/// All library functions that return `NdResult` uphold the non-empty invariants above.
#[non_exhaustive]
pub enum NdResult<T>
where
    T: StandardSample,
{
    /// A single-channel (mono) result stored as a 1-D ndarray.
    Mono(Array1<T>),
    /// A multi-channel result stored as a 2-D ndarray (channels × samples).
    MultiChannel(Array2<T>),
}

impl<T> NdResult<T>
where
    T: StandardSample,
{
    /// Extracts the inner 1-D array if this result is [`Mono`](NdResult::Mono).
    ///
    /// # Returns
    ///
    /// `Some(array)` when the variant is `Mono`; `None` when it is `MultiChannel`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use audio_samples::NdResult;
    /// use ndarray::array;
    ///
    /// let result: NdResult<f32> = NdResult::Mono(array![0.1, 0.2, 0.3]);
    /// assert!(result.into_array1().is_some());
    ///
    /// let result: NdResult<f32> = NdResult::MultiChannel(ndarray::array![[0.1, 0.2], [0.3, 0.4]]);
    /// assert!(result.into_array1().is_none());
    /// ```
    #[inline]
    #[must_use]
    pub fn into_array1(self) -> Option<Array1<T>> {
        match self {
            Self::Mono(arr) => Some(arr),
            Self::MultiChannel(_) => None,
        }
    }

    /// Extracts the inner 1-D array without checking the variant.
    ///
    /// # Safety
    ///
    /// The caller must ensure that this `NdResult` is the [`Mono`](NdResult::Mono)
    /// variant. Calling this on a [`MultiChannel`](NdResult::MultiChannel) value is
    /// a programmer error and will panic in debug builds. There is no undefined
    /// behaviour, but the panic is not recoverable.
    ///
    /// # Returns
    ///
    /// The inner `Array1<T>`.
    ///
    /// # Panics
    ///
    /// Panics if the variant is `MultiChannel`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use audio_samples::NdResult;
    /// use ndarray::array;
    ///
    /// let result: NdResult<f32> = NdResult::Mono(array![0.1, 0.2, 0.3]);
    /// // SAFETY: we just constructed a Mono variant above
    /// let arr = unsafe { result.into_array1_unchecked() };
    /// assert_eq!(arr.len(), 3);
    /// ```
    #[inline]
    #[must_use]
    pub unsafe fn into_array1_unchecked(self) -> Array1<T> {
        match self {
            Self::Mono(arr) => arr,
            Self::MultiChannel(_) => panic!("Cannot convert MultiChannel to Array1"),
        }
    }

    /// Extracts the inner 2-D array if this result is [`MultiChannel`](NdResult::MultiChannel).
    ///
    /// # Returns
    ///
    /// `Some(array)` when the variant is `MultiChannel`; `None` when it is `Mono`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use audio_samples::NdResult;
    /// use ndarray::array;
    ///
    /// let result: NdResult<f32> = NdResult::MultiChannel(array![[0.1, 0.2], [0.3, 0.4]]);
    /// assert!(result.into_array2().is_some());
    ///
    /// let result: NdResult<f32> = NdResult::Mono(array![0.1, 0.2, 0.3]);
    /// assert!(result.into_array2().is_none());
    /// ```
    #[inline]
    #[must_use]
    pub fn into_array2(self) -> Option<Array2<T>> {
        match self {
            Self::Mono(_) => None,
            Self::MultiChannel(arr) => Some(arr),
        }
    }

    /// Extracts the inner 2-D array without checking the variant.
    ///
    /// # Safety
    ///
    /// The caller must ensure that this `NdResult` is the
    /// [`MultiChannel`](NdResult::MultiChannel) variant. Calling this on a
    /// [`Mono`](NdResult::Mono) value is a programmer error and will panic in debug
    /// builds. There is no undefined behaviour, but the panic is not recoverable.
    ///
    /// # Returns
    ///
    /// The inner `Array2<T>`.
    ///
    /// # Panics
    ///
    /// Panics if the variant is `Mono`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use audio_samples::NdResult;
    /// use ndarray::array;
    ///
    /// let result: NdResult<f32> = NdResult::MultiChannel(array![[0.1, 0.2], [0.3, 0.4]]);
    /// // SAFETY: we just constructed a MultiChannel variant above
    /// let arr = unsafe { result.into_array2_unchecked() };
    /// assert_eq!(arr.nrows(), 2);
    /// ```
    #[inline]
    #[must_use]
    pub unsafe fn into_array2_unchecked(self) -> Array2<T> {
        match self {
            Self::Mono(_) => panic!("Cannot convert Mono to Array2"),
            Self::MultiChannel(arr) => arr,
        }
    }
}
