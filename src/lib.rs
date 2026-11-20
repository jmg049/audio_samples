//! Audio as a typed value. An [`AudioSamples<T>`] holds the samples together with
//! their sample rate and channel count, and the stored format `T` is part of its
//! type. Filters, spectrograms, pitch and beat trackers, effects and the resampler
//! read the rate and the channel layout from the value, so they take no sample-rate
//! argument and cannot be handed a stale one.
//!
//! The crate is for offline analysis, feature extraction, processing chains and
//! teaching. It does no file I/O: [`audio_samples_io`](https://docs.rs/audio_samples_io)
//! reads and writes WAV, FLAC and AIFF as `AudioSamples`.
//!
//! ```rust
//! # #[cfg(all(feature = "processing", feature = "transforms", feature = "resampling"))] {
//! use audio_samples::operations::types::ResamplingQuality;
//! use audio_samples::{
//!     AudioIirFiltering, AudioProcessing, AudioSamples, AudioSpectrum, AudioTypeConversion,
//!     MelSpectrogramConfig, NormalizationConfig, nzu, sample_rate, sine_wave,
//! };
//! use ndarray::Array3;
//! use std::time::Duration;
//!
//! // One second of 440 Hz at 44.1 kHz. The sample rate is part of the value.
//! let tone: AudioSamples<'static, f32> =
//!     sine_wave(440.0, Duration::from_secs(1), sample_rate!(44_100), 0.5);
//!
//! // Operations are trait methods. The cutoff is in Hz: the filter reads the rate
//! // from `tone`. The resampled audio carries its new rate to the next stage.
//! let processed = tone
//!     .butterworth_lowpass(nzu!(4), 1_000.0)?
//!     .normalize(NormalizationConfig::peak(0.9))?
//!     .resample(sample_rate!(16_000), ResamplingQuality::High)?;
//! assert_eq!(processed.sample_rate().get(), 16_000);
//! assert_eq!(processed.samples_per_channel().get(), 16_000);
//!
//! // Analysis takes no sample-rate argument. The shape is (channels, mel bands, frames).
//! let mel: Array3<f32> = processed.melspectrogram(&MelSpectrogramConfig::default())?;
//! assert_eq!(mel.dim(), (1, 128, 32));
//!
//! // Store as 16-bit PCM. Analysis of 16-bit audio still computes in f32.
//! let pcm: AudioSamples<'static, i16> = processed.to_format();
//! let mel_pcm: Array3<f32> = pcm.melspectrogram(&MelSpectrogramConfig::default())?;
//! assert_eq!(mel_pcm.dim(), mel.dim());
//! # }
//! # Ok::<(), audio_samples::AudioSampleError>(())
//! ```
//!
//! The example needs the `processing`, `transforms` and `resampling` features:
//! `cargo add audio_samples --features processing,transforms,resampling`, or
//! `--features full` for everything (see [Feature flags](#feature-flags)).
//!
//! ## Key concepts
//!
//! ### The audio value
//!
//! - **The sample rate and channel count are never zero.** Both are `NonZeroU32`,
//!   and signatures show that type. [`sample_rate!`]`(44_100)` and
//!   [`channels!`]`(2)` check their argument at compile time, so `sample_rate!(0)`
//!   does not compile; `NonZeroU32::new` covers values known only at run time.
//! - **Audio is never empty.** Every value has at least one channel and one sample.
//!   Constructors given empty data return an error, and lengths are `NonZeroUsize`
//!   (call `.get()` for a `usize`).
//! - **Storage is planar.** Multi-channel audio is a `(channels, samples)` array with
//!   each channel contiguous. Interleaved data is converted at the edges, by
//!   [`AudioSamples::from_interleaved_vec`], [`AudioSamples::to_interleaved_vec`] and
//!   `audio_samples_io`.
//! - **The type is the format.** `T` is `u8`, `i16`, [`I24`], `i32`, `f32` or `f64`.
//!   [`to_format::<U>()`](AudioTypeConversion::to_format) converts between formats
//!   and rescales, so full scale stays full scale;
//!   [`cast_as::<U>()`](AudioTypeConversion::cast_as) keeps the numeric values.
//! - **Borrowed or owned.** `AudioSamples<'a, T>` can borrow an existing buffer
//!   without copying it ([`AudioSamples::new_mono_from_slice`]). Operations that
//!   build new audio return an owned `AudioSamples<'static, T>`.
//!
//! ### Precision
//!
//! | Sample type | Full scale | Silence | Analysis float ([`AudioSample::Float`]) |
//! |---|---|---|---|
//! | `u8` | 0 to 255 | 128 | `f32` |
//! | `i16` | −2¹⁵ to 2¹⁵ − 1 | 0 | `f32` |
//! | [`I24`] | −2²³ to 2²³ − 1 | 0 | `f32` |
//! | `i32` | −2³¹ to 2³¹ − 1 | 0 | `f64` |
//! | `f32` | −1.0 to 1.0 | 0.0 | `f32` |
//! | `f64` | −1.0 to 1.0 | 0.0 | `f64` |
//!
//! Frame-wise analysis (spectrograms, features, chroma, constant-Q transforms, onset
//! envelopes, pitch tracks) converts the samples to the smallest float type that
//! holds every value of `T` exactly, computes in that type and returns arrays of it.
//! Analysing 16-bit audio therefore takes half the memory it would in `f64`, and
//! `i32` audio loses no precision. Audio-to-audio operations return audio of the
//! input type `T`, and scalar statistics return `f64`.
//!
//! Analysis works in units of full scale: a sample of `i16::MAX` enters it as about
//! `1.0`. There are two exceptions, each documented where it applies.
#![cfg_attr(
    feature = "transforms",
    doc = r#" [`AudioTransforms`] computes in `f64` for every sample type, and"#
)]
#![cfg_attr(
    not(feature = "transforms"),
    doc = r#" `AudioTransforms` computes in `f64` for every sample type, and"#
)]
#![cfg_attr(
    feature = "statistics",
    doc = r#" the moments of [`AudioStatistics`] (mean, RMS, variance)"#
)]
#![cfg_attr(
    not(feature = "statistics"),
    doc = r#" the moments of `AudioStatistics` (mean, RMS, variance)"#
)]
#![cfg_attr(
    feature = "envelopes",
    doc = r#" and the [`operations::envelopes`] use the raw stored values."#
)]
#![cfg_attr(
    not(feature = "envelopes"),
    doc = r#" and the `operations::envelopes` use the raw stored values."#
)]
//!
//! ### Array layout
//!
//! Frame-wise results are `ndarray` arrays shaped `(channels, features, frames)`.
//! Mono input gives one channel, and a per-frame scalar (a centroid, an RMS value)
//! has one feature row. With centring on, the default, frame `t` is centred on
//! sample `t × hop_length`, at `t × hop_length / sample_rate` seconds.
//!
//! ### Configuration
//!
//! An operation with more than a few parameters takes a config struct. Configs
//! implement `Default` unless a parameter has no sensible default; then `new` takes
//! it (the frequency range of a pitch tracker, for example). Configs that match a
//! librosa function also have a `librosa()` constructor with librosa's defaults.
//! Where `Default` differs from `librosa()`, the config's documentation says how.
//!
//! ## How the crate is organised
//!
//! ### Operations
//!
//! Most operations are methods of one trait per area, implemented for
//! `AudioSamples` and, in most cases, re-exported at the crate root: bring the trait
//! into scope and call the method. Transforming operations come in pairs. `op(&self, ..)` returns
//! a new owned value and leaves the receiver unchanged, so calls chain;
//! `op_in_place(&mut self, ..)` modifies the receiver. Free functions cover the
//! rest, including the `*_from_*` variants that take a precomputed spectrogram or
//! envelope, so that one STFT can feed several features.
//!
//! Fallible calls return [`AudioSampleResult`]. Every [`AudioSampleError`] carries a
//! stable code and a help message through [`miette::Diagnostic`]. With the `fancy`
//! feature, miette renders them in colour, with carets under the offending input.
//!
//! ### Where to find things
//!
//! | Task | Where |
//! |---|---|
//! | Read or write a file | [`audio_samples_io`](https://docs.rs/audio_samples_io) |
//! | Generate test signals and example clips | [`sine_wave`], [`chirp`], [`example_clip`] and the rest of [`utils::generation`] |
#![cfg_attr(
    feature = "statistics",
    doc = r#" | Peak, RMS, zero crossings, correlation | [`AudioStatistics`] |"#
)]
#![cfg_attr(
    not(feature = "statistics"),
    doc = r#" | Peak, RMS, zero crossings, correlation | `AudioStatistics` |"#
)]
#![cfg_attr(
    feature = "processing",
    doc = r#" | Gain, normalisation, clipping, DC removal, μ-law | [`AudioProcessing`] |"#
)]
#![cfg_attr(
    not(feature = "processing"),
    doc = r#" | Gain, normalisation, clipping, DC removal, μ-law | `AudioProcessing` |"#
)]
#![cfg_attr(
    feature = "editing",
    doc = r#" | Trim, pad, fade, mix, concatenate, split | [`AudioEditing`] |"#
)]
#![cfg_attr(
    not(feature = "editing"),
    doc = r#" | Trim, pad, fade, mix, concatenate, split | `AudioEditing` |"#
)]
#![cfg_attr(
    feature = "channels",
    doc = r#" | Mono and stereo conversion, panning, interleaving | [`AudioChannelOps`] |"#
)]
#![cfg_attr(
    not(feature = "channels"),
    doc = r#" | Mono and stereo conversion, panning, interleaving | `AudioChannelOps` |"#
)]
#![cfg_attr(
    feature = "iir-filtering",
    doc = r#" | Butterworth, Chebyshev, elliptic and Bessel filters | [`AudioIirFiltering`] |"#
)]
#![cfg_attr(
    not(feature = "iir-filtering"),
    doc = r#" | Butterworth, Chebyshev, elliptic and Bessel filters | `AudioIirFiltering` |"#
)]
#![cfg_attr(
    feature = "parametric-eq",
    doc = r#" | Parametric and three-band EQ | [`AudioParametricEq`] |"#
)]
#![cfg_attr(
    not(feature = "parametric-eq"),
    doc = r#" | Parametric and three-band EQ | `AudioParametricEq` |"#
)]
#![cfg_attr(
    feature = "dynamic-range",
    doc = r#" | Compressor, limiter, gate, expander | [`AudioDynamicRange`] |"#
)]
#![cfg_attr(
    not(feature = "dynamic-range"),
    doc = r#" | Compressor, limiter, gate, expander | `AudioDynamicRange` |"#
)]
#![cfg_attr(
    feature = "envelopes",
    doc = r#" | Amplitude, RMS and analytic envelopes | [`AudioEnvelopes`](operations::AudioEnvelopes) |"#
)]
#![cfg_attr(
    not(feature = "envelopes"),
    doc = r#" | Amplitude, RMS and analytic envelopes | `AudioEnvelopes` |"#
)]
#![cfg_attr(
    feature = "resampling",
    doc = r#" | Sample-rate conversion, whole-buffer or streamed | [`resample`], [`StreamingResampler`] |"#
)]
#![cfg_attr(
    not(feature = "resampling"),
    doc = r#" | Sample-rate conversion, whole-buffer or streamed | `resample`, `StreamingResampler` |"#
)]
#![cfg_attr(
    feature = "transforms",
    doc = r#" | STFT, mel spectrogram, MFCC, decibels | [`AudioSpectrum`] |"#
)]
#![cfg_attr(
    not(feature = "transforms"),
    doc = r#" | STFT, mel spectrogram, MFCC, decibels | `AudioSpectrum` |"#
)]
#![cfg_attr(
    feature = "feature-extraction",
    doc = r#" | Frame-wise spectral features, RMS, zero-crossing rate | [`AudioFeatures`] |"#
)]
#![cfg_attr(
    not(feature = "feature-extraction"),
    doc = r#" | Frame-wise spectral features, RMS, zero-crossing rate | `AudioFeatures` |"#
)]
#![cfg_attr(
    feature = "feature-extraction",
    doc = r#" | Chroma and tuning | [`AudioTonal`] |"#
)]
#![cfg_attr(
    not(feature = "feature-extraction"),
    doc = r#" | Chroma and tuning | `AudioTonal` |"#
)]
#![cfg_attr(
    all(feature = "transforms", feature = "resampling"),
    doc = r#" | Constant-Q and variable-Q transforms | [`AudioConstantQ`] |"#
)]
#![cfg_attr(
    not(all(feature = "transforms", feature = "resampling")),
    doc = r#" | Constant-Q and variable-Q transforms | `AudioConstantQ` |"#
)]
#![cfg_attr(
    feature = "pitch-analysis",
    doc = r#" | Pitch tracking (YIN, pYIN) | [`AudioPitchTracking`] |"#
)]
#![cfg_attr(
    not(feature = "pitch-analysis"),
    doc = r#" | Pitch tracking (YIN, pYIN) | `AudioPitchTracking` |"#
)]
#![cfg_attr(
    feature = "pitch-analysis",
    doc = r#" | Key estimation | [`AudioPitchAnalysis::estimate_key`] |"#
)]
#![cfg_attr(
    not(feature = "pitch-analysis"),
    doc = r#" | Key estimation | `AudioPitchAnalysis::estimate_key` |"#
)]
#![cfg_attr(
    feature = "beat-tracking",
    doc = r#" | Onsets, tempo and beats | [`AudioRhythm`] |"#
)]
#![cfg_attr(
    not(feature = "beat-tracking"),
    doc = r#" | Onsets, tempo and beats | `AudioRhythm` |"#
)]
#![cfg_attr(
    feature = "effects",
    doc = r#" | Time stretch, pitch shift, HPSS, silence trimming | [`AudioEffects`] |"#
)]
#![cfg_attr(
    not(feature = "effects"),
    doc = r#" | Time stretch, pitch shift, HPSS, silence trimming | `AudioEffects` |"#
)]
#![cfg_attr(
    feature = "vad",
    doc = r#" | Voice activity detection | [`AudioVoiceActivityDetection`] |"#
)]
#![cfg_attr(
    not(feature = "vad"),
    doc = r#" | Voice activity detection | `AudioVoiceActivityDetection` |"#
)]
#![cfg_attr(
    feature = "sequence",
    doc = r#" | DTW, Viterbi decoding, recurrence matrices | [`operations::sequence`] |"#
)]
#![cfg_attr(
    not(feature = "sequence"),
    doc = r#" | DTW, Viterbi decoding, recurrence matrices | `operations::sequence` |"#
)]
#![cfg_attr(
    feature = "psychoacoustic",
    doc = r#" | Masking thresholds and perceptual coding | [`codecs`] |"#
)]
#![cfg_attr(
    not(feature = "psychoacoustic"),
    doc = r#" | Masking thresholds and perceptual coding | `codecs` |"#
)]
#![cfg_attr(
    feature = "plotting",
    doc = r#" | Interactive plots | [`AudioPlotting`](operations::AudioPlotting) |"#
)]
#![cfg_attr(
    not(feature = "plotting"),
    doc = r#" | Interactive plots | `AudioPlotting` |"#
)]
//!
#![cfg_attr(
    feature = "transforms",
    doc = r#" Some tasks have two APIs, such as [`AudioTransforms::mfcc`] and"#
)]
#![cfg_attr(
    not(feature = "transforms"),
    doc = r#" Some tasks have two APIs, such as `AudioTransforms::mfcc` and"#
)]
#![cfg_attr(
    feature = "transforms",
    doc = r#" [`AudioSpectrum::mfcc_frames`]. They follow different conventions; the table"#
)]
#![cfg_attr(
    not(feature = "transforms"),
    doc = r#" `AudioSpectrum::mfcc_frames`. They follow different conventions; the table"#
)]
//! [Choosing between overlapping APIs](operations#choosing-between-overlapping-apis)
//! says which to use for what.
//!
//! ### Feature flags
//!
//! The default build (`bare-bones`) has no optional features. It contains
//! [`AudioSamples`], the sample-format conversions, the iterators, [`utils`],
//! [`operations::traits`] and [`operations::types`]. Each feature below adds the
//! modules next to it and turns on the features in the last column. On docs.rs,
//! every item names the feature it needs. A module name in the table is a link when
//! this documentation was built with its feature, and plain text otherwise. In a
//! checkout of the repository, `cargo docs` builds and opens the documentation with
//! every feature that builds offline.
//!
//! | Feature | Adds | Also enables |
//! |---|---|---|
#![cfg_attr(
    feature = "statistics",
    doc = r#" | `statistics` | [`operations::statistics`] | |"#
)]
#![cfg_attr(
    not(feature = "statistics"),
    doc = r#" | `statistics` | `operations::statistics` | |"#
)]
#![cfg_attr(
    feature = "processing",
    doc = r#" | `processing` | [`operations::processing`] | `statistics`, `iir-filtering` |"#
)]
#![cfg_attr(
    not(feature = "processing"),
    doc = r#" | `processing` | `operations::processing` | `statistics`, `iir-filtering` |"#
)]
#![cfg_attr(
    feature = "editing",
    doc = r#" | `editing` | [`operations::editing`], window iterators | `statistics`, `random-generation` |"#
)]
#![cfg_attr(
    not(feature = "editing"),
    doc = r#" | `editing` | `operations::editing`, window iterators | `statistics`, `random-generation` |"#
)]
#![cfg_attr(
    feature = "channels",
    doc = r#" | `channels` | [`operations::channels`] | |"#
)]
#![cfg_attr(
    not(feature = "channels"),
    doc = r#" | `channels` | `operations::channels` | |"#
)]
#![cfg_attr(
    feature = "iir-filtering",
    doc = r#" | `iir-filtering` | [`operations::iir_filtering`] | |"#
)]
#![cfg_attr(
    not(feature = "iir-filtering"),
    doc = r#" | `iir-filtering` | `operations::iir_filtering` | |"#
)]
#![cfg_attr(
    feature = "parametric-eq",
    doc = r#" | `parametric-eq` | [`operations::parametric_eq`] | `iir-filtering`, `channels` |"#
)]
#![cfg_attr(
    not(feature = "parametric-eq"),
    doc = r#" | `parametric-eq` | `operations::parametric_eq` | `iir-filtering`, `channels` |"#
)]
#![cfg_attr(
    feature = "dynamic-range",
    doc = r#" | `dynamic-range` | [`operations::dynamic_range`] | |"#
)]
#![cfg_attr(
    not(feature = "dynamic-range"),
    doc = r#" | `dynamic-range` | `operations::dynamic_range` | |"#
)]
#![cfg_attr(
    feature = "envelopes",
    doc = r#" | `envelopes` | [`operations::envelopes`] | `dynamic-range`, `editing` |"#
)]
#![cfg_attr(
    not(feature = "envelopes"),
    doc = r#" | `envelopes` | `operations::envelopes` | `dynamic-range`, `editing` |"#
)]
#![cfg_attr(feature = "vad", doc = r#" | `vad` | [`operations::vad`] | |"#)]
#![cfg_attr(not(feature = "vad"), doc = r#" | `vad` | `operations::vad` | |"#)]
#![cfg_attr(
    feature = "peak-picking",
    doc = r#" | `peak-picking` | [`operations::peak_picking`] | |"#
)]
#![cfg_attr(
    not(feature = "peak-picking"),
    doc = r#" | `peak-picking` | `operations::peak_picking` | |"#
)]
#![cfg_attr(
    feature = "resampling",
    doc = r#" | `resampling` | [`resampling`] | |"#
)]
#![cfg_attr(
    not(feature = "resampling"),
    doc = r#" | `resampling` | `resampling` | |"#
)]
#![cfg_attr(
    feature = "transforms",
    doc = r#" | `transforms` | [`operations::transforms`], [`operations::spectrum`], [`operations::filterbanks`] | |"#
)]
#![cfg_attr(
    not(feature = "transforms"),
    doc = r#" | `transforms` | `operations::transforms`, `operations::spectrum`, `operations::filterbanks` | |"#
)]
#![cfg_attr(
    all(feature = "transforms", feature = "resampling"),
    doc = r#" | `transforms` + `resampling` | [`operations::constant_q`] | |"#
)]
#![cfg_attr(
    not(all(feature = "transforms", feature = "resampling")),
    doc = r#" | `transforms` + `resampling` | `operations::constant_q` | |"#
)]
#![cfg_attr(
    feature = "feature-extraction",
    doc = r#" | `feature-extraction` | [`operations::features`], [`operations::tonal`] | `transforms` |"#
)]
#![cfg_attr(
    not(feature = "feature-extraction"),
    doc = r#" | `feature-extraction` | `operations::features`, `operations::tonal` | `transforms` |"#
)]
#![cfg_attr(
    feature = "onset-detection",
    doc = r#" | `onset-detection` | [`operations::onset`] | `transforms`, `peak-picking`, `processing` |"#
)]
#![cfg_attr(
    not(feature = "onset-detection"),
    doc = r#" | `onset-detection` | `operations::onset` | `transforms`, `peak-picking`, `processing` |"#
)]
#![cfg_attr(
    feature = "beat-tracking",
    doc = r#" | `beat-tracking` | [`operations::beat`], [`operations::rhythm`] | `onset-detection` |"#
)]
#![cfg_attr(
    not(feature = "beat-tracking"),
    doc = r#" | `beat-tracking` | `operations::beat`, `operations::rhythm` | `onset-detection` |"#
)]
#![cfg_attr(
    feature = "decomposition",
    doc = r#" | `decomposition` | [`operations::hpss`] | `onset-detection` |"#
)]
#![cfg_attr(
    not(feature = "decomposition"),
    doc = r#" | `decomposition` | `operations::hpss` | `onset-detection` |"#
)]
#![cfg_attr(
    feature = "pitch-analysis",
    doc = r#" | `pitch-analysis` | [`operations::pitch_analysis`], [`operations::pitch_tracking`] | `transforms`, `sequence` |"#
)]
#![cfg_attr(
    not(feature = "pitch-analysis"),
    doc = r#" | `pitch-analysis` | `operations::pitch_analysis`, `operations::pitch_tracking` | `transforms`, `sequence` |"#
)]
#![cfg_attr(
    feature = "effects",
    doc = r#" | `effects` | [`operations::effects`] | `transforms`, `resampling`, `editing`, `sequence` |"#
)]
#![cfg_attr(
    not(feature = "effects"),
    doc = r#" | `effects` | `operations::effects` | `transforms`, `resampling`, `editing`, `sequence` |"#
)]
#![cfg_attr(
    feature = "sequence",
    doc = r#" | `sequence` | [`operations::sequence`] | |"#
)]
#![cfg_attr(
    not(feature = "sequence"),
    doc = r#" | `sequence` | `operations::sequence` | |"#
)]
#![cfg_attr(
    feature = "linalg",
    doc = r#" | `linalg` | [`operations::linalg`] | |"#
)]
#![cfg_attr(
    not(feature = "linalg"),
    doc = r#" | `linalg` | `operations::linalg` | |"#
)]
#![cfg_attr(
    feature = "feature-inversion",
    doc = r#" | `feature-inversion` | [`operations::inverse`] | `feature-extraction`, `effects`, `linalg` |"#
)]
#![cfg_attr(
    not(feature = "feature-inversion"),
    doc = r#" | `feature-inversion` | `operations::inverse` | `feature-extraction`, `effects`, `linalg` |"#
)]
#![cfg_attr(
    feature = "factorization",
    doc = r#" | `factorization` | [`operations::factorization`] | `linalg` |"#
)]
#![cfg_attr(
    not(feature = "factorization"),
    doc = r#" | `factorization` | `operations::factorization` | `linalg` |"#
)]
#![cfg_attr(
    feature = "pitch-filterbank",
    doc = r#" | `pitch-filterbank` | [`operations::pitch_filterbank`] | `iir-filtering`, `resampling`, `transforms` |"#
)]
#![cfg_attr(
    not(feature = "pitch-filterbank"),
    doc = r#" | `pitch-filterbank` | `operations::pitch_filterbank` | `iir-filtering`, `resampling`, `transforms` |"#
)]
#![cfg_attr(
    feature = "mellin",
    doc = r#" | `mellin` | [`operations::mellin`] | `transforms` |"#
)]
#![cfg_attr(
    not(feature = "mellin"),
    doc = r#" | `mellin` | `operations::mellin` | `transforms` |"#
)]
#![cfg_attr(
    feature = "segmentation",
    doc = r#" | `segmentation` | [`operations::segment`] | `sequence`, `transforms` |"#
)]
#![cfg_attr(
    not(feature = "segmentation"),
    doc = r#" | `segmentation` | `operations::segment` | `sequence`, `transforms` |"#
)]
#![cfg_attr(
    feature = "psychoacoustic",
    doc = r#" | `psychoacoustic` | [`codecs`], [`codecs::perceptual`] | `transforms` |"#
)]
#![cfg_attr(
    not(feature = "psychoacoustic"),
    doc = r#" | `psychoacoustic` | `codecs`, `codecs::perceptual` | `transforms` |"#
)]
#![cfg_attr(
    feature = "opus-codec",
    doc = r#" | `opus-codec` | [`codecs::opus`] | `psychoacoustic`, `statistics` |"#
)]
#![cfg_attr(
    not(feature = "opus-codec"),
    doc = r#" | `opus-codec` | `codecs::opus` | `psychoacoustic`, `statistics` |"#
)]
#![cfg_attr(
    feature = "plotting",
    doc = r#" | `plotting` | [`operations::plotting`] | `channels`, `transforms`, `editing` |"#
)]
#![cfg_attr(
    not(feature = "plotting"),
    doc = r#" | `plotting` | `operations::plotting` | `channels`, `transforms`, `editing` |"#
)]
#![cfg_attr(
    feature = "fixed-size-audio",
    doc = r#" | `fixed-size-audio` | [`fixed_audio`] | |"#
)]
#![cfg_attr(
    not(feature = "fixed-size-audio"),
    doc = r#" | `fixed-size-audio` | `fixed_audio` | |"#
)]
#![cfg_attr(
    feature = "educational",
    doc = r#" | `educational` | [`educational`] | `processing`, `fancy` |"#
)]
#![cfg_attr(
    not(feature = "educational"),
    doc = r#" | `educational` | `educational` | `processing`, `fancy` |"#
)]
#![cfg_attr(
    all(feature = "educational", feature = "full_no_plotting"),
    doc = r#" | `educational` + `full_no_plotting` | [`guide`] | |"#
)]
#![cfg_attr(
    not(all(feature = "educational", feature = "full_no_plotting")),
    doc = r#" | `educational` + `full_no_plotting` | `guide` | |"#
)]
#![cfg_attr(
    feature = "full_no_plotting",
    doc = r#" | `full_no_plotting` | [`librosa_migration`], [`moving_from_librosa_rust`] | every feature above except `plotting` and `educational` |"#
)]
#![cfg_attr(
    not(feature = "full_no_plotting"),
    doc = r#" | `full_no_plotting` | `librosa_migration`, `moving_from_librosa_rust` | every feature above except `plotting` and `educational` |"#
)]
//! | `full` | | `full_no_plotting`, `plotting` |
//!
//! Features that change behaviour rather than add modules:
//!
//! | Feature | Effect |
//! |---|---|
//! | `random-generation` | White, pink and brown noise generators, and the operations that add noise |
//! | `parallel` | Encodes and decodes the segments of the perceptual codec on several threads |
//! | `simd` | Vectorised conversion between `f32` and `i16` or `i32` in [`simd_conversions`] |
//! | `fancy` | Coloured diagnostics, with carets, for [`AudioSampleError`] |
//! | `static-plots` | PNG and SVG export of plots; enables `plotting` |
//! | `html_view` | Opens plots in a browser window |
//! | `python` | Python conversion of [`I24`], for the Python bindings |
//!
//! `full` enables everything except `educational`, `fancy`, `simd`,
//! `static-plots`, `html_view` and `python`.
//!
//! ### Performance
//!
//! Constructors can borrow existing buffers, and channel and frame access returns
//! views, so reading audio rarely copies it. Analysis of 8-, 16- and 24-bit audio
//! computes in `f32`, and FFT plans are cached per thread and reused across calls.
//!
//! ## Learning and related crates
//!
#![cfg_attr(
    all(feature = "educational", feature = "full_no_plotting"),
    doc = r#" - [`guide`]: a short course in speech and audio processing, fourteen chapters"#
)]
#![cfg_attr(
    not(all(feature = "educational", feature = "full_no_plotting")),
    doc = r#" - `guide`: a short course in speech and audio processing, fourteen chapters"#
)]
//!   from sampling to perceptual coding. Every code block is a doctest, and each
//!   chapter has a runnable lab in `examples/tutorial_NN_*.rs`.
//! - [The illustrated book](https://jmg049.github.io/audio_samples/): the same
//!   chapters with figures computed by the library.
#![cfg_attr(
    feature = "educational",
    doc = r#" - [`educational`]: the explain API. Configs and topics return documents with"#
)]
#![cfg_attr(
    not(feature = "educational"),
    doc = r#" - `educational`: the explain API. Configs and topics return documents with"#
)]
//!   the formula the code evaluates, each symbol with its unit, and the numbers for
//!   your configuration.
#![cfg_attr(
    feature = "full_no_plotting",
    doc = r#" - [`librosa_migration`] and [`moving_from_librosa_rust`]: librosa functions"#
)]
#![cfg_attr(
    not(feature = "full_no_plotting"),
    doc = r#" - `librosa_migration` and `moving_from_librosa_rust`: librosa functions"#
)]
//!   mapped to their counterparts here, and a worked port of a librosa pipeline.
//! - [`audio_samples_io`](https://docs.rs/audio_samples_io): read and write WAV
//!   (including RF64), FLAC and AIFF as `AudioSamples`, whole or streamed.
//! - [`audio_samples_streaming`](https://docs.rs/audio_samples_streaming): chunked
//!   streaming of `AudioSamples` between files, audio devices and async runtimes.
//! - [`audio_samples_python`](https://github.com/jmg049/audio_samples_python): the
//!   same types and operations from Python and NumPy (`pip install audio_samples`),
//!   with the guide's chapters as notebooks.
//! - [`spectrograms`](https://docs.rs/spectrograms) and [`i24`](https://docs.rs/i24):
//!   dependencies whose types appear in public signatures (`WindowType`,
//!   `StftParams`, `I24`).

// docs.rs builds with `--cfg docsrs` on nightly: label feature-gated items with the
// features they need.
#![cfg_attr(docsrs, feature(doc_cfg))]

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

#[cfg(feature = "psychoacoustic")]
pub mod codecs;
pub mod conversions;
pub mod iterators;
pub mod operations;
#[cfg(feature = "resampling")]
pub mod resampling;
pub mod traits;
pub mod utils;

mod audio_float;
mod error;
#[cfg(feature = "fixed-size-audio")]
pub mod fixed_audio;
mod repr;

pub mod simd_conversions;

#[cfg(feature = "educational")]
pub mod educational;

// Documentation only: renders documentation/LIBROSA_MIGRATION.md and runs its
// examples as doctests. The module has no items.
#[cfg(all(feature = "full_no_plotting", any(doc, doctest)))]
pub mod librosa_migration {
    #![doc = include_str!("../documentation/LIBROSA_MIGRATION.md")]
}

// Documentation only: renders documentation/MOVING_FROM_LIBROSA_RUST.md and runs
// its examples as doctests. The module has no items.
#[cfg(all(feature = "full_no_plotting", any(doc, doctest)))]
pub mod moving_from_librosa_rust {
    #![doc = include_str!("../documentation/MOVING_FROM_LIBROSA_RUST.md")]
}

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
    /// variant. Calling this on a [`MultiChannel`](NdResult::MultiChannel) value
    /// panics in every build profile. It cannot cause undefined behaviour.
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
    /// [`Mono`](NdResult::Mono) value panics in every build profile. It cannot
    /// cause undefined behaviour.
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
