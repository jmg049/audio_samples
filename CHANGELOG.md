# Changelog

All notable changes to this project are documented here.

## [Unreleased]

- **`ResamplingQuality::VeryHigh`**: windowed sinc with a 185 dB stopband from the Nyquist
  frequency on (ripple below 1e-9) and its passband to 0.95 of the lower Nyquist frequency for
  decimation by a power of two, 0.98 for other ratios whose terms have no prime factor above 7
  (1 : 2, 22050 to 16000 Hz, 44100 to 48000 Hz) and 0.996 for the rest (pitch shifts, 44100 to
  44101 Hz), where it is flat to 2e-11 at 0.995. Interpolated stage-two tables (ratios with more
  than 2^15 output phases) are interpolated quadratically at this tier, so they reach its
  stopband. With `f32` samples the result is limited by `f32` rounding (about -140 dB). Parsed
  from `"very_high"` or `"vhq"`.
- **Long stage-one filters run by FFT up to 2^16 taps** (was 2^14): decimation by up to about
  700 : 1 at `High` no longer falls back to the direct filter. Such stages keep a second, smaller
  transform for short inputs, and the transform size model now favours sizes up to 2^17 points
  for long filters.
- **`StreamingResampler` equals `resample` for pitch-shift ratios too.** Downsampling by a ratio
  whose terms have a prime factor above 7 (22050 to 17501 Hz and the like) now uses one plan for
  every input length (the first stage at the input rate), so the stream gives `resample`'s output
  to rounding from its first sample, holding back only the filters' look-ahead. Inputs of one to
  three cycles of such a ratio read each row of the interpolation table once for two outputs.
- **`AudioSamples` implements `audioadapter` 5's `Adapter`** (previously `audioadapter` 3, through
  `rubato` 3). The `resampling` feature no longer depends on `rubato` or `audioadapter-buffers`.
- **Conversion by a ratio whose terms have no prime factor above 7 runs block by block in the
  frequency domain** (22050 to 16000 or 8000 Hz, 44100 or 48000 to 16000 Hz, 48000 to 44100 Hz,
  16000 to 22050 Hz, 44100 to 48000 Hz and the like; decimation by a power of two and 1 : 2
  upsampling keep their own paths). Each output is the
  windowed-sinc filter evaluated at its exact position, without the second, interpolating stage;
  the filter's response above the lower Nyquist frequency is dropped instead of aliased into the
  band or left as images.
  Outputs change at the stopband level (about 1e-8 on 22050 to 8000 Hz), stopbands measure
  equal or lower, and 10 s conversions take 40-55 % less time. `StreamingResampler` converts the
  same blocks, so it still matches `resample`, but for these ratios it holds back up to one
  block of input (a few thousand samples) before it returns output.
- **`resample_scaled`** resamples float audio and multiplies each output sample by a gain as it
  is written, the same values as multiplying afterwards without the second pass.

- **`wavelet_lengths_with_alpha` and `wavelet_with_alpha` accept one frequency.** The
  bandwidths are given, so the two-frequency minimum, which exists to derive them from the
  spacing, no longer applies. `wavelet_lengths` and `wavelet` still need two.
- **More of librosa's option values.** `operations::transforms::magphase_powf` takes any
  `power`. `constant_q::griffinlim_cqt_random` starts Griffin-Lim from random phases
  (`init="random"`, this crate's RNG). `spectral_{centroid,bandwidth,rolloff}_from_spectrogram_with_freqs`
  and `poly_features_from_spectrogram_with_freqs` take a frequency for every bin of every frame.
  `rhythm::onset_strength_multi_slices` / `onset_strength_multi_from_db_slices` and
  `sequence::sync_slices` / `sync_slices_with` take `ndarray::Slice`s. `rhythm::metrogram_per_bin`
  is `metrogram(aggregate=None)`. `effects::nn_filter_sparse` takes a compressed sparse column
  matrix (`SparseRecurrence`, with `i32` or `i64` indices), and `nn_filter_with`,
  `nn_filter_sparse_with` and `nn_filter_recurrence_with` any aggregate. `effects::trim_db_with`,
  `split_db_with` and `nonsilent_frames_with` take closures for the reference level and the
  channel combination. Each is checked against librosa 1.0.0 golden fixtures.
- **`trim_db`, `split_db` and `nonsilent_frames` sum each frame from hop-long blocks** when the
  frame length and padding are whole hops (the defaults), so every square is summed once. A frame
  whose level lies within round-off of the threshold can be classified differently than before.
- **Minimum supported Rust version is now 1.95**, the version the `non-empty-slice`,
  `non-empty-iter` and `non-zero-size` dependencies require.
- **Everything else librosa 1.0 computes.** New modules: `operations::linalg` (feature `linalg`:
  `nnls`, `nnls_matrix`, `pinv`, `eigh`, `svd`, `randomized_svd`); `operations::inverse` (feature
  `feature-inversion`: `mel_to_stft`, `mel_to_audio`, `mfcc_to_mel`, `mfcc_to_audio`);
  `operations::factorization` (feature `factorization`: `nmf`, `nmf_transform`, `decompose`, with
  coordinate descent and multiplicative updates and the NNDSVD initialisations);
  `operations::pitch_filterbank` (feature `pitch-filterbank`: `mr_frequencies`,
  `semitone_filterbank`, `iirt`); `operations::mellin` (feature `mellin`: `mellin_transform`);
  `operations::segment` (feature `segmentation`: `diagonal_filter`, `path_enhance`,
  `timelag_filter`, `agglomerative`, `subsegment`); `operations::plotting::matrix` (`plot_matrix`,
  `plot_waveform_bars`, `plot_waveform_f0`); `utils::array_ops` (`localmax`, `localmin`,
  `match_events`, `match_intervals`, `interp1d`, `pad_center`, `axis_sort`, `sparsify_rows`,
  `zero_crossings`, `tone`, `chirp` and others); `utils::notation` (tuning systems, the Functional Just System, svara, mela and thaat
  names, and `midi_name` with `octave`, `cents`, `key` and `unicode`). Also
  `sequence::path_to_steps`, `audio_math::blocks_to_{frames,samples,time}`,
  `audio_math::note_to_midi_fractional`, `ExampleClip::description`,
  `constant_q::vqt_with_intervals` for bins in any tuning system, and
  `constant_q::{wavelet_with_alpha, wavelet_lengths_with_alpha}` for explicit bandwidths. All of
  them are checked against librosa 1.0.0 golden fixtures; `documentation/LIBROSA_MIGRATION.md`
  maps each librosa name.
- **Integer arithmetic saturates.** `+`, `-`, `*`, `/` and negation of integer `AudioSamples` overflowed the
  sample type, which panicked in a debug build and wrapped in a release build; dividing by zero panicked. They now saturate
  at the type's limits, and an integer divided by zero gives the limit with its sign (zero divided by zero gives
  zero). `u8` is offset binary, so its arithmetic works on the offset value and 128 is the neutral element of `+`.
- **Example clips.** `utils::generation::example_clip` renders named clips of music and speech
  (`ExampleClip::{Chords, DrumLoop, ChordsAndDrums, Speech, NoisyRecording}`) at any sample rate,
  duration and sample type. Nothing is stored or downloaded. Rendering uses only basic IEEE 754
  arithmetic, so the same arguments give the same bytes on any platform with IEEE 754 `f64`
  (checked on x86-64 glibc and musl, and on aarch64 musl under QEMU). `EXAMPLE_TEMPO_BPM`,
  `EXAMPLE_NOISY_SNR_DB` and `example_speech_f0` give the known contents.
- **Faster, with the same results.** The median filter behind `hpss_masks` and `hpss_magnitude` slides a
  sorted window. `melspectrogram` and `mfcc_frames` project each block of frames rather than
  holding the whole spectrogram. `to_mono` makes one pass over interleaved stereo.
- **Plotting.** A spectrogram zoomed below 1 kHz is labelled in kHz, the unit of its values.
- **librosa-compatible API.** A librosa 1.0-compatible API alongside the existing one:
  the `AudioSpectrum`, `AudioFeatures`, `AudioTonal`, `AudioConstantQ`,
  `AudioRhythm`, `AudioEffects` and `AudioPitchTracking` traits, the
  `operations::sequence` module and librosa unit conversions in
  `utils::audio_math`, with new features `feature-extraction`, `effects` and
  `sequence` (all in `full_no_plotting`). Every new config has a `librosa()`
  constructor with librosa's defaults; `Default` equals it except where librosa's
  default changes the data without being asked (see *librosa-compatible API:
  precision and defaults*). Outputs are `(channels, features, frames)` arrays in
  the input's float type (`f32` for `u8`/`i16`/`I24`/`f32` audio, `f64` for
  `i32`/`f64`), and results are verified against librosa 1.0.0 golden fixtures
  (`tests/analysis_reference/`). Bugs in existing analysis code are fixed as
  well (YIN, `estimate_key`, autocorrelation, mu-law, the resampler, ...; see
  *Fixed*), and the default onset peak picker finds onsets (see *Changed*). `documentation/LIBROSA_MIGRATION.md`
  (rendered as the `librosa_migration` module on docs.rs) maps every librosa
  function to its counterpart and lists every difference.

### Output changes in released (2.0.x) APIs

These change the output of existing operations for **integer** sample types
(`u8`, `i16`, `I24`, `i32`). Output for `f32` and `f64` audio is unchanged
except for the resampler (below).

- **`normalize` with `NormalizationMethod::ZScore` returns
  `AudioSampleError::Unsupported` for integer sample types.** It used to return
  z-scores truncated to the integers -1..=1, which is silence. Convert first
  with `to_format::<f64>()`.
- **`to_mono(MonoConversionMethod::Weighted(..))` and
  `to_mono(MonoConversionMethod::Center)` returned all-zero output** for integer
  audio; they now return the weighted sum and the channel average.
- `normalize` with `Mean` or `Median`, and `remove_dc_offset`, round integer
  results to the nearest value and saturate, instead of truncating toward zero
  (Mean of `[1, 2, 4]` as `i16` is now `[-1, 0, 2]`, was `[-1, 0, 1]`). `I24`
  no longer wraps on overflow.
- `resample`, `resample_by_ratio` and `StreamingResampler` (windowed-sinc
  tiers) evaluate the same filter specification as a two-stage cascade (FFT
  low-pass, then a short interpolator) and compute in the sample type's float
  (`f32` for `f32`, `i16`, `I24`, `u8`). This changes `f32` and `f64` output
  too: by about 1e-15 of the peak for 2:1 and 1:2, up to 3.5e-8 (f64) for
  other ratios, and by f32 round-off (about 4e-7) for `f32` input. Passband
  ripple, passband edge and stopband attenuation are unchanged (measured within
  0.5 dB).
- `to_mono(Average)`, `to_stereo(StereoConversionMethod::Pan(..))`, `pan` and
  `balance` round and saturate integer results instead of truncating. `pan` and
  `balance` no longer wrap `I24` on overflow, and for `u8` they scale around the
  mid-scale value 128.

### librosa-compatible API: precision and defaults

The librosa-compatible API is unreleased, so these are changes to the
`[Unreleased]` additions listed below, not to 2.0.x.

- **`magphase` computes `|D|` as `sqrt(re² + im²)`** in one vectorised pass
  (`hypot` only where the squares overflow or underflow), about 2-3x faster.
  The magnitude differs from `hypot` by at most 1 ulp (about 11% of the values
  of a 10 s STFT change, max relative change 2.2e-16), and the phase, divided
  by it, by at most 1-2 ulp.
- **`phase_vocoder` / `phase_vocoder_at` in `f64` carry the phase as a unit
  phasor** (`D / |D|` products) instead of summing `atan2` angles and taking
  `cos` / `sin`: the same quantity without the transcendental calls (about 6x
  faster). Output differs from the angle form by round-off (2.6e-14 of the peak
  on a 10 s STFT), within the existing 1e-9 agreement with librosa. `f32`
  keeps librosa's float32 angle arithmetic and is unchanged.

- **Analysis output follows the input precision.** The traits gain
  `type Float` (= `T::Float`), and their outputs are `Array*<Self::Float>` /
  `Array*<Complex<Self::Float>>`: `f32` for `u8`, `i16`, `I24` and `f32` audio,
  `f64` for `i32` and `f64` audio. The computation runs natively in that type,
  so `f32` analysis needs half the memory. The free functions on arrays are
  generic over `F: AudioFloat` and return the element type of their input;
  untyped float literals passed to them can need a suffix (`1.0_f64`). Filter
  tables (windows, mel and chroma bases, wavelets, weightings) stay `f64`
  functions and are rounded once when applied. Tempo, tuning estimates and
  event times stay `f64`. Result structs take a type parameter defaulting to
  `f64` (`PyinResult<F>`, `ReassignedSpectrogram<F>`, `DtwResult<F>`,
  `RqaResult<F>`, `ViterbiResult<F>`, `ViterbiBinaryResult<F>`,
  `FrequencyGrid<'a, F>`); unconstrained `transition_*` calls need `::<f64>`.
  Exceptions that compute in `f64` and round once: LPC's Burg recursion,
  the analysis STFT of `time_stretch` / `pitch_shift`, the polynomial fit of
  `poly_features`, and the resampling inside `pitch_shift` and the CQT.
- **Audio-returning operations return the input's sample type.**
  `AudioSpectrum::istft_matrix`, `AudioConstantQ::icqt`,
  `AudioConstantQ::griffinlim_cqt` and `AudioEffects::griffinlim_audio` return
  `AudioSamples<Self::Sample>` (they returned `AudioSamples<f64>`), and
  `operations::effects::clicks` is generic: `clicks::<T>`. The `AudioEffects`
  methods (`time_stretch`, `pitch_shift`, `preemphasis`, `trim_db`, `remix`,
  `harmonic`, ...) compute in `T::Float` instead of `f64`; `i16` results are
  within 1 LSB of the `f64` path re-quantised.
- **`librosa()` constructors** on every librosa-compatible config return
  librosa 1.0's defaults exactly (`DbConfig::librosa_power()` /
  `librosa_amplitude()` for the dB configs; `YinConfig::librosa(fmin, fmax)` and
  `PyinConfig::librosa(fmin, fmax)`). The golden-fixture tests use them.
- **Defaults that no longer follow librosa,** because librosa's default changes
  the data without being asked:
  - `DbConfig::power()` / `amplitude()` (and so `DbConfig::default()` and
    `MfccConfig::default()`) apply no dynamic-range floor (librosa:
    `top_db=80`).
  - `ChromaStftConfig.tuning` and `ChromaCqtConfig.tuning` (and
    `ChromaCensConfig`, `TonnetzConfig`) default to `Some(0.0)`; `None` still
    estimates the tuning from the signal (librosa's default).
  - `BeatTrackConfig.trim` defaults to `false` (librosa: `True`, which drops
    weak leading and trailing beats).
  - `GriffinLimInit::default()` is `Random(Some(0))`, so repeated calls give
    the same output (librosa: unseeded).
- **Values the caller must give:** `YinConfig` and `PyinConfig` have no
  `Default` (`new(fmin, fmax)`; librosa requires both, the crate used to
  supply C2..C7). `PeakPickConfig` has no `Default`/`new()`; use
  `for_onsets(sample_rate, hop_length)` (the old default hard-coded 22050 Hz
  and hop 512). `PcenConfig` has no `Default` (`new(hop_length)`; it must equal
  the spectrogram's hop). `SilenceConfig` has no `Default` (`new(top_db)`; the
  threshold decides how much `trim_db` removes). `ClicksConfig.hop_length` moved
  into `ClickPositions::Frames { frames, hop_length }`.
- Every other default equals librosa's. The defaults, with the choices a user
  may want to revisit (frame sizes in samples rather than seconds, `n_mels`,
  the mel scale, chroma and tempogram normalisation, tempo priors, CQT range),
  are summarised in `documentation/LIBROSA_MIGRATION.md` under *Defaults that
  differ from librosa*.

### Added

- `AudioFloat`, a sealed trait implemented by `f32` and `f64` that bundles what
  generic analysis code needs (`num_traits::Float` and `FloatConst` through
  `ndarray::NdFloat`, `Sum`, `Product`, `rustfft::FftNum` with `transforms`),
  with `F::of(f64)`, `F::of_usize(usize)` and `x.as_f64()` for conversions.
  `AudioSample` gains the associated type `Float` (the analysis float type of
  the sample type) and `to_float` / `from_float`. Implementations of
  `AudioSample` outside this crate must add them.
- `AudioProcessing::apply_window_f64` / `apply_window_f64_in_place` and
  `apply_filter_f64` / `apply_filter_f64_in_place` take the window or FIR taps
  as `f64` for every sample type, compute in `f64` and return the input's
  sample type (integer results rounded and saturated). For integer audio this
  allows fractional windows and taps outside [-1, 1] (`apply_filter` with `i16`
  taps clamps `[1.5, -0.5]` to `[32767, -16384]`). For `f32` / `f64` audio
  they give the same results as `apply_window` / `apply_filter`.

- `educational::explain` (`educational` feature): explanations computed from
  your settings, before anything runs. `ExplainDoc` holds a summary, the
  formulas the code evaluates, derived numbers, tables, trade-offs and
  warnings, and renders to text (`term-maths`), Markdown (`$$` maths for
  Jupyter) or an HTML fragment. The `Explain` trait is implemented for
  `StftParams`, `WindowType`, `SpectrogramParams`, `MelParams`, `MfccParams`,
  `IirFilterDesign`, `EqBand`, `CompressorConfig`, `LimiterConfig`,
  `VadConfig` and `OnsetDetectionConfig`; `audio.explain(&config)`
  (`ExplainFor`) uses the signal's sample rate and length; `explain::topics`
  covers sampling, aliasing, quantisation, the DFT, windows, YIN,
  autocorrelation pitch, LPC, the cepstrum, resampling and FIR filters.
- Learning guide (`guide` module, `educational` + `full_no_plotting`): a
  14-chapter course mapped to a speech and audio processing syllabus, with
  doctested chapters, a runnable lab per chapter
  (`examples/tutorial_NN_*.rs`) and a page for instructors. The chapter
  sources are plain Markdown with standard LaTeX maths, so they render in
  Markdown viewers and on GitHub; `build.rs` escapes the maths for rustdoc
  (checked by `tests/guide_math_roundtrip.rs` and a pre-commit hook that
  compares the maths in the `cargo doc` HTML with the source).
- [Illustrated book](https://jmg049.github.io/audio_samples/) (`book/`, not published with the crate):
  the guide as an mdBook with KaTeX maths and figures generated by a small crate
  (`book/figures/`) from the same library calls the chapters assert.
  `book/build.sh` builds it.

- `AudioSpectrum` trait (feature `transforms`, `operations::spectrum`): librosa-compatible
  `stft_matrix`, `istft_matrix`, `spectrogram_matrix`, `melspectrogram` and `mfcc_frames` for
  mono and multichannel audio, returning `(channels, features, frames)` arrays. Config structs
  `StftConfig` (with `win_length`, `pad_mode`), `MelSpectrogramConfig`, `MfccConfig` default to
  librosa 1.0's defaults.
- Free functions in `operations::spectrum`: `stft`, `istft` (with `length`), `frame`,
  `power_to_db` / `amplitude_to_db` (`DbConfig`: `ref` value or max, `amin`, `top_db`, per-channel
  axes), `db_to_power`, `db_to_amplitude`, `melspectrogram_from_spectrogram`, `mfcc_from_mel_db`
  (DCT types I-III, ortho/unnormalised, lifter), `delta` (Savitzky-Golay, all edge modes) and
  `stack_memory`.
- `operations::filterbanks` (feature `transforms`): `Window` / `get_window` (the standard
  window families, periodic or symmetric), `window_sumsquare`, `window_bandwidth`, and `mel_filterbank` with
  `MelFilterbankConfig` (Slaney or HTK scale, Slaney / none / Lp normalisation).
- `utils::audio_math`: `MelScale` with `hz_to_mel_with_scale`, `mel_to_hz_with_scale`,
  `mel_frequencies_with_scale`; librosa-compatible `time_to_samples`, `samples_to_frames`,
  `frames_to_samples`, `time_to_frames_floor`, `frames_to_time_with_offset` (floor semantics,
  negative values, optional `n_fft` offset), their `_slice` forms, `times_like`, `samples_like`.
- `utils::audio_math`: `hz_to_octs`, `octs_to_hz`, `a4_to_tuning`, `tuning_to_a4`,
  `cqt_frequencies`, `key_to_notes` and `key_to_degrees` (librosa semantics, Unicode/ASCII
  accidentals, modes, natural signs).
- `AudioFeatures` trait (feature `feature-extraction`, module `operations::features`): librosa 1.0-compatible
  frame-wise features returning `(channels, n, frames)` `f64` arrays, with configs whose `Default` is librosa's:
  `spectral_centroid_frames`, `spectral_bandwidth_frames`, `spectral_rolloff_frames`, `spectral_flatness_frames`,
  `spectral_contrast_frames`, `poly_features`, `rms_frames`, `zero_crossing_rate_frames`, `reassigned_spectrogram`.
- `*_from_spectrogram` free functions (librosa's `S=` path) for the spectral descriptors, and `rms_from_spectrogram`.
- `pcen` / `pcen_with_state` (librosa `pcen`, including `max_size`, `ref`, `zi`/`zf` streaming).
- Frequency weightings `a_weighting`, `b_weighting`, `c_weighting`, `d_weighting`, `z_weighting`,
  `frequency_weighting`, `multi_frequency_weighting` (`FrequencyWeighting`, `min_db` clipping) and
  `perceptual_weighting`.
- `operations::tonal` (feature `feature-extraction`): trait `AudioTonal` with librosa-compatible
  `chroma_stft` (`librosa.feature.chroma_stft`), `piptrack` (`librosa.piptrack`) and
  `estimate_tuning` (`librosa.estimate_tuning`); outputs are `(channels, n, frames)`, tuning
  estimates pool channels. Configs `ChromaStftConfig`, `PiptrackConfig` (+ `PiptrackReference`),
  `TuningConfig`, `ChromaFilterbankConfig`, `NormalizeConfig` default to librosa 1.0 defaults.
- Free functions `chroma_filterbank` (`librosa.filters.chroma`), `chroma_stft_from_spectrogram`,
  `piptrack_from_spectrogram`, `estimate_tuning_from_signal`, `estimate_tuning_from_spectrogram`,
  `pitch_tuning` and `normalize_array` (`librosa.util.normalize`: inf/-inf/0/p norms, axis,
  threshold, fill).
- `operations::constant_q` with trait `AudioConstantQ` (features `transforms` + `resampling`; chroma/tonnetz also need `feature-extraction`): librosa 1.0-compatible `cqt`, `vqt` (ERB-based `gamma` by default), `pseudo_cqt`, `hybrid_cqt`, `icqt`, `griffinlim_cqt` (deterministic `init=None`), `chroma_cqt`, `chroma_cens`, `chroma_vqt`, `tonnetz`, with configs `CqtConfig`, `VqtConfig`, `ChromaCqtConfig`, `ChromaCensConfig`, `ChromaVqtConfig`, `TonnetzConfig`, `CqtMode` (defaults = librosa defaults; `tuning = None` estimates the tuning as librosa does).
- Free functions `cq_to_chroma`, `wavelet`, `wavelet_lengths` (`WaveletConfig`), `chroma_cens_from_chroma`, `tonnetz_from_chroma` (librosa `filters.cq_to_chroma`, `filters.wavelet`, `filters.wavelet_lengths`, and the CENS / tonnetz post-processing).
- `operations::rhythm` (feature `beat-tracking`): trait `AudioRhythm`, a librosa 1.0-compatible
  rhythm pipeline. It provides `onset_strength`, `onset_strength_multi`, `onset_detect`,
  `tempogram`, `fourier_tempogram`, `tempo`, `beat_track` (Ellis dynamic programming, as in
  librosa 1.0), `plp`, `tempogram_ratio` and `hybrid_tempogram`.
  - Free functions: `peak_pick` (librosa `util.peak_pick`, with greedy, `dp_count` and
    `dp_value` methods), `onset_backtrack`, `tempo_frequencies`, `fourier_tempo_frequencies`
    and `metrogram`.
  - Precomputed-input forms: `*_from_db`, `*_from_envelope`, `*_from_tempogram` and
    `beat_track_with_bpm`, which also accepts time-varying tempo.
  - Config structs whose `Default` is librosa's default: `OnsetStrengthConfig`,
    `OnsetDetectConfig`, `PeakPickConfig`, `TempogramConfig`, `FourierTempogramConfig`,
    `TempoConfig`, `BeatTrackConfig`, `PlpConfig`, `TempogramRatioConfig` and
    `HybridTempogramConfig`.
  - Supporting types: `Aggregate`, `TempoLogPrior`, `FrameEvents` and `BeatTrack`.
- `operations::effects` (feature `effects`, now also enabling `sequence`) with the
  `AudioEffects` trait, librosa 1.0-compatible:
  `time_stretch` / `pitch_shift` / `preemphasis` / `deemphasis` / `trim_db` / `remix` /
  `harmonic` / `percussive` (each with an `_in_place` variant), `harmonic_percussive`,
  `split_db`, `lpc` (Burg), and `griffinlim_audio`.
- Free functions in `operations::effects`: `phase_vocoder` / `phase_vocoder_at`
  (librosa 1.0 algorithm), `preemphasis_with_state` / `deemphasis_with_state` (explicit
  `zi` / `zf`), `nonsilent_frames`, `softmask`, `hpss_masks` / `hpss_magnitude` /
  `hpss_spectrogram` (`librosa.decompose.hpss`: kernel sizes, power, margins,
  `reflect`-mode medians), `griffinlim` (fast Griffin-Lim with momentum; deterministic
  `ZeroPhase` or seeded random init), `nn_filter` / `nn_filter_recurrence`, `lpc`, `clicks`.
- Config types `PitchShiftConfig`, `SilenceConfig`, `ChannelAggregate`, `HpssMaskConfig`,
  `HarmonicPercussiveConfig`, `GriffinLimConfig`, `GriffinLimInit`, `NnAggregate`,
  `ClicksConfig`, `ClickPositions` (defaults = librosa defaults), re-exported at the crate root.
- `operations::pitch_tracking` (feature `pitch-analysis`): trait `AudioPitchTracking` with
  librosa-compatible frame-wise `yin(&YinConfig)` (`librosa.yin`) and `pyin(&PyinConfig) -> PyinResult`
  (`librosa.pyin`: f0, voiced flag, voicing probability), outputs `(channels, n_frames)`; array-level
  free functions `yin` / `pyin`; `UnvoicedFill` (librosa's `fill_na`). Config defaults are librosa
  1.0's, with `fmin`/`fmax` defaulting to librosa's recommended C2..C7.
- Free functions `interp_harmonics`, `salience` (`SalienceConfig`, `SalienceAggregate`) and
  `f0_harmonics` (`librosa.interp_harmonics` / `salience` / `f0_harmonics`) on
  `(channels, bins, frames)` spectrograms, with fixed or time-varying `FrequencyGrid`.
- `sequence` feature: `operations::sequence` module with librosa-compatible (1.0.0) sequence tools on
  `ndarray` feature matrices:
  - `dtw`, `dtw_with_cost`, `dtw_backtracking` with `DtwConfig` (metrics euclidean / sqeuclidean /
    cosine / cityblock / chebyshev, custom steps and weights, subsequence alignment, Sakoe-Chiba band)
    and `DtwResult` (accumulated cost, step matrix, warping path in librosa order).
  - `viterbi`, `viterbi_discriminative`, `viterbi_binary` (`bool` states; log-domain HMM decoding, librosa
    tie-breaking, optional `transition_min_prob` pruning).
  - Transition matrix builders `transition_uniform`, `transition_loop`, `transition_cycle`,
    `transition_local` (`TransitionWindow`: triangle, hann, hamming, blackman, boxcar).
  - `rqa` (recurrence quantification analysis) with `RqaConfig`.
  - `recurrence_matrix`, `cross_similarity` (dense `bool` connectivity links, as librosa) and
    `recurrence_matrix_weighted`, `cross_similarity_weighted` (distance / affinity weights with all
    librosa bandwidth heuristics), `recurrence_to_lag`, `lag_to_recurrence`.
  - `sync` / `sync_with` (beat-synchronous aggregation over the last axis; mean, median, max, min,
    sum or a custom reduction).
- `mu_law_encode(&[f64], mu) -> Vec<i64>` and `mu_law_decode(&[i64], mu) -> Vec<f64>` (`processing` feature, crate root): librosa's default quantised mu-law (`mu_compress`/`mu_expand` with `quantize=True`), bit-exact codes (`[-128, 127]` for mu = 255).
- librosa 1.0.0 golden-reference suite: generator `tests/reference/analysis/` (a `uv` project
  pinning librosa, NumPy, SciPy and soxr), fixtures `tests/fixtures/analysis/*.json`, tests
  `tests/analysis_reference/` (spectrum, features, tonal, constant_q, rhythm, effects, pitch,
  sequence, resample / mu-law, autocorrelation, YIN and note names), and `pre-commit` hooks that
  run them.
  - Constant-Q (`constant_q.json`): the non-resampled top octave matches librosa to round-off;
    decimated octaves agree to 1.25e-4 of the peak for the default CQT and <= 3.5e-4 away from
    the signal edges in every configuration (librosa resamples with soxr, the crate with its own
    resampler); edge-dominated
    configurations with deep decimation differ more at the edges (up to 0.28 of the peak for
    `early_downsample`, where librosa itself moves by 0.33 when switched to `res_type="polyphase"`).
  - Rhythm (`rhythm.json`): peak, onset and beat frames and tempi match exactly; envelopes and
    tempograms match to about 1e-14.
- `documentation/LIBROSA_MIGRATION.md`, rendered on docs.rs as the doc-only `librosa_migration`
  module: a librosa 1.0 to audio_samples equivalence table, every documented difference from
  librosa, and how agreement is verified.

- `IirFilterDesign::sos_coefficients`, `IirFilterDesign::frequency_response`,
  `SosFilter::from_sos_coefficients`, `SosFilter::sos_coefficients` and
  `SosFilter::filtfilt_in_place` (the last two return an error for
  non-biquad sections): SOS interchange as `[b0, b1, b2, a0, a1, a2]` rows,
  the common second-order-section layout.
- `BesselNorm` and `IirFilterDesign::with_bessel_norm`: Bessel filters
  normalised for magnitude (-3 dB at the cutoff, the default), phase or group
  delay.
- `IirFilterDesign::butterworth_bandstop`, `IirFilterDesign::chebyshev_i_band`,
  `EqBand::band_pass`, `EqBand::band_stop`.
- Golden-reference filter test suite (`tests/filter_reference.rs`): 594 IIR
  designs checked coefficient by coefficient, time-domain checks of causal and
  zero-phase filtering, and 512 RBJ biquads checked against the Audio EQ
  Cookbook formulas. Fixtures are regenerated by
  `tests/reference/generate_filter_fixtures.py`.

- `StreamingResampler` (`resampling` feature): a stateful, chunk-by-chunk resampler for continuous sources (e.g. audio being decoded live and fed to a playback device at a different sample rate). Uses the same delay-free windowed-sinc engine as `resample`/`resample_by_ratio` (no delay, exact total length, output equal to `resample` of the whole input), but keeps its filter state and any leftover input alive across calls instead of requiring the whole clip up front.

- `examples/librosa_bench/`: wall-clock benchmark of the librosa-compatible operations (JSON
  output), driven by the comparison harness in `benches/librosa_compare/`.

- `PeakPickConfig::new()` (same as `Default`).

### Deprecated

- `AudioProcessing::apply_window` / `apply_window_in_place`: the window has the
  sample type and the product is computed in that type, so integer audio can
  only use the coefficients 0 and 1, and larger values overflow. Use
  `apply_window_f64`. Behaviour is unchanged.

### Fixed

- **`VadMethod::Spectral` works.** It returned `AudioSampleError::Unsupported`. A frame is now
  speech when its RMS level reaches `energy_threshold_db` and at least `spectral_ratio_threshold`
  of its energy (Hann window, real FFT of length `frame_size`) lies between `speech_band_low_hz`
  and `speech_band_high_hz` (defaults: 0.6 of the energy in 300-3400 Hz). It rejects loud
  broadband noise, hum and hiss that the energy method accepts. A band above the Nyquist
  frequency, or one with no FFT bin at the frame size, is an `AudioSampleError::Parameter`. The
  `vad` feature now depends on `realfft`.
- **No dead-code warnings on partial feature sets.** Building with `iir-filtering` but not
  `pitch-filterbank` warned that `bandpass_order` was never used, and building with `sequence`
  but not `effects` warned about the recurrence-link helpers. Each is now compiled only with the
  feature that uses it.

- **IIR high-pass filters of order ≠ 2 were mirrored around fs/4.** The
  Butterworth and Chebyshev I high-pass designs were derived from the
  low-pass with `z → -z`, which maps the cutoff `fc` to `fs/2 - fc`: a
  "4th-order 100 Hz high-pass" at 48 kHz attenuated everything by > 120 dB.
  All families and responses now use one analog-prototype pipeline: prototype
  zeros, poles and gain, the low-pass to low/high/band transformation, the
  bilinear transform, then pairing into second-order sections.
- **Odd-order Butterworth / Chebyshev I filters had a doubled real pole**,
  i.e. an order-`N` request produced a wrong order-`N+1` filter (-6 dB instead
  of -3 dB at the cutoff).
- Chebyshev I filters were renormalised to 0 dB at DC (even orders) or the
  band centre, shifting the passband up by the ripple; the passband now
  ripples between 0 and -`rp` dB, as the Chebyshev I definition requires. Butterworth /
  Chebyshev II / Bessel band-pass filters were normalised at the un-warped
  geometric centre, leaving a small gain error.
- `filtfilt` uses steady-state initial conditions for each pass and pads by
  `3 · (2·n_sections + 1)` samples (less for first-order sections);
  previously the edges were off by up to ~6e-3.
- Bessel prototypes are accurate to f64 precision up to order 24 (the
  Durand-Kerner root finder lost accuracy beyond order ~12).
- `low_pass_filter`, `high_pass_filter` and `band_pass_filter` were not
  real first-order filters: the low-pass used `α = 2π·fc/fs` and became
  unstable (output → ∞) for cutoffs above ~fs/π (e.g. 20 kHz at 48 kHz); the
  high-pass computed in the sample type, so for integer audio (e.g. 16-bit WAV)
  its coefficient truncated to zero and it output silence. They are now
  first-order Butterworth filters, and `band_pass_filter` is a first-order
  Butterworth band-pass between its two edges.
- RBJ low/high shelf EQ bands used `sqrt(2A)·α` instead of the cookbook's
  `2·sqrt(A)·α`, effectively scaling Q by √2 (up to ~0.8 dB error).
- `apply_filter` (FIR) computed a correlation rather than a convolution, so
  asymmetric kernels were applied time-reversed. It now computes the
  convolution over the positions where the kernel lies wholly inside the
  signal ("valid" mode), accumulates in `f64`, and no longer overflows for
  integer samples.
- EQ band validation rejects NaN frequency / Q / gain.
- Numerically degenerate IIR specifications (e.g. a ripple so small that
  `10^(rp/10) - 1` underflows) return a `Parameter` error instead of
  panicking or producing NaN / all-zero output, and edges within ~1e-5 of
  Nyquist no longer overflow the gain computation.
- **Float → `i16`/`i32` sample conversion was not the inverse of
  `i16`/`i32` → float**: it scaled negatives by `MAX` instead of `-MIN` and
  truncated instead of rounding, so every negative sample moved one LSB
  towards zero on an int → float → int round trip and any processing of
  16-bit audio (filters, EQ, gain) added a half-LSB bias. Conversion now
  scales asymmetrically and rounds half away from zero (as `I24` and `u8`
  already did): `-1.0 → i16::MIN`, and int → float → int is the identity for
  every value. The `simd` conversion paths match bit-for-bit.
- YIN (`detect_pitch_yin`, `track_pitch` with `PitchDetectionMethod::Yin`, `yin_pitch_detection`): the cumulative-mean normalised difference now includes every lag from 1 (lags below the `max_frequency` limit were zero, inflating the CMND); the first sub-threshold dip is followed to its local minimum (a ±5-lag window stopped on the flank of low-pitch dips, up to +53 cents at 82 Hz); and the period is refined by parabolic interpolation (it was an integer: 440 Hz read as 441 Hz). Errors on a 110-880 Hz glide drop from 8-17 cents median / 32-72 max to 0.8-2.7 / 2.4-4.8 cents.
- `estimate_key` rotated the key profiles the wrong way, reporting every non-C key as its mirror image (D major as A# major).
- `frequency_to_note` labelled every frequency below ~7.9 Hz as "C-1" (saturating cast); it now returns `ParameterError::OutOfRange` outside MIDI `0..=127`, as above G9.
- `AudioStatistics::autocorrelation` divided by the FFT size twice, so values were too small by the zero-padded FFT length; it now returns the documented unbiased estimate (`r[k] * (n - k)` is the raw lag sum `Σ x[j] x[j+k]`).
- `complex_onset_detection` / `onset_detection_function_complex` panicked (`unreachable!`) for every preset on any input long enough for two CQT frames.
- `spectral_centroid`, `spectral_rolloff`, `spectral_bandwidth` and `spectral_slope` used a wrong frequency axis for odd-length signals (bin spacing sr/(n±1) instead of sr/n).
- `zero_crossing_rate` on multichannel audio summed the crossings of all channels but divided by one channel's duration (two identical channels gave twice the mono rate); it is now the mean of the per-channel rates.
- Opus codec: `levinson_durbin` returned the order p-1 LPC solution padded with a stale coefficient instead of the order-p solution.
- `mu_compress` used `ln(mu|x| + ln(1+mu))` instead of `ln(1 + mu|x|)` (0 mapped to 0.309, ±1 to ±1.003); it now computes `sign(x) ln(1 + mu|x|) / ln(1 + mu)`. `mu_compress`/`mu_expand` on integer sample types rescaled `mu` like a sample (255 became 255/32768); the raw numeric value is now used. Both reject non-finite or non-positive `mu`.
- `trim_all_silence` discarded the audio before every silent gap shorter than the minimum (a 1 s tone came back as 24 samples); it now removes exactly the silent runs at least `min_silence_duration_seconds` long.
- Resampling (`resample`, `resample_by_ratio`, `AudioProcessing::resample*`): Medium/High output was delayed by the filter delay (64/128 input samples) and lost the tail; `Fast` downsampling was plain decimation with no anti-aliasing. All sinc paths now use a delay-free Kaiser-windowed sinc interpolator at exact rational positions (f64), with exact output length `ceil(n * out / in)`; `High` has a passband to 0.913 of the lower Nyquist frequency and 125 dB stopband attenuation. `Fast` downsampling is anti-aliased; `Fast` upsampling is unchanged (linear interpolation).
- `to_mono(Average)` averaged stereo f64/i32 in f32 (precision loss) and summed >2 integer channels in the sample type (overflow); `to_stereo(Left/Right)` on **mono** input returned mono / errored; it now gives `[m, 0]` / `[0, m]`. On multi-channel input `Left`/`Right` still extract channel 0 / 1 as a mono signal, exactly as in 2.0 (unchanged; use `extract_channel` for clarity).
- Pitch-shift perturbation (`PerturbationMethod::PitchShift`): wrong instantaneous-frequency estimate (no bin de-trending, sign-preserving `%` wrap), incoherent phases for colliding bins, output 68 samples short, and multichannel input failed with a shape error. A 440 Hz tone shifted +2 st now peaks at 494.0 Hz (was 505.7 Hz).
- `pad_samples_right` could return one sample fewer than requested (samples -> seconds -> samples truncation).

- Resampling: the per-thread filter-table cache is now a bounded LRU (at most 8 tables / 64 MiB per
  thread). It previously grew without limit, by up to tens of MB per distinct rate pair (e.g. pitch
  shifting by arbitrary steps).

- **Resampling: excessive memory use for steep decimation.** Strongly decimating ratios built a very large
  interpolated filter table (48000 -> 101 Hz at `High`: 2.8 GB and 12 s, kept in the per-thread cache;
  192000 -> 101 Hz aborted the process). The interpolated table now uses `ceil(4096 * L/M)` rows per input
  sample when downsampling (same interpolation accuracy), kernels larger than the 64 MiB cache budget are
  not retained, and ratios that would need more than 2^25 coefficients return a `ParameterError` from
  `resample`, `resample_by_ratio` and `StreamingResampler::new` instead of aborting.
- **Non-finite audio is rejected by the librosa-compatible operations** (`AudioSpectrum`, `AudioFeatures`,
  `AudioTonal`, `AudioConstantQ`, `AudioRhythm`, `AudioEffects`) with a `ParameterError`; previously a
  single NaN produced meaningless output without an error.
- **Extreme parameters no longer abort the process**: `mu_law_encode` (very large `mu`; `mu > 2^53` is an error),
  `pyin` (very small `resolution`), `tempo`/tempograms (very large `ac_size`/`win_length`), `estimate_tuning`/
  `pitch_tuning` (very small `resolution`), `clicks` and `time_stretch` (very small `rate`) validate derived buffer sizes.
- `peak_pick` no longer overflows with very large window/`wait` parameters.
- `PeakPickingConfig::set_min_peak_separation_ms` no longer panics when the separation rounds to zero
  frames (or is NaN / negative); it clamps to one frame.
- `resample_by_ratio` rejects ratios whose target rate exceeds `u32::MAX` Hz, and non-finite ratios.
- **`resample` and `resample_by_ratio` with `ResamplingQuality::Fast` upsampling
  interpolate `i32` and `f64` audio in `f64`**, as every other resampling path
  does; they converted to `f32` first, so output was only accurate to about
  1.2e-7 of full scale. `i32` and `f64` output changes by up to that amount;
  `u8`, `i16`, `I24` and `f32` output is unchanged. Fast upsampling now reads
  the input in place and writes the output directly, which makes it several
  times faster.

- Size-driven building blocks return `ParameterError` instead of aborting on a failed allocation (or
  panicking with "capacity overflow"): `get_window`, `window_bandwidth`, `window_sumsquare`,
  `mel_filterbank`, `stft`, `istft`, `frame`, `stack_memory`, `transition_*` (`n_states^2 > 2^28`),
  `chroma_filterbank`, `cq_to_chroma`, `wavelet`, `beat_track_with_bpm`.
- The maximum filters behind `onset_strength(max_size=)` and `pcen(max_size=)` scan at most one padding
  period, so a very large `max_size` no longer runs for hours (results unchanged).
- `clicks` uses saturating arithmetic when placing clicks.


### Fixed (correctness for teaching use)

Each fix below has a regression test (`tests/*_regressions.rs`). Several
change numerical output so that it follows the documented definition. Related
fixes are listed under *Fixed* above: YIN, `estimate_key`, `autocorrelation`
scaling, `levinson_durbin`, the resampler, `complex_onset_detection`, the
odd-length spectral bin spacing, `frequency_to_note` and
`mu_compress`/`mu_expand`.

- **Dynamics:** the envelope follower used the release time almost
  everywhere (a 1 ms attack reached 3 % instead of 63 % after 1 ms), so every
  compressor, limiter, gate, expander and `attack_decay_envelope` was too slow
  to react. Lookahead had no effect. The soft knee was non-monotone (now the
  Giannoulis/Massberg/Reiss quadratic knee); the soft limiter overshot its
  ceiling statically; gate ratio now means a 1:R downward expander;
  `set_detection_method` resizes the RMS window. `LimiterConfig::isp_limiting`
  is documented as having no effect yet.
- **Statistics:** `mean`/`variance`/`cross_correlation` wrapped or panicked
  on integer audio (also breaking `remove_dc_offset`); multi-channel
  `variance` now pools all samples; u8 zero crossings are counted about 128;
  `align_signals` never searched any lag; `detect_fundamental_frequency`
  returned sub-harmonics; `detect_sample_rate` tested the wrong bins and now
  returns `Ok(None)` when no cutoff is found.
- **Processing:** gains, peak/min-max normalisation, fades and `mix` on
  integer audio truncated the gain to an integer (silence on i16); median
  normalisation subtracted the middle sample, not the median; `GaussianNoise`
  perturbation added uniform noise at the wrong SNR (and pink was white);
  `am_signal` depth now follows the standard AM definition.
- **Spectral:** `power_spectral_density` now has Welch density scaling
  (power per Hz; it was off by N·Σw²/2); spectrograms, mel, MFCC, chroma, gammatone
  and CQT of integer audio used a raw cast (i16 power 2³⁰ too large);
  `HpssConfig::num_freq_bins` for odd FFT sizes; spectrum/phase plot axes and
  spectrogram plot sample rate.
- **Rhythm:** the beat grid drifted; onset envelope smoothing used a sample
  count as frames; `pick_peaks` ignored the adjusted threshold window.
- **VAD:** a signal shorter than one frame with `pad_end = false` returns an
  error (was undefined behaviour); `VadMethod::Spectral` returns
  `Unsupported` (it is not implemented).
- **Documentation:** about 150 formulas and claims corrected so that the docs
  state what the code computes (conventions, normalisation, units, dB
  references), with the conventions that vary between references stated
  explicitly (symmetric Hann, Slaney mel on power, unnormalised DCT-II with
  lifter 22, absolute dB floor, HTK `audio_math` mel helpers, ZCR in crossings
  per second). The perceptual
  codec docs were scientifically corrected and the Opus module is described
  as an Opus-inspired teaching codec, not bitstream compatible.

### Changed

- **Onset peak-picking presets are specified in seconds and take the sample rate.**
  `PeakPickingConfig::music`, `speech` and `drums` used lengths sized as audio samples (a
  2048-frame threshold window and a 1024-frame separation for `music`), which on an onset
  strength function span many seconds, so they found almost nothing. They now take
  `(sample_rate, hop_size)` and convert fixed durations to frames: threshold windows of 0.2 s
  (`music`, `speech`) and 0.1 s (`drums`), minimum separations of 0.1 s, 80 ms and 30 ms. Like
  the default, each floors the threshold at 0.1 of the normalised strength and uses no median
  filter. The onset presets that use them take the sample rate and build the peak picker for
  their own hop: `OnsetDetectionConfig::{percussive, musical, speech}(sample_rate)`,
  `SpectralFluxConfig::{percussive, musical}(sample_rate)` and
  `ComplexOnsetConfig::{percussive, musical, speech}(sample_rate)`. Pass the rate of the audio
  the configuration is used on, e.g. `OnsetDetectionConfig::percussive(audio.sample_rate())`.
- **`AudioSamples`' `Display` output is one line, and `{:#}` previews the channels.** `{}`
  prints `AudioSamples<f32>: 2 ch × 48000 samples @ 48000 Hz (1.000 s)` (sample type, channels,
  samples per channel, sample rate, duration) with no trailing newline and no samples. `{:#}`
  adds one line per channel, up to eight, with its first and last three samples (all of them
  when there are at most six). It used to print `x` instead of `×`, end the header with a
  newline, add a sample preview to `{}`, and lay `{:#}` out differently for mono and
  multi-channel audio. The sample type is the crate's label (`I24`, not `i24::I24`).
- **Setters and lookups that panicked on bad arguments return errors.**
  `HpssConfig::set_stft_params` returns `AudioSampleResult<()>` and rejects a
  `hop_size` larger than `n_fft` with `AudioSampleError::Parameter`, leaving the
  configuration unchanged. `PitchClass::from_index` returns
  `AudioSampleResult<PitchClass>` (an index above 11 is an error);
  `PitchClass::ALL` lists the twelve classes in order.
  `DistanceMetric::distance` returns `AudioSampleResult<F>` (vectors of
  different lengths are an error). `SampleType::from_bits` returns
  `Option<SampleType>` (`None` for a width no sample type has).
- **`AudioSamples::apply_to_channel_data` accepts any memory layout.** Channels
  of non-contiguous multi-channel storage are copied into a buffer, passed to
  the callback and written back; `try_apply_to_channel_data` still reports
  such storage as an error.
- **Waveform and spectrogram plots with `ChannelManagementStrategy::Separate`
  (the default) reject audio with more than `MAX_SEPARATE_CHANNELS` (8)
  channels** with `AudioSampleError::Parameter`; they panicked.

- **`f64` decimal logarithms are correctly rounded.** `power_to_db`,
  `amplitude_to_db`, `perceptual_weighting`, the `A`/`B`/`C`/`D` frequency
  weightings (`frequency_weighting`, `multi_frequency_weighting`) and the
  features built on them (`mfcc`, `spectral_contrast`) take `log10` of `f64`
  values with a vectorised kernel instead of the C library's `log10`. Each
  logarithm is the correctly rounded value except within about 2^-70 of a
  rounding boundary, and differs from glibc's (which is not correctly rounded)
  by at most 1 ulp, toward the exact result. Output values change by at most
  2 ulp for the dB conversions and at most 8e-14 dB for the weightings. Results are the same on machines
  with and without FMA.

- Filter parameter validation accepts any edge in `(0, Nyquist)` (the previous Chebyshev/elliptic `0.95·Nyquist` limit is gone), any
  positive ripple is accepted (was ≤ 5 dB), and the maximum order is 24
  (was 12).
- `processing` now enables `iir-filtering` (the simple filters are built on it).
- A multi-band `ParametricEq` is applied as one SOS cascade in a single pass,
  with the output gain folded in.
- `AudioIirFiltering::frequency_response` is deprecated: it never had a filter
  to describe and always returned a flat response.
- `PeakPickingConfig::default()` (used by `OnsetDetectionConfig::default()`, `ComplexOnsetConfig::default()` and `SpectralFluxConfig::complex()`) now uses a 9-frame delta threshold (delta 0.07) with an absolute floor of 0.1, a 3-frame minimum separation and no median filter. The old defaults (1024-frame threshold window, 3-point median filter, 512-frame separation) found about 10% of onsets; `detect_onsets` with the default config now finds all 10 notes of a test sequence (was 1).
- `note_to_midi` (and so `note_to_frequency`) accepts a wider note grammar: lowercase letters, any run of `#` `♯` `b` `!` `♭` `𝄪` `𝄫` `♮` accidentals (B#3 = 60, Cb4 = 59, double accidentals), signed octaves (`C-1` = 0) and a cents suffix rounded half to even (`A4+50` = 70). The octave remains required and the result remains a `u8` in `0..=127`.
- `BeatTrackingData::beat_times` / `track_beats_core` are now sorted chronologically and de-duplicated (they were in detection order). `BeatTrackingData::tempo_bpm` is documented as the requested tempo (it is not estimated); its `Display` prints "Tempo:".
- `resampling` feature: Medium/High/Fast downsampling take 2-3x longer than before for the same ratio, because their anti-aliasing filters are longer and now correct.
- The `pitch-analysis` feature now also enables `sequence` (pYIN's Viterbi decoding).
- `realfft` is now a direct optional dependency of the `transforms` feature (it was already a
  transitive dependency via `spectrograms`).
- Documentation: `hz_to_mel`, `mel_to_hz`, `mel_scale` and `mel_frequencies` state that they use
  the HTK scale; `time_to_frames` documents its rounding; the two `seconds_to_samples` functions
  document their different rounding.

- Performance (results unchanged up to floating-point round-off):
  - `stft`: blocked output transpose, no whole-signal scratch buffer (~2x faster).
  - Spectrogram magnitudes use `re² + im²` instead of `hypot`; mel products visit only each
    filter's non-zero bins (melspectrogram ~2.8x, mfcc ~2.4x faster).
  - Whole-buffer resampling with small rate ratios (2:1, 1:2, 3:1, 3:2, ...) evaluates the same
    polyphase filter with FFTs (44100 -> 22050 Hz ~4x faster; `cqt` ~3x faster).
  - `spectral_contrast` uses partial selection for the band quantiles (~2.4x faster, bit-identical).
  - Phase vocoder (`time_stretch`, `pitch_shift`) computes each frame's phase and magnitude once
    (~1.45x / ~1.5x faster, bit-identical).
  - Fewer whole-signal copies in `AudioEffects` methods returning audio.
- The `resampling` feature depends on `realfft` directly.

- **Resampling / `pitch_shift`: filter tables are shared and cheap to build.** Interpolated (awkward-ratio)
  tables depend only on quality and relative bandwidth, so all upsampling ratios of one quality share one
  table, sampled from a per-quality prototype, so `pitch_shift` with a new `n_steps` no longer builds a
  table of its own and a sweep over many distinct shifts runs about five times faster. Common (small-L) ratios are
  bit-identical; awkward ratios change below the table-interpolation error (< -120 dB).

### Known issues

- Upstream limitations of the `spectrograms` crate (2.1.3) that this crate
  inherits:
  - `WindowType::Kaiser` is inaccurate for β > 3.75.
  - The gammatone filterbank's default centre frequencies are spaced linearly
    in Hz, not on the ERB-number scale.
  - A mel spectrogram's `frequencies()` axis ignores `f_min`/`f_max`; it is
    correct only for the range 0 Hz to Nyquist.

## 2.0.0

A breaking release that unifies the public API. See
[documentation/MIGRATING_TO_2.0.md](documentation/MIGRATING_TO_2.0.md) for
upgrade steps.

### Added

- IIR filters: Chebyshev II, Elliptic (Cauer), and Bessel designs, plus
  band-stop responses and a corrected band-pass transform, applied through
  `apply_iir_filter`, checked against reference designs.
- Zero-phase `filtfilt` and a design-once streaming `SosFilter`
  (`from_design` / `process_block`).
- Spectral feature suite on `AudioStatistics`: bandwidth, flatness, contrast,
  slope, and crest, alongside the existing centroid and rolloff.
- Tempo estimation (`estimate_tempo`) on `AudioBeatTracking`.
- Generators: `exponential_chirp`, band-limited square/sawtooth/triangle
  oscillators, and `fm_signal`.
- Comparison metrics: `psnr`, `segmental_snr`, `log_spectral_distance`, and
  per-channel `correlation`/`mse`/`snr`.
- Plots: phase spectrum and Lissajous (stereo X-Y).
- `windows_ref()`: a borrowing, zero-copy window iterator.
- Educational explanations for the dynamics, EQ, and resampling operations.
- Benchmarks for conversions, transforms, generation, and dynamics.

### Changed

- `AudioSamples` fields are private; use `data()`, `data_mut()`,
  `into_data()`, `into_data_borrowed()`, and `sample_rate()`.
- Every transforming operation now has a borrowing `op(&self) -> Result<Self>`
  (the unsuffixed name) and an `op_in_place(&mut self) -> Result<()>`. Methods
  that previously consumed `self` now borrow; methods that previously mutated
  `&mut self` are renamed with the `_in_place` suffix.
- Channel indices are `usize` (were `u32`); counts remain `ChannelCount`.
- `power_spectral_density` returns `Psd`; `track_pitch` returns `PitchContour`;
  `estimate_key` returns `Key { tonic, mode, confidence }`.
- `apply_gate`, `apply_expander`, and `apply_three_band_eq` take config structs
  (`GateConfig`, `ExpanderConfig`, `ThreeBandEqConfig`).
- `spectral_centroid` and `spectral_rolloff` take a `ChannelReduction` argument.
- `median` renamed to `midpoint_sample`; `AudioTypeConversion::as_float`
  removed (use `as_f64`).
- `SampleType: TryFrom<&str>` errors with `EnumParseError` (was `()`), and gains
  `FromStr`.
- `EnvelopeFollower::process` no longer takes a per-call detection method; it
  uses the one set at construction.
- Minimum supported Rust version is 1.87. Requires `spectrograms` 1.4.4.

### Fixed

- Peak/notch EQ gain used base `10.04` instead of `10.0`.
- The non-rectified complex onset detection function read the phase buffer as
  magnitude.
- `compound_tone` applied `sin` to the raw time value rather than the full
  phase argument.
- Seeded `brown_noise` returned a single sample regardless of duration.
- `detect_dynamic_range` computed RMS from the sample mean instead of the mean
  of squares.
- `from_interleaved_slice` did not deinterleave multi-channel input.
- `swap_channels` transposed the array instead of swapping the two rows.
- `apply_eq_band` corrupted multi-channel input.
- YIN's backward neighbourhood search was disabled by a shadowed variable.
- Multi-channel `trim_silence` compared signed values rather than magnitude.

### Performance

- Thread-local FFT planner reuse across the transforms and statistics paths,
  and in FFT convolution.
- Hoisted per-frame and per-window allocations in the iterators, pitch
  tracking, and HPSS median filtering.
- Vectorized sample conversions (bit-identical to the scalar path) where the
  `simd` feature is enabled, with a scalar fallback elsewhere.
- Compressor and limiter use a lookahead-sized ring buffer; the complex onset
  detection function computes its CQT once.
