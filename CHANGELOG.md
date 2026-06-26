# Changelog

All notable changes to this project are documented here.

## 2.0.0

A breaking release that unifies the public API. See
[documentation/MIGRATING_TO_2.0.md](documentation/MIGRATING_TO_2.0.md) for
upgrade steps.

### Added

- IIR filters: Chebyshev II, Elliptic (Cauer), and Bessel designs, plus
  band-stop responses and a corrected band-pass transform, applied through
  `apply_iir_filter`. Cross-checked against `scipy.signal`.
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
