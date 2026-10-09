# Changelog

All notable changes to this project are documented here.

## [Unreleased]

### Fixed

- **IIR high-pass filters of order ≠ 2 were mirrored around fs/4.** The
  Butterworth and Chebyshev I high-pass designs were derived from the
  low-pass with `z → -z`, which maps the cutoff `fc` to `fs/2 - fc`: a
  "4th-order 100 Hz high-pass" at 48 kHz attenuated everything by > 120 dB.
  All families and responses now use one analog-prototype pipeline that is a
  line-by-line port of SciPy's (`iirfilter` → `*ap` → `lp2*_zpk` →
  `bilinear_zpk` → `zpk2sos`).
- **Odd-order Butterworth / Chebyshev I filters had a doubled real pole**,
  i.e. an order-`N` request produced a wrong order-`N+1` filter (-6 dB instead
  of -3 dB at the cutoff).
- Chebyshev I filters were renormalised to 0 dB at DC (even orders) or the
  band centre, shifting the passband up by the ripple; the passband now
  ripples between 0 and -`rp` dB exactly as in SciPy/MATLAB. Butterworth /
  Chebyshev II / Bessel band-pass filters were normalised at the un-warped
  geometric centre, leaving a small gain error.
- `filtfilt` now matches `scipy.signal.sosfiltfilt` exactly (steady-state
  initial conditions per pass and SciPy's pad length); previously the edges
  differed by up to ~6e-3.
- Bessel prototypes are accurate to f64 precision up to order 24 (the
  Durand-Kerner root finder lost accuracy beyond order ~12).
- `low_pass_filter`, `high_pass_filter` and `band_pass_filter` were not
  real first-order filters: the low-pass used `α = 2π·fc/fs` and became
  unstable (output → ∞) for cutoffs above ~fs/π (e.g. 20 kHz at 48 kHz); the
  high-pass computed in the sample type, so for integer audio (e.g. 16-bit WAV)
  its coefficient truncated to zero and it output silence. They are now
  first-order Butterworth filters (`scipy.signal.butter(1, …)`), and
  `band_pass_filter` is `butter(1, [low, high], 'bandpass')`.
- RBJ low/high shelf EQ bands used `sqrt(2A)·α` instead of the cookbook's
  `2·sqrt(A)·α`, effectively scaling Q by √2 (up to ~0.8 dB error).
- `apply_filter` (FIR) computed a correlation rather than a convolution, so
  asymmetric kernels were applied time-reversed. It now equals
  `numpy.convolve(x, h, 'valid')`, accumulates in `f64`, and no longer
  overflows for integer samples.
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


### Fixed (correctness audit for teaching use)

A formula-by-formula audit checked the documentation against the code and
against SciPy/librosa/closed-form results. Each fix below has a regression
test (`tests/audit_*.rs`). Several change numerical output; they now match
their documented definitions.

- **Dynamics:** the envelope follower used the release time almost
  everywhere (a 1 ms attack reached 3 % instead of 63 % after 1 ms), so every
  compressor, limiter, gate, expander and `attack_decay_envelope` was too slow
  to react. Lookahead had no effect. The soft knee was non-monotone (now the
  Giannoulis/Massberg/Reiss quadratic knee); the soft limiter overshot its
  ceiling statically; gate ratio now means a 1:R downward expander;
  `set_detection_method` resizes the RMS window. `LimiterConfig::isp_limiting`
  is documented as having no effect yet.
- **Statistics:** `autocorrelation` was divided by the FFT size twice;
  `mean`/`variance`/`cross_correlation` wrapped or panicked on integer audio
  (also breaking `remove_dc_offset`); multi-channel `variance` now pools all
  samples; u8 zero crossings are counted about 128; spectral centroid,
  rolloff, bandwidth and slope used the wrong bin spacing for odd lengths;
  `align_signals` never searched any lag; `detect_fundamental_frequency`
  returned sub-harmonics; `detect_sample_rate` tested the wrong bins and now
  returns `Ok(None)` when no cutoff is found.
- **Processing:** gains, peak/min-max normalisation, fades and `mix` on
  integer audio truncated the gain to an integer (silence on i16);
  `mu_compress`/`mu_expand` rescaled μ like a sample; median normalisation
  subtracted the middle sample, not the median; `GaussianNoise` perturbation
  added uniform noise at the wrong SNR (and pink was white); `am_signal`
  depth now follows the standard AM definition; Medium/High resampling is
  time-aligned (the sinc delay of ~69/139 samples at 44.1→48 kHz is
  compensated and the tail is kept).
- **Spectral:** `power_spectral_density` now has SciPy `welch` density
  scaling (it was off by N·Σw²/2); spectrograms, mel, MFCC, chroma, gammatone
  and CQT of integer audio used a raw cast (i16 power 2³⁰ too large);
  `HpssConfig::num_freq_bins` for odd FFT sizes; spectrum/phase plot axes and
  spectrogram plot sample rate; `frequency_to_note` errors outside MIDI 0–127.
- **LPC:** `levinson_durbin` returned the order-(p−1) solution padded with a
  zero; it now returns the order-p predictor (LPC/SILK output changes).
- **Pitch and rhythm:** YIN now follows de Cheveigné & Kawahara with
  parabolic interpolation (440 Hz read as 450 Hz at 44.1 kHz, and +3.9 % at
  100 Hz); `estimate_key` rotated its profile the wrong way (D major reported
  as A# major); `complex_onset_detection` panicked on every input; the beat
  grid drifted; onset envelope smoothing used a sample count as frames;
  `pick_peaks` ignored the adjusted threshold window.
- **VAD:** a signal shorter than one frame with `pad_end = false` returns an
  error (was undefined behaviour); `VadMethod::Spectral` returns
  `Unsupported` (it is not implemented).
- **Documentation:** about 150 formulas and claims corrected so that the docs
  state what the code computes (conventions, normalisation, units, dB
  references), with librosa/SciPy differences called out (symmetric Hann,
  Slaney mel on power, unnormalised DCT-II with lifter 22, absolute dB floor,
  HTK `audio_math` mel helpers, ZCR in crossings per second). The perceptual
  codec docs were scientifically corrected and the Opus module is described
  as an Opus-inspired teaching codec, not bitstream compatible.

### Changed

- Filter parameter validation follows SciPy: any edge in `(0, Nyquist)` is
  accepted (the previous Chebyshev/elliptic `0.95·Nyquist` limit is gone), any
  positive ripple is accepted (was ≤ 5 dB), and the maximum order is 24
  (was 12).
- `processing` now enables `iir-filtering` (the simple filters are built on it).
- A multi-band `ParametricEq` is applied as one SOS cascade in a single pass,
  with the output gain folded in.
- `AudioIirFiltering::frequency_response` is deprecated: it never had a filter
  to describe and always returned a flat response.

### Added

- `IirFilterDesign::sos_coefficients`, `IirFilterDesign::frequency_response`,
  `SosFilter::from_sos_coefficients`, `SosFilter::sos_coefficients` and
  `SosFilter::filtfilt_in_place` (the last two return an error for
  non-biquad sections): SciPy/MATLAB-layout SOS interchange
  (`[b0, b1, b2, a0, a1, a2]` rows).
- `BesselNorm` and `IirFilterDesign::with_bessel_norm` (SciPy's `norm=`
  `'mag'` (default here), `'phase'` (SciPy's default) and `'delay'`).
- `IirFilterDesign::butterworth_bandstop`, `IirFilterDesign::chebyshev_i_band`,
  `EqBand::band_pass`, `EqBand::band_stop`.
- Golden-reference filter test suite (`tests/filter_reference.rs`): 594 IIR
  designs checked coefficient-by-coefficient against SciPy and against
  DSPFilters and liquid-dsp, time-domain checks against `sosfilt` /
  `sosfiltfilt`, and 512 RBJ biquads checked against the cookbook and FFmpeg.
  Fixtures are regenerated by `tests/reference/generate_filter_fixtures.py`.

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
  (`examples/tutorial_NN_*.rs`) and a page for instructors.
- `StreamingResampler::output_delay()`.

- `StreamingResampler` (`resampling` feature): a stateful, chunk-by-chunk resampler for continuous sources (e.g. audio being decoded live and fed to a playback device at a different sample rate). Wraps the same rubato `Async<f32>` engine and quality tuning as `resample`/`resample_by_ratio`, but keeps the delay line and any leftover input alive across calls instead of requiring the whole clip up front.

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
