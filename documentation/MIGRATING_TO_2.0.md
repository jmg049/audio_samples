# Migrating to audio_samples 2.0

audio_samples 2.0 is a clean break (no deprecation shims) that unifies the
public API into a single convention. The headline change is that every
transforming operation now provides two consistently named variants: an
unmarked, non-mutating form that borrows `&self` and returns a new value, and
an `_in_place` form that takes `&mut self` and mutates in place. Alongside that,
`AudioSamples` fields are now private (use accessors), channel indices are
`usize`, several multi-parameter operations take config structs instead of
positional arguments, and a handful of operations return structured types
(`Psd`, `PitchContour`, `Key`) instead of bare tuples. This guide lists every
breaking change with a before/after snippet and the reason, followed by the new
requirements and the additive features you may want to adopt.

## Contents

1. [Sealed fields on AudioSamples](#1-sealed-fields-on-audiosamples)
2. [Dual in-place / borrowing operation variants](#2-dual-in-place--borrowing-operation-variants)
3. [Channel indices are now usize](#3-channel-indices-are-now-usize)
4. [Config structs replace positional arguments](#4-config-structs-replace-positional-arguments)
5. [Structured return types](#5-structured-return-types)
6. [ChannelReduction on spectral_centroid and spectral_rolloff](#6-channelreduction-on-spectral_centroid-and-spectral_rolloff)
7. [Renames and removals](#7-renames-and-removals)
8. [Explicitly NOT changed](#8-explicitly-not-changed)
9. [Requirements](#9-requirements)
10. [New in 2.0 (additive, no action needed)](#10-new-in-20-additive-no-action-needed)

---

## 1. Sealed fields on AudioSamples

Why: the `data` and `sample_rate` fields are now private so the internal
copy-on-write representation can evolve without breaking callers. Access them
through methods.

```rust
// before (1.x)
let samples = audio.data;            // moved/borrowed the public field
let sr = audio.sample_rate;

// after (2.0)
let samples = audio.data();          // &AudioData<'a, T>
let samples_mut = audio.data_mut();  // &mut AudioData<'a, T>
let sr = audio.sample_rate();        // SampleRate
let owned = audio.into_data();       // AudioData<'static, T> (consumes self)
let borrowed = audio.into_data_borrowed(); // AudioData<'a, T> (consumes self)
```

Accessors (all on `AudioSamples<'a, T>`):

```rust
pub const fn sample_rate(&self) -> SampleRate;
pub fn data(&self) -> &AudioData<'a, T>;
pub fn data_mut(&mut self) -> &mut AudioData<'a, T>;
pub fn into_data(self) -> AudioData<'static, T>;
pub fn into_data_borrowed(self) -> AudioData<'a, T>;
```

---

## 2. Dual in-place / borrowing operation variants

This is the largest change. Every Category-1 transforming operation now exists
in two forms with a fixed naming rule:

- The **unmarked name** is **non-mutating**: `op(&self, ..) -> Result<Self>`
  (clones, returns a new value, leaves the receiver untouched).
- The **`_in_place` suffix** is **mutating**: `op_in_place(&mut self, ..) -> Result<()>`.

Infallible operations drop the `Result` (for example `reverse`/`reverse_in_place`,
`scale`/`scale_in_place`). The `_in_place` method holds the logic; the
non-mutating twin is a provided default, so the two never drift.

There are two distinct migration cases, depending on what the 1.x method did.

### 2a. Methods that USED to consume `self` and return `Self`

These now borrow `&self`. Chaining still works; the only difference is the
original is no longer moved, so you can keep using it.

Why: borrowing is consistent with the rest of the API and avoids forcing a move
just to read a transformed copy.

Affected (`AudioProcessing`): `normalize`, `scale`, `clip`, `apply_window`,
`apply_filter`, `mu_compress`, `mu_expand`, `low_pass_filter`,
`high_pass_filter`, `band_pass_filter`, `remove_dc_offset`.

```rust
// before (1.x) - consumed `audio`
let out = audio.normalize(config)?;     // audio is moved, unusable afterward

// after (2.0) - borrows `audio`
let out = audio.normalize(config)?;     // audio still usable
// or mutate in place with no clone:
audio.normalize_in_place(config)?;
```

New signatures:

```rust
fn normalize(&self, config: NormalizationConfig<Self::Sample>) -> AudioSampleResult<Self>;
fn normalize_in_place(&mut self, config: NormalizationConfig<Self::Sample>) -> AudioSampleResult<()>;

fn scale(&self, factor: f64) -> Self;
fn scale_in_place(&mut self, factor: f64);

fn apply_window(&self, window: &NonEmptySlice<Self::Sample>) -> AudioSampleResult<Self>;
fn apply_filter(&self, filter_coeffs: &NonEmptySlice<Self::Sample>) -> AudioSampleResult<Self>;
fn mu_compress(&self, mu: Self::Sample) -> AudioSampleResult<Self>;
fn mu_expand(&self, mu: Self::Sample) -> AudioSampleResult<Self>;
fn low_pass_filter(&self, cutoff_hz: f64) -> AudioSampleResult<Self>;
fn high_pass_filter(&self, cutoff_hz: f64) -> AudioSampleResult<Self>;
fn band_pass_filter(&self, low_cutoff_hz: f64, high_cutoff_hz: f64) -> AudioSampleResult<Self>;
fn remove_dc_offset(&self) -> AudioSampleResult<Self>;
fn clip(&self, min_val: Self::Sample, max_val: Self::Sample) -> AudioSampleResult<Self>;
```

### 2b. Methods that USED to mutate in place (`&mut self -> Result<()>`)

These keep their old behaviour under a new name with the `_in_place` suffix.
The unmarked name is now the non-mutating form. This is the case that silently
changes behaviour if you do not rename: calling the unmarked name no longer
mutates.

Why: the unmarked name is reserved for the non-mutating variant across the whole
API; the in-place primitive moves to the `_in_place` suffix.

```rust
// before (1.x) - mutated `audio` in place
audio.butterworth_lowpass(order, cutoff)?;

// after (2.0) - rename to keep mutating
audio.butterworth_lowpass_in_place(order, cutoff)?;
// or take a filtered copy instead:
let out = audio.butterworth_lowpass(order, cutoff)?;
```

Affected traits and methods (rename the call to `<name>_in_place` to preserve
1.x mutating behaviour):

- `AudioIirFiltering`: `apply_iir_filter`, `filtfilt`, `butterworth_lowpass`,
  `butterworth_highpass`, `butterworth_bandpass`, `chebyshev_i`.

  ```rust
  fn butterworth_lowpass_in_place(&mut self, order: NonZeroUsize, cutoff_frequency: f64) -> AudioSampleResult<()>;
  fn butterworth_lowpass(&self, order: NonZeroUsize, cutoff_frequency: f64) -> AudioSampleResult<Self>;
  fn apply_iir_filter_in_place(&mut self, design: &IirFilterDesign) -> AudioSampleResult<()>;
  fn apply_iir_filter(&self, design: &IirFilterDesign) -> AudioSampleResult<Self>;
  ```

- `AudioParametricEq`: `apply_parametric_eq`, `apply_eq_band`,
  `apply_peak_filter`, `apply_low_shelf`, `apply_high_shelf`,
  `apply_three_band_eq`.

  ```rust
  fn apply_eq_band_in_place(&mut self, band: &EqBand) -> AudioSampleResult<()>;
  fn apply_eq_band(&self, band: &EqBand) -> AudioSampleResult<Self>;
  ```

- `AudioDynamicRange`: `apply_compressor`, `apply_limiter`, `apply_gate`,
  `apply_expander`, and the `*_sidechain` variants.

  ```rust
  fn apply_compressor_in_place(&mut self, config: &CompressorConfig) -> AudioSampleResult<()>;
  fn apply_compressor(&self, config: &CompressorConfig) -> AudioSampleResult<Self>;
  ```

- `AudioEditing`: `fade_in`, `fade_out` (also `trim`, `pad`, `reverse`).

  ```rust
  fn fade_in_in_place(&mut self, duration_seconds: f64, curve: FadeCurve) -> AudioSampleResult<()>;
  fn fade_in(&self, duration_seconds: f64, curve: FadeCurve) -> AudioSampleResult<Self>;
  ```

- `AudioChannelOps`: `swap_channels`, `pan`, `balance`, `apply_to_channel`
  (also `to_mono`, `to_stereo`).

  ```rust
  fn swap_channels_in_place(&mut self, channel1: usize, channel2: usize) -> AudioSampleResult<()>;
  fn swap_channels(&self, channel1: usize, channel2: usize) -> AudioSampleResult<Self>;
  ```

---

## 3. Channel indices are now usize

Why: channel indices were `u32` in `AudioChannelOps` and `usize` elsewhere; 2.0
unifies all indices and lengths to `usize`. Channel counts continue to use the
existing `ChannelCount` newtype where applicable.

```rust
// before (1.x)
audio.swap_channels(0u32, 1u32)?;
let ch = audio.extract_channel(2u32)?;

// after (2.0)
audio.swap_channels_in_place(0usize, 1usize)?;
let ch = audio.extract_channel(2usize)?;
```

Affected signatures:

```rust
fn swap_channels_in_place(&mut self, channel1: usize, channel2: usize) -> AudioSampleResult<()>;
fn apply_to_channel_in_place<F>(&mut self, channel_index: usize, func: F) -> AudioSampleResult<()>
    where F: FnMut(Self::Sample) -> Self::Sample;
fn extract_channel(&self, channel_index: usize) -> AudioSampleResult<AudioSamples<'static, Self::Sample>>;
fn borrow_channel(&self, channel_index: usize) -> AudioSampleResult<AudioSamples<'_, Self::Sample>>;
```

---

## 4. Config structs replace positional arguments

Why: operations with several parameters took loose positional `f64` arguments;
2.0 replaces them with validated config structs (mirroring the existing
`CompressorConfig`). `EnvelopeFollower::process` likewise drops its per-call
argument in favour of state fixed at construction.

### apply_gate / apply_expander take config structs

```rust
// before (1.x) - positional arguments
audio.apply_gate(threshold_db, ratio, attack_ms, release_ms)?;
audio.apply_expander(threshold_db, ratio, attack_ms, release_ms)?;

// after (2.0) - config structs by reference
audio.apply_gate_in_place(&GateConfig::default())?;
audio.apply_expander_in_place(&ExpanderConfig::default())?;
```

```rust
fn apply_gate_in_place(&mut self, config: &GateConfig) -> AudioSampleResult<()>;
fn apply_gate(&self, config: &GateConfig) -> AudioSampleResult<Self>;
fn apply_expander_in_place(&mut self, config: &ExpanderConfig) -> AudioSampleResult<()>;
fn apply_expander(&self, config: &ExpanderConfig) -> AudioSampleResult<Self>;
```

### apply_three_band_eq takes ThreeBandEqConfig

```rust
// before (1.x) - seven positional f64 arguments
audio.apply_three_band_eq(low_freq, low_gain, mid_freq, mid_gain, mid_q, high_freq, high_gain)?;

// after (2.0)
audio.apply_three_band_eq_in_place(&config)?; // config: &ThreeBandEqConfig
```

```rust
fn apply_three_band_eq_in_place(&mut self, config: &ThreeBandEqConfig) -> AudioSampleResult<()>;
fn apply_three_band_eq(&self, config: &ThreeBandEqConfig) -> AudioSampleResult<Self>;
```

### EnvelopeFollower::process drops the detection_method argument

```rust
// before (1.x) - method passed on every call
let follower = EnvelopeFollower::new(attack_ms, release_ms, sample_rate);
let out = follower.process(sample, detection_method);

// after (2.0) - method set once at construction
let mut follower = EnvelopeFollower::new(attack_ms, release_ms, sample_rate, detection_method);
let out = follower.process(sample); // f64 -> f64
```

```rust
pub fn new(attack_ms: f64, release_ms: f64, sample_rate: f64, detection_method: DynamicRangeMethod) -> Self;
pub fn process(&mut self, input: f64) -> f64;
```

---

## 5. Structured return types

Why: tuples carried no field names and could not grow. 2.0 returns named
structs with accessors instead.

### power_spectral_density returns Psd

```rust
// before (1.x)
let (frequencies, density): (Vec<f64>, Vec<f64>) = audio.power_spectral_density(window_size, overlap)?;

// after (2.0)
let psd: Psd = audio.power_spectral_density(window_size, overlap)?;
let frequencies: &[f64] = psd.frequencies();
let density: &[f64] = psd.density();
let (frequencies, density) = psd.into_parts(); // if you need the owned tuple back
```

### track_pitch returns PitchContour

```rust
// before (1.x)
let frames: Vec<(f64, Option<f64>)> = audio.track_pitch(window_size, hop_size, method, threshold, min_freq, max_freq)?;

// after (2.0)
let contour: PitchContour = audio.track_pitch(window_size, hop_size, method, threshold, min_freq, max_freq)?;
let mean = contour.mean_pitch();                 // Option<f64>
for (time, frequency) in contour.voiced_frames() { /* both f64 */ }
let frames: &[PitchFrame] = contour.frames();    // each has .time: f64, .frequency: Option<f64>
```

### estimate_key returns Key

```rust
// before (1.x)
let (tonic_index, confidence): (usize, f64) = audio.estimate_key(&stft_params)?;

// after (2.0)
let key: Key = audio.estimate_key(&stft_params)?;
let tonic: PitchClass = key.tonic;   // enum, e.g. PitchClass::A
let mode: Mode = key.mode;           // Mode::Major | Mode::Minor
let confidence: f64 = key.confidence;
```

`Key` fields are public; `PitchClass` and `Mode` are enums (`PitchClass` also
implements `Display` and `to_index()` / `from_index()`).

---

## 6. ChannelReduction on spectral_centroid and spectral_rolloff

Why: multi-channel handling was inconsistent (one errored, one silently used
channel 0). Both now take a trailing `ChannelReduction` argument so the policy
is explicit. These methods live on `AudioStatistics`.

```rust
// before (1.x)
let centroid = audio.spectral_centroid()?;
let rolloff = audio.spectral_rolloff(0.85)?;

// after (2.0)
use audio_samples::operations::types::ChannelReduction;
let centroid = audio.spectral_centroid(ChannelReduction::Average)?;
let rolloff = audio.spectral_rolloff(0.85, ChannelReduction::First)?;
```

```rust
fn spectral_centroid(&self, reduction: ChannelReduction) -> AudioSampleResult<f64>;
fn spectral_rolloff(&self, rolloff_percent: f64, reduction: ChannelReduction) -> AudioSampleResult<f64>;
```

`ChannelReduction` (`#[non_exhaustive]`) has variants `Error` (default),
`First`, `Average`, and `Channel(usize)`.

---

## 7. Renames and removals

Why: removing duplicated and inconsistently named items.

### median renamed to midpoint_sample

```rust
// before (1.x)
let m = audio.median();          // Option<f64>

// after (2.0)
let m = audio.midpoint_sample(); // Option<f64>
```

### AudioTypeConversion::as_float removed

Use `as_f64` (the two were duplicates).

```rust
// before (1.x)
let f = audio.as_float();

// after (2.0)
let f = audio.as_f64();  // AudioSamples<'static, f64>
```

### SampleType parsing now errors with EnumParseError, and gains FromStr

`TryFrom<&str>` previously used `type Error = ()`. It now carries a span-bearing
`EnumParseError`, and `FromStr` is implemented, so `.parse()` works.

```rust
// before (1.x) - opaque unit error
let t = SampleType::try_from("f32").map_err(|()| "bad sample type")?;

// after (2.0) - structured error, and FromStr
let t: SampleType = "f32".parse()?;            // FromStr, Err = EnumParseError
let t = SampleType::try_from("f32")?;          // TryFrom, Error = EnumParseError
```

```rust
impl TryFrom<&str> for SampleType { type Error = EnumParseError; /* ... */ }
impl core::str::FromStr for SampleType { type Err = EnumParseError; /* ... */ }
```

---

## 8. Explicitly NOT changed

To prevent confusion, these were considered but deliberately left as-is:

- **`NdResult` is retained.** It is the canonical return for analysis ops whose
  dimensionality tracks the input layout (1D for mono, 2D for multi-channel).
  `AudioEnvelopes` still returns `NdResult` (for example
  `amplitude_envelope(&self) -> NdResult<Self::Sample>`); no caller action is
  needed.
- **The `ConvertTo` bound on `cast_as` / `map_into` was dropped.** This is a
  relaxation, not a break: `cast_as` now bounds only `CastInto`, and `map_into`
  is generic over `Fn(T) -> O`. Existing callers continue to compile.

---

## 9. Requirements

- **MSRV is now 1.87** (`rust-version = "1.87"`). Build with Rust 1.87 or newer.
- The `transforms` feature requires **`spectrograms` 1.4.4**.

---

## 10. New in 2.0 (additive, no action needed)

These are new capabilities a 1.x user may want to adopt. Existing code does not
need to change to use them.

- **More IIR filter designs.** Beyond Butterworth and Chebyshev I, the
  `IirFilterType` enum now includes `ChebyshevII`, `Elliptic` (Cauer), and
  `Bessel`, and `FilterResponse` includes `BandStop`. Use them by building an
  `IirFilterDesign` and applying it with `apply_iir_filter` /
  `apply_iir_filter_in_place`.
- **Zero-phase `filtfilt` and a streaming `SosFilter`.** `filtfilt` /
  `filtfilt_in_place` apply a design forward and backward for zero phase
  distortion. `IirFilterDesign::to_sos(sample_rate)` returns a design-once
  `SosFilter` that streams consecutive blocks
  (`process_sample`, `process_samples`, `process_samples_in_place`,
  `process_block`, `reset`).
- **Spectral feature suite and tempo estimation.** On `AudioStatistics`:
  `spectral_centroid`, `spectral_rolloff`, `spectral_bandwidth`,
  `spectral_flatness`, `spectral_crest`, `spectral_slope`, `spectral_contrast`
  (all taking `ChannelReduction`); plus `estimate_tempo(&BeatTrackingConfig)`.
- **New generators.** `exponential_chirp`, `fm_signal`, and band-limited
  oscillators `square_wave_bandlimited`, `sawtooth_wave_bandlimited`,
  `triangle_wave_bandlimited` (all in `utils::generation`).
- **Comparison metrics.** `psnr`, `segmental_snr`, `log_spectral_distance`, and
  per-channel variants (`correlation_per_channel`, `mse_per_channel`,
  `snr_per_channel`) in `utils::comparison`.
- **Phase-spectrum and Lissajous plots** via the plotting module.
- **Zero-copy `windows_ref()`** on `AudioSamples`, an iterator yielding window
  views without copying:
  `windows_ref(&self, window_size: NonZeroUsize, hop_size: NonZeroUsize)`.
