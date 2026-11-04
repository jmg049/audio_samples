# Example gallery

Every example in the crate's `examples/` directory, grouped by task. `book/build.sh`
compiles all of them with the `full` and `educational` features and runs all but two,
and the build fails if one fails, so the output below is what the current code prints.
An example asserts its own results, and a figure repeats the example's calls or
comes from the chapter that covers the same operation, where the same numbers are
asserted.

Run one from a checkout of the crate with the command shown under it.


## Getting started and loading audio

The crate's core types, sample conversion and the helpers around them. Reading and writing files is in the separate `audio_samples_io` crate, which has no example here (see gaps below).

### Create audio and convert samples

Builds mono and stereo `AudioSamples` and converts samples between `i16` and `f32`, showing how each sample type maps to the other.

[`examples/basics.rs`](https://github.com/jmg049/audio_samples/blob/main/examples/basics.rs)

```sh
cargo run --example basics --features statistics
```

<!-- output: basics -->

### Statistics of a signal

Peak, RMS, mean, variance, zero crossings and autocorrelation of a generated tone.

[`examples/statistics.rs`](https://github.com/jmg049/audio_samples/blob/main/examples/statistics.rs)

```sh
cargo run --example statistics --features statistics
```

<!-- output: statistics -->

### Unit conversions

Hz, mel and MIDI conversions, decibels, and the other small helpers in `audio_samples::utils`.

[`examples/utils_demo.rs`](https://github.com/jmg049/audio_samples/blob/main/examples/utils_demo.rs)

```sh
cargo run --example utils_demo
```

<!-- output: utils_demo -->

### Detection helpers

Finds the fundamental frequency and the silent regions of a signal without configuring an analysis.

[`examples/detection.rs`](https://github.com/jmg049/audio_samples/blob/main/examples/detection.rs)

```sh
cargo run --example detection
```

<!-- output: detection -->

### Errors as values

How to inspect a typed error, react to it and walk its chain of causes.

[`examples/errors.rs`](https://github.com/jmg049/audio_samples/blob/main/examples/errors.rs)

```sh
cargo run --example errors --features fancy
```

<!-- output: errors -->

### Errors as diagnostics

Renders the same errors as `miette` diagnostics with codes, help text and source spans.

[`examples/error_diagnostics.rs`](https://github.com/jmg049/audio_samples/blob/main/examples/error_diagnostics.rs)

```sh
cargo run --example error_diagnostics --features fancy
```

<!-- output: error_diagnostics -->

## Spectrograms

STFT, spectra and the transforms built on them.

### FFT, STFT, mel, MFCC, chroma, CQT

One call of each main transform on a tone, with the resulting shapes, then reconstruction with the inverse STFT.

[`examples/transforms.rs`](https://github.com/jmg049/audio_samples/blob/main/examples/transforms.rs)

```sh
cargo run --example transforms --features transforms,statistics
```

<!-- figure: ch04-chirp -->

<!-- output: transforms -->

### Tutorial 3: the DFT, windows and leakage

Measures leakage for bin-centred and off-bin tones and compares windows. Text in [chapter 3](03_dft_windows.md).

[`examples/tutorial_03_dft_windows.rs`](https://github.com/jmg049/audio_samples/blob/main/examples/tutorial_03_dft_windows.rs)

```sh
cargo run --example tutorial_03_dft_windows --features educational,full_no_plotting
```

<!-- figure: ch03-windows -->

<!-- output: tutorial_03_dft_windows -->

### Tutorial 4: STFT and spectrograms

Frame counts, the inverse STFT and the time-frequency trade-off. Text in [chapter 4](04_stft_spectrograms.md).

[`examples/tutorial_04_stft_spectrograms.rs`](https://github.com/jmg049/audio_samples/blob/main/examples/tutorial_04_stft_spectrograms.rs)

```sh
cargo run --example tutorial_04_stft_spectrograms --features educational,full_no_plotting
```

<!-- figure: ch04-tradeoff -->

<!-- output: tutorial_04_stft_spectrograms -->

## Mel and MFCC

### Tutorial 10: cepstrum, mel and MFCC

Derives the real cepstrum, the mel filterbank and MFCCs and checks them against the library. Text in [chapter 10](10_cepstrum_mel_mfcc.md).

[`examples/tutorial_10_cepstrum_mel_mfcc.rs`](https://github.com/jmg049/audio_samples/blob/main/examples/tutorial_10_cepstrum_mel_mfcc.rs)

```sh
cargo run --example tutorial_10_cepstrum_mel_mfcc --features educational,full_no_plotting
```

<!-- figure: ch10-filterbank -->

<!-- figure: ch10-mfcc -->

<!-- output: tutorial_10_cepstrum_mel_mfcc -->

### Psychoacoustic analysis

Band layouts on the Bark and mel scales and the perceptual importance of each band.

[`examples/psychoacoustic.rs`](https://github.com/jmg049/audio_samples/blob/main/examples/psychoacoustic.rs)

```sh
cargo run --example psychoacoustic --features psychoacoustic
```

<!-- figure: ch14-scales -->

<!-- output: psychoacoustic -->

## Pitch

### Pitch and harmonics

YIN against autocorrelation, a pitch track, harmonic magnitudes and the harmonic-to-noise ratio.

[`examples/pitch_analysis.rs`](https://github.com/jmg049/audio_samples/blob/main/examples/pitch_analysis.rs)

```sh
cargo run --example pitch_analysis --features pitch-analysis
```

<!-- figure: ch09-track -->

<!-- output: pitch_analysis -->

### Tutorial 9: autocorrelation and YIN

Derives both estimators and shows why they make octave errors. Text in [chapter 9](09_pitch.md).

[`examples/tutorial_09_pitch.rs`](https://github.com/jmg049/audio_samples/blob/main/examples/tutorial_09_pitch.rs)

```sh
cargo run --example tutorial_09_pitch --features educational,full_no_plotting
```

<!-- figure: ch09-yin -->

<!-- output: tutorial_09_pitch -->

### Tutorial 8: LPC and formants

Linear prediction of a synthetic vowel and the formants read from its poles. Text in [chapter 8](08_lpc_formants.md).

[`examples/tutorial_08_lpc_formants.rs`](https://github.com/jmg049/audio_samples/blob/main/examples/tutorial_08_lpc_formants.rs)

```sh
cargo run --example tutorial_08_lpc_formants --features educational,full_no_plotting
```

<!-- figure: ch08-envelope -->

<!-- output: tutorial_08_lpc_formants -->

## Beats and onsets

### Tutorial 13: onsets and tempo

Onset detection and tempo estimation on a click track, with the peak picking derived by hand. Text in [chapter 13](13_onsets_tempo.md).

[`examples/tutorial_13_onsets_tempo.rs`](https://github.com/jmg049/audio_samples/blob/main/examples/tutorial_13_onsets_tempo.rs)

```sh
cargo run --example tutorial_13_onsets_tempo --features educational,full_no_plotting
```

<!-- figure: ch13-onsets -->

<!-- figure: ch13-tempo -->

<!-- output: tutorial_13_onsets_tempo -->

### Onsets and beats on a waveform

Detects onsets and beats in a percussive pattern and marks them on an interactive waveform.

[`examples/plotting_event_markers.rs`](https://github.com/jmg049/audio_samples/blob/main/examples/plotting_event_markers.rs)

```sh
cargo run --example plotting_event_markers --features plotting,onset-detection,beat-tracking,statistics
```

<!-- plot: plotting_event_markers_combined Onsets and beats from the example, as markers on the waveform. -->

<!-- output: plotting_event_markers -->

## Separation

### Harmonic and percussive separation

Mixes a tone with impulses, then separates them with `AudioDecomposition::hpss`.

[`examples/hpss.rs`](https://github.com/jmg049/audio_samples/blob/main/examples/hpss.rs)

```sh
cargo run --example hpss --features decomposition,statistics,editing
```

<!-- figure: ch15-hpss -->

<!-- output: hpss -->

## Effects and editing

### Gain, clipping and windows

Scaling, clipping, windowing and a low-pass filter applied one after another, with the level after each.

[`examples/processing.rs`](https://github.com/jmg049/audio_samples/blob/main/examples/processing.rs)

```sh
cargo run --example processing --features processing,editing,transforms
```

<!-- output: processing -->

### Trim, pad, fade, split and join

The editing operations on a tone, with the length after each step.

[`examples/editing.rs`](https://github.com/jmg049/audio_samples/blob/main/examples/editing.rs)

```sh
cargo run --example editing --features editing
```

<!-- output: editing -->

### Channels

Mono to stereo and back, extracting a channel, panning and balance, interleaving.

[`examples/channels.rs`](https://github.com/jmg049/audio_samples/blob/main/examples/channels.rs)

```sh
cargo run --example channels --features channels,statistics
```

<!-- output: channels -->

### Compressor and limiter

The static curve of a compressor, its gain reduction on a modulated tone, then a mastering limiter.

[`examples/dynamic_range.rs`](https://github.com/jmg049/audio_samples/blob/main/examples/dynamic_range.rs)

```sh
cargo run --example dynamic_range --features dynamic-range,statistics
```

<!-- figure: ch12-static-curve -->

<!-- output: dynamic_range -->

### Tutorial 12: dynamics

Compressor and limiter behaviour measured on steps in level. Text in [chapter 12](12_dynamics.md).

[`examples/tutorial_12_dynamics.rs`](https://github.com/jmg049/audio_samples/blob/main/examples/tutorial_12_dynamics.rs)

```sh
cargo run --example tutorial_12_dynamics --features educational,full_no_plotting
```

<!-- figure: ch12-compressor-step -->

<!-- output: tutorial_12_dynamics -->

### Voice activity detection

Per-frame level and zero-crossing rate on a synthetic vowel in noise, then the energy and combined detectors.

[`examples/vad.rs`](https://github.com/jmg049/audio_samples/blob/main/examples/vad.rs)

```sh
cargo run --example vad --features vad,editing
```

<!-- figure: ch11-vad -->

<!-- output: vad -->

### Tutorial 11: energy and VAD

Short-time energy, zero crossings and a detector built from them. Text in [chapter 11](11_energy_vad.md).

[`examples/tutorial_11_energy_vad.rs`](https://github.com/jmg049/audio_samples/blob/main/examples/tutorial_11_energy_vad.rs)

```sh
cargo run --example tutorial_11_energy_vad --features educational,full_no_plotting
```

<!-- output: tutorial_11_energy_vad -->

## Resampling and sampling

### Resample a signal

Changes the rate by target and by ratio at two quality settings, and checks the resulting length.

[`examples/resampling.rs`](https://github.com/jmg049/audio_samples/blob/main/examples/resampling.rs)

```sh
cargo run --example resampling --features resampling,statistics,processing
```

<!-- figure: ch07-tier-responses -->

<!-- output: resampling -->

### Tutorial 7: resampling

The quality settings measured against each other. Text in [chapter 7](07_resampling.md).

[`examples/tutorial_07_resampling.rs`](https://github.com/jmg049/audio_samples/blob/main/examples/tutorial_07_resampling.rs)

```sh
cargo run --example tutorial_07_resampling --features educational,full_no_plotting
```

<!-- figure: ch07-alias-suppression -->

<!-- output: tutorial_07_resampling -->

### Tutorial 1: sampling and aliasing

Predicts and measures where a tone above Nyquist lands. Text in [chapter 1](01_sampling_aliasing.md).

[`examples/tutorial_01_sampling_aliasing.rs`](https://github.com/jmg049/audio_samples/blob/main/examples/tutorial_01_sampling_aliasing.rs)

```sh
cargo run --example tutorial_01_sampling_aliasing --features educational,full_no_plotting
```

<!-- figure: ch01-samples -->

<!-- output: tutorial_01_sampling_aliasing -->

### Tutorial 2: quantisation and dither

Quantisation error with and without dither. Text in [chapter 2](02_quantisation_dither.md).

[`examples/tutorial_02_quantisation_dither.rs`](https://github.com/jmg049/audio_samples/blob/main/examples/tutorial_02_quantisation_dither.rs)

```sh
cargo run --example tutorial_02_quantisation_dither --features educational,full_no_plotting
```

<!-- figure: ch02-dither-spectra -->

<!-- output: tutorial_02_quantisation_dither -->

## Filters

### IIR filters

Applies a Butterworth low-pass to a two-tone signal and checks the frequency response of the design.

[`examples/iir_filtering.rs`](https://github.com/jmg049/audio_samples/blob/main/examples/iir_filtering.rs)

```sh
cargo run --example iir_filtering --features iir-filtering,editing,statistics
```

<!-- figure: ch06-magnitude -->

<!-- output: iir_filtering -->

### Parametric EQ

A three-band equaliser and a single band applied to a harmonic tone, with the response at chosen frequencies.

[`examples/parametric_eq.rs`](https://github.com/jmg049/audio_samples/blob/main/examples/parametric_eq.rs)

```sh
cargo run --example parametric_eq --features parametric-eq,statistics
```

<!-- figure: ch15-eq -->

<!-- output: parametric_eq -->

### Tutorial 5: convolution and FIR filters

Designs a windowed-sinc low-pass and measures its response. Text in [chapter 5](05_convolution_fir.md).

[`examples/tutorial_05_convolution_fir.rs`](https://github.com/jmg049/audio_samples/blob/main/examples/tutorial_05_convolution_fir.rs)

```sh
cargo run --example tutorial_05_convolution_fir --features educational,full_no_plotting
```

<!-- figure: ch05-magnitude-response -->

<!-- output: tutorial_05_convolution_fir -->

### Tutorial 6: IIR filters

Poles, zeros, stability and notch filters. Text in [chapter 6](06_iir_filters.md).

[`examples/tutorial_06_iir_filters.rs`](https://github.com/jmg049/audio_samples/blob/main/examples/tutorial_06_iir_filters.rs)

```sh
cargo run --example tutorial_06_iir_filters --features educational,full_no_plotting
```

<!-- figure: ch06-pole-zero -->

<!-- output: tutorial_06_iir_filters -->

## Plotting

The `plotting` feature draws interactive Plotly pages. Each example renders its plots to HTML; the frames below show one plot from each. They load the Plotly library from a CDN.

### Waveform and spectrogram

The two basic plots of a stereo tone.

[`examples/plotting_basic.rs`](https://github.com/jmg049/audio_samples/blob/main/examples/plotting_basic.rs)

```sh
cargo run --example plotting_basic --features plotting,transforms
```

<!-- plot: plotting_basic Mel spectrogram from the example. -->

<!-- output: plotting_basic -->

### Magnitude spectrum

dB and linear scales and a zoomed frequency range.

[`examples/plotting_spectrum.rs`](https://github.com/jmg049/audio_samples/blob/main/examples/plotting_spectrum.rs)

```sh
cargo run --example plotting_spectrum --features plotting,transforms
```

<!-- plot: plotting_spectrum Spectrum zoomed to 0 to 2 kHz. -->

<!-- output: plotting_spectrum -->

### Spectrogram frequency range

The automatic zoom to the occupied frequencies, turned off, and overridden by hand.

[`examples/plotting_auto_zoom.rs`](https://github.com/jmg049/audio_samples/blob/main/examples/plotting_auto_zoom.rs)

```sh
cargo run --example plotting_auto_zoom --features plotting,transforms
```

<!-- plot: plotting_auto_zoom Spectrogram with the automatic zoom. -->

<!-- output: plotting_auto_zoom -->

### Frequency padding

The padding around the automatic frequency range.

[`examples/plotting_configurable_padding.rs`](https://github.com/jmg049/audio_samples/blob/main/examples/plotting_configurable_padding.rs)

```sh
cargo run --example plotting_configurable_padding --features plotting,transforms
```

<!-- plot: plotting_configurable_padding_30 Spectrogram with 30 % padding. -->

<!-- output: plotting_configurable_padding -->

### Overlays and shaded regions

Lines and shaded regions over a waveform, and a pitch contour over a spectrogram.

[`examples/plotting_overlays.rs`](https://github.com/jmg049/audio_samples/blob/main/examples/plotting_overlays.rs)

```sh
cargo run --example plotting_overlays --features plotting,transforms
```

<!-- plot: plotting_overlays Waveform with overlays. -->

<!-- output: plotting_overlays -->

### RMS, peak, ZCR and energy overlays

Waveforms with analysis curves computed by the `dsp_overlays` module.

[`examples/plotting_dsp_overlays.rs`](https://github.com/jmg049/audio_samples/blob/main/examples/plotting_dsp_overlays.rs)

```sh
cargo run --example plotting_dsp_overlays --features plotting,statistics
```

<!-- plot: plotting_dsp_overlays_combined Waveform with several overlays together. -->

<!-- output: plotting_dsp_overlays -->

### Spectral centroid and rolloff overlays

Spectrograms of a chirp with centroid and rolloff curves on top.

[`examples/plotting_spectral_overlays.rs`](https://github.com/jmg049/audio_samples/blob/main/examples/plotting_spectral_overlays.rs)

```sh
cargo run --example plotting_spectral_overlays --features plotting,statistics
```

<!-- plot: plotting_spectral_overlays_combined Spectrogram with centroid and rolloff. -->

<!-- output: plotting_spectral_overlays -->

### Composite figures

A waveform and a spectrogram arranged vertically and horizontally.

[`examples/plotting_composite_basics.rs`](https://github.com/jmg049/audio_samples/blob/main/examples/plotting_composite_basics.rs)

```sh
cargo run --example plotting_composite_basics --features plotting,transforms
```

<!-- plot: plotting_composite_basics Vertical composite. -->

<!-- output: plotting_composite_basics -->

### Composite figures with overlays

A modulated tone with its envelope, stacked over its spectrum.

[`examples/plotting_composite_advanced.rs`](https://github.com/jmg049/audio_samples/blob/main/examples/plotting_composite_advanced.rs)

```sh
cargo run --example plotting_composite_advanced --features plotting,envelopes
```

<!-- plot: plotting_composite_advanced Waveform with overlays over the spectrum. -->

<!-- output: plotting_composite_advanced -->

## Explanations, and the rest

### Explanations in the terminal

Prints what common settings mean, using the same text as the `explain` API.

[`examples/explain.rs`](https://github.com/jmg049/audio_samples/blob/main/examples/explain.rs)

```sh
cargo run --example explain --features educational,full_no_plotting
```

<!-- output: explain -->

### Explanation documents

Runs a chain of operations in explain mode and opens the resulting document in a browser, so it is compiled but not run for this page.

[`examples/educational.rs`](https://github.com/jmg049/audio_samples/blob/main/examples/educational.rs)

```sh
cargo run --example educational --features educational,processing
```

### Tutorial 14: psychoacoustics and codecs

Hearing thresholds, masking, mu-law and the perceptual codec. Text in [chapter 14](14_psychoacoustics_codecs.md).

[`examples/tutorial_14_psychoacoustics_codecs.rs`](https://github.com/jmg049/audio_samples/blob/main/examples/tutorial_14_psychoacoustics_codecs.rs)

```sh
cargo run --example tutorial_14_psychoacoustics_codecs --features educational,full_no_plotting
```

<!-- figure: ch14-masking -->

<!-- output: tutorial_14_psychoacoustics_codecs -->

### Benchmark harness

The native side of the benchmark against librosa. It reads signal files written by `benches/librosa_compare/run.sh`, so it is compiled but not run for this page.

[`examples/librosa_bench/main.rs`](https://github.com/jmg049/audio_samples/blob/main/examples/librosa_bench/main.rs)

```sh
cargo run --example librosa_bench --features full_no_plotting
```

## Gaps

Tasks with no Rust example yet:

- Reading and writing files. That code is in the `audio_samples_io` crate, and this
  crate does not depend on it.
- Streaming or block-by-block processing of a long signal.
