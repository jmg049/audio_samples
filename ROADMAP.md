# AudioSamples Roadmap

This document lists planned work.

Each section corresponds to a minor version release. Each minor release will focus on a specific theme or set of features.

The exact ordering of the minor releases may change and there is no guarantee they will be worked on in the order they are listed here.

## API stability

Version 2.0 unified the public API into a single convention:

- Dual-variant operations: every transforming operation exposes both an
  in-place primitive (`op_in_place(&mut self) -> Result<()>`) and a
  non-mutating borrowing form (`op(&self) -> Result<Self>`).

- Encapsulation: `AudioSamples` fields are sealed behind accessors so the
  invariant-preserving guarantees cannot be bypassed.

- Consistency: unified channel-index types (`usize`), a single
  `ChannelReduction` policy for multi-channel analysis, config structs with
  builders/validation for multi-parameter operations, and structured return
  types (`Psd`, `PitchContour`, `Key`) in place of bare tuples.

From 2.0 on the API is intended to remain stable: changes should be additive,
with deprecations rather than silent breakage.

Quality criteria for every release:

- Comprehensive documentation: all public APIs fully documented with clear
  explanations, usage examples, and notes on edge cases.

- Thorough testing: unit, integration, and documentation tests covering every
  feature, backed by CI across the feature matrix.

- Examples: a wide range of runnable, self-verifying examples.

## Error handling

- Check that every error case is covered by the most specific error type, and that each message and diagnostic says what went wrong and how to fix it.

## Plotting and DSP overlays

- Tidy up and expand the plotting and DSP overlay features: more examples, better documentation, and a more intuitive API.

- Improve the `html_view` crate and its integration with `audio_samples` for more interactive and informative visualisations of audio data and DSP overlays: live plotting, interactive controls for adjusting DSP parameters and hearing the effect in real time, and richer views of audio features and processing results.

- The plotting system is built on plotly and HTML, which leaves room for interactive features.

## Educational content

- A long-term goal of the project is to help new and existing audio programmers learn audio programming concepts and techniques through the library and its documentation. Libraries are often used without understanding the mechanisms underneath, which is fine, but users who want to learn those concepts should find it easy.

- Use the plotting and DSP overlay features, `manim` and `audio_samples_python` to create educational content that explains audio programming concepts and techniques: short videos, interactive notebooks, or full courses on audio programming topics.

## ML integration

- `audio_samples_python` implements the array protocol and the DLPack protocol, so audio converts to and from Python arrays and DLPack tensors. The Rust crate does not implement DLPack yet; adding it would ease integration with machine learning frameworks that support it, such as `candle`, `ort` and `tch`.

## General extensions

- Integrate further features from the `spectrograms` crate, such as binaural spectrograms.
