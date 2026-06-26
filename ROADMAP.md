# AudioSamples Roadmap

Where are we going and what do we want to achieve?

Each section corresponds to a minor version release. Each minor release will focus on a specific theme or set of features.

The exact ordering of the minor releases may change and there is no guarantee they will be worked on in the order they are listed here.

## 2.0 --- A unified API

AudioSamples 2.0 is a deliberate breaking release that unifies the public
API into a single, consistent convention. The library is broad and feature
rich; 2.0 makes its surface predictable and ergonomic.

Headline changes in 2.0:

- Dual-variant operations: every transforming operation exposes both an
  in-place primitive (`op_in_place(&mut self) -> Result<()>`) and a
  non-mutating borrowing form (`op(&self) -> Result<Self>`).

- Encapsulation: `AudioSamples` fields are sealed behind accessors so the
  invariant-preserving guarantees cannot be bypassed.

- Consistency: unified channel-index types (`usize`), a single
  `ChannelReduction` policy for multi-channel analysis, config structs with
  builders/validation for multi-parameter operations, and structured return
  types (`Psd`, `PitchContour`, `Key`) in place of bare tuples.

Post-2.0 the API is intended to remain stable: subsequent changes should be
additive, with deprecations rather than silent breakage.

Ongoing quality criteria (across releases):

- Comprehensive documentation: all public APIs fully documented with clear
  explanations, usage examples, and notes on edge cases.

- Thorough testing: unit, integration, and documentation tests covering every
  feature, backed by CI across the feature matrix.

- Examples: a wide range of runnable, self-verifying examples.

## Error handling

- Need to move towards using `miette` for improved error reporting and diagnostics. This will involve updating the existing error handling code to use `miette`'s error types and reporting mechanisms, as well as ensuring that all errors are properly propagated and handled throughout the library.

- Need to ensure proper usage and coverage of the existing error types in terms of correct usage, informative information and messages, and comprehensive coverage of all error cases.

## Plotting and DSP overlays

- Need to tidy up and expand the plotting and DSP overlay features, including adding more examples, improving documentation, and ensuring that the API is intuitive and easy to use.

- A large chunk of this will be improving the `html_view` crate and its integration with `audio_samples` to allow for more interactive and informative visualizations of audio data and DSP overlays. This could include features such as live plotting of audio data, interactive controls for adjusting DSP parameters and seeing the effects in real time, and more advanced visualizations of audio features and processing results.

- The plotting system is built on top of plotly and html. We can do a lot with html.

## Educational content

- A long term / high in the sky goal of the project is to enable new and existing audio programmers to learn about audio programming concepts and techniques through the use of the library and its documentation. We often use libraries without understanding the underlying mechanisms and techniques, which is fine, but we should make it easy for users to learn about the underlying concepts if they want to.

- Use the plotting and DSP overlay features. Use `manim` and `audio_samples_python` to create educational content that explains audio programming concepts and techniques in an accessible and engaging way. These could take the form of short form videos, interactive notebooks, or even full courses on audio programming topics.

## ML Integration

- `audio_samples_python` implements the numpy array protocol and the dlpack protocol. This means that we can easily convert audio data to and from numpy arrays, and we can also convert audio data to and from dlpack tensors. The Rust version `audio_samples` does not yet implement the dlpack protocol, but this is something that could be added in the future to allow for even easier integration with machine learning frameworks that support dlpack such as `candle`, `ort`, `tch`, etc.

## General Extensions

- Need to integrate newer features from the ``Spectrograms`` crate, such as the binaural spectrograms.