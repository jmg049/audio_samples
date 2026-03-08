# AudioSamples Roadmap

Where are we going and what do we want to achieve?

Each section corresponds to a minor version release. Each minor release will focus on a specific theme or set of features.

The exact ordering of the minor releases may change and there is no guarantee they will be worked on in the order they are listed here.

## 1.0 --- What is it?

AudioSamples (currently v0.11.0) is near the end of its initial development phase. 
The library is broad and feature rich, the API is mostly stable, and the documentation is comprehensive.

The main criteria for the 1.0 release will be:

- API stability: No breaking changes to the public API. Any necessary changes will be made in a way that allows for migration without breaking existing code. This will take the form of deprecations and additions rather than outright removals and breaking changes.

- Comprehensive documentation: All public APIs will be fully documented with clear explanations, usage examples, and any necessary warnings or notes about edge cases.

- Thorough testing: All features will be covered by unit tests, integration tests, and documentation tests to ensure reliability and correctness.

- Examples: A wide range of examples will be provided to demonstrate the capabilities of the library and to serve as a reference for users.

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