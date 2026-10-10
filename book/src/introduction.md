# Introduction

This book is the learning guide of the Rust crate
[`audio_samples`](https://docs.rs/audio_samples): fourteen chapters on speech
and audio signal processing, from sampling and quantisation to pitch, MFCCs,
dynamics processing and perceptual coding.

Each chapter has the same structure:

1. **Concept**: the theory, with every symbol and its unit defined.
2. **With the library**: the `audio_samples` calls that compute it.
3. **How it works**: the same result derived from first principles in a few
   lines of plain code and checked against the library.
4. **Ask the library to explain**: the `educational` feature's explanations of
   a configuration (formulas, derived numbers, warnings).
5. **Exercises** with worked answers, common pitfalls, references and links to
   the API documentation.

## How the book relates to the crate

The chapters are the Markdown files in `documentation/guide/` of the
repository. The same files are compiled into the API documentation as the
`audio_samples::guide` module, where every code block is a doctest: the
numbers stated in the text are asserted by the code next to them, and the
test suite fails if the library stops producing them.

The figures are computed by the library as well. A small program in `book/figures/`
makes the same calls with the same parameters as the chapter, asserts the
same numbers, and draws the result. A figure therefore shows the values the
chapter's code computes, not an independent illustration.

Code blocks hide their setup lines; the eye icon on a block shows them. Links
on item names (for example [`resample`](crate::resample)) go to the API
documentation on docs.rs.

## Running the code

Every chapter has a lab, `examples/tutorial_NN_<topic>.rs`, which runs the
chapter's computations and prints the results:

```sh
cargo run --release --example tutorial_01_sampling_aliasing --features educational,full_no_plotting
```

To use the snippets in a crate of your own, see
[Guide index and setup](README.md) for the dependencies and their versions.
Lecturers considering the crate for a course should read
[For instructors](FOR_INSTRUCTORS.md).

## Prerequisites

The chapters assume calculus, complex numbers and basic linear algebra. They
do not assume prior signal processing: chapter 1 starts from sampling, and
later chapters refer back to earlier ones where they build on them. Some Rust
is needed to read the code; the API calls are explained where they are used.
