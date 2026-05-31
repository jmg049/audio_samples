//! Renders `audio_samples` errors as miette diagnostics.
//!
//! Run with the `fancy` feature to see the coloured caret-underline output,
//! help hints, and stable error codes:
//!
//! ```sh
//! cargo run --example error_diagnostics --features fancy
//! ```
//!
//! Without `fancy`, the same errors print as plain `Display` text — the
//! diagnostic metadata (code, help, spans) is still present, just not rendered
//! graphically.

use audio_samples::utils::audio_math::note_to_midi;
use audio_samples::{AudioSampleError, EnumParseError, ParameterError};
use miette::Report;

/// Convert an error into a miette `Report` and print it.
///
/// The `{:?}` formatting of a `Report` dispatches to miette's graphical handler
/// when the `fancy` feature is enabled, producing the caret-underlined output.
fn show(err: AudioSampleError) {
    let report: Report = err.into();
    println!("{report:?}\n");
}

fn main() {
    // 1. A note name whose pitch letter is not recognised — the caret points at
    //    the offending letter and the help suggests the correct notation.
    if let Err(e) = note_to_midi("H4") {
        show(e);
    }

    // 2. A note name with a non-numeric octave — the caret moves to the octave.
    if let Err(e) = note_to_midi("C#x") {
        show(e);
    }

    // 3. A parameter out of range — note the actionable help and the stable
    //    `audio_samples::parameter::out_of_range` code.
    show(
        ParameterError::out_of_range(
            "cutoff_hz",
            25_000,
            20,
            22_050,
            "exceeds the Nyquist limit for 44.1 kHz audio",
        )
        .into(),
    );

    // 4. An unrecognised enum value — the caret underlines the whole token and
    //    the help lists the accepted alternatives.
    show(EnumParseError::new("PadSide", "middle", &["left", "right"]).into());
}
