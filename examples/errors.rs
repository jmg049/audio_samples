//! A guided tour of `audio_samples`' diagnostic errors.
//!
//! Every fallible API in the crate returns [`AudioSampleError`], which now
//! implements miette's [`Diagnostic`] trait on top of the usual
//! [`std::error::Error`]. That buys four things this example shows off:
//!
//! 1. **Stable error codes** (`audio_samples::<domain>::<variant>`) you can
//!    match on or look up.
//! 2. **Actionable help text** that tells the caller *how to fix it*.
//! 3. **Source spans** that point a caret at the exact offending character of a
//!    parsed string.
//! 4. **Preserved cause chains** — foreign errors are kept as sources instead of
//!    being flattened to a string.
//!
//! Run it with the `fancy` feature to see the coloured, caret-underlined
//! rendering:
//!
//! ```sh
//! cargo run --example errors --features fancy
//! ```
//!
//! Without `fancy` the diagnostics still carry all of their metadata — it just
//! prints as plain text rather than a graphical report.

use std::error::Error;

use audio_samples::utils::audio_math::note_to_midi;
use audio_samples::{
    AudioSampleError, ChannelRequirement, ConversionError, EnumParseError, FeatureError,
    LayoutError, ParameterError, ProcessingError, SampleType,
};
use miette::{Diagnostic, Report};

fn main() {
    section("1. Source spans — the caret points at what's wrong");
    spans_demo();

    section("2. Inspecting a diagnostic programmatically");
    inspect_demo();

    section("3. Matching on the *structured* taxonomy (no string parsing)");
    taxonomy_demo();

    section("4. Preserved cause chains (foreign errors kept as sources)");
    cause_chain_demo();

    section("5. A gallery across every error domain");
    gallery();
}

// ---------------------------------------------------------------------------
// 1. Source spans
// ---------------------------------------------------------------------------

/// Real parse failures from [`note_to_midi`]. The rendered report underlines the
/// offending substring — the unrecognised pitch letter, then the bad octave.
fn spans_demo() {
    // `H` is not a pitch letter — the caret lands on the first character.
    if let Err(e) = note_to_midi("H4") {
        render(e);
    }

    // The note is fine but the octave `x` is not a number — the caret moves to
    // the octave position.
    if let Err(e) = note_to_midi("C#x") {
        render(e);
    }

    // An unrecognised enum token, with the accepted values listed in the help.
    render(EnumParseError::new("PadSide", "middle", &["left", "right", "both"]).into());
}

// ---------------------------------------------------------------------------
// 2. Programmatic inspection
// ---------------------------------------------------------------------------

/// The `Diagnostic` trait exposes the metadata as data, not just rendered text —
/// useful for logging, telemetry, or building your own reporter.
fn inspect_demo() {
    let err: AudioSampleError = ParameterError::out_of_range(
        "cutoff_hz",
        25_000,
        20,
        22_050,
        "exceeds the Nyquist limit for 44.1 kHz audio",
    )
    .into();

    inspect(&err);

    // Spans are data too: pull the labelled offsets straight off the diagnostic.
    let parse = EnumParseError::new("WindowType", "hann", &["hanning", "hamming", "blackman"]);
    println!("  labels on EnumParseError(\"hann\"):");
    if let Some(labels) = parse.labels() {
        for label in labels {
            println!(
                "    • offset {}, len {}{}",
                label.offset(),
                label.len(),
                label
                    .label()
                    .map(|t| format!(" — \"{t}\""))
                    .unwrap_or_default(),
            );
        }
    }
}

/// Print the structured diagnostic fields for any error in the crate.
fn inspect(err: &AudioSampleError) {
    println!("  display  : {err}");
    print_opt("  code     :", err.code());
    print_opt("  severity :", err.severity().map(|s| format!("{s:?}")));
    print_opt("  help     :", err.help());
    print_opt("  docs url :", err.url());
}

// ---------------------------------------------------------------------------
// 3. Structured taxonomy
// ---------------------------------------------------------------------------

/// Because failures now carry typed fields, callers branch on real data —
/// `ChannelRequirement` and `SampleType` — rather than scraping the message.
fn taxonomy_demo() {
    // A mono-only operation handed stereo audio.
    let layout = LayoutError::channel_count_unsupported("stft", ChannelRequirement::Mono, 2);
    match &layout {
        LayoutError::ChannelCountUnsupported {
            operation,
            required,
            actual,
            ..
        } => {
            println!("  `{operation}` needs {required}, got {actual} channel(s)");
            if matches!(required, ChannelRequirement::Mono) {
                println!("  → caller can react: down-mix with `.to_mono()` and retry");
            }
        }
        _ => unreachable!(),
    }
    render(AudioSampleError::from(layout));

    // A conversion error that carries the exact `SampleType` pair.
    let conv = ConversionError::audio_conversion(
        1.5_f32,
        SampleType::F32,
        SampleType::I16,
        "value lies outside the normalised [-1.0, 1.0] range",
    );
    if let ConversionError::AudioConversion { from, to, .. } = &conv {
        println!("  conversion failed: {from} → {to} (typed, not strings)");
    }
    render(AudioSampleError::from(conv));
}

// ---------------------------------------------------------------------------
// 4. Cause chains
// ---------------------------------------------------------------------------

/// Foreign errors are nested as sources, so the whole chain is recoverable —
/// here a real [`ndarray::ShapeError`] sits underneath a `LayoutError`.
fn cause_chain_demo() {
    // 5 elements cannot fill a 2×3 array — ndarray returns a genuine ShapeError.
    let shape_err = ndarray::Array2::<f32>::from_shape_vec((2, 3), vec![0.0; 5])
        .expect_err("a 2x3 array needs 6 elements, so this must fail");

    let err: AudioSampleError = LayoutError::shape_error("reshape_to_stereo", shape_err).into();

    println!("  walking the cause chain:");
    let mut current: Option<&dyn Error> = Some(&err);
    let mut depth = 0;
    while let Some(e) = current {
        println!("    {}{e}", "  ".repeat(depth));
        current = e.source();
        depth += 1;
    }
    render(err);
}

// ---------------------------------------------------------------------------
// 5. Gallery
// ---------------------------------------------------------------------------

/// One representative error per domain, to survey the codes and help hints.
fn gallery() {
    let errors: Vec<AudioSampleError> = vec![
        ParameterError::missing("sample_rate").into(),
        LayoutError::borrowed_mutation("normalize_in_place", "audio is borrowed immutably").into(),
        ProcessingError::algorithm_failure("butterworth", "filter became unstable at this order")
            .into(),
        ConversionError::numeric_cast(
            300.0_f64,
            SampleType::F64,
            SampleType::U8,
            "value exceeds the 0..=255 range of u8",
        )
        .into(),
        FeatureError::not_enabled("transforms", "stft").into(),
        AudioSampleError::empty_data("rms_envelope"),
        AudioSampleError::unsupported("band_stop_butterworth", "not implemented for order > 8"),
    ];

    for err in errors {
        let code = err
            .code()
            .map(|c| c.to_string())
            .unwrap_or_else(|| "<none>".into());
        println!("  [{code}]");
        println!("    {err}");
        if let Some(help) = err.help() {
            println!("    help: {help}");
        }
        println!();
    }
}

// ---------------------------------------------------------------------------
// Rendering helpers
// ---------------------------------------------------------------------------

/// Convert an error into a miette [`Report`] and print it. The `{:?}` formatting
/// dispatches to the graphical handler when the `fancy` feature is enabled.
fn render(err: AudioSampleError) {
    let report: Report = err.into();
    println!("{report:?}\n");
}

fn section(title: &str) {
    println!("\n\x1b[1m{title}\x1b[0m");
    println!("{}", "─".repeat(title.len()));
}

fn print_opt(label: &str, value: Option<impl std::fmt::Display>) {
    match value {
        Some(v) => println!("{label} {v}"),
        None => println!("{label} <none>"),
    }
}
