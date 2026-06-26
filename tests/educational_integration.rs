//! Integration tests for the `educational` explanation surface.
//!
//! Verifies that the explanation dispatcher returns non-empty, concept-bearing
//! text for the wired operations and that unrecognized ops resolve to the
//! generic fallback (`None`) rather than fabricated text. `normalize` is wired
//! through the auto-generated `AudioProcessingExplainText` trait rather than the
//! string dispatcher, so it is checked via that trait method.
//!
//! Run with:
//! ```bash
//! cargo test --test educational_integration --features educational,processing
//! ```

use audio_samples::educational::explain_text_for_op;
use audio_samples::operations::traits::AudioProcessingExplainText;
use audio_samples::{AudioSamples, sample_rate};
use ndarray::array;

fn sample() -> AudioSamples<'static, f32> {
    AudioSamples::new_mono(
        array![0.1f32, 0.8, -0.2, 0.9, -0.7, 0.3],
        sample_rate!(44100),
    )
    .unwrap()
}

/// Every wired dispatcher op returns `Some` non-empty text tagged with its
/// `[operation: …]` marker and bearing concept-specific keywords.
#[test]
fn wired_dispatcher_ops_return_concept_text() {
    let a = sample();

    let cases: &[(&str, &str, &[&str])] = &[
        (
            "Compressor",
            "[operation: Compressor]",
            &["threshold", "ratio"],
        ),
        ("EQ Band", "[operation: EQ Band]", &["frequency", "gain"]),
        ("Resample", "[operation: Resample]", &["sample rate"]),
        ("Limiter", "[operation: Limiter]", &["ceiling"]),
    ];

    for (op, tag, keywords) in cases {
        let text = explain_text_for_op(op, &a, &a)
            .unwrap_or_else(|| panic!("op {op} should be wired in the dispatcher"));
        assert!(!text.is_empty(), "explanation for {op} was empty");
        assert!(
            text.contains(tag),
            "explanation for {op} missing tag {tag}: {text}"
        );
        for kw in *keywords {
            assert!(
                text.contains(kw),
                "explanation for {op} missing concept keyword '{kw}': {text}"
            );
        }
    }
}

/// `normalize` is wired via the generated `AudioProcessingExplainText` trait,
/// not the string dispatcher. It must produce non-empty, concept-bearing text.
#[test]
fn normalize_explanation_is_wired_via_trait() {
    let a = sample();
    let text =
        <AudioSamples<'static, f32> as AudioProcessingExplainText>::explain_text_normalize(&a, &a);
    assert!(!text.is_empty(), "normalize explanation was empty");
    assert!(
        text.contains("[operation: Normalize]"),
        "normalize explanation missing operation tag: {text}"
    );
    assert!(
        text.contains("normaliz") || text.contains("Normaliz"),
        "normalize explanation lacks concept text: {text}"
    );

    // The string dispatcher intentionally does NOT handle normalize.
    assert!(
        explain_text_for_op("Normalize", &a, &a).is_none(),
        "dispatcher should not resolve 'Normalize'"
    );
}

/// Unrecognized ops resolve to the generic fallback (`None`), not fabricated
/// text — this is what stops a newly wired op from silently using the default.
#[test]
fn unrecognized_op_returns_fallback_none() {
    let a = sample();
    assert!(explain_text_for_op("Nonexistent", &a, &a).is_none());
    assert!(explain_text_for_op("", &a, &a).is_none());
    assert!(explain_text_for_op("compressor", &a, &a).is_none()); // case-sensitive
}
