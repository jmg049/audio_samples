//! Integration test for the `educational::explain` API: every explainer runs,
//! renders to text, Markdown and HTML, and carries formulas.

use std::time::Duration;

use audio_samples::educational::{
    Explain, ExplainContext, ExplainDoc, ExplainFor, WarningLevel, topics,
};
use audio_samples::operations::onset::OnsetDetectionConfig;
use audio_samples::operations::types::{EqBand, IirFilterDesign, VadConfig};
use audio_samples::{CompressorConfig, LimiterConfig, nzu, sample_rate, sine_wave};
use spectrograms::{MelParams, MfccParams, SpectrogramParams, StftParams, WindowType};

fn all_docs() -> Vec<ExplainDoc> {
    let audio = sine_wave::<f32>(220.0, Duration::from_millis(500), sample_rate!(16000), 0.5);
    let ctx = ExplainContext::for_audio(&audio)
        .with_f0_range(80.0, 400.0)
        .with_n_fft(512)
        .with_n_mels(40);
    let stft = StftParams::new(nzu!(512), nzu!(128), WindowType::Hanning, true).unwrap();
    vec![
        topics::sampling(16_000.0),
        topics::aliasing(7_000.0, 8_000.0),
        topics::quantisation(16),
        topics::dft(512, 16_000.0),
        topics::window(&WindowType::Hanning, 512, 16_000.0),
        topics::yin(0.15, 80.0, 400.0, 16_000.0, Some(1024)),
        topics::autocorrelation_pitch(80.0, 400.0, 16_000.0),
        topics::lpc(18, 16_000.0),
        topics::cepstrum(1024, 16_000.0),
        topics::resampling(48_000.0, 16_000.0),
        topics::fir(&[0.25, 0.5, 0.25], 16_000.0),
        audio.explain(&stft),
        WindowType::Blackman.explain_with(&ctx),
        SpectrogramParams::new(stft.clone(), 16_000.0)
            .unwrap()
            .explain(),
        MelParams::new(nzu!(40), 0.0, 8_000.0)
            .unwrap()
            .explain_with(&ctx),
        MfccParams::new(nzu!(13)).explain_with(&ctx),
        audio.explain(&IirFilterDesign::butterworth_lowpass(nzu!(4), 1_000.0)),
        audio.explain(&EqBand::peak(1_000.0, 6.0, 1.0)),
        audio.explain(&CompressorConfig::vocal()),
        audio.explain(&LimiterConfig::mastering()),
        audio.explain(&VadConfig::default()),
        audio.explain(&OnsetDetectionConfig::percussive()),
    ]
}

#[test]
fn every_explainer_renders_with_formulas() {
    for doc in all_docs() {
        assert!(!doc.formulas.is_empty(), "{} has no formulas", doc.title);
        assert!(!doc.summary.is_empty(), "{} has no summary", doc.title);
        let text = doc.to_text();
        let md = doc.to_markdown();
        let html = doc.to_html();
        assert!(text.starts_with(&doc.title));
        assert!(
            !text.contains('\\'),
            "{}: raw LaTeX leaked into the terminal rendering:\n{text}",
            doc.title
        );
        assert!(
            md.contains("$$"),
            "{} markdown has no display maths",
            doc.title
        );
        assert!(md.starts_with("### "));
        assert!(html.starts_with("<section") && !html.contains("<script"));
        assert_eq!(doc.to_string(), text);
        for q in &doc.derived {
            assert!(
                q.value.is_finite(),
                "{}: {} is not finite",
                doc.title,
                q.name
            );
        }
        for t in &doc.tables {
            assert!(
                t.rows.iter().all(|r| r.len() == t.headers.len()),
                "{}: ragged table {}",
                doc.title,
                t.caption
            );
        }
        // Markdown dollar signs are balanced so Jupyter does not swallow text.
        assert_eq!(
            md.matches('$').count() % 2,
            0,
            "{}: unbalanced $ in markdown",
            doc.title
        );
    }
}

#[test]
fn signal_context_reaches_the_explainer() {
    let audio = sine_wave::<f32>(220.0, Duration::from_millis(500), sample_rate!(16000), 0.5);
    let stft = StftParams::new(nzu!(512), nzu!(128), WindowType::Hanning, true).unwrap();
    let doc = audio.explain(&stft);
    assert_eq!(doc.quantity("Frequency resolution").unwrap().value, 31.25);
    assert!(doc.quantity("Frames").is_some());
    assert!(!doc.has_warning(WarningLevel::Problem));
}
