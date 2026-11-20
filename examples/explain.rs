//! Print explanations of common settings in the terminal.
//!
//! ```text
//! cargo run --example explain --features educational,full_no_plotting
//! cargo run --example explain --features educational,full_no_plotting -- all
//! cargo run --example explain --features educational,full_no_plotting -- markdown
//! ```
//!
//! The default prints a handful of explanations; `all` prints every
//! explainer, and `markdown` prints the same handful as Markdown (the format
//! notebooks render).

use std::time::Duration;

use audio_samples::educational::{Explain, ExplainContext, ExplainDoc, ExplainFor, topics};
use audio_samples::operations::onset::OnsetDetectionConfig;
use audio_samples::operations::types::{EqBand, IirFilterDesign, VadConfig};
use audio_samples::{CompressorConfig, LimiterConfig, nzu, sample_rate, sine_wave};
use spectrograms::{MelParams, MfccParams, StftParams, WindowType};

fn main() {
    let mode = std::env::args().nth(1).unwrap_or_default();

    // A 16 kHz speech-like test signal: half a second of a 120 Hz tone.
    let audio = sine_wave::<f32>(120.0, Duration::from_millis(500), sample_rate!(16000), 0.5);

    // An STFT that is too short for a low male voice: the explanation warns.
    let stft = StftParams::new(nzu!(256), nzu!(128), WindowType::Hanning, true)
        .expect("valid STFT parameters");
    let ctx = ExplainContext::for_audio(&audio).with_f0_range(80.0, 300.0);

    let mut docs: Vec<ExplainDoc> = vec![
        topics::aliasing(7_000.0, 8_000.0),
        topics::quantisation(16),
        stft.explain_with(&ctx),
        audio.explain(&IirFilterDesign::butterworth_lowpass(nzu!(4), 1_000.0)),
        audio.explain(&CompressorConfig::vocal()),
    ];

    if mode == "all" {
        let ctx = ctx.with_n_fft(256).with_n_mels(40);
        docs.extend([
            topics::sampling(16_000.0),
            topics::dft(256, 16_000.0),
            topics::window(&WindowType::Hanning, 256, 16_000.0),
            topics::yin(0.15, 80.0, 300.0, 16_000.0, Some(1024)),
            topics::autocorrelation_pitch(80.0, 300.0, 16_000.0),
            topics::lpc(18, 16_000.0),
            topics::cepstrum(1024, 16_000.0),
            topics::resampling(48_000.0, 16_000.0),
            topics::fir(&[0.25, 0.5, 0.25], 16_000.0),
            MelParams::new(nzu!(40), 0.0, 8_000.0)
                .expect("valid mel parameters")
                .explain_with(&ctx),
            MfccParams::new(nzu!(13)).explain_with(&ctx),
            audio.explain(&EqBand::peak(1_000.0, 6.0, 1.0)),
            audio.explain(&LimiterConfig::mastering()),
            audio.explain(&VadConfig::default()),
            audio.explain(&OnsetDetectionConfig::percussive(audio.sample_rate())),
        ]);
    }

    for doc in &docs {
        if mode == "markdown" {
            println!("{}", doc.to_markdown());
        } else {
            println!("{doc}");
        }
        println!();
    }
}
