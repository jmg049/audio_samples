use audio_samples::{
    AudioSampleResult, AudioSamples,
    operations::{AudioPlotBuilders, PlotComposer},
};
use std::time::Duration;

pub fn main() -> AudioSampleResult<()> {
    let note_a: AudioSamples<'_, f32> =
        audio_samples::sine_wave::<f32, f32>(440.0, Duration::from_secs_f64(10.0), 20, 0.5);
    println!("{:#}", note_a);
    PlotComposer::<f32>::new()
        .add_element(note_a.waveform_plot(None).unwrap())
        .with_title("Sine Wave A4")
        .show(true)
        .unwrap();

    Ok(())
}
