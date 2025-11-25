use audio_samples::{
    AudioTransforms,
    operations::types::{SpectrogramScale, WindowType},
    sine_wave,
};
use std::time::Duration;

fn main() -> audio_samples::AudioSampleResult<()> {
    let sample_rate = 44_100;
    let duration = Duration::from_secs_f64(2.0);

    let audio = sine_wave::<f32, f32>(220.0, duration, sample_rate, 0.8);

    let window_size = 2048;
    let hop_size = 512;

    let window = WindowType::<f32>::Hanning;

    let _stft = audio.stft::<f32>(window_size, hop_size, window)?;

    let _spectrogram = audio.spectrogram::<f32>(
        window_size,
        hop_size,
        WindowType::<f32>::Hanning,
        SpectrogramScale::Log,
        true,
    )?;

    let _mfcc = audio.mfcc::<f32>(13, 40, 80.0, (sample_rate as f32) / 2.0)?;

    Ok(())
}
