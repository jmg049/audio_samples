//! librosa rejects audio containing NaN or infinity ("Audio buffer is not
//! finite everywhere", `librosa.util.valid_audio`). The audio-level
//! librosa-compatible operations do the same: one shared entry check returns a
//! `ParameterError` instead of silently producing garbage (e.g. a tempo of
//! 129.2 BPM or a trim of `(1536, 4096)` from a single NaN sample).

use audio_samples::operations::constant_q::{ChromaCqtConfig, CqtConfig, TonnetzConfig};
use audio_samples::operations::effects::{
    HarmonicPercussiveConfig, PitchShiftConfig, SilenceConfig,
};
use audio_samples::operations::features::{
    RmsConfig, SpectralCentroidConfig, ZeroCrossingRateConfig,
};
use audio_samples::operations::rhythm::{
    BeatTrackConfig, OnsetDetectConfig, OnsetStrengthConfig, TempoConfig,
};
use audio_samples::operations::spectrum::{MelSpectrogramConfig, MfccConfig, StftConfig};
use audio_samples::operations::tonal::{ChromaStftConfig, PiptrackConfig, TuningConfig};
use audio_samples::{
    AudioConstantQ, AudioEffects, AudioFeatures, AudioRhythm, AudioSampleError, AudioSampleResult,
    AudioSamples, AudioSpectrum, AudioTonal, nzu, sample_rate,
};
use ndarray::Array1;

fn signal(bad: f64) -> AudioSamples<'static, f64> {
    let mut x: Vec<f64> = (0..22050)
        .map(|i| 0.3 * (i as f64 * 0.0627).sin() + 0.05 * (i as f64 * 0.91).cos())
        .collect();
    x[5000] = bad;
    AudioSamples::new_mono(Array1::from(x), sample_rate!(22050)).unwrap()
}

fn assert_rejected<R>(name: &str, r: AudioSampleResult<R>) {
    match r {
        Err(AudioSampleError::Parameter(e)) => {
            assert!(
                e.to_string().contains("not finite everywhere"),
                "{name}: unexpected error {e}"
            );
        }
        Err(e) => panic!("{name}: expected a ParameterError, got {e}"),
        Ok(_) => panic!("{name}: non-finite input was accepted"),
    }
}

#[test]
fn non_finite_audio_is_rejected() {
    for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let y = signal(bad);
        assert_rejected("stft_matrix", y.stft_matrix(&StftConfig::default()));
        assert_rejected(
            "melspectrogram",
            y.melspectrogram(&MelSpectrogramConfig::default()),
        );
        assert_rejected("mfcc_frames", y.mfcc_frames(&MfccConfig::default()));
        assert_rejected(
            "spectral_centroid_frames",
            y.spectral_centroid_frames(&SpectralCentroidConfig::default()),
        );
        assert_rejected("rms_frames", y.rms_frames(&RmsConfig::default()));
        assert_rejected(
            "zero_crossing_rate_frames",
            y.zero_crossing_rate_frames(&ZeroCrossingRateConfig::default()),
        );
        assert_rejected("chroma_stft", y.chroma_stft(&ChromaStftConfig::default()));
        assert_rejected("piptrack", y.piptrack(&PiptrackConfig::default()));
        assert_rejected(
            "estimate_tuning",
            y.estimate_tuning(&TuningConfig::default()),
        );
        assert_rejected("cqt", y.cqt(&CqtConfig::default()));
        assert_rejected("hybrid_cqt", y.hybrid_cqt(&CqtConfig::default()));
        assert_rejected("chroma_cqt", y.chroma_cqt(&ChromaCqtConfig::default()));
        assert_rejected("tonnetz", y.tonnetz(&TonnetzConfig::default()));
        assert_rejected(
            "onset_strength",
            y.onset_strength(&OnsetStrengthConfig::default()),
        );
        assert_rejected(
            "onset_detect",
            y.onset_detect(&OnsetDetectConfig::default()),
        );
        assert_rejected("tempo", y.tempo(&TempoConfig::default()));
        assert_rejected("beat_track", y.beat_track(&BeatTrackConfig::default()));
        assert_rejected("time_stretch", y.time_stretch(1.5, &StftConfig::default()));
        assert_rejected(
            "pitch_shift",
            y.pitch_shift(2.0, &PitchShiftConfig::librosa()),
        );
        assert_rejected("preemphasis", y.preemphasis(0.97));
        assert_rejected("trim_db", y.trim_db(&SilenceConfig::librosa()));
        assert_rejected("split_db", y.split_db(&SilenceConfig::librosa()));
        assert_rejected("harmonic", y.harmonic(&HarmonicPercussiveConfig::librosa()));
        assert_rejected("lpc", y.lpc(nzu!(4)));
        let mut z = y.clone();
        assert_rejected(
            "time_stretch_in_place",
            z.time_stretch_in_place(2.0, &StftConfig::default()),
        );
    }
    // Finite input (and integer input, which cannot be non-finite) still works.
    assert!(signal(0.0).stft_matrix(&StftConfig::default()).is_ok());
    let pcm =
        AudioSamples::new_mono(Array1::from(vec![100i16; 4096]), sample_rate!(22050)).unwrap();
    assert!(pcm.melspectrogram(&MelSpectrogramConfig::default()).is_ok());
}

/// `f32` audio is analysed in `f32` by `AudioRhythm`; the finiteness check
/// applies there too.
#[test]
fn non_finite_f32_audio_is_rejected_by_rhythm() {
    for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let mut x: Vec<f32> = signal(0.0)
            .as_mono()
            .unwrap()
            .iter()
            .map(|&v| v as f32)
            .collect();
        x[5000] = bad;
        let y = AudioSamples::new_mono(Array1::from(x), sample_rate!(22050)).unwrap();
        assert_rejected(
            "onset_strength f32",
            y.onset_strength(&OnsetStrengthConfig::librosa()),
        );
        assert_rejected(
            "onset_detect f32",
            y.onset_detect(&OnsetDetectConfig::librosa()),
        );
        assert_rejected("tempo f32", y.tempo(&TempoConfig::librosa()));
        assert_rejected("beat_track f32", y.beat_track(&BeatTrackConfig::librosa()));
    }
}

#[test]
fn non_finite_audio_is_rejected_by_pitch_tracking() {
    use audio_samples::{AudioPitchTracking, PyinConfig, YinConfig};
    for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let y = signal(bad);
        assert_rejected("yin", y.yin(&YinConfig::librosa(65.0, 2093.0)));
        assert_rejected("pyin", y.pyin(&PyinConfig::librosa(65.0, 2093.0)));
        let mut x32: Vec<f32> = vec![0.1; 8192];
        x32[100] = bad as f32;
        let y32 = AudioSamples::new_mono(Array1::from(x32), sample_rate!(22050)).unwrap();
        assert_rejected("yin f32", y32.yin(&YinConfig::librosa(65.0, 2093.0)));
    }
}

#[test]
fn non_finite_f32_audio_is_rejected_by_effects() {
    // f32 audio is processed in f32; the entry check is the same.
    let mut x: Vec<f32> = (0..22050)
        .map(|i| 0.3 * (i as f32 * 0.0627).sin())
        .collect();
    x[5000] = f32::NAN;
    let y = AudioSamples::new_mono(Array1::from(x), sample_rate!(22050)).unwrap();
    assert_rejected(
        "f32 time_stretch",
        y.time_stretch(1.5, &StftConfig::librosa()),
    );
    assert_rejected("f32 preemphasis", y.preemphasis(0.97));
    assert_rejected("f32 trim_db", y.trim_db(&SilenceConfig::librosa()));
    assert_rejected("f32 lpc", y.lpc(nzu!(4)));
}
