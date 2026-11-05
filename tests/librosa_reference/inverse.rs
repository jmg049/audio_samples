//! Golden-reference checks for `operations::inverse`.
//!
//! `mel_to_stft`: the reference stops L-BFGS-B early on an underdetermined problem, so the
//! unknowns themselves are not unique: the check is that the mel fit is at least as good,
//! and that the total energy is within 15 %. `mfcc_to_mel` is a closed form, held to 1e-12
//! of the peak. `mel_to_audio` starts from different magnitudes, so its reconstruction is
//! compared through the distance of its mel spectrogram to the input in dB, on bands
//! within 60 dB of the peak: it may exceed the reference's by 1 dB, the spread between
//! the two phase paths after few iterations.

use std::num::NonZeroUsize;

use audio_samples::nzu;
use audio_samples::operations::effects::{GriffinLimConfig, GriffinLimInit};
use audio_samples::operations::filterbanks::{MelFilterbankConfig, mel_filterbank};
use audio_samples::operations::inverse::{
    MelInversionConfig, MfccInversionConfig, mel_to_audio, mel_to_stft, mfcc_to_mel,
};
use audio_samples::operations::spectrum::{
    DctNorm, DctType, StftConfig, melspectrogram_from_spectrogram, stft,
};
use audio_samples::utils::audio_math::MelScale;
use ndarray::{Array2, Axis};

use crate::common::{arr2, load};

fn fx() -> serde_json::Value {
    load("inverse")
}

#[test]
fn mel_to_stft_fit_is_no_worse() {
    let f = fx();
    for (name, c) in f["mel_to_stft"].as_object().unwrap() {
        let m = arr2(&c["M"]);
        let kw = &c["kw"];
        let n_fft = NonZeroUsize::new(kw["n_fft"].as_u64().unwrap() as usize).unwrap();
        let power = kw["power"].as_f64().unwrap();
        let mut mel = MelFilterbankConfig::new()
            .with_n_mels(NonZeroUsize::new(kw["n_mels"].as_u64().unwrap() as usize).unwrap());
        if kw["htk"].as_bool() == Some(true) {
            mel = mel
                .with_scale(MelScale::Htk)
                .with_fmin(kw["fmin"].as_f64().unwrap())
                .with_fmax(kw["fmax"].as_f64());
        }
        let cfg = MelInversionConfig::new(22050.0)
            .with_n_fft(n_fft)
            .with_power(power)
            .with_mel(mel.clone());
        let s = mel_to_stft(m.view().insert_axis(Axis(0)), &cfg).unwrap();
        let s = s.index_axis(Axis(0), 0);
        assert!(s.iter().all(|v| v.is_finite() && *v >= 0.0));
        let basis = mel_filterbank(22050.0, n_fft, &mel).unwrap();
        let powered = s.mapv(|v: f64| v.powf(power));
        let objective = 0.5 * (&basis.dot(&powered) - &m).mapv(|v| v * v).sum();
        let reference = c["objective"].as_f64().unwrap();
        assert!(
            objective <= reference * 1.001,
            "{name}: {objective:e} vs {reference:e}"
        );
        let energy = s.mapv(|v: f64| v * v).sum();
        let want = c["energy"].as_f64().unwrap();
        assert!(
            (energy - want).abs() <= 0.15 * want,
            "{name}: energy {energy} vs {want}"
        );
    }
}

#[test]
fn mfcc_to_mel_closed_form() {
    let f = fx();
    let c = &f["mfcc_to_mel"];
    let mfcc = arr2(&c["mfcc"]);
    for (name, case) in c["cases"].as_object().unwrap() {
        let kw = &case["kw"];
        let mut cfg = MfccInversionConfig::new()
            .with_n_mels(NonZeroUsize::new(kw["n_mels"].as_u64().unwrap() as usize).unwrap());
        if let Some(l) = kw["lifter"].as_f64() {
            cfg = cfg.with_lifter(l);
        }
        if let Some(r) = kw["ref"].as_f64() {
            cfg = cfg.with_reference(r);
        }
        cfg = cfg.with_dct_type(match kw["dct_type"].as_u64() {
            Some(1) => DctType::One,
            Some(3) => DctType::Three,
            _ => DctType::Two,
        });
        if kw.get("norm").is_some_and(serde_json::Value::is_null) {
            cfg = cfg.with_dct_norm(DctNorm::Unnormalized);
        }
        let scale = case["scale"].as_f64().unwrap();
        let input = mfcc.mapv(|v| v * scale);
        let m = mfcc_to_mel(input.view().insert_axis(Axis(0)), &cfg).unwrap();
        let m = m.index_axis(Axis(0), 0).to_owned();
        let want = arr2(&case["mel"]);
        assert_eq!(m.dim(), want.dim(), "{name}");
        let peak = want.iter().fold(0.0_f64, |a, v| a.max(v.abs()));
        let err = (&m - &want).iter().fold(0.0_f64, |a, v| a.max(v.abs()));
        assert!(err <= 1e-12 * peak, "{name}: {err:e} of {peak:e}");
    }
}

#[test]
fn mel_to_audio_keeps_the_mel_spectrogram() {
    let f = fx();
    let c = &f["mel_to_audio"];
    let m = arr2(&c["M"]);
    let n_fft = nzu!(512);
    let hop = nzu!(128);
    let stft_cfg = StftConfig::new()
        .with_n_fft(n_fft)
        .with_hop_length(Some(hop));
    let gl = GriffinLimConfig::new()
        .with_n_iter(8)
        .with_init(GriffinLimInit::ZeroPhase)
        .with_stft(stft_cfg.clone())
        .with_length(Some(8192));
    let cfg = MelInversionConfig::new(22050.0).with_n_fft(n_fft);
    let y = mel_to_audio(m.view().insert_axis(Axis(0)), &cfg, &gl).unwrap();
    assert_eq!(y.ncols(), 8192);
    let spec = stft(y.view(), &stft_cfg).unwrap().mapv(|z| z.norm_sqr());
    let mel_cfg = MelFilterbankConfig::new().with_n_mels(nzu!(40));
    let ours: Array2<f64> = melspectrogram_from_spectrogram(spec.view(), 22050.0, n_fft, &mel_cfg)
        .unwrap()
        .index_axis_move(Axis(0), 0);
    let want = arr2(&c["mel_of_audio"]);
    assert_eq!(ours.dim(), want.dim());
    let peak = m.iter().fold(0.0_f64, |a, &v| a.max(v));
    let mean_db = |x: &Array2<f64>| {
        let (mut sum, mut count) = (0.0, 0_usize);
        for (a, b) in x.iter().zip(m.iter()) {
            if *b > peak * 1e-6 {
                sum += 10.0 * (a.max(1e-30) / b).log10().abs();
                count += 1;
            }
        }
        sum / count as f64
    };
    let (ours_db, want_db) = (mean_db(&ours), mean_db(&want));
    assert!(ours_db <= want_db + 1.0, "{ours_db} dB vs {want_db} dB");
}
