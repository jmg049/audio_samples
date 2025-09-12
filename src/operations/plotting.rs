use crate::{AudioPlottingUtils, AudioSample, AudioSamples, AudioTypeConversion, ConvertTo, I24};
/// Choose a "nice" step in seconds given a rough desired step.
/// Uses 1–2–5 × 10^k progression.
fn nice_step_seconds(rough: f64) -> f64 {
    assert!(rough.is_finite() && rough > 0.0);
    let exp = rough.log10().floor();
    let base = 10f64.powf(exp);
    let mant = rough / base;
    let nice_mant = if mant <= 1.0 {
        1.0
    } else if mant <= 2.0 {
        2.0
    } else if mant <= 5.0 {
        5.0
    } else {
        10.0
    };
    nice_mant * base
}
/// Seconds from 0 to duration with ~target_ticks "nice" spacing (1–2–5).
pub fn time_ticks_seconds<T: AudioSample>(audio: &AudioSamples<T>, target_ticks: usize) -> Vec<f64>
where
    i16: ConvertTo<T>,
    I24: ConvertTo<T>,
    i32: ConvertTo<T>,
    f32: ConvertTo<T>,
    f64: ConvertTo<T>,
{
    let sr = audio.sample_rate() as f64;
    let duration = audio.duration_seconds();
    let rough = (duration / target_ticks.max(2) as f64).max(1.0 / sr);
    let dt = nice_step_seconds(rough);

    let mut ticks = Vec::new();
    let mut t = 0.0;
    while t <= duration + 1e-12 {
        ticks.push(t);
        t += dt;
    }
    if ticks.last().copied().unwrap_or(0.0) < duration - 1e-9 {
        ticks.push(duration);
    }
    ticks
}

impl<T: AudioSample> AudioPlottingUtils<T> for AudioSamples<T>
where
    i16: ConvertTo<T>,
    I24: ConvertTo<T>,
    i32: ConvertTo<T>,
    f32: ConvertTo<T>,
    f64: ConvertTo<T>,
{
    /// Return a time axis (in seconds) with a sensible spacing for plotting.
    /// If `step_seconds` is `Some(dt)`, it is used directly; otherwise, a "nice" dt is chosen
    /// so there are roughly `TARGET_POINTS` points across the clip.
    ///
    /// Invariants:
    /// - spacing is in seconds
    /// - internal stepping uses whole samples (>= 1)
    fn time_axis(&self, step_seconds: Option<f64>) -> Vec<f64> {
        let samples_per_channel = self.samples_per_channel();
        if samples_per_channel == 0 {
            return Vec::new();
        }

        let sr = self.sample_rate() as f64;
        assert!(
            sr.is_finite() && sr > 0.0,
            "sample_rate must be positive and finite"
        );

        let duration = self.duration_seconds(); // should equal samples_per_channel as f64 / sr

        // If the caller didn’t specify a step, pick a "nice" one so we get ~TARGET_POINTS samples.
        const TARGET_POINTS: usize = 20;

        let dt = match step_seconds {
            Some(dt) => dt.max(1.0 / sr), // at least one sample
            None => {
                let rough = duration / TARGET_POINTS as f64;
                nice_step_seconds(rough)
            }
        };

        // Convert to an integer sample stride (at least 1).
        let step_samples = (dt * sr).round().max(1.0) as usize;

        // Number of points we’ll generate (ensure we don’t overflow).
        let n = (samples_per_channel + step_samples - 1) / step_samples;

        // Build the axis using integer math for indices.
        let mut axis = Vec::with_capacity(n + 1);
        let mut idx = 0usize;
        while idx < samples_per_channel {
            axis.push(idx as f64 / sr);
            idx = idx.saturating_add(step_samples);
        }
        // Ensure the last point (duration) is present for clean axes.
        if let Some(&last) = axis.last() {
            if (duration - last) > (0.5 / sr) {
                axis.push(duration);
            }
        }
        axis
    }

    fn frequency_axis(&self) -> Vec<T> {
        todo!()
    }

    fn time_ticks_seconds(&self, target_ticks: usize) -> Vec<f64> {
        time_ticks_seconds(self, target_ticks)
    }
}
