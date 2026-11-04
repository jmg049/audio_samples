//! Named example clips, synthesised on demand.
//!
//! [`example_clip`] renders a short piece of realistic audio from nothing but its
//! arguments, so documentation and experiments have something to analyse without an
//! audio file or a download. Each clip has known contents, listed on [`ExampleClip`],
//! and those contents are what the crate's tests recover with real analysis.
//!
//! Rendering uses only addition, subtraction, multiplication, division, square root,
//! `floor` and a hand-written PRNG, on `f64`. IEEE 754 fixes the result of each of those
//! operations, so a clip does not depend on the platform's maths library, on the optimiser
//! or on the CPU. The same arguments give the same bytes on every target whose `f64`
//! arithmetic follows IEEE 754 exactly, which excludes x87 builds such as `i586`.
//!
//! ```rust
//! use audio_samples::sample_rate;
//! use audio_samples::utils::generation::{ExampleClip, example_clip};
//! use std::time::Duration;
//!
//! let clip = example_clip::<f32>(ExampleClip::DrumLoop, Duration::from_secs(4), sample_rate!(22050));
//! assert_eq!(clip.sample_rate().get(), 22050);
//! assert_eq!(clip.samples_per_channel().get(), 4 * 22050);
//! assert_eq!(clip.num_channels().get(), 1);
//! ```

use std::f64::consts::{LN_2, LN_10, PI, TAU};
use std::num::NonZeroU32;
use std::time::Duration;

use ndarray::Array1;

use crate::{AudioSamples, ConvertTo, StandardSample};

/// Tempo of [`ExampleClip::DrumLoop`], [`ExampleClip::Chords`] and
/// [`ExampleClip::ChordsAndDrums`] in beats per minute.
pub const EXAMPLE_TEMPO_BPM: f64 = 110.0;

/// Signal-to-noise ratio of [`ExampleClip::NoisyRecording`] in dB, measured over the voiced
/// parts of the speech.
pub const EXAMPLE_NOISY_SNR_DB: f64 = 15.0;

/// A named clip for [`example_clip`].
///
/// | Clip | Contents |
/// |------|----------|
/// | [`Chords`](Self::Chords) | C, G, Am, F in equal temperament (A4 = 440 Hz), one chord per four beats at [`EXAMPLE_TEMPO_BPM`], soft attacks, harmonic partials only |
/// | [`DrumLoop`](Self::DrumLoop) | Kick on beats 1 and 3, snare on beats 2 and 4, hi-hat on every eighth note, at [`EXAMPLE_TEMPO_BPM`], with the first kick at time zero |
/// | [`ChordsAndDrums`](Self::ChordsAndDrums) | Exactly `0.5 * Chords + 0.5 * DrumLoop` in `f64`, before conversion to the sample type |
/// | [`Speech`](Self::Speech) | Source-filter synthesis: a voiced source following [`example_speech_f0`] through formant resonators, repeating every 6 s with a pause after each 2.4 s phrase |
/// | [`NoisyRecording`](Self::NoisyRecording) | `Speech` plus low-passed noise and 50 Hz mains hum with harmonics, at [`EXAMPLE_NOISY_SNR_DB`] over the voiced parts |
///
/// The vowels of `Speech`, each 0.6 s long, with their first three formants in Hz:
/// /a/ (730, 1090, 2440), /e/ (530, 1840, 2480), /i/ (270, 2290, 3010), /o/ (570, 840, 2410),
/// /u/ (300, 870, 2240). The phrase starting at 0 s sings /a e i o/ and the phrase starting at
/// 3 s sings /u a i e/.
///
/// Components above 95 % of the Nyquist frequency are omitted, so a low sample rate gives the
/// same clip with less top end and no aliasing from the partials.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum ExampleClip {
    /// Pitched music with a known chord progression.
    Chords,
    /// A drum pattern with a known tempo and known onset times.
    DrumLoop,
    /// [`Chords`](Self::Chords) and [`DrumLoop`](Self::DrumLoop) mixed equally.
    ChordsAndDrums,
    /// A voiced speech-like signal with formants and a known F0 contour.
    Speech,
    /// [`Speech`](Self::Speech) with background noise and mains hum at a known level.
    NoisyRecording,
}

impl ExampleClip {
    /// Every clip, in the order of the table on [`ExampleClip`].
    pub const ALL: [Self; 5] = [
        Self::Chords,
        Self::DrumLoop,
        Self::ChordsAndDrums,
        Self::Speech,
        Self::NoisyRecording,
    ];

    /// The clip's lower-case name, such as `"drum_loop"`.
    #[inline]
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Chords => "chords",
            Self::DrumLoop => "drum_loop",
            Self::ChordsAndDrums => "chords_and_drums",
            Self::Speech => "speech",
            Self::NoisyRecording => "noisy_recording",
        }
    }

    /// Finds a clip by the name given by [`name`](Self::name).
    ///
    /// # Returns
    ///
    /// The clip, or `None` when no clip has that name.
    #[inline]
    #[must_use]
    pub fn from_name(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|clip| clip.name() == name)
    }
}

/// Renders a named example clip.
///
/// The clip contains no stored audio. The same arguments give the same samples on any platform
/// (see the [module documentation](self)), and a shorter clip is a prefix of a longer one with
/// the same clip, sample rate and sample type. The audio is computed in `f64` and converted to
/// `T` with [`ConvertTo`]; its peak stays below full scale, so integer types do not clip.
///
/// # Arguments
///
/// - `clip` - Which clip to render.
/// - `duration` - Length of the clip. Clips repeat their pattern for as long as requested.
/// - `sample_rate` - Sample rate of the returned audio.
///
/// # Returns
///
/// A mono [`AudioSamples`] buffer of `T` at `sample_rate`, with `duration * sample_rate` samples
/// rounded down.
///
/// # Panics
///
/// If `duration` gives zero samples at `sample_rate`.
///
/// # Examples
///
/// ```rust
/// use audio_samples::sample_rate;
/// use audio_samples::utils::generation::{ExampleClip, example_clip};
/// use std::time::Duration;
///
/// let speech = example_clip::<i16>(ExampleClip::Speech, Duration::from_secs(2), sample_rate!(16000));
/// assert_eq!(speech.samples_per_channel().get(), 32000);
///
/// let again = example_clip::<i16>(ExampleClip::Speech, Duration::from_secs(2), sample_rate!(16000));
/// assert_eq!(speech.as_slice(), again.as_slice());
/// ```
#[must_use]
pub fn example_clip<T>(
    clip: ExampleClip,
    duration: Duration,
    sample_rate: NonZeroU32,
) -> AudioSamples<'static, T>
where
    T: StandardSample,
{
    let sr = f64::from(sample_rate.get());
    let n = (duration.as_secs_f64() * sr) as usize;
    assert!(n > 0, "example_clip: duration gives zero samples");

    let rendered = match clip {
        ExampleClip::Chords => render_chords(n, sr),
        ExampleClip::DrumLoop => render_drums(n, sr),
        ExampleClip::ChordsAndDrums => {
            let mut mix = render_chords(n, sr);
            let drums = render_drums(n, sr);
            for (m, d) in mix.iter_mut().zip(&drums) {
                *m = 0.5 * *m + 0.5 * *d;
            }
            mix
        }
        ExampleClip::Speech => render_speech(n, sr),
        ExampleClip::NoisyRecording => render_noisy(n, sr),
    };

    let samples: Vec<T> = rendered.into_iter().map(ConvertTo::convert_to).collect();
    // safety: `n > 0` was checked above, so the array is non-empty
    unsafe { AudioSamples::new_mono_unchecked(Array1::from_vec(samples), sample_rate) }
}

/// The F0 of [`ExampleClip::Speech`] and [`ExampleClip::NoisyRecording`] at a given time.
///
/// # Arguments
///
/// - `time_secs` - Time from the start of the clip, in seconds.
///
/// # Returns
///
/// The fundamental frequency in Hz, or `None` during a pause and at negative or non-finite
/// times.
///
/// # Examples
///
/// ```rust
/// use audio_samples::utils::generation::example_speech_f0;
///
/// assert!(example_speech_f0(1.0).is_some());
/// assert_eq!(example_speech_f0(2.7), None);
/// ```
#[must_use]
pub fn example_speech_f0(time_secs: f64) -> Option<f64> {
    if !time_secs.is_finite() || time_secs < 0.0 {
        return None;
    }
    let (phrase, u) = phrase_position(time_secs)?;
    Some(if phrase == 0 {
        // Falling statement with an accent in the middle.
        150.0 - 40.0 * u + 18.0 * sin_turns(0.5 * u)
    } else {
        // Rising question.
        105.0 + 55.0 * u * u
    })
}

const PHRASE_SECS: f64 = 2.4;
const PHRASE_PERIOD_SECS: f64 = 3.0;
const VOWEL_SECS: f64 = 0.6;

/// Which phrase a time falls in and how far through it, from 0 to 1. `None` during a pause.
fn phrase_position(t: f64) -> Option<(usize, f64)> {
    let cycles = (t / PHRASE_PERIOD_SECS).floor();
    let offset = t - cycles * PHRASE_PERIOD_SECS;
    if offset >= PHRASE_SECS {
        return None;
    }
    let phrase = usize::from(cycles - 2.0 * (cycles / 2.0).floor() != 0.0);
    Some((phrase, offset / PHRASE_SECS))
}

// ---------------------------------------------------------------------------
// Portable maths: basic IEEE 754 operations only. `f64::sin`, `exp`, `powf`, `powi` and the
// like are not portable, because their results depend on the platform's maths library.
// ---------------------------------------------------------------------------

/// `sin(2 * pi * x)`, accurate to about 1e-15 for moderate `x`.
#[inline]
fn sin_turns(x: f64) -> f64 {
    let wrapped = x - (x + 0.5).floor();
    let reduced = if wrapped > 0.25 {
        0.5 - wrapped
    } else if wrapped < -0.25 {
        -0.5 - wrapped
    } else {
        wrapped
    };
    let a = TAU * reduced;
    let z = a * a;
    let p = -1.0 / 121_645_100_408_832_000.0;
    let p = p * z + 1.0 / 355_687_428_096_000.0;
    let p = p * z - 1.0 / 1_307_674_368_000.0;
    let p = p * z + 1.0 / 6_227_020_800.0;
    let p = p * z - 1.0 / 39_916_800.0;
    let p = p * z + 1.0 / 362_880.0;
    let p = p * z - 1.0 / 5040.0;
    let p = p * z + 1.0 / 120.0;
    let p = p * z - 1.0 / 6.0;
    a * (p * z + 1.0)
}

/// `cos(2 * pi * x)`.
#[inline]
fn cos_turns(x: f64) -> f64 {
    sin_turns(x + 0.25)
}

/// `e^x`, accurate to about 1e-15 over the range used here. Returns 0 below `e^-700`.
fn exp(x: f64) -> f64 {
    if x < -700.0 {
        return 0.0;
    }
    let k = (x / LN_2 + 0.5).floor();
    let r = x - k * LN_2;
    let mut term = 1.0;
    let mut sum = 1.0;
    for i in 1..=13 {
        term = term * r / f64::from(i);
        sum += term;
    }
    let exponent = 1023 + k as i64;
    sum * f64::from_bits((exponent as u64) << 52)
}

/// Equal-tempered frequency of a MIDI note, with A4 (note 69) at 440 Hz.
fn midi_hz(note: f64) -> f64 {
    440.0 * exp((note - 69.0) / 12.0 * LN_2)
}

/// A rising curve from 0 at `x <= 0` to 1 at `x >= 1`.
#[inline]
fn smoothstep(x: f64) -> f64 {
    let x = x.clamp(0.0, 1.0);
    x * x * (3.0 - 2.0 * x)
}

/// `SplitMix64`, used so that the noise does not depend on another crate's stream.
struct Noise(u64);

impl Noise {
    #[inline]
    const fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    /// Uniform in `[-1, 1)`.
    #[inline]
    fn next(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / 4_503_599_627_370_496.0 - 1.0
    }
}

/// Sample index of a time, rounded to nearest.
fn index_of(seconds: f64, sr: f64) -> usize {
    (seconds * sr + 0.5).floor() as usize
}

// ---------------------------------------------------------------------------
// Chords
// ---------------------------------------------------------------------------

/// MIDI notes per chord: C, G, Am, F, each with a bass note.
const CHORDS: [[f64; 4]; 4] = [
    [48.0, 60.0, 64.0, 67.0],
    [43.0, 59.0, 62.0, 67.0],
    [45.0, 60.0, 64.0, 69.0],
    [41.0, 60.0, 65.0, 69.0],
];
const CHORD_PARTIALS: usize = 8;
const CHORD_GAIN: f64 = 0.14;
const CHORD_ATTACK_SECS: f64 = 0.04;
const CHORD_RELEASE_SECS: f64 = 0.08;

fn beat_secs() -> f64 {
    60.0 / EXAMPLE_TEMPO_BPM
}

fn render_chords(n: usize, sr: f64) -> Vec<f64> {
    let mut out = vec![0.0; n];
    let bar_secs = 4.0 * beat_secs();
    let hold = bar_secs;
    let note_len = index_of(hold + CHORD_RELEASE_SECS, sr);
    let attack = index_of(CHORD_ATTACK_SECS, sr).max(1);
    let release = index_of(CHORD_RELEASE_SECS, sr).max(1);
    let mut note = vec![0.0; note_len];

    let mut bar = 0usize;
    loop {
        let start = index_of(bar as f64 * bar_secs, sr);
        if start >= n {
            break;
        }
        let audible = note_len.min(n - start);
        for &midi in &CHORDS[bar % CHORDS.len()] {
            note[..audible].fill(0.0);
            let f0 = midi_hz(midi);
            for k in 1..=CHORD_PARTIALS {
                let harmonic = k as f64;
                let freq = f0 * harmonic;
                if freq >= 0.475 * sr {
                    break;
                }
                add_decaying_partial(
                    &mut note[..audible],
                    freq / sr,
                    CHORD_GAIN / harmonic,
                    exp(-(0.25 + 0.2 * harmonic) / sr),
                );
            }
            for (i, s) in note[..audible].iter_mut().enumerate() {
                if i < attack {
                    *s *= smoothstep(i as f64 / attack as f64);
                }
                if i + release > note_len {
                    *s *= smoothstep((note_len - i) as f64 / release as f64);
                }
            }
            for (o, s) in out[start..start + audible].iter_mut().zip(&note) {
                *o += *s;
            }
        }
        bar += 1;
    }
    out
}

/// Adds `amp * d^i * sin(2 * pi * freq * i)` to `out`, where `freq` is in cycles per sample.
/// The sine comes from rotating a unit vector, which needs one complex multiplication a sample.
#[inline]
fn add_decaying_partial(out: &mut [f64], freq: f64, amp: f64, d: f64) {
    let (wr, wi) = (cos_turns(freq) * d, sin_turns(freq) * d);
    let (mut re, mut im) = (amp, 0.0_f64);
    for o in out {
        *o += im;
        let next_re = re * wr - im * wi;
        im = re * wi + im * wr;
        re = next_re;
    }
}

// ---------------------------------------------------------------------------
// Drums
// ---------------------------------------------------------------------------

const DRUM_GAIN: f64 = 0.65;

fn kick(sr: f64) -> Vec<f64> {
    let len = index_of(0.25, sr).max(1);
    let (amp_decay, sweep) = (exp(-1.0 / (0.07 * sr)), exp(-1.0 / (0.03 * sr)));
    let mut noise = Noise(1);
    let (mut amp, mut extra_hz, mut phase) = (0.9, 110.0, 0.0);
    let click_len = index_of(0.002, sr);
    (0..len)
        .map(|i| {
            let tone = amp * sin_turns(phase);
            let click = if i < click_len {
                0.15 * noise.next()
            } else {
                0.0
            };
            phase += (50.0 + extra_hz) / sr;
            extra_hz *= sweep;
            amp *= amp_decay;
            tone + click
        })
        .collect()
}

fn snare(sr: f64) -> Vec<f64> {
    let len = index_of(0.2, sr).max(1);
    let (tone_decay, noise_decay) = (exp(-1.0 / (0.05 * sr)), exp(-1.0 / (0.06 * sr)));
    let mut noise = Noise(2);
    let (mut tone_amp, mut noise_amp) = (0.35, 0.6);
    let mut previous = 0.0;
    (0..len)
        .map(|i| {
            let t = i as f64 / sr;
            let tone = tone_amp * (sin_turns(185.0 * t) + 0.6 * sin_turns(330.0 * t));
            let white = noise.next();
            // A first difference tilts the noise towards high frequencies.
            let hiss = noise_amp * (white - 0.7 * previous);
            previous = white;
            tone_amp *= tone_decay;
            noise_amp *= noise_decay;
            tone + hiss
        })
        .collect()
}

fn hat(sr: f64) -> Vec<f64> {
    let len = index_of(0.06, sr).max(1);
    let decay = exp(-1.0 / (0.012 * sr));
    let mut noise = Noise(3);
    let mut amp = 0.22;
    let (mut x1, mut x2) = (0.0, 0.0);
    (0..len)
        .map(|_| {
            let x = noise.next();
            // A second difference removes everything but the top of the band.
            let s = amp * (x - 2.0 * x1 + x2) / 4.0;
            x2 = x1;
            x1 = x;
            amp *= decay;
            s
        })
        .collect()
}

fn mix_in(out: &mut [f64], shot: &[f64], start: usize) {
    if start < out.len() {
        for (o, s) in out[start..].iter_mut().zip(shot) {
            *o += *s;
        }
    }
}

fn render_drums(n: usize, sr: f64) -> Vec<f64> {
    let mut out = vec![0.0; n];
    let (kick, snare, hat) = (kick(sr), snare(sr), hat(sr));
    let eighth = beat_secs() / 2.0;
    let mut step = 0usize;
    loop {
        let start = index_of(step as f64 * eighth, sr);
        if start >= n {
            break;
        }
        match step % 8 {
            0 | 4 => mix_in(&mut out, &kick, start),
            2 | 6 => mix_in(&mut out, &snare, start),
            _ => {}
        }
        mix_in(&mut out, &hat, start);
        step += 1;
    }
    for o in &mut out {
        *o *= DRUM_GAIN;
    }
    out
}

// ---------------------------------------------------------------------------
// Speech
// ---------------------------------------------------------------------------

/// First three formants in Hz of /a/, /e/, /i/, /o/, /u/.
const VOWELS: [[f64; 3]; 5] = [
    [730.0, 1090.0, 2440.0],
    [530.0, 1840.0, 2480.0],
    [270.0, 2290.0, 3010.0],
    [570.0, 840.0, 2410.0],
    [300.0, 870.0, 2240.0],
];
/// Indices into [`VOWELS`]: /a e i o/, then /u a i e/.
const PHRASE_VOWELS: [[usize; 4]; 2] = [[0, 1, 2, 3], [4, 0, 2, 1]];
/// Fourth and fifth formants, the same for every vowel.
const UPPER_FORMANTS: [f64; 2] = [3500.0, 4500.0];
/// Formant bandwidths in Hz.
const BANDWIDTHS: [f64; 5] = [80.0, 100.0, 140.0, 200.0, 250.0];
const FORMANT_GLIDE_SECS: f64 = 0.08;
const SPEECH_GAIN: f64 = 0.13;
/// Coefficients are updated this many times a second.
const CONTROL_RATE: f64 = 1000.0;

/// The formants at a point in a phrase, gliding from one vowel to the next around each boundary.
fn formants_at(phrase: usize, u: f64) -> [f64; 3] {
    let vowels = &PHRASE_VOWELS[phrase];
    let seconds = u * PHRASE_SECS;
    let index = ((seconds / VOWEL_SECS).floor() as usize).min(vowels.len() - 1);
    let into = seconds - index as f64 * VOWEL_SECS;
    let current = VOWELS[vowels[index]];
    if index + 1 < vowels.len() && into > VOWEL_SECS - FORMANT_GLIDE_SECS {
        let next = VOWELS[vowels[index + 1]];
        let w = smoothstep((into - (VOWEL_SECS - FORMANT_GLIDE_SECS)) / FORMANT_GLIDE_SECS);
        return [
            current[0] + w * (next[0] - current[0]),
            current[1] + w * (next[1] - current[1]),
            current[2] + w * (next[2] - current[2]),
        ];
    }
    current
}

/// Source amplitude: phrase attack and release times a dip between syllables.
fn speech_amplitude(u: f64) -> f64 {
    let seconds = u * PHRASE_SECS;
    let fade = smoothstep(seconds / 0.03) * smoothstep((PHRASE_SECS - seconds) / 0.05);
    let syllable = sin_turns(0.5 * seconds / VOWEL_SECS);
    fade * (0.5 + 0.5 * syllable * syllable)
}

/// A band-limited sawtooth sample at phase `t` in `[0, 1)`, with polynomial corrections around
/// the discontinuity.
#[inline]
fn poly_blep_saw(t: f64, dt: f64) -> f64 {
    let naive = 2.0 * t - 1.0;
    if t < dt {
        let x = t / dt;
        naive - (x + x - x * x - 1.0)
    } else if t > 1.0 - dt {
        let x = (t - 1.0) / dt;
        naive - (x * x + x + x + 1.0)
    } else {
        naive
    }
}

#[derive(Clone, Copy, Default)]
struct Resonator {
    a1: f64,
    a2: f64,
    b0: f64,
    y1: f64,
    y2: f64,
}

impl Resonator {
    /// Sets a two-pole resonance with unit gain at DC, or a pass-through when the formant is out
    /// of band. The filter state is kept, so the coefficients can change while it runs.
    fn tune(&mut self, freq: f64, bandwidth: f64, sr: f64) {
        if freq >= 0.45 * sr {
            self.a1 = 0.0;
            self.a2 = 0.0;
            self.b0 = 1.0;
            return;
        }
        let r = exp(-PI * bandwidth / sr);
        self.a1 = 2.0 * r * cos_turns(freq / sr);
        self.a2 = -r * r;
        self.b0 = 1.0 - self.a1 - self.a2;
    }

    #[inline]
    fn process(&mut self, x: f64) -> f64 {
        let y = self.b0 * x + self.a1 * self.y1 + self.a2 * self.y2;
        self.y2 = self.y1;
        self.y1 = y;
        y
    }
}

fn render_speech(n: usize, sr: f64) -> Vec<f64> {
    let mut out = Vec::with_capacity(n);
    let block = ((sr / CONTROL_RATE) as usize).max(1);
    let mut filters = [Resonator::default(); 5];
    for (filter, (&freq, &bw)) in filters[3..]
        .iter_mut()
        .zip(UPPER_FORMANTS.iter().zip(&BANDWIDTHS[3..]))
    {
        filter.tune(freq, bw, sr);
    }
    let (mut phase, mut dt, mut amp, mut amp_step) = (0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64);
    let mut voiced = false;

    for i in 0..n {
        if i % block == 0 {
            let t = i as f64 / sr;
            let position = phrase_position(t);
            voiced = position.is_some();
            let target = if let Some((phrase, u)) = position {
                let formants = formants_at(phrase, u);
                for (j, filter) in filters[..3].iter_mut().enumerate() {
                    filter.tune(formants[j], BANDWIDTHS[j], sr);
                }
                dt = example_speech_f0(t).unwrap_or(100.0) / sr;
                speech_amplitude(u)
            } else {
                0.0
            };
            // The amplitude moves in a straight line over the block, so it has no steps.
            amp_step = (target - amp) / block as f64;
        }

        let source = if voiced || amp > 0.0 {
            let s = poly_blep_saw(phase, dt);
            phase += dt;
            if phase >= 1.0 {
                phase -= 1.0;
            }
            s * amp
        } else {
            phase = 0.0;
            0.0
        };
        amp = (amp + amp_step).max(0.0);

        let mut y = source;
        for filter in &mut filters {
            y = filter.process(y);
        }
        out.push(SPEECH_GAIN * y);
    }
    out
}

// ---------------------------------------------------------------------------
// Noisy recording
// ---------------------------------------------------------------------------

/// RMS of the speech over its voiced parts. It changes by less than 1 dB between 16 kHz and
/// 96 kHz.
const VOICED_RMS: f64 = 0.1365;
const HISS_CUTOFF_HZ: f64 = 2500.0;

fn render_noisy(n: usize, sr: f64) -> Vec<f64> {
    // 10^(-SNR / 20), with the SNR written out so that no `powf` is needed.
    let noise_rms = VOICED_RMS * exp(-EXAMPLE_NOISY_SNR_DB / 20.0 * LN_10);
    // 90 % of the noise power is hiss and 10 % is hum.
    let hiss_rms = noise_rms * (0.9_f64).sqrt();
    let hum_rms = noise_rms * (0.1_f64).sqrt();

    let mut out = render_speech(n, sr);
    let mut noise = Noise(4);
    // A one-pole low-pass divides the variance of white noise by (1 + a) / (1 - a), and uniform
    // noise on [-1, 1) has variance 1/3.
    let a = exp(-TAU * HISS_CUTOFF_HZ / sr);
    let hiss_gain = hiss_rms * (3.0 * (1.0 + a) / (1.0 - a)).sqrt() * (1.0 - a);
    let mut low = 0.0;

    // Hum at 50, 100 and 150 Hz with amplitudes 1, 0.5 and 0.25, whose power adds up to
    // (1 + 0.25 + 0.0625) / 2.
    let hum_gain = hum_rms / (1.3125_f64 / 2.0).sqrt();
    let hum_components = [(50.0, 1.0), (100.0, 0.5), (150.0, 0.25)];
    let mut hum = [(0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64); 3];
    for (state, &(freq, level)) in hum.iter_mut().zip(&hum_components) {
        let freq = freq / sr;
        *state = (hum_gain * level, 0.0, cos_turns(freq), sin_turns(freq));
    }

    for o in &mut out {
        low = a * low + noise.next();
        let mut sample = hiss_gain * low;
        for (re, im, wr, wi) in &mut hum {
            sample += *im;
            let next_re = *re * *wr - *im * *wi;
            *im = *re * *wi + *im * *wr;
            *re = next_re;
        }
        *o += sample;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sin_turns_matches_std_to_round_off() {
        for i in -2000..2000 {
            let x = f64::from(i) * 0.013_7;
            // Wrapped first, so that the error of `TAU * x` does not count against the result.
            let wrapped = TAU * (x - x.round());
            assert!((sin_turns(x) - wrapped.sin()).abs() < 3e-14, "x = {x}");
            assert!((cos_turns(x) - wrapped.cos()).abs() < 3e-14, "x = {x}");
        }
    }

    #[test]
    fn exp_matches_std_to_round_off() {
        for i in -700..200 {
            let x = f64::from(i) * 0.1;
            let relative = (exp(x) - x.exp()).abs() / x.exp();
            assert!(relative < 2e-14, "x = {x}");
        }
        assert_eq!(exp(-1000.0), 0.0);
    }

    #[test]
    fn midi_hz_is_equal_temperament() {
        assert!((midi_hz(69.0) - 440.0).abs() < 1e-12);
        assert!((midi_hz(60.0) - 261.625_565_300_598_6).abs() < 1e-9);
    }

    /// The rendering must not use a floating-point function whose result comes from the
    /// platform's maths library. This is a check on the source text, up to the test module.
    #[test]
    fn rendering_avoids_platform_maths() {
        let source = include_str!("example_clips.rs");
        let rendering = source.split("#[cfg(test)]").next().unwrap();
        for call in [
            ".sin(",
            ".cos(",
            ".tan(",
            ".exp(",
            ".exp2(",
            ".ln(",
            ".log(",
            ".log2(",
            ".log10(",
            ".powf(",
            ".powi(",
            ".sinh(",
            ".cosh(",
            ".tanh(",
            ".atan(",
            ".atan2(",
            ".asin(",
            ".acos(",
            ".hypot(",
            ".cbrt(",
            ".mul_add(",
            ".sin_cos(",
            ".exp_m1(",
            ".ln_1p(",
        ] {
            assert!(!rendering.contains(call), "uses {call}");
        }
    }

    #[test]
    fn f0_contour_has_pauses_and_stays_in_speech_range() {
        for i in 0..6000 {
            let t = f64::from(i) / 1000.0;
            let in_pause = (2.4..3.0).contains(&t) || t >= 5.4;
            match example_speech_f0(t) {
                Some(f0) => {
                    assert!(!in_pause, "t = {t}");
                    assert!((80.0..200.0).contains(&f0), "t = {t}, f0 = {f0}");
                }
                None => assert!(in_pause, "t = {t}"),
            }
        }
        assert_eq!(example_speech_f0(-1.0), None);
        assert_eq!(example_speech_f0(f64::NAN), None);
        assert_eq!(example_speech_f0(f64::INFINITY), None);
    }

    #[test]
    fn names_round_trip() {
        for clip in ExampleClip::ALL {
            assert_eq!(ExampleClip::from_name(clip.name()), Some(clip));
        }
        assert_eq!(ExampleClip::from_name("trumpet"), None);
    }
}
