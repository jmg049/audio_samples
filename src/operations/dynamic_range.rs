//! Dynamic range processors: compressor, limiter, noise gate and expander, as the
//! methods of [`AudioDynamicRange`], configured by [`CompressorConfig`],
//! [`LimiterConfig`], [`GateConfig`] and [`ExpanderConfig`] (each with presets).
//! The building blocks, an [`EnvelopeFollower`] for level detection and a
//! [`LookaheadBuffer`], are public for use in other processors.
//!
//! ```
//! use audio_samples::{AudioDynamicRange, CompressorConfig, sample_rate, sine_wave};
//! use std::time::Duration;
//!
//! // A loud tone through the vocal preset: -18 dBFS threshold, 3:1, +3 dB makeup.
//! let loud = sine_wave::<f32>(440.0, Duration::from_millis(500), sample_rate!(44100), 0.9);
//! let squashed = loud.apply_compressor(&CompressorConfig::vocal()).unwrap();
//!
//! let peak = |a: &[f32]| a.iter().fold(0.0f32, |m, x| m.max(x.abs()));
//! let tail = &squashed.as_slice().unwrap()[11_025..]; // after the attack settles
//! assert!(peak(tail) < 0.6);
//! ```
//!
//! ## How it works
//!
//! Every processor is a feed-forward design with three stages, run per sample `n`
//! and per channel:
//!
//! 1. **Detector and ballistics.** An [`EnvelopeFollower`] turns the input `x[n]`
//!    into a smoothed level `e[n]` (linear amplitude, full scale = 1.0).
//! 2. **Static gain computer.** The level in dBFS, `L[n] = 20·log10(e[n])`
//!    (an envelope of exactly 0 is mapped to −80 dBFS), is mapped to a gain reduction `r[n] ≥ 0` in dB.
//! 3. **Gain application.** `y[n] = x[n] · 10^{−r[n]/20} · 10^{M/20}`, with `M`
//!    the makeup gain in dB (compressor only).
//!
//! The gain is computed in the level domain: the envelope is smoothed first and
//! the static curve is applied to the smoothed level.
//!
//! ### Detector
//!
//! The detector value `d[n]` depends on [`DynamicRangeMethod`]:
//!
//! - `Peak`: `d[n] = |x[n]|`.
//! - `Rms`: `d[n] = sqrt(mean(x²))` over the last 10 ms (`⌊0.01·f_s⌋` samples).
//!   A sine of peak amplitude `A` reads `A/√2`, i.e. 3.01 dB below `Peak`.
//! - `Hybrid`: `d[n] = max(|x[n]|, RMS)`.
//!
//! The detector choice therefore shifts the effective threshold.
//!
//! ### Attack and release ballistics
//!
//! The envelope is a one-pole smoother with two coefficients:
//!
//! ```text
//! e[n] = a·e[n−1] + (1 − a)·d[n],   a = a_A if d[n] > e[n−1] else a_R
//! a_A = exp(−1 / (τ_A·f_s)),        a_R = exp(−1 / (τ_R·f_s))
//! ```
//!
//! `τ_A` (`attack_ms`) and `τ_R` (`release_ms`) are time constants in seconds and
//! `f_s` is the sample rate in Hz. After an upward step the envelope follows
//! `1 − e^{−t/τ_A}`: it reaches 63.2 % of the step after `τ_A`, 99 % after about
//! `4.6·τ_A`, and its 10–90 % rise time is `τ_A·ln 9 ≈ 2.2·τ_A`. A downward step
//! decays as `e^{−t/τ_R}`. The envelope never reaches "full" gain reduction in
//! finite time. Commercial units quote attack and release with different
//! conventions (some 10–90 %, some to 63 %), so the numbers are not directly
//! comparable between products.
//!
//! ### Static gain computer
//!
//! With input level `L` (dBFS), threshold `T` (dBFS), ratio `R` (≥ 1) and knee
//! width `W` (dB), the compressor output level `Y = L − r` is
//!
//! ```text
//! hard knee:  Y = L                         for L ≤ T
//!             Y = T + (L − T)/R             for L > T
//!
//! soft knee:  Y = L                         for 2(L − T) < −W
//!             Y = L + (1/R − 1)·(L − T + W/2)² / (2W)   for 2|L − T| ≤ W
//!             Y = T + (L − T)/R             for 2(L − T) > W
//! ```
//!
//! The soft knee is the quadratic knee of Giannoulis, Massberg & Reiss (2012),
//! "Digital Dynamic Range Compressor Design — A Tutorial and Analysis", JAES 60(6).
//! It is centred on `T`, spans `[T − W/2, T + W/2]`, is monotone and has a
//! continuous slope (1 below the knee, `1/R` above it). At `L = T` it gives
//! `Y = T − (1 − 1/R)·W/8`. `W = 0` is a hard knee.
//!
//! The limiter uses the same curve with `R → ∞` and the ceiling `C` in place of
//! `T`, so its static output never exceeds `C`. With a soft knee, gain reduction
//! starts at `C − W/2`.
//!
//! The gate and the expander are downward expanders: below the threshold
//! the output level is `Y = T − R·(T − L)`, i.e. the reduction is
//! `r = (R − 1)·(T − L)` dB for `L < T`, and `r = 0` above. A ratio `R` therefore
//! means "1 dB below the threshold in gives `R` dB below it out" (1:R expansion).
//! `R = 10` attenuates a signal 20 dB under the threshold by 180 dB.
//! Ratios below 1 produce upward expansion (a boost below the threshold).
//!
//! ### Lookahead and latency
//!
//! The compressor and limiter accept `lookahead_ms`, converted to
//! `N_L = round(lookahead_ms · 10⁻³ · f_s)` samples. The detector sees the
//! undelayed signal, the signal path is delayed by `N_L` samples, and the gain
//! applied to sample `n` uses the largest reduction computed for samples
//! `n … n + N_L` (a max-hold over the lookahead window):
//!
//! ```text
//! y[n] = x[n] · 10^{−max(r[n], …, r[n+N_L])/20}
//! ```
//!
//! Gain reduction is thus in place up to `N_L` samples before a peak arrives.
//! With `N_L ≥ 5·τ_A·f_s` the envelope has fully settled when the peak reaches
//! the output. A real-time implementation would add `N_L` samples of latency;
//! because these functions process the whole buffer, they compensate that
//! delay, so output index `n` stays aligned with input index `n`. Beyond the
//! end of the signal the detector is fed zeros. The gate and expander have no
//! lookahead.
//!
//! ## Limitations
//!
//! - The limiter is not a brick-wall (clipping-free) limiter. Its detector is
//!   the smoothed envelope above, so the ceiling is exact only once the envelope
//!   has settled on a steady level. Transients shorter than a few `τ_A`, and the
//!   peaks of low-frequency waveforms when `τ_A` is long compared with the
//!   period, can exceed the ceiling.
//! - [`LimiterConfig::isp_limiting`] detects true peaks from a 4× oversampled
//!   detector signal, which can still read a peak up to 0.7 dB low near the
//!   Nyquist frequency.
//! - The side-chain filters of [`SideChainConfig`] shape only the detector
//!   input; the signal path is not filtered.
//!
//! [`SideChainConfig`]: crate::operations::types::SideChainConfig

use ndarray::{ArrayView1, ArrayViewMut1, Axis, s};
use non_empty_slice::NonEmptySlice;

use crate::operations::iir_design::{Sos, design_sos};
use crate::operations::iir_filtering::sosfilt_in_place;
use crate::operations::traits::AudioDynamicRange;
use crate::operations::types::{
    CompressorConfig, DynamicRangeMethod, ExpanderConfig, GateConfig, IirFilterDesign, KneeType,
    LimiterConfig, SideChainConfig,
};
use crate::repr::AudioData;
use crate::traits::StandardSample;
use crate::utils::audio_math::{
    amplitude_to_db as linear_to_db, db_to_amplitude as db_to_linear, ms_to_samples,
};
use crate::{
    AudioSampleError, AudioSampleResult, AudioSamples, AudioTypeConversion, ConvertTo,
    ParameterError,
};
use std::collections::VecDeque;
use std::num::NonZeroUsize;

/// A first-order RC-style envelope follower with separate attack and release time constants.
///
/// ## Purpose
///
/// Tracks the amplitude envelope of an audio signal over time. Each call to
/// [`process`][EnvelopeFollower::process] returns the current smoothed level
/// `e[n] = a·e[n−1] + (1 − a)·d[n]`, where `d[n]` is the detector value (peak,
/// RMS or hybrid). The coefficient is the attack coefficient `a_A` while
/// `d[n] > e[n−1]` (the envelope rises) and the release coefficient `a_R`
/// otherwise (the envelope falls). Each coefficient is `a = exp(−1/(τ·f_s))`,
/// with `τ` the time constant in seconds and `f_s` the sample rate in Hz, so a
/// step in `d` is followed as `1 − e^{−t/τ}` (63.2 % after `τ`). See the module
/// documentation for details.
///
/// ## Intended Usage
///
/// Used internally by the compressor and limiter implementations to derive a smoothed
/// level estimate from the input signal. It can also be used directly when implementing
/// custom dynamic range processors.
///
/// ## Invariants
///
/// - The envelope value is always non-negative.
/// - Attack and release coefficients are in `[0.0, 1.0]`; a value of `0.0` means
///   instantaneous response with no smoothing.
#[derive(Debug, Clone)]
pub struct EnvelopeFollower {
    /// Current envelope value
    envelope: f64,
    /// Attack coefficient (0.0 to 1.0)
    attack_coeff: f64,
    /// Release coefficient (0.0 to 1.0)
    release_coeff: f64,
    /// Sample rate in Hz, kept so the RMS window can be resized when the
    /// detection method changes.
    sample_rate: f64,
    /// RMS window for RMS detection
    rms_window: VecDeque<f64>,
    /// RMS window size
    rms_window_size: usize,
    /// RMS window sum
    rms_sum: f64,
    /// Detection method chosen at construction time
    detection_method: DynamicRangeMethod,
}

impl EnvelopeFollower {
    /// Creates a new envelope follower with the given time constants.
    ///
    /// # Arguments
    ///
    /// - `attack_ms` - Attack time constant `τ_A` in milliseconds: after a rising step
    ///   the envelope reaches 63.2 % of the step after `τ_A`. Use `0.0` for
    ///   instantaneous response.
    /// - `release_ms` - Release time constant `τ_R` in milliseconds: after a falling
    ///   step the envelope has covered 63.2 % of the drop after `τ_R`. Use `0.0` for
    ///   instantaneous response.
    /// - `sample_rate` - Sample rate in Hz used to convert time constants to
    ///   per-sample coefficients.
    /// - `detection_method` - Determines whether peak, RMS, or hybrid level detection
    ///   is used. For `Rms` and `Hybrid` modes, an internal window of approximately
    ///   10 ms is allocated.
    ///
    /// # Returns
    ///
    /// A new `EnvelopeFollower` with the envelope initialised to `0.0`.
    ///
    /// # Example
    ///
    /// ```
    /// use audio_samples::operations::dynamic_range::EnvelopeFollower;
    /// use audio_samples::operations::types::DynamicRangeMethod;
    ///
    /// let follower = EnvelopeFollower::new(5.0, 50.0, 44100.0, DynamicRangeMethod::Peak);
    /// ```
    #[inline]
    #[must_use]
    pub fn new(
        attack_ms: f64,
        release_ms: f64,
        sample_rate: f64,
        detection_method: DynamicRangeMethod,
    ) -> Self {
        let attack_coeff = if attack_ms > 0.0 {
            (-1.0 / (attack_ms * 0.001 * sample_rate)).exp()
        } else {
            0.0
        };

        let release_coeff = if release_ms > 0.0 {
            (-1.0 / (release_ms * 0.001 * sample_rate)).exp()
        } else {
            0.0
        };

        let rms_window_size = Self::rms_window_len(detection_method, sample_rate);

        Self {
            envelope: 0.0,
            attack_coeff,
            release_coeff,
            sample_rate,
            rms_window: VecDeque::with_capacity(rms_window_size),
            rms_window_size,
            rms_sum: 0.0,
            detection_method,
        }
    }

    /// Processes a single sample and returns the current envelope level.
    ///
    /// The detector value is computed from `input` according to `detection_method`,
    /// then smoothed with the attack coefficient when it is above the current envelope
    /// and with the release coefficient otherwise.
    ///
    /// The detection algorithm is the one supplied to [`new`][EnvelopeFollower::new];
    /// it is not chosen per call.
    ///
    /// # Arguments
    ///
    /// - `input` - The input sample value. The follower tracks amplitude, so any
    ///   numeric scale is accepted.
    ///
    /// # Returns
    ///
    /// The smoothed envelope value (always `≥ 0.0`).
    ///
    /// # Example
    ///
    /// ```
    /// use audio_samples::operations::dynamic_range::EnvelopeFollower;
    /// use audio_samples::operations::types::DynamicRangeMethod;
    ///
    /// let mut follower = EnvelopeFollower::new(1.0, 10.0, 44100.0, DynamicRangeMethod::Peak);
    /// let level = follower.process(0.5);
    /// assert!(level > 0.0);
    /// ```
    #[inline]
    pub fn process(&mut self, input: f64) -> f64 {
        self.process_with_peak(input, input.abs())
    }

    /// [`process`](Self::process) with the peak level given separately: `peak`
    /// replaces `|input|` as the peak detector value (`Peak`, and the peak half of
    /// `Hybrid`), while the RMS window still sees `input`. The limiter passes the
    /// true (inter-sample) peak here.
    #[inline]
    fn process_with_peak(&mut self, input: f64, peak: f64) -> f64 {
        let detector_value = match self.detection_method {
            DynamicRangeMethod::Peak => peak,
            DynamicRangeMethod::Rms => {
                // Add new sample to RMS window
                let sample_squared: f64 = input * input;
                self.rms_window.push_back(sample_squared);
                self.rms_sum += sample_squared;

                // Remove old samples if window is full
                if self.rms_window.len() > self.rms_window_size
                    && let Some(old_sample) = self.rms_window.pop_front()
                {
                    self.rms_sum -= old_sample;
                }

                // Calculate RMS
                if self.rms_window.is_empty() {
                    0.0
                } else {
                    (self.rms_sum / self.rms_window.len() as f64).sqrt()
                }
            }
            DynamicRangeMethod::Hybrid => {
                // Hybrid: max of peak and RMS
                // RMS calculation
                let sample_squared = input * input;
                self.rms_window.push_back(sample_squared);
                self.rms_sum += sample_squared;

                if self.rms_window.len() > self.rms_window_size
                    && let Some(old_sample) = self.rms_window.pop_front()
                {
                    self.rms_sum -= old_sample;
                }

                let rms = if self.rms_window.is_empty() {
                    0.0
                } else {
                    (self.rms_sum / self.rms_window.len() as f64).sqrt()
                };

                peak.max(rms)
            }
        };

        // Attack while the detector is above the current envelope (the
        // envelope must rise), release while it is at or below it.
        let coeff = if detector_value > self.envelope {
            self.attack_coeff
        } else {
            self.release_coeff
        };

        self.envelope = coeff * self.envelope + (1.0 - coeff) * detector_value;
        self.envelope
    }

    /// Overrides the detection method used by [`process`][EnvelopeFollower::process].
    ///
    /// The detection method is normally fixed at construction. This setter exists
    /// for callers that reuse a single configured follower across detection modes
    /// (for example envelope analysis). The RMS window is resized to match the new
    /// method (10 ms for `Rms` and `Hybrid`, one sample for `Peak`) and its
    /// contents are cleared; the envelope value itself is kept.
    ///
    /// # Arguments
    ///
    /// - `detection_method` - The detection algorithm to apply from now on.
    #[inline]
    pub fn set_detection_method(&mut self, detection_method: DynamicRangeMethod) {
        self.detection_method = detection_method;
        self.rms_window_size = Self::rms_window_len(detection_method, self.sample_rate);
        self.rms_window.clear();
        self.rms_sum = 0.0;
    }

    /// RMS window length in samples: 10 ms for `Rms`/`Hybrid`, 1 for `Peak`.
    fn rms_window_len(detection_method: DynamicRangeMethod, sample_rate: f64) -> usize {
        match detection_method {
            DynamicRangeMethod::Rms | DynamicRangeMethod::Hybrid => {
                (0.01 * sample_rate).max(1.0) as usize
            }
            DynamicRangeMethod::Peak => 1,
        }
    }

    /// Resets all internal state to zero.
    ///
    /// Clears the envelope value and any accumulated RMS window data.
    /// Call this between unrelated audio segments to prevent state leaking across
    /// independent signals.
    ///
    /// # Example
    ///
    /// ```
    /// use audio_samples::operations::dynamic_range::EnvelopeFollower;
    /// use audio_samples::operations::types::DynamicRangeMethod;
    ///
    /// let mut follower = EnvelopeFollower::new(1.0, 10.0, 44100.0, DynamicRangeMethod::Peak);
    /// follower.process(0.8);
    /// follower.reset();
    /// let level = follower.process(0.0);
    /// assert_eq!(level, 0.0);
    /// ```
    #[inline]
    pub fn reset(&mut self) {
        self.envelope = 0.0;
        self.rms_window.clear();
        self.rms_sum = 0.0;
    }
}

/// Sliding-window maximum over the most recent `len` gain-reduction values (dB).
///
/// Implements the lookahead "max-hold": the gain applied to the delayed sample
/// `x[n − L]` is the largest reduction computed for any of the samples
/// `x[n − L] … x[n]`, so gain reduction is already in place when a peak that
/// lies up to `L` samples ahead arrives. A monotonic deque gives O(1)
/// amortised cost per sample.
#[derive(Debug, Clone)]
struct GainMaxHold {
    window: VecDeque<(usize, f64)>,
    len: usize,
    count: usize,
}

impl GainMaxHold {
    #[inline]
    fn new(len: usize) -> Self {
        Self {
            window: VecDeque::with_capacity(len),
            len: len.max(1),
            count: 0,
        }
    }

    /// Pushes a reduction and returns the maximum of the last `len` pushed values.
    #[inline]
    fn push(&mut self, reduction_db: f64) -> f64 {
        let idx = self.count;
        self.count += 1;
        while let Some(&(_, v)) = self.window.back() {
            if v <= reduction_db {
                self.window.pop_back();
            } else {
                break;
            }
        }
        self.window.push_back((idx, reduction_db));
        while let Some(&(j, _)) = self.window.front() {
            if j + self.len <= idx {
                self.window.pop_front();
            } else {
                break;
            }
        }
        self.window.front().map_or(reduction_db, |&(_, v)| v)
    }
}

/// Runs a feed-forward, lookahead gain stage over one channel in place.
///
/// The detector (envelope follower) sees the undelayed control signal:
/// `detector` when given (the side-chain signal), otherwise the channel itself.
/// `peaks`, when given, replaces the magnitude of the control signal as the
/// peak detector value (true-peak limiting). The static gain computer
/// `reduction_db` maps the envelope level in dBFS to a gain reduction in dB.
/// Output sample `y[n] = x[n] · 10^{−r_held[n]/20} · output_gain`, where
/// `r_held[n] = max(r[n], …, r[n + L])` and `L = lookahead` samples. Beyond the
/// end of the signal the detector is fed zeros. Because the whole signal is
/// available, the `L`-sample delay is compensated: output index `n` is aligned
/// with input index `n`.
fn process_lookahead_channel<T, F>(
    mut channel: ArrayViewMut1<'_, T>,
    detector: Option<&[f64]>,
    peaks: Option<&[f64]>,
    mut follower: EnvelopeFollower,
    lookahead: usize,
    reduction_db: F,
    output_gain: f64,
) where
    T: StandardSample,
    F: Fn(f64) -> f64,
{
    let n = channel.len();
    let mut delay = NonZeroUsize::new(lookahead).map(LookaheadBuffer::new);
    let mut hold = GainMaxHold::new(lookahead + 1);

    for i in 0..n + lookahead {
        let x: f64 = if i < n { channel[i].convert_to() } else { 0.0 };
        let detector_input: f64 = match detector {
            Some(d) if i < n => d[i],
            Some(_) => 0.0,
            None => x,
        };

        let envelope = match peaks {
            Some(p) => follower.process_with_peak(detector_input, p.get(i).copied().unwrap_or(0.0)),
            None => follower.process(detector_input),
        };
        let held_db = hold.push(reduction_db(linear_to_db(envelope)));

        let delayed = match delay.as_mut() {
            Some(buffer) => buffer.process(x),
            None => x,
        };

        if i >= lookahead {
            let output = delayed * db_to_linear(-held_db) * output_gain;
            channel[i - lookahead] = output.convert_to();
        }
    }
}

/// Second-order sections that condition the detector input of a compressor or
/// limiter when `side_chain.enabled`: a 2nd-order Butterworth high-pass at
/// `high_pass_freq`, a 2nd-order Butterworth low-pass at `low_pass_freq` and
/// the pre-emphasis shelf, each only when set. Empty when the side chain is
/// disabled or sets no filter.
fn side_chain_sections(
    side_chain: &SideChainConfig,
    sample_rate: f64,
) -> AudioSampleResult<Vec<Sos>> {
    let mut sections = Vec::new();
    if !side_chain.enabled {
        return Ok(sections);
    }
    if let Some(f) = side_chain.high_pass_freq {
        sections.extend(design_sos(
            &IirFilterDesign::butterworth_highpass(crate::nzu!(2), f),
            sample_rate,
        )?);
    }
    if let Some(f) = side_chain.low_pass_freq {
        sections.extend(design_sos(
            &IirFilterDesign::butterworth_lowpass(crate::nzu!(2), f),
            sample_rate,
        )?);
    }
    if side_chain.pre_emphasis_db != 0.0 {
        sections.push(pre_emphasis_shelf(side_chain.pre_emphasis_db, sample_rate));
    }
    Ok(sections)
}

/// Midpoint of the side-chain pre-emphasis shelf, in Hz (a quarter of the
/// sample rate when that is lower).
pub(crate) const PRE_EMPHASIS_MIDPOINT_HZ: f64 = 2_000.0;

/// First-order high shelf with gain 1 at DC and `gain_db` at the Nyquist
/// frequency, and half the gain in dB at its midpoint
/// `f_m = min(PRE_EMPHASIS_MIDPOINT_HZ, f_s/4)`.
///
/// It is the bilinear transform, prewarped at `f_m`, of
/// `H(s) = (g·s/ω_p + 1)/(s/ω_p + 1)` with `g = 10^{gain_db/20}` and
/// `ω_p = ω_m·√g`, for which `|H(jω_m)| = √g`.
fn pre_emphasis_shelf(gain_db: f64, sample_rate: f64) -> Sos {
    let g = db_to_linear(gain_db);
    let midpoint = PRE_EMPHASIS_MIDPOINT_HZ.min(sample_rate / 4.0);
    let k = (std::f64::consts::PI * midpoint / sample_rate).tan();
    let a = 1.0 / (k * g.sqrt());
    let norm = a + 1.0;
    [
        (g * a + 1.0) / norm,
        (1.0 - g * a) / norm,
        0.0,
        1.0,
        (1.0 - a) / norm,
        0.0,
    ]
}

/// The signal that drives the detector for one channel when it differs from
/// the channel: `(1 − m)·x + m·e` for an external side-chain signal `e` with
/// mix `m`, then filtered causally by `sections`. `None` when there is neither
/// an external signal nor a filter, and the detector sees the channel itself.
fn detector_signal<T: StandardSample>(
    channel: ArrayView1<'_, T>,
    external: Option<(ArrayView1<'_, T>, f64)>,
    sections: &[Sos],
) -> Option<Vec<f64>> {
    if external.is_none() && sections.is_empty() {
        return None;
    }
    let mut signal: Vec<f64> = match external {
        Some((ext, mix)) => channel
            .iter()
            .zip(ext.iter())
            .map(|(&x, &e)| {
                let (x, e): (f64, f64) = (x.convert_to(), e.convert_to());
                (1.0 - mix).mul_add(x, mix * e)
            })
            .collect(),
        None => channel.iter().map(|&x| x.convert_to()).collect(),
    };
    if !sections.is_empty() {
        sosfilt_in_place(sections, &mut signal);
    }
    Some(signal)
}

/// Oversampling factor of the true-peak detector.
const TRUE_PEAK_OVERSAMPLING: usize = 4;
/// Half-length, in input samples, of the true-peak interpolation filter
/// (12 taps per phase, as in ITU-R BS.1770).
const TRUE_PEAK_HALF_TAPS: usize = 6;
/// Kaiser window parameter of the true-peak interpolation filter.
const TRUE_PEAK_KAISER_BETA: f64 = 5.0;

/// Modified Bessel function of the first kind, order zero (power series).
fn bessel_i0(x: f64) -> f64 {
    let q = 0.25 * x * x;
    let (mut term, mut sum) = (1.0, 1.0);
    for k in 1..64 {
        term *= q / (k * k) as f64;
        sum += term;
        if term < sum * 1e-17 {
            break;
        }
    }
    sum
}

/// Taps `h_μ[j]`, `j = −5 … 6`, of the fractional-delay filter for offset
/// `μ = k/4`, `k = 1 … 3`: a Kaiser-windowed (β = 5) sinc, normalised to unit
/// DC gain. `x̂(n + μ) = Σ_j h_μ[j]·x[n + j]`.
fn true_peak_taps() -> [[f64; 2 * TRUE_PEAK_HALF_TAPS]; TRUE_PEAK_OVERSAMPLING - 1] {
    let half = TRUE_PEAK_HALF_TAPS as f64;
    let mut taps = [[0.0; 2 * TRUE_PEAK_HALF_TAPS]; TRUE_PEAK_OVERSAMPLING - 1];
    for (k, phase) in taps.iter_mut().enumerate() {
        let mu = (k + 1) as f64 / TRUE_PEAK_OVERSAMPLING as f64;
        for (idx, h) in phase.iter_mut().enumerate() {
            let t = idx as f64 - (half - 1.0) - mu;
            let sinc = if t == 0.0 {
                1.0
            } else {
                (std::f64::consts::PI * t).sin() / (std::f64::consts::PI * t)
            };
            let r = (1.0 - (t / half).powi(2)).max(0.0);
            *h = sinc * bessel_i0(TRUE_PEAK_KAISER_BETA * r.sqrt())
                / bessel_i0(TRUE_PEAK_KAISER_BETA);
        }
        let sum: f64 = phase.iter().sum();
        phase.iter_mut().for_each(|h| *h /= sum);
    }
    taps
}

/// True-peak magnitude around each sample of `signal`.
///
/// The signal (zero outside its ends) is interpolated at the three points
/// `n + k/4` between each pair of samples by [`true_peak_taps`] (4×
/// oversampling). `peaks[n]` is the largest magnitude among `x[n]` and the
/// interpolated points in `(n − 1, n + 1)`, so the gain applied to the samples
/// on either side of an inter-sample peak accounts for it.
fn true_peaks(signal: &[f64]) -> Vec<f64> {
    let n = signal.len();
    let half = TRUE_PEAK_HALF_TAPS;
    let taps = true_peak_taps();
    // Zero padding so every tap of every interval reads inside the buffer.
    let mut padded = vec![0.0; n + 2 * half + 1];
    padded[half..half + n].copy_from_slice(signal);
    // interval[m + 1]: largest interpolated magnitude in (m, m + 1), m = −1 … n − 1.
    let interval: Vec<f64> = (0..=n)
        .map(|m| {
            // Taps of interval m − 1 start at sample m − 1 − (half − 1) = m − half.
            let window = &padded[m..m + 2 * half];
            taps.iter()
                .map(|h| h.iter().zip(window).map(|(a, b)| a * b).sum::<f64>().abs())
                .fold(0.0, f64::max)
        })
        .collect();
    (0..n)
        .map(|i| signal[i].abs().max(interval[i]).max(interval[i + 1]))
        .collect()
}

/// One channel of a compressor or limiter: the detector signal (side-chain
/// blend and filters), its true peaks when `true_peak`, then the gain stage.
#[allow(clippy::too_many_arguments)]
fn process_channel<T, F>(
    channel: ArrayViewMut1<'_, T>,
    external: Option<(ArrayView1<'_, T>, f64)>,
    sections: &[Sos],
    true_peak: bool,
    follower: EnvelopeFollower,
    lookahead: usize,
    reduction_db: F,
    output_gain: f64,
) where
    T: StandardSample,
    F: Fn(f64) -> f64,
{
    let detector = detector_signal(channel.view(), external, sections);
    let peaks = true_peak.then(|| match &detector {
        Some(d) => true_peaks(d),
        None => true_peaks(
            &channel
                .iter()
                .map(|&x| x.convert_to())
                .collect::<Vec<f64>>(),
        ),
    });
    process_lookahead_channel(
        channel,
        detector.as_deref(),
        peaks.as_deref(),
        follower,
        lookahead,
        reduction_db,
        output_gain,
    );
}

/// Runs `process_channel` over every channel of `main`, driven by the
/// external side-chain signal `external` (mixed in by `mix`) when given.
///
/// A mono side-chain signal drives every channel; a multi-channel one must
/// have as many channels as `main`, and channel `c` drives channel `c`.
#[allow(clippy::too_many_arguments)]
fn process_all_channels<T, F>(
    main: &mut AudioData<'_, T>,
    external: Option<(&AudioData<'_, T>, f64)>,
    sections: &[Sos],
    true_peak: bool,
    new_follower: impl Fn() -> EnvelopeFollower,
    lookahead: usize,
    reduction_db: F,
    output_gain: f64,
) -> AudioSampleResult<()>
where
    T: StandardSample,
    F: Fn(f64) -> f64 + Copy,
{
    if let Some((ext, _)) = external {
        if ext.samples_per_channel() != main.samples_per_channel() {
            return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
                "signal_lengths",
                "Main signal and sidechain signal must have the same length",
            )));
        }
        if matches!(ext, AudioData::Multi(_)) && ext.num_channels() != main.num_channels() {
            return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
                "sidechain_signal",
                format!(
                    "a multi-channel side-chain signal must have as many channels as the main \
                     signal ({} vs {}); pass a mono side-chain signal to drive every channel",
                    ext.num_channels(),
                    main.num_channels()
                ),
            )));
        }
    }
    // The side-chain row that drives channel `c` of the main signal.
    let external_row = |c: usize| {
        external.map(|(ext, mix)| {
            let row = match ext {
                AudioData::Mono(m) => m.as_view(),
                AudioData::Multi(m) => m.as_view().index_axis_move(Axis(0), c),
            };
            (row, mix)
        })
    };
    match main {
        AudioData::Mono(samples) => {
            process_channel(
                samples.slice_mut(s![..]),
                external_row(0),
                sections,
                true_peak,
                new_follower(),
                lookahead,
                reduction_db,
                output_gain,
            );
        }
        AudioData::Multi(samples) => {
            for c in 0..samples.nrows().get() {
                process_channel(
                    samples.row_mut(c),
                    external_row(c),
                    sections,
                    true_peak,
                    new_follower(),
                    lookahead,
                    reduction_db,
                    output_gain,
                );
            }
        }
    }
    Ok(())
}

/// A circular delay buffer that enables lookahead processing in limiters and compressors.
///
/// ## Purpose
///
/// Introduces a fixed sample delay between gain computation and the signal it controls.
/// This allows the processor to begin attenuating the signal *before* a peak arrives,
/// preventing transient clipping without the distortion that zero-latency processing
/// can cause.
///
/// ## Intended Usage
///
/// Used internally by [`apply_limiter`][crate::AudioDynamicRange::apply_limiter] and
/// [`apply_compressor`][crate::AudioDynamicRange::apply_compressor]. Can be used
/// directly when implementing custom lookahead processors.
///
/// ## Invariants
///
/// - The buffer always stores exactly `lookahead_samples` samples.
/// - Until the buffer is full (the first `lookahead_samples` calls to
///   [`process`][LookaheadBuffer::process]), the output is `0.0`.
#[derive(Debug, Clone)]
pub struct LookaheadBuffer {
    /// Circular buffer for samples
    buffer: Vec<f64>,
    /// Write position in buffer
    write_pos: usize,
    /// Read position in buffer
    read_pos: usize,
    /// Buffer size
    size: usize,
    /// Whether buffer is full
    is_full: bool,
}

impl LookaheadBuffer {
    /// Creates a new lookahead buffer for the given delay length.
    ///
    /// # Arguments
    ///
    /// - `lookahead_samples` - Number of samples of delay. A value of `N` means the
    ///   output lags the input by `N` samples.
    ///
    /// # Returns
    ///
    /// A new `LookaheadBuffer` initialised with all zeros.
    ///
    /// # Example
    ///
    /// ```
    /// use audio_samples::operations::dynamic_range::LookaheadBuffer;
    /// use std::num::NonZeroUsize;
    ///
    /// // 10 ms lookahead at 44100 Hz
    /// let buf = LookaheadBuffer::new(NonZeroUsize::new(441).unwrap());
    /// ```
    #[inline]
    #[must_use]
    pub fn new(lookahead_samples: NonZeroUsize) -> Self {
        Self {
            buffer: vec![0.0; lookahead_samples.get()],
            write_pos: 0,
            read_pos: 0,
            size: lookahead_samples.get(),
            is_full: false,
        }
    }

    /// Adds a sample to the buffer and returns the correspondingly delayed output.
    ///
    /// The buffer behaves as a FIFO queue with a fixed length equal to the number of
    /// lookahead samples. The first `N` calls return `0.0` while the buffer fills;
    /// subsequent calls return samples in FIFO order, where `N` is the buffer size.
    ///
    /// # Arguments
    ///
    /// - `input` - The incoming sample to enqueue.
    ///
    /// # Returns
    ///
    /// The delayed sample, or `0.0` if the buffer has not yet been fully filled.
    ///
    /// # Example
    ///
    /// ```
    /// use audio_samples::operations::dynamic_range::LookaheadBuffer;
    /// use std::num::NonZeroUsize;
    ///
    /// let mut buf = LookaheadBuffer::new(NonZeroUsize::new(3).unwrap());
    /// assert_eq!(buf.process(1.0), 0.0); // filling
    /// assert_eq!(buf.process(2.0), 0.0); // filling
    /// assert_eq!(buf.process(3.0), 0.0); // filling
    /// assert_eq!(buf.process(4.0), 1.0); // first delayed output
    /// ```
    #[inline]
    pub fn process(&mut self, input: f64) -> f64 {
        // For a lookahead buffer, we need to return the oldest sample
        // when the buffer is full, otherwise return 0.0

        let output = if self.is_full {
            // Return the oldest sample (at read position)
            self.buffer[self.read_pos]
        } else {
            0.0
        };

        // Write new sample at write position
        self.buffer[self.write_pos] = input;

        // If buffer is full, advance read position
        if self.is_full {
            self.read_pos = (self.read_pos + 1) % self.size;
        }

        // Always advance write position
        self.write_pos = (self.write_pos + 1) % self.size;

        // Check if buffer becomes full (write position caught up to read position)
        if self.write_pos == self.read_pos && !self.is_full {
            self.is_full = true;
        }

        output
    }

    /// Returns the sample currently at the write position without consuming it.
    ///
    /// Returns the value stored at the current write position - the sample that will
    /// next be displaced when [`process`][LookaheadBuffer::process] is called.
    ///
    /// # Returns
    ///
    /// The sample at the write position.
    ///
    /// # Example
    ///
    /// ```
    /// use audio_samples::operations::dynamic_range::LookaheadBuffer;
    /// use std::num::NonZeroUsize;
    ///
    /// let buf = LookaheadBuffer::new(NonZeroUsize::new(4).unwrap());
    /// // Newly created buffer contains all zeros.
    /// assert_eq!(buf.peek(), 0.0);
    /// ```
    #[inline]
    #[must_use]
    pub fn peek(&self) -> f64 {
        self.buffer[self.write_pos]
    }

    /// Resets the buffer to its initial all-zero state.
    ///
    /// Clears all stored samples to `0.0` and resets read/write positions. Call this
    /// between unrelated processing passes to prevent stale samples from affecting output.
    ///
    /// # Example
    ///
    /// ```
    /// use audio_samples::operations::dynamic_range::LookaheadBuffer;
    /// use std::num::NonZeroUsize;
    ///
    /// let mut buf = LookaheadBuffer::new(NonZeroUsize::new(2).unwrap());
    /// buf.process(0.5);
    /// buf.reset();
    /// assert_eq!(buf.peek(), 0.0);
    /// ```
    #[inline]
    pub fn reset(&mut self) {
        self.buffer.fill(0.0);
        self.write_pos = 0;
        self.read_pos = 0;
        self.is_full = false;
    }
}

/// Computes the gain reduction in dB for a given input level and compressor settings.
///
/// Returns the amount `r` (in dB, positive = reduction) that is subtracted from the
/// input level to give the output level `Y = L − r` of the static compressor curve.
/// The hard knee gives `r = (1 − 1/R)·max(L − T, 0)`. The soft knee is the quadratic
/// knee of width `W` centred on `T` (see the [module documentation](self)), so
/// reduction already begins at `T − W/2`.
///
/// # Arguments
///
/// - `input_level_db` - Current signal level in dBFS.
/// - `threshold_db` - Compression threshold in dBFS. Gain reduction begins above this level.
/// - `ratio` - Compression ratio (e.g. `4.0` means 4:1). Must be ≥ 1.0.
/// - `knee_type` - Whether the transition at the threshold is abrupt ([`KneeType::Hard`])
///   or gradual ([`KneeType::Soft`]).
/// - `knee_width_db` - Total width `W` of the soft-knee region in dB, spanning
///   `[T − W/2, T + W/2]`. Only used when `knee_type` is [`KneeType::Soft`];
///   `W ≤ 0` behaves as a hard knee.
///
/// # Returns
///
/// Gain reduction in dB (≥ 0.0 for `ratio ≥ 1`). Subtract from the input level to
/// obtain the compressed output level.
///
/// # Example
///
/// ```
/// use audio_samples::operations::dynamic_range::calculate_compression_gain;
/// use audio_samples::operations::types::KneeType;
///
/// // 6 dB overshoot at 4:1 ratio → 4.5 dB gain reduction
/// let reduction = calculate_compression_gain(-6.0, -12.0, 4.0, KneeType::Hard, 2.0);
/// assert!((reduction - 4.5).abs() < 1e-10);
///
/// // Soft knee, W = 12 dB: at the threshold the reduction is (1 − 1/R)·W/8 = 1.125 dB
/// let at_threshold = calculate_compression_gain(-20.0, -20.0, 4.0, KneeType::Soft, 12.0);
/// assert!((at_threshold - 1.125).abs() < 1e-10);
/// ```
#[inline]
#[must_use]
pub fn calculate_compression_gain(
    input_level_db: f64,
    threshold_db: f64,
    ratio: f64,
    knee_type: KneeType,
    knee_width_db: f64,
) -> f64 {
    // Overshoot o = L − T. Hard knee: r = (1 − 1/R)·max(o, 0).
    // Soft knee of width W (Giannoulis, Massberg & Reiss 2012), centred on T:
    //   o ≤ −W/2       : r = 0
    //   |o| ≤ W/2      : r = (1 − 1/R)·(o + W/2)² / (2W)
    //   o ≥ W/2        : r = (1 − 1/R)·o
    let overshoot = input_level_db - threshold_db;
    let slope = 1.0 - 1.0 / ratio;
    quadratic_knee_reduction(overshoot, slope, knee_type, knee_width_db)
}

/// Gain reduction `slope · max(o, 0)` with an optional quadratic soft knee of
/// width `knee_width_db` centred on `o = 0` (Giannoulis, Massberg & Reiss 2012).
#[inline]
fn quadratic_knee_reduction(
    overshoot: f64,
    slope: f64,
    knee_type: KneeType,
    knee_width_db: f64,
) -> f64 {
    let half_knee = knee_width_db / 2.0;
    if knee_type == KneeType::Soft && half_knee > 0.0 && overshoot.abs() <= half_knee {
        let x = overshoot + half_knee;
        slope * x * x / (2.0 * knee_width_db)
    } else if overshoot > 0.0 {
        slope * overshoot
    } else {
        0.0
    }
}

/// Computes the gain reduction in dB for a given input level and limiter settings.
///
/// Returns the amount `r` (in dB, positive = reduction) that is subtracted from the
/// input level. This is the compressor curve with `R → ∞`: the hard knee gives
/// `r = max(L − C, 0)`; the soft knee gives `r = (L − C + W/2)²/(2W)` for
/// `|L − C| ≤ W/2`. The static output level `L − r` never exceeds the ceiling `C`.
///
/// # Arguments
///
/// - `input_level_db` - Current signal level in dBFS.
/// - `ceiling_db` - Limiting ceiling `C` in dBFS. The static output level never
///   exceeds this value (the dynamic output can, see the
///   [module documentation](self)).
/// - `knee_type` - Whether the transition is abrupt ([`KneeType::Hard`]) or gradual
///   ([`KneeType::Soft`]).
/// - `knee_width_db` - Total width `W` of the soft-knee region in dB, spanning
///   `[C − W/2, C + W/2]`. Only used when `knee_type` is [`KneeType::Soft`].
///
/// # Returns
///
/// Gain reduction in dB (≥ 0.0). Subtract from the input level to obtain the limited
/// output level.
///
/// # Example
///
/// ```
/// use audio_samples::operations::dynamic_range::calculate_limiting_gain;
/// use audio_samples::operations::types::KneeType;
///
/// // 0.5 dB overshoot → 0.5 dB gain reduction (hard limiting)
/// let reduction = calculate_limiting_gain(-0.5, -1.0, KneeType::Hard, 1.0);
/// assert!((reduction - 0.5).abs() < 1e-10);
/// ```
#[inline]
#[must_use]
pub fn calculate_limiting_gain(
    input_level_db: f64,
    ceiling_db: f64,
    knee_type: KneeType,
    knee_width_db: f64,
) -> f64 {
    // The compressor knee with R → ∞ (slope 1): the static output
    // y = L − r never exceeds the ceiling C, also inside the soft knee.
    quadratic_knee_reduction(input_level_db - ceiling_db, 1.0, knee_type, knee_width_db)
}

/// The envelope follower of a compressor.
fn compressor_follower(config: &CompressorConfig, sample_rate: f64) -> EnvelopeFollower {
    EnvelopeFollower::new(
        config.attack_ms,
        config.release_ms,
        sample_rate,
        config.detection_method,
    )
}

/// The envelope follower of a limiter.
fn limiter_follower(config: &LimiterConfig, sample_rate: f64) -> EnvelopeFollower {
    EnvelopeFollower::new(
        config.attack_ms,
        config.release_ms,
        sample_rate,
        config.detection_method,
    )
}

/// The static gain computer of a compressor: envelope level in dBFS to gain
/// reduction in dB.
fn compressor_reduction(config: &CompressorConfig) -> impl Fn(f64) -> f64 + Copy + '_ {
    move |envelope_db| {
        calculate_compression_gain(
            envelope_db,
            config.threshold_db,
            config.ratio,
            config.knee_type,
            config.knee_width_db,
        )
    }
}

/// The static gain computer of a limiter.
fn limiter_reduction(config: &LimiterConfig) -> impl Fn(f64) -> f64 + Copy + '_ {
    move |envelope_db| {
        calculate_limiting_gain(
            envelope_db,
            config.ceiling_db,
            config.knee_type,
            config.knee_width_db,
        )
    }
}

/// The `*_sidechain` methods need `side_chain.enabled`.
fn require_side_chain(side_chain: &SideChainConfig) -> AudioSampleResult<()> {
    if side_chain.enabled {
        Ok(())
    } else {
        Err(AudioSampleError::Parameter(ParameterError::invalid_value(
            "side_chain_config",
            "Side-chain is not enabled in configuration",
        )))
    }
}

impl<T> AudioDynamicRange for AudioSamples<'_, T>
where
    T: StandardSample,
    Self: AudioTypeConversion<Sample = T>,
{
    /// Applies compression to reduce the dynamic range of the signal.
    ///
    /// The detector envelope is mapped through the static curve (hard knee or quadratic
    /// soft knee, threshold `T`, ratio `R`); levels above the threshold are reduced so
    /// that the output level is `T + (L − T)/R`. `attack_ms` and `release_ms` are
    /// one-pole time constants (63 % of a step after one time constant). With
    /// `lookahead_ms > 0` gain reduction starts up to `lookahead_ms` before a peak; the
    /// delay is compensated, so the output stays time-aligned with the input.
    /// Multi-channel audio is compressed independently per channel (no stereo linking).
    /// With `side_chain.enabled` the detector input passes through the side-chain filters
    /// of [`SideChainConfig`](crate::operations::types::SideChainConfig).
    /// The formulas are in the [`dynamic_range`](crate::operations::dynamic_range)
    /// module documentation.
    ///
    /// # Arguments
    ///
    /// - `config` - Compressor parameters including threshold, ratio, attack, release,
    ///   knee type, makeup gain, lookahead, and detection method. Use preset constructors
    ///   such as [`CompressorConfig::vocal`] or [`CompressorConfig::drum`] for common
    ///   configurations.
    ///
    /// # Returns
    ///
    /// `Ok(())` on success. The audio is modified in place.
    ///
    /// # Errors
    ///
    /// Returns [crate::AudioSampleError::Parameter] if the configuration fails validation
    /// (e.g. threshold above 0 dBFS, ratio below 1.0, or negative time constants).
    ///
    /// # Example
    ///
    /// ```
    /// use audio_samples::{AudioSamples, AudioDynamicRange, sample_rate};
    /// use audio_samples::operations::types::CompressorConfig;
    /// use ndarray::Array1;
    ///
    /// let data = Array1::from_vec(vec![0.1f32, 0.8, 0.2, 0.9, 0.1]);
    /// let mut audio = AudioSamples::new_mono(data, sample_rate!(44100)).unwrap();
    /// let config = CompressorConfig::vocal();
    /// audio.apply_compressor_in_place(&config).unwrap();
    /// ```
    fn apply_compressor_in_place(&mut self, config: &CompressorConfig) -> AudioSampleResult<()> {
        let sample_rate = self.sample_rate_hz();
        config.validate(sample_rate)?;
        let sections = side_chain_sections(&config.side_chain, sample_rate)?;
        process_all_channels(
            self.data_mut(),
            None,
            &sections,
            false,
            || compressor_follower(config, sample_rate),
            ms_to_samples(config.lookahead_ms, sample_rate),
            compressor_reduction(config),
            db_to_linear(config.makeup_gain_db),
        )
    }

    /// Prevents the signal from exceeding a specified ceiling level.
    ///
    /// Applies gain reduction when the smoothed envelope level exceeds the ceiling (the
    /// compressor curve with an infinite ratio). The detector sees each sample up to
    /// `lookahead_ms` before it is output and the largest reduction over that window is
    /// applied, so gain reduction begins before peaks arrive; the delay is compensated.
    /// This is not a brick-wall limiter: the ceiling is exact only once the envelope has
    /// settled, and transients shorter than a few attack time constants can exceed it.
    /// With `isp_limiting` the peak detector reads true (inter-sample) peaks of a 4×
    /// oversampled copy of the detector signal instead of sample peaks. With
    /// `side_chain.enabled` the detector input passes through the side-chain filters.
    /// Multi-channel audio is limited independently per channel. See the
    /// [`dynamic_range`](crate::operations::dynamic_range) module documentation.
    ///
    /// # Arguments
    ///
    /// - `config` - Limiter parameters including ceiling, attack, release, knee type,
    ///   lookahead, and detection method. Use preset constructors such as
    ///   [`LimiterConfig::transparent`] or [`LimiterConfig::mastering`] for common
    ///   configurations.
    ///
    /// # Returns
    ///
    /// `Ok(())` on success. The audio is modified in place.
    ///
    /// # Errors
    ///
    /// Returns [crate::AudioSampleError::Parameter] if the configuration fails validation
    /// (e.g. ceiling above 0 dBFS or invalid time constants).
    ///
    /// # Example
    ///
    /// ```
    /// use audio_samples::{AudioSamples, AudioDynamicRange, sample_rate};
    /// use audio_samples::operations::types::LimiterConfig;
    /// use ndarray::Array1;
    ///
    /// let data = Array1::from_vec(vec![0.1f32, 0.8, 0.2, 0.9, 0.1]);
    /// let mut audio = AudioSamples::new_mono(data, sample_rate!(44100)).unwrap();
    /// let config = LimiterConfig::mastering();
    /// audio.apply_limiter_in_place(&config).unwrap();
    /// ```
    fn apply_limiter_in_place(&mut self, config: &LimiterConfig) -> AudioSampleResult<()> {
        let sample_rate = self.sample_rate_hz();
        let config = config.validate(sample_rate)?;
        let sections = side_chain_sections(&config.side_chain, sample_rate)?;
        process_all_channels(
            self.data_mut(),
            None,
            &sections,
            config.isp_limiting,
            || limiter_follower(&config, sample_rate),
            ms_to_samples(config.lookahead_ms, sample_rate),
            limiter_reduction(&config),
            1.0,
        )
    }

    /// Applies compression driven by an external sidechain signal.
    ///
    /// The gain reduction is determined by the level of `sidechain_signal` rather than
    /// the main audio, allowing one signal to control the dynamics of another. A common
    /// use case is ducking: a voice track causes background music to decrease in level
    /// whenever speech is present. Lookahead and latency compensation work as in
    /// [`apply_compressor_in_place`](Self::apply_compressor_in_place).
    ///
    /// The detector input is `(1 − m)·x + m·s`, the main signal `x` and the side-chain
    /// signal `s` mixed by `m = config.side_chain.external_mix` (1 by default: the
    /// side-chain signal alone), then passed through the side-chain filters
    /// (`high_pass_freq`, `low_pass_freq`, `pre_emphasis_db`; see
    /// [`SideChainConfig`](crate::operations::types::SideChainConfig)). A mono
    /// side-chain signal drives every channel of the main signal; a multi-channel one
    /// must have as many channels, and drives them channel by channel.
    ///
    /// # Arguments
    ///
    /// - `config` - Compressor configuration. Sidechain processing must be enabled;
    ///   call `config.side_chain.enable()` before passing the config to this method.
    /// - `sidechain_signal` - External control signal. Must be the same length as the
    ///   main signal.
    ///
    /// # Returns
    ///
    /// `Ok(())` on success. The main audio is modified in place.
    ///
    /// # Errors
    ///
    /// - [crate::AudioSampleError::Parameter] if sidechain is not enabled in `config`.
    /// - [crate::AudioSampleError::Parameter] if `config` fails validation.
    /// - [crate::AudioSampleError::Parameter] if the main and sidechain signals have
    ///   different lengths, or the sidechain signal is multi-channel with a different
    ///   channel count.
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use audio_samples::{AudioSamples, AudioDynamicRange, sample_rate};
    /// use audio_samples::operations::types::CompressorConfig;
    /// use ndarray::Array1;
    ///
    /// let mut audio = AudioSamples::new_mono(
    ///     Array1::from_vec(vec![0.1f32, 0.8, 0.2, 0.9, 0.1]),
    ///     sample_rate!(44100),
    /// ).unwrap();
    /// let sidechain = AudioSamples::new_mono(
    ///     Array1::from_vec(vec![0.0f32, 1.0, 0.0, 1.0, 0.0]),
    ///     sample_rate!(44100),
    /// ).unwrap();
    /// let mut config = CompressorConfig::new();
    /// config.side_chain.enable();
    /// audio.apply_compressor_sidechain_in_place(&config, &sidechain).unwrap();
    /// ```
    fn apply_compressor_sidechain_in_place(
        &mut self,
        config: &CompressorConfig,
        sidechain_signal: &Self,
    ) -> AudioSampleResult<()> {
        let sample_rate = self.sample_rate_hz();
        config.validate(sample_rate)?;
        require_side_chain(&config.side_chain)?;
        let sections = side_chain_sections(&config.side_chain, sample_rate)?;
        process_all_channels(
            self.data_mut(),
            Some((sidechain_signal.data(), config.side_chain.external_mix)),
            &sections,
            false,
            || compressor_follower(config, sample_rate),
            ms_to_samples(config.lookahead_ms, sample_rate),
            compressor_reduction(config),
            db_to_linear(config.makeup_gain_db),
        )
    }

    /// Applies limiting driven by an external sidechain signal.
    ///
    /// The gain reduction ceiling is enforced based on the level of `sidechain_signal`
    /// rather than the main audio. Useful for frequency-selective limiting, where a
    /// filtered copy of the signal controls gain reduction.
    /// Lookahead and `isp_limiting` work as in
    /// [`apply_limiter_in_place`](Self::apply_limiter_in_place). The detector input is
    /// mixed and filtered as in
    /// [`apply_compressor_sidechain_in_place`](Self::apply_compressor_sidechain_in_place),
    /// and the same channel combinations are accepted.
    ///
    /// # Arguments
    ///
    /// - `config` - Limiter configuration. Sidechain processing must be enabled;
    ///   call `config.side_chain.enable()` before passing the config to this method.
    /// - `sidechain_signal` - External control signal. Must be the same length as the
    ///   main signal.
    ///
    /// # Returns
    ///
    /// `Ok(())` on success. The main audio is modified in place.
    ///
    /// # Errors
    ///
    /// - [crate::AudioSampleError::Parameter] if sidechain is not enabled in `config`.
    /// - [crate::AudioSampleError::Parameter] if `config` fails validation.
    /// - [crate::AudioSampleError::Parameter] if the main and sidechain signals have
    ///   different lengths, or the sidechain signal is multi-channel with a different
    ///   channel count.
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use audio_samples::{AudioSamples, AudioDynamicRange, sample_rate};
    /// use audio_samples::operations::types::LimiterConfig;
    /// use ndarray::Array1;
    ///
    /// let mut audio = AudioSamples::new_mono(
    ///     Array1::from_vec(vec![0.1f32, 0.8, 0.2, 0.9, 0.1]),
    ///     sample_rate!(44100),
    /// ).unwrap();
    /// let sidechain = AudioSamples::new_mono(
    ///     Array1::from_vec(vec![0.0f32, 1.0, 0.0, 1.0, 0.0]),
    ///     sample_rate!(44100),
    /// ).unwrap();
    /// let mut config = LimiterConfig::default();
    /// config.side_chain.enable();
    /// audio.apply_limiter_sidechain_in_place(&config, &sidechain).unwrap();
    /// ```
    fn apply_limiter_sidechain_in_place(
        &mut self,
        config: &LimiterConfig,
        sidechain_signal: &Self,
    ) -> AudioSampleResult<()> {
        let sample_rate = self.sample_rate_hz();
        let config = config.validate(sample_rate)?;
        require_side_chain(&config.side_chain)?;
        let sections = side_chain_sections(&config.side_chain, sample_rate)?;
        process_all_channels(
            self.data_mut(),
            Some((sidechain_signal.data(), config.side_chain.external_mix)),
            &sections,
            config.isp_limiting,
            || limiter_follower(&config, sample_rate),
            ms_to_samples(config.lookahead_ms, sample_rate),
            limiter_reduction(&config),
            1.0,
        )
    }

    /// Computes the static compression input-output curve for given input levels.
    ///
    /// Maps each input level in `input_levels_db` through the compressor's static gain
    /// characteristic (threshold, ratio, knee) plus makeup gain, returning the resulting
    /// output level in dBFS for each input. Does not use time-varying envelope following -
    /// the result depends only on the static transfer function.
    ///
    /// Useful for visualising and verifying compressor behaviour without processing actual
    /// audio samples.
    ///
    /// # Arguments
    ///
    /// - `config` - Compressor configuration parameters.
    /// - `input_levels_db` - Non-empty slice of input levels in dBFS to evaluate.
    ///
    /// # Returns
    ///
    /// A `Vec<f64>` of output levels in dBFS, one per entry in `input_levels_db`,
    /// in the same order.
    ///
    /// # Example
    ///
    /// ```
    /// use audio_samples::{AudioSamples, AudioDynamicRange, sample_rate};
    /// use audio_samples::operations::types::CompressorConfig;
    /// use non_empty_slice::NonEmptySlice;
    /// use ndarray::Array1;
    ///
    /// let audio = AudioSamples::new_mono(
    ///     Array1::from_elem(10, 0.5f32), sample_rate!(44100),
    /// ).unwrap();
    /// let config = CompressorConfig::new();
    /// let levels = [-40.0f64, -20.0, -12.0, 0.0];
    /// let curve = audio.get_compression_curve(
    ///     &config,
    ///     NonEmptySlice::new(&levels).unwrap(),
    /// ).unwrap();
    /// assert_eq!(curve.len(), 4);
    /// // Levels above the threshold are reduced (output < input)
    /// assert!(curve[3] < 0.0);
    /// ```
    fn get_compression_curve(
        &self,
        config: &CompressorConfig,
        input_levels_db: &NonEmptySlice<f64>,
    ) -> AudioSampleResult<Vec<f64>> {
        let output_levels = input_levels_db
            .iter()
            .map(|&input_db| {
                let gain_reduction_db = calculate_compression_gain(
                    input_db,
                    config.threshold_db,
                    config.ratio,
                    config.knee_type,
                    config.knee_width_db,
                );
                input_db - gain_reduction_db + config.makeup_gain_db
            })
            .collect();

        Ok(output_levels)
    }

    /// Returns the per-sample gain reduction that would be applied by the compressor.
    ///
    /// Passes the audio through the envelope follower and compression gain calculation,
    /// collecting the gain reduction (in dB) at every sample without modifying the
    /// signal. For multi-channel audio, only the first channel is analysed. With
    /// `config.side_chain.enabled` the detector input passes through the side-chain
    /// filters, as in `apply_compressor`.
    ///
    /// Useful for metering, visualising, and analysing compressor activity.
    ///
    /// # Arguments
    ///
    /// - `config` - Compressor configuration parameters.
    ///
    /// # Returns
    ///
    /// A `Vec<f64>` of gain reduction values in dB (each value ≥ 0.0), one per sample
    /// in the signal (or first channel for multi-channel audio).
    ///
    /// The values come from the detector and static curve only:
    /// `config.lookahead_ms` is ignored here, so they show the gain reduction
    /// without the lookahead max-hold that `apply_compressor` applies.
    ///
    /// # Errors
    ///
    /// Returns [crate::AudioSampleError::Parameter] if `config` fails validation.
    ///
    /// # Example
    ///
    /// ```
    /// use audio_samples::{AudioSamples, AudioDynamicRange, sample_rate};
    /// use audio_samples::operations::types::CompressorConfig;
    /// use ndarray::Array1;
    ///
    /// let data = Array1::from_vec(vec![0.1f32, 0.8, 0.2, 0.9, 0.1]);
    /// let audio = AudioSamples::new_mono(data, sample_rate!(44100)).unwrap();
    /// let config = CompressorConfig::new();
    /// let reductions = audio.get_gain_reduction(&config).unwrap();
    /// assert_eq!(reductions.len(), 5);
    /// assert!(reductions.iter().all(|&r| r >= 0.0));
    /// ```
    fn get_gain_reduction(&self, config: &CompressorConfig) -> AudioSampleResult<Vec<f64>> {
        let sample_rate = self.sample_rate_hz();
        config.validate(sample_rate)?;
        let sections = side_chain_sections(&config.side_chain, sample_rate)?;
        let first = match self.data() {
            AudioData::Mono(samples) => samples.as_view(),
            AudioData::Multi(samples) => samples.as_view().index_axis_move(Axis(0), 0),
        };
        let detector = detector_signal(first, None, &sections)
            .unwrap_or_else(|| first.iter().map(|&x| x.convert_to()).collect());
        let mut follower = compressor_follower(config, sample_rate);
        let reduction = compressor_reduction(config);
        Ok(detector
            .iter()
            .map(|&x| reduction(linear_to_db(follower.process(x))))
            .collect())
    }

    /// Attenuates the signal when it falls below a threshold (noise gate).
    ///
    /// A gate mutes or reduces quiet passages - typically background noise, room tone,
    /// or bleed - while leaving louder content unaffected. When the envelope level `L`
    /// is below the threshold `T`, the gain is reduced by `(R − 1)·(T − L)` dB, a 1:R
    /// downward expansion; above the threshold the signal passes unchanged. Peak
    /// detection is always used; there is no hold time, range floor or lookahead.
    ///
    /// # Arguments
    ///
    /// - `config` - Gate parameters (threshold, ratio, attack, release). Use
    ///   [`GateConfig::new`] or the [`GateConfig::noise_gate`] preset.
    ///
    /// # Returns
    ///
    /// `Ok(())` on success. The audio is modified in place.
    ///
    /// # Errors
    ///
    /// Returns [crate::AudioSampleError::Parameter] if the configuration fails
    /// [`GateConfig::validate`] (e.g. ratio ≤ 0).
    ///
    /// # Example
    ///
    /// ```
    /// use audio_samples::{AudioSamples, AudioDynamicRange, sample_rate};
    /// use audio_samples::operations::types::GateConfig;
    /// use ndarray::Array1;
    ///
    /// let data = Array1::from_vec(vec![0.001f32, 0.8, 0.002, 0.9, 0.001]);
    /// let mut audio = AudioSamples::new_mono(data, sample_rate!(44100)).unwrap();
    /// // Gate at -20 dBFS, ratio 10: 1 dB under the threshold becomes 10 dB under
    /// audio.apply_gate_in_place(&GateConfig::with_params(-20.0, 10.0, 1.0, 10.0)).unwrap();
    /// ```
    fn apply_gate_in_place(&mut self, config: &GateConfig) -> AudioSampleResult<()> {
        config.validate()?;

        let GateConfig {
            threshold_db,
            ratio,
            attack_ms,
            release_ms,
            ..
        } = *config;

        let sample_rate = self.sample_rate_hz();
        // Gate is essentially a compressor with inverted threshold logic
        // and very high ratio for signals below threshold

        match self.data_mut() {
            AudioData::Mono(samples) => {
                let mut envelope_follower = EnvelopeFollower::new(
                    attack_ms,
                    release_ms,
                    sample_rate,
                    DynamicRangeMethod::Peak, // Use peak detection for gates
                );

                for sample in samples.iter_mut() {
                    let sample_f: f64 = (*sample).convert_to();
                    let envelope = envelope_follower.process(sample_f);
                    let envelope_db = linear_to_db(envelope);

                    // Gate logic: attenuate if below threshold
                    let gain_reduction_db = if envelope_db < threshold_db {
                        let undershoot = threshold_db - envelope_db;
                        undershoot * (ratio - 1.0)
                    } else {
                        0.0
                    };

                    let gain_linear = db_to_linear(-gain_reduction_db);
                    let output_sample = sample_f * gain_linear;

                    *sample = output_sample.convert_to();
                }
            }
            AudioData::Multi(samples) => {
                let num_channels = samples.nrows().get();
                let num_samples = samples.ncols().get();

                for channel in 0..num_channels {
                    let mut envelope_follower = EnvelopeFollower::new(
                        attack_ms,
                        release_ms,
                        sample_rate,
                        DynamicRangeMethod::Peak,
                    );

                    for sample_idx in 0..num_samples {
                        let sample_f: f64 = samples[[channel, sample_idx]].convert_to();
                        let envelope = envelope_follower.process(sample_f);
                        let envelope_db = linear_to_db(envelope);

                        // Gate logic: attenuate if below threshold
                        let gain_reduction_db = if envelope_db < threshold_db {
                            let undershoot = threshold_db - envelope_db;
                            undershoot * (ratio - 1.0)
                        } else {
                            0.0
                        };

                        let gain_linear = db_to_linear(-gain_reduction_db);
                        let output_sample = sample_f * gain_linear;

                        samples[[channel, sample_idx]] = output_sample.convert_to();
                    }
                }
            }
        }

        Ok(())
    }

    /// Increases dynamic range by expanding signals below a threshold.
    ///
    /// An expander is the complement of compression: signals below `threshold_db` are
    /// attenuated by `(R − 1)·(T − L)` dB, where `L` is the envelope level, `T` the
    /// threshold and `R` the ratio, making quiet passages quieter while leaving loud
    /// passages unchanged. RMS detection (10 ms window) is always used for expansion.
    ///
    /// # Arguments
    ///
    /// - `config` - Expander parameters (threshold, ratio, attack, release). Use
    ///   [`ExpanderConfig::new`] or the [`ExpanderConfig::gentle`] preset.
    ///
    /// # Returns
    ///
    /// `Ok(())` on success. The audio is modified in place.
    ///
    /// # Errors
    ///
    /// Returns [crate::AudioSampleError::Parameter] if the configuration fails
    /// [`ExpanderConfig::validate`] (e.g. ratio ≤ 0).
    ///
    /// # Example
    ///
    /// ```
    /// use audio_samples::{AudioSamples, AudioDynamicRange, sample_rate};
    /// use audio_samples::operations::types::ExpanderConfig;
    /// use ndarray::Array1;
    ///
    /// let data = Array1::from_vec(vec![0.1f32, 0.8, 0.2, 0.9, 0.1]);
    /// let mut audio = AudioSamples::new_mono(data, sample_rate!(44100)).unwrap();
    /// // Expand at -20 dBFS with 2:1 ratio
    /// audio.apply_expander_in_place(&ExpanderConfig::with_params(-20.0, 2.0, 1.0, 10.0)).unwrap();
    /// ```
    fn apply_expander_in_place(&mut self, config: &ExpanderConfig) -> AudioSampleResult<()> {
        config.validate()?;

        let ExpanderConfig {
            threshold_db,
            ratio,
            attack_ms,
            release_ms,
            ..
        } = *config;

        let sample_rate = self.sample_rate_hz();
        // Expander increases dynamic range by expanding signals below threshold

        match self.data_mut() {
            AudioData::Mono(samples) => {
                let mut envelope_follower = EnvelopeFollower::new(
                    attack_ms,
                    release_ms,
                    sample_rate,
                    DynamicRangeMethod::Rms, // Use RMS detection for expanders
                );

                for sample in samples.iter_mut() {
                    let sample_f: f64 = (*sample).convert_to();
                    let envelope = envelope_follower.process(sample_f);
                    let envelope_db = linear_to_db(envelope);

                    // Expander logic: amplify the difference below threshold
                    let gain_change_db = if envelope_db < threshold_db {
                        let undershoot = threshold_db - envelope_db;
                        undershoot * (ratio - 1.0) // Expand by ratio
                    } else {
                        0.0
                    };

                    let gain_linear = db_to_linear(-gain_change_db); // Negative because we're reducing level
                    let output_sample = sample_f * gain_linear;

                    *sample = output_sample.convert_to();
                }
            }
            AudioData::Multi(samples) => {
                let num_channels = samples.nrows().get();
                let num_samples = samples.ncols().get();

                for channel in 0..num_channels {
                    let mut envelope_follower = EnvelopeFollower::new(
                        attack_ms,
                        release_ms,
                        sample_rate,
                        DynamicRangeMethod::Rms,
                    );

                    for sample_idx in 0..num_samples {
                        let sample_f: f64 = samples[[channel, sample_idx]].convert_to();
                        let envelope = envelope_follower.process(sample_f);
                        let envelope_db = linear_to_db(envelope);

                        // Expander logic: amplify the difference below threshold
                        let gain_change_db = if envelope_db < threshold_db {
                            let undershoot = threshold_db - envelope_db;
                            undershoot * (ratio - 1.0) // Expand by ratio
                        } else {
                            0.0
                        };

                        let gain_linear = db_to_linear(-gain_change_db); // Negative because we're reducing level
                        let output_sample = sample_f * gain_linear;

                        samples[[channel, sample_idx]] = output_sample.convert_to();
                    }
                }
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::AudioSamples;
    use crate::sample_rate;
    use ndarray::Array1;
    use non_empty_slice::non_empty_vec;

    #[test]
    fn test_compressor_basic() {
        let data = Array1::from_vec(vec![0.1f32, 0.8, 0.2, 0.9, 0.1]);
        let mut audio = AudioSamples::new_mono(data, sample_rate!(44100)).unwrap();

        let config = CompressorConfig::new();
        let result = audio.apply_compressor_in_place(&config);

        assert!(result.is_ok());
    }

    #[test]
    fn test_limiter_basic() {
        let data = Array1::from_vec(vec![0.1f32, 0.8, 0.2, 0.9, 0.1]);
        let mut audio = AudioSamples::new_mono(data, sample_rate!(44100)).unwrap();

        let config = LimiterConfig::default();
        let result = audio.apply_limiter_in_place(&config);

        assert!(result.is_ok());
    }

    #[test]
    fn test_compression_curve() {
        let data = Array1::from_vec(vec![0.1f32, 0.2, 0.3, 0.4, 0.5]);
        let audio = AudioSamples::new_mono(data, sample_rate!(44100)).unwrap();

        let config = CompressorConfig::default();
        let input_levels = non_empty_vec![-40.0, -30.0, -20.0, -10.0, 0.0];
        let result = audio.get_compression_curve(&config, input_levels.as_non_empty_slice());

        assert!(result.is_ok());
        let output_levels = result.unwrap();
        assert_eq!(output_levels.len(), input_levels.len().get());

        // Check that compression reduces output levels above threshold
        for (i, &input_db) in input_levels.iter().enumerate() {
            if input_db > config.threshold_db {
                assert!(
                    output_levels[i] < input_db,
                    "Output level {} should be less than input level {} for compression",
                    output_levels[i],
                    input_db
                );
            }
        }
    }

    #[test]
    fn test_gain_reduction() {
        let data = Array1::from_vec(vec![0.1f32, 0.8, 0.2, 0.9, 0.1]);
        let audio = AudioSamples::new_mono(data, sample_rate!(44100)).unwrap();

        let config = CompressorConfig::new();
        let result = audio.get_gain_reduction(&config);

        assert!(result.is_ok());
        let gain_reductions = result.unwrap();
        assert_eq!(gain_reductions.len(), 5);

        // All gain reductions should be non-negative
        for &gr in &gain_reductions {
            assert!(gr >= 0.0, "Gain reduction should be non-negative: {}", gr);
        }
    }

    #[test]
    fn test_gate_basic() {
        let data = Array1::from_vec(vec![0.001f32, 0.8, 0.002, 0.9, 0.001]);
        let mut audio = AudioSamples::new_mono(data, sample_rate!(44100)).unwrap();

        let result = audio.apply_gate_in_place(&GateConfig::with_params(-20.0, 10.0, 1.0, 10.0));

        assert!(result.is_ok());
    }

    #[test]
    fn test_expander_basic() {
        let data = Array1::from_vec(vec![0.1f32, 0.8, 0.2, 0.9, 0.1]);
        let mut audio = AudioSamples::new_mono(data, sample_rate!(44100)).unwrap();

        let result =
            audio.apply_expander_in_place(&ExpanderConfig::with_params(-20.0, 2.0, 1.0, 10.0));

        assert!(result.is_ok());
    }

    #[test]
    fn test_envelope_follower() {
        let mut envelope = EnvelopeFollower::new(1.0, 10.0, 44100.0, DynamicRangeMethod::Peak);

        // Test that envelope responds to input
        let output1 = envelope.process(0.5);
        let output2 = envelope.process(0.5);

        assert!(output1 > 0.0);
        assert!(output2 > output1); // Should be increasing on attack
    }

    #[test]
    fn test_lookahead_buffer() {
        let mut buffer = LookaheadBuffer::new(crate::nzu!(3));

        // Test buffer filling
        assert_eq!(buffer.process(1.0), 0.0); // Not full yet
        assert_eq!(buffer.process(2.0), 0.0); // Not full yet
        assert_eq!(buffer.process(3.0), 0.0); // Not full yet
        assert_eq!(buffer.process(4.0), 1.0); // Now returns delayed sample
        assert_eq!(buffer.process(5.0), 2.0); // Returns next delayed sample
    }

    #[test]
    fn test_db_linear_conversion() {
        let linear: f64 = 0.5;
        let db = linear_to_db(linear);
        let back_to_linear = db_to_linear(db);

        assert!((linear - back_to_linear).abs() < 1e-10);
    }

    #[test]
    fn test_compression_gain_calculation() {
        let gain_reduction: f64 = calculate_compression_gain(
            -6.0,  // input_level_db
            -12.0, // threshold_db
            4.0,   // ratio
            KneeType::Hard,
            2.0, // knee_width_db
        );

        // With 6dB overshoot and 4:1 ratio, we should get 4.5dB gain reduction
        assert!((gain_reduction - 4.5).abs() < 1e-10);
    }

    #[test]
    fn test_limiting_gain_calculation() {
        let gain_reduction: f64 = calculate_limiting_gain(
            -0.5, // input_level_db
            -1.0, // ceiling_db
            KneeType::Hard,
            1.0, // knee_width_db
        );

        // With 0.5dB overshoot, we should get 0.5dB gain reduction
        assert!((gain_reduction - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_compressor_config_validation() {
        let mut config: CompressorConfig = CompressorConfig::new();

        // Valid config should pass
        assert!(config.validate(44100.0).is_ok());

        // Invalid threshold should fail
        config.threshold_db = 5.0;
        assert!(config.validate(44100.0).is_err());

        // Fix threshold, test invalid ratio
        config.threshold_db = -12.0;
        config.ratio = 0.5;
        assert!(config.validate(44100.0).is_err());
    }

    #[test]
    fn test_limiter_config_validation() {
        let config = LimiterConfig::default();

        let config = config.validate(44100.0);
        // Valid config should pass
        assert!(
            config.is_ok(),
            "Valid limiter config failed validation: {:?}",
            config
        );
        let mut config = config.expect("safe unwrap");

        // Invalid ceiling should fail

        config.ceiling_db = 5.0;
        let config = config.validate(44100.0);

        assert!(
            config.is_err(),
            "Invalid ceiling_db passed validation: {:?}",
            config
        );

        // Fix ceiling, test invalid attack time
        let config = LimiterConfig {
            ceiling_db: -1.0,
            attack_ms: 0.0,
            ..Default::default()
        };
        assert!(config.validate(44100.0).is_err());
    }

    #[test]
    fn test_multi_channel_compressor() {
        let data = ndarray::Array2::from_shape_vec(
            (2, 5),
            vec![0.1f32, 0.8, 0.2, 0.9, 0.1, 0.2f32, 0.7, 0.3, 0.8, 0.2],
        )
        .unwrap();
        let mut audio = AudioSamples::new_multi_channel(data, sample_rate!(44100)).unwrap();

        let config = CompressorConfig::new();
        let result = audio.apply_compressor_in_place(&config);

        assert!(result.is_ok());
    }

    #[test]
    fn test_multi_channel_limiter() {
        let data = ndarray::Array2::from_shape_vec(
            (2, 5),
            vec![0.1f32, 0.8, 0.2, 0.9, 0.1, 0.2f32, 0.7, 0.3, 0.8, 0.2],
        )
        .unwrap();
        let mut audio = AudioSamples::new_multi_channel(data, sample_rate!(44100)).unwrap();

        let config = LimiterConfig::default();
        let result = audio.apply_limiter_in_place(&config);

        assert!(result.is_ok());
    }

    #[test]
    fn test_compressor_presets() {
        let data = Array1::from_vec(vec![0.1f32, 0.8, 0.2, 0.9, 0.1]);
        let mut audio = AudioSamples::new_mono(data, sample_rate!(44100)).unwrap();

        // Test different presets
        let vocal_config = CompressorConfig::vocal();
        assert!(audio.apply_compressor_in_place(&vocal_config).is_ok());

        let drum_config = CompressorConfig::drum();
        assert!(audio.apply_compressor_in_place(&drum_config).is_ok());

        let bus_config = CompressorConfig::bus();
        assert!(audio.apply_compressor_in_place(&bus_config).is_ok());
    }

    #[test]
    fn test_limiter_presets() {
        let data = Array1::from_vec(vec![0.1f32, 0.8, 0.2, 0.9, 0.1]);
        let mut audio = AudioSamples::new_mono(data, sample_rate!(44100)).unwrap();

        // Test different presets
        let transparent_config = LimiterConfig::transparent();
        assert!(audio.apply_limiter_in_place(&transparent_config).is_ok());

        let mastering_config = LimiterConfig::mastering();
        assert!(audio.apply_limiter_in_place(&mastering_config).is_ok());

        let broadcast_config = LimiterConfig::broadcast();
        assert!(audio.apply_limiter_in_place(&broadcast_config).is_ok());
    }

    /// The mono expander applies no gain change (`gain_change_db = 0.0`) to
    /// signals above the threshold: a loud signal passes through at unity gain,
    /// not attenuated by ~1 dB (`db_to_linear(-1.0) ~= 0.891`).
    #[test]
    fn test_expander_above_threshold_mono_unchanged() {
        // Constant loud signal, well above a low threshold. RMS envelope settles
        // to the signal level, so every sample stays above threshold.
        let amp = 0.8f64;
        let data = Array1::from_vec(vec![amp as f32; 4096]);
        let mut audio = AudioSamples::new_mono(data, sample_rate!(44100)).unwrap();

        // Threshold (-40 dBFS) far below the signal (~ -1.9 dBFS).
        audio
            .apply_expander_in_place(&ExpanderConfig::with_params(-40.0, 2.0, 1.0, 10.0))
            .unwrap();

        if let AudioData::Mono(samples) = audio.data() {
            // Check the settled tail: gain must be ~1.0 (unchanged), not 0.891.
            for &s in samples.iter().skip(2048) {
                let v: f64 = s.convert_to();
                assert!(
                    (v - amp).abs() < 1e-4,
                    "above-threshold sample changed: got {v}, expected {amp}"
                );
            }
        } else {
            panic!("expected mono");
        }
    }

    /// A `ratio` of `0.0` would divide by zero in the gate gain formula
    /// `(ratio - 1.0) / ratio`. Both `apply_gate` and `apply_expander` must
    /// reject `ratio <= 0.0` with an `Err`; the rejection is done in
    /// `GateConfig::validate` / `ExpanderConfig::validate`.
    #[test]
    fn test_gate_and_expander_reject_zero_ratio() {
        let data = Array1::from_vec(vec![0.1f32, 0.8, 0.2, 0.9, 0.1]);

        let mut gate_audio = AudioSamples::new_mono(data.clone(), sample_rate!(44100)).unwrap();
        let gate_result =
            gate_audio.apply_gate_in_place(&GateConfig::with_params(-20.0, 0.0, 1.0, 10.0));
        assert!(gate_result.is_err(), "apply_gate(ratio=0.0) should error");

        let mut exp_audio = AudioSamples::new_mono(data, sample_rate!(44100)).unwrap();
        let exp_result =
            exp_audio.apply_expander_in_place(&ExpanderConfig::with_params(-20.0, 0.0, 1.0, 10.0));
        assert!(
            exp_result.is_err(),
            "apply_expander(ratio=0.0) should error"
        );

        // The audio must be untouched (no NaN/inf written) when validation fails.
        if let AudioData::Mono(samples) = gate_audio.data() {
            assert!(samples.iter().all(|&s| {
                let v: f64 = s.convert_to();
                v.is_finite()
            }));
        }
    }

    #[test]
    fn test_gate_config_validate_rejects_bad_input() {
        // Ratio must be > 0.
        assert!(
            GateConfig::with_params(-20.0, 0.0, 1.0, 10.0)
                .validate()
                .is_err()
        );
        // Attack out of range.
        assert!(
            GateConfig::with_params(-20.0, 10.0, 0.0, 10.0)
                .validate()
                .is_err()
        );
        // Release out of range.
        assert!(
            GateConfig::with_params(-20.0, 10.0, 1.0, 0.0)
                .validate()
                .is_err()
        );
        // A valid config passes.
        assert!(GateConfig::noise_gate().validate().is_ok());
    }

    #[test]
    fn test_expander_config_validate_rejects_bad_input() {
        // Ratio must be > 0.
        assert!(
            ExpanderConfig::with_params(-20.0, 0.0, 1.0, 10.0)
                .validate()
                .is_err()
        );
        // Attack out of range.
        assert!(
            ExpanderConfig::with_params(-20.0, 2.0, 2000.0, 10.0)
                .validate()
                .is_err()
        );
        // Release out of range.
        assert!(
            ExpanderConfig::with_params(-20.0, 2.0, 1.0, 20000.0)
                .validate()
                .is_err()
        );
        // A valid config passes.
        assert!(ExpanderConfig::gentle().validate().is_ok());
    }

    #[test]
    fn test_apply_compressor_dual_variant() {
        let data = Array1::from_vec(vec![0.1f32, 0.8, 0.2, 0.9, 0.1, 0.7, 0.3, 0.95]);
        let original = AudioSamples::new_mono(data.clone(), sample_rate!(44100)).unwrap();
        let config = CompressorConfig::vocal();

        // Non-mutating: returns a new copy, leaves `original` unchanged.
        let compressed = original.apply_compressor(&config).unwrap();

        // In-place on a clone produces an equal result.
        let mut in_place = original.clone();
        in_place.apply_compressor_in_place(&config).unwrap();

        assert_eq!(
            compressed.as_slice().unwrap(),
            in_place.as_slice().unwrap(),
            "non-mutating and in-place variants must produce equal results"
        );

        // The original is untouched by the non-mutating call.
        let pristine = AudioSamples::new_mono(data, sample_rate!(44100)).unwrap();
        assert_eq!(
            original.as_slice().unwrap(),
            pristine.as_slice().unwrap(),
            "non-mutating variant must not modify the original"
        );
    }

    // ---- Ring buffer against a two-pass reference -----------------------
    //
    // These reference implementations work in two passes: they compute all
    // gains into a Vec, then apply them with the
    // `gain_reductions[i - lookahead]` delay plus the trailing flush. The
    // fixed-capacity ring buffer of the processors must give output that is
    // bit-for-bit identical to these references.

    /// Brute-force reference for the lookahead gain stage: the detector runs
    /// over the zero-padded signal (length `n + L`), and sample `j` receives
    /// the maximum reduction computed for samples `j ..= j + L`.
    fn reference_lookahead_mono(
        input: &[f64],
        lookahead: usize,
        mut env: EnvelopeFollower,
        reduction: impl Fn(f64) -> f64,
        output_gain: f64,
    ) -> Vec<f64> {
        let n = input.len();
        let gains: Vec<f64> = (0..n + lookahead)
            .map(|i| {
                let x = if i < n { input[i] } else { 0.0 };
                reduction(linear_to_db(env.process(x)))
            })
            .collect();
        (0..n)
            .map(|j| {
                let held = gains[j..=j + lookahead]
                    .iter()
                    .copied()
                    .fold(f64::NEG_INFINITY, f64::max);
                input[j] * db_to_linear(-held) * output_gain
            })
            .collect()
    }

    fn reference_compressor_mono(
        input: &[f64],
        config: &CompressorConfig,
        sample_rate: f64,
    ) -> Vec<f64> {
        reference_lookahead_mono(
            input,
            ms_to_samples(config.lookahead_ms, sample_rate),
            EnvelopeFollower::new(
                config.attack_ms,
                config.release_ms,
                sample_rate,
                config.detection_method,
            ),
            |db| {
                calculate_compression_gain(
                    db,
                    config.threshold_db,
                    config.ratio,
                    config.knee_type,
                    config.knee_width_db,
                )
            },
            db_to_linear(config.makeup_gain_db),
        )
    }

    fn reference_limiter_mono(input: &[f64], config: &LimiterConfig, sample_rate: f64) -> Vec<f64> {
        reference_lookahead_mono(
            input,
            ms_to_samples(config.lookahead_ms, sample_rate),
            EnvelopeFollower::new(
                config.attack_ms,
                config.release_ms,
                sample_rate,
                config.detection_method,
            ),
            |db| {
                calculate_limiting_gain(
                    db,
                    config.ceiling_db,
                    config.knee_type,
                    config.knee_width_db,
                )
            },
            1.0,
        )
    }

    fn assert_bit_identical(a: &[f64], b: &[f64], ctx: &str) {
        assert_eq!(a.len(), b.len(), "{ctx}: length mismatch");
        for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
            assert_eq!(
                x.to_bits(),
                y.to_bits(),
                "{ctx}: sample {i} differs: {x} vs {y}"
            );
        }
    }

    #[test]
    fn test_compressor_ring_buffer_bit_identical() {
        // Input long enough to exercise both the main loop (i >= lookahead)
        // and the trailing flush.
        let sr = 44100.0;
        let input: Vec<f64> = (0..256)
            .map(|i| 0.5 * (i as f64 * 0.13).sin() + 0.4 * (i as f64 * 0.017).sin())
            .collect();

        let mut config = CompressorConfig::new();
        config.lookahead_ms = 1.0; // ~44 samples of lookahead
        config.makeup_gain_db = 3.0;

        let expected = reference_compressor_mono(&input, &config, sr);

        let mut audio =
            AudioSamples::new_mono(Array1::from_vec(input), sample_rate!(44100)).unwrap();
        audio.apply_compressor_in_place(&config).unwrap();

        assert_bit_identical(audio.as_slice().unwrap(), &expected, "compressor mono ring");
    }

    #[test]
    fn test_limiter_ring_buffer_bit_identical() {
        let sr = 44100.0;
        let input: Vec<f64> = (0..256).map(|i| 0.9 * (i as f64 * 0.21).sin()).collect();

        let config = LimiterConfig {
            lookahead_ms: 1.0,
            ..Default::default()
        };

        let expected = reference_limiter_mono(&input, &config, sr);

        let mut audio =
            AudioSamples::new_mono(Array1::from_vec(input), sample_rate!(44100)).unwrap();
        audio.apply_limiter_in_place(&config).unwrap();

        assert_bit_identical(audio.as_slice().unwrap(), &expected, "limiter mono ring");
    }

    #[test]
    fn test_compressor_ring_buffer_short_input_no_lookahead_reached() {
        // Input SHORTER than the lookahead: every sample must still be
        // processed, not returned unchanged.
        let sr = 44100.0;
        let input: Vec<f64> = vec![0.3, -0.6, 0.2, 0.9, -0.1];
        let mut config = CompressorConfig::new();
        config.lookahead_ms = 1.0; // ~44 samples >> 5
        config.makeup_gain_db = 2.0;

        let expected = reference_compressor_mono(&input, &config, sr);

        let mut audio =
            AudioSamples::new_mono(Array1::from_vec(input.clone()), sample_rate!(44100)).unwrap();
        audio.apply_compressor_in_place(&config).unwrap();

        assert_bit_identical(
            audio.as_slice().unwrap(),
            &expected,
            "compressor short input",
        );
        // The makeup gain must have been applied, so the output is not the input.
        let out = audio.as_slice().unwrap();
        assert!((out[0] - input[0] * db_to_linear(config.makeup_gain_db)).abs() < 1e-3);
    }

    #[test]
    fn test_compressor_ring_buffer_multi_channel_bit_identical() {
        let sr = 44100.0;
        let n = 200;
        let ch0: Vec<f64> = (0..n).map(|i| 0.5 * (i as f64 * 0.11).sin()).collect();
        let ch1: Vec<f64> = (0..n).map(|i| 0.7 * (i as f64 * 0.23).cos()).collect();

        let mut config = CompressorConfig::new();
        config.lookahead_ms = 0.5;
        config.makeup_gain_db = 1.5;

        // Per-channel reference (processing is independent per channel).
        let exp0 = reference_compressor_mono(&ch0, &config, sr);
        let exp1 = reference_compressor_mono(&ch1, &config, sr);

        let mut flat = ch0;
        flat.extend_from_slice(&ch1);
        let data = ndarray::Array2::from_shape_vec((2, n), flat).unwrap();
        let mut audio = AudioSamples::new_multi_channel(data, sample_rate!(44100)).unwrap();
        audio.apply_compressor_in_place(&config).unwrap();

        let out = audio.as_slice().unwrap();
        assert_bit_identical(&out[0..n], &exp0, "compressor multi ch0");
        assert_bit_identical(&out[n..2 * n], &exp1, "compressor multi ch1");
    }

    #[test]
    fn test_limiter_ring_buffer_multi_channel_bit_identical() {
        let sr = 44100.0;
        let n = 200;
        let ch0: Vec<f64> = (0..n).map(|i| 0.95 * (i as f64 * 0.19).sin()).collect();
        let ch1: Vec<f64> = (0..n).map(|i| 0.85 * (i as f64 * 0.07).cos()).collect();

        let config = LimiterConfig {
            lookahead_ms: 0.5,
            ..Default::default()
        };

        let exp0 = reference_limiter_mono(&ch0, &config, sr);
        let exp1 = reference_limiter_mono(&ch1, &config, sr);

        let mut flat = ch0;
        flat.extend_from_slice(&ch1);
        let data = ndarray::Array2::from_shape_vec((2, n), flat).unwrap();
        let mut audio = AudioSamples::new_multi_channel(data, sample_rate!(44100)).unwrap();
        audio.apply_limiter_in_place(&config).unwrap();

        let out = audio.as_slice().unwrap();
        assert_bit_identical(&out[0..n], &exp0, "limiter multi ch0");
        assert_bit_identical(&out[n..2 * n], &exp1, "limiter multi ch1");
    }

    /// A full-scale sine at f_s/4 sampled 45° off its peaks has sample peaks of
    /// 1/√2 (−3 dBFS); the true-peak detector reads its amplitude, 1.0, within
    /// the accuracy of the interpolator. A slow sine reads its sample peak.
    #[test]
    fn true_peaks_recover_inter_sample_peaks() {
        let quarter: Vec<f64> = (0..400)
            .map(|n| (std::f64::consts::FRAC_PI_2 * n as f64 + std::f64::consts::FRAC_PI_4).sin())
            .collect();
        assert!((peak_of(&quarter) - std::f64::consts::FRAC_1_SQRT_2).abs() < 1e-12);
        let tp = true_peaks(&quarter);
        for &v in &tp[20..380] {
            assert!((20.0 * v.log10()).abs() < 0.05, "true peak {v}");
        }
        let slow: Vec<f64> = (0..4800)
            .map(|n| 0.5 * (std::f64::consts::TAU * 100.0 * n as f64 / 48_000.0).sin())
            .collect();
        let tp = true_peaks(&slow);
        let max = tp[100..4700].iter().copied().fold(0.0, f64::max);
        assert!((max - 0.5).abs() < 1e-4, "slow sine true peak {max}");
        assert!(tp.iter().zip(&slow).all(|(t, x)| *t >= x.abs()));
    }

    fn peak_of(x: &[f64]) -> f64 {
        x.iter().fold(0.0, |m, v| m.max(v.abs()))
    }

    /// The interpolation phases have unit DC gain and the half-sample phase is
    /// symmetric.
    #[test]
    fn true_peak_taps_are_normalised() {
        let taps = true_peak_taps();
        for phase in &taps {
            assert!((phase.iter().sum::<f64>() - 1.0).abs() < 1e-12);
        }
        let half = &taps[1];
        for j in 0..half.len() {
            assert!((half[j] - half[half.len() - 1 - j]).abs() < 1e-15);
        }
    }

    /// The pre-emphasis shelf has unity gain at DC, the full gain at the Nyquist
    /// frequency and half the gain in dB at its midpoint.
    #[test]
    fn pre_emphasis_shelf_response() {
        for (gain_db, fs) in [(12.0, 48_000.0), (-6.0, 44_100.0), (6.0, 6_000.0)] {
            let shelf = pre_emphasis_shelf(gain_db, fs);
            let midpoint = PRE_EMPHASIS_MIDPOINT_HZ.min(fs / 4.0);
            let h =
                crate::operations::iir_design::sosfreqz(&[shelf], &[0.0, midpoint, fs / 2.0], fs);
            let db = |c: num_complex::Complex<f64>| 20.0 * c.norm().log10();
            assert!(
                db(h[0]).abs() < 1e-9,
                "{gain_db} dB at {fs}: DC {}",
                db(h[0])
            );
            assert!(
                (db(h[1]) - gain_db / 2.0).abs() < 1e-9,
                "midpoint {}",
                db(h[1])
            );
            assert!((db(h[2]) - gain_db).abs() < 1e-9, "Nyquist {}", db(h[2]));
        }
    }

    /// The side-chain filters are designed only when the side chain is enabled.
    #[test]
    fn side_chain_sections_follow_the_config() {
        let mut sc = SideChainConfig::disabled();
        sc.set_high_pass(100.0);
        assert!(side_chain_sections(&sc, 48_000.0).unwrap().is_empty());
        sc.enable();
        assert_eq!(side_chain_sections(&sc, 48_000.0).unwrap().len(), 1);
        sc.set_low_pass(5_000.0);
        sc.pre_emphasis_db = 6.0;
        assert_eq!(side_chain_sections(&sc, 48_000.0).unwrap().len(), 3);
    }
}
